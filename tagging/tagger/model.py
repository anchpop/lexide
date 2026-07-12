"""Models for the lexide tagger.

Two independently-trainable components make up the deployable pipeline:

1. CharBoundaryTagger  -- raw text -> token spans. A tiny bidirectional minGRU over
   bytes predicting per-char {O, B, I} so token boundaries need not agree with any
   subword vocabulary. This is the piece that replaces the LLM's implicit tokenizer.

2. MultiTaskTagger     -- token spans -> {POS, lemma, dep-head, dep-rel}. A shared
   multilingual subword encoder with offset-based subword->word pooling, then a POS
   head, a lemma edit-script classifier, and a Dozat-Manning biaffine dependency head.

Splitting the two lets each train on the signal it needs and keeps the tagger's
per-token heads operating on exactly one vector per *our* token, regardless of how
the encoder's own subword tokenizer split the string.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
# NOTE: `transformers` is imported lazily inside MultiTaskTagger.__init__ so that the
# tiny byte-level CharBoundaryTagger (and its sentence-segmenter twin) can be trained /
# exported in a minimal torch-only environment without pulling in transformers.


# --------------------------------------------------------------------------------------
# minGRU (Feng et al., "Were RNNs All We Needed?") -- a linear-recurrence GRU with no
# hidden-state dependence in the candidate, so it parallelizes. Sequences here are short
# (per-sentence character strings), so a plain sequential scan is fast and exact.
# --------------------------------------------------------------------------------------
class MinGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.to_z = nn.Linear(in_dim, hidden_dim)
        self.to_h = nn.Linear(in_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, reverse=False):
        # x: [B, L, D]
        z = torch.sigmoid(self.to_z(x))
        h_cand = self.to_h(x)
        B, L, H = z.shape
        idx = range(L - 1, -1, -1) if reverse else range(L)
        h = x.new_zeros(B, H)
        outs = [None] * L
        for t in idx:
            h = (1 - z[:, t]) * h + z[:, t] * h_cand[:, t]
            outs[t] = h
        return torch.stack(outs, dim=1)


class BiMinGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fwd = MinGRU(in_dim, hidden_dim)
        self.bwd = MinGRU(in_dim, hidden_dim)

    def forward(self, x):
        return torch.cat([self.fwd(x, reverse=False), self.bwd(x, reverse=True)], dim=-1)


class CharBoundaryTagger(nn.Module):
    """Byte-level bidirectional minGRU tagging each character as O/B/I.

    O = not part of a token (whitespace between tokens), B = token begins here,
    I = token continues. Token spans are recovered as each B and its trailing I run.
    """
    LABELS = ["O", "B", "I"]

    def __init__(self, vocab_size=259, emb_dim=64, hidden_dim=128, layers=3, dropout=0.1):
        super().__init__()
        # vocab: 256 byte values + PAD(256) + BOS(257) + EOS(258)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=256)
        self.layers = nn.ModuleList()
        d = emb_dim
        for _ in range(layers):
            self.layers.append(BiMinGRU(d, hidden_dim))
            d = hidden_dim * 2
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 3)

    def forward(self, byte_ids):
        h = self.emb(byte_ids)
        for layer in self.layers:
            h = self.drop(layer(h))
        return self.out(self.norm(h))  # [B, L, 3]


# --------------------------------------------------------------------------------------
# Multi-task subword tagger
# --------------------------------------------------------------------------------------
class Biaffine(nn.Module):
    """Biaffine scorer: for head/dep representations produces pairwise scores.

    out[b, i, j] = h_dep[b, i]^T W h_head[b, j] (+ bias terms via appended 1s).
    Set n_out>1 for labeled scoring -> out[b, i, j, r].
    """
    def __init__(self, in_dim, n_out=1, bias_x=True, bias_y=True):
        super().__init__()
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.W = nn.Parameter(torch.zeros(n_out, in_dim + int(bias_x), in_dim + int(bias_y)))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # x (dep): [B, T, D], y (head): [B, U, D]
        if self.bias_x:
            x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
        # [B, n_out, T, U]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.W, y)
        if self.n_out == 1:
            return s.squeeze(1)          # [B, T, U]
        return s.permute(0, 2, 3, 1)      # [B, T, U, n_out]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.33):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.lin(x)))


@dataclass
class TaggerOutput:
    loss: torch.Tensor
    pos_logits: torch.Tensor
    lemma_logits: torch.Tensor
    arc_scores: torch.Tensor
    rel_scores: torch.Tensor
    parts: dict


class MultiTaskTagger(nn.Module):
    def __init__(self, encoder_name, n_pos, n_dep, n_lemma,
                 arc_dim=256, rel_dim=128, dropout=0.2,
                 loss_weights=None):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(encoder_name)
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)

        self.pos_head = nn.Linear(H, n_pos)
        self.lemma_head = nn.Linear(H, n_lemma)

        # learned ROOT word representation (head candidate index 0)
        self.root = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.normal_(self.root, std=0.02)

        self.arc_dep = MLP(H, arc_dim, dropout)
        self.arc_head = MLP(H, arc_dim, dropout)
        self.rel_dep = MLP(H, rel_dim, dropout)
        self.rel_head = MLP(H, rel_dim, dropout)
        self.arc_biaf = Biaffine(arc_dim, n_out=1, bias_x=True, bias_y=False)
        self.rel_biaf = Biaffine(rel_dim, n_out=n_dep, bias_x=True, bias_y=True)

        self.n_dep = n_dep
        self.lw = loss_weights or {"pos": 1.0, "lemma": 1.0, "arc": 1.0, "rel": 1.0}

    def pool_words(self, hidden, word_first_sub, word_mask):
        """Gather the first-subword vector for each word. word_first_sub: [B, W] long."""
        B, W = word_first_sub.shape
        H = hidden.size(-1)
        idx = word_first_sub.clamp(min=0).unsqueeze(-1).expand(B, W, H)
        pooled = torch.gather(hidden, 1, idx)
        return pooled * word_mask.unsqueeze(-1)

    def forward(self, input_ids, attention_mask, word_first_sub, word_mask,
                pos=None, lemma=None, head=None, rel=None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        enc = self.drop(enc)
        words = self.pool_words(enc, word_first_sub, word_mask)   # [B, W, H]

        pos_logits = self.pos_head(words)
        lemma_logits = self.lemma_head(words)

        B, W, H = words.shape
        root = self.root.expand(B, 1, H)
        head_cands = torch.cat([root, words], dim=1)              # [B, W+1, H]

        arc_d = self.arc_dep(words)                               # [B, W, arc]
        arc_h = self.arc_head(head_cands)                         # [B, W+1, arc]
        arc_scores = self.arc_biaf(arc_d, arc_h)                  # [B, W, W+1]

        rel_d = self.rel_dep(words)                               # [B, W, rel]
        rel_h = self.rel_head(head_cands)                         # [B, W+1, rel]
        rel_scores = self.rel_biaf(rel_d, rel_h)                  # [B, W, W+1, n_dep]

        # mask invalid head candidates (padding words) to -inf on the arc scores
        head_mask = torch.cat([word_mask.new_ones(B, 1), word_mask], dim=1)  # [B, W+1]
        arc_scores = arc_scores.masked_fill(~head_mask.bool().unsqueeze(1), float("-inf"))

        loss = None
        parts = {}
        if pos is not None:
            m = word_mask.bool()
            wl = self.lw
            pos_l = F.cross_entropy(pos_logits[m], pos[m])
            lemma_l = F.cross_entropy(lemma_logits[m], lemma[m])
            arc_l = F.cross_entropy(arc_scores[m], head[m])
            # relation scored at the gold head
            gold_head = head[m].clamp(min=0)
            rel_at_gold = rel_scores[m][torch.arange(m.sum(), device=words.device), gold_head]
            rel_l = F.cross_entropy(rel_at_gold, rel[m])
            loss = wl["pos"] * pos_l + wl["lemma"] * lemma_l + wl["arc"] * arc_l + wl["rel"] * rel_l
            parts = {"pos": pos_l.item(), "lemma": lemma_l.item(),
                     "arc": arc_l.item(), "rel": rel_l.item()}

        return TaggerOutput(loss, pos_logits, lemma_logits, arc_scores, rel_scores, parts)
