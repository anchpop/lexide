"""Models for the lexide tagger.

Two independently-trainable components make up the deployable pipeline:

1. CharBoundaryTagger  -- raw text -> token spans. A tiny bidirectional minGRU over
   bytes predicting per-char {O, B, I} so token boundaries need not agree with any
   subword vocabulary. This is the piece that replaces the LLM's implicit tokenizer.

2. MultiTaskTagger     -- token spans -> {POS, lemma, dep-head, dep-rel}. A shared
   multilingual subword encoder with offset-based subword->word pooling, then a POS
   head, a lemma edit-script classifier, and a Dozat-Manning biaffine dependency head.

JointTagger adds character-resolved representations and jointly learns boundaries
and word tasks. Unlike first-subword pooling, distinct words sharing one encoder
piece receive distinct character states.
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
        a = 1.0 - z
        b = z * self.to_h(x)
        if reverse:
            a = a.flip(1)
            b = b.flip(1)
        # Hillis-Steele doubling scan of the affine recurrence h_t = a_t*h_{t-1} + b_t
        # (h_0 = 0): log2(L) rounds of full-size elementwise kernels. The per-timestep
        # python loop this replaces was kernel-launch-bound (~0.06 it/s on an A10 at
        # L=768); the math is identical up to fp reassociation.
        L = b.shape[1]
        s = 1
        while s < L:
            b = b + a * F.pad(b[:, :-s], (0, 0, s, 0))
            a = a * F.pad(a[:, :-s], (0, 0, s, 0), value=1.0)
            s *= 2
        return b.flip(1) if reverse else b


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

    def __init__(self, vocab_size=259, emb_dim=64, hidden_dim=128, layers=3, dropout=0.1,
                 prior_vocab=0, prior_mode="add", prior_dim=8):
        super().__init__()
        # vocab: 256 byte values + PAD(256) + BOS(257) + EOS(258)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=256)
        # Optional per-byte boundary proposal (see prior.py). prior_vocab=0 adds no
        # parameters at all, so pre-prior checkpoints load unchanged.
        #   "add"    — summed into the byte embedding, as BERT does with segment embeddings.
        #              Shares the byte's 96 dims, so the table must keep the two separable.
        #   "concat" — its own coordinates, so layer 0 weights the two signals independently
        #              and cannot swamp the prior early in training. Widens layer 0 only.
        self.prior_mode = prior_mode
        self.prior_emb = None
        d = emb_dim
        if prior_vocab:
            width = emb_dim if prior_mode == "add" else prior_dim
            self.prior_emb = nn.Embedding(prior_vocab, width)
            if prior_mode == "concat":
                d = emb_dim + prior_dim
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(BiMinGRU(d, hidden_dim))
            d = hidden_dim * 2
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 3)

    def forward(self, byte_ids, prior_ids=None):
        h = self.emb(byte_ids)
        if self.prior_emb is not None and prior_ids is not None:
            p = self.prior_emb(prior_ids)
            h = h + p if self.prior_mode == "add" else torch.cat([h, p], dim=-1)
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


class JointTagger(nn.Module):
    """One encoder/character pass, followed by gold- or predicted-span word heads."""

    def __init__(self, encoder_name, n_pos, n_dep, n_lemma, n_langs=12,
                 encoder_layers=18, encoder_revision=None, char_dim=256,
                 char_hidden=256, char_buckets=65536, word_dim=768,
                 arc_dim=256, rel_dim=128, dropout=0.2, lang_dropout=0.15,
                 loss_weights=None, encoder_config=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if encoder_config is None:
            self.encoder = AutoModel.from_pretrained(encoder_name, revision=encoder_revision)
        else:
            cfg = dict(encoder_config)
            model_type = cfg.pop("model_type")
            self.encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **cfg))
        if encoder_layers:
            layers = self.encoder.encoder.layer
            self.encoder.encoder.layer = nn.ModuleList(layers[:encoder_layers])
            self.encoder.config.num_hidden_layers = len(self.encoder.encoder.layer)
        self.encoder.pooler = None
        hidden = self.encoder.config.hidden_size
        self.char_buckets = char_buckets
        self.lang_dropout = lang_dropout
        self.sub_projection = nn.Linear(hidden, char_dim)
        self.char_embedding = nn.Embedding(char_buckets + 1, char_dim, padding_idx=0)
        # Three binary features packed into one small (8-entry) table.
        self.feature_embedding = nn.Embedding(8, char_dim)
        self.lang_embedding = nn.Embedding(n_langs + 1, char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, num_layers=2,
                                 batch_first=True, bidirectional=True, dropout=dropout)
        self.char_norm = nn.LayerNorm(char_hidden * 2)
        self.boundary_head = nn.Linear(char_hidden * 2, 3)
        self.word_projection = nn.Linear(char_hidden * 4 + hidden, word_dim)
        self.drop = nn.Dropout(dropout)
        self.pos_head = nn.Sequential(MLP(word_dim, word_dim, dropout), nn.Linear(word_dim, n_pos))
        self.lemma_head = nn.Sequential(MLP(word_dim, word_dim, dropout), nn.Linear(word_dim, n_lemma))
        self.root = nn.Parameter(torch.empty(1, 1, word_dim))
        nn.init.normal_(self.root, std=0.02)
        self.arc_dep = MLP(word_dim, arc_dim, dropout)
        self.arc_head = MLP(word_dim, arc_dim, dropout)
        self.rel_dep = MLP(word_dim, rel_dim, dropout)
        self.rel_head = MLP(word_dim, rel_dim, dropout)
        self.arc_biaf = Biaffine(arc_dim, bias_y=False)
        self.rel_biaf = Biaffine(rel_dim, n_out=n_dep)
        self.lw = {name: 1. for name in ("boundary", "pos", "lemma", "arc", "rel")}
        self.lw.update(loss_weights or {})

    def encode_chars(self, batch):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        ids, attention = batch["input_ids"], batch["attention_mask"]
        limit = self.encoder.config.max_position_embeddings - 2
        if ids.size(1) <= limit:
            sub = self.encoder(input_ids=ids, attention_mask=attention).last_hidden_state
        else:
            # Long inference: encode all pieces in windows, then run ONE full
            # character LSTM and ONE sentence-level dependency tree. No lost tail
            # or multiple artificial roots. Training stays within max_subwords.
            rows = []
            for row, mask in zip(ids, attention):
                length = int(mask.sum())
                pieces = []
                for start in range(1, length - 1, limit - 2):
                    end = min(start + limit - 2, length - 1)
                    window = torch.cat([row[:1], row[start:end], row[length - 1:length]])
                    states = self.encoder(input_ids=window[None], attention_mask=torch.ones_like(window)[None]).last_hidden_state[0]
                    pieces.append(states[0:1] if start == 1 else states[:0])
                    pieces.append(states[1:-1])
                    if end == length - 1:
                        pieces.append(states[-1:])
                if not pieces:  # Empty sentence alongside a long sentence.
                    pieces = [self.encoder(input_ids=row[:length][None]).last_hidden_state[0]]
                hidden = torch.cat(pieces)
                rows.append(F.pad(hidden, (0, 0, 0, ids.size(1) - length)))
            sub = torch.stack(rows)
        mapping = batch["char_to_sub"]
        covered = mapping >= 0
        at_char = sub.gather(1, mapping.clamp(min=0).unsqueeze(-1).expand(-1, -1, sub.size(-1)))
        at_char = at_char * covered.unsqueeze(-1)
        lang = batch["lang_ids"]
        if self.training and self.lang_dropout:
            lang = lang.masked_fill(torch.rand_like(lang.float()) < self.lang_dropout, 0)
        features = (self.sub_projection(at_char) + self.char_embedding(batch["char_ids"])
                    + self.feature_embedding(batch["char_features"])
                    + self.lang_embedding(lang).unsqueeze(1))
        lengths = batch["char_mask"].sum(1).clamp(min=1).cpu()
        # Packed cuDNN LSTM bf16 is not uniformly supported. Keep this modest layer
        # FP32, including inputs, while the much larger encoder/heads use autocast.
        with torch.autocast(device_type=sub.device.type, enabled=False):
            packed = pack_padded_sequence(features.float(), lengths, batch_first=True, enforce_sorted=False)
            packed, _ = self.char_lstm(packed)
            chars, _ = pad_packed_sequence(packed, batch_first=True, total_length=mapping.size(1))
            chars = self.char_norm(chars)
        chars = self.drop(chars) * batch["char_mask"].unsqueeze(-1)
        return {"chars": chars, "sub_at_char": at_char,
                "boundary_logits": self.boundary_head(chars)}

    def word_heads(self, encoded, starts, ends, word_mask):
        chars = encoded["chars"]
        def gather(states, positions):
            return states.gather(1, positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, states.size(-1)))
        features = torch.cat([gather(chars, starts), gather(chars, ends - 1),
                              gather(encoded["sub_at_char"], starts)], dim=-1)
        words = self.drop(self.word_projection(features)) * word_mask.unsqueeze(-1)
        heads = torch.cat([self.root.expand(words.size(0), -1, -1), words], dim=1)
        arcs = self.arc_biaf(self.arc_dep(words), self.arc_head(heads))
        rels = self.rel_biaf(self.rel_dep(words), self.rel_head(heads))
        valid = torch.cat([word_mask.new_ones(words.size(0), 1), word_mask], dim=1)
        arcs = arcs.masked_fill(~valid[:, None, :].bool(), float("-inf"))
        # Self arcs never represent a valid dependency.
        w = words.size(1)
        arcs = arcs.masked_fill(torch.eye(w, w + 1, device=words.device, dtype=torch.bool).roll(1, dims=1)[None], float("-inf"))
        return {"pos_logits": self.pos_head(words), "lemma_logits": self.lemma_head(words),
                "arc_scores": arcs, "rel_scores": rels}

    def forward(self, batch):
        encoded = self.encode_chars(batch)
        out = self.word_heads(encoded, batch["starts"], batch["ends"], batch["word_mask"])
        out.update(encoded)
        mask = batch["word_mask"].bool()
        # A differentiable zero for empty sentences; avoid multiplying -inf arcs by zero.
        zero = out["pos_logits"].sum() * 0
        parts = {}
        def ce(logits, targets, active):
            return F.cross_entropy(logits[active].float(), targets[active]) if active.any() else zero
        parts["boundary"] = ce(out["boundary_logits"], batch["boundary"], batch["boundary"] >= 0)
        parts["pos"] = ce(out["pos_logits"], batch["pos"], mask)
        parts["lemma"] = ce(out["lemma_logits"], batch["lemma"], mask)
        arc_mask = mask & (batch["head"] >= 0)
        parts["arc"] = ce(out["arc_scores"], batch["head"], arc_mask)
        if arc_mask.any():
            rels = out["rel_scores"][arc_mask]
            rels = rels[torch.arange(len(rels), device=rels.device), batch["head"][arc_mask]]
            parts["rel"] = F.cross_entropy(rels.float(), batch["rel"][arc_mask])
        else:
            parts["rel"] = zero
        out["loss"] = sum(self.lw[name] * value for name, value in parts.items())
        out["parts"] = {name: value.detach() for name, value in parts.items()}
        return out
