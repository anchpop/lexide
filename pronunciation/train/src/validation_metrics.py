"""Sentence-disjoint holdout and fixed, production-decoded validation probes."""
import hashlib
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from torch.utils.data import ConcatDataset, Subset

from .dataset import collate_fn

# Match inference's standalone import support when training runs as `-m src...`.
_PRONUNCIATION = str(Path(__file__).resolve().parents[2])
if _PRONUNCIATION not in sys.path:
    sys.path.insert(0, _PRONUNCIATION)
from inference.infer import decide_frames


def normalize_sentence(sentence: str) -> str:
    """Ignore case, whitespace variation and trailing Unicode punctuation."""
    sentence = " ".join(sentence.lower().split())
    while sentence and (sentence[-1].isspace()
                        or unicodedata.category(sentence[-1]).startswith("P")):
        sentence = sentence[:-1]
    return sentence


def sample_records(dataset):
    if isinstance(dataset, Subset):
        records = sample_records(dataset.dataset)
        return [records[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [r for ds in dataset.datasets for r in sample_records(ds)]
    return dataset.samples


def sentence_split(dataset, fraction, seed=42):
    """Keep all recordings of normalized text together, across sources/languages."""
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    groups = defaultdict(list)
    for i, record in enumerate(sample_records(dataset)):
        sentence = normalize_sentence(record.get("sentence") or "")
        # Missing text cannot establish equivalence with any other recording.
        key = ("sentence", sentence) if sentence else ("clip", record["wav_path"])
        groups[key].append(i)
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    target = int(len(dataset) * fraction)
    val = []
    for key in keys:
        if len(val) >= target:
            break
        val.extend(groups[key])
    selected = set(val)
    train = [i for i in range(len(dataset)) if i not in selected]
    if not train or not val:
        raise ValueError("sentence-disjoint split requires nonempty train and validation sets")
    return Subset(dataset, train), Subset(dataset, sorted(val))


class IdentifiedSubset(Subset):
    """Carry path identity through workers/collation, never infer it from batch order."""
    def __getitem__(self, index):
        item = dict(self.dataset[self.indices[index]])
        item["clip_id"] = self.clip_ids[index]
        return item

    def __getitems__(self, indices):
        return [self[i] for i in indices]


def identify_validation(dataset, seed=42, limit=100):
    records = sample_records(dataset)
    identified = IdentifiedSubset(dataset.dataset, dataset.indices)
    identified.clip_ids = [f"{r['lang']}/{Path(r['wav_path']).name}" for r in records]
    by_lang = defaultdict(set)
    for r, clip_id in zip(records, identified.clip_ids):
        by_lang[r["lang"]].add(clip_id)
    selected = set()
    for ids in by_lang.values():
        selected.update(sorted(ids, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).digest())[:limit])
    return identified, selected


def collate_identified(batch):
    result = collate_fn(batch)
    result["clip_ids"] = [item["clip_id"] for item in batch]
    return result


def collapse_phones(ids, blank_id, stress=None):
    result, previous = [], None
    for i, phone in enumerate(ids):
        symbol = (phone, stress[i]) if stress is not None else phone
        if phone != blank_id and symbol != previous:
            result.append(phone)
        previous = symbol
    return result


def edit_counts(reference, hypothesis):
    """Levenshtein (substitutions, deletions, insertions), diagonal-first ties."""
    row = [(0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        new = [(0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            s, d, ins = row[j - 1]
            diagonal = (s + (ref != hyp), d, ins)
            s, d, ins = row[j]
            deletion = (s, d + 1, ins)
            s, d, ins = new[j - 1]
            insertion = (s, d, ins + 1)
            new.append(min((diagonal, deletion, insertion), key=sum))
        row = new
    return row[-1]


class DecodeMetrics:
    def __init__(self, tokenizer, blank_id, selected):
        self.tokenizer, self.blank_id, self.selected = tokenizer, blank_id, selected
        self.totals = defaultdict(lambda: defaultdict(int))

    def update(self, outputs, batch, n_frames):
        indices = [i for i, clip_id in enumerate(batch["clip_ids"]) if clip_id in self.selected]
        if not indices:
            return
        ids = decide_frames(outputs["log_probs"][indices], outputs["nonblank_logit"][indices],
                            self.tokenizer, self.blank_id).cpu().tolist()
        stresses = outputs["stress_logits"][indices].argmax(-1).cpu().tolist()
        for k, i in enumerate(indices):
            n = int(n_frames[i])
            ref = batch["phoneme_ids"][i, :int(batch["phoneme_lens"][i])].tolist()
            hyp = collapse_phones(ids[k][:n], self.blank_id)
            factor_hyp = collapse_phones(ids[k][:n], self.blank_id, stresses[k][:n])
            s, d, ins = edit_counts(ref, hyp)
            total = self.totals[batch["langs"][i]]
            for name, value in dict(errors=s+d+ins, deletions=d, insertions=ins,
                                    references=len(ref), hypotheses=len(hyp), clips=1,
                                    empty=not hyp, factor_errors=sum(edit_counts(ref, factor_hyp))).items():
                total[name] += value

    def compute(self):
        return {lang: {"per": t["errors"] / max(t["references"], 1),
                       "deletions": t["deletions"] / max(t["references"], 1),
                       "insertions": t["insertions"] / max(t["references"], 1),
                       "empty_fraction": t["empty"] / t["clips"],
                       "hyp_ref_length_ratio": t["hypotheses"] / max(t["references"], 1),
                       "factor_aware_per": t["factor_errors"] / max(t["references"], 1),
                       "clips": t["clips"]}
                for lang, t in sorted(self.totals.items())}
