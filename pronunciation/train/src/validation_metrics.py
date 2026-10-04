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


def collapse_phones(ids, blank_id, stress=None, *, return_starts=False):
    result, starts, previous = [], [], None
    for i, phone in enumerate(ids):
        symbol = (phone, stress[i]) if stress is not None else phone
        if phone != blank_id and symbol != previous:
            result.append(phone)
            starts.append(i)
        previous = symbol
    return (result, starts) if return_starts else result


def edit_counts(reference, hypothesis, *, return_matches=False):
    """S/D/I, optionally with exact-phone index pairs; ties prefer diagonal, D, I."""
    trace = []
    row = [(0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        new = [(0, i, 0)]
        steps = []
        for j, hyp in enumerate(hypothesis, 1):
            s, d, ins = row[j - 1]
            diagonal = (s + (ref != hyp), d, ins)
            s, d, ins = row[j]
            deletion = (s, d + 1, ins)
            s, d, ins = new[j - 1]
            insertion = (s, d, ins + 1)
            choices = (diagonal, deletion, insertion)
            step = min(range(3), key=lambda k: sum(choices[k]))
            new.append(choices[step])
            if return_matches:
                steps.append(step)
        row = new
        if return_matches:
            trace.append(steps)
    if return_matches:
        matches = []
        i, j = len(reference), len(hypothesis)
        while i and j:
            step = trace[i - 1][j - 1]
            if step == 0 and reference[i - 1] == hypothesis[j - 1]:
                matches.append((i - 1, j - 1))
            i -= step != 2
            j -= step != 1
        return row[-1], matches[::-1]
    return row[-1]


class DecodeMetrics:
    """Factor accuracy uses fixed decode probes and exact-phone matches only.

    Labels come from the first frame of each phone-only run, as in hosting.
    Pitch's L/H majority baseline uses every available validation target instead.
    Rates are micro-averaged fractions; zero-denominator factors are omitted.
    Stress accuracy covers all available matched phones; recall requires an exact
    primary/secondary label on stressed references. Tone/pitch accuracy covers
    reference bearers, while fp is the nonzero prediction rate on nonbearers.
    """
    def __init__(self, tokenizer, blank_id, selected, language_head_specs=None):
        self.tokenizer, self.blank_id, self.selected = tokenizer, blank_id, selected
        self.language_head_specs = language_head_specs or {}
        self.totals = defaultdict(lambda: defaultdict(int))
        self.factors = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.pitch_counts = defaultdict(lambda: [0, 0])

    def update(self, outputs, batch, n_frames):
        for spec in self.language_head_specs.values():
            if spec["target"] != "pitch_accent":
                continue
            for i, lang in enumerate(batch["langs"]):
                if lang == spec["lang"] and batch["pitch_accent_available"][i]:
                    refs = batch["pitch_accent_seq"][i, :int(batch["phoneme_lens"][i])]
                    for label in (1, 2):
                        self.pitch_counts[lang][label - 1] += int((refs == label).sum())
        indices = [i for i, clip_id in enumerate(batch["clip_ids"]) if clip_id in self.selected]
        if not indices:
            return
        ids = decide_frames(outputs["log_probs"][indices], outputs["nonblank_logit"][indices],
                            self.tokenizer, self.blank_id).cpu().tolist()
        stresses = outputs["stress_logits"][indices].argmax(-1).cpu().tolist()
        head_labels = {
            name: logits[indices].argmax(-1).cpu().tolist()
            for name, logits in outputs["language_head_logits"].items()
            if name in self.language_head_specs
        }
        for k, i in enumerate(indices):
            n = int(n_frames[i])
            ref = batch["phoneme_ids"][i, :int(batch["phoneme_lens"][i])].tolist()
            hyp, starts = collapse_phones(ids[k][:n], self.blank_id, return_starts=True)
            factor_hyp = collapse_phones(ids[k][:n], self.blank_id, stresses[k][:n])
            (s, d, ins), matches = edit_counts(ref, hyp, return_matches=True)
            lang = batch["langs"][i]
            total = self.totals[lang]
            for name, value in dict(errors=s+d+ins, deletions=d, insertions=ins,
                                    references=len(ref), hypotheses=len(hyp), clips=1,
                                    empty=not hyp,
                                    factor_errors=(s+d+ins if factor_hyp == hyp
                                                   else sum(edit_counts(ref, factor_hyp)))).items():
                total[name] += value
            factors = []
            if batch["stress_available"][i]:
                factors.append(("stress", batch["stress_seq"][i].tolist(), stresses[k]))
            for name, labels in head_labels.items():
                spec = self.language_head_specs[name]
                target = spec["target"]
                if lang == spec["lang"] and batch[f"{target}_available"][i]:
                    factors.append(("pitch" if target == "pitch_accent" else target,
                                    batch[f"{target}_seq"][i].tolist(), labels[k]))
            for factor, refs, labels in factors:
                for r, h in matches:
                    expected, predicted = refs[r], labels[starts[h]]
                    name = f"{factor}_acc" if factor == "stress" or expected else f"{factor}_fp"
                    count = self.factors[lang][name]
                    count[0] += (predicted != 0) if name.endswith("_fp") else (predicted == expected)
                    count[1] += 1
                    if factor == "stress" and expected:
                        count = self.factors[lang]["stress_recall"]
                        count[0] += predicted == expected
                        count[1] += 1

    def compute(self):
        result = {lang: {"per": t["errors"] / max(t["references"], 1),
                       "deletions": t["deletions"] / max(t["references"], 1),
                       "insertions": t["insertions"] / max(t["references"], 1),
                       "empty_fraction": t["empty"] / t["clips"],
                       "hyp_ref_length_ratio": t["hypotheses"] / max(t["references"], 1),
                       "factor_aware_per": t["factor_errors"] / max(t["references"], 1),
                       "clips": t["clips"]}
                for lang, t in sorted(self.totals.items())}
        for lang, metrics in self.factors.items():
            result.setdefault(lang, {}).update({name: correct / count
                                               for name, (correct, count) in metrics.items() if count})
        for lang, counts in self.pitch_counts.items():
            if sum(counts):
                result.setdefault(lang, {})["pitch_majority_baseline"] = max(counts) / sum(counts)
        return result
