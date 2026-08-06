import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import random


RANDOM_SEED = 1234


class MultilingualNLPDataset:
    def __init__(self, data_dir: str = "data", max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.language_map = {
            "eng": "English",
            "deu": "German",
            "fra": "French",
            "spa": "Spanish",
            "kor": "Korean",
            "por": "Portuguese",
            "ita": "Italian",
            "rus": "Russian",
            "jpn": "Japanese",
            "hin": "Hindi",
            "tha": "Thai",
            "zho-hans": "Chinese",
        }

    def load_data(self, languages: Optional[List[str]] = None):
        if languages is None:
            languages = list(self.language_map.keys())

        grouped_data = []
        rng = random.Random(RANDOM_SEED)

        for lang in languages:
            file_path = self.data_dir / f"cleaned_{lang}.jsonl"
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping {lang}")
                continue

            language_data = []
            with jsonlines.open(file_path) as reader:
                for line_num, obj in enumerate(reader, 1):
                    try:
                        processed = self.process_sample(obj, lang)
                        if processed:
                            language_data.append(processed)
                    except Exception as e:
                        print(f"Error parsing {file_path} line {line_num}:")
                        print(f"  Error: {e}")
                        print(f"  Sample: {obj}")
                        raise

            # Group only within a language. Mixing a globally shuffled list here labels
            # multilingual groups as whichever language happened to come first.
            rng.shuffle(language_data)
            grouped_data.extend(self.group_sentences(language_data, rng))

        rng.shuffle(grouped_data)

        train_size = int(0.9 * len(grouped_data))
        val_size = int(0.05 * len(grouped_data))

        train_data = grouped_data[:train_size]
        val_data = grouped_data[train_size : train_size + val_size]
        test_data = grouped_data[train_size + val_size :]

        return {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        }

    def ends_with_punctuation(self, sample: Dict) -> bool:
        """Check if a sentence ends with punctuation."""
        sentence = sample["sentence"].strip()
        if not sentence:
            return False

        # Common punctuation marks across languages
        punctuation_marks = {
            ".",
            "!",
            "?",
            "...",
            ":",
            ";",
            "।",
            "。",
            "！",
            "？",
            "؟",
            "¿",
            "¡",
        }
        return sentence[-1] in punctuation_marks

    def group_sentences(self, all_data: List[Dict], rng: random.Random) -> List[Dict]:
        """Group sentences randomly, 1-5 per group. Only merge sentences ending with punctuation."""
        grouped = []
        i = 0

        while i < len(all_data):
            sentence_group = [all_data[i]]
            i += 1

            # If this sentence doesn't end with punctuation, it stays alone
            if not self.ends_with_punctuation(sentence_group[0]):
                grouped.append(
                    {
                        "sentences": sentence_group,
                        "language": sentence_group[0]["language"],
                        "language_name": sentence_group[0]["language_name"],
                    }
                )
                continue

            # 1 in 10 chance of single sentence, otherwise try to add 1-4 more
            if rng.random() < 0.1:
                max_additional = 0
            else:
                max_additional = rng.randint(1, 4)

            # Try to add more sentences (only if they end with punctuation)
            added = 0
            while added < max_additional and i < len(all_data):
                next_sentence = all_data[i]

                # Add this sentence to the group
                sentence_group.append(next_sentence)
                i += 1
                added += 1

                # If this sentence doesn't end with punctuation, stop here
                if not self.ends_with_punctuation(next_sentence):
                    break

            grouped.append(
                {
                    "sentences": sentence_group,
                    "language": sentence_group[0]["language"],
                    "language_name": sentence_group[0]["language_name"],
                }
            )

        return grouped

    def process_sample(self, sample: Dict, language: str) -> Optional[Dict]:
        tokens = []
        whitespaces = []
        pos_tags = []
        lemmas = []
        dep_tags = []
        dep_heads = []

        # Handle both old format ("doc") and new format ("tokens")
        token_list = sample.get("tokens", sample.get("doc", []))
        canonical_sentence = "".join(token["text"] + token["whitespace"] for token in token_list)

        # A head outside [0, n] means the annotation itself is corrupt (e.g. a
        # merged token whose neighbors were never renumbered). Skip the sample
        # loudly rather than teach the model a malformed tree.
        bad_heads = [int(t["head"]) for t in token_list
                     if not 0 <= int(t["head"]) <= len(token_list)]
        if bad_heads:
            print(f"Warning: skipping {language} sample with out-of-range "
                  f"dependency heads {bad_heads}: {canonical_sentence[:60]!r}")
            return None

        for token in token_list:
            tokens.append(token["text"])
            whitespaces.append(token["whitespace"])  # Default to space if missing
            pos_tags.append(token["pos"])
            lemmas.append(token["lemma"])
            dep_tags.append(token["dep"])
            dep_heads.append(str(int(token["head"])))

        # Tokenization-aware transformation: move trailing normal spaces to next token's beginning
        for i in range(len(tokens) - 1):
            if whitespaces[i] == " ":
                tokens[i + 1] = " " + tokens[i + 1]
                whitespaces[i] = ""

        return {
            "language": language,
            "language_name": self.language_map[language],
            # The annotations are authoritative when a source sentence disagrees around
            # whitespace or a merged term: input text must reconstruct from target tokens.
            "sentence": canonical_sentence,
            "tokens": tokens,
            "whitespaces": whitespaces,
            "pos_tags": pos_tags,
            "lemmas": lemmas,
            "dep_tags": dep_tags,
            "dep_heads": dep_heads,
        }

    def create_prompt_response(self, example: Dict) -> Dict:
        # Handle both single sentence (old format) and grouped sentences (new format)
        if "sentences" in example:
            sentences = example["sentences"]
        else:
            # Old format - single sentence
            sentences = [example]

        # Concatenate sentence texts (no space for Japanese/Chinese)
        no_space_langs = {"Japanese", "Chinese"}
        lang_name = example.get("language_name", "")
        joiner = "" if lang_name in no_space_langs else " "
        sentence_texts = []
        for sent in sentences:
            text = sent["sentence"].strip()
            sentence_texts.append(text)
        combined_sentence = joiner.join(sentence_texts)

        prompt_parts = [
            f"Language: {example['language_name']}",
            f"Sentence: {combined_sentence}",
            "Task: Analyze tokens (idx,token,ws,POS,lemma,dep,head)",
        ]
        prompt = "\n".join(prompt_parts)

        # Format each sentence's tokens with separator between sentences
        all_response_parts = []
        for sent_idx, sent in enumerate(sentences):
            # For subsequent sentences, prepend joiner to first token
            tokens = sent["tokens"].copy()
            if sent_idx > 0 and len(tokens) > 0 and joiner:
                tokens[0] = joiner + tokens[0]

            response_parts = []
            for i, token in enumerate(tokens):
                # Compact tab-separated format with whitespace column
                # Represent whitespace as '_' for space, 'none' for no space
                ws_reprs = {
                    " ": "_",
                    "\u00a0": "nbsp",
                    "\u2009": "thinsp",
                    "\u200a": "hairsp",
                    "\u200b": "zwsp",
                    "\u202f": "narnbsp",
                    "\u3000": "ideogrp",
                }
                ws_repr = (
                    "none"
                    if sent["whitespaces"][i] == ""
                    else ws_reprs.get(sent["whitespaces"][i], sent["whitespaces"][i])
                )
                response_parts.append(
                    f"{i+1}\t{token}\t{ws_repr}\t{sent['pos_tags'][i]}\t"
                    f"{sent['lemmas'][i]}\t{sent['dep_tags'][i]}\t"
                    f"{sent['dep_heads'][i]}"
                )
            all_response_parts.append("\n".join(response_parts))

        # Join sentences with ----- separator
        tokens_data = "\n-----\n".join(all_response_parts)

        # Add the conversational prefix that instruction-tuned models naturally produce
        # Add explicit end marker for constrained decoding
        response = f"Here's the token analysis:\n\n{tokens_data}\n\n</analysis>"

        return {
            "prompt": prompt,
            "response": response,
            "full_text": f"{prompt}\n\n{response}<eos>",  # Explicitly add EOS token
        }


def prepare_dataset_for_training(tokenizer, max_length: int = 512):
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["full_text"],
            max_length=max_length,
            # Let the collator pad each batch only to its longest example. Gold
            # analyses are usually far shorter than 512 tokens, so global max-length
            # padding wastes most of the Gemma training compute.
            padding=False,
            truncation=True,
            return_tensors=None,
        )

        return model_inputs

    return tokenize_function


def load_and_prepare_data(
    tokenizer,
    data_dir: str = "data",
    languages: Optional[List[str]] = None,
    max_length: int = 512,
):
    dataset_loader = MultilingualNLPDataset(data_dir=data_dir, max_length=max_length)
    datasets = dataset_loader.load_data(languages=languages)

    for split in datasets:
        datasets[split] = datasets[split].map(
            dataset_loader.create_prompt_response,
            remove_columns=datasets[split].column_names,
        )

    tokenize_fn = prepare_dataset_for_training(tokenizer, max_length=max_length)

    tokenized_datasets = {}
    for split in datasets:
        tokenized_datasets[split] = datasets[split].map(
            tokenize_fn, batched=True, remove_columns=datasets[split].column_names
        )

    return DatasetDict(tokenized_datasets)
