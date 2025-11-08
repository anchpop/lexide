import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import numpy as np
from inference import MultilingualNLPInference
from tqdm import tqdm


class NLPEvaluator:
    def __init__(self, model_path: str = "./final_model", base_model: str = "google/gemma-2b"):
        self.inferencer = MultilingualNLPInference(
            base_model_name=base_model,
            adapter_path=model_path
        )
        
    def load_test_data(self, data_dir: str = "data", languages: List[str] = None, max_samples: int = 100):
        if languages is None:
            languages = ["eng", "deu", "fra", "spa", "kor"]
        
        test_data = []
        
        for lang in languages:
            file_path = Path(data_dir) / f"{lang}.jsonl"
            if not file_path.exists():
                continue
            
            with jsonlines.open(file_path) as reader:
                lang_samples = []
                for obj in reader:
                    lang_samples.append((obj, lang))
                    if len(lang_samples) >= max_samples:
                        break
                test_data.extend(lang_samples)
        
        return test_data
    
    def extract_tokens_and_labels(self, sample: Dict) -> Tuple[List[str], List[str], List[str], List[str]]:
        tokens = []
        pos_tags = []
        lemmas = []
        dep_tags = []
        
        for token in sample["doc"]:
            tokens.append(token["text"])
            pos_tags.append(token["pos"])
            lemmas.append(token["lemma"])
            dep_tags.append(token["dep"])
        
        return tokens, pos_tags, lemmas, dep_tags
    
    def evaluate_predictions(
        self,
        test_data: List[Tuple[Dict, str]],
        verbose: bool = True
    ) -> Dict:
        all_true_pos = []
        all_pred_pos = []
        all_true_lemmas = []
        all_pred_lemmas = []
        all_true_deps = []
        all_pred_deps = []
        
        language_scores = defaultdict(lambda: {
            "pos_correct": 0,
            "lemma_correct": 0,
            "dep_correct": 0,
            "total_tokens": 0
        })
        
        for sample, lang in tqdm(test_data, desc="Evaluating"):
            sentence = sample["sentence"]
            tokens, true_pos, true_lemmas, true_deps = self.extract_tokens_and_labels(sample)
            
            try:
                analysis = self.inferencer.analyze_sentence(sentence, lang)
                parsed = self.inferencer.parse_analysis(analysis)
                
                pred_pos = []
                pred_lemmas = []
                pred_deps = []
                
                for i, token in enumerate(tokens[:len(parsed)]):
                    if i < len(parsed):
                        pred_pos.append(parsed[i]["pos"])
                        pred_lemmas.append(parsed[i]["lemma"])
                        pred_deps.append(parsed[i]["dep"])
                    else:
                        pred_pos.append("UNK")
                        pred_lemmas.append(token)
                        pred_deps.append("UNK")
                
                for i in range(min(len(true_pos), len(pred_pos))):
                    all_true_pos.append(true_pos[i])
                    all_pred_pos.append(pred_pos[i])
                    all_true_lemmas.append(true_lemmas[i])
                    all_pred_lemmas.append(pred_lemmas[i])
                    all_true_deps.append(true_deps[i])
                    all_pred_deps.append(pred_deps[i])
                    
                    language_scores[lang]["total_tokens"] += 1
                    if true_pos[i] == pred_pos[i]:
                        language_scores[lang]["pos_correct"] += 1
                    if true_lemmas[i] == pred_lemmas[i]:
                        language_scores[lang]["lemma_correct"] += 1
                    if true_deps[i] == pred_deps[i]:
                        language_scores[lang]["dep_correct"] += 1
                        
            except Exception as e:
                if verbose:
                    print(f"Error processing sample: {e}")
                continue
        
        overall_results = {
            "pos_accuracy": accuracy_score(all_true_pos, all_pred_pos),
            "lemma_accuracy": accuracy_score(all_true_lemmas, all_pred_lemmas),
            "dep_accuracy": accuracy_score(all_true_deps, all_pred_deps),
            "language_scores": {}
        }
        
        for lang, scores in language_scores.items():
            if scores["total_tokens"] > 0:
                overall_results["language_scores"][lang] = {
                    "pos_accuracy": scores["pos_correct"] / scores["total_tokens"],
                    "lemma_accuracy": scores["lemma_correct"] / scores["total_tokens"],
                    "dep_accuracy": scores["dep_correct"] / scores["total_tokens"],
                    "total_tokens": scores["total_tokens"]
                }
        
        if verbose:
            unique_pos = list(set(all_true_pos + all_pred_pos))
            pos_report = classification_report(
                all_true_pos,
                all_pred_pos,
                labels=unique_pos[:20],
                target_names=unique_pos[:20],
                zero_division=0
            )
            overall_results["pos_classification_report"] = pos_report
        
        return overall_results
    
    def print_results(self, results: Dict):
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Accuracy:")
        print(f"  POS Tagging:    {results['pos_accuracy']:.3f}")
        print(f"  Lemmatization:  {results['lemma_accuracy']:.3f}")
        print(f"  Dependencies:   {results['dep_accuracy']:.3f}")
        
        print(f"\nPer-Language Results:")
        for lang, scores in results["language_scores"].items():
            print(f"\n  {lang.upper()} ({scores['total_tokens']} tokens):")
            print(f"    POS:    {scores['pos_accuracy']:.3f}")
            print(f"    Lemma:  {scores['lemma_accuracy']:.3f}")
            print(f"    Dep:    {scores['dep_accuracy']:.3f}")
        
        if "pos_classification_report" in results:
            print("\nPOS Tag Classification Report (Top 20 tags):")
            print(results["pos_classification_report"])


def main():
    evaluator = NLPEvaluator(
        model_path="./final_model",
        base_model="google/gemma-2b"
    )
    
    print("Loading test data...")
    test_data = evaluator.load_test_data(
        data_dir="data",
        languages=["eng", "deu", "fra", "spa", "kor"],
        max_samples=50
    )
    
    print(f"Loaded {len(test_data)} test samples")
    
    print("\nEvaluating model...")
    results = evaluator.evaluate_predictions(test_data, verbose=True)
    
    evaluator.print_results(results)
    
    with open("evaluation_results.json", "w") as f:
        json_results = {k: v for k, v in results.items() if k != "pos_classification_report"}
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()