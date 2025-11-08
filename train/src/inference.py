import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from typing import List, Dict, Optional
from pathlib import Path


class MultilingualNLPInference:
    def __init__(
        self,
        base_model_name: str = "google/gemma-2b",
        adapter_path: str = "./final_model",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading tokenizer from {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model from {base_model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if Path(adapter_path).exists():
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
        if self.device == "cpu":
            self.model = self.model.to(self.device)
    
    def analyze_sentence(
        self,
        sentence: str,
        language: str,
        max_length: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95
    ) -> str:
        language_names = {
            "eng": "English",
            "deu": "German",
            "fra": "French",
            "spa": "Spanish",
            "kor": "Korean"
        }
        
        language_name = language_names.get(language, language)
        
        prompt = f"""Language: {language_name}
Sentence: {sentence}
Task: Analyze tokens (idx,token,ws,POS,lemma,dep,head)

Analysis:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        analysis = response.split("Analysis:")[-1].strip()
        
        return analysis
    
    def parse_analysis(self, analysis: str) -> List[Dict]:
        """Parse tab-separated token analysis.
        Format: idx\ttoken\tws\tPOS\tlemma\tdep\thead
        Skips the conversational prefix "Here's the token analysis:"
        """
        results = []

        # Skip the conversational prefix if present
        if "Here's the token analysis:" in analysis:
            analysis = analysis.split("Here's the token analysis:")[-1]

        lines = analysis.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 7:
                idx = parts[0].strip()
                token = parts[1].strip()
                whitespace = parts[2].strip()
                pos = parts[3].strip()
                lemma = parts[4].strip()
                dep = parts[5].strip()
                head = parts[6].strip()

                results.append({
                    "index": idx,
                    "token": token,
                    "whitespace": whitespace if whitespace != '_' else ' ',
                    "pos": pos,
                    "lemma": lemma,
                    "dep": dep,
                    "head": head
                })

        return results
    
    def analyze_batch(
        self,
        sentences: List[str],
        languages: List[str],
        batch_size: int = 8
    ) -> List[str]:
        results = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_languages = languages[i:i+batch_size]
            
            for sent, lang in zip(batch_sentences, batch_languages):
                analysis = self.analyze_sentence(sent, lang)
                results.append(analysis)
        
        return results


def main():
    inferencer = MultilingualNLPInference(
        base_model_name="google/gemma-2b",
        adapter_path="./final_model"
    )
    
    test_sentences = [
        ("I don't have them.", "eng"),
        ("Je ne les ai pas.", "fra"),
        ("Ich werde bald zurück sein.", "deu"),
        ("Estoy pensando.", "spa"),
        ("저거로군!", "kor")
    ]
    
    print("\n" + "="*60)
    print("Testing Multilingual NLP Analysis")
    print("="*60 + "\n")
    
    for sentence, language in test_sentences:
        print(f"Language: {language}")
        print(f"Sentence: {sentence}")
        print("-" * 40)
        
        analysis = inferencer.analyze_sentence(sentence, language)
        print("Raw Analysis:")
        print(analysis)
        
        parsed = inferencer.parse_analysis(analysis)
        print("\nParsed Results:")
        for token_info in parsed:
            print(f"  [{token_info['index']}] {token_info['token']:15} "
                  f"POS: {token_info['pos']:10} Lemma: {token_info['lemma']:15} "
                  f"Dep: {token_info['dep']:10} Head: {token_info['head']}")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()