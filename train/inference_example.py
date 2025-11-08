#!/usr/bin/env python
from src.inference import MultilingualNLPInference

def main():
    print("Loading model...")
    inferencer = MultilingualNLPInference(
        base_model_name="google/gemma-2b",
        adapter_path="./final_model"
    )
    
    while True:
        print("\nEnter a sentence (or 'quit' to exit):")
        sentence = input("> ").strip()
        
        if sentence.lower() == 'quit':
            break
        
        print("Enter language code (eng/deu/fra/spa/kor):")
        language = input("> ").strip().lower()
        
        if language not in ["eng", "deu", "fra", "spa", "kor"]:
            print(f"Unknown language: {language}. Using 'eng'.")
            language = "eng"
        
        print("\nAnalyzing...")
        analysis = inferencer.analyze_sentence(sentence, language)
        
        print("\nRaw Analysis:")
        print(analysis)
        
        print("\nParsed Results:")
        parsed = inferencer.parse_analysis(analysis)
        for token_info in parsed:
            print(f"  {token_info['token']:15} POS: {token_info['pos']:10} "
                  f"Lemma: {token_info['lemma']:15} Dep: {token_info['dep']}")

if __name__ == "__main__":
    main()