import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, logging
logging.set_verbosity_error()

# Exercise 2: TokenBudgetChecker
class TokenBudgetChecker:
    def __init__(self, model_name: str = "gpt2", max_tokens: int = 100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def filter_texts(self, texts: list[str]) -> list[tuple[str, int]]:
        passed = []
        for t in texts:
            count = len(self.tokenizer.encode(t))
            if count <= self.max_tokens:
                passed.append((t, count))
        return passed

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 12 EXERCISES - TokenBudgetChecker")
    print("=" * 60)
    
    checker = TokenBudgetChecker(max_tokens=20)
    samples = [
        "Short sentence.",
        "This is a much longer sentence that might exceed the small twenty token threshold limit.",
        "RAG combines retrieval and generation.",
    ]
    
    print("\nInput Samples:")
    for s in samples:
        print(f"- '{s}'")
        
    print("\nSamples yang Lolos Budget (<= 20 tokens):")
    results = checker.filter_texts(samples)
    for text, count in results:
        print(f" [Tokens: {count:2d}] {text}")
        