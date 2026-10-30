# 12.3 Tokenizers (Halaman 137-138)
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Hilangkan warning symlinks Windows dari Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, logging
logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# GPT-2 tidak punya pad token bawaan, gunakan eos_token sebagai pad_token
tokenizer.pad_token = tokenizer.eos_token

text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."

tokens = tokenizer.encode(text)
print(f"Token count: {len(tokens)}")
print(f"Token IDs: {tokens}")

decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

token_strings = tokenizer.convert_ids_to_tokens(tokens)
print(f"Tokens: {token_strings}")

# Batch encoding with padding and truncation
batch = ["What is RAG?", "Explain vector databases in detail."]
encoded = tokenizer(
    batch,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(f"\nBatch input_ids shape: {encoded['input_ids'].shape}")
print(f"Attention mask shape: {encoded['attention_mask'].shape}")

def count_tokens(text: str, tok=tokenizer) -> int:
    return len(tok.encode(text))

prompt = "Summarise the following document: " + "word " * 50
print(f"Prompt token count: {count_tokens(prompt)}")