import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Hilangkan peringatan symlinks Windows dan oneDNN
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, logging
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset

logging.set_verbosity_error()

# Folder penyimpanan adapter dan output training di lokal folder proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "lora_output")
ADAPTER_DIR = os.path.join(BASE_DIR, "lora_adapter")

if __name__ == "__main__":
    print("12.5 PEFT / LoRA Fine-Tuning Demo\n")

    # 1. Load base model (distilgpt2: model bahasa kecil ~82M parameter)
    model_name = "distilgpt2"
    print(f"1. Memuat base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Konfigurasi LoRA (Low-Rank Adaptation)
    print("2. Mengonfigurasi LoRA adapter (Rank=8, Alpha=32)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        fan_in_fan_out=True,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 3. Siapkan dataset pelatihan tanya-jawab AI
    print("\n3. Menyiapkan dataset fine-tuning...")
    training_examples = [
        "Question: What is RAG? Answer: RAG stands for Retrieval-Augmented Generation.",
        "Question: What is fine-tuning? Answer: Fine-tuning trains a pre-trained model on new data.",
        "Question: What is an embedding? Answer: An embedding is a dense vector representing text meaning.",
        "Question: What is a vector database? Answer: A vector database stores embeddings for similarity search.",
        "Question: What is LangChain? Answer: LangChain is a framework for building LLM applications.",
    ]

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=64,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": training_examples})
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

    # 4. Argumen Training (Disesuaikan untuk CPU agar cepat dan ringan)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
    )

    print("\n4. Memulai proses fine-tuning LoRA...")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()

    # 5. Simpan hanya LoRA adapter weights
    print(f"\n5. Menyimpan LoRA adapter ke: {ADAPTER_DIR}")
    peft_model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("LoRA adapter dan tokenizer berhasil disimpan!")

    # 6. Uji inferensi dengan model yang telah di-fine-tune
    print("\n6. Menguji inferensi model hasil fine-tuning:")
    test_prompt = "Question: What is RAG? Answer:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = peft_model.generate(
            **inputs,
            max_new_tokens=25,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Hasil : {generated_text.strip()}\n")

    print("12.5 LoRA Fine-Tuning Selesai Dijalankan!")