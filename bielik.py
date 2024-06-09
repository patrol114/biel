from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline, EarlyStoppingCallback, AutoConfig
import torch
from datasets import load_dataset, DatasetDict
import os
import deepspeed
import pandas as pd

# Konfiguracja środowiska
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Załadowanie modelu i tokenizera
model_name = "speakleash/Bielik-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
model.gradient_checkpointing_enable()
model.is_parallelizable = True
model.model_parallel = True
model.tie_weights = False

# Ustawienie generatora tekstu
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def generate_text(prompt, temperature=1.0):
    torch.cuda.empty_cache()
    return text_generator(text_inputs=prompt, max_new_tokens=100, do_sample=True, top_k=50, eos_token_id=tokenizer.eos_token_id, temperature=temperature)

def main():
    mode = input("Wybierz tryb: generowanie tekstu (1) czy trening (2): ")
    if mode == "1":
        prompt = input("Podaj tekst wejściowy: ")
        results = generate_text(prompt, temperature=0.7)
        for result in results:
            print(f"Wynik: {result['generated_text']}")
    elif mode == "2":
        dataset = load_and_prepare_dataset(tokenizer)
        train_model(dataset)

def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("allegro/polish-question-passage-pairs")
    def tokenize_function(examples):
        return tokenizer(examples['question'], padding="max_length", truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'question', 'passage'])
    return DatasetDict({
        "train": tokenized_dataset["train"],
        "eval": tokenized_dataset["validation"]
    })

def train_model(dataset):
    # Konfiguracja treningu z użyciem DeepSpeed
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_medium=1,
        save_steps=2000,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=500,
        learning_rate=3e-5,
        weight_decay=0.1,
        bf16=False,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_b=False,
        report_to="none",
        dataloader_num_workers=2,
        run_name="bielik-training",
        logging_dir="./logs",
        save_strategy="epoch",
        gradient_accumulation_steps=16,
        lr_scheduler_type='cosine',
        warmup_steps=2000,
        max_steps=17350,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        deepspeed=deepspeed_config
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    main()
