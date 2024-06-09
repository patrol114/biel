# Etap 1: Importowanie bibliotek i ustawienia środowiska
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline, EarlyStoppingCallback, AutoConfig
import torch
from datasets import load_dataset, DatasetDict
import os
import deepspeed

# Ustawienia środowiskowe
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Etap 2: Załadowanie tokenizera i konfiguracji modelu
model_name = "speakleash/Bielik-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Konfiguracja DeepSpeed z uwzględnieniem ograniczenia pamięci RAM
deepspeed_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 16,
    "gradient_clipping": 1.0,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": True
    },
}

# Funkcja menu
def menu():
    choice = input("Wybierz opcję: 1. Generowanie tekstu, 2. Trening modelu: ")
    if choice == '1':
        text = input("Podaj tekst wejściowy: ")
        print(generate_text(text, temperature=0.7))
    elif choice == '2':
        dataset = load_and_prepare_dataset(tokenizer)
        train_model(dataset)
    else:
        print("Niepoprawny wybór, spróbuj ponownie.")

# Funkcja generowania tekstu
def generate_text(prompt, temperature=0.7):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16).to(device)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    generated_text = text_generator(prompt, max_length=4096, temperature=temperature, num_return_sequences=1, do_sample=True)[0]['generated_text']
    del model  # Uwalnianie pamięci
    torch.cuda.empty_cache()
    return generated_text

# Funkcja ładowania i przygotowania zestawu danych
def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("allegro/polish-question-passage-pairs", cache_format='arrow')
    def tokenize_function(examples):
        return tokenizer(examples['question'], padding="max_length", truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'question', 'passage'], num_proc=4)
    return DatasetDict({
        "train": tokenized_dataset["train"],
        "eval": tokenized_dataset["validation"]
    })

# Funkcja treningu modelu
def train_model(dataset):
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16).to(device)
    model.gradient_checkpointing_enable()
    model.is_parallelizable = True
    model.model_parallel = True
    model.tie_weights = False
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
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
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=4,
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
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    try:
        print("Rozpoczynam trening modelu...")
        trainer.train()
        print("Trening zakończony sukcesem.")
    except Exception as e:
        print(f"Trening nie powiódł się z błędem: {e}")
    del model  # Uwalnianie pamięci
    torch.cuda.empty_cache()

# Główna funkcja sterująca
if __name__ == "__main__":
    print("Witaj w aplikacji modelowania językowego!")
    while True:
        menu()
        kontynuacja = input("Czy chcesz kontynuować? (tak/nie): ")
        if kontynuacja.lower() != "tak":
            break

# Dodatkowa funkcja ewaluacji modelu, która może być wywołana w ramach treningu lub osobno
def evaluate_model(model_name, tokenizer):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16).to(device)
    tasks = ["sentiment-analysis", "text-classification", "question-answering"]
    results = {}
    for task in tasks:
        evaluator = pipeline(task, model=model, tokenizer=tokenizer, device=device)
        inputs = {
            "sentiment-analysis": "To był wspaniały dzień!",
            "text-classification": "Przykład tekstu do klasyfikacji.",
            "question-answering": {
                "context": "Książka opisuje przygody bohatera w magicznym świecie.",
                "question": "Gdzie rozgrywa się akcja książki?"
            }
        }
        if task == "question-answering":
            result = evaluator(question=inputs[task]["question"], context=inputs[task]["context"])
        else:
            result = evaluator(inputs[task])
        results[task] = result
    del model  # Uwalnianie pamięci
    torch.cuda.empty_cache()
    return results
