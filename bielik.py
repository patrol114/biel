from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline, EarlyStoppingCallback
import torch
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Nazwa modelu
model_name = "speakleash/Bielik-7B-v0.1"

# Załaduj tokenizer i model z mniejszą precyzją
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)

# Użycie gradient checkpointing do oszczędzania pamięci
model.gradient_checkpointing_enable()

# Inicjalizacja pipeline do generowania tekstu
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Funkcja generowania tekstu
def generate_text(prompt):
    sequences = text_generator(
        text_inputs=prompt,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

# Tekst wejściowy
text = input("Podaj tekst wejściowy: ")

# Generowanie tekstu
generate_text(text)

# Funkcja do tworzenia datasetu z listy par pytań i odpowiedzi
def create_dataset(pairs):
    df = pd.DataFrame(pairs)
    train_df, eval_df = train_test_split(df, test_size=0.2)
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval": Dataset.from_pandas(eval_df)
    })
    return dataset

# Przykładowe pary pytań i odpowiedzi
pairs = [
    {"text": "To był wspaniały dzień!", "label": "positive"},
    {"text": "Przykład tekstu do klasyfikacji.", "label": "neutral"},
    {"text": "Książka opisuje przygody bohatera w magicznym świecie.", "label": "context"},
    {"text": "Gdzie rozgrywa się akcja książki?", "label": "question"}
]

# Tworzenie datasetu
dataset = create_dataset(pairs)

# Przygotowanie danych do trenowania
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Ustawienia treningowe
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Zwiększono liczbę epok treningowych
    per_device_train_batch_size=4,  # Zmniejszono rozmiar batcha
    per_device_eval_batch_size=4,  # Zmniejszono rozmiar batcha dla ewaluacji
    save_steps=2000,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=2000,
    logging_steps=500,
    learning_rate=3e-5,  # Dostosowano learning rate
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Metryka używana do wyboru najlepszego modelu
    greater_is_better=False,  # Wartość mniejsza jest lepsza dla straty
    report_to="none",  # Wyłączenie raportowania do WANDB
    dataloader_num_workers=2,  # Zwiększono liczbę wątków do ładowania danych
)

# Inicjalizacja trenera
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Dodano wczesne zatrzymanie
)

# Rozpoczęcie treningu
trainer.train()

# Funkcja oceny modelu z dodatkowymi metrykami
def evaluate_model(model, tokenizer):
    tasks = ["sentiment-analysis", "text-classification", "question-answering"]
    results = {}

    for task in tasks:
        evaluator = pipeline(task, model=model, tokenizer=tokenizer)
        # Przykładowe dane wejściowe dla każdego zadania
        inputs = {
            "sentiment-analysis": "To był wspaniały dzień!",
            "text-classification": "Przykład tekstu do klasyfikacji.",
            "question-answering": {
                "context": "Książka opisuje przygody bohatera w magicznym świecie.",
                "question": "Gdzie rozgrywa się akcja książki?"
            }
        }
        
        # Wykonanie oceny dla każdego zadania
        if task == "question-answering":
            result = evaluator(question=inputs[task]["question"], context=inputs[task]["context"])
        else:
            result = evaluator(inputs[task])
        
        results[task] = result
    
    return results

# Przeprowadzenie oceny modelu
evaluation_results = evaluate_model(model, tokenizer)
print(evaluation_results)