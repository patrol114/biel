# Etap 1: Importowanie bibliotek i ustawienia środowiska
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline, EarlyStoppingCallback, AutoConfig
import torch
from datasets import Dataset, DatasetDict
import os
import deepspeed
import pandas as pd

# Ustawienia środowiskowe
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Etap 2: Załadowanie modelu i tokenizeraa
model_name = "speakleash/Bielik-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Załadowanie modelu z niską precyzją i włączenie optymalizacji pamięci
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
model.gradient_checkpointing_enable()
model.is_parallelizable = True
model.model_parallel = True
model.tie_weights = False

# Etap 3: Konfiguracja DeepSpeed
deepspeed_config = {
    "train_micro_batch_size_per_gpu": 1,
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
    "zero_allow_untested_optimizer": True,
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": True
    },
}

# Etap 4: Generowanie tekstu
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def generate_text(prompt, temperature=1.0):
    torch.cuda.empty_cache()
    sequences = text_generator(text_inputs=prompt, max_new_tokens=100, do_sample=True, top_k=50, eos_token_id=tokenizer.eos_token_id, temperature=temperature)
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

text = input("Podaj tekst wejściowy: ")
generate_text(text, temperature=0.7)

# Etap 5: Przygotowanie zestawu danych
def create_dataset(pairs, tokenizer):
    dataset = Dataset.from_pandas(pd.DataFrame(pairs))
    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2).values()
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    return DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset
    })

pairs = [
    {"text": "To był wspaniały dzień!", "label": "positive"},
    {"text": "Przykład tekstu do klasyfikacji.", "label": "neutral"},
    {"text": "Książka opisuje przygody bohatera w magicznym świecie.", "label": "context"},
    {"text": "Gdzie rozgrywa się akcja książki?", "label": "question"}
]
dataset = create_dataset(pairs, tokenizer)

# Etap 6: Konfiguracja treningu
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
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
    max_grad_norm=1.0
)

# Etap 7: Trening modelu
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

torch.cuda.empty_cache()
try:
    trainer.train()
except Exception as e:
    print(f"Training failed with error: {e}")

# Etap 8: Ewaluacja modelu
torch.cuda.empty_cache()

def evaluate_model(model, tokenizer):
    tasks = ["sentiment-analysis", "text-classification", "question-answering"]
    results = {}

    for task in tasks:
        torch.cuda.empty_cache()
        evaluator = pipeline(task, model=model, tokenizer=tokenizer)
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
    
    return results

evaluation_results = evaluate_model(model, tokenizer)
print(evaluation_results)
