from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline, EarlyStoppingCallback, AutoConfig
import torch
from datasets import load_dataset, DatasetDict
import os
import deepspeed
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box

console = Console()

# Ustawienia środowiskowe
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Etap 2: Załadowanie tokenizera i konfiguracji modelu
model_name = "speakleash/Bielik-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
config = AutoConfig.from_pretrained(model_name, force_download=True)

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
    table = Table(title="Opcje Menu", box=box.ROUNDED)
    table.add_column("Numer", justify="right", style="cyan", no_wrap=True)
    table.add_column("Opis", style="magenta")
    table.add_row("1", "Generowanie tekstu")
    table.add_row("2", "Trening modelu")
    table.add_row("3", "Strojenie hiperparametrów")
    table.add_row("4", "Ewaluacja modelu")
    table.add_row("5", "Logowanie danych tekstowych")
    console.print(table)
    
    choice = input("Wybierz opcję: ")
    if choice == '1':
        text = input("Podaj tekst wejściowy: ")
        console.print(generate_text(text, temperature=0.7), style="bold green")
    elif choice == '2':
        dataset = load_and_prepare_dataset(tokenizer)
        train_model(dataset)
    elif choice == '3':
        hyperparameter_tuning()
    elif choice == '4':
        results = evaluate_model(model_name, tokenizer)
        console.print(results, style="bold blue")
    elif choice == '5':
        log_text_data()
    else:
        console.print("Niepoprawny wybór, spróbuj ponownie.", style="bold red")

# Funkcja generowania tekstu
def generate_text(prompt, temperature=0.7):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, force_download=True).to(device)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    generated_text = text_generator(prompt, max_length=2096, temperature=temperature, num_return_sequences=1, do_sample=True)[0]['generated_text']
    del model  # Uwalnianie pamięci
    torch.cuda.empty_cache()
    return generated_text

# Funkcja ładowania i przygotowania zestawu danych
def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("allegro/polish-question-passage-pairs", cache_format='arrow')
    
    def tokenize_function(examples):
        # Dodano obsługę wyjątków
        try:
            return tokenizer(examples['question'], padding="max_length", truncation=True)
        except Exception as e:
            console.print(f"Błąd podczas tokenizacji: {e}", style="bold red")
            return None
    
    # Tokenizacja datasetu z obsługą błędów
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['id', 'question', 'passage'], num_proc=4)
    
    # Usunięcie None wartości, jeśli występują błędy w tokenizacji
    tokenized_dataset = tokenized_dataset.filter(lambda x: x is not None)

    return DatasetDict({
        "train": tokenized_dataset["train"],
        "eval": tokenized_dataset["validation"]
    })

# Funkcja treningu modelu
def train_model(dataset):
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, force_download=True).to(device)
    model.gradient_checkpointing_enable()
    model.is_parallelizable = True
    model.model_parallel = True
    model.tie_weights = False
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    
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
        report_to="tensorboard",
        logging_dir=log_dir,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), tensorboard_callback]
    )
    
    try:
        console.print("Rozpoczynam trening modelu...", style="bold blue")
        trainer.train()
        console.print("Trening zakończony sukcesem.", style="bold green")
    except Exception as e:
        console.print(f"Trening nie powiódł się z błędem: {e}", style="bold red")
    del model  # Uwalnianie pamięci
    torch.cuda.empty_cache()

# Dodatkowa funkcja ewaluacji modelu, która może być wywołana w ramach treningu lub osobno
def evaluate_model(model_name, tokenizer):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float16, force_download=True).to(device)
    tasks = ["sentiment-analysis", "text-classification", "question-answering"]
    results = {}
    for task in tasks:
        console.print(f"Rozpoczynam ewaluację dla zadania: {task}", style="bold blue")
        evaluator = pipeline(task, model=model, tokenizer=tokenizer, device=0)
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
        console.print(f"Wynik dla {task}: {result}", style="bold green")
    del model  # Uwalnianie pamięci
    torch.cuda.empty_cache()
    return results

# Logowanie danych tekstowych
def log_text_data():
    logdir = "logs/text_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        tf.summary.text("Sample text", "This is a sample log of text data", step=0)
    # Zamiast magicznego polecenia, instrukcja jak uruchomić tensorboard z terminala
    console.print(f"Aby uruchomić TensorBoard, wykonaj polecenie: tensorboard --logdir {logdir}", style="bold blue")

# Funkcja treningu modelu z hiperparametrami
def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(x_train, y_train, epochs=1)  # Trening z jednym epokiem dla szybszej demonstracji
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # zapisanie użytych wartości hiperparametrów
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

# Funkcja strojenia hiperparametrów
def hyperparameter_tuning():
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    METRIC_ACCURACY = 'accuracy'
    
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )
    
    session_num = 0
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer
                }
                run_name = f"run-{session_num}"
                console.print(f"--- Uruchamianie: {run_name}", style="bold blue")
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1

# Uruchomienie menu, aby wybrać operację
if __name__ == "__main__":
    menu()
