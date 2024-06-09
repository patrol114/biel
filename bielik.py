from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    pipeline, 
    EarlyStoppingCallback, 
    AutoConfig
)
import torch
from datasets import Dataset, DatasetDict
import os
import deepspeed

# Set environment variables to avoid parallelism warning and manage CUDA memory allocation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model name
model_name = "speakleash/Bielik-7B-v0.1"

# Load tokenizer and model with lower precision
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    config=config, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(device)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Enable model parallelism and state dict offloading
model.is_parallelizable = True
model.model_parallel = True
model.tie_weights = False  # Prevent weight tying to save memory

# Initialize DeepSpeed
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

# Clear GPU cache before large operations
torch.cuda.empty_cache()

# Initialize the text generation pipeline
text_generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device=0
)

# Function to generate text with added temperature
def generate_text(prompt, temperature=1.0):
    torch.cuda.empty_cache()  # Clear GPU cache before generating text
    sequences = text_generator(
        text_inputs=prompt,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

# Prompt for user input
text = input("Podaj tekst wejściowy: ")

# Generate text
generate_text(text, temperature=0.7)  # Example with temperature set to 0.7

# Function to create dataset from list of question-answer pairs
def create_dataset(pairs, tokenizer):
    dataset = Dataset.from_pandas(pd.DataFrame(pairs))
    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2).values()
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.remove_columns(['text'])
    eval_dataset = eval_dataset.remove_columns(['text'])
    
    return DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset
    })

# Example question-answer pairs
pairs = [
    {"text": "To był wspaniały dzień!", "label": "positive"},
    {"text": "Przykład tekstu do klasyfikacji.", "label": "neutral"},
    {"text": "Książka opisuje przygody bohatera w magicznym świecie.", "label": "context"},
    {"text": "Gdzie rozgrywa się akcja książki?", "label": "question"}
]

# Create dataset
dataset = create_dataset(pairs, tokenizer)

# Prepare data for training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Number of training epochs
    per_device_train_batch_size=1,  # Reduce train batch size to save memory
    per_device_eval_batch_size=1,  # Reduce eval batch size to save memory
    save_steps=2000,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=2000,
    logging_steps=500,
    learning_rate=3e-5,  # Learning rate (cosine annealing will be applied manually)
    weight_decay=0.1,  # Weight decay
    bf16=False,  # Use float16 precision
    fp16=True,  # Enable mixed precision training
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Metric for selecting best model
    greater_is_better=False,  # Lower loss is better
    report_to="none",  # Disable reporting to WANDB
    dataloader_num_workers=2,  # Number of data loader workers
    run_name="bielik-training",  # Run name for tracking
    logging_dir="./logs",  # Directory for storing logs
    save_strategy="epoch",  # Save model at the end of each epoch
    gradient_accumulation_steps=16,  # Increase gradient accumulation steps to reduce memory usage
    lr_scheduler_type='cosine',  # Cosine learning rate schedule
    warmup_steps=2000,  # Warmup iterations
    max_steps=17350,  # Total training iterations
    optim="adamw_torch",  # Use AdamW optimizer
    adam_beta1=0.9,  # AdamW β1
    adam_beta2=0.95,  # AdamW β2
    adam_epsilon=1e-8,  # AdamW epsilon
    max_grad_norm=1.0  # Gradient clipping
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early stopping
)

# Clear GPU cache before training
torch.cuda.empty_cache()

# Start training
try:
    trainer.train()
except Exception as e:
    print(f"Training failed with error: {e}")

# Clear GPU cache before evaluation
torch.cuda.empty_cache()

# Function to evaluate the model with additional metrics
def evaluate_model(model, tokenizer):
    tasks = ["sentiment-analysis", "text-classification", "question-answering"]
    results = {}

    for task in tasks:
        torch.cuda.empty_cache()  # Clear GPU cache before each task
        evaluator = pipeline(task, model=model, tokenizer=tokenizer)
        # Example inputs for each task
        inputs = {
            "sentiment-analysis": "To był wspaniały dzień!",
            "text-classification": "Przykład tekstu do klasyfikacji.",
            "question-answering": {
                "context": "Książka opisuje przygody bohatera w magicznym świecie.",
                "question": "Gdzie rozgrywa się akcja książki?"
            }
        }
        
        # Perform evaluation for each task
        if task == "question-answering":
            result = evaluator(question=inputs[task]["question"], context=inputs[task]["context"])
        else:
            result = evaluator(inputs[task])
        
        results[task] = result
    
    return results

# Conduct model evaluation
evaluation_results = evaluate_model(model, tokenizer)
print(evaluation_results)
