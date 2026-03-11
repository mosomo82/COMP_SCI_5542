import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_NAME = "microsoft/phi-2"
DATASET_PATH = "instruction_dataset.json"
OUTPUT_DIR = "adapted_model"

# Load dataset
print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Load tokenizer
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization via bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load base model
print(f"Loading base model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# Apply QLoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
def formatting_prompts_func(example):
    instruction = example['instruction']
    input_text = example['input']
    output = example['output']

    prompt = f"Instruct: {instruction}\n"
    if input_text:
        prompt += f"Context: {input_text}\n"
    prompt += f"Output:\n{output}"

    return prompt

# Setup Training Arguments
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_8bit"
)

# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    formatting_func=formatting_prompts_func,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the adapted model
print(f"Saving adapted model to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Inference on 5 sample queries (baseline vs adapted)
print("-" * 50)
print("Running Inference Comparison")
print("-" * 50)

sample_queries = data[:5]

def generate_response(model_to_use, query_dict):
    instruction = query_dict["instruction"]
    input_text = query_dict["input"]
    prompt = f"Instruct: {instruction}\n"
    if input_text:
        prompt += f"Context: {input_text}\n"
    prompt += "Output:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model_to_use.device)
    with torch.no_grad():
        outputs = model_to_use.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Output:\n")[-1].strip()

print("Loading raw base model for baseline comparison...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

for i, query in enumerate(sample_queries):
    print(f"\n--- Sample {i+1} ---")
    print(f"Query: {query['instruction']}")
    
    baseline_response = generate_response(baseline_model, query)
    print(f"\n[Baseline Model]:\n{baseline_response}")
    
    # Reload model for generation or use current model wrapper
    # Here `model` is the PEFT model
    model.eval()
    adapted_response = generate_response(model, query)
    model.train() # switch back if necessary, but we are done training
    print(f"\n[Adapted Model]:\n{adapted_response}")
