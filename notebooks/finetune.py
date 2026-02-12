# %% [markdown]
# # LoRA Fine-Tuning: Mistral-7B
#
# Parameter-efficient fine-tuning using QLoRA (4-bit quantization + LoRA adapters).
# This notebook trains a Mistral-7B model on the Alpaca instruction-following dataset.
#
# **Requirements:** GPU with ≥16GB VRAM (T4, L4, A100)

# %% Imports
import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb

# %% Load configs
with open("configs/lora_config.yaml") as f:
    lora_cfg = yaml.safe_load(f)

with open("configs/training_config.yaml") as f:
    train_cfg = yaml.safe_load(f)

MODEL_NAME = lora_cfg["model"]["name"]
MAX_SEQ_LEN = lora_cfg["model"]["max_seq_length"]

print(f"Model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_SEQ_LEN}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% Load and prepare dataset
dataset = load_dataset(
    train_cfg["data"]["dataset_name"],
    split="train",
    trust_remote_code=True,
)

if train_cfg["data"]["max_samples"]:
    dataset = dataset.select(range(train_cfg["data"]["max_samples"]))

# Train/val split
split = dataset.train_test_split(
    test_size=train_cfg["data"]["val_split_ratio"],
    seed=train_cfg["training"]["seed"],
)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train samples: {len(train_dataset):,}")
print(f"Eval samples: {len(eval_dataset):,}")

# %% Format into instruction template
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


def format_alpaca(example: dict) -> dict:
    """Format Alpaca examples into instruction-following prompts."""
    if example.get("input", "").strip():
        text = PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text}


train_dataset = train_dataset.map(
    format_alpaca,
    num_proc=train_cfg["data"]["preprocessing_num_workers"],
    remove_columns=train_dataset.column_names,
)
eval_dataset = eval_dataset.map(
    format_alpaca,
    num_proc=train_cfg["data"]["preprocessing_num_workers"],
    remove_columns=eval_dataset.column_names,
)

print(f"\nSample formatted prompt:\n{train_dataset[0]['text'][:500]}")

# %% Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %% Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=lora_cfg["quantization"]["load_in_4bit"],
    bnb_4bit_quant_type=lora_cfg["quantization"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, lora_cfg["quantization"]["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=lora_cfg["quantization"]["bnb_4bit_use_double_quant"],
)

# %% Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Prepare model for k-bit training (freeze base, cast layernorm to fp32)
model = prepare_model_for_kbit_training(model)

# %% Configure LoRA
peft_config = LoraConfig(
    r=lora_cfg["lora"]["r"],
    lora_alpha=lora_cfg["lora"]["lora_alpha"],
    lora_dropout=lora_cfg["lora"]["lora_dropout"],
    target_modules=lora_cfg["lora"]["target_modules"],
    bias=lora_cfg["lora"]["bias"],
    task_type=lora_cfg["lora"]["task_type"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %% Training arguments
tc = train_cfg["training"]
training_args = TrainingArguments(
    output_dir=tc["output_dir"],
    num_train_epochs=tc["num_train_epochs"],
    per_device_train_batch_size=tc["per_device_train_batch_size"],
    per_device_eval_batch_size=tc["per_device_eval_batch_size"],
    gradient_accumulation_steps=tc["gradient_accumulation_steps"],
    learning_rate=tc["learning_rate"],
    weight_decay=tc["weight_decay"],
    warmup_ratio=tc["warmup_ratio"],
    lr_scheduler_type=tc["lr_scheduler_type"],
    logging_steps=tc["logging_steps"],
    eval_strategy=tc["eval_strategy"],
    eval_steps=tc["eval_steps"],
    save_strategy=tc["save_strategy"],
    save_steps=tc["save_steps"],
    save_total_limit=tc["save_total_limit"],
    fp16=tc["fp16"],
    bf16=tc["bf16"],
    gradient_checkpointing=tc["gradient_checkpointing"],
    optim=tc["optim"],
    max_grad_norm=tc["max_grad_norm"],
    report_to=tc["report_to"],
    seed=tc["seed"],
)

# %% Initialize W&B
wandb.init(
    project=train_cfg["wandb"]["project"],
    name=train_cfg["wandb"].get("run_name"),
    config={
        "model": MODEL_NAME,
        "lora": lora_cfg["lora"],
        "training": tc,
    },
)

# %% Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    max_seq_length=MAX_SEQ_LEN,
)

print("Starting training...")
trainer.train()

# %% Save adapter
adapter_path = os.path.join(tc["output_dir"], "final")
trainer.save_model(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"\nAdapter saved to: {adapter_path}")

# %% Log adapter as W&B artifact
artifact = wandb.Artifact("lora-adapter", type="model")
artifact.add_dir(adapter_path)
wandb.log_artifact(artifact)
wandb.finish()

print("Training complete!")
