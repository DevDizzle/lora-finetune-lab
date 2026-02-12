# %% [markdown]
# # Evaluation: LoRA Fine-Tuned Mistral-7B
#
# Load the fine-tuned adapter and evaluate on held-out test data.
# Metrics: perplexity, ROUGE-L, instruction-following accuracy.
# Results logged to Weights & Biases.

# %% Imports
import os
import yaml
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
import wandb

# %% Load configs
with open("configs/lora_config.yaml") as f:
    lora_cfg = yaml.safe_load(f)

with open("configs/training_config.yaml") as f:
    train_cfg = yaml.safe_load(f)

MODEL_NAME = lora_cfg["model"]["name"]
MAX_SEQ_LEN = lora_cfg["model"]["max_seq_length"]
ADAPTER_PATH = os.path.join(train_cfg["training"]["output_dir"], "final")

print(f"Base model: {MODEL_NAME}")
print(f"Adapter: {ADAPTER_PATH}")

# %% Initialize W&B
wandb.init(
    project=train_cfg["wandb"]["project"],
    name="evaluation",
    job_type="eval",
)

# %% Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # left-pad for generation

# %% Load quantized base model + LoRA adapter
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model loaded successfully.")

# %% Load evaluation dataset
# FIX: Removed trust_remote_code=True
dataset = load_dataset(
    train_cfg["data"]["dataset_name"],
    split="train",
)

# Use the same split logic to get the eval portion
split = dataset.train_test_split(
    test_size=train_cfg["data"]["val_split_ratio"],
    seed=train_cfg["training"]["seed"],
)
eval_dataset = split["test"]

# Cap eval set for reasonable runtime
MAX_EVAL = 500
if len(eval_dataset) > MAX_EVAL:
    eval_dataset = eval_dataset.select(range(MAX_EVAL))

print(f"Evaluation samples: {len(eval_dataset)}")

# %% Prompt formatting (instruction only — no output)
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
"""


def make_prompt(example: dict) -> str:
    if example.get("input", "").strip():
        return PROMPT_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
        )
    return PROMPT_TEMPLATE_NO_INPUT.format(instruction=example["instruction"])


# %% 1. Compute Perplexity
print("\n--- Computing Perplexity ---")

total_loss = 0.0
total_tokens = 0

for example in tqdm(eval_dataset, desc="Perplexity"):
    prompt = make_prompt(example)
    full_text = prompt + example["output"]

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)
print(f"Perplexity: {perplexity:.2f}")

# %% 2. Generate responses and compute ROUGE
print("\n--- Generating Responses & Computing ROUGE ---")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
generations = []

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.15,
}

for example in tqdm(eval_dataset.select(range(min(200, len(eval_dataset)))), desc="Generation"):
    prompt = make_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(
        model.device
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    reference = example["output"].strip()

    # ROUGE scores
    scores = scorer.score(reference, generated)
    for key in rouge_scores:
        rouge_scores[key].append(scores[key].fmeasure)

    generations.append(
        {
            "instruction": example["instruction"],
            "reference": reference,
            "generated": generated,
        }
    )

# Aggregate ROUGE
avg_rouge = {k: np.mean(v) for k, v in rouge_scores.items()}
for k, v in avg_rouge.items():
    print(f"{k}: {v:.4f}")

# %% 3. Task-specific: instruction-following quality (length & format heuristics)
print("\n--- Instruction-Following Quality ---")

non_empty_rate = sum(1 for g in generations if len(g["generated"]) > 10) / len(generations)
avg_gen_length = np.mean([len(g["generated"].split()) for g in generations])
avg_ref_length = np.mean([len(g["reference"].split()) for g in generations])

print(f"Non-empty response rate: {non_empty_rate:.1%}")
print(f"Avg generated length: {avg_gen_length:.0f} words")
print(f"Avg reference length: {avg_ref_length:.0f} words")

# %% Log all metrics to W&B
metrics = {
    "eval/perplexity": perplexity,
    "eval/rouge1": avg_rouge["rouge1"],
    "eval/rouge2": avg_rouge["rouge2"],
    "eval/rougeL": avg_rouge["rougeL"],
    "eval/non_empty_rate": non_empty_rate,
    "eval/avg_gen_length_words": avg_gen_length,
    "eval/avg_ref_length_words": avg_ref_length,
    "eval/num_samples": len(eval_dataset),
}
wandb.log(metrics)

# Log sample generations as a W&B table
columns = ["instruction", "reference", "generated"]
table = wandb.Table(columns=columns)
for g in generations[:50]:  # log first 50
    table.add_data(g["instruction"], g["reference"], g["generated"])
wandb.log({"eval/generations": table})

# %% Summary
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"  Perplexity:    {perplexity:.2f}")
print(f"  ROUGE-1:       {avg_rouge['rouge1']:.4f}")
print(f"  ROUGE-2:       {avg_rouge['rouge2']:.4f}")
print(f"  ROUGE-L:       {avg_rouge['rougeL']:.4f}")
print(f"  Response rate: {non_empty_rate:.1%}")
print("=" * 60)

wandb.finish()
print("\nEvaluation complete. Results logged to W&B.")
