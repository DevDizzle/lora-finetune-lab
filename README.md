# 🧬 LoRA Fine-Tune Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://huggingface.co/docs/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Parameter-efficient fine-tuning of Mistral-7B using LoRA**, demonstrating production-grade ML engineering: dataset preparation, 4-bit quantized training, evaluation with automated metrics, and efficient inference serving.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LoRA Fine-Tune Lab                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Data Prep   │───▶│  Fine-Tune   │───▶│    Evaluation    │  │
│  │              │    │              │    │                  │  │
│  │ • Load HF    │    │ • QLoRA 4bit │    │ • Perplexity     │  │
│  │   dataset    │    │ • r=16 α=32  │    │ • ROUGE scores   │  │
│  │ • Tokenize   │    │ • Grad accum │    │ • Task metrics   │  │
│  │ • Train/val  │    │ • Checkpoints│    │ • W&B logging    │  │
│  │   split      │    │              │    │                  │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────┘  │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │   LoRA Adapter  │                          │
│                    │   (~16 MB)      │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │   Inference     │                          │
│                    │   CLI / API     │                          │
│                    │                 │                          │
│                    │ Base + Adapter  │                          │
│                    │ 4-bit quantized │                          │
│                    └─────────────────┘                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Tech: Transformers · PEFT · Accelerate · bitsandbytes · W&B   │
└─────────────────────────────────────────────────────────────────┘
```

## What This Demonstrates

| Skill | Implementation |
|-------|---------------|
| **Parameter-efficient fine-tuning** | LoRA with rank-16 decomposition on attention projections |
| **Memory-efficient training** | QLoRA 4-bit quantization via bitsandbytes NF4 |
| **Training engineering** | Gradient accumulation, cosine scheduling, warmup |
| **Evaluation rigor** | Perplexity, ROUGE-L, task-specific accuracy |
| **Experiment tracking** | Weights & Biases integration with artifact logging |
| **Production inference** | Merged adapter loading with CLI interface |
| **Reproducibility** | YAML configs, fixed seeds, pinned dependencies |

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USER/lora-finetune-lab.git
cd lora-finetune-lab
pip install -r requirements.txt

# Prepare data
python src/data_prep.py --dataset tatsu-lab/alpaca --output data/

# Fine-tune (requires GPU — see GUIDE.md)
python -m notebooks.finetune

# Evaluate
python -m notebooks.evaluate

# Run inference
python src/inference.py --adapter outputs/lora-adapter --prompt "Explain LoRA in one paragraph."
```

## Project Structure

```
lora-finetune-lab/
├── configs/
│   ├── lora_config.yaml          # LoRA hyperparameters
│   └── training_config.yaml      # Training loop config
├── notebooks/
│   ├── finetune.py               # Fine-tuning notebook (percent script)
│   └── evaluate.py               # Evaluation notebook (percent script)
├── src/
│   ├── data_prep.py              # Dataset preprocessing
│   └── inference.py              # Inference CLI
├── requirements.txt
├── GUIDE.md                      # Step-by-step execution guide
└── README.md
```

## Results

**Run Date:** February 12, 2026
**Hardware:** NVIDIA A100
**Training Time:** ~3 hours (2h 59m 54s)
**WandB Report:** [View Run Logs](https://wandb.ai/eraphaelparra-evanparra-ai/lora-finetune-lab/runs/0gxeyqwe)

| Metric | Value | Notes |
|--------|-------|-------|
| **Eval Loss** | 1.187 | Low validation loss indicates good generalization |
| **Eval Entropy** | 0.888 | Measure of prediction uncertainty |
| **Token Accuracy** | 70.78% | Percentage of tokens predicted correctly |
| **Train Loss** | 0.890 | Final training loss after 3 epochs |

**Example Inference:**
> **Instruction:** Write a python function to reverse a list.
>
> **Response:**
> ```python
> def reverse_list(lst):
>    return lst[::-1]
> ```

**Training Details:**
- Base model: `mistralai/Mistral-7B-v0.3`
- Trainable parameters: ~4.2M / 7.2B (0.06%)
- Epochs: 3
- Speed: 13.73 samples/sec (train), 61.35 samples/sec (eval)

## License

MIT
