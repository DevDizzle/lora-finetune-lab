# 🗺️ Execution Guide

Step-by-step instructions to run the full fine-tuning pipeline.  
**Estimated time: 4–6 hours** (mostly GPU training time).

---

## Prerequisites

- HuggingFace account (free) — [huggingface.co](https://huggingface.co)
- Weights & Biases account (free) — [wandb.ai](https://wandb.ai)
- GPU runtime with ≥16 GB VRAM (see Step 2)

---

## Step 1: Pick a Dataset

The default is **tatsu-lab/alpaca** (52K instruction-following examples). Other options:

| Dataset | Size | Task |
|---------|------|------|
| `tatsu-lab/alpaca` | 52K | General instruction following |
| `databricks/databricks-dolly-15k` | 15K | General instruction following |
| `OpenAssistant/oasst1` | 88K | Conversational |

To change, edit `configs/training_config.yaml` → `data.dataset_name`.

---

## Step 2: Get a GPU Runtime

### Option A: Google Colab Pro ($10/month)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU** (or A100 if available)
3. Upload the repo or clone from GitHub:
```bash
!git clone https://github.com/YOUR_USER/lora-finetune-lab.git
%cd lora-finetune-lab
!pip install -r requirements.txt
```

### Option B: Vertex AI Workbench (GCP)
1. Console → Vertex AI → Workbench → Create Instance
2. Select **n1-standard-8** with **NVIDIA T4** or **L4** GPU
3. Open JupyterLab, clone repo, install deps

### Option C: RunPod / Lambda Labs (on-demand)
1. Rent an A100 40GB instance (~$1.50/hr)
2. SSH in, clone repo, install deps

---

## Step 3: Setup & Authenticate

```bash
# Clone and install
git clone https://github.com/YOUR_USER/lora-finetune-lab.git
cd lora-finetune-lab
pip install -r requirements.txt

# Authenticate HuggingFace (for gated models like Mistral)
huggingface-cli login
# Paste your HF token (get from https://huggingface.co/settings/tokens)

# Authenticate W&B
wandb login
# Paste your API key (get from https://wandb.ai/authorize)
```

---

## Step 4: Prepare Data

```bash
python src/data_prep.py --dataset tatsu-lab/alpaca --output data/
```

This downloads, formats, splits, and saves the dataset. Check `data/stats.json` for summary.

---

## Step 5: Fine-Tune

```bash
# Run as script
python -m notebooks.finetune

# OR open in Jupyter/Colab as notebook
# (rename .py → .ipynb or use jupytext)
jupytext --to notebook notebooks/finetune.py
jupyter notebook notebooks/finetune.ipynb
```

**Expected runtime:**
- T4 (16GB): ~3–4 hours for 3 epochs on Alpaca
- A100 (40GB): ~45 min

Monitor training at [wandb.ai](https://wandb.ai) → your project.

---

## Step 6: Evaluate

```bash
python -m notebooks.evaluate
```

This computes perplexity, ROUGE scores, and logs sample generations to W&B.

---

## Step 7: Test Inference

```bash
# Single prompt
python src/inference.py --adapter outputs/lora-adapter/final \
    --prompt "Explain the difference between LoRA and full fine-tuning."

# Interactive mode
python src/inference.py --adapter outputs/lora-adapter/final --interactive
```

---

## Step 8: Update README with Results

1. Copy metrics from W&B or terminal output
2. Fill in the Results table in `README.md`
3. Add training time and hardware used
4. Take a screenshot of the W&B dashboard (optional)
5. Commit and push:

```bash
git add -A
git commit -m "Add fine-tuning results"
git push origin main
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on T4 | Reduce `per_device_train_batch_size` to 2, increase `gradient_accumulation_steps` to 16 |
| Flash Attention not available | Remove `attn_implementation="flash_attention_2"` from model loading |
| HF gated model error | Run `huggingface-cli login` and accept model license on HF website |
| W&B not logging | Run `wandb login` or set `report_to: "none"` in training config |

---

## Time Budget

| Task | Time |
|------|------|
| Setup & auth | 15 min |
| Data prep | 5 min |
| Fine-tuning (T4) | 3–4 hrs |
| Evaluation | 30 min |
| Inference testing | 15 min |
| Update README | 15 min |
| **Total** | **4–5 hrs** |
