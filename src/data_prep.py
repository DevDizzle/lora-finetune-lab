"""
Dataset preprocessing utilities for LoRA fine-tuning.

Usage:
    python src/data_prep.py --dataset tatsu-lab/alpaca --output data/
    python src/data_prep.py --dataset databricks/databricks-dolly-15k --output data/
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, load_dataset


# ── Prompt templates ──────────────────────────────────────────────

ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


# ── Dataset formatters ────────────────────────────────────────────

def format_alpaca(example: dict) -> dict:
    """Format tatsu-lab/alpaca examples."""
    if example.get("input", "").strip():
        text = ALPACA_TEMPLATE.format(**example)
    else:
        text = ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text}


def format_dolly(example: dict) -> dict:
    """Format databricks/databricks-dolly-15k examples."""
    return format_alpaca({
        "instruction": example["instruction"],
        "input": example.get("context", ""),
        "output": example["response"],
    })


def format_oasst(example: dict) -> dict:
    """Format OpenAssistant examples (first turn only)."""
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}


FORMATTERS = {
    "tatsu-lab/alpaca": format_alpaca,
    "databricks/databricks-dolly-15k": format_dolly,
    "OpenAssistant/oasst1": format_oasst,
}


# ── Core functions ────────────────────────────────────────────────

def load_and_format(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
    num_workers: int = 4,
) -> Dataset:
    """Load a HuggingFace dataset and format it for instruction tuning."""
    print(f"Loading {dataset_name} (split={split})...")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"  Subsampled to {len(ds)} examples")

    formatter = FORMATTERS.get(dataset_name)
    if formatter is None:
        raise ValueError(
            f"No formatter for '{dataset_name}'. "
            f"Supported: {list(FORMATTERS.keys())}"
        )

    ds = ds.map(
        formatter,
        num_proc=num_workers,
        remove_columns=ds.column_names,
        desc="Formatting",
    )

    print(f"  Formatted {len(ds)} examples")
    return ds


def create_splits(
    dataset: Dataset,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """Split dataset into train and validation sets."""
    splits = dataset.train_test_split(test_size=val_ratio, seed=seed)
    print(f"  Train: {len(splits['train']):,} | Val: {len(splits['test']):,}")
    return splits


def compute_stats(dataset: Dataset) -> dict:
    """Compute basic dataset statistics."""
    lengths = [len(ex["text"].split()) for ex in dataset]
    return {
        "num_examples": len(dataset),
        "avg_words": sum(lengths) / len(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "median_words": sorted(lengths)[len(lengths) // 2],
    }


def save_dataset(splits: DatasetDict, output_dir: str) -> None:
    """Save processed dataset splits to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits.save_to_disk(str(output_path / "processed"))

    # Also save as JSONL for inspection
    for split_name, ds in splits.items():
        jsonl_path = output_path / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for example in ds:
                f.write(json.dumps(example) + "\n")
        print(f"  Saved {jsonl_path} ({len(ds)} examples)")

    # Save stats
    stats = {
        split_name: compute_stats(ds) for split_name, ds in splits.items()
    }
    stats_path = output_path / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA fine-tuning")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help=f"HuggingFace dataset name. Supported: {list(FORMATTERS.keys())}",
    )
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for debugging)")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    dataset = load_and_format(
        args.dataset,
        max_samples=args.max_samples,
        num_workers=args.workers,
    )

    splits = create_splits(dataset, val_ratio=args.val_ratio, seed=args.seed)
    save_dataset(splits, args.output)

    print("\nDone! Dataset ready for training.")


if __name__ == "__main__":
    main()
