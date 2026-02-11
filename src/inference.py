"""
Inference CLI for LoRA fine-tuned Mistral-7B.

Usage:
    python src/inference.py --adapter outputs/lora-adapter/final --prompt "Your prompt here"
    python src/inference.py --adapter outputs/lora-adapter/final --interactive
"""

import argparse
import sys

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer


def load_model(
    adapter_path: str,
    base_model: str | None = None,
    use_4bit: bool = True,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load base model with LoRA adapter merged."""
    if base_model is None:
        with open("configs/lora_config.yaml") as f:
            cfg = yaml.safe_load(f)
        base_model = cfg["model"]["name"]

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def generate(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = True,
) -> str:
    """Generate a response from an instruction prompt."""
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_special_tokens=True) if stream else None

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    return response


def interactive_loop(model: PeftModel, tokenizer: AutoTokenizer) -> None:
    """Run an interactive chat loop."""
    print("\n🧬 LoRA Fine-Tune Lab — Interactive Inference")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("\nModel: ", end="", flush=True)
        generate(model, tokenizer, prompt, stream=True)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA inference CLI")
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/lora-adapter/final",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (reads from config if not specified)",
    )
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-4bit", action="store_true", help="Load in full precision")

    args = parser.parse_args()

    model, tokenizer = load_model(
        adapter_path=args.adapter,
        base_model=args.base_model,
        use_4bit=not args.no_4bit,
    )

    if args.interactive:
        interactive_loop(model, tokenizer)
    elif args.prompt:
        response = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if not response:
            print("(empty response)")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
