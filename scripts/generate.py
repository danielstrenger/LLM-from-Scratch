from __future__ import annotations

import argparse
import os
from argparse import Namespace

import tiktoken
import torch
from safetensors.torch import load_file

from config.config import load_config
from models.gptmodel import GPTModel
from samplers.samplers import sample_with_temperature


def parse_args() -> Namespace:
    """Parse command-line arguments for text generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config (same as training, e.g. config/default.json)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="And the Lord",
        help="Beginning of the text to generate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Controls randomness: lower values favor likely tokens; "
            "higher values increase diversity."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load model from checkpoint and print a sampled continuation of ``--prompt``."""
    args = parse_args()

    cfg = load_config(args.config)
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    embedding_dim = cfg["embedding_dim"]
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    checkpoint_path = os.path.join(cfg["checkpoint_dir"], cfg["model_checkpoint"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} (from config {args.config}). "
            "Train the model first with scripts/train.py."
        )

    model = GPTModel(
        vocab_size, context_length, embedding_dim, num_layers, num_heads
    )
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    context_ids = tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"})
    input_ids = torch.tensor([context_ids]).to(device)
    output_ids = sample_with_temperature(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        context_size=context_length,
        temperature=args.temperature,
    )
    output_text = tokenizer.decode(output_ids[0].tolist())
    print("Generated Text:\n", output_text)


if __name__ == "__main__":
    main()
