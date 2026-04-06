import argparse
import os
import random
import json
import numpy as np
from argparse import Namespace

import tiktoken
import torch
from safetensors.torch import load_file, save_file

from config.config import load_config
from data.datasets import create_dataloader_v1
from models.gptmodel import GPTModel
from trainers.trainer import train_model_simple


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config (e.g. config/default.json)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_epochs from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, train (optionally resume), and save checkpoints."""
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    num_epochs = args.epochs if args.epochs is not None else cfg["num_epochs"]

    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    embedding_dim = cfg["embedding_dim"]
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    weight_decay = cfg["weight_decay"]
    train_split = cfg["train_split"]
    data_path = cfg["data_path"]
    checkpoint_dir = cfg["checkpoint_dir"]
    model_ckpt_path = os.path.join(checkpoint_dir, cfg["model_checkpoint"])
    optim_ckpt_path = os.path.join(checkpoint_dir, cfg["optim_checkpoint"])
    history_path = os.path.join(checkpoint_dir, "loss_history.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPTModel(
        vocab_size, context_length, embedding_dim, num_layers, num_heads
    ).to(device)

    # If a checkpoint file exists already, resume training.
    if os.path.exists(model_ckpt_path):
        state_dict = load_file(model_ckpt_path, device=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {model_ckpt_path}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the dataset exists.")

    with open(data_path, "r", encoding="utf-8") as f:
        txt = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

    split_idx = int(train_split * len(token_ids))
    split_idx = max(split_idx, context_length + 1)

    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_txt = tokenizer.decode(train_ids)
    dataloader = create_dataloader_v1(
        train_txt,
        batch_size=batch_size,
        max_length=context_length,
        shuffle=True,
    )

    if len(val_ids) > context_length + 1:
        val_txt = tokenizer.decode(val_ids)
        val_dataloader = create_dataloader_v1(
            val_txt,
            batch_size=batch_size,
            max_length=context_length,
            shuffle=False,
            drop_last=False,
        )
    else:
        val_dataloader = None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Load optimizer state.
    if os.path.exists(optim_ckpt_path):
        optimizer.load_state_dict(
            torch.load(optim_ckpt_path, map_location=device)
        )
        print(f"Loaded optimizer state: {optim_ckpt_path}")
    
    # Simple Linear Warmup or StepLR could be added here
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    loss_fn = torch.nn.CrossEntropyLoss()

    history = train_model_simple(
        model,
        dataloader,
        optimizer,
        loss_fn,
        device=device,
        num_epochs=num_epochs,
        val_loader=val_dataloader,
        scheduler=scheduler,
        max_grad_norm=1.0
    )

    # Save model, optimizer state, and loss history.
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_file(model.state_dict(), model_ckpt_path)
    torch.save(optimizer.state_dict(), optim_ckpt_path)
    
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Loss history saved to {history_path}")


if __name__ == "__main__":
    main()
