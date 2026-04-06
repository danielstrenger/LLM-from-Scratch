"""Shared JSON schema and loader for train / generate scripts."""

import json
from typing import Any, Dict

CONFIG_REQUIRED_KEYS = [
    "vocab_size",
    "context_length",
    "embedding_dim",
    "num_layers",
    "num_heads",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "weight_decay",
    "train_split",
    "data_path",
    "checkpoint_dir",
    "model_checkpoint",
    "optim_checkpoint",
]


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate training / inference JSON config from ``path``.

    Args:
        path: Filesystem path to a JSON file containing all keys in
            ``CONFIG_REQUIRED_KEYS``.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        ValueError: If any required key is missing.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    missing = [k for k in CONFIG_REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    return cfg
