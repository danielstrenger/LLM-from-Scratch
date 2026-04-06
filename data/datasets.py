from typing import Any, List, Tuple

import tiktoken
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    """Sliding-window dataset of input/target token sequences for LM training."""

    def __init__(self, txt: str, tokenizer: Any, max_length: int, stride: int) -> None:
        """Build overlapping chunks of length ``max_length`` from ``txt``.

        Args:
            txt: Raw text to tokenize.
            tokenizer: Object with ``encode`` (e.g. tiktoken ``Encoding``).
            max_length: Sequence length for each example.
            stride: Step between window starts (smaller = more overlap).
        """
        self.input_ids: List[Tensor] = []
        self.target_ids: List[Tensor] = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, (
            "Number of tokenized inputs must at least be equal to max_length+1"
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a ``DataLoader`` over ``GPTDatasetV1`` using GPT-2 tiktoken encoding.

    Args:
        txt: Corpus text.
        batch_size: Batch size for the loader.
        max_length: Token length per sequence.
        stride: Window stride for chunking.
        shuffle: Whether to shuffle batches.
        drop_last: Drop last incomplete batch if True.
        num_workers: ``DataLoader`` worker processes.

    Returns:
        PyTorch ``DataLoader`` yielding ``(input_ids, target_ids)`` batches.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
