from typing import Optional

import torch
from torch import Tensor
from torch.nn import Embedding, Module


class SimpleEmbedding(Module):
    """Sum of learned token and position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.token_embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = Embedding(context_length, embedding_dim)

    def forward(
        self,
        token_ids: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Embed tokens and positions.

        Args:
            token_ids: Shape ``(batch, seq)`` integer token indices.
            position_ids: Optional same shape as ``token_ids``; if omitted,
                positions ``0 .. seq-1`` are used per row.

        Returns:
            Sum of token and position embeddings, shape ``(batch, seq, dim)``.
        """
        if position_ids is None:
            position_ids = torch.arange(
                token_ids.size(1),
                dtype=torch.long,
                device=token_ids.device,
            ).unsqueeze(0).expand_as(token_ids)
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)
        return token_embeddings + position_embeddings
