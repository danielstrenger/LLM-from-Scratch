from typing import Optional

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, MultiheadAttention


class CausalSelfAttention(Module):
    """Multi-head self-attention with causal masking (batch-first)."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal self-attention.

        Args:
            x: Input of shape ``(batch, seq_len, embedding_dim)``.

        Returns:
            Same shape as ``x``.
        """
        _, seq_len, _ = x.shape
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_output, _ = self.attention(
            x, x, x, is_causal=True, attn_mask=mask, need_weights=False
        )
        return attn_output


class TransformerBlock(Module):
    """Pre-norm transformer block: attention + feed-forward with residuals."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.norm1 = LayerNorm(embedding_dim)

        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embedding_dim
        self.ffn = torch.nn.Sequential(
            Linear(embedding_dim, ff_hidden_dim),
            torch.nn.GELU(),
            Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Run attention sublayer then FFN sublayer.

        Args:
            x: Tensor of shape ``(batch, seq_len, embedding_dim)``.

        Returns:
            Same shape as ``x``.
        """
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
