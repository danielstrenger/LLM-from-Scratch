from torch import Tensor, nn

from models.attention import TransformerBlock
from models.embeddings import SimpleEmbedding


class GPTModel(nn.Module):
    """Decoder-only GPT-style language model (embeddings, blocks, LM head)."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = SimpleEmbedding(vocab_size, context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute next-token logits for every position.

        Args:
            input_ids: Token indices of shape ``(batch, seq)``.

        Returns:
            Logits of shape ``(batch, seq, vocab_size)``.
        """
        x = self.embedding(input_ids)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
