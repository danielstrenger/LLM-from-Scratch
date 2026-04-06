import torch
from torch import nn


def sample_with_temperature(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Autoregressively sample tokens with temperature-scaled multinomial sampling.

    Temporarily sets ``model`` to eval mode (restored afterward). Appends
    ``max_new_tokens`` tokens to ``input_ids`` along sequence dimension 1.

    Args:
        model: Module that maps token ids ``(batch, seq)`` to logits
            ``(batch, seq, vocab)``.
        input_ids: Starting token indices, shape ``(batch, n_tokens)``.
        max_new_tokens: How many new tokens to generate.
        context_size: Maximum sequence length passed to the model each step
            (last ``context_size`` positions are used).
        temperature: Softmax temperature; must be strictly positive.

    Returns:
        Token ids including the original context and generated suffix,
        shape ``(batch, n_tokens + max_new_tokens)``.

    Raises:
        ValueError: If ``temperature <= 0``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)

            logits = logits[:, -1, :]  # (batch, vocab_size)

            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values

            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            idx_next = torch.multinomial(probas, num_samples=1)  # (batch, 1)

            input_ids = torch.cat((input_ids, idx_next), dim=1)  # (batch, n_tokens+1)
    finally:
        if was_training:
            model.train()

    return input_ids


def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedy autoregressive generation (argmax next token each step).

    Args:
        model: Module that maps token ids ``(batch, seq)`` to logits
            ``(batch, seq, vocab)``.
        idx: Starting token indices, shape ``(batch, n_tokens)``.
        max_new_tokens: How many new tokens to generate.
        context_size: Maximum sequence length passed to the model each step.

    Returns:
        Token ids including the original context and generated suffix,
        shape ``(batch, n_tokens + max_new_tokens)``.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # (batch, vocab_size)

        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
