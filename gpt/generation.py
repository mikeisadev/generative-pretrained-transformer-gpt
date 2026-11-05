"""Ultimo step del percorso: generazione autoregressiva.

Il modello prevede UN token alla volta. Lo aggiungiamo al contesto e
ripetiamo: è lo stesso meccanismo "token by token" che nella L16 vedevamo
arrivare in streaming nel frontend React.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


@torch.no_grad()  # in generazione non serve calcolare i gradienti: meno memoria, più velocità
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Allunga ``idx`` (B, T) di ``max_new_tokens`` token campionati dal modello.

    Args:
        model: qualsiasi modello che restituisce ``(logits, loss)`` con logits (B, T, vocab_size).
        block_size: se indicato, il contesto viene tagliato agli ultimi ``block_size``
            token (il GPT ha embedding di posizione solo fino a block_size).
        temperature: <1 rende le scelte più "sicure", >1 più creative. È lo stesso
            parametro ``temperature`` che impostavamo in Spring AI nella L16.

    Returns:
        (B, T + max_new_tokens) la sequenza originale seguita dai token generati.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens non può essere negativo")
    if temperature <= 0:
        raise ValueError("temperature deve essere positiva")

    was_training = model.training
    model.eval()  # disattiva il dropout: in generazione vogliamo il modello "completo"
    try:
        for _ in range(max_new_tokens):
            idx_cond = idx if block_size is None else idx[:, -block_size:]  # (B, ≤block_size)
            logits, _ = model(idx_cond)  # (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size): ci serve solo l'ultima posizione
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size) probabilità che sommano a 1
            idx_next = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    finally:
        model.train(was_training)  # rimettiamo il modello com'era
    return idx
