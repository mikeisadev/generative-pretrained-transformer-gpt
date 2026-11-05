"""Loss condivisa da tutti i language model del progetto (bigram e GPT)."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor | None:
    """Cross-entropy tra i logits previsti e il token che arriva davvero dopo.

    Args:
        logits: (B, T, V) un punteggio per ognuno dei V token del vocabolario.
        targets: (B, T) il token corretto per ogni posizione, oppure ``None`` in generazione.

    PyTorch vuole i logits come (N, V) e i target come (N,), quindi "appiattiamo"
    batch e tempo: B*T previsioni indipendenti.

    Nota didattica: se il modello non sa nulla e assegna a tutti i V token la
    stessa probabilità 1/V, la loss vale -ln(1/V) = ln(V). Con V=65 è ≈ 4.17:
    è il valore che ci aspettiamo all'inizio del training.
    """
    if targets is None:
        return None
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
