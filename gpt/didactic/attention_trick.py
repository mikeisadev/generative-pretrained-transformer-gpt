"""Step 3 del percorso: il "trucco matematico" della self-attention.

Obiettivo: ogni token t riceve la MEDIA dei token 0..t (solo passato, mai futuro).
Lo otteniamo in quattro versioni sempre più vicine all'attention vera:

    v1  doppio ciclo for                         → chiaro ma lento
    v2  moltiplicazione per matrice triangolare  → stesso risultato, in un colpo solo
    v3  softmax su una matrice con -inf          → pesi "ammorbiditi", pronti a diventare appresi
    v4  self-attention: i pesi dipendono dai dati (query · key)

Tutte le funzioni lavorano su x di forma (B, T, C).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


def average_past_loop(x: torch.Tensor) -> torch.Tensor:
    """v1: xbow[b, t] = media di x[b, 0..t]. ("bow" = bag of words)"""
    B, T, C = x.shape
    xbow = torch.zeros_like(x)
    for b in range(B):
        for t in range(T):
            xbow[b, t] = x[b, : t + 1].mean(dim=0)  # (t+1, C) → (C,)
    return xbow


def causal_average_weights(T: int) -> torch.Tensor:
    """Matrice (T, T) triangolare inferiore con righe normalizzate: riga t = 1/(t+1) sulle prime t+1 colonne."""
    wei = torch.tril(torch.ones(T, T))
    return wei / wei.sum(dim=1, keepdim=True)


def average_past_matmul(x: torch.Tensor) -> torch.Tensor:
    """v2: la stessa media come prodotto matriciale. (T, T) @ (B, T, C) → (B, T, C)"""
    return causal_average_weights(x.shape[1]) @ x


def causal_softmax_weights(scores: torch.Tensor) -> torch.Tensor:
    """Maschera il futuro con -inf e normalizza con softmax. scores: (..., T, T)"""
    T = scores.shape[-1]
    tril = torch.tril(torch.ones(T, T, device=scores.device))
    return F.softmax(scores.masked_fill(tril == 0, float("-inf")), dim=-1)


def average_past_softmax(x: torch.Tensor) -> torch.Tensor:
    """v3: punteggi tutti a 0 + maschera + softmax = di nuovo la media uniforme."""
    T = x.shape[1]
    return causal_softmax_weights(torch.zeros(T, T)) @ x


def self_attention_single_head(x: torch.Tensor, head_size: int, seed: int = 1337) -> tuple[torch.Tensor, torch.Tensor]:
    """v4: i punteggi non sono più zeri ma affinità query·key, apprendibili.

    Restituisce (out, wei): out (B, T, head_size), wei (B, T, T).
    """
    torch.manual_seed(seed)
    C = x.shape[-1]
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k, q, v = key(x), query(x), value(x)  # (B, T, hs) ciascuno
    scores = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T)
    wei = causal_softmax_weights(scores)
    return wei @ v, wei
