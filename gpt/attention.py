"""Step 4 e 5 del percorso: singola head di self-attention e multi-head attention.

L'attention è un meccanismo di COMUNICAZIONE tra token. Ogni token produce tre vettori:
    - query (Q): "che cosa sto cercando?"
    - key   (K): "che cosa contengo?"
    - value (V): "che cosa ti comunico se mi scegli?"
Il prodotto scalare query·key misura quanto due token sono "affini"; la softmax
trasforma le affinità in pesi che sommano a 1; l'output è la media pesata dei value.

Formula (Vaswani et al., 2017):  Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.config import GPTConfig


class Head(nn.Module):
    """Una head di self-attention causale (ogni token vede solo sé stesso e il passato)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.head_size = config.head_size
        # Tre proiezioni lineari apprese, senza bias: pesi (head_size, n_embd) ciascuna.
        self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, self.head_size, bias=False)
        # Maschera causale triangolare inferiore (block_size, block_size).
        # È un "buffer": si sposta su GPU col modello ma NON viene appresa.
        # persistent=False: non finisce nel checkpoint, la ricostruiamo sempre uguale.
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size)), persistent=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Quanto ogni token "guarda" ogni token precedente: (B, T, T), ogni riga somma a 1."""
        B, T, C = x.shape  # C = n_embd
        k = self.key(x)  # (B, T, hs)  hs = head_size
        q = self.query(x)  # (B, T, hs)

        # Affinità tra ogni coppia di token. Dividiamo per √head_size (la d_k del paper):
        # così, se q e k hanno varianza 1, anche i punteggi hanno varianza ≈1 e la
        # softmax non "satura" su un solo token.
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, hs) @ (B, hs, T) → (B, T, T)

        # Il futuro non si guarda: dove la maschera vale 0 mettiamo -inf,
        # che dopo la softmax diventa probabilità 0.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        return F.softmax(wei, dim=-1)  # (B, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wei = self.dropout(self.attention_weights(x))  # (B, T, T)
        v = self.value(x)  # (B, T, hs)
        return wei @ v  # media pesata dei value: (B, T, T) @ (B, T, hs) → (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Più head in parallelo, ognuna libera di cercare un tipo diverso di relazione.

    Analogia: in una riunione ogni collega ascolta l'intervento con un "filtro"
    diverso (chi i numeri, chi le scadenze, chi i rischi). Alla fine si
    mettono insieme gli appunti di tutti.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        # Proiezione finale: rimescola gli appunti delle head. Pesi (n_embd, n_embd) + bias.
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # n_head uscite (B, T, hs) concatenate sull'ultima dimensione → (B, T, n_head*hs) = (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))  # (B, T, C)
