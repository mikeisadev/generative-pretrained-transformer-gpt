"""Step 7 del percorso: il blocco transformer = comunicazione + elaborazione.

Due ingredienti rendono addestrabile una pila profonda di blocchi:

- Connessioni residue (``x = x + ...``): ogni sotto-strato aggiunge una
  "correzione" al flusso invece di riscriverlo. Il gradiente ha così
  un'autostrada diretta fino ai primi strati.
- LayerNorm: riporta ogni vettore a media 0 e varianza 1 (poi γ e β appresi
  riscalano e spostano). Qui è applicata PRIMA di ogni sotto-strato
  ("pre-norm"), come nei GPT moderni; nel paper del 2017 era dopo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gpt.attention import MultiHeadAttention
from gpt.config import GPTConfig
from gpt.feedforward import FeedForward


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)  # γ, β: (C,) ciascuno
        self.sa = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # comunicazione tra token   (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # elaborazione per token     (B, T, C)
        return x
