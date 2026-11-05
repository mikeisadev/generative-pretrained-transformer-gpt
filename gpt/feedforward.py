"""Step 6 del percorso: la rete feed-forward (MLP) applicata a ogni token.

Dopo che i token si sono "parlati" con l'attention, ognuno ELABORA da solo
ciò che ha raccolto. Lo strato centrale è 4 volte più largo (come nel paper):
più spazio per combinare le informazioni prima di tornare a n_embd.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gpt.config import GPTConfig


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),  # pesi (4C, C) + bias (4C)
            nn.ReLU(),  # non-linearità: senza, due Linear di fila equivalgono a una sola
            nn.Linear(hidden, config.n_embd),  # pesi (C, 4C) + bias (C)
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, C) → (B, T, 4C) → (B, T, C)
