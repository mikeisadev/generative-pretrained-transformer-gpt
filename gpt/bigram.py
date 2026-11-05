"""Step 2 del percorso: il modello bigram, la baseline più semplice.

Ogni token guarda SOLO se stesso e legge da una tabella i punteggi del token
successivo. Esempio: dopo "q" in inglese arriva quasi sempre "u". Non c'è
nessuna comunicazione tra i token: è proprio il limite che la self-attention
risolverà.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gpt.generation import generate
from gpt.loss import next_token_loss


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # Tabella (V, V): la riga i contiene i logits del token che segue il token i.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)  # (B, T) → (B, T, V)
        return logits, next_token_loss(logits, targets)

    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        # Il bigram guarda solo l'ultimo token: non serve tagliare il contesto.
        return generate(self, idx, max_new_tokens, block_size=None, **kwargs)
