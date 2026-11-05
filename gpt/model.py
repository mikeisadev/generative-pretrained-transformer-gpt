"""Step 8 del percorso: il GPT completo (decoder-only transformer).

Flusso dei dati:

    idx (B, T) token id
      │  token embedding  +  position embedding
      ▼
    x (B, T, C)
      │  n_layer × Block  (attention + feed-forward, con residui e LayerNorm)
      ▼
    x (B, T, C)
      │  LayerNorm finale
      │  lm_head (Linear C → V)
      ▼
    logits (B, T, V)  →  softmax  →  probabilità del prossimo token

Nel notebook originale questa classe si chiamava ancora ``BigramLanguageModel``
(residuo del primo esperimento): qui ha il nome che merita.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gpt.block import Block
from gpt.config import GPTConfig
from gpt.generation import generate
from gpt.loss import next_token_loss


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        # "Che cosa" è il token: una riga di C numeri appresi per ognuno dei V token.
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)  # (V, C)
        # "Dove" si trova: l'attention da sola non conosce l'ordine, glielo diciamo noi.
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)  # (T_max, C)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # LayerNorm finale
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # (V, C) + bias (V)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequenza lunga {T}, il massimo è block_size={self.config.block_size}")

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # Le posizioni 0..T-1 vengono create sullo STESSO device dell'input
        # (l'originale usava una variabile globale `device`).
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding_table(pos)  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) broadcasting: la stessa posizione vale per tutto il batch
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)
        return logits, next_token_loss(logits, targets)

    def generate(self, idx: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        return generate(self, idx, max_new_tokens, block_size=self.config.block_size, **kwargs)

    def num_parameters(self) -> int:
        """Numero di parametri appresi (i buffer come la maschera tril non contano)."""
        return sum(p.numel() for p in self.parameters())
