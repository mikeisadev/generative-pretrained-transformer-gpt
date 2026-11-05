"""Esperimento 5 — LayerNorm a mano vs nn.LayerNorm.

    python experiments/05_layernorm.py
"""

import torch
from _common import section

from gpt.didactic.layernorm import LayerNorm1d

torch.manual_seed(1337)
x = torch.randn(32, 100) * 3 + 5  # 32 esempi da 100 feature, media 5 e deviazione 3
y = LayerNorm1d(100)(x)

section("Ogni RIGA (un esempio) viene normalizzata")
print(f"riga 0:    media {y[0].mean():+.4f}, std {y[0].std(unbiased=False):.4f}")
print(f"colonna 0: media {y[:, 0].mean():+.4f}, std {y[:, 0].std():.4f}  (le colonne NO)")

section("Coincide con nn.LayerNorm")
print("allclose:", torch.allclose(y, torch.nn.LayerNorm(100)(x), atol=1e-5))
