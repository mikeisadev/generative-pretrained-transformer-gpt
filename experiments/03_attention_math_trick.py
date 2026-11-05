"""Esperimento 3 — Il trucco matematico della self-attention (versioni 1-4).

    python experiments/03_attention_math_trick.py
"""

import torch
from _common import section

from gpt.didactic.attention_trick import (
    average_past_loop,
    average_past_matmul,
    average_past_softmax,
    causal_average_weights,
    self_attention_single_head,
)

section("Esempio giocattolo: a @ b fa la media cumulativa delle righe di b")
torch.manual_seed(42)
a = causal_average_weights(3)
b = torch.randint(0, 10, (3, 2)).float()
print("a =", a, "b =", b, "a @ b =", a @ b, sep="\n")

torch.manual_seed(1337)
x = torch.randn(4, 8, 2)  # (B, T, C)
v1, v2, v3 = average_past_loop(x), average_past_matmul(x), average_past_softmax(x)
section("Le versioni 1, 2 e 3 danno lo stesso risultato")
print("v1 == v2:", torch.allclose(v1, v2))
print("v1 == v3:", torch.allclose(v1, v3))

section("Versione 4: pesi che dipendono dai dati")
torch.manual_seed(1337)
x = torch.randn(4, 8, 32)
out, wei = self_attention_single_head(x, head_size=16)
print(f"out: {tuple(out.shape)}")
print("pesi di attention del primo esempio (righe = chi guarda, colonne = chi è guardato):")
torch.set_printoptions(precision=2, sci_mode=False)
print(wei[0])
