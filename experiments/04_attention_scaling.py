"""Esperimento 4 — Perché dividere per √head_size.

    python experiments/04_attention_scaling.py
"""

import torch
from _common import section

torch.manual_seed(1337)
B, T, head_size = 4, 8, 16
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)

section("Varianza dei punteggi query·key")
print(f"var(k) = {k.var():.3f}, var(q) = {q.var():.3f}")
print(f"senza scaling:  var(q·k) = {(q @ k.transpose(-2, -1)).var():.3f}  (≈ head_size = {head_size})")
print(f"con scaling:    var(q·k/√hs) = {(q @ k.transpose(-2, -1) * head_size**-0.5).var():.3f}  (≈ 1)")

section("Softmax di valori grandi → quasi one-hot")
scores = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
print("softmax(s)     =", torch.softmax(scores, dim=-1))
print("softmax(s * 8) =", torch.softmax(scores * 8, dim=-1))
