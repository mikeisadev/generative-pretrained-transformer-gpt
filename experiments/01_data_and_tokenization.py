"""Esperimento 1 — Dati e tokenizzazione.

    python experiments/01_data_and_tokenization.py
"""

import torch
from _common import section

from gpt.data import CharTokenizer, get_batch, load_text, train_val_split

text = load_text()
section("Il dataset")
print(f"caratteri totali: {len(text):,}")
print(text[:300])

tokenizer = CharTokenizer.from_text(text)
section("Il vocabolario")
print(repr("".join(tokenizer.chars)))
print(f"vocab_size = {tokenizer.vocab_size}")

section("Encode / decode")
ids = tokenizer.encode("hii there")
print(f"encode('hii there') = {ids}")
print(f"decode(...)         = {tokenizer.decode(ids)!r}")

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
splits = train_val_split(data)
print(f"\ntrain: {len(splits.train):,} token | val: {len(splits.val):,} token")

section("Una finestra contiene block_size esempi")
block_size = 8
x, y = splits.train[:block_size], splits.train[1 : block_size + 1]
for t in range(block_size):
    print(f"contesto {x[: t + 1].tolist()} → target {y[t].item()}")

section("Un batch (B=4, T=8)")
torch.manual_seed(1337)
xb, yb = get_batch(splits.train, block_size=8, batch_size=4)
print("x =", xb, sep="\n")
print("y =", yb, sep="\n")
