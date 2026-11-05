"""Esperimento 2 — Il modello bigram: prima e dopo un breve training.

    python experiments/02_bigram.py
"""

import math

import torch
from _common import section

from gpt.bigram import BigramLanguageModel
from gpt.data import CharTokenizer, get_batch, load_text, train_val_split

torch.manual_seed(1337)
text = load_text()
tokenizer = CharTokenizer.from_text(text)
splits = train_val_split(torch.tensor(tokenizer.encode(text), dtype=torch.long))

model = BigramLanguageModel(tokenizer.vocab_size)
xb, yb = get_batch(splits.train, block_size=8, batch_size=4)
logits, loss = model(xb, yb)
section("Prima del training")
print(f"logits: {tuple(logits.shape)}  (B, T, V)")
print(f"loss: {loss.item():.4f}  — atteso circa ln({tokenizer.vocab_size}) = {math.log(tokenizer.vocab_size):.4f}")
start = torch.zeros((1, 1), dtype=torch.long)
print(tokenizer.decode(model.generate(start, max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(10_000):
    xb, yb = get_batch(splits.train, block_size=8, batch_size=32)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

section("Dopo 10.000 step")
print(f"loss: {loss.item():.4f}")
print(tokenizer.decode(model.generate(start, max_new_tokens=300)[0].tolist()))
