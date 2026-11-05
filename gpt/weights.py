"""Inventario dei pesi: dove sono, che forma hanno, quanti sono.

I pesi (o parametri) sono i numeri che il training impara. Tutto il resto
(la maschera ``tril``, il dropout, gli iperparametri) NON viene appreso.
Questo modulo alimenta sia la lezione sia la web app 3D.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.data import CharTokenizer


@dataclass(frozen=True)
class WeightInfo:
    name: str  # es. "blocks.0.sa.heads.1.query.weight"
    shape: tuple[int, ...]
    numel: int
    learned: bool  # True = parametro appreso, False = buffer fisso

    def to_dict(self) -> dict:
        # Liste invece di tuple: il formato che si ottiene rileggendo il JSON.
        return {**asdict(self), "shape": list(self.shape)}


def weight_inventory(model: nn.Module) -> list[WeightInfo]:
    """Elenca parametri appresi e buffer non appresi, nell'ordine del flusso dei dati."""
    learned = [
        WeightInfo(name, tuple(p.shape), p.numel(), learned=True) for name, p in model.named_parameters()
    ]
    fixed = [WeightInfo(name, tuple(b.shape), b.numel(), learned=False) for name, b in model.named_buffers()]
    return learned + fixed


def count_by_prefix(model: nn.Module, prefix: str) -> int:
    """Parametri appresi il cui nome inizia con ``prefix`` (es. "blocks.0.sa")."""
    return sum(p.numel() for name, p in model.named_parameters() if name.startswith(prefix))


@contextmanager
def edited_weight(model: nn.Module, name: str, index: tuple[int, ...], value: float) -> Iterator[float]:
    """Cambia UN singolo numero di un peso, poi lo ripristina all'uscita dal ``with``.

    Esempio:
        with edited_weight(model, "lm_head.bias", (stoi["z"],), 8.0) as old:
            ...  # qui il modello "ama" la lettera z
        # qui il modello è tornato identico a prima

    Restituisce (via ``as``) il valore originale.
    """
    param = dict(model.named_parameters())[name]
    with torch.no_grad():
        original = param[index].item()
        param[index] = value
    try:
        yield original
    finally:
        with torch.no_grad():
            param[index] = original


@torch.no_grad()
def next_token_distribution(
    model: nn.Module, tokenizer: CharTokenizer, prompt: str, top_k: int = 5
) -> list[tuple[str, float]]:
    """I ``top_k`` caratteri più probabili dopo ``prompt``, con la loro probabilità."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    block_size = getattr(getattr(model, "config", None), "block_size", None)
    if block_size is not None:
        idx = idx[:, -block_size:]
    logits, _ = model(idx)
    probs = F.softmax(logits[0, -1], dim=-1)
    values, indices = probs.topk(top_k)
    model.train(was_training)
    return [(tokenizer.decode([i]), v) for i, v in zip(indices.tolist(), values.tolist(), strict=True)]
