"""Step 9 del percorso: training e salvataggio del modello.

Il ciclo di training ripete quattro passi:
    1. prendi un batch (x, y)
    2. forward: calcola la loss
    3. backward: calcola il gradiente di ogni peso
    4. step: l'ottimizzatore (AdamW) sposta ogni peso un po' nella direzione che abbassa la loss
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, DataSplits, get_batch
from gpt.model import GPTLanguageModel

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class EvalPoint:
    step: int
    train_loss: float
    val_loss: float


class TrainingCallback:
    """Osservatore del training: ridefinisci solo i metodi che ti servono.

    I callback GUARDANO il training (loss, gradienti, pesi) senza cambiarlo:
    non devono modificare i pesi né usare il generatore casuale globale di torch,
    altrimenti i batch estratti dopo sarebbero diversi.
    Va bene anche un oggetto qualsiasi con solo alcuni di questi metodi.
    """

    def on_train_start(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_cfg: TrainConfig) -> None:
        """Prima del primo step."""

    def on_eval(self, step: int, losses: dict[str, float]) -> None:
        """Dopo ``estimate_loss``: ``losses = {"train": ..., "val": ...}``."""

    def on_after_backward(self, step: int, model: nn.Module, loss: torch.Tensor) -> None:
        """I gradienti sono pronti, i pesi NON sono ancora stati aggiornati."""

    def on_step_end(
        self, step: int, model: nn.Module, loss: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        """Subito dopo ``optimizer.step()``: i pesi sono già aggiornati."""

    def on_train_end(self, model: nn.Module, history: list[EvalPoint]) -> None:
        """Dopo l'ultimo step."""


class _CallbackList:
    """Chiama lo stesso metodo su tutti i callback che lo definiscono, nell'ordine della lista."""

    def __init__(self, callbacks: Iterable[object] | None) -> None:
        self._callbacks = list(callbacks or ())

    def __call__(self, hook: str, *args: object) -> None:
        for callback in self._callbacks:
            method = getattr(callback, hook, None)
            if method is not None:
                method(*args)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    splits: DataSplits,
    block_size: int,
    train_cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Media della loss su ``eval_iters`` batch: una singola misura sarebbe troppo rumorosa."""
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for split_name, data in (("train", splits.train), ("val", splits.val)):
        losses = torch.zeros(train_cfg.eval_iters)
        for k in range(train_cfg.eval_iters):
            x, y = get_batch(data, block_size, train_cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train(was_training)
    return out


def train(
    model: nn.Module,
    splits: DataSplits,
    block_size: int,
    train_cfg: TrainConfig,
    device: torch.device,
    log: Callable[[str], None] = print,
    callbacks: Iterable[TrainingCallback] | None = None,
) -> list[EvalPoint]:
    """Addestra ``model`` in place e restituisce la storia delle loss misurate.

    ``callbacks`` (facoltativi) osservano il training, per esempio per registrarlo
    (vedi ``gpt.recording``); senza callback il training è identico.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    history: list[EvalPoint] = []
    notify = _CallbackList(callbacks)
    model.train()
    notify("on_train_start", model, optimizer, train_cfg)

    for step in range(train_cfg.max_iters):
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_iters - 1:
            losses = estimate_loss(model, splits, block_size, train_cfg, device)
            history.append(EvalPoint(step, losses["train"], losses["val"]))
            log(f"step {step:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            notify("on_eval", step, losses)

        xb, yb = get_batch(splits.train, block_size, train_cfg.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        notify("on_after_backward", step, model, loss)
        optimizer.step()
        notify("on_step_end", step, model, loss, optimizer)

    notify("on_train_end", model, history)
    return history


def save_checkpoint(path: Path, model: GPTLanguageModel, tokenizer: CharTokenizer) -> None:
    """Salva pesi, configurazione e vocabolario: tutto ciò che serve per rigenerare testo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config": model.config.to_dict(),
            "chars": tokenizer.chars,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[GPTLanguageModel, CharTokenizer]:
    """Ricarica un checkpoint in modo sicuro.

    ``weights_only=True`` impedisce a torch.load di eseguire codice arbitrario
    contenuto nel file (pickle): un checkpoint scaricato da internet è input non fidato.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"formato checkpoint non supportato: {payload.get('format_version')}")
    model = GPTLanguageModel(GPTConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, CharTokenizer(payload["chars"])
