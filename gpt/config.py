"""Iperparametri del modello e del training.

Gli iperparametri sono le "manopole" che scegliamo NOI prima del training
(a differenza dei pesi, che il training impara da solo). Li raccogliamo in
dataclass immutabili invece che in variabili globali: così ogni modulo riceve
esplicitamente la configurazione e possiamo creare modelli diversi nello
stesso processo (per esempio uno piccolo nei test e uno standard nel training).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Architettura del transformer. I default sono quelli del notebook di Karpathy."""

    vocab_size: int = 65  # caratteri distinti in Tiny Shakespeare
    block_size: int = 32  # lunghezza massima del contesto (T massimo)
    n_embd: int = 64  # dimensione dei vettori di embedding (C)
    n_head: int = 4  # head di attention in parallelo in ogni blocco
    n_layer: int = 4  # blocchi transformer impilati
    dropout: float = 0.0  # probabilità di "spegnere" un'attivazione durante il training

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} deve essere positivo, ricevuto {getattr(self, name)}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) deve essere divisibile per n_head ({self.n_head}): "
                "ogni head lavora su una fetta di uguale dimensione."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout deve stare in [0, 1), ricevuto {self.dropout}")

    @property
    def head_size(self) -> int:
        """Dimensione di ogni head: le n_head fette, concatenate, tornano n_embd."""
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Parametri del ciclo di training (non fanno parte del modello salvato)."""

    batch_size: int = 16  # sequenze indipendenti elaborate in parallelo (B)
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 200  # batch usati per stimare la loss media
    learning_rate: float = 1e-3
    seed: int = 1337

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_iters", "eval_interval", "eval_iters"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} deve essere positivo, ricevuto {getattr(self, name)}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate deve essere positivo, ricevuto {self.learning_rate}")
