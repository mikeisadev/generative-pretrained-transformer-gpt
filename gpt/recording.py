"""Registrazione del training vero, per i grafici e le animazioni della web app.

``TrainingRecorder`` è un callback di ``gpt.training.train``: guarda il training
senza cambiarlo. Tutto ciò che richiede numeri casuali (il batch fisso di
validation, il testo d'esempio) usa ``torch.Generator`` DEDICATI, mai il
generatore globale: così i batch del training restano quelli di ``train.py`` e
le loss coincidono con il training di riferimento.

Convenzione sugli step: "step s" indica i pesi dopo ``s`` aggiornamenti
dell'ottimizzatore, gli stessi che vede la loss del passo ``s``. Lo snapshot
``0`` è quindi l'inizializzazione casuale e lo snapshot ``max_iters`` (sempre
presente) sono i pesi finali, identici al checkpoint (a meno del float16).

Schema di ``trace.json`` (float arrotondati; entropie in nat, cioè con ln)::

    {
      "format_version": 1,
      "config": {...},                 # GPTConfig.to_dict()
      "train_config": {...},           # TrainConfig: batch_size, max_iters, eval_interval, eval_iters,
                                       #              learning_rate, seed
      "total_params": 209729,
      "vocab": ["\\n", " ", ...],        # token id → carattere
      "device": "cpu",
      "duration_s": 170.3,             # durata del training registrato
      "recorder": {                    # come sono stati misurati i dati qui sotto
        "log_every", "snapshot_steps", "sample_prompt", "sample_length", "sample_seed",
        "eval_batch_size", "eval_batch_seed", "example_prompt"
      },
      "groups": ["token_embedding", "position_embedding", "blocks.0.attention", "blocks.0.ffwd",
                 "blocks.0.layernorm", ..., "ln_f", "lm_head"],
      "group_params": {"blocks.0.attention": ["blocks.0.sa.heads.0.key.weight", ...], ...},
      "snapshot_dtype": "float16",
      "byte_order": "little",
      "params": [{"name", "shape", "offset", "numel"}],   # layout dei file snap_*.f16.bin,
                                                         # offset/numel in numero di elementi
      "train_loss": [4.36, 4.31, ...], # loss del batch di training, UNA PER STEP (indice = step)
      "evals": {"steps": [0, 100, ..., 4999], "train": [...], "val": [...]},
                                       # le stime di estimate_loss (media su eval_iters batch)
      "metrics": {                     # ogni log_every step (e all'ultimo)
        "steps": [0, 25, ...],
        "lr": [...],                   # learning rate dell'ottimizzatore
        "grad_norm": [...],            # norma L2 del gradiente di tutti i pesi
        "grad_norm_by_group": {gruppo: [...]},
        "update_ratio": [...],         # ||Δw|| / ||w|| su tutti i pesi in quello step
        "update_ratio_by_group": {gruppo: [...]},
        "weight_std_by_group": {gruppo: [...]}   # deviazione standard (popolazione) dei pesi
      },
      "snapshots": [{
        "step": 500,
        "file": "snap_00500.f16.bin",  # relativo a trace.json
        "sample_text": "...",          # sample_length caratteri generati dopo sample_prompt
        "example_next_token_probs": [65 float],   # sul prompt d'esempio della web app
        "loss": 2.1,                   # cross-entropy sul batch fisso di validation
        "top1": 0.38, "top5": 0.74,    # accuracy del prossimo carattere sul batch fisso
        "entropy": 2.0,                # entropia media della distribuzione predetta
        "attention_entropy": [[...n_head] ...n_layer]   # entropia media di ogni head
      }]
    }

I file ``snap_XXXXX.f16.bin`` contengono tutti i parametri in float16
little-endian, concatenati nell'ordine di ``model.named_parameters()`` (lo
stesso layout di ``weights.f32.bin``, vedi ``gpt.export``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.config import TrainConfig
from gpt.data import CharTokenizer, get_batch
from gpt.export import EXAMPLE_PROMPT, capture_attention, parameter_layout, parameters_to_bytes
from gpt.generation import generate
from gpt.model import GPTLanguageModel
from gpt.training import EvalPoint, TrainingCallback

TRACE_FORMAT_VERSION = 1
TRACE_FILE = "trace.json"
SNAPSHOT_PATTERN = "snap_*.f16.bin"
# Più fitti all'inizio, dove i pesi cambiano di più. 5000 = pesi finali del training di riferimento.
DEFAULT_SNAPSHOT_STEPS = (0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000)

# Sotto-moduli di un Block → gruppo (attention = heads + proj; le due LayerNorm insieme).
_BLOCK_PARTS = {"sa": "attention", "ffwd": "ffwd", "ln1": "layernorm", "ln2": "layernorm"}
_TOP_LEVEL = {
    "token_embedding_table": "token_embedding",
    "position_embedding_table": "position_embedding",
    "ln_f": "ln_f",
    "lm_head": "lm_head",
}


def snapshot_file(step: int) -> str:
    return f"snap_{step:05d}.f16.bin"


def group_of(param_name: str) -> str:
    """Gruppo di un parametro, es. ``blocks.2.sa.heads.0.key.weight`` → ``blocks.2.attention``."""
    parts = param_name.split(".")
    if parts[0] in _TOP_LEVEL:
        return _TOP_LEVEL[parts[0]]
    if parts[0] == "blocks" and len(parts) > 2 and parts[2] in _BLOCK_PARTS:
        return f"blocks.{parts[1]}.{_BLOCK_PARTS[parts[2]]}"
    raise ValueError(f"il parametro '{param_name}' non appartiene a nessun gruppo: aggiorna group_of")


def parameter_groups(model: GPTLanguageModel) -> dict[str, list[str]]:
    """Gruppi nell'ordine del flusso dei dati → nomi dei loro parametri."""
    order = ["token_embedding", "position_embedding"]
    for i in range(model.config.n_layer):
        order += [f"blocks.{i}.attention", f"blocks.{i}.ffwd", f"blocks.{i}.layernorm"]
    order += ["ln_f", "lm_head"]
    groups: dict[str, list[str]] = {g: [] for g in order}
    for name, _ in model.named_parameters():
        groups[group_of(name)].append(name)
    empty = [g for g, names in groups.items() if not names]
    if empty:
        raise ValueError(f"gruppi senza parametri: {empty}")
    return groups


def _sig(x: float, digits: int = 6) -> float:
    """Arrotonda a ``digits`` cifre significative: le metriche vanno da 1e-5 a 1e+1."""
    return float(f"{x:.{digits}g}")


def _attention_entropy(weights: torch.Tensor) -> float:
    """Entropia media delle righe di attention (B, T, T): 0 = guarda un solo token, ln(t+1) = media uniforme.

    xlogy(0, 0) = 0: le posizioni future (peso 0 per la maschera causale) non contano.
    """
    return float(-torch.special.xlogy(weights, weights).sum(-1).mean())


@dataclass(frozen=True)
class RecorderConfig:
    log_every: int = 25  # ogni quanti step misurare gradienti, update e pesi
    snapshot_steps: tuple[int, ...] = DEFAULT_SNAPSHOT_STEPS
    sample_prompt: str = "\n"
    sample_length: int = 150
    sample_seed: int = 42  # stesso seme a ogni snapshot: cambia solo il modello, non il "dado"
    eval_batch_size: int = 64  # sequenze del batch fisso di validation
    eval_batch_seed: int = 2024

    def __post_init__(self) -> None:
        for name in ("log_every", "sample_length", "eval_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} deve essere positivo, ricevuto {getattr(self, name)}")
        if any(s < 0 for s in self.snapshot_steps):
            raise ValueError("gli step degli snapshot non possono essere negativi")


class TrainingRecorder(TrainingCallback):
    """Callback che registra loss, metriche e snapshot dei pesi; poi ``save(out_dir)``.

    Uso: ``train(..., callbacks=[recorder])``, poi ``recorder.save(cartella)``.
    Un recorder registra UN solo training: per un altro training creane uno nuovo.
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        val_data: torch.Tensor,
        block_size: int,
        config: RecorderConfig = RecorderConfig(),
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.block_size = block_size
        self.example_prompt = EXAMPLE_PROMPT[:block_size]
        # Il batch fisso viene estratto ORA, con un generatore suo: è lo stesso a ogni
        # snapshot (le metriche sono confrontabili) e non sposta il generatore globale.
        generator = torch.Generator().manual_seed(config.eval_batch_seed)
        self._eval_x, self._eval_y = get_batch(val_data, block_size, config.eval_batch_size, generator=generator)
        self._example_ids = torch.tensor([tokenizer.encode(self.example_prompt)], dtype=torch.long)
        self._sample_ids = torch.tensor([tokenizer.encode(config.sample_prompt)], dtype=torch.long)

        self.train_loss: list[float] = []
        self.evals: dict[str, list] = {"steps": [], "train": [], "val": []}
        self.metrics: dict[str, Any] = {}
        self.snapshots: list[dict[str, Any]] = []
        self._blobs: dict[str, bytes] = {}  # file → contenuto float16
        self._weights_before: dict[str, torch.Tensor] | None = None
        self._model_config: dict | None = None
        self._train_config: TrainConfig | None = None
        self._snapshot_steps: set[int] = set()
        self._groups: dict[str, list[str]] = {}
        self._layout: list[dict] = []
        self._device = "cpu"
        self._started_at = 0.0
        self.duration_s: float | None = None

    # --- hook del training -------------------------------------------------------------

    def on_train_start(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_cfg: TrainConfig) -> None:
        self._train_config = train_cfg
        self._model_config = model.config.to_dict()
        self._groups = parameter_groups(model)
        self._layout = parameter_layout(model)
        self._device = next(model.parameters()).device
        # Gli snapshot oltre la fine non esistono; quello finale c'è sempre.
        self._snapshot_steps = {s for s in self.config.snapshot_steps if s <= train_cfg.max_iters}
        self._snapshot_steps.add(train_cfg.max_iters)
        self.metrics = {
            "steps": [],
            "lr": [],
            "grad_norm": [],
            "grad_norm_by_group": {g: [] for g in self._groups},
            "update_ratio": [],
            "update_ratio_by_group": {g: [] for g in self._groups},
            "weight_std_by_group": {g: [] for g in self._groups},
        }
        self._started_at = time.perf_counter()

    def on_eval(self, step: int, losses: dict[str, float]) -> None:
        self.evals["steps"].append(step)
        self.evals["train"].append(round(losses["train"], 5))
        self.evals["val"].append(round(losses["val"], 5))

    def on_after_backward(self, step: int, model: nn.Module, loss: torch.Tensor) -> None:
        if step != len(self.train_loss):
            raise RuntimeError(f"step {step} ricevuto fuori ordine (atteso {len(self.train_loss)})")
        self.train_loss.append(round(loss.item(), 5))
        # Qui i pesi sono ancora quelli dopo `step` aggiornamenti: è il momento giusto per lo snapshot.
        if step in self._snapshot_steps:
            self._snapshot(step, model)
        if self._is_log_step(step):
            self._log_gradients_and_weights(step, model)

    def on_step_end(
        self, step: int, model: nn.Module, loss: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        if self._weights_before is None:
            return
        params = dict(model.named_parameters())
        delta_sq, weight_sq = {}, {}
        with torch.no_grad():
            for group, names in self._groups.items():
                delta_sq[group] = sum(float((params[n] - self._weights_before[n]).pow(2).sum()) for n in names)
                weight_sq[group] = sum(float(self._weights_before[n].pow(2).sum()) for n in names)
        for group in self._groups:
            ratio = math.sqrt(delta_sq[group]) / max(math.sqrt(weight_sq[group]), 1e-12)
            self.metrics["update_ratio_by_group"][group].append(_sig(ratio))
        total = math.sqrt(sum(delta_sq.values())) / max(math.sqrt(sum(weight_sq.values())), 1e-12)
        self.metrics["update_ratio"].append(_sig(total))
        self.metrics["lr"].append(_sig(optimizer.param_groups[0]["lr"]))
        self._weights_before = None

    def on_train_end(self, model: nn.Module, history: list[EvalPoint]) -> None:
        self.duration_s = round(time.perf_counter() - self._started_at, 1)
        final = self._train_config.max_iters
        if final in self._snapshot_steps:
            self._snapshot(final, model)

    # --- misure -----------------------------------------------------------------------

    def _is_log_step(self, step: int) -> bool:
        return step % self.config.log_every == 0 or step == self._train_config.max_iters - 1

    @torch.no_grad()
    def _log_gradients_and_weights(self, step: int, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        total_sq = 0.0
        for group, names in self._groups.items():
            grad_sq = sum(float(params[n].grad.pow(2).sum()) for n in names if params[n].grad is not None)
            total_sq += grad_sq
            self.metrics["grad_norm_by_group"][group].append(_sig(math.sqrt(grad_sq)))
            values = torch.cat([params[n].detach().flatten() for n in names])
            self.metrics["weight_std_by_group"][group].append(_sig(values.std(correction=0).item()))
        self.metrics["steps"].append(step)
        self.metrics["grad_norm"].append(_sig(math.sqrt(total_sq)))
        # Copia dei pesi PRIMA di optimizer.step(): serve a on_step_end per misurare l'update.
        self._weights_before = {n: p.detach().clone() for n, p in params.items()}

    @torch.no_grad()
    def _snapshot(self, step: int, model: GPTLanguageModel) -> None:
        was_training = model.training
        model.eval()
        try:
            entry = {"step": step, "file": snapshot_file(step)}
            entry["sample_text"] = self._sample(model)
            entry["example_next_token_probs"] = self._next_token_probs(model, self._example_ids)
            entry.update(self._eval_batch_metrics(model))
        finally:
            model.train(was_training)
        self._blobs[entry["file"]] = parameters_to_bytes(model, "<f2")
        self.snapshots.append(entry)

    def _sample(self, model: GPTLanguageModel) -> str:
        generator = torch.Generator(device=self._device).manual_seed(self.config.sample_seed)
        idx = self._sample_ids.to(self._device)
        out = generate(model, idx, self.config.sample_length, block_size=self.block_size, generator=generator)
        return self.tokenizer.decode(out[0, idx.shape[1] :].tolist())

    def _next_token_probs(self, model: GPTLanguageModel, ids: torch.Tensor) -> list[float]:
        logits, _ = model(ids.to(self._device))
        return [round(p, 5) for p in F.softmax(logits[0, -1], dim=-1).tolist()]

    def _eval_batch_metrics(self, model: GPTLanguageModel) -> dict[str, Any]:
        x, y = self._eval_x.to(self._device), self._eval_y.to(self._device)
        with capture_attention(model) as attention:
            logits, loss = model(x, y)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        top5 = logits.topk(5, dim=-1).indices  # (B, T, 5)
        cfg = model.config
        return {
            "loss": round(loss.item(), 5),
            "top1": round(float((logits.argmax(-1) == y).float().mean()), 5),
            "top5": round(float((top5 == y.unsqueeze(-1)).any(-1).float().mean()), 5),
            "entropy": round(entropy.item(), 5),
            "attention_entropy": [
                [round(_attention_entropy(attention[(layer, h)]), 5) for h in range(cfg.n_head)]
                for layer in range(cfg.n_layer)
            ],
        }

    # --- serializzazione ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        if self._train_config is None:
            raise RuntimeError("il recorder non ha ancora visto un training: passalo a train(callbacks=[...])")
        return {
            "format_version": TRACE_FORMAT_VERSION,
            "config": self._model_config,
            "train_config": asdict(self._train_config),
            "total_params": sum(p["numel"] for p in self._layout),
            "vocab": self.tokenizer.chars,
            "device": str(self._device),
            "duration_s": self.duration_s,
            "recorder": {
                **asdict(self.config),
                "snapshot_steps": sorted(self._snapshot_steps),
                "example_prompt": self.example_prompt,
            },
            "groups": list(self._groups),
            "group_params": self._groups,
            "snapshot_dtype": "float16",
            "byte_order": "little",
            "params": self._layout,
            "train_loss": self.train_loss,
            "evals": self.evals,
            "metrics": self.metrics,
            "snapshots": self.snapshots,
        }

    def save(self, out_dir: Path) -> Path:
        """Scrive ``trace.json`` e gli snapshot in ``out_dir``; restituisce il percorso di trace.json.

        Gli snapshot di registrazioni precedenti vengono rimossi: nella cartella
        restano solo i file citati da questo ``trace.json``.
        """
        trace = self.to_dict()
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob(SNAPSHOT_PATTERN):
            if stale.name not in self._blobs:
                stale.unlink()
        for name, blob in self._blobs.items():
            (out_dir / name).write_bytes(blob)
        path = out_dir / TRACE_FILE
        path.write_text(json.dumps(trace, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return path
