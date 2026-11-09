"""Registrazione del training vero, per i grafici e le animazioni della web app.

``TrainingRecorder`` è un callback di ``gpt.training.train``: guarda il training
senza cambiarlo. Tutto ciò che richiede numeri casuali (il batch fisso di
validation, il testo d'esempio) usa ``torch.Generator`` DEDICATI, mai il
generatore globale: così i batch del training restano quelli di ``train.py`` e
le loss coincidono con il training di riferimento.

Funziona con QUALUNQUE architettura costruita con il codice di ``gpt/``, anche
modificata (un'attivazione, una norma, un'attention diversa, moduli nuovi): basta
l'interfaccia di ``train``, cioè ``logits, loss = model(idx, targets)``. Le due cose
che dipendono dai nomi del GPT del corso si adattano, e il trace lo dice in ``notes``:

- i GRUPPI di parametri (``group_of``): per i nomi del corso sono quelli di sempre
  (li usano la web app e i dati committati); un nome nuovo va nel suo modulo, con la
  regola di ``module_group``, la stessa di ``groupOf`` nel browser
  (``webapp/js/shared/weights/groups.js``);
- l'ENTROPIA DELL'ATTENTION viene dai moduli che espongono ``attention_weights``,
  ovunque siano (interfaccia in ``gpt.export.capture_attention``); se il modello non
  ne ha, la metrica si salta. Tutte le altre metriche valgono per ogni architettura.

Convenzione sugli step: "step s" indica i pesi dopo ``s`` aggiornamenti
dell'ottimizzatore, gli stessi che vede la loss del passo ``s``. Lo snapshot
``0`` è quindi l'inizializzazione casuale e lo snapshot ``max_iters`` (sempre
presente) sono i pesi finali, identici al checkpoint (a meno del float16).

Schema di ``trace.json`` (float arrotondati; entropie in nat, cioè con ln)::

    {
      "format_version": 1,
      "config": {...},                 # GPTConfig.to_dict(), anche con i campi aggiunti (null senza config)
      "train_config": {...},           # TrainConfig: batch_size, max_iters, eval_interval, eval_iters,
                                       #              learning_rate, seed
      "total_params": 209729,          # valori di uno snapshot (per il GPT del corso: i parametri)
      "vocab": ["\\n", " ", ...],        # token id → carattere
      "device": "cpu",
      "duration_s": 170.3,             # durata del training registrato
      "recorder": {                    # come sono stati misurati i dati qui sotto
        "log_every", "snapshot_steps", "sample_prompt", "sample_length", "sample_seed",
        "eval_batch_size", "eval_batch_seed", "example_prompt"
      },
      "groups": ["token_embedding", "position_embedding", "blocks.0.attention", "blocks.0.ffwd",
                 "blocks.0.layernorm", ..., "ln_f", "lm_head"],   # ordine del flusso dei dati
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
        "attention_entropy": [[...head] ...strato]   # entropia media di ogni head; SOLO se il
      }],                                            # modello espone attention_weights
      "notes": ["..."]                 # SOLO se c'è qualcosa da dire (mai per il GPT del corso):
    }                                  # nomi nuovi, attention saltata, valori non finiti, ...

I file ``snap_XXXXX.f16.bin`` contengono i tensori in virgola mobile dello
``state_dict`` in float16 little-endian, concatenati nel suo ordine: i parametri
e gli eventuali buffer persistenti, cioè gli stessi tensori che ``scripts/export_run.py``
esporta dal checkpoint. Per il GPT del corso sono esattamente ``model.named_parameters()``
(la maschera ``tril`` non è persistente), lo stesso layout di ``weights.f32.bin``
(vedi ``gpt.export``). Un valore non finito (NaN, infinito: training divergente)
diventa ``null`` nel JSON, che resta così leggibile dal browser.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.config import TrainConfig
from gpt.data import CharTokenizer, get_batch
from gpt.export import (
    EXAMPLE_PROMPT,
    attention_sites,
    capture_attention,
    parameter_layout,
    parameters_to_bytes,
)
from gpt.generation import generate
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
# Un indice di nn.Sequential o nn.ModuleList: solo cifre ASCII, come /^\d+$/ in groups.js.
_INDEX = re.compile(r"[0-9]+")
_LAYERNORM_GROUP = re.compile(r"blocks\.([^.]+)\.layernorm")
# Le head del GPT del corso: con questi nomi l'entropia dell'attention non ha bisogno di spiegazioni.
_COURSE_HEAD = re.compile(r"blocks\.[0-9]+\.sa\.heads\.[0-9]+")
TOP_K = 5


def snapshot_file(step: int) -> str:
    return f"snap_{step:05d}.f16.bin"


# --------------------------------------------------------------------------- gruppi di parametri


def course_group(param_name: str) -> str | None:
    """Gruppo di un parametro del GPT del corso, o None se il nome non è del corso.

    Esempio: ``blocks.2.sa.heads.0.key.weight`` → ``blocks.2.attention``.
    """
    parts = param_name.split(".")
    if parts[0] in _TOP_LEVEL:
        return _TOP_LEVEL[parts[0]]
    if parts[0] == "blocks" and len(parts) > 2 and parts[2] in _BLOCK_PARTS:
        return f"blocks.{parts[1]}.{_BLOCK_PARTS[parts[2]]}"
    return None


def module_group(param_name: str) -> str:
    """La regola generale, per i nomi che il GPT del corso non ha: il modulo che contiene il parametro.

    Dentro un blocco ripetuto (il primo livello numerato del nome) è il sotto-modulo del blocco,
    altrimenti i primi due livelli del nome; un parametro senza modulo è un gruppo da solo:

        blocks.0.norm1.weight             → blocks.0.norm1
        transformer.h.3.attn.c_attn.bias  → transformer.h.3.attn
        transformer.wte.weight            → transformer.wte
        scale                             → scale

    È la regola di ``groupOf`` nel browser (js/shared/weights/groups.js): Python e web app
    raggruppano allo stesso modo anche i run che non hanno un trace.
    """
    modules = param_name.split(".")[:-1]
    if not modules:
        return param_name
    index = next((i for i, part in enumerate(modules) if _INDEX.fullmatch(part)), -1)
    return ".".join(modules[: index + 2 if index > 0 else 2])


def group_of(param_name: str) -> str:
    """Gruppo di un parametro: quello del GPT del corso se il nome è del corso, altrimenti il suo modulo.

    ``blocks.2.sa.heads.0.key.weight`` → ``blocks.2.attention``; ``blocks.2.norm1.weight`` →
    ``blocks.2.norm1`` (vedi ``module_group``). Non fallisce mai: un'architettura con moduli dai
    nomi nuovi si registra lo stesso.
    """
    return course_group(param_name) or module_group(param_name)


def parameter_groups(model: nn.Module) -> dict[str, list[str]]:
    """Gruppi nell'ordine del flusso dei dati → nomi dei loro parametri.

    L'ordine è quello in cui i gruppi compaiono in ``named_parameters``, con una sola eccezione
    presa dal GPT del corso: il gruppo ``layernorm`` di un blocco (ln1 e ln2 insieme, che stanno
    prima dell'attention e prima del feed-forward) va in fondo al suo blocco, come nei dati
    committati: token_embedding, position_embedding, blocks.0.attention, blocks.0.ffwd,
    blocks.0.layernorm, …, ln_f, lm_head.
    """
    groups: dict[str, list[str]] = {}
    for name, _ in model.named_parameters():
        groups.setdefault(group_of(name), []).append(name)
    return {group: groups[group] for group in _flow_order(list(groups))}


def _flow_order(groups: list[str]) -> list[str]:
    """``groups`` (in ordine di comparsa) con ``blocks.<i>.layernorm`` dopo gli altri gruppi del blocco."""

    def position(item: tuple[int, str]) -> float:
        index, group = item
        match = _LAYERNORM_GROUP.fullmatch(group)
        if match is None:
            return index
        block = f"blocks.{match.group(1)}."
        mates = [i for i, other in enumerate(groups) if other != group and other.startswith(block)]
        return max(mates) + 0.5 if mates else index

    return [group for _, group in sorted(enumerate(groups), key=position)]


def grouping_note(names: Iterable[str]) -> str | None:
    """La nota del trace per i parametri con nomi che il GPT del corso non ha (None se non ce ne sono)."""
    new = [name for name in names if course_group(name) is None]
    if not new:
        return None
    groups = list(dict.fromkeys(module_group(name) for name in new))
    return (
        f"gruppi: {_count(len(new), 'parametro', 'parametri')} con nomi che il GPT del corso non ha "
        f"({_few(new)}) sono raggruppati per modulo con la regola generale di group_of (gpt/recording.py, "
        "come groupOf nel browser): il sotto-modulo del blocco numerato, altrimenti i primi due livelli "
        f"del nome. Gruppi nuovi: {_few(groups)}; i parametri di ogni gruppo sono in group_params."
    )


# --------------------------------------------------------------------------- note sull'attention


def attention_notes(
    sites: list[tuple[str, str]], rows: list[list[float]] | None, problems: Iterable[str] = ()
) -> list[str]:
    """Le note del trace su ``attention_entropy`` (nessuna per il GPT del corso).

    Args:
        sites: ``(percorso del modulo, percorso dello strato)`` dei moduli con ``attention_weights``.
        rows: ``attention_entropy`` dell'ultimo snapshot che ce l'ha (strati × head), o None.
        problems: moduli saltati, con il motivo (``capture_attention(..., problems=...)``).
    """
    if not sites:
        return [(
            "attention: nessun modulo del modello espone il metodo attention_weights (l'interfaccia di "
            "Head in gpt/attention.py, descritta in capture_attention di gpt/export.py), quindi "
            "l'entropia dell'attention per head non è stata misurata: le altre metriche ci sono tutte. "
            "Per averla aggiungi all'attention un metodo attention_weights con gli stessi argomenti di "
            "forward che restituisce le probabilità (B, T, T) di una head o (B, H, T, T) di H head."
        )]
    notes = [f"attention: {problem} (modulo saltato nell'entropia dell'attention)" for problem in problems]
    if not rows or not any(rows):
        notes.append("attention: i moduli con attention_weights non hanno dato probabilità valide nel "
                     "forward (errori qui sopra, o moduli mai chiamati): l'entropia dell'attention non c'è")
        return notes
    if _course_attention(sites):
        return notes
    layers = list(dict.fromkeys(layer or "il modello intero" for _, layer in sites))
    patterns = list(dict.fromkeys(_pattern(path) for path, _ in sites))
    heads = sorted({len(row) for row in rows})
    per_layer = str(heads[0]) if len(heads) == 1 else f"da {heads[0]} a {heads[-1]}"
    notes.append(
        f"attention: attention_entropy[strato][head] viene dai moduli con attention_weights "
        f"{_few(patterns)}: {_count(len(rows), 'strato', 'strati')} ({_span(layers)}) con {per_layer} head "
        "ciascuno, numerate nell'ordine del forward"
    )
    return notes


def _course_attention(sites: list[tuple[str, str]]) -> bool:
    """Le head stanno dove le mette il GPT del corso (blocks.<i>.sa.heads.<h>, strato i = blocks.i)?"""
    layers = list(dict.fromkeys(layer for _, layer in sites))
    return all(_COURSE_HEAD.fullmatch(path) for path, _ in sites) and layers == [
        f"blocks.{i}" for i in range(len(layers))
    ]


def _attention_entropy(weights: torch.Tensor) -> float:
    """Entropia media delle righe di attention (B, T, T): 0 = guarda un solo token, ln(t+1) = media uniforme.

    xlogy(0, 0) = 0: le posizioni future (peso 0 per la maschera causale) non contano.
    """
    return float(-torch.special.xlogy(weights, weights).sum(-1).mean())


def _entropy_rows(attention: Mapping[tuple[int, int], torch.Tensor]) -> list[list[float]]:
    """``{(strato, head): pesi}`` → entropia media di ogni head, una riga per strato."""
    rows: list[list[float]] = [[] for _ in range(max(layer for layer, _ in attention) + 1)]
    for (layer, _head), weights in sorted(attention.items(), key=lambda item: item[0]):
        rows[layer].append(round(_attention_entropy(weights), 5))
    return rows


# --------------------------------------------------------------------------- snapshot e JSON


class _Tensors:
    """Tensori con nome e l'interfaccia che usano ``parameter_layout`` e ``parameters_to_bytes``.

    Così layout e byte degli snapshot li scrive il codice di gpt/export.py, lo stesso di weights.f32.bin.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def named_parameters(self):
        return iter(self._tensors.items())

    def parameters(self):
        return iter(self._tensors.values())


def snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """I tensori di uno snapshot: quelli in virgola mobile dello ``state_dict``, nel suo ordine.

    Sono i pesi che ``scripts/export_run.py`` esporta dal checkpoint (i parametri e gli eventuali
    buffer persistenti, es. le statistiche di una BatchNorm): così il Laboratorio confronta ogni
    snapshot con i pesi del run e il benchmark li misura tutti. Per il GPT del corso coincidono con
    ``named_parameters`` (la maschera ``tril`` non è persistente).
    """
    return {name: t for name, t in model.state_dict().items() if t.is_floating_point()}


def _model_config(model: nn.Module) -> tuple[dict[str, Any] | None, list[str]]:
    """La config del modello scrivibile in JSON (anche i campi aggiunti a GPTConfig) e le note."""
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
    elif dataclasses.is_dataclass(config) and not isinstance(config, type):
        raw = dataclasses.asdict(config)
    else:
        return None, ["config: il modello non ha un attributo config con to_dict(): il trace non la contiene"]
    clean = {key: value for key, value in raw.items() if isinstance(key, str) and _json_value(value)}
    dropped = [str(key) for key in raw if key not in clean]
    notes = [f"config: saltati i campi non scrivibili in JSON ({', '.join(dropped)})"] if dropped else []
    return clean, notes


def _json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_json_value(v) for v in value)
    return False


def _finite_json(value: Any) -> tuple[Any, int]:
    """``value`` con ogni float non finito (NaN, ±infinito) al posto di None, e quanti ne ha sostituiti.

    In JSON NaN non esiste: ``JSON.parse`` del browser rifiuterebbe il file intero. ``null`` invece
    si legge, e i grafici della web app saltano i punti che non sono numeri.
    """
    if isinstance(value, float):
        return (value, 0) if math.isfinite(value) else (None, 1)
    if isinstance(value, dict):
        out, total = {}, 0
        for key, item in value.items():
            out[key], count = _finite_json(item)
            total += count
        return out, total
    if isinstance(value, (list, tuple)):
        items = [_finite_json(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    return value, 0


def _few(items: list[str], limit: int = 3) -> str:
    """I primi nomi di un elenco: "a, b, c e altri 4" (con uno solo in più: "a, b, c, d")."""
    if len(items) <= limit + 1:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} e altri {len(items) - limit}"


def _span(items: list[str]) -> str:
    """Un elenco ordinato in breve: "blocks.0 … blocks.5"."""
    return ", ".join(items) if len(items) <= 3 else f"{items[0]} … {items[-1]}"


def _pattern(path: str) -> str:
    """Il percorso con gli indici al posto di "*": blocks.3.attn → blocks.*.attn (come la web app)."""
    return ".".join("*" if _INDEX.fullmatch(part) else part for part in path.split("."))


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _sig(x: float, digits: int = 6) -> float:
    """Arrotonda a ``digits`` cifre significative: le metriche vanno da 1e-5 a 1e+1."""
    return float(f"{x:.{digits}g}")


# --------------------------------------------------------------------------- il recorder


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
        self._config_notes: list[str] = []
        self._train_config: TrainConfig | None = None
        self._snapshot_steps: set[int] = set()
        self._groups: dict[str, list[str]] = {}
        self._layout: list[dict] = []
        self._attention_sites: list[tuple[str, str]] = []  # (modulo, strato) con attention_weights
        self._attention_problems: list[str] = []  # moduli di attention saltati, con il motivo
        self._device = "cpu"
        self._started_at = 0.0
        self.duration_s: float | None = None

    # --- hook del training -------------------------------------------------------------

    def on_train_start(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_cfg: TrainConfig) -> None:
        self._train_config = train_cfg
        self._model_config, self._config_notes = _model_config(model)
        self._groups = parameter_groups(model)
        self._layout = parameter_layout(_Tensors(snapshot_state(model)))
        self._attention_sites = [(site.path, site.layer_path) for site in attention_sites(model)]
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
    def _snapshot(self, step: int, model: nn.Module) -> None:
        was_training = model.training
        model.eval()
        try:
            entry = {"step": step, "file": snapshot_file(step)}
            entry["sample_text"] = self._sample(model)
            entry["example_next_token_probs"] = self._next_token_probs(model, self._example_ids)
            entry.update(self._eval_batch_metrics(model))
        finally:
            model.train(was_training)
        self._blobs[entry["file"]] = self._snapshot_bytes(model)
        self.snapshots.append(entry)

    def _snapshot_bytes(self, model: nn.Module) -> bytes:
        """I tensori del layout fissato all'inizio, in float16: tutti gli snapshot hanno lo stesso layout."""
        state = snapshot_state(model)
        tensors = {}
        for p in self._layout:
            tensor = state.get(p["name"])
            if tensor is None or list(tensor.shape) != p["shape"]:
                raise RuntimeError(f"il tensore {p['name']} è cambiato (o sparito) durante il training: gli "
                                   "snapshot devono avere tutti gli stessi tensori con le stesse forme")
            tensors[p["name"]] = tensor
        return parameters_to_bytes(_Tensors(tensors), "<f2")

    def _sample(self, model: nn.Module) -> str:
        generator = torch.Generator(device=self._device).manual_seed(self.config.sample_seed)
        idx = self._sample_ids.to(self._device)
        out = generate(model, idx, self.config.sample_length, block_size=self.block_size, generator=generator)
        return self.tokenizer.decode(out[0, idx.shape[1] :].tolist())

    def _next_token_probs(self, model: nn.Module, ids: torch.Tensor) -> list[float]:
        logits, _ = model(ids.to(self._device))
        return [round(p, 5) for p in F.softmax(logits[0, -1], dim=-1).tolist()]

    def _eval_batch_metrics(self, model: nn.Module) -> dict[str, Any]:
        x, y = self._eval_x.to(self._device), self._eval_y.to(self._device)
        # problems: un'attention che non dà probabilità valide si salta con una nota, non ferma il training.
        with capture_attention(model, problems=self._attention_problems) as attention:
            logits, loss = model(x, y)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        top5 = logits.topk(min(TOP_K, logits.shape[-1]), dim=-1).indices  # (B, T, 5)
        metrics = {
            "loss": round(loss.item(), 5),
            "top1": round(float((logits.argmax(-1) == y).float().mean()), 5),
            "top5": round(float((top5 == y.unsqueeze(-1)).any(-1).float().mean()), 5),
            "entropy": round(entropy.item(), 5),
        }
        if attention:  # senza moduli con attention_weights la metrica si salta (la nota è in notes)
            metrics["attention_entropy"] = _entropy_rows(attention)
        return metrics

    # --- serializzazione ---------------------------------------------------------------

    def notes(self) -> list[str]:
        """Che cosa è stato adattato o saltato per questa architettura (vuoto per il GPT del corso)."""
        with_attention = [s["attention_entropy"] for s in self.snapshots if "attention_entropy" in s]
        rows = with_attention[-1] if with_attention else None
        group_note = grouping_note(name for names in self._groups.values() for name in names)
        return [
            *self._config_notes,
            *([group_note] if group_note else []),
            *attention_notes(self._attention_sites, rows, self._attention_problems),
        ]

    def to_dict(self) -> dict[str, Any]:
        if self._train_config is None:
            raise RuntimeError("il recorder non ha ancora visto un training: passalo a train(callbacks=[...])")
        trace = {
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
        trace, non_finite = _finite_json(trace)
        notes = self.notes()
        if non_finite:
            notes.append(f"{_count(non_finite, 'valore non finito', 'valori non finiti')} (NaN o infinito) "
                         "scritti come null: il training è divergente? (learning rate troppo alto, una "
                         "divisione per zero in un modulo nuovo…)")
        if notes:  # il GPT del corso non ne ha: il suo trace resta quello di sempre
            trace["notes"] = notes
        return trace

    def save(self, out_dir: Path) -> Path:
        """Scrive ``trace.json`` e gli snapshot in ``out_dir``; restituisce il percorso di trace.json.

        Gli snapshot di registrazioni precedenti vengono rimossi: nella cartella
        restano solo i file citati da questo ``trace.json``.
        """
        trace = self.to_dict()
        # allow_nan=False: un JSON con NaN non si aprirebbe nel browser (to_dict li ha già tolti).
        text = json.dumps(trace, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob(SNAPSHOT_PATTERN):
            if stale.name not in self._blobs:
                stale.unlink()
        for name, blob in self._blobs.items():
            (out_dir / name).write_bytes(blob)
        path = out_dir / TRACE_FILE
        path.write_text(text, encoding="utf-8")
        return path
