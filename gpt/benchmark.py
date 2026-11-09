"""Benchmark deterministico e confrontabile del GPT: lo stesso metro per ogni run.

I pesi si eseguono con la classe del codice ATTUALE (``gpt/model.py``): se migliori l'architettura,
il modello nuovo si misura con lo stesso protocollo, sullo stesso testo, del GPT del corso.

Protocollo ``finestre-contigue-v1`` (identico in Python e nel browser)
-----------------------------------------------------------------------

1. Testo: la validation di Tiny Shakespeare, cioè l'ultimo 10% di ``data/input.txt`` (lo split di
   ``gpt/data.py``), codificata con il vocabolario del run: un carattere = un token, N token.
2. Finestre: il testo si taglia in finestre CONTIGUE che non si sovrappongono. La finestra k ha
   come ingresso i token ``[k·T, k·T + T)`` e come target gli stessi spostati di uno, con
   ``T = block_size`` del modello; l'ultima può essere più corta. Ogni token tranne il primo è
   previsto esattamente una volta (N − 1 previsioni), dal contesto dei token che lo precedono
   nella sua finestra (da 1 a T): le stesse condizioni del training, che fa la media della loss
   su tutte le posizioni di finestre lunghe block_size. Niente casualità: stessi pesi, stessi numeri.
3. Per ogni posizione prevista: logits del forward della finestra, log-softmax e
   ``nll = −ln p(token giusto)``. Poi:

   - ``loss`` = media delle nll (nat); ``perplexity`` = exp(loss);
   - ``bpc`` (bit per carattere) = Σ nll / (ln 2 · caratteri previsti): con un token per carattere
     vale loss / ln 2;
   - ``top1``, ``top5``: la previsione è giusta se MENO di k logits sono strettamente maggiori di
     quello del token giusto (nessuna ambiguità a pari merito, uguale in ogni linguaggio);
   - ``entropy``: media di −Σ p·ln p della distribuzione prevista (quanto il modello è indeciso).
4. Testo generato con prompt e seme fissi (predefiniti ``"ROMEO:\\n"``, seme 1337, temperatura 1,
   200 token): ``sampled`` con ``torch.multinomial`` e un ``torch.Generator`` dedicato (non tocca il
   generatore globale) e ``greedy``, sempre il token più probabile (a pari merito il primo indice).
   Il motore JS usa un altro generatore casuale: riproduce ``greedy``, non ``sampled``.
5. Prestazioni, che dipendono dalla macchina (confrontabili solo sullo stesso computer):
   ``tokens_per_s`` della generazione (batch 1, contesto ricalcolato a ogni token come
   ``gpt/generation.py``), ``eval_tokens_per_s`` (previsioni al secondo della valutazione a gruppi
   di finestre) e ``duration_s`` (tutto il benchmark). Su una macchina già carica conviene
   ``BenchmarkSettings(threads=1)``: thread in più dei core liberi si ostacolano a vicenda.

Lo stesso protocollo si applica all'estratto fisso ``webapp/data/benchmark/val_excerpt.txt`` (circa
4 KB della validation, discorsi interi): il browser lo valuta con il motore JS in un secondo circa e
confronta i suoi numeri con quelli calcolati qui.

Schema di ``benchmark.json`` (``builtin.json`` è il benchmark di ``webapp/data``)::

    {
      "format_version": 1,
      "name": "GPT del corso",
      "created": "2026-11-09T10:15:00+00:00",
      "source": "webapp/data",                 # checkpoint o cartella misurata (relativa al progetto)
      "protocol": {"id": "finestre-contigue-v1", "description": "...", "block_size": 32,
                   "top_k": [1, 5], "log_base": "e"},
      "data": {"name": "Tiny Shakespeare, validation", "file": "data/input.txt",
               "offset": 1003854, "chars": 111540, "sha256": "..."},   # il testo valutato
      "model": {"class": "gpt.model.GPTLanguageModel", "config": {...}, "params": 209729,
                "vocab_size": 65, "weights_sha256": "...",       # sha256 dei byte di weights.f32.bin
                "code_sha256": "...", "code_files": ["gpt/attention.py", ...]},  # sorgenti delle classi
      "metrics": {"loss", "perplexity", "bpc", "top1", "top5", "entropy",    # validation completa
                  "predictions", "params", "tokens_per_s", "eval_tokens_per_s", "duration_s"},
      "counts": {"predictions", "windows", "chars", "top1", "top5"},        # conteggi esatti
      "excerpt": {"file": "val_excerpt.txt", "chars", "sha256",
                  "offset",                      # dove inizia nella validation (null se non ci sta)
                  "loss", "perplexity", "bpc", "top1", "top5", "entropy", "counts": {...},
                  "surprisal_bits": [...]} | null,   # bit di sorpresa (−log2 p) di ogni carattere previsto
      "generation": {"prompt", "seed", "temperature", "max_new_tokens", "sampled", "greedy",
                     "tokens_per_s"} | null,
      "snapshots": [{"step", "file", "dtype": "float16",
                     "metrics": {loss, perplexity, bpc, top1, top5, entropy, predictions},
                     "excerpt": {loss, perplexity, bpc, top1, top5, entropy, counts} | null}],
      "reference": {"uniform_loss": 4.174387, "uniform_bpc": 6.022368},   # un modello che tira a caso
      "environment": {"python", "torch", "numpy", "device", "threads", "system", "machine"},
      "notes": ["..."]
    }

``snapshots`` c'è quando la cartella misurata ha la storia del training (``training/``): le stesse
metriche per i pesi di ogni snapshot, sulla validation completa e sull'estratto.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import math
import platform
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.config import GPTConfig
from gpt.data import CharTokenizer, load_text, train_val_split
from gpt.export import PROJECT_ROOT, TRAINING_DIR, TRAINING_TRACE, WEBAPP_DATA_DIR, WEIGHTS_MANIFEST
from gpt.generation import generate
from gpt.model import GPTLanguageModel
from gpt.runs import (
    RunWeights,
    TrainingHistory,
    read_checkpoint,
    read_run_folder,
    read_run_info,
    read_training_history,
    snapshot_tensors,
    weights_from_model,
)

BENCHMARK_FORMAT_VERSION = 1
PROTOCOL_ID = "finestre-contigue-v1"
PROTOCOL_DESCRIPTION = (
    "Validation di Tiny Shakespeare (ultimo 10% di data/input.txt, lo split di gpt/data.py) tagliata in "
    "finestre contigue di block_size token che non si sovrappongono (l'ultima può essere più corta): ogni "
    "carattere tranne il primo è previsto una volta, dal contesto dei caratteri che lo precedono nella sua "
    "finestra. Loss = media della cross-entropy in nat su tutte le posizioni previste; top-k giusto se meno "
    "di k logits superano quello del carattere vero."
)
TOP_K = (1, 5)
BENCHMARK_DATA_DIR = WEBAPP_DATA_DIR / "benchmark"
EXCERPT_FILE = "val_excerpt.txt"
BUILTIN_FILE = "builtin.json"
# 4096 previsioni al massimo: con il motore JS sono circa 128 forward da 32 token, ~1 s su un desktop.
EXCERPT_MAX_CHARS = 4097
BUILTIN_NAME = "GPT del corso"
VALIDATION_NAME = "Tiny Shakespeare, validation"
DATASET_FILE = "data/input.txt"

# Tolleranze per dire "stesso risultato" (benchmark_differences): pesi identici danno differenze
# di ~1e-6 tra macchine diverse (ordine delle somme in virgola mobile), ben sotto queste soglie.
LOSS_ATOL = 1e-4
HITS_ATOL = 2  # previsioni giuste in più o in meno ammesse (un quasi pari merito che si rompe)
SURPRISAL_ATOL = 2e-3  # i bit di sorpresa sono salvati con 3 decimali


@dataclass(frozen=True)
class BenchmarkSettings:
    """Le scelte fisse del benchmark: cambiarle rende i numeri non confrontabili con quelli salvati."""

    prompt: str = "ROMEO:\n"
    seed: int = 1337  # lo stesso seme del training del corso (e di DEFAULT_SEED nel motore JS)
    max_new_tokens: int = 200
    temperature: float = 1.0
    batch_windows: int = 512  # finestre per forward: cambia la velocità, non i numeri
    device: str = "cpu"  # la CPU dà gli stessi numeri ovunque meglio di una GPU
    # Thread di PyTorch (None = quelli predefiniti). Su una macchina già carica, più thread che core
    # liberi si ostacolano a vicenda (un forward da 1 ms può durare 1 s): lì conviene 1 o 2.
    threads: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("il prompt non può essere vuoto")
        if self.max_new_tokens < 0 or self.batch_windows <= 0:
            raise ValueError("max_new_tokens deve essere ≥ 0 e batch_windows positivo")
        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads deve essere positivo")
        if not self.temperature > 0:
            raise ValueError("temperature deve essere positiva")


DEFAULT_SETTINGS = BenchmarkSettings()


@dataclass(frozen=True)
class EvalText:
    """Un testo da valutare con la sua carta d'identità: dice se due misure sono confrontabili."""

    text: str
    name: str
    file: str
    offset: int | None = None  # dove inizia nel testo di partenza, se si sa

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "file": self.file, "offset": self.offset, "chars": len(self.text),
                "sha256": self.sha256}


@dataclass
class Scores:
    """Somme e conteggi di una valutazione, accumulati finestra dopo finestra."""

    predictions: int = 0
    windows: int = 0
    chars: int = 0
    nll: float = 0.0  # Σ −ln p(token giusto), in nat
    entropy: float = 0.0  # Σ entropia della distribuzione prevista, in nat
    hits: dict[int, int] = field(default_factory=lambda: {k: 0 for k in TOP_K})

    def metrics(self) -> dict[str, float]:
        loss = self.nll / self.predictions
        return {
            "loss": round(loss, 6),
            "perplexity": round(math.exp(loss), 6),
            "bpc": round(self.nll / (math.log(2) * self.chars), 6),
            **{f"top{k}": round(self.hits[k] / self.predictions, 6) for k in TOP_K},
            "entropy": round(self.entropy / self.predictions, 6),
        }

    def counts(self) -> dict[str, int]:
        return {"predictions": self.predictions, "windows": self.windows, "chars": self.chars,
                **{f"top{k}": self.hits[k] for k in TOP_K}}


class ArchitectureMismatch(ValueError):
    """I pesi non hanno i nomi e le forme che la classe del codice attuale si aspetta."""

    def __init__(self, missing: list[str], extra: list[str], mismatched: list[tuple[str, tuple, tuple]]):
        self.missing, self.extra, self.mismatched = missing, extra, mismatched

        def few(items: list[str]) -> str:
            return ", ".join(items[:3]) + (f" e altri {len(items) - 3}" if len(items) > 3 else "")

        parts = []
        if missing:
            parts.append(f"mancano {len(missing)} tensori ({few(missing)})")
        if extra:
            parts.append(f"{len(extra)} tensori che il codice non conosce ({few(extra)})")
        if mismatched:
            shapes = "; ".join(f"{n} {list(g)} invece di {list(e)}" for n, g, e in mismatched[:3])
            more = f" e altri {len(mismatched) - 3}" if len(mismatched) > 3 else ""
            parts.append(f"forme diverse in {len(mismatched)} tensori: {shapes}{more}")
        super().__init__(
            f"il codice attuale (gpt/model.py) non ricostruisce questi pesi: {'; '.join(parts)}. "
            "Per misurarli serve il codice che li ha prodotti; per confrontarli usa il loro benchmark.json."
        )


# --------------------------------------------------------------------------- protocollo


def contiguous_windows(n_tokens: int, block_size: int) -> list[tuple[int, int]]:
    """Finestre ``(inizio, lunghezza)`` del protocollo: ogni token tranne il primo è previsto una volta.

    La finestra che parte da ``s`` legge i token ``[s, s + lunghezza)`` e prevede quelli
    ``[s + 1, s + lunghezza + 1)``.
    """
    if block_size <= 0:
        raise ValueError("block_size deve essere positivo")
    if n_tokens < 2:
        raise ValueError("servono almeno 2 token: uno di contesto e uno da prevedere")
    return [(s, min(block_size, n_tokens - 1 - s)) for s in range(0, n_tokens - 1, block_size)]


@torch.no_grad()
def score_tokens(
    model: nn.Module,
    ids: Sequence[int] | torch.Tensor,
    *,
    block_size: int,
    token_chars: torch.Tensor | None = None,
    batch_windows: int = 512,
    surprisal: bool = False,
) -> tuple[Scores, list[float] | None]:
    """Valuta ``ids`` con il protocollo a finestre contigue.

    Args:
        token_chars: caratteri di ogni token del vocabolario (per i bit per carattere); senza, 1.
        surprisal: se vero restituisce anche i bit di sorpresa di ogni posizione, in ordine di testo.

    Le finestre complete hanno tutte la stessa lunghezza: sono le righe di una matrice e passano nel
    modello a gruppi di ``batch_windows``. L'ultima, più corta, passa da sola.
    """
    ids = torch.as_tensor(ids, dtype=torch.long).flatten().cpu()
    windows = contiguous_windows(ids.numel(), block_size)
    full = sum(1 for _, length in windows if length == block_size)
    device = next(model.parameters()).device
    scores = Scores()
    bits: list[float] | None = [] if surprisal else None
    was_training = model.training
    model.eval()
    try:
        inputs = ids[: full * block_size].view(full, block_size)
        targets = ids[1 : full * block_size + 1].view(full, block_size)
        for i in range(0, full, batch_windows):
            batch = slice(i, i + batch_windows)
            _accumulate(model, inputs[batch], targets[batch], device, scores, token_chars, bits)
        if len(windows) > full:
            start, length = windows[-1]
            x, y = ids[start : start + length], ids[start + 1 : start + length + 1]
            _accumulate(model, x[None], y[None], device, scores, token_chars, bits)
    finally:
        model.train(was_training)
    return scores, bits


def _accumulate(model, x, y, device, scores: Scores, token_chars, bits) -> None:
    logits, _ = model(x.to(device))  # (B, t, V)
    logits = logits.float()
    target = y.to(device).unsqueeze(-1)  # (B, t, 1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, target).squeeze(-1)  # (B, t)
    entropy = -(log_probs.exp() * log_probs).sum(-1)  # (B, t)
    # Quanti logits superano STRETTAMENTE quello giusto: 0 = primo posto (top-1), < 5 = top-5.
    greater = (logits > logits.gather(-1, target)).sum(-1)  # (B, t)
    scores.predictions += y.numel()
    scores.windows += x.shape[0]
    scores.chars += int(token_chars[y].sum()) if token_chars is not None else y.numel()
    scores.nll += nll.double().sum().item()  # somme in float64: 111 mila termini senza perdere cifre
    scores.entropy += entropy.double().sum().item()
    for k in TOP_K:
        scores.hits[k] += int((greater < k).sum())
    if bits is not None:
        bits.extend((nll.double().cpu() / math.log(2)).flatten().tolist())


@torch.no_grad()
def greedy_continuation(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, block_size: int):
    """Sempre il token più probabile (``argmax``: a pari merito il primo indice, come il motore JS)."""
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -block_size:])
            idx = torch.cat((idx, logits[:, -1, :].argmax(dim=-1, keepdim=True)), dim=1)
    finally:
        model.train(was_training)
    return idx


def generation_sample(
    model: nn.Module, tokenizer: CharTokenizer, settings: BenchmarkSettings, block_size: int
) -> dict[str, Any]:
    """Testo campionato con seme fisso, continuazione greedy e token al secondo della generazione."""
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(settings.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        model.eval()
        model(idx[:, -block_size:])  # riscaldamento: le allocazioni del primo forward non contano nel tempo
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    start = time.perf_counter()
    sampled = generate(model, idx, settings.max_new_tokens, block_size=block_size,
                       temperature=settings.temperature, generator=generator)
    seconds = time.perf_counter() - start
    greedy = greedy_continuation(model, idx, settings.max_new_tokens, block_size)
    n = idx.shape[1]
    speed = settings.max_new_tokens / seconds if seconds > 0 and settings.max_new_tokens else None
    return {
        "prompt": settings.prompt,
        "seed": settings.seed,
        "temperature": settings.temperature,
        "max_new_tokens": settings.max_new_tokens,
        "sampled": tokenizer.decode(sampled[0, n:].tolist()),
        "greedy": tokenizer.decode(greedy[0, n:].tolist()),
        "tokens_per_s": round(speed, 1) if speed else None,
    }


# --------------------------------------------------------------------------- testi


def validation_text(text: str | None = None) -> EvalText:
    """La validation di Tiny Shakespeare: la regola di ``train_val_split`` applicata ai caratteri.

    Con un tokenizer a caratteri un token è un carattere, quindi tagliare il testo o i token è uguale;
    la regola (il 90%) la decide ``gpt/data.py``, qui non si ricopia.
    """
    text = load_text() if text is None else text
    n = len(train_val_split(torch.empty(len(text), dtype=torch.bool)).train)
    return EvalText(text[n:], VALIDATION_NAME, DATASET_FILE, n)


def excerpt_bounds(validation: str, max_chars: int = EXCERPT_MAX_CHARS) -> tuple[int, int]:
    """Dove sta l'estratto per il browser nella validation: discorsi interi, dall'inizio del primo.

    Parte dopo la prima riga vuota (la validation comincia a metà di una battuta) e finisce con
    l'ultimo discorso completo che entra in ``max_chars`` caratteri.
    """
    blank = validation.find("\n\n")
    start = blank + 2 if blank >= 0 else 0
    chunk = validation[start : start + max_chars]
    cut = chunk.rfind("\n\n")
    return start, start + (cut + 1 if cut > 0 else len(chunk))


def read_excerpt(path: Path, validation: EvalText | None = None) -> EvalText:
    """L'estratto salvato su disco (byte UTF-8 esatti: niente conversioni degli a capo)."""
    path = Path(path)
    text = path.read_bytes().decode("utf-8")
    offset = validation.text.find(text) if validation is not None else -1
    return EvalText(text, "estratto della validation", path.name, offset if offset >= 0 else None)


# --------------------------------------------------------------------------- modelli da misurare


@dataclass(frozen=True)
class BenchmarkSource:
    """Che cosa si misura: pesi, nome da mostrare e, per una cartella di run, la storia del training."""

    name: str
    source: str
    weights: RunWeights
    history: TrainingHistory | None = None


def open_source(path: Path, *, name: str | None = None, with_history: bool = True) -> BenchmarkSource:
    """Un checkpoint ``.pt`` oppure una cartella con ``weights.json`` (un run esportato o ``webapp/data``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} non esiste: serve un checkpoint .pt o la cartella di un run")
    if not path.is_dir():
        weights = read_checkpoint(path)
        return BenchmarkSource(name or path.stem, weights.source, weights)
    if not (path / WEIGHTS_MANIFEST).is_file():
        raise FileNotFoundError(f"{path} non contiene {WEIGHTS_MANIFEST}: non è la cartella di un run")
    weights = read_run_folder(path)
    declared = (read_run_info(path) or {}).get("name")
    if not isinstance(declared, str):
        declared = BUILTIN_NAME if path.resolve() == WEBAPP_DATA_DIR.resolve() else path.name
    trace = path / TRAINING_DIR / TRAINING_TRACE
    history = read_training_history(trace.parent) if with_history and trace.is_file() else None
    return BenchmarkSource(name or declared, weights.source, weights, history)


def build_model(weights: RunWeights) -> GPTLanguageModel:
    """Il modello del codice ATTUALE con i pesi del run: nomi e forme devono coincidere esattamente.

    Si confrontano i tensori in virgola mobile dello state_dict, gli unici che un run contiene (l'export
    salta quelli interi, come ``num_batches_tracked`` di una BatchNorm, e lo scrive in ``skipped``): un
    tensore intero resta quello del modello appena costruito, invece di rendere il run "non ricostruibile".
    """
    if weights.config is None:
        raise ValueError("manca la config del modello (gli iperparametri): il codice non sa che forme creare")
    try:
        config = GPTConfig(**weights.config)
    except TypeError as error:
        raise ValueError(f"la config del run non va bene per GPTConfig del codice attuale: {error}") from None
    # Prima le sole forme, sul device "meta" (nessuna memoria): la config viene da un file (weights.json o
    # run.json, magari scaricato) e una sbagliata o enorme (n_embd 4096 accanto a pesi da 64) si scopre
    # senza allocare. Prima il modello vero nasceva subito: 25 GB per quella config, o un RuntimeError.
    with torch.device("meta"):
        shapes = GPTLanguageModel(config).state_dict()
    expected = {n: tuple(t.shape) for n, t in shapes.items() if t.is_floating_point()}
    got = {n: tuple(t.shape) for n, t in weights.tensors.items()}
    missing = [n for n in expected if n not in got]
    extra = [n for n in got if n not in expected]
    mismatched = [(n, got[n], expected[n]) for n in expected if n in got and got[n] != expected[n]]
    if missing or extra or mismatched:
        raise ArchitectureMismatch(missing, extra, mismatched)
    model = GPTLanguageModel(config)  # ora le forme combaciano: il modello vero, grande quanto i pesi
    # stretto: tutto lo state_dict, con i pesi del run sopra (i tensori interi restano quelli del codice)
    model.load_state_dict({**model.state_dict(), **weights.tensors})
    return model.eval()


def run_tokenizer(weights: RunWeights, default_vocab: list[str] | None = None) -> CharTokenizer:
    """Il tokenizer del run; senza vocabolario nel run, ``default_vocab`` se ha la misura giusta."""
    size = (weights.config or {}).get("vocab_size")
    vocab = weights.vocab
    if vocab is None and default_vocab is not None and len(default_vocab) == size:
        vocab = default_vocab
    if vocab is None:
        raise ValueError(f"serve un vocabolario di {size} token: il run non lo contiene")
    return CharTokenizer(vocab)


# --------------------------------------------------------------------------- benchmark


def benchmark_model(
    model: nn.Module,
    tokenizer: CharTokenizer,
    *,
    name: str,
    source: str,
    validation: EvalText,
    excerpt: EvalText | None = None,
    history: TrainingHistory | None = None,
    weights: RunWeights | None = None,
    settings: BenchmarkSettings = DEFAULT_SETTINGS,
    created: datetime | None = None,
) -> dict[str, Any]:
    """Misura ``model`` (con ``config.block_size``) e restituisce il contenuto di ``benchmark.json``.

    ``model`` viene spostato su ``settings.device`` e messo in modalità eval. ``weights`` (se dato)
    sono i pesi da cui viene, per l'impronta; altrimenti la si calcola dal modello.
    """
    with _torch_threads(settings.threads):
        return _benchmark_model(model, tokenizer, name=name, source=source, validation=validation,
                                excerpt=excerpt, history=history, weights=weights, settings=settings,
                                created=created)


def _benchmark_model(model, tokenizer, *, name, source, validation, excerpt, history, weights, settings,
                     created):
    started = time.perf_counter()
    device = torch.device(settings.device)
    model.to(device).eval()
    block_size = model.config.block_size
    token_chars = torch.tensor([len(t) for t in tokenizer.chars])
    notes: list[str] = []
    try:
        val_ids = tokenizer.encode(validation.text)
    except ValueError as error:
        raise ValueError(f"il vocabolario del modello non copre la validation ({error})") from None

    def score(net, text_ids):
        return score_tokens(net, text_ids, block_size=block_size, token_chars=token_chars,
                            batch_windows=settings.batch_windows)[0]

    eval_start = time.perf_counter()
    scores = score(model, val_ids)
    eval_seconds = time.perf_counter() - eval_start
    excerpt_section = None
    if excerpt is not None:
        excerpt_section = _text_section(model, tokenizer, excerpt, block_size, token_chars, settings, notes,
                                        surprisal=True)
    try:
        generation = generation_sample(model, tokenizer, settings, block_size)
    except ValueError as error:  # un prompt con caratteri fuori vocabolario
        generation = None
        notes.append(f"testo generato saltato: {error}")

    snapshots = []
    if history is not None:
        for entry, snap in _snapshot_models(model, history, notes):
            snap_excerpt = None
            if excerpt is not None:
                section = _text_section(snap, tokenizer, excerpt, block_size, token_chars, settings, notes)
                snap_excerpt = section and {k: section[k] for k in (*_METRIC_KEYS, "counts")}
            snap_scores = score(snap, val_ids)
            snapshots.append({
                "step": entry["step"],
                "file": f"{TRAINING_DIR}/{entry['file']}",
                "dtype": "float16",
                "metrics": {**snap_scores.metrics(), "predictions": snap_scores.predictions},
                "excerpt": snap_excerpt,
            })

    params = sum(p.numel() for p in model.parameters())
    vocab_size = len(tokenizer.chars)
    sources = code_files(model)
    config = getattr(model, "config", None)
    return {
        "format_version": BENCHMARK_FORMAT_VERSION,
        "name": name,
        "created": (created or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "source": source,
        "protocol": {"id": PROTOCOL_ID, "description": PROTOCOL_DESCRIPTION, "block_size": block_size,
                     "top_k": list(TOP_K), "log_base": "e"},
        "data": validation.info(),
        "model": {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": config.to_dict() if hasattr(config, "to_dict") else None,
            "params": params,
            "vocab_size": vocab_size,
            "weights_sha256": (weights or weights_from_model(model)).sha256(),
            "code_sha256": code_sha256(sources),
            "code_files": [f.relative_to(PROJECT_ROOT).as_posix() for f in sources],
        },
        "metrics": {
            **scores.metrics(),
            "predictions": scores.predictions,
            "params": params,
            "tokens_per_s": generation["tokens_per_s"] if generation else None,
            "eval_tokens_per_s": round(scores.predictions / eval_seconds, 1) if eval_seconds > 0 else None,
            "duration_s": round(time.perf_counter() - started, 2),
        },
        "counts": scores.counts(),
        "excerpt": excerpt_section,
        "generation": generation,
        "snapshots": snapshots,
        "reference": {"uniform_loss": round(math.log(vocab_size), 6),
                      "uniform_bpc": round(math.log2(vocab_size), 6)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "threads": torch.get_num_threads(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "notes": notes,
    }


def benchmark_path(
    path: Path,
    *,
    name: str | None = None,
    settings: BenchmarkSettings = DEFAULT_SETTINGS,
    text: str | None = None,
    excerpt_path: Path | None = BENCHMARK_DATA_DIR / EXCERPT_FILE,
    snapshots: bool = True,
) -> dict[str, Any]:
    """Benchmark di un checkpoint o di una cartella di run (con gli snapshot, se ci sono e ``snapshots``)."""
    src = open_source(path, name=name, with_history=snapshots)
    text = load_text() if text is None else text
    model = build_model(src.weights)
    tokenizer = run_tokenizer(src.weights, default_vocab=CharTokenizer.from_text(text).chars)
    validation = validation_text(text)
    excerpt = None
    if excerpt_path is not None and Path(excerpt_path).is_file():
        excerpt = read_excerpt(excerpt_path, validation)
    bench = benchmark_model(model, tokenizer, name=src.name, source=src.source, validation=validation,
                            excerpt=excerpt, history=src.history, weights=src.weights, settings=settings)
    note = rebuilt_tensors_note(model)
    if note:
        bench["notes"].append(note)
    return bench


def rebuilt_tensors_note(model: nn.Module) -> str | None:
    """La nota sui tensori interi o booleani di un modello ricostruito da un run con ``build_model`` (None se
    non ne ha): il run non li contiene (l'export li salta e li scrive in ``skipped``), quindi valgono quanto
    li crea il codice. La aggiungono al benchmark ``benchmark_path`` e ``scripts/export_run.py --benchmark``.
    """
    rebuilt = [n for n, t in model.state_dict().items() if not t.is_floating_point()]
    if not rebuilt:
        return None
    return (f"tensori interi o booleani che il run non contiene ({_few_names(rebuilt)}): "
            "misurati con il valore che dà loro il codice")


def load_benchmark(path: Path) -> dict[str, Any]:
    """Un ``benchmark.json`` già calcolato (per confrontare senza rifare i conti)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} non trovato (un benchmark.json si crea con --out o export_run.py)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: JSON non valido ({error})") from None
    if not isinstance(data, dict) or data.get("format_version") != BENCHMARK_FORMAT_VERSION:
        raise ValueError(f"{path}: non è un benchmark.json con format_version {BENCHMARK_FORMAT_VERSION}")
    if not isinstance(data.get("metrics"), dict):
        raise ValueError(f"{path}: manca la sezione metrics")
    return data


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """JSON valido per il browser (niente NaN), compatto come gli altri dati della web app."""
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    Path(path).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- dati per il browser


def write_benchmark_data(
    out_dir: Path = BENCHMARK_DATA_DIR,
    *,
    data_dir: Path = WEBAPP_DATA_DIR,
    text: str | None = None,
    settings: BenchmarkSettings = DEFAULT_SETTINGS,
) -> dict[str, Any]:
    """Scrive ``val_excerpt.txt`` e ``builtin.json`` (benchmark dei pesi committati); restituisce builtin."""
    builtin, excerpt = compute_benchmark_data(data_dir=data_dir, text=text, settings=settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EXCERPT_FILE).write_bytes(excerpt.encode("utf-8"))
    write_json(out_dir / BUILTIN_FILE, builtin)
    return builtin


def compute_benchmark_data(
    *,
    data_dir: Path = WEBAPP_DATA_DIR,
    text: str | None = None,
    settings: BenchmarkSettings = DEFAULT_SETTINGS,
) -> tuple[dict[str, Any], str]:
    """``(builtin, testo dell'estratto)`` ricalcolati da ``data/input.txt`` e dai pesi in ``data_dir``."""
    text = load_text() if text is None else text
    validation = validation_text(text)
    start, end = excerpt_bounds(validation.text)
    excerpt = EvalText(validation.text[start:end], "estratto della validation", EXCERPT_FILE, start)
    src = open_source(data_dir, name=BUILTIN_NAME)
    model, tokenizer = build_model(src.weights), run_tokenizer(src.weights)
    builtin = benchmark_model(model, tokenizer, name=BUILTIN_NAME, source=src.source, validation=validation,
                              excerpt=excerpt, history=src.history, weights=src.weights, settings=settings)
    return builtin, excerpt.text


def benchmark_differences(committed: Mapping[str, Any], fresh: Mapping[str, Any]) -> list[str]:
    """Che cosa cambia tra un benchmark salvato e uno ricalcolato (lista vuota = aggiornato).

    Si confrontano i numeri deterministici con le tolleranze del modulo; data, tempi, ambiente e
    impronta del codice (che cambia anche per un commento) si ignorano.
    """
    diffs: list[str] = []

    def same(path: str, a: Any, b: Any) -> None:
        if a != b:
            diffs.append(f"{path}: {a!r} nel file, {b!r} ricalcolato")

    def close(path: str, a: Any, b: Any, atol: float) -> None:
        if not (_is_number(a) and _is_number(b) and abs(a - b) <= atol):
            diffs.append(f"{path}: {a!r} nel file, {b!r} ricalcolato (tolleranza {atol:g})")

    def section(path: str, a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> None:
        if (a is None) != (b is None):
            diffs.append(f"{path}: presente solo in uno dei due")
            return
        if a is None:
            return
        for key in ("loss", "bpc", "entropy"):
            close(f"{path}.{key}", a.get(key), b.get(key), LOSS_ATOL)
        perplexity = b.get("perplexity") or 1.0
        close(f"{path}.perplexity", a.get("perplexity"), perplexity, LOSS_ATOL * max(1.0, perplexity))
        n = b.get("predictions") or (b.get("counts") or {}).get("predictions") or 1
        for k in TOP_K:
            close(f"{path}.top{k}", a.get(f"top{k}"), b.get(f"top{k}"), HITS_ATOL / n)

    for key in ("format_version", "name"):
        same(key, committed.get(key), fresh.get(key))
    for key in ("id", "block_size", "top_k"):
        same(f"protocol.{key}", (committed.get("protocol") or {}).get(key), fresh["protocol"].get(key))
    same("data", committed.get("data"), fresh.get("data"))
    for key in ("config", "params", "vocab_size", "weights_sha256"):
        same(f"model.{key}", (committed.get("model") or {}).get(key), fresh["model"].get(key))
    section("metrics", committed.get("metrics"), fresh.get("metrics"))
    for key in ("predictions", "windows", "chars"):
        same(f"counts.{key}", (committed.get("counts") or {}).get(key), fresh["counts"].get(key))

    excerpt_a, excerpt_b = committed.get("excerpt"), fresh.get("excerpt")
    section("excerpt", excerpt_a, excerpt_b)
    if excerpt_a and excerpt_b:
        for key in ("file", "chars", "sha256", "offset"):
            same(f"excerpt.{key}", excerpt_a.get(key), excerpt_b.get(key))
        bits_a, bits_b = excerpt_a.get("surprisal_bits") or [], excerpt_b.get("surprisal_bits") or []
        same_length = len(bits_a) == len(bits_b)
        if not same_length or any(abs(x - y) > SURPRISAL_ATOL for x, y in zip(bits_a, bits_b, strict=True)):
            diffs.append("excerpt.surprisal_bits: valori diversi")

    gen_a, gen_b = committed.get("generation") or {}, fresh.get("generation") or {}
    for key in ("prompt", "seed", "temperature", "max_new_tokens", "greedy"):
        same(f"generation.{key}", gen_a.get(key), gen_b.get(key))
    # torch.multinomial può cambiare algoritmo tra versioni di torch: il testo campionato si
    # confronta solo con la stessa versione con cui è stato salvato.
    if (committed.get("environment") or {}).get("torch") == torch.__version__:
        same("generation.sampled", gen_a.get("sampled"), gen_b.get("sampled"))

    snaps_a = {s.get("step"): s for s in committed.get("snapshots") or []}
    snaps_b = {s["step"]: s for s in fresh.get("snapshots") or []}
    same("snapshots (step)", sorted(snaps_a), sorted(snaps_b))
    for step in sorted(set(snaps_a) & set(snaps_b)):
        section(f"snapshots[{step}].metrics", snaps_a[step].get("metrics"), snaps_b[step].get("metrics"))
        section(f"snapshots[{step}].excerpt", snaps_a[step].get("excerpt"), snaps_b[step].get("excerpt"))
    return diffs


# --------------------------------------------------------------------------- testo per il terminale

_METRIC_KEYS = ("loss", "perplexity", "bpc", "top1", "top5", "entropy")
_ROWS = (
    # chiave in metrics, etichetta, formato, verso: −1 più basso è meglio, +1 più alto è meglio, 0 nessuno
    ("loss", "loss (nat)", "{:.4f}", -1),
    ("perplexity", "perplexity", "{:.3f}", -1),
    ("bpc", "bit per carattere", "{:.3f}", -1),
    ("top1", "top-1 (il giusto è il primo)", "pct", 1),
    ("top5", "top-5 (il giusto è tra i primi 5)", "pct", 1),
    ("entropy", "entropia media (nat)", "{:.3f}", 0),
    ("params", "parametri", "int", 0),
    # Velocità: dipendono dalla macchina e dal suo carico, quindi nessun verdetto automatico.
    ("tokens_per_s", "token/s in generazione", "{:.0f}", 0),
    ("eval_tokens_per_s", "previsioni/s in valutazione", "{:.0f}", 0),
)


def comparison_rows(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Righe del confronto A/B: ``{key, label, a, b, delta, better}``, ``better`` in {"A", "B", "=", None}."""
    rows = []
    for key, label, _, direction in _ROWS:
        va, vb = (a.get("metrics") or {}).get(key), (b.get("metrics") or {}).get(key)
        delta = vb - va if _is_number(va) and _is_number(vb) else None
        better = None
        if delta is not None and direction:
            better = "=" if abs(delta) < 1e-12 else ("B" if delta * direction > 0 else "A")
        rows.append({"key": key, "label": label, "a": va, "b": vb, "delta": delta, "better": better})
    return rows


def comparison_warnings(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(livello, testo)``: "attenzione" se i numeri non sono confrontabili, "nota" se li distingue."""
    pa, pb = a.get("protocol") or {}, b.get("protocol") or {}
    ma, mb = a.get("model") or {}, b.get("model") or {}
    ea, eb = a.get("environment") or {}, b.get("environment") or {}
    out: list[tuple[str, str]] = []

    def add(level: str, *parts: str) -> None:
        out.append((level, "".join(parts)))

    if pa.get("id") != pb.get("id"):
        add("attenzione", f"protocolli diversi ({pa.get('id')} contro {pb.get('id')}): ",
            "i numeri non sono confrontabili")
    if (a.get("data") or {}).get("sha256") != (b.get("data") or {}).get("sha256"):
        add("attenzione", "testi di validation diversi: i numeri non sono confrontabili")
    if pa.get("block_size") != pb.get("block_size"):
        add("nota", f"contesti diversi (block_size {pa.get('block_size')} contro {pb.get('block_size')}): ",
            "un contesto più lungo aiuta a prevedere")
    if ma.get("vocab_size") != mb.get("vocab_size"):
        add("attenzione", f"vocabolari diversi ({ma.get('vocab_size')} contro {mb.get('vocab_size')} token)")
    if ma.get("weights_sha256") and ma.get("weights_sha256") == mb.get("weights_sha256"):
        add("nota", "stessi pesi, bit per bit")
    elif ma.get("code_sha256") and mb.get("code_sha256") and ma["code_sha256"] != mb["code_sha256"]:
        add("nota", "sorgenti del modello diversi (code_sha256): è un confronto prima/dopo una modifica ",
            "al codice")
    if any(ea.get(k) != eb.get(k) for k in ("system", "machine", "threads", "torch", "device")):
        add("nota", "velocità misurate in condizioni diverse (macchina, thread o versione di torch): ",
            "non confrontarle")
    return out


# Caratteri di controllo (C0 tranne a capo e tabulazione, DEL, C1) e controlli bidirezionali: i testi stampati nel
# terminale vengono anche da file altrui (nome, note e testi generati di un benchmark.json, il vocabolario di un
# run scaricato) e un ESC comanderebbe il terminale (colori, titolo della finestra, appunti con OSC 52), un "\r"
# riscriverebbe la riga. Si stampano visibili, come escape.
_TERMINAL_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def terminal_text(text: str) -> str:
    """``text`` da stampare in un terminale: i caratteri di ``_TERMINAL_UNSAFE`` scritti come ``\\x1b``."""
    return _TERMINAL_UNSAFE.sub(lambda m: f"\\x{ord(m.group()):02x}" if ord(m.group()) < 0x100
                                else f"\\u{ord(m.group()):04x}", text)


def format_summary(bench: Mapping[str, Any]) -> str:
    """Il benchmark di un run come testo da terminale, in italiano."""
    m, c, ref = bench["metrics"], bench.get("counts") or {}, bench.get("reference") or {}
    env = bench.get("environment") or {}
    block = (bench.get("protocol") or {}).get("block_size")
    uniform = ref.get("uniform_loss", float("nan"))
    lines = [
        f"Benchmark: {bench['name']} ({bench.get('source')})",
        (f"Validation di Tiny Shakespeare: {format_int(m['predictions'])} caratteri previsti in "
         f"{format_int(c.get('windows'))} finestre contigue da {block}"),
        f"  loss               {m['loss']:.4f} nat   (un modello che tira a caso: {uniform:.4f})",
        (f"  perplexity         {m['perplexity']:.3f}       (come scegliere tra ~{m['perplexity']:.0f} "
         "caratteri equiprobabili)"),
        f"  bit per carattere  {m['bpc']:.3f}",
        f"  top-1              {_pct(m['top1'])}      (il carattere giusto è il più probabile)",
        f"  top-5              {_pct(m['top5'])}      (il carattere giusto è tra i 5 più probabili)",
        f"  entropia media     {m['entropy']:.3f} nat",
    ]
    ex = bench.get("excerpt")
    if ex:
        predictions = format_int(ex["counts"]["predictions"])
        lines.append(f"Estratto per il browser ({ex['file']}, {predictions} previsioni): "
                     f"loss {ex['loss']:.4f} · top-1 {_pct(ex['top1'])} · top-5 {_pct(ex['top5'])}")
    speed = [f"{format_int(m['params'])} parametri"]
    if m.get("tokens_per_s"):
        speed.append(f"{m['tokens_per_s']:.0f} token/s in generazione")
    if m.get("eval_tokens_per_s"):
        speed.append(f"{format_int(round(m['eval_tokens_per_s']))} previsioni/s in valutazione")
    speed.append(f"{m['duration_s']:.1f} s ({env.get('device')}, {env.get('threads')} thread)")
    lines.append(" · ".join(speed))
    snaps = bench.get("snapshots") or []
    if snaps:
        lines.append("Durante il training (snapshot float16, validation completa):")
        for s in snaps:
            sm = s["metrics"]
            lines.append(f"  step {s['step']:>5}: loss {sm['loss']:.4f} · top-1 {_pct(sm['top1'])} "
                         f"· top-5 {_pct(sm['top5'])}")
    gen = bench.get("generation")
    if gen:
        lines.append(f"Testo generato (prompt {gen['prompt']!r}, seme {gen['seed']}, "
                     f"temperatura {gen['temperature']}):")
        lines += _indent(gen["prompt"] + gen["sampled"])
        lines.append("Greedy (sempre il carattere più probabile):")
        lines += _indent(gen["prompt"] + gen["greedy"])
    lines += [f"Nota: {n}" for n in bench.get("notes") or []]
    return terminal_text("\n".join(lines))  # nomi, note e testi possono venire da un benchmark.json altrui


def format_comparison(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    """Tabella A/B leggibile in italiano, con avvisi di confrontabilità e testi generati."""
    lines = [f"A = {a['name']} ({a.get('source')})", f"B = {b['name']} ({b.get('source')})", ""]
    header = ("metrica", "A", "B", "B − A", "meglio")
    table = [header]
    formats = {key: fmt for key, _, fmt, _ in _ROWS}
    verdicts = {"A": "A", "B": "B", "=": "pari"}
    for row in comparison_rows(a, b):
        fmt = formats[row["key"]]
        table.append((row["label"], _value(row["a"], fmt), _value(row["b"], fmt), _delta(row["delta"], fmt),
                      verdicts.get(row["better"], "")))
    ea, eb = a.get("excerpt"), b.get("excerpt")
    if ea and eb:
        delta = eb["loss"] - ea["loss"]
        verdict = "pari" if abs(delta) < 1e-12 else ("B" if delta < 0 else "A")
        table.append(("loss sull'estratto (browser)", f"{ea['loss']:.4f}", f"{eb['loss']:.4f}",
                      f"{delta:+.4f}", verdict))
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for i, row in enumerate(table):
        numbers = (cell.rjust(w) for cell, w in zip(row[1:4], widths[1:4], strict=True))
        cells = [row[0].ljust(widths[0]), *numbers, row[4]]
        lines.append("  ".join(cells).rstrip())
        if i == 0:
            lines.append("  ".join("─" * w for w in widths))
    lines.append("Loss, perplexity e bit per carattere: più bassi è meglio. Top-1 e top-5: più alti è meglio "
                 "(differenza in punti percentuali).")

    sa = {s["step"]: s for s in a.get("snapshots") or []}
    sb = {s["step"]: s for s in b.get("snapshots") or []}
    common = sorted(set(sa) & set(sb))
    if common:
        lines += ["", "Loss sulla validation durante il training (snapshot):"]
        for step in common:
            loss_a, loss_b = sa[step]["metrics"]["loss"], sb[step]["metrics"]["loss"]
            lines.append(f"  step {step:>5}:  A {loss_a:.4f}   B {loss_b:.4f}")
    warnings = comparison_warnings(a, b)
    if warnings:
        lines += ["", *[f"{level.capitalize()}: {text}" for level, text in warnings]]
    ga, gb = a.get("generation"), b.get("generation")
    if ga and gb:
        lines += ["", f"Testo generato (prompt {ga['prompt']!r}, seme {ga['seed']}):",
                  "A:", *_indent(ga["prompt"] + ga["sampled"]), "B:", *_indent(gb["prompt"] + gb["sampled"])]
        if (ga["prompt"], ga["seed"]) != (gb["prompt"], gb["seed"]):
            lines.append("Nota: prompt o seme diversi, i due testi non sono confrontabili")
    return terminal_text("\n".join(lines))  # nomi, note e testi possono venire da un benchmark.json altrui


# --------------------------------------------------------------------------- interno


@contextlib.contextmanager
def _torch_threads(threads: int | None):
    """Imposta i thread di PyTorch per la durata del benchmark e poi rimette quelli di prima."""
    if threads is None:
        yield
        return
    before = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def _text_section(model, tokenizer, text: EvalText, block_size, token_chars, settings, notes, *,
                  surprisal=False):
    """Metriche di un testo (l'estratto), o None se il vocabolario non lo copre."""
    try:
        ids = tokenizer.encode(text.text)
    except ValueError as error:
        notes.append(f"{text.file} saltato: {error}")
        return None
    scores, bits = score_tokens(model, ids, block_size=block_size, token_chars=token_chars,
                                batch_windows=settings.batch_windows, surprisal=surprisal)
    section = {"file": text.file, "chars": len(text.text), "sha256": text.sha256, "offset": text.offset,
               **scores.metrics(), "counts": scores.counts()}
    if bits is not None:
        section["surprisal_bits"] = [round(b, 3) for b in bits]
    return section


def _snapshot_models(model: nn.Module, history: TrainingHistory, notes: list[str]):
    """``(voce, modello)`` per ogni snapshot del training, con la stessa classe del modello misurato.

    Uno snapshot contiene i tensori IN VIRGOLA MOBILE dello state_dict (``snapshot_state`` di
    gpt/recording.py): si confrontano e si caricano solo quelli. Un tensore intero o booleano, come
    ``num_batches_tracked`` di una BatchNorm, negli snapshot non c'è e non deve farli saltare tutti:
    resta quello del modello misurato, e una nota lo dice.
    """
    state = model.state_dict()
    expected = {n: tuple(t.shape) for n, t in state.items() if t.is_floating_point()}
    if {p["name"]: tuple(p["shape"]) for p in history.layout} != expected:
        notes.append("snapshot del training saltati: hanno tensori diversi da quelli del modello")
        return
    others = [n for n, t in state.items() if not t.is_floating_point()]
    if others:
        notes.append(f"snapshot: i tensori interi o booleani dello state_dict ({_few_names(others)}) non "
                     "sono negli snapshot, che hanno solo quelli in virgola mobile: misurando gli snapshot "
                     "restano quelli del modello misurato")
    if history.missing:
        notes.append(f"snapshot senza file, non misurati: step {', '.join(map(str, history.missing))}")
    for entry in history.snapshots:
        snap = copy.deepcopy(model)
        # Caricamento stretto: lo state_dict completo, con i tensori dello snapshot al posto dei loro.
        snap.load_state_dict({**snap.state_dict(), **snapshot_tensors(history, entry)})
        yield entry, snap


def code_files(model: nn.Module) -> list[Path]:
    """I file del progetto che definiscono le classi dei moduli del modello (gpt/model.py, gpt/block.py…).

    Con ``code_sha256`` è l'impronta del codice scritta in ``benchmark.json`` (``model.code_files``,
    ``model.code_sha256``); ``scripts/record_training.py`` la usa per riconoscere il codice del corso.
    Le classi di PyTorch (nn.Linear…) non contano: non sono codice del corso e la .venv, che sta
    dentro la cartella del progetto, le farebbe sembrare tali.
    """
    files = set()
    for module in model.modules():
        cls = type(module)
        if cls.__module__.split(".")[0] == "torch":
            continue
        try:
            source = inspect.getsourcefile(cls)
        except TypeError:  # classi senza sorgente Python
            continue
        path = Path(source).resolve() if source else None
        if path and path.is_relative_to(PROJECT_ROOT) and "site-packages" not in path.parts:
            files.add(path)
    return sorted(files)


def code_sha256(files: list[Path]) -> str | None:
    """sha256 dei file di ``code_files`` (nome relativo al progetto e contenuto), o None senza file.

    Cambia con qualunque modifica di quei file, anche un commento: dice "codice diverso", non "calcolo
    diverso".
    """
    if not files:
        return None
    digest = hashlib.sha256()
    for file in files:
        name = file.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(name + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest()


# I nomi di prima, per chi li importava (scripts/record_training.py fino a questo cambiamento).
_code_files = code_files
_code_sha256 = code_sha256


def _few_names(names: list[str], limit: int = 3) -> str:
    """I primi nomi di un elenco per una nota: "a, b, c e altri 4"."""
    more = f" e altri {len(names) - limit}" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def format_int(value: Any) -> str:
    """Intero con il punto delle migliaia, all'italiana (209.729)."""
    return f"{value:,}".replace(",", ".") if isinstance(value, int) else str(value)


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _value(value: Any, fmt: str) -> str:
    if not _is_number(value):
        return "—"
    if fmt == "pct":
        return _pct(value)
    if fmt == "int":
        return format_int(int(value))
    return fmt.format(value)


def _delta(delta: float | None, fmt: str) -> str:
    if delta is None:
        return "—"
    if delta == 0:
        return "0"
    if fmt == "pct":
        return f"{100 * delta:+.2f} punti"
    if fmt == "int":
        return f"{int(delta):+,}".replace(",", ".")
    return ("{:+" + fmt[2:]).format(delta)


def _indent(text: str, width: int = 4) -> list[str]:
    return [" " * width + line for line in text.splitlines()] or [" " * width]
