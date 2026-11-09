"""Run esportabili: i pesi di un training in una cartella che il Laboratorio pesi della web app carica.

Il flusso pensato per lo studente:

    1. migliori il GPT (iperparametri, pesi o architettura, anche con un assistente come Claude);
    2. lo riaddestri (``train.py`` o ``scripts/record_training.py``, che registra anche la storia);
    3. lo esporti con ``scripts/export_run.py`` in ``runs/<nome>/``;
    4. nel tab "Laboratorio pesi" carichi la cartella e lo confronti con il GPT del corso.

Per ESPORTARE non serve la classe del modello: bastano i tensori dello state_dict. Così si esporta
anche un checkpoint con l'architettura modificata (nomi o forme diversi), che il codice attuale non
saprebbe più ricostruire. Per MISURARLO (``gpt/benchmark.py``) serve invece una classe che lo esegua.

Cartella di un run::

    runs/<nome>/
      run.json                     descrizione del run (schema sotto)
      weights.json                 manifest dei pesi: formato di webapp/data/weights.json (gpt/export.py)
      weights.f32.bin              tutti i tensori in float32 little-endian, nell'ordine dello state_dict
      benchmark.json               (facoltativo) metriche di gpt/benchmark.py
      training/trace.json          (facoltativo) storia del training di scripts/record_training.py
      training/snap_XXXXX.f16.bin  (facoltativo) gli snapshot citati da trace.json

``weights.json`` e ``weights.f32.bin`` li scrive ``gpt.export.export_weights``, la stessa funzione dei
dati della web app: per il modello del corso i file sono identici byte per byte. Con un'architettura
diversa cambiano solo nomi e forme in ``params``; ``config`` e ``vocab`` possono valere ``null``.

Schema di ``run.json`` (testi in italiano; un campo facoltativo può valere ``null``)::

    {
      "format_version": 1,
      "name": "GPT con GELU",                  # nome mostrato nel Laboratorio (al massimo 200 caratteri)
      "description": "...",                     # al massimo 2000 caratteri
      "created": "2026-11-09T10:15:00+00:00",   # data dell'export, ISO 8601 in UTC
      "source": "checkpoints/gpt.pt",           # da dove vengono i pesi (relativo al progetto)
      "config": {...} | null,                   # iperparametri del checkpoint come sono (anche chiavi nuove)
      "total_params": 209729,                   # valori esportati (somma dei numel)
      "tensor_count": 94,
      "modules": {...},                         # albero dei moduli, vedi sotto
      "buffers": ["..."] | null,                # tensori che sono buffer non appresi; null = non si sa
      "skipped": [{"name": "...", "reason": "..."}],   # voci dello state_dict NON esportate
      "files": {"weights": "weights.json", "benchmark": "benchmark.json" | null,
                "trace": "training/trace.json" | null},
      "training": {"steps", "snapshots": [0, 50, ...], "eval_step", "train_loss", "val_loss",
                   "device", "duration_s"} | null,      # riassunto di trace.json, se copiato
      "engine_compatible": false,               # c'è solo se dichiarato (vedi sotto)
      "notes": ["..."]                          # note dell'autore, poi quelle di trace.json con il prefisso
    }                                           # "storia del training: ", poi gli avvisi dell'export

Albero dei moduli, dai nomi puntati dello state_dict (come ``moduleTree.js`` nel browser)::

    {"path": "", "key": "", "kind": "module", "class": "GPTLanguageModel" | null,
     "params": 209729, "tensors": 94, "children": [
       {"path": "blocks", "key": "blocks", "kind": "module", "class": "nn.Sequential", ...,
        "children": [...]},
       {"path": "lm_head.bias", "key": "bias", "kind": "param", "shape": [65], "numel": 65,
        "dtype": "float32", "source_dtype": "float32"}
     ]}

``params`` conta i valori sotto il nodo, ``tensors`` i tensori. ``kind`` di una foglia è
``"param"`` oppure ``"buffer"`` (un buffer persistente, es. le statistiche di una BatchNorm).
``class`` e la distinzione tra parametri e buffer ci sono solo se si conosce il modulo: uno
state_dict da solo non sa di che classe erano i moduli (``class`` vale null, ``buffers`` null).

``engine_compatible``: il Laboratorio decide da solo se il motore JavaScript sa eseguire i pesi
(stessi nomi e forme del GPT del corso). Forme uguali però non garantiscono lo stesso calcolo (es.
GELU al posto di ReLU): in quel caso si esporta con ``engine_compatible=False``
(``scripts/export_run.py --not-engine-compatible``) e il browser mostra i pesi senza eseguirli.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pickle
import re
import shutil
import types
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from gpt.export import (
    PROJECT_ROOT,
    TRAINING_DIR,
    TRAINING_TRACE,
    WEIGHTS_BINARY,
    WEIGHTS_FORMAT_VERSION,
    WEIGHTS_MANIFEST,
    export_weights,
    parameter_layout,
    parameters_to_bytes,
)
from gpt.recording import TRACE_FORMAT_VERSION

RUN_FORMAT_VERSION = 1
RUN_FILE = "run.json"
BENCHMARK_FILE = "benchmark.json"
RUNS_DIR = PROJECT_ROOT / "runs"

# Limiti dei testi: gli stessi che il Laboratorio applica leggendo run.json (shared/weights/formats.js).
MAX_NAME_CHARS = 200
MAX_TEXT_CHARS = 2000
# Nomi dei tensori: stesso limite del browser (shared/weights/validate.js, LIMITS.maxNameLength).
MAX_TENSOR_NAME_CHARS = 512
# Oltre questo numero di valori il browser rifiuta i pesi (LIMITS.maxParams): lo diciamo subito.
BROWSER_MAX_VALUES = 256 * 1024 * 1024
# Prefissi aggiunti dagli involucri di PyTorch: torch.compile (_orig_mod.) e DataParallel/DDP (module.).
_WRAPPER_PREFIXES = ("_orig_mod.", "module.")
# Contenitori dello state_dict nei formati più comuni: gpt/training.py, nanoGPT, esempi di PyTorch.
_STATE_DICT_KEYS = ("state_dict", "model", "model_state_dict")
_CONFIG_KEYS = ("config", "model_args")
_VOCAB_KEYS = ("chars", "vocab")
# Le chiavi della config che il browser vuole intere positive (shared/weights/validate.js, cleanConfig).
_POSITIVE_INT_KEYS = ("vocab_size", "block_size", "n_embd", "n_head", "n_layer")
_SNAPSHOT_NAME = re.compile(r"^snap_\d{5,}\.f16\.bin$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# Un surrogato UTF-16 spaiato ("\ud800" scritto in un JSON, o un byte non UTF-8 di un argomento della riga
# di comando) è un str valido ma non si scrive in UTF-8: write_text fallirebbe alla fine dell'export, con
# pesi e storia già scritti e run.json vuoto. Nei testi che finiscono in run.json diventa "�" (U+FFFD),
# come fa toWellFormed() nel browser (js/tabs/lab/runs/describe.js).
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
# Le note di trace.json (gpt/recording.py) arrivano in run.json con questo prefisso, così nella scheda
# del run si capisce da dove vengono; il Laboratorio usa lo stesso (js/tabs/lab/runs/describe.js).
TRAINING_NOTE_PREFIX = "storia del training: "
# Il registratore scrive al più una nota per argomento: di più vuol dire un trace scritto a mano.
MAX_TRAINING_NOTES = 20


# --------------------------------------------------------------------------- tensori di un run


@dataclass(frozen=True)
class RunWeights:
    """I tensori di un run pronti da scrivere, più quello che si sa di loro.

    ``tensors`` sono float32 contigui su CPU, nell'ordine dello state_dict: è l'ordine del file
    binario, lo stesso di ``model.named_parameters()`` per il GPT del corso.
    """

    tensors: dict[str, torch.Tensor]
    source: str  # da dove vengono (percorso relativo al progetto o descrizione)
    config: dict[str, Any] | None = None
    vocab: list[str] | None = None
    trained: bool = True  # False = inizializzazione casuale (come "trained" di weights.json)
    source_dtypes: dict[str, str] = field(default_factory=dict)  # nome → dtype originale ("bfloat16"…)
    classes: dict[str, str] | None = None  # percorso del modulo → classe; None = non si sa
    buffers: frozenset[str] | None = None  # tensori che sono buffer; None = non si sa
    skipped: dict[str, str] = field(default_factory=dict)  # voci non esportate → motivo
    notes: tuple[str, ...] = ()  # avvisi raccolti leggendo i pesi

    @property
    def total_values(self) -> int:
        return sum(t.numel() for t in self.tensors.values())

    def sha256(self) -> str:
        """Impronta dei pesi: sha256 dei byte di weights.f32.bin (abbina un benchmark ai suoi pesi)."""
        return hashlib.sha256(parameters_to_bytes(_TensorModule(self), "<f4")).hexdigest()

    def with_module_info(self, model: nn.Module, note: str | None = None) -> RunWeights:
        """Aggiunge classi dei moduli e buffer presi da ``model``, che deve avere gli stessi tensori."""
        if not set(self.tensors) <= set(model.state_dict()):
            raise ValueError("il modello non ha gli stessi tensori del run: classi e buffer non si abbinano")
        # Solo i buffer esportati: un contatore intero (es. num_batches_tracked) è stato saltato.
        persistent = {name for name, _ in model.named_buffers()} & set(self.tensors)
        notes = self.notes + ((note,) if note else ())
        return dataclasses.replace(self, classes=module_classes(model), buffers=frozenset(persistent),
                                   notes=notes)


def module_classes(model: nn.Module) -> dict[str, str]:
    """Percorso di ogni sotto-modulo → classe, scritta come nel codice (``nn.Linear``, ``Block``)."""
    classes = {}
    for path, module in model.named_modules():
        cls = type(module)
        classes[path] = f"nn.{cls.__name__}" if cls.__module__.startswith("torch.nn") else cls.__name__
    return classes


def weights_from_model(
    model: nn.Module,
    *,
    vocab: list[str] | None = None,
    source: str = "modello in memoria",
    trained: bool = True,
) -> RunWeights:
    """Run dai pesi di un modello in memoria: si conoscono anche le classi e i buffer."""
    to_dict = getattr(getattr(model, "config", None), "to_dict", None)
    config = to_dict() if callable(to_dict) else None
    weights = weights_from_state_dict(model.state_dict(), config=config, vocab=vocab, source=source,
                                      trained=trained)
    return weights.with_module_info(model)


def weights_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    vocab: Iterable[str] | None = None,
    source: str = "state_dict",
    trained: bool = True,
    notes: Iterable[str] = (),
) -> RunWeights:
    """Run da uno state_dict qualunque: nessuna classe necessaria, nomi e forme restano quelli dati.

    Si esportano i tensori in virgola mobile (convertiti in float32); interi, booleani e oggetti
    che non sono tensori si saltano, con il motivo in ``skipped``.
    """
    notes = list(notes)
    renamed, prefix_note = _canonical_names(list(state_dict))
    if prefix_note:
        notes.append(prefix_note)
    tensors: dict[str, torch.Tensor] = {}
    source_dtypes: dict[str, str] = {}
    skipped: dict[str, str] = {}
    converted: Counter[str] = Counter()
    for original, value in state_dict.items():
        name = renamed[original]
        _check_tensor_name(name)
        reason = _not_exportable(value)
        if reason:
            skipped[name] = reason
            continue
        dtype = str(value.dtype).removeprefix("torch.")
        if value.dtype != torch.float32:
            converted[dtype] += 1
        tensors[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        source_dtypes[name] = dtype
    if not tensors:
        raise ValueError("nessun tensore in virgola mobile da esportare: è lo state_dict di un modello?")
    _check_tree_names(list(tensors))

    for dtype, count in sorted(converted.items()):
        notes.append(f"{count} tensori in {dtype} convertiti in float32 (il formato dei pesi della web app)")
    if skipped:
        notes.append(f"{len(skipped)} voci dello state_dict non esportate: non sono pesi in virgola mobile")
    non_finite = sum(int((~torch.isfinite(t)).sum()) for t in tensors.values())
    if non_finite:
        notes.append(f"{non_finite} valori non finiti (NaN o infinito): il training è divergente?")
    total = sum(t.numel() for t in tensors.values())
    if total > BROWSER_MAX_VALUES:
        notes.append(f"{total} valori: oltre il limite del browser ({BROWSER_MAX_VALUES}), il Laboratorio "
                     "non li carica")

    clean_config = _clean_config(config, notes)
    return RunWeights(
        tensors=tensors,
        source=source,
        config=clean_config,
        vocab=_clean_vocab(vocab, clean_config),
        trained=trained,
        source_dtypes=source_dtypes,
        skipped=skipped,
        notes=tuple(notes),
    )


def read_checkpoint(path: Path) -> RunWeights:
    """Legge un checkpoint SENZA costruire il modello: funziona anche con un'architettura modificata.

    Formati accettati: quello di ``gpt/training.py`` (``{"format_version", "config", "chars",
    "state_dict"}``), uno state_dict "nudo" e i dizionari con lo state_dict in ``"model"`` (nanoGPT)
    o ``"model_state_dict"``. ``weights_only=True``: un checkpoint è input NON fidato e un pickle
    qualsiasi potrebbe eseguire codice; così torch accetta solo tensori e tipi semplici.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} non trovato")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        detail = str(error).strip().splitlines()[0][:200] if str(error).strip() else type(error).__name__
        raise ValueError(
            f"{path.name}: non si legge come checkpoint di PyTorch con weights_only=True. È danneggiato "
            "oppure contiene oggetti Python che non sono tensori o tipi semplici (per sicurezza non si "
            f"caricano). Dettaglio: {detail}"
        ) from None
    state_dict, config, vocab, notes = _unpack_checkpoint(payload, path.name)
    return weights_from_state_dict(state_dict, config=config, vocab=vocab, source=display_path(path),
                                   notes=notes)


def read_run_folder(path: Path) -> RunWeights:
    """Pesi di una cartella con ``weights.json`` e il binario float32 (un run esportato o ``webapp/data``).

    I file si controllano come input non fidato: versione, tipo, layout (offset nei limiti, niente
    sovrapposizioni) e dimensione del binario, che deve stare nella stessa cartella.
    """
    path = Path(path)
    manifest_path = path / WEIGHTS_MANIFEST
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != WEIGHTS_FORMAT_VERSION:
        raise ValueError(f"{manifest_path}: format_version {manifest.get('format_version')!r} non supportato")
    if manifest.get("dtype") != "float32" or manifest.get("byte_order") != "little":
        raise ValueError(f"{manifest_path}: pesi attesi in float32 little-endian")
    binary = _file_beside(path, manifest.get("file"), manifest_path)
    layout, total = _check_layout(manifest.get("params"), manifest.get("total_params"), str(manifest_path))
    size = binary.stat().st_size
    if size != 4 * total:
        raise ValueError(f"{binary}: attesi {4 * total} byte ({total} float32), trovati {size}")
    flat = np.fromfile(binary, dtype="<f4")
    tensors = {p["name"]: _tensor_from_flat(flat, p) for p in layout}
    config = manifest.get("config")
    if config is None:
        config = (read_run_info(path) or {}).get("config")
    return weights_from_state_dict(tensors, config=config, vocab=manifest.get("vocab"),
                                   source=display_path(path), trained=manifest.get("trained") is not False)


def read_run_info(path: Path) -> dict[str, Any] | None:
    """``run.json`` di una cartella, se c'è (None altrimenti)."""
    file = Path(path) / RUN_FILE
    return _read_json(file) if file.is_file() else None


# --------------------------------------------------------------------------- storia del training


@dataclass(frozen=True)
class TrainingHistory:
    """``trace.json`` già controllato, con gli snapshot che hanno davvero il loro file accanto."""

    directory: Path
    trace: dict[str, Any]
    layout: list[dict[str, Any]]  # trace["params"]: layout dei file snap_*.f16.bin
    total_params: int
    snapshots: list[dict[str, Any]]  # voci di trace["snapshots"] con il file presente, in ordine di step
    missing: list[int]  # step citati in trace.json ma senza file

    @property
    def max_iters(self) -> int | None:
        value = (self.trace.get("train_config") or {}).get("max_iters")
        return value if _is_int(value) else None

    def summary(self) -> dict[str, Any]:
        """Il riassunto che finisce in run.json: pochi numeri per la scheda del run."""
        evals = self.trace.get("evals") if isinstance(self.trace.get("evals"), dict) else {}

        def last(key: str) -> Any:
            values = evals.get(key)
            return values[-1] if isinstance(values, list) and values and _is_number(values[-1]) else None

        device, duration = self.trace.get("device"), self.trace.get("duration_s")
        return {
            "steps": self.max_iters,
            "snapshots": [s["step"] for s in self.snapshots],
            "eval_step": last("steps"),
            "train_loss": last("train"),
            "val_loss": last("val"),
            "device": _one_line(device) if isinstance(device, str) else None,  # testo del file: "cpu", "cuda"…
            "duration_s": duration if _is_number(duration) else None,
        }


def read_training_history(directory: Path) -> TrainingHistory:
    """Legge ``trace.json`` (scripts/record_training.py) e controlla gli snapshot float16 accanto.

    I nomi dei file degli snapshot vengono dal JSON: devono essere nomi semplici
    ``snap_XXXXX.f16.bin``, così copiarli non può toccare file fuori dalla cartella. Uno snapshot
    senza file viene solo segnalato (il browser ne mostra comunque le metriche); uno della dimensione
    sbagliata è un errore.
    """
    directory = Path(directory)
    trace_path = directory / TRAINING_TRACE
    trace = _read_json(trace_path)
    if trace.get("format_version") != TRACE_FORMAT_VERSION:
        raise ValueError(f"{trace_path}: format_version {trace.get('format_version')!r} non supportato")
    if trace.get("snapshot_dtype") != "float16" or trace.get("byte_order", "little") != "little":
        raise ValueError(f"{trace_path}: snapshot attesi in float16 little-endian")
    layout, total = _check_layout(trace.get("params"), trace.get("total_params"), str(trace_path))
    entries = trace.get("snapshots")
    if not isinstance(entries, list):
        raise ValueError(f"{trace_path}: snapshots deve essere una lista")
    present, missing, steps = [], [], set()
    for i, entry in enumerate(entries):
        where = f"{trace_path} snapshots[{i}]"
        if not isinstance(entry, dict) or not _is_int(entry.get("step")) or entry["step"] < 0:
            raise ValueError(f"{where}: serve un oggetto con uno step intero ≥ 0")
        if entry["step"] in steps:
            raise ValueError(f"{where}: lo step {entry['step']} compare due volte")
        steps.add(entry["step"])
        name = entry.get("file")
        if not isinstance(name, str) or not _SNAPSHOT_NAME.match(name):
            raise ValueError(f"{where}: nome di file non ammesso {name!r} (atteso snap_XXXXX.f16.bin)")
        file = directory / name
        if not file.is_file():
            missing.append(entry["step"])
            continue
        size = file.stat().st_size
        if size != 2 * total:
            raise ValueError(f"{file}: attesi {2 * total} byte ({total} float16), trovati {size}")
        present.append(entry)
    present.sort(key=lambda s: s["step"])
    return TrainingHistory(directory, trace, layout, total, present, sorted(missing))


def snapshot_tensors(history: TrainingHistory, entry: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """I pesi di uno snapshot: float16 convertiti in float32 senza perdita, come fa il motore JS."""
    flat = np.fromfile(history.directory / entry["file"], dtype="<f2").astype(np.float32)
    return {p["name"]: _tensor_from_flat(flat, p) for p in history.layout}


def training_notes(trace: Mapping[str, Any]) -> list[str]:
    """Le note di ``trace.json`` pronte per ``run.json``: ``"storia del training: …"``, una per nota.

    Sono le spiegazioni di gpt/recording.py su che cosa si è adattato all'architettura (gruppi nuovi,
    pesi condivisi, attention saltata, valori non finiti). Il trace è un file letto dal disco: si tengono
    solo i testi, su una riga (``_one_line``: spazi al posto dei caratteri di controllo, "�" al posto dei
    surrogati spaiati, che run.json non potrebbe contenere), accorciati al limite che il browser legge
    (``MAX_TEXT_CHARS``) e al più ``MAX_TRAINING_NOTES`` (le altre si contano in una nota).
    """
    raw = trace.get("notes")
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    texts = [_one_line(item) for item in items if isinstance(item, str)]
    texts = [text for text in texts if text]
    notes = [_shorten(TRAINING_NOTE_PREFIX + text, MAX_TEXT_CHARS) for text in texts[:MAX_TRAINING_NOTES]]
    if len(texts) > MAX_TRAINING_NOTES:
        notes.append(f"{TRAINING_NOTE_PREFIX}altre {len(texts) - MAX_TRAINING_NOTES} note in "
                     f"{TRAINING_DIR}/{TRAINING_TRACE}")
    return notes


# --------------------------------------------------------------------------- export


def export_run(
    weights: RunWeights,
    out_dir: Path,
    *,
    name: str,
    description: str = "",
    notes: Iterable[str] = (),
    training_dir: Path | None = None,
    benchmark: Mapping[str, Any] | None = None,
    engine_compatible: bool | None = None,
    overwrite: bool = False,
    created: datetime | None = None,
) -> dict[str, Any]:
    """Scrive la cartella del run e restituisce il contenuto di ``run.json``.

    Args:
        training_dir: cartella con ``trace.json`` e gli snapshot (``scripts/record_training.py --out``),
            copiata in ``training/``. Le note del trace (``notes``: che cosa la registrazione ha adattato
            all'architettura) finiscono anche in ``run.json``, dopo quelle dell'autore, con il prefisso
            ``"storia del training: "`` (``training_notes``).
        benchmark: il risultato di ``gpt.benchmark`` (diventa ``benchmark.json``).
        engine_compatible: ``False`` se il calcolo è diverso da quello del GPT del corso anche con
            le stesse forme (es. GELU invece di ReLU); ``None`` (predefinito) lascia decidere al browser.
        overwrite: sostituisce i file di un run già presente in ``out_dir``. Si cancellano SOLO i file
            che l'export scrive: un file estraneo nella cartella resta dov'è (e viene segnalato).
    """
    out_dir = Path(out_dir)
    name = _check_text(name, "il nome", MAX_NAME_CHARS, required=True)
    description = _check_text(description, "la descrizione", MAX_TEXT_CHARS)
    author_notes = [_check_text(n, "una nota", MAX_TEXT_CHARS, required=True) for n in notes]
    # Prima tutto ciò che può fallire per colpa dei dati, poi si scrive: niente cartelle a metà.
    history = read_training_history(training_dir) if training_dir is not None else None
    benchmark_text = None
    if benchmark is not None:
        benchmark_text = json.dumps(benchmark, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    export_notes = [*weights.notes, *_benchmark_notes(benchmark, weights)]
    export_notes += _prepare_folder(out_dir, overwrite)

    module = _TensorModule(weights)
    if history is not None:
        export_notes += _copy_history(history, out_dir / TRAINING_DIR, module)
    # La stessa funzione dei dati della web app: weights.json punta a training/trace.json se c'è.
    vocab = types.SimpleNamespace(chars=weights.vocab)  # export_weights legge solo tokenizer.chars
    source = _well_formed(weights.source)  # il percorso del checkpoint: anche un nome di file non UTF-8
    export_weights(module, vocab, out_dir, source, weights.trained)
    if benchmark_text is not None:
        (out_dir / BENCHMARK_FILE).write_text(benchmark_text, encoding="utf-8")

    info: dict[str, Any] = {
        "format_version": RUN_FORMAT_VERSION,
        "name": name,
        "description": description,
        "created": (created or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "source": source,
        "config": weights.config,
        "total_params": weights.total_values,
        "tensor_count": len(weights.tensors),
        "modules": module_tree(weights),
        "buffers": sorted(weights.buffers) if weights.buffers is not None else None,
        "skipped": [{"name": n, "reason": r} for n, r in weights.skipped.items()],
        "files": {
            "weights": WEIGHTS_MANIFEST,
            "benchmark": BENCHMARK_FILE if benchmark is not None else None,
            "trace": f"{TRAINING_DIR}/{TRAINING_TRACE}" if history is not None else None,
        },
        "training": history.summary() if history is not None else None,
    }
    if engine_compatible is not None:
        info["engine_compatible"] = bool(engine_compatible)
    history_notes = training_notes(history.trace) if history is not None else []
    info["notes"] = author_notes + history_notes + export_notes
    (out_dir / RUN_FILE).write_text(readable_json(info) + "\n", encoding="utf-8")
    return info


def module_tree(weights: RunWeights) -> dict[str, Any]:
    """Albero dei moduli dai nomi puntati (schema nella docstring del modulo)."""
    classes = weights.classes or {}
    buffers = weights.buffers or frozenset()

    def module_node(path: str, key: str) -> dict[str, Any]:
        return {"path": path, "key": key, "kind": "module", "class": classes.get(path), "params": 0,
                "tensors": 0, "children": []}

    root = module_node("", "")
    modules = {"": root}
    for name, tensor in weights.tensors.items():
        parts = name.split(".")
        parent = root
        for depth in range(1, len(parts)):
            path = ".".join(parts[:depth])
            if path not in modules:
                modules[path] = module_node(path, parts[depth - 1])
                parent["children"].append(modules[path])
            parent = modules[path]
        parent["children"].append({
            "path": name,
            "key": parts[-1],
            "kind": "buffer" if name in buffers else "param",
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "dtype": "float32",
            "source_dtype": weights.source_dtypes.get(name, "float32"),
        })
    _count(root)
    return root


def readable_json(value: Any, level: int = 0) -> str:
    """JSON da leggere a occhio: un oggetto o una lista "piatti" (solo valori semplici) sta su una riga.

    Così in run.json ogni tensore dell'albero è una riga sola, con la forma ``[65, 64]`` accanto al
    nome, invece di una decina di righe con un numero per riga.
    """
    flat = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if not isinstance(value, (dict, list)) or not value or _is_flat(value):
        return flat
    pad, inner = "  " * level, "  " * (level + 1)
    if isinstance(value, dict):
        items = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {readable_json(v, level + 1)}"
                 for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    return "[\n" + ",\n".join(inner + readable_json(v, level + 1) for v in value) + f"\n{pad}]"


# --------------------------------------------------------------------------- percorsi e nomi


def display_path(path: Path) -> str:
    """Percorso da scrivere nei JSON: relativo al progetto se ci sta dentro.

    Fuori dal progetto si tengono solo le ultime due parti: nei file condivisi non deve finire la
    cartella personale di nessuno (es. /Users/<nome>/...).
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return "…/" + "/".join(resolved.parts[-2:])


def slugify(name: str) -> str:
    """Nome di cartella sicuro per un run, da minuscole senza accenti, cifre e trattini.

    Esempio: "GPT con GELU" → "gpt-con-gelu". Niente punti né barre: il nome non esce da ``runs/``.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:64].strip("-")
    if not slug:
        raise ValueError(f"dal nome {name!r} non si ricava un nome di cartella: usa lettere o cifre")
    return slug


# --------------------------------------------------------------------------- interno


class _TensorModule:
    """Un insieme di tensori con l'interfaccia che usano ``export_weights`` e ``parameter_layout``.

    Così i file del run li scrive il codice di ``gpt/export.py`` senza modificarlo e senza la classe
    del modello: quelle funzioni chiedono solo nomi, tensori, numero di parametri e config.
    """

    def __init__(self, weights: RunWeights) -> None:
        self._tensors = weights.tensors
        self.config = types.SimpleNamespace(to_dict=lambda: weights.config)

    def named_parameters(self):
        return iter(self._tensors.items())

    def parameters(self):
        return iter(self._tensors.values())

    def num_parameters(self) -> int:
        return sum(t.numel() for t in self._tensors.values())


def _unpack_checkpoint(payload: Any, label: str) -> tuple[Mapping[str, Any], Any, Any, list[str]]:
    """Separa state_dict, config e vocabolario dei formati di checkpoint più comuni."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}: il checkpoint non è un dizionario con uno state_dict")
    notes: list[str] = []
    container = next((k for k in _STATE_DICT_KEYS if isinstance(payload.get(k), Mapping)), None)
    if container is None:
        if not any(isinstance(v, torch.Tensor) for v in payload.values()):
            keys = ", ".join(map(str, list(payload)[:8]))
            raise ValueError(f"{label}: nessuno state_dict trovato (chiavi: {keys})")
        return payload, None, None, notes  # uno state_dict "nudo"
    version = payload.get("format_version")
    if version is not None and version != 1:
        notes.append(f"checkpoint con format_version {version!r} (gpt/training.py scrive 1): letti solo "
                     "state_dict, config e vocabolario")
    config = next((payload[k] for k in _CONFIG_KEYS if isinstance(payload.get(k), Mapping)), None)
    vocab = next((payload[k] for k in _VOCAB_KEYS if payload.get(k) is not None), None)
    return payload[container], config, vocab, notes


def _canonical_names(names: list[Any]) -> tuple[dict[Any, Any], str | None]:
    """Toglie ``_orig_mod.``/``module.`` (torch.compile, DataParallel), se non crea nomi doppi."""
    renamed, prefixes = {}, set()
    for name in names:
        new = name
        while isinstance(new, str) and new.startswith(_WRAPPER_PREFIXES):
            prefix = next(p for p in _WRAPPER_PREFIXES if new.startswith(p))
            prefixes.add(prefix.rstrip("."))
            new = new[len(prefix):]
        renamed[name] = new
    if not prefixes:
        return renamed, None
    if len(set(renamed.values())) != len(renamed):
        return ({n: n for n in names},
                "alcuni nomi iniziano con module. o _orig_mod., ma toglierli creerebbe nomi doppi: lasciati")
    joined = " e ".join(sorted(prefixes))
    return renamed, f"tolti i prefissi {joined} dai nomi (aggiunti da torch.compile o DataParallel)"


def _not_exportable(value: Any) -> str | None:
    if not isinstance(value, torch.Tensor):
        return f"non è un tensore ({type(value).__name__})"
    if value.layout != torch.strided:
        return "tensore sparso"
    if value.is_complex() or not value.is_floating_point():
        dtype = str(value.dtype).removeprefix("torch.")
        return f"tensore {dtype}: non è un peso in virgola mobile (es. un contatore)"
    return None


def _check_tensor_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"nome di tensore non valido: {name!r}")
    if len(name) > MAX_TENSOR_NAME_CHARS or _CONTROL_CHARS.search(name) or "" in name.split("."):
        raise ValueError(f"nome di tensore non valido: {name[:80]!r} (troppo lungo, caratteri di controllo "
                         "o punti vuoti)")


def _check_tree_names(names: list[str]) -> None:
    """Un tensore non può essere anche il modulo di un altro ("a.b" e "a.b.c"): l'albero non avrebbe senso."""
    known = set(names)
    for name in names:
        parts = name.split(".")
        for depth in range(1, len(parts)):
            prefix = ".".join(parts[:depth])
            if prefix in known:
                raise ValueError(f"il tensore {prefix!r} è anche il modulo che contiene {name!r}")


def _clean_config(config: Mapping[str, Any] | None, notes: list[str]) -> dict[str, Any] | None:
    """Config scrivibile in JSON: valori semplici e finiti; gli altri si saltano (con una nota)."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("la config deve essere un dizionario")
    clean, dropped = {}, []
    for key, value in config.items():
        if isinstance(key, str) and _json_safe(value):
            clean[key] = value
        else:
            dropped.append(str(key))
    if dropped:
        notes.append(f"config: saltate le chiavi non scrivibili in JSON ({', '.join(dropped)})")
    for key in _POSITIVE_INT_KEYS:
        if key in clean and not (_is_int(clean[key]) and clean[key] > 0):
            raise ValueError(f"config.{key} deve essere un intero positivo, trovato {clean[key]!r}")
    return clean


def _clean_vocab(vocab: Iterable[str] | None, config: Mapping[str, Any] | None) -> list[str] | None:
    if vocab is None:
        return None
    # Una stringa è iterabile, ma "abc" non è un vocabolario.
    vocab = None if isinstance(vocab, str) else list(vocab)
    if vocab is None or not all(isinstance(t, str) for t in vocab):
        raise ValueError("il vocabolario deve essere una lista di stringhe (token id → testo)")
    if len(set(vocab)) != len(vocab):
        raise ValueError("il vocabolario contiene token duplicati")
    expected = (config or {}).get("vocab_size")
    if expected is not None and expected != len(vocab):
        raise ValueError(f"il vocabolario ha {len(vocab)} voci ma config.vocab_size è {expected}")
    return vocab


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


def _is_flat(value: dict | list) -> bool:
    items = value.values() if isinstance(value, dict) else value
    simple = (str, int, float, bool, type(None))
    return all(isinstance(v, simple) or (isinstance(v, list) and all(isinstance(x, simple) for x in v))
               for v in items)


def _check_text(value: Any, what: str, max_chars: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{what} deve essere un testo")
    value = _well_formed(value.strip())  # un byte non UTF-8 della riga di comando non blocca run.json
    if required and not value:
        raise ValueError(f"{what} non può essere vuoto")
    if len(value) > max_chars:
        raise ValueError(f"{what} è lungo {len(value)} caratteri: il massimo è {max_chars}")
    return value


def _shorten(text: str, max_chars: int) -> str:
    """``text`` di al più ``max_chars`` caratteri, con "…" se è stato tagliato (testi letti da file)."""
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _well_formed(text: str) -> str:
    """``text`` scrivibile in UTF-8: "�" al posto dei surrogati spaiati (vedi ``_LONE_SURROGATE``)."""
    return _LONE_SURROGATE.sub("\ufffd", text)


def _one_line(text: str) -> str:
    """Un testo letto da un file su una riga: spazi al posto di caratteri di controllo e a capo, scrivibile
    in UTF-8. Le stesse regole di ``oneLine`` nel browser: una nota copiata in run.json si riconosce."""
    return " ".join(_well_formed(_CONTROL_CHARS.sub(" ", text)).split())


def _prepare_folder(out_dir: Path, overwrite: bool) -> list[str]:
    """Crea la cartella o, con ``overwrite``, toglie solo i file scritti da un export precedente."""
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return []
    if not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} esiste e non è una cartella")
    if not any(out_dir.iterdir()):
        return []
    if not overwrite:
        raise FileExistsError(
            f"la cartella {out_dir} contiene già dei file: scegli un altro nome oppure sostituisci il run "
            "(--force; si cancellano solo i file del run)"
        )
    training = out_dir / TRAINING_DIR
    if training.is_symlink():
        raise ValueError(f"{training} è un collegamento simbolico: per sicurezza non si cancella niente lì")
    # Solo i nomi fissi che l'export scrive: mai un file indicato da un JSON della cartella.
    managed = [out_dir / n for n in (RUN_FILE, WEIGHTS_MANIFEST, WEIGHTS_BINARY, BENCHMARK_FILE)]
    if training.is_dir():
        managed += [training / TRAINING_TRACE, *training.glob("snap_*.f16.bin")]
    for file in managed:
        if file.is_file() or file.is_symlink():
            file.unlink()
    if training.is_dir() and not any(training.iterdir()):
        training.rmdir()
    leftovers = sorted(p.name for p in out_dir.iterdir())
    if leftovers:
        return [f"nella cartella restano altri file non del run: {', '.join(leftovers[:5])}"]
    return []


def _benchmark_notes(benchmark: Mapping[str, Any] | None, weights: RunWeights) -> list[str]:
    """Un benchmark.json misurato su altri pesi sarebbe fuorviante: lo si segnala."""
    declared = ((benchmark or {}).get("model") or {}).get("weights_sha256")
    if declared and declared != weights.sha256():
        return ["benchmark.json è stato misurato su pesi diversi da quelli esportati (altro weights_sha256)"]
    return []


def _copy_history(history: TrainingHistory, dest: Path, module: _TensorModule) -> list[str]:
    """Copia trace.json e gli snapshot presenti; controlla che la storia sia di QUESTI pesi."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(history.directory / TRAINING_TRACE, dest / TRAINING_TRACE)
    for entry in history.snapshots:
        shutil.copyfile(history.directory / entry["file"], dest / entry["file"])
    notes = []
    if history.missing:
        steps = ", ".join(map(str, history.missing))
        notes.append(f"trace.json cita snapshot senza file (step {steps}): di questi restano le metriche")
    if history.layout != parameter_layout(module):
        notes.append("gli snapshot del training hanno tensori diversi dai pesi esportati: il Laboratorio li "
                     "mostra a parte")
        return notes
    # Lo snapshot finale è la copia float16 dei pesi dopo l'ultimo step (gpt/recording.py).
    final = history.snapshots[-1] if history.snapshots else None
    is_final = final is not None and final["step"] == history.max_iters
    if is_final and (history.directory / final["file"]).read_bytes() != parameters_to_bytes(module, "<f2"):
        notes.append(f"l'ultimo snapshot (step {final['step']}) non coincide con i pesi esportati: "
                     "la storia viene da un altro training?")
    return notes


def _check_layout(params: Any, total: Any, where: str) -> tuple[list[dict[str, Any]], int]:
    """Layout ``[{name, shape, offset, numel}]`` (offset e numel in numero di elementi) controllato."""
    if not isinstance(params, list) or not params:
        raise ValueError(f"{where}: manca la lista dei parametri (params)")
    if not (_is_int(total) and total >= 0):
        raise ValueError(f"{where}: total_params non valido ({total!r})")
    layout, seen = [], set()
    for i, p in enumerate(params):
        if not isinstance(p, dict) or not isinstance(p.get("name"), str):
            raise ValueError(f"{where} params[{i}]: serve un oggetto {{name, shape, offset, numel}}")
        name, shape, offset = p["name"], p.get("shape"), p.get("offset")
        _check_tensor_name(name)
        if name in seen:
            raise ValueError(f"{where}: il tensore {name} compare due volte")
        seen.add(name)
        if not isinstance(shape, list) or not all(_is_int(d) and d >= 0 for d in shape):
            raise ValueError(f"{where}: forma di {name} non valida ({shape!r})")
        numel = math.prod(shape)
        if p.get("numel", numel) != numel or not (_is_int(offset) and offset >= 0):
            raise ValueError(f"{where}: numel o offset di {name} non validi")
        layout.append({"name": name, "shape": list(shape), "offset": offset, "numel": numel})
    end = 0
    for p in sorted(layout, key=lambda q: (q["offset"], q["numel"])):
        if p["offset"] < end:
            raise ValueError(f"{where}: il tensore {p['name']} si sovrappone al precedente")
        end = p["offset"] + p["numel"]
    if end > total or sum(p["numel"] for p in layout) != total:
        raise ValueError(f"{where}: i tensori non coprono esattamente i {total} valori del file")
    _check_tree_names([p["name"] for p in layout])
    return layout, total


def _tensor_from_flat(flat: np.ndarray, p: Mapping[str, Any]) -> torch.Tensor:
    values = flat[p["offset"] : p["offset"] + p["numel"]].reshape(p["shape"])
    return torch.from_numpy(values.copy())  # copia: il tensore non deve dipendere dal buffer del file


def _file_beside(folder: Path, name: Any, where: Path) -> Path:
    """Un file indicato da un JSON: solo un nome semplice nella stessa cartella (niente "..", né assoluti)."""
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"{where}: campo 'file' non ammesso ({name!r}): serve il nome di un file accanto")
    file = folder / name
    if not file.is_file():
        raise FileNotFoundError(f"{where}: manca il file dei pesi {name}")
    return file


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} non trovato") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: JSON non valido ({error})") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: atteso un oggetto JSON")
    return data


def _count(node: dict[str, Any]) -> tuple[int, int]:
    """Riempie ``params`` e ``tensors`` dei moduli (somma dei figli); restituisce i totali del nodo."""
    if node["kind"] != "module":
        return node["numel"], 1
    for child in node["children"]:
        params, tensors = _count(child)
        node["params"] += params
        node["tensors"] += tensors
    return node["params"], node["tensors"]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
