"""Registra un training (loss, metriche, snapshot dei pesi) per i grafici della web app e del Laboratorio.

    python scripts/record_training.py              # il GPT del corso, come train.py: 5000 step, 3-4 minuti
    python scripts/record_training.py --device cpu
    # un esperimento (mai in webapp/data, che contiene il GPT del corso):
    python scripts/record_training.py --n-layer 6 --max-iters 2000 --threads 1 \\
        --checkpoint checkpoints/sei-blocchi/gpt.pt --out checkpoints/sei-blocchi/training --no-export

Produce, con le opzioni predefinite (il training di riferimento del corso):
    checkpoints/gpt.pt                          il modello addestrato (come train.py)
    webapp/data/training/trace.json             loss per step, metriche, snapshot (vedi gpt/recording.py)
    webapp/data/training/snap_XXXXX.f16.bin     i pesi completi a step scelti, in float16
e poi riesporta webapp/data (model.json, weights.json, weights.f32.bin) dal nuovo checkpoint.

Architettura modificata. Il modello è gpt.model.GPTLanguageModel del codice ATTUALE: per provare
un'idea cambia i file in gpt/ (attivazione, norme, attention, moduli nuovi, anche con Claude) e
rilancia questo script. La registrazione funziona con qualunque architettura (gpt/recording.py):
moduli dai nomi nuovi vanno in gruppi nuovi e un'attention che non espone i pesi salta solo la sua
metrica, con una nota nel trace. Le opzioni del modello nascono dai campi di GPTConfig: un campo che
aggiungi (es. ``ffn_mult: int = 4`` in gpt/config.py) diventa da solo un'opzione (``--ffn-mult 3``) e
finisce nella config del checkpoint e del trace; quelle del training vengono da TrainConfig. Poi

    python scripts/export_run.py CHECKPOINT --name "..." --trace CARTELLA --benchmark

porta il run nel Laboratorio pesi, misurato con lo stesso codice (lo script lo ricorda alla fine).
Una classe tua fuori da gpt/? Usa gpt.recording.TrainingRecorder come callback di gpt.training.train
(vedi la sua docstring): funziona con ogni modello con l'interfaccia ``logits, loss = model(idx, targets)``.

Iperparametri e seme predefiniti sono quelli di train.py, e il recorder non tocca il generatore
casuale globale: i batch, e quindi le loss, sono gli stessi del training di riferimento. Una storia
che NON è quella di riferimento (pesi, pesi condivisi, config, codice dei moduli o training diversi)
non può finire in webapp/data per sbaglio: lo script chiede --out, --checkpoint e --no-export prima
di cominciare, e controlla prima del training anche che le uscite si possano scrivere.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shlex
import sys
import time
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.nn as nn

# _code_files e _code_sha256 sono l'impronta del codice dei moduli che gpt/benchmark.py scrive in
# benchmark.json: usare le stesse funzioni garantisce che il confronto con builtin.json sia alla pari.
from gpt.benchmark import BENCHMARK_DATA_DIR, BUILTIN_FILE, _code_files, _code_sha256
from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, load_text, train_val_split
from gpt.device import resolve_device
from gpt.export import PROJECT_ROOT, TRAINING_DIR, WEBAPP_DATA_DIR, WEIGHTS_MANIFEST, export
from gpt.model import GPTLanguageModel
from gpt.recording import TRACE_FILE, RecorderConfig, TrainingRecorder, shared_parameters, snapshot_state
from gpt.training import save_checkpoint, train

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "gpt.pt"
DEFAULT_OUT = WEBAPP_DATA_DIR / TRAINING_DIR
# Prefissi delle destinazioni di argparse: le opzioni di GPTConfig e di TrainConfig restano separate.
MODEL, TRAIN = "model__", "train__"
_OPTION_TYPES = (bool, int, float, str)  # i campi con un default di questi tipi diventano opzioni
_TYPE_NAMES = {int: "un numero intero", float: "un numero", str: "un testo"}
# I messaggi di argparse che si vedono più spesso, in italiano (gli altri restano com'erano).
_ARGPARSE_ITALIAN = (
    ("unrecognized arguments: ", "opzioni sconosciute: "),
    ("the following arguments are required: ", "mancano le opzioni: "),
    ("expected one argument", "serve un valore"),
    ("ambiguous option: ", "opzione ambigua: "),
    (" could match ", " potrebbe essere "),
    ("ignored explicit argument ", "valore non previsto "),
    ("argument ", "opzione "),
)


# --------------------------------------------------------------------------- opzioni


class _ItalianHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, "uso: " if prefix is None else prefix)


class _ItalianParser(argparse.ArgumentParser):
    """argparse con gli errori in italiano: lo studente legge "Errore: …", non "error: argument …"."""

    def error(self, message: str) -> NoReturn:
        for english, italian in _ARGPARSE_ITALIAN:
            message = message.replace(english, italian)
        self.print_usage(sys.stderr)
        self.exit(2, f"Errore: {message} (l'elenco delle opzioni: --help)\n")


def add_dataclass_options(group: Any, cls: type, prefix: str, skip: Collection[str] = ()) -> None:
    """Un'opzione per ogni campo semplice della dataclass: ``n_layer`` → ``--n-layer``.

    Il default resta quello della dataclass (qui ``None`` vuol dire "non cambiato"); i campi di altri
    tipi (tuple, oggetti) si cambiano nel codice. Va chiamata DOPO le opzioni fisse dello script: così un
    campo che si chiama come una di loro dà l'errore qui sotto, in italiano.
    """
    for field in dataclasses.fields(cls):
        default = field.default
        if field.name in skip or default is dataclasses.MISSING or type(default) not in _OPTION_TYPES:
            continue
        flag, dest = "--" + field.name.replace("_", "-"), prefix + field.name
        try:
            if isinstance(default, bool):
                group.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction, default=None,
                                   help=f"{cls.__name__}.{field.name} (predefinito: {default})")
            else:
                group.add_argument(flag, dest=dest, type=_parser_for(type(default)), default=None,
                                   metavar=field.name.upper(),
                                   help=f"{cls.__name__}.{field.name} (predefinito: {default})")
        except argparse.ArgumentError:
            raise SystemExit(f"Errore: il campo {field.name} di {cls.__name__} ha lo stesso nome di un'altra "
                             "opzione dello script: rinominalo") from None


def _parser_for(kind: type):
    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"serve {_TYPE_NAMES[kind]}, non {text!r}") from None
        # float("nan") e float("inf") si leggono, ma un learning rate NaN rovinerebbe il training in silenzio.
        if kind is float and not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"serve un numero finito, non {text!r}")
        return value

    return parse


def _positive_int(text: str) -> int:
    value = _parser_for(int)(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"serve un intero positivo, non {text!r}")
    return value


def overrides(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    """I campi che la riga di comando ha cambiato: ``{"n_layer": 6}``."""
    return {k[len(prefix):]: v for k, v in vars(args).items() if k.startswith(prefix) and v is not None}


def _step_list(text: str) -> tuple[int, ...]:
    try:
        steps = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"servono step interi separati da virgole, non {text!r}") from None
    if not steps:
        raise argparse.ArgumentTypeError("serve almeno uno step")
    return steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _ItalianParser(description=__doc__, formatter_class=_ItalianHelpFormatter, add_help=False)
    # I gruppi si mostrano nell'ordine in cui nascono, ma si riempiono dopo le opzioni fisse (vedi sotto).
    general = parser.add_argument_group("uscite e dispositivo")
    model = parser.add_argument_group("modello: i campi di GPTConfig (gpt/config.py)")
    training = parser.add_argument_group("training: i campi di TrainConfig (gpt/config.py)")
    recording = parser.add_argument_group("registrazione (gpt/recording.py)")
    general.add_argument("-h", "--help", action="help", help="mostra questo aiuto ed esce")
    general.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    general.add_argument("--threads", type=_positive_int,
                         help="thread di PyTorch (su una macchina già carica: 1; i risultati non cambiano)")
    general.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                         help="dove salvare il modello")
    general.add_argument("--out", type=Path, default=DEFAULT_OUT,
                         help="cartella di trace.json e degli snapshot (predefinita: webapp/data/training)")
    general.add_argument("--no-export", action="store_true", help="non rigenerare model.json e weights.*")
    general.add_argument("--update-webapp", action="store_true",
                         help="scrive in webapp/data anche un training che non è quello di riferimento: "
                         "serve solo a chi cambia il GPT del corso")
    recording.add_argument("--log-every", type=_positive_int, help="ogni quanti step misurare gradienti, "
                           f"update e pesi (predefinito: {RecorderConfig.log_every})")
    recording.add_argument("--snapshot-steps", type=_step_list, metavar="0,50,100",
                           help="step degli snapshot dei pesi (quello finale c'è sempre; predefiniti: "
                           f"{','.join(map(str, RecorderConfig.snapshot_steps))})")
    # Per ultime le opzioni generate: un campo nuovo che si chiama come un'opzione fissa (es. log_every in
    # TrainConfig) incontra qui il conflitto e add_dataclass_options lo spiega, invece di un traceback.
    add_dataclass_options(model, GPTConfig, MODEL, skip={"vocab_size"})  # vocab_size viene dal testo
    add_dataclass_options(training, TrainConfig, TRAIN)
    return parser.parse_args(argv)


def recorder_config(args: argparse.Namespace) -> RecorderConfig:
    chosen = {"log_every": args.log_every, "snapshot_steps": args.snapshot_steps}
    return RecorderConfig(**{k: v for k, v in chosen.items() if v is not None})


# --------------------------------------------------------------------------- il training di riferimento


def reference_differences(
    model: nn.Module,
    train_cfg: TrainConfig,
    recorder_cfg: RecorderConfig,
    course: Mapping[str, Any] | None,
    course_code: str | None = None,
) -> tuple[list[str], list[str]]:
    """Che cosa distingue questo training da quello di riferimento del GPT del corso.

    Args:
        course: il manifest committato ``webapp/data/weights.json`` (nomi e forme dei pesi, config).
        course_code: l'impronta del codice dei moduli del GPT del corso (``model.code_sha256`` di
            ``webapp/data/benchmark/builtin.json``); None = non si controlla.

    Returns:
        ``(architettura, training)``: le differenze del modello (pesi, pesi condivisi, config, codice) e
        quelle del training o della registrazione. Due liste vuote = è il training di webapp/data.
    """
    architecture = []
    if course is not None:
        # I tensori che si REGISTRANO e si esportano (lo state_dict in virgola mobile), non solo i
        # parametri: un buffer persistente aggiunto cambia gli snapshot anche senza parametri nuovi.
        ours = [(name, list(t.shape)) for name, t in snapshot_state(model).items()]
        theirs = [(p.get("name"), p.get("shape")) for p in course.get("params") or []]
        if ours != theirs:
            architecture.append(f"pesi diversi da quelli del GPT del corso: {_layout_change(ours, theirs)}")
        shared = [f"{alias} è lo stesso tensore di {original}"
                  for alias, original in shared_parameters(model).items()]
        if shared:
            architecture.append(f"pesi condivisi: {_few(shared)}")
        to_dict = getattr(getattr(model, "config", None), "to_dict", None)
        architecture += _changes("config", to_dict() if callable(to_dict) else {}, course.get("config") or {})
    if course_code is not None:
        files = _code_files(model)
        if _code_sha256(files) != course_code:
            names = ", ".join(f.relative_to(PROJECT_ROOT).as_posix() for f in files) or "nessun file"
            architecture.append(
                f"codice dei moduli del modello ({names}) diverso da quello del GPT del corso misurato in "
                "webapp/data/benchmark/builtin.json: un calcolo nuovo anche con gli stessi pesi (es. GELU), "
                "o anche solo un commento"
            )
    training = _changes("training", dataclasses.asdict(train_cfg), dataclasses.asdict(TrainConfig()))
    recording = dataclasses.asdict(recorder_cfg)
    training += _changes("registrazione", recording, dataclasses.asdict(RecorderConfig()))
    return architecture, training


def same_layout_as_course(model: nn.Module, course: Mapping[str, Any] | None) -> bool:
    """Nomi e forme dei tensori registrati sono quelli del GPT del corso (anche se il calcolo è cambiato)?"""
    if course is None:  # senza i dati del corso non si sa: il promemoria che ne segue è solo una domanda
        return True
    ours = [(name, list(t.shape)) for name, t in snapshot_state(model).items()]
    return ours == [(p.get("name"), p.get("shape")) for p in course.get("params") or []]


def _layout_change(ours: list[tuple[str, list[int]]], theirs: list[tuple[str, list[int]]]) -> str:
    """In breve che cosa cambia tra due layout ``[(nome, forma)]``: tensori nuovi, mancanti, forme."""
    mine, course = dict(ours), dict(theirs)
    new = [name for name in mine if name not in course]
    missing = [name for name in course if name not in mine]
    reshaped = [f"{name} {mine[name]} invece di {course[name]}" for name in mine
                if name in course and mine[name] != course[name]]
    parts = [f"{label} {_few(items)}" for label, items in
             (("nuovi", new), ("mancanti", missing), ("forme diverse", reshaped)) if items]
    return "; ".join(parts) or "gli stessi tensori in un altro ordine"


def _few(items: list[str], limit: int = 2) -> str:
    if len(items) <= limit + 1:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} e altri {len(items) - limit}"


def _changes(what: str, values: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    """``"config: n_layer=6 invece di 4, ffn_mult=3 (campo nuovo)"``, o nessuna riga se sono uguali."""
    changed = []
    for key in dict.fromkeys([*values, *reference]):
        if key not in reference:
            changed.append(f"{key}={values[key]!r} (campo nuovo)")
        elif key not in values:
            changed.append(f"{key} assente (nel corso {reference[key]!r})")
        elif values[key] != reference[key]:
            changed.append(f"{key}={values[key]!r} invece di {reference[key]!r}")
    return [f"{what}: {', '.join(changed)}"] if changed else []


def _read_course_manifest() -> dict[str, Any] | None:
    try:
        manifest = json.loads((WEBAPP_DATA_DIR / WEIGHTS_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # niente dati committati: non c'è un GPT del corso da proteggere
    return manifest if isinstance(manifest, dict) else None


def _read_course_code() -> str | None:
    """L'impronta del codice dei moduli del GPT del corso, da ``builtin.json`` (None se non c'è)."""
    try:
        bench = json.loads((BENCHMARK_DATA_DIR / BUILTIN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    model = bench.get("model") if isinstance(bench, dict) else None
    code = model.get("code_sha256") if isinstance(model, dict) else None
    return code if isinstance(code, str) else None


def webapp_problems(
    args: argparse.Namespace, architecture: list[str], training: list[str]
) -> list[tuple[str, str]]:
    """Le scelte da cambiare perché un esperimento non sovrascriva i dati o il checkpoint del corso.

    Returns:
        ``[(perché, opzione da aggiungere)]``; vuota per il training di riferimento scritto dove ha
        sempre scritto, o con ``--update-webapp``.
    """
    if args.update_webapp:
        return []
    if not (architecture or training):
        # Il training di riferimento: pesi della web app e storia si rigenerano INSIEME, o nessuno dei due.
        inside = _inside(args.out, WEBAPP_DATA_DIR)
        if inside and args.no_export:
            why = ("con --no-export la storia nuova finirebbe in webapp/data accanto ai pesi di prima, che "
                   "non sono quelli del training registrato (per rigenerare tutto, togli --no-export)")
            return [(why, "--out checkpoints/esperimento/training")]
        if not inside and not args.no_export:
            why = "l'export aggiornerebbe i pesi di webapp/data, ma la loro storia resterebbe quella di prima"
            return [(why, "--no-export")]
        return []
    problems = []
    if _inside(args.out, WEBAPP_DATA_DIR):
        why = "--out è in webapp/data, dove c'è la storia del GPT del corso (la usano la web app e i test)"
        problems.append((why, "--out checkpoints/esperimento/training"))
    if not args.no_export:
        why = "l'export riscriverebbe model.json e weights.* di webapp/data con questo modello"
        problems.append((why, "--no-export"))
    if architecture and _same_file(args.checkpoint, DEFAULT_CHECKPOINT):
        why = "checkpoints/gpt.pt è il checkpoint del GPT del corso (lo usano generate.py e serve_webapp.py)"
        problems.append((why, "--checkpoint checkpoints/esperimento/gpt.pt"))
    return problems


def _inside(path: Path, folder: Path) -> bool:
    """``path`` è ``folder`` o sta dentro? Anche con un'altra grafia della stessa cartella.

    ``resolve()`` toglie ``..`` e collegamenti simbolici; il confronto per identità (``samestat``) delle
    cartelle che esistono copre anche i file system che non distinguono maiuscole e minuscole (macOS):
    ``WebApp/Data/training`` è comunque la cartella del corso.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(folder.resolve()):
        return True
    try:
        target = folder.stat()
    except OSError:
        return False
    return any(os.path.samestat(p.stat(), target) for p in (resolved, *resolved.parents) if p.exists())


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    return a.exists() and b.exists() and os.path.samefile(a, b)


def output_problems(checkpoint: Path, out: Path) -> list[str]:
    """Le uscite che fallirebbero solo DOPO il training (minuti persi): si controllano prima di cominciare."""
    problems = []
    blocked = _blocking_file(out)
    if blocked is not None:
        problems.append(f"--out {out}: non si può usare come cartella, {blocked} è un file")
    elif (out / TRACE_FILE).is_dir():
        problems.append(f"--out {out}: contiene una cartella {TRACE_FILE}, al posto del file della storia")
    elif not _writable(out):
        problems.append(f"--out {out}: non hai i permessi per scriverci")
    if checkpoint.is_dir():
        problems.append(f"--checkpoint {checkpoint}: è una cartella, serve un file (es. prova/gpt.pt)")
    elif (blocked := _blocking_file(checkpoint.parent)) is not None:
        problems.append(f"--checkpoint {checkpoint}: {blocked} è un file, non una cartella")
    elif not _writable(checkpoint.parent):
        problems.append(f"--checkpoint {checkpoint}: non hai i permessi per scrivere in {checkpoint.parent}")
    return problems


def _blocking_file(folder: Path) -> Path | None:
    """Il primo antenato esistente di ``folder`` (o ``folder`` stessa), se è un file e non una cartella."""
    for candidate in (folder, *folder.parents):
        if candidate.exists():
            return None if candidate.is_dir() else candidate
    return None


def _writable(folder: Path) -> bool:
    existing = next(p for p in (folder, *folder.resolve().parents) if p.exists())
    return os.access(existing, os.W_OK)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads is not None:
        torch.set_num_threads(args.threads)  # prima di ogni calcolo
    try:
        device = resolve_device(args.device)
        train_cfg = TrainConfig(**overrides(args, TRAIN))
        recorder_cfg = recorder_config(args)
        # Stessa sequenza di train.py: il seme va fissato PRIMA di creare il modello.
        torch.manual_seed(train_cfg.seed)
        text = load_text()
        tokenizer = CharTokenizer.from_text(text)
        splits = train_val_split(torch.tensor(tokenizer.encode(text), dtype=torch.long))
        config = GPTConfig(vocab_size=tokenizer.vocab_size, **overrides(args, MODEL))
    except (ValueError, RuntimeError, TypeError, OSError) as error:  # opzioni non valide, device, dataset
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    model = GPTLanguageModel(config).to(device)
    course = _read_course_manifest()
    course_code = _read_course_code()
    architecture, training = reference_differences(model, train_cfg, recorder_cfg, course, course_code)
    problems = webapp_problems(args, architecture, training)
    if problems:
        print_refusal(architecture + training, problems)
        return 1
    unusable = output_problems(args.checkpoint, args.out)
    if unusable:  # prima del training: dopo, i minuti di calcolo andrebbero persi
        print("\n".join(["Errore: le uscite non si possono scrivere:", *(f"  - {p}" for p in unusable)]),
              file=sys.stderr)
        return 1

    recorder = TrainingRecorder(tokenizer, splits.val, config.block_size, recorder_cfg)
    print(f"device: {device} | parametri: {model.num_parameters():,} | step: {train_cfg.max_iters}"
          f" | thread: {torch.get_num_threads()}")
    for difference in architecture + training:
        print(f"  diverso dal GPT del corso: {difference}")

    start = time.perf_counter()
    history = train(model, splits, config.block_size, train_cfg, device, callbacks=[recorder])
    print(f"training completato in {time.perf_counter() - start:.0f} s")
    final = history[-1]
    print(f"loss finali (step {final.step}): train {final.train_loss:.4f}, val {final.val_loss:.4f}")

    save_checkpoint(args.checkpoint, model, tokenizer)
    print(f"checkpoint salvato in {args.checkpoint}")
    trace = recorder.save(args.out)
    print(f"traccia salvata in {trace} ({len(recorder.snapshots)} snapshot)")
    for note in recorder.to_dict().get("notes", []):
        print(f"  nota: {note}")

    if not args.no_export:
        # Dopo la traccia: così weights.json sa che la storia del training esiste.
        spec = export(args.checkpoint)
        print(f"web app aggiornata da: {spec['source']}")
    if architecture or training or not _inside(args.out, WEBAPP_DATA_DIR):
        same_layout = same_layout_as_course(model, course)
        print("\n".join(next_steps(args.checkpoint, args.out, same_layout=same_layout)))
    return 0


def print_refusal(differences: list[str], problems: list[tuple[str, str]]) -> None:
    """Spiega, PRIMA del training, perché non si parte e che cosa aggiungere al comando."""
    if differences:
        lines = ["Errore: questo non è il training di riferimento del GPT del corso:"]
        lines += [f"  - {difference}" for difference in differences]
        lines += ["e così sovrascriverebbe i file del corso:"]
    else:
        lines = ["Errore: così i dati della web app resterebbero a metà:"]
    lines += [f"  - {why}" for why, _ in problems]
    lines += [f"Aggiungi al comando: {' '.join(fix for _, fix in problems)}",
              "(--update-webapp scrive comunque in webapp/data: serve solo a chi cambia il GPT del corso)"]
    print("\n".join(lines), file=sys.stderr)


def next_steps(checkpoint: Path, out: Path, *, same_layout: bool) -> list[str]:
    """Il comando per portare il run nel Laboratorio pesi, con i percorsi di questo training."""
    # La cartella dell'esperimento (checkpoints/sei-blocchi/gpt.pt → sei-blocchi), se c'è.
    folder = checkpoint.parent.name
    name = folder if folder not in {"", ".", "checkpoints"} else checkpoint.stem
    command = (f"python scripts/export_run.py {shlex.quote(str(checkpoint))} --name {shlex.quote(name)} "
               f"--trace {shlex.quote(str(out))} --benchmark")
    lines = ["Per confrontarlo nel tab «Laboratorio pesi» della web app (poi carica la cartella del run):",
             f"  {command}"]
    if same_layout:
        lines.append("  Hai cambiato il calcolo senza cambiare nomi e forme dei pesi (es. GELU al posto "
                     "di ReLU)? Aggiungi --not-engine-compatible: il motore JavaScript farebbe i conti "
                     "del GPT del corso.")
    else:
        lines.append("  Nomi o forme diversi dal GPT del corso: il Laboratorio mostra pesi, storia e "
                     "benchmark, senza eseguire il modello nel browser.")
    return lines


if __name__ == "__main__":
    sys.exit(main())
