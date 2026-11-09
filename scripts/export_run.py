"""Esporta un checkpoint come "run" da caricare nel Laboratorio pesi della web app.

    python scripts/export_run.py checkpoints/gpt.pt --name "GPT del corso, rifatto"
    python scripts/export_run.py runs/_gelu/gpt.pt --name "GPT con GELU" --trace runs/_gelu/training \\
        --benchmark
    python scripts/export_run.py mio.pt --name prova --out /tmp/prova --force

Scrive runs/<nome>/ con run.json, weights.json + weights.f32.bin (lo stesso formato di webapp/data),
più, a richiesta, la storia del training (training/, da scripts/record_training.py --out) e
benchmark.json (gpt/benchmark.py). Il checkpoint si legge con torch.load(weights_only=True): il file
non può eseguire codice. Per esportare non serve che il codice attuale sappia ricostruire il modello
(un'architettura modificata va bene); per --benchmark sì, perché i pesi vanno eseguiti.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpt.benchmark import (
    BENCHMARK_DATA_DIR,
    EXCERPT_FILE,
    ArchitectureMismatch,
    BenchmarkSettings,
    benchmark_model,
    build_model,
    format_int,
    read_excerpt,
    rebuilt_tensors_note,
    run_tokenizer,
    terminal_text,
    validation_text,
)
from gpt.data import CharTokenizer, load_text
from gpt.runs import RUNS_DIR, export_run, read_checkpoint, read_training_history, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="checkpoint .pt (di gpt/training.py o uno state_dict)")
    parser.add_argument("--name", required=True, help='nome del run nel Laboratorio (es. "GPT con GELU")')
    parser.add_argument("--description", default="", help="che cosa è cambiato rispetto al GPT del corso")
    parser.add_argument("--note", action="append", default=[], help="nota da salvare nel run (ripetibile)")
    parser.add_argument("--out", type=Path, help="cartella del run (predefinita: runs/<nome>)")
    parser.add_argument("--trace", type=Path,
                        help="cartella con trace.json e gli snapshot (scripts/record_training.py --out)")
    parser.add_argument("--benchmark", action="store_true", help="misura il run e scrive benchmark.json")
    parser.add_argument("--threads", type=int, help="thread di PyTorch per il benchmark (macchina carica: 1)")
    parser.add_argument(
        "--not-engine-compatible",
        action="store_true",
        help="il calcolo è diverso da quello del GPT del corso anche con le stesse forme (es. GELU al posto "
        "di ReLU): il Laboratorio mostrerà i pesi senza eseguirli con il motore JavaScript",
    )
    parser.add_argument("--force", action="store_true", help="sostituisce un run già esportato lì")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out = args.out or RUNS_DIR / slugify(args.name)
        weights = read_checkpoint(args.checkpoint)
        history = read_training_history(args.trace) if args.trace else None
        model = None
        try:
            model = build_model(weights)
            note = "classi dei moduli dal codice attuale (gpt/), che ricostruisce questi pesi senza errori"
            weights = weights.with_module_info(model, note=note)
        except ValueError as error:  # architettura diversa o config mancante
            if args.benchmark:
                raise
            print(f"Nota: {error}")
            print("      L'export funziona lo stesso: i pesi si esportano anche senza la classe del modello.")

        benchmark = None
        if args.benchmark:
            text = load_text()
            tokenizer = run_tokenizer(weights, default_vocab=CharTokenizer.from_text(text).chars)
            validation = validation_text(text)
            excerpt_path = BENCHMARK_DATA_DIR / EXCERPT_FILE
            excerpt = read_excerpt(excerpt_path, validation) if excerpt_path.is_file() else None
            print("Benchmark in corso (validation completa, estratto, generazione, snapshot)…")
            benchmark = benchmark_model(model, tokenizer, name=args.name, source=weights.source,
                                        validation=validation, excerpt=excerpt, history=history,
                                        weights=weights, settings=BenchmarkSettings(threads=args.threads))
            # come benchmark_path: i tensori interi del modello ricostruito sono quelli del codice, e lo si dice
            note = rebuilt_tensors_note(model)
            if note:
                benchmark["notes"].append(note)

        info = export_run(
            weights,
            out,
            name=args.name,
            description=args.description,
            notes=args.note,
            training_dir=args.trace,
            benchmark=benchmark,
            engine_compatible=False if args.not_engine_compatible else None,
            overwrite=args.force,
        )
    except ArchitectureMismatch as error:
        print(f"Errore: {error}", file=sys.stderr)
        print("Esporta senza --benchmark per guardare i pesi nel Laboratorio.", file=sys.stderr)
        return 1
    except (ValueError, OSError) as error:  # file mancanti, cartelle occupate, dati non validi
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print_summary(info, out, benchmark)
    return 0


def print_summary(info: dict, out: Path, benchmark: dict | None) -> None:
    # config, note e nomi dei tensori vengono dal checkpoint (magari scaricato): niente caratteri di controllo
    # nel terminale (un ESC lo comanderebbe), ogni riga passa da terminal_text
    def say(text: str) -> None:
        print(terminal_text(text))

    megabytes = 4 * info["total_params"] / 1e6
    say(f'Run "{info["name"]}" esportato in {out}')
    say(f"  pesi: {format_int(info['total_params'])} valori in {info['tensor_count']} tensori "
        f"({megabytes:.2f} MB in float32)")
    config = info.get("config")
    described = ", ".join(f"{k}={v}" for k, v in config.items()) if config else "assente nel checkpoint"
    say(f"  config: {described}")
    training = info.get("training")
    if training:
        steps = training["snapshots"]
        span = f"step {steps[0]} … {steps[-1]}" if steps else "nessuno snapshot"
        say(f"  storia del training: {len(steps)} snapshot ({span}), "
            f"val loss finale {training['val_loss']}")
    if benchmark:
        m = benchmark["metrics"]
        say(f"  benchmark: loss {m['loss']:.4f} · top-1 {100 * m['top1']:.2f}% · "
            f"top-5 {100 * m['top5']:.2f}% (validation completa, {format_int(m['predictions'])} caratteri)")
    for note in info["notes"]:
        say(f"  nota: {note}")
    say(f'Nella web app: tab "Laboratorio pesi", carica la cartella {out}')


if __name__ == "__main__":
    sys.exit(main())
