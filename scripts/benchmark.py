"""Misura un GPT con il protocollo deterministico di gpt/benchmark.py, oppure confronta due run.

    python scripts/benchmark.py checkpoints/gpt.pt                       # metriche a schermo
    python scripts/benchmark.py runs/gpt-con-gelu --out runs/gpt-con-gelu/benchmark.json
    python scripts/benchmark.py webapp/data runs/gpt-con-gelu            # confronto A/B
    python scripts/benchmark.py webapp/data/benchmark/builtin.json runs/gpt-con-gelu/benchmark.json

Ogni sorgente può essere un checkpoint (.pt), una cartella di run (weights.json + weights.f32.bin,
come runs/<nome>/ o webapp/data) oppure un benchmark.json già calcolato, che non si ricalcola: così
si confronta anche un run che il codice attuale non sa più eseguire (architettura cambiata).
I pesi si eseguono con la classe del codice ATTUALE: un'architettura migliorata si misura con lo
stesso metro del GPT del corso (validation completa di Tiny Shakespeare, finestre contigue).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpt.benchmark import (
    BenchmarkSettings,
    benchmark_path,
    format_comparison,
    format_summary,
    load_benchmark,
    write_json,
)

_DEFAULTS = BenchmarkSettings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path, metavar="SORGENTE",
                        help="checkpoint .pt, cartella di run o benchmark.json (due per confrontarli)")
    parser.add_argument("--out", type=Path, help="scrive il benchmark.json della sorgente (una sola)")
    parser.add_argument("--name", help="nome da mostrare (una sola sorgente)")
    parser.add_argument("--prompt", default=_DEFAULTS.prompt, help="prompt del testo generato")
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument("--tokens", type=int, default=_DEFAULTS.max_new_tokens, help="token da generare")
    parser.add_argument("--temperature", type=float, default=_DEFAULTS.temperature)
    parser.add_argument("--threads", type=int, help="thread di PyTorch (su una macchina già carica: 1)")
    parser.add_argument("--no-snapshots", action="store_true", help="non misurare gli snapshot del training")
    args = parser.parse_args()
    if len(args.sources) > 2:
        parser.error("al massimo due sorgenti: A e B")
    if len(args.sources) == 2 and (args.out or args.name):
        parser.error("--out e --name valgono con una sola sorgente")
    return args


def load(path: Path, args: argparse.Namespace, settings: BenchmarkSettings) -> dict:
    if path.suffix == ".json":  # già calcolato: si legge e basta
        return load_benchmark(path)
    print(f"Misuro {path}…", file=sys.stderr)
    return benchmark_path(path, name=args.name, settings=settings, snapshots=not args.no_snapshots)


def main() -> int:
    args = parse_args()
    try:
        settings = BenchmarkSettings(prompt=args.prompt, seed=args.seed, max_new_tokens=args.tokens,
                                     temperature=args.temperature, threads=args.threads)
        results = [load(path, args, settings) for path in args.sources]
    except (ValueError, OSError) as error:  # file mancanti o illeggibili, architettura diversa
        print(f"Errore: {error}", file=sys.stderr)
        return 1
    if len(results) == 1:
        print(format_summary(results[0]))
        if args.out:
            write_json(args.out, results[0])
            print(f"benchmark salvato in {args.out}")
    else:
        print(format_comparison(*results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
