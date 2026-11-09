"""Rigenera i dati del benchmark per il browser dai pesi committati di webapp/data.

    python scripts/export_benchmark_data.py            # riscrive val_excerpt.txt e builtin.json
    python scripts/export_benchmark_data.py --check    # controlla soltanto che siano aggiornati
    python scripts/export_benchmark_data.py --threads 1

Produce in webapp/data/benchmark/:
    val_excerpt.txt   un estratto fisso della validation (~4 KB, discorsi interi) da valutare nel browser
    builtin.json      il benchmark del GPT del corso (gpt/benchmark.py): validation completa, estratto
                      (con i bit di sorpresa di ogni carattere), testo generato e gli 11 snapshot del training
Serve data/input.txt (si scarica da solo). Va rilanciato solo se cambiano i pesi committati o il
protocollo: un test (tests/test_benchmark.py) controlla che i file siano aggiornati.
"""

from __future__ import annotations

import argparse
import json
import sys

from gpt.benchmark import (
    BENCHMARK_DATA_DIR,
    BUILTIN_FILE,
    EXCERPT_FILE,
    BenchmarkSettings,
    benchmark_differences,
    compute_benchmark_data,
    format_summary,
    write_benchmark_data,
)
from gpt.runs import display_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="non scrive: esce con 1 se i file non sono aggiornati")
    parser.add_argument("--threads", type=int, help="thread di PyTorch (su una macchina già carica: 1)")
    args = parser.parse_args()
    settings = BenchmarkSettings(threads=args.threads)
    excerpt_path, builtin_path = BENCHMARK_DATA_DIR / EXCERPT_FILE, BENCHMARK_DATA_DIR / BUILTIN_FILE

    if not args.check:
        builtin = write_benchmark_data(settings=settings)
        print(format_summary(builtin))
        print(f"scritti {display_path(excerpt_path)} e {display_path(builtin_path)}")
        return 0

    fresh, excerpt = compute_benchmark_data(settings=settings)
    problems = []
    if not excerpt_path.is_file() or excerpt_path.read_bytes() != excerpt.encode("utf-8"):
        problems.append(f"{display_path(excerpt_path)} mancante o diverso dall'estratto della validation")
    if builtin_path.is_file():
        problems += benchmark_differences(json.loads(builtin_path.read_text(encoding="utf-8")), fresh)
    else:
        problems.append(f"{display_path(builtin_path)} mancante")
    for problem in problems:
        print(f"- {problem}")
    print("dati aggiornati" if not problems else "rilancia: python scripts/export_benchmark_data.py")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
