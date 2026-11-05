"""Rifà il training di riferimento registrandone la storia per la web app.

    python scripts/record_training.py              # come train.py: 5000 step, ~3-4 minuti su CPU
    python scripts/record_training.py --device cpu
    python scripts/record_training.py --max-iters 300 --out /tmp/prova --checkpoint /tmp/prova/gpt.pt

Produce:
    checkpoints/gpt.pt                          il modello addestrato (come train.py)
    webapp/data/training/trace.json             loss per step, metriche, snapshot (vedi gpt/recording.py)
    webapp/data/training/snap_XXXXX.f16.bin     i pesi completi a step scelti, in float16
e poi riesporta webapp/data (model.json, weights.json, weights.f32.bin) dal nuovo checkpoint.

Iperparametri e seme sono quelli di train.py, e il recorder non tocca il generatore
casuale globale: i batch, e quindi le loss, sono gli stessi del training di riferimento.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, load_text, train_val_split
from gpt.device import resolve_device
from gpt.export import PROJECT_ROOT, TRAINING_DIR, WEBAPP_DATA, export
from gpt.model import GPTLanguageModel
from gpt.recording import TrainingRecorder
from gpt.training import save_checkpoint, train


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-iters", type=int, default=defaults.max_iters)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "gpt.pt")
    parser.add_argument("--out", type=Path, default=WEBAPP_DATA.parent / TRAINING_DIR, help="cartella della traccia")
    parser.add_argument("--no-export", action="store_true", help="non rigenerare model.json e weights.*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    train_cfg = TrainConfig(max_iters=args.max_iters)
    # Stessa sequenza di train.py: il seme va fissato PRIMA di creare il modello.
    torch.manual_seed(train_cfg.seed)

    text = load_text()
    tokenizer = CharTokenizer.from_text(text)
    splits = train_val_split(torch.tensor(tokenizer.encode(text), dtype=torch.long))

    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config).to(device)
    recorder = TrainingRecorder(tokenizer, splits.val, config.block_size)
    print(f"device: {device} | parametri: {model.num_parameters():,} | step: {train_cfg.max_iters}")

    start = time.perf_counter()
    history = train(model, splits, config.block_size, train_cfg, device, callbacks=[recorder])
    print(f"training completato in {time.perf_counter() - start:.0f} s")
    final = history[-1]
    print(f"loss finali (step {final.step}): train {final.train_loss:.4f}, val {final.val_loss:.4f}")

    save_checkpoint(args.checkpoint, model, tokenizer)
    print(f"checkpoint salvato in {args.checkpoint}")
    trace = recorder.save(args.out)
    print(f"traccia salvata in {trace} ({len(recorder.snapshots)} snapshot)")

    if not args.no_export:
        # Dopo la traccia: così weights.json sa che la storia del training esiste.
        spec = export(args.checkpoint)
        print(f"web app aggiornata da: {spec['source']}")


if __name__ == "__main__":
    main()
