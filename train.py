"""Addestra il GPT su Tiny Shakespeare e salva il checkpoint.

Esempi:
    python train.py                      # training completo (5000 step, iperparametri del notebook)
    python train.py --max-iters 500      # prova veloce
    python train.py --device cpu         # forza la CPU
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, load_text, train_val_split
from gpt.device import resolve_device
from gpt.model import GPTLanguageModel
from gpt.training import save_checkpoint, train

DEFAULT_CHECKPOINT = Path("checkpoints/gpt.pt")


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-iters", type=int, default=defaults.max_iters)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval-iters", type=int, default=defaults.eval_iters)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    train_cfg = TrainConfig(
        max_iters=args.max_iters, eval_interval=args.eval_interval, eval_iters=args.eval_iters, seed=args.seed
    )
    torch.manual_seed(train_cfg.seed)

    text = load_text()
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    splits = train_val_split(data)

    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config).to(device)
    print(f"device: {device} | parametri: {model.num_parameters():,} ({model.num_parameters() / 1e6:.6f} M)")

    start = time.perf_counter()
    train(model, splits, config.block_size, train_cfg, device)
    print(f"training completato in {time.perf_counter() - start:.0f} s")

    save_checkpoint(args.checkpoint, model, tokenizer)
    print(f"checkpoint salvato in {args.checkpoint}")


if __name__ == "__main__":
    main()
