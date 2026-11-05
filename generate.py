"""Genera testo "alla Shakespeare" con un checkpoint addestrato.

Esempi:
    python generate.py
    python generate.py --prompt "ROMEO:" --max-new-tokens 300 --temperature 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpt.device import resolve_device
from gpt.training import load_checkpoint

DEFAULT_CHECKPOINT = Path("checkpoints/gpt.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="\n", help="testo iniziale (default: un a capo)")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=None, help="fissalo per risultati riproducibili")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint {args.checkpoint} non trovato: lancia prima `python train.py`.")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
