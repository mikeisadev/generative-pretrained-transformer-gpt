"""Esperimento 6 — Modificare a mano UN peso e osservare l'effetto.

Il modello ha 209.729 pesi. Ne cambiamo uno solo: il bias di ``lm_head`` per
la lettera scelta. È il "punteggio di partenza" di quella lettera prima di
ogni previsione: alzandolo, la lettera diventa molto più probabile ovunque.

    python train.py                          # serve un checkpoint addestrato
    python experiments/06_edit_a_weight.py --char z --value 8
"""

import argparse
from pathlib import Path

import torch
from _common import section

from gpt.training import load_checkpoint
from gpt.weights import edited_weight, next_token_distribution

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gpt.pt"))
parser.add_argument("--char", default="z", help="il carattere da 'potenziare'")
parser.add_argument("--value", type=float, default=8.0, help="nuovo valore del bias")
parser.add_argument("--prompt", default="ROMEO:\nWhat is ")
args = parser.parse_args()

if not args.checkpoint.exists():
    raise SystemExit("Checkpoint non trovato: lancia prima `python train.py`.")

model, tokenizer = load_checkpoint(args.checkpoint, "cpu")
token_id = tokenizer.encode(args.char)[0]
weight_name = "lm_head.bias"


def show(title: str) -> None:
    section(title)
    full = next_token_distribution(model, tokenizer, args.prompt, top_k=tokenizer.vocab_size)
    for ch, p in full[:5]:
        print(f"  {ch!r:6} {p:6.1%}")
    print(f"  probabilità di {args.char!r}: {dict(full)[args.char]:.2%}")
    torch.manual_seed(0)  # stesso seed → l'unica differenza è il peso modificato
    ctx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    print("\n" + tokenizer.decode(model.generate(ctx, max_new_tokens=150)[0].tolist()))


show(f"PRIMA — prossimo carattere dopo {args.prompt!r}")

with edited_weight(model, weight_name, (token_id,), args.value) as old:
    print(f"\n>>> {weight_name}[{token_id}] (carattere {args.char!r}): {old:+.4f} → {args.value:+.4f}")
    show("DOPO la modifica di un solo peso")

show("RIPRISTINATO — di nuovo identico a PRIMA")
