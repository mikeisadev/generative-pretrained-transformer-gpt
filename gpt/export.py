"""Esporta l'architettura per la web app 3D, leggendola dal codice Python.

Tutto ciò che la web app mostra (iperparametri, forme e numero dei pesi,
frammenti di codice, valori dei pesi e pattern di attention) viene prodotto
qui a partire dal modello vero: niente numeri copiati a mano che possono
andare fuori sincrono. Un test verifica che il JSON committato sia aggiornato.
"""

from __future__ import annotations

import base64
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from gpt import bigram, block, config, data, feedforward, generation, loss, model, training, weights
from gpt.attention import Head, MultiHeadAttention
from gpt.didactic import attention_trick
from gpt.model import GPTLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DATA = PROJECT_ROOT / "webapp" / "data" / "model.json"
EXAMPLE_PROMPT = "ROMEO:\nBut, soft! what light through yonder window breaks?"

# chiave usata dalla web app → oggetto Python di cui mostrare il sorgente
SNIPPETS: dict[str, Any] = {
    "config": config.GPTConfig,
    "data.download": data.download_tiny_shakespeare,
    "data.tokenizer": data.CharTokenizer,
    "data.get_batch": data.get_batch,
    "bigram": bigram.BigramLanguageModel,
    "trick.loop": attention_trick.average_past_loop,
    "trick.matmul": attention_trick.average_past_matmul,
    "trick.softmax": attention_trick.average_past_softmax,
    "trick.v4": attention_trick.self_attention_single_head,
    "model.init": model.GPTLanguageModel.__init__,
    "model.forward": model.GPTLanguageModel.forward,
    "head": Head,
    "mha": MultiHeadAttention,
    "ffwd": feedforward.FeedForward,
    "block": block.Block,
    "loss": loss.next_token_loss,
    "train": training.train,
    "generate": generation.generate,
    "weights.edit": weights.edited_weight,
}


def _snippet(obj: Any) -> dict[str, Any]:
    obj = inspect.unwrap(obj)  # es. funzioni decorate con @contextmanager o @torch.no_grad
    lines, start = inspect.getsourcelines(obj)
    return {
        "file": Path(inspect.getsourcefile(obj)).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "start_line": start,
        "code": textwrap.dedent("".join(lines)).rstrip(),
    }


def _b64(values: torch.Tensor) -> str:
    return base64.b64encode(values.to(torch.uint8).contiguous().numpy().tobytes()).decode("ascii")


def _quantize_signed(t: torch.Tensor) -> dict[str, Any]:
    """Pesi → byte 0..255 con 128 = zero, simmetrico rispetto al massimo in valore assoluto."""
    matrix = t.detach().float().cpu()
    matrix = matrix.unsqueeze(0) if matrix.ndim == 1 else matrix
    scale = float(matrix.abs().max()) or 1.0
    q = torch.clamp(torch.round(matrix / scale * 127) + 128, 0, 255)
    return {"shape": list(matrix.shape), "scale": round(scale, 6), "data": _b64(q)}


@torch.no_grad()
def _example_activity(net: GPTLanguageModel, tokenizer: data.CharTokenizer) -> dict[str, Any]:
    """Pattern di attention di ogni head e previsione del prossimo token su un prompt d'esempio."""
    prompt = EXAMPLE_PROMPT[: net.config.block_size]
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=next(net.parameters()).device)

    attention: dict[str, str] = {}
    hooks = []
    for layer, blk in enumerate(net.blocks):
        for h, head in enumerate(blk.sa.heads):

            def capture(module, inputs, _output, key=f"{layer}.{h}"):
                wei = module.attention_weights(inputs[0])[0]  # (T, T), primo elemento del batch
                attention[key] = _b64(torch.round(wei.cpu() * 255))

            hooks.append(head.register_forward_hook(capture))
    was_training = net.training
    net.eval()
    try:
        logits, _ = net(idx)
    finally:
        for hk in hooks:
            hk.remove()
        net.train(was_training)

    probs = F.softmax(logits[0, -1], dim=-1).cpu()
    return {
        "prompt": prompt,
        "tokens": idx[0].tolist(),
        "attention": attention,
        "next_token_probs": [round(p, 5) for p in probs.tolist()],
    }


def build_spec(net: GPTLanguageModel, tokenizer: data.CharTokenizer, source: str, trained: bool) -> dict[str, Any]:
    inventory = weights.weight_inventory(net)
    return {
        "source": source,
        "trained": trained,
        "config": net.config.to_dict(),
        "head_size": net.config.head_size,
        "total_params": net.num_parameters(),
        "vocab": tokenizer.chars,
        "params": [w.to_dict() for w in inventory if w.learned],
        "buffers": [w.to_dict() for w in inventory if not w.learned],
        "tensors": {name: _quantize_signed(p) for name, p in net.named_parameters()},
        "example": _example_activity(net, tokenizer),
        "snippets": {key: _snippet(obj) for key, obj in SNIPPETS.items()},
    }


def export(checkpoint: Path | None, out: Path = WEBAPP_DATA) -> dict[str, Any]:
    """Esporta dal checkpoint se esiste, altrimenti da un modello appena inizializzato."""
    if checkpoint is not None and checkpoint.exists():
        net, tokenizer = training.load_checkpoint(checkpoint)
        spec = build_spec(net, tokenizer, source=checkpoint.as_posix(), trained=True)
    else:
        torch.manual_seed(1337)
        tokenizer = data.CharTokenizer.from_text(data.load_text())
        net = GPTLanguageModel(config.GPTConfig(vocab_size=tokenizer.vocab_size))
        spec = build_spec(net, tokenizer, source="inizializzazione casuale (seed 1337)", trained=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return spec
