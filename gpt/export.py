"""Esporta l'architettura per la web app 3D, leggendola dal codice Python.

Tutto ciò che la web app mostra (iperparametri, forme e numero dei pesi,
frammenti di codice, valori dei pesi e pattern di attention) viene prodotto
qui a partire dal modello vero: niente numeri copiati a mano che possono
andare fuori sincrono. Un test verifica che il JSON committato sia aggiornato.

File prodotti in ``webapp/data/``:

``model.json``
    config, pesi quantizzati a 8 bit per le heatmap, snippet di codice,
    attention e previsione sul prompt d'esempio (vedi ``build_spec``).

``weights.json`` + ``weights.f32.bin``
    TUTTI i pesi in float32, per il motore di inferenza JavaScript.
    ``weights.f32.bin`` è la concatenazione dei tensori, nell'ordine di
    ``model.named_parameters()``, ognuno appiattito riga per riga (C order),
    float32 little-endian, senza intestazione. ``weights.json`` è il manifest::

        {
          "format_version": 1,
          "dtype": "float32",
          "byte_order": "little",
          "file": "weights.f32.bin",          # percorso relativo a weights.json
          "source": "checkpoints/gpt.pt",     # da dove vengono i pesi
          "trained": true,                    # false = inizializzazione casuale
          "config": {vocab_size, block_size, n_embd, n_head, n_layer, dropout},
          "vocab": ["\n", " ", "!", ...],      # token id → carattere
          "total_params": 209729,
          "params": [{"name", "shape", "offset", "numel"}, ...],
          "training": {"dir": "training", "trace": "training/trace.json"} | null
        }

    ``offset`` e ``numel`` sono in NUMERO DI FLOAT (non in byte): in JavaScript
    ``new Float32Array(buffer, offset * 4, numel)``. ``training`` è ``null`` se
    la storia del training (``scripts/record_training.py``) non è stata registrata.
"""

from __future__ import annotations

import base64
import inspect
import json
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt import bigram, block, config, data, feedforward, generation, loss, model, training, weights
from gpt.attention import Head, MultiHeadAttention
from gpt.didactic import attention_trick
from gpt.model import GPTLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DATA_DIR = PROJECT_ROOT / "webapp" / "data"
WEBAPP_DATA = WEBAPP_DATA_DIR / "model.json"
WEIGHTS_MANIFEST = "weights.json"
WEIGHTS_BINARY = "weights.f32.bin"
WEIGHTS_FORMAT_VERSION = 1
# Storia del training registrata da scripts/record_training.py (relativa a webapp/data).
TRAINING_DIR = "training"
TRAINING_TRACE = "trace.json"
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


@contextmanager
def capture_attention(net: GPTLanguageModel) -> Iterator[dict[tuple[int, int], torch.Tensor]]:
    """Dentro il ``with``, ogni forward salva i pesi di attention (B, T, T) di ogni head.

    Le chiavi sono ``(layer, head)``. Gli hook vengono rimossi all'uscita
    anche in caso di errore, così il modello torna esattamente com'era.
    """
    attention: dict[tuple[int, int], torch.Tensor] = {}
    hooks = []
    for layer, blk in enumerate(net.blocks):
        for h, head in enumerate(blk.sa.heads):

            def capture(module, inputs, _output, key=(layer, h)):
                attention[key] = module.attention_weights(inputs[0]).detach()

            hooks.append(head.register_forward_hook(capture))
    try:
        yield attention
    finally:
        for hk in hooks:
            hk.remove()


@torch.no_grad()
def _example_activity(net: GPTLanguageModel, tokenizer: data.CharTokenizer) -> dict[str, Any]:
    """Pattern di attention di ogni head e previsione del prossimo token su un prompt d'esempio."""
    prompt = EXAMPLE_PROMPT[: net.config.block_size]
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=next(net.parameters()).device)

    was_training = net.training
    net.eval()
    try:
        with capture_attention(net) as maps:
            logits, _ = net(idx)
    finally:
        net.train(was_training)

    # primo (e unico) elemento del batch, ogni peso in 0..255
    attention = {f"{layer}.{h}": _b64(torch.round(wei[0].cpu() * 255)) for (layer, h), wei in maps.items()}
    probs = F.softmax(logits[0, -1], dim=-1).cpu()
    return {
        "prompt": prompt,
        "tokens": idx[0].tolist(),
        "attention": attention,
        "next_token_probs": [round(p, 5) for p in probs.tolist()],
    }


def parameter_layout(net: nn.Module) -> list[dict[str, Any]]:
    """Dove si trova ogni tensore nel file binario: offset e numel in numero di elementi."""
    layout, offset = [], 0
    for name, p in net.named_parameters():
        layout.append({"name": name, "shape": list(p.shape), "offset": offset, "numel": p.numel()})
        offset += p.numel()
    return layout


def parameters_to_bytes(net: nn.Module, dtype: str) -> bytes:
    """Tutti i parametri concatenati (ordine di ``named_parameters``, C order).

    ``dtype`` è un tipo numpy con byte order esplicito, es. ``"<f4"`` (float32
    little-endian) o ``"<f2"`` (float16): così il file è uguale su ogni macchina.
    """
    chunks = [p.detach().cpu().float().numpy().ravel().astype(dtype) for p in net.parameters()]
    return np.concatenate(chunks).tobytes()


def build_weights_manifest(
    net: GPTLanguageModel, tokenizer: data.CharTokenizer, source: str, trained: bool, training: dict | None
) -> dict[str, Any]:
    return {
        "format_version": WEIGHTS_FORMAT_VERSION,
        "dtype": "float32",
        "byte_order": "little",
        "file": WEIGHTS_BINARY,
        "source": source,
        "trained": trained,
        "config": net.config.to_dict(),
        "vocab": tokenizer.chars,
        "total_params": net.num_parameters(),
        "params": parameter_layout(net),
        "training": training,
    }


def export_weights(
    net: GPTLanguageModel, tokenizer: data.CharTokenizer, out_dir: Path, source: str, trained: bool
) -> dict[str, Any]:
    """Scrive ``weights.json`` e ``weights.f32.bin`` in ``out_dir`` e restituisce il manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / TRAINING_DIR / TRAINING_TRACE
    training = {"dir": TRAINING_DIR, "trace": f"{TRAINING_DIR}/{TRAINING_TRACE}"} if trace.exists() else None
    manifest = build_weights_manifest(net, tokenizer, source, trained, training)
    (out_dir / WEIGHTS_BINARY).write_bytes(parameters_to_bytes(net, "<f4"))
    (out_dir / WEIGHTS_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


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


def _display_path(path: Path) -> str:
    """Percorso relativo alla radice del progetto: nei JSON committati non finisce la home di nessuno."""
    resolved = path.resolve()
    return resolved.relative_to(PROJECT_ROOT).as_posix() if resolved.is_relative_to(PROJECT_ROOT) else path.as_posix()


def export(checkpoint: Path | None, out: Path = WEBAPP_DATA) -> dict[str, Any]:
    """Esporta dal checkpoint se esiste, altrimenti da un modello appena inizializzato.

    Scrive ``out`` (model.json) e, nella stessa cartella, i pesi completi
    (weights.json + weights.f32.bin). Restituisce il contenuto di model.json.
    """
    if checkpoint is not None and checkpoint.exists():
        net, tokenizer = training.load_checkpoint(checkpoint)
        spec = build_spec(net, tokenizer, source=_display_path(checkpoint), trained=True)
    else:
        torch.manual_seed(1337)
        tokenizer = data.CharTokenizer.from_text(data.load_text())
        net = GPTLanguageModel(config.GPTConfig(vocab_size=tokenizer.vocab_size))
        spec = build_spec(net, tokenizer, source="inizializzazione casuale (seed 1337)", trained=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    export_weights(net, tokenizer, out.parent, source=spec["source"], trained=spec["trained"])
    return spec
