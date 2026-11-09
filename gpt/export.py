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
import functools
import inspect
import json
import re
import textwrap
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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


# --- attention di un'architettura qualsiasi ------------------------------------------------------
#
# Interfaccia (la stessa di Head in gpt/attention.py): un modulo è un'attention se la sua classe ha il
# METODO ``attention_weights``, che riceve gli stessi argomenti di ``forward`` e restituisce le
# probabilità di attention dopo maschera e softmax (prima del dropout): ``(B, T, T)`` per una head, come
# Head, oppure ``(B, H, T, T)`` per H head calcolate insieme (un'attention "fusa", con una sola Linear
# per Q, K e V). La prima dimensione è il batch dell'ingresso e ogni riga somma a 1. Un'attention che non
# lo espone (per esempio con ``F.scaled_dot_product_attention``, che non restituisce le probabilità)
# semplicemente non c'è: chi usa capture_attention salta le metriche dell'attention invece di fermarsi.

ATTENTION_METHOD = "attention_weights"
# Un indice di nn.Sequential o nn.ModuleList ("blocks.3"): solo cifre ASCII, come /^\d+$/ nel browser.
_INDEX = re.compile(r"[0-9]+")
# Le righe delle probabilità sommano a 1; il margine copre float16 e bfloat16, non i punteggi grezzi.
_ROW_SUM_ATOL = 1e-2


class AttentionWeightsError(ValueError):
    """``attention_weights`` ha restituito qualcosa che non sono probabilità di attention leggibili."""


@dataclass(frozen=True)
class AttentionSite:
    """Un modulo che espone le probabilità di attention e lo strato del modello a cui appartiene."""

    path: str  # nome in named_modules, es. "blocks.2.sa.heads.0"
    layer: int  # 0, 1, … nell'ordine in cui gli strati compaiono nel modello
    layer_path: str  # il blocco ripetuto che lo contiene, es. "blocks.2" ("" = il modello intero)
    module: nn.Module


def attention_layer_path(path: str) -> str:
    """Il blocco ripetuto che contiene un modulo: il percorso fino al primo indice numerico SOPRA di lui.

    ``blocks.2.sa.heads.0`` → ``blocks.2``; ``transformer.h.2.attn`` → ``transformer.h.2``. L'indice
    del modulo stesso non conta (in ``block.sa.heads.0``, con un solo blocco, ``0`` è la head): un
    modulo fuori da blocchi numerati appartiene a ``""``, il modello intero, cioè a un solo strato.
    """
    containers = path.split(".")[:-1]
    for i, part in enumerate(containers):
        if _INDEX.fullmatch(part):
            return ".".join(containers[: i + 1])
    return ""


def attention_sites(net: nn.Module) -> list[AttentionSite]:
    """I moduli di ``net`` con il metodo ``attention_weights``, nell'ordine di ``named_modules``.

    Gli strati si numerano nell'ordine in cui compaiono (per il GPT del corso ``blocks.<i>`` → i).
    Se un modulo con il metodo ne contiene altri che lo hanno, conta solo il più esterno: le sue
    probabilità comprendono già quelle dei figli, che altrimenti verrebbero contate due volte.
    Un modulo riusato (lo stesso oggetto in più punti) compare una volta sola, come in ``named_modules``:
    le sue chiamate ripetute in un forward le separa ``capture_attention``.
    """
    sites: list[AttentionSite] = []
    layers: dict[str, int] = {}
    for path, module in net.named_modules():
        if not _has_attention_method(module):
            continue
        if any(site.path == "" or path.startswith(site.path + ".") for site in sites):
            continue
        layer_path = attention_layer_path(path)
        layer = layers.setdefault(layer_path, len(layers))
        sites.append(AttentionSite(path, layer, layer_path, module))
    return sites


def _has_attention_method(module: nn.Module) -> bool:
    """La CLASSE del modulo definisce il metodo ``attention_weights``?

    ``getattr_static`` guarda la classe senza eseguire niente: un sotto-modulo, un tensore o una property
    che si chiamano così non sono l'interfaccia, e una property non viene calcolata solo per cercarla.
    """
    return isinstance(inspect.getattr_static(type(module), ATTENTION_METHOD, None), types.FunctionType)


@dataclass
class _ForwardCounters:
    """I contatori di UN forward del modello (si azzerano all'inizio di ogni forward)."""

    layers: int  # strati di attention_sites: le righe oltre questo numero sono passaggi ripetuti
    heads: dict[int, int] = field(default_factory=dict)  # riga → prossima head libera
    calls: dict[str, int] = field(default_factory=dict)  # modulo → chiamate già viste in questo forward

    def reset(self, *_hook_args: Any) -> None:
        self.heads.clear()
        self.calls.clear()

    def row(self, site: AttentionSite) -> int:
        """La riga di questa chiamata: lo strato del modulo e, se lo stesso modulo lavora di nuovo nello
        stesso forward (un blocco riusato: pesi condivisi tra gli strati), una riga nuova per ogni passaggio.

        Per il GPT del corso ogni head lavora una volta sola: la riga è lo strato, come prima.
        """
        repeat = self.calls.get(site.path, 0)
        self.calls[site.path] = repeat + 1
        return site.layer + repeat * self.layers


@contextmanager
def capture_attention(
    net: nn.Module, problems: list[str] | None = None
) -> Iterator[dict[tuple[int, int], torch.Tensor]]:
    """Dentro il ``with``, ogni forward salva i pesi di attention (B, T, T) di ogni head.

    Funziona con qualunque architettura: gli hook vanno sui moduli di ``attention_sites`` (quelli
    che espongono ``attention_weights``, ovunque siano nel modello). Le chiavi sono ``(layer, head)``:
    ``layer`` è lo strato di ``attention_sites``, le head di uno strato si numerano nell'ordine del
    forward (un modulo ``(B, H, T, T)`` ne dà H). Per il GPT del corso la chiave di
    ``blocks.<l>.sa.heads.<h>`` è ``(l, h)``. Un modulo che lavora più volte nello stesso forward (un
    blocco riusato in più strati) dà una riga in più per ogni passaggio: ``layer + passaggio × strati``.
    Senza moduli che espongono il metodo il dizionario resta vuoto.

    Args:
        problems: se è una lista, un modulo il cui ``attention_weights`` fallisce o restituisce
            qualcosa che non sono probabilità viene saltato e il motivo finisce qui, una volta sola
            (per chi registra un training: un errore nella misura non deve fermarlo). Senza lista
            l'errore arriva al chiamante.

    Gli hook vengono rimossi all'uscita anche in caso di errore, così il modello torna esattamente com'era.
    """
    attention: dict[tuple[int, int], torch.Tensor] = {}
    sites = attention_sites(net)
    counters = _ForwardCounters(layers=1 + max((site.layer for site in sites), default=-1))
    hooks = []
    if sites:
        # Ogni forward del modello ricomincia da head 0: un secondo forward sovrascrive il primo.
        hooks.append(net.register_forward_pre_hook(counters.reset))
    for site in sites:
        capture = functools.partial(_capture_site, site, attention, counters, problems)
        hooks.append(site.module.register_forward_hook(capture, with_kwargs=True))
    try:
        yield attention
    finally:
        for hk in hooks:
            hk.remove()


def _capture_site(site, attention, counters, problems, module, args, kwargs, _output) -> None:
    """Hook di un modulo di attention: chiede le probabilità con gli stessi argomenti del forward."""
    row = counters.row(site)  # anche se la misura fallisce: i passaggi dopo restano nelle loro righe
    where = f"{site.path}.{ATTENTION_METHOD}()"
    try:
        value = getattr(module, ATTENTION_METHOD)(*args, **kwargs)
        weights = _attention_probabilities(value, args[0] if args else None)
    except AttentionWeightsError as error:
        _report(f"{where}: {error}", problems)
        return
    except Exception as error:  # il codice di chi ha modificato il modello: si segnala, non si ferma
        if problems is None:
            raise
        _report(f"{where} ha dato un errore ({type(error).__name__}: {error})", problems)
        return
    first = counters.heads.get(row, 0)
    if weights.dim() == 3:  # una head, come Head del corso: il tensore così com'è
        attention[(row, first)] = weights
        counters.heads[row] = first + 1
        return
    for h in range(weights.shape[1]):
        attention[(row, first + h)] = weights[:, h]
    counters.heads[row] = first + weights.shape[1]


def _report(message: str, problems: list[str] | None) -> None:
    if problems is None:
        raise AttentionWeightsError(message)
    if message not in problems:  # una volta sola, anche se il forward si ripete a ogni snapshot
        problems.append(message)


@torch.no_grad()
def _attention_probabilities(value: Any, first_input: Any = None) -> torch.Tensor:
    """Il risultato di ``attention_weights``, staccato dal grafo, se sono davvero probabilità leggibili.

    ``first_input`` è il primo argomento del forward: se è un ``(B, T, C)`` con lo stesso T, la prima
    dimensione delle probabilità deve essere il suo B. Un ``(B·H, T, T)`` (head e batch fusi, come con
    ``.view(B * H, T, hs)``) passerebbe tutti gli altri controlli e diventerebbe UNA head con la media
    di tutte: meglio dirlo e saltarlo.
    """
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise AttentionWeightsError(f"atteso un tensore float, ottenuto {type(value).__name__}")
    if value.dim() not in (3, 4) or value.shape[-1] != value.shape[-2]:
        raise AttentionWeightsError(f"forma {tuple(value.shape)}, attese (B, T, T) o (B, H, T, T)")
    batch = _batch_of(first_input, value.shape[-1])
    if batch is not None and value.shape[0] != batch:
        raise AttentionWeightsError(
            f"forma {tuple(value.shape)} con un ingresso di forma {tuple(first_input.shape)}: la prima "
            f"dimensione deve essere il batch ({batch}). Con head e batch insieme, (B·H, T, T), restituisci "
            "weights.view(B, H, T, T)"
        )
    value = value.detach()
    if not bool(torch.isfinite(value).all()):
        raise AttentionWeightsError("probabilità non finite (NaN o infinito): il training è divergente?")
    rows = value.float().sum(-1)
    if bool((value < 0).any()) or not torch.allclose(rows, torch.ones_like(rows), atol=_ROW_SUM_ATOL):
        raise AttentionWeightsError("non sono probabilità: ogni riga deve sommare a 1 (softmax e maschera)")
    return value


def _batch_of(first_input: Any, seq_len: int) -> int | None:
    """Il batch B di un ingresso ``(B, T, C)`` con lo stesso T delle probabilità, altrimenti None.

    Solo in quel caso si sa con certezza dov'è il batch (un ingresso ``(T, B, C)``, per esempio, non si
    controlla): il controllo non deve mai scartare un'attention giusta.
    """
    if isinstance(first_input, torch.Tensor) and first_input.dim() == 3 and first_input.shape[1] == seq_len:
        return int(first_input.shape[0])
    return None


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


def display_path(path: Path) -> str:
    """Percorso da scrivere nei JSON (model.json, weights.json, run.json): relativo al progetto, se ci sta.

    Fuori dal progetto resta il nome del file, con la cartella che lo contiene solo se non può essere quella
    personale: nei file condivisi o committati non deve finire il nome di nessuno. ``/home/mario/gpt.pt`` e
    ``C:\\Users\\mario\\gpt.pt`` → ``"…/gpt.pt"``; ``/home/mario/esperimenti/gpt.pt`` →
    ``"…/esperimenti/gpt.pt"``. (Prima: model.json scriveva il percorso assoluto, run.json
    ``"…/mario/gpt.pt"`` per un file nella home.)
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    # la cartella si tiene a partire dal terzo livello (/home/<nome>/<cartella>) e se non è la home
    keep_parent = len(resolved.parts) >= 5 and not _is_home(resolved.parent)
    return "…/" + "/".join(resolved.parts[-2:] if keep_parent else resolved.parts[-1:])


def _is_home(folder: Path) -> bool:
    try:
        return folder == Path.home().resolve()
    except (RuntimeError, OSError):  # HOME non definita: resta la regola della profondità
        return False


# Il nome di prima (privato): per chi lo importava.
_display_path = display_path


def export(checkpoint: Path | None, out: Path = WEBAPP_DATA) -> dict[str, Any]:
    """Esporta dal checkpoint se esiste, altrimenti da un modello appena inizializzato.

    Scrive ``out`` (model.json) e, nella stessa cartella, i pesi completi
    (weights.json + weights.f32.bin). Restituisce il contenuto di model.json.
    """
    if checkpoint is not None and checkpoint.exists():
        net, tokenizer = training.load_checkpoint(checkpoint)
        spec = build_spec(net, tokenizer, source=display_path(checkpoint), trained=True)
    else:
        torch.manual_seed(1337)
        tokenizer = data.CharTokenizer.from_text(data.load_text())
        net = GPTLanguageModel(config.GPTConfig(vocab_size=tokenizer.vocab_size))
        spec = build_spec(net, tokenizer, source="inizializzazione casuale (seed 1337)", trained=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    export_weights(net, tokenizer, out.parent, source=spec["source"], trained=spec["trained"])
    return spec
