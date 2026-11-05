"""Step 1 del percorso: dati e tokenizzazione.

1. Scarichiamo Tiny Shakespeare (~1 MB di testo, tutte le opere di Shakespeare).
2. Costruiamo un tokenizer a livello di carattere: ogni carattere distinto
   diventa un numero intero (il suo "token id").
3. Dividiamo i dati in train (90%) e validation (10%).
4. Estraiamo batch casuali di sequenze: l'input ``x`` e il target ``y``,
   che è lo stesso testo spostato di una posizione a destra.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Impronta SHA-256 del file originale: se il download è corrotto o manomesso ce ne accorgiamo.
DATASET_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DOWNLOAD_TIMEOUT_S = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_tiny_shakespeare(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Scarica il dataset (solo la prima volta) e ne verifica l'integrità.

    Sostituisce il comando da notebook ``!wget``, che non è Python valido.
    Il file viene scritto prima in un file temporaneo e poi spostato: se il
    download si interrompe non resta un ``input.txt`` a metà.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "input.txt"
    if target.exists() and _sha256(target) == DATASET_SHA256:
        return target

    with tempfile.NamedTemporaryFile(dir=data_dir, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(DATASET_URL, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            shutil.copyfileobj(response, tmp)

    if _sha256(tmp_path) != DATASET_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Il file scaricato non corrisponde all'impronta attesa: download annullato.")
    tmp_path.replace(target)
    return target


def load_text(data_dir: Path = DEFAULT_DATA_DIR) -> str:
    return download_tiny_shakespeare(data_dir).read_text(encoding="utf-8")


class CharTokenizer:
    """Tokenizer a caratteri: il più semplice possibile.

    I modelli reali (GPT-4, Claude, ecc.) usano tokenizer a sotto-parole
    (es. BPE) con vocabolari da decine di migliaia di token; qui il
    vocabolario sono i 65 caratteri del testo. L'idea però è identica:
    testo → lista di interi → testo.
    """

    def __init__(self, chars: list[str]) -> None:
        if len(set(chars)) != len(chars):
            raise ValueError("il vocabolario contiene caratteri duplicati")
        self.chars: list[str] = list(chars)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}  # string → int
        self.itos: dict[int, str] = dict(enumerate(self.chars))  # int → string

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"caratteri non presenti nel vocabolario: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


@dataclass(frozen=True)
class DataSplits:
    train: torch.Tensor  # (N_train,) token id
    val: torch.Tensor  # (N_val,)


def train_val_split(data: torch.Tensor, train_fraction: float = 0.9) -> DataSplits:
    """Il primo 90% per imparare, il restante 10% per verificare che il modello generalizzi."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction deve stare in (0, 1)")
    n = int(train_fraction * len(data))
    return DataSplits(train=data[:n], val=data[n:])


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estrae ``batch_size`` finestre casuali lunghe ``block_size``.

    Restituisce:
        x: (B, T) i token di input
        y: (B, T) i target, cioè x spostato di uno: y[b, t] è il token che segue x[b, :t+1]

    Una sola finestra contiene quindi T esempi di training: "dato 1 carattere
    prevedi il 2°", "dati 2 caratteri prevedi il 3°", ... fino a T.
    """
    if len(data) <= block_size:
        raise ValueError("il dataset è più corto di block_size + 1")
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
