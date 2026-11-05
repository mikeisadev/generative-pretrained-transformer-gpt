"""GPT didattico: il transformer del video "Let's build GPT" di Andrej Karpathy,
riorganizzato in moduli che seguono il percorso della lezione L20.

Ordine di lettura consigliato:
    config → data → bigram → didactic.attention_trick → attention → feedforward
    → block → model → training → generation → weights
"""

from gpt.bigram import BigramLanguageModel
from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, get_batch, load_text, train_val_split
from gpt.device import resolve_device
from gpt.model import GPTLanguageModel

__all__ = [
    "BigramLanguageModel",
    "CharTokenizer",
    "GPTConfig",
    "GPTLanguageModel",
    "TrainConfig",
    "get_batch",
    "load_text",
    "resolve_device",
    "train_val_split",
]
