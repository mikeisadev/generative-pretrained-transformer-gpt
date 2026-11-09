"""Fixture condivise: una configurazione piccola per test veloci e deterministici.

Durante i test PyTorch usa pochi thread (1, oppure ``GPT_TEST_THREADS``). Su una macchina condivisa o
già carica, i thread in più dei core liberi si ostacolano a vicenda: un test da un secondo ne ha
impiegati 73. I risultati non cambiano: i test confrontano numeri calcolati nello stesso processo,
oppure con tolleranze molto più larghe delle differenze (~1e-6) dovute all'ordine delle somme.
Su una macchina libera: ``GPT_TEST_THREADS=4 pytest``.
"""

import os

import pytest
import torch

from gpt.config import GPTConfig
from gpt.model import GPTLanguageModel


def _test_threads() -> int:
    raw = os.environ.get("GPT_TEST_THREADS", "1")
    try:
        value = int(raw)
    except ValueError:  # anche "²", che per str.isdigit è una cifra ma int() non legge
        value = 0
    if value <= 0:
        raise pytest.UsageError(f"GPT_TEST_THREADS deve essere un intero positivo, trovato {raw!r}")
    return value


TEST_THREADS = _test_threads()
torch.set_num_threads(TEST_THREADS)
# Anche gli script che i test lanciano in un processo a parte (torch legge queste variabili all'avvio):
# gli stessi thread del processo dei test, anche se nella shell c'era già un altro OMP_NUM_THREADS.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = str(TEST_THREADS)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(1337)


@pytest.fixture
def small_config() -> GPTConfig:
    return GPTConfig(vocab_size=11, block_size=8, n_embd=16, n_head=4, n_layer=2, dropout=0.0)


@pytest.fixture
def small_model(small_config: GPTConfig) -> GPTLanguageModel:
    return GPTLanguageModel(small_config)
