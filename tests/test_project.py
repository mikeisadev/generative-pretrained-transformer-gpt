"""Controlli sul progetto: sintassi, nomi, configurazione, device."""

import py_compile
from pathlib import Path

import pytest
import torch

from gpt.config import GPTConfig
from gpt.device import resolve_device

ROOT = Path(__file__).resolve().parent.parent
PY_SOURCES = [
    p for p in ROOT.rglob("*.py") if not any(part in {".venv", "reference"} for part in p.relative_to(ROOT).parts)
]


@pytest.mark.parametrize("path", PY_SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_python_file_compiles(path):
    """BUG: il file esportato conteneva `!wget`, che non è Python valido."""
    py_compile.compile(str(path), doraise=True)


def test_class_names_are_fixed():
    """BUG: il GPT finale si chiamava `BigramLanguageModel` e c'era il refuso `FeedFoward`."""
    import gpt.feedforward
    import gpt.model

    assert hasattr(gpt.model, "GPTLanguageModel") and not hasattr(gpt.model, "BigramLanguageModel")
    assert hasattr(gpt.feedforward, "FeedForward") and not hasattr(gpt.feedforward, "FeedFoward")


def test_config_validation():
    with pytest.raises(ValueError, match="divisibile"):
        GPTConfig(n_embd=64, n_head=5)
    with pytest.raises(ValueError):
        GPTConfig(dropout=1.0)


def test_two_configs_can_coexist():
    """BUG: con gli iperparametri globali era impossibile avere due modelli diversi nello stesso processo."""
    from gpt.model import GPTLanguageModel

    small = GPTLanguageModel(GPTConfig(n_embd=16, n_head=2, n_layer=1))
    large = GPTLanguageModel(GPTConfig())
    assert small.num_parameters() < large.num_parameters()


def test_resolve_device():
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
