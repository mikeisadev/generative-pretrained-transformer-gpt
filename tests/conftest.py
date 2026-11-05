"""Fixture condivise: una configurazione piccola per test veloci e deterministici."""

import pytest
import torch

from gpt.config import GPTConfig
from gpt.model import GPTLanguageModel


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(1337)


@pytest.fixture
def small_config() -> GPTConfig:
    return GPTConfig(vocab_size=11, block_size=8, n_embd=16, n_head=4, n_layer=2, dropout=0.0)


@pytest.fixture
def small_model(small_config: GPTConfig) -> GPTLanguageModel:
    return GPTLanguageModel(small_config)
