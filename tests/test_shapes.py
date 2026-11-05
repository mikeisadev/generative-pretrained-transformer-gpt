"""Forma dei tensori in uscita da ogni modulo: (B, T, C) entra, (B, T, ...) esce."""

import torch

from gpt.attention import Head, MultiHeadAttention
from gpt.bigram import BigramLanguageModel
from gpt.block import Block
from gpt.feedforward import FeedForward
from gpt.model import GPTLanguageModel

B, T = 3, 5


def test_head_shape(small_config):
    x = torch.randn(B, T, small_config.n_embd)
    assert Head(small_config)(x).shape == (B, T, small_config.head_size)


def test_multi_head_shape(small_config):
    x = torch.randn(B, T, small_config.n_embd)
    assert MultiHeadAttention(small_config)(x).shape == (B, T, small_config.n_embd)


def test_feedforward_shape(small_config):
    x = torch.randn(B, T, small_config.n_embd)
    assert FeedForward(small_config)(x).shape == (B, T, small_config.n_embd)


def test_block_shape(small_config):
    x = torch.randn(B, T, small_config.n_embd)
    assert Block(small_config)(x).shape == (B, T, small_config.n_embd)


def test_gpt_logits_shape(small_model, small_config):
    idx = torch.randint(small_config.vocab_size, (B, T))
    logits, loss = small_model(idx)
    assert logits.shape == (B, T, small_config.vocab_size)
    assert loss is None


def test_gpt_logits_shape_is_the_same_with_targets(small_model, small_config):
    """L'originale restituiva logits (B*T, V) con i target e (B, T, V) senza: ora è sempre (B, T, V)."""
    idx = torch.randint(small_config.vocab_size, (B, T))
    logits, loss = small_model(idx, idx)
    assert logits.shape == (B, T, small_config.vocab_size)
    assert loss.ndim == 0


def test_bigram_shape():
    model = BigramLanguageModel(vocab_size=7)
    logits, _ = model(torch.randint(7, (B, T)))
    assert logits.shape == (B, T, 7)


def test_gpt_rejects_context_longer_than_block_size(small_model, small_config):
    idx = torch.zeros((1, small_config.block_size + 1), dtype=torch.long)
    try:
        small_model(idx)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("atteso ValueError")


def test_default_config_has_209_729_parameters():
    """Il numero stampato nel video di Karpathy: 0.209729 M parametri."""
    from gpt.config import GPTConfig

    assert GPTLanguageModel(GPTConfig()).num_parameters() == 209_729
