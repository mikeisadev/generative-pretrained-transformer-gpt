"""Gli strumenti esplorativi devono dire la verità."""

import torch

from gpt.didactic.attention_trick import (
    average_past_loop,
    average_past_matmul,
    average_past_softmax,
    self_attention_single_head,
)
from gpt.didactic.layernorm import LayerNorm1d


def test_three_versions_of_the_trick_agree():
    x = torch.randn(4, 8, 2)
    v1 = average_past_loop(x)
    assert torch.allclose(v1, average_past_matmul(x), atol=1e-6)
    assert torch.allclose(v1, average_past_softmax(x), atol=1e-6)


def test_single_head_weights_are_causal_and_normalised():
    out, wei = self_attention_single_head(torch.randn(4, 8, 32), head_size=16)
    assert out.shape == (4, 8, 16)
    assert torch.allclose(wei.sum(-1), torch.ones(4, 8))
    assert torch.all(wei.triu(diagonal=1) == 0)


def test_handmade_layernorm_matches_pytorch():
    """BUG: il LayerNorm1d del notebook usava la varianza corretta (n-1) e non coincideva con nn.LayerNorm."""
    x = torch.randn(32, 100) * 3 + 5
    assert torch.allclose(LayerNorm1d(100)(x), torch.nn.LayerNorm(100)(x), atol=1e-5)
