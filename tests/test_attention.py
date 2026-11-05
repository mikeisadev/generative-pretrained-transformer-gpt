"""Maschera causale e scaling dell'attention."""

import pytest
import torch
from torch.nn import functional as F

from gpt.attention import Head
from gpt.config import GPTConfig
from gpt.model import GPTLanguageModel


def test_head_output_ignores_future_tokens(small_config):
    """Cambiare il token in posizione t non deve cambiare l'output nelle posizioni < t."""
    head = Head(small_config)
    x = torch.randn(1, small_config.block_size, small_config.n_embd)
    x_changed = x.clone()
    t = 5
    x_changed[:, t] += 10.0
    out, out_changed = head(x), head(x_changed)
    assert torch.allclose(out[:, :t], out_changed[:, :t])
    assert not torch.allclose(out[:, t:], out_changed[:, t:])


def test_full_model_is_causal(small_model, small_config):
    """Stessa proprietà sull'intero GPT: i logits passati non dipendono dai token futuri."""
    idx = torch.randint(small_config.vocab_size, (2, small_config.block_size))
    idx_changed = idx.clone()
    idx_changed[:, -1] = (idx[:, -1] + 1) % small_config.vocab_size
    logits, _ = small_model(idx)
    logits_changed, _ = small_model(idx_changed)
    assert torch.allclose(logits[:, :-1], logits_changed[:, :-1], atol=1e-6)


def test_attention_scaling_uses_head_size():
    """BUG: l'originale scalava per C**-0.5 (C = n_embd) invece che per head_size**-0.5.

    Ricalcoliamo a mano l'attention con la formula del paper e confrontiamo.
    Con n_embd=64 e head_size=16 i due fattori differiscono di 2x, quindi il test
    distingue nettamente la versione corretta da quella sbagliata.
    """
    config = GPTConfig(n_embd=64, n_head=4, block_size=8)
    head = Head(config)
    x = torch.randn(2, 8, 64)

    q, k, v = head.query(x), head.key(x), head.value(x)
    mask = torch.tril(torch.ones(8, 8)) == 0

    def attention(scale: float) -> torch.Tensor:
        scores = (q @ k.transpose(-2, -1) * scale).masked_fill(mask, float("-inf"))
        return F.softmax(scores, dim=-1) @ v

    expected = attention(config.head_size**-0.5)
    buggy = attention(config.n_embd**-0.5)
    actual = head(x)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(actual, buggy, atol=1e-3)


def test_scaled_scores_have_unit_variance():
    """Il motivo dello scaling: con q, k a varianza 1 i punteggi restano a varianza ≈ 1."""
    torch.manual_seed(0)
    hs = 16
    q, k = torch.randn(64, 32, hs), torch.randn(64, 32, hs)
    raw = (q @ k.transpose(-2, -1)).var().item()
    scaled = (q @ k.transpose(-2, -1) * hs**-0.5).var().item()
    assert raw == pytest.approx(hs, rel=0.15)
    assert scaled == pytest.approx(1.0, rel=0.15)


def test_tril_is_a_buffer_not_a_parameter(small_config):
    head = Head(small_config)
    assert "tril" in dict(head.named_buffers())
    assert all("tril" not in name for name, _ in head.named_parameters())
    assert "tril" not in head.state_dict()  # non persistente: non finisce nel checkpoint


def test_positions_are_created_on_the_input_device(small_config, monkeypatch):
    """BUG: l'originale creava le posizioni con la variabile globale `device`.

    Su una macchina con GPU, un modello spostato su un device diverso da quella
    variabile falliva. Senza GPU non possiamo provocare l'errore, quindi
    "spiamo" torch.arange e verifichiamo che usi il device dell'input.
    Usiamo il device "meta" (tensori senza dati) come device diverso dalla CPU.
    """
    import gpt.model

    seen = []
    real_arange = torch.arange

    def spy_arange(*args, **kwargs):
        seen.append(torch.device(kwargs.get("device", "cpu")))
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(gpt.model.torch, "arange", spy_arange)
    model = GPTLanguageModel(small_config).to("meta")
    logits, _ = model(torch.zeros((2, 4), dtype=torch.long, device="meta"))
    assert seen == [torch.device("meta")]
    assert logits.shape == (2, 4, small_config.vocab_size)
