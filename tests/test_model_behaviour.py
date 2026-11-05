"""Loss iniziale, generazione e overfitting."""

import math

import pytest
import torch

from gpt.bigram import BigramLanguageModel
from gpt.config import GPTConfig
from gpt.model import GPTLanguageModel


def test_initial_loss_is_close_to_ln_vocab_size():
    """Un modello appena nato non sa nulla: probabilità ≈ uniformi → loss ≈ ln(V)."""
    config = GPTConfig()  # V = 65
    model = GPTLanguageModel(config)
    idx = torch.randint(config.vocab_size, (16, config.block_size))
    targets = torch.randint(config.vocab_size, (16, config.block_size))
    _, loss = model(idx, targets)
    assert loss.item() == pytest.approx(math.log(config.vocab_size), abs=0.3)


@pytest.mark.parametrize("prompt_len,new_tokens", [(1, 0), (1, 10), (3, 40)])
def test_generate_output_length(small_model, small_config, prompt_len, new_tokens):
    """Anche oltre block_size (40 > 8): il contesto viene tagliato, la sequenza no."""
    idx = torch.zeros((2, prompt_len), dtype=torch.long)
    out = small_model.generate(idx, max_new_tokens=new_tokens)
    assert out.shape == (2, prompt_len + new_tokens)
    assert torch.equal(out[:, :prompt_len], idx)
    assert int(out.max()) < small_config.vocab_size


def test_bigram_generate_output_length():
    model = BigramLanguageModel(vocab_size=5)
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)


def test_generate_runs_in_eval_mode_without_gradients_then_restores_mode():
    """BUG: l'originale generava in modalità training (dropout attivo) costruendo il grafo dei gradienti.

    Un forward hook registra, a ogni passo, lo stato del modello durante la generazione.
    """
    model = GPTLanguageModel(GPTConfig(vocab_size=11, block_size=8, n_embd=16, n_head=2, n_layer=1, dropout=0.5))
    model.train()
    observed = []
    model.register_forward_hook(lambda m, _in, _out: observed.append((m.training, torch.is_grad_enabled())))

    model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)

    assert observed == [(False, False)] * 5, "ogni passo: dropout spento e nessun gradiente"
    assert model.training, "alla fine il modello torna nella modalità in cui era"


def test_temperature_must_be_positive(small_model):
    with pytest.raises(ValueError):
        small_model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1, temperature=0)


@pytest.mark.slow
def test_model_can_overfit_a_single_batch(small_config):
    """Se il modello non riesce a memorizzare 4 sequenze, c'è un bug nel flusso dei gradienti."""
    model = GPTLanguageModel(small_config)
    x = torch.randint(small_config.vocab_size, (4, small_config.block_size))
    y = torch.randint(small_config.vocab_size, (4, small_config.block_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    _, initial = model(x, y)
    for _ in range(400):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert initial.item() > 2.0
    assert loss.item() < 0.1
