"""Training, checkpoint e inventario dei pesi."""

import pytest
import torch

from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, train_val_split
from gpt.model import GPTLanguageModel
from gpt.training import load_checkpoint, save_checkpoint, train
from gpt.weights import count_by_prefix, edited_weight, next_token_distribution, weight_inventory

TEXT = "to be or not to be, that is the question. " * 40


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(TEXT)


@pytest.fixture
def tiny_model(tokenizer):
    return GPTLanguageModel(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=1))


@pytest.mark.slow
def test_training_lowers_the_loss(tiny_model, tokenizer):
    splits = train_val_split(torch.tensor(tokenizer.encode(TEXT)))
    cfg = TrainConfig(batch_size=8, max_iters=300, eval_interval=100, eval_iters=10)
    history = train(tiny_model, splits, 8, cfg, torch.device("cpu"), log=lambda _msg: None)
    assert [p.step for p in history] == [0, 100, 200, 299]
    assert history[-1].train_loss < history[0].train_loss - 0.5


def test_checkpoint_roundtrip(tmp_path, tiny_model, tokenizer):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, tiny_model, tokenizer)
    loaded, loaded_tok = load_checkpoint(path)
    idx = torch.tensor([tokenizer.encode("to be")])
    tiny_model.eval()
    assert torch.allclose(tiny_model(idx)[0], loaded(idx)[0])
    assert loaded_tok.chars == tokenizer.chars


def test_inventory_separates_learned_weights_from_fixed_buffers():
    model = GPTLanguageModel(GPTConfig())
    inventory = weight_inventory(model)
    learned = [w for w in inventory if w.learned]
    fixed = [w for w in inventory if not w.learned]
    assert sum(w.numel for w in learned) == 209_729
    assert fixed and all(w.name.endswith("tril") for w in fixed)
    assert len(fixed) == 4 * 4  # una maschera per head, 4 head × 4 blocchi
    assert not any("dropout" in w.name for w in inventory)  # il dropout non ha pesi


def test_parameter_counts_per_component():
    model = GPTLanguageModel(GPTConfig())
    assert count_by_prefix(model, "token_embedding_table") == 65 * 64
    assert count_by_prefix(model, "position_embedding_table") == 32 * 64
    assert count_by_prefix(model, "blocks.0.sa.heads") == 4 * 3 * 64 * 16
    assert count_by_prefix(model, "blocks.0.sa.proj") == 64 * 64 + 64
    assert count_by_prefix(model, "blocks.0.ffwd") == (64 * 256 + 256) + (256 * 64 + 64)
    assert count_by_prefix(model, "blocks.0.ln1") == 2 * 64
    assert count_by_prefix(model, "blocks.0.") == 49_792
    assert count_by_prefix(model, "lm_head") == 64 * 65 + 65


def test_editing_one_bias_boosts_that_character_and_is_reverted(tiny_model, tokenizer):
    target = "q"
    token_id = tokenizer.encode(target)[0]

    def prob_of_target() -> float:
        dist = dict(next_token_distribution(tiny_model, tokenizer, "to be", top_k=tokenizer.vocab_size))
        return dist[target]

    before = prob_of_target()
    with edited_weight(tiny_model, "lm_head.bias", (token_id,), 20.0):
        during = prob_of_target()
    after = prob_of_target()

    assert during > 0.99 > before
    assert after == pytest.approx(before)
