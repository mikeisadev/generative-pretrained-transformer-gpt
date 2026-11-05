"""Tokenizer, batch e download del dataset (senza rete: urlopen viene simulato)."""

import io

import pytest
import torch

from gpt import data as data_module
from gpt.data import CharTokenizer, get_batch, train_val_split


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    assert tok.vocab_size == len(set("hello world"))
    assert tok.decode(tok.encode("low old")) == "low old"


def test_tokenizer_rejects_unknown_characters():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="non presenti"):
        tok.encode("abz")


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_train_val_split_is_90_10():
    splits = train_val_split(torch.arange(1000))
    assert len(splits.train) == 900 and len(splits.val) == 100


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_is_skipped_when_file_is_valid(tmp_path, monkeypatch):
    payload = b"First Citizen:\n"
    monkeypatch.setattr(data_module, "DATASET_SHA256", data_module.hashlib.sha256(payload).hexdigest())
    (tmp_path / "input.txt").write_bytes(payload)

    def fail(*_args, **_kwargs):
        raise AssertionError("non doveva scaricare")

    monkeypatch.setattr(data_module.urllib.request, "urlopen", fail)
    assert data_module.download_tiny_shakespeare(tmp_path).read_bytes() == payload


def test_download_replaces_corrupted_file(tmp_path, monkeypatch):
    """BUG: `!wget` non è Python. Ora il download è automatico e verificato."""
    payload = b"First Citizen:\n"
    monkeypatch.setattr(data_module, "DATASET_SHA256", data_module.hashlib.sha256(payload).hexdigest())
    (tmp_path / "input.txt").write_bytes(b"file troncato")
    monkeypatch.setattr(data_module.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))
    assert data_module.download_tiny_shakespeare(tmp_path).read_bytes() == payload


def test_download_rejects_tampered_content(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"malevolo"))
    with pytest.raises(RuntimeError, match="impronta"):
        data_module.download_tiny_shakespeare(tmp_path)
    assert not list(tmp_path.iterdir()), "nessun file parziale deve restare su disco"
