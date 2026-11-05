"""Export dei pesi completi float32 per il motore JavaScript (weights.json + weights.f32.bin)."""

import json

import numpy as np
import pytest
import torch

from gpt.config import GPTConfig
from gpt.data import CharTokenizer
from gpt.export import (
    EXAMPLE_PROMPT,
    PROJECT_ROOT,
    TRAINING_DIR,
    TRAINING_TRACE,
    WEBAPP_DATA_DIR,
    WEIGHTS_BINARY,
    WEIGHTS_MANIFEST,
    export,
    export_weights,
    parameter_layout,
)
from gpt.model import GPTLanguageModel
from gpt.training import load_checkpoint, save_checkpoint

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "gpt.pt"


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(EXAMPLE_PROMPT + " abcdefghij")


@pytest.fixture
def model(tokenizer):
    return GPTLanguageModel(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=2))


def read_tensors(manifest, binary_path):
    """Ricostruzione indipendente dal codice di export: solo numpy, offset e forme del manifest."""
    flat = np.fromfile(binary_path, dtype="<f4")
    return {
        p["name"]: torch.from_numpy(flat[p["offset"] : p["offset"] + p["numel"]].reshape(p["shape"]).copy())
        for p in manifest["params"]
    }


def assert_matches_state_dict(manifest, binary_path, state_dict):
    assert binary_path.stat().st_size == 4 * manifest["total_params"]
    tensors = read_tensors(manifest, binary_path)
    assert tensors.keys() == state_dict.keys()  # la maschera tril non è nello state_dict
    for name, value in state_dict.items():
        assert torch.equal(tensors[name], value.cpu()), name  # uguali bit a bit, non "vicini"


def test_export_writes_every_weight_exactly(tmp_path, model, tokenizer):
    manifest = export_weights(model, tokenizer, tmp_path, source="test", trained=False)
    on_disk = json.loads((tmp_path / WEIGHTS_MANIFEST).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["total_params"] == model.num_parameters()
    assert_matches_state_dict(manifest, tmp_path / manifest["file"], model.state_dict())


def test_manifest_layout(tmp_path, model, tokenizer):
    manifest = export_weights(model, tokenizer, tmp_path, source="test", trained=False)
    assert {k: manifest[k] for k in ("format_version", "dtype", "byte_order", "file")} == {
        "format_version": 1, "dtype": "float32", "byte_order": "little", "file": WEIGHTS_BINARY,
    }
    assert manifest["config"] == model.config.to_dict()
    assert manifest["vocab"] == tokenizer.chars
    names = [p["name"] for p in manifest["params"]]
    assert names == [n for n, _ in model.named_parameters()]
    offset = 0
    for p in manifest["params"]:  # tensori contigui, senza buchi né sovrapposizioni
        assert p["offset"] == offset and p["numel"] == int(np.prod(p["shape"]))
        offset += p["numel"]
    assert offset == manifest["total_params"]


def test_manifest_points_to_the_training_trace_only_if_it_exists(tmp_path, model, tokenizer):
    assert export_weights(model, tokenizer, tmp_path, source="test", trained=False)["training"] is None
    (tmp_path / TRAINING_DIR).mkdir()
    (tmp_path / TRAINING_DIR / TRAINING_TRACE).write_text("{}", encoding="utf-8")
    training = export_weights(model, tokenizer, tmp_path, source="test", trained=False)["training"]
    assert training == {"dir": "training", "trace": "training/trace.json"}


def test_export_from_checkpoint_also_writes_the_full_weights(tmp_path, model, tokenizer):
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, tokenizer)
    export(ckpt, out=tmp_path / "web" / "model.json")
    manifest = json.loads((tmp_path / "web" / WEIGHTS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["trained"] is True
    assert_matches_state_dict(manifest, tmp_path / "web" / WEIGHTS_BINARY, model.state_dict())


def test_layout_of_the_standard_model():
    layout = parameter_layout(GPTLanguageModel(GPTConfig()))
    assert layout[0] == {"name": "token_embedding_table.weight", "shape": [65, 64], "offset": 0, "numel": 4160}
    assert layout[-1]["offset"] + layout[-1]["numel"] == 209_729


# --- file committati della web app -----------------------------------------------------------


@pytest.fixture(scope="module")
def committed():
    path = WEBAPP_DATA_DIR / WEIGHTS_MANIFEST
    if not path.exists():
        pytest.skip("webapp/data/weights.json non ancora generato: rilancia python scripts/serve_webapp.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_manifest_matches_the_code(committed):
    fresh = GPTLanguageModel(GPTConfig(vocab_size=len(committed["vocab"])))
    assert committed["params"] == parameter_layout(fresh), "rilancia python scripts/serve_webapp.py"
    assert (WEBAPP_DATA_DIR / WEIGHTS_BINARY).stat().st_size == 4 * committed["total_params"]
    trace_exists = (WEBAPP_DATA_DIR / TRAINING_DIR / TRAINING_TRACE).exists()
    assert (committed["training"] is not None) == trace_exists


def test_committed_weights_are_the_checkpoint(committed):
    if not CHECKPOINT.exists() or not committed["trained"]:
        pytest.skip("serve il checkpoint addestrato (checkpoints/gpt.pt)")
    net, tokenizer = load_checkpoint(CHECKPOINT)
    assert committed["vocab"] == tokenizer.chars
    assert_matches_state_dict(committed, WEBAPP_DATA_DIR / WEIGHTS_BINARY, net.state_dict())
