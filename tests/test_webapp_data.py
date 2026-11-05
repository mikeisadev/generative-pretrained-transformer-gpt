"""La web app deve mostrare esattamente il codice e l'architettura del pacchetto `gpt`."""

import json

import pytest

from gpt.config import GPTConfig
from gpt.data import CharTokenizer
from gpt.export import SNIPPETS, WEBAPP_DATA, build_spec
from gpt.model import GPTLanguageModel


@pytest.fixture(scope="module")
def committed():
    if not WEBAPP_DATA.exists():
        pytest.skip("webapp/data/model.json non ancora generato")
    return json.loads(WEBAPP_DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh(committed):
    tokenizer = CharTokenizer(committed["vocab"])
    model = GPTLanguageModel(GPTConfig(vocab_size=tokenizer.vocab_size))
    return build_spec(model, tokenizer, source="test", trained=False)


def test_architecture_matches_the_code(committed, fresh):
    for key in ("config", "head_size", "total_params", "params", "buffers"):
        assert committed[key] == fresh[key], f"'{key}' non aggiornato: rilancia python scripts/serve_webapp.py"


def test_code_snippets_match_the_source(committed, fresh):
    assert set(committed["snippets"]) == set(SNIPPETS)
    for key, snippet in fresh["snippets"].items():
        assert committed["snippets"][key] == snippet, f"frammento '{key}' non aggiornato: rilancia l'export"


def test_every_tensor_is_exported_with_its_shape(committed, fresh):
    for param in fresh["params"]:
        tensor = committed["tensors"][param["name"]]
        expected = param["shape"] if len(param["shape"]) == 2 else [1, *param["shape"]]
        assert tensor["shape"] == expected


def test_example_has_one_attention_map_per_head(committed):
    cfg = committed["config"]
    assert len(committed["example"]["attention"]) == cfg["n_layer"] * cfg["n_head"]
    assert len(committed["example"]["next_token_probs"]) == cfg["vocab_size"]
    assert sum(committed["example"]["next_token_probs"]) == pytest.approx(1.0, abs=1e-3)
