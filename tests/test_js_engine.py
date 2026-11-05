"""Il motore JavaScript (webapp/js/engine) deve calcolare le stesse cose di PyTorch.

Il test lancia ``node webapp/tests/engine_fixture.mjs``, che carica gli stessi pesi esportati
(``webapp/data/weights.*``) ed esegue forward, logit lens e distribuzione del prossimo token
su vari prompt. Qui si rifà tutto con i moduli PyTorch di ``gpt/`` e si confronta
attivazione per attivazione. Lo stesso confronto si ripete con i pesi di alcuni snapshot
float16 del training (in PyTorch convertiti float16 → float32, come fa il motore).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gpt.export import EXAMPLE_PROMPT, PROJECT_ROOT, TRAINING_DIR, TRAINING_TRACE, WEBAPP_DATA_DIR, WEIGHTS_MANIFEST
from gpt.model import GPTLanguageModel
from gpt.training import load_checkpoint

NODE = shutil.which("node")
FIXTURE = PROJECT_ROOT / "webapp" / "tests" / "engine_fixture.mjs"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "gpt.pt"
MANIFEST = WEBAPP_DATA_DIR / WEIGHTS_MANIFEST
TRACE = WEBAPP_DATA_DIR / TRAINING_DIR / TRAINING_TRACE

# Errore massimo ammesso. Pesi identici, calcolo in float32 (PyTorch) contro float64 con
# risultati salvati in float32 (JS): le differenze osservate sono intorno a 1e-6.
ATOL = 1e-4
PROB_ATOL = 1e-5

CASES = [
    {"text": "\n"},
    {"text": "A"},
    {"text": "ROMEO:"},
    {"text": EXAMPLE_PROMPT[:32]},  # esattamente block_size, con un a capo
    {"text": "First Citizen:\nBefore we\nproceed"},  # 32 caratteri, due a capo
    {"text": "hi\n\nyo", "temperature": 0.8, "topK": 5},
    # Più lungo di block_size: solo la distribuzione, sugli ultimi 32 token come in generation.py.
    {"text": EXAMPLE_PROMPT, "temperature": 1.3, "topK": 10},
]

pytestmark = [
    pytest.mark.skipif(NODE is None, reason="node non è nel PATH: serve Node 22 per confrontare il motore JS con PyTorch"),
    pytest.mark.skipif(
        not (CHECKPOINT.exists() and MANIFEST.exists()),
        reason="mancano checkpoints/gpt.pt o webapp/data/weights.json: lancia train.py e l'export",
    ),
]


def run_engine(snapshot: int | None) -> dict:
    request = {"dataDir": str(WEBAPP_DATA_DIR), "snapshot": snapshot, "cases": CASES}
    proc = subprocess.run(
        [NODE, str(FIXTURE)], input=json.dumps(request), capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, f"il motore JS è fallito:\n{proc.stderr}"
    return json.loads(proc.stdout)


def snapshot_model(step: int) -> GPTLanguageModel:
    """Modello PyTorch con i pesi dello snapshot float16, letti con numpy (indipendente dal JS)."""
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    entry = next(s for s in trace["snapshots"] if s["step"] == step)
    flat = np.fromfile(WEBAPP_DATA_DIR / TRAINING_DIR / entry["file"], dtype="<f2").astype(np.float32)
    net, _ = load_checkpoint(CHECKPOINT)
    params = dict(net.named_parameters())
    with torch.no_grad():
        for p in trace["params"]:
            values = flat[p["offset"] : p["offset"] + p["numel"]].reshape(p["shape"])
            params[p["name"]].copy_(torch.from_numpy(values.copy()))
    return net.eval()


@torch.no_grad()
def reference_trace(net: GPTLanguageModel, ids: list[int]) -> dict:
    """Le stesse attivazioni del ForwardTrace JS, calcolate chiamando i moduli PyTorch."""
    T = len(ids)
    idx = torch.tensor([ids])
    tok = net.token_embedding_table(idx)[0]
    pos = net.position_embedding_table(torch.arange(T))
    x = tok + pos
    out = {"tokEmb": tok, "posEmb": pos, "x0": x, "blocks": []}
    for blk in net.blocks:
        ln1 = blk.ln1(x)
        heads = []
        for head in blk.sa.heads:
            q, k = head.query(ln1), head.key(ln1)
            scores = (q @ k.T * head.head_size**-0.5).masked_fill(head.tril[:T, :T] == 0, float("-inf"))
            heads.append(
                {
                    "q": q,
                    "k": k,
                    "v": head.value(ln1),
                    "scores": scores,
                    "att": head.attention_weights(ln1[None])[0],
                    "out": head(ln1[None])[0],
                }
            )
        concat = torch.cat([h["out"] for h in heads], dim=-1)
        proj = blk.sa.proj(concat)
        resid1 = x + proj
        ln2 = blk.ln2(resid1)
        ffn_pre = blk.ffwd.net[0](ln2)
        ffn_post = blk.ffwd.net[1](ffn_pre)
        ffn_out = blk.ffwd.net[2](ffn_post)
        resid2 = resid1 + ffn_out
        # La scomposizione a mano deve coincidere col blocco vero (altrimenti il test non vale nulla).
        torch.testing.assert_close(resid2, blk(x[None])[0], atol=1e-6, rtol=0)
        out["blocks"].append(
            {"ln1": ln1, "heads": heads, "concat": concat, "proj": proj, "resid1": resid1, "ln2": ln2,
             "ffnPre": ffn_pre, "ffnPost": ffn_post, "ffnOut": ffn_out, "resid2": resid2}
        )
        x = resid2
    out["lnF"] = net.ln_f(x)
    out["logits"] = net.lm_head(out["lnF"])
    torch.testing.assert_close(out["logits"], net(idx)[0][0], atol=1e-6, rtol=0)
    return out


def reference_next(net: GPTLanguageModel, ids: list[int], temperature: float, top_k: int | None):
    with torch.no_grad():
        logits = net(torch.tensor([ids[-net.config.block_size :]]))[0][0, -1]
    scaled = logits / temperature
    if top_k is not None:
        threshold = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    return logits, F.softmax(scaled, dim=-1)


def max_error(js_values, expected: torch.Tensor) -> float:
    actual = torch.tensor(np.asarray(js_values, dtype=np.float64)).reshape(expected.shape)
    return (actual - expected.double()).abs().max().item()


def assert_close(js_values, expected: torch.Tensor, what: str, atol: float = ATOL) -> None:
    err = max_error(js_values, expected)
    assert err < atol, f"{what}: errore massimo {err:.3g} (ammesso {atol})"


def assert_scores(js_scores, expected: torch.Tensor, what: str) -> None:
    """Punteggi dell'attention: sopra la diagonale −∞ (null nel JSON), sotto uguali a PyTorch."""
    T = expected.shape[0]
    masked = ~torch.tril(torch.ones(T, T, dtype=torch.bool))
    js = np.asarray([np.nan if v is None else v for v in js_scores], dtype=np.float64).reshape(T, T)
    assert np.isnan(js[masked.numpy()]).all(), f"{what}: il futuro non è mascherato"
    js_tensor = torch.tensor(np.where(np.isnan(js), 0.0, js))
    err = (js_tensor - expected.double().masked_fill(masked, 0.0)).abs().max().item()
    assert err < ATOL, f"{what}: errore massimo {err:.3g}"


def check_against_pytorch(output: dict, net: GPTLanguageModel) -> None:
    assert output["config"]["n_layer"] == net.config.n_layer
    for case, result in zip(CASES, output["results"], strict=True):
        name = repr(case["text"][:12])
        ids = result["ids"]
        temperature, top_k = case.get("temperature", 1.0), case.get("topK")

        logits, probs = reference_next(net, ids, temperature, top_k)
        assert_close(result["next"]["logits"], logits, f"{name} logits del prossimo token")
        assert_close(result["next"]["probs"], probs, f"{name} probabilità", PROB_ATOL)
        if top_k is not None:
            assert sum(p > 0 for p in result["next"]["probs"]) == top_k

        if len(ids) > net.config.block_size:
            assert "forward" not in result
            continue
        ref = reference_trace(net, ids)
        js = result["forward"]["trace"]
        assert js["ids"] == ids
        assert_close(result["forward"]["logits"], ref["logits"], f"{name} logits")
        assert_close(js["logits"], ref["logits"], f"{name} trace.logits")
        for key in ("tokEmb", "posEmb", "x0", "lnF"):
            assert_close(js[key], ref[key], f"{name} {key}")
        for b, (js_block, ref_block) in enumerate(zip(js["blocks"], ref["blocks"], strict=True)):
            for key in ("ln1", "concat", "proj", "resid1", "ln2", "ffnPre", "ffnPost", "ffnOut", "resid2"):
                assert_close(js_block[key], ref_block[key], f"{name} blocco {b} {key}")
            for h, (js_head, ref_head) in enumerate(zip(js_block["heads"], ref_block["heads"], strict=True)):
                for key in ("q", "k", "v", "att", "out"):
                    assert_close(js_head[key], ref_head[key], f"{name} blocco {b} head {h} {key}")
                assert_scores(js_head["scores"], ref_head["scores"], f"{name} blocco {b} head {h} scores")

        # Logit lens: ln_f + lm_head sul residual stream dopo l'embedding e dopo ogni blocco.
        streams = [ref["x0"]] + [blk["resid2"] for blk in ref["blocks"]]
        assert len(result["lens"]) == len(streams)
        with torch.no_grad():
            for layer, (js_probs, x) in enumerate(zip(result["lens"], streams, strict=True)):
                expected = F.softmax(net.lm_head(net.ln_f(x)), dim=-1)
                assert_close(js_probs, expected, f"{name} logit lens strato {layer}", PROB_ATOL)


def test_js_engine_matches_pytorch_on_trained_weights():
    net, _ = load_checkpoint(CHECKPOINT)
    output = run_engine(snapshot=None)
    assert output["weightsSource"] == "trained"
    check_against_pytorch(output, net.eval())


@pytest.mark.skipif(not TRACE.exists(), reason="manca webapp/data/training/trace.json: lancia scripts/record_training.py")
@pytest.mark.parametrize("step", [0, 1000])
def test_js_engine_matches_pytorch_on_float16_snapshot(step):
    output = run_engine(snapshot=step)
    assert output["weightsSource"] == "snapshot" and output["snapshotStep"] == step
    check_against_pytorch(output, snapshot_model(step))


def test_fixture_reports_engine_errors():
    """Se il motore fallisce (carattere fuori vocabolario), il ponte esce con errore e lo spiega."""
    request = {"dataDir": str(WEBAPP_DATA_DIR), "snapshot": None, "cases": [{"text": "ciao €"}]}
    proc = subprocess.run([NODE, str(FIXTURE)], input=json.dumps(request), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1
    assert "non presenti nel vocabolario" in proc.stderr and "€" in proc.stderr
