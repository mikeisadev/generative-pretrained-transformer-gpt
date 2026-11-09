"""Callback del training e registrazione della sua storia (gpt/recording.py)."""

import json
import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, get_batch, train_val_split
from gpt.export import EXAMPLE_PROMPT, TRAINING_DIR, TRAINING_TRACE, WEBAPP_DATA_DIR, WEIGHTS_BINARY
from gpt.model import GPTLanguageModel
from gpt.recording import RecorderConfig, TrainingRecorder, group_of, parameter_groups
from gpt.training import EvalPoint, TrainingCallback, estimate_loss, train

# Il testo contiene il prompt d'esempio della web app, che il recorder deve saper codificare.
TEXT = (EXAMPLE_PROMPT + " to be or not to be, that is the question. ") * 20
BLOCK = 8
CPU = torch.device("cpu")
TRAIN_CFG = TrainConfig(batch_size=8, max_iters=60, eval_interval=20, eval_iters=5)
REC_CFG = RecorderConfig(log_every=10, snapshot_steps=(0, 10, 30, 1000), sample_length=40, eval_batch_size=16)


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(TEXT)


@pytest.fixture
def splits(tokenizer):
    return train_val_split(torch.tensor(tokenizer.encode(TEXT)))


def new_model(tokenizer, seed=1337):
    torch.manual_seed(seed)
    return GPTLanguageModel(
        GPTConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK, n_embd=16, n_head=2, n_layer=2)
    )


def reference_train(model, splits, cfg):
    """Il ciclo di train() com'era PRIMA dei callback: il comportamento da non cambiare."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    history = []
    model.train()
    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model, splits, BLOCK, cfg, CPU)
            history.append(EvalPoint(step, losses["train"], losses["val"]))
        xb, yb = get_batch(splits.train, BLOCK, cfg.batch_size, CPU)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


def run(tokenizer, splits, callbacks=None, use_reference=False):
    """Training completo da seme fisso; restituisce storia, pesi finali e stato del RNG globale."""
    model = new_model(tokenizer)
    if use_reference:
        history = reference_train(model, splits, TRAIN_CFG)
    else:
        history = train(model, splits, BLOCK, TRAIN_CFG, CPU, log=lambda _msg: None, callbacks=callbacks)
    return history, model.state_dict(), torch.get_rng_state()


class LossSpy(TrainingCallback):
    """Callback minimo: annota la loss e l'ordine in cui arrivano gli hook."""

    def __init__(self):
        self.losses, self.calls = [], []

    def on_train_start(self, model, optimizer, train_cfg):
        self.calls.append("start")

    def on_eval(self, step, losses):
        self.calls.append(f"eval{step}")

    def on_after_backward(self, step, model, loss):
        assert all(p.grad is not None for p in model.parameters())
        self.losses.append(loss.item())
        self.calls.append(f"backward{step}")

    def on_step_end(self, step, model, loss, optimizer):
        self.calls.append(f"step{step}")

    def on_train_end(self, model, history):
        self.calls.append("end")


def assert_same_run(a, b):
    history_a, weights_a, rng_a = a
    history_b, weights_b, rng_b = b
    assert history_a == history_b  # loss identiche, non solo vicine
    assert weights_a.keys() == weights_b.keys()
    assert all(torch.equal(weights_a[k], weights_b[k]) for k in weights_a)
    assert torch.equal(rng_a, rng_b)  # il generatore globale ha fatto esattamente la stessa strada


def test_train_without_callbacks_matches_the_original_loop(tokenizer, splits):
    assert_same_run(run(tokenizer, splits, use_reference=True), run(tokenizer, splits, callbacks=None))


@pytest.mark.parametrize("make_callbacks", [lambda: [], lambda: [object()], lambda: [TrainingCallback()]])
def test_no_op_callbacks_do_not_change_the_training(tokenizer, splits, make_callbacks):
    assert_same_run(run(tokenizer, splits), run(tokenizer, splits, callbacks=make_callbacks()))


def test_recorder_does_not_change_the_training(tokenizer, splits):
    recorder = TrainingRecorder(tokenizer, splits.val, BLOCK, REC_CFG)
    assert_same_run(run(tokenizer, splits), run(tokenizer, splits, callbacks=[recorder, LossSpy()]))


def test_hooks_are_called_in_order(tokenizer, splits):
    spy = LossSpy()
    run(tokenizer, splits, callbacks=[spy])
    assert spy.calls[:5] == ["start", "eval0", "backward0", "step0", "backward1"]
    assert spy.calls[-4:] == ["eval59", "backward59", "step59", "end"]
    assert len(spy.losses) == TRAIN_CFG.max_iters


@pytest.fixture
def recorded(tokenizer, splits, tmp_path):
    model = new_model(tokenizer)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    recorder = TrainingRecorder(tokenizer, splits.val, BLOCK, REC_CFG)
    spy = LossSpy()
    history = train(model, splits, BLOCK, TRAIN_CFG, CPU, log=lambda _msg: None, callbacks=[recorder, spy])
    path = recorder.save(tmp_path)
    trace = json.loads(path.read_text(encoding="utf-8"))
    return {"model": model, "initial": initial, "history": history, "spy": spy, "trace": trace, "dir": tmp_path}


def all_numbers(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from all_numbers(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from all_numbers(value)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


def read_snapshot(path, layout):
    flat = np.fromfile(path, dtype="<f2")
    return {p["name"]: torch.from_numpy(flat[p["offset"] : p["offset"] + p["numel"]].reshape(p["shape"]).copy())
            for p in layout}


def test_trace_has_the_documented_keys(recorded):
    trace = recorded["trace"]
    assert set(trace) == {
        "format_version", "config", "train_config", "total_params", "vocab", "device", "duration_s",
        "recorder", "groups", "group_params", "snapshot_dtype", "byte_order", "params",
        "train_loss", "evals", "metrics", "snapshots",
    }
    assert trace["train_config"]["max_iters"] == TRAIN_CFG.max_iters
    assert trace["total_params"] == recorded["model"].num_parameters()
    assert all(math.isfinite(x) for x in all_numbers(trace))


def test_losses_and_evals_are_the_real_ones(recorded):
    trace = recorded["trace"]
    assert trace["train_loss"] == pytest.approx(recorded["spy"].losses, abs=1e-5)
    evals = trace["evals"]
    assert evals["steps"] == [p.step for p in recorded["history"]] == [0, 20, 40, 59]
    assert evals["val"] == pytest.approx([p.val_loss for p in recorded["history"]], abs=1e-5)


def test_metrics_have_one_value_per_logged_step(recorded):
    trace = recorded["trace"]
    metrics = trace["metrics"]
    assert metrics["steps"] == [0, 10, 20, 30, 40, 50, 59]
    n = len(metrics["steps"])
    for key in ("lr", "grad_norm", "update_ratio"):
        assert len(metrics[key]) == n
    for key in ("grad_norm_by_group", "update_ratio_by_group", "weight_std_by_group"):
        assert list(metrics[key]) == trace["groups"]
        assert all(len(values) == n for values in metrics[key].values())
    assert metrics["lr"] == [TRAIN_CFG.learning_rate] * n
    # la norma globale è la radice della somma dei quadrati delle norme dei gruppi
    for i in range(n):
        by_group = math.sqrt(sum(v[i] ** 2 for v in metrics["grad_norm_by_group"].values()))
        assert metrics["grad_norm"][i] == pytest.approx(by_group, rel=1e-4)
    assert all(0 < r < 1 for r in metrics["update_ratio"])


def test_groups_cover_every_parameter_once(recorded):
    trace = recorded["trace"]
    names = [n for group in trace["groups"] for n in trace["group_params"][group]]
    assert sorted(names) == sorted(n for n, _ in recorded["model"].named_parameters())
    assert trace["groups"][:3] == ["token_embedding", "position_embedding", "blocks.0.attention"]
    assert trace["groups"][-2:] == ["ln_f", "lm_head"]


def test_group_of_keeps_the_course_groups_and_never_rejects_a_name():
    assert group_of("blocks.3.sa.heads.1.query.weight") == "blocks.3.attention"
    assert group_of("blocks.0.sa.proj.bias") == "blocks.0.attention"
    assert group_of("blocks.1.ln2.weight") == "blocks.1.layernorm"
    assert group_of("blocks.1.ffwd.net.2.weight") == "blocks.1.ffwd"
    assert group_of("lm_head.bias") == "lm_head"
    # Un nome che il GPT del corso non ha va nel suo modulo (prima fermava la registrazione).
    assert group_of("decoder.weight") == "decoder"
    assert group_of("blocks.1.norm1.weight") == "blocks.1.norm1"
    assert len(parameter_groups(GPTLanguageModel(GPTConfig()))) == 2 + 3 * 4 + 2


def test_snapshots_are_float16_copies_of_the_weights(recorded):
    trace, model = recorded["trace"], recorded["model"]
    # 1000 > max_iters viene scartato; lo step finale (60 = pesi dopo l'ultimo update) c'è sempre
    assert [s["step"] for s in trace["snapshots"]] == [0, 10, 30, 60]
    assert trace["recorder"]["snapshot_steps"] == [0, 10, 30, 60]
    for snap in trace["snapshots"]:
        path = recorded["dir"] / snap["file"]
        assert path.stat().st_size == 2 * trace["total_params"]
    first = read_snapshot(recorded["dir"] / trace["snapshots"][0]["file"], trace["params"])
    last = read_snapshot(recorded["dir"] / trace["snapshots"][-1]["file"], trace["params"])
    for name, p in model.named_parameters():
        assert torch.equal(first[name], recorded["initial"][name].half())
        assert torch.equal(last[name], p.detach().half())


def test_snapshot_measurements(recorded):
    trace, cfg = recorded["trace"], recorded["model"].config
    for snap in trace["snapshots"]:
        assert len(snap["sample_text"]) == REC_CFG.sample_length
        assert set(snap["sample_text"]) <= set(trace["vocab"])
        probs = snap["example_next_token_probs"]
        assert len(probs) == cfg.vocab_size and sum(probs) == pytest.approx(1.0, abs=1e-3)
        assert 0 <= snap["top1"] <= snap["top5"] <= 1
        assert 0 < snap["entropy"] <= math.log(cfg.vocab_size) + 1e-4
        assert len(snap["attention_entropy"]) == cfg.n_layer
        # un token che guarda al massimo BLOCK token ha entropia tra 0 e ln(BLOCK)
        assert all(len(row) == cfg.n_head for row in snap["attention_entropy"])
        assert all(0 <= e <= math.log(BLOCK) for row in snap["attention_entropy"] for e in row)
    # il modello migliora: il batch fisso è sempre lo stesso, quindi i numeri sono confrontabili
    assert trace["snapshots"][-1]["loss"] < trace["snapshots"][0]["loss"]


def test_samples_are_reproducible(tokenizer, splits):
    def sample_texts():
        recorder = TrainingRecorder(tokenizer, splits.val, BLOCK, REC_CFG)
        train(new_model(tokenizer), splits, BLOCK, TRAIN_CFG, CPU, log=lambda _msg: None, callbacks=[recorder])
        return [s["sample_text"] for s in recorder.snapshots]

    assert sample_texts() == sample_texts()


def test_save_removes_stale_snapshots(recorded, tokenizer, splits):
    stale = recorded["dir"] / "snap_99999.f16.bin"
    stale.write_bytes(b"vecchio")
    recorder = TrainingRecorder(tokenizer, splits.val, BLOCK, REC_CFG)
    train(new_model(tokenizer), splits, BLOCK, TRAIN_CFG, CPU, log=lambda _msg: None, callbacks=[recorder])
    recorder.save(recorded["dir"])
    assert not stale.exists()
    assert len(list(recorded["dir"].glob("snap_*.f16.bin"))) == len(recorder.snapshots)


def test_save_replaces_links_instead_of_writing_through_them(tokenizer, splits, tmp_path):
    outside = tmp_path / "fuori.txt"
    outside.write_text("da non toccare", encoding="utf-8")
    out = tmp_path / "training"
    out.mkdir()
    try:
        (out / "trace.json").symlink_to(outside)
        (out / "snap_00000.f16.bin").symlink_to(outside)
    except OSError:  # Windows senza i permessi per i collegamenti simbolici
        pytest.skip("il file system non permette collegamenti simbolici")
    (out / "snap_00007.f16.bin").mkdir()  # una cartella con il nome di uno snapshot non si tocca
    recorder = TrainingRecorder(tokenizer, splits.val, BLOCK, REC_CFG)
    train(new_model(tokenizer), splits, BLOCK, TRAIN_CFG, CPU, log=lambda _msg: None, callbacks=[recorder])
    recorder.save(out)
    assert outside.read_text(encoding="utf-8") == "da non toccare"  # niente scritto fuori dalla cartella
    assert not (out / "trace.json").is_symlink() and json.loads((out / "trace.json").read_text(encoding="utf-8"))
    assert not (out / "snap_00000.f16.bin").is_symlink()
    assert (out / "snap_00007.f16.bin").is_dir()
    assert not list(out.glob(".*.tmp"))  # nessun file temporaneo rimasto


def test_recorder_without_training_cannot_be_saved(tokenizer, splits, tmp_path):
    with pytest.raises(RuntimeError):
        TrainingRecorder(tokenizer, splits.val, BLOCK).save(tmp_path)


def test_invalid_recorder_config_is_rejected():
    with pytest.raises(ValueError):
        RecorderConfig(log_every=0)
    with pytest.raises(ValueError):
        RecorderConfig(snapshot_steps=(-1,))


# --- dati committati della web app ---------------------------------------------------------

TRACE_PATH = WEBAPP_DATA_DIR / TRAINING_DIR / TRAINING_TRACE


@pytest.fixture(scope="module")
def committed_trace():
    if not TRACE_PATH.exists():
        pytest.skip("webapp/data/training/trace.json non ancora generato (scripts/record_training.py)")
    return json.loads(TRACE_PATH.read_text(encoding="utf-8"))


def test_committed_trace_is_the_reference_training(committed_trace):
    trace = committed_trace
    assert trace["config"] == GPTConfig().to_dict()
    assert trace["train_config"] == asdict(TrainConfig())
    assert len(trace["train_loss"]) == TrainConfig().max_iters
    assert trace["evals"]["steps"][-1] == TrainConfig().max_iters - 1
    for snap in trace["snapshots"]:
        assert (TRACE_PATH.parent / snap["file"]).stat().st_size == 2 * trace["total_params"]
    assert all(math.isfinite(x) for x in all_numbers(trace))


def test_committed_final_snapshot_matches_the_exported_weights(committed_trace):
    weights = WEBAPP_DATA_DIR / WEIGHTS_BINARY
    if not weights.exists():
        pytest.skip("weights.f32.bin non ancora generato")
    final = committed_trace["snapshots"][-1]
    assert final["step"] == committed_trace["train_config"]["max_iters"]
    f32 = np.fromfile(weights, dtype="<f4")
    f16 = np.fromfile(TRACE_PATH.parent / final["file"], dtype="<f2")
    assert np.array_equal(f32.astype("<f2"), f16)
