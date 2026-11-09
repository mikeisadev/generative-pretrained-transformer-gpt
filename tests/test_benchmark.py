"""Benchmark deterministico (gpt/benchmark.py) e dati del benchmark per il browser (webapp/data/benchmark)."""

import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt.benchmark import (
    BENCHMARK_DATA_DIR,
    BUILTIN_FILE,
    EXCERPT_FILE,
    HITS_ATOL,
    LOSS_ATOL,
    PROTOCOL_ID,
    SURPRISAL_ATOL,
    ArchitectureMismatch,
    BenchmarkSettings,
    EvalText,
    benchmark_differences,
    benchmark_model,
    benchmark_path,
    build_model,
    comparison_rows,
    compute_benchmark_data,
    contiguous_windows,
    excerpt_bounds,
    format_comparison,
    format_summary,
    generation_sample,
    load_benchmark,
    open_source,
    run_tokenizer,
    score_tokens,
    validation_text,
)
from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, train_val_split
from gpt.export import EXAMPLE_PROMPT, PROJECT_ROOT, WEBAPP_DATA_DIR
from gpt.model import GPTLanguageModel
from gpt import benchmark
from gpt.recording import RecorderConfig, TrainingRecorder
from gpt.runs import (
    export_run,
    read_run_folder,
    read_training_history,
    snapshot_tensors,
    weights_from_model,
    weights_from_state_dict,
)
from gpt.training import save_checkpoint, train

TEXT = ("ROMEO:\n" + EXAMPLE_PROMPT + " to be or not to be, that is the question.\n\n") * 12
NODE = shutil.which("node")
BUILTIN = BENCHMARK_DATA_DIR / BUILTIN_FILE
EXCERPT = BENCHMARK_DATA_DIR / EXCERPT_FILE
DATASET = PROJECT_ROOT / "data" / "input.txt"
RERUN = "rilancia python scripts/export_benchmark_data.py"
# Pochi thread: su una macchina carica, più thread che core liberi rallentano di decine di volte.
THREADS = min(2, os.cpu_count() or 1)
QUICK = BenchmarkSettings(max_new_tokens=20, threads=THREADS)
TIMINGS = ("tokens_per_s", "eval_tokens_per_s", "duration_s")


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(TEXT)


def small_model(tokenizer, seed=1337):
    torch.manual_seed(seed)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=2)
    return GPTLanguageModel(config).eval()


@contextlib.contextmanager
def torch_threads(n):
    before = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def results(bench):
    """Il benchmark senza ciò che dipende dal momento o dalla macchina (date, tempi, ambiente)."""
    out = {k: v for k, v in bench.items() if k not in ("created", "environment")}
    out["metrics"] = {k: v for k, v in bench["metrics"].items() if k not in TIMINGS}
    if out.get("generation"):
        out["generation"] = {k: v for k, v in bench["generation"].items() if k != "tokens_per_s"}
    return out


# --------------------------------------------------------------------------- protocollo


@pytest.mark.parametrize(("n", "block"), [(2, 8), (9, 8), (10, 8), (50, 8), (111_540, 32)])
def test_contiguous_windows_predict_every_token_once(n, block):
    windows = contiguous_windows(n, block)
    targets = [s + 1 + t for s, length in windows for t in range(length)]
    assert targets == list(range(1, n))  # ognuno una volta, in ordine di testo
    assert all(1 <= length <= block for _, length in windows)
    assert all(length == block for _, length in windows[:-1])  # solo l'ultima può essere più corta
    assert [s for s, _ in windows] == list(range(0, n - 1, block))


def test_contiguous_windows_reject_impossible_inputs():
    with pytest.raises(ValueError):
        contiguous_windows(1, 8)
    with pytest.raises(ValueError):
        contiguous_windows(10, 0)


def test_scores_match_a_direct_computation(tokenizer):
    model = small_model(tokenizer)
    ids = torch.tensor(tokenizer.encode(TEXT[:51]))  # 50 previsioni: 6 finestre da 8 e una da 2
    nll, entropy, top1, top5, bits = 0.0, 0.0, 0, 0, []
    with torch.no_grad():
        for start, length in contiguous_windows(len(ids), 8):
            logits = model(ids[start : start + length][None])[0][0]
            target = ids[start + 1 : start + length + 1]
            losses = F.cross_entropy(logits, target, reduction="none")
            nll += losses.sum().item()
            bits += (losses / math.log(2)).tolist()
            probs = logits.softmax(-1)
            entropy += -(probs * probs.log()).sum().item()
            top1 += int((logits.argmax(-1) == target).sum())
            top5 += int((logits.topk(5).indices == target[:, None]).any(-1).sum())
    for batch in (1, 3, 512):  # i gruppi di finestre cambiano solo la velocità
        scores, surprisal = score_tokens(model, ids, block_size=8, batch_windows=batch, surprisal=True)
        assert (scores.predictions, scores.windows, scores.chars) == (50, 7, 50)
        assert scores.nll == pytest.approx(nll, abs=1e-4)
        assert scores.entropy == pytest.approx(entropy, abs=1e-4)
        assert (scores.hits[1], scores.hits[5]) == (top1, top5)
        assert surprisal == pytest.approx(bits, abs=1e-5)
    metrics = scores.metrics()
    assert metrics["loss"] == pytest.approx(nll / 50, abs=1e-6)
    assert metrics["perplexity"] == pytest.approx(math.exp(nll / 50), rel=1e-5)
    assert metrics["bpc"] == pytest.approx(nll / 50 / math.log(2), abs=1e-6)  # un carattere per token


class Uniform(nn.Module):
    """Un modello che non sa niente: tutti i logits uguali."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.unused = nn.Parameter(torch.zeros(()))

    def forward(self, idx):
        return torch.zeros(*idx.shape, self.vocab_size) + 0 * self.unused, None


def test_ties_count_for_the_right_token_and_uniform_is_ln_v():
    """Top-k: giusto se MENO di k logits sono strettamente maggiori; a pari merito vince il token giusto."""
    scores, _ = score_tokens(Uniform(65), list(range(40)), block_size=8)
    metrics = scores.metrics()
    assert metrics["top1"] == metrics["top5"] == 1.0
    assert metrics["loss"] == pytest.approx(math.log(65), abs=1e-6) == metrics["entropy"]
    assert metrics["bpc"] == pytest.approx(math.log2(65), abs=1e-6)
    doubled, _ = score_tokens(Uniform(65), list(range(40)), block_size=8, token_chars=torch.full((65,), 2))
    assert doubled.metrics()["bpc"] == pytest.approx(math.log2(65) / 2, abs=1e-6)  # due caratteri per token


def test_validation_and_excerpt_follow_the_split_of_gpt_data(tokenizer):
    validation = validation_text(TEXT)
    expected = tokenizer.decode(train_val_split(torch.tensor(tokenizer.encode(TEXT))).val.tolist())
    assert validation.text == expected and validation.offset == len(TEXT) - len(expected)
    text = "a?\n\nGREMIO:\nUno.\n\nBAPTISTA:\nDue.\n\nPETRUCHIO:\nTre parole lunghe."
    start, end = excerpt_bounds(text, max_chars=30)
    assert text[start:end] == "GREMIO:\nUno.\n\nBAPTISTA:\nDue.\n"  # discorsi interi, finisce con un a capo
    assert excerpt_bounds("senza righe vuote", max_chars=5) == (0, 5)


# --------------------------------------------------------------------------- benchmark


def test_benchmark_is_deterministic_and_documented(tokenizer):
    threads_before = torch.get_num_threads()
    model = small_model(tokenizer)
    validation = validation_text(TEXT)
    excerpt = EvalText(validation.text[:60], "estratto", "prova.txt", 0)
    common = {"name": "Prova", "source": "memoria", "validation": validation, "excerpt": excerpt}
    bench = benchmark_model(model, tokenizer, settings=QUICK, **common)
    other_batches = BenchmarkSettings(max_new_tokens=20, threads=1, batch_windows=1)
    assert results(bench) == results(benchmark_model(model, tokenizer, settings=other_batches, **common))
    json.dumps(bench, allow_nan=False)  # JSON valido anche per il browser
    assert set(bench) == {"format_version", "name", "created", "source", "protocol", "data", "model",
                          "metrics", "counts", "excerpt", "generation", "snapshots", "reference",
                          "environment", "notes"}
    assert set(bench["metrics"]) == {"loss", "perplexity", "bpc", "top1", "top5", "entropy", "predictions",
                                     "params", *TIMINGS}
    assert bench["protocol"]["id"] == PROTOCOL_ID and bench["protocol"]["block_size"] == 8
    assert bench["metrics"]["predictions"] == len(validation.text) - 1 == bench["counts"]["predictions"]
    assert bench["model"]["weights_sha256"] == weights_from_model(model).sha256()
    code_files = ["gpt/attention.py", "gpt/block.py", "gpt/feedforward.py", "gpt/model.py"]
    assert bench["model"]["code_files"] == code_files  # solo il codice del corso, non quello di torch
    assert bench["reference"]["uniform_loss"] == pytest.approx(math.log(tokenizer.vocab_size), abs=1e-6)
    assert len(bench["excerpt"]["surprisal_bits"]) == 59 and bench["excerpt"]["offset"] == 0
    generation = bench["generation"]
    assert len(generation["sampled"]) == len(generation["greedy"]) == 20
    assert bench["environment"]["threads"] == THREADS
    assert torch.get_num_threads() == threads_before  # il benchmark rimette i thread com'erano
    assert "Benchmark: Prova" in format_summary(bench)


def test_generation_uses_its_own_random_generator(tokenizer):
    model = small_model(tokenizer)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    first = generation_sample(model, tokenizer, QUICK, 8)
    assert torch.equal(torch.get_rng_state(), before)  # il generatore globale non si è mosso
    assert generation_sample(model, tokenizer, QUICK, 8)["sampled"] == first["sampled"]
    other = generation_sample(model, tokenizer, BenchmarkSettings(max_new_tokens=20, seed=7), 8)
    assert other["greedy"] == first["greedy"]  # il greedy non dipende dal seme


def test_checkpoint_and_run_folder_give_the_same_benchmark(tmp_path, tokenizer):
    splits = train_val_split(torch.tensor(tokenizer.encode(TEXT)))
    model = small_model(tokenizer)
    recorder_cfg = RecorderConfig(log_every=10, snapshot_steps=(0, 10), sample_length=10, eval_batch_size=4)
    recorder = TrainingRecorder(tokenizer, splits.val, 8, recorder_cfg)
    train(model, splits, 8, TrainConfig(batch_size=8, max_iters=20, eval_interval=10, eval_iters=2),
          torch.device("cpu"), log=lambda _msg: None, callbacks=[recorder])
    recorder.save(tmp_path / "trace")
    save_checkpoint(tmp_path / "ckpt.pt", model, tokenizer)
    export_run(weights_from_model(model, vocab=tokenizer.chars), tmp_path / "run", name="Run",
               training_dir=tmp_path / "trace")

    from_ckpt = benchmark_path(tmp_path / "ckpt.pt", text=TEXT, excerpt_path=None, settings=QUICK)
    from_run = benchmark_path(tmp_path / "run", text=TEXT, excerpt_path=None, settings=QUICK)
    assert from_ckpt["name"] == "ckpt" and from_run["name"] == "Run"  # il nome viene da run.json
    assert from_ckpt["model"]["weights_sha256"] == from_run["model"]["weights_sha256"]
    assert results(from_ckpt)["metrics"] == results(from_run)["metrics"]
    assert results(from_ckpt)["generation"] == results(from_run)["generation"]
    assert from_ckpt["snapshots"] == [] and [s["step"] for s in from_run["snapshots"]] == [0, 10, 20]
    final = from_run["snapshots"][-1]["metrics"]  # lo snapshot finale è la copia float16 dei pesi
    assert final["loss"] == pytest.approx(from_run["metrics"]["loss"], abs=1e-2)
    assert from_run["snapshots"][0]["metrics"]["loss"] > final["loss"]  # il training ha imparato qualcosa


class TokenBatchNorm(nn.BatchNorm1d):
    """Una BatchNorm sui canali di (B, T, C): nello state_dict ha un tensore intero, num_batches_tracked."""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class BatchNormGPT(GPTLanguageModel):
    """Il GPT del corso con una BatchNorm al posto di ln_f, come in un gpt/model.py modificato."""

    def __init__(self, config):
        super().__init__(config)
        self.ln_f = TokenBatchNorm(config.n_embd)


def test_an_integer_buffer_does_not_stop_the_benchmark_or_its_snapshots(tmp_path, tokenizer, monkeypatch):
    """Snapshot e run hanno solo i tensori in virgola mobile: num_batches_tracked non li fa più saltare."""
    monkeypatch.setattr(benchmark, "GPTLanguageModel", BatchNormGPT)  # il "codice attuale" di build_model
    torch.manual_seed(1337)
    model = BatchNormGPT(small_model(tokenizer).config)
    splits = train_val_split(torch.tensor(tokenizer.encode(TEXT)))
    recorder_cfg = RecorderConfig(log_every=10, snapshot_steps=(0, 10), sample_length=10, eval_batch_size=4)
    recorder = TrainingRecorder(tokenizer, splits.val, 8, recorder_cfg)
    train(model, splits, 8, TrainConfig(batch_size=8, max_iters=20, eval_interval=10, eval_iters=2),
          torch.device("cpu"), log=lambda _msg: None, callbacks=[recorder])
    history = read_training_history(recorder.save(tmp_path / "trace").parent)
    layout = {p["name"] for p in history.layout}
    assert model.state_dict()["ln_f.num_batches_tracked"].item() > 0  # un contatore int64 vero
    assert "ln_f.running_mean" in layout and "ln_f.num_batches_tracked" not in layout

    # 1. il modello in memoria (scripts/record_training.py, una classe propria)
    bench = benchmark_model(model, tokenizer, name="BatchNorm", source="memoria",
                            validation=validation_text(TEXT), history=history, settings=QUICK)
    assert [s["step"] for s in bench["snapshots"]] == [0, 10, 20]  # prima: [] e "snapshot ... saltati"
    assert not any("saltati" in n for n in bench["notes"])
    assert any("ln_f.num_batches_tracked" in n for n in bench["notes"])  # lo si dice
    final = bench["snapshots"][-1]["metrics"]["loss"]  # lo snapshot finale è la copia float16 dei pesi
    assert final == pytest.approx(bench["metrics"]["loss"], abs=1e-2)
    assert bench["snapshots"][0]["metrics"]["loss"] > final  # i pesi di ogni snapshot, statistiche comprese

    # 2. il run esportato (export_run.py --benchmark, benchmark_path): build_model lo ricostruisce
    weights = weights_from_model(model, vocab=tokenizer.chars)
    assert "ln_f.num_batches_tracked" in weights.skipped  # l'export salta il contatore
    export_run(weights, tmp_path / "run", name="BatchNorm", training_dir=tmp_path / "trace")
    rebuilt = build_model(read_run_folder(tmp_path / "run"))  # prima: ArchitectureMismatch ("mancano …")
    assert rebuilt.state_dict()["ln_f.num_batches_tracked"].item() == 0  # il valore iniziale del codice
    assert torch.equal(rebuilt.ln_f.running_var, model.ln_f.running_var)  # pesi del run, statistiche comprese
    from_run = benchmark_path(tmp_path / "run", text=TEXT, excerpt_path=None, settings=QUICK)
    # in eval la BatchNorm usa le statistiche (nel run), non il contatore: i numeri del modello in memoria
    assert from_run["metrics"]["loss"] == pytest.approx(bench["metrics"]["loss"], abs=1e-6)
    assert [s["step"] for s in from_run["snapshots"]] == [0, 10, 20]
    assert any(n.startswith("tensori interi o booleani che il run non contiene (ln_f.num_batches_tracked)")
               for n in from_run["notes"])


def test_the_code_fingerprint_is_public_and_keeps_its_old_names():
    """record_training.py importa code_files e code_sha256; i nomi privati restano per chi li usava."""
    assert benchmark._code_files is benchmark.code_files and benchmark._code_sha256 is benchmark.code_sha256
    files = benchmark.code_files(small_model(CharTokenizer.from_text(TEXT)))
    assert [f.relative_to(PROJECT_ROOT).as_posix() for f in files] == [
        "gpt/attention.py", "gpt/block.py", "gpt/feedforward.py", "gpt/model.py"]
    digest = benchmark.code_sha256(files)
    assert len(digest) == 64 and benchmark.code_sha256([]) is None


def test_benchmark_needs_the_current_architecture(tmp_path, tokenizer):
    state = small_model(tokenizer).state_dict()
    state["blocks.0.ffwd.gate.weight"] = torch.zeros(4, 16)
    state["blocks.1.ln1.weight"] = torch.zeros(3)
    del state["lm_head.bias"]
    config = small_model(tokenizer).config.to_dict()
    with pytest.raises(ArchitectureMismatch) as error:
        build_model(weights_from_state_dict(state, config=config))
    assert error.value.missing == ["lm_head.bias"] and error.value.extra == ["blocks.0.ffwd.gate.weight"]
    assert error.value.mismatched == [("blocks.1.ln1.weight", (3,), (16,))]
    assert "benchmark.json" in str(error.value)
    good = small_model(tokenizer).state_dict()
    with pytest.raises(ValueError, match="GPTConfig"):
        build_model(weights_from_state_dict(good, config={**config, "activation": "gelu"}))
    with pytest.raises(ValueError, match="config"):
        build_model(weights_from_state_dict(good))
    weights = weights_from_state_dict(good, config=config)  # senza vocabolario
    assert run_tokenizer(weights, default_vocab=tokenizer.chars).chars == tokenizer.chars
    with pytest.raises(ValueError, match="vocabolario"):
        run_tokenizer(weights, default_vocab=tokenizer.chars[:-1])
    with pytest.raises(FileNotFoundError, match="weights.json"):
        open_source(tmp_path)


# --------------------------------------------------------------------------- confronto


@pytest.fixture(scope="module")
def builtin():
    if not BUILTIN.exists():
        pytest.skip(f"manca webapp/data/benchmark/builtin.json: {RERUN}")
    return json.loads(BUILTIN.read_text(encoding="utf-8"))


def improved(bench):
    """Una copia "migliore" di un benchmark, misurata su un altro testo."""
    other = copy.deepcopy(bench)
    other["name"] = "Migliorato"
    other["metrics"].update(loss=bench["metrics"]["loss"] - 0.1, top1=bench["metrics"]["top1"] + 0.02)
    other["data"]["sha256"] = "0" * 64
    return other


def test_comparison_says_which_run_is_better(builtin):
    rows = {row["key"]: row for row in comparison_rows(builtin, improved(builtin))}
    assert rows["loss"]["better"] == "B" and rows["loss"]["delta"] == pytest.approx(-0.1)
    assert rows["top1"]["better"] == "B" and rows["top5"]["better"] == "="
    assert rows["params"]["better"] is None and rows["tokens_per_s"]["better"] is None  # nessun verdetto
    table = format_comparison(builtin, improved(builtin))
    assert "A = GPT del corso" in table and "B = Migliorato" in table and "meglio" in table
    assert "+2.00 punti" in table and "Attenzione: testi di validation diversi" in table
    assert "Loss sulla validation durante il training" in table  # entrambi hanno gli snapshot
    same = format_comparison(builtin, builtin)
    assert "stessi pesi, bit per bit" in same and "Attenzione" not in same


def test_benchmark_differences_ignore_timing_but_not_results(builtin):
    noisy = copy.deepcopy(builtin)
    noisy["created"] = "2000-01-01T00:00:00+00:00"
    noisy["environment"] = {"torch": "0.0"}
    noisy["metrics"].update(duration_s=99.0, tokens_per_s=1.0, eval_tokens_per_s=1.0)
    noisy["metrics"]["loss"] += LOSS_ATOL / 2
    assert benchmark_differences(builtin, noisy) == []

    def bump_bit(b):
        b["excerpt"]["surprisal_bits"][0] += 0.1

    def bump_snapshot(b):
        b["snapshots"][3]["excerpt"]["top1"] += 0.01

    for change in (
        lambda b: b["metrics"].update(loss=b["metrics"]["loss"] + 1e-3),
        lambda b: b["model"].update(weights_sha256="0" * 64),
        lambda b: b["generation"].update(greedy="altro"),
        lambda b: b["snapshots"].pop(),
        bump_bit,
        bump_snapshot,
    ):
        changed = copy.deepcopy(builtin)
        change(changed)
        assert benchmark_differences(builtin, changed), change


def test_cli_compares_two_benchmark_files(tmp_path, builtin):
    (tmp_path / "b.json").write_text(json.dumps(improved(builtin)), encoding="utf-8")
    script = str(PROJECT_ROOT / "scripts" / "benchmark.py")

    def run(*args):
        return subprocess.run([sys.executable, script, *map(str, args)], capture_output=True, text=True,
                              timeout=120, check=False)

    done = run(BUILTIN, tmp_path / "b.json")
    assert done.returncode == 0, done.stderr
    assert "loss (nat)" in done.stdout and "B = Migliorato" in done.stdout
    wrong = run(BUILTIN, BUILTIN, "--out", tmp_path / "x.json")
    assert wrong.returncode == 2 and "una sola sorgente" in wrong.stderr
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark(tmp_path / "bad.json")
    with pytest.raises(FileNotFoundError, match="non trovato"):
        load_benchmark(tmp_path / "manca.json")


# --------------------------------------------------------------------------- dati committati


def test_committed_benchmark_matches_the_committed_weights(builtin):
    """Veloce e senza dataset: estratto, impronte, snapshot sull'estratto e testi dai pesi committati."""
    excerpt = EXCERPT.read_bytes()
    assert hashlib.sha256(excerpt).hexdigest() == builtin["excerpt"]["sha256"]
    weights_bin = (WEBAPP_DATA_DIR / "weights.f32.bin").read_bytes()
    assert builtin["model"]["weights_sha256"] == hashlib.sha256(weights_bin).hexdigest()
    assert builtin["protocol"]["id"] == PROTOCOL_ID and builtin["name"] == "GPT del corso"

    src = open_source(WEBAPP_DATA_DIR)
    model, tokenizer = build_model(src.weights), run_tokenizer(src.weights)
    ids = tokenizer.encode(excerpt.decode("utf-8"))
    block = model.config.block_size

    def check(section, scores):
        counts = scores.counts()
        assert (counts["predictions"], counts["windows"]) == (section["counts"]["predictions"],
                                                              section["counts"]["windows"])
        for key in ("loss", "bpc", "entropy"):
            assert scores.metrics()[key] == pytest.approx(section[key], abs=LOSS_ATOL)
        for k in (1, 5):
            assert abs(scores.hits[k] - section["counts"][f"top{k}"]) <= HITS_ATOL

    with torch_threads(THREADS):
        scores, bits = score_tokens(model, ids, block_size=block, surprisal=True)
        check(builtin["excerpt"], scores)
        assert bits == pytest.approx(builtin["excerpt"]["surprisal_bits"], abs=SURPRISAL_ATOL)
        assert [s["step"] for s in builtin["snapshots"]] == [s["step"] for s in src.history.snapshots]
        for entry, saved in zip(src.history.snapshots, builtin["snapshots"], strict=True):
            snap = copy.deepcopy(model)
            snap.load_state_dict(snapshot_tensors(src.history, entry))
            check(saved["excerpt"], score_tokens(snap, ids, block_size=block)[0])
        keys = ("prompt", "seed", "temperature", "max_new_tokens")
        settings = BenchmarkSettings(**{k: builtin["generation"][k] for k in keys})
        generation = generation_sample(model, tokenizer, settings, block)
    assert generation["greedy"] == builtin["generation"]["greedy"]
    if builtin["environment"]["torch"] == torch.__version__:
        assert generation["sampled"] == builtin["generation"]["sampled"]


@pytest.mark.slow
@pytest.mark.skipif(not DATASET.exists(), reason="serve data/input.txt (si scarica con python train.py)")
def test_builtin_benchmark_is_up_to_date(builtin):
    """Tutto builtin.json ricalcolato dai pesi committati (validation completa e 11 snapshot)."""
    fresh, excerpt = compute_benchmark_data(settings=BenchmarkSettings(threads=THREADS))
    assert EXCERPT.read_bytes() == excerpt.encode("utf-8"), RERUN
    assert benchmark_differences(builtin, fresh) == [], RERUN


# Il protocollo riscritto in JavaScript con il motore vero (js/engine/gpt.js): così si verifica che il
# browser possa ottenere gli STESSI numeri di builtin.json. Il Laboratorio pesi ne ha la sua versione.
JS_PROTOCOL = """
import { readFile } from "node:fs/promises";
import { loadEngine } from %(engine)s;

const dir = %(data)s;
const loaders = {
  loadJson: async (f) => JSON.parse(await readFile(f, "utf8")),
  loadBuffer: async (f) => {
    const b = await readFile(f);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  },
};
function score(engine, ids) {
  const T = engine.config.block_size, V = engine.config.vocab_size;
  let nll = 0, entropy = 0, top1 = 0, top5 = 0, predictions = 0, windows = 0;
  const bits = [];
  for (let start = 0; start + 1 < ids.length; start += T) {
    const len = Math.min(T, ids.length - 1 - start);
    const { logits } = engine.forward(ids.slice(start, start + len));
    windows++;
    for (let t = 0; t < len; t++) {
      const off = t * V, target = ids[start + t + 1];
      let max = -Infinity, sum = 0, h = 0, greater = 0;
      for (let i = 0; i < V; i++) max = Math.max(max, logits[off + i]);
      for (let i = 0; i < V; i++) sum += Math.exp(logits[off + i] - max);
      const logZ = max + Math.log(sum);
      for (let i = 0; i < V; i++) {
        const lp = logits[off + i] - logZ;
        h -= Math.exp(lp) * lp;
        if (logits[off + i] > logits[off + target]) greater++;
      }
      const lp = logits[off + target] - logZ;
      nll -= lp; entropy += h; predictions++; bits.push(-lp / Math.LN2);
      if (greater < 1) top1++;
      if (greater < 5) top5++;
    }
  }
  return { predictions, windows, loss: nll / predictions, entropy: entropy / predictions, top1, top5, bits };
}
const builtin = JSON.parse(await readFile(dir + "/benchmark/builtin.json", "utf8"));
const engine = await loadEngine(dir + "/", loaders);
const ids = engine.encode(await readFile(dir + "/benchmark/val_excerpt.txt", "utf8"));
const out = { final: score(engine, ids), snapshots: [] };
const prompt = engine.encode(builtin.generation.prompt), context = [...prompt];
for (let i = 0; i < builtin.generation.max_new_tokens; i++) {
  const { logits } = engine.nextTokenDistribution(context);
  let best = 0;
  for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
  context.push(best);
}
out.greedy = engine.decode(context.slice(prompt.length));
for (const step of %(steps)s) {
  await engine.loadSnapshot(step);
  const { bits, ...rest } = score(engine, ids);
  out.snapshots.push({ step, ...rest });
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="serve Node per eseguire il motore JS")
def test_js_engine_reproduces_the_builtin_benchmark(builtin):
    # Inizializzazione, metà training e fine (float16): ~1 s di motore JS ciascuno. Tutti e 11 gli
    # snapshot danno lo stesso risultato (verificato), ma triplicherebbero la durata del test.
    saved = {s["step"]: s for s in builtin["snapshots"]}
    steps = [step for step in (0, 1000, 5000) if step in saved]
    engine = (PROJECT_ROOT / "webapp" / "js" / "engine" / "gpt.js").as_uri()
    script = JS_PROTOCOL % {"engine": json.dumps(engine), "data": json.dumps(str(WEBAPP_DATA_DIR)),
                            "steps": json.dumps(steps)}
    done = subprocess.run([NODE, "--input-type=module"], input=script, capture_output=True, text=True,
                          timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    js = json.loads(done.stdout)

    def check(result, section):
        expected = (section["counts"]["predictions"], section["counts"]["windows"])
        assert (result["predictions"], result["windows"]) == expected
        assert result["loss"] == pytest.approx(section["loss"], abs=LOSS_ATOL)
        assert result["entropy"] == pytest.approx(section["entropy"], abs=LOSS_ATOL)
        for k in (1, 5):
            assert abs(result[f"top{k}"] - section["counts"][f"top{k}"]) <= HITS_ATOL

    check(js["final"], builtin["excerpt"])
    assert js["final"]["bits"] == pytest.approx(builtin["excerpt"]["surprisal_bits"], abs=SURPRISAL_ATOL)
    assert js["greedy"] == builtin["generation"]["greedy"]  # argmax: stesso testo in JS e in PyTorch
    assert [r["step"] for r in js["snapshots"]] == steps
    for result in js["snapshots"]:
        check(result, saved[result["step"]]["excerpt"])
