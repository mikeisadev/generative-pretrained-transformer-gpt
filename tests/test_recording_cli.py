"""scripts/record_training.py: opzioni per le architetture modificate, protezione dei dati del corso."""

import argparse
import dataclasses
import importlib.util
import json

import pytest
import torch

from gpt.config import GPTConfig, TrainConfig
from gpt.export import EXAMPLE_PROMPT, PROJECT_ROOT, TRAINING_DIR, WEBAPP_DATA_DIR, WEIGHTS_MANIFEST
from gpt.model import GPTLanguageModel
from gpt.recording import RecorderConfig
from gpt.runs import export_run, read_checkpoint

TEXT = (EXAMPLE_PROMPT + " to be or not to be, that is the question. ") * 20
SCRIPT = PROJECT_ROOT / "scripts" / "record_training.py"
TINY = ["--block-size", "8", "--n-embd", "16", "--n-head", "2", "--n-layer", "1", "--max-iters", "6",
        "--eval-interval", "3", "--eval-iters", "1", "--batch-size", "4", "--snapshot-steps", "0,3",
        "--log-every", "2", "--threads", "1"]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("record_training", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_text(script, monkeypatch):
    """Il testo dei test al posto di Tiny Shakespeare: niente download, vocabolario piccolo."""
    monkeypatch.setattr(script, "load_text", lambda: TEXT)
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)  # --threads cambia i thread del processo: i test dopo non se ne accorgono


def course_manifest():
    path = WEBAPP_DATA_DIR / WEIGHTS_MANIFEST
    if not path.is_file():
        pytest.skip("webapp/data/weights.json non ancora generato")
    return json.loads(path.read_text(encoding="utf-8"))


def test_options_come_from_the_config_dataclasses(script):
    args = script.parse_args(["--n-layer", "6", "--dropout", "0.1", "--learning-rate", "3e-4",
                              "--snapshot-steps", "0,100,50", "--log-every", "5"])
    assert script.overrides(args, script.MODEL) == {"n_layer": 6, "dropout": 0.1}
    assert script.overrides(args, script.TRAIN) == {"learning_rate": 3e-4}
    assert script.recorder_config(args) == RecorderConfig(log_every=5, snapshot_steps=(0, 50, 100))
    # Ogni campo di GPTConfig ha la sua opzione, tranne vocab_size, che viene dal testo.
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    options = _option_strings(script)
    assert {f"--{name.replace('_', '-')}" for name in fields - {"vocab_size"}} <= options
    assert "--vocab-size" not in options
    defaults = script.parse_args([])
    assert script.overrides(defaults, script.MODEL) == {} and script.overrides(defaults, script.TRAIN) == {}
    assert script.recorder_config(defaults) == RecorderConfig()


def _option_strings(script):
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("modello")
    script.add_dataclass_options(group, GPTConfig, script.MODEL, skip={"vocab_size"})
    return {s for action in parser._actions for s in action.option_strings}


def test_a_field_added_to_the_config_becomes_an_option(script):
    @dataclasses.dataclass(frozen=True)
    class MyConfig:  # come GPTConfig dopo una modifica dello studente
        n_layer: int = 4
        ffn_mult: int = 4
        use_bias: bool = True
        norm: str = "layernorm"
        steps: tuple = (1, 2)  # un tipo complesso: si cambia nel codice, niente opzione

    parser = argparse.ArgumentParser()
    script.add_dataclass_options(parser, MyConfig, "x__")
    args = parser.parse_args(["--ffn-mult", "3", "--no-use-bias", "--norm", "rmsnorm"])
    assert script.overrides(args, "x__") == {"ffn_mult": 3, "use_bias": False, "norm": "rmsnorm"}
    assert "--steps" not in {s for action in parser._actions for s in action.option_strings}

    @dataclasses.dataclass(frozen=True)
    class Clash:
        device: str = "cpu"

    parser.add_argument("--device")
    with pytest.raises(SystemExit, match="stesso nome"):
        script.add_dataclass_options(parser, Clash, "y__")


def test_a_new_field_named_like_a_fixed_option_is_explained(script, monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class MyTrainConfig:  # TrainConfig con un campo aggiunto che lo script ha già come opzione sua
        max_iters: int = 5000
        log_every: int = 50

    monkeypatch.setattr(script, "TrainConfig", MyTrainConfig)
    with pytest.raises(SystemExit, match="log_every di MyTrainConfig ha lo stesso nome"):
        script.parse_args([])  # prima: argparse.ArgumentError, un traceback in inglese


def test_bad_options_are_explained_in_italian(script, capsys):
    cases = ((["--n-layer", "sei"], "serve un numero intero"), (["--threads", "0"], "intero positivo"),
             (["--snapshot-steps", "a,b"], "step interi"), (["--learning-rate", "nan"], "numero finito"),
             (["--n-leyer", "3"], "opzioni sconosciute: --n-leyer"), (["--n-layer"], "serve un valore"))
    for argv, message in cases:
        with pytest.raises(SystemExit):
            script.parse_args(argv)
        err = capsys.readouterr().err
        assert message in err and "Errore:" in err and "error:" not in err and err.startswith("uso: ")


def test_the_reference_training_is_recognized(script):
    course = course_manifest()
    model = GPTLanguageModel(GPTConfig(vocab_size=len(course["vocab"])))
    assert script.reference_differences(model, TrainConfig(), RecorderConfig(), course) == ([], [])
    deeper = GPTLanguageModel(GPTConfig(vocab_size=len(course["vocab"]), n_layer=5))
    architecture, training = script.reference_differences(deeper, TrainConfig(max_iters=10),
                                                          RecorderConfig(log_every=5), course)
    assert any("nuovi blocks.4" in d for d in architecture)
    assert any("n_layer=5 invece di 4" in d for d in architecture)
    assert training == ["training: max_iters=10 invece di 5000", "registrazione: log_every=5 invece di 25"]


def test_the_protection_sees_buffers_shared_weights_and_model_code(script):
    course = course_manifest()
    config = GPTConfig(vocab_size=len(course["vocab"]))
    defaults = (TrainConfig(), RecorderConfig())

    with_buffer = GPTLanguageModel(config)  # nessun parametro nuovo, ma un tensore in più negli snapshot
    with_buffer.register_buffer("temperature", torch.ones(1))
    architecture, _ = script.reference_differences(with_buffer, *defaults, course)
    assert any("nuovi temperature" in d for d in architecture)

    tied = GPTLanguageModel(config)  # nomi e forme del corso, ma un tensore condiviso (non "mancante")
    tied.lm_head.weight = tied.token_embedding_table.weight
    assert script.reference_differences(tied, *defaults, course)[0] == [
        "pesi condivisi: lm_head.weight è lo stesso tensore di token_embedding_table.weight"]

    # Il codice dei moduli: un calcolo nuovo con gli stessi pesi (es. GELU) cambia l'impronta.
    course_model = GPTLanguageModel(config)
    other_code = script.reference_differences(course_model, *defaults, course, "0" * 64)[0]
    assert len(other_code) == 1 and "codice dei moduli" in other_code[0]
    assert "gpt/feedforward.py" in other_code[0]
    code = script._read_course_code()
    if code is None:
        pytest.skip("webapp/data/benchmark/builtin.json non ancora generato")
    # Il codice attuale è quello misurato in builtin.json: il comando del README resta permesso.
    assert script.reference_differences(course_model, *defaults, course, code) == ([], [])


def test_experiments_never_write_into_the_course_files(script):
    def problems(*argv, architecture=("config: n_layer=6 invece di 4",), training=()):
        return [fix for _, fix in script.webapp_problems(script.parse_args(list(argv)), list(architecture),
                                                         list(training))]

    assert problems() == ["--out checkpoints/esperimento/training", "--no-export",
                          "--checkpoint checkpoints/esperimento/gpt.pt"]
    assert problems("--out", "x/training", "--no-export", "--checkpoint", "x/gpt.pt") == []
    inside = str(WEBAPP_DATA_DIR / TRAINING_DIR / "altro")  # anche una sottocartella di webapp/data
    fixes = problems("--out", inside, "--no-export", "--checkpoint", "x.pt")
    assert fixes == ["--out checkpoints/esperimento/training"]
    # Cambia solo il training: il checkpoint predefinito va bene (come train.py --max-iters 500).
    assert problems("--out", "x", "--no-export", architecture=(), training=("training: max_iters=6",)) == []
    assert problems(architecture=()) == []  # il training di riferimento scrive dove ha sempre scritto
    assert problems("--out", "x", architecture=()) == ["--no-export"]  # pesi nuovi con la storia vecchia
    # ... e il contrario: storia nuova in webapp/data accanto ai pesi vecchi
    assert problems("--no-export", architecture=()) == ["--out checkpoints/esperimento/training"]
    assert problems("--update-webapp") == []  # chi cambia il GPT del corso lo dice esplicitamente


def test_unusable_outputs_are_refused_before_training(script, tiny_text, tmp_path, monkeypatch, capsys):
    def no_training(*_args, **_kwargs):
        raise AssertionError("il training non doveva cominciare")

    monkeypatch.setattr(script, "train", no_training)
    a_file = tmp_path / "un-file"
    a_file.write_text("x", encoding="utf-8")
    checkpoint = str(tmp_path / "c.pt")
    cases = ((["--out", str(a_file), "--checkpoint", checkpoint], "non si può usare come cartella"),
             (["--out", str(a_file / "sotto"), "--checkpoint", checkpoint], "è un file"),
             (["--out", str(tmp_path / "out"), "--checkpoint", str(tmp_path)], "è una cartella"))
    for argv, message in cases:
        assert script.main([*TINY, "--no-export", *argv]) == 1  # prima: un traceback DOPO il training
        err = capsys.readouterr().err
        assert "le uscite non si possono scrivere" in err and message in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["un-file"]  # niente scritto


def test_the_script_refuses_before_training(script, tiny_text, monkeypatch, capsys):
    trace = WEBAPP_DATA_DIR / TRAINING_DIR / "trace.json"
    before = trace.read_bytes() if trace.is_file() else None
    checkpoint = PROJECT_ROOT / "checkpoints" / "gpt.pt"
    existed = checkpoint.exists()

    def no_training(*_args, **_kwargs):
        raise AssertionError("il training non doveva cominciare")

    monkeypatch.setattr(script, "train", no_training)
    assert script.main(["--n-layer", "6", "--max-iters", "10"]) == 1
    err = capsys.readouterr().err
    assert "non è il training di riferimento" in err and "--no-export" in err and "--checkpoint" in err
    assert (trace.read_bytes() if trace.is_file() else None) == before
    assert checkpoint.exists() == existed


def test_a_modified_training_is_recorded_from_the_command_line(script, tiny_text, tmp_path, capsys):
    ckpt, out = tmp_path / "esperimento" / "gpt.pt", tmp_path / "esperimento" / "training"
    code = script.main([*TINY, "--checkpoint", str(ckpt), "--out", str(out), "--no-export"])
    printed = capsys.readouterr().out
    assert code == 0, printed
    trace = json.loads((out / "trace.json").read_text(encoding="utf-8"))
    assert trace["config"]["n_layer"] == 1 and trace["train_config"]["max_iters"] == 6
    assert [s["step"] for s in trace["snapshots"]] == [0, 3, 6] and "notes" not in trace
    assert "diverso dal GPT del corso" in printed and "scripts/export_run.py" in printed
    # Il formato che export_run.py --trace sa leggere, abbinato ai pesi del checkpoint.
    info = export_run(read_checkpoint(ckpt), tmp_path / "run", name="Esperimento", training_dir=out)
    assert info["training"]["snapshots"] == [0, 3, 6] and info["training"]["steps"] == 6
    assert not any("non coincide" in n or "tensori diversi" in n for n in info["notes"])
