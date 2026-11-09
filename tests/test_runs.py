"""Run esportabili (gpt/runs.py): cartelle che il Laboratorio pesi della web app sa caricare."""

import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, train_val_split
from gpt.export import EXAMPLE_PROMPT, PROJECT_ROOT, WEBAPP_DATA_DIR, export_weights
from gpt.model import GPTLanguageModel
from gpt.recording import RecorderConfig, TrainingRecorder
from gpt.runs import (
    BENCHMARK_FILE,
    MAX_TEXT_CHARS,
    MAX_TRAINING_NOTES,
    RUN_FILE,
    TRAINING_NOTE_PREFIX,
    display_path,
    export_run,
    module_tree,
    read_checkpoint,
    read_run_folder,
    read_training_history,
    readable_json,
    slugify,
    training_notes,
    weights_from_model,
    weights_from_state_dict,
)
from gpt.training import save_checkpoint, train

TEXT = (EXAMPLE_PROMPT + " to be or not to be, that is the question. ") * 20
NODE = shutil.which("node")
FIXTURE = PROJECT_ROOT / "webapp" / "tests" / "engine_fixture.mjs"
RUN_KEYS = {"format_version", "name", "description", "created", "source", "config", "total_params",
            "tensor_count", "modules", "buffers", "skipped", "files", "training", "notes"}


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(TEXT)


def small_model(tokenizer, seed=1337):
    torch.manual_seed(seed)
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=2)
    return GPTLanguageModel(config)


def read_flat(folder):
    """Rilettura indipendente dal codice di export: solo numpy, offset e forme del manifest."""
    manifest = json.loads((folder / "weights.json").read_text(encoding="utf-8"))
    flat = np.fromfile(folder / manifest["file"], dtype="<f4")
    tensors = {p["name"]: flat[p["offset"] : p["offset"] + p["numel"]].reshape(p["shape"])
               for p in manifest["params"]}
    return manifest, tensors


def leaves(node):
    if node["kind"] != "module":
        yield node
    for child in node.get("children", []):
        yield from leaves(child)


# --------------------------------------------------------------------------- formato


def test_run_weights_are_written_like_the_webapp_data(tmp_path, tokenizer):
    """Stessa funzione di gpt/export.py: per il modello del corso i file sono identici byte per byte."""
    model = small_model(tokenizer)
    export_weights(model, tokenizer, tmp_path / "webapp", source="prova", trained=True)
    weights = weights_from_model(model, vocab=tokenizer.chars, source="prova")
    export_run(weights, tmp_path / "run", name="Prova")
    for name in ("weights.f32.bin", "weights.json"):
        assert (tmp_path / "run" / name).read_bytes() == (tmp_path / "webapp" / name).read_bytes()


def test_checkpoint_export_has_the_documented_run_json(tmp_path, tokenizer):
    model = small_model(tokenizer)
    save_checkpoint(tmp_path / "ckpt.pt", model, tokenizer)
    weights = read_checkpoint(tmp_path / "ckpt.pt")
    info = export_run(weights, tmp_path / "run", name="  GPT piccolo ", description="prova",
                      notes=["una nota"])

    on_disk = json.loads((tmp_path / "run" / RUN_FILE).read_text(encoding="utf-8"))
    assert on_disk == info and set(info) == RUN_KEYS  # engine_compatible manca: non dichiarato
    assert info["name"] == "GPT piccolo" and info["notes"][0] == "una nota"
    assert info["config"] == model.config.to_dict() and info["training"] is None
    assert info["files"] == {"weights": "weights.json", "benchmark": None, "trace": None}
    assert info["buffers"] is None  # da un checkpoint non si sa quali tensori sono buffer
    manifest, tensors = read_flat(tmp_path / "run")
    assert manifest["vocab"] == tokenizer.chars and manifest["trained"] is True
    assert manifest["training"] is None
    state = model.state_dict()
    assert list(tensors) == list(state)
    assert all(np.array_equal(tensors[n], state[n].numpy()) for n in state)  # bit per bit

    tree = info["modules"]
    assert tree["params"] == info["total_params"] == model.num_parameters()
    found = list(leaves(tree))
    assert [leaf["path"] for leaf in found] == list(state) and len(found) == info["tensor_count"]
    head = next(leaf for leaf in found if leaf["path"] == "blocks.1.sa.heads.0.query.weight")
    assert head == {"path": "blocks.1.sa.heads.0.query.weight", "key": "weight", "kind": "param",
                    "shape": [8, 16], "numel": 128, "dtype": "float32", "source_dtype": "float32"}


def test_module_info_names_the_classes_and_the_buffers():
    model = nn.Sequential(OrderedDict(fc=nn.Linear(4, 3), norm=nn.BatchNorm1d(3)))
    weights = weights_from_model(model, source="prova")
    assert set(weights.skipped) == {"norm.num_batches_tracked"}  # un contatore intero, non un peso
    assert weights.buffers == {"norm.running_mean", "norm.running_var"}
    tree = module_tree(weights)
    assert tree["class"] == "nn.Sequential"
    assert [c["class"] for c in tree["children"]] == ["nn.Linear", "nn.BatchNorm1d"]
    kinds = {leaf["path"]: leaf["kind"] for leaf in leaves(tree)}
    assert kinds["norm.running_mean"] == "buffer" and kinds["fc.weight"] == "param"


def test_modified_architecture_is_exported_without_its_class(tmp_path, tokenizer):
    """Nomi e forme nuovi e una chiave di config sconosciuta: l'export non costruisce il modello."""
    state = small_model(tokenizer).state_dict()
    state["blocks.0.ffwd.gate.weight"] = torch.randn(64, 16)
    state["blocks.0.ffwd.net.0.weight"] = torch.randn(48, 16)
    state["head.weight"] = state.pop("lm_head.weight")
    config = {**small_model(tokenizer).config.to_dict(), "activation": "gelu"}
    payload = {"format_version": 1, "config": config, "chars": tokenizer.chars, "state_dict": state}
    torch.save(payload, tmp_path / "x.pt")

    info = export_run(read_checkpoint(tmp_path / "x.pt"), tmp_path / "run", name="GPT modificato")
    assert info["config"]["activation"] == "gelu"
    manifest, tensors = read_flat(tmp_path / "run")
    assert [p["name"] for p in manifest["params"]] == list(state)
    shapes = {p["name"]: p["shape"] for p in manifest["params"]}
    assert shapes["blocks.0.ffwd.net.0.weight"] == [48, 16]
    assert shapes["blocks.0.ffwd.gate.weight"] == [64, 16]
    assert all(np.array_equal(tensors[n], state[n].numpy()) for n in state)
    assert manifest["total_params"] == sum(t.numel() for t in state.values()) == info["total_params"]


def test_other_checkpoint_formats_and_dtypes(tmp_path):
    state = OrderedDict([
        ("_orig_mod.emb.weight", torch.randn(5, 3).to(torch.bfloat16)),
        ("_orig_mod.out.bias", torch.randn(5).half()),
        ("_orig_mod.steps", torch.tensor(7)),
    ])
    torch.save({"model": state, "model_args": {"n_embd": 3, "bias": True}}, tmp_path / "nano.pt")  # nanoGPT
    weights = read_checkpoint(tmp_path / "nano.pt")
    assert list(weights.tensors) == ["emb.weight", "out.bias"] and set(weights.skipped) == {"steps"}
    assert weights.config == {"n_embd": 3, "bias": True} and weights.vocab is None
    bf16 = state["_orig_mod.emb.weight"]
    assert torch.equal(weights.tensors["emb.weight"], bf16.float())  # bfloat16 → float32 senza perdita
    assert weights.source_dtypes == {"emb.weight": "bfloat16", "out.bias": "float16"}
    notes = " ".join(weights.notes)
    assert "_orig_mod" in notes and "bfloat16" in notes and "float16" in notes

    torch.save(state, tmp_path / "bare.pt")  # uno state_dict "nudo"
    assert list(read_checkpoint(tmp_path / "bare.pt").tensors) == ["emb.weight", "out.bias"]
    # Togliere i prefissi creerebbe nomi doppi: restano com'erano.
    clash = weights_from_state_dict({"a.w": torch.ones(1), "module.a.w": torch.zeros(1)})
    assert list(clash.tensors) == ["a.w", "module.a.w"]


class _Boom:
    """Se un pickle qualunque venisse eseguito, questo oggetto scriverebbe un file "segnale"."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (Path.write_text, (Path(self.marker), "codice eseguito"))


def test_unsafe_or_broken_checkpoints_are_refused_without_running_code(tmp_path, tokenizer):
    marker = tmp_path / "eseguito.txt"
    payload = {"state_dict": small_model(tokenizer).state_dict(), "regalo": _Boom(marker)}
    torch.save(payload, tmp_path / "evil.pt")
    with pytest.raises(ValueError, match="weights_only"):
        read_checkpoint(tmp_path / "evil.pt")
    assert not marker.exists()

    (tmp_path / "rotto.pt").write_bytes(b"non sono un checkpoint")
    with pytest.raises(ValueError, match="danneggiato"):
        read_checkpoint(tmp_path / "rotto.pt")
    (tmp_path / "vuoto.pt").write_bytes(b"")
    with pytest.raises(ValueError):
        read_checkpoint(tmp_path / "vuoto.pt")
    with pytest.raises(FileNotFoundError):
        read_checkpoint(tmp_path / "manca.pt")
    torch.save({"pesi": [1, 2, 3]}, tmp_path / "senza.pt")
    with pytest.raises(ValueError, match="state_dict"):
        read_checkpoint(tmp_path / "senza.pt")


def test_invalid_inputs_are_rejected_with_clear_errors():
    one = {"w": torch.ones(1)}
    with pytest.raises(ValueError, match="virgola mobile"):
        weights_from_state_dict({"contatore": torch.tensor([1, 2])})
    with pytest.raises(ValueError, match="nome di tensore"):
        weights_from_state_dict({"a\nb": torch.ones(1)})
    with pytest.raises(ValueError, match="anche il modulo"):
        weights_from_state_dict({"a.b": torch.ones(1), "a.b.c": torch.ones(1)})
    with pytest.raises(ValueError, match="vocab_size"):
        weights_from_state_dict(one, config={"vocab_size": 3}, vocab=["a", "b"])
    with pytest.raises(ValueError, match="duplicati"):
        weights_from_state_dict(one, vocab=["a", "a"])
    with pytest.raises(ValueError, match="lista di stringhe"):
        weights_from_state_dict(one, vocab="abc")
    with pytest.raises(ValueError, match="intero positivo"):
        weights_from_state_dict(one, config={"n_layer": 0})
    odd = weights_from_state_dict(one, config={"n_layer": 2, "dtype": torch.float16, "lr": float("nan")})
    assert odd.config == {"n_layer": 2} and "dtype" in " ".join(odd.notes)
    diverged = weights_from_state_dict({"w": torch.tensor([1.0, float("nan"), float("inf")])})
    assert "2 valori non finiti" in " ".join(diverged.notes)


# --------------------------------------------------------------------------- storia del training


@pytest.fixture
def recorded(tmp_path, tokenizer):
    """Un training vero e minuscolo, registrato come fa scripts/record_training.py."""
    splits = train_val_split(torch.tensor(tokenizer.encode(TEXT)))
    model = small_model(tokenizer)
    recorder_cfg = RecorderConfig(log_every=10, snapshot_steps=(0, 10, 99), sample_length=20,
                                  eval_batch_size=8)
    recorder = TrainingRecorder(tokenizer, splits.val, 8, recorder_cfg)
    cfg = TrainConfig(batch_size=8, max_iters=30, eval_interval=10, eval_iters=3)
    train(model, splits, 8, cfg, torch.device("cpu"), log=lambda _msg: None, callbacks=[recorder])
    recorder.save(tmp_path / "trace")
    save_checkpoint(tmp_path / "ckpt.pt", model, tokenizer)
    return {"model": model, "trace": tmp_path / "trace", "ckpt": tmp_path / "ckpt.pt"}


def test_training_history_is_copied_and_matched_to_the_weights(tmp_path, recorded):
    weights = read_checkpoint(recorded["ckpt"])
    info = export_run(weights, tmp_path / "run", name="Con storia", training_dir=recorded["trace"])
    run = tmp_path / "run"
    assert sorted(p.name for p in (run / "training").iterdir()) == [
        "snap_00000.f16.bin", "snap_00010.f16.bin", "snap_00030.f16.bin", "trace.json"]
    assert (run / "training" / "trace.json").read_bytes() == (recorded["trace"] / "trace.json").read_bytes()
    manifest = json.loads((run / "weights.json").read_text(encoding="utf-8"))
    assert manifest["training"] == {"dir": "training", "trace": "training/trace.json"}
    assert info["files"]["trace"] == "training/trace.json"
    assert info["training"]["steps"] == 30 and info["training"]["snapshots"] == [0, 10, 30]
    assert info["training"]["eval_step"] == 29 and info["training"]["val_loss"] > 0
    assert not any("non coincide" in n for n in info["notes"])

    # La storia di un altro training (pesi diversi, stesso layout) viene segnalata.
    other = weights_from_model(small_model(CharTokenizer.from_text(TEXT), seed=7))
    info = export_run(other, tmp_path / "altro", name="Storia sbagliata", training_dir=recorded["trace"])
    assert any("non coincide" in n for n in info["notes"])
    # Tensori diversi: gli snapshot si guardano a parte.
    different = weights_from_state_dict({"w": torch.ones(3)})
    info = export_run(different, tmp_path / "diverso", name="Diverso", training_dir=recorded["trace"])
    assert any("tensori diversi" in n for n in info["notes"])


def test_trace_notes_go_into_run_json_after_the_author_notes(tmp_path, recorded):
    """Le note della registrazione (gpt/recording.py) si leggono anche nella scheda del run."""
    trace_file = recorded["trace"] / "trace.json"
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    assert "notes" not in trace  # il GPT del corso non ne ha: run.json resta com'era
    weights = read_checkpoint(recorded["ckpt"])
    plain = export_run(weights, tmp_path / "corso", name="Corso", training_dir=recorded["trace"])
    assert not any(n.startswith(TRAINING_NOTE_PREFIX) for n in plain["notes"])

    trace["notes"] = ["attention: nessun modulo del modello espone il metodo attention_weights",
                      "gruppi: 2 parametri\tnuovi\n(norm1)"]
    trace_file.write_text(json.dumps(trace), encoding="utf-8")
    other = weights_from_model(small_model(CharTokenizer.from_text(TEXT), seed=7))  # + un avviso dell'export
    info = export_run(other, tmp_path / "run", name="Con note", notes=["mia nota"],
                      training_dir=recorded["trace"])
    assert info["notes"][:3] == ["mia nota",
                                 "storia del training: attention: nessun modulo del modello espone il metodo "
                                 "attention_weights",
                                 "storia del training: gruppi: 2 parametri nuovi (norm1)"]
    assert any("non coincide" in n for n in info["notes"][3:])  # poi gli avvisi dell'export, come prima
    assert json.loads((tmp_path / "run" / RUN_FILE).read_text(encoding="utf-8"))["notes"] == info["notes"]
    # trace.json resta la copia esatta: le note pulite stanno solo in run.json
    assert (tmp_path / "run" / "training" / "trace.json").read_bytes() == trace_file.read_bytes()


def test_trace_notes_are_read_as_untrusted_text():
    assert training_notes({}) == [] and training_notes({"notes": None}) == []
    assert training_notes({"notes": {"a": 1}}) == [] and training_notes({"notes": [1, None, "", "  "]}) == []
    assert training_notes({"notes": "una sola"}) == [f"{TRAINING_NOTE_PREFIX}una sola"]
    assert training_notes({"notes": ["a\x00b\x1b[31m"]}) == [f"{TRAINING_NOTE_PREFIX}a b [31m"]
    long = training_notes({"notes": ["x" * 5000]})[0]
    assert len(long) == MAX_TEXT_CHARS and long.endswith("…")  # il limite che il browser legge
    many = training_notes({"notes": [f"nota {i}" for i in range(MAX_TRAINING_NOTES + 5)]})
    assert len(many) == MAX_TRAINING_NOTES + 1
    assert many[-1] == f"{TRAINING_NOTE_PREFIX}altre 5 note in training/trace.json"
    # un surrogato spaiato ("\ud800" nel JSON) non si scrive in UTF-8: diventa "�"; le coppie valide restano
    assert training_notes(json.loads('{"notes": ["a \\ud800 b \\udfff c \\ud83d\\ude00"]}')) == [
        f"{TRAINING_NOTE_PREFIX}a \ufffd b \ufffd c \U0001f600"]


def test_strange_trace_notes_never_leave_a_run_half_written(tmp_path, recorded):
    """Prima: un surrogato spaiato in una nota del trace faceva fallire run.json DOPO pesi e storia, e
    run.json restava vuoto (0 byte) accanto ai pesi; ora il run è intero e la nota ha "�"."""
    trace_file = recorded["trace"] / "trace.json"
    text = trace_file.read_text(encoding="utf-8")  # JSON scritto a mano: json.dumps non darebbe "\ud800"
    strange = ',"device":"cpu\\ud800","notes":["<img src=x onerror=alert(1)> \\ud800 \\u202efine"]}'
    trace_file.write_text(text[:-1] + strange, encoding="utf-8")
    # "\udcff": un byte non UTF-8 di un argomento della riga di comando (surrogateescape)
    info = export_run(read_checkpoint(recorded["ckpt"]), tmp_path / "run", name="Nome \udcff",
                      notes=["nota \udcff"], training_dir=recorded["trace"])
    written = json.loads((tmp_path / "run" / RUN_FILE).read_text(encoding="utf-8"))
    assert written == info and written["name"] == "Nome \ufffd" and written["training"]["device"] == "cpu\ufffd"
    html = f"{TRAINING_NOTE_PREFIX}<img src=x onerror=alert(1)> \ufffd \u202efine"  # testo: lo mostra textContent
    assert written["notes"][:2] == ["nota \ufffd", html]
    assert (tmp_path / "run" / "training" / "trace.json").read_bytes() == trace_file.read_bytes()  # copia esatta


def test_training_history_is_validated_before_copying(tmp_path, recorded):
    trace = json.loads((recorded["trace"] / "trace.json").read_text(encoding="utf-8"))
    bad = tmp_path / "bad"
    shutil.copytree(recorded["trace"], bad)
    trace["snapshots"][0]["file"] = "../../segreto.bin"  # un JSON non deve poter far copiare file altrui
    (bad / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="non ammesso"):
        export_run(read_checkpoint(recorded["ckpt"]), tmp_path / "run", name="x", training_dir=bad)
    assert not (tmp_path / "run").exists()  # niente cartelle a metà

    shutil.rmtree(bad)
    shutil.copytree(recorded["trace"], bad)
    (bad / "snap_00010.f16.bin").unlink()
    history = read_training_history(bad)
    assert [s["step"] for s in history.snapshots] == [0, 30] and history.missing == [10]
    (bad / "snap_00030.f16.bin").write_bytes(b"\0" * 10)
    with pytest.raises(ValueError, match="attesi"):
        read_training_history(bad)


# --------------------------------------------------------------------------- cartelle e nomi


def test_existing_runs_are_protected_and_force_replaces_only_run_files(tmp_path, recorded):
    weights = read_checkpoint(recorded["ckpt"])
    run = tmp_path / "run"
    bench = {"metrics": {"loss": 1.0}}
    export_run(weights, run, name="Primo", training_dir=recorded["trace"], benchmark=bench)
    with pytest.raises(FileExistsError, match="--force"):
        export_run(weights, run, name="Secondo")
    (run / "appunti.txt").write_text("miei", encoding="utf-8")
    info = export_run(weights, run, name="Secondo", overwrite=True)
    assert sorted(p.name for p in run.iterdir()) == ["appunti.txt", "run.json", "weights.f32.bin",
                                                     "weights.json"]
    assert (run / "appunti.txt").read_text(encoding="utf-8") == "miei"
    assert any("appunti.txt" in n for n in info["notes"]) and info["files"]["benchmark"] is None
    assert json.loads((run / "weights.json").read_text(encoding="utf-8"))["training"] is None
    with pytest.raises(FileExistsError):
        export_run(weights, run / "run.json", name="Un file")


def test_benchmark_is_written_and_checked_against_the_weights(tmp_path, tokenizer):
    weights = weights_from_model(small_model(tokenizer))
    bench = {"format_version": 1, "metrics": {"loss": 2.5}, "model": {"weights_sha256": weights.sha256()}}
    info = export_run(weights, tmp_path / "a", name="A", benchmark=bench, engine_compatible=False)
    assert json.loads((tmp_path / "a" / BENCHMARK_FILE).read_text(encoding="utf-8")) == bench
    assert info["files"]["benchmark"] == BENCHMARK_FILE and info["engine_compatible"] is False
    assert not any("pesi diversi" in n for n in info["notes"])
    wrong = {**bench, "model": {"weights_sha256": "0" * 64}}
    notes = export_run(weights, tmp_path / "b", name="B", benchmark=wrong)["notes"]
    assert any("pesi diversi" in n for n in notes)
    with pytest.raises(ValueError):  # JSON non valido per il browser: niente file scritti
        export_run(weights, tmp_path / "c", name="C", benchmark={"metrics": {"loss": float("nan")}})
    assert not (tmp_path / "c").exists()


def test_names_texts_and_paths(tmp_path):
    assert slugify("GPT con GELU è più bello!") == "gpt-con-gelu-e-piu-bello"
    assert slugify("../../etc") == "etc"
    with pytest.raises(ValueError):
        slugify("???")
    weights = weights_from_state_dict({"w": torch.ones(1)})
    bad_texts = [{"name": ""}, {"name": "x" * 201}, {"name": "ok", "description": "x" * 2001},
                 {"name": "ok", "notes": [""]}]
    for bad in bad_texts:
        with pytest.raises(ValueError):
            export_run(weights, tmp_path / "mai", **bad)
    assert not (tmp_path / "mai").exists()  # i testi si controllano prima di toccare il disco
    assert display_path(PROJECT_ROOT / "runs" / "a") == "runs/a"
    assert display_path(Path("/home/mario/esperimenti/gpt.pt")) == "…/esperimenti/gpt.pt"


def test_run_folders_are_read_as_untrusted_input(tmp_path, tokenizer):
    export_run(weights_from_model(small_model(tokenizer), vocab=tokenizer.chars), tmp_path / "run", name="R")
    run = tmp_path / "run"
    weights = read_run_folder(run)
    assert weights.vocab == tokenizer.chars and weights.config["n_layer"] == 2
    manifest = json.loads((run / "weights.json").read_text(encoding="utf-8"))

    def broken(change):
        bad = json.loads(json.dumps(manifest))
        change(bad)
        (run / "weights.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError)):
            read_run_folder(run)

    broken(lambda m: m.update(file="../weights.f32.bin"))
    broken(lambda m: m.update(file="/etc/passwd"))
    broken(lambda m: m["params"][1].update(offset=0))  # si sovrappone al primo
    broken(lambda m: m.update(total_params=m["total_params"] + 1))
    broken(lambda m: m.update(format_version=2))
    broken(lambda m: m["params"][0].update(shape=[-1]))


def test_readable_json_is_valid_and_keeps_simple_objects_on_one_line():
    value = {"a": [1, 2], "b": {"c": "x", "d": [{"e": [3, 4], "f": None}]}, "g": [], "h": {}, "è": "à"}
    text = readable_json(value)
    assert json.loads(text) == value
    assert '{"e": [3, 4], "f": null}' in text and '"a": [1, 2]' in text


# --------------------------------------------------------------------------- integrazione


def test_cli_exports_a_checkpoint(tmp_path, tokenizer):
    model = small_model(tokenizer)
    save_checkpoint(tmp_path / "ckpt.pt", model, tokenizer)
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "export_run.py"), str(tmp_path / "ckpt.pt"),
               "--name", "Da riga di comando", "--note", "prova", "--out", str(tmp_path / "run")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert "esportato" in done.stdout and "Laboratorio pesi" in done.stdout
    info = json.loads((tmp_path / "run" / RUN_FILE).read_text(encoding="utf-8"))
    assert info["modules"]["class"] == "GPTLanguageModel"  # il codice attuale ricostruisce i pesi
    again = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert again.returncode == 1 and "contiene già dei file" in again.stderr


@pytest.mark.skipif(NODE is None, reason="serve Node per eseguire il motore JS")
@pytest.mark.skipif(not (WEBAPP_DATA_DIR / "weights.json").exists(), reason="mancano i pesi committati")
def test_exported_run_runs_in_the_js_engine_like_the_webapp_data(tmp_path):
    """Un run esportato dai pesi committati (con la storia) si carica nel motore JS come webapp/data."""
    weights = read_run_folder(WEBAPP_DATA_DIR)
    export_run(weights, tmp_path / "run", name="GPT del corso", training_dir=WEBAPP_DATA_DIR / "training")

    def engine(data_dir, snapshot):
        request = {"dataDir": str(data_dir), "snapshot": snapshot, "cases": [{"text": "ROMEO:"}]}
        done = subprocess.run([NODE, str(FIXTURE)], input=json.dumps(request), capture_output=True, text=True,
                              timeout=120, check=False)
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)

    for snapshot in (None, 1000):
        assert engine(tmp_path / "run", snapshot) == engine(WEBAPP_DATA_DIR, snapshot)
