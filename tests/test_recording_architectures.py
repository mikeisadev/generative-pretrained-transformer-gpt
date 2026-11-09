"""Registrazione di architetture modificate: gruppi, attention, note (gpt/recording.py, gpt/export.py).

Il GPT del corso deve restare com'era (stessi gruppi, stesso ordine, stesse mappe di attention, nessuna
nota); un'architettura con nomi nuovi o un'attention diversa deve registrarsi tutta, con le note al posto
degli errori.
"""

import json
import math
import shutil
import subprocess
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt import export
from gpt.attention import Head
from gpt.block import Block
from gpt.config import GPTConfig, TrainConfig
from gpt.data import CharTokenizer, train_val_split
from gpt.export import (
    EXAMPLE_PROMPT,
    PROJECT_ROOT,
    AttentionWeightsError,
    attention_layer_path,
    attention_sites,
    build_spec,
    capture_attention,
    parameter_layout,
    parameters_to_bytes,
)
from gpt.loss import next_token_loss
from gpt.model import GPTLanguageModel
from gpt.recording import (
    RecorderConfig,
    TrainingRecorder,
    course_group,
    group_of,
    module_group,
    parameter_groups,
    snapshot_state,
)
from gpt.runs import export_run, read_training_history, weights_from_model
from gpt.training import train

TEXT = (EXAMPLE_PROMPT + " to be or not to be, that is the question. ") * 20
BLOCK = 8
CPU = torch.device("cpu")
TRAIN_CFG = TrainConfig(batch_size=8, max_iters=12, eval_interval=6, eval_iters=2)
REC_CFG = RecorderConfig(log_every=4, snapshot_steps=(0, 6), sample_length=12, eval_batch_size=8)
NODE = shutil.which("node")
GROUPS_JS = PROJECT_ROOT / "webapp" / "js" / "shared" / "weights" / "groups.js"


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(TEXT)


@pytest.fixture
def splits(tokenizer):
    return train_val_split(torch.tensor(tokenizer.encode(TEXT)))


def tiny_config(tokenizer, **changes):
    return GPTConfig(**{"vocab_size": tokenizer.vocab_size, "block_size": BLOCK, "n_embd": 16, "n_head": 2,
                        "n_layer": 2, **changes})


# --------------------------------------------------------------------------- architetture "dello studente"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FusedAttention(nn.Module):
    """Tutte le head in un modulo (una Linear per Q, K, V): ``attention_weights`` dà (B, H, T, T)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head, self.head_size = config.n_head, config.head_size
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.out = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("tril", mask, persistent=False)

    def _heads(self, x):
        B, T, C = x.shape
        return [t.view(B, T, self.n_head, self.head_size).transpose(1, 2) for t in self.qkv(x).split(C, -1)]

    def attention_weights(self, x):
        q, k, _ = self._heads(x)
        T = x.shape[1]
        scores = q @ k.transpose(-2, -1) * self.head_size**-0.5
        return F.softmax(scores.masked_fill(self.tril[:T, :T] == 0, float("-inf")), dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        return self.out((self.attention_weights(x) @ self._heads(x)[2]).transpose(1, 2).reshape(B, T, C))


class ScoresAttention(FusedAttention):
    """Un errore tipico: ``attention_weights`` restituisce i punteggi PRIMA della softmax."""

    def attention_weights(self, x):
        q, k, _ = self._heads(x)
        return q @ k.transpose(-2, -1)

    def forward(self, x):
        B, T, C = x.shape
        y = F.softmax(self.attention_weights(x), -1) @ self._heads(x)[2]
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class FlashAttention(nn.Module):
    """``F.scaled_dot_product_attention``: veloce, ma le probabilità di attention non escono."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head, self.head_size = config.n_head, config.head_size
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in self.c_attn(x).split(C, -1))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


class ModernBlock(nn.Module):
    def __init__(self, config: GPTConfig, attention: type[nn.Module]) -> None:
        super().__init__()
        C = config.n_embd
        self.norm1 = RMSNorm(C)
        self.attn = attention(config)
        self.norm2 = RMSNorm(C)
        self.mlp = nn.Sequential(nn.Linear(C, 3 * C), nn.GELU(), nn.Linear(3 * C, C))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ModernGPT(nn.Module):
    """Il GPT del corso "migliorato" come lo scriverebbe uno studente: blocchi nuovi, stessa interfaccia."""

    def __init__(self, config: GPTConfig, attention: type[nn.Module] = FusedAttention) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[ModernBlock(config, attention) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        # Un buffer persistente in virgola mobile: è nello state_dict, quindi negli snapshot e nel run.
        self.register_buffer("logit_scale", torch.ones(1))

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        logits = self.lm_head(self.ln_f(self.blocks(x))) * self.logit_scale
        return logits, next_token_loss(logits, targets)


class SharedBlockGPT(nn.Module):
    """UN Block del corso riusato n_layer volte: pesi condivisi tra gli strati, come ALBERT."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.block = Block(config)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        for _ in range(self.config.n_layer):
            x = self.block(x)
        logits = self.lm_head(self.ln_f(x))
        return logits, next_token_loss(logits, targets)


class TiedGPT(GPTLanguageModel):
    """Il GPT del corso con il weight tying: lm_head usa la matrice (V, C) dell'embedding dei token."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.lm_head.weight = self.token_embedding_table.weight


def record(model, tokenizer, splits, out_dir):
    recorder = TrainingRecorder(tokenizer, splits.val, BLOCK, REC_CFG)
    train(model, splits, BLOCK, TRAIN_CFG, CPU, log=lambda _msg: None, callbacks=[recorder])
    path = recorder.save(out_dir)
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_no_constants), recorder


def _no_constants(name):  # JSON.parse del browser non accetta NaN e Infinity: nemmeno questo test
    raise ValueError(f"il JSON contiene {name}")


# --------------------------------------------------------------------------- il comportamento di prima


def legacy_parameter_groups(model):
    """parameter_groups com'era prima (solo il GPT del corso): l'ordine da non cambiare."""
    order = ["token_embedding", "position_embedding"]
    for i in range(model.config.n_layer):
        order += [f"blocks.{i}.attention", f"blocks.{i}.ffwd", f"blocks.{i}.layernorm"]
    order += ["ln_f", "lm_head"]
    groups = {g: [] for g in order}
    for name, _ in model.named_parameters():
        groups[course_group(name)].append(name)
    return groups


@contextmanager
def legacy_capture_attention(net):
    """capture_attention com'era prima: presupponeva blocks[i].sa.heads."""
    attention, hooks = {}, []
    for layer, blk in enumerate(net.blocks):
        for h, head in enumerate(blk.sa.heads):

            def capture(module, inputs, _output, key=(layer, h)):
                attention[key] = module.attention_weights(inputs[0]).detach()

            hooks.append(head.register_forward_hook(capture))
    try:
        yield attention
    finally:
        for hk in hooks:
            hk.remove()


@pytest.mark.parametrize("n_layer", [1, 3, 4])
def test_course_groups_and_their_order_do_not_change(n_layer):
    model = GPTLanguageModel(GPTConfig(n_layer=n_layer))
    groups, legacy = parameter_groups(model), legacy_parameter_groups(model)
    assert list(groups) == list(legacy) and groups == legacy


def test_capture_attention_is_unchanged_for_the_course(tokenizer):
    model = GPTLanguageModel(tiny_config(tokenizer, n_head=4)).eval()
    idx = torch.randint(tokenizer.vocab_size, (3, BLOCK))
    with legacy_capture_attention(model) as old:
        model(idx)
    with capture_attention(model) as new:
        model(idx)
    assert list(new) == list(old) == [(layer, h) for layer in range(2) for h in range(4)]  # stesso ordine
    assert all(torch.equal(new[key], old[key]) for key in old)


def test_model_json_of_the_course_is_byte_for_byte_the_same(tokenizer, monkeypatch):
    model = GPTLanguageModel(tiny_config(tokenizer))

    def spec_bytes():
        spec = build_spec(model, tokenizer, source="prova", trained=False)
        return json.dumps(spec, ensure_ascii=False, separators=(",", ":"))

    fresh = spec_bytes()
    monkeypatch.setattr(export, "capture_attention", legacy_capture_attention)
    assert fresh == spec_bytes()


def test_course_trace_has_no_notes_and_the_same_snapshots(tokenizer, splits, tmp_path):
    model = GPTLanguageModel(tiny_config(tokenizer))
    trace, _ = record(model, tokenizer, splits, tmp_path)
    assert "notes" not in trace  # il formato del GPT del corso non cambia
    assert trace["params"] == parameter_layout(model)  # state_dict float = named_parameters
    final = (tmp_path / trace["snapshots"][-1]["file"]).read_bytes()
    assert final == parameters_to_bytes(model, "<f2")  # i byte che scriveva il codice di prima


# --------------------------------------------------------------------------- gruppi per nomi nuovi


def test_new_names_go_to_their_module():
    assert course_group("decoder.weight") is None
    cases = {
        "blocks.0.norm1.weight": "blocks.0.norm1",
        "blocks.3.attn.qkv.weight": "blocks.3.attn",
        "transformer.h.3.attn.c_attn.weight": "transformer.h.3.attn",
        "model.layers.0.mlp.up_proj.weight": "model.layers.0.mlp",
        "transformer.wte.weight": "transformer.wte",
        "blocks.2.weight": "blocks.2",
        "logit_scale": "logit_scale",
        "0.weight": "0",
        "0.1.weight": "0.1",
        "0.attn.qkv.weight": "0.attn",  # il modello è un nn.Sequential di blocchi
        "h.0.attn.c_attn.weight": "h.0.attn",
        # Un solo blocco riusato (nessun contenitore numerato di blocchi): gli indici sono di head e strati.
        "block.sa.heads.0.key.weight": "block.sa",
        "block.sa.heads.3.value.weight": "block.sa",
        "block.sa.proj.bias": "block.sa",
        "block.ffwd.net.0.weight": "block.ffwd",
        "block.ffwd.net.2.bias": "block.ffwd",
        "block.ln1.weight": "block.ln1",
        "block.mlp.0.weight": "block.mlp",  # lo 0 è il modulo del parametro: uno strato di nn.Sequential
        "blocks.0.mlp.0.weight": "blocks.0.mlp",
    }
    assert {name: module_group(name) for name in cases} == cases
    assert {name: group_of(name) for name in cases} == cases
    assert group_of("blocks.0.sa.c_attn.weight") == "blocks.0.attention"  # dentro sa: attention del corso


def test_a_single_reused_block_is_grouped_by_its_modules(tokenizer):
    """Prima: block.sa.heads.0.key, block.ffwd.net.0… (gli indici di head e strati presi per blocchi)."""
    model = SharedBlockGPT(tiny_config(tokenizer))
    groups = parameter_groups(model)
    # l'ordine di Block in gpt/block.py: ln1, sa, ln2, ffwd
    assert list(groups) == ["token_embedding", "position_embedding", "block.ln1", "block.sa", "block.ln2",
                            "block.ffwd", "ln_f", "lm_head"]
    assert groups["block.sa"] == [n for n, _ in model.named_parameters() if n.startswith("block.sa.")]
    assert groups["block.ffwd"] == [f"block.ffwd.net.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")]


@pytest.mark.skipif(NODE is None, reason="serve Node.js")
def test_group_of_is_the_same_as_groupOf_of_the_web_app(tokenizer):
    names = [n for n, _ in GPTLanguageModel(GPTConfig()).named_parameters()] + [
        "blocks.0.norm1.weight", "blocks.3.attn.qkv.weight", "transformer.h.3.attn.c_attn.bias",
        "model.layers.0.mlp.up_proj.weight", "transformer.wte.weight", "blocks.2.weight", "logit_scale",
        "0.weight", "0.1.weight", "blocks.x.sa.w", "blocks.٣.w", "constructor.weight", "__proto__.weight",
        "a.b.c.d.e.weight", "0.attn.qkv.weight", "h.0.attn.c_attn.weight", "block.mlp.0.weight",
        "blocks.0.mlp.0.weight", "model.decoder.layers.0.self_attn.k_proj.weight", "layers.0.weight",
        "net.layers.0.weight", "x.y.z.0.w.weight", "a.0.b.1.c.weight", "", ".", "a..b", "1.2.3.4.weight",
    ]
    names += [n for n, _ in SharedBlockGPT(tiny_config(tokenizer)).named_parameters()]
    names += [n for n, _ in ModernGPT(tiny_config(tokenizer)).named_parameters()]
    script = (f"import {{ groupOf }} from {json.dumps(GROUPS_JS.as_uri())};\n"
              f"console.log(JSON.stringify({json.dumps(names)}.map(groupOf)));")
    done = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                          timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [group_of(n) for n in names]


def test_groups_of_a_modified_model_follow_the_data_flow(tokenizer):
    model = ModernGPT(tiny_config(tokenizer))
    groups = parameter_groups(model)
    assert list(groups) == ["token_embedding", "position_embedding",
                            "blocks.0.norm1", "blocks.0.attn", "blocks.0.norm2", "blocks.0.mlp",
                            "blocks.1.norm1", "blocks.1.attn", "blocks.1.norm2", "blocks.1.mlp",
                            "ln_f", "lm_head"]
    grouped = sorted(n for names in groups.values() for n in names)
    assert grouped == sorted(n for n, _ in model.named_parameters())


# --------------------------------------------------------------------------- attention ovunque


def test_attention_layer_is_the_numbered_block_above_the_module():
    assert attention_layer_path("blocks.2.sa.heads.0") == "blocks.2"
    assert attention_layer_path("transformer.h.2.attn") == "transformer.h.2"
    assert attention_layer_path("block.sa.heads.3") == ""  # un blocco solo: il 3 è la head, non lo strato
    assert attention_layer_path("attn") == ""


def test_attention_is_found_wherever_it_is(tokenizer):
    config = tiny_config(tokenizer, n_head=4)
    model = ModernGPT(config)
    sites = attention_sites(model)
    assert [(s.path, s.layer, s.layer_path) for s in sites] == [("blocks.0.attn", 0, "blocks.0"),
                                                               ("blocks.1.attn", 1, "blocks.1")]
    with capture_attention(model) as attention:
        model(torch.randint(config.vocab_size, (2, BLOCK)))
    assert list(attention) == [(layer, h) for layer in range(2) for h in range(4)]
    assert all(w.shape == (2, BLOCK, BLOCK) for w in attention.values())

    class OneBlock(nn.Module):  # un blocco del corso, senza nn.Sequential numerato
        def __init__(self):
            super().__init__()
            self.block = Block(config)

        def forward(self, x):
            return self.block(x)

    one = OneBlock()
    with capture_attention(one) as attention:
        one(torch.randn(1, BLOCK, config.n_embd))
    assert list(attention) == [(0, h) for h in range(4)]

    class Outer(nn.Module):  # espone le head insieme e contiene le Head del corso: contano solo una volta
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleList([Head(config) for _ in range(2)])
            self.attention_weights_table = nn.Linear(2, 2)  # un nome qualsiasi non è l'interfaccia

        def attention_weights(self, x):
            return torch.stack([h.attention_weights(x) for h in self.heads], dim=1)

        def forward(self, x):
            return torch.cat([h(x) for h in self.heads], dim=-1)

    nested = nn.Sequential(Outer())
    assert [s.path for s in attention_sites(nested)] == ["0"]
    with capture_attention(nested) as attention:
        nested(torch.randn(1, BLOCK, config.n_embd))
    assert list(attention) == [(0, 0), (0, 1)]

    class Confusing(nn.Module):  # un sotto-modulo che si chiama come il metodo non è un'attention
        def __init__(self):
            super().__init__()
            self.attention_weights = nn.Linear(2, 2)

    assert attention_sites(Confusing()) == []


def test_bad_attention_weights_are_reported_instead_of_raised(tokenizer):
    class WrongShape(nn.Module):
        def attention_weights(self, x):
            return x  # (B, T, C): non è (B, T, T)

        def forward(self, x):
            return x

    class Scores(nn.Module):
        def attention_weights(self, x):
            return x @ x.transpose(-2, -1)  # punteggi grezzi, non probabilità

        def forward(self, x):
            return x

    class OtherSignature(nn.Module):
        def attention_weights(self, x, mask):  # argomenti diversi da forward
            return mask

        def forward(self, x):
            return x

    class FoldedHeads(nn.Module):
        def attention_weights(self, x):  # head e batch fusi: (B·H, T, T) sembra UNA head con B·H esempi
            B, T, C = x.shape
            q = x.view(B, T, 2, C // 2).transpose(1, 2).reshape(B * 2, T, C // 2)
            future = torch.ones(T, T, dtype=torch.bool).triu(1)
            return F.softmax((q @ q.transpose(-2, -1)).masked_fill(future, float("-inf")), dim=-1)

        def forward(self, x):
            return x

    net = nn.Sequential(WrongShape(), Scores(), OtherSignature(), FoldedHeads())
    x = torch.randn(2, BLOCK, 16)
    problems = []
    with capture_attention(net, problems=problems) as attention:
        net(x)
        net(x)  # una volta sola per modulo, anche con più forward
    assert attention == {} and len(problems) == 4
    assert "forma" in problems[0] and "non sono probabilità" in problems[1] and "TypeError" in problems[2]
    assert "batch (2)" in problems[3] and "view(B, H, T, T)" in problems[3]
    with pytest.raises(AttentionWeightsError, match="forma"), capture_attention(net):
        net(x)
    assert not net[0]._forward_hooks  # gli hook se ne vanno anche dopo un errore


def test_the_attention_method_is_looked_up_without_running_anything():
    class Lazy(nn.Module):  # una property con quel nome non è il metodo, e cercarla non deve calcolarla
        @property
        def attention_weights(self):
            raise AssertionError("la property non doveva essere calcolata")

    assert attention_sites(nn.Sequential(Lazy())) == []


def test_a_block_reused_in_every_layer_gives_one_row_per_pass(tokenizer, splits, tmp_path):
    config = tiny_config(tokenizer, n_layer=3)
    model = SharedBlockGPT(config)
    sites = attention_sites(model)
    assert [(s.path, s.layer, s.layer_path) for s in sites] == [("block.sa.heads.0", 0, ""),
                                                               ("block.sa.heads.1", 0, "")]
    with capture_attention(model) as attention:
        model(torch.randint(config.vocab_size, (2, BLOCK)))
    # 3 passaggi dello stesso blocco: una riga ciascuno (prima: un solo strato con 6 head)
    assert list(attention) == [(row, h) for row in range(3) for h in range(2)]
    assert not torch.equal(attention[(0, 0)], attention[(1, 0)])  # la stessa head su ingressi diversi

    trace, _ = record(model, tokenizer, splits, tmp_path)
    assert all([len(row) for row in s["attention_entropy"]] == [2, 2, 2] for s in trace["snapshots"])
    note = next(n for n in trace["notes"] if n.startswith("attention:"))
    assert "3 strati con 2 head" in note and "1 (il modello intero)" in note and "passaggio" in note


def test_shared_weights_are_recorded_once_per_name_with_a_note(tokenizer, splits, tmp_path):
    model = TiedGPT(tiny_config(tokenizer))
    trace, _ = record(model, tokenizer, splits, tmp_path / "training")
    names = [p["name"] for p in trace["params"]]
    # Negli snapshot ci sono i due nomi (come nello state_dict e nel run), nei gruppi solo il primo.
    assert "lm_head.weight" in names and "token_embedding_table.weight" in names
    grouped = [n for group in trace["group_params"].values() for n in group]
    assert "lm_head.weight" not in grouped and trace["group_params"]["lm_head"] == ["lm_head.bias"]
    note = next(n for n in trace["notes"] if n.startswith("pesi condivisi:"))
    assert "lm_head.weight = token_embedding_table.weight" in note and "token_embedding" in note
    info = export_run(weights_from_model(model, vocab=tokenizer.chars), tmp_path / "run", name="Legati",
                      training_dir=tmp_path / "training")
    assert not any("tensori diversi" in n or "non coincide" in n for n in info["notes"])


# --------------------------------------------------------------------------- registrazione completa


def test_a_modified_architecture_is_recorded_completely(tokenizer, splits, tmp_path):
    model = ModernGPT(tiny_config(tokenizer))
    trace, _ = record(model, tokenizer, splits, tmp_path / "training")

    names = [n for names in trace["group_params"].values() for n in names]
    assert sorted(names) == sorted(n for n, _ in model.named_parameters())
    assert trace["groups"] == list(trace["group_params"]) == list(parameter_groups(model))
    metrics = trace["metrics"]
    for key in ("grad_norm_by_group", "update_ratio_by_group", "weight_std_by_group"):
        assert list(metrics[key]) == trace["groups"]
        assert all(len(v) == len(metrics["steps"]) for v in metrics[key].values())
    assert len(trace["train_loss"]) == TRAIN_CFG.max_iters and trace["evals"]["steps"] == [0, 6, 11]

    # Snapshot: i tensori float dello state_dict, buffer persistente compreso (come export_run).
    assert [p["name"] for p in trace["params"]] == list(snapshot_state(model))
    assert "logit_scale" in {p["name"] for p in trace["params"]}
    assert [s["step"] for s in trace["snapshots"]] == [0, 6, 12]
    for snap in trace["snapshots"]:
        assert (tmp_path / "training" / snap["file"]).stat().st_size == 2 * trace["total_params"]
        assert len(snap["sample_text"]) == REC_CFG.sample_length
        assert sum(snap["example_next_token_probs"]) == pytest.approx(1.0, abs=1e-3)
        assert 0 <= snap["top1"] <= snap["top5"] <= 1
        rows = snap["attention_entropy"]  # dalle head di FusedAttention: (strato, head)
        assert [len(row) for row in rows] == [2, 2]
        assert all(0 <= e <= math.log(BLOCK) for row in rows for e in row)

    notes = trace["notes"]
    assert any(n.startswith("gruppi:") and "blocks.0.norm1" in n and "group_params" in n for n in notes)
    assert any(n.startswith("attention:") and "blocks.*.attn" in n and "2 strati" in n for n in notes)

    # Il run si esporta con la sua storia: stessi tensori, ultimo snapshot = pesi finali (in float16).
    info = export_run(weights_from_model(model, vocab=tokenizer.chars), tmp_path / "run", name="Moderno",
                      training_dir=tmp_path / "training")
    assert not any("tensori diversi" in n or "non coincide" in n for n in info["notes"])
    assert read_training_history(tmp_path / "run" / "training").snapshots[-1]["step"] == TRAIN_CFG.max_iters
    # Le note della registrazione arrivano anche in run.json: la scheda del run nel Laboratorio le mostra.
    copied = [n for n in info["notes"] if n.startswith("storia del training: ")]
    assert copied == [f"storia del training: {n}" for n in notes]


@pytest.mark.parametrize("attention, reason", [(FlashAttention, "nessun modulo"),
                                               (ScoresAttention, "non sono probabilità")])
def test_attention_without_probabilities_is_skipped_with_a_note(attention, reason, tokenizer, splits,
                                                                tmp_path):
    model = ModernGPT(tiny_config(tokenizer), attention=attention)
    trace, _ = record(model, tokenizer, splits, tmp_path)
    assert all("attention_entropy" not in snap for snap in trace["snapshots"])
    assert any(n.startswith("attention:") and reason in n for n in trace["notes"])
    for snap in trace["snapshots"]:  # tutto il resto c'è
        assert {"sample_text", "example_next_token_probs", "loss", "top1", "top5", "entropy"} <= set(snap)
    assert all(math.isfinite(v) for v in trace["metrics"]["update_ratio"])


def test_non_finite_values_become_null_with_a_note(tokenizer, splits, tmp_path):
    model = GPTLanguageModel(tiny_config(tokenizer))
    _, recorder = record(model, tokenizer, splits, tmp_path)
    recorder.train_loss[3] = float("nan")  # come dopo un training divergente
    recorder.metrics["grad_norm"][0] = float("inf")
    trace = json.loads(recorder.save(tmp_path).read_text(encoding="utf-8"), parse_constant=_no_constants)
    assert trace["train_loss"][3] is None and trace["metrics"]["grad_norm"][0] is None
    assert any("2 valori non finiti" in n and "null" in n for n in trace["notes"])
