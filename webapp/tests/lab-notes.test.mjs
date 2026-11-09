// Note della storia del training nel Laboratorio pesi e famiglie dei gruppi nei grafici:
//   node --test webapp/tests/lab-notes.test.mjs
// - le note di trace.json (gpt/recording.py) nella scheda del run, una volta sola anche quando run.json le
//   ha già copiate (gpt/runs.py, "storia del training: …"), sempre come testo (anche con dell'HTML dentro);
// - al posto della heatmap vuota dell'entropia dell'attention, il perché con la nota del trace;
// - le famiglie dei gruppi per blocco: "*" solo al posto dell'indice del blocco (norm1 e norm2 restano due).
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { expectedParameters } from "../js/engine/gpt.js";
import { groupOf } from "../js/shared/weights/groups.js";
import { loadBuiltinRun, runsFromFiles } from "../js/shared/weights/run.js";
import { installAnalysisCharts } from "../js/tabs/lab/analysis/charts.js";
import { attentionEntropy, blockFamily, groupFamilies, missingAttentionText, moduleFamily } from "../js/tabs/lab/analysis/series.js";
import { runCard } from "../js/tabs/lab/runs/card.js";
import { TRAINING_NOTE_PREFIX, runNotes, traceNotes } from "../js/tabs/lab/runs/describe.js";

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const loadJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const loadBuffer = async (file) => {
  const bytes = await readFile(file);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};
const builtin = await loadBuiltinRun({ baseUrl: DATA_DIR, loadJson, loadBuffer });

// Le note che gpt/recording.py scrive per un'attention con F.scaled_dot_product_attention (testo vero).
const SDPA_NOTE =
  "attention: nessun modulo del modello espone il metodo attention_weights (l'interfaccia di Head in gpt/attention.py, descritta in capture_attention di gpt/export.py), quindi l'entropia dell'attention per head non è stata misurata: le altre metriche ci sono tutte.";
const GROUPS_NOTE = "gruppi: 8 parametri con nomi che il GPT del corso non ha (blocks.0.norm1.weight …) sono raggruppati per modulo";
const HTML_NOTE = 'attention: <img src=x onerror="globalThis.hacked = true"><b>grassetto</b> & altro';

function fileLike(path, content) {
  const bytes = typeof content === "string" ? new TextEncoder().encode(content) : content;
  return { name: path.split("/").pop(), size: bytes.byteLength, webkitRelativePath: path, text: async () => new TextDecoder().decode(bytes), arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
}

/**
 * Un run esportato come fa gpt/runs.py (pesi, run.json, training/ con trace.json e uno snapshot), in
 * oggetti "tipo File": l'architettura del corso a una config minuscola, senza entropia dell'attention.
 */
function exportedRun(name, { info = {}, traceNotes: notes = [SDPA_NOTE, GROUPS_NOTE] } = {}) {
  const config = { vocab_size: 5, block_size: 4, n_embd: 4, n_head: 2, n_layer: 1, dropout: 0 };
  let offset = 0;
  const params = expectedParameters(config).map(({ name: n, shape }) => {
    const numel = shape.reduce((a, b) => a * b, 1);
    const p = { name: n, shape, offset, numel };
    offset += numel;
    return p;
  });
  const vocab = ["\n", " ", "a", "b", "c"];
  const weights = { format_version: 1, dtype: "float32", byte_order: "little", file: "weights.f32.bin", config, vocab, total_params: offset, params, training: { dir: "training", trace: "training/trace.json" } };
  const trace = { format_version: 1, config, vocab, total_params: offset, snapshot_dtype: "float16", byte_order: "little", params, train_config: { max_iters: 0 }, snapshots: [{ step: 0, file: "snap_00000.f16.bin", loss: 1.6 }], notes };
  const run = { format_version: 1, name, description: "prova", config, notes: [], ...info };
  return [
    fileLike(`${name}/weights.json`, JSON.stringify(weights)),
    fileLike(`${name}/weights.f32.bin`, new Uint8Array(4 * offset)),
    fileLike(`${name}/run.json`, JSON.stringify(run)),
    fileLike(`${name}/training/trace.json`, JSON.stringify(trace)),
    fileLike(`${name}/training/snap_00000.f16.bin`, new Uint8Array(2 * offset)),
  ];
}

async function load(files) {
  const { runs, errors } = await runsFromFiles(files, { defaultVocab: builtin.vocab });
  assert.deepEqual(errors, []);
  return runs[0];
}

// ---------------------------------------------------------------- un DOM finto, per la scheda

/** Quello che el() di core/dom.js usa del DOM. innerHTML è vietato: il testo dei file deve restare testo. */
class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attributes = new Map();
    this.dataset = {};
    this.style = {};
    this.className = "";
  }
  set innerHTML(_value) {
    throw new Error("innerHTML usato con un testo che viene da un file");
  }
  setAttribute(key, value) {
    this.attributes.set(key, String(value));
  }
  append(...nodes) {
    this.children.push(...nodes);
  }
  addEventListener() {}
  get textContent() {
    return this.children.map((c) => (c instanceof FakeElement ? c.textContent : String(c))).join("");
  }
  *walk() {
    yield this;
    for (const c of this.children) if (c instanceof FakeElement) yield* c.walk();
  }
}
globalThis.document = { createElement: (tag) => new FakeElement(tag) };

/** Le voci dell'elenco "Note" della scheda. */
function cardNotes(card) {
  const facts = [...card.walk()].filter((e) => e.className === "lab-fact");
  const notes = facts.find((f) => f.children[0]?.textContent === "Note");
  return notes ? [...notes.walk()].filter((e) => e.tagName === "LI") : [];
}

// ---------------------------------------------------------------- note della storia del training

test("note di trace.json lette come testo non fidato: solo stringhe, su una riga, accorciate, al più 20", () => {
  for (const t of [null, undefined, 3, "x", [], {}, { notes: null }, { notes: { a: 1 } }, { notes: [1, null, "", "  "] }]) assert.deepEqual(traceNotes(t), []);
  assert.deepEqual(traceNotes({ notes: "una sola" }), ["una sola"]);
  assert.deepEqual(traceNotes({ notes: ["a\nb\t c\u0000d\u0085e\u3000f"] }), ["a b c d e f"]);
  // surrogati spaiati \u2192 "\ufffd" come _one_line di gpt/runs.py; le coppie valide (\ud83d\ude00) restano
  assert.deepEqual(traceNotes({ notes: ["a \ud800 b \udfff c \ud83d\ude00"] }), ["a \ufffd b \ufffd c \ud83d\ude00"]);
  assert.equal(traceNotes({ notes: ["x".repeat(100_000)] })[0].length, 2000);
  assert.equal(traceNotes({ notes: Array.from({ length: 5000 }, (_, i) => `nota ${i}`) }).length, 20);
});

test("scheda del run: le note del trace che run.json non ha (un export vecchio), con il prefisso, una volta sola", async () => {
  const old = await load(exportedRun("vecchio", { info: { notes: ["nota dell'autore", "classi dei moduli dal codice attuale"] } }));
  assert.deepEqual(old.trace.notes, [SDPA_NOTE, GROUPS_NOTE], "parseTrace tiene le note del file");
  const { notes, hidden } = runNotes(old);
  assert.deepEqual(notes, ["nota dell'autore", "classi dei moduli dal codice attuale", `${TRAINING_NOTE_PREFIX}${SDPA_NOTE}`, `${TRAINING_NOTE_PREFIX}${GROUPS_NOTE}`]);
  assert.equal(hidden, 0);

  // export_run di gpt/runs.py le copia già in run.json (anche accorciate): la scheda non le ripete
  const long = `attention: ${"lunga ".repeat(500)}`;
  const copied = [`${TRAINING_NOTE_PREFIX}${SDPA_NOTE}`, `${(TRAINING_NOTE_PREFIX + long.trim()).slice(0, 1999)}…`, "classi dei moduli dal codice attuale"];
  const fresh = await load(exportedRun("nuovo", { info: { notes: copied }, traceNotes: [SDPA_NOTE, long] }));
  assert.deepEqual(runNotes(fresh).notes, copied);
  // un surrogato spaiato nel trace: gpt/runs.py lo scrive "�" in run.json, e la nota si riconosce lo stesso
  const odd = await load(exportedRun("strano", { info: { notes: [`${TRAINING_NOTE_PREFIX}attention: x \ufffd y`] }, traceNotes: ["attention: x \ud800 y"] }));
  assert.deepEqual(runNotes(odd).notes, [`${TRAINING_NOTE_PREFIX}attention: x \ufffd y`]);

  // tante note: le prime 12 e quante ne restano fuori
  const many = await load(exportedRun("tante", { info: { notes: Array.from({ length: 15 }, (_, i) => `nota ${i}`) } }));
  const shown = runNotes(many);
  assert.equal(shown.notes.length, 12);
  assert.equal(shown.hidden, 5, "15 di run.json + 2 del trace");
  const card = runCard({ name: "tante", description: "" }, many, builtin);
  assert.equal(cardNotes(card).at(-1).textContent, "… e altre 5, non mostrate qui");
});

test("scheda del run: una nota con dell'HTML resta testo (textContent, mai innerHTML)", async () => {
  const run = await load(exportedRun("<b>html</b>", { info: { notes: ['<script>globalThis.hacked = true</script>'] }, traceNotes: [HTML_NOTE] }));
  const card = runCard({ name: run.name, description: "<i>desc</i>" }, run, builtin);
  const items = cardNotes(card);
  assert.deepEqual(items.map((li) => li.textContent), ['<script>globalThis.hacked = true</script>', `${TRAINING_NOTE_PREFIX}${HTML_NOTE}`]);
  for (const li of items) assert.ok(li.children.every((c) => typeof c === "string"), "nessun elemento creato dal testo della nota");
  assert.equal(globalThis.hacked, undefined);
  assert.ok([...card.walk()].some((e) => e.tagName === "H4" && e.textContent === "<b>html</b>"), "anche il nome è testo");
});

// ---------------------------------------------------------------- entropia dell'attention che manca

test("entropia dell'attention che manca: il perché, con la nota del trace", async () => {
  assert.equal(missingAttentionText(null), null);
  assert.equal(missingAttentionText(builtin.trace), null, "il GPT del corso ce l'ha");
  const run = await load(exportedRun("sdpa"));
  assert.equal(attentionEntropy(run.trace), null);
  const text = missingAttentionText(run.trace);
  assert.match(text, /^la registrazione del training non ha misurato l'entropia dell'attention \(in trace\.json manca attention_entropy\)/);
  // la nota tra «» e isolata (FSI … PDI): si vede dove comincia e finisce il testo del file
  assert.ok(text.includes("e la sua nota dice perché: «\u2068nessun modulo del modello espone il metodo attention_weights"), text);
  assert.ok(text.endsWith("\u2069»"), text);
  // un U+202E (da destra a sinistra) in una nota resta chiuso nella nota: non rovescia il resto della frase
  const rlo = missingAttentionText({ snapshots: [{ step: 0 }], notes: ["attention: \u202eexe.png", "attention: seconda"] });
  assert.ok(rlo.endsWith("«\u2068\u202eexe.png\u2069» «\u2068seconda\u2069»"), rlo);
  assert.ok(!text.includes("gruppi:"), "solo le note sull'attention");
  assert.match(missingAttentionText({ snapshots: [] }), /non ha snapshot/);
  assert.match(missingAttentionText({ snapshots: [{ step: 0 }], notes: "gruppi: altro" }), /non dice perché/);
  assert.match(missingAttentionText({ snapshots: [{ step: 0 }], notes: [HTML_NOTE] }), /<img src=x/, "il testo della nota passa com'è: lo mostra textContent");
});

/** Il pannello dei grafici del nucleo (ChartsPanel), quanto basta: sezioni, card, dati, descrizioni. */
function fakeCharts() {
  const make = (options) => ({ type: options.type, options: { ...options }, data: null, setData(d) { this.data = d; }, setHighlight() {}, clear() { this.data = null; } });
  const sectionOf = (title, description) => ({
    title,
    description,
    charts: new Map(),
    chart(id, options = {}) {
      const c = this.charts.get(id);
      if (c && c.type === (options.type ?? c.type)) {
        for (const k of ["title", "description"]) if (options[k] !== undefined) c.options[k] = options[k];
        return c;
      }
      const fresh = make(options);
      this.charts.set(id, fresh);
      return fresh;
    },
    clear() {
      this.charts.clear();
    },
  });
  return {
    sections: new Map(),
    section(id, title = id, description = "") {
      let s = this.sections.get(id);
      if (!s) this.sections.set(id, (s = sectionOf(title, description)));
      else if (arguments.length > 1) Object.assign(s, { title, description });
      return s;
    },
    open() {},
  };
}

/** Lo stato del Laboratorio, quanto ne usa charts.js: run A e B, i loro nomi, gli eventi. */
function fakeState(runs) {
  const listeners = new Map();
  return {
    runA: null,
    runB: null,
    step: null,
    stepB: null,
    run: (id) => runs.get(id) ?? null,
    entry: (id) => (runs.has(id) ? { name: runs.get(id).name } : null),
    on(type, fn) {
      listeners.set(type, [...(listeners.get(type) ?? []), fn]);
      return () => {};
    },
    emit(type) {
      for (const fn of listeners.get(type) ?? []) fn({}, this);
    },
  };
}

test("grafici: dove il trace non ha l'entropia dell'attention, niente heatmap vuota ma il perché con la nota", async () => {
  const sdpa = await load(exportedRun("Attention SDPA", { traceNotes: [HTML_NOTE.replace("attention: ", "attention: SDPA ") , SDPA_NOTE] }));
  const runs = new Map([["corso", builtin], ["sdpa", sdpa]]);
  const state = fakeState(runs);
  state.runA = "corso";
  const charts = fakeCharts();
  installAnalysisCharts({ state, ui: { charts } }, { bench: { of: async () => null } });
  const attention = charts.sections.get("lab-an-attention");
  assert.ok(attention, "la sezione dell'entropia dell'attention");
  assert.deepEqual([...attention.charts.keys()], ["att-a"], "il GPT del corso ha la sua heatmap");
  assert.equal(attention.charts.get("att-a").data.rows, 16);
  assert.ok(!attention.description.includes("Nessuna heatmap"));

  state.runA = "sdpa";
  state.runB = "corso";
  state.emit("runA");
  assert.deepEqual([...attention.charts.keys()], ["att-b"], "A non ha l'entropia: resta solo la heatmap di B");
  // il nome viene da run.json: isolato (FSI … PDI) dentro la frase
  assert.match(attention.description, /Nessuna heatmap per A \(\u2068Attention SDPA\u2069\): la registrazione del training non ha misurato l'entropia dell'attention/);
  assert.ok(attention.description.includes("nessun modulo del modello espone il metodo attention_weights"), attention.description);
  assert.ok(attention.description.includes('<img src=x onerror="globalThis.hacked = true">'), "testo semplice: il nucleo lo mette con textContent");
  assert.match(attention.charts.get("att-b").options.description, /^Una riga per head/, "senza «come sopra»: sopra non c'è niente");

  state.runB = "sdpa"; // stesso run: B è un momento della stessa storia
  state.emit("runB");
  assert.deepEqual([...attention.charts.keys()], []);
  assert.match(attention.description, /B è un momento dello stesso training di A: vale lo stesso\.$/);

  state.runA = "corso";
  state.runB = null;
  state.emit("runA");
  assert.deepEqual([...attention.charts.keys()], ["att-a"], "la heatmap torna con un run che ce l'ha");
});

// ---------------------------------------------------------------- famiglie dei gruppi

test("famiglie dei gruppi: '*' solo al posto dell'indice del blocco (norm1 e norm2 restano due famiglie)", () => {
  assert.deepEqual(blockFamily("blocks.0.norm1"), { id: "blocks.*.norm1", index: "0" });
  assert.deepEqual(blockFamily("transformer.h.11.attn"), { id: "transformer.h.*.attn", index: "11" });
  assert.deepEqual(blockFamily("0.1"), { id: "*.1", index: "0" }, "solo il primo livello numerato");
  assert.equal(blockFamily("ln1"), null);
  assert.equal(blockFamily("block.sa"), null);
  assert.equal(blockFamily("blocks.٣.norm"), null, "solo cifre ASCII, come gpt/recording.py");

  const groups = ["token_embedding", "position_embedding", "blocks.0.norm1", "blocks.0.attn", "blocks.0.norm2", "blocks.0.mlp", "blocks.1.norm1", "blocks.1.attn", "blocks.1.norm2", "blocks.1.mlp", "ln1", "ln2", "ln_f", "lm_head"];
  const families = groupFamilies({ groups });
  assert.deepEqual(families.map((f) => f.id), ["blocks.*.norm1", "blocks.*.attn", "blocks.*.norm2", "blocks.*.mlp", "outside"]);
  for (const f of families.slice(0, 4)) assert.deepEqual(f.members.map((m) => m.label), ["blocco 0", "blocco 1"], f.id);
  // prima: ln1 e ln2 erano "blocco 1" e "blocco 2" della famiglia "ln*"
  assert.deepEqual(families.at(-1).members.map((m) => m.label), ["embedding dei token", "embedding delle posizioni", "ln1", "ln2", "ln_f", "lm_head"]);
  assert.equal(families.at(-1).label, "fuori dai blocchi numerati");
  assert.equal(groupFamilies(builtin.trace).at(-1).label, "fuori dai blocchi", "il GPT del corso: embedding, ln_f, lm_head");

  // un solo blocco riusato (gruppi di gpt/recording.py, uguali a groupOf): niente blocchi inventati, e i
  // moduli del blocco in una famiglia loro (prima: 8 linee insieme agli embedding, al limite dei colori)
  const shared = ["token_embedding", "position_embedding", "block.ln1", "block.sa", "block.ln2", "block.ffwd", "ln_f", "lm_head"];
  const sharedFamilies = groupFamilies({ groups: shared });
  assert.deepEqual(sharedFamilies.map((f) => [f.id, f.label, f.members.map((m) => m.label)]), [
    ["module:block", "block", ["ln1", "sa", "ln2", "ffwd"]],
    ["outside", "fuori dai blocchi", ["embedding dei token", "embedding delle posizioni", "ln_f", "lm_head"]],
  ]);
  assert.deepEqual(sharedFamilies[0].members.map((m) => m.group), ["block.ln1", "block.sa", "block.ln2", "block.ffwd"]);
  // un modulo con un gruppo solo resta fuori; un modulo che si chiama come una famiglia del corso resta a parte
  assert.deepEqual(groupFamilies({ groups: ["sa_heads.proj", "ffwd.net", "ffwd.gate", "blocks.0.ffwd", "norm_f"] }).map((f) => [f.id, f.members.map((m) => m.label)]), [
    ["module:ffwd", ["net", "gate"]],
    ["ffwd", ["blocco 0"]],
    ["outside", ["sa_heads.proj", "norm_f"]],
  ]);
  assert.deepEqual(moduleFamily("transformer.wte"), { id: "transformer", label: "wte" });
  for (const g of ["norm_f", "blocks.0.norm1", ".x", "x.", ""]) assert.equal(moduleFamily(g), null, g);
  assert.equal(groupOf("block.sa.heads.0.key.weight"), "block.sa");
  assert.equal(groupOf("block.ffwd.net.2.bias"), "block.ffwd");
  assert.equal(groupOf("blocks.0.norm2.weight"), "blocks.0.norm2");
});
