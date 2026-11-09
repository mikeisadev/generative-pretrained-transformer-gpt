// Test della vista "Architettura LLM moderni completa" sui moduli puri e di soli dati (niente
// three.js né DOM): compressione delle ripetizioni sui 3 modelli, conti che tornano con quelli della
// ricerca verificata (llm_specs.json → derived), ricostruzione del checkpoint, ancore e id univoci,
// indice della ricerca globale caricato senza la scena.
import assert from "node:assert/strict";
import test from "node:test";

import { FULL_ACTION_IDS, FULL_ACTIONS, FULL_LEVELS, FULL_SECTIONS, MARKER_LABELS } from "../js/tabs/modern-llm/full/catalog.js";
import { FULL_PATH } from "../js/tabs/modern-llm/full/content.js";
import { anatomyLayout, macroLayout, stripLayout } from "../js/tabs/modern-llm/full/layout.js";
import { anatomyCards, anchorsOf, buildStrip, cardState, compressLayers, formatCount, formatLayers, layerByNumber, layerForAnchor, layerList, paramTotals } from "../js/tabs/modern-llm/full/model.js";
import { MODEL_BY_ID, MODELS, PRIMARY_MODEL_ID } from "../js/tabs/modern-llm/full/models/index.js";
import { CAPABILITY_FACTS, LANDSCAPE, NAME_CHECK } from "../js/tabs/modern-llm/full/research.js";
import { FULL_SNIPPETS } from "../js/tabs/modern-llm/full/snippets.js";
import { enumerateCheckpoint } from "../js/tabs/modern-llm/full/tensors.js";
import { OVERVIEW_PATH, PATH } from "../js/tabs/modern-llm/content.js";
import { REGIONS } from "../js/tabs/modern-llm/regionData.js";
import { ACTION_IDS } from "../js/tabs/modern-llm/actionCatalog.js";

const V41 = MODEL_BY_ID.get("deepseek-v4.1-flash");
const KIMI = MODEL_BY_ID.get("kimi-k3.1-pro");
const OSS = MODEL_BY_ID.get("gpt-oss-120b");

/** Descrizione compatta dell'uscita di compressLayers: "S×2 | 3×[F,R×5] | …". */
function describe(items) {
  return items.map((i) => (i.kind === "run" ? `${i.key}×${i.count}` : `${i.repeat}×[${i.unit.map((u) => (u.count > 1 ? `${u.key}×${u.count}` : u.key)).join(",")}]`)).join(" | ");
}

test("compressione delle ripetizioni: i tre modelli come li descrive la ricerca", () => {
  // DeepSeek-V4.1-Flash: encoder = 2 SWA + 3 × [Full + 5 Reuse]; decoder = Full + 3 Reuse + 4 × [Reindex + 3 Reuse]
  assert.equal(describe(compressLayers(V41.layers.kinds)), "S×2 | 3×[F,R×5] | D×1 | U×3 | 4×[X,U×3]");
  // Kimi: 23 blocchi da [3 KDA + 1 Gated MLA] + un Gated MLA finale (69 + 24)
  assert.equal(describe(compressLayers(KIMI.layers.kinds)), "23×[K×3,M] | M×1");
  // gpt-oss: 36 layer, finestra e densa alternate
  assert.equal(describe(compressLayers(OSS.layers.kinds)), "18×[B,G]");
});

test("compressione: copre ogni layer una volta sola, ricostruisce la sequenza, casi limite", () => {
  const expand = (items, keys) => {
    const out = [];
    for (const it of items) {
      if (it.kind === "run") for (let i = 0; i < it.count; i++) out.push(it.key);
      else for (let r = 0; r < it.repeat; r++) for (const u of it.unit) for (let i = 0; i < u.count; i++) out.push(u.key);
    }
    assert.equal(out.join(""), Array.from(keys).join(""));
  };
  for (const keys of [V41.layers.kinds, KIMI.layers.kinds, OSS.layers.kinds, "A", "AB", "AAAA", "ABABAB", "AABAAB", "ABCABCABCD", "XYZ"]) expand(compressLayers(keys), keys);
  assert.deepEqual(compressLayers(""), []);
  assert.equal(describe(compressLayers("AAAA")), "A×4");
  assert.equal(describe(compressLayers("ABABAB")), "3×[A,B]");
  // a parità di costo vince l'unità che comincia con il tipo più raro (la "testa" del gruppo)
  assert.equal(describe(compressLayers("FRRRXRRRXRRR")), "F×1 | R×3 | 2×[X,R×3]");
});

test("schema dei layer: coincide con i campi del config da cui la ricerca lo ricava", () => {
  // V4.1: compress_ratio 0 → solo finestra; sorgenti KV → Full (decoder dal 20); sorgenti dell'indexer → Reindex
  const c = V41.config;
  const derived = Array.from({ length: c.layers }, (_, l) => {
    if (c.compressRatios[l] === 0) return "S";
    if (c.kvSourceLayers.includes(l)) return l >= 20 ? "D" : "F";
    if (c.indexSourceLayers.includes(l)) return "X";
    return l >= 20 ? "U" : "R";
  }).join("");
  assert.equal(V41.layers.kinds, derived);
  assert.equal(c.compressRatios.length, c.layers + c.dspark.layers);
  // Kimi: full_attn_layers → Gated MLA, gli altri KDA (numerati da 1)
  const kimi = Array.from({ length: KIMI.config.layers }, (_, i) => (KIMI.config.fullAttnLayers.includes(i + 1) ? "M" : "K")).join("");
  assert.equal(KIMI.layers.kinds, kimi);
  assert.equal([...KIMI.layers.kinds].filter((k) => k === "K").length, 69);
  // gpt-oss: finestra nei layer pari
  assert.equal(OSS.layers.kinds, Array.from({ length: 36 }, (_, l) => (l % 2 === 0 ? "B" : "G")).join(""));
  for (const m of MODELS) {
    assert.equal(m.layers.kinds.length, m.config.layers, `${m.id}: numero di layer`);
    for (const k of new Set(m.layers.kinds)) assert.ok(m.kinds[k], `${m.id}: tipo ${k} senza descrizione`);
  }
});

test("parametri: totali e attivi coincidono con i conti della ricerca", () => {
  const v = paramTotals(V41);
  assert.equal(v.backbone, 551880850544); // derived.backbone_total (paper: 552B)
  assert.equal(v.checkpoint, 763205315794); // + Engram 196,61B + DSpark 14,23B + visione 0,49B
  assert.equal(formatCount(v.active, { digits: 2 }), "16,13B"); // decode (paper: 16B)
  assert.equal(formatCount(v.prefill, { digits: 2 }), "7,89B"); // prefill solo encoder (paper: 8B)
  const routed = v.components.find((c) => c.id === "routed");
  assert.equal(routed.params, 543581798400);
  assert.equal(routed.share.toFixed(3), "0.985"); // 98,5% del backbone
  assert.equal(v.components.find((c) => c.id === "attention").params, 5126564864);
  assert.equal(v.components.find((c) => c.id === "embedding").params, 661913600);

  const k = paramTotals(KIMI);
  assert.equal(k.backbone, 2779480454752); // 2.779,48B (Table 1: 2,78T)
  assert.equal(k.active, 104185591392); // 104,19B (Table 1: 104,2B)
  assert.equal(k.components[0].share.toFixed(2), "0.98");

  const o = paramTotals(OSS);
  assert.equal(o.backbone, 116829156672); // model card: 116,83B
  assert.equal(o.active, 5132849472); // model card: 5,13B
  assert.equal(o.prefill, null);
});

test("checkpoint di V4.1 ricostruito tensore per tensore: 96.085 tensori, 510,3 GB", () => {
  const c = enumerateCheckpoint(V41);
  assert.equal(c.tensors, 96085); // ricetta vLLM
  assert.equal(c.bytes, 510286023000); // 510,3 GB
  assert.equal(c.params, 763205315794);
  const g = Object.fromEntries(c.groups.map((x) => [x.id, x]));
  assert.equal(g.routed.tensors, 92160); // 40 × 384 × 3 pesi + le loro scale
  assert.equal(g.routed.params, 543581798400);
  assert.equal(g["engram-tables"].params, 196613849600);
  assert.equal(g.dspark.params, 14225362530);
  assert.equal(g.vision.params, 485253120);
  // byte per precisione dei dati (GB con un decimale) = totale della ricostruzione
  const gb = V41.precision.reduce((a, p) => a + p.gb, 0);
  assert.ok(Math.abs(gb - c.bytes / 1e9) < 0.1, `${gb} contro ${c.bytes / 1e9}`);
});

test("stime dei byte di Kimi e gpt-oss: tornano con ricetta vLLM e model card", () => {
  const sum = (m) => m.precision.reduce((a, p) => a + p.gb, 0);
  assert.ok(Math.abs(sum(KIMI) - 1561) < 2, `Kimi: ${sum(KIMI)} GB contro ≈ 1.561`);
  assert.ok(Math.abs(sum(OSS) / 1.073741824 - 60.8) < 0.3, `gpt-oss: ${sum(OSS) / 1.073741824} GiB contro 60,8`);
  for (const m of [KIMI, OSS]) assert.ok(m.precision.every((p) => p.estimate), `${m.id}: le stime vanno dichiarate`);
});

test("striscia dei Blocchi: gruppi «× N», segmenti, tutti i layer rappresentati", () => {
  for (const m of MODELS) {
    for (const expanded of [new Set(), "all"]) {
      const strip = buildStrip(m, { expanded });
      const covered = new Set(strip.blocks.flatMap((b) => b.layers));
      assert.equal(covered.size, m.config.layers, `${m.id}: ogni layer in un blocco`);
      for (const s of strip.segments) assert.ok(s.from !== undefined && s.to >= s.from, `${m.id}: segmento ${s.id}`);
      const layout = stripLayout(strip, m);
      for (let i = 1; i < layout.blocks.length; i++) {
        const a = layout.blocks[i - 1];
        const b = layout.blocks[i];
        assert.ok(a.x + a.w / 2 <= b.x - b.w / 2, `${m.id}: blocchi ${i - 1} e ${i} sovrapposti`);
      }
      assert.ok(layout.box.minX < layout.blocks[0].x && layout.box.maxX > layout.blocks.at(-1).x);
    }
  }
  const v = buildStrip(V41);
  assert.deepEqual(v.groups.map((g) => g.label), ["× 2", "× 3", "× 3", "× 4"]);
  assert.equal(v.groups[1].sub.startsWith("layer 2-19"), true);
  assert.deepEqual(v.segments.map((s) => s.id), ["encoder", "decoder"]);
  // Engram al 14: segnato sul blocco Full del gruppo compresso
  const full = v.blocks.find((b) => b.kind === "F");
  assert.deepEqual(full.layers, [2, 8, 14]);
  assert.deepEqual(full.flags.find((f) => f.id === "engram").layers, [14]);
  assert.equal(formatLayers([3, 4, 5, 6, 7, 9, 10]), "3-7, 9-10");
});

test("anatomia: schede presenti per tipo di layer, nessuna sovrapposizione", () => {
  for (const m of MODELS) {
    const seen = new Set();
    for (const layer of layerList(m)) {
      const sig = `${layer.kind}|${[...layer.flags].join()}`;
      if (seen.has(sig)) continue;
      seen.add(sig);
      const L = anatomyLayout(m, layer);
      const rects = [...L.cards.entries()];
      for (let i = 0; i < rects.length; i++)
        for (let j = i + 1; j < rects.length; j++) {
          const [ia, a] = rects[i];
          const [ib, b] = rects[j];
          const overlap = Math.abs(a.x - b.x) * 2 < a.w + b.w - 0.01 && Math.abs(a.y - b.y) * 2 < a.h + b.h - 0.01;
          assert.ok(!overlap, `${m.id} layer ${layer.number}: ${ia} e ${ib} si sovrappongono`);
        }
      assert.ok(L.cards.has(m.anatomy.attention.core.id), `${m.id}: manca il nucleo dell'attention`);
    }
  }
  // V4.1: il compressore lavora nei Full, è riusato (tratteggiato) nei Reuse, non c'è nei layer solo-finestra
  const compressor = anatomyCards(V41).find((c) => c.id === "compressor");
  assert.equal(cardState(compressor, layerByNumber(V41, 2)), "on");
  assert.equal(cardState(compressor, layerByNumber(V41, 5)), "ghost");
  assert.equal(cardState(compressor, layerByNumber(V41, 0)), "off");
  // Engram solo nei layer 1 e 14; FFN densa solo nel layer 1 di Kimi
  assert.ok(anatomyLayout(V41, layerByNumber(V41, 14)).branch);
  assert.equal(anatomyLayout(V41, layerByNumber(V41, 15)).branch, null);
  assert.equal(anatomyLayout(KIMI, layerByNumber(KIMI, 1)).grid, null);
  assert.ok(anatomyLayout(KIMI, layerByNumber(KIMI, 2)).grid.count === 896);
});

test("i 16 meccanismi: ogni modello dice dove vive ognuno, su un'ancora che esiste", () => {
  assert.equal(REGIONS.length, 16);
  for (const m of MODELS) {
    const anchors = anchorsOf(m);
    for (const r of REGIONS) {
      const mech = m.mechanisms[r.id];
      assert.ok(mech, `${m.id}: manca il meccanismo ${r.id}`);
      assert.ok(anchors.has(mech.at), `${m.id}/${r.id}: ancora inesistente ${mech.at}`);
      assert.ok(["present", "absent", "study"].includes(mech.status), `${m.id}/${r.id}: stato ${mech.status}`);
      assert.ok(mech.text.length > 30, `${m.id}/${r.id}: testo`);
      if (mech.status === "absent") assert.match(mech.text, /vivrebbe|entrerebbero/i, `${m.id}/${r.id}: va detto che il modello non lo usa`);
      if (mech.at.startsWith("detail:")) {
        const n = layerForAnchor(m, mech.at, mech.layer ?? null);
        assert.ok(layerByNumber(m, n), `${m.id}/${r.id}: layer ${n}`);
      }
    }
    assert.equal(Object.keys(m.mechanisms).length, 16, `${m.id}: meccanismi in più`);
  }
  for (const r of REGIONS) assert.ok(MARKER_LABELS[r.id], `etichetta del segnaposto per ${r.id}`);
  // ciò che manca davvero: RoPE in Kimi (NoPE), la multi-token prediction in gpt-oss, le mesh in tutti
  assert.equal(KIMI.mechanisms.rope.status, "absent");
  assert.equal(OSS.mechanisms.spec.status, "absent");
  for (const m of MODELS) assert.equal(m.mechanisms.mesh.status, "absent");
});

test("dati dei modelli: fonti, cose da verificare, nessun riferimento rotto", () => {
  for (const m of MODELS) {
    assert.ok(m.sources.length >= 3 && m.sources.every((s) => /^https:\/\//.test(s.url) && s.title && s.note), `${m.id}: fonti`);
    const doubts = new Set(m.unverified.map((u) => u.id));
    for (const s of m.pipeline) for (const id of s.unverified ?? []) assert.ok(doubts.has(id), `${m.id}: pipeline ${s.id} cita ${id}`);
    assert.ok(m.unverified.every((u) => u.value && u.why.length > 20), `${m.id}: spiegazioni`);
    for (const c of m.caches) assert.ok(c.layers === "all" || typeof c.layers === "string" || c.layers.length, `${m.id}: cache ${c.id}`);
  }
  // i valori segnati dalla ricerca come NON verificati sono marcati
  const v41Doubts = V41.unverified.map((u) => u.id);
  for (const id of ["shards", "sparse-64k", "context-34t", "kv-qat"]) assert.ok(v41Doubts.includes(id), `V4.1: ${id} va marcato da verificare`);
  assert.equal(PRIMARY_MODEL_ID, "deepseek-v4.1-flash");
  assert.equal(LANDSCAPE.length, 15);
  assert.ok(LANDSCAPE.filter((r) => !r.verified).every((r) => r.check === "S"), "le righe solo stampa sono quelle da verificare");
  assert.ok(NAME_CHECK.some((n) => n.asked === "DeepSeek V4.1 Pro" && !n.exists));
  assert.ok(CAPABILITY_FACTS.find((f) => f.id === "emergent").caution.includes("Schaeffer"), "la critica di Schaeffer et al. accompagna le abilità emergenti");
});

test("percorso, azioni, viste: id univoci e collegamenti esistenti", () => {
  const steps = FULL_PATH.flatMap((s) => s.children);
  assert.ok(steps.length >= 10 && steps.length <= 14, `passi: ${steps.length}`);
  const overviewSteps = new Set(OVERVIEW_PATH.flatMap((s) => s.children.map((c) => c.id)));
  const anchors = anchorsOf(V41);
  for (const item of steps) {
    assert.ok(item.id.startsWith("full-"), `${item.id}: prefisso`);
    assert.ok(!overviewSteps.has(item.id), `${item.id}: id già usato dalla vista d'insieme`);
    if (item.anchor) assert.ok(anchors.has(item.anchor), `${item.id}: ancora ${item.anchor}`);
    if (item.play) assert.ok(FULL_ACTION_IDS.has(item.play), `${item.id}: azione ${item.play}`);
    if (item.code) assert.ok(FULL_SNIPPETS[item.code], `${item.id}: codice ${item.code}`);
    if (item.layer !== undefined) assert.ok(layerByNumber(V41, item.layer), `${item.id}: layer ${item.layer}`);
    assert.ok(!/[\u{1F300}-\u{1FAFF}]/u.test(item.text), `${item.id}: niente emoji nei testi didattici`);
  }
  // PATH del tab = vista d'insieme + architettura completa, id tutti diversi
  const all = PATH.flatMap((s) => s.children.map((c) => c.id));
  assert.equal(new Set(all).size, all.length);
  for (const a of FULL_ACTIONS) assert.ok(!ACTION_IDS.has(a.id) && a.id.startsWith("full-"), `${a.id}: id d'azione`);
  assert.equal(FULL_ACTION_IDS.size, FULL_ACTIONS.length);
  const views = [...FULL_LEVELS.map((l) => l.id), ...FULL_SECTIONS.map((s) => s.id)];
  assert.equal(new Set(views).size, views.length);
  for (const s of FULL_SECTIONS) assert.ok(anchorsOf(V41).has(s.anchor), `vista ${s.id}: ancora ${s.anchor}`);
  // estratti veri: file e riga d'origine; pseudocodice dichiarato
  for (const [key, s] of Object.entries(FULL_SNIPPETS)) {
    assert.ok(s.code.length > 40 && s.title && s.file, `codice ${key}`);
    if (/^pseudocodice/.test(s.file)) assert.match(s.code, /NON è il codice ufficiale|Conti della ricerca/, `${key}: pseudocodice dichiarato`);
  }
});

test("Panoramica: fasi in ordine con i pesi al centro, scala con la striscia", () => {
  for (const m of MODELS) {
    const L = macroLayout(m, 1);
    assert.equal(L.stages.length, m.pipeline.length);
    assert.equal(L.stages.find((s) => s.id === "weights").x, 0);
    for (let i = 1; i < L.stages.length; i++) assert.ok(L.stages[i].x - L.stages[i].w / 2 > L.stages[i - 1].x + L.stages[i - 1].w / 2);
    const big = macroLayout(m, 3);
    assert.equal(big.stages[0].w, 3 * L.stages[0].w);
  }
});

test("indice della ricerca: voci della vista completa, nessun duplicato, niente three.js", async () => {
  const { searchIndex } = await import("../js/tabs/modern-llm/search.js");
  const keys = new Set(searchIndex.map((e) => `${e.kind}:${e.id}`));
  assert.equal(keys.size, searchIndex.length, "voci duplicate");
  for (const id of ["overview", "full", ...FULL_LEVELS.map((l) => l.id), ...FULL_SECTIONS.map((s) => s.id), ...MODELS.map((m) => `full-model-${m.id}`), ...REGIONS.map((r) => `full-mech-${r.id}`)]) assert.ok(keys.has(`view:${id}`), `manca view:${id}`);
  for (const a of FULL_ACTIONS) assert.ok(keys.has(`action:${a.id}`), `manca action:${a.id}`);
  for (const s of FULL_PATH) for (const c of s.children) assert.ok(keys.has(`step:${c.id}`), `manca step:${c.id}`);
  const terms = searchIndex.find((e) => e.kind === "view" && e.id === "full").keywords.join(" ");
  for (const t of ["Engram", "mHC", "CSA2", "lightning indexer", "DSpark", "KDA", "MXFP4", "safetensors", "KV cache compressa"]) assert.ok(terms.includes(t), `parola chiave ${t}`);
  // l'indice è caricato senza la scena: nessun modulo three.js nel grafo (lo garantisce l'import qui in Node)
  assert.equal(typeof globalThis.document, "undefined");
});
