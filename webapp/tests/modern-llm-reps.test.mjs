// Test delle rappresentazioni del tab "LLM moderni" (moduli puri, niente three.js né DOM):
//   node --test webapp/tests/modern-llm-reps.test.mjs
// Rete neurale (js/tabs/modern-llm/network.js, networkMechanisms.js): validazione del nucleo senza avvisi,
// ordine dei layer e «× N» uguali alla striscia dei Blocchi del modulo dati, pezzi di ogni layer uguali al
// config, forme dichiarate uguali ai numeri del modulo dati, attention causale (finestra e voci già
// complete), ogni token verso esattamente top-k esperti, limiti con margine, onestà nei testi.
// Cervello (brain.js): strict senza avvisi, tutto nell'atlante, fonti citate esistenti.
// Ricerca e installazione: voci view:rete e view:cervello, pannello Cervello prima di Fonti, goto, barra
// dei sottotab sopra il livello alternativo.
// BRAIN_RESEARCH=/percorso/brain_data.json confronta anche i testi del cervello con la ricerca, parola per
// parola (senza, quel test si salta: la ricerca sta fuori dal repository).
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { atlasKind } from "../js/core/brain/atlas.js";
import { NEURON_TYPES } from "../js/core/brain/morphology.js";
import { STRENGTHS, normalizeBrainSpec, safeUrl } from "../js/core/brain/spec.js";
import { decodeUnit } from "../js/core/neuralNet/layout.js";
import { LINK_LIMITS } from "../js/core/neuralNet/links.js";
import { buildNetworkModel } from "../js/core/neuralNet/model.js";
import { LIMITS, normalizeSpec } from "../js/core/neuralNet/spec.js";
import { prepareEntry, search } from "../js/core/search.js";
import { brainSpec } from "../js/tabs/modern-llm/brain.js";
import { buildStrip, compressLayers, formatLayers, layerList } from "../js/tabs/modern-llm/full/model.js";
import { MODEL_BY_ID, PRIMARY_MODEL_ID } from "../js/tabs/modern-llm/full/models/index.js";
import { PROMPT, SHOWN, buildNetworkSpec, layerSectionIds, networkSpec, networkSpecFor, readableText } from "../js/tabs/modern-llm/network.js";
import { drawnLayers, entryCount, entryReady, layerAttention, layerRecipe, selectEntries } from "../js/tabs/modern-llm/networkMechanisms.js";
import { searchIndex } from "../js/tabs/modern-llm/search.js";

const source = (path) => readFile(new URL(path, import.meta.url), "utf8");
const RESEARCH = process.env.BRAIN_RESEARCH ?? "";
const model = MODEL_BY_ID.get(PRIMARY_MODEL_ID);
const c = model.config;
const T = PROMPT.tokens.length;
const it = (n) => Number(n).toLocaleString("it-IT", { useGrouping: "always" });
const net = normalizeSpec(networkSpec);
const built = buildNetworkModel(networkSpec);
const sectionOf = (id) => net.byId.get(id);
const drawn = drawnLayers(model);
const prefix = (d) => `l${d.rep}`;

// ------------------------------------------------------------------------------------------ rete neurale

test("rete: la specifica passa la validazione del nucleo senza avvisi, entro i limiti con ampio margine", () => {
  assert.deepEqual(net.warnings, []);
  assert.equal(built.links.capped, false, "nessun collegamento tagliato dal tetto di sicurezza");
  assert.equal(built.links.scale, 1, "densità come scritte nella specifica");
  assert.ok(net.sections.length < LIMITS.sections / 4, `sezioni ${net.sections.length}`);
  assert.ok(net.links.length < LIMITS.links / 10, `collegamenti ${net.links.length}`);
  assert.ok(net.units < LIMITS.units / 8, `neuroni ${net.units}`);
  assert.ok(built.links.segments.count < LINK_LIMITS.maxSegments / 10, `segmenti ${built.links.segments.count}`);
  assert.ok(built.links.arcs.count < LINK_LIMITS.maxArcs / 4, `archi ${built.links.arcs.count}`);
  assert.ok(built.links.arcs.count > 500, "gli archi di attention e dei residui ci sono");
  assert.deepEqual(net.sources.map((s) => s.url), model.sources.map((s) => s.url), "le fonti sono quelle del modulo dati");
  assert.ok(net.sources.every((s) => /^https:\/\//.test(s.url)));
});

test("rete: i layer disegnati e i «× N» sono la striscia dei Blocchi del modulo dati, nell'ordine vero", () => {
  const strip = buildStrip(model);
  const layerBlocks = strip.blocks.filter((b) => b.type === "layer");
  // un layer disegnato per blocco della striscia, nello stesso ordine e con lo stesso tipo
  assert.deepEqual(drawn.blocks.map((d) => d.rep), layerBlocks.map((b) => b.rep));
  assert.equal(drawn.blocks.map((d) => d.block.kind).join(""), "SFRDUXU", "S×2 | 3×[F, R×5] | D | U×3 | 4×[X, U×3]");
  // la compressione è quella di compressLayers sullo schema dei layer (lo stesso dei Blocchi)
  const items = compressLayers(layerList(model).map((l) => l.kind));
  const expected = items.filter((i) => (i.kind === "run" ? i.count > 1 : true)).map((i) => (i.kind === "run" ? i.count : i.repeat));
  assert.deepEqual(net.repeats.filter((r) => r.id !== "rep.mtp").map((r) => r.count), expected);
  assert.deepEqual(expected, [2, 3, 3, 4]);
  // ogni ripetizione copre esattamente le sezioni dei layer del suo gruppo, e la sua etichetta dice «× N» e i layer veri
  for (const g of strip.groups) {
    const rep = net.repeats.find((r) => r.id === `rep.${g.id}`);
    const ids = drawn.blocks.filter((d) => d.index >= g.from && d.index <= g.to).flatMap((d) => layerSectionIds(prefix(d), d.recipe));
    assert.deepEqual(rep.of.map((s) => s.id), ids, rep.id);
    assert.equal(rep.label, `× ${g.repeat} layer ${formatLayers(g.layers)}`);
  }
  // l'ordine di tutto il flusso: ingresso, layer (con le ripetizioni subito dopo il loro gruppo), uscita, DSpark
  const order = [];
  for (const d of drawn.blocks) {
    order.push(...layerSectionIds(prefix(d), d.recipe));
    if (d.group?.to === d.index) order.push(`rep.${d.group.id}`);
  }
  assert.deepEqual(net.items.map((i) => i.id), ["tok", "emb", "mhc", ...order, "fnorm", "head", "softmax", "mtp.fuse", "mtp.block", "rep.mtp", "mtp.out"]);
  assert.equal(sectionOf("tok").index, 0);
  // il blocco R rappresenta 5 layer in fila dentro il gruppo × 3: la parentesi lo dice
  assert.ok(net.groups.some((gr) => gr.label === "5 layer Reuse" && gr.from.id === "l3.norm1" && gr.to.id === "l3.add2"));
  assert.ok(net.groups.some((gr) => gr.label === "3 layer Reuse" && gr.from.id === "l25.norm1"));
});

test("rete: ogni layer ha i pezzi che il config gli dà (compress_ratios, kv_source, index_source)", () => {
  // la ricetta di ognuno dei 40 layer, rifatta dai campi del config
  for (const l of layerList(model)) {
    const r = layerRecipe(model, l.number);
    const m = c.compressRatios[l.number - model.layers.base];
    assert.equal(r.compress, m, `layer ${l.number}`);
    assert.equal(r.writesKV, m > 0 && c.kvSourceLayers.includes(l.number));
    assert.equal(r.indexes, m > 0 && c.indexSourceLayers.includes(l.number));
    if (m > 0) assert.equal(r.kvSource, Math.max(...c.kvSourceLayers.filter((n) => n <= l.number)));
  }
  assert.deepEqual(drawn.blocks.map((d) => d.recipe.compress), [0, 2, 2, 1, 1, 1, 1]);
  for (const d of drawn.blocks) {
    const p = prefix(d);
    const r = d.recipe;
    assert.equal(net.byId.has(`${p}.gkv`), r.writesKV, `${p}: KV globale solo nei layer sorgente`);
    assert.equal(net.byId.has(`${p}.idx`), r.indexes, `${p}: indexer solo dove si ricalcola la scelta`);
    const incoming = (id) => net.links.filter((k) => k.to.id === id);
    if (r.writesKV && r.compress > 1) {
      const down = incoming(`${p}.gkv`).find((k) => k.pattern === "down");
      assert.ok(down && down.factor === r.compress && sectionOf(`${p}.gkv`).depth.count === T / r.compress, `${p}: ${r.compress} token → 1 voce`);
    }
    if (r.writesKV && r.compress === 1) {
      // nel decoder la voce globale è una proiezione dell'uscita dell'encoder (H20), non dello stato del layer
      const from = incoming(`${p}.gkv`)[0].from.id;
      const holder = drawn.blocks.find((b) => b.block.layers.includes(r.fromSegment.to));
      assert.equal(from, `${prefix(holder)}.add2`);
      assert.equal(r.fromSegment.id, model.layers.prefillSegment);
    }
    // le voci globali arrivano all'attention dal layer sorgente (riuso) o dal layer stesso
    const global = incoming(`${p}.attn`).filter((k) => k.pattern === "attention" && k.from.id.endsWith(".gkv"));
    if (r.compress === 0) assert.equal(global.length, 0, `${p}: solo finestra`);
    else {
      assert.equal(global.length, 1, p);
      assert.equal(global[0].from.id, `${prefix(drawn.blocks.find((b) => b.index === d.kvBlock))}.gkv`);
      assert.ok(global[0].from.order < net.byId.get(`${p}.attn`).order, "la KV riusata viene da sinistra");
    }
    // la finestra locale: dal KV latente dello stesso layer, in ogni layer
    assert.equal(incoming(`${p}.attn`).filter((k) => k.pattern === "attention" && k.from.id === `${p}.kv`).length, 1);
    // residuo, norme per token, router → esperti → somma
    assert.equal(incoming(`${p}.add1`).filter((k) => k.pattern === "residual").length, 1);
    assert.equal(incoming(`${p}.add2`).filter((k) => k.pattern === "residual").length, 1);
    assert.equal(sectionOf(`${p}.norm1`).role, "norm");
    assert.equal(sectionOf(`${p}.norm2`).role, "norm");
    assert.equal(incoming(`${p}.experts`).filter((k) => k.pattern === "route" && k.from.id === `${p}.router`).length, 1);
    assert.equal(net.links.filter((k) => k.pattern === "route" && k.from.id === `${p}.experts` && k.to.id === `${p}.add2`).length, 1);
  }
});

test("rete: le forme dichiarate sono i numeri del modulo dati (sottotitoli e «nel modello vero»)", () => {
  const embedding = model.input.find((b) => b.id === "embedding");
  const head = model.output.find((b) => b.id === "head");
  assert.deepEqual([sectionOf("tok").real, sectionOf("tok").activation], [c.vocab, "onehot"]);
  assert.ok(sectionOf("tok").sub.startsWith(`${it(c.vocab)} voci`));
  assert.ok(sectionOf("emb").sub.startsWith(embedding.sub) && sectionOf("emb").real === c.hidden);
  assert.ok(sectionOf("head").sub.startsWith(head.sub) && sectionOf("head").real === c.vocab);
  assert.equal(sectionOf("mhc").slabs, c.hcMult);
  for (const d of drawn.blocks) {
    const p = prefix(d);
    for (const id of ["norm1", "norm2"]) assert.ok(sectionOf(`${p}.${id}`).sub.includes(it(c.hidden)) && sectionOf(`${p}.${id}`).real === c.hidden);
    const kv = sectionOf(`${p}.kv`);
    assert.ok(kv.sub.startsWith(`${it(c.headDim)} (${it(c.headDim - c.ropeDim)} + ${c.ropeDim} RoPE)`) && kv.real === c.headDim);
    const attn = sectionOf(`${p}.attn`);
    assert.ok(attn.sub.startsWith(`${c.heads} query · ${c.kvHeads} KV head`) && attn.real === c.heads * c.headDim && attn.heads === SHOWN.heads);
    for (const id of ["add1", "add2"]) assert.deepEqual([sectionOf(`${p}.${id}`).slabs, sectionOf(`${p}.${id}`).real], [c.hcMult, c.hcMult * c.hidden]);
    const router = sectionOf(`${p}.router`);
    assert.ok(router.sub.includes(`${c.scoring} → top-${c.topK}`) && router.neurons === SHOWN.experts && router.real === c.experts);
    const experts = sectionOf(`${p}.experts`);
    assert.deepEqual([experts.experts, experts.active, experts.shared, experts.real, experts.router.id], [SHOWN.experts, c.topK, c.sharedExperts, c.expertHidden, `${p}.router`]);
    assert.ok(experts.sub.startsWith(`${it(c.experts)} + ${c.sharedExperts}`) && experts.sub.endsWith(`${c.topK} attivi`));
    if (d.recipe.writesKV) {
      const gkv = sectionOf(`${p}.gkv`);
      assert.ok(gkv.sub.includes(d.recipe.compress > 1 ? `${d.recipe.compress} token → 1 voce · ${it(model.kv.entryBytes)} B` : `da H${d.recipe.fromSegment.to + 1}`), gkv.sub);
    }
    if (d.recipe.indexes) assert.ok(sectionOf(`${p}.idx`).sub.includes(`top-${it(c.indexTopK)}`) && sectionOf(`${p}.idx`).real === c.indexHeads * c.indexHeadDim);
  }
  // DSpark: le feature dei layer bersaglio, i blocchi draft, i token proposti
  assert.deepEqual(sectionOf("mtp.fuse").slabNames, c.dspark.targets.map((n) => `layer ${n}`));
  assert.equal(net.repeats.find((r) => r.id === "rep.mtp").count, c.dspark.layers);
  assert.equal(sectionOf("mtp.out").slabs, c.dspark.blockSize);
  // la scala ridotta è dichiarata con i numeri veri accanto a quelli disegnati
  for (const piece of [`${SHOWN.hidden} numeri per token invece di ${it(c.hidden)}`, `${SHOWN.experts} dei ${it(c.experts)}`, `ne accende ${c.topK}`, `finestra locale di ${SHOWN.window} token invece di ${it(c.window)}`, `${SHOWN.entries} voci globali scelte invece di ${it(c.indexTopK)}`, `${drawn.blocks.length} disegnati dei ${c.layers}`]) {
    assert.ok(networkSpec.scale.includes(piece), piece);
  }
});

test("rete: attention causale (nessun arco verso il futuro), finestra locale e voci globali già complete", () => {
  const { arcs } = built.links;
  const seen = { window: 0, entries: 0 };
  for (let a = 0; a < arcs.count; a++) {
    const link = net.links[arcs.link[a]];
    if (link.pattern !== "attention") continue;
    const key = decodeUnit(link.from, arcs.from[a] - link.from.offset).depth;
    const query = decodeUnit(link.to, arcs.to[a] - link.to.offset).depth;
    if (link.from.id.endsWith(".gkv")) {
      const m = T / link.from.depth.count;
      assert.ok(entryReady(key, query, m), `${link.from.id} → ${link.to.id}: voce ${key} non ancora completa per la query ${query}`);
      seen.entries++;
    } else {
      assert.ok(key <= query, `${link.from.id} → ${link.to.id}: arco verso il futuro (${key} → ${query})`);
      assert.ok(query - key < SHOWN.window, `${link.from.id}: fuori dalla finestra (${key} → ${query})`);
      seen.window++;
    }
  }
  assert.ok(seen.window > 300 && seen.entries > 100, JSON.stringify(seen));
  // nessun arco di attention vuoto: ogni layer ne ha di finestra, e di voci globali se ha una KV globale
  for (const d of drawn.blocks) {
    const linksTo = net.links.filter((k) => k.pattern === "attention" && k.to.id === `${prefix(d)}.attn`).map((k) => k.index);
    for (const li of linksTo) assert.ok(arcs.link.some((x) => x === li), `${prefix(d)}: collegamento ${li} senza archi`);
  }
});

test("meccanismi: pesi di attention puri, causali, con il sink in un'unica softmax; scelta delle voci causale", () => {
  const selection = selectEntries({ tokens: 12, m: 2, k: 2, seed: "prova" });
  assert.deepEqual(selection, selectEntries({ tokens: 12, m: 2, k: 2, seed: "prova" }), "deterministica");
  selection.forEach((chosen, i) => {
    assert.ok(chosen.length <= 2 && chosen.every((e) => entryReady(e, i, 2)), `query ${i}`);
    assert.equal(chosen.length, Math.min(2, Math.floor((i + 1) / 2)), "le k migliori tra quelle pronte");
  });
  assert.equal(entryCount(12, 2), 6);
  const H = 3;
  const W = 4;
  const { window, entries, sink, entryCount: E } = layerAttention({ tokens: 12, heads: H, window: W, m: 2, selection, seed: "prova" });
  for (let h = 0; h < H; h++) {
    for (let i = 0; i < 12; i++) {
      let sum = sink[h * 12 + i];
      for (let j = 0; j < 12; j++) {
        const w = window[(h * 12 + i) * 12 + j];
        if (j > i || i - j >= W) assert.equal(w, 0, `finestra h${h} ${i}←${j}`);
        sum += w;
      }
      for (let e = 0; e < E; e++) {
        const w = entries[(h * 12 + i) * E + e];
        if (!selection[i].includes(e)) assert.equal(w, 0, `voce non scelta h${h} ${i}←${e}`);
        sum += w;
      }
      assert.ok(Math.abs(sum - 1) < 1e-5, "finestra + voci + sink = 1");
      assert.ok(sink[h * 12 + i] > 0 && sink[h * 12 + i] < 1);
    }
  }
  const onlyWindow = layerAttention({ tokens: 8, heads: 2, window: 4, seed: "x" });
  assert.equal(onlyWindow.entries, null);
});

test("rete: ogni token va esattamente ai suoi top-k esperti (più il condiviso), e solo a loro", () => {
  const { segments } = built.links;
  for (const d of drawn.blocks) {
    const E = sectionOf(`${prefix(d)}.experts`);
    const route = built.values.routing.get(E.id);
    assert.equal(route.source, "router", "punteggi presi dal router disegnato");
    assert.equal(route.k, c.topK);
    const dispatch = net.links.find((k) => k.pattern === "route" && k.to === E);
    const combine = net.links.find((k) => k.pattern === "route" && k.from === E);
    const reached = Array.from({ length: T }, () => new Set());
    for (let s = 0; s < segments.count; s++) {
      if (segments.link[s] !== dispatch.index && segments.link[s] !== combine.index) continue;
      const unit = segments.link[s] === dispatch.index ? segments.to[s] : segments.from[s];
      const { slab, depth } = decodeUnit(E, unit - E.offset);
      assert.equal(route.mask[slab * T + depth], 1, `${E.id}: segmento verso un esperto spento (lastra ${slab}, token ${depth})`);
      if (segments.link[s] === dispatch.index && slab >= E.shared) reached[depth].add(slab - E.shared);
    }
    for (let t = 0; t < T; t++) {
      const chosen = new Set(route.experts.slice(t * c.topK, (t + 1) * c.topK));
      assert.equal(chosen.size, c.topK, `${E.id}, token ${t}: ${c.topK} esperti diversi`);
      assert.deepEqual([...reached[t]].sort((a, b) => a - b), [...chosen].sort((a, b) => a - b), `${E.id}, token ${t}: instradato proprio ai suoi top-${c.topK}`);
      let lit = 0;
      for (let slab = 0; slab < E.slabs; slab++) lit += route.mask[slab * T + t];
      assert.equal(lit, c.topK + c.sharedExperts, "top-k + condiviso accesi, gli altri spenti");
    }
  }
});

test("rete: deterministica, token del prompt nella scheda, testi semplici e onesti", async () => {
  const again = buildNetworkModel(buildNetworkSpec());
  assert.deepEqual([...again.links.segments.from], [...built.links.segments.from]);
  assert.deepEqual([...again.links.arcs.weight], [...built.links.arcs.weight]);
  assert.deepEqual([...again.values.raw], [...built.values.raw]);
  assert.ok(T >= 8 && T <= 16);
  assert.deepEqual(net.slices.names, [...PROMPT.tokens], "le fette della scheda sono i token del prompt");
  assert.equal(readableText(PROMPT.tokens.slice(0, PROMPT.prompt)), "Perché il cielo è blu?");
  assert.ok(networkSpec.intro.includes("«Perché il cielo è blu?»") && networkSpec.intro.includes("illustrativa"));
  // i token speciali sono quelli del tokenizer di V4.1 (modulo dati)
  const tokenizer = model.input.find((b) => b.id === "tokenizer").detail.join(" ");
  for (const t of PROMPT.tokens.filter((x) => /^<.*>$/.test(x))) assert.ok(tokenizer.includes(t), t);
  // etichette brevi, niente HTML (a parte i nomi dei token speciali), niente emoji
  for (const s of net.sections) assert.ok(s.label.length < 20, `etichetta lunga: ${s.label}`);
  for (const g of net.groups) assert.ok(g.label.length < 20, `etichetta lunga: ${g.label}`);
  for (const r of net.repeats) assert.ok(r.label.length < 20, `etichetta lunga: ${r.label}`);
  const texts = [net.title, net.intro, net.scale, net.note, ...net.sections.flatMap((s) => [s.label, s.sub, s.text]), ...net.repeats.map((r) => r.text), ...net.groups.map((g) => g.label + g.text), ...net.links.map((k) => k.label)];
  for (const t of texts) {
    assert.ok(!/<(?!｜|think>)\/?[a-z][^>]*>/i.test(t), `HTML in: ${t.slice(0, 60)}`);
    assert.ok(!/\p{Extended_Pictographic}/u.test(t), `emoji in: ${t.slice(0, 60)}`);
  }
  // onestà: ogni sezione dice che i suoi valori (o le scelte) sono illustrativi; router ed esperti anche la scelta
  for (const s of net.sections) assert.match(s.text, /illustrativ/, `${s.id}: manca la frase di onestà`);
  for (const d of drawn.blocks) for (const id of ["router", "experts"]) assert.match(sectionOf(`${prefix(d)}.${id}`).text, /Quali esperti si accendono è illustrativo/);
  assert.match(networkSpec.note, /da verificare/i);
  assert.ok(net.simulated, "la scheda mostra «simulazione didattica»");
  // il modulo è puro: nessun import di three.js o del DOM
  for (const file of ["../js/tabs/modern-llm/network.js", "../js/tabs/modern-llm/networkMechanisms.js"]) assert.doesNotMatch(await source(file), /from "three"|document\.|window\./);
});

test("rete: i testi dicono il ruolo vero di ogni layer disegnato (Reindex sceglie le voci, Reuse no) in italiano corretto", () => {
  for (const d of drawn.blocks) {
    const p = prefix(d);
    const r = d.recipe;
    const attn = sectionOf(`${p}.attn`).text;
    if (r.compress > 0 && !r.writesKV && r.indexes) {
      // il layer Reindex ricalcola la scelta: non «calcola solo la sua finestra»
      assert.match(attn, /^.*Reindex: nessuna KV globale propria/m, p);
      assert.match(attn, new RegExp(`KV globale del layer ${r.kvSource}, ma le voci le sceglie l'indexer di questo layer`), p);
      assert.doesNotMatch(attn, /calcola solo la sua finestra/, p);
    } else if (r.compress > 0 && !r.writesKV) {
      assert.match(attn, new RegExp(`KV globale del layer ${r.kvSource}, con le voci scelte dall'indexer del layer ${r.indexSource}; questo layer calcola solo la sua finestra`), p);
    }
    // in fondo a ogni sezione: quali layer veri rappresenta («Layer 20 (Full ← H20).» se è uno solo)
    const kind = model.kinds[d.block.kind];
    const where = d.block.layers.length > 1 ? `Layer ${kind.label}: ${formatLayers(d.block.layers)}, ne vedi uno (${d.rep === 0 ? "lo 0" : `il ${d.rep}`}).` : `Layer ${d.rep} (${kind.label}).`;
    for (const id of layerSectionIds(p, r)) assert.ok(sectionOf(id).text.includes(where), `${id}: ${where}`);
    // la parentesi del layer elenca i layer veri, non «12 layer · 25 … 39» (sembrerebbe una fila continua)
    const group = net.groups.find((g) => g.from.id === `${p}.norm1`);
    assert.ok(group.text.includes(d.block.layers.length > 1 ? `Layer ${formatLayers(d.block.layers)} (${d.block.layers.length} layer)` : `Layer ${d.rep}: è l'unico di questo tipo.`), group.text);
  }
  const texts = [net.intro, net.scale, net.note, ...net.sections.map((s) => s.text), ...net.repeats.map((r) => r.text), ...net.groups.map((g) => g.text ?? "")];
  for (const t of texts) {
    assert.doesNotMatch(t, /\bil 0\b/, "«lo 0», non «il 0»");
    for (const para of t.split(/\n{2,}/)) assert.match(para, /^[\p{Lu}\d«<+×]/u, `paragrafo che comincia in minuscolo: ${para.slice(0, 40)}`);
  }
  // «× 2»: il nucleo disegna una sola lastra fantasma
  assert.match(net.repeats.find((x) => x.count === 2).text, /la lastra fantasma è l'altro/);
  // l'ordine dei token speciali del prompt non è nelle fonti verificate: lo si dice
  assert.match(networkSpec.intro, /template di chat di V4\.1-Flash non è tra le fonti verificate/);
});

test("rete: gli altri modelli della vista completa sono rifiutati invece di essere disegnati sbagliati", () => {
  for (const id of ["kimi-k3.1-pro", "gpt-oss-120b"]) assert.throws(() => buildNetworkSpec(MODEL_BY_ID.get(id)), /compress_ratios/);
});

test("rete del modello scelto nel selettore: il principale, o la sua rete con la frase che dice perché", () => {
  assert.equal(networkSpecFor(null), networkSpec);
  assert.equal(networkSpecFor(MODEL_BY_ID.get(PRIMARY_MODEL_ID)), networkSpec);
  for (const id of ["kimi-k3.1-pro", "gpt-oss-120b"]) {
    const model = MODEL_BY_ID.get(id);
    const spec = networkSpecFor(model);
    assert.ok(spec.intro.startsWith(`Nel selettore è scelto ${model.name}`), id);
    assert.ok(spec.intro.endsWith(networkSpec.intro), "poi l'introduzione di sempre");
    assert.equal(spec.sections, networkSpec.sections, "la rete è quella del modello principale");
    assert.deepEqual(normalizeSpec(spec).warnings, []);
  }
});

// ------------------------------------------------------------------------------------------ cervello

test("cervello: strict senza avvisi; regioni, marcatori e percorsi nell'atlante", () => {
  const spec = normalizeBrainSpec(brainSpec, { strict: true });
  assert.deepEqual(spec.warnings, []);
  for (const id of brainSpec.regions) assert.equal(atlasKind(id), "region", id);
  for (const id of brainSpec.markers) assert.equal(atlasKind(id), "marker", id);
  for (const p of brainSpec.pathways) assert.equal(atlasKind(typeof p === "string" ? p : p.id), "pathway");
  for (const a of brainSpec.analogies) {
    for (const r of a.regions) assert.ok(["region", "marker"].includes(atlasKind(r)), `${a.id}: ${r}`);
    for (const m of a.markers ?? []) assert.equal(atlasKind(m), "marker");
    for (const p of a.pathways ?? []) assert.equal(atlasKind(p), "pathway");
  }
  for (const g of brainSpec.circuit.groups) assert.equal(atlasKind(g.region), "region", g.id);
  for (const e of brainSpec.circuit.edges) if (e.pathway) assert.equal(atlasKind(e.pathway), "pathway");
  assert.deepEqual(spec.circuit.groups.map((g) => g.id), ["ctx", "d1", "d2", "ind", "gpi", "thal", "act", "da"], "l'anello da sinistra a destra");
});

test("cervello: ogni analogia dice dove non regge, con forza e fonti valide", () => {
  const spec = normalizeBrainSpec(brainSpec, { strict: true });
  assert.equal(spec.analogies.length, 10);
  for (const a of spec.analogies) {
    for (const field of ["ai", "brain", "text", "similar", "different"]) assert.ok(a[field].length > 10, `${a.id}.${field}`);
    assert.ok(a.strength in STRENGTHS, a.id);
    assert.ok(a.sources.length >= 1 && a.sources.every((s) => s.title && safeUrl(s.url)), `${a.id}: fonti`);
  }
  assert.ok(spec.analogies.some((a) => a.id === "mod-moe" && a.strength === "media"), "MoE ↔ specializzazione: forza media");
  assert.ok(spec.analogies.some((a) => a.id === "mod-rl" && a.strength === "forte"), "RL ↔ dopamina: il legame forte");
});

test("cervello: circuito, numeri e fonti citate esistono", async () => {
  const { circuit, facts, sources } = brainSpec;
  for (const g of circuit.groups) {
    assert.ok(g.type in NEURON_TYPES, `${g.id}: tipo di neurone ${g.type}`);
    assert.ok(g.count >= 1 && g.count <= 18 && g.ai, g.id);
  }
  const ids = new Set(circuit.groups.map((g) => g.id));
  for (const e of circuit.edges) assert.ok(ids.has(e.from) && ids.has(e.to) && ["excitatory", "inhibitory", "modulatory"].includes(e.kind));
  assert.ok(facts.length >= 5);
  for (const f of facts) assert.ok(f.claim && f.value && safeUrl(f.source?.url), f.claim);
  // ogni fonte citata (analogie, circuito, numeri) è nell'elenco delle fonti, con un link
  const listed = new Map(sources.map((s) => [s.id, s]));
  for (const s of [...brainSpec.analogies.flatMap((a) => a.sources), ...circuit.sources, ...facts.map((f) => f.source)]) {
    assert.ok(listed.has(s.id), `fonte citata ma non elencata: ${s.id}`);
    assert.ok(safeUrl(s.url), s.id);
  }
  assert.equal(listed.size, sources.length, "nessun doppione");
  const header = await source("../js/tabs/modern-llm/brain.js");
  assert.match(header, /FILE GENERATO/);
  assert.match(header, /brain_data\.json/);
  assert.match(header, /gen_brain\.mjs/);
});

test("cervello: i testi sono quelli della ricerca verificata, parola per parola", { skip: !(RESEARCH && existsSync(RESEARCH)) && "BRAIN_RESEARCH non indicato (la ricerca sta fuori dal repository)" }, async () => {
  const data = JSON.parse(await readFile(RESEARCH, "utf8"));
  const r = data.tabs["modern-llm"];
  assert.equal(brainSpec.intro, r.intro);
  assert.deepEqual(brainSpec.regions, r.regions);
  assert.deepEqual(brainSpec.pathways, r.pathways);
  assert.deepEqual(brainSpec.markers, r.markers);
  assert.deepEqual(brainSpec.analogies.map((a) => a.id), r.analogies.map((a) => a.id));
  for (const [i, a] of brainSpec.analogies.entries()) {
    const x = r.analogies[i];
    for (const k of ["ai", "brain", "text", "similar", "different"]) assert.equal(a[k], x[k], `${a.id}.${k}`);
    assert.equal(a.aiSection, x.ai_section);
    assert.deepEqual(a.regions, x.regions);
    assert.deepEqual(a.pathways ?? [], x.pathways ?? []);
    assert.equal(a.strength, x.strength.startsWith("debole") ? "debole" : x.strength);
    assert.deepEqual(a.sources.map((s) => [s.id, s.title, s.url]), x.sources.map((s) => [s.id, s.title, s.url]));
  }
  const c = r.circuit;
  for (const k of ["title", "text", "caveat"]) assert.equal(brainSpec.circuit[k], c[k], `circuit.${k}`);
  const aiOf = new Map(c.analogy_map.map((m) => [m.group, m.ai_component]));
  assert.deepEqual(brainSpec.circuit.groups.map((g) => [g.id, g.label, g.region, g.count, g.ai]), c.groups.map((g) => [g.id, g.label, g.region, g.count_shown, aiOf.get(g.id)]));
  assert.deepEqual(brainSpec.circuit.edges.map((e) => [e.from, e.to, e.label ?? "", e.pathway ?? ""]), c.edges.map((e) => [e.from, e.to, e.label ?? "", e.pathway ?? ""]));
  // i numeri: claim e valore identici a quelli della ricerca (fatti globali o dati dei tipi di neurone)
  const texts = new Set([...data.facts.map((f) => f.value), ...Object.values(data.neuron_types).flatMap((t) => t.data.map((d) => d.text))]);
  for (const f of brainSpec.facts) assert.ok(texts.has(f.value), f.claim);
});

// ------------------------------------------------------------------------------------------ ricerca e installazione

test("ricerca: voci view:rete e view:cervello, trovabili con le parole dei meccanismi e delle regioni", () => {
  const index = searchIndex.map((e) => prepareEntry({ ...e, tab: "modern-llm" }));
  const top = (q, n = 1) => search(index, q).slice(0, n).map((r) => r.entry.id);
  assert.equal(top("rete neurale")[0], "rete");
  assert.ok(top("KV globale", 3).includes("rete"));
  assert.ok(top("attention causale", 3).includes("rete"));
  assert.ok(top("router esperti neuroni", 3).includes("rete"));
  assert.equal(top("cervello")[0], "cervello");
  assert.equal(top("dopamina striato")[0], "cervello");
  assert.equal(top("gangli della base")[0], "cervello");
  for (const id of ["rete", "cervello"]) {
    const entries = searchIndex.filter((e) => e.kind === "view" && e.id === id);
    assert.equal(entries.length, 1);
    assert.ok(entries[0].title.endsWith("(LLM moderni)"));
  }
});

test("installazione: rappresentazioni prima di Fonti, goto da reps.goto, convivenza con i sottotab", async () => {
  const index = await source("../js/tabs/modern-llm/index.js");
  const install = index.indexOf("installRepresentations(ui");
  assert.ok(install > index.indexOf('addPane({ id: "modelli"') && install < index.indexOf('addPane({ id: "fonti"'), "il pannello Cervello sta prima di Fonti, dopo i pannelli del tab");
  assert.match(index, /import\("\.\/network\.js"\)\.then\(\(m\) => m\.networkSpecFor\(tab\.full\?\.model \?\? null\)\),\s+cache: false,/);
  assert.match(index, /import\("\.\/brain\.js"\)\.then\(\(m\) => m\.brainSpec\)/);
  assert.match(index, /async function goto\(\{ kind, id \}\) \{[\s\S]{0,300}?const shown = reps\.goto\(\{ kind, id \}\);\s*if \(shown\) return shown;/);
  // scegliere un sottotab (anche quello attivo) chiude prima il livello; "Cervello" resta prima di "Fonti" nei due
  assert.match(index, /async function showSubtab\(id, \{ restore = true \} = \{\}\) \{[\s\S]{0,300}?closeRepresentation\(\);/);
  assert.match(index, /onReselect: \(\) => closeRepresentation\(\)/, "la scheda già attiva chiude il livello (callback di subtabs.js)");
  assert.match(index, /function placeBrainPane\(to\)/);
  const css = await source("../css/tabs/modern-llm.css");
  assert.match(css, /\.tab\.tab-root\[data-tab="modern-llm"\]\.rep-alt-active > \.mllm-subtabs \{ visibility: visible; \}/);
});
