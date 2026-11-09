// Test delle rappresentazioni del tab RAG (logica pura): la rete neurale costruita dai calcoli veri del
// tab (js/tabs/rag/network.js) e la specifica del cervello (js/tabs/rag/brain.js).
// node --test webapp/tests/rag-reps.test.mjs
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { atlasKind } from "../js/core/brain/atlas.js";
import { normalizeBrainSpec } from "../js/core/brain/spec.js";
import { buildNetworkModel } from "../js/core/neuralNet/model.js";
import { LINK_LIMITS } from "../js/core/neuralNet/links.js";
import { LIMITS, normalizeSpec } from "../js/core/neuralNet/spec.js";
import { EXAMPLE_QUESTIONS, SHARED_DIMENSION, UNKNOWN_DIMENSION, cosineSimilarity } from "../js/shared/rag/index.js";
import { brainSpec } from "../js/tabs/rag/brain.js";
import { ENCODER, GENERATOR, SECTION_IDS, SHOWN, buildRagNetworkSpec, chunkLabel, encoderTokens, featureLabel, hnswNote, questionDims, shownFeatures, textTokens } from "../js/tabs/rag/network.js";
import { buildKnowledgeBase, runSearch } from "../js/tabs/rag/pipeline.js";
import { searchIndex } from "../js/tabs/rag/search.js";

// La ricerca verificata sta fuori dal repository: per il confronto parola per parola si indica il file con
// BRAIN_RESEARCH=/percorso/brain_data.json (senza, quel test si salta).
const RESEARCH = process.env.BRAIN_RESEARCH ?? "";

const kb = buildKnowledgeBase(); // configurazione iniziale del tab: splitter e HNSW didattici
const DEFAULT_QUESTION = EXAMPLE_QUESTIONS[2].question; // la domanda con cui il tab si apre
const LONG_QUESTION = "Vorrei sapere quali sono tutte le regole per i rimborsi delle spese di viaggio, gli alberghi, i pasti, i taxi, i treni e gli aerei quando vado in trasferta all'estero per un cliente importante dell'azienda nel periodo estivo";
const QUESTIONS = [...new Set([DEFAULT_QUESTION, ...EXAMPLE_QUESTIONS.map((q) => q.question), "come?", LONG_QUESTION])];

/** Come nel tab: la ricerca con i parametri del pannello Azioni (topK 5, soglia 0,5). */
const build = (question, options = {}, base = kb) => {
  const run = runSearch(base, question, { topK: 5, similarityThreshold: 0.5, ...options });
  return { run, spec: buildRagNetworkSpec({ kb: base, run }) };
};
const sectionOf = (spec, id) => spec.sections.find((s) => s.id === id);
const linksFrom = (spec, from, to) => spec.links.filter((l) => l.from === from && l.to === to);
const close = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

test("la rete del RAG passa la validazione del nucleo senza avvisi, per ogni domanda d'esempio e nei casi limite", () => {
  for (const question of QUESTIONS) {
    const { spec } = build(question);
    const model = buildNetworkModel(spec);
    assert.deepEqual(model.net.warnings, [], `avvisi per ${JSON.stringify(question)}`);
    // il badge e "simulato" valgono solo per le parti illustrative: le parti 1 e 2 hanno valori veri
    for (const id of [...SECTION_IDS.embedder, ...SECTION_IDS.database]) assert.ok(!model.values.simulated.has(id), `${id} dovrebbe avere valori veri`);
    for (const id of [...SECTION_IDS.encoder, ...SECTION_IDS.generator]) assert.ok(model.values.simulated.has(id), `${id} è illustrativa`);
  }
});

test("ordine e conteggi delle sezioni: quattro parti da sinistra a destra, con i blocchi ripetuti compressi", () => {
  const { spec } = build(DEFAULT_QUESTION);
  const net = normalizeSpec(spec);
  const expected = [...SECTION_IDS.embedder, ...SECTION_IDS.database, ...SECTION_IDS.encoder, ...SECTION_IDS.generator];
  assert.deepEqual(net.sections.map((s) => s.id), expected);
  assert.equal(net.sections.length, 22);
  // × 12 layer dell'encoder e × 32 del decoder, ognuno subito dopo il suo blocco
  assert.deepEqual(net.repeats.map((r) => [r.id, r.count, r.of.map((s) => s.id)]), [
    ["enc.x12", 12, ["enc.attn", "enc.add1", "enc.ffn", "enc.add2"]],
    ["llm.x32", 32, ["llm.attn", "llm.add1", "llm.ffn", "llm.add2"]],
  ]);
  const order = net.items.map((i) => i.id);
  assert.equal(order.indexOf("enc.x12"), order.indexOf("enc.add2") + 1);
  assert.equal(order.indexOf("llm.x32"), order.indexOf("llm.add2") + 1);
  // le quattro parti, ognuna seguita dalla parentesi grande (vero / illustrativo) che la contiene: a parità di
  // importanza il nucleo piazza le etichette nell'ordine dell'elenco, così le piccole stanno sulle loro parentesi
  assert.deepEqual(net.groups.map((g) => [g.from.id, g.to.id]), [
    ["emb.features", "emb.q"],
    ["db.chunks", "db.topk"],
    ["emb.features", "db.topk"],
    ["enc.tokens", "enc.vec"],
    ["llm.prompt", "llm.next"],
    ["enc.tokens", "llm.next"],
  ]);
  assert.match(net.groups[2].label, /^Valori veri/);
  assert.match(net.groups[5].label, /^Illustrativo/);
  // etichette corte: di lato, a rete intera, le parentesi delle parti vere sono larghe poche decine di pixel
  for (const s of net.sections) assert.ok(s.label.length <= 24, `${s.id}: etichetta lunga «${s.label}»`);
  for (const g of net.groups) assert.ok(g.label.length <= 30, `gruppo: etichetta lunga «${g.label}»`);
  for (const s of net.sections) assert.ok(s.text.length > 40, `${s.id}: manca il testo della scheda`);
  assert.ok(net.sources.length >= 10);
});

test("forme uguali ai numeri dichiarati: chunk, feature, dimensioni, token e modelli veri", () => {
  const { run, spec } = build(DEFAULT_QUESTION);
  const net = normalizeSpec(spec);
  const N = kb.inserts.length;
  const by = (id) => net.byId.get(id);
  // profondità della rete = i chunk del database, con i loro nomi
  assert.equal(net.slices.count, N);
  assert.deepEqual(net.slices.names, kb.inserts.map((r) => chunkLabel(r.entry.id)));
  for (const id of SECTION_IDS.database) assert.equal(by(id).depth.count, N, id);
  // parte 1: niente profondità (sacchetto di feature), una colonna per feature e per dimensione non nulla
  const { dims } = questionDims(run.explanation.vector);
  const F = run.explanation.features.length;
  assert.equal(by("emb.features").neurons, F);
  assert.equal(by("emb.features").depth.count, 1);
  for (const id of ["emb.raw", "emb.q", "db.chunks", "db.compare"]) assert.equal(by(id).neurons, dims.length, id);
  assert.equal(by("emb.q").real, 256);
  assert.match(by("db.chunks").sub, new RegExp(`${N} chunk × ${dims.length} delle 256`));
  // parti 3 e 4: i numeri dei config ufficiali in `real`, le dimensioni disegnate in SHOWN
  assert.deepEqual([ENCODER.layers, ENCODER.width, ENCODER.heads, ENCODER.ffn, ENCODER.vocab], [12, 384, 12, 1536, 250037]);
  assert.equal(ENCODER.vocab * ENCODER.width, 96_014_208); // parametri dell'embedding nei metadati MTEB
  assert.deepEqual([GENERATOR.layers, GENERATOR.width, GENERATOR.heads, GENERATOR.kvHeads, GENERATOR.ffn, GENERATOR.vocab], [32, 4096, 32, 8, 14336, 128256]);
  // SwiGLU di Llama: 2/3 di 4·d, × 1,3, arrotondato al multiplo di 1024 (sku_list.py)
  assert.equal(Math.ceil(Math.floor(1.3 * Math.floor((2 * 4 * 4096) / 3)) / 1024) * 1024, GENERATOR.ffn);
  assert.deepEqual([by("enc.tokens").real, by("enc.embed").real, by("enc.ffn").real, by("enc.vec").real], [250037, 384, 1536, 384]);
  assert.deepEqual([by("llm.prompt").real, by("llm.embed").real, by("llm.ffn").real, by("llm.next").real], [128256, 4096, 14336, 128256]);
  assert.equal(by("enc.attn").heads, SHOWN.encoder.heads);
  assert.equal(by("llm.attn").heads, SHOWN.generator.heads);
  assert.equal(by("enc.attn").neurons % by("enc.attn").heads, 0);
  // token: <s> "query" ":" … </s> in profondità nell'encoder, il prompt aumentato nel decoder
  const tokens = by("enc.tokens").depth;
  assert.equal(tokens.names[0], "<s>");
  assert.equal(tokens.names.at(-1), "</s>");
  // i token stimati dal tab (tokenizer.js spezza le parole lunghe: "query" → "que" + "ry")
  assert.deepEqual(tokens.names.slice(1, -1), textTokens(`query: ${DEFAULT_QUESTION}`));
  assert.ok(tokens.count <= SHOWN.encoder.tokens);
  const prompt = by("llm.prompt").depth;
  assert.equal(prompt.count, SHOWN.generator.tokens);
  // la domanda intera, poi l'inizio di ogni chunk scelto (almeno due token a testa)
  const question = textTokens(DEFAULT_QUESTION);
  assert.deepEqual(prompt.names.slice(0, question.length), question);
  const rest = prompt.names.slice(question.length);
  const each = Math.floor(rest.length / run.results.length);
  assert.ok(each >= 2);
  run.results.forEach((r, i) => assert.deepEqual(rest.slice(i * each, (i + 1) * each), textTokens(r.text).slice(0, each)));
  assert.match(by("llm.prompt").sub, new RegExp(`dei ${run.prompt.tokens.total.toLocaleString("it-IT", { useGrouping: "always" })} token`));
  // le sezioni della stessa parte condividono la stessa profondità
  for (const id of SECTION_IDS.encoder.slice(0, 6)) assert.equal(by(id).depth.count, tokens.count, id);
  for (const id of SECTION_IDS.generator.slice(0, 6)) assert.equal(by(id).depth.count, prompt.count, id);
  // le parti illustrative restano basse (al più i neuroni nascosti disegnati): la parte vera della domanda
  // iniziale, con le sue 27 feature e 24 dimensioni, è la più alta
  const illustrative = [...SECTION_IDS.encoder, ...SECTION_IDS.generator].map((id) => by(id).neurons);
  assert.ok(Math.max(...illustrative) <= Math.max(SHOWN.encoder.ffn, SHOWN.generator.ffn));
  assert.ok(by("emb.features").neurons > Math.max(...illustrative) && by("emb.q").neurons >= Math.max(...illustrative));
  for (const id of ["enc.pool", "enc.vec", "llm.next", "emb.raw", "emb.q"]) assert.equal(by(id).depth.count, 1, id);
});

test("schemi giusti: hashing, confronto in profondità, top-k, attention bidirezionale e causale, pooling", () => {
  const { run, spec } = build(DEFAULT_QUESTION);
  const net = normalizeSpec(spec);
  const link = (from, to) => net.links.filter((l) => l.from.id === from && l.to.id === to);
  // hashing: un collegamento per feature, verso la dimensione scelta dall'hash, peso ±1 (le sconosciute in 1, +1)
  const [hash] = link("emb.features", "emb.raw");
  assert.equal(hash.pattern, "explicit");
  const features = shownFeatures(run.explanation.features);
  const { dims } = questionDims(run.explanation.vector);
  assert.equal(hash.pairs.count, features.length);
  for (let i = 0; i < hash.pairs.count; i++) {
    const f = features[hash.pairs.from[i]];
    assert.equal(dims[hash.pairs.to[i]], f.known ? kb.embedder.slot(f.feature).dimension : UNKNOWN_DIMENSION);
    assert.equal(hash.pairs.weight[i], f.known ? f.sign : 1);
  }
  assert.equal(link("emb.raw", "emb.q")[0].pattern, "identity");
  // la domanda raggiunge ogni chunk (broadcast), i vettori salvati arrivano uno a uno, la somma è sulla fetta
  assert.equal(link("emb.q", "db.compare")[0].pattern, "broadcast");
  assert.equal(link("db.chunks", "db.compare")[0].pattern, "identity");
  const [sum] = link("db.compare", "db.cosine");
  assert.equal(sum.pattern, "explicit");
  assert.equal(sum.slice, "selected");
  assert.deepEqual([...sum.pairs.weight], dims.map(() => 1));
  // top-k: un collegamento per chunk scelto da HNSW, alla sua posizione, e nessun altro
  const topk = link("db.cosine", "db.topk");
  assert.equal(topk.length, run.results.length);
  assert.deepEqual(topk.map((l) => l.slice), run.results.map((r) => kb.inserts.findIndex((x) => x.entry.id === r.id)));
  // encoder: attention SENZA maschera, residui, pooling che schiaccia i token; decoder: attention causale
  const encAttn = net.links.find((l) => l.within && l.from.id === "enc.attn");
  assert.equal(encAttn.pattern, "attention");
  assert.equal(encAttn.causal, false);
  const llmAttn = net.links.find((l) => l.within && l.from.id === "llm.attn");
  assert.equal(llmAttn.causal, true);
  assert.equal(link("enc.add2", "enc.pool")[0].pattern, "pool");
  assert.equal(link("enc.tokens", "enc.embed")[0].pattern, "lookup");
  assert.equal(link("llm.prompt", "llm.embed")[0].pattern, "lookup");
  assert.equal(link("enc.embed", "enc.add1")[0].pattern, "residual");
  assert.equal(link("llm.add1", "llm.add2")[0].pattern, "residual");
  // solo l'ultima posizione del prompt predice il prossimo token
  const [head] = link("llm.add2", "llm.next");
  assert.equal(head.slice, net.byId.get("llm.add2").depth.count - 1);
  // nessuna cross-attention: le parti si toccano solo dove passa un dato vero (embedder → database)
  assert.ok(!net.links.some((l) => l.pattern === "attention" && !l.within));
  const parts = (id) => id.split(".")[0];
  for (const l of net.links) {
    if (parts(l.from.id) !== parts(l.to.id)) assert.deepEqual([l.from.id, l.to.id], ["emb.q", "db.compare"]);
  }
});

test("valori veri: uguali a quelli calcolati dalla pipeline del tab (embedder, vector store, ricerca)", () => {
  for (const question of [DEFAULT_QUESTION, EXAMPLE_QUESTIONS[0].question, EXAMPLE_QUESTIONS[12].question, LONG_QUESTION]) {
    const { run, spec } = build(question);
    const ex = kb.embedder.explain(question);
    const N = kb.inserts.length;
    const { dims, total } = questionDims(ex.vector);
    const features = shownFeatures(ex.features);
    assert.deepEqual(Array.from(sectionOf(spec, "emb.features").values), features.map((f) => f.weight));
    assert.deepEqual(sectionOf(spec, "emb.features").neuronNames, features.map(featureLabel));
    assert.deepEqual(Array.from(sectionOf(spec, "emb.raw").values), dims.map((d) => ex.raw[d]));
    assert.deepEqual(Array.from(sectionOf(spec, "emb.q").values), dims.map((d) => ex.vector[d]));
    // la dimensione 0 è la componente comune (0,66), la 1 raccoglie le feature sconosciute
    if (dims.includes(SHARED_DIMENSION)) assert.ok(close(ex.vector[SHARED_DIMENSION], kb.embedder.sharedComponent));
    const chunks = sectionOf(spec, "db.chunks").values;
    const products = sectionOf(spec, "db.compare").values;
    const cosine = sectionOf(spec, "db.cosine").values;
    const topk = sectionOf(spec, "db.topk").values;
    const exact = new Map(run.exactAll.map((e) => [e.id, e.score]));
    const results = new Map(run.results.map((r) => [r.id, r.score]));
    kb.inserts.forEach((r, n) => {
      let sum = 0;
      dims.forEach((d, k) => {
        assert.equal(chunks[k * N + n], r.vector[d]);
        assert.ok(close(products[k * N + n], ex.vector[d] * r.vector[d]));
        sum += products[k * N + n];
      });
      // il coseno è quello della ricerca esatta del tab (score = 1 − distanza)…
      assert.ok(close(cosine[n], exact.get(r.entry.id), 1e-6), `${r.entry.id}: coseno`);
      assert.ok(close(cosine[n], cosineSimilarity(ex.vector, r.vector), 1e-6));
      // …e, con tutte le dimensioni non nulle disegnate, è proprio la somma dei prodotti mostrati
      if (dims.length === total) assert.ok(close(sum, cosine[n], 1e-5), `${r.entry.id}: somma ${sum} contro ${cosine[n]}`);
      // passano solo i risultati di similaritySearch (HNSW + topK + soglia), con il loro score
      assert.ok(close(topk[n], results.get(r.entry.id) ?? 0, 1e-6), `${r.entry.id}: top-k`);
    });
  }
});

test("il testo su HNSW dice quanto ha lavorato davvero l'indice del tab, e perché non conviene", () => {
  const { run, spec } = build(DEFAULT_QUESTION);
  const text = sectionOf(spec, "db.cosine").text;
  assert.ok(text.includes(hnswNote(run, kb.inserts.length, kb.store.index.efSearch)));
  assert.ok(text.includes(`${run.comparisons} distanze`));
  // grafo didattico a più livelli: con 27 chunk rilegge alcuni nodi (più distanze dei nodi visitati)
  const distinct = run.frames.at(-1).visited.length;
  assert.ok(run.comparisons > kb.inserts.length && run.comparisons > distinct, `${run.comparisons} distanze, ${distinct} nodi diversi`);
  assert.match(text, /non conviene \(rilegge alcuni nodi a ogni livello del grafo\)/);
  // valori di pgvector: un livello solo ed ef_search 40 > 21 chunk → legge tutto il grafo, ogni nodo una volta
  // (prima il testo diceva anche qui "rilegge alcuni nodi a ogni livello", che non è vero)
  const pg = buildKnowledgeBase({ splitter: "l17", hnsw: "pgvector" });
  const pgRun = runSearch(pg, DEFAULT_QUESTION, { topK: 5, similarityThreshold: 0.5 });
  assert.equal(pg.maxLevel, 0);
  assert.equal(pgRun.comparisons, pg.inserts.length);
  const pgText = sectionOf(buildRagNetworkSpec({ kb: pg, run: pgRun }), "db.cosine").text;
  assert.match(pgText, /con ef_search 40 e solo 21 chunk legge tutto il grafo/);
  assert.doesNotMatch(pgText, /rilegge/);
  assert.match(hnswNote({ comparisons: 120, recall: 0.8 }, 10_000), /120 distanze invece di 10\.000 \(recall 80%\)/);
});

test("normalizzazione descritta com'è nell'embedder: parte specifica a √(1 − s²), dimensione 0 = s, lunghezza 1", () => {
  const { run, spec } = build(DEFAULT_QUESTION);
  const s = kb.embedder.sharedComponent;
  const q = sectionOf(spec, "emb.q");
  const v = run.explanation.vector;
  // i numeri del testo sono quelli del vettore vero
  let specific = 0;
  for (let d = 0; d < v.length; d++) if (d !== SHARED_DIMENSION) specific += v[d] * v[d];
  assert.ok(close(Math.sqrt(specific), Math.sqrt(1 - s * s), 1e-6));
  assert.ok(close(v[SHARED_DIMENSION], s, 1e-7));
  const rest = Math.sqrt(1 - s * s).toLocaleString("it-IT", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
  const shared = s.toLocaleString("it-IT", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
  assert.ok(q.text.includes(`fino a lunghezza ${rest}`) && q.text.includes(`prende il valore ${shared}`), q.text);
  assert.doesNotMatch(q.text, /fino a lunghezza 1[,;:. ]/, "il vettore grezzo non va a lunghezza 1: ci va il vettore intero, con la dimensione 0");
  const [scaleLink] = linksFrom(spec, "emb.raw", "emb.q");
  assert.match(scaleLink.label, new RegExp(`riscalato a lunghezza ${rest}, più ${shared} nella dimensione 0`));
  // le feature conosciute vanno nelle dimensioni dalla 2 alla 255 (0 e 1 riservate)
  assert.match(sectionOf(spec, "emb.raw").text, /dalla 2 alla 255: la 0 e la 1 sono riservate/);
  for (const f of run.explanation.features) if (f.known) assert.ok(f.dimension >= 2 && f.dimension <= 255);
  // vettore nullo: nemmeno la dimensione 0 si accende, e il coseno non indica un "più simile"
  const empty = build("come?");
  assert.ok(empty.run.explanation.vector.every((x) => x === 0));
  assert.match(sectionOf(empty.spec, "emb.q").text, /resta tutto zero, anche la componente comune/);
  assert.match(sectionOf(empty.spec, "db.cosine").text, /tutte le similarità valgono 0/);
  assert.doesNotMatch(sectionOf(empty.spec, "db.cosine").text, /La più alta/);
});

test("l'introduzione dice da dove viene la domanda disegnata (pannello, ultima ricerca, esempio)", () => {
  const run = runSearch(kb, DEFAULT_QUESTION, { topK: 5, similarityThreshold: 0.5 });
  const intro = (origin) => buildRagNetworkSpec({ kb, run, origin }).intro;
  assert.match(intro(undefined), /quella del pannello Azioni/);
  assert.match(intro("panel"), /quella del pannello Azioni/);
  assert.match(intro("last"), /l'ultima cercata: il campo del pannello Azioni è vuoto/);
  assert.match(intro("example"), /una domanda d'esempio: il campo del pannello Azioni è vuoto/);
  for (const o of ["panel", "last", "example"]) assert.ok(intro(o).includes(`«${DEFAULT_QUESTION}»`));
  // index.js passa la provenienza: networkSearch → { run, origin }
  return readFile(new URL("../js/tabs/rag/index.js", import.meta.url), "utf8").then((source) => {
    assert.match(source, /builder\(\{ kb: tab\.kb, \.\.\.networkSearch\(\) \}\)/);
    for (const o of ["panel", "last", "example"]) assert.ok(source.includes(`origin: "${o}"`), o);
  });
});

test("il vettore grezzo è davvero la somma delle feature con il segno dell'hash", () => {
  const ex = kb.embedder.explain(DEFAULT_QUESTION);
  const sums = new Map();
  let unknown = 0;
  for (const f of ex.features) {
    if (f.known) sums.set(f.dimension, (sums.get(f.dimension) ?? 0) + f.sign * f.weight);
    else unknown += f.weight * f.weight;
  }
  for (const [d, v] of sums) assert.ok(close(ex.raw[d], v, 1e-5), `dim ${d}`);
  assert.ok(close(ex.raw[UNKNOWN_DIMENSION], Math.sqrt(unknown), 1e-5)); // le sconosciute «in quadratura»
  assert.ok(unknown > 0, "nella domanda iniziale «cena» non compare nei documenti");
});

test("domanda fuori tema, solo parole vuote, domanda lunga: la rete resta onesta", () => {
  // fuori tema: nessun chunk sopra la soglia, nessun collegamento al top-k, il prompt dice che non ci sono documenti
  const off = build(EXAMPLE_QUESTIONS[12].question);
  assert.equal(off.run.results.length, 0);
  assert.equal(linksFrom(off.spec, "db.cosine", "db.topk").length, 0);
  assert.ok(sectionOf(off.spec, "db.topk").values.every((v) => v === 0));
  assert.match(sectionOf(off.spec, "db.topk").text, /Nessun chunk supera la soglia/);
  assert.ok(sectionOf(off.spec, "llm.prompt").depth.names.join("").includes("Nessundocumentopertinente"));
  // solo parole vuote: vettore nullo, si disegna la dimensione 0 a zero e una feature segnaposto
  const empty = build("come?");
  assert.deepEqual(sectionOf(empty.spec, "emb.q").neuronNames, ["dim 0 · comune"]);
  assert.deepEqual(Array.from(sectionOf(empty.spec, "emb.q").values), [0]);
  assert.equal(sectionOf(empty.spec, "emb.features").neurons, 1);
  assert.equal(buildNetworkModel(empty.spec).net.warnings.length, 0);
  // domanda lunga: al più SHOWN feature e dimensioni, con la scala dichiarata
  const long = build(LONG_QUESTION);
  const ex = kb.embedder.explain(LONG_QUESTION);
  assert.ok(ex.features.length > SHOWN.features);
  assert.equal(sectionOf(long.spec, "emb.features").neurons, SHOWN.features);
  assert.equal(sectionOf(long.spec, "emb.features").real, ex.features.length);
  assert.match(long.spec.scale, new RegExp(`le ${SHOWN.features} feature più pesanti delle ${ex.features.length}`));
  assert.ok(sectionOf(long.spec, "emb.q").neurons <= SHOWN.dims);
  assert.equal(sectionOf(long.spec, "enc.tokens").depth.count, SHOWN.encoder.tokens);
  assert.equal(encoderTokens(LONG_QUESTION).names.at(-2), "…");
});

test("limiti del nucleo rispettati con margine; tutti i collegamenti disegnati già alla densità iniziale", () => {
  for (const question of QUESTIONS) {
    const model = buildNetworkModel(build(question).spec);
    assert.ok(model.net.units <= LIMITS.units / 10, `${question}: ${model.net.units} neuroni`);
    assert.ok(model.links.segments.count <= 16_000, `${question}: ${model.links.segments.count} segmenti`); // DEFAULT_SEGMENTS di scene.js
    assert.ok(model.links.arcs.count <= LINK_LIMITS.maxArcs / 5, `${question}: ${model.links.arcs.count} archi`);
    assert.equal(model.links.capped, false);
    assert.equal(model.links.scale, 1);
  }
});

test("la configurazione L17 (chunkSize 500) cambia i chunk, non la forma della rete", () => {
  const l17 = buildKnowledgeBase({ splitter: "l17", hnsw: "pgvector" });
  const { spec } = build(DEFAULT_QUESTION, {}, l17);
  const net = normalizeSpec(spec);
  assert.equal(net.slices.count, l17.inserts.length);
  assert.notEqual(l17.inserts.length, kb.inserts.length);
  assert.match(spec.scale, /Lezione L17/);
  assert.deepEqual(buildNetworkModel(spec).net.warnings, []);
});

test("deterministica: stessa domanda, stessa specifica (anche dopo una ricerca diversa)", () => {
  const a = JSON.stringify(build(DEFAULT_QUESTION).spec, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v));
  build(EXAMPLE_QUESTIONS[5].question);
  const b = JSON.stringify(build(DEFAULT_QUESTION).spec, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v));
  assert.equal(a, b);
});

test("testi onesti: domanda citata come testo, scala e simulazioni dichiarate, nessun HTML", () => {
  const tricky = "Il limite è <b>500</b> & \"oltre\"?";
  const { spec } = build(tricky);
  assert.ok(spec.intro.includes(`«${tricky}»`), "la domanda entra nell'introduzione come testo");
  const texts = [spec.title, spec.intro, spec.scale, spec.note, ...spec.sections.flatMap((s) => [s.label, s.sub ?? "", s.text ?? ""]), ...spec.groups.flatMap((g) => [g.label, g.text ?? ""])];
  const withoutQuestion = texts.map((t) => t.replace(tricky, ""));
  for (const t of withoutQuestion) assert.ok(!/<\/?[a-z][^>]*>/i.test(t.replace(/<s>|<\/s>|<documento>|<=>/g, "")), `HTML nel testo: ${t.slice(0, 60)}`);
  assert.match(spec.note, /simulate con un seme nelle parti 3 e 4/);
  assert.match(spec.scale, /Parte 3 \(illustrativa\)/);
  assert.match(sectionOf(spec, "enc.tokens").text, /non è pubblica/);
  for (const s of spec.sources) assert.match(s.url, /^https:\/\//);
});

// ------------------------------------------------------------------------------------------ cervello
test("brainSpec: normalizeBrainSpec strict senza avvisi, con le analogie della ricerca complete", () => {
  const spec = normalizeBrainSpec(brainSpec, { strict: true });
  assert.deepEqual(spec.warnings, []);
  assert.equal(spec.analogies.length, 10);
  for (const a of spec.analogies) {
    assert.ok(a.ai && a.brain && a.text && a.similar && a.different, `${a.id}: campi`);
    assert.ok(["forte", "media", "debole"].includes(a.strength), `${a.id}: forza`);
    assert.ok(a.sources.length > 0 && a.sources.every((s) => s.url?.startsWith("https://")), `${a.id}: fonti`);
  }
  assert.equal(spec.circuit.groups.length, 8);
  assert.equal(spec.circuit.edges.length, 14);
  assert.deepEqual(spec.circuit.groups.map((g) => g.id), ["ctx", "ec", "dg", "int", "ca3", "ca1", "sub", "ctx_out"]);
  assert.ok(spec.circuit.groups.every((g) => g.ai), "ogni gruppo dice a cosa corrisponde nel RAG");
  assert.ok(spec.circuit.caveat.length > 40);
  assert.equal(spec.facts.length, 5);
});

test("brainSpec: regioni, percorsi e marcatori nell'atlante; ogni fonte citata è nell'elenco delle fonti", () => {
  for (const id of brainSpec.regions) assert.equal(atlasKind(id), "region", id);
  for (const id of brainSpec.markers) assert.equal(atlasKind(id), "marker", id);
  for (const p of brainSpec.pathways) assert.equal(atlasKind(typeof p === "string" ? p : p.id), "pathway");
  for (const a of brainSpec.analogies) {
    for (const r of a.regions) assert.ok(["region", "marker"].includes(atlasKind(r)), `${a.id}: ${r}`);
    for (const p of a.pathways ?? []) assert.equal(atlasKind(p), "pathway", `${a.id}: ${p}`);
  }
  for (const g of brainSpec.circuit.groups) assert.equal(atlasKind(g.region), "region", g.id);
  for (const e of brainSpec.circuit.edges) if (e.pathway) assert.equal(atlasKind(e.pathway), "pathway");
  const ids = new Set(brainSpec.sources.map((s) => s.id));
  assert.equal(ids.size, brainSpec.sources.length, "fonti senza doppioni");
  const cited = [
    ...brainSpec.analogies.flatMap((a) => a.sources.map((s) => s.id)),
    ...brainSpec.circuit.sources.map((s) => s.id),
    ...brainSpec.circuit.groups.map((g) => g.realSource).filter(Boolean),
    ...brainSpec.facts.map((f) => f.source.id),
  ];
  for (const id of cited) assert.ok(ids.has(id), `fonte citata ma non elencata: ${id}`);
  for (const s of brainSpec.sources) assert.ok(s.title && s.authors && s.year && s.url?.startsWith("https://"), s.id);
  // le cautele della verifica della ricerca restano scritte accanto ai numeri incerti: Rolls 2016 (via
  // perforante) ed Eyal et al. 2018 (spine), entrambe "numero da ricontrollare online" in brain_data.json
  assert.match(brainSpec.facts.find((f) => f.source.id === "rolls2016").note, /ricontrollare/);
  assert.match(brainSpec.facts.find((f) => f.source.id === "eyal2018").note, /ricontrollare/);
});

test("brainSpec: i testi sono quelli della ricerca verificata, parola per parola", { skip: !(RESEARCH && existsSync(RESEARCH)) && "BRAIN_RESEARCH non indicato (la ricerca sta fuori dal repository)" }, async () => {
  const data = JSON.parse(await readFile(RESEARCH, "utf8"));
  const r = data.tabs.rag;
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
  const ai = new Map(c.analogy_map.map((m) => [m.group, m.ai_component]));
  assert.deepEqual(
    brainSpec.circuit.groups.map((g) => [g.id, g.label, g.region, g.type, g.count, g.real, g.realText, g.ai, g.note]),
    c.groups.map((g) => [g.id, g.label, g.region, g.neuron_type, g.count_shown, g.real_count, g.real_count_text, ai.get(g.id), g.note]),
  );
  const KIND = { eccitatoria: "excitatory", inibitoria: "inhibitory", modulatoria: "modulatory" };
  assert.deepEqual(
    brainSpec.circuit.edges.map((e) => [e.from, e.to, e.kind, e.label, e.pathway, e.note]),
    c.edges.map((e) => [e.from, e.to, KIND[e.kind], e.label, e.pathway, e.note]),
  );
  // i numeri sono quelli della lista della ricerca (claim, valore e fonte)
  for (const f of brainSpec.facts) {
    const x = data.facts.find((y) => y.claim === f.claim);
    assert.ok(x, f.claim);
    assert.equal(f.value, x.value);
    assert.equal(f.source.id, x.source.id);
  }
});

// ------------------------------------------------------------------------------------------ ricerca
test("link: gli id delle voci di ricerca sono quelli di REP_LINKS del nucleo", async () => {
  // letto dal sorgente: representations.js importa moduli con three.js, che qui non c'è
  const core = await readFile(new URL("../js/core/representations.js", import.meta.url), "utf8");
  assert.match(core, /REP_LINKS = \{ rete: "network", cervello: "brain" \}/);
  const tab = await readFile(new URL("../js/tabs/rag/index.js", import.meta.url), "utf8");
  assert.match(tab, /const shown = reps\.goto\(\{ kind, id \}\);\s+if \(shown\) return shown;/, "goto comincia con reps.goto");
  assert.ok(tab.indexOf("installRepresentations(ui") < tab.indexOf('addPane({ id: "fonti"'), "il pannello Cervello va prima di Fonti");
});

test("search.js: voci per la Rete neurale e il Cervello con parole chiave utili", () => {
  const rete = searchIndex.find((e) => e.kind === "view" && e.id === "rete");
  const cervello = searchIndex.find((e) => e.kind === "view" && e.id === "cervello");
  // titoli senza il nome del tab (la palette raggruppa già per tab: "Database vettoriale (RAG)") e senza
  // parentesi annidate; il titolo della rete è quello della sua scheda
  assert.equal(rete.title, "Rete neurale del RAG: dalla domanda alla risposta");
  assert.equal(cervello.title, "Cervello: il RAG e la memoria");
  for (const k of ["feature hashing", "top-k", "attention bidirezionale", "mean pooling", "attention causale"]) assert.ok(rete.keywords.includes(k), k);
  for (const k of ["ippocampo", "corteccia entorinale", "giro dentato", "CA3"]) assert.ok(cervello.keywords.includes(k), k);
  // i livelli dello zoom (goto di sempre) ci sono ancora
  for (const id of ["macro", "blocks", "detail"]) assert.ok(searchIndex.some((e) => e.kind === "view" && e.id === id), id);
});
