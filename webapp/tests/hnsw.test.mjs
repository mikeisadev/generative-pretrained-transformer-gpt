// Test dell'indice HNSW (modulo puro, niente DOM): node --test webapp/tests/
import assert from "node:assert/strict";
import { test } from "node:test";
import { EXAMPLE_QUESTIONS, HNSWIndex, createDemoStore, createRng, euclideanDistance, gaussian, normalize } from "../js/shared/rag/index.js";

/** Punti gaussiani normalizzati (direzioni casuali): un caso difficile e senza struttura. */
function randomUnitVectors(count, dimensions, seed) {
  const rng = createRng(seed);
  return Array.from({ length: count }, () => normalize(Array.from({ length: dimensions }, () => gaussian(rng))));
}

function buildIndex(points, options) {
  const index = new HNSWIndex(options);
  points.forEach((p, i) => index.insert(p, { id: `p${i}` }));
  return index;
}

/** recall@k media: quota dei veri k vicini (ricerca esatta) trovati da HNSW. */
function recallAtK(index, queries, k) {
  let hits = 0;
  for (const q of queries) {
    const approx = new Set(index.search(q, k).map((r) => r.index));
    for (const r of index.exactSearch(q, k)) if (approx.has(r.index)) hits++;
  }
  return hits / (queries.length * k);
}

const points = randomUnitVectors(1000, 32, 2024);
const queries = randomUnitVectors(100, 32, 7);
const index = buildIndex(points, { seed: 1 });

test("HNSW: recall@5 >= 0.95 su 1000 vettori casuali (selezione euristica e semplice)", () => {
  assert.ok(recallAtK(index, queries, 5) >= 0.95);
  const simple = buildIndex(points, { seed: 1, neighborSelection: "simple" });
  assert.ok(recallAtK(simple, queries, 5) >= 0.95);
  // Anche con un'altra metrica (euclidea) e con le opzioni dell'algoritmo 4 attive.
  const l2 = buildIndex(points, { seed: 1, distance: euclideanDistance, extendCandidates: true, keepPrunedConnections: true });
  assert.ok(recallAtK(l2, queries, 5) >= 0.95);
});

test("HNSW: su un insieme più grande la ricerca visita una piccola parte dei nodi", () => {
  // Con pochi punti HNSW non conviene (ef × M0 è già vicino a N); il vantaggio cresce con N.
  const many = randomUnitVectors(4000, 8, 99);
  const big = buildIndex(many, { M: 8, efConstruction: 32, seed: 3 });
  const probes = randomUnitVectors(20, 8, 5);
  assert.ok(recallAtK(big, probes, 5) >= 0.95);
  for (const q of probes) {
    let visited = 0;
    big.search(q, 5, { ef: 20, onEvent: (e) => e.type === "search-end" && (visited = e.visitedCount) });
    assert.ok(visited > 0 && visited < many.length / 4, `visitati ${visited} nodi su ${many.length}`);
  }
});

test("HNSW: recall@5 >= 0.95 su tutti i chunk del corpus", () => {
  const { store, embedder } = createDemoStore();
  const vectors = store.entries.map((e) => e.vector);
  const asQueries = [...vectors, ...EXAMPLE_QUESTIONS.map((q) => embedder.embed(q.question))];
  assert.ok(recallAtK(store.index, asQueries, 5) >= 0.95);
  // Con M piccolo il grafo ha più piani ed è più "sparso": la recall deve restare alta.
  const { store: sparse } = createDemoStore({ hnsw: { M: 3, efConstruction: 16, efSearch: 10, seed: 5 } });
  assert.ok(sparse.index.maxLevel >= 1);
  assert.ok(recallAtK(sparse.index, asQueries, 5) >= 0.95);
});

test("HNSW: determinismo con lo stesso seed, struttura diversa con un altro seed", () => {
  const sample = points.slice(0, 300);
  const a = buildIndex(sample, { M: 6, seed: 11 });
  const b = buildIndex(sample, { M: 6, seed: 11 });
  const c = buildIndex(sample, { M: 6, seed: 12 });
  assert.deepEqual(a.toGraph(), b.toGraph());
  assert.deepEqual(a.search(queries[3], 5), b.search(queries[3], 5));
  assert.notDeepEqual(a.toGraph().nodes.map((n) => n.level), c.toGraph().nodes.map((n) => n.level));
});

test("HNSW: grado massimo rispettato (M ai piani alti, M0 = 2M al piano 0) e archi validi", () => {
  for (const options of [{ M: 3, seed: 2 }, { M: 5, seed: 4, neighborSelection: "simple" }, { M: 16, seed: 1 }]) {
    const idx = options.M === 16 ? index : buildIndex(points, options);
    assert.equal(idx.M0, 2 * idx.M);
    assert.ok(Math.abs(idx.mL - 1 / Math.log(idx.M)) < 1e-12);
    let pruned = false;
    for (const node of idx.nodes) {
      assert.equal(node.neighbors.length, node.level + 1);
      node.neighbors.forEach((list, layer) => {
        assert.ok(list.length <= (layer === 0 ? idx.M0 : idx.M), `nodo ${node.index} piano ${layer}: ${list.length}`);
        assert.equal(new Set(list).size, list.length, "nessun vicino duplicato");
        for (const n of list) {
          assert.notEqual(n, node.index, "nessun arco verso se stesso");
          assert.ok(idx.nodes[n].level >= layer, "il vicino esiste in quel piano");
        }
        if (list.length === (layer === 0 ? idx.M0 : idx.M)) pruned = true;
      });
    }
    assert.ok(pruned, "almeno un nodo ha raggiunto il massimo (la potatura è stata esercitata)");
    for (const level of idx.stats().levels) assert.ok(level.maxDegree <= level.maxAllowed);
  }
});

test("HNSW: i livelli seguono la distribuzione dell'articolo (circa 1/M dei nodi sopra il piano 0)", () => {
  const stats = index.stats();
  const aboveZero = stats.levels[1].nodes;
  assert.equal(stats.levels[0].nodes, points.length);
  assert.ok(aboveZero > 1000 / 16 / 2 && aboveZero < (1000 / 16) * 2, `nodi al piano 1: ${aboveZero}`);
  assert.equal(index.nodes[index.entryPoint].level, index.maxLevel, "il punto d'ingresso sta al piano più alto");
  for (let l = 1; l < stats.levels.length; l++) assert.ok(stats.levels[l].nodes <= stats.levels[l - 1].nodes);
});

test("HNSW: eventi di inserimento (livello, visite, archi creati e potati)", () => {
  const idx = new HNSWIndex({ M: 2, efConstruction: 8, seed: 9 });
  const events = [];
  points.slice(0, 60).forEach((p) => idx.insert(p, { onEvent: (e) => events.push(e) }));
  const starts = events.filter((e) => e.type === "insert-start");
  const ends = events.filter((e) => e.type === "insert-end");
  assert.equal(starts.length, 60);
  assert.equal(ends.length, 60);
  starts.forEach((e, i) => assert.equal(e.level, idx.nodes[i].level));
  const connect = events.filter((e) => e.type === "connect");
  assert.ok(connect.length >= 59);
  for (const e of connect) assert.ok(e.neighbors.length <= idx.M);
  const prune = events.filter((e) => e.type === "prune");
  assert.ok(prune.length > 0);
  for (const e of prune) {
    assert.ok(e.before.length > e.after.length);
    assert.deepEqual(e.removed, e.before.filter((n) => !e.after.includes(n)));
  }
  assert.ok(events.some((e) => e.type === "layer-step" && e.phase === "insert"));
  assert.ok(ends.some((e) => e.entryPointChanged));
});

test("HNSW: eventi di ricerca dal piano più alto al piano 0", () => {
  const events = [];
  const results = index.search(queries[1], 5, { onEvent: (e) => events.push(e) });
  assert.equal(events[0].type, "search-start");
  assert.equal(events.at(-1).type, "search-end");
  assert.deepEqual(events.at(-1).results, results);
  const layers = events.filter((e) => e.type === "layer-start").map((e) => e.layer);
  assert.deepEqual(layers, Array.from({ length: index.maxLevel + 1 }, (_, i) => index.maxLevel - i));
  const layerStarts = events.filter((e) => e.type === "layer-start");
  assert.ok(layerStarts.slice(0, -1).every((e) => e.ef === 1), "piani alti: discesa golosa con ef = 1");
  assert.equal(layerStarts.at(-1).ef, Math.max(index.efSearch, 5), "piano 0: ef = max(efSearch, k)");
  const steps = events.filter((e) => e.type === "layer-step");
  assert.ok(steps.length > 0);
  for (const step of steps) {
    assert.ok(Array.isArray(step.candidates) && Array.isArray(step.found));
    for (let i = 1; i < step.found.length; i++) assert.ok(step.found[i - 1].distance <= step.found[i].distance);
  }
  const lastLayer = events.filter((e) => e.type === "layer-end").at(-1);
  assert.deepEqual(lastLayer.found.slice(0, 5).map((f) => f.index), results.map((r) => r.index));
});

test("HNSW: struttura per il disegno e casi limite", () => {
  const graph = index.toGraph();
  assert.equal(graph.nodes.length, points.length);
  assert.equal(graph.layers.length, index.maxLevel + 1);
  for (const layer of graph.layers) {
    assert.equal(layer.edges.length, layer.mutual.length);
    const keys = new Set(layer.edges.map(([a, b]) => `${a}-${b}`));
    assert.equal(keys.size, layer.edges.length, "archi non ripetuti");
    for (const [a, b] of layer.edges) assert.ok(a < b);
    assert.equal(layer.nodes.length, index.stats().levels[layer.level].nodes);
  }
  const empty = new HNSWIndex();
  assert.deepEqual(empty.search([1, 0], 3), []);
  const forced = new HNSWIndex({ M: 4, seed: 1 });
  forced.insert([1, 0, 0]);
  forced.insert([0, 1, 0], { level: 3 });
  assert.equal(forced.entryPoint, 1);
  assert.equal(forced.maxLevel, 3);
  assert.throws(() => forced.insert([1, 0]), RangeError);
  assert.throws(() => forced.search([1, 0, 0], 0), RangeError);
  assert.throws(() => new HNSWIndex({ M: 1 }), RangeError);
});

test("HNSW: selezione dei vicini, semplice contro euristica (algoritmi 3 e 4)", () => {
  // Base nell'origine; A e B vicinissimi tra loro, C in un'altra direzione.
  const idx = new HNSWIndex({ M: 2, distance: euclideanDistance, seed: 1 });
  const [A, B, C] = [[1, 0], [1.1, 0.05], [0, 1.5]].map((v) => idx.insert(v));
  const base = [0, 0];
  const candidates = [A, B, C].map((index) => ({ index, distance: euclideanDistance(base, idx.nodes[index].vector) }));
  // La semplice prende i due più vicini, anche se ridondanti.
  assert.deepEqual(idx.selectNeighborsSimple(candidates, 2).map((c) => c.index), [A, B]);
  // L'euristica scarta B (più vicino ad A che alla base) e preferisce la direzione nuova C.
  assert.deepEqual(idx.selectNeighborsHeuristic(base, candidates, 2, 0).map((c) => c.index), [A, C]);
  assert.deepEqual(idx.selectNeighborsHeuristic(base, candidates, 3, 0).map((c) => c.index), [A, C]);
  idx.keepPrunedConnections = true;
  assert.deepEqual(idx.selectNeighborsHeuristic(base, candidates, 3, 0).map((c) => c.index), [A, C, B]);
});
