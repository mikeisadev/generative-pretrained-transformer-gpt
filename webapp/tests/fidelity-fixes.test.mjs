// Correzioni di fedeltà del round E (richieste 30 e 31 dell'integrazione): node --test webapp/tests/fidelity-fixes.test.mjs
// - nei brain.js dei tab nessun numero che la verifica della ricerca segna "da ricontrollare" resta senza cautela;
// - la didascalia dell'azione Ricerca del RAG dice perché HNSW non conviene con pochi chunk (rilegge nodi con
//   l'indice didattico, legge tutto il grafo con i valori di pgvector), come la rete neurale del tab.
// BRAIN_RESEARCH=/percorso/brain_data.json controlla anche che l'elenco delle fonti segnate sia quello della ricerca.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { hnswNote } from "../js/tabs/rag/network.js";
import { hnswCost, hnswSearchCaption } from "../js/tabs/rag/hnswCost.js";
import { buildKnowledgeBase, runSearch } from "../js/tabs/rag/pipeline.js";
import { EXAMPLE_QUESTIONS } from "../js/shared/rag/index.js";

const RESEARCH = process.env.BRAIN_RESEARCH ?? "";

/**
 * Le fonti che la verifica della ricerca dell'8 novembre 2026 segna "da ricontrollare online" o "coerente, numero
 * da ricontrollare online" (brain_data.json → verification.checked_sources): copiate qui perché la ricerca sta
 * fuori dal repository; con BRAIN_RESEARCH il test controlla che siano proprio quelle.
 */
const FLAGGED = [
  "andersen1992", "andersen1993", "arancibia2017", "ashmore2008", "carandini2000", "microns2025", "curcio1990rgc", "deitcher2017",
  "eyal2018", "gardnermedwin1972", "goldstein2025", "grace1984a", "herculano2006", "hijazi2023", "hursh1939", "kincaid1998",
  "leuba1994", "makary2011", "mohan2015", "palkovits1972", "pichitpornchai1994", "rolls2016", "streng2018", "tang2001",
];
/**
 * Immagini e video segue una regola più stretta: il suo generatore scarta i numeri segnati, salvo quelli ricontrollati
 * a parte e dichiarati nell'intestazione di brain.js (Curcio e Allen 1990 sull'abstract, BioNumbers 105617 per V1).
 */
const RECHECKED = { diffusion: new Set(["curcio1990rgc", "leuba1994"]) };
const TABS = ["transformer", "rag", "diffusion", "audio", "modern-llm", "lab"];
const CAUTION = /ricontrollar/i;

test("brain.js: ogni numero e ogni fonte «da ricontrollare» porta la sua cautela, in tutti i tab", async () => {
  const flagged = new Set(FLAGGED);
  let checked = 0;
  for (const tab of TABS) {
    const { brainSpec } = await import(`../js/tabs/${tab}/brain.js`);
    const skip = RECHECKED[tab] ?? new Set();
    for (const f of brainSpec.facts ?? []) {
      const id = f.source?.id;
      if (!flagged.has(id) || skip.has(id) || !/\d/.test(`${f.claim} ${f.value}`)) continue;
      assert.match(f.note ?? "", CAUTION, `${tab}: numero «${f.claim}» (${id}) senza cautela`);
      checked++;
    }
    const walk = (v, path) => {
      if (Array.isArray(v)) return v.forEach((x, i) => walk(x, `${path}[${i}]`));
      if (!v || typeof v !== "object") return;
      if (typeof v.id === "string" && typeof v.title === "string" && flagged.has(v.id) && !skip.has(v.id)) {
        assert.match(v.note ?? "", CAUTION, `${tab}: fonte ${v.id} senza cautela in ${path}`);
        checked++;
      }
      for (const [k, x] of Object.entries(v)) walk(x, `${path}.${k}`);
    };
    walk(brainSpec, `${tab}.brainSpec`);
  }
  assert.ok(checked > 40, `controllate ${checked} voci`);
  // il numero di Eyal et al. 2018 nel Transformer (la richiesta da cui è partito il controllo)
  const { brainSpec: tf } = await import("../js/tabs/transformer/brain.js");
  const spines = tf.facts.find((f) => f.source.id === "eyal2018");
  assert.match(spines.value, /25\.000–30\.000/);
  assert.match(spines.note, /Da ricontrollare \(verifica della ricerca dell'8 novembre 2026\).*25\.000–30\.000 spine va ricontrollato/);
});

test("brain.js: l'elenco delle fonti segnate è quello della verifica della ricerca", { skip: !(RESEARCH && existsSync(RESEARCH)) && "BRAIN_RESEARCH non indicato (la ricerca sta fuori dal repository)" }, async () => {
  const data = JSON.parse(await readFile(RESEARCH, "utf8"));
  const ids = data.verification.checked_sources.filter((v) => /ricontrollare/.test(v.status)).map((v) => v.id);
  assert.deepEqual([...ids].sort(), [...FLAGGED].sort());
});

test("RAG, azione Ricerca: con pochi chunk la didascalia dice perché HNSW non conviene, come la rete del tab", () => {
  const question = EXAMPLE_QUESTIONS[2].question;
  // indice didattico (4 livelli): più distanze che nodi diversi visitati → "rilegge"
  const kb = buildKnowledgeBase();
  const run = runSearch(kb, question, { topK: 5, similarityThreshold: 0.5 });
  const N = kb.inserts.length;
  const cost = hnswCost(run);
  assert.equal(cost.distinct, run.frames.at(-1).visited.length);
  const caption = hnswSearchCaption(run, N, kb.store.index.efSearch);
  if (run.comparisons < N) assert.match(caption, /in meno/);
  else {
    assert.equal(cost.rereads, run.comparisons > cost.distinct);
    assert.match(caption, cost.rereads ? /rilegge alcuni nodi a ogni livello del grafo/ : /legge tutto il grafo/);
  }
  // valori di pgvector (m 16, ef_search 40): un livello solo, ef_search più grande dei 21 chunk → legge tutto
  // il grafo, ogni nodo una volta (prima la didascalia diceva "rilegge alcuni nodi": falso)
  const pg = buildKnowledgeBase({ splitter: "l17", hnsw: "pgvector" });
  const pgRun = runSearch(pg, question, { topK: 5, similarityThreshold: 0.5 });
  const M = pg.inserts.length;
  const pgCost = hnswCost(pgRun);
  assert.equal(pgCost.rereads, false, "ogni nodo letto una volta");
  assert.equal(pgRun.comparisons, M, "tutto il grafo");
  const pgCaption = hnswSearchCaption(pgRun, M, pg.store.index.efSearch);
  assert.match(pgCaption, new RegExp(`con ef_search ${pg.store.index.efSearch} e solo ${M} chunk legge tutto il grafo, ogni nodo una volta`));
  assert.doesNotMatch(pgCaption, /rilegge/);
  // la stessa distinzione della scheda della rete neurale (hnswNote di network.js)
  assert.doesNotMatch(hnswNote(pgRun, M, pg.store.index.efSearch), /rilegge/);
  if (cost.rereads) assert.match(hnswNote(run, N, kb.store.index.efSearch), /rilegge/);
  // tanti chunk: il risparmio
  assert.match(hnswSearchCaption({ comparisons: 120, recall: 0.8, frames: [] }, 10_000, 40), /120 distanze invece di 10000 \(9880 in meno\), recall 80%/);
});
