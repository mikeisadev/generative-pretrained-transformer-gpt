/**
 * Quanto ha lavorato davvero l'indice HNSW del tab in una ricerca (puro, niente three.js né DOM: lo usano
 * l'azione "Ricerca" e `hnswNote` della rete neurale del tab, e si prova in Node, tests/fidelity-fixes.test.mjs).
 *
 * Con pochi chunk HNSW non conviene, ma per due motivi diversi, e la didascalia deve dire quello vero:
 * - con l'indice didattico (più livelli) calcola più distanze dei nodi diversi che visita: rilegge alcuni
 *   nodi a ogni livello;
 * - con i valori di pgvector (m 16, ef_search 40) il grafo di poche decine di chunk ha un livello solo ed
 *   ef_search supera i chunk: legge tutto il grafo, ogni nodo una volta.
 * `hnswNote` in tabs/rag/network.js (rete neurale del tab) fa la stessa distinzione con questi conti.
 */

/**
 * @param {{comparisons: number, frames?: {visited?: unknown[]}[]}} run il risultato di runSearch (eventi registrati dalla libreria)
 * @returns {{comparisons: number, distinct: number, rereads: boolean}} distanze calcolate, nodi diversi visitati
 *   (l'ultimo fotogramma della ricerca li ha tutti) e se qualche nodo è stato riletto
 */
export function hnswCost(run) {
  const comparisons = run.comparisons;
  const distinct = run.frames?.at(-1)?.visited?.length ?? comparisons;
  return { comparisons, distinct, rereads: comparisons > distinct };
}

/**
 * Didascalia del passo "HNSW contro ricerca esatta" dell'azione Ricerca.
 * @param {object} run il risultato di runSearch (comparisons, recall, frames)
 * @param {number} N chunk nel database (la ricerca esatta calcola N distanze)
 * @param {number} [efSearch] ef_search dell'indice (kb.store.index.efSearch)
 */
export function hnswSearchCaption(run, N, efSearch) {
  const { comparisons, rereads } = hnswCost(run);
  const saved = N - comparisons;
  if (saved > 0) {
    return `HNSW ha calcolato ${comparisons} distanze invece di ${N} (${saved} in meno), recall ${Math.round(run.recall * 100)}%. Con milioni di chunk il risparmio diventa enorme: vedi i Grafici.`;
  }
  const why = rereads ? "rilegge alcuni nodi a ogni livello del grafo" : `${efSearch ? `con ef_search ${efSearch} e ` : "con "}solo ${N} chunk legge tutto il grafo, ogni nodo una volta`;
  return `Con soli ${N} chunk HNSW ha calcolato ${comparisons} distanze, la ricerca esatta ${N}: su così pochi vettori l'indice non conviene (${why}). Il vantaggio arriva con tanti vettori: vedi i Grafici.`;
}
