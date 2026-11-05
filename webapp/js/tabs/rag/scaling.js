/**
 * Misura vera di quanto lavoro risparmia HNSW al crescere del numero di vettori.
 *
 * Con i pochi chunk del corso HNSW non conviene (a volte calcola perfino più distanze della
 * ricerca esatta, perché rilegge alcuni nodi a ogni piano). Il vantaggio arriva con tanti
 * vettori: qui costruiamo un indice con i parametri di pgvector (m 16, ef_construction 64,
 * ef_search 40) su vettori sintetici e contiamo, per ogni dimensione dell'indice, i calcoli di
 * distanza e la recall@5 rispetto alla ricerca esatta.
 *
 * Vettori sintetici: 32 dimensioni casuali (distribuzione normale, normalizzati). È un caso
 * difficile (i dati veri hanno più struttura), dichiarato come tale nella UI. Il calcolo gira a
 * pezzi tra un frame e l'altro per non bloccare l'animazione.
 */
import { HNSWIndex, PGVECTOR_HNSW_DEFAULTS, createRng, gaussian, normalize } from "../../shared/rag/index.js";

export const SCALING_CHECKPOINTS = Object.freeze([50, 100, 250, 500, 1000, 2000, 4000]);
const DIMENSIONS = 32;
const QUERIES = 25;
const K = 5;

/** Calcoli di distanza di una ricerca: entry point + vicini visitati per la prima volta in ogni piano. */
function countComparisons(index, query) {
  let comparisons = 0;
  const results = index.search(query, K, {
    ef: index.efSearch,
    onEvent: (e) => {
      if (e.type === "search-start") comparisons = 1;
      else if (e.type === "layer-step") comparisons += e.newlyVisited.length;
    },
  });
  return { comparisons, results };
}

/**
 * @param {(point: {n: number, hnsw: number, exact: number, recall: number}) => void} onPoint
 * @param {{signal?: AbortSignal, budgetMs?: number}} [options]
 */
export async function measureScaling(onPoint, { signal, budgetMs = 8 } = {}) {
  const rng = createRng(2024);
  const randomVector = () => {
    const v = new Float32Array(DIMENSIONS);
    for (let i = 0; i < DIMENSIONS; i++) v[i] = gaussian(rng);
    return normalize(v);
  };
  const index = new HNSWIndex({ ...PGVECTOR_HNSW_DEFAULTS, seed: 1 });
  const queries = Array.from({ length: QUERIES }, randomVector);
  const yieldFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
  for (const n of SCALING_CHECKPOINTS) {
    while (index.size < n) {
      const start = performance.now();
      while (index.size < n && performance.now() - start < budgetMs) index.insert(randomVector());
      await yieldFrame();
      if (signal?.aborted) return;
    }
    let comparisons = 0;
    let recall = 0;
    for (const q of queries) {
      const run = countComparisons(index, q);
      const exact = new Set(index.exactSearch(q, K).map((r) => r.index));
      comparisons += run.comparisons;
      recall += run.results.filter((r) => exact.has(r.index)).length / K;
    }
    onPoint({ n, hnsw: comparisons / QUERIES, exact: n, recall: recall / QUERIES });
    await yieldFrame();
    if (signal?.aborted) return;
  }
}
