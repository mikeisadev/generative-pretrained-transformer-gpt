/**
 * Da dove vengono i benchmark dei run (senza DOM): per il GPT del corso data/benchmark/builtin.json
 * (scaricato una volta, solo quando serve), per i run caricati il loro benchmark.json, letti con
 * readBenchmark. Più l'estratto della validation (val_excerpt.txt) per il calcolo nel browser.
 *
 *   const bench = createBenchSource(state);
 *   await bench.of(runId)     // benchmark normalizzato o null;   await bench.excerpt()  // il testo
 */
import { readBenchmark } from "./benchmarkModel.js";

export const BUILTIN_BENCHMARK_URL = "data/benchmark/builtin.json";
export const EXCERPT_URL = "data/benchmark/val_excerpt.txt";

/**
 * @param {object} state LabState
 * @param {{courseId: string, loadJson?: (url) => Promise<any>, loadText?: (url) => Promise<string>}} o
 */
export function createBenchSource(state, { courseId, loadJson = defaultJson, loadText = defaultText }) {
  const cache = new WeakMap(); // WeightRun → benchmark (un run dimenticato si libera da solo)
  let builtin = null;
  let excerpt = null;

  function builtinBench() {
    builtin ??= loadJson(BUILTIN_BENCHMARK_URL).then(readBenchmark);
    builtin.catch(() => (builtin = null)); // un errore di rete non resta in cache
    return builtin;
  }

  return {
    /** Il benchmark di un run del registro (null se non ne ha uno o se il file è illeggibile). */
    async of(id) {
      if (!id) return null;
      if (id === courseId) return builtinBench();
      const run = state.run(id) ?? (state.has(id) ? await state.loadRun(id) : null);
      if (!run?.benchmark) return null;
      if (!cache.has(run)) cache.set(run, readBenchmark(run.benchmark.raw ?? run.benchmark));
      return cache.get(run);
    },
    /** Il testo dell'estratto della validation (byte esatti: il file è salvato con eol=lf). */
    excerpt() {
      excerpt ??= loadText(EXCERPT_URL);
      excerpt.catch(() => (excerpt = null));
      return excerpt;
    },
  };
}

async function defaultJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`impossibile caricare ${url} (HTTP ${response.status})`);
  return response.json();
}

async function defaultText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`impossibile caricare ${url} (HTTP ${response.status})`);
  return response.text();
}
