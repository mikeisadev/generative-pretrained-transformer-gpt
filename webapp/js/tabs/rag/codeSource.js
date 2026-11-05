/**
 * Frammenti di codice letti dai file veri della libreria RAG (js/shared/rag), al momento:
 * il pannello del codice mostra esattamente ciò che la web app esegue, con i numeri di riga
 * giusti, senza copie da tenere allineate a mano.
 */
const cache = new Map();

/** Testo di un file della libreria (una sola richiesta per file). */
function load(file) {
  if (!cache.has(file)) {
    const url = new URL(`../../shared/rag/${file}`, import.meta.url);
    cache.set(
      file,
      fetch(url).then((r) => {
        if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
        return r.text();
      }),
    );
  }
  return cache.get(file);
}

/**
 * Estrae dalla riga che contiene `start` fino alla riga prima di quella che contiene `end`
 * (cercata dopo `start`).
 * @returns {Promise<{file: string, code: string, startLine: number}>}
 */
export async function librarySnippet(file, start, end) {
  const lines = (await load(file)).split("\n");
  const from = lines.findIndex((l) => l.includes(start));
  if (from < 0) throw new Error(`${file}: marcatore non trovato (${start})`);
  let to = end ? lines.findIndex((l, i) => i > from && l.includes(end)) : lines.length;
  if (to < 0) to = lines.length;
  // via le righe vuote e i commenti JSDoc in coda (appartengono alla funzione dopo)
  while (to > from + 1 && /^\s*(\/\*\*|\*|$)/.test(lines[to - 1])) to--;
  return { file: `js/shared/rag/${file}`, code: lines.slice(from, to).join("\n"), startLine: from + 1 };
}

/** Frammenti della libreria usati dal percorso e dalle azioni. */
export const LIBRARY = Object.freeze({
  clean: { file: "clean.js", start: "const STEPS = [", end: "/** Metadati dei passi", title: "Pulizia: i passi in ordine (libreria didattica)" },
  scan: { file: "clean.js", start: "export function scanForInjection", end: "/** I passi di trasformazione", title: "Controllo di sicurezza: segnalare, non cancellare" },
  split: { file: "chunker.js", start: "  splitDetailed(text) {", end: "  splitText(text) {", title: "TokenTextSplitter.doSplit portato in JavaScript" },
  cosine: { file: "similarity.js", start: "export function cosineSimilarity", end: "export function euclideanDistance", title: "Similarità e distanza del coseno" },
  embed: { file: "embedder.js", start: "  explain(text) {", end: "  embed(text) {", title: "Embedding didattico (simulazione di text-embedding-3-small)" },
  level: { file: "hnsw.js", start: "  randomLevel() {", end: "  nodeDistance(a, b) {", title: "HNSW: livello casuale di un nuovo nodo" },
  insert: { file: "hnsw.js", start: "  insert(vector, { id, level: forcedLevel, onEvent } = {}) {", end: "  shrinkConnections(index, layer, emit) {", title: "HNSW, algoritmo 1: INSERT" },
  searchLayer: { file: "hnsw.js", start: "  searchLayer(query, entryPoints, ef, layer, context) {", end: "  selectNeighborsSimple(candidates, m) {", title: "HNSW, algoritmo 2: SEARCH-LAYER" },
  knn: { file: "hnsw.js", start: "  search(query, k, { ef, onEvent } = {}) {", end: "  exactSearch(query, k) {", title: "HNSW, algoritmo 5: K-NN-SEARCH" },
  exact: { file: "hnsw.js", start: "  exactSearch(query, k) {", end: "  toGraph() {", title: "Ricerca esatta: confronta la domanda con tutti" },
  storeSearch: { file: "store.js", start: "  similaritySearch(request) {", end: "  exactSearch(request) {", title: "similaritySearch come in PgVectorStore" },
  prompt: { file: "prompt.js", start: "export const PROMPT_STYLES", end: "/**", title: "Formato dei documenti nel prompt (L18 e L17)" },
});
