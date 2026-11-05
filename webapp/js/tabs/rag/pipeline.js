/**
 * Dati del tab RAG: tutta la pipeline delle lezioni L17-L18 calcolata davvero con la libreria
 * condivisa js/shared/rag (pulizia, TokenTextSplitter, embedding didattico, HNSW, ricerca).
 *
 * Modulo puro (niente DOM, niente three): prepara le "registrazioni" che le animazioni
 * riproducono passo per passo. Così ogni passo dell'ActionPlayer può ricostruire il proprio
 * stato da zero (serve al "passo indietro") senza ricalcolare nulla.
 *
 * Ordine di inserimento: il database parte VUOTO. Il primo PDF caricato è il documento "in
 * evidenza" (la procedura rimborsi, che contiene la riga di prompt injection della L17), poi
 * arrivano gli altri sei: ogni upload chiama vectorStore.add, proprio come DocumentService.
 */
import {
  CORPUS,
  COURSE_SPLITTER_OPTIONS,
  EXAMPLE_QUESTIONS,
  HashingEmbedder,
  PGVECTOR_HNSW_DEFAULTS,
  VectorStore,
  buildPrompt,
  cleanDocument,
  countTokens,
  pca,
} from "../../shared/rag/index.js";

/** Documento seguito passo per passo dall'azione di indicizzazione. */
export const FEATURED_ID = "rimborsi";

/** Configurazioni del TokenTextSplitter selezionabili nel pannello Azioni. */
export const SPLITTER_PRESETS = Object.freeze({
  didactic: {
    label: "Didattico: chunkSize 200",
    note: "Valore ridotto per vedere i tagli anche su pagine corte. Gli altri parametri sono quelli della L17.",
    options: { ...COURSE_SPLITTER_OPTIONS, chunkSize: 200 },
  },
  l17: {
    label: "Lezione L17: chunkSize 500",
    note: "Esattamente i parametri della L17: quasi tutte le pagine stanno in un solo chunk.",
    options: { ...COURSE_SPLITTER_OPTIONS },
  },
});

/** Configurazioni dell'indice HNSW. */
export const HNSW_PRESETS = Object.freeze({
  didactic: {
    label: "Didattico: m 3, ef_construction 16, ef_search 8",
    note: "Con poche decine di vettori, i valori di pgvector metterebbero quasi tutto in un solo piano: qui li riduciamo per vedere più livelli.",
    options: { M: 3, efConstruction: 16, efSearch: 8, seed: 7 },
  },
  pgvector: {
    label: "pgvector: m 16, ef_construction 64, ef_search 40",
    note: "I valori predefiniti di pgvector. Con meno di 40 chunk la ricerca legge tutto il grafo.",
    options: { ...PGVECTOR_HNSW_DEFAULTS, seed: 42 },
  },
});

/**
 * Ricostruisce lo stato del grafo dopo ogni inserimento, riproducendo gli eventi di INSERT
 * (archi creati) e di "shrink connections" (archi potati). Restituisce, per ogni k = 0…n, gli
 * archi non orientati di ogni piano dopo i primi k inserimenti.
 *
 * @param {{level: number, events: object[]}[]} inserts in ordine di inserimento
 * @returns {{layers: [number, number][][]}[]} states[k].layers[l] = archi [a, b] con a < b
 */
export function replayGraph(inserts) {
  const adjacency = []; // adjacency[node][layer] = vicini (orientati)
  const states = [{ layers: [] }];
  for (const insert of inserts) {
    adjacency.push(Array.from({ length: insert.level + 1 }, () => []));
    for (const e of insert.events) {
      if (e.type === "connect") {
        adjacency[e.node][e.layer] = [...e.neighbors];
        for (const n of e.neighbors) adjacency[n][e.layer].push(e.node);
      } else if (e.type === "prune") {
        adjacency[e.node][e.layer] = [...e.after];
      }
    }
    const maxLevel = Math.max(...inserts.slice(0, states.length).map((i) => i.level));
    const layers = [];
    for (let l = 0; l <= maxLevel; l++) {
      const seen = new Set();
      const edges = [];
      adjacency.forEach((perLayer, a) => {
        for (const b of perLayer[l] ?? []) {
          const lo = Math.min(a, b);
          const hi = Math.max(a, b);
          const key = lo * 10000 + hi;
          if (!seen.has(key)) {
            seen.add(key);
            edges.push([lo, hi]);
          }
        }
      });
      layers.push(edges);
    }
    states.push({ layers });
  }
  return states;
}

/**
 * Trasforma gli eventi di una ricerca HNSW in "fotogrammi" da animare. Ogni fotogramma è uno
 * stato completo (nodo corrente, visitati finora, candidati, migliori trovati, calcoli di
 * distanza eseguiti), quindi si può disegnare qualunque fotogramma senza conoscere i precedenti.
 *
 * Calcoli di distanza = 1 (entry point) + tutti i vicini visitati per la prima volta in ogni
 * piano: è il costo vero della ricerca, da confrontare con gli N calcoli della ricerca esatta.
 */
export function searchFrames(events) {
  const frames = [];
  const visited = new Set();
  const path = []; // archi percorsi dal nodo corrente: [layer, from, to]
  let comparisons = 0;
  let previous = null;
  const snapshot = (extra) => ({
    visited: [...visited],
    path: path.map((p) => [...p]),
    comparisons,
    ...extra,
  });
  for (const e of events) {
    if (e.type === "search-start") {
      comparisons = 1;
      visited.add(e.entryPoint);
      frames.push(snapshot({ kind: "start", layer: e.maxLevel, current: e.entryPoint, candidates: [], found: [], ef: e.ef }));
      previous = e.entryPoint;
    } else if (e.type === "layer-start") {
      const entry = e.entryPoints[0];
      if (previous !== null && previous !== entry) path.push([e.layer, previous, entry]);
      previous = entry;
      frames.push(snapshot({ kind: "layer", layer: e.layer, current: entry, candidates: [], found: [], ef: e.ef }));
    } else if (e.type === "layer-step") {
      if (previous !== null && previous !== e.current) path.push([e.layer, previous, e.current]);
      previous = e.current;
      for (const v of e.newlyVisited) visited.add(v);
      comparisons += e.newlyVisited.length;
      frames.push(
        snapshot({
          kind: "step",
          layer: e.layer,
          current: e.current,
          newlyVisited: [...e.newlyVisited],
          candidates: [...e.candidates],
          found: e.found.map((f) => ({ index: f.index, distance: f.distance })),
        }),
      );
    } else if (e.type === "layer-end") {
      const best = e.found[0]?.index ?? previous;
      if (previous !== null && previous !== best) path.push([e.layer, previous, best]);
      previous = best;
      frames.push(snapshot({ kind: "end", layer: e.layer, current: best, candidates: [], found: e.found.map((f) => ({ index: f.index, distance: f.distance })) }));
    } else if (e.type === "search-end") {
      frames.push(snapshot({ kind: "done", layer: 0, current: e.results[0]?.index ?? previous, candidates: [], found: e.results.map((r) => ({ index: r.index, distance: r.distance })) }));
    }
  }
  return frames;
}

/**
 * Costruisce la base di conoscenza registrando ogni passaggio.
 *
 * @param {{splitter?: keyof SPLITTER_PRESETS, hnsw?: keyof HNSW_PRESETS}} [config]
 */
export function buildKnowledgeBase({ splitter = "didactic", hnsw = "didactic" } = {}) {
  const splitterPreset = SPLITTER_PRESETS[splitter];
  const hnswPreset = HNSW_PRESETS[hnsw];
  const cleaned = CORPUS.map((document) => {
    const { documents, cleaning } = cleanDocument(document);
    return { document, pages: documents, cleaning };
  });
  // L'embedder si "addestra" (idf) una volta su tutte le pagine: come un modello vero, poi resta fisso.
  const embedder = new HashingEmbedder().fit(cleaned.flatMap((c) => c.pages.map((p) => p.text)));
  const store = new VectorStore({ embedder, splitter: splitterPreset.options, hnsw: hnswPreset.options });

  const featuredIndex = cleaned.findIndex((c) => c.document.id === FEATURED_ID);
  const order = [cleaned[featuredIndex], ...cleaned.filter((_, i) => i !== featuredIndex)];

  const uploads = [];
  const inserts = [];
  for (const entry of order) {
    const traces = entry.pages.map((page) => store.splitter.splitDetailed(page.text));
    const firstNode = inserts.length;
    let current = null;
    const added = store.add(entry.pages, {
      onEvent: (e) => {
        if (e.type === "embed") {
          current = { chunkId: e.chunk, vector: e.vector, events: [], level: 0 };
          inserts.push(current);
        } else if (e.type === "index") {
          current.events.push(e.event);
          if (e.event.type === "insert-start") current.level = e.event.level;
        }
      },
    });
    added.forEach((chunk, i) => {
      const record = inserts[firstNode + i];
      record.entry = chunk;
      record.upload = uploads.length;
      record.tokens = countTokens(chunk.text);
    });
    uploads.push({
      ...entry,
      traces,
      firstNode,
      chunks: added,
      response: {
        filename: entry.document.original_filename,
        pages: entry.pages.length,
        chunks: added.length,
        message: "Documento indicizzato correttamente",
      },
    });
  }

  // Proiezione PCA (stessa per grafo, nuvola e grafico 2D): la domanda userà `project`.
  const projection = pca(
    inserts.map((i) => i.vector),
    { components: 3, radius: 1 },
  );
  const graphStates = replayGraph(inserts);
  const maxLevel = Math.max(...inserts.map((i) => i.level));
  const docOrder = order.map((o) => o.document.id);

  return {
    config: { splitter, hnsw, splitterPreset, hnswPreset },
    store,
    embedder,
    uploads,
    featured: uploads[0],
    inserts,
    projection,
    points: projection.points,
    graphStates,
    maxLevel,
    docOrder,
    /** Indice del documento (0…6, ordine di caricamento) a cui appartiene il nodo. */
    docOf: (node) => inserts[node].upload,
    /** Nodi per livello dopo k inserimenti. */
    levelCounts(k = inserts.length) {
      const counts = new Array(maxLevel + 1).fill(0);
      for (let i = 0; i < k; i++) for (let l = 0; l <= inserts[i].level; l++) counts[l]++;
      return counts;
    },
  };
}

/**
 * Esegue una ricerca come RagService.search (topK, soglia) registrando i passi dell'indice e
 * confrontandola con la ricerca esatta.
 */
export function runSearch(kb, question, { topK = 5, similarityThreshold = 0.5 } = {}) {
  const events = [];
  const explanation = kb.embedder.explain(question);
  const results = kb.store.similaritySearch({ query: question, topK, similarityThreshold, onEvent: (e) => events.push(e) });
  const exactAll = kb.store.index.exactSearch(explanation.vector, kb.store.size).map((c) => ({
    node: c.index,
    id: c.id,
    distance: c.distance,
    score: 1 - c.distance,
  }));
  const exact = exactAll.filter((c) => c.distance < 1 - similarityThreshold).slice(0, topK);
  const frames = searchFrames(events);
  const resultIds = new Set(results.map((r) => r.id));
  const recall = exact.length ? exact.filter((e) => resultIds.has(e.id)).length / exact.length : 1;
  // Candidati letti dall'indice (fino a ef_search) prima del filtro sulla soglia, come pgvector.
  const scanned = frames.at(-1)?.found ?? [];
  const prompt = buildPrompt({ question, documents: results, style: "L18" });
  return {
    question,
    topK,
    similarityThreshold,
    explanation,
    vector: explanation.vector,
    point: kb.projection.project(explanation.vector),
    results,
    exact,
    exactAll,
    scanned,
    frames,
    comparisons: frames.at(-1)?.comparisons ?? 0,
    recall,
    prompt,
  };
}

/** Domande d'esempio per i pulsanti della UI (le prime di ogni documento più una fuori tema). */
export function exampleQuestions() {
  const seen = new Set();
  const picks = [];
  for (const q of EXAMPLE_QUESTIONS) {
    const key = q.expected ?? "off";
    if (seen.has(key)) continue;
    seen.add(key);
    picks.push(q.question);
  }
  return picks;
}
