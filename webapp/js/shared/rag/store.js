// VectorStore didattico con la stessa interfaccia e la stessa semantica del PgVectorStore di
// Spring AI 2.0.2 configurato come nelle lezioni (distance-type COSINE_DISTANCE, index-type HNSW).
//
// Cosa fa PgVectorStore.doSimilaritySearch (verificato sul sorgente v2.0.2):
//   distance = 1 - similarityThreshold
//   SELECT *, embedding <=> ? AS distance FROM vector_store
//     WHERE embedding <=> ? < distance ORDER BY distance LIMIT topK
//   e per ogni riga: score = 1 - distance, metadata.distance = distance.
// Con l'indice HNSW, pgvector legge dall'indice al massimo hnsw.ef_search candidati (40 di
// default) e applica il WHERE dopo: lo stesso facciamo qui.
//
// Nota sui default: SearchRequest di Spring AI ha topK = 4 e similarityThreshold = 0.0 (accetta
// tutto); qui i default sono quelli usati nelle lezioni, topK = 5 e similarityThreshold = 0.5.

import { COURSE_SPLITTER_OPTIONS, TokenTextSplitter } from "./chunker.js";
import { HNSWIndex, PGVECTOR_HNSW_DEFAULTS } from "./hnsw.js";
import { cosineDistance } from "./similarity.js";

/** Default di ricerca usati nelle lezioni L17-L18. */
export const COURSE_SEARCH_DEFAULTS = Object.freeze({ topK: 5, similarityThreshold: 0.5 });

/**
 * @typedef {object} StoredChunk
 * @property {string} id
 * @property {string} text
 * @property {Record<string, unknown>} metadata
 * @property {Float32Array} vector
 * @property {number} node  indice del nodo nel grafo HNSW
 */

/**
 * @typedef {object} SearchResult
 * Come un Document restituito da Spring AI: testo, metadati (con `distance`) e score.
 * @property {string} id
 * @property {string} text
 * @property {Record<string, unknown>} metadata
 * @property {number} score     1 - distance (= cosine similarity)
 * @property {number} distance  cosine distance
 * @property {number} node
 */

/**
 * @typedef {object} StoreEvent
 * - {type: "split", document, chunks: string[]}        un documento diviso in chunk
 * - {type: "embed", chunk, vector}                     embedding calcolato
 * - {type: "index", chunk, node, event: HnswEvent}     evento dell'inserimento nel grafo
 * - {type: "stored", chunk, node}                      chunk salvato
 */

/**
 * Controlla e normalizza la richiesta di ricerca. Accetta anche una stringa (solo la query).
 * I controlli ricalcano il RagController della L17 ("Query obbligatoria") e gli Assert di
 * SearchRequest (topK >= 0, soglia in [0, 1]).
 */
function parseRequest(request) {
  const { query, topK = COURSE_SEARCH_DEFAULTS.topK, similarityThreshold = COURSE_SEARCH_DEFAULTS.similarityThreshold, onEvent } =
    typeof request === "string" ? { query: request } : request ?? {};
  if (typeof query !== "string" || query.trim() === "") throw new TypeError("Query obbligatoria");
  if (!Number.isInteger(topK) || topK < 0) throw new RangeError("topK deve essere un intero >= 0");
  if (!(similarityThreshold >= 0 && similarityThreshold <= 1)) {
    throw new RangeError("similarityThreshold deve essere nell'intervallo [0, 1]");
  }
  return { query, topK, similarityThreshold, onEvent };
}

/** Vector store in memoria: chunk → embedding → indice HNSW. */
export class VectorStore {
  /**
   * @param {object} options
   * @param {import("./embedder.js").HashingEmbedder} options.embedder  modello di embedding (già addestrato con `fit`)
   * @param {TokenTextSplitter | import("./chunker.js").SplitterOptions} [options.splitter] splitter o suoi parametri (default: L17)
   * @param {object} [options.hnsw]  parametri di HNSWIndex (default: pgvector, seed 42)
   */
  constructor({ embedder, splitter = COURSE_SPLITTER_OPTIONS, hnsw = {} } = {}) {
    if (!embedder || typeof embedder.embed !== "function") throw new TypeError("VectorStore: serve un embedder");
    this.embedder = embedder;
    this.splitter = splitter instanceof TokenTextSplitter ? splitter : new TokenTextSplitter(splitter);
    this.index = new HNSWIndex({ ...PGVECTOR_HNSW_DEFAULTS, ...hnsw, distance: cosineDistance });
    this.distanceType = "COSINE_DISTANCE";
    this.indexType = "HNSW";
    /** @type {StoredChunk[]} */
    this.entries = [];
    /** @type {Map<string, StoredChunk>} */
    this.byId = new Map();
  }

  /** Numero di chunk salvati. */
  get size() {
    return this.entries.length;
  }

  /**
   * Aggiunge documenti: li divide in chunk, calcola gli embedding e li inserisce nell'indice
   * (in Spring AI: splitter.apply(documents) e poi vectorStore.add(chunks)).
   *
   * @param {import("./clean.js").RagDocument[]} documents
   * @param {{onEvent?: (e: StoreEvent) => void}} [options]
   * @returns {StoredChunk[]} i chunk aggiunti
   */
  add(documents, { onEvent } = {}) {
    if (!Array.isArray(documents)) throw new TypeError("add: serve una lista di documenti");
    const added = [];
    for (const document of documents) {
      const chunks = this.splitter.apply([document]);
      onEvent?.({ type: "split", document: document.id, chunks: chunks.map((c) => c.id) });
      for (const chunk of chunks) {
        if (this.byId.has(chunk.id)) throw new Error(`Chunk già presente: ${chunk.id}`);
        const vector = this.embedder.embed(chunk.text);
        onEvent?.({ type: "embed", chunk: chunk.id, vector });
        const nextNode = this.index.size;
        const node = this.index.insert(vector, {
          id: chunk.id,
          onEvent: onEvent && ((event) => onEvent({ type: "index", chunk: chunk.id, node: nextNode, event })),
        });
        const entry = { id: chunk.id, text: chunk.text, metadata: chunk.metadata, vector, node };
        this.entries.push(entry);
        this.byId.set(entry.id, entry);
        added.push(entry);
        onEvent?.({ type: "stored", chunk: chunk.id, node });
      }
    }
    return added;
  }

  /**
   * Chunk salvato con un certo id.
   *
   * @param {string} id
   * @returns {StoredChunk | undefined}
   */
  get(id) {
    return this.byId.get(id);
  }

  /** Converte un risultato dell'indice nel formato di Spring AI. */
  toResult({ index, distance }) {
    const entry = this.byId.get(this.index.nodes[index].id);
    return {
      id: entry.id,
      text: entry.text,
      metadata: { ...entry.metadata, distance },
      score: 1 - distance,
      distance,
      node: entry.node,
    };
  }

  /**
   * Ricerca semantica con l'indice HNSW (approssimata), come vectorStore.similaritySearch.
   *
   * @param {string | {query: string, topK?: number, similarityThreshold?: number,
   *   onEvent?: (e: import("./hnsw.js").HnswEvent) => void}} request
   * @returns {SearchResult[]} al massimo topK risultati con distance < 1 - similarityThreshold
   */
  similaritySearch(request) {
    const { query, topK, similarityThreshold, onEvent } = parseRequest(request);
    if (topK === 0 || this.size === 0) return [];
    const maxDistance = 1 - similarityThreshold;
    // L'indice restituisce fino a ef_search candidati (almeno topK); il WHERE si applica dopo.
    const width = Math.max(this.index.efSearch, topK);
    const candidates = this.index.search(this.embedder.embed(query), width, { ef: width, onEvent });
    return candidates
      .filter((c) => c.distance < maxDistance)
      .slice(0, topK)
      .map((c) => this.toResult(c));
  }

  /**
   * Ricerca esatta (forza bruta, come pgvector senza indice): confronta la domanda con tutti i
   * chunk. Serve per mostrare che HNSW trova (quasi sempre) gli stessi risultati visitando
   * molti meno nodi.
   *
   * @param {string | {query: string, topK?: number, similarityThreshold?: number}} request
   * @returns {SearchResult[]}
   */
  exactSearch(request) {
    const { query, topK, similarityThreshold } = parseRequest(request);
    if (topK === 0 || this.size === 0) return [];
    const maxDistance = 1 - similarityThreshold;
    return this.index
      .exactSearch(this.embedder.embed(query), this.size)
      .filter((c) => c.distance < maxDistance)
      .slice(0, topK)
      .map((c) => this.toResult(c));
  }

  /**
   * Struttura del grafo HNSW con gli id dei chunk, per disegnarla.
   */
  graph() {
    return this.index.toGraph();
  }

  /**
   * Statistiche: numero di chunk e documenti, dimensione dei vettori, livelli HNSW.
   */
  stats() {
    const parents = new Set(this.entries.map((e) => e.metadata.parent_document_id ?? e.id));
    const files = new Set(this.entries.map((e) => e.metadata.original_filename).filter(Boolean));
    return {
      chunks: this.size,
      documents: parents.size,
      files: files.size,
      dimensions: this.index.dimensions ?? this.embedder.dimensions,
      distanceType: this.distanceType,
      indexType: this.indexType,
      splitter: this.splitter.options,
      embedder: this.embedder.info,
      hnsw: this.index.stats(),
    };
  }
}
