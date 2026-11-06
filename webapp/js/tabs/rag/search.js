/**
 * Indice della ricerca globale per il tab RAG (contratto in ARCHITECTURE.md, "Ricerca globale").
 *
 * Solo dati leggeri: viene caricato anche se il tab non è mai stato aperto, quindi niente three.js
 * né moduli della scena. I passi vengono dal Percorso (content.js, stessi id del Menu); azioni e
 * viste hanno gli stessi id usati da index.js (player.register e livelli dello zoom).
 */
import { PATH } from "./content.js";

/**
 * Sinonimi e parole che uno studente potrebbe cercare per arrivare a un passo (italiano e inglese).
 * La ricerca guarda anche titolo e testo del passo: qui solo ciò che lì non c'è o è scritto diverso.
 */
const STEP_KEYWORDS = {
  "intro.problema": ["ricerca semantica", "semantic search", "significato", "LIKE", "SQL", "parole chiave", "keyword search"],
  "intro.db": ["vector database", "vector db", "pgvector", "PostgreSQL", "biblioteca", "nuvola di punti"],
  "intro.flusso": ["RAG", "retrieval augmented generation", "pipeline", "due fasi", "flusso"],
  "index.upload": ["caricamento", "upload PDF", "DocumentController", "React", "FormData", "multipart", "WebFlux", "UUID"],
  "index.extract": ["PagePdfDocumentReader", "lettore PDF", "pagine", "Document", "metadati", "metadata", "PDF reader"],
  "index.clean": ["pulizia", "pre-processing", "preprocessing", "cleaning", "intestazioni", "a capo", "sanitizzazione"],
  "index.chunk": ["chunking", "chunk", "TokenTextSplitter", "text splitter", "dividere il testo", "token", "pezzi"],
  "index.embed": ["embedding", "vettore", "vector", "coseno", "cosine similarity", "similarità", "EmbeddingModel", "1536"],
  "index.pgvector": ["vector_store", "tabella", "table", "INSERT", "vectorStore.add", "Postgres", "colonna vector"],
  "hnsw.why": ["indice", "index", "ricerca esatta", "brute force", "forza bruta", "scansione", "lento"],
  "hnsw.simple": ["HNSW", "hierarchical navigable small world", "grafo", "graph", "livelli", "layers", "autostrade"],
  "hnsw.build": ["costruzione", "inserimento", "insert", "vicini", "neighbors", "livello casuale", "M"],
  "hnsw.search": ["ricerca nel grafo", "greedy", "discesa golosa", "beam search", "fascio", "ef_search", "candidati"],
  "hnsw.params": ["parametri", "parametro M", "ef_construction", "ef_search", "complessità", "complexity", "recall", "log n"],
  "search.query": ["domanda", "query", "embedding della domanda", "query embedding", "similaritySearch"],
  "search.topk": ["topK", "top-k", "top k", "soglia", "threshold", "similarityThreshold", "risultati"],
  "search.prompt": ["prompt", "system prompt", "contesto", "context", "prompt caching", "cache", "template"],
  "search.llm": ["ChatClient", "LLM", "modello", "model", "Spring AI", "chiamata", "Flux"],
  "search.stream": ["streaming", "SSE", "server-sent events", "token per token", "React", "EventSource"],
  "security.injection": ["prompt injection", "indirect prompt injection", "attacco", "sicurezza", "security", "documenti non fidati"],
  "security.layers": ["difesa a strati", "defense in depth", "guardrail", "sicurezza", "mitigazione", "validazione"],
};

const steps = PATH.flatMap((section) =>
  section.children.map((item) => ({
    id: item.id,
    kind: "step",
    title: item.title,
    text: item.text,
    keywords: [section.title, ...(STEP_KEYWORDS[item.id] ?? [])],
  })),
);

const actions = [
  {
    id: "indicizzazione",
    kind: "action",
    title: "Indicizzazione: dal PDF al database vettoriale",
    text: "Animazione passo per passo: upload del PDF, estrazione del testo, pulizia, chunking, embedding di ogni chunk, INSERT nella tabella vector_store e inserimento nell'indice HNSW nodo per nodo.",
    keywords: ["indexing", "ingestion", "caricare i documenti", "riempire il database", "ETL", "animazione"],
  },
  {
    id: "ricerca",
    kind: "action",
    title: "Ricerca: dalla domanda alla risposta",
    text: "Animazione passo per passo: embedding della domanda, ricerca HNSW (discesa e fascio) contro ricerca esatta, topK e soglia, prompt con i chunk, chiamata all'LLM e risposta in streaming SSE.",
    keywords: ["retrieval", "search", "query", "domanda", "risposta", "answer", "RAG", "animazione"],
  },
];

/** I tre livelli dello zoom semantico (stessi id del LODController del tab). */
const views = [
  {
    id: "macro",
    kind: "view",
    title: "Panoramica: il RAG in tre blocchi",
    text: "Lo schema più astratto: Indicizzazione → Database vettoriale → Recupero e generazione.",
    keywords: ["panoramica", "overview", "schema", "diagramma a blocchi", "vista d'insieme", "zoom out"],
  },
  {
    id: "blocks",
    kind: "view",
    title: "Blocchi: le undici fasi della pipeline",
    text: "Un blocco per fase, da Upload a Risposta in streaming, con una frase su cosa fa.",
    keywords: ["blocchi", "fasi", "stazioni", "pipeline", "vista d'insieme", "mappa"],
  },
  {
    id: "detail",
    kind: "view",
    title: "Dettaglio: le stazioni da vicino",
    text: "Ogni chunk, ogni vettore e ogni nodo dell'indice HNSW, stazione per stazione.",
    keywords: ["dettaglio", "detail", "zoom in", "da vicino", "stazioni", "chunk", "nodi"],
  },
];

export const searchIndex = [...steps, ...actions, ...views];
