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

/**
 * Le rappresentazioni del nucleo (selettore "Architettura · Rete neurale · Cervello"): stessi id di
 * REP_LINKS in core/representations.js, aperti da goto con reps.goto.
 */
const representations = [
  {
    id: "rete",
    kind: "view",
    title: "Rete neurale (Database vettoriale (RAG))",
    text: "La pipeline del RAG come rete, da sinistra a destra: il nostro embedder (feature, hashing con il segno, normalizzazione L2) e la ricerca nel database con i chunk in profondità (confronto con la domanda, similarità del coseno, top-k e soglia), con i numeri veri della domanda; poi, illustrativi, un encoder neurale con l'attention bidirezionale e il mean pooling e un LLM con l'attention causale che legge il prompt aumentato.",
    keywords: [
      "neural network", "neuroni", "rete orizzontale", "profondità", "feature hashing", "trucco dell'hashing", "tf-idf",
      "normalizzazione L2", "prodotto scalare", "cosine similarity", "top-k", "soglia", "encoder", "E5", "multilingual-e5-small",
      "BERT", "attention bidirezionale", "mean pooling", "decoder", "attention causale", "Llama 3", "prompt aumentato",
    ],
  },
  {
    id: "cervello",
    kind: "view",
    title: "Cervello (Database vettoriale (RAG))",
    text: "Il RAG e la memoria del cervello: sistemi di apprendimento complementari (ippocampo veloce, neocorteccia lenta), teoria dell'indice ippocampale, completamento di pattern in CA3, separazione nel giro dentato, con dove l'analogia non regge e le fonti; il circuito trisinaptico dell'ippocampo in 3D.",
    keywords: [
      "brain", "neuroscienze", "memoria", "ippocampo", "hippocampus", "corteccia entorinale", "via perforante", "giro dentato",
      "CA3", "CA1", "subiculum", "circuito trisinaptico", "neocorteccia", "concept cells", "pattern completion", "Hopfield",
      "prefrontale", "confabulazione", "analogie",
    ],
  },
];

export const searchIndex = [...steps, ...actions, ...views, ...representations];
