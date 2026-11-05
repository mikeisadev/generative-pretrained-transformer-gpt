// Punto d'ingresso della libreria RAG didattica (moduli puri, senza DOM).
// Pipeline delle lezioni L17-L18:
//   PDF → PagePdfDocumentReader (una pagina = un Document) → pulizia → TokenTextSplitter
//   → EmbeddingModel → PgVectorStore (HNSW) → similaritySearch → prompt per l'LLM.

export * from "./random.js";
export * from "./similarity.js";
export * from "./tokenizer.js";
export * from "./corpus.js";
export * from "./clean.js";
export * from "./chunker.js";
export * from "./embedder.js";
export * from "./hnsw.js";
export * from "./store.js";
export * from "./projection.js";
export * from "./prompt.js";
export * from "./sources.js";

import { cleanDocument } from "./clean.js";
import { CORPUS } from "./corpus.js";
import { HashingEmbedder } from "./embedder.js";
import { VectorStore } from "./store.js";

/**
 * Domande d'esempio con il documento che dovrebbe rispondere (null = fuori tema: con la soglia
 * 0.5 non deve tornare nessun chunk). Usate dai test e utili come suggerimenti nella UI.
 */
export const EXAMPLE_QUESTIONS = Object.freeze([
  { question: "Quanti giorni di ferie ho all'anno?", expected: "ferie" },
  { question: "Come chiedo un permesso per una visita medica?", expected: "ferie" },
  { question: "Qual è il limite di spesa per una cena in trasferta?", expected: "rimborsi" },
  { question: "Entro quando devo presentare la nota spese?", expected: "rimborsi" },
  { question: "Ogni quanto devo cambiare la password?", expected: "sicurezza" },
  { question: "Ho ricevuto una email sospetta, cosa faccio?", expected: "sicurezza" },
  { question: "Cosa devo fare il primo giorno di lavoro?", expected: "onboarding" },
  { question: "Quanti giorni a settimana posso lavorare da casa?", expected: "smart-working" },
  { question: "In quali fasce orarie devo essere reperibile in smart working?", expected: "smart-working" },
  { question: "Come carico un documento PDF nel gestionale?", expected: "manuale" },
  { question: "Chi approva un ordine di acquisto sopra i 5000 euro?", expected: "ordini" },
  { question: "Quanti preventivi servono per scegliere un fornitore?", expected: "ordini" },
  { question: "Che tempo farà domani a Roma?", expected: null },
  { question: "Qual è la ricetta della carbonara?", expected: null },
]);

/**
 * Prepara tutta la pipeline sul corpus del corso: pulizia di ogni PDF, un Document per pagina,
 * embedder addestrato (idf) sulle pagine pulite, chunk indicizzati in HNSW.
 *
 * @param {object} [options]
 * @param {import("./chunker.js").SplitterOptions} [options.splitter] parametri del TokenTextSplitter (default L17)
 * @param {object} [options.hnsw]      parametri di HNSWIndex (default pgvector)
 * @param {object} [options.embedder]  parametri di HashingEmbedder
 * @param {(e: import("./store.js").StoreEvent) => void} [options.onEvent] eventi dell'indicizzazione
 * @returns {{store: VectorStore, embedder: HashingEmbedder, corpus: typeof CORPUS,
 *   cleaned: {document: import("./corpus.js").CorpusDocument, pages: import("./clean.js").RagDocument[],
 *   cleaning: import("./clean.js").CleaningResult}[], pages: import("./clean.js").RagDocument[]}}
 */
export function createDemoStore({ splitter, hnsw, embedder: embedderOptions, onEvent } = {}) {
  const cleaned = CORPUS.map((document) => {
    const { documents, cleaning } = cleanDocument(document);
    return { document, pages: documents, cleaning };
  });
  const pages = cleaned.flatMap((entry) => entry.pages);
  const embedder = new HashingEmbedder(embedderOptions).fit(pages.map((page) => page.text));
  const store = new VectorStore({ embedder, splitter, hnsw });
  store.add(pages, { onEvent });
  return { store, embedder, corpus: CORPUS, cleaned, pages };
}
