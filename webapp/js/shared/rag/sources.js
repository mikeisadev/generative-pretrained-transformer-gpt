// Fonti verificabili per il pannello "Fonti" dei tab che usano la libreria RAG.
// Ogni voce dice che cosa conferma, così chi legge sa dove controllare ogni affermazione.

/** @type {ReadonlyArray<{id: string, title: string, url: string, note: string}>} */
export const RAG_SOURCES = Object.freeze([
  {
    id: "spring-ai-splitter",
    title: "Spring AI 2.0.2 - TokenTextSplitter.java",
    url: "https://github.com/spring-projects/spring-ai/blob/v2.0.2/spring-ai-commons/src/main/java/org/springframework/ai/transformer/splitter/TokenTextSplitter.java",
    note: "Algoritmo di chunking (doSplit), default 800/350/5/10000, segni . ? ! e a capo, cl100k_base.",
  },
  {
    id: "spring-ai-text-splitter",
    title: "Spring AI 2.0.2 - TextSplitter.java",
    url: "https://github.com/spring-projects/spring-ai/blob/v2.0.2/spring-ai-commons/src/main/java/org/springframework/ai/transformer/splitter/TextSplitter.java",
    note: "Metadati copiati nei chunk e aggiunta di parent_document_id, chunk_index, total_chunks.",
  },
  {
    id: "spring-ai-pdf-reader",
    title: "Spring AI 2.0.2 - PagePdfDocumentReader.java",
    url: "https://github.com/spring-projects/spring-ai/blob/v2.0.2/document-readers/spring-ai-pdf-document-reader/src/main/java/org/springframework/ai/reader/pdf/PagePdfDocumentReader.java",
    note: "Un Document per pagina; metadati page_number, end_page_number e file_name.",
  },
  {
    id: "spring-ai-pgvector",
    title: "Spring AI 2.0.2 - PgVectorStore.java",
    url: "https://github.com/spring-projects/spring-ai/blob/v2.0.2/vector-stores/spring-ai-pgvector-store/src/main/java/org/springframework/ai/vectorstore/pgvector/PgVectorStore.java",
    note: "Ricerca con distance < 1 - similarityThreshold, ORDER BY distance LIMIT topK, score = 1 - distance.",
  },
  {
    id: "spring-ai-search-request",
    title: "Spring AI 2.0.2 - SearchRequest.java",
    url: "https://github.com/spring-projects/spring-ai/blob/v2.0.2/spring-ai-vector-store/src/main/java/org/springframework/ai/vectorstore/SearchRequest.java",
    note: "Default di Spring AI: topK = 4, similarityThreshold = 0.0 (le lezioni usano 5 e 0.5).",
  },
  {
    id: "pgvector",
    title: "pgvector - README",
    url: "https://github.com/pgvector/pgvector",
    note: "Indice HNSW: m = 16, ef_construction = 64, hnsw.ef_search = 40; filtri applicati dopo la scansione dell'indice.",
  },
  {
    id: "hnsw-paper",
    title: "Malkov, Yashunin - Efficient and robust approximate nearest neighbor search using HNSW graphs",
    url: "https://arxiv.org/abs/1603.09320",
    note: "Algoritmi 1-5 (INSERT, SEARCH-LAYER, selezione dei vicini, K-NN-SEARCH), mL = 1/ln(M), M0 = 2M.",
  },
  {
    id: "openai-embeddings",
    title: "OpenAI - Embeddings guide",
    url: "https://platform.openai.com/docs/guides/embeddings",
    note: "text-embedding-3-small: 1536 dimensioni di default, vettori normalizzati a lunghezza 1.",
  },
  {
    id: "tiktoken",
    title: "OpenAI tiktoken (cl100k_base)",
    url: "https://github.com/openai/tiktoken",
    note: "Tokenizzatore BPE e regex di pre-tokenizzazione approssimati da tokenizer.js.",
  },
  {
    id: "spring-ai-tools",
    title: "Spring AI - Tool calling",
    url: "https://docs.spring.io/spring-ai/reference/api/tools.html",
    note: "@Tool, @ToolParam, ToolContext: base degli strumenti mostrati nel prompt.",
  },
]);
