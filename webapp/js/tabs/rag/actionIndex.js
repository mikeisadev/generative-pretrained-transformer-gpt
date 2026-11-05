/**
 * Azione "Indicizzazione": il database parte vuoto e si riempie un passo alla volta.
 * Il primo PDF (procedura rimborsi) viene seguito in dettaglio stazione per stazione; gli
 * altri sei entrano poi in sequenza veloce, come sette upload successivi.
 *
 * Ogni passo chiama `baseline(stage)` in enter: porta TUTTE le stazioni allo stato in cui si
 * trovano all'inizio di quella fase, così "passo indietro" e salti funzionano sempre.
 */
import { phase } from "./scene3d.js";

// Fasi in ordine: baseline(s) = tutte le fasi < s completate, le altre allo stato iniziale.
const S = { EMPTY: 0, SEND: 1, EXTRACT: 2, CLEAN: 3, CHUNK: 4, EMBED: 5, TABLE: 6, GRAPH: 7, RESPONSE: 8, BATCH: 9, READY: 10 };

/** Vero se il player ha completato l'ultimo passo dell'azione corrente. */
export function isFinished(player) {
  return player.index >= player.steps.length - 1 && player.t >= 1;
}

export function indexAction(tab) {
  const baseline = (stage) => {
    const { kb, world } = tab;
    const s = world.stations;
    const F = kb.featured.chunks.length;
    const N = kb.inserts.length;
    world.clearSearch();
    s.upload.set(stage > S.BATCH ? "batch" : stage > S.RESPONSE ? "done" : stage > S.SEND ? "saved" : "idle", 1);
    s.extract.set(stage > S.EXTRACT ? 1 : 0);
    s.clean.set(stage > S.CLEAN ? kb.featured.cleaning.steps.length - 1 : -1);
    s.chunk.set(stage > S.CHUNK ? kb.featured.pages.length : 0, stage > S.CHUNK ? 1 : 0);
    s.embed.set(stage > S.EMBED ? F - 1 : 0, stage > S.EMBED ? 1 : 0);
    const rows = stage > S.BATCH ? N : stage > S.TABLE ? F : 0;
    const nodes = stage > S.BATCH ? N : stage > S.GRAPH ? F : 0;
    s.table.set(rows);
    s.hnsw.setGraph(nodes);
    s.hnsw.genericInfo(nodes);
    world.indexed = nodes;
    plotted = -1;
    plotIndexed(stage > S.CHUNK ? Math.max(nodes, F) : 0);
  };

  let before = null; // stato della scena prima dell'azione: si ripristina se viene interrotta

  /** Aggiorna i grafici solo quando il numero di nodi cambia (non a ogni frame). */
  let plotted = -1;
  const plotIndexed = (k) => {
    if (k === plotted) return;
    plotted = k;
    tab.plots.setIndexed(tab.kb, k);
  };

  return {
    id: "indicizzazione",
    title: "Indicizzazione: dal PDF al database vettoriale",
    category: "Indicizzazione",
    summary: "Database vuoto → upload → estrazione → pulizia → chunk → embedding → vector_store → indice HNSW nodo per nodo.",
    onStart() {
      before = tab.snapshot();
      tab.prepareAction();
      plotted = -1;
    },
    onEnd() {
      tab.world.indexed = tab.kb.inserts.length;
    },
    onStop({ player }) {
      // chiusa dopo l'ultimo passo: il database pieno resta; interrotta a metà: com'era prima
      if (!isFinished(player)) tab.restore(before);
    },
    build() {
      const { kb } = tab;
      const f = kb.featured;
      const F = f.chunks.length;
      const N = kb.inserts.length;
      const steps = [];
      const name = f.document.original_filename;

      steps.push({
        title: "Database vuoto",
        caption: `Partiamo da zero: la tabella vector_store non ha righe e l'indice HNSW non ha nodi. Stiamo per caricare ${name}.`,
        duration: 3,
        enter: () => {
          baseline(S.EMPTY);
          tab.focus("table");
          tab.code("properties", ["initialize-schema", "index-type"]);
        },
      });
      steps.push({
        title: "Upload dal frontend React",
        caption: "React mette il PDF in un FormData e fa una POST multipart a /api/documents.",
        duration: 3.5,
        enter: () => {
          baseline(S.SEND);
          tab.focus("upload");
          tab.code("reactUpload", ["formData.append", "method: \"POST\""]);
        },
        update: (p) => tab.world.stations.upload.set("send", p),
      });
      steps.push({
        title: "Il controller salva il file",
        caption: "DocumentController accetta solo .pdf, lo salva come UUID.pdf (mai il nome originale come percorso) e chiama DocumentService.indexPdf.",
        duration: 3.5,
        enter: () => {
          baseline(S.SEND + 1);
          tab.focus("upload");
          tab.code("documentController", ["UUID.randomUUID", "file.transferTo", "documentService.indexPdf"]);
        },
      });
      steps.push({
        title: "Estrazione del testo per pagina",
        caption: `PagePdfDocumentReader legge il PDF con PDFBox: ${f.pages.length} pagine, quindi ${f.pages.length} Document, ognuno con page_number, file_name, source e original_filename.`,
        duration: 4,
        enter: () => {
          baseline(S.EXTRACT);
          tab.focus("extract");
          tab.code("pdfService", ["withPagesPerDocument", "reader.read()", "metadata.put"]);
        },
        update: (p) => tab.world.stations.extract.set(p),
      });
      f.cleaning.steps.forEach((step, i) => {
        const flagged = step.id === "injection-scan";
        const changes = step.changes.filter((c) => c.pageNumber === tab.world.stations.clean.pageNumber).length;
        steps.push({
          title: `Pulizia ${i + 1}/${f.cleaning.steps.length}: ${step.title}`,
          caption: flagged
            ? `Controllo di sicurezza: ${f.cleaning.warnings.length} regole scattano sulla stessa frase ("Nota per l'assistente AI…"). Il testo non si cancella: il chunk viene segnalato (flagged) per una revisione umana.`
            : `${step.description} Su questa pagina: ${changes} ${changes === 1 ? "modifica" : "modifiche"}.`,
          duration: flagged ? 6 : 4,
          enter: () => {
            baseline(S.CLEAN);
            tab.world.stations.clean.set(i);
            tab.focus("clean");
            tab.code(flagged ? "lib:scan" : "lib:clean", flagged ? ["warnings.push"] : [`id: "${step.id}"`]);
          },
        });
      });
      f.pages.forEach((page, i) => {
        const trace = f.traces[i].trace;
        const cut = trace.some((t) => t.cutAtPunctuation);
        steps.push({
          title: `Chunking della pagina ${page.metadata.page_number}`,
          caption: `${f.traces[i].totalTokens} token: ${trace.length === 1 ? "la pagina sta in un solo chunk" : `${trace.length} chunk`}${cut ? ", tagliati all'ultimo segno di punteggiatura dopo minChunkSizeChars (barretta rossa)" : ""}. Ogni chunk eredita i metadati e riceve chunk_index.`,
          duration: 3.5 + trace.length,
          enter: () => {
            baseline(S.CHUNK);
            tab.world.stations.chunk.set(i, 0);
            tab.focus("chunk");
            tab.code("splitDocuments", ["withChunkSize", "withMinChunkSizeChars", "splitter.apply"]);
          },
          update: (p) => {
            tab.world.stations.chunk.set(i, p);
            const before = f.chunks.filter((c) => c.metadata.page_number < page.metadata.page_number).length;
            plotIndexed(before + (p > 0.6 ? trace.filter((t) => t.kept).length : 0));
          },
        });
      });
      f.chunks.forEach((chunk, j) => {
        steps.push({
          title: `Embedding del chunk ${j + 1}/${F}`,
          caption: `${chunk.id}: il testo entra nell'EmbeddingModel ed esce un vettore di ${kb.embedder.dimensions} numeri (simulazione; text-embedding-3-small ne produce 1536). Arancione = positivo, blu = negativo.`,
          duration: j === 0 ? 4.5 : 2.5,
          enter: () => {
            baseline(S.EMBED);
            tab.world.stations.embed.set(j, 0);
            tab.focus("embed");
            tab.code("lib:embed", ["vector[i] = raw[i] * scale", "vector[SHARED_DIMENSION]"]);
          },
          update: (p) => tab.world.stations.embed.set(j, p),
        });
      });
      steps.push({
        title: "INSERT nella tabella vector_store",
        caption: `vectorStore.add(chunks): ${F} righe con id, content, metadata (JSON) ed embedding. Il chunk segnalato in pulizia ha flagged: true nei metadati.`,
        duration: 3.5,
        enter: () => {
          baseline(S.TABLE);
          tab.focus("table");
          tab.code("pgInsert", ["INSERT INTO"]);
        },
        update: (p) => tab.world.stations.table.set(Math.min(F, Math.floor(phase(p, 0, 0.8) * F + 0.999))),
      });
      f.chunks.forEach((chunk, j) => {
        const r = kb.inserts[j];
        steps.push({
          title: `Indice HNSW: inserimento del nodo ${j}`,
          caption:
            j === 0
              ? `Primo nodo: l'indice era vuoto, quindi ${chunk.id} diventa il punto di ingresso (livello estratto: ${r.level}).`
              : `${chunk.id}: livello estratto ${r.level}. Discesa golosa dal punto di ingresso, poi collegamento ai vicini più simili in ogni livello fino al suo (archi azzurri).`,
          duration: 4,
          enter: () => {
            baseline(S.GRAPH);
            tab.world.stations.table.set(F);
            tab.world.stations.hnsw.setGraph(j + 1, { newest: true });
            tab.world.stations.hnsw.insertInfo(j + 1);
            tab.world.indexed = j + 1;
            plotIndexed(Math.max(F, j + 1));
            tab.focus("hnsw.stack");
            tab.code(j === 0 ? "lib:level" : "lib:insert", j === 0 ? ["Math.floor"] : ["searchLayer", "selectNeighbors", "shrinkConnections"]);
          },
          update: (p) => tab.world.stations.hnsw.popNewest(j + 1, phase(p, 0, 0.35)),
        });
      });
      steps.push({
        title: "Risposta al frontend",
        caption: `DocumentService restituisce DocumentUploadResponse: ${f.response.pages} pagine, ${f.response.chunks} chunk, "${f.response.message}".`,
        duration: 3.5,
        enter: () => {
          baseline(S.RESPONSE + 1);
          tab.focus("upload");
          tab.code("documentService", ["vectorStore.add(chunks)", "new DocumentUploadResponse"]);
        },
      });
      steps.push({
        title: "Gli altri 6 documenti",
        caption: `Gli altri PDF seguono la stessa strada: alla fine ${N} chunk in vector_store e ${N} nodi nell'indice. Guarda nei Grafici la piramide dei livelli.`,
        duration: 9,
        enter: () => {
          baseline(S.BATCH);
          tab.focus("hnsw");
          tab.code("sqlPages", ["COUNT(*)"]);
        },
        update: (p) => {
          const k = F + Math.round(phase(p, 0.05, 0.95) * (N - F));
          const s = tab.world.stations;
          s.upload.set("batch", p);
          s.table.set(k);
          s.hnsw.setGraph(k, { newest: k > F });
          if (k > F) s.hnsw.insertInfo(k);
          tab.world.indexed = k;
          plotIndexed(k);
        },
      });
      steps.push({
        title: "La nuvola dei vettori",
        caption: "Gli stessi vettori proiettati in 3D con la PCA: i chunk dello stesso documento tendono a stare vicini. È un'ombra: la ricerca usa tutte le dimensioni.",
        duration: 5,
        enter: () => {
          baseline(S.READY);
          tab.focus("hnsw.cloud");
          tab.code("lib:cosine", ["export function cosineSimilarity"]);
        },
      });
      steps.push({
        title: "Database pronto",
        caption: `Indicizzazione completata: ${N} chunk, indice con ${kb.maxLevel + 1} livelli. Ora prova l'azione "Ricerca" con una domanda.`,
        duration: 3,
        enter: () => {
          baseline(S.READY);
          tab.focus("hnsw");
          tab.code("sqlRows", ["vector_dims"]);
        },
      });
      return steps;
    },
  };
}
