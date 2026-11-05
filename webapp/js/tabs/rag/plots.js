/**
 * Grafici del tab RAG (ChartsPanel del nucleo). Si aggiornano mentre le azioni procedono:
 *   Indicizzazione: lunghezze dei chunk in token, nodi per livello dell'indice HNSW;
 *   Ricerca: embedding proiettati con la domanda, punteggi con soglia e topK, calcoli di distanza
 *            HNSW contro ricerca esatta, recall e risparmio al crescere del numero di vettori.
 */
import { measureScaling } from "./scaling.js";
import { shortId } from "./stationsAnswer.js";

export class RagPlots {
  constructor(charts) {
    this.charts = charts;
    const idx = charts.section("rag-index", "Indicizzazione", "Si aggiornano mentre i documenti entrano nel database.");
    this.lengths = idx.chart("lengths", {
      type: "histogram",
      title: "Lunghezza dei chunk (token)",
      description: "A cosa serve: controllare che il TokenTextSplitter produca pezzi di dimensione simile. Chunk troppo corti hanno poco contesto, troppo lunghi mescolano argomenti diversi.",
      xLabel: "token per chunk",
      yLabel: "chunk",
    });
    this.levels = idx.chart("levels", {
      type: "bar",
      title: "Nodi per livello dell'indice HNSW",
      description: "A cosa serve: vedere la piramide di HNSW. Tutti i nodi stanno al livello 0, ogni livello sopra ne contiene circa M volte meno (livello estratto a caso).",
      xLabel: "livello",
      yLabel: "nodi",
    });
    const search = charts.section("rag-search", "Ricerca", "Valori veri calcolati sulla domanda corrente.");
    this.scatter = search.chart("scatter", {
      type: "scatter",
      title: "Embedding proiettati in 2D (PCA) con la domanda",
      description: "A cosa serve: intuire perché certi chunk vengono recuperati. Punti vicini hanno significato simile; ma 2 dimensioni su centinaia sono solo un'ombra: la ricerca vera usa tutte le dimensioni.",
      xLabel: "componente 1",
      yLabel: "componente 2",
    });
    this.scores = search.chart("scores", {
      type: "line",
      title: "Punteggi di similarità, dal più alto",
      description: "A cosa serve: vedere l'effetto di topK e soglia. Passano solo i chunk sopra la linea della soglia e, tra questi, i primi topK (linea verticale).",
      xLabel: "posizione in classifica",
      yLabel: "score (1 − distanza)",
      yMin: 0,
      yMax: 1,
    });
    this.cost = search.chart("cost", {
      type: "bar",
      title: "Calcoli di distanza: HNSW contro ricerca esatta",
      description: "A cosa serve: misurare il lavoro della ricerca. Con pochi chunk HNSW non conviene (rilegge alcuni nodi a ogni livello); il vantaggio arriva con tanti vettori, vedi il grafico sotto.",
      yLabel: "calcoli di distanza",
    });
    const scale = charts.section("rag-scale", "HNSW al crescere dei vettori", "Misura vera su vettori sintetici (32 dimensioni casuali) con i parametri di pgvector: m 16, ef_construction 64, ef_search 40.");
    this.scaleCost = scale.chart("scale-cost", {
      type: "line",
      title: "Calcoli di distanza per ricerca",
      description: "A cosa serve: vedere perché pgvector usa un indice. La ricerca esatta cresce come N, HNSW molto più lentamente.",
      xLabel: "vettori nell'indice (N)",
      yLabel: "calcoli di distanza",
      yScale: "log",
    });
    this.recall = scale.chart("recall", {
      type: "line",
      title: "Recall@5 dell'indice",
      description: "A cosa serve: ricordare che HNSW è approssimato. Recall = frazione dei 5 vicini veri che l'indice trova davvero; si alza aumentando ef_search, al prezzo di più calcoli.",
      xLabel: "vettori nell'indice (N)",
      yLabel: "recall@5",
      yMin: 0,
      yMax: 1,
    });
    this.scaling = null;
  }

  /** Lunghezze dei primi k chunk inseriti e nodi per livello dopo k inserimenti. */
  setIndexed(kb, k) {
    this.lengths.setData({ values: kb.inserts.slice(0, k).map((r) => r.tokens), bins: 12 });
    const counts = kb.levelCounts(k);
    this.levels.setData({ labels: counts.map((_, l) => `livello ${l}`), values: counts });
    this.setProjection(kb, k, null);
  }

  /** Scatter dei primi k vettori (gruppo = documento) e, se c'è, della domanda. */
  setProjection(kb, k, run) {
    const points = kb.inserts.slice(0, k).map((r, i) => ({
      x: kb.points[i][0],
      y: kb.points[i][1],
      label: shortId(r.entry.id),
      group: r.entry.metadata.original_filename.replace(".pdf", ""),
    }));
    if (run) points.push({ x: run.point[0], y: run.point[1], label: "domanda", group: "domanda" });
    this.scatter.setData({ points });
    this.scatter.setHighlight(run ? { label: "domanda" } : null);
  }

  /** Grafici della ricerca; `comparisons` = calcoli eseguiti finora (cresce durante l'animazione). */
  setSearch(kb, run, comparisons = run?.comparisons ?? 0) {
    if (!run) {
      this.scores.clear();
      this.cost.clear();
      return;
    }
    const ranked = run.exactAll.slice(0, 15);
    this.scores.setData({
      series: [
        { name: "chunk (ricerca esatta)", points: ranked.map((r, i) => [i + 1, r.score]) },
        { name: `soglia ${run.similarityThreshold}`, points: [[1, run.similarityThreshold], [ranked.length, run.similarityThreshold]] },
      ],
    });
    this.scores.setHighlight({ x: run.topK, label: `topK = ${run.topK}` });
    this.cost.setData({ labels: ["HNSW", "ricerca esatta"], values: [comparisons, kb.inserts.length], highlight: 0 });
  }

  /** Avvia una sola volta la misura su vettori sintetici (gira a pezzi tra un frame e l'altro). */
  startScaling(kb, recallOnCorpus) {
    if (this.scaling) return;
    const hnsw = [];
    const exact = [];
    const recall = [];
    this.scaling = measureScaling((pt) => {
      hnsw.push([pt.n, pt.hnsw]);
      exact.push([pt.n, pt.exact]);
      recall.push([pt.n, pt.recall]);
      this.scaleCost.setData({ series: [{ name: "ricerca esatta (= N)", points: exact }, { name: "HNSW", points: hnsw }] });
      this.recall.setData({
        series: [
          { name: "HNSW, vettori sintetici", points: recall },
          { name: "corpus del corso (domande d'esempio)", points: [[kb.inserts.length, recallOnCorpus]] },
        ],
      });
    });
  }
}
