/**
 * Sezione "Contesto" del pannello Grafici. I grafici seguono l'animazione: punteggi dei chunk
 * man mano che la ricerca procede, composizione del contesto pezzo per pezzo, token per chiamata,
 * mappa dei vettori e token della risposta arrivati a React.
 */
import { topChartsSection } from "../views/panel.js";
import { contextSegments } from "./pipeline.js";

const short = (name) => String(name).replace(".pdf", "");

export class ContextCharts {
  constructor(panel) {
    this.panel = panel;
    this.charts = null;
  }

  ensure() {
    if (!this.panel) return null;
    const section = topChartsSection(this.panel, "contesto", "Contesto (RAG → LLM)", "Numeri veri della domanda scelta: ricerca su pgvector simulata, contesto assemblato con prompt.js, token stimati.");
    if (this.charts) return this.charts;
    this.charts = {
      scores: section.chart("scores", {
        type: "line",
        title: "Similarità dei chunk con la domanda",
        description: "Score (1 − distanza del coseno) di ogni chunk, dal più simile. Entrano nel contesto al massimo topK chunk, e solo quelli sopra la soglia (la linea piatta); la linea verticale segna l'ultimo chunk tenuto.",
        xLabel: "posizione in classifica",
        yLabel: "score",
        yMin: 0.3,
        yMax: 0.7,
      }),
      composition: section.chart("composition", {
        type: "bar",
        title: "Di che cosa è fatto il contesto",
        description: "Token stimati di ogni parte, nell'ordine in cui arrivano all'LLM. Serve a capire dove va lo spazio (e il costo) di una chiamata.",
        xLabel: "parte del contesto",
        yLabel: "token",
        highlightLabel: "parte appena aggiunta",
      }),
      calls: section.chart("calls", {
        type: "bar",
        title: "Token in ingresso per chiamata",
        description: "Ogni giro di strumenti rimanda tutto il contesto più richiesta e risultato: la seconda chiamata legge più token della prima.",
        xLabel: "chiamata all'LLM",
        yLabel: "token in ingresso",
        highlightLabel: "chiamata in corso",
      }),
      map: section.chart("map", {
        type: "scatter",
        title: "Mappa dei vettori (PCA)",
        description: "I vettori dei chunk e della domanda proiettati su due assi: punti vicini = significati simili. Le distanze vere si misurano su tutte le dimensioni, qui ne vediamo solo due.",
        xLabel: "componente 1",
        yLabel: "componente 2",
      }),
      stream: section.chart("stream", {
        type: "line",
        title: "Risposta arrivata a React",
        description: "Token della risposta ricevuti dal frontend, evento SSE dopo evento: lo streaming mostra il testo mentre l'LLM lo genera.",
        xLabel: "evento SSE",
        yLabel: "token ricevuti",
      }),
    };
    return this.charts;
  }

  /** Disegna tutti i grafici nello stato finale (vista senza animazione). */
  showAll(s) {
    const c = this.ensure();
    if (!c) return;
    this.scores(s, Infinity);
    this.composition(s, Infinity, true);
    this.calls(s, s.calls.length);
    this.map(s);
    this.stream(s, Infinity);
  }

  scores(s, rows) {
    const c = this.ensure();
    if (!c) return;
    if (!s.search) return c.scores.clear();
    const ranking = s.search.ranking.slice(0, 10);
    const n = Math.min(rows, ranking.length);
    c.scores.setData({
      series: [
        { name: "score del chunk", points: ranking.slice(0, n).map((r, i) => [i + 1, r.score]) },
        { name: "soglia", points: [[1, s.threshold], [ranking.length, s.threshold]] },
      ],
    });
    const cut = s.search.results.length;
    c.scores.setHighlight(cut ? { x: cut, label: `ultimo chunk tenuto (${cut})` } : null);
  }

  composition(s, count, withTool = false) {
    const c = this.ensure();
    if (!c) return;
    if (!s.prompt || count <= 0) return c.composition.clear();
    const segs = contextSegments(s, { includeTool: withTool });
    const n = Math.min(count, segs.length);
    c.composition.setData({ labels: segs.map((g) => g.label), values: segs.map((g, i) => (i < n ? g.tokens : 0)), highlight: n - 1 });
  }

  calls(s, upTo) {
    const c = this.ensure();
    if (!c) return;
    if (!s.calls.length || upTo <= 0) return c.calls.clear();
    c.calls.setData({ labels: s.calls.map((x) => x.label), values: s.calls.map((x, i) => (i < upTo ? x.input : 0)), highlight: Math.max(0, Math.min(upTo, s.calls.length) - 1) });
  }

  map(s, { highlight = true } = {}) {
    const c = this.ensure();
    if (!c) return;
    if (!s.search) return c.map.clear();
    const hit = new Set(s.search.results.map((r) => r.node));
    const entries = s.search.ranking.slice().sort((a, b) => a.node - b.node);
    const points = entries.map((e) => ({
      x: s.search.points[e.node][0],
      y: s.search.points[e.node][1],
      label: `${short(e.metadata.original_filename)} p.${e.metadata.page_number}`,
      group: highlight && hit.has(e.node) ? "chunk recuperato" : "chunk",
    }));
    points.push({ x: s.search.query[0], y: s.search.query[1], label: "domanda", group: "domanda" });
    c.map.setData({ points });
    c.map.setHighlight({ label: "domanda" });
  }

  stream(s, tokens) {
    const c = this.ensure();
    if (!c) return;
    const n = Math.min(tokens, s.answer.tokens.length);
    const offset = s.sse[0]?.event === "sources" ? 1 : 0;
    const points = [[0, 0]];
    for (let i = 1; i <= n; i++) points.push([i + offset, i]);
    c.stream.setData({ series: [{ name: "token ricevuti", points }] });
  }
}
