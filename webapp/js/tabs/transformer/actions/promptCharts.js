/**
 * Grafici della sezione "Prompt" del ChartsPanel, aggiornati passo per passo dall'azione
 * "Dal prompt all'output". Tutti i valori vengono da promptRun.js (motore JS, pesi addestrati).
 */
import { el } from "../../../core/dom.js";
import { visibleChar } from "./math.js";

const LN_V = Math.log(65);

export class PromptCharts {
  /** @param {import("../../../core/ui/charts.js").ChartsPanel} panel */
  constructor(panel, { layers, heads }) {
    this.layers = layers;
    this.heads = heads;
    this.block = 0;
    this.run = null;
    this.token = -1;
    this.shown = ""; // "token:serie" mostrato, per non ridisegnare tutto se non cambia nulla
    const section = panel.section(
      "prompt",
      "Prompt (inferenza)",
      "Dati veri del forward sul tuo prompt (motore JavaScript, pesi addestrati). Si aggiornano mentre l'azione «Dal prompt all'output» procede.",
    );
    this.section = section;
    this.dist = section.chart("dist", {
      type: "bar",
      title: "Distribuzione del prossimo token (top 10)",
      description: "A cosa serve: vedere tra quali caratteri il modello è indeciso. Cosa guardare: una barra dominante = modello sicuro; la barra evidenziata è il token estratto, che non è per forza il più probabile.",
      percent: true,
      highlightLabel: "estratto",
      yLabel: "probabilità",
    });
    this.entropy = section.chart("entropy", {
      type: "line",
      title: "Entropia della distribuzione, token per token",
      description: "A cosa serve: misura l'incertezza (in nat). 0 = certezza, ln 65 ≈ 4,17 = nessuna preferenza. Cosa guardare: picchi a inizio parola, valori bassi dove la parola è già decisa.",
      xLabel: "token generato",
      yLabel: "nat",
      yMin: 0,
    });
    this.chosen = section.chart("chosen", {
      type: "line",
      title: "Probabilità del token estratto",
      description: "A cosa serve: capire se il campionamento ha preso la scelta ovvia o una sorpresa. Cosa guardare: valori bassi = il caso ha scelto un token poco probabile (succede di più con temperatura alta).",
      xLabel: "token generato",
      yLabel: "probabilità",
      yMin: 0,
      yMax: 1,
    });

    this.blockSelect = el(
      "select",
      { className: "tfa-select", ariaLabel: "Blocco mostrato nelle heatmap di attention", onchange: (e) => this.setBlock(Number(e.target.value), true) },
      ...Array.from({ length: layers }, (_, l) => el("option", { value: String(l) }, `Blocco ${l}`)),
    );
    this.attention = Array.from({ length: heads }, (_, h) =>
      section.chart(`att${h}`, {
        type: "heatmap",
        title: `Attention, head ${h}`,
        description: h === 0
          ? "A cosa serve: vedere quali token guarda ogni posizione (righe = chi guarda, colonne = chi viene guardato; ogni riga somma a 1). Cosa guardare: diagonale = guarda sé stesso, colonne accese = token importanti; il triangolo in alto è vuoto per la maschera causale."
          : "Stesso blocco, un'altra head: ognuna cerca relazioni diverse.",
      }),
    );
    this.attention[0].root.before(el("div", { className: "tfa-chart-controls" }, el("label", {}, "Heatmap di attention del ", this.blockSelect)));

    this.lens = section.chart("lens", {
      type: "heatmap",
      title: "Logit lens: cosa prevede il modello dopo ogni blocco",
      description: "A cosa serve: applicare LayerNorm finale + lm_head al residual stream di ogni strato, come se il modello si fermasse lì (righe: emb = solo embedding, b0…b3 = dopo ogni blocco). Cosa guardare: in quale blocco la previsione finale diventa riconoscibile.",
      xLabel: "i 10 token più probabili alla fine",
    });
    this.norms = section.chart("norms", {
      type: "line",
      title: "Norma del residual stream per strato",
      description: "A cosa serve: ogni blocco SOMMA il suo contributo al residual stream; la norma dice quanto è cresciuto. Cosa guardare: una crescita regolare è normale; salti enormi indicherebbero un blocco che domina.",
      xLabel: "strato (0 = embedding, 1 = dopo il blocco 0, …)",
      yLabel: "norma L2",
    });
    this.ffn = section.chart("ffn", {
      type: "histogram",
      title: "Attivazioni del feed-forward (prima della ReLU)",
      description: "A cosa serve: vedere quanti dei 256 neuroni nascosti si accendono. Cosa guardare: tutto ciò che sta a sinistra dello 0 viene azzerato dalla ReLU; la nota dice la quota di neuroni spenti.",
      xLabel: "valore",
    });
  }

  /** Nuovo prompt: svuota le serie. */
  /** Porta la sezione Prompt in vista nel pannello dei grafici. */
  reveal() {
    this.section.root.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  reset(run) {
    this.run = run;
    this.token = -1;
    this.shown = "";
    this.entropy.setData({ series: [{ name: "entropia", points: [] }, { name: "massimo: ln 65", points: [] }] });
    this.chosen.setData({ series: [{ name: "p(token estratto)", points: [] }] });
    this.dist.clear();
    this.lens.clear();
    this.norms.clear();
    this.ffn.clear();
    this.attention.forEach((c) => c.clear());
  }

  setBlock(block, fromUser = false) {
    this.block = block;
    this.blockSelect.value = String(block);
    if (fromUser || this.token >= 0) this.drawBlock();
  }

  /**
   * Mostra i dati del token k (distribuzione, attention, lens…). Le serie nel tempo arrivano fino
   * al token k compreso (`appendSeries`, dopo l'estrazione) o fino al precedente (prima
   * dell'estrazione): ogni passo imposta lo stato completo, anche saltando avanti e indietro.
   */
  showToken(k, { appendSeries = true } = {}) {
    const run = this.run;
    const s = run?.steps[k];
    if (!s) return;
    const key = `${k}:${appendSeries}`;
    if (key === this.shown) return;
    this.shown = key;
    this.token = k;
    const vocab = run.vocab;
    const order = Array.from(s.probs.keys()).sort((a, b) => s.probs[b] - s.probs[a]).slice(0, 10);
    const top = order.includes(s.id) ? order : [...order.slice(0, 9), s.id];
    this.dist.setData({
      labels: top.map((id) => visibleChar(vocab[id])),
      values: top.map((id) => s.probs[id]),
      highlight: top.indexOf(s.id),
    });
    this.appendUpTo(appendSeries ? k : k - 1);
    this.lens.setData({
      rows: s.lens.length,
      cols: order.length,
      values: Float32Array.from(s.lens.flatMap((l) => order.map((id) => l.probs[id]))),
      rowLabels: s.lens.map((_, i) => (i === 0 ? "emb" : `b${i - 1}`)),
      colLabels: order.map((id) => visibleChar(vocab[id])),
      min: 0,
    });
    this.norms.setData({
      series: [
        { name: "ultima posizione", points: s.residualNorm.map((n, i) => [i, n.last]) },
        { name: "media sulle posizioni", points: s.residualNorm.map((n, i) => [i, n.mean]) },
      ],
    });
    this.drawBlock();
  }

  /** Serie "nel tempo" fino al token k (compreso; −1 = nessuno), ricostruite: funziona anche col passo indietro. */
  appendUpTo(k) {
    const steps = this.run.steps.slice(0, k + 1);
    this.entropy.setData({
      series: [
        { name: "entropia", points: steps.map((s, i) => [i + 1, s.entropy]) },
        { name: "massimo: ln 65", points: [[1, LN_V], [Math.max(2, this.run.steps.length), LN_V]] },
      ],
    });
    this.chosen.setData({ series: [{ name: "p(token estratto)", points: steps.map((s, i) => [i + 1, s.chosenProb]) }] });
    const highlight = k >= 0 ? { x: k + 1, label: `token ${k + 1}` } : null;
    this.entropy.setHighlight(highlight);
    this.chosen.setHighlight(highlight);
  }

  drawBlock() {
    const s = this.run?.steps[Math.max(0, this.token)];
    if (!s) return;
    const labels = s.contextIds.map((id) => visibleChar(this.run.vocab[id]));
    this.attention.forEach((chart, h) => {
      chart.setText(`Attention, blocco ${this.block}, head ${h}`);
      chart.setData({ rows: s.T, cols: s.T, values: s.attention[this.block][h], rowLabels: labels, colLabels: labels, min: 0, max: 1 });
    });
    const values = s.ffnPre[this.block];
    let off = 0;
    for (let i = 0; i < values.length; i++) if (values[i] <= 0) off++;
    this.ffn.setText(`Attivazioni del feed-forward, blocco ${this.block} (prima della ReLU)`);
    this.ffn.setData({ values, bins: 48 });
    this.ffn.setHighlight({ x: 0, label: `ReLU: ${Math.round((off / values.length) * 100)}% azzerati` });
  }
}
