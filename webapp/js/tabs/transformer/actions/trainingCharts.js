/**
 * Grafici della sezione "Training": quelli che un AI engineer o un data scientist guarda davvero
 * durante un training. Tutti i numeri vengono da data/training/trace.json (training registrato).
 *
 * `fill(step)` mostra la storia fino allo step dato: l'azione "Il training completo" la chiama
 * mentre avanza; senza azioni in corso i grafici mostrano la storia completa.
 */
import { el } from "../../../core/dom.js";
import { GROUP_KINDS, metricIndex, summarizeByKind } from "./trainingData.js";

const LN_V = Math.log(65);
const UPDATE_REFERENCE = 1e-3;
/** Tensori proposti per l'istogramma dei pesi (nome PyTorch → etichetta). */
const HIST_TENSORS = [
  ["token_embedding_table.weight", "token embedding (65 × 64)"],
  ["blocks.0.sa.heads.0.query.weight", "blocco 0, head 0, query (16 × 64)"],
  ["blocks.0.ffwd.net.0.weight", "blocco 0, feed-forward net.0 (256 × 64)"],
  ["blocks.3.ffwd.net.2.weight", "blocco 3, feed-forward net.2 (64 × 256)"],
  ["blocks.0.ln1.weight", "blocco 0, LayerNorm 1 γ (64)"],
  ["lm_head.weight", "lm_head (65 × 64)"],
];

/** Etichette brevi dei 16 gruppi (la heatmap ha poco spazio a sinistra). */
function shortGroup(g) {
  const m = /^blocks\.(\d+)\.(\w+)$/.exec(g);
  if (m) return `${m[1]}${{ attention: "A", ffwd: "F", layernorm: "N" }[m[2]] ?? "?"}`;
  return { token_embedding: "tok", position_embedding: "pos", ln_f: "lnf", lm_head: "head" }[g] ?? g.slice(0, 4);
}

const upTo = (steps, values, step) => {
  const points = [];
  for (let i = 0; i < steps.length && steps[i] <= step; i++) points.push([steps[i], values[i]]);
  return points;
};

export class TrainingCharts {
  constructor(panel) {
    const s = panel.section(
      "training",
      "Training",
      "La storia vera dei 5000 step di training (trace.json). L'azione «Il training completo» la riscorre; la linea verticale indica lo step mostrato nella scena.",
    );
    this.section = s;
    this.trace = null;
    this.tensor = HIST_TENSORS[0][0];
    this.snapshotValues = null;

    const lossDescription = "A cosa serve: è IL grafico del training. Train = errore sui dati che il modello studia; validation = errore su testo mai visto (il 10% finale del file). Cosa guardare: entrambe scendono; la serie «ln 65» ≈ 4,17 è la loss attesa di un modello che tira a indovinare.";
    this.lossControls = el(
      "div",
      { className: "tfa-chart-controls" },
      el("label", {}, el("input", { type: "checkbox", onchange: (e) => this.setLogScale(e.target.checked) }), " Scala logaritmica per la loss"),
    );
    this.loss = s.chart("loss", { type: "line", title: "Loss: train e validation", description: lossDescription, xLabel: "step", yLabel: "cross-entropy (nat)" });
    this.lossLog = s.chart("loss-log", { type: "line", title: "Loss: train e validation (scala log)", description: lossDescription, xLabel: "step", yLabel: "cross-entropy (nat)", yScale: "log" });
    this.lossLog.root.hidden = true;
    this.loss.root.before(this.lossControls);
    this.perplexity = s.chart("perplexity", {
      type: "line",
      title: "Perplexity",
      description: "A cosa serve: e^loss, più intuitiva: \"tra quanti caratteri il modello è indeciso\". Cosa guardare: parte da circa 82 (peggio del caso, 65) e scende verso 5-6.",
      xLabel: "step",
      yScale: "log",
    });
    this.gap = s.chart("gap", {
      type: "line",
      title: "Gap di generalizzazione (val − train)",
      description: "A cosa serve: segnala l'overfitting. Cosa guardare: se il gap cresce mentre la train continua a scendere, il modello sta imparando a memoria il testo di training invece di generalizzare.",
      xLabel: "step",
      yLabel: "Δ loss",
    });
    this.lr = s.chart("lr", {
      type: "line",
      title: "Learning rate",
      description: "A cosa serve: quanto è grande ogni passo dell'ottimizzatore. Cosa guardare: qui è costante (0,001, nessuno scheduler, come nel notebook); nei modelli grandi di solito sale nel warm-up e poi scende.",
      xLabel: "step",
    });
    this.gradNorm = s.chart("grad-norm", {
      type: "line",
      title: "Norma del gradiente: globale e per gruppo",
      description: "A cosa serve: quanto \"spinge\" la loss su ogni parte del modello. Cosa guardare: picchi improvvisi = instabilità (si usa il gradient clipping); i gruppi dei blocchi sono riassunti come norma dell'unione dei 4 blocchi.",
      xLabel: "step",
      yScale: "log",
    });
    this.updateRatio = s.chart("update-ratio", {
      type: "line",
      title: "Rapporto update/peso ‖Δw‖/‖w‖",
      description: "A cosa serve: dice se il learning rate è sensato. Cosa guardare: la regola pratica (CS231n) è circa 1e-3, la linea grigia. Molto sotto = impara troppo piano; molto sopra = passi troppo grandi. Per i blocchi è la media dei 4.",
      xLabel: "step",
      yScale: "log",
    });
    this.updateHeat = s.chart("update-heat", {
      type: "heatmap",
      title: "Rapporto update/peso per gruppo (log10)",
      description: "A cosa serve: lo stesso rapporto, ma per ognuno dei 16 gruppi di parametri (righe: tok/pos = embedding, 0A = blocco 0 attention, 0F = feed-forward, 0N = LayerNorm, …, lnf, head = lm_head). Cosa guardare: righe più chiare = gruppi che cambiano di più.",
      xLabel: "step →",
      height: 300,
    });
    this.weightStd = s.chart("weight-std", {
      type: "line",
      title: "Deviazione standard dei pesi per gruppo",
      description: "A cosa serve: controlla che i pesi non esplodano né si spengano. Cosa guardare: variazioni lente e regolari; per i blocchi è la media dei 4.",
      xLabel: "step",
    });
    this.histSelect = el(
      "select",
      { className: "tfa-select", ariaLabel: "Tensore dell'istogramma", onchange: (e) => this.setTensor(e.target.value) },
      ...HIST_TENSORS.map(([name, label]) => el("option", { value: name }, label)),
    );
    this.hist = s.chart("weight-hist", {
      type: "histogram",
      title: "Istogramma dei pesi allo snapshot",
      description: "A cosa serve: la forma della distribuzione dei pesi di un tensore. Cosa guardare: come si allarga o si deforma rispetto all'inizializzazione (step 0).",
      xLabel: "valore del peso",
    });
    this.hist.root.before(el("div", { className: "tfa-chart-controls" }, el("label", {}, "Istogramma di: ", this.histSelect)));
    this.accuracy = s.chart("accuracy", {
      type: "line",
      title: "Accuracy top-1 e top-5 (validation)",
      description: "A cosa serve: quante volte il carattere giusto è il primo (top-1) o tra i primi 5 (top-5) della distribuzione. Misurato agli snapshot su un batch fisso di validation (64 × 32).",
      xLabel: "step",
      yMin: 0,
      yMax: 1,
      percent: true,
    });
    this.entropy = s.chart("entropy", {
      type: "line",
      title: "Entropia della distribuzione predetta",
      description: "A cosa serve: quanto il modello è sicuro delle sue previsioni. Cosa guardare: scende da ln 65 ≈ 4,17 man mano che il modello impara; se scendesse sotto la loss il modello sarebbe sicuro e sbagliato (sovraconfidente).",
      xLabel: "step",
      yLabel: "nat",
    });
    this.attEntropy = s.chart("att-entropy", {
      type: "heatmap",
      title: "Entropia dell'attention per blocco × head",
      description: "A cosa serve: una head con entropia alta guarda un po' tutto (media); bassa = si concentra su pochi token. Per un'attention uniforme su 32 posizioni la media è circa 2,55 nat. Righe b0…b3 = blocchi.",
      height: 230,
    });
  }

  /** Porta la sezione Training in vista nel pannello dei grafici. */
  reveal() {
    this.section.root.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  setLogScale(on) {
    this.loss.root.hidden = on;
    this.lossLog.root.hidden = !on;
  }

  setTensor(name) {
    this.tensor = name;
    this.drawHistogram();
  }

  /** Carica la storia (una volta) e la mostra tutta. */
  setTrace(trace) {
    this.trace = trace;
    this.kindsGrad = summarizeByKind(trace.metrics.grad_norm_by_group, "rss");
    this.kindsUpdate = summarizeByKind(trace.metrics.update_ratio_by_group, "mean");
    this.kindsStd = summarizeByKind(trace.metrics.weight_std_by_group, "mean");
    const groups = trace.groups;
    const n = trace.metrics.steps.length;
    this.updateHeat.setData({
      rows: groups.length,
      cols: n,
      values: Float32Array.from(groups.flatMap((g) => trace.metrics.update_ratio_by_group[g].map((v) => Math.log10(Math.max(v, 1e-12))))),
      rowLabels: groups.map(shortGroup),
    });
    this.fill(Infinity);
  }

  /** Mostra la storia fino a `step` (Infinity = tutta). */
  fill(step) {
    const t = this.trace;
    if (!t) return;
    const last = Math.min(step, t.train_config.max_iters);
    const trainSteps = [];
    for (let i = 0; i <= Math.min(last, t.train_loss.length - 1); i++) trainSteps.push([i, t.train_loss[i]]);
    const evalTrain = upTo(t.evals.steps, t.evals.train, last);
    const evalVal = upTo(t.evals.steps, t.evals.val, last);
    const max = t.train_config.max_iters;
    const lossSeries = [
      { name: "train (batch di ogni step)", points: trainSteps },
      { name: "train (stima su 200 batch)", points: evalTrain },
      { name: "validation", points: evalVal },
      { name: "ln 65: loss iniziale attesa", points: [[0, LN_V], [max, LN_V]] },
    ];
    this.loss.setData({ series: lossSeries });
    this.lossLog.setData({ series: lossSeries });
    this.perplexity.setData({
      series: [
        { name: "train", points: evalTrain.map(([x, y]) => [x, Math.exp(y)]) },
        { name: "validation", points: evalVal.map(([x, y]) => [x, Math.exp(y)]) },
        { name: "65: tirare a caso", points: [[0, 65], [max, 65]] },
      ],
    });
    this.gap.setData({ series: [{ name: "val − train", points: evalVal.map(([x, y], i) => [x, y - evalTrain[i][1]]) }] });
    const ms = t.metrics.steps;
    this.lr.setData({ series: [{ name: "learning rate", points: upTo(ms, t.metrics.lr, last) }] });
    this.gradNorm.setData({
      series: [{ name: "globale", points: upTo(ms, t.metrics.grad_norm, last) }, ...this.kindsGrad.map((k) => ({ name: k.label, points: upTo(ms, k.values, last) }))],
    });
    this.updateRatio.setData({
      series: [
        { name: "globale", points: upTo(ms, t.metrics.update_ratio, last) },
        ...this.kindsUpdate.map((k) => ({ name: k.label, points: upTo(ms, k.values, last) })),
        { name: "riferimento 1e-3", points: [[0, UPDATE_REFERENCE], [max, UPDATE_REFERENCE]] },
      ],
    });
    this.weightStd.setData({ series: this.kindsStd.map((k) => ({ name: k.label, points: upTo(ms, k.values, last) })) });
    const snaps = t.snapshots.filter((s) => s.step <= last);
    this.accuracy.setData({
      series: [
        { name: "top-1", points: snaps.map((s) => [s.step, s.top1]) },
        { name: "top-5", points: snaps.map((s) => [s.step, s.top5]) },
      ],
    });
    this.entropy.setData({
      series: [
        { name: "entropia predetta", points: snaps.map((s) => [s.step, s.entropy]) },
        { name: "loss sullo stesso batch", points: snaps.map((s) => [s.step, s.loss]) },
        { name: "ln 65", points: [[0, LN_V], [max, LN_V]] },
      ],
    });
    const snap = snaps.at(-1);
    if (snap) this.showAttentionEntropy(snap);
    this.highlight(Number.isFinite(step) ? last : null);
  }

  /** Linea verticale "siamo qui" su tutti i grafici con l'asse degli step. */
  highlight(step) {
    const h = step === null ? null : { x: step, label: `step ${step}` };
    for (const c of [this.loss, this.lossLog, this.perplexity, this.gap, this.lr, this.gradNorm, this.updateRatio, this.weightStd, this.accuracy, this.entropy]) c.setHighlight(h);
    if (this.trace && step !== null) this.updateHeat.setHighlight({ col: metricIndex(this.trace.metrics.steps, step) });
    else this.updateHeat.setHighlight(null);
  }

  showAttentionEntropy(snap) {
    const H = snap.attention_entropy;
    this.attEntropy.setText(`Entropia dell'attention per blocco × head (step ${snap.step})`);
    this.attEntropy.setData({
      rows: H.length,
      cols: H[0].length,
      values: Float32Array.from(H.flat()),
      rowLabels: H.map((_, l) => `b${l}`),
      colLabels: H[0].map((_, h) => `head ${h}`),
      min: 0,
      max: Math.log(32),
    });
  }

  /** Pesi (Map nome → Float32Array) dello snapshot mostrato, per l'istogramma. */
  setSnapshotWeights(step, views) {
    this.snapshotStep = step;
    this.snapshotViews = views;
    this.drawHistogram();
  }

  drawHistogram() {
    const values = this.snapshotViews?.get(this.tensor);
    if (!values) return;
    const label = HIST_TENSORS.find(([n]) => n === this.tensor)?.[1] ?? this.tensor;
    this.hist.setText(`Istogramma dei pesi: ${label}, step ${this.snapshotStep}`);
    this.hist.setData({ values, bins: 50 });
  }
}

export { GROUP_KINDS };
