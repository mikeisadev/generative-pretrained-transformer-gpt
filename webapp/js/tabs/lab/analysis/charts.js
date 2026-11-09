/**
 * I grafici della fase 2 nel pannello "Grafici" del nucleo (ChartsPanel): la storia vera del training di
 * A e di B (loss train e validation, loss del batch, learning rate, norma del gradiente, update ratio),
 * per gruppo di pesi (update ratio e deviazione standard), le metriche degli snapshot (loss, top-1,
 * top-5, entropia, entropia dell'attention per head, il carattere giusto della frase d'esempio), il
 * prossimo carattere sul prompt dello studente e il benchmark (loss per snapshot, sorpresa lungo
 * l'estratto). Le serie vengono da series.js (puro); qui solo il collegamento con lo stato.
 *
 * Il pannello del nucleo ridisegna solo da aperto e una volta per frame: aggiornare i dati costa poco.
 * Nomi delle serie fissi ("A · train"): i colori del nucleo si legano al nome, quindi restano gli stessi
 * cambiando i run (A blu, B arancione quando le serie sono due).
 */
import { prefersReducedMotion } from "../../../core/dom.js";
import { isolate } from "../runs/describe.js";
import { CHARTS, EXAMPLE_CONTINUATIONS } from "./content.js";
import { fixed, int, num, pct } from "./format.js";
import { barData, logitPairs, sharedIds } from "./probe.js";
import {
  abSeries,
  attentionEntropy,
  batchLossSeries,
  exampleProbSeries,
  finalStep,
  groupFamilies,
  groupMetricSeries,
  isConstant,
  lossSeries,
  metricSeries,
  missingAttentionText,
  movingAverage,
  pointAt,
  snapshotSeries,
} from "./series.js";

const SURPRISAL_WINDOW = 64;
const SECTIONS = { training: "lab-an-training", groups: "lab-an-groups", snapshots: "lab-an-snapshots", attention: "lab-an-attention", prompt: "lab-an-prompt", bench: "lab-an-bench" };
const ATTENTION_TITLE = "Entropia dell'attention di ogni head";
const ATTENTION_DESC = "Misurata a ogni snapshot sullo stesso batch fisso, quando il modello espone le probabilità di attention (il metodo attention_weights, come Head in gpt/attention.py).";
const HEAT_DESC = "Una riga per head (b1h2 = blocco 1, head 2), una colonna per snapshot. Chiaro = l'head guarda molti token insieme, scuro = si concentra su pochi.";
const percentPoints = (points) => points.map(([x, y]) => [x, y * 100]);
const pctFormat = (v) => `${v.toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`;

export function installAnalysisCharts(tab, { bench }) {
  const { state } = tab;
  const charts = tab.ui.charts;

  // ---------------------------------------------------------------- sezioni e grafici fissi
  const training = charts.section(SECTIONS.training, CHARTS.trainingTitle, CHARTS.trainingDesc);
  // le descrizioni con i numeri del training (ogni quanti step, quanti batch) le scrive drawTraining dal trace
  const loss = training.chart("loss", { type: "line", title: "Loss: train e validation", description: "", xLabel: "step", yLabel: "loss (nat)" });
  const batch = training.chart("batch", { type: "line", title: "Loss del batch, step per step", description: "", xLabel: "step", yLabel: "loss (nat)" });
  const lr = training.chart("lr", { type: "line", title: "Learning rate", description: "Quanto è lungo ogni passo dell'ottimizzatore (AdamW).", xLabel: "step", yLabel: "learning rate" });
  const grad = training.chart("grad", { type: "line", title: "Norma del gradiente", description: "Quanto spingono, tutti insieme, i gradienti di tutti i pesi in uno step: picchi improvvisi = un batch che sorprende il modello.", xLabel: "step", yLabel: "‖gradiente‖" });
  const UPDATE_DESC = "Di quanto cambiano tutti i pesi in uno step, in proporzione alla loro grandezza (‖Δw‖ / ‖w‖): un picco = un passo che sposta molto i pesi, verso zero = pesi che non si muovono quasi più.";
  const update = training.chart("update", { type: "line", title: "Update ratio ‖Δw‖ / ‖w‖", description: UPDATE_DESC, xLabel: "step", yLabel: "‖Δw‖ / ‖w‖" });
  const groups = charts.section(SECTIONS.groups, CHARTS.groupsTitle, CHARTS.groupsDesc);
  const snaps = charts.section(SECTIONS.snapshots, CHARTS.snapsTitle, CHARTS.snapsDesc);
  const snapLoss = snaps.chart("loss", { type: "line", title: "Loss sul batch fisso di validation", description: "Ogni punto è uno snapshot: gli stessi pesi che il cursore «Step del training» mette nella scena.", xLabel: "step", yLabel: "loss (nat)" });
  const snapAcc = snaps.chart("acc", { type: "line", title: "Accuratezza: top-1 e top-5", description: "Quante volte il carattere giusto è il primo (top-1) o tra i primi 5 (top-5) della distribuzione prevista.", xLabel: "step", yLabel: "% giuste", format: pctFormat });
  const snapEntropy = snaps.chart("entropy", { type: "line", title: "Entropia della previsione", description: "Quanto il modello è indeciso sul prossimo carattere (nat): all'inizio vicino a ln 65 = 4,17 (tira a caso), poi scende.", xLabel: "step", yLabel: "entropia (nat)" });
  const example = snaps.chart("example", { type: "line", title: "Il carattere giusto della frase d'esempio", description: "", xLabel: "step", yLabel: "probabilità", percent: true });
  // Le heatmap in una sezione loro, ricostruita quando cambia quali ci sono (drawAttention): dove il
  // trace non ha l'entropia, al posto di una heatmap vuota la descrizione della sezione dice perché.
  const attention = charts.section(SECTIONS.attention, ATTENTION_TITLE, ATTENTION_DESC);
  const prompt = charts.section(SECTIONS.prompt, CHARTS.promptTitle, CHARTS.promptDesc);
  const probsA = prompt.chart("probs-a", { type: "bar", title: "Probabilità del prossimo carattere · A", description: "", percent: true, xLabel: "carattere", yLabel: "probabilità", highlightLabel: "il più probabile" });
  const probsB = prompt.chart("probs-b", { type: "bar", title: "Probabilità del prossimo carattere · B", description: "Gli stessi caratteri del grafico di A, nello stesso ordine: le barre si confrontano a colpo d'occhio.", percent: true, xLabel: "carattere", yLabel: "probabilità", highlightLabel: "il più probabile" });
  const logits = prompt.chart("logits", { type: "scatter", title: "Logits: A (asse x) contro B (asse y)", description: "Un punto per ogni carattere del vocabolario: i punteggi grezzi di lm_head prima della softmax. Sulla diagonale i due modelli sono d'accordo.", xLabel: "logit di A", yLabel: "logit di B" });
  const benchSection = charts.section(SECTIONS.bench, CHARTS.benchTitle, CHARTS.benchDesc);
  const benchVal = benchSection.chart("val", { type: "line", title: "Loss sulla validation completa, snapshot per snapshot", description: "Dai benchmark.json del Python: tutte le previsioni della validation per ogni snapshot (protocollo finestre-contigue-v1).", xLabel: "step", yLabel: "loss (nat)" });
  const surprisal = benchSection.chart("surprisal", { type: "line", title: `Sorpresa lungo l'estratto (media mobile di ${SURPRISAL_WINDOW} caratteri)`, description: "I bit di sorpresa −log₂ p(carattere giusto) lungo val_excerpt.txt: dove la curva sale il modello fa fatica (nomi propri, inizi di battuta).", xLabel: "carattere dell'estratto", yLabel: "bit" });

  // I colori del nucleo si legano al nome della serie alla prima comparsa: una prima volta senza punti,
  // nell'ordine giusto, fissa A = primo colore (blu) e B = secondo (arancione) anche se arriva prima B.
  const prime = (chart, names) => chart.setData({ series: names.map((name) => ({ name, points: [] })) });
  prime(loss, ["A · train", "A · validation", "B · train", "B · validation"]);
  prime(snapAcc, ["A · top-1", "A · top-5", "B · top-1", "B · top-5"]);
  for (const c of [batch, lr, grad, update, snapLoss, snapEntropy, example, benchVal, surprisal]) prime(c, ["A", "B"]);

  // ---------------------------------------------------------------- dati dai trace
  const traceOf = (id) => (id ? state.run(id)?.trace ?? null : null);
  // il nome viene da run.json e finisce dentro frasi: isolato, così un U+202E non rovescia il resto
  const nameOf = (id) => isolate(state.entry(id)?.name ?? "");
  /** A e B sono due momenti dello stesso training: una storia sola, B è un punto su quella curva. */
  const sameRun = () => Boolean(state.runB) && state.runB === state.runA;
  /** Lo step di B sull'asse x dei grafici (i pesi finali sono l'ultimo step del training). */
  const stepXB = () => state.stepB ?? finalStep(traceOf(state.runB));
  const who = () => {
    if (sameRun()) return `A e B sono due momenti dello stesso training (${nameOf(state.runA)}): una curva sola, B è il punto allo step ${int(stepXB() ?? 0)}`;
    return [state.runA ? `A = ${nameOf(state.runA)}` : "", state.runB ? `B = ${nameOf(state.runB)}` : ""].filter(Boolean).join(" · ");
  };
  /** Le serie di B: le sue, oppure (stesso run) il punto della curva di A allo step di B. */
  const forB = (seriesA, seriesB) => {
    if (!sameRun()) return seriesB;
    const p = pointAt(seriesA, stepXB());
    return p ? [p] : [];
  };

  /** Un intero positivo di un oggetto del trace (dati non fidati), o null. */
  const positive = (obj, key) => (obj && typeof obj === "object" && Number.isSafeInteger(obj[key]) && obj[key] > 0 ? obj[key] : null);

  function drawTraining() {
    const ta = traceOf(state.runA);
    const tb = traceOf(state.runB);
    const any = ta || tb;
    charts.section(SECTIONS.training, CHARTS.trainingTitle, any ? `${CHARTS.trainingDesc} ${who()}.` : CHARTS.noTrace);
    // i numeri delle descrizioni vengono dal trace (train_config, recorder), non da quello del corso
    const tc = (ta ?? tb)?.train_config;
    const every = positive(tc, "eval_interval");
    const iters = positive(tc, "eval_iters");
    const size = positive(tc, "batch_size");
    training.chart("loss", { type: "line", title: "Loss: train e validation", description: `Le stime di estimate_loss${every ? ` ogni ${int(every)} step` : ""}${iters ? ` (media su ${int(iters)} batch)` : ""}: se la validation si stacca dal train verso l'alto, il modello sta imparando a memoria.` });
    training.chart("batch", { type: "line", title: "Loss del batch, step per step", description: `La loss del batch${size ? ` di ${int(size)} sequenze` : ""} su cui si fa ogni passo di ottimizzazione: rumorosa perché ogni batch è diverso.` });
    const la = lossSeries(ta);
    const lb = lossSeries(tb);
    loss.setData({ series: [...abSeries(la.train, [], "train"), ...abSeries(la.val, [], "validation"), ...abSeries([], forB(la.train, lb.train), "train"), ...abSeries([], forB(la.val, lb.val), "validation")] });
    const ab = (fn) => {
      const a = fn(ta);
      return abSeries(a, forB(a, fn(tb)));
    };
    batch.setData({ series: ab(batchLossSeries) });
    const lrA = metricSeries(ta, "lr");
    const lrB = metricSeries(tb, "lr");
    lr.setData({ series: abSeries(lrA, forB(lrA, lrB)) });
    const constant = [lrA, lrB].filter((s) => s.length).every(isConstant);
    training.chart("lr", { type: "line", title: "Learning rate", description: constant && (lrA.length || lrB.length) ? "Quanto è lungo ogni passo dell'ottimizzatore (AdamW). Qui è costante: il training non usa uno scheduler che lo cambia nel tempo." : "Quanto è lungo ogni passo dell'ottimizzatore (AdamW)." });
    grad.setData({ series: ab((t) => metricSeries(t, "grad_norm")) });
    const updA = metricSeries(ta, "update_ratio", { positive: true });
    update.setData({ series: abSeries(updA, forB(updA, metricSeries(tb, "update_ratio", { positive: true }))) });
    // l'intervallo vero di A, letto dai dati (nessuna soglia "giusta" presa da fuori)
    // (un ciclo e non Math.min(...): un trace enorme farebbe traboccare lo stack degli argomenti)
    let lo = Infinity;
    let hi = -Infinity;
    for (const [, y] of updA) {
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
    training.chart("update", { type: "line", title: "Update ratio ‖Δw‖ / ‖w‖", description: updA.length ? `${UPDATE_DESC} Nel run A va da ${num(lo, 2)} a ${num(hi, 2)}.` : UPDATE_DESC });
    highlight();
  }

  /** La linea verticale "siamo qui": lo step di A (i pesi finali sono l'ultimo step del training). */
  function highlight() {
    const ta = traceOf(state.runA);
    const x = state.step ?? finalStep(ta);
    const h = Number.isFinite(x) ? { x } : null;
    for (const c of [loss, batch, lr, grad, update, snapLoss, snapAcc, snapEntropy, example, benchVal]) c.setHighlight(h);
    for (const c of groupCharts) c.setHighlight(h);
  }

  let groupCharts = [];
  function drawGroups() {
    // i gruppi del run A se ha la storia, altrimenti quelli di B
    const id = traceOf(state.runA) ? state.runA : traceOf(state.runB) ? state.runB : null;
    const trace = traceOf(id);
    const families = groupFamilies(trace);
    groups.clear();
    groupCharts = [];
    const side = id === state.runA ? "A" : "B";
    charts.section(SECTIONS.groups, `${CHARTS.groupsTitle}${id ? ` · ${side}` : ""}`, families.length ? `${CHARTS.groupsDesc} Run ${side}: ${nameOf(id)}.` : CHARTS.noTrace);
    for (const f of families) {
      const extra = f.truncated ? ` (${int(f.truncated)} gruppi in mezzo non disegnati: al più 8 linee leggibili)` : "";
      const logEvery = positive(trace?.recorder, "log_every");
      const up = groups.chart(`upd-${f.id}`, { type: "line", title: `Update ratio · ${f.label}`, description: `‖Δw‖ / ‖w‖ del gruppo a ogni misura${logEvery ? ` (ogni ${int(logEvery)} step)` : ""}${extra}.`, xLabel: "step", yLabel: "‖Δw‖ / ‖w‖", height: 170 });
      up.setData({ series: f.members.map((m) => ({ name: m.label, points: groupMetricSeries(trace, "update_ratio_by_group", m.group, { positive: true }) })).filter((s) => s.points.length) });
      const std = groups.chart(`std-${f.id}`, { type: "line", title: `Deviazione standard dei pesi · ${f.label}`, description: `Quanto sono sparsi i pesi del gruppo: allo step 0 è quella dell'inizializzazione casuale, poi la cambia il training${extra}.`, xLabel: "step", yLabel: "dev. std", height: 170 });
      std.setData({ series: f.members.map((m) => ({ name: m.label, points: groupMetricSeries(trace, "weight_std_by_group", m.group) })).filter((s) => s.points.length) });
      groupCharts.push(up, std);
    }
  }

  function drawSnapshots() {
    const ta = traceOf(state.runA);
    const tb = traceOf(state.runB);
    const snap = (key, scale = (p) => p) => {
      const a = scale(snapshotSeries(ta, key));
      return [a, forB(a, scale(snapshotSeries(tb, key)))];
    };
    snapLoss.setData({ series: abSeries(...snap("loss")) });
    const [top1A, top1B] = snap("top1", percentPoints);
    const [top5A, top5B] = snap("top5", percentPoints);
    snapAcc.setData({ series: [...abSeries(top1A, [], "top-1"), ...abSeries(top5A, [], "top-5"), ...abSeries([], top1B, "top-1"), ...abSeries([], top5B, "top-5")] });
    snapEntropy.setData({ series: abSeries(...snap("entropy")) });
    drawExample(ta, tb);
    drawAttention(ta, tb);
  }

  /** La frase d'esempio del training: la probabilità del carattere che viene davvero dopo (se lo sappiamo). */
  function drawExample(ta, tb) {
    const prompt = ta?.recorder?.example_prompt ?? tb?.recorder?.example_prompt;
    const known = typeof prompt === "string" && Object.hasOwn(EXAMPLE_CONTINUATIONS, prompt) ? EXAMPLE_CONTINUATIONS[prompt] : null;
    const vocabOf = (id) => state.run(id)?.vocab ?? null;
    const idOf = (id) => (known ? vocabOf(id)?.indexOf(known.next) ?? -1 : -1);
    const pa = ta?.recorder?.example_prompt === prompt ? exampleProbSeries(ta, idOf(state.runA)) : [];
    const pb = forB(pa, tb?.recorder?.example_prompt === prompt ? exampleProbSeries(tb, idOf(state.runB)) : []);
    if (!known) {
      snaps.chart("example", { type: "line", title: "Il carattere giusto della frase d'esempio", description: "Il trace non ha la frase d'esempio del corso: niente da mostrare." });
      example.setData({ series: [] });
      return;
    }
    // il carattere preferito dal modello all'ultimo snapshot di A, letto dai dati (non sempre è quello giusto)
    const last = ta?.snapshots?.at(-1)?.example_next_token_probs;
    const vocab = vocabOf(state.runA);
    let favorite = "";
    if (Array.isArray(last) && vocab) {
      let best = 0;
      for (let i = 1; i < last.length; i++) if (last[i] > last[best]) best = i;
      if (vocab[best] !== known.next) favorite = ` All'ultimo snapshot A preferisce «${vocab[best]}» (${pct(last[best])}) alla «${known.next}» (${pct(last[vocab.indexOf(known.next)])}).`;
    }
    snaps.chart("example", { type: "line", title: `p(«${known.next}») dopo «…${prompt.slice(-14).replace(/\n/g, "↵")}»`, description: `${known.note} Le probabilità le ha registrate il training stesso, snapshot per snapshot.${favorite}` });
    example.setData({ series: abSeries(pa, pb) });
  }

  // ---------------------------------------------------------------- entropia dell'attention
  let attentionCards = ""; // quali heatmap ha la sezione ora: "AB", "A", "B" o ""

  /** Le heatmap di A e di B che ci sono; per chi non ce l'ha, il perché nella descrizione della sezione. */
  function drawAttention(ta, tb) {
    const same = sameRun(); // stesso run: la colonna di B è nella heatmap di A
    const ha = attentionEntropy(ta);
    const hb = same ? null : attentionEntropy(tb);
    const cards = `${ha ? "A" : ""}${hb ? "B" : ""}`;
    if (cards !== attentionCards) {
      attention.clear(); // l'unico modo di togliere una card: poi si ricreano quelle che servono
      attentionCards = cards;
    }
    if (ha) heat("att-a", "A", HEAT_DESC, ha);
    if (hb) heat("att-b", "B", ha ? "Come sopra, per il run B." : HEAT_DESC, hb);
    charts.section(SECTIONS.attention, ATTENTION_TITLE, attentionText(ta, tb, ha, hb, same));
  }

  function heat(id, side, description, h) {
    const chart = attention.chart(id, { type: "heatmap", title: `${ATTENTION_TITLE} · ${side}`, description, xLabel: "step", yLabel: "head" });
    chart.setData({ rows: h.rows, cols: h.cols, values: h.values, rowLabels: h.rowLabels, colLabels: h.colLabels });
  }

  /** La descrizione della sezione: che cosa misura e, per A e per B senza heatmap, perché manca. */
  function attentionText(ta, tb, ha, hb, same) {
    if (!ta && !tb) return CHARTS.noTrace;
    const parts = [ATTENTION_DESC];
    const why = (side, id, trace) => `Nessuna heatmap per ${side} (${nameOf(id)}): ${missingAttentionText(trace)}`;
    if (!ta) parts.push("A non ha la storia del training (trace.json).");
    else if (!ha) parts.push(why("A", state.runA, ta));
    if (same) parts.push(ha ? "B è un momento dello stesso training di A: la sua colonna è nella heatmap di A." : "B è un momento dello stesso training di A: vale lo stesso.");
    else if (state.runB && !tb) parts.push("B non ha la storia del training (trace.json).");
    else if (tb && !hb) parts.push(why("B", state.runB, tb));
    return parts.join(" ");
  }

  // ---------------------------------------------------------------- benchmark
  let benchToken = 0;
  async function drawBench() {
    const my = ++benchToken;
    const [ba, bb] = await Promise.all([bench.of(state.runA).catch(() => null), bench.of(state.runB).catch(() => null)]);
    if (my !== benchToken) return;
    const points = (b) => (b?.snapshots ?? []).filter((s) => Number.isFinite(s.metrics.loss)).map((s) => [s.step, s.metrics.loss]);
    const pa = points(ba);
    benchVal.setData({ series: abSeries(pa, forB(pa, points(bb))) });
    const ref = ba?.reference?.uniformLoss;
    benchSection.chart("val", { type: "line", title: "Loss sulla validation completa, snapshot per snapshot", description: `Dai benchmark.json del Python: tutte le previsioni della validation per ogni snapshot (protocollo finestre-contigue-v1).${Number.isFinite(ref) ? ` Tirare a caso darebbe ${fixed(ref, 3)}.` : ""}` });
  }

  // ---------------------------------------------------------------- aggiornamenti
  const drawAll = () => {
    drawTraining();
    drawGroups();
    drawSnapshots();
    void drawBench();
  };
  const offs = [
    state.on("runA", drawAll),
    state.on("runB", drawAll),
    state.on("step", () => highlight()),
    // stesso run: B è un punto sulle curve di A, che si sposta con il suo step
    state.on("stepB", () => {
      if (!sameRun()) return;
      drawTraining();
      drawSnapshots();
      void drawBench();
    }),
    state.on("runs", ({ reason }) => (reason === "load" ? drawAll() : null)),
  ];
  drawAll();

  return {
    /** Il prossimo carattere sul prompt: barre per A e per B sugli stessi caratteri, scatter dei logits. */
    setProbe(a, b, { prompt: text = "" } = {}) {
      const ids = sharedIds(a, b, 8);
      const shown = text.length > 24 ? `…${text.slice(-24)}` : text;
      const where = `dopo «${shown.replace(/\n/g, "↵")}»`;
      prompt.chart("probs-a", { type: "bar", title: `Probabilità del prossimo carattere · A`, description: a.ok ? `${nameOf(state.runA)}, ${where}. I caratteri più probabili per A o per B.` : a.reason || "" });
      probsA.setData(barData(a, ids, "probs"));
      probsB.setData(barData(b, ids, "probs"));
      if (!b.ok && b.reason) prompt.chart("probs-b", { type: "bar", title: "Probabilità del prossimo carattere · B", description: b.reason });
      logits.setData({ points: logitPairs(a, b) });
    },
    /** La sorpresa carattere per carattere sull'estratto (Python o browser), per A e B. */
    setSurprisal({ a = null, b = null } = {}) {
      surprisal.setData({ series: abSeries(a?.bits?.length ? movingAverage(a.bits, SURPRISAL_WINDOW) : [], b?.bits?.length ? movingAverage(b.bits, SURPRISAL_WINDOW) : []) });
      const from = [a?.bits?.length ? `A: ${a.label}` : "", b?.bits?.length ? `B: ${b.label}` : ""].filter(Boolean).join(" · ");
      benchSection.chart("surprisal", { type: "line", title: `Sorpresa lungo l'estratto (media mobile di ${SURPRISAL_WINDOW} caratteri)`, description: `I bit di sorpresa −log₂ p(carattere giusto) lungo val_excerpt.txt: dove la curva sale il modello fa fatica.${from ? ` ${from}.` : " Compaiono quando c'è il calcolo del Python (estratto di benchmark.json) o quello del browser."}` });
    },
    /** Apre il pannello dei grafici su una sezione ("training", "groups", "snapshots", "prompt", "bench"). */
    reveal(id = "training") {
      charts.open();
      const section = charts.sections?.get(SECTIONS[id] ?? id);
      // dopo l'apertura (il pannello scorre dentro con una transizione)
      requestAnimationFrame(() => section?.root?.scrollIntoView?.({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" }));
    },
    redraw: drawAll,
    dispose() {
      for (const off of offs) off();
      benchToken++;
      for (const id of Object.values(SECTIONS)) charts.sections?.get(id)?.clear();
    },
  };
}
