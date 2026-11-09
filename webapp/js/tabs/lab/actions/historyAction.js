/**
 * Azione «I pesi durante il training»: gli snapshot veri del run A, dallo step 0 (numeri casuali) ai
 * pesi finali, SULLA SCENA (lo step del training del tab si sposta davvero: scena, cursore, tabella e
 * targhetta cambiano insieme; a fine azione torna quello di prima). La lavagna mostra la curva vera della
 * loss con "siamo qui", le metriche dello snapshot e il testo che il modello generava a quello step.
 * Funziona con ogni run che ha gli snapshot, anche se il motore JS non lo esegue: sono solo pesi.
 */
import { codeFor } from "./code.js";
import { int, num, pct } from "./format.js";
import { padRect } from "./plan.js";
import { historySeries, valLossNear } from "./training.js";

/** I passi; prima di tutto legge tutti gli snapshot (così durante l'azione ogni cambio di step è immediato). */
export async function historySteps(stage, tab, settings = null) {
  const { state } = tab;
  const runId = state.runA;
  const run = await state.loadRun(runId);
  const steps = state.stepsOf(runId);
  if (steps.length < 2 || !run.trace) {
    return [
      {
        title: "Questo run non ha la storia del training",
        caption: `«${run.name}» ha solo i pesi finali: per vedere i pesi nascere dal rumore servono gli snapshot (trace.json + snap_XXXXX.f16.bin).`,
        duration: 8,
        board: () => [
          { title: "Servono gli snapshot", kicker: "Storia del training" },
          { line: ["Registra il training con scripts/record_training.py --out <cartella> --no-export, poi esporta il run con scripts/export_run.py … --trace <cartella> (pannello Run, «Come esportare un run»)."] },
          { note: "Il GPT del corso ha 11 snapshot veri: scegli il GPT del corso nel pannello Run per vedere questa azione." },
        ],
      },
    ];
  }
  // tutti gli snapshot in memoria prima di partire (40-230 ms ciascuno la prima volta): poi ogni passo è immediato
  await Promise.all(steps.map((s) => state.weightsAt(runId, s)));
  const series = historySeries(run.trace);
  const bySnapshot = new Map(series.snapshots.map((s) => [s.step, s]));
  const finalStep = steps.at(-1);
  // dove guardare: la head scelta del primo blocco (key, query, value e tril), o tutto il modello se non c'è
  const chosen = settings?.clamp(run.config) ?? { block: 0, head: 0 };
  const head = `blocks.${chosen.block}.sa.heads.${chosen.head}`;
  const box = stage.scene.nodeOf(head) ?? stage.scene.nodeOf("blocks.0") ?? stage.scene.layout?.root;
  const rect = box ? padRect(box, 0.08) : null;
  const chartOf = (step) => ({
    series: [
      { name: "train (media del batch)", points: series.train, color: "#38bdf8", width: 1.6, opacity: 0.85 },
      { name: "validation", points: series.val, color: "#fbbf24", width: 2.2 },
    ],
    marker: step,
    xMax: series.maxStep,
    yMin: Math.max(0, Math.floor(series.yMin * 10) / 10),
    yMax: Math.ceil(series.yMax * 10) / 10,
    xLabel: `step (fino a ${int(series.maxStep)})`,
    yLabel: "loss",
    label: `Loss del training per step, con lo step ${int(step)} evidenziato`,
    height: 96,
  });
  const goTo = (step) => {
    stage.touchedStep = true;
    stage.ownStepChange = true;
    const pending = state.setStep(step);
    stage.ownStepChange = false;
    return pending;
  };
  const describe = (step) => {
    const snap = bySnapshot.get(step);
    const val = valLossNear(run.trace, step);
    const lines = [];
    if (snap && Number.isFinite(snap.loss)) lines.push({ line: ["Loss sul batch fisso di validation: ", { text: num(snap.loss, 3), tone: "accent" }, Number.isFinite(snap.top1) ? ` · top-1 ${pct(snap.top1)} · top-5 ${pct(snap.top5)}` : ""] });
    else if (val?.val !== null && val?.val !== undefined) lines.push({ line: [`Loss di validation stimata allo step ${int(val.step)}: `, { text: num(val.val, 3), tone: "accent" }] });
    if (snap?.text) {
      const shown = snap.text.replace(/\s*\n\s*/g, " ↵ ").slice(0, 90);
      lines.push({ formula: ["Testo generato: ", { text: `«${shown}${snap.text.length > 90 ? "…" : ""}»`, tone: "code" }] });
    }
    return lines;
  };
  const caption = (step, i) => {
    const snap = bySnapshot.get(step);
    if (i === 0) return `Step 0: i pesi appena creati, numeri casuali (inizializzazione di PyTorch). Il modello tira a indovinare: loss ${num(snap?.loss ?? NaN, 2)}, vicina a ln ${run.config?.vocab_size ?? 65} = ${num(Math.log(run.config?.vocab_size ?? 65), 2)}.`;
    if (step === finalStep) return `Step ${int(step)}: l'ultimo snapshot, i pesi alla fine del training (in float16). Dal rumore è nata una struttura: è tutto quello che il modello ha imparato.`;
    return `Step ${int(step)}: ${int(step)} passi di AdamW dopo l'inizio. Loss ${num(snap?.loss ?? NaN, 2)}: ${i < 3 ? "il modello impara le lettere e gli spazi più frequenti." : i < 6 ? "parole brevi e la forma del testo (nomi, a capo)." : "le parole migliorano piano: la loss scende sempre più lentamente."}`;
  };

  const specs = steps.map((step, i) => ({
    title: `Step ${int(step)}`,
    caption: caption(step, i),
    duration: i === 0 ? 5 : 3.2,
    code: codeFor("loop", `step ${int(step)}`),
    scene: () => void goTo(step),
    board: () => [{ title: `I pesi allo step ${int(step)} di ${int(series.maxStep)}`, kicker: "Storia del training" }, { chart: chartOf(step) }, ...describe(step)],
    frame: i === 0 ? () => rect : null,
  }));
  specs.push({
    title: "Il viaggio dei pesi",
    caption: "Dallo step 0 ai pesi finali: la loss è scesa da ~4,4 a ~1,8 e le matrici sono passate dal rumore uniforme a una struttura. Chiudendo l'azione lo step del training torna quello di prima.",
    duration: 6,
    code: codeFor("step", "gpt/training.py"),
    scene: () => void goTo(finalStep),
    board: () => [
      { title: "La loss vera, step per step", kicker: "Storia del training" },
      { chart: chartOf(finalStep) },
      { note: `Curva di train: la loss del batch a ogni step, in media su gruppi di ${series.group} step (un batch solo è rumoroso). Validation: la stima ogni ${int(run.trace.train_config?.eval_interval ?? 100)} step. Snapshot: ${steps.map((s) => int(s)).join(", ")}.` },
    ],
    frame: () => rect,
  });
  return specs;
}
