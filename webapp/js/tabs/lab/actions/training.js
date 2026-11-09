/**
 * I dati veri delle due azioni di training (logica pura: niente DOM, niente three.js; testata).
 *
 * "Un passo di training": su un esempio vero (il prompt dello studente fino al token scelto e il
 * carattere che nel prompt viene dopo) il motore calcola la loss; il gradiente di lm_head si ottiene in
 * forma chiusa (math.js, verificato con le differenze finite) e si applica con il learning rate vero
 * del training (trace.json); poi il motore ricalcola la loss con i pesi aggiornati, su una COPIA privata
 * del motore (i motori dello stato non si toccano: lab.md).
 *
 * "I pesi durante il training": la curva della loss vera (train per step, validation ogni 100 step) e
 * le metriche degli snapshot, per la lavagna.
 */
import { GPTEngine } from "../../../engine/gpt.js";
import { adamwFirstStep, crossEntropy, layerNormSteps, lmHeadGradient, lmHeadInputGradient, sgdStep, updateStats } from "./math.js";
import { trainingExample } from "./settings.js";
import { OUTPUT_NAMES, rowOf } from "./data.js";

/** Il learning rate del training registrato: metrics.lr allo step più vicino, o train_config.learning_rate. */
export function learningRateAt(trace, step = null) {
  const m = trace?.metrics;
  if (m?.steps?.length && m?.lr?.length === m.steps.length) {
    let k = m.steps.length - 1;
    if (step !== null && step !== undefined) {
      k = 0;
      for (let i = 0; i < m.steps.length; i++) if (Math.abs(m.steps[i] - step) < Math.abs(m.steps[k] - step)) k = i;
    }
    const lr = Number(m.lr[k]);
    if (lr > 0) return { lr, source: `trace.json, metrics.lr allo step ${m.steps[k]}` };
  }
  const lr = Number(trace?.train_config?.learning_rate);
  if (lr > 0) return { lr, source: "trace.json, train_config.learning_rate" };
  return null;
}

/** Una copia del motore con gli stessi pesi (setWeights su questa non tocca i motori in cache dello stato). */
export function privateEngine(engine) {
  return GPTEngine.fromBuffers(engine.manifest, engine.weights);
}

/** I pesi del motore con lm_head sostituita (W e b nuovi), nel layout di weights.f32.bin. */
export function withLmHead(engine, W, b) {
  const flat = Float32Array.from(engine.weights);
  const w = engine.paramViews.get(OUTPUT_NAMES.lmHead.weight);
  const bias = engine.paramViews.get(OUTPUT_NAMES.lmHead.bias);
  flat.set(W, w.offset);
  flat.set(b, bias.offset);
  return flat;
}

/** La loss (e p del carattere giusto) del motore sull'ultima posizione del contesto. */
export function engineLoss(engine, context, target) {
  const V = engine.config.vocab_size;
  const { logits } = engine.forward(context);
  const ce = crossEntropy(rowOf(logits, context.length - 1, V), target);
  return { loss: ce.loss, p: ce.p, probs: ce.probs };
}

/**
 * Tutto "Un passo di training" con i numeri veri.
 * @param {import("../../../engine/gpt.js").GPTEngine} engine motore del run A allo step scelto (solo letto)
 * @param {number[]} ids il prompt in token
 * @param {{token?: number|null, lr: number, scratch?: GPTEngine}} o `scratch`: motore privato da riusare
 */
export function trainingStepData(engine, ids, { token = null, lr, scratch = null }) {
  const example = trainingExample(ids, token);
  if (!example) return null;
  const { context, target, position } = example;
  const cfg = engine.config;
  const V = cfg.vocab_size;
  const C = cfg.n_embd;
  const { trace } = engine.forward(context, { trace: true });
  const last = context.length - 1;
  const h = Float32Array.from(rowOf(trace.lnF, last, C));
  const logits = Float32Array.from(rowOf(trace.logits, last, V));
  const ce = crossEntropy(logits, target);
  const grad = lmHeadGradient(ce.probs, target, h);
  const W = engine.paramViews.get(OUTPUT_NAMES.lmHead.weight).data;
  const b = engine.paramViews.get(OUTPUT_NAMES.lmHead.bias).data;
  // un passo più indietro: h = ln_f(x) = x̂ · γ + β, quindi ∂L/∂β = ∂L/∂h e ∂L/∂γ = ∂L/∂h · x̂ (verificati
  // con le differenze finite sul motore, tests/lab-actions.test.mjs)
  const dh = lmHeadInputGradient(grad.dLogits, W, C);
  const lnF = layerNormSteps(rowOf(trace.blocks.at(-1).resid2, last, C), engine.paramViews.get(OUTPUT_NAMES.lnF.weight).data, engine.paramViews.get(OUTPUT_NAMES.lnF.bias).data);
  const dGamma = new Float32Array(C);
  for (let j = 0; j < C; j++) dGamma[j] = dh[j] * lnF.normalized[j];
  const sgd = { W: sgdStep(W, grad.dW, lr), b: sgdStep(b, grad.db, lr) };
  const adamw = { W: adamwFirstStep(W, grad.dW, { lr }), b: adamwFirstStep(b, grad.db, { lr }) };
  // la loss dopo l'aggiornamento: la ricalcola il motore (copia privata) con lm_head nuova
  const copy = scratch ?? privateEngine(engine);
  const base = Float32Array.from(engine.weights);
  copy.setWeights(withLmHead(engine, sgd.W, sgd.b));
  const afterSgd = engineLoss(copy, context, target);
  copy.setWeights(withLmHead(engine, adamw.W, adamw.b));
  const afterAdamw = engineLoss(copy, context, target);
  copy.setWeights(base);
  return {
    context,
    target,
    position,
    chars: context.map((id) => engine.vocab[id]),
    targetChar: engine.vocab[target],
    h,
    logits,
    probs: ce.probs,
    loss: ce.loss,
    p: ce.p,
    grad,
    dh,
    lnF: { normalized: lnF.normalized, dGamma, dBeta: Float32Array.from(dh) },
    W,
    b,
    lr,
    sgd: { ...sgd, ...afterSgd, statsW: updateStats(W, sgd.W), statsB: updateStats(b, sgd.b) },
    adamw: { ...adamw, ...afterAdamw, statsW: updateStats(W, adamw.W), statsB: updateStats(b, adamw.b) },
    V,
    C,
  };
}

// ---------------------------------------------------------------- storia del training

/**
 * Le curve della lavagna di "I pesi durante il training": la loss del batch di training (una per step,
 * sfoltita a ~`points` punti con la media di ogni gruppo: il batch singolo è rumoroso), la loss di
 * validation (evals) e le metriche degli snapshot. Tutto da trace.json, niente inventato.
 */
export function historySeries(trace, { points = 160 } = {}) {
  const raw = Array.isArray(trace?.train_loss) ? trace.train_loss : [];
  const group = Math.max(1, Math.ceil(raw.length / points));
  const train = [];
  for (let s = 0; s < raw.length; s += group) {
    let sum = 0;
    let n = 0;
    for (let k = s; k < Math.min(raw.length, s + group); k++) {
      if (Number.isFinite(raw[k])) {
        sum += raw[k];
        n++;
      }
    }
    if (n) train.push([s + (Math.min(raw.length, s + group) - 1 - s) / 2, sum / n]);
  }
  const ev = trace?.evals;
  const val = ev?.steps?.length ? ev.steps.map((s, i) => [s, ev.val?.[i]]).filter(([, v]) => Number.isFinite(v)) : [];
  const snapshots = (trace?.snapshots ?? []).map((s) => ({ step: s.step, loss: s.loss, top1: s.top1, top5: s.top5, entropy: s.entropy, text: typeof s.sample_text === "string" ? s.sample_text : "" }));
  const all = [...train.map((p) => p[1]), ...val.map((p) => p[1])];
  const maxStep = Math.max(raw.length - 1, ...val.map((p) => p[0]), ...snapshots.map((s) => s.step), 1);
  return { train, val, snapshots, yMin: all.length ? Math.min(...all) : 0, yMax: all.length ? Math.max(...all) : 1, maxStep, group };
}

/** La loss di validation stimata più vicina a uno step (evals ogni eval_interval step). */
export function valLossNear(trace, step) {
  const ev = trace?.evals;
  if (!ev?.steps?.length) return null;
  let k = 0;
  for (let i = 0; i < ev.steps.length; i++) if (Math.abs(ev.steps[i] - step) < Math.abs(ev.steps[k] - step)) k = i;
  return { step: ev.steps[k], val: ev.val?.[k] ?? null, train: ev.train?.[k] ?? null };
}
