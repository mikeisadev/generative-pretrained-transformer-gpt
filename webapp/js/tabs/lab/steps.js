/**
 * Step del training (logica pura): il cursore "Step del training" del Laboratorio pesi.
 *
 * Un run con la storia del training ha gli snapshot veri (per il GPT del corso: 0, 50, 100, 250, 500,
 * 1000, 1500, 2000, 3000, 4000, 5000) più i pesi finali. Il cursore si AGGANCIA agli snapshot: ogni
 * posizione è un insieme di pesi davvero salvato durante il training. Solo se lo studente lo chiede
 * ("interpola") il cursore passa anche dagli step intermedi, e allora i pesi sono una media pesata dei
 * due snapshot vicini: un'approssimazione (il training vero non va in linea retta), sempre dichiarata.
 *
 *   const model = sliderModel([0, 50, 100], { interpolate: false });
 *   model.toStep(1) → 50;  model.toValue(null) → 3 (pesi finali, l'ultima posizione)
 */

/** Passo del cursore quando si interpola: abbastanza fine da vedere la transizione, non 5.000 tacche. */
export const INTERPOLATION_STEP = 10;

/** Lo snapshot più vicino a `step` tra quelli disponibili (a pari distanza il precedente); null se non ce ne sono. */
export function snapStep(steps, step) {
  if (!steps?.length || !Number.isFinite(step)) return null;
  let best = steps[0];
  for (const s of steps) if (Math.abs(s - step) < Math.abs(best - step)) best = s;
  return best;
}

/**
 * Il modello del cursore: valori interi da `min` a `max`. L'ultima posizione sono sempre i pesi finali
 * (step null); le altre gli snapshot (agganciato) oppure gli step da INTERPOLATION_STEP in
 * INTERPOLATION_STEP tra il primo e l'ultimo snapshot (interpolato).
 * @param {number[]} steps step degli snapshot disponibili, in ordine crescente
 * @param {{interpolate?: boolean}} [o]
 * @returns {{min: number, max: number, step: number, interpolate: boolean, ticks: number[],
 *   toStep: (value: number) => number|null, toValue: (step: number|null) => number}}
 */
export function sliderModel(steps, { interpolate = false } = {}) {
  const list = [...new Set((steps ?? []).filter((s) => Number.isInteger(s) && s >= 0))].sort((a, b) => a - b);
  if (!list.length) return { min: 0, max: 0, step: 1, interpolate: false, ticks: [], toStep: () => null, toValue: () => 0 };
  if (!interpolate || list.length < 2) {
    const max = list.length; // posizioni 0…n−1 = snapshot, n = pesi finali
    return {
      min: 0,
      max,
      step: 1,
      interpolate: false,
      ticks: list.map((_, i) => i),
      toStep: (value) => {
        const i = Math.round(Number(value));
        return i >= max || !Number.isFinite(i) ? null : list[Math.max(0, i)];
      },
      toValue: (step) => (step === null || step === undefined ? max : Math.max(0, list.indexOf(snapStep(list, step)))),
    };
  }
  const first = list[0];
  const last = list.at(-1);
  // Una posizione in più dopo l'ultimo snapshot: i pesi finali (come nel cursore agganciato).
  const max = last + INTERPOLATION_STEP;
  return {
    min: first,
    max,
    step: INTERPOLATION_STEP,
    interpolate: true,
    ticks: list,
    toStep: (value) => {
      const v = Number(value);
      if (!Number.isFinite(v) || v > last) return null;
      // vicino a uno snapshot (entro mezzo passo) si aggancia: lo snapshot vero è sempre raggiungibile
      const near = snapStep(list, v);
      if (Math.abs(near - v) <= INTERPOLATION_STEP / 2) return near;
      return Math.min(last, Math.max(first, Math.round(v)));
    },
    toValue: (step) => (step === null || step === undefined ? max : Math.min(last, Math.max(first, step))),
  };
}

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });

/**
 * Come si chiama una scelta del cursore, in italiano: "pesi finali", "snapshot dello step 500",
 * "step 750 (interpolato tra 500 e 1.000)". `view` è il risultato di LabState.weightsAt (se c'è dice
 * che cosa è stato caricato davvero).
 */
export function stepLabel(step, view = null) {
  if (view?.source === "interpolated") return `step ${INT.format(view.step)}, interpolato tra ${INT.format(view.from[0])} e ${INT.format(view.from[1])} (approssimato)`;
  if (step === null || step === undefined || view?.source === "final") return "pesi finali";
  return `snapshot dello step ${INT.format(view?.step ?? step)}`;
}

/** Etichetta breve per la barra dello stage e il cursore: "finali", "step 500", "≈ step 750". */
export function shortStepLabel(step, view = null) {
  if (view?.source === "interpolated") return `≈ step ${INT.format(view.step)}`;
  if (step === null || step === undefined || view?.source === "final") return "pesi finali";
  return `step ${INT.format(view?.step ?? step)}`;
}
