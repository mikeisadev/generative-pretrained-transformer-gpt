/**
 * I "confronti pronti" del pannello Confronto (logica pura, testata): le domande che uno studente si
 * fa più spesso, tradotte in una coppia A/B di (run, step). Si calcolano da ciò che c'è nel registro:
 * un preset che non ha senso (un run senza storia del training) non compare.
 *
 *   presets({ runA, courseId, stepsOf, entries })
 *   → [{ id, title, text, a: {run, step}, b: {run, step} }]   step null = pesi finali
 */

/** Gli id dei preset con un link fisso (#lab/go=view:confronto:<id>): il tab li risolve sul GPT del corso. */
export const PRESET_IDS = Object.freeze(["training", "fasi", "corso"]);

/** Lo snapshot più vicino a un decimo dell'ultimo (500 per il GPT del corso): "a che punto era all'inizio". */
export function earlyStep(steps) {
  const list = steps.filter((s) => s > 0);
  if (!list.length) return null;
  const goal = steps.at(-1) / 10;
  return list.reduce((best, s) => (Math.abs(s - goal) < Math.abs(best - goal) ? s : best), list[0]);
}

/**
 * @param {{runA: string, courseId: string, stepsOf: (id: string) => number[], entries: {id: string, name: string, origin: string}[], nameOf?: (id: string) => string}} o
 */
export function presets({ runA, courseId, stepsOf, entries = [], nameOf = (id) => entries.find((e) => e.id === id)?.name ?? id }) {
  const out = [];
  // la storia del training: quella del run A se c'è, altrimenti quella del corso
  const own = stepsOf(runA);
  const historyRun = own.length > 1 ? runA : courseId;
  const steps = stepsOf(historyRun);
  if (steps.length > 1) {
    const first = steps[0];
    const last = steps.at(-1);
    const name = nameOf(historyRun);
    out.push({
      id: "training",
      title: "Che cosa ha cambiato il training",
      text: `${name}: pesi finali (A) contro lo step ${fmt(first)}, i numeri casuali dell'inizio (B).`,
      a: { run: historyRun, step: null },
      b: { run: historyRun, step: first },
    });
    const early = earlyStep(steps);
    if (early !== null && early !== first && early !== last) {
      out.push({
        id: "fasi",
        title: `Step ${fmt(last)} contro step ${fmt(early)}`,
        text: `${name}: che cosa cambia tra l'inizio del training (B, step ${fmt(early)}) e la fine (A, step ${fmt(last)}).`,
        a: { run: historyRun, step: last },
        b: { run: historyRun, step: early },
      });
    }
  }
  // prima e dopo: un run caricato contro il GPT del corso
  const uploaded = entries.filter((e) => e.origin !== "builtin");
  const mine = runA !== courseId ? runA : uploaded[0]?.id ?? null;
  if (mine) {
    out.push({
      id: "corso",
      title: "Il tuo run contro il GPT del corso",
      text: `${nameOf(mine)} (A) contro il GPT del corso (B), pesi finali: che cosa ha cambiato la tua modifica.`,
      a: { run: mine, step: null },
      b: { run: courseId, step: null },
    });
  }
  return out;
}

const fmt = (n) => Number(n).toLocaleString("it-IT", { useGrouping: "always" });
