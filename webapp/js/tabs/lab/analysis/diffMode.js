/**
 * La modalità differenza della scena: le matrici di A colorate con A − B (colormap divergente del
 * nucleo) e, da vicino, i numeri della differenza (NumericMatrix li scrive da sola). Niente DOM: parla
 * con la scena (LabScene: showValues, highlightCells) e con lo stato (setMode), testata in Node con
 * una scena finta.
 *
 * Chi possiede i valori della scena? Più moduli possono mostrare valori che non sono i pesi (le azioni
 * della fase 2, per esempio). Regole:
 * - accesa, questa modalità imposta `state.setMode("differenza", { label: "A − B" })` e ricorda gli
 *   array che ha dato a ogni matrice;
 * - se un altro modulo cambia modalità, si spegne e rimette i pesi solo dove la scena mostra ancora i
 *   suoi array (i valori dell'altro modulo, e la loro etichetta, restano);
 * - quando parte un'azione si spegne e rimette i pesi, ma solo sulle matrici che mostrano ancora i
 *   suoi array (`matrixOf(name).data === suo`): i valori messi dall'azione restano;
 * - i tensori senza un gemello della stessa forma in B si velano (non si confrontano): il velo è suo e
 *   lo toglie spegnendosi.
 */
import { differenceMap } from "./compareModel.js";

export const DIFF_LABEL = "A − B";
export const SCALES = Object.freeze({ common: "divergente", weights: "pesi" });
const VEIL_DIM = 0.78;

/**
 * @param {{scene: object, state: object, actionRunning?: () => boolean}} o `scene`: LabScene (o una finta
 *   con showValues, matrixOf, highlightCells); `state`: LabState (setMode, mode, on); `actionRunning`: c'è
 *   un'azione in corso (per dire allo studente perché la differenza si è spenta)
 */
export function createDiffMode({ scene, state, actionRunning = () => false }) {
  const listeners = new Set();
  const buffers = new Map(); // nome → Float32Array riusato tra un ricalcolo e l'altro
  let on = false;
  let scale = "common";
  let mine = new Map(); // nome → l'array dato alla scena
  let veiled = new Set();
  let last = null; // { result, a, b } dell'ultimo confronto
  let legend = null;
  let selfMode = false;
  let reason = "";

  const notify = () => {
    for (const fn of [...listeners]) fn(snapshot());
  };
  const snapshot = () => ({ on, scale, legend, reason });

  function apply() {
    if (!on || !last?.result) return;
    const { result, a, b } = last;
    const d = differenceMap(result, a.view, b.view, buffers);
    for (const [name, values] of d.values) buffers.set(name, values);
    // le matrici della volta prima che ora non si confrontano più tornano ai pesi
    const back = new Map([...mine.keys()].filter((n) => !d.values.has(n)).map((n) => [n, null]));
    if (back.size) scene.showValues(back);
    const shown = scene.showValues(d.values, { colormap: SCALES[scale], label: DIFF_LABEL });
    mine = new Map([...d.values].filter(([n]) => shown.shown.includes(n)));
    // il velo: solo sui tensori senza gemello che la scena disegna davvero
    const nextVeil = new Set(d.unmatched.filter((n) => scene.matrixOf(n)));
    for (const n of veiled) if (!nextVeil.has(n)) scene.highlightCells(n, null);
    for (const n of nextVeil) scene.highlightCells(n, [], { dim: VEIL_DIM });
    veiled = nextVeil;
    legend = { maxAbs: d.maxAbs, scale, shown: mine.size, veiled: veiled.size, missing: shown.missing.length };
    selfMode = true;
    state.setMode("differenza", { label: DIFF_LABEL });
    selfMode = false;
  }

  /**
   * Rimette i pesi sulle matrici che mostrano ancora i nostri array; toglie i nostri veli. L'etichetta dei
   * valori della scena resta quella che c'è: se un altro modulo mostra i suoi valori, non la cancelliamo.
   */
  function restore() {
    const back = new Map();
    for (const [name, values] of mine) if (scene.matrixOf(name)?.data === values) back.set(name, null);
    if (back.size) scene.showValues(back, { label: scene.valuesLabel ?? "" });
    for (const n of veiled) scene.highlightCells(n, null);
    mine = new Map();
    veiled = new Set();
  }

  function setEnabled(value, why = "") {
    const next = Boolean(value) && Boolean(last?.result);
    if (next === on) return on;
    on = next;
    reason = next ? "" : why;
    if (on) apply();
    else {
      restore();
      legend = null;
      if (state.mode === "differenza") {
        selfMode = true;
        state.setMode("pesi");
        selfMode = false;
      }
    }
    notify();
    return on;
  }

  const offs = [
    // un altro modulo ha preso la scena: ci spegniamo senza toccare i suoi valori
    // (restore tocca solo le matrici che mostrano ancora i nostri array, non quelle passate all'altro)
    state.on("mode", ({ mode }) => {
      if (selfMode || !on || mode === "differenza") return;
      on = false;
      restore();
      legend = null;
      reason = actionRunning() ? "action" : "other";
      notify();
    }),
    // la scena si ricostruisce per un altro run A: i valori e i veli di prima non ci sono più; e il motivo
    // dello spegnimento (es. "è partita un'azione") riguardava il run di prima: non va più detto
    state.on("runA", () => {
      mine = new Map();
      veiled = new Set();
      buffers.clear();
      if (!on && reason) {
        reason = "";
        notify();
      }
    }),
  ];

  return {
    /** Accesa? (`on(fn)` invece è l'ascolto dei cambiamenti, come in comparison.js) */
    get enabled() {
      return on;
    },
    get scale() {
      return scale;
    },
    get legend() {
      return legend;
    },
    get state() {
      return snapshot();
    },
    /**
     * Un nuovo confronto (o nessuno): se la modalità è accesa la scena si aggiorna; senza B si spegne.
     * Durante un ricalcolo ("busy", es. scorrendo gli step) resta com'è: il risultato nuovo arriva subito dopo.
     */
    update(comparison) {
      if (comparison?.status === "busy") return;
      last = comparison?.status === "ready" ? comparison : null;
      if (!on) return;
      if (!last) {
        setEnabled(false, "none");
        return;
      }
      apply();
      notify();
    },
    setEnabled,
    toggle: () => setEnabled(!on),
    setScale(next) {
      if (!Object.hasOwn(SCALES, next) || next === scale) return;
      scale = next;
      if (on) apply();
      notify();
    },
    /** Parte un'azione: torna ai pesi veri (le azioni li animano). */
    stopForAction() {
      if (on) setEnabled(false, "action");
    },
    on(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    dispose() {
      if (on) restore();
      for (const off of offs) off();
      listeners.clear();
      buffers.clear();
    },
  };
}
