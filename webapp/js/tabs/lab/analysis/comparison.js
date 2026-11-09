/**
 * Il confronto A/B "vivo" (logica pura, senza DOM: testata in Node con il LabState vero).
 *
 * Ascolta lo stato del Laboratorio (run A e il suo step, run B e il suo step) e ricalcola il confronto
 * quando serve; tabella, scena (modalità differenza) e istogrammi leggono tutti lo stesso risultato,
 * così non lo calcolano tre volte. Le richieste si accavallano (il cursore dello step manda decine di
 * eventi): vince sempre l'ultima, una superata non annuncia risultati vecchi (gettone).
 *
 *   const cmp = createComparison(state);
 *   cmp.on((snapshot) => …);   // { status: "none" | "busy" | "ready" | "error", a, b, result, message, ms }
 *   cmp.current;               // l'ultimo annunciato
 */
import { compareAB } from "./compareModel.js";

/**
 * @param {import("../state.js").LabState} state
 * @param {{schedule?: (fn: () => void) => void}} [o] `schedule`: quando ricalcolare dopo un evento
 *   (nel browser requestAnimationFrame: più eventi nello stesso frame = un solo confronto)
 */
export function createComparison(state, { schedule = (fn) => queueMicrotask(fn) } = {}) {
  const listeners = new Set();
  let token = 0;
  let pending = false;
  let current = { status: "none", a: null, b: null, result: null, message: "", ms: 0 };

  function emit(next) {
    current = Object.freeze(next);
    for (const fn of [...listeners]) {
      try {
        fn(current);
      } catch (error) {
        (globalThis.reportError ?? console.error)(error); // un ascoltatore rotto non ferma gli altri
      }
    }
  }

  /** Che cosa c'è in A e in B (nomi e step), per le etichette: niente pesi. */
  function side(id, step, view) {
    const entry = state.entry(id);
    return { runId: id, step, name: entry?.name ?? id, view, dtype: view?.dtype ?? null, label: view?.label ?? "" };
  }

  async function compute() {
    pending = false;
    const my = ++token;
    if (!state.runA || !state.runB) {
      emit({ status: "none", a: state.runA ? side(state.runA, state.step, null) : null, b: null, result: null, message: "", ms: 0 });
      return current;
    }
    emit({ ...current, status: "busy" });
    try {
      const [va, vb] = await Promise.all([state.weightsAt(state.runA, state.step), state.weightsAt(state.runB, state.stepB)]);
      if (my !== token) return current; // superata da una richiesta più recente
      const runA = state.run(state.runA);
      const runB = state.run(state.runB);
      const t0 = globalThis.performance?.now() ?? Date.now();
      const result = compareAB(va, vb, { trace: runA?.trace ?? runB?.trace ?? null });
      const ms = (globalThis.performance?.now() ?? Date.now()) - t0;
      emit({ status: "ready", a: side(state.runA, state.step, va), b: side(state.runB, state.stepB, vb), result, message: "", ms });
    } catch (error) {
      if (my !== token) return current;
      emit({ status: "error", a: side(state.runA, state.step, null), b: side(state.runB, state.stepB, null), result: null, message: error?.message ?? String(error), ms: 0 });
    }
    return current;
  }

  function request() {
    if (pending) return;
    pending = true;
    schedule(() => void compute());
  }

  const offs = ["runA", "step", "runB", "stepB", "interpolate", "weights"].map((type) => state.on(type, request));

  return {
    get current() {
      return current;
    },
    /** fn(snapshot) a ogni cambiamento; restituisce la funzione che smette di ascoltare. */
    on(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    /** Ricalcola subito (per i test e dopo un caricamento): Promise del nuovo stato. */
    refresh: () => compute(),
    request,
    dispose() {
      token++;
      for (const off of offs) off();
      listeners.clear();
    },
  };
}
