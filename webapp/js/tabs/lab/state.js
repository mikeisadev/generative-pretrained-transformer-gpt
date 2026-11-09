/**
 * Stato del Laboratorio pesi (logica pura: niente DOM, niente three.js; testato in tests/lab.test.mjs).
 *
 * Tiene il registro dei run (il "GPT del corso" sempre presente, i run caricati da file, quelli salvati
 * nel browser), il run A (quello nella scena) e il run B facoltativo (per il confronto), lo step del
 * training di ciascuno (null = pesi finali), l'elemento scelto (un tensore o un modulo), la modalità
 * della scena. Ogni cambiamento è un evento; i pesi a uno step e il motore JS si chiedono qui, con una
 * cache e la liberazione della memoria quando un run non serve più.
 *
 *   const state = new LabState({ loadBuiltin: () => loadBuiltinRun(), store: new RunStore() });
 *   await state.init();                                   // GPT del corso come run A, pesi finali
 *   state.on("weights", ({ runId, step, view }) => scene.setWeights(view.tensors));
 *   await state.setStep(500);                             // agganciato allo snapshot vero
 *   const { engine, reason } = await state.engineFor(state.runA, 500);
 *
 * Perché un modulo a parte: scena, pannelli, rappresentazioni e i moduli della fase 2 (azioni, confronto,
 * grafici, benchmark) leggono e cambiano la stessa cosa; con un solo stato e i suoi eventi nessuno deve
 * sapere chi altro ascolta, e ogni transizione si prova in Node.
 */

import { BUILTIN_RUN_ID } from "../../shared/weights/run.js";
import { snapStep } from "./steps.js";

export const COURSE_RUN_ID = BUILTIN_RUN_ID;

/** Modalità della scena: i pesi del run A, una differenza A − B o altri valori (fase 2, showValues). */
export const MODES = Object.freeze(["pesi", "differenza", "valori"]);

/** Eventi di LabState.on(tipo, fn): fn(detail, state). */
export const EVENTS = Object.freeze([
  "runs", //        il registro è cambiato: { reason: "init" | "add" | "load" | "save" | "forget" | "saved" }
  "busy", //        un caricamento comincia o finisce: { what: "run" | "weights", id, busy }
  "runA", //        { id, previous, run } (il run è già caricato)
  "runB", //        { id, previous, run } (id null: niente confronto)
  "step", //        { step, previous, runId } dello step di A
  "stepB", //       { step, previous, runId }
  "weights", //     i pesi di A da mostrare sono pronti: { runId, step, view }
  "select", //      { path, previous, kind: "tensor" | "module" | null }
  "mode", //        { mode, previous, label }
  "interpolate", // { interpolate }
  "features", //    { name, enabled } (es. "compare": la fase 2 ha installato il confronto)
  "error", //       { message, cause } (un caricamento è fallito: lo stato resta quello di prima)
]);

const MiB = 1024 * 1024;
const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });

/**
 * Riepilogo leggero di una voce del registro, per i pannelli (niente pesi).
 * @typedef {{id: string, name: string, description: string, origin: "builtin"|"upload"|"saved",
 *   loaded: boolean, saved: boolean, savedBytes: number|null, summary: object|null, error: string|null}} RunEntryInfo
 */

export class LabState {
  /**
   * @param {object} o
   * @param {() => Promise<import("../../shared/weights/run.js").WeightRun>} o.loadBuiltin il GPT del corso
   * @param {object|null} [o.store] un RunStore (o qualcosa con list/load/save/remove/estimate); null = niente salvataggio
   * @param {number} [o.viewCacheBytes] memoria massima delle interpolazioni tenute in cache
   * @param {number} [o.viewCacheSize] quante viste dei pesi (run, step) tenere in cache
   * @param {number} [o.engineCacheSize] quanti motori JS (run, step) tenere vivi
   */
  constructor({ loadBuiltin, store = null, viewCacheBytes = 96 * MiB, viewCacheSize = 24, engineCacheSize = 3 } = {}) {
    if (typeof loadBuiltin !== "function") throw new TypeError("LabState: serve loadBuiltin()");
    this.loadBuiltin = loadBuiltin;
    this.store = store;
    this.limits = { viewCacheBytes, viewCacheSize, engineCacheSize };
    /** id → voce del registro { id, name, description, origin, run, saved, savedBytes, summary, addedAt, loading, error } */
    this._entries = new Map();
    this._listeners = new Map(EVENTS.map((e) => [e, new Set()]));
    this._views = new Map(); // chiave → { promise, bytes, runId } in ordine di uso (LRU)
    this._engines = new Map(); // chiave → { promise, runId } (LRU)
    this._tokens = { a: 0, b: 0, weights: 0 };
    this._seq = 0;
    this.runA = null;
    this.runB = null;
    this.step = null;
    this.stepB = null;
    this.selected = null; // nome di un tensore o percorso di un modulo
    this.mode = "pesi";
    this.modeLabel = "";
    this.interpolate = false;
    this.features = { compare: false };
    /** L'ultima vista dei pesi di A annunciata con "weights": { runId, step, view } o null. */
    this.current = null;
    this.ready = false;
  }

  // ---------------------------------------------------------------- eventi
  /** Ascolta un evento (EVENTS). Restituisce la funzione che smette di ascoltare. */
  on(type, fn) {
    const set = this._listeners.get(type);
    if (!set || typeof fn !== "function") throw new TypeError(`LabState.on: evento sconosciuto "${type}"`);
    set.add(fn);
    return () => set.delete(fn);
  }

  emit(type, detail = {}) {
    for (const fn of [...(this._listeners.get(type) ?? [])]) {
      try {
        fn(detail, this);
      } catch (error) {
        // un ascoltatore rotto non deve fermare gli altri (né lo stato): l'errore resta visibile in console
        (globalThis.reportError ?? console.error)(error);
      }
    }
  }

  // ---------------------------------------------------------------- registro
  /** Il GPT del corso, i run salvati nel browser (senza leggerne i pesi) e i pesi finali come run A. */
  async init() {
    const builtin = await this.loadBuiltin();
    this._put({ id: builtin.id, name: builtin.name, description: builtin.description, origin: "builtin", run: builtin, addedAt: Infinity });
    await this.refreshSaved({ silent: true });
    this.runA = builtin.id;
    this.step = null;
    this.ready = true;
    this.emit("runs", { reason: "init" });
    this.emit("runA", { id: builtin.id, previous: null, run: builtin });
    await this._refreshWeights();
    return builtin;
  }

  /** Il GPT del corso (WeightRun), o null prima di init. */
  get builtin() {
    return this._entries.get(COURSE_RUN_ID)?.run ?? null;
  }

  /** Voci del registro in ordine: il GPT del corso, poi le altre dalla più recente. */
  get entries() {
    return [...this._entries.values()].sort((a, b) => b.addedAt - a.addedAt || a.name.localeCompare(b.name, "it")).map(info);
  }

  has(id) {
    return this._entries.has(id);
  }

  entry(id) {
    const e = this._entries.get(id);
    return e ? info(e) : null;
  }

  /** Il WeightRun già in memoria (null se non c'è o è salvato ma non ancora letto: vedi loadRun). */
  run(id) {
    return this._entries.get(id)?.run ?? null;
  }

  /** I run salvati nel browser entrano nel registro (solo le descrizioni: i pesi si leggono quando servono). */
  async refreshSaved({ silent = false } = {}) {
    if (!this.store) return [];
    const saved = await this.store.list();
    const ids = new Set(saved.map((s) => s.id));
    for (const e of this._entries.values()) {
      if (e.origin !== "builtin") e.saved = ids.has(e.id);
      if (!e.saved) e.savedBytes = null;
    }
    for (const s of saved) {
      const e = this._entries.get(s.id);
      if (e) {
        e.saved = true;
        e.savedBytes = s.bytes ?? null;
        continue;
      }
      this._put({ id: s.id, name: s.name, description: s.description ?? "", origin: "saved", run: null, saved: true, savedBytes: s.bytes ?? null, summary: s, addedAt: s.savedAt ?? 0 });
    }
    if (!silent) this.emit("runs", { reason: "saved" });
    return saved;
  }

  /**
   * Aggiunge i run caricati dallo studente (runsFromFiles). Lo stesso run caricato due volte ha lo
   * stesso id (impronta dei pesi): non si duplica.
   * @returns {{added: string[], duplicates: string[]}}
   */
  addRuns(runs) {
    const added = [];
    const duplicates = [];
    for (const run of runs ?? []) {
      if (!run?.id || !(run.tensors instanceof Map)) continue;
      const e = this._entries.get(run.id);
      if (e) {
        if (!e.run) e.run = run; // era salvato nel browser: ora c'è anche in memoria
        e.addedAt = this._now();
        duplicates.push(run.id);
        continue;
      }
      this._put({ id: run.id, name: run.name, description: run.description ?? "", origin: "upload", run, addedAt: this._now() });
      added.push(run.id);
    }
    if (added.length || duplicates.length) this.emit("runs", { reason: "add", added, duplicates });
    return { added, duplicates };
  }

  /** Il WeightRun di una voce (dal browser se era solo salvato). */
  async loadRun(id) {
    const e = this._entries.get(id);
    if (!e) throw new Error(`run sconosciuto: ${id}`);
    if (e.run) return e.run;
    if (!this.store) throw new Error("salvataggio nel browser non disponibile");
    e.loading ??= (async () => {
      this.emit("busy", { what: "run", id, busy: true });
      try {
        const run = await this.store.load(id);
        if (!run) throw new Error(`«${e.name}» non è più leggibile nel browser (spazio liberato o dati danneggiati): caricalo di nuovo dai file`);
        e.run = run;
        e.error = null;
        this.emit("runs", { reason: "load", id });
        return run;
      } catch (error) {
        e.error = error.message;
        throw error;
      } finally {
        e.loading = null;
        this.emit("busy", { what: "run", id, busy: false });
      }
    })();
    return e.loading;
  }

  /** Salva un run nel browser (RunStore): { ok, bytes } oppure { ok: false, reason }. */
  async saveRun(id) {
    if (!this.store) return { ok: false, reason: "salvataggio nel browser non disponibile" };
    const e = this._entries.get(id);
    if (!e) return { ok: false, reason: `run sconosciuto: ${id}` };
    if (e.origin === "builtin") return { ok: false, reason: "il GPT del corso è già nei dati della web app" };
    const run = await this.loadRun(id);
    const result = await this.store.save(run);
    if (result.ok) {
      e.saved = true;
      e.savedBytes = result.bytes;
      this.emit("runs", { reason: "save", id });
    }
    return result;
  }

  /**
   * "Dimentica": toglie un run dal registro, dalla memoria e dal browser. Il GPT del corso resta sempre.
   * Se era il run A si torna al GPT del corso; se era B il confronto si chiude.
   */
  async forget(id) {
    const e = this._entries.get(id);
    if (!e || e.origin === "builtin") return false;
    if (e.saved && this.store) {
      const removed = await this.store.remove(id);
      if (!removed) {
        this.emit("error", { message: `il browser non ha cancellato «${e.name}»: riprova più tardi` });
        return false;
      }
    }
    this._entries.delete(id);
    this.release(id);
    if (this.runB === id) await this.setRunB(null);
    if (this.runA === id) await this.setRunA(COURSE_RUN_ID);
    if (this.selected && !this._selectionValid(this.selected)) this.select(null);
    this.emit("runs", { reason: "forget", id });
    return true;
  }

  /** Spazio usato dai run salvati nel browser (byte) e, se il browser lo dice, quota del sito. */
  async storageUsage() {
    let saved = 0;
    for (const e of this._entries.values()) if (e.saved && e.savedBytes) saved += e.savedBytes;
    const estimate = (await this.store?.estimate?.()) ?? null;
    return { saved, usage: estimate?.usage ?? null, quota: estimate?.quota ?? null };
  }

  // ---------------------------------------------------------------- run A, run B, step
  /** Step degli snapshot del training di un run che la scena sa mostrare (stessi tensori dei pesi). */
  stepsOf(id) {
    const run = this.run(id);
    if (!run) return [];
    return run.snapshots.filter((s) => s.available && s.layoutMatches !== false).map((s) => s.step);
  }

  /** `step` valido per il run: null (finali), lo snapshot più vicino o, con l'interpolazione, lo step chiesto. */
  normalizeStep(id, step, interpolate = this.interpolate) {
    if (step === null || step === undefined) return null;
    const steps = this.stepsOf(id);
    if (!steps.length || !Number.isFinite(step)) return null;
    if (interpolate && steps.length > 1) return Math.min(steps.at(-1), Math.max(steps[0], Math.round(step)));
    return snapStep(steps, step);
  }

  /**
   * Il run nella scena. Prima si carica (dal browser, se serve), poi si annuncia: chi ascolta "runA"
   * riceve il run pronto, poi "weights" con i pesi allo step chiesto.
   * @returns {Promise<boolean>} false se il run non c'è, il caricamento fallisce o una richiesta più recente lo supera
   */
  async setRunA(id, { step = null } = {}) {
    if (!this._entries.has(id)) return false;
    const token = ++this._tokens.a;
    let run;
    try {
      run = await this.loadRun(id);
    } catch (error) {
      if (token === this._tokens.a) this.emit("error", { message: error.message, cause: error });
      return false;
    }
    if (token !== this._tokens.a) return false;
    const previous = this.runA;
    const previousStep = this.step;
    this.runA = id;
    this.step = this.normalizeStep(id, step);
    if (previous !== id) {
      if (previous && previous !== this.runB) this.release(previous);
      if (this.selected && !this._selectionValid(this.selected)) this.select(null);
      this.emit("runA", { id, previous, run });
    }
    if (this.step !== previousStep || previous !== id) this.emit("step", { step: this.step, previous: previousStep, runId: id });
    await this._refreshWeights();
    return token === this._tokens.a;
  }

  /** Lo step di A (null = pesi finali): agganciato agli snapshot, a meno dell'interpolazione dichiarata. */
  async setStep(step) {
    if (!this.runA) return false;
    const next = this.normalizeStep(this.runA, step);
    if (next === this.step) return true;
    const previous = this.step;
    this.step = next;
    this.emit("step", { step: next, previous, runId: this.runA });
    return this._refreshWeights();
  }

  /** Il run B per il confronto (null = nessuno), con il suo step. */
  async setRunB(id, { step = null } = {}) {
    const token = ++this._tokens.b;
    const previous = this.runB;
    if (id === null || id === undefined) {
      this.runB = null;
      this.stepB = null;
      if (previous && previous !== this.runA) this.release(previous);
      if (previous !== null) this.emit("runB", { id: null, previous, run: null });
      return true;
    }
    if (!this._entries.has(id)) return false;
    let run;
    try {
      run = await this.loadRun(id);
    } catch (error) {
      if (token === this._tokens.b) this.emit("error", { message: error.message, cause: error });
      return false;
    }
    if (token !== this._tokens.b) return false;
    const previousStep = this.stepB;
    this.runB = id;
    this.stepB = this.normalizeStep(id, step);
    if (previous !== id) {
      if (previous && previous !== this.runA) this.release(previous);
      this.emit("runB", { id, previous, run });
    }
    if (this.stepB !== previousStep || previous !== id) this.emit("stepB", { step: this.stepB, previous: previousStep, runId: id });
    return true;
  }

  setStepB(step) {
    if (!this.runB) return false;
    const next = this.normalizeStep(this.runB, step);
    if (next === this.stepB) return true;
    const previous = this.stepB;
    this.stepB = next;
    this.emit("stepB", { step: next, previous, runId: this.runB });
    return true;
  }

  /** Interpolazione tra gli snapshot (approssimata, dichiarata): spegnendola lo step torna su uno snapshot vero. */
  async setInterpolate(on) {
    const value = Boolean(on);
    if (value === this.interpolate) return;
    this.interpolate = value;
    this.emit("interpolate", { interpolate: value });
    if (!value && this.runA && this.step !== null) {
      const snapped = this.normalizeStep(this.runA, this.step, false);
      if (snapped !== this.step) {
        const previous = this.step;
        this.step = snapped;
        this.emit("step", { step: snapped, previous, runId: this.runA });
        await this._refreshWeights();
      }
    }
  }

  // ---------------------------------------------------------------- scelta, modalità, funzioni
  /** Il tensore o il modulo scelto (percorso puntato di PyTorch), null = niente. */
  select(path) {
    const next = typeof path === "string" && path ? path : null;
    if (next === this.selected) return;
    const previous = this.selected;
    this.selected = next;
    this.emit("select", { path: next, previous, kind: this.kindOf(next) });
  }

  /** "tensor" se `path` è un tensore del run A, "module" se ne contiene, null altrimenti. */
  kindOf(path, runId = this.runA) {
    if (!path) return null;
    const run = this.run(runId);
    if (!run) return null;
    if (run.tensors.has(path)) return "tensor";
    const prefix = `${path}.`;
    for (const name of run.tensors.keys()) if (name.startsWith(prefix)) return "module";
    return null;
  }

  /** Il nome del tensore scelto (null se la scelta è un modulo o niente). */
  get tensor() {
    return this.kindOf(this.selected) === "tensor" ? this.selected : null;
  }

  setMode(mode, { label = "" } = {}) {
    if (!MODES.includes(mode)) throw new RangeError(`modalità sconosciuta: ${mode} (disponibili: ${MODES.join(", ")})`);
    if (mode === this.mode && label === this.modeLabel) return;
    const previous = this.mode;
    this.mode = mode;
    this.modeLabel = label;
    this.emit("mode", { mode, previous, label });
  }

  /** Funzioni opzionali installate da altri moduli (es. "compare" dalla fase 2): i pannelli mostrano i comandi relativi. */
  enableFeature(name, enabled = true) {
    if (this.features[name] === Boolean(enabled)) return;
    this.features[name] = Boolean(enabled);
    this.emit("features", { name, enabled: Boolean(enabled) });
  }

  // ---------------------------------------------------------------- pesi e motore
  /**
   * I pesi di un run a uno step: `{ runId, step, requested, exact, from, t, source, tensors, label, dtype }`.
   * step null = pesi finali; uno step degli snapshot = lo snapshot vero; uno step intermedio con
   * `interpolate` = media pesata dei due snapshot vicini (source "interpolated", exact false). In cache.
   */
  async weightsAt(id, step = null, { interpolate = this.interpolate } = {}) {
    const run = await this.loadRun(id);
    const target = step === null || step === undefined ? null : this.normalizeStep(id, step, interpolate);
    if (step !== null && step !== undefined && target === null) throw new Error(`«${run.name}» non ha snapshot del training: ci sono solo i pesi finali`);
    const key = `${id}|${target ?? "finali"}`;
    const hit = this._touch(this._views, key);
    if (hit) return hit.promise;
    const entry = { runId: id, bytes: 0, promise: null };
    entry.promise = (async () => {
      if (target === null) {
        const first = run.tensors.values().next().value;
        return view({ runId: id, step: null, requested: null, exact: true, from: [], t: 0, source: "final", tensors: run.tensors, dtype: first?.dtype ?? "float32" });
      }
      const w = await run.weightsAt(target, { interpolate: true });
      const source = w.exact ? "snapshot" : "interpolated";
      if (source === "interpolated") entry.bytes = w.flat.byteLength; // le interpolazioni sono nostre: contano nel budget
      return view({ runId: id, step: w.step, requested: step, exact: w.exact, from: w.from, t: w.t, source, tensors: w.tensors, dtype: "float16" });
    })();
    entry.promise.catch(() => this._views.get(key) === entry && this._views.delete(key));
    this._views.set(key, entry);
    entry.promise.then(() => this._trimViews(), () => {});
    return entry.promise;
  }

  /**
   * Il motore JS (js/engine/gpt.js) con i pesi del run allo step: `{ engine, runId, step, view, reason }`.
   * Solo per i run compatibili (stessi nomi e forme del GPT del corso, calcolo dichiarato uguale): per
   * gli altri `engine` è null e `reason` dice perché, in italiano. In cache (pochi motori vivi).
   */
  async engineFor(id, step = null) {
    const run = await this.loadRun(id);
    if (!run.compatibleWithEngine) return { engine: null, runId: id, step, view: null, reason: incompatibility(run) };
    const v = await this.weightsAt(id, step);
    const key = `${id}|${v.step ?? "finali"}`;
    const hit = this._touch(this._engines, key);
    if (hit) return hit.promise;
    const entry = { runId: id, promise: null };
    entry.promise = Promise.resolve().then(() => {
      const engine = run.createEngine();
      if (v.source !== "final") engine.setWeights(run.engineFlat(v.tensors));
      return { engine, runId: id, step: v.step, view: v, reason: null };
    });
    entry.promise.catch(() => this._engines.get(key) === entry && this._engines.delete(key));
    this._engines.set(key, entry);
    while (this._engines.size > this.limits.engineCacheSize) this._engines.delete(this._engines.keys().next().value);
    return entry.promise;
  }

  /** Libera le viste dei pesi e i motori di un run (resta nel registro: riaprirlo li ricrea). */
  release(id) {
    for (const map of [this._views, this._engines]) for (const [key, e] of [...map]) if (e.runId === id) map.delete(key);
  }

  /** Memoria delle cache (per i test e la console): viste, interpolazioni in byte, motori. */
  get cacheInfo() {
    let bytes = 0;
    for (const e of this._views.values()) bytes += e.bytes;
    return { views: this._views.size, bytes, engines: this._engines.size };
  }

  dispose() {
    for (const set of this._listeners.values()) set.clear();
    this._views.clear();
    this._engines.clear();
    this._tokens.a++;
    this._tokens.b++;
    this._tokens.weights++;
  }

  // ---------------------------------------------------------------- interni
  _put(e) {
    this._entries.set(e.id, { description: "", saved: false, savedBytes: null, summary: null, loading: null, error: null, ...e });
  }

  /** Un "orologio" che cresce sempre: l'ordine del registro non dipende dalla risoluzione di Date.now(). */
  _now() {
    return Date.now() + ++this._seq / 1000;
  }

  /** Porta in fondo (usata di recente) una voce di una cache LRU e la restituisce. */
  _touch(map, key) {
    const e = map.get(key);
    if (!e) return null;
    map.delete(key);
    map.set(key, e);
    return e;
  }

  _trimViews() {
    const { viewCacheBytes, viewCacheSize } = this.limits;
    const protectedKeys = new Set([this._viewKey(this.runA, this.step), this._viewKey(this.runB, this.stepB)]);
    let bytes = 0;
    for (const e of this._views.values()) bytes += e.bytes;
    for (const [key, e] of [...this._views]) {
      if (this._views.size <= viewCacheSize && bytes <= viewCacheBytes) break;
      if (protectedKeys.has(key)) continue;
      this._views.delete(key);
      bytes -= e.bytes;
    }
  }

  _viewKey(id, step) {
    return id ? `${id}|${step ?? "finali"}` : null;
  }

  /** Carica i pesi di A allo step attuale e li annuncia ("weights"), se nel frattempo nessuno ha chiesto altro. */
  async _refreshWeights() {
    const token = ++this._tokens.weights;
    const runId = this.runA;
    const step = this.step;
    this.emit("busy", { what: "weights", id: runId, busy: true });
    try {
      const v = await this.weightsAt(runId, step);
      if (token !== this._tokens.weights) return false;
      this.current = { runId, step, view: v };
      this.emit("weights", this.current);
      return true;
    } catch (error) {
      if (token === this._tokens.weights) this.emit("error", { message: error.message, cause: error });
      return false;
    } finally {
      if (token === this._tokens.weights) this.emit("busy", { what: "weights", id: runId, busy: false });
    }
  }

  _selectionValid(path) {
    return this.kindOf(path) !== null;
  }
}

/** Riepilogo di una voce del registro (senza il WeightRun: i pannelli non toccano i pesi). */
function info(e) {
  const run = e.run;
  return {
    id: e.id,
    name: e.name,
    description: e.description,
    origin: e.origin,
    loaded: Boolean(run),
    saved: e.saved,
    savedBytes: e.savedBytes,
    summary: run ? run.summary() : e.summary,
    error: e.error,
  };
}

/** La vista dei pesi con la sua etichetta in italiano. */
function view(v) {
  const label =
    v.source === "final"
      ? `pesi finali (${v.dtype})`
      : v.source === "snapshot"
        ? `snapshot dello step ${INT.format(v.step)} (float16)`
        : `step ${INT.format(v.step)}, interpolato tra gli snapshot ${INT.format(v.from[0])} e ${INT.format(v.from[1])}: approssimato`;
  return Object.freeze({ ...v, label });
}

/** Perché il motore JS non esegue un run: i motivi della libreria, in una frase. */
export function incompatibility(run) {
  const issues = run.engine?.issues ?? [];
  return issues.length ? issues.join("; ") : "il motore JS non sa eseguire questi pesi";
}
