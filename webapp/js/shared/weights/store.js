/**
 * Run caricati salvati nel browser (IndexedDB), per ritrovarli alla visita successiva.
 *
 *   const store = new RunStore();
 *   await store.save(run);          // {ok: true, id, bytes} | {ok: false, reason}
 *   await store.list();             // [{id, name, savedAt, bytes, totalParams, …}] (senza i pesi)
 *   const run = await store.load(id);
 *   await store.remove(id);         // "Dimentica"
 *   await store.estimate();         // {usage, quota} del browser, o null
 *
 * Due archivi: `runs` (descrizioni leggere, per l'elenco) e `blobs` (pesi float32 e snapshot
 * float16 così come sono arrivati), così l'elenco non legge megabyte di pesi. Ogni operazione
 * è in try/catch: senza IndexedDB (navigazione privata, file://, quota esaurita) l'elenco è
 * vuoto, `save` risponde `{ok: false, reason}` e la pagina continua a funzionare con i run in
 * memoria. Il GPT del corso non si salva: è già nei dati della web app.
 */

import { WeightRun, snapshotEntry } from "./run.js";
import { megabytes } from "./validate.js";

const DB_NAME = "gpt-didattico-pesi";
const DB_VERSION = 1;
const RUNS = "runs";
const BLOBS = "blobs";
export const RECORD_FORMAT = 1;

const weightsKey = (id) => `${id}|weights`;
const snapshotKey = (id, step) => `${id}|snap|${step}`;

/**
 * Run → descrizione clonabile + blob da salvare (logica pura, testata senza IndexedDB).
 * I pesi si salvano come un unico Float32Array nell'ordine dei tensori; gli snapshot con i
 * loro byte originali float16 (metà spazio).
 * @returns {Promise<{record: object, blobs: {key: string, data: ArrayBuffer}[]}>}
 */
export async function serializeRun(run, { now = Date.now() } = {}) {
  if (run.source === "builtin") throw new Error("il GPT del corso è già nei dati della web app: non serve salvarlo");
  const tensors = [];
  const flat = new Float32Array(run.totalParams);
  let offset = 0;
  for (const t of run.tensors.values()) {
    flat.set(t.data, offset);
    tensors.push({ name: t.name, shape: [...t.shape], numel: t.numel, dtype: t.dtype, offset });
    offset += t.numel;
  }
  const blobs = [{ key: weightsKey(run.id), data: flat.buffer }];
  const snapshots = [];
  for (const s of run.snapshots) {
    let available = s.available;
    if (available) {
      try {
        blobs.push({ key: snapshotKey(run.id, s.step), data: await s.raw() });
      } catch {
        available = false; // file non più leggibile: restano le metriche dello snapshot
      }
    }
    snapshots.push({ step: s.step, file: s.file, meta: s.meta, available, layoutMatches: s.layoutMatches });
  }
  const bytes = blobs.reduce((sum, b) => sum + b.data.byteLength, 0);
  const record = {
    format: RECORD_FORMAT,
    id: run.id,
    name: run.name,
    description: run.description,
    savedAt: now,
    bytes,
    config: run.config,
    vocab: run.vocab,
    trace: run.trace,
    benchmark: run.benchmark,
    info: run.info,
    meta: run.meta,
    tensors,
    snapshots,
    summary: { ...run.summary(), steps: snapshots.filter((s) => s.available).map((s) => s.step) },
  };
  return { record, blobs };
}

/**
 * Descrizione salvata + lettore dei blob → WeightRun (gli snapshot si leggono quando servono).
 * @param {object} record
 * @param {ArrayBuffer} weights blob dei pesi
 * @param {(key: string) => Promise<ArrayBuffer>} readBlob
 */
export function deserializeRun(record, weights, readBlob) {
  if (record?.format !== RECORD_FORMAT) throw new Error(`run salvato in un formato sconosciuto (${record?.format})`);
  if (!(weights instanceof ArrayBuffer) || weights.byteLength % 4 !== 0) throw new Error("pesi salvati illeggibili (blob danneggiato)");
  const flat = new Float32Array(weights);
  // il blob e la descrizione si scrivono nella stessa transazione: se non tornano, il salvataggio è danneggiato
  const described = (record.tensors ?? []).reduce((sum, t) => Math.max(sum, t.offset + t.numel), 0);
  if (!Array.isArray(record.tensors) || described !== flat.length) throw new Error(`pesi salvati danneggiati: ${flat.length} valori invece di ${described}`);
  const tensors = new Map(
    record.tensors.map((t) => [t.name, { name: t.name, shape: t.shape, numel: t.numel, dtype: t.dtype, data: flat.subarray(t.offset, t.offset + t.numel) }]),
  );
  const layout = record.trace?.layout ?? [];
  const totalParams = record.trace?.totalParams ?? 0;
  const snapshots = record.snapshots.map((s) =>
    snapshotEntry({
      step: s.step,
      file: s.file,
      meta: s.meta,
      layout,
      totalParams,
      layoutMatches: s.layoutMatches,
      raw: s.available ? () => readBlob(snapshotKey(record.id, s.step)) : null,
    }),
  );
  return new WeightRun({
    id: record.id,
    name: record.name,
    description: record.description,
    source: "upload",
    tensors,
    flat,
    config: record.config,
    vocab: record.vocab,
    trace: record.trace,
    snapshots,
    benchmark: record.benchmark,
    info: record.info,
    meta: { ...record.meta, savedAt: record.savedAt, warnings: [...(record.meta?.warnings ?? [])] },
  });
}

/** Promessa di una richiesta IndexedDB. */
function request(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Promessa della fine di una transazione (anche annullata, es. per la quota). */
function finished(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error("transazione annullata"));
  });
}

function reasonOf(error) {
  if (error?.name === "QuotaExceededError") return "spazio del browser esaurito: elimina qualche run salvato o libera spazio";
  return `il browser non ha salvato il run (${error?.message || error?.name || "errore sconosciuto"})`;
}

export class RunStore {
  /**
   * @param {{indexedDB?: IDBFactory|null, keyRange?: typeof IDBKeyRange|null, name?: string,
   *   storage?: {estimate: () => Promise<{usage: number, quota: number}>}|null}} [o]
   *   le dipendenze del browser si possono sostituire (test, altri database)
   */
  constructor({ indexedDB = globalThis.indexedDB ?? null, keyRange = globalThis.IDBKeyRange ?? null, name = DB_NAME, storage = globalThis.navigator?.storage ?? null } = {}) {
    this.idb = indexedDB;
    this.keyRange = keyRange;
    this.name = name;
    this.storage = storage;
    this.lastError = null;
    this._db = null;
  }

  /** true se il browser offre IndexedDB (non garantisce che salvare riesca: vedi `save`). */
  get available() {
    return Boolean(this.idb) && this.lastError?.fatal !== true;
  }

  _open() {
    if (!this.idb) return Promise.reject(new Error("IndexedDB non disponibile in questo browser"));
    if (!this._db) {
      const pending = new Promise((resolve, reject) => {
        let req;
        try {
          req = this.idb.open(this.name, DB_VERSION);
        } catch (error) {
          reject(error); // es. navigazione privata di alcuni browser
          return;
        }
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(RUNS)) db.createObjectStore(RUNS, { keyPath: "id" });
          if (!db.objectStoreNames.contains(BLOBS)) db.createObjectStore(BLOBS);
        };
        req.onsuccess = () => {
          const db = req.result;
          // Un'altra scheda aggiorna il database (si lascia fare) o il browser chiude la connessione
          // (spazio liberato a mano, disco pieno): la prossima operazione la riapre invece di fallire.
          const forget = () => {
            if (this._db === pending) this._db = null;
          };
          db.onversionchange = () => {
            db.close();
            forget();
          };
          db.onclose = forget;
          resolve(db);
        };
        req.onerror = () => reject(req.error);
        req.onblocked = () => reject(new Error("database occupato da un'altra scheda"));
      });
      this._db = pending;
      pending.catch((error) => {
        if (this._db === pending) this._db = null;
        this.lastError = { message: error?.message ?? String(error), fatal: true };
      });
    }
    return this._db;
  }

  /** Spazio usato e disponibile per questo sito (byte), o null se il browser non lo dice. */
  async estimate() {
    try {
      const e = await this.storage?.estimate?.();
      return e && Number.isFinite(e.quota) ? { usage: e.usage ?? 0, quota: e.quota } : null;
    } catch {
      return null;
    }
  }

  /** Run salvati, dal più recente: solo le descrizioni (niente pesi). */
  async list() {
    try {
      const db = await this._open();
      const records = await request(db.transaction(RUNS, "readonly").objectStore(RUNS).getAll());
      return records
        .filter((r) => r?.format === RECORD_FORMAT)
        .map((r) => ({ ...r.summary, id: r.id, name: r.name, description: r.description, savedAt: r.savedAt, bytes: r.bytes }))
        .sort((a, b) => b.savedAt - a.savedAt);
    } catch (error) {
      this.lastError = { message: error?.message ?? String(error), fatal: this.lastError?.fatal ?? false };
      return [];
    }
  }

  /** true se il run con quell'id è già salvato. */
  async has(id) {
    try {
      const db = await this._open();
      return (await request(db.transaction(RUNS, "readonly").objectStore(RUNS).count(id))) > 0;
    } catch {
      return false;
    }
  }

  /**
   * Salva (o sostituisce) un run.
   * @returns {Promise<{ok: true, id: string, bytes: number}|{ok: false, reason: string}>}
   */
  async save(run) {
    let payload;
    try {
      payload = await serializeRun(run);
    } catch (error) {
      return { ok: false, reason: error.message };
    }
    const { record, blobs } = payload;
    const space = await this.estimate();
    if (space && space.usage + record.bytes > space.quota) {
      return { ok: false, reason: `servono ${megabytes(record.bytes)} ma nel browser ne restano ${megabytes(Math.max(0, space.quota - space.usage))}` };
    }
    try {
      const db = await this._open();
      // Una sola transazione: se il browser la annulla (quota esaurita a metà) resta la versione di
      // prima, intera. Le richieste di una transazione vanno in ordine: prima si tolgono i blob vecchi
      // (uno snapshot che non c'è più non deve restare orfano), poi si scrivono i nuovi.
      const tx = db.transaction([RUNS, BLOBS], "readwrite");
      const done = finished(tx);
      tx.objectStore(BLOBS).delete(this._blobRange(record.id));
      tx.objectStore(RUNS).put(record);
      for (const b of blobs) tx.objectStore(BLOBS).put(b.data, b.key);
      await done;
      return { ok: true, id: record.id, bytes: record.bytes };
    } catch (error) {
      return { ok: false, reason: reasonOf(error) };
    }
  }

  /** Il run salvato, o null (non c'è o il browser non lo legge). */
  async load(id) {
    try {
      const db = await this._open();
      const tx = db.transaction([RUNS, BLOBS], "readonly");
      const [record, weights] = await Promise.all([request(tx.objectStore(RUNS).get(id)), request(tx.objectStore(BLOBS).get(weightsKey(id)))]);
      if (!record || !weights) return null;
      const readBlob = async (key) => {
        const blob = await request((await this._open()).transaction(BLOBS, "readonly").objectStore(BLOBS).get(key));
        if (!blob) throw new Error("snapshot non più presente nel browser");
        return blob;
      };
      return deserializeRun(record, weights, readBlob);
    } catch (error) {
      this.lastError = { message: error?.message ?? String(error), fatal: this.lastError?.fatal ?? false };
      return null;
    }
  }

  /** Dimentica un run (descrizione e tutti i suoi blob). */
  async remove(id) {
    try {
      const db = await this._open();
      const tx = db.transaction([RUNS, BLOBS], "readwrite");
      const done = finished(tx);
      tx.objectStore(BLOBS).delete(this._blobRange(id));
      tx.objectStore(RUNS).delete(id);
      await done;
      return true;
    } catch {
      return false;
    }
  }

  /** Dimentica tutti i run salvati. */
  async clear() {
    try {
      const db = await this._open();
      const tx = db.transaction([RUNS, BLOBS], "readwrite");
      const done = finished(tx);
      tx.objectStore(RUNS).clear();
      tx.objectStore(BLOBS).clear();
      await done;
      return true;
    } catch {
      return false;
    }
  }

  /** Tutte le chiavi dei blob di un run ("id|…"): l'intervallo da "id|" a "id|\uffff" (il carattere più alto). */
  _blobRange(id) {
    if (!this.keyRange) throw new Error("IDBKeyRange non disponibile in questo browser");
    return this.keyRange.bound(`${id}|`, `${id}|\uffff`);
  }

  /** Chiude il database (es. prima di eliminarlo nei test). */
  async close() {
    try {
      (await this._db)?.close();
    } catch {
      // già chiuso o mai aperto
    }
    this._db = null;
  }
}
