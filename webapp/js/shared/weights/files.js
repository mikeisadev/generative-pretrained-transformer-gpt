/**
 * Riconoscimento dei file caricati dallo studente (cartella con `webkitdirectory`, selezione
 * multipla o drag & drop) e lettura con i limiti di dimensione.
 *
 * Funziona con `FileList`, array di `File` o oggetti "tipo File" `{name, size, text(),
 * arrayBuffer()}` con il percorso in `webkitRelativePath` (cartelle), `relativePath` o `path`
 * (chi converte le voci del drag & drop). `recognizeFiles` guarda SOLO nomi, percorsi e
 * dimensioni: non legge niente, così si può mostrare subito cosa è stato riconosciuto.
 *
 * Regole (una cartella = un run):
 * - una cartella con `weights.json`, `run.json` o un `.safetensors` è la radice di un run;
 * - `trace.json` e `snap_XXXXX.f16.bin` appartengono al run della cartella o della più vicina
 *   cartella antenata che è un run (come `data/` e `data/training/` del corso); se non ce n'è
 *   nessuna formano un run da soli (pesi = l'ultimo snapshot);
 * - più `.safetensors` nella stessa cartella sono run diversi, tranne gli shard
 *   `nome-00001-of-00003.safetensors`, che formano un solo run;
 * - una cartella con sottocartelle che sono run (es. `runs/a/`, `runs/b/`) dà più run.
 */

import { WeightsFormatError, megabytes, withLimits } from "./validate.js";

const SNAPSHOT = /^snap_(\d+)\.f16\.bin$/;
const SHARD = /^(.+)-(\d{5})-of-(\d{5})\.safetensors$/;

/** Percorso relativo di un file caricato, con le barre normali e senza "./" iniziali. */
export function filePath(file) {
  const raw = file.webkitRelativePath || file.relativePath || file.path || file.name || "";
  return String(raw).replace(/\\/g, "/").replace(/^\/+/, "").replace(/^(\.\/)+/, "").replace(/\/{2,}/g, "/");
}

function kindOf(name) {
  if (name.startsWith(".") || name === "Thumbs.db") return { kind: "system" };
  const lower = name.toLowerCase();
  if (lower === "weights.json") return { kind: "manifest" };
  if (lower === "trace.json") return { kind: "trace" };
  if (lower === "run.json") return { kind: "runInfo" };
  if (lower === "benchmark.json") return { kind: "benchmark" };
  const snap = SNAPSHOT.exec(name);
  if (snap) return { kind: "snapshot", step: Number(snap[1]) };
  if (lower.endsWith(".safetensors")) return { kind: "safetensors" };
  if (lower.endsWith(".bin")) return { kind: "binary" };
  return { kind: "other" };
}

const dirOf = (path) => (path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "");
const baseOf = (path) => path.slice(path.lastIndexOf("/") + 1);
const parentOf = (dir) => (dir.includes("/") ? dir.slice(0, dir.lastIndexOf("/")) : dir === "" ? null : "");

function emptyGroup(key, dir, name) {
  return { key, dir, name, manifest: null, binaries: [], safetensors: [], trace: null, snapshots: [], runInfo: null, benchmark: null, bytes: 0, warnings: [], errors: [] };
}

/**
 * Raggruppa i file in run, senza leggerli.
 * @param {FileList|Iterable<object>} files
 * @param {{limits?: object}} [options]
 * @returns {{runs: object[], ignored: {path: string, reason: string}[], totalBytes: number}}
 *   ogni run: `{key, dir, name, manifest, binaries[], safetensors[], trace, snapshots[{step, file, path}],
 *   runInfo, benchmark, bytes, warnings[], errors[]}` (i file sono voci `{file, path, size}`; `errors`:
 *   file oltre il limite della loro dimensione, il run non si può caricare)
 * @throws {WeightsFormatError} se i file da leggere superano `limits.maxTotalBytes`
 */
export function recognizeFiles(files, { limits } = {}) {
  const lim = withLimits(limits);
  const ignored = [];
  const entries = [];
  for (const file of Array.from(files ?? [])) {
    const path = filePath(file);
    const name = baseOf(path);
    if (!name) continue;
    if (path.split("/").some((p) => p === "..")) {
      ignored.push({ path, reason: "percorso non ammesso (contiene \"..\")" });
      continue;
    }
    const kind = kindOf(name);
    if (kind.kind === "system") continue; // .DS_Store e simili: non vale la pena segnalarli
    if (kind.kind === "other") {
      const reason = name === "model.json" ? "model.json serve alla scena del Transformer, non a un run" : "tipo di file non riconosciuto";
      ignored.push({ path, reason });
      continue;
    }
    entries.push({ file, path, dir: dirOf(path), name, size: Number(file.size) || 0, ...kind });
  }

  // 1. Le radici dei run: cartelle con pesi o con run.json.
  const roots = new Set(entries.filter((e) => ["manifest", "safetensors", "runInfo"].includes(e.kind)).map((e) => e.dir));
  // 2. Le cartelle che contengono solo la storia del training vanno al run antenato più vicino.
  const ownerOf = (dir) => {
    for (let d = dir; d !== null; d = parentOf(d)) if (roots.has(d)) return d;
    return null;
  };
  for (const e of entries) {
    if ((e.kind === "trace" || e.kind === "snapshot") && ownerOf(e.dir) === null) roots.add(e.dir);
  }

  const groups = new Map(); // chiave → gruppo
  const mainOf = new Map(); // cartella radice → chiave del run principale
  const groupFor = (dir) => {
    if (!mainOf.has(dir)) {
      const key = dir || ".";
      mainOf.set(dir, key);
      groups.set(key, emptyGroup(key, dir, baseOf(dir) || null));
    }
    return groups.get(mainOf.get(dir));
  };

  // 3. I safetensors: shard uniti, gli altri un run ciascuno (il primo può essere il principale).
  const tensorsByDir = new Map();
  for (const e of entries.filter((x) => x.kind === "safetensors")) {
    if (!tensorsByDir.has(e.dir)) tensorsByDir.set(e.dir, []);
    tensorsByDir.get(e.dir).push(e);
  }
  for (const [dir, list] of tensorsByDir) {
    const shards = new Map();
    const singles = [];
    for (const e of list) {
      const m = SHARD.exec(e.name);
      if (m) {
        const id = `${m[1]}-of-${m[3]}`;
        if (!shards.has(id)) shards.set(id, { prefix: m[1], total: Number(m[3]), parts: [] });
        shards.get(id).parts.push({ ...e, index: Number(m[2]) });
      } else singles.push(e);
    }
    singles.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    const sets = [...[...shards.values()].map((s) => ({ name: s.prefix, entries: s.parts.sort((a, b) => a.index - b.index), shard: s })), ...singles.map((e) => ({ name: e.name.replace(/\.safetensors$/i, ""), entries: [e] }))];
    const hasManifest = entries.some((x) => x.kind === "manifest" && x.dir === dir);
    sets.forEach((set, i) => {
      // Il primo insieme diventa il run "principale" della cartella solo se manca weights.json.
      const group = i === 0 && !hasManifest ? groupFor(dir) : null;
      const target = group ?? (() => {
        const key = `${dir || "."}/${set.name}`;
        const g = emptyGroup(key, dir, set.name);
        groups.set(key, g);
        return g;
      })();
      // Un solo insieme di pesi: il run prende il nome della cartella; più insiemi: quello del file.
      if (group && (!group.name || sets.length > 1)) group.name = set.name;
      target.safetensors.push(...set.entries.map(pick));
      if (set.shard) {
        const have = new Set(set.entries.map((e) => e.index));
        const missing = [];
        for (let k = 1; k <= set.shard.total; k++) if (!have.has(k)) missing.push(k);
        if (missing.length) target.warnings.push(`mancano gli shard ${missing.join(", ")} di ${set.shard.total} di ${set.name}`);
      }
    });
  }

  // 4. Tutto il resto al run principale della propria radice.
  for (const e of entries) {
    if (e.kind === "safetensors") continue;
    const root = e.kind === "trace" || e.kind === "snapshot" ? ownerOf(e.dir) : e.dir;
    if (root === null || !roots.has(root)) {
      ignored.push({ path: e.path, reason: "fuori da una cartella di run (manca weights.json, run.json o un .safetensors)" });
      continue;
    }
    const group = groupFor(root);
    if (e.kind === "binary") group.binaries.push(pick(e));
    else if (e.kind === "snapshot") group.snapshots.push({ ...pick(e), step: e.step });
    else {
      const slot = e.kind;
      if (group[slot]) {
        ignored.push({ path: e.path, reason: `un secondo ${e.name} nello stesso run: si usa ${group[slot].path}` });
        continue;
      }
      group[slot] = pick(e);
    }
  }

  const runs = [...groups.values()].filter((g) => g.manifest || g.safetensors.length || g.trace || g.runInfo);
  let totalBytes = 0;
  for (const g of runs) {
    if (!g.manifest && !g.safetensors.length && !g.trace) g.warnings.push("c'è run.json ma nessun file di pesi");
    g.snapshots.sort((a, b) => a.step - b.step);
    // Si leggono pesi, JSON e snapshot: tutti contano per il limite.
    const all = [g.manifest, g.trace, g.runInfo, g.benchmark, ...g.binaries, ...g.safetensors, ...g.snapshots].filter(Boolean);
    g.bytes = all.reduce((s, e) => s + e.size, 0);
    totalBytes += g.bytes;
    for (const e of all) {
      const json = e.path.toLowerCase().endsWith(".json");
      const max = json ? lim.maxJsonBytes : lim.maxBinaryBytes;
      if (e.size > max) g.errors.push(`${e.path}: ${megabytes(e.size)}, il limite per questo tipo di file è ${megabytes(max)}`);
    }
  }
  runs.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  if (totalBytes > lim.maxTotalBytes) {
    throw new WeightsFormatError(null, `i file selezionati pesano ${megabytes(totalBytes)}: il limite per un caricamento è ${megabytes(lim.maxTotalBytes)}`);
  }
  return { runs, ignored, totalBytes };
}

const pick = (e) => ({ file: e.file, path: e.path, size: e.size });

/** Legge un JSON caricato, con il limite di dimensione (errore chiaro se non è JSON). */
export async function readJsonFile(entry, { limits } = {}) {
  const lim = withLimits(limits);
  const { file, path } = normalizeEntry(entry);
  if (Number(file.size) > lim.maxJsonBytes) throw new WeightsFormatError(path, `${megabytes(file.size)}: troppo grande per un JSON (limite ${megabytes(lim.maxJsonBytes)})`);
  const text = await file.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new WeightsFormatError(path, `non è un JSON valido (${error.message})`);
  }
}

/** Legge un file binario caricato come ArrayBuffer, con il limite di dimensione. */
export async function readBinaryFile(entry, { limits } = {}) {
  const lim = withLimits(limits);
  const { file, path } = normalizeEntry(entry);
  if (Number(file.size) > lim.maxBinaryBytes) throw new WeightsFormatError(path, `${megabytes(file.size)}: oltre il limite di ${megabytes(lim.maxBinaryBytes)}`);
  return file.arrayBuffer();
}

function normalizeEntry(entry) {
  if (entry && entry.file && typeof entry.file.text === "function") return { file: entry.file, path: entry.path ?? filePath(entry.file) };
  return { file: entry, path: filePath(entry) };
}
