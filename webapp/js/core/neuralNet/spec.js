/**
 * Specifica della rete neurale orizzontale: validazione e normalizzazione (logica pura, testata con node).
 *
 * Un tab descrive la sua rete con un oggetto di soli dati (`networkSpec`, formato completo in
 * ARCHITECTURE.md, sezione "Rete neurale orizzontale"). Qui lo si controlla TUTTO prima di disegnare
 * qualcosa: un errore di battitura in un id o una forma sbagliata dei valori veri deve dire dove sta
 * il problema, non produrre una scena muta. Tutti gli errori vengono raccolti in una volta sola
 * (NetworkSpecError.errors), con il percorso del campo: "sections[3].neurons: …".
 *
 * La normalizzazione risolve i valori predefiniti (profondità ereditata, colore del ruolo, slab degli
 * esperti, densità per schema…) così layout.js, links.js e la scena non devono più indovinare niente.
 */
import { ACTIVATION_KINDS, PATTERN_IDS, ROLE_IDS, ROLES } from "./roles.js";

/** Limiti di sicurezza: oltre, la scena diventerebbe inutilizzabile (o il browser si fermerebbe). */
export const LIMITS = Object.freeze({
  sections: 400,
  links: 4096, // l'indice del collegamento di ogni segmento sta in 16 bit (links.js): molto sopra ogni rete vera
  neurons: 4096, // per fetta di profondità (per una griglia: canali)
  slabs: 256,
  depth: 1024, // posizioni in profondità di una sezione
  gridSide: 128,
  experts: 1024,
  heads: 64,
  units: 250_000, // neuroni disegnati in tutto (la prova di carico ne usa ~60.000)
  repeat: 1e12,
});

/** Densità predefinita del campione di collegamenti, per schema (1 = tutti quelli possibili). */
export const DEFAULT_DENSITY = Object.freeze({
  dense: 0.15,
  identity: 1,
  lookup: 1,
  route: 0.35,
  conv: 0.3,
  pool: 1,
  broadcast: 1,
  down: 1,
  up: 1,
  explicit: 1,
  attention: 1,
  residual: 1,
});

const ID_RE = /^[A-Za-z0-9_.:-]+$/;

/** Raccoglie errori e avvisi con il percorso del campo; `count` serve a capire se un blocco ne ha aggiunti. */
class Problems {
  constructor() {
    this.errors = [];
    this.warnings = [];
    this.fail = (path, message) => {
      this.errors.push(`${path}: ${message}`);
    };
    this.broken = new Set(); // id di sezioni scartate per un errore già segnalato
  }

  get count() {
    return this.errors.length;
  }
}
const COLOR_RE = /^#[0-9a-f]{6}$/i;

export class NetworkSpecError extends Error {
  /** @param {string[]} errors */
  constructor(errors) {
    super(`Specifica della rete neurale non valida (${errors.length} ${errors.length === 1 ? "errore" : "errori"}):\n- ${errors.join("\n- ")}`);
    this.name = "NetworkSpecError";
    this.errors = errors;
  }
}

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v) && !ArrayBuffer.isView(v);
const isInt = (v, min, max) => Number.isInteger(v) && v >= min && v <= max;
const isArrayLike = (v) => Array.isArray(v) || ArrayBuffer.isView(v);
/**
 * Float32Array da un elenco di numeri, o null se un elemento non è un numero. Il controllo va fatto
 * PRIMA della copia: Float32Array.from trasforma null in 0 e "1" in 1 senza dire niente (e un NaN
 * passato da JSON.stringify diventa proprio null).
 */
const toFloats = (v) => (v instanceof Float32Array ? v : Array.prototype.some.call(v, (x) => typeof x !== "number") ? null : Float32Array.from(v));
const optString = (v) => (typeof v === "string" ? v : "");
const show = (v) => {
  try {
    const text = JSON.stringify(v);
    return text === undefined ? String(v) : text.length > 40 ? `${text.slice(0, 37)}…` : text;
  } catch {
    return String(v);
  }
};

/**
 * Controlla e normalizza una specifica. Lancia NetworkSpecError con TUTTI gli errori trovati.
 * @param {object} raw la specifica del tab (soli dati)
 * @returns {NormalizedNetwork}
 */
export function normalizeSpec(raw) {
  const problems = new Problems();
  const { errors, warnings } = problems;
  const fail = problems.fail;
  if (!isObject(raw)) throw new NetworkSpecError(["la specifica deve essere un oggetto { title, sections, links, … }"]);

  const title = typeof raw.title === "string" ? raw.title.trim() : "";
  if (!title) fail("title", "serve un titolo (stringa non vuota)");
  for (const key of ["intro", "scale", "note"]) if (raw[key] !== undefined && typeof raw[key] !== "string") fail(key, "deve essere una stringa");
  if (raw.scale === undefined) warnings.push("scale: manca la frase sulla scala (es. \"1 sfera = 1 canale; il modello vero ne ha 1.280\")");
  const seed = raw.seed === undefined ? 1 : raw.seed;
  if (!isInt(seed, 0, 2 ** 32 - 1)) fail("seed", `deve essere un intero tra 0 e 4294967295 (trovato: ${show(raw.seed)})`);

  const axes = { x: "flusso: dall'ingresso all'uscita", y: "neuroni", z: "profondità" };
  if (raw.axes !== undefined) {
    if (!isObject(raw.axes)) fail("axes", "deve essere { x?, y?, z? } con stringhe");
    else for (const k of ["x", "y", "z"]) {
      if (raw.axes[k] === undefined) continue;
      if (typeof raw.axes[k] !== "string") fail(`axes.${k}`, "deve essere una stringa");
      else axes[k] = raw.axes[k];
    }
  }

  // ------------------------------------------------------------ profondità predefinita
  const globalDepth = raw.depth === undefined ? null : parseDepth(raw.depth, "depth", fail, null);
  if (globalDepth && raw.axes?.z === undefined) axes.z = globalDepth.label;

  // ------------------------------------------------------------ sezioni e ripetizioni
  if (!Array.isArray(raw.sections) || raw.sections.length === 0) {
    fail("sections", "serve un elenco non vuoto di sezioni");
    throw new NetworkSpecError(errors);
  }
  if (raw.sections.length > LIMITS.sections) fail("sections", `al massimo ${LIMITS.sections} voci (trovate ${raw.sections.length})`);

  const byId = new Map();
  const sections = [];
  const repeats = [];
  const items = [];
  let units = 0;
  raw.sections.forEach((entry, i) => {
    const path = `sections[${i}]`;
    if (!isObject(entry)) return fail(path, "deve essere un oggetto");
    if (entry.repeat !== undefined) {
      const rep = parseRepeat(entry, path, problems, byId, items, repeats.length);
      if (!rep) return;
      repeats.push(rep);
      items.push(rep);
      if (rep.id) byId.set(rep.id, rep);
      return;
    }
    const section = parseSection(entry, path, problems, globalDepth, byId);
    if (!section) {
      // l'errore della sezione è già nell'elenco: i collegamenti che la citano non devono aggiungere "id sconosciuto"
      if (typeof entry.id === "string") problems.broken.add(entry.id);
      return;
    }
    section.index = sections.length;
    section.order = items.length;
    section.offset = units;
    units += section.units;
    sections.push(section);
    items.push(section);
    byId.set(section.id, section);
  });
  for (const s of sections) {
    if (!s.router) continue;
    const r = byId.get(s.router);
    if (!r || r.kind !== "section" || r.role !== "router") fail(`sections (${s.id}).router`, `"${s.router}" non è una sezione con role: "router"`);
    else if (sliceUnits(r) !== s.experts) fail(`sections (${s.id}).router`, `il router "${r.id}" ha ${sliceUnits(r)} neuroni per posizione: ne serve uno per esperto (${s.experts})`);
    else if (r.depth.count !== s.depth.count && r.depth.count !== 1) fail(`sections (${s.id}).router`, "router ed esperti devono avere la stessa profondità");
    else s.router = r;
  }
  if (units > LIMITS.units) fail("sections", `troppi neuroni da disegnare: ${units} (massimo ${LIMITS.units}); comprimi i blocchi ripetuti con { repeat, of } o riduci neurons/depth`);
  if (sections.length === 0 && !errors.length) fail("sections", "serve almeno una sezione vera (non solo ripetizioni)");

  // ------------------------------------------------------------ collegamenti
  const links = [];
  if (raw.links !== undefined && !Array.isArray(raw.links)) fail("links", "deve essere un elenco");
  else if (Array.isArray(raw.links) && raw.links.length > LIMITS.links) fail("links", `al massimo ${LIMITS.links} voci (trovate ${raw.links.length})`);
  (Array.isArray(raw.links) && raw.links.length <= LIMITS.links ? raw.links : []).forEach((entry, i) => {
    const link = parseLink(entry, `links[${i}]`, problems, byId);
    if (link) {
      link.index = links.length;
      links.push(link);
    }
  });

  // ------------------------------------------------------------ gruppi (parentesi sopra le sezioni)
  const groups = [];
  if (raw.groups !== undefined && !Array.isArray(raw.groups)) fail("groups", "deve essere un elenco");
  (Array.isArray(raw.groups) ? raw.groups : []).forEach((g, i) => {
    const path = `groups[${i}]`;
    if (!isObject(g)) return fail(path, "deve essere un oggetto { label, from, to }");
    if (typeof g.label !== "string" || !g.label) fail(`${path}.label`, "serve un'etichetta");
    const from = byId.get(g.from);
    const to = byId.get(g.to);
    if (problems.broken.has(g.from) || problems.broken.has(g.to)) return;
    if (!from) return fail(`${path}.from`, `id sconosciuto ${show(g.from)}`);
    if (!to) return fail(`${path}.to`, `id sconosciuto ${show(g.to)}`);
    if (from.order > to.order) return fail(path, `"${g.from}" viene dopo "${g.to}" nel flusso`);
    if (g.color !== undefined && !COLOR_RE.test(g.color)) fail(`${path}.color`, "colore #rrggbb");
    groups.push({ index: groups.length, label: optString(g.label), text: optString(g.text), from, to, color: COLOR_RE.test(g.color ?? "") ? g.color : "#7dd3fc" });
  });

  // ------------------------------------------------------------ fonti (solo link http/https: la scheda li rende cliccabili)
  const sources = [];
  if (raw.sources !== undefined && !Array.isArray(raw.sources)) fail("sources", "deve essere un elenco");
  (Array.isArray(raw.sources) ? raw.sources : []).forEach((s, i) => {
    const path = `sources[${i}]`;
    if (!isObject(s) || typeof s.title !== "string" || !s.title) return fail(path, "serve { title, url }");
    if (typeof s.url !== "string" || !/^https?:\/\/[^\s"'<>]+$/i.test(s.url)) return fail(`${path}.url`, "serve un indirizzo http(s)://");
    sources.push({ title: s.title, url: s.url, note: optString(s.note) });
  });

  if (errors.length) throw new NetworkSpecError(errors);

  const simulated = sections.some((s) => !s.values);
  const slices = globalDepth ?? sections.find((s) => s.depth.count > 1)?.depth ?? NO_DEPTH;
  return {
    title,
    intro: optString(raw.intro),
    scale: optString(raw.scale),
    note: optString(raw.note),
    seed,
    axes,
    depth: globalDepth ?? NO_DEPTH,
    // le fette della rete: le posizioni sull'asse z della profondità predefinita (o della prima sezione
    // profonda). Le sezioni con una profondità di altro tipo (es. i token del prompt in una U-Net a
    // griglia) non partecipano alla scelta della fetta.
    slices: { label: slices.label, count: slices.cols, names: sliceNames(slices), kind: slices.kind },
    sections,
    repeats,
    items,
    links,
    groups,
    sources,
    simulated,
    units,
    warnings,
    byId,
  };
}

const NO_DEPTH = Object.freeze({ kind: "none", count: 1, rows: 1, cols: 1, label: "", names: null });

/** Nomi delle posizioni sull'asse z (una per fetta): i nomi dei token, oppure "colonna n" per una griglia. */
function sliceNames(depth) {
  if (depth.kind === "line") return depth.names ? [...depth.names] : Array.from({ length: depth.cols }, (_, i) => `${depth.label} ${i + 1}`);
  if (depth.kind === "grid") return Array.from({ length: depth.cols }, (_, i) => `colonna ${i + 1}`);
  return [];
}

/**
 * Profondità di una sezione (o quella predefinita della rete).
 *   false | undefined | 1     nessuna profondità (un vettore solo)
 *   true                      quella predefinita della rete
 *   n                         n posizioni in fila (token, frame…)
 *   [righe, colonne]          griglia 2D: righe sull'asse y, colonne sull'asse z, un canale per lastra
 *   { count, label, names }   in fila con etichetta e nomi propri
 *   { grid: [r, c], label }   griglia con etichetta
 */
function parseDepth(value, path, fail, inherited) {
  if (value === undefined || value === false || value === 1) return NO_DEPTH;
  if (value === true) {
    if (!inherited) {
      fail(path, "depth: true usa la profondità predefinita della rete, ma la specifica non ha `depth`");
      return null;
    }
    return inherited;
  }
  if (Number.isInteger(value)) {
    if (!isInt(value, 1, LIMITS.depth)) return fail(path, `deve essere tra 1 e ${LIMITS.depth} (trovato: ${value})`), null;
    const same = inherited && inherited.kind === "line" && inherited.count === value;
    return { kind: "line", count: value, rows: 1, cols: value, label: inherited?.kind === "line" ? inherited.label : "posizioni", names: same ? inherited.names : null };
  }
  if (Array.isArray(value)) return parseGrid(value, path, fail, inherited?.kind === "grid" ? inherited.label : "griglia");
  if (isObject(value)) {
    if (value.grid !== undefined) {
      const grid = parseGrid(value.grid, `${path}.grid`, fail, typeof value.label === "string" ? value.label : "griglia");
      return grid;
    }
    if (!isInt(value.count, 1, LIMITS.depth)) return fail(`${path}.count`, `serve un intero tra 1 e ${LIMITS.depth} (trovato: ${show(value.count)})`), null;
    if (value.count === 1) return NO_DEPTH;
    let names = null;
    if (value.names !== undefined) {
      if (!Array.isArray(value.names) || value.names.some((n) => typeof n !== "string")) fail(`${path}.names`, "deve essere un elenco di stringhe");
      else if (value.names.length !== value.count) fail(`${path}.names`, `servono ${value.count} nomi (uno per posizione), trovati ${value.names.length}`);
      else names = value.names;
    }
    return { kind: "line", count: value.count, rows: 1, cols: value.count, label: typeof value.label === "string" && value.label ? value.label : "posizioni", names };
  }
  fail(path, `forma non riconosciuta ${show(value)}: usa true, un numero, [righe, colonne] o { count, label, names }`);
  return null;
}

function parseGrid(value, path, fail, label) {
  if (!Array.isArray(value) || value.length !== 2 || !value.every((n) => isInt(n, 1, LIMITS.gridSide))) {
    fail(path, `una griglia è [righe, colonne] con interi tra 1 e ${LIMITS.gridSide} (trovato: ${show(value)})`);
    return null;
  }
  const [rows, cols] = value;
  if (rows * cols > LIMITS.depth) return fail(path, `griglia troppo grande: ${rows * cols} posizioni (massimo ${LIMITS.depth})`), null;
  return { kind: "grid", count: rows * cols, rows, cols, label, names: null };
}

function parseSection(entry, path, problems, globalDepth, byId) {
  const { fail, warnings } = problems;
  const id = entry.id;
  if (typeof id !== "string" || !ID_RE.test(id)) return fail(`${path}.id`, `serve un id con lettere, cifre, _ . : - (trovato: ${show(id)})`), null;
  if (byId.has(id)) return fail(`${path}.id`, `id "${id}" ripetuto`), null;
  const errorsBefore = problems.count;
  const at = (k) => `${path}.${k} (${id})`;
  if (typeof entry.label !== "string" || !entry.label) fail(at("label"), "serve un'etichetta");
  const role = entry.role ?? "data";
  if (!ROLE_IDS.includes(role)) fail(at("role"), `ruolo sconosciuto ${show(role)}; ruoli: ${ROLE_IDS.join(", ")}`);
  if (!isInt(entry.neurons, 1, LIMITS.neurons)) fail(at("neurons"), `serve un intero tra 1 e ${LIMITS.neurons} (trovato: ${show(entry.neurons)})`);
  for (const key of ["sub", "text", "valueLabel"]) if (entry[key] !== undefined && typeof entry[key] !== "string") fail(at(key), "deve essere una stringa");
  if (entry.color !== undefined && !COLOR_RE.test(entry.color)) fail(at("color"), "colore #rrggbb");
  const depth = parseDepth(entry.depth, at("depth"), fail, globalDepth);
  if (!depth || !Number.isInteger(entry.neurons) || !ROLE_IDS.includes(role)) return null;
  const neurons = entry.neurons;

  // lastre sull'asse x: i canali di una griglia, gli esperti di un MoE, oppure `slabs` esplicito
  let slabs = 1;
  let experts = 0;
  let active = 0;
  let shared = 0;
  if (role === "experts") {
    if (entry.slabs !== undefined) fail(at("slabs"), "negli esperti le lastre sono gli esperti stessi: usa experts / shared");
    if (!isInt(entry.experts, 1, LIMITS.experts)) fail(at("experts"), `serve il numero di esperti disegnati (1…${LIMITS.experts})`);
    experts = entry.experts ?? 1;
    shared = entry.shared ?? 0;
    active = entry.active ?? Math.min(2, experts);
    if (!isInt(shared, 0, 16)) fail(at("shared"), "esperti condivisi: intero tra 0 e 16");
    if (!isInt(active, 1, Math.max(1, experts))) fail(at("active"), `esperti attivi per token: intero tra 1 e ${experts}`);
    if (depth.kind === "grid") fail(at("depth"), "gli esperti lavorano su una profondità in fila (token), non su una griglia");
    slabs = (Number.isInteger(shared) ? shared : 0) + (Number.isInteger(experts) ? experts : 1);
  } else if (entry.experts !== undefined || entry.active !== undefined || entry.shared !== undefined) {
    fail(at("experts"), 'experts / active / shared valgono solo con role: "experts"');
  }
  if (entry.slabs !== undefined && role !== "experts") {
    if (depth.kind === "grid") fail(at("slabs"), "con una griglia ogni canale è già una lastra: togli slabs");
    else if (!isInt(entry.slabs, 1, LIMITS.slabs)) fail(at("slabs"), `serve un intero tra 1 e ${LIMITS.slabs}`);
    else slabs = entry.slabs;
  }
  let slabNames = null;
  if (entry.slabNames !== undefined) {
    if (!Array.isArray(entry.slabNames) || entry.slabNames.some((n) => typeof n !== "string")) fail(at("slabNames"), "elenco di stringhe");
    else if (entry.slabNames.length !== slabs) fail(at("slabNames"), `servono ${slabs} nomi (uno per lastra)`);
    else slabNames = entry.slabNames;
  }

  const heads = entry.heads ?? (role === "attention" ? 1 : 0);
  if (entry.heads !== undefined && !isInt(entry.heads, 1, LIMITS.heads)) fail(at("heads"), `serve un intero tra 1 e ${LIMITS.heads}`);
  else if (heads > 1 && depth.kind !== "grid" && neurons % heads !== 0) warnings.push(`${at("heads")}: ${neurons} neuroni non si dividono in ${heads} head uguali`);

  if (entry.real !== undefined && !(Number.isFinite(entry.real) && entry.real >= 1)) fail(at("real"), "il numero vero di neuroni del modello: un numero ≥ 1");
  else if (entry.real !== undefined && entry.real < neurons) warnings.push(`${at("real")}: il modello vero ha meno neuroni (${entry.real}) di quelli disegnati (${neurons})`);

  const activation = entry.activation ?? defaultActivation(role);
  if (!ACTIVATION_KINDS.includes(activation)) fail(at("activation"), `uno tra ${ACTIVATION_KINDS.join(", ")}`);

  const units = slabs * neurons * depth.count;
  let values = null;
  if (entry.values !== undefined) {
    if (!isArrayLike(entry.values)) fail(at("values"), "valori veri: Float32Array o elenco di numeri (lastra, neurone, posizione)");
    else if (entry.values.length !== units) fail(at("values"), `servono ${units} valori (${slabs} lastre × ${neurons} neuroni × ${depth.count} posizioni), trovati ${entry.values.length}`);
    else {
      values = toFloats(entry.values);
      const bad = values ? values.findIndex((v) => !Number.isFinite(v)) : Array.prototype.findIndex.call(entry.values, (v) => typeof v !== "number");
      if (bad >= 0) fail(at("values"), `valore non numerico o non finito all'indice ${bad} (${show(entry.values[bad])})`);
    }
  }

  let neuronNames = null;
  if (entry.neuronNames !== undefined) {
    if (!Array.isArray(entry.neuronNames) || entry.neuronNames.some((n) => typeof n !== "string")) fail(at("neuronNames"), "elenco di stringhe");
    else if (entry.neuronNames.length > neurons) fail(at("neuronNames"), `al massimo ${neurons} nomi`);
    else neuronNames = entry.neuronNames;
  }

  let routes = null;
  let gates = null;
  if (entry.routes !== undefined) {
    if (role !== "experts") fail(at("routes"), 'routes vale solo con role: "experts"');
    else if (!isArrayLike(entry.routes) || entry.routes.length !== depth.count * active) fail(at("routes"), `servono ${depth.count} × ${active} indici di esperto (posizione per posizione)`);
    else {
      // i valori grezzi si controllano PRIMA di copiarli (Int32Array trasformerebbe 2.5 in 2 e "x" in 0)
      const raw = entry.routes;
      for (let d = 0; d < depth.count; d++) {
        const seen = new Set();
        for (let k = 0; k < active; k++) {
          const e = raw[d * active + k];
          if (!isInt(e, 0, experts - 1)) fail(at("routes"), `indice di esperto ${show(e)} fuori da 0…${experts - 1} (posizione ${d})`);
          else if (seen.has(e)) fail(at("routes"), `esperto ${e} ripetuto alla posizione ${d}`);
          seen.add(e);
        }
      }
      routes = Int32Array.from(raw);
      if (entry.gates !== undefined) {
        if (!isArrayLike(entry.gates) || entry.gates.length !== routes.length) fail(at("gates"), "un peso per ogni voce di routes");
        else if (Array.from(entry.gates).some((g) => typeof g !== "number" || !Number.isFinite(g) || g < 0)) fail(at("gates"), "i pesi del router sono numeri finiti ≥ 0");
        else gates = Float32Array.from(entry.gates);
      }
    }
  }
  if (entry.router !== undefined && (role !== "experts" || typeof entry.router !== "string")) fail(at("router"), 'router: l\'id della sezione router (solo con role: "experts")');
  if (problems.count !== errorsBefore) return null;

  return {
    kind: "section",
    id,
    label: entry.label,
    sub: optString(entry.sub),
    text: optString(entry.text),
    valueLabel: optString(entry.valueLabel),
    role,
    color: entry.color ?? ROLES[role].color,
    neurons,
    slabs,
    slabNames,
    heads,
    experts,
    active,
    shared,
    routes,
    gates,
    router: entry.router ?? null, // id per ora: diventa la sezione dopo aver letto tutte le sezioni
    depth,
    units,
    real: entry.real ?? null,
    activation,
    values,
    neuronNames,
    index: -1,
    order: -1,
    offset: 0,
  };
}

/** Come si leggono i valori se la specifica non lo dice: le uscite sono probabilità, il router anche. */
function defaultActivation(role) {
  if (role === "output" || role === "router") return "probs";
  if (role === "codebook") return "onehot";
  if (role === "experts") return "positive";
  return "signed";
}

function parseRepeat(entry, path, problems, byId, items, index) {
  const { fail } = problems;
  if (!isInt(entry.repeat, 2, LIMITS.repeat)) return fail(`${path}.repeat`, `quante volte si ripete il blocco: intero ≥ 2 (trovato: ${show(entry.repeat)})`), null;
  if (!Array.isArray(entry.of) || entry.of.length === 0) return fail(`${path}.of`, "serve l'elenco degli id delle sezioni che si ripetono"), null;
  const of = [];
  for (const id of entry.of) {
    const s = byId.get(id);
    if (!s) return fail(`${path}.of`, `"${id}" non è una sezione già elencata prima di questa ripetizione`), null;
    if (s.kind !== "section") return fail(`${path}.of`, `"${id}" è una ripetizione: non si possono annidare`), null;
    of.push(s);
  }
  of.sort((a, b) => a.order - b.order);
  for (let i = 1; i < of.length; i++) {
    if (of[i].order !== of[i - 1].order + 1) return fail(`${path}.of`, "le sezioni ripetute devono essere consecutive nel flusso"), null;
  }
  if (of[of.length - 1].order !== items.length - 1) return fail(`${path}.of`, "la ripetizione va subito dopo l'ultima sezione del blocco"), null;
  if (entry.id !== undefined && (typeof entry.id !== "string" || !ID_RE.test(entry.id) || byId.has(entry.id))) return fail(`${path}.id`, "id non valido o già usato"), null;
  return {
    kind: "repeat",
    id: entry.id ?? `repeat-${index}`,
    count: entry.repeat,
    of,
    label: optString(entry.label) || `× ${entry.repeat} blocchi uguali`,
    text: optString(entry.text),
    index,
    order: items.length,
  };
}

/** Numero di unità in una fetta di profondità (lastre × neuroni, o canali di una griglia). */
export const sliceUnits = (s) => s.slabs * s.neurons;

function parseLink(entry, path, problems, byId) {
  const { fail, warnings } = problems;
  if (!isObject(entry)) return fail(path, "deve essere un oggetto { from, to, pattern }"), null;
  const pattern = entry.pattern;
  if (!PATTERN_IDS.includes(pattern)) return fail(`${path}.pattern`, `schema sconosciuto ${show(pattern)}; schemi: ${PATTERN_IDS.join(", ")}`), null;
  const label = `${path} (${pattern})`;
  const section = (id, key) => {
    const s = byId.get(id);
    if (!s && problems.broken.has(id)) return null; // sezione già segnalata come sbagliata
    if (!s) fail(`${label}.${key}`, `id sconosciuto ${show(id)}`);
    else if (s.kind !== "section") fail(`${label}.${key}`, `"${id}" è una ripetizione: collega le sue sezioni`);
    else return s;
    return null;
  };
  let from;
  let to;
  const within = entry.within !== undefined;
  if (within) {
    if (pattern !== "attention") return fail(label, '"within" vale solo per l\'attention (self-attention dentro una sezione)'), null;
    from = to = section(entry.within, "within");
  } else {
    from = section(entry.from, "from");
    to = section(entry.to, "to");
    if (from && to && from === to) return fail(label, "from e to sono la stessa sezione (per la self-attention usa within)"), null;
  }
  if (!from || !to) return null;
  const density = entry.density ?? DEFAULT_DENSITY[pattern];
  if (!(Number.isFinite(density) && density > 0 && density <= 1)) return fail(`${label}.density`, `tra 0 (escluso) e 1 (trovato: ${show(entry.density)})`), null;

  const D = (s) => s.depth.count;
  const link = { kind: "link", pattern, from, to, within, density, label: optString(entry.label), text: optString(entry.text), index: -1 };
  const errorsBefore = problems.count;
  switch (pattern) {
    case "dense":
    case "identity":
    case "explicit":
      if (D(from) !== D(to) && D(from) !== 1 && D(to) !== 1) {
        if (pattern === "dense") warnings.push(`${label}: profondità diverse (${D(from)} e ${D(to)}): le posizioni si abbinano in proporzione`);
        else fail(label, `profondità diverse (${D(from)} e ${D(to)}): servono uguali (o una delle due senza profondità)`);
      }
      if (pattern === "identity" && sliceUnits(from) !== sliceUnits(to)) warnings.push(`${label}: ${sliceUnits(from)} e ${sliceUnits(to)} unità per posizione: si collegano le prime ${Math.min(sliceUnits(from), sliceUnits(to))}`);
      break;
    case "lookup":
      if (D(from) !== D(to)) fail(label, `il lookup lavora posizione per posizione: servono profondità uguali (${D(from)} e ${D(to)})`);
      break;
    case "attention": {
      if (D(from) < 2 || D(to) < 2) fail(label, "l'attention collega posizioni della profondità: servono sezioni con depth ≥ 2");
      link.causal = Boolean(entry.causal);
      if (link.causal && !within) {
        warnings.push(`${label}: causal non ha senso nella cross-attention (ignorato)`);
        link.causal = false;
      }
      if (link.causal && D(from) !== D(to)) fail(label, "causale richiede la stessa profondità");
      const heads = entry.heads ?? (to.heads || 1);
      if (!isInt(heads, 1, LIMITS.heads)) fail(`${label}.heads`, `intero tra 1 e ${LIMITS.heads}`);
      link.heads = heads;
      link.weights = null;
      if (entry.weights !== undefined) {
        const n = heads * D(to) * D(from);
        if (!isArrayLike(entry.weights) || entry.weights.length !== n) fail(`${label}.weights`, `servono ${n} pesi (head × query ${D(to)} × key ${D(from)})`);
        else {
          link.weights = toFloats(entry.weights);
          if (!link.weights || link.weights.some((w) => !(w >= 0) || !Number.isFinite(w))) fail(`${label}.weights`, "i pesi di attention sono probabilità: numeri finiti ≥ 0");
        }
      }
      break;
    }
    case "route": {
      const into = to.role === "experts";
      const outOf = from.role === "experts";
      if (into === outOf) fail(label, 'l\'instradamento collega una sezione e degli esperti (role: "experts"): esattamente una delle due');
      else {
        const experts = into ? to : from;
        const other = into ? from : to;
        if (D(other) !== D(experts) && D(other) !== 1) fail(label, `profondità diverse (${D(other)} e ${D(experts)})`);
        link.experts = experts;
        link.direction = into ? "dispatch" : "combine";
      }
      break;
    }
    case "conv": {
      const kernel = entry.kernel ?? 3;
      if (!isInt(kernel, 1, 9) || kernel % 2 === 0) fail(`${label}.kernel`, "dimensione del kernel: intero dispari tra 1 e 9");
      link.kernel = kernel;
      if (from.depth.kind !== to.depth.kind || from.depth.rows !== to.depth.rows || from.depth.cols !== to.depth.cols) {
        fail(label, "la convoluzione collega due sezioni con la stessa griglia (o la stessa fila): per cambiare dimensione usa down/up");
      }
      break;
    }
    case "pool":
      if (D(from) < 2 || D(to) !== 1) fail(label, "il pooling va da una sezione con profondità a una senza (depth: false)");
      break;
    case "broadcast":
      if (D(from) !== 1 || D(to) < 2) fail(label, "il broadcast va da una sezione senza profondità a una con profondità");
      break;
    case "down":
    case "up": {
      const big = pattern === "down" ? from.depth : to.depth;
      const small = pattern === "down" ? to.depth : from.depth;
      const factor = entry.factor ?? (small.cols > 0 ? big.cols / small.cols : 0);
      const ok = from.depth.kind === to.depth.kind && from.depth.kind !== "none" && Number.isInteger(factor) && factor >= 2
        && small.cols * factor === big.cols && (big.kind === "line" || small.rows * factor === big.rows);
      if (!ok) fail(label, `${pattern === "down" ? "la griglia di arrivo" : "la griglia di partenza"} deve essere più piccola di un fattore intero ≥ 2 in ogni lato (trovate ${big.rows}×${big.cols} e ${small.rows}×${small.cols})`);
      link.factor = factor;
      break;
    }
    case "residual":
      if (from.order >= to.order) fail(label, "il residuo salta in avanti: from deve venire prima di to");
      if (D(from) !== D(to)) fail(label, `profondità diverse (${D(from)} e ${D(to)})`);
      break;
    default:
      break;
  }
  if (pattern === "explicit") link.pairs = parsePairs(entry, label, fail, from, to);
  if (pattern === "explicit" || pattern === "dense") {
    const slice = entry.slice ?? (pattern === "explicit" ? "selected" : "all");
    if (!(slice === "selected" || slice === "all" || isInt(slice, 0, Math.max(D(from), D(to)) - 1))) fail(`${label}.slice`, '"selected", "all" o l\'indice di una posizione');
    link.slice = slice;
  }
  for (const key of ["fromSlab", "toSlab"]) {
    const v = entry[key];
    const owner = key === "fromSlab" ? from : to;
    if (v === undefined) continue;
    if (!(v === "all" || isInt(v, 0, owner.slabs - 1))) fail(`${label}.${key}`, `"all" o un indice di lastra tra 0 e ${owner.slabs - 1}`);
  }
  link.fromSlab = entry.fromSlab ?? (from.role === "experts" ? "all" : from.slabs - 1); // si esce dall'ultima lastra…
  link.toSlab = entry.toSlab ?? (to.role === "experts" ? "all" : 0); // …e si entra dalla prima
  if (problems.count !== errorsBefore) return null;
  return link;
}

/**
 * Coppie vere: { from: [...], to: [...], weight: [...] } oppure [[from, to, peso], …]. Gli indici sono
 * dentro la fetta di profondità (lastra × neuroni + neurone; per una griglia: il canale), i pesi i
 * numeri veri della matrice. I valori grezzi si controllano PRIMA di copiarli in array tipizzati
 * (Int32Array trasformerebbe 2.5 in 2 e "x" in 0 senza dire niente).
 */
function parsePairs(entry, label, fail, from, to) {
  const p = entry.pairs;
  let rawFrom;
  let rawTo;
  let rawWeight;
  if (Array.isArray(p)) {
    if (p.some((t) => !Array.isArray(t) || t.length < 2)) return fail(`${label}.pairs`, "ogni coppia è [from, to, peso]"), null;
    rawFrom = p.map((t) => t[0]);
    rawTo = p.map((t) => t[1]);
    rawWeight = p.map((t) => t[2] ?? 1);
  } else if (isObject(p) && isArrayLike(p.from) && isArrayLike(p.to) && p.from.length === p.to.length) {
    rawFrom = p.from;
    rawTo = p.to;
    rawWeight = p.weight ?? new Float32Array(p.from.length).fill(1);
    if (!isArrayLike(rawWeight) || rawWeight.length !== rawFrom.length) return fail(`${label}.pairs.weight`, "un peso per coppia"), null;
  } else {
    return fail(`${label}.pairs`, "servono le coppie vere: { from: [...], to: [...], weight: [...] } oppure [[from, to, peso], …]"), null;
  }
  const maxA = sliceUnits(from);
  const maxB = sliceUnits(to);
  for (let i = 0; i < rawFrom.length; i++) {
    if (!isInt(rawFrom[i], 0, maxA - 1)) return fail(`${label}.pairs`, `indice di partenza ${show(rawFrom[i])} fuori da 0…${maxA - 1} (coppia ${i})`), null;
    if (!isInt(rawTo[i], 0, maxB - 1)) return fail(`${label}.pairs`, `indice di arrivo ${show(rawTo[i])} fuori da 0…${maxB - 1} (coppia ${i})`), null;
    if (typeof rawWeight[i] !== "number" || !Number.isFinite(rawWeight[i])) return fail(`${label}.pairs`, `peso non finito (coppia ${i})`), null;
  }
  return { from: Int32Array.from(rawFrom), to: Int32Array.from(rawTo), weight: Float32Array.from(rawWeight), count: rawFrom.length };
}

/** Testo leggibile di una NetworkSpecError (o di un errore qualunque) per la scheda. */
export function describeSpecError(error) {
  return error instanceof NetworkSpecError ? error.errors : [String(error?.message ?? error)];
}

/**
 * @typedef {ReturnType<typeof normalizeSpec>} NormalizedNetwork
 */
