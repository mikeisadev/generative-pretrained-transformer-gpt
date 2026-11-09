/**
 * Logica PURA della vista "Architettura LLM moderni completa" (niente DOM, niente three.js; test in
 * tests/modern-llm-full.test.mjs): dai dati di un modello (full/models/*.js) alle cose da disegnare.
 *
 *   compressLayers(keys)      le ripetizioni come blocchi "× N" (vale per qualsiasi schema di layer)
 *   layerList(model)          i layer con numero, tipo e flag
 *   layerParams / paramTotals parametri per layer, totali, attivi per token, ripartizione per componente
 *   addOnTotal(model)         parametri aggiunti a quelli dei conteggi ufficiali (indexer DSA di Kimi K3.1)
 *   buildStrip(model, opts)   i blocchi del livello Blocchi (compressi o srotolati), parentesi e segmenti
 *   anchorsOf(model)          le ancore della scena (livello:pezzo) a cui puntano percorso e meccanismi
 *   formatCount, formatGB, formatBytes, formatLayers   numeri all'italiana
 */

/** Spareggio tra descrizioni con lo stesso numero di pezzi (vedi compressLayers). */
const TIE_BREAK = 0.01;

/**
 * La descrizione più corta di una sequenza di tipi di layer come ripetizioni.
 *
 * Perché la programmazione dinamica e non un algoritmo goloso: nel decoder di DeepSeek-V4.1-Flash
 * (Full, 3 Reuse, 4 × [Reindex + 3 Reuse]) il goloso trova prima 4 × [3 Reuse + Reindex], sfasato di
 * un layer; qui si cerca il minimo su tutta la sequenza. Costi: un "run" (stesso tipo ripetuto) vale 1;
 * un gruppo (unità di più layer ripetuta almeno 2 volte) vale 1 + i run della sua unità. A parità di
 * costo vince l'unità che comincia con il suo tipo più raro: la "testa" che dà il ritmo al gruppo
 * (Full, Reindex, Gated MLA), come la si descrive a voce.
 *
 * @param {ArrayLike<string>} keys un tipo per layer (es. la stringa "SSFRRRRR…")
 * @param {{maxUnit?: number}} [o] lunghezza massima dell'unità di un gruppo
 * @returns {Array<{kind: "run", key: string, start: number, count: number} |
 *   {kind: "group", start: number, repeat: number, unitLength: number, unit: {key: string, offset: number, count: number}[]}>}
 *   posizioni (start, offset) relative all'inizio della sequenza
 */
export function compressLayers(keys, { maxUnit = 8 } = {}) {
  const k = Array.from(keys);
  const n = k.length;
  const best = new Float64Array(n + 1);
  const choice = new Array(n + 1);
  for (let i = n - 1; i >= 0; i--) {
    best[i] = Infinity;
    // run: lo stesso tipo da i in avanti (prima i più lunghi: a parità di costo vince il run più lungo)
    let run = 1;
    while (i + run < n && k[i + run] === k[i]) run++;
    for (let c = run; c >= 1; c--) consider(i, 1 + best[i + c], { kind: "run", key: k[i], start: i, count: c });
    // gruppi: unità di u layer ripetuta r ≥ 2 volte di fila
    for (let u = 2; u <= maxUnit && i + 2 * u <= n; u++) {
      let r = 1;
      while (i + (r + 1) * u <= n && sameBlock(k, i, i + r * u, u)) r++;
      if (r < 2) continue;
      const unit = runsOf(k, i, u);
      if (unit.length < 2) continue; // un'unità di un solo tipo è già un run
      const cost = 1 + unit.length + (startsWithRarest(k, i, u) ? 0 : TIE_BREAK);
      for (let reps = r; reps >= 2; reps--) consider(i, cost + best[i + reps * u], { kind: "group", start: i, repeat: reps, unitLength: u, unit });
    }
  }
  const items = [];
  for (let i = 0; i < n; ) {
    const c = choice[i];
    items.push(c);
    i += c.kind === "run" ? c.count : c.repeat * c.unitLength;
  }
  return items;

  function consider(i, cost, option) {
    if (cost < best[i] - 1e-9) {
      best[i] = cost;
      choice[i] = option;
    }
  }
}

function sameBlock(k, a, b, u) {
  for (let j = 0; j < u; j++) if (k[a + j] !== k[b + j]) return false;
  return true;
}

/** Run consecutivi dell'unità k[i .. i+u): [{key, offset, count}]. */
function runsOf(k, i, u) {
  const runs = [];
  for (let j = 0; j < u; j++) {
    const last = runs.at(-1);
    if (last && last.key === k[i + j]) last.count++;
    else runs.push({ key: k[i + j], offset: j, count: 1 });
  }
  return runs;
}

function startsWithRarest(k, i, u) {
  const counts = new Map();
  for (let j = 0; j < u; j++) counts.set(k[i + j], (counts.get(k[i + j]) ?? 0) + 1);
  return counts.get(k[i]) === Math.min(...counts.values());
}

// ------------------------------------------------------------------------------------------------
// Layer e parametri
// ------------------------------------------------------------------------------------------------

/** Numeri di layer di un flag o di una selezione ("all", un codice di tipo, un elenco). */
export function selectLayers(model, selector) {
  const all = layerList(model);
  if (selector === "all") return all.map((l) => l.number);
  if (typeof selector === "string") return all.filter((l) => selector.includes(l.kind)).map((l) => l.number);
  return Array.isArray(selector) ? [...selector] : [];
}

const layerCache = new WeakMap();

/** I layer del modello: [{position, number, kind, flags: Set}] (number = numerazione del modello). */
export function layerList(model) {
  if (layerCache.has(model)) return layerCache.get(model);
  const { base, kinds, flags } = model.layers;
  const list = Array.from(kinds, (kind, position) => ({ position, number: base + position, kind, flags: new Set() }));
  for (const [flag, { layers }] of Object.entries(flags ?? {})) for (const n of layers) list[n - base]?.flags.add(flag);
  layerCache.set(model, list);
  return list;
}

/** Il layer con quel numero (o null). */
export function layerByNumber(model, number) {
  return layerList(model)[number - model.layers.base] ?? null;
}

/**
 * Parametri di un layer: totali (tutti gli esperti) e attivi per token (solo i top-k), con le parti.
 * Un layer con il flag "dense" ha una FFN densa al posto del MoE (Kimi, layer 1).
 */
export function layerParams(model, layer) {
  const p = model.params;
  const attention = p.attention[layer.kind] ?? 0;
  const isDense = Boolean(p.dense?.layers.includes(layer.number));
  const moe = p.moe;
  const routed = isDense ? 0 : moe.routed * moe.expert;
  const routedActive = isDense ? 0 : moe.active * moe.expert;
  const shared = isDense ? 0 : moe.shared * moe.sharedExpert;
  const router = isDense ? 0 : moe.router;
  const latent = isDense ? 0 : moe.latent;
  const dense = isDense ? p.dense.params : 0;
  let flagged = 0;
  for (const flag of layer.flags) flagged += p.flagParams?.[flag] ?? 0;
  const common = attention + shared + router + latent + dense + (p.residual ?? 0) + (p.norms ?? 0) + flagged;
  return {
    total: common + routed,
    active: common + routedActive,
    parts: { attention, routed, shared, router, latent, dense, residual: p.residual ?? 0, norms: p.norms ?? 0, flagged },
  };
}

/** Etichette dei componenti della ripartizione dei parametri (pannello "Pesi" e grafico). */
export const COMPONENT_LABELS = {
  routed: "Esperti instradati",
  attention: "Attention",
  shared: "Esperti condivisi",
  embedding: "Embedding",
  head: "Head",
  latent: "Proiezioni latenti del MoE",
  dense: "FFN densa",
  router: "Router",
  flagged: "Proiezioni Engram",
  residual: "Residuo mHC",
  norms: "Norme",
  extra: "Marcatori immagine",
};

const totalsCache = new WeakMap();

/**
 * Totali del modello: backbone (embedding e head compresi), attivi per token in decode (embedding
 * escluso: è una lettura, come nelle convenzioni DeepSeek/OpenAI) e in prefill (solo il segmento
 * `prefillSegment`, se il modello lo dichiara), ripartizione per componente con la quota del backbone.
 */
export function paramTotals(model) {
  if (totalsCache.has(model)) return totalsCache.get(model);
  const p = model.params;
  const sums = { routed: 0, attention: 0, shared: 0, latent: 0, dense: 0, router: 0, flagged: 0, residual: 0, norms: 0 };
  let active = p.head + (p.finalNorm ?? 0);
  const byKind = {};
  for (const layer of layerList(model)) {
    const lp = layerParams(model, layer);
    for (const key of Object.keys(sums)) sums[key] += lp.parts[key];
    active += lp.active;
    byKind[layer.kind] ??= { layers: 0, perLayer: lp.total };
    byKind[layer.kind].layers++;
  }
  sums.norms += p.finalNorm ?? 0;
  const parts = { ...sums, embedding: p.embedding, head: p.head, extra: p.extra ?? 0 };
  const backbone = Object.values(parts).reduce((a, b) => a + b, 0);
  let prefill = null;
  const segment = model.layers.segments.find((s) => s.id === model.layers.prefillSegment);
  if (segment) {
    prefill = 0;
    for (const layer of layerList(model)) if (layer.number >= segment.from && layer.number <= segment.to) prefill += layerParams(model, layer).active;
  }
  const components = Object.entries(parts)
    .filter(([, value]) => value > 0)
    .map(([id, value]) => ({ id, label: id === "flagged" ? flaggedLabel(model) : COMPONENT_LABELS[id], params: value, share: value / backbone }))
    .sort((a, b) => b.params - a.params);
  const outside = (p.outside ?? []).map((o) => ({ ...o }));
  const checkpoint = backbone + outside.reduce((a, o) => a + (o.approx ? 0 : o.params), 0);
  const result = { backbone, active, prefill, components, outside, checkpoint, byKind, layers: layerList(model).length };
  totalsCache.set(model, result);
  return result;
}

/**
 * Parametri che il modello aggiunge al backbone dei conteggi ufficiali, mostrati a parte: Kimi K3.1 Pro
 * mette l'indexer DSA nei 24 layer MLA, che non è nei 2,78T / 104,2B del report di Kimi K3.
 * @returns {{label: string, kind: string, perLayer: number, layers: number, params: number} | null}
 */
export function addOnTotal(model) {
  const a = model.params.addOn;
  if (!a) return null;
  const layers = layerList(model).filter((l) => l.kind === a.kind).length;
  return { label: a.label, kind: a.kind, perLayer: a.params, layers, params: a.params * layers };
}

function flaggedLabel(model) {
  const names = Object.keys(model.params.flagParams ?? {});
  return names.length === 1 && names[0] === "engram" ? COMPONENT_LABELS.flagged : "Parametri dei layer speciali";
}

// ------------------------------------------------------------------------------------------------
// Livello Blocchi: la striscia dell'architettura
// ------------------------------------------------------------------------------------------------

/**
 * I blocchi del livello Blocchi, da sinistra a destra.
 * @param {object} model
 * @param {{expanded?: Set<string> | "all"}} [o] gruppi srotolati nei layer veri (id "rep-<n>")
 * @returns {{
 *   blocks: object[],   // {id, type: "input"|"layer"|"output", kind?, title, sub, text, color, layers, count, repeat, params, paramsLabel, flags, rep, item?}
 *   groups: object[],   // parentesi "× N": {id, from, to, repeat, layers, params, label, sub, expanded}
 *   segments: object[], // parentesi dei segmenti (encoder, decoder…): {id, label, sub, color, from, to}
 *   items: object[],    // uscita di compressLayers con id e numeri di layer
 * }}  from/to = indici dei blocchi
 */
export function buildStrip(model, { expanded = new Set() } = {}) {
  const list = layerList(model);
  const isExpanded = (id) => expanded === "all" || expanded.has(id);
  const items = compressLayers(list.map((l) => l.kind)).map((item, i) => ({ ...item, id: `rep-${i}` }));
  const blocks = [];
  const groups = [];
  for (const b of model.input) blocks.push(ioBlock(b, "input"));
  for (const item of items) {
    if (item.kind === "run") {
      const layers = list.slice(item.start, item.start + item.count);
      if (item.count > 1 && !isExpanded(item.id)) {
        const first = blocks.length;
        blocks.push(layerBlock(model, layers, { count: item.count, repeat: 1, item: item.id }));
        groups.push(groupInfo(model, item, first, first, layers, false));
      } else {
        const first = blocks.length;
        for (const l of layers) blocks.push(layerBlock(model, [l], { item: item.id }));
        if (item.count > 1) groups.push(groupInfo(model, item, first, blocks.length - 1, layers, true));
      }
      continue;
    }
    // gruppo: unità ripetuta
    const first = blocks.length;
    const span = list.slice(item.start, item.start + item.repeat * item.unitLength);
    if (isExpanded(item.id)) {
      for (const l of span) blocks.push(layerBlock(model, [l], { item: item.id }));
    } else {
      for (const run of item.unit) {
        // lo stesso pezzo dell'unità in tutte le ripetizioni (es. i Full 2, 8, 14)
        const layers = [];
        for (let r = 0; r < item.repeat; r++) for (let j = 0; j < run.count; j++) layers.push(list[item.start + r * item.unitLength + run.offset + j]);
        blocks.push(layerBlock(model, layers, { count: run.count, repeat: item.repeat, item: item.id }));
      }
    }
    groups.push(groupInfo(model, item, first, blocks.length - 1, span, isExpanded(item.id)));
  }
  for (const b of model.output) blocks.push(ioBlock(b, "output"));
  const segments = (model.layers.segments ?? []).map((s) => {
    const inside = blocks.map((b, i) => (b.type === "layer" && b.layers.every((n) => n >= s.from && n <= s.to) ? i : -1)).filter((i) => i >= 0);
    return { ...s, from: inside[0], to: inside.at(-1) };
  });
  return { blocks, groups, segments, items: items.map((it) => ({ ...it, layers: rangeOf(list, it) })) };
}

function rangeOf(list, item) {
  const len = item.kind === "run" ? item.count : item.repeat * item.unitLength;
  return list.slice(item.start, item.start + len).map((l) => l.number);
}

function ioBlock(b, type) {
  return { ...b, type, layers: [], count: 1, repeat: 1, flags: [], paramsLabel: b.params ? formatCount(b.params) : "" };
}

function layerBlock(model, layers, { count = 1, repeat = 1, item = null }) {
  const first = layers[0];
  const kind = model.kinds[first.kind];
  const numbers = layers.map((l) => l.number);
  // il valore più frequente: un'eccezione (es. il layer 1 di Kimi con la FFN densa) è segnata dai flag
  const per = mostFrequent(layers.map((l) => layerParams(model, l).total));
  // il layer con un flag che cambia i parametri (Engram, FFN densa) dice i suoi: il numero del blocco è
  // quello degli altri layer, e senza la nota "2 × 13,75B" sembrerebbe valere anche per il layer 1
  const flags = [];
  for (const [flag, info] of Object.entries(model.layers.flags ?? {})) {
    const hit = layers.filter((l) => l.flags.has(flag));
    if (!hit.length) continue;
    const own = hit.map((l) => layerParams(model, l).total);
    const note = own.some((t) => t !== per) ? (hit.length === 1 ? formatCount(own[0]) : "parametri diversi") : null;
    flags.push({ id: flag, label: info.label, color: info.color, layers: hit.map((l) => l.number), note });
  }
  return {
    id: `layer-${numbers[0]}${layers.length > 1 ? `-x${layers.length}` : ""}`,
    type: "layer",
    kind: first.kind,
    title: kind.label,
    sub: layerSummary(numbers),
    text: kind.text,
    color: kind.color,
    layers: numbers,
    rep: numbers[0],
    count,
    repeat,
    item,
    params: per * count, // una ripetizione: il totale del gruppo sta nella sua parentesi
    paramsLabel: count > 1 ? `${count} × ${formatCount(per)}` : formatCount(per),
    flags,
  };
}

function mostFrequent(values) {
  const counts = new Map();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  let best = values[0];
  for (const [v, c] of counts) if (c > counts.get(best)) best = v;
  return best;
}

function groupInfo(model, item, from, to, layers, expanded) {
  const total = layers.reduce((a, l) => a + layerParams(model, l).total, 0);
  const numbers = layers.map((l) => l.number);
  const repeat = item.kind === "group" ? item.repeat : item.count;
  return {
    id: item.id,
    from,
    to,
    repeat,
    kind: item.kind,
    layers: numbers,
    params: total,
    label: `× ${repeat}`,
    sub: `layer ${formatLayers(numbers)} · ${formatCount(total)}`,
    expanded,
  };
}

/** Il blocco che rappresenta un layer (anche dentro un gruppo compresso): indice o -1. */
export function blockOfLayer(strip, number) {
  return strip.blocks.findIndex((b) => b.type === "layer" && b.layers.includes(number));
}

// ------------------------------------------------------------------------------------------------
// Ancore della scena
// ------------------------------------------------------------------------------------------------

/**
 * Le ancore (livello:pezzo) che la scena sa inquadrare per un modello:
 *   macro:<id della pipeline>, macro:capabilities, macro:weights-share
 *   blocks:<id di ingresso, uscita, ramo>, blocks:cache-<id>, blocks:segment-<id>, blocks:layers
 *   detail:layer, detail:residual, detail:experts, detail:<id di ogni scheda dell'anatomia>
 */
export function anchorsOf(model) {
  const set = new Set(["macro:capabilities", "macro:weights-share", "blocks:layers", "detail:layer", "detail:residual", "detail:experts"]);
  for (const s of model.pipeline) set.add(`macro:${s.id}`);
  for (const b of [...model.input, ...model.output, ...model.branches]) set.add(`blocks:${b.id}`);
  for (const c of model.caches) set.add(`blocks:cache-${c.id}`);
  for (const s of model.layers.segments ?? []) set.add(`blocks:segment-${s.id}`);
  for (const card of anatomyCards(model)) set.add(`detail:${card.id}`);
  return set;
}

/** Tutte le schede dell'anatomia di un modello (per ogni tipo di layer), con la sezione di appartenenza. */
export function anatomyCards(model) {
  const a = model.anatomy;
  const cards = [];
  for (const row of a.attention.rows) for (const c of row.cards) cards.push({ ...c, section: "attention", row: row.id, rowIn: row.in });
  cards.push({ ...a.attention.core, section: "attention" });
  for (const c of a.attention.output) cards.push({ ...c, section: "attention", row: "output" });
  const f = a.ffn;
  for (const c of [f.router, f.expert, f.sum, f.dense, ...(f.latent ?? []), ...(f.shared ?? [])]) if (c) cards.push({ ...c, section: "ffn" });
  for (const c of a.branch?.cards ?? []) cards.push({ ...c, section: "branch" });
  return cards;
}

/**
 * Stato di una scheda dell'anatomia in un layer: "on" (lavora), "ghost" (riusata da un altro layer:
 * tratteggiata), "off" (non c'è). Regole dei dati: `in` / `ghost` sono stringhe di codici di tipo;
 * `when` / la riga con `in` limitano per flag o per tipo.
 */
export function cardState(card, layer) {
  if (card.when && !layer.flags.has(card.when)) return "off";
  if (card.rowIn && !card.rowIn.includes(layer.kind)) return "off";
  if (!card.in) return "on";
  if (card.in.includes(layer.kind)) return "on";
  if (card.ghost?.includes(layer.kind)) return "ghost";
  return "off";
}

/** Il layer da mostrare per un'ancora del Dettaglio (dato del meccanismo, o il primo layer che la usa). */
export function layerForAnchor(model, anchor, preferred = null) {
  const id = anchor.split(":")[1];
  const list = layerList(model);
  if (preferred !== null && layerByNumber(model, preferred)) return preferred;
  const card = anatomyCards(model).find((c) => c.id === id);
  if (!card) return list[0].number;
  const hit = list.find((l) => cardState(card, l) === "on");
  return (hit ?? list[0]).number;
}

// ------------------------------------------------------------------------------------------------
// Numeri all'italiana
// ------------------------------------------------------------------------------------------------

const it = (n, digits) => n.toLocaleString("it-IT", { minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: true });

/**
 * Conteggi di parametri: "551,88B", "137,3M", "15.360". Con `digits` si fissano le cifre decimali.
 * "B" = miliardi, come nel resto del tab ("671B parametri").
 */
export function formatCount(n, { digits } = {}) {
  const a = Math.abs(n);
  if (a >= 1e12) return `${it(n / 1e12, digits ?? 2)}T`;
  if (a >= 1e9) return `${it(n / 1e9, digits ?? (a >= 1e11 ? 1 : 2))}B`;
  if (a >= 1e6) return `${it(n / 1e6, digits ?? 1)}M`;
  return it(n, 0);
}

/** Gigabyte con una cifra decimale: "510,3 GB". */
export function formatGB(gb, digits = 1) {
  return `${it(gb, digits)} GB`;
}

/** Byte per token: "890 B", "70.272 B". */
export function formatBytes(bytes) {
  return `${it(bytes, 0)} B`;
}

/** Percentuale con una cifra: "98,5%". */
export function formatShare(x, digits = 1) {
  return `${it(100 * x, digits)}%`;
}

/**
 * Riga breve dei layer di un blocco: l'elenco se è corto ("layer 3-7, 9-13, 15-19"), altrimenti
 * quanti e da dove a dove ("69 layer · 1 … 91"): sulla lastra deve restare leggibile.
 */
export function layerSummary(numbers) {
  const list = formatLayers(numbers);
  if (numbers.length === 1) return `layer ${list}`;
  if (list.length <= 22) return `layer ${list}`;
  return `${numbers.length} layer · ${Math.min(...numbers)} … ${Math.max(...numbers)}`;
}

/** Elenco di layer come intervalli: [3,4,5,6,7,9,10] → "3-7, 9-10". */
export function formatLayers(numbers) {
  const sorted = [...numbers].sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < sorted.length; ) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    parts.push(j > i ? `${sorted[i]}-${sorted[j]}` : String(sorted[i]));
    i = j + 1;
  }
  return parts.join(", ");
}
