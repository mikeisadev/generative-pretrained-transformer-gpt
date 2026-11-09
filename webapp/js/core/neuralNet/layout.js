/**
 * Disposizione della rete orizzontale con profondità (logica pura, testata con node).
 *
 *   x  il flusso: le sezioni da sinistra a destra, ognuna con una o più lastre (lastre = parti di un
 *      blocco, canali di una griglia, esperti di un MoE)
 *   y  i neuroni della sezione (per una griglia 2D: le righe della griglia)
 *   z  la profondità vera: i token, le colonne della griglia, i frame nel tempo. Centrata su z = 0:
 *      la posizione 0 sta in fondo, l'ultima davanti (la più vicina alla camera di lato, che la vede
 *      sopra le altre: per un LLM è il token da cui si predice il prossimo).
 *
 * Tutte le sezioni sono centrate su y = 0 (la rete si legge come una fascia orizzontale) e usano lo
 * stesso passo in profondità: il token 3 sta alla stessa z in ogni sezione, così il suo percorso si
 * segue a occhio. Le sezioni molto alte (o molto profonde) stringono il passo invece di diventare
 * enormi: i neuroni diventano più piccoli, la rete resta proporzionata.
 *
 * Un blocco ripetuto ({ repeat: N, of: [...] }) si disegna UNA volta: dopo vengono poche lastre
 * "fantasma" con l'ingombro del blocco e la parentesi "× N" (vedi ghosts).
 */
import { sliceUnits } from "./spec.js";

export const SPACING = Object.freeze({
  neuron: 0.32, // passo tra neuroni sull'asse y
  depth: 0.56, // passo tra posizioni in profondità (token, frame…)
  grid: 0.84, // passo di una griglia 2D (righe su y e colonne su z, celle quadrate): lastre grandi come feature map
  slab: 0.52, // tra le lastre di una sezione
  attentionSlab: 1.6, // tra le lastre di una self-attention: lo spazio dove corrono gli archi tra i token
  plate: 0.46, // tra i canali (lastre) di una griglia
  expert: 0.4, // tra le lastre degli esperti
  section: 2.1, // spazio vuoto tra due sezioni: di lato la rete resta compatta e grande sullo schermo
  headGap: 0.8, // spazio in più (in passi) tra i neuroni di due head
  maxHeight: 15, // altezza massima di una sezione: oltre, il passo si stringe
  maxDepth: 18, // profondità massima della rete
  radius: 0.3, // raggio del neurone in frazione del passo più piccolo (aria tra i neuroni per vedere gli archi)
  minRadius: 0.012,
  maxRadius: 0.13,
  loopRoom: 1.4, // spazio dopo una self-attention a una sola lastra: gli archi escono a destra
  ghosts: 4, // lastre fantasma al massimo per un blocco ripetuto
  ghostGap: 0.85,
  ghostWidth: 0.34,
  groupRise: 2.6, // le parentesi dei gruppi stanno così in alto sopra la sezione più alta
  groupStep: 1.25,
  residualRise: 1.1,
  residualStep: 0.75,
});

/**
 * @param {import("./spec.js").NormalizedNetwork} net
 * @returns {NetworkLayout}
 */
export function computeLayout(net, spacing = SPACING) {
  // le sezioni che campionano l'asse delle fette con un altro passo (EnCodec a 36 posizioni accanto ai 12
  // frame, una KV compressa 2 → 1) non dettano la profondità: si stendono sulla stessa estensione
  const sized = net.sections.filter((s) => !proportionalDepth(s, net));
  const maxCols = Math.max(1, net.slices.count, ...sized.map((s) => s.depth.cols), ...sized.map((s) => s.depth.rows));
  // un passo in profondità per tutta la rete (le fette si allineano), stretto se la profondità è grande;
  // con le griglie 2D le celle sono più grandi: ogni canale è una lastra quadrata, come una feature map
  const preferred = net.sections.some((s) => s.depth.kind === "grid") ? spacing.grid : spacing.depth;
  const dz = Math.min(preferred, spacing.maxDepth / Math.max(1, maxCols - 1));
  const selfAttention = net.links.filter((l) => l.pattern === "attention" && l.within);
  const attentionLoops = new Set(selfAttention.filter((l) => l.from.slabs === 1).map((l) => l.from.id));
  const attentionWide = new Set(selfAttention.filter((l) => l.from.slabs > 1).map((l) => l.from.id));

  const sections = [];
  const repeats = [];
  let cursor = 0;
  for (const item of net.items) {
    if (item.kind === "section") {
      const box = placeSection(item, cursor, depthStep(item, net, dz), spacing, attentionWide.has(item.id));
      sections.push(box);
      cursor = box.x1 + spacing.section + (attentionLoops.has(item.id) ? spacing.loopRoom : 0);
    } else {
      const rep = placeRepeat(item, cursor, sections, spacing);
      repeats.push(rep);
      cursor = rep.x1 + spacing.section;
    }
  }

  // posizioni di tutte le unità (un neurone per fetta di profondità e per lastra)
  const positions = new Float32Array(net.units * 3);
  const depthU = new Float32Array(net.units);
  const sectionOf = new Uint16Array(net.units);
  for (const s of net.sections) {
    const box = sections[s.index];
    const sliced = takesPartInSlices(s, net);
    for (let u = 0; u < s.units; u++) {
      const g = s.offset + u;
      unitPosition(s, box, u, positions, 3 * g);
      depthU[g] = sliced ? unitDepthU(s, u) : -1;
      sectionOf[g] = s.index;
    }
  }

  const residuals = stackResiduals(net, sections, spacing);
  const groups = stackGroups(net, sections, repeats, residuals, spacing);
  const bounds = networkBounds(sections, repeats, groups, residuals);
  return { dz, sections, repeats, groups, residuals, positions, depthU, sectionOf, bounds, units: net.units, spacing };
}

/** Passo verticale di una sezione in fila: stretto se i neuroni (più gli spazi tra le head) non ci stanno. */
function rowsOf(s, spacing) {
  if (s.depth.kind === "grid") return s.depth.rows;
  const headGaps = s.heads > 1 ? (s.heads - 1) * spacing.headGap : 0;
  return s.neurons + headGaps;
}

/** Riga (in passi, con gli spazi tra le head) del neurone n di una sezione in fila. */
export function neuronRow(s, n, headGap = SPACING.headGap) {
  if (s.heads <= 1) return n;
  return n + Math.floor((n * s.heads) / s.neurons) * headGap;
}

/** Head a cui appartiene il neurone n (0 se la sezione non ha head). */
export const headOf = (s, n) => (s.heads > 1 ? Math.floor((n * s.heads) / s.neurons) : 0);

function placeSection(s, x0, dz, spacing, wide = false) {
  const grid = s.depth.kind === "grid";
  const plates = grid ? s.neurons : s.slabs; // lastre sull'asse x
  const rows = rowsOf(s, spacing);
  const dy = grid ? dz : Math.min(spacing.neuron, spacing.maxHeight / Math.max(1, rows - 1));
  const height = (rows - 1) * dy;
  const xs = new Float64Array(plates); // doppia precisione: niente "6,2200002" nei confronti dei test
  if (s.role === "experts") {
    // prima gli esperti condivisi, poi (dopo un passo vuoto) quelli instradati
    let x = x0;
    for (let e = 0; e < plates; e++) {
      if (e === s.shared && s.shared > 0) x += spacing.expert;
      xs[e] = x;
      x += spacing.expert;
    }
  } else {
    const step = grid ? spacing.plate : wide ? spacing.attentionSlab : spacing.slab;
    for (let p = 0; p < plates; p++) xs[p] = x0 + p * step;
  }
  const x1 = plates ? xs[plates - 1] : x0;
  const cols = s.depth.cols;
  const depthExtent = (cols - 1) * dz;
  const minStep = Math.min(dy, cols > 1 ? dz : Infinity, plates > 1 ? xs[1] - xs[0] : Infinity);
  const radius = Math.min(spacing.maxRadius, Math.max(spacing.minRadius, spacing.radius * minStep));
  const yTop = height / 2;
  const z0 = -depthExtent / 2;
  return {
    id: s.id,
    index: s.index,
    x0,
    x1,
    xs,
    plates,
    rows,
    dy,
    dz,
    yTop,
    yBottom: -height / 2,
    z0,
    z1: depthExtent / 2,
    radius,
    headGap: spacing.headGap,
    center: [(x0 + x1) / 2, 0, 0],
    box: { min: [x0 - radius, -height / 2 - radius, z0 - radius], max: [x1 + radius, height / 2 + radius, depthExtent / 2 + radius] },
  };
}

/**
 * Lastre fantasma di un blocco ripetuto: hanno l'ingombro (altezza e profondità) del blocco vero,
 * così si capisce che "dopo ce ne sono altri uguali", senza disegnarne N copie.
 */
function placeRepeat(rep, x0, sections, spacing) {
  const block = rep.of.map((s) => sections[s.index]);
  const yTop = Math.max(...block.map((b) => b.yTop));
  const yBottom = Math.min(...block.map((b) => b.yBottom));
  const z0 = Math.min(...block.map((b) => b.z0));
  const z1 = Math.max(...block.map((b) => b.z1));
  const count = Math.min(spacing.ghosts, rep.count - 1);
  const ghosts = [];
  for (let i = 0; i < count; i++) ghosts.push({ x: x0 + i * spacing.ghostGap, width: spacing.ghostWidth, fade: 1 - i / Math.max(1, count) });
  const x1 = x0 + (count - 1) * spacing.ghostGap + spacing.ghostWidth;
  return {
    id: rep.id,
    index: rep.index,
    count: rep.count,
    shown: count,
    x0,
    x1,
    ghosts,
    yTop,
    yBottom,
    z0,
    z1,
    blockX0: block[0].x0,
    blockX1: block[block.length - 1].x1,
    box: { min: [x0, yBottom, z0], max: [x1, yTop, z1] },
  };
}

/** Coordinata x, y, z dell'unità u (locale alla sezione) scritta in out[o…o+2]. */
export function unitPosition(s, box, u, out, o = 0) {
  const D = s.depth.count;
  const d = u % D;
  const t = (u - d) / D;
  const n = t % s.neurons;
  const slab = (t - n) / s.neurons;
  if (s.depth.kind === "grid") {
    const row = Math.floor(d / s.depth.cols);
    const col = d % s.depth.cols;
    out[o] = box.xs[n];
    out[o + 1] = box.yTop - row * box.dy;
    out[o + 2] = box.z0 + col * box.dz;
    return out;
  }
  out[o] = box.xs[slab];
  out[o + 1] = box.yTop - neuronRow(s, n, box.headGap) * box.dy;
  out[o + 2] = D > 1 ? box.z0 + d * box.dz : 0;
  return out;
}

/**
 * true se la sezione si accende e si spegne con la fetta scelta: ha profondità dello stesso tipo delle
 * fette e non l'ha esclusa la specifica (`slices: false`).
 */
export const takesPartInSlices = (s, net) => s.depth.count > 1 && s.depth.kind === net.slices.kind && s.slices !== false;

/**
 * true se la sezione campiona l'asse delle fette (in fila) con un altro numero di posizioni: 36 istanti
 * accanto a 12 frame, una KV compressa 2 → 1 accanto ai token. Le sue posizioni si stendono sulla stessa
 * estensione in z delle fette (passo proporzionale): la posizione p sta alla z del suo istante.
 */
export const proportionalDepth = (s, net) => net.slices.kind === "line" && takesPartInSlices(s, net) && s.depth.count !== net.slices.count;

/**
 * Passo in profondità di una sezione: quello della rete, o (passo proporzionale) quello che porta il
 * centro della posizione p alla frazione (p + 0,5) / D dell'asse, come la fetta t sta a (t + 0,5) / T.
 */
export const depthStep = (s, net, dz) => (proportionalDepth(s, net) ? (dz * net.slices.count) / s.depth.count : dz);

/** Posizione normalizzata in profondità (0…1, centro della colonna) per la scelta della fetta; -1 = senza profondità. */
export function unitDepthU(s, u) {
  const D = s.depth.count;
  if (D <= 1) return -1;
  const col = (u % D) % s.depth.cols;
  return (col + 0.5) / s.depth.cols;
}

/** Scompone l'indice locale di un'unità: { slab, neuron, depth, row, col }. */
export function decodeUnit(s, u) {
  const D = s.depth.count;
  const depth = u % D;
  const t = (u - depth) / D;
  const neuron = t % s.neurons;
  const slab = (t - neuron) / s.neurons;
  return { slab, neuron, depth, row: Math.floor(depth / s.depth.cols), col: depth % s.depth.cols };
}

/** Indice locale dell'unità (lastra, neurone, posizione): l'ordine dei `values` della specifica. */
export const unitIndex = (s, slab, neuron, depth) => (slab * s.neurons + neuron) * s.depth.count + depth;

/** Indice nella fetta (lastra × neuroni + neurone): quello usato dalle coppie dei collegamenti "explicit". */
export const sliceIndex = (s, slab, neuron) => slab * s.neurons + neuron;

/**
 * Archi dei residui: sopra le sezioni che saltano. Si piazzano dal più corto al più lungo; un arco che
 * si sovrappone a uno già piazzato sale di un gradino (due residui consecutivi A→B e B→C restano alla
 * stessa altezza: si toccano solo in B).
 */
function stackResiduals(net, sections, spacing) {
  const list = net.links.filter((l) => l.pattern === "residual").map((l) => ({ link: l.index, from: l.from.index, to: l.to.index }));
  list.sort((a, b) => a.to - a.from - (b.to - b.from) || a.from - b.from);
  const placed = [];
  for (const r of list) {
    let top = -Infinity;
    for (let i = r.from; i <= r.to; i++) top = Math.max(top, sections[i].yTop);
    const level = overlapLevel(placed, r.from, r.to);
    placed.push({ ...r, x0: sections[r.from].center[0], x1: sections[r.to].center[0], top, level, height: top + spacing.residualRise + level * spacing.residualStep });
  }
  return placed.sort((a, b) => a.link - b.link);
}

/** Gradino libero per l'intervallo (a, b): uno sopra il più alto degli intervalli già piazzati che lo toccano dentro. */
function overlapLevel(placed, a, b) {
  let level = 0;
  for (const p of placed) if (p.from < b && a < p.to) level = Math.max(level, p.level + 1);
  return level;
}

/**
 * Parentesi dei gruppi sopra le sezioni (e sopra i residui che passano di lì): dalle più strette alle
 * più larghe, una parentesi che ne contiene o ne taglia un'altra già piazzata sale di un gradino.
 */
function stackGroups(net, sections, repeats, residuals, spacing) {
  const extent = (item) => (item.kind === "repeat" ? repeats.find((r) => r.id === item.id) : sections[item.index]);
  const groups = net.groups.map((g) => {
    const a = extent(g.from);
    const b = extent(g.to);
    const x0 = Math.min(a.x0, a.blockX0 ?? a.x0);
    const x1 = b.x1;
    let top = -Infinity;
    for (const s of sections) if (s.x1 >= x0 - 1e-6 && s.x0 <= x1 + 1e-6) top = Math.max(top, s.yTop);
    for (const r of repeats) if (r.x1 >= x0 - 1e-6 && r.x0 <= x1 + 1e-6) top = Math.max(top, r.yTop);
    for (const r of residuals) if (r.x1 >= x0 - 1e-6 && r.x0 <= x1 + 1e-6) top = Math.max(top, r.height + 0.35);
    return { index: g.index, label: g.label, color: g.color, x0, x1, top, level: 0, y: 0 };
  });
  const placed = [];
  for (const g of [...groups].sort((p, q) => p.x1 - p.x0 - (q.x1 - q.x0) || p.x0 - q.x0)) {
    let level = 0;
    let floor = g.top + spacing.groupRise;
    for (const p of placed) {
      if (p.x0 < g.x1 - 1e-6 && g.x0 < p.x1 - 1e-6) {
        level = Math.max(level, p.level + 1);
        floor = Math.max(floor, p.y + spacing.groupStep);
      }
    }
    g.level = level;
    g.y = floor;
    placed.push(g);
  }
  return groups;
}

function networkBounds(sections, repeats, groups, residuals) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const add = (b) => {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], b.min[i]);
      max[i] = Math.max(max[i], b.max[i]);
    }
  };
  for (const s of sections) add(s.box);
  for (const r of repeats) add(r.box);
  for (const g of groups) max[1] = Math.max(max[1], g.y + 0.6);
  for (const r of residuals) max[1] = Math.max(max[1], r.height + 0.3);
  return { min, max };
}

/** Riquadro (min/max) di più sezioni o ripetizioni per id: per inquadrarle con la camera. */
export function boxOf(layout, net, ids) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const id of ids) {
    const item = net.byId.get(id);
    if (!item) continue;
    const b = item.kind === "repeat" ? layout.repeats.find((r) => r.id === id).box : layout.sections[item.index].box;
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], b.min[i]);
      max[i] = Math.max(max[i], b.max[i]);
    }
  }
  return Number.isFinite(min[0]) ? { min, max } : null;
}

/** Unità di una fetta di profondità: per comodità dei collegamenti e dei test. */
export const sliceSize = sliceUnits;

/**
 * @typedef {ReturnType<typeof computeLayout>} NetworkLayout
 */
