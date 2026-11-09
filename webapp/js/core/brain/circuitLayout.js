/**
 * Disposizione del microcircuito: i gruppi di neuroni in colonne da sinistra a destra (nell'ordine
 * della specifica, come la rete AI che rappresentano), i neuroni di ogni gruppo in una piccola griglia
 * (x, profondità z), le fibre tra i gruppi (dall'assone di un neurone ai dendriti, o al soma per le
 * sinapsi inibitorie a canestro, di un neurone del gruppo di arrivo). Puro: niente three.js.
 *
 *   const layout = layoutCircuit(spec.circuit, { detail: 1 });
 *   layout.columns  // [{ id, index, x, top, bottom, label, ai, region, type, count, color, fire }]
 *   layout.cells    // [{ group, column, position, scale, morph, fire }]
 *   layout.fibers   // [{ edge, kind, direction: "forward" | "backward" | "self", points, from, to }]
 *   layout.box      // { min, max } in unità della scena del circuito
 *
 * Le dimensioni dei neuroni NON sono in scala tra i tipi (una cellula granulare è minuscola accanto a
 * una di Purkinje): ogni tipo è mostrato a una grandezza leggibile, e la vista lo dichiara.
 */
import { buildMorphology } from "./morphology.js";

/** Grandezza di disegno per tipo (moltiplica l'altezza ≈ 1 della morfologia). */
export const DISPLAY_SCALE = Object.freeze({
  pyramidal: 1.25,
  stellate: 1.55,
  relay: 1.55,
  basket: 1.45,
  granule: 1.45,
  purkinje: 1.3,
  dentate: 1.6,
  photoreceptor: 1.4,
  ganglion: 1.45,
  hair: 1.7,
  bipolar: 1.5,
  projection: 1.5,
  spiny: 1.6,
  pallidal: 1.45,
  dopaminergic: 1.35,
});

/** Famiglia di un tipo (colore del neurone = cosa fa alle sinapsi di arrivo, come nella legenda). */
export const FAMILY = Object.freeze({
  pyramidal: "excitatory",
  stellate: "excitatory",
  relay: "excitatory",
  granule: "excitatory",
  dentate: "excitatory",
  ganglion: "excitatory",
  bipolar: "excitatory",
  projection: "excitatory",
  basket: "inhibitory",
  purkinje: "inhibitory",
  spiny: "inhibitory",
  pallidal: "inhibitory",
  dopaminergic: "modulatory",
  hair: "sensory",
  photoreceptor: "sensory",
});

export const FAMILY_COLORS = Object.freeze({
  excitatory: "#ecc9a6",
  inhibitory: "#9fc3f0",
  modulatory: "#d6b4f7",
  sensory: "#8fe3d3",
});

/** Spaziature (unità della scena del circuito). */
export const SPACING = Object.freeze({ column: 0.5, dx: 0.42, dz: 0.56, jitter: 0.06 });

/** Secondi tra lo scarico di un gruppo e quello del successivo (animazione degli impulsi). */
export const FIRE_STEP = 0.9;

/**
 * @param {object} circuit circuito normalizzato (spec.js)
 * @param {{ detail?: number }} [o] dettaglio delle morfologie (1 con una GPU, ~0,6 con il rendering software)
 */
export function layoutCircuit(circuit, { detail = 1 } = {}) {
  const columns = [];
  const cells = [];
  let x = 0;
  circuit.groups.forEach((group, index) => {
    const scale = DISPLAY_SCALE[group.type] ?? 1.4;
    // al più due file in larghezza, il resto in profondità: il circuito resta compatto da sinistra a
    // destra (si legge come la rete AI) e girando la camera si vede lo spessore
    const nx = group.count <= 6 ? 1 : group.count <= 12 ? 2 : 3;
    const nz = Math.ceil(group.count / nx);
    const morphs = [];
    let minX = Infinity;
    let maxX = -Infinity;
    let top = -Infinity;
    let bottom = Infinity;
    for (let k = 0; k < group.count; k++) {
      const morph = buildMorphology(group.type, `${group.id}:${k}`, { detail });
      morphs.push(morph);
      minX = Math.min(minX, morph.extent.minX * scale);
      maxX = Math.max(maxX, morph.extent.maxX * scale);
      top = Math.max(top, morph.extent.maxY * scale);
      bottom = Math.min(bottom, morph.extent.minY * scale);
    }
    // la colonna inizia dove finisce la precedente: larghezza = griglia + ingombro di una cellula
    const gridWidth = (nx - 1) * SPACING.dx;
    const left = x - minX;
    const column = {
      id: group.id,
      index,
      x: left + gridWidth / 2,
      left: x,
      right: left + gridWidth + maxX,
      top: top + SPACING.jitter,
      bottom: bottom - SPACING.jitter,
      depth: (nz - 1) * SPACING.dz,
      label: group.label,
      ai: group.ai,
      region: group.region,
      marker: group.marker,
      type: group.type,
      count: group.count,
      family: FAMILY[group.type] ?? "excitatory",
      fire: index * FIRE_STEP,
      cells: [],
    };
    column.color = FAMILY_COLORS[column.family];
    morphs.forEach((morph, k) => {
      const ix = k % nx;
      const iz = Math.floor(k / nx);
      // piccolo scarto deterministico: una fila perfetta sembrerebbe un disegno tecnico, non un tessuto
      const jx = (hash01(index * 131 + k * 17) - 0.5) * SPACING.jitter * 2;
      const jy = (hash01(index * 71 + k * 29) - 0.5) * SPACING.jitter * 2;
      const jz = (hash01(index * 53 + k * 41) - 0.5) * SPACING.jitter * 2;
      const position = [left + ix * SPACING.dx + jx, jy, (iz - (nz - 1) / 2) * SPACING.dz + jz + (ix % 2) * SPACING.dz * 0.35];
      const cell = { group: group.id, column: index, index: cells.length, position, scale, morph, fire: column.fire + jy * 0.4 };
      cells.push(cell);
      column.cells.push(cell);
    });
    columns.push(column);
    x = column.right + SPACING.column;
  });

  const byId = new Map(columns.map((c) => [c.id, c]));
  const top = Math.max(...columns.map((c) => c.top));
  const fibers = [];
  circuit.edges.forEach((edge, e) => {
    const from = byId.get(edge.from);
    const to = byId.get(edge.to);
    if (!from || !to) return;
    const direction = from === to ? "self" : from.index < to.index ? "forward" : "backward";
    const count = Math.max(2, Math.min(5, Math.round((from.count + to.count) / 4)));
    for (let k = 0; k < count; k++) {
      const source = from.cells[(k * 5 + 1) % from.cells.length];
      let target = to.cells[(k * 3) % to.cells.length];
      if (direction === "self" && target === source) target = to.cells[(k * 3 + 1) % to.cells.length];
      const start = place(source, source.morph.axonEnd);
      const end = edge.kind === "inhibitory" ? somaContact(target, start) : place(target, nearestInput(target, start));
      fibers.push({ edge: e, kind: edge.kind, direction, from: edge.from, to: edge.to, label: edge.label, points: route(start, end, direction, top, k), source: source.index, target: target.index, fire: from.fire });
    }
  });

  const box = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  for (const c of columns) {
    extend(box, [c.left, c.bottom, -c.depth / 2 - 0.6]);
    extend(box, [c.right, c.top, c.depth / 2 + 0.6]);
  }
  for (const f of fibers) for (const p of f.points) extend(box, p);
  return { columns, cells, fibers, box, top };
}

/** Punto della morfologia (coordinate locali) nella scena del circuito. */
export function place(cell, local) {
  return [cell.position[0] + local[0] * cell.scale, cell.position[1] + local[1] * cell.scale, cell.position[2] + local[2] * cell.scale];
}

/** L'ingresso (punta di dendrite) del neurone d'arrivo più comodo da raggiungere da `from`. */
function nearestInput(cell, from) {
  let best = cell.morph.inputs[0] ?? [0, 0, 0];
  let score = Infinity;
  for (const p of cell.morph.inputs) {
    const w = place(cell, p);
    const s = Math.abs(w[1] - from[1]) + 0.4 * Math.abs(w[2] - from[2]) + 0.25 * Math.abs(w[0] - from[0]);
    if (s < score) {
      score = s;
      best = p;
    }
  }
  return best;
}

/** Contatto sul soma (sinapsi inibitoria a canestro), dal lato da cui arriva la fibra. */
function somaContact(cell, from) {
  const c = cell.position;
  const d = [from[0] - c[0], from[1] - c[1], from[2] - c[2]];
  const l = Math.hypot(d[0], d[1], d[2]) || 1;
  const r = 0.07 * cell.scale;
  return [c[0] + (d[0] / l) * r, c[1] + (d[1] / l) * r, c[2] + (d[2] / l) * r];
}

/** Punti di controllo di una fibra: in avanti a "S", all'indietro ad arco sopra le colonne, ricorrente ad anello. */
function route(start, end, direction, top, k) {
  if (direction === "forward") {
    return [start, [start[0] + 0.35, start[1] - 0.08, start[2]], [end[0] - 0.4, end[1] - 0.05, end[2]], end];
  }
  if (direction === "backward") {
    const h = top + 0.55 + 0.16 * k;
    return [start, [start[0] + 0.3, start[1] + 0.4, start[2]], [start[0] - 0.2, h, start[2] * 0.6], [end[0] + 0.3, h, end[2] * 0.6], [end[0] + 0.12, end[1] + 0.3, end[2]], end];
  }
  const side = (k % 2 ? -1 : 1) * 0.55;
  return [start, [start[0] + 0.25, start[1] + 0.2, start[2] + side], [end[0] - 0.2, end[1] + 0.1, end[2] + side * 0.8], end];
}

function extend(box, p) {
  for (let a = 0; a < 3; a++) {
    box.min[a] = Math.min(box.min[a], p[a]);
    box.max[a] = Math.max(box.max[a], p[a]);
  }
}

/** Numero pseudo-casuale in [0, 1) da un intero (scarti deterministici). */
function hash01(n) {
  let h = (n | 0) ^ 0x9e3779b9;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
