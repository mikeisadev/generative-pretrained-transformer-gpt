/**
 * Geometrie della rete orizzontale (three.js): ogni funzione costruisce UN oggetto per tutta la rete.
 *
 *   neuronPoints   tutti i neuroni in un THREE.Points (un vertice per neurone, disco nello shader)
 *   slabMesh       una "lastra" (scatola sottile) per lastra di ogni sezione, con la heatmap delle
 *                  attivazioni presa da un atlante: da lontano sostituisce i neuroni (LOD)
 *   segmentLines   i collegamenti dritti, nell'ordine di priorità (la densità è un drawRange)
 *   arcLines       gli archi (attention, cross-attention, residui) come polilinee in un'unica geometria
 *   ghostMeshes    le lastre fantasma dei blocchi ripetuti
 *   structure      basamenti delle sezioni, parentesi dei gruppi e delle ripetizioni, assi
 * Colori nello spazio dello schermo (vedi shaders.js). Niente qui gira a ogni frame.
 */
import * as THREE from "three";
import { MODE } from "./activations.js";
import { ARC_KIND } from "./links.js";
import { ACTIVATION, AXIS_COLORS, HEAD_COLORS, LINK_COLORS } from "./roles.js";
import { ghostMaterial, lineMaterial, neuronMaterial, rgb, slabMaterial } from "./shaders.js";

const COLOR = Object.fromEntries(Object.entries(ACTIVATION).map(([k, v]) => [k, rgb(v)]));

/** Colore di un'unità per la heatmap (stessa regola dello shader dei neuroni), in byte 0…255. */
export function unitRGB(display, mode, out, o) {
  let c;
  let t;
  if (mode === MODE.off) {
    c = COLOR.off;
    t = 1;
  } else {
    const a = Math.min(1, Math.abs(display));
    if (mode === MODE.probs) [c, t] = [COLOR.probability, a];
    else if (mode === MODE.onehot) [c, t] = [COLOR.onehot, a];
    else [c, t] = [display < 0 ? COLOR.negative : COLOR.positive, Math.pow(a, 0.6)];
  }
  const z = mode === MODE.off ? c : COLOR.zero;
  out[o] = Math.round(255 * (z[0] + (c[0] - z[0]) * t));
  out[o + 1] = Math.round(255 * (z[1] + (c[1] - z[1]) * t));
  out[o + 2] = Math.round(255 * (z[2] + (c[2] - z[2]) * t));
  out[o + 3] = 255;
}

/** Spegne il raycast di un oggetto (le linee non si "colpiscono" col doppio clic: costerebbe decine di ms). */
const noRaycast = (object) => {
  object.raycast = () => {};
  return object;
};

// ------------------------------------------------------------------------------------------ neuroni
export function neuronPoints(model, uniforms) {
  const { net, layout, values } = model;
  const n = net.units;
  const geometry = new THREE.BufferGeometry();
  const radius = new Float32Array(n);
  const section = new Float32Array(n);
  for (let u = 0; u < n; u++) {
    section[u] = layout.sectionOf[u];
    radius[u] = layout.sections[layout.sectionOf[u]].radius;
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(layout.positions, 3));
  const value = new THREE.BufferAttribute(values.display, 1);
  const mode = new THREE.BufferAttribute(Float32Array.from(values.mode), 1);
  value.setUsage(THREE.DynamicDrawUsage);
  mode.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("aValue", value);
  geometry.setAttribute("aMode", mode);
  geometry.setAttribute("aSection", new THREE.BufferAttribute(section, 1));
  geometry.setAttribute("aDepthU", new THREE.BufferAttribute(layout.depthU, 1));
  geometry.setAttribute("aRadius", new THREE.BufferAttribute(radius, 1));
  geometry.boundingSphere = new THREE.Sphere(); // calcolata sotto: serve a "Ti sei perso?" della navigazione
  geometry.computeBoundingSphere();
  const points = new THREE.Points(geometry, neuronMaterial(uniforms));
  points.frustumCulled = false;
  points.name = "nn-neurons";
  // Raycast proprio (doppio clic della navigazione): un solo ciclo sulle posizioni, nessuna allocazione
  // per punto, e restituisce solo il neurone più vicino al raggio entro il suo raggio.
  const local = new THREE.Ray();
  const inverse = new THREE.Matrix4();
  const p = new THREE.Vector3();
  points.raycast = (raycaster, intersects) => {
    inverse.copy(points.matrixWorld).invert();
    local.copy(raycaster.ray).applyMatrix4(inverse);
    const pos = layout.positions;
    let best = -1;
    let bestT = Infinity;
    for (let u = 0; u < n; u++) {
      p.set(pos[3 * u], pos[3 * u + 1], pos[3 * u + 2]);
      const t = p.sub(local.origin).dot(local.direction);
      if (t < 0 || t >= bestT) continue;
      const d2 = p.lengthSq() - t * t;
      const r = 1.6 * radius[u];
      if (d2 <= r * r) {
        best = u;
        bestT = t;
      }
    }
    if (best < 0) return;
    const point = local.at(bestT, new THREE.Vector3()).applyMatrix4(points.matrixWorld);
    intersects.push({ distance: raycaster.ray.origin.distanceTo(point), point, index: best, object: points });
  };
  return points;
}

// ------------------------------------------------------------------------------------------ lastre-heatmap
/**
 * Una scatola sottile per lastra. Le facce laterali (piani y-z) mostrano tutta la heatmap neuroni ×
 * profondità; la faccia davanti la fetta più vicina (l'ultimo token), quella dietro la prima, sopra e
 * sotto la prima e l'ultima riga: girando la camera la lastra resta leggibile da ogni lato.
 */
export function slabMesh(model, uniforms) {
  const { net, layout, values } = model;
  const blocks = [];
  for (const s of net.sections) {
    const box = layout.sections[s.index];
    const grid = s.depth.kind === "grid";
    for (let p = 0; p < box.plates; p++) blocks.push({ s, box, plate: p, w: s.depth.cols, h: grid ? s.depth.rows : s.neurons, x: 0, y: 0 });
  }
  const atlas = packBlocks(blocks);
  const data = new Uint8Array(atlas.width * atlas.height * 4);
  for (const b of blocks) writeBlock(b, values, atlas, data);
  const texture = new THREE.DataTexture(data, atlas.width, atlas.height, THREE.RGBAFormat);
  texture.magFilter = THREE.NearestFilter; // ogni texel è un neurone: bordi netti
  texture.minFilter = THREE.NearestFilter;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;

  const count = blocks.length * 24;
  const pos = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const sec = new Float32Array(count);
  const du = new Float32Array(count);
  const index = new Uint32Array(blocks.length * 36);
  let v = 0;
  let k = 0;
  const o = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  /** Una faccia: i triangoli girano in senso antiorario visti da fuori (`normal`), così basta FrontSide. */
  const quad = (corners, uvs, depthU, section, normal) => {
    for (let c = 0; c < 4; c++) {
      pos.set(corners[c], 3 * v + 3 * c);
      uv.set(uvs[c], 2 * v + 2 * c);
      sec[v + c] = section;
      du[v + c] = depthU[c];
    }
    o.fromArray(corners[0]);
    ab.fromArray(corners[1]).sub(o);
    ac.fromArray(corners[2]).sub(o);
    const outward = ab.cross(ac).dot(normal) > 0;
    index.set(outward ? [v, v + 1, v + 2, v, v + 2, v + 3] : [v, v + 2, v + 1, v, v + 3, v + 2], k);
    v += 4;
    k += 6;
  };
  const N = { px: new THREE.Vector3(1, 0, 0), nx: new THREE.Vector3(-1, 0, 0), pz: new THREE.Vector3(0, 0, 1), nz: new THREE.Vector3(0, 0, -1), py: new THREE.Vector3(0, 1, 0), ny: new THREE.Vector3(0, -1, 0) };
  for (const b of blocks) {
    const { s, box } = b;
    const deep = s.depth.count > 1;
    const t = Math.min(0.24, Math.max(0.06, 2.2 * box.radius)); // spessore della lastra in x
    const x = box.xs[b.plate];
    const half = (deep ? box.dz : t) / 2;
    const [x0, x1] = [x - t / 2, x + t / 2];
    const [y0, y1] = [box.yBottom - box.dy / 2, box.yTop + box.dy / 2];
    const [z0, z1] = deep ? [box.z0 - half, box.z1 + half] : [-t / 2, t / 2];
    const { u0, u1, v0, v1, uFront, uBack, vTop, vBottom } = atlasRect(b, atlas);
    const d = (u) => (deep ? u : -1);
    const W = b.w;
    const S = s.index;
    // facce laterali (normale ±x): heatmap intera (u = profondità, v = neuroni, neurone 0 in alto)
    quad([[x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]], [[u0, v0], [u1, v0], [u1, v1], [u0, v1]], [d(0), d(1), d(1), d(0)], S, N.px);
    quad([[x0, y0, z1], [x0, y0, z0], [x0, y1, z0], [x0, y1, z1]], [[u1, v0], [u0, v0], [u0, v1], [u1, v1]], [d(1), d(0), d(0), d(1)], S, N.nx);
    // davanti (+z): l'ultima posizione; dietro (−z): la prima
    const front = d(1 - 0.5 / W);
    const back = d(0.5 / W);
    quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [[uFront, v0], [uFront, v0], [uFront, v1], [uFront, v1]], [front, front, front, front], S, N.pz);
    quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [[uBack, v0], [uBack, v0], [uBack, v1], [uBack, v1]], [back, back, back, back], S, N.nz);
    // sopra: la prima riga lungo la profondità; sotto: l'ultima
    quad([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [[u1, vTop], [u1, vTop], [u0, vTop], [u0, vTop]], [d(1), d(1), d(0), d(0)], S, N.py);
    quad([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], [[u0, vBottom], [u0, vBottom], [u1, vBottom], [u1, vBottom]], [d(0), d(0), d(1), d(1)], S, N.ny);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geometry.setAttribute("aSection", new THREE.BufferAttribute(sec, 1));
  geometry.setAttribute("aDepthU", new THREE.BufferAttribute(du, 1));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const mesh = new THREE.Mesh(geometry, slabMaterial(uniforms, texture));
  mesh.frustumCulled = false;
  mesh.name = "nn-slabs";
  return { mesh, texture, blocks, atlas, data };
}

/** Riscrive nell'atlante i colori di una sezione (dopo setSectionValues). */
export function repaintSection(slabs, values, section) {
  for (const b of slabs.blocks) if (b.s === section) writeBlock(b, values, slabs.atlas, slabs.data);
  slabs.texture.needsUpdate = true;
}

function writeBlock(b, values, atlas, data) {
  const { s } = b;
  const grid = s.depth.kind === "grid";
  const D = s.depth.count;
  for (let r = 0; r < b.h; r++) {
    for (let c = 0; c < b.w; c++) {
      // unità della cella: in fila (lastra, neurone r, posizione c); in una griglia (canale = lastra, riga r, colonna c)
      const u = grid ? s.offset + b.plate * D + r * s.depth.cols + c : s.offset + (b.plate * s.neurons + r) * D + c;
      const texel = (b.y + (b.h - 1 - r)) * atlas.width + b.x + c; // riga 0 in alto
      unitRGB(values.display[u], values.mode[u], data, 4 * texel);
    }
  }
}

function packBlocks(blocks) {
  const order = blocks.map((_, i) => i).sort((a, b) => blocks[b].h - blocks[a].h);
  const total = blocks.reduce((a, b) => a + (b.w + 1) * (b.h + 1), 0);
  const maxW = Math.min(4096, Math.max(64, 2 ** Math.ceil(Math.log2(Math.sqrt(total) * 1.3))));
  let x = 0;
  let y = 0;
  let shelf = 0;
  let width = 0;
  for (const i of order) {
    const b = blocks[i];
    if (x + b.w > maxW) {
      x = 0;
      y += shelf + 1;
      shelf = 0;
    }
    b.x = x;
    b.y = y;
    x += b.w + 1; // un texel vuoto tra i blocchi
    shelf = Math.max(shelf, b.h);
    width = Math.max(width, x);
  }
  return { width: Math.max(1, width), height: Math.max(1, y + shelf) };
}

function atlasRect(b, atlas) {
  const e = 0.002; // dentro il blocco: il filtro "nearest" non deve pescare dal vicino
  const W = atlas.width;
  const H = atlas.height;
  return {
    u0: (b.x + e) / W,
    u1: (b.x + b.w - e) / W,
    v0: (b.y + e) / H,
    v1: (b.y + b.h - e) / H,
    uFront: (b.x + b.w - 0.5) / W,
    uBack: (b.x + 0.5) / W,
    vTop: (b.y + b.h - 0.5) / H,
    vBottom: (b.y + 0.5) / H,
  };
}

// ------------------------------------------------------------------------------------------ collegamenti dritti
/** Colore e opacità di base di un collegamento, per schema (pesi: segno e intensità); scrive in `out`. */
const STRUCTURAL = rgb(LINK_COLORS.structural);
const LOOKUP = rgb(LINK_COLORS.lookup);
const ROUTE = rgb(LINK_COLORS.route);
function linkStyle(link, weight, norm, out) {
  switch (link.pattern) {
    case "dense":
    case "conv":
    case "explicit":
      out.c = weight < 0 ? COLOR.negative : COLOR.positive;
      out.a = 0.22 + 0.78 * Math.pow(Math.min(1, Math.abs(weight) / norm), 0.8);
      break;
    case "lookup":
      out.c = LOOKUP;
      out.a = 0.65;
      break;
    case "route":
      out.c = ROUTE;
      out.a = 0.3 + 0.7 * Math.min(1, weight);
      break;
    default:
      out.c = STRUCTURAL;
      out.a = 0.5;
  }
  return out;
}

/** Scrive il colore RGBA (byte) dei due vertici del segmento i. */
function writePairColor(color, i, c, alpha) {
  for (let e = 0; e < 2; e++) {
    const o = 8 * i + 4 * e;
    color[o] = 255 * c[0];
    color[o + 1] = 255 * c[1];
    color[o + 2] = 255 * c[2];
    color[o + 3] = 255 * alpha;
  }
}

/**
 * @param {{ from, to, weight, link, count }} list segmenti (o pesi della fetta) già ordinati
 * @param {(i: number, out: Float32Array, o: number) => void} [place] posizioni dei due estremi (predefinito: dalle unità)
 */
export function segmentLines(model, uniforms, list, { place = null, owner = null } = {}) {
  const { net, layout } = model;
  const n = list.count;
  const pos = new Float32Array(Math.max(1, n) * 6);
  const color = new Uint8Array(Math.max(1, n) * 8);
  const secA = new Float32Array(Math.max(1, n) * 2);
  const secB = new Float32Array(Math.max(1, n) * 2);
  const du = new Float32Array(Math.max(1, n) * 2);
  const norms = net.links.map(() => 1e-9);
  for (let i = 0; i < n; i++) norms[list.link[i]] = Math.max(norms[list.link[i]], Math.abs(list.weight[i]));
  const P = layout.positions;
  const style = { c: null, a: 0 };
  for (let i = 0; i < n; i++) {
    const link = net.links[list.link[i]];
    const a = list.from[i];
    const b = list.to[i];
    if (place) place(i, pos, 6 * i);
    else {
      pos[6 * i] = P[3 * a];
      pos[6 * i + 1] = P[3 * a + 1];
      pos[6 * i + 2] = P[3 * a + 2];
      pos[6 * i + 3] = P[3 * b];
      pos[6 * i + 4] = P[3 * b + 1];
      pos[6 * i + 5] = P[3 * b + 2];
    }
    linkStyle(link, list.weight[i], norms[list.link[i]], style);
    writePairColor(color, i, style.c, style.a);
    secA[2 * i] = secA[2 * i + 1] = link.to.index;
    secB[2 * i] = secB[2 * i + 1] = link.from.index;
    // la fetta del segmento: quella dell'arrivo se ha profondità, altrimenti quella della partenza
    const slice = owner ? owner(i) : layout.depthU[b] >= 0 ? layout.depthU[b] : layout.depthU[a];
    du[2 * i] = du[2 * i + 1] = slice;
  }
  return lineObject(uniforms, { pos, color, secA, secB, du, count: n * 2, name: "nn-links" });
}

/** LineSegments con lo shader dei collegamenti (enfasi: la più alta delle due sezioni toccate). */
function lineObject(uniforms, { pos, color, secA, secB, du, count, name }) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(color, 4, true));
  geometry.setAttribute("aSection", new THREE.BufferAttribute(secA, 1));
  geometry.setAttribute("aSection2", new THREE.BufferAttribute(secB, 1));
  geometry.setAttribute("aDepthU", new THREE.BufferAttribute(du, 1));
  geometry.setDrawRange(0, count);
  const lines = new THREE.LineSegments(geometry, lineMaterial(uniforms));
  lines.frustumCulled = false;
  lines.name = name;
  return noRaycast(lines);
}

// ------------------------------------------------------------------------------------------ archi
// segmenti per curva: abbastanza per sembrare morbida, pochi perché ogni segmento costa (rendering software)
const ARC_STEPS = { [ARC_KIND.attention]: 5, [ARC_KIND.cross]: 8, [ARC_KIND.residual]: 10 };

/**
 * Archi come polilinee: attention tra due lastre = ponte che sale tanto più quanto i token sono lontani;
 * attention dentro una lastra sola = ansa verso destra; cross-attention = arco alto tra due sezioni;
 * residuo = arco sopra le sezioni saltate, all'altezza decisa dal layout (archi annidati non si coprono).
 */
export function arcLines(model, uniforms) {
  const { net, layout, links } = model;
  const arcs = links.arcs;
  let segments = 0;
  for (let i = 0; i < arcs.count; i++) segments += ARC_STEPS[arcs.kind[i]];
  const pos = new Float32Array(Math.max(1, segments) * 6);
  const color = new Uint8Array(Math.max(1, segments) * 8);
  const secA = new Float32Array(Math.max(1, segments) * 2);
  const secB = new Float32Array(Math.max(1, segments) * 2);
  const du = new Float32Array(Math.max(1, segments) * 2);
  const residualHeight = new Map(layout.residuals.map((r) => [r.link, r.height]));
  const heads = HEAD_COLORS.map(rgb);
  const residual = rgb(LINK_COLORS.residual);
  const P = layout.positions;
  const p0 = new THREE.Vector3();
  const p1 = new THREE.Vector3();
  const c1 = new THREE.Vector3();
  const c2 = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const quadratic = new THREE.QuadraticBezierCurve3(p0, c1, p1);
  const cubic = new THREE.CubicBezierCurve3(p0, c1, c2, p1);
  let s = 0;
  for (let i = 0; i < arcs.count; i++) {
    const kind = arcs.kind[i];
    const link = net.links[arcs.link[i]];
    const from = arcs.from[i];
    const to = arcs.to[i];
    p0.set(P[3 * from], P[3 * from + 1], P[3 * from + 2]);
    p1.set(P[3 * to], P[3 * to + 1], P[3 * to + 2]);
    let curve = quadratic;
    if (kind === ARC_KIND.residual) {
      const h = residualHeight.get(link.index) ?? Math.max(p0.y, p1.y) + 1;
      const y0 = Math.max(p0.y, p1.y);
      const yc = y0 + (h - y0) / 0.75; // il punto più alto di questa cubica è a 3/4 dei controlli
      c1.set(p0.x, yc, p0.z);
      c2.set(p1.x, yc, p1.z);
      curve = cubic;
    } else if (kind === ARC_KIND.cross) {
      c1.addVectors(p0, p1).multiplyScalar(0.5);
      c1.y += 0.6 + 0.08 * Math.abs(p1.x - p0.x) + 0.25 * Math.abs(p1.z - p0.z);
    } else if (Math.abs(p1.x - p0.x) > 1e-4) {
      c1.addVectors(p0, p1).multiplyScalar(0.5);
      c1.y += Math.min(2.6, 0.18 + 0.34 * Math.hypot(p1.z - p0.z, p1.y - p0.y));
    } else {
      c1.addVectors(p0, p1).multiplyScalar(0.5);
      c1.x += Math.min(1.25, 0.22 + 0.32 * p0.distanceTo(p1));
    }
    const steps = ARC_STEPS[kind];
    const rgbA = kind === ARC_KIND.residual ? residual : heads[arcs.head[i] % heads.length];
    const alpha = kind === ARC_KIND.residual ? 0.6 : 0.16 + 0.84 * Math.sqrt(arcs.weight[i]);
    const slice = layout.depthU[to] >= 0 ? layout.depthU[to] : layout.depthU[from];
    curve.getPoint(0, a);
    for (let k = 1; k <= steps; k++) {
      curve.getPoint(k / steps, b);
      pos[6 * s] = a.x;
      pos[6 * s + 1] = a.y;
      pos[6 * s + 2] = a.z;
      pos[6 * s + 3] = b.x;
      pos[6 * s + 4] = b.y;
      pos[6 * s + 5] = b.z;
      writePairColor(color, s, rgbA, alpha);
      secA[2 * s] = secA[2 * s + 1] = link.to.index;
      secB[2 * s] = secB[2 * s + 1] = link.from.index;
      du[2 * s] = du[2 * s + 1] = slice;
      a.copy(b);
      s++;
    }
  }
  return lineObject(uniforms, { pos, color, secA, secB, du, count: s * 2, name: "nn-arcs" });
}

// ------------------------------------------------------------------------------------------ fantasmi e struttura
/** Accumula triangoli e linee con colore per vertice (shader dei fantasmi: onda compresa). */
class Builder {
  constructor() {
    this.tri = [];
    this.triColor = [];
    this.line = [];
    this.lineColor = [];
  }

  box(x0, x1, y0, y1, z0, z1, rgbA, faceAlpha, edgeAlpha) {
    const c = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
    if (faceAlpha > 0) {
      for (const [p, q, r, t] of [[0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7], [1, 5, 6, 2], [3, 2, 6, 7], [4, 5, 1, 0]]) this.quad(c[p], c[q], c[r], c[t], rgbA, faceAlpha);
    }
    if (edgeAlpha > 0) for (const [p, q] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) this.segment(c[p], c[q], rgbA, edgeAlpha);
  }

  quad(a, b, c, d, rgbA, alpha) {
    for (const p of [a, b, c, a, c, d]) {
      this.tri.push(...p);
      this.triColor.push(255 * rgbA[0], 255 * rgbA[1], 255 * rgbA[2], 255 * alpha);
    }
  }

  segment(a, b, rgbA, alpha) {
    this.line.push(...a, ...b);
    for (let e = 0; e < 2; e++) this.lineColor.push(255 * rgbA[0], 255 * rgbA[1], 255 * rgbA[2], 255 * alpha);
  }

  /** { mesh, lines }: oggetti pronti (o null se vuoti). */
  objects(uniforms, name) {
    const out = { mesh: null, lines: null };
    if (this.tri.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(this.tri, 3));
      g.setAttribute("aColor", new THREE.BufferAttribute(Uint8Array.from(this.triColor), 4, true));
      g.computeBoundingBox();
      out.mesh = new THREE.Mesh(g, ghostMaterial(uniforms));
      out.mesh.name = `${name}-faces`;
    }
    if (this.line.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(this.line, 3));
      g.setAttribute("aColor", new THREE.BufferAttribute(Uint8Array.from(this.lineColor), 4, true));
      out.lines = noRaycast(new THREE.LineSegments(g, ghostMaterial(uniforms, { lines: true })));
      out.lines.name = `${name}-lines`;
    }
    return out;
  }
}

/** Lastre fantasma dei blocchi ripetuti, la parentesi sotto il blocco e i basamenti delle sezioni. */
export function structure(model, uniforms) {
  const { net, layout } = model;
  const faces = new Builder();
  const ink = rgb("#cbd5e1");
  // basamento di ogni sezione: mostra quanto è profonda anche quando la si guarda di lato
  for (const s of net.sections) {
    const box = layout.sections[s.index];
    const y = box.yBottom - box.dy / 2 - 0.22;
    const pad = Math.max(0.16, box.radius * 1.5);
    const deep = s.depth.count > 1;
    const z0 = deep ? box.z0 - box.dz / 2 : -pad;
    const z1 = deep ? box.z1 + box.dz / 2 : pad;
    faces.quad([box.x0 - pad, y, z1], [box.x1 + pad, y, z1], [box.x1 + pad, y, z0], [box.x0 - pad, y, z0], rgb(s.color), 0.12);
  }
  for (const r of layout.repeats) {
    for (const g of r.ghosts) faces.box(g.x, g.x + g.width, r.yBottom - 0.1, r.yTop + 0.1, r.z0 - 0.2, r.z1 + 0.2, ink, 0.07 + 0.1 * g.fade, 0.25 + 0.4 * g.fade);
    // parentesi sotto il blocco vero e i suoi fantasmi: "tutto questo × N"
    const y = r.yBottom - 1.05;
    faces.segment([r.blockX0, y, r.z1 + 0.2], [r.x1, y, r.z1 + 0.2], ink, 0.75);
    faces.segment([r.blockX0, y, r.z1 + 0.2], [r.blockX0, y + 0.35, r.z1 + 0.2], ink, 0.75);
    faces.segment([r.x1, y, r.z1 + 0.2], [r.x1, y + 0.35, r.z1 + 0.2], ink, 0.75);
  }
  for (const g of layout.groups) {
    const c = rgb(g.color);
    faces.segment([g.x0, g.y, 0], [g.x1, g.y, 0], c, 0.8);
    faces.segment([g.x0, g.y, 0], [g.x0, g.y - 0.4, 0], c, 0.8);
    faces.segment([g.x1, g.y, 0], [g.x1, g.y - 0.4, 0], c, 0.8);
  }
  return faces.objects(uniforms, "nn-structure");
}

/** Assi in un angolo (davanti, in basso a sinistra): x flusso, y neuroni, z profondità. Restituisce le punte. */
export function axesGizmo(model) {
  const { min, max } = model.layout.bounds;
  const origin = new THREE.Vector3(min[0] - 1.4, model.layout.sections.reduce((m, s) => Math.min(m, s.yBottom), 0) - 0.9, max[2] + 0.6);
  const length = { x: 2.4, y: 1.6, z: Math.max(1.4, Math.min(3, (max[2] - min[2]) * 0.5)) };
  const group = new THREE.Group();
  group.name = "nn-axes";
  const tips = {};
  for (const [axis, dir] of [["x", new THREE.Vector3(1, 0, 0)], ["y", new THREE.Vector3(0, 1, 0)], ["z", new THREE.Vector3(0, 0, -1)]]) {
    // z verso il fondo: le posizioni crescono verso la camera, quindi la freccia indica "più indietro nel prompt"
    const end = origin.clone().addScaledVector(dir, length[axis]);
    const color = new THREE.Color(AXIS_COLORS[axis]);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, end]), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }));
    noRaycast(line);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.3, 10), new THREE.MeshBasicMaterial({ color }));
    cone.position.copy(end);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    cone.userData.noBounds = true;
    noRaycast(cone);
    group.add(line, cone);
    tips[axis] = end.clone().addScaledVector(dir, 0.35);
  }
  return { group, tips, origin };
}

/** Segmenti riusabili per un contorno (riquadro della sezione scelta, piano della fetta): posizioni scritte da fuori. */
export function outline(segments, color, opacity) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(segments * 6), 3));
  const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity, depthWrite: false }));
  lines.frustumCulled = false;
  lines.visible = false;
  lines.renderOrder = 5;
  return noRaycast(lines);
}
