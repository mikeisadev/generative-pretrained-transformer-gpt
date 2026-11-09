/**
 * Forma del cervello stilizzato, in millimetri MNI (x sinistra → destra, y dietro → avanti, z sotto →
 * sopra). Puro: niente three.js (tests/brain.test.mjs lo prova in Node).
 *
 *   const base = buildHemisphereBase({ cell: 1.6 });            // superficie liscia dell'emisfero sinistro
 *   const left = buildCortex(base, { side: "left", seed: 11 });  // + solchi e giri, colori, lobi
 *   const right = buildCortex(mirrorMesh(base), { side: "right", seed: 23 });
 *   lobeOf([-47, 21, 17]) → "frontale"
 *
 * Come nasce l'emisfero:
 *  1. una funzione distanza con segno (SDF) unisce ellissoidi per i lobi (frontale, centrale,
 *     parietale, occipitale e temporale) e taglia la parete interna, la base orbitale e il piano del
 *     tentorio sopra il cervelletto. Il lobo temporale si unisce con un raccordo stretto davanti e
 *     largo dietro: la piega che resta è la scissura laterale (di Silvio), con il polo temporale che
 *     sporge sotto il lobo frontale come nel cervello vero (una SDF ammette le sporgenze);
 *  2. "surface nets" estrae la superficie con vertici a passo quasi costante (niente triangoli
 *     stirati come su una sfera deformata) e una proiezione di Newton la rende liscia;
 *  3. i solchi principali (centrale, laterale, frontali, temporali, intraparietale, calcarino…) sono
 *     curve proiettate sulla superficie; i giri minori vengono da rumore "ridged" (|rumore| vicino a
 *     zero = fondo del solco): tutto spostato lungo la normale, verso l'interno.
 * La profondità del solco diventa anche il colore (fondo più scuro): si legge anche da trasparente.
 */
import { computeNormals } from "./geometry.js";
import { createNoise3D } from "./noise.js";

// ---------------------------------------------------------------- funzioni distanza
/** Ellissoide (approssimazione di Quilez: quasi una distanza vera vicino alla superficie). */
function ellipsoid(x, y, z, cx, cy, cz, rx, ry, rz) {
  const px = (x - cx) / rx;
  const py = (y - cy) / ry;
  const pz = (z - cz) / rz;
  const k0 = Math.sqrt(px * px + py * py + pz * pz);
  const qx = px / rx;
  const qy = py / ry;
  const qz = pz / rz;
  const k1 = Math.sqrt(qx * qx + qy * qy + qz * qz);
  return k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz);
}

/** Unione morbida (polinomiale): k = raggio del raccordo in mm. */
export function smin(a, b, k) {
  if (k <= 0) return Math.min(a, b);
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

/** Intersezione morbida. */
export function smax(a, b, k) {
  return -smin(-a, -b, k);
}

const smoothstep = (e0, e1, x) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/**
 * Lobi come ellissoidi (centro e semiassi in mm, emisfero sinistro). Scelti guardando i limiti del
 * cervello MNI152: polo frontale a y ≈ +70, polo occipitale a y ≈ −104, vertice a z ≈ +78, faccia
 * laterale a x ≈ −68, polo temporale a (−42, 16, −32), base temporale a z ≈ −40.
 */
export const LOBE_SHAPES = Object.freeze({
  frontal: { c: [-27, 28, 12], r: [29, 46, 48] },
  central: { c: [-33, -10, 33], r: [33, 40, 44] },
  parietal: { c: [-34, -45, 30], r: [34, 42, 43] },
  occipital: { c: [-21, -79, 10], r: [21, 27, 27] },
  // asse dal polo temporale (davanti, in basso) verso la parte posteriore (più in alto)
  temporal: { c: [-48, -22, -14], r: [20, 46, 25], pitch: (-19 * Math.PI) / 180 },
  // lobo temporale mediale (giro paraippocampale, uncus): ippocampo ed entorinale stanno qui dentro
  medialTemporal: { c: [-27, -18, -20], r: [10, 32, 12] },
});

const TEMPORAL_COS = Math.cos(LOBE_SHAPES.temporal.pitch);
const TEMPORAL_SIN = Math.sin(LOBE_SHAPES.temporal.pitch);

/**
 * Piano del tentorio (la "tenda" di dura madre tra lobi occipitali e cervelletto): z sotto cui
 * l'emisfero finisce, in funzione di y. Dietro (y < −40) sale verso il polo occipitale.
 */
export const tentoriumZ = (y) => -32 - 0.5 * (y + 40);
const TENTORIUM_COS = 1 / Math.hypot(1, 0.5);

/** Funzione distanza con segno (mm, negativa dentro) dell'emisfero SINISTRO liscio. */
export function hemisphereSdf(x, y, z) {
  const L = LOBE_SHAPES;
  let frontal = ellipsoid(x, y, z, L.frontal.c[0], L.frontal.c[1], L.frontal.c[2], L.frontal.r[0], L.frontal.r[1], L.frontal.r[2]);
  frontal = smax(frontal, -24 - z, 8); // faccia orbitale: la base del lobo frontale è quasi piatta
  const central = ellipsoid(x, y, z, L.central.c[0], L.central.c[1], L.central.c[2], L.central.r[0], L.central.r[1], L.central.r[2]);
  const parietal = ellipsoid(x, y, z, L.parietal.c[0], L.parietal.c[1], L.parietal.c[2], L.parietal.r[0], L.parietal.r[1], L.parietal.r[2]);
  const occipital = ellipsoid(x, y, z, L.occipital.c[0], L.occipital.c[1], L.occipital.c[2], L.occipital.r[0], L.occipital.r[1], L.occipital.r[2]);
  let d = smin(smin(smin(frontal, central, 18), parietal, 18), occipital, 16);
  // lobo temporale nel suo sistema ruotato (asse lungo inclinato di 19°)
  const T = L.temporal;
  const dy = y - T.c[1];
  const dz = z - T.c[2];
  const s = dy * TEMPORAL_COS + dz * TEMPORAL_SIN;
  const t = -dy * TEMPORAL_SIN + dz * TEMPORAL_COS;
  const MT = L.medialTemporal;
  const temporal = smin(ellipsoid(x, s, t, T.c[0], 0, 0, T.r[0], T.r[1], T.r[2]), ellipsoid(x, y, z, MT.c[0], MT.c[1], MT.c[2], MT.r[0], MT.r[1], MT.r[2]), 9);
  // raccordo stretto davanti (resta la scissura laterale), largo dietro (il lobo temporale continua
  // nel parietale e nell'occipitale senza piega)
  const k = 2.5 + 15 * smoothstep(-34, -66, y);
  d = smin(d, temporal, k);
  d = smax(d, x + 2, 6); // parete interna: la scissura longitudinale separa i due emisferi (4 mm)
  d = smax(d, (tentoriumZ(y) - z) * TENTORIUM_COS, 6); // sopra il cervelletto
  return d;
}

/** Box (mm) che contiene l'emisfero sinistro, con un margine per la griglia. */
export const HEMISPHERE_BOX = Object.freeze({ min: [-74, -110, -50], max: [2, 76, 84] });

// ---------------------------------------------------------------- surface nets
/**
 * Estrae la superficie f = 0 di una funzione distanza con "naive surface nets": un vertice per cella
 * attraversata (la media dei punti di attraversamento sugli spigoli) e un quadrilatero per ogni
 * spigolo della griglia che cambia segno. Mesh chiusa, vertici a passo ~ `cell`.
 * @returns {{ positions: Float32Array, indices: Uint32Array }}
 */
export function surfaceNets(sdf, { min, max }, cell) {
  const nx = Math.ceil((max[0] - min[0]) / cell) + 1;
  const ny = Math.ceil((max[1] - min[1]) / cell) + 1;
  const nz = Math.ceil((max[2] - min[2]) / cell) + 1;
  const field = new Float32Array(nx * ny * nz);
  let idx = 0;
  for (let k = 0; k < nz; k++) {
    const z = min[2] + k * cell;
    for (let j = 0; j < ny; j++) {
      const y = min[1] + j * cell;
      for (let i = 0; i < nx; i++) field[idx++] = sdf(min[0] + i * cell, y, z);
    }
  }
  const at = (i, j, k) => field[i + nx * (j + ny * k)];
  const cx = nx - 1;
  const cy = ny - 1;
  const cz = nz - 1;
  const cellVertex = new Int32Array(cx * cy * cz).fill(-1);
  let positions = new Float32Array(1 << 16);
  let count = 0;
  // gli 8 angoli di una cella e i 12 spigoli (coppie di angoli)
  const corner = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
  ];
  const edges = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];
  const v = new Float32Array(8);
  for (let k = 0; k < cz; k++) {
    for (let j = 0; j < cy; j++) {
      for (let i = 0; i < cx; i++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          v[c] = at(i + corner[c][0], j + corner[c][1], k + corner[c][2]);
          if (v[c] < 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let n = 0;
        for (const [a, b] of edges) {
          const va = v[a];
          const vb = v[b];
          if (va < 0 === vb < 0) continue;
          const w = va / (va - vb);
          sx += corner[a][0] + (corner[b][0] - corner[a][0]) * w;
          sy += corner[a][1] + (corner[b][1] - corner[a][1]) * w;
          sz += corner[a][2] + (corner[b][2] - corner[a][2]) * w;
          n++;
        }
        if (count * 3 + 3 > positions.length) {
          const grown = new Float32Array(positions.length * 2);
          grown.set(positions);
          positions = grown;
        }
        positions[count * 3] = min[0] + (i + sx / n) * cell;
        positions[count * 3 + 1] = min[1] + (j + sy / n) * cell;
        positions[count * 3 + 2] = min[2] + (k + sz / n) * cell;
        cellVertex[i + cx * (j + cy * k)] = count++;
      }
    }
  }
  const cellAt = (i, j, k) => cellVertex[i + cx * (j + cy * k)];
  const indices = [];
  const quad = (a, b, c, d, inside) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    // antiorario visto da fuori: l'ordine dipende da che parte dello spigolo sta l'interno
    if (inside) indices.push(a, b, c, a, c, d);
    else indices.push(a, d, c, a, c, b);
  };
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const f0 = at(i, j, k);
        // spigolo lungo x: celle attorno nel piano y-z
        if (i < cx && j > 0 && k > 0 && j < cy && k < cz) {
          const f1 = at(i + 1, j, k);
          if (f0 < 0 !== f1 < 0) quad(cellAt(i, j - 1, k - 1), cellAt(i, j, k - 1), cellAt(i, j, k), cellAt(i, j - 1, k), f0 < 0);
        }
        // spigolo lungo y: celle attorno nel piano z-x
        if (j < cy && i > 0 && k > 0 && i < cx && k < cz) {
          const f1 = at(i, j + 1, k);
          if (f0 < 0 !== f1 < 0) quad(cellAt(i - 1, j, k - 1), cellAt(i - 1, j, k), cellAt(i, j, k), cellAt(i, j, k - 1), f0 < 0);
        }
        // spigolo lungo z: celle attorno nel piano x-y
        if (k < cz && i > 0 && j > 0 && i < cx && j < cy) {
          const f1 = at(i, j, k + 1);
          if (f0 < 0 !== f1 < 0) quad(cellAt(i - 1, j - 1, k), cellAt(i, j - 1, k), cellAt(i, j, k), cellAt(i - 1, j, k), f0 < 0);
        }
      }
    }
  }
  return { positions: positions.slice(0, count * 3), indices: Uint32Array.from(indices) };
}

/**
 * Porta ogni vertice sulla superficie f = 0 (passi di Newton lungo il gradiente): toglie la
 * "scalettatura" di surface nets. In place.
 */
export function projectToSurface(positions, sdf, { iterations = 2, h = 0.4 } = {}) {
  for (let i = 0; i < positions.length; i += 3) {
    let x = positions[i];
    let y = positions[i + 1];
    let z = positions[i + 2];
    for (let it = 0; it < iterations; it++) {
      const f = sdf(x, y, z);
      const gx = (sdf(x + h, y, z) - sdf(x - h, y, z)) / (2 * h);
      const gy = (sdf(x, y + h, z) - sdf(x, y - h, z)) / (2 * h);
      const gz = (sdf(x, y, z + h) - sdf(x, y, z - h)) / (2 * h);
      const g2 = gx * gx + gy * gy + gz * gz;
      if (g2 < 1e-8) break;
      // passo limitato: dove la funzione non è una distanza vera (raccordi) niente salti lontani
      const step = Math.max(-1.5, Math.min(1.5, f / g2));
      x -= gx * step;
      y -= gy * step;
      z -= gz * step;
    }
    positions[i] = x;
    positions[i + 1] = y;
    positions[i + 2] = z;
  }
  return positions;
}

/** Laplaciano a una passata (media con i vicini) per ammorbidire i quadrilateri di surface nets. In place. */
export function relax(mesh, amount = 0.5) {
  const { positions, indices } = mesh;
  const n = positions.length / 3;
  const sum = new Float32Array(n * 3);
  const deg = new Uint16Array(n);
  for (let f = 0; f < indices.length; f += 3) {
    for (let e = 0; e < 3; e++) {
      const a = indices[f + e];
      const b = indices[f + ((e + 1) % 3)];
      sum[a * 3] += positions[b * 3];
      sum[a * 3 + 1] += positions[b * 3 + 1];
      sum[a * 3 + 2] += positions[b * 3 + 2];
      deg[a]++;
    }
  }
  for (let i = 0; i < n; i++) {
    if (!deg[i]) continue;
    for (let a = 0; a < 3; a++) positions[i * 3 + a] += (sum[i * 3 + a] / deg[i] - positions[i * 3 + a]) * amount;
  }
  return mesh;
}

/**
 * Superficie liscia dell'emisfero sinistro (prima di solchi e giri). `cell`: passo della griglia in
 * mm (1,4-1,6 con una GPU, 2,2-2,6 con il rendering software).
 */
export function buildHemisphereBase({ cell = 1.6 } = {}) {
  const mesh = surfaceNets(hemisphereSdf, HEMISPHERE_BOX, cell);
  relax(mesh, 0.5);
  projectToSurface(mesh.positions, hemisphereSdf);
  computeNormals(mesh);
  mesh.cell = cell;
  return mesh;
}

/** La stessa mesh nell'altro emisfero (x → −x, triangoli girati perché restino rivolti all'esterno). */
export function mirrorMesh(mesh) {
  const positions = mesh.positions.slice();
  for (let i = 0; i < positions.length; i += 3) positions[i] = -positions[i];
  const indices = mesh.indices.slice();
  for (let f = 0; f < indices.length; f += 3) {
    const t = indices[f + 1];
    indices[f + 1] = indices[f + 2];
    indices[f + 2] = t;
  }
  const out = { positions, indices, cell: mesh.cell };
  if (mesh.normals) {
    out.normals = mesh.normals.slice();
    for (let i = 0; i < out.normals.length; i += 3) out.normals[i] = -out.normals[i];
  }
  return out;
}

// ---------------------------------------------------------------- solchi principali
/**
 * Solchi principali dell'emisfero sinistro: punti di passaggio (mm) vicini alla superficie, che
 * buildCortex proietta sulla superficie liscia. `depth` (mm) e `width` (mm, mezza larghezza a metà
 * profondità) del solco. Tracciati guardando un atlante di anatomia: sono indicativi.
 */
export const SULCI = Object.freeze([
  // scissura laterale (di Silvio): dal polo temporale indietro e in su fino al giro sopramarginale
  { id: "lateral", depth: 10, width: 2.6, points: [[-40, 16, -14], [-50, 4, -4], [-56, -12, 6], [-58, -28, 16], [-56, -40, 24]] },
  // solco centrale (di Rolando): tra lobo frontale e parietale, scende in avanti dal vertice
  { id: "central", depth: 9, width: 2, points: [[-4, -38, 74], [-22, -32, 72], [-38, -24, 62], [-50, -16, 48], [-58, -10, 33]] },
  { id: "precentral", depth: 7, width: 1.7, points: [[-8, -20, 76], [-26, -14, 70], [-42, -6, 56], [-54, 2, 40], [-56, 6, 28]] },
  { id: "postcentral", depth: 7, width: 1.7, points: [[-6, -52, 72], [-26, -46, 68], [-42, -38, 56], [-55, -30, 40], [-60, -24, 28]] },
  { id: "superior_frontal", depth: 6.5, width: 1.8, points: [[-22, -14, 70], [-22, 6, 64], [-24, 26, 56], [-22, 46, 40], [-20, 60, 22]] },
  { id: "inferior_frontal", depth: 6.5, width: 1.8, points: [[-46, -2, 46], [-46, 14, 40], [-44, 32, 30], [-40, 50, 16]] },
  { id: "superior_temporal", depth: 7.5, width: 1.9, points: [[-50, 10, -26], [-60, -6, -14], [-63, -26, -6], [-60, -46, 4], [-52, -60, 16]] },
  { id: "inferior_temporal", depth: 6, width: 1.7, points: [[-52, 2, -36], [-62, -16, -28], [-63, -36, -20], [-56, -56, -12]] },
  { id: "intraparietal", depth: 7, width: 1.9, points: [[-46, -34, 50], [-38, -48, 54], [-30, -62, 50], [-24, -76, 38]] },
  { id: "lateral_occipital", depth: 5, width: 1.6, points: [[-42, -80, 8], [-34, -90, 12], [-24, -98, 14]] },
  // faccia interna (mediale): parieto-occipitale, calcarina, cingolato
  { id: "parieto_occipital", depth: 8, width: 2, points: [[-3, -70, 56], [-3, -68, 40], [-3, -64, 24], [-3, -60, 10]] },
  { id: "calcarine", depth: 7, width: 1.8, points: [[-3, -60, 8], [-4, -72, 6], [-6, -86, 5], [-10, -100, 4]] },
  { id: "cingulate", depth: 6.5, width: 1.8, points: [[-3, 34, 0], [-3, 42, 20], [-3, 24, 40], [-3, -6, 46], [-3, -30, 50], [-3, -42, 66]] },
]);

/** Punti densi lungo una polilinea (ogni `step` mm), per misurare la distanza dal solco. */
function densify(points, step) {
  const out = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const n = Math.max(1, Math.ceil(d / step));
    for (let s = 0; s < n; s++) out.push([a[0] + ((b[0] - a[0]) * s) / n, a[1] + ((b[1] - a[1]) * s) / n, a[2] + ((b[2] - a[2]) * s) / n]);
  }
  out.push(points[points.length - 1].slice());
  return out;
}

/**
 * I solchi come polilinee dense sulla superficie liscia (emisfero sinistro). Con un seme i punti
 * serpeggiano un poco (pochi millimetri, diversi nei due emisferi): un solco vero non è mai una retta.
 */
export function projectedSulci(step = 1.5, seed = 0) {
  const wobble = seed ? createNoise3D(seed + 7) : null;
  return SULCI.map((s, k) => {
    const points = densify(s.points, step);
    if (wobble) {
      for (const p of points) {
        const f = 0.07;
        p[0] += 2.2 * wobble(p[0] * f, p[1] * f, p[2] * f + k * 13);
        p[1] += 2.8 * wobble(p[0] * f + 31, p[1] * f, p[2] * f + k * 13);
        p[2] += 2.8 * wobble(p[0] * f, p[1] * f + 57, p[2] * f + k * 13);
      }
    }
    const flat = new Float32Array(points.flat());
    projectToSurface(flat, hemisphereSdf, { iterations: 8 });
    const out = [];
    for (let i = 0; i < flat.length; i += 3) out.push([flat[i], flat[i + 1], flat[i + 2]]);
    return { ...s, points: out };
  });
}

/** Griglia di celle (passo `size` mm) per trovare in fretta i punti dei solchi vicini a un vertice. */
function spatialHash(segments, size) {
  const map = new Map();
  // chiave numerica (niente stringhe: la si calcola 27 volte per ognuno delle decine di migliaia di vertici)
  const key = (i, j, k) => ((i + 512) * 1024 + (j + 512)) * 1024 + (k + 512);
  segments.forEach((seg, index) => {
    for (const p of seg.points) {
      const k = key(Math.floor(p[0] / size), Math.floor(p[1] / size), Math.floor(p[2] / size));
      if (!map.has(k)) map.set(k, []);
      map.get(k).push(index, p[0], p[1], p[2]);
    }
  });
  return {
    near(x, y, z, visit) {
      const ci = Math.floor(x / size);
      const cj = Math.floor(y / size);
      const ck = Math.floor(z / size);
      for (let i = ci - 1; i <= ci + 1; i++) for (let j = cj - 1; j <= cj + 1; j++) for (let k = ck - 1; k <= ck + 1; k++) {
        const list = map.get(key(i, j, k));
        if (!list) continue;
        for (let n = 0; n < list.length; n += 4) visit(list[n], list[n + 1], list[n + 2], list[n + 3]);
      }
    },
  };
}

// ---------------------------------------------------------------- corteccia: giri, solchi, colori, lobi
/**
 * Solchi e giri sulla superficie liscia di un emisfero: sposta i vertici lungo la normale (verso
 * l'interno nei solchi) e calcola per ogni vertice la profondità (0 = cresta del giro, 1 = fondo del
 * solco) e il lobo. Restituisce una nuova mesh { positions, normals, indices, depth, lobe }.
 * @param {object} base mesh liscia (buildHemisphereBase o mirrorMesh)
 * @param {{ side: "left" | "right", seed?: number, minor?: number }} o
 *   minor = ampiezza dei giri minori in mm (0 = solo i solchi principali)
 */
export function buildCortex(base, { side = "left", seed = 11, minor = 4.2, gyrus = 4.2 } = {}) {
  const mirror = side === "right" ? -1 : 1;
  const sulci = projectedSulci(1.5, seed);
  const hash = spatialHash(sulci, 8);
  const centralIndex = sulci.findIndex((s) => s.id === "central");
  // un rumore per famiglia di giri: allungato lungo la direzione in cui quei giri corrono
  // `gyrus` = mezza larghezza di un giro (mm): con una mesh più rada (rendering software) i giri si
  // fanno più larghi, perché un solco più stretto di due vertici sparirebbe
  const families = GYRI.map((g, k) => ({ ...g, freq: g.freq * (4.2 / gyrus), noise: createNoise3D(seed + 17 * k), dir: normalize3(g.dir) }));
  const warp = createNoise3D(seed + 101);
  const n = base.positions.length / 3;
  const positions = base.positions.slice();
  const normals = base.normals ?? computeNormals({ positions: base.positions, indices: base.indices }).normals;
  const depth = new Float32Array(n);
  const lobe = new Uint8Array(n);
  const best = new Float32Array(sulci.length);
  const weights = new Float32Array(families.length);
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    const x = positions[o];
    const y = positions[o + 1];
    const z = positions[o + 2];
    const lx = x * mirror; // coordinate dell'emisfero sinistro (solchi e famiglie di giri sono definiti lì)
    // solchi principali: profilo a "V" arrotondata (fondo stretto, spalle morbide = giri ai lati)
    best.fill(Infinity);
    hash.near(lx, y, z, (index, px, py, pz) => {
      const d = (px - lx) * (px - lx) + (py - y) * (py - y) + (pz - z) * (pz - z);
      if (d < best[index]) best[index] = d;
    });
    let major = 0;
    let majorDepth01 = 0;
    let nearest = Infinity;
    for (let s = 0; s < sulci.length; s++) {
      if (best[s] === Infinity) continue;
      const d = Math.sqrt(best[s]);
      const w = Math.max(sulci[s].width * 2.6, gyrus * 1.5);
      const a = 1 - arch(d / w);
      if (sulci[s].depth * a > major) {
        major = sulci[s].depth * a;
        majorDepth01 = a;
      }
      nearest = Math.min(nearest, d / w);
    }
    // giri minori: il fondo del solco è dove il rumore vale zero. La distanza (in mm) da lì si stima
    // con |rumore| / |gradiente|: così ogni solco minore ha la stessa larghezza e tra due solchi resta
    // un giro pieno e arrotondato, invece di una superficie tutta a pendii
    const wx = lx + 6 * warp(lx * 0.035, y * 0.035, z * 0.035);
    const wy = y + 6 * warp(lx * 0.035 + 17, y * 0.035, z * 0.035);
    const wz = z + 6 * warp(lx * 0.035, y * 0.035 + 31, z * 0.035);
    gyrusWeights(lx, y, z, best[centralIndex], weights);
    let distance = 0;
    let wsum = 0;
    for (let k = 0; k < families.length; k++) {
      const w = weights[k];
      if (w < 0.02) continue;
      const g = families[k];
      const v0 = familyNoise(g, wx, wy, wz);
      const gx = (familyNoise(g, wx + GRAD_H, wy, wz) - v0) / GRAD_H;
      const gy = (familyNoise(g, wx, wy + GRAD_H, wz) - v0) / GRAD_H;
      const gz = (familyNoise(g, wx, wy, wz + GRAD_H) - v0) / GRAD_H;
      // solo la parte del gradiente lungo la superficie: è lungo la superficie che si misura il solco
      const nx = normals[o] * mirror;
      const gn = gx * nx + gy * normals[o + 1] + gz * normals[o + 2];
      const tx = gx - gn * nx;
      const ty = gy - gn * normals[o + 1];
      const tz = gz - gn * normals[o + 2];
      distance += w * Math.min(8, Math.abs(v0) / Math.max(1e-4, Math.hypot(tx, ty, tz)));
      wsum += w;
    }
    const minorA = 1 - arch((wsum > 0 ? distance / wsum : 8) / gyrus); // 1 = fondo, 0 = cresta
    // accanto a un solco principale niente solchi minori: le sue due sponde restano giri pieni
    const fade = smoothstep(0.45, 1.15, nearest);
    // nella striscia centrale i giri precentrale e postcentrale restano lisci e lunghi: il solco
    // centrale (il confine tra lobo frontale e parietale) si riconosce subito
    const strip = 1 - 0.8 * weights[1];
    const minorDepth = minor * (minorA * fade * strip - 0.3);
    const total = Math.max(major, minorDepth);
    positions[o] = x - normals[o] * total;
    positions[o + 1] = y - normals[o + 1] * total;
    positions[o + 2] = z - normals[o + 2] * total;
    depth[i] = Math.max(majorDepth01, minorA * fade * strip);
    lobe[i] = LOBE_INDEX[lobeOf([lx, y, z])];
  }
  const mesh = computeNormals({ positions, indices: base.indices });
  mesh.depth = depth;
  mesh.lobe = lobe;
  mesh.side = side;
  return mesh;
}

/**
 * Famiglie di giri: dove si trovano (gyrusWeights), in che direzione corrono (dir, mm MNI dell'emisfero
 * sinistro) e quanto sono allungati (stretch < 1: il rumore cambia più piano lungo dir). Così i giri
 * frontali corrono avanti-indietro, il precentrale e il postcentrale seguono il solco centrale, quelli
 * temporali seguono il lobo, mentre parietale e occipitale restano più intricati.
 */
const GYRI = Object.freeze([
  { id: "frontal", dir: [0, 1, 0.1], stretch: 0.45, freq: 1 / 12.5 },
  { id: "central", dir: [-0.74, 0.38, -0.56], stretch: 0.25, freq: 1 / 12 },
  { id: "temporal", dir: [0, 0.94, -0.33], stretch: 0.42, freq: 1 / 12 },
  { id: "posterior", dir: [0, 1, 0], stretch: 0.85, freq: 1 / 11 },
]);

const GRAD_H = 0.35; // passo (mm) per stimare il gradiente del rumore

/**
 * Profilo di un giro tra due solchi (t = distanza dal fondo del solco / mezza larghezza del giro):
 * 0 sul fondo, 1 sulla cresta, ad arco (parabola): pendii morbidi che la mesh riesce a rappresentare,
 * fondo a "V" stretta che diventa la linea scura del solco.
 */
function arch(t) {
  const u = Math.min(1, Math.max(0, t));
  return 1 - (1 - u) * (1 - u);
}

/** Rumore di una famiglia di giri: compresso lungo la direzione dei giri (cambia piano lungo, in fretta di traverso). */
function familyNoise(g, x, y, z) {
  const along = (x * g.dir[0] + y * g.dir[1] + z * g.dir[2]) * (g.stretch - 1);
  return g.noise((x + g.dir[0] * along) * g.freq, (y + g.dir[1] * along) * g.freq, (z + g.dir[2] * along) * g.freq);
}

/** Pesi delle famiglie di giri in un punto (somma ~1). `centralD2`: distanza² dal solco centrale. */
function gyrusWeights(x, y, z, centralD2, out) {
  const central = Number.isFinite(centralD2) ? Math.exp(-centralD2 / (2 * 11 * 11)) : 0;
  const temporal = smoothstep(lateralFissureZ(y) + 3, lateralFissureZ(y) - 7, z) * smoothstep(-74, -58, y);
  const frontal = smoothstep(centralSulcusY(z) + 8, centralSulcusY(z) + 22, y) * (1 - temporal);
  const rest = Math.max(0, 1 - central - temporal - frontal);
  out[0] = frontal * (1 - central);
  out[1] = central;
  out[2] = temporal * (1 - central);
  out[3] = rest;
  return out;
}

function normalize3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Ordine dei lobi negli attributi per vertice (il colore lo dà atlas.js LOBES). */
export const LOBE_ORDER = Object.freeze(["frontale", "parietale", "temporale", "occipitale", "insula", "limbico"]);
const LOBE_INDEX = Object.fromEntries(LOBE_ORDER.map((id, i) => [id, i]));

/** y del solco centrale a una certa altezza z (scende in avanti): il confine frontale-parietale. */
export const centralSulcusY = (z) => -35 + (72 - z) * (27 / 42);
/**
 * z della scissura laterale a una certa y (sale all'indietro, da circa (y 15, z −10) a (y −40, z 25)):
 * il confine sopra il lobo temporale. Il giro di Heschl (−42, −19, 10) e il giro temporale superiore
 * (−53, −21, 7) restano sotto, come devono.
 */
export const lateralFissureZ = (y) => -0.5 - 0.64 * y;
/** y del confine occipitale (solco parieto-occipitale e incisura preoccipitale, in obliquo). */
export const occipitalBorderY = (z) => -63 - 0.18 * z;

/**
 * Lobo di un punto (mm MNI, qualsiasi emisfero) della corteccia: confini semplificati (solco centrale,
 * scissura laterale, linea parieto-occipitale) dello stesso modello stilizzato.
 * @returns {"frontale" | "parietale" | "temporale" | "occipitale" | "insula" | "limbico"}
 */
export function lobeOf(p) {
  const ax = Math.abs(p[0]);
  const y = p[1];
  const z = p[2];
  if (y < occipitalBorderY(z)) return "occipitale";
  // in fondo alla scissura laterale, coperta dagli opercoli: l'insula (solo i punti profondi)
  if (ax > 27 && ax < 41 && y > -30 && y < 22 && z > -12 && z < 20 && Math.abs(z - lateralFissureZ(y)) < 16) return "insula";
  // lobo temporale mediale (ippocampo, giro paraippocampale, entorinale)
  if (ax < 30 && z < -6 && y < 6 && y > -60) return "limbico";
  // sotto la scissura laterale e, dietro, sotto il confine con il lobo parietale (z ≈ 20)
  if (y > -64 && z < Math.min(lateralFissureZ(y), 20)) return "temporale";
  // giro del cingolo, sulla faccia interna sopra il corpo calloso
  if (ax < 9 && z > 0 && z < 40 && y > -45 && y < 40) return "limbico";
  if (y > centralSulcusY(z)) return "frontale";
  return "parietale";
}

// ---------------------------------------------------------------- cervelletto
/** Forma del cervelletto (mm MNI): centro, semiassi, numero di folia (sul mezzo giro dal basso all'alto). */
export const CEREBELLUM = Object.freeze({ c: [0, -64, -38], r: [48, 28, 22], folia: 23 });

/**
 * Cervelletto parametrico in latitudine e longitudine attorno a un asse verticale: le folia, che nel
 * cervelletto vero corrono da sinistra a destra attraverso verme ed emisferi, sono anelli orizzontali
 * (le righe della mesh), quindi restano nette e parallele sia di lato sia da dietro. Solco sulla linea
 * mediana dietro (tra i due emisferi, sopra il verme), incavo davanti per il tronco encefalico, faccia
 * superiore schiacciata sotto il tentorio, qualche fessura più profonda (come la fessura primaria).
 * @returns {{ positions, normals, indices, depth }} depth = 0 sulla cresta della folia, 1 in fondo
 */
export function buildCerebellum({ rows = 220, cols = 96, seed = 5 } = {}) {
  const { c, r, folia } = CEREBELLUM;
  const noise = createNoise3D(seed);
  const positions = new Float32Array((rows + 1) * (cols + 1) * 3);
  const depthAttr = new Float32Array((rows + 1) * (cols + 1));
  const fissures = [0.42, -0.08, -0.62]; // latitudini (rad) delle fessure profonde
  let v = 0;
  for (let i = 0; i <= rows; i++) {
    const psi = -Math.PI / 2 + (i / rows) * Math.PI; // latitudine: −90° sotto, +90° sopra
    const cp = Math.cos(psi);
    const sp = Math.sin(psi);
    for (let j = 0; j <= cols; j++) {
      const lambda = (j / cols) * Math.PI * 2; // longitudine attorno all'asse verticale
      const dx = cp * Math.cos(lambda);
      const dy = cp * Math.sin(lambda);
      const dz = sp;
      let rad = 1 / Math.sqrt((dx / r[0]) ** 2 + (dy / r[1]) ** 2 + (dz / r[2]) ** 2);
      // incavo davanti, al centro: lì passano il tronco encefalico e il quarto ventricolo
      rad *= 1 - 0.36 * Math.exp(-((dx / 0.32) ** 2)) * smoothstep(0.1, 0.8, dy);
      // incisura posteriore sulla linea mediana (il verme un po' più in dentro dei due emisferi)
      rad *= 1 - 0.07 * Math.exp(-((dx / 0.12) ** 2)) * smoothstep(-0.1, -0.7, dy);
      // folia: creste parallele con solchi stretti; poche fessure più profonde
      const wobble = 0.035 * noise(dx * 2.5, dy * 2.5, dz * 2.5);
      const valley = 0.5 - 0.5 * Math.cos(2 * folia * (psi + wobble));
      let fissure = 0;
      for (const f of fissures) fissure += Math.exp(-(((psi + wobble - f) / 0.03) ** 2));
      const groove = valley ** 4 * 1.5 + fissure * 3.5;
      rad -= groove;
      const px = c[0] + dx * rad;
      const py = c[1] + dy * rad;
      let pz = c[2] + dz * rad;
      // faccia superiore sotto il tentorio (che sale verso dietro)
      const top = tentoriumZ(py) - 1.5;
      if (pz > top) pz = top + (pz - top) * 0.15;
      positions[v * 3] = px;
      positions[v * 3 + 1] = py;
      positions[v * 3 + 2] = pz;
      depthAttr[v] = Math.min(1, groove / 3.2);
      v++;
    }
  }
  const indices = [];
  const row = cols + 1;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const a = i * row + j;
      const b = a + row;
      // antiorario visto da fuori: la longitudine (verso a + 1) e la latitudine (verso b) formano una
      // base destrorsa con la normale uscente
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }
  const mesh = computeNormals({ positions, indices: Uint32Array.from(indices) });
  // ai poli (sopra e sotto) i triangoli sono degeneri: la normale è verticale
  const nrm = mesh.normals;
  for (const i of [0, rows]) {
    for (let j = 0; j <= cols; j++) {
      const o = (i * row + j) * 3;
      nrm[o] = 0;
      nrm[o + 1] = 0;
      nrm[o + 2] = i === 0 ? -1 : 1;
    }
  }
  mesh.depth = depthAttr;
  return mesh;
}

// ---------------------------------------------------------------- tronco encefalico
/** Linea mediana del tronco encefalico (mm MNI) con le semilarghezze [laterale, antero-posteriore]. */
export const BRAINSTEM_PATH = Object.freeze([
  { p: [0, -14, 6], r: [11, 9] }, // sotto il talamo (diencefalo)
  { p: [0, -19, -8], r: [14, 11] }, // mesencefalo
  { p: [0, -22, -22], r: [15, 13] }, // ponte (rigonfio davanti)
  { p: [0, -26, -34], r: [15, 13] },
  { p: [0, -32, -47], r: [11, 9] }, // giunzione ponte-bulbo
  { p: [0, -38, -59], r: [9, 8] }, // bulbo
  { p: [0, -42, -72], r: [6.5, 6.5] }, // inizio del midollo spinale
]);
