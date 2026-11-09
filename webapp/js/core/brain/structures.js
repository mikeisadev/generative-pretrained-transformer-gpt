/**
 * Strutture profonde e organi di senso del cervello stilizzato, in millimetri MNI (emisfero sinistro:
 * il destro si ottiene specchiando x). Puro: restituisce mesh di array (geometry.js), three.js arriva
 * dopo (anatomy.js). Ogni struttura dice a quale regione dell'atlante appartiene (`region`), così
 * l'evidenziazione delle regioni del tab colora proprio lei.
 *
 * Forme semplificate ma riconoscibili, con le posizioni dei centroidi di atlas.js:
 *  - ippocampo: un tubo curvo dalla testa (davanti, gonfia, con le "digitazioni") alla coda che sale
 *    dietro il talamo, come un cavalluccio marino;
 *  - amigdala: una mandorla davanti alla testa dell'ippocampo; talamo: un uovo per lato;
 *  - striato: il caudato a "C" (testa, corpo sopra il talamo, coda che scende nel lobo temporale) e il
 *    putamen a lente; corpo calloso: un arco nella linea mediana dal ginocchio allo splenio;
 *  - VTA e substantia nigra (scura, per la neuromelanina); tronco encefalico; occhio con retina, nervo
 *    ottico, chiasma e tratto ottico fino al genicolato laterale; coclea a spirale (2,5 giri, ingrandita
 *    nel modello) con i canali semicircolari e il nervo acustico fino al nucleo cocleare.
 */
import { ellipsoidMesh, rotationMatrix, sampleCurve, sweepTube } from "./geometry.js";
import { BRAINSTEM_PATH } from "./shape.js";

/** Colori "anatomici" stilizzati (lo sfondo è scuro: toni chiari e poco saturi). */
export const STRUCTURE_COLORS = Object.freeze({
  hippocampus: "#e2b878",
  amygdala: "#e39588",
  thalamus: "#8fb0d6",
  striatum: "#b9a0d8",
  corpus_callosum: "#ece4d6",
  vta_snc: "#5b5560",
  brainstem: "#bdb2a6",
  eye: "#e7e9ee",
  retina: "#d9624a",
  iris: "#3b6fb6",
  nerve: "#e6d9b8",
  cochlea: "#eadfc4",
});

/** Coclea ingrandita nel modello (dichiarato nella legenda): vera ≈ 9 mm, qui ≈ 16 mm. */
export const COCHLEA_SCALE = 1.8;

const tube = (control, radius, { samples = 32, radial = 12, side = null, caps = true } = {}) => {
  const { points, lengths } = sampleCurve(control, samples);
  return sweepTube(points, { radius, radial, side, caps, lengths });
};

/** Ippocampo sinistro: testa (davanti, vicino all'amigdala), corpo, coda (dietro, sale). */
export function hippocampusMesh({ samples = 40, radial = 14 } = {}) {
  const control = [
    [-27, -9, -19],
    [-29, -15, -15],
    [-28, -22, -11],
    [-26, -29, -7],
    [-22, -35, -2],
    [-18, -38, 3],
  ];
  return tube(
    control,
    (t, angle) => {
      // testa gonfia con le digitazioni (piccoli bozzi), poi si assottiglia fino alla coda
      const head = Math.max(0, 1 - t / 0.35);
      const r = 2.6 + 4.6 * Math.pow(1 - t, 1.4);
      const digit = head * 0.9 * Math.max(0, Math.sin(angle * 4));
      return [r * 1.05 + digit, r * 0.82 + digit * 0.6];
    },
    { samples, radial, side: [0, 0, 1] },
  );
}

/** Amigdala sinistra: mandorla. */
export function amygdalaMesh() {
  return ellipsoidMesh([-23, -1, -17], [7.5, 8.5, 6.5], { widthSegments: 22, heightSegments: 14, rotation: rotationMatrix([0, 0, 1], -0.35) });
}

/** Talamo sinistro: un uovo (la parte mediale quasi tocca quella dell'altro lato). */
export function thalamusMesh() {
  return ellipsoidMesh([-11, -18, 8], [8.5, 15, 9], {
    widthSegments: 26,
    heightSegments: 16,
    rotation: rotationMatrix([0, 0, 1], 0.12),
    // più gonfio dietro, dove sporge il pulvinar
    bump: (x, y) => 1 + 0.1 * Math.max(0, -y),
  });
}

/** Striato sinistro: caudato (tubo a "C") + putamen (lente) + accumbens (che li unisce in basso). */
export function striatumMeshes() {
  const caudate = tube(
    [
      [-13, 15, 6],
      [-14, 8, 16],
      [-16, -6, 21],
      [-20, -20, 20],
      [-26, -30, 12],
      [-31, -28, 0],
      [-31, -18, -8],
    ],
    (t) => 1.2 + 5.2 * Math.pow(1 - t, 2.2),
    { samples: 42, radial: 12 },
  );
  const putamen = ellipsoidMesh([-25, 3, 2], [5.5, 15, 9.5], { widthSegments: 24, heightSegments: 14, rotation: rotationMatrix([0, 0, 1], -0.08) });
  const accumbens = ellipsoidMesh([-11, 11, -7], [5, 5.5, 4.5], { widthSegments: 16, heightSegments: 10 });
  return [caudate, putamen, accumbens];
}

/** VTA (vicino alla linea mediana) e substantia nigra (una mezzaluna di lato), lato sinistro. */
export function vtaSncMeshes() {
  const vta = ellipsoidMesh([-4, -16, -13], [3, 4.5, 3], { widthSegments: 14, heightSegments: 10 });
  const snc = tube(
    [
      [-7, -12, -14],
      [-11, -17, -12],
      [-12, -22, -9],
    ],
    () => [2.2, 1.6],
    { samples: 12, radial: 10 },
  );
  return [vta, snc];
}

/** Corpo calloso: arco nella linea mediana, largo in x, sottile in verticale (splenio più spesso). */
export function corpusCallosumMesh() {
  return tube(
    [
      [0, 15, -5],
      [0, 30, 4],
      [0, 28, 17],
      [0, 10, 25],
      [0, -12, 26],
      [0, -30, 22],
      [0, -39, 13],
    ],
    (t) => {
      const ends = Math.exp(-(((t - 0.12) / 0.12) ** 2)) * 1.8 + Math.exp(-(((t - 0.95) / 0.1) ** 2)) * 3.2;
      return [13 - 4 * Math.abs(t - 0.5), 2.6 + ends];
    },
    { samples: 48, radial: 14, side: [1, 0, 0] },
  );
}

/** Tronco encefalico (mesencefalo, ponte con il rigonfiamento davanti, bulbo, inizio del midollo). */
export function brainstemMesh({ samples = 40, radial = 20 } = {}) {
  const control = BRAINSTEM_PATH.map((s) => s.p);
  const radii = BRAINSTEM_PATH.map((s) => s.r);
  return tube(
    control,
    (t, angle) => {
      const f = t * (radii.length - 1);
      const i = Math.min(radii.length - 2, Math.floor(f));
      const w = f - i;
      const rx = radii[i][0] * (1 - w) + radii[i + 1][0] * w;
      const ry = radii[i][1] * (1 - w) + radii[i + 1][1] * w;
      // il ponte sporge davanti (verso +y): v ≈ direzione antero-posteriore della sezione
      const pons = Math.exp(-(((t - 0.42) / 0.14) ** 2));
      const front = Math.max(0, Math.sin(angle));
      return [rx, ry * (1 + 0.28 * pons * front)];
    },
    { samples, radial, side: [1, 0, 0] },
  );
}

/** Occhio sinistro (centro, raggio), retina e via ottica fino al genicolato laterale. */
export const EYE = Object.freeze({ center: [-32, 54, -38], radius: 12 });

export function eyeMeshes() {
  const { center, radius } = EYE;
  const sclera = ellipsoidMesh(center, [radius, radius, radius], { widthSegments: 28, heightSegments: 18 });
  // retina: la metà posteriore della parete interna (calotta)
  const retina = ellipsoidMesh([center[0], center[1] - 0.6, center[2]], [radius * 0.93, radius * 0.93, radius * 0.93], {
    widthSegments: 28,
    heightSegments: 18,
    rotation: rotationMatrix([1, 0, 0], Math.PI / 2), // poli lungo y: la calotta posteriore è metà sfera
  });
  keepHalf(retina, center, [0, -1, 0], 0.15);
  // iride e pupilla davanti (verso +y)
  const iris = ellipsoidMesh([center[0], center[1] + radius * 0.93, center[2]], [4.6, 1.2, 4.6], { widthSegments: 20, heightSegments: 8 });
  return { sclera, retina, iris };
}

/** Toglie i triangoli dalla parte opposta a `dir` (oltre `offset` raggi): una calotta da una sfera. */
function keepHalf(mesh, center, dir, offset) {
  const { positions, indices } = mesh;
  const kept = [];
  for (let f = 0; f < indices.length; f += 3) {
    let side = 0;
    for (let e = 0; e < 3; e++) {
      const o = indices[f + e] * 3;
      side += (positions[o] - center[0]) * dir[0] + (positions[o + 1] - center[1]) * dir[1] + (positions[o + 2] - center[2]) * dir[2];
    }
    if (side / 3 > -offset * EYE.radius) kept.push(indices[f], indices[f + 1], indices[f + 2]);
  }
  mesh.indices = Uint32Array.from(kept);
}

/**
 * Via ottica: nervo ottico sinistro dal fondo dell'occhio al chiasma, poi tratto ottico fino al
 * genicolato laterale (nel chiasma metà delle fibre passa dall'altra parte: qui un solo tubo per lato).
 */
export function opticPathMesh() {
  return tube(
    [
      [-29, 43, -38],
      [-20, 28, -26],
      [-7, 8, -16],
      [-2, 3, -14],
      [-8, -4, -13],
      [-15, -12, -10],
      [-21, -22, -7],
    ],
    (t) => (t < 0.45 ? 2.1 : 1.6),
    { samples: 40, radial: 10 },
  );
}

/** Coclea sinistra (spirale), canali semicircolari e nervo acustico fino al nucleo cocleare. */
export const COCHLEA = Object.freeze({ center: [-42, -18, -44] });

export function cochleaMeshes() {
  const s = COCHLEA_SCALE;
  const [cx, cy, cz] = COCHLEA.center;
  // spirale conica: 2,5 giri, raggio da 4,6 a 1,2 mm, alta 4 mm (poi ingrandita di COCHLEA_SCALE)
  const turns = 2.5;
  const control = [];
  const steps = 60;
  // asse della spirale verso l'esterno e un po' in avanti (come la coclea nella rocca petrosa)
  const axis = [-0.55, 0.65, -0.52];
  const ax = norm3(axis);
  const u = norm3(cross3(ax, [0, 0, 1]));
  const v = cross3(ax, u);
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const angle = t * turns * Math.PI * 2;
    const r = (4.6 - 3.4 * t) * s;
    const h = 4 * t * s;
    control.push([
      cx + (u[0] * Math.cos(angle) + v[0] * Math.sin(angle)) * r + ax[0] * h,
      cy + (u[1] * Math.cos(angle) + v[1] * Math.sin(angle)) * r + ax[1] * h,
      cz + (u[2] * Math.cos(angle) + v[2] * Math.sin(angle)) * r + ax[2] * h,
    ]);
  }
  const spiral = sweepTube(control, { radius: (t) => (1.5 - 0.9 * t) * s, radial: 12, caps: true });
  // tre canali semicircolari, perpendicolari tra loro, dietro la coclea
  const base = [cx + 3 * s, cy - 9 * s, cz + 3 * s];
  const canals = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ].map((n, k) => {
    const a = norm3(cross3(n, k === 2 ? [1, 0, 0] : [0, 0, 1]));
    const b = cross3(n, a);
    const ring = [];
    for (let i = 0; i <= 28; i++) {
      const ang = (i / 28) * Math.PI * 1.6 - 0.3;
      const rr = 3.1 * s;
      ring.push([base[0] + (a[0] * Math.cos(ang) + b[0] * Math.sin(ang)) * rr + a[0] * rr, base[1] + (a[1] * Math.cos(ang) + b[1] * Math.sin(ang)) * rr + a[1] * rr, base[2] + (a[2] * Math.cos(ang) + b[2] * Math.sin(ang)) * rr + a[2] * rr]);
    }
    return sweepTube(ring, { radius: () => 0.42 * s, radial: 8, caps: true });
  });
  // nervo acustico (VIII): dal centro della spirale al nucleo cocleare, al confine ponte-bulbo
  const nerve = tube(
    [
      [cx + 1, cy - 1, cz + 1],
      [-30, -28, -42],
      [-19, -36, -41],
      [-12, -41, -40],
    ],
    () => 1.5,
    { samples: 20, radial: 10 },
  );
  return { spiral, canals, nerve };
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm3(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
