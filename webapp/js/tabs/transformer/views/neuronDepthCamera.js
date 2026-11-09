/**
 * Inquadrature della disposizione "Orizzontale con profondità" (logica pura, testata con node).
 *
 * Il punto di partenza sono i punti da far entrare (spigoli delle sezioni, culmini di residui e
 * parentesi: NetworkScene.framePoints) e una direzione di vista. I punti si proiettano sulla base
 * della camera (destra, su, profondità), si sceglie la distanza che li fa entrare nella parte
 * visibile dello stage e si sposta il punto guardato in modo che il loro centro cada al centro della
 * fascia libera tra i comandi in alto e la didascalia in basso. In orizzontale non serve spostare
 * niente: il SceneHost del tab sposta già il centro della proiezione nella parte non coperta dal
 * codice (view offset), quindi basta l'aspetto della parte visibile.
 */

/**
 * Direzioni della camera (dal punto guardato verso la camera). "side": di lato con un filo di
 * prospettiva (si intuisce lo spessore); "thick": tre quarti, da sinistra e dall'alto, con un angolo
 * dolce per tutta la rete (il flusso resta quasi orizzontale); "thickPart": più deciso per una parte
 * sola (le facce delle lastre, neuroni × token, si leggono meglio). Le stesse del nucleo (VIEW_DIRECTIONS).
 */
export const DEPTH_DIRECTIONS = Object.freeze({
  side: Object.freeze([-0.07, 0.1, 1]),
  thick: Object.freeze([-0.366, 0.438, 0.821]),
  thickPart: Object.freeze([-0.6, 0.45, 0.66]),
});

const normalize = (v) => {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Base della camera per una direzione di vista: destra, su (mai capovolta), direzione. */
export function cameraBasis(direction) {
  const d = normalize(direction);
  const worldUp = Math.abs(d[1]) > 0.98 ? [0, 0, -1] : [0, 1, 0];
  const right = normalize(cross(worldUp, d));
  const up = cross(d, right);
  return { right, up, d };
}

/**
 * Posa della camera che inquadra tutti i punti dalla direzione data.
 * @param {ArrayLike<number>[]} points punti [x, y, z] nel mondo
 * @param {ArrayLike<number>} direction dal punto guardato verso la camera
 * @param {object} o
 * @param {number} o.fov campo visivo verticale in gradi
 * @param {number} o.aspect larghezza / altezza della parte visibile dello stage
 * @param {number} [o.top] frazione dell'altezza occupata in alto (comandi)
 * @param {number} [o.bottom] frazione dell'altezza occupata in basso (didascalia, codice ridotto)
 * @param {number} [o.margin] aria attorno ai punti (1 = a filo)
 * @param {number} [o.minDistance] distanza minima dal punto guardato
 * @returns {{ position: number[], target: number[], distance: number }}
 */
export function framePose(points, direction, { fov, aspect, top = 0, bottom = 0, margin = 1.06, minDistance = 2 }) {
  const { right, up, d } = cameraBasis(direction);
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    const a = [dot(p, right), dot(p, up), dot(p, d)];
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], a[k]);
      hi[k] = Math.max(hi[k], a[k]);
    }
  }
  if (!Number.isFinite(lo[0])) return { position: [0, 0, minDistance], target: [0, 0, 0], distance: minDistance };
  const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const center = [0, 1, 2].map((k) => right[k] * mid[0] + up[k] * mid[1] + d[k] * mid[2]);
  const free = Math.max(0.2, 1 - top - bottom); // parte dell'altezza libera dai comandi
  const tanV = Math.tan((fov * Math.PI) / 360);
  const tanH = tanV * aspect;
  const halfR = (hi[0] - lo[0]) / 2;
  const halfU = (hi[1] - lo[1]) / 2;
  const halfD = (hi[2] - lo[2]) / 2;
  // la faccia più vicina dei punti sta a distanza − halfD: lì devono entrare in larghezza e in altezza
  const distance = Math.max(minDistance, Math.max(halfU / (tanV * free), halfR / tanH) * margin + halfD);
  // centro della fascia libera in coordinate normalizzate (−1 in basso, +1 in alto): ci si porta il centro dei punti
  const cy = bottom - top;
  const target = center.map((c, k) => c - up[k] * cy * distance * tanV);
  return { position: target.map((c, k) => c + d[k] * distance), target, distance };
}
