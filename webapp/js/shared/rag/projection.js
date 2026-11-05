// Proiezione dei vettori in 3D per disegnarli nello spazio, con la PCA (analisi delle componenti
// principali): cerchiamo le 3 direzioni lungo cui i punti sono più "sparsi" e usiamo quelle come
// assi x, y, z. È deterministica (power iteration partendo da vettori iniziali con seed fisso)
// e lineare: anche la domanda dell'utente si proietta con la stessa formula, quindi finisce
// vicino ai chunk che le somigliano.
//
// Attenzione nella UI: 3 dimensioni su 256 conservano solo una parte delle distanze (la quota
// è in `explainedVariance`). Due punti vicini nel disegno possono essere lontani nello spazio
// originale: la ricerca vera usa sempre tutte le dimensioni.

import { createRng, gaussian } from "./random.js";

/**
 * @typedef {object} Projection
 * @property {number[][]} points            coordinate 3D (o `components`D) di ogni vettore in ingresso
 * @property {Float64Array[]} components    direzioni principali (vettori di norma 1)
 * @property {Float64Array} mean            media dei vettori (la PCA lavora sui dati centrati)
 * @property {number[]} variances           varianza lungo ogni componente
 * @property {number[]} explainedVariance   frazione della varianza totale spiegata da ogni componente
 * @property {number} totalVariance
 * @property {number} scale                 fattore applicato alle coordinate (vedi opzione `radius`)
 * @property {(vector: ArrayLike<number>) => number[]} project proietta un nuovo vettore
 */

/** Prodotto scalare tra un vettore e una riga centrata, senza allocare la riga. */
function centeredDot(row, mean, direction) {
  let sum = 0;
  for (let j = 0; j < direction.length; j++) sum += (row[j] - mean[j]) * direction[j];
  return sum;
}

/**
 * Calcola la PCA dei vettori e la loro proiezione.
 *
 * Power iteration: moltiplicando ripetutamente un vettore per la matrice di covarianza
 * C = Xᵀ X / n, il vettore si allinea alla direzione di massima varianza. Per la componente
 * successiva si toglie ogni volta la parte lungo le componenti già trovate (deflazione).
 * Non serve costruire C (256 × 256): Cv = Xᵀ (X v) / n.
 *
 * @param {ArrayLike<number>[]} vectors
 * @param {object} [options]
 * @param {number} [options.components=3]
 * @param {number} [options.iterations=200]  massimo di iterazioni per componente
 * @param {number} [options.tolerance=1e-10] arresto quando la direzione non cambia più
 * @param {number} [options.seed=1]
 * @param {number} [options.radius]          se dato, scala le coordinate perché il punto più
 *   lontano dall'origine stia a questa distanza (comodo per la scena 3D); la stessa scala vale per `project`
 * @returns {Projection}
 */
export function pca(vectors, { components = 3, iterations = 200, tolerance = 1e-10, seed = 1, radius } = {}) {
  if (!Array.isArray(vectors) || vectors.length === 0) throw new TypeError("pca: serve almeno un vettore");
  const n = vectors.length;
  const d = vectors[0].length;
  for (const v of vectors) if (v.length !== d) throw new RangeError("pca: i vettori devono avere la stessa dimensione");

  const mean = new Float64Array(d);
  for (const v of vectors) for (let j = 0; j < d; j++) mean[j] += v[j] / n;

  let totalVariance = 0;
  for (const v of vectors) for (let j = 0; j < d; j++) totalVariance += (v[j] - mean[j]) ** 2 / n;

  const rng = createRng(seed);
  const found = [];
  const variances = [];
  const projections = new Float64Array(n);
  for (let c = 0; c < Math.min(components, d); c++) {
    let direction = Float64Array.from({ length: d }, () => gaussian(rng));
    const initialLength = Math.hypot(...direction);
    for (let j = 0; j < d; j++) direction[j] /= initialLength;
    for (let iter = 0; iter < iterations; iter++) {
      const next = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        const p = centeredDot(vectors[i], mean, direction);
        for (let j = 0; j < d; j++) next[j] += (p * (vectors[i][j] - mean[j])) / n;
      }
      // Deflazione: la nuova direzione deve restare perpendicolare alle componenti già trovate.
      for (const previous of found) {
        let overlap = 0;
        for (let j = 0; j < d; j++) overlap += next[j] * previous[j];
        for (let j = 0; j < d; j++) next[j] -= overlap * previous[j];
      }
      const length = Math.hypot(...next);
      if (length < 1e-15) break; // nessuna varianza residua: dati tutti su un sottospazio più piccolo
      for (let j = 0; j < d; j++) next[j] /= length;
      let change = 0;
      for (let j = 0; j < d; j++) change += (next[j] - direction[j]) ** 2;
      direction = next;
      if (change < tolerance) break;
    }
    // Il segno di un autovettore è arbitrario: lo fissiamo (componente più grande positiva)
    // così la figura non si ribalta tra un'esecuzione e l'altra.
    let largest = 0;
    for (let j = 1; j < d; j++) if (Math.abs(direction[j]) > Math.abs(direction[largest])) largest = j;
    if (direction[largest] < 0) for (let j = 0; j < d; j++) direction[j] = -direction[j];
    // Varianza lungo la direzione trovata (è l'autovalore corrispondente).
    let variance = 0;
    for (let i = 0; i < n; i++) {
      projections[i] = centeredDot(vectors[i], mean, direction);
      variance += projections[i] ** 2 / n;
    }
    found.push(direction);
    variances.push(variance);
  }

  const raw = vectors.map((v) => found.map((component) => centeredDot(v, mean, component)));
  let scale = 1;
  if (radius !== undefined) {
    const farthest = Math.max(...raw.map((p) => Math.hypot(...p)));
    scale = farthest > 0 ? radius / farthest : 1;
  }
  const project = (vector) => {
    if (vector.length !== d) throw new RangeError("project: dimensione diversa da quella dei dati");
    return found.map((component) => centeredDot(vector, mean, component) * scale);
  };
  return {
    points: raw.map((p) => p.map((x) => x * scale)),
    components: found,
    mean,
    variances,
    explainedVariance: variances.map((v) => (totalVariance > 0 ? v / totalVariance : 0)),
    totalVariance,
    scale,
    project,
  };
}

/**
 * Scorciatoia per la scena: proiezione 3D con raggio fissato.
 *
 * @param {ArrayLike<number>[]} vectors
 * @param {number} [radius=10]
 * @returns {Projection}
 */
export function projectTo3D(vectors, radius = 10) {
  return pca(vectors, { components: 3, radius });
}
