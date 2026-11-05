// Misure di somiglianza e distanza tra vettori, con le stesse definizioni usate da pgvector:
//   <=>  cosine distance      = 1 - cos(a, b)
//   <->  distanza euclidea    = ||a - b||
//   <#>  inner product negato = -(a · b)
// Spring AI con distance-type COSINE_DISTANCE calcola score = 1 - distance, cioè la cosine similarity.

/**
 * Controlla che due vettori abbiano la stessa dimensione: confrontare vettori di modelli
 * diversi (es. 1536 contro 256 dimensioni) è un errore di configurazione, non un caso limite.
 *
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 */
function assertSameLength(a, b) {
  if (a.length !== b.length) {
    throw new RangeError(`Dimensioni diverse: ${a.length} contro ${b.length}`);
  }
}

/**
 * Prodotto scalare a · b = Σ aᵢ bᵢ.
 *
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @returns {number}
 */
export function dot(a, b) {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * Norma euclidea (lunghezza) ||a|| = √(a · a).
 *
 * @param {ArrayLike<number>} a
 * @returns {number}
 */
export function norm(a) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * a[i];
  return Math.sqrt(sum);
}

/**
 * Restituisce una copia del vettore con norma 1 (normalizzazione L2). Il vettore nullo resta
 * nullo: non ha una direzione, quindi non c'è un modo sensato di allungarlo.
 *
 * @param {ArrayLike<number>} a
 * @returns {Float32Array}
 */
export function normalize(a) {
  const length = norm(a);
  const out = new Float32Array(a.length);
  if (length === 0) return out;
  for (let i = 0; i < a.length; i++) out[i] = a[i] / length;
  return out;
}

/**
 * Cosine similarity: coseno dell'angolo tra i due vettori, in [-1, 1]. Guarda solo la
 * direzione (il "significato"), non la lunghezza. Con un vettore nullo vale 0.
 *
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @returns {number}
 */
export function cosineSimilarity(a, b) {
  assertSameLength(a, b);
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  if (aa === 0 || bb === 0) return 0;
  return ab / Math.sqrt(aa * bb);
}

/**
 * Cosine distance come l'operatore `<=>` di pgvector: 1 - cosine similarity, in [0, 2].
 * 0 = stessa direzione, 1 = perpendicolari, 2 = opposti.
 *
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @returns {number}
 */
export function cosineDistance(a, b) {
  return 1 - cosineSimilarity(a, b);
}

/**
 * Distanza euclidea ||a - b|| (operatore `<->` di pgvector).
 *
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @returns {number}
 */
export function euclideanDistance(a, b) {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Le metriche disponibili, con il nome usato da pgvector/Spring AI. Utile alla UI per
 * mostrare un selettore e spiegare l'operatore SQL corrispondente.
 */
export const METRICS = Object.freeze({
  COSINE_DISTANCE: { operator: "<=>", label: "Cosine distance", distance: cosineDistance },
  EUCLIDEAN_DISTANCE: { operator: "<->", label: "Distanza euclidea", distance: euclideanDistance },
  NEGATIVE_INNER_PRODUCT: { operator: "<#>", label: "Prodotto scalare (negato)", distance: (a, b) => -dot(a, b) },
});
