/**
 * Operazioni numeriche elementari del transformer, su Float32Array piatte (riga per riga).
 *
 * Perché cicli espliciti e non una libreria: il modello è piccolo (64 dimensioni), così
 * ogni formula del corso resta leggibile riga per riga. Gli accumulatori sono numeri
 * JavaScript (float64): il risultato, salvato in float32, è vicino a PyTorch entro ~1e-6.
 *
 * Convenzioni comuni:
 * - una matrice (rows × cols) è una Float32Array lunga almeno rows·cols, riga r a partire da r·cols;
 * - `first` e `rows` scelgono le righe su cui lavorare (stessa riga in ingresso e in uscita):
 *   in generazione serve calcolare solo l'ultima riga senza creare viste nuove;
 * - nessuna funzione alloca memoria: le uscite sono buffer preparati dal chiamante.
 */

/**
 * Strato lineare con il layout di `nn.Linear`: W ha forma [out, in] e y = x · Wᵀ + b.
 * @param {Float32Array} y uscita (righe × outDim)
 * @param {Float32Array} x ingresso (righe × inDim)
 * @param {Float32Array} W pesi [outDim, inDim]
 * @param {Float32Array|null} b bias [outDim] oppure null (le Q, K, V non hanno bias)
 */
export function linear(y, x, W, b, inDim, outDim, first, rows) {
  for (let t = first; t < first + rows; t++) {
    const xOff = t * inDim;
    const yOff = t * outDim;
    for (let o = 0; o < outDim; o++) {
      const wOff = o * inDim;
      let acc = 0;
      for (let i = 0; i < inDim; i++) acc += x[xOff + i] * W[wOff + i];
      y[yOff + o] = b === null ? acc : acc + b[o];
    }
  }
}

/**
 * LayerNorm come `nn.LayerNorm`: media e varianza "biased" (divisa per C, non C−1)
 * sull'ultima dimensione, poi γ (gamma) e β (beta) appresi.
 */
export function layerNorm(y, x, gamma, beta, C, eps, first, rows) {
  for (let t = first; t < first + rows; t++) {
    const off = t * C;
    let mean = 0;
    for (let i = 0; i < C; i++) mean += x[off + i];
    mean /= C;
    let variance = 0;
    for (let i = 0; i < C; i++) {
      const d = x[off + i] - mean;
      variance += d * d;
    }
    variance /= C;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let i = 0; i < C; i++) y[off + i] = (x[off + i] - mean) * inv * gamma[i] + beta[i];
  }
}

/** y[i] += x[i] per i in [0, n): la connessione residua. */
export function addInPlace(y, x, n) {
  for (let i = 0; i < n; i++) y[i] += x[i];
}

/** ReLU sul posto: max(0, x). */
export function reluInPlace(x, n) {
  for (let i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

/**
 * Softmax di `src[srcOff … srcOff+n)` scritta in `dst[dstOff …)`. Si sottrae il massimo
 * prima di exp (stessa distribuzione, nessun overflow). Gli -Infinity diventano 0.
 */
export function softmax(dst, dstOff, src, srcOff, n) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (src[srcOff + i] > max) max = src[srcOff + i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(src[srcOff + i] - max);
    dst[dstOff + i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) dst[dstOff + i] /= sum;
}

/**
 * Una head di self-attention causale, come `Head.forward` di gpt/attention.py.
 *
 *   scores = q · kᵀ · head_size^-0.5, con −∞ sopra la diagonale (il futuro non si guarda)
 *   att    = softmax(scores) riga per riga
 *   out    = att · v
 *
 * q, k, v, out sono (T × hs); scores e att sono (T × T) compatte (passo T).
 */
export function causalSelfAttention(q, k, v, scores, att, out, T, hs) {
  const scale = hs ** -0.5;
  for (let i = 0; i < T; i++) {
    const row = i * T;
    for (let j = 0; j < T; j++) {
      if (j > i) {
        scores[row + j] = -Infinity;
        continue;
      }
      let dot = 0;
      for (let d = 0; d < hs; d++) dot += q[i * hs + d] * k[j * hs + d];
      // Come PyTorch: prima il prodotto scalare in float32, poi la scalatura.
      scores[row + j] = Math.fround(dot) * scale;
    }
    softmax(att, row, scores, row, T);
    for (let d = 0; d < hs; d++) {
      let acc = 0;
      for (let j = 0; j <= i; j++) acc += att[row + j] * v[j * hs + d];
      out[i * hs + d] = acc;
    }
  }
}
