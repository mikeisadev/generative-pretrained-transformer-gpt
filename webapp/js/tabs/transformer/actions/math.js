/**
 * Calcoli puri usati dalle azioni animate (nessun DOM, nessun three.js): così si testano in
 * Node (webapp/tests/transformer-actions.test.mjs) e restano gli stessi in tutte le viste.
 *
 * Convenzioni: matrici Float32Array piatte riga per riga, logaritmi naturali (nat), come
 * F.cross_entropy di PyTorch.
 */

/** Entropia di una distribuzione, in nat: H(p) = −Σ p·ln p (i termini con p = 0 valgono 0). */
export function entropy(probs) {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

/**
 * log-softmax numericamente stabile di una riga di logits (sottraendo il massimo, come fa
 * PyTorch): restituisce ln p_i per ogni i.
 */
export function logSoftmaxRow(logits, offset, V, out = new Float32Array(V)) {
  let max = -Infinity;
  for (let i = 0; i < V; i++) max = Math.max(max, logits[offset + i]);
  let sum = 0;
  for (let i = 0; i < V; i++) sum += Math.exp(logits[offset + i] - max);
  const logZ = max + Math.log(sum);
  for (let i = 0; i < V; i++) out[i] = logits[offset + i] - logZ;
  return out;
}

/** Softmax di logits / temperatura (una riga). È la formula di gpt/generation.py. */
export function softmaxWithTemperature(logits, temperature = 1, out = new Float32Array(logits.length)) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / temperature);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] / temperature - max);
    sum += out[i];
  }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

/**
 * Cross-entropy di una sequenza: logits (T × V) e target (T). Restituisce la media e, per ogni
 * posizione, la probabilità del target e la sua loss −ln p. Coincide con
 * F.cross_entropy(logits.view(B*T, V), targets.view(B*T)) per B = 1.
 */
export function crossEntropy(logits, targets, V) {
  const T = targets.length;
  const row = new Float32Array(V);
  const perPosition = new Float32Array(T);
  const targetProb = new Float32Array(T);
  let sum = 0;
  for (let t = 0; t < T; t++) {
    logSoftmaxRow(logits, t * V, V, row);
    const lp = row[targets[t]];
    perPosition[t] = -lp;
    targetProb[t] = Math.exp(lp);
    sum -= lp;
  }
  return { mean: sum / T, perPosition, targetProb };
}

/**
 * Gradiente della loss media rispetto ai logits di UNA posizione: (softmax − one-hot) / N.
 * È il primo numero che la backward calcola, e lo si ottiene in forma chiusa.
 */
export function logitsGradient(logits, offset, V, target, count = 1) {
  const grad = logSoftmaxRow(logits, offset, V);
  for (let i = 0; i < V; i++) grad[i] = (Math.exp(grad[i]) - (i === target ? 1 : 0)) / count;
  return grad;
}

/**
 * Punteggi di attention q·kᵀ/√hs per TUTTE le coppie (i, j), anche quelle future (j > i) che il
 * motore non calcola (le mette subito a −∞). Servono a mostrare i punteggi "prima della maschera".
 * q e k sono (T × hs) riga per riga; il risultato è (T × T).
 */
export function scaledScores(q, k, T, hs, out = new Float32Array(T * T)) {
  const scale = 1 / Math.sqrt(hs);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let c = 0; c < hs; c++) dot += q[i * hs + c] * k[j * hs + c];
      out[i * T + j] = dot * scale;
    }
  }
  return out;
}

/** Somma cumulativa nell'ordine dato (è la "ruota" di sample() in engine/sampling.js). */
export function cumulative(probs) {
  const out = new Float32Array(probs.length + 1);
  for (let i = 0; i < probs.length; i++) out[i + 1] = out[i] + probs[i];
  return out;
}

/** Norma L2 di ogni riga di una matrice (rows × cols). */
export function rowNorms(matrix, rows, cols) {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) s += matrix[r * cols + c] ** 2;
    out[r] = Math.sqrt(s);
  }
  return out;
}

/** Massimo valore assoluto (finito) di un array: la scala delle heatmap divergenti. */
export function maxAbs(values) {
  let m = 0;
  for (let i = 0; i < values.length; i++) {
    const v = Math.abs(values[i]);
    if (Number.isFinite(v) && v > m) m = v;
  }
  return m;
}

/** Interpolazione lineare elemento per elemento, scritta in `out` (nessuna allocazione). */
export function lerpInto(out, a, b, t) {
  for (let i = 0; i < out.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}

/**
 * Quantizzazione dei pesi come gpt/export.py (_quantize_signed): byte 0..255, 128 = zero,
 * simmetrica rispetto a `scale` (il massimo in valore assoluto del tensore addestrato). Una sola
 * implementazione per la web app, quella della libreria dei pesi (riesportata qui per chi la
 * importava da questo modulo): per valori finiti e scala > 0 è identica a quella che stava qui, in
 * più un NaN diventa lo zero (128) invece di un indice NaN nella tabella dei colori.
 */
export { quantizeSigned } from "../../../shared/weights/display.js";

/**
 * Disposizione dei caratteri di un testo in una griglia "da macchina da scrivere": si va a capo
 * dopo ogni "\n" e quando la riga raggiunge `cols` celle. Restituisce [{col, row}] per carattere.
 */
export function layoutText(chars, cols) {
  const cells = [];
  let col = 0;
  let row = 0;
  for (const ch of chars) {
    if (col >= cols) {
      col = 0;
      row++;
    }
    cells.push({ col, row });
    col++;
    if (ch === "\n") {
      col = 0;
      row++;
    }
  }
  return cells;
}

/** Il carattere come lo mostriamo: spazio e a capo diventano simboli visibili. */
export function visibleChar(ch) {
  return ch === "\n" ? "↵" : ch === " " ? "␣" : ch;
}

/**
 * Finestre di training da un testo già tokenizzato, come get_batch di gpt/data.py:
 * x = data[i : i+T], y = data[i+1 : i+T+1]. Gli indici di partenza si estraggono con `rng`.
 */
export function makeBatch(ids, blockSize, batchSize, rng) {
  if (ids.length <= blockSize) throw new RangeError("il testo è più corto di block_size + 1");
  const windows = [];
  for (let b = 0; b < batchSize; b++) {
    const start = Math.floor(rng() * (ids.length - blockSize));
    windows.push({ start, x: ids.slice(start, start + blockSize), y: ids.slice(start + 1, start + blockSize + 1) });
  }
  return windows;
}

/** Indice del segmento [a_i, a_{i+1}) che contiene x in una lista ordinata di punti (clamp ai bordi). */
export function segmentIndex(sortedPoints, x) {
  let lo = 0;
  let hi = sortedPoints.length - 1;
  if (x <= sortedPoints[0]) return 0;
  if (x >= sortedPoints[hi]) return hi - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sortedPoints[mid] <= x) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** Curva di easing per i movimenti (morbida in partenza e in arrivo). */
export const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * Limitatore di frequenza in tempo REALE: `ready()` è vero al massimo `perSecond` volte al secondo.
 * Serve agli aggiornamenti costosi (texture, grafici) dentro `update`: contare in base al
 * progresso del passo li moltiplicherebbe con la velocità del player (4× = 4 volte tanti).
 * @param {number} perSecond
 * @param {() => number} [now] orologio in millisecondi (iniettabile nei test)
 */
export function rateLimiter(perSecond, now = () => performance.now()) {
  const interval = 1000 / perSecond;
  let last = -Infinity;
  return {
    ready(force = false) {
      const t = now();
      if (!force && t - last < interval) return false;
      last = t;
      return true;
    },
    reset() {
      last = -Infinity;
    },
  };
}
