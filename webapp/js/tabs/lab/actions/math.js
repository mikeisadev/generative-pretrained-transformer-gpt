/**
 * I calcoli delle azioni del Laboratorio, passo per passo (logica pura: niente DOM, niente three.js;
 * test in tests/lab-actions.test.mjs, che li confrontano con il motore JS e con le differenze finite).
 *
 * Perché rifarli qui invece di leggere solo la traccia del motore (js/engine/gpt.js): la traccia dà i
 * risultati (k, i punteggi, l'uscita di ln1…), ma un'animazione deve mostrare anche i pezzi intermedi
 * che il motore non conserva: i 64 prodotti di una riga, la media e la varianza di una LayerNorm, i
 * punteggi del futuro prima della maschera, il gradiente. Ogni funzione segue lo STESSO ordine delle
 * operazioni dei kernel del motore (js/engine/kernels.js: somme in double, risultato salvato in float32),
 * così il numero finale mostrato coincide con quello del motore, bit per bit (i test lo controllano).
 *
 * Convenzioni: matrici Float32Array piatte riga per riga, nn.Linear con W di forma (uscite, ingressi) e
 * y = x · Wᵀ + b; logaritmi naturali (nat), come F.cross_entropy di PyTorch.
 */

/** eps di nn.LayerNorm (default di PyTorch, lo stesso del motore). */
export const LAYER_NORM_EPS = 1e-5;

/** Gli iperparametri di torch.optim.AdamW quando non se ne passano (gpt/training.py passa solo lr). */
export const ADAMW_DEFAULTS = Object.freeze({ beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01 });

// ---------------------------------------------------------------- nn.Linear

/**
 * Una riga di una nn.Linear: i prodotti x[i]·W[row, i], la loro somma, la bias e il valore finale.
 * @param {ArrayLike<number>} x ingresso (in)
 * @param {Float32Array} W pesi (out × in)
 * @param {Float32Array|null} b bias (out) o null (bias=False)
 * @param {number} inDim
 * @param {number} row l'uscita da calcolare
 * @returns {{products: Float32Array, sum: number, bias: number|null, value: number}} `value` è il numero
 *   float32 che salva il motore; `sum` la somma dei prodotti (in double, prima della bias)
 */
export function linearRow(x, W, b, inDim, row) {
  const products = new Float32Array(inDim);
  const off = row * inDim;
  let sum = 0;
  for (let i = 0; i < inDim; i++) {
    const p = x[i] * W[off + i];
    products[i] = p;
    sum += p; // stesso ordine del kernel: acc += x[i] * W[o * in + i]
  }
  const bias = b ? b[row] : null;
  return { products, sum, bias, value: Math.fround(b ? sum + bias : sum) };
}

/** y = x · Wᵀ + b per un vettore x (in) → Float32Array (out), come `linear` del motore. */
export function linearForward(x, W, b, inDim, outDim) {
  const y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    const off = o * inDim;
    let acc = 0;
    for (let i = 0; i < inDim; i++) acc += x[i] * W[off + i];
    y[o] = b ? acc + b[o] : acc;
  }
  return y;
}

/** Somme parziali dei prodotti (per il contatore che cresce durante l'animazione): s[k] = Σ_{i<k} p[i]. */
export function runningSums(products) {
  const out = new Float64Array(products.length + 1);
  for (let i = 0; i < products.length; i++) out[i + 1] = out[i] + products[i];
  return out;
}

// ---------------------------------------------------------------- self-attention di una head

/**
 * Punteggi q·kᵀ·hs^-0.5 per TUTTE le coppie (i, j), anche quelle del futuro (j > i) che il motore non
 * calcola (le mette subito a −∞): servono a mostrare i punteggi "prima della maschera". Per j ≤ i il
 * numero è esattamente quello del motore (prodotto scalare arrotondato a float32, poi la scala).
 * @param {Float32Array} q (T × hs)
 * @param {Float32Array} k (T × hs)
 * @returns {{dots: Float32Array, scaled: Float32Array, scale: number}} `dots` prima della scala (T × T)
 */
export function attentionScores(q, k, T, hs) {
  const scale = hs ** -0.5;
  const dots = new Float32Array(T * T);
  const scaled = new Float32Array(T * T);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let d = 0; d < hs; d++) dot += q[i * hs + d] * k[j * hs + d];
      dots[i * T + j] = dot;
      scaled[i * T + j] = Math.fround(dot) * scale;
    }
  }
  return { dots, scaled, scale };
}

/** La maschera causale (tril): sopra la diagonale −∞, come `masked_fill(tril[:T, :T] == 0, -inf)`. */
export function causalMask(scores, T) {
  const out = Float32Array.from(scores);
  for (let i = 0; i < T; i++) for (let j = i + 1; j < T; j++) out[i * T + j] = -Infinity;
  return out;
}

/**
 * Softmax di una riga (src[off … off+n) → dst[dstOff …)), con lo stesso procedimento del kernel del
 * motore: si sottrae il massimo, exp, si divide per la somma. Gli −∞ diventano 0.
 * @returns {{max: number, sum: number}} il massimo e la somma degli esponenziali (per la lavagna)
 */
export function softmaxInto(dst, dstOff, src, off, n) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (src[off + i] > max) max = src[off + i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(src[off + i] - max);
    dst[dstOff + i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) dst[dstOff + i] /= sum;
  return { max, sum };
}

/** Softmax riga per riga di una matrice (T × T) già mascherata: i pesi di attention. */
export function softmaxRows(masked, T) {
  const att = new Float32Array(T * T);
  for (let i = 0; i < T; i++) softmaxInto(att, i * T, masked, i * T, T);
  return att;
}

/** La riga i dei pesi di attention con i passaggi della softmax: esponenziali, somma, massimo. */
export function softmaxRowSteps(masked, T, i) {
  const exps = new Float64Array(T);
  let max = -Infinity;
  for (let j = 0; j < T; j++) if (masked[i * T + j] > max) max = masked[i * T + j];
  let sum = 0;
  for (let j = 0; j < T; j++) {
    exps[j] = Math.exp(masked[i * T + j] - max);
    sum += exps[j];
  }
  const probs = new Float32Array(T);
  for (let j = 0; j < T; j++) probs[j] = exps[j] / sum;
  return { max, exps, sum, probs };
}

/** out = att · v (T × hs): la media dei value pesata dall'attention (solo j ≤ i, come il motore). */
export function attentionOutput(att, v, T, hs) {
  const out = new Float32Array(T * hs);
  for (let i = 0; i < T; i++) {
    for (let d = 0; d < hs; d++) {
      let acc = 0;
      for (let j = 0; j <= i; j++) acc += att[i * T + j] * v[j * hs + d];
      out[i * hs + d] = acc;
    }
  }
  return out;
}

/** torch.cat(uscite delle head, dim=-1): la head h occupa le colonne h·hs … h·hs + hs − 1. */
export function concatHeads(outs, T, hs) {
  const H = outs.length;
  const C = H * hs;
  const out = new Float32Array(T * C);
  for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) for (let d = 0; d < hs; d++) out[t * C + h * hs + d] = outs[h][t * hs + d];
  return out;
}

/** a + b elemento per elemento (la connessione residua), in float32 come addInPlace del motore. */
export function addVectors(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

// ---------------------------------------------------------------- LayerNorm

/**
 * nn.LayerNorm di un vettore, passo per passo: media, varianza "biased" (divisa per C), 1/√(σ² + ε),
 * x − μ, la normalizzazione, × γ e + β. `out` coincide con il motore (stesso ordine delle operazioni).
 */
export function layerNormSteps(x, gamma, beta, eps = LAYER_NORM_EPS) {
  const C = x.length;
  let mean = 0;
  for (let i = 0; i < C; i++) mean += x[i];
  mean /= C;
  let variance = 0;
  for (let i = 0; i < C; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= C;
  const invStd = 1 / Math.sqrt(variance + eps);
  const centered = new Float32Array(C);
  const normalized = new Float32Array(C);
  const scaled = new Float32Array(C);
  const out = new Float32Array(C);
  for (let i = 0; i < C; i++) {
    const n = (x[i] - mean) * invStd;
    centered[i] = x[i] - mean;
    normalized[i] = n;
    scaled[i] = n * gamma[i];
    out[i] = n * gamma[i] + beta[i];
  }
  return { mean, variance, std: Math.sqrt(variance + eps), invStd, eps, centered, normalized, scaled, out };
}

/** Media e varianza (di popolazione) di un vettore: per dire "ora media 0 e varianza 1". */
export function meanVariance(v) {
  let mean = 0;
  for (let i = 0; i < v.length; i++) mean += v[i];
  mean /= v.length || 1;
  let variance = 0;
  for (let i = 0; i < v.length; i++) variance += (v[i] - mean) ** 2;
  return { mean, variance: variance / (v.length || 1) };
}

// ---------------------------------------------------------------- feed-forward

/** ReLU: max(0, x). `off` = neuroni spenti (valore ≤ 0 prima della ReLU, 0 dopo), `on` = accesi. */
export function relu(pre) {
  const post = new Float32Array(pre.length);
  let off = 0;
  for (let i = 0; i < pre.length; i++) {
    if (pre[i] > 0) post[i] = pre[i];
    else off++;
  }
  return { post, off, on: pre.length - off };
}

// ---------------------------------------------------------------- logits, probabilità, loss

/** Softmax di un vettore di logits → probabilità (Float32Array) e log-normalizzatore ln Σ e^z. */
export function softmax(logits) {
  const probs = new Float32Array(logits.length);
  const { max, sum } = softmaxInto(probs, 0, logits, 0, logits.length);
  return { probs, logZ: max + Math.log(sum) };
}

/** I k valori più grandi: [{id, value}] in ordine decrescente (a pari merito l'indice più piccolo). */
export function topK(values, k) {
  const order = Array.from(values, (_, i) => i).sort((a, b) => values[b] - values[a] || a - b);
  return order.slice(0, Math.max(0, k)).map((id) => ({ id, value: values[id] }));
}

/**
 * Cross-entropy di UNA posizione: −ln p(giusto). In double con il log-softmax stabile (come
 * F.cross_entropy), più le probabilità per i grafici.
 */
export function crossEntropy(logits, target) {
  const { probs, logZ } = softmax(logits);
  const logp = logits[target] - logZ;
  return { probs, loss: -logp, p: Math.exp(logp), logZ };
}

/**
 * Il gradiente ESATTO della loss di una posizione rispetto a lm_head (logits = h · Wᵀ + b, L = −ln p_y):
 *   ∂L/∂logits = p − onehot(y)          (V)
 *   ∂L/∂W      = (p − y) · hᵀ           (V × C): riga v = (p_v − y_v) · h
 *   ∂L/∂b      = p − y                  (V)
 * Nel training vero la loss è la media su B·T posizioni: il gradiente di ognuna è diviso per B·T e i
 * contributi si sommano (qui un esempio solo: niente divisione). Verificato con le differenze finite.
 * @param {ArrayLike<number>} probs (V) @param {number} target @param {ArrayLike<number>} h (C)
 */
export function lmHeadGradient(probs, target, h) {
  const V = probs.length;
  const C = h.length;
  const dLogits = new Float32Array(V);
  for (let v = 0; v < V; v++) dLogits[v] = probs[v] - (v === target ? 1 : 0);
  const dW = new Float32Array(V * C);
  for (let v = 0; v < V; v++) for (let j = 0; j < C; j++) dW[v * C + j] = dLogits[v] * h[j];
  return { dLogits, dW, db: Float32Array.from(dLogits) };
}

/**
 * Il gradiente che continua all'indietro, verso ln_f: ∂L/∂h = Wᵀ · (p − y), cioè
 * dh[j] = Σ_c (p_c − y_c) · W[c, j]. È il primo passo della backpropagation oltre lm_head (verificato
 * con le differenze finite); da qui il gradiente attraversa ln_f e i blocchi (qui non mostrato).
 */
export function lmHeadInputGradient(dLogits, W, C) {
  const V = dLogits.length;
  const dh = new Float32Array(C);
  for (let j = 0; j < C; j++) {
    let acc = 0;
    for (let c = 0; c < V; c++) acc += dLogits[c] * W[c * C + j];
    dh[j] = acc;
  }
  return dh;
}

/** Un passo di discesa del gradiente puro (SGD senza momento): w − lr · g. */
export function sgdStep(w, g, lr) {
  const out = new Float32Array(w.length);
  for (let i = 0; i < w.length; i++) out[i] = w[i] - lr * g[i];
  return out;
}

/**
 * Il PRIMO passo di torch.optim.AdamW (momenti a zero, step = 1), nello stesso ordine di PyTorch:
 *   w ← w · (1 − lr·λ)                         weight decay "disaccoppiato"
 *   m = (1−β₁)·g,  v = (1−β₂)·g²,  m̂ = m/(1−β₁),  v̂ = v/(1−β₂)
 *   w ← w − lr · m̂ / (√v̂ + ε)                  ≈ w − lr · segno(g): ogni peso si sposta di circa lr
 * Il training vero è al passo 5.000 con i momenti accumulati, che il checkpoint non salva: per questo
 * qui si mostra il primo passo di un AdamW appena creato (e lo si dice).
 */
export function adamwFirstStep(w, g, { lr, beta1 = ADAMW_DEFAULTS.beta1, beta2 = ADAMW_DEFAULTS.beta2, eps = ADAMW_DEFAULTS.eps, weightDecay = ADAMW_DEFAULTS.weightDecay } = {}) {
  const out = new Float32Array(w.length);
  const bc1 = 1 - beta1;
  const bc2Sqrt = Math.sqrt(1 - beta2);
  const stepSize = lr / bc1;
  for (let i = 0; i < w.length; i++) {
    const decayed = Math.fround(w[i] * (1 - lr * weightDecay));
    const m = Math.fround((1 - beta1) * g[i]);
    const v = Math.fround((1 - beta2) * g[i] * g[i]);
    const denom = Math.sqrt(v) / bc2Sqrt + eps;
    out[i] = decayed - stepSize * (m / denom);
  }
  return out;
}

/** Statistiche di uno spostamento dei pesi (nuovi − vecchi): |Δ| medio e massimo, quanti si sono mossi. */
export function updateStats(before, after) {
  let sum = 0;
  let max = 0;
  let moved = 0;
  for (let i = 0; i < before.length; i++) {
    const d = Math.abs(after[i] - before[i]);
    sum += d;
    if (d > max) max = d;
    if (d > 0) moved++;
  }
  return { meanAbs: sum / (before.length || 1), maxAbs: max, moved, count: before.length };
}

/** Lo scarto massimo |a − b| (per dire "uguale al motore": deve essere 0 o quasi). NaN se le lunghezze differiscono. */
export function maxAbsDiff(a, b) {
  if (!a || !b || a.length !== b.length) return NaN;
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m || Number.isNaN(d)) m = Number.isNaN(d) ? Infinity : d;
  }
  return m;
}
