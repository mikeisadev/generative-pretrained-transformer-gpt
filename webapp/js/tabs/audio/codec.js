/**
 * Codec "giocattolo" calcolato davvero: encoder fisso → RVQ addestrata con k-means → decoder.
 *
 *   encoder : forma d'onda → un vettore di feature per frame (spettro di ampiezza in scala log,
 *             50 frame al secondo, come il frame rate dei token di MusicGen)
 *   RVQ     : ogni vettore diventa Q numeri interi (un indice per codebook); ogni codebook
 *             quantizza l'errore lasciato dai precedenti
 *   decoder : indici → somma dei centroidi → spettro → Griffin-Lim → forma d'onda
 *
 * Differenze dichiarate rispetto a SoundStream/EnCodec: lì encoder e decoder sono reti
 * convoluzionali addestrate insieme alla RVQ (con loss di ricostruzione e avversarie);
 * qui encoder e decoder sono fissi e solo i codebook vengono "imparati" (k-means).
 *
 * Modulo PURO, testabile in Node.
 */
import { griffinLim, stft } from "./dsp.js";

export const CODEC = Object.freeze({
  nFft: 1024, // 64 ms a 16 kHz
  hop: 320, // 20 ms → 50 frame al secondo
  bins: 257, // bin 0…256 = 0…4000 Hz: basta per i nostri strumenti
  logScale: 20, // feature = log(1 + 20·|X|): comprime la dinamica come fa l'orecchio
  codebookSize: 64, // 6 bit per indice
  levels: 8, // codebook addestrati (per il grafico dell'errore)
  used: 4, // codebook usati dal "MusicGen giocattolo" (come i 4 di MusicGen)
});

/* --------------------------------------------------------------------------------- encoder --- */

/** Forma d'onda → feature (frames × bins) del codec giocattolo. */
export function encodeFeatures(samples) {
  const { nFft, hop, bins, logScale } = CODEC;
  const spec = stft(samples, { nFft, hop, maxBins: bins });
  const features = new Float32Array(spec.mag.length);
  for (let i = 0; i < features.length; i++) features[i] = Math.log1p(logScale * spec.mag[i]);
  return { frames: spec.frames, dim: bins, features, mag: spec.mag };
}

/** Feature → forma d'onda (inversa del logaritmo, poi Griffin-Lim). */
export function decodeFeatures(features, frames, length, { iterations = 16, random } = {}) {
  const { nFft, hop, bins, logScale } = CODEC;
  const mag = new Float32Array(features.length);
  for (let i = 0; i < mag.length; i++) mag[i] = Math.max(0, Math.expm1(features[i]) / logScale);
  return griffinLim(mag, { frames, bins, nFft, hop, length, iterations, random });
}

/* --------------------------------------------------------------------------------- k-means --- */

function sqDist(a, ao, b, bo, dim) {
  let s = 0;
  for (let d = 0; d < dim; d++) {
    const v = a[ao + d] - b[bo + d];
    s += v * v;
  }
  return s;
}

/** Indice del centroide più vicino al vettore x[xo…xo+dim). */
export function nearest(centroids, k, x, xo, dim) {
  let best = 0;
  let bestD = Infinity;
  for (let c = 0; c < k; c++) {
    const d = sqDist(x, xo, centroids, c * dim, dim);
    if (d < bestD) (bestD = d), (best = c);
  }
  return best;
}

/**
 * k-means con inizializzazione k-means++ (i centri iniziali sono scelti lontani tra loro).
 * @param {Float32Array} data n × dim
 * @returns {Float32Array} centroids k × dim
 */
export function kmeans(data, n, dim, k, { iterations = 12, random = Math.random } = {}) {
  const centroids = new Float32Array(k * dim);
  const dist = new Float64Array(n).fill(Infinity);
  let pick = Math.floor(random() * n);
  for (let c = 0; c < k; c++) {
    centroids.set(data.subarray(pick * dim, pick * dim + dim), c * dim);
    let total = 0;
    for (let i = 0; i < n; i++) {
      dist[i] = Math.min(dist[i], sqDist(data, i * dim, centroids, c * dim, dim));
      total += dist[i];
    }
    if (total <= 0) {
      pick = Math.floor(random() * n); // tutti i punti coincidono con un centro: ne basta uno qualsiasi
      continue;
    }
    let r = random() * total;
    pick = n - 1;
    for (let i = 0; i < n; i++) if ((r -= dist[i]) <= 0) (pick = i), (i = n);
  }
  const assign = new Int32Array(n);
  const sums = new Float64Array(k * dim);
  const counts = new Int32Array(k);
  for (let it = 0; it < iterations; it++) {
    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      const c = nearest(centroids, k, data, i * dim, dim);
      assign[i] = c;
      counts[c]++;
      for (let d = 0; d < dim; d++) sums[c * dim + d] += data[i * dim + d];
    }
    for (let c = 0; c < k; c++) {
      if (!counts[c]) continue; // centroide senza punti: resta dov'è
      for (let d = 0; d < dim; d++) centroids[c * dim + d] = sums[c * dim + d] / counts[c];
    }
  }
  return centroids;
}

/* --------------------------------------------------------------------------------- RVQ --- */

/**
 * Addestra una quantizzazione vettoriale residua: il codebook l è il k-means dei residui
 * lasciati dai codebook 0…l-1 (come l'algoritmo 1 di SoundStream, ma senza rete).
 * Funzione asincrona: tra un livello e l'altro cede il controllo al browser.
 * @returns {Promise<{codebooks: Float32Array[], k: number, dim: number, mse: number[]}>}
 *          mse[l] = errore quadratico medio per componente usando l codebook (mse[0] = senza codebook)
 */
export async function trainRvq(data, n, dim, { levels = CODEC.levels, k = CODEC.codebookSize, random = Math.random, onLevel = () => {}, yieldFn = () => Promise.resolve() } = {}) {
  const residual = Float32Array.from(data);
  const codebooks = [];
  const mse = [meanSquare(residual)];
  for (let l = 0; l < levels; l++) {
    const cb = kmeans(residual, n, dim, k, { random });
    for (let i = 0; i < n; i++) {
      const c = nearest(cb, k, residual, i * dim, dim);
      for (let d = 0; d < dim; d++) residual[i * dim + d] -= cb[c * dim + d];
    }
    codebooks.push(cb);
    mse.push(meanSquare(residual));
    onLevel(l, mse);
    await yieldFn();
  }
  return { codebooks, k, dim, mse };
}

function meanSquare(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return s / x.length;
}

/**
 * Codifica: per ogni frame e per ogni codebook, l'indice del centroide più vicino al residuo.
 * @returns {Int32Array} codes q × frames (riga = codebook, come il tensore [n_q, T] di EnCodec)
 */
export function rvqEncode(rvq, data, n, q = rvq.codebooks.length) {
  const { dim, k } = rvq;
  const codes = new Int32Array(q * n);
  const r = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    r.set(data.subarray(i * dim, i * dim + dim));
    for (let l = 0; l < q; l++) {
      const cb = rvq.codebooks[l];
      const c = nearest(cb, k, r, 0, dim);
      codes[l * n + i] = c;
      for (let d = 0; d < dim; d++) r[d] -= cb[c * dim + d];
    }
  }
  return codes;
}

/** Decodifica: somma dei centroidi scelti nei primi q codebook. */
export function rvqDecode(rvq, codes, n, q) {
  if (q > codes.length / n) throw new RangeError("q oltre i codebook disponibili");
  const { dim } = rvq;
  const out = new Float32Array(n * dim);
  for (let l = 0; l < q; l++) {
    const cb = rvq.codebooks[l];
    for (let i = 0; i < n; i++) {
      const c = codes[l * n + i];
      for (let d = 0; d < dim; d++) out[i * dim + d] += cb[c * dim + d];
    }
  }
  return out;
}

/** Bit al secondo: frame/s × codebook × bit per indice. */
export const bitrate = (frameRate, q, k) => frameRate * q * Math.log2(k);

/* ------------------------------------------------------------- PCA 2D (per disegnare i codebook) --- */

/** Proiezione sulle prime due componenti principali (metodo delle potenze), per la scena 3D. */
export function pca2(data, n, dim, { iterations = 40, random = Math.random } = {}) {
  const mean = new Float64Array(dim);
  for (let i = 0; i < n; i++) for (let d = 0; d < dim; d++) mean[d] += data[i * dim + d] / n;
  const axes = [];
  for (let a = 0; a < 2; a++) {
    let v = Float64Array.from({ length: dim }, () => random() - 0.5);
    for (let it = 0; it < iterations; it++) {
      const next = new Float64Array(dim);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += (data[i * dim + d] - mean[d]) * v[d];
        for (let d = 0; d < dim; d++) next[d] += dot * (data[i * dim + d] - mean[d]);
      }
      for (const u of axes) {
        let p = 0;
        for (let d = 0; d < dim; d++) p += next[d] * u[d];
        for (let d = 0; d < dim; d++) next[d] -= p * u[d];
      }
      const len = Math.hypot(...next) || 1;
      v = next.map((x) => x / len);
    }
    axes.push(v);
  }
  return {
    mean,
    axes,
    project(x, xo = 0) {
      let px = 0;
      let py = 0;
      for (let d = 0; d < dim; d++) {
        const c = x[xo + d] - mean[d];
        px += c * axes[0][d];
        py += c * axes[1][d];
      }
      return [px, py];
    },
  };
}

/* ------------------------------------------------------------ "MusicGen giocattolo" --- */

/**
 * Delay pattern di MusicGen: al passo s il modello produce il token del codebook k per il
 * frame s − delays[k]. Con delays = [0, 1, 2, 3] i 4 codebook scorrono "a scala".
 * Restituisce per ogni passo l'elenco [{k, frame}] (frame < 0 o ≥ T → token speciale).
 */
export function delayPattern(frames, q, delays = Array.from({ length: q }, (_, k) => k)) {
  const steps = frames + Math.max(...delays);
  const out = [];
  for (let s = 0; s < steps; s++) out.push(delays.map((d, k) => ({ k, frame: s - d, valid: s - d >= 0 && s - d < frames })));
  return out;
}

/**
 * Modello statistico a conteggi AL POSTO del transformer (simulazione didattica).
 * Per il codebook 0 guarda il token precedente dello stesso codebook; per il codebook k > 0
 * guarda anche il token del codebook k − 1 dello stesso frame, che grazie al delay pattern
 * è già stato generato al passo prima. Un transformer vero impara dipendenze molto più lunghe.
 * Il "testo" (stile) sceglie quali conteggi usare: è il nostro condizionamento.
 */
export class TokenModel {
  /**
   * @param {{[style: string]: Int32Array}} codesByStyle codici q × T per stile
   * @param {number} q @param {number} k dimensione del codebook
   */
  constructor(codesByStyle, q, k, { alpha = 0.5 } = {}) {
    this.q = q;
    this.k = k;
    this.alpha = alpha;
    this.tables = {};
    for (const [style, codes] of Object.entries(codesByStyle)) {
      const T = codes.length / q;
      const uni = Array.from({ length: q }, () => new Float64Array(k).fill(1e-3));
      const ctx = Array.from({ length: q }, () => new Map());
      for (let l = 0; l < q; l++) {
        for (let t = 0; t < T; t++) {
          const tok = codes[l * T + t];
          uni[l][tok]++;
          const key = this.key(l, codes, T, t);
          if (!ctx[l].has(key)) ctx[l].set(key, new Float64Array(k));
          ctx[l].get(key)[tok]++;
        }
      }
      this.tables[style] = { uni, ctx, first: Array.from({ length: q }, (_, l) => codes[l * T]) };
    }
  }

  /** Contesto del token (codebook l, frame t): token precedente e, per l > 0, il codebook sopra. */
  key(l, codes, T, t) {
    const prev = t > 0 ? codes[l * T + t - 1] : -1;
    const above = l > 0 ? codes[(l - 1) * T + t] : -1;
    return prev * 4096 + above;
  }

  /** Distribuzione di probabilità del prossimo token (Float64Array di k valori che sommano a 1). */
  distribution(style, l, codes, T, t, temperature = 1) {
    const table = this.tables[style];
    const counts = table.ctx[l].get(this.key(l, codes, T, t));
    const uni = table.uni[l];
    let uniSum = 0;
    for (let i = 0; i < this.k; i++) uniSum += uni[i];
    const p = new Float64Array(this.k);
    let total = 0;
    for (let i = 0; i < this.k; i++) {
      // conteggio del contesto + un po' di probabilità "di riserva" dal conteggio generale
      const v = (counts ? counts[i] : 0) + (this.alpha * uni[i]) / uniSum;
      p[i] = Math.pow(v, 1 / Math.max(0.05, temperature));
      total += p[i];
    }
    for (let i = 0; i < this.k; i++) p[i] /= total;
    return p;
  }

  /**
   * Genera T frame seguendo il delay pattern. Restituisce i codici (q × T) e, per ogni passo,
   * cosa è stato prodotto (utile all'animazione e ai grafici).
   */
  generate(style, T, { temperature = 1, random = Math.random } = {}) {
    const codes = new Int32Array(this.q * T);
    const steps = [];
    for (const slots of delayPattern(T, this.q)) {
      const produced = [];
      for (const { k: l, frame, valid } of slots) {
        if (!valid) {
          produced.push({ k: l, frame, token: null });
          continue;
        }
        const p = this.distribution(style, l, codes, T, frame, temperature);
        let r = random();
        let tok = this.k - 1;
        for (let i = 0; i < this.k; i++) if ((r -= p[i]) <= 0) (tok = i), (i = this.k);
        if (frame === 0) tok = this.tables[style].first[l]; // il primo frame è il "prompt" audio
        codes[l * T + frame] = tok;
        produced.push({ k: l, frame, token: tok, p: l === 0 ? p : null });
      }
      steps.push(produced);
    }
    return { codes, steps };
  }
}

/** Conteggio dei token di un codebook (istogramma per il grafico "distribuzione dei token"). */
export function tokenCounts(codes, T, level, k) {
  const counts = new Array(k).fill(0);
  for (let t = 0; t < T; t++) counts[codes[level * T + t]]++;
  return counts;
}
