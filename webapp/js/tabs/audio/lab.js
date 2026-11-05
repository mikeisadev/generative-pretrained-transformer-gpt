/**
 * Il "laboratorio" del tab: calcola davvero, nel browser, tutto ciò che la scena mostra.
 *
 *   brani (sintesi additiva) → STFT e Mel per la scena → feature del codec → RVQ (k-means)
 *   → token per stile → modello dei token (al posto del transformer) → generazione → audio
 *
 * I calcoli pesanti (k-means, Griffin-Lim) cedono il controllo al browser tra un pezzo e
 * l'altro, così la pagina resta reattiva. I risultati audio sono messi in cache.
 */
import { CODEC, TokenModel, bitrate, decodeFeatures, encodeFeatures, nearest, pca2, rvqDecode, rvqEncode, tokenCounts, trainRvq } from "./codec.js";
import { applyMel, hzToMel, melFilterbank, melToHz, powerToDb, seededRandom, stft, whisperLogMel } from "./dsp.js";
import { fmt } from "../../core/dom.js";
import { SAMPLE_RATE, STYLES, synthesize } from "./song.js";

/** Errore relativo (0…1) come percentuale leggibile: 2 decimali sotto l'1 %. */
export const formatPercent = (v) => `${fmt(v * 100, v < 0.01 ? 2 : 1)} %`;

const nextTick = () => new Promise((resolve) => setTimeout(resolve, 0));

/** Righe della superficie 3D dello spettrogramma (asse delle frequenze). */
export const DISPLAY_ROWS = 128;
/** Bande Mel mostrate nei grafici (filtri sulla FFT da 1024 punti, 0…8 kHz). */
export const DISPLAY_MELS = 64;

/**
 * Viste dello stesso spettrogramma per animare STFT → Mel → log-Mel nella scena 3D.
 * Le righe della superficie sono equidistanti in scala Mel: la riga r ha frequenza rowHz[r].
 * Così lo stesso reticolo si può disegnare con l'asse lineare (y ∝ Hz) o Mel (y ∝ r).
 * Tutti i valori sono frames × DISPLAY_ROWS, normalizzati in 0…1:
 *   linear : |X| (massimo dei bin che cadono nella riga)
 *   mel    : √(energia della banda Mel) interpolata alla frequenza della riga
 *   logMel : la stessa energia in decibel (70 dB di dinamica)
 */
function displaySpectra(samples) {
  const nFft = CODEC.nFft;
  const spec = stft(samples, { nFft, hop: CODEC.hop });
  const bank = melFilterbank({ sampleRate: SAMPLE_RATE, nFft, nMels: DISPLAY_MELS });
  const mel = applyMel(spec, bank);
  const melDb = powerToDb(mel, { range: 70 });
  const T = spec.frames;
  const R = DISPLAY_ROWS;
  const nyquist = SAMPLE_RATE / 2;
  const melMax = hzToMel(nyquist);
  const rowHz = Float32Array.from({ length: R }, (_, r) => melToHz((r / (R - 1)) * melMax));
  const binHz = SAMPLE_RATE / nFft;
  const linear = new Float32Array(T * R);
  const melAmp = new Float32Array(T * R);
  const logMel = new Float32Array(T * R);
  // per ogni riga: intervallo di bin e posizione tra due bande Mel (interpolazione lineare)
  const binRange = [];
  const bandPos = [];
  for (let r = 0; r < R; r++) {
    const lo = r > 0 ? (rowHz[r - 1] + rowHz[r]) / 2 : 0;
    const hi = r < R - 1 ? (rowHz[r] + rowHz[r + 1]) / 2 : nyquist;
    const a = Math.min(spec.bins - 1, Math.round(lo / binHz));
    binRange.push([a, Math.max(a + 1, Math.min(spec.bins, Math.round(hi / binHz)))]);
    let m = 0;
    while (m < DISPLAY_MELS - 1 && bank.centers[m + 1] < rowHz[r]) m++;
    const f = m < DISPLAY_MELS - 1 ? Math.min(1, Math.max(0, (rowHz[r] - bank.centers[m]) / (bank.centers[m + 1] - bank.centers[m]))) : 0;
    bandPos.push([m, Math.min(DISPLAY_MELS - 1, m + 1), f]);
  }
  let maxLin = 0;
  let maxMel = 0;
  let lo = Infinity;
  let hi = -Infinity;
  for (let t = 0; t < T; t++) {
    for (let r = 0; r < R; r++) {
      const [a, b] = binRange[r];
      let v = 0;
      for (let k = a; k < b; k++) v = Math.max(v, spec.mag[t * spec.bins + k]);
      const i = t * R + r;
      linear[i] = v;
      maxLin = Math.max(maxLin, v);
      const [m0, m1, f] = bandPos[r];
      const p = mel[t * DISPLAY_MELS + m0] * (1 - f) + mel[t * DISPLAY_MELS + m1] * f;
      melAmp[i] = Math.sqrt(p);
      maxMel = Math.max(maxMel, melAmp[i]);
      logMel[i] = melDb[t * DISPLAY_MELS + m0] * (1 - f) + melDb[t * DISPLAY_MELS + m1] * f;
      lo = Math.min(lo, logMel[i]);
      hi = Math.max(hi, logMel[i]);
    }
  }
  for (let i = 0; i < T * R; i++) {
    linear[i] /= maxLin || 1;
    melAmp[i] /= maxMel || 1;
    logMel[i] = (logMel[i] - lo) / (hi - lo || 1);
  }
  return { frames: T, rows: R, rowHz, linear, mel: melAmp, logMel, melDb, bank, spec };
}

/**
 * Costruisce il laboratorio. onProgress(testo, frazione) per la barra di caricamento.
 * @returns {Promise<AudioLab>}
 */
export async function createLab({ onProgress = () => {} } = {}) {
  const random = seededRandom(2026);
  onProgress("Sintesi dei brani…", 0.05);
  const songs = {};
  for (const style of Object.values(STYLES)) songs[style.id] = { style, ...synthesize(style) };
  await nextTick();

  onProgress("STFT e banco di filtri Mel…", 0.15);
  const display = {};
  const features = {};
  for (const id of Object.keys(songs)) {
    display[id] = displaySpectra(songs[id].samples);
    features[id] = encodeFeatures(songs[id].samples);
    await nextTick();
  }

  // Codebook condivisi tra i due stili: un unico "vocabolario" di pezzi di suono.
  const ids = Object.keys(songs);
  const dim = CODEC.bins;
  const frames = features[ids[0]].frames;
  const all = new Float32Array(ids.length * frames * dim);
  ids.forEach((id, i) => all.set(features[id].features, i * frames * dim));
  const rvq = await trainRvq(all, ids.length * frames, dim, {
    random,
    yieldFn: nextTick,
    onLevel: (l) => onProgress(`k-means del codebook ${l + 1} di ${CODEC.levels}…`, 0.25 + (0.6 * (l + 1)) / CODEC.levels),
  });

  // Riferimento per l'errore: la varianza delle feature, cioè l'errore di chi rispondesse
  // sempre con il "frame medio" (zero codebook).
  const variance = featureVariance(all, ids.length * frames, dim);

  onProgress("Token e modello dei token…", 0.9);
  const codes = {};
  for (const id of ids) codes[id] = rvqEncode(rvq, features[id].features, frames, CODEC.levels);
  const firstQ = (c) => c.subarray(0, CODEC.used * frames);
  const model = new TokenModel(Object.fromEntries(ids.map((id) => [id, firstQ(codes[id])])), CODEC.used, CODEC.codebookSize);

  // Nuvole dei codebook in 2D (PCA dei residui di ogni livello) per la scena.
  const clouds = codebookClouds(rvq, all, ids.length * frames, dim, random);
  await nextTick();
  onProgress("Pronto", 1);

  const audioCache = new Map();
  /** Ricostruzione di uno stile con i primi q codebook (Griffin-Lim, in cache). */
  async function reconstruct(id, q) {
    const key = `rec:${id}:${q}`;
    if (!audioCache.has(key)) {
      await nextTick();
      const feats = rvqDecode(rvq, codes[id], frames, q);
      audioCache.set(key, decodeFeatures(feats, frames, songs[id].samples.length, { random: seededRandom(11) }));
    }
    return audioCache.get(key);
  }

  /** Genera un brano nuovo token per token (delay pattern) e ne calcola l'audio solo su richiesta. */
  function generate(id, { temperature = 0.9, seed = 1 } = {}) {
    const result = model.generate(id, frames, { temperature, random: seededRandom(seed) });
    let audio = null;
    return {
      ...result,
      style: id,
      async audio() {
        if (!audio) {
          await nextTick();
          const feats = rvqDecode(rvq, result.codes, frames, CODEC.used);
          audio = decodeFeatures(feats, frames, songs[id].samples.length, { random: seededRandom(seed + 100) });
        }
        return audio;
      },
    };
  }

  const whisperCache = new Map();
  return {
    sampleRate: SAMPLE_RATE,
    frames,
    frameRate: SAMPLE_RATE / CODEC.hop,
    songs,
    display,
    features,
    rvq,
    codes,
    model,
    clouds,
    reconstruct,
    generate,
    /** log-Mel con gli stessi parametri e la stessa formula di Whisper (3000 × 80). */
    whisper(id) {
      if (!whisperCache.has(id)) whisperCache.set(id, whisperLogMel(songs[id].samples));
      return whisperCache.get(id);
    },
    /** Errore relativo con 0…levels codebook: 1 = come rispondere sempre con il frame medio. */
    relativeError() {
      return rvq.mse.map((v, l) => (l === 0 ? 1 : v / variance));
    },
    bitrate: (q) => bitrate(SAMPLE_RATE / CODEC.hop, q, CODEC.codebookSize),
    tokenCounts: (c, level) => tokenCounts(c, frames, level, CODEC.codebookSize),
  };
}

function featureVariance(data, n, dim) {
  const mean = new Float64Array(dim);
  for (let i = 0; i < n; i++) for (let d = 0; d < dim; d++) mean[d] += data[i * dim + d] / n;
  let s = 0;
  for (let i = 0; i < n; i++) for (let d = 0; d < dim; d++) s += (data[i * dim + d] - mean[d]) ** 2;
  return s / (n * dim);
}

/**
 * Per ogni livello: la PCA 2D dei residui, la posizione dei centroidi e quella di ogni frame.
 * Serve solo a disegnare: il codec lavora nello spazio completo a 257 dimensioni.
 */
function codebookClouds(rvq, data, n, dim, random) {
  const residual = Float32Array.from(data);
  const out = [];
  for (let l = 0; l < CODEC.used; l++) {
    const cb = rvq.codebooks[l];
    // per la PCA bastano 160 frame presi a intervalli regolari
    const step = Math.max(1, Math.floor(n / 160));
    const sample = new Float32Array(Math.ceil(n / step) * dim);
    let m = 0;
    for (let i = 0; i < n; i += step, m++) sample.set(residual.subarray(i * dim, i * dim + dim), m * dim);
    const p = pca2(sample, m, dim, { random, iterations: 20 });
    const centroids = new Float32Array(rvq.k * 2);
    for (let c = 0; c < rvq.k; c++) centroids.set(p.project(cb, c * dim), c * 2);
    const points = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) points.set(p.project(residual, i * dim), i * 2);
    // scala robusta: l'85° percentile delle coordinate dei centroidi (pochi outlier non schiacciano la nuvola)
    const mags = Array.from(centroids, Math.abs).sort((a, b) => a - b);
    const scale = mags[Math.floor(mags.length * 0.85)] * 1.25 || 1;
    out.push({ centroids, points, scale });
    for (let i = 0; i < n; i++) {
      const best = nearest(cb, rvq.k, residual, i * dim, dim);
      for (let j = 0; j < dim; j++) residual[i * dim + j] -= cb[best * dim + j];
    }
  }
  return out;
}
