/**
 * Elaborazione del segnale per il tab "Audio e musica": tutto calcolato davvero nel browser.
 *
 * Modulo PURO (niente DOM, niente Three.js): si può importare anche da Node per i controlli.
 * Contiene: finestra di Hann, FFT radix-2, DFT diretta (per lunghezze non potenze di 2, come
 * le 400 di Whisper), STFT, banco di filtri Mel (formula di Slaney come librosa, o HTK),
 * log-Mel "alla Whisper", ISTFT e Griffin-Lim (per tornare dallo spettro al suono).
 */

/* ------------------------------------------------------------------------------ finestre --- */

/** Finestra di Hann "periodica" (come torch.hann_window): attenua i bordi del frame. */
export function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/* ------------------------------------------------------------------------------------ FFT --- */

const fftTables = new Map();

/** Tabelle (bit-reversal, seni e coseni) calcolate una volta per dimensione. */
function tablesFor(n) {
  if (fftTables.has(n)) return fftTables.get(n);
  if (n & (n - 1)) throw new RangeError(`FFT: la lunghezza deve essere una potenza di 2 (ricevuto ${n})`);
  const rev = new Uint32Array(n);
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = -Math.sin((2 * Math.PI * i) / n);
  }
  const t = { rev, cos, sin };
  fftTables.set(n, t);
  return t;
}

/**
 * FFT di Cooley-Tukey sul posto: re/im vengono sovrascritti con X[k] = Σ x[n]·e^{-2πikn/N}.
 * @param {Float64Array|Float32Array} re @param {Float64Array|Float32Array} im
 * @param {boolean} [inverse=false] trasformata inversa (già divisa per N)
 */
export function fft(re, im, inverse = false) {
  const n = re.length;
  const { rev, cos, sin } = tablesFor(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  const sign = inverse ? -1 : 1;
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step];
        const wi = sign * sin[k * step];
        const a = start + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) (re[i] /= n), (im[i] /= n);
}

/* ----------------------------------------------------------------------------------- STFT --- */

/**
 * Short-Time Fourier Transform "centrata" (come torch.stft con center=True): il segnale viene
 * esteso di nFft/2 campioni per lato, così il frame t è centrato sul campione t·hop.
 * Restituisce le ampiezze |X| di ogni frame (righe = frame, colonne = bin 0…nFft/2).
 * @param {Float32Array} x segnale
 * @param {object} o
 * @param {number} o.nFft lunghezza del frame (potenza di 2 per la FFT; altrimenti DFT diretta)
 * @param {number} o.hop passo tra un frame e il successivo
 * @param {"reflect"|"zeros"} [o.pad="reflect"]
 * @param {boolean} [o.phase=false] restituisce anche la fase (per la ricostruzione)
 * @param {number} [o.maxBins] tiene solo i primi maxBins bin (es. fino a 4 kHz)
 * @param {number} [o.maxFrames] calcola solo i primi maxFrames frame
 * @returns {{frames: number, bins: number, mag: Float32Array, phase?: Float32Array}}
 */
export function stft(x, { nFft, hop, pad = "reflect", phase = false, maxBins, maxFrames = Infinity } = {}) {
  const half = nFft >> 1;
  const frames = Math.min(maxFrames, 1 + Math.floor(x.length / hop));
  const bins = Math.min(half + 1, maxBins ?? half + 1);
  const mag = new Float32Array(frames * bins);
  const ph = phase ? new Float32Array(frames * bins) : null;
  const window = hannWindow(nFft);
  const pow2 = (nFft & (nFft - 1)) === 0;
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const dft = pow2 ? null : dftTables(nFft, bins);
  const sample = (i) => {
    if (i >= 0 && i < x.length) return x[i];
    if (pad === "zeros") return 0;
    // riflessione senza ripetere il bordo (come numpy/torch "reflect")
    const n = x.length;
    let j = i < 0 ? -i : 2 * (n - 1) - i;
    if (j < 0 || j >= n) j = Math.max(0, Math.min(n - 1, j));
    return x[j];
  };
  for (let t = 0; t < frames; t++) {
    const start = t * hop - half;
    for (let i = 0; i < nFft; i++) {
      re[i] = sample(start + i) * window[i];
      im[i] = 0;
    }
    if (pow2) fft(re, im);
    else dft(re, im);
    const o = t * bins;
    for (let k = 0; k < bins; k++) {
      mag[o + k] = Math.hypot(re[k], im[k]);
      if (ph) ph[o + k] = Math.atan2(im[k], re[k]);
    }
  }
  return { frames, bins, mag, ...(ph ? { phase: ph } : {}) };
}

/** DFT diretta X[k] = Σ x[n] e^{-2πikn/N} per i primi `bins` coefficienti (N qualsiasi, es. 400). */
function dftTables(n, bins) {
  const cos = new Float64Array(n * bins);
  const sin = new Float64Array(n * bins);
  for (let k = 0; k < bins; k++) {
    for (let i = 0; i < n; i++) {
      const a = (-2 * Math.PI * k * i) / n;
      cos[k * n + i] = Math.cos(a);
      sin[k * n + i] = Math.sin(a);
    }
  }
  return (re, im) => {
    const out = new Float64Array(2 * bins);
    for (let k = 0; k < bins; k++) {
      let sr = 0;
      let si = 0;
      const o = k * n;
      for (let i = 0; i < n; i++) {
        sr += re[i] * cos[o + i];
        si += re[i] * sin[o + i];
      }
      out[2 * k] = sr;
      out[2 * k + 1] = si;
    }
    for (let k = 0; k < bins; k++) {
      re[k] = out[2 * k];
      im[k] = out[2 * k + 1];
    }
  };
}

/* ------------------------------------------------------------------------------- scala Mel --- */

/**
 * Da hertz a mel. Due varianti in uso:
 * - HTK: m = 2595·log10(1 + f/700)
 * - Slaney (default di librosa, quindi anche dei filtri di Whisper): lineare sotto 1 kHz,
 *   logaritmica sopra.
 */
export function hzToMel(f, { htk = false } = {}) {
  if (htk) return 2595 * Math.log10(1 + f / 700);
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return f < minLogHz ? f / fSp : minLogMel + Math.log(f / minLogHz) / logStep;
}

export function melToHz(m, { htk = false } = {}) {
  if (htk) return 700 * (Math.pow(10, m / 2595) - 1);
  const fSp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fSp;
  const logStep = Math.log(6.4) / 27;
  return m < minLogMel ? fSp * m : minLogHz * Math.exp(logStep * (m - minLogMel));
}

/**
 * Banco di filtri Mel triangolari, stessa costruzione di librosa.filters.mel
 * (fmin = 0, fmax = sr/2, normalizzazione "slaney": ogni triangolo ha area costante).
 * @returns {{nMels: number, bins: number, weights: Float32Array, centers: Float32Array}}
 *          weights: riga per filtro (nMels × bins); centers: frequenza centrale in Hz
 */
export function melFilterbank({ sampleRate, nFft, nMels, fMin = 0, fMax = sampleRate / 2, htk = false, normalize = true }) {
  const bins = 1 + (nFft >> 1);
  const melMin = hzToMel(fMin, { htk });
  const melMax = hzToMel(fMax, { htk });
  const edges = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) edges[i] = melToHz(melMin + ((melMax - melMin) * i) / (nMels + 1), { htk });
  const weights = new Float32Array(nMels * bins);
  for (let m = 0; m < nMels; m++) {
    const lo = edges[m];
    const mid = edges[m + 1];
    const hi = edges[m + 2];
    const norm = normalize ? 2 / (hi - lo) : 1;
    for (let k = 0; k < bins; k++) {
      const f = (k * sampleRate) / nFft;
      const up = (f - lo) / (mid - lo);
      const down = (hi - f) / (hi - mid);
      const w = Math.max(0, Math.min(up, down));
      weights[m * bins + k] = w * norm;
    }
  }
  return { nMels, bins, weights, centers: Float32Array.from({ length: nMels }, (_, m) => edges[m + 1]) };
}

/**
 * Applica il banco di filtri a uno spettro (frames × binsSpec, ampiezze) → energie Mel (frames × nMels).
 * Usa la potenza |X|² come Whisper e librosa.feature.melspectrogram.
 */
export function applyMel(spec, bank) {
  const { frames, bins: specBins, mag } = spec;
  const { nMels, bins } = bank;
  const use = Math.min(specBins, bins);
  const out = new Float32Array(frames * nMels);
  for (let t = 0; t < frames; t++) {
    const s = t * specBins;
    for (let m = 0; m < nMels; m++) {
      let acc = 0;
      const w = m * bins;
      for (let k = 0; k < use; k++) {
        const a = mag[s + k];
        acc += bank.weights[w + k] * a * a;
      }
      out[t * nMels + m] = acc;
    }
  }
  return out;
}

/** Decibel 10·log10(potenza), con un pavimento: i valori sotto (max − range) dB vengono alzati. */
export function powerToDb(values, { range = 80, floor = 1e-10 } = {}) {
  const out = new Float32Array(values.length);
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    out[i] = 10 * Math.log10(Math.max(floor, values[i]));
    if (out[i] > max) max = out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i], max - range);
  return out;
}

/* ---------------------------------------------------------------------------- log-Mel Whisper --- */

/** Iperparametri audio di Whisper (whisper/audio.py nel repository openai/whisper). */
export const WHISPER = Object.freeze({ sampleRate: 16000, nFft: 400, hop: 160, chunkSeconds: 30, nFrames: 3000, nMels: 80 });

/**
 * Lo stesso calcolo di whisper.audio.log_mel_spectrogram: STFT (n_fft 400, hop 160, Hann),
 * potenza, filtri Mel (80), log10 con pavimento 1e-10, compressione della dinamica a 8
 * (cioè 80 dB) e riscalatura (x + 4) / 4. Il segnale è riempito di silenzio fino a 30 s
 * (3000 frame): i frame di solo silenzio hanno potenza 0, quindi non serve calcolarne la DFT.
 * @param {Float32Array} audio a 16 kHz
 * @returns {{frames: number, nMels: number, values: Float32Array, signalFrames: number}}
 *          values: frame per frame (3000 × 80)
 */
export function whisperLogMel(audio) {
  const { nFft, hop, nFrames, nMels, sampleRate } = WHISPER;
  // torch.stft(center=True) su (audio + 30 s di silenzio): i frame oltre il segnale vedono solo zeri.
  const signalFrames = Math.min(nFrames, Math.ceil((audio.length + nFft / 2) / hop));
  // a sinistra torch riflette il segnale (pad_mode "reflect"), a destra trova il silenzio aggiunto
  const padded = new Float32Array(signalFrames * hop + nFft);
  padded.set(audio);
  const spec = stft(padded, { nFft, hop, maxFrames: signalFrames });
  const bank = melFilterbank({ sampleRate, nFft, nMels });
  const mel = applyMel({ frames: signalFrames, bins: spec.bins, mag: spec.mag }, bank);
  const values = new Float32Array(nFrames * nMels).fill(-10); // log10(1e-10) = silenzio
  let max = -Infinity;
  for (let i = 0; i < mel.length; i++) {
    const v = Math.log10(Math.max(1e-10, mel[i]));
    values[i] = v;
    if (v > max) max = v;
  }
  for (let i = 0; i < values.length; i++) values[i] = (Math.max(values[i], max - 8) + 4) / 4;
  return { frames: nFrames, nMels, values, signalFrames: Math.min(signalFrames, Math.round(audio.length / hop)) };
}

/* ------------------------------------------------------------------ dallo spettro al suono --- */

/**
 * Griffin-Lim: ricostruisce un segnale da sole ampiezze cercando fasi coerenti.
 * Alterna ISTFT (overlap-add) e STFT tenendo le ampiezze date e le fasi stimate.
 * È il "decoder" del nostro codec giocattolo: EnCodec invece ha un decoder neurale addestrato.
 * @param {Float32Array} mag frames × bins (bins ≤ nFft/2+1; i bin mancanti valgono 0)
 * @param {object} o @param {number} o.frames @param {number} o.bins @param {number} o.nFft
 * @param {number} o.hop @param {number} o.length lunghezza del segnale in uscita
 * @param {number} [o.iterations=16] @param {() => number} [o.random] generatore per le fasi iniziali
 */
export function griffinLim(mag, { frames, bins, nFft, hop, length, iterations = 16, random = Math.random }) {
  const half = nFft >> 1;
  const window = hannWindow(nFft);
  const phase = new Float32Array(frames * bins);
  for (let i = 0; i < phase.length; i++) phase[i] = 2 * Math.PI * random();
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const total = (frames - 1) * hop + nFft;
  const signal = new Float64Array(total);
  const norm = new Float64Array(total);
  for (let t = 0; t < frames; t++) for (let i = 0; i < nFft; i++) norm[t * hop + i] += window[i] * window[i];

  const synthesize = () => {
    signal.fill(0);
    for (let t = 0; t < frames; t++) {
      re.fill(0);
      im.fill(0);
      for (let k = 0; k < bins; k++) {
        const a = mag[t * bins + k];
        const p = phase[t * bins + k];
        re[k] = a * Math.cos(p);
        im[k] = a * Math.sin(p);
        if (k > 0 && k < half) {
          re[nFft - k] = re[k]; // simmetria hermitiana: il segnale resta reale
          im[nFft - k] = -im[k];
        }
      }
      fft(re, im, true);
      const o = t * hop;
      for (let i = 0; i < nFft; i++) signal[o + i] += re[i] * window[i];
    }
    for (let i = 0; i < total; i++) if (norm[i] > 1e-8) signal[i] /= norm[i];
  };
  const analyze = () => {
    for (let t = 0; t < frames; t++) {
      const o = t * hop;
      for (let i = 0; i < nFft; i++) {
        re[i] = signal[o + i] * window[i];
        im[i] = 0;
      }
      fft(re, im);
      for (let k = 0; k < bins; k++) phase[t * bins + k] = Math.atan2(im[k], re[k]);
    }
  };
  for (let it = 0; it < iterations; it++) {
    synthesize();
    analyze();
  }
  synthesize();
  // il frame t è centrato sul campione t·hop: si scarta la mezza finestra iniziale
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = signal[i + half] ?? 0;
  return out;
}

/* -------------------------------------------------------------------------------- utilità --- */

/** Generatore pseudo-casuale deterministico (mulberry32): stessi risultati a ogni caricamento. */
export function seededRandom(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Porta il picco del segnale a `peak` (evita la saturazione in riproduzione). */
export function normalizePeak(x, peak = 0.8) {
  let max = 0;
  for (let i = 0; i < x.length; i++) max = Math.max(max, Math.abs(x[i]));
  if (max > 0) for (let i = 0; i < x.length; i++) x[i] *= peak / max;
  return x;
}

/** Inviluppo min/max per colonne: per disegnare una forma d'onda lunga con poche linee. */
export function envelope(x, columns, start = 0, end = x.length) {
  const out = new Float32Array(columns * 2);
  const span = (end - start) / columns;
  for (let c = 0; c < columns; c++) {
    let lo = Infinity;
    let hi = -Infinity;
    const a = Math.floor(start + c * span);
    const b = Math.max(a + 1, Math.floor(start + (c + 1) * span));
    for (let i = a; i < b && i < x.length; i++) {
      if (x[i] < lo) lo = x[i];
      if (x[i] > hi) hi = x[i];
    }
    out[2 * c] = Number.isFinite(lo) ? lo : 0;
    out[2 * c + 1] = Number.isFinite(hi) ? hi : 0;
  }
  return out;
}
