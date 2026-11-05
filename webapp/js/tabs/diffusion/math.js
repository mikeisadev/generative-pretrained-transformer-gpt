/**
 * Matematica della diffusione, senza DOM e senza Three.js (si può provare anche in Node).
 *
 * Le formule sono quelle dei paper e del codice di riferimento:
 *   - schedule lineare di β (DDPM, Ho et al. 2020: β da 1e-4 a 0,02, T = 1000)
 *   - schedule "scaled_linear" di Stable Diffusion (β da 0,00085 a 0,012, interpolando le radici)
 *   - processo diretto  x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε
 *   - passo DDIM deterministico (η = 0) e classifier-free guidance.
 * Tutte le funzioni che lavorano su tensori scrivono in un buffer `out` passato da fuori:
 * così le animazioni non allocano memoria a ogni frame.
 */

/** Schedule di rumore: β_t, α_t = 1 − β_t, ᾱ_t = ∏ α_s. Indici 0…T−1 (t = 0 è il primo passo). */
export function makeSchedule({ kind = "linear", T = 1000, start = 1e-4, end = 0.02 } = {}) {
  const betas = new Float64Array(T);
  for (let i = 0; i < T; i++) {
    const f = T === 1 ? 0 : i / (T - 1);
    betas[i] = kind === "scaled_linear" ? (Math.sqrt(start) + f * (Math.sqrt(end) - Math.sqrt(start))) ** 2 : start + f * (end - start);
  }
  const alphaBar = new Float64Array(T);
  let prod = 1;
  for (let i = 0; i < T; i++) {
    prod *= 1 - betas[i];
    alphaBar[i] = prod;
  }
  return { kind, T, start, end, betas, alphaBar };
}

/** Le due schedule usate nel tab, con i valori dei file di configurazione ufficiali. */
export const SCHEDULES = {
  ddpm: makeSchedule({ kind: "linear", T: 1000, start: 1e-4, end: 0.02 }),
  sd: makeSchedule({ kind: "scaled_linear", T: 1000, start: 0.00085, end: 0.012 }),
};

/** Rapporto segnale/rumore: SNR(t) = ᾱ_t / (1 − ᾱ_t). */
export const snr = (alphaBar) => alphaBar / (1 - alphaBar);

/**
 * Passi temporali di un campionatore con `steps` passi su `T` (spaziatura "leading" di diffusers
 * con steps_offset = 1, come nella configurazione dello scheduler di Stable Diffusion).
 * Esempio: T = 1000, steps = 50 → 981, 961, …, 21, 1.
 */
export function samplingTimesteps(T, steps, offset = 1) {
  const ratio = Math.floor(T / steps);
  const out = [];
  for (let i = steps - 1; i >= 0; i--) out.push(Math.min(T - 1, i * ratio + offset));
  return out;
}

/** Generatore pseudo-casuale con seme (mulberry32): stesso seme → stesso rumore. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Riempie `out` di campioni gaussiani N(0, 1) (Box–Muller). */
export function fillGaussian(out, rng) {
  for (let i = 0; i < out.length; i += 2) {
    const u = Math.max(rng(), 1e-12);
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < out.length) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** Processo diretto (DDPM, eq. 4): out = √ᾱ · x0 + √(1 − ᾱ) · ε. */
export function qSample(out, x0, eps, alphaBar) {
  const a = Math.sqrt(alphaBar);
  const b = Math.sqrt(1 - alphaBar);
  for (let i = 0; i < out.length; i++) out[i] = a * x0[i] + b * eps[i];
  return out;
}

/** Stima dell'immagine pulita a partire dal rumore previsto: x̂0 = (x_t − √(1 − ᾱ) · ε̂) / √ᾱ. */
export function predictX0(out, xt, eps, alphaBar) {
  const a = Math.sqrt(alphaBar);
  const b = Math.sqrt(1 - alphaBar);
  for (let i = 0; i < out.length; i++) out[i] = (xt[i] - b * eps[i]) / a;
  return out;
}

/**
 * Passo DDIM deterministico (η = 0, Song et al. 2020, eq. 12):
 *   x_{t−1} = √ᾱ_prev · x̂0 + √(1 − ᾱ_prev) · ε̂
 * `x0hat` può essere un buffer di lavoro: viene riempito con la stima di x0.
 */
export function ddimStep(out, xt, eps, alphaBar, alphaBarPrev, x0hat) {
  predictX0(x0hat, xt, eps, alphaBar);
  const a = Math.sqrt(alphaBarPrev);
  const b = Math.sqrt(1 - alphaBarPrev);
  for (let i = 0; i < out.length; i++) out[i] = a * x0hat[i] + b * eps[i];
  return out;
}

/** Classifier-free guidance (convenzione di Stable Diffusion): ε̂ = ε_u + s · (ε_c − ε_u). */
export function cfgCombine(out, epsUncond, epsCond, scale) {
  for (let i = 0; i < out.length; i++) out[i] = epsUncond[i] + scale * (epsCond[i] - epsUncond[i]);
  return out;
}

/**
 * Rumore "previsto" da un denoiser SIMULATO che conosce già la risposta x0:
 *   ε = (x_t − √ᾱ · x0) / √(1 − ᾱ)
 * È la previsione perfetta: una rete vera la approssima dopo l'addestramento.
 */
export function oracleEps(out, xt, x0, alphaBar) {
  const a = Math.sqrt(alphaBar);
  const b = Math.sqrt(Math.max(1 - alphaBar, 1e-12));
  for (let i = 0; i < out.length; i++) out[i] = (xt[i] - a * x0[i]) / b;
  return out;
}

/** Media e deviazione standard di un array. */
export function meanStd(values) {
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i];
  const mean = s / values.length;
  let v = 0;
  for (let i = 0; i < values.length; i++) v += (values[i] - mean) ** 2;
  return { mean, std: Math.sqrt(v / values.length) };
}

/** Differenza assoluta media tra due array della stessa lunghezza. */
export function meanAbsDiff(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
}

/** Embedding sinusoidale del passo t (come in DDPM/DiT: metà coseni, metà seni). */
export function timestepEmbedding(out, t, maxPeriod = 10000) {
  const half = Math.floor(out.length / 2);
  for (let i = 0; i < half; i++) {
    const freq = Math.exp((-Math.log(maxPeriod) * i) / half);
    out[i] = Math.cos(t * freq);
    out[half + i] = Math.sin(t * freq);
  }
  return out;
}
