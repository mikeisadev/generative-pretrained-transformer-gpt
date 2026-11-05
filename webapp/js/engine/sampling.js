/**
 * Dalla distribuzione al token: temperatura, top-k, campionamento e un generatore di numeri
 * casuali deterministico.
 *
 * PyTorch usa `torch.multinomial` con il suo generatore: in JavaScript non possiamo
 * riprodurre gli STESSI token estratti, ma le probabilità da cui si estrae sono identiche.
 * Con lo stesso seed, la generazione JS si ripete uguale (utile per le animazioni).
 */

import { softmax } from "./kernels.js";

/**
 * mulberry32: generatore pseudo-casuale a 32 bit, piccolo e veloce, con stato riproducibile.
 * @param {number} seed intero (viene ridotto a 32 bit senza segno)
 * @returns {() => number} funzione che restituisce numeri in [0, 1)
 */
export function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Estrae un indice con probabilità `probs[i]` (metodo della "ruota": somma cumulativa).
 * @param {ArrayLike<number>} probs distribuzione (somma 1)
 * @param {() => number} rng numeri in [0, 1), per esempio `mulberry32(seed)`
 * @returns {number} indice estratto
 */
export function sample(probs, rng) {
  const u = rng();
  let cumulative = 0;
  let lastPositive = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] <= 0) continue;
    cumulative += probs[i];
    lastPositive = i;
    if (u < cumulative) return i;
  }
  // Gli arrotondamenti possono lasciare la somma appena sotto 1: vince l'ultimo token possibile.
  if (lastPositive < 0) throw new Error("distribuzione vuota: nessuna probabilità positiva");
  return lastPositive;
}

/**
 * Controlla le opzioni di campionamento e restituisce i valori normalizzati.
 * @returns {{temperature: number, topK: number|null}}
 */
export function samplingOptions({ temperature = 1, topK = null } = {}, vocabSize) {
  if (!(Number.isFinite(temperature) && temperature > 0)) {
    throw new RangeError(`temperature deve essere un numero positivo, ricevuto ${temperature}`);
  }
  if (topK !== null && topK !== undefined) {
    if (!Number.isInteger(topK) || topK < 1 || topK > vocabSize) {
      throw new RangeError(`topK deve essere un intero tra 1 e ${vocabSize}, ricevuto ${topK}`);
    }
    return { temperature, topK };
  }
  return { temperature, topK: null };
}

/**
 * Probabilità del prossimo token da una riga di logits:
 * logits / temperature, poi (se richiesto) solo i topK più alti, poi softmax.
 * Come in nanoGPT, a parità di valore col k-esimo si tengono tutti i pari merito.
 *
 * @param {Float32Array} probs uscita, lunga esattamente V
 * @param {Float32Array} logits sorgente
 * @param {number} offset inizio della riga in `logits`
 * @param {number} V dimensione del vocabolario
 * @param {{temperature: number, topK: number|null}} options già validate da `samplingOptions`
 * @param {Float32Array} scratch buffer di lavoro lungo esattamente V (evita allocazioni)
 */
export function distributionFromLogits(probs, logits, offset, V, { temperature, topK }, scratch) {
  for (let i = 0; i < V; i++) probs[i] = logits[offset + i] / temperature;
  if (topK !== null && topK < V) {
    scratch.set(probs);
    scratch.sort(); // TypedArray.sort è numerico e crescente
    const threshold = scratch[V - topK];
    for (let i = 0; i < V; i++) if (probs[i] < threshold) probs[i] = -Infinity;
  }
  softmax(probs, 0, probs, 0, V);
}
