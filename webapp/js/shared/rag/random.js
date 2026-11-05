// Utilità deterministiche condivise dalla libreria RAG: generatore pseudo-casuale con seed
// e hash di stringhe. Tutto deve essere riproducibile: stessa entrata, stessa uscita, così
// le animazioni e i test vedono sempre gli stessi numeri.

/**
 * Crea un generatore pseudo-casuale con seed (algoritmo mulberry32: 32 bit di stato, veloce
 * e con una distribuzione più che sufficiente per scopi didattici; non è crittografico).
 *
 * @param {number} seed intero qualsiasi
 * @returns {() => number} funzione che restituisce un numero in [0, 1)
 */
export function createRng(seed) {
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
 * Hash FNV-1a a 32 bit di una stringa (UTF-16). Serve al "trucco dell'hashing" dell'embedder
 * e per derivare seed stabili dai nomi.
 *
 * @param {string} text
 * @param {number} [seed=0x811c9dc5] base dell'hash; seed diversi danno hash indipendenti
 * @returns {number} intero senza segno a 32 bit
 */
export function hashString(text, seed = 0x811c9dc5) {
  let hash = seed >>> 0;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  // Rimescolamento finale (avalanche): FNV da solo distribuisce male i bit alti,
  // che invece usiamo per il segno dell'hashing.
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b) >>> 0;
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}

/**
 * Numero gaussiano standard (media 0, varianza 1) con il metodo di Box-Muller.
 *
 * @param {() => number} rng generatore uniforme in [0, 1)
 * @returns {number}
 */
export function gaussian(rng) {
  // 1 - rng() è in (0, 1]: evita log(0).
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
