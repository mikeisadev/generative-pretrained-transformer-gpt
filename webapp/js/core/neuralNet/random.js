/**
 * Numeri pseudo-casuali riproducibili per la rete neurale didattica (logica pura, testata con node).
 *
 * Le attivazioni simulate e i collegamenti campionati devono essere gli stessi a ogni apertura
 * (screenshot confrontabili, uno studente che torna trova la stessa rete): niente Math.random,
 * solo generatori con seme. Ogni sezione e ogni collegamento ha un flusso suo (hashSeed con il suo
 * id): aggiungere una sezione alla specifica non cambia i numeri delle altre.
 */

/** mulberry32: 32 bit di stato, veloce, buona distribuzione per usi grafici. Restituisce () => [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seme a 32 bit da più parti (numeri o stringhe): FNV-1a sui caratteri, con un separatore tra le parti. */
export function hashSeed(...parts) {
  let h = 0x811c9dc5;
  for (const part of parts) {
    const text = String(part);
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    h ^= 0x1f; // separatore: ("ab", "c") ≠ ("a", "bc")
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Generatore con seme derivato dalle parti: rng(seed, "link", 3). */
export const rngFor = (...parts) => mulberry32(hashSeed(...parts));

/** Normale standard (Box-Muller) dal generatore uniforme `rand`. */
export function gaussian(rand) {
  const u = Math.max(1e-12, rand());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/**
 * Salto geometrico: quante prove fallite prima del prossimo successo con probabilità p.
 * Serve a campionare k coppie su milioni possibili in O(k) invece che in O(milioni).
 */
export function geometricSkip(rand, p) {
  if (p >= 1) return 0;
  if (p <= 0) return Infinity;
  return Math.floor(Math.log(1 - rand()) / Math.log(1 - p));
}
