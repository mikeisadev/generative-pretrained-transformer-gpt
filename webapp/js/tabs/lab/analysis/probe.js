/**
 * Il prossimo carattere sul prompt dello studente, per A e per B (logica pura sul motore JS, testata).
 * Il prompt è testo non fidato: i caratteri fuori vocabolario non sono un'eccezione ma un messaggio, e
 * oltre block_size contano solo gli ultimi caratteri (come in generazione: gpt/generation.py).
 *
 *   const a = nextTokenProbe(engineA, "ROMEO:\nBut, soft! what light thr");
 *   a.top → [{ id, char, prob, logit }…];   sharedIds(a, b, 8) → token da mostrare negli stessi grafici
 */
import { shownChar } from "../inspect/meaning.js";

export const MAX_PROMPT = 2000; // il motore guarda solo gli ultimi block_size caratteri: oltre è solo peso morto

/**
 * @param {import("../../../engine/gpt.js").GPTEngine} engine
 * @param {string} text
 * @param {{top?: number}} [o]
 * @returns {{ok: true, ids: number[], used: number, cropped: boolean, probs: Float32Array, logits: Float32Array,
 *   top: {id: number, char: string, prob: number, logit: number}[], vocab: string[]}
 *   | {ok: false, reason: string, unknown?: string[]}}
 */
export function nextTokenProbe(engine, text, { top = 8 } = {}) {
  const prompt = String(text ?? "").slice(-MAX_PROMPT);
  if (!prompt.length) return { ok: false, reason: "scrivi almeno un carattere" };
  let ids;
  try {
    ids = engine.encode(prompt);
  } catch (error) {
    const unknown = Array.isArray(error?.unknown) ? error.unknown : [];
    return { ok: false, reason: unknown.length ? `caratteri che il vocabolario non ha: ${unknown.map((c) => `«${shownChar(c)}»`).join(" ")}` : String(error?.message ?? error), unknown };
  }
  const T = engine.config.block_size;
  const { probs, logits } = engine.nextTokenDistribution(ids);
  const vocab = engine.vocab;
  const order = Array.from(probs.keys()).sort((x, y) => probs[y] - probs[x] || x - y);
  return {
    ok: true,
    ids,
    used: Math.min(ids.length, T),
    cropped: ids.length > T,
    probs,
    logits,
    top: order.slice(0, top).map((id) => ({ id, char: vocab[id], prob: probs[id], logit: logits[id] })),
    vocab,
  };
}

/**
 * I token da mostrare negli stessi grafici per A e B: l'unione dei primi `k` di ciascuno, nell'ordine
 * delle probabilità di A (poi di B). Stesse etichette = barre confrontabili a colpo d'occhio.
 */
export function sharedIds(a, b, k = 8) {
  const ids = new Set();
  for (const p of [a, b]) if (p?.ok) for (const t of p.top.slice(0, k)) ids.add(t.id);
  const pa = a?.ok ? a.probs : null;
  const pb = b?.ok ? b.probs : null;
  return [...ids].sort((x, y) => (pa ? pa[y] - pa[x] : 0) || (pb ? pb[y] - pb[x] : 0) || x - y);
}

/** I dati di un grafico a barre (probabilità o logits) sugli stessi token, con il più probabile evidenziato. */
export function barData(probe, ids, key = "probs") {
  if (!probe?.ok) return { labels: [], values: [] };
  const values = ids.map((id) => probe[key][id]);
  const best = probe.top[0]?.id;
  const highlight = ids.indexOf(best);
  return { labels: ids.map((id) => shownChar(probe.vocab[id])), values, highlight: highlight >= 0 ? highlight : undefined };
}

/** Punti dello scatter "logits di A contro logits di B", un punto per carattere del vocabolario. */
export function logitPairs(a, b) {
  if (!a?.ok || !b?.ok || a.logits.length !== b.logits.length) return [];
  return Array.from(a.logits, (x, id) => ({ x, y: b.logits[id], label: shownChar(a.vocab[id]) }));
}
