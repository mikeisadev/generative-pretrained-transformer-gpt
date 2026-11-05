/**
 * Esegue davvero il prompt con il motore GPT in JavaScript e raccoglie, token per token, i dati
 * che l'animazione e i grafici mostrano. Nessun DOM: si usa anche nei test in Node.
 *
 * Tutto viene calcolato PRIMA che l'animazione parta (in build): durante i passi non si fa
 * nessun calcolo pesante, così l'animazione resta fluida.
 */
import { mulberry32 } from "../../../engine/gpt.js";
import { entropy, rowNorms } from "./math.js";

/** Valori ammessi dal pannello: gli stessi limiti sono mostrati nel form. */
export const PROMPT_LIMITS = Object.freeze({
  temperature: { min: 0.1, max: 2, step: 0.05, value: 0.8 },
  tokens: { min: 1, max: 80, value: 24 },
  seed: { min: 0, max: 2 ** 31 - 1, value: 1337 },
});

/** Prompt di default: un dialogo nello stile del dataset, che il modello deve continuare. */
export const DEFAULT_PROMPT = "ROMEO:\nBut, soft! what light through yonder window breaks?\n\nJULIET:\n";

/**
 * Controlla un prompt senza lanciare eccezioni.
 * @returns {{ok: true, ids: number[]} | {ok: false, message: string, unknown?: string[]}}
 */
export function checkPrompt(engine, text) {
  if (!text) return { ok: false, message: "Scrivi almeno un carattere: il modello ha bisogno di un punto di partenza." };
  try {
    return { ok: true, ids: engine.encode(text) };
  } catch (error) {
    if (!error.unknown) throw error;
    const shown = error.unknown.map((c) => (c === "\t" ? "tab" : c === "\r" ? "a capo Windows (\\r)" : `«${c}»`)).join(", ");
    return {
      ok: false,
      unknown: error.unknown,
      message: `Questi caratteri non esistono nel vocabolario del modello (i 65 caratteri di Tiny Shakespeare): ${shown}. Toglili o sostituiscili (per esempio, niente lettere accentate né cifre diverse da 3).`,
    };
  }
}

/**
 * @param {import("../../../engine/gpt.js").GPTEngine} engine
 * @param {string} text prompt già controllato
 * @param {{temperature: number, tokens: number, seed: number}} options
 */
export function runPrompt(engine, text, { temperature, tokens, seed }) {
  const { block_size: blockSize, n_layer: L, n_head: H, n_embd: C } = engine.config;
  const promptIds = engine.encode(text);
  const base = mulberry32(seed);
  let lastU = 0;
  // Stesso generatore di engine.generate, ma ci annotiamo il numero casuale estratto:
  // è la "pallina" che cade sulla ruota delle probabilità.
  const rng = () => (lastU = base());

  const sequence = [...promptIds];
  const steps = [];
  let first = null;
  for (const out of engine.generate(promptIds, tokens, { temperature, rng, trace: true })) {
    const contextIds = sequence.slice(out.contextStart);
    const tr = out.trace;
    const T = tr.T;
    const lens = engine.logitLens(contextIds, { topK: 10 });
    const lastRow = (probs) => probs.subarray((T - 1) * tr.V, T * tr.V);
    const stages = [tr.x0, ...tr.blocks.map((b) => b.resid2)];
    const norms = stages.map((x) => rowNorms(x, T, C));
    const step = {
      index: steps.length,
      contextStart: out.contextStart,
      contextIds,
      T,
      id: out.id,
      char: out.token,
      u: lastU,
      probs: out.probs,
      logits: out.logits,
      entropy: entropy(out.probs),
      chosenProb: out.probs[out.id],
      attention: tr.blocks.map((b) => b.heads.map((h) => h.att)), // [L][H] (T × T)
      ffnPre: tr.blocks.map((b) => b.ffnPre), // [L] (T × 4C)
      lens: lens.layers.map((layer) => ({ label: layer.label, probs: lastRow(layer.probs) })),
      residualNorm: norms.map((n) => ({ last: n[T - 1], mean: n.reduce((a, v) => a + v, 0) / T })),
    };
    if (!first) first = tr; // il forward completo del primo token: lo mostriamo passo per passo
    steps.push(step);
    sequence.push(out.id);
  }
  return {
    text,
    promptIds,
    chars: [...text],
    vocab: engine.vocab,
    sequence,
    generated: engine.decode(sequence.slice(promptIds.length)),
    steps,
    trace: first,
    config: { blockSize, L, H, C, V: engine.config.vocab_size, headSize: engine.config.head_size },
    options: { temperature, tokens, seed },
  };
}
