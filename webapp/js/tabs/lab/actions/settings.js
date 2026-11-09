/**
 * Le scelte dello studente per le azioni (logica pura: niente DOM, niente three.js; testata):
 * il prompt, il token da seguire, il blocco, la head e la nn.Linear di «x · Wᵀ riga per colonna».
 *
 * Il prompt è testo non fidato: si controlla carattere per carattere contro il vocabolario del run (un
 * carattere = un token, come CharTokenizer in gpt/data.py) e non si mostra mai come HTML.
 *
 *   const s = new ActionSettings();
 *   s.on((change) => …);                 // { key, value }
 *   s.set({ prompt: "ROMEO:\n" });
 *   const p = checkPrompt(s.prompt, vocab, 32);   // { ok, ids, chars, message, … }
 */
import { NETWORK_PROMPT } from "../network.js";

/**
 * Il prompt d'esempio: lo stesso della rappresentazione "Rete neurale" del tab (network.js), l'inizio
 * del prompt del benchmark ("ROMEO:\n") e una battuta di Romeo che in Tiny Shakespeare c'è davvero.
 */
export const DEFAULT_PROMPT = NETWORK_PROMPT;

/** Le nn.Linear che «x · Wᵀ riga per colonna» sa mostrare, con l'ingresso che ricevono (dalla traccia del motore). */
export const LINEAR_CHOICES = Object.freeze([
  { id: "key", label: "key della head (Wk)", needsHead: true },
  { id: "query", label: "query della head (Wq)", needsHead: true },
  { id: "value", label: "value della head (Wv)", needsHead: true },
  { id: "proj", label: "proj dell'attention (con bias)", needsHead: false },
  { id: "ffn1", label: "feed-forward, 1° strato (con bias)", needsHead: false },
  { id: "ffn2", label: "feed-forward, 2° strato (con bias)", needsHead: false },
  { id: "lm_head", label: "lm_head (con bias)", needsHead: false },
]);

/** Come si scrive un carattere del prompt in un'etichetta (spazio e a capo resterebbero invisibili). */
export const visibleChar = (c) => (c === " " ? "␣" : c === "\n" ? "↵" : c === "\t" ? "⇥" : String(c ?? ""));

/**
 * Controlla il prompt per un run: caratteri nel vocabolario, lunghezza (il modello vede al più
 * `blockSize` token: gli ultimi).
 * @param {string} text
 * @param {string[]|null} vocab token id → carattere
 * @param {number} blockSize
 * @returns {{ok: boolean, ids: number[], chars: string[], unknown: string[], truncated: number, message: string}}
 */
export function checkPrompt(text, vocab, blockSize) {
  const clean = String(text ?? "").replace(/\r\n?/g, "\n");
  const chars = [...clean];
  if (!Array.isArray(vocab) || !vocab.length) return { ok: false, ids: [], chars: [], unknown: [], truncated: 0, message: "Il run non ha un vocabolario: non si può scrivere un prompt." };
  if (!chars.length) return { ok: false, ids: [], chars: [], unknown: [], truncated: 0, message: "Scrivi almeno un carattere." };
  const stoi = new Map(vocab.map((c, i) => [c, i]));
  const unknown = [...new Set(chars.filter((c) => !stoi.has(c)))];
  if (unknown.length) {
    const shown = unknown.slice(0, 8).map((c) => `«${visibleChar(c)}»`).join(" ");
    return { ok: false, ids: [], chars: [], unknown, truncated: 0, message: `Caratteri che il modello non conosce: ${shown}${unknown.length > 8 ? " …" : ""}. Il vocabolario ha solo i ${vocab.length} caratteri di Tiny Shakespeare.` };
  }
  const truncated = Math.max(0, chars.length - blockSize);
  const kept = chars.slice(truncated);
  const ids = kept.map((c) => stoi.get(c));
  const n = kept.length;
  const message = truncated
    ? `${chars.length} caratteri: il modello ne vede solo gli ultimi ${blockSize} (block_size), le animazioni usano quelli.`
    : `${n} ${n === 1 ? "carattere" : "caratteri"} = ${n} token: ${n === 1 ? "sta" : "stanno tutti"} nel contesto (${blockSize}).`;
  return { ok: true, ids, chars: kept, unknown: [], truncated, message };
}

/** Il token da seguire: `token` null = l'ultimo; fuori misura si ferma ai bordi. */
export function resolveToken(token, length) {
  if (!(length > 0)) return 0;
  if (token === null || token === undefined || !Number.isFinite(token)) return length - 1;
  return Math.min(length - 1, Math.max(0, Math.round(token)));
}

/**
 * L'esempio di training da un prompt: contesto = i token fino a `token` compreso, carattere giusto =
 * quello che nel prompt viene dopo. Se il token scelto è l'ultimo si usa il penultimo (dopo l'ultimo
 * il prompt non dice che cosa viene). null se il prompt ha un solo carattere.
 * @returns {{context: number[], target: number, position: number}|null}
 */
export function trainingExample(ids, token) {
  if (!ids || ids.length < 2) return null;
  const t = Math.min(resolveToken(token, ids.length), ids.length - 2);
  return { context: ids.slice(0, t + 1), target: ids[t + 1], position: t };
}

/** Lo stato delle scelte, con un evento a ogni cambiamento. */
export class ActionSettings {
  constructor({ prompt = DEFAULT_PROMPT, token = null, block = 0, head = 0, linear = "key" } = {}) {
    this.prompt = prompt;
    this.token = token; // null = l'ultimo token del prompt
    this.block = block;
    this.head = head;
    this.linear = linear;
    this._listeners = new Set();
  }

  /** Cambia una o più scelte; avvisa chi ascolta solo se qualcosa è cambiato davvero. */
  set(values) {
    const changed = [];
    for (const [key, value] of Object.entries(values)) {
      if (!Object.hasOwn(this, key) || key.startsWith("_") || this[key] === value) continue;
      this[key] = value;
      changed.push({ key, value });
    }
    for (const change of changed) for (const fn of [...this._listeners]) fn(change, this);
    return changed.length > 0;
  }

  /** fn({key, value}, settings) a ogni cambiamento; restituisce la funzione che smette di ascoltare. */
  on(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  /** Blocco e head dentro i limiti di una config (un run con meno blocchi o head). */
  clamp(config) {
    const L = Math.max(1, config?.n_layer ?? 1);
    const H = Math.max(1, config?.n_head ?? 1);
    return { block: Math.min(L - 1, Math.max(0, this.block | 0)), head: Math.min(H - 1, Math.max(0, this.head | 0)) };
  }
}
