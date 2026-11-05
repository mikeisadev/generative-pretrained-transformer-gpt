// Tokenizzatore didattico che APPROSSIMA cl100k_base, il tokenizzatore BPE di OpenAI usato da
// Spring AI nel TokenTextSplitter (libreria JTokkit, EncodingType.CL100K_BASE).
//
// Come funziona il vero cl100k_base:
//   1. una regex divide il testo in "pre-token" (parole con lo spazio davanti, gruppi di al più
//      3 cifre, punteggiatura, spazi e a capo);
//   2. ogni pre-token viene poi fuso in pezzi più grandi con il Byte Pair Encoding, usando una
//      tabella di circa 100.000 fusioni imparate dai dati: le parole frequenti diventano 1 token,
//      quelle rare vengono spezzate in più token.
//
// Qui facciamo il passo 1 con la stessa regex di cl100k_base e sostituiamo il passo 2 (che
// richiederebbe la tabella delle fusioni, qualche MB) con una regola semplice: una parola fino a
// 4 lettere vale 1 token, una parola più lunga viene divisa in pezzi di circa 4 lettere.
// Il conteggio è quindi una stima: sull'italiano somiglia all'ordine di grandezza reale (circa
// 3-4 caratteri per token), ma non coincide token per token con quello di OpenAI.

/**
 * Regex di pre-tokenizzazione di cl100k_base (dal codice di tiktoken), tradotta in sintassi
 * JavaScript: `(?i:'s|…)` diventa un gruppo con il flag `i` globale, `\p{L}` e `\p{N}` richiedono `u`.
 */
const PRE_TOKEN_PATTERN =
  /'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu;

/** Parole fino a questa lunghezza (in lettere) contano come un solo token. */
export const MAX_LETTERS_SINGLE_TOKEN = 4;
/** Lunghezza tipica dei pezzi in cui si spezzano le parole più lunghe. */
export const LETTERS_PER_PIECE = 4;

/** Descrizione dichiarata dell'approssimazione, da mostrare nella UI. */
export const TOKENIZER_INFO = Object.freeze({
  name: "cl100k-approx",
  approximates: "cl100k_base (BPE di OpenAI, usato da TokenTextSplitter tramite JTokkit)",
  description:
    "Simulazione didattica: stessa regex di pre-tokenizzazione di cl100k_base, ma al posto del " +
    "Byte Pair Encoding una regola fissa (parole fino a 4 lettere = 1 token, parole più lunghe " +
    "divise in pezzi da circa 4 lettere). I conteggi sono stime, non i token reali di OpenAI.",
});

/**
 * @typedef {object} Token
 * @property {string} text  testo del token (spazi iniziali inclusi, come in cl100k)
 * @property {number} start indice del primo carattere nel testo originale
 * @property {number} end   indice successivo all'ultimo carattere
 * @property {"word"|"number"|"punct"|"space"|"newline"} kind categoria, utile per colorare
 */

/**
 * Classifica un pezzo di testo per la visualizzazione.
 *
 * @param {string} text
 * @returns {Token["kind"]}
 */
function kindOf(text) {
  if (/\p{L}/u.test(text)) return "word";
  if (/\p{N}/u.test(text)) return "number";
  if (/[\r\n]/.test(text) && text.trim() === "") return "newline";
  if (text.trim() === "") return "space";
  return "punct";
}

/**
 * Divide le lettere di una parola lunga in pezzi di lunghezza il più possibile uguale.
 * Il primo pezzo conserva l'eventuale prefisso (spazio o apostrofo), come fa cl100k.
 *
 * @param {string} piece pre-token
 * @param {number} offset posizione del pre-token nel testo
 * @param {Token[]} out
 */
function pushPieces(piece, offset, out) {
  const match = /^([^\p{L}]?)(\p{L}+)$/u.exec(piece);
  const letters = match ? match[2] : "";
  if (!match || letters.length <= MAX_LETTERS_SINGLE_TOKEN) {
    out.push({ text: piece, start: offset, end: offset + piece.length, kind: kindOf(piece) });
    return;
  }
  const count = Math.ceil(letters.length / LETTERS_PER_PIECE);
  const prefix = match[1];
  let cursor = 0;
  for (let i = 0; i < count; i++) {
    // Pezzi bilanciati: la divisione intera distribuisce il resto sui primi pezzi.
    const size = Math.floor(letters.length / count) + (i < letters.length % count ? 1 : 0);
    const text = (i === 0 ? prefix : "") + letters.slice(cursor, cursor + size);
    const start = offset + (i === 0 ? 0 : prefix.length + cursor);
    out.push({ text, start, end: start + text.length, kind: "word" });
    cursor += size;
  }
}

/**
 * Divide il testo in token. Garanzia importante (come per il vero BPE): concatenando i token
 * si riottiene esattamente il testo di partenza, quindi `decode(tokenize(t)) === t`.
 *
 * @param {string} text
 * @returns {Token[]}
 */
export function tokenize(text) {
  if (typeof text !== "string") throw new TypeError("tokenize: serve una stringa");
  const tokens = [];
  for (const match of text.matchAll(PRE_TOKEN_PATTERN)) {
    pushPieces(match[0], match.index, tokens);
  }
  return tokens;
}

/**
 * Ricompone il testo da una sequenza di token (l'inverso di `tokenize`).
 *
 * @param {Token[]} tokens
 * @param {number} [from=0]
 * @param {number} [to=tokens.length]
 * @returns {string}
 */
export function decode(tokens, from = 0, to = tokens.length) {
  let text = "";
  for (let i = from; i < to; i++) text += tokens[i].text;
  return text;
}

/**
 * Conta i token (stimati) di un testo.
 *
 * @param {string} text
 * @returns {number}
 */
export function countTokens(text) {
  return tokenize(text).length;
}
