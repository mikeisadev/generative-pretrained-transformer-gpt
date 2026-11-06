/**
 * Ricerca globale: logica pura (nessun DOM), testata in tests/search.test.mjs.
 *
 * - normalizzazione tollerante: minuscole, senza accenti ("perché" = "perche"), punteggiatura → spazi;
 * - punteggio: ogni parola della domanda deve trovarsi da qualche parte (titolo, parole chiave, testo);
 *   pesano di più il titolo e l'inizio delle parole; le parole parziali valgono ("embed" → "embedding");
 *   un errore di battitura nelle parole lunghe è tollerato ("sofmax" → "softmax");
 * - link diretti: "#<tab>/go=<kind>:<id>" (vedi parseHash / formatHash).
 *
 * Una voce dell'indice (contratto di js/tabs/<id>/search.js) è
 *   { id, kind: "step" | "action" | "view", title, text?, keywords?: string[] }
 * La palette aggiunge `tab` (id del tab) e le voci speciali "tab" e "term" (glossario).
 */

/** Tipi di risultato con l'etichetta mostrata all'utente. */
export const KINDS = {
  step: "Passo",
  action: "Azione",
  view: "Vista",
  term: "Termine",
  tab: "Tab",
};

/**
 * Parole troppo comuni per distinguere un risultato: vengono ignorate se nella domanda c'è
 * almeno un'altra parola ("come funziona la softmax" cerca "funziona softmax").
 */
const STOPWORDS = new Set(
  ("a ad al alla alle allo agli ai che chi ci come con cosa cos da dal dalla dalle dai degli dei del della delle dello " +
    "di e ed gli i il in l la le lo ma mi ne nel nella nelle nei nello o per piu quale quali quando se si sono su sul " +
    "sulla tra fra un una uno d c the of and to is").split(" "),
);

// --- normalizzazione -----------------------------------------------------------------------

/** Una lettera "piegata": minuscola e senza segni diacritici (può diventare più caratteri, es. "ß"). */
function foldChar(ch) {
  return ch.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/** Testo confrontabile: minuscole, senza accenti, solo lettere e cifre separate da uno spazio. */
export function normalize(text) {
  return String(text ?? "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/** Parole della domanda (senza le parole vuote, a meno che non ci siano solo quelle). */
export function queryTokens(query) {
  const words = normalize(query).split(" ").filter(Boolean);
  const useful = words.filter((w) => !STOPWORDS.has(w));
  return [...new Set(useful.length ? useful : words)];
}

const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", egrave: "è", eacute: "é", agrave: "à", ugrave: "ù", ograve: "ò", igrave: "ì", rarr: "→", larr: "←", middot: "·", hellip: "…" };

/**
 * Testo semplice da HTML scritto da noi (content.js): toglie i tag, decodifica le entità più
 * comuni, i segnaposto {{chiave}} e i delimitatori LaTeX. Non è un sanitizer: il risultato si
 * mostra sempre come testo (textContent), mai come HTML.
 */
export function plainText(html) {
  return String(html ?? "")
    .replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi, " ")
    .replace(/<[^>]*>/g, " ")
    .replace(/&(#\d+|#x[0-9a-f]+|\w+);/gi, (m, code) => {
      if (code[0] === "#") {
        const n = code[1] === "x" || code[1] === "X" ? parseInt(code.slice(2), 16) : parseInt(code.slice(1), 10);
        return Number.isFinite(n) ? String.fromCodePoint(n) : " ";
      }
      return ENTITIES[code.toLowerCase()] ?? m;
    })
    .replace(/\{\{\s*(\w+)\s*\}\}/g, "$1")
    .replace(/\$+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// --- indice ----------------------------------------------------------------------------------

/**
 * Prepara una voce per la ricerca (si fa una volta sola, quando l'indice viene caricato).
 * @returns {object} la voce originale con i campi normalizzati in `_s`
 */
export function prepareEntry(entry) {
  const title = plainText(entry.title);
  const text = plainText(entry.text ?? "");
  const keywords = (entry.keywords ?? []).map(plainText).join(" ");
  const titleN = normalize(title);
  const keywordsN = normalize(keywords);
  const textN = normalize(text);
  return {
    ...entry,
    title,
    text,
    _s: {
      title: titleN,
      titleWords: titleN.split(" ").filter(Boolean),
      titleCompact: titleN.replace(/ /g, ""),
      keywords: keywordsN,
      keywordWords: keywordsN.split(" ").filter(Boolean),
      text: textN,
      textWords: new Set(textN.split(" ").filter(Boolean)),
    },
  };
}

/** Distanza di Damerau-Levenshtein limitata (si ferma appena supera `max`). */
export function editDistance(a, b, max = 2) {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const rows = [];
  for (let i = 0; i <= a.length; i++) {
    rows.push(new Array(b.length + 1).fill(0));
    rows[i][0] = i;
  }
  for (let j = 0; j <= b.length; j++) rows[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    let rowMin = Infinity;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let d = Math.min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, rows[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d = Math.min(d, rows[i - 2][j - 2] + 1);
      rows[i][j] = d;
      rowMin = Math.min(rowMin, d);
    }
    if (rowMin > max) return max + 1;
  }
  return rows[a.length][b.length];
}

/** Errori di battitura tollerati per una parola della domanda: nessuno sotto i 5 caratteri. */
const allowedTypos = (token) => (token.length >= 8 ? 2 : token.length >= 5 ? 1 : 0);

function bestWordScore(words, token, exact, prefix, typo) {
  let best = 0;
  const typos = allowedTypos(token);
  for (const w of words) {
    if (w === token) return exact;
    if (w.startsWith(token)) best = Math.max(best, prefix);
    else if (typos && best < typo && fuzzyPrefix(token, w, typos)) best = typo;
  }
  return best;
}

/** true se `token` coincide con l'inizio di `word` a meno di `typos` errori. */
function fuzzyPrefix(token, word, typos) {
  for (let len = token.length - typos; len <= token.length + typos; len++) {
    if (len < 4 || len > word.length) continue;
    if (editDistance(token, word.slice(0, len), typos) <= typos) return true;
  }
  return false;
}

/** Punteggio di una parola della domanda dentro una voce (0 = non trovata). */
function tokenScore(s, token) {
  const short = token.length < 3; // "k", "kv": solo come parola intera o inizio di parola
  let score = bestWordScore(s.titleWords, token, 10, 8, 3);
  if (score < 5 && !short && (s.title.includes(token) || (token.length >= 4 && s.titleCompact.includes(token)))) score = 5;
  if (score < 7) score = Math.max(score, bestWordScore(s.keywordWords, token, 7, 6, 2.5));
  if (score < 4 && !short && s.keywords.includes(token)) score = 4;
  if (score < 3) {
    if (s.textWords.has(token)) score = 3;
    else if (!short) {
      for (const w of s.textWords) {
        if (w.startsWith(token)) {
          score = 2.5;
          break;
        }
      }
      if (score < 1.5 && token.length >= 4 && s.text.includes(token)) score = 1.5;
    }
  }
  return score;
}

/** Peso del tipo: a parità di testo, il termine del glossario e il tab vengono prima. */
const KIND_BONUS = { tab: 2, term: 1.5, view: 1, step: 0.5, action: 0.5 };

/**
 * Punteggio di una voce preparata (0 = esclusa): tutte le parole devono trovarsi.
 * @param {object} prepared output di prepareEntry
 * @param {string[]} tokens output di queryTokens
 */
export function scoreEntry(prepared, tokens, phrase = tokens.join(" ")) {
  if (!tokens.length) return 0;
  const s = prepared._s;
  let total = 0;
  for (const token of tokens) {
    const score = tokenScore(s, token);
    if (!score) return 0;
    total += score;
  }
  if (s.title === phrase) total += 12;
  else if (s.title.startsWith(phrase)) total += 6;
  else if (phrase.length >= 3 && s.title.includes(phrase)) total += 3;
  if (s.keywordWords.includes(phrase)) total += 4;
  return total + (KIND_BONUS[prepared.kind] ?? 0);
}

/**
 * Cerca nell'indice preparato.
 * @param {object[]} index voci preparate (prepareEntry)
 * @param {string} query testo scritto dall'utente
 * @param {object} [options]
 * @param {(entry) => boolean} [options.filter] es. solo un tipo
 * @param {(entry) => number} [options.boost] punti in più (es. tab aperto)
 * @param {number} [options.limit]
 * @returns {{entry: object, score: number}[]} in ordine di punteggio decrescente
 */
export function search(index, query, { filter, boost, limit = 60 } = {}) {
  const tokens = queryTokens(query);
  if (!tokens.length) return [];
  const phrase = tokens.join(" ");
  const results = [];
  for (const entry of index) {
    if (filter && !filter(entry)) continue;
    const score = scoreEntry(entry, tokens, phrase);
    if (score > 0) results.push({ entry, score: score + (boost?.(entry) ?? 0) });
  }
  results.sort((a, b) => b.score - a.score || a.entry.title.localeCompare(b.entry.title, "it"));
  return results.slice(0, limit);
}

/**
 * Raggruppa i risultati per `group` (di solito il tab) mantenendo l'ordine di merito:
 * il gruppo con il risultato migliore viene per primo.
 * @returns {{group: string, results: object[]}[]}
 */
export function groupResults(results, groupOf = (r) => r.entry.group ?? r.entry.tab) {
  const groups = new Map();
  for (const r of results) {
    const key = groupOf(r);
    if (!groups.has(key)) groups.set(key, { group: key, results: [] });
    groups.get(key).results.push(r);
  }
  return [...groups.values()];
}

// --- evidenziazione ed estratti --------------------------------------------------------------

/**
 * Intervalli [inizio, fine) del testo ORIGINALE che corrispondono alle parole cercate
 * (inizio di parola, senza badare ad accenti e maiuscole). Servono per i <mark>.
 */
export function highlightRanges(text, tokens) {
  const source = String(text ?? "");
  if (!tokens.length || !source) return [];
  // Testo piegato carattere per carattere, con la posizione originale di ogni carattere.
  let folded = "";
  const origin = [];
  for (let i = 0; i < source.length; i++) {
    const f = foldChar(source[i]);
    for (const c of f) {
      folded += /[\p{L}\p{N}]/u.test(c) ? c : " ";
      origin.push(i);
    }
  }
  origin.push(source.length);
  const ranges = [];
  for (const token of tokens) {
    let from = 0;
    for (;;) {
      const at = folded.indexOf(token, from);
      if (at < 0) break;
      const wordStart = at === 0 || folded[at - 1] === " ";
      if (wordStart || token.length >= 4) ranges.push([origin[at], origin[at + token.length - 1] + 1]);
      from = at + token.length;
    }
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([...r]);
  }
  return merged;
}

/** Estratto di `text` di circa `length` caratteri centrato sulla prima parola trovata. */
export function excerpt(text, tokens, length = 120) {
  const source = String(text ?? "");
  if (source.length <= length) return source;
  const first = highlightRanges(source, tokens)[0];
  if (!first) return source.slice(0, length).replace(/\s+\S*$/, "") + "…";
  let start = Math.max(0, first[0] - Math.round(length * 0.3));
  if (start > 0) {
    const space = source.indexOf(" ", start);
    start = space >= 0 && space < first[0] ? space + 1 : start;
  }
  const end = Math.min(source.length, start + length);
  let piece = source.slice(start, end);
  if (end < source.length) piece = piece.replace(/\s+\S*$/, "");
  return (start > 0 ? "…" : "") + piece + (end < source.length ? "…" : "");
}

// --- link diretti ----------------------------------------------------------------------------

/**
 * Legge l'indirizzo: "#transformer/go=step:head.qkv" → { tab: "transformer", go: {kind, id}, rest }.
 * `rest` è ciò che segue "<tab>/" (es. "q=…" del tab RAG), `go` è null se manca o non è valido.
 */
export function parseHash(hash) {
  const raw = String(hash ?? "").replace(/^#/, "");
  const slash = raw.indexOf("/");
  const tab = safeDecode(slash < 0 ? raw : raw.slice(0, slash));
  const rest = slash < 0 ? "" : raw.slice(slash + 1);
  const match = /^go=([a-z]+):(.+)$/.exec(rest);
  const go = match ? { kind: match[1], id: safeDecode(match[2]) } : null;
  return { tab, rest, go: go && go.id ? go : null };
}

/** Indirizzo di un link diretto: formatHash("rag", {kind: "step", id: "index.chunk"}) → "#rag/go=step:index.chunk". */
export function formatHash(tab, target) {
  if (!target?.kind || target.id === undefined || target.id === null || target.kind === "tab") return `#${tab}`;
  return `#${tab}/go=${target.kind}:${encodeURIComponent(String(target.id))}`;
}

function safeDecode(text) {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}
