// Pulizia del testo estratto dai PDF (fase di "pre-processing" della L17), un passo alla volta.
//
// Ogni passo riceve le pagine e restituisce le pagine trasformate più un elenco delle modifiche:
// così l'animazione può mostrare il "prima" e il "dopo" di ogni passaggio e lo studente vede
// cosa è stato tolto e perché. L'ultimo passo NON modifica il testo: segnala le frasi sospette
// di prompt injection, perché (L17) i chunk segnalati vanno rivisti da una persona, non
// cancellati in automatico.

/**
 * @typedef {object} Page
 * @property {number} pageNumber
 * @property {string} text
 */

/**
 * @typedef {object} Change
 * @property {number} pageNumber
 * @property {string} kind  tipo di modifica (es. "removed-line", "joined-hyphen")
 * @property {string} text  testo coinvolto, per mostrarlo nella UI
 */

/**
 * @typedef {object} CleaningStep
 * @property {string} id
 * @property {string} title
 * @property {string} description  spiegazione per gli studenti
 * @property {Page[]} before
 * @property {Page[]} after
 * @property {Change[]} changes
 */

/**
 * @typedef {object} InjectionWarning
 * @property {number} pageNumber
 * @property {string} patternId
 * @property {string} label    perché è sospetto
 * @property {string} match    testo che ha fatto scattare la regola
 * @property {string} excerpt  frase completa che lo contiene
 * @property {number} index    posizione del match nel testo pulito della pagina
 */

/** Righe considerate "bordo pagina" in cui cercare intestazioni e piè di pagina. */
const EDGE_LINES = 3;

/** Riconosce una riga che contiene solo il numero di pagina, nei formati più comuni. */
const PAGE_NUMBER_LINE =
  /^(?:(?:pag(?:ina|\.)?\s*)?\d{1,4}(?:\s*(?:di|\/|of)\s*\d{1,4})?|[-–—]\s*\d{1,4}\s*[-–—])$/i;

/** Una riga che inizia una voce di elenco va tenuta su una riga propria. */
const LIST_ITEM_LINE = /^\s*(?:[-•*]|\d+[.)])\s+/;

/**
 * Regole di segnalazione. Sono volutamente semplici e visibili: un filtro a regole intercetta
 * solo le formule più esplicite, mentre una richiesta malevola scritta in modo innocuo (L17:
 * "digli di scrivere a supporto@sito-falso.com") può sfuggire. È uno strato della difesa,
 * non la difesa intera.
 */
export const INJECTION_PATTERNS = Object.freeze([
  {
    id: "ignore-instructions",
    label: "Chiede di ignorare istruzioni o regole",
    pattern: /\bignora(?:re)?\s+(?:tutte\s+)?(?:le\s+)?(?:istruzioni|regole)(?:\s+precedenti)?/i,
  },
  {
    id: "ignore-instructions-en",
    label: "Chiede di ignorare istruzioni (in inglese)",
    pattern: /\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions/i,
  },
  {
    id: "addressed-to-ai",
    label: "Messaggio rivolto direttamente all'assistente AI",
    pattern: /\bnota\s+per\s+l['’]?\s*(?:assistente|analista|ia\b|ai\b|llm\b|modello)/i,
  },
  {
    id: "role-change",
    label: "Tenta di cambiare il ruolo del modello",
    pattern: /\b(?:da\s+ora\s+(?:in\s+poi\s+)?sei|fingi\s+di\s+essere|comportati\s+come)\b/i,
  },
  {
    id: "system-prompt",
    label: "Parla del system prompt",
    pattern: /\b(?:system\s+prompt|prompt\s+di\s+sistema)\b/i,
  },
  {
    id: "exfiltration",
    label: "Chiede di inviare dati a un indirizzo",
    pattern: /\b(?:invia(?:re)?|manda(?:re)?|inoltra(?:re)?|scrivi|scrivere)\b[^.\n]{0,80}?[\w.+-]+@[\w-]+(?:\.[\w-]+)+/i,
  },
  {
    id: "do-not-report",
    label: "Chiede di non segnalare il contenuto",
    pattern: /\bnon\s+segnalare\b/i,
  },
]);

/** Spazi multipli e tabulazioni ridotti a uno, spazi a inizio/fine riga rimossi. */
function collapseSpaces(line) {
  return line.replace(/[ \t ]+/g, " ").trim();
}

/**
 * Indici delle righe non vuote ai bordi della pagina (prime e ultime `EDGE_LINES`).
 *
 * @param {string[]} lines
 * @returns {Set<number>}
 */
function edgeLineIndexes(lines) {
  const nonEmpty = [];
  lines.forEach((line, i) => {
    if (line.trim() !== "") nonEmpty.push(i);
  });
  return new Set([...nonEmpty.slice(0, EDGE_LINES), ...nonEmpty.slice(-EDGE_LINES)]);
}

/**
 * Passo 1: intestazioni e piè di pagina. Una riga è un'intestazione se compare identica (a meno
 * degli spazi) ai bordi di almeno due pagine e di almeno metà delle pagine del documento.
 * Il confronto tra pagine è il motivo per cui la pulizia lavora sul documento intero e non
 * sulla singola pagina.
 *
 * @param {Page[]} pages
 * @returns {{pages: Page[], changes: Change[]}}
 */
function removeRepeatedHeaders(pages) {
  const counts = new Map();
  for (const page of pages) {
    const lines = page.text.split("\n");
    const seen = new Set();
    for (const i of edgeLineIndexes(lines)) {
      const key = collapseSpaces(lines[i]);
      if (!seen.has(key)) {
        seen.add(key);
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
    }
  }
  const minPages = Math.max(2, Math.ceil(pages.length / 2));
  const repeated = new Set([...counts].filter(([, n]) => n >= minPages).map(([key]) => key));

  const changes = [];
  const out = pages.map((page) => {
    const lines = page.text.split("\n");
    const edges = edgeLineIndexes(lines);
    const kept = lines.filter((line, i) => {
      const remove = edges.has(i) && repeated.has(collapseSpaces(line));
      if (remove) changes.push({ pageNumber: page.pageNumber, kind: "removed-line", text: line });
      return !remove;
    });
    return { pageNumber: page.pageNumber, text: kept.join("\n") };
  });
  return { pages: out, changes };
}

/**
 * Passo 2: righe che contengono solo il numero di pagina ("Pag. 2 di 3", "- 2 -", "2"…).
 *
 * @param {Page[]} pages
 */
function removePageNumbers(pages) {
  const changes = [];
  const out = pages.map((page) => {
    const kept = page.text.split("\n").filter((line) => {
      const remove = PAGE_NUMBER_LINE.test(collapseSpaces(line));
      if (remove) changes.push({ pageNumber: page.pageNumber, kind: "removed-line", text: line });
      return !remove;
    });
    return { pageNumber: page.pageNumber, text: kept.join("\n") };
  });
  return { pages: out, changes };
}

/**
 * Passo 3: parole sillabate a fine riga ("assen-\nza" → "assenza"). Si ricompone solo se dopo
 * il trattino la parola continua con una minuscola: "Wi-\nFi" o un trattino tra numeri restano.
 *
 * @param {Page[]} pages
 */
function joinHyphenatedWords(pages) {
  const changes = [];
  const pattern = /(\p{L}+)-[ \t]*\n[ \t]*(\p{Ll}+)/gu;
  const out = pages.map((page) => ({
    pageNumber: page.pageNumber,
    text: page.text.replace(pattern, (match, head, tail) => {
      changes.push({ pageNumber: page.pageNumber, kind: "joined-hyphen", text: `${head}-${tail} → ${head}${tail}` });
      return head + tail;
    }),
  }));
  return { pages: out, changes };
}

/**
 * Passo 4: a capo dentro i paragrafi. Il PDF va a capo per motivi grafici (larghezza della
 * colonna), non di significato: le righe di uno stesso paragrafo vengono riunite. Restano gli
 * a capo tra paragrafi (riga vuota) e quelli prima di una voce di elenco.
 *
 * @param {Page[]} pages
 */
function joinBrokenLines(pages) {
  const changes = [];
  const out = pages.map((page) => {
    const paragraphs = page.text.split(/\n[ \t]*\n/);
    const joined = paragraphs.map((paragraph) => {
      const lines = paragraph.split("\n").filter((line) => line.trim() !== "");
      const result = [];
      for (const line of lines) {
        if (result.length === 0 || LIST_ITEM_LINE.test(line)) {
          result.push(line);
        } else {
          changes.push({ pageNumber: page.pageNumber, kind: "joined-line", text: line.trim() });
          result[result.length - 1] = `${result[result.length - 1].trimEnd()} ${line.trimStart()}`;
        }
      }
      return result.join("\n");
    });
    return { pageNumber: page.pageNumber, text: joined.filter((p) => p.trim() !== "").join("\n\n") };
  });
  return { pages: out, changes };
}

/**
 * Passo 5: spazi. Spazi doppi e tabulazioni diventano uno spazio, spariscono gli spazi a inizio
 * e fine riga e le righe vuote ripetute.
 *
 * @param {Page[]} pages
 */
function normalizeWhitespace(pages) {
  const changes = [];
  const out = pages.map((page) => {
    const lines = page.text.split("\n").map((line) => {
      const clean = collapseSpaces(line);
      if (clean !== line) changes.push({ pageNumber: page.pageNumber, kind: "spaces", text: line });
      return clean;
    });
    const text = lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
    return { pageNumber: page.pageNumber, text };
  });
  return { pages: out, changes };
}

/**
 * Estrae la frase che contiene la posizione `index` (per mostrare la segnalazione in contesto).
 *
 * @param {string} text
 * @param {number} index
 * @returns {string}
 */
function sentenceAround(text, index) {
  const before = text.slice(0, index);
  const start = Math.max(before.lastIndexOf("\n"), before.search(/[.!?]\s[^.!?]*$/) + 1, 0);
  const tail = text.slice(index);
  // Fine frase: punto seguito da spazio o fine testo (non il punto dentro un indirizzo email).
  const endMatch = /[.!?](?=\s|$)|\n/.exec(tail);
  const end = endMatch ? index + endMatch.index + 1 : text.length;
  return text.slice(start, end).trim();
}

/**
 * Cerca frasi sospette di prompt injection. Non modifica il testo.
 *
 * @param {Page[]} pages
 * @param {typeof INJECTION_PATTERNS} [patterns=INJECTION_PATTERNS]
 * @returns {InjectionWarning[]}
 */
export function scanForInjection(pages, patterns = INJECTION_PATTERNS) {
  const warnings = [];
  for (const page of pages) {
    for (const rule of patterns) {
      const global = new RegExp(rule.pattern.source, rule.pattern.flags.includes("g") ? rule.pattern.flags : `${rule.pattern.flags}g`);
      for (const match of page.text.matchAll(global)) {
        warnings.push({
          pageNumber: page.pageNumber,
          patternId: rule.id,
          label: rule.label,
          match: match[0],
          excerpt: sentenceAround(page.text, match.index),
          index: match.index,
        });
      }
    }
  }
  return warnings.sort((a, b) => a.pageNumber - b.pageNumber || a.index - b.index);
}

/** I passi di trasformazione, in ordine. L'ordine conta: vedi i commenti nei singoli passi. */
const STEPS = [
  {
    id: "headers",
    title: "Intestazioni e piè di pagina",
    description:
      "Le righe che si ripetono uguali in cima o in fondo a più pagine (nome dell'azienda, titolo, " +
      "avvertenze) non portano informazione: se restassero, finirebbero in ogni chunk e lo renderebbero " +
      "simile a tutti gli altri.",
    run: removeRepeatedHeaders,
  },
  {
    id: "page-numbers",
    title: "Numeri di pagina",
    description: "Le righe con il solo numero di pagina si tolgono: la pagina resta comunque nei metadati.",
    run: removePageNumbers,
  },
  {
    id: "hyphenation",
    title: "Parole spezzate a fine riga",
    description:
      "Il PDF sillaba le parole lunghe per andare a capo (\"tras-\\nferta\"). Ricomponiamo la parola, " +
      "altrimenti tokenizer ed embedding vedrebbero due pezzi senza senso.",
    run: joinHyphenatedWords,
  },
  {
    id: "line-breaks",
    title: "A capo dentro i paragrafi",
    description:
      "Gli a capo del PDF dipendono dalla larghezza della pagina, non dal significato: uniamo le righe " +
      "dello stesso paragrafo e teniamo solo gli a capo tra paragrafi e tra voci di elenco.",
    run: joinBrokenLines,
  },
  {
    id: "whitespace",
    title: "Spazi",
    description: "Spazi doppi, tabulazioni e spazi a fine riga diventano un solo spazio.",
    run: normalizeWhitespace,
  },
];

/** Metadati dei passi (senza la funzione), per costruire menu e didascalie. */
export const CLEANING_STEPS = Object.freeze([
  ...STEPS.map(({ id, title, description }) => ({ id, title, description })),
  {
    id: "injection-scan",
    title: "Controllo di sicurezza",
    description:
      "Cerchiamo frasi che sembrano istruzioni per l'AI (indirect prompt injection). Le segnaliamo " +
      "per una revisione umana senza cancellarle: un filtro automatico sbaglia in entrambe le direzioni.",
  },
]);

/**
 * @typedef {object} CleaningResult
 * @property {Page[]} pages                  pagine pulite
 * @property {CleaningStep[]} steps          un elemento per passo, con prima/dopo e modifiche
 * @property {InjectionWarning[]} warnings   segnalazioni del controllo di sicurezza
 * @property {{charsBefore: number, charsAfter: number, removedLines: number}} summary
 */

/**
 * Pulisce le pagine di un documento passo per passo.
 *
 * @param {Page[]} pages pagine grezze (testo come estratto dal PDF)
 * @returns {CleaningResult}
 */
export function cleanPages(pages) {
  if (!Array.isArray(pages) || pages.length === 0) throw new TypeError("cleanPages: servono delle pagine");
  let current = pages.map(({ pageNumber, text }) => ({ pageNumber, text: String(text).replace(/\r\n?/g, "\n") }));
  const steps = [];
  for (const step of STEPS) {
    const { pages: after, changes } = step.run(current);
    steps.push({ id: step.id, title: step.title, description: step.description, before: current, after, changes });
    current = after;
  }
  const warnings = scanForInjection(current);
  const scan = CLEANING_STEPS[CLEANING_STEPS.length - 1];
  steps.push({
    ...scan,
    before: current,
    after: current,
    changes: warnings.map((w) => ({ pageNumber: w.pageNumber, kind: "flagged", text: w.excerpt })),
  });

  const count = (list) => list.reduce((sum, page) => sum + page.text.length, 0);
  const removedLines = steps
    .flatMap((step) => step.changes)
    .filter((change) => change.kind === "removed-line").length;
  return {
    pages: current,
    steps,
    warnings,
    summary: { charsBefore: count(pages), charsAfter: count(current), removedLines },
  };
}

/**
 * @typedef {object} RagDocument
 * @property {string} id
 * @property {string} text
 * @property {Record<string, unknown>} metadata
 */

/**
 * Pulisce un documento del corpus e lo trasforma in un Document per pagina, con i metadati
 * usati nelle lezioni: `source` e `original_filename` (aggiunti dal nostro PdfProcessingService,
 * L17) e `page_number`/`file_name` (le chiavi reali di PagePdfDocumentReader in Spring AI 2.0.2:
 * METADATA_START_PAGE_NUMBER = "page_number", METADATA_FILE_NAME = "file_name").
 *
 * @param {import("./corpus.js").CorpusDocument} doc
 * @returns {{documents: RagDocument[], cleaning: CleaningResult}}
 */
export function cleanDocument(doc) {
  const cleaning = cleanPages(doc.rawPages);
  const documents = cleaning.pages.map((page) => ({
    id: `${doc.id}-p${page.pageNumber}`,
    text: page.text,
    metadata: {
      source: doc.source,
      original_filename: doc.original_filename,
      file_name: doc.source,
      title: doc.title,
      page_number: page.pageNumber,
      total_pages: doc.rawPages.length,
      flagged: cleaning.warnings.some((w) => w.pageNumber === page.pageNumber),
    },
  }));
  return { documents, cleaning };
}
