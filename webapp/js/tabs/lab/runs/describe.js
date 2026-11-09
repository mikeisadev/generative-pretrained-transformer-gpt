/**
 * Come si racconta un run allo studente (logica pura, testata): config con le differenze dal GPT del
 * corso, che cosa cambia nell'architettura, perché il motore JS lo esegue o no, date e dimensioni.
 * Tutto testo semplice: chi lo mostra usa textContent (nomi e note vengono dai file dello studente).
 */

export const CONFIG_KEYS = Object.freeze(["vocab_size", "block_size", "n_embd", "n_head", "n_layer", "dropout"]);

/** Che cosa vuol dire ogni voce della config (GPTConfig di gpt/config.py). */
export const CONFIG_LABELS = Object.freeze({
  vocab_size: "caratteri del vocabolario (V)",
  block_size: "contesto massimo, in token (T)",
  n_embd: "numeri per token (C)",
  n_head: "head di attention per blocco",
  n_layer: "blocchi transformer",
  dropout: "dropout durante il training",
});

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const DEC = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 2 });
export const fmtInt = (v) => INT.format(v);

/** Le voci della config: valore, valore del corso e se è cambiato (anche le chiavi nuove di run.json). */
export function configDiff(config, course) {
  const keys = [...CONFIG_KEYS, ...Object.keys(config ?? {}).filter((k) => !CONFIG_KEYS.includes(k) && k !== "__proto__")];
  return keys
    .filter((k) => config && k in config)
    .map((key) => {
      const value = config[key];
      const reference = course?.[key];
      return { key, label: CONFIG_LABELS[key] ?? "voce nuova della config", value, course: reference ?? null, changed: reference !== undefined && reference !== value, extra: !(key in (course ?? {})) };
    });
}

/** Un valore della config come testo (numeri all'italiana, il resto com'è, accorciato). */
export function configValue(v) {
  if (typeof v === "number") return Number.isInteger(v) ? fmtInt(v) : DEC.format(v);
  if (typeof v === "boolean") return v ? "sì" : "no";
  if (v === null || v === undefined) return "—";
  return String(v).slice(0, 80);
}

const few = (list, k = 4) => `${list.slice(0, k).join(", ")}${list.length > k ? ` e altri ${fmtInt(list.length - k)}` : ""}`;
const shapeText = (s) => `(${s.join(", ")})`;

/**
 * Che cosa cambia rispetto al GPT del corso, guardando nomi e forme dei tensori (le uniche cose certe:
 * il calcolo, es. GELU al posto di ReLU, nei pesi non si vede; lo dice run.json).
 * @param {Map<string, {shape: number[]}>} tensors del run
 * @param {Map<string, {shape: number[]}>} courseTensors del GPT del corso
 * @param {{engineDeclared?: boolean|null}} [o] `engine_compatible` di run.json
 * @returns {{same: boolean, notes: string[], onlyRun: string[], onlyCourse: string[], reshaped: {name, shape, course}[], blocks: number, courseBlocks: number}}
 */
export function architectureDiff(tensors, courseTensors, { engineDeclared = null } = {}) {
  const onlyRun = [];
  const reshaped = [];
  for (const [name, t] of tensors) {
    const c = courseTensors.get(name);
    if (!c) onlyRun.push(name);
    else if (c.shape.length !== t.shape.length || c.shape.some((v, i) => v !== t.shape[i])) reshaped.push({ name, shape: t.shape, course: c.shape });
  }
  const onlyCourse = [...courseTensors.keys()].filter((n) => !tensors.has(n));
  const count = (map) => new Set([...map.keys()].map((n) => /^blocks\.(\d+)\./.exec(n)?.[1]).filter((x) => x !== undefined)).size;
  const blocks = count(tensors);
  const courseBlocks = count(courseTensors);
  const notes = [];
  if (blocks !== courseBlocks) notes.push(blocks ? `${fmtInt(blocks)} blocchi transformer invece di ${fmtInt(courseBlocks)}` : "nessun blocco blocks.<i>: i nomi non seguono il codice del corso");
  // la feed-forward: primo strato (hidden, C) del blocco 0
  const ffn = reshaped.find((r) => r.name === "blocks.0.ffwd.net.0.weight");
  if (ffn) notes.push(`feed-forward con ${fmtInt(ffn.shape[0])} neuroni nascosti invece di ${fmtInt(ffn.course[0])} (${DEC.format(ffn.shape[0] / ffn.shape[1])} × n_embd invece di ${DEC.format(ffn.course[0] / ffn.course[1])} ×)`);
  const others = reshaped.filter((r) => r !== ffn && !/^blocks\.\d+\.ffwd\.net\.[02]\.(weight|bias)$/.test(r.name));
  if (others.length) notes.push(`forme diverse: ${few(others.map((r) => `${r.name} ${shapeText(r.shape)} invece di ${shapeText(r.course)}`), 3)}`);
  const newOnes = onlyRun.filter((n) => !/^blocks\.(\d+)\./.test(n) || Number(/^blocks\.(\d+)\./.exec(n)[1]) < courseBlocks);
  if (newOnes.length) notes.push(`${fmtInt(newOnes.length)} tensori che il codice del corso non crea: ${few(newOnes)}`);
  const missing = onlyCourse.filter((n) => !/^blocks\.(\d+)\./.test(n) || Number(/^blocks\.(\d+)\./.exec(n)[1]) < blocks);
  if (missing.length) notes.push(`mancano ${fmtInt(missing.length)} tensori del GPT del corso: ${few(missing)}`);
  if (engineDeclared === false) notes.push("run.json dichiara che il calcolo è diverso dal GPT del corso (engine_compatible: false), per esempio un'altra attivazione: nei pesi non si vede");
  const same = !notes.length && !onlyRun.length && !onlyCourse.length && !reshaped.length;
  return { same, notes, onlyRun, onlyCourse, reshaped, blocks, courseBlocks };
}

/** Compatibilità con il motore JS in parole semplici. */
export function compatibilityText(run) {
  if (run.compatibleWithEngine) {
    return { ok: true, title: "Il motore JS lo esegue", text: "Stessi nomi e forme del GPT del corso, calcolo dichiarato uguale: il browser può fare il passaggio in avanti con questi pesi (rete neurale con le attivazioni vere, animazioni, benchmark sull'estratto)." };
  }
  const issues = run.engine?.issues ?? [];
  return { ok: false, title: "Il motore JS non lo esegue", text: "Puoi guardare e confrontare tutti i pesi, ma il browser non sa fare il passaggio in avanti di questa architettura: niente attivazioni vere.", reasons: issues };
}

/** Data ISO 8601 → "9 novembre 2026, 01:14" (ora locale); testo com'è se non è una data. */
export function formatDate(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 60);
  return new Intl.DateTimeFormat("it-IT", { day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(d);
}

/** Byte → "839 kB", "2,6 MB", "1,2 GB". */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1000) return `${fmtInt(bytes)} byte`;
  const units = ["kB", "MB", "GB"];
  let v = bytes / 1000;
  let u = 0;
  while (v >= 1000 && u < units.length - 1) {
    v /= 1000;
    u++;
  }
  return `${new Intl.NumberFormat("it-IT", { maximumFractionDigits: v < 10 ? 1 : 0 }).format(v)} ${units[u]}`;
}

/** Da dove viene un run, in una riga. */
export function originText(entry, run) {
  if (entry.origin === "builtin") return "dati della web app (webapp/data, esportati da gpt/export.py)";
  const files = run?.meta?.files ?? [];
  const where = run?.meta?.origin ? `esportato da ${run.meta.origin}` : "caricato da file";
  const saved = entry.saved ? " · salvato in questo browser" : " · solo in questa pagina";
  return `${where}${files.length ? ` (${few(files, 2)})` : ""}${saved}`;
}

/**
 * Un testo dei file dello studente (nome di un run, nota) DENTRO una frase: tra FSI e PDI (U+2068 … U+2069),
 * i caratteri invisibili che ne isolano la direzione di scrittura. Senza, un U+202E (da destra a sinistra)
 * nel nome rovescerebbe anche il resto della frase ("ilanif isep" al posto di "pesi finali"). Un testo da
 * solo in un elemento (una voce dell'elenco delle note, il titolo della scheda) non ne ha bisogno.
 */
export const isolate = (text) => `\u2068${text}\u2069`;

/** Il prefisso delle note di trace.json copiate in run.json (TRAINING_NOTE_PREFIX di gpt/runs.py). */
export const TRAINING_NOTE_PREFIX = "storia del training: ";
const MAX_NOTE_CHARS = 2000; // come parseRunInfo (shared/weights/formats.js) e MAX_TEXT_CHARS di gpt/runs.py
const MAX_TRACE_NOTES = 20; // come MAX_TRAINING_NOTES di gpt/runs.py
const CARD_NOTES = 12;

// Caratteri di controllo e spazi di Python (str.split()): gli stessi che training_notes di gpt/runs.py
// toglie scrivendo run.json, così una nota copiata là si riconosce qui (e non compare due volte).
const BLANKS = /[\u0000-\u001f\u007f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000 ]+/g;

/**
 * Una riga semplice: uno spazio al posto di a capo, tabulazioni, caratteri di controllo, e "�" al posto dei
 * surrogati spaiati (un "\ud800" scritto nel JSON), come _one_line di gpt/runs.py.
 */
const oneLine = (text) => wellFormed(text).replace(BLANKS, " ").trim();
// toWellFormed (ES2024) manca nei browser più vecchi: lì il testo resta com'è (textContent lo mostra lo stesso).
const wellFormed = (text) => (typeof text.toWellFormed === "function" ? text.toWellFormed() : text);

/** Le stringhe di una lista presa da un file (o una stringa sola), accorciate e su una riga. */
function texts(value, max) {
  const items = typeof value === "string" ? [value] : Array.isArray(value) ? value.slice(0, 10 * max) : [];
  const out = [];
  for (const item of items) {
    if (typeof item !== "string") continue;
    // prima si taglia, poi si pulisce: un testo enorme non passa intero dalla regex
    const text = oneLine(item.slice(0, 2 * MAX_NOTE_CHARS)).slice(0, MAX_NOTE_CHARS);
    if (text) out.push(text);
    if (out.length === max) break;
  }
  return out;
}

/**
 * Le note di trace.json (gpt/recording.py: gruppi nuovi, pesi condivisi, attention saltata, valori non
 * finiti), senza prefisso. Il trace è un file dello studente: solo testi non vuoti, accorciati, al più 20.
 */
export function traceNotes(trace) {
  return texts(trace?.notes, MAX_TRACE_NOTES);
}

/**
 * Le note della scheda di un run, in ordine: quelle di run.json (dell'autore, della storia del training,
 * dell'export), poi le note di trace.json che run.json non riporta già (un run esportato prima che
 * export_run le copiasse, o una cartella con il solo trace) con il prefisso «storia del training: », poi
 * gli avvisi del caricamento. Testo semplice: chi le mostra usa textContent.
 * @returns {{notes: string[], hidden: number}} `hidden` = quante restano fuori dalle prime `limit`
 */
export function runNotes(run, { limit = CARD_NOTES } = {}) {
  const fromInfo = texts(run?.meta?.notes, 200);
  const fromTrace = traceNotes(run?.trace)
    .map((n) => `${TRAINING_NOTE_PREFIX}${n}`)
    .filter((n) => !fromInfo.some((info) => sameNote(info, n)));
  const all = [...fromInfo, ...fromTrace, ...texts(run?.meta?.warnings, 200)];
  return { notes: all.slice(0, limit), hidden: Math.max(0, all.length - limit) };
}

/** `copy` (da run.json) è la nota `note`, anche accorciata da chi l'ha scritta o letta (testi lunghi). */
function sameNote(copy, note) {
  if (copy === note) return true;
  const cut = copy.replace(/…$/, "");
  return cut.length >= MAX_NOTE_CHARS / 2 && note.startsWith(cut);
}

/** Il training del run in una riga: snapshot, durata, loss finali (dai dati, se ci sono). */
export function trainingText(run) {
  const steps = run.steps ?? [];
  const parts = [];
  if (steps.length) parts.push(`${fmtInt(steps.length)} snapshot (step ${few(steps.map(fmtInt), 12)})`);
  // trace.json (gpt/recording.py): evals = { steps: [...], train: [...], val: [...] }
  const val = run.trace?.evals?.val;
  const last = Array.isArray(val) ? val.at(-1) : null;
  if (Number.isFinite(last)) parts.push(`loss di validazione finale ${DEC.format(last)}`);
  const iters = run.trace?.train_config?.max_iters;
  if (Number.isInteger(iters)) parts.push(`${fmtInt(iters)} step di training`);
  return parts.length ? parts.join(" · ") : "nessuna storia del training (solo i pesi finali)";
}
