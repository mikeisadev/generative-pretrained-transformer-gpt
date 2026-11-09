/**
 * Controlli comuni ai formati dei pesi (manifest, trace, safetensors, file caricati).
 *
 * I file arrivano anche dallo studente (upload): vanno trattati come dati NON fidati. Per
 * questo ogni numero che diventa un indice o una dimensione viene controllato qui, con
 * messaggi in italiano che dicono quale file e quale campo non va. Niente `eval`, niente
 * oggetti indicizzati con nomi presi dai file (si usano Map: un tensore chiamato "__proto__"
 * resta un nome qualunque).
 */

const MiB = 1024 * 1024;

/**
 * Limiti predefiniti (sovrascrivibili con l'opzione `limits` delle funzioni di caricamento).
 * Il GPT del corso pesa 0,8 MB: i limiti lasciano spazio a modelli molto più grandi ma
 * impediscono che un file sbagliato (o malevolo) blocchi la pagina allocando gigabyte.
 */
export const LIMITS = Object.freeze({
  maxJsonBytes: 64 * MiB, // weights.json, trace.json, run.json, benchmark.json
  maxBinaryBytes: 1024 * MiB, // un file di pesi (.bin, .safetensors)
  maxTotalBytes: 1536 * MiB, // tutti i file letti in un caricamento
  maxSafetensorsHeaderBytes: 16 * MiB, // l'intestazione JSON di un safetensors
  maxTensors: 100_000,
  maxDims: 8,
  maxParams: 256 * MiB, // 268 milioni di valori = 1 GiB in float32
  maxNameLength: 512,
});

/**
 * Controlli bidirezionali espliciti: incassamenti e sostituzioni (U+202A–U+202E: LRE, RLE, PDF, LRO, RLO)
 * e isolamenti (U+2066–U+2069: LRI, RLI, FSI, PDI). Aprono un livello di direzione che vale fino alla fine
 * del paragrafo: un U+202E nel nome di un run rovescerebbe anche la frase del Laboratorio che lo contiene
 * («Nella scena: «…», 209.729 parametri.» si leggerebbe al contrario dopo il nome), e un U+2069 chiuderebbe
 * prima del tempo l'isolamento che il Laboratorio mette attorno ai testi dei file (isolate di
 * tabs/lab/runs/describe.js). Restano i segni U+200E, U+200F e U+061C: sono lettere invisibili con una
 * direzione, non aprono livelli, e un nome in arabo o in ebraico può averne bisogno.
 */
const BIDI_CONTROLS = /[\u202a-\u202e\u2066-\u2069]/g;
const BIDI_CONTROL = /[\u202a-\u202e\u2066-\u2069]/; // per test(): senza /g, niente lastIndex tra una chiamata e l'altra

/**
 * `value` senza i controlli bidirezionali espliciti (BIDI_CONTROLS). Un testo che ne contiene solo (o
 * soltanto spazi) non dice niente: null, come un campo assente (il nome del run torna quello della cartella).
 * @param {string|null} value
 * @returns {string|null}
 */
export function withoutBidiControls(value) {
  if (typeof value !== "string") return null;
  const clean = value.replace(BIDI_CONTROLS, "");
  return clean.trim() ? clean : null;
}

/**
 * Un testo da mettere dentro una frase dell'interfaccia (un percorso del disco dello studente, un messaggio
 * che cita un file): senza i controlli bidirezionali, il resto com'è (anche vuoto, mai null).
 */
export function stripBidiControls(text) {
  return String(text ?? "").replace(BIDI_CONTROLS, "");
}

/** `text` con i controlli bidirezionali scritti come «\u202e»: per dire nel messaggio che cosa non va. */
const showBidiControls = (text) => text.replace(BIDI_CONTROLS, (c) => `\\u${c.charCodeAt(0).toString(16)}`);

/**
 * Errore di formato: `file` dice quale file (o campo) ha il problema. Il messaggio non contiene controlli
 * bidirezionali: cita percorsi e nomi presi dai file dello studente e finisce nello stato del caricamento.
 */
export class WeightsFormatError extends Error {
  constructor(file, message) {
    super(stripBidiControls(file ? `${file}: ${message}` : message));
    this.name = "WeightsFormatError";
    this.file = file ?? null;
  }
}

/** Unisce i limiti predefiniti con quelli passati dal chiamante. */
export function withLimits(limits) {
  return limits ? { ...LIMITS, ...limits } : LIMITS;
}

export const isPlainObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

/** Prodotto delle dimensioni (1 per gli scalari, come numel di PyTorch). */
export function product(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** Megabyte con una cifra decimale, per i messaggi. */
export function megabytes(bytes) {
  return `${(bytes / MiB).toFixed(1).replace(".", ",")} MB`;
}

/**
 * Forma di un tensore: array di interi ≥ 0, al massimo `maxDims` dimensioni, numel entro il limite.
 * @returns {number[]} copia della forma
 */
export function checkShape(shape, where, limits = LIMITS) {
  if (!Array.isArray(shape)) throw new WeightsFormatError(where, "la forma (shape) deve essere una lista di interi");
  if (shape.length > limits.maxDims) {
    throw new WeightsFormatError(where, `forma con ${shape.length} dimensioni: il massimo è ${limits.maxDims}`);
  }
  for (const d of shape) {
    if (!Number.isSafeInteger(d) || d < 0) {
      throw new WeightsFormatError(where, `dimensione non valida nella forma [${shape.join(", ")}]: servono interi ≥ 0`);
    }
  }
  const numel = product(shape);
  if (numel > limits.maxParams) {
    throw new WeightsFormatError(where, `il tensore ha ${numel} valori: il limite è ${limits.maxParams}`);
  }
  return [...shape];
}

/**
 * Nome di un tensore: stringa non vuota, senza caratteri di controllo, lunghezza limitata. Niente controlli
 * bidirezionali (BIDI_CONTROLS): il nome compare in molte frasi del Laboratorio (scheda del run, motivi del
 * motore JS, didascalie, etichette della scena) e le rovescerebbe; nessun modello di PyTorch li ha nei nomi.
 */
export function checkName(name, where, limits = LIMITS) {
  if (typeof name !== "string" || name.length === 0) throw new WeightsFormatError(where, "nome del tensore mancante o vuoto");
  if (name.length > limits.maxNameLength) {
    throw new WeightsFormatError(where, `nome del tensore lungo ${name.length} caratteri: il massimo è ${limits.maxNameLength}`);
  }
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f\u007f]/.test(name)) throw new WeightsFormatError(where, `il nome ${JSON.stringify(name)} contiene caratteri di controllo`);
  if (BIDI_CONTROL.test(name)) {
    throw new WeightsFormatError(where, `il nome "${showBidiControls(name)}" contiene caratteri che cambiano la direzione del testo (U+202A–U+202E, U+2066–U+2069)`);
  }
  return name;
}

/**
 * Percorso relativo sicuro (es. il campo `file` di weights.json): niente percorsi assoluti,
 * niente "..", niente barre rovesciate. Restituisce il percorso normalizzato.
 */
export function checkRelativePath(path, where) {
  if (typeof path !== "string" || path.length === 0) throw new WeightsFormatError(where, "percorso del file mancante");
  if (path.includes("\\") || path.startsWith("/") || /^[a-z]+:/i.test(path)) {
    throw new WeightsFormatError(where, `percorso non ammesso: ${JSON.stringify(path)} (serve un percorso relativo)`);
  }
  const parts = path.split("/").filter((p) => p !== "" && p !== ".");
  if (parts.some((p) => p === "..")) throw new WeightsFormatError(where, `percorso non ammesso: ${JSON.stringify(path)} (contiene "..")`);
  if (parts.length === 0) throw new WeightsFormatError(where, "percorso del file vuoto");
  return parts.join("/");
}

/**
 * Layout di tensori in un file piatto: `[{name, shape, offset, numel}]` con offset e numel in
 * NUMERO DI ELEMENTI (come weights.json e trace.json). Controlla nomi unici, numel = prodotto
 * della forma, intervalli dentro il file, nessuna sovrapposizione e nessun buco (la somma dei
 * numel deve dare il totale): un file con byte non descritti è corrotto o di un altro formato.
 * @param {unknown} params la lista letta dal JSON
 * @param {number|undefined|null} declaredTotal total_params dichiarato (se manca: somma dei numel)
 * @returns {{params: {name: string, shape: number[], offset: number, numel: number}[], total: number}}
 */
export function checkLayout(params, declaredTotal, file, limits = LIMITS) {
  if (!Array.isArray(params) || params.length === 0) throw new WeightsFormatError(file, "manca la lista dei parametri (params)");
  if (params.length > limits.maxTensors) {
    throw new WeightsFormatError(file, `${params.length} tensori: il massimo è ${limits.maxTensors}`);
  }
  const seen = new Set();
  let sum = 0;
  const out = params.map((p, i) => {
    const where = `${file} params[${i}]`;
    if (!isPlainObject(p)) throw new WeightsFormatError(where, "ogni parametro deve essere un oggetto {name, shape, offset, numel}");
    const name = checkName(p.name, where, limits);
    if (seen.has(name)) throw new WeightsFormatError(where, `il tensore ${name} compare due volte`);
    seen.add(name);
    const shape = checkShape(p.shape, `${where} (${name})`, limits);
    const numel = product(shape);
    if (p.numel !== undefined && p.numel !== numel) {
      throw new WeightsFormatError(where, `numel di ${name} è ${p.numel}, ma la forma [${shape.join(", ")}] ne contiene ${numel}`);
    }
    if (!Number.isSafeInteger(p.offset) || p.offset < 0) {
      throw new WeightsFormatError(where, `offset di ${name} non valido (${p.offset}): serve un intero ≥ 0`);
    }
    sum += numel;
    return { name, shape, offset: p.offset, numel };
  });
  if (sum > limits.maxParams) throw new WeightsFormatError(file, `${sum} valori in tutto: il limite è ${limits.maxParams}`);
  const total = declaredTotal === undefined || declaredTotal === null ? sum : declaredTotal;
  if (!Number.isSafeInteger(total) || total < 0) throw new WeightsFormatError(file, `total_params non valido (${declaredTotal})`);
  // Ordinati per offset: ognuno deve iniziare dopo la fine del precedente e finire dentro il file.
  const sorted = [...out].sort((a, b) => a.offset - b.offset || a.numel - b.numel);
  let end = 0;
  for (const p of sorted) {
    if (p.offset < end) throw new WeightsFormatError(file, `il tensore ${p.name} (offset ${p.offset}) si sovrappone al precedente (che finisce a ${end})`);
    if (p.offset + p.numel > total) {
      throw new WeightsFormatError(file, `il tensore ${p.name} esce dal file: offset ${p.offset} + numel ${p.numel} > total_params ${total}`);
    }
    end = p.offset + p.numel;
  }
  if (sum !== total) throw new WeightsFormatError(file, `total_params è ${total} ma i parametri sommano a ${sum} (ci sono byte non descritti)`);
  return { params: out, total };
}

/**
 * Configurazione dichiarata (GPTConfig.to_dict() o simile): si tengono solo i valori semplici
 * (numeri, stringhe, booleani), così nulla di strano arriva all'interfaccia.
 */
export function cleanConfig(config, where) {
  if (config === undefined || config === null) return null;
  if (!isPlainObject(config)) throw new WeightsFormatError(where, "config deve essere un oggetto");
  const out = {};
  for (const [key, value] of Object.entries(config)) {
    // chiave e valore vanno nella scheda del run (`n_embd = 64 · …`): niente controlli bidirezionali, che
    // rovescerebbero la riga. Una chiave che ne ha non è di GPTConfig e si salta (pulita potrebbe coprirne
    // una vera, es. "\u202en_embd" → "n_embd").
    if (!["number", "string", "boolean"].includes(typeof value) || key.length > 64 || BIDI_CONTROL.test(key)) continue;
    out[key] = typeof value === "string" ? stripBidiControls(value) : value;
  }
  for (const key of ["vocab_size", "block_size", "n_embd", "n_head", "n_layer"]) {
    if (key in out && !(Number.isSafeInteger(out[key]) && out[key] > 0)) {
      throw new WeightsFormatError(where, `config.${key} deve essere un intero positivo, trovato ${JSON.stringify(out[key])}`);
    }
  }
  return out;
}

/** Vocabolario: lista di stringhe senza duplicati (token id → testo del token). */
export function checkVocab(vocab, where, expectedSize = null) {
  if (vocab === undefined || vocab === null) return null;
  if (!Array.isArray(vocab) || !vocab.every((t) => typeof t === "string")) {
    throw new WeightsFormatError(where, "il vocabolario (vocab) deve essere una lista di stringhe");
  }
  if (expectedSize !== null && vocab.length !== expectedSize) {
    throw new WeightsFormatError(where, `il vocabolario ha ${vocab.length} voci ma vocab_size è ${expectedSize}`);
  }
  if (new Set(vocab).size !== vocab.length) throw new WeightsFormatError(where, "il vocabolario contiene token duplicati");
  return [...vocab];
}

/**
 * Nome "canonico" di un parametro: toglie i prefissi che aggiungono gli involucri di PyTorch
 * (`module.` di DataParallel/DDP, `_orig_mod.` di torch.compile). Così un checkpoint salvato da
 * un modello compilato si abbina ai nomi del codice.
 */
export function canonicalName(name) {
  let out = name;
  for (;;) {
    if (out.startsWith("_orig_mod.")) out = out.slice(10);
    else if (out.startsWith("module.")) out = out.slice(7);
    else return out;
  }
}

/** Ordinamento "naturale": blocks.2 prima di blocks.10 (i numeri si confrontano come numeri). */
export function naturalCompare(a, b) {
  const pa = a.split(/(\d+)/);
  const pb = b.split(/(\d+)/);
  for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
    if (pa[i] === pb[i]) continue;
    const na = Number(pa[i]);
    const nb = Number(pb[i]);
    if (i % 2 === 1 && Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
    return pa[i] < pb[i] ? -1 : 1;
  }
  return pa.length - pb.length;
}
