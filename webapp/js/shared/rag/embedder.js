// Modello di embedding DIDATTICO e deterministico.
//
// SIMULAZIONE DIDATTICA: nelle lezioni l'embedding è calcolato da text-embedding-3-small di
// OpenAI, una rete neurale che produce 1536 numeri per testo e che ha imparato da enormi quantità
// di dati a mettere vicini testi con significato simile (anche senza parole in comune).
// Qui non c'è nessuna rete neurale: il vettore si costruisce con regole esplicite, così ogni
// passaggio si può mostrare e capire.
//
//   1. normalizzazione: minuscole, accenti tolti, solo lettere e cifre;
//   2. parole vuote (articoli, preposizioni…) scartate, le altre ridotte a una radice
//      ("giorni", "giorno" → "giorn") con poche regole sui suffissi italiani e un piccolo
//      dizionario di sinonimi (la "conoscenza" che il modello vero impara dai dati);
//   3. feature: radici (unigrammi), coppie di radici vicine (bigrammi) e trigrammi di caratteri
//      (per riconoscere varianti della stessa parola);
//   4. peso di ogni feature = tf sublineare (1 + ln tf) × idf (parole rare nel corpus pesano di
//      più) × peso del tipo di feature;
//   5. hashing: ogni feature finisce in una delle 256 dimensioni scelta da una funzione di hash,
//      con un segno + o − anch'esso dato dall'hash (le collisioni si compensano in media).
//      Eccezione: le feature mai viste nel corpus finiscono tutte in una dimensione riservata
//      ("sconosciute"). Pesano sulla lunghezza del vettore, ma non possono avvicinarlo a nessun
//      documento: senza questo accorgimento collisioni casuali farebbero sembrare simile a un
//      documento anche una domanda fuori tema;
//   6. normalizzazione L2: il vettore viene portato a lunghezza 1, come fanno i modelli OpenAI,
//      così la cosine similarity coincide con il prodotto scalare.
//
// Ultimo dettaglio, dichiarato: i modelli neurali reali producono vettori che condividono una
// componente comune (anisotropia), per cui anche due testi scollegati hanno similarità
// positiva. Lo simuliamo con un peso costante su una dimensione riservata
// (`sharedComponent`): alza tutte le similarità della stessa quantità e rende sensata la soglia
// similarityThreshold = 0.5 usata nelle lezioni.

import { hashString } from "./random.js";

/**
 * Parole vuote italiane: non distinguono un documento da un altro. Passano dalla stessa
 * normalizzazione del testo ("perché" → "perche"), altrimenti quelle accentate non
 * verrebbero mai riconosciute.
 */
const STOPWORDS = new Set(
  (
    "a ad al allo alla ai agli alle anche ancora avere aver ha hai ho hanno abbiamo avete c che chi ci " +
    "come con col coi cosa cui da dal dallo dalla dai dagli dalle dei degli del della delle dello di " +
    "dove e ed è era essere fa gli i il in io l la le lei lo loro lui ma mi mia mie miei mio ne negli " +
    "nei nel nella nelle nello noi non nostro nostra o od ogni per perché più po poi può qual quale " +
    "quali qualche quando quanto quanti quante questo questa questi queste quello quella se sei si " +
    "sia siamo sono sta stato sto su sua sue sui sul sulla suo suoi te ti tra fra tu tua tuo tuoi " +
    "un una uno vi voi devo deve devono posso puoi possono sempre solo molto fare faccio " +
    "the of and to"
  )
    .split(" ")
    .map(normalizeText),
);

/**
 * Sinonimi ricondotti a una radice comune (dopo la riduzione dei suffissi). Un vero modello
 * di embedding impara queste equivalenze dai dati; qui ne scriviamo a mano poche, legate al
 * dominio del gestionale, e lo dichiariamo.
 */
const SYNONYMS = new Map([
  ["vacanz", "fer"],
  ["remot", "smart"],
  ["agil", "smart"],
  ["cas", "smart"],
  ["domicil", "smart"],
  ["pc", "portatil"],
  ["laptop", "portatil"],
  ["computer", "portatil"],
  ["credenzial", "password"],
  ["chiav", "password"],
  ["pdf", "document"],
  ["fil", "document"],
  ["acquist", "ordin"],
  ["compr", "ordin"],
  ["neoassunt", "assunt"],
  ["truff", "phishing"],
  ["sospett", "phishing"],
  ["medic", "malatt"],
  ["medich", "malatt"],
  ["colleg", "dipendent"],
  ["collegh", "dipendent"],
]);

/** Pesi relativi predefiniti dei tre tipi di feature. */
export const DEFAULT_FEATURE_WEIGHTS = Object.freeze({ word: 1, bigram: 0.3, char: 0.2 });

/** Descrizione dichiarata del modello, da mostrare nella UI. */
export const EMBEDDER_INFO = Object.freeze({
  name: "didactic-hashing-embedder",
  simulates: "text-embedding-3-small (OpenAI)",
  realDimensions: 1536,
  description:
    "Simulazione didattica di un modello di embedding: niente rete neurale, ma radici delle parole, " +
    "coppie di parole e trigrammi di caratteri pesati con tf-idf e distribuiti con l'hashing in un " +
    "vettore di lunghezza 1. Il vero text-embedding-3-small è una rete neurale che produce 1536 " +
    "dimensioni e coglie il significato anche senza parole in comune.",
});

/**
 * Minuscole, accenti rimossi (NFD + rimozione dei segni diacritici), apostrofi e punteggiatura
 * trasformati in spazi.
 *
 * @param {string} text
 * @returns {string}
 */
export function normalizeText(text) {
  return String(text)
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

/**
 * Riduce una parola alla sua radice con poche regole sui suffissi italiani. È uno stemmer
 * volutamente semplice (molto più rozzo dello stemmer di Porter per l'italiano): sbaglia in
 * qualche caso, ma rende vicine le forme più comuni della stessa parola.
 *
 * @param {string} word parola già normalizzata
 * @returns {string}
 */
export function stem(word) {
  if (/\d/.test(word) || word.length <= 3) return SYNONYMS.get(word) ?? word;
  let root = word;
  if (root.length > 6 && /(?:are|ere|ire)$/.test(root)) root = root.slice(0, -3); // infiniti
  else if (root.length > 7 && /zion[ei]$/.test(root)) root = root.slice(0, -1); // -zione/-zioni
  else if (root.length > 7 && root.endsWith("mente")) root = root.slice(0, -5); // avverbi
  // Al massimo due vocali finali (genere e numero: "uffici"/"ufficio" → "uffic").
  for (let i = 0; i < 2 && root.length > 3 && /[aeiou]$/.test(root); i++) root = root.slice(0, -1);
  // Participi passati: "cambiata" → "cambiat" → "camb", come "cambiare" → "camb".
  if (root.length > 5 && /(?:at|it|ut)$/.test(root)) root = root.slice(0, -2);
  return SYNONYMS.get(root) ?? root;
}

/**
 * @typedef {object} AnalyzedWord
 * @property {string} word   parola normalizzata
 * @property {string} stem   radice (vuota se parola vuota)
 * @property {boolean} stop  true se scartata come parola vuota
 */

/**
 * Passi 1-2: normalizzazione, parole vuote, radici.
 *
 * @param {string} text
 * @returns {AnalyzedWord[]}
 */
export function analyze(text) {
  const normalized = normalizeText(text);
  if (normalized === "") return [];
  return normalized.split(" ").map((word) => {
    const stop = STOPWORDS.has(word) || (word.length === 1 && !/\d/.test(word));
    return { word, stem: stop ? "" : stem(word), stop };
  });
}

/**
 * Passo 3: feature con il loro conteggio (tf). Le chiavi hanno un prefisso per tipo, così la
 * stessa stringa in tipi diversi finisce (di solito) in dimensioni diverse.
 *
 * @param {AnalyzedWord[]} words
 * @returns {Map<string, {kind: "word"|"bigram"|"char", value: string, tf: number}>}
 */
function extractFeatures(words) {
  const features = new Map();
  const add = (kind, value) => {
    const key = `${kind}:${value}`;
    const entry = features.get(key);
    if (entry) entry.tf++;
    else features.set(key, { kind, value, tf: 1 });
  };
  const stems = words.filter((w) => !w.stop).map((w) => w.stem);
  stems.forEach((root, i) => {
    add("word", root);
    if (i > 0) add("bigram", `${stems[i - 1]} ${root}`);
    const padded = `#${root}#`;
    for (let j = 0; j + 3 <= padded.length; j++) add("char", padded.slice(j, j + 3));
  });
  return features;
}

/**
 * @typedef {object} FeatureExplanation
 * @property {string} feature   chiave (tipo:valore)
 * @property {"word"|"bigram"|"char"} kind
 * @property {string} value
 * @property {number} tf        quante volte compare
 * @property {number} idf       rarità nel corpus
 * @property {number} weight    peso finale (prima della normalizzazione)
 * @property {number} dimension dimensione in cui finisce
 * @property {1|-1} sign        segno dato dall'hash
 * @property {boolean} known    false se la feature non compare nel corpus (va in UNKNOWN_DIMENSION)
 */

/**
 * @typedef {object} EmbeddingExplanation
 * @property {string} text
 * @property {string} normalized
 * @property {AnalyzedWord[]} words
 * @property {FeatureExplanation[]} features
 * @property {Float32Array} raw          vettore prima della normalizzazione
 * @property {number} norm               lunghezza del vettore grezzo
 * @property {Float32Array} vector       vettore finale (lunghezza 1)
 * @property {{dimension: number, value: number}[]} topDimensions le dimensioni più "accese"
 * @property {{id: string, title: string, description: string}[]} steps passi per l'animazione
 */

/** Dimensione riservata alla componente comune (anisotropia simulata). */
export const SHARED_DIMENSION = 0;
/** Dimensione riservata alle feature che non compaiono nel corpus usato per `fit`. */
export const UNKNOWN_DIMENSION = 1;
/** Prima dimensione disponibile per l'hashing. */
const FIRST_HASHED_DIMENSION = 2;

/** Modello di embedding didattico basato sull'hashing delle feature. */
export class HashingEmbedder {
  /**
   * @param {object} [options]
   * @param {number} [options.dimensions=256]        dimensione dei vettori
   * @param {number} [options.seed=7]                seed dell'hash (cambia la disposizione delle dimensioni)
   * @param {number} [options.sharedComponent=0.66]  peso della componente comune (0 = disattivata):
   *   due testi senza feature in comune hanno similarità sharedComponent², circa 0.44. Il valore è
   *   calibrato sul corpus del corso perché la soglia 0.5 delle lezioni separi domande pertinenti
   *   e domande fuori tema
   * @param {{word: number, bigram: number, char: number}} [options.featureWeights] pesi dei tipi di feature
   */
  constructor({
    dimensions = 256,
    seed = 7,
    sharedComponent = 0.66,
    featureWeights = DEFAULT_FEATURE_WEIGHTS,
  } = {}) {
    if (!Number.isInteger(dimensions) || dimensions < 8) throw new RangeError("dimensions deve essere un intero >= 8");
    if (!(sharedComponent >= 0 && sharedComponent < 1)) throw new RangeError("sharedComponent deve essere in [0, 1)");
    this.dimensions = dimensions;
    this.seed = seed >>> 0;
    this.sharedComponent = sharedComponent;
    this.featureWeights = Object.freeze({ ...DEFAULT_FEATURE_WEIGHTS, ...featureWeights });
    /** @type {Map<string, number>} */
    this.documentFrequency = new Map();
    this.documentCount = 0;
    this.info = { ...EMBEDDER_INFO, dimensions };
  }

  /** true dopo `fit`: senza corpus tutte le feature hanno idf = 1 e nessuna è "sconosciuta". */
  get fitted() {
    return this.documentCount > 0;
  }

  /**
   * Calcola l'idf sul corpus. Va chiamato una volta, prima di indicizzare: come un modello vero,
   * dopo l'addestramento l'embedder resta fisso (altrimenti i vettori già salvati non sarebbero
   * più confrontabili con quelli nuovi).
   *
   * @param {string[]} texts
   * @returns {this}
   */
  fit(texts) {
    if (!Array.isArray(texts)) throw new TypeError("fit: serve una lista di testi");
    this.documentFrequency = new Map();
    this.documentCount = texts.length;
    for (const text of texts) {
      for (const key of extractFeatures(analyze(text)).keys()) {
        this.documentFrequency.set(key, (this.documentFrequency.get(key) ?? 0) + 1);
      }
    }
    return this;
  }

  /**
   * idf smussato: ln((1 + N) / (1 + df)) + 1. Una feature mai vista nel corpus (df = 0)
   * prende il valore massimo.
   *
   * @param {string} key
   * @returns {number}
   */
  idf(key) {
    if (!this.fitted) return 1;
    const df = this.documentFrequency.get(key) ?? 0;
    return Math.log((1 + this.documentCount) / (1 + df)) + 1;
  }

  /**
   * Dimensione e segno di una feature (trucco dell'hashing). Le feature sconosciute vanno
   * nella dimensione riservata UNKNOWN_DIMENSION.
   *
   * @param {string} key
   * @returns {{dimension: number, sign: 1|-1, known: boolean}}
   */
  slot(key) {
    const known = !this.fitted || this.documentFrequency.has(key);
    if (!known) return { dimension: UNKNOWN_DIMENSION, sign: 1, known };
    const hash = hashString(key, this.seed);
    return {
      dimension: FIRST_HASHED_DIMENSION + (hash % (this.dimensions - FIRST_HASHED_DIMENSION)),
      sign: hashString(key, this.seed ^ 0x9e3779b9) & 1 ? 1 : -1,
      known,
    };
  }

  /**
   * Calcola embedding e spiegazione completa dei passaggi.
   *
   * @param {string} text
   * @returns {EmbeddingExplanation}
   */
  explain(text) {
    const words = analyze(text);
    const features = [];
    const raw = new Float32Array(this.dimensions);
    let unknownEnergy = 0;
    for (const [key, { kind, value, tf }] of extractFeatures(words)) {
      const idf = this.idf(key);
      const { dimension, sign, known } = this.slot(key);
      const weight = (1 + Math.log(tf)) * idf * this.featureWeights[kind];
      // Le sconosciute si sommano "in quadratura": ognuna conta come se avesse una direzione
      // tutta sua, che nessun documento del corpus condivide.
      if (known) raw[dimension] += sign * weight;
      else unknownEnergy += weight * weight;
      features.push({ feature: key, kind, value, tf, idf, weight, dimension, sign, known });
    }
    raw[UNKNOWN_DIMENSION] = Math.sqrt(unknownEnergy);
    let sumSquares = 0;
    for (let i = 0; i < raw.length; i++) sumSquares += raw[i] * raw[i];
    const norm = Math.sqrt(sumSquares);

    const vector = new Float32Array(this.dimensions);
    if (norm > 0) {
      // Parte "specifica" scalata a lunghezza √(1 - s²), componente comune s: totale lunghezza 1.
      const s = this.sharedComponent;
      const scale = Math.sqrt(1 - s * s) / norm;
      for (let i = 0; i < raw.length; i++) vector[i] = raw[i] * scale;
      vector[SHARED_DIMENSION] = s;
    }
    const topDimensions = [...vector]
      .map((value, dimension) => ({ dimension, value }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || a.dimension - b.dimension)
      .slice(0, 12);

    return {
      text,
      normalized: normalizeText(text),
      words,
      features: features.sort((a, b) => b.weight - a.weight || (a.feature < b.feature ? -1 : 1)),
      raw,
      norm,
      vector,
      topDimensions,
      steps: [
        { id: "normalize", title: "Normalizzazione", description: "Minuscole, accenti e punteggiatura tolti." },
        { id: "words", title: "Parole e radici", description: "Via le parole vuote; le altre ridotte alla radice." },
        { id: "features", title: "Feature", description: "Radici, coppie di radici vicine e trigrammi di caratteri." },
        { id: "weights", title: "Pesi tf-idf", description: "Contano di più le feature frequenti nel testo e rare nel corpus." },
        {
          id: "hashing",
          title: "Hashing",
          description: `Ogni feature cade in una delle ${this.dimensions} dimensioni, con segno + o −; quelle mai viste nel corpus in una dimensione riservata.`,
        },
        { id: "normalize-l2", title: "Normalizzazione L2", description: "Il vettore viene portato a lunghezza 1." },
      ],
    };
  }

  /**
   * Embedding di un testo: vettore di `dimensions` numeri con norma 1 (vettore nullo se il testo
   * non contiene parole utili).
   *
   * @param {string} text
   * @returns {Float32Array}
   */
  embed(text) {
    return this.explain(text).vector;
  }

  /**
   * Embedding di più testi (come EmbeddingModel.embed(List<String>) in Spring AI).
   *
   * @param {string[]} texts
   * @returns {Float32Array[]}
   */
  embedAll(texts) {
    return texts.map((text) => this.embed(text));
  }
}
