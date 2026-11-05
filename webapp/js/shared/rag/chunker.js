// TokenTextSplitter didattico: porta in JavaScript la logica di
// org.springframework.ai.transformer.splitter.TokenTextSplitter (Spring AI 2.0.2,
// metodo doSplit) e di TextSplitter (metadati ereditati dai chunk).
// Sorgente: https://github.com/spring-projects/spring-ai/blob/v2.0.2/spring-ai-commons/src/main/java/org/springframework/ai/transformer/splitter/TokenTextSplitter.java
//
// L'algoritmo, passo per passo:
//   1. il testo viene trasformato in token (in Spring AI con cl100k_base; qui con il tokenizzatore
//      didattico di tokenizer.js, che lo approssima);
//   2. si prendono i primi `chunkSize` token e li si ritrasforma in testo;
//   3. se dopo questo chunk restano altri token, si cerca nel testo del chunk l'ultimo segno tra
//      '.', '?', '!' e '\n': se si trova OLTRE il carattere `minChunkSizeChars`, il chunk viene
//      accorciato fino a quel segno (così non si taglia a metà frase); altrimenti si tiene il taglio
//      netto a `chunkSize` token;
//   4. il chunk viene ripulito dagli spazi ai bordi (con keepSeparator=false anche gli a capo
//      diventano spazi) e tenuto solo se è più lungo di `minChunkLengthToEmbed` caratteri;
//   5. si tolgono dalla lista tanti token quanti ne servono a codificare il testo del chunk e si
//      ricomincia, fino a esaurire i token o a raggiungere `maxNumChunks`.
// In Spring AI 2.0.2 non c'è sovrapposizione (overlap) tra chunk: arriva con la 2.1.

import { decode, tokenize } from "./tokenizer.js";

/** Valori predefiniti di Spring AI 2.0.2 (costanti della classe TokenTextSplitter). */
export const SPRING_AI_DEFAULTS = Object.freeze({
  chunkSize: 800,
  minChunkSizeChars: 350,
  minChunkLengthToEmbed: 5,
  maxNumChunks: 10000,
  keepSeparator: true,
  punctuationMarks: Object.freeze([".", "?", "!", "\n"]),
});

/** Parametri usati nella L17 (TokenTextSplitter.builder() del corso). */
export const COURSE_SPLITTER_OPTIONS = Object.freeze({
  chunkSize: 500,
  minChunkSizeChars: 200,
  minChunkLengthToEmbed: 20,
  maxNumChunks: 10000,
  keepSeparator: true,
});

/**
 * @typedef {object} SplitterOptions
 * @property {number} [chunkSize]             dimensione target del chunk, in token
 * @property {number} [minChunkSizeChars]     sotto questa posizione (caratteri) non si arretra alla punteggiatura
 * @property {number} [minChunkLengthToEmbed] i chunk lunghi al massimo questo numero di caratteri vengono scartati
 * @property {number} [maxNumChunks]          numero massimo di chunk per testo
 * @property {boolean} [keepSeparator]        true: tiene gli a capo; false: li trasforma in spazi
 * @property {string[]} [punctuationMarks]    segni a cui si può arretrare
 */

/**
 * @typedef {object} ChunkTrace
 * Descrizione di un chunk prodotto (o scartato), per l'animazione.
 * @property {number} index               ordine di produzione
 * @property {string} text                testo finale del chunk (dopo trim)
 * @property {boolean} kept               false se scartato perché troppo corto
 * @property {"chunk"|"remainder"} phase  "remainder" = coda oltre maxNumChunks
 * @property {number} startToken          primo token consumato
 * @property {number} tokenCount          token consumati da questo chunk
 * @property {number} windowTokens        token della finestra iniziale (≤ chunkSize)
 * @property {boolean} cutAtPunctuation   true se il chunk è stato accorciato all'ultimo segno
 * @property {number} punctuationIndex    posizione del segno nel testo della finestra (-1 se nessuno)
 * @property {number} startChar           inizio nel testo originale
 * @property {number} endChar             fine (esclusa) nel testo originale del testo consumato
 * @property {string} droppedText         caratteri persi al confine (vedi nota in `splitDetailed`)
 */

/**
 * Controlla che un parametro sia un intero nell'intervallo atteso, con lo stesso spirito degli
 * Assert del costruttore Java.
 */
function assertInteger(name, value, min) {
  if (!Number.isInteger(value) || value < min) {
    throw new RangeError(`${name} deve essere un intero >= ${min} (ricevuto ${value})`);
  }
}

/** Splitter a token con la logica di Spring AI 2.0.2. */
export class TokenTextSplitter {
  /**
   * @param {SplitterOptions} [options] i valori mancanti prendono i default di Spring AI
   */
  constructor(options = {}) {
    const merged = { ...SPRING_AI_DEFAULTS, ...options };
    assertInteger("chunkSize", merged.chunkSize, 1);
    assertInteger("maxNumChunks", merged.maxNumChunks, 1);
    assertInteger("minChunkSizeChars", merged.minChunkSizeChars, 0);
    assertInteger("minChunkLengthToEmbed", merged.minChunkLengthToEmbed, 0);
    if (!Array.isArray(merged.punctuationMarks) || merged.punctuationMarks.length === 0) {
      throw new RangeError("punctuationMarks non può essere vuoto");
    }
    /** @type {Required<SplitterOptions>} */
    this.options = Object.freeze({ ...merged, punctuationMarks: [...merged.punctuationMarks], keepSeparator: !!merged.keepSeparator });
  }

  /**
   * Builder con gli stessi nomi dei metodi Java, così il codice JS somiglia a quello della L17:
   * `TokenTextSplitter.builder().withChunkSize(500).withMinChunkSizeChars(200)…build()`.
   */
  static builder() {
    const options = {};
    const builder = {
      withChunkSize: (v) => ((options.chunkSize = v), builder),
      withMinChunkSizeChars: (v) => ((options.minChunkSizeChars = v), builder),
      withMinChunkLengthToEmbed: (v) => ((options.minChunkLengthToEmbed = v), builder),
      withMaxNumChunks: (v) => ((options.maxNumChunks = v), builder),
      withKeepSeparator: (v) => ((options.keepSeparator = v), builder),
      withPunctuationMarks: (v) => ((options.punctuationMarks = v), builder),
      build: () => new TokenTextSplitter(options),
    };
    return builder;
  }

  /**
   * Posizione dell'ultimo segno di punteggiatura nel testo (getLastPunctuationIndex in Java).
   *
   * @param {string} text
   * @returns {number}
   */
  lastPunctuationIndex(text) {
    let max = -1;
    for (const mark of this.options.punctuationMarks) max = Math.max(max, text.lastIndexOf(mark));
    return max;
  }

  /**
   * Divide il testo e descrive ogni chunk (anche quelli scartati).
   *
   * Nota di fedeltà: come in Java, dopo l'arretramento i token da togliere si contano
   * ricodificando il testo accorciato. Se il segno di punteggiatura stava dentro un token più
   * lungo (per esempio ".\n\n"), il resto di quel token sparisce: lo riportiamo in `droppedText`.
   *
   * @param {string} text
   * @returns {{chunks: string[], trace: ChunkTrace[], totalTokens: number}}
   */
  splitDetailed(text) {
    if (typeof text !== "string") throw new TypeError("splitDetailed: serve una stringa");
    const { chunkSize, minChunkSizeChars, minChunkLengthToEmbed, maxNumChunks, keepSeparator } = this.options;
    const tokens = tokenize(text);
    const chunks = [];
    const trace = [];
    if (text.trim() === "") return { chunks, trace, totalTokens: tokens.length };

    let position = 0;
    let produced = 0;
    while (position < tokens.length && produced < maxNumChunks) {
      const windowEnd = Math.min(position + chunkSize, tokens.length);
      let chunkText = decode(tokens, position, windowEnd);

      // Finestra fatta solo di spazi: si salta senza contarla come chunk (come in Java).
      if (chunkText.trim() === "") {
        position = windowEnd;
        continue;
      }

      let cutAtPunctuation = false;
      let punctuationIndex = -1;
      // Si arretra alla punteggiatura solo se dopo questa finestra resta altro testo.
      if (tokens.length - position > chunkSize) {
        punctuationIndex = this.lastPunctuationIndex(chunkText);
        if (punctuationIndex !== -1 && punctuationIndex > minChunkSizeChars) {
          chunkText = chunkText.slice(0, punctuationIndex + 1);
          cutAtPunctuation = true;
        }
      }

      const finalText = keepSeparator ? chunkText.trim() : chunkText.replaceAll("\n", " ").trim();
      const kept = finalText.length > minChunkLengthToEmbed;
      if (kept) chunks.push(finalText);

      const consumed = tokenize(chunkText).length;
      const startChar = tokens[position].start;
      const endChar = tokens[position + consumed - 1].end;
      trace.push({
        index: trace.length,
        text: finalText,
        kept,
        phase: "chunk",
        startToken: position,
        tokenCount: consumed,
        windowTokens: windowEnd - position,
        cutAtPunctuation,
        punctuationIndex,
        startChar,
        endChar,
        droppedText: text.slice(startChar + chunkText.length, endChar),
      });
      position += consumed;
      produced++;
    }

    // Coda oltre maxNumChunks: in Java diventa un ultimo chunk con gli a capo trasformati in spazi.
    if (position < tokens.length) {
      const remainder = decode(tokens, position).replaceAll("\n", " ").trim();
      const kept = remainder.length > minChunkLengthToEmbed;
      if (kept) chunks.push(remainder);
      trace.push({
        index: trace.length,
        text: remainder,
        kept,
        phase: "remainder",
        startToken: position,
        tokenCount: tokens.length - position,
        windowTokens: tokens.length - position,
        cutAtPunctuation: false,
        punctuationIndex: -1,
        startChar: tokens[position].start,
        endChar: text.length,
        droppedText: "",
      });
    }
    return { chunks, trace, totalTokens: tokens.length };
  }

  /**
   * Divide un testo in chunk (splitText in Java).
   *
   * @param {string} text
   * @returns {string[]}
   */
  splitText(text) {
    return this.splitDetailed(text).chunks;
  }

  /**
   * Divide una lista di documenti (apply/split in Java). Ogni chunk eredita i metadati del
   * documento di partenza (le chiavi con valore null/undefined vengono tolte, come in Java) e
   * riceve `parent_document_id`, `chunk_index` e `total_chunks`.
   * Differenza voluta: Spring AI assegna ai chunk un UUID casuale; qui l'id è deterministico
   * (`<id documento>:<indice>`) per avere animazioni e test riproducibili.
   *
   * @param {import("./clean.js").RagDocument[]} documents
   * @returns {import("./clean.js").RagDocument[]}
   */
  apply(documents) {
    if (!Array.isArray(documents)) throw new TypeError("apply: serve una lista di documenti");
    return documents.flatMap((doc) => {
      const chunks = this.splitText(doc.text ?? "");
      const inherited = Object.fromEntries(
        Object.entries(doc.metadata ?? {}).filter(([, value]) => value !== null && value !== undefined),
      );
      return chunks.map((text, chunkIndex) => ({
        id: `${doc.id}:${chunkIndex}`,
        text,
        metadata: { ...inherited, parent_document_id: doc.id, chunk_index: chunkIndex, total_chunks: chunks.length },
      }));
    });
  }

  /** Alias di `apply`, come `split(List<Document>)` in Java. */
  split(documents) {
    return this.apply(documents);
  }
}
