/**
 * Lettura dei file `.safetensors` (formato di Hugging Face, https://github.com/huggingface/safetensors).
 *
 * Struttura del file:
 *   8 byte   N = lunghezza dell'intestazione, intero senza segno a 64 bit little-endian
 *   N byte   intestazione JSON in UTF-8: { "__metadata__": {stringa: stringa},
 *            "nome": {"dtype": "F32", "shape": [r, c], "data_offsets": [inizio, fine]}, … }
 *   resto    i dati dei tensori, uno dopo l'altro; gli offset sono relativi all'inizio di questa parte
 *
 * Perché è sicuro da aprire nel browser: è solo JSON + numeri (a differenza dei checkpoint
 * pickle di PyTorch, che possono eseguire codice). Qui si controlla comunque tutto, come fa la
 * libreria ufficiale: intestazione di lunghezza limitata e dentro il file, offset interi e nei
 * limiti, byte coerenti con forma e dtype, nessuna sovrapposizione, nessun buco tra i tensori e
 * nessun byte in più dopo l'ultimo (la regola ufficiale contro i file "poliglotti").
 *
 * dtype supportati: F32, F16, BF16 (convertiti in Float32Array). Gli altri danno un errore chiaro.
 */

import { halfToFloat } from "../../core/binary.js";
import { WeightsFormatError, checkName, checkShape, isPlainObject, product, withLimits } from "./validate.js";

/** Byte per elemento dei dtype che sappiamo convertire in float32. */
export const SAFETENSORS_DTYPES = Object.freeze({ F32: 4, F16: 2, BF16: 2 });
/** dtype del formato che riconosciamo ma non convertiamo (per un messaggio più preciso). */
const KNOWN_UNSUPPORTED = new Set(["F64", "F8_E4M3", "F8_E5M2", "BOOL", "U8", "I8", "I16", "U16", "I32", "U32", "I64", "U64"]);
const DTYPE_NAMES = { F32: "float32", F16: "float16", BF16: "bfloat16" };

let halfTable = null;
/** Tabella dei 65.536 half-float → float32 (256 KB, costruita una volta con halfToFloat di core/binary.js). */
function halfLookup() {
  if (!halfTable) {
    halfTable = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) halfTable[h] = halfToFloat(h);
  }
  return halfTable;
}

function bytesOf(data) {
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new TypeError("atteso un ArrayBuffer o un typed array con i byte del file");
}

/**
 * Legge un safetensors.
 * @param {ArrayBuffer|ArrayBufferView} data byte del file
 * @param {{file?: string, limits?: object}} [options] `file`: nome da citare negli errori
 * @returns {{tensors: Map<string, {name: string, shape: number[], numel: number, dtype: string, data: Float32Array}>,
 *            flat: Float32Array, metadata: Map<string, string>, order: string[]}}
 *   `flat` contiene tutti i valori, nell'ordine dei byte del file; ogni `data` è una vista su `flat`;
 *   `dtype` è quello originale ("float32", "float16", "bfloat16")
 */
export function parseSafetensors(data, { file = "file safetensors", limits } = {}) {
  const lim = withLimits(limits);
  const bytes = bytesOf(data);
  if (bytes.byteLength < 8) throw new WeightsFormatError(file, "troppo corto per un safetensors (servono almeno 8 byte di intestazione)");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // u64 little-endian: la parte alta deve essere piccola, altrimenti il numero non sta nei limiti.
  const low = view.getUint32(0, true);
  const high = view.getUint32(4, true);
  const headerLength = high * 2 ** 32 + low;
  if (headerLength > lim.maxSafetensorsHeaderBytes) {
    throw new WeightsFormatError(file, `intestazione dichiarata di ${headerLength} byte: il limite è ${lim.maxSafetensorsHeaderBytes} (file non valido?)`);
  }
  if (8 + headerLength > bytes.byteLength) {
    throw new WeightsFormatError(file, `intestazione dichiarata di ${headerLength} byte ma il file ne ha solo ${bytes.byteLength - 8} dopo la lunghezza`);
  }
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(8, 8 + headerLength));
  } catch {
    throw new WeightsFormatError(file, "l'intestazione non è testo UTF-8 valido");
  }
  let header;
  try {
    header = JSON.parse(text);
  } catch (error) {
    throw new WeightsFormatError(file, `intestazione JSON non valida (${error.message})`);
  }
  if (!isPlainObject(header)) throw new WeightsFormatError(file, "l'intestazione deve essere un oggetto JSON {nome: {dtype, shape, data_offsets}}");

  const metadata = new Map();
  const entries = [];
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") {
      if (!isPlainObject(info)) throw new WeightsFormatError(file, "__metadata__ deve essere un oggetto {stringa: stringa}");
      for (const [k, v] of Object.entries(info)) {
        if (typeof v !== "string") throw new WeightsFormatError(file, `__metadata__.${k} deve essere una stringa`);
        metadata.set(k, v);
      }
      continue;
    }
    entries.push([name, info]);
  }
  if (entries.length > lim.maxTensors) throw new WeightsFormatError(file, `${entries.length} tensori: il massimo è ${lim.maxTensors}`);

  const dataStart = 8 + headerLength;
  const dataLength = bytes.byteLength - dataStart;
  let totalNumel = 0;
  const tensors = entries.map(([name, info]) => {
    const where = `${file} (tensore ${JSON.stringify(name)})`;
    checkName(name, file, lim);
    if (!isPlainObject(info)) throw new WeightsFormatError(where, "descrizione del tensore mancante: servono dtype, shape e data_offsets");
    const { dtype } = info;
    if (typeof dtype !== "string") throw new WeightsFormatError(where, "dtype mancante");
    // hasOwn e non `in`: "toString", "constructor" o "__proto__" arrivano dal file e sono nel prototipo
    if (!Object.hasOwn(SAFETENSORS_DTYPES, dtype)) {
      const known = KNOWN_UNSUPPORTED.has(dtype) ? "" : " (dtype sconosciuto)";
      throw new WeightsFormatError(
        where,
        `dtype ${dtype} non supportato${known}: sono supportati F32, F16 e BF16. In Python: salva i pesi con model.float() o .half()`,
      );
    }
    const shape = checkShape(info.shape, where, lim);
    const offsets = info.data_offsets;
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every((o) => Number.isSafeInteger(o) && o >= 0)) {
      throw new WeightsFormatError(where, "data_offsets deve essere [inizio, fine] con interi ≥ 0");
    }
    const [begin, end] = offsets;
    if (begin > end) throw new WeightsFormatError(where, `data_offsets [${begin}, ${end}]: l'inizio supera la fine`);
    if (end > dataLength) throw new WeightsFormatError(where, `data_offsets [${begin}, ${end}] escono dai dati (${dataLength} byte dopo l'intestazione)`);
    const numel = product(shape);
    const expected = numel * SAFETENSORS_DTYPES[dtype];
    if (end - begin !== expected) {
      throw new WeightsFormatError(where, `occupa ${end - begin} byte ma forma [${shape.join(", ")}] in ${dtype} ne richiede ${expected}`);
    }
    totalNumel += numel;
    if (totalNumel > lim.maxParams) throw new WeightsFormatError(file, `più di ${lim.maxParams} valori in tutto: oltre il limite`);
    return { name, dtype, shape, numel, begin, end };
  });

  // I tensori devono coprire i dati da 0 alla fine, uno dopo l'altro: niente sovrapposizioni, niente buchi.
  const byOffset = [...tensors].sort((a, b) => a.begin - b.begin || a.end - b.end);
  let cursor = 0;
  for (const t of byOffset) {
    if (t.begin < cursor) throw new WeightsFormatError(file, `il tensore ${JSON.stringify(t.name)} (byte ${t.begin}-${t.end}) si sovrappone al precedente`);
    if (t.begin > cursor) throw new WeightsFormatError(file, `buco nei dati: i byte ${cursor}-${t.begin} non appartengono a nessun tensore`);
    cursor = t.end;
  }
  if (cursor !== dataLength) {
    throw new WeightsFormatError(file, `${dataLength - cursor} byte in più dopo l'ultimo tensore: il formato non li ammette`);
  }

  // Un solo Float32Array per tutti i valori (nell'ordine del file): ogni tensore ne è una vista.
  const flat = new Float32Array(totalNumel);
  const flatBits = new Uint32Array(flat.buffer);
  const out = new Map();
  let at = 0;
  for (const t of byOffset) {
    const start = dataStart + t.begin;
    decodeInto(t.dtype, view, start, t.numel, flat, flatBits, at);
    out.set(t.name, { name: t.name, shape: t.shape, numel: t.numel, dtype: DTYPE_NAMES[t.dtype], data: flat.subarray(at, at + t.numel) });
    at += t.numel;
  }
  return { tensors: out, flat, metadata, order: byOffset.map((t) => t.name) };
}

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * Converte `count` valori da `view[start…]` in float32, scrivendoli in `flat[at…]`.
 * Con i byte allineati (il caso normale) si copia con un typed array, molto più veloce di
 * DataView elemento per elemento; altrimenti (intestazione di lunghezza dispari) DataView.
 */
function decodeInto(dtype, view, start, count, flat, flatBits, at) {
  const offset = view.byteOffset + start;
  const size = SAFETENSORS_DTYPES[dtype];
  const aligned = LITTLE_ENDIAN && offset % size === 0;
  if (dtype === "F32") {
    if (aligned) flat.set(new Float32Array(view.buffer, offset, count), at);
    else for (let i = 0; i < count; i++) flat[at + i] = view.getFloat32(start + 4 * i, true);
    return;
  }
  const halves = aligned ? new Uint16Array(view.buffer, offset, count) : null;
  const read = halves ? (i) => halves[i] : (i) => view.getUint16(start + 2 * i, true);
  if (dtype === "F16") {
    const table = halfLookup();
    for (let i = 0; i < count; i++) flat[at + i] = table[read(i)];
  } else {
    // bfloat16 = i 16 bit alti di un float32: basta spostarli a sinistra (conversione esatta).
    for (let i = 0; i < count; i++) flatBits[at + i] = read(i) << 16;
  }
}

/**
 * Metadati di un safetensors scritti come JSON (es. `{"config": "{…}", "vocab": "[…]"}`): li
 * decodifica quando sono JSON validi e ignora gli altri.
 * @param {Map<string, string>} metadata
 * @returns {{config: object|null, vocab: string[]|null, other: Record<string, string>}}
 */
export function safetensorsInfo(metadata) {
  const parse = (key) => {
    const text = metadata.get(key);
    if (typeof text !== "string") return null;
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  };
  const config = parse("config");
  const vocab = parse("vocab");
  const other = {};
  for (const [k, v] of metadata) if (k !== "config" && k !== "vocab") other[k] = v;
  return {
    config: isPlainObject(config) ? config : null,
    vocab: Array.isArray(vocab) && vocab.every((t) => typeof t === "string") ? vocab : null,
    other,
  };
}
