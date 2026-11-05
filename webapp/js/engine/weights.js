/**
 * Pesi del GPT: quali tensori ci aspettiamo, controllo del manifest e lettura dei file binari.
 *
 * Formati (vedi gpt/export.py e gpt/recording.py):
 * - `weights.json` elenca i parametri `{name, shape, offset, numel}`, con offset e numel in
 *   NUMERO DI FLOAT; `weights.f32.bin` è la concatenazione float32 little-endian.
 * - gli snapshot del training `training/snap_XXXXX.f16.bin` hanno lo stesso layout in float16.
 */

import { float16ToFloat32 } from "../core/binary.js";

export const WEIGHTS_FORMAT_VERSION = 1;

/** Moltiplicatore dello strato nascosto della feed-forward (gpt/feedforward.py: 4 · n_embd). */
export const FFN_MULTIPLIER = 4;

// I typed array usano l'ordine dei byte della macchina: i file sono little-endian.
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * Parametri appresi del GPT con nomi e forme di `model.named_parameters()` in PyTorch,
 * nello stesso ordine. Serve a controllare che i dati corrispondano all'architettura
 * implementata qui (e, nei test, a costruire modelli sintetici).
 * @param {{vocab_size:number, block_size:number, n_embd:number, n_head:number, n_layer:number}} config
 * @returns {{name: string, shape: number[]}[]}
 */
export function expectedParameters(config) {
  const { vocab_size: V, block_size: B, n_embd: C, n_head: H, n_layer: L } = config;
  for (const [key, value] of Object.entries({ vocab_size: V, block_size: B, n_embd: C, n_head: H, n_layer: L })) {
    if (!Number.isInteger(value) || value <= 0) throw new Error(`config.${key} deve essere un intero positivo, ricevuto ${value}`);
  }
  if (C % H !== 0) throw new Error(`n_embd (${C}) deve essere divisibile per n_head (${H})`);
  const hs = C / H;
  const hidden = FFN_MULTIPLIER * C;
  const params = [
    { name: "token_embedding_table.weight", shape: [V, C] },
    { name: "position_embedding_table.weight", shape: [B, C] },
  ];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}`;
    params.push({ name: `${p}.ln1.weight`, shape: [C] }, { name: `${p}.ln1.bias`, shape: [C] });
    for (let h = 0; h < H; h++) {
      for (const kind of ["key", "query", "value"]) params.push({ name: `${p}.sa.heads.${h}.${kind}.weight`, shape: [hs, C] });
    }
    params.push(
      { name: `${p}.sa.proj.weight`, shape: [C, C] },
      { name: `${p}.sa.proj.bias`, shape: [C] },
      { name: `${p}.ln2.weight`, shape: [C] },
      { name: `${p}.ln2.bias`, shape: [C] },
      { name: `${p}.ffwd.net.0.weight`, shape: [hidden, C] },
      { name: `${p}.ffwd.net.0.bias`, shape: [hidden] },
      { name: `${p}.ffwd.net.2.weight`, shape: [C, hidden] },
      { name: `${p}.ffwd.net.2.bias`, shape: [C] },
    );
  }
  params.push(
    { name: "ln_f.weight", shape: [C] },
    { name: "ln_f.bias", shape: [C] },
    { name: "lm_head.weight", shape: [V, C] },
    { name: "lm_head.bias", shape: [V] },
  );
  return params;
}

const product = (shape) => shape.reduce((a, b) => a * b, 1);
const sameShape = (a, b) => Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * Controlla il manifest (weights.json) e restituisce il layout validato.
 * Errori chiari invece di numeri sbagliati in silenzio se i dati non corrispondono al codice.
 * @returns {{name: string, shape: number[], offset: number, numel: number}[]}
 */
export function validateManifest(manifest) {
  if (!manifest || typeof manifest !== "object") throw new Error("manifest dei pesi mancante");
  if (manifest.format_version !== WEIGHTS_FORMAT_VERSION) {
    throw new Error(`formato dei pesi non supportato: ${manifest.format_version} (atteso ${WEIGHTS_FORMAT_VERSION})`);
  }
  if (manifest.dtype !== "float32" || manifest.byte_order !== "little") {
    throw new Error(`pesi attesi in float32 little-endian, trovato ${manifest.dtype}/${manifest.byte_order}`);
  }
  const { config, vocab, params, total_params: total } = manifest;
  if (!Array.isArray(vocab) || vocab.length !== config?.vocab_size) {
    throw new Error(`il vocabolario ha ${vocab?.length} voci ma vocab_size è ${config?.vocab_size}`);
  }
  if (new Set(vocab).size !== vocab.length) throw new Error("il vocabolario contiene token duplicati");

  const byName = new Map((params ?? []).map((p) => [p.name, p]));
  const expected = expectedParameters(config);
  const layout = expected.map(({ name, shape }) => {
    const p = byName.get(name);
    if (!p) throw new Error(`parametro mancante nel manifest: ${name}`);
    if (!sameShape(p.shape, shape)) throw new Error(`forma di ${name}: attesa [${shape}], trovata [${p.shape}]`);
    if (p.numel !== product(shape)) throw new Error(`numel di ${name} non coerente con la forma`);
    if (!Number.isInteger(p.offset) || p.offset < 0 || p.offset + p.numel > total) {
      throw new Error(`offset di ${name} fuori dal file dei pesi`);
    }
    return { name, shape: [...shape], offset: p.offset, numel: p.numel };
  });
  if (byName.size !== expected.length) {
    const known = new Set(expected.map((p) => p.name));
    const extra = [...byName.keys()].filter((n) => !known.has(n));
    throw new Error(`parametri inattesi nel manifest (architettura diversa?): ${extra.join(", ")}`);
  }
  const counted = layout.reduce((sum, p) => sum + p.numel, 0);
  if (counted !== total) throw new Error(`total_params è ${total} ma i parametri sommano a ${counted}`);
  return layout;
}

/** Accetta ArrayBuffer o una vista (Uint8Array, Buffer di Node…) e restituisce i byte come ArrayBuffer. */
export function toArrayBuffer(data) {
  if (data instanceof ArrayBuffer) return data;
  if (ArrayBuffer.isView(data)) {
    return data.byteOffset === 0 && data.byteLength === data.buffer.byteLength
      ? data.buffer
      : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  throw new TypeError("atteso un ArrayBuffer o un typed array");
}

function checkByteLength(buffer, expected, what) {
  if (!LITTLE_ENDIAN) throw new Error("questa piattaforma è big-endian: i file dei pesi sono little-endian");
  if (buffer.byteLength !== expected) {
    throw new Error(`${what}: attesi ${expected} byte, ricevuti ${buffer.byteLength}`);
  }
}

/** File float32 dei pesi → Float32Array di `count` valori (vista sui byte, senza copia). */
export function decodeFloat32(data, count) {
  const buffer = toArrayBuffer(data);
  checkByteLength(buffer, 4 * count, "file dei pesi float32");
  return new Float32Array(buffer);
}

/** Snapshot float16 del training → Float32Array di `count` valori (conversione esatta). */
export function decodeFloat16(data, count) {
  const buffer = toArrayBuffer(data);
  checkByteLength(buffer, 2 * count, "snapshot float16");
  return float16ToFloat32(buffer);
}

/** Il layout degli snapshot (trace.json) deve coincidere con quello dei pesi. */
export function assertSameLayout(traceParams, layout) {
  const byName = new Map((traceParams ?? []).map((p) => [p.name, p]));
  for (const p of layout) {
    const q = byName.get(p.name);
    if (!q || q.offset !== p.offset || q.numel !== p.numel) {
      throw new Error(`layout degli snapshot diverso da quello dei pesi (parametro ${p.name})`);
    }
  }
}
