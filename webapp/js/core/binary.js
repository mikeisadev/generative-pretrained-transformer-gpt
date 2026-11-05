/**
 * Decodifica dei dati binari esportati dal Python (gpt/export.py).
 */

/** Base64 → Uint8Array. */
export function decodeBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Scarica un file binario e lo restituisce come ArrayBuffer. */
export async function fetchBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Impossibile caricare ${url} (HTTP ${response.status})`);
  return response.arrayBuffer();
}

/** Scarica un JSON. */
export async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Impossibile caricare ${url} (HTTP ${response.status})`);
  return response.json();
}

/** Converte un half-float (IEEE 754 binary16, come uint16) in float32. */
export function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

/** ArrayBuffer di float16 little-endian → Float32Array. */
export function float16ToFloat32(buffer) {
  const half = new Uint16Array(buffer);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = halfToFloat(half[i]);
  return out;
}
