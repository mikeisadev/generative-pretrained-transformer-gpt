/**
 * Colori delle heatmap, presi dal nucleo (core/visuals.js) invece di essere riscritti qui.
 *
 * visuals.js espone solo `weightTexture` e `attentionTexture` (non le funzioni di colore), quindi
 * le "interroghiamo" una volta sola con dati sintetici e ne leggiamo i pixel: otteniamo tabelle
 * di 256 colori (LUT) identiche a quelle usate dalla scena. Così le texture che aggiorniamo
 * durante le azioni hanno esattamente i colori originali, senza duplicare la colormap.
 */
import { attentionTexture, weightTexture } from "../../../core/visuals.js";

let cache = null;

/** base64 di un Uint8Array (piccolo: solo per costruire le LUT). */
function toBase64(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

/**
 * @returns {{signed: Uint8Array, attention: Uint8Array, masked: Uint8Array}}
 *   signed: 256 × RGBA per i pesi quantizzati (128 = zero, blu negativo, arancione positivo)
 *   attention: 256 × RGBA per i pesi di attention 0..255 (scuro → azzurro)
 *   masked: 4 × RGBA del tratteggio del "futuro" mascherato, indicizzato da (riga + colonna) % 4
 */
export function colormaps() {
  if (cache) return cache;
  const ramp = Uint8Array.from({ length: 256 }, (_, i) => i);

  const w = weightTexture({ shape: [1, 256], data: toBase64(ramp) });
  const signed = new Uint8Array(w.image.data);
  w.dispose();

  // Matrice 256×256 la cui ultima riga contiene 0..255: tutta sotto la diagonale, quindi colorata
  // con la scala dell'attention. La riga 0 (sopra la diagonale dalla colonna 1) dà il tratteggio.
  const T = 256;
  const bytes = new Uint8Array(T * T);
  bytes.set(ramp, (T - 1) * T);
  const a = attentionTexture(toBase64(bytes), T);
  const data = a.image.data;
  // attentionTexture capovolge le righe (DataTexture parte dal basso): la riga i sta in T-1-i.
  const attention = new Uint8Array(data.subarray(0, 256 * 4));
  const row0 = (T - 1) * T * 4;
  const masked = new Uint8Array(16);
  for (let k = 0; k < 4; k++) masked.set(data.subarray(row0 + (4 + k) * 4, row0 + (5 + k) * 4), k * 4);
  a.dispose();

  cache = { signed, attention, masked };
  return cache;
}
