/**
 * Colori delle heatmap, presi dal nucleo (core/visuals.js) invece di essere riscritti qui: sono le
 * stesse tabelle che usano weightTexture e attentionTexture, quindi le texture che aggiorniamo durante
 * le azioni hanno esattamente i colori originali, senza duplicare la colormap.
 */
import { ATTENTION_LUT, DIVERGING_LUT, MASKED_LUT } from "../../../core/visuals.js";

/**
 * @returns {{signed: Uint8Array, attention: Uint8Array, masked: Uint8Array}} tabelle di sola lettura
 *   signed: 256 × RGBA per i pesi quantizzati (128 = zero, blu negativo, arancione positivo)
 *   attention: 256 × RGBA per i pesi di attention 0..255 (scuro → azzurro)
 *   masked: 4 × RGBA del tratteggio del "futuro" mascherato, indicizzato da (riga + colonna) % 4
 */
export function colormaps() {
  return { signed: DIVERGING_LUT, attention: ATTENTION_LUT, masked: MASKED_LUT };
}
