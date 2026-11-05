/**
 * Posizioni dei componenti nella scena del tab audio. Le usano sia il livello "Dettaglio"
 * sia i livelli "Blocchi" e "Panoramica" del LOD: così un blocco sta esattamente dove sta
 * il componente che riassume.
 *
 *   riga principale (y = 0):  Audio → Encoder → RVQ → Transformer → Delay → Decoder → Uscita
 *   riga del testo  (y = 13): Prompt → T5 ──cross-attention──▶ Transformer
 *   riga di Whisper (y = −15): log-Mel → Encoder → Decoder → Testo
 */
export const SLOT = Object.freeze({
  audio: { x: -50, y: 0, w: 14, h: 6 },
  encoder: { x: -32, y: 0, w: 14, h: 7 },
  rvq: { x: -14, y: 0, w: 14, h: 11 },
  transformer: { x: 3, y: 0, w: 9, h: 9 },
  delay: { x: 18, y: 0, w: 13, h: 10 },
  decoder: { x: 32, y: 0, w: 7, h: 7 },
  output: { x: 46, y: 0, w: 14, h: 6 },
  text: { x: -10, y: 13, w: 13, h: 3 },
  t5: { x: 3, y: 13, w: 9, h: 3 },
  wMel: { x: -30, y: -15, w: 22, h: 5 },
  wEnc: { x: -9, y: -15, w: 9, h: 7 },
  wDec: { x: 5, y: -15, w: 9, h: 7 },
  wText: { x: 24, y: -15, w: 20, h: 3.4 },
});

/** Spostamento dell'intera scena, così il centro del disegno coincide con l'origine. */
export const SCENE_OFFSET = { x: 2, y: 0 };

/** Soglie del LOD (distanza camera → punto guardato). */
export const LOD_DISTANCES = { detail: 52, blocks: 128 };
