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
  wMel: { x: -31, y: -16, w: 20, h: 7 },
  wEnc: { x: -11.5, y: -16, w: 9, h: 7 },
  wDec: { x: 1.5, y: -16, w: 9, h: 7 },
  wText: { x: 17.5, y: -16, w: 15, h: 5 },
});

/** Spostamento dell'intera scena, così il centro del disegno coincide con l'origine. */
export const SCENE_OFFSET = { x: 2, y: 0 };

/** Soglie del LOD (distanza camera → punto guardato). */
export const LOD_DISTANCES = { detail: 82, blocks: 165 };

/**
 * Rettangoli da inquadrare (coordinate della scena, titoli compresi). Le linee e le etichette
 * non entrano nel bounding box calcolato da CameraDirector: per questo li dichiariamo qui.
 * [x0, x1, y0, y1]
 */
export const FOCUS = Object.freeze({
  audio: [-57.4, -42.6, -3.9, 4.5],
  "audio.lens": [-57.4, -42.6, -3.9, 10.6],
  encoder: [-40.6, -23.4, -4.6, 5.4],
  rvq: [-21.9, -6.1, -5.6, 6.2],
  transformer: [-2.2, 8.2, -5, 6],
  delay: [11.2, 24.8, -4.3, 6.2],
  delayCore: [-2.2, 24.8, -5, 6.2],
  genCore: [-2.2, 8.2, -5, 15.8],
  conditioning: [-17, 8, 10.8, 15.8],
  t5: [-17, 8.2, -5, 15.8],
  decoder: [27.8, 36.2, -3.6, 4.3],
  decoderOut: [27.8, 53.4, -3.6, 4.4],
  output: [38.6, 53.4, -3.4, 4.4],
  wMel: [-41.4, -20.6, -20.4, -10.6],
  wEnc: [-16, 6, -20.8, -10.4],
  wDec: [-3, 25.4, -20.8, -10.4],
  whisperOut: [-3, 25.4, -20.8, -10.4],
  wText: [9.6, 25.6, -19.2, -12.2],
  whisperAll: [-41.4, 25.4, -21, -10.4],
  overview: [-57.6, 53.6, -21, 16],
});
