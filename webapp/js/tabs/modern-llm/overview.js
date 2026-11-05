/**
 * Livelli astratti del LOD.
 *   "Blocchi"    → un rettangolo per meccanismo, nello stesso posto della sua regione di dettaglio
 *   "Panoramica" → Dati → Pre-training → Post-training → Modello → Inferenza (ragionamento, tool)
 * I testi sono ingranditi (bigBlock con fattore S) perché questi livelli si guardano da lontano.
 */
import * as THREE from "three";
import { label } from "../../core/visuals.js";
import { REGION_H, REGION_W, REGIONS, ROWS, regionCenter } from "./regions.js";
import { arrow, bigBlock, PALETTE } from "./shapes.js";

/** Titoli delle righe sopra la prima regione di ogni riga (in ogni livello che li usa). */
export function rowTitles(group, size = 1.6) {
  ROWS.forEach((row, r) => {
    const sprite = label(row.title.toUpperCase(), { size, color: row.color });
    const c = regionCenter(REGIONS.find((g) => g.row === r && g.col === 0));
    sprite.position.set(c.x - REGION_W / 2 + sprite.scale.x / 2, c.y + REGION_H / 2 + size * 0.95, 0);
    group.add(sprite);
  });
}

export function buildBlocks() {
  const group = new THREE.Group();
  for (const region of REGIONS) {
    const c = regionCenter(region);
    bigBlock(group, { title: region.title, subtitle: region.sub, text: region.block, color: region.color }, { x: c.x, y: c.y, w: REGION_W - 2, h: REGION_H - 2, S: 8 });
  }
  rowTitles(group, 2.2);
  return group;
}

export const MACRO_STAGES = [
  { id: "data", title: "Dati", subtitle: "testo e codice", text: "Trilioni di token ripuliti e tokenizzati (BPE a byte)." },
  { id: "pre", title: "Pre-training", subtitle: "prossimo token", text: "Il modello impara a continuare il testo. Leggi di scala: più N e più D." },
  { id: "post", title: "Post-training", subtitle: "SFT, preferenze", text: "Segue istruzioni, preferisce risposte utili; RL su compiti verificabili." },
  { id: "model", title: "Modello", subtitle: "blocchi decoder", text: "RMSNorm, GQA + RoPE, SwiGLU o MoE; pesi aperti o chiusi." },
  { id: "infer", title: "Inferenza", subtitle: "token per token", text: "Genera token per token; FlashAttention, decodifica speculativa." },
];

const MACRO_Y = 8;
const MACRO_W = 50;
const MACRO_STEP = 64;
export const macroX = (i) => (i - 2) * MACRO_STEP;

export function buildMacro() {
  const group = new THREE.Group();
  const S = 14.5;
  MACRO_STAGES.forEach((s, i) => bigBlock(group, { ...s, color: i === 3 ? PALETTE.modern : i < 3 ? PALETTE.ok : "#22d3ee" }, { x: macroX(i), y: MACRO_Y, w: MACRO_W, h: 40, S }));
  for (let i = 0; i < 4; i++) arrow(group, [[macroX(i) + MACRO_W / 2 + 0.8, MACRO_Y], [macroX(i + 1) - MACRO_W / 2 - 0.8, MACRO_Y]], { color: 0x67e8f9, radius: 0.7, head: 4.5, z: 0.5 });
  // sotto "Modello" e "Inferenza": ragionamento e strumenti, usati al momento della risposta
  const subY = -44;
  const reasonX = macroX(3) + 8;
  const toolX = macroX(4) + 8;
  bigBlock(group, { title: "Ragionamento", subtitle: "più calcolo", text: "Catena di pensiero, più tentativi, verifica.", color: "#a3e635" }, { x: reasonX, y: subY, w: 50, h: 30, S: 12 });
  bigBlock(group, { title: "Tool e agenti", subtitle: "chiamate a strumenti", text: "Il modello chiede, il programma esegue, il risultato torna.", color: PALETTE.tool }, { x: toolX, y: subY, w: 50, h: 30, S: 12 });
  arrow(group, [[macroX(4) - 8, MACRO_Y - 20.5], [reasonX + 6, subY + 15.5]], { color: 0xa3e635, radius: 0.6, head: 4 });
  arrow(group, [[macroX(4) + 4, MACRO_Y - 20.5], [toolX, subY + 15.5]], { color: 0x34d399, radius: 0.6, head: 4 });
  // ritorno: le risposte verificate diventano segnale di addestramento (RL con ricompense verificabili)
  arrow(group, [[reasonX - 25.5, subY], [macroX(2), subY], [macroX(2), MACRO_Y - 20.5]], { color: 0xfacc15, radius: 0.55, head: 4 });
  const back = label("risposte verificate → ricompensa (RL)", { size: 4.4, color: "#fde68a" });
  back.position.set(macroX(2) + 3 + back.scale.x / 2, subY + 3.6, 0.5);
  group.add(back);
  const title = label("Dalla raccolta dei dati alla risposta", { size: 7, color: "#e2e8f0" });
  title.position.set(0, MACRO_Y + 30, 0);
  group.add(title);
  return group;
}

/** Riquadro del livello Panoramica (per inquadrarlo). */
export function macroBox() {
  return new THREE.Box3(new THREE.Vector3(macroX(0) - MACRO_W / 2 - 4, -62, -1), new THREE.Vector3(macroX(4) + MACRO_W / 2 + 10, MACRO_Y + 38, 1));
}
