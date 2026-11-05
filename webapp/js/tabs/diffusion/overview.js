/**
 * Livelli astratti del LOD:
 *   "Blocchi"    → un rettangolo per componente, con titolo e descrizione di cosa fa
 *   "Panoramica" → pochi rettangoli grandi e le frecce del flusso
 * Per l'immagine (in alto) e per il video (più in basso, a VIDEO_Y).
 */
import * as THREE from "three";
import { makeArrow, makeBlock } from "../../core/diagram.js";
import { LAYOUT } from "./pipeline.js";
import { placeLabel } from "./shapes.js";

export const VIDEO_Y = -62;

function block(group, spec, x, y) {
  const b = makeBlock({ width: 10, height: 8, ...spec });
  b.position.set(x, y, 0);
  group.add(b);
  return b;
}

function arrow(group, x1, y1, x2, y2, color = 0x67e8f9, radius = 0.16) {
  group.add(makeArrow(new THREE.Vector3(x1, y1, 0.2), new THREE.Vector3(x2, y2, 0.2), { color, radius, headLength: 1.1 }));
}

/** Polilinea con freccia solo sull'ultimo tratto. points: [[x, y], …] */
function elbow(group, points, color = 0x67e8f9, radius = 0.16) {
  for (let i = 1; i < points.length; i++) {
    const [x1, y1] = points[i - 1];
    const [x2, y2] = points[i];
    const last = i === points.length - 1;
    group.add(makeArrow(new THREE.Vector3(x1, y1, 0.2), new THREE.Vector3(x2, y2, 0.2), { color, radius, headLength: last ? 1.1 : 0.001 }));
  }
}

/** Livello "Blocchi". Restituisce anche i due blocchi alternativi del denoiser. */
export function buildBlocks() {
  const group = new THREE.Group();
  const y = 0;
  block(group, { title: "Tokenizer", subtitle: "testo → token", text: "Spezza il prompt in token (al massimo 77 per CLIP) e li trasforma in numeri.", color: "#c4b5fd", width: 9 }, LAYOUT.prompt.x, y);
  block(group, { title: "Text encoder", subtitle: "CLIP · transformer", text: "Trasforma i token in 77 vettori da 768 numeri che descrivono il significato del prompt.", color: "#c4b5fd", width: 10 }, LAYOUT.encoder.x + 1.5, y);
  block(group, { title: "Rumore", subtitle: "latente 4 × 64 × 64", text: "Si parte da rumore gaussiano casuale: il seed decide quale. Il passo t dice quanto rumore c'è.", color: "#cbd5e1", width: 10 }, LAYOUT.latent.x - 1, y);
  const unet = block(group, { title: "U-Net", subtitle: "denoiser · ResNet + attention", text: "Guarda il latente rumoroso e predice il rumore che contiene. Riceve il passo t e, con la cross-attention, il testo. Encoder e decoder sono uniti da skip connection.", color: "#fcd34d", width: 22, height: 11 }, 0, 1.5);
  const dit = block(group, { title: "Diffusion Transformer", subtitle: "denoiser · blocchi transformer", text: "Taglia il latente in patch, le tratta come token e le fa passare in una pila di blocchi transformer. Il passo t entra con adaLN; in SD3 il testo partecipa all'attention.", color: "#fcd34d", width: 22, height: 11 }, 0, 1.5);
  dit.visible = false;
  block(group, { title: "Scheduler", subtitle: "DDIM, Euler…", text: "Usa la previsione per togliere un po' di rumore e passa al t successivo. Ripete 20-50 volte.", color: "#a5f3fc", width: 16, height: 5.2 }, 0, LAYOUT.scheduler.y);
  block(group, { title: "VAE decoder", subtitle: "latente → pixel", text: "Riporta il latente 64×64×4 a un'immagine 512×512 a colori (fattore 8).", color: "#7dd3fc", width: 10 }, LAYOUT.vae.x, y);
  block(group, { title: "Immagine", subtitle: "512 × 512 × 3", text: "Il risultato: ogni pixel è stato raffinato insieme agli altri, non uno alla volta.", color: "#e2e8f0", width: 9 }, LAYOUT.image.x, y);
  arrow(group, LAYOUT.prompt.x + 4.6, y, LAYOUT.encoder.x - 3.6, y);
  elbow(group, [[LAYOUT.encoder.x + 1.5, 4.1], [LAYOUT.encoder.x + 1.5, 9.5], [-11.2, 9.5]], 0xa78bfa);
  arrow(group, LAYOUT.latent.x + 4.1, y, -11.1, y);
  arrow(group, LAYOUT.vae.x + 5.1, y, LAYOUT.image.x - 4.6, y);
  arrow(group, 0, -4.1, 0, LAYOUT.scheduler.y + 2.7);
  elbow(group, [[-8.1, LAYOUT.scheduler.y], [LAYOUT.latent.x - 1, LAYOUT.scheduler.y], [LAYOUT.latent.x - 1, -4.1]]);
  elbow(group, [[8.1, LAYOUT.scheduler.y], [LAYOUT.vae.x, LAYOUT.scheduler.y], [LAYOUT.vae.x, -4.1]], 0x7dd3fc);
  placeLabel(group, "z_0 dopo l'ultimo passo", 21, LAYOUT.scheduler.y + 1, 0.3, { size: 0.6, color: "#7dd3fc" });
  placeLabel(group, "testo (cross-attention)", -27, 10.6, 0.3, { size: 0.6, color: "#c4b5fd" });
  placeLabel(group, "ripeti per ogni passo", -17, LAYOUT.scheduler.y + 1, 0.3, { size: 0.7, color: "#a5f3fc" });

  // video (sotto)
  const vy = VIDEO_Y;
  block(group, { title: "Testo + rumore", subtitle: "latenti spazio-temporali", text: "Il rumore iniziale copre TUTTI i frame insieme: un cubo larghezza × altezza × tempo.", color: "#cbd5e1", width: 12, height: 9 }, -44, vy);
  block(group, { title: "VAE 3D causale", subtitle: "compressione spazio + tempo", text: "In HunyuanVideo: 8 volte nello spazio, 4 volte nel tempo, 16 canali. Causale: un frame dipende solo dal passato.", color: "#7dd3fc", width: 12, height: 9 }, -26, vy);
  block(group, { title: "Patch spazio-temporali", subtitle: "cubetti → token", text: "Il cubo latente è tagliato in cubetti: ognuno diventa un token, come i caratteri del nostro GPT.", color: "#22d3ee", width: 12, height: 9 }, -8, vy);
  block(group, { title: "Transformer video", subtitle: "attention spazio + tempo", text: "Ogni token guarda gli altri dello stesso frame e degli altri frame: i frame vengono raffinati insieme, ripetendo T passi.", color: "#fcd34d", width: 14, height: 9 }, 11, vy);
  block(group, { title: "VAE decoder 3D", subtitle: "latente → frame", text: "Riporta il cubo latente a frame a piena risoluzione.", color: "#7dd3fc", width: 11, height: 9 }, 29.5, vy);
  block(group, { title: "Video", subtitle: "tanti frame coerenti", text: "Gli oggetti non saltano da un frame all'altro perché sono nati insieme.", color: "#e2e8f0", width: 10, height: 9 }, 46, vy);
  for (const [a, b] of [[-38, -32], [-20, -14], [-2, 4], [18, 24], [35, 41]]) arrow(group, a, vy, b, vy);
  return { group, unet, dit };
}

/** Livello "Panoramica": tre rettangoli grandi e il ciclo ripetuto T volte. */
export function buildMacro() {
  const group = new THREE.Group();
  const big = (title, subtitle, text, x, y, color, width = 26) => block(group, { title, subtitle, text, width, height: 15, color }, x, y);
  big("Testo", "prompt → vettori", "Il text encoder (un transformer) trasforma la frase in vettori che guidano il disegno.", -50, 0, "#c4b5fd");
  big("Denoiser × T", "ripetuto T volte (es. 50)", "Parte da puro rumore e a ogni passo ne toglie un po', guidato dal testo.", 0, 0, "#fcd34d", 30);
  big("Immagine", "VAE decoder → pixel", "Dopo l'ultimo passo il latente diventa un'immagine.", 50, 0, "#e2e8f0");
  arrow(group, -37, 0, -15.2, 0, 0xa78bfa, 0.35);
  arrow(group, 15.2, 0, 37, 0, 0x67e8f9, 0.35);
  // anello: il denoiser si ripete
  const curve = new THREE.EllipseCurve(0, -7.5, 9, 4.5, Math.PI * 0.05, Math.PI * 0.95, true);
  const pts = curve.getPoints(40).map((p) => new THREE.Vector3(p.x, p.y, 0.3));
  const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 60, 0.3, 8), new THREE.MeshBasicMaterial({ color: 0x67e8f9 }));
  group.add(tube);
  placeLabel(group, "× T passi", 0, -14.2, 0.4, { size: 1.6, color: "#a5f3fc" });

  big("Testo", "prompt → vettori", "Lo stesso text encoder guida tutti i frame.", -50, VIDEO_Y, "#c4b5fd");
  big("Denoiser video × T", "tutti i frame insieme", "Raffina un cubo di latenti spazio-temporali: i frame nascono insieme, non uno alla volta.", 0, VIDEO_Y, "#fcd34d", 30);
  big("Video", "VAE decoder 3D → frame", "Frame coerenti tra loro.", 50, VIDEO_Y, "#e2e8f0");
  arrow(group, -37, VIDEO_Y, -15.2, VIDEO_Y, 0xa78bfa, 0.35);
  arrow(group, 15.2, VIDEO_Y, 37, VIDEO_Y, 0x67e8f9, 0.35);
  placeLabel(group, "Immagini", -50, 11, 0, { size: 2.2, color: "#e2e8f0" });
  placeLabel(group, "Video", -50, VIDEO_Y + 11, 0, { size: 2.2, color: "#e2e8f0" });
  return { group };
}
