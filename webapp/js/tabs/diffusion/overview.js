/**
 * Livelli astratti del LOD:
 *   "Blocchi"    → un rettangolo per componente (Text encoder, Rumore, U-Net o DiT, Scheduler,
 *                  VAE, Immagine), con titolo e descrizione di cosa fa
 *   "Panoramica" → Testo → Denoiser (ripetuto T volte) → Immagine, e lo stesso per il video
 *
 * Leggibilità: questi livelli si vedono da lontano, quindi il testo deve essere grande rispetto
 * alla scena. makeBlock disegna il testo con un'altezza fissa in unità di scena (titolo ≈ 0,42):
 * costruiamo blocchi piccoli e li INGRANDIAMO con `scale` (S), così titolo e descrizione crescono
 * insieme al rettangolo.
 */
import * as THREE from "three";
import { makeArrow, makeBlock } from "../../core/diagram.js";
import { placeLabel } from "./shapes.js";

export const VIDEO_Y = -66;
const MACRO_SCALE = 1.5;

/** Blocco di larghezza w e altezza h (unità di scena) con il testo ingrandito di S. */
function block(group, spec, { x, y, w, h, S }) {
  const b = makeBlock({ ...spec, width: w / S, height: h / S });
  b.scale.setScalar(S);
  b.position.set(x, y, 0);
  group.add(b);
  return b;
}

/** Polilinea con freccia solo sull'ultimo tratto. points: [[x, y], …] */
function arrow(group, points, { color = 0x67e8f9, radius = 0.35, head = 2.2 } = {}) {
  for (let i = 1; i < points.length; i++) {
    const [x1, y1] = points[i - 1];
    const [x2, y2] = points[i];
    const last = i === points.length - 1;
    group.add(makeArrow(new THREE.Vector3(x1, y1, 0.3), new THREE.Vector3(x2, y2, 0.3), { color, radius, headLength: last ? head : 0.001 }));
  }
}

/** Livello "Blocchi". Restituisce anche i due blocchi alternativi del denoiser. */
export function buildBlocks() {
  const group = new THREE.Group();
  const S = 6.5;
  const H = 15;
  block(group, { title: "Text encoder", subtitle: "tokenizer + CLIP", text: "Spezza il prompt in token (max 77) e li trasforma in vettori di significato.", color: "#c4b5fd" }, { x: -66, y: 0, w: 24, h: H, S });
  block(group, { title: "Rumore", subtitle: "latente 4×64×64", text: "Si parte da rumore casuale: lo sceglie il seed.", color: "#cbd5e1" }, { x: -38, y: 0, w: 20, h: H, S });
  const unet = block(group, { title: "U-Net", subtitle: "il denoiser", text: "Predice il rumore nel latente. Riceve il passo t e, con la cross-attention, il testo.", color: "#fcd34d" }, { x: 0, y: 0, w: 40, h: H, S });
  const dit = block(group, { title: "Diffusion Transformer", subtitle: "il denoiser", text: "Patch del latente come token in una pila di blocchi transformer; t entra con adaLN.", color: "#fcd34d" }, { x: 0, y: 0, w: 40, h: H, S });
  dit.visible = false;
  block(group, { title: "Scheduler", subtitle: "DDIM, Euler…", text: "Toglie il rumore previsto e passa al t successivo. Ripete 20-50 volte.", color: "#a5f3fc" }, { x: 0, y: -27, w: 40, h: 14, S });
  block(group, { title: "VAE decoder", subtitle: "latente → pixel", text: "Da 64×64×4 a 512×512 a colori.", color: "#7dd3fc" }, { x: 38, y: 0, w: 20, h: H, S });
  block(group, { title: "Immagine", subtitle: "512 × 512", text: "Tutti i pixel raffinati insieme.", color: "#e2e8f0" }, { x: 64, y: 0, w: 20, h: H, S });
  arrow(group, [[-53.6, 0], [-48.4, 0]], { color: 0xa78bfa });
  arrow(group, [[-27.6, 0], [-20.4, 0]]);
  arrow(group, [[-66, 7.8], [-66, 12], [-12, 12], [-12, 8.2]], { color: 0xa78bfa });
  arrow(group, [[0, -7.8], [0, -19.6]]);
  arrow(group, [[-20.4, -27], [-38, -27], [-38, -8.2]]);
  arrow(group, [[20.4, -27], [38, -27], [38, -8.2]], { color: 0x7dd3fc });
  arrow(group, [[48.4, 0], [53.6, 0]]);
  placeLabel(group, "ripeti con t più piccolo", -29, -24.2, 0.4, { size: 2, color: "#a5f3fc" });
  placeLabel(group, "z_0 alla fine", 29, -24.2, 0.4, { size: 2, color: "#7dd3fc" });
  placeLabel(group, "testo (cross-attention)", -39, 14.6, 0.4, { size: 2, color: "#c4b5fd" });

  // video (sotto), nell'ordine della GENERAZIONE: il rumore nasce già nel latente spazio-temporale,
  // il decoder del VAE 3D torna ai frame solo alla fine (l'encoder serve nel training)
  const vy = VIDEO_Y;
  const V = { h: 16, S };
  block(group, { title: "Rumore 3D", subtitle: "nel latente", text: "Un cubo di rumore: tempo × altezza × larghezza.", color: "#cbd5e1" }, { x: -66, y: vy, w: 22, ...V });
  block(group, { title: "Patch", subtitle: "spazio-temporali", text: "Cubetti del latente che diventano token in fila.", color: "#22d3ee" }, { x: -39, y: vy, w: 22, ...V });
  block(group, { title: "Transformer", subtitle: "attention spazio + tempo", text: "Raffina tutti i frame insieme, per T passi.", color: "#fcd34d" }, { x: -11, y: vy, w: 26, ...V });
  block(group, { title: "VAE 3D decoder", subtitle: "latente → frame", text: "HunyuanVideo: 4× tempo, 8× spazio, 16 canali.", color: "#7dd3fc" }, { x: 17, y: vy, w: 22, ...V });
  block(group, { title: "Video", subtitle: "frame coerenti", text: "Tutti i frame nascono insieme.", color: "#e2e8f0" }, { x: 44, y: vy, w: 22, ...V });
  for (const [a, b] of [[-54.6, -50.4], [-27.6, -24.4], [2.4, 5.6], [28.4, 32.6]]) arrow(group, [[a, vy], [b, vy]]);
  arrow(group, [[-11, vy + 13.5], [-11, vy + 8.6]], { color: 0xa78bfa });
  placeLabel(group, "testo (T5, CLIP…)", -11, vy + 15.4, 0.4, { size: 2, color: "#c4b5fd" });
  placeLabel(group, "Nel training l'encoder del VAE 3D comprime i video veri: il transformer impara in quello spazio.", -11, vy - 11.5, 0.4, { size: 1.7, color: "#94a3b8" });
  placeLabel(group, "Immagini", -66, 15, 0.4, { size: 3 });
  placeLabel(group, "Video", -66, vy + 12, 0.4, { size: 3 });
  return { group, unet, dit };
}

/** Livello "Panoramica": tre rettangoli grandi e il ciclo ripetuto T volte, per immagine e video. */
export function buildMacro() {
  const group = new THREE.Group();
  const S = 15;
  const big = (title, subtitle, text, x, y, color, w = 56) => block(group, { title, subtitle, text, color }, { x, y, w, h: 36, S });
  big("Testo", "prompt → vettori", "Un transformer legge la frase.", -84, 0, "#c4b5fd");
  big("Denoiser × T", "es. 50 passi", "Dal rumore all'immagine, un po' alla volta.", 0, 0, "#fcd34d", 64);
  big("Immagine", "VAE → pixel", "Il risultato finale.", 84, 0, "#e2e8f0");
  arrow(group, [[-55.5, 0], [-32.5, 0]], { color: 0xa78bfa, radius: 0.9, head: 5 });
  arrow(group, [[32.5, 0], [55.5, 0]], { radius: 0.9, head: 5 });
  // anello: il denoiser si ripete
  const curve = new THREE.EllipseCurve(0, -18.5, 20, 8, Math.PI * 0.08, Math.PI * 0.92, true);
  const pts = curve.getPoints(40).map((p) => new THREE.Vector3(p.x, p.y, 0.5));
  group.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 60, 0.8, 8), new THREE.MeshBasicMaterial({ color: 0x67e8f9 })));
  placeLabel(group, "× T passi", 0, -32, 0.6, { size: 6.5, color: "#a5f3fc" });

  const vy = -76;
  big("Testo", "prompt → vettori", "Lo stesso testo guida ogni frame.", -84, vy, "#c4b5fd");
  big("Denoiser video × T", "tutti i frame insieme", "Un cubo di latenti spazio-temporali.", 0, vy, "#fcd34d", 64);
  big("Video", "VAE 3D → frame", "Frame coerenti tra loro.", 84, vy, "#e2e8f0");
  arrow(group, [[-55.5, vy], [-32.5, vy]], { color: 0xa78bfa, radius: 0.9, head: 5 });
  arrow(group, [[32.5, vy], [55.5, vy]], { radius: 0.9, head: 5 });
  placeLabel(group, "Immagini", -84, 25, 0.6, { size: 6.5 });
  placeLabel(group, "Video", -84, vy + 25, 0.6, { size: 6.5 });
  // La Panoramica compare oltre la soglia del livello Blocchi: ingrandita così, a quella distanza
  // riempie lo schermo invece di occuparne metà.
  group.scale.setScalar(MACRO_SCALE);
  return { group };
}
