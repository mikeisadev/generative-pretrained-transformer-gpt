/**
 * Regione "Multimodale" (schema di LLaVA, Liu et al. 2023): immagine → patch → encoder visivo
 * (CLIP ViT-L/14) → proiettore (lineare in LLaVA, MLP in LLaVA-1.5) → token visivi che entrano
 * nel modello insieme ai token del testo. L'immagine è disegnata nel browser.
 */
import * as THREE from "three";
import { arrow, card, Cells, ease, PALETTE, phase, Plate, TokenStrip } from "./shapes.js";

const GRID = 4;
const IMG = 6.4;
const IMG_X = -13;
const IMG_Y = 4.2;
const ENC_X = -4.4;
const PROJ_X = 3.4;
const LLM_X = 12.6;
const ROW_Y = -4.6;

/** Piccola scena (cielo, sole, prato, casa) su canvas: niente immagini esterne. */
function sceneCanvas() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = "#7dd3fc";
  g.fillRect(0, 0, 128, 128);
  g.fillStyle = "#4ade80";
  g.fillRect(0, 84, 128, 44);
  g.fillStyle = "#fde047";
  g.beginPath();
  g.arc(100, 26, 14, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = "#f87171";
  g.fillRect(26, 62, 40, 32);
  g.fillStyle = "#7c2d12";
  g.beginPath();
  g.moveTo(20, 64);
  g.lineTo(46, 40);
  g.lineTo(72, 64);
  g.fill();
  return c;
}

export function buildMulti(group) {
  const canvas = sceneCanvas();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const image = new THREE.Mesh(new THREE.PlaneGeometry(IMG, IMG), new THREE.MeshBasicMaterial({ map: texture }));
  image.position.set(IMG_X, IMG_Y, 0);
  group.add(image);
  // colore medio di ogni patch (letto una volta sola dal canvas)
  const data = canvas.getContext("2d").getImageData(0, 0, 128, 128).data;
  const patchColor = [];
  const s = 128 / GRID;
  for (let r = 0; r < GRID; r++)
    for (let c = 0; c < GRID; c++) {
      let R = 0, G = 0, B = 0;
      for (let y = r * s; y < (r + 1) * s; y++)
        for (let x = c * s; x < (c + 1) * s; x++) {
          const o = (y * 128 + x) * 4;
          R += data[o];
          G += data[o + 1];
          B += data[o + 2];
        }
      const n = s * s;
      patchColor.push(new THREE.Color(`rgb(${Math.round(R / n)},${Math.round(G / n)},${Math.round(B / n)})`));
    }

  const encoder = card(group, { x: ENC_X, y: IMG_Y, w: 6.2, h: 5, title: "Encoder visivo", sub: "ViT (es. CLIP ViT-L/14)", color: "#fda4af", size: 0.46 });
  const projector = card(group, { x: PROJ_X, y: IMG_Y, w: 5.6, h: 3.4, title: "Proiettore", sub: "lineare o MLP", color: "#fda4af", size: 0.46 });
  const llm = card(group, { x: LLM_X, y: IMG_Y, w: 6, h: 5, title: "LLM", sub: "legge token visivi + testo", color: PALETTE.modern, size: 0.5 });
  arrow(group, [[IMG_X + IMG / 2 + 0.2, IMG_Y], [ENC_X - 3.2, IMG_Y]], { color: 0xfda4af });
  arrow(group, [[ENC_X + 3.2, IMG_Y], [PROJ_X - 2.9, IMG_Y]], { color: 0xfda4af });
  arrow(group, [[LLM_X, ROW_Y + 1.2], [LLM_X, IMG_Y - 2.6]], { color: 0x38bdf8 });

  const patches = new Cells(GRID * GRID, { w: 1.4, h: 1.4, d: 0.25 });
  group.add(patches.mesh);
  for (let i = 0; i < GRID * GRID; i++) patches.color(i, patchColor[i]);
  const text = new TokenStrip({ w: 13, h: 2, font: 0.78, mono: false, gap: 0.15 }).place(group, 6.9, ROW_Y);
  text.set(["Cosa", "c'è", "nella", "foto", "?"].map((t) => ({ text: t, fill: "#1e3a8a", stroke: "#93c5fd" })));
  const caption = new Plate({ w: 34, h: 3.2, size: 0.57, align: "center" }).place(group, 0, -9.4);

  const patchHex = patchColor.map((c) => `#${c.getHexString()}`);
  const CAPTIONS = [
    [{ text: "Immagine → patch → vettori → token visivi accanto ai token del testo.", bold: true }, { text: "In LLaVA il proiettore è una matrice (in LLaVA-1.5 un piccolo MLP).", color: PALETTE.muted }],
    [{ text: "1. L'immagine viene tagliata in patch (qui 4 × 4; CLIP ViT-L/14 usa patch di 14 pixel).", bold: true }],
    [{ text: "2. L'encoder visivo trasforma ogni patch in un vettore; il proiettore lo porta", bold: true }, { text: "nello spazio degli embedding del modello di linguaggio." }],
    [{ text: "3. I vettori proiettati sono «token visivi»: si mettono in fila prima del testo.", bold: true }],
    [{ text: "4. Il modello legge tutto con la stessa attention e risponde a parole.", bold: true }],
  ];
  const start = (i) => [IMG_X - IMG / 2 + (i % GRID + 0.5) * (IMG / GRID), IMG_Y + IMG / 2 - (Math.floor(i / GRID) + 0.5) * (IMG / GRID)];
  const end = (i) => [-15.6 + i * 0.98, ROW_Y];

  const api = {
    /**
     * t in 0..1: 0-0.25 patch che si staccano, 0.25-0.55 attraverso encoder e proiettore,
     * 0.55-0.85 in fila come token, 0.85-1 il modello li legge.
     */
    set(t) {
      for (let i = 0; i < GRID * GRID; i++) {
        const [sx, sy] = start(i);
        const [ex, ey] = end(i);
        const a = ease(phase(t, 0.05 + i * 0.008, 0.25));
        const b = ease(phase(t, 0.25, 0.55));
        const c = ease(phase(t, 0.55, 0.85));
        let x = sx;
        let y = sy;
        let z = 0.25 + a * 0.6;
        if (b > 0) {
          x = ENC_X + (PROJ_X - ENC_X) * b;
          y = IMG_Y + (Math.floor(i / GRID) - 1.5) * 0.6 * (1 - b);
        }
        if (c > 0) {
          x = PROJ_X + (ex - PROJ_X) * c;
          y = IMG_Y + (ey - IMG_Y) * c;
        }
        if (t <= 0) z = -1; // nascoste dietro l'immagine
        const scale = b > 0 && c === 0 ? 0.55 : 0.7 + 0.3 * (1 - a);
        patches.place(i, x, y, z, t <= 0 ? 0 : scale * 0.72, t <= 0 ? 0 : scale, 1);
        // nell'encoder il colore "si astrae": diventa il colore dei vettori
        patches.mix(i, patchHex[i], "#fb7185", b * (1 - c * 0.2));
      }
      encoder.set(t > 0.25 && t < 0.45 ? 0.9 : 0);
      projector.set(t > 0.4 && t < 0.6 ? 0.9 : 0);
      llm.set(t > 0.85 ? 0.9 : 0);
      caption.set(CAPTIONS[t <= 0 ? 0 : t < 0.25 ? 1 : t < 0.55 ? 2 : t < 0.85 ? 3 : 4]);
    },
  };
  api.set(0);
  return api;
}
