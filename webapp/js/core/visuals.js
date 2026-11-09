/**
 * Mattoni grafici: matrici di pesi come heatmap, etichette, nodi di somma.
 *
 * Convenzione: 1 cella = 1 peso, lato di una cella = CELL unità di scena.
 * Una matrice nn.Linear di forma (out, in) è disegnata alta `out` e larga `in`.
 */
import * as THREE from "three";
import { decodeBytes } from "./binary.js";

export const CELL = 0.04;

export const COLORS = {
  background: 0x0d1117,
  learned: 0xf59e0b,
  fixed: 0x94a3b8,
  highlight: 0x38bdf8,
  flow: 0x67e8f9,
  frame: 0x334155,
};

/** Colormap divergente su sfondo scuro: blu (negativo) → scuro (zero) → arancione (positivo). */
function signedColor(v) {
  const m = Math.pow(Math.abs(v), 0.6);
  const base = [22, 27, 34];
  const tip = v >= 0 ? [251, 146, 60] : [59, 130, 246];
  return base.map((b, i) => Math.round(b + (tip[i] - b) * m));
}

/** Colormap dell'attention (0…1): scuro → azzurro, con la radice per far vedere anche i pesi piccoli. */
function attentionColor(a) {
  const s = Math.pow(a, 0.5);
  return [Math.round(20 + 83 * s), Math.round(30 + 202 * s), Math.round(40 + 209 * s)];
}

/** Tabella di 256 colori RGBA (sRGB) da una funzione di colore del byte 0…255. */
function lut(colorOf) {
  const table = new Uint8Array(256 * 4);
  for (let q = 0; q < 256; q++) table.set([...colorOf(q), 255], q * 4);
  return table;
}

/**
 * Le tabelle dei colori delle heatmap, una volta sola per pagina: chi colora celle con gli stessi colori
 * della scena (matrixNumbers.js, le azioni del Transformer) le legge da qui invece di ricavarle
 * disegnando una rampa. Sono esattamente i colori di weightTexture e attentionTexture (che le usano):
 * byte quantizzato q → RGBA in [4q, 4q + 4). Condivise e di sola lettura: non scriverci (un array
 * tipizzato non si può congelare).
 * - DIVERGING_LUT: pesi quantizzati come gpt/export.py (128 = zero, v = (q − 128) / 127);
 * - ATTENTION_LUT: pesi di attention 0…255 (v = q / 255);
 * - MASKED_LUT: i 4 colori RGBA del tratteggio del futuro mascherato, indice (riga + colonna) % 4.
 */
export const DIVERGING_LUT = lut((q) => signedColor((q - 128) / 127));
export const ATTENTION_LUT = lut((q) => attentionColor(q / 255));
export const MASKED_LUT = Uint8Array.from([0, 1, 2, 3].flatMap((k) => maskedColor(k)));

/** Colore del tratteggio grigio del futuro mascherato: (riga + colonna) % 4 → RGBA. */
function maskedColor(k) {
  const stripe = k % 4 < 2 ? 52 : 34;
  return [stripe, stripe + 4, stripe + 10, 255];
}

function textureFromRGBA(rgba, width, height) {
  const tex = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter; // celle nette: si vede ogni singolo peso
  tex.minFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/** Texture di un tensore quantizzato da export.py (128 = zero). Riga 0 in alto. */
export function weightTexture(tensor) {
  const [rows, cols] = tensor.shape;
  const bytes = decodeBytes(tensor.data);
  const rgba = new Uint8Array(rows * cols * 4);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const q = 4 * bytes[r * cols + c];
      rgba.set(DIVERGING_LUT.subarray(q, q + 4), ((rows - 1 - r) * cols + c) * 4); // DataTexture parte dal basso
    }
  }
  return textureFromRGBA(rgba, cols, rows);
}

/** Pattern di attention (T×T, valori 0..255). Sopra la diagonale: tratteggio grigio = futuro mascherato. */
export function attentionTexture(b64, T) {
  const bytes = decodeBytes(b64);
  const rgba = new Uint8Array(T * T * 4);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const o = ((T - 1 - i) * T + j) * 4;
      const k = j > i ? 4 * ((i + j) % 4) : 4 * bytes[i * T + j];
      rgba.set((j > i ? MASKED_LUT : ATTENTION_LUT).subarray(k, k + 4), o);
    }
  }
  return textureFromRGBA(rgba, T, T);
}

/** Piano rettangolare con texture, dimensioni proporzionali alla forma. */
export function matrixPlane(texture, rows, cols, { opacity = 1 } = {}) {
  const geo = new THREE.PlaneGeometry(cols * CELL, rows * CELL);
  const mat = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, transparent: opacity < 1, opacity });
  const mesh = new THREE.Mesh(geo, mat);
  const frame = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: COLORS.frame }),
  );
  mesh.add(frame);
  return mesh;
}

/** Etichetta di testo sempre rivolta verso la camera. */
export function label(text, { size = 0.32, color = "#e2e8f0", sub = null, subColor = "#94a3b8" } = {}) {
  const lines = sub ? [text, sub] : [text];
  const fontPx = 64;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.font = `600 ${fontPx}px Inter, system-ui, sans-serif`;
  const width = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 24;
  canvas.width = Math.ceil(width);
  canvas.height = Math.ceil(fontPx * 1.25 * lines.length + 12);
  ctx.font = `600 ${fontPx}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  lines.forEach((l, i) => {
    ctx.fillStyle = i === 0 ? color : subColor;
    ctx.font = `${i === 0 ? 600 : 500} ${i === 0 ? fontPx : fontPx * 0.8}px Inter, system-ui, sans-serif`;
    ctx.fillText(l, 12, 6 + i * fontPx * 1.25);
  });
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true }));
  const aspect = canvas.width / canvas.height;
  const h = size * lines.length * 1.1;
  sprite.scale.set(h * aspect, h, 1);
  sprite.renderOrder = 10;
  return sprite;
}

/** Nodo "+" delle somme (embedding e connessioni residue). */
export function addNode(radius = 0.28) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0x1e293b, emissive: 0x0e7490, emissiveIntensity: 0.6, roughness: 0.4 }),
  );
  const plus = label("+", { size: radius * 1.6, color: "#e0f2fe" });
  plus.position.z = radius + 0.01;
  mesh.add(plus);
  return mesh;
}

/** Cubo di un token con il suo carattere. */
export function tokenCube(char, size = 0.3, { color = 0x1f2937, emissive = 0x000000 } = {}) {
  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(size, size, size),
    new THREE.MeshStandardMaterial({ color, emissive, roughness: 0.5 }),
  );
  const shown = char === "\n" ? "↵" : char === " " ? "␣" : char;
  const tag = label(shown, { size: size * 0.7 });
  tag.position.z = size / 2 + 0.02;
  cube.add(tag);
  return cube;
}
