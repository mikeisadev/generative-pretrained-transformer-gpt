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
      const v = (bytes[r * cols + c] - 128) / 127;
      const [R, G, B] = signedColor(v);
      const o = ((rows - 1 - r) * cols + c) * 4; // DataTexture parte dal basso
      rgba.set([R, G, B, 255], o);
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
      if (j > i) {
        const stripe = (i + j) % 4 < 2 ? 52 : 34;
        rgba.set([stripe, stripe + 4, stripe + 10, 255], o);
      } else {
        const a = Math.pow(bytes[i * T + j] / 255, 0.5);
        rgba.set([Math.round(20 + 83 * a), Math.round(30 + 202 * a), Math.round(40 + 209 * a), 255], o);
      }
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
