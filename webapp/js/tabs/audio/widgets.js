/**
 * Mattoni 3D del tab audio (sopra quelli del nucleo): colori, forme d'onda come LineSegments,
 * griglie di token come InstancedMesh, pannelli con testo disegnato su canvas, "lastre"
 * per gli strati delle reti.
 *
 * Regola: ogni funzione crea materiali propri (il LOD sfuma l'opacità dei materiali).
 */
import * as THREE from "three";
import { label } from "../../core/visuals.js";

/** Rampa per spettrogrammi su sfondo scuro: blu notte → viola → arancio → giallo chiaro. */
const RAMP = [
  [13, 17, 40],
  [49, 31, 104],
  [128, 38, 120],
  [205, 69, 87],
  [247, 140, 52],
  [252, 230, 140],
];

/** v in 0…1 → colore della rampa (scrive in `out`, nessuna allocazione). */
export function rampColor(v, out) {
  const x = Math.min(1, Math.max(0, v)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = RAMP[i];
  const b = RAMP[i + 1];
  out.r = (a[0] + (b[0] - a[0]) * f) / 255;
  out.g = (a[1] + (b[1] - a[1]) * f) / 255;
  out.b = (a[2] + (b[2] - a[2]) * f) / 255;
  return out;
}

/**
 * Colore di un token: identità, non grandezza. Tinta "a passo d'oro" sul cerchio dei colori:
 * indici vicini hanno tinte lontane, così due token diversi si distinguono anche se adiacenti.
 */
export function tokenColor(id, out = new THREE.Color()) {
  const hue = (id * 0.618034) % 1;
  return out.setHSL(hue, 0.62, 0.56);
}
export const tokenCss = (id) => `#${tokenColor(id).getHexString()}`;

/** Forma d'onda come inviluppo min/max: una linea verticale per colonna (un solo draw call). */
export function waveformMesh(env, { width, height, color = 0x67e8f9, capacity = env.length / 2 } = {}) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(capacity * 6);
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mesh = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true }));
  mesh.userData.setEnvelope = (e) => {
    const columns = e.length / 2;
    for (let c = 0; c < columns; c++) {
      const x = -width / 2 + (c / Math.max(1, columns - 1)) * width;
      positions.set([x, (e[2 * c] * height) / 2, 0, x, (Math.max(e[2 * c + 1], e[2 * c] + 0.004) * height) / 2, 0], c * 6);
    }
    geometry.setDrawRange(0, columns * 2);
    geometry.attributes.position.needsUpdate = true;
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  };
  mesh.userData.setEnvelope(env);
  return mesh;
}

/** Polilinea con capacità fissa (aggiornabile senza allocare): restituisce {line, set(points, count)}. */
export function dynamicLine(capacity, { color = 0xe2e8f0, opacity = 1 } = {}) {
  const positions = new Float32Array(capacity * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setDrawRange(0, 0);
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  line.frustumCulled = false;
  return {
    line,
    positions,
    commit(count) {
      geometry.setDrawRange(0, count);
      geometry.attributes.position.needsUpdate = true;
    },
  };
}

/**
 * Griglia di token q × T come q InstancedMesh (una per codebook): il colore dice QUALE token è,
 * `reveal(n)` mostra solo i primi n frame (animazioni senza ricreare nulla).
 */
export function tokenGrid(q, T, { width, rowHeight = 0.7, gap = 0.12, depth = 0.18 }) {
  const group = new THREE.Group();
  const cell = width / T;
  const rows = [];
  const color = new THREE.Color();
  const matrix = new THREE.Matrix4();
  for (let k = 0; k < q; k++) {
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(cell * 0.86, rowHeight, depth), new THREE.MeshStandardMaterial({ roughness: 0.6, transparent: true }), T);
    const y = ((q - 1) / 2 - k) * (rowHeight + gap);
    for (let t = 0; t < T; t++) {
      matrix.makeTranslation(-width / 2 + (t + 0.5) * cell, y, 0);
      mesh.setMatrixAt(t, matrix);
      mesh.setColorAt(t, color.set(0x334155));
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
    group.add(mesh);
    rows.push(mesh);
  }
  return {
    group,
    rows,
    cell,
    rowY: (k) => ((q - 1) / 2 - k) * (rowHeight + gap),
    frameX: (t) => -width / 2 + (t + 0.5) * cell,
    /** codes: Int32Array q×T (riga = codebook) */
    setCodes(codes, total = T) {
      for (let k = 0; k < q; k++) {
        for (let t = 0; t < T; t++) rows[k].setColorAt(t, tokenColor(codes[k * total + t], color));
        rows[k].instanceColor.needsUpdate = true;
      }
    },
    /** Mostra i primi n frame di ogni riga (perRow: array di n diversi per riga). */
    reveal(n, perRow = null) {
      rows.forEach((mesh, k) => (mesh.count = Math.max(0, Math.min(T, perRow ? perRow[k] : n))));
    },
  };
}

/**
 * Pannello piatto con testo/disegni su canvas, ridisegnabile (es. la "lente" sui numeri dei token).
 * draw(ctx, w, h) disegna; panel.redraw() ridisegna e aggiorna la texture.
 */
export function canvasPanel(width, height, draw, { pxPerUnit = 110 } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.min(2048, Math.round(width * pxPerUnit));
  canvas.height = Math.min(2048, Math.round(height * pxPerUnit));
  const ctx = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  const redraw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    draw(ctx, canvas.width, canvas.height);
    texture.needsUpdate = true;
  };
  redraw();
  mesh.userData.redraw = redraw;
  mesh.userData.dispose = () => texture.dispose();
  return mesh;
}

/** Lastra (strato di rete) con un'etichetta accanto. */
export function slab({ width, height, depth = 1.2, color = 0x1e293b, emissive = 0x0e7490, intensity = 0.25, text = null, labelSize = 0.42, labelSide = "right" }) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: intensity, roughness: 0.5, transparent: true, opacity: 0.95 }),
  );
  if (text) {
    const tag = label(text, { size: labelSize });
    tag.position.set(labelSide === "right" ? width / 2 + 0.3 + tag.scale.x / 2 : 0, labelSide === "right" ? 0 : height / 2 + 0.45, 0);
    mesh.add(tag);
  }
  return mesh;
}

/** Etichetta posizionata (comodità). */
export function placeLabel(parent, text, x, y, z = 0, options = {}) {
  const tag = label(text, options);
  tag.position.set(x, y, z);
  parent.add(tag);
  return tag;
}

/** Libera geometrie, materiali e texture di un sottoalbero. */
export function disposeTree(root) {
  root.traverse((o) => {
    o.geometry?.dispose?.();
    const materials = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
    for (const m of materials) {
      m.map?.dispose?.();
      m.dispose?.();
    }
  });
}
