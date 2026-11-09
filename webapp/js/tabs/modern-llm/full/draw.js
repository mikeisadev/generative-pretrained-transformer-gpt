/**
 * Mattoni grafici della vista completa, sopra quelli del tab (shapes.js) e del nucleo (diagram.js).
 * Regole: ogni oggetto ha materiali propri (il LOD sfuma l'opacità materiale per materiale), le
 * texture di testo si disegnano una volta sola, linee molte in un solo LineSegments, e `disposeTree`
 * libera geometrie, materiali e texture quando si cambia modello o si ricostruisce un livello.
 */
import * as THREE from "three";
import { arrow, bigBlock, Plate } from "../shapes.js";

export { arrow, bigBlock };

/** Larghezza massima del canvas di un pannello di testo: oltre, la memoria cresce senza guadagno visibile. */
const MAX_CANVAS = 1024;
/** Limite assoluto (le GPU accettano texture fino a 4096-8192 px): un pannello largo usa meno pixel per unità. */
const HARD_CANVAS = 2048;

/**
 * Pannello di testo (righe colorate) con una risoluzione adatta alla sua misura: piccolo e vicino
 * = più pixel per unità, grande e lontano = meno (il canvas resta sotto i 1024 px di larghezza).
 * @param {object} o {w, h, lines: string | {text, color?, bold?, scale?}[], size, align, mono, background, border, px}
 */
export function textPlate({ w, h, lines, size = 1, align = "left", mono = false, background = null, border = null, color, px }) {
  const density = Math.min(px ?? Math.max(6, Math.min(52, MAX_CANVAS / w)), HARD_CANVAS / w);
  const plate = new Plate({ w, h, px: density, size, align, mono, background, border, color });
  plate.set(lines);
  return plate;
}

/**
 * Testo a capo per un pannello largo `width` unità con caratteri alti `size`: stima della larghezza
 * media di un carattere (0,52 em), poi Plate rimpicciolisce da sé la riga che non entrasse.
 */
export function wrapText(text, width, size) {
  const max = Math.max(8, Math.floor(width / (size * 0.52)));
  const lines = [];
  let line = "";
  for (const word of String(text).split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (next.length > max && line) {
      lines.push(line);
      line = word;
    } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

/** Rettangolo di contorno (LineLoop) nel piano z; `dashed` per ciò che è riusato o assente. */
export function outline(parent, { x, y, w, h }, { color = "#334155", z = 0, dashed = false, opacity = 1 } = {}) {
  const pts = [new THREE.Vector3(x - w / 2, y - h / 2, z), new THREE.Vector3(x + w / 2, y - h / 2, z), new THREE.Vector3(x + w / 2, y + h / 2, z), new THREE.Vector3(x - w / 2, y + h / 2, z)];
  const geometry = new THREE.BufferGeometry().setFromPoints(pts);
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: Math.max(0.4, Math.min(w, h) / 10), gapSize: Math.max(0.3, Math.min(w, h) / 14), transparent: opacity < 1, opacity })
    : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  const loop = new THREE.LineLoop(geometry, material);
  if (dashed) loop.computeLineDistances();
  loop.userData.noBounds = true;
  parent.add(loop);
  return loop;
}

/** Pannello pieno (sfondo di un riquadro), dietro al testo: non entra nei confini della camera. */
export function backPlane(parent, { x, y, w, h }, { color = "#0b1220", opacity = 0.7, z = -0.4 } = {}) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
  mesh.position.set(x, y, z);
  mesh.userData.isFrame = true;
  parent.add(mesh);
  return mesh;
}

/** Tanti segmenti di un colore in un solo LineSegments: [[x1, y1, x2, y2], …] alla quota z. */
export function segments(parent, list, { color = "#475569", z = 0, opacity = 1 } = {}) {
  const positions = new Float32Array(list.length * 6);
  list.forEach(([x1, y1, x2, y2], i) => positions.set([x1, y1, z, x2, y2, z], i * 6));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
  lines.userData.noBounds = true;
  parent.add(lines);
  return lines;
}

/** Parentesi orizzontale sopra un gruppo di blocchi: linea con due stanghette verso il basso. */
export function bracket(parent, x0, x1, y, { color = "#94a3b8", tick = 2.2, z = 0.2 } = {}) {
  return segments(parent, [[x0, y, x1, y], [x0, y, x0, y - tick], [x1, y, x1, y - tick]], { color, z });
}

/** Scatola piatta colorata (barre, celle singole) con materiale proprio. */
export function flatBox(parent, { x, y, w, h, d = 0.3 }, { color = "#1e293b", emissive = "#000000", intensity = 0, opacity = 1 } = {}) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: intensity, roughness: 0.6, transparent: opacity < 1, opacity }),
  );
  mesh.position.set(x, y, 0);
  parent.add(mesh);
  return mesh;
}

/**
 * Segnaposto numerato di un meccanismo della vista d'insieme: cerchio con il numero e nome breve.
 * `status`: "present" (pieno, colore della regione), "absent" (grigio tratteggiato: qui vivrebbe),
 * "study" (anello viola: non è un pezzo del modello). Sprite sempre rivolto alla camera.
 */
export function markerSprite(number, label, color, status = "present", height = 4) {
  const font = 64;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  ctx.font = `600 ${font * 0.62}px Inter, system-ui, sans-serif`;
  const textW = ctx.measureText(label).width;
  const r = font * 0.62;
  canvas.width = Math.ceil(2 * r + 16 + textW + 26);
  canvas.height = Math.ceil(2 * r + 12);
  const cx = r + 6;
  const cy = canvas.height / 2;
  const absent = status === "absent";
  const tint = absent ? "#64748b" : status === "study" ? "#c084fc" : color;
  // pillola di fondo: il nome resta leggibile anche sopra un blocco colorato
  ctx.fillStyle = "rgba(11, 18, 32, 0.9)";
  ctx.beginPath();
  ctx.roundRect(2, 4, canvas.width - 4, canvas.height - 8, (canvas.height - 8) / 2);
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = absent ? "#475569" : tint;
  if (absent) ctx.setLineDash([8, 6]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(cx, cy, r - 4, 0, Math.PI * 2);
  ctx.fillStyle = absent ? "#1e293b" : tint;
  ctx.fill();
  if (status === "study") {
    ctx.fillStyle = "#1e1b2e";
    ctx.beginPath();
    ctx.arc(cx, cy, r - 11, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = absent ? "#cbd5e1" : status === "study" ? "#f5f3ff" : "#0b1220";
  ctx.font = `800 ${font * 0.7}px Inter, system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(number), cx, cy + 2);
  ctx.textAlign = "left";
  ctx.font = `600 ${font * 0.62}px Inter, system-ui, sans-serif`;
  ctx.fillStyle = absent ? "#94a3b8" : "#f1f5f9";
  ctx.fillText(label, 2 * r + 14, cy + 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthWrite: false, depthTest: false, transparent: true }));
  sprite.scale.set((height * canvas.width) / canvas.height, height, 1);
  sprite.renderOrder = 20; // sopra blocchi e schede: un segnaposto non deve mai sparire dietro
  sprite.center.set(0, 0.5); // ancorato a sinistra: il cerchio sta sul punto dell'architettura
  return sprite;
}

/**
 * Linee di molti colori in UN solo LineSegments (colori per vertice): contorni di blocchi e schede.
 * Con il rendering software ogni oggetto in più costa una chiamata di disegno a ogni frame.
 *   const lines = new LineBatch(); lines.rect(r, "#38bdf8", 0.5); lines.flush(group);
 */
export class LineBatch {
  constructor() {
    this.positions = [];
    this.colors = [];
  }

  segment(x1, y1, x2, y2, color, z = 0) {
    const c = scratch.set(color);
    this.positions.push(x1, y1, z, x2, y2, z);
    this.colors.push(c.r, c.g, c.b, c.r, c.g, c.b);
  }

  rect({ x, y, w, h }, color, z = 0) {
    const [x0, x1, y0, y1] = [x - w / 2, x + w / 2, y - h / 2, y + h / 2];
    this.segment(x0, y0, x1, y0, color, z);
    this.segment(x1, y0, x1, y1, color, z);
    this.segment(x1, y1, x0, y1, color, z);
    this.segment(x0, y1, x0, y0, color, z);
  }

  flush(parent) {
    if (!this.positions.length) return null;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(this.positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(this.colors, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true }));
    lines.userData.noBounds = true;
    parent.add(lines);
    return lines;
  }
}

/**
 * Tante scatole piatte non illuminate in UN solo InstancedMesh (colore per istanza): barre
 * colorate, sfondi delle lastre fantasma. `opacity` vale per tutte.
 */
export function boxBatch(parent, boxes, { opacity = 1, depth = 0.3 } = {}) {
  if (!boxes.length) return null;
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, depth), new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: opacity < 1, opacity, depthWrite: opacity >= 1 }), boxes.length);
  const o = new THREE.Object3D();
  boxes.forEach((b, i) => {
    o.position.set(b.x, b.y, b.z ?? 0);
    o.scale.set(b.w, b.h, 1);
    o.updateMatrix();
    mesh.setMatrixAt(i, o.matrix);
    mesh.setColorAt(i, scratch.set(b.color));
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.userData.noBounds = true;
  parent.add(mesh);
  return mesh;
}
const scratch = new THREE.Color();

/** Libera geometrie, materiali e texture di un sotto-albero e lo stacca dal genitore. */
export function disposeTree(object) {
  object.traverse((child) => {
    child.geometry?.dispose();
    const materials = Array.isArray(child.material) ? child.material : child.material ? [child.material] : [];
    for (const m of materials) {
      m.map?.dispose();
      m.dispose();
    }
  });
  object.removeFromParent();
}

/** Svuota un gruppo (figli liberati) lasciando il gruppo stesso al suo posto. */
export function clearGroup(group) {
  for (const child of [...group.children]) disposeTree(child);
}

/** Box3 di un rettangolo {x, y, w, h} (spessore piccolo in z): per CameraDirector.focus. */
export function rectBox({ x, y, w, h }, pad = 0, offset = { x: 0, y: 0 }) {
  return new THREE.Box3(new THREE.Vector3(offset.x + x - w / 2 - pad, offset.y + y - h / 2 - pad, -0.5), new THREE.Vector3(offset.x + x + w / 2 + pad, offset.y + y + h / 2 + pad, 0.5));
}
