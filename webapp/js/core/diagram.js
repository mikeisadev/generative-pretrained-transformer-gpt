/**
 * Mattoni per diagrammi a blocchi 3D: rettangoli con titolo e descrizione, frecce,
 * flussi animati. Usati dai livelli "Blocchi" e "Panoramica" del LOD e dai tab
 * concettuali (RAG, diffusione, audio, LLM moderni).
 */
import * as THREE from "three";

const PX_PER_UNIT = 96;
const MAX_TEXTURE = 2048;

function wrapLines(ctx, text, maxWidth) {
  const lines = [];
  for (const paragraph of String(text).split("\n")) {
    let line = "";
    for (const word of paragraph.split(/\s+/)) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxWidth && line) {
        lines.push(line);
        line = word;
      } else line = test;
    }
    lines.push(line);
  }
  return lines;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/**
 * Rettangolo 3D con titolo, sottotitolo e testo (disegnati su canvas).
 * @param {object} o
 * @param {string} o.title
 * @param {string} [o.subtitle] riga breve sotto il titolo (es. forma o numero di pesi)
 * @param {string} [o.text] descrizione, va a capo da sola
 * @param {number} [o.width=6] @param {number} [o.height=3] @param {number} [o.depth=0.25]
 * @param {string} [o.color="#38bdf8"] colore del bordo e del titolo
 * @param {string} [o.fill="rgba(15,23,42,0.92)"] riempimento
 * @returns {THREE.Group} con userData.block = o
 */
export function makeBlock({ title, subtitle = "", text = "", width = 6, height = 3, depth = 0.25, color = "#38bdf8", fill = "rgba(15,23,42,0.92)" }) {
  const scale = Math.min(PX_PER_UNIT, MAX_TEXTURE / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  const pad = 0.08 * scale + 6;
  const radius = Math.min(0.25 * scale, canvas.height / 4);

  roundRect(ctx, 3, 3, canvas.width - 6, canvas.height - 6, radius);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = Math.max(3, scale * 0.04);
  ctx.strokeStyle = color;
  ctx.stroke();

  const titlePx = Math.max(14, Math.min(scale * 0.42, canvas.height * 0.2));
  ctx.textBaseline = "top";
  ctx.fillStyle = color;
  ctx.font = `700 ${titlePx}px Inter, system-ui, sans-serif`;
  let y = pad;
  for (const line of wrapLines(ctx, title, canvas.width - 2 * pad)) {
    ctx.fillText(line, pad, y);
    y += titlePx * 1.15;
  }
  if (subtitle) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = `500 ${titlePx * 0.62}px "JetBrains Mono", ui-monospace, monospace`;
    ctx.fillText(subtitle, pad, y + 2);
    y += titlePx * 0.62 * 1.4 + 2;
  }
  if (text) {
    // La descrizione deve entrare tutta: se non ci sta, il corpo del testo si rimpicciolisce
    // (fino al 60% della misura normale); solo oltre quel limite le ultime righe vengono tagliate.
    const fits = (px) => {
      ctx.font = `400 ${px}px Inter, system-ui, sans-serif`;
      const lines = wrapLines(ctx, text, canvas.width - 2 * pad);
      return { lines, ok: y + px * 0.3 + lines.length * px * 1.3 - px * 0.3 <= canvas.height - pad };
    };
    let bodyPx = titlePx * 0.6;
    let layout = fits(bodyPx);
    while (!layout.ok && bodyPx > titlePx * 0.36) layout = fits((bodyPx *= 0.92));
    ctx.fillStyle = "#e2e8f0";
    y += bodyPx * 0.3;
    for (const line of layout.lines) {
      if (y + bodyPx > canvas.height - pad) break;
      ctx.fillText(line, pad, y);
      y += bodyPx * 1.3;
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const group = new THREE.Group();
  const face = new THREE.Mesh(
    new THREE.PlaneGeometry(width, height),
    // polygonOffset: da lontano 0,01 unità non bastano a separare la faccia col testo dal corpo
    // (z-fighting: righe scure sul testo); l'offset la porta sempre davanti.
    new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 }),
  );
  face.position.z = depth / 2 + 0.01;
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.985, height * 0.97, depth),
    new THREE.MeshStandardMaterial({ color: 0x0f172a, transparent: true, opacity: 0.85, roughness: 0.6 }),
  );
  group.add(body, face);
  group.userData.block = { title, subtitle, text, width, height, depth, color };
  return group;
}

/** Freccia da `from` a `to` (Vector3), cilindro + cono. */
export function makeArrow(from, to, { color = 0x38bdf8, radius = 0.06, headLength = 0.4 } = {}) {
  const dir = new THREE.Vector3().subVectors(to, from);
  const length = dir.length();
  const group = new THREE.Group();
  if (length < 1e-6) return group;
  const material = new THREE.MeshBasicMaterial({ color });
  const shaftLength = Math.max(0.001, length - headLength);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, shaftLength, 10), material);
  shaft.position.y = shaftLength / 2;
  const head = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.6, headLength, 14), material);
  head.position.y = shaftLength + headLength / 2;
  group.add(shaft, head);
  group.position.copy(from);
  group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return group;
}

/**
 * Tubo lungo una polilinea con trattini che scorrono (direzione del flusso).
 * Chiama `flow.userData.tick(dt)` a ogni frame (es. host.onFrame(dt => flow.userData.tick(dt))).
 */
export function makeFlow(points, { color = "#67e8f9", radius = 0.05, speed = 0.8, dashes = 6 } = {}) {
  const path = new THREE.CurvePath();
  for (let i = 1; i < points.length; i++) path.add(new THREE.LineCurve3(points[i - 1], points[i]));
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 4;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(103,232,249,0.15)";
  ctx.fillRect(0, 0, 64, 4);
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 28, 4);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.repeat.set(Math.max(1, path.getLength() / 0.6) / (dashes / 6), 1);
  const mesh = new THREE.Mesh(
    new THREE.TubeGeometry(path, Math.max(16, points.length * 24), radius, 8, false),
    new THREE.MeshBasicMaterial({ map: texture, transparent: true }),
  );
  mesh.userData.path = path;
  mesh.userData.tick = (dt) => {
    texture.offset.x -= dt * speed;
  };
  return mesh;
}
