/**
 * Mattoni 3D del tab "LLM moderni", costruiti sopra il nucleo (diagram.js, visuals.js).
 *
 * Regole di prestazione seguite qui:
 * - tante celle uguali → un solo InstancedMesh (Cells);
 * - tante linee → un solo LineSegments con buffer preallocato (Links);
 * - testo che cambia → una sola CanvasTexture ridisegnata SOLO quando il testo cambia (Plate);
 * - nessuna allocazione nei metodi chiamati a ogni frame (oggetti di appoggio riusati).
 * Ogni mesh ha il proprio materiale: il LOD sfuma l'opacità materiale per materiale.
 */
import * as THREE from "three";
import { makeArrow, makeBlock } from "../../core/diagram.js";
import { label } from "../../core/visuals.js";

export const PALETTE = {
  ours: "#f59e0b", // il nostro GPT (come i pesi "appresi" del tab Transformer)
  modern: "#38bdf8",
  moe: "#a78bfa",
  ok: "#4ade80",
  bad: "#f87171",
  warn: "#facc15",
  text: "#e2e8f0",
  muted: "#94a3b8",
  data: "#f472b6",
  tool: "#34d399",
};

/** Le etichette sprite delle regioni si leggono a ~40 unità di distanza: le ingrandiamo tutte. */
const TEXT_SCALE = 1.45;

const tmpObject = new THREE.Object3D();
const tmpColor = new THREE.Color();

/** Scatola con materiale proprio. `glow` (0..1) usa l'emissive per "accendere" un componente. */
export function box({ w = 1, h = 1, d = 0.4, color = "#1e293b", emissive = "#000000", opacity = 1 } = {}) {
  const material = new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: 0, roughness: 0.55, transparent: opacity < 1, opacity });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  mesh.userData.size = { w, h, d };
  return mesh;
}

/** Accende (k = 1) o spegne (k = 0) una scatola creata con box(). */
export function glow(mesh, k, color) {
  const m = mesh.material;
  if (color) m.emissive.set(color);
  m.emissiveIntensity = k;
}

/** Etichetta (sprite del nucleo) posizionata. */
export function text(parent, value, x, y, { size = 0.6, color = PALETTE.text, sub = null, z = 0.4 } = {}) {
  const sprite = label(value, { size: size * TEXT_SCALE, color, sub });
  sprite.position.set(x, y, z);
  parent.add(sprite);
  return sprite;
}

/**
 * "Card": scatola piatta con titolo (e sottotitolo) sopra. Restituisce { mesh, label, set(k) }
 * dove set(k) la accende con il colore della card.
 */
export function card(parent, { x = 0, y = 0, w = 6, h = 2, title = "", sub = null, color = PALETTE.modern, size = 0.5, fill = "#111827" }) {
  const mesh = box({ w, h, d: 0.35, color: fill, emissive: color });
  mesh.position.set(x, y, 0);
  parent.add(mesh);
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color }));
  mesh.add(edge);
  // testo disegnato sulla faccia della card: dimensione in unità di scena, quindi proporzionato alla card
  const font = Math.min(size * 1.7, (h * 0.8) / (sub ? 2.3 : 1.35));
  const tag = title ? new Plate({ w: w - 0.2, h: h - 0.1, size: font, align: "center" }).place(parent, x, y, 0.2) : null;
  tag?.set(sub ? [{ text: title, bold: true, color: "#f8fafc" }, { text: sub, color: "#cbd5e1", scale: 0.68 }] : [{ text: title, bold: true, color: "#f8fafc" }]);
  return {
    mesh,
    label: tag?.mesh ?? null,
    edge,
    set(k) {
      glow(mesh, 0.32 * k);
    },
  };
}

/** Freccia lungo una polilinea [[x,y], …] con la punta solo sull'ultimo tratto. */
export function arrow(parent, points, { color = 0x67e8f9, radius = 0.06, head = 0.45, z = 0.1 } = {}) {
  const group = new THREE.Group();
  for (let i = 1; i < points.length; i++) {
    const [x1, y1] = points[i - 1];
    const [x2, y2] = points[i];
    const last = i === points.length - 1;
    group.add(makeArrow(new THREE.Vector3(x1, y1, z), new THREE.Vector3(x2, y2, z), { color, radius, headLength: last ? head : 0.001 }));
  }
  parent.add(group);
  return group;
}

/** makeBlock "ingrandito": blocco largo w e alto h con il testo scalato di S (leggibile da lontano). */
export function bigBlock(parent, spec, { x, y, w, h, S }) {
  const b = makeBlock({ ...spec, width: w / S, height: h / S });
  b.scale.setScalar(S);
  b.position.set(x, y, 0);
  parent.add(b);
  return b;
}

/** Cornice rettangolare sottile (contorno di una regione). */
export function frame(parent, w, h, color = "#334155") {
  const shape = new THREE.Shape();
  const r = 0.8;
  shape.moveTo(-w / 2 + r, -h / 2);
  shape.lineTo(w / 2 - r, -h / 2);
  shape.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  shape.lineTo(w / 2, h / 2 - r);
  shape.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  shape.lineTo(-w / 2 + r, h / 2);
  shape.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  shape.lineTo(-w / 2, -h / 2 + r);
  shape.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  const points = shape.getPoints(6).map((p) => new THREE.Vector3(p.x, p.y, -0.3));
  const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color }));
  line.userData.noBounds = true;
  const back = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: "#0b1220", transparent: true, opacity: 0.55, depthWrite: false }));
  back.position.z = -0.35;
  back.userData.isFrame = true;
  parent.add(back, line);
  return line;
}

/**
 * Pannello di testo che cambia (contatori, righe di codice, didascalie nella scena).
 * La texture si ridisegna solo quando il contenuto cambia davvero.
 */
export class Plate {
  constructor({ w = 8, h = 2, px = 52, size = 0.45, color = PALETTE.text, align = "left", mono = false, background = null, border = null } = {}) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = Math.round(w * px);
    this.canvas.height = Math.round(h * px);
    this.ctx = this.canvas.getContext("2d");
    this.opts = { w, h, px, size, color, align, mono, background, border };
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 4;
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthWrite: false }));
    this.mesh.renderOrder = 5;
    this.key = null;
  }

  /**
   * @param {string | {text: string, color?: string, bold?: boolean}[]} content testo (con \n) o righe colorate
   */
  set(content) {
    if (content === this.ref) return this; // stesso oggetto (o stessa stringa): niente da ridisegnare
    this.ref = content;
    const key = typeof content === "string" ? content : JSON.stringify(content);
    if (key === this.key) return this;
    this.key = key;
    const { px, size, color, align, mono, background, border } = this.opts;
    const ctx = this.ctx;
    const { width, height } = this.canvas;
    ctx.clearRect(0, 0, width, height);
    if (background) {
      ctx.fillStyle = background;
      ctx.beginPath();
      ctx.roundRect(2, 2, width - 4, height - 4, 12);
      ctx.fill();
      if (border) {
        ctx.strokeStyle = border;
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    }
    const lines = typeof content === "string" ? content.split("\n").map((t) => ({ text: t })) : content;
    const pad = size * px * 0.5;
    // se le righe non entrano in altezza si rimpicciolisce tutto il testo (mai testo tagliato)
    const natural = lines.reduce((a, l) => a + size * px * (l.scale ?? 1) * 1.3, 0);
    const fontPx = size * px * Math.min(1, (height - pad * 0.6) / natural);
    const heights = lines.map((l) => fontPx * (l.scale ?? 1) * 1.3);
    const totalH = heights.reduce((a, b) => a + b, 0);
    ctx.textBaseline = "top";
    ctx.textAlign = align;
    const x = align === "center" ? width / 2 : align === "right" ? width - pad : pad;
    let y = Math.max(pad * 0.4, (height - totalH) / 2);
    lines.forEach((line, i) => {
      // la riga si rimpicciolisce da sola se non entra in larghezza (niente testo tagliato)
      let f = fontPx * (line.scale ?? 1);
      const family = mono ? '"JetBrains Mono", ui-monospace, monospace' : "Inter, system-ui, sans-serif";
      const weight = line.bold ? 700 : 500;
      ctx.font = `${weight} ${f}px ${family}`;
      const room = width - 2 * pad;
      const w = ctx.measureText(line.text).width;
      if (w > room) {
        f *= room / w;
        ctx.font = `${weight} ${f}px ${family}`;
      }
      ctx.fillStyle = line.color ?? color;
      ctx.fillText(line.text, x, y + (heights[i] - f * 1.3) / 2);
      y += heights[i];
    });
    this.texture.needsUpdate = true;
    return this;
  }

  place(parent, x, y, z = 0.3) {
    this.mesh.position.set(x, y, z);
    parent.add(this.mesh);
    return this;
  }
}

/**
 * Tante scatole uguali in un solo InstancedMesh. Ogni istanza ha posizione, scala e colore.
 * hide(i) la nasconde con scala zero (niente ricostruzioni di geometria).
 */
export class Cells {
  constructor(count, { w = 1, h = 1, d = 0.3, color = "#334155", emissive = "#000000", emissiveIntensity = 0 } = {}) {
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ color: "#ffffff", emissive, emissiveIntensity, roughness: 0.5 }), count);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.count = count;
    this.pos = new Float32Array(count * 3);
    this.scale = new Float32Array(count * 3).fill(1);
    for (let i = 0; i < count; i++) this.mesh.setColorAt(i, tmpColor.set(color));
    this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  }

  /** Posizione (e scala opzionale) dell'istanza i. */
  place(i, x, y, z = 0, sx = 1, sy = 1, sz = 1) {
    this.pos[i * 3] = x;
    this.pos[i * 3 + 1] = y;
    this.pos[i * 3 + 2] = z;
    this.scale[i * 3] = sx;
    this.scale[i * 3 + 1] = sy;
    this.scale[i * 3 + 2] = sz;
    tmpObject.position.set(x, y, z);
    tmpObject.scale.set(sx || 1e-4, sy || 1e-4, sz || 1e-4);
    tmpObject.updateMatrix();
    this.mesh.setMatrixAt(i, tmpObject.matrix);
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  /** Cambia solo la scala (es. barre che crescono). */
  resize(i, sx, sy, sz) {
    this.place(i, this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2], sx, sy, sz);
  }

  hide(i) {
    this.place(i, this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2], 0, 0, 0);
  }

  color(i, value) {
    this.mesh.setColorAt(i, typeof value === "number" || typeof value === "string" ? tmpColor.set(value) : value);
    this.mesh.instanceColor.needsUpdate = true;
  }

  /** Colore interpolato tra a e b (stringhe esadecimali), t in 0..1. */
  mix(i, a, b, t) {
    tmpColor.set(a).lerp(scratchColor.set(b), Math.min(1, Math.max(0, t)));
    this.color(i, tmpColor);
  }
}
const scratchColor = new THREE.Color();

/** Tanti segmenti in un solo LineSegments con buffer preallocato (posizioni e colori). */
export class Links {
  constructor(max, { opacity = 1 } = {}) {
    this.max = max;
    this.positions = new Float32Array(max * 6);
    this.colors = new Float32Array(max * 6);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("color", new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    this.line = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: opacity < 1, opacity }));
    this.line.userData.noBounds = true;
    this.line.frustumCulled = false;
  }

  set(i, ax, ay, az, bx, by, bz) {
    const p = this.positions;
    const o = i * 6;
    p[o] = ax;
    p[o + 1] = ay;
    p[o + 2] = az;
    p[o + 3] = bx;
    p[o + 4] = by;
    p[o + 5] = bz;
    this.line.geometry.attributes.position.needsUpdate = true;
  }

  color(i, value) {
    tmpColor.set(value);
    for (let k = 0; k < 2; k++) {
      this.colors[i * 6 + k * 3] = tmpColor.r;
      this.colors[i * 6 + k * 3 + 1] = tmpColor.g;
      this.colors[i * 6 + k * 3 + 2] = tmpColor.b;
    }
    this.line.geometry.attributes.color.needsUpdate = true;
  }

  /** Nasconde il segmento i (lunghezza zero): niente linee "nere" dimenticate sullo sfondo. */
  hide(i) {
    this.positions.fill(0, i * 6, i * 6 + 6);
    this.line.geometry.attributes.position.needsUpdate = true;
  }

  setCount(n) {
    this.line.geometry.setDrawRange(0, 2 * Math.min(n, this.max));
  }
}

/** Sfera "pacchetto" che si muove lungo i percorsi (dati, messaggi, token). */
export function packet(color = "#fde047", radius = 0.35) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.8 }));
  mesh.userData.noBounds = true;
  return mesh;
}

/** Interpolazione lungo una polilinea [[x,y], …] con t in 0..1 (scrive in `out`). */
export function alongPath(points, t, out) {
  let total = 0;
  for (let i = 1; i < points.length; i++) total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  let left = Math.min(1, Math.max(0, t)) * total;
  for (let i = 1; i < points.length; i++) {
    const seg = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    if (left <= seg || i === points.length - 1) {
      const k = seg > 0 ? Math.min(1, left / seg) : 0;
      out.x = points[i - 1][0] + (points[i][0] - points[i - 1][0]) * k;
      out.y = points[i - 1][1] + (points[i][1] - points[i - 1][1]) * k;
      return out;
    }
    left -= seg;
  }
  return out;
}

export const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const clamp01 = (t) => Math.min(1, Math.max(0, t));
/** Porzione [a, b] del progresso p, rimappata in 0..1. */
export const phase = (p, a, b) => clamp01((p - a) / (b - a));

/**
 * Riga (o più righe) di token disegnata su una sola texture: rettangoli arrotondati con testo.
 * Usata per BPE, decodifica speculativa, KV cache, multimodale… Ogni token:
 * { text, fill?, stroke?, color?, width? (in "unità carattere") }.
 */
export class TokenStrip {
  constructor({ w = 20, h = 2, px = 50, font = 0.5, gap = 0.18, mono = true, align = "left" } = {}) {
    this.plate = new Plate({ w, h, px });
    this.opts = { w, h, px, font, gap, mono, align };
    this.mesh = this.plate.mesh;
    this.key = null;
  }

  set(tokens) {
    if (tokens === this.ref) return this;
    this.ref = tokens;
    const key = JSON.stringify(tokens);
    if (key === this.key) return this;
    this.key = key;
    const { px, font, gap, mono, align } = this.opts;
    const { canvas, ctx, texture } = this.plate;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const fontPx = font * px;
    ctx.font = `600 ${fontPx}px ${mono ? '"JetBrains Mono", ui-monospace, monospace' : "Inter, system-ui, sans-serif"}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    const padX = fontPx * 0.35;
    const boxH = fontPx * 1.55;
    const gapPx = gap * px;
    const rows = [[]];
    let rowW = 0;
    for (const t of tokens) {
      const w = Math.max(fontPx * 0.9, ctx.measureText(t.text).width + 2 * padX);
      if (rowW + w > canvas.width - 8 && rows.at(-1).length) {
        rows.push([]);
        rowW = 0;
      }
      rows.at(-1).push({ t, w });
      rowW += w + gapPx;
    }
    const totalH = rows.length * boxH + (rows.length - 1) * gapPx;
    let y = (canvas.height - totalH) / 2;
    for (const row of rows) {
      const width = row.reduce((s, r) => s + r.w + gapPx, -gapPx);
      let x = align === "center" ? (canvas.width - width) / 2 : 4;
      for (const { t, w } of row) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, boxH, fontPx * 0.25);
        ctx.fillStyle = t.fill ?? "#1e293b";
        ctx.fill();
        ctx.lineWidth = t.strokeWidth ?? 2;
        ctx.strokeStyle = t.stroke ?? "#475569";
        ctx.stroke();
        ctx.fillStyle = t.color ?? "#f1f5f9";
        ctx.fillText(t.text, x + w / 2, y + boxH / 2 + 1);
        x += w + gapPx;
      }
      y += boxH + gapPx;
    }
    texture.needsUpdate = true;
    return this;
  }

  place(parent, x, y, z = 0.3) {
    this.plate.place(parent, x, y, z);
    return this;
  }
}
