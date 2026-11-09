/**
 * Le "carte" delle azioni (three.js): matrici di attivazione TEMPORANEE appoggiate davanti alla scena,
 * accanto ai pesi veri. Ogni carta è una NumericMatrix del nucleo (celle colorate con la stessa colormap
 * dei pesi e, da vicino, i numeri veri scritti nelle celle) su un foglio scuro, con un'etichetta e un
 * bordo colorato; si possono scoprire poco a poco (celle "da calcolare" a strisce) e marcare celle,
 * righe o colonne con riquadri gialli.
 *
 * Tutto ciò che crea si libera in dispose() (geometrie, materiali, texture, canvas). Gli oggetti non si
 * colpiscono con il raycaster e stanno fuori dalle inquadrature automatiche (userData.noBounds).
 */
import * as THREE from "three";
import { NumericMatrix } from "../../../core/matrixNumbers.js";
import { CELL } from "../scene/layout.js";
import { LabelAtlas, disposeLabelMesh, labelMesh } from "../scene/labels.js";

const FONT = `600 64px Inter, system-ui, -apple-system, "Segoe UI", sans-serif`;
const MONO = `500 56px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
export const CARD_COLORS = Object.freeze({ input: "#38bdf8", output: "#4ade80", product: "#fbbf24", grad: "#f472b6", mask: "#94a3b8", probs: "#a78bfa" });
const MARK_COLOR = 0xfde047;
/** Ordine di disegno: foglio, celle (numeri +1), bordo, marcature, etichetta. Sopra la scena (velo dei pesi: +3). */
const ORDER = { plate: 9, cells: 10, border: 12, mark: 13, label: 12 };

const quiet = (o) => {
  o.raycast = () => {};
  o.userData.noBounds = true;
  return o;
};

/** Altezza delle etichette delle carte SULLO SCHERMO (px): restano leggibili da lontano e piccole da vicino. */
export const LABEL_PX = 15;

/**
 * Un'etichetta di testo (canvas) su una "pillola" scura: un piano unitario da scalare (larghezza =
 * altezza × userData.aspect). Il testo è disegnato con fillText: nessun HTML.
 */
export function textPlane(text, { color = "#e2e8f0", mono = false, createCanvas = () => document.createElement("canvas") } = {}) {
  const canvas = createCanvas();
  const ctx = canvas.getContext("2d");
  ctx.font = mono ? MONO : FONT;
  const pad = 18;
  const h = 84;
  const w = Math.max(h, Math.ceil(ctx.measureText(text).width + 2 * pad));
  canvas.width = w;
  canvas.height = h;
  ctx.fillStyle = "rgba(10, 17, 32, 0.94)";
  roundRect(ctx, 1, 1, w - 2, h - 2, 18);
  ctx.fill();
  ctx.font = mono ? MONO : FONT;
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.fillText(text, pad, h / 2 + 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const mesh = quiet(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, toneMapped: false })));
  mesh.renderOrder = ORDER.label;
  mesh.userData.aspect = w / h;
  mesh.userData.dispose = () => {
    mesh.geometry.dispose();
    mesh.material.dispose();
    texture.dispose();
    canvas.width = 0;
    canvas.height = 0;
  };
  return mesh;
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

/** Contorno di un rettangolo (x0, x1, y0, y1) come LineLoop. */
function rectLoop(color, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, -0.5, 0), new THREE.Vector3(0.5, 0.5, 0), new THREE.Vector3(-0.5, 0.5, 0)]);
  return quiet(new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false, toneMapped: false })));
}

export class Card {
  /**
   * @param {object} o
   * @param {{x0: number, y1: number, rows: number, cols: number, cellW?: number, cellH?: number}} o.rect angolo in alto a sinistra e celle
   * @param {ArrayLike<number>} o.data valori riga per riga
   * @param {"diverging"|"attention"} [o.colormap]
   * @param {number|null} [o.range] |valore| del colore pieno (predefinito il massimo)
   * @param {string} [o.label] etichetta (testo semplice: anche caratteri del prompt)
   * @param {"top"|"left"|"bottom"|"right"} [o.labelSide]
   * @param {string} [o.accent] colore del bordo
   * @param {number} [o.z]
   * @param {string} [o.maskedText] testo delle celle "mascherate" (−∞ per i punteggi del futuro; "" per quelle da calcolare)
   * @param {import("../../../core/matrixNumbers.js").NumbersGroup} o.group il budget dei numeri
   * @param {(r: number) => string} [o.rowLabel] @param {(c: number) => string} [o.colLabel]
   */
  constructor({ rect, data, colormap = "diverging", range = null, label = "", labelSide = "top", accent = CARD_COLORS.input, z = 0.42, maskedText = "", group, rowLabel = null, colLabel = null, createCanvas }) {
    const cellW = rect.cellW ?? CELL;
    const cellH = rect.cellH ?? CELL;
    this.rows = rect.rows;
    this.cols = rect.cols;
    this.rect = { x0: rect.x0, x1: rect.x0 + rect.cols * cellW, y0: rect.y1 - rect.rows * cellH, y1: rect.y1, rows: rect.rows, cols: rect.cols, cellW, cellH };
    this.z = z;
    this.object3d = new THREE.Group();
    this.object3d.name = "lab-act-card";
    this._hidden = null; // celle non ancora calcolate: (r, c) => boolean, o null
    this._maskFn = null; // celle mascherate per davvero (−∞), o null
    this.m = new NumericMatrix({
      rows: this.rows,
      cols: this.cols,
      data,
      cell: CELL,
      colormap,
      range: typeof range === "number" && range > 0 ? range : null, // "auto" o null = il massimo dei valori
      group,
      masked: (r, c) => Boolean(this._hidden?.(r, c) || this._maskFn?.(r, c)),
      maskedText,
      rowLabel,
      colLabel,
    });
    const { x0, x1, y0, y1 } = this.rect;
    this.m.mesh.scale.set(cellW / CELL, cellH / CELL, 1);
    this.m.mesh.renderOrder = ORDER.cells;
    this.m.object3d.position.set((x0 + x1) / 2, (y0 + y1) / 2, z);
    quiet(this.m.mesh);
    this.object3d.add(this.m.object3d);

    // il foglio scuro dietro le celle: la carta si legge anche sopra le matrici della scena
    const pad = Math.max(0.008, Math.min(cellW, cellH) * 0.3);
    const plate = { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad };
    this.box = plate;
    this.pad = pad;
    // l'etichetta: grandezza costante sullo schermo (scaleLabel, chiamata dalla regia quando la camera si muove)
    this.label = label ? textPlane(label, { color: accent, createCanvas }) : null;
    this.labelSide = labelSide;
    if (this.label) {
      this.object3d.add(this.label);
      this.scaleLabel(Math.max(0.03, Math.min(cellW, cellH)));
    }
    const plateMesh = quiet(new THREE.Mesh(new THREE.PlaneGeometry(plate.x1 - plate.x0, plate.y1 - plate.y0), new THREE.MeshBasicMaterial({ color: 0x0a1120, toneMapped: false })));
    plateMesh.position.set((plate.x0 + plate.x1) / 2, (plate.y0 + plate.y1) / 2, z - 0.004);
    plateMesh.renderOrder = ORDER.plate;
    this.plate = plateMesh;
    this.object3d.add(plateMesh);

    this.border = rectLoop(new THREE.Color(accent), 0.95);
    this.border.position.set((x0 + x1) / 2, (y0 + y1) / 2, z + 0.003);
    this.border.scale.set(x1 - x0 + pad * 0.9, y1 - y0 + pad * 0.9, 1);
    this.border.renderOrder = ORDER.border;
    this.object3d.add(this.border);

    this.marks = null; // LineSegments dei riquadri gialli
    this._markGeometry = null;
  }

  /**
   * L'etichetta alta `h` nel mondo, appoggiata al suo lato della carta (a sinistra: finisce dove
   * cominciano le celle; sopra: comincia dall'angolo in alto a sinistra).
   */
  scaleLabel(h) {
    const label = this.label;
    if (!label) return;
    const w = h * label.userData.aspect;
    const { x0, x1, y0, y1 } = this.rect;
    const gap = this.pad + h * 0.25;
    label.scale.set(w, h, 1);
    const z = this.z + 0.002;
    if (this.labelSide === "left") label.position.set(x0 - gap - w / 2, (y0 + y1) / 2, z);
    else if (this.labelSide === "right") label.position.set(x1 + gap + w / 2, (y0 + y1) / 2, z);
    else if (this.labelSide === "bottom") label.position.set(x0 + w / 2, y0 - gap - h / 2, z);
    else label.position.set(x0 + w / 2, y1 + gap + h / 2, z);
    this.labelHeight = h;
  }

  /** Quanti pixel prende l'etichetta di lato (larghezza a sinistra/destra, altezza sopra/sotto) a LABEL_PX. */
  labelReservePx(px = LABEL_PX) {
    if (!this.label) return 0;
    return this.labelSide === "left" || this.labelSide === "right" ? px * this.label.userData.aspect + px * 0.6 : px * 1.5;
  }

  /** Nuovi valori (stessa forma). `range` come in NumericMatrix.setData ("auto" = ricalcola). */
  setData(data, options) {
    this.m.setData(data, options);
  }

  /** Ridisegna con gli stessi valori (dopo aver cambiato le celle nascoste o la maschera). */
  refresh() {
    this.m.setData(this.m.data);
  }

  /**
   * Scopre le prime `count` celle in ordine di lettura (riga per riga); le altre restano "da calcolare"
   * (a strisce, senza numero). `count` ≥ righe × colonne = tutte.
   */
  reveal(count) {
    if (this._setRevealed(count)) this.refresh();
  }

  /** Celle mascherate davvero (es. il futuro dell'attention, −∞): `fn(r, c)` o null. */
  setMask(fn) {
    this._maskFn = fn ?? null;
    this.refresh();
  }

  /** Celle scoperte e maschera insieme, con un solo ridisegno (all'inizio di ogni passo). */
  configure({ reveal = Infinity, mask = null } = {}) {
    const changed = this._setRevealed(reveal);
    if (changed || mask || this._maskFn) {
      this._maskFn = mask ?? null;
      this.refresh();
    }
  }

  /** `count`: quante celle scoprire in ordine di lettura, oppure una funzione (r, c) → scoperta. */
  _setRevealed(count) {
    if (typeof count === "function") {
      this._revealed = count;
      this._hidden = (r, c) => !count(r, c);
      return true;
    }
    const n = this.rows * this.cols;
    const k = Math.max(0, Math.min(n, Math.floor(count)));
    const key = k >= n ? null : k;
    if (this._revealed === key) return false;
    this._revealed = key;
    this._hidden = key === null ? null : (r, c) => r * this.cols + c >= key;
    return true;
  }

  /** Il centro della cella (i, j) nel mondo (davanti alla carta). */
  cellCenter(i, j, target = new THREE.Vector3()) {
    const { x0, y1, cellW, cellH } = this.rect;
    return target.set(x0 + (j + 0.5) * cellW, y1 - (i + 0.5) * cellH, this.z + 0.004);
  }

  /** Il rettangolo delle celle [r0, r1) × [c0, c1) nel mondo. */
  cellsRect(r0, r1, c0, c1) {
    const { x0, y1, cellW, cellH } = this.rect;
    return { x0: x0 + c0 * cellW, x1: x0 + c1 * cellW, y0: y1 - r1 * cellH, y1: y1 - r0 * cellH };
  }

  /**
   * Riquadri gialli su gruppi di celle: [{r0, r1, c0, c1}] (esclusi r1, c1) | null. Un solo LineSegments.
   */
  mark(rects) {
    if (!rects?.length) {
      if (this.marks) this.marks.visible = false;
      return;
    }
    const pos = new Float32Array(rects.length * 8 * 3);
    const z = this.z + 0.006;
    rects.forEach((q, k) => {
      const r = this.cellsRect(q.r0, q.r1, q.c0, q.c1);
      const pts = [r.x0, r.y0, r.x1, r.y0, r.x1, r.y0, r.x1, r.y1, r.x1, r.y1, r.x0, r.y1, r.x0, r.y1, r.x0, r.y0];
      for (let i = 0; i < 8; i++) pos.set([pts[2 * i], pts[2 * i + 1], z], (k * 8 + i) * 3);
    });
    if (!this.marks) {
      this._markGeometry = new THREE.BufferGeometry();
      this.marks = quiet(new THREE.LineSegments(this._markGeometry, new THREE.LineBasicMaterial({ color: MARK_COLOR, transparent: true, opacity: 1, depthWrite: false, toneMapped: false })));
      this.marks.renderOrder = ORDER.mark;
      this.object3d.add(this.marks);
    }
    this._markGeometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this._markGeometry.computeBoundingSphere();
    this.marks.visible = true;
  }

  set visible(on) {
    this.object3d.visible = Boolean(on);
  }

  get visible() {
    return this.object3d.visible;
  }

  dispose() {
    this.object3d.removeFromParent();
    this.m.dispose();
    this.label?.userData.dispose();
    this.plate.geometry.dispose();
    this.plate.material.dispose();
    this.border.geometry.dispose();
    this.border.material.dispose();
    if (this.marks) {
      this._markGeometry.dispose();
      this.marks.material.dispose();
    }
  }
}

/**
 * I caratteri del prompt accanto alle righe o alle colonne di una carta (chi è la riga 5? «s»): tutte le
 * etichette in un atlante, un mesh solo. Il testo è disegnato con fillText: nessun HTML.
 */
export class TokenStrip {
  /**
   * @param {string[]} texts un'etichetta per posizione (già "visibili": ␣ per lo spazio, ↵ per l'a capo)
   * @param {{x: number, y: number}[]} positions centro di ogni etichetta nel mondo
   * @param {{height?: number, z?: number, color?: string, marked?: number|null}} [o]
   */
  constructor(texts, positions, { height = 0.032, z = 0.43, color = "#cbd5e1", marked = null, createCanvas } = {}) {
    this.atlas = createCanvas ? new LabelAtlas({ createCanvas }) : new LabelAtlas();
    const px = 40;
    const items = texts.map((text, i) => {
      const key = this.atlas.add({ key: `${i === marked ? "m" : "t"}|${text}`, align: "center", lines: [[{ text, font: "code", px, color: i === marked ? "#fde047" : color }]] });
      return { key, x: positions[i].x, y: positions[i].y, anchor: "center", scale: height / (px * 1.28), z };
    });
    this.object3d = labelMesh(items, this.atlas.build(), { renderOrder: 12 });
    this.object3d.traverse((o) => {
      o.raycast = () => {};
      o.userData.noBounds = true;
    });
  }

  dispose() {
    disposeLabelMesh(this.object3d);
    this.atlas.dispose();
  }
}
