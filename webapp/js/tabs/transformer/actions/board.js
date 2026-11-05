/**
 * "Lavagna" 3D: un pannello nella scena, accanto al componente inquadrato, su cui disegniamo i
 * numeri veri del passo corrente (matrici di attivazioni, probabilità, testo).
 *
 * È un solo piano con una CanvasTexture: si ridisegna solo quando cambia il passo (enter), mai a
 * ogni frame. Le matrici si disegnano 1 pixel per valore su un canvas di appoggio e poi si
 * ingrandiscono senza sfumature: veloce anche per matrici 32 × 256.
 */
import * as THREE from "three";
import { colormaps } from "./colormaps.js";
import { maxAbs, visibleChar } from "./math.js";

export const BOARD_W = 1500;
export const BOARD_H = 1050;
const FONT = "Inter, system-ui, sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, monospace";
export const INK = { text: "#e2e8f0", muted: "#94a3b8", accent: "#38bdf8", warm: "#fbbf24", good: "#4ade80", bad: "#f87171", grid: "#334155" };

export class Board {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = BOARD_W;
    this.canvas.height = BOARD_H;
    this.ctx = this.canvas.getContext("2d");
    this.scratch = document.createElement("canvas"); // appoggio per le matrici (1 px = 1 valore)
    this.scratchCtx = this.scratch.getContext("2d");
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 8;
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(BOARD_W / BOARD_H, 1),
      // depthTest false: la lavagna si disegna sopra la scena (altrimenti le righe vicine la "bucano")
      new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide }),
    );
    plane.renderOrder = 20;
    this.mesh = new THREE.Group();
    this.mesh.add(plane);
    this.mesh.visible = false;
    this.mesh.userData.board = true;
    this.plane = plane;
    // Ago della "ruota" delle probabilità: un mesh sopra la lavagna, che si muove senza ridisegnare il canvas.
    this.needle = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, depthWrite: false, depthTest: false }));
    this.needle.renderOrder = 21;
    this.needle.visible = false;
    this.mesh.add(this.needle);
  }

  /** Mostra l'ago come linea verticale alla x di canvas `px`, da `top` a `bottom` (pixel di canvas). */
  needleAt(px, top, bottom, widthPx = 7) {
    const a = this.aspect;
    this.needle.visible = true;
    this.needle.scale.set((widthPx / BOARD_W) * a, (bottom - top) / BOARD_H, 1);
    this.needle.position.set((px / BOARD_W - 0.5) * a, 0.5 - (top + bottom) / 2 / BOARD_H, 0.003);
  }

  hideNeedle() {
    this.needle.visible = false;
  }

  get aspect() {
    return BOARD_W / BOARD_H;
  }

  /** Altezza del pannello in unità di scena (la larghezza segue l'aspetto). */
  setHeight(h) {
    this.mesh.scale.set(h, h, 1);
  }

  /** Pulisce e disegna intestazione; restituisce la y da cui partire con il contenuto. */
  begin(title, subtitle = "") {
    const g = this.ctx;
    g.setTransform(1, 0, 0, 1, 0, 0);
    this.hideNeedle();
    g.clearRect(0, 0, BOARD_W, BOARD_H);
    roundRect(g, 4, 4, BOARD_W - 8, BOARD_H - 8, 28);
    g.fillStyle = "rgba(8, 13, 24, 0.94)";
    g.fill();
    g.lineWidth = 4;
    g.strokeStyle = "#1e3a5f";
    g.stroke();
    g.textBaseline = "top";
    g.fillStyle = INK.accent;
    g.font = `700 56px ${FONT}`;
    g.fillText(title, 44, 30, BOARD_W - 88);
    let y = 100;
    if (subtitle) {
      y = this.paragraph(subtitle, 44, y, BOARD_W - 88, { size: 38, color: "#cbd5e1" });
    }
    return y + 18;
  }

  /** Testo che va a capo; restituisce la y sotto l'ultima riga. */
  paragraph(text, x, y, width, { size = 32, color = INK.text, weight = 400, mono = false, lineHeight = 1.3 } = {}) {
    const g = this.ctx;
    g.font = `${weight} ${size}px ${mono ? MONO : FONT}`;
    g.fillStyle = color;
    g.textBaseline = "top";
    for (const paragraph of String(text).split("\n")) {
      let line = "";
      for (const word of paragraph.split(" ")) {
        const test = line ? `${line} ${word}` : word;
        if (g.measureText(test).width > width && line) {
          g.fillText(line, x, y);
          y += size * lineHeight;
          line = word;
        } else line = test;
      }
      g.fillText(line, x, y);
      y += size * lineHeight;
    }
    return y;
  }

  /** Una riga di testo singola (troncata a maxWidth). */
  label(text, x, y, { size = 28, color = INK.text, weight = 500, mono = false, align = "left", maxWidth } = {}) {
    const g = this.ctx;
    g.font = `${weight} ${size}px ${mono ? MONO : FONT}`;
    g.fillStyle = color;
    g.textAlign = align;
    g.textBaseline = "top";
    g.fillText(text, x, y, maxWidth);
    g.textAlign = "left";
  }

  /**
   * Matrice come heatmap.
   * @param {object} o
   * @param {"signed"|"attention"} [o.kind="signed"] signed: blu/arancione rispetto a ±scale;
   *        attention: 0..1 scuro → azzurro, con il futuro (−∞ o j > i) tratteggiato
   * @param {number} [o.scale] per "signed": default max |v|
   * @param {string[]} [o.rowLabels] etichette delle righe (es. i caratteri del contesto)
   * @param {number} [o.highlightRow] riga da incorniciare
   * @param {string} [o.title] titolo sopra la matrice
   * @param {boolean} [o.mask] per "signed": tratteggia il triangolo sopra la diagonale (futuro)
   */
  matrix(x, y, w, h, values, rows, cols, { kind = "signed", scale, rowLabels, highlightRow = -1, title, caption, mask = false, square = false } = {}) {
    const g = this.ctx;
    const { signed, attention, masked } = colormaps();
    const titleH = title ? 44 : 0;
    h -= titleH + (caption ? 46 : 0);
    const labelW = rowLabels ? 34 : 0;
    let showLabels = rowLabels && h / rows >= 13;
    let mx = x + (showLabels ? labelW : 0);
    let mw = w - (showLabels ? labelW : 0);
    if (square) {
      // celle quadrate (es. attention T × T): la matrice si restringe nel lato più lungo e si
      // centra nello spazio che aveva, insieme al titolo
      const cell = Math.min(mw / cols, h / rows);
      showLabels = rowLabels && cell >= 13;
      const shift = (mw - cell * cols) / 2;
      x += shift;
      mx += shift;
      mw = cell * cols;
      h = cell * rows;
    }
    if (title) {
      this.label(title, x, y, { size: 34, weight: 600, maxWidth: w + 60 });
      y += titleH;
    }
    const cellH = h / rows;

    this.scratch.width = cols;
    this.scratch.height = rows;
    const image = this.scratchCtx.createImageData(cols, rows);
    const px = image.data;
    const s = kind === "signed" ? scale || maxAbs(values) || 1 : 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r * cols + c];
        let src;
        let k;
        if (kind === "attention" || (mask && c > r)) {
          if (!Number.isFinite(v) || c > r) {
            src = masked;
            k = ((r + c) % 4) * 4;
          } else {
            src = attention;
            k = Math.max(0, Math.min(255, Math.round(v * 255))) * 4;
          }
        } else {
          src = signed;
          k = Math.max(0, Math.min(255, Math.round((v / s) * 127) + 128)) * 4;
        }
        const o = (r * cols + c) * 4;
        px[o] = src[k];
        px[o + 1] = src[k + 1];
        px[o + 2] = src[k + 2];
        px[o + 3] = 255;
      }
    }
    this.scratchCtx.putImageData(image, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(this.scratch, mx, y, mw, h);
    g.imageSmoothingEnabled = true;
    g.lineWidth = 2;
    g.strokeStyle = INK.grid;
    g.strokeRect(mx, y, mw, h);
    if (showLabels) {
      g.font = `500 ${Math.min(24, cellH * 0.9)}px ${MONO}`;
      g.fillStyle = INK.muted;
      g.textAlign = "right";
      g.textBaseline = "middle";
      rowLabels.forEach((l, r) => g.fillText(visibleChar(l), mx - 6, y + (r + 0.5) * cellH));
      g.textAlign = "left";
    }
    if (highlightRow >= 0) {
      g.lineWidth = 4;
      g.strokeStyle = INK.warm;
      g.strokeRect(mx - 3, y + highlightRow * cellH - 2, mw + 6, cellH + 4);
    }
    if (caption) this.label(caption, x, y + h + 8, { size: 30, color: INK.muted, maxWidth: w + 40 });
    return { x: mx, y, w: mw, h, cellH, cellW: mw / cols };
  }

  /**
   * Barre orizzontali etichettate (probabilità, gradienti…). I valori possono essere negativi:
   * le barre partono dalla linea dello zero.
   * items: [{label, value, highlight?, color?}]
   */
  bars(x, y, w, h, items, { max, min, format = (v) => v.toFixed(3), labelWidth = 120, valueWidth = 150, color = "#475569", highlightColor = INK.accent } = {}) {
    const g = this.ctx;
    const rowH = h / items.length;
    const size = Math.min(34, rowH * 0.66);
    const lo = Math.min(0, min ?? Math.min(...items.map((i) => i.value)));
    const hi = Math.max(max ?? Math.max(...items.map((i) => i.value)), lo + 1e-9);
    const left = x + labelWidth;
    const span = w - labelWidth - valueWidth; // spazio a destra per il valore scritto
    const toX = (v) => left + ((v - lo) / (hi - lo)) * span;
    const zeroX = toX(0);
    items.forEach((item, i) => {
      const cy = y + i * rowH;
      const vx = toX(item.value);
      g.fillStyle = item.color ?? (item.highlight ? highlightColor : color);
      g.fillRect(Math.min(zeroX, vx), cy + rowH * 0.16, Math.max(2, Math.abs(vx - zeroX)), rowH * 0.68);
      g.font = `${item.highlight ? 700 : 500} ${size}px ${MONO}`;
      g.fillStyle = item.highlight ? INK.text : INK.muted;
      g.textBaseline = "middle";
      g.textAlign = "right";
      g.fillText(item.label, left - 12, cy + rowH / 2, labelWidth - 14);
      g.textAlign = "left";
      g.fillText(format(item.value), Math.max(zeroX, vx) + 10, cy + rowH / 2);
    });
    if (lo < 0) {
      g.strokeStyle = INK.muted;
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(zeroX, y);
      g.lineTo(zeroX, y + h);
      g.stroke();
    }
  }

  /**
   * Testo come fila di celle (un carattere per cella), con evidenziazioni per intervallo.
   * marks: [{from, to, color}] con indici nel testo. wrapNewlines false: "↵" è solo una cella
   * (per mostrare finestre di training su una riga).
   */
  charCells(text, x, y, { cell = 34, cols = 38, marks = [], ids = null, wrapNewlines = true } = {}) {
    const g = this.ctx;
    const chars = [...text];
    let col = 0;
    let row = 0;
    const rowH = ids ? cell * 1.75 : cell * 1.18;
    chars.forEach((ch, i) => {
      if (col >= cols) {
        col = 0;
        row++;
      }
      const cx = x + col * cell;
      const cy = y + row * rowH;
      const mark = marks.find((m) => i >= m.from && i < m.to);
      g.fillStyle = mark?.color ?? "rgba(30, 41, 59, 0.9)";
      g.fillRect(cx + 1, cy + 1, cell - 2, cell - 2);
      g.fillStyle = ch === "\n" || ch === " " ? INK.muted : INK.text;
      g.font = `600 ${cell * 0.62}px ${MONO}`;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(visibleChar(ch), cx + cell / 2, cy + cell / 2);
      if (ids) {
        g.font = `500 ${cell * 0.4}px ${MONO}`;
        g.fillStyle = INK.warm;
        g.fillText(String(ids[i]), cx + cell / 2, cy + cell * 1.33);
      }
      g.textAlign = "left";
      col++;
      if (ch === "\n" && wrapNewlines) {
        col = 0;
        row++;
      }
    });
    return y + (row + 1) * rowH;
  }

  /** Riquadro con un numero grande e la sua didascalia. */
  stat(x, y, w, value, caption, color = INK.warm) {
    const g = this.ctx;
    roundRect(g, x, y, w, 130, 16);
    g.fillStyle = "rgba(30, 41, 59, 0.85)";
    g.fill();
    this.label(value, x + 22, y + 16, { size: 54, weight: 700, color, mono: true, maxWidth: w - 40 });
    this.label(caption, x + 22, y + 84, { size: 28, color: INK.muted, maxWidth: w - 40 });
  }

  /** Pubblica il disegno sulla texture. */
  commit() {
    this.texture.needsUpdate = true;
  }

  dispose() {
    this.needle.geometry.dispose();
    this.needle.material.dispose();
    this.texture.dispose();
    this.plane.geometry.dispose();
    this.plane.material.dispose();
  }
}

export function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}
