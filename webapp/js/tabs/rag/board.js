/**
 * "Lavagne" di testo per la scena: un piano con una CanvasTexture su cui disegniamo pagine PDF,
 * chunk, tabelle, prompt e chat.
 *
 * Il canvas si ridisegna SOLO quando il contenuto cambia (redraw con una chiave): le animazioni
 * muovono gli oggetti a ogni frame ma non toccano le texture, così si resta a 60 fps.
 *
 * Il testo usa un font monospazio: ogni carattere ha la stessa larghezza, quindi evidenziare un
 * intervallo di caratteri (un token, una riga tolta dalla pulizia) è solo questione di colonne.
 */
import * as THREE from "three";

export const MONO = '"JetBrains Mono", "DejaVu Sans Mono", ui-monospace, monospace';
export const SANS = "Inter, system-ui, sans-serif";

/** Stili degli intervalli evidenziati (colori leggibili su sfondo scuro). */
export const MARKS = {
  removed: { color: "#fca5a5", bg: "rgba(239,68,68,0.18)", strike: "#f87171" },
  hyphen: { color: "#fde68a", bg: "rgba(250,204,21,0.30)" },
  joined: { color: "#e2e8f0", bg: "rgba(34,211,238,0.14)" },
  spaces: { color: "#fde68a", bg: "rgba(250,204,21,0.45)" },
  flagged: { color: "#fed7aa", bg: "rgba(249,115,22,0.30)", underline: "#fb923c" },
  added: { color: "#bbf7d0", bg: "rgba(34,197,94,0.20)" },
  tokenA: { color: "#dbeafe", bg: "rgba(56,189,248,0.16)" },
  tokenB: { color: "#dbeafe", bg: "rgba(129,140,248,0.22)" },
  chunk0: { color: "#e0f2fe", bg: "rgba(56,189,248,0.22)" },
  chunk1: { color: "#dcfce7", bg: "rgba(74,222,128,0.22)" },
  chunk2: { color: "#fef3c7", bg: "rgba(251,191,36,0.22)" },
  dim: { color: "#64748b" },
  cut: { color: "#fecaca", bg: "rgba(239,68,68,0.55)" },
};

/** Colori dei chunk: i token pari e dispari di uno stesso chunk alternano due intensità. */
export const CHUNK_RGB = ["56,189,248", "74,222,128", "251,191,36", "244,114,182"];
CHUNK_RGB.forEach((rgb, k) => {
  MARKS[`c${k}e`] = { color: "#f1f5f9", bg: `rgba(${rgb},0.30)` };
  MARKS[`c${k}o`] = { color: "#f1f5f9", bg: `rgba(${rgb},0.16)` };
});

export class Board {
  /**
   * @param {object} o
   * @param {number} o.width   larghezza in unità di scena
   * @param {number} o.height  altezza in unità di scena
   * @param {number} [o.ppu=56] pixel di canvas per unità (risoluzione)
   */
  constructor({ width, height, ppu = 56 }) {
    const scale = Math.min(ppu, 2048 / Math.max(width, height));
    this.canvas = document.createElement("canvas");
    this.canvas.width = Math.round(width * scale);
    this.canvas.height = Math.round(height * scale);
    // willReadFrequently: canvas in memoria di sistema. Il testo si disegna una volta e poi si copia
    // nella texture: senza, il browser lo rasterizza sulla GPU e a ogni upload deve rileggerlo
    // (con il rendering software un ridisegno costava circa il doppio).
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.W = this.canvas.width;
    this.H = this.canvas.height;
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 4;
    this.mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(width, height),
      new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthWrite: false, side: THREE.DoubleSide }),
    );
    this.key = null;
  }

  /** Ridisegna con fn(ctx, W, H) solo se `key` è cambiata. */
  redraw(key, fn) {
    if (key === this.key) return;
    this.key = key;
    this.ctx.clearRect(0, 0, this.W, this.H);
    fn(this.ctx, this.W, this.H);
    this.texture.needsUpdate = true;
  }

  dispose() {
    this.texture.dispose();
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}

export function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Sfondo di un pannello con bordo e titolo opzionale; restituisce la y da cui scrivere. */
export function panel(ctx, W, H, { title = "", subtitle = "", border = "#334155", fill = "rgba(11,18,32,0.96)", accent = "#7dd3fc" } = {}) {
  const pad = Math.round(W * 0.025);
  roundRect(ctx, 2, 2, W - 4, H - 4, Math.min(24, H * 0.05));
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = border;
  ctx.stroke();
  let y = pad;
  if (title) {
    const px = Math.round(Math.min(W * 0.045, 40));
    ctx.font = `700 ${px}px ${SANS}`;
    ctx.fillStyle = accent;
    ctx.textBaseline = "top";
    ctx.fillText(title, pad, y, W - 2 * pad);
    y += px * 1.25;
    if (subtitle) {
      ctx.font = `500 ${Math.round(px * 0.62)}px ${SANS}`;
      ctx.fillStyle = "#94a3b8";
      ctx.fillText(subtitle, pad, y, W - 2 * pad);
      y += px * 0.85;
    }
    y += px * 0.25;
  }
  return { x: pad, y, w: W - 2 * pad, pad };
}

/**
 * Va a capo mantenendo le posizioni dei caratteri: restituisce [{start, end}] (end escluso)
 * riferiti al testo originale. Rispetta gli a capo del testo e spezza le righe lunghe sugli spazi.
 */
export function wrapOffsets(text, cols) {
  const lines = [];
  let start = 0;
  while (start <= text.length) {
    let nl = text.indexOf("\n", start);
    if (nl === -1) nl = text.length;
    let lineStart = start;
    while (nl - lineStart > cols) {
      let cut = text.lastIndexOf(" ", lineStart + cols);
      if (cut <= lineStart) cut = lineStart + cols;
      lines.push({ start: lineStart, end: cut });
      lineStart = text[cut] === " " ? cut + 1 : cut;
    }
    lines.push({ start: lineStart, end: nl });
    if (nl >= text.length) break;
    start = nl + 1;
  }
  return lines;
}

/**
 * Testo monospazio con intervalli evidenziati.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} text
 * @param {{x: number, y: number, w: number, h: number}} box
 * @param {object} [o]
 * @param {{start: number, end: number, style: string}[]} [o.ranges] stile = chiave di MARKS
 * @param {number} [o.cols] colonne (determina la dimensione del font)
 * @param {string} [o.color]
 * @param {{index: number, style: string}[]} [o.markers] barrette verticali (es. punto di taglio)
 * @param {boolean} [o.fit] riduce il font se il testo non sta in altezza
 * @returns {{lines: number, overflow: boolean}}
 */
export function richText(ctx, text, box, { ranges = [], cols = 60, color = "#cbd5e1", markers = [], lineGap = 1.42, eol = null, fit = false } = {}) {
  ctx.font = `400 100px ${MONO}`;
  const charRatio = ctx.measureText("M").width / 100;
  let px = Math.max(8, Math.floor(box.w / (cols * charRatio)));
  // fit: rimpicciolisce il font finché tutte le righe stanno nell'altezza disponibile
  if (fit) px = Math.max(8, Math.min(px, Math.floor(box.h / (wrapOffsets(text, cols).length * lineGap))));
  ctx.font = `400 ${px}px ${MONO}`;
  const cw = ctx.measureText("M").width;
  const lh = px * lineGap;
  const styles = new Uint8Array(text.length + 1);
  const keys = [null];
  for (const r of ranges) {
    let id = keys.indexOf(r.style);
    if (id === -1) id = keys.push(r.style) - 1;
    styles.fill(id, Math.max(0, r.start), Math.min(text.length, r.end));
  }
  const lines = wrapOffsets(text, cols);
  ctx.textBaseline = "top";
  let y = box.y;
  let drawn = 0;
  for (const line of lines) {
    if (y + lh > box.y + box.h + 1) break;
    let i = line.start;
    while (i < line.end) {
      const id = styles[i];
      let j = i + 1;
      while (j < line.end && styles[j] === id) j++;
      const mark = id ? MARKS[keys[id]] : null;
      const x = box.x + (i - line.start) * cw;
      const w = (j - i) * cw;
      if (mark?.bg) {
        ctx.fillStyle = mark.bg;
        ctx.fillRect(x, y - px * 0.12, w, lh * 0.92);
      }
      ctx.fillStyle = mark?.color ?? color;
      ctx.fillText(text.slice(i, j), x, y);
      if (mark?.strike) {
        ctx.fillStyle = mark.strike;
        ctx.fillRect(x, y + px * 0.5, w, Math.max(2, px * 0.09));
      }
      if (mark?.underline) {
        ctx.fillStyle = mark.underline;
        ctx.fillRect(x, y + px * 1.05, w, Math.max(2, px * 0.1));
      }
      i = j;
    }
    // segno di fine riga (es. "⏎" per gli a capo che la pulizia toglie)
    const eolStyle = eol?.(line);
    if (eolStyle) {
      const x = box.x + (line.end - line.start) * cw + cw * 0.3;
      ctx.fillStyle = MARKS[eolStyle]?.bg ?? "#22d3ee";
      ctx.fillRect(x - cw * 0.15, y - px * 0.12, cw * 1.3, lh * 0.92);
      ctx.fillStyle = "#67e8f9";
      ctx.fillText("⏎", x, y);
    }
    for (const m of markers) {
      if (m.index >= line.start && m.index <= line.end) {
        const x = box.x + (m.index - line.start) * cw;
        ctx.fillStyle = MARKS[m.style]?.bg ?? "#ef4444";
        ctx.fillRect(x - 2, y - px * 0.3, 5, lh * 1.1);
      }
    }
    y += lh;
    drawn++;
  }
  if (drawn < lines.length) {
    ctx.fillStyle = "#64748b";
    ctx.fillText("…", box.x, Math.min(y, box.y + box.h - lh * 0.5));
  }
  return { lines: drawn, overflow: drawn < lines.length, lineHeight: lh, fontPx: px };
}

/** Testo proporzionale a capo automatico (per didascalie dentro le lavagne). */
export function paragraph(ctx, text, x, y, w, { px = 22, color = "#e2e8f0", weight = 400, font = SANS, lineGap = 1.35, maxLines = 99 } = {}) {
  ctx.font = `${weight} ${px}px ${font}`;
  ctx.fillStyle = color;
  ctx.textBaseline = "top";
  let lines = 0;
  for (const para of String(text).split("\n")) {
    let line = "";
    for (const word of para.split(/\s+/)) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > w && line) {
        if (lines >= maxLines) return y;
        ctx.fillText(line, x, y);
        y += px * lineGap;
        lines++;
        line = word;
      } else line = test;
    }
    if (lines >= maxLines) return y;
    ctx.fillText(line, x, y);
    y += px * lineGap;
    lines++;
  }
  return y;
}

/** Cornice di una finestra del browser (il frontend React del corso); restituisce l'area utile. */
export function browserFrame(ctx, W, H, { url = "localhost:5173", title = "Gestionale" } = {}) {
  roundRect(ctx, 2, 2, W - 4, H - 4, 18);
  ctx.fillStyle = "#0b1220";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#475569";
  ctx.stroke();
  const bar = Math.round(H * 0.085);
  ctx.fillStyle = "#1e293b";
  ctx.fillRect(4, 4, W - 8, bar);
  ["#f87171", "#facc15", "#4ade80"].forEach((c, i) => {
    ctx.beginPath();
    ctx.arc(bar * 0.6 + i * bar * 0.55, 4 + bar / 2, bar * 0.17, 0, Math.PI * 2);
    ctx.fillStyle = c;
    ctx.fill();
  });
  const px = Math.round(bar * 0.42);
  ctx.font = `500 ${px}px ${MONO}`;
  ctx.fillStyle = "#94a3b8";
  ctx.textBaseline = "middle";
  ctx.fillText(`${url}  ·  ${title} (React)`, bar * 2.4, 4 + bar / 2, W - bar * 3);
  ctx.textBaseline = "top";
  const pad = Math.round(W * 0.04);
  return { x: pad, y: bar + pad, w: W - 2 * pad, h: H - bar - 2 * pad };
}

/** Etichetta "pillola" colorata (es. badge "da rivedere"). */
export function pill(ctx, text, x, y, { px = 20, color = "#0f172a", bg = "#fb923c" } = {}) {
  ctx.font = `700 ${px}px ${SANS}`;
  const w = ctx.measureText(text).width + px;
  roundRect(ctx, x, y, w, px * 1.5, px * 0.75);
  ctx.fillStyle = bg;
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + px / 2, y + px * 0.78);
  ctx.textBaseline = "top";
  return w;
}

/** Libera geometrie, materiali e texture di un sottoalbero (quando si ricostruisce la scena). */
export function disposeTree(root) {
  root.traverse((o) => {
    o.geometry?.dispose();
    const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
    for (const m of mats) {
      m.map?.dispose();
      m.dispose();
    }
  });
}
