/**
 * Schede del livello Dettaglio: piani 3D con una texture disegnata su canvas.
 *
 * Una scheda si ridisegna SOLO quando cambia il suo stato (es. un controllo in più superato,
 * un token in più arrivato a React), mai a ogni frame: caricare una texture sulla GPU costa.
 * Le unità sono quelle della scena: `u` = pixel per unità (le misure dei font sono in unità,
 * così il testo ha la stessa grandezza su tutte le schede).
 */
import * as THREE from "three";

const PX = 64; // pixel per unità di scena: 20 unità → 1280 px
const FONT = "Inter, system-ui, sans-serif";
const MONO = '"JetBrains Mono", ui-monospace, monospace';
export const INK = { text: "#e2e8f0", muted: "#94a3b8", dim: "#64748b", ok: "#34d399", bad: "#f87171", panel: "rgba(15,23,42,0.94)", line: "#334155" };

export class Card {
  /** @param {{w: number, h: number, color?: string}} slot dimensioni nel mondo */
  constructor({ w, h, color = "#38bdf8" }) {
    this.w = w;
    this.h = h;
    this.color = color;
    this.canvas = document.createElement("canvas");
    this.canvas.width = Math.round(w * PX);
    this.canvas.height = Math.round(h * PX);
    this.ctx = this.canvas.getContext("2d");
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 4;
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthWrite: false }));
    this.key = null;
  }

  /** Ridisegna solo se `key` (una stringa che riassume lo stato) è cambiata. */
  draw(key, painter) {
    if (key === this.key) return;
    this.key = key;
    const { ctx, canvas } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(PX, PX); // da qui in poi si disegna in unità di scena
    frame(ctx, this.w, this.h, this.color);
    painter(new Pen(ctx, this.w, this.h, this.color));
    this.texture.needsUpdate = true;
  }

  dispose() {
    this.texture.dispose();
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
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

function frame(ctx, w, h, color) {
  roundRect(ctx, 0.06, 0.06, w - 0.12, h - 0.12, 0.35);
  ctx.fillStyle = INK.panel;
  ctx.fill();
  ctx.lineWidth = 0.08;
  ctx.strokeStyle = color;
  ctx.stroke();
}

/** Penna con un cursore verticale: scrive righe una sotto l'altra, andando a capo da sola. */
export class Pen {
  constructor(ctx, w, h, color) {
    this.ctx = ctx;
    this.w = w;
    this.h = h;
    this.color = color;
    this.pad = 0.55;
    this.y = 0.45;
  }

  get bottom() {
    return this.h - this.pad * 0.7;
  }

  get width() {
    return this.w - 2 * this.pad;
  }

  font(size, { weight = 400, mono = false } = {}) {
    this.ctx.font = `${weight} ${size}px ${mono ? MONO : FONT}`;
  }

  /** Titolo della scheda con il sottotitolo a destra. */
  header(title, subtitle = "") {
    const { ctx } = this;
    ctx.textBaseline = "top";
    this.font(0.95, { weight: 700 });
    ctx.fillStyle = this.color;
    ctx.fillText(title, this.pad, this.y);
    if (subtitle) {
      this.font(0.52, { mono: true });
      ctx.fillStyle = INK.muted;
      ctx.textAlign = "right";
      ctx.fillText(subtitle, this.w - this.pad, this.y + 0.28);
      ctx.textAlign = "left";
    }
    this.y += 1.35;
    ctx.fillStyle = INK.line;
    ctx.fillRect(this.pad, this.y - 0.18, this.width, 0.04);
  }

  /** Spezza `text` in righe larghe al massimo `width` (rispetta gli a capo). */
  lines(text, width = this.width) {
    const out = [];
    for (const paragraph of String(text).split("\n")) {
      let line = "";
      for (const word of paragraph.split(/(\s+)/)) {
        if (!word) continue;
        const test = line + word;
        if (this.ctx.measureText(test).width > width && line.trim()) {
          out.push(line.trimEnd());
          line = word.trimStart();
        } else line = test;
      }
      out.push(line);
    }
    return out;
  }

  /**
   * Scrive un paragrafo. Restituisce false se il testo non è entrato tutto (ultima riga con "…").
   * @param {object} [o] size, color, mono, weight, x (rientro), maxLines, gap (spazio dopo)
   */
  text(text, { size = 0.62, color = INK.text, mono = false, weight = 400, x = 0, width = this.width - x, maxLines = Infinity, gap = 0.2, limit = this.bottom } = {}) {
    const { ctx } = this;
    this.font(size, { weight, mono });
    ctx.fillStyle = color;
    ctx.textBaseline = "top";
    const lh = size * 1.28;
    const lines = this.lines(text, width);
    let complete = true;
    for (let i = 0; i < lines.length; i++) {
      const last = i === maxLines - 1 || this.y + 2 * lh > limit;
      let line = lines[i];
      if (last && i < lines.length - 1) {
        while (line && ctx.measureText(`${line}…`).width > width) line = line.slice(0, -1);
        line = `${line}…`;
        complete = false;
      }
      if (this.y + lh > limit + 0.01) {
        complete = false;
        break;
      }
      ctx.fillText(line, this.pad + x, this.y);
      this.y += lh;
      if (last && i < lines.length - 1) break;
    }
    this.y += gap;
    return complete;
  }

  /** Riga con un segno di spunta (ok) o una croce (fallito): disegnati, non caratteri. */
  check(label, detail, ok, { size = 0.62 } = {}) {
    const { ctx } = this;
    const cx = this.pad + 0.3;
    const cy = this.y + size * 0.55;
    ctx.lineWidth = 0.1;
    ctx.lineCap = "round";
    ctx.strokeStyle = ok ? INK.ok : INK.bad;
    ctx.beginPath();
    if (ok) {
      ctx.moveTo(cx - 0.22, cy);
      ctx.lineTo(cx - 0.05, cy + 0.18);
      ctx.lineTo(cx + 0.25, cy - 0.2);
    } else {
      ctx.moveTo(cx - 0.18, cy - 0.18);
      ctx.lineTo(cx + 0.18, cy + 0.18);
      ctx.moveTo(cx + 0.18, cy - 0.18);
      ctx.lineTo(cx - 0.18, cy + 0.18);
    }
    ctx.stroke();
    this.font(size, { weight: 600 });
    ctx.fillStyle = INK.text;
    ctx.fillText(label, this.pad + 0.8, this.y);
    const lw = ctx.measureText(label).width;
    this.font(size * 0.9, { mono: true });
    ctx.fillStyle = INK.muted;
    let d = detail;
    const room = this.width - 0.8 - lw - 0.4;
    while (d && ctx.measureText(d).width > room) d = d.slice(0, -2) + "…";
    ctx.fillText(d, this.pad + 0.8 + lw + 0.4, this.y + size * 0.08);
    this.y += size * 1.45;
  }

  /** Fascia colorata piena larghezza con testo (es. esito del guardrail). */
  band(text, color, { size = 0.6, height = 1.1 } = {}) {
    const { ctx } = this;
    roundRect(ctx, this.pad, this.y, this.width, height, 0.2);
    ctx.fillStyle = `${color}33`;
    ctx.fill();
    ctx.lineWidth = 0.05;
    ctx.strokeStyle = color;
    ctx.stroke();
    this.font(size, { weight: 700 });
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(text, this.pad + 0.35, this.y + height / 2);
    ctx.textBaseline = "top";
    this.y += height + 0.25;
  }

  /** Barra orizzontale: valore in [0, max], con una tacca opzionale (es. la soglia). */
  bar(x, y, w, h, value, max, color, mark = null) {
    const { ctx } = this;
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, Math.max(0.03, (w * Math.min(value, max)) / max), h);
    if (mark !== null) {
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(x + (w * mark) / max - 0.025, y - 0.12, 0.05, h + 0.24);
    }
  }

  /** "Pillola" con testo: restituisce la larghezza occupata. */
  chip(x, y, text, color, { size = 0.5, mono = true, fill = true } = {}) {
    const { ctx } = this;
    this.font(size, { mono });
    const w = ctx.measureText(text).width + 0.4;
    roundRect(ctx, x, y, w, size * 1.5, 0.15);
    ctx.fillStyle = fill ? `${color}2e` : "transparent";
    ctx.fill();
    ctx.lineWidth = 0.04;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.fillStyle = INK.text;
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 0.2, y + size * 0.78);
    ctx.textBaseline = "top";
    return w;
  }

  /** Fascia di una parte del contesto: barra del colore del ruolo a sinistra, titolo, token e anteprima. */
  part({ title, tokens, preview, color, lines = 2, dim = false, size = 0.5 }) {
    const { ctx } = this;
    const top = this.y;
    this.font(0.58, { weight: 700 });
    ctx.globalAlpha = dim ? 0.28 : 1;
    ctx.fillStyle = color;
    ctx.fillText(title, this.pad + 0.45, this.y);
    this.font(0.5, { mono: true });
    ctx.fillStyle = INK.muted;
    ctx.textAlign = "right";
    ctx.fillText(tokens, this.w - this.pad, this.y + 0.06);
    ctx.textAlign = "left";
    this.y += 0.78;
    if (preview) this.text(preview, { size, mono: true, color: INK.text, x: 0.45, maxLines: lines, gap: 0 });
    ctx.globalAlpha = dim ? 0.28 : 1;
    ctx.fillStyle = color;
    ctx.fillRect(this.pad, top, 0.16, this.y - top);
    ctx.globalAlpha = 1;
    this.y += 0.3;
  }

  /** Testo in basso nella scheda (nota, legenda). */
  footer(text, { size = 0.5, color = INK.muted } = {}) {
    const lines = this.lines(text);
    this.font(size);
    this.ctx.fillStyle = color;
    this.ctx.textBaseline = "top";
    lines.slice(-2).forEach((line, i, arr) => this.ctx.fillText(line, this.pad, this.h - this.pad * 0.7 - (arr.length - i) * size * 1.3));
  }
}

/** Testo mostrato senza a capo e spazi multipli (anteprime compatte). */
export const compact = (text) => String(text).replace(/\s+/g, " ").trim();
