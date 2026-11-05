/**
 * Texture aggiornabili per immagini e latenti, ed etichette con testo modificabile.
 *
 * Ogni oggetto alloca i propri buffer una volta sola: durante le animazioni si riscrive
 * il contenuto e si segna `needsUpdate`, senza creare nuovi oggetti a ogni frame.
 */
import * as THREE from "three";

const clamp255 = (x) => (x < 0 ? 0 : x > 255 ? 255 : x | 0);

function dataTexture(width, height, nearest) {
  const data = new Uint8Array(width * height * 4);
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = nearest ? THREE.NearestFilter : THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.flipY = false;
  texture.needsUpdate = true;
  return { data, texture };
}

/** Texture di un'immagine RGB (valori in [−1, 1], righe dall'alto in basso). */
export class RGBTexture {
  constructor(size, { nearest = false } = {}) {
    this.size = size;
    Object.assign(this, dataTexture(size, size, nearest));
  }

  /** @param {Float32Array} rgb immagine intercalata size·size·3 */
  set(rgb) {
    const { size, data } = this;
    for (let j = 0; j < size; j++) {
      const row = (size - 1 - j) * size; // DataTexture parte dal basso
      for (let i = 0; i < size; i++) {
        const p = (j * size + i) * 3;
        const o = (row + i) * 4;
        data[o] = clamp255((rgb[p] + 1) * 127.5);
        data[o + 1] = clamp255((rgb[p + 1] + 1) * 127.5);
        data[o + 2] = clamp255((rgb[p + 2] + 1) * 127.5);
        data[o + 3] = 255;
      }
    }
    this.texture.needsUpdate = true;
    return this;
  }

  dispose() {
    this.texture.dispose();
  }
}

/**
 * Texture di UN canale di un tensore "channels first" con colormap divergente
 * (blu = negativo, scuro = zero, arancione = positivo), la stessa delle matrici di pesi.
 */
export class ChannelTexture {
  constructor(width, { height = width, range = 2.5 } = {}) {
    this.width = width;
    this.height = height;
    this.range = range;
    Object.assign(this, dataTexture(width, height, true));
  }

  /** @param {Float32Array} tensor [c][y][x] @param {number} channel */
  set(tensor, channel = 0) {
    const { width, height, data, range } = this;
    const n = width * height;
    for (let j = 0; j < height; j++) {
      const row = (height - 1 - j) * width;
      for (let i = 0; i < width; i++) {
        const v = tensor[channel * n + j * width + i] / range;
        const m = Math.pow(Math.min(1, Math.abs(v)), 0.7);
        const o = (row + i) * 4;
        if (v >= 0) {
          data[o] = clamp255(22 + (251 - 22) * m);
          data[o + 1] = clamp255(27 + (146 - 27) * m);
          data[o + 2] = clamp255(34 + (60 - 34) * m);
        } else {
          data[o] = clamp255(22 + (59 - 22) * m);
          data[o + 1] = clamp255(27 + (130 - 27) * m);
          data[o + 2] = clamp255(34 + (246 - 34) * m);
        }
        data[o + 3] = 255;
      }
    }
    this.texture.needsUpdate = true;
    return this;
  }

  dispose() {
    this.texture.dispose();
  }
}

/** Texture di una piccola matrice di valori qualsiasi (es. embedding del testo), stessa colormap. */
export function matrixTexture(rows, cols, valueAt) {
  const { data, texture } = dataTexture(cols, rows, true);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = Math.max(-1, Math.min(1, valueAt(r, c)));
      const m = Math.pow(Math.abs(v), 0.7);
      const o = ((rows - 1 - r) * cols + c) * 4;
      const tip = v >= 0 ? [251, 146, 60] : [59, 130, 246];
      data[o] = clamp255(22 + (tip[0] - 22) * m);
      data[o + 1] = clamp255(27 + (tip[1] - 27) * m);
      data[o + 2] = clamp255(34 + (tip[2] - 34) * m);
      data[o + 3] = 255;
    }
  }
  texture.needsUpdate = true;
  return texture;
}

/**
 * Etichetta (sprite) con testo che si può cambiare: ridisegna il proprio canvas solo quando il
 * testo cambia davvero. `lines` è una stringa con "\n" per andare a capo.
 */
export class DynamicLabel {
  constructor(text = "", { height = 0.5, color = "#e2e8f0", sub = "#94a3b8", maxWidth = 1400, align = "center" } = {}) {
    this.height = height;
    this.color = color;
    this.subColor = sub;
    this.maxWidth = maxWidth;
    this.align = align;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.texture, transparent: true, depthWrite: false }));
    this.sprite.renderOrder = 12;
    this.text = null;
    this.set(text);
  }

  /** Prima riga in evidenza, le altre più piccole e grigie. */
  set(text) {
    if (text === this.text) return this;
    this.text = text;
    const lines = String(text).split("\n");
    const px = 56;
    const { ctx, canvas } = this;
    const font = (i) => `${i === 0 ? 600 : 500} ${i === 0 ? px : px * 0.78}px Inter, system-ui, sans-serif`;
    let width = 1;
    lines.forEach((l, i) => {
      ctx.font = font(i);
      width = Math.max(width, ctx.measureText(l).width);
    });
    canvas.width = Math.ceil(Math.min(this.maxWidth * 2, width + 24));
    canvas.height = Math.ceil(px * 1.22 * lines.length + 10);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.textBaseline = "top";
    lines.forEach((l, i) => {
      ctx.font = font(i);
      ctx.fillStyle = i === 0 ? this.color : this.subColor;
      const w = ctx.measureText(l).width;
      const x = this.align === "left" ? 12 : (canvas.width - w) / 2;
      ctx.fillText(l, x, 5 + i * px * 1.22);
    });
    this.texture.dispose(); // la dimensione del canvas può cambiare: serve una nuova texture GPU
    this.texture = new THREE.CanvasTexture(canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.sprite.material.map = this.texture;
    this.sprite.material.needsUpdate = true;
    const h = this.height * lines.length * 1.1;
    this.sprite.scale.set((h * canvas.width) / canvas.height, h, 1);
    if (this.align === "left") this.sprite.center.set(0, 0.5);
    return this;
  }

  dispose() {
    this.texture.dispose();
    this.sprite.material.dispose();
  }
}
