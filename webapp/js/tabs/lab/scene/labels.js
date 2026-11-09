/**
 * Etichette della scena del Laboratorio in un atlante di testo (three.js + canvas 2D).
 *
 *   const atlas = new LabelAtlas();
 *   atlas.add({ key: "h:key", lines: [[{ text: "key · nn.Linear", font: "title", px: 32, color: "#38bdf8" }]] });
 *   const built = atlas.build();                       // pagine (canvas + texture), posto di ogni etichetta
 *   group.add(labelMesh(items, built));                // un mesh per pagina, quad di tutte le etichette
 *
 * Ogni etichetta è un riquadro di righe; ogni riga è una fila di segmenti (es. titolo e file:riga in
 * coda). Facoltativo un "piatto" dietro al testo (sfondo arrotondato con bordo) per le etichette grandi
 * dei livelli Moduli e Panoramica. Le etichette con la stessa chiave (lo stesso testo) si disegnano una
 * volta sola. Il testo è sempre disegnato con fillText: nessun HTML, quindi i nomi dei tensori caricati
 * dallo studente non possono iniettare niente.
 */
import * as THREE from "three";
import { packShelves } from "./pack.js";

export const FONTS = Object.freeze({
  title: (px) => `600 ${px}px Inter, system-ui, -apple-system, "Segoe UI", sans-serif`,
  big: (px) => `700 ${px}px Inter, system-ui, -apple-system, "Segoe UI", sans-serif`,
  sub: (px) => `500 ${px}px Inter, system-ui, -apple-system, "Segoe UI", sans-serif`,
  code: (px) => `${px}px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono", monospace`,
});

const LINE = 1.28; // interlinea
const PAD_PX = 6;
const MAX_LABEL_W = 2040;

function defaultCanvas() {
  return document.createElement("canvas");
}

/** Taglia un testo con "…" finché non sta in `maxW` pixel (solo per testi davvero troppo lunghi). */
function fit(ctx, text, maxW) {
  if (ctx.measureText(text).width <= maxW) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= maxW) lo = mid;
    else hi = mid - 1;
  }
  return `${text.slice(0, lo)}…`;
}

export class LabelAtlas {
  constructor({ pageWidth = 2048, maxPageHeight = 2048, createCanvas = defaultCanvas } = {}) {
    this.pageWidth = pageWidth;
    this.maxPageHeight = maxPageHeight;
    this.createCanvas = createCanvas;
    this.specs = new Map();
    this.scratch = createCanvas().getContext("2d");
    this.built = null;
  }

  /** Aggiunge un'etichetta (se la chiave c'è già, non la duplica). Restituisce la chiave. */
  add(spec) {
    if (this.specs.has(spec.key)) return spec.key;
    const ctx = this.scratch;
    let w = 0;
    let h = 0;
    const lines = spec.lines.map((segments) => {
      let lw = 0;
      let lh = 0;
      const segs = segments
        .filter((s) => s.text)
        .map((s) => {
          ctx.font = FONTS[s.font ?? "title"](s.px);
          const text = fit(ctx, String(s.text), MAX_LABEL_W - 2 * PAD_PX - lw);
          const sw = ctx.measureText(text).width + (s.gap ?? 0);
          lw += sw;
          lh = Math.max(lh, s.px * LINE);
          return { ...s, text, w: sw };
        });
      w = Math.max(w, lw);
      h += lh;
      return { segs, w: lw, h: lh };
    });
    const pad = spec.plate ? Math.round((spec.plate.pad ?? 0.6) * (spec.lines[0]?.[0]?.px ?? 24)) : PAD_PX;
    this.specs.set(spec.key, { ...spec, lines, w: Math.ceil(Math.min(MAX_LABEL_W, w + 2 * pad)), h: Math.ceil(h + 2 * pad * (spec.plate ? 0.8 : 0.5)), pad });
    return spec.key;
  }

  /** Impacchetta e disegna tutte le etichette: { pages: [{canvas, texture}], entries: Map(chiave → posto) }. */
  build() {
    const specs = [...this.specs.values()];
    const { pages, places } = packShelves(
      specs.map((s) => ({ w: s.w, h: s.h })),
      // 8 px di margine: con le mipmap le etichette vicine non si "sporcano" a vicenda da lontano
      { pageWidth: this.pageWidth, maxPageHeight: this.maxPageHeight, padding: 8 },
    );
    const canvases = pages.map((p) => {
      const canvas = this.createCanvas();
      canvas.width = p.width;
      canvas.height = p.height;
      return canvas;
    });
    const entries = new Map();
    specs.forEach((s, i) => {
      const { page, x, y } = places[i];
      const ctx = canvases[page].getContext("2d");
      draw(ctx, s, x, y);
      const { width: W, height: H } = pages[page];
      entries.set(s.key, { page, w: s.w, h: s.h, u0: x / W, u1: (x + s.w) / W, v1: 1 - y / H, v0: 1 - (y + s.h) / H, lines: s.lines.length, firstLine: s.lines[0]?.h ?? s.h });
    });
    const textures = canvases.map((canvas) => {
      const t = new THREE.CanvasTexture(canvas);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 4;
      t.minFilter = THREE.LinearMipmapLinearFilter; // da lontano il testo si sfuma invece di sfarfallare
      t.generateMipmaps = true;
      return t;
    });
    this.built = { pages: canvases.map((canvas, i) => ({ canvas, texture: textures[i] })), entries };
    return this.built;
  }

  dispose() {
    for (const p of this.built?.pages ?? []) {
      p.texture.dispose();
      p.canvas.width = 0; // libera subito la memoria del canvas
      p.canvas.height = 0;
    }
    this.built = null;
    this.specs.clear();
  }
}

function draw(ctx, s, x, y) {
  ctx.save();
  if (s.plate) {
    const r = Math.min(s.h / 3, 18);
    roundRect(ctx, x + 1.5, y + 1.5, s.w - 3, s.h - 3, r);
    ctx.fillStyle = s.plate.fill ?? "rgba(15, 23, 42, 0.92)";
    ctx.fill();
    if (s.plate.stroke) {
      ctx.lineWidth = 3;
      ctx.strokeStyle = s.plate.stroke;
      ctx.stroke();
    }
  }
  ctx.textBaseline = "middle";
  let yy = y + (s.plate ? s.pad * 0.8 : s.pad * 0.5);
  for (const line of s.lines) {
    let xx = s.align === "center" ? x + (s.w - line.w) / 2 : x + s.pad;
    for (const seg of line.segs) {
      ctx.font = FONTS[seg.font ?? "title"](seg.px);
      ctx.fillStyle = seg.color ?? "#e2e8f0";
      ctx.fillText(seg.text, xx, yy + line.h / 2);
      xx += seg.w;
    }
    yy += line.h;
  }
  ctx.restore();
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
 * I quad delle etichette di un livello: un mesh per pagina dell'atlante, con materiali propri (il LOD
 * ne sfuma l'opacità). `items`: { key, x, y, scale (mondo per pixel), anchor: "top-left" | "center" |
 * "top-center" | "bottom-center", maxW?, z }. Se l'etichetta è più larga di `maxW` si rimpicciolisce per starci.
 */
export function labelMesh(items, built, { renderOrder = 3 } = {}) {
  const group = new THREE.Group();
  const perPage = built.pages.map(() => []);
  for (const it of items) {
    const e = built.entries.get(it.key);
    if (!e) continue;
    let scale = it.scale;
    if (it.maxW && e.w * scale > it.maxW) scale = it.maxW / e.w;
    const w = e.w * scale;
    const h = e.h * scale;
    let x0 = it.x;
    let y1 = it.y;
    if (it.anchor === "center") {
      x0 -= w / 2;
      y1 += h / 2;
    } else if (it.anchor === "top-center") x0 -= w / 2;
    else if (it.anchor === "bottom-center") {
      x0 -= w / 2;
      y1 += h;
    }
    perPage[e.page].push({ x0, x1: x0 + w, y0: y1 - h, y1, z: it.z ?? 0, e });
    it.size = { w, h, x0, y1 };
  }
  perPage.forEach((quads, page) => {
    if (!quads.length) return;
    const pos = new Float32Array(quads.length * 12);
    const uv = new Float32Array(quads.length * 8);
    const index = new Uint32Array(quads.length * 6);
    quads.forEach((q, i) => {
      pos.set([q.x0, q.y0, q.z, q.x1, q.y0, q.z, q.x1, q.y1, q.z, q.x0, q.y1, q.z], i * 12);
      uv.set([q.e.u0, q.e.v0, q.e.u1, q.e.v0, q.e.u1, q.e.v1, q.e.u0, q.e.v1], i * 8);
      index.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.computeBoundingSphere();
    const material = new THREE.MeshBasicMaterial({ map: built.pages[page].texture, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2, toneMapped: false });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = renderOrder;
    mesh.raycast = () => {}; // i clic si risolvono con la disposizione (layout.hitTest), non sul testo
    mesh.userData.noBounds = true;
    group.add(mesh);
  });
  return group;
}

/** Libera geometrie e materiali di un gruppo di etichette (le texture sono dell'atlante). */
export function disposeLabelMesh(group) {
  group.traverse((o) => {
    if (o.isMesh) {
      o.geometry.dispose();
      o.material.dispose();
    }
  });
  group.removeFromParent();
}
