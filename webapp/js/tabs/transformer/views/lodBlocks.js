/**
 * Rettangoli del LOD leggibili da lontano.
 *
 * `makeBlock` (core/diagram.js) disegna il testo con una dimensione fissa rispetto alla
 * propria unità (titolo ≈ 0,42 unità). Per i livelli "Blocchi" e "Panoramica" servono
 * scritte molto più grandi, perché la camera è lontana: qui costruiamo il blocco piccolo
 * e poi lo ingrandiamo in modo uniforme (scala `s`), così il testo cresce con lui.
 *
 * `fitBlock` sceglie la scala più grande (fino a `maxScale`) per cui TUTTO il testo entra nel
 * rettangolo: ogni parola del titolo e della descrizione nella larghezza, il sottotitolo su una
 * riga e tutte le righe della descrizione nell'altezza (makeBlock altrimenti le taglierebbe in
 * silenzio). La simulazione usa gli stessi font e le stesse proporzioni di makeBlock.
 */
import * as THREE from "three";
import { makeBlock } from "../../../core/diagram.js";

// Costanti di core/diagram.js (makeBlock): se cambiano là, vanno aggiornate qui.
const PX_PER_UNIT = 96;
const MAX_TEXTURE = 2048;
const TITLE_FONT = (px) => `700 ${px}px Inter, system-ui, sans-serif`;
const SUBTITLE_FONT = (px) => `500 ${px}px "JetBrains Mono", ui-monospace, monospace`;
const BODY_FONT = (px) => `400 ${px}px Inter, system-ui, sans-serif`;
const MIN_SCALE = 0.35;
const SCALE_STEP = 0.94; // a ogni tentativo la scala scende del 6%

let measureCtx = null;
function textWidth(text, font) {
  measureCtx ??= document.createElement("canvas").getContext("2d");
  measureCtx.font = font;
  return measureCtx.measureText(text).width;
}

/** Numero di righe di `text` a capo come wrapLines di makeBlock, o Infinity se una parola non entra. */
function lineCount(text, font, maxWidth) {
  let lines = 0;
  for (const paragraph of String(text).split("\n")) {
    let line = "";
    for (const word of paragraph.split(/\s+/)) {
      if (textWidth(word, font) > maxWidth) return Infinity;
      const test = line ? `${line} ${word}` : word;
      if (line && textWidth(test, font) > maxWidth) {
        lines++;
        line = word;
      } else line = test;
    }
    lines++;
  }
  return lines;
}

/** Il testo di un makeBlock grande `w`×`h` unità (non scalato) entra tutto? Stessa impaginazione di makeBlock. */
export function textFits({ title, subtitle = "", text = "" }, w, h) {
  const px = Math.min(PX_PER_UNIT, MAX_TEXTURE / Math.max(w, h));
  const cw = Math.round(w * px);
  const ch = Math.round(h * px);
  const pad = 0.08 * px + 6;
  const inner = cw - 2 * pad;
  const titlePx = Math.max(14, Math.min(px * 0.42, ch * 0.2));
  let y = pad + lineCount(title, TITLE_FONT(titlePx), inner) * titlePx * 1.15;
  if (subtitle) {
    if (textWidth(subtitle, SUBTITLE_FONT(titlePx * 0.62)) > inner) return false;
    y += titlePx * 0.62 * 1.4 + 2;
  }
  if (text) {
    const bodyPx = titlePx * 0.6;
    const lines = lineCount(text, BODY_FONT(bodyPx), inner);
    y += bodyPx * 0.3 + (lines - 1) * bodyPx * 1.3 + bodyPx; // fondo dell'ultima riga
  }
  return Number.isFinite(y) && y <= ch - pad;
}

/**
 * @param {object} o opzioni di makeBlock (title, subtitle, text, color, fill) +
 * @param {number} o.width larghezza nel mondo @param {number} o.height altezza nel mondo
 * @param {number} [o.maxScale=3] ingrandimento massimo del testo
 * @param {number} [o.depth=0.2] spessore nel mondo
 * @param {number} [o.bodyOpacity] opacità del corpo (makeBlock usa 0,85)
 * @returns {THREE.Group} con userData.block, userData.scale, userData.face (il piano con il testo)
 */
export function fitBlock({ width, height, maxScale = 3, depth = 0.2, bodyOpacity, ...rest }) {
  let scale = maxScale;
  while (scale * SCALE_STEP >= MIN_SCALE && !textFits(rest, width / scale, height / scale)) scale *= SCALE_STEP;
  const group = makeBlock({ ...rest, width: width / scale, height: height / scale, depth: depth / scale });
  group.scale.setScalar(scale);
  const [body, face] = group.children;
  if (bodyOpacity !== undefined) body.material.opacity = bodyOpacity;
  group.userData.scale = scale;
  group.userData.face = face;
  group.userData.body = body;
  return group;
}

/** Contorno rettangolare (per evidenziare il blocco attivo): una sola LineSegments riusabile. */
export function makeOutline(color = 0x67e8f9) {
  const geometry = new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1));
  const outline = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 1 }));
  outline.visible = false;
  outline.userData.noBounds = true;
  return outline;
}

/** Porta il contorno attorno a un blocco di fitBlock (in coordinate del genitore comune). */
export function wrapOutline(outline, block, pad = 0.25) {
  const { width, height, depth } = block.userData.block;
  const s = block.userData.scale ?? 1;
  outline.position.set(block.position.x, block.position.y, block.position.z + (depth * s) / 2 + 0.05);
  outline.scale.set(width * s + pad, height * s + pad, 1);
  outline.visible = true;
}

/** Libera geometrie, materiali e texture di un sottoalbero (quando una vista viene ricostruita). */
export function disposeTree(object) {
  object.traverse((child) => {
    child.geometry?.dispose?.();
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    for (const m of materials) {
      m.map?.dispose?.();
      m.dispose?.();
    }
  });
}
