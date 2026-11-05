/**
 * Rettangoli del LOD leggibili da lontano.
 *
 * `makeBlock` (core/diagram.js) disegna il testo con una dimensione fissa rispetto alla
 * propria unità (titolo ≈ 0,42 unità). Per i livelli "Blocchi" e "Panoramica" servono
 * scritte molto più grandi, perché la camera è lontana: qui costruiamo il blocco piccolo
 * e poi lo ingrandiamo in modo uniforme (scala `s`), così il testo cresce con lui.
 *
 * `fitBlock` sceglie la scala più grande (fino a `maxScale`) per cui ogni parola del titolo,
 * il sottotitolo e la parola più lunga della descrizione entrano nella larghezza: niente
 * testo tagliato ai bordi. Le misure usano gli stessi font di makeBlock.
 */
import * as THREE from "three";
import { makeBlock } from "../../../core/diagram.js";

const PX_PER_UNIT = 96; // come in core/diagram.js
const TITLE_PX = 0.42 * PX_PER_UNIT;
const PAD_PX = 0.08 * PX_PER_UNIT + 6;

let measureCtx = null;
function textWidth(text, font) {
  measureCtx ??= document.createElement("canvas").getContext("2d");
  measureCtx.font = font;
  return measureCtx.measureText(text).width;
}

/** Larghezza (in unità del blocco non scalato) necessaria per la riga/parola più larga. */
function neededWidth({ title, subtitle = "", text = "" }) {
  const longest = (s) => String(s).split(/\s+/).filter(Boolean);
  const titleWords = String(title).split("\n").map((line) => line); // le righe del titolo non vanno spezzate a metà parola
  const widths = [
    ...titleWords.flatMap((line) => longest(line).map((w) => textWidth(w, `700 ${TITLE_PX}px Inter, system-ui, sans-serif`))),
    subtitle ? textWidth(subtitle, `500 ${TITLE_PX * 0.62}px "JetBrains Mono", ui-monospace, monospace`) : 0,
    ...longest(text).map((w) => textWidth(w, `400 ${TITLE_PX * 0.6}px Inter, system-ui, sans-serif`)),
  ];
  return (Math.max(...widths) + 2 * PAD_PX + 8) / PX_PER_UNIT;
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
  const need = neededWidth(rest);
  // Il titolo è alto al massimo il 20% del canvas: sotto 2,1 unità di altezza il testo rimpicciolisce.
  const scale = Math.max(0.35, Math.min(maxScale, width / need, height / 2.1));
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
