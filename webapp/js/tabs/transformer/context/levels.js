/**
 * Livelli astratti dello zoom semantico per la vista "Contesto".
 *
 *   Blocchi    → un rettangolo per fase (titolo, sottotitolo, cosa fa) e frecce del flusso,
 *                negli stessi slot delle schede del Dettaglio.
 *   Panoramica → tre rettangoli grandi: Contesto (RAG) → [LLM: la pila] → Risposta,
 *                più il giro degli strumenti. Si affiancano ai rettangoli Input/Transformer/Output
 *                che lod.js disegna per la pila.
 *
 * Regola di core/lod.js: ogni livello ha materiali propri (l'opacità si sfuma sui materiali).
 */
import * as THREE from "three";
import { fitBlock } from "../views/lodBlocks.js";
import { SLOTS, STACK, connections } from "./layout.js";
import { MACRO, STAGES } from "./texts.js";

/**
 * Freccia lungo una polilinea: un tubo e una punta conica sull'ultimo tratto.
 * Un solo materiale per freccia (proprio, come chiede il LOD).
 */
export function polyArrow(points, { color = 0x67e8f9, radius = 0.12, head = 0.9, opacity = 1 } = {}) {
  const group = new THREE.Group();
  const material = new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity });
  const last = points.at(-1);
  const prev = points.at(-2);
  const dir = new THREE.Vector3().subVectors(last, prev);
  const len = dir.length();
  dir.normalize();
  const tip = last.clone().addScaledVector(dir, -Math.min(head, len * 0.6));
  const path = new THREE.CurvePath();
  const pts = [...points.slice(0, -1), tip];
  for (let i = 1; i < pts.length; i++) path.add(new THREE.LineCurve3(pts[i - 1], pts[i]));
  group.add(new THREE.Mesh(new THREE.TubeGeometry(path, Math.max(2, pts.length * 4), radius, 8, false), material));
  const cone = new THREE.Mesh(new THREE.ConeGeometry(radius * 3, Math.min(head, len * 0.6), 14), material);
  cone.position.copy(tip).addScaledVector(dir, Math.min(head, len * 0.6) / 2);
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  group.add(cone);
  group.traverse((o) => (o.userData.noBounds = true));
  return group;
}

/** Livello Blocchi: un fitBlock per fase + frecce. Restituisce anche i blocchi per id (per il contorno). */
export function buildBlocksLevel() {
  const group = new THREE.Group();
  group.name = "context-blocks";
  const blocks = new Map();
  for (const stage of STAGES) {
    const s = SLOTS[stage.id];
    const block = fitBlock({ width: s.w, height: s.h, maxScale: 4.2, title: stage.title, subtitle: stage.subtitle, text: stage.blocks, color: stage.color });
    block.position.set(s.x, s.y, 0);
    group.add(block);
    blocks.set(stage.id, block);
  }
  for (const c of connections()) group.add(polyArrow(c.points, { color: c.decorative ? 0x7dd3fc : 0x67e8f9, radius: c.decorative ? 0.1 : 0.16, head: 1.3, opacity: c.decorative ? 0.6 : 1 }));
  return { group, blocks };
}

/** Livello Panoramica: Contesto (RAG) → LLM → Risposta, con il giro degli strumenti. */
export function buildMacroLevel() {
  const group = new THREE.Group();
  group.name = "context-macro";
  const box = { x: -70, w: 56 };
  // Scale grandi: la Panoramica si guarda da 200-250 unità di distanza, il testo deve restare leggibile.
  const context = fitBlock({ width: box.w, height: 32, maxScale: 12, color: "#34d399", depth: 0.3, ...MACRO.context });
  context.position.set(box.x, -33, 0);
  const answer = fitBlock({ width: box.w, height: 20, maxScale: 12, color: "#38bdf8", depth: 0.3, ...MACRO.answer });
  answer.position.set(box.x, 41.5, 0);
  const loop = fitBlock({ width: 44, height: 13, maxScale: 9, color: "#fb7185", ...MACRO.loop });
  loop.position.set(box.x, 14, 0);
  group.add(context, answer, loop);

  const V = (x, y) => new THREE.Vector3(x, y, 0.5);
  const right = box.x + box.w / 2;
  const style = { color: 0x67e8f9, radius: 0.45, head: 2.6 };
  group.add(
    // il contesto entra nei token di input della pila (il rettangolo "Input" di lod.js)
    polyArrow([V(right + 0.3, STACK.inputY - 4), V(-18, STACK.inputY - 4)], style),
    // dall'uscita della pila alla risposta
    polyArrow([V(-18, 41.5), V(right + 0.3, 41.5)], style),
    // giro degli strumenti: la risposta chiede uno strumento, il risultato torna nel contesto
    polyArrow([V(box.x, 41.5 - 10.2), V(box.x, 14 + 6.8)], { ...style, color: 0xfb7185 }),
    polyArrow([V(box.x, 14 - 6.8), V(box.x, -33 + 16.2)], { ...style, color: 0xfb7185 }),
  );
  return { group };
}
