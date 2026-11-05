/**
 * Regione "RoPE": quadranti con una freccia per posizione. Riga in alto: coppia di dimensioni
 * "veloce" (θ = 1 radiante per posizione); riga in basso: coppia "lenta" (θ = 0,1).
 * Sono le prime due frequenze di θ_i = 10000^(-2i/d) con d = 4 (RoFormer): numeri veri.
 */
import * as THREE from "three";
import { ropeFrequency } from "./sim.js";
import { PALETTE, Plate, text } from "./shapes.js";

const POSITIONS = 6;
const RADIUS = 1.75;
const dialX = (m) => -12.5 + m * 5;
const ROWS = [
  { y: 3.7, theta: ropeFrequency(0, 4), name: "coppia veloce: θ = 1 radiante per posizione" },
  { y: -3.6, theta: ropeFrequency(1, 4), name: "coppia lenta: θ = 0,01 radianti per posizione" },
];

/** Freccia 2D (cilindro + cono) con materiale proprio, che parte dall'origine lungo +x. */
function dialArrow(color) {
  const material = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.3 });
  const g = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, RADIUS - 0.45, 8), material);
  shaft.rotation.z = -Math.PI / 2;
  shaft.position.x = (RADIUS - 0.45) / 2;
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.24, 0.45, 12), material);
  head.rotation.z = -Math.PI / 2;
  head.position.x = RADIUS - 0.22;
  g.add(shaft, head);
  g.userData.material = material;
  return g;
}

export function buildRope(group) {
  const arrows = [];
  ROWS.forEach((row, r) => {
    text(group, row.name, 0, row.y + 3, { size: 0.48, color: PALETTE.muted });
    for (let m = 0; m < POSITIONS; m++) {
      const ring = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(Array.from({ length: 48 }, (_, k) => new THREE.Vector3(Math.cos((k / 48) * 2 * Math.PI) * RADIUS, Math.sin((k / 48) * 2 * Math.PI) * RADIUS, 0))),
        new THREE.LineBasicMaterial({ color: "#475569" }),
      );
      ring.position.set(dialX(m), row.y, 0);
      group.add(ring);
      const a = dialArrow(r === 0 ? "#c084fc" : "#f0abfc");
      a.position.set(dialX(m), row.y, 0.1);
      group.add(a);
      arrows.push({ arrow: a, m, theta: row.theta, row: r });
    }
  });
  for (let m = 0; m < POSITIONS; m++) text(group, `pos ${m}`, dialX(m), -6.6, { size: 0.45, color: PALETTE.muted });
  const note = new Plate({ w: 34, h: 3.2, size: 0.59, align: "center" }).place(group, 0, -9.9);

  const api = {
    /** Angolo di ogni freccia = (m + shift)·θ·grow. `focus` = posizioni da evidenziare. */
    set(shift = 0, focus = [], grow = 1) {
      for (const a of arrows) {
        a.arrow.rotation.z = (a.m + shift) * a.theta * (a.row === 1 ? 10 : 1) * grow; // vedi nota sotto
        const on = focus.includes(a.m);
        a.arrow.userData.material.emissiveIntensity = on ? 1 : 0.25;
        a.arrow.scale.setScalar(on ? 1.12 : 1);
      }
    },
    note(lines) {
      note.set(lines);
    },
  };
  // Nota: la riga lenta usa θ = 0,01 rad (d = 4); per renderla visibile la disegniamo 10 volte più
  // ampia, e la didascalia lo dichiara. La riga veloce è in scala reale.
  api.set(0, []);
  api.note([{ text: "Ogni coppia di dimensioni di q e k ruota di (posizione × θ).", bold: true }, { text: "Riga lenta disegnata con angoli × 10 per renderla visibile.", color: PALETTE.muted }]);
  return api;
}
