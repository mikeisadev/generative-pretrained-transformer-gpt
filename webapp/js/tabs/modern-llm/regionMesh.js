/**
 * Regione "Modelli 3D": una mesh scritta come testo OBJ, riga per riga, come in LLaMA-Mesh
 * (Wang et al. 2024: coordinate scalate in [0, 64] e arrotondate a interi, formato OBJ come
 * testo normale, nessun token nuovo). A destra la mesh si costruisce faccia dopo faccia.
 * La casetta è un esempio scritto a mano, non generato da un modello.
 */
import * as THREE from "three";
import { PALETTE, Plate } from "./shapes.js";

/** Casetta: 8 vertici del cubo + 2 del colmo del tetto, coordinate intere in [0, 64]. */
export const MESH_VERTICES = [
  [8, 0, 8], [56, 0, 8], [56, 0, 40], [8, 0, 40],
  [8, 32, 8], [56, 32, 8], [56, 32, 40], [8, 32, 40],
  [8, 52, 24], [56, 52, 24],
];
/** Facce (indici da 1 come in OBJ). */
export const MESH_FACES = [
  [1, 2, 6], [1, 6, 5], // davanti
  [2, 3, 7], [2, 7, 6], // destra
  [3, 4, 8], [3, 8, 7], // dietro
  [4, 1, 5], [4, 5, 8], // sinistra
  [5, 6, 10], [5, 10, 9], // falda davanti
  [7, 8, 9], [7, 9, 10], // falda dietro
  [6, 7, 10], [8, 5, 9], // timpani
];

export const OBJ_LINES = [...MESH_VERTICES.map((v) => `v ${v.join(" ")}`), ...MESH_FACES.map((f) => `f ${f.join(" ")}`)];

export function buildMesh(group) {
  const listing = new Plate({ w: 12.5, h: 19.4, size: 0.81, mono: true, background: "#0f172a", border: "#334155" }).place(group, -10.4, -0.6);
  const positions = new Float32Array(MESH_FACES.length * 9);
  MESH_FACES.forEach((f, i) =>
    f.forEach((vi, k) => {
      const v = MESH_VERTICES[vi - 1];
      positions.set([(v[0] - 32) / 6, v[1] / 6, (v[2] - 24) / 6], i * 9 + k * 3);
    }),
  );
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  geometry.setDrawRange(0, 0);
  const solid = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: "#fcd34d", roughness: 0.6, side: THREE.DoubleSide, flatShading: true }));
  const wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: "#78350f", wireframe: true }));
  const holder = new THREE.Group();
  holder.add(solid, wire);
  holder.position.set(7.2, -4.4, 0);
  holder.rotation.set(0.28, 0.65, 0);
  group.add(holder);
  // punti dei vertici: si accendono quando la riga "v" corrispondente è stata scritta
  const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.22, 10, 8), new THREE.MeshBasicMaterial({ color: "#f8fafc" }), MESH_VERTICES.length);
  const tmp = new THREE.Object3D();
  holder.add(dots);
  const note = new Plate({ w: 20, h: 3.6, size: 0.52, align: "center" }).place(group, 7.2, 7.4);
  const footer = new Plate({ w: 34, h: 1.3, size: 0.47, align: "center", color: PALETTE.muted }).place(group, 0, -11.6);
  footer.set("Altra strada: l'LLM scrive uno script Python per Blender (bpy) che costruisce la scena.");

  const api = {
    holder,
    /** n = righe OBJ scritte (0 … OBJ_LINES.length). */
    set(n) {
      if (n === api.n) return;
      api.n = n;
      const shown = OBJ_LINES.slice(0, n);
      const window = shown.slice(-24);
      listing.set(window.length ? window.map((line, i) => ({ text: line, color: line.startsWith("v") ? "#fde68a" : "#93c5fd", bold: i === window.length - 1 })) : [{ text: "(vuoto)", color: PALETTE.muted }]);
      const verts = Math.min(n, MESH_VERTICES.length);
      MESH_VERTICES.forEach((v, i) => {
        tmp.position.set((v[0] - 32) / 6, v[1] / 6, (v[2] - 24) / 6);
        tmp.scale.setScalar(i < verts ? 1 : 0.0001);
        tmp.updateMatrix();
        dots.setMatrixAt(i, tmp.matrix);
      });
      dots.instanceMatrix.needsUpdate = true;
      const faces = Math.max(0, n - MESH_VERTICES.length);
      geometry.setDrawRange(0, faces * 3);
      note.set([
        { text: `${verts} vertici, ${faces} facce di ${MESH_FACES.length}`, bold: true },
        { text: "«v x y z» = un vertice (interi 0-64)", color: "#fde68a" },
        { text: "«f a b c» = un triangolo tra 3 vertici", color: "#93c5fd" },
      ]);
    },
    /** Rotazione lenta per vedere la forma (chiamata a ogni frame dall'azione). */
    spin(dt) {
      holder.rotation.y += dt * 0.5;
    },
  };
  api.set(OBJ_LINES.length);
  return api;
}
