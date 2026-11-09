/**
 * Contenitori della scena (three.js): tutte le "vaschette" dei moduli in DUE oggetti per livello del
 * LOD, un mesh per le velature e un LineSegments per i bordi, invece di centinaia di oggetti. Con il
 * rendering software ogni disegno in più costa: così i contenitori sono 2 disegni.
 *
 * Ogni contenitore è un rettangolo nel piano (x, y) alla sua z, con un bordo davanti e uno un poco
 * dietro uniti agli angoli: girando la camera si vede lo spessore, e i livelli di annidamento (ogni
 * livello un poco più avanti) si leggono a strati.
 */
import * as THREE from "three";

const DEPTH = 0.03; // spessore della vaschetta

const _c = new THREE.Color();

/**
 * @param {{x0: number, x1: number, y0: number, y1: number, z: number, color: string, alpha?: number, edge?: number}[]} rects
 *   `alpha`: opacità della velatura (0 = solo bordo); `edge`: luminosità del bordo (0…1)
 * @param {{renderOrder?: number, opaque?: boolean}} [o] `opaque`: velature piene (il colore è già quello
 *   finale). Perché: con il rendering software ogni velatura trasparente che copre lo schermo costa
 *   ~14 ms (misurato: 7 livelli annidati = ~100 ms per frame); piene e disegnate dalla più vicina alla più
 *   lontana, il test di profondità scarta subito i pixel coperti e ogni pixel si colora una volta sola.
 * @returns {THREE.Group} con `userData.fills` (Mesh) e `userData.lines` (LineSegments)
 */
export function buildFrames(rects, { renderOrder = 0, opaque = false } = {}) {
  const group = new THREE.Group();
  const filled = rects.filter((r) => (r.alpha ?? 0) > 0).sort((a, b) => (opaque ? b.z - a.z : 0));
  if (filled.length) {
    const pos = new Float32Array(filled.length * 12);
    const col = new Float32Array(filled.length * 16);
    const index = new Uint32Array(filled.length * 6);
    filled.forEach((r, i) => {
      pos.set([r.x0, r.y0, r.z, r.x1, r.y0, r.z, r.x1, r.y1, r.z, r.x0, r.y1, r.z], i * 12);
      _c.set(r.color);
      for (let k = 0; k < 4; k++) col.set([_c.r, _c.g, _c.b, opaque ? 1 : r.alpha], i * 16 + k * 4);
      index.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(col, 4));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.computeBoundingSphere();
    const material = opaque
      ? new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, toneMapped: false })
      : new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    const fills = new THREE.Mesh(geometry, material);
    fills.renderOrder = renderOrder;
    fills.raycast = () => {};
    group.add(fills);
    group.userData.fills = fills;
  }
  // Bordi: davanti (z), dietro (z − DEPTH) e i 4 spigoli che li uniscono: 12 segmenti per contenitore.
  const pos = new Float32Array(rects.length * 12 * 6);
  const col = new Float32Array(rects.length * 12 * 6);
  rects.forEach((r, i) => {
    const zf = r.z + 0.002;
    const zb = r.z - DEPTH;
    const corners = [
      [r.x0, r.y0],
      [r.x1, r.y0],
      [r.x1, r.y1],
      [r.x0, r.y1],
    ];
    const segs = [];
    for (let k = 0; k < 4; k++) {
      const [ax, ay] = corners[k];
      const [bx, by] = corners[(k + 1) % 4];
      segs.push([ax, ay, zf, bx, by, zf], [ax, ay, zb, bx, by, zb], [ax, ay, zf, ax, ay, zb]);
    }
    _c.set(r.color).multiplyScalar(r.edge ?? 0.75);
    segs.forEach((s, k) => {
      pos.set(s, (i * 12 + k) * 6);
      // il bordo dietro e gli spigoli più scuri: il davanti si legge, lo spessore si intuisce
      const dim = k % 3 === 0 ? 1 : 0.45;
      col.set([_c.r * dim, _c.g * dim, _c.b * dim, _c.r * dim, _c.g * dim, _c.b * dim], (i * 12 + k) * 6);
    });
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geometry.computeBoundingSphere();
  const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, toneMapped: false }));
  lines.renderOrder = renderOrder + 1;
  lines.raycast = () => {};
  group.add(lines);
  group.userData.lines = lines;
  return group;
}

/** Libera geometrie e materiali dei contenitori. */
export function disposeFrames(group) {
  group.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  group.removeFromParent();
}

/**
 * Frecce piatte del flusso (Panoramica): un triangolo per punta e un rettangolo per asta, tutte in un mesh.
 * @param {{x0: number, x1: number, y: number, z: number}[]} arrows da sinistra a destra
 */
export function buildArrows(arrows, { color = "#67e8f9", width = 0.5, head = 1.4 } = {}) {
  const pos = [];
  for (const a of arrows) {
    const len = a.x1 - a.x0;
    if (len <= 0.1) continue;
    const hl = Math.min(head, len * 0.6);
    const xb = a.x1 - hl;
    const w = width / 2;
    // asta (due triangoli) + punta
    pos.push(a.x0, a.y - w, a.z, xb, a.y - w, a.z, xb, a.y + w, a.z, a.x0, a.y - w, a.z, xb, a.y + w, a.z, a.x0, a.y + w, a.z);
    pos.push(xb, a.y - w * 2.6, a.z, a.x1, a.y, a.z, xb, a.y + w * 2.6, a.z);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
  geometry.computeBoundingSphere();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }));
  mesh.raycast = () => {};
  mesh.userData.noBounds = true;
  return mesh;
}
