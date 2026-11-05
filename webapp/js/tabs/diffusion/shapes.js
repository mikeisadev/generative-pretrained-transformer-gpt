/**
 * Piccoli costruttori di forme usati dalle scene del tab (oltre a quelli del nucleo).
 * Ogni funzione crea materiali PROPRI: il LOD sfuma l'opacità per materiale.
 */
import * as THREE from "three";
import { label } from "../../core/visuals.js";

/** Palette del tab (coerente con il resto della web app). */
export const PALETTE = {
  text: 0xa78bfa, // testo e cross-attention (viola)
  resnet: 0xf59e0b, // ResNet block (arancione, pesi addestrati)
  attn: 0x8b5cf6, // blocchi di attention
  sample: 0x2dd4bf, // down/upsample
  time: 0x4ade80, // passo t
  noise: 0x94a3b8,
  vae: 0x38bdf8,
  flow: 0x67e8f9,
  skip: 0x22d3ee,
};

/** Parallelepipedo semplice con materiale proprio. */
export function box(w, h, d, color, { opacity = 1, emissive = 0x000000 } = {}) {
  return new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: 0.35, roughness: 0.55, transparent: opacity < 1, opacity }),
  );
}

/** Piano con texture (immagine o latente), con cornice sottile. */
export function texturedPlane(texture, w, h, { frame = 0x334155 } = {}) {
  const geo = new THREE.PlaneGeometry(w, h);
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }));
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: frame }));
  edges.userData.noBounds = true;
  mesh.add(edges);
  return mesh;
}

/**
 * Tante scatole uguali come un solo InstancedMesh (una draw call).
 * items: [{ x, y, z, sx, sy, sz, color }] → restituisce il mesh; userData.items conserva i dati.
 */
export function instancedBoxes(items, { opacity = 1 } = {}) {
  const mesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, transparent: opacity < 1, opacity }),
    items.length,
  );
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const c = new THREE.Color();
  items.forEach((it, i) => {
    m.compose(new THREE.Vector3(it.x, it.y, it.z ?? 0), q, new THREE.Vector3(it.sx, it.sy, it.sz));
    mesh.setMatrixAt(i, m);
    mesh.setColorAt(i, c.set(it.color));
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();
  mesh.userData.items = items;
  // Il calcolo dei bounding box della regia (meshBounds) non conosce le istanze: lo escludiamo
  // e aggiungiamo un "proxy" invisibile grande quanto l'insieme delle scatole.
  mesh.userData.noBounds = true;
  const bounds = new THREE.Box3();
  for (const it of items) {
    bounds.expandByPoint(new THREE.Vector3(it.x - it.sx / 2, it.y - it.sy / 2, (it.z ?? 0) - it.sz / 2));
    bounds.expandByPoint(new THREE.Vector3(it.x + it.sx / 2, it.y + it.sy / 2, (it.z ?? 0) + it.sz / 2));
  }
  mesh.add(boundsProxy(bounds));
  return mesh;
}

/** Mesh invisibile che occupa `box3`: serve solo alla regia per inquadrare gruppi "speciali". */
export function boundsProxy(box3) {
  const size = box3.getSize(new THREE.Vector3());
  const proxy = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(size.x, 0.01), Math.max(size.y, 0.01), Math.max(size.z, 0.01)),
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  box3.getCenter(proxy.position);
  proxy.raycast = () => {};
  return proxy;
}

/** Ricolora le istanze: color(i, item) → numero esadecimale. Nessuna allocazione. */
const tmpColor = new THREE.Color();
export function recolor(mesh, color) {
  mesh.userData.items.forEach((it, i) => mesh.setColorAt(i, tmpColor.set(color(i, it))));
  mesh.instanceColor.needsUpdate = true;
}

/** Segmenti di linea da coppie di punti [[a, b], …] in un solo LineSegments. */
export function lineSegments(pairs, color, { opacity = 0.6, maxSegments = pairs.length } = {}) {
  const positions = new Float32Array(Math.max(1, maxSegments) * 6);
  pairs.forEach(([a, b], i) => {
    positions.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setDrawRange(0, pairs.length * 2);
  const lines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  lines.userData.noBounds = true;
  lines.frustumCulled = false;
  return lines;
}

/** Etichetta sprite posizionata. */
export function placeLabel(parent, text, x, y, z = 0, options = {}) {
  const sprite = label(text, options);
  sprite.position.set(x, y, z);
  parent.add(sprite);
  return sprite;
}

/** Libera geometrie, materiali e texture di un sottoalbero. */
export function disposeTree(object) {
  object.traverse((child) => {
    if (!child.isSprite) child.geometry?.dispose?.(); // gli sprite condividono la geometria
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    for (const m of materials) {
      m.map?.dispose?.();
      m.dispose();
    }
  });
}
