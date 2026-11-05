/**
 * "Banco di lavoro": un'area sopra la pipeline dove le azioni mettono immagini, latenti
 * e formule ingranditi. Ogni azione costruisce i propri pannelli all'avvio e li libera
 * alla fine con clear() (geometrie, materiali e texture vengono rilasciati).
 */
import * as THREE from "three";
import { DynamicLabel } from "./textures.js";

export class Lab {
  constructor() {
    this.group = new THREE.Group();
    this.group.visible = false;
    this.items = new Map();
    this.owned = []; // texture create per l'azione corrente, da liberare in clear()
  }

  /**
   * Pannello con texture e didascalia sopra.
   * @returns {{mesh: THREE.Mesh, title: DynamicLabel}}
   */
  panel(key, { texture, x, y, w, h = w, title = "", titleHeight = 0.5, frame = 0x475569 }) {
    const geo = new THREE.PlaneGeometry(w, h);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }));
    mesh.position.set(x, y, 0);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: frame }));
    edges.userData.noBounds = true;
    mesh.add(edges);
    const label = new DynamicLabel(title, { height: titleHeight });
    label.sprite.position.set(x, y + h / 2 + titleHeight * 1.1 * Math.max(1, title.split("\n").length) * 0.55 + 0.25, 0.05);
    this.group.add(mesh, label.sprite);
    const item = { mesh, title: label, edges };
    this.items.set(key, item);
    return item;
  }

  /** Testo libero (formula, nota). */
  text(key, text, x, y, options = {}) {
    const label = new DynamicLabel(text, options);
    label.sprite.position.set(x, y, 0.05);
    this.group.add(label.sprite);
    this.items.set(key, { label });
    return label;
  }

  /** Una texture creata dall'azione: verrà liberata in clear(). */
  own(textureHolder) {
    this.owned.push(textureHolder);
    return textureHolder;
  }

  get(key) {
    return this.items.get(key);
  }

  /** Riquadro (invisibile) per la regia: inquadra tutto ciò che c'è nel banco. */
  bounds() {
    this.group.updateWorldMatrix(true, true);
    const box = new THREE.Box3();
    for (const item of this.items.values()) {
      const obj = item.mesh ?? item.label?.sprite;
      if (!obj) continue;
      if (item.mesh) box.union(new THREE.Box3().setFromObject(item.mesh));
      else box.expandByPoint(obj.getWorldPosition(new THREE.Vector3()));
    }
    return box;
  }

  show() {
    this.group.visible = true;
  }

  clear() {
    // le texture dei pannelli possono essere condivise (es. quella dell'immagine finale):
    // qui liberiamo solo geometrie e materiali; le texture dell'azione stanno in `owned`
    // (gli sprite condividono una geometria interna di three.js: non va liberata)
    this.group.traverse((child) => {
      if (child.isSprite) return;
      child.geometry?.dispose();
      child.material?.dispose();
    });
    for (const item of this.items.values()) (item.title ?? item.label)?.dispose();
    this.group.clear();
    this.items.clear();
    for (const t of this.owned) t.dispose();
    this.owned = [];
    this.group.visible = false;
  }
}

