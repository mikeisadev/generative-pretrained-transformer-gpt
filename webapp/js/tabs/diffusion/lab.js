/**
 * "Banco di lavoro": un'area sopra la pipeline dove le azioni mettono immagini, latenti
 * e formule ingranditi. Ogni azione costruisce i propri pannelli all'avvio e li libera
 * alla fine con clear() (geometrie, materiali e texture vengono rilasciati).
 */
import * as THREE from "three";
import { ChannelTexture, DynamicLabel } from "./textures.js";

/** Direzione della camera sul banco: quasi di fronte, così testi e immagini restano leggibili. */
const LAB_DIR = new THREE.Vector3(0.06, 0.08, 1).normalize();

export class Lab {
  /** @param {number} textScale moltiplica tutte le altezze dei testi (leggibilità da lontano) */
  constructor({ textScale = 1 } = {}) {
    this.textScale = textScale;
    this.group = new THREE.Group();
    this.group.visible = false;
    this.items = new Map();
    this.owned = []; // texture create per l'azione corrente, da liberare in clear()
  }

  /**
   * Pannello con texture e didascalia sopra.
   * @returns {{mesh: THREE.Mesh, title: DynamicLabel}}
   */
  panel(key, { texture, x, y, w, h = w, title = "", titleHeight = 0.5, below = false, frame = 0x475569 }) {
    titleHeight *= this.textScale;
    const geo = new THREE.PlaneGeometry(w, h);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }));
    mesh.position.set(x, y, 0);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: frame }));
    edges.userData.noBounds = true;
    mesh.add(edges);
    const label = new DynamicLabel(title, { height: titleHeight });
    const offset = h / 2 + titleHeight * 1.1 * Math.max(1, title.split("\n").length) * 0.55 + 0.25;
    label.sprite.position.set(x, below ? y - offset : y + offset, 0.05);
    this.group.add(mesh, label.sprite);
    const item = { mesh, title: label, edges };
    this.items.set(key, item);
    return item;
  }

  /** Testo libero (formula, nota). */
  text(key, text, x, y, options = {}) {
    const label = new DynamicLabel(text, { ...options, height: (options.height ?? 0.5) * this.textScale });
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

// ---------------------------------------------------------------- aiuti per le azioni

/** Quattro canali di un latente in una griglia 2 × 2 nel banco di lavoro. */
export function latentGrid(lab, key, x, y, w, title, titleHeight = 0.85) {
  const textures = Array.from({ length: 4 }, () => lab.own(new ChannelTexture(32)));
  const half = w / 2 + 0.15;
  textures.forEach((t, c) => {
    lab.panel(`${key}${c}`, { texture: t.texture, x: x + (c % 2 ? half / 2 + 0.1 : -half / 2 - 0.1), y: y + (c < 2 ? half / 2 + 0.1 : -half / 2 - 0.1), w: w / 2, title: "" });
  });
  lab.text(`${key}-title`, title, x, y + w / 2 + 0.6 + titleHeight * title.split("\n").length * 0.6, { height: titleHeight });
  return { set: (z) => textures.forEach((t, c) => t.set(z, c)) };
}

/** Inquadra il banco di lavoro lasciando spazio in basso alla barra del player. */
export function focusLab(tab, margin = 1.05) {
  // rimandato: all'avvio di un'azione il pannello del codice si apre subito dopo e cambia lo spazio visibile
  return new Promise((resolve) =>
    setTimeout(() => {
      const box = tab.lab.bounds();
      box.min.y -= (box.max.y - box.min.y) * 0.3;
      tab.ui.director.focus(box, { showFrame: false, margin, direction: LAB_DIR }).then(resolve);
    }, 0),
  );
}

/** Porta la camera sul banco solo se non c'è già (serve quando si salta tra i passi). */
export function ensureLab(tab, ctx) {
  if (ctx.data.view === "lab") return;
  ctx.data.view = "lab";
  focusLab(tab);
}
