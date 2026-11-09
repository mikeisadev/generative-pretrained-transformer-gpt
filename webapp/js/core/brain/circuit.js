/**
 * Il microcircuito del tab in 3D: i gruppi di neuroni in colonne da sinistra a destra (circuitLayout.js),
 * i neuroni modellati (neurons.js), le fibre colorate per tipo di sinapsi e, per ogni colonna, le linee
 * che la legano alla sua regione sul cervello in miniatura ("ingrandimento": da dove viene quel pezzo).
 *
 *   const circuit = new CircuitScene(spec.circuit, { quality });
 *   circuit.root            // il gruppo da mettere nella scena (centrato nell'origine)
 *   circuit.labelItems()    // etichette delle colonne (nome del gruppo e componente AI)
 *   circuit.pickCell(x, y, camera, w, h) → la cellula (e la sua colonna) sotto il puntatore
 *   circuit.setLinks(points) // punti delle regioni sul cervello (coordinate mondo): linee di collegamento
 */
import * as THREE from "three";
import { FIRE_STEP, layoutCircuit } from "./circuitLayout.js";
import { NeuronField } from "./neurons.js";

export class CircuitScene {
  /**
   * @param {object} circuit circuito normalizzato (spec.js)
   * @param {{ quality?: "high" | "low" }} [o]
   */
  constructor(circuit, { quality = "high" } = {}) {
    const t0 = performance.now();
    this.circuit = circuit;
    this.layout = layoutCircuit(circuit, { detail: quality === "low" ? 0.45 : 1 });
    const { box } = this.layout;
    // centrato nell'origine: la vista sposta il gruppo, non le coordinate dei neuroni
    this.offset = new THREE.Vector3(-(box.min[0] + box.max[0]) / 2, -(box.min[1] + box.max[1]) / 2, 0);
    this.root = new THREE.Group();
    this.root.name = "circuit";
    this.inner = new THREE.Group();
    this.inner.position.copy(this.offset);
    this.root.add(this.inner);
    // un giro di animazione: tutti i gruppi scaricano in ordine, poi una pausa
    this.period = (this.layout.columns.length + 1.6) * FIRE_STEP;
    this.field = new NeuronField(this.layout, { quality, period: this.period });
    this.inner.add(this.field.root);
    // linee di collegamento colonna → regione sul cervello (aggiornate da setLinks)
    const linkGeometry = new THREE.BufferGeometry();
    linkGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.layout.columns.length * 6), 3));
    this.links = new THREE.LineSegments(linkGeometry, new THREE.LineBasicMaterial({ color: "#7dd3fc", transparent: true, opacity: 0.32, depthWrite: false }));
    this.links.userData.noBounds = true;
    this.links.frustumCulled = false;
    this.links.visible = false;
    this.buildMs = performance.now() - t0;
  }

  /** Box del circuito nelle coordinate del suo gruppo `root` (centrato). */
  localBox(target = new THREE.Box3()) {
    const { box } = this.layout;
    target.min.set(box.min[0], box.min[1], box.min[2]).add(this.offset);
    target.max.set(box.max[0], box.max[1], box.max[2]).add(this.offset);
    return target;
  }

  /** Box di una colonna (coordinate mondo). */
  columnBox(id, target = new THREE.Box3()) {
    const c = this.layout.columns.find((col) => col.id === id);
    target.makeEmpty();
    if (!c) return target;
    target.min.set(c.left, c.bottom, -c.depth / 2 - 0.4).add(this.offset);
    target.max.set(c.right, c.top, c.depth / 2 + 0.4).add(this.offset);
    this.root.updateMatrixWorld();
    return target.applyMatrix4(this.root.matrixWorld);
  }

  /** Punto in cima a una colonna (coordinate mondo): lì l'etichetta e la partenza della linea verso la regione. */
  columnTop(column, target = new THREE.Vector3()) {
    this.root.updateMatrixWorld();
    return target.set(column.x, column.top + 0.12, 0).add(this.offset).applyMatrix4(this.root.matrixWorld);
  }

  /** Punto sotto una colonna (coordinate mondo). */
  columnBottom(column, target = new THREE.Vector3()) {
    this.root.updateMatrixWorld();
    return target.set(column.x, column.bottom - 0.08, 0).add(this.offset).applyMatrix4(this.root.matrixWorld);
  }

  /** Etichette delle colonne: il nome del gruppo, sopra o sotto la colonna. */
  labelItems() {
    // sopra e sotto a colonne alterne: dieci nomi in fila non entrerebbero; il componente AI sta
    // nella scheda e nel suggerimento (qui solo per la colonna scelta, vedi la vista)
    return this.layout.columns.map((c) => ({
      id: `col:${c.id}`,
      text: shortLabel(c.label),
      sub: "",
      ai: c.ai ? `AI: ${clip(c.ai, 52)}` : "",
      color: c.color,
      anchors: [(c.index % 2 ? this.columnBottom(c) : this.columnTop(c)).toArray()],
      direction: c.index % 2 ? "down" : "up",
      priority: 1,
      data: { kind: "group", id: c.id },
    }));
  }

  /**
   * Linee dalla cima di ogni colonna alla sua regione sul cervello in miniatura.
   * @param {Map<string, number[]>} regionPoints id regione → punto (mondo), o null per nasconderle
   */
  setLinks(regionPoints) {
    if (!regionPoints) {
      this.links.visible = false;
      return;
    }
    const pos = this.links.geometry.attributes.position;
    const inv = new THREE.Matrix4().copy(this.links.parent?.matrixWorld ?? new THREE.Matrix4()).invert();
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    let n = 0;
    for (const c of this.layout.columns) {
      const target = (c.marker && regionPoints.get(c.marker)) || regionPoints.get(c.region);
      if (!target) continue;
      this.columnTop(c, a).applyMatrix4(inv);
      b.fromArray(target).applyMatrix4(inv);
      pos.setXYZ(n++, a.x, a.y, a.z);
      pos.setXYZ(n++, b.x, b.y, b.z);
    }
    pos.needsUpdate = true;
    this.links.geometry.setDrawRange(0, n);
    this.links.visible = n > 0;
  }

  /** La cellula più vicina al puntatore (entro `radius` px), proiettando i somi: niente raycast su migliaia di istanze. */
  pickCell(x, y, camera, width, height, radius = 22) {
    this.root.updateMatrixWorld();
    const v = new THREE.Vector3();
    let best = null;
    let bestD = radius;
    for (const cell of this.layout.cells) {
      v.set(cell.position[0], cell.position[1], cell.position[2]).add(this.offset).applyMatrix4(this.root.matrixWorld).project(camera);
      if (v.z > 1 || v.z < -1) continue;
      const sx = (v.x * 0.5 + 0.5) * width;
      const sy = (1 - (v.y * 0.5 + 0.5)) * height;
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestD) {
        bestD = d;
        best = cell;
      }
    }
    return best ? { cell: best, column: this.layout.columns[best.column] } : null;
  }

  setPlaying(on) {
    this.field.setPlaying(on);
  }

  setTime(t) {
    this.field.setTime(t);
  }

  dispose() {
    this.field.dispose();
    this.links.geometry.dispose();
    this.links.material.dispose();
    this.links.removeFromParent();
    this.root.removeFromParent();
  }
}

/** Etichetta breve di un gruppo ("Talamo: neuroni di relè (l'ingresso)" → "Talamo: neuroni di relè"). */
function shortLabel(label) {
  return clip(label.replace(/\s*\([^)]*\)\s*$/, ""), 38);
}

function clip(text, max) {
  return text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;
}
