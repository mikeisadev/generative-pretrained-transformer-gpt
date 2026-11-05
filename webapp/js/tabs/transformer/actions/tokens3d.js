/**
 * Oggetti 3D dei token usati dall'azione "Dal prompt all'output":
 *   TokenTape     il testo completo (prompt + generati) come griglia di cubi con carattere e id
 *   StoiTable     la tabella stoi (carattere → id) del tokenizer, con i caratteri usati evidenziati
 *   ContextColumn la colonna dei token che il modello vede davvero (al massimo block_size)
 *   LinkLines     fili dai token alle righe delle tabelle di embedding (un solo LineSegments)
 *   RowMarks      evidenziatori delle righe lette nelle tabelle di embedding (InstancedMesh)
 *
 * Tante istanze uguali → InstancedMesh; i caratteri sono disegnati su UN canvas per oggetto,
 * ridisegnato solo quando cambia il contenuto. Le animazioni per frame aggiornano solo matrici
 * e attributi già allocati.
 */
import * as THREE from "three";
import { easeInOut, layoutText, visibleChar } from "./math.js";

const MONO = "'JetBrains Mono', ui-monospace, monospace";
const PX = 64; // pixel di canvas per unità di cella

/** Stati dei cubi → colore (il colore dell'istanza moltiplica il bianco del materiale). */
export const TOKEN_COLORS = {
  idle: new THREE.Color(0x334155),
  context: new THREE.Color(0x0e7490),
  outside: new THREE.Color(0x1e293b),
  generated: new THREE.Color(0xb45309),
  current: new THREE.Color(0x0ea5e9),
};

function makeLabelPlane(widthUnits, heightUnits) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(8, Math.round(widthUnits * PX));
  canvas.height = Math.max(8, Math.round(heightUnits * PX));
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(widthUnits, heightUnits),
    new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }),
  );
  mesh.renderOrder = 12;
  return { canvas, ctx: canvas.getContext("2d"), texture, mesh };
}

function disposeObject(root) {
  root.traverse((o) => {
    o.geometry?.dispose();
    const m = o.material;
    if (m) (Array.isArray(m) ? m : [m]).forEach((mm) => {
      mm.map?.dispose();
      mm.dispose();
    });
  });
}

/* ============================================================================================ */

export class TokenTape {
  /**
   * @param {number} capacity numero massimo di caratteri (prompt + generati)
   * @param {{cols?: number, cell?: number}} [o] celle per riga e lato della cella
   */
  constructor(capacity, { cols = 16, cell = 0.62 } = {}) {
    this.capacity = capacity;
    this.cols = cols;
    this.cell = cell;
    this.group = new THREE.Group();
    const size = cell * 0.82;
    this.cubes = new THREE.InstancedMesh(new THREE.BoxGeometry(size, size, size * 0.6), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55 }), capacity);
    this.cubes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.cubes.count = 0;
    this.cubes.frustumCulled = false;
    for (let i = 0; i < capacity; i++) this.cubes.setColorAt(i, TOKEN_COLORS.idle);
    this.group.add(this.cubes);
    this.chars = [];
    this.ids = [];
    this.cells = [];
    this.states = [];
    this.rows = 1;
    this.labels = null;
    this.dummy = new THREE.Object3D();
    this.bracket = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x38bdf8 }));
    this.bracket.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(4 * 2 * 3 * 64), 3));
    this.bracket.geometry.setDrawRange(0, 0);
    this.bracket.frustumCulled = false;
    this.group.add(this.bracket);
  }

  /** Imposta tutto il testo (le righe della griglia si ricalcolano), senza mostrarlo ancora. */
  setText(chars, ids, maxRows) {
    this.chars = chars;
    this.ids = ids;
    this.cells = layoutText(chars, this.cols);
    this.rows = Math.max(maxRows ?? 1, (this.cells.at(-1)?.row ?? 0) + 1);
    this.states = chars.map(() => "idle");
    if (!this.labels || this.labels.rowsAllocated < this.rows) {
      if (this.labels) {
        this.group.remove(this.labels.mesh);
        disposeObject(this.labels.mesh);
      }
      this.labels = makeLabelPlane(this.cols * this.cell, this.rows * this.cell * 1.45);
      this.labels.rowsAllocated = this.rows;
      this.labels.mesh.position.set((this.cols * this.cell) / 2 - this.cell / 2, this.cell / 2 - (this.rows * this.cell * 1.45) / 2, this.cell * 0.32);
      this.group.add(this.labels.mesh);
    }
    this.visible = 0;
    this.idsShown = 0;
  }

  /** Centro (coordinate locali) della cella i. */
  cellPosition(i, target = new THREE.Vector3()) {
    const { col, row } = this.cells[i];
    return target.set(col * this.cell, -row * this.cell * 1.45, 0);
  }

  /** Posizione nel mondo della cella i (per i fili verso gli embedding). */
  worldPosition(i, target = new THREE.Vector3()) {
    this.cellPosition(i, target);
    this.group.updateWorldMatrix(true, false);
    return this.group.localToWorld(target);
  }

  /**
   * Mostra i primi `count` cubi; `appear` (0..1) fa "crescere" l'ultimo, per l'effetto di
   * scrittura. Nessuna allocazione: si riusa un solo Object3D di appoggio.
   */
  setVisible(count, appear = 1) {
    count = Math.min(count, this.chars.length);
    const changed = count !== this.visible;
    this.cubes.count = count;
    for (let i = 0; i < count; i++) {
      this.cellPosition(i, this.dummy.position);
      const s = i === count - 1 ? Math.max(0.01, easeInOut(appear)) : 1;
      this.dummy.scale.setScalar(s);
      this.dummy.updateMatrix();
      this.cubes.setMatrixAt(i, this.dummy.matrix);
    }
    this.cubes.instanceMatrix.needsUpdate = true;
    this.visible = count;
    if (changed) this.redraw();
  }

  /** Stato di ogni cubo ("idle" | "context" | "outside" | "generated" | "current"). */
  setStates(fn) {
    for (let i = 0; i < this.chars.length; i++) {
      this.states[i] = fn(i);
      this.cubes.setColorAt(i, TOKEN_COLORS[this.states[i]] ?? TOKEN_COLORS.idle);
    }
    if (this.cubes.instanceColor) this.cubes.instanceColor.needsUpdate = true;
  }

  /** Mostra gli id sotto i primi `count` caratteri (0 = nessuno). */
  setIds(count) {
    if (count === this.idsShown) return;
    this.idsShown = count;
    this.redraw();
  }

  /** Cornice attorno alle celle [from, to): la finestra di contesto. */
  setBracket(from, to) {
    const pos = this.bracket.geometry.attributes.position.array;
    let n = 0;
    const half = this.cell * 0.5;
    const p = new THREE.Vector3();
    // un rettangolo per ogni riga toccata dall'intervallo
    let i = from;
    while (i < to && n < 64) {
      const row = this.cells[i].row;
      let j = i;
      while (j + 1 < to && this.cells[j + 1].row === row) j++;
      this.cellPosition(i, p);
      const x0 = p.x - half;
      const y0 = p.y + half * 1.1;
      this.cellPosition(j, p);
      const x1 = p.x + half;
      const y1 = p.y - half * 2.1;
      const z = this.cell * 0.4;
      const segs = [x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0];
      for (let k = 0; k < 8; k++) {
        pos[n * 3] = segs[k * 2];
        pos[n * 3 + 1] = segs[k * 2 + 1];
        pos[n * 3 + 2] = z;
        n++;
      }
      i = j + 1;
    }
    this.bracket.geometry.setDrawRange(0, n);
    this.bracket.geometry.attributes.position.needsUpdate = true;
    this.bracket.geometry.computeBoundingSphere();
  }

  /** Ridisegna caratteri e id sul canvas (solo quando cambia il contenuto). */
  redraw() {
    const { ctx: g, canvas, texture } = this.labels;
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.textAlign = "center";
    g.textBaseline = "middle";
    const c = this.cell * PX;
    for (let i = 0; i < this.visible; i++) {
      const { col, row } = this.cells[i];
      const x = col * c + c / 2;
      const y = row * c * 1.45 + c / 2;
      const ch = this.chars[i];
      g.font = `600 ${c * 0.5}px ${MONO}`;
      g.fillStyle = ch === " " || ch === "\n" ? "#94a3b8" : "#f8fafc";
      g.fillText(visibleChar(ch), x, y);
      if (i < this.idsShown) {
        g.font = `600 ${c * 0.3}px ${MONO}`;
        g.fillStyle = this.states[i] === "outside" ? "#64748b" : "#fbbf24";
        g.fillText(String(this.ids[i]), x, y + c * 0.82);
      }
    }
    texture.needsUpdate = true;
  }

  /** Box3 nel mondo delle prime `count` celle (le InstancedMesh non hanno un bounding box utile). */
  bounds(count = this.chars.length) {
    const box = new THREE.Box3();
    const p = new THREE.Vector3();
    const half = this.cell * 0.55;
    const last = Math.max(1, Math.min(count, this.chars.length));
    for (let i = 0; i < last; i++) {
      this.worldPosition(i, p);
      box.expandByPoint(p.set(p.x - half, p.y + half, p.z));
      box.expandByPoint(p.set(p.x + 2 * half, p.y - 3.4 * half, p.z));
    }
    return box;
  }

  /** Larghezza e altezza della griglia in unità di scena. */
  size() {
    return { width: this.cols * this.cell, height: this.rows * this.cell * 1.45 };
  }

  dispose() {
    disposeObject(this.group);
  }
}

/* ============================================================================================ */

export class StoiTable {
  /** @param {string[]} vocab i 65 caratteri, indice = id */
  constructor(vocab, { cols = 13, cell = 0.62 } = {}) {
    this.vocab = vocab;
    this.cols = cols;
    this.cell = cell;
    this.rows = Math.ceil(vocab.length / cols);
    this.w = cols * cell * 1.25;
    this.h = this.rows * cell * 1.1 + 0.9;
    const plane = makeLabelPlane(this.w, this.h);
    this.plane = plane;
    this.group = new THREE.Group();
    this.group.add(plane.mesh);
    this.used = new Set();
    this.current = -1;
  }

  /** used: id presenti nel prompt; current: id appena tokenizzato (evidenziato). */
  draw(used, current = -1) {
    this.used = used;
    this.current = current;
    const { ctx: g, canvas, texture } = this.plane;
    const sx = canvas.width / this.w;
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.fillStyle = "rgba(8, 13, 24, 0.92)";
    g.fillRect(0, 0, canvas.width, canvas.height);
    g.textBaseline = "middle";
    g.fillStyle = "#38bdf8";
    g.font = `700 ${0.36 * sx}px Inter, system-ui, sans-serif`;
    g.fillText("stoi: carattere → id (65 caratteri)", 0.2 * sx, 0.42 * sx);
    const cw = this.cell * 1.25 * sx;
    const ch = this.cell * 1.1 * sx;
    this.vocab.forEach((c, id) => {
      const x = (id % this.cols) * cw;
      const y = 0.9 * sx + Math.floor(id / this.cols) * ch;
      const isCurrent = id === current;
      g.fillStyle = isCurrent ? "#0ea5e9" : used.has(id) ? "#0e4a5c" : "#1e293b";
      g.fillRect(x + 3, y + 3, cw - 6, ch - 6);
      g.textAlign = "center";
      g.font = `600 ${ch * 0.42}px ${MONO}`;
      g.fillStyle = isCurrent || used.has(id) ? "#f8fafc" : "#64748b";
      g.fillText(visibleChar(c), x + cw * 0.3, y + ch / 2);
      g.font = `500 ${ch * 0.32}px ${MONO}`;
      g.fillStyle = isCurrent ? "#0f172a" : "#fbbf24";
      g.fillText(String(id), x + cw * 0.72, y + ch / 2);
    });
    g.textAlign = "left";
    texture.needsUpdate = true;
  }

  dispose() {
    disposeObject(this.group);
  }
}

/* ============================================================================================ */

export class ContextColumn {
  /**
   * Colonna verticale di block_size cubi, con lo stesso passo dei token di architecture.js
   * (0.27): prende il posto della colonna d'esempio mentre l'azione è in corso.
   */
  constructor(blockSize, { size = 0.22, step = 0.27 } = {}) {
    this.blockSize = blockSize;
    this.size = size;
    this.step = step;
    this.group = new THREE.Group();
    this.cubes = new THREE.InstancedMesh(new THREE.BoxGeometry(size, size, size), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 }), blockSize);
    this.cubes.frustumCulled = false;
    const dummy = new THREE.Object3D();
    for (let t = 0; t < blockSize; t++) {
      dummy.position.set(0, this.y(t, blockSize), 0);
      dummy.updateMatrix();
      this.cubes.setMatrixAt(t, dummy.matrix);
      this.cubes.setColorAt(t, TOKEN_COLORS.context);
    }
    this.group.add(this.cubes);
    // canvas più largo della colonna: a sinistra gli id, sopra i cubi i caratteri
    this.labelW = 1.6;
    this.plane = makeLabelPlane(this.labelW, blockSize * step + 0.6);
    this.plane.mesh.position.set(-this.labelW / 2 + size / 2 + 0.06, 0, size / 2 + 0.02);
    this.group.add(this.plane.mesh);
    this.T = 0;
  }

  /** y della posizione t (la prima in alto), centrata come buildTokens di architecture.js. */
  y(t, T = this.blockSize) {
    return ((T - 1) / 2 - t) * this.step;
  }

  /** Mostra il contesto (lista di caratteri e id); `highlight` = posizione evidenziata. */
  set(chars, ids, highlight = -1) {
    this.T = chars.length;
    this.cubes.count = this.T;
    for (let t = 0; t < this.T; t++) this.cubes.setColorAt(t, t === highlight ? TOKEN_COLORS.current : t >= this.T - 1 && highlight === -2 ? TOKEN_COLORS.generated : TOKEN_COLORS.context);
    this.cubes.instanceColor.needsUpdate = true;
    const { ctx: g, canvas, texture } = this.plane;
    const sx = canvas.width / this.labelW;
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.textBaseline = "middle";
    const top = 0.3 * sx;
    for (let t = 0; t < this.T; t++) {
      const cy = top + t * this.step * sx + (this.step * sx) / 2;
      g.textAlign = "center";
      g.font = `600 ${this.size * 0.7 * sx}px ${MONO}`;
      g.fillStyle = "#f8fafc";
      g.fillText(visibleChar(chars[t]), canvas.width - (this.size / 2 + 0.06) * sx, cy);
      g.textAlign = "right";
      g.font = `500 ${this.size * 0.55 * sx}px ${MONO}`;
      g.fillStyle = "#fbbf24";
      g.fillText(`${t}: ${ids[t]}`, canvas.width - (this.size + 0.18) * sx, cy);
    }
    texture.needsUpdate = true;
  }

  worldPosition(t, target = new THREE.Vector3()) {
    target.set(this.size / 2, this.y(t), 0);
    this.group.updateWorldMatrix(true, false);
    return this.group.localToWorld(target);
  }

  dispose() {
    disposeObject(this.group);
  }
}

/* ============================================================================================ */

export class LinkLines {
  constructor(capacity = 96, color = 0xfbbf24) {
    this.capacity = capacity;
    this.positions = new Float32Array(capacity * 2 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geometry.setDrawRange(0, 0);
    this.lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 }));
    this.lines.frustumCulled = false;
    this.lines.userData.noBounds = true;
    this.count = 0;
  }

  clear() {
    this.count = 0;
    this.lines.geometry.setDrawRange(0, 0);
  }

  add(a, b) {
    if (this.count >= this.capacity) return;
    this.positions.set([a.x, a.y, a.z, b.x, b.y, b.z], this.count * 6);
    this.count++;
    this.lines.geometry.attributes.position.needsUpdate = true;
  }

  /** Mostra solo i primi `n` fili (rivelazione progressiva, senza toccare i buffer). */
  reveal(n) {
    this.lines.geometry.setDrawRange(0, Math.min(this.count, Math.max(0, n)) * 2);
  }

  dispose() {
    disposeObject(this.lines);
  }
}

/* ============================================================================================ */

export class RowMarks {
  /** Barre luminose sopra le righe di una matrice (una istanza per riga letta). */
  constructor(capacity = 64, color = 0xfbbf24) {
    this.mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide }), capacity);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 15;
    this.mesh.userData.noBounds = true;
    this.capacity = capacity;
    this.dummy = new THREE.Object3D();
  }

  /**
   * Evidenzia le righe `rows` della matrice `mesh` (piano di matrixPlane con `nRows` righe).
   * Restituisce la posizione nel mondo del bordo sinistro di ogni riga (per i fili).
   */
  mark(matrixMesh, nRows, rows, cell) {
    matrixMesh.updateWorldMatrix(true, false);
    const width = matrixMesh.geometry.parameters.width;
    const height = matrixMesh.geometry.parameters.height;
    const ends = [];
    this.mesh.count = 0;
    rows.forEach((r, i) => {
      if (i >= this.capacity) return;
      const y = height / 2 - (r + 0.5) * cell;
      this.dummy.position.set(0, y, 0.02);
      this.dummy.scale.set(width + 0.06, cell * 1.1, 1);
      this.dummy.updateMatrix();
      this.dummy.matrix.premultiply(matrixMesh.matrixWorld);
      this.mesh.setMatrixAt(i, this.dummy.matrix);
      this.mesh.count = i + 1;
      ends.push(new THREE.Vector3(-width / 2, y, 0.02).applyMatrix4(matrixMesh.matrixWorld));
    });
    this.mesh.instanceMatrix.needsUpdate = true;
    return ends;
  }

  /** Mostra solo le prime n righe evidenziate. */
  reveal(n, total) {
    this.mesh.count = Math.min(total, Math.max(0, n));
  }

  dispose() {
    disposeObject(this.mesh);
  }
}

/** Cubo singolo con il suo carattere (il token estratto che vola verso il contesto). */
export class FlyingToken {
  constructor(size = 0.6) {
    this.size = size;
    this.group = new THREE.Group();
    this.cube = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), new THREE.MeshStandardMaterial({ color: 0xb45309, emissive: 0x7c2d12, roughness: 0.4 }));
    this.plane = makeLabelPlane(size, size);
    this.plane.mesh.position.z = size / 2 + 0.01;
    this.group.add(this.cube, this.plane.mesh);
    this.group.visible = false;
    this.from = new THREE.Vector3();
    this.to = new THREE.Vector3();
  }

  set(char) {
    const { ctx: g, canvas, texture } = this.plane;
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.fillStyle = "#f8fafc";
    g.font = `700 ${canvas.height * 0.62}px ${MONO}`;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(visibleChar(char), canvas.width / 2, canvas.height / 2);
    texture.needsUpdate = true;
  }

  /** Volo ad arco da `from` a `to` (p in 0..1), senza allocazioni. */
  fly(p) {
    const k = easeInOut(p);
    this.group.position.lerpVectors(this.from, this.to, k);
    this.group.position.z += Math.sin(Math.PI * k) * 3;
    this.group.scale.setScalar(1 - 0.45 * k);
  }

  dispose() {
    disposeObject(this.group);
  }
}

export { disposeObject };
