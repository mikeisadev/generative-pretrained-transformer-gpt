/**
 * La scena 3D del Laboratorio pesi (three.js): l'albero dei moduli come il codice Python, con ogni
 * parametro come NumericMatrix (celle colorate e, da vicino, i numeri veri).
 *
 * Tre livelli di dettaglio (LOD, come negli altri tab):
 *   detail  "Dettaglio"   contenitori annidati con la riga Python che li crea, una matrice per parametro;
 *                         i numeri compaiono da soli quando le celle sono abbastanza grandi (NumbersGroup)
 *   blocks  "Moduli"      un riquadro per modulo (ln1, sa, ln2, ffwd di ogni blocco…) con i parametri
 *   macro   "Panoramica"  le isole del modello (embedding, Block ×N, ln_f, lm_head) con le quote e il flusso
 * Le matrici stanno in un gruppo a parte (`cells`), visibile solo nel Dettaglio: da lontano una cella è
 * meno di un pixel e conterebbe solo come costo.
 *
 * API per gli altri moduli (fase 2: animazioni, confronto) — vedi lab.md:
 *   scene.showValues(map | null, { colormap: "divergente" | "pesi", label })   valori qualsiasi sulle stesse matrici
 *   scene.setWeights(map)            i pesi di un altro step (name → Float32Array | {data})
 *   scene.focusTensor(name)          scene.focusModule(path)        scene.focusCell(name, i, j)
 *   scene.matrixOf(name) → NumericMatrix           scene.highlightCells(name, celle | null)
 *   scene.layoutOf(name) → { rect, center, cellW, cellH, rows, cols, cellCenter(i, j) }   scene.boxOf(path) → Box3
 */
import * as THREE from "three";
import { NumericMatrix, NumbersGroup } from "../../../core/matrixNumbers.js";
import { tensorMeaning } from "../inspect/meaning.js";
import { buildArrows, buildFrames, disposeFrames } from "./frames.js";
import { LabelAtlas, disposeLabelMesh, labelMesh } from "./labels.js";
import { CELL, Z_STEP, coveredBy, fmtInt, fmtPct, hitTest, macroItems, moduleItems, subtractRects } from "./layout.js";
import { LAB_ROLES } from "./roles.js";

/** Budget della scena: oltre, le matrici più grandi diventano riquadri (i numeri restano nel pannello Tensore). */
export const SCENE_LIMITS = Object.freeze({ maxCells: 8_000_000, maxCellsPerMatrix: 4_000_000, maxVeilCells: 1_048_576 });

const ACCENT = "#38bdf8";
const BACKGROUND = "#0d1117"; // lo sfondo della scena (SceneHost)
const PX = { title: 30, code: 22, sub: 19, param: 20, big: 64, bigSub: 40 };
const colorOf = (role) => (LAB_ROLES[role] ?? LAB_ROLES.other).color;
const FRONT = new THREE.Vector3(0, 0.02, 1).normalize();

export class LabScene {
  /**
   * @param {{numbers?: NumbersGroup, createCanvas?: () => HTMLCanvasElement}} [o]
   */
  constructor({ numbers = null, createCanvas } = {}) {
    this.root = new THREE.Group();
    this.root.name = "lab-scene";
    this.levels = { detail: new THREE.Group(), blocks: new THREE.Group(), macro: new THREE.Group() };
    this.cells = new THREE.Group();
    this.root.add(this.levels.detail, this.levels.blocks, this.levels.macro, this.cells);
    this.numbers = numbers ?? new NumbersGroup({ maxOverlays: 4, maxCellsPerOverlay: 2500 });
    this.createCanvas = createCanvas;
    /** nome → { m: NumericMatrix, node, weights, range, custom, veil } */
    this.matrices = new Map();
    this.placeholders = [];
    this.layout = null;
    this.run = null;
    this.level = "detail";
    this.valuesLabel = "";
    this.customValues = new Set();
    this.limited = [];
    this._built = [];
    this._atlases = [];
    this._outline = null;
    this._hover = null;
    this._ray = new THREE.Ray();
    this._plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    this._point = new THREE.Vector3();
    this._byDepth = [];
  }

  // ---------------------------------------------------------------- costruzione
  /**
   * Costruisce (o ricostruisce) la scena per un run.
   * @param {{run: object, layout: object, view: {tensors: Map}, config?: object, vocab?: string[]}} o
   */
  build({ run, layout, view, config = run?.config ?? null, vocab = run?.vocab ?? null }) {
    this.clear();
    this.run = run;
    this.layout = layout;
    this.config = config;
    this.vocab = vocab;
    this._buildDetail(layout);
    this._buildMatrices(layout, view?.tensors ?? run.tensors);
    this._buildModules(layout);
    this._buildMacro(layout);
    this._outline = outlineLoop();
    this.root.add(this._outline);
    // per la scelta col mouse: i riquadri per profondità (dal più profondo), ognuno alla sua z
    const depths = new Map();
    for (const b of layout.boxes) {
      if (b.kind === "group" || b.kind === "param") continue;
      if (!depths.has(b.depth)) depths.set(b.depth, []);
      depths.get(b.depth).push(b);
    }
    this._byDepth = [...depths.entries()].sort((a, b) => b[0] - a[0]);
    this.setLevel(this.level);
    this.numbers.invalidate();
  }

  _buildDetail(layout) {
    // velature piene: il colore di ogni livello è già quello che darebbe la sovrapposizione delle
    // velature trasparenti (sfondo + un poco del colore del ruolo, di più a ogni livello di annidamento)
    // velature piene, ognuna solo dove i figli non la coprono (subtractRects): ogni pixel si colora una
    // volta. Il colore di ogni livello è quello che darebbero velature trasparenti sovrapposte.
    const fills = [];
    const borders = [];
    for (const b of layout.boxes) {
      if (b.kind !== "module" && b.kind !== "virtual") continue;
      const tint = b.kind === "virtual" ? 0.05 : Math.min(0.2, 0.035 + 0.026 * b.depth);
      const color = mix(BACKGROUND, colorOf(b.role), tint);
      for (const r of subtractRects(b, coveredBy(b))) fills.push({ ...r, z: b.z, color, alpha: 1, edge: 0 });
      borders.push({ x0: b.x0, x1: b.x1, y0: b.y0, y1: b.y1, z: b.z, color: colorOf(b.role), alpha: 0, edge: b.kind === "virtual" ? 0.55 : b.depth === 0 ? 0.6 : 0.85 });
    }
    const fillMesh = buildFrames(fills, { opaque: true });
    fillMesh.userData.lines.visible = false; // i bordi sono per contenitore, non per striscia
    const frames = buildFrames(borders);
    this.levels.detail.add(fillMesh, frames);
    this._built.push(() => disposeFrames(fillMesh), () => disposeFrames(frames));

    const atlas = new LabelAtlas({ createCanvas: this.createCanvas });
    const items = [];
    for (const b of layout.boxes) {
      if ((b.kind === "module" || b.kind === "virtual") && b.header) {
        const h = b.header;
        const color = lighten(colorOf(b.role));
        const tagSeg = h.tag ? { text: h.tag, font: "code", px: Math.round(PX.title * 0.68), color: "#94a3b8" } : null;
        const lines = b.kind === "virtual"
          ? [[{ text: h.title, font: "title", px: PX.title, color }], [{ text: h.code, font: "code", px: PX.code, color: "#cbd5e1", gap: PX.code }, tagSeg], [{ text: h.sub, font: "sub", px: PX.sub, color: "#94a3b8" }]]
          : [[{ text: h.title, font: "title", px: PX.title, color, gap: PX.title * 0.6 }, tagSeg], h.code ? [{ text: h.code, font: "code", px: PX.code, color: "#e2e8f0" }] : [], [{ text: h.sub, font: "sub", px: PX.sub, color: "#94a3b8" }]];
        const key = atlas.add({ key: `h|${lines.flat().filter(Boolean).map((s) => s.text).join("|")}`, lines: lines.map((l) => l.filter(Boolean)).filter((l) => l.length) });
        // scala: il titolo alto h.titleH nel mondo; se il testo vero è più largo del previsto si stringe
        items.push({ key, x: b.headerBox.x0, y: b.headerBox.y1, scale: h.titleH / (PX.title * 1.28), maxW: b.headerBox.w, z: b.z + 0.004 });
      } else if ((b.kind === "param" || b.kind === "buffer") && b.label) {
        const color = b.kind === "buffer" ? "#94a3b8" : "#cbd5e1";
        const key = atlas.add({ key: `p|${b.label}`, lines: [[{ text: b.label, font: "code", px: PX.param, color }]] });
        items.push({ key, x: b.labelBox.x0, y: b.labelBox.y1, scale: b.labelBox.h / (PX.param * 1.28), maxW: Math.max(b.labelBox.w, b.matrix.w) * 1.4, z: b.z + 0.004 });
      }
    }
    const built = atlas.build();
    const labels = labelMesh(items, built);
    this.levels.detail.add(labels);
    this._atlases.push(atlas);
    this._built.push(() => disposeLabelMesh(labels));
  }

  _buildMatrices(layout, tensors) {
    this.limited = [];
    let budget = SCENE_LIMITS.maxCells;
    // le matrici più piccole prima: con un budget finito restano fuori solo le enormi
    const order = [...layout.boxes.filter((b) => b.kind === "param" || b.kind === "buffer")].sort((a, b) => a.matrix.rows * a.matrix.cols - b.matrix.rows * b.matrix.cols);
    for (const b of order) {
      const { rows, cols } = b.matrix;
      const cells = rows * cols;
      if (cells > SCENE_LIMITS.maxCellsPerMatrix || cells > budget) {
        this.limited.push(b.path);
        this._placeholder(b);
        continue;
      }
      budget -= cells;
      let data;
      let masked = null;
      let colormap = "diverging";
      let range = null;
      if (b.kind === "buffer") {
        // il buffer tril: 1 sotto la diagonale (e sulla diagonale), 0 sopra = il futuro, tratteggiato come nel Transformer
        data = new Float32Array(cells);
        for (let r = 0; r < rows; r++) for (let c = 0; c <= r && c < cols; c++) data[r * cols + c] = 1;
        masked = (r, c) => c > r;
        colormap = "attention";
        range = 1;
      } else {
        data = tensors.get(b.path)?.data;
        if (!data || data.length < cells) {
          this.limited.push(b.path);
          this._placeholder(b);
          continue;
        }
      }
      const meaning = tensorMeaning(b.path, b.shape, { vocab: this.vocab, config: this.config });
      // etichette di righe e colonne con il loro significato (riga 43 = «e»…): per un vettore steso in
      // riga gli elementi sono le colonne, per un vettore in colonna (bias accanto a W) sono le righe
      const element = (k) => meaning.rows.label(k);
      const vector = b.matrix.vector;
      const m = new NumericMatrix({
        rows,
        cols,
        data,
        cell: b.matrix.cell,
        colormap,
        range,
        group: this.numbers,
        masked,
        maskedText: "0",
        rowLabel: vector === "row" ? null : element,
        colLabel: vector === "row" ? element : vector === "column" ? null : (c) => meaning.cols.label(c),
      });
      // i vettori sono stirati sul lato sottile (layout.VECTOR_STRETCH): una cella resta un peso
      m.mesh.scale.set(b.matrix.cellW / CELL, b.matrix.cellH / CELL, 1);
      m.object3d.position.set(b.matrix.cx, b.matrix.cy, b.z);
      m.mesh.userData.labPath = b.path;
      this.cells.add(m.object3d);
      this.matrices.set(b.path, { m, node: b, weights: data, range: m.range, custom: false, veil: null, buffer: b.kind === "buffer" });
    }
  }

  _placeholder(b) {
    const geometry = new THREE.PlaneGeometry(b.matrix.w, b.matrix.h);
    const material = new THREE.MeshBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0.9 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(b.matrix.cx, b.matrix.cy, b.z);
    mesh.userData.labPath = b.path;
    mesh.userData.placeholder = true;
    this.cells.add(mesh);
    this.placeholders.push(mesh);
  }

  _buildModules(layout) {
    const mods = moduleItems(layout);
    const rects = mods.map((n) => ({ x0: n.x0, x1: n.x1, y0: n.y0, y1: n.y1, z: n.z + 0.01, color: mix("#0b1220", colorOf(n.role), 0.18), alpha: 0.96, edge: 0 }));
    const borders = mods.map((n) => ({ x0: n.x0, x1: n.x1, y0: n.y0, y1: n.y1, z: n.z + 0.012, color: colorOf(n.role), alpha: 0, edge: 0.95 }));
    const fills = buildFrames(rects, { opaque: true });
    fills.userData.lines.visible = false;
    const lines = buildFrames(borders);
    this.levels.blocks.add(fills, lines);
    this._built.push(() => disposeFrames(fills), () => disposeFrames(lines));

    const atlas = new LabelAtlas({ createCanvas: this.createCanvas });
    // titolo breve (il nome nel codice), poi il tipo e i parametri: da lontano si legge almeno il nome
    const items = mods.map((n) => {
      const type = n.type ?? "modulo";
      const sub = n.kind === "virtual" ? "nessun peso" : `${fmtInt(n.params)} parametri · ${fmtPct(n.share)}`;
      const key = atlas.add({
        key: `m|${n.title}|${type}|${sub}`,
        align: "center",
        lines: [[{ text: n.title, font: "big", px: PX.big, color: lighten(colorOf(n.role)) }], [{ text: type, font: "sub", px: PX.bigSub, color: "#e2e8f0" }], [{ text: sub, font: "sub", px: PX.bigSub, color: "#94a3b8" }]],
      });
      const titleH = Math.min(n.h * 0.22, 2.6);
      return { key, x: n.cx, y: n.cy, anchor: "center", scale: titleH / (PX.big * 1.28), maxW: n.w * 0.92, z: n.z + 0.03 };
    });
    const labels = labelMesh(items, atlas.build());
    this.levels.blocks.add(labels);
    this._atlases.push(atlas);
    this._built.push(() => disposeLabelMesh(labels));
  }

  _buildMacro(layout) {
    const items = macroItems(layout);
    this.macro = items;
    const rects = items.map((it) => ({ x0: it.x0, x1: it.x1, y0: it.y0, y1: it.y1, z: it.z + 0.02, color: mix("#0b1220", colorOf(it.role), it.role === "block" ? 0.05 : 0.22), alpha: 0.97, edge: 0 }));
    const borders = items.map((it) => ({ x0: it.x0, x1: it.x1, y0: it.y0, y1: it.y1, z: it.z + 0.022, color: colorOf(it.role), alpha: 0, edge: 1 }));
    const fills = buildFrames(rects, { opaque: true });
    fills.userData.lines.visible = false;
    const lines = buildFrames(borders);
    // le frecce del flusso: dal bordo destro di un'isola al sinistro della successiva, a metà altezza del modello
    const r = layout.root;
    const midY = (layout.nodes.get("blocks")?.cy ?? r.cy);
    const arrows = buildArrows(
      items.slice(0, -1).map((it, i) => ({ x0: it.x1 + 0.1, x1: items[i + 1].x0 - 0.1, y: midY, z: it.z + 0.03 })),
      { width: 0.45, head: 0.9 },
    );
    this.levels.macro.add(fills, lines, arrows);
    this._built.push(() => disposeFrames(fills), () => disposeFrames(lines), () => {
      arrows.geometry.dispose();
      arrows.material.dispose();
      arrows.removeFromParent();
    });
    const atlas = new LabelAtlas({ createCanvas: this.createCanvas });
    // le isole grandi (i blocchi) hanno l'etichetta dentro; quelle strette (embedding, ln_f, lm_head)
    // sopra di sé, con il solo nome: dentro un riquadro largo 7 unità il testo sarebbe illeggibile
    const quads = items.map((it) => {
      const w = it.x1 - it.x0;
      const small = w < 14;
      const title = small ? it.title.split(" · ")[0] : it.title;
      const sub = small ? fmtInt(it.params) : `${fmtInt(it.params)} parametri · ${fmtPct(it.share)}`;
      const key = atlas.add({ key: `M|${title}|${sub}`, align: "center", lines: [[{ text: title, font: "big", px: PX.big, color: lighten(colorOf(it.role)) }], [{ text: sub, font: "sub", px: PX.bigSub, color: "#e2e8f0" }]] });
      if (small) return { key, x: (it.x0 + it.x1) / 2, y: it.y1 + 0.5, anchor: "bottom-center", scale: 1.7 / (PX.big * 1.28), maxW: w + 1.2, z: it.z + 0.05 };
      const titleH = Math.min((it.y1 - it.y0) * 0.16, 3.4);
      return { key, x: (it.x0 + it.x1) / 2, y: (it.y0 + it.y1) / 2, anchor: "center", scale: titleH / (PX.big * 1.28), maxW: w * 0.9, z: it.z + 0.05 };
    });
    const title = this.run?.name ? `${layout.root.title} · ${this.run.name}` : layout.root.title;
    const head = `${fmtInt(layout.total)} parametri in ${fmtInt(layout.params.length)} tensori`;
    const titleKey = atlas.add({ key: `T|${title}|${head}`, align: "center", lines: [[{ text: title, font: "big", px: PX.big, color: "#e0f2fe" }], [{ text: head, font: "sub", px: PX.bigSub, color: "#94a3b8" }]] });
    quads.push({ key: titleKey, x: r.cx, y: r.y1 + 0.6, anchor: "top-center", scale: Math.min(3.2, (r.x1 - r.x0) * 0.02) / (PX.big * 1.28), maxW: r.x1 - r.x0, z: 0.05 });
    // il riquadro del titolo sta sopra il modello: anche lui conta per l'inquadratura della Panoramica
    const labels = labelMesh(quads, atlas.build());
    const titleSize = quads.at(-1).size;
    this.macroTitleBox = titleSize ? { x0: titleSize.x0, x1: titleSize.x0 + titleSize.w, y0: titleSize.y1 - titleSize.h, y1: titleSize.y1 } : null;
    this.levels.macro.add(labels);
    this._atlases.push(atlas);
    this._built.push(() => disposeLabelMesh(labels));
  }

  /** Livello del LOD attivo: le matrici si vedono (e costano) solo nel Dettaglio. */
  setLevel(level) {
    this.level = level;
    this.cells.visible = level === "detail";
    if (this._outline) this._outline.visible = this._outline.userData.active && level === "detail";
    this.numbers.invalidate();
  }

  // ---------------------------------------------------------------- dati
  /** I pesi di un altro step (o di un altro run con gli stessi nomi e forme): name → Float32Array | {data}. */
  setWeights(tensors) {
    for (const [name, value] of entriesOf(tensors)) {
      const e = this.matrices.get(name);
      const data = value?.data ?? value;
      if (!e || e.buffer || !data || data.length < e.m.rows * e.m.cols) continue;
      e.weights = data;
      if (!e.custom) e.m.setData(data); // la scala resta quella dei pesi finali: si vede quanto crescono
    }
  }

  /**
   * Valori qualsiasi sulle stesse matrici (es. la differenza A − B). `map`: name → Float32Array (stessa
   * forma) o null (torna ai pesi); `map` null = tutte tornano ai pesi.
   * colormap "pesi": la scala dei pesi (confrontabile a occhio); "divergente": una scala sua, simmetrica,
   * calcolata sui valori (comune a tutti i tensori passati, così due matrici si confrontano).
   * @returns {{shown: string[], missing: string[], label: string}}
   */
  showValues(map, { colormap = "divergente", label = "" } = {}) {
    if (map === null || map === undefined) {
      for (const [, e] of this.matrices) if (e.custom) this._restore(e);
      this.customValues.clear();
      this.valuesLabel = "";
      return { shown: [], missing: [], label: "" };
    }
    const shown = [];
    const missing = [];
    const entries = entriesOf(map);
    let common = 0;
    if (colormap !== "pesi") {
      for (const [, v] of entries) if (v) for (let i = 0; i < v.length; i++) common = Math.max(common, Math.abs(v[i]) || 0);
    }
    for (const [name, values] of entries) {
      const e = this.matrices.get(name);
      if (!e || e.buffer) {
        missing.push(name);
        continue;
      }
      if (values === null) {
        this._restore(e);
        this.customValues.delete(name);
        continue;
      }
      if (values.length < e.m.rows * e.m.cols) {
        missing.push(name);
        continue;
      }
      e.custom = true;
      e.m.setData(values, { range: colormap === "pesi" ? e.range : common > 0 ? common : 1 });
      this.customValues.add(name);
      shown.push(name);
    }
    this.valuesLabel = this.customValues.size ? label : "";
    return { shown, missing, label: this.valuesLabel };
  }

  _restore(e) {
    e.custom = false;
    e.m.setData(e.weights, { range: e.range });
  }

  /** La NumericMatrix di un tensore (o del buffer tril), null se non c'è. */
  matrixOf(name) {
    return this.matrices.get(name)?.m ?? null;
  }

  /** Il nodo della disposizione (contenitore, parametro, buffer, nodo senza pesi) di un percorso. */
  nodeOf(path) {
    return this.layout?.nodes.get(path) ?? null;
  }

  /**
   * Dove sta un tensore, per le animazioni: rettangolo della matrice (mondo), centro, lato delle celle,
   * centro di una cella. null se non c'è.
   */
  layoutOf(name) {
    const b = this.layout?.nodes.get(name);
    if (!b?.matrix) return null;
    const m = b.matrix;
    return {
      name,
      rows: m.rows,
      cols: m.cols,
      cellW: m.cellW,
      cellH: m.cellH,
      rect: { x0: m.x0, x1: m.x1, y0: m.y0, y1: m.y1, z: b.z },
      center: new THREE.Vector3(m.cx, m.cy, b.z),
      cellCenter: (i, j, target = new THREE.Vector3()) => target.set(m.x0 + (j + 0.5) * m.cellW, m.y1 - (i + 0.5) * m.cellH, b.z + 0.01),
    };
  }

  /** Box3 di un modulo o di un tensore (per inquadrarlo): in z fino al contenuto più avanti. */
  boxOf(path) {
    const b = this.layout?.nodes.get(path) ?? (path === "" ? this.layout?.root : null);
    if (!b) return null;
    const r = b.matrix ?? b;
    const front = (n) => Math.max(n.z, ...(n.children ?? []).map(front));
    return new THREE.Box3(new THREE.Vector3(r.x0, r.y0, b.z - 0.05), new THREE.Vector3(r.x1, r.y1, front(b) + 0.05));
  }

  /** Box3 di tutto il modello (con il titolo della Panoramica). */
  bounds() {
    const L = this.layout;
    if (!L) return new THREE.Box3();
    const box = new THREE.Box3(new THREE.Vector3(L.bounds.x0, L.bounds.y0, -0.05), new THREE.Vector3(L.bounds.x1, L.bounds.y1, L.bounds.z1 + 0.05));
    if (this.macroTitleBox) box.expandByPoint(new THREE.Vector3(this.macroTitleBox.x0, this.macroTitleBox.y1, 0)).expandByPoint(new THREE.Vector3(this.macroTitleBox.x1, this.macroTitleBox.y1, 0));
    return box;
  }

  /**
   * Evidenzia alcune celle di una matrice e vela le altre (per le animazioni: una riga di W, una colonna,
   * celle singole). `cells`: [{i, j}] | {rows: [..]} | {cols: [..]} | {r0, r1, c0, c1} (esclusi r1, c1) | null.
   * @returns {boolean} false se la matrice non c'è o è troppo grande per il velo
   */
  highlightCells(name, cells, { dim = 0.62 } = {}) {
    const e = this.matrices.get(name);
    if (!e) return false;
    if (!cells) {
      if (e.veil) e.veil.mesh.visible = false;
      return true;
    }
    const { rows, cols } = e.m;
    if (rows * cols > SCENE_LIMITS.maxVeilCells) return false;
    if (!e.veil) e.veil = makeVeil(e.m);
    const alpha = e.veil.texture.image.data;
    const base = Math.round(255 * dim);
    for (let k = 3; k < alpha.length; k += 4) alpha[k] = base;
    const open = (r, c) => {
      if (r < 0 || c < 0 || r >= rows || c >= cols) return;
      alpha[((rows - 1 - r) * cols + c) * 4 + 3] = 0; // la texture parte dal basso (riga 0 in alto)
    };
    if (Array.isArray(cells)) for (const c of cells) open(c.i ?? c.row, c.j ?? c.col);
    else if (cells.rows) for (const r of cells.rows) for (let c = 0; c < cols; c++) open(r, c);
    else if (cells.cols) for (const c of cells.cols) for (let r = 0; r < rows; r++) open(r, c);
    else if (Number.isInteger(cells.r0)) for (let r = cells.r0; r < Math.min(rows, cells.r1); r++) for (let c = cells.c0; c < Math.min(cols, cells.c1); c++) open(r, c);
    e.veil.texture.needsUpdate = true;
    e.veil.mesh.visible = true;
    return true;
  }

  /** Il riquadro di selezione attorno a un tensore o a un modulo (null lo toglie). */
  setSelected(path) {
    const o = this._outline;
    if (!o) return;
    const b = path ? this.layout?.nodes.get(path) : null;
    o.userData.active = Boolean(b);
    o.visible = Boolean(b) && (this.level === "detail" || !b.matrix);
    if (!b) return;
    const r = b.matrix ?? b;
    const pad = b.matrix ? Math.max(0.02, Math.min(r.x1 - r.x0, r.y1 - r.y0) * 0.04) : 0.05;
    o.position.set((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, b.z + 0.03);
    o.scale.set(r.x1 - r.x0 + 2 * pad, r.y1 - r.y0 + 2 * pad, 1);
  }

  /** La cella sotto il mouse: riquadro sottile (null lo toglie). */
  hoverCell(name, cell) {
    if (this._hover && (this._hover.name !== name || !cell)) {
      this.matrices.get(this._hover.name)?.m.highlight(null);
      this._hover = null;
    }
    if (!name || !cell) return;
    this.matrices.get(name)?.m.highlight(cell);
    this._hover = { name };
  }

  // ---------------------------------------------------------------- scelta col mouse
  /**
   * Che cosa c'è sotto il puntatore. `hits`: le intersezioni del raycaster con `pickables()` (già
   * calcolate con host.pick, che ha anche impostato `ray`). Nel Dettaglio vince la matrice colpita (con
   * la cella), poi il contenitore più profondo; negli altri livelli i riquadri del livello.
   * @returns {{path: string, kind: string, cell?: object, node: object}|null}
   */
  pickFrom(ray, hits) {
    if (!this.layout) return null;
    if (this.level === "detail") {
      for (const hit of hits) {
        const path = hit.object.userData.labPath;
        const e = this.matrices.get(path);
        if (e) return { path, kind: e.buffer ? "buffer" : "tensor", cell: e.m.cellAt(hit), node: e.node };
        if (hit.object.userData.placeholder) return { path, kind: "tensor", cell: null, node: this.layout.nodes.get(path) };
      }
      for (const [depth, boxes] of this._byDepth) {
        const p = this._intersect(ray, depth * Z_STEP);
        if (!p) continue;
        for (const b of boxes) if (p.x >= b.x0 && p.x <= b.x1 && p.y >= b.y0 && p.y <= b.y1) return { path: b.path, kind: b.kind === "virtual" ? "virtual" : b.kind === "buffer" ? "buffer" : "module", node: b };
      }
      return null;
    }
    const items = this.level === "macro" ? this.macro.map((it) => ({ ...it, path: it.members[0], z: it.z })) : moduleItems(this.layout);
    for (const it of items) {
      const p = this._intersect(ray, it.z);
      if (p && p.x >= it.x0 && p.x <= it.x1 && p.y >= it.y0 && p.y <= it.y1) {
        const node = this.layout.nodes.get(it.path);
        return { path: it.path, kind: node?.kind === "param" ? "tensor" : node?.kind ?? "module", node };
      }
    }
    return null;
  }

  /** Gli oggetti da passare al raycaster (solo le matrici visibili: i contenitori si risolvono con la disposizione). */
  pickables() {
    if (this.level !== "detail") return [];
    const list = [];
    for (const e of this.matrices.values()) list.push(e.m.mesh);
    return list.concat(this.placeholders);
  }

  _intersect(ray, z) {
    this._plane.constant = -z;
    return ray.intersectPlane(this._plane, this._point);
  }

  /** Il nodo più profondo in un punto del piano (test e console). */
  nodeAt(x, y) {
    return this.layout ? hitTest(this.layout, x, y) : null;
  }

  // ---------------------------------------------------------------- pulizia
  clear() {
    for (const e of this.matrices.values()) {
      if (e.veil) {
        e.veil.mesh.removeFromParent();
        e.veil.mesh.geometry.dispose();
        e.veil.mesh.material.dispose();
        e.veil.texture.dispose();
      }
      e.m.dispose();
    }
    this.matrices.clear();
    for (const p of this.placeholders) {
      p.removeFromParent();
      p.geometry.dispose();
      p.material.dispose();
    }
    this.placeholders = [];
    for (const undo of this._built.splice(0)) undo();
    for (const a of this._atlases.splice(0)) a.dispose();
    if (this._outline) {
      this._outline.removeFromParent();
      this._outline.geometry.dispose();
      this._outline.material.dispose();
      this._outline = null;
    }
    this._hover = null;
    this.customValues.clear();
    this.valuesLabel = "";
    this.layout = null;
    this.macro = [];
  }

  dispose() {
    this.clear();
    this.root.removeFromParent();
  }

  /** Statistiche per i test e la console. */
  get stats() {
    let cells = 0;
    for (const e of this.matrices.values()) cells += e.m.rows * e.m.cols;
    return { matrices: this.matrices.size, cells, placeholders: this.placeholders.length, limited: [...this.limited], level: this.level, atlases: this._atlases.length };
  }
}

// ---------------------------------------------------------------- aiuti
function entriesOf(map) {
  if (!map) return [];
  if (map instanceof Map) return [...map.entries()];
  return Object.entries(map);
}

/** Il velo per highlightCells: un piano sopra la matrice (e sopra i suoi numeri) con un alfa per cella. */
function makeVeil(m) {
  const { rows, cols } = m;
  const texture = new THREE.DataTexture(new Uint8Array(rows * cols * 4), cols, rows, THREE.RGBAFormat);
  const data = texture.image.data;
  for (let k = 0; k < data.length; k += 4) {
    data[k] = 8;
    data[k + 1] = 11;
    data[k + 2] = 18;
  }
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.flipY = false;
  texture.needsUpdate = true;
  m.mesh.geometry.computeBoundingBox();
  const bb = m.mesh.geometry.boundingBox;
  const geometry = new THREE.PlaneGeometry(bb.max.x - bb.min.x, bb.max.y - bb.min.y);
  const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -8, toneMapped: false });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.raycast = () => {};
  mesh.userData.noBounds = true;
  mesh.renderOrder = (m.mesh.renderOrder ?? 0) + 3; // sopra le celle e sopra i numeri (overlay: +1)
  mesh.position.z = 1e-4;
  m.mesh.add(mesh);
  return { mesh, texture };
}

function outlineLoop() {
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, -0.5, 0), new THREE.Vector3(0.5, 0.5, 0), new THREE.Vector3(-0.5, 0.5, 0)]);
  const loop = new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 1, depthWrite: false, toneMapped: false }));
  loop.raycast = () => {};
  loop.userData.noBounds = true;
  loop.renderOrder = 8;
  loop.visible = false;
  return loop;
}

const _a = new THREE.Color();
const _b = new THREE.Color();
/** Mescola due colori (t = quota del secondo), in esadecimale. */
function mix(a, b, t) {
  return `#${_a.set(a).lerp(_b.set(b), t).getHexString()}`;
}
/** Il colore del ruolo un poco più chiaro: titoli leggibili sul fondo scuro. */
function lighten(color) {
  return mix(color, "#ffffff", 0.25);
}

export { FRONT };
