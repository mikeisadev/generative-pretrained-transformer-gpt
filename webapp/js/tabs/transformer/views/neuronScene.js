/**
 * Scena 3D della vista "Rete neurale" (dati e calcoli in neuronModel.js).
 *
 * Prestazioni: tutti i neuroni sono UNA InstancedMesh (colore per istanza = attivazione);
 * tutti i 207.552 archi con peso sono UNA LineSegments, scritta nell'ordine "dal più forte"
 * (neuronModel.buildEdges → order): la soglia "mostra solo i pesi più forti" è soltanto
 * geometry.setDrawRange, senza ricostruire niente. Il livello "Blocchi" riusa la stessa
 * geometria con un materiale proprio (regola del LOD). Nessuna allocazione per frame.
 *
 * Livelli (registrati in tab.lod):
 *   detail  neuroni, archi, attention tratteggiata, residui, etichette delle colonne
 *   blocks  un rettangolo per colonna (strato) con titolo e descrizione, archi in trasparenza
 *   macro   un rettangolo per riga (ingresso, blocchi, uscita) e le frecce del flusso
 */
import * as THREE from "three";
import { makeArrow, makeFlow } from "../../../core/diagram.js";
import { fmt } from "../../../core/dom.js";
import { label } from "../../../core/visuals.js";
import { residualArc } from "../architecture.js";
import { KIND_COLORS } from "./archLevels.js";
import { disposeTree, fitBlock } from "./lodBlocks.js";
import { ROLE } from "./neuronModel.js";

export const HEAD_COLORS = ["#c084fc", "#34d399", "#f472b6", "#facc15"];
const POS = [251, 146, 60].map((v) => v / 255); // arancione: positivo (come le heatmap dei pesi)
const NEG = [59, 130, 246].map((v) => v / 255); // blu: negativo
const BASE = [0.3, 0.34, 0.42]; // neurone a zero: grigio visibile sullo sfondo scuro
const PENDING = [0.13, 0.16, 0.21]; // non ancora calcolato (animazione)
const STRIP_ROW = 0.4; // altezza di una riga della striscia dell'attention
const STRIP_W = 2.2;

/** Descrizioni delle colonne per il livello Blocchi. */
const COLUMN_TEXT = {
  tok_onehot: ["65 neuroni", "Un solo neurone acceso: il carattere in ingresso."],
  pos_onehot: ["32 neuroni", "Acceso solo quello della posizione t."],
  tok_emb: ["64 · tabella", "La riga della tabella del token."],
  pos_emb: ["64 · tabella", "La riga della tabella della posizione."],
  x0: ["64 · somma", "Token + posizione, numero per numero."],
  x: ["64 · ingresso", "Lo stesso vettore uscito dalla riga sotto."],
  ln1: ["64 · γ, β", "Normalizza: media 0, varianza 1."],
  qkv: ["4 head × 3 × 16", "Query, key e value di ogni head."],
  ctx: ["0 pesi", "Coefficienti calcolati dai dati: softmax(q·k)."],
  concat: ["64 = 4 × 16", "Le uscite delle head affiancate."],
  proj: ["64 + bias", "Rimescola le head."],
  resid1: ["residuo", "x + attention"],
  ln2: ["64 · γ, β", "Normalizza di nuovo."],
  hidden: ["256 · ReLU", "Espande e scarta i negativi."],
  ffnOut: ["64 + bias", "Comprime di nuovo a 64."],
  resid2: ["residuo", "x + feed-forward"],
  ln_f: ["64 · γ, β", "Normalizzazione finale."],
  logits: ["65 + bias", "Un punteggio per carattere."],
  probs: ["65", "Probabilità che sommano a 1."],
};

function columnColor(col) {
  if (col.role === ROLE.input || col.key.endsWith("_emb")) return KIND_COLORS.embedding;
  if (col.role === ROLE.output) return KIND_COLORS.output;
  if (col.key.startsWith("ln")) return KIND_COLORS.norm;
  if (["qkv", "ctx", "concat", "proj"].includes(col.key)) return KIND_COLORS.attention;
  if (["hidden", "ffnOut"].includes(col.key)) return KIND_COLORS.ffwd;
  return KIND_COLORS.data;
}

export class NeuronScene {
  /** @param {ReturnType<import("./neuronModel.js").buildNeuronLayout>} layout */
  constructor(layout, spec) {
    this.layout = layout;
    this.spec = spec;
    this.detail = new THREE.Group();
    this.blocks = new THREE.Group();
    this.macro = new THREE.Group();
    this.detail.name = "neurons-detail";
    this.blocks.name = "neurons-blocks";
    this.macro.name = "neurons-macro";
    this.flows = [];
    this.tmp = new THREE.Vector3();
    this.colorBuffer = new Float32Array(layout.count * 3);
    this.buildNeurons();
    this.buildStructure();
    this.buildBlocksLevel();
    this.buildMacroLevel();
    this.buildAttention();
    this.buildSelection();
  }

  // ------------------------------------------------------------------ neuroni
  buildNeurons() {
    const { count, positions, radii } = this.layout;
    const mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), // 80 triangoli: 4.387 sfere restano leggere
      new THREE.MeshBasicMaterial({ color: 0xffffff }), count);
    const m = new THREE.Matrix4();
    for (let k = 0; k < count; k++) {
      const r = radii[k];
      m.makeScale(r, r, r).setPosition(positions[3 * k], positions[3 * k + 1], positions[3 * k + 2]);
      mesh.setMatrixAt(k, m);
    }
    mesh.instanceColor = new THREE.InstancedBufferAttribute(this.colorBuffer, 3);
    mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    this.neurons = mesh;
    this.detail.add(mesh);
  }

  /**
   * Colori dei neuroni: segno e intensità dell'attivazione, normalizzata per colonna.
   * @param {Float32Array} values @param {(colIndex: number) => number} reveal 0 = non calcolato, 1 = tutto
   */
  setValues(values, reveal = () => 1) {
    const { columns } = this.layout;
    const c = this.colorBuffer;
    columns.forEach((col, ci) => {
      if (!col.n) return;
      const shown = Math.round(reveal(ci) * col.n);
      let max = 1e-9;
      for (let i = 0; i < col.n; i++) max = Math.max(max, Math.abs(values[col.offset + i]));
      for (let i = 0; i < col.n; i++) {
        const k = col.offset + i;
        let rgb;
        let a;
        if (i >= shown) {
          c.set(PENDING, 3 * k);
          continue;
        }
        const v = values[k];
        if (col.key === "probs") {
          a = Math.pow(v / max, 0.5);
          rgb = [0.4, 0.91, 0.98];
        } else if (col.role === ROLE.input) {
          a = v > 0 ? 1 : 0;
          rgb = [0.88, 0.95, 1];
        } else {
          a = Math.pow(Math.abs(v) / max, 0.6);
          rgb = v >= 0 ? POS : NEG;
        }
        c[3 * k] = BASE[0] + (rgb[0] - BASE[0]) * a;
        c[3 * k + 1] = BASE[1] + (rgb[1] - BASE[1]) * a;
        c[3 * k + 2] = BASE[2] + (rgb[2] - BASE[2]) * a;
      }
    });
    this.neurons.instanceColor.needsUpdate = true;
  }

  // ------------------------------------------------------------------ archi
  /** Scrive tutti gli archi nell'ordine dal più forte (una volta per insieme di pesi). */
  setEdges(edges) {
    const { positions } = this.layout;
    const n = edges.count;
    const pos = new Float32Array(n * 6);
    const col = new Float32Array(n * 8);
    for (let s = 0; s < n; s++) {
      const e = edges.order[s];
      const a = 3 * edges.from[e];
      const b = 3 * edges.to[e];
      pos[6 * s] = positions[a];
      pos[6 * s + 1] = positions[a + 1];
      pos[6 * s + 2] = -0.02;
      pos[6 * s + 3] = positions[b];
      pos[6 * s + 4] = positions[b + 1];
      pos[6 * s + 5] = -0.02;
      const rgb = edges.weight[e] >= 0 ? POS : NEG;
      const alpha = 0.12 + 0.88 * Math.pow(edges.strength[e], 0.8);
      for (let v = 0; v < 2; v++) {
        col[8 * s + 4 * v] = rgb[0];
        col[8 * s + 4 * v + 1] = rgb[1];
        col[8 * s + 4 * v + 2] = rgb[2];
        col[8 * s + 4 * v + 3] = alpha;
      }
    }
    if (!this.edgeGeometry) {
      this.edgeGeometry = new THREE.BufferGeometry();
      const mat = () => new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, opacity: 1 });
      this.edgeLines = new THREE.LineSegments(this.edgeGeometry, mat());
      this.edgeLinesFar = new THREE.LineSegments(this.edgeGeometry, mat()); // livello Blocchi: stessa geometria, materiale suo
      this.edgeLines.frustumCulled = false;
      this.edgeLinesFar.frustumCulled = false;
      this.detail.add(this.edgeLines);
      this.blocks.add(this.edgeLinesFar);
    }
    else this.edgeGeometry.dispose(); // pesi cambiati (es. snapshot del training): libera i vecchi buffer sulla GPU
    this.edgeGeometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.edgeGeometry.setAttribute("color", new THREE.BufferAttribute(col, 4));
    this.edgeCount = n;
  }

  /**
   * Mostra la frazione `f` (0…1) degli archi più forti di ogni matrice.
   * @param {(m: THREE.Material, o: number) => void} setOpacity compatibile con il LOD
   */
  setFraction(f, setOpacity) {
    const shown = Math.max(1, Math.round(this.edgeCount * f));
    this.edgeGeometry.setDrawRange(0, shown * 2);
    // Più archi = ognuno più trasparente, altrimenti tutto diventa una macchia.
    const opacity = Math.max(0.06, Math.min(1, Math.pow(0.02 / f, 0.7)));
    this.edgeOpacity = opacity;
    setOpacity(this.edgeLines.material, opacity * (this.dimEdges ? 0.25 : 1));
    setOpacity(this.edgeLinesFar.material, opacity * 0.35);
    return shown;
  }

  // ------------------------------------------------------------------ struttura fissa: etichette, residui, collegamenti
  buildStructure() {
    const { columns, rows, positions, byId } = this.layout;
    const top = (col) => col.yTop + col.y;
    // Etichette delle colonne
    for (const col of columns) {
      const text = col.n ? `${col.short} (${col.n})` : col.short;
      const tag = label(text, { size: 0.3, color: columnColor(col) });
      const above = col.id === "pos_onehot" || col.id === "pos_emb" ? col.y + col.yBottom - 0.55 : top(col) + 0.55;
      tag.position.set(col.x, col.key === "ctx" ? col.y + 7.2 : above, 0.2);
      this.detail.add(tag);
    }
    // Q, K, V: tre gruppi nella stessa colonna (4 head da 16 ciascuno)
    for (let l = 0; l < this.spec.config.n_layer; l++) {
      const col = byId.get(`b${l}.qkv`);
      const C = this.spec.config.n_embd;
      ["Q", "K", "V"].forEach((name, part) => {
        const a = positions[3 * (col.offset + part * C) + 1];
        const b = positions[3 * (col.offset + part * C + C - 1) + 1];
        const tag = label(name, { size: 0.36, color: "#7dd3fc" });
        tag.position.set(col.x - 0.55, (a + b) / 2, 0.1);
        this.detail.add(tag);
      });
    }
    // Titoli delle righe
    for (const row of rows) {
      const first = columns.find((c) => c.row === row.index);
      const tag = label(row.title, { size: 0.7, color: "#7dd3fc" });
      tag.center.set(1, 0.5);
      tag.position.set(first.x - 1.4, row.y, 0);
      this.detail.add(tag);
    }
    // Collegamenti fissi uno a uno (somme, copie, softmax): grigi, non sono pesi.
    const pairs = [["tok_emb", "x0"], ["pos_emb", "x0"], ["logits", "probs"]];
    for (let l = 0; l < this.spec.config.n_layer; l++) pairs.push([`b${l}.proj`, `b${l}.resid1`], [`b${l}.ffnOut`, `b${l}.resid2`]);
    const seg = [];
    for (const [a, b] of pairs) {
      const A = byId.get(a);
      const B = byId.get(b);
      for (let i = 0; i < Math.min(A.n, B.n); i++) {
        const p = 3 * (A.offset + i);
        const q = 3 * (B.offset + i);
        seg.push(positions[p], positions[p + 1], -0.02, positions[q], positions[q + 1], -0.02);
      }
    }
    const idGeo = new THREE.BufferGeometry();
    idGeo.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
    this.detail.add(new THREE.LineSegments(idGeo, new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.45, depthWrite: false })));

    // Connessioni residue: archi sopra le colonne del blocco (peso fisso 1).
    for (let l = 0; l < this.spec.config.n_layer; l++) {
      const x = byId.get(`b${l}.x`);
      const r1 = byId.get(`b${l}.resid1`);
      const r2 = byId.get(`b${l}.resid2`);
      const arcs = new THREE.Group();
      arcs.position.set(0, x.y + x.yTop + 1.1, 0.1);
      arcs.add(residualArc(x.x, r1.x, 1.6, { radius: 0.05, opacity: 0.8, z: 0 }), residualArc(r1.x, r2.x, 1.6, { radius: 0.05, opacity: 0.8, z: 0 }));
      const tag = label("connessione residua: copia uno a uno, peso fisso 1", { size: 0.24, color: "#7dd3fc" });
      tag.position.set((x.x + r2.x) / 2, 2.0, 0);
      arcs.add(tag);
      this.detail.add(arcs);
    }
    // Il vettore x passa da una riga alla successiva (come il binario dell'architettura).
    const ends = [["x0", "b0.x"], ...[...Array(this.spec.config.n_layer).keys()].map((l) => [`b${l}.resid2`, l + 1 < this.spec.config.n_layer ? `b${l + 1}.x` : "out.x"])];
    for (const [a, b] of ends) {
      const A = byId.get(a);
      const B = byId.get(b);
      const midY = (A.y + B.y) / 2;
      const flow = makeFlow([new THREE.Vector3(A.x + 0.6, A.y, 0), new THREE.Vector3(A.x + 1.6, A.y, 0), new THREE.Vector3(A.x + 1.6, midY, 0), new THREE.Vector3(B.x - 1.6, midY, 0), new THREE.Vector3(B.x - 1.6, B.y, 0), new THREE.Vector3(B.x - 0.6, B.y, 0)], { radius: 0.035, speed: 1.2 });
      this.detail.add(flow);
      this.flows.push(flow);
    }
  }

  // ------------------------------------------------------------------ livello Blocchi: un rettangolo per strato
  buildBlocksLevel() {
    const { columns, rows } = this.layout;
    for (const col of columns) {
      const same = columns.filter((c) => c.row === col.row && c !== col && Math.abs(c.x - col.x) > 0.01);
      const gap = Math.min(...same.map((c) => Math.abs(c.x - col.x)), 4);
      const width = Math.max(1.6, Math.min(3.4, gap - 0.35));
      const height = col.key === "ctx" ? 13.2 : col.yTop - col.yBottom + 0.7;
      const [sub, text] = COLUMN_TEXT[col.key] ?? ["", ""];
      const block = fitBlock({ title: col.short, subtitle: sub, text, width, height, maxScale: 1.7, color: columnColor(col), fill: "rgba(15,23,42,0.62)", bodyOpacity: 0.35, depth: 0.1 });
      block.position.set(col.x, col.y + (col.yTop + col.yBottom) / 2, 0.25);
      this.blocks.add(block);
    }
    for (const row of rows) {
      const first = columns.find((c) => c.row === row.index);
      const tag = label(row.title, { size: 1.1, color: "#7dd3fc" });
      tag.center.set(1, 0.5);
      tag.position.set(first.x - 2, row.y, 0);
      this.blocks.add(tag);
    }
  }

  // ------------------------------------------------------------------ livello Panoramica: un rettangolo per riga
  buildMacroLevel() {
    const { columns, rows } = this.layout;
    const params = this.spec.params;
    const sum = (test) => params.filter((p) => test(p.name)).reduce((a, p) => a + p.numel, 0);
    const neuronsIn = (row) => columns.filter((c) => c.row === row.index && c.role !== ROLE.copy).reduce((a, c) => a + c.n, 0);
    const L = this.spec.config.n_layer;
    const blocksOut = [];
    for (const row of rows) {
      const cols = columns.filter((c) => c.row === row.index);
      const x0 = Math.min(...cols.map((c) => c.x)) - 1.5;
      const x1 = Math.max(...cols.map((c) => c.x)) + 1.5;
      let title;
      let n;
      let text;
      if (row.index === 0) {
        title = "Ingresso";
        n = sum((name) => name.includes("embedding_table"));
        text = "One-hot del carattere e della posizione → due tabelle di embedding → x.";
      } else if (row.index > L) {
        title = "Uscita";
        n = sum((name) => name.startsWith("ln_f.") || name.startsWith("lm_head."));
        text = "LayerNorm finale → 65 logits → softmax.";
      } else {
        title = `Blocco ${row.layer}`;
        n = sum((name) => name.startsWith(`blocks.${row.layer}.`));
        text = "LayerNorm → Q, K, V → attention → proiezione → + → LayerNorm → 256 nascosti → 64 → +";
      }
      const block = fitBlock({
        title,
        subtitle: `${fmt(neuronsIn(row))} neuroni · ${fmt(n)} parametri`,
        text,
        width: Math.max(12, x1 - x0),
        height: 14,
        maxScale: 6.5,
        color: row.index === 0 ? KIND_COLORS.embedding : row.index > L ? KIND_COLORS.output : "#7dd3fc",
      });
      block.position.set((x0 + x1) / 2, row.y, 0);
      this.macro.add(block);
      blocksOut.push(block);
    }
    for (let i = 1; i < blocksOut.length; i++) {
      const a = blocksOut[i - 1].position.y + 7;
      const b = blocksOut[i].position.y - 7;
      this.macro.add(makeArrow(new THREE.Vector3(0, a + 0.1, 0.3), new THREE.Vector3(0, b - 0.1, 0.3), { color: 0x67e8f9, radius: 0.3, headLength: 1.3 }));
    }
  }

  // ------------------------------------------------------------------ attention: striscia delle posizioni e archi tratteggiati
  buildAttention() {
    const { n_layer: L, n_head: H, block_size: T } = this.spec.config;
    this.strips = [];
    for (let l = 0; l < L; l++) {
      const col = this.layout.byId.get(`b${l}.ctx`);
      const canvas = document.createElement("canvas");
      canvas.width = 220;
      canvas.height = T * 24;
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(STRIP_W, T * STRIP_ROW), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }));
      plane.position.set(col.x, col.y + col.yTop - (T * STRIP_ROW) / 2, 0.05);
      this.detail.add(plane);
      this.strips.push({ canvas, texture, plane, col });
    }
    // Archi tratteggiati: L × H × T curve da 8 segmenti, preallocati.
    this.arcSegments = 8;
    const maxSeg = L * H * T * this.arcSegments;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(maxSeg * 6), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(maxSeg * 8), 4));
    geo.setDrawRange(0, 0);
    this.attLines = new THREE.LineSegments(geo, new THREE.LineDashedMaterial({ vertexColors: true, transparent: true, depthWrite: false, dashSize: 0.18, gapSize: 0.12 }));
    this.attLines.frustumCulled = false;
    this.detail.add(this.attLines);
  }

  /** Disegna l'attention vera della posizione t (trace del motore). */
  setAttention(trace, t, vocab) {
    const { n_layer: L, n_head: H } = this.spec.config;
    const T = trace.T;
    const hs = this.layout.headSize;
    const pos = this.attLines.geometry.attributes.position.array;
    const colr = this.attLines.geometry.attributes.color.array;
    const headRgb = HEAD_COLORS.map((h) => new THREE.Color(h));
    let s = 0;
    const p0 = new THREE.Vector3();
    const p1 = new THREE.Vector3();
    const pc = new THREE.Vector3();
    const curve = new THREE.QuadraticBezierCurve3(p0, pc, p1);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    for (let l = 0; l < L; l++) {
      const { canvas, texture, col } = this.strips[l];
      const concat = this.layout.byId.get(`b${l}.concat`);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "600 17px 'JetBrains Mono', ui-monospace, monospace";
      ctx.textBaseline = "middle";
      for (let src = 0; src <= t; src++) {
        const y = src * 24;
        ctx.fillStyle = src === t ? "rgba(56,189,248,0.25)" : "rgba(30,41,59,0.85)";
        ctx.fillRect(0, y + 1, canvas.width, 22);
        const ch = vocab[trace.ids[src]];
        ctx.fillStyle = "#e2e8f0";
        ctx.fillText(ch === " " ? "␣" : ch === "\n" ? "↵" : ch, 8, y + 12);
        ctx.fillStyle = "#64748b";
        ctx.fillText(String(src).padStart(2, " "), 30, y + 12);
        for (let h = 0; h < H; h++) {
          const w = trace.blocks[l].heads[h].att[t * T + src];
          const c = headRgb[h];
          ctx.fillStyle = `rgba(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)},${0.08 + 0.92 * Math.sqrt(w)})`;
          ctx.fillRect(66 + h * 38, y + 3, 34, 18);
        }
      }
      texture.needsUpdate = true;
      // Archi: dalla riga `src` della striscia al gruppo della head h nella colonna concat.
      for (let h = 0; h < H; h++) {
        const first = 3 * (concat.offset + h * hs);
        const last = 3 * (concat.offset + h * hs + hs - 1);
        const hy = (this.layout.positions[first + 1] + this.layout.positions[last + 1]) / 2;
        const c = headRgb[h];
        for (let src = 0; src <= t; src++) {
          const w = trace.blocks[l].heads[h].att[t * T + src];
          if (w < 0.02) continue; // sotto il 2% l'arco sarebbe invisibile: lo saltiamo
          p0.set(col.x + STRIP_W / 2, col.y + col.yTop - (src + 0.5) * STRIP_ROW, 0.05);
          p1.set(concat.x - 0.15, hy, 0.05);
          pc.set((p0.x + p1.x) / 2, (p0.y + p1.y) / 2 + 0.6 * (h - 1.5), 0.05);
          for (let k = 0; k < this.arcSegments; k++) {
            curve.getPoint(k / this.arcSegments, a);
            curve.getPoint((k + 1) / this.arcSegments, b);
            pos.set([a.x, a.y, a.z, b.x, b.y, b.z], 6 * s);
            const alpha = 0.15 + 0.85 * Math.sqrt(w);
            colr.set([c.r, c.g, c.b, alpha, c.r, c.g, c.b, alpha], 8 * s);
            s++;
          }
        }
      }
    }
    const geo = this.attLines.geometry;
    geo.setDrawRange(0, s * 2);
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    this.attLines.computeLineDistances(); // solo al cambio di prompt o posizione, non a ogni frame
  }

  // ------------------------------------------------------------------ selezione di un neurone
  buildSelection() {
    this.marker = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.9 }));
    this.marker.visible = false;
    this.detail.add(this.marker);
    const max = 512;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(max * 6), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(max * 8), 4));
    geo.setDrawRange(0, 0);
    this.inLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    this.inLines.frustumCulled = false;
    this.inLines.renderOrder = 3;
    this.detail.add(this.inLines);
    this.maxIn = max;
  }

  /**
   * Evidenzia un neurone e i suoi archi in ingresso (colore = segno del peso, opacità = |w·x|).
   * @param {number|null} neuron @param {{from, to, weight}} edges @param {Float32Array} values
   */
  select(neuron, edges, values) {
    const geo = this.inLines.geometry;
    if (neuron === null) {
      this.marker.visible = false;
      geo.setDrawRange(0, 0);
      return;
    }
    const { positions, radii } = this.layout;
    const r = radii[neuron] * 2.2;
    this.marker.position.set(positions[3 * neuron], positions[3 * neuron + 1], 0);
    this.marker.scale.setScalar(Math.max(r, 0.08));
    this.marker.visible = true;
    // Archi in ingresso: scansione lineare (una volta per clic).
    const found = [];
    let maxC = 1e-9;
    for (let e = 0; e < edges.count; e++) {
      if (edges.to[e] !== neuron) continue;
      const contrib = Math.abs(edges.weight[e] * values[edges.from[e]]);
      found.push([e, contrib]);
      maxC = Math.max(maxC, contrib);
    }
    const pos = geo.attributes.position.array;
    const col = geo.attributes.color.array;
    const n = Math.min(found.length, this.maxIn);
    for (let s = 0; s < n; s++) {
      const [e, contrib] = found[s];
      const a = 3 * edges.from[e];
      pos.set([positions[a], positions[a + 1], 0.03, positions[3 * neuron], positions[3 * neuron + 1], 0.03], 6 * s);
      const rgb = edges.weight[e] >= 0 ? POS : NEG;
      const alpha = 0.15 + 0.85 * Math.sqrt(contrib / maxC);
      col.set([rgb[0], rgb[1], rgb[2], alpha, rgb[0], rgb[1], rgb[2], alpha], 8 * s);
    }
    geo.setDrawRange(0, n * 2);
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
  }

  /** Neurone più vicino al raggio (tolleranza in proporzione alla distanza: i neuroni piccoli restano cliccabili). */
  pick(ray) {
    const { count, positions, radii } = this.layout;
    let best = null;
    let bestScore = Infinity;
    const p = this.tmp;
    for (let k = 0; k < count; k++) {
      p.set(positions[3 * k], positions[3 * k + 1], positions[3 * k + 2]);
      const d2 = ray.distanceSqToPoint(p);
      const tol = Math.max(radii[k] * 1.6, ray.origin.distanceTo(p) * 0.008);
      if (d2 < tol * tol && d2 / (tol * tol) < bestScore) {
        bestScore = d2 / (tol * tol);
        best = k;
      }
    }
    return best;
  }

  tick(dt) {
    for (const f of this.flows) f.userData.tick(dt);
  }

  /** Riquadro di una colonna (o di più colonne) per inquadrarla con la camera. */
  boxOf(columnIds) {
    const box = new THREE.Box3();
    for (const id of columnIds) {
      const col = this.layout.byId.get(id);
      const half = col.key === "ctx" ? STRIP_W / 2 : col.sub > 1 ? 0.9 : 0.4;
      box.expandByPoint(this.tmp.set(col.x - half, col.y + col.yBottom - 0.3, 0));
      box.expandByPoint(this.tmp.set(col.x + half, col.y + col.yTop + 0.9, 0));
    }
    return box;
  }

  dispose() {
    for (const g of [this.detail, this.blocks, this.macro]) {
      g.removeFromParent();
      disposeTree(g);
    }
  }
}
