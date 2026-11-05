/**
 * Mattoni 3D comuni alle stazioni del tab RAG: posizioni delle stazioni lungo il percorso,
 * contenitore di una stazione (titolo + riquadro per la camera), barre di un vettore, colori.
 *
 * Percorso a "U": in alto l'indicizzazione da sinistra a destra (Upload → … → Indice HNSW),
 * in basso la ricerca da destra a sinistra (Ricerca → Prompt → LLM → Risposta), così il
 * database vettoriale sta nell'angolo dove le due metà si incontrano.
 */
import * as THREE from "three";
import { makeBlock } from "../../core/diagram.js";
import { label } from "../../core/visuals.js";

export const ROW_TOP = 0;
export const ROW_BOTTOM = -46;
export const STATION_H = 26;

/** Centro e larghezza di ogni stazione (unità di scena). */
export const STATIONS = Object.freeze({
  upload: { x: -132, y: ROW_TOP, w: 32, n: 1, title: "Upload" },
  extract: { x: -94, y: ROW_TOP, w: 32, n: 2, title: "Estrazione del testo" },
  clean: { x: -56, y: ROW_TOP, w: 30, h: 29, n: 3, title: "Pulizia e sicurezza" },
  chunk: { x: -20, y: ROW_TOP, w: 26, h: 30, n: 4, title: "Chunking" },
  embed: { x: 14, y: ROW_TOP, w: 30, h: 28, n: 5, title: "Embedding" },
  table: { x: 54, y: ROW_TOP, w: 34, n: 6, title: "pgvector: vector_store" },
  hnsw: { x: 112, y: ROW_TOP, w: 62, n: 7, title: "Indice HNSW" },
  search: { x: 112, y: ROW_BOTTOM, w: 50, n: 8, title: "Ricerca" },
  prompt: { x: 62, y: ROW_BOTTOM, w: 38, n: 9, title: "Prompt strutturato" },
  llm: { x: 22, y: ROW_BOTTOM, w: 30, n: 10, title: "LLM (ChatClient)" },
  answer: { x: -20, y: ROW_BOTTOM, w: 40, n: 11, title: "Risposta in streaming" },
});

/** Un colore per documento (ordine di caricamento), leggibile su sfondo scuro. */
export const DOC_COLORS = ["#fb923c", "#38bdf8", "#a78bfa", "#4ade80", "#f472b6", "#facc15", "#2dd4bf"];

export const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const clamp01 = (t) => Math.min(1, Math.max(0, t));
/** Porzione [a, b] di un avanzamento p ∈ [0, 1], riscalata a [0, 1]. */
export const phase = (p, a, b) => clamp01((p - a) / (b - a));

/**
 * Contenitore di una stazione: gruppo posizionato, titolo e un box invisibile usato dalla
 * camera per inquadrarla (meshBounds conta anche i mesh invisibili).
 */
export function makeStation(id) {
  const spec = STATIONS[id];
  const group = new THREE.Group();
  group.position.set(spec.x, spec.y, 0);
  group.name = `station-${id}`;
  const h = spec.h ?? STATION_H;
  // il riquadro per la camera comprende anche il titolo (sprite, che meshBounds non conta)
  const bounds = new THREE.Mesh(new THREE.BoxGeometry(spec.w, h + 4, 2), new THREE.MeshBasicMaterial({ visible: false }));
  bounds.position.y = 2;
  bounds.visible = false;
  group.add(bounds);
  const title = label(`${spec.n} · ${spec.title}`, { size: 1.6, color: "#e0f2fe" });
  title.position.set(0, h / 2 + 1.6, 0);
  group.add(title);
  return { id, spec, group, bounds, title };
}

/**
 * Le componenti di un vettore come griglia di celle (una per dimensione, riga per riga):
 * arancione = valore positivo, blu = negativo, più intenso = più grande in valore assoluto.
 * Un solo InstancedMesh; aggiornare i colori non alloca niente.
 */
export class VectorGrid {
  constructor({ dims, size = 12 }) {
    this.dims = dims;
    this.cols = Math.ceil(Math.sqrt(dims));
    const cell = size / this.cols;
    this.mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(cell * 0.86, cell * 0.86), new THREE.MeshBasicMaterial({ transparent: true }), dims);
    this.mesh.userData.noBounds = true;
    const dummy = new THREE.Object3D();
    for (let i = 0; i < dims; i++) {
      const r = Math.floor(i / this.cols);
      const c = i % this.cols;
      dummy.position.set((c + 0.5) * cell - size / 2, size / 2 - (r + 0.5) * cell, 0);
      dummy.updateMatrix();
      this.mesh.setMatrixAt(i, dummy.matrix);
    }
    this.frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(size + 0.3, size + 0.3)),
      new THREE.LineBasicMaterial({ color: 0x475569, transparent: true }),
    );
    this.group = new THREE.Group();
    this.group.add(this.mesh, this.frame);
    this.colors = { empty: new THREE.Color("#1e293b"), pos: new THREE.Color("#fb923c"), neg: new THREE.Color("#60a5fa"), tmp: new THREE.Color() };
    this.set(null, 0);
  }

  /** Mostra il vettore v; `grow` ∈ [0, 1] = frazione di celle già "accese" (riga per riga). */
  set(v, grow = 1) {
    const { empty, pos, neg, tmp } = this.colors;
    // scala robusta: la dimensione 0 (componente comune dell'embedder didattico) non conta
    let max = 1e-6;
    if (v) for (let i = 1; i < this.dims; i++) max = Math.max(max, Math.abs(v[i]));
    const lit = Math.round(grow * this.dims);
    for (let i = 0; i < this.dims; i++) {
      if (!v || i >= lit) tmp.copy(empty);
      else {
        const a = Math.pow(Math.min(1, Math.abs(v[i]) / max), 0.6);
        tmp.lerpColors(empty, v[i] >= 0 ? pos : neg, a);
      }
      this.mesh.setColorAt(i, tmp);
    }
    this.mesh.instanceColor.needsUpdate = true;
  }
}

/**
 * makeBlock con testo ingrandito: il blocco si costruisce piccolo (w/S × h/S) e si scala di S,
 * così titolo e descrizione crescono con il rettangolo e restano leggibili.
 *
 * Scalando, la faccia col testo resta a 0.01·S dal corpo del blocco e da lontano il depth buffer
 * non distingue più le due superfici (z-fighting: righe scure sul testo). Il polygon offset tiene
 * la faccia sempre davanti al corpo.
 */
export function bigBlock(spec, { w, h, S = 2.2 }) {
  const block = makeBlock({ ...spec, width: w / S, height: h / S });
  block.scale.setScalar(S);
  block.traverse((o) => {
    if (!o.material?.map) return;
    o.material.polygonOffset = true;
    o.material.polygonOffsetFactor = -2;
    o.material.polygonOffsetUnits = -4;
  });
  return block;
}

/** Posiziona un oggetto lungo un arco tra due punti (per i "voli" di carte e fogli). */
export function arcLerp(out, from, to, t, lift = 4) {
  out.lerpVectors(from, to, t);
  out.y += Math.sin(Math.PI * t) * lift;
  out.z += Math.sin(Math.PI * t) * lift * 0.4;
  return out;
}
