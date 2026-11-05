/**
 * Livelli astratti del LOD:
 *   "Blocchi"    → un rettangolo per fase (Upload … Risposta) nella stessa posizione della sua
 *                  stazione, con titolo e descrizione di cosa fa;
 *   "Panoramica" → tre blocchi grandi: Indicizzazione → Database vettoriale → Recupero + Generazione.
 *
 * Il testo di makeBlock ha un'altezza fissa in unità di scena: costruiamo blocchi piccoli e li
 * ingrandiamo con `scale` (S) così titolo e descrizione restano leggibili da lontano.
 */
import * as THREE from "three";
import { makeArrow, makeBlock } from "../../core/diagram.js";
import { label } from "../../core/visuals.js";
import { ROW_BOTTOM, ROW_TOP, STATIONS } from "./scene3d.js";

const PHASES = {
  upload: { subtitle: "React → Spring Boot", text: "Il PDF arriva al DocumentController di Spring Boot, che lo salva con un nome UUID.", color: "#4ade80" },
  extract: { subtitle: "lettore PDF", text: "PagePdfDocumentReader: un Document per pagina, con testo e metadati.", color: "#4ade80" },
  clean: { subtitle: "pre-processing", text: "Toglie intestazioni e a capo del PDF; segnala testi sospetti di prompt injection.", color: "#fb923c" },
  chunk: { subtitle: "TokenTextSplitter", text: "Divide il testo in pezzi di pochi token, tagliando a fine frase.", color: "#38bdf8" },
  embed: { subtitle: "EmbeddingModel", text: "Ogni chunk diventa un vettore di numeri che ne riassume il significato.", color: "#c084fc" },
  table: { subtitle: "vectorStore.add()", text: "Ogni chunk è una riga: id, content, metadata, embedding.", color: "#38bdf8" },
  hnsw: { subtitle: "grafo a livelli", text: "Collega ogni vettore ai vicini: la ricerca salta tra pochi nodi invece di leggerli tutti.", color: "#38bdf8" },
  search: { subtitle: "similaritySearch", text: "La domanda diventa un vettore; si prendono i topK chunk sopra la soglia.", color: "#fde047" },
  prompt: { subtitle: "system + doc + domanda", text: "I chunk recuperati entrano nel prompt come dati, con fonte e pagina.", color: "#c084fc" },
  llm: { subtitle: "ChatClient", text: "Il modello legge il prompt e genera la risposta un token alla volta.", color: "#c084fc" },
  answer: { subtitle: "SSE → React", text: "I token arrivano al frontend in streaming, insieme alle fonti.", color: "#e2e8f0" },
};

function scaledBlock(spec, { x, y, w, h, S }) {
  const b = makeBlock({ ...spec, width: w / S, height: h / S });
  b.scale.setScalar(S);
  b.position.set(x, y, 0);
  return b;
}

/** Freccia tra due punti (array [x, y]) appena davanti ai blocchi. */
function arrow(group, a, b, { color = 0x67e8f9, radius = 0.4, head = 2.4 } = {}) {
  group.add(makeArrow(new THREE.Vector3(a[0], a[1], 0.4), new THREE.Vector3(b[0], b[1], 0.4), { color, radius, headLength: head }));
}

export function buildBlocks() {
  const group = new THREE.Group();
  const S = 9;
  const H = 24;
  const ids = Object.keys(STATIONS);
  const boxes = {};
  for (const id of ids) {
    const s = STATIONS[id];
    const w = Math.min(Math.max(s.w - 2, 29), 44);
    boxes[id] = { x: s.x, y: s.y, w };
    group.add(scaledBlock({ title: `${s.n}. ${s.title}`, ...PHASES[id] }, { x: s.x, y: s.y, w, h: H, S }));
  }
  // frecce del flusso: riga in alto verso destra, poi giù, poi riga in basso verso sinistra
  for (let i = 0; i < 6; i++) {
    const a = boxes[ids[i]];
    const b = boxes[ids[i + 1]];
    arrow(group, [a.x + a.w / 2 + 0.3, ROW_TOP], [b.x - b.w / 2 - 0.3, ROW_TOP]);
  }
  arrow(group, [STATIONS.hnsw.x, ROW_TOP - H / 2 - 0.3], [STATIONS.search.x, ROW_BOTTOM + H / 2 + 0.3], { color: 0xfde047 });
  for (let i = 7; i < 10; i++) {
    const a = boxes[ids[i]];
    const b = boxes[ids[i + 1]];
    arrow(group, [a.x - a.w / 2 - 0.3, ROW_BOTTOM], [b.x + b.w / 2 + 0.3, ROW_BOTTOM], { color: 0xfde047 });
  }
  const top = label("Indicizzazione: una volta per ogni documento caricato", { size: 4.5, color: "#bae6fd" });
  top.position.set(-40, ROW_TOP + H / 2 + 5.5, 0);
  const bottom = label("Ricerca e generazione: a ogni domanda", { size: 4.5, color: "#fef08a" });
  bottom.position.set(40, ROW_BOTTOM - H / 2 - 5.5, 0);
  group.add(top, bottom);
  return { group };
}

export function buildMacro() {
  const group = new THREE.Group();
  const S = 26;
  const y0 = (ROW_TOP + ROW_BOTTOM) / 2;
  const W = 116;
  const Hm = 64;
  group.add(
    scaledBlock(
      { title: "Indicizzazione", subtitle: "PDF → chunk → vettori", text: "Si fa una volta per documento: estrazione, pulizia, chunking, embedding.", color: "#4ade80" },
      { x: -150, y: y0, w: W, h: Hm, S },
    ),
    scaledBlock(
      { title: "Database vettoriale", subtitle: "PostgreSQL + pgvector", text: "Tabella vector_store e indice HNSW: trova i vettori vicini in fretta.", color: "#38bdf8" },
      { x: 0, y: y0, w: W, h: Hm, S },
    ),
    scaledBlock(
      { title: "Recupero + risposta", subtitle: "ricerca → prompt → LLM", text: "A ogni domanda: i chunk più simili entrano nel prompt e l'LLM risponde.", color: "#fde047" },
      { x: 150, y: y0, w: W, h: Hm, S },
    ),
  );
  arrow(group, [-91, y0], [-59, y0], { radius: 2, head: 9 });
  arrow(group, [59, y0], [91, y0], { color: 0xfde047, radius: 2, head: 9 });
  const t = label("RAG: Retrieval Augmented Generation", { size: 12, color: "#e0f2fe" });
  t.position.set(0, y0 + Hm / 2 + 14, 0);
  group.add(t);
  return { group };
}
