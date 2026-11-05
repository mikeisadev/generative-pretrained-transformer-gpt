/**
 * Geometria del diagramma "Contesto" (coordinate del mondo, z = 0 è il piano della pila).
 *
 * Il diagramma sta a SINISTRA della pila del transformer (x tra -120 e -28; la pila occupa
 * x tra -22 e +22): è la fase che prepara l'input. Il percorso fa un anello:
 *
 *   React ─▶ guardrail ─▶ embedding ─▶ pgvector ─▶ chunk ─▶ contesto ─▶ token ─▶ [pila: LLM]
 *     ▲                                                     ▲                        │
 *     │                                       risultato ────┘        richiesta di    │
 *     │                                       strumento ◀── Java ◀── strumento ◀─────┤
 *     └──────────────── streaming SSE ◀──────────────────────────────────────────────┘
 *
 * Ogni fase ha uno "slot" (centro e dimensioni): lo usano allo stesso modo il livello Dettaglio
 * (schede con i dati veri), il livello Blocchi (un rettangolo con titolo e descrizione) e il
 * pacchetto animato, così a ogni zoom le cose restano allineate.
 */
import * as THREE from "three";

/** Punto d'ingresso nella pila: la colonna dei token di input (registro "tokens", x ≈ -5,15, riga 0). */
export const STACK = {
  tokensX: -6.6, // bordo sinistro della colonna dei cubi dei token
  inputY: -37.5, // centro della riga 0 (pila centrata: root.position.y = -37,5)
  outputLeftX: -11.5, // poco a sinistra della riga di uscita (ln_f → lm_head → probabilità)
  outputY: 41,
  outputTopX: -4,
  outputTopY: 45.5,
  rightX: 23, // bordo destro della pila (blocchi larghi ~44 unità, centrati in x = 0)
};

/** Slot delle fasi: centro (x, y) e dimensioni (w, h). */
export const SLOTS = {
  react: { x: -108, y: 40, w: 20, h: 12 },
  guard: { x: -108, y: 22, w: 20, h: 12 },
  embed: { x: -108, y: 4, w: 20, h: 12 },
  vectordb: { x: -108, y: -14, w: 20, h: 12 },
  chunks: { x: -108, y: -33, w: 20, h: 14 },
  prompt: { x: -72, y: -8, w: 26, h: 34 },
  tokenize: { x: -72, y: -38, w: 26, h: 8 },
  llm: { x: -38, y: -10, w: 16, h: 16 },
  toolcall: { x: -38, y: 41, w: 18, h: 10 },
  java: { x: -72, y: 41, w: 20, h: 10 },
  toolresult: { x: -72, y: 22, w: 20, h: 9 },
  stream: { x: -72, y: 58, w: 26, h: 11 },
};

const V = (x, y) => new THREE.Vector3(x, y, 0.35);
const edge = (id, side, t = 0.5) => {
  const s = SLOTS[id];
  if (side === "top") return V(s.x - s.w / 2 + s.w * t, s.y + s.h / 2);
  if (side === "bottom") return V(s.x - s.w / 2 + s.w * t, s.y - s.h / 2);
  if (side === "left") return V(s.x - s.w / 2, s.y + s.h / 2 - s.h * t);
  return V(s.x + s.w / 2, s.y + s.h / 2 - s.h * t);
};

/**
 * Collegamenti tra le fasi come polilinee (punti nel mondo). L'ordine segue il flusso dei dati;
 * `id` è usato dall'azione animata per far correre il pacchetto sul tratto giusto.
 */
export function connections() {
  const s = SLOTS;
  const chunkRight = edge("chunks", "right");
  const promptLeft = edge("prompt", "left", 0.78);
  const elbowX = (chunkRight.x + promptLeft.x) / 2;
  return [
    { id: "react-guard", points: [edge("react", "bottom"), edge("guard", "top")] },
    { id: "guard-embed", points: [edge("guard", "bottom"), edge("embed", "top")] },
    { id: "embed-vectordb", points: [edge("embed", "bottom"), edge("vectordb", "top")] },
    { id: "vectordb-chunks", points: [edge("vectordb", "bottom"), edge("chunks", "top")] },
    { id: "chunks-prompt", points: [chunkRight, V(elbowX, chunkRight.y), V(elbowX, promptLeft.y), promptLeft] },
    { id: "prompt-tokenize", points: [edge("prompt", "bottom"), edge("tokenize", "top")] },
    { id: "tokenize-llm", points: [edge("tokenize", "right"), V(STACK.tokensX, STACK.inputY)] },
    { id: "llm-note", points: [edge("llm", "right"), V(-21.2, SLOTS.llm.y)], decorative: true },
    { id: "llm-toolcall", points: [V(STACK.outputLeftX, STACK.outputY), edge("toolcall", "right", 0.5)] },
    { id: "toolcall-java", points: [edge("toolcall", "left"), edge("java", "right")] },
    { id: "java-toolresult", points: [edge("java", "bottom"), edge("toolresult", "top")] },
    { id: "toolresult-prompt", points: [edge("toolresult", "bottom"), edge("prompt", "top")] },
    {
      id: "llm-stream",
      points: [V(STACK.outputTopX, STACK.outputTopY), V(STACK.outputTopX, s.stream.y), edge("stream", "right")],
    },
    { id: "stream-react", points: [edge("stream", "left"), V(s.react.x, s.stream.y), edge("react", "top")] },
  ];
}

/** Box3 di uno o più slot (per inquadrarli con la camera). */
export function slotsBox(ids, pad = 1) {
  const box = new THREE.Box3();
  for (const id of ids) {
    const s = SLOTS[id];
    box.expandByPoint(new THREE.Vector3(s.x - s.w / 2 - pad, s.y - s.h / 2 - pad, -0.5));
    box.expandByPoint(new THREE.Vector3(s.x + s.w / 2 + pad, s.y + s.h / 2 + pad, 0.5));
  }
  return box;
}

/** Area del diagramma (tutte le fasi) e area di diagramma + pila (vista d'insieme). */
export const DIAGRAM_IDS = Object.keys(SLOTS);
export function overviewBox() {
  const box = slotsBox(DIAGRAM_IDS, 2);
  box.expandByPoint(new THREE.Vector3(STACK.rightX, -50, 0));
  box.expandByPoint(new THREE.Vector3(STACK.rightX, 50, 0));
  return box;
}

/** Soglie del LOD in questa vista (stage di riferimento, vedi index.js): il diagramma con la pila è largo ~145 unità. */
export const CONTEXT_DISTANCES = { detail: 72, blocks: 195 };
