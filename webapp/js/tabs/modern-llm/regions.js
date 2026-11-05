/**
 * Mappa della scena: 16 "regioni" su una griglia 4 × 4, una per meccanismo.
 * Lo stesso posto ospita i tre livelli del LOD:
 *   Dettaglio  → la regione con i componenti interni (file region*.js)
 *   Blocchi    → un rettangolo con titolo e descrizione (overview.js, testo `block`)
 *   Panoramica → il diagramma Dati → Pre-training → Post-training → Modello → Inferenza
 * Le righe seguono le categorie delle azioni.
 */
import * as THREE from "three";
import { frame, PALETTE, text } from "./shapes.js";

export const REGION_W = 36;
export const REGION_H = 26;
const STEP_X = 40;
const STEP_Y = 30;

export const ROWS = [
  { title: "Architettura", color: PALETTE.modern },
  { title: "Inferenza efficiente", color: "#22d3ee" },
  { title: "Addestramento e ragionamento", color: PALETTE.ok },
  { title: "Agenti e oltre il testo", color: PALETTE.data },
];

/**
 * id, posizione nella griglia, titolo, colore e testo del livello "Blocchi".
 * I testi dei blocchi sono brevi: si leggono da lontano.
 */
export const REGIONS = [
  { id: "compare", row: 0, col: 0, title: "Il nostro GPT e un blocco moderno", color: PALETTE.modern, sub: "stessa ossatura", block: "LayerNorm → RMSNorm, posizioni apprese → RoPE, attention → GQA con KV cache, MLP ReLU → SwiGLU o MoE." },
  { id: "moe", row: 0, col: 1, title: "Mixture of Experts", color: PALETTE.moe, sub: "router + esperti", block: "Un router sceglie pochi esperti (MLP) per ogni token: tanti parametri in totale, pochi usati alla volta." },
  { id: "gqa", row: 0, col: 2, title: "MHA, GQA e MQA", color: "#60a5fa", sub: "K e V condivisi", block: "Più head di query condividono la stessa coppia K/V: la KV cache diventa molte volte più piccola." },
  { id: "rope", row: 0, col: 3, title: "RoPE", color: "#c084fc", sub: "posizione = rotazione", block: "Query e key vengono ruotate di un angolo proporzionale alla posizione: conta la distanza tra i token." },
  { id: "bpe", row: 1, col: 0, title: "Tokenizer BPE a byte", color: "#fbbf24", sub: "byte → sotto-parole", block: "Si parte dai 256 byte e si fondono le coppie più frequenti: nessun carattere è mai sconosciuto." },
  { id: "kv", row: 1, col: 1, title: "KV cache", color: "#22d3ee", sub: "solo il token nuovo", block: "K e V dei token già visti si salvano: a ogni passo si calcolano solo quelli del token nuovo." },
  { id: "flash", row: 1, col: 2, title: "Attention efficiente", color: "#2dd4bf", sub: "FlashAttention", block: "Calcolo a blocchi nella memoria veloce della GPU; finestra scorrevole; contesti fino a 128K token." },
  { id: "spec", row: 1, col: 3, title: "Decodifica speculativa", color: "#f472b6", sub: "bozza + verifica", block: "Un modello piccolo propone alcuni token, quello grande li verifica tutti in un solo passaggio." },
  { id: "scaling", row: 2, col: 0, title: "Leggi di scala", color: "#fb923c", sub: "Chinchilla", block: "Con un budget di calcolo fissato conviene crescere insieme parametri e dati: la loss scende in modo prevedibile." },
  { id: "training", row: 2, col: 1, title: "Pipeline di addestramento", color: PALETTE.ok, sub: "tre fasi", block: "Prima si impara a continuare il testo, poi a seguire istruzioni, poi le preferenze (RLHF, DPO, CAI)." },
  { id: "reasoning", row: 2, col: 2, title: "Ragionamento", color: "#a3e635", sub: "pensiero e verifica", block: "Passi intermedi, più tentativi e un verificatore; RL con ricompense verificabili (GRPO, DeepSeek-R1)." },
  { id: "interp", row: 2, col: 3, title: "Interpretabilità", color: "#e879f9", sub: "head e feature", block: "Si apre la scatola: head che copiano schemi, autoencoder sparsi che trovano concetti leggibili." },
  { id: "agent", row: 3, col: 0, title: "Tool use e agenti", color: PALETTE.tool, sub: "chiamate a strumenti", block: "Il modello scrive una chiamata strutturata, il programma esegue lo strumento e restituisce il risultato." },
  { id: "coding", row: 3, col: 1, title: "Capacità di coding", color: "#93c5fd", sub: "FIM, test, SWE-bench", block: "Dati di codice, riempimento nel mezzo (FIM), ciclo scrivi → esegui → correggi con il feedback dei test." },
  { id: "multi", row: 3, col: 2, title: "Multimodale", color: "#fda4af", sub: "immagini come token", block: "Un encoder trasforma l'immagine in vettori, un proiettore li rende token che il modello legge." },
  { id: "mesh", row: 3, col: 3, title: "Modelli 3D", color: "#fcd34d", sub: "mesh come testo", block: "Una mesh scritta come testo OBJ (LLaMA-Mesh) o uno script per un programma 3D come Blender." },
];

export const REGION_BY_ID = new Map(REGIONS.map((r) => [r.id, r]));

/** Centro della regione in coordinate di scena (griglia centrata nell'origine). */
export function regionCenter(region) {
  return new THREE.Vector3((region.col - 1.5) * STEP_X, (1.5 - region.row) * STEP_Y, 0);
}

/** Box3 della regione (per inquadrarla con il CameraDirector). */
export function regionBox(region, pad = 0) {
  const c = regionCenter(region);
  return new THREE.Box3(new THREE.Vector3(c.x - REGION_W / 2 - pad, c.y - REGION_H / 2 - pad, -0.5), new THREE.Vector3(c.x + REGION_W / 2 + pad, c.y + REGION_H / 2 + pad, 0.5));
}

/** Riquadro di tutta la griglia. */
export function gridBox() {
  return new THREE.Box3(new THREE.Vector3(-2 * STEP_X, -2 * STEP_Y, -1), new THREE.Vector3(2 * STEP_X, 2 * STEP_Y + 9, 1));
}

/** Cornice e titolo comuni a tutte le regioni del livello Dettaglio. */
export function regionShell(group, region) {
  frame(group, REGION_W, REGION_H, "#334155");
  text(group, region.title, 0, REGION_H / 2 - 1.3, { size: 0.95, color: region.color });
}
