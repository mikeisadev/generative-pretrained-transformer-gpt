/**
 * Livelli astratti del LOD (zoom semantico):
 *   "Blocchi"    → un rettangolo per componente, con titolo e una frase su cosa fa
 *   "Panoramica" → pochi rettangoli grandi: Suono → Token → Suono (+ testo e Whisper)
 * e, nel livello "Dettaglio", i tubi animati che collegano i componenti.
 * I blocchi stanno nelle stesse posizioni dei componenti (layout3d.js).
 */
import * as THREE from "three";
import { makeArrow, makeBlock, makeFlow } from "../../core/diagram.js";
import { SLOT } from "./layout3d.js";

const BLOCKS = [
  { slot: "audio", title: "Audio", subtitle: "16.000 campioni/s", text: "La forma d'onda: un numero ogni 1/16.000 di secondo. Continua e lunghissima.", color: "#67e8f9" },
  { slot: "encoder", title: "Encoder", subtitle: "un vettore ogni 20 ms", text: "Riassume ogni frame in un vettore. Nei codec veri è una rete convoluzionale; qui spettro + logaritmo.", color: "#fbbf24" },
  { slot: "rvq", title: "RVQ", subtitle: "4 codebook da 64 voci", text: "Ogni vettore diventa 4 numeri interi: i token. Ogni codebook corregge l'errore dei precedenti.", color: "#f472b6" },
  { slot: "transformer", title: "Transformer", subtitle: "prossimo token", text: "Come il GPT della lezione, ma i token sono pezzi di suono. Il delay pattern sfasa i 4 codebook.", color: "#38bdf8", spanTo: "delay" },
  { slot: "decoder", title: "Decoder", subtitle: "token → suono", text: "Dai token ricostruisce la forma d'onda (nei codec veri: convoluzioni che aumentano la risoluzione).", color: "#34d399" },
  { slot: "output", title: "Audio generato", subtitle: "ascoltalo", text: "Il suono nuovo, prodotto token dopo token e decodificato.", color: "#34d399" },
  { slot: "text", title: "Prompt testuale", subtitle: "\"melodia allegra…\"", text: "La descrizione di ciò che vuoi sentire.", color: "#a78bfa" },
  { slot: "t5", title: "Text encoder (T5)", subtitle: "cross-attention", text: "Trasforma il testo in vettori che il transformer consulta a ogni passo.", color: "#a78bfa" },
  { slot: "wMel", title: "log-Mel (Whisper)", subtitle: "30 s · 3000 × 80", text: "Whisper non usa token audio: legge direttamente il log-Mel.", color: "#fbbf24" },
  { slot: "wEnc", title: "Encoder Whisper", subtitle: "Conv1d + transformer", text: "Capisce l'audio guardandolo tutto insieme (attention bidirezionale).", color: "#38bdf8" },
  { slot: "wDec", title: "Decoder Whisper", subtitle: "prossimo token di testo", text: "Scrive il testo un token alla volta consultando l'encoder.", color: "#a78bfa" },
  { slot: "wText", title: "Testo", subtitle: "token speciali + parole", text: "Trascrizione, traduzione o lingua: lo decidono i token speciali iniziali.", color: "#34d399" },
];

const FLOW_MAIN = ["audio", "encoder", "rvq", "transformer", "delay", "decoder", "output"];

const right = (s) => new THREE.Vector3(SLOT[s].x + SLOT[s].w / 2 + 0.3, SLOT[s].y, 0);
const left = (s) => new THREE.Vector3(SLOT[s].x - SLOT[s].w / 2 - 0.3, SLOT[s].y, 0);

/** Livello "Blocchi": restituisce { group, bySlot } (bySlot: id → blocco, per inquadrarlo). */
export function buildBlocks() {
  const group = new THREE.Group();
  const bySlot = new Map();
  for (const b of BLOCKS) {
    const s = SLOT[b.slot];
    const x0 = s.x - s.w / 2;
    const x1 = b.spanTo ? SLOT[b.spanTo].x + SLOT[b.spanTo].w / 2 : s.x + s.w / 2;
    const width = x1 - x0;
    const height = b.slot === "text" || b.slot === "t5" || b.slot === "wText" ? 4 : 5.6;
    const block = makeBlock({ title: b.title, subtitle: b.subtitle, text: b.text, width, height, color: b.color });
    block.position.set((x0 + x1) / 2, s.y, 0);
    group.add(block);
    bySlot.set(b.slot, block);
  }
  const arrows = [
    ["audio", "encoder"],
    ["encoder", "rvq"],
    ["rvq", "transformer"],
    ["delay", "decoder"],
    ["decoder", "output"],
    ["text", "t5"],
    ["wMel", "wEnc"],
    ["wEnc", "wDec"],
    ["wDec", "wText"],
  ];
  for (const [a, b] of arrows) group.add(makeArrow(right(a), left(b), { color: 0x94a3b8, radius: 0.09, headLength: 0.6 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.t5.x, SLOT.t5.y - 2.1, 0), new THREE.Vector3(SLOT.t5.x, SLOT.transformer.y + 3, 0), { color: 0xa78bfa, radius: 0.09, headLength: 0.6 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.audio.x, SLOT.audio.y - 3, 0), new THREE.Vector3(SLOT.wMel.x - SLOT.wMel.w / 2 + 2, SLOT.wMel.y + 2.1, 0), { color: 0xfbbf24, radius: 0.09, headLength: 0.6 }));
  return { group, bySlot };
}

/** Livello "Panoramica": diagramma con pochi blocchi grandi. */
export function buildMacro() {
  const group = new THREE.Group();
  const big = (title, subtitle, text, x0, x1, y, color, height = 9) => {
    const b = makeBlock({ title, subtitle, text, width: x1 - x0, height, color });
    b.position.set((x0 + x1) / 2, y, 0);
    group.add(b);
    return b;
  };
  const sx0 = SLOT.audio.x - SLOT.audio.w / 2;
  const sx1 = SLOT.encoder.x + SLOT.encoder.w / 2;
  const tx0 = SLOT.rvq.x - SLOT.rvq.w / 2;
  const tx1 = SLOT.delay.x + SLOT.delay.w / 2;
  const ox0 = SLOT.decoder.x - SLOT.decoder.w / 2;
  const ox1 = SLOT.output.x + SLOT.output.w / 2;
  big("Suono", "forma d'onda → vettori", "Il suono è continuo: l'encoder lo riassume in un vettore ogni pochi millisecondi.", sx0, sx1, 0, "#67e8f9");
  big("Token", "numeri interi · prossimo token", "Il codec trasforma i vettori in token; un transformer genera nuovi token, come il GPT con le lettere.", tx0, tx1, 0, "#38bdf8");
  big("Suono", "token → forma d'onda", "Il decoder del codec trasforma i token in suono da ascoltare.", ox0, ox1, 0, "#34d399");
  big("Testo", "il prompt", "Descrive la musica da generare (cross-attention).", SLOT.text.x - 6, SLOT.t5.x + 4.5, 13, "#a78bfa", 5);
  big("Voce → Testo (Whisper)", "log-Mel → encoder → decoder", "Il percorso opposto: dall'audio alle parole.", SLOT.wMel.x - SLOT.wMel.w / 2, SLOT.wText.x + SLOT.wText.w / 2, -15, "#fbbf24", 6);
  const y = 0;
  group.add(makeArrow(new THREE.Vector3(sx1 + 0.4, y, 0), new THREE.Vector3(tx0 - 0.4, y, 0), { color: 0xe2e8f0, radius: 0.16, headLength: 1 }));
  group.add(makeArrow(new THREE.Vector3(tx1 + 0.4, y, 0), new THREE.Vector3(ox0 - 0.4, y, 0), { color: 0xe2e8f0, radius: 0.16, headLength: 1 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.t5.x - 2, 10.4, 0), new THREE.Vector3(SLOT.t5.x - 2, 4.6, 0), { color: 0xa78bfa, radius: 0.16, headLength: 1 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.audio.x, -4.6, 0), new THREE.Vector3(SLOT.audio.x + 2, -11.9, 0), { color: 0xfbbf24, radius: 0.16, headLength: 1 }));
  return group;
}

/** Tubi animati del livello "Dettaglio" lungo il percorso principale e verso Whisper. */
export function buildFlows() {
  const group = new THREE.Group();
  const flows = [];
  for (let i = 1; i < FLOW_MAIN.length; i++) {
    const pa = right(FLOW_MAIN[i - 1]);
    const pb = left(FLOW_MAIN[i]);
    flows.push(makeFlow([new THREE.Vector3(pa.x - 0.3, 0, 0), new THREE.Vector3(pb.x + 0.3, 0, 0)], { radius: 0.06 }));
  }
  flows.push(makeFlow([new THREE.Vector3(SLOT.audio.x - 4, -SLOT.audio.h / 2 - 0.8, 0), new THREE.Vector3(SLOT.audio.x - 4, SLOT.wMel.y + SLOT.wMel.h / 2 + 1.9, 0), new THREE.Vector3(SLOT.wMel.x - SLOT.wMel.w / 2 + 1, SLOT.wMel.y + SLOT.wMel.h / 2 + 1.9, 0)], { radius: 0.06, color: "#fbbf24" }));
  flows.push(makeFlow([right("wMel"), left("wEnc")], { radius: 0.06 }));
  flows.push(makeFlow([right("wDec"), left("wText")], { radius: 0.06 }));
  for (const f of flows) {
    f.userData.noBounds = true;
    group.add(f);
  }
  return { group, tick: (dt) => flows.forEach((f) => f.userData.tick(dt)) };
}
