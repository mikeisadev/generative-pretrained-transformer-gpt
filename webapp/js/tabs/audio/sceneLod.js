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
  { slot: "audio", title: "Audio", subtitle: "16.000 campioni/s", text: "La forma d'onda: continua e lunghissima.", color: "#67e8f9" },
  { slot: "encoder", title: "Encoder", subtitle: "un vettore ogni 20 ms", text: "Riassume ogni frame in un vettore di numeri.", color: "#fbbf24" },
  { slot: "rvq", title: "RVQ", subtitle: "4 codebook × 64 voci", text: "Ogni vettore diventa 4 numeri interi: i token.", color: "#f472b6" },
  { slot: "transformer", title: "Transformer", subtitle: "prossimo token + delay pattern", text: "Come il GPT, ma i token sono pezzi di suono.", color: "#38bdf8", spanTo: "delay" },
  // il componente è stretto: il blocco si allarga di 1,5 unità per lato, così il titolo ci sta
  { slot: "decoder", title: "Decoder", subtitle: "token → suono", text: "Dai token alla forma d'onda.", color: "#34d399", widen: 1.5 },
  { slot: "output", title: "Audio generato", subtitle: "da ascoltare", text: "Il brano nuovo, token dopo token.", color: "#34d399" },
  { slot: "text", title: "Prompt", subtitle: "\"melodia allegra…\"", text: "Che musica vuoi sentire.", color: "#a78bfa" },
  { slot: "t5", title: "Text encoder", subtitle: "T5", text: "Il testo diventa vettori per la cross-attention.", color: "#a78bfa" },
  { slot: "wMel", title: "log-Mel", subtitle: "30 s · 3000 × 80", text: "L'ingresso di Whisper: niente token audio.", color: "#fbbf24" },
  { slot: "wEnc", title: "Encoder", subtitle: "Whisper", text: "Legge tutto l'audio insieme.", color: "#38bdf8" },
  { slot: "wDec", title: "Decoder", subtitle: "Whisper", text: "Scrive il testo un token alla volta.", color: "#a78bfa" },
  { slot: "wText", title: "Testo", subtitle: "token speciali + parole", text: "Trascrizione, traduzione o lingua.", color: "#34d399" },
];

/**
 * makeBlock disegna il testo con una dimensione fissa in unità di scena: per renderlo leggibile
 * da lontano si costruisce il blocco in piccolo e lo si ingrandisce (testo compreso).
 */
function scaledBlock({ width, height, ...rest }, scale) {
  const block = makeBlock({ ...rest, width: width / scale, height: height / scale, depth: 0.08 });
  block.scale.setScalar(scale);
  return block;
}

const FLOW_MAIN = ["audio", "encoder", "rvq", "transformer", "delay", "decoder", "output"];

const right = (s) => new THREE.Vector3(SLOT[s].x + SLOT[s].w / 2 + 0.3, SLOT[s].y, 0);
const left = (s) => new THREE.Vector3(SLOT[s].x - SLOT[s].w / 2 - 0.3, SLOT[s].y, 0);

/** Livello "Blocchi": restituisce { group, bySlot } (bySlot: id → blocco, per inquadrarlo). */
export function buildBlocks() {
  const group = new THREE.Group();
  const bySlot = new Map();
  const extent = new Map(); // slot → [x0, x1] del blocco disegnato (le frecce partono dai bordi veri)
  for (const b of BLOCKS) {
    const s = SLOT[b.slot];
    const widen = b.widen ?? 0;
    const x0 = s.x - s.w / 2 - widen;
    const x1 = (b.spanTo ? SLOT[b.spanTo].x + SLOT[b.spanTo].w / 2 : s.x + s.w / 2) + widen;
    extent.set(b.slot, [x0, x1]);
    if (b.spanTo) extent.set(b.spanTo, [x0, x1]);
    const width = x1 - x0;
    const height = b.slot === "text" || b.slot === "t5" ? 5 : b.slot.startsWith("w") ? 6 : 8;
    const block = scaledBlock({ title: b.title, subtitle: b.subtitle, text: b.text, width, height, color: b.color }, 4.2);
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
  const edge = (slot, side) => new THREE.Vector3(extent.get(slot)[side] + (side ? 0.3 : -0.3), SLOT[slot].y, 0);
  for (const [a, b] of arrows) group.add(makeArrow(edge(a, 1), edge(b, 0), { color: 0x94a3b8, radius: 0.12, headLength: 0.8 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.t5.x, SLOT.t5.y - 2.6, 0), new THREE.Vector3(SLOT.t5.x, SLOT.transformer.y + 4.2, 0), { color: 0xa78bfa, radius: 0.12, headLength: 0.8 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.audio.x, SLOT.audio.y - 4.2, 0), new THREE.Vector3(SLOT.wMel.x - SLOT.wMel.w / 2 + 2, SLOT.wMel.y + 3.2, 0), { color: 0xfbbf24, radius: 0.12, headLength: 0.8 }));
  return { group, bySlot };
}

/** Livello "Panoramica": diagramma con pochi blocchi grandi. */
export function buildMacro() {
  const group = new THREE.Group();
  const big = (title, subtitle, text, x0, x1, y, color, height = 13) => {
    const b = scaledBlock({ title, subtitle, text, width: x1 - x0, height, color }, 6.5);
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
  big("Suono", "forma d'onda → vettori", "Un'onda continua, riassunta in un vettore ogni 20 ms.", sx0, sx1, 0, "#67e8f9");
  big("Token", "numeri interi · prossimo token", "Il codec fa i token; il transformer ne genera di nuovi, come il GPT con le lettere.", tx0, tx1, 0, "#38bdf8");
  big("Suono", "token → forma d'onda", "Il decoder rende i token suono da ascoltare.", ox0, ox1, 0, "#34d399");
  big("Testo", "il prompt", "Descrive la musica da generare.", SLOT.text.x - 8, SLOT.t5.x + 8, 17, "#a78bfa", 8.5);
  big("Voce → Testo (Whisper)", "il percorso opposto", "Dall'audio alle parole.", SLOT.wMel.x - SLOT.wMel.w / 2, SLOT.wText.x + SLOT.wText.w / 2, -19, "#fbbf24", 9);
  const y = 0;
  group.add(makeArrow(new THREE.Vector3(sx1 + 0.4, y, 0), new THREE.Vector3(tx0 - 0.4, y, 0), { color: 0xe2e8f0, radius: 0.16, headLength: 1 }));
  group.add(makeArrow(new THREE.Vector3(tx1 + 0.4, y, 0), new THREE.Vector3(ox0 - 0.4, y, 0), { color: 0xe2e8f0, radius: 0.16, headLength: 1 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.t5.x - 2, 12.5, 0), new THREE.Vector3(SLOT.t5.x - 2, 6.9, 0), { color: 0xa78bfa, radius: 0.18, headLength: 1.1 }));
  group.add(makeArrow(new THREE.Vector3(SLOT.audio.x, -6.9, 0), new THREE.Vector3(SLOT.wMel.x - SLOT.wMel.w / 2 + 1, -14.2, 0), { color: 0xfbbf24, radius: 0.18, headLength: 1.1 }));
  return group;
}

/** Tubi animati del livello "Dettaglio" lungo il percorso principale e verso Whisper. */
export function buildFlows() {
  const group = new THREE.Group();
  const flows = [];
  for (let i = 1; i < FLOW_MAIN.length; i++) {
    const pa = right(FLOW_MAIN[i - 1]);
    const pb = left(FLOW_MAIN[i]);
    // dal bordo destro di un componente a poco prima del successivo (lascia spazio alle etichette degli assi)
    flows.push(makeFlow([new THREE.Vector3(pa.x + 0.1, 0, 0), new THREE.Vector3(pb.x - 1.2, 0, 0)], { radius: 0.06 }));
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
