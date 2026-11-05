/**
 * Livelli astratti dell'architettura per lo zoom semantico (vedi ../lod.js).
 *
 *   "Blocchi":    un rettangolo per componente, nello stesso slot della scena di dettaglio
 *                 (architecture.js → arch.slots), con titolo, forma/numero di pesi e una frase.
 *                 Ogni blocco transformer è un contenitore con i suoi sotto-rettangoli.
 *   "Panoramica": tre rettangoli grandi (Input, Transformer, Output) e le frecce del flusso.
 *
 * I rettangoli stanno esattamente dove passa il binario del flusso (stesse righe, stessi slot):
 * il pacchetto animato attraversa il rettangolo del componente che sta "visitando".
 * `rectFor(level, stationId)` restituisce il rettangolo da evidenziare per una stazione.
 */
import * as THREE from "three";
import { makeArrow } from "../../../core/diagram.js";
import { fmt } from "../../../core/dom.js";
import { residualArc } from "../architecture.js";
import { fitBlock } from "./lodBlocks.js";
import { componentTotals, countParams, shapeText } from "./paramGroups.js";

/** Colori per tipo di componente (stessi della legenda nella sidebar). */
export const KIND_COLORS = {
  data: "#94a3b8", // dati e operazioni senza pesi
  embedding: "#f59e0b",
  norm: "#a78bfa",
  attention: "#38bdf8",
  ffwd: "#34d399",
  output: "#f472b6",
};

const GAP = 0.9; // come in architecture.js: i rettangoli occupano slot + metà spazio per lato
const ROW_CONTENT = 13.4; // altezza utile di una riga (le righe distano 15)

const pesi = (n) => (n === 0 ? "0 pesi" : `${fmt(n)} pesi`);

function place(group, block, x, y, z = 0) {
  block.position.set(x, y, z);
  group.add(block);
  return block;
}

/** Rettangolo nel livello Blocchi centrato su uno slot. */
function slotBlock(group, slot, o, { y = 0, height = 8, shrink = 0.3, z = 0 } = {}) {
  const width = slot.width + GAP - shrink;
  const block = fitBlock({ width, height, maxScale: 2.2, ...o });
  return place(group, block, slot.x, slot.y + y, z);
}

const rect = (x, y, w, h) => ({ x, y, w, h });
const rectOf = (block) => {
  const { width, height } = block.userData.block;
  const s = block.userData.scale;
  return rect(block.position.x, block.position.y, width * s, height * s);
};
const union = (a, b) => {
  const x0 = Math.min(a.x - a.w / 2, b.x - b.w / 2);
  const x1 = Math.max(a.x + a.w / 2, b.x + b.w / 2);
  const y0 = Math.min(a.y - a.h / 2, b.y - b.h / 2);
  const y1 = Math.max(a.y + a.h / 2, b.y + b.h / 2);
  return rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0);
};

function buildBlocksLevel(tab) {
  const { spec, arch } = tab;
  const cfg = spec.config;
  const C = cfg.n_embd;
  const totals = componentTotals(spec.params);
  const count = (id) => totals.get(id)?.count ?? 0;
  const shapeOf = (name) => shapeText(spec.params.find((p) => p.name === name).shape);
  const slots = arch.slots;
  const group = new THREE.Group();
  group.name = "arch-blocks";
  const rects = new Map(); // stationId → rettangolo da evidenziare

  // --- riga 0: input ed embedding
  const T = spec.example.prompt.length;
  rects.set("tokens", rectOf(slotBlock(group, slots.get("tokens"), {
    title: "Token", subtitle: `${T} caratteri · 0 pesi`, color: KIND_COLORS.data,
    text: "Il prompt diventa una sequenza di numeri interi (token id), uno per carattere.",
  })));
  const emb = slots.get("embeddings");
  const tok = slotBlock(group, emb, {
    title: "Token embedding", subtitle: `${shapeOf("token_embedding_table.weight")} · ${pesi(count("tok_emb"))}`, color: KIND_COLORS.embedding,
    text: `Ogni token id sceglie una riga: il suo vettore di ${cfg.n_embd} numeri.`,
  }, { y: 2.25, height: 4 });
  const pos = slotBlock(group, emb, {
    title: "Position embedding", subtitle: `${shapeOf("position_embedding_table.weight")} · ${pesi(count("pos_emb"))}`, color: KIND_COLORS.embedding,
    text: `Un vettore per ogni posizione 0…${cfg.block_size - 1}: dice dove sta il token.`,
  }, { y: -2.15, height: 3.8 });
  rects.set("embeddings", union(rectOf(tok), rectOf(pos)));
  rects.set("embed_add", rectOf(slotBlock(group, slots.get("embed_add"), {
    title: "+", subtitle: "somma", color: KIND_COLORS.data, text: "x = token + posizione",
  }, { height: 4 })));

  // --- righe dei blocchi transformer: contenitore + sotto-rettangoli + archi residui sotto
  for (let l = 0; l < cfg.n_layer; l++) {
    const p = `blocks.${l}`;
    const first = slots.get(`${p}.ln1`);
    const last = slots.get(`${p}.add2`);
    const x0 = first.x - first.width / 2 - GAP / 2;
    const x1 = last.x + last.width / 2 + GAP / 2;
    const y = first.y;
    const container = fitBlock({
      width: x1 - x0 + 0.6, height: ROW_CONTENT, maxScale: 3.6, depth: 0.1, bodyOpacity: 0.55,
      title: `Blocco ${l}  (blocks.${l})`,
      subtitle: `${fmt(countParams(spec.params, `${p}.`))} pesi · x = x + sa(ln1(x));  x = x + ffwd(ln2(x))`,
      color: "#7dd3fc", fill: "rgba(15,23,42,0.55)",
    });
    place(group, container, (x0 + x1) / 2, y, -0.35);
    // Spazio occupato da titolo e sottotitolo del contenitore (stesse proporzioni di makeBlock).
    const header = (0.14 + 0.42 * 1.15 + 0.26 * 1.4 + 0.12) * container.userData.scale;
    const top = y + ROW_CONTENT / 2 - header; // sotto il titolo del contenitore
    const bottom = y - ROW_CONTENT / 2 + 1.6; // sopra gli archi residui
    const h = top - bottom;
    const cy = (top + bottom) / 2 - y;
    const sub = (id, o) => rects.set(id, rectOf(slotBlock(group, slots.get(id), o, { y: cy, height: h })));
    sub(`${p}.ln1`, { title: "LayerNorm", subtitle: `ln1 · ${count(`${p}.ln1`)}`, color: KIND_COLORS.norm, text: "Riporta ogni vettore a media 0 e varianza 1, poi applica γ e β." });
    sub(`${p}.heads`, { title: "Attention", subtitle: `${cfg.n_head} head · ${fmt(count(`${p}.heads`))}`, color: KIND_COLORS.attention, text: "Ogni token guarda i token precedenti e raccoglie da loro informazioni (Q, K, V)." });
    sub(`${p}.proj`, { title: "Proiezione", subtitle: `sa.proj · ${fmt(count(`${p}.proj`))}`, color: KIND_COLORS.attention, text: `Rimescola le uscite delle head in un unico vettore da ${cfg.n_embd}.` });
    sub(`${p}.add1`, { title: "+", subtitle: "residuo", color: KIND_COLORS.data, text: "x + attention" });
    sub(`${p}.ln2`, { title: "LayerNorm", subtitle: `ln2 · ${count(`${p}.ln2`)}`, color: KIND_COLORS.norm, text: "Normalizza di nuovo, prima della feed-forward." });
    sub(`${p}.ffwd`, { title: "Feed-forward", subtitle: `${C} → ${4 * C} → ${C} · ${fmt(count(`${p}.ffwd`))}`, color: KIND_COLORS.ffwd, text: `Elabora ogni token da solo: espande a ${4 * C} numeri, applica la ReLU, comprime di nuovo a ${C}.` });
    sub(`${p}.add2`, { title: "+", subtitle: "residuo", color: KIND_COLORS.data, text: "x + feed-forward" });

    // Connessioni residue: archi sotto i rettangoli, dall'ingresso del blocco ai due "+".
    const arcs = new THREE.Group();
    arcs.position.set(0, bottom - 0.15, 0.3);
    const arcStart = first.x - first.width / 2;
    arcs.add(
      residualArc(arcStart, slots.get(`${p}.add1`).x, -1.0, { radius: 0.07, opacity: 0.8, z: 0 }),
      residualArc(slots.get(`${p}.add1`).x, last.x, -1.0, { radius: 0.07, opacity: 0.8, z: 0 }),
    );
    group.add(arcs);
  }

  // --- riga di uscita
  const lmShape = shapeOf("lm_head.weight");
  rects.set("ln_f", rectOf(slotBlock(group, slots.get("ln_f"), {
    title: "LayerNorm", subtitle: `ln_f · ${count("ln_f")}`, color: KIND_COLORS.norm, text: "Normalizzazione finale del vettore di ogni posizione.",
  })));
  rects.set("lm_head", rectOf(slotBlock(group, slots.get("lm_head"), {
    title: "lm_head", subtitle: `${lmShape} + bias · ${fmt(count("lm_head"))}`, color: KIND_COLORS.output,
    text: `Trasforma il vettore in ${cfg.vocab_size} punteggi (logits), uno per carattere.`,
  })));
  rects.set("logits", rectOf(slotBlock(group, slots.get("logits"), {
    title: "Softmax", subtitle: `${cfg.vocab_size} probabilità · 0 pesi`, color: KIND_COLORS.data,
    text: "I punteggi diventano probabilità positive che sommano a 1.",
  })));
  const probs = spec.example.next_token_probs;
  const best = probs.indexOf(Math.max(...probs));
  const ch = spec.vocab[best] === " " ? "␣" : spec.vocab[best] === "\n" ? "↵" : spec.vocab[best];
  rects.set("next_token", rectOf(slotBlock(group, slots.get("next_token"), {
    title: "Prossimo token", subtitle: `'${ch}' ${(probs[best] * 100).toFixed(1).replace(".", ",")}%`, color: KIND_COLORS.data,
    text: "Si sceglie un carattere e lo si aggiunge al contesto.",
  })));
  return { group, rects };
}

function buildMacroLevel(tab) {
  const { spec, arch } = tab;
  const cfg = spec.config;
  const group = new THREE.Group();
  group.name = "arch-macro";
  const rowY = (row) => row * arch.rowSpacing;
  const blockRows = arch.rows.filter((r) => r.index >= 1 && r.index <= cfg.n_layer);
  const width = Math.max(...blockRows.map((r) => r.width)) + 3;
  const total = spec.total_params;
  const inBlocks = countParams(spec.params, "blocks.");
  const inInput = countParams(spec.params, (n) => n.includes("embedding_table."));
  const inOutput = total - inBlocks - inInput;
  const pct = (n) => `${((n / total) * 100).toFixed(1).replace(".", ",")}%`;

  // Input e Output sono alti (16 unità) perché il titolo di makeBlock non supera il 20% dell'altezza:
  // servono scritte grandi, la Panoramica si guarda da 150-250 unità di distanza.
  const ioHeight = 16;
  const ioShift = ioHeight / 2 - 4; // il binario (centro della riga) attraversa comunque il rettangolo
  const input = fitBlock({
    width: width * 0.8, height: ioHeight, maxScale: 7, color: KIND_COLORS.embedding,
    title: "Input: i token",
    subtitle: `fino a ${cfg.block_size} token · ${fmt(inInput)} pesi`,
    text: "Ogni carattere diventa un vettore: embedding del token + embedding della posizione.",
  });
  place(group, input, 0, rowY(0) - ioShift);

  const yBottom = rowY(1) - ROW_CONTENT / 2;
  const yTop = rowY(cfg.n_layer) + ROW_CONTENT / 2;
  const transformer = fitBlock({
    width, height: yTop - yBottom, maxScale: 10, color: "#7dd3fc", depth: 0.3,
    title: "Transformer",
    subtitle: `${cfg.n_layer} blocchi uguali`, // il sottotitolo non va a capo: corto, o rimpicciolirebbe tutto il testo
    text: `${fmt(inBlocks)} parametri (${pct(inBlocks)} del modello). Lo stesso blocco (attention + feed-forward) ripetuto ${cfg.n_layer} volte, dal basso verso l'alto. L'attention fa comunicare i token tra loro, la feed-forward elabora ogni token da solo, le connessioni residue sommano ogni risultato al vettore di partenza.`,
  });
  place(group, transformer, 0, (yBottom + yTop) / 2);

  const output = fitBlock({
    width: width * 0.8, height: ioHeight, maxScale: 7, color: KIND_COLORS.output,
    title: "Output: il prossimo token",
    subtitle: `${cfg.vocab_size} probabilità · ${fmt(inOutput)} pesi`,
    text: `LayerNorm, lm_head e softmax: una probabilità per ognuno dei ${cfg.vocab_size} caratteri.`,
  });
  place(group, output, 0, rowY(cfg.n_layer + 1) + ioShift);

  // Frecce del flusso tra i tre rettangoli (dal basso verso l'alto, come nel paper).
  const arrowX = 0;
  const inTop = input.position.y + ioHeight / 2;
  const outBottom = output.position.y - ioHeight / 2;
  group.add(
    makeArrow(new THREE.Vector3(arrowX, inTop + 0.1, 0.4), new THREE.Vector3(arrowX, yBottom - 0.1, 0.4), { color: 0x67e8f9, radius: 0.3, headLength: 1.4 }),
    makeArrow(new THREE.Vector3(arrowX, yTop + 0.1, 0.4), new THREE.Vector3(arrowX, outBottom - 0.1, 0.4), { color: 0x67e8f9, radius: 0.3, headLength: 1.4 }),
  );

  const rects = new Map();
  const inputRect = rectOf(input);
  const outputRect = rectOf(output);
  const transformerRect = rectOf(transformer);
  for (const station of arch.stations) {
    if (station.row === 0) rects.set(station.id, inputRect);
    else if (station.row > cfg.n_layer) rects.set(station.id, outputRect);
    else rects.set(station.id, transformerRect); // un solo rettangolo: la fascia di un blocco taglierebbe il testo
  }
  const info = { width, yBottom: input.position.y - ioHeight / 2, yTop: output.position.y + ioHeight / 2, transformer: { y0: yBottom, y1: yTop }, input: rectOf(input), output: rectOf(output), counts: { input: inInput, blocks: inBlocks, output: inOutput } };
  return { group, rects, info };
}

/**
 * Costruisce i due livelli astratti dell'architettura, in coordinate di arch.root
 * (il chiamante li posiziona con lo stesso offset verticale della pila).
 */
export function buildArchitectureLevels(tab) {
  const blocks = buildBlocksLevel(tab);
  const macro = buildMacroLevel(tab);
  return { blocks: blocks.group, macro: macro.group, rects: { blocks: blocks.rects, macro: macro.rects }, macroInfo: macro.info };
}
