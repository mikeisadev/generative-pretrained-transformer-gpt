/**
 * Livello Blocchi della vista completa: l'architettura da sinistra a destra (tokenizer → embedding →
 * residuo → layer raggruppati "× N" → norm → head → softmax), con le lastre fantasma delle
 * ripetizioni, le parentesi dei gruppi e dei segmenti (encoder, decoder), la riga delle cache sotto
 * e i rami (visione sopra, draft multi-token sotto). Ogni blocco mostra i suoi parametri.
 *
 *   const blocks = buildBlocks(group, model, strip, layout);
 *   blocks.anchors   Map("blocks:<id>" → Box3)        blocks.blockBoxes[i]  Box3 del blocco i
 *   blocks.pickables oggetti cliccabili: userData.block = indice, .toggle = id del gruppo, .anchor
 *   blocks.highlight(indices | null), blocks.packet (sfera per le azioni), blocks.flowY
 */
import * as THREE from "three";
import { glow, packet as makePacket } from "../shapes.js";
import { arrow, boxBatch, bracket, flatBox, LineBatch, outline, rectBox, segments, textPlate, wrapText } from "./draw.js";
import { STRIP } from "./layout.js";

/**
 * Misure del testo dei blocchi (unità di scena): il titolo si legge con tutta la striscia
 * inquadrata, numero di layer e parametri avvicinandosi. Il testo lungo (cosa fa il blocco) sta nel
 * suggerimento al passaggio del mouse e nella didascalia al clic: sulla lastra solo l'essenziale.
 */
const TITLE = 7.2;
const SUB = 3.9;
const BODY = 2.9;
const PARAMS = 3.6;
/** Quota delle corsie del residuo che attraversano i layer (sotto il centro dei blocchi). */
const RAIL_Y = -15;
const RAIL_COLORS = ["#38bdf8", "#0ea5e9", "#7dd3fc", "#0284c7"];
const BACKGROUND = new THREE.Color("#0d1117");

/**
 * Lastra di un blocco: corpo scuro, barra del colore del blocco in alto, bordo, titolo e riga
 * piccola che si rimpiccioliscono da soli se non entrano (Plate), parametri in basso.
 * Restituisce il corpo (per accenderlo e per il clic).
 */
function stripBlock(group, r, { title, sub, text, params, color }, batch) {
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(r.w, r.h, 1),
    new THREE.MeshStandardMaterial({ color: "#0f172a", emissive: color, emissiveIntensity: 0, roughness: 0.6, transparent: true, opacity: 0.94 }),
  );
  body.position.set(r.x, r.y, 0);
  group.add(body);
  // contorno e barra colorata in alto: raccolti in due oggetti per tutta la striscia (meno disegni)
  batch.lines.rect(r, color, 0.55);
  batch.bars.push({ x: r.x, y: r.y + r.h / 2 - 0.9, w: r.w - 0.5, h: 1.3, z: 0.56, color });
  const lines = [{ text: title, bold: true, color }];
  if (sub) lines.push({ text: sub, color: "#cbd5e1", scale: SUB / TITLE });
  textPlate({ w: r.w - 1.6, h: TITLE * 2.3, size: TITLE, align: "center", lines }).place(group, r.x, r.y + r.h / 2 - 2 - TITLE * 1.15, 0.7);
  // una frase su cosa fa: si legge avvicinandosi (da lontano bastano titolo e numeri)
  if (text) {
    const body = wrapText(text, r.w - 3, BODY).slice(0, 6);
    textPlate({ w: r.w - 2, h: body.length * BODY * 1.32 + 1, size: BODY, align: "center", lines: body.map((t) => ({ text: t, color: "#94a3b8" })) }).place(group, r.x, r.y - 1.5, 0.7);
  }
  if (params) textPlate({ w: r.w - 1.6, h: PARAMS * 1.7, size: PARAMS, align: "center", mono: true, lines: [{ text: params, color: "#e2e8f0", bold: true }] }).place(group, r.x, r.y - r.h / 2 + PARAMS * 1.15, 0.7);
  return body;
}

export function buildBlocks(group, model, strip, layout) {
  const anchors = new Map();
  const blockBoxes = [];
  const slabs = [];
  const pickables = [];
  const batch = { lines: new LineBatch(), bars: [], ghosts: [] };

  // corsie del residuo: dal blocco del residuo alla norm finale, dietro ai layer (4 per mHC)
  const resIndex = strip.blocks.findIndex((b) => b.id === "residual");
  const normIndex = strip.blocks.findIndex((b) => b.id === "norm");
  if (resIndex >= 0 && normIndex > resIndex) {
    const x0 = layout.blocks[resIndex].x;
    const x1 = layout.blocks[normIndex].x;
    const streams = model.anatomy.residual.streams;
    for (let s = 0; s < streams; s++) segments(group, [[x0, RAIL_Y - s * 1.1, x1, RAIL_Y - s * 1.1]], { color: RAIL_COLORS[s % RAIL_COLORS.length], z: -0.9, opacity: 0.85 });
  }

  strip.blocks.forEach((b, i) => {
    const r = layout.blocks[i];
    const color = b.color ?? "#38bdf8";
    // lastre fantasma dietro ai blocchi che ne rappresentano molti (al massimo 3: bastano a dire "pila")
    const ghosts = b.type === "layer" ? Math.min(3, b.count * b.repeat - 1) : 0;
    for (let g = ghosts; g >= 1; g--) {
      const o = STRIP.ghostOffset * g;
      batch.ghosts.push({ x: r.x + o, y: r.y + o, w: r.w, h: r.h, z: -0.6 - g * 0.5, color: "#0f172a" });
      // più lontana = più scura: il colore del blocco mescolato con lo sfondo
      batch.lines.rect({ x: r.x + o, y: r.y + o, w: r.w, h: r.h }, new THREE.Color(color).lerp(BACKGROUND, 0.35 + g * 0.12), -0.55 - g * 0.5);
    }
    const slab = stripBlock(group, r, { title: b.title, sub: b.sub, text: b.narrow ? null : b.text, params: b.paramsLabel, color }, batch);
    slab.userData.block = i;
    slabs.push(slab);
    pickables.push(slab);
    blockBoxes.push(rectBox(r, 1));
    if (b.type !== "layer") anchors.set(`blocks:${b.id}`, rectBox(r, 2));

    // flag dei layer rappresentati (Engram, → DSpark): pillole sopra i parametri, con i parametri del
    // layer speciale quando sono diversi da quelli del blocco (model.js, layerBlock)
    b.flags?.forEach((f, j) => {
      const p = textPlate({ w: r.w - 2, h: 4.2, size: 2.6, align: "center", lines: [{ text: `${f.label} ${f.layers.join(", ")}${f.note ? ` · ${f.note}` : ""}`, color: f.color ?? "#e2e8f0", bold: true }], background: "rgba(15, 23, 42, 0.95)", border: f.color ?? "#64748b" });
      p.place(group, r.x, r.y - r.h / 2 + PARAMS * 2.6 + 2.4 + j * 4.8, 0.8);
    });
    // distintivo "× N" sul blocco (run compresso, o pezzo di un gruppo con più layer per ripetizione)
    if (b.type === "layer" && b.count > 1) {
      const badge = textPlate({ w: 11, h: 6.6, size: 4.6, align: "center", lines: [{ text: `× ${b.count}`, color: "#0b1220", bold: true }], background: color, border: "#e2e8f0" });
      badge.place(group, r.x + r.w / 2 - 3, r.y + r.h / 2 + 1.6, 1);
      const run = strip.groups.find((g) => g.id === b.item && g.kind === "run");
      if (run) {
        badge.mesh.userData.toggle = run.id;
        pickables.push(badge.mesh);
      }
    }
  });

  // frecce tra ingresso, layer e uscita (tra layer consecutivi passano le corsie del residuo)
  layout.flows.forEach(([x0, x1], i) => {
    const a = strip.blocks[i];
    const c = strip.blocks[i + 1];
    if (a.type === "layer" && c.type === "layer") return;
    arrow(group, [[x0 + 0.4, 0], [x1 - 0.4, 0]], { color: 0x67e8f9, radius: 0.38, head: 2.2, z: 0.4 });
  });

  // parentesi dei gruppi "× N" (i run compressi hanno il distintivo sul blocco)
  strip.groups.forEach((g, i) => {
    const r = layout.groups[i];
    if (r.onBlock) return;
    bracket(group, r.x0, r.x1, r.y, { color: "#cbd5e1" });
    const text = g.expanded ? [{ text: `${g.layers.length} layer veri`, color: "#e2e8f0", bold: true }, { text: "clic: comprimi in «× N»", color: "#94a3b8", scale: 0.62 }] : [{ text: g.label, color: "#f8fafc", bold: true }, { text: g.sub, color: "#cbd5e1", scale: 0.42 }];
    const w = Math.max(20, Math.min(r.x1 - r.x0, 64));
    const plate = textPlate({ w, h: 11.5, size: g.expanded ? 3.4 : 6.4, align: "center", lines: text, background: g.expanded ? "rgba(30, 41, 59, 0.9)" : "rgba(14, 116, 144, 0.35)", border: g.expanded ? "#475569" : "#22d3ee" });
    plate.place(group, (r.x0 + r.x1) / 2, r.y + 6.6, 0.6);
    plate.mesh.userData.toggle = g.id;
    pickables.push(plate.mesh);
  });

  // parentesi dei segmenti (encoder, decoder…)
  strip.segments.forEach((s, i) => {
    const r = layout.segments[i];
    if (!r) return;
    bracket(group, r.x0, r.x1, r.y, { color: s.color, tick: 3 });
    const plate = textPlate({ w: Math.min(140, Math.max(40, r.x1 - r.x0)), h: 11, size: 5.6, align: "center", lines: [{ text: s.label, color: s.color, bold: true }, { text: s.sub, color: "#94a3b8", scale: 0.58 }] });
    plate.place(group, (r.x0 + r.x1) / 2, r.y + 6.4, 0.4);
    anchors.set(`blocks:segment-${s.id}`, new THREE.Box3(new THREE.Vector3(r.x0 - 3, -STRIP.blockH / 2 - 6, -1), new THREE.Vector3(r.x1 + 3, r.y + 8, 1)));
  });

  // riga delle cache: barra per cache, stanghette verso i layer che la scrivono o la leggono
  model.caches.forEach((c, i) => {
    const r = layout.caches[i];
    const rect = { x: (r.x0 + r.x1) / 2, y: r.y, w: r.x1 - r.x0, h: r.h };
    const bar = flatBox(group, rect, { color: "#0b1324", emissive: c.color, intensity: 0.12 });
    bar.userData.anchor = `blocks:cache-${c.id}`;
    pickables.push(bar);
    batch.lines.rect(rect, c.color, 0.2);
    // il testo sta all'inizio della barra (una barra srotolata è lunga centinaia di unità) o, se la
    // barra è più corta del testo, subito dopo la sua fine (layout.js ha già contato l'ingombro)
    // un'etichetta fuori dalla barra ha uno sfondo scuro davanti alle frecce (il tubo di raggio 0,35 a
    // z 0,4 con la punta più larga arriva a z 1,3): una freccia che passa di lì (in Kimi quella verso il
    // Draft MTP) le scorre dietro invece di attraversare il testo
    const outside = r.labelX0 >= r.x1;
    const plate = textPlate({ w: r.labelW, h: r.h - 0.6, size: 3.4, align: "left", lines: [{ text: `${c.title}  ·  ${c.sub}`, color: c.color, bold: true }], background: outside ? "rgba(13, 17, 23, 0.92)" : null });
    plate.place(group, r.labelX0 + r.labelW / 2, rect.y, outside ? 1.4 : 0.4);
    segments(group, r.ticks.map((x) => [x, r.y + r.h / 2, x, -STRIP.blockH / 2]), { color: c.color, z: -0.3, opacity: 0.45 });
    // l'ancora comprende l'etichetta: il segnaposto del meccanismo va dopo, e inquadrare la cache la fa leggere
    anchors.set(`blocks:cache-${c.id}`, rectBox({ x: (r.x0 + r.labelEnd) / 2, y: r.y, w: r.labelEnd - r.x0, h: r.h }, 3));
  });

  // rami: visione sopra l'embedding, draft multi-token sotto l'uscita
  model.branches.forEach((br, i) => {
    const r = layout.branches[i];
    const params = br.params ? `${br.approx ? "≈ " : ""}${(br.params / 1e9).toLocaleString("it-IT", { maximumFractionDigits: 2 })}B` : br.external ? "fuori dal backbone" : "";
    const slab = stripBlock(group, r, { title: br.title, sub: br.sub, text: null, params, color: br.color }, batch);
    slab.userData.anchor = `blocks:${br.id}`;
    slab.userData.branch = br.id;
    pickables.push(slab);
    if (br.external) outline(group, { x: r.x, y: r.y, w: r.w + 1.6, h: r.h + 1.6 }, { color: br.color, dashed: true, z: 0.1 });
    arrow(group, [[r.from.x, r.from.y], [r.to.x, r.to.y]], { color: new THREE.Color(br.color).getHex(), radius: 0.35, head: 2.2, z: 0.4 });
    anchors.set(`blocks:${br.id}`, rectBox(r, 2));
  });

  const layerIdx = strip.blocks.map((b, i) => (b.type === "layer" ? i : -1)).filter((i) => i >= 0);
  const first = layout.blocks[layerIdx[0]];
  const last = layout.blocks[layerIdx.at(-1)];
  anchors.set("blocks:layers", new THREE.Box3(new THREE.Vector3(first.x - first.w / 2 - 4, -STRIP.blockH / 2 - 10, -1), new THREE.Vector3(last.x + last.w / 2 + 4, STRIP.segmentY + 9, 1)));
  const box = layout.box;
  anchors.set("blocks:all", new THREE.Box3(new THREE.Vector3(box.minX, box.minY, -1), new THREE.Vector3(box.maxX, box.maxY, 1)));

  batch.lines.flush(group);
  boxBatch(group, batch.bars);
  boxBatch(group, batch.ghosts, { opacity: 0.9 });

  const packet = makePacket("#fde047", 2);
  packet.visible = false;
  group.add(packet);

  const lit = new Set();
  return {
    anchors,
    blockBoxes,
    pickables,
    packet,
    railY: RAIL_Y,
    /** Accende i blocchi con questi indici (glow del colore del blocco); null spegne tutto. */
    highlight(indices) {
      const want = new Set(indices ?? []);
      for (const i of new Set([...lit, ...want])) {
        const on = want.has(i);
        if (slabs[i]) glow(slabs[i], on ? 0.5 : 0, strip.blocks[i].color);
        if (on) lit.add(i);
        else lit.delete(i);
      }
    },
  };
}
