/**
 * Livello Dettaglio della vista completa: l'anatomia di UN layer (quello scelto), costruita dai dati
 * del modello con le posizioni di layout.js, più il contesto intorno: le sagome dei layer della
 * striscia (per sapere dove sei) e le schede dei pezzi che non sono layer (tokenizer, head, cache…).
 *
 *   const anatomy = buildAnatomy(model, layer, totals);   // un gruppo in coordinate locali
 *   anatomy.anchors Map("detail:<id>" → {x, y, w, h})   anatomy.cards Map(id → {meshes, state})
 *   anatomy.highlight(id | null)   anatomy.route(esperti, t)   anatomy.setPulse(id, k)
 *   buildContext(group, model, strip, layout)            sagome dei layer e schede dei pezzi fissi
 * Le schede dicono il nome vero del tensore e la sua precisione (colore = precisione).
 */
import * as THREE from "three";
import { glow } from "../shapes.js";
import { backPlane, LineBatch, outline, rectBox, segments, textPlate } from "./draw.js";
import { anatomyLayout } from "./layout.js";
import { anatomyCards, cardState, formatCount, layerParams } from "./model.js";
import { DTYPE_COLORS } from "./sceneMacro.js";

/** Colori delle sezioni dell'anatomia. */
const SECTION = { attention: "#38bdf8", ffn: "#a78bfa", branch: "#fb923c", node: "#94a3b8" };
const EXPERT_OFF = new THREE.Color("#2e2850");
const EXPERT_ON = new THREE.Color("#ddd6fe");
const tmpObject = new THREE.Object3D();

/**
 * Scheda dell'anatomia: scatola piatta, bordo del colore della sezione (tratteggiato se il pezzo è
 * riusato da un altro layer) e testo su tre righe: nome, cosa fa, tensore (colore = precisione).
 */
function anatomyCard(parent, rect, spec, { state = "on", accent = SECTION.attention, kind, lines: batch } = {}) {
  const s = { ...spec, ...(spec.alt?.[kind] ?? {}) };
  const ghost = state === "ghost";
  const material = new THREE.MeshStandardMaterial({ color: "#111827", emissive: accent, emissiveIntensity: 0, roughness: 0.6, transparent: true, opacity: ghost ? 0.45 : 0.95 });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(rect.w, rect.h, 0.4), material);
  mesh.position.set(rect.x, rect.y, 0);
  mesh.userData.card = spec.id;
  parent.add(mesh);
  if (ghost || !batch) outline(parent, rect, { color: ghost ? "#64748b" : accent, z: 0.25, dashed: ghost });
  else batch.rect(rect, accent, 0.25); // contorni pieni: tutti in un solo oggetto
  const dtype = s.dtype ? ` · ${s.dtype}` : "";
  const lines = [{ text: s.title, bold: true, color: ghost ? "#94a3b8" : "#f8fafc" }, { text: ghost ? `riusato · ${s.sub}` : s.sub, color: ghost ? "#64748b" : "#cbd5e1", scale: 0.66 }];
  if (s.tensor) lines.push({ text: `${s.tensor}${dtype}`, color: ghost ? "#475569" : (DTYPE_COLORS[(s.dtype ?? "").split(" ")[0]] ?? "#94a3b8"), scale: 0.5 });
  const plate = textPlate({ w: rect.w - 0.4, h: rect.h - 0.3, size: Math.min(1.9, rect.h / 3.2), align: "center", lines });
  plate.place(parent, rect.x, rect.y, 0.3);
  plate.mesh.userData.card = spec.id;
  return { mesh, plate, accent, state };
}

/** Freccia corta (segmento + punta) per i collegamenti interni: tutte in pochi oggetti. */
function connectors(parent, pairs, color, z = 0.15) {
  const lines = [];
  const heads = [];
  for (const [x1, y1, x2, y2] of pairs) {
    lines.push([x1, y1, x2, y2]);
    const a = Math.atan2(y2 - y1, x2 - x1);
    const h = 0.9;
    heads.push([x2, y2, x2 - h * Math.cos(a - 0.45), y2 - h * Math.sin(a - 0.45)], [x2, y2, x2 - h * Math.cos(a + 0.45), y2 - h * Math.sin(a + 0.45)]);
  }
  segments(parent, [...lines, ...heads], { color, z });
}

/**
 * L'anatomia del layer `layer` ({number, kind, flags}) del modello: un gruppo nuovo, da liberare
 * con disposeTree quando si cambia tipo di layer.
 */
export function buildAnatomy(model, layer, segmentName = "") {
  const group = new THREE.Group();
  const L = anatomyLayout(model, layer);
  const a = model.anatomy;
  const kind = model.kinds[layer.kind];
  const box = L.box;
  const anchors = new Map();
  const cards = new Map();
  const lines = new LineBatch();

  // fondo che scrive la profondità: copre le sagome e le schede del contesto che stanno dietro
  // (i pannelli di testo hanno renderOrder alto: senza profondità passerebbero davanti)
  const back = backPlane(group, { x: (box.minX + box.maxX) / 2, y: (box.minY + box.maxY) / 2, w: box.maxX - box.minX, h: box.maxY - box.minY }, { color: "#0a101d", opacity: 0.97, z: -0.8 });
  back.material.depthWrite = true;
  outline(group, { x: (box.minX + box.maxX) / 2, y: (box.minY + box.maxY) / 2, w: box.maxX - box.minX, h: box.maxY - box.minY }, { color: kind.color, z: -0.7 });

  // titolo: numero e tipo del layer, parametri con le parti
  const lp = layerParams(model, layer);
  const parts = [
    lp.parts.routed && `esperti ${formatCount(lp.parts.routed)}`,
    lp.parts.dense && `FFN densa ${formatCount(lp.parts.dense)}`,
    `attention ${formatCount(lp.parts.attention)}`,
    lp.parts.shared && `condivisi ${formatCount(lp.parts.shared)}`,
    lp.parts.latent && `latente ${formatCount(lp.parts.latent)}`,
    lp.parts.router && `router ${formatCount(lp.parts.router)}`,
    lp.parts.residual && `mHC ${formatCount(lp.parts.residual)}`,
    lp.parts.flagged && `Engram ${formatCount(lp.parts.flagged)}`,
  ].filter(Boolean);
  const title = textPlate({ w: 150, h: 5, size: 3, align: "center", lines: [{ text: `Layer ${layer.number}${segmentName ? ` · ${segmentName}` : ""} · ${kind.name}`, color: "#f8fafc", bold: true }] });
  title.place(group, (box.minX + box.maxX) / 2 + (L.branch ? 14 : 0), 50, 0.3);
  const sub = textPlate({ w: 170, h: 3.6, size: 1.8, align: "center", lines: [{ text: `${formatCount(lp.total)} parametri: ${parts.join(" · ")}`, color: "#94a3b8" }] });
  sub.place(group, (box.minX + box.maxX) / 2 + (L.branch ? 14 : 0), 45.2, 0.3);

  // corsie del residuo
  const railColors = ["#38bdf8", "#0ea5e9", "#7dd3fc", "#0284c7"];
  L.rails.y.forEach((y, i) => segments(group, [[L.rails.x0, y, L.rails.x1, y]], { color: railColors[i % 4], z: 0.05 }));
  const railLabel = textPlate({ w: 60, h: 3, size: 1.6, lines: [{ text: a.residual.label, color: "#7dd3fc", bold: true }] });
  railLabel.place(group, L.rails.x0 + 31, L.rails.y[0] + 2.4, 0.3);
  const railTop = L.rails.y[0];
  const railBottom = L.rails.y.at(-1);
  anchors.set("detail:residual", { x: 0, y: (railTop + railBottom) / 2, w: L.rails.x1 - L.rails.x0, h: railTop - railBottom + 6 });

  // nodi di ingresso e uscita delle due metà (pre-mix, norm, post-mix)
  const n = L.nodes;
  const nodeCard = (rect, id, titleText, subText, tensor, dtype) => rect && cards.set(id, anatomyCard(group, rect, { id, title: titleText, sub: subText, tensor, dtype }, { accent: SECTION.node, lines }));
  nodeCard(n.preA, "preA", a.residual.pre, a.residual.preSub ?? "4 → 1", a.residual.tensor, a.residual.dtype);
  nodeCard(n.normA, "normA", a.norm.title, "prima dell'attention", a.norm.attnTensor, a.norm.dtype);
  nodeCard(n.postA, "postA", a.residual.post, "1 → 4", null);
  nodeCard(n.preF, "preF", a.residual.pre, a.residual.preSub ?? "4 → 1", null);
  nodeCard(n.normF, "normF", a.norm.title, "prima della FFN", a.norm.ffnTensor, a.norm.dtype);
  nodeCard(n.postF, "postF", a.residual.post, "1 → 4", null);

  // pannelli
  const panelRect = (p) => ({ x: (p.x0 + p.x1) / 2, y: (p.y0 + p.y1) / 2, w: p.x1 - p.x0, h: p.y1 - p.y0 });
  const attnRect = panelRect(L.panels.attention);
  const ffnRect = panelRect(L.panels.ffn);
  outline(group, attnRect, { color: SECTION.attention, z: -0.2, opacity: 0.8 });
  outline(group, ffnRect, { color: SECTION.ffn, z: -0.2, opacity: 0.8 });
  const attnTitle = a.attention.titles?.[layer.kind] ?? a.attention.title;
  textPlate({ w: attnRect.w - 4, h: 3.6, size: 2.2, lines: [{ text: `${attnTitle} · ${kind.label}`, color: SECTION.attention, bold: true }] }).place(group, attnRect.x, L.panels.attention.y1 - 2.4, 0.3);
  textPlate({ w: ffnRect.w - 2, h: 3.6, size: 2.1, lines: [{ text: L.grid ? a.ffn.title : "FFN densa (questo layer non ha il MoE)", color: SECTION.ffn, bold: true }] }).place(group, ffnRect.x, L.panels.ffn.y1 - 2.4, 0.3);
  anchors.set("detail:layer", { x: (box.minX + box.maxX) / 2, y: (box.minY + box.maxY) / 2, w: box.maxX - box.minX, h: box.maxY - box.minY });

  // righe dell'attention: etichetta, note per le righe vuote
  for (const row of L.rows) {
    textPlate({ w: 12, h: 3, size: 1.45, align: "left", lines: [{ text: row.label, color: "#7dd3fc", bold: true }] }).place(group, L.panels.attention.x0 + 7.4, row.y, 0.3);
    if (row.note) textPlate({ w: 54, h: 3, size: 1.4, align: "left", lines: [{ text: row.note, color: "#64748b" }] }).place(group, -66 + 27, row.y, 0.3);
  }

  // schede: attention, FFN, ramo
  const all = anatomyCards(model);
  for (const spec of all) {
    const rect = L.cards.get(spec.id);
    if (!rect) continue;
    cards.set(spec.id, anatomyCard(group, rect, spec, { state: rect.state, accent: SECTION[spec.section] ?? SECTION.attention, kind: layer.kind, lines }));
    anchors.set(`detail:${spec.id}`, rect);
  }

  // collegamenti: lungo ogni riga, righe → nucleo, nucleo → uscita, nodi e corsie
  const flow = [];
  const rowsOf = new Map();
  for (const spec of all) {
    const r = L.cards.get(spec.id);
    if (!r || spec.section !== "attention" || !spec.row || spec.row === "output") continue;
    if (!rowsOf.has(spec.row)) rowsOf.set(spec.row, []);
    rowsOf.get(spec.row).push(r);
  }
  const core = L.cards.get(a.attention.core.id);
  for (const list of rowsOf.values()) {
    list.sort((p, q) => p.x - q.x);
    for (let i = 1; i < list.length; i++) flow.push([list[i - 1].x + list[i - 1].w / 2, list[i].y, list[i].x - list[i].w / 2, list[i].y]);
    const last = list.at(-1);
    if (core) flow.push([last.x + last.w / 2, last.y, core.x - core.w / 2, Math.max(core.y - core.h / 2 + 1, Math.min(core.y + core.h / 2 - 1, last.y))]);
  }
  const outs = a.attention.output.map((c) => L.cards.get(c.id)).filter(Boolean);
  if (core && outs.length) {
    // uscita in colonna sotto il nucleo: nucleo → prima scheda → … (frecce verso il basso)
    flow.push([core.x, core.y - core.h / 2, outs[0].x, outs[0].y + outs[0].h / 2]);
    for (let i = 1; i < outs.length; i++) flow.push([outs[i - 1].x, outs[i - 1].y - outs[i - 1].h / 2, outs[i].x, outs[i].y + outs[i].h / 2]);
  }
  // ingresso dell'attention: dalla norm un "bus" verticale, poi una freccia verso ogni riga
  const firsts = [...rowsOf.values()].map((list) => list[0]);
  if (n.normA && firsts.length) {
    const busX = Math.min(...firsts.map((r) => r.x - r.w / 2)) - 2.2;
    const ys = firsts.map((r) => r.y);
    const bus = [[n.normA.x + n.normA.w / 2, n.normA.y, busX, n.normA.y], [busX, Math.max(n.normA.y, ...ys), busX, Math.min(n.normA.y, ...ys)]];
    segments(group, bus, { color: "#38bdf8", z: 0.15 });
    for (const r of firsts) flow.push([busX, r.y, r.x - r.w / 2, r.y]);
  }
  connectors(group, flow, "#38bdf8");

  // corsie ↔ nodi: giù nel pre-mix, su dal post-mix
  const mix = [];
  const railY = L.rails.y.at(-1);
  if (n.preA) mix.push([n.preA.x, railY, n.preA.x, n.preA.y + n.preA.h / 2], [n.preA.x, n.preA.y - n.preA.h / 2, n.normA.x, n.normA.y + n.normA.h / 2]);
  else mix.push([n.normA.x, railY, n.normA.x, n.normA.y + n.normA.h / 2]);
  const attnExit = outs.at(-1);
  if (n.postA) {
    if (attnExit) mix.push([attnExit.x + attnExit.w / 2, attnExit.y, n.postA.x, attnExit.y], [n.postA.x, attnExit.y, n.postA.x, n.postA.y - n.postA.h / 2]);
    mix.push([n.postA.x, n.postA.y + n.postA.h / 2, n.postA.x, railY]);
    mix.push([n.preF.x, railY, n.preF.x, n.preF.y + n.preF.h / 2], [n.preF.x, n.preF.y - n.preF.h / 2, n.normF.x, n.normF.y + n.normF.h / 2]);
  } else {
    // residuo semplice: l'uscita dell'attention si somma alla corsia (nodo "+") e la norm della FFN la rilegge
    if (attnExit) mix.push([attnExit.x + attnExit.w / 2, attnExit.y, n.normF.x - 6, attnExit.y], [n.normF.x - 6, attnExit.y, n.normF.x - 6, railY]);
    mix.push([n.normF.x, railY, n.normF.x, n.normF.y + n.normF.h / 2]);
  }
  connectors(group, mix, "#7dd3fc");

  // FFN: norm → router (o FFN densa), router → griglia, griglia → somma, somma → corsie
  const fflow = [];
  const f = a.ffn;
  const router = L.cards.get(f.router?.id);
  const sumRect = L.cards.get(f.sum?.id);
  const denseRect = f.dense ? L.cards.get(f.dense.id) : null;
  if (denseRect) fflow.push([n.normF.x + n.normF.w / 2, n.normF.y, denseRect.x - denseRect.w / 2, denseRect.y]);
  if (router) fflow.push([n.normF.x + n.normF.w / 2, n.normF.y, router.x - router.w / 2, router.y]);
  if (router && L.grid) fflow.push([router.x, router.y - router.h / 2, router.x, L.grid.y + L.grid.h / 2 + 0.5]);
  const exitX = n.postF ? n.postF.x : L.panels.ffn.x1 + 4;
  const exitFrom = sumRect ?? denseRect;
  if (exitFrom) fflow.push([exitFrom.x + exitFrom.w / 2, exitFrom.y, exitX, exitFrom.y], [exitX, exitFrom.y, exitX, n.postF ? n.postF.y - n.postF.h / 2 : railY]);
  if (n.postF) fflow.push([n.postF.x, n.postF.y + n.postF.h / 2, n.postF.x, railY]);
  connectors(group, fflow, "#a78bfa");

  // griglia degli esperti (InstancedMesh): spenti, si accendono con route()
  let grid = null;
  if (L.grid) {
    const g = L.grid;
    const geometry = new THREE.BoxGeometry(g.cell * 0.82, g.cell * 0.82, 0.25);
    const material = new THREE.MeshStandardMaterial({ color: "#ffffff", emissive: "#7c3aed", emissiveIntensity: 0.0, roughness: 0.55 });
    const mesh = new THREE.InstancedMesh(geometry, material, g.count);
    for (let i = 0; i < g.count; i++) {
      const col = i % g.cols;
      const row = Math.floor(i / g.cols);
      tmpObject.position.set(g.x - g.w / 2 + (col + 0.5) * g.cell, g.y + g.h / 2 - (row + 0.5) * g.cell, 0);
      tmpObject.updateMatrix();
      mesh.setMatrixAt(i, tmpObject.matrix);
      mesh.setColorAt(i, EXPERT_OFF);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.userData.card = "experts";
    group.add(mesh);
    const moe = model.params.moe;
    textPlate({ w: g.w + 2, h: 2.6, size: 1.35, align: "center", lines: [{ text: `${f.grid.label} · ${f.grid.dtype} · ${moe.active} accesi per token (simulazione didattica)`, color: "#c4b5fd", bold: true }] }).place(group, g.x, g.labelY, 0.3);
    anchors.set("detail:experts", { x: g.x, y: g.y + 2.2, w: g.w + 2, h: g.h + 8.4 }); // griglia + la sua etichetta
    grid = { mesh, count: g.count, lit: new Set() };
  }

  // ramo laterale (Engram): colonna a sinistra, freccia nel residuo
  if (L.branch) {
    const b = L.branch;
    outline(group, { x: b.x, y: (b.y0 + b.y1) / 2, w: b.x1 - b.x0, h: b.y1 - b.y0 }, { color: SECTION.branch, z: -0.2, dashed: true });
    textPlate({ w: b.x1 - b.x0, h: 3.2, size: 1.7, align: "center", lines: [{ text: b.label, color: SECTION.branch, bold: true }] }).place(group, b.x, b.y1 + 2.2, 0.3);
    const list = a.branch.cards.map((c) => L.cards.get(c.id));
    const pairs = [];
    for (let i = 1; i < list.length; i++) pairs.push([b.x, list[i - 1].y - list[i - 1].h / 2, b.x, list[i].y + list[i].h / 2]);
    const last = list.at(-1);
    pairs.push([last.x + last.w / 2, last.y, L.rails.x0 + 4, last.y], [L.rails.x0 + 4, last.y, L.rails.x0 + 4, L.rails.y.at(-1)]);
    connectors(group, pairs, SECTION.branch);
  }

  lines.flush(group);

  let lit = null;
  return {
    group,
    anchors,
    cards,
    box,
    /** Accende una scheda (o "experts" per la griglia, "residual" per le corsie); null spegne. */
    highlight(id) {
      if (lit === id) return;
      if (lit && cards.has(lit)) glow(cards.get(lit).mesh, 0);
      if (id && cards.has(id)) glow(cards.get(id).mesh, 0.55, cards.get(id).accent);
      lit = id;
    },
    /** Esperti accesi (indici) con intensità t (0..1): simulazione didattica dichiarata nella scena. */
    route(indices, t = 1) {
      if (!grid) return;
      const want = new Set(indices ?? []);
      for (const i of new Set([...grid.lit, ...want])) {
        const on = want.has(i) && t > 0;
        grid.mesh.setColorAt(i, on ? EXPERT_OFF.clone().lerp(EXPERT_ON, t) : EXPERT_OFF);
        if (on) grid.lit.add(i);
        else grid.lit.delete(i);
      }
      grid.mesh.material.emissiveIntensity = want.size && t > 0 ? 0.08 : 0;
      grid.mesh.instanceColor.needsUpdate = true;
    },
    /** Box3 di un'ancora in coordinate di scena, sapendo dove sta il gruppo. */
    anchorBox(id, pad = 1.5) {
      const r = anchors.get(id);
      return r ? rectBox(r, pad, group.position) : null;
    },
    /** Il nome del layer nel titolo cambia senza ricostruire (stesso tipo e stessi flag). */
    setTitle(text) {
      title.set([{ text, color: "#f8fafc", bold: true }]);
    },
  };
}

/** Firma di un layer per l'anatomia: due layer con la stessa firma si disegnano uguali. */
export function anatomySignature(layer) {
  return `${layer.kind}|${[...layer.flags].sort().join(",")}`;
}

/**
 * Contesto del Dettaglio: le sagome dei layer della striscia (dove sei e dove andare) e le schede
 * dei pezzi che non sono layer (tokenizer, embedding, head, rami, cache) con i dettagli dei dati.
 * Restituisce gli oggetti cliccabili (sagome: userData.block = indice del blocco).
 */
export function buildContext(group, model, strip, layout) {
  const pickables = [];
  strip.blocks.forEach((b, i) => {
    const r = layout.blocks[i];
    if (b.type === "layer") {
      const back = backPlane(group, r, { color: "#0b1220", opacity: 0.6, z: -1.2 });
      back.userData.block = i;
      back.userData.isFrame = false; // cliccabile e parte dei confini del Dettaglio
      pickables.push(back);
      outline(group, r, { color: b.color, z: -1.1, opacity: 0.7 });
      textPlate({ w: r.w - 1, h: 6, size: 2.3, align: "center", lines: [{ text: b.title, color: b.color, bold: true }, { text: b.sub, color: "#94a3b8", scale: 0.75 }] }).place(group, r.x, r.y + r.h / 2 - 4, -1);
      return;
    }
    const lines = [{ text: b.title, color: b.color ?? "#e2e8f0", bold: true }, ...(b.detail ?? [b.sub, b.text]).map((t) => ({ text: t, color: "#cbd5e1", scale: 0.62 }))];
    const plate = textPlate({ w: r.w + 10, h: r.h, size: 2.4, align: "center", lines, background: "rgba(15, 23, 42, 0.95)", border: b.color ?? "#475569" });
    plate.place(group, r.x, r.y, -0.5);
  });
  model.branches.forEach((br, i) => {
    const r = layout.branches[i];
    const lines = [{ text: br.title, color: br.color, bold: true }, ...(br.detail ?? [br.sub, br.text]).map((t) => ({ text: t, color: "#cbd5e1", scale: 0.6 }))];
    textPlate({ w: r.w + 12, h: r.h + 4, size: 2.4, align: "center", lines, background: "rgba(15, 23, 42, 0.95)", border: br.color }).place(group, r.x, r.y, -0.5);
  });
  model.caches.forEach((c, i) => {
    const r = layout.caches[i];
    const lines = [{ text: `${c.title} · ${c.sub}`, color: c.color, bold: true }, ...(c.detail ?? [c.text]).map((t) => ({ text: t, color: "#cbd5e1", scale: 0.8 }))];
    const w = Math.min(r.x1 - r.x0, 140);
    textPlate({ w, h: Math.max(r.h, 2.6 + 2.2 * lines.length), size: 1.8, align: "left", lines, background: "rgba(15, 23, 42, 0.95)", border: c.color }).place(group, r.x0 + w / 2, r.y, -0.5);
  });
  return { pickables };
}
