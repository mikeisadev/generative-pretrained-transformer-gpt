/**
 * Posizioni PURE della vista completa (niente three.js; test in tests/modern-llm-full.test.mjs).
 * La scena è una parete (piano x-y) letta da sinistra a destra, centrata nell'origine:
 *   Blocchi     stripLayout(strip, model)      la striscia dell'architettura, con parentesi e cache
 *   Panoramica  macroLayout(model, scale)       la pipeline, centrata sulla fase "Pesi" (x = 0): dallo
 *                                              zoom più lontano la fase dei pesi è dove sta l'architettura
 *   Dettaglio   anatomyLayout(model, layer)     l'anatomia di un layer (coordinate locali del gruppo)
 * Ogni rettangolo è {x, y, w, h} con (x, y) = centro.
 */
import { MARKER_LABELS } from "./catalog.js";
import { anatomyCards, cardState, selectLayers } from "./model.js";

// ------------------------------------------------------------------------------------------------
// Livello Blocchi
// ------------------------------------------------------------------------------------------------

export const STRIP = {
  layerW: 32,
  ioW: 34,
  narrowW: 20,
  blockH: 62,
  gap: 6,
  itemGap: 4, // spazio in più tra due ripetizioni diverse
  segmentGap: 10, // spazio in più al confine di un segmento (encoder | decoder)
  bracketY: 38, // parentesi "× N" sopra i blocchi
  segmentY: 53, // parentesi dei segmenti
  branchGap: 18, // distanza dei rami (visione, draft) dalla striscia
  cacheTop: -40, // prima riga delle cache
  cacheH: 7,
  cacheStep: 9.5,
  ghostOffset: 2.6, // spostamento delle lastre fantasma dietro un blocco "× N"
  cacheLabelChar: 1.85, // larghezza media di un carattere dell'etichetta di una cache (unità)
};

/**
 * @param {ReturnType<import("./model.js").buildStrip>} strip
 * @param {object} model
 * @returns {{blocks, groups, segments, caches, branches, flows, box: {minX, maxX, minY, maxY}, width}}
 */
/** Larghezza (unità) del segnaposto di un meccanismo alto 7 nei Blocchi: draw.js markerSprite. */
export function markerWidth(text, height = 7) {
  return height * (1.33 + 0.24 * text.length);
}

export function stripLayout(strip, model) {
  const S = STRIP;
  const xs = [];
  let x = 0;
  strip.blocks.forEach((b, i) => {
    const w = b.type === "layer" ? S.layerW : b.narrow ? S.narrowW : S.ioW;
    if (i > 0) {
      x += S.gap;
      const prev = strip.blocks[i - 1];
      if (b.type === "layer" && prev.type === "layer" && b.item !== prev.item) x += S.itemGap;
      if (strip.segments.some((s) => s.from === i && i > 0)) x += S.segmentGap;
      if (b.type !== prev.type) x += S.itemGap; // ingresso | layer | uscita
    }
    xs.push({ x0: x, w });
    x += w;
  });
  const width = x;
  const shift = -width / 2;
  const blocks = xs.map(({ x0, w }) => ({ x: x0 + shift + w / 2, y: 0, w, h: S.blockH }));
  const spanOf = (from, to) => ({ x0: blocks[from].x - blocks[from].w / 2, x1: blocks[to].x + blocks[to].w / 2 });

  // parentesi "× N" dei gruppi compressi (i run "× N" di un solo blocco hanno il distintivo sul blocco)
  const groups = strip.groups.map((g) => ({ ...spanOf(g.from, g.to), y: S.bracketY, id: g.id, onBlock: g.kind === "run" && !g.expanded }));
  const segments = strip.segments.filter((s) => s.from !== undefined).map((s) => ({ ...spanOf(s.from, s.to), y: S.segmentY, id: s.id }));

  // cache: una barra per cache, sotto i blocchi dei layer che la scrivono o la leggono; barre che si
  // sovrappongono in x vanno su righe diverse
  const layerBlocks = strip.blocks.map((b, i) => ({ b, i })).filter(({ b }) => b.type === "layer");
  const rows = [];
  const caches = model.caches.map((c) => {
    const wanted = new Set(selectLayers(model, c.layers));
    const hit = layerBlocks.filter(({ b }) => b.layers.some((n) => wanted.has(n))).map(({ i }) => i);
    const x0 = Math.min(...hit.map((i) => blocks[i].x - blocks[i].w / 2));
    const x1 = Math.max(...hit.map((i) => blocks[i].x + blocks[i].w / 2));
    // l'etichetta può uscire dalla barra a destra (barra corta sotto un solo blocco): conta anche
    // lei, e così il segnaposto di un meccanismo che vive su questa cache (va dopo l'etichetta)
    const labelW = Math.min(160, (c.title.length + c.sub.length + 5) * S.cacheLabelChar);
    // dentro la barra se ci sta, altrimenti subito dopo la sua fine (barra | etichetta)
    const labelInside = labelW <= x1 - x0 - 1;
    const labelX0 = labelInside ? x0 + 0.5 : x1 + 1.5;
    const labelEnd = labelInside ? x1 : labelX0 + labelW;
    const hosted = Object.entries(model.mechanisms ?? {}).filter(([, m]) => m.at === `blocks:cache-${c.id}`);
    const markerW = hosted.length ? 4.5 + Math.max(...hosted.map(([id]) => markerWidth(MARKER_LABELS[id] ?? id))) : 0;
    const reach = labelEnd + markerW;
    let row = 0;
    while ((rows[row] ?? []).some((r) => r.x0 < reach + 2 && r.x1 > x0 - 2)) row++;
    (rows[row] ??= []).push({ x0, x1: reach });
    return { id: c.id, x0, x1, labelW, labelX0, labelEnd, reach, y: S.cacheTop - row * S.cacheStep, h: S.cacheH, ticks: hit.map((i) => blocks[i].x) };
  });
  const cacheBottom = caches.length ? Math.min(...caches.map((c) => c.y - c.h / 2)) : -S.blockH / 2;

  // rami: sopra il blocco a cui si attaccano (visione → embedding), sotto l'uscita (draft multi-token)
  const lastLayer = layerBlocks.at(-1)?.i ?? 0;
  const firstOutput = strip.blocks.findIndex((b) => b.type === "output");
  const branches = model.branches.map((br) => {
    const w = S.ioW + 8;
    const h = 30;
    if (br.side === "top") {
      const at = strip.blocks.findIndex((b) => b.id === br.attach);
      const bx = blocks[Math.max(0, at)].x;
      const y = S.blockH / 2 + S.branchGap + h / 2;
      return { id: br.id, x: bx, y, w, h, from: { x: bx, y: y - h / 2 }, to: { x: bx, y: S.blockH / 2 } };
    }
    // sotto le uscite, accanto alla riga delle cache: freccia dal layer finale e verso l'uscita
    const right = blocks[strip.blocks.length - 1];
    const left = blocks[firstOutput >= 0 ? firstOutput : lastLayer];
    const bx = (left.x - left.w / 2 + right.x + right.w / 2) / 2;
    const y = -S.blockH / 2 - S.branchGap - h / 2 + 4;
    const src = blocks[lastLayer];
    return { id: br.id, x: bx, y, w: Math.max(w, right.x + right.w / 2 - (left.x - left.w / 2)), h, from: { x: src.x + src.w / 2 - 3, y: -S.blockH / 2 }, to: { x: bx, y: y + h / 2 } };
  });

  // frecce del flusso tra blocchi consecutivi
  const flows = [];
  for (let i = 1; i < blocks.length; i++) flows.push([blocks[i - 1].x + blocks[i - 1].w / 2, blocks[i].x - blocks[i].w / 2]);

  // un meccanismo su un ramo ha il segnaposto sopra il ramo alto e sotto quello basso (index.js
  // placeMarkers): il riquadro della striscia lo comprende, così la vista di casa non lo taglia
  const hostsMarker = (id) => Object.values(model.mechanisms ?? {}).some((m) => m.at === `blocks:${id}`);
  const top = Math.max(S.segmentY + 5, ...branches.map((b) => b.y + b.h / 2 + (b.from.y < b.y && hostsMarker(b.id) ? 11 : 0)));
  const bottom = Math.min(cacheBottom, ...branches.map((b) => b.y - b.h / 2 - (b.from.y > b.y && hostsMarker(b.id) ? 11 : 0)));
  // a destra conta anche un'etichetta di cache (con il suo segnaposto) che esce oltre l'ultimo blocco
  const right = Math.max(-shift + 6, ...caches.map((c) => c.reach + 2));
  return { blocks, groups, segments, caches, branches, flows, width, box: { minX: shift - 6, maxX: right, minY: bottom - 4, maxY: top + 3 } };
}

// ------------------------------------------------------------------------------------------------
// Livello Panoramica
// ------------------------------------------------------------------------------------------------

export const MACRO = { stageW: 112, stageH: 120, step: 134, noteY: -128, noteH: 76, titleY: 96 };

/**
 * La pipeline da sinistra a destra, con la fase "weights" in x = 0. `scale` ingrandisce tutto (testi
 * compresi) quando la striscia dei Blocchi si allunga (gruppi srotolati): la Panoramica resta lo zoom
 * più lontano.
 */
export function macroLayout(model, scale = 1) {
  const M = MACRO;
  const center = Math.max(0, model.pipeline.findIndex((s) => s.id === "weights"));
  const stages = model.pipeline.map((s, i) => ({ id: s.id, x: (i - center) * M.step * scale, y: 0, w: M.stageW * scale, h: M.stageH * scale }));
  const first = stages[0];
  const last = stages.at(-1);
  const split = stages[Math.min(stages.length - 1, Math.max(0, center - 1))];
  const noteY = M.noteY * scale;
  const capabilities = { x: (first.x - first.w / 2 + split.x + split.w / 2) / 2, y: noteY, w: split.x + split.w / 2 - (first.x - first.w / 2), h: M.noteH * scale };
  const weightsShare = { x: (stages[center].x - stages[center].w / 2 + last.x + last.w / 2) / 2, y: noteY, w: last.x + last.w / 2 - (stages[center].x - stages[center].w / 2), h: M.noteH * scale };
  const minX = first.x - first.w / 2 - 8 * scale;
  const maxX = last.x + last.w / 2 + 8 * scale;
  return {
    stages,
    capabilities,
    weightsShare,
    title: { x: (minX + maxX) / 2, y: M.titleY * scale },
    box: { minX, maxX, minY: noteY - (M.noteH / 2 + 8) * scale, maxY: (M.titleY + 10) * scale },
  };
}

// ------------------------------------------------------------------------------------------------
// Livello Dettaglio: l'anatomia di un layer
// ------------------------------------------------------------------------------------------------

export const ANATOMY = {
  halfW: 98,
  titleY: 50,
  subY: 45,
  railTop: 37,
  railStep: 1.25,
  colA: -89,
  colB: 21,
  colC: 35,
  colD: 92,
  mixY: 25,
  normY: 15,
  nodeW: 14,
  nodeH: 6,
  attn: { x0: -80, x1: 14, y0: -50, y1: 29 },
  ffn: { x0: 44, x1: 86, y0: -50, y1: 29 },
  rowY0: 18,
  rowStep: 11,
  cardW: 13,
  cardH: 7,
  cardGap: 2.2,
  cardsX0: -66,
  core: { x0: -4, x1: 12 },
  outY: -37,
  branchX: -116,
};

/**
 * Rettangoli dell'anatomia per un layer (kind e flag): solo le schede presenti ("on") o riusate
 * ("ghost"); le righe vuote per quel tipo diventano una nota. Coordinate locali del gruppo
 * dell'anatomia (centro del Dettaglio in 0, 0).
 * @returns {{cards: Map<string, {x, y, w, h, state}>, rows: {id, label, y, note?}[], panels: {attention, ffn, branch?},
 *   nodes: {preA, normA, postA, preF, normF, postF}, rails: {y: number[], x0, x1}, grid: {x, y, w, h, cols, rows, cell}, box}}
 */
export function anatomyLayout(model, layer) {
  const A = ANATOMY;
  const anatomy = model.anatomy;
  const cards = new Map();
  const put = (id, x, y, w, h, state = "on") => cards.set(id, { x, y, w, h, state });

  // corsie del residuo (4 per mHC, 1 altrimenti)
  const streams = anatomy.residual.streams;
  const rails = { y: Array.from({ length: streams }, (_, i) => A.railTop - i * A.railStep), x0: -A.halfW, x1: A.halfW };

  // nodi di ingresso e uscita delle due metà del layer (pre-mix, norm, post-mix)
  const node = (x, y) => ({ x, y, w: A.nodeW, h: A.nodeH });
  const hasMix = Boolean(anatomy.residual.pre);
  const nodes = {
    preA: hasMix ? node(A.colA, A.mixY) : null,
    normA: node(A.colA, hasMix ? A.normY : A.mixY - 4),
    postA: anatomy.residual.post ? node(A.colB, A.mixY) : null,
    preF: anatomy.residual.post ? node(A.colC, A.mixY) : null,
    normF: node(A.colC, anatomy.residual.post ? A.normY : A.mixY - 4),
    postF: anatomy.residual.post ? node(A.colD, A.mixY) : null,
  };

  // attention: righe presenti in questo layer, schede da sinistra a destra
  const all = anatomyCards(model);
  const byId = new Map(all.map((c) => [c.id, c]));
  const rows = [];
  let rowIndex = 0;
  for (const row of anatomy.attention.rows) {
    if (row.in && !row.in.includes(layer.kind)) continue;
    const y = A.rowY0 - rowIndex * A.rowStep;
    rowIndex++;
    const visible = row.cards.filter((c) => cardState(byId.get(c.id), layer) !== "off");
    const note = visible.length ? null : (row.empty?.[layer.kind] ?? "non usato in questo layer");
    rows.push({ id: row.id, label: row.label, y, note });
    visible.forEach((c, i) => put(c.id, A.cardsX0 + A.cardW / 2 + i * (A.cardW + A.cardGap), y, A.cardW, A.cardH, cardState(byId.get(c.id), layer)));
  }
  const coreTop = A.rowY0 + A.cardH / 2;
  const coreBottom = A.rowY0 - (Math.max(1, rowIndex) - 1) * A.rowStep - A.cardH / 2;
  put(anatomy.attention.core.id, (A.core.x0 + A.core.x1) / 2, (coreTop + coreBottom) / 2, A.core.x1 - A.core.x0, Math.max(A.cardH, coreTop - coreBottom));
  // uscita sotto il nucleo, in colonna: il dato scende dal nucleo e poi esce a destra verso il post-mix
  const outputs = anatomy.attention.output.filter((c) => cardState(byId.get(c.id), layer) !== "off");
  outputs.forEach((c, i) => put(c.id, (A.core.x0 + A.core.x1) / 2, coreBottom - 3 - A.cardH / 2 - i * (A.cardH + 3), A.core.x1 - A.core.x0, A.cardH, cardState(byId.get(c.id), layer)));

  // FFN: router, (latente), griglia degli esperti, (latente), esperto, somma; oppure la FFN densa
  const f = anatomy.ffn;
  const fx0 = A.ffn.x0 + 2;
  const fx1 = A.ffn.x1 - 2;
  let grid = null;
  const dense = f.dense && layer.flags.has(f.dense.when);
  if (dense) {
    put(f.dense.id, (fx0 + fx1) / 2, 0, fx1 - fx0, 12);
  } else {
    const shared = f.shared ?? [];
    const routerW = shared.length ? 22 : fx1 - fx0;
    put(f.router.id, fx0 + routerW / 2, 18, routerW, A.cardH);
    const sharedH = shared.length > 1 ? 5.5 : A.cardH;
    shared.forEach((c, i) => put(c.id, fx1 - (fx1 - fx0 - routerW - 2) / 2, shared.length > 1 ? 20.5 - i * 6.5 : 18, fx1 - fx0 - routerW - 2, sharedH));
    // sotto il router e sotto la pila degli esperti condivisi (Kimi ne ha 2)
    const sharedBottom = shared.length > 1 ? 20.5 - (shared.length - 1) * 6.5 - sharedH / 2 : 18 - A.cardH / 2;
    let top = Math.min(18 - A.cardH / 2, sharedBottom) - 2;
    const latent = f.latent ?? [];
    if (latent[0]) {
      put(latent[0].id, (fx0 + fx1) / 2, top - 2.5, fx1 - fx0, 5);
      top -= 7;
    }
    const bottomReserved = (latent[1] ? 7 : 0) + A.cardH + 2 + A.cardH + 4; // (latente) + esperto + somma
    const areaH = top - (A.ffn.y0 + 2 + bottomReserved);
    const cols = f.grid.cols;
    const count = model.params.moe.routed;
    const gridRows = Math.ceil(count / cols);
    const cell = Math.min((fx1 - fx0) / cols, (areaH - 3) / gridRows);
    const gw = cell * cols;
    const gh = cell * gridRows;
    grid = { x: (fx0 + fx1) / 2, y: top - 3 - gh / 2, w: gw, h: gh, cols, rows: gridRows, cell, count, labelY: top - 1.2 };
    let y = grid.y - gh / 2 - 2;
    if (latent[1]) {
      put(latent[1].id, (fx0 + fx1) / 2, y - 2.5, fx1 - fx0, 5);
      y -= 7;
    }
    put(f.expert.id, (fx0 + fx1) / 2, y - A.cardH / 2, fx1 - fx0, A.cardH);
    y -= A.cardH + 2;
    put(f.sum.id, fx0 + 11, y - A.cardH / 2, 22, A.cardH);
  }

  // ramo laterale (Engram): colonna a sinistra, dall'alto verso il residuo
  let branch = null;
  const br = anatomy.branch;
  if (br && layer.flags.has(br.when)) {
    const w = 24;
    const step = 9;
    br.cards.forEach((c, i) => put(c.id, A.branchX, 33 - i * step, w, 7));
    branch = { x: A.branchX, x0: A.branchX - w / 2 - 2, x1: A.branchX + w / 2 + 2, y0: 33 - (br.cards.length - 1) * step - 5.5, y1: 40.5, label: br.label };
  }

  const minX = (branch ? branch.x0 : -A.halfW) - 2;
  const box = { minX, maxX: A.halfW + 2, minY: A.attn.y0 - 3, maxY: A.titleY + 4 };
  return { cards, rows, nodes, rails, grid, branch, panels: { attention: A.attn, ffn: A.ffn }, box };
}
