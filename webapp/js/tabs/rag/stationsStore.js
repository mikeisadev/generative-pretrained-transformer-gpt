/**
 * Stazioni 4-6: Chunking (TokenTextSplitter), Embedding (EmbeddingModel) e inserimento nella
 * tabella vector_store di pgvector.
 */
import * as THREE from "three";
import { fmt } from "../../core/dom.js";
import { tokenize } from "../../shared/rag/index.js";
import { Board, CHUNK_RGB, MONO, panel, paragraph, pill, richText, roundRect } from "./board.js";
import { VectorGrid, arcLerp, bigBlock, ease, makeStation, phase } from "./scene3d.js";

/** Carta di un chunk: anteprima del testo e conteggio dei token. */
function drawChunkCard(ctx, W, H, record, k) {
  const rgb = CHUNK_RGB[k % CHUNK_RGB.length];
  roundRect(ctx, 3, 3, W - 6, H - 6, 18);
  ctx.fillStyle = "rgba(15,23,42,0.97)";
  ctx.fill();
  ctx.lineWidth = 5;
  ctx.strokeStyle = `rgb(${rgb})`;
  ctx.stroke();
  const m = record.entry.metadata;
  ctx.font = `700 44px Inter, sans-serif`;
  ctx.fillStyle = `rgb(${rgb})`;
  ctx.textBaseline = "top";
  ctx.fillText(`pag. ${m.page_number} · chunk ${m.chunk_index} · ${record.tokens} token`, 20, 14, W - 40);
  ctx.font = `600 30px ${MONO}`;
  ctx.fillStyle = m.flagged ? "#fdba74" : "#94a3b8";
  ctx.fillText(`${record.entry.text.length} caratteri${m.flagged ? " · SEGNALATO in pulizia" : ""}`, 20, 68, W - 40);
  richText(ctx, record.entry.text, { x: 20, y: 112, w: W - 40, h: H - 126 }, { cols: 40, color: "#cbd5e1", lineGap: 1.25 });
}

// ------------------------------------------------------------------ 4 · Chunking

export function buildChunk(kb) {
  const st = makeStation("chunk");
  const { featured, inserts, config } = kb;
  const chunkSize = config.splitterPreset.options.chunkSize;
  const records = featured.chunks.map((c) => inserts.find((r) => r.entry.id === c.id));
  // pagina in alto, carte dei chunk in una griglia 2 × 2 sotto
  const board = new Board({ width: 25, height: 16.4, ppu: 60 });
  board.mesh.position.set(0, 6.6, 0);
  st.group.add(board.mesh);
  const tmp = new THREE.Vector3();
  // una carta per chunk del documento in evidenza; arrivano impilate a destra
  const cards = records.map((record, i) => {
    const card = new Board({ width: 12.2, height: 5.6, ppu: 56 });
    card.redraw("c", (ctx, W, H) => drawChunkCard(ctx, W, H, record, record.entry.metadata.chunk_index));
    card.slot = new THREE.Vector3(i % 2 ? 6.3 : -6.3, -5.2 - Math.floor(i / 2) * 6, 0.3);
    card.page = record.entry.metadata.page_number;
    st.group.add(card.mesh);
    return card;
  });
  const from = new THREE.Vector3(0, 5, 0.6);

  function drawPage(pageIndex) {
    const page = featured.pages[pageIndex];
    const { trace, totalTokens } = featured.traces[pageIndex];
    const tokens = tokenize(page.text);
    board.redraw(`P${pageIndex}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, {
        title: `Pagina ${page.metadata.page_number} · ${totalTokens} token`,
        subtitle: `finestre di ${chunkSize} token · taglio all'ultimo . ? ! o a capo oltre ${config.splitterPreset.options.minChunkSizeChars} caratteri`,
        border: "#475569",
      });
      const ranges = [];
      const markers = [];
      trace.forEach((t, k) => {
        const kept = t.kept ? k : -1;
        tokens.forEach((tok, j) => {
          if (j < t.startToken || j >= t.startToken + t.tokenCount) return;
          ranges.push({ start: tok.start, end: tok.end, style: kept < 0 ? "dim" : `c${k % CHUNK_RGB.length}${j % 2 ? "o" : "e"}` });
        });
        if (t.cutAtPunctuation) markers.push({ index: t.startChar + t.punctuationIndex + 1, style: "cut" });
      });
      const res = richText(ctx, page.text, { x: box.x, y: box.y, w: box.w, h: H - box.y - 30 }, { ranges, markers, cols: 66, lineGap: 1.3, fit: true });
      return res;
    });
  }

  /**
   * @param {number} pageIndex pagina in lavorazione (0…n-1); n = tutte finite
   * @param {number} p avanzamento: le carte della pagina si staccano e volano nello stack
   */
  function set(pageIndex, p = 0) {
    const n = featured.pages.length;
    drawPage(Math.min(pageIndex, n - 1));
    const currentPage = featured.pages[Math.min(pageIndex, n - 1)].metadata.page_number;
    let order = 0;
    cards.forEach((card) => {
      const done = card.page < currentPage || pageIndex >= n;
      const active = !done && card.page === currentPage;
      card.mesh.visible = done || (active && p > 0);
      if (done) card.mesh.position.copy(card.slot);
      else if (active) {
        const t = ease(phase(p, 0.15 + order * 0.25, 0.65 + order * 0.25));
        card.mesh.position.copy(arcLerp(tmp, from, card.slot, t, 3));
        card.mesh.rotation.z = (1 - t) * -0.25;
        order++;
      }
      if (done) card.mesh.rotation.z = 0;
    });
  }
  set(0, 0);
  return { ...st, set, cards, records };
}

// ------------------------------------------------------------------ 5 · Embedding

export function buildEmbed(kb) {
  const st = makeStation("embed");
  const { featured, inserts, embedder } = kb;
  const records = featured.chunks.map((c) => inserts.find((r) => r.entry.id === c.id));
  const card = new Board({ width: 12.2, height: 5.6, ppu: 56 });
  const model = bigBlock({ title: "EmbeddingModel", subtitle: "text-embedding-3-small", text: "Qui simulato nel browser: niente rete neurale.", color: "#c084fc" }, { w: 12, h: 6.4, S: 2.2 });
  model.position.set(-8, 1, 0);
  const grid = new VectorGrid({ dims: embedder.dimensions, size: 12.5 });
  grid.group.position.set(7.4, 3, 0);
  const numbers = new Board({ width: 29, height: 6.4, ppu: 56 });
  numbers.mesh.position.set(0, -9.6, 0);
  const arrow = new Board({ width: 3, height: 1.6, ppu: 60 });
  arrow.mesh.position.set(-1.1, 1, 0.2);
  arrow.redraw("a", (ctx, W, H) => {
    ctx.fillStyle = "#c084fc";
    ctx.font = `700 ${H * 0.8}px Inter, sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillText("→", W * 0.2, H / 2);
  });
  st.group.add(card.mesh, model, grid.group, numbers.mesh, arrow.mesh);
  const cardFrom = new THREE.Vector3(-8, 9.2, 0.4);
  const cardTo = new THREE.Vector3(-8, 1, 0.6);
  const tmp = new THREE.Vector3();

  function drawNumbers(record, k) {
    numbers.redraw(`N${k}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, { title: `Il vettore: ${embedder.dimensions} numeri (griglia 16 × 16)`, border: "#c084fc", accent: "#d8b4fe" });
      const v = record.vector;
      const shown = [...v.slice(0, 8)].map((x) => x.toFixed(3)).join(", ");
      paragraph(ctx, `[${shown}, …]`, box.x, box.y, box.w, { px: 30, color: "#f5d0fe", font: MONO, maxLines: 1 });
      paragraph(ctx, "Simulazione didattica, lunghezza 1. Il vero text-embedding-3-small dà 1536 numeri: per questo la L17 imposta dimensions=1536.", box.x, box.y + 46, box.w, { px: 28, color: "#cbd5e1", maxLines: 2 });
    });
  }

  /** chunk = indice tra i chunk del documento in evidenza; p: la carta entra nel modello, poi la griglia si accende. */
  function set(chunk, p = 0) {
    const record = records[Math.min(chunk, records.length - 1)];
    card.redraw(`C${chunk}`, (ctx, W, H) => drawChunkCard(ctx, W, H, record, record.entry.metadata.chunk_index));
    const t = ease(phase(p, 0, 0.4));
    card.mesh.position.copy(arcLerp(tmp, cardFrom, cardTo, t, 1.5));
    card.mesh.scale.setScalar(1 - 0.55 * t);
    card.mesh.visible = p < 0.42;
    const grow = phase(p, 0.4, 1);
    grid.set(grow > 0 ? record.vector : null, grow);
    numbers.mesh.visible = grow > 0.3;
    drawNumbers(record, chunk);
  }
  set(0, 0);
  return { ...st, set, records };
}

// ------------------------------------------------------------------ 6 · Tabella vector_store

const VISIBLE_ROWS = 10;

export function buildTable(kb) {
  const st = makeStation("table");
  const { inserts, embedder } = kb;
  const board = new Board({ width: 34, height: 23, ppu: 54 });
  board.mesh.position.set(0, 0.4, 0);
  st.group.add(board.mesh);

  /** n = righe inserite (in ordine di inserimento); la più recente è evidenziata. */
  function set(n) {
    board.redraw(`T${n}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, {
        title: `vector_store · ${fmt(n)} ${n === 1 ? "riga" : "righe"}`,
        subtitle: "INSERT INTO vector_store (id, content, metadata, embedding) VALUES (?, ?, ?::jsonb, ?) …",
        border: "#38bdf8",
      });
      const cols = [
        { name: "id", w: 0.2 },
        { name: "content (text)", w: 0.37 },
        { name: "metadata (json)", w: 0.25 },
        { name: `embedding vector(${embedder.dimensions})`, w: 0.18 },
      ];
      const rowH = 70;
      let y = box.y + 6;
      ctx.fillStyle = "#1e293b";
      ctx.fillRect(box.x, y, box.w, rowH);
      ctx.font = `700 32px Inter, sans-serif`;
      ctx.fillStyle = "#7dd3fc";
      ctx.textBaseline = "middle";
      let x = box.x + 10;
      for (const c of cols) {
        ctx.fillText(c.name, x, y + rowH / 2, c.w * box.w - 16);
        x += c.w * box.w;
      }
      y += rowH;
      if (n === 0) {
        paragraph(ctx, "Tabella vuota. Spring AI la crea all'avvio (initialize-schema=true) insieme all'indice HNSW:", box.x + 10, y + 30, box.w - 20, { px: 36, color: "#94a3b8" });
        paragraph(ctx, "CREATE INDEX ON vector_store USING HNSW (embedding vector_cosine_ops)", box.x + 10, y + 150, box.w - 20, { px: 32, color: "#bae6fd", font: MONO });
      }
      const first = Math.max(0, n - VISIBLE_ROWS);
      ctx.font = `400 28px ${MONO}`;
      for (let i = first; i < n; i++) {
        const r = inserts[i];
        const newest = i === n - 1;
        ctx.fillStyle = newest ? "rgba(56,189,248,0.18)" : i % 2 ? "rgba(30,41,59,0.6)" : "rgba(15,23,42,0.6)";
        ctx.fillRect(box.x, y, box.w, rowH);
        const m = r.entry.metadata;
        const cells = [
          r.entry.id,
          r.entry.text.replace(/\s+/g, " "),
          `{"original_filename":"${m.original_filename}","page_number":${m.page_number}${m.flagged ? ',"flagged":true' : ""}}`,
          `[${r.vector[0].toFixed(2)}, ${r.vector[1].toFixed(2)}, …]`,
        ];
        ctx.fillStyle = newest ? "#f0f9ff" : m.flagged ? "#fdba74" : "#cbd5e1";
        x = box.x + 10;
        cells.forEach((cell, j) => {
          ctx.fillText(clip(ctx, cell, cols[j].w * box.w - 18), x, y + rowH / 2);
          x += cols[j].w * box.w;
        });
        y += rowH;
      }
      if (first > 0) {
        ctx.fillStyle = "#64748b";
        ctx.fillText(`… e altre ${first} righe più vecchie`, box.x + 10, y + 30);
      }
      ctx.textBaseline = "top";
      if (n > 0) pill(ctx, `SELECT COUNT(*) FROM vector_store; → ${n}`, box.x, H - box.pad - 54, { px: 32, bg: "#38bdf8" });
    });
  }
  set(0);
  return { ...st, set };
}

function clip(ctx, text, w) {
  if (ctx.measureText(text).width <= w) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= w) lo = mid;
    else hi = mid - 1;
  }
  return `${text.slice(0, lo)}…`;
}
