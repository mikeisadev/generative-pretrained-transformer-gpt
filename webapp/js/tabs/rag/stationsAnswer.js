/**
 * Stazioni 8-11 della ricerca: domanda ed embedding, punteggi (topK e soglia), prompt
 * strutturato, chiamata all'LLM con ChatClient e risposta in streaming nel frontend React.
 */
import * as THREE from "three";
import { Board, MONO, browserFrame, panel, paragraph, pill, roundRect } from "./board.js";
import { DOC_COLORS, VectorGrid, bigBlock, ease, makeStation, phase } from "./scene3d.js";

const SCORE_BARS = 8;
const BAR_H = 13; // altezza di una barra con score 1

/** Etichetta breve di un chunk: "rimborsi p2·0". */
export function shortId(id) {
  const [doc, rest] = id.split("-p");
  return rest ? `${doc} p${rest.replace(":", "·")}` : id;
}

// ------------------------------------------------------------------ 8 · Ricerca

export function buildSearch(kb) {
  const st = makeStation("search");
  const card = new Board({ width: 22, height: 6.4, ppu: 56 });
  card.mesh.position.set(-13.5, 8.6, 0);
  const bars = new VectorGrid({ dims: kb.embedder.dimensions, size: 9.5 });
  bars.group.position.set(-19.2, -3.6, 0);
  const note = new Board({ width: 11.4, height: 13, ppu: 56 });
  note.mesh.position.set(-8, -3.4, 0);
  // barre dei punteggi (un InstancedMesh) + soglia
  const scoreMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ transparent: true }), SCORE_BARS);
  scoreMesh.userData.noBounds = true;
  const x0 = 4;
  const dx = 2.4;
  const base = -5.6;
  const threshold = new THREE.Mesh(new THREE.PlaneGeometry(SCORE_BARS * dx + 1.5, 0.18), new THREE.MeshBasicMaterial({ color: 0xf87171, transparent: true }));
  const axis = new Board({ width: 26, height: 22.5, ppu: 50 });
  axis.mesh.position.set(x0 + (SCORE_BARS - 1) * dx * 0.5 + 0.9, 0.9, -0.6);
  st.group.add(card.mesh, bars.group, note.mesh, axis.mesh, scoreMesh, threshold);
  const dummy = new THREE.Object3D();
  const colors = { top: new THREE.Color("#22c55e"), above: new THREE.Color("#facc15"), below: new THREE.Color("#64748b") };

  function setQuestion(run, p = 1) {
    card.redraw(`Q${run?.question}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, { title: "Domanda dell'utente", border: "#fde047", accent: "#fde047" });
      paragraph(ctx, run ? `"${run.question}"` : "Scrivi una domanda nel pannello Azioni.", box.x, box.y, box.w, { px: 38, color: "#fef9c3", weight: 600, maxLines: 3 });
    });
    const grow = run ? phase(p, 0.2, 1) : 0;
    bars.set(run && grow > 0 ? run.vector : null, grow);
    note.redraw(`N${run?.question}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, { title: "Embedding della domanda", border: "#c084fc", accent: "#d8b4fe" });
      const text = run
        ? `Stesso EmbeddingModel dei chunk: domanda e documenti devono stare sulla stessa "mappa" (L18 §12.1).\n\n${kb.embedder.dimensions} numeri (simulazione; il modello vero ne dà 1536).`
        : "In attesa della domanda.";
      paragraph(ctx, text, box.x, box.y, box.w, { px: 32, color: "#e2e8f0", maxLines: 12 });
    });
  }

  /**
   * Punteggi dei chunk più simili (ricerca esatta, valori veri), confrontati con la soglia.
   * p fa crescere le barre; con p > 0.6 compaiono i colori del filtro topK/soglia.
   */
  function setScores(run, p = 1) {
    const top = run ? run.exactAll.slice(0, SCORE_BARS) : [];
    const resultIds = new Set(run?.results.map((r) => r.id) ?? []);
    const grow = ease(phase(p, 0, 0.6));
    const judged = p > 0.6;
    for (let i = 0; i < SCORE_BARS; i++) {
      const s = top[i]?.score ?? 0;
      const h = Math.max(0.01, s * BAR_H * grow);
      dummy.position.set(x0 + i * dx, base + h / 2, 0);
      dummy.scale.set(1.5, h, 1.2);
      dummy.updateMatrix();
      scoreMesh.setMatrixAt(i, dummy.matrix);
      const c = !judged ? colors.below : resultIds.has(top[i]?.id) ? colors.top : s >= (run?.similarityThreshold ?? 0.5) ? colors.above : colors.below;
      scoreMesh.setColorAt(i, c);
    }
    scoreMesh.count = top.length;
    scoreMesh.instanceMatrix.needsUpdate = true;
    if (scoreMesh.instanceColor) scoreMesh.instanceColor.needsUpdate = true;
    const thr = run?.similarityThreshold ?? 0.5;
    threshold.position.set(x0 + (SCORE_BARS - 1) * dx * 0.5, base + thr * BAR_H, 0.7);
    threshold.visible = !!run;
    axis.redraw(`A${run?.question}|${judged}|${run?.topK}|${thr}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, { title: "Similarità con la domanda", subtitle: "score = 1 − distanza del coseno (operatore <=> di pgvector)", border: "#334155" });
      if (!run) return;
      // etichette sotto le barre: stesse coordinate della scena (ppu 50, origine al centro della lavagna)
      const ppu = W / 26;
      const toX = (sceneX) => (sceneX - (axis.mesh.position.x - 13)) * ppu;
      const toY = (sceneY) => (11.25 - (sceneY - 0.9)) * ppu;
      ctx.font = `600 26px ${MONO}`;
      ctx.textBaseline = "top";
      top.forEach((t, i) => {
        const x = toX(x0 + i * dx);
        ctx.save();
        ctx.translate(x, toY(base) + 8);
        ctx.rotate(Math.PI / 5);
        ctx.fillStyle = DOC_COLORS[kb.docOf(t.node) % DOC_COLORS.length];
        ctx.fillText(shortId(t.id), 0, 0);
        ctx.restore();
        if (grow > 0) {
          ctx.fillStyle = "#e2e8f0";
          ctx.fillText(t.score.toFixed(2), x - 30, toY(base + t.score * BAR_H) - 32);
        }
      });
      ctx.fillStyle = "#fca5a5";
      ctx.font = `700 28px Inter, sans-serif`;
      ctx.fillText(`soglia ${thr}`, toX(x0 + SCORE_BARS * dx - 2.2), toY(base + thr * BAR_H) - 34);
      if (judged) {
        const y = H - 58;
        let x = 18;
        x += pill(ctx, `restituiti: ${run.results.length} (topK ${run.topK})`, x, y, { px: 26, bg: "#22c55e" }) + 12;
        x += pill(ctx, "oltre topK", x, y, { px: 26, bg: "#facc15" }) + 12;
        pill(ctx, "sotto soglia", x, y, { px: 26, bg: "#64748b", color: "#f8fafc" });
      }
    });
  }

  setQuestion(null);
  setScores(null, 0);
  return { ...st, setQuestion, setScores };
}

// ------------------------------------------------------------------ 9 · Prompt

const ROLE_COLORS = { tools: "#94a3b8", system: "#c084fc", question: "#fde047", "documents-header": "#7dd3fc", document: "#38bdf8" };

export function buildPrompt() {
  const st = makeStation("prompt");
  const board = new Board({ width: 37, height: 24, ppu: 50 });
  board.mesh.position.set(0, 0.3, 0);
  st.group.add(board.mesh);

  /** Mostra le prime `count` parti del prompt della ricerca `run`. */
  function set(run, count = 99) {
    board.redraw(`P${run?.question}|${count}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, {
        title: "Prompt strutturato (L18): ordine pensato per il prompt caching",
        subtitle: "strumenti e system prompt restano uguali (prefisso in cache); domanda e documenti cambiano a ogni chiamata",
        border: "#c084fc",
        accent: "#d8b4fe",
      });
      if (!run) return;
      const parts = run.prompt.parts.slice(0, count);
      const free = H - box.y - box.pad - 60;
      const slot = Math.min(170, free / Math.max(6, run.prompt.parts.length));
      let y = box.y;
      for (const part of parts) {
        const color = part.metadata?.flagged ? "#fb923c" : ROLE_COLORS[part.role] ?? "#94a3b8";
        roundRect(ctx, box.x, y, box.w, slot - 8, 10);
        ctx.fillStyle = "rgba(30,41,59,0.75)";
        ctx.fill();
        ctx.fillStyle = color;
        ctx.fillRect(box.x, y, 8, slot - 8);
        ctx.font = `700 32px Inter, sans-serif`;
        ctx.textBaseline = "top";
        ctx.fillText(`${part.title}${part.cacheable ? " · in cache" : ""}${part.metadata?.flagged ? " · SEGNALATO in pulizia" : ""}`, box.x + 20, y + 6, box.w - 220);
        ctx.font = `600 28px ${MONO}`;
        ctx.fillStyle = "#94a3b8";
        ctx.textAlign = "right";
        ctx.fillText(`${part.tokens} token`, box.x + box.w - 12, y + 10);
        ctx.textAlign = "left";
        const excerpt = part.content.replace(/\s+/g, " ").trim().slice(0, 160);
        paragraph(ctx, excerpt, box.x + 20, y + 46, box.w - 36, { px: 26, color: "#cbd5e1", font: MONO, maxLines: Math.max(1, Math.floor((slot - 56) / 34)) });
        y += slot;
      }
      if (count >= run.prompt.parts.length) {
        pill(ctx, `Totale stimato: ${run.prompt.tokens.total} token in un'unica chiamata`, box.x, H - box.pad - 48, { px: 30, bg: "#c084fc" });
      }
    });
  }
  set(null);
  return { ...st, set };
}

// ------------------------------------------------------------------ 10 · LLM

const TOKEN_POOL = 14;

export function buildLLM() {
  const st = makeStation("llm");
  const stack = new THREE.Group();
  const slabMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, emissive: 0x3b0764, emissiveIntensity: 0.6, transparent: true, opacity: 0.95 });
  for (let i = 0; i < 4; i++) {
    const slab = new THREE.Mesh(new THREE.BoxGeometry(11, 1.3, 6), slabMat);
    slab.position.y = -4 + i * 2;
    stack.add(slab);
  }
  stack.position.set(2, -1, 0);
  const title = bigBlock({ title: "LLM", subtitle: "via API (OpenAI, Claude, …)", text: "Un transformer come il nostro GPT, molto più grande: genera un token alla volta.", color: "#c084fc" }, { w: 16, h: 7.4, S: 2.4 });
  title.position.set(2, 7.6, 0);
  const call = new Board({ width: 28, height: 3.4, ppu: 50 });
  call.mesh.position.set(0, -10.3, 0);
  call.redraw("c", (ctx, W, H) => {
    ctx.fillStyle = "rgba(15,23,42,0.92)";
    ctx.fillRect(0, 0, W, H);
    paragraph(ctx, "chatClient.prompt().user(question).advisors(…).stream().content()  →  Flux<String>", 14, 12, W - 28, { px: 27, color: "#bae6fd", font: MONO, maxLines: 2 });
  });
  const tokens = new THREE.InstancedMesh(new THREE.BoxGeometry(0.8, 0.8, 0.8), new THREE.MeshBasicMaterial({ color: 0xfde047, transparent: true }), TOKEN_POOL);
  tokens.userData.noBounds = true;
  const inbound = new THREE.Mesh(new THREE.BoxGeometry(2.4, 3, 0.3), new THREE.MeshBasicMaterial({ color: 0xc084fc, transparent: true, opacity: 0.85 }));
  st.group.add(stack, title, call.mesh, tokens, inbound);
  const dummy = new THREE.Object3D();

  /**
   * @param {"idle"|"prompt"|"stream"} mode prompt: il prompt entra nel modello; stream: escono token
   * @param {number} p avanzamento del passo
   */
  function set(mode, p = 0) {
    inbound.visible = mode === "prompt";
    inbound.position.set(16 - 14 * ease(p), -1, 0.5);
    inbound.scale.setScalar(1 - 0.6 * ease(p));
    tokens.visible = mode === "stream";
    slabMat.emissiveIntensity = mode === "idle" ? 0.4 : 0.6 + 0.5 * Math.abs(Math.sin(p * Math.PI * 10));
    if (mode !== "stream") return;
    for (let i = 0; i < TOKEN_POOL; i++) {
      const t = (p * 5 + i / TOKEN_POOL) % 1; // flusso continuo verso il frontend (a sinistra)
      dummy.position.set(-3 - t * 22, -1 + Math.sin(t * Math.PI) * 1.5, 0.5);
      dummy.scale.setScalar(t < 0.08 ? t / 0.08 : 1);
      dummy.updateMatrix();
      tokens.setMatrixAt(i, dummy.matrix);
    }
    tokens.instanceMatrix.needsUpdate = true;
  }
  set("idle");
  return { ...st, set };
}

// ------------------------------------------------------------------ 11 · Risposta

export function buildAnswer() {
  const st = makeStation("answer");
  const board = new Board({ width: 38, height: 24, ppu: 50 });
  board.mesh.position.set(0, 0.3, 0);
  st.group.add(board.mesh);

  /**
   * @param {object|null} run ricerca
   * @param {{tokens: string[], skipped: string[]}|null} answer risposta composta
   * @param {number} shown token già arrivati (streaming); -1 = solo la domanda
   */
  function set(run, answer, shown = -1) {
    board.redraw(`A${run?.question}|${shown}`, (ctx, W, H) => {
      const box = browserFrame(ctx, W, H, { title: "Chat con l'assistente" });
      if (!run) {
        paragraph(ctx, "La chat è vuota: avvia la ricerca dal pannello Azioni.", box.x, box.y, box.w, { px: 28, color: "#64748b" });
        return;
      }
      // domanda (a destra)
      ctx.font = `500 34px Inter, sans-serif`;
      const qw = Math.min(box.w * 0.8, ctx.measureText(run.question).width + 44);
      roundRect(ctx, box.x + box.w - qw, box.y, qw, 64, 18);
      ctx.fillStyle = "#0369a1";
      ctx.fill();
      paragraph(ctx, run.question, box.x + box.w - qw + 22, box.y + 13, qw - 34, { px: 34, color: "#f0f9ff", maxLines: 1 });
      let y = box.y + 86;
      if (shown < 0) {
        paragraph(ctx, "POST /api/chat/rag/stream: in attesa della risposta…", box.x, y, box.w, { px: 30, color: "#64748b", font: MONO });
        return;
      }
      // prima le fonti (ChatEvent.Sources), poi i token (ChatEvent.Token)
      const sources = [...new Set(run.results.map((r) => `${r.metadata.original_filename} p.${r.metadata.page_number}`))];
      ctx.font = `600 28px Inter, sans-serif`;
      ctx.fillStyle = "#94a3b8";
      ctx.fillText("Fonti:", box.x, y + 8);
      let x = box.x + 96;
      for (const s of sources.length ? sources : ["nessuna"]) {
        const wpx = pill(ctx, s, x, y, { px: 26, bg: "#1e293b", color: "#7dd3fc" });
        x += wpx + 12;
        if (x > box.x + box.w - 380) break;
      }
      y += 68;
      const text = answer.tokens.slice(0, Math.max(0, shown)).join("") + (shown < answer.tokens.length ? "▍" : "");
      roundRect(ctx, box.x, y, box.w * 0.94, H - y - 150, 18);
      ctx.fillStyle = "#1e293b";
      ctx.fill();
      paragraph(ctx, text, box.x + 22, y + 18, box.w * 0.94 - 44, { px: 33, color: "#f1f5f9", maxLines: 9 });
      const last = answer.tokens.slice(Math.max(0, shown - 3), Math.max(0, shown)).map((t) => `data:${JSON.stringify(t)}`).join("  ");
      paragraph(ctx, `SSE ${last}`, box.x, H - 140, box.w, { px: 26, color: "#64748b", font: MONO, maxLines: 1 });
      paragraph(
        ctx,
        `Risposta simulata: frasi estratte dai chunk recuperati, con la fonte.${answer.skipped.length ? " Una frase con istruzioni sospette è stata trattata come dato e ignorata." : ""}`,
        box.x,
        H - 100,
        box.w,
        { px: 27, color: "#fbbf24", maxLines: 2 },
      );
    });
  }
  set(null, null);
  return { ...st, set };
}
