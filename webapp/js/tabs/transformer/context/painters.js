/**
 * Che cosa disegna ogni scheda del livello Dettaglio, dato lo scenario (pipeline.js) e lo stato
 * dell'animazione (`st`, vedi scene.js → FULL_STATE). Funzioni pure sul canvas: nessun oggetto 3D.
 */
import { tokenize } from "../../../shared/rag/index.js";
import { INK, compact } from "./cards.js";
import { PRODUCTION_MODEL, ROLE_STYLE, contextSegments } from "./pipeline.js";

const it = (n, d = 0) => Number(n).toLocaleString("it-IT", { maximumFractionDigits: d, minimumFractionDigits: d });
const shown = (t) => t.replace(/\n/g, "⏎").replace(/ /g, "·");
const unused = (pen, why) => pen.text(why, { size: 0.6, color: INK.dim });

/** Fonti raggruppate per file: "rimborsi-spese-trasferte.pdf p. 1, 2, 3". */
function sourcesLine(sources) {
  const byFile = new Map();
  for (const x of sources) byFile.set(x.filename, [...(byFile.get(x.filename) ?? []), x.page]);
  return [...byFile].map(([f, pages]) => `${f} p. ${pages.join(", ")}`).join(" · ");
}

/**
 * Ogni painter riceve la penna (cards.js), lo scenario, lo stato dell'animazione e `info`
 * (numeri del nostro GPT da model.json: parametri, block_size, vocabolario).
 */
export const PAINTERS = {
  react(pen, s, st) {
    pen.header("Chat del gestionale", "frontend React");
    pen.text("Tu", { size: 0.48, color: INK.muted, weight: 600, gap: 0.05 });
    pen.text(s.question, { size: 0.6, color: "#bae6fd", maxLines: 3 });
    pen.text("Assistente", { size: 0.48, color: INK.muted, weight: 600, gap: 0.05 });
    const n = Math.min(st.streamed, s.answer.tokens.length);
    if (n <= 0) pen.text(st.streamed < 0 ? "" : "in attesa della risposta…", { size: 0.55, color: INK.dim });
    else pen.text(s.answer.tokens.slice(0, n).join(""), { size: 0.55, maxLines: 8, limit: pen.h - 1.5 });
    if (n >= s.answer.tokens.length && n > 0 && s.sources.length) {
      pen.y = pen.h - 1.35;
      pen.text(`Fonti: ${sourcesLine(s.sources)}`, { size: 0.5, color: INK.ok, maxLines: 1, weight: 600 });
    }
  },

  guard(pen, s, st) {
    pen.header("InputGuard", "prima del modello");
    s.guard.checks.slice(0, st.checks).forEach((c) => pen.check(c.label, c.detail, c.ok));
    if (st.checks >= s.guard.checks.length) {
      pen.y += 0.2;
      if (s.blocked) pen.band(`Bloccata: «${s.guard.text}»`, INK.bad, { size: 0.65, height: 1.2 });
      else {
        pen.band("Passa: la domanda pulita prosegue", INK.ok, { size: 0.65, height: 1.2 });
        pen.text("Nessuna chiamata al modello in questo passo: è normale codice Java, veloce e gratuito.", { size: 0.52, color: INK.muted, maxLines: 2 });
      }
    }
  },

  embed(pen, s) {
    pen.header("Embedding della domanda", s.embedding ? `${it(s.embedding.dimensions)} numeri` : "");
    if (!s.embedding) return unused(pen, "Domanda bloccata dal guardrail: nessun embedding.");
    const feats = s.embedding.features.slice(0, 5).map((f) => `${f.value} ${it(f.weight, 1)}`).join(" · ");
    pen.text(`Radici con più peso: ${feats}`, { size: 0.5, color: INK.muted, maxLines: 2 });
    pen.footer("Ogni barra è un numero del vettore (arancione > 0, blu < 0). Simulazione didattica di text-embedding-3-small (1536 numeri).");
  },

  vectordb(pen, s, st) {
    pen.header("pgvector · indice HNSW", "distanza del coseno");
    if (!s.search) return unused(pen, "Nessuna ricerca: domanda bloccata.");
    const v = Math.min(st.visited, s.search.visitOrder.length);
    pen.text(`Nodi visitati: ${v} su ${s.search.total}`, { size: 0.55, color: INK.muted });
    pen.footer("Pallino = chunk (posizione: PCA del vettore). Viola: la domanda. Azzurro: visitato. Verde: recuperato.");
  },

  chunks(pen, s, st) {
    pen.header("Chunk recuperati", `topK ${s.topK} · soglia ${it(s.threshold, 1)}`);
    if (!s.search) return unused(pen, "Nessun chunk: domanda bloccata.");
    const rows = s.search.ranking.slice(0, 7);
    const passed = new Set(s.search.results.map((r) => r.id));
    const { ctx } = pen;
    rows.slice(0, st.rows).forEach((r, i) => {
      const ok = passed.has(r.id);
      const y = pen.y + i * 1.32;
      pen.font(0.55, { weight: ok ? 700 : 400 });
      ctx.fillStyle = ok ? INK.text : INK.dim;
      ctx.fillText(`${i + 1}. ${r.metadata.original_filename.replace(".pdf", "")} p.${r.metadata.page_number}`, pen.pad, y);
      pen.bar(pen.pad + 11.2, y + 0.14, 4.6, 0.36, r.score, 0.8, ok ? INK.ok : "#475569", s.threshold);
      pen.font(0.52, { mono: true });
      ctx.fillStyle = ok ? INK.ok : INK.dim;
      ctx.textAlign = "right";
      ctx.fillText(it(r.score, 3), pen.w - pen.pad, y + 0.04);
      ctx.textAlign = "left";
    });
    pen.footer(s.search.results.length ? `Verdi: score > ${it(s.threshold, 1)}, entrano nel contesto con file e pagina. Tacca bianca = soglia.` : "Nessun chunk supera la soglia: nel contesto andrà «Nessun documento pertinente trovato».");
  },

  prompt(pen, s, st) {
    pen.header(st.call >= 2 ? "Contesto della 2ª chiamata" : "Contesto inviato all'LLM", s.prompt ? `${it(st.call >= 2 && s.calls[1] ? s.calls[1].input : s.prompt.tokens.total)} token (stima)` : "");
    if (!s.prompt) return unused(pen, "Nessun contesto: la domanda è stata bloccata dal guardrail e non arriva all'LLM.");
    const segs = contextSegments(s, { includeTool: true });
    const previews = {
      tools: () => s.prompt.tools.map((t) => `${t.name}(${Object.keys(t.inputSchema.properties).join(", ")})`).join("  "),
      system: (seg) => compact(seg.parts[0].content),
      history: (seg) => seg.parts.map((p) => `${p.messageRole === "user" ? "utente" : "assistente"}: ${compact(p.content)}`).join("\n"),
      question: (seg) => compact(seg.parts[0].content.replace("DOMANDA DELL'UTENTE:", "")),
      document: (seg) => seg.parts.filter((p) => p.role === "document").map((p) => `[${p.title}] ${compact(p.content.replace(/<\/?documento[^>]*>/g, ""))}`).join("\n"),
      "tool-call": () => compact(s.toolCall.json),
      "tool-result": () => compact(s.toolResult.json),
    };
    const lines = { tools: 2, system: 4, history: 2, question: 2, document: 9, "tool-call": 2, "tool-result": 4 };
    segs.forEach((seg, i) => {
      const visible = i < st.segments;
      pen.part({ title: seg.label, tokens: `${it(seg.tokens)} tok`, preview: visible ? previews[seg.role](seg) : "", color: seg.color, lines: lines[seg.role], dim: !visible });
    });
    // barra di occupazione, come nella L18 §2.4
    const total = segs.reduce((a, g) => a + g.tokens, 0);
    const y = pen.h - 1.25;
    let x = pen.pad;
    segs.forEach((seg, i) => {
      const w = (pen.width * seg.tokens) / total;
      pen.ctx.globalAlpha = i < st.segments ? 1 : 0.15;
      pen.ctx.fillStyle = seg.color;
      pen.ctx.fillRect(x, y, Math.max(0.05, w - 0.05), 0.5);
      x += w;
    });
    pen.ctx.globalAlpha = 1;
  },

  tokenize(pen, s, st, info) {
    pen.header("Tokenizzazione", "testo → token");
    if (!s.prompt) return unused(pen, "Nessun token: domanda bloccata.");
    const pieces = tokenize(s.question).slice(0, 20);
    let x = pen.pad;
    for (const p of pieces) {
      const w = pen.chip(x, pen.y, shown(p.text), ROLE_STYLE.question.color, { size: 0.68 });
      x += w + 0.16;
      if (x > pen.w - 3.5) break;
    }
    pen.y += 1.35;
    const n = st.call >= 2 && s.calls[1] ? s.calls[1].input : s.prompt.tokens.total;
    pen.text(`≈ ${it(n)} token BPE per ${PRODUCTION_MODEL}. Il nostro GPT (1 token = 1 carattere) ne legge al massimo ${info.block}: qui ne servirebbero ${it(s.prompt.text.length)}.`, { size: 0.6, color: INK.muted, maxLines: 3 });
  },

  llm(pen, s, st, info) {
    pen.header("LLM", "la pila →");
    if (!s.calls.length) return unused(pen, "Nessuna chiamata: il guardrail ha risposto da solo, senza spendere token.");
    pen.text(`Nel corso: ${PRODUCTION_MODEL}. A destra: il nostro GPT, stessa struttura (transformer decoder) ma addestrato solo su Shakespeare.`, { size: 0.55, maxLines: 4 });
    if (info) pen.text(`Il nostro GPT: ${info.layers} blocchi, ${info.params} parametri, vocabolario di ${info.vocab} caratteri, finestra di ${info.block} token.`, { size: 0.5, color: INK.muted, maxLines: 3 });
    pen.y += 0.2;
    s.calls.forEach((c, i) => {
      const active = st.call === i + 1;
      pen.text(`${c.label}: ${it(c.input)} token in ingresso → ${c.produces} (${it(c.output)} token)`, { size: 0.52, color: active ? "#67e8f9" : st.call > i + 1 ? INK.text : INK.dim, weight: active ? 700 : 400, maxLines: 2 });
    });
    pen.footer("Leggere il contesto non cambia i pesi: il modello non impara dai documenti.");
  },

  toolcall(pen, s, st) {
    pen.header("Richiesta di strumento", "1ª chiamata");
    if (!s.toolCall) return unused(pen, "Per questa domanda non serve nessuno strumento: l'LLM risponde subito.");
    if (!st.tool) return unused(pen, "In attesa della risposta dell'LLM…");
    pen.text(s.toolCall.json, { size: 0.55, mono: true, color: "#fecdd3", maxLines: 8 });
    pen.footer("Formato semplificato (ogni provider ha il suo). Scelta simulata.");
  },

  java(pen, s, st) {
    pen.header("EmployeeTools.java", "backend Spring");
    if (!s.toolCall) return unused(pen, "Nessuno strumento da eseguire.");
    if (st.tool < 2) return unused(pen, "Il metodo non è ancora stato chiamato.");
    pen.text(`findEmployeesByDepartment("${s.toolCall.department}", toolContext)`, { size: 0.55, mono: true, color: "#fecdd3", maxLines: 2 });
    pen.check("Utente dal login", "ToolContext", true);
    pen.check("Permessi", "canAccessDepartment", true);
    pen.check("Query JPA", "findTop50ByDepartment…", true);
    pen.text(`→ ${s.toolResult.rows.length} EmployeeSummary: solo nome, ruolo, email`, { size: 0.52, color: INK.muted, maxLines: 2 });
  },

  toolresult(pen, s, st) {
    pen.header("Risultato (JSON)", s.toolResult ? `${it(s.toolResult.tokens)} token` : "");
    if (!s.toolResult) return unused(pen, "Nessun risultato: nessuno strumento chiamato.");
    if (st.tool < 3) return unused(pen, "In attesa del metodo Java…");
    pen.text(compact(s.toolResult.json), { size: 0.5, mono: true, color: "#fbcfe8", maxLines: 6 });
    pen.footer("Dati d'esempio inventati (AZIENDA ESEMPIO).");
  },

  stream(pen, s, st) {
    pen.header("Server-Sent Events → React", `${it(s.sse.length)} eventi`);
    const offset = s.sse[0]?.event === "sources" ? 1 : 0;
    const n = Math.max(0, Math.min(Math.max(st.streamed, 0) + offset, s.sse.length));
    if (n === 0) return unused(pen, "In attesa: prima l'evento sources, poi un evento token per ogni pezzo di risposta.");
    // a sinistra gli ultimi frame del flusso (come li scrive Spring), a destra il testo ricostruito da React
    const top = pen.y;
    const frames = s.sse.slice(0, n);
    const shownFrames = frames.length > 5 ? [frames[0], ...frames.slice(-4)] : frames;
    pen.font(0.5, { mono: true });
    shownFrames.forEach((f, i) => {
      const text = f.event === "sources" ? `event:sources  data:[${s.sources.length} fonti]` : `event:token    data:${shown(f.data)}`;
      pen.chip(pen.pad, top + i * 0.95, text, f.event === "sources" ? "#34d399" : "#38bdf8", { size: 0.48 });
    });
    const col = 10.2;
    pen.y = top;
    pen.text(s.answer.tokens.slice(0, Math.max(0, st.streamed)).join(""), { size: 0.52, x: col, width: pen.w - col - 2 * pen.pad, maxLines: 6, limit: pen.h - 1.3 });
    pen.footer(s.answer.kind === "example" ? "Risposta d'esempio scritta da noi (stile di un LLM di produzione)." : s.answer.kind === "extractive" ? "Risposta simulata: frase dei chunk più simile alla domanda." : "");
  },
};
