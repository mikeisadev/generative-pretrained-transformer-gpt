/**
 * Stazioni 1-3 dell'indicizzazione: Upload (React → DocumentController), Estrazione del testo
 * per pagina (PagePdfDocumentReader) e Pulizia con controllo di sicurezza.
 *
 * Ogni stazione espone dei "set" che portano la scena in uno stato preciso (idempotenti): le
 * animazioni li chiamano dentro enter/update dei passi.
 */
import * as THREE from "three";
import { Board, browserFrame, panel, paragraph, pill, richText, MONO } from "./board.js";
import { arcLerp, bigBlock, ease, makeStation, phase } from "./scene3d.js";

// ------------------------------------------------------------------ miniature di un PDF

/** Disegna una pagina PDF "sporca" in miniatura (intestazione, righe di testo, numero di pagina). */
function drawPdfThumb(ctx, W, H, { title, text, color = "#fb923c" }) {
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, W, H * 0.06);
  ctx.fillStyle = "#0f172a";
  ctx.font = `700 ${Math.round(W * 0.075)}px Inter, sans-serif`;
  ctx.textBaseline = "top";
  ctx.fillText("PDF", W * 0.07, H * 0.09);
  paragraph(ctx, title, W * 0.07, H * 0.17, W * 0.86, { px: Math.round(W * 0.06), color: "#1e293b", weight: 600, maxLines: 2 });
  ctx.font = `400 ${Math.round(W * 0.034)}px ${MONO}`;
  ctx.fillStyle = "#475569";
  const lines = text.split("\n").slice(0, 26);
  lines.forEach((l, i) => ctx.fillText(l.slice(0, 44), W * 0.07, H * 0.3 + i * W * 0.045));
}

function pdfSheet(doc, { width = 3.6, color }) {
  const board = new Board({ width, height: width * 1.36, ppu: 80 });
  board.redraw("pdf", (ctx, W, H) => drawPdfThumb(ctx, W, H, { title: doc.document.title, text: doc.document.rawPages[0].text, color }));
  return board;
}

// ------------------------------------------------------------------ 1 · Upload

export function buildUpload(kb) {
  const st = makeStation("upload");
  const featured = kb.featured;
  const react = new Board({ width: 15, height: 12.5, ppu: 64 });
  react.mesh.position.set(-7.5, 2, 0);
  const server = bigBlock(
    { title: "Spring Boot", subtitle: "DocumentController", text: "Riceve il file (FilePart), lo salva come UUID.pdf e chiama DocumentService.indexPdf().", color: "#4ade80" },
    { w: 12, h: 9.5, S: 2.4 },
  );
  server.position.set(9.8, 3.6, 0);
  const sheet = pdfSheet(featured, { width: 2.7, color: "#fb923c" });
  const home = new THREE.Vector3(-2.4, 3.6, 0.6); // nella zona di caricamento della finestra
  const target = new THREE.Vector3(9.5, -2.2, 0.6);
  sheet.mesh.position.copy(home);
  // gli altri PDF aspettano in coda sotto la finestra
  const queue = kb.uploads.slice(1).map((u, i) => {
    const s = pdfSheet(u, { width: 1.9, color: "#94a3b8" });
    s.mesh.position.set(-13.5 + i * 2.3, -8.4, 0.3);
    s.home = s.mesh.position.clone();
    st.group.add(s.mesh);
    return s;
  });
  const queueLabel = new Board({ width: 15, height: 1.4, ppu: 60 });
  queueLabel.mesh.position.set(-7.5, -11.4, 0.2);
  queueLabel.redraw("q", (ctx, W, H) => paragraph(ctx, "In coda: gli altri 6 PDF del gestionale", 6, 6, W, { px: 50, color: "#94a3b8" }));
  st.group.add(react.mesh, server, sheet.mesh, queueLabel.mesh);
  const tmp = new THREE.Vector3();

  function drawReact(state) {
    const r = featured.response;
    react.redraw(state, (ctx, W, H) => {
      const box = browserFrame(ctx, W, H, { title: "Carica documenti" });
      let y = box.y;
      ctx.font = `700 48px Inter, sans-serif`;
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText("Documenti aziendali", box.x, y);
      y += 72;
      // zona di caricamento (a destra resta il posto per il PDF scelto)
      ctx.setLineDash([14, 10]);
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 3;
      ctx.strokeRect(box.x, y, box.w, 190);
      ctx.setLineDash([]);
      paragraph(ctx, featured.document.original_filename, box.x + 20, y + 20, box.w * 0.66, { px: 36, color: "#fdba74", weight: 600, maxLines: 2 });
      paragraph(ctx, `${featured.document.rawPages.length} pagine, accept=".pdf"`, box.x + 20, y + 120, box.w * 0.66, { px: 30, color: "#94a3b8" });
      y += 220;
      if (state === "idle") {
        pill(ctx, "Carica PDF", box.x, y, { px: 38, bg: "#38bdf8" });
        paragraph(ctx, "Database vuoto: 0 righe in vector_store.", box.x, y + 84, box.w, { px: 34, color: "#94a3b8" });
      } else if (state === "send") {
        paragraph(ctx, "POST /api/documents", box.x, y, box.w, { px: 40, color: "#7dd3fc", font: MONO, weight: 600 });
        paragraph(ctx, "body: FormData con il campo \"file\" (multipart/form-data)", box.x, y + 64, box.w, { px: 32, color: "#e2e8f0" });
      } else {
        paragraph(ctx, "Risposta: DocumentUploadResponse", box.x, y, box.w, { px: 32, color: "#94a3b8" });
        const done = state === "done" || state === "batch";
        const json = `{ "filename": "${r.filename}",\n  "pages": ${r.pages}, "chunks": ${done ? r.chunks : "…"},\n  "message": "${done ? r.message : "in corso…"}" }`;
        richText(ctx, json, { x: box.x, y: y + 50, w: box.w, h: 200 }, { cols: 40, color: "#bbf7d0" });
        if (state === "batch") pill(ctx, `Caricati ${kb.uploads.length} PDF: ${kb.inserts.length} chunk`, box.x, box.y + box.h - 56, { px: 34, bg: "#4ade80" });
      }
    });
  }

  /**
   * @param {"idle"|"send"|"saved"|"done"|"batch"} state
   * @param {number} [p] avanzamento del volo del PDF (stato "send") o della coda ("batch")
   */
  function set(state, p = 1) {
    drawReact(state);
    sheet.mesh.visible = state !== "batch" || p < 0.05;
    if (state === "idle") sheet.mesh.position.copy(home);
    else if (state === "send") sheet.mesh.position.copy(arcLerp(tmp, home, target, ease(p), 5));
    else sheet.mesh.position.copy(target);
    queue.forEach((q, i) => {
      const t = state === "batch" ? phase(p, i / queue.length, (i + 1) / queue.length) : 0;
      q.mesh.visible = !(state === "batch" && t >= 1);
      q.mesh.position.copy(arcLerp(tmp, q.home, target, ease(t), 6));
    });
  }
  set("idle");
  return { ...st, set, react, server };
}

// ------------------------------------------------------------------ 2 · Estrazione

export function buildExtract(kb) {
  const st = makeStation("extract");
  const pages = kb.featured.document.rawPages;
  const metas = kb.featured.pages.map((p) => p.metadata);
  const items = pages.map((page, i) => {
    const sheet = new Board({ width: 9, height: 12.5, ppu: 70 });
    sheet.redraw("p", (ctx, W, H) => {
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#0f172a";
      ctx.font = `700 46px Inter, sans-serif`;
      ctx.textBaseline = "top";
      ctx.fillText(`Document ${i + 1} · pagina ${page.pageNumber}`, 22, 14);
      ctx.fillStyle = "#64748b";
      ctx.fillRect(22, 64, W - 44, 2);
      richText(ctx, page.text, { x: 22, y: 78, w: W - 44, h: H - 100 }, { cols: 76, color: "#334155", lineGap: 1.3 });
    });
    const tag = new Board({ width: 9.6, height: 5.2, ppu: 64 });
    const m = metas[i];
    tag.redraw("m", (ctx, W, H) => {
      const box = panel(ctx, W, H, { title: "metadata", border: "#38bdf8" });
      const json = `page_number: ${m.page_number}\nfile_name: "${m.file_name.slice(0, 8)}….pdf"\nsource: "${m.source.slice(0, 8)}….pdf"\noriginal_filename:\n "${m.original_filename}"`;
      richText(ctx, json, { x: box.x, y: box.y - 6, w: box.w, h: H - box.y }, { cols: 29, color: "#bae6fd", lineGap: 1.2 });
    });
    const group = new THREE.Group();
    group.add(sheet.mesh, tag.mesh);
    tag.mesh.position.set(0, -9.3, 0.05);
    st.group.add(group);
    return { group, tag, final: new THREE.Vector3((i - 1) * 10, 2.6, 0), start: new THREE.Vector3(-14, 2.6, -0.2 - i * 0.05) };
  });
  const reader = new Board({ width: 30, height: 1.6, ppu: 60 });
  reader.mesh.position.set(0, 10.6, 0);
  reader.redraw("r", (ctx, W, H) =>
    paragraph(ctx, "PagePdfDocumentReader: un Document per pagina, con i metadati", 8, 8, W - 16, { px: 48, color: "#cbd5e1" }),
  );
  st.group.add(reader.mesh);

  /** p = 0: pagine impilate e nascoste; p = 1: pagine aperte a ventaglio con i metadati. */
  function set(p) {
    items.forEach((it, i) => {
      const t = ease(phase(p, i * 0.15, 0.55 + i * 0.15));
      it.group.visible = p > 0;
      it.group.position.lerpVectors(it.start, it.final, t);
      it.group.rotation.z = (1 - t) * 0.18 * (i - 1);
      it.tag.mesh.visible = phase(p, 0.6 + i * 0.1, 0.8 + i * 0.1) > 0;
      it.tag.mesh.scale.setScalar(Math.max(0.01, ease(phase(p, 0.6 + i * 0.1, 0.8 + i * 0.1))));
    });
  }
  set(0);
  return { ...st, set };
}

// ------------------------------------------------------------------ 3 · Pulizia

/** Intervalli da evidenziare nel "prima" e nel "dopo" di un passo di pulizia (una pagina). */
export function cleaningMarks(step, warnings, pageNumber) {
  const before = step.before.find((p) => p.pageNumber === pageNumber).text;
  const after = step.after.find((p) => p.pageNumber === pageNumber).text;
  const changes = step.changes.filter((c) => c.pageNumber === pageNumber);
  const left = [];
  const right = [];
  const joinedStarts = new Set();
  for (const c of changes) {
    if (c.kind === "removed-line") {
      const i = before.indexOf(c.text);
      if (i >= 0) left.push({ start: i, end: i + c.text.length, style: "removed" });
    } else if (c.kind === "joined-hyphen") {
      const [pair, word] = c.text.split(" → ");
      const [head, tail] = pair.split("-");
      const m = new RegExp(`${head}-\\s*\\n\\s*${tail}`).exec(before);
      if (m) left.push({ start: m.index, end: m.index + m[0].length, style: "hyphen" });
      const j = after.indexOf(word);
      if (j >= 0) right.push({ start: j, end: j + word.length, style: "added" });
    } else if (c.kind === "joined-line") {
      const i = before.indexOf(c.text.slice(0, 30));
      if (i >= 0) joinedStarts.add(i);
    }
  }
  if (step.id === "whitespace") {
    for (const m of before.matchAll(/ {2,}| +$/gm)) left.push({ start: m.index, end: m.index + m[0].length, style: "spaces" });
  }
  if (step.id === "injection-scan") {
    for (const w of warnings.filter((x) => x.pageNumber === pageNumber)) {
      const i = after.indexOf(w.excerpt);
      if (i >= 0) {
        left.push({ start: i, end: i + w.excerpt.length, style: "flagged" });
        right.push({ start: i, end: i + w.excerpt.length, style: "flagged" });
      }
    }
  }
  // "⏎" sulle righe il cui a capo viene tolto: la riga successiva inizia in joinedStarts
  const eol = (line) => {
    let next = line.end + 1;
    while (before[next] === " ") next++;
    return joinedStarts.has(next) ? "joined" : null;
  };
  return { before, after, left, right, eol };
}

export function buildClean(kb) {
  const st = makeStation("clean");
  const { cleaning } = kb.featured;
  const pageNumber = cleaning.warnings[0]?.pageNumber ?? 1; // la pagina con l'injection mostra tutti i casi
  // "prima" sopra, "dopo" sotto: la stazione è più alta che larga, come lo spazio libero
  // accanto al pannello del codice
  const left = new Board({ width: 28, height: 15.2, ppu: 62 });
  const right = new Board({ width: 28, height: 11.6, ppu: 62 });
  left.mesh.position.set(0, 6.5, 0);
  right.mesh.position.set(0, -7.4, 0);
  st.group.add(left.mesh, right.mesh); // la descrizione del passo è nella didascalia
  const total = cleaning.steps.length;

  /** i = indice del passo di pulizia (0…5), -1 = solo il testo grezzo. */
  function set(i) {
    const step = cleaning.steps[Math.max(0, i)];
    const marks = i >= 0 ? cleaningMarks(step, cleaning.warnings, pageNumber) : { before: step.before.find((p) => p.pageNumber === pageNumber).text, left: [], right: [], eol: null };
    left.redraw(`L${i}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, {
        title: i >= 0 ? `Prima · passo ${i + 1}/${total}` : "Testo estratto (grezzo)",
        subtitle: `pagina ${pageNumber}, come la restituisce il lettore PDF`,
        border: "#475569",
      });
      richText(ctx, marks.before, { x: box.x, y: box.y, w: box.w, h: H - box.y - box.pad }, { ranges: marks.left, cols: 78, eol: marks.eol, lineGap: 1.25, fit: true });
    });
    right.redraw(`R${i}`, (ctx, W, H) => {
      if (i < 0) {
        const box = panel(ctx, W, H, { title: "Dopo", border: "#334155" });
        paragraph(ctx, "Qui comparirà il testo dopo ogni passo di pulizia.", box.x, box.y, box.w, { px: 30, color: "#64748b" });
        return;
      }
      const flagged = step.id === "injection-scan";
      const box = panel(ctx, W, H, { title: `Dopo · ${step.title}`, subtitle: flagged ? "il testo NON cambia: si segnala soltanto" : "testo trasformato", border: flagged ? "#fb923c" : "#4ade80", accent: flagged ? "#fdba74" : "#86efac" });
      const res = richText(ctx, marks.after, { x: box.x, y: box.y, w: box.w, h: H - box.y - box.pad - (flagged ? 64 : 0) }, { ranges: marks.right, cols: 78, lineGap: 1.25, fit: true });
      if (flagged) {
        const w = cleaning.warnings.filter((x) => x.pageNumber === pageNumber);
        const y = H - box.pad - 52;
        const pw = pill(ctx, "Da rivedere: revisione umana", box.x, y, { px: 30 });
        paragraph(ctx, `${w.length} regole: ${w.map((x) => x.label.toLowerCase()).join("; ")}`, box.x + pw + 16, y + 4, box.w - pw - 16, { px: 24, color: "#fed7aa", maxLines: 2 });
      }
      return res;
    });
  }
  set(-1);
  return { ...st, set, pageNumber };
}
