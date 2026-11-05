/**
 * Contenuto della lavagna per ogni passo di "Dal prompt all'output". Ogni funzione riceve la
 * lavagna e i dati veri del forward (promptRun.js) e disegna: titolo, matrici, barre, testo.
 * Nessun numero inventato: tutto viene da `run` (motore JS con i pesi addestrati).
 */
import { BOARD_H, BOARD_W, INK } from "./board.js";
import { cumulative, scaledScores, visibleChar } from "./math.js";

const PAD = 44;
const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",") : v === -Infinity ? "−∞" : String(v));
const pct = (p) => `${(p * 100).toFixed(1).replace(".", ",")}%`;

/** Matrici affiancate su una riga: items = [{values, rows, cols, title, ...opzioni di matrix}]. */
export function matrixRow(board, y, items, { height = BOARD_H - y - 40, gap = 46 } = {}) {
  const width = (BOARD_W - 2 * PAD - gap * (items.length - 1)) / items.length;
  return items.map((it, i) => board.matrix(PAD + i * (width + gap), y, width, height, it.values, it.rows, it.cols, it));
}

/** Griglia di matrici (per le 4 head): `cols` matrici per riga. */
export function matrixGrid(board, y, items, cols, { height = BOARD_H - y - 30, gap = 36 } = {}) {
  const rows = Math.ceil(items.length / cols);
  const w = (BOARD_W - 2 * PAD - gap * (cols - 1)) / cols;
  const h = (height - gap * (rows - 1)) / rows;
  items.forEach((it, i) => board.matrix(PAD + (i % cols) * (w + gap), y + Math.floor(i / cols) * (h + gap), w, h, it.values, it.rows, it.cols, it));
}

/** Etichette (caratteri) delle righe = le posizioni del contesto. */
export const contextLabels = (run, k = 0) => run.steps[k].contextIds.map((id) => run.vocab[id]);

/* ------------------------------------------------------------------ dal testo ai token */

export function drawPrompt(board, run) {
  const { temperature, tokens, seed } = run.options;
  let y = board.begin("1. Il prompt", "Il testo che scrivi è solo una sequenza di caratteri. Il modello lo continuerà un carattere alla volta.");
  y = board.charCells(run.text, PAD, y + 10, { cell: 54, cols: 25 });
  board.stat(PAD, BOARD_H - 170, 430, `${run.chars.length}`, "caratteri nel prompt");
  board.stat(PAD + 460, BOARD_H - 170, 300, fmt(temperature, 2), "temperatura");
  board.stat(PAD + 790, BOARD_H - 170, 300, String(tokens), "token da generare");
  board.stat(PAD + 1120, BOARD_H - 170, 290, String(seed), "seed");
}

export function drawCut(board, run) {
  const n = run.promptIds.length;
  const T = run.config.blockSize;
  const start = run.steps[0].contextStart;
  let y = board.begin(
    "3. Taglio al contesto",
    n > T
      ? `Il prompt ha ${n} token, ma il modello ha posizioni solo per ${T} (block_size). Teniamo gli ultimi ${T}: i primi ${start} restano fuori, il modello non li vede.`
      : `Il prompt ha ${n} token: sta tutto nel contesto di ${T} posizioni. Nessun taglio.`,
  );
  y = board.charCells(run.text, PAD, y + 10, { cell: 54, cols: 25, marks: [{ from: 0, to: start, color: "rgba(30,41,59,0.35)" }, { from: start, to: n, color: "rgba(14,116,144,0.85)" }] });
  board.paragraph("idx_cond = idx[:, -block_size:]   # (B, ≤32)", PAD, BOARD_H - 110, BOARD_W - 2 * PAD, { size: 34, mono: true, color: INK.warm });
}

export function drawLookup(board, run, which) {
  const tr = run.trace;
  const labels = contextLabels(run);
  const isTok = which === "tok";
  const y = board.begin(
    isTok ? "4. Token embedding: una riga per token" : "5. Position embedding: una riga per posizione",
    isTok
      ? `Ogni id sceglie una riga della tabella (65 × 64). Il token «${visibleChar(labels.at(-1))}» (id ${run.steps[0].contextIds.at(-1)}) legge la riga ${run.steps[0].contextIds.at(-1)}. Risultato: ${tr.T} vettori da 64 numeri.`
      : `La posizione t (0…${tr.T - 1}) sceglie la riga t della tabella (32 × 64): così il modello sa DOVE sta ogni carattere. L'attention da sola non conosce l'ordine.`,
  );
  matrixRow(board, y + 6, [
    { values: isTok ? tr.tokEmb : tr.posEmb, rows: tr.T, cols: tr.C, title: isTok ? "tok_emb: righe lette (T × 64)" : "pos_emb: righe 0…T−1 (T × 64)", rowLabels: labels, highlightRow: tr.T - 1, caption: "blu = negativo, arancione = positivo; riquadro = ultima posizione" },
  ]);
}

export function drawSum(board, run) {
  const tr = run.trace;
  const labels = contextLabels(run);
  const y = board.begin("6. Somma: x = tok_emb + pos_emb", "Il vettore di ogni posizione ora dice CHE COSA è il carattere e DOVE si trova. È l'inizio del residual stream, la \"corsia\" che attraversa tutti i blocchi.");
  matrixRow(board, y + 6, [
    { values: tr.tokEmb, rows: tr.T, cols: tr.C, title: "tok_emb", rowLabels: labels },
    { values: tr.posEmb, rows: tr.T, cols: tr.C, title: "+ pos_emb" },
    { values: tr.x0, rows: tr.T, cols: tr.C, title: "= x (T × 64)", highlightRow: tr.T - 1 },
  ]);
}

/* ------------------------------------------------------------------ dentro un blocco */

const blockInput = (tr, l) => (l === 0 ? tr.x0 : tr.blocks[l - 1].resid2);

export function drawLayerNorm(board, run, l, which) {
  const tr = run.trace;
  const b = tr.blocks[l];
  const input = which === 1 ? blockInput(tr, l) : b.resid1;
  const output = which === 1 ? b.ln1 : b.ln2;
  const y = board.begin(
    `Blocco ${l} · LayerNorm ${which}`,
    "Ogni riga (un token) viene centrata e riscalata: media 0, varianza 1. Poi γ e β, appresi, la rimodellano. Serve a tenere i numeri in un intervallo comodo per lo strato che segue.",
  );
  const last = tr.T - 1;
  const stats = (m) => {
    let mean = 0;
    for (let c = 0; c < tr.C; c++) mean += m[last * tr.C + c];
    mean /= tr.C;
    let v = 0;
    for (let c = 0; c < tr.C; c++) v += (m[last * tr.C + c] - mean) ** 2;
    return `ultima riga: media ${fmt(mean, 2)}, dev. std ${fmt(Math.sqrt(v / tr.C), 2)}`;
  };
  matrixRow(board, y + 6, [
    { values: input, rows: tr.T, cols: tr.C, title: which === 1 ? "x (ingresso del blocco)" : "x dopo l'attention", rowLabels: contextLabels(run), caption: stats(input), highlightRow: last },
    { values: output, rows: tr.T, cols: tr.C, title: `ln${which}(x)`, caption: stats(output), highlightRow: last },
  ]);
}

export function drawQKV(board, run, l) {
  const tr = run.trace;
  const hs = tr.headSize;
  const y = board.begin(`Blocco ${l} · Query, Key, Value di ogni head`, `Ogni head moltiplica ln1(x) per tre matrici (64 → ${hs}): query = "che cosa cerco", key = "che cosa offro", value = "che cosa passo a chi mi sceglie".`);
  const items = [];
  tr.blocks[l].heads.forEach((h, i) => {
    items.push({ values: h.q, rows: tr.T, cols: hs, title: `head ${i} · Q`, rowLabels: i === 0 ? contextLabels(run) : undefined });
    items.push({ values: h.k, rows: tr.T, cols: hs, title: "K" });
    items.push({ values: h.v, rows: tr.T, cols: hs, title: "V" });
  });
  matrixGrid(board, y + 4, items, 6, { gap: 22 });
}

export function drawScores(board, run, l, masked) {
  const tr = run.trace;
  const y = board.begin(
    masked ? `Blocco ${l} · Maschera causale` : `Blocco ${l} · Punteggi query · key / √${tr.headSize}`,
    masked
      ? "Sopra la diagonale c'è il futuro: lì il punteggio diventa −∞ (tratteggio). Ogni token può guardare solo sé stesso e i token precedenti, come quando generiamo."
      : `Per ogni coppia (riga i, colonna j): quanto la query di i "assomiglia" alla key di j. Si divide per √${tr.headSize} = ${Math.sqrt(tr.headSize)} perché i numeri non crescano troppo e la softmax non diventi "tutto o niente".`,
  );
  matrixGrid(
    board,
    y + 4,
    // punteggi di tutte le coppie (anche il futuro): la maschera li tratteggia solo nel passo successivo
    tr.blocks[l].heads.map((h, i) => ({ values: scaledScores(h.q, h.k, tr.T, tr.headSize), rows: tr.T, cols: tr.T, title: `head ${i}`, mask: masked, square: true, rowLabels: i % 2 === 0 ? contextLabels(run) : undefined })),
    2,
  );
}

export function drawSoftmax(board, run, l) {
  const tr = run.trace;
  const y = board.begin(`Blocco ${l} · Softmax: i pesi di attention`, "Ogni riga diventa una distribuzione che somma a 1: quanto ogni token \"ascolta\" ciascuno dei precedenti. Gli stessi pattern ora sono anche nella scena, accanto alle matrici Q·K·V.");
  matrixGrid(board, y + 4, tr.blocks[l].heads.map((h, i) => ({ values: h.att, rows: tr.T, cols: tr.T, kind: "attention", square: true, title: `head ${i}`, rowLabels: i % 2 === 0 ? contextLabels(run) : undefined })), 2);
}

export function drawWeighted(board, run, l) {
  const tr = run.trace;
  const y = board.begin(`Blocco ${l} · Media pesata dei value`, "Ogni token raccoglie i value dei token che ascolta, pesati con l'attention: out = att · V. È qui che l'informazione passa da un token all'altro.");
  matrixGrid(board, y + 4, tr.blocks[l].heads.map((h, i) => ({ values: h.out, rows: tr.T, cols: tr.headSize, title: `head ${i}: att · V (T × ${tr.headSize})`, rowLabels: i === 0 ? contextLabels(run) : undefined })), 4);
}

export function drawProjection(board, run, l) {
  const tr = run.trace;
  const b = tr.blocks[l];
  const y = board.begin(`Blocco ${l} · Concatenazione e proiezione`, `Le 4 uscite da ${tr.headSize} colonne si affiancano (4 × ${tr.headSize} = ${tr.C}) e una matrice 64 × 64 le rimescola: gli "appunti" delle head diventano un solo contributo.`);
  matrixRow(board, y + 6, [
    { values: b.concat, rows: tr.T, cols: tr.C, title: "concat delle 4 head", rowLabels: contextLabels(run) },
    { values: b.proj, rows: tr.T, cols: tr.C, title: "proj(concat) + bias" },
  ]);
}

export function drawResidual(board, run, l, which) {
  const tr = run.trace;
  const b = tr.blocks[l];
  const before = which === 1 ? blockInput(tr, l) : b.resid1;
  const delta = which === 1 ? b.proj : b.ffnOut;
  const after = which === 1 ? b.resid1 : b.resid2;
  const y = board.begin(
    `Blocco ${l} · Connessione residua ${which}`,
    which === 1
      ? "Il risultato dell'attention si SOMMA a x invece di sostituirlo: x = x + attention. Il blocco aggiunge informazione senza cancellare quella che c'era (e il gradiente scorre meglio)."
      : "Anche il feed-forward si somma: x = x + ffwd. Questo vettore è l'uscita del blocco e l'ingresso del successivo.",
  );
  matrixRow(board, y + 6, [
    { values: before, rows: tr.T, cols: tr.C, title: "x", rowLabels: contextLabels(run) },
    { values: delta, rows: tr.T, cols: tr.C, title: which === 1 ? "+ attention" : "+ feed-forward" },
    { values: after, rows: tr.T, cols: tr.C, title: "= nuovo x", highlightRow: tr.T - 1 },
  ]);
}

export function drawFeedForward(board, run, l) {
  const tr = run.trace;
  const b = tr.blocks[l];
  let off = 0;
  for (let i = 0; i < b.ffnPre.length; i++) if (b.ffnPre[i] <= 0) off++;
  const y = board.begin(
    `Blocco ${l} · Feed-forward 64 → 256 → 64`,
    `Ogni token da solo passa in una piccola rete: espansione a ${tr.hidden} neuroni, ReLU (i negativi diventano 0: qui il ${Math.round((off / b.ffnPre.length) * 100)}% dei valori), compressione a 64. È dove il modello "elabora" ciò che l'attention ha raccolto.`,
  );
  matrixRow(board, y + 6, [
    { values: b.ffnPre, rows: tr.T, cols: tr.hidden, title: "net.0 (T × 256)", rowLabels: contextLabels(run) },
    { values: b.ffnPost, rows: tr.T, cols: tr.hidden, title: "dopo la ReLU" },
    { values: b.ffnOut, rows: tr.T, cols: tr.C, title: "net.2 (T × 64)" },
  ]);
}

/* ------------------------------------------------------------------ uscita */

export function drawFinalNorm(board, run) {
  const tr = run.trace;
  const y = board.begin("LayerNorm finale", "Dopo l'ultimo blocco, un'ultima normalizzazione prepara i vettori per lm_head.");
  matrixRow(board, y + 6, [
    { values: tr.blocks.at(-1).resid2, rows: tr.T, cols: tr.C, title: "uscita del blocco 3", rowLabels: contextLabels(run) },
    { values: tr.lnF, rows: tr.T, cols: tr.C, title: "ln_f(x)", highlightRow: tr.T - 1 },
  ]);
}

export function drawLogits(board, run) {
  const tr = run.trace;
  const y = board.begin("lm_head: un punteggio per ogni carattere", `Una matrice 65 × 64 trasforma ogni vettore in 65 punteggi (logits), uno per carattere del vocabolario. Per generare serve solo l'ULTIMA riga: la previsione di ciò che viene dopo «${visibleChar(contextLabels(run).at(-1))}».`);
  matrixRow(board, y + 6, [{ values: tr.logits, rows: tr.T, cols: tr.V, title: "logits (T × 65): righe = posizioni, colonne = caratteri", rowLabels: contextLabels(run), highlightRow: tr.T - 1 }]);
}

function topIds(values, k) {
  return Array.from(values.keys()).sort((a, b) => values[b] - values[a]).slice(0, k);
}

export function drawTemperature(board, run, k = 0) {
  const s = run.steps[k];
  const tau = run.options.temperature;
  const y = board.begin(
    `Temperatura τ = ${fmt(tau, 2)}`,
    tau < 1
      ? `Dividiamo i logits per ${fmt(tau, 2)}: le differenze si ALLARGANO e la distribuzione diventa più "sicura". È lo stesso parametro temperature che si imposta in Spring AI (L16).`
      : tau > 1
        ? `Dividiamo i logits per ${fmt(tau, 2)}: le differenze si RESTRINGONO e il modello diventa più "creativo" (e più incline a sbagliare).`
        : "Con τ = 1 i logits restano uguali: si usa la distribuzione del modello così com'è.",
  );
  const ids = topIds(s.logits, 12);
  const half = (BOARD_W - 2 * PAD - 40) / 2;
  const rows = ids.map((id) => ({ label: `«${visibleChar(run.vocab[id])}»`, id }));
  board.label("logits grezzi (top 12)", PAD, y + 4, { size: 30, weight: 600 });
  board.label(`logits / ${fmt(tau, 2)}`, PAD + half + 40, y + 4, { size: 30, weight: 600 });
  const lo = Math.min(...ids.map((id) => Math.min(s.logits[id], s.logits[id] / tau)), 0);
  const hi = Math.max(...ids.map((id) => Math.max(s.logits[id], s.logits[id] / tau)));
  board.bars(PAD, y + 54, half, BOARD_H - y - 100, rows.map((r) => ({ label: r.label, value: s.logits[r.id] })), { min: lo, max: hi, format: (v) => fmt(v, 2), labelWidth: 90 });
  board.bars(PAD + half + 40, y + 54, half, BOARD_H - y - 100, rows.map((r) => ({ label: r.label, value: s.logits[r.id] / tau, highlight: r.id === ids[0] })), { min: lo, max: hi, format: (v) => fmt(v, 2), labelWidth: 90 });
}

export function drawSoftmaxOut(board, run, k = 0) {
  const s = run.steps[k];
  const y = board.begin("Softmax: dai punteggi alle probabilità", `e^(z/τ) rende tutto positivo, la divisione per la somma fa sì che le 65 probabilità sommino a 1. Entropia: ${fmt(s.entropy, 2)} nat su un massimo di ln 65 ≈ 4,17.`);
  const ids = topIds(s.probs, 12);
  board.bars(PAD, y + 20, BOARD_W - 2 * PAD, BOARD_H - y - 60, ids.map((id) => ({ label: `«${visibleChar(run.vocab[id])}»`, value: s.probs[id], highlight: id === ids[0] })), { min: 0, format: pct, labelWidth: 110 });
}

/**
 * La ruota: le 65 probabilità in fila, in ordine di id (come sample() in engine/sampling.js).
 * Restituisce le coordinate (canvas) della barra, per muovere l'ago.
 */
export function drawWheel(board, run, k = 0, { title } = {}) {
  const s = run.steps[k];
  const y = board.begin(
    title ?? "Estrazione del token: la ruota",
    `Le probabilità messe in fila (in ordine di id) dividono il segmento [0, 1) in spicchi. Un numero casuale u = ${fmt(s.u, 4)} (seed ${run.options.seed}) cade nello spicchio di «${visibleChar(s.char)}», che aveva probabilità ${pct(s.chosenProb)}.`,
  );
  const cum = cumulative(s.probs);
  const x0 = PAD;
  const w = BOARD_W - 2 * PAD;
  const top = y + 70;
  const h = 120;
  const g = board.ctx;
  const ids = new Set(topIds(s.probs, 8));
  for (let i = 0; i < s.probs.length; i++) {
    const a = x0 + cum[i] * w;
    const b = x0 + cum[i + 1] * w;
    if (b - a < 0.5) continue;
    g.fillStyle = i === s.id ? INK.warm : i % 2 ? "#334155" : "#475569";
    g.fillRect(a, top, Math.max(1, b - a - 1), h);
    if (ids.has(i) && b - a > 26) board.label(visibleChar(run.vocab[i]), (a + b) / 2, top + h / 2 - 20, { size: 34, weight: 700, mono: true, align: "center", color: i === s.id ? "#0f172a" : INK.text });
  }
  board.label("0", x0, top + h + 10, { size: 26, color: INK.muted });
  board.label("1", x0 + w, top + h + 10, { size: 26, color: INK.muted, align: "right" });
  board.label(`u = ${fmt(s.u, 4)}`, x0 + s.u * w, top - 46, { size: 30, weight: 700, color: INK.warm, align: "center", mono: true });
  // le prime voci della distribuzione, per leggere i numeri
  // il token estratto è sempre nella lista, anche quando non è tra i più probabili
  const likely = topIds(s.probs, 6);
  const list = likely.includes(s.id) ? likely : [...likely.slice(0, 5), s.id];
  board.bars(PAD, top + h + 70, BOARD_W - 2 * PAD, BOARD_H - (top + h + 70) - 30, list.map((id) => ({ label: `«${visibleChar(run.vocab[id])}» id ${id}`, value: s.probs[id], highlight: id === s.id })), { min: 0, format: pct, labelWidth: 220 });
  return { x0, w, top, bottom: top + h, u: s.u };
}

export function drawAppend(board, run, k = 0) {
  const s = run.steps[k];
  const y = board.begin(
    `Il token «${visibleChar(s.char)}» si aggiunge al testo`,
    "torch.cat((idx, idx_next)): il carattere estratto diventa parte dell'input. Al giro successivo il modello rifà TUTTO il forward sul contesto allungato (gli ultimi 32 token) e prevede il carattere dopo.",
  );
  const shown = run.sequence.slice(0, run.promptIds.length + k + 1);
  const from = Math.max(0, shown.length - 6 * 25);
  const text = shown.slice(from).map((id) => run.vocab[id]).join("");
  board.charCells(text, PAD, y + 10, { cell: 54, cols: 25, marks: [{ from: run.promptIds.length - from, to: shown.length - from, color: "rgba(180,83,9,0.85)" }] });
}

/** Passo veloce di un token successivo: distribuzione + ruota + testo finora. */
export function drawFastToken(board, run, k) {
  const s = run.steps[k];
  const wheel = drawWheel(board, run, k, { title: `Token ${k + 1} di ${run.steps.length}: «${visibleChar(s.char)}»` });
  return wheel;
}

export function drawFinal(board, run) {
  const y = board.begin(
    "Il testo finale",
    `Prompt (azzurro) + ${run.steps.length} caratteri generati (arancione). Ogni carattere ha richiesto un forward completo con i pesi addestrati. Entropia media: ${fmt(run.steps.reduce((a, s) => a + s.entropy, 0) / run.steps.length, 2)} nat.`,
  );
  const n = run.promptIds.length;
  const text = run.sequence.map((id) => run.vocab[id]).join("");
  board.charCells(text, PAD, y + 10, { cell: 46, cols: 30, marks: [{ from: 0, to: n, color: "rgba(14,116,144,0.7)" }, { from: n, to: text.length, color: "rgba(180,83,9,0.85)" }] });
}
