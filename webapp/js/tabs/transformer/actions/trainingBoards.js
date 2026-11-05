/**
 * Contenuto della lavagna per le azioni di training. I numeri vengono da trace.json (training
 * registrato) o da forward veri del motore JS con i pesi degli snapshot.
 */
import { BOARD_H, BOARD_W, INK } from "./board.js";
import { visibleChar } from "./math.js";

const PAD = 44;
const fmt = (v, d = 3) => v.toFixed(d).replace(".", ",");
const sci = (v) => {
  const e = Math.floor(Math.log10(Math.abs(v)));
  return `${fmt(v / 10 ** e, 2)}e${e}`;
};
const LN_V = Math.log(65);

export function drawStart(board, m) {
  const snap = m.snapA;
  const y = board.begin(
    `Lo stato allo step ${m.s0}`,
    m.s0 === 0
      ? "Step 0: i pesi sono appena stati inizializzati a caso. Le heatmap nella scena mostrano proprio questi pesi casuali (stessa scala di colori dei pesi finali)."
      : `I pesi dopo ${m.s0} passi di AdamW (snapshot registrato). Le heatmap nella scena li mostrano con la stessa scala di colori dei pesi finali.`,
  );
  board.stat(PAD, y + 10, 440, fmt(m.trace.train_loss[m.s0]), `loss registrata allo step ${m.s0}`);
  board.stat(PAD + 470, y + 10, 440, fmt(snap.loss), "loss sul batch fisso di validation");
  board.stat(PAD + 940, y + 10, 470, fmt(LN_V), "ln 65: tirare a indovinare");
  board.label(`Testo generato con questi pesi (150 caratteri dopo "\\n", seed 42):`, PAD, y + 180, { size: 28, weight: 600 });
  board.paragraph(snap.sample_text.trim(), PAD, y + 230, BOARD_W - 2 * PAD, { size: 30, mono: true, color: INK.warm });
}

export function drawBatch(board, m) {
  const y = board.begin(
    "1. Estrazione del batch",
    `get_batch sceglie ${m.windows.length} punti a caso nel testo di training e ritaglia ${m.blockSize + 1} caratteri da ognuno. Ecco le prime 6 finestre, prese da un estratto vero di input.txt.`,
  );
  const colors = ["rgba(14,116,144,0.85)", "rgba(180,83,9,0.85)", "rgba(126,34,206,0.8)", "rgba(21,128,61,0.85)", "rgba(190,24,93,0.8)", "rgba(71,85,105,0.9)"];
  const cell = 38;
  m.windows.slice(0, 6).forEach((w, i) => {
    const row = y + 20 + i * 104;
    board.label(`finestra ${i + 1} · posizione ${w.start}`, PAD, row, { size: 26, color: INK.muted });
    board.charCells(m.text.slice(w.start, w.start + m.blockSize + 1), PAD, row + 36, { cell, cols: 34, wrapNewlines: false, marks: [{ from: 0, to: m.blockSize + 1, color: colors[i] }] });
  });
  board.label(`… e altre ${m.windows.length - 6} finestre: il batch ha forma (B, T + 1) = (${m.windows.length}, ${m.blockSize + 1}).`, PAD, BOARD_H - 60, { size: 28, color: INK.text });
}

export function drawInputTarget(board, m) {
  const w = m.windows[0];
  const y = board.begin(
    "2. Input e target",
    `Una sola finestra contiene ${m.blockSize} esempi: dato il 1° carattere prevedi il 2°, dati i primi 2 prevedi il 3°, … Il target di ogni posizione è semplicemente il carattere successivo.`,
  );
  const cell = 40;
  const cols = 32;
  const chars = (ids) => ids.map((id) => m.vocab[id]).join("");
  board.label("x (input)", PAD, y + 20, { size: 30, weight: 600, color: INK.accent });
  board.charCells(chars(w.x), PAD + 150, y + 10, { cell, cols, ids: w.x, wrapNewlines: false });
  board.label("y (target)", PAD, y + 170, { size: 30, weight: 600, color: INK.warm });
  board.charCells(chars(w.y), PAD + 150, y + 160, { cell, cols, ids: w.y, wrapNewlines: false, marks: [{ from: 0, to: 32, color: "rgba(146,64,14,0.6)" }] });
  board.paragraph(
    `Il batch intero: x e y di forma (B, T) = (${m.windows.length}, ${m.blockSize}). Sono ${m.windows.length * m.blockSize} previsioni da valutare in un colpo solo.`,
    PAD,
    y + 340,
    BOARD_W - 2 * PAD,
    { size: 32 },
  );
}

export function drawForward(board, m) {
  const pw = m.lossA.perWindow[0];
  const y = board.begin(
    "3. Forward",
    `Le ${m.windows.length} finestre attraversano il modello con i pesi dello step ${m.s0}: escono logits di forma (B, T, V) = (${m.windows.length}, ${m.blockSize}, 65). Sotto: i logits della prima finestra (32 posizioni × 65 caratteri).`,
  );
  board.matrix(PAD, y + 6, BOARD_W - 2 * PAD, BOARD_H - y - 50, pw.logits, m.blockSize, 65, { title: "logits della finestra 0", rowLabels: m.windows[0].x.map((id) => m.vocab[id]) });
}

export function drawLoss(board, m) {
  const pw = m.lossA.perWindow[0];
  const w = m.windows[0];
  const y = board.begin(
    "4. La loss: quanto ha sbagliato",
    "Per ogni posizione si guarda la probabilità che il modello dava al carattere GIUSTO: −ln p. Sicuro e giusto → quasi 0; sicuro e sbagliato → numero grande. La loss è la media su tutte le 512 previsioni.",
  );
  board.stat(PAD, y + 4, 450, fmt(m.lossA.mean), `loss di questo batch (pesi step ${m.s0})`);
  board.stat(PAD + 480, y + 4, 450, fmt(m.trace.train_loss[m.s0]), "loss registrata (batch vero dello step)");
  board.stat(PAD + 960, y + 4, 450, fmt(LN_V), "ln 65: modello uniforme");
  const n = 14;
  const items = [];
  for (let t = 0; t < n; t++) {
    items.push({ label: `«${visibleChar(m.vocab[w.x[t]])}»→«${visibleChar(m.vocab[w.y[t]])}»`, value: pw.perPosition[t], color: pw.targetProb[t] > 0.3 ? "#15803d" : pw.targetProb[t] > 0.05 ? "#a16207" : "#b91c1c" });
  }
  board.label("finestra 0, prime 14 posizioni: −ln p(target)", PAD, y + 160, { size: 28, weight: 600 });
  board.bars(PAD, y + 205, BOARD_W - 2 * PAD, BOARD_H - y - 240, items, { min: 0, format: (v) => `${fmt(v, 2)}  (p = ${fmt(Math.exp(-v), 3)})`, labelWidth: 190, valueWidth: 300 });
}

export function drawLogitGrad(board, m) {
  const y = board.begin(
    "5. Il primo gradiente: sui logits",
    `Per la cross-entropy il gradiente rispetto ai logits è semplice: p − 1 sul carattere giusto, p sugli altri (diviso per le ${m.windows.length * m.blockSize} previsioni). Negativo = "alza questo punteggio", positivo = "abbassalo".`,
  );
  const { grad, target, context } = m.logitGrad;
  const order = Array.from(grad.keys()).sort((a, b) => Math.abs(grad[b]) - Math.abs(grad[a])).slice(0, 10);
  board.label(`finestra 0, ultima posizione: dopo «…${context}» il target è «${visibleChar(m.vocab[target])}»`, PAD, y + 4, { size: 28, weight: 600 });
  board.bars(PAD, y + 54, BOARD_W - 2 * PAD, BOARD_H - y - 90, order.map((id) => ({ label: `«${visibleChar(m.vocab[id])}»`, value: grad[id] * m.windows.length * m.blockSize, highlight: id === target })), {
    format: (v) => fmt(v, 3),
    labelWidth: 110,
    color: "#be185d",
    highlightColor: INK.good,
  });
  board.label("valori moltiplicati per 512 per leggerli meglio", BOARD_W - PAD, BOARD_H - 46, { size: 22, color: INK.muted, align: "right" });
}

export function drawBackward(board, m) {
  const y = board.begin(
    "6. Backward: l'errore torna indietro",
    `loss.backward() applica la regola della catena dall'uscita fino agli embedding. Ogni gruppo di pesi riceve il suo gradiente: qui la norma VERA misurata allo step ${m.gradStep} del training registrato. Le particelle più luminose = gradiente più forte.`,
  );
  const items = m.groups.map((g) => ({ label: g.replace("blocks.", "b").replace("_embedding", " emb"), value: m.gradByGroup[g], color: "#be185d" }));
  board.bars(PAD, y + 10, BOARD_W - 2 * PAD, BOARD_H - y - 40, items, { min: 0, format: (v) => fmt(v, 3), labelWidth: 230 });
}

export function drawOptimizer(board, m) {
  const y = board.begin(
    "7. AdamW aggiorna i pesi",
    `Ogni peso si sposta di pochissimo contro il suo gradiente (lr = ${m.trace.metrics.lr[0]}). In UN passo il rapporto ‖Δw‖/‖w‖ è circa ${sci(m.updateGlobal)}: invisibile a occhio. Per vederlo, la scena anima i ${m.s1 - m.s0} passi veri tra lo snapshot ${m.s0} e lo snapshot ${m.s1}.`,
  );
  // scala logaritmica: le barre partono da 10^lo, la linea tratteggiata è il riferimento 1e-3
  const logs = m.groups.map((g) => Math.log10(m.updateByGroup[g]));
  const lo = Math.floor(Math.min(...logs, -3.5));
  const hi = Math.ceil(Math.max(...logs, -2.5));
  const items = m.groups.map((g, i) => ({ label: g.replace("blocks.", "b").replace("_embedding", " emb"), value: logs[i] - lo, color: "#7c3aed" }));
  board.label("rapporto update/peso per gruppo, scala logaritmica (riferimento CS231n: circa 1e-3)", PAD, y, { size: 26, weight: 600 });
  board.bars(PAD, y + 44, BOARD_W - 2 * PAD, BOARD_H - y - 80, items, { min: 0, max: hi - lo, format: (v) => sci(10 ** (v + lo)), labelWidth: 230 });
  const left = PAD + 230;
  const span = BOARD_W - 2 * PAD - 230 - 150;
  const x = left + ((-3 - lo) / (hi - lo)) * span;
  const g = board.ctx;
  g.strokeStyle = INK.warm;
  g.setLineDash([10, 8]);
  g.lineWidth = 3;
  g.beginPath();
  g.moveTo(x, y + 40);
  g.lineTo(x, BOARD_H - 36);
  g.stroke();
  g.setLineDash([]);
  board.label("1e-3", x, BOARD_H - 34, { size: 22, color: INK.warm, align: "center" });
}

export function drawResult(board, m) {
  const better = m.lossA.mean - m.lossB.mean;
  const y = board.begin(
    "8. Il risultato",
    `Stesso batch, pesi dello step ${m.s1}: la loss è ${better > 0 ? "scesa" : "salita"} di ${fmt(Math.abs(better))}. Il modello ha imparato un po' a prevedere questo testo. Ripetere 5000 volte è il training.`,
  );
  board.stat(PAD, y + 4, 450, fmt(m.lossA.mean), `prima (step ${m.s0})`);
  board.stat(PAD + 480, y + 4, 450, fmt(m.lossB.mean), `dopo (step ${m.s1})`, better > 0 ? INK.good : INK.bad);
  board.stat(PAD + 960, y + 4, 450, fmt(m.snapB.top1 * 100, 1) + "%", `accuracy top-1 allo step ${m.s1} (validation)`);
  board.label(`Testo generato con i pesi dello step ${m.s1}:`, PAD, y + 180, { size: 28, weight: 600 });
  board.paragraph(m.snapB.sample_text.trim(), PAD, y + 230, BOARD_W - 2 * PAD, { size: 30, mono: true, color: INK.warm });
}

/* ------------------------------------------------------------------ training completo */

export function drawRunSegment(board, r, a, b) {
  const snap = r.snapshots[a];
  const prev = r.snapshots[a - 1];
  const y = board.begin(
    `Step ${snap.step} → ${r.snapshots[b].step}`,
    a === 0
      ? "Pesi casuali: il modello produce caratteri a caso. Guarda le heatmap: cambieranno colore man mano che il training procede."
      : `Snapshot dello step ${snap.step}: loss di validation ${fmt(snap.loss)}${prev ? ` (era ${fmt(prev.loss)} allo step ${prev.step})` : ""}, accuracy top-1 ${fmt(snap.top1 * 100, 1)}%.`,
  );
  board.label(`Il modello allo step ${snap.step} scrive (150 caratteri, seed 42):`, PAD, y + 10, { size: 28, weight: 600 });
  board.paragraph(snap.sample_text.trim(), PAD, y + 60, BOARD_W - 2 * PAD, { size: 31, mono: true, color: INK.warm });
}

export function drawRunEnd(board, r) {
  const last = r.snapshots.at(-1);
  const first = r.snapshots[0];
  const y = board.begin(
    "Fine del training: i pesi finali",
    `5000 step, ${r.trace.train_config.batch_size} finestre per step. Loss di validation da ${fmt(first.loss)} a ${fmt(last.loss)}; accuracy top-1 da ${fmt(first.top1 * 100, 1)}% a ${fmt(last.top1 * 100, 1)}%. Questi pesi sono esattamente quelli del checkpoint usato dal resto della web app.`,
  );
  board.label("Testo generato alla fine:", PAD, y + 10, { size: 28, weight: 600 });
  board.paragraph(last.sample_text.trim(), PAD, y + 60, BOARD_W - 2 * PAD, { size: 31, mono: true, color: INK.warm });
}
