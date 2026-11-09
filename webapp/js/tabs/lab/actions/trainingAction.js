/**
 * Azione «Un passo di training su lm_head»: un esempio vero (il prompt fino al token scelto e il
 * carattere che viene dopo), la loss del motore, il gradiente ESATTO di lm_head scritto sulla scena
 * (p − y accanto alle righe; (p − y) · hᵀ appoggiato sopra W, cella per cella: ha la sua stessa forma),
 * l'aggiornamento con il learning rate del training
 * registrato (SGD puro e primo passo di AdamW), la loss ricalcolata dal motore che scende, e il gradiente
 * ∂L/∂h che continua all'indietro verso ln_f (i blocchi: dichiarati, non mostrati).
 */
import { CARD_COLORS } from "./cards.js";
import { codeFor } from "./code.js";
import { OUTPUT_NAMES } from "./data.js";
import { factor, int, num, pct, sci, signed } from "./format.js";
import { linearLayout, linearParts, sceneRect } from "./linearKit.js";
import { ADAMW_DEFAULTS, topK } from "./math.js";
import { TEMP_SIDE, cardRect, cellsRect, readableColumns, unionRect } from "./plan.js";
import { visibleChar } from "./settings.js";
import { onChange, progressCount } from "./stepKit.js";
import { learningRateAt, trainingStepData } from "./training.js";

/**
 * @param {import("./stage.js").ActionStage} stage
 * @param {{tab: object, engine: object, ids: number[], token: number|null, run: object}} o
 */
export function trainingStepSteps(stage, { tab, engine, ids, token, run }) {
  const S = stage.scene;
  const N = OUTPUT_NAMES.lmHead;
  const lrInfo = learningRateAt(run.trace ?? tab.state.builtin?.trace, tab.state.step);
  const lr = lrInfo?.lr ?? 1e-3;
  const lrSource = lrInfo?.source ?? "train_config del corso (gpt/config.py: learning_rate = 1e-3): il run non ha trace.json";
  const D = trainingStepData(engine, ids, { token, lr });
  if (!D) return null;
  const G = linearLayout(S, N.weight, N.bias, { outputs: 4 });
  if (!G) return [];
  const { V, C, target } = D;
  const chars = engine.vocab.map(visibleChar);
  const tch = `«${chars[target]}»`;
  const context = D.chars.map(visibleChar).join("");
  const rowLabel = (r) => chars[r];
  const [logitsCol, probsCol, oneHotCol, gradCol] = G.columns;
  // il gradiente di W ha la forma di W: si appoggia SOPRA W (h sopra le colonne, p − y accanto alle righe:
  // la cella (c, j) sta all'incrocio, come nello schema del prodotto esterno)
  const dWRect = G.W;
  const dWGradCol = logitsCol;
  const dhRow = cardRect({ x0: G.input.x0, y1: G.input.y1 + 0.08 + TEMP_SIDE + 0.05, rows: 1, cols: C, cellW: G.W.cellW, cellH: TEMP_SIDE });
  const oneHot = new Float32Array(V);
  oneHot[target] = 1;
  const gradRange = Math.max(...D.grad.dLogits.map(Math.abs));
  const dWRange = Math.max(...D.grad.dW.map(Math.abs));
  const jStar = topK(D.h.map(Math.abs), 1)[0].id; // la dimensione di h più grande: il suo peso si muove di più
  const best = topK(D.probs, 1)[0].id;
  const parts = linearParts(stage, G, { x: D.h, W: D.W, b: D.b, y: D.logits, names: N, inDim: C, outDim: V, inLabel: "h", outLabel: "logits", rowLabel });
  const strip = (marked = target) => stage.strip("tr-chars", chars, chars.map((_, r) => ({ x: G.columns.at(-1).x1 + 0.05, y: G.W.y1 - (r + 0.5) * G.W.cellH })), { height: Math.min(0.034, G.W.cellH * 0.85), marked });
  const cards = {
    logits: (o = {}) => stage.card("lin-output", { rect: logitsCol, data: D.logits, label: "logits", labelSide: "top", accent: CARD_COLORS.output, range: Math.max(...D.logits.map(Math.abs)), rowLabel, ...o }),
    probs: (o = {}) => stage.card("tr-probs", { rect: probsCol, data: D.probs, label: "p", labelSide: "top", accent: CARD_COLORS.probs, colormap: "attention", range: Math.max(...D.probs), rowLabel, ...o }),
    oneHot: (o = {}) => stage.card("tr-onehot", { rect: oneHotCol, data: oneHot, label: "y", labelSide: "top", accent: CARD_COLORS.mask, colormap: "attention", range: 1, rowLabel, ...o }),
    grad: (o = {}) => stage.card("tr-grad", { rect: gradCol, data: D.grad.dLogits, label: "p − y", labelSide: "top", accent: CARD_COLORS.grad, range: gradRange, rowLabel, ...o }),
    dW: (o = {}) => stage.card("tr-dw", { rect: dWRect, data: D.grad.dW, label: "∂L/∂W = (p − y) · hᵀ", labelSide: "bottom", accent: CARD_COLORS.grad, range: dWRange, rowLabel, ...o }),
    dWGrad: (o = {}) => stage.card("tr-dw-grad", { rect: dWGradCol, data: D.grad.dLogits, label: "p − y", labelSide: "top", accent: CARD_COLORS.grad, range: gradRange, ...o }),
    dh: (o = {}) => stage.card("tr-dh", { rect: dhRow, data: D.dh, label: "∂L/∂h", labelSide: "left", accent: CARD_COLORS.grad, ...o }),
  };
  const forward = () => {
    parts.input();
    cards.logits();
    cards.probs().mark([{ r0: target, r1: target + 1, c0: 0, c1: 1 }]);
    strip();
  };
  const top = G.all;
  const withGrad = unionRect(G.input, G.W, G.B, ...G.columns);
  const withDW = unionRect(G.input, G.W, G.B, dWGradCol);
  // la finestra dei numeri sulla riga del carattere giusto, attorno alla colonna jStar (pesi prima/dopo)
  const targetWindow = () => {
    const n = readableColumns(stage.freeArea().width, { minCellPx: 58, min: 4, max: 12 });
    const c0 = Math.max(0, Math.min(C - n, jStar - Math.floor(n / 2)));
    return cellsRect(G.W, Math.max(0, target - 1), Math.min(V, target + 2), c0, c0 + n);
  };
  const around = [jStar - 1, jStar, jStar + 1].filter((j) => j >= 0 && j < C);
  // γ e β di ln_f nella scena: i loro gradienti si appoggiano subito sopra e sotto
  const lnFGamma = sceneRect(S, OUTPUT_NAMES.lnF.weight);
  const lnFBeta = sceneRect(S, OUTPUT_NAMES.lnF.bias);
  const rowAboveRect = (r) => cardRect({ x0: r.x0, y1: r.y1 + 0.13 + TEMP_SIDE, rows: 1, cols: r.cols, cellW: r.cellW, cellH: TEMP_SIDE });
  const rowBelowRect = (r) => cardRect({ x0: r.x0, y1: r.y0 - 0.05, rows: 1, cols: r.cols, cellW: r.cellW, cellH: TEMP_SIDE });
  const totalParams = run.totalParams ?? 0;
  const lossCode = codeFor("loss", "gpt/loss.py");
  const trainCode = (id) => codeFor(id, "gpt/training.py");
  const wBefore = D.W[target * C + jStar];
  const wSgd = D.sgd.W[target * C + jStar];
  const wAdam = D.adamw.W[target * C + jStar];
  const uniform = Math.log(V);

  return [
    {
      title: "L'esempio",
      caption: `Nel training ogni carattere del testo è la risposta giusta per i caratteri prima di lui. L'esempio qui è il tuo prompt: il modello legge «${context}» e deve prevedere ${tch}.`,
      duration: 6,
      code: trainCode("forwardLoss"),
      scene: forward,
      board: () => [
        { title: `«${context}» → ${tch}`, kicker: "Un passo di training" },
        { line: [`Il motore JS esegue il modello (run nella scena, ${tab.state.step === null ? "pesi finali" : `step ${int(tab.state.step)}`}) sul contesto di ${D.context.length} caratteri; sull'ultimo dà le probabilità del prossimo.`] },
        { note: `Nel training vero: ogni passo prende 16 finestre di 32 caratteri (batch) e fa la media su 16 × 32 = 512 previsioni come questa.` },
      ],
      frame: () => top,
    },
    {
      title: "La loss: −ln p(giusto)",
      caption: `Il modello dà a ${tch} il ${pct(D.p)}. La loss è −ln di quel numero: ${num(D.loss, 3)}. Più il modello è sicuro della risposta giusta, più la loss è vicina a 0; tirando a caso sarebbe ln ${V} = ${num(uniform, 2)}.`,
      duration: 6,
      code: lossCode,
      scene: forward,
      board: () => [
        { title: `loss = −ln p(${tch})`, kicker: "Un passo di training" },
        { formula: ["−ln ", { text: num(D.p, 5), tone: "num" }, " = ", { text: num(D.loss, 4), tone: "accent" }] },
        { bars: [{ label: `p(${tch})`, value: D.p, text: pct(D.p), mark: true }, { label: `p(«${chars[best]}»)`, value: D.probs[best], text: `${pct(D.probs[best])} (il più probabile)` }], max: 1 },
      ],
      frame: () => top,
    },
    {
      title: "∂L/∂logits = p − y",
      caption: `Il gradiente della loss rispetto ai logits è semplicissimo: le probabilità meno la risposta giusta (y = 1 su ${tch}, 0 altrove). Negativo su ${tch} (il suo punteggio deve salire), positivo sugli altri (devono scendere).`,
      duration: 6,
      code: trainCode("backward"),
      scene: () => {
        forward();
        cards.oneHot();
        cards.grad({ reveal: 0 }).mark([{ r0: target, r1: target + 1, c0: 0, c1: 1 }]);
      },
      board: () => [
        { title: "p − y, carattere per carattere", kicker: "Gradiente" },
        { formula: [`${tch}: `, { text: num(D.p, 4), tone: "num" }, " − 1 = ", { text: num(D.grad.dLogits[target], 4), tone: "neg" }] },
        { formula: [`«${chars[best]}»: `, { text: num(D.probs[best], 4), tone: "num" }, " − 0 = ", { text: signed(D.grad.dLogits[best], 4), tone: "pos" }] },
        { note: "È esatto (derivata della cross-entropy con la softmax), non un'approssimazione: un test lo confronta con le differenze finite." },
      ],
      update: (() => {
        const stepFn = onChange((k) => stage.cards.get("tr-grad").reveal(k));
        return (p) => stepFn(progressCount(p, V, { hold: 0.3 }));
      })(),
      frame: () => withGrad,
    },
    {
      title: "∂L/∂b = p − y",
      caption: "La bias di lm_head si somma direttamente ai logits: il suo gradiente è proprio p − y, carattere per carattere. È la colonna della bias nella scena.",
      duration: 5,
      code: trainCode("backward"),
      scene: () => {
        forward();
        cards.oneHot();
        cards.grad().mark([{ r0: target, r1: target + 1, c0: 0, c1: 1 }]);
        if (G.B) stage.outline(G.B, { pad: 0.02 });
      },
      board: () => [
        { title: "∂L/∂b = p − y", kicker: "Gradiente" },
        { formula: [`∂L/∂b[${tch}] = `, { text: num(D.grad.db[target], 4), tone: "neg" }, `   b[${tch}] = `, { text: num(D.b[target], 4), tone: "num" }] },
        { line: ["Un gradiente negativo vuol dire: alzare quel numero abbassa la loss. La discesa del gradiente va nella direzione opposta al gradiente."] },
      ],
      frame: () => withGrad,
    },
    {
      title: "∂L/∂W = (p − y) · hᵀ",
      caption: `Ogni peso W[c, j] moltiplica h[j] per arrivare al logit di c: il suo gradiente è (p − y)[c] · h[j]. Ha la forma di W e si appoggia sopra W, cella per cella: la riga di ${tch} è h per un numero negativo, le altre h per la loro probabilità (quasi tutte minuscole).`,
      duration: 7,
      code: trainCode("backward"),
      scene: () => {
        parts.input();
        cards.dW({ reveal: 0 });
        cards.dWGrad().mark([{ r0: target, r1: target + 1, c0: 0, c1: 1 }]);
      },
      board: () => [
        { title: "∂L/∂W[c, j] = (p − y)[c] · h[j]", kicker: "Gradiente" },
        { formula: [`∂L/∂W[${tch}, ${jStar}] = `, { text: factor(D.grad.dLogits[target], 4), tone: "num" }, " · ", { text: factor(D.h[jStar], 4), tone: "num" }, " = ", { text: num(D.grad.dW[target * C + jStar], 4), tone: "accent" }] },
        { line: ["Riga in corso: ", { text: "", tone: "num", live: "dw" }] },
      ],
      update: (() => {
        const stepFn = onChange((r) => {
          const card = stage.cards.get("tr-dw");
          card.reveal(r * C);
          card.mark(r < V ? [{ r0: r, r1: r + 1, c0: 0, c1: C }] : [{ r0: target, r1: target + 1, c0: 0, c1: C }]);
          stage.board.setLive("dw", r < V ? `«${chars[r]}» (${r + 1} di ${V})` : `tutte le ${V}`);
        });
        return (p) => stepFn(progressCount(p, V));
      })(),
      frame: () => withDW,
    },
    {
      title: "Un passo di discesa del gradiente",
      caption: `Ogni peso si sposta di −lr · gradiente, con il learning rate del training registrato: lr = ${num(lr, 4)}. I numeri nelle celle di lm_head ora sono quelli DOPO il passo (avvicinati per leggerli): cambiano alla terza o quarta cifra.`,
      duration: 7,
      code: trainCode("step"),
      scene: () => {
        stage.showValues(new Map([[N.weight, D.sgd.W], [N.bias, D.sgd.b]]), "lm_head dopo un passo SGD");
        stage.highlight(N.weight, { rows: [target] }, { dim: 0.35 });
        stage.scene.hoverCell(N.weight, { row: target, col: jStar });
      },
      board: () => [
        { title: "W ← W − lr · ∂L/∂W", kicker: "SGD" },
        { formula: [`W[${tch}, ${jStar}]: `, { text: num(wBefore, 5), tone: "num" }, " − ", num(lr, 4), " · ", factor(D.grad.dW[target * C + jStar], 4), " = ", { text: num(wSgd, 5), tone: "good" }] },
        { table: { head: [`W[${tch}, j]`, ...around.map((j) => `j = ${j}`)], rows: [["prima", ...around.map((j) => num(D.W[target * C + j], 5))], ["dopo", ...around.map((j) => num(D.sgd.W[target * C + j], 5))]] } },
        { line: [`lr = ${num(lr, 4)} (${lrSource}). |Δ| medio dei pesi: `, { text: sci(D.sgd.statsW.meanAbs, 2), tone: "num" }, ", massimo ", { text: sci(D.sgd.statsW.maxAbs, 2), tone: "num" }, "."] },
        { note: `Spostamenti così piccoli che ${int(D.sgd.statsW.count - D.sgd.statsW.moved)} pesi su ${int(D.sgd.statsW.count)} non cambiano nemmeno nel float32: il training fa migliaia di passi.` },
      ],
      frame: targetWindow,
    },
    {
      title: "La loss scende",
      caption: `Il motore JS ricalcola il modello con lm_head aggiornata: la loss passa da ${num(D.loss, 4)} a ${num(D.sgd.loss, 4)} (${signed(D.sgd.loss - D.loss, 4)}) e p(${tch}) da ${pct(D.p)} a ${pct(D.sgd.p)}. Un passo piccolo nella direzione giusta.`,
      duration: 6,
      code: trainCode("step"),
      scene: () => {
        parts.input();
        cards.logits();
        stage.card("tr-probs-after", { rect: probsCol, data: D.sgd.probs, label: "p dopo", labelSide: "top", accent: CARD_COLORS.probs, colormap: "attention", range: Math.max(...D.probs), rowLabel }).mark([{ r0: target, r1: target + 1, c0: 0, c1: 1 }]);
        strip();
        stage.showValues(new Map([[N.weight, D.sgd.W], [N.bias, D.sgd.b]]), "lm_head dopo un passo SGD");
      },
      board: () => [
        { title: "Prima e dopo il passo", kicker: "SGD" },
        {
          table: {
            head: ["", "prima", "dopo"],
            rows: [
              ["loss", num(D.loss, 4), num(D.sgd.loss, 4)],
              [`p(${tch})`, pct(D.p, 2), pct(D.sgd.p, 2)],
            ],
          },
        },
        { note: "Il motore usa una copia dei pesi: la scena e i pesi del run non cambiano davvero (a fine azione torna tutto com'era)." },
      ],
      frame: () => top,
    },
    {
      title: "Il training vero: AdamW",
      caption: `gpt/training.py usa AdamW, non SGD: ogni peso si sposta di circa lr nella direzione giusta, quasi qualunque sia la grandezza del suo gradiente (lo divide per una media della sua grandezza), più un piccolo «weight decay» che lo tira verso zero. Al primo passo: loss ${num(D.adamw.loss, 4)}.`,
      duration: 8,
      code: trainCode("adamw"),
      scene: () => {
        stage.showValues(new Map([[N.weight, D.adamw.W], [N.bias, D.adamw.b]]), "lm_head dopo il 1° passo di AdamW");
        stage.highlight(N.weight, { rows: [target] }, { dim: 0.35 });
        stage.scene.hoverCell(N.weight, { row: target, col: jStar });
      },
      board: () => [
        { title: "SGD e AdamW sullo stesso gradiente", kicker: "Ottimizzatore" },
        {
          table: {
            head: ["", "SGD", "AdamW (1° passo)"],
            rows: [
              [`W[${tch}, ${jStar}]`, num(wSgd, 5), num(wAdam, 5)],
              ["|Δ| medio", sci(D.sgd.statsW.meanAbs, 2), sci(D.adamw.statsW.meanAbs, 2)],
              ["loss dopo", num(D.sgd.loss, 4), num(D.adamw.loss, 4)],
            ],
          },
        },
        { note: `I numeri nelle celle di lm_head ora sono quelli dopo il primo passo di AdamW. Primo passo di un AdamW appena creato (β₁ = ${ADAMW_DEFAULTS.beta1}, β₂ = ${ADAMW_DEFAULTS.beta2}, ε = 10⁻⁸, weight decay ${ADAMW_DEFAULTS.weightDecay}: i valori predefiniti di PyTorch). Il training vero è al passo 5.000 con le medie m e v accumulate, che il checkpoint non salva: il suo passo vero non si può rifare qui.` },
      ],
      frame: targetWindow,
    },
    {
      title: "E il resto della rete?",
      caption: `Il gradiente non si ferma a lm_head: ∂L/∂h = Wᵀ · (p − y) entra in ln_f, che ha i suoi pesi γ e β (qui sotto, calcolati e verificati), e poi all'indietro nei blocchi (backpropagation). Ogni peso dei ${int(totalParams)} riceve il suo; i blocchi qui non sono mostrati.`,
      duration: 8,
      code: trainCode("backward"),
      scene: () => {
        if (lnFGamma && lnFBeta) {
          stage.card("tr-dgamma", { rect: rowAboveRect(lnFGamma), data: D.lnF.dGamma, label: "∂L/∂γ", labelSide: "left", accent: CARD_COLORS.grad });
          stage.card("tr-dbeta", { rect: rowBelowRect(lnFBeta), data: D.lnF.dBeta, label: "∂L/∂β = ∂L/∂h", labelSide: "left", accent: CARD_COLORS.grad });
          stage.outline([lnFGamma, lnFBeta], { pad: 0.012 });
        } else {
          cards.dh();
          parts.input();
        }
      },
      board: () => [
        { title: "∂L/∂h = Wᵀ · (p − y), poi ln_f", kicker: "Backpropagation" },
        { vector: { values: D.dh, max: 12, caption: `∂L/∂h (i primi 12 di ${C}): il segnale d'errore che torna indietro.` } },
        { formula: ["ln_f: h = x̂ · γ + β  →  ∂L/∂β = ∂L/∂h,  ∂L/∂γ = ∂L/∂h · x̂"] },
        { note: "Verificati con le differenze finite sul motore JS (un test sposta γ e β di ln_f e misura la loss). Il resto (i 4 blocchi, gli embedding) non è mostrato: nel training vero torch.autograd calcola tutti i gradienti e AdamW aggiorna tutti i pesi insieme." },
      ],
      frame: () => (lnFGamma && lnFBeta ? unionRect(rowAboveRect(lnFGamma), rowBelowRect(lnFBeta)) : unionRect(dhRow, G.input, G.W)),
    },
  ];
}
