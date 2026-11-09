/**
 * Le due azioni della self-attention di una head (gpt/attention.py), con le attivazioni vere del motore:
 *
 * «q · kᵀ, scala, maschera e softmax»: Q a sinistra, Kᵀ sopra, i punteggi scritti SOPRA il buffer tril
 * della scena (la maschera è proprio lì); la scala 1/√head_size, il futuro a −∞, la softmax riga per riga.
 *
 * «· V, concatenazione, proj e residuo»: la media dei value pesata dall'attention (V sopra l'uscita, come
 * nello schema di una moltiplicazione), le head affiancate sopra la W di proj, la proiezione e il residuo.
 */
import { CARD_COLORS } from "./cards.js";
import { codeFor } from "./code.js";
import { attentionData, rowOf } from "./data.js";
import { factor, int, num, pct, vectorPreview } from "./format.js";
import { linearParts, sceneRect } from "./linearKit.js";
import { addVectors, linearForward, maxAbsDiff, softmaxRowSteps, topK } from "./math.js";
import { CARD_GAP, LABEL_CLEARANCE, TEMP_SIDE, blockAbove, blockLeft, blockRight, cardRect, cellCenter, columnRight, padRect, subRect, unionRect } from "./plan.js";
import { visibleChar } from "./settings.js";
import { onChange, progressCount } from "./stepKit.js";

/** La geometria comune: punteggi sopra il tril, Q a sinistra, Kᵀ sopra, uscita e V a destra. */
function attentionLayout(S, f) {
  const tril = sceneRect(S, f.names.tril);
  if (!tril || tril.rows < f.T) return null;
  const scores = subRect(tril, f.T, f.T);
  const q = blockLeft(scores, { cols: f.hs });
  const kT = blockAbove(scores, { rows: f.hs });
  const out = blockRight(scores, { cols: f.hs, gap: CARD_GAP * 2.4 });
  const v = blockAbove(out, { rows: f.T });
  return { tril, scores, q, kT, out, v };
}

/**
 * Le etichette dei token: a sinistra delle righe (le query) e sopra le colonne (le key). `beside`:
 * "q" = a sinistra di Q (quando Q è visibile), "scores" = subito a sinistra dei punteggi.
 */
function tokenStrips(stage, f, A, { rows = true, cols = true, marked = null, beside = "q" } = {}) {
  const texts = f.chars.map(visibleChar);
  const h = A.scores.cellH * 1.05;
  if (rows) {
    const x = (beside === "q" ? A.q.x0 : A.scores.x0) - h * 0.75;
    stage.strip(`att-rows-${beside}`, texts, texts.map((_, i) => ({ x, y: cellCenter(A.scores, i, 0).y })), { height: h, marked });
  }
  if (cols) {
    const y = A.kT.y1 + h * 0.75;
    stage.strip("att-cols", texts, texts.map((_, j) => ({ x: cellCenter(A.scores, 0, j).x, y })), { height: h });
  }
}

/** Il rettangolo dei punteggi con la fila dei token a sinistra (per i passi senza Q e Kᵀ). */
function scoresWithTokens(A) {
  return padRect({ ...A.scores, x0: A.scores.x0 - A.scores.cellW * 1.6 }, 0.04);
}

/** La key a cui la query i dà più peso (la più interessante da mostrare). */
function strongestKey(att, T, i) {
  return topK(att.subarray(i * T, i * T + i + 1), 1)[0]?.id ?? i;
}

/** «q · kᵀ, scala, maschera e softmax». */
export function attentionSteps(stage, f) {
  const S = stage.scene;
  const A = attentionLayout(S, f);
  if (!A) return null;
  const D = attentionData(f);
  const { T, hs, t: i } = f;
  const j = strongestKey(D.att, T, i);
  const ch = (k) => `«${visibleChar(f.chars[k])}»`;
  const qRow = rowOf(D.q, i, hs);
  const kRow = rowOf(D.k, j, hs);
  const kT = transpose(D.k, T, hs); // (hs × T): una colonna per token
  const dotsMax = maxAbs(D.dots);
  const scaledMax = maxAbs(D.scaled);
  const head = `head ${f.head} del blocco ${f.block}`;
  const scoresDiff = maxAbsDiffFinite(D.masked, D.engine.scores);
  const attDiff = maxAbsDiff(D.att, D.engine.att);
  const all = unionRect(A.q, A.kT, A.scores);
  const future = (r, c) => c > r;
  const rowSteps = softmaxRowSteps(D.masked, T, i);

  const cardQ = (o = {}) => stage.card("att-q", { rect: A.q, data: D.q, label: "Q", labelSide: "top", accent: CARD_COLORS.input, ...o });
  const cardK = (o = {}) => stage.card("att-k", { rect: A.kT, data: kT, label: "Kᵀ", labelSide: "left", accent: CARD_COLORS.input, ...o });
  const cardScores = (data, o = {}) => stage.card("att-scores", { rect: A.scores, data, label: "punteggi", labelSide: "bottom", accent: CARD_COLORS.product, range: dotsMax, maskedText: "", ...o });

  return [
    {
      title: "Q e K di ogni token",
      caption: `Ogni token del prompt ha una query (che cosa cerco) e una key (che cosa contengo): ${hs} numeri ciascuna nella ${head}. Kᵀ è K girata: una colonna per token.`,
      duration: 5,
      code: codeFor("query", head),
      scene: () => {
        cardQ();
        cardK();
        cardScores(D.dots, { reveal: 0 });
        tokenStrips(stage, f, A);
      },
      board: () => [
        { title: `Q (${T} × ${hs}) e Kᵀ (${hs} × ${T}) della ${head}`, kicker: "Attention" },
        { line: [`head_size = ${f.C} / ${f.config.n_head} = ${hs}: ogni head lavora su una fetta dei ${f.C} numeri. q = x · Wqᵀ e k = x · Wkᵀ (azione «x · Wᵀ riga per colonna»).`] },
        { line: ["Il punteggio tra il token i e il token j è q[i] · k[j]: la riga i di Q per la colonna j di Kᵀ, come in ogni moltiplicazione tra matrici."] },
      ],
      frame: () => all,
    },
    {
      title: "Un punteggio: q · k",
      caption: `La query di ${ch(i)} incontra la key di ${ch(j)}: ${hs} prodotti sommati. Più i due vettori puntano nella stessa direzione, più il punteggio è alto (come due liste di interessi che si somigliano).`,
      duration: 6,
      code: codeFor("scores", head),
      scene: () => {
        cardQ().mark([{ r0: i, r1: i + 1, c0: 0, c1: hs }]);
        cardK().mark([{ r0: 0, r1: hs, c0: j, c1: j + 1 }]);
        const sc = cardScores(D.dots, { reveal: (r, c) => r === i && c === j });
        sc.mark([{ r0: i, r1: i + 1, c0: j, c1: j + 1 }]);
        tokenStrips(stage, f, A, { marked: i });
        const target = sc.cellCenter(i, j);
        stage.link([
          [{ x: A.q.x1, y: target.y }, target],
          [{ x: target.x, y: A.kT.y0 }, target],
        ]);
      },
      board: () => [
        { title: `q[${ch(i)}] · k[${ch(j)}]`, kicker: "Attention" },
        { formula: [`q · k = `, { text: factor(qRow[0], 3), tone: "num" }, "·", { text: factor(kRow[0], 3), tone: "num" }, " + ", { text: factor(qRow[1], 3), tone: "num" }, "·", { text: factor(kRow[1], 3), tone: "num" }, ` + … (${hs} prodotti) = `, { text: num(D.dots[i * T + j]), tone: "accent" }] },
        { note: `È il numero nella cella (riga ${i}, colonna ${j}) dei punteggi, scritta sopra il buffer tril della scena.` },
      ],
      frame: () => all,
    },
    {
      title: "Tutti i punteggi",
      caption: `Ogni riga di Q per ogni colonna di Kᵀ: ${T} × ${T} = ${int(T * T)} punteggi. Per ora anche quelli del futuro (sopra la diagonale): la maschera arriva tra poco.`,
      duration: 6,
      code: codeFor("scores", head),
      scene: () => {
        cardQ();
        cardK();
        cardScores(D.dots, { reveal: 0 });
        tokenStrips(stage, f, A);
      },
      board: () => [
        { title: "S = Q · Kᵀ", kicker: "Attention" },
        { line: [`La riga i dice quanto il token i «si interessa» a ogni token j. Il punteggio più alto della riga di ${ch(i)}: `, { text: `${ch(j)} = ${num(D.dots[i * T + j], 3)}`, tone: "accent" }, "."] },
        { line: ["Riga in corso: ", { text: "", tone: "num", live: "rows" }] },
      ],
      update: (() => {
        const step = onChange((r) => {
          stage.cards.get("att-scores").reveal(r * T);
          stage.cards.get("att-q").mark(r < T ? [{ r0: r, r1: r + 1, c0: 0, c1: hs }] : null);
          stage.board.setLive("rows", r < T ? `${r + 1} di ${T} (token ${ch(r)})` : `tutte le ${T} righe`);
        });
        return (p) => step(progressCount(p, T));
      })(),
      frame: () => all,
    },
    {
      title: `La scala 1/√${hs}`,
      caption: `Tutti i punteggi si moltiplicano per 1/√${hs} = ${num(D.scale, 2)}: i colori si spengono. Senza, con vettori lunghi i punteggi crescerebbero e la softmax darebbe quasi tutto a un solo token.`,
      duration: 5,
      code: codeFor("scores", head),
      scene: () => {
        cardQ();
        cardK();
        cardScores(D.dots);
        tokenStrips(stage, f, A);
      },
      board: () => [
        { title: `× 1/√${hs} = × ${num(D.scale, 2)}`, kicker: "Attention" },
        { formula: [`${ch(i)} → ${ch(j)}: `, { text: num(D.dots[i * T + j], 3), tone: "num" }, ` × ${num(D.scale, 2)} = `, { text: num(D.scaled[i * T + j], 3), tone: "accent" }] },
        { note: "Vaswani et al. (2017): si divide per √d_k perché, con q e k di varianza 1, il prodotto scalare ha varianza d_k. Dopo la scala torna intorno a 1." },
      ],
      update: (() => {
        const mixed = new Float32Array(T * T);
        const step = onChange((k) => {
          const w = k / 10;
          for (let n = 0; n < mixed.length; n++) mixed[n] = D.dots[n] + (D.scaled[n] - D.dots[n]) * w;
          stage.cards.get("att-scores").setData(mixed);
        });
        return (p) => step(Math.round(Math.min(1, p / 0.7) * 10));
      })(),
      frame: () => all,
    },
    {
      title: "La maschera causale (tril)",
      caption: `Il buffer tril sotto i punteggi: 1 sotto la diagonale, 0 sopra. Dove c'è 0 il punteggio diventa −∞: il token ${ch(i)} non può guardare i token che vengono dopo di lui. Il futuro non si sbircia.`,
      duration: 6,
      code: codeFor("mask", head),
      scene: () => {
        const sc = cardScores(D.scaled, { mask: future, maskedText: "−∞", range: scaledMax });
        sc.visible = false;
        tokenStrips(stage, f, A, { cols: false, beside: "scores" });
        stage.highlight(f.names.tril, { r0: 0, r1: T, c0: 0, c1: T }, { dim: 0.45 });
      },
      board: () => [
        { title: "masked_fill(tril == 0, −∞)", kicker: "Attention" },
        { line: [`tril è un buffer (register_buffer in gpt/attention.py): numeri fissi, non è un peso e il training non lo cambia. Qui serve la sua parte ${T} × ${T}: il prompt ha ${T} token.`] },
        { line: [maskExample(i, T, ch)] },
      ],
      update: (() => {
        const step = onChange((show) => {
          stage.cards.get("att-scores").visible = show;
        });
        return (p) => step(p > 0.4);
      })(),
      frame: () => padRect(A.tril, 0.05),
    },
    {
      title: "Softmax riga per riga",
      caption: `Ogni riga diventa una distribuzione: e^(punteggio − massimo), diviso per la somma della riga. I −∞ diventano 0, gli altri numeri tra 0 e 1 che sommano a 1: è il 100% dell'attenzione di quel token.`,
      duration: 6,
      code: codeFor("softmax", head),
      scene: () => {
        stage.card("att-weights", { rect: A.scores, data: D.att, label: "pesi di attention", labelSide: "bottom", accent: CARD_COLORS.probs, colormap: "attention", range: 1, maskedText: "0", mask: future, reveal: 0 });
        tokenStrips(stage, f, A, { cols: false, marked: i, beside: "scores" });
      },
      board: () => {
        const top = topK(rowSteps.probs.subarray(0, i + 1), 3);
        return [
          { title: `softmax della riga di ${ch(i)}`, kicker: "Attention" },
          ...top.map((e) => ({ formula: [`${ch(e.id)}: e^(${num(D.masked[i * T + e.id], 3)} − ${num(rowSteps.max, 3)}) = ${num(rowSteps.exps[e.id], 4)} → ${num(rowSteps.exps[e.id], 4)} / ${num(rowSteps.sum, 4)} = `, { text: num(e.value, 4), tone: "accent" }] })),
          { line: [`Somma della riga: ${num(sumRow(D.att, T, i), 6)}. I token dopo ${ch(i)}: 0 (erano −∞).`] },
        ];
      },
      update: (() => {
        const step = onChange((r) => {
          const card = stage.cards.get("att-weights");
          card.reveal(r * T);
          card.mark(r < T ? [{ r0: r, r1: r + 1, c0: 0, c1: T }] : [{ r0: i, r1: i + 1, c0: 0, c1: T }]);
        });
        return (p) => step(progressCount(p, T));
      })(),
      frame: () => scoresWithTokens(A),
    },
    {
      title: "Chi guarda chi",
      caption: `I pesi di attention della ${head}: ${ch(i)} guarda soprattutto ${ch(j)} (${pct(D.att[i * T + j])}). Come in una riunione: ognuno ascolta chi ha parlato prima di lui, e dà a ciascuno una quota della sua attenzione.`,
      duration: 6,
      code: codeFor("softmax", head),
      scene: () => {
        stage.card("att-weights", { rect: A.scores, data: D.att, label: "pesi di attention", labelSide: "bottom", accent: CARD_COLORS.probs, colormap: "attention", range: 1, maskedText: "0", mask: future }).mark([{ r0: i, r1: i + 1, c0: 0, c1: i + 1 }]);
        tokenStrips(stage, f, A, { cols: false, marked: i, beside: "scores" });
      },
      board: () => [
        { title: `La riga di ${ch(i)}: a chi dà attenzione`, kicker: "Attention" },
        { bars: topK(D.att.subarray(i * T, i * T + i + 1), Math.min(6, i + 1)).map((e) => ({ label: `${e.id}: ${ch(e.id)}`, value: e.value, text: pct(e.value), mark: e.id === j })), max: 1 },
        { note: scoresDiff === 0 && attDiff === 0 ? "Controllo: punteggi e pesi uguali a quelli del motore JS, bit per bit." : `Controllo: scarto massimo dal motore ${num(Math.max(scoresDiff, attDiff), 8)}.` },
      ],
      frame: () => scoresWithTokens(A),
    },
  ];
}

/** «· V, concatenazione, proj e residuo». */
export function attentionValueSteps(stage, f) {
  const S = stage.scene;
  const A = attentionLayout(S, f);
  const P = sceneRect(S, f.names.proj.weight);
  const PB = sceneRect(S, f.names.proj.bias);
  if (!A || !P) return null;
  const D = attentionData(f);
  const { T, hs, C, t: i } = f;
  const H = f.config.n_head;
  const ch = (k) => `«${visibleChar(f.chars[k])}»`;
  const head = `head ${f.head} del blocco ${f.block}`;
  const future = (r, c) => c > r;
  const terms = topK(D.att.subarray(i * T, i * T + i + 1), 3);
  const blk = f.trace.blocks[f.block];
  const concatDiff = maxAbsDiff(D.concat, blk.concat);
  const outDiff = maxAbsDiff(D.out, D.engine.out);

  // proj: la riga i della concatenazione (carta sopra W, una colonna per numero) → colonna d'uscita
  const concatRect = blockAbove(P, { rows: T, gap: LABEL_CLEARANCE });
  const projOut = columnRight(P, { after: PB });
  const xIn = rowOf(f.blockInput, i, C);
  const proj = rowOf(blk.proj, i, C);
  const resid = rowOf(blk.resid1, i, C);
  const projRecomputed = linearForward(rowOf(D.concat, i, C), f.weight(f.names.proj.weight), f.weight(f.names.proj.bias), C, C);
  const residRecomputed = addVectors(xIn, projRecomputed);
  const residDiff = maxAbsDiff(residRecomputed, resid);
  // il residuo: tre righe sopra la concatenazione (x, + sa, = la somma), allineate alle stesse colonne
  const rowAt = (level) => cardRect({ x0: P.x0, y1: concatRect.y1 + 0.16 + (3 - level) * (TEMP_SIDE + CARD_GAP * 1.4), rows: 1, cols: C, cellW: P.cellW, cellH: TEMP_SIDE });
  const resRows = [rowAt(0), rowAt(1), rowAt(2)];
  const parts = linearParts(stage, { W: P, B: PB, input: concatRect, products: null, output: projOut, columns: [projOut], all: null }, { x: rowOf(D.concat, i, C), W: f.weight(f.names.proj.weight), b: f.weight(f.names.proj.bias), y: proj, names: f.names.proj, inDim: C, outDim: C, outLabel: "sa" });
  const allV = unionRect(A.scores, A.v, A.out);

  const weightsCard = (o = {}) => stage.card("av-weights", { rect: A.scores, data: D.att, label: "pesi di attention", labelSide: "bottom", accent: CARD_COLORS.probs, colormap: "attention", range: 1, maskedText: "0", mask: future, ...o });
  const vCard = (o = {}) => stage.card("av-v", { rect: A.v, data: D.v, label: "V", labelSide: "top", accent: CARD_COLORS.input, ...o });
  const outCard = (o = {}) => stage.card("av-out", { rect: A.out, data: D.out, label: "out = att · V", labelSide: "bottom", accent: CARD_COLORS.output, ...o });
  const concatCard = (o = {}) => stage.card("av-concat", { rect: concatRect, data: D.concat, label: "concat delle head", labelSide: "top", accent: CARD_COLORS.output, ...o });
  const vStrip = () => {
    const texts = f.chars.map(visibleChar);
    stage.strip("av-v-rows", texts, texts.map((_, k) => ({ x: A.v.x0 - 0.03, y: cellCenter(A.v, k, 0).y })), { height: Math.min(0.034, A.v.cellH * 0.85), marked: null });
    stage.strip("av-rows", texts, texts.map((_, k) => ({ x: A.scores.x0 - 0.03, y: cellCenter(A.scores, k, 0).y })), { height: Math.min(0.034, A.scores.cellH * 0.85), marked: i });
  };

  return [
    {
      title: "La media pesata dei value",
      caption: `L'uscita della head per ${ch(i)} è una media dei value dei token che guarda, pesata dall'attention: chi riceve più attenzione conta di più (come un riassunto che cita di più le fonti più rilevanti).`,
      duration: 6,
      code: codeFor("weightedSum", head),
      scene: () => {
        weightsCard().mark([{ r0: i, r1: i + 1, c0: 0, c1: i + 1 }]);
        vCard().mark(terms.map((e) => ({ r0: e.id, r1: e.id + 1, c0: 0, c1: hs })));
        outCard({ reveal: () => false });
        vStrip();
      },
      board: () => [
        { title: `out[${ch(i)}] = Σⱼ att[${i}, j] · v[j]`, kicker: "Attention · V" },
        { formula: terms.flatMap((e, k) => [k ? " + " : "", { text: num(e.value, 3), tone: "accent" }, ` · v[${ch(e.id)}]`]).concat(i + 1 > terms.length ? [" + …"] : []) },
        { line: [`out[${ch(i)}] = `, { text: vectorPreview(rowOf(D.out, i, hs), { count: 4 }), tone: "num" }, ` (${hs} numeri)`] },
      ],
      update: (() => {
        const step = onChange((done) => {
          const card = stage.cards.get("av-out");
          card.reveal(done ? (r) => r === i : () => false);
          card.mark(done ? [{ r0: i, r1: i + 1, c0: 0, c1: hs }] : null);
        });
        return (p) => step(p > 0.6);
      })(),
      frame: () => allV,
    },
    {
      title: "Tutte le posizioni",
      caption: `Lo stesso per ogni token: ${T} righe, una media pesata ciascuna. La riga 0 (il primo token) può guardare solo sé stessa: la sua uscita è il suo value.`,
      duration: 5,
      code: codeFor("weightedSum", head),
      scene: () => {
        weightsCard();
        vCard();
        outCard({ reveal: 0 });
        vStrip();
      },
      board: () => [
        { title: `out (${T} × ${hs}) della ${head}`, kicker: "Attention · V" },
        { line: ["Riga in corso: ", { text: "", tone: "num", live: "out" }] },
        { note: outDiff === 0 ? "Uguale all'uscita della head nel motore JS, bit per bit." : `Scarto massimo dal motore: ${num(outDiff, 8)}.` },
      ],
      update: (() => {
        const step = onChange((r) => {
          stage.cards.get("av-out").reveal(r * hs);
          stage.cards.get("av-weights").mark(r < T ? [{ r0: r, r1: r + 1, c0: 0, c1: r + 1 }] : null);
          stage.board.setLive("out", r < T ? `${r + 1} di ${T} (${ch(r)})` : `tutte le ${T}`);
        });
        return (p) => step(progressCount(p, T));
      })(),
      frame: () => allV,
    },
    {
      title: `Le ${H} head insieme`,
      caption: `Le ${H} head lavorano in parallelo, ognuna con le sue Wq, Wk, Wv. torch.cat mette le loro uscite una accanto all'altra: ${H} × ${hs} = ${C} numeri per token, sopra la W di proj.`,
      duration: 5,
      code: codeFor("concat", `blocco ${f.block}`),
      scene: () => {
        const card = concatCard({ reveal: 0 });
        card.mark(Array.from({ length: H }, (_, h) => ({ r0: 0, r1: T, c0: h * hs, c1: (h + 1) * hs })));
      },
      board: () => [
        { title: `concat (${T} × ${C}): le ${H} head affiancate`, kicker: "Multi-head" },
        { line: [`La head h occupa le colonne da h·${hs} a h·${hs} + ${hs - 1}. Head pronte: `, { text: "", tone: "accent", live: "heads" }] },
        { note: concatDiff === 0 ? "Uguale alla concatenazione del motore JS." : `Scarto dal motore: ${num(concatDiff, 8)}.` },
      ],
      update: (() => {
        const step = onChange((k) => {
          stage.cards.get("av-concat").reveal((r, c) => Math.floor(c / hs) < k);
          stage.board.setLive("heads", `${k} di ${H}`);
        });
        return (p) => step(progressCount(p, H));
      })(),
      frame: () => unionRect(concatRect, P, PB),
    },
    {
      title: "proj: rimescolare le head",
      caption: `La riga di ${ch(i)} nella concatenazione passa per proj (${C} × ${C} con bias): ogni uscita mescola i numeri di tutte le head. Le head si sono divise il lavoro, proj lo rimette insieme.`,
      duration: Math.min(7, Math.max(4, C * 0.08)),
      code: codeFor("proj", `blocco ${f.block}`),
      scene: () => {
        concatCard().mark([{ r0: i, r1: i + 1, c0: 0, c1: C }]);
        stage.card("lin-output", { rect: projOut, data: proj, label: "sa(x)", labelSide: "top", accent: CARD_COLORS.output, reveal: 0 });
      },
      board: () => [
        { title: `sa(x) = concat[${ch(i)}] · Wᵀ + b`, kicker: "proj" },
        { line: ["Riga in corso: ", { text: "", tone: "num", live: "row" }] },
        { note: "Il Dropout dopo proj nel training del corso ha p = 0 (config.dropout): non spegne niente." },
      ],
      update: (() => {
        let sweep = null;
        return (p) => {
          sweep ??= parts.sweepRows(stage.cards.get("lin-output"), null);
          sweep(progressCount(p, C));
        };
      })(),
      frame: () => unionRect(concatRect, P, PB, projOut),
    },
    {
      title: "Il residuo: x + sa(ln1(x))",
      caption: `L'uscita dell'attention non sostituisce x: si SOMMA a x, numero per numero (connessione residua). Come aggiungere note a margine invece di riscrivere il quaderno: quello che c'era resta.`,
      duration: 6,
      code: codeFor("residualAttention", `blocco ${f.block}`),
      scene: () => {
        stage.card("res-x", { rect: resRows[0], data: xIn, label: "x", labelSide: "left", accent: CARD_COLORS.input });
        stage.card("res-sa", { rect: resRows[1], data: proj, label: "+ sa(x)", labelSide: "left", accent: CARD_COLORS.product });
        stage.card("res-sum", { rect: resRows[2], data: resid, label: "= x", labelSide: "left", accent: CARD_COLORS.output, reveal: 0 });
      },
      board: () => [
        { title: `x + sa(ln1(x)) per ${ch(i)}`, kicker: "Residuo" },
        { formula: [`x[0] + sa[0] = `, { text: num(xIn[0], 3), tone: "num" }, " + ", { text: factor(proj[0], 3), tone: "num" }, " = ", { text: num(resid[0], 3), tone: "good" }] },
        { formula: [`x[1] + sa[1] = `, { text: num(xIn[1], 3), tone: "num" }, " + ", { text: factor(proj[1], 3), tone: "num" }, " = ", { text: num(resid[1], 3), tone: "good" }] },
        { note: residDiff === 0 ? `… e così per tutti i ${C} numeri: uguale al flusso residuo del motore JS, bit per bit.` : `Scarto dal motore: ${num(residDiff, 8)}.` },
      ],
      update: (() => {
        const step = onChange((k) => stage.cards.get("res-sum").reveal(k));
        return (p) => step(progressCount(p, C, { hold: 0.35 }));
      })(),
      frame: () => unionRect(...resRows),
    },
  ];
}

// ---------------------------------------------------------------- aiuti

/** Che cosa fa la maschera su una riga: per l'ultimo token niente (vede tutto), quindi si cita anche una riga a metà. */
function maskExample(i, T, ch) {
  if (i < T - 1) return `Dopo la maschera la riga di ${ch(i)} tiene ${i + 1} punteggi su ${T}: i token da 0 a ${i}; gli altri ${T - 1 - i} sono −∞.`;
  const m = Math.max(0, Math.floor((T - 1) / 2));
  return `${ch(i)} è l'ultimo token: vede tutto il prompt e la sua riga resta intera. La riga di ${ch(m)} (posizione ${m}) invece tiene ${m + 1} punteggi su ${T}: gli altri ${T - 1 - m} sono −∞.`;
}

function transpose(m, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = m[r * cols + c];
  return out;
}

function maxAbs(a) {
  let m = 0;
  for (let k = 0; k < a.length; k++) if (Number.isFinite(a[k])) m = Math.max(m, Math.abs(a[k]));
  return m || 1;
}

function sumRow(att, T, i) {
  let s = 0;
  for (let j = 0; j < T; j++) s += att[i * T + j];
  return s;
}

/** Scarto massimo contando uguali anche i −∞ nella stessa cella (i punteggi mascherati). */
function maxAbsDiffFinite(a, b) {
  let m = 0;
  for (let k = 0; k < a.length; k++) {
    if (a[k] === b[k]) continue;
    m = Math.max(m, Math.abs(a[k] - b[k]));
  }
  return m;
}
