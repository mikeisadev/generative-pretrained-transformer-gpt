/**
 * Una nn.Linear nella scena, per tutte le azioni che ne mostrano una (x · Wᵀ, proj, la feed-forward,
 * lm_head, il passo di training): dove stanno x, i prodotti e l'uscita rispetto alla W VERA della scena,
 * e i pezzi di passo che si ripetono (l'ingresso, una riga colonna per colonna, la somma, tutte le righe).
 *
 * Disposizione (come lo schema classico della moltiplicazione):
 *          [ x            ]   ← l'ingresso, x[j] sopra la colonna j di W
 *          [ x ⊙ W[d, :]  ]   ← i prodotti della riga d
 *          [ W (scena)    ] [b] [y]   ← la bias della scena (se c'è) e l'uscita: y[d] accanto alla riga d
 */
import { CARD_COLORS } from "./cards.js";
import { CARD_GAP, TEMP_SIDE, cardRect, cellsRect, columnRight, unionRect } from "./plan.js";
import { factor, num, signed } from "./format.js";
import { linearRow, runningSums } from "./math.js";
import { onChange, progressCount } from "./stepKit.js";

/** Spazio tra le righe temporanee e W: le carte coprono l'etichetta "weight (…)" (sono opache). */
const ROW_GAP = 0.06;

/** Il rettangolo delle celle di un tensore della scena (null se la scena non lo disegna). */
export function sceneRect(S, name) {
  const w = S.layoutOf(name);
  if (!w) return null;
  return { ...w.rect, rows: w.rows, cols: w.cols, cellW: w.cellW, cellH: w.cellH };
}

/**
 * Dove stanno le carte di una nn.Linear (pesi `weight`, bias `bias` o null) nella scena.
 * `outputLevel`: quante colonne di uscita (logits, probabilità, gradiente…) si affiancano a destra.
 */
export function linearLayout(S, weight, bias = null, { outputs = 1 } = {}) {
  const W = sceneRect(S, weight);
  if (!W) return null;
  const B = bias ? sceneRect(S, bias) : null;
  const products = cardRect({ x0: W.x0, y1: W.y1 + ROW_GAP + TEMP_SIDE, rows: 1, cols: W.cols, cellW: W.cellW, cellH: TEMP_SIDE });
  const input = cardRect({ x0: W.x0, y1: products.y1 + CARD_GAP + TEMP_SIDE, rows: 1, cols: W.cols, cellW: W.cellW, cellH: TEMP_SIDE });
  const columns = Array.from({ length: outputs }, (_, level) => columnRight(W, { after: B, level }));
  return { W, B, input, products, output: columns[0], columns, all: unionRect(input, products, W, B, ...columns) };
}

/** La finestra di numeri leggibili: le prime `n` colonne di x, dei prodotti e della riga d di W. */
export function windowRect(G, d, c0, n) {
  const c1 = Math.min(G.W.cols, c0 + n);
  return unionRect(cellsRect(G.input, 0, 1, c0, c1), cellsRect(G.products, 0, 1, c0, c1), cellsRect(G.W, d, d + 1, c0, c1));
}

/** Etichette delle righe di un'uscita (i caratteri per lm_head, gli indici per le altre). */
export function outputLabels(n, vocab, isVocab) {
  return Array.from({ length: n }, (_, i) => (isVocab && vocab?.[i] !== undefined ? vocab[i] : String(i)));
}

/**
 * I pezzi comuni di una nn.Linear su un token, legati a una disposizione G e ai dati veri:
 *   x, W, b (Float32Array), y = l'uscita del motore, names = { weight, bias }.
 * Restituisce funzioni che mettono le carte nella scena (per gli scene() dei passi) e i testi.
 */
export function linearParts(stage, G, { x, W, b, y, names, inDim, outDim, inLabel = "x", outLabel = "y", outRange = null, rowLabel = null }) {
  const S = stage.scene;
  const productsOf = (d) => linearRow(x, W, b, inDim, d);
  return {
    G,
    productsOf,
    /** La carta dell'ingresso x (riga sopra W). */
    input(options = {}) {
      return stage.card("lin-input", { rect: G.input, data: x, label: inLabel, labelSide: "left", accent: CARD_COLORS.input, ...options });
    },
    /** I prodotti della riga d (riga tra x e W). */
    products(d, options = {}) {
      return stage.card("lin-products", { rect: G.products, data: productsOf(d).products, label: `x·W[${d}]`, labelSide: "left", accent: CARD_COLORS.product, range: "auto", ...options });
    },
    /** L'uscita y (colonna a destra di W o della bias). */
    output(options = {}) {
      return stage.card("lin-output", { rect: G.output, data: y, label: outLabel, labelSide: "top", accent: CARD_COLORS.output, range: outRange, rowLabel, ...options });
    },
    /** Velo su W tranne la riga d (e la cella della bias, se c'è). */
    highlightRow(d) {
      stage.highlight(names.weight, { rows: [d] });
      if (names.bias) stage.highlight(names.bias, [{ i: d, j: 0 }]);
    },
    /** La linea dalla fine della riga d di W (e dalla bias) alla cella d dell'uscita. */
    linkRow(d, outCard) {
      const yRow = G.W.y1 - (d + 0.5) * G.W.cellH;
      const to = outCard.cellCenter(d, 0);
      stage.link([[{ x: G.W.x1, y: yRow, z: to.z }, { x: to.x - G.output.cellW / 2, y: to.y, z: to.z }]]);
    },
    /** Le righe della lavagna con i primi prodotti scritti per esteso. */
    productLines(d, count = 3) {
      const r = productsOf(d);
      const lines = [];
      for (let j = 0; j < Math.min(count, inDim); j++) {
        lines.push({ formula: [`x[${j}] · W[${d}, ${j}] = `, { text: factor(x[j]), tone: "num" }, " · ", { text: factor(W[d * inDim + j]), tone: "num" }, " = ", { text: num(r.products[j]), tone: r.products[j] >= 0 ? "pos" : "neg" }] });
      }
      if (inDim > count) lines.push({ note: `… e così per tutte le ${inDim} colonne: un prodotto per colonna.` });
      return lines;
    },
    /** La somma della riga d con la bias, scritta per esteso con i numeri veri. */
    sumLines(d, { live = false } = {}) {
      const r = productsOf(d);
      const head = r.products.slice(0, 2);
      const lines = [
        { formula: [`${outLabel}[${d}] = Σⱼ x[j] · W[${d}, j] = `, { text: num(head[0]), tone: "num" }, ` ${signed(head[1]).replace(/^([+−])/, "$1 ")} + … = `, { text: num(r.sum), tone: "accent", live: live ? "sum" : undefined }] },
      ];
      if (b) lines.push({ formula: [`+ b[${d}] = `, { text: signed(r.bias), tone: "num" }, " → ", `${outLabel}[${d}] = `, { text: num(r.value), tone: "good" }] });
      else lines.push({ note: "bias=False: niente bias, il risultato è la somma." });
      return lines;
    },
    /** update(p) che scorre la riga d colonna per colonna: prodotti scoperti e somma che cresce sulla lavagna. */
    sweepColumns(d, card) {
      const r = productsOf(d);
      const sums = runningSums(r.products);
      return onChange((k) => {
        card.reveal(k);
        card.mark(k > 0 && k <= inDim ? [{ r0: 0, r1: 1, c0: k - 1, c1: k }] : null);
        if (k > 0 && k <= inDim) S.hoverCell(names.weight, { row: d, col: k - 1 });
        stage.board.setLive("sum", k >= inDim ? num(r.sum) : `${num(sums[k])} (dopo ${k} prodotti su ${inDim})`);
      });
    },
    /** update(p) che passa tutte le righe: velo sulla riga, prodotti della riga, uscita che si riempie. */
    sweepRows(outCard, productsCard, { keep = null } = {}) {
      return onChange((r) => {
        const row = Math.min(outDim - 1, r);
        outCard.reveal((i) => i < r || i === keep);
        if (r < outDim) {
          this.highlightRow(row);
          productsCard?.setData(productsOf(row).products, { range: "auto" });
          outCard.mark([{ r0: row, r1: row + 1, c0: 0, c1: 1 }]);
          this.linkRow(row, outCard);
          stage.board.setLive("row", `${outLabel}[${row}] = ${num(y[row])}`);
        } else {
          outCard.mark(null);
          stage.clearLinks();
          stage.board.setLive("row", `tutte le ${outDim} righe: fatto`);
        }
      });
    },
    progressCount,
  };
}
