/**
 * Azione «Feed-forward: 64 → 256, ReLU, 256 → 64» (gpt/feedforward.py) sul token scelto: il primo
 * strato accende i neuroni nascosti (colonna accanto a W₁, una cella per riga), la ReLU spegne i negativi
 * (gli stessi 256 numeri anche in due griglie 16 × 16 accanto al nodo della ReLU: si vede quanti si
 * spengono), il secondo strato riceve quei numeri come riga sopra W₂ e torna a 64; poi il residuo.
 * Pesi della scena, attivazioni del motore.
 */
import { CARD_COLORS } from "./cards.js";
import { codeFor } from "./code.js";
import { ffnData } from "./data.js";
import { factor, int, num, pct } from "./format.js";
import { linearLayout, linearParts } from "./linearKit.js";
import { maxAbsDiff } from "./math.js";
import { CARD_GAP, TEMP_SIDE, cardRect, unionRect } from "./plan.js";
import { visibleChar } from "./settings.js";
import { onChange, progressCount } from "./stepKit.js";

export function ffnSteps(stage, f) {
  const S = stage.scene;
  const N = f.names;
  const D = ffnData(f);
  const G1 = linearLayout(S, N.ffn1.weight, N.ffn1.bias, { outputs: 2 });
  const G2 = linearLayout(S, N.ffn2.weight, N.ffn2.bias);
  if (!G1 || !G2) return null;
  const C = f.C;
  const H4 = f.hidden;
  const token = `«${visibleChar(f.chars[f.t])}»`;
  const preCol = G1.columns[0];
  const postCol = G1.columns[1];
  const reluBox = S.nodeOf(N.relu);
  const W1 = f.weight(N.ffn1.weight);
  const b1 = f.weight(N.ffn1.bias);
  const W2 = f.weight(N.ffn2.weight);
  const b2 = f.weight(N.ffn2.bias);
  const p1 = linearParts(stage, G1, { x: D.x, W: W1, b: b1, y: D.pre, names: N.ffn1, inDim: C, outDim: H4, inLabel: "x", outLabel: "h" });
  const p2 = linearParts(stage, G2, { x: D.post, W: W2, b: b2, y: D.out, names: N.ffn2, inDim: H4, outDim: C, inLabel: "ReLU(h)", outLabel: "y" });
  const diffs = { pre: maxAbsDiff(D.pre, D.engine.pre), post: maxAbsDiff(D.post, D.engine.post), out: maxAbsDiff(D.out, D.engine.out) };
  // il residuo: tre righe sopra l'ingresso del secondo strato (x, + feed-forward, = la somma)
  const step = TEMP_SIDE + CARD_GAP * 1.4;
  const resRow = (level) => cardRect({ x0: G2.W.x0, y1: G2.input.y1 + 0.2 + (3 - level) * step, rows: 1, cols: C, cellW: G2.W.cellW, cellH: TEMP_SIDE });
  const res = [resRow(0), resRow(1), resRow(2)];
  const hiddenRange = Math.max(...D.pre.map(Math.abs));
  // le due griglie (h e ReLU(h)): gli stessi numeri della colonna, a righe di `side`, sopra il nodo della ReLU
  const side = Math.ceil(Math.sqrt(H4));
  const gridRows = Math.ceil(H4 / side);
  const anchor = reluBox ?? { x0: postCol.x1 + 0.2, x1: postCol.x1 + 1.6, y0: G1.W.y1 - 1, y1: G1.W.y1 - 0.6, cx: postCol.x1 + 0.9 };
  const gridW = side * 0.04;
  const preGrid = cardRect({ x0: (anchor.x0 + anchor.x1) / 2 - gridW - 0.06, y1: anchor.y1 + 0.18 + gridRows * 0.04, rows: gridRows, cols: side });
  const postGrid = cardRect({ x0: (anchor.x0 + anchor.x1) / 2 + 0.06, y1: preGrid.y1, rows: gridRows, cols: side });
  const padded = (values) => {
    const out = new Float32Array(gridRows * side);
    out.set(values.subarray(0, H4));
    return out;
  };
  const outside = (r, c) => r * side + c >= H4; // celle oltre l'ultimo neurone (se H4 non è un quadrato)
  const strongest = topIndex(D.post);

  const where = `blocco ${f.block}`;
  const preCard = (o = {}) => stage.card("ffn-pre", { rect: preCol, data: D.pre, label: "h", labelSide: "top", accent: CARD_COLORS.output, range: hiddenRange, ...o });
  const postCard = (o = {}) => stage.card("ffn-post", { rect: postCol, data: D.post, label: "ReLU(h)", labelSide: "top", accent: CARD_COLORS.product, range: hiddenRange, ...o });
  const sweep1 = (() => {
    let sweep = null;
    return (p) => {
      sweep ??= p1.sweepRows(stage.cards.get("ffn-pre"), null);
      sweep(progressCount(p, H4));
    };
  })();
  const sweep2 = (() => {
    let sweep = null;
    return (p) => {
      sweep ??= p2.sweepRows(stage.cards.get("lin-output"), null);
      sweep(progressCount(p, C));
    };
  })();

  return [
    {
      title: "Il vettore x, dopo ln2",
      caption: `Dopo che i token si sono parlati (attention), ognuno elabora da solo quello che ha raccolto. Il token ${token} entra nella feed-forward come x di ${C} numeri.`,
      duration: 5,
      code: codeFor("ffnForward", where),
      scene: () => {
        p1.input();
        preCard({ reveal: 0 });
      },
      board: () => [
        { title: `x = ln2(x) · token ${token}`, kicker: "Feed-forward" },
        { line: [`nn.Sequential: Linear ${C} → ${H4}, ReLU, Linear ${H4} → ${C}, Dropout. Lo strato in mezzo è 4 volte più largo: più spazio per combinare le informazioni.`] },
        { vector: { values: D.x, max: 16, caption: "x (i primi 16):" } },
      ],
      frame: () => unionRect(G1.input, G1.products, G1.W, G1.B, preCol),
    },
    {
      title: `Primo strato: ${C} → ${H4}`,
      caption: `Ogni riga di W₁ è un neurone nascosto: guarda x e dà un punteggio (più la sua bias). ${H4} righe, ${H4} neuroni, come ${H4} rilevatori di caratteristiche diverse.`,
      duration: 7,
      code: codeFor("ffn1", where),
      scene: () => {
        p1.input();
        preCard({ reveal: 0 });
      },
      board: () => [
        { title: `h = x · W₁ᵀ + b₁ (${H4} neuroni)`, kicker: "Feed-forward" },
        { line: ["Riga in corso: ", { text: "", tone: "accent", live: "row" }] },
        { note: `${int(H4)} × ${int(C)} = ${int(H4 * C)} moltiplicazioni per un token.` },
      ],
      update: sweep1,
      frame: () => unionRect(G1.input, G1.W, G1.B, preCol),
    },
    {
      title: "ReLU: i negativi si spengono",
      caption: `ReLU(h) = max(0, h): ogni neurone con un punteggio negativo si spegne (0). Qui si spengono ${D.off} neuroni su ${H4} (${pct(D.off / H4, 0)}). Senza questa curva, due nn.Linear di fila sarebbero una sola nn.Linear.`,
      duration: 6,
      code: codeFor("relu", where),
      scene: () => {
        preCard();
        postCard({ reveal: 0 });
        stage.card("ffn-pre-grid", { rect: preGrid, data: padded(D.pre), label: `h: ${H4} = ${gridRows} × ${side}`, labelSide: "top", accent: CARD_COLORS.output, range: hiddenRange, mask: outside });
        stage.card("ffn-post-grid", { rect: postGrid, data: padded(D.post), label: "ReLU(h)", labelSide: "top", accent: CARD_COLORS.product, range: hiddenRange, mask: outside, reveal: 0 });
        if (reluBox) stage.outline(reluBox, { pad: 0.03 });
      },
      board: () => [
        { title: "ReLU(h) = max(0, h)", kicker: "Feed-forward" },
        { formula: [`h[${strongest}] = `, { text: num(D.pre[strongest], 4), tone: "num" }, " → ", { text: num(D.post[strongest], 4), tone: "good" }, "  (acceso: il più forte)"] },
        { line: ["Spenti: ", { text: `${D.off} di ${H4}`, tone: "accent" }, ` · accesi: ${D.on}. Neuroni spenti finora: `, { text: "", tone: "num", live: "off" }] },
      ],
      update: (() => {
        const stepFn = onChange((k) => {
          stage.cards.get("ffn-post").reveal(k);
          stage.cards.get("ffn-post-grid")?.reveal(k);
          let off = 0;
          for (let n = 0; n < k; n++) if (!(D.pre[n] > 0)) off++;
          stage.board.setLive("off", `${off} (dopo ${k} neuroni)`);
        });
        return (p) => stepFn(progressCount(p, H4, { hold: 0.25 }));
      })(),
      frame: () => unionRect(preGrid, postGrid, reluBox),
    },
    {
      title: `Secondo strato: ${H4} → ${C}`,
      caption: `I ${H4} neuroni (spenti compresi: valgono 0) diventano la riga d'ingresso di W₂, sopra le sue colonne. Ogni riga di W₂ ne fa una somma pesata: ${C} numeri, la dimensione del flusso residuo.`,
      duration: 6,
      code: codeFor("ffn2", where),
      scene: () => {
        p2.input();
        p2.output({ reveal: 0 });
      },
      board: () => [
        { title: `y = ReLU(h) · W₂ᵀ + b₂ (${C} numeri)`, kicker: "Feed-forward" },
        { line: ["Riga in corso: ", { text: "", tone: "accent", live: "row" }] },
        { note: `Dropout dopo il secondo strato: nel training del corso p = ${f.config.dropout ?? 0}, quindi non spegne niente.` },
      ],
      update: sweep2,
      frame: () => unionRect(G2.input, G2.W, G2.B, G2.output),
    },
    {
      title: "Il residuo: x + ffwd(ln2(x))",
      caption: `Come per l'attention, l'uscita si somma al flusso residuo: è l'uscita del blocco ${f.block}, uguale al motore JS${diffs.out === 0 && diffs.pre === 0 ? " bit per bit" : ""}.`,
      duration: 6,
      code: codeFor("residualFfn", where),
      scene: () => {
        stage.card("res-x", { rect: res[0], data: D.residIn, label: "x", labelSide: "left", accent: CARD_COLORS.input });
        stage.card("res-ffn", { rect: res[1], data: D.out, label: "+ ffwd(x)", labelSide: "left", accent: CARD_COLORS.product });
        stage.card("res-sum", { rect: res[2], data: D.resid, label: "= x", labelSide: "left", accent: CARD_COLORS.output, reveal: 0 });
      },
      board: () => [
        { title: `x + ffwd(ln2(x)) per ${token}`, kicker: "Residuo" },
        { formula: ["x[0] + y[0] = ", { text: num(D.residIn[0], 3), tone: "num" }, " + ", { text: factor(D.out[0], 3), tone: "num" }, " = ", { text: num(D.resid[0], 3), tone: "good" }] },
        { note: diffs.pre === 0 && diffs.post === 0 && diffs.out === 0 ? "Controllo: h, ReLU(h) e y uguali al motore JS, bit per bit." : `Controllo: scarti dal motore ${num(diffs.pre, 8)} / ${num(diffs.post, 8)} / ${num(diffs.out, 8)}.` },
      ],
      update: (() => {
        const stepFn = onChange((k) => stage.cards.get("res-sum").reveal(k));
        return (p) => stepFn(progressCount(p, C, { hold: 0.35 }));
      })(),
      frame: () => unionRect(...res),
    },
  ];
}

function topIndex(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
