/**
 * Azione «LayerNorm: media, varianza, γ e β» (nn.LayerNorm di ln1 del blocco scelto, sul token scelto):
 * sopra la riga γ della scena x, x − μ e x normalizzato (le carte coprono l'intestazione del modulo:
 * la riga di codice è nel pannello del codice); γ e β veri evidenziati; sotto β x̂ · γ e l'uscita.
 * Tutto allineato alle colonne di γ: la dimensione j sta sempre lì. Da vicino i numeri si leggono.
 */
import { CARD_COLORS } from "./cards.js";
import { codeFor } from "./code.js";
import { layerNormData } from "./data.js";
import { factor, num, sci } from "./format.js";
import { sceneRect } from "./linearKit.js";
import { maxAbsDiff, meanVariance } from "./math.js";
import { CARD_GAP, LABEL_CLEARANCE, TEMP_SIDE, cardRect, cellsRect, readableColumns, unionRect } from "./plan.js";
import { visibleChar } from "./settings.js";
import { onChange, progressCount } from "./stepKit.js";

/** Le righe sopra γ e sotto β, allineate alle colonne di γ (riga 0 = la più in alto). */
function layerNormLayout(S, data) {
  const gamma = sceneRect(S, data.names.weight);
  const beta = sceneRect(S, data.names.bias);
  const module = S.nodeOf(data.names.weight.replace(/\.weight$/, ""));
  if (!gamma || !beta || !module) return null;
  const row = (y1) => cardRect({ x0: gamma.x0, y1, rows: 1, cols: gamma.cols, cellW: gamma.cellW, cellH: TEMP_SIDE });
  const step = TEMP_SIDE + CARD_GAP;
  const above = (level) => row(gamma.y1 + LABEL_CLEARANCE + TEMP_SIDE + (2 - level) * step);
  const below = (level) => row(beta.y0 - CARD_GAP - level * step);
  return { gamma, beta, module, x: above(0), centered: above(1), normalized: above(2), scaled: below(0), out: below(1) };
}

/** Le prime n dimensioni di alcune righe: la finestra in cui si leggono i numeri. */
function columnsWindow(rows, n) {
  return unionRect(...rows.map((r) => cellsRect(r, 0, 1, 0, Math.min(r.cols, n))));
}

export function layerNormSteps(stage, f, which = "ln1") {
  const S = stage.scene;
  const data = layerNormData(f, which);
  const G = layerNormLayout(S, data);
  if (!G) return null;
  const s = data.steps;
  const C = f.C;
  const x = data.input;
  const token = `«${visibleChar(f.chars[f.t])}»`;
  const before = meanVariance(x);
  const after = meanVariance(s.normalized);
  const diff = maxAbsDiff(s.out, data.output);
  const code = codeFor(data.code, data.label);
  const all = unionRect(G.x, G.out);
  // quante dimensioni leggibili stanno nella parte libera (con le etichette a sinistra)
  const readable = () => readableColumns(stage.freeArea().width, { minCellPx: 54, min: 4, max: 14, labelPx: Math.max(0, ...["ln-x", "ln-centered", "ln-norm"].map((k) => stage.cards.get(k)?.labelReservePx() ?? 0)) });
  const xCard = () => stage.card("ln-x", { rect: G.x, data: x, label: "x", labelSide: "left", accent: CARD_COLORS.input });
  const centeredCard = (o = {}) => stage.card("ln-centered", { rect: G.centered, data: s.centered, label: "x − μ", labelSide: "left", accent: CARD_COLORS.product, ...o });
  const normCard = (o = {}) => stage.card("ln-norm", { rect: G.normalized, data: s.normalized, label: "x̂", labelSide: "left", accent: CARD_COLORS.product, ...o });
  const scaledCard = (o = {}) => stage.card("ln-scaled", { rect: G.scaled, data: s.scaled, label: "x̂ · γ", labelSide: "left", accent: CARD_COLORS.product, ...o });
  const outCard = (o = {}) => stage.card("ln-out", { rect: G.out, data: s.out, label: "y", labelSide: "left", accent: CARD_COLORS.output, ...o });
  const reveal = (key) => {
    const step = onChange((k) => stage.cards.get(key)?.reveal(k));
    return (p) => step(progressCount(p, C, { hold: 0.3 }));
  };

  return [
    {
      title: "Il vettore x",
      caption: `Il token ${token} entra in ${data.label} con ${C} numeri di scala qualsiasi. La LayerNorm li riporta tutti alla stessa scala, come si fa con i voti di classi diverse prima di confrontarli.`,
      duration: 5,
      code,
      scene: () => xCard(),
      board: () => [
        { title: `${data.label} · token ${token}`, kicker: "LayerNorm" },
        { vector: { values: x, max: 16, caption: `x (i primi 16 di ${C}): media ${num(before.mean, 3)}, varianza ${num(before.variance, 3)}.` } },
        { line: ["nn.LayerNorm(n_embd) ha due vettori appresi: γ (weight) e β (bias), un numero per dimensione: sono le due righe nel modulo qui sotto."] },
      ],
      frame: () => all,
    },
    {
      title: "La media μ",
      caption: `Prima la media dei ${C} numeri; poi a ciascuno si toglie la media: x − μ è centrato sullo zero.`,
      duration: 5,
      code,
      scene: () => {
        xCard();
        centeredCard({ reveal: 0 });
      },
      board: () => [
        { title: "μ = (x[0] + x[1] + … ) / " + C, kicker: "LayerNorm" },
        { formula: ["μ = (", { text: num(x[0], 3), tone: "num" }, " + ", { text: factor(x[1], 3), tone: "num" }, ` + … ) / ${C} = `, { text: num(s.mean, 5), tone: "accent" }] },
        { formula: ["x[0] − μ = ", { text: num(x[0], 4), tone: "num" }, " − ", { text: factor(s.mean, 4), tone: "num" }, " = ", { text: num(s.centered[0], 4), tone: "good" }] },
      ],
      update: reveal("ln-centered"),
      frame: () => columnsWindow([G.x, G.centered], readable()),
    },
    {
      title: "La varianza σ²",
      caption: "Quanto i numeri sono sparsi: la media dei quadrati di x − μ. Divisa per " + C + ", non " + (C - 1) + " (varianza «biased», come nn.LayerNorm). Si aggiunge un ε piccolissimo per non dividere mai per zero.",
      duration: 6,
      code,
      scene: () => {
        xCard();
        centeredCard();
      },
      board: () => [
        { title: "σ² = Σ (x[j] − μ)² / " + C, kicker: "LayerNorm" },
        { formula: ["σ² = (", { text: num(s.centered[0], 3), tone: "num" }, "² + ", { text: factor(s.centered[1], 3), tone: "num" }, ` ² + … ) / ${C} = `, { text: num(s.variance, 5), tone: "accent" }] },
        { formula: ["√(σ² + ε) = √(", num(s.variance, 5), " + ", sci(s.eps, 1), ") = ", { text: num(s.std, 5), tone: "accent" }] },
      ],
      frame: () => columnsWindow([G.x, G.centered], readable()),
    },
    {
      title: "Normalizzazione",
      caption: "Ogni numero di x − μ si divide per √(σ² + ε): ora media 0 e varianza 1. Token con numeri grandi e token con numeri piccoli diventano confrontabili.",
      duration: 5,
      code,
      scene: () => {
        xCard();
        centeredCard();
        normCard({ reveal: 0 });
      },
      board: () => [
        { title: "x̂ = (x − μ) / √(σ² + ε)", kicker: "LayerNorm" },
        { formula: ["x̂[0] = ", { text: num(s.centered[0], 4), tone: "num" }, " / ", num(s.std, 4), " = ", { text: num(s.normalized[0], 4), tone: "good" }] },
        { line: ["Controllo su x̂: media ", { text: sci(after.mean, 2), tone: "num" }, ", varianza ", { text: num(after.variance, 5), tone: "num" }, ` (1 a meno di ε).`] },
      ],
      update: reveal("ln-norm"),
      frame: () => columnsWindow([G.centered, G.normalized], readable()),
    },
    {
      title: "× γ: allargare o stringere",
      caption: "Ora i pesi: γ (la riga weight del modulo) moltiplica ogni dimensione. Il training ha imparato quanto ogni dimensione deve contare: γ grande = la allarga, γ vicino a zero = quasi la spegne.",
      duration: 5,
      code,
      scene: () => {
        normCard();
        scaledCard({ reveal: 0 });
        stage.outline(G.gamma, { pad: 0.015 });
      },
      board: () => [
        { title: "x̂ · γ, dimensione per dimensione", kicker: "LayerNorm" },
        { formula: ["x̂[0] · γ[0] = ", { text: num(s.normalized[0], 4), tone: "num" }, " · ", { text: factor(data.gamma[0], 4), tone: "num" }, " = ", { text: num(s.scaled[0], 4), tone: "good" }] },
        { formula: ["x̂[1] · γ[1] = ", { text: num(s.normalized[1], 4), tone: "num" }, " · ", { text: factor(data.gamma[1], 4), tone: "num" }, " = ", { text: num(s.scaled[1], 4), tone: "good" }] },
      ],
      update: reveal("ln-scaled"),
      frame: () => unionRect(G.normalized, G.gamma, G.scaled),
    },
    {
      title: "+ β: l'uscita",
      caption: `Infine β (la riga bias) sposta ogni dimensione. L'uscita y è quella che entra ${which === "ln1" ? "nell'attention" : which === "ln2" ? "nella feed-forward" : "in lm_head"}: uguale al motore JS${diff === 0 ? ", bit per bit" : ""}.`,
      duration: 6,
      code,
      scene: () => {
        scaledCard();
        outCard({ reveal: 0 });
        stage.outline(G.beta, { pad: 0.015 });
      },
      board: () => [
        { title: "y = x̂ · γ + β", kicker: "LayerNorm" },
        { formula: ["y[0] = ", { text: num(s.scaled[0], 4), tone: "num" }, " + ", { text: factor(data.beta[0], 4), tone: "num" }, " = ", { text: num(s.out[0], 4), tone: "good" }] },
        { vector: { values: s.out, max: 16, caption: "y (i primi 16):" } },
        { note: diff === 0 ? "Controllo: uguale all'uscita della LayerNorm nel motore JS, bit per bit." : `Controllo: scarto massimo dal motore ${num(diff, 8)}.` },
      ],
      update: reveal("ln-out"),
      frame: () => unionRect(G.gamma, G.scaled, G.out),
    },
  ];
}
