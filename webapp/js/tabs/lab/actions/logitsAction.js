/**
 * Azione «Logits e probabilità del prossimo carattere»: lm_head (la W vera della scena, una riga per
 * carattere) trasforma l'uscita di ln_f del token scelto in un punteggio per carattere; la softmax li
 * fa diventare probabilità; il carattere più probabile e quello che nel prompt viene davvero dopo.
 */
import { CARD_COLORS } from "./cards.js";
import { codeFor } from "./code.js";
import { OUTPUT_NAMES, logitsData } from "./data.js";
import { num, pct } from "./format.js";
import { linearLayout, linearParts } from "./linearKit.js";
import { topK } from "./math.js";
import { unionRect } from "./plan.js";
import { visibleChar } from "./settings.js";
import { onChange, progressCount } from "./stepKit.js";

/**
 * Un esempio di riga dal vocabolario VERO del run: «la riga 43 è «e»…» nel GPT del corso (una costante
 * scritta a mano sarebbe sbagliata per un altro vocabolario, e lo era anche per questo: la riga 12 è «?»).
 */
export function rowExample(vocab) {
  const list = Array.isArray(vocab) ? vocab : [];
  const id = list.indexOf("e") >= 0 ? list.indexOf("e") : list.findIndex((c) => /\p{L}/u.test(c));
  return id >= 0 ? `la riga ${id} è «${visibleChar(list[id])}»…` : "una riga per token";
}

export function logitsSteps(stage, f) {
  const S = stage.scene;
  const N = OUTPUT_NAMES.lmHead;
  const G = linearLayout(S, N.weight, N.bias, { outputs: 2 });
  if (!G) return null;
  const D = logitsData(f);
  const V = f.V;
  const C = f.C;
  const token = `«${visibleChar(f.chars[f.t])}»`;
  const chars = f.vocab.map(visibleChar);
  const rowLabel = (r) => chars[r];
  const best = topK(D.probs, 1)[0].id;
  const top = topK(D.probs, 8);
  const logitRange = Math.max(...D.logits.map(Math.abs));
  const maxProb = D.probs[best];
  const parts = linearParts(stage, G, { x: D.h, W: f.weight(N.weight), b: f.weight(N.bias), y: D.logits, names: N, inDim: C, outDim: V, inLabel: "x", outLabel: "logits", rowLabel });
  const probsCol = G.columns[1];
  const logitsCard = (o = {}) => stage.card("lin-output", { rect: G.output, data: D.logits, label: "logits", labelSide: "top", accent: CARD_COLORS.output, range: logitRange, rowLabel, ...o });
  const probsCard = (o = {}) => stage.card("lg-probs", { rect: probsCol, data: D.probs, label: "p", labelSide: "top", accent: CARD_COLORS.probs, colormap: "attention", range: maxProb, rowLabel, ...o });
  const strip = (marked = null) => stage.strip("lg-chars", chars, chars.map((_, r) => ({ x: probsCol.x1 + 0.05, y: probsCol.y1 - (r + 0.5) * probsCol.cellH })), { height: Math.min(0.034, probsCol.cellH * 0.85), marked });
  const nextText = D.next === null ? null : `«${visibleChar(f.vocab[D.next])}»`;
  const all = unionRect(G.input, G.W, G.B, G.output, probsCol);

  return [
    {
      title: "L'ultima tappa: ln_f",
      caption: `Dopo l'ultimo blocco e la LayerNorm finale, il token ${token} è un vettore di ${C} numeri. lm_head lo trasforma in un punteggio per ognuno dei ${V} caratteri del vocabolario.`,
      duration: 5,
      code: codeFor("lnF", "poi lm_head"),
      scene: () => {
        parts.input();
        logitsCard({ reveal: 0 });
        strip();
      },
      board: () => [
        { title: `x = ln_f(x) · token ${token}`, kicker: "Logits" },
        { line: [`lm_head ha ${V} righe, una per carattere (${rowExample(f.vocab)}): ogni riga dà il punteggio del suo carattere. Più il vettore del token «somiglia» alla riga, più il punteggio è alto.`] },
        { vector: { values: D.h, max: 16, caption: "x (i primi 16):" } },
      ],
      frame: () => all,
    },
    {
      title: `${V} punteggi: i logits`,
      caption: `Riga per riga: x · W[c]ᵀ + b[c] per ogni carattere c. I logits non sono ancora probabilità: possono essere negativi e non sommano a 1.`,
      duration: 6,
      code: codeFor("lmHead", "lm_head"),
      scene: () => {
        parts.input();
        logitsCard({ reveal: 0 });
        strip();
      },
      board: () => [
        { title: "logits = x · Wᵀ + b", kicker: "Logits" },
        { line: ["Carattere in corso: ", { text: "", tone: "accent", live: "row" }] },
        { line: ["I tre più alti: ", ...topK(D.logits, 3).flatMap((e, k) => [k ? ", " : "", { text: `«${chars[e.id]}» ${num(e.value, 2)}`, tone: "num" }])] },
      ],
      update: (() => {
        let sweep = null;
        return (p) => {
          sweep ??= parts.sweepRows(stage.cards.get("lin-output"), null);
          sweep(progressCount(p, V));
        };
      })(),
      frame: () => all,
    },
    {
      title: "Softmax: le probabilità",
      caption: "La softmax fa e^logit per ogni carattere e divide per la somma: numeri tra 0 e 1 che sommano a 1. Un logit più alto di 1 vale circa 2,7 volte la probabilità.",
      duration: 6,
      code: codeFor("probs", "gpt/generation.py"),
      scene: () => {
        logitsCard();
        probsCard({ reveal: 0 });
        strip();
      },
      board: () => [
        { title: "p = softmax(logits)", kicker: "Probabilità" },
        { bars: top.map((e) => ({ label: `«${chars[e.id]}»`, value: e.value, text: pct(e.value), mark: e.id === best })), max: maxProb },
        { note: `I ${V} numeri sommano a 1. Qui gli 8 più probabili.` },
      ],
      update: (() => {
        const stepFn = onChange((k) => stage.cards.get("lg-probs").reveal(k));
        return (p) => stepFn(progressCount(p, V, { hold: 0.3 }));
      })(),
      frame: () => unionRect(G.B, G.output, probsCol),
    },
    {
      title: "Il carattere scelto",
      caption: `Il più probabile dopo ${token} è «${chars[best]}» (${pct(D.probs[best])}). In generazione non si prende sempre il primo: si estrae a caso con queste probabilità (torch.multinomial), così il testo non si ripete.${nextText ? ` Nel prompt dopo ${token} viene ${nextText}: il modello gli dava ${pct(D.nextP)}.` : ""}`,
      duration: 7,
      code: codeFor("sample", "gpt/generation.py"),
      scene: () => {
        logitsCard().mark([{ r0: best, r1: best + 1, c0: 0, c1: 1 }]);
        const pc = probsCard();
        pc.mark([{ r0: best, r1: best + 1, c0: 0, c1: 1 }, ...(D.next !== null && D.next !== best ? [{ r0: D.next, r1: D.next + 1, c0: 0, c1: 1 }] : [])]);
        strip(best);
        stage.highlight(N.weight, { rows: [best] });
      },
      board: () => [
        { title: `Dopo ${token}: «${chars[best]}»`, kicker: "Previsione" },
        { line: ["Il più probabile: ", { text: `«${chars[best]}» ${pct(D.probs[best])}`, tone: "good" }, `, la riga ${best} di lm_head (evidenziata nella scena).`] },
        nextText ? { line: [`Nel prompt viene ${nextText}: p = `, { text: pct(D.nextP), tone: "accent" }, `, loss −ln p = `, { text: num(D.nextLoss, 3), tone: "num" }, " (è il numero che il training cerca di abbassare)."] } : { note: "Il token scelto è l'ultimo del prompt: che cosa viene dopo lo decide la generazione." },
      ],
      frame: () => unionRect(G.W, G.B, G.output, probsCol),
    },
  ];
}
