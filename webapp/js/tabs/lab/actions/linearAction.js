/**
 * Azione «x · Wᵀ riga per colonna»: una nn.Linear al microscopio, con i pesi veri della scena e
 * l'ingresso vero del token scelto (traccia del motore JS). Predefinita: la key della head scelta.
 *
 *   1. l'ingresso x (sopra W, x[j] sopra la colonna j)     4. tutte le righe: l'uscita si riempie
 *   2. una riga di W, colonna per colonna (numeri leggibili)  5. il risultato, uguale al motore
 *   3. i prodotti si sommano (+ la bias, se c'è)
 */
import { codeFor } from "./code.js";
import { visibleChar } from "./settings.js";
import { linearData } from "./data.js";
import { int, num, vectorPreview } from "./format.js";
import { linearLayout, linearParts, outputLabels, windowRect } from "./linearKit.js";
import { maxAbsDiff, topK } from "./math.js";
import { readableColumns } from "./plan.js";

/** Che cosa fa ogni nn.Linear, in parole semplici (per didascalie e lavagna). */
const ROLE = {
  key: (f) => ({ what: `la key della head ${f.head} del blocco ${f.block}`, does: "trasforma i numeri del token nella sua key: «che cosa contengo», l'etichetta che il token espone agli altri" }),
  query: (f) => ({ what: `la query della head ${f.head} del blocco ${f.block}`, does: "trasforma i numeri del token nella sua query: «che cosa cerco» negli altri token" }),
  value: (f) => ({ what: `il value della head ${f.head} del blocco ${f.block}`, does: "trasforma i numeri del token nel suo value: «che cosa comunico se mi scelgono»" }),
  proj: (f) => ({ what: `la proiezione proj del blocco ${f.block}`, does: "rimescola le uscite delle head affiancate, prima del residuo" }),
  ffn1: (f) => ({ what: `il primo strato della feed-forward del blocco ${f.block}`, does: "accende i neuroni nascosti della feed-forward (prima della ReLU)" }),
  ffn2: (f) => ({ what: `il secondo strato della feed-forward del blocco ${f.block}`, does: "riporta i neuroni nascosti alla dimensione del flusso residuo" }),
  lm_head: () => ({ what: "lm_head, l'ultima nn.Linear", does: "dà un punteggio (logit) a ogni carattere del vocabolario" }),
};

/** I passi dell'azione. `f` = forwardData del motore, `choice` = quale nn.Linear. */
export function linearSteps(stage, f, choice) {
  const S = stage.scene;
  const L = linearData(f, choice);
  const G = linearLayout(S, L.weight, L.bias);
  if (!G) return null;
  const isVocab = L.choice === "lm_head";
  const labels = outputLabels(L.outDim, f.vocab, isVocab);
  const rowLabel = isVocab ? (r) => visibleChar(labels[r]) : null;
  const parts = linearParts(stage, G, { x: L.input, W: L.W, b: L.b, y: L.output, names: { weight: L.weight, bias: L.bias }, inDim: L.inDim, outDim: L.outDim, inLabel: L.inputLabel, outLabel: L.outName, rowLabel });
  // la riga spiegata nel dettaglio: per lm_head quella del carattere più probabile, altrimenti la prima
  const d = isVocab ? topK(L.output, 1)[0].id : 0;
  const role = ROLE[L.choice](f);
  const token = `«${visibleChar(f.chars[f.t])}»`;
  const shape = `${int(L.outDim)} × ${int(L.inDim)}`;
  const code = codeFor(L.code, L.weight);
  const diff = maxAbsDiff(L.recomputed, L.output);
  const rowName = isVocab ? `la riga del carattere «${visibleChar(labels[d])}»` : `la riga ${d}`;
  const sweepSeconds = Math.min(7, Math.max(3.5, L.outDim * 0.09));

  return [
    {
      title: "L'ingresso x",
      caption: `Il token ${token} (posizione ${f.t} del prompt) arriva a ${role.what} come vettore x di ${L.inDim} numeri. Una nn.Linear è come una giuria: ogni riga di W è un giudice che dà un peso a ciascun numero di x.`,
      duration: 5,
      code,
      scene: () => {
        parts.input();
        parts.output({ reveal: 0 });
      },
      board: () => [
        { title: `${L.inputText} · token ${token}`, kicker: "x · Wᵀ" },
        { line: [`W (${shape}) ${role.does}. La sua riga d dice quanto ogni numero di x conta per l'uscita d.`] },
        { vector: { values: L.input, max: 16, caption: `x, i ${L.inDim} numeri che entrano (i primi 16):` } },
      ],
      frame: () => G.all,
    },
    {
      title: "Una riga di W, colonna per colonna",
      caption: `${capital(rowName)} di W incontra x: ogni numero di x si moltiplica per il peso che sta sotto di lui, nella stessa colonna.`,
      duration: 6,
      code,
      scene: () => {
        parts.input();
        parts.products(d, { reveal: 0 });
        parts.output({ reveal: 0 });
        parts.highlightRow(d);
      },
      board: () => [{ title: `Riga ${d} di W per x`, kicker: "x · Wᵀ" }, ...parts.productLines(d)],
      update: (() => {
        let sweep = null;
        return (p) => {
          sweep ??= parts.sweepColumns(d, stage.cards.get("lin-products"));
          sweep(Math.min(windowColumns(stage), progressCount(p, windowColumns(stage))));
        };
      })(),
      frame: () => windowRect(G, d, 0, windowColumns(stage)),
    },
    {
      title: `${L.inDim} prodotti, una somma`,
      caption: `I ${L.inDim} prodotti si sommano: è il voto del giudice ${d}.${L.b ? " Poi si aggiunge la bias, il suo punto di partenza." : " Questa nn.Linear non ha bias (bias=False)."}`,
      duration: 6,
      code,
      scene: () => {
        parts.input();
        parts.products(d, { reveal: 0 });
        parts.output({ reveal: () => false });
        parts.highlightRow(d);
      },
      board: () => [{ title: `${L.outName}[${d}]: la somma della riga ${d}`, kicker: "x · Wᵀ" }, ...parts.sumLines(d, { live: true })],
      update: (() => {
        let sweep = null;
        let done = null;
        return (p) => {
          sweep ??= parts.sweepColumns(d, stage.cards.get("lin-products"));
          sweep(progressCount(p, L.inDim, { hold: 0.3 }));
          const finished = p >= 0.72;
          if (finished === done) return;
          done = finished;
          const out = stage.cards.get("lin-output");
          out.reveal(finished ? (i) => i === d : () => false);
          out.mark(finished ? [{ r0: d, r1: d + 1, c0: 0, c1: 1 }] : null);
          if (finished) parts.linkRow(d, out);
          else stage.clearLinks();
        };
      })(),
      frame: () => G.all,
    },
    {
      title: "Tutte le righe",
      caption: `Ogni riga di W fa lo stesso calcolo con lo stesso x: ${int(L.outDim)} righe, ${int(L.outDim)} numeri d'uscita. Sulla GPU tutte insieme; qui una alla volta.`,
      duration: sweepSeconds,
      code,
      scene: () => {
        parts.input();
        parts.products(0);
        parts.output({ reveal: (i) => i === d });
      },
      board: () => [
        { title: `${L.outName} = x · Wᵀ${L.b ? " + b" : ""}: riga per riga`, kicker: "x · Wᵀ" },
        { line: ["Riga in corso: ", { text: "", tone: "accent", live: "row" }] },
        { note: `${int(L.outDim)} righe × ${int(L.inDim)} colonne = ${int(L.outDim * L.inDim)} moltiplicazioni per un solo token.` },
      ],
      update: (() => {
        let sweep = null;
        return (p) => {
          sweep ??= parts.sweepRows(stage.cards.get("lin-output"), stage.cards.get("lin-products"), { keep: d });
          sweep(progressCount(p, L.outDim));
        };
      })(),
      frame: () => G.all,
    },
    {
      title: "Il risultato",
      caption: `Ecco ${L.outName}: ${int(L.outDim)} numeri per il token ${token}, calcolati qui riga per colonna e uguali a quelli del motore JS${diff === 0 ? " (scarto 0: stesse operazioni, stesso ordine)" : ` (scarto massimo ${num(diff, 8)})`}.`,
      duration: 6,
      code,
      scene: () => {
        parts.input();
        parts.output();
      },
      board: () => [
        { title: `${L.outName} del token ${token} (${int(L.outDim)} numeri)`, kicker: "x · Wᵀ" },
        isVocab
          ? { bars: topK(L.output, 5).map((e) => ({ label: `«${visibleChar(labels[e.id])}»`, value: e.value, text: num(e.value, 3), mark: e.id === d })), max: Math.max(...L.output) }
          : { vector: { values: L.output, max: 24 } },
        { line: [`${L.outName} = `, { text: vectorPreview(L.output, { count: 5 }), tone: "num" }] },
        { note: diff === 0 ? "Controllo: il motore JS dà gli stessi numeri, bit per bit." : `Controllo: scarto massimo dal motore ${num(diff, 8)}.` },
      ],
      frame: () => G.all,
    },
  ];
}

/** Quante colonne di numeri leggibili (celle di ~54 px) stanno adesso nella parte libera, etichette comprese. */
function windowColumns(stage) {
  let labelPx = 0;
  for (const key of ["lin-input", "lin-products"]) {
    const card = stage.cards.get(key);
    if (card?.visible) labelPx = Math.max(labelPx, card.labelReservePx());
  }
  return readableColumns(stage.freeArea().width, { minCellPx: 54, min: 4, max: 14, labelPx });
}

function progressCount(p, n, options) {
  const hold = options?.hold ?? 0.15;
  return Math.min(n, Math.floor(Math.min(1, p / (1 - hold)) * n));
}

const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);
