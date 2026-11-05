/**
 * Tabelle dei pesi del tab Transformer: per componente e riepilogo dell'intero modello.
 * I numeri vengono da data/model.json, cioè dal modello PyTorch vero.
 */
import { el, fillTemplate, fmt } from "../../core/dom.js";
import { summarize } from "./data.js";

export function weightsTable(spec, refs, note) {
  const fill = (t) => fillTemplate(t, spec.vars);
  const rows = refs.map((ref) => {
    const { count, hits } = summarize(spec, ref.prefix, ref.buffer);
    const shapes = hits.length <= 3 ? hits.map((h) => `(${h.shape.join(", ")})`).join(" + ") : `${hits.length} tensori`;
    return el(
      "tr",
      {},
      el("td", { textContent: fill(ref.label) }),
      el("td", { className: "mono", textContent: shapes }),
      el("td", { className: "num", textContent: ref.buffer ? "—" : fmt(count) }),
      el("td", {}, el("span", { className: ref.buffer ? "badge fixed" : "badge learned", textContent: ref.buffer ? "non appreso" : "appreso" })),
    );
  });
  const learned = refs.filter((r) => !r.buffer).reduce((acc, r) => acc + summarize(spec, r.prefix).count, 0);
  const table = el(
    "table",
    { className: "weights" },
    el("thead", {}, el("tr", {}, ...["Peso", "Forma", "Parametri", ""].map((h) => el("th", { textContent: h })))),
    el("tbody", {}, ...rows),
  );
  if (refs.filter((r) => !r.buffer).length > 1) {
    table.append(el("tfoot", {}, el("tr", {}, el("td", { textContent: "Totale" }), el("td"), el("td", { className: "num", textContent: fmt(learned) }), el("td"))));
  }
  return el("div", { className: "weights-box" }, el("h4", { textContent: "Pesi" }), table, note ? el("p", { className: "hint", textContent: fill(note) }) : null);
}

export function modelSummaryTable(spec) {
  const cfg = spec.config;
  const parts = [
    ["Token embedding", "token_embedding_table."],
    ["Position embedding", "position_embedding_table."],
    ...[...Array(cfg.n_layer).keys()].map((i) => [`Blocco ${i}`, `blocks.${i}.`]),
    ["LayerNorm finale", "ln_f."],
    ["lm_head", "lm_head."],
  ];
  const total = spec.total_params;
  const rows = parts.map(([name, prefix]) => {
    const n = summarize(spec, prefix).count;
    return el(
      "tr",
      {},
      el("td", { textContent: name }),
      el("td", { className: "num", textContent: fmt(n) }),
      el("td", { className: "num", textContent: `${((n / total) * 100).toFixed(1)}%` }),
    );
  });
  const bufferCount = spec.buffers.length;
  return el(
    "div",
    { className: "weights-box" },
    el("h4", { textContent: "Tutti i pesi del modello" }),
    el(
      "table",
      { className: "weights" },
      el("thead", {}, el("tr", {}, ...["Componente", "Parametri", "%"].map((h) => el("th", { textContent: h })))),
      el("tbody", {}, ...rows),
      el("tfoot", {}, el("tr", {}, el("td", { textContent: "Totale appreso" }), el("td", { className: "num", textContent: fmt(total) }), el("td", { className: "num", textContent: "100%" }))),
    ),
    el("p", {
      className: "hint",
      textContent: `Non appresi: ${bufferCount} maschere tril (${spec.buffers[0].shape.join("×")}), una per head; dropout = ${cfg.dropout} (nessun peso). Pesi mostrati: ${spec.trained ? "addestrati" : "inizializzazione casuale"} (${spec.source}).`,
    }),
  );
}
