/**
 * Menu laterale a fisarmonica: punti principali → sottopunti → spiegazione,
 * tabella dei pesi e riquadro "Approfondimento" con le formule (KaTeX).
 */
import katex from "../../vendor/katex/katex.mjs";
import { fillTemplate, summarize } from "../data.js";

const fmt = (n) => n.toLocaleString("it-IT", { useGrouping: "always" });

function el(tag, attrs = {}, ...children) {
  const node = Object.assign(document.createElement(tag), attrs);
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

function weightsTable(spec, refs, note) {
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

function modelSummaryTable(spec) {
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

function mathBox(spec, items) {
  const box = el("details", { className: "deep" }, el("summary", { textContent: "Approfondimento" }));
  items.forEach(({ tex, note }) => {
    const holder = el("div", { className: "formula" });
    katex.render(fillTemplate(tex, spec.vars), holder, { displayMode: true, throwOnError: false });
    box.append(holder);
    if (note) box.append(el("p", { className: "hint", textContent: fillTemplate(note, spec.vars) }));
  });
  return box;
}

export class Menu {
  constructor(root, steps, spec, onSelect) {
    this.root = root;
    this.spec = spec;
    this.onSelect = onSelect;
    this.items = new Map();
    this.order = [];
    this.render(steps);
  }

  render(steps) {
    const fill = (t) => fillTemplate(t, this.spec.vars);
    steps.forEach((step) => {
      const section = el("section", { className: "step" });
      const header = el("button", { className: "step-header", type: "button", textContent: fill(step.title) });
      header.setAttribute("aria-expanded", "false");
      const list = el("div", { className: "step-children" });
      header.addEventListener("click", () => {
        const open = header.getAttribute("aria-expanded") !== "true";
        header.setAttribute("aria-expanded", String(open));
        section.classList.toggle("open", open);
      });

      step.children.forEach((item) => {
        const button = el("button", { className: "item-header", type: "button", textContent: fill(item.title) });
        const body = el("div", { className: "item-body" });
        body.innerHTML = fill(item.text);
        if (item.weights) body.append(weightsTable(this.spec, item.weights, item.weightsNote));
        if (item.weightsSummary) body.append(modelSummaryTable(this.spec));
        if (item.math) body.append(mathBox(this.spec, item.math));
        const wrapper = el("div", { className: "item" }, button, body);
        button.addEventListener("click", () => this.select(item.id));
        list.append(wrapper);
        this.items.set(item.id, { item, wrapper, section, header, title: fill(item.title) });
        this.order.push(item.id);
      });

      section.append(header, list);
      this.root.append(section);
    });
  }

  select(id) {
    const entry = this.items.get(id);
    if (!entry) return;
    this.root.querySelectorAll(".item.active").forEach((n) => n.classList.remove("active"));
    entry.wrapper.classList.add("active");
    entry.section.classList.add("open");
    entry.header.setAttribute("aria-expanded", "true");
    entry.wrapper.scrollIntoView({ block: "nearest", behavior: "smooth" });
    this.current = id;
    this.onSelect(entry.item, entry.title);
  }

  /** Passa allo step successivo (+1) o precedente (-1): comodo con le frecce in aula. */
  move(delta) {
    const i = this.order.indexOf(this.current);
    const next = this.order[Math.min(this.order.length - 1, Math.max(0, i + delta))];
    this.select(next ?? this.order[0]);
  }
}
