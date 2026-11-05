/**
 * Menu a fisarmonica condiviso: punti principali → sottopunti → spiegazione e
 * riquadro "Approfondimento" con le formule (KaTeX).
 *
 * Formato dei passi:
 *   { id, title, children: [ { id, title, text (HTML fidato, scritto da noi),
 *                              math?: [{tex, note?}], extra?: (item) => Node, ...qualsiasi } ] }
 * Quando l'utente seleziona un sottopunto viene chiamato onSelect(item, title).
 */
import katex from "../../../vendor/katex/katex.mjs";
import { el, fillTemplate } from "../dom.js";

/** Riquadro "Approfondimento" con formule LaTeX. */
export function mathBox(items, vars = {}, title = "Approfondimento") {
  const box = el("details", { className: "deep" }, el("summary", {}, title));
  for (const { tex, note } of items) {
    const holder = el("div", { className: "formula" });
    katex.render(fillTemplate(tex, vars), holder, { displayMode: true, throwOnError: false });
    box.append(holder);
    if (note) box.append(el("p", { className: "hint" }, fillTemplate(note, vars)));
  }
  return box;
}

export class Menu {
  /**
   * @param {HTMLElement} root
   * @param {object[]} steps
   * @param {object} [options]
   * @param {object} [options.vars] valori per i segnaposto {{...}}
   * @param {(item: object, title: string) => void} [options.onSelect]
   */
  constructor(root, steps, { vars = {}, onSelect = () => {} } = {}) {
    this.root = root;
    this.vars = vars;
    this.onSelect = onSelect;
    this.items = new Map();
    this.order = [];
    this.current = null;
    this.render(steps);
  }

  render(steps) {
    const fill = (t) => fillTemplate(t, this.vars);
    for (const step of steps) {
      const section = el("section", { className: "step" });
      const header = el("button", { className: "step-header", type: "button" }, fill(step.title));
      header.setAttribute("aria-expanded", "false");
      const list = el("div", { className: "step-children" });
      header.addEventListener("click", () => {
        const open = header.getAttribute("aria-expanded") !== "true";
        header.setAttribute("aria-expanded", String(open));
        section.classList.toggle("open", open);
      });
      for (const item of step.children) {
        const title = fill(item.title);
        const button = el("button", { className: "item-header", type: "button" }, title);
        const body = el("div", { className: "item-body" });
        body.innerHTML = fill(item.text ?? "");
        const extra = item.extra?.(item);
        if (extra) body.append(extra);
        if (item.math?.length) body.append(mathBox(item.math, this.vars));
        const wrapper = el("div", { className: "item" }, button, body);
        button.addEventListener("click", () => this.select(item.id));
        list.append(wrapper);
        this.items.set(item.id, { item, wrapper, section, header, title });
        this.order.push(item.id);
      }
      section.append(header, list);
      this.root.append(section);
    }
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

  /** Passa allo step successivo (+1) o precedente (-1). */
  move(delta) {
    const i = this.order.indexOf(this.current);
    const next = this.order[Math.min(this.order.length - 1, Math.max(0, i + delta))];
    this.select(next ?? this.order[0]);
  }
}
