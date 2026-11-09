/**
 * L'indice dei moduli nel pannello "Modello" (solo DOM): lo stesso albero della scena scritto come
 * `print(model)` di PyTorch, una riga per modulo o peso, da scegliere anche da tastiera (è la via
 * accessibile alla scena 3D). Clic su una riga = la camera vola lì, il codice compare nel pannello e il
 * tensore nel pannello "Tensore". I blocchi e le head si aprono e chiudono.
 */
import { el } from "../../../core/dom.js";
import { fmtInt } from "../scene/layout.js";

/** Gli argomenti di un modulo come nel repr di PyTorch, dalle forme dei suoi pesi. */
export function moduleArgs(node) {
  const params = (node.children ?? []).filter((c) => c.kind === "param");
  const w = params.find((p) => p.key === "weight");
  const b = params.find((p) => p.key === "bias");
  if (node.type === "nn.Linear" && w?.shape?.length === 2) return `(in=${w.shape[1]}, out=${w.shape[0]}${b ? "" : ", bias=False"})`;
  if (node.type === "nn.Embedding" && w?.shape?.length === 2) return `(${w.shape[0]}, ${w.shape[1]})`;
  if (/LayerNorm/.test(node.type ?? "") && w?.shape) return `(${w.shape.join(", ")})`;
  return "";
}

/** Le righe dell'indice: {node, depth, label, count, kids} in ordine (i gruppi della disposizione sono trasparenti). */
export function outlineRows(layout) {
  const rows = [];
  const visit = (n, depth) => {
    const kids = [];
    const collect = (x) => {
      for (const c of x.children ?? []) {
        if (c.kind === "group") collect(c);
        else kids.push(c);
      }
    };
    collect(n);
    const label = n.kind === "param" || n.kind === "buffer" ? n.label : n.kind === "virtual" ? `${n.title}: ${n.type ?? ""}` : !n.path ? n.title : `${n.title}: ${n.type ?? "modulo"}${moduleArgs(n)}`;
    const count = n.kind === "virtual" ? "nessun peso" : n.kind === "buffer" ? "buffer" : fmtInt(n.params);
    rows.push({ node: n, depth, label, count, kids: kids.length });
    for (const c of kids) visit(c, depth + 1);
  };
  visit(layout.root, 0);
  return rows;
}

export function createOutline(container, tab) {
  const list = el("ul", { className: "lab-outline", role: "list", ariaLabel: "Moduli del modello, come print(model)" });
  container.append(list);
  let rows = [];
  const open = new Set([""]); // percorsi aperti: all'inizio la radice e blocks
  const lines = new Map();

  function render() {
    const layout = tab.scene.scene.layout;
    list.replaceChildren();
    lines.clear();
    if (!layout) return;
    rows = outlineRows(layout);
    if (open.size === 1) {
      open.add("blocks");
      open.add("blocks.0");
    }
    const hiddenUnder = [];
    for (const r of rows) {
      while (hiddenUnder.length && hiddenUnder.at(-1) >= r.depth) hiddenUnder.pop();
      const visible = !hiddenUnder.length;
      const n = r.node;
      const isOpen = open.has(n.path);
      if (r.kids && !isOpen) hiddenUnder.push(r.depth);
      if (!visible) continue;
      const toggle = r.kids
        ? el("button", { type: "button", className: "lab-tree-toggle", ariaExpanded: String(isOpen), ariaLabel: `${isOpen ? "Chiudi" : "Apri"} ${n.title}`, onclick: () => flip(n.path) }, isOpen ? "▾" : "▸")
        : el("span", { className: "lab-tree-toggle", ariaHidden: "true" });
      const line = el(
        "button",
        { type: "button", className: `lab-tree-line lab-role-${n.role ?? "other"}`, dataset: { path: n.path }, onclick: () => tab.inspect(n.path, { focus: true, code: true, source: "outline" }) },
        el("span", { className: "lab-tree-label mono" }, r.label),
        el("span", { className: "lab-tree-count" }, r.count),
      );
      lines.set(n.path, line);
      list.append(el("li", { style: { paddingLeft: `${r.depth * 12}px` } }, toggle, line));
    }
    mark(tab.state.selected);
  }

  function flip(path) {
    if (open.has(path)) open.delete(path);
    else open.add(path);
    render();
    lines.get(path)?.previousSibling?.focus?.(); // il fuoco resta sul pulsante che apre e chiude
  }

  /** Mette in evidenza la riga scelta (apre i suoi genitori). */
  function mark(path) {
    for (const [p, line] of lines) {
      if (p === path) line.setAttribute("aria-current", "true");
      else line.removeAttribute("aria-current");
    }
  }

  function reveal(path) {
    if (!path) return;
    const parts = path.split(".");
    let changed = false;
    for (let k = 0; k < parts.length; k++) {
      const p = parts.slice(0, k).join(".");
      if (!open.has(p)) {
        open.add(p);
        changed = true;
      }
    }
    if (changed || !lines.has(path)) render();
    mark(path);
    // solo se il pannello Modello è quello aperto: altrimenti si sposterebbe lo scorrimento (condiviso) della sidebar
    if (!container.closest("[hidden]")) lines.get(path)?.scrollIntoView({ block: "nearest" });
  }

  tab.state.on("select", ({ path }) => reveal(path));
  return { render, reveal, get rows() {
    return rows;
  } };
}
