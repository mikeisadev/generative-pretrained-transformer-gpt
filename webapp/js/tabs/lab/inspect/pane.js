/**
 * Pannello "Tensore" della sidebar (solo DOM): il tensore o il modulo scelto, con i numeri veri.
 *
 * Per un tensore: nome Python, forma, da dove vengono i numeri (pesi finali, snapshot, interpolazione),
 * la riga di codice che lo crea, che cosa sono righe e colonne, la tabella dei numeri (TensorTable del
 * nucleo, virtualizzata), le statistiche e l'istogramma. Passando sopra una cella (qui o nella scena):
 * valore e significato di riga e colonna. Clic su una cella della tabella → la scena ci vola; clic nella
 * scena → la tabella ci va.
 * Per un modulo: classe, riga di codice, parametri e sotto-moduli (pulsanti per sceglierli).
 *
 * Tutti i testi che vengono dai file dello studente (nomi dei tensori, del run) passano da textContent.
 */
import { el } from "../../../core/dom.js";
import { Histogram } from "../../../core/ui/histogram.js";
import { TensorTable, renderTensorStats } from "../../../core/ui/tensorTable.js";
import { formatPrecise } from "../../../shared/weights/display.js";
import { pythonPath } from "../../../shared/weights/moduleTree.js";
import { codeLine, fmtInt, fmtPct } from "../scene/layout.js";
import { tensorMeaning } from "./meaning.js";

const EMPTY_TEXT = "Scegli una matrice nella scena (clic) oppure un modulo nel pannello Modello: qui compaiono i suoi numeri veri, le statistiche e l'istogramma.";

export function createInspector(pane, tab) {
  const { state } = tab;
  const empty = el("p", { className: "lab-empty" }, EMPTY_TEXT);
  const kicker = el("p", { className: "lab-kicker" });
  const title = el("h3", { className: "lab-tensor-name mono" });
  const meta = el("p", { className: "lab-meta" });
  const source = el("p", { className: "lab-source" });
  const role = el("p", { className: "lab-role" });
  const code = el("div", { className: "lab-code-line" });
  const axes = el("p", { className: "lab-axes hint" });
  const readout = el("p", { className: "lab-readout", ariaLive: "polite" });
  const tableBox = el("div", { className: "lab-table" });
  const statsBox = el("div", { className: "lab-stats" });
  const histBox = el("div", { className: "lab-hist" });
  const members = el("div", { className: "lab-members" });
  const tensorParts = el("div", { className: "lab-tensor-parts" }, readout, tableBox, el("h4", { className: "lab-h4" }, "Statistiche"), statsBox, el("h4", { className: "lab-h4" }, "Distribuzione dei valori"), histBox);
  const body = el("section", { className: "lab-inspect", hidden: true }, kicker, title, meta, source, role, code, axes, tensorParts, members);
  pane.append(empty, body);

  let current = null; // { path, kind, node, meaning, data }
  const table = new TensorTable(tableBox, {
    rowLabel: (i) => current?.meaning?.rows?.label(i) ?? String(i),
    colLabel: (j) => current?.meaning?.cols?.label(j) ?? String(j),
    onHover: (cell) => {
      if (!current || !cell) return void (cell === null && current && tab.scene.scene.hoverCell(null, null));
      readCell(cell.i, cell.j, cell.value);
      tab.scene.scene.hoverCell(current.path, toMatrixCell(cell.i, cell.j));
    },
    onSelect: (cell) => {
      if (!current) return;
      readCell(cell.i, cell.j, cell.value);
      tab.showCell(current.path, cell.i, cell.j, { fly: true, from: "table" });
    },
  });
  const hist = new Histogram(histBox, { bins: 60, height: 120, label: "" });

  /** La cella (i, j) del tensore → la cella della matrice disegnata (un vettore in riga ha gli elementi nelle colonne). */
  function toMatrixCell(i, j) {
    const vector = current?.node?.matrix?.vector;
    if (vector === "row") return { row: 0, col: i };
    return { row: i, col: j ?? 0 };
  }

  function readCell(i, j, value) {
    if (!current?.meaning) return;
    const m = current.meaning;
    const isVector = !m.cols || current.shape.length === 1;
    const where = isVector ? m.rows.describe(i) : `${m.rows.describe(i)} · ${m.cols.describe(j)}`;
    const shown = current.kind === "buffer" ? (value ? "1" : "0") : formatPrecise(value);
    readout.replaceChildren(el("strong", {}, isVector ? `[${i}] = ${shown}` : `[${i}, ${j}] = ${shown}`), el("span", {}, ` ${where}.`), el("span", { className: "lab-cell-meaning" }, ` ${capital(m.cell(i, j))}.`));
  }

  function show(path) {
    const node = path ? tab.scene.scene.nodeOf(path) : null;
    if (!node) {
      current = null;
      body.hidden = true;
      empty.hidden = false;
      return;
    }
    empty.hidden = true;
    body.hidden = false;
    const isTensor = node.kind === "param" || node.kind === "buffer";
    const pyName = pythonPath(path) || node.title;
    title.textContent = pyName;
    code.replaceChildren();
    if (node.code) {
      const line = node.code.createdBy && /^class\s/.test(String(node.code.text)) ? node.code.createdBy : node.code;
      code.append(
        el("code", {}, codeLine(line)),
        el("span", { className: "lab-code-where" }, `${String(line.file ?? "").replace(/^gpt\//, "gpt/")}:${line.line}`),
        el("button", { type: "button", className: "lab-link", onclick: () => showFullCode(path) }, "Mostra il codice"),
      );
    } else {
      code.append(el("span", { className: "hint" }, "Questo modulo non è nel codice del corso (gpt/*.py): viene da un'architettura modificata. L'albero è costruito dai nomi dei pesi."));
    }
    if (isTensor) showTensor(path, node);
    else showModule(path, node);
  }

  /** "Mostra il codice": il pannello intero (sul telefono il cassetto si chiude, se no lo coprirebbe). */
  function showFullCode(path) {
    tab.showCode(path, { expand: true });
    tab.ui.codePanel.root.classList.remove("minimized");
    if (window.matchMedia("(max-width: 900px)").matches) tab.ui.sidebar.toggleCollapsed(true);
  }

  function showTensor(path, node) {
    const isBuffer = node.kind === "buffer";
    const shape = node.shape ?? [];
    const meaning = tensorMeaning(path, shape, { vocab: tab.scene.scene.vocab, config: tab.scene.scene.config });
    const data = isBuffer ? tab.scene.scene.matrixOf(path)?.data : state.current?.view?.tensors.get(path)?.data;
    current = { path, kind: isBuffer ? "buffer" : "tensor", node, meaning, shape, data };
    kicker.textContent = isBuffer ? "Buffer (non è un peso)" : "Tensore";
    meta.textContent = `forma (${shape.join(", ")}) · ${fmtInt(node.params || shape.reduce((a, b) => a * b, 1))} ${isBuffer ? "numeri" : "pesi"}${node.share ? ` · ${fmtPct(node.share)} del modello` : ""}`;
    const view = state.current?.view;
    const custom = tab.scene.scene.customValues.has(path);
    source.textContent = isBuffer
      ? "Costruito dal codice: torch.tril(torch.ones(block_size, block_size)). Non sta nei pesi salvati (persistent=False) e il training non lo cambia."
      : custom
        ? `Nella scena: ${tab.scene.scene.valuesLabel || "altri valori"}. Qui sotto i pesi del run A (${view?.label ?? ""}).`
        : `Numeri veri del run «${state.entry(state.runA)?.name ?? ""}»: ${view?.label ?? ""}.`;
    role.textContent = [node.code?.role ? capital(node.code.role) : meaning.title, meaning.formula].filter(Boolean).join(" · ");
    axes.textContent = meaning.cols ? `Righe: ${meaning.rows.name} (${fmtInt(rowsOf(shape))}) · colonne: ${meaning.cols.name} (${fmtInt(shape.at(-1))}).` : `Un vettore di ${fmtInt(shape[0] ?? 0)} numeri: ${meaning.rows.describe(0).replace(/\b0\b/, "i")}…`;
    readout.textContent = "Passa sopra una cella (qui o nella scena) per il suo significato.";
    tensorParts.hidden = !data;
    members.hidden = true;
    if (!data) return;
    const range = tab.scene.scene.matrices.get(path)?.range ?? null;
    table.setTensor({ name: path, shape, data, range, rowTitle: meaning.rows.name, colTitle: meaning.cols?.name ?? "", caption: meaning.title });
    renderTensorStats(statsBox, data, { label: isBuffer ? "Statistiche del buffer" : "Statistiche" });
    hist.setData([{ label: isBuffer ? "tril" : view?.label ?? "pesi", data }]);
  }

  function showModule(path, node) {
    current = { path, kind: node.kind, node, meaning: null, shape: [], data: null };
    tensorParts.hidden = true;
    kicker.textContent = node.kind === "virtual" ? "Modulo senza pesi" : path === "" ? "Modello" : "Modulo";
    meta.textContent = node.kind === "virtual" ? node.note : `${node.type ?? "modulo"} · ${fmtInt(node.params)} parametri${node.path ? ` · ${fmtPct(node.share)} del modello` : ""}`;
    source.textContent = "";
    role.textContent = node.kind === "virtual" ? "Fa un calcolo senza pesi da imparare: non ha numeri da mostrare." : "";
    axes.textContent = "";
    members.hidden = false;
    members.replaceChildren();
    const params = [];
    const modules = [];
    const walk = (n) => {
      for (const c of n.children ?? []) {
        if (c.kind === "group") walk(c);
        else if (c.kind === "param" || c.kind === "buffer") params.push(c);
        else modules.push(c);
      }
    };
    walk(node);
    if (params.length) members.append(el("h4", { className: "lab-h4" }, "Pesi di questo modulo"), list(params));
    if (modules.length) members.append(el("h4", { className: "lab-h4" }, "Sotto-moduli"), list(modules));
  }

  function list(nodes) {
    return el(
      "ul",
      { className: "lab-member-list" },
      ...nodes.map((n) =>
        el(
          "li",
          {},
          el(
            "button",
            { type: "button", className: "lab-member", onclick: () => tab.inspect(n.path, { focus: true, source: "pane" }) },
            el("span", { className: "mono" }, n.kind === "param" || n.kind === "buffer" ? n.label : `${n.title}${n.type ? ` · ${n.type}` : ""}`),
            el("small", {}, n.kind === "virtual" ? n.note : n.kind === "buffer" ? "buffer" : `${fmtInt(n.params)} ${n.kind === "param" ? "pesi" : "parametri"}`),
          ),
        ),
      ),
    );
  }

  /** I pesi di A sono cambiati (un altro step): stessi tensori, numeri nuovi; la cella scelta resta. */
  function refresh() {
    if (!current || current.kind !== "tensor") return;
    const sel = current.selectedCell;
    show(current.path);
    if (sel && current) {
      current.selectedCell = sel;
      table.select(sel.i, sel.j ?? 0, { scroll: false });
    }
  }

  state.on("select", ({ path }) => show(path));
  state.on("weights", () => refresh());

  return {
    show,
    refresh,
    /** Porta la tabella su una cella (dalla scena): la seleziona e la mostra. */
    selectCell(i, j) {
      if (!current || current.kind === "module") return;
      current.selectedCell = { i, j };
      table.select(i, j ?? 0, { scroll: true });
      const value = current.data ? current.data[current.shape.length === 1 ? i : i * current.shape.at(-1) + (j ?? 0)] : NaN;
      readCell(i, j ?? 0, value);
    },
    get current() {
      return current;
    },
    table,
    histogram: hist,
    dispose() {
      table.dispose();
      hist.dispose();
    },
  };
}

const rowsOf = (shape) => (shape.length > 1 ? shape.slice(0, -1).reduce((a, b) => a * b, 1) : shape[0] ?? 0);
const capital = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
