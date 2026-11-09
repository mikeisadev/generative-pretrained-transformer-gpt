/**
 * L'oggetto `tab` del Laboratorio pesi: lo ricevono tutti i moduli del tab e quelli della fase 2
 * (contratto in lab.md, "L'oggetto tab"). Raccoglie layout, stato e scena e offre le operazioni comuni:
 * scegliere un tensore o un modulo, mostrarne il codice vero, aprire un pannello, registrare viste e link.
 */
import { buildModuleTree, pythonPath } from "../../shared/weights/moduleTree.js";
import { SOURCES } from "./content.js";
import { classesFromRunInfo, codeLine, layoutModel } from "./scene/layout.js";

const narrow = window.matchMedia("(max-width: 900px)");

export /** L'oggetto `tab` passato ai moduli (e alla fase 2): vedi lab.md, "L'oggetto tab". */
function createTabObject({ app, ui, spec, state }) {
  const trees = new WeakMap();
  const layouts = new WeakMap();
  const views = new Map();
  const links = new Map();
  const tab = {
    app,
    ui,
    spec,
    state,
    panes: {},
    stats: {},
    sources: SOURCES.map((g) => ({ ...g, items: [...g.items] })),
    disposers: [],
    engineFor: (id, step) => state.engineFor(id, step),
    weightsAt: (id, step, options) => state.weightsAt(id, step, options),

    /** L'albero dei moduli di un run (buildModuleTree con il codice vero del corso), in cache. */
    treeOf(run) {
      if (!trees.has(run)) trees.set(run, buildModuleTree(run.params, { snippets: spec.snippets, config: run.config }));
      return trees.get(run);
    },
    /** La disposizione della scena di un run (scene/layout.js), in cache. */
    layoutOf(run) {
      if (!layouts.has(run)) layouts.set(run, layoutModel(tab.treeOf(run), { config: run.config, snippets: spec.snippets, classes: classesFromRunInfo(run.info?.modules) }));
      return layouts.get(run);
    },

    /**
     * Sceglie un tensore o un modulo: riquadro nella scena, codice vero, pannello Tensore.
     * @param {string} path nome PyTorch ("" = il modello)
     * @param {{cell?: {i: number, j: number}, focus?: boolean, code?: boolean, source?: "scene"|"outline"|"pane"|"link"}} [o]
     */
    inspect(path, { cell = null, focus = false, code = true, source = "code" } = {}) {
      const node = tab.scene.scene.nodeOf(path) ?? (path === "" ? tab.scene.scene.layout?.root : null);
      if (!node) return false;
      state.select(path);
      // dalla scena (e sempre sul telefono) il codice si apre ridotto: aprirlo intero sposterebbe la scena
      // sotto il puntatore (il pannello copre la destra); la riga di codice è comunque nel pannello Tensore
      if (code) tab.showCode(path, { expand: source !== "scene" && !narrow.matches });
      const isTensor = node.kind === "param" || node.kind === "buffer";
      // dalla scena (o per un tensore) i numeri sono nel pannello Tensore; sul telefono il cassetto resta chiuso
      if (!narrow.matches && (source === "scene" || source === "link" || isTensor)) ui.sidebar.select("tensore");
      if (cell) tab.inspector.selectCell(cell.i, cell.j);
      if (focus) {
        ui.director.noteUserIntent();
        void (isTensor ? tab.scene.focusTensor(path) : tab.scene.focusModule(path));
      }
      return true;
    },

    /** Una cella (riga i, colonna j del tensore): riquadro nella scena e, con `fly`, la camera vicino al numero. */
    showCell(name, i, j, { fly = false } = {}) {
      const node = tab.scene.scene.nodeOf(name);
      if (!node?.matrix) return;
      const vector = node.matrix.vector;
      const cell = vector === "row" ? { row: 0, col: i } : { row: i, col: j ?? 0 };
      tab.scene.scene.hoverCell(name, cell);
      if (fly) {
        ui.director.noteUserIntent();
        void tab.scene.focusCell(name, cell.row, cell.col);
      }
    },

    /**
     * Il codice vero che crea un modulo o un peso (snippet di data/model.json, riga evidenziata).
     * `expand: false` lascia il pannello ridotto se era nascosto o ridotto (cambia solo l'intestazione).
     */
    showCode(path, { expand = true } = {}) {
      const panel = ui.codePanel;
      const keepSmall = !expand && (panel.root.hidden || panel.root.classList.contains("minimized"));
      const done = () => keepSmall && panel.root.classList.add("minimized");
      const node = tab.scene.scene.nodeOf(path) ?? (path === "" ? tab.scene.scene.layout?.root : null);
      const c = node?.code;
      const snippet = c?.snippet ? spec.snippets[c.snippet] : null;
      const name = pythonPath(path) || node?.title || "modello";
      if (!snippet) {
        panel.show({ title: name, code: "# Questo modulo non è nel codice del corso (gpt/*.py): il run viene da un'architettura\n# modificata. Il Laboratorio ne ricava la struttura dai nomi dei pesi; il codice vero\n# è quello del repository in cui il run è stato addestrato.", language: "python" });
        done();
        return;
      }
      let title = name;
      let highlight = [c.first ?? codeLine(c)];
      if (node.kind === "module" && c.createdBy && /^class\s/.test(String(c.text))) {
        // un elemento di una lista (Block, Head): la sua classe, e nel titolo la riga che lo crea
        title = `${name}: ${c.className} (creato da ${String(c.createdBy.file).split("/").pop()}:${c.createdBy.line})`;
        highlight = [codeLine(c)];
      } else if (node.kind === "param") title = `${name} (${node.shape.join(" × ")})`;
      panel.show({ title, file: snippet.file, startLine: snippet.start_line, code: snippet.code, language: "python", highlight });
      done();
    },

    setCaption(text) {
      ui.caption.textContent = text;
    },

    /** Apre un pannello della sidebar (anche il cassetto sul telefono); `focus: "step"` porta al cursore. */
    openPane(id, { focus } = {}) {
      ui.sidebar.select(id);
      ui.sidebar.toggleCollapsed(false);
      if (focus === "step") tab.model.timeline.focus();
    },

    /** Viste raggiungibili con #lab/go=view:<id> registrate dalla fase 2 (es. "confronto"). */
    views: {
      register(id, { title = id, open }) {
        if (typeof open !== "function") throw new TypeError("tab.views.register: serve open()");
        views.set(id, { id, title, open });
      },
      has: (id) => views.has(id),
      open: (id) => views.get(id)?.open(),
    },
    /** Tipi di link nuovi (#lab/go=<kind>:<id>) della fase 2: handler(id) → Promise<boolean>. */
    links: {
      register(kind, handler) {
        if (!/^[a-z]+$/.test(kind) || typeof handler !== "function") throw new TypeError("tab.links.register(kind, handler): kind di sole lettere minuscole");
        links.set(kind, handler);
      },
      handler: (kind) => links.get(kind) ?? null,
    },
  };
  return tab;
}

