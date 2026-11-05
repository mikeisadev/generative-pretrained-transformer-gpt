/**
 * Tab "Transformer": il GPT del corso in 3D.
 *
 * Questo file è il punto di raccordo: costruisce layout, scena, menu e il gestore delle
 * viste, poi installa i moduli specializzati (ognuno in un file proprio):
 *   lod.js           zoom semantico Dettaglio / Blocchi / Panoramica
 *   views/weights.js vista "Pesi" (sovrapposizione: dove sono i parametri)
 *   views/neurons.js vista "Rete neurale" (tutti i neuroni e le connessioni)
 *   actions/index.js azioni animate: prompt → token → … → output, training (+ grafici)
 *   context/index.js vista "Contesto": RAG + system prompt + tool → richiesta all'LLM
 * Ogni modulo esporta `install(tab)`; l'oggetto `tab` è descritto in webapp/ARCHITECTURE.md.
 */
import * as THREE from "three";
import { meshBounds } from "../../core/director.js";
import { el } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { Menu } from "../../core/ui/menu.js";
import { installActions } from "./actions/index.js";
import { buildArchitecture } from "./architecture.js";
import { SOURCES, STEPS } from "./content.js";
import { loadEngine } from "../../engine/gpt.js";
import { installContext } from "./context/index.js";
import { loadModelSpec } from "./data.js";
import { DataFlow } from "./flow.js";
import { installLod } from "./lod.js";
import { installNeuronsView } from "./views/neurons.js";
import { installWeightsView } from "./views/weights.js";
import { modelSummaryTable, weightsTable } from "./weightsTables.js";

/** Gestore delle viste: una sola attiva alla volta, elencate nel pannello "Viste" della sidebar. */
class ViewManager {
  constructor(tab, pane) {
    this.tab = tab;
    this.pane = pane;
    this.views = [];
    this.current = null;
    this.list = el("div", { className: "view-switch" });
    pane.append(this.list);
  }

  /** view = { id, label, description, order?, enter(tab), exit(tab) } */
  register(view) {
    this.views.push(view);
    this.views.sort((a, b) => (a.order ?? 50) - (b.order ?? 50));
    this.list.replaceChildren(
      ...this.views.map((v) =>
        el(
          "button",
          { type: "button", dataset: { view: v.id }, className: v.id === this.current ? "active" : "", onclick: () => this.activate(v.id) },
          el("strong", {}, v.label),
          el("small", {}, v.description ?? ""),
        ),
      ),
    );
  }

  async activate(id) {
    if (id === this.current) return;
    const next = this.views.find((v) => v.id === id);
    if (!next) return;
    const prev = this.views.find((v) => v.id === this.current);
    await prev?.exit?.(this.tab);
    this.current = id;
    this.list.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.view === id));
    await next.enter?.(this.tab);
    this.tab.events.dispatchEvent(new CustomEvent("view", { detail: id }));
  }
}

export async function createTab(root, app) {
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint: "Scegli uno step o un'azione: la camera vola sul componente e a destra compare il codice. Rotellina per lo zoom: più ti allontani, più la vista si semplifica.",
  });
  const { host, director, sidebar, codePanel, stage } = ui;
  const spec = await loadModelSpec();

  // --- scena di base
  const grid = new THREE.GridHelper(220, 110, 0x1e293b, 0x111827);
  grid.position.y = -48;
  host.scene.add(grid);
  const arch = buildArchitecture(spec);
  host.scene.add(arch.root, arch.rail);
  const flow = new DataFlow(host.scene, arch.path, arch.stations, (station) => (ui.caption.textContent = station.caption));
  host.onFrame((dt, elapsed) => flow.update(dt, elapsed));

  // --- sidebar: Percorso, Azioni, Viste
  const panePath = sidebar.addPane({ id: "percorso", title: "Percorso" });
  const paneActions = sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Panoramica delle azioni animate: scegli se seguire un prompt o il training." });
  const paneViews = sidebar.addPane({ id: "viste", title: "Viste", hint: "Cambia il modo di guardare lo stesso modello." });
  // (il pannello "Contesto" lo aggiunge context/index.js; "Fonti" va in fondo, vedi sotto)
  panePath.append(
    el(
      "p",
      { className: "legend" },
      el("span", { className: "swatch pos" }),
      " peso positivo ",
      el("span", { className: "swatch neg" }),
      " peso negativo ",
      el("span", { className: "swatch zero" }),
      " ≈ 0 · 1 cella = 1 peso",
    ),
  );

  const events = new EventTarget();
  let enginePromise = null;
  const tab = {
    app,
    ui,
    spec,
    vars: spec.vars,
    arch,
    flow,
    events,
    panes: { path: panePath, actions: paneActions, views: paneViews },
    menu: null,
    views: null,
    /** Mostra un frammento di codice esportato da gpt/export.py (chiave di SNIPPETS). */
    showCode(key, title, highlight = []) {
      const s = spec.snippets[key];
      if (!s) return codePanel.show({ title, code: `# frammento '${key}' non trovato in model.json`, language: "python" });
      codePanel.show({ title, file: s.file, startLine: s.start_line, code: s.code, language: "python", highlight });
    },
    /** Vola su un componente per id del registro (vedi architecture.js) o su un oggetto. */
    focus(target, options) {
      const object = typeof target === "string" ? arch.registry.get(target) : target;
      // 1.25: la riga di uscita in alto non finisce sotto la barra dei comandi dello stage
      if (target === "overview") return director.focus(arch.registry.get("overview"), { showFrame: false, margin: 1.25, ...options });
      return object ? director.focus(object, options) : Promise.resolve();
    },
    setCaption(text) {
      ui.caption.textContent = text;
    },
    /** Motore GPT in JavaScript (pesi addestrati veri), caricato una sola volta e condiviso tra i moduli. */
    getEngine() {
      enginePromise ??= loadEngine("data/");
      return enginePromise;
    },
  };

  // --- menu del percorso (contenuti in content.js, pesi dal modello vero)
  const steps = STEPS.map((step) => ({
    ...step,
    children: step.children.map((item) => ({
      ...item,
      extra: () => {
        const box = el("div");
        if (item.weights) box.append(weightsTable(spec, item.weights, item.weightsNote));
        if (item.weightsSummary) box.append(modelSummaryTable(spec));
        return box.childNodes.length ? box : null;
      },
    })),
  }));
  tab.menu = new Menu(panePath, steps, {
    vars: spec.vars,
    onSelect: (item, title) => {
      tab.focus(item.focus);
      tab.showCode(item.code, title, item.highlight ?? []);
    },
  });
  panePath.append(panePath.querySelector(".legend"));

  // --- viste
  tab.views = new ViewManager(tab, paneViews);
  let returning = false; // la prima attivazione è l'avvio del tab: camera e codice li sceglie createTab
  tab.views.register({
    id: "architettura",
    label: "Architettura",
    description: "Il modello come lo descrive il codice: matrici, head, blocchi e flusso dei dati.",
    order: 0,
    enter: (t) => {
      arch.root.visible = true;
      arch.rail.visible = true;
      flow.packet.visible = true;
      // Tornando da un'altra vista: camera sulla pila, codice e didascalia dello step scelto nel
      // Percorso (le altre viste non devono sapere come ripristinare questa).
      const firstEntry = !returning;
      returning = true;
      if (firstEntry || !t.menu?.current) return;
      const entry = t.menu.items.get(t.menu.current);
      t.focus("overview");
      t.showCode(entry.item.code, entry.title, entry.item.highlight ?? []);
      t.setCaption("");
    },
  });
  tab.views.activate("architettura");

  // --- moduli specializzati
  installLod(tab);
  installWeightsView(tab);
  installNeuronsView(tab);
  installActions(tab);
  installContext(tab);
  sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Da dove vengono codice, architettura e scelte di training." }).append(renderSources(SOURCES));

  // --- controlli generali
  const playFlow = el("button", { type: "button", className: "flow-toggle", onclick: () => (playFlow.textContent = flow.toggle() ? "⏸ Flusso" : "▶ Flusso") }, "⏸ Flusso");
  const overview = el("button", { type: "button", className: "overview-button", onclick: () => tab.focus("overview") }, "Vista d'insieme");
  stage.append(el("div", { className: "stage-toolbar" }, overview, playFlow));
  root.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowDown") tab.menu.move(+1);
    if (e.key === "ArrowUp") tab.menu.move(-1);
    if (e.key === " " && ui.player?.action) {
      e.preventDefault();
      ui.player.toggle();
    }
  });
  root.tabIndex = -1;

  installLevelPicking(tab);

  // Altri tab possono chiedere una vista precisa (es. il tab RAG apre la vista "contesto").
  // detail = id della vista, oppure { view, question } per passare anche la domanda alla vista Contesto.
  app.bus.addEventListener("transformer:view", (e) => {
    const { view, question } = typeof e.detail === "string" ? { view: e.detail } : e.detail ?? {};
    app.open("transformer");
    if (question) app.bus.dispatchEvent(new CustomEvent("context:question", { detail: { question } }));
    if (view) tab.views.activate(view);
  });

  tab.menu.select("data.dataset");
  tab.focus("overview");
  // Sul telefono il codice è un foglio in basso che coprirebbe metà della pila: all'avvio resta ridotto.
  if (window.matchMedia("(max-width: 900px)").matches) codePanel.root.classList.add("minimized");
  // Le azioni di questo tab (actions/, context/) inquadrano la scena sapendo che sul telefono il
  // codice è un foglio in basso: niente spostamento in alto durante le azioni (regola di core.css).
  root.classList.add("code-stays-bottom");
  window.__gptApp = tab; // utile per il debug dalla console e per i test automatici

  return {
    tab,
    activate: () => {
      host.start();
      host.resize();
    },
    deactivate: () => host.stop(),
  };
}

/**
 * Scelta di un livello dal selettore dello zoom: la camera inquadra quel livello (senza, scegliere
 * "Panoramica" da vicino mostrerebbe un rettangolo enorme tagliato). Durante un'azione la regia
 * della camera è dell'azione, quindi qui non si vola.
 */
function installLevelPicking(tab) {
  const FRONT = new THREE.Vector3(0, 0.08, 1);
  tab.lod.controller.onPick = (mode) => {
    if (mode === "auto" || tab.ui.player?.action) return;
    if (tab.views.current === "architettura") {
      if (mode === "detail") {
        const entry = tab.menu.items.get(tab.menu.current);
        const focus = entry?.item.focus;
        tab.focus(focus && focus !== "overview" ? focus : "blocks.0");
      } else {
        // sul telefono le due file di comandi in alto occupano ~100 px: serve più margine
        const margin = window.matchMedia("(max-width: 900px)").matches ? 1.45 : 1.25;
        tab.focus(tab.lod.arch[mode], { showFrame: false, margin, direction: FRONT });
      }
      return;
    }
    // Altre viste: inquadra ciò che di quel livello è visibile ora (es. il diagramma della vista Contesto).
    if (mode === "detail") return;
    const box = meshBounds(tab.lod.levels[mode], { visibleOnly: true });
    if (!box.isEmpty()) tab.ui.director.focus(box, { showFrame: false, margin: 1.12, direction: FRONT });
  };
}

/** Elenco delle fonti raggruppate (stile condiviso .sources-list di core.css). */
function renderSources(groups) {
  return el(
    "div",
    { className: "sources-list" },
    ...groups.map((group) =>
      el(
        "section",
        {},
        el("h3", {}, group.title),
        el("ul", {}, ...group.items.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener" }, s.title), el("p", { className: "hint" }, s.note)))),
      ),
    ),
  );
}
