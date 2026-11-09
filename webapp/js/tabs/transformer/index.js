/**
 * Tab "Transformer": il GPT del corso in 3D.
 *
 * Questo file è il punto di raccordo: costruisce layout, scena, menu e il gestore delle
 * viste, poi installa i moduli specializzati (ognuno in un file proprio):
 *   lod.js           zoom semantico Dettaglio / Blocchi / Panoramica
 *   views/weights.js vista "Pesi" (sovrapposizione: dove sono i parametri)
 *   views/neurons.js vista "Rete neurale" (tutti i neuroni e le connessioni)
 *   views/numbers.js vista "Pesi numerici" (i numeri veri nelle matrici, anche durante il training)
 *   actions/index.js azioni animate: prompt → token → … → output, training (+ grafici)
 *   context/index.js vista "Contesto": RAG + system prompt + tool → richiesta all'LLM
 * Ogni modulo esporta `install(tab)`; l'oggetto `tab` è descritto in webapp/ARCHITECTURE.md.
 * Rappresentazioni del nucleo (selettore "Architettura · Rete neurale · Cervello"): installReps qui sotto.
 */
import * as THREE from "three";
import { meshBounds } from "../../core/director.js";
import { el } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { installRepresentations } from "../../core/representations.js";
import { installStepKeys } from "../../core/stepKeys.js";
import { Menu } from "../../core/ui/menu.js";
import { installActions } from "./actions/index.js";
import { buildArchitecture } from "./architecture.js";
import { SOURCES, STEPS } from "./content.js";
import { loadEngine } from "../../engine/gpt.js";
import { installContext } from "./context/index.js";
import { loadModelSpec } from "./data.js";
import { DataFlow } from "./flow.js";
import { installLod } from "./lod.js";
import { NEURON_LINKS, installNeuronsView } from "./views/neurons.js";
import { BIRTH_VIEW_ID, installNumbersView } from "./views/numbers.js";
import { installWeightsView } from "./views/weights.js";
import { modelSummaryTable, weightsTable } from "./weightsTables.js";

/** Schermi stretti (telefono): sidebar a cassetto, codice in basso, due file di comandi in alto (core.css). */
const narrowScreen = () => window.matchMedia("(max-width: 900px)").matches;

/**
 * Gestore delle viste: una sola attiva alla volta, elencate nel pannello "Viste" della sidebar.
 * Ogni vista può dire qual è la sua inquadratura iniziale (`home`): la usano "Centra" della
 * navigazione libera (tasto C) e il pulsante "Vista d'insieme".
 */
class ViewManager {
  constructor(tab, pane) {
    this.tab = tab;
    this.pane = pane;
    this.views = [];
    this.current = null;
    this.list = el("div", { className: "view-switch" });
    pane.append(this.list);
  }

  /**
   * view = { id, label, description, order?, enter(tab), exit(tab), home?(tab) → Promise, flow? }
   * flow: true se nella vista il flusso dei dati gira da solo sul binario (il pulsante "Flusso" ha senso)
   */
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
    // Sul telefono il codice è un foglio che coprirebbe metà della scena: in una vista conta la scena,
    // il codice resta ridotto alla sua intestazione (un tocco lo riapre).
    if (narrowScreen()) this.tab.ui.codePanel?.root.classList.add("minimized");
    this.tab.events.dispatchEvent(new CustomEvent("view", { detail: id }));
  }

  has(id) {
    return this.views.some((v) => v.id === id);
  }

  /** La vista attiva (oggetto registrato) o null. */
  get currentView() {
    return this.views.find((v) => v.id === this.current) ?? null;
  }

  /** Vola sull'inquadratura iniziale della vista attiva (predefinita: la pila intera). */
  home() {
    const view = this.views.find((v) => v.id === this.current);
    return view?.home ? view.home(this.tab) : this.tab.focus("overview");
  }
}

export async function createTab(root, app) {
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint: "Scegli uno step o un'azione: la camera vola sul componente e a destra compare il codice. Poi muoviti come vuoi: trascina per ruotare, tasto destro o W A S D per spostarti, doppio clic su un punto per avvicinarti, C per tornare alla vista d'insieme (? mostra tutti i comandi). Più ti allontani, più la vista si semplifica.",
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
      // 1.25: la riga di uscita in alto non finisce sotto la barra dei comandi dello stage;
      // sul telefono le due file di comandi in alto e il codice ridotto in basso tolgono ~1/3 dell'altezza
      if (target === "overview") return director.focus(arch.registry.get("overview"), { showFrame: false, margin: narrowScreen() ? 1.6 : 1.25, ...options });
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
    tabId: "transformer",
    onSelect: async (item, title) => {
      // I passi parlano della pila: se la vista attiva la nasconde (Rete neurale), si torna all'Architettura.
      if (!arch.root.visible) await tab.views.activate("architettura");
      // un passo inquadra un componente da vicino: lo zoom torna su Auto, così si vede il dettaglio
      // anche se prima lo studente aveva bloccato "Panoramica" (durante un'azione decide l'azione)
      if (!ui.player?.action) tab.lod?.controller.setMode("auto");
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
    flow: true,
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
  installNumbersView(tab);
  installActions(tab);
  installContext(tab);
  // dopo i pannelli e PRIMA di "Fonti": il pannello "Cervello" del nucleo sta prima delle fonti
  installReps(tab);
  sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Da dove vengono codice, architettura e scelte di training." }).append(renderSources(SOURCES));

  // --- controlli generali
  const playFlow = installFlowButton(tab);
  // "Vista d'insieme" fa la stessa cosa di "Centra" (tasto C): l'inquadratura iniziale della vista attiva.
  const overview = el("button", { type: "button", className: "overview-button", title: "Torna alla vista d'insieme (tasto C)", onclick: () => ui.nav.home() }, "Vista d'insieme");
  stage.append(el("div", { className: "stage-toolbar" }, overview, playFlow));
  installStepKeys({ root, ui, menu: () => tab.menu, pathPane: tab.panes.path }); // ↑ ↓ Percorso, Spazio player

  installLevelPicking(tab);
  installNavigation(tab);

  // Altri tab possono chiedere una vista precisa (es. il tab RAG apre la vista "contesto").
  // detail = id della vista, oppure { view, question } per passare anche la domanda alla vista Contesto.
  app.bus.addEventListener("transformer:view", (e) => {
    const { view, question } = typeof e.detail === "string" ? { view: e.detail } : e.detail ?? {};
    app.open("transformer");
    if (question) app.bus.dispatchEvent(new CustomEvent("context:question", { detail: { question } }));
    if (!view) return;
    // una richiesta che non passa da un gesto nel tab: prima la scena del tab (si chiude un Cervello aperto)
    tab.reps?.sync(view === "neuroni" ? "network" : "main");
    tab.views.activate(view);
  });

  tab.menu.select("data.dataset");
  tab.focus("overview");
  // Sul telefono il codice è un foglio in basso che coprirebbe metà della pila: all'avvio resta ridotto.
  if (narrowScreen()) codePanel.root.classList.add("minimized");
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
    deactivate: () => host.stop(), // la navigazione si ferma da sola (niente scivolate al ritorno)
    goto: (target) => gotoTarget(tab, target),
  };
}

/**
 * Pulsante "Flusso" della barra dello stage: ferma o fa ripartire il pacchetto che scorre sul binario.
 * L'etichetta segue SEMPRE lo stato vero (anche quando una vista o un'azione fermano il flusso da sole,
 * evento onChange di DataFlow) e il pulsante si disattiva dove il flusso non si vede o non è suo: nelle
 * viste senza `flow: true` (Rete neurale, Pesi numerici, Contesto) e durante un'azione, che muove il
 * pacchetto da sé. Prima, nella Rete neurale, faceva partire un flusso invisibile che sostituiva la
 * didascalia e restava "⏸ Flusso" anche a flusso fermo.
 */
function installFlowButton(tab) {
  const { ui, flow } = tab;
  const available = () => tab.views.currentView?.flow === true && !ui.player?.action;
  const button = el("button", { type: "button", className: "flow-toggle", onclick: () => available() && flow.toggle() });
  const sync = () => {
    const can = available();
    button.textContent = flow.playing ? "⏸ Flusso" : "▶ Flusso";
    button.disabled = !can;
    button.setAttribute("aria-label", flow.playing ? "Ferma il flusso dei dati" : "Fai scorrere il flusso dei dati");
    button.title = can
      ? flow.playing
        ? "Ferma il pacchetto dei dati che scorre sul binario."
        : "Fai ripartire il pacchetto dei dati che scorre sul binario."
      : ui.player?.action
        ? "Durante un'azione il flusso dei dati lo gestisce l'azione (alcune lo fermano, altre muovono il pacchetto da sé)."
        : "Il flusso dei dati scorre nelle viste Architettura e Pesi.";
  };
  flow.onChange(sync);
  tab.events.addEventListener("view", sync);
  ui.player?.on("state", sync);
  sync();
  return button;
}

/**
 * Rappresentazioni del nucleo (core/representations.js): "Architettura" è la scena del tab con le sue
 * viste; "Rete neurale" è la vista `neuroni` del tab (modalità personalizzata { enter, exit }: stessa
 * scena, la disposizione che lo studente aveva scelto, "A righe" se non ne ha scelta una, con
 * l'interruttore visibile); "Cervello" è il livello del nucleo con brain.js. Lasciando la Rete neurale
 * dal selettore si torna alla vista di prima (o all'Architettura). Quando il tab cambia vista da solo
 * (pannello Viste, passo del Percorso, azione, link) il selettore si risincronizza (reps.sync).
 */
function installReps(tab) {
  const { ui, app } = tab;
  let back = "architettura"; // la vista prima della Rete neurale
  let leaving = false; // exit() in corso: l'evento "view" che ne segue è nostro, non va risincronizzato
  const reps = installRepresentations(ui, {
    tabId: "transformer",
    app,
    network: {
      description: "Tutti i neuroni del GPT del corso e i pesi che li collegano: a righe, un blocco per riga, oppure da sinistra a destra con i token del prompt in profondità (interruttore in alto a sinistra).",
      enter: () => tab.views.activate("neuroni"),
      exit: async () => {
        if (tab.views.current !== "neuroni") return;
        leaving = true;
        try {
          await tab.views.activate(tab.views.has(back) ? back : "architettura");
        } finally {
          leaving = false;
        }
      },
    },
    brain: { load: () => import("./brain.js").then((m) => m.brainSpec) },
  });
  tab.events.addEventListener("view", (e) => {
    if (e.detail !== "neuroni") back = e.detail;
    if (!leaving) reps.sync(e.detail === "neuroni" ? "network" : "main");
  });
  tab.reps = reps;
  return reps;
}

/**
 * Navigazione libera (ui.nav del nucleo) adattata al tab:
 * - "Centra" (C, Home, pulsante del pad) = l'inquadratura iniziale della vista attiva;
 * - confini = ciò che si vede nella vista attiva, calcolato sui contenuti fissi della scena
 *   (non sui livelli dello zoom che si accendono e spengono, così il bordo non cambia allo zoom).
 */
function installNavigation(tab) {
  const { ui, arch } = tab;
  ui.nav.setHome(() => tab.views.home());
  const box = new THREE.Box3();
  const part = new THREE.Box3();
  const levelGroups = new Set(Object.values(tab.lod.levels));
  const archLevels = new Set([arch.root, tab.lod.arch.blocks, tab.lod.arch.macro]);
  const add = (object) => box.union(part.copy(meshBounds(object, { visibleOnly: true })));
  ui.nav.setBounds(() => {
    box.makeEmpty();
    // La pila conta tutta, a ogni livello (i rettangoli di Blocchi/Panoramica stanno nelle stesse
    // coordinate): il livello che si accende o si spegne zoomando non sposta i confini.
    if (arch.root.visible) {
      add(arch.root);
      add(tab.lod.arch.macro);
    }
    for (const child of ui.host.scene.children) {
      if (!child.visible) continue;
      if (!levelGroups.has(child)) add(child); // binario, lavagna delle azioni…
      // dentro i livelli: ciò che le viste aggiungono (rete neurale, diagramma del Contesto)
      else for (const c of child.children) if (c.visible && !archLevels.has(c)) add(c);
    }
    return box;
  });
}

/**
 * Link diretti e ricerca globale (app.goto → instance.goto): porta il tab su un passo del Percorso,
 * un'azione, una vista, un livello dello zoom o un passo della vista Contesto. Gli id sono quelli
 * di search.js. Restituisce false se l'id non esiste (il guscio avvisa lo studente).
 */
async function gotoTarget(tab, target = {}) {
  const { ui } = tab;
  const { kind, id } = target;
  // un salto chiesto dallo studente: i voli che seguono partono anche con la camera libera
  ui.director.noteUserIntent(1500);
  // view:rete e view:cervello mostrano la rappresentazione; per tutto il resto un Cervello aperto si chiude
  const shown = tab.reps?.goto(target);
  if (shown) return shown;
  // tensor:<nome>[@<step>]: la vista "Pesi numerici" su quel tensore (lo stesso formato del Laboratorio pesi)
  if (kind === "tensor") return gotoTensor(tab, id);
  if (kind === "step") {
    if (!tab.menu.items.has(id)) return false;
    stopOtherAction(tab, null);
    await tab.views.activate("architettura");
    return tab.menu.reveal(id, { sidebar: ui.sidebar });
  }
  if (kind === "action") {
    if (!ui.player?.actions.has(id)) return false;
    ui.sidebar.select("azioni"); // l'azione può poi scegliere il suo pannello (es. Contesto, Viste)
    showSceneOnPhone(tab);
    void ui.player.start(id); // la preparazione può richiedere il motore: non la aspettiamo
    return true;
  }
  if (kind !== "view") return false;
  // vista Rete neurale in una disposizione: view:neuroni (a righe, come sempre) e view:neuroni-profondita
  if (Object.hasOwn(NEURON_LINKS, id)) {
    stopOtherAction(tab, null);
    await tab.neuronsView.open(NEURON_LINKS[id]);
    ui.sidebar.select("viste");
    showSceneOnPhone(tab);
    return true;
  }
  // "i pesi che nascono": la vista Pesi numerici sul training, dallo step 0 (▶ per farli nascere)
  if (id === BIRTH_VIEW_ID) {
    stopOtherAction(tab, null);
    await tab.numbersView.open({ birth: true });
    ui.sidebar.select("viste");
    showSceneOnPhone(tab);
    return true;
  }
  if (tab.views.has(id)) {
    stopOtherAction(tab, null);
    await tab.views.activate(id);
    if (id !== "contesto") ui.sidebar.select("viste");
    showSceneOnPhone(tab);
    return true;
  }
  const level = /^zoom\.(detail|blocks|macro)$/.exec(id)?.[1];
  if (level) {
    stopOtherAction(tab, null);
    // I testi di questi risultati parlano della pila (matrici, rettangoli, diagramma del paper): anche
    // da Contesto o Pesi, dove la pila si vede ma il livello inquadrerebbe altro, si torna all'Architettura.
    await tab.views.activate("architettura");
    tab.lod.controller.pick(level); // come se lo studente lo scegliesse dal selettore: blocca il livello e lo inquadra
    ui.sidebar.select("viste"); // lì c'è "Zoom semantico: cosa vedi a ogni livello"
    showSceneOnPhone(tab);
    return true;
  }
  if (id.startsWith("ctx.") && tab.contextView) {
    stopOtherAction(tab, "context-pipeline");
    return tab.contextView.reveal(id);
  }
  return false;
}

/** tensor:<nome>[@<step>] → vista Pesi numerici su quel tensore, ai pesi finali o allo step chiesto. */
async function gotoTensor(tab, id) {
  // moduli caricati solo qui (e dalla vista): l'avvio del tab non li scarica
  const [run, { parseTensorTarget }] = await Promise.all([import("../../shared/weights/run.js").then((m) => m.loadBuiltinRun()), import("./views/numbersModel.js")]);
  const target = parseTensorTarget(id, { names: new Set(run.tensors.keys()), steps: run.steps });
  if (!target) return false;
  stopOtherAction(tab, null);
  const ok = await tab.numbersView.open({ tensor: target.name, step: target.step, interpolate: target.interpolate });
  if (!ok) return false;
  tab.ui.sidebar.select("viste");
  showSceneOnPhone(tab);
  return true;
}

/**
 * Sul telefono la sidebar è un cassetto sopra la scena: per una vista, un livello o un'azione
 * conta la scena, quindi il cassetto si chiude (per un passo invece resta aperto sul suo testo).
 */
function showSceneOnPhone(tab) {
  if (narrowScreen()) tab.ui.sidebar.toggleCollapsed(true);
}

/** Un salto della ricerca chiude l'azione in corso (tranne `keep`): la regia torna allo studente. */
function stopOtherAction(tab, keep) {
  const player = tab.ui.player;
  if (player?.action && player.action.id !== keep) player.stop();
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
        const margin = narrowScreen() ? 1.45 : 1.25;
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
