/**
 * Tab "LLM moderni": cosa hanno in più i modelli di oggi rispetto al nostro GPT didattico.
 * Due sottotab (barra in cima allo stage, subtabs.js):
 *   "Vista d'insieme tecnologie LLM moderni"   16 meccanismi, uno per regione (questo file)
 *   "Architettura LLM moderni completa"         un modello aperto vero smontato pezzo per pezzo
 *                                               (full/index.js, caricato al primo ingresso)
 *
 * Come è fatto (un modulo per cosa):
 *   regions.js        mappa della scena: 16 regioni su una griglia 4 × 4 (una per meccanismo)
 *   region*.js        livello Dettaglio di ogni regione, con una piccola API usata dalle azioni
 *   overview.js       livelli Blocchi (un rettangolo per meccanismo) e Panoramica (pipeline)
 *   actions*.js       azioni animate per categoria (Architettura, Addestramento, Ragionamento
 *                     e agenti, Oltre il testo)
 *   plots.js          grafici; sim.js matematica pura (formule pubblicate e simulazioni dichiarate)
 *   content.js        percorsi didattici (OVERVIEW_PATH qui, FULL_PATH della vista completa), schede
 *                     dei modelli, fonti; snippets.js codice vero
 *   ourModel.js       numeri del nostro GPT (data/model.json) per i testi con segnaposto {{chiave}}
 *   regionData.js, actionCatalog.js, palette.js, full/catalog.js, full/models/   soli dati (li usa
 *                     anche search.js, l'indice della ricerca globale, caricato senza la scena)
 *   regionCompass.js  "Dove sei": la regione su cui si trova chi passeggia nella scena
 *   subtabs.js        la barra dei sottotab; sceneClick.js il clic sulla scena (non trascinamento)
 *
 * Una sola scena per i due sottotab (un SceneHost, nessun contesto WebGL in più): cambiando
 * sottotab l'azione in corso si ferma, codice e didascalia si azzerano, ogni sottotab ricorda la sua
 * camera, si vede un solo selettore dello zoom e solo i pannelli e i grafici di quel sottotab.
 *
 * Navigazione libera (ui.nav, dal nucleo): "Centra" torna alla mappa dei 16 meccanismi (o
 * all'architettura intera), i confini sono quelli del sottotab; la camera non va dietro né sopra la
 * parete, né dentro gli schemi.
 * Link diretti e ricerca: goto({kind, id}) con
 *   step:<id del Percorso>, action:<id dell'azione>, view:macro|blocks|detail|<id di una regione>,
 *   view:overview | view:full | view:full-<…> | action:full-<…> | step:full-<…> (full/catalog.js).
 */
import * as THREE from "three";
import { el, fillTemplate, fmt, sleep } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { installStepKeys } from "../../core/stepKeys.js";
import { LODController } from "../../core/lod.js";
import { parseHash } from "../../core/search.js";
import { Menu } from "../../core/ui/menu.js";
import { ACTION_CATALOG, ACTION_IDS } from "./actionCatalog.js";
import { architectureActions } from "./actionsArch.js";
import { beyondActions } from "./actionsBeyond.js";
import { inferenceActions } from "./actionsInference.js";
import { reasoningActions } from "./actionsReasoning.js";
import { trainingActions } from "./actionsTraining.js";
import { MODEL_CARDS, OVERVIEW_PATH, SOURCES } from "./content.js";
import { FULL_ACTION_IDS } from "./full/catalog.js";
import { FULL_PATH } from "./full/content.js";
import { MODEL_BY_ID, PRIMARY_MODEL_ID } from "./full/models/index.js";
import { buildBlocks, buildMacro, macroBox } from "./overview.js";
import { loadModel, modelVars } from "./ourModel.js";
import { installPlots } from "./plots.js";
import { buildAgent } from "./regionAgent.js";
import { buildBpe } from "./regionBpe.js";
import { buildCoding } from "./regionCoding.js";
import { buildCompare } from "./regionCompare.js";
import { buildFlash } from "./regionFlash.js";
import { buildGqa } from "./regionGqa.js";
import { buildInterp } from "./regionInterp.js";
import { buildKv } from "./regionKv.js";
import { buildMesh } from "./regionMesh.js";
import { buildMoe } from "./regionMoe.js";
import { buildMulti } from "./regionMulti.js";
import { buildReasoning } from "./regionReasoning.js";
import { buildRope } from "./regionRope.js";
import { buildScaling } from "./regionScaling.js";
import { buildSpec } from "./regionSpec.js";
import { buildTraining } from "./regionTraining.js";
import { RegionCompass } from "./regionCompass.js";
import { gridBox, nearestRegionAt, REGION_BY_ID, REGION_H, REGION_W, REGIONS, ROWS, regionBox, regionCenter, regionShell, STEP_X, STEP_Y } from "./regions.js";
import { installSceneClick } from "./sceneClick.js";
import { Snippets } from "./snippets.js";
import { createSubtabs, RESET_TITLE } from "./subtabs.js";

const BUILDERS = {
  compare: buildCompare,
  moe: buildMoe,
  gqa: buildGqa,
  rope: buildRope,
  bpe: buildBpe,
  kv: buildKv,
  flash: buildFlash,
  spec: buildSpec,
  scaling: buildScaling,
  training: buildTraining,
  reasoning: buildReasoning,
  interp: buildInterp,
  agent: buildAgent,
  coding: buildCoding,
  multi: buildMulti,
  mesh: buildMesh,
};

const CATEGORIES = ["Architettura", "Addestramento", "Ragionamento e agenti", "Oltre il testo"];
/** Schemi e testi si guardano quasi di fronte: il testo resta leggibile. */
const FRONT = new THREE.Vector3(0, 0.06, 1).normalize();
/**
 * Durante un'azione la barra del player copre il fondo dello stage: il riquadro da inquadrare si
 * allunga verso il basso (in unità di scena). Sugli schermi stretti la barra occupa più spazio.
 */
const PLAYER_PAD = 9;
const PLAYER_PAD_NARROW = 16;
/** Soglie del LOD (distanza camera → punto guardato) pensate per uno stage più largo che alto. */
const LOD_DETAIL_MAX = 74;
const LOD_BLOCKS_MAX = 235;
/**
 * La scena è una parete verticale (piano x-y) con testo, guardata di fronte: da sopra o da sotto si
 * vedrebbe solo una linea, da dietro il testo a specchio, e troppo vicino si finisce dentro gli schemi.
 * Si gira attorno e ci si avvicina liberamente, ma entro questi limiti (mouse, tasti, pad e touch);
 * gli stessi del tab Audio, che ha una scena simile. Valgono per entrambi i sottotab.
 */
const MAX_TURN_DEG = 75;
const MAX_TILT_DEG = 60;
const MIN_ZOOM_DISTANCE = 6;
/**
 * Schermo stretto: stessa soglia di core/ui/charts.js. Sotto i 900 px i pannelli non stanno accanto
 * alla scena ma sopra (grafici a tutto schermo, codice in basso).
 */
const NARROW = window.matchMedia("(max-width: 899px)");
/** Sidebar a cassetto sopra la scena (stessa soglia di core/layout.js e core.css). */
const DRAWER = window.matchMedia("(max-width: 900px)");
/** Confini della navigazione libera: la griglia dei meccanismi, oppure il diagramma della Panoramica. */
const GRID_BOX = gridBox();
const MACRO_BOX = macroBox();
/**
 * Fin dove può arrivare il punto guardato fuori dalla Panoramica: il riquadro dei centri delle
 * regioni (il nucleo lo allarga del 12%, quindi il bordo cade circa sul bordo delle regioni esterne).
 * Così, anche da vicino al livello Dettaglio, al confine metà dello schermo mostra ancora una regione;
 * con l'intera griglia (GRID_BOX) si poteva scivolare di quasi una regione nel vuoto.
 */
const WALK_BOX = new THREE.Box3(new THREE.Vector3(-1.5 * STEP_X, -1.5 * STEP_Y, -1), new THREE.Vector3(1.5 * STEP_X, 1.5 * STEP_Y, 1));
/** Primo passo del Percorso e prima azione che parlano di una regione (per "Leggi" e "▶ Anima"). */
const STEP_OF_REGION = new Map();
for (const section of OVERVIEW_PATH) for (const item of section.children) if (item.focus && !STEP_OF_REGION.has(item.focus)) STEP_OF_REGION.set(item.focus, item.id);
const ACTION_OF_REGION = new Map();
for (const a of ACTION_CATALOG) if (!ACTION_OF_REGION.has(a.region)) ACTION_OF_REGION.set(a.region, a.id);

/** I due sottotab (etichette brevi sugli schermi stretti, titolo intero nel tooltip). */
const SUBTABS = [
  { id: "overview", label: "Vista d'insieme tecnologie LLM moderni", short: "Vista d'insieme" },
  { id: "full", label: "Architettura LLM moderni completa", short: "Architettura completa" },
];
/** Pannelli corrispondenti nei due sottotab: cambiando sottotab si resta sullo stesso tipo di pannello. */
const PANE_PAIRS = [
  ["percorso", "full-percorso"],
  ["azioni", "full-azioni"],
  ["viste", "full-viste"],
  ["modelli", "full-modelli"],
  ["fonti", "full-fonti"],
];
const OVERVIEW_PANES = ["percorso", "azioni", "viste", "modelli", "fonti"];
const FULL_STEP_IDS = new Set(FULL_PATH.flatMap((s) => s.children.map((c) => c.id)));

export async function createTab(root, app) {
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint: "Segui il percorso o avvia un'azione. Rotellina: da vicino i componenti, più lontano un rettangolo per meccanismo, da lontanissimo la pipeline completa. Clic su una regione per avvicinarti; con W A S D (o tasto destro + trascina) passeggi tra i 16 meccanismi e in basso a sinistra leggi dove sei.",
  });
  const { host, director, sidebar, codePanel, stage, nav } = ui;
  // numeri del nostro GPT per i testi (segnaposto {{chiave}}), dal model.json esportato
  const model = await loadModel();
  const vars = modelVars(model);

  // --- scena: Dettaglio (16 regioni), Blocchi, Panoramica
  const detail = new THREE.Group();
  const regions = {};
  const frames = {};
  for (const region of REGIONS) {
    const g = new THREE.Group();
    g.position.copy(regionCenter(region));
    g.userData.regionId = region.id;
    frames[region.id] = regionShell(g, region);
    regions[region.id] = BUILDERS[region.id](g, vars);
    detail.add(g);
  }
  const blocks = buildBlocks();
  const macro = buildMacro();
  blocks.children.forEach((child, i) => (child.userData.regionId = REGIONS[i]?.id));
  host.scene.add(detail, blocks, macro);

  const lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: LOD_DETAIL_MAX, group: detail },
      { id: "blocks", label: "Blocchi", maxDistance: LOD_BLOCKS_MAX, group: blocks },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: macro },
    ],
    widgetParent: stage,
    // Il clic dello studente su un livello lo inquadra sempre, anche se era già quello attivo
    // (dopo una passeggiata "Blocchi" riporta alla mappa intera, "Dettaglio" alla regione dove sei).
    onPick: (mode) => mode !== "auto" && flyToLevel(mode),
  });
  /** "Dove sei" (regionCompass.js): creata più sotto, quando menu e player esistono. */
  let compass = null;

  /**
   * Su uno stage stretto (telefono in verticale, pannelli aperti) la camera deve allontanarsi di più
   * per far stare in larghezza una regione: le soglie del LOD crescono nella stessa proporzione,
   * così una regione inquadrata per intero resta al livello Dettaglio e la griglia al livello Blocchi.
   */
  const REF_ASPECT = REGION_W / REGION_H;
  function scaleLodToStage() {
    const k = Math.max(1, REF_ASPECT / host.visibleAspect());
    lod.levels[0].maxDistance = LOD_DETAIL_MAX * k;
    lod.levels[1].maxDistance = LOD_BLOCKS_MAX * k;
  }
  scaleLodToStage();
  new ResizeObserver(scaleLodToStage).observe(stage);
  ui.charts.onResize(scaleLodToStage);
  codePanel.onResize(scaleLodToStage);

  /** Scelta di un livello: la camera si porta a una distanza adatta ("Dettaglio": la regione dove sei). */
  function flyToLevel(id) {
    if (id === "detail") return tab.focusRegion(nearestRegion().id);
    return frameWhole(id === "blocks" ? GRID_BOX : MACRO_BOX);
  }

  /** Inquadra tutta la griglia (o tutta la Panoramica) di fronte. */
  function frameWhole(box, { duration } = {}) {
    compass?.hide();
    return director.focus(box, { showFrame: false, margin: 1.02, direction: FRONT, duration }); // undefined = durata predefinita
  }

  /** La regione più vicina al punto guardato: con la navigazione libera, quella su cui sei arrivato. */
  function nearestRegion() {
    const t = host.controls.target;
    return nearestRegionAt(t.x, t.y);
  }

  // --- oggetto condiviso con i moduli delle azioni e dei grafici
  const snippets = new Snippets();
  let codeRequest = 0;
  const tab = {
    app,
    ui,
    lod,
    vars,
    regions,
    /** Inquadra una regione di fronte. `pad` in basso lascia spazio alla barra del player. */
    focusRegion(id, { bottomPad = 0, duration = 1.1 } = {}) {
      const region = REGION_BY_ID.get(id);
      if (!region) return Promise.resolve();
      compass?.hide();
      // una regione si guarda al livello Dettaglio: un livello bloccato (Blocchi, Panoramica) torna automatico
      if (lod.mode !== "auto" && lod.mode !== "detail") lod.setMode("auto");
      const box = regionBox(region, 0.4);
      box.min.y -= bottomPad;
      box.max.y += 2.5; // spazio per il selettore dello zoom in alto
      // si aspetta che il pannello del codice richiesto nello stesso momento sia aperto:
      // l'inquadratura deve tenere conto dello spazio che copre
      return sleep(30)
        .then(() => tab.codePending)
        .then(() => director.focus(box, { showFrame: false, margin: 1.03, direction: FRONT, minDistance: 8, duration }));
    },
    /** Preparazione comune delle azioni: livello automatico, grafici richiesti, inquadratura. */
    prepare(id, { charts = null } = {}) {
      lod.setMode("auto");
      if (charts) tab.openCharts(charts);
      tab.actionRegion = id;
      return tab.focusAction();
    },
    /** Inquadra la regione dell'azione in corso, lasciando spazio sotto per la barra del player. */
    focusAction() {
      return tab.focusRegion(tab.actionRegion, { bottomPad: NARROW.matches ? PLAYER_PAD_NARROW : PLAYER_PAD });
    },
    /** Regione inquadrata dall'azione in corso (per "Segui l'azione"). */
    actionRegion: null,
    /**
     * Apre i grafici su una sezione. Con `auto` (aperture decise da un'azione o dal percorso) lo fa
     * solo se il pannello sta accanto alla scena: sugli schermi stretti la coprirebbe tutta e
     * l'animazione non si vedrebbe più; lì i grafici si aprono con il pulsante «Grafici».
     */
    openCharts(sectionId, { auto = true } = {}) {
      if (auto && NARROW.matches) return;
      ui.charts.open();
      if (sectionId) ui.charts.section(sectionId).root?.scrollIntoView({ block: "start", behavior: "smooth" });
    },
    /** Mostra uno snippet: chiave di snippets.js oppure un oggetto {title, file, code, language, highlight}. */
    showCode(keyOrSnippet) {
      const request = ++codeRequest;
      tab.codePending = (async () => {
        const s = typeof keyOrSnippet === "string" ? await snippets.get(keyOrSnippet) : keyOrSnippet;
        if (!s || request !== codeRequest) return;
        codePanel.show({ title: s.title, file: s.file, startLine: s.startLine ?? 1, code: s.code, language: s.language, highlight: s.highlight ?? [] });
        // su schermo stretto il codice coprirebbe la scena: resta ridotto, si espande con il suo pulsante
        if (NARROW.matches) codePanel.root.classList.add("minimized");
      })();
      return tab.codePending;
    },
    codePending: Promise.resolve(),
    hideCode() {
      codeRequest++;
      codePanel.hide();
    },
    setCaption(text) {
      ui.caption.textContent = text;
    },
  };

  // --- sidebar
  const panePath = sidebar.addPane({ id: "percorso", title: "Percorso" });
  const paneActions = sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Animazioni passo per passo: in basso pausa, passo avanti/indietro e velocità. Le simulazioni sono dichiarate." });
  const paneViews = sidebar.addPane({ id: "viste", title: "Viste", hint: "Vai a un meccanismo o a uno dei livelli di zoom." });
  const paneModels = sidebar.addPane({ id: "modelli", title: "Modelli", hint: "Numeri dalle configurazioni e dai paper ufficiali dei modelli aperti." });
  const paneSources = sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Paper, codice e configurazioni ufficiali da cui vengono numeri e formule." });

  // percorso: ogni sottopunto con un'azione ha il pulsante per avviarla; quelli su una regione anche
  // il collegamento all'architettura completa ("Dove sta in DeepSeek-V4.1-Flash")
  const primaryModel = MODEL_BY_ID.get(PRIMARY_MODEL_ID);
  const steps = OVERVIEW_PATH.map((step) => ({
    ...step,
    children: step.children.map((item) => ({
      ...item,
      text: item.text && fillTemplate(item.text, vars),
      extra:
        item.action || item.focus
          ? () =>
              el(
                "div",
                { className: "mllm-step-buttons" },
                item.action ? el("button", { type: "button", className: "mllm-run", onclick: (e) => (e.stopPropagation(), ui.player.start(item.action)) }, "▶ Avvia l'animazione") : null,
                item.focus ? el("button", { type: "button", className: "mllm-where-full", title: `Il punto dell'architettura di ${primaryModel.name} dove vive questo meccanismo`, onclick: (e) => (e.stopPropagation(), crossToFull(item.focus)) }, `Dove sta in ${primaryModel.name} →`) : null,
              )
          : undefined,
    })),
  }));
  let initial = true;
  tab.menu = new Menu(panePath, steps, {
    tabId: "modern-llm",
    onSelect: (item) => {
      if (initial) return; // all'avvio si mostra la vista d'insieme, non la prima regione
      if (item.code) tab.showCode(item.code);
      else tab.hideCode();
      if (item.focus) tab.focusRegion(item.focus);
      if (item.charts) tab.openCharts(item.charts);
      tab.setCaption("");
    },
  });

  // i due Percorsi condividono i progressi del tab: "Azzera" li azzera entrambi, e lo dice
  if (tab.menu.resetButton) tab.menu.resetButton.title = RESET_TITLE;

  const views = buildViews(tab, frameWhole);
  paneViews.append(renderViews(tab, views));
  paneModels.append(renderModels(model));
  paneSources.append(renderSources());

  // --- moduli
  installPlots(tab);
  for (const make of [architectureActions, inferenceActions, trainingActions, reasoningActions, beyondActions]) for (const action of make(tab)) ui.player.register(action);
  const actionsList = el("div", { className: "mllm-actions" });
  paneActions.append(el("p", { className: "mllm-note" }, "Ogni azione inquadra la sua regione e sincronizza codice e grafici. Le animazioni con dati inventati lo dichiarano nella didascalia."), actionsList);
  ui.player.renderOverview(actionsList, { categories: CATEGORIES });
  ui.player.on("end", () => director.hideFrame());
  // Le schede "Azioni" dei due sottotab: si accende quella dell'azione in corso (il player da solo
  // tiene il conto di un solo elenco, l'ultimo disegnato).
  ui.player.on("state", () => {
    const id = ui.player.action?.id ?? null;
    root.querySelectorAll(".mllm-actions .action-card").forEach((c) => c.classList.toggle("active", c.dataset.action === id));
  });
  // Sul telefono la sidebar è un cassetto sopra la scena: un'azione avviata dal suo elenco (o con
  // "Avvia l'animazione" in un passo) partiva dietro il cassetto. Alla partenza di un'azione si chiude.
  let lastRun = null;
  ui.player.on("state", () => {
    const run = ui.player.action ? ui.player.run : null;
    if (run !== null && run !== lastRun && DRAWER.matches) sidebar.toggleCollapsed(true);
    lastRun = run;
  });

  // --- "Dove sei": etichetta della regione più vicina mentre lo studente passeggia
  compass = new RegionCompass({
    ui,
    lod,
    frames,
    onFrame: (id) => tab.focusRegion(id),
    stepFor: (id) => STEP_OF_REGION.get(id) ?? null,
    actionFor: (id) => ACTION_OF_REGION.get(id) ?? null,
    onRead: (stepId) => stepId && tab.menu.reveal(stepId, { sidebar }),
    onAnimate: (actionId) => actionId && ui.player.start(actionId),
  });
  // Un'azione arrivata in fondo non è più "in corso": lo studente può passeggiare senza camera libera.
  // Se poi la fa ripartire (▶ dall'ultimo passo) la camera torna sulla sua regione, come all'avvio:
  // le azioni la inquadrano solo in onStart, che qui non viene richiamato.
  let actionFinished = false;
  ui.player.on("state", () => {
    const p = ui.player;
    if (!p.action) {
      actionFinished = false;
      return;
    }
    compass.hide();
    const finished = !p.playing && p.steps.length > 0 && p.index >= p.steps.length - 1 && p.t >= 1;
    if (subtab === "overview" && actionFinished && !finished && tab.actionRegion) tab.focusAction();
    actionFinished = finished;
  });
  // "Segui l'azione" dopo una passeggiata durante un'azione: le azioni di questo tab inquadrano la loro
  // regione solo all'inizio (nessun volo da recuperare), quindi la si inquadra di nuovo qui.
  director.onFollowChange((following) => {
    if (subtab === "overview" && following && director.actionActive && tab.actionRegion) tab.focusAction();
  });

  // --- clic su una regione (senza trascinare): avvicinati
  // Nel livello Dettaglio un clic sulla regione dove sei già non fa niente: così il doppio clic del
  // nucleo (centra il punto e avvicinati) funziona anche dentro una regione.
  installSceneClick(host.renderer.domElement, (e) => {
    if (subtab !== "overview") return;
    const level = lod.current === "detail" ? detail : lod.current === "blocks" ? blocks : null;
    if (!level) return;
    for (const hit of host.pick(e, [level])) {
      let o = hit.object;
      while (o && !o.userData.regionId) o = o.parent;
      if (!o) continue;
      const id = o.userData.regionId;
      if (lod.current === "detail" && id === nearestRegion().id) return;
      tab.focusRegion(id);
      return;
    }
  });

  // ==============================================================================================
  // Sottotab: barra, camera di ciascuno, collegamenti tra le due viste
  // ==============================================================================================
  let subtab = "overview";
  let full = null; // architettura completa: creata al primo ingresso
  let fullLoading = null;
  let switching = 0;
  /** Camera di ciascun sottotab (tornando si ritrova dov'era). */
  const poses = { overview: null, full: null };
  /** Ritorno dopo un collegamento tra le due viste: { subtab, pose, regionId }. */
  let returnTo = null;

  const bar = createSubtabs({ root, items: SUBTABS, current: "overview", onSelect: (id) => showSubtab(id), isBlocked: () => nav.modalOpen });
  addHelpShortcut(nav);

  const capturePose = () => ({ position: host.camera.position.clone(), target: host.controls.target.clone() });
  const restorePose = (p) => {
    director.cancel();
    // Le OrbitControls tagliano la distanza a maxDistance a ogni frame, e il limite della navigazione è
    // ancora quello dell'altro sottotab (si ricalcola più tardi): un volo di 0,01 s finisce prima che la
    // navigazione lo allarghi, e la camera ricordata tornava più vicina (es. 450 invece di 1.900).
    host.controls.maxDistance = Math.max(host.controls.maxDistance, 1.01 * p.position.distanceTo(p.target));
    return director.flyTo(p.position, p.target, 0.01);
  };

  function loadFull() {
    fullLoading ??= import("./full/index.js").then(({ createFullArchitecture }) => {
      full = createFullArchitecture({ ui, tab, regions: REGIONS, onMarker: (regionId) => crossToOverview(regionId), onModelChange: () => updateBarContext() });
      tab.full = full;
      return full;
    });
    return fullLoading;
  }

  /** Mostra solo i pannelli di un sottotab, restando sullo stesso tipo di pannello. */
  function swapPanes(to) {
    const current = sidebar.current;
    const pair = PANE_PAIRS.find(([a, b]) => a === current || b === current);
    const show = to === "full" ? full.panes : OVERVIEW_PANES;
    const hide = to === "full" ? OVERVIEW_PANES : full?.panes ?? [];
    for (const id of show) sidebar.setPaneHidden(id, false);
    sidebar.select(pair ? pair[to === "full" ? 1 : 0] : show[0]);
    for (const id of hide) sidebar.setPaneHidden(id, true);
  }

  /** Grafici: solo le sezioni del sottotab attivo (quelle della vista completa hanno id "full-…"). */
  function swapCharts(to) {
    for (const [id, section] of ui.charts.sections) section.root.hidden = id.startsWith("full-") !== (to === "full");
  }

  /**
   * Cambia sottotab. `restore`: riporta la camera dov'era in quel sottotab (o, la prima volta,
   * all'inquadratura iniziale); false quando chi chiama inquadra da sé (collegamenti, link diretti).
   */
  async function showSubtab(id, { restore = true } = {}) {
    if (id === subtab) return true;
    const token = ++switching;
    bar.setCurrent(id);
    ui.player.stop();
    director.cancel();
    director.hideFrame();
    if (returnTo?.subtab === id) returnTo = null; // tornato da solo: il ritorno non serve più
    poses[subtab] = capturePose();
    tab.hideCode();
    tab.setCaption("");
    if (id === "full") {
      await loadFull();
      if (token !== switching) return false; // nel frattempo lo studente ha cambiato idea
      lod.setEnabled(false);
      detail.visible = blocks.visible = macro.visible = false;
      compass.setEnabled(false);
      full.activate();
    } else {
      full?.deactivate();
      lod.setEnabled(true);
      compass.setEnabled(true);
    }
    subtab = id;
    root.classList.toggle("mllm-full", id === "full"); // regole CSS solo della vista completa
    swapPanes(id);
    swapCharts(id);
    nav.setBounds(navBounds); // confini da ricalcolare: il contenuto è cambiato
    if (restore) {
      if (poses[id]) await restorePose(poses[id]);
      else if (id === "full") await full.home();
    }
    updateBarContext();
    return true;
  }

  /** Dall'architettura completa (clic su un segnaposto) alla regione della vista d'insieme. */
  async function crossToOverview(regionId) {
    const pose = capturePose();
    const from = full.model;
    await showSubtab("overview", { restore: false });
    returnTo = { subtab: "full", pose, regionId, model: from };
    updateBarContext();
    await tab.focusRegion(regionId);
  }

  /** Dalla vista d'insieme al punto dell'architettura completa dove vive un meccanismo. */
  async function crossToFull(regionId) {
    const pose = capturePose();
    await showSubtab("full", { restore: false });
    returnTo = { subtab: "overview", pose, regionId };
    updateBarContext();
    await full.focusMechanism(regionId);
  }

  async function goBack() {
    const r = returnTo;
    if (!r) return;
    returnTo = null;
    await showSubtab(r.subtab, { restore: false });
    await restorePose(r.pose);
    updateBarContext();
  }

  /** Area di contesto della barra: il ritorno (se c'è) e, nella vista d'insieme, "Dove sta in …". */
  let shownLink = null;
  function updateBarContext() {
    const modelName = (full?.model ?? primaryModel).name;
    const region = subtab === "overview" && lod.current === "detail" ? nearestRegion() : null;
    let back = null;
    if (returnTo) {
      const r = REGION_BY_ID.get(returnTo.regionId);
      back =
        returnTo.subtab === "full"
          ? { label: "← Torna all'architettura completa", short: "← Torna", title: `Torna all'architettura completa di ${returnTo.model?.name ?? modelName}, esattamente dov'eri (${r?.title ?? ""})`, onClick: goBack }
          : { label: "← Torna alla vista d'insieme", short: "← Torna", title: `Torna alla vista d'insieme, esattamente dov'eri (${r?.title ?? ""})`, onClick: goBack };
    }
    const link = region && !(returnTo && returnTo.regionId === region.id) ? { label: `Dove sta in ${modelName}: ${region.title} →`, short: `In ${(full?.model ?? primaryModel).short} →`, title: `Il punto dell'architettura di ${modelName} dove vive «${region.title}»`, onClick: () => crossToFull(region.id) } : null;
    const key = `${back?.label}|${back?.title}|${link?.label}`;
    if (key === shownLink) return;
    shownLink = key;
    bar.setContext({ back, link });
  }
  // la regione "dove sei" cambia solo quando la camera si muove: al più un aggiornamento per frame
  let contextFrame = 0;
  host.controls.addEventListener("change", () => {
    if (subtab !== "overview" || contextFrame) return;
    contextFrame = requestAnimationFrame(() => {
      contextFrame = 0;
      updateBarContext();
    });
  });

  // --- tasti: ↑/↓ scorrono il Percorso del sottotab attivo, Spazio mette in pausa l'azione
  // (core/stepKeys.js); sui pulsanti di "Dove sei" Spazio attiva il pulsante
  const activePathPane = { contains: (node) => (subtab === "full" ? root.querySelector('.sidebar-pane[data-pane="full-percorso"]') : panePath)?.contains(node) ?? false };
  installStepKeys({ root, ui, menu: () => (subtab === "full" ? full?.menu : tab.menu), pathPane: activePathPane, spaceIgnore: ".mllm-where" });

  // Aperto da un link diretto a un passo: si parte da quel passo (goto lo mostrerà subito dopo), così
  // il primo passo del Percorso non viene segnato come visto senza che lo studente l'abbia letto.
  const link = parseHash(location.hash);
  const firstStep = link.tab === "modern-llm" && link.go?.kind === "step" && tab.menu.items.has(link.go.id) ? link.go.id : "idea.ponte";
  tab.menu.select(firstStep, { scroll: false });
  initial = false;
  tab.hideCode();
  frameWhole(GRID_BOX, { duration: 0.01 });

  // --- navigazione libera: "Centra" e confini del sottotab attivo
  function navBounds() {
    if (subtab === "full" && full) return full.bounds();
    return lod.current === "macro" ? MACRO_BOX : WALK_BOX;
  }
  nav.setHome(() => {
    if (subtab === "full" && full) return full.home();
    lod.setMode("auto");
    return frameWhole(GRID_BOX);
  });
  nav.setBounds(navBounds);
  nav.setOrbitLimits({ turn: MAX_TURN_DEG, tilt: MAX_TILT_DEG, minDistance: MIN_ZOOM_DISTANCE });

  tab.subtabs = {
    show: showSubtab,
    get current() {
      return subtab;
    },
    crossToFull,
    crossToOverview,
    goBack,
  };
  window.__modernLlmTab = tab; // per il debug dalla console e per gli screenshot automatici

  /**
   * Ricerca globale e link diretti (#modern-llm/go=<kind>:<id>). Funziona anche alla prima apertura:
   * il guscio la chiama dopo activate(). → false se l'id non esiste (il guscio avvisa lo studente).
   */
  async function goto({ kind, id }) {
    director.noteUserIntent(1500); // i voli che seguono sono chiesti dallo studente
    // Azioni e viste si guardano nella scena: sul telefono il cassetto della sidebar la coprirebbe.
    const showScene = (pane) => (NARROW.matches ? sidebar.toggleCollapsed(true) : sidebar.select(pane));
    if (isFullTarget(kind, id)) {
      if (kind === "step") ui.player.stop();
      await showSubtab("full", { restore: kind === "view" && id === "full" });
      if (kind === "action") showScene("full-azioni");
      else if (kind === "view" && !id.startsWith("full-model-")) showScene("full-viste");
      return full.goto({ kind, id });
    }
    if (kind === "view" && id === "overview") {
      await showSubtab("overview");
      return true;
    }
    if (subtab !== "overview" && (kind === "step" || kind === "action" || kind === "view")) {
      if (!(kind === "step" ? tab.menu.items.has(id) : kind === "action" ? ACTION_IDS.has(id) : views.has(id))) return false;
      await showSubtab("overview");
    }
    if (kind === "step") {
      if (!tab.menu.items.has(id)) return false;
      // come scegliere un'altra strada: l'azione in corso si chiude (codice e camera sono del passo)
      ui.player.stop();
      return tab.menu.reveal(id, { sidebar });
    }
    if (kind === "action") {
      if (!ACTION_IDS.has(id)) return false;
      showScene("azioni");
      ui.player.start(id);
      return true;
    }
    if (kind === "view") {
      const view = views.get(id);
      if (!view) return false;
      ui.player.stop();
      showScene("viste");
      pointAt(view.button);
      tab.hideCode();
      await view.run();
      return true;
    }
    return false;
  }

  /** Mostra per un attimo il pulsante della vista scelta dalla ricerca (le viste qui sono voli, non modi). */
  function pointAt(button) {
    if (!button || NARROW.matches) return;
    button.scrollIntoView({ block: "nearest" });
    button.classList.add("active");
    setTimeout(() => button.classList.remove("active"), 1800);
  }

  return {
    goto,
    activate() {
      host.start();
      host.resize();
    },
    deactivate() {
      ui.player.stop();
      host.stop();
    },
  };
}

/** Gli id che appartengono all'architettura completa (full/catalog.js). */
function isFullTarget(kind, id) {
  if (kind === "step") return FULL_STEP_IDS.has(id);
  if (kind === "action") return FULL_ACTION_IDS.has(id);
  return kind === "view" && (id === "full" || id.startsWith("full-"));
}

/**
 * Il tasto V del tab nell'aiuto della navigazione ("?"): una riga in fondo, se l'aiuto c'è. Il pad
 * del nucleo non ha un'API per le scorciatoie dei tab: si aggiunge una sezione alla sua griglia.
 */
function addHelpShortcut(nav) {
  const grid = nav.pad?.help?.querySelector(".nav-help-grid");
  if (!grid) return;
  grid.append(el("section", {}, el("h3", {}, "Questo tab"), el("dl", {}, el("dt", {}, "V"), el("dd", {}, "Vista d'insieme ↔ Architettura completa"))));
}

/**
 * Le viste del tab per id (pannello "Viste" e goto): i due livelli d'insieme, il Dettaglio della
 * regione dove sei e una vista per meccanismo. `run()` restituisce la promessa del volo.
 */
function buildViews(tab, frameWhole) {
  const views = new Map([
    ["macro", { title: "Panoramica: dalla raccolta dei dati alla risposta", text: "Dati → Pre-training → Post-training → Modello → Inferenza.", run: () => (tab.lod.setMode("macro"), frameWhole(MACRO_BOX)) }],
    ["blocks", { title: "Tutti i meccanismi (livello Blocchi)", text: "Un rettangolo per meccanismo con cosa fa.", run: () => (tab.lod.setMode("blocks"), frameWhole(GRID_BOX)) }],
    ["detail", { title: "Dettaglio: il meccanismo dove sei", text: "I componenti della regione più vicina.", run: () => (tab.lod.setMode("auto"), tab.focusRegion(nearestRegionAt(tab.ui.host.controls.target.x, tab.ui.host.controls.target.y).id)) }],
  ]);
  for (const g of REGIONS) views.set(g.id, { title: g.title, text: g.sub, row: g.row, run: () => (tab.lod.setMode("auto"), tab.focusRegion(g.id)) });
  return views;
}

/** Pannello "Viste": livelli di zoom e un pulsante per meccanismo, raggruppati per riga. */
function renderViews(tab, views) {
  const button = (v) =>
    (v.button = el(
      "button",
      {
        type: "button",
        onclick: () => {
          tab.hideCode();
          v.run();
        },
      },
      el("strong", {}, v.title),
      v.text ? el("small", {}, v.text) : null,
    ));
  return el(
    "div",
    { className: "view-switch mllm-views" },
    button(views.get("macro")),
    button(views.get("blocks")),
    ...ROWS.flatMap((row, r) => [el("h3", { className: "mllm-h3" }, row.title), ...REGIONS.filter((g) => g.row === r).map((g) => button(views.get(g.id)))]),
  );
}

/** Pannello "Modelli": il nostro GPT (dal model.json esportato) e le schede dei modelli aperti. */
function renderModels(model) {
  let ours = null;
  if (model?.config) {
    const c = model.config;
    ours = {
      name: "Il nostro GPT (questa lezione)",
      who: "gpt/ · data/model.json",
      rows: [["parametri", fmt(model.total_params)], ["layer", String(c.n_layer)], ["head", `${c.n_head} (MHA, head da ${model.head_size})`], ["dimensione", String(c.n_embd)], ["MLP", `ReLU, ${4 * c.n_embd}`], ["norm", "LayerNorm"], ["posizioni", `apprese, ${c.block_size}`], ["vocabolario", `${c.vocab_size} caratteri`]],
    };
  }
  const table = (card) =>
    el(
      "article",
      { className: `mllm-model${card.who === "modelli chiusi" ? " closed" : ""}` },
      el("h3", {}, card.name),
      el("p", { className: "hint" }, card.who),
      el("table", {}, el("tbody", {}, ...card.rows.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, v))))),
    );
  return el("div", { className: "mllm-models" }, ours ? table(ours) : null, ...MODEL_CARDS.map(table));
}

/** Pannello "Fonti": elenco ragionato con link. */
function renderSources() {
  return el(
    "div",
    { className: "mllm-sources" },
    el("p", { className: "mllm-note" }, "Le architetture dei modelli chiusi (GPT-4/5, Claude, Gemini) non sono pubbliche: i numeri di questo tab vengono da modelli aperti."),
    ...SOURCES.map((group) =>
      el(
        "section",
        {},
        el("h3", {}, group.title),
        el(
          "ul",
          {},
          ...group.items.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener" }, s.title), s.who ? el("span", { className: "muted" }, ` · ${s.who}`) : null, el("p", { className: "hint" }, s.note))),
        ),
      ),
    ),
  );
}
