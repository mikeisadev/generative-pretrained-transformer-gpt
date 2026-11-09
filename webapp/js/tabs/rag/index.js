/**
 * Tab "Database vettoriale (RAG)": la pipeline delle lezioni L17-L18 come percorso a stazioni.
 *
 * Moduli (uno per responsabilità):
 *   pipeline.js       calcoli veri con js/shared/rag: base di conoscenza, ricerche registrate
 *   answer.js         risposta simulata, estratta dai chunk recuperati
 *   scaling.js        misura HNSW contro ricerca esatta al crescere dei vettori
 *   world.js          livello Dettaglio: stazioni (stationsIngest, stationsStore, stationHnsw, stationsAnswer)
 *   overview.js       livelli Blocchi e Panoramica del LOD
 *   actionIndex.js    azione "Indicizzazione"; actionSearch.js azione "Ricerca"
 *   plots.js          grafici; content.js percorso e fonti; snippets.js / codeSource.js codice
 *   board.js, scene3d.js mattoni grafici comuni
 *   network.js        rappresentazione "Rete neurale" (specifica ricalcolata con la domanda attuale)
 *   brain.js          rappresentazione e pannello "Cervello" (soli dati, dalla ricerca verificata)
 *
 * Eventi tra tab (app.bus):
 *   ascolta  "rag:query"  {detail: {question}}  → imposta la domanda e avvia la ricerca
 *   invia    "transformer:view" {view: "contesto", question} → il tab Transformer mostra la vista Contesto
 * In alternativa all'evento, l'indirizzo #rag/q=<domanda> avvia la ricerca (all'apertura del tab o
 * quando l'indirizzo cambia). I link diretti #rag/go=<kind>:<id> (ricerca globale) arrivano a goto():
 *   step:<id del Percorso>, action:indicizzazione|ricerca, view:detail|blocks|macro (livelli dello zoom),
 *   view:rete|cervello (rappresentazioni del nucleo).
 * L'indice della ricerca globale sta in search.js (solo dati, senza la scena).
 */
import * as THREE from "three";
import { el, fmt } from "../../core/dom.js";
import { meshBounds } from "../../core/director.js";
import { createTabLayout } from "../../core/layout.js";
import { installStepKeys } from "../../core/stepKeys.js";
import { LODController } from "../../core/lod.js";
import { installRepresentations } from "../../core/representations.js";
import { parseHash } from "../../core/search.js";
import { Menu } from "../../core/ui/menu.js";
import { indexAction } from "./actionIndex.js";
import { searchAction } from "./actionSearch.js";
import { composeAnswer } from "./answer.js";
import { LIBRARY, librarySnippet } from "./codeSource.js";
import { PATH, SOURCES } from "./content.js";
import { buildBlocks, buildMacro } from "./overview.js";
import { FEATURED_ID, HNSW_PRESETS, SPLITTER_PRESETS, buildKnowledgeBase, exampleQuestions, runSearch } from "./pipeline.js";
import { RagPlots } from "./plots.js";
import { DOC_COLORS, ROW_BOTTOM, ROW_TOP } from "./scene3d.js";
import { SNIPPETS } from "./snippets.js";
import { World } from "./world.js";
import { CORPUS, EXAMPLE_QUESTIONS } from "../../shared/rag/index.js";

/** Riquadri delle viste d'insieme (larghezza e altezza in unità di scena) e soglie del LOD. */
const SCENE_CENTER = new THREE.Vector3(2, (ROW_TOP + ROW_BOTTOM) / 2 - 2, 0);
const VIEWS = { overview: { w: 318, h: 96, min: 230 }, macro: { w: 430, h: 130 } };
const LOD_DETAIL = 200;
const LOD_BLOCKS = 440;
/**
 * Zoom avanti massimo dello studente (rotella, pizzico, +, doppio clic): più vicino la camera entra
 * nelle carte dei chunk e lo schermo resta nero. I voli del tab restano sopra le 44 unità.
 */
const MIN_ZOOM_DISTANCE = 12;
const MAX_TURN_DEG = 75; // rotazione massima attorno alla scena, dalla vista frontale (come LLM moderni)
const MAX_TILT_DEG = 60; // inclinazione massima dall'alto o dal basso
const FRONT = new THREE.Vector3(0, 0.06, 1).normalize();
const DIRECTIONS = {
  default: new THREE.Vector3(0.1, 0.12, 1).normalize(),
  "hnsw.stack": new THREE.Vector3(0.22, 0.62, 1).normalize(),
  "hnsw.cloud": new THREE.Vector3(0.35, 0.3, 1).normalize(),
  hnsw: new THREE.Vector3(0.12, 0.35, 1).normalize(),
};

/** Id delle azioni (player.register) raggiungibili con un link diretto. */
const ACTION_IDS = new Set(["indicizzazione", "ricerca"]);
/** Didascalia di un livello dello zoom aperto con un link diretto (dice anche come cambiarlo). */
const LEVEL_CAPTIONS = {
  detail: "Dettaglio: ogni chunk, ogni vettore, ogni nodo. Spostati lungo la pipeline per passare da una stazione all'altra; allontanati per lo schema a blocchi.",
  blocks: "Blocchi: le undici fasi, in alto l'indicizzazione e in basso ricerca e generazione. Avvicinati a una fase per vederla da vicino.",
  macro: "Panoramica: il RAG in tre blocchi. Avvicinati per vedere le singole fasi.",
};

/**
 * Suggerimento sopra il Percorso: come cambiare livello di dettaglio e come scorrere lungo la
 * pipeline (che è lunga e orizzontale). Sui touch screen i gesti, altrimenti mouse e tastiera.
 */
const HINT = window.matchMedia("(pointer: coarse)").matches
  ? "Pizzica per lo zoom: da vicino ogni chunk e ogni nodo, da lontano lo schema a blocchi. Due dita: scorri lungo la pipeline."
  : "Rotellina: da vicino ogni chunk e ogni nodo, da lontano lo schema a blocchi. A / D o tasto destro + trascina: scorri lungo la pipeline.";

export async function createTab(root, app) {
  const ui = createTabLayout(root, { code: true, charts: true, player: true, hint: HINT });
  const { host, director, sidebar, codePanel, stage, player } = ui;
  root.classList.add("rag-tab");
  // La pipeline è un plastico guardato da davanti: si gira attorno e si inclina, ma senza arrivare dietro
  // (testo a specchio) né sopra (una linea); il volo più obliquo del tab (pila HNSW) resta ben dentro.
  ui.nav.setOrbitLimits({ turn: MAX_TURN_DEG, tilt: MAX_TILT_DEG, minDistance: MIN_ZOOM_DISTANCE });

  // --- scena: tre livelli di LOD
  const world = new World(host);
  const blocks = buildBlocks();
  const macro = buildMacro();
  host.scene.add(world.group, blocks.group, macro.group);

  let lod = null;
  let onLevelChange = () => {}; // assegnata più sotto, quando confini e didascalie sono pronti
  lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: LOD_DETAIL, group: world.group },
      { id: "blocks", label: "Blocchi", maxDistance: LOD_BLOCKS, group: blocks.group },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: macro.group },
    ],
    widgetParent: stage,
    onChange: (id) => onLevelChange(id),
    // il clic dello studente inquadra sempre il livello scelto, anche se era già quello attivo
    // (es. dopo essersi spostato con il pan: "Blocchi" con il livello già "Blocchi" lo riporta al centro)
    onPick: (mode) => {
      if (mode !== "auto") flyToLevel(mode);
    },
  });
  function flyToLevel(id) {
    if (id === "detail") return tab.focus(tab.lastStation ?? "upload"); // all'inizio: la prima stazione
    return tab.focus(id === "macro" ? "macro" : "overview");
  }

  /** Distanza a cui la vista d'insieme `name` sta tutta nella parte visibile dello stage. */
  function fitDistance(name) {
    const v = VIEWS[name];
    const tan = Math.tan(THREE.MathUtils.degToRad(host.camera.fov / 2));
    return Math.max(v.h / 2 / tan, v.w / 2 / (tan * host.visibleAspect())) * 1.04;
  }

  /**
   * Soglia tra Blocchi e Panoramica. Sugli schermi stretti (telefono in verticale) lo schema a
   * blocchi intero sta nello schermo solo da più lontano: alziamo la soglia quanto serve, così il
   * livello Blocchi resta attivo anche lì. Su schermi larghi resta LOD_BLOCKS.
   */
  function updateBlocksLimit() {
    const limit = Math.max(LOD_BLOCKS, fitDistance("overview") * 1.12);
    lod.levels[1].maxDistance = limit;
    return limit;
  }
  new ResizeObserver(() => updateBlocksLimit()).observe(stage);

  // --- navigazione libera: "Centra" (C, Home, ◎) torna alla vista d'insieme ricalcolata per lo
  // schermo attuale e per i pannelli aperti. I confini seguono il livello dello zoom: da vicino le
  // stazioni (non si scivola nel vuoto oltre l'Upload o l'indice HNSW), in Panoramica lo schema a
  // tre blocchi, più largo. Si calcolano a ogni ricostruzione della scena, non a ogni frame.
  const navBoxes = { detail: new THREE.Box3(), macro: new THREE.Box3() };
  function updateNavBounds() {
    navBoxes.detail.copy(meshBounds(world.group)).union(meshBounds(blocks.group));
    navBoxes.macro.copy(meshBounds(macro.group));
  }
  const navBounds = () => (lod.current === "macro" ? navBoxes.macro : navBoxes.detail);
  // A ogni cambio di livello i confini si aggiornano subito (la navigazione li terrebbe fino a 3 s)
  // e la didascalia di un livello aperto con un link diretto segue lo zoom dello studente: chi apre
  // "Panoramica" e poi si avvicina non legge più "il RAG in tre blocchi" davanti alle stazioni.
  const levelCaptions = new Set(Object.values(LEVEL_CAPTIONS));
  onLevelChange = (id) => {
    ui.nav.setBounds(navBounds);
    if (levelCaptions.has(ui.caption.textContent)) ui.caption.textContent = LEVEL_CAPTIONS[id];
  };
  ui.nav.setHome(() => {
    // la vista d'insieme è quella dell'avvio: anche lo zoom torna automatico (se lo studente aveva
    // bloccato "Panoramica", da questa distanza vedrebbe ancora i tre blocchi grandi)
    if (lod.mode !== "auto") lod.setMode("auto");
    return tab.focus("overview");
  });

  // --- stato: configurazione, base di conoscenza, ricerca corrente
  const settings = { splitter: "didactic", hnsw: "didactic", question: EXAMPLE_QUESTIONS[2].question, topK: 5, threshold: 0.5 };
  const caches = { search: new Map(), answer: new Map(), recall: null };
  let codeToken = 0;

  const tab = {
    app,
    ui,
    world,
    lod,
    settings,
    kb: null,
    plots: new RagPlots(ui.charts),
    lastStation: null,

    /** Inquadra una stazione, una sua parte, un elenco di stazioni o una vista d'insieme. */
    focus(target, options = {}) {
      return Promise.resolve().then(() => {
        if (target === "overview" || target === "macro") {
          const limit = updateBlocksLimit();
          const fit = fitDistance(target);
          const d = target === "macro" ? Math.max(fit, limit * 1.15) : Math.min(Math.max(fit, VIEWS.overview.min), limit * 0.92);
          director.hideFrame();
          return director.flyTo(SCENE_CENTER.clone().addScaledVector(FRONT, d), SCENE_CENTER, options.duration ?? 1.2);
        }
        const s = world.stations;
        let object;
        if (Array.isArray(target)) object = { userData: { members: target.map((id) => s[id].bounds) } };
        else if (target.startsWith("hnsw.")) object = s.hnsw.parts[target.slice(5)];
        else object = s[target]?.bounds;
        if (!object) return undefined;
        tab.lastStation = Array.isArray(target) ? target[0] : target.split(".")[0];
        const direction = DIRECTIONS[Array.isArray(target) ? "default" : target] ?? DIRECTIONS.default;
        // durante un'azione la barra del player copre il fondo dello stage: allunghiamo il
        // riquadro verso il basso così la stazione resta nella parte libera;
        // in alto ci sono il titolo della stazione e i comandi dello zoom
        const box = meshBounds(object);
        const h = box.max.y - box.min.y;
        box.max.y += h * 0.05;
        if (player.action) box.min.y -= h * 0.3;
        return director.focus(box, { showFrame: false, margin: 1.04, direction, ...options });
      });
    },

    /** Codice nel pannello: chiave di SNIPPETS (dispense) oppure "lib:<chiave>" (file veri della libreria). */
    code(key, highlight = []) {
      const token = ++codeToken;
      codePanel.root.hidden = false; // subito: l'inquadratura deve tenere conto del pannello
      if (key.startsWith("lib:")) {
        const spec = LIBRARY[key.slice(4)];
        librarySnippet(spec.file, spec.start, spec.end)
          .then((s) => token === codeToken && showCode({ title: spec.title, file: s.file, startLine: s.startLine, code: s.code, language: "javascript", highlight }))
          .catch((err) => console.warn("codice della libreria non disponibile:", err.message));
        return;
      }
      const s = SNIPPETS[key];
      const language = s.language === "sql" || s.language === "properties" ? "bash" : s.language;
      showCode({ title: s.source, file: s.file, startLine: s.startLine ?? 1, code: s.code, language, highlight });
    },

    /** Nasconde il codice (e annulla un frammento della libreria ancora in caricamento). */
    hideCode() {
      codeToken++;
      codePanel.hide();
    },

    /** Ricerca con i parametri attuali (memorizzata: stessi parametri, stesso risultato). */
    currentSearch() {
      const key = `${settings.question}|${settings.topK}|${settings.threshold}`;
      if (!caches.search.has(key)) caches.search.set(key, runSearch(tab.kb, settings.question, { topK: settings.topK, similarityThreshold: settings.threshold }));
      return caches.search.get(key);
    },

    answerFor(run) {
      if (!caches.answer.has(run)) caches.answer.set(run, composeAnswer({ vector: run.vector, results: run.results, embedder: tab.kb.embedder }));
      return caches.answer.get(run);
    },

    /** Recall media dell'indice sulle domande d'esempio (punto "corpus del corso" nel grafico). */
    corpusRecall() {
      if (caches.recall === null) {
        const runs = EXAMPLE_QUESTIONS.map((q) => runSearch(tab.kb, q.question));
        caches.recall = runs.reduce((s, r) => s + r.recall, 0) / runs.length;
      }
      return caches.recall;
    },

    /** Database pieno (senza animazione) e grafici dell'indicizzazione coerenti. */
    ensureIndexed() {
      if (world.indexed >= tab.kb.inserts.length) return;
      world.showIndexed();
      tab.plots.setIndexed(tab.kb, tab.kb.inserts.length);
    },

    /** Prima di ogni azione: zoom automatico (le azioni inquadrano i dettagli). */
    prepareAction() {
      if (lod.mode !== "auto") lod.setMode("auto");
      codeExpandedByUser = false;
      ui.caption.textContent = ""; // durante le azioni parla la didascalia del player
    },

    /** Stato della scena da ripristinare se un'azione viene interrotta. */
    snapshot() {
      return { indexed: world.indexed >= tab.kb.inserts.length, search: world.search };
    },

    /** Riporta scena e grafici allo stato `snap` (vedi snapshot) e toglie il codice dell'azione. */
    restore(snap) {
      const kb = tab.kb;
      if (snap.indexed) world.showIndexed();
      else world.reset();
      if (snap.search) world.showSearch(snap.search.run, snap.search.answer);
      else world.clearSearch();
      tab.plots.setIndexed(kb, snap.indexed ? kb.inserts.length : 0);
      tab.plots.setProjection(kb, snap.indexed ? kb.inserts.length : 0, snap.search?.run ?? null);
      tab.plots.setSearch(kb, snap.search?.run ?? null);
      tab.hideCode();
      tab.setCaption("Azione interrotta: la scena è tornata com'era prima. Puoi riavviarla dal pannello Azioni.");
    },

    setCaption(text) {
      ui.caption.textContent = text;
    },
  };

  // --- pannello del codice: sui telefoni, durante le azioni, parte ridotto alla sola intestazione
  // (altrimenti copre la scena); se lo studente lo espande, resta espanso fino alla fine dell'azione.
  const phone = window.matchMedia("(max-width: 900px)");
  let codeExpandedByUser = false;
  codePanel.root.querySelector("header button")?.addEventListener("click", () => {
    codeExpandedByUser = !codePanel.root.classList.contains("minimized");
  });
  function showCode(spec) {
    codePanel.show(spec);
    if (phone.matches && player.action && !codeExpandedByUser) codePanel.root.classList.add("minimized");
  }

  // --- costruzione (e ricostruzione quando cambia la configurazione)
  function rebuild() {
    player.stop();
    tab.kb = buildKnowledgeBase({ splitter: settings.splitter, hnsw: settings.hnsw });
    caches.search.clear();
    caches.answer.clear();
    caches.recall = null;
    world.rebuild(tab.kb);
    world.reset();
    updateNavBounds();
    ui.nav.setBounds(navBounds); // anche per far ricalcolare subito i confini alla navigazione
    tab.plots.setIndexed(tab.kb, 0);
    tab.plots.setSearch(tab.kb, null);
    renderMenu();
  }

  // --- sidebar
  const panePath = sidebar.addPane({ id: "percorso", title: "Percorso", hint: HINT });
  const paneActions = sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Animazioni passo per passo: controlli in basso per pausa, passo avanti/indietro e velocità." });

  // --- rappresentazioni del nucleo: "Rete neurale" e "Cervello" (il pannello Cervello va prima di Fonti).
  // Azioni, passi del Percorso e link riportano da soli all'Architettura (ritorno automatico del nucleo).
  const reps = installRepresentations(ui, {
    tabId: "rag",
    app,
    network: {
      // cache: false = load() a ogni apertura del livello (anche quando il tab torna visibile con la rete
      // aperta): la rete si apre sempre con la domanda e la configurazione di adesso (~0,6 ms). Una rete
      // già aperta resta quella della domanda scritta nella sua introduzione: "Cerca" fa partire un'azione
      // e il nucleo torna da solo all'Architettura.
      load: () => import("./network.js").then((m) => m.buildRagNetworkSpec({ kb: tab.kb, ...networkSearch() })),
      cache: false,
      description: "La pipeline come rete: il nostro embedder e la ricerca nel database con i numeri veri della tua domanda, poi l'encoder e l'LLM di un RAG vero (illustrativi). Girando la camera vedi i chunk in profondità.",
    },
    brain: {
      load: () => import("./brain.js").then((m) => m.brainSpec),
      description: "L'ippocampo e la memoria: le analogie tra il RAG e il cervello, con i loro limiti e le fonti, e il circuito dell'ippocampo in 3D.",
    },
  });
  const paneSources = sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Documentazione e sorgenti da cui vengono codice, numeri e parametri." });

  /**
   * La ricerca da disegnare e da dove viene (l'introduzione della rete lo dice): la domanda del pannello
   * Azioni o, se il campo è vuoto, l'ultima ricerca mostrata, altrimenti la domanda d'esempio iniziale.
   */
  function networkSearch() {
    if (settings.question.trim()) return { run: tab.currentSearch(), origin: "panel" };
    if (world.search?.run) return { run: world.search.run, origin: "last" };
    return { run: runSearch(tab.kb, EXAMPLE_QUESTIONS[2].question, { topK: settings.topK, similarityThreshold: settings.threshold }), origin: "example" };
  }

  function menuVars() {
    const kb = tab.kb;
    return {
      chunks: kb.inserts.length,
      dims: kb.embedder.dimensions,
      featuredTitle: kb.featured.document.title,
      flaggedPage: kb.featured.cleaning.warnings[0]?.pageNumber ?? 1,
      splitterLabel: kb.config.splitterPreset.label,
      hnswLabel: kb.config.hnswPreset.label,
    };
  }

  function renderMenu() {
    const current = tab.menu?.current;
    tab.menu?.dispose();
    panePath.replaceChildren();
    const steps = PATH.map((step) => ({
      ...step,
      children: step.children.map((item) => ({ ...item, extra: () => itemExtra(item) })),
    }));
    tab.menu = new Menu(panePath, steps, { vars: menuVars(), onSelect: (item, title) => selectItem(item, title), tabId: "rag" });
    if (current) tab.menu.select(current);
  }

  /** Pulsanti in fondo a un passo del percorso (altro codice, collegamento al Transformer). */
  function itemExtra(item) {
    const buttons = [];
    if (item.extraCode) {
      const key = item.extraCode;
      const label = key.startsWith("lib:") ? LIBRARY[key.slice(4)].title : SNIPPETS[key].file;
      buttons.push(el("button", { type: "button", className: "rag-link", onclick: () => tab.code(key) }, `Mostra il codice: ${label}`));
    }
    if (item.transformerLink) buttons.push(transformerButton());
    return buttons.length ? el("div", { className: "rag-item-extra" }, ...buttons) : null;
  }

  function selectItem(item, title) {
    if (player.action) player.stop();
    if (item.scene === "indexed") {
      world.showIndexed();
      tab.plots.setIndexed(tab.kb, tab.kb.inserts.length);
    }
    if (item.scene === "search") {
      tab.ensureIndexed();
      const run = tab.currentSearch();
      world.showSearch(run, tab.answerFor(run));
      tab.plots.setProjection(tab.kb, tab.kb.inserts.length, run);
      tab.plots.setSearch(tab.kb, run);
    }
    if (item.focus) tab.focus(item.focus);
    if (item.code) tab.code(item.code, item.highlight ?? []);
    else tab.hideCode(); // passi senza codice: tutta la scena resta visibile
    tab.setCaption(item.caption ? item.caption.replace(/\{\{(\w+)\}\}/g, (m, k) => menuVars()[k] ?? m) : title);
  }

  // --- pannello Azioni: domanda, azioni, configurazione, legenda
  const questionInput = el("textarea", {
    className: "rag-question",
    rows: 2,
    value: settings.question,
    ariaLabel: "Domanda per la ricerca",
    oninput: (e) => (settings.question = e.target.value),
  });
  const examples = el(
    "div",
    { className: "rag-examples" },
    ...exampleQuestions().map((q) => el("button", { type: "button", onclick: () => setQuestion(q, true) }, q)),
  );
  const thresholdOut = el("output", {}, fmt(settings.threshold, 2));
  const params = el(
    "div",
    { className: "rag-params" },
    el("label", {}, "topK ", el("input", { type: "number", min: "1", max: "10", value: String(settings.topK), onchange: (e) => (settings.topK = clampInt(e.target.value, 1, 10, 5)) })),
    el(
      "label",
      {},
      "soglia ",
      thresholdOut,
      el("input", {
        type: "range",
        min: "0",
        max: "0.9",
        step: "0.05",
        value: String(settings.threshold),
        oninput: (e) => {
          settings.threshold = Number(e.target.value);
          thresholdOut.textContent = fmt(settings.threshold, 2);
        },
      }),
    ),
  );
  const searchButton = el("button", { type: "button", className: "rag-primary", onclick: () => startSearch() }, "Cerca: avvia l'azione Ricerca");
  const actionsList = el("div");
  const config = el(
    "details",
    { className: "rag-box rag-config" },
    el("summary", {}, "Configurazione della pipeline"),
    configSelect("TokenTextSplitter", SPLITTER_PRESETS, "splitter"),
    configSelect("Indice HNSW", HNSW_PRESETS, "hnsw"),
    el("p", { className: "hint" }, "Cambiando configurazione il database torna vuoto e tutto viene ricalcolato."),
  );
  paneActions.append(
    el(
      "section",
      { className: "rag-box" },
      el("h3", {}, "Domanda dell'utente"),
      questionInput,
      el("p", { className: "hint" }, "Esempi (l'ultimo è fuori tema):"),
      examples,
      params,
      searchButton,
    ),
    actionsList,
    transformerButton(),
    config,
    legend(),
  );

  function configSelect(title, presets, key) {
    const note = el("p", { className: "hint" }, presets[settings[key]].note);
    const select = el(
      "select",
      {
        onchange: (e) => {
          settings[key] = e.target.value;
          note.textContent = presets[settings[key]].note;
          // il database torna vuoto: si guarda nell'Architettura (una rete aperta sarebbe quella di prima)
          void reps.show("main");
          rebuild();
          tab.setCaption("Nuova configurazione: database vuoto. Avvia l'Indicizzazione.");
        },
      },
      ...Object.entries(presets).map(([id, p]) => el("option", { value: id, selected: id === settings[key] }, p.label)),
    );
    return el("label", { className: "rag-select" }, el("span", {}, title), select, note);
  }

  function transformerButton() {
    return el("button", { type: "button", className: "rag-link rag-transformer", onclick: () => openTransformerContext() }, "Guarda come il contesto entra nell'LLM");
  }

  function setQuestion(q, start = false) {
    settings.question = q;
    questionInput.value = q;
    if (start) startSearch();
  }

  function startSearch() {
    if (!settings.question.trim()) {
      tab.setCaption("Scrivi una domanda prima di cercare.");
      return;
    }
    sidebar.select("azioni");
    player.start("ricerca");
  }

  paneSources.append(renderSources());

  // --- azioni
  player.register(indexAction(tab));
  player.register(searchAction(tab));
  player.renderOverview(actionsList, { categories: ["Indicizzazione", "Ricerca"] });

  // --- tra i tab
  app.bus.addEventListener("rag:query", (e) => {
    const question = e.detail?.question;
    if (typeof question !== "string" || !question.trim()) return;
    app.open("rag");
    setQuestion(question.trim(), true);
  });

  /** Apre la vista Contesto del tab Transformer con la domanda scritta qui. */
  async function openTransformerContext() {
    await app.open("transformer"); // si risolve quando il tab (caricato alla prima apertura) è pronto
    const question = settings.question.trim();
    app.bus.dispatchEvent(new CustomEvent("transformer:view", { detail: question ? { view: "contesto", question } : "contesto" }));
  }

  // grafici "HNSW al crescere dei vettori": si misurano la prima volta che il pannello si apre
  ui.charts.onResize(() => {
    if (ui.charts.isOpen) tab.plots.startScaling(tab.kb, tab.corpusRecall());
  });

  // --- tasti: ↑ ↓ scorrono i passi del Percorso, Spazio mette in pausa l'azione (core/stepKeys.js);
  // il menu si ricostruisce quando cambia la configurazione, quindi lo si legge ogni volta
  installStepKeys({ root, ui, menu: () => tab.menu, pathPane: panePath });

  rebuild();
  // Aperto da un link diretto a un passo: si parte da quel passo (goto lo mostrerà subito dopo),
  // così il primo passo del Percorso non viene segnato come visto senza che lo studente l'abbia letto.
  const initial = parseHash(location.hash);
  const firstStep = initial.tab === "rag" && initial.go?.kind === "step" && tab.menu.items.has(initial.go.id) ? initial.go.id : "intro.problema";
  tab.menu.select(firstStep, { scroll: false });
  tab.focus("overview", { duration: 0.01 });

  // #rag/q=<domanda>: avvio diretto di una ricerca (alternativa all'evento "rag:query"), sia
  // all'apertura del tab sia quando l'indirizzo cambia a pagina aperta. Lo stesso indirizzo non
  // riparte due volte (es. tornando con "indietro" da un altro tab).
  let handledQuery = null;
  function routeQuery() {
    const match = /^#rag\/q=(.+)$/.exec(location.hash);
    if (!match || match[1] === handledQuery) return;
    handledQuery = match[1];
    let question;
    try {
      question = decodeURIComponent(match[1]).trim();
    } catch {
      return; // indirizzo malformato (es. "%" da solo): lo ignoriamo
    }
    if (question) setQuestion(question, true);
  }
  routeQuery();
  window.addEventListener("hashchange", routeQuery);

  window.__ragTab = tab; // per il debug dalla console e per gli screenshot automatici

  /** Link diretto o risultato della ricerca globale: porta il tab su un passo, un'azione, un livello o una rappresentazione. */
  async function goto({ kind, id }) {
    // view:rete | view:cervello aprono la rappresentazione; per tutto il resto il nucleo chiude prima il
    // livello alternativo (null) e si prosegue nella scena del tab
    const shown = reps.goto({ kind, id });
    if (shown) return shown;
    if (kind === "step") return tab.menu.reveal(id, { sidebar }); // apre il Percorso sul passo
    // Azioni e livelli si guardano nella scena: sul telefono il cassetto della sidebar la coprirebbe.
    const showScene = () => phone.matches && sidebar.toggleCollapsed(true);
    if (kind === "action") {
      if (!ACTION_IDS.has(id)) return false;
      sidebar.select("azioni");
      showScene();
      if (id === "ricerca") startSearch();
      else player.start(id);
      return true;
    }
    if (kind === "view") {
      if (!Object.hasOwn(LEVEL_CAPTIONS, id)) return false;
      // come scegliere un passo dal Percorso: l'azione in corso si chiude, altrimenti il suo passo
      // successivo riporterebbe subito la camera altrove
      if (player.action) player.stop();
      // zoom automatico: il volo porta alla distanza del livello e lo studente può ancora zoomare
      if (lod.mode !== "auto") lod.setMode("auto");
      showScene();
      tab.setCaption(LEVEL_CAPTIONS[id]);
      await flyToLevel(id);
      return true;
    }
    return false;
  }

  return {
    tab,
    goto,
    activate() {
      // con la rete aperta il nucleo la ricrea qui (host.start) con una specifica nuova (cache: false)
      host.start();
      host.resize();
    },
    deactivate() {
      player.stop();
      host.stop();
    },
  };
}

function clampInt(value, min, max, fallback) {
  const n = Math.round(Number(value));
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

/** Legenda dei colori della scena. */
function legend() {
  // stesso ordine di pipeline.js: prima il documento in evidenza, poi gli altri
  const docs = [...CORPUS.filter((d) => d.id === FEATURED_ID), ...CORPUS.filter((d) => d.id !== FEATURED_ID)].map((d) => d.title);
  const sw = (color, text) => el("li", {}, el("span", { className: "rag-swatch", style: { background: color } }), text);
  return el(
    "details",
    { className: "rag-box rag-legend" },
    el("summary", {}, "Legenda dei colori"),
    el("h4", {}, "Documenti (ordine di caricamento)"),
    el("ul", {}, ...docs.map((d, i) => sw(DOC_COLORS[i], d))),
    el("h4", {}, "Ricerca nell'indice HNSW"),
    el(
      "ul",
      {},
      sw("#fde047", "domanda (rombo) e percorso"),
      sw("#ffffff", "nodo corrente"),
      sw("#facc15", "candidati da esplorare (C)"),
      sw("#4ade80", "migliori trovati (W) e risultati"),
      sw("#64748b", "nodi visitati"),
      sw("#f472b6", "punto di ingresso (anello)"),
    ),
  );
}

/** Pannello "Fonti": elenco ragionato con link. */
function renderSources() {
  return el(
    "div",
    { className: "rag-sources" },
    ...SOURCES.map((group) =>
      el(
        "section",
        {},
        el("h3", {}, group.title),
        el(
          "ul",
          {},
          ...group.items.map((s) =>
            el("li", {}, s.url ? el("a", { href: s.url, target: "_blank", rel: "noopener" }, s.title) : el("strong", {}, s.title), el("p", { className: "hint" }, s.note)),
          ),
        ),
      ),
    ),
    el("p", { className: "hint" }, "Le simulazioni (embedding, risposta dell'LLM) sono dichiarate come tali nella scena e nel percorso."),
  );
}
