/**
 * Tab "LLM moderni": cosa hanno in più i modelli di oggi rispetto al nostro GPT didattico.
 *
 * Come è fatto (un modulo per cosa):
 *   regions.js        mappa della scena: 16 regioni su una griglia 4 × 4 (una per meccanismo)
 *   region*.js        livello Dettaglio di ogni regione, con una piccola API usata dalle azioni
 *   overview.js       livelli Blocchi (un rettangolo per meccanismo) e Panoramica (pipeline)
 *   actions*.js       azioni animate per categoria (Architettura, Addestramento, Ragionamento
 *                     e agenti, Oltre il testo)
 *   plots.js          grafici; sim.js matematica pura (formule pubblicate e simulazioni dichiarate)
 *   content.js        percorso didattico, schede dei modelli, fonti; snippets.js codice vero
 *   ourModel.js       numeri del nostro GPT (data/model.json) per i testi con segnaposto {{chiave}}
 */
import * as THREE from "three";
import { el, fillTemplate, fmt, sleep } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { LODController } from "../../core/lod.js";
import { Menu } from "../../core/ui/menu.js";
import { architectureActions } from "./actionsArch.js";
import { beyondActions } from "./actionsBeyond.js";
import { inferenceActions } from "./actionsInference.js";
import { reasoningActions } from "./actionsReasoning.js";
import { trainingActions } from "./actionsTraining.js";
import { MODEL_CARDS, PATH, SOURCES } from "./content.js";
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
import { gridBox, REGION_BY_ID, REGION_H, REGION_W, REGIONS, ROWS, regionBox, regionCenter, regionShell } from "./regions.js";
import { Snippets } from "./snippets.js";

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
 * Schermo stretto: stessa soglia di core/ui/charts.js. Sotto i 900 px i pannelli non stanno accanto
 * alla scena ma sopra (grafici a tutto schermo, codice in basso).
 */
const NARROW = window.matchMedia("(max-width: 899px)");

export async function createTab(root, app) {
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint: "Segui il percorso o avvia un'azione. Rotellina: da vicino i componenti, più lontano un rettangolo per meccanismo, da lontanissimo la pipeline completa. Clic su una regione per avvicinarti.",
  });
  const { host, director, sidebar, codePanel, stage } = ui;
  // numeri del nostro GPT per i testi (segnaposto {{chiave}}), dal model.json esportato
  const model = await loadModel();
  const vars = modelVars(model);

  // --- scena: Dettaglio (16 regioni), Blocchi, Panoramica
  const detail = new THREE.Group();
  const regions = {};
  for (const region of REGIONS) {
    const g = new THREE.Group();
    g.position.copy(regionCenter(region));
    g.userData.regionId = region.id;
    regionShell(g, region);
    regions[region.id] = BUILDERS[region.id](g, vars);
    detail.add(g);
  }
  const blocks = buildBlocks();
  const macro = buildMacro();
  blocks.children.forEach((child, i) => (child.userData.regionId = REGIONS[i]?.id));
  host.scene.add(detail, blocks, macro);

  let lod = null;
  lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: LOD_DETAIL_MAX, group: detail },
      { id: "blocks", label: "Blocchi", maxDistance: LOD_BLOCKS_MAX, group: blocks },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: macro },
    ],
    widgetParent: stage,
    onChange: (id) => {
      if (lod && lod.mode !== "auto" && lod.mode === id) flyToLevel(id);
    },
  });

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

  /** Scelta manuale di un livello: la camera si porta a una distanza adatta. */
  function flyToLevel(id) {
    if (id === "detail") {
      const near = nearestRegion(host.controls.target);
      return tab.focusRegion(near.id);
    }
    if (id === "blocks") return director.focus(gridBox(), { showFrame: false, margin: 1.02, direction: FRONT });
    return director.focus(macroBox(), { showFrame: false, margin: 1.02, direction: FRONT });
  }

  function nearestRegion(point) {
    let best = REGIONS[0];
    let bestD = Infinity;
    for (const r of REGIONS) {
      const d = regionCenter(r).distanceToSquared(point);
      if (d < bestD) [best, bestD] = [r, d];
    }
    return best;
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
      return tab.focusRegion(id, { bottomPad: NARROW.matches ? PLAYER_PAD_NARROW : PLAYER_PAD });
    },
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

  // percorso: ogni sottopunto con un'azione ha il pulsante per avviarla
  const steps = PATH.map((step) => ({
    ...step,
    children: step.children.map((item) => ({
      ...item,
      text: item.text && fillTemplate(item.text, vars),
      extra: item.action ? () => el("button", { type: "button", className: "mllm-run", onclick: (e) => (e.stopPropagation(), ui.player.start(item.action)) }, "▶ Avvia l'animazione") : undefined,
    })),
  }));
  let initial = true;
  tab.menu = new Menu(panePath, steps, {
    onSelect: (item) => {
      if (initial) return; // all'avvio si mostra la vista d'insieme, non la prima regione
      if (item.code) tab.showCode(item.code);
      else tab.hideCode();
      if (item.focus) tab.focusRegion(item.focus);
      if (item.charts) tab.openCharts(item.charts);
      tab.setCaption("");
    },
  });

  paneViews.append(renderViews(tab, director));
  paneModels.append(renderModels(model));
  paneSources.append(renderSources());

  // --- moduli
  installPlots(tab);
  for (const make of [architectureActions, inferenceActions, trainingActions, reasoningActions, beyondActions]) for (const action of make(tab)) ui.player.register(action);
  const actionsList = el("div", { className: "mllm-actions" });
  paneActions.append(el("p", { className: "mllm-note" }, "Ogni azione inquadra la sua regione e sincronizza codice e grafici. Le animazioni con dati inventati lo dichiarano nella didascalia."), actionsList);
  ui.player.renderOverview(actionsList, { categories: CATEGORIES });
  ui.player.on("end", () => director.hideFrame());

  // --- clic su una regione (senza trascinare): avvicinati
  let down = null;
  host.renderer.domElement.addEventListener("pointerdown", (e) => (down = { x: e.clientX, y: e.clientY }));
  host.renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
    const level = lod.current === "detail" ? detail : lod.current === "blocks" ? blocks : null;
    if (!level) return;
    for (const hit of host.pick(e, [level])) {
      let o = hit.object;
      while (o && !o.userData.regionId) o = o.parent;
      if (o) {
        tab.focusRegion(o.userData.regionId);
        return;
      }
    }
  });

  // --- tasti
  root.tabIndex = -1;
  root.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowDown") tab.menu.move(+1);
    if (e.key === "ArrowUp") tab.menu.move(-1);
    if (e.key === " " && ui.player.action) {
      e.preventDefault();
      ui.player.toggle();
    }
  });

  tab.menu.select("idea.ponte");
  initial = false;
  tab.hideCode();
  director.focus(gridBox(), { showFrame: false, margin: 1.02, direction: FRONT, duration: 0.01 });
  window.__modernLlmTab = tab; // per il debug dalla console e per gli screenshot automatici

  return {
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

/** Pannello "Viste": livelli di zoom e un pulsante per meccanismo, raggruppati per riga. */
function renderViews(tab, director) {
  const views = [
    { title: "Panoramica: dalla raccolta dei dati alla risposta", text: "Dati → Pre-training → Post-training → Modello → Inferenza.", run: () => (tab.lod.setMode("macro"), director.focus(macroBox(), { showFrame: false, margin: 1.02, direction: FRONT })) },
    { title: "Tutti i meccanismi (livello Blocchi)", text: "Un rettangolo per meccanismo con cosa fa.", run: () => (tab.lod.setMode("blocks"), director.focus(gridBox(), { showFrame: false, margin: 1.02, direction: FRONT })) },
  ];
  const button = (v) =>
    el(
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
    );
  return el(
    "div",
    { className: "view-switch mllm-views" },
    ...views.map(button),
    ...ROWS.flatMap((row, r) => [
      el("h3", { className: "mllm-h3" }, row.title),
      ...REGIONS.filter((g) => g.row === r).map((g) => button({ title: g.title, text: g.sub, run: () => (tab.lod.setMode("auto"), tab.focusRegion(g.id)) })),
    ]),
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
