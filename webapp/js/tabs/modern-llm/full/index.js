/**
 * Sottotab "Architettura LLM moderni completa" del tab "LLM moderni": un modello aperto vero
 * (DeepSeek-V4.1-Flash, con Kimi K3.1 Pro e gpt-oss-120b come alternative) smontato in tre livelli
 * di zoom, guidato dai moduli di soli dati in full/models/.
 *
 * Vive nella STESSA scena del tab (un solo SceneHost, nessun contesto WebGL in più): tre gruppi con
 * il loro LODController (Panoramica / Blocchi / Dettaglio), i loro pannelli nella sidebar (nascosti
 * nell'altro sottotab), le loro azioni nel player, i loro grafici. index.js del tab lo crea al primo
 * ingresso (import dinamico) e lo accende/spegne con activate/deactivate.
 *
 * Moduli: models/ (dati) → model.js (logica pura) → layout.js (posizioni pure) → scene*.js
 * (three.js per livello) · panes.js (DOM) · actions.js · plots.js · content.js / catalog.js (testi).
 *
 *   const full = createFullArchitecture({ ui, tab, regions, onMarker, onModelChange });
 *   full.activate(); full.deactivate(); await full.goto({kind, id}); full.home(); full.bounds();
 *   full.focusMechanism(regionId); full.setModel(id); full.model; full.dispose();
 */
import * as THREE from "three";
import { el, prefersReducedMotion, sleep } from "../../../core/dom.js";
import { LODController, setGroupOpacity } from "../../../core/lod.js";
import { Menu } from "../../../core/ui/menu.js";
import { installSceneClick } from "../sceneClick.js";
import { RESET_TITLE } from "../subtabs.js";
import { fullActions } from "./actions.js";
import { FULL_ACTION_IDS, FULL_CATEGORIES, FULL_LEVELS, FULL_SECTIONS, MARKER_LABELS } from "./catalog.js";
import { FULL_PATH } from "./content.js";
import { clearGroup, disposeTree, markerSprite, rectBox } from "./draw.js";
import { macroLayout, stripLayout } from "./layout.js";
import { anatomyCards, anchorsOf, blockOfLayer, buildStrip, cardState, layerByNumber, layerForAnchor, layerList, paramTotals } from "./model.js";
import { MODEL_BY_ID, MODELS, PRIMARY_MODEL_ID } from "./models/index.js";
import { renderModels, renderSources, renderViews, renderWeights } from "./panes.js";
import { FULL_CHART_SECTIONS, installFullPlots } from "./plots.js";
import { buildBlocks } from "./sceneBlocks.js";
import { anatomySignature, buildAnatomy, buildContext } from "./sceneDetail.js";
import { buildMacro } from "./sceneMacro.js";
import { FULL_SNIPPETS } from "./snippets.js";

/** La scena si guarda quasi di fronte, come la vista d'insieme. */
const FRONT = new THREE.Vector3(0, 0.05, 1).normalize();
/** Larghezza della striscia compressa di V4.1 (circa): oltre, la Panoramica si ingrandisce con lei. */
const REF_STRIP_WIDTH = 430;
/** Misura dell'anatomia per le soglie del LOD (unità di scena). */
const ANATOMY_SIZE = { w: 200, h: 108 };
/** Camera della vista completa: la striscia srotolata e la Panoramica stanno oltre i 2.000 del nucleo. */
const NEAR = 0.5;
const FAR = 40000;
/** Segnaposto: altezza in unità di scena per livello (si leggono alla distanza tipica del livello). */
const MARKER_H = { macro: 16, blocks: 7, detail: 2.7 };
/** Lo zoom apre il Dettaglio del layer sotto il punto guardato solo subito dopo un gesto dello studente. */
const USER_INPUT_WINDOW_MS = 1500;
const NARROW = window.matchMedia("(max-width: 899px)");
/** Pannello dei grafici (css/charts.css) e scena che deve restare accanto perché si aprano da soli. */
const CHARTS_WIDTH = 460;
const MIN_SCENE_BESIDE_CHARTS = 700;
const CHARTS_HINT_MS = 2600;
/** Scena libera sotto la quale il codice di un passo parte ridotto (solo intestazione). */
const MIN_SCENE_BESIDE_CODE = 600;
/** Telefono: stage più stretto di così (larghezza/altezza visibili) e larghezza mostrata alla soglia del Dettaglio. */
const PHONE_ASPECT = 0.85;
const PHONE_DETAIL_SPAN = 140;
const PHONE_HINT = "Pizzica con due dita per leggere i blocchi; tocca un layer per aprirne il Dettaglio.";
/** Pane della sidebar della vista completa (id univoci nel tab). */
export const FULL_PANES = [
  { id: "full-percorso", title: "Percorso" },
  { id: "full-azioni", title: "Azioni", hint: "Animazioni passo per passo sul modello scelto. Le scelte degli esperti sono simulate (non sono pubbliche): le didascalie lo dicono." },
  { id: "full-viste", title: "Viste", hint: "Livelli dello zoom, ripetizioni «× N», il layer del Dettaglio, i pezzi dell'architettura e i 16 meccanismi." },
  { id: "full-modelli", title: "Modelli", hint: "I tre modelli aperti della vista e il panorama degli open-weight, con la marcatura di verifica." },
  { id: "full-pesi", title: "Pesi", hint: "Dove sono i parametri, in quali precisioni e con quali nomi di tensore." },
  { id: "full-fonti", title: "Fonti", hint: "Come sono stati ottenuti i numeri, fonti e ciò che resta da verificare." },
];

/**
 * @param {object} o
 * @param {object} o.ui layout del tab (createTabLayout)
 * @param {object} o.tab oggetto del tab: showCode(snippet), hideCode(), setCaption(text), openCharts(id), codePending
 * @param {object[]} o.regions le 16 regioni della vista d'insieme (regionData.js)
 * @param {(regionId: string) => void} o.onMarker clic su un segnaposto: passa alla vista d'insieme
 * @param {(model: object) => void} [o.onModelChange]
 */
export function createFullArchitecture({ ui, tab, regions, onMarker, onModelChange = () => {} }) {
  const { host, director, sidebar, stage, nav } = ui;
  const camera = host.camera;
  const savedCamera = { near: camera.near, far: camera.far };
  const roots = { macro: new THREE.Group(), blocks: new THREE.Group(), detail: new THREE.Group() };
  for (const g of Object.values(roots)) {
    g.visible = false;
    host.scene.add(g);
  }

  // --- stato
  let active = false;
  let model = MODEL_BY_ID.get(PRIMARY_MODEL_ID);
  let expanded = new Set();
  let selectedLayer = defaultLayer(model);
  let strip, stripL, macroL, totals, macroScene, blocksScene, contextScene;
  let anatomy = null;
  let anatomySig = null;
  const markers = { macro: [], blocks: [], detail: [] };

  const lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: 200, group: roots.detail },
      { id: "blocks", label: "Blocchi", maxDistance: 600, group: roots.blocks },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: roots.macro },
    ],
    widgetParent: stage,
    // (active per primo: il costruttore chiama onChange quando `lod` non esiste ancora)
    onChange: (id) => active && id === "detail" && lod.mode === "auto" && followTargetLayer(),
    // come nella vista d'insieme: il livello scelto dallo studente resta bloccato ("Auto" lo sblocca)
    onPick: (mode) => mode !== "auto" && flyToLevel(mode, { lock: true }),
  });
  lod.setEnabled(false);
  for (const g of Object.values(roots)) g.visible = false;

  // --- sidebar: pannelli propri, nascosti finché il sottotab non è attivo
  const panes = Object.fromEntries(FULL_PANES.map((p) => [p.id, sidebar.addPane(p)]));
  for (const p of FULL_PANES) sidebar.setPaneHidden(p.id, true);

  // Percorso
  let silentSelect = false;
  const steps = FULL_PATH.map((section) => ({
    ...section,
    children: section.children.map((item) => ({
      ...item,
      extra: item.play ? () => el("button", { type: "button", className: "mllm-run", onclick: (e) => (e.stopPropagation(), ui.player.start(item.play)) }, "▶ Avvia l'animazione") : undefined,
    })),
  }));
  const menu = new Menu(panes["full-percorso"], steps, { tabId: "modern-llm", onSelect: (item) => !silentSelect && selectStep(item) });
  if (menu.resetButton) menu.resetButton.title = RESET_TITLE;

  // Azioni
  const fx = createActionContext();
  for (const action of fullActions(fx)) ui.player.register(action);
  const actionsList = el("div", { className: "mllm-actions" });
  panes["full-azioni"].append(el("p", { className: "mllm-note" }, "Ogni azione inquadra il suo pezzo dell'architettura e sincronizza codice e grafici. I dati sono quelli del modello scelto; dove una scelta non è pubblica la simulazione è dichiarata."), actionsList);
  ui.player.renderOverview(actionsList, { categories: FULL_CATEGORIES });

  // Viste, Modelli, Pesi, Fonti
  const views = renderViews({
    regions,
    onLevel: (level) => flyToLevel(level, { lock: true }),
    onSection: (s) => focusSection(s),
    onLayer: (n) => selectLayer(n, { fly: true }),
    onToggleExpand: () => toggleExpandAll(),
    onMechanism: (id) => focusMechanism(id),
    onOverview: (id) => onMarker(id),
  });
  panes["full-viste"].append(views.root);
  const modelsPane = renderModels({ onModel: (id) => setModel(id) });
  panes["full-modelli"].append(modelsPane.root);
  const weightsPane = renderWeights();
  panes["full-pesi"].append(weightsPane.root);
  const sourcesPane = renderSources();
  panes["full-fonti"].append(sourcesPane.root);

  // barra strumenti dello stage (solo in questo sottotab): il selettore del modello
  const modelSelect = el(
    "select",
    { className: "mllm-select", ariaLabel: "Modello mostrato nell'architettura completa", onchange: (e) => setModel(e.target.value) },
    ...MODELS.map((m) => el("option", { value: m.id }, m.name)),
  );
  const toolbar = el("div", { className: "stage-toolbar mllm-toolbar" }, el("label", { className: "mllm-toolbar-label" }, el("span", {}, "Modello"), modelSelect));

  // grafici
  fx.plots = installFullPlots(ui.charts);

  // suggerimento al passaggio sui segnaposto
  const tip = el("div", { className: "mllm-tip", role: "tooltip", hidden: true });
  stage.append(tip);

  rebuild();

  // ultimo gesto di navigazione dello studente (rotellina, trascinamento, tasti, pad)
  let lastUserInput = -Infinity;
  const offInput = nav.on("input", (source) => {
    if (source !== "home") lastUserInput = performance.now();
    // il suggerimento del telefono sparisce al primo gesto: ha già fatto il suo lavoro
    if (active && ui.caption.textContent === PHONE_HINT) tab.setCaption("");
  });

  // --- interazione con la scena
  const canvas = host.renderer.domElement;
  const offClick = installSceneClick(canvas, (e) => active && onSceneClick(e));
  let hoverFrame = 0;
  let lastHover = null;
  const onMove = (e) => {
    if (!active || e.pointerType === "touch" || hoverFrame) return;
    lastHover = e;
    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      // il clic su un segnaposto può aver appena spento il sottotab (si passa alla vista d'insieme):
      // il raggio non guarda la visibilità, quindi il suggerimento ricomparirebbe sopra l'altra scena
      if (active) updateHover(lastHover);
    });
  };
  const onLeave = () => hideTip();
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  const offResize = host.onResize(() => active && updateThresholds());
  ui.charts.onResize(() => {
    if (!active) return;
    syncCodeWithCharts();
    updateThresholds();
  });

  // =============================================================================================

  /** Il layer predefinito del Dettaglio: il primo del tipo con più pezzi al lavoro (V4.1: il 2). */
  function defaultLayer(m) {
    const cards = anatomyCards(m);
    let best = layerList(m)[0];
    let bestScore = -1;
    const seen = new Set();
    for (const l of layerList(m)) {
      const sig = `${l.kind}|${[...l.flags].join()}`;
      if (seen.has(sig)) continue;
      seen.add(sig);
      const score = cards.filter((c) => cardState(c, l) === "on").length - l.flags.size * 0.5;
      if (score > bestScore) [best, bestScore] = [l, score];
    }
    return best.number;
  }

  /** Ricostruisce le tre scene del modello scelto (al cambio di modello o di ripetizioni srotolate). */
  function rebuild() {
    for (const g of Object.values(roots)) clearGroup(g);
    anatomy = null;
    anatomySig = null;
    markers.macro = [];
    markers.blocks = [];
    markers.detail = [];
    totals = paramTotals(model);
    strip = buildStrip(model, { expanded });
    stripL = stripLayout(strip, model);
    macroL = macroLayout(model, Math.max(1, stripL.width / REF_STRIP_WIDTH));
    macroScene = buildMacro(roots.macro, model, macroL, totals);
    blocksScene = buildBlocks(roots.blocks, model, strip, stripL);
    contextScene = buildContext(roots.detail, model, strip, stripL);
    placeMarkers("macro", macroScene.anchors);
    placeMarkers("blocks", blocksScene.anchors);
    ensureAnatomy();
    updateThresholds();
    for (const level of lod.levels) setGroupOpacity(level.group, lod.alphas.get(level.id));
    if (!active) for (const g of Object.values(roots)) g.visible = false;
    views.update({ model, expanded: expanded.size > 0, layer: selectedLayer });
    modelsPane.update({ model });
    weightsPane.update({ model });
    sourcesPane.update({ model });
    fx.plots?.setModel(model);
    modelSelect.value = model.id;
  }

  /** L'anatomia del layer scelto, nel punto della striscia dove sta quel layer. */
  function ensureAnatomy() {
    const layer = layerByNumber(model, selectedLayer) ?? layerList(model)[0];
    const sig = anatomySignature(layer);
    const segment = model.layers.segments.find((s) => layer.number >= s.from && layer.number <= s.to && model.layers.segments.length > 1);
    if (sig !== anatomySig || !anatomy) {
      if (anatomy) disposeTree(anatomy.group);
      anatomy = buildAnatomy(model, layer, segment ? capitalize(segment.label) : "");
      roots.detail.add(anatomy.group);
      anatomySig = sig;
      markers.detail = [];
      placeMarkers("detail", anatomy.anchors, anatomy.group);
      setGroupOpacity(roots.detail, lod.alphas.get("detail"));
      if (!active) roots.detail.visible = false;
      if (strip) updateThresholds(); // la soglia del Dettaglio dipende dalla larghezza dell'anatomia
    } else {
      anatomy.setTitle(`Layer ${layer.number}${segment ? ` · ${capitalize(segment.label)}` : ""} · ${model.kinds[layer.kind].name}`);
    }
    const bi = blockOfLayer(strip, layer.number);
    anatomy.group.position.set(stripL.blocks[Math.max(0, bi)].x, 0, 1.4);
    anatomy.group.updateMatrixWorld(true);
  }

  /**
   * Segnaposto dei 16 meccanismi per un livello: sopra l'ancora (a destra delle cache, sotto i rami bassi),
   * impilati se più meccanismi vivono nello stesso punto. `parent`: gruppo che li porta (anatomia).
   */
  function placeMarkers(level, anchors, parent = roots[level]) {
    const byAnchor = new Map();
    regions.forEach((r, i) => {
      const m = model.mechanisms[r.id];
      if (!m || !m.at.startsWith(`${level}:`)) return;
      const box = anchors.get(m.at);
      if (!box) return;
      const list = byAnchor.get(m.at) ?? [];
      list.push({ region: r, index: i, mech: m, box });
      byAnchor.set(m.at, list);
    });
    const h = MARKER_H[level] * (level === "macro" ? macroL.stages[0].w / 112 : 1);
    for (const [anchor, list] of byAnchor) {
      // più meccanismi nello stesso punto: in fila da sinistra a destra (in colonna accanto alle cache)
      let dx = 0;
      list.forEach(({ region, index, mech, box }, j) => {
        const sprite = markerSprite(index + 1, MARKER_LABELS[region.id] ?? region.title, region.color, mech.status, h);
        const rect = box.isBox3 ? { x0: box.min.x, x1: box.max.x, y1: box.max.y, y0: box.min.y } : { x0: box.x - box.w / 2, x1: box.x + box.w / 2, y1: box.y + box.h / 2, y0: box.y - box.h / 2 };
        const side = anchor.startsWith("blocks:cache-");
        const below = model.branches.some((b) => b.side === "bottom" && anchor === `blocks:${b.id}`);
        if (anchor === "detail:experts") sprite.position.set(rect.x0 - sprite.scale.x - 1, (rect.y0 + rect.y1) / 2 - j * h * 1.15, 2); // sopra c'è il router
        else if (side) sprite.position.set(rect.x1 + 1.5, (rect.y0 + rect.y1) / 2 - j * h * 1.15, 2);
        else if (below) sprite.position.set(rect.x0 + 0.5 + dx, rect.y0 - h * 0.75, 2); // sopra arriva la freccia dal layer finale
        else sprite.position.set(rect.x0 + (level === "detail" ? 0.2 : 0.5) + dx, rect.y1 + h * 0.75, 2);
        dx += sprite.scale.x + h * 0.25;
        sprite.userData.mechanism = region.id;
        parent.add(sprite);
        markers[level].push(sprite);
      });
    }
    // segnaposto di ancore vicine sulla stessa riga: chi si sovrappone al precedente scivola a destra
    const row = (sp) => Math.round(sp.position.y / h);
    const sorted = [...markers[level]].sort((a, b) => row(a) - row(b) || a.position.x - b.position.x);
    for (let i = 1; i < sorted.length; i++) {
      const prev = sorted[i - 1];
      const cur = sorted[i];
      if (row(prev) !== row(cur)) continue;
      const minX = prev.position.x + prev.scale.x + h * 0.25;
      if (cur.position.x < minX) cur.position.x = minX;
    }
  }

  /** Soglie del LOD dalla misura dei tre livelli e dallo stage visibile (pannelli aperti compresi). */
  function updateThresholds() {
    const t = thresholds();
    lod.levels[0].maxDistance = t.detail;
    lod.levels[1].maxDistance = t.blocks;
  }

  function fitDistance(w, h) {
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    return Math.max(h / 2 / tanHalf, w / 2 / (tanHalf * host.visibleAspect()));
  }

  function thresholds() {
    const s = stripL.box;
    const m = macroL.box;
    // l'anatomia vera: con il ramo Engram (layer 1 e 14) è più larga, e il suo volo finirebbe oltre la
    // soglia (meno l'isteresi dell'8% di core/lod.js), restando nei Blocchi
    const aw = anatomy ? Math.max(ANATOMY_SIZE.w, anatomy.box.maxX - anatomy.box.minX) : ANATOMY_SIZE.w;
    let detail = 1.3 * fitDistance(aw, ANATOMY_SIZE.h);
    // Telefono in verticale: l'anatomia (200 unità) entra intera già a una distanza a cui i titoli dei
    // Blocchi sono di 8 px, e pizzicando si passava al Dettaglio prima di poterli leggere. Qui la soglia
    // è la distanza che mostra circa 4 blocchi; i voli verso l'anatomia bloccano il Dettaglio (focusAnchor).
    if (phoneStage()) detail = Math.min(detail, fitDistance(PHONE_DETAIL_SPAN, 0));
    const blocks = Math.max(detail * 1.5, Math.sqrt(1.05 * fitDistance(s.maxX - s.minX, s.maxY - s.minY) * 1.05 * fitDistance(m.maxX - m.minX, m.maxY - m.minY)));
    return { detail, blocks };
  }

  /** Stage stretto e verticale (telefono): la scena è molto più alta che larga. */
  function phoneStage() {
    return NARROW.matches && host.visibleAspect() < PHONE_ASPECT;
  }

  /** Box3 di un'ancora (livello:pezzo) in coordinate di scena, o null se il modello non ce l'ha. */
  function anchorBox(anchor) {
    const [level] = anchor.split(":");
    if (level === "macro") return macroScene.anchors.get(anchor)?.clone() ?? null;
    if (level === "blocks") return blocksScene.anchors.get(anchor)?.clone() ?? null;
    return anatomy?.anchorBox(anchor) ?? null;
  }

  /**
   * Inquadra un'ancora al suo livello: la distanza minima tiene lo zoom automatico sul livello
   * giusto (un blocco piccolo inquadrato da vicino finirebbe nel Dettaglio). `pad`: spazio in basso
   * per la barra del player; `wide`: un pezzo dell'anatomia con il suo intorno; `with`: blocchi in più.
   * Se il livello dell'ancora è quello bloccato nel selettore (dallo studente o da un'azione sulla
   * Panoramica) resta bloccato e la camera può avvicinarsi quanto serve per leggere: senza blocco la
   * Panoramica da vicino lascerebbe il posto ai Blocchi. `keepMode`: blocca comunque (scelta di un livello).
   */
  async function focusAnchor(anchor, { layer = null, pad = false, wide = false, with: extra = [], also = [], duration = 1.1, margin = 1.05, keepMode = false } = {}) {
    const [level] = anchor.split(":");
    let keep = keepMode || lod.mode === level;
    // sul telefono la soglia del Dettaglio è più vicina dell'anatomia intera (thresholds): il volo blocca
    // il livello, altrimenti atterrerebbe nei Blocchi (il selettore mostra «Dettaglio», «Auto» lo sblocca)
    if (!keep && level === "detail" && phoneStage()) {
      lod.setMode("detail");
      keep = true;
    }
    // un pezzo della Panoramica (una fase, i pesi, le capacità) si legge solo da vicino, dove lo zoom
    // automatico mostrerebbe i Blocchi: la Panoramica resta bloccata (vale per passi, viste, segnaposto)
    if (!keep && level === "macro" && anchor !== "macro:all") {
      lod.setMode("macro");
      keep = true;
    }
    if (level === "detail") selectLayer(layer ?? layerForAnchor(model, anchor, selectedLayer), { fly: false });
    const box = anchorBox(anchor);
    if (!box) return false;
    for (const i of extra) if (i >= 0 && blocksScene.blockBoxes[i]) box.union(blocksScene.blockBoxes[i]);
    for (const a of also) {
      const other = anchorBox(a);
      if (other) box.union(other);
    }
    // un pezzo piccolo del Dettaglio si guarda con il suo intorno (mai una scheda a tutto schermo)
    const minHalf = wide ? { x: 56, y: 34 } : level === "detail" && anchor !== "detail:layer" ? { x: 48, y: 30 } : null;
    if (minHalf) {
      const c = box.getCenter(new THREE.Vector3());
      box.expandByPoint(new THREE.Vector3(c.x - minHalf.x, c.y - minHalf.y, 0)).expandByPoint(new THREE.Vector3(c.x + minHalf.x, c.y + minHalf.y, 0));
    }
    // una fase della Panoramica vista da vicino: si vedono anche le frecce verso le fasi accanto
    if (keep && level === "macro" && model.pipeline.some((s) => anchor === `macro:${s.id}`)) box.expandByVector(new THREE.Vector3(macroL.stages[0].w * 0.4, 0, 0));
    const height = box.max.y - box.min.y;
    if (pad) box.min.y -= height * (NARROW.matches ? 0.5 : 0.32);
    box.max.y += height * 0.06; // spazio per il selettore dello zoom in alto
    if (!keep && lod.mode !== "auto") lod.setMode("auto");
    hideTip();
    await sleep(30);
    await tab.codePending;
    // le soglie dipendono dalla scena libera: si calcolano dopo che il codice chiesto insieme è aperto
    // (o ridotto, vedi showSnippet), altrimenti un passo dei Blocchi poteva atterrare nella Panoramica
    const t = thresholds();
    const minDistance = keep ? 6 : level === "macro" ? t.blocks * 1.1 : level === "blocks" ? t.detail * 1.12 : 6;
    await director.focus(box, { showFrame: false, margin, direction: FRONT, minDistance, duration });
    return true;
  }

  /**
   * Scelta di un livello (selettore dello zoom, pannello Viste, link diretti): inquadra tutto il livello
   * e, con `lock`, lo blocca subito (il selettore lo mostra durante tutto il volo, non solo alla fine).
   */
  function flyToLevel(level, { lock = false } = {}) {
    if (lock) lod.setMode(level);
    if (level === "detail") return focusAnchor("detail:layer", { keepMode: lock });
    const anchor = level === "macro" ? "macro:all" : "blocks:all";
    return focusAnchor(anchor, { keepMode: lock, margin: level === "macro" ? 1.05 : 1.14 });
  }

  function focusSection(section) {
    if (!anchorsOf(model).has(section.anchor)) {
      tab.setCaption(`${section.title}: ${model.name} non ha questo pezzo.`);
      return Promise.resolve(false);
    }
    return focusAnchor(section.anchor, { layer: section.layer ?? null, also: section.also ?? [] });
  }

  /** Cambia il layer del Dettaglio (pannello Viste, clic su un blocco, ancore del percorso). */
  function selectLayer(number, { fly = false } = {}) {
    if (!layerByNumber(model, number)) return;
    selectedLayer = number;
    ensureAnatomy();
    views.update({ model, expanded: expanded.size > 0, layer: selectedLayer });
    if (fly) focusAnchor("detail:layer");
  }

  /** Entrando nel Dettaglio con la rotellina: il layer è quello del blocco sotto il punto guardato. */
  function followTargetLayer() {
    // solo lo zoom dello studente cambia layer: un volo del tab (o l'attesa prima del volo) sa già dove va
    if (director.flight || performance.now() - lastUserInput > USER_INPUT_WINDOW_MS) return;
    const x = host.controls.target.x;
    let best = -1;
    let bestD = Infinity;
    strip.blocks.forEach((b, i) => {
      if (b.type !== "layer") return;
      const r = stripL.blocks[i];
      const d = Math.abs(r.x - x);
      if (d < bestD) [best, bestD] = [i, d];
    });
    if (best < 0 || bestD > STRIP_SNAP) return;
    const rep = strip.blocks[best].rep;
    if (blockOfLayer(strip, selectedLayer) !== best) selectLayer(rep);
  }

  function toggleExpandAll() {
    expanded = expanded.size ? new Set() : new Set(strip.groups.map((g) => g.id));
    afterStripChange(expanded.size > 0 ? "Ripetizioni srotolate: ogni layer ha il suo blocco." : "Ripetizioni compresse in blocchi «× N».");
  }

  function toggleGroup(id) {
    // la didascalia parla del gruppo cliccato: altri gruppi possono restare srotolati
    const opening = !expanded.has(id);
    if (opening) expanded.add(id);
    else expanded.delete(id);
    afterStripChange(opening ? "Gruppo srotolato nei layer veri: clic sulla sua etichetta per comprimerlo di nuovo." : "Gruppo compresso in «× N».");
  }

  function afterStripChange(text) {
    ui.player.stop();
    rebuild();
    tab.setCaption(text);
    focusAnchor("blocks:layers");
  }

  /**
   * Cambio di modello: azione ferma, scene del vecchio liberate, camera sulla nuova architettura.
   * `stopAction: false` quando lo chiede un'azione che sta partendo (deve tornare al modello principale).
   */
  async function setModel(id, { fly = true, stopAction = true } = {}) {
    const next = MODEL_BY_ID.get(id);
    if (!next || next === model) return false;
    if (stopAction) ui.player.stop();
    // il codice aperto parla del modello di prima (gli estratti e il pseudocodice sono di V4.1): accanto a
    // un altro modello sarebbe fuorviante. Chi cambia modello e poi mostra codice (passo, azione) lo riapre.
    tab.hideCode();
    model = next;
    expanded = new Set();
    selectedLayer = defaultLayer(model);
    rebuild();
    onModelChange(model);
    tab.setCaption(`${model.name}: ${model.headline}.`);
    if (fly) await home();
    return true;
  }

  /** Vista di casa: tutta la striscia, con un po' d'aria per la barra dello zoom e il NavPad negli angoli. */
  function home() {
    if (lod.mode !== "auto") lod.setMode("auto");
    return focusAnchor("blocks:all", { margin: 1.14 });
  }

  /** Dove vive un meccanismo della vista d'insieme: inquadra il suo segnaposto e lo fa pulsare. */
  async function focusMechanism(regionId) {
    const mech = model.mechanisms[regionId];
    if (!mech) return false;
    const ok = await focusAnchor(mech.at, { layer: mech.layer ?? null, wide: mech.at.startsWith("detail:") && mech.at !== "detail:layer" });
    const index = regions.findIndex((r) => r.id === regionId);
    tab.setCaption(`${index + 1}. ${regions[index].title} in ${model.name}: ${mech.text}`);
    pulseMarker(regionId);
    return ok;
  }

  /** Il segnaposto pulsa per 1,6 s (un ascoltatore per frame solo per quel tempo). */
  let stopPulse = null;
  function pulseMarker(regionId) {
    stopPulse?.();
    const sprites = Object.values(markers).flat().filter((s) => s.userData.mechanism === regionId);
    if (!sprites.length) return;
    const base = sprites.map((s) => s.scale.clone());
    if (prefersReducedMotion()) {
      // niente movimento: il segnaposto resta un po' più grande per lo stesso tempo, senza pulsare
      sprites.forEach((s, i) => s.scale.copy(base[i]).multiplyScalar(1.25));
      const timer = setTimeout(() => stopPulse?.(), 1600);
      stopPulse = () => {
        clearTimeout(timer);
        sprites.forEach((s, i) => s.scale.copy(base[i]));
        stopPulse = null;
      };
      return;
    }
    let t = 0;
    const off = host.onFrame((dt) => {
      t += dt;
      const k = 1 + 0.28 * Math.sin(t * 9) * Math.max(0, 1 - t / 1.6);
      sprites.forEach((s, i) => s.scale.copy(base[i]).multiplyScalar(k));
      if (t >= 1.6) stopPulse();
    });
    stopPulse = () => {
      off();
      sprites.forEach((s, i) => s.scale.copy(base[i]));
      stopPulse = null;
    };
  }

  // --- percorso ---------------------------------------------------------------------------------
  async function selectStep(item) {
    if (model.id !== PRIMARY_MODEL_ID) {
      await setModel(PRIMARY_MODEL_ID, { fly: false });
      tab.setCaption(`Il percorso segue ${model.name}: torno a questo modello.`);
    } else tab.setCaption("");
    if (item.code) showSnippet(item.code);
    else tab.hideCode();
    if (item.charts) openChartsIfRoom(item.charts);
    if (item.anchor) focusAnchor(item.anchor, { layer: item.layer ?? null, also: item.also ?? [], wide: item.anchor.startsWith("detail:") && item.anchor !== "detail:layer" });
  }

  /**
   * Grafici chiesti da un passo o da un'azione: si aprono da soli solo se accanto resta abbastanza
   * scena (l'architettura è una striscia larga: a 1280 px con la sidebar aperta resterebbero 440 px).
   * Altrimenti il pulsante «Grafici» si illumina per qualche secondo: sono lì quando li vuoi.
   */
  function openChartsIfRoom(sectionId) {
    if (ui.charts.isOpen || (!NARROW.matches && stage.clientWidth - CHARTS_WIDTH >= MIN_SCENE_BESIDE_CHARTS)) {
      tab.openCharts(sectionId);
      return;
    }
    const fab = ui.charts.button;
    clearTimeout(chartsHintTimer);
    fab.classList.remove("mllm-charts-hint");
    void fab.offsetWidth; // l'animazione riparte anche se era già accesa
    fab.classList.add("mllm-charts-hint");
    fab.title = "Questo passo ha dei grafici: aprili qui";
    chartsHintTimer = setTimeout(() => {
      fab.classList.remove("mllm-charts-hint");
      fab.removeAttribute("title");
    }, CHARTS_HINT_MS);
  }
  let chartsHintTimer = 0;

  function showSnippet(key) {
    const s = FULL_SNIPPETS[key];
    if (!s) return tab.hideCode();
    const request = tab.showCode(s);
    request.then(() => {
      if (request !== tab.codePending || NARROW.matches) return;
      const panel = ui.codePanel.root;
      if (panel.hidden) return;
      // sotto i grafici aperti il codice sarebbe una striscia tagliata: resta ridotto finché si chiudono
      if (ui.charts.isOpen) return minimizeCodeForCharts();
      // Accanto alla striscia larga dell'architettura, con meno di ~600 px di scena libera (1280 px con la
      // sidebar aperta: ne restavano 350) il codice parte ridotto alla sua intestazione, come sul telefono:
      // il titolo dice cosa c'è, ▁ lo espande. Se lo studente l'ha espanso a mano, resta espanso.
      if (!codeExpandedByUser && stage.clientWidth - panel.offsetWidth - 14 < MIN_SCENE_BESIDE_CODE) panel.classList.add("minimized");
    });
  }

  /**
   * Grafici e codice stanno entrambi a destra, i grafici sopra: del codice restava visibile una striscia
   * di 100 px con il titolo tagliato, che in più toglieva quello spazio alla scena. Con i grafici aperti
   * il codice si riduce alla sua intestazione (nascosta sotto i grafici) e alla chiusura torna com'era.
   */
  let codeMinimizedForCharts = false;
  function minimizeCodeForCharts() {
    const panel = ui.codePanel.root;
    if (panel.hidden || panel.classList.contains("minimized")) return;
    panel.classList.add("minimized");
    codeMinimizedForCharts = true;
  }
  function syncCodeWithCharts() {
    if (NARROW.matches) return;
    if (ui.charts.isOpen) minimizeCodeForCharts();
    else if (codeMinimizedForCharts) {
      codeMinimizedForCharts = false;
      if (!ui.codePanel.root.hidden && !ui.codePanel.userMinimized) ui.codePanel.root.classList.remove("minimized");
    }
  }
  /** Lo studente ha espanso il codice con ▁: la sua scelta vale per i frammenti seguenti. */
  let codeExpandedByUser = false;
  const codeToggle = ui.codePanel.root.querySelector("header button");
  const onCodeToggle = () => (codeExpandedByUser = !ui.codePanel.root.classList.contains("minimized"));
  codeToggle?.addEventListener("click", onCodeToggle);

  // --- clic e passaggio del mouse -----------------------------------------------------------------
  function visibleMarkers() {
    return markers[lod.current] ?? [];
  }

  function onSceneClick(e) {
    director.noteUserIntent();
    const hitMarker = host.pick(e, visibleMarkers(), false)[0];
    if (hitMarker) {
      hideTip();
      onMarker(hitMarker.object.userData.mechanism);
      return;
    }
    if (lod.current === "blocks") {
      for (const hit of host.pick(e, blocksScene.pickables)) {
        const u = hit.object.userData;
        if (u.toggle) return toggleGroup(u.toggle);
        if (u.block !== undefined) {
          const b = strip.blocks[u.block];
          if (b.type === "layer") {
            tab.setCaption(`${b.title} · ${b.sub} · ${b.paramsLabel}: ${b.text}`);
            selectLayer(b.rep, { fly: true });
          } else focusAnchor(`blocks:${b.id}`);
          return;
        }
        if (u.anchor) return focusAnchor(u.anchor);
      }
      return;
    }
    if (lod.current === "detail") {
      const objects = anatomy ? [anatomy.group, ...contextScene.pickables] : contextScene.pickables;
      for (const hit of host.pick(e, objects)) {
        const u = hit.object.userData;
        if (u.card) return showCardInfo(u.card);
        if (u.block !== undefined) {
          const b = strip.blocks[u.block];
          if (b?.type === "layer") return selectLayer(b.rep, { fly: true });
        }
      }
    }
  }

  /** Clic su una scheda dell'anatomia: la accende e scrive nella didascalia il suo tensore. */
  function showCardInfo(id) {
    anatomy.highlight(id);
    if (id === "experts") {
      const moe = model.params.moe;
      tab.setCaption(`${moe.routed} esperti instradati, ${moe.active} attivi per token: ${model.anatomy.ffn.expert.tensor} (${model.anatomy.ffn.expert.dtype ?? ""}).`);
      return;
    }
    const spec = anatomyCards(model).find((c) => c.id === id);
    if (!spec) return;
    const s = { ...spec, ...(spec.alt?.[layerByNumber(model, selectedLayer).kind] ?? {}) };
    tab.setCaption([s.title, s.sub, s.tensor, s.dtype].filter(Boolean).join(" · "));
  }

  function updateHover(e) {
    const hit = host.pick(e, visibleMarkers(), false)[0];
    if (!hit) {
      hideTip();
      const over = lod.current === "blocks" && host.pick(e, blocksScene.pickables).length > 0;
      canvas.style.cursor = over ? "pointer" : "";
      return;
    }
    const region = regions.find((r) => r.id === hit.object.userData.mechanism);
    const mech = model.mechanisms[region.id];
    const status = mech.status === "absent" ? "Qui vivrebbe, ma questo modello non lo usa. " : mech.status === "study" ? "Non è un pezzo del modello. " : "";
    tip.replaceChildren(el("strong", {}, `${regions.indexOf(region) + 1}. ${region.title}`), el("span", {}, `${status}${mech.text}`), el("small", {}, "Clic: apri nella vista d'insieme"));
    const r = stage.getBoundingClientRect();
    tip.hidden = false;
    const x = Math.min(e.clientX - r.left + 14, r.width - tip.offsetWidth - 8);
    const y = Math.min(e.clientY - r.top + 14, r.height - tip.offsetHeight - 8);
    tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
    canvas.style.cursor = "pointer";
  }

  function hideTip() {
    tip.hidden = true;
    if (active) canvas.style.cursor = "";
  }

  // --- contesto per le azioni ----------------------------------------------------------------------
  function createActionContext() {
    const ctx = {
      plots: null,
      get model() {
        return model;
      },
      get strip() {
        return strip;
      },
      get blocks() {
        return blocksScene;
      },
      get macro() {
        return macroScene;
      },
      anatomy: () => anatomy,
      prepare(level, { charts = null } = {}) {
        // La Panoramica si legge solo da vicino, dove lo zoom automatico mostrerebbe già i Blocchi: le
        // azioni sulla Panoramica la bloccano (il selettore lo mostra; "Auto" o un passo la sbloccano).
        lod.setMode(level === "macro" ? "macro" : "auto");
        tab.hideCode(); // il codice lasciato aperto dall'azione di prima non c'entra con questa
        tab.setCaption(""); // l'azione ha la sua didascalia nella barra del player
        director.hideFrame();
        if (charts) openChartsIfRoom(charts);
      },
      rest() {
        blocksScene?.highlight(null);
        if (blocksScene) blocksScene.packet.visible = false;
        anatomy?.highlight(null);
        anatomy?.route([], 0);
        macroScene?.highlight(null);
        macroScene?.fillPrecision(1);
      },
      focus: (anchor, o = {}) => focusAnchor(anchor, o),
      focusBlocks(indices, { anchor = null } = {}) {
        const box = new THREE.Box3();
        for (const i of indices) if (i >= 0) box.union(blocksScene.blockBoxes[i]);
        if (anchor) box.union(anchorBox(anchor));
        const height = box.max.y - box.min.y;
        box.min.y -= height * (NARROW.matches ? 0.6 : 0.4);
        box.max.y += height * 0.15;
        const t = thresholds();
        if (lod.mode !== "auto") lod.setMode("auto");
        return director.focus(box, { showFrame: false, margin: 1.15, direction: FRONT, minDistance: t.detail * 1.12, duration: 1 });
      },
      setLayer: (n) => selectLayer(n),
      ensurePrimary: () => (model.id === PRIMARY_MODEL_ID ? Promise.resolve() : setModel(PRIMARY_MODEL_ID, { fly: false, stopAction: false })),
      showCode: (key) => (key ? showSnippet(key) : tab.hideCode()),
      openCharts: (id) => openChartsIfRoom(id),
      segmentBlocks(id) {
        const s = strip.segments.find((x) => x.id === id);
        if (!s || s.from === undefined) return [];
        return strip.blocks.map((b, i) => i).filter((i) => i >= s.from && i <= s.to && strip.blocks[i].type === "layer");
      },
      layerBlocks: (numbers) => [...new Set(numbers.map((n) => blockOfLayer(strip, n)).filter((i) => i >= 0))],
      blockIndex: (id) => strip.blocks.findIndex((b) => b.id === id),
      packetAt(i) {
        const p = blocksScene.packet;
        if (i < 0 || !stripL.blocks[i]) {
          p.visible = false;
          return;
        }
        p.visible = true;
        p.position.set(stripL.blocks[i].x, blocksScene.railY, 3);
      },
      packetAlong(indices, t) {
        const a = stripL.blocks[indices[0]];
        const b = stripL.blocks[indices.at(-1)];
        const p = blocksScene.packet;
        p.visible = true;
        p.position.set(a.x + (b.x - a.x) * t, blocksScene.railY, 3);
      },
    };
    return ctx;
  }

  // --- API -----------------------------------------------------------------------------------------
  let initialized = false;
  return {
    menu,
    panes: FULL_PANES.map((p) => p.id),
    get model() {
      return model;
    },
    get lod() {
      return lod;
    },
    /** Accende il sottotab: LOD, gruppi, pannelli, barra strumenti, camera della vista completa. */
    activate() {
      active = true;
      camera.near = NEAR;
      camera.far = FAR;
      camera.updateProjectionMatrix();
      lod.setEnabled(true);
      stage.append(toolbar); // i pannelli li mostra index.js, insieme a quelli dell'altro sottotab
      updateThresholds();
      if (!initialized) {
        initialized = true;
        silentSelect = true; // il primo passo conta come visto, ma la camera va sull'architettura intera
        menu.select("full-intro", { scroll: false });
        silentSelect = false;
        // sul telefono la striscia intera è piccola: la prima volta si dice come avvicinarsi
        if (phoneStage()) tab.setCaption(PHONE_HINT);
      }
    },
    /** Spegne il sottotab: niente LOD né gruppi, pannelli nascosti, camera come prima. */
    deactivate() {
      active = false;
      stopPulse?.();
      if (hoverFrame) cancelAnimationFrame(hoverFrame);
      hoverFrame = 0;
      hideTip();
      canvas.style.cursor = "";
      lod.setEnabled(false);
      for (const g of Object.values(roots)) g.visible = false;
      toolbar.remove();
      camera.near = savedCamera.near;
      camera.far = savedCamera.far;
      camera.updateProjectionMatrix();
    },
    home,
    /** Confini della navigazione libera per il livello attivo. */
    bounds() {
      const toBox = (b) => new THREE.Box3(new THREE.Vector3(b.minX, b.minY, -2), new THREE.Vector3(b.maxX, b.maxY, 2));
      if (lod.current === "macro") return toBox(macroL.box);
      const box = toBox(stripL.box);
      if (lod.current === "detail" && anatomy) box.union(rectBox({ x: (anatomy.box.minX + anatomy.box.maxX) / 2, y: (anatomy.box.minY + anatomy.box.maxY) / 2, w: anatomy.box.maxX - anatomy.box.minX, h: anatomy.box.maxY - anatomy.box.minY }, 0, anatomy.group.position));
      return box;
    },
    focusMechanism,
    setModel,
    selectLayer,
    /** goto per gli id della vista completa (vedi catalog.js). → false se l'id non esiste. */
    async goto({ kind, id }) {
      if (kind === "step") return menu.reveal(id, { sidebar });
      if (kind === "action") {
        if (!FULL_ACTION_IDS.has(id)) return false;
        ui.player.start(id);
        return true;
      }
      if (kind !== "view") return false;
      if (id === "full") return true;
      const level = FULL_LEVELS.find((l) => l.id === id);
      if (level) {
        await flyToLevel(level.level, { lock: true });
        return true;
      }
      if (id.startsWith("full-model-")) {
        const target = MODEL_BY_ID.get(id.slice("full-model-".length));
        if (!target) return false;
        if (target !== model) await setModel(target.id);
        else await home();
        sidebar.select("full-modelli");
        return true;
      }
      if (id.startsWith("full-layer-")) {
        const n = Number(id.slice("full-layer-".length));
        if (!Number.isInteger(n) || !layerByNumber(model, n)) return false;
        selectLayer(n, { fly: true });
        return true;
      }
      if (id.startsWith("full-mech-")) {
        const region = id.slice("full-mech-".length);
        if (!model.mechanisms[region]) return false;
        await focusMechanism(region);
        return true;
      }
      const section = FULL_SECTIONS.find((s) => s.id === id);
      if (section) {
        await focusSection(section);
        return true;
      }
      return false;
    },
    /** Ripristino delle scelte di un altro sottotab non necessario: qui si espone solo lo stato. */
    get actionIds() {
      return FULL_ACTION_IDS;
    },
    chartSections: FULL_CHART_SECTIONS,
    dispose() {
      this.deactivate();
      clearTimeout(chartsHintTimer);
      codeToggle?.removeEventListener("click", onCodeToggle);
      offClick();
      offResize();
      offInput();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      if (hoverFrame) cancelAnimationFrame(hoverFrame);
      lod.dispose();
      menu.dispose();
      for (const g of Object.values(roots)) disposeTree(g);
      tip.remove();
      toolbar.remove();
    },
  };
}

/** Distanza massima (unità) dal centro di un blocco perché lo zoom ne apra il Dettaglio. */
const STRIP_SNAP = 60;

const capitalize = (s) => s.charAt(0) + s.slice(1).toLowerCase();
