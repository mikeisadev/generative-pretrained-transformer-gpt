/**
 * Layout standard di un tab: sidebar a sinistra, stage 3D al centro, pannelli fluttuanti
 * opzionali (codice, grafici), barra delle azioni e didascalia.
 *
 *   const ui = createTabLayout(root, { code: true, charts: true, player: true });
 *   ui.host.scene.add(...); ui.sidebar.addPane({ id: "percorso", title: "Percorso" });
 *
 * createSceneStack(stage, { root }) monta solo la pila di scena (SceneHost, CameraDirector,
 * Navigation, NavPad): serve a chi ha bisogno di una seconda scena con la stessa navigazione
 * (le rappresentazioni alternative di js/core/representations.js).
 */
import { CameraDirector } from "./director.js";
import { el } from "./dom.js";
import { Navigation } from "./navigation.js";
import { SceneHost } from "./sceneHost.js";
import { ChartsPanel } from "./ui/charts.js";
import { CodePanel } from "./ui/codePanel.js";
import { NavPad } from "./ui/navPad.js";
import { ActionPlayer } from "./ui/player.js";
import { Sidebar } from "./ui/sidebar.js";

/**
 * @param {HTMLElement} root contenitore del tab (vuoto)
 * @param {object} [o]
 * @param {boolean} [o.code] finestra del codice a destra
 * @param {boolean} [o.charts] pulsante fluttuante "Grafici" + pannello
 * @param {boolean} [o.player] barra delle azioni animate
 * @param {string} [o.hint] suggerimento in cima alla sidebar
 * @param {number} [o.background] colore di sfondo della scena
 * @returns {{root, sidebar, stage, caption, host, director, nav, codePanel, charts, player}}
 *   `nav` è la navigazione libera (js/core/navigation.js): nav.setHome(fn), nav.setBounds(box), …
 */
export function createTabLayout(root, { code = false, charts = false, player = false, hint = "", background } = {}) {
  root.classList.add("tab");
  const sidebar = new Sidebar(root, { hint });
  const stage = el("main", { className: "tab-stage" });
  const caption = el("footer", { className: "caption", ariaLive: "polite" });
  root.append(stage, caption);

  const codePanel = code ? new CodePanel(root) : null;
  const chartsPanel = charts ? new ChartsPanel(root) : null;
  const coveredRight = () => Math.max(codePanel?.coveredWidth() ?? 0, chartsPanel?.coveredWidth() ?? 0);
  const { host, director, nav, player: actions } = createSceneStack(stage, { root, player, codePanel, charts: chartsPanel, coveredRight, background });
  codePanel?.onResize(() => host.resize());
  if (chartsPanel?.onResize) chartsPanel.onResize(() => host.resize());
  // Grafici aperti: il codice si riduce accanto a loro invece di restarne una striscia sotto (codePanel.js).
  if (codePanel && chartsPanel) codePanel.followCharts(chartsPanel);

  const toggle = el(
    "button",
    { className: "icon-button sidebar-toggle", type: "button", ariaLabel: "Mostra o nascondi la sidebar", onclick: () => root.classList.toggle("sidebar-collapsed") },
    "☰",
  );
  stage.append(toggle);
  // Su schermi stretti la sidebar è un cassetto sopra lo stage: parte chiusa e si chiude toccando la scena.
  const narrow = window.matchMedia("(max-width: 900px)");
  if (narrow.matches) root.classList.add("sidebar-collapsed");
  host.renderer.domElement.addEventListener("pointerdown", () => {
    if (narrow.matches) root.classList.add("sidebar-collapsed");
  });
  new ResizeObserver(() => host.resize()).observe(root);

  return { root, sidebar, stage, caption, host, director, nav, codePanel, charts: chartsPanel, player: actions };
}

/**
 * Pila di scena: SceneHost + CameraDirector (+ ActionPlayer) + Navigation + NavPad, collegati come in
 * ogni tab. La usa createTabLayout per lo stage del tab e js/core/representations.js per il livello
 * delle viste alternative: così la navigazione libera è identica ovunque (pad, aiuto, "Centra", foto,
 * schermo intero) senza duplicarne il montaggio.
 *
 *   const { host, director, nav } = createSceneStack(layerElement, { root: layerElement });
 *
 * @param {HTMLElement} stage elemento che ospita il canvas, la barra del player, il pad e l'aiuto
 * @param {object} o
 * @param {HTMLElement} o.root radice della navigazione e del pad: il suo `hidden` dice se la scena è
 *   visibile, i suoi clic contano come richieste dello studente, `data-tab` dà il nome alla foto
 * @param {boolean} [o.player] crea la barra delle azioni animate (dentro `stage`)
 * @param {import("./ui/codePanel.js").CodePanel | null} [o.codePanel] pannello di cui il pad tiene conto
 * @param {import("./ui/charts.js").ChartsPanel | null} [o.charts] pannello di cui il pad tiene conto
 * @param {() => number} [o.coveredRight] pixel coperti a destra (view offset della camera)
 * @param {number} [o.background] colore di sfondo della scena
 * @returns {{host: SceneHost, director: CameraDirector, nav: Navigation, player: ActionPlayer | null}}
 *   `nav.pad` è il NavPad; `nav.dispose()` toglie anche il pad, `host.dispose()` il canvas
 */
export function createSceneStack(stage, { root, player = false, codePanel = null, charts = null, coveredRight = () => 0, background } = {}) {
  const host = new SceneHost(stage, { coveredRight, ...(background !== undefined ? { background } : {}) });
  const director = new CameraDirector(host);
  const actions = player ? new ActionPlayer(host, stage) : null;
  // Durante un'azione la camera segue l'azione; a fine azione (o quando si ferma) torna a seguirla.
  // Un'azione arrivata all'ultimo passo non è più "in corso": lo studente si muove senza camera libera.
  actions?.on("state", () => director.setActionActive(Boolean(actions.action) && !actionFinished(actions)));
  // "⟲ Ricomincia" con la camera libera: l'azione riparte dall'inizio e la camera torna a seguirla.
  actions?.on("restart", () => director.resumeFollow());

  // Navigazione libera (mouse, tastiera, touch, pad a schermo): uguale in tutti i tab.
  const nav = new Navigation(host, director, { root, player: actions });
  nav.pad = new NavPad(nav, { root, stage, player: actions, codePanel, charts, coveredRight });
  return { host, director, nav, player: actions };
}

/** true se l'azione è ferma sull'ultimo passo completato (il player resta aperto sul risultato). */
function actionFinished(player) {
  return !player.playing && player.steps.length > 0 && player.index >= player.steps.length - 1 && player.t >= 1;
}
