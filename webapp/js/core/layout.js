/**
 * Layout standard di un tab: sidebar a sinistra, stage 3D al centro, pannelli fluttuanti
 * opzionali (codice, grafici), barra delle azioni e didascalia.
 *
 *   const ui = createTabLayout(root, { code: true, charts: true, player: true });
 *   ui.host.scene.add(...); ui.sidebar.addPane({ id: "percorso", title: "Percorso" });
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
  const host = new SceneHost(stage, { coveredRight, ...(background !== undefined ? { background } : {}) });
  codePanel?.onResize(() => host.resize());
  if (chartsPanel?.onResize) chartsPanel.onResize(() => host.resize());

  const director = new CameraDirector(host);
  const actions = player ? new ActionPlayer(host, stage) : null;
  // Durante un'azione la camera segue l'azione; a fine azione (o quando si ferma) torna a seguirla.
  // Un'azione arrivata all'ultimo passo non è più "in corso": lo studente si muove senza camera libera.
  actions?.on("state", () => director.setActionActive(Boolean(actions.action) && !actionFinished(actions)));
  // "⟲ Ricomincia" con la camera libera: l'azione riparte dall'inizio e la camera torna a seguirla.
  actions?.on("restart", () => director.resumeFollow());

  // Navigazione libera (mouse, tastiera, touch, pad a schermo): uguale in tutti i tab.
  const nav = new Navigation(host, director, { root, player: actions });
  nav.pad = new NavPad(nav, { root, stage, player: actions, codePanel, charts: chartsPanel, coveredRight });
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

/** true se l'azione è ferma sull'ultimo passo completato (il player resta aperto sul risultato). */
function actionFinished(player) {
  return !player.playing && player.steps.length > 0 && player.index >= player.steps.length - 1 && player.t >= 1;
}
