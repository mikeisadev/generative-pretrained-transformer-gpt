/**
 * Layout standard di un tab: sidebar a sinistra, stage 3D al centro, pannelli fluttuanti
 * opzionali (codice, grafici), barra delle azioni e didascalia.
 *
 *   const ui = createTabLayout(root, { code: true, charts: true, player: true });
 *   ui.host.scene.add(...); ui.sidebar.addPane({ id: "percorso", title: "Percorso" });
 */
import { CameraDirector } from "./director.js";
import { el } from "./dom.js";
import { SceneHost } from "./sceneHost.js";
import { ChartsPanel } from "./ui/charts.js";
import { CodePanel } from "./ui/codePanel.js";
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
  const toggle = el(
    "button",
    { className: "icon-button sidebar-toggle", type: "button", ariaLabel: "Mostra o nascondi la sidebar", onclick: () => root.classList.toggle("sidebar-collapsed") },
    "☰",
  );
  stage.append(toggle);
  if (window.matchMedia("(max-width: 900px)").matches) root.classList.add("sidebar-collapsed");
  new ResizeObserver(() => host.resize()).observe(root);

  return { root, sidebar, stage, caption, host, director, codePanel, charts: chartsPanel, player: actions };
}
