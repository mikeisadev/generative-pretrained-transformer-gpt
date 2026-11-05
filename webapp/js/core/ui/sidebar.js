/**
 * Sidebar sinistra collassabile, divisa in "pannelli" selezionabili con una barra a schede
 * (es. "Percorso", "Azioni", "Viste"). Ogni tab della web app ne ha una propria.
 *
 *   const sidebar = new Sidebar(tabRoot);
 *   const pane = sidebar.addPane({ id: "azioni", title: "Azioni" });   // restituisce l'elemento del pannello
 *   sidebar.select("azioni");
 */
import { el } from "../dom.js";

export class Sidebar {
  constructor(parent, { hint = "" } = {}) {
    this.nav = el("nav", { className: "sidebar-nav", role: "tablist" });
    this.body = el("div", { className: "sidebar-body" });
    this.hint = el("p", { className: "menu-hint" }, hint);
    this.root = el("aside", { className: "tab-sidebar" }, this.nav, this.hint, this.body);
    parent.append(this.root);
    this.panes = new Map();
    this.listeners = new Set();
  }

  /** Aggiunge un pannello e restituisce il suo elemento contenitore. */
  addPane({ id, title, hint = "" }) {
    const button = el("button", { type: "button", role: "tab", dataset: { pane: id }, onclick: () => this.select(id) }, title);
    const pane = el("div", { className: "sidebar-pane", role: "tabpanel", hidden: true, dataset: { pane: id } });
    this.nav.append(button);
    this.body.append(pane);
    this.panes.set(id, { button, pane, hint });
    if (this.panes.size === 1) this.select(id);
    return pane;
  }

  select(id) {
    for (const [paneId, { button, pane, hint }] of this.panes) {
      const active = paneId === id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      pane.hidden = !active;
      if (active) this.hint.textContent = hint;
    }
    this.hint.hidden = !this.hint.textContent;
    for (const fn of this.listeners) fn(id);
  }

  /** fn(idPannello) a ogni cambio di pannello. */
  onSelect(fn) {
    this.listeners.add(fn);
  }

  toggleCollapsed(force) {
    this.root.closest(".tab")?.classList.toggle("sidebar-collapsed", force);
  }
}
