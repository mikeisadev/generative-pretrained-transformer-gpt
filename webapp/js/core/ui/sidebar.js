/**
 * Sidebar sinistra collassabile, divisa in "pannelli" selezionabili con una barra a schede
 * (es. "Percorso", "Azioni", "Viste"). Ogni tab della web app ne ha una propria.
 *
 *   const sidebar = new Sidebar(tabRoot);
 *   const pane = sidebar.addPane({ id: "azioni", title: "Azioni" });   // restituisce l'elemento del pannello
 *   sidebar.select("azioni");
 *   sidebar.movePane("cervello", "fonti");                              // la scheda "Cervello" prima di "Fonti"
 *
 * I pannelli condividono il contenitore che scorre (`body`): ognuno ricorda il suo scorrimento, così
 * tornando a un pannello lo si ritrova dov'era e uno nuovo si apre dall'alto.
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
    this.scrolls = new Map(); // id del pannello → scrollTop quando lo si è lasciato
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
    const previous = this.current;
    if (previous !== null && previous !== id) this.scrolls.set(previous, this.body.scrollTop);
    for (const [paneId, { button, pane, hint }] of this.panes) {
      const active = paneId === id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      pane.hidden = !active;
      if (active) this.hint.textContent = hint;
    }
    this.hint.hidden = !this.hint.textContent;
    // Lo scorrimento del pannello scelto (dall'alto la prima volta): prima degli ascoltatori, che possono
    // portare in vista un elemento preciso (es. menu.reveal di un passo).
    if (previous !== id && this.panes.has(id)) this.body.scrollTop = this.scrolls.get(id) ?? 0;
    for (const fn of this.listeners) fn(id);
  }

  /** fn(idPannello) a ogni cambio di pannello. */
  onSelect(fn) {
    this.listeners.add(fn);
  }

  /**
   * Sposta la scheda (e il pannello) `id` subito prima di `beforeId`, o in fondo se `beforeId` è null:
   * es. un pannello aggiunto dal nucleo che deve stare prima di "Fonti" anche quando i pannelli del tab
   * nascono dopo. Restituisce false se uno dei due pannelli non esiste.
   */
  movePane(id, beforeId = null) {
    const entry = this.panes.get(id);
    const before = beforeId === null ? null : this.panes.get(beforeId);
    if (!entry || (beforeId !== null && !before) || id === beforeId) return false;
    if (before) {
      if (entry.button.nextElementSibling !== before.button) this.nav.insertBefore(entry.button, before.button);
      if (entry.pane.nextElementSibling !== before.pane) this.body.insertBefore(entry.pane, before.pane);
    } else {
      this.nav.append(entry.button);
      this.body.append(entry.pane);
    }
    return true;
  }

  /**
   * Nasconde (o rimostra) un pannello e la sua scheda: es. i pannelli del sottotab non attivo di un
   * tab con due viste. Se il pannello nascosto era quello aperto, si apre il primo ancora visibile.
   */
  setPaneHidden(id, hidden) {
    const entry = this.panes.get(id);
    if (!entry) return;
    entry.button.hidden = Boolean(hidden);
    if (!hidden || entry.pane.hidden) return;
    const next = [...this.panes].find(([, e]) => !e.button.hidden);
    if (next) this.select(next[0]);
    else entry.pane.hidden = true;
  }

  /** Il pannello aperto (id) o null. */
  get current() {
    for (const [id, { pane }] of this.panes) if (!pane.hidden) return id;
    return null;
  }

  toggleCollapsed(force) {
    this.root.closest(".tab")?.classList.toggle("sidebar-collapsed", force);
  }
}
