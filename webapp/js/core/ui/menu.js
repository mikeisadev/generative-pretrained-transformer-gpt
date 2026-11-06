/**
 * Menu a fisarmonica condiviso: punti principali → sottopunti → spiegazione e
 * riquadro "Approfondimento" con le formule (KaTeX).
 *
 * Formato dei passi:
 *   { id, title, children: [ { id, title, text (HTML fidato, scritto da noi),
 *                              math?: [{tex, note?}], extra?: (item) => Node, ...qualsiasi } ] }
 * Quando l'utente seleziona un sottopunto viene chiamato onSelect(item, title).
 *
 * Con l'opzione `tabId` il menu segna i passi già visti (✓, "n/m passi visti", azzerabile;
 * vedi core/progress.js) e mette in fondo a ogni passo "Copia link a questo passo"
 * (#<tabId>/go=step:<id>, aperto da app.goto → tab.goto → menu.select).
 */
import katex from "../../../vendor/katex/katex.mjs";
import { el, fillTemplate, prefersReducedMotion } from "../dom.js";
import { progress } from "../progress.js";
import { formatHash } from "../search.js";
import { copyLink } from "./share.js";

/** Riquadro "Approfondimento" con formule LaTeX. */
export function mathBox(items, vars = {}, title = "Approfondimento") {
  const box = el("details", { className: "deep" }, el("summary", {}, title));
  for (const { tex, note } of items) {
    const holder = el("div", { className: "formula" });
    katex.render(fillTemplate(tex, vars), holder, { displayMode: true, throwOnError: false });
    box.append(holder);
    if (note) box.append(el("p", { className: "hint" }, fillTemplate(note, vars)));
  }
  return box;
}

export class Menu {
  /**
   * @param {HTMLElement} root
   * @param {object[]} steps
   * @param {object} [options]
   * @param {object} [options.vars] valori per i segnaposto {{...}}
   * @param {(item: object, title: string) => void} [options.onSelect]
   * @param {string} [options.tabId] id del tab (es. "rag"): abilita i progressi ("n/m passi", ✓ sui
   *   passi visti) e il pulsante "Copia link a questo passo" (#<tabId>/go=step:<id>). Senza, il menu
   *   funziona come prima (serve ai menu secondari, es. quello della vista Contesto).
   * @param {boolean} [options.trackProgress] false per non segnare i passi visti anche con tabId
   */
  constructor(root, steps, { vars = {}, onSelect = () => {}, tabId = null, trackProgress = true } = {}) {
    this.root = root;
    this.vars = vars;
    this.onSelect = onSelect;
    this.tabId = tabId;
    this.trackProgress = Boolean(tabId) && trackProgress;
    /** id → { item, wrapper, section, header, title, button, sectionId } */
    this.items = new Map();
    /** id della sezione → { section, header, count, ids } */
    this.sections = new Map();
    this.order = [];
    this.current = null;
    if (this.trackProgress) this.root.append(this.renderProgress());
    this.render(steps);
    if (this.trackProgress) {
      this.onProgress = (e) => {
        if (e.detail.tab === this.tabId) this.refreshProgress();
      };
      progress.addEventListener("change", this.onProgress);
      this.refreshProgress();
    }
  }

  render(steps) {
    const fill = (t) => fillTemplate(t, this.vars);
    for (const step of steps) {
      const section = el("section", { className: "step" });
      // Con i progressi l'intestazione è "titolo · 2/3 · ▸" (flex: vedi css/palette.css).
      const count = this.trackProgress ? el("span", { className: "step-count" }) : null;
      const header = el("button", { className: `step-header${count ? " tracked" : ""}`, type: "button" }, count ? el("span", { className: "step-title" }, fill(step.title)) : fill(step.title), count);
      header.setAttribute("aria-expanded", "false");
      const list = el("div", { className: "step-children" });
      header.addEventListener("click", () => {
        const open = header.getAttribute("aria-expanded") !== "true";
        header.setAttribute("aria-expanded", String(open));
        section.classList.toggle("open", open);
      });
      const ids = [];
      for (const item of step.children) {
        const title = fill(item.title);
        const button = el("button", { className: "item-header", type: "button" }, title);
        const body = el("div", { className: "item-body" });
        body.innerHTML = fill(item.text ?? "");
        const extra = item.extra?.(item);
        if (extra) body.append(extra);
        if (item.math?.length) body.append(mathBox(item.math, this.vars));
        if (this.tabId) body.append(this.linkButton(item.id));
        const wrapper = el("div", { className: "item", dataset: { step: item.id } }, button, body);
        button.addEventListener("click", () => this.select(item.id));
        list.append(wrapper);
        this.items.set(item.id, { item, wrapper, section, header, title, button, sectionId: step.id });
        this.order.push(item.id);
        ids.push(item.id);
      }
      this.sections.set(step.id ?? `section-${this.sections.size}`, { section, header, count, ids });
      section.append(header, list);
      this.root.append(section);
    }
  }

  /**
   * Seleziona un passo: apre la sua sezione, lo evidenzia, lo porta in vista e chiama onSelect.
   * @param {string} id
   * @param {object} [options]
   * @param {"nearest"|"start"|"center"|false} [options.scroll] come portarlo in vista (false = non scorrere)
   * @returns {boolean} false se il passo non esiste
   */
  select(id, { scroll = "nearest" } = {}) {
    const entry = this.items.get(id);
    if (!entry) return false;
    this.root.querySelectorAll(".item.active").forEach((n) => n.classList.remove("active"));
    entry.wrapper.classList.add("active");
    entry.section.classList.add("open");
    entry.header.setAttribute("aria-expanded", "true");
    if (scroll) {
      // Al fotogramma dopo: se chi chiama ha appena reso visibile il pannello (sidebar.select),
      // ora il layout c'è e lo scorrimento funziona.
      requestAnimationFrame(() => entry.wrapper.scrollIntoView({ block: scroll, behavior: prefersReducedMotion() ? "auto" : "smooth" }));
    }
    this.current = id;
    if (this.trackProgress) progress.mark(this.tabId, id);
    this.onSelect(entry.item, entry.title);
    return true;
  }

  /**
   * Come select, ma prima rende visibile il menu: apre la sidebar (anche il cassetto sul telefono)
   * e il pannello che contiene il menu, poi porta il passo in cima. È ciò che serve a tab.goto
   * per un passo: `goto: ({kind, id}) => kind === "step" && menu.reveal(id, { sidebar: ui.sidebar })`.
   * @param {string} id
   * @param {object} [options]
   * @param {import("./sidebar.js").Sidebar} [options.sidebar] la sidebar del tab
   * @param {boolean} [options.openSidebar] false per lasciare la sidebar com'è
   * @returns {boolean} false se il passo non esiste
   */
  reveal(id, { sidebar = null, openSidebar = true } = {}) {
    if (!this.items.has(id)) return false;
    if (sidebar) {
      if (openSidebar) sidebar.toggleCollapsed(false);
      const pane = this.root.closest(".sidebar-pane")?.dataset.pane;
      if (pane && sidebar.panes.has(pane)) sidebar.select(pane);
    }
    return this.select(id, { scroll: "start" });
  }

  /** Passa allo step successivo (+1) o precedente (-1). */
  move(delta) {
    const i = this.order.indexOf(this.current);
    const next = this.order[Math.min(this.order.length - 1, Math.max(0, i + delta))];
    this.select(next ?? this.order[0]);
  }

  /** Indirizzo del link diretto a un passo (null se il menu non conosce il suo tab). */
  linkTo(id) {
    return this.tabId ? formatHash(this.tabId, { kind: "step", id }) : null;
  }

  /** Rimuove l'ascoltatore dei progressi (da chiamare se il menu viene distrutto prima della pagina). */
  dispose() {
    if (this.onProgress) progress.removeEventListener("change", this.onProgress);
  }

  // --- interni ------------------------------------------------------------------------------

  linkButton(id) {
    return el(
      "button",
      {
        type: "button",
        className: "item-link",
        title: "Copia negli appunti l'indirizzo che apre direttamente questo passo",
        onclick: () => copyLink(this.linkTo(id), { label: "Link al passo copiato negli appunti" }),
      },
      linkIcon(),
      "Copia link a questo passo",
    );
  }

  renderProgress() {
    this.progressText = el("span", { className: "menu-progress-text" });
    this.progressFill = el("span", { className: "menu-progress-fill" });
    this.progressBar = el("span", { className: "menu-progress-bar", role: "progressbar", "aria-valuemin": "0" }, this.progressFill);
    this.resetButton = el("button", { type: "button", className: "menu-progress-reset", onclick: () => this.askReset() }, "Azzera");
    return el("div", { className: "menu-progress" }, this.progressText, this.progressBar, this.resetButton);
  }

  /** "Azzera" chiede conferma con un secondo clic (niente finestre di dialogo). */
  askReset() {
    if (this.resetButton.classList.contains("confirm")) {
      clearTimeout(this.resetTimer);
      this.resetButton.classList.remove("confirm");
      this.resetButton.textContent = "Azzera";
      progress.reset(this.tabId);
      // Il passo aperto resta quello che lo studente sta leggendo: lo contiamo di nuovo.
      if (this.current) progress.mark(this.tabId, this.current);
      return;
    }
    this.resetButton.classList.add("confirm");
    this.resetButton.textContent = "Confermi?";
    this.resetTimer = setTimeout(() => {
      this.resetButton.classList.remove("confirm");
      this.resetButton.textContent = "Azzera";
    }, 3500);
  }

  refreshProgress() {
    const seen = progress.visited(this.tabId);
    let total = 0;
    for (const [id, { button, wrapper }] of this.items) {
      const visited = seen.has(id);
      if (visited) total++;
      wrapper.classList.toggle("visited", visited);
      let check = button.querySelector(".item-check");
      if (visited && !check) {
        check = el("span", { className: "item-check", role: "img", "aria-label": "già visto" }, "✓");
        button.append(check);
      } else if (!visited && check) check.remove();
    }
    for (const { section, count, ids } of this.sections.values()) {
      const done = ids.filter((id) => seen.has(id)).length;
      count.textContent = `${done}/${ids.length}`;
      count.setAttribute("aria-label", `${done} passi visti su ${ids.length}`);
      section.classList.toggle("done", done === ids.length);
    }
    const n = this.order.length;
    this.progressText.textContent = `${total}/${n} passi visti`;
    this.progressFill.style.width = `${n ? (100 * total) / n : 0}%`;
    this.progressBar.setAttribute("aria-valuemax", String(n));
    this.progressBar.setAttribute("aria-valuenow", String(total));
    this.progressBar.setAttribute("aria-label", "Passi del percorso già visti");
    this.resetButton.hidden = total === 0;
  }
}

function linkIcon() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  svg.innerHTML = '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>';
  return svg;
}
