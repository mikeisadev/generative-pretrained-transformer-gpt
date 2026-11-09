/**
 * Ricerca globale ("palette"): Ctrl/⌘+K, "/" oppure il pulsante "Cerca" della barra in alto.
 *
 * Cerca in tutti i tab, anche in quelli mai aperti: ogni tab espone un indice leggero in
 * js/tabs/<id>/search.js (`export const searchIndex = [{id, kind, title, text?, keywords?}]`),
 * caricato con un import dinamico alla prima apertura. Se un indice manca o non si carica, la
 * palette funziona con gli altri. In più: i tab stessi e il glossario (core/glossary.js).
 *
 * Scegliere un risultato chiama `goto(tab, {kind, id})` (di solito app.goto); un termine del
 * glossario si apre in una scheda dentro la palette, con il link "Guarda nel tab …".
 *
 *   const palette = new SearchPalette({ tabs: TABS, currentTab: () => app.current, goto: (t, x) => app.goto(t, x) });
 *   palette.open("softmax"); palette.showTerm("rope"); palette.close();
 *
 * Tastiera: ↑ ↓ scelgono, Invio apre, Maiusc+Invio copia il link, Esc chiude (dalla scheda di un
 * termine torna ai risultati). Il fuoco resta nel campo di testo: i risultati sono un listbox
 * con aria-activedescendant, quindi il lettore di schermo annuncia la voce attiva.
 */
import { el } from "../dom.js";
import { glossarySearchIndex, glossaryTerm } from "../glossary.js";
import { progress } from "../progress.js";
import { withRepKeywords } from "../repLinks.js";
import { KINDS, excerpt, formatHash, groupResults, highlightRanges, prepareEntry, queryTokens, search } from "../search.js";
import { copyLink } from "./share.js";

const GLOSSARY_GROUP = "glossario";
const RECENT_KEY = "gpt-lesson.search.recent";
const RECENT_MAX = 6;
const PER_GROUP = 5;
const VALID_KINDS = new Set(["step", "action", "view"]);

const FILTERS = [
  { id: "all", label: "Tutto" },
  { id: "step", label: "Passi" },
  { id: "action", label: "Azioni" },
  { id: "view", label: "Viste" },
  { id: "term", label: "Glossario" },
];

export class SearchPalette {
  /**
   * @param {object} options
   * @param {{id: string, title: string, search?: () => Promise<{searchIndex?: object[]}>}[]} options.tabs
   * @param {() => string|null} options.currentTab id del tab visibile (i suoi risultati salgono un po')
   * @param {(tab: string, target: {kind: string, id: string}|null) => Promise<unknown>} options.goto
   */
  constructor({ tabs, currentTab = () => null, goto }) {
    this.tabs = tabs;
    this.currentTab = currentTab;
    this.goto = goto;
    this.filter = "all";
    this.expanded = new Set();
    this.options = []; // [{node, run, copy?}] nell'ordine in cui appaiono
    this.active = -1;
    this.term = null; // id del termine mostrato nella scheda (null = elenco dei risultati)
    this.index = null;
    this.failedIndexes = []; // tab il cui search.js manca o non si carica (per il debug)
    this.build();
  }

  get isOpen() {
    return !this.backdrop.hidden;
  }

  // --- apertura e chiusura -------------------------------------------------------------------

  /** Apre la palette (con un testo già scritto, se dato) e mette il fuoco nel campo di ricerca. */
  open(query) {
    if (!this.isOpen) {
      this.returnFocus = document.activeElement;
      this.backdrop.hidden = false;
      document.body.classList.add("palette-open");
    }
    if (typeof query === "string") this.input.value = query;
    this.term = null;
    this.input.focus({ preventScroll: true });
    this.input.select();
    this.render();
    this.loadIndex().then(() => this.isOpen && !this.term && this.render());
  }

  close({ restoreFocus = true } = {}) {
    if (!this.isOpen) return;
    this.backdrop.hidden = true;
    document.body.classList.remove("palette-open");
    this.term = null;
    const back = this.returnFocus;
    this.returnFocus = null;
    // Il fuoco non deve restare nel campo nascosto: i tasti finirebbero ancora alla palette.
    if (this.dialog.contains(document.activeElement)) document.activeElement.blur();
    if (restoreFocus && back?.isConnected && back !== document.body) back.focus?.({ preventScroll: true });
  }

  toggle() {
    if (this.isOpen) this.close();
    else this.open();
  }

  /** Mostra la scheda di un termine del glossario (apre la palette se è chiusa). */
  showTerm(id) {
    const term = glossaryTerm(id);
    if (!term) return false;
    if (!this.isOpen) this.open();
    this.term = id;
    this.remember({ tab: GLOSSARY_GROUP, kind: "term", id });
    this.renderTerm(term);
    return true;
  }

  // --- indice ----------------------------------------------------------------------------------

  /** Carica (una volta sola) gli indici di tutti i tab, il glossario e i tab stessi. */
  loadIndex() {
    this.indexPromise ??= (async () => {
      const entries = [];
      for (const tab of this.tabs) {
        entries.push({ id: tab.id, kind: "tab", tab: tab.id, group: tab.id, title: tab.title, text: tab.description ?? "", keywords: tab.keywords ?? [] });
      }
      const loaded = await Promise.all(
        this.tabs.map(async (tab) => {
          try {
            const module = await tab.search?.();
            return { tab, items: Array.isArray(module?.searchIndex) ? module.searchIndex : [] };
          } catch {
            this.failedIndexes.push(tab.id);
            return { tab, items: [] };
          }
        }),
      );
      for (const { tab, items } of loaded) {
        for (const item of items) {
          if (!item || typeof item.title !== "string" || item.id === undefined || !VALID_KINDS.has(item.kind)) continue;
          // view:rete e view:cervello: anche i nomi dei comandi comuni delle due viste (repLinks.js)
          entries.push({ ...item, id: String(item.id), keywords: withRepKeywords(item), tab: tab.id, group: tab.id });
        }
      }
      for (const item of glossarySearchIndex()) entries.push({ ...item, tab: GLOSSARY_GROUP, group: GLOSSARY_GROUP });
      this.index = entries.map(prepareEntry);
      return this.index;
    })();
    return this.indexPromise;
  }

  // --- costruzione del DOM -------------------------------------------------------------------

  build() {
    this.input = el("input", {
      type: "search",
      className: "palette-input",
      placeholder: "Cerca passi, azioni, viste e termini…",
      autocomplete: "off",
      spellcheck: false,
      enterKeyHint: "go",
      "aria-label": "Cerca nella lezione",
      role: "combobox",
      "aria-autocomplete": "list",
      "aria-expanded": "true",
      "aria-controls": "palette-list",
      "aria-keyshortcuts": "ArrowUp ArrowDown Enter Shift+Enter Escape",
    });
    this.input.addEventListener("input", () => {
      this.term = null;
      this.expanded.clear();
      this.render();
    });
    this.filterButtons = FILTERS.map(({ id, label }) =>
      el("button", { type: "button", className: "palette-filter", dataset: { filter: id }, "aria-pressed": String(id === this.filter), onclick: () => this.setFilter(id) }, label),
    );
    this.list = el("div", { className: "palette-list", id: "palette-list", role: "listbox", "aria-label": "Risultati" });
    this.card = el("div", { className: "palette-term-view", hidden: true });
    this.status = el("span", { className: "palette-status", role: "status", "aria-live": "polite" });
    const closeButton = el("button", { type: "button", className: "palette-close", "aria-label": "Chiudi la ricerca", onclick: () => this.close() }, el("span", { className: "palette-close-key" }, "Esc"), el("span", { className: "palette-close-x", "aria-hidden": "true" }, "✕"));
    this.dialog = el(
      "div",
      { className: "palette", role: "dialog", "aria-modal": "true", "aria-label": "Ricerca nella lezione" },
      el("div", { className: "palette-head" }, searchIcon(), this.input, closeButton),
      el("div", { className: "palette-filters", role: "group", "aria-label": "Filtra per tipo" }, ...this.filterButtons),
      el("div", { className: "palette-body" }, this.list, this.card),
      el(
        "div",
        { className: "palette-foot" },
        el("span", { className: "palette-keys" }, kbd("↑"), kbd("↓"), " scegli · ", kbd("Invio"), " apri · ", kbd("Maiusc"), "+", kbd("Invio"), " copia link · ", kbd("Esc"), " chiudi"),
        this.status,
      ),
    );
    this.backdrop = el("div", { className: "palette-backdrop", hidden: true }, this.dialog);
    // Clic sullo sfondo (fuori dalla finestra) = chiudi.
    this.backdrop.addEventListener("mousedown", (e) => {
      if (e.target === this.backdrop) this.close();
    });
    // Un clic su una voce non deve togliere il fuoco al campo di testo.
    this.list.addEventListener("mousedown", (e) => {
      if (!e.target.closest("button")) e.preventDefault();
    });
    this.dialog.addEventListener("keydown", (e) => this.onKey(e));
    document.body.append(this.backdrop);
  }

  setFilter(id) {
    this.filter = id;
    for (const b of this.filterButtons) b.setAttribute("aria-pressed", String(b.dataset.filter === id));
    this.term = null;
    this.expanded.clear();
    this.input.focus({ preventScroll: true });
    this.render();
  }

  // --- tastiera ------------------------------------------------------------------------------

  onKey(e) {
    // I tasti dati alla palette non arrivano alla scena (WASD, frecce, Spazio del player…).
    e.stopPropagation();
    if (e.key === "Escape") {
      e.preventDefault();
      if (this.term) this.backToResults();
      else this.close();
      return;
    }
    if (e.key === "Tab") return this.trapFocus(e);
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault(); // la stessa scorciatoia che la apre la chiude
      this.close();
      return;
    }
    if (this.term) {
      if (e.key === "Enter" && e.target === this.input) {
        e.preventDefault();
        this.card.querySelector(".palette-primary")?.click();
      } else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && e.target === this.input) {
        e.preventDefault();
        this.backToResults();
      }
      return;
    }
    if (e.target !== this.input) return; // i pulsanti (filtri, chiudi) gestiscono da soli Invio e Spazio
    const step = { ArrowDown: 1, ArrowUp: -1, PageDown: 5, PageUp: -5 }[e.key];
    if (step) {
      e.preventDefault();
      this.move(step);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const option = this.options[this.active];
      if (!option) return;
      if (e.shiftKey) option.copy?.();
      else option.run();
    }
  }

  move(delta) {
    const n = this.options.length;
    if (!n) return;
    const big = Math.abs(delta) > 1;
    let next = this.active + delta;
    // Le frecce girano in tondo; Pagina su/giù si fermano ai bordi.
    next = big ? Math.max(0, Math.min(n - 1, next)) : (next + n) % n;
    this.setActive(next);
  }

  setActive(i, { scroll = true } = {}) {
    this.options[this.active]?.node.setAttribute("aria-selected", "false");
    this.active = i;
    const option = this.options[i];
    if (!option) {
      this.input.removeAttribute("aria-activedescendant");
      return;
    }
    option.node.setAttribute("aria-selected", "true");
    this.input.setAttribute("aria-activedescendant", option.node.id);
    if (scroll) option.node.scrollIntoView({ block: "nearest" });
  }

  /** Tab e Maiusc+Tab restano dentro la finestra (è modale). */
  trapFocus(e) {
    const focusable = [...this.dialog.querySelectorAll("input, button:not([tabindex='-1'])")].filter((n) => !n.closest("[hidden]") && n.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  // --- elenco dei risultati ------------------------------------------------------------------

  render() {
    this.card.hidden = true;
    this.list.hidden = false;
    this.input.setAttribute("aria-expanded", "true");
    this.list.replaceChildren();
    this.options = [];
    this.active = -1;
    if (!this.index) {
      this.list.append(el("p", { className: "palette-empty" }, "Carico l'indice della lezione…"));
      this.status.textContent = "";
      return;
    }
    const query = this.input.value.trim();
    const tokens = queryTokens(query);
    if (!tokens.length) this.renderBrowse();
    else this.renderResults(query, tokens);
    if (!this.options.length && tokens.length) {
      this.list.append(
        el(
          "div",
          { className: "palette-empty" },
          el("p", {}, `Nessun risultato per «${query}».`),
          el("p", { className: "hint" }, this.filter === "all" ? "Prova con una parola sola o più corta (\"embed\" trova anche \"embedding\")." : "Prova a cercare in tutti i tipi: scegli \"Tutto\"."),
        ),
      );
    }
    this.setActive(this.options.length ? 0 : -1, { scroll: false });
    this.list.scrollTop = 0;
  }

  renderResults(query, tokens) {
    const current = this.currentTab();
    const filter = this.filter === "all" ? null : (e) => e.kind === this.filter;
    const results = search(this.index, query, { filter, boost: (e) => (e.tab === current ? 1.5 : 0), limit: 80 });
    for (const { group, results: items } of groupResults(results)) this.renderGroup(group, items, tokens);
    this.status.textContent = results.length === 1 ? "1 risultato" : `${results.length} risultati`;
  }

  /** Senza testo: recenti e tab (filtro "Tutto"), altrimenti tutte le voci del tipo scelto. */
  renderBrowse() {
    if (this.filter === "all") {
      const recent = this.recent()
        .map((r) => this.index.find((e) => e.kind === r.kind && e.id === r.id && (e.tab === r.tab || e.kind === "term")))
        .filter(Boolean);
      if (recent.length) this.renderGroup("recenti", recent.map((entry) => ({ entry })), [], { title: "Aperti di recente", limit: RECENT_MAX });
      const tabs = this.index.filter((e) => e.kind === "tab").map((entry) => ({ entry }));
      this.renderGroup("tabs", tabs, [], { title: "Tab della lezione", limit: tabs.length });
      this.list.append(
        el(
          "p",
          { className: "palette-tip" },
          "Scrivi una parola: cerca nei passi del Percorso, nelle azioni animate, nelle viste e nel glossario di tutti i tab, anche quelli non ancora aperti. Accenti e maiuscole non contano.",
        ),
      );
      this.status.textContent = `${this.index.length} voci nell'indice`;
      return;
    }
    const entries = this.index.filter((e) => e.kind === this.filter);
    if (this.filter === "term") entries.sort((a, b) => a.title.localeCompare(b.title, "it"));
    for (const { group, results } of groupResults(entries.map((entry) => ({ entry })))) {
      this.renderGroup(group, results, [], { limit: this.filter === "term" ? Infinity : PER_GROUP });
    }
    this.status.textContent = `${entries.length} voci`;
  }

  renderGroup(group, items, tokens, { title = this.groupTitle(group), limit = PER_GROUP } = {}) {
    if (!items.length) return;
    const headId = `palette-group-${group}`;
    const box = el("div", { className: "palette-group", role: "group", "aria-labelledby": headId });
    const count = items.length > 1 ? ` · ${items.length}` : "";
    const here = group === this.currentTab() ? " (sei qui)" : "";
    box.append(el("div", { className: "palette-group-title", id: headId, role: "presentation" }, title + here + count));
    const shown = this.expanded.has(group) ? items : items.slice(0, limit);
    for (const { entry } of shown) box.append(this.renderOption(entry, tokens));
    const hidden = items.length - shown.length;
    if (hidden > 0) {
      const position = this.options.length;
      const label = el("span", { className: "palette-title" }, hidden === 1 ? `Mostra un altro risultato in ${title}` : `Mostra altri ${hidden} risultati in ${title}`);
      const more = el("div", { className: "palette-option palette-more" }, el("span", { className: "palette-main" }, label));
      box.append(
        this.addOption(more, {
          run: () => {
            this.expanded.add(group);
            this.render();
            this.setActive(Math.min(position, this.options.length - 1));
          },
        }),
      );
    }
    this.list.append(box);
  }

  groupTitle(group) {
    if (group === GLOSSARY_GROUP) return "Glossario";
    return this.tabs.find((t) => t.id === group)?.title ?? group;
  }

  renderOption(entry, tokens) {
    const kindLabel = entry.kind === "tab" ? "Tab" : KINDS[entry.kind] ?? entry.kind;
    const title = el("span", { className: "palette-title" }, marked(entry.title, tokens));
    const main = el("span", { className: "palette-main" }, title);
    const detail = this.detailText(entry, tokens);
    if (detail) main.append(el("span", { className: "palette-snippet" }, marked(detail, tokens)));
    const node = el("div", { className: "palette-option" }, el("span", { className: `palette-kind kind-${entry.kind}` }, kindLabel), main);
    if (entry.kind === "step" && progress.visited(entry.tab).has(entry.id)) {
      node.append(el("span", { className: "palette-visited", role: "img", "aria-label": "già visto", title: "Già visto" }, "✓"));
    }
    const hash = this.hashOf(entry);
    const copy = () => copyLink(hash);
    const copyButton = el("button", { type: "button", className: "palette-copy", tabIndex: -1, title: "Copia il link (Maiusc+Invio)", "aria-label": `Copia il link a ${entry.title}` }, linkIcon());
    copyButton.addEventListener("click", (e) => {
      e.stopPropagation();
      copy();
    });
    node.append(copyButton);
    return this.addOption(node, { run: () => this.choose(entry), copy });
  }

  /** Riga secondaria: per i tab i progressi, per le altre voci un estratto del testo. */
  detailText(entry, tokens) {
    if (entry.kind === "tab") {
      const steps = this.index.filter((e) => e.tab === entry.tab && e.kind === "step").map((e) => e.id);
      const parts = [];
      if (steps.length) parts.push(`${progress.count(entry.tab, steps)}/${steps.length} passi visti`);
      if (entry.text) parts.push(entry.text);
      return parts.join(" · ");
    }
    return excerpt(entry.text, tokens, 110);
  }

  addOption(node, handlers) {
    const index = this.options.length;
    node.id = `palette-opt-${index}`;
    node.setAttribute("role", "option");
    node.setAttribute("aria-selected", "false");
    node.addEventListener("click", () => handlers.run());
    node.addEventListener("mousemove", () => {
      if (this.active !== index) this.setActive(index, { scroll: false });
    });
    this.options.push({ node, ...handlers });
    return node;
  }

  hashOf(entry) {
    if (entry.kind === "term") {
      const term = glossaryTerm(entry.id);
      return formatHash(term?.tab ?? this.currentTab() ?? this.tabs[0].id, { kind: "term", id: entry.id });
    }
    return formatHash(entry.tab, entry.kind === "tab" ? null : { kind: entry.kind, id: entry.id });
  }

  choose(entry) {
    if (entry.kind === "term") {
      this.showTerm(entry.id);
      return;
    }
    this.remember({ tab: entry.tab, kind: entry.kind, id: entry.id });
    this.close({ restoreFocus: false });
    this.goto(entry.tab, entry.kind === "tab" ? null : { kind: entry.kind, id: entry.id });
  }

  // --- scheda di un termine ------------------------------------------------------------------

  renderTerm(term) {
    this.list.hidden = true;
    this.card.hidden = false;
    this.options = [];
    this.active = -1;
    this.input.removeAttribute("aria-activedescendant");
    this.input.setAttribute("aria-expanded", "false");
    const tabTitle = this.groupTitle(term.tab);
    const related = (term.related ?? []).map(glossaryTerm).filter(Boolean);
    const card = el(
      "article",
      { className: "palette-term", "aria-labelledby": "palette-term-title" },
      el("button", { type: "button", className: "palette-back", onclick: () => this.backToResults() }, "← Torna ai risultati"),
      el("span", { className: "palette-kind kind-term" }, "Termine"),
      el("h2", { id: "palette-term-title" }, term.term),
      el("p", { className: "palette-term-def" }, term.definition),
      el("div", { className: "palette-term-analogy" }, el("strong", {}, "Come immaginarlo"), el("p", {}, term.analogy)),
      related.length
        ? el(
            "div",
            { className: "palette-term-related" },
            el("span", { className: "muted" }, "Vedi anche: "),
            ...related.map((r) => el("button", { type: "button", className: "palette-chip", onclick: () => this.showTerm(r.id) }, r.term)),
          )
        : null,
      el(
        "div",
        { className: "palette-term-actions" },
        el(
          "button",
          {
            type: "button",
            className: "palette-primary",
            onclick: () => {
              this.close({ restoreFocus: false });
              this.goto(term.tab, term.go ?? null);
            },
          },
          `Guarda nel tab ${tabTitle} →`,
        ),
        el("button", { type: "button", onclick: () => copyLink(formatHash(term.tab, { kind: "term", id: term.id }), { label: "Link al termine copiato negli appunti" }) }, linkIcon(), " Copia link"),
      ),
    );
    this.card.replaceChildren(card);
    this.card.scrollTop = 0;
    this.status.textContent = `Termine: ${term.term}`;
    this.input.focus({ preventScroll: true });
  }

  backToResults() {
    this.term = null;
    this.render();
    this.input.focus({ preventScroll: true });
  }

  // --- recenti (localStorage, con try/catch: può non esserci) --------------------------------

  recent() {
    try {
      const list = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]");
      return Array.isArray(list) ? list.filter((r) => r && typeof r.id === "string" && typeof r.kind === "string") : [];
    } catch {
      return [];
    }
  }

  remember(item) {
    const list = [item, ...this.recent().filter((r) => !(r.kind === item.kind && r.id === item.id && r.tab === item.tab))].slice(0, RECENT_MAX);
    try {
      localStorage.setItem(RECENT_KEY, JSON.stringify(list));
    } catch {
      // senza localStorage i recenti semplicemente non vengono ricordati
    }
  }
}

// --- piccoli elementi grafici ----------------------------------------------------------------

/** Testo con le parti trovate dentro <mark> (solo nodi di testo: nessun HTML interpretato). */
function marked(text, tokens) {
  const fragment = document.createDocumentFragment();
  let at = 0;
  for (const [start, end] of highlightRanges(text, tokens)) {
    if (start > at) fragment.append(text.slice(at, start));
    fragment.append(el("mark", {}, text.slice(start, end)));
    at = end;
  }
  if (at < text.length) fragment.append(text.slice(at));
  return fragment;
}

function kbd(text) {
  return el("kbd", {}, text);
}

function svgIcon(path, className) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", className);
  svg.innerHTML = path;
  return svg;
}

export function searchIcon(className = "palette-icon") {
  return svgIcon('<circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="m20 20-3.5-3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>', className);
}

function linkIcon() {
  return svgIcon(
    '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    "icon",
  );
}
