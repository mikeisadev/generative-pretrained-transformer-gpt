/**
 * Selettore delle rappresentazioni di un tab: "Architettura · Rete neurale · Cervello".
 *
 * Un radiogroup accessibile (frecce, Home/Fine, focus visibile, suggerimento con la descrizione) che
 * sta in alto nello stage: nella prima fila dei comandi se c'è posto, altrimenti nella seconda; con
 * icone ed etichette, solo con le icone (l'etichetta resta per gli screen reader e nel suggerimento)
 * o, se non c'è proprio spazio, come menu compatto (un pulsante che apre l'elenco). Gli altri comandi
 * del tab (☰, selettore dello zoom, barra strumenti, "Grafici", codice, grafici aperti, player, pad)
 * sono ostacoli da non coprire. Sul telefono, con il cassetto della sidebar aperto (copre la scena),
 * il selettore sale in cima al cassetto: resta sempre raggiungibile.
 *
 * Il posto si ricalcola solo a eventi di layout (ridimensionamenti, pannelli, sidebar), mai per frame.
 * La scelta del posto è pura (freeIntervals, candidateRows, placeSwitch: tests/reps.test.mjs).
 * Il resto (che cosa succede scegliendo una voce) è di js/core/representations.js.
 */
import { el } from "../dom.js";

/** Varianti in linea del selettore, dalla più ricca alla più compatta. */
export const VARIANTS = ["full", "icons", "menu"];

/**
 * Classe della radice del tab mentre una vista alternativa copre la scena (la mette representations.js,
 * reps.css nasconde i comandi del tab con visibility): quei comandi restano ostacoli, così il selettore
 * non salta da un posto all'altro proprio mentre lo studente lo usa.
 */
export const ALT_ACTIVE_CLASS = "rep-alt-active";

/**
 * Ordine di preferenza [variante, fila]: prima con le etichette nella fila dei comandi in alto, poi
 * solo icone nella stessa fila (il selettore non salta di fila quando un pannello si apre), poi la
 * seconda fila, infine il menu compatto.
 */
const PREFERENCE = [
  ["full", 0],
  ["icons", 0],
  ["full", 1],
  ["icons", 1],
  ["menu", 0],
  ["menu", 1],
  ["icons", 2],
  ["menu", 2],
];

const EDGE = 10; // distanza dai bordi dello stage (come ☰ e il selettore dello zoom)
const GAP = 8; // spazio minimo da ogni altro comando
const FIRST_ROW = 10; // la prima fila è allineata a ☰
const ROW_OBSTACLE_MAX = 120; // più alto di così è un pannello laterale, non un comando della fila
const SHEET_FRACTION = 0.9; // largo almeno così è un foglio a tutto stage, non un comando
const KEY_SETTLE_MS = 160; // frecce: la vista cambia quando lo studente si ferma su una voce

/**
 * Intervalli orizzontali liberi della fascia [top, top + height] dello stage (coordinate dello stage),
 * tolti gli ostacoli allargati di `gap`. Restituisce [{ start, end }] ordinati.
 * @param {{width: number, top: number, height: number, obstacles?: {left, right, top, bottom}[], gap?: number, edge?: number}} o
 */
export function freeIntervals({ width, top, height, obstacles = [], gap = GAP, edge = EDGE }) {
  let free = [{ start: edge, end: width - edge }];
  for (const o of obstacles) {
    if (o.bottom <= top - gap || o.top >= top + height + gap) continue; // non tocca la fascia
    const cutStart = o.left - gap;
    const cutEnd = o.right + gap;
    const next = [];
    for (const f of free) {
      if (cutEnd <= f.start || cutStart >= f.end) {
        next.push(f);
        continue;
      }
      if (cutStart > f.start) next.push({ start: f.start, end: cutStart });
      if (cutEnd < f.end) next.push({ start: cutEnd, end: f.end });
    }
    free = next;
  }
  return free.filter((f) => f.end > f.start);
}

/**
 * Le file candidate (il loro `top`): la prima allineata a ☰, ognuna delle successive subito sotto i
 * comandi che toccano la precedente. Gli ostacoli alti (codice, grafici, cassetto) non fanno fila:
 * bloccano solo un tratto orizzontale.
 */
export function candidateRows({ obstacles = [], height, first = FIRST_ROW, gap = GAP, count = 3, rowObstacleMax = ROW_OBSTACLE_MAX }) {
  const small = obstacles.filter((o) => o.bottom - o.top <= rowObstacleMax);
  const rows = [first];
  while (rows.length < count) {
    const top = rows[rows.length - 1];
    let bottom = top + height;
    for (const o of small) if (o.top < top + height + gap && o.bottom > top - gap) bottom = Math.max(bottom, o.bottom);
    rows.push(bottom + gap);
  }
  return rows;
}

/**
 * Dove mettere il selettore. Nella prima fila si preferisce il centro della parte visibile dello stage
 * (`center`); nelle altre il bordo sinistro, sotto ☰, come il selettore dello zoom sul telefono.
 * Gli ostacoli con `soft: true` sono comandi del tab nascosti da una vista alternativa: contano (il
 * selettore non salta mentre lo si usa), ma se così non c'è posto i pannelli alti tra loro (codice,
 * grafici) liberano la loro area. Se nessun posto è libero resta il menu compatto nella seconda fila
 * (meglio che sparire), con fits: false.
 * @param {object} o
 * @param {number} o.width larghezza dello stage (px)
 * @param {number} o.stageHeight altezza dello stage (px): una fila che non ci sta non vale
 * @param {{full: number, icons: number, menu: number}} o.sizes larghezza di ogni variante (px)
 * @param {number} o.height altezza del selettore (px)
 * @param {{left, right, top, bottom, soft?}[]} [o.obstacles] comandi da non coprire (coordinate dello stage)
 * @param {number} [o.center] ascissa preferita per il centro del selettore nella prima fila
 * @returns {{variant: "full"|"icons"|"menu", row: number, top: number, left: number, fits: boolean}}
 */
export function placeSwitch({ width, stageHeight = Infinity, sizes, height, obstacles = [], center = width / 2, gap = GAP, edge = EDGE, first = FIRST_ROW }) {
  // Un foglio largo quanto lo stage (i grafici sul telefono) non è un comando da evitare: copre tutto,
  // selettore compreso, come gli altri comandi; il selettore resta al suo posto e lì lo si ritrova.
  const all = obstacles.filter((o) => o.right - o.left < SHEET_FRACTION * width);
  const options = { width, stageHeight, sizes, height, center, gap, edge, first };
  const relaxed = all.filter((o) => !o.soft || o.bottom - o.top <= ROW_OBSTACLE_MAX);
  return (
    firstFreeSpot({ ...options, obstacles: all }) ??
    (relaxed.length < all.length ? firstFreeSpot({ ...options, obstacles: relaxed }) : null) ?? {
      variant: "menu",
      row: 1,
      top: candidateRows({ obstacles: all, height, first, gap })[1],
      left: edge,
      fits: false,
    }
  );
}

/** Il primo posto libero nell'ordine di PREFERENCE, o null. */
function firstFreeSpot({ width, stageHeight, sizes, height, obstacles, center, gap, edge, first }) {
  const rows = candidateRows({ obstacles, height, first, gap });
  for (const [variant, row] of PREFERENCE) {
    const w = sizes[variant];
    const top = rows[row];
    if (!(w > 0) || top + height > stageHeight - edge) continue;
    const spans = freeIntervals({ width, top, height, obstacles, gap, edge }).filter((f) => f.end - f.start >= w);
    if (!spans.length) continue;
    if (row > 0) return { variant, row, top, left: spans[0].start, fits: true };
    // prima fila: il punto più vicino al centro voluto, dentro un tratto libero
    const want = center - w / 2;
    let best = null;
    for (const f of spans) {
      const left = Math.min(Math.max(want, f.start), f.end - w);
      if (!best || Math.abs(left - want) < Math.abs(best.left - want)) best = { variant, row, top, left, fits: true };
    }
    return best;
  }
  return null;
}

/** Icone SVG (stroke = colore del testo). Stringhe costanti, mai dati esterni. */
const ICONS = {
  // schema a blocchi: un componente che ne alimenta due
  architecture: '<rect x="7" y="3" width="10" height="6" rx="1.5"/><rect x="2.5" y="15" width="8" height="6" rx="1.5"/><rect x="13.5" y="15" width="8" height="6" rx="1.5"/><path d="M12 9v3M6.5 15v-3h11v3"/>',
  // neuroni in colonne collegati da sinistra a destra
  network:
    '<circle cx="4" cy="8" r="2"/><circle cx="4" cy="16" r="2"/><circle cx="12" cy="4.6" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="19.4" r="2"/><circle cx="20" cy="12" r="2"/><path d="M5.9 7.3l4.2-1.9M5.9 8.9l4.2 2.2M5.9 15.1l4.2-2.2M5.9 16.7l4.2 1.9M13.5 6.2l5 4.3M14 12h4M13.5 17.8l5-4.3"/>',
  // cervello visto dall'alto: due emisferi, la scissura in mezzo e qualche giro
  brain:
    '<path d="M12 4.6C10.9 3.4 8.6 3.4 7.6 4.9 5.8 4.9 4.5 6.5 4.8 8.2 3.4 9 3 11 3.9 12.4 3.2 13.9 3.7 15.9 5.3 16.6 5.4 18.5 7.3 19.9 9.2 19.4 10 20.4 11.3 20.4 12 19.5M12 4.6C13.1 3.4 15.4 3.4 16.4 4.9 18.2 4.9 19.5 6.5 19.2 8.2 20.6 9 21 11 20.1 12.4 20.8 13.9 20.3 15.9 18.7 16.6 18.6 18.5 16.7 19.9 14.8 19.4 14 20.4 12.7 20.4 12 19.5M12 4.6v14.9"/><path d="M7.9 8.7c1.2.1 2 .9 2.1 2.1M16.1 8.7c-1.2.1-2 .9-2.1 2.1M7.4 14.2c1.1-.6 2.4-.4 3.2.6M16.6 14.2c-1.1-.6-2.4-.4-3.2.6"/>',
  chevron: '<path d="M7 10l5 5 5-5"/>',
};

function icon(name, className = "rep-icon") {
  const span = el("span", { className, ariaHidden: "true" });
  span.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] ?? ""}</svg>`;
  return span;
}

/** Figli diretti dello stage o della radice che non sono comandi da evitare. */
const NOT_OBSTACLES = "canvas, .rep-layer, .rep-loading, .rep-switch, .nav-hud, .nav-help, .tab-stage, .caption, script, style, template";
let switchCount = 0;

export class RepSwitch {
  /**
   * @param {object} o
   * @param {HTMLElement} o.root radice del tab
   * @param {HTMLElement} o.stage stage del tab (il selettore ci sta sopra, in alto)
   * @param {HTMLElement | null} [o.sidebar] la sidebar (sul telefono, a cassetto aperto, il selettore va lì)
   * @param {{id: string, label: string, description: string, icon: string}[]} o.options voci disponibili
   * @param {string} o.current voce scelta all'inizio
   * @param {(id: string) => void} o.onPick lo studente ha scelto una voce
   */
  constructor({ root, stage, sidebar = null, options, current, onPick }) {
    this.root = root;
    this.stage = stage;
    this.sidebar = sidebar;
    this.options = options;
    this.onPick = onPick;
    this.current = current;
    this.variant = "full";
    this.open = false;
    this.disposers = [];
    this.narrow = window.matchMedia("(max-width: 900px)");
    this.coarse = window.matchMedia("(pointer: coarse)");
    const uid = `rep-switch-${++switchCount}`;

    this.buttons = new Map(
      options.map((o) => {
        const desc = el("span", { className: "rep-desc", id: `${uid}-${o.id}-desc` }, o.description);
        const button = el(
          "button",
          { type: "button", className: "rep-option", role: "radio", dataset: { rep: o.id }, onclick: () => this.choose(o.id) },
          icon(o.icon),
          el("span", { className: "rep-text" }, el("span", { className: "rep-label" }, o.label), desc),
        );
        button.setAttribute("aria-describedby", desc.id);
        this.hoverTip(button, o);
        return [o.id, button];
      }),
    );
    this.group = el("div", { className: "rep-options", id: `${uid}-options`, role: "radiogroup", ariaLabel: "Rappresentazione" }, ...this.buttons.values());
    this.group.addEventListener("keydown", (e) => this.onKey(e));
    this.triggerIcon = el("span", { className: "rep-trigger-icon" });
    this.trigger = el(
      "button",
      { type: "button", className: "rep-trigger", onclick: () => this.setOpen(!this.open, { focus: true }) },
      this.triggerIcon,
      icon("chevron", "rep-chevron"),
    );
    this.trigger.setAttribute("aria-haspopup", "menu");
    this.trigger.setAttribute("aria-expanded", "false");
    this.trigger.setAttribute("aria-controls", this.group.id);
    this.trigger.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault(); // la freccia apre il menu: non arriva al Percorso (stepKeys) né alla scena
      this.setOpen(true, { focus: true });
    });
    this.tip = el("div", { className: "rep-tip", ariaHidden: "true" });
    this.element = el("div", { className: "rep-switch", dataset: { variant: this.variant } }, this.trigger, this.group, this.tip);
    this.element.addEventListener("keydown", (e) => {
      // Esc chiude il menu compatto (e non arriva alla navigazione). Spazio e Invio su una voce sono
      // della voce: non devono mettere in pausa l'azione (Spazio di stepKeys), come sul pad.
      if (e.key === "Escape" && this.open) {
        e.stopPropagation();
        this.setOpen(false, { focus: true });
      } else if ((e.key === " " || e.key === "Enter") && e.target.closest?.("button")) e.stopPropagation();
    });
    stage.append(this.element);
    this.setCurrent(current);
    this.watch();
    this.scheduleLayout();
  }

  // ---------------------------------------------------------------- stato
  /** La voce mostrata (anche quando a cambiarla è il codice, non lo studente). */
  setCurrent(id) {
    this.current = id;
    for (const [rep, b] of this.buttons) {
      const on = rep === id;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1; // tabindex mobile: Tab entra sulla voce scelta, le frecce girano
    }
    const option = this.options.find((o) => o.id === id);
    this.triggerIcon.replaceChildren(icon(option?.icon ?? "architecture"));
    this.trigger.setAttribute("aria-label", `Rappresentazione: ${option?.label ?? ""}`);
    this.trigger.title = `Rappresentazione: ${option?.label ?? ""}`;
  }

  /** Voce in preparazione (la vista si sta caricando): un segno discreto, letto anche come "occupato". */
  setBusy(id) {
    const busy = (node, on) => (on ? node.setAttribute("aria-busy", "true") : node.removeAttribute("aria-busy"));
    for (const [rep, b] of this.buttons) busy(b, rep === id);
    busy(this.trigger, Boolean(id));
  }

  /** Porta il focus sulla voce scelta (es. quando la vista che aveva il focus viene chiusa). */
  focusCurrent() {
    const target = this.variant === "menu" && !this.open ? this.trigger : this.buttons.get(this.current);
    target?.focus({ preventScroll: true });
  }

  choose(id, { fromKeys = false } = {}) {
    clearTimeout(this.keyTimer);
    if (this.variant === "menu") this.setOpen(false, { focus: true });
    if (id === this.current && !fromKeys) return;
    this.setCurrent(id);
    if (!fromKeys) {
      this.onPick(id);
      return;
    }
    // Con le frecce lo studente passa sopra le voci: la vista cambia quando si ferma (una sola apertura).
    this.keyTimer = setTimeout(() => this.onPick(this.current), KEY_SETTLE_MS);
  }

  onKey(e) {
    const ids = [...this.buttons.keys()];
    const index = ids.indexOf(e.target.closest?.(".rep-option")?.dataset.rep);
    if (index < 0) return;
    let next = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (index + 1) % ids.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (index - 1 + ids.length) % ids.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = ids.length - 1;
    else if (this.variant === "menu" && e.key === "Tab") this.setOpen(false);
    if (next === null) return;
    e.preventDefault(); // niente scorrimento della pagina, niente passo del Percorso
    const button = this.buttons.get(ids[next]);
    button.focus();
    // Nel menu compatto le frecce spostano solo il focus (Invio sceglie); in linea, come ogni
    // radiogroup, la scelta segue il focus.
    if (this.variant !== "menu") this.choose(ids[next], { fromKeys: true });
  }

  setOpen(open, { focus = false } = {}) {
    open = Boolean(open) && this.variant === "menu";
    if (open === this.open) return;
    this.open = open;
    this.element.classList.toggle("open", open);
    this.trigger.setAttribute("aria-expanded", String(open));
    this.hideTip();
    if (open) {
      this.buttons.get(this.current)?.focus({ preventScroll: true });
      const outside = (e) => !this.element.contains(e.target) && this.setOpen(false);
      document.addEventListener("pointerdown", outside, true);
      this.closeOutside = () => document.removeEventListener("pointerdown", outside, true);
    } else {
      this.closeOutside?.();
      this.closeOutside = null;
      if (focus) this.trigger.focus({ preventScroll: true });
    }
  }

  /** Nel menu compatto il contenitore è un menu di voci a scelta singola; in linea un radiogroup. */
  setVariant(variant) {
    if (variant === this.variant) return;
    const hadFocus = this.element.contains(document.activeElement);
    if (this.open) this.setOpen(false);
    this.variant = variant;
    this.element.dataset.variant = variant;
    const menu = variant === "menu";
    this.group.setAttribute("role", menu ? "menu" : "radiogroup");
    for (const b of this.buttons.values()) b.setAttribute("role", menu ? "menuitemradio" : "radio");
    if (hadFocus) this.focusCurrent();
  }

  // ---------------------------------------------------------------- suggerimento
  /** Etichetta e descrizione della voce sotto il mouse (o col focus da tastiera), sotto il selettore. */
  hoverTip(button, option) {
    const show = () => {
      if (this.variant === "menu" || this.variant === "sidebar") return; // lì la descrizione è già scritta
      this.tip.replaceChildren(el("strong", {}, option.label), el("span", {}, option.description));
      this.tip.style.left = `${Math.max(0, button.offsetLeft - 3)}px`;
      this.tip.classList.add("visible");
    };
    button.addEventListener("pointerenter", (e) => e.pointerType === "mouse" && show());
    button.addEventListener("pointerleave", () => this.hideTip());
    button.addEventListener("focus", () => button.matches(":focus-visible") && show());
    button.addEventListener("blur", () => this.hideTip());
    button.addEventListener("click", () => this.hideTip());
  }

  hideTip() {
    this.tip.classList.remove("visible");
  }

  // ---------------------------------------------------------------- posizionamento
  /** Ascolta solo gli eventi che possono spostare i comandi: ridimensionamenti, classi, `hidden`, figli. */
  watch() {
    const schedule = () => this.scheduleLayout();
    const resize = new ResizeObserver(schedule);
    const attrs = new MutationObserver((records) => {
      for (const r of records) if (r.type === "childList") r.addedNodes.forEach((n) => observe(n));
      schedule();
    });
    const observe = (node) => {
      if (!(node instanceof Element) || node === this.element || node.matches("canvas, script, style")) return;
      resize.observe(node);
      attrs.observe(node, { attributes: true, attributeFilter: ["class", "hidden"] });
    };
    resize.observe(this.stage);
    attrs.observe(this.root, { attributes: true, attributeFilter: ["class", "hidden"], childList: true });
    attrs.observe(this.stage, { childList: true });
    for (const node of [...this.root.children, ...this.stage.children]) observe(node);
    // I pannelli entrano ed escono con una transizione (grafici, sidebar): solo alla fine sono al loro
    // posto, e uno spostamento con transform non avvisa né ResizeObserver né MutationObserver.
    this.root.addEventListener("transitionend", schedule);
    this.disposers.push(() => this.root.removeEventListener("transitionend", schedule));
    const media = () => {
      this.sizes = null;
      schedule();
    };
    this.narrow.addEventListener("change", media);
    this.coarse.addEventListener("change", media);
    document.fonts?.ready?.then(media);
    this.disposers.push(
      () => resize.disconnect(),
      () => attrs.disconnect(),
      () => this.narrow.removeEventListener("change", media),
      () => this.coarse.removeEventListener("change", media),
    );
  }

  scheduleLayout() {
    if (this.raf || this.disposed) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.layout();
    });
  }

  /**
   * Sul telefono a cassetto aperto il selettore sta in cima alla sidebar; altrimenti nello stage, nel
   * primo posto libero (placeSwitch). Scrive anche --rep-safe-top sulla radice del tab: le viste
   * alternative mettono i loro comandi in alto sotto quella riga, senza finire sotto ☰ o il selettore.
   */
  layout() {
    if (this.disposed || this.root.hidden || !this.root.isConnected) return;
    const inDrawer = this.narrow.matches && !this.root.classList.contains("sidebar-collapsed") && this.sidebar;
    this.mount(inDrawer ? "sidebar" : "stage");
    if (inDrawer) return;
    const stageRect = this.stage.getBoundingClientRect();
    if (!stageRect.width || !stageRect.height) return; // tab nascosto
    const sizes = this.measure();
    const obstacles = this.obstacles(stageRect);
    const charts = this.root.querySelector(":scope > .charts-panel.open");
    const chartsLeft = charts ? charts.getBoundingClientRect().left - stageRect.left : stageRect.width;
    const spot = placeSwitch({
      width: stageRect.width,
      stageHeight: stageRect.height,
      sizes,
      height: sizes.height,
      obstacles,
      center: Math.min(stageRect.width, Math.max(0, chartsLeft)) / 2,
    });
    this.setVariant(spot.variant);
    this.element.style.left = `${Math.round(spot.left)}px`;
    this.element.style.top = `${Math.round(spot.top)}px`;
    // il menu compatto si apre verso l'interno dello stage
    this.element.dataset.align = spot.left + sizes[spot.variant] / 2 > stageRect.width / 2 ? "end" : "start";
    const toggle = this.stage.querySelector(":scope > .sidebar-toggle");
    const toggleBottom = toggle ? toggle.getBoundingClientRect().bottom - stageRect.top : 0;
    this.root.style.setProperty("--rep-safe-top", `${Math.round(Math.max(toggleBottom, spot.top + sizes.height) + GAP)}px`);
    this.root.classList.toggle("rep-switch-low", spot.row > 0);
  }

  mount(place) {
    if (this.place === place) return;
    const hadFocus = this.element.contains(document.activeElement) ? document.activeElement : null;
    this.place = place;
    if (place === "sidebar") {
      this.setVariant("sidebar");
      this.element.style.left = "";
      this.element.style.top = "";
      this.sidebar.prepend(this.element);
      this.root.style.removeProperty("--rep-safe-top");
      this.root.classList.remove("rep-switch-low");
    } else {
      this.setVariant("full");
      this.stage.append(this.element);
    }
    hadFocus?.isConnected && hadFocus.focus({ preventScroll: true }); // spostarlo nel DOM toglie il focus
  }

  /** Larghezze delle tre varianti in linea (max-content: non dipendono dal posto) e altezza. */
  measure() {
    const key = `${this.narrow.matches}|${this.coarse.matches}|${document.fonts?.status ?? ""}`;
    if (this.sizes && this.sizesKey === key) return this.sizes;
    const keep = this.element.dataset.variant;
    const sizes = {};
    for (const variant of VARIANTS) {
      this.element.dataset.variant = variant;
      sizes[variant] = this.element.offsetWidth;
      if (variant === "full") sizes.height = this.element.offsetHeight;
    }
    this.element.dataset.variant = keep;
    this.sizes = sizes;
    this.sizesKey = key;
    return sizes;
  }

  /**
   * I comandi da non coprire, in coordinate dello stage. Gli elementi invisibili non contano, tranne
   * quelli del tab nascosti da una vista alternativa (ALT_ACTIVE_CLASS): il selettore resta dov'era.
   * Con una vista alternativa contano anche i suoi comandi (figli del livello): una vista che mette la
   * sua scheda sotto --rep-safe-top non tocca mai la fila del selettore, una che non lo fa non viene coperta.
   */
  obstacles(stageRect) {
    const list = [];
    const alt = this.root.classList.contains(ALT_ACTIVE_CLASS);
    const layer = alt ? this.stage.querySelector(":scope > .rep-layer") : null;
    for (const node of [...this.stage.children, ...this.root.children, ...(layer?.children ?? [])]) {
      if (node.hidden || node.matches(NOT_OBSTACLES)) continue;
      const style = getComputedStyle(node);
      if (style.display === "none" || style.pointerEvents === "none" || style.position === "static") continue;
      if ((!alt || node.parentElement === layer) && style.visibility === "hidden") continue;
      const r = node.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) continue;
      // soft: comando del tab nascosto dalla vista alternativa (non si vede, ma tiene il suo posto)
      const soft = alt && node.parentElement !== layer;
      const o = { left: r.left - stageRect.left, right: r.right - stageRect.left, top: r.top - stageRect.top, bottom: r.bottom - stageRect.top, soft };
      if (o.right <= 0 || o.left >= stageRect.width || o.bottom <= 0 || o.top >= stageRect.height) continue;
      list.push(o);
    }
    return list;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    clearTimeout(this.keyTimer);
    this.closeOutside?.();
    for (const off of this.disposers.splice(0)) off();
    this.element.remove();
    this.root.style.removeProperty("--rep-safe-top");
    this.root.classList.remove("rep-switch-low");
  }
}
