/**
 * Comandi di navigazione a schermo (il "pad"), aiuto, suggerimenti e chip "Camera libera".
 *
 * Il pad sta in basso a destra della parte di scena che si vede: accanto al pannello del codice se
 * questo arriva fin giù, a sinistra dei grafici se sono aperti, sopra la barra del player se la
 * toccherebbe. Le frecce tenute premute muovono in continuo (come i tasti W A S D); un tocco breve
 * fa un piccolo passo. Si riduce a un solo pulsante (sul telefono parte ridotto).
 *
 * La logica del movimento è in js/core/navigation.js: qui solo DOM, posizionamento e testi.
 */
import { el } from "../dom.js";
import { captionPlacement, padPlacement, snapshotName } from "../navMath.js";
import { readPref, writePref } from "../navigation.js";

const PREF_COLLAPSED = "nav.padCollapsed";
const PREF_TIP_SEEN = "nav.tipSeen";
const NARROW = "(max-width: 900px)";
const FAB_ROOM = 56; // spazio a destra della didascalia per il pulsante unico sul telefono (8 + 40 + 8 px)
const HUD_MIN_WIDTH = 360; // in un livello, i chip vanno a destra della scheda della vista se resta almeno tanto
let tipShownThisSession = false; // il suggerimento iniziale compare in un solo tab

/** Icone SVG (stroke = colore del testo). Stringhe costanti, mai dati esterni. */
const ICONS = {
  up: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  down: '<path d="M12 5v14M19 12l-7 7-7-7"/>',
  left: '<path d="M19 12H5M12 19l-7-7 7-7"/>',
  right: '<path d="M5 12h14M12 5l7 7-7 7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  orbitLeft: '<path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1"/><path d="M3 4v5h5"/>',
  orbitRight: '<path d="M20.5 12a8.5 8.5 0 1 1-2.6-6.1"/><path d="M21 4v5h-5"/>',
  center: '<circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/>',
  rotate: '<ellipse cx="12" cy="12" rx="9" ry="4.2"/><path d="M12 3v18"/><path d="M15.5 6.2 12 3 8.5 6.2"/>',
  pan: '<path d="M12 3v18M3 12h18M12 3 9.5 5.5M12 3l2.5 2.5M12 21l-2.5-2.5M12 21l2.5-2.5M3 12l2.5-2.5M3 12l2.5 2.5M21 12l-2.5-2.5M21 12l-2.5 2.5"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 0 1 4.8 1c0 1.7-2.4 2.1-2.4 3.7"/><path d="M12 17.3h.01"/>',
  fullscreen: '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>',
  fullscreenExit: '<path d="M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5"/>',
  camera: '<path d="M4 8h3.2l1.8-2.6h6l1.8 2.6H20v11H4z"/><circle cx="12" cy="13.2" r="3.4"/>',
  collapse: '<path d="M6 9l6 6 6-6"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
};

function icon(name) {
  const span = el("span", { className: "nav-icon", ariaHidden: "true" });
  span.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;
  return span;
}

/** I pulsanti di movimento: disposti 3×3 come una croce direzionale, "Centra" nel mezzo. */
const MOVES = [
  { axis: "orbit", dir: -1, icon: "orbitLeft", label: "Gira attorno verso sinistra", keys: "Q" },
  { axis: "panY", dir: +1, icon: "up", label: "Sposta in su", keys: "W" },
  { axis: "orbit", dir: +1, icon: "orbitRight", label: "Gira attorno verso destra", keys: "E" },
  { axis: "panX", dir: -1, icon: "left", label: "Sposta a sinistra", keys: "A o ←" },
  { center: true, icon: "center", label: "Centra: torna alla vista iniziale", keys: "C" },
  { axis: "panX", dir: +1, icon: "right", label: "Sposta a destra", keys: "D o →" },
  { axis: "zoom", dir: -1, icon: "minus", label: "Allontanati (zoom indietro)", keys: "−" },
  { axis: "panY", dir: -1, icon: "down", label: "Sposta in giù", keys: "S" },
  { axis: "zoom", dir: +1, icon: "plus", label: "Avvicinati (zoom avanti)", keys: "+" },
];

/** Contenuto dell'aiuto: [gesto, effetto]. `touch` = sezione da mostrare per prima sui telefoni. */
const HELP = [
  {
    title: "Mouse e trackpad",
    rows: [
      ["Trascina", "ruota attorno al punto che stai guardando"],
      ["Tasto destro, rotella premuta o Shift + trascina", "sposta la scena"],
      ["Rotella (due dita in verticale sul trackpad, o pizzico)", "zoom verso il puntatore"],
      ["Due dita in orizzontale sul trackpad, o Shift + rotella", "sposta a destra / sinistra"],
      ["Doppio clic su un punto", "lo porta al centro e ti avvicina"],
    ],
  },
  {
    title: "Tastiera",
    rows: [
      ["W A S D  oppure  ← →", "sposta (tieni premuto per continuare)"],
      ["Q E", "gira attorno"],
      ["R F", "guarda più dall'alto / dal basso"],
      ["+ −", "avvicinati / allontanati"],
      ["Shift / Alt", "più veloce / più preciso"],
      ["C oppure Home", "torna alla vista iniziale"],
      ["M", "trascinare: ruota ↔ sposta"],
      ["? oppure H", "apri o chiudi questo aiuto"],
    ],
  },
  {
    title: "Touch",
    touch: true,
    rows: [
      ["Un dito", "ruota (in modalità Sposta: sposta)"],
      ["Due dita", "pizzica per lo zoom, trascina per spostare"],
      ["Doppio tocco", "centra quel punto"],
    ],
  },
  {
    title: "Pulsanti a schermo",
    touch: true,
    rows: [
      ["Ruota / Sposta", "scegli cosa fa il trascinamento (comodo con il solo trackpad)"],
      ["Frecce, ⟲ ⟳, + −", "tieni premuto per muoverti in continuo, tocca per un piccolo passo"],
      ["Mirino al centro", "torna alla vista iniziale"],
      ["Schermo intero, foto", "più spazio alla scena; salva un'immagine PNG"],
    ],
  },
];

export class NavPad {
  /**
   * @param {import("../navigation.js").Navigation} nav
   * @param {object} o
   * @param {HTMLElement} o.root radice del tab
   * @param {HTMLElement} o.stage stage della scena
   * @param {import("./player.js").ActionPlayer | null} [o.player]
   * @param {import("./codePanel.js").CodePanel | null} [o.codePanel]
   * @param {import("./charts.js").ChartsPanel | null} [o.charts]
   * @param {() => number} [o.coveredRight] pixel coperti a destra (codice/grafici)
   */
  constructor(nav, { root, stage, player = null, codePanel = null, charts = null, coveredRight = () => 0 }) {
    this.nav = nav;
    this.root = root;
    this.stage = stage;
    this.player = player;
    this.codePanel = codePanel;
    this.charts = charts;
    this.coveredRight = coveredRight;
    this.narrow = window.matchMedia(NARROW);
    this.disposers = [];
    root.classList.add("has-nav-pad");

    this.buildPad();
    this.buildHud();
    this.buildHelp();
    if (player) this.buildFollowChip();
    this.setCollapsed(this.narrow.matches ? true : readPref(PREF_COLLAPSED, "0") === "1", { persist: false });
    this.wire();
    this.scheduleLayout();
  }

  // ---------------------------------------------------------------- costruzione
  buildPad() {
    this.tip = el("div", { className: "nav-pad-tip", ariaHidden: "true" });
    this.modeButtons = {
      rotate: this.button("rotate", "Trascinare ruota la scena", "M", () => this.nav.setDragMode("rotate"), "nav-mode-btn", "Ruota"),
      pan: this.button("pan", "Trascinare sposta la scena (comodo con il trackpad)", "M", () => this.nav.setDragMode("pan"), "nav-mode-btn", "Sposta"),
    };
    const collapse = this.button("collapse", "Riduci i comandi", "", () => this.setCollapsed(true, { keepFocus: true }), "nav-collapse");
    const grid = el("div", { className: "nav-grid" }, ...MOVES.map((m) => (m.center ? this.button(m.icon, m.label, m.keys, () => this.nav.home(), "nav-center") : this.moveButton(m))));
    this.fullscreenButton = this.button("fullscreen", "Schermo intero", "", () => this.toggleFullscreen());
    this.fullscreenButton.hidden = !fullscreenSupported();
    const tools = el(
      "div",
      { className: "nav-tools" },
      this.button("help", "Tutti i comandi", "?", () => this.toggleHelp()),
      this.fullscreenButton,
      this.button("camera", "Foto della scena (PNG)", "", () => this.takePhoto()),
    );
    this.body = el(
      "div",
      { className: "nav-pad-body" },
      el("div", { className: "nav-pad-head" }, el("div", { className: "nav-mode", role: "group", ariaLabel: "Cosa fa il trascinamento" }, this.modeButtons.rotate, this.modeButtons.pan), collapse),
      el("div", { className: "nav-pad-main" }, grid, tools),
    );
    this.fab = el(
      "button",
      { type: "button", className: "nav-fab", ariaLabel: "Mostra i comandi di navigazione", ariaExpanded: "false", onclick: () => this.expand() },
      icon("pan"),
    );
    this.hoverTip(this.fab, "Comandi di navigazione", "");
    this.pad = el("div", { className: "nav-pad", role: "group", ariaLabel: "Comandi di navigazione" }, this.tip, this.body, this.fab);
    // Spazio e Invio su un pulsante del pad sono per quel pulsante: non devono arrivare al tab (dove
    // Spazio mette in pausa l'azione) né alla navigazione.
    this.pad.addEventListener("keydown", (e) => (e.key === " " || e.key === "Enter") && e.target.closest("button") && e.stopPropagation());
    this.stage.append(this.pad);
    this.syncMode(this.nav.mode);
  }

  /** Pulsante con icona, etichetta accessibile e suggerimento (con il tasto corrispondente). */
  button(iconName, label, keys, onclick, className = "", text = "") {
    const b = el(
      "button",
      { type: "button", className: `nav-btn ${className}`.trim(), ariaLabel: keys ? `${label} (${keys})` : label, onclick },
      icon(iconName),
      text ? el("span", { className: "nav-btn-text" }, text) : null,
    );
    this.hoverTip(b, label, keys);
    return b;
  }

  /** Freccia del pad: tenuta premuta muove in continuo; da tastiera (Invio/Spazio) fa un passo. */
  moveButton({ axis, dir, icon: iconName, label, keys }) {
    const b = this.button(iconName, label, keys, (e) => {
      if (e.detail === 0) this.nav.nudge(axis, dir); // attivato da tastiera: niente pointerdown
    }, "nav-move");
    let release = null;
    const end = () => {
      release?.();
      release = null;
      b.classList.remove("pressed");
    };
    b.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault(); // niente focus né selezione: il dito o il mouse restano liberi
      b.setPointerCapture?.(e.pointerId);
      end();
      release = this.nav.hold(axis, dir);
      b.classList.add("pressed");
    });
    b.addEventListener("pointerup", end);
    b.addEventListener("pointercancel", end);
    b.addEventListener("lostpointercapture", end);
    // Da tastiera (Tab fino al pulsante): Invio o Spazio tenuti premuti muovono in continuo come col mouse.
    b.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault(); // niente click ripetuti a ogni ripetizione del tasto
      if (e.repeat || release) return;
      release = this.nav.hold(axis, dir);
      b.classList.add("pressed");
    });
    b.addEventListener("keyup", (e) => (e.key === "Enter" || e.key === " ") && end());
    b.addEventListener("blur", end);
    b.addEventListener("contextmenu", (e) => e.preventDefault()); // pressione lunga sul telefono
    return b;
  }

  /** Suggerimento sopra il pad per il pulsante sotto il mouse (o col focus da tastiera): testo in data-tip. */
  hoverTip(button, label, keys) {
    button.dataset.tip = keys ? `${label} · ${keys}` : label;
    const show = () => {
      this.tipOwner = button;
      this.tip.textContent = button.dataset.tip;
      this.tip.classList.add("visible");
    };
    const hide = () => {
      if (this.tipOwner === button) this.tipOwner = null;
      this.tip.classList.remove("visible");
    };
    button.addEventListener("pointerenter", (e) => e.pointerType === "mouse" && show());
    button.addEventListener("pointerleave", hide);
    button.addEventListener("focus", () => button.matches(":focus-visible") && show());
    button.addEventListener("blur", hide);
  }

  buildHud() {
    this.lostHint = el(
      "div",
      { className: "nav-chip nav-lost", role: "status", hidden: true },
      el("span", {}, "Ti sei perso?"),
      el("button", { type: "button", className: "nav-chip-action", onclick: () => this.nav.home() }, "Torna al centro", el("kbd", {}, "C")),
    );
    this.toast = el("div", { className: "nav-chip nav-toast", role: "status", hidden: true });
    this.hud = el("div", { className: "nav-hud" }, this.lostHint, this.toast);
    this.stage.append(this.hud);
  }

  buildHelp() {
    this.helpClose = el("button", { type: "button", className: "nav-btn nav-help-close", ariaLabel: "Chiudi l'aiuto (Esc)", onclick: () => this.toggleHelp(false) }, icon("close"));
    const titleId = `nav-help-title-${Math.random().toString(36).slice(2, 8)}`;
    const card = el(
      "div",
      { className: "nav-help-card", role: "dialog", ariaModal: "true", "aria-labelledby": titleId },
      el("header", {}, el("h2", { id: titleId }, "Muoversi nella scena"), this.helpClose),
      el("p", { className: "nav-help-lead" }, "Pensa alla scena come a un plastico sul tavolo: puoi girarci attorno, spostarti di lato, avvicinarti a un dettaglio e tornare alla vista iniziale quando vuoi."),
      el(
        "div",
        { className: "nav-help-grid" },
        ...helpSections().map((section) =>
          el("section", {}, el("h3", {}, section.title), el("dl", {}, ...section.rows.flatMap(([gesture, effect]) => [el("dt", {}, gesture), el("dd", {}, effect)]))),
        ),
      ),
      el("p", { className: "nav-help-foot" }, "Durante un'azione animata puoi guardarti intorno liberamente: la camera smette di seguire l'azione finché non premi \"Segui l'azione\"."),
    );
    this.help = el("div", { className: "nav-help", hidden: true }, card);
    this.help.addEventListener("pointerdown", (e) => e.target === this.help && this.toggleHelp(false));
    this.help.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        this.toggleHelp(false);
      } else if (e.key === "Tab") {
        e.preventDefault(); // finestra modale: il focus resta sul suo unico pulsante
        this.helpClose.focus();
      }
    });
    this.stage.append(this.help);
  }

  /**
   * Una scorciatoia del tab nell'aiuto ("?"), es. `nav.pad.addShortcut({ keys: "V", text: "Vista d'insieme ↔
   * Architettura completa" })`: una riga nella sezione `section` (creata la prima volta, in fondo alla
   * griglia). Testi come testo. Restituisce la funzione che toglie la riga (e la sezione rimasta vuota).
   * @param {{keys: string, text: string, section?: string}} o
   */
  addShortcut({ keys, text, section = "Questo tab" }) {
    this.shortcutSections ??= new Map();
    let block = this.shortcutSections.get(section);
    if (!block) {
      const dl = el("dl", {});
      block = { node: el("section", { className: "nav-help-shortcuts" }, el("h3", {}, section), dl), dl };
      this.help.querySelector(".nav-help-grid").append(block.node);
      this.shortcutSections.set(section, block);
    }
    const row = [el("dt", {}, keys), el("dd", {}, text)];
    block.dl.append(...row);
    return () => {
      for (const node of row) node.remove();
      if (!block.dl.children.length && this.shortcutSections.get(section) === block) {
        block.node.remove();
        this.shortcutSections.delete(section);
      }
    };
  }

  buildFollowChip() {
    this.follow = el(
      "div",
      { className: "nav-chip nav-follow", role: "status", hidden: true },
      el("span", {}, "Camera libera"),
      el("button", { type: "button", className: "nav-chip-action", onclick: () => this.nav.director.resumeFollow() }, "Segui l'azione"),
    );
    this.player.bar.append(this.follow);
  }

  // ---------------------------------------------------------------- collegamenti
  wire() {
    const { nav } = this;
    const sub = (off) => this.disposers.push(off);
    sub(nav.on("mode", (mode) => {
      this.syncMode(mode);
      // conferma visibile anche col pad ridotto (tasto M); non per il cambio arrivato da un altro tab
      if (nav.isActive()) this.showToast(mode === "pan" ? "Trascinare ora sposta la scena · M per tornare a ruotare" : "Trascinare ora ruota la scena · M per spostare");
    }));
    sub(nav.on("help", () => this.toggleHelp()));
    sub(nav.on("escape", () => !this.help.hidden && this.toggleHelp(false)));
    sub(nav.on("lost", (lost) => {
      this.lostHint.hidden = !lost;
      if (lost) this.placeHud();
    }));
    sub(nav.on("input", () => this.onFirstInput()));
    sub(nav.on("dispose", () => this.dispose()));
    sub(nav.director.onFollowChange((following) => {
      if (!this.follow) return;
      this.follow.hidden = following;
      this.scheduleLayout(); // il chip occupa spazio sopra la barra: il pad gli fa posto
    }));
    sub(nav.host.onResize(() => this.scheduleLayout()));
    if (this.player) sub(this.player.on("state", () => this.scheduleLayout()));

    const ro = new ResizeObserver(() => this.scheduleLayout());
    const caption = this.root.querySelector(":scope > .caption");
    for (const node of [this.stage, this.pad, this.player?.bar, this.codePanel?.root, caption]) if (node) ro.observe(node);
    sub(() => ro.disconnect());

    const onNarrow = () => {
      if (this.narrow.matches) this.setCollapsed(true, { persist: false });
      this.scheduleLayout();
    };
    this.narrow.addEventListener("change", onNarrow);
    sub(() => this.narrow.removeEventListener("change", onNarrow));

    const onFullscreen = () => this.syncFullscreen();
    document.addEventListener("fullscreenchange", onFullscreen);
    document.addEventListener("webkitfullscreenchange", onFullscreen);
    sub(() => document.removeEventListener("fullscreenchange", onFullscreen));
    sub(() => document.removeEventListener("webkitfullscreenchange", onFullscreen));

    // Il suggerimento iniziale compare al primo frame disegnato in cui i chip si vedono davvero (il tab è
    // visibile, nessun livello lo copre, il cassetto del telefono è chiuso). Finché non si vedono si riprova
    // ogni mezzo secondo, solo mentre la scena gira; poi la funzione per frame si toglie.
    let nextCheck = 0;
    const offFrame = nav.host.onFrame((dt, elapsed) => {
      if (elapsed < nextCheck) return;
      nextCheck = elapsed + 0.5;
      if (this.maybeShowWelcome() !== "later") offFrame();
    });
    sub(offFrame);
  }

  dispose() {
    for (const off of this.disposers.splice(0)) off();
    cancelAnimationFrame(this.raf);
    clearTimeout(this.toastTimer);
    this.pad.remove();
    this.hud.remove();
    this.help.remove();
    this.follow?.remove();
    this.root.classList.remove("has-nav-pad", "nav-caption-squeezed");
    this.root.style.removeProperty("--nav-caption-right");
    this.root.style.removeProperty("--nav-caption-bottom");
  }

  // ---------------------------------------------------------------- stato
  syncMode(mode) {
    for (const [id, b] of Object.entries(this.modeButtons ?? {})) b.setAttribute("aria-pressed", String(id === mode));
    this.pad?.classList.toggle("pan-mode", mode === "pan");
  }

  /** Il pulsante unico: riapre il pad (anche se era ridotto per mancanza di spazio: lo studente lo vuole). */
  expand() {
    if (this.collapsed) return this.setCollapsed(false, { keepFocus: true });
    this.forceOpen = true;
    this.setSqueezed(false);
    this.modeButtons[this.nav.mode].focus();
    this.scheduleLayout();
  }

  /** keepFocus: il pulsante premuto sparisce, quindi il focus passa all'altro (chi usa la tastiera non si perde). */
  setCollapsed(collapsed, { persist = true, keepFocus = false } = {}) {
    const hadFocus = keepFocus && this.pad.contains(document.activeElement);
    this.collapsed = collapsed;
    this.forceOpen = false;
    this.setSqueezed(false);
    this.pad.classList.toggle("collapsed", collapsed);
    if (hadFocus) (collapsed ? this.fab : this.modeButtons[this.nav.mode]).focus();
    this.fab.setAttribute("aria-expanded", String(!collapsed));
    this.tip.classList.remove("visible");
    if (persist && !this.narrow.matches) writePref(PREF_COLLAPSED, collapsed ? "1" : "0");
    this.scheduleLayout();
  }

  toggleHelp(open = this.help.hidden) {
    if (open === !this.help.hidden) return;
    this.help.hidden = !open;
    this.nav.modalOpen = open;
    if (open) {
      this.returnFocus = document.activeElement;
      this.helpClose.focus();
    } else if (this.returnFocus?.isConnected) this.returnFocus.focus();
  }

  /**
   * Schermo intero di tutta la pagina, non del solo tab: la ricerca, il glossario e gli avvisi stanno
   * fuori dal tab e devono restare visibili, e cambiare tab non deve lasciare uno schermo nero.
   */
  toggleFullscreen() {
    const element = document.documentElement;
    const current = document.fullscreenElement ?? document.webkitFullscreenElement;
    const request = element.requestFullscreen ?? element.webkitRequestFullscreen;
    const exit = document.exitFullscreen ?? document.webkitExitFullscreen;
    try {
      const result = current ? exit.call(document) : request.call(element);
      result?.catch?.(() => this.showToast("Lo schermo intero non è disponibile qui."));
    } catch {
      this.showToast("Lo schermo intero non è disponibile qui.");
    }
  }

  syncFullscreen() {
    const on = Boolean(document.fullscreenElement ?? document.webkitFullscreenElement);
    this.fullscreenButton.replaceChildren(icon(on ? "fullscreenExit" : "fullscreen"));
    const label = on ? "Esci dallo schermo intero" : "Schermo intero";
    this.fullscreenButton.ariaLabel = label;
    this.fullscreenButton.dataset.tip = label;
    if (this.tipOwner === this.fullscreenButton) this.tip.textContent = label;
  }

  takePhoto() {
    try {
      const name = snapshotName(this.root.dataset.tab);
      const link = el("a", { href: this.nav.host.snapshot(), download: name });
      link.click();
      this.stage.classList.add("nav-flash");
      setTimeout(() => this.stage.classList.remove("nav-flash"), 350);
      this.showToast(`Foto salvata: ${name}`);
    } catch {
      this.showToast("Non è stato possibile salvare la foto.");
    }
  }

  showToast(text, ms = 3200, node = null) {
    clearTimeout(this.toastTimer);
    this.toast.replaceChildren(node ?? el("span", {}, text));
    this.toast.hidden = false;
    this.placeHud();
    this.toastTimer = setTimeout(() => (this.toast.hidden = true), ms);
  }

  /**
   * I chip in alto (benvenuto, avvisi, "Ti sei perso?") nella pila di un livello delle rappresentazioni
   * (radice = stage: core/representations.js) evitano la scheda della vista, che sta in alto a sinistra:
   * a destra della scheda se lì c'è posto, altrimenti sotto (sul telefono copriva i suoi comandi). Nello
   * stage di un tab non fa niente: lì le regole dei tab (CSS) sanno dove stanno le loro barre.
   * A eventi (un chip che compare, un ridimensionamento), mai per frame.
   */
  placeHud() {
    if (this.root !== this.stage) return;
    this.hud.style.left = "";
    this.hud.style.top = "";
    const sw = this.stage.clientWidth;
    const sh = this.stage.clientHeight;
    if (!sw || !sh) return;
    const box = this.stage.getBoundingClientRect();
    const right = parseFloat(this.hud.style.right) || 0;
    let left = 0;
    let top = parseFloat(getComputedStyle(this.hud).top) || 0;
    const cssTop = top;
    for (const node of this.stage.children) {
      if (node === this.pad || node === this.hud || node === this.help || node instanceof HTMLCanvasElement) continue;
      const r = node.getBoundingClientRect();
      const w = r.width;
      const h = r.height;
      // livelli a tutto schermo (etichette, linee guida) e suggerimenti nascosti non contano
      if (w < 2 || h < 2 || w * h > 0.5 * sw * sh || r.top - box.top > sh * 0.4 || !node.checkVisibility?.({ visibilityProperty: true })) continue;
      if (r.bottom - box.top <= cssTop) continue; // già sopra i chip
      const besideLeft = r.right - box.left + 12;
      if (sw - right - besideLeft >= HUD_MIN_WIDTH) left = Math.max(left, besideLeft);
      else top = Math.max(top, r.bottom - box.top + 10);
    }
    if (left) this.hud.style.left = `${Math.round(left)}px`;
    if (top !== cssTop) this.hud.style.top = `${Math.round(top)}px`;
  }

  /**
   * Il suggerimento di benvenuto, una volta per pagina e una volta per sempre (nav.tipSeen), solo se i chip
   * di questo pad si vedono davvero: non sotto un livello delle rappresentazioni (link diretto a
   * view:rete) né dietro il cassetto aperto del telefono. @returns {"done"|"later"|"shown"}
   */
  maybeShowWelcome() {
    if (tipShownThisSession || readPref(PREF_TIP_SEEN, "0") === "1") return "done";
    if (!(this.hud.checkVisibility?.({ visibilityProperty: true }) ?? Boolean(this.hud.offsetParent))) return "later";
    tipShownThisSession = true;
    writePref(PREF_TIP_SEEN, "1");
    const touch = window.matchMedia("(pointer: coarse)").matches;
    const text = touch
      ? "Muoviti nella scena: un dito per ruotare, due dita per zoom e spostamento, doppio tocco per centrare. Altri comandi nel pulsante in basso a destra."
      : "Muoviti nella scena: trascina per ruotare, tasto destro o Shift + trascina per spostarti, rotella per lo zoom, W A S D da tastiera.";
    const close = el("button", { type: "button", className: "nav-chip-close", ariaLabel: "Chiudi il suggerimento", onclick: () => (this.toast.hidden = true) }, icon("close"));
    const more = el("button", { type: "button", className: "nav-chip-action", onclick: () => this.toggleHelp(true) }, "Tutti i comandi", el("kbd", {}, "?"));
    this.welcomeOpen = true;
    this.showToast("", 12000, el("span", { className: "nav-welcome" }, el("span", {}, text), more, close));
  }

  /** Dopo il primo movimento il suggerimento di benvenuto ha fatto il suo lavoro: sparisce poco dopo. */
  onFirstInput() {
    if (!this.welcomeOpen) return;
    this.welcomeOpen = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => (this.toast.hidden = true), 2500);
  }

  // ---------------------------------------------------------------- posizionamento
  scheduleLayout() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.layout();
    });
  }

  /**
   * Sceglie dove mettere il pad: nell'angolo in basso a destra se è libero, altrimenti accanto al
   * pannello del codice; in entrambi i casi sopra la barra del player se la toccherebbe. Se non c'è
   * un posto che lasci libera la scena (finestra stretta con codice e player aperti), il pad si
   * riduce da solo al pulsante unico, senza ricordarlo: tornato lo spazio, si riapre.
   * Gira solo quando cambia qualcosa (ridimensionamenti, pannelli, player), mai per frame.
   */
  layout() {
    const sw = this.stage.clientWidth;
    const sh = this.stage.clientHeight;
    if (!sw || !sh) return; // tab nascosto
    const stageRect = this.stage.getBoundingClientRect();
    const rel = (r) => ({ left: r.left - stageRect.left, right: r.right - stageRect.left, top: r.top - stageRect.top, bottom: r.bottom - stageRect.top });
    if (!this.collapsed && !this.squeezed) this.openSize = { w: this.pad.offsetWidth, h: this.pad.offsetHeight };
    const narrow = this.narrow.matches;
    const codeRoot = this.codePanel?.root;
    // visibility: hidden (un tab può nascondere il codice senza toglierlo) conta come nascosto
    const codeRect = codeRoot && !codeRoot.hidden && codeRoot.offsetHeight && getComputedStyle(codeRoot).visibility !== "hidden" ? rel(codeRoot.getBoundingClientRect()) : null;
    const bar = this.placeFollowChip(this.player && !this.player.bar.hidden ? rel(this.player.bar.getBoundingClientRect()) : null, codeRect);
    const chartsWidth = this.charts?.coveredWidth() ?? 0;
    const caption = this.root.querySelector(":scope > .caption");
    const captionShown = Boolean(caption?.textContent.trim());
    const rootWidth = this.root.clientWidth;
    // il `left` del CSS (un tab può restringere la didascalia al suo testo e centrarla)
    const captionLeft = caption ? parseFloat(getComputedStyle(caption).left) || 16 : 16;
    const captionHeight = caption?.offsetHeight ?? 0;
    // la didascalia è figlia della radice: la barra del player passa dalle coordinate dello stage alle sue
    const shift = stageRect.left - this.root.getBoundingClientRect().left;
    const barInRoot = bar ? { ...bar, left: bar.left + shift, right: bar.right + shift } : null;
    // Sul telefono prima la didascalia (sopra il codice ridotto se gli finirebbe sotto), poi il pad: il
    // pulsante unico sta sotto tutto, accanto a lei; aperto, il pad sale sopra la didascalia.
    let captionSpot = null;
    if (narrow) captionSpot = captionPlacement({ width: rootWidth, left: captionLeft, sh, height: captionHeight, code: codeRect, bar: barInRoot, narrow: true, fabRoom: FAB_ROOM });
    const captionTop = captionShown ? (captionSpot?.bottom ?? 14) + captionHeight + 6 : 14;
    const base = narrow ? (this.collapsed ? 4 : Math.max(46, captionTop)) : 14;
    // Sul telefono il codice (anche ridotto) sta in basso per tutta la larghezza: il pad aperto gli sta sopra.
    const lowest = narrow && !this.collapsed && codeRect && codeRect.top > sh / 2 ? codeRect : null;
    const where = (size) => padPlacement({ sw, sh, size, bar, code: !narrow && !chartsWidth ? codeRect : null, lowest, chartsWidth, edge: narrow ? 8 : 14, base });

    const open = !this.collapsed;
    let spot = where({ w: this.openSize?.w ?? this.pad.offsetWidth, h: this.openSize?.h ?? this.pad.offsetHeight });
    if (spot.ok) this.forceOpen = false;
    this.setSqueezed(open && !narrow && !spot.ok && !this.forceOpen);
    if (this.squeezed) spot = where({ w: this.fab.offsetWidth || 40, h: this.fab.offsetHeight || 40 });
    this.pad.style.right = `${spot.right}px`;
    this.pad.style.bottom = `${spot.bottom}px`;
    // I chip in alto (benvenuto, "Ti sei perso?") stanno sopra la parte visibile della scena (in una parte
    // stretta il benvenuto mette il testo sopra e i pulsanti sotto: container query in nav.css).
    this.hud.style.right = `${this.coveredRight()}px`;
    this.placeHud();
    // La didascalia in basso: accanto al pad se stanno sulla stessa riga e resta larga, altrimenti sopra.
    if (!narrow) {
      const padSize = this.pad.offsetWidth ? { w: this.pad.offsetWidth, h: this.pad.offsetHeight } : { w: 0, h: 0 };
      captionSpot = captionPlacement({ width: rootWidth, left: captionLeft, sh, height: captionHeight, pad: { right: spot.right, bottom: spot.bottom, ...padSize }, chartsWidth, bar: barInRoot });
    }
    this.root.style.setProperty("--nav-caption-right", `${captionSpot.right}px`);
    this.root.style.setProperty("--nav-caption-bottom", `${captionSpot.bottom}px`);
    this.root.classList.toggle("nav-caption-squeezed", captionShown && captionSpot.hidden);
  }

  /**
   * Il chip "Camera libera · Segui l'azione" sta sopra la barra del player; se subito sopra la barra c'è
   * il codice ridotto (telefono con .code-stays-bottom) sale sopra il codice, invece di coprirne
   * l'intestazione. Restituisce l'ingombro di barra + chip visibile: il pad sta sopra entrambi, così il
   * pulsante "Segui l'azione" resta sempre toccabile.
   */
  placeFollowChip(bar, code) {
    if (!bar || !this.follow) return bar;
    const touching = code && code.left < bar.right && code.right > bar.left && code.bottom <= bar.top + 4 && code.bottom >= bar.top - 40;
    const lift = touching ? bar.top - code.top + 8 : 8;
    this.follow.style.bottom = `calc(100% + ${lift}px)`;
    if (this.follow.hidden) return bar;
    return { ...bar, top: Math.min(bar.top, bar.top - lift - this.follow.offsetHeight) };
  }

  /** Pad ridotto "per mancanza di spazio" (non è una scelta dello studente: non si ricorda). */
  setSqueezed(squeezed) {
    if (squeezed === Boolean(this.squeezed)) return;
    const hadFocus = this.pad.contains(document.activeElement);
    this.squeezed = squeezed;
    this.pad.classList.toggle("squeezed", squeezed);
    this.fab.setAttribute("aria-expanded", String(!this.collapsed && !squeezed));
    this.tip.classList.remove("visible");
    if (hadFocus && squeezed) this.fab.focus();
  }
}

function fullscreenSupported() {
  return Boolean(document.fullscreenEnabled ?? document.webkitFullscreenEnabled);
}

/** Le sezioni dell'aiuto: sui touch screen prima i gesti e i pulsanti, poi mouse e tastiera. */
function helpSections() {
  if (!window.matchMedia("(pointer: coarse)").matches) return HELP;
  return [...HELP.filter((s) => s.touch), ...HELP.filter((s) => !s.touch)];
}
