/**
 * Guscio della web app: barra dei tab, caricamento pigro dei moduli, routing con #hash,
 * ricerca globale e link diretti.
 *
 * Ogni tab è un modulo in js/tabs/<id>/index.js che esporta:
 *   export async function createTab(root, app) → { activate(), deactivate(), goto?({kind, id}) }
 * Il modulo viene importato solo alla prima apertura; quando il tab non è visibile
 * deactivate() ferma la sua scena 3D (i tab nascosti non consumano GPU).
 * L'indice della ricerca sta a parte, in js/tabs/<id>/search.js (solo dati leggeri): così la
 * ricerca trova anche i passi dei tab mai aperti senza caricarne la scena.
 *
 * API per i tab (oggetto `app`):
 *   app.bus                     EventTarget condiviso: eventi tra tab (elenco in ARCHITECTURE.md)
 *   await app.open(id)          apre un tab (anche "rag/q=…") e attende che sia pronto
 *   await app.whenReady(id)     attende che il tab sia stato creato (senza aprirlo); → istanza o null
 *   await app.goto(id, target)  apre il tab su un passo/azione/vista: target = {kind, id} (vedi sotto)
 *   app.openSearch(query?)      apre la ricerca globale
 *   app.showTerm(id)            mostra un termine del glossario (senza cambiare tab)
 *
 * Link diretti: "#<tab>/go=<kind>:<id>" (es. #transformer/go=step:head.qkv) fa la stessa cosa di
 * app.goto, sia al caricamento della pagina sia quando l'hash cambia; avanti/indietro del browser
 * ripercorrono anche questi salti.
 */
import { el } from "./core/dom.js";
import { formatHash, parseHash } from "./core/search.js";

export const TABS = [
  {
    id: "transformer",
    title: "Transformer",
    description: "Il GPT del corso in 3D: embedding, attention, blocchi, training e generazione.",
    keywords: ["gpt", "modello", "architettura"],
    load: () => import("./tabs/transformer/index.js"),
    search: () => import("./tabs/transformer/search.js"),
  },
  {
    id: "lab",
    title: "Laboratorio pesi",
    description: "I pesi del GPT in Python come nel codice: numeri, run caricati dal training, confronti e benchmark.",
    keywords: ["pesi", "weights", "run", "snapshot", "safetensors", "confronto", "benchmark", "laboratorio"],
    load: () => import("./tabs/lab/index.js"),
    search: () => import("./tabs/lab/search.js"),
  },
  {
    id: "rag",
    title: "Database vettoriale (RAG)",
    description: "Dal PDF ai vettori, indice HNSW, ricerca e risposta dell'LLM (lezioni L17-L18).",
    keywords: ["rag", "pgvector", "spring ai", "retrieval"],
    load: () => import("./tabs/rag/index.js"),
    search: () => import("./tabs/rag/search.js"),
  },
  {
    id: "diffusion",
    title: "Immagini e video",
    description: "Modelli di diffusione: rumore, U-Net, VAE, DiT e generazione video.",
    keywords: ["diffusione", "stable diffusion", "immagini", "video"],
    load: () => import("./tabs/diffusion/index.js"),
    search: () => import("./tabs/diffusion/search.js"),
  },
  {
    id: "audio",
    title: "Audio e musica",
    description: "Onde, spettrogrammi, codec neurali con RVQ, generazione di musica e Whisper.",
    keywords: ["suono", "musica", "voce", "speech"],
    load: () => import("./tabs/audio/index.js"),
    search: () => import("./tabs/audio/search.js"),
  },
  {
    id: "modern-llm",
    title: "LLM moderni",
    description: "Dal nostro GPT ai modelli di oggi: RoPE, MoE, KV cache, ragionamento, agenti.",
    keywords: ["llm", "modelli moderni", "frontier"],
    load: () => import("./tabs/modern-llm/index.js"),
    search: () => import("./tabs/modern-llm/search.js"),
  },
];

/** true se il tasto premuto finirebbe in un campo di testo (lì "/" va scritto, non intercettato). */
const isTyping = (target) => Boolean(target?.closest?.("input, textarea, select, [contenteditable=''], [contenteditable='true']"));

class App {
  constructor() {
    this.nav = document.querySelector("[data-tabs]");
    this.main = document.querySelector("#app");
    /** id → { root, instance, ready: Promise<istanza|null> } */
    this.instances = new Map();
    this.current = null;
    this.bus = new EventTarget();
    /** ultimo hash già gestito: l'evento hashchange che segue un goto non lo ripete */
    this.routedHash = null;
    for (const tab of TABS) {
      this.nav.append(el("a", { href: `#${tab.id}`, className: "tab-link", dataset: { tab: tab.id } }, tab.title));
    }
    window.addEventListener("hashchange", () => this.route());
    this.installSearch();
    this.route();
  }

  /** Apre un tab da codice (es. un pulsante "Vai al tab RAG"); si risolve quando il tab è pronto. */
  open(id) {
    if (location.hash !== `#${id}`) location.hash = id;
    this.route(); // subito, senza aspettare hashchange (che poi non ripete il lavoro)
    return this.whenReady(id.split("/")[0]);
  }

  /** Si risolve con l'istanza del tab quando è stato creato (null se il caricamento è fallito). */
  async whenReady(id) {
    // Il tab può non essere ancora in costruzione (hashchange arriva dopo): aspettiamo che lo sia.
    for (let i = 0; i < 200 && !this.instances.has(id); i++) await new Promise((r) => setTimeout(r, 25));
    return (await this.instances.get(id)?.ready) ?? null;
  }

  /**
   * Apre il tab `id` e lo porta su un elemento: target = {kind: "step"|"action"|"view", id}
   * (lo gestisce `instance.goto(target)` del tab) oppure {kind: "term", id} (scheda del glossario).
   * Senza target apre solo il tab. Aggiorna l'indirizzo (#<tab>/go=<kind>:<id>), quindi il
   * pulsante "indietro" del browser riporta al punto precedente.
   * @returns {Promise<boolean>} true se l'elemento è stato mostrato, false se si è solo aperto il tab
   */
  async goto(id, target = null) {
    if (!TABS.some((t) => t.id === id)) return false;
    const hash = formatHash(id, target);
    if (location.hash !== hash) location.hash = hash;
    // Anche se l'hash non cambia (stesso link scelto due volte) il salto va rifatto.
    this.routedHash = location.hash;
    this.routing = this.applyRoute(parseHash(location.hash));
    return this.routing;
  }

  route() {
    if (location.hash === this.routedHash) return this.routing;
    this.routedHash = location.hash;
    this.routing = this.applyRoute(parseHash(location.hash));
    return this.routing;
  }

  async applyRoute({ tab: id, rest, go }) {
    const tab = TABS.find((t) => t.id === id) ?? TABS[0];
    // Un cambio di indirizzo non chiesto dalla palette (es. "indietro") la chiude.
    if (go?.kind !== "term") this.palette?.close({ restoreFocus: false });
    const entry = await this.show(tab);
    if (!go && rest.startsWith("go=") && this.current === tab.id) {
      // link diretto rovinato (es. "#audio/go=step:" senza id): si apre il tab e lo si dice
      await this.warnMissing(tab);
      return false;
    }
    if (!go || this.current !== tab.id || !entry.instance) return false;
    return this.applyTarget(tab, entry.instance, go);
  }

  async warnMissing(tab) {
    const { showToast } = await import("./core/ui/share.js");
    showToast(`Non trovo questo elemento nel tab ${tab.title}: ho aperto il tab.`, { duration: 4000 });
  }

  /** Rende visibile un tab (creandolo la prima volta); si risolve quando è pronto e attivo. */
  async show(tab) {
    if (this.current === tab.id) {
      const entry = this.instances.get(tab.id);
      await entry.ready;
      return entry;
    }
    this.nav.querySelectorAll(".tab-link").forEach((a) => {
      const active = a.dataset.tab === tab.id;
      a.classList.toggle("active", active);
      if (active) {
        a.setAttribute("aria-current", "page");
        a.scrollIntoView({ block: "nearest", inline: "nearest" }); // sul telefono la barra scorre
      } else a.removeAttribute("aria-current");
    });
    if (this.current) {
      const prev = this.instances.get(this.current);
      if (prev) prev.root.hidden = true;
      prev?.instance?.deactivate?.();
    }
    this.current = tab.id;
    let entry = this.instances.get(tab.id);
    if (!entry) {
      const root = el("section", { className: "tab-root", dataset: { tab: tab.id } });
      this.main.append(root);
      entry = { root, instance: null, ready: null };
      this.instances.set(tab.id, entry);
      entry.ready = this.create(tab, entry);
    }
    entry.root.hidden = false;
    await entry.ready;
    // Durante il caricamento l'utente può essere già passato a un altro tab: allora resta nascosto.
    if (this.current === tab.id) entry.instance?.activate?.();
    else entry.root.hidden = true;
    return entry;
  }

  async applyTarget(tab, instance, go) {
    if (go.kind === "term") return this.showTerm(go.id);
    if (typeof instance.goto !== "function") return false;
    let shown;
    try {
      shown = await instance.goto(go);
    } catch (err) {
      console.error(err);
      shown = false;
    }
    if (shown === false) {
      await this.warnMissing(tab);
      return false;
    }
    return true;
  }

  async create(tab, entry) {
    try {
      const module = await tab.load();
      entry.instance = await module.createTab(entry.root, this);
    } catch (err) {
      console.error(err);
      entry.root.append(el("p", { className: "tab-error" }, `Errore nel caricamento del tab: ${err.message}`));
    }
    return entry.instance;
  }

  // --- ricerca globale -----------------------------------------------------------------------

  /** La palette si carica alla prima richiesta (non pesa sull'avvio della pagina). */
  getPalette() {
    this.palettePromise ??= import("./core/ui/palette.js").then(({ SearchPalette }) => {
      this.palette = new SearchPalette({ tabs: TABS, currentTab: () => this.current, goto: (id, target) => this.goto(id, target) });
      return this.palette;
    });
    return this.palettePromise;
  }

  async openSearch(query) {
    (await this.getPalette()).open(query);
  }

  async showTerm(id) {
    return (await this.getPalette()).showTerm(id);
  }

  installSearch() {
    const button = document.querySelector("[data-search]");
    if (button) {
      button.addEventListener("click", () => this.openSearch());
      // Al passaggio del mouse o al fuoco la palette e gli indici si preparano: l'apertura è immediata.
      const prefetch = () => this.getPalette().then((p) => p.loadIndex());
      button.addEventListener("pointerenter", prefetch, { once: true });
      button.addEventListener("focus", prefetch, { once: true });
      const shortcut = /Mac|iPhone|iPad/.test(navigator.userAgentData?.platform ?? navigator.platform ?? "") ? "⌘ K" : "Ctrl K";
      const key = button.querySelector("kbd");
      if (key) key.textContent = shortcut;
      button.title = `Cerca in tutta la lezione (${shortcut} oppure /)`;
    }
    window.addEventListener("keydown", (e) => {
      if (e.defaultPrevented) return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && k === "k") {
        e.preventDefault(); // altrimenti il browser apre la sua ricerca
        this.getPalette().then((p) => p.toggle());
      } else if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey && !isTyping(e.target) && !this.palette?.isOpen) {
        e.preventDefault();
        this.openSearch();
      }
    });
  }
}

window.__app = new App();
