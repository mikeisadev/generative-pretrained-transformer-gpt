/**
 * Guscio della web app: barra dei tab, caricamento pigro dei moduli, routing con #hash.
 *
 * Ogni tab è un modulo in js/tabs/<id>/index.js che esporta:
 *   export async function createTab(root, app) → { activate(), deactivate() }
 * Il modulo viene importato solo alla prima apertura; quando il tab non è visibile
 * deactivate() ferma la sua scena 3D (i tab nascosti non consumano GPU).
 *
 * API per i tab (oggetto `app`):
 *   app.bus                  EventTarget condiviso: eventi tra tab (elenco in ARCHITECTURE.md)
 *   await app.open(id)       apre un tab (anche "rag/q=…") e attende che sia pronto
 *   await app.whenReady(id)  attende che il tab sia stato creato (senza aprirlo); → istanza o null
 */
import { el } from "./core/dom.js";

export const TABS = [
  { id: "transformer", title: "Transformer", load: () => import("./tabs/transformer/index.js") },
  { id: "rag", title: "Database vettoriale (RAG)", load: () => import("./tabs/rag/index.js") },
  { id: "diffusion", title: "Immagini e video", load: () => import("./tabs/diffusion/index.js") },
  { id: "audio", title: "Audio e musica", load: () => import("./tabs/audio/index.js") },
  { id: "modern-llm", title: "LLM moderni", load: () => import("./tabs/modern-llm/index.js") },
];

class App {
  constructor() {
    this.nav = document.querySelector("[data-tabs]");
    this.main = document.querySelector("#app");
    /** id → { root, instance, ready: Promise<istanza|null> } */
    this.instances = new Map();
    this.current = null;
    this.bus = new EventTarget();
    for (const tab of TABS) {
      this.nav.append(el("a", { href: `#${tab.id}`, className: "tab-link", dataset: { tab: tab.id } }, tab.title));
    }
    window.addEventListener("hashchange", () => this.route());
    this.route();
  }

  /** Apre un tab da codice (es. un pulsante "Vai al tab RAG"); si risolve quando il tab è pronto. */
  open(id) {
    if (location.hash !== `#${id}`) location.hash = id; // l'evento hashchange chiama route()
    else this.route();
    return this.whenReady(id.split("/")[0]);
  }

  /** Si risolve con l'istanza del tab quando è stato creato (null se il caricamento è fallito). */
  async whenReady(id) {
    // Il tab può non essere ancora in costruzione (hashchange arriva dopo): aspettiamo che lo sia.
    for (let i = 0; i < 200 && !this.instances.has(id); i++) await new Promise((r) => setTimeout(r, 25));
    return (await this.instances.get(id)?.ready) ?? null;
  }

  async route() {
    const id = location.hash.slice(1).split("/")[0] || TABS[0].id;
    const tab = TABS.find((t) => t.id === id) ?? TABS[0];
    if (this.current === tab.id) return;
    this.nav.querySelectorAll(".tab-link").forEach((a) => {
      const active = a.dataset.tab === tab.id;
      a.classList.toggle("active", active);
      if (active) {
        a.setAttribute("aria-current", "page");
        a.scrollIntoView({ block: "nearest", inline: "nearest" }); // sul telefono la barra scorre
      }
      else a.removeAttribute("aria-current");
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
}

window.__app = new App();
