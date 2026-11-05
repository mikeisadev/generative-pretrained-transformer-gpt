/**
 * Guscio della web app: barra dei tab, caricamento pigro dei moduli, routing con #hash.
 *
 * Ogni tab è un modulo in js/tabs/<id>/index.js che esporta:
 *   export async function createTab(root, app) → { activate(), deactivate() }
 * Il modulo viene importato solo alla prima apertura; quando il tab non è visibile
 * deactivate() ferma la sua scena 3D.
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
    this.instances = new Map();
    this.current = null;
    this.bus = new EventTarget(); // eventi tra tab, es. "rag:open" per collegare RAG e Transformer
    for (const tab of TABS) {
      this.nav.append(el("a", { href: `#${tab.id}`, className: "tab-link", dataset: { tab: tab.id } }, tab.title));
    }
    window.addEventListener("hashchange", () => this.route());
    this.route();
  }

  /** Apre un tab da codice (es. un pulsante "Vai al tab RAG"). */
  open(id) {
    if (location.hash !== `#${id}`) location.hash = id;
    else this.route();
  }

  async route() {
    const id = location.hash.slice(1).split("/")[0] || TABS[0].id;
    const tab = TABS.find((t) => t.id === id) ?? TABS[0];
    if (this.current === tab.id) return;
    this.nav.querySelectorAll(".tab-link").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab.id));
    if (this.current) {
      const prev = this.instances.get(this.current);
      prev?.root && (prev.root.hidden = true);
      prev?.instance?.deactivate?.();
    }
    this.current = tab.id;
    let entry = this.instances.get(tab.id);
    if (!entry) {
      const root = el("section", { className: "tab-root", dataset: { tab: tab.id } });
      this.main.append(root);
      entry = { root, instance: null };
      this.instances.set(tab.id, entry);
      try {
        const module = await tab.load();
        entry.instance = await module.createTab(root, this);
      } catch (err) {
        console.error(err);
        root.append(el("p", { className: "tab-error" }, `Errore nel caricamento del tab: ${err.message}`));
      }
    }
    entry.root.hidden = false;
    if (this.current === tab.id) entry.instance?.activate?.();
  }
}

window.__app = new App();
