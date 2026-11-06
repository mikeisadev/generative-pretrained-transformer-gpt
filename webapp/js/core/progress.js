/**
 * Progressi dello studente: quali passi del Percorso ha già aperto, per ogni tab.
 *
 * I dati stanno nel localStorage del browser (solo su questo dispositivo). Se il localStorage
 * non è disponibile (finestra privata, dati bloccati) i progressi valgono solo per la sessione.
 *
 *   import { progress } from "../progress.js";
 *   progress.mark("transformer", "head.qkv");      // true se è la prima volta
 *   progress.visited("transformer")                 // Set degli id visti
 *   progress.reset("transformer");
 *   progress.addEventListener("change", (e) => e.detail.tab);
 */

const PREFIX = "gpt-lesson.progress.";

/** localStorage se usabile, altrimenti null (ogni accesso può lanciare un'eccezione). */
function browserStorage() {
  try {
    const storage = globalThis.localStorage;
    if (!storage) return null;
    const probe = `${PREFIX}probe`;
    storage.setItem(probe, "1");
    storage.removeItem(probe);
    return storage;
  } catch {
    return null;
  }
}

export class ProgressStore extends EventTarget {
  /** @param {Storage|null} [storage] un oggetto con getItem/setItem/removeItem (null = solo memoria) */
  constructor(storage = browserStorage()) {
    super();
    this.storage = storage;
    /** tab → Set degli id visti (letti una volta, poi tenuti in memoria) */
    this.cache = new Map();
  }

  visited(tab) {
    if (!this.cache.has(tab)) this.cache.set(tab, new Set(this.read(tab)));
    return this.cache.get(tab);
  }

  /** Segna un passo come visto; restituisce true se non lo era. */
  mark(tab, id) {
    const set = this.visited(tab);
    if (set.has(id)) return false;
    set.add(id);
    this.write(tab, set);
    this.dispatchEvent(new CustomEvent("change", { detail: { tab, id } }));
    return true;
  }

  reset(tab) {
    this.visited(tab).clear();
    try {
      this.storage?.removeItem(PREFIX + tab);
    } catch {
      // nulla da fare: in memoria è già azzerato
    }
    this.dispatchEvent(new CustomEvent("change", { detail: { tab, id: null } }));
  }

  /** Quanti dei passi `ids` sono stati visti. */
  count(tab, ids) {
    const set = this.visited(tab);
    let n = 0;
    for (const id of ids) if (set.has(id)) n++;
    return n;
  }

  read(tab) {
    try {
      const parsed = JSON.parse(this.storage?.getItem(PREFIX + tab) ?? "[]");
      return Array.isArray(parsed) ? parsed.filter((id) => typeof id === "string") : [];
    } catch {
      return [];
    }
  }

  write(tab, set) {
    try {
      this.storage?.setItem(PREFIX + tab, JSON.stringify([...set]));
    } catch {
      // spazio esaurito o accesso negato: i progressi restano validi per questa sessione
    }
  }
}

/** Istanza condivisa da Menu e palette di ricerca. */
export const progress = new ProgressStore();
