/**
 * Condivisione: copia di un link diretto negli appunti e un piccolo avviso ("toast").
 *
 *   await copyLink("#rag/go=step:index.chunk");   // copia l'indirizzo completo e mostra "Link copiato"
 *   showToast("Testo breve");
 */
import { el } from "../dom.js";

let toastBox = null;
let toastTimer = 0;

/** Avviso breve in basso al centro; viene letto anche dagli screen reader (aria-live). */
export function showToast(text, { duration = 2200 } = {}) {
  if (!toastBox) {
    toastBox = el("div", { className: "app-toast", role: "status", "aria-live": "polite" });
    document.body.append(toastBox);
  }
  toastBox.textContent = text;
  toastBox.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastBox.classList.remove("visible"), duration);
}

/** Indirizzo completo della pagina con un hash diverso (es. "#audio/go=step:freq.mel"). */
export function urlWithHash(hash) {
  const { origin, pathname, search } = location;
  const base = origin && origin !== "null" ? origin + pathname + search : location.href.split("#")[0];
  return base + hash;
}

/** Copia un testo negli appunti; restituisce true se ci è riuscito. */
export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // permesso negato: si prova con il metodo classico
  }
  // Fuori da https/localhost l'API Clipboard non c'è: textarea temporanea + execCommand.
  const area = el("textarea", { value: text, readOnly: true, className: "clipboard-helper", "aria-hidden": "true" });
  const active = document.activeElement;
  document.body.append(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  area.remove();
  active?.focus?.({ preventScroll: true });
  return ok;
}

/** Copia il link a un hash della web app e lo dice all'utente. */
export async function copyLink(hash, { label = "Link copiato negli appunti" } = {}) {
  const url = urlWithHash(hash);
  const ok = await copyText(url);
  showToast(ok ? label : `Copia questo indirizzo: ${url}`, { duration: ok ? 2200 : 6000 });
  return ok;
}
