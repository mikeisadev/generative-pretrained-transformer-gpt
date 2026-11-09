/**
 * Barra dei sottotab del tab "LLM moderni": "Vista d'insieme tecnologie LLM moderni" /
 * "Architettura LLM moderni completa". Solo DOM (niente three.js).
 *
 * Sta nel flusso, sopra lo stage e allineata a lui (segue l'apertura della sidebar): lo stage parte
 * sotto la barra (classe `mllm-has-subtabs` sulla radice del tab, regole in css/tabs/modern-llm.css),
 * così nessun comando dello stage viene coperto. Schede accessibili (role=tablist, frecce, Home/Fine,
 * focus visibile); sugli schermi stretti etichette brevi con il titolo intero nel tooltip e
 * nell'aria-label. Tasto V: passa all'altro sottotab (mai nei campi, nelle finestre di dialogo, con
 * l'aiuto aperto o con Ctrl/⌘/Alt). A destra un'area di contesto per i collegamenti tra le due viste
 * ("← Torna all'architettura completa", "Dove sta in …").
 *
 *   const bar = createSubtabs({ root, items, current: "overview", onSelect: (id) => showSubtab(id), isBlocked });
 *   bar.setCurrent("full"); bar.setContext({ back, link }); bar.dispose();
 */
import { el } from "../../core/dom.js";

/** Tasto che passa all'altro sottotab (fuori da navigazione, Percorso e player: vedi navMath.js). */
export const SUBTAB_KEY = "KeyV";
/** Tooltip di "Azzera" nei due Percorsi: i passi visti sono del tab (stesso tabId), non del sottotab. */
export const RESET_TITLE = "Azzera i passi visti di tutto il tab: vista d'insieme e architettura completa";
const TEXT_TARGETS = "input, textarea, select, [contenteditable=''], [contenteditable='true'], [role='dialog'], dialog";

/**
 * @param {object} o
 * @param {HTMLElement} o.root radice del tab (riceve la classe che sposta lo stage)
 * @param {{id: string, label: string, short: string}[]} o.items
 * @param {string} o.current
 * @param {(id: string) => void} o.onSelect lo studente ha scelto un sottotab
 * @param {() => boolean} [o.isBlocked] true se i tasti non devono agire (es. aiuto della navigazione aperto)
 */
export function createSubtabs({ root, items, current, onSelect, isBlocked = () => false }) {
  const buttons = new Map();
  const tablist = el("div", { className: "mllm-subtabs-list", role: "tablist", ariaLabel: "Viste del tab LLM moderni" });
  for (const item of items) {
    const button = el(
      "button",
      {
        type: "button",
        role: "tab",
        className: "mllm-subtab",
        id: `mllm-subtab-${item.id}`,
        title: `${item.label} (tasto V)`,
        ariaLabel: item.label,
        dataset: { subtab: item.id },
        onclick: () => pick(item.id),
      },
      el("span", { className: "mllm-subtab-long" }, item.label),
      el("span", { className: "mllm-subtab-short", ariaHidden: "true" }, item.short),
    );
    button.setAttribute("aria-keyshortcuts", "V");
    buttons.set(item.id, button);
    tablist.append(button);
  }
  tablist.addEventListener("keydown", (e) => {
    const ids = [...buttons.keys()];
    const at = ids.indexOf(document.activeElement?.dataset?.subtab);
    if (at < 0) return;
    let next = -1;
    if (e.key === "ArrowRight") next = (at + 1) % ids.length;
    else if (e.key === "ArrowLeft") next = (at - 1 + ids.length) % ids.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = ids.length - 1;
    if (next < 0) return;
    e.preventDefault();
    buttons.get(ids[next]).focus();
    pick(ids[next]);
  });

  const context = el("div", { className: "mllm-subtabs-context", ariaLive: "polite" });
  const element = el("div", { className: "mllm-subtabs" }, tablist, context);
  // Primo nel DOM (la posizione la decide il CSS): con il tasto Tab, dopo la barra dei tab si arriva qui
  // e poi alla sidebar e alla scena. In fondo alla radice del tab servivano più di 40 pressioni.
  root.prepend(element);
  root.classList.add("mllm-has-subtabs");

  let selected = null;
  setCurrent(current);

  function pick(id) {
    if (id === selected) return;
    setCurrent(id);
    onSelect(id);
  }

  function setCurrent(id) {
    selected = id;
    for (const [bid, button] of buttons) {
      const on = bid === id;
      button.setAttribute("aria-selected", String(on));
      button.tabIndex = on ? 0 : -1; // tabindex mobile: Tab entra sulla scheda attiva, le frecce scelgono
      button.classList.toggle("active", on);
    }
  }

  /** Tasto V: l'altro sottotab. Solo con il tab visibile e fuori da campi e finestre di dialogo. */
  const onKey = (e) => {
    if (e.code !== SUBTAB_KEY || e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
    if (root.hidden || !root.isConnected || isBlocked()) return;
    if (e.target?.closest?.(TEXT_TARGETS)) return;
    const ids = [...buttons.keys()];
    e.preventDefault();
    pick(ids[(ids.indexOf(selected) + 1) % ids.length]);
  };
  window.addEventListener("keydown", onKey);

  /**
   * Area di contesto a destra: `back` (pulsante evidenziato per tornare dove eri) e `link`
   * (collegamento verso l'altra vista). Ognuno {label, short, title, onClick} oppure null.
   */
  function setContext({ back = null, link = null } = {}) {
    const make = (spec, className) =>
      spec
        ? el(
            "button",
            { type: "button", className, title: spec.title ?? spec.label, ariaLabel: spec.title ?? spec.label, onclick: spec.onClick },
            el("span", { className: "mllm-subtab-long" }, spec.label),
            el("span", { className: "mllm-subtab-short", ariaHidden: "true" }, spec.short ?? spec.label),
          )
        : null;
    context.replaceChildren(...[make(back, "mllm-subtabs-back"), make(link, "mllm-subtabs-link")].filter(Boolean));
  }

  return {
    element,
    get current() {
      return selected;
    },
    setCurrent,
    setContext,
    dispose() {
      window.removeEventListener("keydown", onKey);
      element.remove();
      root.classList.remove("mllm-has-subtabs");
    },
  };
}
