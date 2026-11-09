/**
 * Interruttore a voci (radiogroup accessibile) con lo stile del tab: lo usano la disposizione della
 * Rete neurale ("A righe / Orizzontale con profondità") e la sorgente dei pesi della vista "Pesi
 * numerici" ("Pesi finali / Training"). Solo DOM.
 *
 * Tastiera come un gruppo di pulsanti radio: frecce e Home/Fine scelgono (tabindex mobile: un solo
 * punto di tabulazione), Spazio e Invio sono della voce (preventDefault: niente pausa del player, niente
 * passo del Percorso, la pagina non scorre).
 *
 *   const s = segmentedSwitch({ options: [{ id, label, short?, title?, icon? }], value, onChange,
 *                               ariaLabel, dataKey: "arrangement", className: "tf-arrange" });
 *   s.element; s.set(id);
 */
import { el } from "../../../core/dom.js";

/**
 * @param {object} o
 * @param {{id: string, label: string, short?: string, title?: string, icon?: () => Element}[]} o.options
 * @param {string} o.value voce scelta all'inizio
 * @param {(id: string) => void} o.onChange
 * @param {string} o.ariaLabel nome del gruppo per gli screen reader
 * @param {string} [o.dataKey] nome del data-attribute con l'id della voce (es. "arrangement" → data-arrangement)
 * @param {boolean} [o.compact] variante della barra dello stage (classe "compact")
 * @returns {{element: HTMLElement, set: (id: string) => void}}
 */
export function segmentedSwitch({ options, value, onChange, ariaLabel, dataKey = "option", compact = false }) {
  const group = el("div", { className: `tf-arrange${compact ? " compact" : ""}`, role: "radiogroup", ariaLabel });
  const buttons = options.map((o) =>
    el(
      "button",
      { type: "button", role: "radio", className: "tf-arrange-option", dataset: { [dataKey]: o.id }, title: o.title ?? "", onclick: () => choose(o.id) },
      o.icon ? o.icon() : null,
      el("span", { className: "tf-arrange-label" }, o.label),
      el("span", { className: "tf-arrange-short" }, o.short ?? o.label),
    ),
  );
  group.append(...buttons);
  const idOf = (b) => b.dataset[dataKey];
  const choose = (id, focus = false) => {
    set(id);
    if (focus) buttons.find((b) => idOf(b) === id)?.focus();
    onChange(id);
  };
  group.addEventListener("keydown", (e) => {
    const i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    const keys = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 };
    let next = null;
    if (e.key in keys) next = (i + keys[e.key] + buttons.length) % buttons.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = buttons.length - 1;
    else if (e.key === " " || e.key === "Enter") next = i;
    if (next === null) return;
    e.preventDefault(); // la pagina non scorre, il Percorso non cambia passo, il player non va in pausa
    choose(idOf(buttons[next]), true);
  });
  function set(id) {
    for (const b of buttons) {
      const on = idOf(b) === id;
      b.setAttribute("aria-checked", String(on));
      b.classList.toggle("active", on);
      b.tabIndex = on ? 0 : -1;
    }
  }
  set(value);
  return { element: group, set };
}
