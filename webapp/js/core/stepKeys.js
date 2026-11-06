/**
 * Tasti del Percorso e del player, uguali in tutti i tab:
 *   ↑ ↓     passo precedente / successivo del Percorso (un passo per pressione: tenendo premuto non si
 *           "segnano come visti" dieci passi di fila)
 *   Spazio  pausa / riprendi l'azione in corso
 * Le lettere e ← → sono della navigazione libera (js/core/navigation.js).
 *
 * Ascolto sulla finestra, come la navigazione: appena aperto il tab (anche da un link diretto o dalla
 * ricerca) il focus è ancora sulla pagina, non dentro il tab, e ↓ deve funzionare senza un clic prima.
 * Niente se il tab è nascosto, con Ctrl/⌘/Alt, con l'aiuto dei comandi aperto, mentre si scrive, in una
 * finestra di dialogo (la ricerca) o sul pad. Negli altri pannelli della sidebar (Azioni, Viste, Fonti…)
 * le frecce scorrono quel pannello.
 *
 *   installStepKeys({ root, ui, menu: () => tab.menu, pathPane });
 */
const NOT_FOR_US = "input, textarea, select, [contenteditable], [role='dialog'], dialog, .nav-pad, .nav-help";

/**
 * @param {object} o
 * @param {HTMLElement} o.root radice del tab
 * @param {{nav: import("./navigation.js").Navigation, player: import("./ui/player.js").ActionPlayer | null}} o.ui
 * @param {() => {move(delta: number): void} | null} o.menu il Menu del Percorso (funzione: può essere ricostruito)
 * @param {HTMLElement} [o.pathPane] il pannello che contiene il menu
 * @param {string} [o.spaceIgnore] selettore dove Spazio resta al pulsante (es. pulsanti in sovrimpressione del tab)
 * @returns {() => void} rimuove l'ascolto
 */
export function installStepKeys({ root, ui, menu, pathPane = null, spaceIgnore = "" }) {
  root.tabIndex = -1; // un clic su una parte non interattiva del tab gli dà il focus
  const onKey = (e) => {
    if (root.hidden || !root.isConnected || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || ui.nav?.modalOpen) return;
    const target = e.target instanceof Element ? e.target : document.body;
    if (target !== document.body && target !== document.documentElement && !root.contains(target)) return;
    if (target.closest(NOT_FOR_US)) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (e.repeat || (pathPane && target.closest(".tab-sidebar") && !pathPane.contains(target))) return;
      const m = menu();
      if (!m) return;
      e.preventDefault(); // la pagina non scorre
      m.move(e.key === "ArrowDown" ? +1 : -1);
    } else if (e.key === " " && ui.player?.action && !(spaceIgnore && target.closest(spaceIgnore))) {
      e.preventDefault(); // niente clic sul pulsante che ha il focus ("▶ Avvia" non riparte) né scorrimento
      if (!e.repeat) ui.player.toggle(); // tenuto premuto: una sola pausa, non un lampeggio
    }
  };
  window.addEventListener("keydown", onKey);
  return () => window.removeEventListener("keydown", onKey);
}
