/**
 * Finestra fluttuante con un frammento di codice evidenziato (Python, Java, JS, Bash, JSON, XML).
 * Costruisce da sola il proprio DOM dentro `parent`.
 *
 *   const panel = new CodePanel(tabRoot);
 *   panel.show({ title, file, startLine, code, language: "python", highlight: ["masked_fill"] });
 *   panel.minimize(); panel.expand(); panel.minimized;      // da codice (non è la scelta dello studente)
 *   panel.userMinimized; panel.userExpanded;                // l'ultima scelta dello studente con "▁"
 *   panel.onToggle((minimized) => …);                       // quando è lo studente a premere "▁"
 *
 * Il pulsante "▁" riduce il pannello alla sola intestazione. Con i grafici aperti (schermi larghi,
 * `followCharts`) il pannello si riduce da solo e si aggancia in fondo alla colonna dei grafici
 * (core.css): sotto i grafici ne restava una striscia tagliata di ~110 px; chiusi i grafici torna com'era.
 */
import hljs from "../../../vendor/highlight/core.min.js";
import bash from "../../../vendor/highlight/bash.min.js";
import java from "../../../vendor/highlight/java.min.js";
import javascript from "../../../vendor/highlight/javascript.min.js";
import json from "../../../vendor/highlight/json.min.js";
import python from "../../../vendor/highlight/python.min.js";
import xml from "../../../vendor/highlight/xml.min.js";
import { el } from "../dom.js";

for (const [name, grammar] of Object.entries({ python, java, javascript, bash, json, xml })) hljs.registerLanguage(name, grammar);

const WIDE = "(min-width: 901px)";

export class CodePanel {
  constructor(parent) {
    this.title = el("strong", {}, "Codice");
    this.file = el("span", { className: "muted mono" });
    this.gutter = el("pre", { className: "code-gutter" });
    this.code = el("code", { className: "hljs" });
    this.marks = el("div", { className: "code-marks" });
    this.scroller = el("div", { className: "code-scroll" }, this.gutter, el("div", { className: "code-area" }, this.marks, el("pre", {}, this.code)));
    // Le scelte dello studente con "▁": show() le rispetta (vedi show) e i tab possono leggerle.
    this.userMinimized = false;
    this.userExpanded = false;
    // Ridotto da followCharts perché i grafici aperti lo coprirebbero (non è una scelta dello studente).
    this.minimizedForCharts = false;
    this.chartsCover = () => false;
    this.toggleListeners = new Set();
    this.toggleButton = el(
      "button",
      { className: "icon-button", type: "button", ariaLabel: "Riduci o espandi il codice", ariaExpanded: "true", onclick: () => this.toggleByUser() },
      "▁",
    );
    this.root = el("section", { className: "code-panel", ariaLabel: "Codice dello step" }, el("header", {}, el("div", {}, this.title, this.file), this.toggleButton), this.scroller);
    parent.append(this.root);
    // aria-expanded segue la classe anche quando un tab la cambia direttamente (vecchia API documentata)
    this.classObserver = new MutationObserver(() => this.toggleButton.setAttribute("aria-expanded", String(!this.minimized)));
    this.classObserver.observe(this.root, { attributes: true, attributeFilter: ["class"] });
    this.listeners = new Set();
  }

  /** true se il pannello mostra solo l'intestazione. */
  get minimized() {
    return this.root.classList.contains("minimized");
  }

  /** Riduce il pannello alla sua intestazione (da codice: non cambia le scelte dello studente). */
  minimize() {
    this.root.classList.add("minimized");
  }

  /** Riapre il pannello (da codice). */
  expand() {
    this.root.classList.remove("minimized");
  }

  /** fn(minimized) quando lo studente preme "▁". Restituisce la funzione di rimozione. */
  onToggle(fn) {
    this.toggleListeners.add(fn);
    return () => this.toggleListeners.delete(fn);
  }

  /** "▁": la scelta dello studente vale per i frammenti seguenti, e vince sulla riduzione per i grafici. */
  toggleByUser() {
    const minimized = this.root.classList.toggle("minimized");
    this.userMinimized = minimized;
    this.userExpanded = !minimized;
    this.minimizedForCharts = false;
    for (const fn of this.toggleListeners) fn(minimized);
  }

  /**
   * Chiamato quando il pannello cambia dimensione o visibilità (ridotto, nascosto): la scena ricalcola il
   * view offset. Restituisce la funzione che smette di osservare.
   */
  onResize(fn) {
    const observer = new MutationObserver(fn);
    observer.observe(this.root, { attributes: true, attributeFilter: ["class", "hidden"] });
    return () => observer.disconnect();
  }

  /**
   * Con i grafici aperti su uno schermo largo il codice si riduce alla sua intestazione (che core.css aggancia
   * in fondo alla colonna dei grafici) e, chiusi i grafici, torna com'era, a meno che nel frattempo lo studente
   * non abbia scelto con "▁". Il fondo dei grafici lascia posto al codice agganciato: la sua altezza va nella
   * variabile CSS --code-dock-h del pannello dei grafici (a ogni cambio di dimensione, mai per frame).
   * Lo collega createTabLayout (core/layout.js). Restituisce la funzione di rimozione.
   * @param {import("./charts.js").ChartsPanel} charts
   */
  followCharts(charts) {
    const wide = window.matchMedia(WIDE);
    this.chartsCover = () => charts.isOpen && wide.matches;
    const dock = () => {
      const h = this.chartsCover() && !this.root.hidden ? this.root.offsetHeight : 0;
      if (h) charts.root.style.setProperty("--code-dock-h", `${h}px`);
      else charts.root.style.removeProperty("--code-dock-h");
    };
    const sync = () => {
      if (this.chartsCover()) {
        if (!this.root.hidden && !this.minimized) {
          this.minimize();
          this.minimizedForCharts = true;
        }
      } else if (this.minimizedForCharts) {
        this.minimizedForCharts = false;
        if (!this.root.hidden && !this.userMinimized) this.expand();
      }
      dock();
    };
    const off = charts.onResize(sync);
    wide.addEventListener("change", sync);
    const resize = typeof ResizeObserver === "function" ? new ResizeObserver(dock) : null;
    resize?.observe(this.root);
    return () => {
      off();
      wide.removeEventListener("change", sync);
      resize?.disconnect();
      charts.root.style.removeProperty("--code-dock-h");
      this.chartsCover = () => false;
    };
  }

  /**
   * Larghezza coperta sul lato destro dello stage, dal suo bordo sinistro al bordo destro della radice
   * (0 se ridotto, nascosto o su schermi stretti): di solito larghezza + 14 px; agganciato ai grafici aperti
   * (e riaperto con ▁), i 460 px della loro colonna.
   */
  coveredWidth() {
    const wide = window.matchMedia(WIDE).matches;
    if (!wide || this.root.hidden || this.minimized || !this.root.offsetWidth) return 0;
    const parent = this.root.parentElement;
    if (!parent) return this.root.offsetWidth + 14;
    return Math.max(0, Math.round(parent.getBoundingClientRect().right - this.root.getBoundingClientRect().left));
  }

  /**
   * @param {object} s
   * @param {string} s.title @param {string} [s.file] @param {number} [s.startLine=1]
   * @param {string} s.code @param {string} [s.language="python"]
   * @param {string[]} [s.highlight] le righe che contengono uno di questi testi vengono evidenziate
   */
  show({ title, file = "", startLine = 1, code, language = "python", highlight = [] }) {
    if (this.root.hidden) this.userMinimized = false; // riaperto da capo: si parte espansi
    this.root.hidden = false;
    const lines = code.split("\n");
    this.title.textContent = title;
    this.file.textContent = file ? `${file}:${startLine}` : "";
    this.code.className = `hljs language-${language}`;
    // highlight.js produce HTML con il codice ESCAPATO (i frammenti sono codice del corso, non testo dello studente)
    this.code.innerHTML = hljs.getLanguage(language) ? hljs.highlight(code, { language }).value : hljs.highlightAuto(code).value;
    this.gutter.textContent = lines.map((_, i) => startLine + i).join("\n");
    // Evidenziazione per riga su un livello sotto il codice: non spezza gli span di highlight.js.
    this.marks.replaceChildren(
      ...lines.flatMap((line, i) => (highlight.some((h) => line.includes(h)) ? [el("div", { style: { top: `calc(${i} * var(--code-line))` } })] : [])),
    );
    const first = lines.findIndex((line) => highlight.some((h) => line.includes(h)));
    const lineHeight = parseFloat(getComputedStyle(this.code).lineHeight) || 18;
    this.scroller.scrollTo({ top: first > 3 ? (first - 3) * lineHeight : 0, behavior: "smooth" });
    // Un pannello ridotto dal codice (es. durante un'azione sul telefono) si riapre al nuovo frammento;
    // uno ridotto a mano dallo studente resta ridotto: l'intestazione mostra comunque il nuovo titolo.
    // Con i grafici aperti resta (o diventa) ridotto accanto a loro, se lo studente non l'ha riaperto lì.
    if (this.chartsCover() && !this.userExpanded) {
      if (!this.minimized) {
        this.minimize();
        this.minimizedForCharts = true;
      }
    } else if (!this.userMinimized) this.expand();
  }

  hide() {
    this.root.hidden = true;
  }
}
