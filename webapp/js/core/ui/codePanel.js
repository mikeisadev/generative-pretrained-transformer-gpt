/**
 * Finestra fluttuante con un frammento di codice evidenziato (Python, Java, JS, Bash, JSON, XML).
 * Costruisce da sola il proprio DOM dentro `parent`.
 *
 *   const panel = new CodePanel(tabRoot);
 *   panel.show({ title, file, startLine, code, language: "python", highlight: ["masked_fill"] });
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

export class CodePanel {
  constructor(parent) {
    this.title = el("strong", {}, "Codice");
    this.file = el("span", { className: "muted mono" });
    this.gutter = el("pre", { className: "code-gutter" });
    this.code = el("code", { className: "hljs" });
    this.marks = el("div", { className: "code-marks" });
    this.scroller = el("div", { className: "code-scroll" }, this.gutter, el("div", { className: "code-area" }, this.marks, el("pre", {}, this.code)));
    const toggle = el("button", { className: "icon-button", type: "button", ariaLabel: "Riduci o espandi il codice", onclick: () => this.root.classList.toggle("minimized") }, "▁");
    this.root = el("section", { className: "code-panel", ariaLabel: "Codice dello step" }, el("header", {}, el("div", {}, this.title, this.file), toggle), this.scroller);
    parent.append(this.root);
    this.listeners = new Set();
  }

  /** Chiamato quando il pannello cambia dimensione (es. ridotto): la scena ricalcola il view offset. */
  onResize(fn) {
    new MutationObserver(fn).observe(this.root, { attributes: true, attributeFilter: ["class", "hidden"] });
  }

  /** Larghezza coperta sul lato destro dello stage (0 se ridotto, nascosto o su schermi stretti). */
  coveredWidth() {
    const wide = window.matchMedia("(min-width: 901px)").matches;
    if (!wide || this.root.hidden || this.root.classList.contains("minimized")) return 0;
    return this.root.offsetWidth + 14;
  }

  /**
   * @param {object} s
   * @param {string} s.title @param {string} [s.file] @param {number} [s.startLine=1]
   * @param {string} s.code @param {string} [s.language="python"]
   * @param {string[]} [s.highlight] le righe che contengono uno di questi testi vengono evidenziate
   */
  show({ title, file = "", startLine = 1, code, language = "python", highlight = [] }) {
    this.root.hidden = false;
    const lines = code.split("\n");
    this.title.textContent = title;
    this.file.textContent = file ? `${file}:${startLine}` : "";
    this.code.className = `hljs language-${language}`;
    this.code.innerHTML = hljs.getLanguage(language) ? hljs.highlight(code, { language }).value : hljs.highlightAuto(code).value;
    this.gutter.textContent = lines.map((_, i) => startLine + i).join("\n");
    // Evidenziazione per riga su un livello sotto il codice: non spezza gli span di highlight.js.
    this.marks.replaceChildren(
      ...lines.flatMap((line, i) => (highlight.some((h) => line.includes(h)) ? [el("div", { style: { top: `calc(${i} * var(--code-line))` } })] : [])),
    );
    const first = lines.findIndex((line) => highlight.some((h) => line.includes(h)));
    const lineHeight = parseFloat(getComputedStyle(this.code).lineHeight) || 18;
    this.scroller.scrollTo({ top: first > 3 ? (first - 3) * lineHeight : 0, behavior: "smooth" });
    this.root.classList.remove("minimized");
  }

  hide() {
    this.root.hidden = true;
  }
}
