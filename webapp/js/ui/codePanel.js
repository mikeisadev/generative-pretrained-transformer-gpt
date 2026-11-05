/**
 * Finestra fluttuante con il frammento di codice Python dello step selezionato.
 * Il codice arriva da data/model.json, cioè dal sorgente vero del pacchetto `gpt`.
 */
import hljs from "../../vendor/highlight/core.min.js";
import python from "../../vendor/highlight/python.min.js";

hljs.registerLanguage("python", python);

export class CodePanel {
  constructor(root, snippets) {
    this.root = root;
    this.snippets = snippets;
    this.title = root.querySelector("[data-code-title]");
    this.file = root.querySelector("[data-code-file]");
    this.gutter = root.querySelector("[data-code-gutter]");
    this.code = root.querySelector("[data-code-body]");
    this.marks = root.querySelector("[data-code-marks]");
    root.querySelector("[data-code-toggle]").addEventListener("click", () => this.root.classList.toggle("minimized"));
  }

  /**
   * @param {string} key chiave del frammento (vedi gpt/export.py → SNIPPETS)
   * @param {string[]} highlight le righe che contengono uno di questi testi vengono evidenziate
   */
  show(key, title, highlight = []) {
    const snippet = this.snippets[key];
    if (!snippet) {
      this.title.textContent = title;
      this.file.textContent = "";
      this.code.textContent = `# frammento '${key}' non trovato in model.json`;
      return;
    }
    const lines = snippet.code.split("\n");
    this.title.textContent = title;
    this.file.textContent = `${snippet.file}:${snippet.start_line}`;
    this.code.innerHTML = hljs.highlight(snippet.code, { language: "python" }).value;
    this.gutter.textContent = lines.map((_, i) => snippet.start_line + i).join("\n");

    // Evidenziazione per riga con un livello sotto il codice: non spezza gli span di highlight.js.
    this.marks.replaceChildren(
      ...lines.flatMap((line, i) =>
        highlight.some((h) => line.includes(h)) ? [Object.assign(document.createElement("div"), { style: `top: calc(${i} * var(--code-line))` })] : [],
      ),
    );
    const first = lines.findIndex((line) => highlight.some((h) => line.includes(h)));
    const scroller = this.root.querySelector(".code-scroll");
    const lineHeight = parseFloat(getComputedStyle(this.code).lineHeight) || 18;
    scroller.scrollTo({ top: first > 3 ? (first - 3) * lineHeight : 0, behavior: "smooth" });
    this.root.classList.remove("minimized");
  }
}
