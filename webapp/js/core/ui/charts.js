/**
 * SEGNAPOSTO — implementazione completa a cura del workflow A (vedi ARCHITECTURE.md, "ChartsPanel").
 * Espone già l'API definitiva, così i tab possono usarla: i metodi qui non disegnano nulla.
 */
import { el } from "../dom.js";

class ChartStub {
  setData() {}
  append() {}
  clear() {}
  setHighlight() {}
}

class SectionStub {
  chart() {
    return new ChartStub();
  }
  clear() {}
}

export class ChartsPanel {
  constructor(parent, { title = "Grafici" } = {}) {
    this.button = el("button", { type: "button", className: "charts-fab", onclick: () => this.toggle() }, "📈 ", title);
    this.root = el("aside", { className: "charts-panel", hidden: true });
    parent.append(this.button, this.root);
  }
  section() {
    return new SectionStub();
  }
  open() {}
  close() {}
  toggle() {}
  coveredWidth() {
    return 0;
  }
}
