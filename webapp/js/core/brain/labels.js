/**
 * Etichette HTML con linee guida per la scena del cervello (regioni, percorsi, colonne del
 * microcircuito): come nelle tavole di anatomia, un punto sulla struttura, una linea sottile e il nome
 * fuori, dove non copre niente.
 *
 *   const labels = new LabelOverlay(layer, { onPick: (item) => … });
 *   labels.set([{ id, text, sub, color, anchors: [[x, y, z], …], priority, direction }]);
 *   labels.update(camera, width, height, { center, state: (item, anchor) => "on" | "dim" | "off" });
 *
 * Perché HTML e non sprite: testo nitido a ogni zoom e con gli accenti, pulsanti veri (clic = scheda),
 * e poche decine di etichette costano pochissimo. update() va chiamato solo quando la camera si muove
 * (o cambiano le etichette): proietta i punti, evita le sovrapposizioni (prima le più importanti) e
 * scrive solo transform e coordinate delle linee. Nessun lavoro a riposo.
 */
import { el } from "../dom.js";

const SVG = "http://www.w3.org/2000/svg";
const MARGIN = 4; // px tra due etichette
const DISTANCES = [34, 56, 82, 112];
const ANGLES = [0, 0.38, -0.38, 0.8, -0.8, 1.25, -1.25];

export class LabelOverlay {
  /**
   * @param {HTMLElement} layer dove vivono le etichette (sopra il canvas)
   * @param {{ onPick?: (item) => void, className?: string }} [o]
   */
  constructor(layer, { onPick = () => {}, className = "" } = {}) {
    this.layer = layer;
    this.onPick = onPick;
    this.svg = document.createElementNS(SVG, "svg");
    this.svg.setAttribute("class", `br-leaders ${className}`.trim());
    this.svg.setAttribute("aria-hidden", "true");
    // le etichette ripetono ciò che la scheda elenca già con pulsanti accessibili: fuori dal tab order
    this.root = el("div", { className: `br-labels ${className}`.trim(), ariaHidden: "true" });
    layer.append(this.svg, this.root);
    this.items = [];
    this.visible = true;
  }

  /**
   * @param {{ id: string, text: string, sub?: string, color?: string, anchors: number[][], priority?: number,
   *   direction?: "auto" | "up", data?: any }[]} items priority: 0 = la più importante
   */
  set(items) {
    for (const item of this.items) {
      item.button.remove();
      item.line.remove();
      item.dot.remove();
    }
    this.items = items.map((spec) => {
      const button = el(
        "button",
        { type: "button", className: "br-label", tabIndex: -1, onclick: () => this.onPick(spec) },
        el("span", { className: "br-label-text" }, spec.text),
        spec.sub ? el("small", { className: "br-label-sub" }, spec.sub) : null,
      );
      button.style.setProperty("--c", spec.color ?? "#e2e8f0"); // proprietà personalizzata: niente Object.assign
      const line = document.createElementNS(SVG, "line");
      line.setAttribute("stroke", spec.color ?? "#e2e8f0");
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("r", "3.2");
      dot.setAttribute("fill", spec.color ?? "#e2e8f0");
      this.svg.append(line, dot);
      this.root.append(button);
      return { ...spec, priority: spec.priority ?? 2, button, line, dot, size: null, shown: null };
    });
  }

  setVisible(on) {
    this.visible = on;
    this.root.hidden = !on;
    this.svg.style.display = on ? "" : "none";
  }

  /**
   * Ricalcola le posizioni. `state(item, anchor)` dice se l'etichetta va mostrata ("on"), attenuata
   * perché il punto è nascosto dietro qualcosa ("dim") o tolta ("off"); `pick(item, anchors)` sceglie
   * quale dei punti usare (es. l'emisfero più vicino alla camera). `obstacles`: rettangoli
   * [x0, y0, x1, y1] dello stage coperti da comandi DOM (scheda, pad, selettore) dove nessuna etichetta
   * va a finire: sotto la scheda o il pad sarebbe tagliata e terrebbe il posto a un'altra.
   */
  update(camera, width, height, { center = null, state = () => "on", pick = null, obstacles = [] } = {}) {
    if (!this.visible || !this.items.length) return;
    const project = (p) => {
      const v = [p[0], p[1], p[2], 1];
      const e = camera.matrixWorldInverse.elements;
      const x = e[0] * v[0] + e[4] * v[1] + e[8] * v[2] + e[12];
      const y = e[1] * v[0] + e[5] * v[1] + e[9] * v[2] + e[13];
      const z = e[2] * v[0] + e[6] * v[1] + e[10] * v[2] + e[14];
      const pe = camera.projectionMatrix.elements;
      const cx = pe[0] * x + pe[4] * y + pe[8] * z + pe[12];
      const cy = pe[1] * x + pe[5] * y + pe[9] * z + pe[13];
      const cw = pe[3] * x + pe[7] * y + pe[11] * z + pe[15];
      if (cw <= 0.01) return null;
      return [((cx / cw) * 0.5 + 0.5) * width, (1 - ((cy / cw) * 0.5 + 0.5)) * height];
    };
    const c = center ? project(center) : [width / 2, height / 2];
    const centerScreen = c ?? [width / 2, height / 2];
    const order = [...this.items].sort((a, b) => a.priority - b.priority);
    const rects = obstacles.filter((o) => o[2] > o[0] && o[3] > o[1]);
    for (const item of order) {
      const anchor = pick ? pick(item) : item.anchors[0];
      const mode = anchor ? state(item, anchor) : "off";
      const p = mode !== "off" ? project(anchor) : null;
      if (!p || p[0] < -20 || p[1] < -20 || p[0] > width + 20 || p[1] > height + 20) {
        this.hide(item);
        continue;
      }
      if (!item.size) item.size = [item.button.offsetWidth || 90, item.button.offsetHeight || 22];
      const [w, h] = item.size;
      let dx = p[0] - centerScreen[0];
      let dy = p[1] - centerScreen[1];
      const len = Math.hypot(dx, dy);
      if (item.direction === "up" || item.direction === "down") {
        dx = 0;
        dy = item.direction === "up" ? -1 : 1;
      } else if (len < 18) {
        dx = 0.7;
        dy = -0.7;
      } else {
        dx /= len;
        dy /= len;
      }
      let chosen = null;
      search: for (const dist of DISTANCES) {
        for (const ang of item.direction === "up" || item.direction === "down" ? [0, 0.5, -0.5] : ANGLES) {
          const ca = Math.cos(ang);
          const sa = Math.sin(ang);
          const ux = dx * ca - dy * sa;
          const uy = dx * sa + dy * ca;
          const tx = p[0] + ux * dist;
          const ty = p[1] + uy * dist;
          // il punto d'aggancio dell'etichetta è il lato rivolto verso la struttura
          const left = ux >= 0.25 ? tx : ux <= -0.25 ? tx - w : tx - w / 2;
          const top = uy >= 0.25 ? ty : uy <= -0.25 ? ty - h : ty - h / 2;
          const rect = [left, top, left + w, top + h];
          if (rect[0] < 4 || rect[1] < 4 || rect[2] > width - 4 || rect[3] > height - 4) continue;
          if (rects.some((r) => rect[0] < r[2] + MARGIN && r[0] < rect[2] + MARGIN && rect[1] < r[3] + MARGIN && r[1] < rect[3] + MARGIN)) continue;
          chosen = rect;
          break search;
        }
      }
      if (!chosen) {
        this.hide(item);
        continue;
      }
      rects.push(chosen);
      this.show(item, p, chosen, mode === "dim");
    }
  }

  show(item, p, rect, dim) {
    const { button, line, dot } = item;
    button.style.transform = `translate(${Math.round(rect[0])}px, ${Math.round(rect[1])}px)`;
    if (item.shown !== true) {
      button.hidden = false;
      line.style.display = "";
      dot.style.display = "";
      item.shown = true;
    }
    button.classList.toggle("dim", dim);
    // la linea va dal punto al bordo più vicino dell'etichetta
    const x = Math.min(Math.max(p[0], rect[0]), rect[2]);
    const y = Math.min(Math.max(p[1], rect[1]), rect[3]);
    line.setAttribute("x1", p[0].toFixed(1));
    line.setAttribute("y1", p[1].toFixed(1));
    line.setAttribute("x2", x.toFixed(1));
    line.setAttribute("y2", y.toFixed(1));
    line.setAttribute("class", dim ? "dim" : "");
    dot.setAttribute("cx", p[0].toFixed(1));
    dot.setAttribute("cy", p[1].toFixed(1));
    dot.setAttribute("class", dim ? "dim" : "");
  }

  hide(item) {
    if (item.shown === false) return;
    item.button.hidden = true;
    item.line.style.display = "none";
    item.dot.style.display = "none";
    item.shown = false;
  }

  /** L'etichetta di una voce (per evidenziarla, es. la regione scelta). */
  mark(id) {
    for (const item of this.items) item.button.classList.toggle("active", item.id === id);
  }

  dispose() {
    this.svg.remove();
    this.root.remove();
    this.items = [];
  }
}
