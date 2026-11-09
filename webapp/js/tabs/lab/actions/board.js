/**
 * La "lavagna" delle azioni: un riquadro DOM sopra la barra del player con il calcolo del passo scritto
 * con i numeri veri (formula, prodotti, somme, probabilità, la curva della loss). Perché DOM e non un
 * pannello 3D: testo nitido a ogni zoom, leggibile dagli screen reader (aria-live), a capo da solo sul
 * telefono; la scena 3D resta libera per le matrici (la camera inquadra la parte di stage sopra la lavagna).
 *
 * Il contenuto è una lista di blocchi di soli dati, scritti sempre con textContent / createElement:
 *   { title, kicker? }                       titolo del calcolo
 *   { line: [segmento…] }                    una riga di testo; segmento = "testo" | { text, tone }
 *   { formula: [segmento…] }                 una riga in monospazio (va a capo se serve)
 *   { vector: { values, labels?, mark?, range?, colormap?, max? } }   celle colorate con i numeri
 *   { bars: [{ label, value, mark? }], max? }                         barre (probabilità)
 *   { chart: { series, marker?, xMax, yMin, yMax, xLabel?, yLabel? } } curva (SVG)
 *   { table: { head: [...], rows: [[...]] } }                         tabellina di confronto
 *   { note: "testo" }                        nota piccola (onestà: cosa è vero, cosa è approssimato)
 * tone: num | pos | neg | muted | accent | good | bad | code
 */
import { colormapLUT } from "../../../core/matrixNumbers.js";
import { el } from "../../../core/dom.js";
import { quantizeSigned, quantizeUnsigned, textColorFor } from "../../../shared/weights/display.js";
import { num } from "./format.js";

const SVG = "http://www.w3.org/2000/svg";
const TONES = new Set(["num", "pos", "neg", "muted", "accent", "good", "bad", "code"]);
const SERIES_COLORS = ["#38bdf8", "#fbbf24", "#f472b6", "#4ade80"];

export class Board {
  /** @param {HTMLElement} parent lo stage del tab */
  constructor(parent) {
    this.body = el("div", { className: "lab-act-board-body" });
    this.toggle = el("button", { type: "button", className: "lab-act-board-toggle", ariaExpanded: "true", ariaLabel: "Riduci o espandi la lavagna", onclick: () => this.setCollapsed(!this.collapsed, { user: true }) }, "▾");
    this.heading = el("strong", { className: "lab-act-board-title" });
    this.root = el(
      "section",
      { className: "lab-act-board", hidden: true, ariaLabel: "Lavagna: il calcolo del passo con i numeri veri", ariaLive: "polite" },
      el("header", { className: "lab-act-board-head" }, this.heading, this.toggle),
      this.body,
    );
    this.collapsed = false;
    parent.append(this.root);
  }

  /** Mostra un passo: `blocks` come sopra (il primo { title } va nell'intestazione). */
  show(blocks = []) {
    const list = blocks.filter(Boolean);
    const head = list[0]?.title !== undefined ? list.shift() : null;
    this.heading.replaceChildren(...(head?.kicker ? [el("span", { className: "lab-act-kicker" }, head.kicker), " "] : []), head?.title ?? "");
    this.body.replaceChildren(...list.map(renderBlock).filter(Boolean));
    this.root.hidden = false;
  }

  hide() {
    this.root.hidden = true;
  }

  /** Aggiorna un pezzo "vivo" della lavagna ({ live: chiave } in una riga) senza ridisegnare il resto. */
  setLive(key, text) {
    const node = this.body.querySelector(`[data-live="${key}"]`);
    if (node && node.textContent !== text) node.textContent = text;
  }

  /** Ridotta: resta l'intestazione (sul telefono la scena ha bisogno di spazio). `user`: l'ha scelto lo studente. */
  setCollapsed(on, { user = false } = {}) {
    this.collapsed = Boolean(on);
    this.root.classList.toggle("collapsed", this.collapsed);
    this.toggle.setAttribute("aria-expanded", String(!this.collapsed));
    this.toggle.textContent = this.collapsed ? "▴" : "▾";
    this.onToggle?.(this.collapsed, user);
  }

  get visible() {
    return !this.root.hidden;
  }

  dispose() {
    this.root.remove();
  }
}

function segment(s) {
  if (s === null || s === undefined || s === false) return null;
  if (typeof s === "string" || typeof s === "number") return document.createTextNode(String(s));
  const tone = TONES.has(s.tone) ? ` lab-act-${s.tone}` : "";
  // la chiave di un pezzo "vivo" è nostra (mai testo dello studente): solo lettere e cifre
  const live = typeof s.live === "string" && /^[a-z0-9-]+$/i.test(s.live) ? { dataset: { live: s.live } } : {};
  return el(s.tone === "code" ? "code" : "span", { className: `lab-act-seg${tone}`, ...live }, String(s.text ?? ""));
}

function renderBlock(b) {
  if (b.line) return el("p", { className: "lab-act-line" }, ...b.line.map(segment).filter(Boolean));
  if (b.formula) return el("p", { className: "lab-act-formula" }, ...b.formula.map(segment).filter(Boolean));
  if (b.note) return el("p", { className: "lab-act-note" }, b.note);
  if (b.vector) return renderVector(b.vector);
  if (b.bars) return renderBars(b.bars, b.max);
  if (b.chart) return renderChart(b.chart);
  if (b.table) return renderTable(b.table);
  return null;
}

/** Celle colorate con i numeri (al più `max`): gli stessi colori della scena (colormap di core/visuals.js). */
function renderVector({ values, labels = null, mark = null, range = null, colormap = "diverging", max = 24, digits = 2, caption = "" }) {
  const n = values.length;
  const lut = colormapLUT(colormap);
  let scale = range;
  if (!(scale > 0)) {
    scale = 0;
    for (let i = 0; i < n; i++) if (Number.isFinite(values[i])) scale = Math.max(scale, Math.abs(values[i]));
    scale ||= 1;
  }
  const shown = Math.min(n, max);
  const cells = [];
  for (let i = 0; i < shown; i++) {
    const v = values[i];
    const known = Number.isFinite(v);
    const q = known ? (colormap === "attention" ? quantizeUnsigned(v, scale) : quantizeSigned(v, scale)) : 0;
    const [r, g, bl] = [lut[4 * q], lut[4 * q + 1], lut[4 * q + 2]];
    cells.push(
      el(
        "span",
        {
          className: `lab-act-cell${mark === i ? " marked" : ""}${known ? "" : " empty"}`,
          style: known ? { background: `rgb(${r}, ${g}, ${bl})`, color: textColorFor(r, g, bl) } : {},
          ...(labels ? { title: `${labels[i]}: ${known ? num(v, 6) : "da calcolare"}` } : {}),
        },
        labels ? el("small", {}, labels[i]) : null,
        known ? num(v, digits) : "·",
      ),
    );
  }
  return el("div", { className: "lab-act-vector-box" }, caption ? el("p", { className: "lab-act-note" }, caption) : null, el("div", { className: "lab-act-vector" }, ...cells), n > shown ? el("p", { className: "lab-act-note" }, `… altri ${n - shown} numeri (tutti nella scena)`) : null);
}

/** Barre orizzontali (probabilità): etichetta, barra, valore. */
function renderBars(bars, max = null) {
  const top = max ?? Math.max(1e-9, ...bars.map((b) => b.value));
  return el(
    "div",
    { className: "lab-act-bars", role: "list" },
    ...bars.map((b) =>
      el(
        "div",
        { className: `lab-act-bar${b.mark ? " marked" : ""}`, role: "listitem" },
        el("span", { className: "lab-act-bar-label" }, b.label),
        el("span", { className: "lab-act-bar-track", ariaHidden: "true" }, el("span", { className: "lab-act-bar-fill", style: { width: `${Math.max(0.5, Math.min(100, (b.value / top) * 100))}%` } })),
        el("span", { className: "lab-act-bar-value" }, b.text ?? num(b.value, 3)),
      ),
    ),
  );
}

/** Una curva in SVG (la loss del training): assi essenziali, serie, linea "siamo qui". */
function renderChart({ series, marker = null, xMax, yMin, yMax, xLabel = "", yLabel = "", height = 120, label = "" }) {
  const W = 560;
  const H = height;
  const pad = { l: 40, r: 10, t: 8, b: 22 };
  const x = (v) => pad.l + (v / (xMax || 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "lab-act-chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", label || `${yLabel} per ${xLabel}`);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };
  add("line", { x1: pad.l, y1: H - pad.b, x2: W - pad.r, y2: H - pad.b, class: "axis" });
  add("line", { x1: pad.l, y1: pad.t, x2: pad.l, y2: H - pad.b, class: "axis" });
  for (const v of [yMin, (yMin + yMax) / 2, yMax]) add("text", { x: pad.l - 5, y: y(v) + 4, "text-anchor": "end", class: "tick" }, num(v, 1));
  add("text", { x: W - pad.r, y: H - 6, "text-anchor": "end", class: "tick" }, xLabel);
  add("text", { x: pad.l, y: H - 6, "text-anchor": "start", class: "tick" }, "0");
  series.forEach((s, k) => {
    if (!s.points.length) return;
    const d = s.points.map(([px, py], i) => `${i ? "L" : "M"}${x(px).toFixed(1)},${y(py).toFixed(1)}`).join("");
    add("path", { d, fill: "none", stroke: s.color ?? SERIES_COLORS[k % SERIES_COLORS.length], "stroke-width": s.width ?? 2, "stroke-linejoin": "round", opacity: s.opacity ?? 1 });
  });
  if (marker !== null && Number.isFinite(marker)) {
    add("line", { x1: x(marker), y1: pad.t, x2: x(marker), y2: H - pad.b, class: "marker" });
  }
  const legend = el("p", { className: "lab-act-legend" }, ...series.filter((s) => s.name).map((s, k) => el("span", {}, el("i", { style: { background: s.color ?? SERIES_COLORS[k % SERIES_COLORS.length] } }), s.name)));
  return el("div", { className: "lab-act-chart-box" }, svg, legend);
}

function renderTable({ head = [], rows = [] }) {
  return el(
    "table",
    { className: "lab-act-table" },
    head.length ? el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { scope: "col" }, h)))) : null,
    el("tbody", {}, ...rows.map((r) => el("tr", {}, ...r.map((c, i) => el(i === 0 ? "th" : "td", i === 0 ? { scope: "row" } : {}, c))))),
  );
}
