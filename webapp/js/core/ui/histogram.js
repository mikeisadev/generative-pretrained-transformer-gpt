/**
 * Istogramma dei valori di un tensore su canvas: una serie (barre) o più serie sovrapposte con gli
 * stessi intervalli (confronto A/B: contorno a gradini e velatura leggera), assi e tacche leggibili,
 * legenda con più serie, riassunto per gli screen reader e suggerimento al passaggio del mouse (o con
 * le frecce, dopo Tab) con l'intervallo e i conteggi di ogni serie.
 *
 *   const h = new Histogram(container, { bins: 60, height: 120, label: "Valori di lm_head.weight" });
 *   h.setData([{ label: "step 0", data: a }, { label: "step 5000", data: b }]);   // o una serie sola
 *   h.dispose();
 *
 * Canvas nitido (devicePixelRatio), ridisegnato solo quando cambiano dati, dimensioni o l'intervallo
 * indicato: niente lavoro per frame. Colori e testi dei grafici dell'app (chartMath.js: palette dei
 * grafici in ordine fisso, testo mai del colore della serie). La logica pura è in histogramMath.js.
 */
import { el } from "../dom.js";
import { THEME, formatTick } from "./chartMath.js";
import { binAt, binLabel, binRows, countHeight, countText, histogramModel, summaryText, valuesText } from "./histogramMath.js";

const FONT = "11px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
const PAD = { top: 8, right: 10, bottom: 20 };
const WASH_ALPHA = 0.16; // velatura delle serie sovrapposte: si vedono entrambe

let instances = 0;

export class Histogram {
  /**
   * @param {HTMLElement} container
   * @param {{bins?: number, height?: number, label?: string, logY?: boolean}} [o]
   *   logY: conteggi in scala logaritmica (log(1 + n)): utile quando pochi valori grandi spariscono
   *   accanto al picco attorno allo zero
   */
  constructor(container, { bins = 60, height = 120, label = "", logY = false } = {}) {
    if (!container?.append) throw new TypeError("Histogram: serve un elemento contenitore");
    this.id = `hist${++instances}`;
    this.options = { bins, height, label, logY };
    this.model = null;
    this.hover = -1;
    this.keyboard = false;
    this.raf = 0;
    this.disposed = false;
    this.width = 0;

    this.canvas = el("canvas", { className: "hist-canvas" });
    this.canvas.setAttribute("aria-hidden", "true");
    this.tip = el("div", { className: "hist-tip", hidden: true });
    this.live = el("p", { className: "hist-live", id: `${this.id}-live`, ariaLive: "polite" });
    this.plot = el("div", { className: "hist-plot", role: "img", tabIndex: 0 }, this.canvas, this.tip);
    this.plot.style.height = `${height}px`;
    this.plot.setAttribute("aria-describedby", this.live.id);
    this.title = el("figcaption", { className: "hist-title", hidden: !label }, label);
    this.legend = el("ul", { className: "hist-legend", hidden: true });
    this.root = el("figure", { className: "hist" }, this.title, this.plot, this.legend, this.live);
    container.append(this.root);

    this.onMove = (e) => this.pointer(e);
    this.onLeave = () => this.setHover(-1);
    this.onKey = (e) => this.key(e);
    this.onBlur = () => this.setHover(-1);
    this.plot.addEventListener("pointermove", this.onMove);
    this.plot.addEventListener("pointerleave", this.onLeave);
    this.plot.addEventListener("keydown", this.onKey);
    this.plot.addEventListener("blur", this.onBlur);
    this.resize = typeof ResizeObserver === "function" ? new ResizeObserver(() => this.schedule()) : null;
    this.resize?.observe(this.plot);
    this.setData([]);
  }

  /**
   * Una serie o più serie sovrapposte con gli stessi intervalli (calcolati sull'unione dei valori).
   * @param {{label?: string, data: ArrayLike<number>, color?: string}[]|{label?: string, data: ArrayLike<number>, color?: string}} series
   */
  setData(series) {
    if (this.disposed) return;
    const list = Array.isArray(series) ? series : series ? [series] : [];
    this.model = histogramModel(list, { bins: this.options.bins, logY: this.options.logY });
    this.hover = -1;
    this.tip.hidden = true;
    this.plot.setAttribute("aria-label", summaryText(this.model, { label: this.options.label }));
    this.renderLegend();
    this.schedule();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resize?.disconnect();
    this.plot.removeEventListener("pointermove", this.onMove);
    this.plot.removeEventListener("pointerleave", this.onLeave);
    this.plot.removeEventListener("keydown", this.onKey);
    this.plot.removeEventListener("blur", this.onBlur);
    this.root.remove();
    this.model = null;
  }

  // ---------------------------------------------------------------- disegno

  schedule() {
    if (this.raf || this.disposed) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  renderLegend() {
    const series = this.model?.series ?? [];
    // Una serie sola non ha bisogno di legenda: il titolo dice già cosa si vede.
    this.legend.hidden = series.length < 2;
    this.legend.replaceChildren(
      ...(series.length < 2
        ? []
        : series.map((s) => {
            const key = el("i", { className: "hist-key" });
            key.style.setProperty("--c", s.color);
            return el("li", {}, key, el("span", {}, s.label), el("small", {}, ` ${valuesText(s.total)}`));
          })),
    );
  }

  /** Geometria del grafico in px CSS: area dei dati e scala x. */
  frame(width, height) {
    const m = this.model;
    const ctx = this.ctx;
    ctx.font = FONT;
    let labelW = 0;
    for (const t of m.yTicks) labelW = Math.max(labelW, ctx.measureText(countText(t, { compact: true })).width);
    const left = Math.ceil(labelW) + 8;
    const plotW = Math.max(10, width - left - PAD.right);
    const plotH = Math.max(10, height - PAD.top - PAD.bottom);
    const x = (v) => left + ((v - m.min) / (m.max - m.min)) * plotW;
    return { left, top: PAD.top, plotW, plotH, bottom: PAD.top + plotH, x };
  }

  draw() {
    if (this.disposed || !this.model) return;
    const width = Math.round(this.plot.clientWidth);
    const height = this.options.height;
    if (width < 40) return; // nascosto (pannello chiuso): si disegna quando torna visibile
    const dpr = Math.min(3, globalThis.devicePixelRatio || 1);
    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    }
    this.width = width;
    const ctx = (this.ctx ??= this.canvas.getContext("2d"));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const m = this.model;
    const f = (this.geom = this.frame(width, height));
    const crisp = (v) => Math.round(v) + 0.5;

    // griglia orizzontale e tacche dei conteggi (a sinistra), sotto i dati
    ctx.font = FONT;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    ctx.lineWidth = 1;
    let lastLabelY = Infinity;
    for (const t of m.yTicks) {
      const y = f.bottom - countHeight(t, m) * f.plotH;
      ctx.strokeStyle = t === 0 ? THEME.axis : THEME.grid;
      ctx.beginPath();
      ctx.moveTo(f.left, crisp(y));
      ctx.lineTo(f.left + f.plotW, crisp(y));
      ctx.stroke();
      if (lastLabelY - y < 12) continue; // in scala log "1" sta appena sopra "0": la linea resta, l'etichetta no
      ctx.fillStyle = THEME.muted;
      ctx.fillText(countText(t, { compact: true }), f.left - 5, y);
      lastLabelY = y;
    }

    // intervallo indicato (mouse o tastiera): una fascia chiara dietro le barre
    const bw = f.plotW / m.bins;
    if (this.hover >= 0) {
      ctx.fillStyle = "rgba(226, 232, 240, 0.08)";
      ctx.fillRect(f.left + this.hover * bw, f.top, bw, f.plotH);
    }

    if (!m.empty) {
      if (m.series.length === 1) this.drawBars(ctx, f, m.series[0], bw);
      else for (const s of m.series) this.drawStep(ctx, f, s, bw);
    }

    // lo zero dei pesi: una linea sottile se cade dentro l'intervallo
    if (!m.empty && m.min < 0 && m.max > 0) {
      const zx = crisp(f.x(0));
      ctx.strokeStyle = "rgba(148, 163, 184, 0.55)";
      ctx.beginPath();
      ctx.moveTo(zx, f.top);
      ctx.lineTo(zx, f.bottom);
      ctx.stroke();
    }

    // tacche dei valori (in basso)
    ctx.textBaseline = "top";
    ctx.fillStyle = THEME.muted;
    let lastRight = -Infinity;
    for (const t of m.xTicks.ticks) {
      const x = f.x(t);
      const text = formatTick(t, m.xTicks.step, Math.max(Math.abs(m.min), Math.abs(m.max))).replace("-", "−");
      const w = ctx.measureText(text).width;
      const lx = Math.min(width - w / 2 - 1, Math.max(w / 2 + 1, x));
      if (lx - w / 2 < lastRight + 6) continue; // etichette che si toccherebbero: si salta
      ctx.strokeStyle = THEME.axis;
      ctx.beginPath();
      ctx.moveTo(crisp(x), f.bottom);
      ctx.lineTo(crisp(x), f.bottom + 3);
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(text, lx, f.bottom + 5);
      lastRight = lx + w / 2;
    }
    if (m.logY) {
      // in alto a destra: a sinistra c'è l'etichetta della tacca più alta
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.fillText("conteggi in scala log", f.left + f.plotW, 0);
    }
  }

  /** Una serie: barre dal basso, separate da 1 px dello sfondo quando c'è spazio, angoli in cima arrotondati. */
  drawBars(ctx, f, s, bw) {
    ctx.fillStyle = s.color;
    const gap = bw >= 4 ? 1 : 0;
    const radius = Math.min(2, (bw - gap) / 2);
    for (let b = 0; b < s.counts.length; b++) {
      const n = s.counts[b];
      if (!n) continue;
      const h = Math.max(1, countHeight(n, this.model) * f.plotH);
      const x = f.left + b * bw + gap / 2;
      const w = Math.max(1, bw - gap);
      const y = f.bottom - h;
      ctx.beginPath();
      if (ctx.roundRect && h > radius * 2) ctx.roundRect(x, y, w, h, [radius, radius, 0, 0]);
      else ctx.rect(x, y, w, h);
      ctx.fill();
    }
  }

  /** Più serie: contorno a gradini di 2 px e velatura leggera, così le forme si confrontano senza coprirsi. */
  drawStep(ctx, f, s, bw) {
    ctx.beginPath();
    ctx.moveTo(f.left, f.bottom);
    for (let b = 0; b < s.counts.length; b++) {
      const y = f.bottom - countHeight(s.counts[b], this.model) * f.plotH;
      ctx.lineTo(f.left + b * bw, y);
      ctx.lineTo(f.left + (b + 1) * bw, y);
    }
    ctx.lineTo(f.left + f.plotW, f.bottom);
    ctx.closePath();
    ctx.globalAlpha = WASH_ALPHA;
    ctx.fillStyle = s.color;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  // ---------------------------------------------------------------- suggerimento (mouse e tastiera)

  pointer(e) {
    if (!this.model || this.model.empty || !this.geom) return;
    const rect = this.plot.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const f = this.geom;
    if (px < f.left || px > f.left + f.plotW) return this.setHover(-1);
    const value = this.model.min + ((px - f.left) / f.plotW) * (this.model.max - this.model.min);
    this.keyboard = false;
    this.setHover(binAt(this.model, value));
  }

  key(e) {
    if (!this.model || this.model.empty) return;
    const last = this.model.bins - 1;
    let next = null;
    if (e.key === "ArrowRight") next = this.hover < 0 ? 0 : Math.min(last, this.hover + 1);
    else if (e.key === "ArrowLeft") next = this.hover < 0 ? last : Math.max(0, this.hover - 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = last;
    else if (e.key === "Escape" && this.hover >= 0) next = -1;
    if (next === null) return;
    e.preventDefault(); // le frecce sono del grafico: la scena 3D non si sposta
    this.keyboard = true;
    this.setHover(next);
  }

  setHover(index) {
    if (index === this.hover) return;
    this.hover = index;
    this.schedule();
    if (index < 0 || !this.model) {
      this.tip.hidden = true;
      return;
    }
    const m = this.model;
    const rows = binRows(m, index);
    this.tip.replaceChildren(
      el("strong", {}, binLabel(m, index)),
      ...rows.map((r) => {
        const key = el("i", { className: "hist-tip-key" });
        key.style.setProperty("--c", r.color);
        return el("span", { className: "hist-tip-row" }, key, el("b", {}, r.text), el("small", {}, ` ${r.share}${m.series.length > 1 ? ` · ${r.label}` : ""}`));
      }),
    );
    this.tip.hidden = false;
    // accanto all'intervallo, dentro il riquadro (a sinistra se a destra non c'è posto)
    const f = this.geom;
    if (f) {
      const bw = f.plotW / m.bins;
      const center = f.left + (index + 0.5) * bw;
      const tipW = this.tip.offsetWidth || 160;
      const left = center + 12 + tipW > this.width ? Math.max(0, center - 12 - tipW) : center + 12;
      this.tip.style.left = `${Math.round(left)}px`;
    }
    if (this.keyboard) this.live.textContent = `${binLabel(m, index)}: ${rows.map((r) => `${m.series.length > 1 ? `${r.label} ` : ""}${r.text} (${r.share})`).join(", ")}`;
  }
}
