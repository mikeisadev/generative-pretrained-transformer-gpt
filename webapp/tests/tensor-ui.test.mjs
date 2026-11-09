// Test dei componenti condivisi per i numeri dei tensori: la logica pura (js/core/ui/tensorTableMath.js e
// histogramMath.js: forma della tabella, finestra virtuale con e senza compressione, tastiera, testi delle
// celle e delle statistiche, intervalli e testi dell'istogramma) e, in fondo, TensorTable su un DOM finto
// minimo (intestazioni dopo un cambio di tensore, virtualizzazione, tastiera, dispose). Disegno, focus
// visibile e contrasto si provano nel browser.
import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { binAt, binLabel, binRows, countHeight, countText, edgeDecimals, histogramModel, logCountTicks, seriesColor, summaryText, valuesText } from "../js/core/ui/histogramMath.js";
import { PALETTE } from "../js/core/ui/chartMath.js";
import { MAX_SCROLL_PX, axisWindow, cellText, describeCell, finiteAbsMax, headerText, moveActive, preciseText, scientificText, scrollToShow, statsRows, tableShape, widestText } from "../js/core/ui/tensorTableMath.js";
import { tensorStats } from "../js/shared/weights/stats.js";

// ---------------------------------------------------------------- forma

test("tabella: forma del tensore → righe × colonne (1D = una colonna, oltre 2D reshape(-1, ultima))", () => {
  assert.deepEqual(tableShape([256, 64]), [256, 64]);
  assert.deepEqual(tableShape([65, 64]), [65, 64]);
  assert.deepEqual(tableShape([65536]), [65536, 1]);
  assert.deepEqual(tableShape([2, 3, 4]), [6, 4]);
  assert.deepEqual(tableShape([]), [1, 1]);
  assert.deepEqual(tableShape(undefined), [1, 1]);
  assert.throws(() => tableShape([2, -1]), RangeError);
  assert.throws(() => tableShape([2.5]), RangeError);
});

// ---------------------------------------------------------------- finestra virtuale

test("finestra virtuale: solo le righe visibili più il margine, posizioni esatte", () => {
  // 256 righe da 24 px in una finestra di 286 px (12 righe circa), in cima
  let w = axisWindow({ count: 256, size: 24, viewport: 286, scroll: 0, overscan: 4 });
  assert.equal(w.content, 256 * 24);
  assert.equal(w.first, 0);
  assert.equal(w.last, Math.ceil(286 / 24) + 4);
  assert.equal(w.shift, 0);
  assert.equal(w.scale, 1);
  // a metà: la prima riga visibile è floor(scroll / 24)
  w = axisWindow({ count: 256, size: 24, viewport: 286, scroll: 2400, overscan: 4 });
  assert.equal(w.first, 100 - 4);
  assert.equal(w.last, Math.ceil((2400 + 286) / 24) + 4);
  // in fondo (anche con uno scroll oltre il limite): l'ultima riga è dentro la finestra
  w = axisWindow({ count: 256, size: 24, viewport: 286, scroll: 1e9, overscan: 4 });
  assert.equal(w.last, 256);
  assert.ok(255 * 24 + w.shift >= w.virtual && 256 * 24 <= w.virtual + 286 + 1e-9);
  // 64 colonne da 62 px in 300 px: poche celle per riga, mai più di quante ne stanno
  w = axisWindow({ count: 64, size: 62, viewport: 300, scroll: 0, overscan: 2 });
  assert.equal(w.last - w.first, Math.ceil(300 / 62) + 2);
  // tabella vuota o finestra nulla
  assert.deepEqual([axisWindow({ count: 0, size: 24, viewport: 300, scroll: 0 }).first, axisWindow({ count: 0, size: 24, viewport: 300, scroll: 0 }).last], [0, 0]);
  const hidden = axisWindow({ count: 100, size: 24, viewport: 0, scroll: 0, overscan: 0 });
  assert.equal(hidden.last - hidden.first, 0);
});

test("finestra virtuale: un vettore di 65.536 valori disegna poche decine di righe", () => {
  const w = axisWindow({ count: 65536, size: 24, viewport: 286, scroll: 65536 * 12, overscan: 4 });
  assert.equal(w.content, 65536 * 24); // 1,57 milioni di px: sotto il limite, nessuna compressione
  assert.equal(w.scale, 1);
  assert.ok(w.last - w.first <= Math.ceil(286 / 24) + 9, `${w.last - w.first} righe disegnate`);
  assert.equal(w.first, 32768 - 4);
});

test("finestra virtuale compressa (tensori enormi): inizio e fine raggiungibili, celle contigue", () => {
  const count = 1_000_000; // 24 milioni di px: oltre il limite dei browser
  const size = 24;
  const viewport = 300;
  const top = axisWindow({ count, size, viewport, scroll: 0, overscan: 0 });
  assert.equal(top.content, MAX_SCROLL_PX);
  assert.ok(top.scale > 1);
  assert.equal(top.first, 0);
  const end = axisWindow({ count, size, viewport, scroll: MAX_SCROLL_PX, overscan: 0 });
  assert.equal(end.last, count);
  // l'ultima cella finisce esattamente sul bordo inferiore della finestra
  const scrollMax = MAX_SCROLL_PX - viewport;
  assert.ok(Math.abs(count * size + end.shift - (scrollMax + viewport)) < 1e-6);
  // a metà: le celle disegnate sono contigue e coprono la finestra visibile
  const s = scrollMax / 2;
  const mid = axisWindow({ count, size, viewport, scroll: s, overscan: 0 });
  const y0 = mid.first * size + mid.shift;
  const y1 = mid.last * size + mid.shift;
  assert.ok(y0 <= s + 1e-6 && y1 >= s + viewport - 1e-6, `celle da ${y0} a ${y1}, finestra da ${s} a ${s + viewport}`);
});

test("scorrimento per portare in vista una cella: si muove il meno possibile, anche compresso", () => {
  // già visibile → nessuno spostamento
  assert.equal(scrollToShow({ index: 5, count: 256, size: 24, viewport: 286, current: 0 }), 0);
  // sotto la finestra → la cella finisce sul bordo inferiore
  assert.equal(scrollToShow({ index: 100, count: 256, size: 24, viewport: 286, current: 0 }), 101 * 24 - 286);
  // sopra la finestra → la cella finisce sul bordo superiore
  assert.equal(scrollToShow({ index: 3, count: 256, size: 24, viewport: 286, current: 2000 }), 3 * 24);
  // tutto visibile → 0
  assert.equal(scrollToShow({ index: 3, count: 4, size: 24, viewport: 286, current: 0 }), 0);
  // compressa: la cella chiesta cade dentro la finestra che ne risulta
  const count = 1_000_000;
  const scroll = scrollToShow({ index: 777_777, count, size: 24, viewport: 300, current: 0 });
  const w = axisWindow({ count, size: 24, viewport: 300, scroll, overscan: 0 });
  const y = 777_777 * 24 + w.shift;
  assert.ok(y >= scroll - 1e-6 && y + 24 <= scroll + 300 + 1e-6, `cella a ${y}, finestra da ${scroll}`);
});

// ---------------------------------------------------------------- tastiera

test("tastiera: frecce, PagSu/PagGiù, Home/Fine (con Ctrl la tabella intera), vettori", () => {
  const o = { rows: 256, cols: 64, pageRows: 10 };
  assert.deepEqual(moveActive({ row: 5, col: 5 }, "ArrowDown", o), { row: 6, col: 5 });
  assert.deepEqual(moveActive({ row: 0, col: 0 }, "ArrowUp", o), { row: 0, col: 0 }); // fermo sul bordo
  assert.deepEqual(moveActive({ row: 5, col: 63 }, "ArrowRight", o), { row: 5, col: 63 });
  assert.deepEqual(moveActive({ row: 5, col: 5 }, "ArrowLeft", o), { row: 5, col: 4 });
  assert.deepEqual(moveActive({ row: 5, col: 5 }, "PageDown", o), { row: 15, col: 5 });
  assert.deepEqual(moveActive({ row: 5, col: 5 }, "PageUp", o), { row: 0, col: 5 });
  assert.deepEqual(moveActive({ row: 250, col: 5 }, "PageDown", o), { row: 255, col: 5 });
  assert.deepEqual(moveActive({ row: 9, col: 9 }, "Home", o), { row: 9, col: 0 });
  assert.deepEqual(moveActive({ row: 9, col: 9 }, "End", o), { row: 9, col: 63 });
  assert.deepEqual(moveActive({ row: 9, col: 9 }, "Home", { ...o, ctrl: true }), { row: 0, col: 0 });
  assert.deepEqual(moveActive({ row: 9, col: 9 }, "End", { ...o, ctrl: true }), { row: 255, col: 63 });
  // un vettore (una colonna): Home e Fine vanno alla prima e all'ultima riga
  assert.deepEqual(moveActive({ row: 300, col: 0 }, "End", { rows: 65536, cols: 1 }), { row: 65535, col: 0 });
  assert.deepEqual(moveActive({ row: 300, col: 0 }, "Home", { rows: 65536, cols: 1 }), { row: 0, col: 0 });
  // tasti che non sono della tabella
  assert.equal(moveActive({ row: 1, col: 1 }, "a", o), null);
  assert.equal(moveActive({ row: 1, col: 1 }, "Enter", o), null);
  assert.equal(moveActive({ row: 0, col: 0 }, "ArrowDown", { rows: 0, cols: 0 }), null);
});

// ---------------------------------------------------------------- testi

test("testi delle celle: virgola italiana, segno −, precisione del valore esatto", () => {
  assert.equal(cellText(-0.12345), "−0,1235");
  assert.equal(cellText(0.5), "0,5000");
  assert.equal(cellText(1.5, 2), "1,50");
  assert.equal(cellText(NaN), "NaN");
  assert.equal(preciseText(-0.0123456789), "−0,012346");
  assert.equal(preciseText(0), "0");
  assert.equal(preciseText(3.2e-7), "3,2 · 10⁻⁷");
  assert.equal(preciseText(-4.4e-6), "−4,4 · 10⁻⁶");
  assert.equal(preciseText(12345.678), "12345,68");
  assert.equal(preciseText(Infinity), "∞");
  assert.equal(scientificText(5.2781e-5), "5,278 · 10⁻⁵");
  assert.equal(scientificText(-0.00012345, 3), "−1,23 · 10⁻⁴");
  // la colonna è larga quanto il numero più largo (segno compreso)
  assert.equal(widestText(3.5332), "−3,5332".length);
  assert.equal(widestText(0.04), "−0,0400".length);
  assert.equal(finiteAbsMax(Float32Array.from([0.1, -2, Infinity, NaN, 1])), 2);
  assert.equal(finiteAbsMax(Float32Array.from([0.1, -2, 5]), 2), 2); // solo i primi n
});

test("intestazioni e descrizione della cella attiva (annuncio e riga di stato)", () => {
  assert.equal(headerText(7), "7");
  assert.equal(headerText(7, (i) => `'${"abcdefgh"[i]}'`), "'h'");
  assert.equal(headerText(7, () => ""), "7"); // etichetta vuota → l'indice
  const vocab = [..."\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"];
  const text = describeCell({ row: 39, col: 3, value: -0.1234567 }, { rowLabel: (i) => `'${vocab[i]}'`, rowTitle: "carattere", colTitle: "dimensione" });
  assert.equal(text, "carattere 39 ('a') · dimensione 3 · −0,123457");
  assert.equal(describeCell({ row: 2, col: 0, value: 1 }), "riga 2 · colonna 0 · 1,000000");
});

test("statistiche: otto righe dai numeri di shared/weights/stats.js, all'italiana", () => {
  const data = Float32Array.from([-0.5, -0.005, 0, 0.002, 0.25, 0.75]);
  const s = tensorStats(data, { nearZero: 0.01, percentiles: [], histogram: false });
  const rows = statsRows(s);
  assert.deepEqual(rows.map((r) => r.id), ["count", "mean", "std", "min", "max", "absMean", "nearZero", "l2"]);
  const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
  assert.equal(byId.count.value, "6");
  assert.equal(byId.min.value, "−0,5");
  assert.equal(byId.max.value, "0,75");
  assert.equal(byId.nearZero.label, "|w| < 0,01");
  assert.equal(byId.nearZero.value, "50,0%"); // −0,005, 0 e 0,002
  const l2 = Math.sqrt([...data].reduce((a, v) => a + v * v, 0));
  assert.equal(byId.l2.value, l2.toPrecision(4).replace(".", ","));
  assert.equal(byId.mean.value, (data.reduce((a, v) => a + v, 0) / 6).toPrecision(4).replace(".", ","));
  // media minuscola (come nei pesi veri): notazione scientifica leggibile
  const tiny = statsRows(tensorStats(Float32Array.from([0.5, -0.4999472]), { nearZero: 0.01, percentiles: [], histogram: false }));
  assert.match(tiny.find((r) => r.id === "mean").value, /^2,6\d\d · 10⁻⁵$/);
  // valori non finiti: esclusi e dichiarati
  const bad = statsRows(tensorStats(Float32Array.from([1, NaN, Infinity, 2]), { nearZero: 0.01, percentiles: [], histogram: false }));
  assert.equal(bad[0].value, "4 (2 non finiti esclusi)");
  assert.equal(bad.find((r) => r.id === "mean").value, "1,5");
  // nessun valore: trattini, niente NaN in pagina
  const empty = statsRows(tensorStats(new Float32Array(0), { nearZero: 0.01, percentiles: [], histogram: false }));
  assert.ok(empty.slice(1).every((r) => r.value === "–" || r.id === "l2"), JSON.stringify(empty));
});

// ---------------------------------------------------------------- istogramma

test("istogramma: intervalli uguali, l'ultimo comprende il massimo, NaN e ∞ esclusi", () => {
  // un valore al centro di ogni intervallo, più gli estremi 0 e 1 (niente valori sui bordi interni:
  // in float32 0,9 vale 0,89999998 e cadrebbe nell'intervallo prima)
  const data = Float32Array.from([0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1, NaN, Infinity]);
  const m = histogramModel([{ label: "w", data }], { bins: 10 });
  assert.equal(m.empty, false);
  assert.equal(m.min, 0);
  assert.equal(m.max, 1);
  assert.equal(m.bins, 10);
  assert.ok(Math.abs(m.binWidth - 0.1) < 1e-12);
  const counts = [...m.series[0].counts];
  assert.equal(counts.reduce((a, b) => a + b, 0), 12);
  assert.equal(counts[0], 2, "0 e 0,05 nel primo intervallo");
  assert.equal(counts[9], 2, "0,95 e il massimo 1 nell'ultimo intervallo");
  assert.deepEqual(counts.slice(1, 9), [1, 1, 1, 1, 1, 1, 1, 1]);
  assert.equal(m.series[0].total, 12);
  assert.equal(m.series[0].excluded, 2);
  assert.equal(m.maxCount, 2);
  assert.equal(m.series[0].color, PALETTE[0]);
  assert.equal(binAt(m, 0), 0);
  assert.equal(binAt(m, 1), 9);
  assert.equal(binAt(m, 0.55), 5);
  assert.equal(binAt(m, -0.1), -1);
  assert.equal(binAt(m, NaN), -1);
});

test("istogramma A/B: stessi intervalli per tutte le serie (unione dei valori), colori in ordine fisso", () => {
  const a = Float32Array.from({ length: 1000 }, (_, i) => (i / 999) * 2 - 1); // da −1 a 1
  const b = Float32Array.from({ length: 500 }, (_, i) => (i / 499) * 0.5); // da 0 a 0,5
  const m = histogramModel([{ label: "step 0", data: a }, { label: "step 5000", data: b }], { bins: 40 });
  assert.equal(m.min, -1);
  assert.equal(m.max, 1);
  assert.equal(m.series.length, 2);
  assert.equal(m.series[0].counts.length, 40);
  assert.equal(m.series[1].counts.length, 40);
  assert.equal(m.series[1].total, 500);
  // B sta tutto tra 0 e 0,5: intervalli 20…29, più il valore 0,5 che apre il 30 ([0,5; 0,55))
  const countsB = [...m.series[1].counts];
  assert.equal(countsB.slice(0, 20).reduce((s, v) => s + v, 0), 0);
  assert.equal(countsB[30], 1);
  assert.equal(countsB.slice(31).reduce((s, v) => s + v, 0), 0);
  assert.deepEqual(m.series.map((s) => s.color), [seriesColor(0), seriesColor(1)]);
  const rows = binRows(m, 25);
  assert.equal(rows.length, 2);
  assert.ok(rows[1].count > 0 && rows[1].share.endsWith("%"));
});

test("istogramma: valori tutti uguali, nessun valore, scala logaritmica dei conteggi", () => {
  const same = histogramModel([{ data: Float32Array.from([2, 2, 2]) }], { bins: 5 });
  assert.ok(same.min < 2 && same.max > 2, "l'intervallo si allarga attorno al valore");
  assert.equal(same.series[0].counts[2], 3, "la barra unica sta al centro");
  const none = histogramModel([{ data: new Float32Array(0) }], { bins: 5 });
  assert.equal(none.empty, true);
  assert.match(summaryText(none), /senza valori/);
  assert.deepEqual(logCountTicks(1234), [0, 1, 10, 100, 1000, 10000]);
  const log = histogramModel([{ data: Float32Array.from([...Array(1000).fill(0), 1]) }], { bins: 2, logY: true });
  assert.equal(log.yTop, 1000);
  assert.ok(countHeight(1, log) > 0.09, "un solo valore resta visibile in scala logaritmica");
  assert.equal(countHeight(0, log), 0);
  assert.equal(countHeight(1000, log), 1);
  const lin = histogramModel([{ data: Float32Array.from([0, 0, 1]) }], { bins: 2 });
  assert.ok(lin.yTicks.includes(0) && lin.yTop >= 2);
  assert.equal(countHeight(lin.yTop, lin), 1);
});

test("istogramma: testi dell'intervallo, conteggi e riassunto per gli screen reader", () => {
  const m = histogramModel([{ label: "blocks.0.ffwd.net.0.weight", data: Float32Array.from([-0.5, -0.25, 0, 0.25, 0.5]) }], { bins: 4 });
  assert.equal(edgeDecimals(m), 2); // bordi esatti a due decimali (−0,5, −0,25, …)
  const weights = histogramModel([{ data: Float32Array.from([-0.5237, 0.4411]) }], { bins: 60 });
  assert.equal(edgeDecimals(weights), 4, "bordi qualunque: un decimale in più del passo");
  assert.equal(binLabel(m, 0), "[−0,50; −0,25)");
  assert.equal(binLabel(m, 3), "[0,25; 0,50]");
  assert.equal(countText(1234), "1.234");
  assert.equal(countText(99999, { compact: true }), "99.999"); // sotto i 100.000 per intero
  assert.equal(countText(123456, { compact: true }), "123,5K");
  assert.match(countText(1234567, { compact: true }), /^1,2\sMln$/); // Intl mette uno spazio non separabile
  assert.equal(valuesText(1), "1 valore");
  assert.equal(valuesText(2048), "2.048 valori");
  const text = summaryText(m, { label: "Valori del tensore" });
  assert.match(text, /^Valori del tensore\. Istogramma da −0,50 a 0,50 in 4 intervalli/);
  assert.match(text, /5 valori, il picco in/);
  const two = histogramModel([{ label: "A", data: Float32Array.from([0, 1]) }, { label: "B", data: Float32Array.from([0.5]) }], { bins: 2 });
  assert.match(summaryText(two), /2 serie sovrapposte con gli stessi intervalli/);
  assert.match(summaryText(two), /A: 2 valori/);
  assert.match(summaryText(two), /B: 1 valore,/);
});

// ---------------------------------------------------------------- TensorTable su un DOM finto

// three.js (colormap delle heatmap, via core/matrixNumbers.js) sta in webapp/vendor: nel browser lo trova
// l'importmap, qui un hook di risoluzione (come in weights.test.mjs).
const VENDOR = pathToFileURL(fileURLToPath(new URL("../vendor/three/", import.meta.url))).href;
register(
  "data:text/javascript," +
    encodeURIComponent(`export async function resolve(specifier, context, next) {
      if (specifier === "three") return { url: ${JSON.stringify(`${VENDOR}three.module.min.js`)}, shortCircuit: true };
      return next(specifier, context);
    }`),
);

/**
 * Elemento finto: quello che TensorTable usa del DOM (attributi, classi, stile, figli, listener, misure).
 * La griglia è 356 × 312 px come nella sidebar; il campione di 10 caratteri è largo 72 px.
 */
class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parent = null;
    this.attributes = new Map();
    this.listeners = new Map();
    this.dataset = {};
    const props = new Map();
    this.style = { setProperty: (k, v) => props.set(k, v), getPropertyValue: (k) => props.get(k) ?? "" };
    const classes = new Set();
    this.classList = { add: (c) => classes.add(c), remove: (c) => classes.delete(c), contains: (c) => classes.has(c), toggle: (c, on = !classes.has(c)) => (on ? classes.add(c) : classes.delete(c), on) };
    Object.assign(this, { id: "", title: "", hidden: false, tabIndex: -1, scrollTop: 0, scrollLeft: 0, connected: false });
    this._text = "";
  }
  set className(v) {
    for (const c of String(v).split(/\s+/).filter(Boolean)) this.classList.add(c);
  }
  set textContent(v) {
    this.children = [];
    this._text = String(v);
  }
  get textContent() {
    return this._text + this.children.map((c) => (c instanceof FakeElement ? c.textContent : String(c))).join("");
  }
  setAttribute(k, v) {
    this.attributes.set(k, String(v));
  }
  getAttribute(k) {
    return this.attributes.get(k) ?? null;
  }
  removeAttribute(k) {
    this.attributes.delete(k);
  }
  hasAttribute(k) {
    return this.attributes.has(k);
  }
  append(...nodes) {
    for (const n of nodes) {
      if (n instanceof FakeElement) {
        n.remove();
        n.parent = this;
      }
      this.children.push(n);
    }
  }
  remove() {
    if (!this.parent) return;
    this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }
  get isConnected() {
    let e = this;
    while (e.parent) e = e.parent;
    return e.connected;
  }
  addEventListener(type, fn) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  removeEventListener(type, fn) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((f) => f !== fn));
  }
  dispatch(type, event = {}) {
    let prevented = false;
    const e = { type, target: this, preventDefault: () => (prevented = true), ...event };
    for (const fn of this.listeners.get(type) ?? []) fn(e);
    return prevented;
  }
  get clientWidth() {
    return this.classList.contains("tt-grid") ? 356 : 0;
  }
  get clientHeight() {
    return this.classList.contains("tt-grid") ? 312 : 0;
  }
  getBoundingClientRect() {
    return { width: this.classList.contains("tt-probe") ? 72 : this.clientWidth, height: this.clientHeight };
  }
  matches() {
    return false;
  }
  focus() {}
  *walk() {
    yield this;
    for (const c of this.children) if (c instanceof FakeElement) yield* c.walk();
  }
  all(cls) {
    return [...this.walk()].filter((e) => e.classList.contains(cls) && e.isConnected);
  }
}
globalThis.document = { createElement: (tag) => new FakeElement(tag) };
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);

const { TensorTable } = await import("../js/core/ui/tensorTable.js");
const page = () => Object.assign(new FakeElement("div"), { connected: true });
const texts = (els) => els.map((e) => e.textContent);
const rowHeads = (t) => texts(t.root.all("tt-rowhead"));
const colHeads = (t) => texts(t.root.all("tt-colhead"));

test("TensorTable: un tensore nuovo riscrive le intestazioni (anche sulle righe già visibili)", () => {
  let kind = "carattere";
  const t = new TensorTable(page(), { rowLabel: (i) => (kind === "carattere" ? `'${"abcdefghijklmnop"[i]}'` : `k${i}`), colLabel: (j) => (kind === "carattere" ? `${j}` : `x${j}`) });
  t.setTensor({ name: "token_embedding_table.weight", shape: [16, 64], data: new Float32Array(16 * 64) });
  assert.deepEqual(rowHeads(t).slice(0, 3), ["'a'", "'b'", "'c'"]);
  kind = "key"; // le etichette leggono una variabile del chiamante: cambia prima di setTensor
  t.setTensor({ name: "blocks.1.sa.heads.2.key.weight", shape: [16, 64], data: new Float32Array(16 * 64).fill(0.5) });
  assert.deepEqual(rowHeads(t).slice(0, 3), ["k0", "k1", "k2"], "le righe riciclate mostrano le etichette del tensore nuovo");
  assert.deepEqual(colHeads(t).slice(0, 2), ["x0", "x1"]);
  assert.equal(t.grid.getAttribute("aria-rowcount"), "17");
  assert.equal(t.grid.getAttribute("aria-colcount"), "65");
  assert.ok(texts(t.root.all("tt-cell")).every((x) => x === "0,5000"));
  t.dispose();
});

test("TensorTable: virtualizzata (256 × 64 e un vettore di 65.536 valori), lo scorrimento ricicla le celle", async () => {
  const t = new TensorTable(page(), { rowLabel: (i) => `n${i}` });
  t.setTensor({ name: "ffwd", shape: [256, 64], data: Float32Array.from({ length: 256 * 64 }, (_, k) => k / 1e5) });
  const cells = t.root.all("tt-cell").length;
  assert.ok(cells > 0 && cells <= 20 * 8, `${cells} celle nel DOM per 16.384 valori`);
  t.grid.scrollTop = 2400; // riga 100 in cima
  t.grid.dispatch("scroll");
  await new Promise((r) => setTimeout(r, 5));
  const heads = rowHeads(t);
  assert.ok(heads.includes("n100") && !heads.includes("n0"), heads.join(" "));
  const after = t.root.all("tt-cell").length; // a metà c'è il margine sopra e sotto: qualche riga in più, mai tutte
  assert.ok(after >= cells && after <= 20 * 8, `${after} celle dopo lo scorrimento`);
  const v = new TensorTable(page(), {});
  v.setTensor({ name: "vettore", shape: [65536], data: new Float32Array(65536) });
  assert.equal(v.cols, 1);
  assert.ok(v.root.all("tt-cell").length <= 20);
  v.grid.scrollTop = 65536 * 24; // in fondo
  v.grid.dispatch("scroll");
  await new Promise((r) => setTimeout(r, 5));
  assert.ok(rowHeads(v).includes("65535"), "l'ultima riga si raggiunge");
  t.dispose();
  v.dispose();
});

test("TensorTable: tastiera (frecce, PagGiù, Ctrl+Fine, Invio), cella attiva annunciata, dispose", () => {
  const container = page();
  const chosen = [];
  const t = new TensorTable(container, { rowLabel: (i) => `r${i}`, onSelect: (c) => chosen.push(c) });
  const data = Float32Array.from({ length: 40 * 8 }, (_, k) => (k % 2 ? -k : k) / 1000);
  t.setTensor({ name: "m", shape: [40, 8], data, rowTitle: "riga del test", colTitle: "colonna del test" });
  const key = (k, extra = {}) => t.grid.dispatch("keydown", { key: k, ctrlKey: false, metaKey: false, altKey: false, ...extra });
  assert.equal(key("ArrowDown"), true, "le frecce della tabella non muovono la scena (preventDefault)");
  key("ArrowRight");
  assert.deepEqual(t.selected, { i: 1, j: 1, value: data[9] });
  const id = t.grid.getAttribute("aria-activedescendant");
  const active = t.root.all("tt-cell").find((c) => c.id === id);
  assert.ok(active, "aria-activedescendant punta a una cella che esiste");
  assert.equal(active.getAttribute("aria-label"), "riga del test 1 (r1) · colonna del test 1 · −0,009000");
  key("PageDown");
  assert.ok(t.selected.i > 1);
  key("End", { ctrlKey: true });
  assert.deepEqual([t.selected.i, t.selected.j], [39, 7]);
  key("Enter");
  assert.deepEqual(chosen.at(-1), { i: 39, j: 7, value: data[39 * 8 + 7] });
  assert.equal(key("a"), false, "le lettere restano alla navigazione");
  t.dispose();
  assert.equal(container.children.length, 0, "dispose toglie la tabella dal DOM");
  for (const [type, list] of t.grid.listeners) assert.equal(list.length, 0, `listener "${type}" rimasto`);
});

test("TensorTable: refresh dopo un cambio sul posto; colonne più larghe se il numero si allarga", () => {
  const t = new TensorTable(page(), {});
  const data = new Float32Array(4).fill(0.25);
  t.setTensor({ name: "v", shape: [2, 2], data });
  const narrow = t.sizes.cellW;
  data.fill(-12.5); // "−12,5000": un carattere in più di "−0,2500"
  t.refresh();
  assert.ok(t.sizes.cellW > narrow, `${t.sizes.cellW} > ${narrow}`);
  assert.ok(texts(t.root.all("tt-cell")).every((x) => x === "−12,5000"));
  t.dispose();
});
