// Test della matematica dei grafici (modulo puro, niente DOM): node --test webapp/tests/
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  bandLayout,
  buildColorTable,
  colorDomain,
  colorPosition,
  decimalsForStep,
  decimateMinMax,
  extent,
  formatTick,
  formatValue,
  histogram,
  linearScale,
  logScale,
  logTicks,
  nearestIndexSorted,
  niceStep,
  niceTicks,
  placeLabels,
  summarize,
  visibleLabel,
  visibleRange,
} from "../js/core/ui/chartMath.js";

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} ≉ ${b}`);

test("niceStep sceglie 1, 2, 2,5, 5 × 10^k", () => {
  assert.equal(niceStep(10, 5), 2);
  assert.equal(niceStep(1, 4), 0.25);
  assert.equal(niceStep(5000, 5), 1000);
  assert.equal(niceStep(3, 5), 1);
  assert.equal(niceStep(0.07, 5), 0.02);
});

test("niceTicks allarga il dominio a tick arrotondati, senza errori di virgola mobile", () => {
  const t = niceTicks(1.62, 4.38);
  assert.deepEqual(t.ticks, [1, 2, 3, 4, 5]);
  assert.equal(t.min, 1);
  assert.equal(t.max, 5);
  assert.deepEqual(niceTicks(1.62, 4.38, { count: 8 }).ticks, [1.5, 2, 2.5, 3, 3.5, 4, 4.5]);
  const small = niceTicks(0.1, 0.7);
  assert.ok(small.ticks.every((v) => String(v).length <= 4), `tick sporchi: ${small.ticks}`);
  assert.ok(small.min <= 0.1 && small.max >= 0.7);
});

test("niceTicks senza expand tiene il dominio esatto e i tick dentro", () => {
  const t = niceTicks(0, 4999, { expand: false });
  assert.equal(t.min, 0);
  assert.equal(t.max, 4999);
  assert.deepEqual(t.ticks, [0, 1000, 2000, 3000, 4000]);
});

test("niceTicks gestisce un solo valore e input non finiti", () => {
  const t = niceTicks(3, 3);
  assert.ok(t.min < 3 && t.max > 3 && t.ticks.length >= 2);
  const z = niceTicks(0, 0);
  assert.ok(z.min < 0 && z.max > 0);
  const bad = niceTicks(NaN, 1);
  assert.equal(bad.min, 0);
  assert.equal(bad.max, 1);
});

test("logTicks: meno di una decade → 1…9, molte decadi → potenze di 10", () => {
  const loss = logTicks(1.66, 4.4);
  assert.deepEqual(loss.ticks, [1, 2, 3, 4, 5]);
  const wide = logTicks(0.001, 1000);
  assert.deepEqual(wide.ticks, [0.001, 0.01, 0.1, 1, 10, 100, 1000]);
  const mid = logTicks(0.3, 40);
  assert.deepEqual(mid.ticks, [0.2, 0.5, 1, 2, 5, 10, 20, 50]);
  assert.deepEqual(logTicks(-1, 5).ticks, [1, 10]); // dominio non valido: fallback
  const padded = logTicks(1.66, 4.4, { pad: 1.08 });
  assert.deepEqual(padded.ticks, [2, 3, 4]); // dominio stretto attorno ai dati, tick dentro
  assert.ok(padded.min < 1.66 && padded.min > 1.5 && padded.max > 4.4 && padded.max < 4.8);
});

test("scale lineari e logaritmiche con invert", () => {
  const s = linearScale([0, 10], [100, 0]);
  assert.equal(s(0), 100);
  assert.equal(s(10), 0);
  close(s.invert(25), 7.5);
  const l = logScale([1, 100], [0, 200]);
  close(l(10), 100);
  close(l.invert(100), 10);
  assert.ok(Number.isNaN(l(0)));
});

test("decimalsForStep e formatTick all'italiana", () => {
  assert.equal(decimalsForStep(0.25), 2);
  assert.equal(decimalsForStep(0.5), 1);
  assert.equal(decimalsForStep(2.5), 1);
  assert.equal(decimalsForStep(1000), 0);
  assert.equal(formatTick(2.5, 0.5), "2,5");
  assert.equal(formatTick(1000, 1000), "1.000");
  assert.equal(formatTick(20000, 10000, 50000), "20K");
  assert.equal(formatTick(-1e-17, 0.1), "0,0");
});

test("formatValue: virgola decimale e cifre significative", () => {
  assert.equal(formatValue(1.66223), "1,662");
  assert.equal(formatValue(209729), "209.729");
  assert.equal(formatValue(NaN), "–");
  assert.match(formatValue(0.00001234), /^1,23E-5$/);
});

test("extent e summarize ignorano NaN e Infinity", () => {
  assert.deepEqual(extent([3, NaN, -2, Infinity, 7]), { min: -2, max: 7 });
  assert.equal(extent([NaN, Infinity]), null);
  const s = summarize(new Float32Array([1, 2, 3, 4, NaN]));
  assert.equal(s.count, 4);
  close(s.mean, 2.5);
  close(s.std, Math.sqrt(1.25));
});

test("histogram: bin uguali, massimo nell'ultimo bin, valori non finiti scartati", () => {
  const h = histogram([0, 0.5, 1, 1.5, 2, NaN, Infinity], { bins: 4 });
  assert.deepEqual([...h.counts], [1, 1, 1, 2]);
  assert.deepEqual([...h.edges], [0, 0.5, 1, 1.5, 2]);
  assert.equal(h.total, 5);
  assert.equal(h.ignored, 2);
});

test("histogram: somma dei conteggi = valori validi, anche su dati grandi", () => {
  const values = new Float32Array(100000);
  for (let i = 0; i < values.length; i++) values[i] = Math.sin(i) * 3;
  const h = histogram(values, { bins: 40 });
  assert.equal(h.counts.length, 40);
  assert.equal(h.counts.reduce((a, b) => a + b, 0), values.length);
});

test("histogram: valori tutti uguali, limiti espliciti e input vuoto", () => {
  const same = histogram([2, 2, 2], { bins: 5 });
  assert.equal(same.min, 1.5);
  assert.equal(same.max, 2.5);
  assert.equal(same.counts[2], 3);
  const clipped = histogram([-5, 0, 0.2, 5], { bins: 2, min: -1, max: 1 });
  assert.equal(clipped.total, 2);
  assert.equal(clipped.ignored, 2);
  assert.equal(histogram([], {}), null);
  assert.equal(histogram([NaN]), null);
});

test("decimateMinMax: al più 4 punti per colonna, picchi ed estremi conservati", () => {
  const n = 300000;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = i;
    ys[i] = Math.sin(i / 1000);
  }
  ys[123457] = 9; // picco isolato
  ys[200001] = -9;
  const out = decimateMinMax(xs, ys, n, 0, n - 1, 400);
  assert.ok(out.length / 2 <= 400 * 4, `troppi punti: ${out.length / 2}`);
  const outYs = out.filter((_, i) => i % 2 === 1);
  assert.equal(Math.max(...outYs), 9);
  assert.equal(Math.min(...outYs), -9);
  assert.equal(out[0], 0); // primo punto
  assert.equal(out[out.length - 2], n - 1); // ultimo punto
  for (let i = 2; i < out.length; i += 2) assert.ok(out[i] > out[i - 2], "x non crescenti");
});

test("nearestIndexSorted e visibleRange", () => {
  const xs = [0, 10, 20, 30];
  assert.equal(nearestIndexSorted(xs, 4, 14), 1);
  assert.equal(nearestIndexSorted(xs, 4, 16), 2);
  assert.equal(nearestIndexSorted(xs, 4, -50), 0);
  assert.equal(nearestIndexSorted(xs, 4, 99), 3);
  assert.equal(nearestIndexSorted(xs, 0, 5), -1);
  assert.deepEqual(visibleRange(xs, 4, 12, 22), [1, 4]);
  assert.deepEqual(visibleRange([0, 10, 20, 30, 40, 50], 6, 20, 30), [1, 5]);
});

test("bandLayout: barre sottili (max 24 px) con almeno 2 px di spazio", () => {
  const wide = bandLayout(5, 400);
  assert.equal(wide.bar, 24);
  const dense = bandLayout(100, 300);
  assert.ok(dense.band - dense.bar >= 0.8 && dense.bar >= 1);
  assert.deepEqual(bandLayout(0, 100), { band: 0, bar: 0, offset: 0 });
});

test("placeLabels scarta le etichette sovrapposte", () => {
  const keep = placeLabels([
    { x: 0, y: 0, w: 20, h: 10 },
    { x: 10, y: 5, w: 20, h: 10 },
    { x: 40, y: 0, w: 20, h: 10 },
  ]);
  assert.deepEqual(keep, [true, false, true]);
  const blocked = placeLabels([{ x: 0, y: 0, w: 10, h: 10 }], 0, [{ x: 5, y: 5, w: 4, h: 4 }]);
  assert.deepEqual(blocked, [false]); // un ostacolo (un punto) blocca l'etichetta
});

test("rampe di colore: estremi esatti e dominio divergente simmetrico", () => {
  const table = buildColorTable(["#000000", "#ffffff"], 3);
  assert.deepEqual([...table.slice(0, 3)], [0, 0, 0]);
  assert.deepEqual([...table.slice(6, 9)], [255, 255, 255]);
  const d = colorDomain([-0.2, 0.5], { diverging: true });
  assert.deepEqual(d, { min: -0.5, max: 0.5 });
  assert.equal(colorPosition(0, d), 0.5);
  assert.equal(colorPosition(9, d), 1);
  assert.deepEqual(colorDomain([0, 1], { min: 0, max: 0.5 }), { min: 0, max: 0.5 });
});

test("visibleLabel rende visibili spazi e a capo", () => {
  assert.equal(visibleLabel(" "), "␣");
  assert.equal(visibleLabel("\n"), "⏎");
  assert.equal(visibleLabel("a"), "a");
});
