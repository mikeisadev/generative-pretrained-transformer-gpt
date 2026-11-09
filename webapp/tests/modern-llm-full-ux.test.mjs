// Test della revisione UX della vista "Architettura LLM moderni completa" (moduli puri e di soli dati):
// la colonna sotto la fase dei pesi non tocca le schede di sintesi, la versione dei pesi è un testo
// (non l'oggetto del checkpoint), le ancore "da inquadrare insieme" esistono.
import assert from "node:assert/strict";
import test from "node:test";

import { FULL_SECTIONS } from "../js/tabs/modern-llm/full/catalog.js";
import { FULL_PATH } from "../js/tabs/modern-llm/full/content.js";
import { bigPrecisions, MACRO, MACRO_WEIGHTS, macroLayout, macroWeightsColumn } from "../js/tabs/modern-llm/full/layout.js";
import { anchorsOf } from "../js/tabs/modern-llm/full/model.js";
import { MODELS } from "../js/tabs/modern-llm/full/models/index.js";

test("Panoramica: la legenda dei byte per precisione non finisce sopra le schede di sintesi", () => {
  for (const m of MODELS) {
    for (const scale of [1, 2.5]) {
      const L = macroLayout(m, scale);
      const legendBottom = -macroWeightsColumn(m) * scale;
      for (const note of [L.capabilities, L.weightsShare]) assert.ok(note.y + note.h / 2 < legendBottom, `${m.id} (scala ${scale}): scheda a ${note.y + note.h / 2} sopra la legenda (${legendBottom})`);
      assert.ok(L.box.minY <= L.weightsShare.y - L.weightsShare.h / 2, `${m.id}: il riquadro della Panoramica comprende le schede`);
    }
  }
  // DeepSeek-V4.1-Flash: una pillola «da verificare» e due precisioni grandi (MXFP4 degli esperti, FP8 di Engram)
  const v41 = MODELS[0];
  assert.equal(bigPrecisions(v41).length, 2);
  const W = MACRO_WEIGHTS;
  assert.equal(macroWeightsColumn(v41), MACRO.stageH / 2 + W.barGap + W.doubtStep + W.barH / 2 + W.legendGap + W.legendHead + 2 * W.legendLine);
});

test("modelli: la versione dei pesi è un testo e il checkpoint resta un oggetto (niente «[object Object]»)", () => {
  for (const m of MODELS) {
    assert.equal(typeof m.version, "string", `${m.id}: version`);
    assert.equal(typeof m.checkpoint, "object", `${m.id}: checkpoint`);
    assert.ok(m.checkpoint.format && m.checkpoint.gb > 0, `${m.id}: formato e dimensione del checkpoint`);
  }
  assert.equal(MODELS[0].version, "checkpoint 1017");
});

test("ancore inquadrate insieme (also): esistono in ogni modello che ha l'ancora principale", () => {
  const withAlso = [...FULL_SECTIONS, ...FULL_PATH.flatMap((s) => s.children)].filter((x) => x.also);
  assert.ok(withAlso.length >= 2, "la vista e il passo «I pesi» inquadrano anche «Dove sono i pesi»");
  for (const m of MODELS) {
    const anchors = anchorsOf(m);
    for (const x of withAlso) {
      if (!anchors.has(x.anchor)) continue;
      for (const a of x.also) assert.ok(anchors.has(a), `${m.id}: ${x.id} → ${a}`);
    }
  }
});
