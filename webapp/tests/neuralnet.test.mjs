// Test della rete neurale orizzontale con profondità (js/core/neuralNet/: moduli puri, niente three.js né DOM):
//   node --test webapp/tests/*.test.mjs
// Specifica (normalizzazione ed errori), disposizione x/y/z, compressione dei blocchi ripetuti,
// attivazioni simulate e instradamento MoE, generazione deterministica dei collegamenti, indice sullo schermo.
import assert from "node:assert/strict";
import { test } from "node:test";

import { MODE, computeRouting, resolveValues, simulateAttention, topK } from "../js/core/neuralNet/activations.js";
import { DEMO_SPECS, loadTestSpec } from "../js/core/neuralNet/demoSpecs.js";
import { SPACING, boxOf, decodeUnit, unitIndex } from "../js/core/neuralNet/layout.js";
import { ARC_KIND, LINK_LIMITS, pairDepths } from "../js/core/neuralNet/links.js";
import { ScreenIndex } from "../js/core/neuralNet/pickIndex.js";
import { gaussian, hashSeed, mulberry32 } from "../js/core/neuralNet/random.js";
import { PATTERNS, ROLES } from "../js/core/neuralNet/roles.js";
import { buildNetworkModel } from "../js/core/neuralNet/model.js";
import { LIMITS, NetworkSpecError, normalizeSpec } from "../js/core/neuralNet/spec.js";

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} ≉ ${b}`);
const TOKENS = ["Il", "gatto", "dorme", "sul", "divano"];

/** Specifica piccola ma completa: token in profondità, attention, router, esperti, residuo. */
function tinySpec(extra = {}) {
  return {
    title: "Prova",
    scale: "ridotta",
    depth: { count: TOKENS.length, label: "token", names: TOKENS },
    sections: [
      { id: "in", label: "Token", role: "input", neurons: 6, depth: true, activation: "onehot" },
      { id: "emb", label: "Embedding", role: "embedding", neurons: 8, depth: true },
      { id: "att", label: "Attention", role: "attention", neurons: 8, heads: 2, slabs: 2, depth: true },
      { id: "add", label: "+", role: "add", neurons: 8, depth: true },
      { id: "router", label: "Router", role: "router", neurons: 6, depth: true },
      { id: "exp", label: "Esperti", role: "experts", neurons: 3, experts: 6, active: 2, shared: 1, router: "router", depth: true },
      { id: "out", label: "Uscita", role: "output", neurons: 4, depth: true },
    ],
    links: [
      { from: "in", to: "emb", pattern: "lookup" },
      { from: "emb", to: "att", pattern: "dense", density: 0.5, toSlab: "all" },
      { within: "att", pattern: "attention", causal: true },
      { from: "att", to: "add", pattern: "dense", density: 0.3 },
      { from: "emb", to: "add", pattern: "residual" },
      { from: "add", to: "router", pattern: "dense", density: 0.4 },
      { from: "router", to: "exp", pattern: "route" },
      { from: "exp", to: "out", pattern: "route", density: 0.5 },
    ],
    ...extra,
  };
}

const build = (spec, limits) => buildNetworkModel(spec, { limits });

/** Errori di una specifica sbagliata (fallisce se NON lancia NetworkSpecError). */
function errorsOf(spec) {
  try {
    normalizeSpec(spec);
  } catch (e) {
    assert.ok(e instanceof NetworkSpecError, `errore inatteso: ${e}`);
    return e.errors;
  }
  assert.fail("la specifica doveva essere rifiutata");
}

// ------------------------------------------------------------------------------------------ random
test("random: mulberry32 e hashSeed sono deterministici e ben distribuiti", () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  for (let i = 0; i < 50; i++) assert.equal(a(), b());
  assert.notEqual(hashSeed("a", "bc"), hashSeed("ab", "c"));
  const rand = mulberry32(1);
  let sum = 0;
  let sq = 0;
  for (let i = 0; i < 20000; i++) {
    const g = gaussian(rand);
    sum += g;
    sq += g * g;
  }
  near(sum / 20000, 0, 0.05);
  near(sq / 20000, 1, 0.06);
});

// ------------------------------------------------------------------------------------------ spec
test("spec: valori predefiniti risolti (colore del ruolo, lastre, profondità, densità, offset)", () => {
  const net = normalizeSpec(tinySpec());
  assert.equal(net.title, "Prova");
  assert.equal(net.sections.length, 7);
  const [inp, emb, att, , router, exp] = net.sections;
  assert.equal(emb.color, ROLES.embedding.color);
  assert.equal(att.slabs, 2);
  assert.equal(att.depth.kind, "line");
  assert.deepEqual(att.depth.names, TOKENS);
  assert.equal(exp.slabs, 7, "esperti condivisi + instradati");
  assert.equal(exp.router, router, "il router dichiarato diventa la sezione");
  assert.equal(inp.offset, 0);
  assert.equal(emb.offset, inp.units);
  assert.equal(net.units, net.sections.reduce((a, s) => a + s.units, 0));
  assert.equal(net.links[1].density, 0.5);
  assert.equal(net.links[0].density, 1, "lookup: tutti");
  assert.equal(net.links[1].fromSlab, 0);
  assert.equal(net.links[1].toSlab, "all");
  assert.equal(net.links[3].fromSlab, 1, "si esce dall'ultima lastra");
  assert.equal(net.simulated, true);
  assert.deepEqual(net.slices, { label: "token", count: 5, names: TOKENS, kind: "line" });
  assert.deepEqual(net.warnings, []);
});

test("spec: forme della profondità (fila, griglia, oggetto, ereditata) e nomi", () => {
  const net = normalizeSpec({
    title: "Griglie",
    scale: "x",
    depth: { grid: [4, 4], label: "griglia" },
    sections: [
      { id: "a", label: "A", role: "conv", neurons: 3, depth: true },
      { id: "b", label: "B", role: "conv", neurons: 3, depth: [2, 2] },
      { id: "c", label: "C", role: "latent", neurons: 3, depth: 7 },
      { id: "d", label: "D", role: "latent", neurons: 3 },
      { id: "e", label: "E", role: "latent", neurons: 3, depth: { count: 3, label: "frame", names: ["0", "1", "2"] } },
    ],
  });
  const [a, b, c, d, e] = net.sections;
  assert.deepEqual([a.depth.kind, a.depth.rows, a.depth.cols, a.depth.count], ["grid", 4, 4, 16]);
  assert.deepEqual([b.depth.rows, b.depth.cols], [2, 2]);
  assert.deepEqual([c.depth.kind, c.depth.count, c.depth.label], ["line", 7, "posizioni"]);
  assert.deepEqual([d.depth.kind, d.depth.count], ["none", 1]);
  assert.deepEqual(e.depth.names, ["0", "1", "2"]);
  assert.equal(net.slices.count, 4, "fette = colonne della griglia predefinita");
  assert.equal(net.slices.names[0], "colonna 1");
});

test("spec: tutti gli errori in una volta, con il percorso del campo", () => {
  const errors = errorsOf({
    title: "",
    sections: [
      { id: "a", label: "A", role: "magia", neurons: 3 },
      { id: "b", label: "B", role: "mlp", neurons: 0 },
      { id: "b", label: "B2", role: "mlp", neurons: 2 },
      { id: "c", label: "C", role: "mlp", neurons: 2, values: [1, 2, 3] },
    ],
    links: [{ from: "c", to: "nessuno", pattern: "dense" }, { from: "c", to: "c", pattern: "teletrasporto" }],
  });
  const text = errors.join("\n");
  assert.match(text, /^title: /m);
  assert.match(text, /sections\[0\]\.role \(a\): ruolo sconosciuto "magia"/);
  assert.match(text, /sections\[1\]\.neurons \(b\)/);
  assert.match(text, /sections\[3\]\.values \(c\): servono 2 valori/);
  assert.match(text, /links\[0\] \(dense\)\.to: id sconosciuto "nessuno"/);
  assert.match(text, /links\[1\]\.pattern: schema sconosciuto/);
  // "a" è già segnalata: i collegamenti che la citano non aggiungono "id sconosciuto"
  const cascade = errorsOf({ title: "x", sections: [{ id: "a", label: "A", role: "boh", neurons: 2 }, { id: "b", label: "B", neurons: 2 }], links: [{ from: "a", to: "b", pattern: "dense" }] });
  assert.equal(cascade.length, 1);
});

test("spec: regole degli schemi (profondità richieste, griglie, residui, esperti)", () => {
  const base = (sections, links) => ({ title: "x", scale: "x", depth: { count: 4, label: "token" }, sections, links });
  const s = (id, extra = {}) => ({ id, label: id, role: "mlp", neurons: 4, depth: true, ...extra });
  assert.match(errorsOf(base([s("a"), s("b")], [{ within: "a", pattern: "dense" }])).join(), /within" vale solo/);
  assert.match(errorsOf(base([s("a", { depth: false }), s("b")], [{ within: "a", pattern: "attention" }])).join(), /depth ≥ 2/);
  assert.match(errorsOf(base([s("a"), s("b")], [{ from: "a", to: "b", pattern: "pool" }])).join(), /pooling va da una sezione con profondità/);
  assert.match(errorsOf(base([s("a", { depth: false }), s("b", { depth: false })], [{ from: "a", to: "b", pattern: "broadcast" }])).join(), /broadcast/);
  assert.match(errorsOf(base([s("a"), s("b")], [{ from: "b", to: "a", pattern: "residual" }])).join(), /salta in avanti/);
  assert.match(errorsOf(base([s("a", { depth: [4, 4] }), s("b", { depth: [3, 3] })], [{ from: "a", to: "b", pattern: "down" }])).join(), /fattore intero/);
  assert.match(errorsOf(base([s("a", { depth: [4, 4] }), s("b", { depth: [2, 2] })], [{ from: "a", to: "b", pattern: "conv" }])).join(), /stessa griglia/);
  assert.match(errorsOf(base([s("a"), s("b")], [{ from: "a", to: "b", pattern: "route" }])).join(), /esattamente una delle due/);
  assert.match(errorsOf(base([s("a", { depth: [2, 2], slabs: 2 })], [])).join(), /ogni canale è già una lastra/);
  assert.match(errorsOf(base([s("e", { role: "experts", experts: 4, active: 5 })], [])).join(), /esperti attivi per token/);
  assert.match(errorsOf(base([s("r", { role: "router", neurons: 3 }), s("e", { role: "experts", experts: 4, router: "r" })], [])).join(), /ne serve uno per esperto/);
  assert.match(errorsOf(base([s("a"), s("b")], [{ from: "a", to: "b", pattern: "explicit", pairs: [[0, 9, 1]] }])).join(), /indice di arrivo 9 fuori/);
  assert.match(errorsOf(base([s("a"), s("b")], [{ from: "a", to: "b", pattern: "explicit", pairs: [[0.5, 1, 1]] }])).join(), /indice di partenza 0.5/);
  assert.match(errorsOf(base([s("a"), s("b")], [{ from: "a", to: "b", pattern: "dense", density: 2 }])).join(), /density/);
  assert.match(errorsOf(base([s("a"), s("b")], [{ within: "b", pattern: "attention", weights: [1, 2] }])).join(), /servono 16 pesi/);
});

test("spec: ripetizioni, gruppi, fonti sicure e limiti", () => {
  const sec = (id) => ({ id, label: id, role: "mlp", neurons: 2 });
  const ok = normalizeSpec({ title: "r", scale: "s", sections: [sec("a"), sec("b"), sec("c"), { repeat: 12, of: ["b", "c"] }, sec("d")], groups: [{ label: "G", from: "b", to: "repeat-0" }] });
  assert.equal(ok.repeats[0].count, 12);
  assert.deepEqual(ok.repeats[0].of.map((s) => s.id), ["b", "c"]);
  assert.equal(ok.repeats[0].label, "× 12 blocchi uguali");
  assert.equal(ok.items.length, 5);
  assert.equal(ok.groups[0].to.kind, "repeat");
  assert.match(errorsOf({ title: "r", sections: [sec("a"), sec("b"), sec("c"), { repeat: 3, of: ["a", "c"] }] }).join(), /consecutive/);
  assert.match(errorsOf({ title: "r", sections: [sec("a"), sec("b"), { repeat: 3, of: ["a"] }] }).join(), /subito dopo/);
  assert.match(errorsOf({ title: "r", sections: [sec("a"), { repeat: 1, of: ["a"] }] }).join(), /intero ≥ 2/);
  assert.match(errorsOf({ title: "r", sections: [sec("a")], sources: [{ title: "x", url: "javascript:alert(1)" }] }).join(), /http\(s\)/);
  assert.match(errorsOf({ title: "r", sections: [{ id: "big", label: "B", neurons: 4096, depth: 100 }] }).join(), new RegExp(`massimo ${LIMITS.units}`));
  assert.match(errorsOf({ title: "r", sections: [sec("a")], groups: [{ label: "G", from: "a", to: "zzz" }] }).join(), /id sconosciuto "zzz"/);
});

test("spec: valori veri accettati (anche come elenco) e rete non più simulata", () => {
  const values = Array.from({ length: 3 * 2 }, (_, i) => i - 2);
  const net = normalizeSpec({ title: "v", scale: "s", sections: [{ id: "a", label: "A", role: "mlp", neurons: 3, depth: 2, values }] });
  assert.ok(net.sections[0].values instanceof Float32Array);
  assert.equal(net.simulated, false);
  assert.match(errorsOf({ title: "v", sections: [{ id: "a", label: "A", neurons: 1, values: [NaN] }] }).join(), /non finito/);
});

// ------------------------------------------------------------------------------------------ layout
test("layout: sezioni da sinistra a destra, centrate su y, fette allineate in profondità", () => {
  const { net, layout } = build(tinySpec());
  for (let i = 1; i < layout.sections.length; i++) {
    assert.ok(layout.sections[i].x0 >= layout.sections[i - 1].x1 + SPACING.section - 1e-9, "spazio tra le sezioni");
  }
  for (const box of layout.sections) near(box.yTop, -box.yBottom);
  // stessa posizione (token 3) → stessa z in tutte le sezioni con la stessa profondità
  const zOf = (s, d) => layout.positions[3 * (s.offset + unitIndex(s, 0, 0, d)) + 2];
  const zs = net.sections.map((s) => zOf(s, 3));
  for (const z of zs) near(z, zs[0]);
  // il token 0 sta in fondo, l'ultimo davanti (verso la camera di lato)
  assert.ok(zOf(net.sections[0], 0) < zOf(net.sections[0], 4));
  near(zOf(net.sections[0], 0), -zOf(net.sections[0], 4));
});

test("layout: la fetta è una posizione dell'asse z; le profondità di altro tipo non partecipano", () => {
  const { net, layout } = build(DEMO_SPECS.unet);
  const text = net.byId.get("text");
  const noisy = net.byId.get("noisy");
  const small = net.byId.get("down1"); // griglia 4 × 4
  assert.equal(net.slices.kind, "grid");
  for (let u = 0; u < text.units; u++) assert.equal(layout.depthU[text.offset + u], -1, "i token del prompt non sono colonne della griglia");
  // colonna 3 di 8 nella griglia grande ↔ colonna 1 di 4 nella griglia piccola (stessa frazione dell'asse z)
  const uBig = layout.depthU[noisy.offset + unitIndex(noisy, 0, 0, 3)];
  const uSmall = layout.depthU[small.offset + unitIndex(small, 0, 0, 1)];
  near(uBig, 3.5 / 8);
  assert.ok(Math.floor(uBig * 4) === Math.floor(uSmall * 4));
});

test("layout: indici delle unità (lastra, neurone, posizione) e posizioni coerenti", () => {
  const { net, layout } = build(tinySpec());
  const att = net.sections[2];
  for (let u = 0; u < att.units; u++) {
    const { slab, neuron, depth } = decodeUnit(att, u);
    assert.equal(unitIndex(att, slab, neuron, depth), u);
  }
  const box = layout.sections[att.index];
  const x = (slab) => layout.positions[3 * (att.offset + unitIndex(att, slab, 0, 0))];
  near(x(0), box.xs[0]);
  near(x(1), box.xs[1]);
  // spazio extra tra le due head: tra il neurone 3 (head 0) e il 4 (head 1) il salto è più grande
  const y = (n) => layout.positions[3 * (att.offset + unitIndex(att, 0, n, 0)) + 1];
  assert.ok(y(3) - y(4) > y(2) - y(3) + 1e-6);
  // gli esperti: prima il condiviso, poi un passo vuoto, poi gli instradati
  const ex = layout.sections[net.sections[5].index];
  assert.ok(ex.xs[1] - ex.xs[0] > ex.xs[2] - ex.xs[1] + 1e-6);
});

test("layout: griglia 2D = righe su y, colonne su z, un canale per lastra; le sezioni alte si stringono", () => {
  const { net, layout } = build({
    title: "g",
    scale: "s",
    sections: [
      { id: "g", label: "G", role: "conv", neurons: 3, depth: [4, 6] },
      { id: "tall", label: "T", role: "mlp", neurons: 400 },
    ],
  });
  const g = net.sections[0];
  const at = (u) => Array.from(layout.positions.subarray(3 * (g.offset + u), 3 * (g.offset + u) + 3));
  const box = layout.sections[0];
  const [x0, y0, z0] = at(unitIndex(g, 0, 0, 0));
  const [, yRow1] = at(unitIndex(g, 0, 0, 6)); // riga 1, colonna 0
  const [, , zCol1] = at(unitIndex(g, 0, 0, 1)); // riga 0, colonna 1
  const [xCh2] = at(unitIndex(g, 0, 2, 0)); // canale 2
  near(y0 - yRow1, box.dy);
  near(zCol1 - z0, box.dz);
  near(xCh2 - x0, 2 * SPACING.plate);
  const tall = layout.sections[1];
  assert.ok(tall.yTop - tall.yBottom <= SPACING.maxHeight + 1e-9);
  assert.ok(tall.radius < layout.sections[0].radius, "neuroni più piccoli dove sono fitti");
});

test("layout: blocco ripetuto = poche lastre fantasma dopo il blocco, con il suo ingombro", () => {
  const { net, layout } = build(DEMO_SPECS.llm);
  const moe = layout.repeats.find((r) => r.id === "moe_x58");
  assert.equal(moe.count, 58);
  assert.equal(moe.shown, SPACING.ghosts);
  const block = net.repeats.find((r) => r.id === "moe_x58").of;
  const last = layout.sections[block[block.length - 1].index];
  assert.ok(moe.x0 > last.x1, "dopo il blocco");
  assert.equal(moe.blockX0, layout.sections[block[0].index].x0);
  near(moe.yTop, Math.max(...block.map((s) => layout.sections[s.index].yTop)));
  const dense = layout.repeats.find((r) => r.id === "dense_x3");
  assert.equal(dense.shown, 2, "× 3 = uno vero + 2 fantasmi");
  const box = boxOf(layout, net, ["moe_x58"]);
  near(box.min[0], moe.x0);
});

test("layout: residui consecutivi alla stessa altezza, sovrapposti un gradino sopra; gruppi sopra i residui", () => {
  const sec = (id, n = 4) => ({ id, label: id, role: "mlp", neurons: n });
  const { layout } = build({
    title: "r",
    scale: "s",
    sections: [sec("a"), sec("b"), sec("c"), sec("d")],
    links: [
      { from: "a", to: "b", pattern: "residual" },
      { from: "b", to: "c", pattern: "residual" },
      { from: "a", to: "d", pattern: "residual" },
    ],
    groups: [{ label: "tutto", from: "a", to: "d" }, { label: "inizio", from: "a", to: "b" }],
  });
  const [ab, bc, ad] = layout.residuals;
  assert.equal(ab.level, 0);
  assert.equal(bc.level, 0);
  assert.equal(ad.level, 1);
  assert.ok(ad.height > ab.height);
  const [all, start] = layout.groups;
  assert.ok(start.y > ab.height, "la parentesi sta sopra i residui");
  assert.ok(all.y > start.y, "la parentesi più larga sta sopra quella che contiene");
  assert.ok(layout.bounds.max[1] >= all.y);
});

// ------------------------------------------------------------------------------------------ valori
test("valori: simulazione deterministica, one-hot, probabilità, LayerNorm per posizione", () => {
  const a = resolveValues(normalizeSpec(tinySpec()));
  const b = resolveValues(normalizeSpec(tinySpec()));
  assert.deepEqual(a.raw, b.raw);
  const other = resolveValues(normalizeSpec(tinySpec({ seed: 99 })));
  assert.notDeepEqual(a.raw, other.raw);
  const net = normalizeSpec(tinySpec());
  const v = resolveValues(net);
  const [inp, , , , , , out] = net.sections;
  for (let d = 0; d < 5; d++) {
    let hot = 0;
    let sum = 0;
    for (let i = 0; i < 6; i++) hot += v.raw[inp.offset + i * 5 + d];
    for (let i = 0; i < 4; i++) sum += v.raw[out.offset + i * 5 + d];
    assert.equal(hot, 1, "un solo token acceso per posizione");
    near(sum, 1, 1e-5);
  }
  assert.equal(v.mode[inp.offset], MODE.onehot);
  assert.equal(v.mode[out.offset], MODE.probs);
  const norm = normalizeSpec({ title: "n", scale: "s", sections: [{ id: "n", label: "N", role: "norm", neurons: 16, depth: 3 }] });
  const nv = resolveValues(norm).raw;
  for (let d = 0; d < 3; d++) {
    let mean = 0;
    for (let i = 0; i < 16; i++) mean += nv[i * 3 + d];
    near(mean / 16, 0, 1e-5);
  }
});

test("valori: gli esperti si accendono solo per i token che li scelgono (top-k del router + condivisi)", () => {
  const net = normalizeSpec(tinySpec());
  const v = resolveValues(net);
  const exp = net.sections[5];
  const router = net.sections[4];
  const route = v.routing.get("exp");
  assert.equal(route.source, "router");
  for (let d = 0; d < 5; d++) {
    const scores = Array.from({ length: 6 }, (_, e) => v.raw[router.offset + e * 5 + d]);
    const chosen = Array.from(route.experts.subarray(d * 2, d * 2 + 2));
    assert.deepEqual(chosen, topK(scores, 2), "gli esperti con il punteggio più alto");
    near(route.gates[d * 2] + route.gates[d * 2 + 1], 1, 1e-6);
    assert.equal(route.mask[0 * 5 + d], 1, "il condiviso lavora sempre");
    for (let slab = 0; slab < exp.slabs; slab++) {
      const on = route.mask[slab * 5 + d] === 1;
      for (let n = 0; n < exp.neurons; n++) {
        const u = exp.offset + unitIndex(exp, slab, n, d);
        if (!on) {
          assert.equal(v.raw[u], 0);
          assert.equal(v.mode[u], MODE.off);
        }
      }
    }
  }
  // instradamento scritto nella specifica: rispettato così com'è
  const fixed = normalizeSpec(tinySpec({ sections: tinySpec().sections.map((s) => (s.id === "exp" ? { ...s, router: undefined, routes: [0, 1, 2, 3, 4, 5, 0, 2, 1, 3] } : s)) }));
  const r2 = computeRouting(fixed, fixed.sections[5], new Float32Array(fixed.units));
  assert.equal(r2.source, "spec");
  assert.deepEqual(Array.from(r2.experts), [0, 1, 2, 3, 4, 5, 0, 2, 1, 3]);
});

test("valori: attention simulata = righe che sommano a 1, futuro a zero con la maschera causale", () => {
  const w = simulateAttention(4, 6, 6, { causal: true, seed: [3] });
  for (let h = 0; h < 4; h++) {
    for (let i = 0; i < 6; i++) {
      let sum = 0;
      for (let j = 0; j < 6; j++) {
        const x = w[(h * 6 + i) * 6 + j];
        if (j > i) assert.equal(x, 0);
        sum += x;
      }
      near(sum, 1, 1e-5);
    }
  }
  // head 0 = "token precedente": per la query 4 il massimo è sulla chiave 3
  const row = Array.from(w.subarray(4 * 6, 5 * 6));
  assert.equal(row.indexOf(Math.max(...row)), 3);
  const cross = simulateAttention(2, 64, 6, { qCols: 8, kCols: 6, seed: [4] });
  assert.equal(cross.length, 2 * 64 * 6);
});

// ------------------------------------------------------------------------------------------ collegamenti
test("collegamenti: stessi input → stessi segmenti; ordinati per priorità; tetto rispettato", () => {
  const a = build(tinySpec()).links;
  const b = build(tinySpec()).links;
  assert.deepEqual(a.segments.from, b.segments.from);
  assert.deepEqual(a.segments.to, b.segments.to);
  assert.deepEqual(a.arcs.weight, b.arcs.weight);
  for (const list of [a.segments, a.arcs]) {
    const bucket = (p) => Math.min(2047, Math.floor(p * 2048));
    for (let i = 1; i < list.count; i++) assert.ok(bucket(list.priority[i]) >= bucket(list.priority[i - 1]), "prima i più importanti");
  }
  const capped = build(tinySpec(), { maxSegments: 50 }).links;
  assert.ok(capped.segments.count <= 50);
  assert.ok(capped.scale < 1 && capped.capped);
  const big = build(loadTestSpec());
  assert.ok(big.net.units > 55000 && big.net.units < 70000, `~60.000 neuroni (${big.net.units})`);
  assert.ok(big.links.segments.count <= LINK_LIMITS.maxSegments);
  assert.ok(big.links.segments.count > 100000, `${big.links.segments.count} segmenti`);
});

test("collegamenti: dense per posizione (stessa profondità) con la densità chiesta", () => {
  const { net, links } = build(tinySpec());
  const emb = net.sections[1];
  const att = net.sections[2];
  const dense = [];
  for (let i = 0; i < links.segments.count; i++) if (links.segments.link[i] === 1) dense.push(i);
  const possible = 5 * emb.neurons * att.slabs * att.neurons;
  near(dense.length / possible, 0.5, 0.08);
  for (const i of dense) {
    const a = decodeUnit(emb, links.segments.from[i] - emb.offset);
    const b = decodeUnit(att, links.segments.to[i] - att.offset);
    assert.equal(a.depth, b.depth, "un nn.Linear lavora token per token");
  }
  assert.deepEqual(pairDepths(3, 3), [[0, 0], [1, 1], [2, 2]]);
  assert.deepEqual(pairDepths(1, 2), [[0, 0], [0, 1]]);
  assert.deepEqual(pairDepths(4, 2), [[0, 0], [1, 0], [2, 1], [3, 1]]);
});

test("collegamenti: lookup dal token acceso; attention causale; residui uno per posizione", () => {
  const { net, values, links } = build(tinySpec());
  const [inp, , att] = net.sections;
  for (let i = 0; i < links.segments.count; i++) {
    if (links.segments.link[i] !== 0) continue;
    assert.equal(values.raw[links.segments.from[i]], 1, "parte dall'unità accesa");
  }
  let attention = 0;
  let residual = 0;
  const perQuery = new Map();
  for (let i = 0; i < links.arcs.count; i++) {
    const kind = links.arcs.kind[i];
    if (kind === ARC_KIND.attention) {
      attention++;
      const key = decodeUnit(att, links.arcs.from[i] - att.offset);
      const query = decodeUnit(att, links.arcs.to[i] - att.offset);
      assert.ok(key.depth <= query.depth, "mai verso il futuro");
      assert.equal(key.slab, 0);
      assert.equal(query.slab, 1);
      assert.ok(links.arcs.weight[i] >= LINK_LIMITS.arcMinWeight);
      const id = `${links.arcs.head[i]}:${query.depth}`;
      perQuery.set(id, (perQuery.get(id) ?? 0) + 1);
    }
    if (kind === ARC_KIND.residual) residual++;
  }
  assert.ok(attention > 0);
  assert.ok(Math.max(...perQuery.values()) <= LINK_LIMITS.arcKeysPerQuery);
  assert.equal(residual, inp.depth.count, "un arco del residuo per token");
});

test("collegamenti: instradamento solo verso gli esperti attivi; dal router, il neurone e va solo all'esperto e", () => {
  const { net, values, links } = build(tinySpec());
  const router = net.sections[4];
  const exp = net.sections[5];
  const route = values.routing.get("exp");
  let toExperts = 0;
  for (let i = 0; i < links.segments.count; i++) {
    const link = links.segments.link[i];
    if (link !== 6 && link !== 7) continue;
    const u = link === 6 ? links.segments.to[i] : links.segments.from[i];
    const { slab, depth } = decodeUnit(exp, u - exp.offset);
    assert.equal(route.mask[slab * exp.depth.count + depth], 1, "esperto spento per quel token");
    if (link === 6 && slab >= exp.shared) {
      const r = decodeUnit(router, links.segments.from[i] - router.offset);
      assert.equal(r.neuron, slab - exp.shared, "punteggio dell'esperto → esperto");
      assert.equal(r.depth, depth);
      toExperts++;
    }
  }
  assert.equal(toExperts, exp.depth.count * exp.active * exp.neurons, "la scelta del router si vede tutta");
});

test("collegamenti: convoluzione locale, down/up, pooling e broadcast", () => {
  const { net, links } = build({
    title: "g",
    scale: "s",
    sections: [
      { id: "a", label: "A", role: "conv", neurons: 2, depth: [6, 6] },
      { id: "b", label: "B", role: "conv", neurons: 2, depth: [6, 6] },
      { id: "c", label: "C", role: "conv", neurons: 2, depth: [3, 3] },
      { id: "d", label: "D", role: "conv", neurons: 2, depth: [6, 6] },
      { id: "p", label: "P", role: "latent", neurons: 2 },
      { id: "q", label: "Q", role: "conv", neurons: 2, depth: [6, 6] },
    ],
    links: [
      { from: "a", to: "b", pattern: "conv", kernel: 3, density: 1 },
      { from: "b", to: "c", pattern: "down" },
      { from: "c", to: "d", pattern: "up" },
      { from: "d", to: "p", pattern: "pool" },
      { from: "p", to: "q", pattern: "broadcast" },
    ],
  });
  const [a, b, c, d, p, q] = net.sections;
  const count = [0, 0, 0, 0, 0];
  for (let i = 0; i < links.segments.count; i++) {
    const l = links.segments.link[i];
    count[l]++;
    const from = links.segments.from[i];
    const to = links.segments.to[i];
    if (l === 0) {
      const x = decodeUnit(a, from - a.offset);
      const y = decodeUnit(b, to - b.offset);
      assert.ok(Math.abs(x.row - y.row) <= 1 && Math.abs(x.col - y.col) <= 1, "solo i vicini 3 × 3");
    } else if (l === 1) {
      const x = decodeUnit(b, from - b.offset);
      const y = decodeUnit(c, to - c.offset);
      assert.deepEqual([Math.floor(x.row / 2), Math.floor(x.col / 2), x.neuron], [y.row, y.col, y.neuron]);
    } else if (l === 2) {
      const x = decodeUnit(c, from - c.offset);
      const y = decodeUnit(d, to - d.offset);
      assert.deepEqual([x.row, x.col], [Math.floor(y.row / 2), Math.floor(y.col / 2)]);
    } else if (l === 3) {
      assert.equal(decodeUnit(d, from - d.offset).neuron, decodeUnit(p, to - p.offset).neuron);
    } else if (l === 4) {
      assert.equal(decodeUnit(p, from - p.offset).neuron, decodeUnit(q, to - q.offset).neuron);
    }
  }
  // conv con densità 1: 36 posizioni × 2 canali × 2 canali × finestra (9 dentro, meno ai bordi)
  const window = (n) => (n === 0 || n === 5 ? 2 : 3);
  let expected = 0;
  for (let r = 0; r < 6; r++) for (let col = 0; col < 6; col++) expected += window(r) * window(col) * 4;
  assert.equal(count[0], expected);
  assert.equal(count[1], 36 * 2, "down: ogni posizione grande va nella sua piccola");
  assert.equal(count[2], 36 * 2, "up: ogni posizione grande riceve dalla sua piccola");
  assert.equal(count[3], 36 * 2, "pool: tutte le posizioni in una");
  assert.equal(count[4], 36 * 2, "broadcast: una in tutte");
});

test("collegamenti: pesi veri (explicit) con il loro segno; 'selected' va sulla fetta scelta", () => {
  const sec = (id) => ({ id, label: id, role: "mlp", neurons: 3, depth: true });
  const pairs = { from: [0, 1, 2], to: [2, 1, 0], weight: [0.9, -0.1, -0.5] };
  const { links } = build({ title: "e", scale: "s", depth: 4, sections: [sec("a"), sec("b")], links: [{ from: "a", to: "b", pattern: "explicit", pairs }] });
  assert.equal(links.segments.count, 0);
  assert.equal(links.sliceSegments.count, 3);
  assert.deepEqual(Array.from(links.sliceSegments.weight), [0.9, -0.5, -0.1].map(Math.fround), "prima i pesi più forti");
  const all = build({ title: "e", scale: "s", depth: 4, sections: [sec("a"), sec("b")], links: [{ from: "a", to: "b", pattern: "explicit", pairs, slice: "all" }] }).links;
  assert.equal(all.segments.count, 3 * 4);
  const one = build({ title: "e", scale: "s", depth: 4, sections: [sec("a"), sec("b")], links: [{ from: "a", to: "b", pattern: "explicit", pairs, slice: 2 }] });
  assert.equal(one.links.segments.count, 3);
  const b = one.net.sections[1];
  assert.equal(decodeUnit(b, one.links.segments.to[0] - b.offset).depth, 2);
});

// ------------------------------------------------------------------------------------------ indice sullo schermo
test("indice sullo schermo: il neurone più vicino alla camera vince; fuori dal raggio niente", () => {
  // matrice ortografica semplice (per colonne): x, y in [-1, 1] → schermo; w = 1 + z (profondità)
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[11] = 1; // w = z + 1 (riga 4, colonna 3)
  m[15] = 1;
  const positions = new Float32Array([0, 0, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0, -2]);
  const index = new ScreenIndex();
  const kept = index.build(positions, 4, m, 200, 200, { pickRadius: () => 0.05, pxPerUnit: 100, minPx: 6 });
  assert.equal(kept, 3, "l'unità con w ≤ 0 è dietro la camera");
  // le unità 0 e 1 proiettano vicino al centro: vince la 1 (w = 1, davanti alla 0 con w = 1,5)
  assert.equal(index.query(100, 100), 1);
  assert.equal(index.query(100, 100, { prefer: (i) => i === 0 }), 0, "la fetta scelta ha la precedenza");
  assert.equal(index.query(10, 10), -1);
  const p2 = index.sx[2];
  assert.equal(index.query(p2 + 2, index.sy[2] - 1), 2);
});

// ------------------------------------------------------------------------------------------ esempi
test("esempi: le tre reti e la prova di carico sono valide, oneste sulla scala, con fonti", () => {
  for (const [name, spec] of Object.entries(DEMO_SPECS)) {
    const { net, links } = build(spec);
    assert.deepEqual(net.warnings, [], `${name}: nessun avviso`);
    assert.ok(net.scale.length > 20, `${name}: frase sulla scala`);
    assert.ok(net.sources.length >= 2, `${name}: fonti`);
    for (const s of net.sections) assert.ok(s.text.length > 10, `${name}/${s.id}: testo della sezione`);
    assert.ok(links.arcs.count > 0, `${name}: meccanismi in profondità`);
    assert.equal(links.scale, 1, `${name}: niente tetto`);
    for (const l of net.links) assert.ok(PATTERNS[l.pattern]);
  }
  const llm = normalizeSpec(DEMO_SPECS.llm);
  assert.deepEqual(llm.repeats.map((r) => r.count), [3, 58], "3 layer densi + 58 MoE = 61");
  const unet = normalizeSpec(DEMO_SPECS.unet);
  assert.ok(unet.links.some((l) => l.pattern === "attention" && !l.within), "cross-attention al testo");
  assert.ok(unet.links.some((l) => l.pattern === "down") && unet.links.some((l) => l.pattern === "up"));
  const audio = normalizeSpec(DEMO_SPECS.audio);
  assert.ok(audio.sections.some((s) => s.role === "codebook"));
  assert.equal(audio.repeats[0].count, 7);
});
