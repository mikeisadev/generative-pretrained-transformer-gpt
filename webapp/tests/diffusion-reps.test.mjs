// Test delle rappresentazioni del tab Immagini e video (moduli di soli dati, niente three.js né DOM):
//   node --test webapp/tests/diffusion-reps.test.mjs
// Rete neurale (js/tabs/diffusion/network.js): validazione del nucleo senza avvisi, ordine e forme delle
// sezioni uguali ai numeri dichiarati (v1-inference.yaml, CLIP), schemi giusti (attention causale nel testo,
// non causale nella U-Net, cross-attention dal testo, skip connection, broadcast del passo t, maschere del
// video), limiti con margine, valori veri uguali a quelli calcolati dalle funzioni del tab (math.js).
// Cervello (js/tabs/diffusion/brain.js): strict senza avvisi, tutto nell'atlante, fonti valide.
// Ricerca e installazione: voci view:rete e view:cervello, pannello Cervello prima di Fonti, goto.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { atlasKind } from "../js/core/brain/atlas.js";
import { resolveNeuronType } from "../js/core/brain/morphology.js";
import { STRENGTHS, safeUrl, normalizeBrainSpec } from "../js/core/brain/spec.js";
import { buildNetworkModel } from "../js/core/neuralNet/model.js";
import { LINK_LIMITS } from "../js/core/neuralNet/links.js";
import { LIMITS, normalizeSpec } from "../js/core/neuralNet/spec.js";
import { prepareEntry, search } from "../js/core/search.js";
import { brainSpec } from "../js/tabs/diffusion/brain.js";
import { fillGaussian, mulberry32, samplingTimesteps, timestepEmbedding } from "../js/tabs/diffusion/math.js";
import {
  FIRST_T,
  GENERATION,
  TEXT_TOKENS,
  TIME_INDICES,
  UNET_LEVELS,
  VIDEO_FRAMES,
  VIDEO_PATCHES,
  buildDiffusionNetworkSpec,
  initialNoiseCrop,
  networkSpec,
  timestepValues,
  videoAttentionMask,
} from "../js/tabs/diffusion/network.js";
import { searchIndex } from "../js/tabs/diffusion/search.js";

const source = (path) => readFile(new URL(path, import.meta.url), "utf8");
const net = normalizeSpec(networkSpec);
const sectionOf = (id) => net.byId.get(id);
const linksOf = (pattern) => net.links.filter((l) => l.pattern === pattern);

// ------------------------------------------------------------------------------------------ rete neurale

test("rete: la specifica passa la validazione del nucleo senza avvisi, con margine sui limiti", () => {
  assert.deepEqual(net.warnings, []);
  const model = buildNetworkModel(networkSpec);
  assert.equal(model.links.capped, false, "nessun collegamento tagliato dal tetto di sicurezza");
  assert.equal(model.links.scale, 1, "densità come scritte nella specifica");
  assert.ok(net.units < LIMITS.units / 10, `neuroni ${net.units}`);
  assert.ok(model.links.segments.count < LINK_LIMITS.maxSegments / 10, `segmenti ${model.links.segments.count}`);
  assert.ok(model.links.arcs.count < LINK_LIMITS.maxArcs / 2, `archi ${model.links.arcs.count}`);
  assert.ok(net.sources.length >= 10 && net.sources.every((s) => /^https:\/\//.test(s.url)));
  assert.ok(net.scale.length > 100 && net.note.includes("Valori veri"), "scala e onestà dichiarate");
});

test("rete: ordine da sinistra a destra (testo, U-Net, scheduler e VAE, video) e ripetizioni", () => {
  const order = net.items.map((i) => i.id);
  assert.deepEqual(order, [
    "txt_tokens", "txt_embed", "txt_attn", "txt_add1", "txt_mlp", "txt_add2", "txt_x12", "txt_out",
    "un_zt", "un_t", "un_tmlp", "un_conv_in",
    "un_e0_res", "un_e0_attn", "un_e0_x2", "un_e1_res", "un_e1_attn", "un_e1_x2", "un_e2_res", "un_e2_attn", "un_e2_x2", "un_e3_res", "un_e3_x2",
    "un_m_res1", "un_m_attn", "un_m_res2",
    "un_d3_res", "un_d3_x3", "un_d2_res", "un_d2_attn", "un_d2_x3", "un_d1_res", "un_d1_attn", "un_d1_x3", "un_d0_res", "un_d0_attn", "un_d0_x3",
    "un_eps", "sch", "vae_mid", "vae_up", "vae_out",
    "vid_in", "vid_spatial", "vid_add1", "vid_temporal", "vid_add2", "vid_out",
  ]);
  assert.equal(net.sections.length, 39);
  const repeats = Object.fromEntries(net.repeats.map((r) => [r.id, [r.count, r.of.map((s) => s.id)]]));
  assert.deepEqual(repeats.txt_x12, [12, ["txt_attn", "txt_add1", "txt_mlp", "txt_add2"]]);
  for (const L of [0, 1, 2]) assert.deepEqual(repeats[`un_e${L}_x2`], [2, [`un_e${L}_res`, `un_e${L}_attn`]], "encoder: num_res_blocks 2");
  for (const L of [0, 1, 2]) assert.deepEqual(repeats[`un_d${L}_x3`], [3, [`un_d${L}_res`, `un_d${L}_attn`]], "decoder: num_res_blocks + 1");
  assert.deepEqual(repeats.un_e3_x2, [2, ["un_e3_res"]], "il livello più piccolo non ha attention");
  assert.deepEqual(repeats.un_d3_x3, [3, ["un_d3_res"]]);
});

test("rete: le forme disegnate seguono i numeri veri (canali 1 ogni 320, griglia divisa per 8, mai sotto 2 × 2)", () => {
  const sideOf = (res) => Math.max(2, res / 8);
  for (const L of UNET_LEVELS) {
    for (const prefix of [`un_e${L.level}`, `un_d${L.level}`]) {
      const ids = L.attention ? [`${prefix}_res`, `${prefix}_attn`] : [`${prefix}_res`];
      for (const id of ids) {
        const s = sectionOf(id);
        assert.equal(s.depth.kind, "grid", id);
        assert.deepEqual([s.depth.rows, s.depth.cols], [sideOf(L.res), sideOf(L.res)], `${id}: griglia`);
        assert.equal(s.neurons, L.channels / 320, `${id}: canali disegnati`);
        assert.equal(s.real, L.channels, `${id}: canali veri`);
        assert.ok(s.sub.includes(`${L.res} × ${L.res}`), `${id}: forma vera nel sottotitolo`);
      }
    }
  }
  assert.deepEqual(UNET_LEVELS.map((L) => L.channels), [320, 640, 1280, 1280], "model_channels 320 × channel_mult [1, 2, 4, 4]");
  assert.deepEqual(UNET_LEVELS.map((L) => L.attention), [true, true, true, false], "attention_resolutions [4, 2, 1]");
  // latente, rumore previsto, scheduler: 4 canali (tutti), 64 × 64 diviso per 8
  for (const id of ["un_zt", "un_eps", "sch"]) assert.deepEqual([sectionOf(id).neurons, sectionOf(id).depth.rows, sectionOf(id).depth.cols], [4, 8, 8], id);
  // testo: 10 token, 12 dei 768 numeri, 4 delle 12 head, MLP 4 volte più largo
  const txt = (id) => sectionOf(id);
  assert.equal(net.depth.count, TEXT_TOKENS.length);
  assert.deepEqual([txt("txt_attn").heads, txt("txt_attn").real, txt("txt_mlp").neurons, txt("txt_mlp").real], [4, 768, 48, 3072]);
  assert.equal(txt("txt_tokens").real, 49408);
  assert.equal(txt("txt_tokens").activation, "onehot");
  assert.ok(txt("txt_out").sub.includes("77 × 768"));
  // passo t: 8 dei 320 numeri, poi l'MLP fino a 1.280
  assert.deepEqual([sectionOf("un_t").neurons, sectionOf("un_t").real, sectionOf("un_tmlp").real], [8, 320, 1280]);
  // VAE: la griglia raddoppia una volta (8 → 16), l'immagine ha 3 canali
  assert.deepEqual([sectionOf("vae_mid").depth.cols, sectionOf("vae_up").depth.cols, sectionOf("vae_out").depth.cols, sectionOf("vae_out").neurons], [8, 16, 16, 3]);
  // video: una riga per frame, una colonna per patch
  for (const id of ["vid_in", "vid_spatial", "vid_add1", "vid_temporal", "vid_add2", "vid_out"]) assert.deepEqual([sectionOf(id).depth.rows, sectionOf(id).depth.cols], [VIDEO_FRAMES, VIDEO_PATCHES], id);
});

test("rete: attention causale nel testo, non causale nella U-Net, cross-attention dal testo in ogni blocco", () => {
  const within = linksOf("attention").filter((l) => l.within);
  const self = Object.fromEntries(within.map((l) => [l.from.id, l]));
  assert.equal(self.txt_attn.causal, true, "CLIP: build_attention_mask");
  const unetAttention = net.sections.filter((s) => s.id.startsWith("un_") && s.role === "attention").map((s) => s.id);
  assert.deepEqual(unetAttention, ["un_e0_attn", "un_e1_attn", "un_e2_attn", "un_m_attn", "un_d2_attn", "un_d1_attn", "un_d0_attn"]);
  for (const id of unetAttention) {
    assert.equal(self[id].causal, false, `${id}: un'immagine non ha un prima e un dopo`);
    const cross = linksOf("attention").filter((l) => !l.within && l.to.id === id);
    assert.equal(cross.length, 1, `${id}: una cross-attention`);
    assert.equal(cross[0].from.id, "txt_out", `${id}: key e value dal testo codificato`);
    assert.deepEqual([self[id].heads, cross[0].heads], [2, 1], `${id}: head disegnate (self 2, cross 1, dichiarate nel sottotitolo)`);
    assert.ok(sectionOf(id).sub.includes("8 head, qui 2 (cross: 1)"), id);
  }
  const crossTargets = linksOf("attention").filter((l) => !l.within).map((l) => l.to.id);
  assert.deepEqual(crossTargets, unetAttention, "nessun'altra sezione riceve il testo");
  assert.equal(self.vae_mid.heads, 1, "VAE: attention a una head");
});

test("rete: skip connection, griglia che si dimezza e raddoppia, broadcast del passo t", () => {
  const residual = linksOf("residual").map((l) => `${l.from.id}>${l.to.id}`);
  for (const [enc, dec] of [["un_e0_attn", "un_d0_res"], ["un_e1_attn", "un_d1_res"], ["un_e2_attn", "un_d2_res"], ["un_e3_res", "un_d3_res"]]) {
    assert.ok(residual.includes(`${enc}>${dec}`), `skip ${enc} → ${dec}`);
    assert.equal(sectionOf(enc).depth.count, sectionOf(dec).depth.count, "stessa griglia ai due capi");
  }
  assert.ok(residual.includes("un_zt>sch"), "lo scheduler riceve z_t");
  assert.ok(residual.includes("txt_embed>txt_add1") && residual.includes("txt_add1>txt_add2"), "residui del testo");
  const pairs = (pattern) => linksOf(pattern).map((l) => [l.from.id, l.to.id, l.from.depth.cols, l.to.depth.cols]);
  assert.deepEqual(pairs("down"), [["un_e0_attn", "un_e1_res", 8, 4], ["un_e1_attn", "un_e2_res", 4, 2]]);
  assert.deepEqual(pairs("up"), [["un_d2_attn", "un_d1_res", 2, 4], ["un_d1_attn", "un_d0_res", 4, 8], ["vae_mid", "vae_up", 8, 16]]);
  // il terzo dimezzamento e il primo raddoppio non si vedono (griglia minima 2 × 2) e il collegamento lo dice
  const hidden = net.links.filter((l) => l.pattern === "conv" && /resta 2 × 2/.test(l.label)).map((l) => `${l.from.id}>${l.to.id}`);
  assert.deepEqual(hidden, ["un_e2_attn>un_e3_res", "un_d3_res>un_d2_res"]);
  const broadcast = linksOf("broadcast");
  assert.ok(broadcast.every((l) => l.from.id === "un_tmlp"));
  assert.deepEqual(broadcast.map((l) => l.to.id).sort(), ["un_d0_res", "un_d1_res", "un_d2_res", "un_d3_res", "un_e0_res", "un_e1_res", "un_e2_res", "un_e3_res", "un_m_res1", "un_m_res2"]);
  assert.ok(linksOf("conv").every((l) => l.from.depth.cols === l.to.depth.cols && l.kernel === 3), "convoluzioni 3 × 3 sulla stessa griglia");
});

test("rete: maschere del video, spaziale lungo le righe (frame) e temporale lungo le colonne (patch)", () => {
  const n = VIDEO_FRAMES * VIDEO_PATCHES;
  for (const kind of ["spatial", "temporal"]) {
    const w = videoAttentionMask(kind);
    assert.equal(w.length, n * n);
    for (let q = 0; q < n; q++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += w[q * n + k];
        const same = kind === "spatial" ? Math.floor(q / VIDEO_PATCHES) === Math.floor(k / VIDEO_PATCHES) : q % VIDEO_PATCHES === k % VIDEO_PATCHES;
        assert.equal(w[q * n + k] > 0, same, `${kind} q=${q} k=${k}`);
      }
      assert.ok(Math.abs(sum - 1) < 1e-6);
    }
  }
  const links = Object.fromEntries(linksOf("attention").filter((l) => l.within && l.from.id.startsWith("vid_")).map((l) => [l.from.id, l]));
  assert.deepEqual([...links.vid_spatial.weights], [...videoAttentionMask("spatial")]);
  assert.deepEqual([...links.vid_temporal.weights], [...videoAttentionMask("temporal")]);
  assert.ok(sectionOf("vid_spatial").text.includes("tutti uguali"), "pesi uguali dichiarati");
});

test("rete: valori veri uguali a quelli calcolati dalle funzioni del tab, il resto dichiarato simulato", async () => {
  // il primo passo del campionatore con le impostazioni del tab (20 passi, spaziatura leading): t = 951
  assert.equal(FIRST_T, samplingTimesteps(1000, GENERATION.steps)[0]);
  assert.equal(FIRST_T, 951);
  const index = await source("../js/tabs/diffusion/index.js");
  assert.match(index, new RegExp(`settings: \\{ seed: ${GENERATION.seed}, scale: [\\d.]+, steps: ${GENERATION.steps} \\}`), "stesse impostazioni del tab");
  // rumore iniziale: lo stesso fillGaussian dell'azione Generazione (latente 4 × 32 × 32), angolo in alto a sinistra 8 × 8
  const full = fillGaussian(new Float32Array(4 * 32 * 32), mulberry32(GENERATION.seed));
  const crop = initialNoiseCrop();
  for (let c = 0; c < 4; c++) for (let r = 0; r < 8; r++) for (let col = 0; col < 8; col++) assert.equal(crop[c * 64 + r * 8 + col], full[c * 1024 + r * 32 + col]);
  assert.deepEqual([...sectionOf("un_zt").values], [...crop]);
  // passo t: 8 dei 320 numeri di timestep_embedding (coseni e seni a frequenze 1, 1/10, 1/100, 1/1000)
  const emb = timestepEmbedding(new Float32Array(320), FIRST_T);
  const expected = [...TIME_INDICES.map((i) => emb[i]), ...TIME_INDICES.map((i) => emb[160 + i])];
  assert.deepEqual([...sectionOf("un_t").values], [...Float32Array.from(expected)]);
  assert.ok(Math.abs(sectionOf("un_t").values[0] - Math.cos(951)) < 1e-6 && Math.abs(sectionOf("un_t").values[7] - Math.sin(0.951)) < 1e-6);
  assert.deepEqual([...timestepValues(999)], [...timestepValues(999)], "pura");
  // tutto il resto è simulato (la vista lo dichiara: "simulazione didattica")
  const { values } = buildNetworkModel(networkSpec);
  assert.deepEqual(net.sections.filter((s) => !values.simulated.has(s.id)).map((s) => s.id), ["un_zt", "un_t"]);
  assert.ok(net.simulated);
});

test("rete: deterministica, token del prompt come nel tab, testi semplici", async () => {
  const a = buildNetworkModel(buildDiffusionNetworkSpec());
  const b = buildNetworkModel(buildDiffusionNetworkSpec());
  assert.deepEqual([...a.links.segments.from], [...b.links.segments.from]);
  assert.deepEqual([...a.links.arcs.weight], [...b.links.arcs.weight]);
  assert.deepEqual([...a.values.raw], [...b.values.raw]);
  // i primi 9 token sono quelli della scena (pipeline.js); il decimo è una posizione di riempimento
  const pipeline = await source("../js/tabs/diffusion/pipeline.js");
  const sceneTokens = JSON.parse(pipeline.match(/export const TOKENS = (\[[^\]]*\]);/)[1]);
  assert.deepEqual(TEXT_TOKENS.slice(0, 9), sceneTokens.slice(0, 9));
  assert.deepEqual(net.slices.names, [...TEXT_TOKENS], "le fette della scheda sono le parole del prompt");
  // niente HTML (a parte i nomi dei token speciali) e niente emoji: la vista mostra tutto come testo
  const texts = [net.title, net.intro, net.scale, net.note, ...net.sections.flatMap((s) => [s.label, s.sub, s.text]), ...net.repeats.map((r) => r.text), ...net.groups.map((g) => g.label + g.text)];
  for (const t of texts) {
    assert.ok(!/<(?!inizio>|fine>)\/?[a-z][^>]*>/i.test(t), `HTML in: ${t.slice(0, 60)}`);
    assert.ok(!/\p{Extended_Pictographic}/u.test(t), `emoji in: ${t.slice(0, 60)}`);
  }
  for (const s of net.sections) assert.ok(s.text.length > 20, `${s.id}: testo della scheda`);
});

// ------------------------------------------------------------------------------------------ cervello

test("cervello: strict senza avvisi, regioni, marcatori e percorsi nell'atlante", () => {
  const spec = normalizeBrainSpec(brainSpec, { strict: true });
  assert.deepEqual(spec.warnings, []);
  for (const id of brainSpec.regions) assert.equal(atlasKind(id), "region", id);
  for (const id of brainSpec.markers) assert.equal(atlasKind(id), "marker", id);
  for (const p of brainSpec.pathways) assert.equal(atlasKind(typeof p === "string" ? p : p.id), "pathway");
  for (const a of brainSpec.analogies) {
    for (const r of a.regions) assert.ok(["region", "marker"].includes(atlasKind(r)), `${a.id}: ${r}`);
    for (const m of a.markers ?? []) assert.equal(atlasKind(m), "marker");
    for (const p of a.pathways ?? []) assert.equal(atlasKind(p), "pathway");
  }
  for (const g of brainSpec.circuit.groups) {
    assert.equal(atlasKind(g.region), "region", g.id);
    if (g.marker) assert.equal(atlasKind(g.marker), "marker", g.id);
  }
  assert.deepEqual(spec.circuit.groups.map((g) => g.id), ["photo", "rgc", "lgn", "v1", "v2v4", "it", "pfc"], "dalla retina alla prefrontale, da sinistra a destra");
});

test("cervello: ogni analogia dice dove non regge, con forza e fonti valide", () => {
  const spec = normalizeBrainSpec(brainSpec, { strict: true });
  assert.equal(spec.analogies.length, 10);
  for (const a of spec.analogies) {
    for (const field of ["ai", "brain", "text", "similar", "different"]) assert.ok(a[field].length > 10, `${a.id}.${field}`);
    assert.ok(a.strength in STRENGTHS, a.id);
    assert.ok(a.sources.length >= 1 && a.sources.every((s) => s.title && s.url && safeUrl(s.url)), `${a.id}: fonti`);
  }
  assert.ok(spec.intro.includes("fisica"), "l'introduzione dice che la diffusione viene dalla fisica, non dal cervello");
});

test("cervello: circuito, fatti e fonti citate esistono", async () => {
  const { circuit, facts, sources } = brainSpec;
  for (const g of circuit.groups) {
    assert.ok(resolveNeuronType(g.neuron_type), `${g.id}: tipo di neurone`);
    assert.ok(g.count_shown >= 1 && g.count_shown <= 18);
    if (g.real_count_source) assert.ok(circuit.sources.some((s) => s.id === g.real_count_source), `${g.id}: fonte del conteggio`);
  }
  const ids = new Set(circuit.groups.map((g) => g.id));
  for (const e of circuit.edges) assert.ok(ids.has(e.from) && ids.has(e.to));
  for (const m of circuit.analogy_map) assert.ok(ids.has(m.group));
  assert.ok(facts.length >= 3);
  for (const f of facts) assert.ok(f.claim && f.value && safeUrl(f.source?.url), f.claim);
  for (const s of [...sources, ...circuit.sources]) assert.ok(s.title && safeUrl(s.url), s.title);
  const header = await source("../js/tabs/diffusion/brain.js");
  assert.match(header, /FILE GENERATO/);
  assert.match(header, /brain_data\.json/);
});

// ------------------------------------------------------------------------------------------ ricerca e installazione

test("ricerca: voci view:rete e view:cervello trovabili con le parole dei meccanismi e delle regioni", () => {
  const index = searchIndex.map((e) => prepareEntry({ ...e, tab: "diffusion" }));
  const top = (q) => search(index, q)[0]?.entry.id;
  assert.equal(top("rete neurale"), "rete");
  assert.equal(top("cross-attention neuroni"), "rete");
  assert.equal(top("cervello"), "cervello");
  assert.equal(top("corteccia visiva V1"), "cervello");
  for (const id of ["rete", "cervello"]) assert.equal(searchIndex.filter((e) => e.kind === "view" && e.id === id).length, 1);
});

test("installazione: rappresentazioni prima di Fonti e goto che comincia da reps.goto", async () => {
  const index = await source("../js/tabs/diffusion/index.js");
  const install = index.indexOf("installRepresentations(ui");
  assert.ok(install > 0 && install < index.indexOf('addPane({ id: "fonti"'), "il pannello Cervello sta prima delle Fonti");
  assert.ok(install > index.indexOf('addPane({ id: "viste"'), "dopo i pannelli del tab");
  assert.match(index, /import\("\.\/network\.js"\)\.then\(\(m\) => m\.networkSpec\)/);
  assert.match(index, /import\("\.\/brain\.js"\)\.then\(\(m\) => m\.brainSpec\)/);
  assert.match(index, /async function goto\(target\) \{[\s\S]{0,400}?const shown = reps\.goto\(target\);\s*if \(shown\) return shown;/);
});
