// Test delle rappresentazioni del tab Audio e musica (moduli puri, niente three.js né DOM):
//   node --test webapp/tests/audio-reps.test.mjs
// Rete neurale (js/tabs/audio/network.js): validazione del nucleo senza avvisi per i due brani, ordine e
// conteggi delle sezioni, forme uguali ai numeri dichiarati (EnCodec, MusicGen small, Whisper tiny, codec del
// tab), schemi giusti (attention causale o no, cross-attention verso la sezione giusta, convoluzioni, passo 2,
// collegamenti con i pesi veri delle voci, delay pattern), limiti con margine, valori veri uguali a quelli
// calcolati dal laboratorio del tab (lab.js e codec.js, ricalcolati qui in modo indipendente).
// Cervello (js/tabs/audio/brain.js): strict senza avvisi, tutto nell'atlante, fonti citate con un link.
// Ricerca e installazione: voci view:rete e view:cervello, pannello Cervello prima di Fonti, goto.
// BRAIN_RESEARCH=/percorso/brain_data.json confronta anche i testi con la ricerca (senza, quel test si salta).
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { atlasKind } from "../js/core/brain/atlas.js";
import { resolveNeuronType } from "../js/core/brain/morphology.js";
import { normalizeBrainSpec, safeUrl, STRENGTHS } from "../js/core/brain/spec.js";
import { LINK_LIMITS } from "../js/core/neuralNet/links.js";
import { REP_LINKS } from "../js/core/repLinks.js";
import { buildNetworkModel } from "../js/core/neuralNet/model.js";
import { LIMITS, normalizeSpec } from "../js/core/neuralNet/spec.js";
import { prepareEntry, search } from "../js/core/search.js";
import { brainSpec } from "../js/tabs/audio/brain.js";
import { CODEC, nearest, rvqDecode } from "../js/tabs/audio/codec.js";
import { WHISPER } from "../js/tabs/audio/dsp.js";
import { createLab } from "../js/tabs/audio/lab.js";
import {
  ENCODEC,
  FEATURE_BINS,
  MEL_STEP,
  MUSICGEN,
  POSITION_CHANNELS,
  WHISPER_TEXT,
  WHISPER_TINY,
  WINDOW_FRAMES,
  buildAudioNetworkSpec,
  codecWindow,
  musicgenWindow,
  onsetFrame,
  whisperWindow,
  windowStart,
} from "../js/tabs/audio/network.js";
import { searchIndex } from "../js/tabs/audio/search.js";

const RESEARCH = process.env.BRAIN_RESEARCH ?? "";
const source = (path) => readFile(new URL(path, import.meta.url), "utf8");

// Il laboratorio vero del tab, calcolato una volta (circa 2 s): gli stessi numeri del browser.
const lab = await createLab();
const STYLES = Object.keys(lab.songs);
const specs = Object.fromEntries(STYLES.map((style) => [style, buildAudioNetworkSpec(lab, { style })]));
const nets = Object.fromEntries(STYLES.map((style) => [style, normalizeSpec(specs[style])]));
const net = nets[STYLES[0]];
const sectionOf = (id, n = net) => n.byId.get(id);
const linksOf = (pattern, n = net) => n.links.filter((l) => l.pattern === pattern);
const pair = (l) => `${l.from.id}>${l.to.id}`;
/** Valore della sezione s al neurone n (lastra slab) e alla posizione d, nell'ordine del nucleo. */
const valueAt = (s, n, d, slab = 0) => s.values[(slab * s.neurons + n) * s.depth.count + d];

// ------------------------------------------------------------------------------------------ rete neurale

test("rete: la specifica passa la validazione del nucleo senza avvisi per i due brani, con margine sui limiti", () => {
  assert.ok(STYLES.length >= 2, "il tab ha due brani");
  for (const style of STYLES) {
    const n = nets[style];
    assert.deepEqual(n.warnings, [], style);
    const model = buildNetworkModel(specs[style]);
    assert.equal(model.links.capped, false, "nessun collegamento tagliato dal tetto di sicurezza");
    assert.equal(model.links.scale, 1, "densità come scritte nella specifica");
    assert.ok(n.units < LIMITS.units / 10, `neuroni ${n.units}`);
    assert.ok(model.links.segments.count <= 16000, `segmenti ${model.links.segments.count}: tutti disegnati già alla densità iniziale`);
    assert.ok(model.links.segments.count < LINK_LIMITS.maxSegments / 10);
    assert.ok(model.links.arcs.count < LINK_LIMITS.maxArcs / 10, `archi ${model.links.arcs.count}`);
    assert.ok(model.links.sliceSegments.count < 2000, `pesi veri sulla fetta: ${model.links.sliceSegments.count}`);
    assert.ok(n.sources.length >= 12 && n.sources.every((s) => /^https:\/\//.test(s.url)));
    assert.ok(n.scale.length > 300 && n.note.startsWith("Valori veri"), "scala e onestà dichiarate");
    assert.equal(n.simulated, true, "non tutto è vero: il badge dice «in parte vere»");
  }
});

test("rete: ordine da sinistra a destra (EnCodec, codec del tab, MusicGen, Whisper) e blocchi ripetuti", () => {
  assert.deepEqual(net.items.map((i) => i.id), [
    "ec_wave", "ec_conv", "ec_blocks", "ec_latent", "ec_cb1", "ec_r1", "ec_cb2", "ec_rvq_x7", "ec_quant", "ec_dec", "ec_up", "ec_out",
    "toy_wave", "toy_feat", "toy_cb1", "toy_r1", "toy_cb2", "toy_r2", "toy_cb3", "toy_r3", "toy_cb4", "toy_sum",
    "mg_tokens", "mg_emb", "mg_text", "mg_attn", "mg_add1", "mg_cross", "mg_add2", "mg_ffn", "mg_add3", "mg_x24", "mg_norm", "mg_heads",
    "wh_mel", "wh_conv1", "wh_conv2", "wh_pos", "wh_in", "wh_attn", "wh_add1", "wh_mlp", "wh_add2", "wh_enc_x4", "wh_enc_out",
    "wh_tok", "wh_temb", "wh_dattn", "wh_dadd1", "wh_cross", "wh_dadd2", "wh_dmlp", "wh_dadd3", "wh_dec_x4", "wh_dln", "wh_logits",
  ]);
  assert.equal(net.sections.length, 52);
  const repeats = Object.fromEntries(net.repeats.map((r) => [r.id, [r.count, r.of.map((s) => s.id)]]));
  assert.deepEqual(repeats, {
    ec_rvq_x7: [7, ["ec_r1", "ec_cb2"]], // 8 codebook a 6 kbps: il primo + 7 coppie
    mg_x24: [MUSICGEN.layers, ["mg_attn", "mg_add1", "mg_cross", "mg_add2", "mg_ffn", "mg_add3"]],
    wh_enc_x4: [WHISPER_TINY.layers, ["wh_attn", "wh_add1", "wh_mlp", "wh_add2"]],
    wh_dec_x4: [WHISPER_TINY.layers, ["wh_dattn", "wh_dadd1", "wh_cross", "wh_dadd2", "wh_dmlp", "wh_dadd3"]],
  });
  // da lontano si leggono i nomi delle quattro reti: le loro parentesi vengono prima delle sotto-parti
  assert.deepEqual(net.groups.slice(0, 4).map((g) => [g.from.id, g.to.id]), [["ec_wave", "ec_out"], ["toy_wave", "toy_sum"], ["mg_tokens", "mg_heads"], ["wh_mel", "wh_logits"]]);
});

test("rete: i numeri dichiarati sono coerenti tra loro e con le fonti del tab", () => {
  // EnCodec 24 kHz: passi 2·4·5·8 = 320 → 75 frame/s; canali che raddoppiano; banda = frame/s × codebook × 10 bit
  assert.equal(ENCODEC.encoderStrides.reduce((a, b) => a * b, 1), ENCODEC.hop);
  assert.deepEqual([...ENCODEC.decoderStrides].reverse(), [...ENCODEC.encoderStrides]);
  assert.equal(ENCODEC.sampleRate / ENCODEC.hop, ENCODEC.frameRate);
  assert.deepEqual(ENCODEC.channels, [32, 64, 128, 256, 512]);
  assert.equal(Math.log2(ENCODEC.codebookSize), 10);
  ENCODEC.bandwidths.forEach((kbps, i) => assert.equal(kbps * 1000, ENCODEC.frameRate * ENCODEC.codebooks[i] * 10));
  // MusicGen small e Whisper tiny
  assert.deepEqual([MUSICGEN.layers, MUSICGEN.dim, MUSICGEN.heads, MUSICGEN.ffn, MUSICGEN.codebooks, MUSICGEN.card], [24, 1024, 16, 4096, 4, 2048]);
  assert.deepEqual([...MUSICGEN.delays], [0, 1, 2, 3]);
  assert.equal(WHISPER_TINY.mlp, 4 * WHISPER_TINY.width);
  assert.equal(WHISPER_TINY.audioPositions, WHISPER.nFrames / 2, "la seconda convoluzione ha passo 2");
  // MusicGen small, un layer (audiocraft/modules/transformer.py, niente bias): self-attention e cross-attention hanno 4
  // matrici dim × dim ciascuna, il feed-forward 2 matrici dim × ffn. Il feed-forward è quindi metà dei pesi del layer,
  // non "la maggior parte": il testo della sezione deve dire così
  assert.equal(2 * MUSICGEN.dim * MUSICGEN.ffn, 2 * 4 * MUSICGEN.dim ** 2);
  assert.match(sectionOf("mg_ffn").text, /metà dei pesi del layer/);
  assert.doesNotMatch(sectionOf("mg_ffn").text, /maggior parte/);
  // l'errore relativo del laboratorio è misurato sui frame dei due brani insieme (lab.js: trainRvq su entrambi)
  assert.doesNotMatch(sectionOf("toy_r2").text, /sul brano/);
  // le stesse cifre compaiono nelle fonti del tab (content.js), non sono state scritte a parte
  return source("../js/tabs/audio/content.js").then((content) => {
    for (const fact of ["8·5·4·2 = 320", "75 al secondo", "1024 voci", "2048 voci per codebook", "ritardi 0, 1, 2, 3", "3000 frame", "1500 posizioni", "80 bande Mel", "39 milioni"]) assert.ok(content.includes(fact), fact);
  });
});

test("rete: le forme disegnate seguono i numeri veri (sottotitoli, campi real, profondità)", () => {
  const s = (id) => sectionOf(id);
  // stessi 240 ms ovunque: 12 frame da 20 ms; EnCodec 18 frame da 13,3 ms (36 passi da 6,7 ms); log-Mel 24 frame da 10 ms
  assert.equal(net.slices.count, WINDOW_FRAMES);
  assert.equal(WINDOW_FRAMES * (1000 / lab.frameRate), 240);
  assert.equal(s("ec_latent").depth.count, (0.24 * ENCODEC.frameRate));
  assert.equal(s("ec_wave").depth.count, 2 * s("ec_latent").depth.count);
  assert.equal(s("wh_mel").depth.count, 0.24 * (WHISPER.sampleRate / WHISPER.hop));
  for (const id of ["toy_wave", "toy_feat", "toy_cb1", "toy_sum", "mg_tokens", "mg_heads", "wh_conv2", "wh_pos", "wh_enc_out"]) assert.equal(s(id).depth.count, WINDOW_FRAMES, id);
  assert.equal(s("mg_text").depth.count, lab.songs[STYLES[0]].style.prompt.split(" ").length, "una posizione per parola del prompt");
  assert.deepEqual(s("wh_tok").depth.names, WHISPER_TEXT.map((t) => t.replace(/^ /, "·")));
  // codec del tab: 17 dei 257 numeri (uno ogni 250 Hz), tutte le 64 voci, 8 campioni per frame
  assert.deepEqual(FEATURE_BINS, Array.from({ length: 17 }, (_, i) => 16 * i));
  assert.equal((FEATURE_BINS.at(-1) * lab.sampleRate) / CODEC.nFft, 4000);
  for (const id of ["toy_feat", "toy_r1", "toy_r2", "toy_r3", "toy_sum"]) assert.deepEqual([s(id).neurons, s(id).real], [17, CODEC.bins], id);
  for (let l = 1; l <= CODEC.used; l++) {
    const cb = s(`toy_cb${l}`);
    assert.deepEqual([cb.role, cb.activation, cb.neurons], ["codebook", "onehot", lab.rvq.k], cb.id);
    assert.equal(cb.neuronNames[cb.neurons - 1], `voce ${lab.rvq.k - 1}`, "il neurone n è la voce n: niente scarti di uno");
  }
  // EnCodec, MusicGen, Whisper: i campi real e i sottotitoli dicono le dimensioni vere
  const real = (id) => s(id).real;
  assert.deepEqual([real("ec_conv"), real("ec_blocks"), real("ec_latent"), real("ec_cb1"), real("ec_cb2"), real("ec_dec")], [32, 512, 128, 1024, 1024, 512]);
  assert.deepEqual([real("mg_emb"), real("mg_attn"), real("mg_ffn"), real("mg_text")], [1024, 1024, 4096, 768]);
  assert.deepEqual([real("wh_conv1"), real("wh_attn"), real("wh_mlp"), real("wh_tok"), real("wh_logits")], [384, 384, 1536, 51865, 51865]);
  assert.match(s("ec_blocks").sub, /2 · 4 · 5 · 8 = 320/);
  assert.match(s("ec_up").sub, /8 · 5 · 4 · 2/);
  assert.match(s("ec_cb1").sub, /1\.024 voci/);
  assert.match(s("mg_ffn").sub, /4\.096 neuroni/);
  assert.match(s("mg_attn").sub, /16 head · qui 4/);
  assert.match(s("wh_mel").sub, /80 bande × 3\.000 frame/);
  assert.match(s("wh_conv2").sub, /3\.000 → 1\.500 posizioni/);
  assert.match(s("wh_attn").sub, /6 head · qui 3/);
  for (const x of net.sections) assert.ok(/\d/.test(x.sub), `sottotitolo con la forma vera: ${x.id}`);
  // le head dividono i neuroni in fasce uguali; le lastre di MusicGen sono i 4 codebook
  for (const x of net.sections.filter((x) => x.heads > 1)) assert.equal(x.neurons % x.heads, 0, x.id);
  assert.deepEqual([s("mg_tokens").slabs, s("mg_heads").slabs], [MUSICGEN.codebooks, MUSICGEN.codebooks]);
  assert.deepEqual(s("mg_tokens").slabNames, ["codebook 1 (ritardo 0)", "codebook 2 (ritardo 1)", "codebook 3 (ritardo 2)", "codebook 4 (ritardo 3)"]);
});

test("rete: attention causale nei decoder, bidirezionale nell'encoder di Whisper, cross-attention verso la sezione giusta", () => {
  const attention = linksOf("attention").map((l) => [pair(l), l.within, l.causal]);
  assert.deepEqual(attention, [
    ["mg_attn>mg_attn", true, true], // MusicGen: ogni passo guarda solo il passato
    ["mg_text>mg_cross", false, false], // cross-attention: key e value dalle parole del prompt
    ["wh_attn>wh_attn", true, false], // encoder di Whisper: tutte le direzioni
    ["wh_dattn>wh_dattn", true, true], // decoder di Whisper: causale come il GPT
    ["wh_enc_out>wh_cross", false, false], // cross-attention: key e value dall'uscita dell'encoder
  ]);
  assert.equal(sectionOf("mg_text").depth.label, "parola del prompt");
  assert.equal(sectionOf("wh_cross").depth.label, "token di testo");
  // le query della cross-attention vengono dal flusso del decoder
  assert.ok(net.links.some((l) => pair(l) === "mg_add1>mg_cross" && l.label === "query"));
  assert.ok(net.links.some((l) => pair(l) === "wh_dadd1>wh_cross" && l.label === "query"));
});

test("rete: convoluzioni tra frame vicini, tempo che si accorcia e si riallunga, quantizzazione con i pesi veri, residui", () => {
  assert.deepEqual(linksOf("conv").map((l) => [pair(l), l.kernel]), [["ec_wave>ec_conv", 3], ["ec_up>ec_out", 3], ["toy_wave>toy_feat", 3], ["wh_mel>wh_conv1", 3]]);
  assert.deepEqual(linksOf("down").map((l) => [pair(l), l.factor]), [["ec_conv>ec_blocks", 2], ["wh_conv1>wh_conv2", 2]]);
  assert.deepEqual(linksOf("up").map((l) => [pair(l), l.factor]), [["ec_dec>ec_up", 2]]);
  assert.deepEqual(linksOf("explicit").map(pair), ["toy_feat>toy_cb1", "toy_r1>toy_cb2", "toy_r2>toy_cb3", "toy_r3>toy_cb4"]);
  assert.ok(linksOf("explicit").every((l) => l.slice === "selected"), "i pesi veri si disegnano sul frame scelto");
  assert.deepEqual(linksOf("lookup").map((l) => `${pair(l)}${l.from.slabs > 1 ? `@${l.fromSlab}` : ""}`), [
    "ec_cb1>ec_r1", "ec_cb1>ec_quant", "ec_cb2>ec_quant",
    "toy_cb1>toy_r1", "toy_cb2>toy_r2", "toy_cb3>toy_r3", "toy_cb1>toy_sum", "toy_cb2>toy_sum", "toy_cb3>toy_sum", "toy_cb4>toy_sum",
    "mg_tokens>mg_emb@0", "mg_tokens>mg_emb@1", "mg_tokens>mg_emb@2", "mg_tokens>mg_emb@3", // i 4 embedding sommati
    "wh_tok>wh_temb",
  ]);
  assert.deepEqual(linksOf("residual").map(pair), [
    "ec_latent>ec_r1", "toy_feat>toy_r1", "toy_r1>toy_r2", "toy_r2>toy_r3",
    "mg_emb>mg_add1", "mg_add1>mg_add2", "mg_add2>mg_add3",
    "wh_conv2>wh_in", "wh_in>wh_add1", "wh_add1>wh_add2", "wh_temb>wh_dadd1", "wh_dadd1>wh_dadd2", "wh_dadd2>wh_dadd3",
  ]);
  assert.deepEqual(linksOf("route"), [], "nessun instradamento MoE: le voci dei codebook non sono esperti");
});

test("rete: valori veri uguali a quelli calcolati dal laboratorio del tab (codec, voce più vicina, somma)", () => {
  const T = lab.frames;
  const { dim, k, codebooks } = lab.rvq;
  for (const style of STYLES) {
    const n = nets[style];
    const start = windowStart(lab, style);
    const codes = lab.codes[style];
    const features = lab.features[style].features;
    const samples = lab.songs[style].samples;
    const decoded = rvqDecode(lab.rvq, codes, T, CODEC.used); // il decoder del tab (somma delle voci)
    for (let d = 0; d < WINDOW_FRAMES; d++) {
      const t = start + d;
      for (let i = 0; i < 8; i++) assert.equal(valueAt(sectionOf("toy_wave", n), i, d), samples[t * 320 - 160 + 20 + 40 * i], "8 campioni del tratto del frame");
      // residui ricalcolati qui, in Float32 come rvqEncode
      const r = Float32Array.from(features.subarray(t * dim, (t + 1) * dim));
      FEATURE_BINS.forEach((bin, b) => assert.equal(valueAt(sectionOf("toy_feat", n), b, d), features[t * dim + bin]));
      for (let l = 0; l < CODEC.used; l++) {
        const q = codes[l * T + t];
        assert.equal(nearest(codebooks[l], k, r, 0, dim), q, `${style} frame ${t} codebook ${l + 1}: la voce accesa è la più vicina`);
        const cb = sectionOf(`toy_cb${l + 1}`, n);
        for (let c = 0; c < k; c++) assert.equal(valueAt(cb, c, d), c === q ? 1 : 0);
        for (let x = 0; x < dim; x++) r[x] -= codebooks[l][q * dim + x];
        if (l < CODEC.used - 1) FEATURE_BINS.forEach((bin, b) => assert.equal(valueAt(sectionOf(`toy_r${l + 1}`, n), b, d), r[bin], `residuo ${l + 1}`));
      }
      FEATURE_BINS.forEach((bin, b) => assert.ok(Math.abs(valueAt(sectionOf("toy_sum", n), b, d) - decoded[t * dim + bin]) < 1e-5, "somma = rvqDecode"));
    }
    // collegamenti verso i codebook: pesi veri delle voci (centroidi), verso le voci usate nella finestra
    const window = codecWindow(lab, style, start);
    linksOf("explicit", n).forEach((link, l) => {
      const used = new Set(window.choices[l]);
      assert.equal(link.pairs.count, FEATURE_BINS.length * used.size);
      for (let i = 0; i < link.pairs.count; i++) {
        const c = link.pairs.to[i];
        assert.ok(used.has(c), "solo le voci usate");
        assert.equal(link.pairs.weight[i], codebooks[l][c * dim + FEATURE_BINS[link.pairs.from[i]]]);
      }
    });
  }
});

test("rete: MusicGen legge i token veri sfasati dal delay pattern e dà le probabilità del modello a conteggi", () => {
  const T = lab.frames;
  for (const style of STYLES) {
    const n = nets[style];
    const start = windowStart(lab, style);
    const codes = lab.codes[style];
    const tokens = sectionOf("mg_tokens", n);
    const heads = sectionOf("mg_heads", n);
    const cb1 = sectionOf("toy_cb1", n);
    for (let q = 0; q < MUSICGEN.codebooks; q++) {
      for (let d = 0; d < WINDOW_FRAMES; d++) {
        const s = start + d;
        const token = codes[q * T + (s - q)]; // passo s, codebook q → frame s − q
        let lit = 0;
        for (let c = 0; c < lab.rvq.k; c++) lit += valueAt(tokens, c, d, q);
        assert.equal(lit, 1, "un token per codebook e per passo");
        assert.equal(valueAt(tokens, token, d, q), 1);
        const p = lab.model.distribution(style, q, codes, T, s + 1 - q, 1);
        let sum = 0;
        for (let c = 0; c < lab.rvq.k; c++) {
          assert.ok(Math.abs(valueAt(heads, c, d, q) - p[c]) < 1e-6);
          sum += valueAt(heads, c, d, q);
        }
        assert.ok(Math.abs(sum - 1) < 1e-4, "le probabilità di ogni testa sommano a 1");
      }
    }
    // il codebook 1 di MusicGen è lo stesso token del codebook 1 del codec allo stesso frame (ritardo 0) …
    for (let d = 0; d < WINDOW_FRAMES; d++) for (let c = 0; c < lab.rvq.k; c++) assert.equal(valueAt(tokens, c, d, 0), valueAt(cb1, c, d));
    // … e il codebook k ritrova al passo d + k il token che il codebook k del codec ha al frame d
    const mg = musicgenWindow(lab, style, start);
    const cw = codecWindow(lab, style, start);
    for (let q = 1; q < MUSICGEN.codebooks; q++) for (let d = 0; d + q < WINDOW_FRAMES; d++) assert.equal(mg.read[q][d + q], cw.choices[q][d], `delay del codebook ${q + 1}`);
  }
});

test("rete: Whisper riceve il log-Mel vero del brano e la codifica di posizione vera; il testo è l'esempio della scena", async () => {
  for (const style of STYLES) {
    const n = nets[style];
    const start = windowStart(lab, style);
    const lm = lab.whisper(style);
    const mel = sectionOf("wh_mel", n);
    assert.equal(mel.neurons, WHISPER.nMels / MEL_STEP);
    for (let b = 0; b < mel.neurons; b++) for (let j = 0; j < 2 * WINDOW_FRAMES; j++) assert.equal(valueAt(mel, b, j), lm.values[(2 * start + j) * WHISPER.nMels + b * MEL_STEP]);
    assert.ok(2 * start + 2 * WINDOW_FRAMES <= lm.signalFrames, "la finestra sta nel brano, non nel silenzio di riempimento");
    // sinusoids(length, channels) di whisper/model.py, scritta qui di nuovo: seni nei primi 192 canali, coseni negli altri
    const pos = sectionOf("wh_pos", n);
    const half = WHISPER_TINY.width / 2;
    const inc = Math.log(10000) / (half - 1);
    const channels = [...POSITION_CHANNELS.map((i) => [Math.sin, i]), ...POSITION_CHANNELS.map((i) => [Math.cos, i])];
    channels.forEach(([f, i], c) => {
      for (let d = 0; d < WINDOW_FRAMES; d++) assert.ok(Math.abs(valueAt(pos, c, d) - f((start + d) * Math.exp(-inc * i))) < 1e-6);
    });
    const tok = sectionOf("wh_tok", n);
    for (let i = 0; i < WHISPER_TEXT.length; i++) for (let j = 0; j < WHISPER_TEXT.length; j++) assert.equal(valueAt(tok, i, j), i === j ? 1 : 0);
    assert.equal(tok.valueLabel, "(token d'esempio)", "il valore dice che è un esempio");
  }
  // lo stesso testo della scena: il modulo di soli dati che la scena di Whisper importa e riesporta
  const { WHISPER_TOKENS } = await import("../js/tabs/audio/whisperTokens.js");
  assert.deepEqual([...WHISPER_TOKENS], [...WHISPER_TEXT]);
  assert.match(await source("../js/tabs/audio/sceneWhisper.js"), /import \{ WHISPER_TOKENS \} from "\.\/whisperTokens\.js";/);
  const w = whisperWindow(lab, STYLES[0]);
  assert.deepEqual(w.channels, [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352]);
});

test("rete: valori veri solo dove il tab calcola davvero, tutto il resto simulato e dichiarato", () => {
  const real = net.sections.filter((s) => s.values).map((s) => s.id);
  assert.deepEqual(real, ["toy_wave", "toy_feat", "toy_cb1", "toy_r1", "toy_cb2", "toy_r2", "toy_cb3", "toy_r3", "toy_cb4", "toy_sum", "mg_tokens", "mg_heads", "wh_mel", "wh_pos", "wh_tok"]);
  assert.ok(net.sections.filter((s) => s.id.startsWith("ec_")).every((s) => !s.values), "EnCodec non gira nel browser");
  assert.match(sectionOf("mg_heads").text, /modello a conteggi del tab/);
  assert.match(sectionOf("ec_wave").text, /Valori simulati/);
  assert.match(sectionOf("mg_text").text, /Valori simulati/);
  assert.match(sectionOf("wh_logits").text, /Probabilità simulate/);
  assert.match(sectionOf("wh_tok").text, /le parole sono un esempio/);
  assert.match(net.note, /Tutto il resto è una simulazione didattica/);
  assert.match(net.note, /sulle parole del prompt e sui token di testo non ha un significato/);
});

test("rete: la finestra comincia 6 frame prima della seconda nota e il codebook 1 cambia voce lì", () => {
  for (const style of STYLES) {
    const start = windowStart(lab, style);
    const onset = onsetFrame(lab, style);
    assert.equal(start, onset - 6, style);
    assert.ok(start >= 3 && start + WINDOW_FRAMES < lab.frames, "token veri per tutti i ritardi, e un passo dopo l'ultimo");
    const cb1 = codecWindow(lab, style, start).choices[0];
    assert.ok(cb1.slice(0, 4).every((q) => q === cb1[0]), `${style}: la voce resta uguale finché suona la stessa nota (${cb1.join(" ")})`);
    const changes = [];
    for (let d = 1; d < WINDOW_FRAMES; d++) if (cb1[d] !== cb1[d - 1]) changes.push(d);
    assert.ok(changes.some((d) => Math.abs(d - 6) <= 1), `${style}: cambia attorno all'attacco (${changes.join(",")})`);
    // il testo della scheda dice anche che dopo l'attacco la voce cambia ancora qualche volta (il volume della nota
    // nuova si assesta): nessun cambio prima dell'attacco, almeno uno dopo
    assert.ok(changes.every((d) => d >= 5) && changes.some((d) => d > 7), `${style}: cambi solo attorno e dopo l'attacco (${changes.join(",")})`);
    // nomi delle fette: gli istanti dei frame, gli stessi del log-Mel (due frame di Whisper per frame)
    const n = nets[style];
    assert.deepEqual(n.slices.names, Array.from({ length: WINDOW_FRAMES }, (_, d) => `${(start + d) * 20} ms`));
    assert.equal(sectionOf("wh_mel", n).depth.names[2 * 3], n.slices.names[3]);
    assert.match(sectionOf("toy_cb1", n).text, new RegExp(`attorno a ${onset * 20} ms`));
  }
});

test("rete: deterministica e testi semplici (niente HTML né emoji), con il prompt come testo", () => {
  const plain = (spec) => JSON.stringify(spec, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v));
  assert.equal(plain(buildAudioNetworkSpec(lab, { style: STYLES[0] })), plain(specs[STYLES[0]]));
  assert.notEqual(plain(specs[STYLES[0]]), plain(specs[STYLES[1]]), "un altro brano, altri numeri");
  for (const spec of Object.values(specs)) {
    const texts = [spec.title, spec.intro, spec.scale, spec.note, ...spec.sections.flatMap((s) => [s.label, s.sub, s.text]), ...spec.groups.flatMap((g) => [g.label, g.text])].filter(Boolean);
    for (const t of texts) {
      assert.ok(!/<\/?[a-z][^>]*>/i.test(t.replace(/<\|[a-z]+\|>/g, "")), `niente HTML: ${t.slice(0, 60)}`);
      assert.ok(!/\p{Extended_Pictographic}/u.test(t), `niente emoji: ${t.slice(0, 60)}`);
    }
    for (const s of spec.sections.filter((x) => x.label)) assert.ok(s.text && s.text.length > 20, `ogni sezione ha il suo testo: ${s.id}`);
  }
  assert.ok(specs[STYLES[0]].intro.includes(lab.songs[STYLES[0]].style.prompt));
});

// ------------------------------------------------------------------------------------------ cervello

test("cervello: strict senza avvisi, regioni, marcatori e percorsi nell'atlante, tipi di neurone noti", () => {
  const spec = normalizeBrainSpec(brainSpec, { strict: true });
  assert.deepEqual(spec.warnings, []);
  for (const id of brainSpec.regions) assert.equal(atlasKind(id), "region", id);
  for (const id of brainSpec.markers) assert.equal(atlasKind(id), "marker", id);
  for (const p of brainSpec.pathways) assert.equal(atlasKind(typeof p === "string" ? p : p.id), "pathway");
  for (const a of brainSpec.analogies) {
    for (const r of a.regions) assert.ok(["region", "marker"].includes(atlasKind(r)), `${a.id}: ${r}`);
    for (const m of a.markers ?? []) assert.equal(atlasKind(m), "marker", `${a.id}: ${m}`);
    for (const p of a.pathways ?? []) assert.equal(atlasKind(p), "pathway", `${a.id}: ${p}`);
  }
  for (const g of brainSpec.circuit.groups) {
    assert.equal(atlasKind(g.region), "region", g.id);
    if (g.marker) assert.equal(atlasKind(g.marker), "marker", g.id);
    assert.ok(resolveNeuronType(g.neuron_type), `${g.id}: ${g.neuron_type}`);
  }
  for (const e of brainSpec.circuit.edges) if (e.pathway) assert.equal(atlasKind(e.pathway), "pathway");
  // la via uditiva: dalla coclea alla corteccia, nell'ordine del flusso (come la rete AI)
  assert.deepEqual(spec.circuit.groups.map((g) => g.id), ["ohc", "ihc", "sgn", "cn", "so", "ic", "mgn", "a1", "stg", "wern"]);
  assert.equal(spec.circuit.edges.length, 12);
  assert.ok(spec.circuit.caveat.length > 50, "l'avvertenza della ricerca resta");
  assert.deepEqual(spec.pathways.map((p) => p.id), ["arcuate", "auditory_radiation", "mesolimbic"]);
});

test("cervello: ogni analogia dice cosa ha in comune e dove non regge, con forza e fonti con un link", () => {
  assert.equal(brainSpec.analogies.length, 9);
  for (const a of normalizeBrainSpec(brainSpec, { strict: true }).analogies) {
    assert.ok(a.ai && a.brain && a.text.length > 80, a.id);
    assert.ok(a.similar.length > 20 && a.different.length > 40, `${a.id}: In comune e Dove non regge`);
    assert.ok(a.strength in STRENGTHS, a.id);
    assert.ok(a.sources.length >= 1 && a.sources.every((s) => safeUrl(s.url)), `${a.id}: fonti con un link`);
  }
});

test("cervello: le fonti citate da circuito, fatti e introduzione esistono e hanno un link", () => {
  const circuitSources = new Set(brainSpec.circuit.sources.map((s) => s.id));
  for (const g of brainSpec.circuit.groups) if (g.real_count_source) assert.ok(circuitSources.has(g.real_count_source), `${g.id}: fonte del conteggio`);
  for (const s of brainSpec.circuit.sources) assert.ok(s.title && safeUrl(s.url), s.id);
  assert.ok(brainSpec.facts.length >= 5);
  for (const f of brainSpec.facts) assert.ok(f.claim && f.value && f.source?.title && safeUrl(f.source.url), f.claim);
  for (const s of brainSpec.sources) assert.ok(s.title && s.authors && s.year && safeUrl(s.url), s.id);
  // i numeri del circuito hanno il loro fatto con la stessa fonte (Ashmore 2008, Makary et al. 2011)
  for (const id of ["ashmore2008", "makary2011"]) assert.ok(brainSpec.facts.some((f) => f.source.id === id), id);
});

test("cervello: i testi sono quelli della ricerca verificata, parola per parola", { skip: !(RESEARCH && existsSync(RESEARCH)) && "BRAIN_RESEARCH non indicato (la ricerca sta fuori dal repository)" }, async () => {
  const data = JSON.parse(await readFile(RESEARCH, "utf8"));
  const r = data.tabs.audio;
  // L'unica aggiunta ai testi della ricerca: la cautela della verifica in fondo alla nota di ogni fonte segnata
  // "da ricontrollare" (verification.checked_sources), con le parole della verifica (round E, richiesta 31).
  const flagged = new Map(data.verification.checked_sources.filter((v) => /ricontrollare/.test(v.status)).map((v) => [v.id, v.note]));
  const withCautions = (v) => {
    if (Array.isArray(v)) return v.map(withCautions);
    if (!v || typeof v !== "object") return v;
    const out = Object.fromEntries(Object.entries(v).map(([k, x]) => [k, withCautions(x)]));
    if (typeof out.title === "string" && flagged.has(out.id) && !/ricontrollar/i.test(out.note ?? "")) {
      out.note = [out.note, `Da ricontrollare (verifica della ricerca dell'8 novembre 2026): ${flagged.get(out.id)}`].filter(Boolean).join(" ");
    }
    return out;
  };
  assert.equal(brainSpec.intro, r.intro);
  assert.deepEqual([brainSpec.regions, brainSpec.markers, brainSpec.pathways], [r.regions, r.markers, r.pathways]);
  assert.deepEqual(brainSpec.analogies, withCautions(r.analogies));
  const { layout, ...circuit } = r.circuit;
  assert.ok(layout, "il campo descrittivo della ricerca esiste ed è l'unico tolto");
  assert.deepEqual(brainSpec.circuit, withCautions(circuit));
  for (const f of brainSpec.facts) {
    const global = data.facts.find((y) => y.value === f.value && y.source.id === f.source.id);
    const typed = Object.values(data.neuron_types).flatMap((t) => t.data).find((d) => d.text === f.value && d.source?.id === f.source.id);
    assert.ok(global || typed, `fatto della ricerca: ${f.claim}`);
  }
});

// ------------------------------------------------------------------------------------------ ricerca e installazione

test("ricerca: voci view:rete e view:cervello trovabili con le parole dei meccanismi e delle regioni", () => {
  const index = searchIndex.map((e) => prepareEntry({ ...e, tab: "audio" }));
  const top = (q) => search(index, q)[0]?.entry;
  for (const q of ["rete neurale audio", "delay pattern rete neurale", "codebook residuo rete", "attention bidirezionale Whisper rete", "convoluzione stride EnCodec"]) {
    assert.deepEqual([top(q)?.kind, top(q)?.id], ["view", "rete"], q);
  }
  for (const q of ["cervello audio", "coclea", "cellule ciliate", "corteccia uditiva", "giro di Heschl", "genicolato mediale"]) {
    assert.deepEqual([top(q)?.kind, top(q)?.id], ["view", "cervello"], q);
  }
  assert.equal(searchIndex.find((e) => e.id === "rete").title, "Rete neurale (Audio e musica)");
  assert.equal(searchIndex.find((e) => e.id === "cervello").title, "Cervello (Audio e musica)");
});

test("installazione: rappresentazioni dopo i pannelli del tab e prima di Fonti, goto che comincia da reps.goto", async () => {
  assert.deepEqual({ ...REP_LINKS }, { rete: "network", cervello: "brain" }, "gli id delle voci sono quelli del nucleo (core/repLinks.js)");
  const index = await source("../js/tabs/audio/index.js");
  const install = index.indexOf("installRepresentations(ui");
  assert.ok(install > index.indexOf('addPane({ id: "viste"'), "dopo i pannelli del tab");
  assert.ok(install < index.indexOf('addPane({ id: "fonti"'), "il pannello Cervello sta prima delle Fonti");
  assert.match(index, /async function goto\(target\) \{[^}]*?const shown = reps\.goto\(target\);\s+if \(shown\) return shown;/, "goto comincia con reps.goto");
  assert.match(index, /import\("\.\/brain\.js"\)\.then\(\(m\) => m\.brainSpec\)/);
  assert.match(index, /buildAudioNetworkSpec\(lab, \{ style: state\.style \}\)\),\s+cache: false,/, "la rete si costruisce a ogni apertura con i numeri del laboratorio");
  assert.match(index, /if \(reps\.current === "network"\) void reps\.refresh\("network"\);/, "cambiando brano la rete aperta si rifà (stessa scena, stessa camera)");
});
