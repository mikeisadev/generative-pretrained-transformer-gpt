// Fedeltà della vista "Architettura LLM moderni completa" alla ricerca verificata (llm_frontier.md e
// llm_specs.json dell'8/11/2026). Completa tests/modern-llm-full.test.mjs: là i totali, qui le formule
// rifatte dal config, i nomi dei tensori confrontati con gli elenchi della ricerca e le regole di
// onestà sui testi (valori non verificati marcati, simulazioni dichiarate, assenze dette come tali).
import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { FULL_ACTIONS } from "../js/tabs/modern-llm/full/catalog.js";
import { FULL_PATH } from "../js/tabs/modern-llm/full/content.js";
import { addOnTotal, anatomyCards, buildStrip, cardState, formatCount, layerByNumber, layerList, layerParams, paramTotals } from "../js/tabs/modern-llm/full/model.js";
import { MODEL_BY_ID, MODELS } from "../js/tabs/modern-llm/full/models/index.js";
import { CAPABILITY_FACTS } from "../js/tabs/modern-llm/full/research.js";
import { FULL_SNIPPETS } from "../js/tabs/modern-llm/full/snippets.js";

// le didascalie delle azioni stanno in full/actions.js, che porta con sé three.js (shapes.js): three.js
// è in webapp/vendor (nel browser lo trova l'importmap), qui un hook di risoluzione come in weights.test.mjs
const VENDOR = pathToFileURL(fileURLToPath(new URL("../vendor/three/", import.meta.url))).href;
register(
  "data:text/javascript," +
    encodeURIComponent(`export async function resolve(specifier, context, next) {
      if (specifier === "three") return { url: ${JSON.stringify(`${VENDOR}three.module.min.js`)}, shortCircuit: true };
      if (specifier.startsWith("three/addons/")) return { url: ${JSON.stringify(`${VENDOR}addons/`)} + specifier.slice(13), shortCircuit: true };
      return next(specifier, context);
    }`),
);
const { fullActions } = await import("../js/tabs/modern-llm/full/actions.js");

/** Azioni che mostrano pezzi propri di V4.1 (tornano al modello principale): solo con V4.1. */
const V41_ONLY = new Set(["full-kv", "full-ced", "full-engram", "full-dspark"]);

/** Titoli e didascalie di tutte le azioni per un modello, costruite come fa il player (senza scena). */
async function actionCaptions(m) {
  const strip = buildStrip(m);
  const fx = {
    model: m,
    strip,
    blockIndex: (id) => strip.blocks.findIndex((b) => b.id === id),
    segmentBlocks: (id) => {
      const s = strip.segments.find((x) => x.id === id);
      return s ? strip.blocks.map((_, i) => i).filter((i) => i >= s.from && i <= s.to && strip.blocks[i].type === "layer") : [];
    },
  };
  const out = [];
  for (const a of fullActions(fx)) {
    if (V41_ONLY.has(a.id) && m !== V41) continue;
    for (const s of a.steps ?? (await a.build())) out.push({ where: `${m.id}/${a.id}/${s.title}`, text: `${s.title}. ${s.caption}` });
  }
  return out;
}

const V41 = MODEL_BY_ID.get("deepseek-v4.1-flash");
const KIMI = MODEL_BY_ID.get("kimi-k3.1-pro");
const OSS = MODEL_BY_ID.get("gpt-oss-120b");

/** Tutte le stringhe di un oggetto di dati, con il percorso (per i messaggi d'errore). */
function strings(value, where = "", out = []) {
  if (typeof value === "string") out.push({ where, text: value });
  else if (Array.isArray(value)) value.forEach((v, i) => strings(v, `${where}[${i}]`, out));
  else if (value && typeof value === "object") for (const [k, v] of Object.entries(value)) strings(v, `${where}.${k}`, out);
  return out;
}

/** "≈ 32,3 GFLOP per token (…)" → 32.3 */
const firstNumber = (text) => Number(text.match(/\d+(?:\.\d{3})*(?:,\d+)?/)[0].replace(/\./g, "").replace(",", "."));

test("KV cache per token: i byte rifatti dal config con le formule della ricerca (§4.14, §5, §6)", () => {
  // V4.1 (cache_engine.py di DeepInfer): voce FP4 = d/2 + d/16 (una scala ogni 16), chiave dell'indexer
  // = d_idx/2 + d_idx/32; voci per token = Σ 1/m sui layer sorgente (Full: m = 2 nell'encoder, 1 al 20)
  const c = V41.config;
  const entry = (c.headDim * 4) / 8 + c.headDim / 16;
  const key = c.indexHeadDim / 2 + c.indexHeadDim / 32;
  const entries = c.kvSourceLayers.reduce((a, l) => a + 1 / c.compressRatios[l], 0);
  assert.deepEqual([entry, key, entries], [V41.kv.entryBytes, V41.kv.indexKeyBytes, V41.kv.entriesPerToken]);
  assert.equal(entries * (entry + key), V41.kv.perToken); // 890
  // confronti: DeepSeek-V3 (MLA, 61 layer × 576 × 2 B) e DeepSeek LLM 67B (GQA, 95 × 2 × 8 × 128 × 2 B)
  const cmp = Object.fromEntries(V41.kv.compare.map((x) => [x.id, x.bytes]));
  assert.equal(cmp.v3, 61 * (512 + 64) * 2);
  assert.equal(cmp.llm67b, 95 * 2 * 8 * 128 * 2);
  assert.equal(V41.kv.ratio, Math.round(cmp.llm67b / V41.kv.perToken)); // 437
  // Kimi: solo i layer Gated MLA crescono col contesto, (kv_lora 512 + rope 64) valori in BF16
  const mla = [...KIMI.layers.kinds].filter((k) => k === "M").length;
  assert.equal(mla, 24);
  assert.equal(KIMI.kv.perToken, mla * (KIMI.config.kvLora + KIMI.config.ropeDim) * 2); // 27.648
  // gpt-oss: i layer densi (dispari) × K e V × 8 head × 64 × 2 B
  const dense = [...OSS.layers.kinds].filter((k) => k === "G").length;
  assert.equal(OSS.kv.perToken, dense * 2 * OSS.config.kvHeads * OSS.config.headDim * 2); // 36.864
});

test("FLOP per token ≈ 2 × parametri attivi, come scritto nei dati (V4.1 anche il prefill)", () => {
  for (const m of MODELS) {
    const t = paramTotals(m);
    const want = (2 * t.active) / 1e9;
    const got = firstNumber(m.flops.decode);
    assert.ok(Math.abs(got - want) / want < 0.005, `${m.id}: ${got} GFLOP contro 2 × ${t.active}`);
  }
  const prefill = (2 * paramTotals(V41).prefill) / 1e9;
  assert.ok(Math.abs(firstNumber(V41.flops.prefill) - prefill) < 0.05, `prefill: ${V41.flops.prefill}`);
  // nucleo dell'attention sparsa: 4·n_h·d_h per chiave, 38 layer compressi × (512 + 128) chiavi, 2 × 128
  const core = (4 * V41.config.heads * V41.config.headDim * (38 * (V41.config.indexTopK + V41.config.window) + 2 * V41.config.window)) / 1e9;
  assert.equal(firstNumber(V41.flops.core), Math.round(core * 10) / 10); // 3,2 (la ricerca ha corretto 3,4)
});

test("valori NON verificati dalla ricerca: dovunque compaiano, il testo dice che sono da verificare", () => {
  // 48 shard, attention sparsa a 64K, estensione a 1M dopo 34T token (llm_frontier.md §4.11, §4.13)
  const doubtful = /48 shard|64K|34T/;
  const texts = [...strings(V41, "V41"), ...strings(FULL_PATH, "PATH"), ...strings(FULL_ACTIONS, "ACTIONS")];
  for (const { where, text } of texts) {
    if (!doubtful.test(text) || /\.unverified\[\d+\]\.(value|why)$/.test(where)) continue;
    // V41.checkpoint.shards: il pannello Pesi gli mette accanto la pillola "da verificare" (panes.js)
    if (where === "V41.checkpoint.shards") continue;
    assert.match(text, /verificat|da verificare/i, `${where}: «${text.slice(0, 90)}…» senza la marcatura`);
  }
  // e i pezzi della Panoramica che li citano rimandano all'elenco
  const ids = new Set(V41.unverified.map((u) => u.id));
  assert.ok(V41.pipeline.find((s) => s.id === "context").unverified.every((id) => ids.has(id)));
  // il tipo dei pesi del gate di Engram è "BF16 (non verificato)" nella ricerca
  const gate = V41.anatomy.branch.cards.find((c) => c.id === "gate");
  assert.match(gate.dtype, /da verificare/);
});

test("simulazioni dichiarate: le scelte degli esperti non sono pubblicate", () => {
  assert.match(FULL_ACTIONS.find((a) => a.id === "full-router").summary, /simulat/);
  assert.match(FULL_PATH.flatMap((s) => s.children).find((c) => c.id === "full-moe").text, /simulazione didattica/);
});

test("affermazioni corrette dalla verifica: niente attribuzioni sbagliate né riuso inventato", async () => {
  const captions = (await Promise.all(MODELS.map(actionCaptions))).flat();
  const all = [...MODELS.flatMap((m) => strings(m, m.id)), ...strings(FULL_PATH, "PATH"), ...strings(FULL_SNIPPETS, "snippets"), ...captions];
  for (const { where, text } of all) {
    // DSpark propone i 5 token; la testa di Markov è una correzione di primo ordine dei logit (§4.9)
    assert.doesNotMatch(text, /markov[^.]*propon/i, `${where}: la bozza la propone DSpark, non la testa di Markov`);
    // i layer 0-1 (solo finestra) e DSpark non leggono il KV globale
    assert.doesNotMatch(text, /letto da tutti|letta da tutti/i, `${where}: i layer solo-finestra non leggono il KV globale`);
  }
  // nell'anatomia: nei layer senza compressione nessuna scheda del KV globale né dell'indexer
  const globalRows = new Set(["global", "index"]);
  for (const card of anatomyCards(V41).filter((c) => globalRows.has(c.row))) {
    for (const n of [0, 1]) assert.equal(cardState(card, layerByNumber(V41, n)), "off", `${card.id} nel layer ${n}`);
  }
  // il codice dichiarato "dal README" di safetensors è pseudocodice (il README descrive il formato)
  assert.match(FULL_SNIPPETS.safetensors.file, /^pseudocodice/);
});

/**
 * Nomi dei tensori elencati dalla ricerca (llm_frontier.md §4.13, §5, §6; llm_specs.json → checkpoint),
 * senza il prefisso del layer. Le graffe con le virgole si espandono, quelle con un indice ({e}, {L},
 * {0..383}) diventano {}.
 */
const RESEARCH_NAMES = {
  "deepseek-v4.1-flash": [
    "attn_norm.weight", "attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b}.weight", "attn.{wq_a,wq_b,wkv,wo_a,wo_b}.scale", "attn.attn_sink",
    "attn.compressor.{wkv,wgate,norm}.weight", "attn.indexer.{wq_b,weights_proj,wk,k_norm}.weight",
    "ffn_norm.weight", "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl", "ffn.shared_experts.{w1,w2,w3}.weight", "ffn.experts.{}.{w1,w2,w3}.{weight,scale}",
    "hc_{attn,ffn}_{base,fn,scale}", "engram.embed.{weight,scale}", "engram.wkv.weight", "engram.q_weight", "engram.k_weight",
    "embed.weight", "head.weight", "norm.weight", "image_start", "image_end", "image_newline",
    "mtp.{}.blocco", "mtp.0.main_proj.weight", "mtp.0.main_norm.weight", "mtp.2.norm.weight", "mtp.2.markov_head.{embed,head}.weight", "mtp.2.confidence_head.proj.weight",
    "vision.patch_embed.proj.weight", "vision.blocks.{}.{norm1,attn.wqkv,attn.wo,norm2,mlp.w1,mlp.w2}.weight", "vision.norm.weight", "aligner.{w1,w2}.{weight,bias}",
  ],
  "kimi-k3.1-pro": [
    "block_sparse_moe.gate.weight", "block_sparse_moe.gate.e_score_correction_bias", "block_sparse_moe.experts.{}.{w1,w2,w3}.{weight_packed,weight_scale}",
    "block_sparse_moe.shared_experts.x", "block_sparse_moe.{routed_expert_down_proj,routed_expert_norm,routed_expert_up_proj}",
    "self_attn.{q,k,v,g,b,f_a,f_b,o}_proj", "self_attn.{q,k,v}_conv1d", "self_attn.{A_log,dt_bias,o_norm}",
    "self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,g_proj,o_proj}", "self_attn.indexer.{wq_b,wk,weights_proj,k_norm}",
    "mlp.{gate,up,down}_proj", "language_model.model.layers.0.mlp.{gate,up,down}_proj.weight",
    "w_l", "index_topk", // pseudo-query di AttnRes (§5) e campo del dsa_config
  ],
  "gpt-oss-120b": [
    "input_layernorm.weight", "self_attn.{q,k,v,o}_proj.{weight,bias}", "self_attn.sinks", "post_attention_layernorm.weight", "mlp.router.{weight,bias}",
    "mlp.experts.gate_up_proj_{blocks,scales,bias}", "mlp.experts.down_proj_{blocks,scales,bias}", "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    "sliding_window", // campo del config
  ],
};

/** Espande le graffe con le virgole; gli indici ({e}, {L}, {N}, {b}, {i}, {0..383}) diventano {}. */
function expand(text) {
  const t = text.replace(/\{[^{}]*\.\.[^{}]*\}/g, "{}").replace(/\{[A-Za-z]\}/g, "{}");
  const m = t.match(/\{([^{}]*,[^{}]*)\}/);
  if (!m) return [t];
  return m[1].split(",").flatMap((alt) => expand(t.slice(0, m.index) + alt.trim() + t.slice(m.index + m[0].length)));
}

/** Nomi di tensore in un testo: identificatori con un punto (attn.wq_a.weight) o con un trattino basso. */
function tensorTokens(text) {
  return expand(text)
    .flatMap((t) => t.match(/[A-Za-z_][\w{}]*(?:\.[\w{}]+)+|\b[A-Za-z]+_[\w]+/g) ?? [])
    .map((t) => t.replace(/^(?:model\.)?layers\.(?:\{\}|\d+)\./, ""));
}

test("nomi dei tensori: solo quelli degli elenchi della ricerca (schede del Dettaglio e pannello Pesi)", () => {
  for (const m of MODELS) {
    const allowed = RESEARCH_NAMES[m.id].flatMap(expand);
    const texts = [
      ...anatomyCards(m).flatMap((c) => [c.tensor, c.tensor2, ...Object.values(c.alt ?? {}).map((a) => a.tensor)]),
      m.anatomy.residual.tensor,
      m.anatomy.norm.attnTensor,
      m.anatomy.norm.ffnTensor,
      ...m.tensorGroups.flatMap((g) => g.tensors.map(([name]) => name)),
    ].filter(Boolean);
    for (const text of texts)
      for (const token of tensorTokens(text)) assert.ok(allowed.some((name) => name.includes(token)), `${m.id}: «${token}» (da «${text}») non è un nome della ricerca`);
  }
});

test("didascalie delle azioni: numeri e strutture del modello mostrato", async () => {
  for (const m of MODELS) {
    const caps = await actionCaptions(m);
    const t = paramTotals(m);
    const moe = m.params.moe;
    const text = caps.map((c) => c.text).join("\n");
    assert.ok(text.includes(`Gli altri ${(moe.routed - moe.active).toLocaleString("it-IT")} non lavorano`), `${m.id}: esperti spenti`);
    const share = ((100 * t.active) / t.backbone).toLocaleString("it-IT", { maximumFractionDigits: 1 });
    assert.ok(text.includes(`${formatCount(t.active)} attivi su ${formatCount(t.backbone)} (${share}%)`), `${m.id}: quota attiva`);
    // esperti condivisi al plurale quando sono più di uno (Kimi: 2), nessuno in gpt-oss
    const sum = caps.find((c) => c.where.endsWith("/full-router/Somma pesata")).text;
    if (moe.shared > 1) assert.match(sum, new RegExp(`i ${moe.shared} esperti condivisi`));
    else if (moe.shared === 1) assert.match(sum, /l'esperto condiviso/);
    else assert.doesNotMatch(sum, /condivis/);
    if (moe.latent) assert.match(sum, /spazio latente/); // LatentMoE: somma nello spazio latente, poi W↑
    // i layer con FFN densa non passano dal MoE (Kimi, layer 1)
    for (const n of m.params.dense?.layers ?? []) assert.ok(caps.some((c) => c.where.includes("/full-token/") && c.text.includes(`il layer ${n} ha invece una FFN densa`)), `${m.id}: layer ${n} denso`);
  }
  // V4.1: le azioni proprie dicono i numeri della ricerca
  const v = (await actionCaptions(V41)).map((c) => c.text).join("\n");
  for (const fact of ["7,89B", "16,13B", "890 byte", "437 volte", "288 byte", "140 token", "196,6B", "183 GiB", "14,2B", "15,8 GFLOP", "32,3"]) assert.ok(v.includes(fact), `V4.1: manca ${fact}`);
});

test("Kimi K3.1 Pro: i conteggi sono del backbone K3 e l'indexer DSA aggiunto da K3.1 si vede a parte", () => {
  const addOn = addOnTotal(KIMI);
  assert.equal(addOn.layers, 24); // i layer Gated MLA (dsa_layers)
  assert.equal(addOn.params, 24 * 7438592); // llm_specs.json → dsa_indexer_per_layer_k31
  assert.equal(addOnTotal(V41), null);
  assert.match(KIMI.headline, /backbone K3/);
  assert.match(KIMI.params.official, /K3, Table 1/);
});

test("blocchi «× N»: il layer speciale dice i suoi parametri (Engram, FFN densa)", () => {
  const flagOf = (m, kind, flag) => buildStrip(m).blocks.find((b) => b.kind === kind && b.flags.some((f) => f.id === flag)).flags.find((f) => f.id === flag);
  // V4.1: layer 1 = SWA + proiezioni Engram (157,3M in più dei layer 0); layer 14 = Full + Engram
  assert.equal(flagOf(V41, "S", "engram").note, formatCount(layerParams(V41, layerByNumber(V41, 1)).total));
  assert.equal(flagOf(V41, "F", "engram").note, formatCount(layerParams(V41, layerByNumber(V41, 14)).total));
  assert.equal(layerParams(V41, layerByNumber(V41, 1)).total - layerParams(V41, layerByNumber(V41, 0)).total, V41.params.flagParams.engram);
  // Kimi: il layer 1 ha la FFN densa al posto del MoE
  assert.equal(flagOf(KIMI, "K", "dense").note, formatCount(KIMI.params.attention.K + KIMI.params.dense.params));
  // i target di DSpark hanno gli stessi parametri dei loro vicini: nessuna nota
  assert.equal(flagOf(V41, "U", "dsparkTarget").note, null);
  assert.equal(layerList(V41).length, 40);
});

test("fonti: ogni studio citato nei fatti sulle capacità ha il suo link", () => {
  const urls = (f) => [f.url, ...(f.links ?? []).map((l) => l.url)];
  const cited = { Kaplan: "2001.08361", Hoffmann: "2203.15556", Wei: "2206.07682", Schaeffer: "2304.15004", Olsson: "in-context-learning-and-induction-heads", Power: "2201.02177", Mukherjee: "2505.11711" };
  for (const f of CAPABILITY_FACTS) {
    for (const [author, id] of Object.entries(cited)) {
      if (!`${f.source} ${f.caution}`.includes(author)) continue;
      assert.ok(urls(f).some((u) => u.includes(id)), `${f.id}: manca il link a ${author}`);
    }
  }
  // la cautela della critica alle abilità emergenti c'è, con la sua fonte
  const emergent = CAPABILITY_FACTS.find((f) => f.id === "emergent");
  assert.match(emergent.caution, /metriche/);
  assert.ok(urls(emergent).some((u) => u.includes("2304.15004")));
  for (const m of MODELS) for (const s of m.sources) assert.match(s.url, /^https:\/\//, `${m.id}: ${s.title}`);
});
