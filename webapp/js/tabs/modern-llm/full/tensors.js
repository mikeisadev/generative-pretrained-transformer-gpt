/**
 * Ricostruzione PURA del checkpoint di DeepSeek-V4.1-Flash: tutti i tensori, raggruppati per
 * componente, con forme, tipi e byte. Segue i moduli del motore ufficiale DeepInfer (mla_attn.py,
 * moe.py, hc_blocks.py, heads.py, engram_parameters.py, vit.py, mlp_projector.py, names.py) e i
 * campi del config del modello: è la stessa enumerazione della verifica della ricerca, rifatta in
 * JavaScript. Il risultato si confronta con i numeri ufficiali (ricetta vLLM): 96.085 tensori e
 * 510,3 GB. Nessun tensore viene creato davvero: si contano i gruppi (un esperto × 384 × 40 …).
 *
 *   const c = enumerateCheckpoint(DEEPSEEK_V41_FLASH);
 *   c.tensors   // 96.085
 *   c.bytes     // 510.286.023.000
 *   c.groups    // [{id, label, tensors, bytes, params}] per il pannello "Pesi"
 */

const BYTES = { BF16: 2, F32: 4, F8_E4M3: 1, F8_E8M0: 1, I8_FP4: 1 };
/** Granularità delle scale FP8 (UE8M0) dei pesi densi: blocchi 32 × 32. */
const FP8_BLOCK = 32;

const GROUP_LABELS = {
  routed: "Esperti instradati (MXFP4)",
  attention: "Attention",
  shared: "Esperti condivisi",
  router: "Router (+ bias)",
  mhc: "mHC",
  norms: "Norme",
  "engram-tables": "Tabelle Engram",
  "engram-proj": "Proiezioni Engram",
  embedding: "Embedding",
  head: "Head",
  extra: "Marcatori immagine",
  dspark: "DSpark (mtp.0-2)",
  vision: "Visione + aligner",
};

/**
 * @param {object} model un modello con `config` nel formato di full/models/deepseekV41Flash.js
 * @returns {{tensors: number, bytes: number, params: number, groups: {id, label, tensors, bytes, params}[]}}
 */
export function enumerateCheckpoint(model) {
  const c = model.config;
  const H = c.hidden;
  const groups = new Map();
  /** Aggiunge `count` tensori uguali: forma memorizzata, tipo, parametri logici per tensore. */
  const add = (group, shape, dtype, count = 1, logical = null) => {
    const stored = shape.reduce((a, b) => a * b, 1);
    const g = groups.get(group) ?? { id: group, label: GROUP_LABELS[group], tensors: 0, bytes: 0, params: 0 };
    g.tensors += count;
    g.bytes += count * stored * BYTES[dtype];
    g.params += count * (logical ?? (dtype === "F8_E8M0" ? 0 : stored));
    groups.set(group, g);
  };
  /** Peso FP8 con la sua scala a blocchi 32 × 32. */
  const fp8 = (group, out, inp, count = 1) => {
    add(group, [out, inp], "F8_E4M3", count);
    add(group, [Math.ceil(out / FP8_BLOCK), Math.ceil(inp / FP8_BLOCK)], "F8_E8M0", count);
  };

  /** Un blocco (layer del backbone o blocco DSpark): stessa struttura, esperti diversi. */
  const block = (where, { ratio, experts, kvSource, indexSource, indexKeySource, engramRows }) => {
    const g = (name) => (where === "dspark" ? "dspark" : name);
    add(g("norms"), [H], "BF16", 2); // attn_norm, ffn_norm
    const channels = (2 + c.hcMult) * c.hcMult;
    for (let k = 0; k < 2; k++) {
      add(g("mhc"), [channels], "F32");
      add(g("mhc"), [channels, c.hcMult * H], "F32");
      add(g("mhc"), [3], "F32");
    }
    add(g("attention"), [c.heads], "F32"); // attn_sink
    fp8(g("attention"), c.qLora, H); // wq_a
    add(g("attention"), [c.qLora], "BF16"); // q_norm
    fp8(g("attention"), c.heads * c.headDim, c.qLora); // wq_b
    fp8(g("attention"), c.headDim, H); // wkv
    add(g("attention"), [c.headDim], "BF16"); // kv_norm
    fp8(g("attention"), c.oGroups * c.oLora, c.headDim * (c.heads / c.oGroups)); // wo_a
    fp8(g("attention"), H, c.oGroups * c.oLora); // wo_b
    if (ratio !== 0) {
      if (kvSource) {
        add(g("attention"), [c.headDim, H], "BF16"); // compressor.wkv
        if (ratio !== 1) add(g("attention"), [c.headDim, H], "BF16"); // compressor.wgate
        add(g("attention"), [c.headDim], "BF16"); // compressor.norm
      }
      if (indexSource) {
        fp8(g("attention"), c.indexHeads * c.indexHeadDim, c.qLora); // indexer.wq_b
        add(g("attention"), [c.indexHeads, H], "BF16"); // indexer.weights_proj
        if (indexKeySource) {
          add(g("attention"), [c.indexHeadDim, c.headDim], "BF16"); // indexer.wk
          add(g("attention"), [c.indexHeadDim], "BF16"); // indexer.k_norm
        }
      }
    }
    add(g("router"), [experts, H], "BF16"); // ffn.gate.weight
    add(g("router"), [experts], "F32", 2); // ffn.gate.bias, bias_vl
    const I = c.expertHidden;
    fp8(g("shared"), I * c.sharedExperts, H, 2); // shared w1, w3
    fp8(g("shared"), H, I * c.sharedExperts); // shared w2
    // esperti MXFP4: 2 valori per byte (I8) + una scala UE8M0 ogni 32 valori
    for (const [out, inp] of [[I, H], [I, H], [H, I]]) {
      add(g("routed"), [out, inp / 2], "I8_FP4", experts, out * inp);
      add(g("routed"), [out, inp / 32], "F8_E8M0", experts);
    }
    if (engramRows) {
      const e = c.engram;
      add("engram-tables", [engramRows, e.headDim], "F8_E4M3");
      add("engram-tables", [engramRows, e.headDim / FP8_BLOCK], "F8_E8M0");
      fp8("engram-proj", H * (c.hcMult + 1), (e.maxNgram - 1) * e.heads * e.headDim); // engram.wkv
      add("engram-proj", [c.hcMult, H], "BF16", 2); // q_weight, k_weight
    }
  };

  // globali
  add("embedding", [c.vocab, H], "BF16");
  add("extra", [H], "BF16", 3); // image_start, image_end, image_newline
  add("head", [c.vocab, H], "BF16");
  add("norms", [H], "BF16"); // norma finale
  // backbone
  const engramRows = new Map(c.engram.layers.map((l, i) => [l, c.engram.rows[i]]));
  for (let l = 0; l < c.layers; l++) {
    block("backbone", {
      ratio: c.compressRatios[l],
      experts: c.experts,
      kvSource: c.kvSourceLayers.includes(l),
      indexSource: c.indexSourceLayers.includes(l),
      indexKeySource: c.kvSourceLayers.includes(l), // index_k_source_layer_ids = kv_source (default)
      engramRows: engramRows.get(l),
    });
  }
  // DSpark: 3 blocchi solo-finestra con 128 esperti, più proiezione d'ingresso e teste
  const d = c.dspark;
  for (let i = 0; i < d.layers; i++) {
    block("dspark", { ratio: 0, experts: d.experts });
    if (i === 0) {
      fp8("dspark", H, d.targets.length * H); // mtp.0.main_proj
      add("dspark", [H], "BF16"); // main_norm
    }
    if (i === d.layers - 1) {
      add("dspark", [H], "BF16"); // mtp.2.norm
      add("dspark", [c.vocab, d.markovRank], "BF16", 2); // markov_head.embed, head
      add("dspark", [1, H + d.markovRank], "BF16"); // confidence_head.proj
    }
  }
  // visione (vit.py) e aligner (mlp_projector.py)
  const v = c.vision;
  add("vision", [v.hidden, 3 * v.patch * v.patch], "BF16");
  add("vision", [v.hidden], "BF16");
  for (let b = 0; b < v.layers; b++) {
    add("vision", [v.hidden], "BF16", 2); // norm1, norm2
    add("vision", [3 * v.hidden, v.hidden], "BF16"); // wqkv
    add("vision", [3 * v.hidden], "BF16");
    add("vision", [v.hidden, v.hidden], "BF16"); // wo
    add("vision", [v.hidden], "BF16");
    add("vision", [2 * v.mlp, v.hidden], "BF16"); // mlp.w1
    add("vision", [v.hidden, v.mlp], "BF16"); // mlp.w2
  }
  add("vision", [v.hidden], "BF16"); // vision.norm
  const merged = v.hidden * v.downsample * v.downsample;
  add("vision", [H, merged], "BF16");
  add("vision", [H], "BF16");
  add("vision", [H, H], "BF16");
  add("vision", [H], "BF16");

  const list = [...groups.values()].sort((a, b) => b.bytes - a.bytes);
  return {
    tensors: list.reduce((a, g) => a + g.tensors, 0),
    bytes: list.reduce((a, g) => a + g.bytes, 0),
    params: list.reduce((a, g) => a + g.params, 0),
    groups: list,
  };
}
