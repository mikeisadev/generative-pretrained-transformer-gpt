/**
 * gpt-oss-120b (OpenAI, 05/08/2025): SOLI DATI, modello alternativo della vista completa.
 *
 * Fonte: ricerca verificata dell'8/11/2026. Valori dal ModelConfig del codice di riferimento OpenAI
 * (github.com/openai/gpt-oss, gpt_oss/torch/model.py) e dai default di transformers GptOssConfig;
 * config.json non scaricato. Totale e attivi ricalcolati: 116,83B / 5,13B, identici alla Table 1
 * della model card. Non è "frontiera 2026", ma è l'ultimo open di OpenAI e il miglior contrasto
 * didattico: GQA classica, finestra alternata, MoE senza esperti condivisi.
 * Formato dei campi: full/models/index.js.
 */
export const GPT_OSS_120B = {
  id: "gpt-oss-120b",
  name: "gpt-oss-120b",
  short: "gpt-oss-120b",
  checkpoint: "",
  org: "OpenAI",
  role: "Alternativo: la scelta classica",
  released: "05/08/2025",
  license: "Apache 2.0 (+ usage policy)",
  repo: "openai/gpt-oss-120b",
  repoUrl: "https://huggingface.co/openai/gpt-oss-120b",
  headline: "116,83B parametri, 5,13B attivi · 36 layer · GQA 64/8 · 128 esperti, 4 attivi · contesto 131.072 · entra in una GPU da 80 GB",
  why: "È la scelta «classica»: GQA con 8 KV head, layer a finestra di 128 token alternati a layer densi con attention sink, MoE top-4 senza esperti condivisi, bias nelle proiezioni, MXFP4.",

  config: {
    vocab: 201088,
    hidden: 2880,
    layers: 36,
    heads: 64,
    kvHeads: 8,
    headDim: 64,
    experts: 128,
    sharedExperts: 0,
    topK: 4,
    expertHidden: 2880,
    swigluLimit: 7,
    swigluAlpha: 1.702,
    window: 128,
    initialContext: 4096,
    maxPositions: 131072,
    ropeTheta: 150000,
    yarn: { factor: 32, original: 4096 },
  },

  layers: {
    base: 0,
    kinds: "BGBGBGBGBGBGBGBGBGBGBGBGBGBGBGBGBGBG",
    flags: {},
    segments: [{ id: "stack", label: "36 LAYER", sub: "finestra di 128 e densa, alternate", from: 0, to: 35, color: "#38bdf8" }],
  },

  kinds: {
    B: { label: "Finestra 128", name: "GQA a finestra (banded)", color: "#2dd4bf", text: "Ogni token guarda solo gli ultimi 128 (layer pari), più il sink." },
    G: { label: "Densa", name: "GQA densa", color: "#38bdf8", text: "Ogni token guarda tutto il contesto (layer dispari), più il sink." },
  },

  params: {
    embedding: 579133440,
    head: 579133440,
    finalNorm: 2880,
    extra: 0,
    attention: { B: 26553024, G: 26553024 }, // q, k, v, o con bias + sinks + input_layernorm
    moe: { routed: 128, expert: 24891840, active: 4, shared: 0, sharedExpert: 0, router: 368768, latent: 0 },
    residual: 0,
    norms: 2880, // post_attention_layernorm
    flagParams: {},
    outside: [],
    official: "116,83B totali e 5,13B attivi (model card, Table 1)",
  },

  input: [
    { id: "tokenizer", title: "Tokenizer", sub: "o200k_harmony · 201.088", text: "BPE che estende o200k; formato di chat harmony.", params: 0, color: "#fbbf24" },
    { id: "embedding", title: "Embedding", sub: "[201.088 × 2.880]", text: "Una riga da 2.880 numeri per token.", params: 579133440, color: "#fbbf24" },
    { id: "residual", title: "Residuo", sub: "una corsia", text: "Residuo semplice, come nel nostro GPT.", params: 0, color: "#38bdf8", narrow: true },
  ],
  output: [
    { id: "norm", title: "Norm", sub: "RMSNorm", text: "Normalizzazione finale.", params: 2880, color: "#e2e8f0", narrow: true },
    { id: "head", title: "Head", sub: "[201.088 × 2.880]", text: "Unembedding non legato all'embedding.", params: 579133440, color: "#fbbf24" },
    { id: "softmax", title: "Softmax", sub: "→ prossimo token", text: "Probabilità di ogni token.", params: 0, color: "#4ade80", narrow: true },
  ],
  branches: [],
  caches: [
    { id: "kv", title: "KV dei 18 layer densi", sub: "36.864 B per token (BF16)", text: "18 layer × 2 × 8 head × 64 × 2 B.", layers: "G", color: "#22d3ee" },
    { id: "window", title: "Finestre dei 18 layer pari", sub: "128 token · ≈ 4,7 MB per sequenza", text: "Si fermano a 128 token: non crescono con il contesto.", layers: "B", color: "#2dd4bf" },
  ],

  pipeline: [
    { id: "data", title: "Dati", sub: "trilioni di token", text: "Solo testo: STEM, codice e conoscenza generale. Il numero esatto non è dichiarato." },
    { id: "pretrain", title: "Pre-training", sub: "2,1M ore H100", text: "NVIDIA H100 con PyTorch e kernel Triton: 2,1 milioni di ore GPU per la versione 120b." },
    { id: "context", title: "Contesto lungo", sub: "YaRN ×32: 4.096 → 131.072", text: "RoPE con θ = 150.000 estesa con YaRN, fattore 32." },
    { id: "post", title: "Post-training", sub: "SFT + RL", text: "SFT e RL ad alto calcolo (come o4-mini), con reasoning effort low, medium, high.", steps: ["SFT", "RL"] },
    { id: "weights", title: "Pesi", sub: "60,8 GiB", text: "Esperti in MXFP4 (quantizzazione nel post-training), il resto in BF16.", unverified: ["shards"] },
    { id: "deploy", title: "Deployment", sub: "una GPU da 80 GB", text: "Grazie a MXFP4 il modello entra in una sola GPU da 80 GB." },
    { id: "infer", title: "Inferenza", sub: "5,13B attivi", text: "≈ 10,3 GFLOP per token; KV solo nei 18 layer densi." },
  ],
  macroNotes: { tokensPerParam: "token di training non dichiarati («trilioni»): il rapporto non si può calcolare" },

  /** Stime: parametri × bit per valore (MXFP4 = 4,25 bit; BF16 = 16 bit); model card: 60,8 GiB. */
  precision: [
    { id: "mxfp4", dtype: "MXFP4", what: "esperti (stima)", gb: 60.9, estimate: true },
    { id: "bf16", dtype: "BF16", what: "attention, router, embedding, head, norme (stima)", gb: 4.3, estimate: true },
  ],
  checkpoint: { format: "safetensors", gb: 65.3, gib: 60.8, tensors: null, shards: null, dtypes: "esperti MXFP4 (blocks + scales), resto BF16" },

  kv: {
    perToken: 36864,
    compare: [{ id: "self", label: "gpt-oss-120b (18 layer densi, BF16)", bytes: 36864 }],
    stateNote: "più le finestre dei 18 layer pari: ≈ 4,7 MB per sequenza",
  },
  flops: { decode: "≈ 10,3 GFLOP per token (2 × 5,13B)" },
  memory: "Entra in una GPU da 80 GB grazie a MXFP4.",

  anatomy: {
    residual: { streams: 1, label: "Residuo (una corsia)", pre: null, post: null },
    norm: { title: "RMSNorm", attnTensor: "input_layernorm.weight [2880]", ffnTensor: "post_attention_layernorm.weight [2880]" },
    attention: {
      title: "Attention GQA + sink",
      titles: { B: "GQA a finestra di 128 token", G: "GQA densa" },
      rows: [
        {
          id: "query",
          label: "Query",
          cards: [
            { id: "q_proj", title: "q_proj", sub: "2.880 → 64 × 64", tensor: "self_attn.q_proj.{weight,bias} [4096, 2880]" },
            { id: "rope", title: "RoPE", sub: "θ 150.000 · YaRN ×32", tensor: "da 4.096 a 131.072 token" },
          ],
        },
        {
          id: "kv",
          label: "Key e Value",
          cards: [
            { id: "k_proj", title: "k_proj", sub: "2.880 → 8 × 64", tensor: "self_attn.k_proj [512, 2880]" },
            { id: "v_proj", title: "v_proj", sub: "2.880 → 8 × 64", tensor: "self_attn.v_proj [512, 2880]" },
          ],
        },
        {
          id: "mask",
          label: "Chi guarda chi",
          cards: [
            { id: "window", title: "Finestra 128", sub: "layer pari", tensor: "sliding_window 128", in: "B" },
            { id: "dense", title: "Tutto il contesto", sub: "layer dispari", tensor: "causale, senza finestra", in: "G" },
            { id: "sink", title: "Sink", sub: "1 logit per head", tensor: "self_attn.sinks [64]" },
          ],
        },
      ],
      core: { id: "core", title: "GQA", sub: "64 query · 8 K/V head", tensor: "8 query per ogni coppia K/V" },
      output: [{ id: "o_proj", title: "o_proj", sub: "4.096 → 2.880", tensor: "self_attn.o_proj [2880, 4096]" }],
    },
    ffn: {
      title: "MoE: 128 esperti, nessun condiviso",
      router: { id: "router", title: "Router", sub: "softmax sui top-4", tensor: "mlp.router.{weight,bias} [128, 2880]" },
      grid: { cols: 16, label: "128 esperti", dtype: "MXFP4" },
      shared: [],
      expert: { id: "expert", title: "Un esperto: SwiGLU", sub: "2.880 → 2.880 → 2.880 · clamp 7, α 1,702", tensor: "mlp.experts.gate_up_proj · down_proj", dtype: "MXFP4" },
      sum: { id: "sum", title: "Somma pesata", sub: "softmax sui 4 scelti" },
    },
    branch: null,
  },

  mechanisms: {
    compare: { at: "detail:layer", status: "present", layer: 1, text: "Il più vicino al nostro GPT: norm → attention → norm → MoE con un residuo semplice; bias nelle proiezioni di attention e negli esperti." },
    moe: { at: "detail:experts", status: "present", layer: 1, text: "128 esperti, 4 attivi per token, nessun esperto condiviso: softmax sui 4 punteggi scelti." },
    gqa: { at: "detail:core", status: "present", layer: 1, text: "GQA vera e propria: 64 head di query e 8 head K/V da 64 (8 query per ogni coppia K/V)." },
    rope: { at: "detail:rope", status: "present", layer: 1, text: "RoPE con θ = 150.000 ed estensione YaRN ×32 (da 4.096 a 131.072 token)." },
    bpe: { at: "blocks:tokenizer", status: "present", text: "o200k_harmony: BPE da 201.088 voci, formato di chat harmony." },
    kv: { at: "blocks:cache-kv", status: "present", text: "36.864 B per token (i 18 layer densi, BF16); i 18 layer a finestra si fermano a 128 token." },
    flash: { at: "detail:window", status: "present", layer: 0, text: "Finestra di 128 token nei layer pari, attention densa nei dispari; un logit «sink» per head in tutti." },
    spec: { at: "blocks:head", status: "absent", text: "Qui vivrebbe, ma gpt-oss non ha teste di multi-token prediction (0 layer MTP)." },
    scaling: { at: "macro:data", status: "present", text: "Trilioni di token di solo testo (numero esatto non dichiarato) e 2,1 milioni di ore H100." },
    training: { at: "macro:post", status: "present", text: "SFT + RL ad alto calcolo, come o4-mini." },
    reasoning: { at: "macro:post", status: "present", text: "Reasoning effort low / medium / high e il canale «analysis» del formato harmony." },
    interp: { at: "detail:residual", status: "study", layer: 1, text: "Non è un pezzo del modello: è lo studio delle attivazioni che passano nel residuo, nelle head e negli esperti." },
    agent: { at: "blocks:tokenizer", status: "present", text: "Formato harmony con i canali analysis / commentary / final." },
    coding: { at: "macro:data", status: "present", text: "Pre-training su STEM, codice e conoscenza generale." },
    multi: { at: "blocks:embedding", status: "absent", text: "Qui entrerebbero i token immagine, ma gpt-oss è solo testo." },
    mesh: { at: "blocks:softmax", status: "absent", text: "Qui vivrebbe: una mesh OBJ è solo testo in uscita. Questo modello non dichiara un addestramento per le mesh 3D." },
  },

  tensorGroups: [
    {
      id: "layer",
      label: "Ogni layer N (36)",
      tensors: [
        ["model.layers.{N}.input_layernorm.weight", "[2880]", ""],
        ["model.layers.{N}.self_attn.q_proj.{weight,bias}", "[4096, 2880]", ""],
        ["model.layers.{N}.self_attn.k_proj / v_proj.{weight,bias}", "[512, 2880]", ""],
        ["model.layers.{N}.self_attn.o_proj.{weight,bias}", "[2880, 4096]", ""],
        ["model.layers.{N}.self_attn.sinks", "[64]", ""],
        ["model.layers.{N}.post_attention_layernorm.weight", "[2880]", ""],
        ["model.layers.{N}.mlp.router.{weight,bias}", "[128, 2880]", ""],
        ["model.layers.{N}.mlp.experts.gate_up_proj_{blocks,scales,bias}", "[128, 2880, 5760]", "MXFP4"],
        ["model.layers.{N}.mlp.experts.down_proj_{blocks,scales,bias}", "[128, 2880, 2880]", "MXFP4"],
      ],
    },
    { id: "global", label: "Globali", tensors: [["model.embed_tokens.weight", "[201088, 2880]", ""], ["model.norm.weight", "[2880]", ""], ["lm_head.weight", "[201088, 2880]", ""]] },
  ],
  tensorNote: "Nomi dal modeling di transformers e dall'integrazione MXFP4. Tutti i 36 layer sono MoE.",

  unverified: [
    { id: "shards", value: "numero di shard", why: "Non verificato dalla ricerca." },
    { id: "config", value: "config.json", why: "Non scaricato: valori dal ModelConfig del codice di riferimento OpenAI e dai default di transformers, coincidenti tra loro." },
  ],

  sources: [
    { title: "openai/gpt-oss: codice di riferimento", url: "https://github.com/openai/gpt-oss", note: "ModelConfig, sinks, finestra a layer alterni, 117B / 5,1B, Apache 2.0." },
    { title: "gpt-oss-120b & gpt-oss-20b Model Card", url: "https://arxiv.org/abs/2508.10925", note: "Table 1, 2,1 milioni di ore H100, o200k_harmony." },
    { title: "transformers: GptOssConfig", url: "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt_oss/configuration_gpt_oss.py", note: "Default del config." },
  ],
};
