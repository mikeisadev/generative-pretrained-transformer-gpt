/**
 * DeepSeek-V4.1-Flash (checkpoint 1017): SOLI DATI della vista "Architettura LLM moderni completa".
 *
 * Da dove vengono i numeri: ricerca verificata dell'8/11/2026 (llm_frontier.md + llm_specs.json). Il
 * config.json NON è stato letto direttamente (huggingface.co non era raggiungibile): è stato
 * ricostruito campo per campo dal codice ufficiale DeepInfer (parser del config e mappa dei nomi dei
 * tensori), dalle classi di config di NVIDIA NeMo AutoModel ("Defaults match the released Flash text
 * configuration"), dagli estratti del config.json ufficiale e dalle ricette vLLM. La ricostruzione
 * torna con i totali ufficiali: 96.085 tensori, 510,3 GB, 552B di backbone.
 *
 * Regole: nessun numero inventato; ciò che la ricerca segna come NON verificato sta in `unverified`
 * e l'interfaccia lo marca "da verificare" con la spiegazione; dove un dato non è pubblico lo si dice.
 * Il formato dei campi è descritto in full/models/index.js; tests/modern-llm-full.test.mjs controlla
 * che i conti tornino con quelli della ricerca.
 */
export const DEEPSEEK_V41_FLASH = {
  id: "deepseek-v4.1-flash",
  name: "DeepSeek-V4.1-Flash",
  short: "V4.1-Flash",
  checkpoint: "1017",
  org: "DeepSeek-AI",
  role: "Modello principale",
  released: "lancio 9-10/09/2026 (checkpoint 0909), refresh DeepSeek-V4.1-Flash-1017 del 17/10/2026",
  license: "MIT (pesi e codice); motore DeepInfer Apache-2.0",
  repo: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
  repoUrl: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017",
  headline: "552B di backbone + 196B di memoria Engram · 8B attivi per token di prompt, 16B per token generato · contesto 1M · testo e immagini",
  why: "È il modello chiesto: il più recente open-weight DeepSeek. Ha una config pubblica, un report tecnico dedicato (arXiv 2609.19969) e un motore di inferenza ufficiale (DeepInfer) dal cui codice l'architettura è stata ricostruita fino all'ultimo tensore.",

  /** Campi del config usati dalla vista (nomi brevi; tra parentesi il nome nel config.json). */
  config: {
    vocab: 129280, // vocab_size
    hidden: 5120, // hidden_size
    layers: 40, // num_hidden_layers
    heads: 64, // num_attention_heads
    kvHeads: 1, // num_key_value_heads
    headDim: 512, // head_dim
    ropeDim: 64, // qk_rope_head_dim
    qLora: 1280, // q_lora_rank
    oLora: 1024, // o_lora_rank
    oGroups: 8, // o_groups
    window: 128, // sliding_window
    experts: 384, // n_routed_experts
    sharedExperts: 1, // n_shared_experts
    topK: 6, // num_experts_per_tok
    expertHidden: 2304, // moe_intermediate_size
    swigluLimit: 10, // swiglu_limit
    routedScaling: 1.5, // routed_scaling_factor
    scoring: "sqrtsoftplus", // scoring_func (topk_method noaux_tc, norm_topk_prob true)
    maxPositions: 1048576, // max_position_embeddings
    ropeTheta: 10000, // rope_theta (layer solo-finestra e DSpark)
    compressRopeTheta: 160000, // compress_rope_theta (layer compressi e indexer)
    yarn: { factor: 16, original: 65536, betaFast: 32, betaSlow: 1 }, // rope_scaling
    indexHeads: 32, // index_n_heads
    indexHeadDim: 128, // index_head_dim
    indexTopK: 512, // index_topk
    candidateBlocks: 2048, // candidate_topk_blocks
    candidateBlockSize: 8, // candidate_block_size
    hcMult: 4, // hc_mult
    sinkhornIters: 20, // hc_sinkhorn_iters
    compressRatios: [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    kvSourceLayers: [2, 8, 14, 20], // kv_source_layer_ids
    indexSourceLayers: [2, 8, 14, 20, 24, 28, 32, 36], // index_source_layer_ids
    engram: { layers: [1, 14], maxNgram: 4, heads: 8, headDim: 256, rows: [384006168, 384016682], compressedVocab: 99092 },
    dspark: { layers: 3, blockSize: 5, targets: [37, 38, 39], markovRank: 256, experts: 128, topK: 3 },
    vision: { layers: 32, hidden: 1024, heads: 16, mlp: 2816, patch: 14, downsample: 3, maxImageTokens: 1024 },
  },

  /**
   * Schema dei layer (0-based): un carattere per layer, il tipo di attention (vedi `kinds`).
   * Il test lo ricava di nuovo da compressRatios, kvSourceLayers e indexSourceLayers.
   */
  layers: {
    base: 0,
    kinds: "SSFRRRRRFRRRRRFRRRRRDUUUXUUUXUUUXUUUXUUU",
    /** Il prefill esegue solo questo segmento (encoder-decoder causale): 8B attivi invece di 16B. */
    prefillSegment: "encoder",
    flags: {
      engram: { layers: [1, 14], label: "Engram", color: "#fb923c" },
      dsparkTarget: { layers: [37, 38, 39], label: "→ DSpark", color: "#f472b6" },
    },
    segments: [
      { id: "encoder", label: "ENCODER causale", sub: "layer 0-19 · il prefill si ferma qui", from: 0, to: 19, color: "#38bdf8" },
      { id: "decoder", label: "DECODER", sub: "layer 20-39 · KV dall'uscita dell'encoder (H20)", from: 20, to: 39, color: "#4ade80" },
    ],
  },

  /** Tipi di layer: etichetta breve (blocchi), nome, colore e cosa fa l'attention. */
  kinds: {
    S: { label: "SWA", name: "Solo finestra (SWA)", color: "#2dd4bf", text: "Finestra di 128 token + attention sink: nessun KV globale." },
    F: { label: "Full", name: "CSA2 Full (m = 2)", color: "#38bdf8", text: "Finestra 128 + KV globale compresso (2 token → 1 voce) + indexer top-512." },
    R: { label: "Reuse", name: "CSA2 Reuse", color: "#94a3b8", text: "Riusa KV globale e top-512 dell'ultimo layer Full; calcola solo la sua finestra." },
    D: { label: "Full ← H20", name: "Full del decoder (m = 1)", color: "#22d3ee", text: "KV globale proiettato dall'uscita dell'encoder (H20); indexer a due livelli." },
    U: { label: "Reuse", name: "Reuse del decoder", color: "#94a3b8", text: "Riusa il KV del 20 e il top-512 del 20 o dell'ultimo Reindex; calcola solo la sua finestra." },
    X: { label: "Reindex", name: "CSA2 Reindex", color: "#a5b4fc", text: "Riusa il KV globale del layer 20 e ricalcola il top-512 dentro i candidati." },
  },

  /**
   * Parametri (conteggi esatti della ricerca, verify_scripts/dsv41_enumerate.py).
   * Per layer: attention del suo tipo + MoE + mHC + norme (+ proiezioni Engram nei layer Engram).
   */
  params: {
    embedding: 661913600, // embed.weight [129280, 5120]
    head: 661913600, // head.weight (non legata all'embedding)
    finalNorm: 5120,
    extra: 15360, // image_start, image_end, image_newline [5120]
    attention: { S: 126617408, F: 137333184, R: 126617408, D: 134711744, U: 126617408, X: 132024128 },
    moe: { routed: 384, expert: 35389440, active: 6, shared: 1, sharedExpert: 35389440, router: 1966848, latent: 0 },
    residual: 983094, // mHC: hc_{attn,ffn}_{fn,base,scale} in FP32
    norms: 10240, // attn_norm + ffn_norm
    flagParams: { engram: 157327360 }, // engram.wkv [25600, 6144] + q_weight + k_weight
    outside: [
      { id: "engram-tables", label: "Tabelle Engram", params: 196613849600, note: "2 tabelle (layer 1 e 14): 384.006.168 e 384.016.682 righe × 256; lette per indirizzo, fuori dal backbone" },
      { id: "dspark", label: "DSpark (3 blocchi draft)", params: 14225362530, note: "3 blocchi SWA + MoE da 128 esperti top-3, testa di Markov e di confidenza" },
      { id: "vision", label: "Visione + aligner", params: 485253120, note: "ViT da 32 blocchi e MLP a 2 strati" },
    ],
    official: "552B di backbone + 196B di Engram (paper, Table 1); 763B nel checkpoint; 16B attivi in decode, 8B in prefill",
  },

  /** Ingresso e uscita del livello Blocchi (da sinistra a destra). */
  input: [
    { id: "tokenizer", title: "Tokenizer", sub: "BPE · 129.280 voci", text: "Byte-level BPE: 128.000 voci + token speciali (<think>, DSML, immagine).", params: 0, color: "#fbbf24", detail: ["<｜begin▁of▁sentence｜> = 0 · <｜end▁of▁sentence｜> = 1", "<think> … </think> · <｜User｜> · <｜Assistant｜>", "｜DSML｜ · <dsml: … </dsml: (strumenti)", "<｜image｜> = 129.264 · <｜ref｜> <｜box｜>", "cifre a gruppi di 3 · CJK separato"] },
    { id: "embedding", title: "Embedding", sub: "[129.280 × 5.120]", text: "Una riga da 5.120 numeri per ogni token del vocabolario (BF16).", params: 661913600, color: "#fbbf24", detail: ["embed.weight [129280, 5120] · BF16", "661,9 milioni di parametri", "non legata alla head (tie = false)", "+ image_start / image_end / image_newline"] },
    { id: "residual", title: "mHC", sub: "1 → 4 flussi", text: "Il residuo diventa 4 flussi da 5.120 che i layer mescolano.", params: 0, color: "#38bdf8", narrow: true, detail: ["4 flussi × 5.120", "mescolati con matrici doppiamente stocastiche", "Sinkhorn: 20 iterazioni"] },
  ],
  output: [
    { id: "norm", title: "Norm", sub: "RMSNorm", text: "Normalizzazione finale.", params: 5120, color: "#e2e8f0", narrow: true, detail: ["norm.weight [5120]", "BF16"] },
    { id: "head", title: "Head", sub: "[129.280 × 5.120]", text: "Un punteggio (logit) per ognuna delle 129.280 voci.", params: 661913600, color: "#fbbf24", detail: ["head.weight [129280, 5120] · BF16", "661,9 milioni di parametri", "serve solo per i token da generare"] },
    { id: "softmax", title: "Softmax", sub: "→ prossimo token", text: "Probabilità di ogni token; se ne sceglie uno.", params: 0, color: "#4ade80", narrow: true, detail: ["129.280 probabilità", "che sommano a 1"] },
  ],
  /** Rami fuori dalla linea principale: visione sopra l'embedding, draft multi-token sotto l'uscita. */
  branches: [
    { id: "vision", title: "Visione (ViT)", sub: "32 blocchi · patch 14", text: "Immagine → ViT → 3×3 → MLP: fino a 1.024 token immagine in 5.120.", params: 485253120, attach: "embedding", side: "top", color: "#fda4af", detail: ["32 blocchi · dim 1.024 · 16 head · SwiGLU 2.816", "RoPE 2D · patch 14", "downsample 3 × 3 → MLP 9.216 → 5.120 → 5.120", "al massimo 1.024 token per immagine"] },
    { id: "dspark", title: "DSpark", sub: "3 blocchi draft · 5 token", text: "Feature dei layer 37-39 → 3 blocchi (SWA + MoE 128e top-3) → 5 token proposti, poi verificati.", params: 14225362530, attach: "last", side: "bottom", color: "#f472b6", detail: ["main_proj 15.360 → 5.120 (layer 37, 38, 39)", "3 blocchi: finestra 128 + MoE 128e top-3", "testa di Markov (rango 256) + confidence head", "5 token per giro · 14,2B parametri"] },
  ],
  /** Riga delle cache sotto i layer: chi le scrive e quanto pesano. */
  caches: [
    { id: "kv", title: "KV globale compressa (FP4)", sub: "890 B per token · sempre in HBM", text: "Scritta dai layer Full 2, 8, 14 (2 token → 1 voce) e 20; la leggono gli altri layer compressi (Reuse e Reindex). I layer 0-1 e DSpark hanno solo la finestra.", layers: [2, 8, 14, 20], color: "#22d3ee", detail: ["voce FP4: 512 × 4 bit + 32 scale = 288 B", "chiave dell'indexer: 64 + 4 = 68 B", "2,5 voci per token × 356 B = 890 B"] },
    { id: "swa", title: "Finestre SWA", sub: "128 token × 43 layer · ≈2,9 MB per sequenza", text: "Ogni layer tiene gli ultimi 128 token (FP8): non cresce con il contesto.", layers: "all", color: "#2dd4bf", detail: ["528 B per token per layer (512 FP8 + 16 di scale)", "128 token × 40 layer ≈ 2,7 MB (≈ 2,9 MB con DSpark)"] },
    { id: "engram", title: "Tabelle Engram", sub: "196,6B valori · 183 GiB FP8", text: "Indirizzo deterministico: possono stare nella DRAM dell'host ed essere prefetchate.", layers: [1, 14], color: "#fb923c", detail: ["384.006.168 righe (layer 1), 384.016.682 (layer 14)", "256 numeri per riga · FP8 con scala ogni 32"] },
  ],

  /** Livello Panoramica: dalla raccolta dei dati al token generato. */
  pipeline: [
    { id: "data", title: "Dati", sub: "45T token multimodali", text: "Pre-training da zero su un corpus di testo e immagini di 45 trilioni di token (dato dichiarato da DeepSeek)." },
    { id: "pretrain", title: "Pre-training", sub: "prossimo token", text: "Da zero, prevedendo il token successivo. Ottimizzatore di V4.1 non dichiarato (V4: Muon per la maggior parte dei moduli)." },
    { id: "context", title: "Contesto lungo", sub: "YaRN ×16: 65.536 → 1M", text: "Le posizioni dei layer compressi sono estese con YaRN (fattore 16 su 65.536): 1.048.576 token.", unverified: ["sparse-64k", "context-34t"] },
    { id: "post", title: "Post-training", sub: "SFT → RL → OPD", text: "Specialisti con RL (GRPO in V4), poi distillazione on-policy in un solo modello. Dati agentici sintetizzati.", steps: ["SFT", "RL", "OPD"] },
    { id: "weights", title: "Pesi", sub: "510,3 GB · 96.085 tensori", text: "Checkpoint safetensors: esperti MXFP4, denso FP8 a blocchi 32×32, embedding/head/router BF16, mHC/sink/bias FP32. La famiglia V4 usa il QAT in FP4 (esperti e Q/K dell'indexer): nei checkpoint V4 su Hugging Face i modelli base sono tutti in FP8, gli instruct hanno gli esperti in FP4.", unverified: ["shards"] },
    { id: "deploy", title: "Deployment", sub: "Encoder · Prefill · Decode", text: "Pool separati per visione, prefill e decode (nel paper; non inclusi nella release open di DeepInfer). MoE in expert parallel con EPLB." },
    { id: "infer", title: "Inferenza", sub: "8B prefill · 16B decode", text: "Il prefill esegue solo l'encoder, il decode tutto il modello; DSpark propone 5 token per giro." },
  ],
  /** Sotto la Panoramica: due schede di sintesi (testi in full/content.js, qui i numeri). */
  macroNotes: { tokensPerParam: "≈ 80 token per parametro totale, oltre 2.800 per parametro attivo (Chinchilla: ≈ 20)" },

  /** Byte del checkpoint per precisione (GB, ricerca §4.13). */
  precision: [
    { id: "mxfp4", dtype: "MXFP4", what: "esperti instradati del backbone", gb: 288.8 },
    { id: "mxfp4-mtp", dtype: "MXFP4", what: "esperti DSpark", gb: 7.2 },
    { id: "fp8-engram", dtype: "FP8", what: "tabelle Engram", gb: 196.6 },
    { id: "scale-engram", dtype: "E8M0", what: "scale delle tabelle Engram", gb: 6.1 },
    { id: "fp8-dense", dtype: "FP8", what: "denso: attention, esperti condivisi, proiezioni", gb: 7.4 },
    { id: "bf16", dtype: "BF16", what: "embedding, head, router, compressori, visione, norme", gb: 4.0 },
    { id: "fp32", dtype: "FP32", what: "mHC, sink, bias del router", gb: 0.2 },
  ],
  checkpoint: { format: "safetensors", gb: 510.3, tensors: 96085, shards: "48", dtypes: "BF16, F32, F8_E4M3, F8_E8M0 (scale), I8 (FP4 impacchettato, 2 valori per byte)" },

  /** KV cache per token (byte) e confronti della ricerca (§4.14). */
  kv: {
    perToken: 890,
    entryBytes: 288, // voce FP4 da 512 dim: 512 × 4 bit + 32 scale × 8 bit
    indexKeyBytes: 68, // chiave dell'indexer: 128 dim FP4 + 4 scale UE8M0
    entriesPerToken: 2.5, // 3 layer Full dell'encoder × ½ (m = 2) + 1 layer Full del decoder
    swaBytesPerTokenLayer: 528, // 512 B FP8 + 16 B di scale
    compare: [
      { id: "self", label: "DeepSeek-V4.1-Flash", bytes: 890 },
      { id: "v3", label: "DeepSeek-V3 (MLA, BF16)", bytes: 70272 },
      { id: "llm67b", label: "DeepSeek LLM 67B (GQA, BF16)", bytes: 389120 },
    ],
    ratio: 437,
    stateNote: "più le finestre SWA: ≈ 2,7 MB per sequenza (128 token × 40 layer × 528 B; ≈ 2,9 MB con DSpark), che non crescono con il contesto",
  },
  flops: { decode: "≈ 32,3 GFLOP per token (2 × 16,13B)", prefill: "≈ 15,8 GFLOP per token (2 × 7,89B)", core: "+ ≈ 3,2 GFLOP del nucleo dell'attention sparsa a contesto lungo" },
  memory: "Pesi 510,3 GB; VRAM minima indicata da vLLM 614 GB. Su 8 × H100 da 80 GB serve spostare le tabelle Engram (183 GiB) nella DRAM dell'host.",

  /**
   * Anatomia di un layer (livello Dettaglio), guidata da questi dati. `in`: tipi di layer in cui il
   * pezzo lavora davvero; `ghost`: tipi in cui è riusato da un altro layer (disegnato tratteggiato);
   * altrove non c'è. `when`: c'è solo nei layer con quel flag. Le forme sono quelle dei tensori veri.
   */
  anatomy: {
    residual: { streams: 4, label: "Residuo mHC: 4 flussi × 5.120", pre: "mHC 4 → 1", post: "mHC 1 → 4", preSub: "Sinkhorn 20 it.", tensor: "hc_attn_fn [24, 20480]", dtype: "FP32" },
    norm: { title: "RMSNorm", attnTensor: "attn_norm.weight [5120]", ffnTensor: "ffn_norm.weight [5120]", dtype: "BF16" },
    attention: {
      title: "Attention CSA2 + finestra 128",
      rows: [
        {
          id: "query",
          label: "Query",
          cards: [
            { id: "wq_a", title: "wq_a", sub: "5.120 → 1.280", tensor: "attn.wq_a.weight [1280, 5120]", dtype: "FP8" },
            { id: "q_norm", title: "q_norm", sub: "RMSNorm", tensor: "attn.q_norm.weight [1280]", dtype: "BF16" },
            { id: "wq_b", title: "wq_b", sub: "1.280 → 64 × 512", tensor: "attn.wq_b.weight [32768, 1280]", dtype: "FP8" },
            { id: "rope", title: "RoPE", sub: "64 di 512 dim", tensor: "θ 10.000 · 160.000 + YaRN" },
          ],
        },
        {
          id: "kv",
          label: "KV locale",
          cards: [
            { id: "wkv", title: "wkv", sub: "5.120 → 512", tensor: "attn.wkv.weight [512, 5120]", dtype: "FP8" },
            { id: "kv_norm", title: "kv_norm", sub: "RMSNorm", tensor: "attn.kv_norm.weight [512]", dtype: "BF16" },
            { id: "latent", title: "Latente 512", sub: "448 + 64 RoPE · K = V", tensor: "una sola KV head" },
            { id: "window", title: "Finestra 128", sub: "+ attention sink", tensor: "attn.attn_sink [64]", dtype: "FP32" },
          ],
        },
        {
          id: "global",
          label: "KV globale",
          empty: { S: "nessun KV globale: solo la finestra" },
          cards: [
            { id: "compressor", title: "Compressore", sub: "wkv + wgate · 2 → 1", tensor: "attn.compressor.wkv/wgate [512, 5120]", dtype: "BF16", in: "FD", ghost: "RUX", alt: { D: { sub: "da H20 · m = 1", tensor: "attn.compressor.wkv [512, 5120]" } } },
            { id: "entry", title: "Voce FP4", sub: "512 dim → 288 B", tensor: "E2M1 + scala ogni 16", in: "FD", ghost: "RUX" },
            { id: "global-read", title: "Legge il KV", sub: "del layer sorgente", tensor: "riuso tra layer (CSA2)", in: "RUX" },
          ],
        },
        {
          id: "index",
          label: "Indexer",
          empty: { S: "nessun indexer" },
          cards: [
            { id: "idx_q", title: "indexer.wq_b", sub: "1.280 → 32 × 128", tensor: "attn.indexer.wq_b.weight [4096, 1280]", dtype: "FP8", in: "FDX", ghost: "RU" },
            { id: "idx_k", title: "indexer.wk", sub: "512 → 128", tensor: "attn.indexer.wk.weight [128, 512]", dtype: "BF16", in: "FD", ghost: "RUX" },
            { id: "idx_w", title: "weights_proj", sub: "5.120 → 32", tensor: "attn.indexer.weights_proj.weight [32, 5120]", dtype: "BF16", in: "FDX", ghost: "RU" },
            { id: "topk", title: "Top-512", sub: "Σ w·ReLU(q·k)", tensor: "dal layer 20: 2.048 blocchi × 8", in: "FDX", ghost: "RU" },
          ],
        },
      ],
      core: { id: "core", title: "MQA sul latente", sub: "64 query · 1 KV head", tensor: "finestra + top-512" },
      output: [
        { id: "wo_a", title: "wo_a", sub: "8 gruppi · 4.096 → 1.024", tensor: "attn.wo_a.weight [8192, 4096]", dtype: "FP8" },
        { id: "wo_b", title: "wo_b", sub: "8.192 → 5.120", tensor: "attn.wo_b.weight [5120, 8192]", dtype: "FP8" },
      ],
    },
    ffn: {
      title: "MoE: 384 esperti + 1 condiviso",
      router: { id: "router", title: "Router", sub: "sqrtsoftplus + bias → top-6", tensor: "ffn.gate.weight [384, 5120]", dtype: "BF16", tensor2: "ffn.gate.bias / bias_vl [384]", dtype2: "FP32" },
      grid: { cols: 24, label: "384 esperti instradati", dtype: "MXFP4" },
      shared: [{ id: "shared-0", title: "Condiviso", sub: "sempre attivo", tensor: "ffn.shared_experts.w1/w2/w3", dtype: "FP8" }],
      expert: { id: "expert", title: "Un esperto: SwiGLU", sub: "5.120 → 2.304 → 5.120 · clamp 10", tensor: "ffn.experts.{e}.w1/w3 [2304, 5120] · w2 [5120, 2304]", dtype: "MXFP4" },
      sum: { id: "sum", title: "Somma pesata", sub: "pesi normalizzati × 1,5" },
    },
    branch: {
      id: "engram",
      when: "engram",
      label: "Engram (solo layer 1 e 14)",
      cards: [
        { id: "ngram", title: "n-gram 2, 3, 4", sub: "vocabolario compresso 99.092", tensor: "NFKC, minuscole, senza accenti" },
        { id: "hash", title: "8 hash per ordine", sub: "→ 24 righe da 256", tensor: "tabelle con un numero primo di righe" },
        { id: "table", title: "Tabella Engram", sub: "≈ 384M righe × 256", tensor: "engram.embed.weight", dtype: "FP8" },
        { id: "proj", title: "engram.wkv", sub: "6.144 → 25.600", tensor: "engram.wkv.weight [25600, 6144]", dtype: "FP8" },
        { id: "gate", title: "Gate", sub: "→ 4 flussi", tensor: "engram.q_weight / k_weight [4, 5120]", dtype: "BF16 (da verificare)" },
      ],
    },
  },

  /**
   * Dove vive ognuno dei 16 meccanismi della vista d'insieme (id delle regioni di regionData.js).
   * `at`: ancora nella scena (livello:pezzo); `status`: "present" (c'è), "absent" (qui vivrebbe, ma
   * questo modello non lo usa), "study" (non è un pezzo del modello: è uno studio sul modello).
   * `layer`: layer da mostrare nel Dettaglio quando l'ancora è dentro un layer.
   */
  mechanisms: {
    compare: { at: "detail:layer", status: "present", text: "Stessa ossatura del blocco del nostro GPT (norm → attention → norm → FFN, con il residuo), ma ogni pezzo è cambiato: RMSNorm, latente KV condiviso, MoE e un residuo a 4 flussi." },
    moe: { at: "detail:experts", status: "present", text: "384 esperti instradati + 1 condiviso in ogni layer: il router ne accende 6 per token." },
    gqa: { at: "detail:core", status: "present", text: "Oltre GQA: MQA sul latente condiviso. Le 64 head di query leggono una sola KV head da 512, con K = V." },
    rope: { at: "detail:rope", status: "present", text: "RoPE solo su 64 delle 512 dimensioni di ogni head: θ = 10.000 nei layer solo-finestra, 160.000 + YaRN nei layer compressi." },
    bpe: { at: "blocks:tokenizer", status: "present", text: "BPE a byte con 129.280 voci (128.000 + token aggiunti); le cifre si raggruppano a 3 a 3." },
    kv: { at: "blocks:cache-kv", status: "present", text: "KV globale compressa in FP4: 890 B per token, circa 437 volte meno di un modello GQA da 67B." },
    flash: { at: "detail:window", status: "present", text: "Attention sparsa: in ogni layer una finestra di 128 token con sink, più le 512 voci scelte dall'indexer nei layer compressi." },
    spec: { at: "blocks:dspark", status: "present", text: "DSpark: 3 blocchi draft propongono 5 token per giro; il modello li verifica tutti insieme." },
    scaling: { at: "macro:data", status: "present", text: "45T token per 552B di parametri: circa 80 token per parametro (Chinchilla suggeriva circa 20)." },
    training: { at: "macro:post", status: "present", text: "Pre-training → SFT → RL → distillazione on-policy (OPD)." },
    reasoning: { at: "macro:post", status: "present", text: "Token <think> … </think> nel tokenizer e RL negli specialisti (GRPO in V4), poi distillati in un solo modello." },
    interp: { at: "detail:residual", status: "study", layer: 2, text: "Non è un pezzo del modello: è lo studio delle attivazioni che passano qui (i 4 flussi del residuo, le head, gli esperti). Nelle fonti consultate non ci sono studi di interpretabilità su V4.1." },
    agent: { at: "blocks:tokenizer", status: "present", text: "Token DSML (｜DSML｜, <dsml: …>) per le chiamate a strumenti in XML, più i marcatori di tool call e di ricerca." },
    coding: { at: "macro:post", status: "present", text: "Entra dai dati e dagli ambienti: il post-training usa task e ambienti agentici sintetizzati; nelle fonti consultate il fill-in-the-middle per V4.1 non compare." },
    multi: { at: "blocks:vision", status: "present", text: "ViT da 32 blocchi, downsample 3×3 e MLP: fino a 1.024 token immagine accanto al testo." },
    mesh: { at: "blocks:softmax", status: "absent", text: "Qui vivrebbe: una mesh OBJ è solo testo in uscita. Questo modello non dichiara un addestramento per le mesh 3D." },
  },

  /** Nomi veri dei tensori per componente ("Dove sono i pesi"), dalla mappa ufficiale DeepInfer. */
  tensorGroups: [
    {
      id: "attention",
      label: "Attention (ogni layer L)",
      tensors: [
        ["layers.{L}.attn_norm.weight", "[5120]", "BF16"],
        ["layers.{L}.attn.wq_a.weight", "[1280, 5120]", "FP8"],
        ["layers.{L}.attn.q_norm.weight", "[1280]", "BF16"],
        ["layers.{L}.attn.wq_b.weight", "[32768, 1280]", "FP8"],
        ["layers.{L}.attn.wkv.weight", "[512, 5120]", "FP8"],
        ["layers.{L}.attn.kv_norm.weight", "[512]", "BF16"],
        ["layers.{L}.attn.wo_a.weight", "[8192, 4096]", "FP8"],
        ["layers.{L}.attn.wo_b.weight", "[5120, 8192]", "FP8"],
        ["layers.{L}.attn.attn_sink", "[64]", "FP32"],
      ],
      note: "Ogni peso FP8 ha accanto la sua scala .scale (UE8M0, blocchi 32 × 32).",
    },
    {
      id: "kv-source",
      label: "Solo nei layer sorgente del KV (2, 8, 14, 20)",
      tensors: [
        ["layers.{L}.attn.compressor.wkv.weight", "[512, 5120]", "BF16"],
        ["layers.{L}.attn.compressor.wgate.weight", "[512, 5120]", "BF16"],
        ["layers.{L}.attn.compressor.norm.weight", "[512]", "BF16"],
        ["layers.{L}.attn.indexer.wk.weight", "[128, 512]", "BF16"],
        ["layers.{L}.attn.indexer.k_norm.weight", "[128]", "BF16"],
      ],
      note: "wgate solo nei layer 2, 8, 14 (m = 2); il layer 20 proietta da H20 senza gate.",
    },
    {
      id: "index-source",
      label: "Solo nei layer dell'indexer (2, 8, 14, 20, 24, 28, 32, 36)",
      tensors: [
        ["layers.{L}.attn.indexer.wq_b.weight", "[4096, 1280]", "FP8"],
        ["layers.{L}.attn.indexer.weights_proj.weight", "[32, 5120]", "BF16"],
      ],
    },
    {
      id: "moe",
      label: "MoE (ogni layer)",
      tensors: [
        ["layers.{L}.ffn_norm.weight", "[5120]", "BF16"],
        ["layers.{L}.ffn.gate.weight", "[384, 5120]", "BF16"],
        ["layers.{L}.ffn.gate.bias", "[384]", "FP32"],
        ["layers.{L}.ffn.gate.bias_vl", "[384]", "FP32"],
        ["layers.{L}.ffn.shared_experts.w1 / w3.weight", "[2304, 5120]", "FP8"],
        ["layers.{L}.ffn.shared_experts.w2.weight", "[5120, 2304]", "FP8"],
        ["layers.{L}.ffn.experts.{0..383}.w1 / w3.weight", "[2304, 5120]", "MXFP4"],
        ["layers.{L}.ffn.experts.{0..383}.w2.weight", "[5120, 2304]", "MXFP4"],
      ],
      note: "Per layer 2.304 tensori di esperti (384 × 3 pesi + 384 × 3 scale UE8M0, una ogni 32 valori). bias_vl: bilanciamento per i token immagine.",
    },
    {
      id: "mhc",
      label: "mHC (ogni layer)",
      tensors: [
        ["layers.{L}.hc_attn_fn / hc_ffn_fn", "[24, 20480]", "FP32"],
        ["layers.{L}.hc_attn_base / hc_ffn_base", "[24]", "FP32"],
        ["layers.{L}.hc_attn_scale / hc_ffn_scale", "[3]", "FP32"],
      ],
    },
    {
      id: "engram",
      label: "Engram (layer 1 e 14)",
      tensors: [
        ["layers.{1,14}.engram.embed.weight", "[384006168, 256]", "FP8"],
        ["layers.{1,14}.engram.wkv.weight", "[25600, 6144]", "FP8"],
        ["layers.{1,14}.engram.q_weight / k_weight", "[4, 5120]", "BF16 (da verificare)"],
      ],
      note: "Il layer 14 ha 384.016.682 righe.",
    },
    {
      id: "global",
      label: "Globali",
      tensors: [
        ["embed.weight", "[129280, 5120]", "BF16"],
        ["head.weight", "[129280, 5120]", "BF16"],
        ["norm.weight", "[5120]", "BF16"],
        ["image_start / image_end / image_newline", "[5120]", "BF16"],
      ],
    },
    {
      id: "mtp",
      label: "DSpark (mtp.0, mtp.1, mtp.2)",
      tensors: [
        ["mtp.{i}.<stessi nomi del blocco>", "esperti 0..127", "MXFP4"],
        ["mtp.0.main_proj.weight", "[5120, 15360]", "FP8"],
        ["mtp.2.markov_head.embed.weight / head.weight", "[129280, 256]", "BF16"],
        ["mtp.2.confidence_head.proj.weight", "[1, 5376]", "BF16"],
      ],
    },
    {
      id: "vision",
      label: "Visione",
      tensors: [
        ["vision.blocks.{0..31}.attn.wqkv.weight", "[3072, 1024]", "BF16"],
        ["vision.blocks.{b}.mlp.w1.weight / w2.weight", "[5632, 1024] / [1024, 2816]", "BF16"],
        ["aligner.w1.weight / aligner.w2.weight", "[5120, 9216] / [5120, 5120]", "BF16"],
      ],
    },
  ],

  /** Ciò che la ricerca segna come NON verificato (marcato "da verificare" nell'interfaccia). */
  unverified: [
    { id: "shards", value: "48 shard", why: "Il numero di shard (model-00001-of-00048 …) non compare né nella ricetta vLLM né nel codice DeepInfer, e l'indice model.safetensors.index.json non era raggiungibile. Verificati invece 96.085 tensori e 510,3 GB." },
    { id: "sparse-64k", value: "sparse attention addestrata a 64K", why: "Nessuna fonte raggiungibile lo riporta." },
    { id: "context-34t", value: "estensione a 1M a 34T token", why: "Nessuna fonte raggiungibile lo riporta; le sintesi del report V4 danno 32T token per V4-Flash e 33T per V4-Pro, senza una fase separata a 34T." },
    { id: "kv-qat", value: "QAT del KV in FP4", why: "Il KV principale in FP4 è confermato dal codice (cache_engine.py di DeepInfer); che sia ottenuto con QAT nel post-training non è confermato dalle fonti leggibili." },
    { id: "clamp", value: "forma del clamp SwiGLU", why: "Il limite 10 (swiglu_limit) è verificato; la forma esatta (simmetrica o solo sul ramo gate) non si legge nel codice Python pubblicato." },
    { id: "paper-pages", value: "citazioni con il numero di pagina del report", why: "Table 1 p. 24, eq. 1, §3.2.x e Fig. 1(b) sono confermati solo tramite una fonte secondaria (Arch 06) e le sintesi dell'abstract: arXiv non era raggiungibile." },
  ],

  sources: [
    { title: "DeepInfer, motore di inferenza ufficiale (solo V4.1-Flash)", url: "https://github.com/deepseek-ai/DeepInfer", note: "Parser del config.json, nomi e forme dei tensori, precisioni; CSA2, YOCO, Engram, mHC, DSpark." },
    { title: "DeepInfer: hugging_face/config.py e names.py", url: "https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/deepinfer/hugging_face/names.py", note: "Mappa dei nomi dei tensori HF; il parser rifiuta valori diversi da quelli del rilascio." },
    { title: "DeepInfer: worker/cache_engine.py", url: "https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/deepinfer/worker/cache_engine.py", note: "288 B per voce FP4 (scala ogni 16), 68 B per chiave dell'indexer, 528 B per token SWA." },
    { title: "NVIDIA NeMo AutoModel: deepseek_v41/config.py", url: "https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/nemo_automodel/components/models/deepseek_v41/config.py", note: "«Defaults match the released Flash text configuration»: testo e visione." },
    { title: "Ricetta vLLM DeepSeek-V4.1-Flash", url: "https://raw.githubusercontent.com/vllm-project/vllm/main/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml", note: "552B, 8-16B attivi, 96.085 tensori, 510,3 GB, Engram 183 GiB FP8, VRAM minima 614 GB." },
    { title: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression", url: "https://arxiv.org/abs/2609.19969", note: "Report tecnico: encoder-decoder causale, CSA2, 890 B per token, 45T token (letto tramite abstract ed estratti)." },
    { title: "Model card Hugging Face", url: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017", note: "SFT → RL → OPD, 384 + 1 esperti (estratti via motore di ricerca)." },
    { title: "deepseek-recipe: tokenizer V4.1", url: "https://github.com/deepseek-ai/deepseek-recipe", note: "tokenizer.json, token speciali, DSML." },
    { title: "Arch 06: Asymmetric Causal Encoder-Decoder (fonte secondaria)", url: "https://github.com/BrendanJamesLynskey/Arch_06_Asymmetric_Causal_Encoder_Decoder", note: "Cita il report pagina per pagina: eq. 1, replay di 128 token, Table 1." },
    { title: "mHC: Manifold-Constrained Hyper-Connections", url: "https://arxiv.org/abs/2512.24880", note: "Residuo a più flussi con matrici doppiamente stocastiche (Sinkhorn-Knopp)." },
    { title: "Engram: Conditional Memory via Scalable Lookup", url: "https://arxiv.org/abs/2601.07372", note: "Memoria di n-gram con accesso per indirizzo." },
    { title: "YOCO: You Only Cache Once", url: "https://arxiv.org/abs/2405.05254", note: "L'idea da cui nasce l'encoder-decoder causale." },
    { title: "HF blog DeepSeek-V4", url: "https://github.com/huggingface/blog/blob/main/deepseekv4.md", note: "Famiglia V4: CSA/HCA, mHC, esperti FP4 (base in FP8, instruct con esperti FP4), 1,6T/49B e 284B/13B." },
    { title: "DeepSeek-V3 Technical Report", url: "https://arxiv.org/abs/2412.19437", note: "Il confronto della KV cache: MLA con 61 layer × (512 + 64) valori in BF16 = 70.272 B per token." },
    { title: "Sistema di inferenza di DeepSeek-V3/R1 (open-infra-index, giorno 6)", url: "https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md", note: "Prefill e decode disaggregati, EP32 / EP144, esperti ridondanti." },
  ],
};
