/**
 * Kimi K3.1 Pro (backbone Kimi K3): SOLI DATI, modello alternativo della vista completa.
 *
 * Fonte: ricerca verificata dell'8/11/2026. Il report tecnico completo copre Kimi K3 (PDF su GitHub,
 * Table 1); K3.1 Pro usa lo stesso backbone e aggiunge DSA (sparse attention) ai 24 layer MLA,
 * documentata dal codice vLLM/NeMo. Valori di config = default NeMo KimiK3TextConfig ("Defaults
 * reproduce the released moonshotai/Kimi-K3 checkpoint"); config.json e indice HF non letti
 * direttamente. I conteggi sono quelli del backbone K3 (2,78T / 104,2B, Table 1): l'indexer DSA di
 * K3.1 è mostrato a parte (`params.addOn`).
 * Formato dei campi: full/models/index.js.
 */
export const KIMI_K31_PRO = {
  id: "kimi-k3.1-pro",
  name: "Kimi K3.1 Pro",
  short: "Kimi K3.1 Pro",
  checkpoint: "backbone K3",
  org: "Moonshot AI",
  role: "Alternativo: attention lineare ibrida",
  released: "K3: API 16/07/2026, pesi 26-27/07/2026; K3.1 Pro: API 23/10/2026, pesi tra il 02 e il 04/11/2026 (fonti discordi)",
  license: "Kimi K3 License / Kimi K3.1-Pro License (stile MIT; accordo per Model-as-a-Service oltre 20 M$ di ricavi, attribuzione oltre 100 M MAU o 20 M$/mese)",
  repo: "moonshotai/Kimi-K3.1-Pro",
  repoUrl: "https://huggingface.co/moonshotai/Kimi-K3.1-Pro",
  headline: "2,78T parametri e 104,2B attivi nel backbone K3 (K3.1 aggiunge l'indexer DSA) · 69 layer KDA + 24 Gated MLA · 896 esperti, 16 attivi · contesto 1M · MXFP4 nativo",
  why: "È la scelta «attention lineare ibrida»: 3 layer KDA (memoria a stato fisso) per ogni layer di attention globale, routing LatentMoE con 896 esperti e precisione MXFP4 nativa (QAT dall'SFT in poi). È il modello open più grande.",

  config: {
    vocab: 163840,
    hidden: 7168,
    layers: 93,
    heads: 96,
    denseHidden: 33792, // intermediate_size (FFN densa del layer 1)
    experts: 896,
    sharedExperts: 2,
    topK: 16,
    expertHidden: 3072, // moe_intermediate_size
    latent: 3584, // routed_expert_hidden_size
    qLora: 1536,
    kvLora: 512,
    ropeDim: 64, // presente nel latente ma senza rotazione (NoPE)
    maxPositions: 1048576,
    kdaHeads: 96,
    kdaHeadDim: 128,
    shortConv: 4,
    attnResBlock: 12,
    fullAttnLayers: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
    dsa: { topK: 2048, heads: 32, headDim: 128 },
    vision: { layers: 27, hidden: 1024, heads: 12, mlp: 4096, patch: 14, merge: "2 × 2" },
  },

  layers: {
    base: 1,
    kinds: "KKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMKKKMM",
    flags: {
      dense: { layers: [1], label: "FFN densa", color: "#fbbf24" },
    },
    segments: [{ id: "stack", label: "93 LAYER IBRIDI", sub: "69 KDA + 24 Gated MLA · globali ogni 4° layer e l'ultimo", from: 1, to: 93, color: "#38bdf8" }],
  },

  kinds: {
    K: { label: "KDA", name: "Kimi Delta Attention (lineare)", color: "#2dd4bf", text: "Delta rule con decadimento per canale: uno stato fisso 128 × 128 per head, nessuna KV cache." },
    M: { label: "Gated MLA", name: "Gated MLA + DSA (K3.1)", color: "#38bdf8", text: "Attention globale sul latente KV (512 + 64, NoPE) con gate d'uscita; in K3.1 l'indexer DSA sceglie 2.048 posizioni." },
  },

  params: {
    embedding: 1174405120,
    head: 1174405120,
    finalNorm: 0,
    extra: 0,
    attention: { K: 443740384, M: 232196096 },
    moe: { routed: 896, expert: 33030144, active: 16, shared: 2, sharedExpert: 66060288, router: 6423424, latent: 51383808 },
    dense: { layers: [1], params: 726663168 },
    residual: 0,
    norms: 0,
    flagParams: {},
    addOn: { kind: "M", params: 7438592, label: "Indexer DSA (aggiunto da K3.1)" },
    outside: [{ id: "vision", label: "Visione MoonViT-V2", params: 401000000, approx: true, note: "≈ 401M (README, Table 1); addestrata da zero con next-token prediction" }],
    note: "Le norme d'ingresso dei layer e le pseudo-query di AttnRes non sono contate, come nei conti della ricerca: sono una parte minima del totale.",
    official: "2,78T totali e 104,2B attivi (report K3, Table 1)",
  },

  input: [
    { id: "tokenizer", title: "Tokenizer", sub: "163.840 voci", text: "160K voci come Kimi K2; template XTML con [open] [sep] [close].", params: 0, color: "#fbbf24" },
    { id: "embedding", title: "Embedding", sub: "[163.840 × 7.168]", text: "Una riga da 7.168 numeri per token.", params: 1174405120, color: "#fbbf24" },
    { id: "residual", title: "AttnRes", sub: "residuo sulla profondità", text: "Ogni layer legge con un'attention appresa le uscite dei blocchi da 12 layer precedenti.", params: 0, color: "#38bdf8", narrow: true },
  ],
  output: [
    { id: "norm", title: "Norm", sub: "RMSNorm", text: "Normalizzazione finale.", params: 0, color: "#e2e8f0", narrow: true },
    { id: "head", title: "Head", sub: "[163.840 × 7.168]", text: "Un logit per ognuna delle 163.840 voci.", params: 1174405120, color: "#fbbf24" },
    { id: "softmax", title: "Softmax", sub: "→ prossimo token", text: "Probabilità di ogni token.", params: 0, color: "#4ade80", narrow: true },
  ],
  branches: [
    { id: "vision", title: "MoonViT-V2", sub: "27 layer · patch 14", text: "Visione nativa: merge 2 × 2, ≈ 401M parametri.", params: 401000000, approx: true, attach: "embedding", side: "top", color: "#fda4af" },
    { id: "mtp", title: "Draft MTP", sub: "1 layer · draft EAGLE-3", text: "1 layer MTP rifinito come draft EAGLE-3; per K3.1 vLLM usa un draft DSpark esterno a 5 layer.", params: 0, attach: "last", side: "bottom", color: "#f472b6", external: true },
  ],
  caches: [
    { id: "kv", title: "KV dei 24 layer MLA", sub: "27.648 B per token (BF16)", text: "24 layer × 576 valori; 13.824 B in FP8. K3.1 aggiunge le chiavi dell'indexer DSA.", layers: "M", color: "#22d3ee" },
    { id: "state", title: "Stato KDA fisso", sub: "≈ 434 MB per sequenza (FP32)", text: "69 × 96 × 128 × 128 valori: non cresce con la lunghezza del contesto.", layers: "K", color: "#2dd4bf" },
  ],

  pipeline: [
    { id: "data", title: "Dati", sub: "token non dichiarati", text: "Pre-training nativo multimodale, testo e immagini da zero. Il numero di token non è dichiarato nel report." },
    { id: "pretrain", title: "Pre-training", sub: "Per-Head Muon", text: "Newton–Schulz per head con il weight clipping di K2; weight decay 0,1; cosine con 1% di warmup. Leggi di scala dedicate: ≈ 2,5 × l'efficienza di K2." },
    { id: "context", title: "Contesto lungo", sub: "8K → 64K → 1M", text: "8K → 64K nel pre-training, 256K → 1M nel cooldown. NoPE: niente YaRN." },
    { id: "post", title: "Post-training", sub: "SFT → RL 3 × 3 → MOPD", text: "SFT (template XTML), RL su 3 domini × 3 livelli di effort = 9 esperti, poi distillazione on-policy multi-teacher.", steps: ["SFT", "RL", "MOPD"] },
    { id: "weights", title: "Pesi", sub: "≈ 1.561 GB · MXFP4", text: "QAT MXFP4 da SFT in poi: training e rollout RL usano la stessa quantizzazione. Esperti MXFP4, resto BF16.", unverified: ["shards"] },
    { id: "deploy", title: "Deployment", sub: "8 GPU classe B300", text: "8 GPU B300, oppure 2 vassoi GB300 o un nodo MI355X; VRAM minima indicata da vLLM 1.874 GB." },
    { id: "infer", title: "Inferenza", sub: "104,2B attivi", text: "≈ 208 GFLOP per token; KV solo nei 24 layer MLA, stato fisso nei 69 KDA." },
  ],
  macroNotes: { tokensPerParam: "token di training non dichiarati: il rapporto token/parametro non si può calcolare" },

  /** Stime: parametri × bit per valore (MXFP4 = 4,25 bit; BF16 = 16 bit); la ricetta vLLM dà ≈ 1.561 GB. */
  precision: [
    { id: "mxfp4", dtype: "MXFP4", what: "esperti instradati (stima)", gb: 1446.5, estimate: true },
    { id: "bf16", dtype: "BF16", what: "tutto il resto, visione e indexer DSA compresi (stima)", gb: 114.6, estimate: true },
  ],
  checkpoint: { format: "safetensors", gb: 1561, tensors: null, shards: "96", dtypes: "esperti MXFP4 (weight_packed + weight_scale), resto BF16" },

  kv: {
    perToken: 27648,
    compare: [{ id: "self", label: "Kimi K3.1 Pro (24 layer MLA, BF16)", bytes: 27648 }],
    stateNote: "più uno stato KDA fisso di ≈ 434 MB per sequenza",
  },
  flops: { decode: "≈ 208 GFLOP per token (2 × 104,2B)" },
  memory: "≈ 1,56 TB su disco; VRAM minima indicata da vLLM 1.874 GB (8 GPU B300 o 2 vassoi GB300).",

  anatomy: {
    residual: { streams: 1, label: "Residuo + Attention Residuals", pre: "AttnRes", preSub: "9 sorgenti", post: null, tensor: "pseudo-query w_l per layer" },
    norm: { title: "RMSNorm", attnTensor: "[7168] · nome non verificato", ffnTensor: "[7168] · nome non verificato" },
    attention: {
      title: "Attention ibrida",
      titles: { K: "Kimi Delta Attention (lineare)", M: "Gated MLA + DSA" },
      rows: [
        {
          id: "kda-proj",
          label: "Proiezioni",
          in: "K",
          cards: [
            { id: "kda_q", title: "q_proj", sub: "7.168 → 96 × 128", tensor: "self_attn.q_proj" },
            { id: "kda_k", title: "k_proj", sub: "7.168 → 96 × 128", tensor: "self_attn.k_proj" },
            { id: "kda_v", title: "v_proj", sub: "7.168 → 96 × 128", tensor: "self_attn.v_proj" },
            { id: "kda_conv", title: "conv corta", sub: "kernel 4 · L2Norm su q, k", tensor: "self_attn.{q,k,v}_conv1d" },
          ],
        },
        {
          id: "kda-gate",
          label: "Gate",
          in: "K",
          cards: [
            { id: "kda_b", title: "β", sub: "forza dell'aggiornamento", tensor: "self_attn.b_proj" },
            { id: "kda_f", title: "Decadimento α", sub: "per canale · in (−5, 0)", tensor: "self_attn.f_a / f_b, A_log, dt_bias" },
            { id: "kda_g", title: "Gate d'uscita", sub: "full-rank", tensor: "self_attn.g_proj" },
          ],
        },
        {
          id: "kda-state",
          label: "Stato",
          in: "K",
          cards: [
            { id: "kda_state", title: "Stato S", sub: "128 × 128 per head", tensor: "S = (I − β k kᵀ) Diag(α) S + β k vᵀ" },
          ],
        },
        {
          id: "mla-q",
          label: "Query",
          in: "M",
          cards: [
            { id: "q_a", title: "q_a_proj", sub: "7.168 → 1.536", tensor: "self_attn.q_a_proj" },
            { id: "q_a_norm", title: "q_a_layernorm", sub: "RMSNorm", tensor: "self_attn.q_a_layernorm" },
            { id: "q_b", title: "q_b_proj", sub: "1.536 → 96 × 192", tensor: "self_attn.q_b_proj" },
          ],
        },
        {
          id: "mla-kv",
          label: "KV latente",
          in: "M",
          cards: [
            { id: "kv_a", title: "kv_a_proj_with_mqa", sub: "7.168 → 512 + 64", tensor: "self_attn.kv_a_proj_with_mqa" },
            { id: "kv_a_norm", title: "kv_a_layernorm", sub: "RMSNorm 512", tensor: "self_attn.kv_a_layernorm" },
            { id: "kv_b", title: "kv_b_proj", sub: "512 → 96 × 256", tensor: "self_attn.kv_b_proj" },
          ],
        },
        {
          id: "mla-dsa",
          label: "DSA (K3.1)",
          in: "M",
          cards: [
            { id: "dsa_q", title: "indexer.wq_b", sub: "1.536 → 32 × 128", tensor: "self_attn.indexer.wq_b" },
            { id: "dsa_k", title: "indexer.wk", sub: "7.168 → 128", tensor: "self_attn.indexer.wk · k_norm" },
            { id: "dsa_w", title: "weights_proj", sub: "7.168 → 32", tensor: "self_attn.indexer.weights_proj" },
            { id: "dsa_topk", title: "Top-2.048", sub: "posizioni scelte", tensor: "index_topk 2048" },
          ],
        },
      ],
      core: { id: "core", title: "Delta rule", sub: "stato fisso, niente KV cache", tensor: "lineare nel contesto", alt: { M: { title: "Attention sul latente", sub: "96 head · NoPE", tensor: "posizione dai layer KDA" } } },
      output: [
        { id: "o_norm", title: "o_norm", sub: "norma dell'uscita", tensor: "self_attn.o_norm", in: "K" },
        { id: "g_proj", title: "g_proj", sub: "gate sigmoide", tensor: "self_attn.g_proj", in: "M" },
        { id: "o_proj", title: "o_proj", sub: "12.288 → 7.168", tensor: "self_attn.o_proj" },
      ],
    },
    ffn: {
      title: "LatentMoE: 896 esperti + 2 condivisi",
      router: { id: "router", title: "Router", sub: "sigmoid + bias → top-16", tensor: "block_sparse_moe.gate.weight [896, 7168]", tensor2: "gate.e_score_correction_bias (Quantile Balancing)" },
      latent: [
        { id: "lat_down", title: "W↓", sub: "7.168 → 3.584", tensor: "routed_expert_down_proj" },
        { id: "lat_up", title: "RMSNorm + W↑", sub: "3.584 → 7.168", tensor: "routed_expert_norm · routed_expert_up_proj" },
      ],
      grid: { cols: 32, label: "896 esperti latenti", dtype: "MXFP4" },
      shared: [
        { id: "shared-0", title: "Condiviso 1", sub: "piena larghezza", tensor: "block_sparse_moe.shared_experts" },
        { id: "shared-1", title: "Condiviso 2", sub: "piena larghezza", tensor: "block_sparse_moe.shared_experts" },
      ],
      expert: { id: "expert", title: "Un esperto: SiTU-GLU", sub: "3.584 → 3.072 → 3.584", tensor: "experts.{e}.w1/w2/w3 (weight_packed, weight_scale)", dtype: "MXFP4" },
      sum: { id: "sum", title: "Somma pesata", sub: "pesi normalizzati" },
      dense: { id: "dense", title: "FFN densa", sub: "7.168 → 33.792 → 7.168", tensor: "mlp.{gate,up,down}_proj", when: "dense" },
    },
    branch: null,
  },

  mechanisms: {
    compare: { at: "detail:layer", status: "present", layer: 4, text: "Stessa ossatura (norm → attention → norm → FFN), ma l'attention è ibrida (KDA lineare o Gated MLA) e il residuo ha le Attention Residuals sulla profondità." },
    moe: { at: "detail:experts", status: "present", layer: 2, text: "LatentMoE: 896 esperti nello spazio latente da 3.584, 16 attivi per token, più 2 condivisi a piena larghezza." },
    gqa: { at: "detail:core", status: "present", layer: 4, text: "Nei 24 layer Gated MLA il KV è un latente da 512 + 64, come in MLA; i 69 layer KDA non hanno KV cache ma uno stato fisso 128 × 128 per head." },
    rope: { at: "detail:core", status: "absent", layer: 4, text: "Qui vivrebbe, ma questo modello non usa RoPE (NoPE): la posizione la danno il gating e il decadimento dei layer KDA." },
    bpe: { at: "blocks:tokenizer", status: "present", text: "Tokenizer da 163.840 voci (160K), invariato rispetto a Kimi K2." },
    kv: { at: "blocks:cache-kv", status: "present", text: "Solo i 24 layer MLA crescono con il contesto: 27.648 B per token in BF16, più uno stato KDA fisso di circa 434 MB per sequenza." },
    flash: { at: "detail:dsa_topk", status: "present", layer: 4, text: "DSA (aggiunta in K3.1): un lightning indexer sceglie le 2.048 posizioni migliori nei layer MLA; i layer KDA sono lineari nel contesto." },
    spec: { at: "blocks:mtp", status: "present", text: "1 layer MTP pre-addestrato e rifinito come draft EAGLE-3 (non nel backbone caricato di default); per K3.1 vLLM usa un draft DSpark esterno a 5 layer." },
    scaling: { at: "macro:pretrain", status: "present", text: "Leggi di scala dedicate: circa 2,5 volte l'efficienza di Kimi K2 (report)." },
    training: { at: "macro:post", status: "present", text: "SFT → RL su 3 domini × 3 livelli di effort (9 esperti) → distillazione on-policy multi-teacher (MOPD)." },
    reasoning: { at: "macro:post", status: "present", text: "Canali think / response / tools e livelli di effort low, high, max addestrati con l'RL." },
    interp: { at: "detail:residual", status: "study", layer: 4, text: "Non è un pezzo del modello: è lo studio delle attivazioni che passano nel residuo, nelle head e negli esperti." },
    agent: { at: "blocks:tokenizer", status: "present", text: "Template XTML con [open] [sep] [close] e il canale tools; l'RL copre anche «general agents» e «coding agents»." },
    coding: { at: "macro:post", status: "present", text: "Uno dei 3 domini dell'RL è «coding agents»." },
    multi: { at: "blocks:vision", status: "present", text: "MoonViT-V2: 27 layer, patch 14, merge 2 × 2, addestrato da zero con next-token prediction." },
    mesh: { at: "blocks:softmax", status: "absent", text: "Qui vivrebbe: una mesh OBJ è solo testo in uscita. Questo modello non dichiara un addestramento per le mesh 3D." },
  },

  tensorGroups: [
    { id: "kda", label: "Layer KDA (69)", tensors: [["…self_attn.{q,k,v,g,b,f_a,f_b,o}_proj", "", ""], ["…self_attn.{q,k,v}_conv1d", "kernel 4", ""], ["…self_attn.A_log, dt_bias, o_norm", "", ""]] },
    { id: "mla", label: "Layer Gated MLA (24)", tensors: [["…self_attn.{q_a_proj, q_a_layernorm, q_b_proj}", "", ""], ["…self_attn.{kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj}", "", ""], ["…self_attn.{g_proj, o_proj}", "", ""], ["…self_attn.indexer.{wq_b, wk, weights_proj, k_norm}", "solo K3.1", ""]] },
    { id: "moe", label: "MoE (92 layer)", tensors: [["…block_sparse_moe.gate.weight / e_score_correction_bias", "[896, 7168]", ""], ["…block_sparse_moe.experts.{e}.w1/w2/w3", "weight_packed, weight_scale", "MXFP4"], ["…block_sparse_moe.shared_experts.*", "", "BF16"], ["…routed_expert_down_proj / routed_expert_norm / routed_expert_up_proj", "", ""]] },
    { id: "dense", label: "Layer 1 (FFN densa)", tensors: [["language_model.model.layers.0.mlp.{gate,up,down}_proj.weight", "33.792", ""]] },
  ],
  tensorNote: "Nomi dedotti dal loader vLLM (vllm/models/kimi_k3/nvidia/model.py): l'indice model.safetensors.index.json non è stato consultato.",

  unverified: [
    { id: "shards", value: "96 shard", why: "Il numero di shard viene dalla stampa; la dimensione (≈ 1.561 GB) dalla ricetta vLLM." },
    { id: "weights-date", value: "data dei pesi K3.1", why: "Fonti discordi: un tracker registra l'upload il 2/11, la ricetta vLLM è del 2/11, la scadenza annunciata era il 4/11, una fonte dice 5/11." },
    { id: "license", value: "licenza Kimi K3.1-Pro", why: "Nota solo dalla stampa (la licenza di K3 è letta dal testo ufficiale)." },
    { id: "config", value: "config.json e indice HF di K3.1", why: "Non letti direttamente: valori dai default NeMo dichiarati uguali al rilascio e dal codice vLLM." },
  ],

  sources: [
    { title: "Kimi K3: README ufficiale", url: "https://github.com/MoonshotAI/Kimi-K3", note: "Tabella dell'architettura (20 valori)." },
    { title: "Kimi K3 Technical Report (PDF, 47 pagine)", url: "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf", note: "KDA, AttnRes, LatentMoE, Quantile Balancing, SiTU-GLU, training, Table 1." },
    { title: "NeMo AutoModel: kimi_k3/config.py", url: "https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/nemo_automodel/components/models/kimi_k3/config.py", note: "Default = config rilasciato di K3 e dsa_config di K3.1." },
    { title: "vLLM: configurazione kimi_linear (DSA di K3.1)", url: "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/transformers_utils/configs/kimi_linear.py", note: "Campi del config." },
    { title: "vLLM: codice del modello Kimi K3 (loader)", url: "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/kimi_k3/nvidia/model.py", note: "Da qui i nomi dei tensori (KDA, MLA, LatentMoE, indexer DSA): l'indice HF non è stato consultato." },
    { title: "Ricetta vLLM Kimi-K3.1-Pro", url: "https://raw.githubusercontent.com/vllm-project/vllm/main/recipes/models/moonshotai/Kimi-K3.1-Pro.yaml", note: "≈ 1.561 GB, MXFP4, hardware." },
    { title: "Licenza Kimi K3", url: "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/LICENSE", note: "Soglie per Model-as-a-Service e attribuzione." },
    { title: "Tech blog Kimi K3 (link dal README ufficiale)", url: "https://www.kimi.com/blog/kimi-k3", note: "Annuncio di K3." },
  ],
};
