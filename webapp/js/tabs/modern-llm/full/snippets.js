/**
 * Codice del pannello per la vista "Architettura LLM moderni completa" (SOLI DATI).
 * - Estratti VERI del motore ufficiale DeepInfer (Apache-2.0), copiati testualmente con il numero di
 *   riga d'origine: names, router, cache, indexkey (file e righe nel campo `file` / `startLine`).
 * - "config": il config.json ricostruito dalla ricerca (estratto), dichiarato come tale.
 * - Pseudocodice scritto dalle formule della ricerca verificata, dichiarato nel titolo e nel commento.
 * Generato da uno script dalle fonti scaricate dalla ricerca; non modificare a mano gli estratti veri.
 */
export const FULL_SNIPPETS = {
  "names": {
    "title": "DeepInfer: dai nomi del checkpoint ai moduli",
    "file": "deepseek-ai/DeepInfer · deepinfer/hugging_face/names.py",
    "startLine": 35,
    "code": "# HF suffix -> internal module suffix.\n_HF_TO_MODEL_SUFFIX = {\n    \"hc_attn_base\": \"hc_attn\",\n    \"hc_ffn_base\": \"hc_ffn\",\n    \"hc_attn_fn\": \"hc_attn_fn\",\n    \"hc_attn_scale\": \"hc_attn_scale\",\n    \"hc_ffn_fn\": \"hc_ffn_fn\",\n    \"hc_ffn_scale\": \"hc_ffn_scale\",\n    \"attn.attn_sink\": \"self_attn.attn_sink_params\",\n    \"attn.wq_a.weight\": \"self_attn.Wq.A.weight\",\n    \"attn.wq_b.weight\": \"self_attn.Wq.B.weight\",\n    \"attn.q_norm.weight\": \"self_attn.Wq.norm.weight\",\n    \"attn.wkv.weight\": \"self_attn.Wkv.A.weight\",\n    \"attn.kv_norm.weight\": \"self_attn.Wkv.norm.weight\",\n    \"attn.wo_a.weight\": \"self_attn.Wv.weight\",\n    \"attn.wo_b.weight\": \"self_attn.out_proj.weight\",\n    \"attn.compressor.wkv.weight\": \"self_attn.PackedWkv.A.weight\",\n    \"attn.compressor.wgate.weight\": \"self_attn.PackedWkv.Gate.weight\",\n    \"attn.compressor.norm.weight\": \"self_attn.PackedWkv.norm.weight\",\n    \"attn.indexer.wq_b.weight\": \"self_attn.index_attn.q_B.weight\",\n    \"attn.indexer.wk.weight\": \"self_attn.index_attn.k_proj.weight\",\n    \"attn.indexer.k_norm.weight\": \"self_attn.index_attn.k_norm.weight\",\n    \"attn.indexer.weights_proj.weight\": \"self_attn.index_attn.weights_proj.weight\",\n    \"attn_norm.weight\": \"norm1.weight\",\n    \"ffn_norm.weight\": \"norm2.weight\",\n    \"ffn.gate.weight\": \"experts.gate.Wgate.weight\",\n    \"ffn.gate.bias\": \"experts.gate.e_score_correction_bias\",\n    \"ffn.gate.bias_vl\": \"experts.gate.e_score_correction_bias_vl\",\n    # linear1 = [w1; w3] concatenated along dim 0.\n    \"ffn.shared_experts.w1.weight\": \"experts.shared_experts.0.linear1.weight\",\n    \"ffn.shared_experts.w3.weight\": \"experts.shared_experts.0.linear1.weight\",\n    \"ffn.shared_experts.w2.weight\": \"experts.shared_experts.0.linear2.weight\",\n    \"engram.q_weight\": \"engram_embedding.weight_hidden\",\n    \"engram.k_weight\": \"engram_embedding.weight_embed\",\n    \"engram.embed.weight\": \"engram_embedding.embedding.embedding\",\n    \"engram.wkv.weight\": \"engram_embedding.fused_kv_projection.weight\",\n}",
    "language": "python",
    "highlight": [
      "attn.wq_a.weight",
      "attn.wkv.weight",
      "attn.compressor.wkv.weight",
      "ffn.gate.weight",
      "engram.embed.weight",
      "hc_attn_fn"
    ]
  },
  "router": {
    "title": "DeepInfer: il router del MoE (sqrtsoftplus + bias)",
    "file": "deepseek-ai/DeepInfer · deepinfer/model_executor/models/v4_modules/moe.py",
    "startLine": 40,
    "code": "class TopKGate(nn.Module):\n    \"\"\"Top-k routing gate of the MoE.\n\n    Scoring uses sqrtsoftplus; top-k selection uses noaux_tc plus a load-correction bias.\n    The parameters are `Wgate` and two FP32 biases (one for text, one for image).\n    \"\"\"\n\n    def __init__(self, cfg: Any, layer_id: int | None = None) -> None:\n        super().__init__()\n        moe_cfg = cfg.moe\n        self.cfg = cfg\n        self.layer_id = layer_id\n        self.top_k = moe_cfg.num_experts_per_tok\n        self.n_routed_experts = moe_cfg.n_routed_experts\n        self.routed_scaling_factor = float(moe_cfg.routed_scaling_factor)\n        self.scoring_func = moe_cfg.get(\"scoring_func\", \"sqrtsoftplus\")\n        self.norm_topk_prob = moe_cfg.norm_topk_prob\n        self.infer_topk_method = moe_cfg.topk_method\n        assert self.scoring_func == \"sqrtsoftplus\"\n        assert self.infer_topk_method == \"noaux_tc\"\n        assert self.norm_topk_prob\n\n        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts, dtype=torch.float32))\n        # HF V4.1 always contains a separate routing bias for image tokens (ffn.gate.bias_vl).\n        self.e_score_correction_bias_vl = nn.Parameter(torch.zeros(self.n_routed_experts, dtype=torch.float32))\n\n        # The gate projection is always bf16 (it does not take part in fp8 quantization)\n        self.Wgate = ReplicatedLinear(cfg.hidden_size, self.n_routed_experts, bias=False, use_fp8_gemm=False)",
    "language": "python",
    "highlight": [
      "sqrtsoftplus",
      "noaux_tc",
      "e_score_correction_bias",
      "bias_vl"
    ]
  },
  "cache": {
    "title": "DeepInfer: quanti byte occupa la KV cache",
    "file": "deepseek-ai/DeepInfer · deepinfer/worker/cache_engine.py",
    "startLine": 205,
    "code": "@dataclass\nclass KVCache:\n    ica_cache: torch.Tensor | None = None\n    \"\"\"index attn cache: [num_blocks, block_size // ratio, 1, index_bytes_per_token], uint8.\"\"\"\n    ica_state: torch.Tensor | None = None\n    \"\"\"index attn state: [num_state_seqs, ring_size, 2, index_dim], fp32.\"\"\"\n    ca_cache: torch.Tensor | None = None\n    \"\"\"compressed attn cache: [num_blocks, block_size // ratio, 1, bytes_per_token], uint8.\n    V41 stores [data | scale] per token: 288 bytes for per-16 FP4 or 528 for MXFP8.\"\"\"\n    ca_state: torch.Tensor | None = None\n    \"\"\"compressed attn state: [num_state_seqs, ring_size, 2, dim], fp32.\"\"\"\n    swa_cache: torch.Tensor | None = None\n    \"\"\"sliding window attn cache: [num_state_seqs, swa_block_size, 1, 528], uint8.\n    Each V41 token stores [512B FP8 data | 16B UE8M0 scale].\"\"\"",
    "language": "python",
    "highlight": [
      "288 bytes",
      "528], uint8",
      "512B FP8"
    ]
  },
  "indexkey": {
    "title": "DeepInfer: byte di una chiave dell'indexer",
    "file": "deepseek-ai/DeepInfer · deepinfer/worker/cache_engine.py",
    "startLine": 76,
    "code": "    def per_token_bytes(ic: AttrDict) -> int:\n        # FP4: two values per byte, one UE8M0 scale per 32 channels.\n        return ic.index_head_dim // 2 + ic.index_head_dim // 32",
    "language": "python",
    "highlight": [
      "two values per byte",
      "index_head_dim // 2"
    ]
  },
  "config": {
    "title": "config.json ricostruito (estratto): DeepSeek-V4.1-Flash",
    "file": "ricostruito da DeepInfer, NeMo e vLLM · non letto direttamente da Hugging Face",
    "startLine": 1,
    "code": "{\n  \"architectures\": [\"DeepseekV41ForCausalLM\"],\n  \"quantization_config\": {\n    \"quant_method\": \"fp8\",\n    \"expert_dtype\": \"fp4\",\n    \"activation_scheme\": \"dynamic\",\n    \"weight_block_size\": [32, 32],\n    \"scale_fmt\": \"ue8m0\"\n  },\n  \"text_config\": {\n    \"vocab_size\": 129280,\n    \"hidden_size\": 5120,\n    \"num_hidden_layers\": 40,\n    \"num_attention_heads\": 64,\n    \"num_key_value_heads\": 1,\n    \"head_dim\": 512,\n    \"qk_rope_head_dim\": 64,\n    \"q_lora_rank\": 1280,\n    \"o_lora_rank\": 1024,\n    \"o_groups\": 8,\n    \"sliding_window\": 128,\n    \"n_routed_experts\": 384,\n    \"n_shared_experts\": 1,\n    \"num_experts_per_tok\": 6,\n    \"moe_intermediate_size\": 2304,\n    \"scoring_func\": \"sqrtsoftplus\",\n    \"routed_scaling_factor\": 1.5,\n    \"swiglu_limit\": 10.0,\n    \"max_position_embeddings\": 1048576,\n    \"rope_scaling\": {\n      \"rope_type\": \"yarn\",\n      \"factor\": 16,\n      \"beta_fast\": 32,\n      \"beta_slow\": 1,\n      \"original_max_position_embeddings\": 65536\n    },\n    \"compress_ratios\": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],\n    \"kv_source_layer_ids\": [2, 8, 14, 20],\n    \"index_source_layer_ids\": [2, 8, 14, 20, 24, 28, 32, 36],\n    \"index_n_heads\": 32,\n    \"index_head_dim\": 128,\n    \"index_topk\": 512,\n    \"hc_mult\": 4,\n    \"engram_layer_ids\": [1, 14],\n    \"num_nextn_predict_layers\": 3,\n    \"dspark_block_size\": 5\n  }\n}",
    "language": "json",
    "highlight": [
      "\"n_routed_experts\"",
      "\"compress_ratios\"",
      "\"kv_source_layer_ids\"",
      "\"engram_layer_ids\""
    ]
  },
  "csa2": {
    "title": "Pseudocodice: l'attention di un layer (CSA2 + finestra)",
    "file": "pseudocodice dalle formule della ricerca verificata",
    "startLine": 1,
    "code": "# Pseudocodice dalle formule della ricerca (§4.4): NON è il codice ufficiale\nq = W_qb(rms_norm(W_qa(x)))                  # 5.120 → 1.280 → 64 head × 512\nq[..., -64:] = rope(q[..., -64:])           # RoPE solo su 64 delle 512 dimensioni\nc = rms_norm(W_kv(x))                       # una sola KV head da 512: K = V\nfinestra = cache_swa[-128:]                 # gli ultimi 128 token, calcolati da questo layer\nif layer in (2, 8, 14):                     # Full dell'encoder: 2 token → 1 voce globale\n    voce = pooling_softmax(wkv(x_coppia), wgate(x_coppia))\nelif layer == 20:                           # Full del decoder: dall'uscita dell'encoder\n    voce = H20 @ W_kv_20\npunteggio = sum_h(w[h] * relu(q_idx[h] @ k_idx))   # lightning indexer: 32 head × 128\nscelte = topk(punteggio, 512)               # dal layer 20: prima 2.048 blocchi × 8\nout = mqa(q, finestra + kv_globale[scelte], sink=attn_sink)\ny = W_ob(W_oa(out))                         # 8 gruppi: 4.096 → 1.024, poi 8.192 → 5.120",
    "language": "python",
    "highlight": [
      "topk(punteggio, 512)",
      "K = V",
      "pooling_softmax"
    ]
  },
  "moe": {
    "title": "Pseudocodice: il router e gli esperti",
    "file": "pseudocodice dalle formule della ricerca verificata",
    "startLine": 1,
    "code": "# Pseudocodice dalla ricerca (§4.5): NON è il codice ufficiale\ns = sqrtsoftplus(W_gate @ x)                # 384 punteggi, uno per esperto\nscelti = topk(s + bias, 6)                  # il bias di bilanciamento conta solo per scegliere\np = s[scelti] / s[scelti].sum() * 1.5       # pesi normalizzati × routed_scaling_factor\ny = condiviso(x) + sum(p[i] * esperto[i](x) for i in scelti)\n\ndef esperto(x):                             # SwiGLU 5.120 → 2.304 → 5.120\n    return w2(swiglu(w1(x), w3(x), limite=10))   # limite 10; la forma esatta del clamp non è pubblicata",
    "language": "python",
    "highlight": [
      "topk(s + bias, 6)",
      "* 1.5"
    ]
  },
  "kvmath": {
    "title": "Il conto dei 890 byte per token",
    "file": "pseudocodice dalle formule della ricerca verificata",
    "startLine": 1,
    "code": "# Conti della ricerca (§4.14), confermati dalle costanti di cache_engine.py\nvoce_fp4 = 512 * 4 // 8 + 512 // 16        # 256 B di numeri + 32 scale da 1 byte = 288 B\nchiave_indexer = 128 // 2 + 128 // 32      # 64 + 4 = 68 B\nvoci_per_token = 3 * (1 / 2) + 1           # 3 Full dell'encoder (2 → 1) + il Full del decoder = 2,5\nkv = voci_per_token * (voce_fp4 + chiave_indexer)   # 2,5 × 356 = 890 B per token\n\nv3 = 61 * (512 + 64) * 2                   # DeepSeek-V3, MLA in BF16: 70.272 B\ngqa_67b = 95 * 2 * 8 * 128 * 2             # DeepSeek LLM 67B, GQA in BF16: 389.120 B ≈ 437 × 890",
    "language": "python",
    "highlight": [
      "kv = voci_per_token",
      "437"
    ]
  },
  "engram": {
    "title": "Pseudocodice: Engram, una memoria di n-gram",
    "file": "pseudocodice dalle formule della ricerca verificata",
    "startLine": 1,
    "code": "# Pseudocodice dalla ricerca (§4.6): NON è il codice ufficiale\nt = comprimi(token)                         # 99.092 classi: NFKC, minuscole, senza accenti\nrighe = []\nfor n in (2, 3, 4):                         # bigrammi, trigrammi, 4-grammi\n    for h in range(8):                      # 8 hash per ordine, tabelle con un numero primo di righe\n        righe.append(tabella[n][h][hash(h, t[-n:]) % righe_tabella[n][h]])   # 256 numeri\ne = concat(righe)                           # 24 × 256 = 6.144 numeri\nkv = engram_wkv(e)                          # 6.144 → 5.120 × (4 + 1)\nflussi = flussi + gate(kv, flussi)          # si somma ai 4 flussi mHC, all'inizio del layer",
    "language": "python",
    "highlight": [
      "for n in (2, 3, 4)",
      "gate(kv, flussi)"
    ]
  },
  "ced": {
    "title": "Pseudocodice: encoder-decoder causale",
    "file": "pseudocodice dalle formule della ricerca verificata",
    "startLine": 1,
    "code": "# Pseudocodice dalla ricerca (§4.8): NON è il codice ufficiale\nh = embedding(prompt)\nfor l in range(0, 20):                      # ENCODER causale: il prefill si ferma qui (8B attivi)\n    h = layer[l](h)\nH20 = h                                     # uscita dell'encoder\nkv_globale = H20 @ W_kv_20                  # il layer 20 proietta il KV di tutto il decoder\n\n# DECODE: per ogni token nuovo girano tutti i 40 layer (16B attivi);\n# il decoder si riesegue solo sugli ultimi 128 token del prompt (replay della finestra)",
    "language": "python",
    "highlight": [
      "range(0, 20)",
      "H20 @ W_kv_20"
    ]
  },
  "dspark": {
    "title": "Pseudocodice: DSpark propone 5 token",
    "file": "pseudocodice dalle formule della ricerca verificata",
    "startLine": 1,
    "code": "# Pseudocodice dalla ricerca (§4.9): NON è il codice ufficiale\nf = main_proj(concat(h37, h38, h39))        # 15.360 → 5.120: le feature degli ultimi 3 layer\nfor blocco in dspark:                       # 3 blocchi: finestra 128 + MoE da 128 esperti, top-3\n    f = blocco(f)\nbozza = proponi(f, 5, correzione=markov_head)  # 5 token; la testa di Markov (rango 256) corregge i logit al primo ordine\nfiducia = confidence_head(f)                # stima della probabilità che vengano accettati\naccettati = verifica(modello, bozza)        # un solo passaggio del modello per tutta la bozza",
    "language": "python",
    "highlight": [
      "markov_head",
      "confidence_head"
    ]
  },
  "safetensors": {
    "title": "Il formato safetensors",
    "file": "pseudocodice dal formato descritto nel README di huggingface/safetensors",
    "startLine": 1,
    "code": "# Pseudocodice dal formato descritto nel README di huggingface/safetensors: NON è il codice ufficiale\nwith open(\"model.safetensors\", \"rb\") as f:\n    n = int.from_bytes(f.read(8), \"little\")  # 8 byte: la lunghezza dell'intestazione\n    header = json.loads(f.read(n))            # {nome: {\"dtype\", \"shape\", \"data_offsets\"}}\n    # es. \"layers.2.attn.wq_a.weight\": {\"dtype\": \"F8_E4M3\", \"shape\": [1280, 5120], …}\n    dati = f.read()                           # poi solo numeri grezzi: niente codice da eseguire",
    "language": "python",
    "highlight": [
      "f.read(8)",
      "json.loads"
    ]
  }
};
