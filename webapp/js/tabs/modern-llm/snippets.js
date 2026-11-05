/**
 * Frammenti di codice VERI mostrati nel pannello del codice.
 * - Codice di modelli aperti, copiato testualmente con il numero di riga d'origine
 *   (generato da uno script dai file ufficiali; repository nel campo `file`).
 * - Il nostro GPT: da data/model.json (esportato da gpt/export.py), letto a runtime.
 * - Le funzioni JavaScript di questo tab: lette dal sorgente di sim.js a runtime, così il codice
 *   mostrato è esattamente quello che fa girare le animazioni.
 */
import { loadModel } from "./ourModel.js";

export const EXTERNAL = {
  "llama.rmsnorm": {
    "title": "Llama 3: RMSNorm",
    "file": "meta-llama/llama3 · llama/model.py",
    "startLine": 35,
    "code": "class RMSNorm(torch.nn.Module):\n    def __init__(self, dim: int, eps: float = 1e-6):\n        super().__init__()\n        self.eps = eps\n        self.weight = nn.Parameter(torch.ones(dim))\n\n    def _norm(self, x):\n        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n\n    def forward(self, x):\n        output = self._norm(x.float()).type_as(x)\n        return output * self.weight",
    "language": "python",
    "highlight": [
      "rsqrt",
      "self.weight"
    ]
  },
  "llama.rope": {
    "title": "Llama 3: RoPE (frequenze e rotazione)",
    "file": "meta-llama/llama3 · llama/model.py",
    "startLine": 49,
    "code": "def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):\n    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))\n    t = torch.arange(end, device=freqs.device, dtype=torch.float32)\n    freqs = torch.outer(t, freqs)\n    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64\n    return freqs_cis\n\n\ndef reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):\n    ndim = x.ndim\n    assert 0 <= 1 < ndim\n    assert freqs_cis.shape == (x.shape[1], x.shape[-1])\n    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]\n    return freqs_cis.view(*shape)\n\n\ndef apply_rotary_emb(\n    xq: torch.Tensor,\n    xk: torch.Tensor,\n    freqs_cis: torch.Tensor,\n) -> Tuple[torch.Tensor, torch.Tensor]:\n    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))\n    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))\n    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)\n    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)\n    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)\n    return xq_out.type_as(xq), xk_out.type_as(xk)",
    "language": "python",
    "highlight": [
      "theta **",
      "torch.polar",
      "xq_ * freqs_cis",
      "xk_ * freqs_cis"
    ]
  },
  "llama.repeat_kv": {
    "title": "Llama 3: GQA, le head K/V ripetute per le query",
    "file": "meta-llama/llama3 · llama/model.py",
    "startLine": 78,
    "code": "def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:\n    \"\"\"torch.repeat_interleave(x, dim=2, repeats=n_rep)\"\"\"\n    bs, slen, n_kv_heads, head_dim = x.shape\n    if n_rep == 1:\n        return x\n    return (\n        x[:, :, :, None, :]\n        .expand(bs, slen, n_kv_heads, n_rep, head_dim)\n        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)\n    )",
    "language": "python",
    "highlight": [
      "n_rep",
      "expand"
    ]
  },
  "llama.init_kv": {
    "title": "Llama 3: proiezioni con meno head K/V",
    "file": "meta-llama/llama3 · llama/model.py",
    "startLine": 90,
    "code": "class Attention(nn.Module):\n    def __init__(self, args: ModelArgs):\n        super().__init__()\n        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads\n        model_parallel_size = fs_init.get_model_parallel_world_size()\n        self.n_local_heads = args.n_heads // model_parallel_size\n        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size\n        self.n_rep = self.n_local_heads // self.n_local_kv_heads\n        self.head_dim = args.dim // args.n_heads\n\n        self.wq = ColumnParallelLinear(\n            args.dim,\n            args.n_heads * self.head_dim,\n            bias=False,\n            gather_output=False,\n            init_method=lambda x: x,\n        )\n        self.wk = ColumnParallelLinear(\n            args.dim,\n            self.n_kv_heads * self.head_dim,\n            bias=False,\n            gather_output=False,\n            init_method=lambda x: x,\n        )\n        self.wv = ColumnParallelLinear(\n            args.dim,\n            self.n_kv_heads * self.head_dim,\n            bias=False,\n            gather_output=False,\n            init_method=lambda x: x,\n        )",
    "language": "python",
    "highlight": [
      "n_kv_heads",
      "self.n_rep",
      "self.wk",
      "self.wv"
    ]
  },
  "llama.cache": {
    "title": "Llama 3: attention con KV cache",
    "file": "meta-llama/llama3 · llama/model.py",
    "startLine": 146,
    "code": "    def forward(\n        self,\n        x: torch.Tensor,\n        start_pos: int,\n        freqs_cis: torch.Tensor,\n        mask: Optional[torch.Tensor],\n    ):\n        bsz, seqlen, _ = x.shape\n        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)\n\n        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)\n        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)\n        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)\n\n        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)\n\n        self.cache_k = self.cache_k.to(xq)\n        self.cache_v = self.cache_v.to(xq)\n\n        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk\n        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv\n\n        keys = self.cache_k[:bsz, : start_pos + seqlen]\n        values = self.cache_v[:bsz, : start_pos + seqlen]\n\n        # repeat k/v heads if n_kv_heads < n_heads\n        keys = repeat_kv(\n            keys, self.n_rep\n        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)\n        values = repeat_kv(\n            values, self.n_rep\n        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)\n\n        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)\n        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)\n        values = values.transpose(\n            1, 2\n        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)\n        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)\n        if mask is not None:\n            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)\n        scores = F.softmax(scores.float(), dim=-1).type_as(xq)\n        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)\n        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)\n        return self.wo(output)",
    "language": "python",
    "highlight": [
      "self.cache_k[:bsz, start_pos",
      "self.cache_v[:bsz, start_pos",
      "keys = self.cache_k",
      "values = self.cache_v",
      "repeat_kv("
    ]
  },
  "llama.ffn": {
    "title": "Llama 3: SwiGLU",
    "file": "meta-llama/llama3 · llama/model.py",
    "startLine": 193,
    "code": "class FeedForward(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        hidden_dim: int,\n        multiple_of: int,\n        ffn_dim_multiplier: Optional[float],\n    ):\n        super().__init__()\n        hidden_dim = int(2 * hidden_dim / 3)\n        # custom dim factor multiplier\n        if ffn_dim_multiplier is not None:\n            hidden_dim = int(ffn_dim_multiplier * hidden_dim)\n        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)\n\n        self.w1 = ColumnParallelLinear(\n            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x\n        )\n        self.w2 = RowParallelLinear(\n            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x\n        )\n        self.w3 = ColumnParallelLinear(\n            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x\n        )\n\n    def forward(self, x):\n        return self.w2(F.silu(self.w1(x)) * self.w3(x))",
    "language": "python",
    "highlight": [
      "int(2 * hidden_dim / 3)",
      "F.silu(self.w1(x)) * self.w3(x)"
    ]
  },
  "llama.block": {
    "title": "Llama 3: un blocco decoder",
    "file": "meta-llama/llama3 · llama/model.py",
    "startLine": 222,
    "code": "class TransformerBlock(nn.Module):\n    def __init__(self, layer_id: int, args: ModelArgs):\n        super().__init__()\n        self.n_heads = args.n_heads\n        self.dim = args.dim\n        self.head_dim = args.dim // args.n_heads\n        self.attention = Attention(args)\n        self.feed_forward = FeedForward(\n            dim=args.dim,\n            hidden_dim=4 * args.dim,\n            multiple_of=args.multiple_of,\n            ffn_dim_multiplier=args.ffn_dim_multiplier,\n        )\n        self.layer_id = layer_id\n        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)\n        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)\n\n    def forward(\n        self,\n        x: torch.Tensor,\n        start_pos: int,\n        freqs_cis: torch.Tensor,\n        mask: Optional[torch.Tensor],\n    ):\n        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)\n        out = h + self.feed_forward(self.ffn_norm(h))\n        return out",
    "language": "python",
    "highlight": [
      "RMSNorm",
      "h = x + self.attention",
      "out = h + self.feed_forward"
    ]
  },
  "mistral.moe": {
    "title": "Mixtral: il layer Mixture of Experts",
    "file": "mistralai/mistral-inference · src/mistral_inference/moe.py",
    "startLine": 16,
    "code": "class MoeLayer(nn.Module):\n    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):\n        super().__init__()\n        assert len(experts) > 0\n        self.experts = nn.ModuleList(experts)\n        self.gate = gate\n        self.args = moe_args\n\n    def forward(self, inputs: torch.Tensor) -> torch.Tensor:\n        gate_logits = self.gate(inputs)\n        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)\n        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)\n        results = torch.zeros_like(inputs)\n        for i, expert in enumerate(self.experts):\n            batch_idx, nth_expert = torch.where(selected_experts == i)\n            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])\n        return results",
    "language": "python",
    "highlight": [
      "torch.topk",
      "F.softmax(weights",
      "results[batch_idx] +="
    ]
  },
  "minbpe.train": {
    "title": "BPE a byte: addestramento",
    "file": "karpathy/minbpe · minbpe/basic.py",
    "startLine": 20,
    "code": "    def train(self, text, vocab_size, verbose=False):\n        assert vocab_size >= 256\n        num_merges = vocab_size - 256\n\n        # input text preprocessing\n        text_bytes = text.encode(\"utf-8\") # raw bytes\n        ids = list(text_bytes) # list of integers in range 0..255\n\n        # iteratively merge the most common pairs to create new tokens\n        merges = {} # (int, int) -> int\n        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes\n        for i in range(num_merges):\n            # count up the number of times every consecutive pair appears\n            stats = get_stats(ids)\n            # find the pair with the highest count\n            pair = max(stats, key=stats.get)\n            # mint a new token: assign it the next available id\n            idx = 256 + i\n            # replace all occurrences of pair in ids with idx\n            ids = merge(ids, pair, idx)\n            # save the merge\n            merges[pair] = idx\n            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]\n            # prints\n            if verbose:\n                print(f\"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences\")\n\n        # save class variables\n        self.merges = merges # used in encode()\n        self.vocab = vocab   # used in decode()\n\n    def decode(self, ids):",
    "language": "python",
    "highlight": [
      "text.encode(\"utf-8\")",
      "get_stats(ids)",
      "max(stats",
      "merge(ids, pair, idx)"
    ]
  },
  "dpo.loss": {
    "title": "DPO: la loss sulle coppie di preferenze",
    "file": "eric-mitchell/direct-preference-optimization · trainers.py",
    "startLine": 45,
    "code": "def preference_loss(policy_chosen_logps: torch.FloatTensor,\n                    policy_rejected_logps: torch.FloatTensor,\n                    reference_chosen_logps: torch.FloatTensor,\n                    reference_rejected_logps: torch.FloatTensor,\n                    beta: float,\n                    label_smoothing: float = 0.0,\n                    ipo: bool = False,\n                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:\n    \"\"\"Compute the DPO loss for a batch of policy and reference model log probabilities.\n\n    Args:\n        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)\n        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)\n        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)\n        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)\n        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.\n        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)\n        ipo: If True, use the IPO loss instead of the DPO loss.\n        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.\n\n    Returns:\n        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).\n        The losses tensor contains the DPO loss for each example in the batch.\n        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.\n    \"\"\"\n    pi_logratios = policy_chosen_logps - policy_rejected_logps\n    ref_logratios = reference_chosen_logps - reference_rejected_logps\n\n    if reference_free:\n        ref_logratios = 0\n\n    logits = pi_logratios - ref_logratios  # also known as h_{\\pi_\\theta}^{y_w,y_l}\n\n    if ipo:\n        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf\n    else:\n        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)\n        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing\n\n    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()\n    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()\n\n    return losses, chosen_rewards, rejected_rewards",
    "language": "python",
    "highlight": [
      "pi_logratios =",
      "ref_logratios =",
      "logits = pi_logratios - ref_logratios",
      "F.logsigmoid(beta * logits)"
    ]
  }
};

/** Chiavi di data/model.json → titolo mostrato (il nostro GPT, per il confronto). */
const OURS = {
  block: { title: "Il nostro GPT: un blocco", highlight: ["nn.LayerNorm", "x = x + self.sa", "x = x + self.ffwd"] },
  ffwd: { title: "Il nostro GPT: MLP con ReLU", highlight: ["nn.ReLU", "hidden = 4 * config.n_embd"] },
  mha: { title: "Il nostro GPT: multi-head attention (K e V per ogni head)", highlight: ["Head(config) for _ in range", "torch.cat"] },
  "model.init": { title: "Il nostro GPT: embedding di posizione appresi", highlight: ["position_embedding_table", "nn.LayerNorm"] },
  "data.tokenizer": { title: "Il nostro GPT: tokenizer a caratteri", highlight: ["class CharTokenizer", "def encode"] },
  generate: { title: "Il nostro GPT: generazione (senza KV cache)", highlight: ["logits, _ = model(idx_cond)", "idx = torch.cat"] },
};

const SIM_URL = new URL("./sim.js", import.meta.url);

export class Snippets {
  constructor() {
    this.model = loadModel();
    this.sim = fetch(SIM_URL).then((r) => (r.ok ? r.text() : "")).catch(() => "");
  }

  /**
   * Restituisce {title, file, startLine, code, language, highlight} oppure null.
   * key: chiave di EXTERNAL, "ours:<chiave di model.json>" o "sim:<nome funzione/classe>".
   */
  async get(key) {
    if (EXTERNAL[key]) return EXTERNAL[key];
    if (key.startsWith("ours:")) {
      const name = key.slice(5);
      const model = await this.model;
      const s = model?.snippets?.[name];
      if (!s) return null;
      return { title: OURS[name]?.title ?? name, file: s.file, startLine: s.start_line, code: s.code, language: "python", highlight: OURS[name]?.highlight ?? [] };
    }
    if (key.startsWith("sim:")) return extractExport(await this.sim, key.slice(4));
    return null;
  }
}

/**
 * Estrae `export function nome` o `export class Nome` dal sorgente, con il commento JSDoc che lo
 * precede e il numero di riga vero. Il blocco finisce alla prima riga "}" in colonna 0.
 */
export function extractExport(source, name) {
  const lines = source.split("\n");
  const start = lines.findIndex((l) => new RegExp(`^export (async )?(function|class) ${name}\\b`).test(l));
  if (start < 0) return null;
  let first = start;
  while (first > 0 && /^\s*(\/\*\*|\*|\*\/)/.test(lines[first - 1])) first--;
  let end = start;
  while (end < lines.length && lines[end] !== "}") end++;
  return {
    title: `Questo tab: ${name}()`,
    file: "webapp/js/tabs/modern-llm/sim.js",
    startLine: first + 1,
    code: lines.slice(first, end + 1).join("\n"),
    language: "javascript",
    highlight: [],
  };
}
