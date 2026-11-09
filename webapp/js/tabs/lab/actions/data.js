/**
 * I dati veri di un'azione (logica pura: niente DOM, niente three.js; testata con il motore vero).
 *
 * Il motore JS (js/engine/gpt.js) esegue il passaggio in avanti del run A allo step scelto sul prompt
 * dello studente e restituisce TUTTE le attivazioni (forward(ids, { trace: true })). Qui si ritagliano i
 * pezzi che ogni animazione mostra (il vettore di un token, le key di una head, i punteggi…) insieme ai
 * pesi veri (le viste del motore: gli stessi numeri della scena) e ai nomi PyTorch dei tensori, così la
 * scena sa su quali matrici lavorare.
 */
import { attentionOutput, attentionScores, causalMask, concatHeads, crossEntropy, layerNormSteps, linearForward, relu, softmaxRows } from "./math.js";
import { resolveToken } from "./settings.js";

/** Una riga t di una matrice (T × n) piatta, come Float32Array (vista, non copia). */
export function rowOf(matrix, t, n) {
  return matrix.subarray(t * n, (t + 1) * n);
}

/** I nomi PyTorch dei pesi di un blocco e di una head (gpt/block.py, gpt/attention.py, gpt/feedforward.py). */
export function blockNames(b, h = 0) {
  const p = `blocks.${b}`;
  return {
    ln1: { weight: `${p}.ln1.weight`, bias: `${p}.ln1.bias` },
    ln2: { weight: `${p}.ln2.weight`, bias: `${p}.ln2.bias` },
    key: `${p}.sa.heads.${h}.key.weight`,
    query: `${p}.sa.heads.${h}.query.weight`,
    value: `${p}.sa.heads.${h}.value.weight`,
    tril: `${p}.sa.heads.${h}.tril`,
    head: `${p}.sa.heads.${h}`,
    proj: { weight: `${p}.sa.proj.weight`, bias: `${p}.sa.proj.bias` },
    ffn1: { weight: `${p}.ffwd.net.0.weight`, bias: `${p}.ffwd.net.0.bias` },
    relu: `${p}.ffwd.net.1`,
    ffn2: { weight: `${p}.ffwd.net.2.weight`, bias: `${p}.ffwd.net.2.bias` },
    block: p,
  };
}

/** Pesi finali di uscita del modello. */
export const OUTPUT_NAMES = Object.freeze({ lnF: { weight: "ln_f.weight", bias: "ln_f.bias" }, lmHead: { weight: "lm_head.weight", bias: "lm_head.bias" } });

/**
 * Il passaggio in avanti del motore sul prompt, con tutto ciò che serve alle azioni.
 * @param {import("../../../engine/gpt.js").GPTEngine} engine con i pesi del run A allo step scelto
 * @param {number[]} ids il prompt in token (al più block_size)
 * @param {{token?: number|null, block?: number, head?: number}} [o]
 */
export function forwardData(engine, ids, { token = null, block = 0, head = 0 } = {}) {
  const cfg = engine.config;
  const { trace } = engine.forward(ids, { trace: true });
  const T = ids.length;
  const C = cfg.n_embd;
  const hs = cfg.head_size;
  const t = resolveToken(token, T);
  const b = Math.min(cfg.n_layer - 1, Math.max(0, block | 0));
  const h = Math.min(cfg.n_head - 1, Math.max(0, head | 0));
  const weight = (name) => engine.paramViews.get(name)?.data ?? null;
  return {
    engine,
    config: cfg,
    vocab: engine.vocab,
    ids,
    chars: ids.map((id) => engine.vocab[id]),
    trace,
    T,
    C,
    hs,
    V: cfg.vocab_size,
    hidden: trace.hidden,
    t,
    block: b,
    head: h,
    names: blockNames(b, h),
    weight,
    /** Il flusso residuo che ENTRA nel blocco b (x0 per il primo, l'uscita del blocco prima per gli altri). */
    blockInput: b === 0 ? trace.x0 : trace.blocks[b - 1].resid2,
  };
}

/**
 * La nn.Linear scelta per «x · Wᵀ riga per colonna»: nomi dei pesi, ingresso vero del token t (dalla
 * traccia), uscita del motore, il codice che la usa. `choice`: key | query | value | proj | ffn1 | ffn2 | lm_head.
 */
export function linearData(f, choice = "key") {
  const { trace, t, C, hs, names } = f;
  const blk = trace.blocks[f.block];
  const headTrace = blk.heads[f.head];
  const row = (m, n) => rowOf(m, t, n);
  const H4 = f.hidden;
  const table = {
    key: { weight: names.key, bias: null, input: row(blk.ln1, C), inputLabel: "x", inputText: "x = ln1(x)", output: row(headTrace.k, hs), outName: "k", code: "key" },
    query: { weight: names.query, bias: null, input: row(blk.ln1, C), inputLabel: "x", inputText: "x = ln1(x)", output: row(headTrace.q, hs), outName: "q", code: "query" },
    value: { weight: names.value, bias: null, input: row(blk.ln1, C), inputLabel: "x", inputText: "x = ln1(x)", output: row(headTrace.v, hs), outName: "v", code: "value" },
    proj: { weight: names.proj.weight, bias: names.proj.bias, input: row(blk.concat, C), inputLabel: "concat", inputText: "la concatenazione delle head", output: row(blk.proj, C), outName: "y", code: "proj" },
    ffn1: { weight: names.ffn1.weight, bias: names.ffn1.bias, input: row(blk.ln2, C), inputLabel: "x", inputText: "x = ln2(x)", output: row(blk.ffnPre, H4), outName: "h", code: "ffn1" },
    ffn2: { weight: names.ffn2.weight, bias: names.ffn2.bias, input: row(blk.ffnPost, H4), inputLabel: "ReLU(h)", inputText: "ReLU(h), i neuroni nascosti", output: row(blk.ffnOut, C), outName: "y", code: "ffn2" },
    lm_head: { weight: OUTPUT_NAMES.lmHead.weight, bias: OUTPUT_NAMES.lmHead.bias, input: row(trace.lnF, C), inputLabel: "x", inputText: "x = ln_f(x)", output: row(trace.logits, f.V), outName: "logits", code: "lmHead" },
  };
  const spec = table[choice] ?? table.key;
  const W = f.weight(spec.weight);
  const bias = spec.bias ? f.weight(spec.bias) : null;
  const inDim = spec.input.length;
  const outDim = spec.output.length;
  return { choice: table[choice] ? choice : "key", ...spec, W, b: bias, inDim, outDim, recomputed: linearForward(spec.input, W, bias, inDim, outDim) };
}

/** La self-attention della head scelta, con i punteggi di tutte le coppie (anche il futuro) e la softmax. */
export function attentionData(f) {
  const { trace, T, hs } = f;
  const blk = trace.blocks[f.block];
  const ht = blk.heads[f.head];
  const { dots, scaled, scale } = attentionScores(ht.q, ht.k, T, hs);
  const masked = causalMask(scaled, T);
  const att = softmaxRows(masked, T);
  const out = attentionOutput(att, ht.v, T, hs);
  const outs = blk.heads.map((x) => x.out);
  return {
    q: ht.q,
    k: ht.k,
    v: ht.v,
    dots,
    scaled,
    scale,
    masked,
    att,
    out,
    engine: { scores: ht.scores, att: ht.att, out: ht.out },
    concat: concatHeads(outs, T, hs),
    heads: outs,
  };
}

/** La LayerNorm scelta ("ln1" | "ln2" del blocco, "ln_f") sul token t: ingresso, passaggi e uscita del motore. */
export function layerNormData(f, which = "ln1") {
  const { trace, t, C } = f;
  const blk = trace.blocks[f.block];
  const spec =
    which === "ln2"
      ? { names: f.names.ln2, input: rowOf(blk.resid1, t, C), output: rowOf(blk.ln2, t, C), code: "residualFfn", label: `ln2 del blocco ${f.block}` }
      : which === "ln_f"
        ? { names: OUTPUT_NAMES.lnF, input: rowOf(trace.blocks.at(-1).resid2, t, C), output: rowOf(trace.lnF, t, C), code: "lnF", label: "ln_f, la LayerNorm finale" }
        : { names: f.names.ln1, input: rowOf(f.blockInput, t, C), output: rowOf(blk.ln1, t, C), code: "residualAttention", label: `ln1 del blocco ${f.block}` };
  const gamma = f.weight(spec.names.weight);
  const beta = f.weight(spec.names.bias);
  return { which, ...spec, gamma, beta, steps: layerNormSteps(spec.input, gamma, beta) };
}

/** La feed-forward del blocco sul token t: i due strati, la ReLU, il residuo (tutto dalla traccia + ricalcolato). */
export function ffnData(f) {
  const { trace, t, C } = f;
  const blk = trace.blocks[f.block];
  const H4 = f.hidden;
  const x = rowOf(blk.ln2, t, C);
  const W1 = f.weight(f.names.ffn1.weight);
  const b1 = f.weight(f.names.ffn1.bias);
  const W2 = f.weight(f.names.ffn2.weight);
  const b2 = f.weight(f.names.ffn2.bias);
  const pre = linearForward(x, W1, b1, C, H4);
  const r = relu(pre);
  const out = linearForward(r.post, W2, b2, H4, C);
  return {
    x,
    pre,
    post: r.post,
    off: r.off,
    on: r.on,
    out,
    residIn: rowOf(blk.resid1, t, C),
    resid: rowOf(blk.resid2, t, C),
    engine: { pre: rowOf(blk.ffnPre, t, H4), post: rowOf(blk.ffnPost, t, H4), out: rowOf(blk.ffnOut, t, C) },
  };
}

/** Logits e probabilità del token t (lm_head su ln_f), con il carattere scelto e quello vero del prompt. */
export function logitsData(f) {
  const { trace, t, C, V } = f;
  const h = rowOf(trace.lnF, t, C);
  const logits = rowOf(trace.logits, t, V);
  const next = t + 1 < f.ids.length ? f.ids[t + 1] : null;
  const ce = crossEntropy(logits, next ?? 0);
  return { h, logits, probs: ce.probs, next, nextLoss: next === null ? null : ce.loss, nextP: next === null ? null : ce.p };
}

/** Le misure di un run "come il GPT del corso" per i testi: 4·C, head_size, ecc. */
export function shapeWords(cfg) {
  return { C: cfg.n_embd, H: cfg.n_head, hs: cfg.n_embd / cfg.n_head, L: cfg.n_layer, V: cfg.vocab_size, T: cfg.block_size, hidden: 4 * cfg.n_embd };
}
