/**
 * Modello della vista "Rete neurale" (logica pura, senza DOM né three.js: testata con node).
 *
 * Il GPT del corso disegnato come rete neurale classica: colonne di neuroni (uno per numero
 * di un vettore di attivazioni) e archi (uno per peso di una matrice). Una riga per blocco,
 * dal basso verso l'alto come nella vista Architettura:
 *
 *   riga 0      one-hot del token (65) e della posizione (32) → embedding (64 + 64) → x (64)
 *   riga 1…L    x → LayerNorm 1 → Q/K/V (4 head × 3 × 16) → [attention] → concat (64) → proj (64)
 *               → + residuo (64) → LayerNorm 2 → nascosto (256, ReLU) → uscita FFN (64) → + residuo
 *   riga L+1    x → LayerNorm finale (64) → logits (65) → softmax (65)
 *
 * Archi con peso (pesi veri da data/weights.json tramite il motore JS):
 *   matrici delle nn.Linear e delle nn.Embedding (206.976) + γ delle LayerNorm, uno a uno (576).
 * I bias e i β delle LayerNorm sono parametri DEL NEURONE (2.177): 206.976 + 576 + 2.177 = 209.729.
 * L'attention NON ha archi con peso: i suoi coefficienti dipendono dai dati (softmax(q·k/√hs)).
 * Le connessioni residue e le somme sono collegamenti fissi (peso 1, non appreso).
 */

export const ROW_GAP = 18; // distanza verticale tra le righe
export const COLUMN_HEIGHT = 13; // altezza di una colonna di neuroni

/** Ruolo didattico delle colonne (legenda: input layer, hidden, output layer). */
export const ROLE = { input: "input", hidden: "hidden", output: "output", copy: "copy" };

/**
 * Posizioni verticali di n neuroni tra yTop e yBottom, con spazi extra tra i gruppi.
 * groups = [{size, gap}] dal più piccolo al più grande: dopo ogni `size` neuroni si aggiungono `gap` posti vuoti.
 */
export function slotPositions(n, yTop, yBottom, groups = []) {
  const slots = new Float64Array(n);
  let pos = 0;
  for (let i = 0; i < n; i++) {
    if (i > 0) for (const g of groups) if (i % g.size === 0) pos += g.gap;
    slots[i] = pos;
    pos += 1;
  }
  const span = Math.max(1e-9, slots[n - 1] || 1);
  const step = (yTop - yBottom) / span;
  return { y: Array.from(slots, (s) => yTop - s * step), spacing: step };
}

/**
 * Costruisce colonne e posizioni di tutti i neuroni.
 * @param {{vocab_size:number, block_size:number, n_embd:number, n_head:number, n_layer:number}} config
 */
export function buildNeuronLayout(config) {
  const { vocab_size: V, block_size: T, n_embd: C, n_head: H, n_layer: L } = config;
  const hs = C / H;
  const F = 4 * C;
  const top = COLUMN_HEIGHT / 2;
  const columns = [];
  const rows = [];
  const add = (row, o) => {
    const col = { row, sub: 1, groups: [], yTop: top, yBottom: -top, role: ROLE.hidden, ...o };
    columns.push(col);
    return col;
  };

  // --- riga 0: ingresso
  add(0, { id: "tok_onehot", key: "tok_onehot", n: V, x: 0, yTop: top, yBottom: 0.5, role: ROLE.input, title: "One-hot del token", short: "One-hot", unit: "token" });
  add(0, { id: "pos_onehot", key: "pos_onehot", n: T, x: 0, yTop: -0.5, yBottom: -top, role: ROLE.input, title: "One-hot della posizione", short: "Posizione", unit: "posizione" });
  add(0, { id: "tok_emb", key: "tok_emb", n: C, x: 4, yTop: top, yBottom: 0.5, title: "Embedding del token", short: "Tok. emb." });
  add(0, { id: "pos_emb", key: "pos_emb", n: C, x: 4, yTop: -0.5, yBottom: -top, title: "Embedding della posizione", short: "Pos. emb." });
  add(0, { id: "x0", key: "x0", n: C, x: 8, title: "x = token + posizione", short: "x" });
  rows.push({ index: 0, title: "Ingresso ed embedding", width: 8 });

  // --- righe dei blocchi
  const qkvGroups = [{ size: hs, gap: 4 }, { size: C, gap: 14 }];
  for (let l = 0; l < L; l++) {
    const r = l + 1;
    const p = `b${l}.`;
    add(r, { id: `${p}x`, key: "x", layer: l, n: C, x: 0, role: ROLE.copy, title: `x in ingresso al blocco ${l}`, short: "x" });
    add(r, { id: `${p}ln1`, key: "ln1", layer: l, n: C, x: 3, title: `Blocco ${l} · LayerNorm 1`, short: "LN 1" });
    add(r, { id: `${p}qkv`, key: "qkv", layer: l, n: 3 * C, x: 7, groups: qkvGroups, title: `Blocco ${l} · query, key, value`, short: "Q K V" });
    add(r, { id: `${p}ctx`, key: "ctx", layer: l, n: 0, x: 10.6, title: `Blocco ${l} · attention`, short: "Attention" });
    add(r, { id: `${p}concat`, key: "concat", layer: l, n: C, x: 14, groups: [{ size: hs, gap: 3 }], title: `Blocco ${l} · uscite delle head (concat)`, short: "Concat" });
    add(r, { id: `${p}proj`, key: "proj", layer: l, n: C, x: 17.5, title: `Blocco ${l} · proiezione`, short: "Proj" });
    add(r, { id: `${p}resid1`, key: "resid1", layer: l, n: C, x: 20.5, title: `Blocco ${l} · x + attention`, short: "+" });
    add(r, { id: `${p}ln2`, key: "ln2", layer: l, n: C, x: 23.5, title: `Blocco ${l} · LayerNorm 2`, short: "LN 2" });
    add(r, { id: `${p}hidden`, key: "hidden", layer: l, n: F, sub: F / C, x: 27.5, title: `Blocco ${l} · feed-forward nascosto (ReLU)`, short: "Nascosto" });
    add(r, { id: `${p}ffnOut`, key: "ffnOut", layer: l, n: C, x: 31.5, title: `Blocco ${l} · uscita della feed-forward`, short: "Uscita FFN" });
    add(r, { id: `${p}resid2`, key: "resid2", layer: l, n: C, x: 34.5, title: `Blocco ${l} · x + feed-forward`, short: "+" });
    rows.push({ index: r, title: `Blocco ${l}`, width: 34.5, layer: l });
  }

  // --- uscita
  const r = L + 1;
  add(r, { id: "out.x", key: "x", n: C, x: 0, role: ROLE.copy, title: "x in uscita dall'ultimo blocco", short: "x" });
  add(r, { id: "ln_f", key: "ln_f", n: C, x: 3, title: "LayerNorm finale", short: "LN finale" });
  add(r, { id: "logits", key: "logits", n: V, x: 7, role: ROLE.output, title: "Logits (lm_head)", short: "Logits" });
  add(r, { id: "probs", key: "probs", n: V, x: 10.5, role: ROLE.output, title: "Softmax: probabilità", short: "Softmax" });
  rows.push({ index: r, title: "Uscita", width: 10.5 });

  // --- posizioni nel mondo (la vista è centrata nell'origine)
  for (const row of rows) row.y = (row.index - (rows.length - 1) / 2) * ROW_GAP;
  let count = 0;
  for (const col of columns) {
    const row = rows[col.row];
    col.x -= row.width / 2;
    col.y = row.y;
    col.offset = count;
    count += col.n;
  }
  const positions = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const columnOf = new Uint16Array(count);
  columns.forEach((col, ci) => {
    if (!col.n) return;
    const perSub = col.n / col.sub;
    const { y, spacing } = slotPositions(perSub, col.yTop, col.yBottom, col.groups);
    col.spacing = spacing;
    col.radius = Math.min(0.08, spacing * 0.36);
    for (let i = 0; i < col.n; i++) {
      const s = Math.floor(i / perSub);
      const k = col.offset + i;
      positions[3 * k] = col.x + (col.sub > 1 ? (s - (col.sub - 1) / 2) * 0.42 : 0);
      positions[3 * k + 1] = col.y + y[i % perSub];
      positions[3 * k + 2] = 0;
      radii[k] = col.radius;
      columnOf[k] = ci;
    }
  });
  const byId = new Map(columns.map((c) => [c.id, c]));
  return { config, columns, rows, byId, count, positions, radii, columnOf, headSize: hs, hidden: F };
}

/** Sottoinsieme di Q/K/V: indice i della colonna qkv → { part: 0 q | 1 k | 2 v, head, d }. */
export function qkvIndex(i, C, hs) {
  return { part: Math.floor(i / C), head: Math.floor((i % C) / hs), d: i % hs };
}
const QKV_NAMES = ["query", "key", "value"];

/**
 * Tutti gli archi con peso. `param(name)` restituisce il Float32Array (riga per riga) del tensore.
 * Ogni arco ha: from, to (indici di neurone), weight, matrix (indice in `matrices`).
 * `order` elenca gli archi dal più forte al più debole PER MATRICE (rango di |w| normalizzato):
 * i primi k archi di `order` sono "il k/N più forte" di ogni matrice, non solo della più grande.
 */
export function buildEdges(layout, param) {
  const { config, byId } = layout;
  const { n_embd: C, n_layer: L, n_head: H } = config;
  const hs = layout.headSize;
  const F = layout.hidden;
  const matrices = [];
  const lists = [];
  /** Aggiunge una matrice di archi: per ogni (i in ingresso, j in uscita) il peso è w(i, j). */
  const dense = (name, fromCol, toCol, nIn, nOut, w, kind = "matrix", toIndex = (j) => j) => {
    const m = { name, kind, from: fromCol.id, to: toCol.id, count: nIn * nOut, start: 0 };
    matrices.push(m);
    lists.push({ m, fromCol, toCol, nIn, nOut, w, toIndex });
  };
  const oneToOne = (name, fromCol, toCol, n) => {
    const g = param(name);
    dense(name, fromCol, toCol, 1, n, null, "gamma");
    lists.at(-1).gamma = g;
  };

  const E = param("token_embedding_table.weight");
  const P = param("position_embedding_table.weight");
  dense("token_embedding_table.weight", byId.get("tok_onehot"), byId.get("tok_emb"), config.vocab_size, C, (i, j) => E[i * C + j]);
  dense("position_embedding_table.weight", byId.get("pos_onehot"), byId.get("pos_emb"), config.block_size, C, (i, j) => P[i * C + j]);
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    const c = (k) => byId.get(`b${l}.${k}`);
    oneToOne(`${p}ln1.weight`, c("x"), c("ln1"), C);
    for (let part = 0; part < 3; part++) {
      for (let h = 0; h < H; h++) {
        const W = param(`${p}sa.heads.${h}.${QKV_NAMES[part]}.weight`); // (hs, C)
        dense(`${p}sa.heads.${h}.${QKV_NAMES[part]}.weight`, c("ln1"), c("qkv"), C, hs, (i, d) => W[d * C + i], "matrix", (d) => part * C + h * hs + d);
      }
    }
    const Wp = param(`${p}sa.proj.weight`);
    dense(`${p}sa.proj.weight`, c("concat"), c("proj"), C, C, (i, j) => Wp[j * C + i]);
    oneToOne(`${p}ln2.weight`, c("resid1"), c("ln2"), C);
    const W1 = param(`${p}ffwd.net.0.weight`); // (F, C)
    dense(`${p}ffwd.net.0.weight`, c("ln2"), c("hidden"), C, F, (i, j) => W1[j * C + i]);
    const W2 = param(`${p}ffwd.net.2.weight`); // (C, F)
    dense(`${p}ffwd.net.2.weight`, c("hidden"), c("ffnOut"), F, C, (i, j) => W2[j * F + i]);
  }
  oneToOne("ln_f.weight", byId.get("out.x"), byId.get("ln_f"), C);
  const Wl = param("lm_head.weight");
  dense("lm_head.weight", byId.get("ln_f"), byId.get("logits"), C, config.vocab_size, (i, v) => Wl[v * C + i]);

  const count = matrices.reduce((a, m) => a + m.count, 0);
  const from = new Uint32Array(count);
  const to = new Uint32Array(count);
  const weight = new Float32Array(count);
  const matrix = new Uint16Array(count);
  const strength = new Float32Array(count); // |w| / max|w| della matrice: per colore e trasparenza
  const BUCKETS = 1024;
  const bucket = new Uint16Array(count);
  let e = 0;
  lists.forEach((entry, mi) => {
    const { m, fromCol, toCol, nIn, nOut, w, gamma, toIndex } = entry;
    m.start = e;
    let maxAbs = 0;
    const start = e;
    if (gamma) {
      for (let j = 0; j < nOut; j++, e++) {
        from[e] = fromCol.offset + j;
        to[e] = toCol.offset + j;
        weight[e] = gamma[j];
      }
    } else {
      for (let j = 0; j < nOut; j++) {
        for (let i = 0; i < nIn; i++, e++) {
          from[e] = fromCol.offset + i;
          to[e] = toCol.offset + toIndex(j);
          weight[e] = w(i, j);
        }
      }
    }
    for (let k = start; k < e; k++) {
      matrix[k] = mi;
      maxAbs = Math.max(maxAbs, Math.abs(weight[k]));
    }
    m.maxAbs = maxAbs;
    // Rango di |w| dentro la matrice → secchio 0 (più forte) … BUCKETS−1 (più debole).
    const idx = Array.from({ length: e - start }, (_, k) => start + k).sort((a, b) => Math.abs(weight[b]) - Math.abs(weight[a]));
    idx.forEach((k, rank) => {
      // I γ delle LayerNorm (uno a uno, 576 in tutto) si vedono sempre: sono il solo "arco" di quei neuroni.
      bucket[k] = gamma ? 0 : Math.min(BUCKETS - 1, Math.floor((rank / idx.length) * BUCKETS));
      strength[k] = maxAbs > 0 ? Math.abs(weight[k]) / maxAbs : 0;
    });
  });
  // Ordinamento per secchio (counting sort, stabile): O(N).
  const starts = new Uint32Array(BUCKETS + 1);
  for (let k = 0; k < count; k++) starts[bucket[k] + 1]++;
  for (let b = 0; b < BUCKETS; b++) starts[b + 1] += starts[b];
  const order = new Uint32Array(count);
  for (let k = 0; k < count; k++) order[starts[bucket[k]]++] = k;
  return { count, from, to, weight, matrix, strength, order, matrices };
}

/** Softmax numericamente stabile di un vettore (nuovo Float32Array). */
export function softmax(z) {
  let max = -Infinity;
  for (const v of z) max = Math.max(max, v);
  const out = new Float32Array(z.length);
  let sum = 0;
  for (let i = 0; i < z.length; i++) sum += out[i] = Math.exp(z[i] - max);
  for (let i = 0; i < z.length; i++) out[i] /= sum;
  return out;
}

/**
 * Attivazione di ogni neurone alla posizione t, dal trace di engine.forward(ids, {trace: true}).
 * @returns {Float32Array} un valore per neurone (stesso ordine delle posizioni)
 */
export function neuronValues(layout, trace, t) {
  const { columns, count, config } = layout;
  const { n_embd: C } = config;
  const hs = layout.headSize;
  const F = layout.hidden;
  const V = config.vocab_size;
  const values = new Float32Array(count);
  const row = (arr, width) => arr.subarray(t * width, (t + 1) * width);
  const probs = softmax(row(trace.logits, V));
  for (const col of columns) {
    if (!col.n) continue;
    const l = col.layer;
    const block = l !== undefined ? trace.blocks[l] : null;
    let src = null;
    switch (col.id === "out.x" ? "out.x" : col.key) {
      case "tok_onehot":
        values[col.offset + trace.ids[t]] = 1;
        continue;
      case "pos_onehot":
        values[col.offset + t] = 1;
        continue;
      case "tok_emb": src = row(trace.tokEmb, C); break;
      case "pos_emb": src = row(trace.posEmb, C); break;
      case "x0": src = row(trace.x0, C); break;
      case "x": src = l === 0 ? row(trace.x0, C) : row(trace.blocks[l - 1].resid2, C); break;
      case "out.x": src = row(trace.blocks.at(-1).resid2, C); break;
      case "ln1": src = row(block.ln1, C); break;
      case "qkv":
        for (let i = 0; i < col.n; i++) {
          const { part, head, d } = qkvIndex(i, C, hs);
          values[col.offset + i] = block.heads[head]["qkv"[part]][t * hs + d];
        }
        continue;
      case "concat": src = row(block.concat, C); break;
      case "proj": src = row(block.proj, C); break;
      case "resid1": src = row(block.resid1, C); break;
      case "ln2": src = row(block.ln2, C); break;
      case "hidden": src = row(block.ffnPost, F); break;
      case "ffnOut": src = row(block.ffnOut, C); break;
      case "resid2": src = row(block.resid2, C); break;
      case "ln_f": src = row(trace.lnF, C); break;
      case "logits": src = row(trace.logits, V); break;
      case "probs": src = probs; break;
      default: continue;
    }
    values.set(src, col.offset);
  }
  return values;
}

/** Media e varianza (biased, come nn.LayerNorm) di un vettore. */
export function meanVar(x) {
  let mean = 0;
  for (const v of x) mean += v;
  mean /= x.length;
  let variance = 0;
  for (const v of x) variance += (v - mean) ** 2;
  return { mean, variance: variance / x.length };
}

const LN_EPS = 1e-5;

/**
 * Spiegazione del calcolo di un neurone con i numeri veri.
 * @returns {{kind, column, index, label, value, params, terms?, rest?, bias?, sum?, ...}}
 *   kind: "onehot" | "lookup" | "add" | "layernorm" | "linear" | "attention" | "softmax"
 */
export function explainNeuron(layout, neuron, trace, t, param, vocab) {
  const { config } = layout;
  const { n_embd: C } = config;
  const hs = layout.headSize;
  const F = layout.hidden;
  const V = config.vocab_size;
  const col = layout.columns[layout.columnOf[neuron]];
  const i = neuron - col.offset;
  const values = neuronValues(layout, trace, t);
  const valueOf = (id, k) => values[layout.byId.get(id).offset + k];
  const vec = (id) => values.subarray(layout.byId.get(id).offset, layout.byId.get(id).offset + layout.byId.get(id).n);
  const l = col.layer;
  const b = (k) => `b${l}.${k}`;
  const base = { column: col, index: i, value: values[neuron], label: col.title };
  /** Somma pesata con i contributi ordinati per |w·x|. */
  const linear = (inputId, w, bias, extra = {}) => {
    const x = vec(inputId);
    const terms = Array.from(x, (xi, k) => ({ k, w: w(k), x: xi, product: w(k) * xi }));
    const sum = terms.reduce((a, s) => a + s.product, 0) + (bias ?? 0);
    terms.sort((a, c) => Math.abs(c.product) - Math.abs(a.product));
    return { ...base, kind: "linear", input: layout.byId.get(inputId), terms, bias, sum, params: x.length + (bias === undefined ? 0 : 1), ...extra };
  };
  const layerNorm = (inputId, prefix) => {
    const x = vec(inputId);
    const { mean, variance } = meanVar(x);
    const gamma = param(`${prefix}.weight`)[i];
    const beta = param(`${prefix}.bias`)[i];
    const xhat = (x[i] - mean) / Math.sqrt(variance + LN_EPS);
    return { ...base, kind: "layernorm", input: layout.byId.get(inputId), x: x[i], mean, variance, eps: LN_EPS, xhat, gamma, beta, sum: gamma * xhat + beta, params: 2 };
  };
  switch (col.id === "out.x" ? "out.x" : col.key) {
    case "tok_onehot":
      return { ...base, kind: "onehot", token: vocab[i], active: trace.ids[t] === i, params: 0 };
    case "pos_onehot":
      return { ...base, kind: "onehot", position: i, active: i === t, params: 0 };
    case "tok_emb": {
      const id = trace.ids[t];
      return { ...base, kind: "lookup", table: "token_embedding_table.weight", row: id, rowLabel: vocab[id], sum: param("token_embedding_table.weight")[id * C + i], params: V };
    }
    case "pos_emb":
      return { ...base, kind: "lookup", table: "position_embedding_table.weight", row: t, rowLabel: `posizione ${t}`, sum: param("position_embedding_table.weight")[t * C + i], params: config.block_size };
    case "x0":
      return { ...base, kind: "add", parts: [["tok_emb", valueOf("tok_emb", i)], ["pos_emb", valueOf("pos_emb", i)]], sum: valueOf("tok_emb", i) + valueOf("pos_emb", i), params: 0 };
    case "x":
    case "out.x":
      return { ...base, kind: "copy", sum: values[neuron], params: 0 };
    case "ln1":
      return layerNorm(b("x"), `blocks.${l}.ln1`);
    case "ln2":
      return layerNorm(b("resid1"), `blocks.${l}.ln2`);
    case "ln_f":
      return layerNorm("out.x", "ln_f");
    case "qkv": {
      const { part, head, d } = qkvIndex(i, C, hs);
      const W = param(`blocks.${l}.sa.heads.${head}.${QKV_NAMES[part]}.weight`);
      return linear(b("ln1"), (k) => W[d * C + k], undefined, { part: QKV_NAMES[part], head, d, tensor: `blocks.${l}.sa.heads.${head}.${QKV_NAMES[part]}.weight` });
    }
    case "concat": {
      const head = Math.floor(i / hs);
      const d = i % hs;
      const h = trace.blocks[l].heads[head];
      const T = trace.T;
      const terms = [];
      for (let s = 0; s <= t; s++) terms.push({ s, token: vocab[trace.ids[s]], att: h.att[t * T + s], v: h.v[s * hs + d], product: h.att[t * T + s] * h.v[s * hs + d], score: h.scores[t * T + s] });
      const sum = terms.reduce((a, x) => a + x.product, 0);
      terms.sort((a, c) => c.att - a.att);
      return { ...base, kind: "attention", head, d, terms, sum, params: 0 };
    }
    case "proj": {
      const W = param(`blocks.${l}.sa.proj.weight`);
      return linear(b("concat"), (k) => W[i * C + k], param(`blocks.${l}.sa.proj.bias`)[i], { tensor: `blocks.${l}.sa.proj.weight` });
    }
    case "hidden": {
      const W = param(`blocks.${l}.ffwd.net.0.weight`);
      const out = linear(b("ln2"), (k) => W[i * C + k], param(`blocks.${l}.ffwd.net.0.bias`)[i], { tensor: `blocks.${l}.ffwd.net.0.weight`, relu: true });
      out.post = Math.max(0, out.sum);
      return out;
    }
    case "ffnOut": {
      const W = param(`blocks.${l}.ffwd.net.2.weight`);
      return linear(b("hidden"), (k) => W[i * F + k], param(`blocks.${l}.ffwd.net.2.bias`)[i], { tensor: `blocks.${l}.ffwd.net.2.weight` });
    }
    case "resid1":
      return { ...base, kind: "add", parts: [["x", valueOf(b("x"), i)], ["attention (proj)", valueOf(b("proj"), i)]], sum: valueOf(b("x"), i) + valueOf(b("proj"), i), params: 0 };
    case "resid2":
      return { ...base, kind: "add", parts: [["x", valueOf(b("resid1"), i)], ["feed-forward", valueOf(b("ffnOut"), i)]], sum: valueOf(b("resid1"), i) + valueOf(b("ffnOut"), i), params: 0 };
    case "logits": {
      const W = param("lm_head.weight");
      return linear("ln_f", (k) => W[i * C + k], param("lm_head.bias")[i], { tensor: "lm_head.weight", token: vocab[i] });
    }
    case "probs": {
      const z = vec("logits");
      let max = -Infinity;
      for (const v of z) max = Math.max(max, v);
      let denom = 0;
      for (const v of z) denom += Math.exp(v - max);
      return { ...base, kind: "softmax", token: vocab[i], z: z[i], max, denom, sum: Math.exp(z[i] - max) / denom, params: 0 };
    }
    default:
      return { ...base, kind: "copy", sum: values[neuron], params: 0 };
  }
}

/** Numero di parametri per tipo: archi delle matrici, γ, parametri dei neuroni (bias e β). */
export function parameterBreakdown(edges, params) {
  let matrix = 0;
  let gamma = 0;
  for (const m of edges.matrices) {
    if (m.kind === "gamma") gamma += m.count;
    else matrix += m.count;
  }
  const neuron = params.filter((p) => p.name.endsWith(".bias")).reduce((a, p) => a + p.numel, 0);
  return { matrix, gamma, neuron, total: matrix + gamma + neuron };
}
