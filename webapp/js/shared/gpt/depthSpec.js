/**
 * La rete neurale "Orizzontale con profondità" del GPT del corso (logica pura, senza DOM né three.js): dalla
 * configurazione e dalla traccia vera del motore (engine.forward(ids, { trace: true })) alla `networkSpec` del
 * renderer del nucleo (js/core/neuralNet/, formato in ARCHITECTURE.md "Rete neurale orizzontale").
 *
 * Sta fuori dai tab perché la usano in due: il Transformer (vista Rete neurale, disposizione "Orizzontale con
 * profondità": tabs/transformer/views/neuronDepthSpec.js la riesporta con le parti sue, cioè link, quote del
 * cursore e corrispondenza con la disposizione "A righe") e il Laboratorio pesi (la rete del run A quando il
 * motore JS lo esegue: tabs/lab/network.js, che prima la importava dalla cartella del Transformer).
 * Test: tests/transformer-depth.test.mjs.
 *
 *   x  le sezioni nell'ordine del flusso: one-hot del token e della posizione → embedding → x; per OGNI
 *      blocco LayerNorm 1 → self-attention (key, value, query, uscita) → proiezione → + residuo →
 *      LayerNorm 2 → nascosto 4·C con ReLU → uscita FFN → + residuo; poi LayerNorm finale → logits → softmax
 *   y  i neuroni della sezione: gli stessi neuroni distinti della disposizione "A righe" del Transformer
 *      (tabs/transformer/views/neuronModel.js)
 *   z  le posizioni del prompt (fino a block_size), con le attivazioni VERE di tutte le posizioni
 *
 * Niente simulazioni: valori, pesi di attention e pesi delle nn.Linear vengono dal motore. I blocchi si
 * disegnano tutti (sono pochi e ognuno ha i suoi numeri): nessun "× N".
 *
 * Perché la self-attention è UNA sezione con quattro lastre (key, value, query, uscita): il renderer
 * disegna gli archi di attention dalla prima lastra della sezione (chi dà: key e value del token j)
 * all'ultima (chi riceve: l'uscita del token i, la media dei value pesata dall'attention). Così ogni arco
 * j → i è proprio il peso att[h][i][j] del motore, e la query (chi chiede) sta accanto all'uscita del suo
 * token. Le copie del vettore x a inizio riga della disposizione "A righe" qui non servono: il residuo è
 * un solo flusso da sinistra a destra.
 */
import { LINK_LIMITS } from "../../core/neuralNet/links.js";
import { softmax } from "../../engine/kernels.js";
import { shownChar } from "../weights/display.js";

/** Quota predefinita dei pesi più forti di ogni matrice disegnati sulla fetta scelta (2%). */
export const DEFAULT_DEPTH_FRACTION = 0.02;

const QKV_NAMES = ["query", "key", "value"]; // part 0, 1, 2 (l'ordine di qkvIndex della disposizione "A righe")

/**
 * Lastre della sezione di attention, da sinistra a destra: chi dà (key, value), chi chiede (query) e
 * chi riceve (uscita = colonne della concatenazione). `part`: 0 query, 1 key, 2 value (null: concat).
 */
export const ATTENTION_SLABS = Object.freeze([
  Object.freeze({ name: "key", part: 1 }),
  Object.freeze({ name: "value", part: 2 }),
  Object.freeze({ name: "query", part: 0 }),
  Object.freeze({ name: "uscita (concat)", part: null }),
]);

// Come si scrive un carattere del vocabolario (spazio, a capo, caratteri invisibili): shared/weights/display.js.
export { shownChar };

/** "LN 1 · blocco 2": l'etichetta della sezione con il blocco a cui appartiene. */
export function sectionTitle(s) {
  const block = /^b(\d+)\./.exec(s.id);
  return block ? `${s.label} · blocco ${block[1]}` : s.label;
}

const itNumber = (v, digits) => v.toLocaleString("it-IT", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const SUPERSCRIPT = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

/**
 * Un valore vero da leggere in un'etichetta: tre decimali; sotto 0,001 la notazione scientifica
 * ("3,2 · 10⁻⁷"), perché molte probabilità sono minuscole e "0,0000" direbbe zero dove zero non è.
 */
export function shownValue(v) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (v === 0 || a >= 0.01) return itNumber(v, 3);
  if (a >= 0.001) return itNumber(v, 4);
  let exp = Math.floor(Math.log10(a));
  let mantissa = v / 10 ** exp;
  if (Math.abs(Number(mantissa.toFixed(1))) >= 10) {
    mantissa /= 10;
    exp += 1;
  }
  return `${itNumber(mantissa, 1)} · 10${[...String(exp)].map((c) => SUPERSCRIPT[c]).join("")}`;
}

/**
 * Le righe dell'etichetta di un'unità (NetworkScene.describe) e della riga "Nella rete orizzontale" della
 * scheda: sezione e blocco, lastra e neurone, head, token, valore. Testo semplice (il prompt lo scrive lo
 * studente: chi lo mostra usa textContent). `names` sono i caratteri del prompt: la profondità
 * normalizzata dal nucleo li perde quando il prompt ha un solo token.
 */
export function unitDescription(info, names = null) {
  const s = info.section;
  const slab = s.slabs > 1 ? `${s.slabNames?.[info.slab] ?? `lastra ${info.slab}`} · ` : "";
  const name = s.neuronNames?.[info.neuron] ? ` «${s.neuronNames[info.neuron]}»` : "";
  const token = names?.[info.depth] ?? s.depth.names?.[info.depth] ?? "";
  return {
    title: sectionTitle(s),
    neuron: `${slab}neurone ${info.neuron}${name} di ${s.neurons}`,
    head: info.head ?? null,
    token: `token ${info.depth}${token ? ` «${token}»` : ""} (di ${s.depth.count})`,
    value: `valore ${shownValue(info.value)}`,
  };
}

/** Neuroni per posizione: quelli distinti della disposizione "A righe" e le copie di x che qui non ci sono. */
export function depthCounts(config, T = config.block_size) {
  const { vocab_size: V, block_size: B, n_embd: C, n_layer: L } = config;
  const input = V + B + 2 * C + C; // one-hot del token e della posizione, due tabelle di embedding, x
  const perBlock = 14 * C; // LN1 C + attention 4C (key, value, query, uscita) + proj C + "+" C + LN2 C + nascosto 4C + uscita C + "+" C
  const output = C + 2 * V; // LayerNorm finale, logits, softmax
  const perPosition = input + L * perBlock + output;
  const copies = (L + 1) * C; // "x in ingresso al blocco" e "x in uscita" della disposizione a righe
  return { perPosition, units: perPosition * T, copies, rowsCount: perPosition + copies, perBlock };
}

/**
 * Le sezioni da sinistra a destra (descrittori di soli dati: id, etichette, forma, colonna della disposizione
 * "A righe" del Transformer, neuronModel.js).
 * Gli id coincidono con quelli delle colonne di neuronModel dove la sezione è una colonna sola; "emb"
 * (tabelle del token e della posizione) e "b<l>.attn" (q, k, v e concat) ne uniscono più d'una.
 */
export function depthSections(config) {
  const { vocab_size: V, block_size: B, n_embd: C, n_head: H, n_layer: L } = config;
  const hs = C / H;
  const F = 4 * C;
  const list = [
    { id: "tok_onehot", label: "Token (one-hot)", sub: `${V} voci del vocabolario`, role: "input", neurons: V, activation: "onehot", text: `Un neurone per carattere del vocabolario (${V}): per ogni token del prompt se ne accende uno solo, quello del suo carattere.` },
    { id: "pos_onehot", label: "Posizione (one-hot)", sub: `${B} posizioni possibili`, role: "input", neurons: B, activation: "onehot", text: `Un neurone per posizione possibile (block_size = ${B}): il token t accende il neurone t. In profondità si vede una diagonale.` },
    { id: "emb", label: "Embedding", sub: "tabella del token · della posizione", role: "embedding", neurons: C, slabs: 2, slabNames: ["token", "posizione"], text: `Due tabelle di pesi: il carattere sceglie una riga di token_embedding_table, la posizione una riga di position_embedding_table (${C} numeri ciascuna). Lookup: per ogni token un solo neurone acceso dà tutto il vettore.` },
    { id: "x0", label: "x = token + posizione", role: "add", neurons: C, text: "Somma numero per numero delle due righe: è l'inizio del flusso residuo (residual stream) che attraversa tutti i blocchi." },
  ];
  for (let l = 0; l < L; l++) {
    const p = `b${l}.`;
    list.push(
      { id: `${p}ln1`, layer: l, label: "LN 1", sub: "LayerNorm", role: "norm", neurons: C, text: `Blocco ${l}. Ogni token normalizza il SUO vettore (media 0, varianza 1) e lo scala con γ e β: un calcolo per posizione, nessun arco in profondità.` },
      {
        id: `${p}attn`, layer: l, label: "Self-attention", sub: `key · value · query · uscita (${H} head × ${hs})`, role: "attention", neurons: C, slabs: ATTENTION_SLABS.length, slabNames: ATTENTION_SLABS.map((s) => s.name), heads: H,
        text: `Blocco ${l}. Da sinistra: key e value (cosa ogni token offre), query (cosa cerca) e uscita delle ${H} head affiancate (concat). Gli archi tra le posizioni sono i pesi di attention veri: dal token j (key, value) all'uscita del token i, un colore per head, più luce = più peso. Maschera causale: un token guarda solo se stesso e quelli prima.`,
      },
      { id: `${p}proj`, layer: l, label: "Proj", sub: "proiezione", role: "attention", neurons: C, text: `Blocco ${l}. Una nn.Linear (${C} → ${C}) rimescola le uscite delle head, token per token.` },
      { id: `${p}resid1`, layer: l, label: "+", sub: "x + attention", role: "add", neurons: C, text: `Blocco ${l}. Connessione residua: al vettore che era entrato nel blocco si somma l'uscita dell'attention (peso fisso 1).` },
      { id: `${p}ln2`, layer: l, label: "LN 2", sub: "LayerNorm", role: "norm", neurons: C, text: `Blocco ${l}. Seconda LayerNorm, di nuovo token per token.` },
      { id: `${p}hidden`, layer: l, label: "Nascosto", sub: `${F} · ReLU`, role: "mlp", neurons: F, activation: "positive", text: `Blocco ${l}. Feed-forward: ${F} neuroni nascosti per token, ognuno somma ${C} ingressi pesati più un bias; la ReLU azzera i negativi (le sfere grigie).` },
      { id: `${p}ffnOut`, layer: l, label: "Uscita FFN", sub: `${F} → ${C}`, role: "mlp", neurons: C, text: `Blocco ${l}. La seconda nn.Linear riporta ${F} numeri a ${C}, token per token.` },
      { id: `${p}resid2`, layer: l, label: "+", sub: "x + feed-forward", role: "add", neurons: C, text: `Blocco ${l}. Seconda connessione residua: è l'uscita del blocco${l + 1 < L ? `, che entra nel blocco ${l + 1}` : ""}.` },
    );
  }
  list.push(
    { id: "ln_f", label: "LN finale", sub: "LayerNorm", role: "norm", neurons: C, text: "LayerNorm finale, token per token." },
    { id: "logits", label: "Logits", sub: `lm_head · ${V} voci`, role: "output", neurons: V, activation: "signed", text: `lm_head: un punteggio per ognuno dei ${V} caratteri, per ogni posizione. Ogni posizione predice il carattere che viene DOPO di lei.` },
    { id: "probs", label: "Softmax", sub: "probabilità", role: "output", neurons: V, activation: "probs", text: "La softmax trasforma i punteggi di ogni posizione in probabilità che sommano a 1. Il prossimo carattere del prompt si legge all'ultima posizione (quella davanti)." },
  );
  return list.map((s) => ({ slabs: 1, ...s }));
}

/** Id delle sezioni nell'ordine del flusso. */
export const depthSectionIds = (config) => depthSections(config).map((s) => s.id);

/**
 * Attivazioni vere di ogni sezione nell'ordine dei `values` del renderer: [lastra][neurone][posizione]
 * (values[(lastra × N + n) × T + t]). La traccia del motore è invece [posizione][neurone].
 * @returns {Map<string, Float32Array>}
 */
export function depthSectionValues(config, trace) {
  const { vocab_size: V, block_size: B, n_embd: C, n_head: H } = config;
  const T = trace.T;
  const hs = C / H;
  const F = 4 * C;
  const out = new Map();
  /** Copia una matrice (T × width) della traccia nella lastra `slab` di una sezione da `width` neuroni. */
  const put = (target, slab, src, width) => {
    for (let t = 0; t < T; t++) for (let n = 0; n < width; n++) target[(slab * width + n) * T + t] = src[t * width + n];
  };
  const section = (id, size) => {
    const values = new Float32Array(size * T);
    out.set(id, values);
    return values;
  };
  const tok = section("tok_onehot", V);
  const pos = section("pos_onehot", B);
  for (let t = 0; t < T; t++) {
    tok[trace.ids[t] * T + t] = 1;
    pos[t * T + t] = 1;
  }
  const emb = section("emb", 2 * C);
  put(emb, 0, trace.tokEmb, C);
  put(emb, 1, trace.posEmb, C);
  put(section("x0", C), 0, trace.x0, C);
  trace.blocks.forEach((block, l) => {
    const p = `b${l}.`;
    put(section(`${p}ln1`, C), 0, block.ln1, C);
    const attn = section(`${p}attn`, ATTENTION_SLABS.length * C);
    ATTENTION_SLABS.forEach(({ part }, slab) => {
      if (part === null) return put(attn, slab, block.concat, C);
      const key = "qkv"[part];
      for (let h = 0; h < H; h++) {
        const src = block.heads[h][key]; // (T × hs)
        for (let t = 0; t < T; t++) for (let d = 0; d < hs; d++) attn[(slab * C + h * hs + d) * T + t] = src[t * hs + d];
      }
    });
    put(section(`${p}proj`, C), 0, block.proj, C);
    put(section(`${p}resid1`, C), 0, block.resid1, C);
    put(section(`${p}ln2`, C), 0, block.ln2, C);
    put(section(`${p}hidden`, F), 0, block.ffnPost, F);
    put(section(`${p}ffnOut`, C), 0, block.ffnOut, C);
    put(section(`${p}resid2`, C), 0, block.resid2, C);
  });
  put(section("ln_f", C), 0, trace.lnF, C);
  put(section("logits", V), 0, trace.logits, V);
  const probs = section("probs", V);
  const p = new Float32Array(V); // la softmax di una posizione (il kernel del motore: stessi numeri, una riga alla volta)
  for (let t = 0; t < T; t++) {
    softmax(p, 0, trace.logits, t * V, V);
    for (let v = 0; v < V; v++) probs[v * T + t] = p[v];
  }
  return out;
}

/** Pesi di attention di un blocco nel formato del renderer: weights[(h × T + query) × T + key]. */
export function attentionWeights(trace, l) {
  const T = trace.T;
  const heads = trace.blocks[l].heads;
  const weights = new Float32Array(heads.length * T * T);
  heads.forEach((head, h) => weights.set(head.att, h * T * T)); // head.att è già (query × key), righe che sommano a 1
  return weights;
}

/**
 * Ordine dei pesi di ogni tensore dal più forte al più debole (|w| decrescente, a parità l'indice):
 * dipende solo dai pesi, quindi chi chiama può tenerlo tra un prompt e l'altro.
 * @param {(name: string) => Float32Array} param
 * @returns {Map<string, Uint32Array>}
 */
export function weightOrders(config, param) {
  const orders = new Map();
  for (const { name } of linearTensors(config)) {
    const w = param(name);
    const idx = new Uint32Array(w.length);
    for (let i = 0; i < idx.length; i++) idx[i] = i;
    idx.sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]) || a - b);
    orders.set(name, idx);
  }
  return orders;
}

/**
 * Le nn.Linear del modello come collegamenti tra sezioni: per ogni tensore (righe = uscite, colonne =
 * ingressi, come in PyTorch) da quale sezione a quale, e dove cade ogni riga e colonna nella fetta
 * (indice = lastra × neuroni + neurone, il formato delle coppie "explicit").
 */
export function linearTensors(config) {
  const { vocab_size: V, n_embd: C, n_head: H, n_layer: L } = config;
  const hs = C / H;
  const F = 4 * C;
  const list = [];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    for (const [slab, { part }] of ATTENTION_SLABS.entries()) {
      if (part === null) continue;
      for (let h = 0; h < H; h++) {
        list.push({ name: `${p}sa.heads.${h}.${QKV_NAMES[part]}.weight`, from: `b${l}.ln1`, to: `b${l}.attn`, rows: hs, cols: C, toIndex: (d) => slab * C + h * hs + d, fromIndex: (i) => i, label: `W${QKV_NAMES[part][0]} della head ${h}` });
      }
    }
    list.push({ name: `${p}sa.proj.weight`, from: `b${l}.attn`, to: `b${l}.proj`, rows: C, cols: C, toIndex: (j) => j, fromIndex: (k) => (ATTENTION_SLABS.length - 1) * C + k, label: "proiezione" });
    list.push({ name: `${p}ffwd.net.0.weight`, from: `b${l}.ln2`, to: `b${l}.hidden`, rows: F, cols: C, toIndex: (j) => j, fromIndex: (i) => i, label: "feed-forward, primo strato" });
    list.push({ name: `${p}ffwd.net.2.weight`, from: `b${l}.hidden`, to: `b${l}.ffnOut`, rows: C, cols: F, toIndex: (j) => j, fromIndex: (i) => i, label: "feed-forward, secondo strato" });
  }
  list.push({ name: "lm_head.weight", from: "ln_f", to: "logits", rows: V, cols: C, toIndex: (v) => v, fromIndex: (i) => i, label: "lm_head" });
  return list;
}

/** Quanti pesi di un tensore da `numel` si disegnano con la quota `fraction` (almeno uno). */
export const shownWeights = (numel, fraction) => Math.min(numel, Math.max(1, Math.round(numel * fraction)));

/**
 * Coppie "explicit" di una nn.Linear: i pesi più forti (quota `fraction` del tensore), con il loro
 * valore vero. W[r × cols + c] collega l'ingresso c all'uscita r.
 */
export function strongestPairs(tensor, W, order, fraction) {
  const k = shownWeights(W.length, fraction);
  const from = new Int32Array(k);
  const to = new Int32Array(k);
  const weight = new Float32Array(k);
  for (let s = 0; s < k; s++) {
    const e = order[s];
    const r = Math.floor(e / tensor.cols);
    from[s] = tensor.fromIndex(e - r * tensor.cols);
    to[s] = tensor.toIndex(r);
    weight[s] = W[e];
  }
  return { from, to, weight };
}

/** γ di una LayerNorm come coppie uno a uno (neurone i → neurone i, peso γ_i): tutti, sono pochi. */
function gammaPairs(gamma) {
  const from = Int32Array.from(gamma, (_, i) => i);
  return { from, to: from.slice(), weight: Float32Array.from(gamma) };
}

/**
 * La networkSpec della rete vera per un prompt.
 * @param {object} o
 * @param {object} o.config configurazione del modello (vocab_size, block_size, n_embd, n_head, n_layer)
 * @param {string[]} o.vocab token id → carattere
 * @param {object} o.trace traccia di engine.forward(ids, { trace: true })
 * @param {(name: string) => Float32Array} o.param pesi veri per nome PyTorch (engine.paramViews)
 * @param {number} [o.fraction] quota dei pesi più forti di ogni matrice disegnata sulla fetta scelta
 * @param {Map<string, Uint32Array>} [o.orders] weightOrders già calcolato (si riusa tra un prompt e l'altro)
 * @param {Map<string, Float32Array>} [o.values] depthSectionValues già calcolato
 */
export function buildDepthSpec({ config, vocab, trace, param, fraction = DEFAULT_DEPTH_FRACTION, orders = null, values = null }) {
  const { n_embd: C, n_head: H, n_layer: L } = config;
  const T = trace.T;
  const counts = depthCounts(config, T);
  const names = trace.ids.map((id) => shownChar(vocab[id]));
  const vocabNames = vocab.map(shownChar);
  const sectionValues = values ?? depthSectionValues(config, trace);
  const sections = depthSections(config).map(({ layer, ...s }) => ({
    ...s,
    depth: true,
    values: sectionValues.get(s.id),
    ...(s.id === "tok_onehot" || s.id === "logits" || s.id === "probs" ? { neuronNames: vocabNames } : {}),
  }));

  const links = [
    { from: "tok_onehot", to: "emb", pattern: "lookup", toSlab: 0, label: "lookup nella tabella dei token" },
    { from: "pos_onehot", to: "emb", pattern: "lookup", toSlab: 1, label: "lookup nella tabella delle posizioni" },
    { from: "emb", to: "x0", pattern: "identity", fromSlab: 0, label: "somma: embedding del token" },
    { from: "emb", to: "x0", pattern: "identity", fromSlab: 1, label: "somma: embedding della posizione" },
  ];
  const tensors = linearTensors(config);
  const orderOf = orders ?? weightOrders(config, param);
  const explicit = (from, to) => {
    for (const t of tensors) {
      if (t.from !== from || t.to !== to) continue;
      // numel: quanti pesi ha la matrice (metadato del tab, il nucleo lo ignora): serve a sliceCounts
      links.push({ from, to, pattern: "explicit", pairs: strongestPairs(t, param(t.name), orderOf.get(t.name), fraction), label: `${t.label} (${t.name})`, numel: t.rows * t.cols });
    }
  };
  const gamma = (from, to, name) => links.push({ from, to, pattern: "explicit", pairs: gammaPairs(param(name)), label: `γ della LayerNorm (${name})` });
  let input = "x0";
  for (let l = 0; l < L; l++) {
    const p = `b${l}.`;
    links.push({ from: input, to: `${p}ln1`, pattern: "identity", label: "LayerNorm: ogni token per conto suo" });
    gamma(input, `${p}ln1`, `blocks.${l}.ln1.weight`);
    explicit(`${p}ln1`, `${p}attn`);
    if (T > 1) links.push({ within: `${p}attn`, pattern: "attention", causal: true, heads: H, weights: attentionWeights(trace, l), label: "attention causale: pesi veri del motore" });
    explicit(`${p}attn`, `${p}proj`);
    links.push({ from: `${p}proj`, to: `${p}resid1`, pattern: "identity", label: "somma: uscita dell'attention" });
    links.push({ from: input, to: `${p}resid1`, pattern: "residual", label: "connessione residua (x + attention)" });
    links.push({ from: `${p}resid1`, to: `${p}ln2`, pattern: "identity", label: "LayerNorm: ogni token per conto suo" });
    gamma(`${p}resid1`, `${p}ln2`, `blocks.${l}.ln2.weight`);
    explicit(`${p}ln2`, `${p}hidden`);
    explicit(`${p}hidden`, `${p}ffnOut`);
    links.push({ from: `${p}ffnOut`, to: `${p}resid2`, pattern: "identity", label: "somma: uscita della feed-forward" });
    links.push({ from: `${p}resid1`, to: `${p}resid2`, pattern: "residual", label: "connessione residua (x + feed-forward)" });
    input = `${p}resid2`;
  }
  links.push({ from: input, to: "ln_f", pattern: "identity", label: "LayerNorm: ogni token per conto suo" });
  gamma(input, "ln_f", "ln_f.weight");
  explicit("ln_f", "logits");
  links.push({ from: "logits", to: "probs", pattern: "identity", label: "softmax (ogni probabilità dipende da tutti i logits della sua posizione)" });

  const fmtInt = (n) => n.toLocaleString("it-IT");
  return {
    title: "Rete neurale vera del GPT del corso, in orizzontale",
    intro: `È la rete VERA del modello, non una simulazione: ogni sfera è un neurone con il valore calcolato dal motore per il tuo prompt. Le stesse sezioni della disposizione a righe stanno una dopo l'altra da sinistra a destra, con la profondità vera: i ${T} token del prompt uno dietro l'altro. Di lato leggi il flusso; girando la camera vedi lo spessore, con l'attention che collega i token tra loro.`,
    scale: `Nessuna riduzione: 1 sfera = 1 neurone vero. ${fmtInt(counts.perPosition)} neuroni per token × ${T} token = ${fmtInt(counts.units)} neuroni, tutti con il valore calcolato dal motore per questo prompt.`,
    note: depthNote(fraction),
    seed: 1,
    axes: { x: "flusso: dall'ingresso all'uscita", y: `neuroni della sezione (${C} per il vettore x)`, z: `posizioni del prompt (${T} token)` },
    depth: { count: T, label: "token", names },
    sections,
    links,
    groups: [
      { label: "Ingresso", from: "tok_onehot", to: "x0", text: "Il carattere e la sua posizione diventano il vettore x." },
      ...Array.from({ length: L }, (_, l) => ({ label: `Blocco ${l}`, from: `b${l}.ln1`, to: `b${l}.resid2`, text: `Blocco transformer ${l}: attention (i token si scambiano informazioni) e feed-forward (ogni token per conto suo), ognuno con la sua connessione residua.` })),
      { label: "Uscita", from: "ln_f", to: "probs", text: "LayerNorm finale, punteggi e probabilità del prossimo carattere per ogni posizione." },
    ],
    sources: [
      { title: "Karpathy, ng-video-lecture (gpt.py)", url: "https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py", note: "Il modello del corso: embedding, blocchi pre-norm, lm_head." },
      { title: "Vaswani et al. (2017), Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", note: "Self-attention con maschera causale nel decoder." },
      { title: "PyTorch: nn.Linear e nn.LayerNorm", url: "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html", note: "y = xWᵀ + b; LayerNorm per posizione con γ e β." },
    ],
  };
}

const fmtPct = (f) => `${(f * 100).toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`;

/**
 * La nota di onestà della rete per la quota `fraction` dei pesi della fetta (il tab la riscrive quando la
 * quota cambia senza ricostruire la rete: vedi DepthLayout.applyFraction).
 */
export function depthNote(fraction) {
  return `I collegamenti colorati sono i pesi veri delle nn.Linear sulla fetta scelta: i più forti (${fmtPct(fraction)} di ogni matrice) più tutti i γ delle LayerNorm. Gli archi di attention sono i pesi veri del motore (per ogni query e head i ${LINK_LIMITS.arcKeysPerQuery} più forti, sopra il ${LINK_LIMITS.arcMinWeight * 100}%). Linee grigie: collegamenti fissi, non appresi, uno a uno e token per token (le somme, con peso 1; LayerNorm e softmax, dove il neurone dipende anche dagli altri neuroni del suo token).`;
}

/**
 * Quanti pesi disegnare di ogni collegamento della specifica con la quota `fraction`: delle nn.Linear i
 * round(quota × pesi della matrice) più forti (shownWeights, come quando la rete si costruisce con quella
 * quota), dei γ delle LayerNorm tutti (Infinity), degli altri collegamenti niente da dire (Infinity).
 * Indice = indice del collegamento in spec.links (lo stesso di model.net.links).
 */
export function sliceCounts(spec, fraction) {
  return spec.links.map((l) => (l.pattern === "explicit" && l.numel ? shownWeights(l.numel, fraction) : Infinity));
}
