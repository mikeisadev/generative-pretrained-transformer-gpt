/**
 * Che cosa vogliono dire righe, colonne e celle di un tensore del GPT (logica pura, testata).
 *
 * Una matrice di numeri senza etichette non insegna niente: la riga 12 di token_embedding_table è "il
 * carattere «e»", la colonna 5 di una key è "la dimensione 5 di x dopo ln1", la colonna 37 di proj è "la
 * head 2, dimensione 5 della concatenazione". Le regole seguono i nomi del codice del corso (gpt/*.py) e
 * le convenzioni di PyTorch (nn.Linear (out, in): riga = uscita, colonna = ingresso); per un nome
 * sconosciuto restano "riga r / colonna c", senza inventare significati.
 *
 *   const m = tensorMeaning("blocks.0.sa.heads.2.key.weight", [16, 64], { vocab, config });
 *   m.rows.describe(3)  → "uscita 3 della key della head 2 (di 16)"
 *   m.cell(3, 5)        → "peso dall'ingresso 5 all'uscita 3: quanto la dimensione 5 di x (dopo ln1) conta per la key"
 */

/** Come si scrive un carattere del vocabolario in un'etichetta (spazio e a capo resterebbero invisibili). */
export const shownChar = (c) => (c === " " ? "␣" : c === "\n" ? "↵" : c === "\t" ? "⇥" : String(c ?? ""));

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const n = (v) => INT.format(v);

function axis(name, describe, label = (i) => String(i)) {
  return { name, describe, label };
}

/**
 * @param {string} name nome PyTorch del tensore (o del buffer, es. "blocks.0.sa.heads.0.tril")
 * @param {number[]} shape
 * @param {{vocab?: string[]|null, config?: object|null}} [o]
 * @returns {{kind: string, title: string, rows: object, cols: object, cell: (i: number, j: number) => string, formula: string}}
 *   per un tensore 1D `rows` descrive gli elementi e `cols` è null
 */
export function tensorMeaning(name, shape, { vocab = null, config = null } = {}) {
  const path = String(name ?? "");
  const parts = path.split(".");
  const last = parts.at(-1);
  const C = config?.n_embd;
  const T = config?.block_size;
  const head = /heads\.(\d+)\.(key|query|value)\.weight$/.exec(path);
  const block = /^blocks\.(\d+)\./.exec(path)?.[1];
  const where = block !== undefined ? ` (blocco ${block})` : "";
  const char = (i) => (vocab?.[i] !== undefined ? `«${shownChar(vocab[i])}»` : `token ${i}`);
  const charAxis = axis("carattere", (i) => `riga ${i}: il carattere ${char(i)}${vocab?.[i] !== undefined ? ` (token ${i})` : ""}`, (i) => (vocab?.[i] !== undefined ? shownChar(vocab[i]) : String(i)));
  const dims = (what) => axis("dimensione", (j) => `dimensione ${j} ${what}${C ? ` (di ${n(C)})` : ""}`);
  const isVector = shape?.length === 1;

  if (/(^|\.)(token_embedding_table|wte|tok_embeddings|embed_tokens)\.weight$/.test(path)) {
    return {
      kind: "embedding-token",
      title: "Tabella degli embedding dei token",
      rows: charAxis,
      cols: dims("del vettore del token"),
      cell: (i, j) => `il numero ${j} del vettore che rappresenta il carattere ${char(i)}`,
      formula: "x = tok_emb[token] + pos_emb[posizione]",
    };
  }
  if (/(^|\.)(position_embedding_table|wpe)\.weight$/.test(path)) {
    return {
      kind: "embedding-position",
      title: "Tabella degli embedding delle posizioni",
      rows: axis("posizione", (i) => `riga ${i}: la posizione ${i} nel contesto${T ? ` (di ${n(T)})` : ""}`),
      cols: dims("del vettore della posizione"),
      cell: (i, j) => `il numero ${j} del vettore che dice "sono alla posizione ${i}"`,
      formula: "x = tok_emb[token] + pos_emb[posizione]",
    };
  }
  if (head) {
    const [, h, kind] = head;
    const role = { key: "la key (che cosa contengo)", query: "la query (che cosa cerco)", value: "il value (che cosa comunico)" }[kind];
    return {
      kind: "head-projection",
      title: `Proiezione ${kind} della head ${h}${where}`,
      rows: axis("uscita", (d) => `uscita ${d} della ${kind} della head ${h}${shape?.[0] ? ` (di ${n(shape[0])})` : ""}`),
      cols: axis("ingresso", (j) => `ingresso ${j}: la dimensione ${j} di x dopo ln1${C ? ` (di ${n(C)})` : ""}`),
      cell: (d, j) => `peso dall'ingresso ${j} all'uscita ${d}: quanto la dimensione ${j} di x conta per ${role}`,
      formula: `${kind[0]} = x · W${kind[0]}ᵀ (nessun bias: bias=False)`,
    };
  }
  if (/\.tril$/.test(path)) {
    return {
      kind: "tril",
      title: "Maschera causale tril (buffer, non è un peso)",
      rows: axis("query", (i) => `riga ${i}: la query del token alla posizione ${i}`),
      cols: axis("key", (j) => `colonna ${j}: la key del token alla posizione ${j}`),
      cell: (i, j) => (j <= i ? `1: il token ${i} può guardare il token ${j} (è nel passato o è lui)` : `0: il token ${j} è nel futuro del token ${i}, mascherato con −∞`),
      formula: "wei.masked_fill(tril[:T, :T] == 0, −∞)",
    };
  }
  const hs = config?.n_head ? C / config.n_head : null;
  if (/\.sa\.proj\.weight$/.test(path)) {
    return {
      kind: "linear",
      title: `Proiezione dell'attention${where}`,
      rows: axis("uscita", (r) => `uscita ${r}: la dimensione ${r} che si somma al flusso residuo`),
      cols: axis("ingresso", (k) => (hs ? `ingresso ${k}: head ${Math.floor(k / hs)}, dimensione ${k % hs} della concatenazione` : `ingresso ${k} (concatenazione delle head)`)),
      cell: (r, k) => `quanto l'ingresso ${k} della concatenazione${hs ? ` (head ${Math.floor(k / hs)})` : ""} conta per l'uscita ${r}`,
      formula: "y = concat(head₀ … head₃) · Wᵀ + b",
    };
  }
  const ffn = /\.ffwd\.net\.(\d+)\.(weight|bias)$/.exec(path);
  if (ffn && ffn[1] === "0") {
    if (ffn[2] === "bias") return vector("bias", `Bias del primo strato della feed-forward${where}`, (i) => `neurone nascosto ${i}${shape?.[0] ? ` (di ${n(shape[0])})` : ""}`, "h = ReLU(x · W₁ᵀ + b₁)");
    return {
      kind: "linear",
      title: `Primo strato della feed-forward${where}`,
      rows: axis("neurone nascosto", (r) => `neurone nascosto ${r}${shape?.[0] ? ` (di ${n(shape[0])})` : ""}`),
      cols: axis("ingresso", (j) => `ingresso ${j}: la dimensione ${j} di x dopo ln2${C ? ` (di ${n(C)})` : ""}`),
      cell: (r, j) => `quanto la dimensione ${j} di x accende il neurone nascosto ${r}`,
      formula: "h = ReLU(x · W₁ᵀ + b₁)",
    };
  }
  if (ffn && ffn[1] === "2") {
    if (ffn[2] === "bias") return vector("bias", `Bias del secondo strato della feed-forward${where}`, (i) => `uscita ${i}: la dimensione ${i} che si somma al flusso residuo`, "y = h · W₂ᵀ + b₂");
    return {
      kind: "linear",
      title: `Secondo strato della feed-forward${where}`,
      rows: axis("uscita", (r) => `uscita ${r}: la dimensione ${r} che si somma al flusso residuo`),
      cols: axis("neurone nascosto", (k) => `neurone nascosto ${k} (dopo la ReLU)${shape?.[1] ? ` (di ${n(shape[1])})` : ""}`),
      cell: (r, k) => `quanto il neurone nascosto ${k} spinge la dimensione ${r} del flusso residuo`,
      formula: "y = h · W₂ᵀ + b₂",
    };
  }
  if (/(^|\.)(ln\w*|norm\w*|ln_f)\.(weight|bias)$/.test(path) && isVector) {
    const gamma = last === "weight";
    const ln = parts.at(-2);
    return vector(
      gamma ? "gamma" : "beta",
      `${gamma ? "γ (gamma)" : "β (beta)"} di ${ln}${where}`,
      (i) => `dimensione ${i} di x${C ? ` (di ${n(C)})` : ""}`,
      gamma ? "y = γ · (x − μ) / √(σ² + ε) + β: γ allarga o stringe la dimensione" : "y = γ · (x − μ) / √(σ² + ε) + β: β sposta la dimensione",
    );
  }
  if (/(^|\.)lm_head\.weight$/.test(path)) {
    return {
      kind: "linear",
      title: "lm_head: dai vettori ai punteggi dei caratteri",
      rows: axis("carattere", (v) => `riga ${v}: il punteggio (logit) del carattere ${char(v)}`, charAxis.label),
      cols: axis("ingresso", (j) => `ingresso ${j}: la dimensione ${j} di x dopo ln_f${C ? ` (di ${n(C)})` : ""}`),
      cell: (v, j) => `quanto la dimensione ${j} di x spinge il punteggio del carattere ${char(v)}`,
      formula: "logits = x · Wᵀ + b",
    };
  }
  if (/(^|\.)lm_head\.bias$/.test(path)) return vector("bias", "Bias di lm_head", (v) => `il carattere ${char(v)}: punteggio di partenza`, "logits = x · Wᵀ + b");
  if (last === "bias" && isVector) return vector("bias", `Bias di ${parts.slice(0, -1).join(".")}`, (i) => `uscita ${i}`, "y = x · Wᵀ + b");
  if (isVector) return vector("generic", path, (i) => `elemento ${i}`, "");
  return {
    kind: "generic",
    title: path,
    rows: axis("riga", (r) => `riga ${r} (in una nn.Linear: l'uscita ${r})`),
    cols: axis("colonna", (c) => `colonna ${c} (in una nn.Linear: l'ingresso ${c})`),
    cell: (r, c) => `il peso alla riga ${r}, colonna ${c}`,
    formula: last === "weight" ? "in una nn.Linear: y = x · Wᵀ + b" : "",
  };
}

function vector(kind, title, describe, formula) {
  return { kind, title, rows: axis("elemento", (i) => describe(i)), cols: null, cell: (i) => describe(i), formula };
}
