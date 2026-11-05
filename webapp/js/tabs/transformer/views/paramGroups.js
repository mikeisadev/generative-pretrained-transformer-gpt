/**
 * Raggruppa i parametri del modello per componente (logica pura, senza DOM: testata con node).
 *
 * I nomi sono quelli di `model.named_parameters()` in PyTorch (data/model.json → spec.params).
 * Ogni parametro appartiene a UN componente, identificato con l'id del registro della scena
 * (architecture.js): così le somme mostrate nei livelli del LOD, nella vista "Pesi" e nelle
 * tabelle coincidono per costruzione con il totale vero (209.729 per il modello del corso).
 */

const RULES = [
  [/^token_embedding_table\./, () => "tok_emb"],
  [/^position_embedding_table\./, () => "pos_emb"],
  [/^blocks\.(\d+)\.ln1\./, (m) => `blocks.${m[1]}.ln1`],
  [/^blocks\.(\d+)\.sa\.heads\.\d+\./, (m) => `blocks.${m[1]}.heads`],
  [/^blocks\.(\d+)\.sa\.proj\./, (m) => `blocks.${m[1]}.proj`],
  [/^blocks\.(\d+)\.ln2\./, (m) => `blocks.${m[1]}.ln2`],
  [/^blocks\.(\d+)\.ffwd\./, (m) => `blocks.${m[1]}.ffwd`],
  [/^ln_f\./, () => "ln_f"],
  [/^lm_head\./, () => "lm_head"],
];

/** Id del componente (registro della scena) a cui appartiene il parametro `name`, o null. */
export function componentOf(name) {
  for (const [re, id] of RULES) {
    const m = name.match(re);
    if (m) return id(m);
  }
  return null;
}

/** Blocco transformer (0…n_layer−1) di un parametro, oppure null se è fuori dai blocchi. */
export function layerOf(name) {
  const m = name.match(/^blocks\.(\d+)\./);
  return m ? Number(m[1]) : null;
}

/** Somma degli elementi dei parametri il cui nome soddisfa `test` (stringa = prefisso). */
export function countParams(params, test) {
  const match = typeof test === "string" ? (n) => n.startsWith(test) : test;
  return params.reduce((acc, p) => (match(p.name) ? acc + p.numel : acc), 0);
}

/**
 * Totale e tensori per componente.
 * @param {{name: string, shape: number[], numel: number}[]} params
 * @returns {Map<string, {id: string, count: number, tensors: object[]}>} in ordine di flusso
 */
export function componentTotals(params) {
  const out = new Map();
  for (const p of params) {
    const id = componentOf(p.name);
    if (!id) throw new Error(`parametro senza componente: ${p.name}`);
    if (!out.has(id)) out.set(id, { id, count: 0, tensors: [] });
    const entry = out.get(id);
    entry.count += p.numel;
    entry.tensors.push(p);
  }
  return out;
}

/**
 * Riepilogo "per grandi parti" usato da grafici e barre: embedding, ogni blocco diviso in
 * attention / feed-forward / LayerNorm, uscita. Le voci sommano esattamente al totale.
 */
export function partsSummary(params, nLayer) {
  const parts = [
    { id: "tok_emb", label: "Token embedding", short: "tok", count: countParams(params, "token_embedding_table.") },
    { id: "pos_emb", label: "Position embedding", short: "pos", count: countParams(params, "position_embedding_table.") },
  ];
  for (let l = 0; l < nLayer; l++) {
    const p = `blocks.${l}.`;
    parts.push(
      { id: `blocks.${l}.attention`, layer: l, label: `Blocco ${l} · attention`, short: `A${l}`, count: countParams(params, `${p}sa.`) },
      { id: `blocks.${l}.ffwd`, layer: l, label: `Blocco ${l} · feed-forward`, short: `F${l}`, count: countParams(params, `${p}ffwd.`) },
      { id: `blocks.${l}.layernorm`, layer: l, label: `Blocco ${l} · LayerNorm`, short: `N${l}`, count: countParams(params, (n) => n.startsWith(`${p}ln1.`) || n.startsWith(`${p}ln2.`)) },
    );
  }
  parts.push(
    { id: "ln_f", label: "LayerNorm finale", short: "ln_f", count: countParams(params, "ln_f.") },
    { id: "lm_head", label: "lm_head", short: "lm", count: countParams(params, "lm_head.") },
  );
  return parts;
}

/** "(65, 64)" */
export const shapeText = (shape) => `(${shape.join(", ")}${shape.length === 1 ? "," : ""})`;
