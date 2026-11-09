/**
 * Gruppi di parametri per i riepiloghi (confronto, grafici del training).
 *
 * Per il GPT del corso i gruppi sono quelli di gpt/recording.py (`group_of`): embedding dei
 * token, embedding delle posizioni, per ogni blocco attention (head + proj), feed-forward e le
 * due LayerNorm insieme, poi ln_f e lm_head. Un trace.json li contiene già (`group_params`):
 * quando c'è, vince lui (`groupsFromTrace`). Per un'architettura diversa si raggruppa per
 * modulo: `blocks.3.mlp.c_fc.weight` → `blocks.3.mlp`, `wte.weight` → `wte`.
 */

const BLOCK_PARTS = { sa: "attention", ffwd: "ffwd", ln1: "layernorm", ln2: "layernorm" };
const TOP_LEVEL = {
  token_embedding_table: "token_embedding",
  position_embedding_table: "position_embedding",
  ln_f: "ln_f",
  lm_head: "lm_head",
};

/** Gruppo di un parametro, con le stesse regole di gpt/recording.py per il GPT del corso. */
export function groupOf(name) {
  const parts = name.split(".");
  if (Object.hasOwn(TOP_LEVEL, parts[0])) return TOP_LEVEL[parts[0]];
  if (parts[0] === "blocks" && parts.length > 2 && Object.hasOwn(BLOCK_PARTS, parts[2])) return `blocks.${parts[1]}.${BLOCK_PARTS[parts[2]]}`;
  // Architettura sconosciuta: il sotto-modulo del blocco ripetuto (es. transformer.h.0.attn,
  // model.layers.3.mlp), oppure i primi due livelli (transformer.wte) se non ci sono blocchi numerati.
  const modules = parts.slice(0, -1);
  if (modules.length === 0) return name;
  const index = modules.findIndex((p) => /^\d+$/.test(p));
  return modules.slice(0, index > 0 ? index + 2 : 2).join(".");
}

/** Mappa nome → gruppo da trace.json (`group_params`), o null se il trace non li ha. */
export function groupsFromTrace(trace) {
  const groups = trace?.group_params;
  if (!groups || typeof groups !== "object") return null;
  const out = new Map();
  for (const [group, names] of Object.entries(groups)) {
    if (!Array.isArray(names)) continue;
    for (const n of names) if (typeof n === "string") out.set(n, group);
  }
  return out.size ? out : null;
}

/** Funzione nome → gruppo: quella del trace se c'è (per i nomi che conosce), altrimenti groupOf. */
export function grouper(trace = null) {
  const fromTrace = groupsFromTrace(trace);
  return fromTrace ? (name) => fromTrace.get(name) ?? groupOf(name) : groupOf;
}
