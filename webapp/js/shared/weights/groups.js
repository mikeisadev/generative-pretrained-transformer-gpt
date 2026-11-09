/**
 * Gruppi di parametri per i riepiloghi (confronto, grafici del training).
 *
 * Per il GPT del corso i gruppi sono quelli di gpt/recording.py (`group_of`): embedding dei
 * token, embedding delle posizioni, per ogni blocco attention (head + proj), feed-forward e le
 * due LayerNorm insieme, poi ln_f e lm_head. Un trace.json li contiene già (`group_params`):
 * quando c'è, vince lui (`groupsFromTrace`). Per un'architettura diversa si raggruppa per
 * modulo: `blocks.3.mlp.c_fc.weight` → `blocks.3.mlp`, `wte.weight` → `wte`, e con un solo blocco
 * riusato `block.sa.heads.0.key.weight` → `block.sa` (lo 0 è una head, non un blocco).
 */

const BLOCK_PARTS = { sa: "attention", ffwd: "ffwd", ln1: "layernorm", ln2: "layernorm" };
const TOP_LEVEL = {
  token_embedding_table: "token_embedding",
  position_embedding_table: "position_embedding",
  ln_f: "ln_f",
  lm_head: "lm_head",
};
// Il contenitore dei blocchi sta vicino alla radice: il suo indice è al più il terzo livello del nome
// (blocks.0, transformer.h.3, model.layers.0). Lo stesso numero è in gpt/recording.py.
const BLOCK_INDEX_MAX_LEVEL = 2;

/** Gruppo di un parametro, con le stesse regole di gpt/recording.py per il GPT del corso. */
export function groupOf(name) {
  const parts = name.split(".");
  if (Object.hasOwn(TOP_LEVEL, parts[0])) return TOP_LEVEL[parts[0]];
  if (parts[0] === "blocks" && parts.length > 2 && Object.hasOwn(BLOCK_PARTS, parts[2])) return `blocks.${parts[1]}.${BLOCK_PARTS[parts[2]]}`;
  // Architettura sconosciuta: il sotto-modulo del blocco ripetuto (es. transformer.h.0.attn,
  // model.layers.3.mlp), oppure i primi due livelli (transformer.wte, block.sa) se non ci sono blocchi numerati.
  const modules = parts.slice(0, -1);
  if (modules.length === 0) return name;
  const index = blockIndex(modules);
  return modules.slice(0, index === -1 ? 2 : index + 2).join(".");
}

/**
 * La posizione dell'indice del blocco ripetuto nei moduli di un nome, o -1 (`_block_index` di
 * gpt/recording.py): il primo livello di sole cifre, se è vicino alla radice e non è l'ultimo modulo.
 * Più in basso un indice numera le parti di un modulo (le head di block.sa.heads.0, gli strati di
 * block.ffwd.net.0); l'ultimo modulo numerato (block.mlp.0.weight) è uno strato di un nn.Sequential.
 */
function blockIndex(modules) {
  const index = modules.findIndex((p) => /^\d+$/.test(p));
  return index !== -1 && index <= BLOCK_INDEX_MAX_LEVEL && index < modules.length - 1 ? index : -1;
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
