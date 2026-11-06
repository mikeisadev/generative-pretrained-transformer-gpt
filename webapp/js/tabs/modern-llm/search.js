/**
 * Indice della ricerca globale per il tab "LLM moderni" (contratto in ARCHITECTURE.md, "Ricerca
 * globale, link diretti e glossario").
 *
 * Solo dati leggeri: viene caricato anche se il tab non è mai stato aperto, quindi niente three.js
 * né moduli della scena. Gli id coincidono con quelli che index.js capisce in goto():
 *   step:<id del Percorso>       (content.js, stessi id del Menu)
 *   action:<id dell'azione>      (actionCatalog.js, stessi id di player.register)
 *   view:macro | blocks | detail (livelli dello zoom) e view:<id di una regione> (regionData.js)
 */
import { ACTION_CATALOG } from "./actionCatalog.js";
import { PATH } from "./content.js";
import { REGIONS, ROWS } from "./regionData.js";

/**
 * Sinonimi e parole che uno studente potrebbe cercare per arrivare a un passo (italiano e inglese).
 * Titolo e testo del passo sono già cercati: qui solo ciò che lì manca o è scritto diverso.
 */
const STEP_KEYWORDS = {
  "idea.ponte": ["LLM moderni", "modern LLM", "Llama 3", "differenze", "confronto", "nostro GPT", "decoder", "ossatura"],
  "idea.chiusi": ["modelli chiusi", "closed models", "GPT-4", "GPT-5", "Claude", "Gemini", "open weights", "pesi aperti", "open source", "fonti"],
  "arch.bpe": ["byte pair encoding", "tokenizer", "tokenizzazione", "tokenization", "sotto-parole", "subword", "merge", "UTF-8", "vocabolario", "tiktoken"],
  "arch.rmsnorm": ["RMSNorm", "LayerNorm", "normalizzazione", "normalization", "layer norm", "root mean square"],
  "arch.rope": ["rotary position embedding", "rotary", "positional encoding", "embedding posizionale", "posizione", "rotazione", "theta"],
  "arch.gqa": ["grouped-query attention", "multi-query attention", "MHA", "MQA", "GQA", "head", "key", "value", "KV cache", "repeat_kv"],
  "arch.swiglu": ["SwiGLU", "SiLU", "Swish", "GLU", "ReLU", "MLP", "feed-forward", "FFN", "attivazione", "activation"],
  "arch.moe": ["MoE", "mixture of experts", "esperti", "experts", "router", "gating", "Mixtral", "DeepSeek-V3", "Switch Transformer", "sparse"],
  "inf.kv": ["KV cache", "cache", "key value", "inferenza", "inference", "generazione", "memoria", "MLA"],
  "inf.flash": ["FlashAttention", "flash attention", "sliding window", "finestra scorrevole", "contesto lungo", "long context", "128K", "GPU", "memoria"],
  "inf.spec": ["speculative decoding", "decodifica speculativa", "draft", "bozza", "modello piccolo", "velocità", "accelerare"],
  "train.scaling": ["scaling laws", "leggi di scala", "Chinchilla", "compute optimal", "FLOP", "parametri", "dati", "budget"],
  "train.pipeline": ["pre-training", "pretraining", "SFT", "supervised fine-tuning", "instruction tuning", "post-training", "fasi"],
  "train.align": ["RLHF", "DPO", "Constitutional AI", "allineamento", "alignment", "reward model", "preferenze", "preferences", "PPO"],
  "reas.cot": ["chain of thought", "catena di pensiero", "CoT", "ragionamento", "reasoning", "test-time compute", "self-consistency", "o1"],
  "reas.rl": ["reinforcement learning", "RL", "GRPO", "RLVR", "ricompense verificabili", "verifiable rewards", "DeepSeek-R1", "reward"],
  "reas.agent": ["agenti", "agents", "tool use", "function calling", "strumenti", "tools", "MCP", "loop"],
  "reas.coding": ["coding", "codice", "programmazione", "FIM", "fill in the middle", "SWE-bench", "test", "Codex"],
  "beyond.multi": ["multimodale", "multimodal", "vision", "immagini", "images", "LLaVA", "ViT", "CLIP", "patch"],
  "beyond.mesh": ["3D", "mesh", "OBJ", "LLaMA-Mesh", "Blender", "modellazione 3D", "vertici"],
  "beyond.emergent": ["emergent abilities", "capacità emergenti", "emergenza", "mirage", "miraggio", "metrica"],
  "beyond.interp": ["interpretability", "interpretabilità", "induction head", "sparse autoencoder", "SAE", "feature", "mechanistic interpretability"],
};

const steps = PATH.flatMap((section) =>
  section.children.map((item) => ({
    id: item.id,
    kind: "step",
    title: item.title,
    text: item.text,
    keywords: [section.title, ...(STEP_KEYWORDS[item.id] ?? [])],
  })),
);

const actions = ACTION_CATALOG.map((a) => ({
  id: a.id,
  kind: "action",
  title: a.title,
  text: `Animazione passo per passo. ${a.summary}`,
  keywords: [a.category, "animazione", "animation", ...a.keywords],
}));

/** I livelli dello zoom (stessi id del LODController del tab). */
const levels = [
  {
    id: "macro",
    kind: "view",
    title: "Panoramica: dalla raccolta dei dati alla risposta",
    text: "Lo schema più astratto: Dati → Pre-training → Post-training → Modello → Inferenza, con ragionamento e strumenti.",
    keywords: ["panoramica", "overview", "zoom out", "schema", "pipeline", "diagramma a blocchi", "vista d'insieme"],
  },
  {
    id: "blocks",
    kind: "view",
    title: "Tutti i meccanismi (livello Blocchi)",
    text: "La mappa dei 16 meccanismi: un rettangolo per meccanismo con una frase su cosa fa.",
    keywords: ["blocchi", "blocks", "mappa", "map", "griglia", "meccanismi", "vista d'insieme", "elenco"],
  },
  {
    id: "detail",
    kind: "view",
    title: "Dettaglio: il meccanismo più vicino",
    text: "Lo zoom più vicino: i componenti interni della regione che stai guardando.",
    keywords: ["dettaglio", "detail", "zoom in", "da vicino", "componenti"],
  },
];

/** Una vista per ogni regione della scena (pannello "Viste"): ci si arriva anche camminando. */
const regions = REGIONS.map((r) => ({
  id: r.id,
  kind: "view",
  title: `Regione: ${r.title}`,
  text: `${r.sub}. ${r.block}`,
  keywords: [ROWS[r.row].title, "regione", "meccanismo", ...ACTION_CATALOG.filter((a) => a.region === r.id).flatMap((a) => a.keywords.slice(0, 4))],
}));

export const searchIndex = [...steps, ...actions, ...levels, ...regions];
