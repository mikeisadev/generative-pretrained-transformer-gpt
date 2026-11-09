/**
 * Indice della ricerca globale per il tab "LLM moderni" (contratto in ARCHITECTURE.md, "Ricerca
 * globale, link diretti e glossario").
 *
 * Solo dati leggeri: viene caricato anche se il tab non è mai stato aperto, quindi niente three.js
 * né moduli della scena. Gli id coincidono con quelli che index.js capisce in goto():
 *   step:<id del Percorso>       (content.js: PATH = vista d'insieme + architettura completa)
 *   action:<id dell'azione>      (actionCatalog.js e full/catalog.js, stessi id di player.register)
 *   view:macro | blocks | detail (livelli dello zoom) e view:<id di una regione> (regionData.js)
 *   view:overview | full | full-<…>   sottotab e viste dell'architettura completa (full/catalog.js)
 *   view:rete | cervello         rappresentazioni del nucleo (Rete neurale e Cervello, per tutto il tab)
 */
import { ACTION_CATALOG } from "./actionCatalog.js";
import { PATH } from "./content.js";
import { FULL_ACTIONS, FULL_LEVELS, FULL_SECTIONS } from "./full/catalog.js";
import { MODELS, PRIMARY_MODEL_ID, MODEL_BY_ID } from "./full/models/index.js";
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
  // architettura completa (DeepSeek-V4.1-Flash)
  "full-intro": ["DeepSeek-V4.1-Flash", "DeepSeek V4.1 Flash", "architettura completa", "modello aperto", "open weights", "config.json", "DeepInfer", "ricostruzione"],
  "full-zoom": ["segnaposto", "16 meccanismi", "Panoramica", "Blocchi", "Dettaglio", "zoom", "vista d'insieme"],
  "full-data": ["45T token", "pre-training", "dati", "token per parametro", "Chinchilla", "YaRN", "contesto lungo", "1M"],
  "full-post": ["SFT", "RL", "GRPO", "OPD", "distillazione on-policy", "post-training", "specialisti", "sottorete"],
  "full-weights": ["checkpoint", "safetensors", "96.085 tensori", "510 GB", "shard", "MXFP4", "FP8", "BF16", "FP32", "precisione", "dove sono i pesi"],
  "full-tokens": ["tokenizer", "BPE", "129.280", "embedding", "token speciali", "think", "DSML", "mHC", "4 flussi"],
  "full-stack": ["layer", "n-Transformer", "blocco ripetuto", "× N", "ripetizioni", "espandi", "srotola", "40 layer", "compressione"],
  "full-ced": ["encoder-decoder causale", "CED", "YOCO", "prefill", "decode", "H20", "8B attivi", "16B attivi"],
  "full-kv": ["KV cache compressa", "890 byte", "FP4", "CSA2", "riuso", "437", "MLA", "memoria"],
  "full-attention": ["CSA2", "MQA", "latente condiviso", "lightning indexer", "top-512", "finestra", "SWA", "attention sink", "RoPE"],
  "full-moe": ["MoE", "384 esperti", "router", "sqrtsoftplus", "top-6", "esperto condiviso", "SwiGLU", "bilanciamento"],
  "full-mhc": ["mHC", "hyper-connections", "Sinkhorn", "residuo", "Engram", "n-gram", "hash", "memoria condizionale"],
  "full-dspark": ["DSpark", "MTP", "multi-token prediction", "decodifica speculativa", "deployment", "EPLB", "expert parallel", "GFLOP"],
  "full-emerge": ["capacità emergenti", "emergent abilities", "Schaeffer", "induction head", "grokking", "leggi di scala", "RL"],
};

const steps = PATH.flatMap((section) =>
  section.children.map((item) => ({
    id: item.id,
    kind: "step",
    title: item.title,
    text: item.text,
    keywords: [section.title, ...(section.subtab === "full" ? ["architettura completa"] : []), ...(STEP_KEYWORDS[item.id] ?? [])],
  })),
);

const actions = [
  ...ACTION_CATALOG.map((a) => ({
    id: a.id,
    kind: "action",
    title: a.title,
    text: `Animazione passo per passo. ${a.summary}`,
    keywords: [a.category, "animazione", "animation", ...a.keywords],
  })),
  ...FULL_ACTIONS.map((a) => ({
    id: a.id,
    kind: "action",
    title: a.title,
    text: `Architettura completa, animazione passo per passo. ${a.summary}`,
    keywords: [a.category, "animazione", "architettura completa", "DeepSeek-V4.1-Flash", ...a.keywords],
  })),
];

/** I livelli dello zoom della vista d'insieme (stessi id del LODController del tab). */
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

// --- architettura completa (sottotab) ------------------------------------------------------

const primary = MODEL_BY_ID.get(PRIMARY_MODEL_ID);

const subtabs = [
  {
    id: "overview",
    kind: "view",
    title: "Vista d'insieme tecnologie LLM moderni",
    text: "I 16 meccanismi dei modelli di oggi, uno per regione, con le loro animazioni.",
    keywords: ["vista d'insieme", "sottotab", "big picture", "16 meccanismi", "tecnologie"],
  },
  {
    id: "full",
    kind: "view",
    title: "Architettura LLM moderni completa",
    text: `Un modello aperto vero smontato pezzo per pezzo: ${primary.name}, con Kimi K3.1 Pro e gpt-oss-120b come alternative.`,
    keywords: ["architettura completa", "sottotab", "DeepSeek-V4.1-Flash", "Kimi K3.1 Pro", "gpt-oss", "modello di frontiera", "open weights", "pesi aperti"],
  },
];

const fullLevels = FULL_LEVELS.map((l) => ({ id: l.id, kind: "view", title: `Architettura completa · ${l.title}`, text: l.text, keywords: ["architettura completa", "zoom", l.level] }));

const fullSections = FULL_SECTIONS.map((s) => ({ id: s.id, kind: "view", title: `Architettura completa · ${s.title}`, text: `Inquadra il pezzo nell'architettura di ${primary.name}.`, keywords: ["architettura completa", ...s.keywords] }));

const fullModels = MODELS.map((m) => ({
  id: `full-model-${m.id}`,
  kind: "view",
  title: `Modello: ${m.name}`,
  text: `${m.role}. ${m.headline}.`,
  keywords: ["modello", "architettura completa", m.org, m.short, ...(m.id === PRIMARY_MODEL_ID ? ["V4.1", "DeepSeek"] : [])],
}));

/** Dove vive ogni meccanismo della vista d'insieme nel modello principale. */
const fullMechanisms = REGIONS.map((r) => {
  const mech = primary.mechanisms[r.id];
  const where = mech.status === "absent" ? "qui vivrebbe, ma questo modello non lo usa" : mech.status === "study" ? "non è un pezzo del modello" : "dove vive";
  return {
    id: `full-mech-${r.id}`,
    kind: "view",
    title: `${r.title} in ${primary.name}`,
    text: `${where}: ${mech.text}`,
    keywords: ["dove sta", "segnaposto", "architettura completa", r.sub, ROWS[r.row].title],
  };
});

/** Parole chiave della vista completa che lo studente potrebbe cercare da sole. */
const TERMS = ["Engram", "mHC", "CSA2", "lightning indexer", "DSpark", "KDA", "Kimi Delta Attention", "MXFP4", "safetensors", "KV cache compressa", "encoder-decoder causale", "LatentMoE", "AttnRes", "attention sink"];
subtabs[1].keywords.push(...TERMS);

// --- rappresentazioni del nucleo (view:rete e view:cervello: reps.goto in index.js) ------------------
// Testi scritti qui (non importati da network.js e brain.js): l'indice deve restare leggero.
const representations = [
  {
    id: "rete",
    kind: "view",
    title: "Rete neurale (LLM moderni)",
    text: `${primary.name} in versione ridotta, disegnato neurone per neurone da sinistra a destra: tokenizer, embedding, le 4 corsie del residuo, i layer nell'ordine vero con le ripetizioni compresse in «× N», head, softmax e DSpark. In profondità i token del prompt: la finestra locale, la KV globale compressa (2 token → 1 voce) con l'indexer che sceglie le voci, l'attention causale, il router che manda ogni token ai suoi esperti. Valori illustrativi.`,
    keywords: [
      "rete neurale", "neural network", "neuroni", "rappresentazione", "profondità", "spessore", "MoE", "mixture of experts", "router", "esperti", "experts", "esperto condiviso",
      "top-k", "KV", "KV cache", "KV latente", "KV globale", "compressione", "attention causale", "causal attention", "finestra", "sliding window", "indexer", "MQA",
      "mHC", "residuo", "RMSNorm", "DeepSeek-V4.1-Flash", "token del prompt",
    ],
  },
  {
    id: "cervello",
    kind: "view",
    title: "Cervello (LLM moderni)",
    text: "Le analogie tra gli LLM moderni e il cervello, ognuna con cosa hanno in comune, dove non regge e le fonti: esperti e specializzazione delle aree, KV cache e memoria di lavoro, RL e dopamina, ragionamento, RoPE e cellule di griglia, leggi di scala; il modello 3D con le regioni e l'anello corteccia, gangli della base e talamo con la dopamina.",
    keywords: [
      "cervello", "brain", "neuroscienze", "analogie", "dopamina", "errore di predizione", "ricompensa", "rinforzo", "gangli della base", "striato", "talamo", "VTA", "substantia nigra",
      "corteccia prefrontale", "memoria di lavoro", "cellule di griglia", "ippocampo", "corteccia entorinale", "area di Broca", "Wernicke", "cervelletto", "rete multiple-demand", "selettività mista",
    ],
  },
];

export const searchIndex = [...steps, ...actions, ...levels, ...regions, ...subtabs, ...fullLevels, ...fullSections, ...fullModels, ...fullMechanisms, ...representations];
