/**
 * Catalogo della vista "Architettura LLM moderni completa" (SOLI DATI): azioni animate e viste con
 * i loro id per goto e per la ricerca globale. Lo usano full/actions.js (che aggiunge i passi), i
 * pannelli e search.js, che viene caricato anche quando il tab non è mai stato aperto.
 *
 * Id (tutti univoci nel tab, con il prefisso "full"):
 *   action:full-<id>                          le azioni qui sotto
 *   view:full                                 il sottotab (con la camera dove l'avevi lasciata)
 *   view:full-macro | full-blocks | full-detail   i tre livelli dello zoom
 *   view:full-<sezione>                       un pezzo dell'architettura (FULL_SECTIONS)
 *   view:full-model-<id del modello>          scegli il modello (full/models/index.js)
 *   view:full-layer-<n>                       il Dettaglio del layer n (non indicizzato: sono tanti)
 *   view:full-mech-<id della regione>         dove vive un meccanismo della vista d'insieme
 *   view:overview                             torna alla vista d'insieme
 */

export const FULL_CATEGORIES = ["Dentro il modello", "Dai dati alle capacità"];

export const FULL_ACTIONS = [
  {
    id: "full-token",
    title: "Un token attraversa il modello",
    category: "Dentro il modello",
    summary: "Dal tokenizer all'embedding, nelle 4 corsie del residuo, attraverso encoder e decoder fino alla head e al token scelto.",
    keywords: ["forward", "flusso", "token", "pipeline", "encoder", "decoder", "head", "softmax"],
  },
  {
    id: "full-router",
    title: "Il router sceglie 6 esperti su 384",
    category: "Dentro il modello",
    summary: "Punteggi sqrtsoftplus, bias di bilanciamento, top-6, esperto condiviso e somma pesata × 1,5 (scelte simulate: non sono pubblicate).",
    keywords: ["MoE", "router", "esperti", "experts", "top-k", "sqrtsoftplus", "bilanciamento", "load balancing", "esperto condiviso"],
  },
  {
    id: "full-kv",
    title: "La KV cache compressa: 890 byte per token",
    category: "Dentro il modello",
    summary: "2 token diventano 1 voce, la voce sta in FP4 (288 B), i layer Reuse la riusano: 437 volte meno di un modello GQA da 67B.",
    keywords: ["KV cache", "compressione", "FP4", "CSA2", "riuso", "memoria", "HBM", "byte per token"],
  },
  {
    id: "full-ced",
    title: "Prefill solo encoder, decode completo",
    category: "Dentro il modello",
    summary: "Il prompt passa solo dai 20 layer dell'encoder (8B attivi); ogni token generato passa da tutti i 40 (16B).",
    keywords: ["encoder-decoder causale", "CED", "YOCO", "prefill", "decode", "H20", "parametri attivi"],
  },
  {
    id: "full-engram",
    title: "Engram: hash → 24 righe → gate",
    category: "Dentro il modello",
    summary: "Nei layer 1 e 14 gli ultimi 2, 3 e 4 token passano per 8 hash ciascuno: 24 righe lette da tabelle enormi e sommate al residuo.",
    keywords: ["Engram", "n-gram", "hash", "memoria", "lookup", "tabelle", "DRAM"],
  },
  {
    id: "full-dspark",
    title: "DSpark: 5 token proposti e verificati",
    category: "Dentro il modello",
    summary: "Le feature dei layer 37-39 alimentano 3 blocchi draft che propongono 5 token; il modello li verifica in un passaggio.",
    keywords: ["DSpark", "MTP", "multi-token prediction", "decodifica speculativa", "draft", "bozza"],
  },
  {
    id: "full-train",
    title: "Dal training ai pesi",
    category: "Dai dati alle capacità",
    summary: "45T token, pre-training, contesto lungo, SFT → RL → OPD, poi il checkpoint: 96.085 tensori, 510,3 GB, precisioni diverse.",
    keywords: ["training", "pre-training", "post-training", "SFT", "RL", "OPD", "checkpoint", "safetensors", "pesi", "MXFP4"],
  },
  {
    id: "full-emerge",
    title: "Come emergono le capacità",
    category: "Dai dati alle capacità",
    summary: "Leggi di scala, abilità «emergenti» e la critica della metrica, induction head, grokking, la sottorete che l'RL modifica.",
    keywords: ["capacità emergenti", "emergent abilities", "leggi di scala", "scaling laws", "induction head", "grokking", "Schaeffer", "RL"],
  },
];

const BY_ID = new Map(FULL_ACTIONS.map((a) => [a.id, a]));

/** I campi che il player mostra: da usare in ogni azione con `...fullActionInfo(id)`. */
export function fullActionInfo(id) {
  const a = BY_ID.get(id);
  if (!a) throw new Error(`Azione sconosciuta nel catalogo della vista completa: ${id}`);
  return { id: a.id, title: a.title, category: a.category, summary: a.summary };
}

export const FULL_ACTION_IDS = new Set(BY_ID.keys());

/** Livelli dello zoom della vista completa. */
export const FULL_LEVELS = [
  { id: "full-macro", level: "macro", title: "Panoramica: dai dati al token", text: "Dati, pre-training, contesto lungo, post-training, pesi, deployment e inferenza del modello scelto." },
  { id: "full-blocks", level: "blocks", title: "Blocchi: l'architettura completa", text: "Tokenizer, embedding, i layer raggruppati in blocchi «× N», head, cache e rami." },
  { id: "full-detail", level: "detail", title: "Dettaglio: dentro un layer", text: "Residuo, norme, attention con tutti i suoi pezzi, MoE con la griglia degli esperti." },
];

/**
 * Pezzi dell'architettura con una vista propria (view:full-<id>): `anchor` è l'ancora della scena
 * (full/model.js, anchorsOf); se il modello scelto non ha quel pezzo la vista lo dice.
 */
export const FULL_SECTIONS = [
  { id: "full-data", anchor: "macro:data", title: "Dati e pre-training", keywords: ["dati", "corpus", "token", "pre-training"] },
  { id: "full-context", anchor: "macro:context", title: "Estensione del contesto", keywords: ["contesto lungo", "YaRN", "1M", "long context"] },
  { id: "full-post", anchor: "macro:post", title: "Post-training: SFT, RL, distillazione", keywords: ["SFT", "RL", "GRPO", "OPD", "distillazione"] },
  { id: "full-weights", anchor: "macro:weights", title: "I pesi: il checkpoint", keywords: ["checkpoint", "safetensors", "shard", "pesi", "precisione"] },
  { id: "full-deploy", anchor: "macro:deploy", title: "Deployment", keywords: ["deployment", "expert parallel", "EPLB", "prefill", "decode", "GPU"] },
  { id: "full-capabilities", anchor: "macro:capabilities", title: "Come emergono le capacità", keywords: ["capacità", "emergenza", "scaling", "induction head", "grokking"] },
  { id: "full-tokenizer", anchor: "blocks:tokenizer", title: "Tokenizer", keywords: ["tokenizer", "BPE", "token speciali", "DSML", "think"] },
  { id: "full-embedding", anchor: "blocks:embedding", title: "Embedding", keywords: ["embedding", "vocabolario"] },
  { id: "full-layers", anchor: "blocks:layers", title: "I layer raggruppati («× N»)", keywords: ["layer", "blocchi ripetuti", "n-Transformer", "ripetizioni", "espandi"] },
  { id: "full-encoder", anchor: "blocks:segment-encoder", title: "Encoder causale", keywords: ["encoder", "prefill", "CED"] },
  { id: "full-decoder", anchor: "blocks:segment-decoder", title: "Decoder", keywords: ["decoder", "H20", "decode"] },
  { id: "full-caches", anchor: "blocks:cache-kv", title: "La riga delle cache", keywords: ["KV cache", "finestra", "SWA", "Engram", "memoria"] },
  { id: "full-dspark", anchor: "blocks:dspark", title: "DSpark (multi-token)", keywords: ["DSpark", "MTP", "draft"] },
  { id: "full-vision", anchor: "blocks:vision", title: "Visione", keywords: ["visione", "ViT", "immagini", "multimodale"] },
  { id: "full-head", anchor: "blocks:head", title: "Norm, head e softmax", keywords: ["head", "logit", "softmax", "lm_head"] },
  { id: "full-attention", anchor: "detail:core", title: "Attention dentro un layer", keywords: ["attention", "MQA", "latente", "indexer", "finestra", "sink"] },
  { id: "full-moe", anchor: "detail:experts", title: "MoE: la griglia degli esperti", keywords: ["MoE", "esperti", "router", "griglia"] },
  { id: "full-residual", anchor: "detail:residual", title: "Il residuo (mHC)", keywords: ["mHC", "hyper-connections", "residuo", "Sinkhorn", "flussi"] },
  { id: "full-engram", anchor: "detail:table", layer: 14, title: "Engram nel layer 14", keywords: ["Engram", "n-gram", "hash", "tabelle"] },
];

/** Nome breve dei 16 meccanismi sui segnaposto della scena (si legge da lontano). */
export const MARKER_LABELS = {
  compare: "GPT e blocco moderno",
  moe: "MoE",
  gqa: "MHA, GQA, MQA",
  rope: "RoPE",
  bpe: "BPE",
  kv: "KV cache",
  flash: "Attention efficiente",
  spec: "Decodifica speculativa",
  scaling: "Leggi di scala",
  training: "Addestramento",
  reasoning: "Ragionamento",
  interp: "Interpretabilità",
  agent: "Tool e agenti",
  coding: "Coding",
  multi: "Multimodale",
  mesh: "Modelli 3D",
};
