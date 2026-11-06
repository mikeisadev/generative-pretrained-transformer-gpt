/**
 * Catalogo delle azioni animate del tab "LLM moderni": id, titolo, categoria, riassunto e regione
 * della scena. Modulo di soli dati (niente three.js): lo usano i moduli actions*.js (che aggiungono
 * i passi) e l'indice della ricerca globale (search.js), che viene caricato anche quando il tab non
 * è mai stato aperto. Un solo posto per i testi: la ricerca trova esattamente ciò che il pannello
 * "Azioni" mostra.
 *
 * `keywords`: sinonimi e parole (italiane e inglesi) che uno studente potrebbe cercare.
 */
export const ACTION_CATALOG = [
  // --- Architettura
  {
    id: "compare",
    title: "Dal nostro GPT a un blocco moderno",
    category: "Architettura",
    region: "compare",
    summary: "I pezzi del blocco cambiano uno alla volta: RoPE, RMSNorm, GQA, SwiGLU, vocabolario.",
    keywords: ["confronto", "comparison", "Llama 3", "blocco decoder", "decoder block", "LayerNorm", "RMSNorm", "SwiGLU", "differenze"],
  },
  {
    id: "moe",
    title: "Mixture of Experts: il router sceglie",
    category: "Architettura",
    region: "moe",
    summary: "Ogni token va a 2 esperti su 8 (come Mixtral); poi il problema del bilanciamento e l'esperto condiviso di DeepSeek-V3.",
    keywords: ["MoE", "esperti", "experts", "router", "gate", "top-k", "Mixtral", "DeepSeek", "load balancing", "bilanciamento", "sparse"],
  },
  {
    id: "gqa",
    title: "GQA e MQA: query che condividono K e V",
    category: "Architettura",
    region: "gqa",
    summary: "Da una coppia K/V per head (MHA) a una per gruppo (GQA) a una sola (MQA): la KV cache si rimpicciolisce.",
    keywords: ["grouped-query attention", "multi-query attention", "MHA", "multi-head", "head", "key", "value", "memoria", "repeat_kv"],
  },
  {
    id: "rope",
    title: "RoPE: ruotare per dire la posizione",
    category: "Architettura",
    region: "rope",
    summary: "Ogni coppia di dimensioni di q e k ruota di posizione × θ; conta solo la distanza tra i token.",
    keywords: ["rotary position embedding", "rotary", "rotazione", "positional encoding", "posizione", "theta", "contesto lungo"],
  },
  {
    id: "bpe",
    title: "Tokenizer BPE a byte, fusione dopo fusione",
    category: "Architettura",
    region: "bpe",
    summary: "Algoritmo vero: dai byte UTF-8 si fonde ogni volta la coppia più frequente.",
    keywords: ["byte pair encoding", "tokenizer", "tokenizzazione", "tokenization", "merge", "fusioni", "UTF-8", "vocabolario", "vocabulary", "minbpe", "tiktoken"],
  },
  {
    id: "kv",
    title: "KV cache: token dopo token",
    category: "Architettura",
    region: "kv",
    summary: "Senza cache si ricalcola tutto il prefisso a ogni token; con la cache solo il token nuovo.",
    keywords: ["KV cache", "cache", "key value", "generazione", "inferenza", "inference", "memoria", "decoding"],
  },
  {
    id: "flash",
    title: "FlashAttention e finestra scorrevole",
    category: "Architettura",
    region: "flash",
    summary: "Attention esatta calcolata a blocchi nella memoria veloce; poi la maschera a finestra di Mistral 7B.",
    keywords: ["FlashAttention", "sliding window", "finestra scorrevole", "tiling", "SRAM", "GPU", "Mistral", "contesto lungo", "long context"],
  },
  {
    id: "spec",
    title: "Decodifica speculativa",
    category: "Architettura",
    region: "spec",
    summary: "Il piccolo propone, il grande verifica in un passaggio: più token per ogni passaggio costoso.",
    keywords: ["speculative decoding", "draft model", "bozza", "verifica", "velocità", "speed", "inferenza"],
  },
  // --- Addestramento
  {
    id: "scaling",
    title: "Leggi di scala: quanti parametri, quanti dati",
    category: "Addestramento",
    region: "scaling",
    summary: "Con la formula di Chinchilla: per ogni budget di calcolo, N e D ottimali e la loss prevista.",
    keywords: ["scaling laws", "Chinchilla", "compute", "calcolo", "FLOP", "parametri", "dati", "token", "loss"],
  },
  {
    id: "pipeline",
    title: "Pre-training → SFT → preferenze",
    category: "Addestramento",
    region: "training",
    summary: "Le tre fasi che trasformano un modello che continua il testo in un assistente; RLHF, DPO e Constitutional AI.",
    keywords: ["pretraining", "fine-tuning", "SFT", "RLHF", "DPO", "Constitutional AI", "allineamento", "alignment", "post-training", "assistente"],
  },
  {
    id: "emergence",
    title: "Capacità emergenti o miraggio?",
    category: "Addestramento",
    region: "scaling",
    summary: "Un salto improvviso con la scala può dipendere dalla metrica: per token la crescita è dolce, la risposta esatta sembra «accendersi».",
    keywords: ["emergent abilities", "emergenza", "mirage", "metrica", "scala", "salto"],
  },
  // --- Ragionamento e agenti
  {
    id: "cot",
    title: "Catena di pensiero e verifica",
    category: "Ragionamento e agenti",
    region: "reasoning",
    summary: "Passi intermedi, più tentativi, un verificatore: spendere più calcolo al momento della risposta.",
    keywords: ["chain of thought", "CoT", "ragionamento", "reasoning", "test-time compute", "self-consistency", "best-of-n", "verificatore"],
  },
  {
    id: "grpo",
    title: "RL con ricompense verificabili (GRPO)",
    category: "Ragionamento e agenti",
    region: "reasoning",
    summary: "Ricompensa da un verificatore, vantaggio = confronto con la media del gruppo: l'idea di DeepSeek-R1.",
    keywords: ["reinforcement learning", "RL", "GRPO", "RLVR", "ricompensa", "reward", "vantaggio", "advantage", "DeepSeek-R1"],
  },
  {
    id: "agent",
    title: "Ciclo agente e tool use",
    category: "Ragionamento e agenti",
    region: "agent",
    summary: "Il modello scrive una chiamata, il programma esegue lo strumento, il risultato torna nel contesto.",
    keywords: ["agenti", "agents", "tool use", "function calling", "strumenti", "tools", "JSON", "loop"],
  },
  {
    id: "coding",
    title: "Coding: fill-in-the-middle e test",
    category: "Ragionamento e agenti",
    region: "coding",
    summary: "Il riordino FIM che insegna a riempire un buco; il ciclo scrivi → esegui i test → correggi.",
    keywords: ["FIM", "fill in the middle", "codice", "code", "programmazione", "test", "SWE-bench", "unit test"],
  },
  // --- Oltre il testo
  {
    id: "multi",
    title: "Un'immagine diventa token",
    category: "Oltre il testo",
    region: "multi",
    summary: "Schema di LLaVA: patch → encoder visivo → proiettore → token visivi accanto al testo.",
    keywords: ["multimodale", "multimodal", "vision", "immagini", "images", "LLaVA", "patch", "ViT", "CLIP", "proiettore"],
  },
  {
    id: "mesh",
    title: "Una mesh 3D scritta come testo",
    category: "Oltre il testo",
    region: "mesh",
    summary: "Come LLaMA-Mesh: vertici e triangoli nel formato OBJ, coordinate intere; la forma si costruisce riga dopo riga.",
    keywords: ["3D", "mesh", "OBJ", "LLaMA-Mesh", "vertici", "vertices", "triangoli", "Blender"],
  },
  {
    id: "interp",
    title: "Interpretabilità: induction head e feature",
    category: "Oltre il testo",
    region: "interp",
    summary: "Una head che copia lo schema [A][B] … [A] → [B]; un autoencoder sparso che trova concetti leggibili.",
    keywords: ["interpretability", "induction head", "sparse autoencoder", "SAE", "feature", "mechanistic", "concetti"],
  },
];

const BY_ID = new Map(ACTION_CATALOG.map((a) => [a.id, a]));

/** I campi che il player mostra (id, titolo, categoria, riassunto): da usare in ogni azione con `...actionInfo(id)`. */
export function actionInfo(id) {
  const a = BY_ID.get(id);
  if (!a) throw new Error(`Azione sconosciuta nel catalogo: ${id}`);
  return { id: a.id, title: a.title, category: a.category, summary: a.summary };
}

export const ACTION_IDS = new Set(BY_ID.keys());
