/**
 * Specifiche di esempio della rete neurale orizzontale con profondità (soli dati).
 *
 * Sono il riferimento per chi scrive il `network.js` di un tab: ogni campo usato qui è spiegato nei
 * commenti e in ARCHITECTURE.md ("Rete neurale orizzontale"). Tre reti realistiche, ridotte con
 * onestà (la frase `scale` dice di quanto), più una rete grande per la prova di carico:
 *
 *   LLM_SPEC        un LLM MoE come DeepSeek-V3: i token in profondità, attention causale, latente KV
 *                   (MLA), router ed esperti instradati, blocchi ripetuti compressi con "× N"
 *   UNET_SPEC       la U-Net di Stable Diffusion: la griglia del latente in profondità (righe su y,
 *                   colonne su z, un canale per lastra), convoluzioni tra vicini, down/up, skip
 *                   connection, self-attention tra posizioni e cross-attention verso il testo
 *   AUDIO_SPEC      un codec neurale come EnCodec: i frame nel tempo in profondità, convoluzioni 1D,
 *                   il tempo che si accorcia e si riallunga, quantizzazione residua (RVQ) con codebook
 *   loadTestSpec()  ~60.000 neuroni e ~150.000 collegamenti: serve solo a misurare le prestazioni
 *
 * I numeri del "modello vero" vengono dalle fonti elencate in `sources` di ogni rete (configurazioni
 * ufficiali). Le attivazioni sono simulate con un seme: la vista lo dichiara ("simulazione didattica").
 */

// ---------------------------------------------------------------------------------------------- LLM

/** Il prompt in profondità: 8 token illustrativi (il tokenizer vero potrebbe spezzare le parole diversamente). */
const PROMPT = ["Il", "gatto", "salta", "sul", "tavolo", "e", "poi", "dorme"];

/**
 * Un blocco transformer di DeepSeek-V3 come sezioni (prefisso p: "d" = layer denso, "m" = layer MoE).
 * La funzione evita di scrivere due volte le stesse sezioni: i due tipi di layer differiscono solo nel
 * feed-forward.
 */
function llmAttentionPart(p) {
  return [
    { id: `${p}_norm1`, label: "RMSNorm", role: "norm", neurons: 16, depth: true, text: "Riporta il vettore di ogni token su una scala comune (radice della media dei quadrati uguale a 1), poi lo moltiplica per i pesi γ. Lavora token per token: nessun arco in profondità." },
    {
      id: `${p}_kv`, label: "Latente KV (MLA)", sub: "512 numeri per token · qui 8", role: "latent", neurons: 8, depth: true, real: 512,
      text: "Multi-head Latent Attention: invece di salvare key e value di tutte le 128 head, ogni token viene compresso in un latente di 512 numeri (kv_lora_rank). È questo che finisce nella KV cache, molto più piccola che con l'attention classica: key e value si ricostruiscono da qui.",
    },
    {
      id: `${p}_attn`, label: "Attention causale", sub: "128 head · qui 4", role: "attention", neurons: 16, heads: 4, depth: true,
      slabs: 2, slabNames: ["key e value", "uscita"],
      text: "Ogni token guarda se stesso e i token prima di lui, mai quelli dopo (maschera causale): gli archi vanno sempre verso il passato. Un colore per head: DeepSeek-V3 ne ha 128, qui 4. Più luminoso l'arco, più peso ha quel token nella media.",
    },
    { id: `${p}_add1`, label: "+ residuo", role: "add", neurons: 16, depth: true, text: "L'uscita dell'attention si somma al vettore che era entrato nel blocco (connessione residua): il blocco aggiunge informazione, non la sostituisce." },
    { id: `${p}_norm2`, label: "RMSNorm", role: "norm", neurons: 16, depth: true, text: "Seconda normalizzazione, prima del feed-forward." },
  ];
}

/** I collegamenti della parte attention di un blocco: `input` è la sezione da cui parte il residuo. */
function llmAttentionLinks(p, input) {
  return [
    { from: input, to: `${p}_norm1`, pattern: "identity" },
    { from: `${p}_norm1`, to: `${p}_kv`, pattern: "dense", density: 0.3 }, // compressione nel latente
    { from: `${p}_kv`, to: `${p}_attn`, pattern: "dense", density: 0.25, toSlab: 0 }, // key e value ricostruiti
    { within: `${p}_attn`, pattern: "attention", causal: true }, // archi tra token, solo verso il passato
    { from: `${p}_attn`, to: `${p}_add1`, pattern: "dense", density: 0.12 }, // proiezione di uscita W_o
    { from: input, to: `${p}_add1`, pattern: "residual" }, // x + attention(x)
    { from: `${p}_add1`, to: `${p}_norm2`, pattern: "identity" },
  ];
}

export const LLM_SPEC = {
  title: "Rete neurale di un LLM MoE (DeepSeek-V3, ridotta)",
  intro: "Un LLM moderno disegnato come rete neurale: da sinistra a destra il flusso, dall'alto in basso i numeri del vettore di ogni token, in profondità i token del prompt. Gira la camera (\"Guarda lo spessore\"): l'attention collega ogni token ai precedenti e il router manda ogni token ai suoi esperti.",
  scale: "1 sfera = 1 numero di un vettore. Qui 16 numeri per token: DeepSeek-V3 ne ha 7.168. Esperti: 16 disegnati invece di 256, 2 attivi per token invece di 8. Dei 61 layer se ne disegnano due, gli altri sono lastre fantasma.",
  seed: 7,
  axes: { y: "numeri del vettore (16 dei 7.168)", z: "token del prompt" },
  // profondità predefinita: le sezioni con depth: true hanno un neurone per token del prompt
  depth: { count: PROMPT.length, label: "token", names: PROMPT },
  sections: [
    {
      id: "tokens", label: "Token", sub: "one-hot · 16 delle 129.280 voci", role: "input", neurons: 16, depth: true, real: 129280,
      activation: "onehot", // un solo neurone acceso per token: la sua voce del vocabolario
      text: "Ogni token del prompt è un indice nel vocabolario (129.280 voci in DeepSeek-V3): un solo neurone acceso per token. Qui il vocabolario è ridotto a 16 voci.",
    },
    {
      id: "embed", label: "Embedding", sub: "7.168 numeri per token · qui 16", role: "embedding", neurons: 16, depth: true, real: 7168,
      text: "La tabella di embedding (129.280 righe × 7.168 colonne di pesi) dà a ogni token la sua riga. Da qui in poi ogni token è un vettore che attraversa la rete restando alla sua profondità.",
    },
    ...llmAttentionPart("d"),
    {
      id: "d_ffn", label: "Feed-forward denso", sub: "18.432 neuroni nascosti · qui 32", role: "mlp", neurons: 32, depth: true, real: 18432,
      activation: "positive",
      text: "Nei primi 3 layer di DeepSeek-V3 il feed-forward è denso: ogni token passa per 18.432 neuroni nascosti. Ogni token per conto suo, con gli stessi pesi: nessun arco in profondità.",
    },
    { id: "d_add2", label: "+ residuo", role: "add", neurons: 16, depth: true, text: "Di nuovo la connessione residua: x + feed-forward(x)." },
    // compressione: il layer denso disegnato sopra vale per 3 (first_k_dense_replace = 3)
    {
      repeat: 3, of: ["d_norm1", "d_kv", "d_attn", "d_add1", "d_norm2", "d_ffn", "d_add2"], id: "dense_x3", label: "× 3 layer densi",
      text: "I primi 3 layer hanno un feed-forward denso. Ne vedi uno: gli altri due hanno la stessa forma e pesi diversi.",
    },
    ...llmAttentionPart("m"),
    {
      id: "router", label: "Router", sub: "un punteggio per esperto", role: "router", neurons: 16, depth: true, // un neurone per esperto
      text: "Per ogni token un punteggio per ognuno dei 256 esperti (qui 16): vincono gli 8 più alti (qui 2). DeepSeek-V3 calcola i punteggi con la sigmoide e bilancia il carico tra gli esperti senza una loss ausiliaria.",
    },
    {
      id: "experts", label: "Esperti", sub: "256 + 1 condiviso · qui 16 + 1", role: "experts", neurons: 6, depth: true,
      experts: 16, active: 2, shared: 1, router: "router", real: 2048, // 6 dei 2.048 neuroni nascosti di ogni esperto
      text: "Ogni esperto è una piccola MLP (2.048 neuroni nascosti). Ogni token accende solo i suoi esperti: guarda la griglia esperti × token dall'alto. L'esperto condiviso (il primo, separato dagli altri) lavora per tutti i token. Così il modello ha 671 miliardi di parametri, ma ogni token ne usa circa 37 miliardi.",
    },
    { id: "m_add2", label: "+ residuo", role: "add", neurons: 16, depth: true, text: "x + MoE(x): l'uscita pesata degli esperti scelti si somma al residuo." },
    {
      repeat: 58, of: ["m_norm1", "m_kv", "m_attn", "m_add1", "m_norm2", "router", "experts", "m_add2"], id: "moe_x58", label: "× 58 layer MoE",
      text: "Gli altri 58 layer (61 in tutto) sono uguali a questo: attention con il latente KV, router ed esperti. Disegnarli tutti vorrebbe dire 58 copie di questo blocco: qui ne vedi uno vero e poche lastre fantasma.",
    },
    { id: "final_norm", label: "RMSNorm finale", role: "norm", neurons: 16, depth: true, text: "L'ultima normalizzazione prima dell'uscita." },
    {
      id: "logits", label: "Prossimo token", sub: "softmax su 129.280 voci · qui 16", role: "output", neurons: 16, depth: true, real: 129280,
      text: "La matrice di uscita (lm_head: 7.168 → 129.280) dà un punteggio a ogni voce del vocabolario, la softmax lo trasforma in probabilità. Ogni posizione predice il token che viene dopo: il prossimo token della frase si legge all'ultima posizione, quella davanti.",
    },
  ],
  links: [
    { from: "tokens", to: "embed", pattern: "lookup" }, // il token sceglie la sua riga della tabella
    ...llmAttentionLinks("d", "embed"),
    { from: "d_norm2", to: "d_ffn", pattern: "dense", density: 0.1 },
    { from: "d_ffn", to: "d_add2", pattern: "dense", density: 0.1 },
    { from: "d_add1", to: "d_add2", pattern: "residual" },
    ...llmAttentionLinks("m", "d_add2"),
    { from: "m_norm2", to: "router", pattern: "dense", density: 0.2 },
    { from: "router", to: "experts", pattern: "route" }, // ogni token verso i suoi top-k (più il condiviso)
    { from: "experts", to: "m_add2", pattern: "route", density: 0.15 }, // e ritorno, pesato dal router
    { from: "m_add1", to: "m_add2", pattern: "residual" },
    { from: "m_add2", to: "final_norm", pattern: "identity" },
    { from: "final_norm", to: "logits", pattern: "dense", density: 0.25 },
  ],
  groups: [
    { label: "Ingresso", from: "tokens", to: "embed" },
    { label: "Layer 1-3: attention + feed-forward denso", from: "d_norm1", to: "dense_x3" },
    { label: "Layer 4-61: attention + MoE", from: "m_norm1", to: "moe_x58" },
    { label: "Uscita", from: "final_norm", to: "logits" },
  ],
  sources: [
    { title: "deepseek-ai/DeepSeek-V3 (codice e configurazione ufficiale)", url: "https://github.com/deepseek-ai/DeepSeek-V3", note: "inference/configs/config_671B.json: 61 layer di cui 3 densi, larghezza 7.168, 128 head, kv_lora_rank 512, 256 esperti instradati + 1 condiviso, 8 attivi, 2.048 neuroni per esperto, vocabolario 129.280, punteggi con sigmoide." },
    { title: "DeepSeek-AI (2024), DeepSeek-V3 Technical Report", url: "https://arxiv.org/abs/2412.19437", note: "671B parametri totali, 37B attivi per token, 14,8T token di pre-training, MLA, bilanciamento degli esperti senza loss ausiliaria." },
  ],
};

// --------------------------------------------------------------------------------------------- U-Net

export const UNET_SPEC = {
  title: "Rete neurale della U-Net di Stable Diffusion (ridotta)",
  intro: "La U-Net che toglie il rumore dal latente, disegnata come rete: ogni lastra è un canale e ogni lastra è la griglia del latente (righe in alto e in basso, colonne in profondità). Scendendo la griglia si rimpicciolisce e i canali aumentano, risalendo torna grande. Gira la camera per vedere le convoluzioni tra vicini, l'attention tra posizioni e la cross-attention verso le parole del prompt.",
  scale: "1 sfera = 1 numero di una feature map. Griglia 8 × 8 invece di 64 × 64, 4-10 canali invece di 320-1.280; due dimezzamenti della griglia invece di tre.",
  seed: 11,
  axes: { x: "flusso; ogni lastra è un canale", y: "righe della griglia", z: "colonne della griglia" },
  // profondità predefinita: una griglia 2D (righe sull'asse y, colonne sull'asse z)
  depth: { grid: [8, 8], label: "griglia del latente" },
  sections: [
    { id: "noisy", label: "Latente rumoroso z_t", sub: "4 canali × 64 × 64", role: "input", neurons: 4, depth: true, text: "Il VAE comprime un'immagine 512 × 512 in un latente di 4 canali × 64 × 64. All'inizio è puro rumore: la U-Net deve stimare quanto rumore c'è, posizione per posizione." },
    { id: "t_emb", label: "Passo t", sub: "embedding del tempo", role: "embedding", neurons: 6, depth: false, text: "Il numero del passo di denoising (quanto rumore resta) diventa un vettore e raggiunge TUTTE le posizioni della griglia (broadcast): ogni ResNet block lo somma alle sue feature." },
    { id: "conv_in", label: "Conv di ingresso", sub: "4 → 320 canali · qui 6", role: "conv", neurons: 6, depth: true, real: 320, text: "Una convoluzione 3 × 3: ogni posizione guarda solo le 9 posizioni vicine, con gli stessi pesi su tutta la griglia." },
    { id: "res1", label: "ResNet block", sub: "64 × 64 · 320 canali", role: "conv", neurons: 6, depth: true, real: 320, text: "Due convoluzioni 3 × 3 con una scorciatoia (residuo); il passo t viene sommato dentro." },
    { id: "self1", label: "Self-attention", sub: "tra tutte le posizioni", role: "attention", neurons: 6, heads: 2, depth: true, text: "Ogni posizione dell'immagine guarda tutte le altre, senza maschera causale (un'immagine non ha un prima e un dopo): così una zona si accorda con una lontana, per esempio i due occhi dello stesso viso." },
    {
      id: "text", label: "Prompt (CLIP)", sub: "77 token × 768 · qui 6 × 8", role: "embedding", neurons: 8, real: 768,
      depth: { count: 6, label: "token del testo", names: ["un", "gatto", "astronauta", "sulla", "luna", "<fine>"] }, // profondità propria
      text: "Il text encoder (CLIP ViT-L/14) trasforma il prompt in 77 vettori da 768 numeri. Non attraversa la U-Net: entra di lato, nella cross-attention.",
    },
    { id: "cross1", label: "Cross-attention al testo", sub: "query: immagine · key e value: testo", role: "attention", neurons: 6, heads: 2, depth: true, text: "Ogni posizione dell'immagine chiede alle parole del prompt cosa disegnare lì: gli archi vanno dai token del testo alle posizioni della griglia. È così che \"gatto\" finisce dove c'è il gatto." },
    { id: "down1", label: "Downsample", sub: "→ 32 × 32 · 640 canali", role: "conv", neurons: 8, depth: [4, 4], real: 640, text: "Una convoluzione con passo 2: ogni posizione riassume un blocco 2 × 2. La griglia si rimpicciolisce e i canali crescono: dai dettagli alla composizione." },
    { id: "res2", label: "ResNet block", sub: "32 × 32 · 640 canali", role: "conv", neurons: 8, depth: [4, 4], real: 640, text: "Convoluzioni sulla griglia più piccola: ogni posizione ora vede una zona più grande dell'immagine." },
    { id: "down2", label: "Downsample", sub: "→ 8 × 8 · 1.280 canali", role: "conv", neurons: 10, depth: [2, 2], real: 1280, text: "Nel modello vero 64 → 32 → 16 → 8: tre dimezzamenti (qui due). Al fondo ogni posizione riassume una zona grande dell'immagine." },
    { id: "mid", label: "Centro della U", sub: "ResNet + attention + ResNet", role: "attention", neurons: 10, heads: 2, depth: [2, 2], real: 1280, text: "Al centro (8 × 8, 1.280 canali nel modello vero) un ResNet, un'attention e un altro ResNet: qui si decide la composizione d'insieme." },
    { id: "up1", label: "Upsample", sub: "→ 32 × 32", role: "conv", neurons: 8, depth: [4, 4], real: 640, text: "Si risale: ogni posizione si copia in un blocco 2 × 2 e una convoluzione rifinisce. La skip connection porta qui le feature dell'encoder dello stesso livello: i dettagli non vanno persi." },
    { id: "up2", label: "Upsample", sub: "→ 64 × 64", role: "conv", neurons: 6, depth: true, real: 320, text: "Di nuovo alla griglia grande, con la skip connection dal primo livello." },
    { id: "conv_out", label: "Conv di uscita", sub: "320 → 4 canali", role: "conv", neurons: 4, depth: true, text: "L'ultima convoluzione torna a 4 canali, come il latente." },
    { id: "eps", label: "Rumore previsto ε̂", sub: "4 × 64 × 64", role: "output", neurons: 4, depth: true, activation: "signed", text: "La U-Net restituisce il rumore che crede ci sia in z_t, della stessa forma del latente. Lo scheduler lo toglie un po' alla volta: 20-50 passi per un'immagine." },
  ],
  links: [
    { from: "noisy", to: "conv_in", pattern: "conv", kernel: 3, density: 0.12 },
    { from: "conv_in", to: "res1", pattern: "conv", density: 0.1 },
    { from: "t_emb", to: "res1", pattern: "broadcast" }, // lo stesso vettore in ogni posizione
    { from: "res1", to: "self1", pattern: "identity" },
    { within: "self1", pattern: "attention" }, // non causale: tutte le posizioni con tutte
    { from: "text", to: "cross1", pattern: "attention" }, // cross-attention: dai token alle posizioni
    { from: "self1", to: "cross1", pattern: "identity" },
    { from: "cross1", to: "down1", pattern: "down" }, // 8 × 8 → 4 × 4 (il fattore si ricava dalle griglie)
    { from: "down1", to: "res2", pattern: "conv", density: 0.1 },
    { from: "res2", to: "down2", pattern: "down", density: 0.4 },
    { from: "down2", to: "mid", pattern: "identity" },
    { within: "mid", pattern: "attention" },
    { from: "mid", to: "up1", pattern: "up", density: 0.4 },
    { from: "up1", to: "up2", pattern: "up", density: 0.4 },
    { from: "up2", to: "conv_out", pattern: "conv", density: 0.08 },
    { from: "conv_out", to: "eps", pattern: "identity" },
    { from: "res2", to: "up1", pattern: "residual", label: "skip connection" }, // stesso livello della U
    { from: "cross1", to: "up2", pattern: "residual", label: "skip connection" },
  ],
  groups: [
    { label: "Encoder: si scende", from: "conv_in", to: "down2" },
    { label: "Centro", from: "mid", to: "mid" },
    { label: "Decoder: si risale", from: "up1", to: "eps" },
  ],
  sources: [
    { title: "CompVis/stable-diffusion: v1-inference.yaml", url: "https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml", note: "U-Net con 320 canali, channel_mult [1, 2, 4, 4], attention a 3 livelli, context_dim 768; latente a 4 canali (fattore 8 del VAE)." },
    { title: "Rombach et al. (2022), High-Resolution Image Synthesis with Latent Diffusion Models", url: "https://arxiv.org/abs/2112.10752", note: "Diffusione nel latente del VAE e condizionamento con la cross-attention." },
    { title: "Ronneberger, Fischer, Brox (2015), U-Net", url: "https://arxiv.org/abs/1505.04597", note: "L'architettura a U con le skip connection." },
    { title: "openai/CLIP", url: "https://github.com/openai/CLIP", note: "Il tokenizer del text encoder: contesto di 77 token." },
  ],
};

// --------------------------------------------------------------------------------------------- Audio

const FRAMES = Array.from({ length: 12 }, (_, i) => `${Math.round((i * 1000) / 75)} ms`); // 75 frame al secondo

export const AUDIO_SPEC = {
  title: "Rete neurale di un codec audio neurale (EnCodec, ridotto)",
  intro: "Un codec neurale comprime il suono in pochi numeri per frame e poi lo ricostruisce. In profondità c'è il tempo, un frame dopo l'altro: le convoluzioni guardano gli istanti vicini, l'encoder accorcia il tempo, la quantizzazione residua (RVQ) sostituisce ogni frame con una voce di codebook dopo l'altra, il decoder riallunga il tempo.",
  scale: "1 sfera = 1 numero. 12 frame (160 ms: EnCodec a 24 kHz ne produce 75 al secondo), pochi canali, 8 delle 1.024 voci di ogni codebook.",
  seed: 5,
  axes: { y: "canali", z: "tempo: frame da 13,3 ms" },
  depth: { count: FRAMES.length, label: "frame", names: FRAMES },
  sections: [
    {
      id: "wave", label: "Forma d'onda", sub: "24.000 campioni al secondo", role: "input", neurons: 6, activation: "signed",
      depth: { count: 24, label: "tempo (blocchi di campioni)" }, // il doppio dei frame: l'encoder lo dimezzerà
      text: "Il suono entra come campioni (24.000 al secondo); qui ogni posizione in profondità è un piccolo blocco di campioni. L'encoder vero riduce il tempo di 320 volte (8 · 5 · 4 · 2) per arrivare a 75 frame al secondo.",
    },
    { id: "enc1", label: "Conv dell'encoder", role: "conv", neurons: 8, depth: { count: 24, label: "tempo (blocchi di campioni)" }, text: "Convoluzioni 1D nel tempo: ogni istante guarda solo gli istanti vicini (kernel 3)." },
    { id: "enc2", label: "Conv con passo 2", sub: "il tempo si dimezza", role: "conv", neurons: 10, depth: true, text: "Una convoluzione con passo (stride) 2: due istanti diventano uno. Nel modello vero le riduzioni sono 8, 5, 4 e 2, da 24.000 campioni a 75 frame al secondo." },
    { id: "latent", label: "Latente", sub: "un vettore per frame", role: "latent", neurons: 8, depth: true, text: "Ogni frame (13,3 ms di suono) è diventato un vettore di numeri: è questo che va compresso." },
    { id: "cb1", label: "Codebook 1", sub: "1.024 voci · qui 8", role: "codebook", neurons: 8, depth: true, real: 1024, text: "Quantizzazione: il vettore del frame si confronta con le 1.024 voci del codebook e viene sostituito dalla più vicina (la voce accesa). Basta salvare il suo indice: 10 bit per frame." },
    { id: "r1", label: "Residuo", sub: "quello che il codebook non ha preso", role: "latent", neurons: 8, depth: true, text: "Al latente si toglie la voce scelta: resta un errore più piccolo. Il codebook dopo quantizza proprio questo residuo (Residual Vector Quantization)." },
    { id: "cb2", label: "Codebook 2", sub: "1.024 voci · qui 8", role: "codebook", neurons: 8, depth: true, real: 1024, text: "Il secondo codebook sceglie la voce più vicina al residuo: aggiunge dettaglio." },
    {
      repeat: 7, of: ["r1", "cb2"], id: "rvq_x7", label: "× 7 residuo → codebook",
      text: "A 6 kbps EnCodec usa 8 codebook: il primo più 7 coppie residuo → codebook come questa (fino a 32 codebook a 24 kbps). 75 frame × 8 codebook × 10 bit = 6.000 bit al secondo.",
    },
    { id: "quant", label: "Latente quantizzato", sub: "somma delle voci scelte", role: "add", neurons: 8, depth: true, text: "La somma delle voci scelte ricostruisce (quasi) il latente. Gli indici delle voci sono anche i token audio che un modello come MusicGen impara a generare." },
    { id: "dec1", label: "Conv trasposta", sub: "il tempo si riallunga", role: "conv", neurons: 10, depth: { count: 24, label: "tempo (blocchi di campioni)" }, text: "Il decoder fa il cammino inverso: convoluzioni trasposte che allungano di nuovo il tempo." },
    { id: "dec2", label: "Conv del decoder", role: "conv", neurons: 8, depth: { count: 24, label: "tempo (blocchi di campioni)" }, text: "Convoluzioni 1D che rifiniscono il segnale." },
    { id: "out", label: "Forma d'onda ricostruita", role: "output", neurons: 6, activation: "signed", depth: { count: 24, label: "tempo (blocchi di campioni)" }, text: "Il suono ricostruito: quasi uguale all'originale, ma descritto da 8 indici per frame invece che da 320 campioni." },
  ],
  links: [
    { from: "wave", to: "enc1", pattern: "conv", density: 0.2 },
    { from: "enc1", to: "enc2", pattern: "down", density: 0.3 }, // 24 → 12 istanti
    { from: "enc2", to: "latent", pattern: "dense", density: 0.3 },
    { from: "latent", to: "cb1", pattern: "dense", density: 0.3, label: "confronto con le voci" },
    { from: "cb1", to: "r1", pattern: "lookup" }, // il vettore della voce scelta
    { from: "latent", to: "r1", pattern: "residual", label: "latente − voce scelta" },
    { from: "r1", to: "cb2", pattern: "dense", density: 0.3 },
    { from: "cb1", to: "quant", pattern: "lookup", density: 0.5 },
    { from: "cb2", to: "quant", pattern: "lookup", density: 0.5 },
    { from: "quant", to: "dec1", pattern: "up", density: 0.3 }, // 12 → 24 istanti
    { from: "dec1", to: "dec2", pattern: "conv", density: 0.15 },
    { from: "dec2", to: "out", pattern: "conv", density: 0.2 },
  ],
  groups: [
    { label: "Encoder", from: "enc1", to: "latent" },
    { label: "Quantizzazione residua (RVQ)", from: "cb1", to: "quant" },
    { label: "Decoder", from: "dec1", to: "out" },
  ],
  sources: [
    { title: "Défossez et al. (2022), High Fidelity Neural Audio Compression (EnCodec)", url: "https://arxiv.org/abs/2210.13438", note: "Encoder e decoder convoluzionali con latente quantizzato a più codebook (RVQ)." },
    { title: "facebookresearch/encodec (codice ufficiale)", url: "https://github.com/facebookresearch/encodec", note: "24 kHz: 1,5/3/6/12/24 kbps con 2/4/8/16/32 codebook da 1.024 voci; riduzione 8 · 5 · 4 · 2 = 320, cioè 75 frame al secondo." },
    { title: "Copet et al. (2023), Simple and Controllable Music Generation (MusicGen)", url: "https://arxiv.org/abs/2306.05284", note: "Un transformer che genera i token di 4 codebook EnCodec (32 kHz, 50 frame al secondo)." },
  ],
};

// ------------------------------------------------------------------------------------ prova di carico

/**
 * Una rete grande e senza compressioni, per misurare le prestazioni (non è un modello vero): con i
 * valori predefiniti 8 blocchi × 32 token → ~63.000 neuroni, ~120.000 collegamenti dritti e ~5.000 archi
 * (curve da 6 segmenti): ~150.000 segmenti di linea in tutto.
 */
export function loadTestSpec({ blocks = 8, tokens = 32, width = 24, hidden = 96, density = 0.072 } = {}) {
  const sections = [{ id: "in", label: "Ingresso", role: "input", neurons: width, depth: true, activation: "onehot" }];
  const links = [];
  let prev = "in";
  for (let b = 0; b < blocks; b++) {
    const p = `b${b}`;
    sections.push(
      { id: `${p}_n1`, label: `Norm ${b + 1}`, role: "norm", neurons: width, depth: true },
      { id: `${p}_att`, label: `Attention ${b + 1}`, role: "attention", neurons: width, heads: 4, slabs: 2, depth: true },
      { id: `${p}_a1`, label: "+", role: "add", neurons: width, depth: true },
      { id: `${p}_n2`, label: `Norm ${b + 1}b`, role: "norm", neurons: width, depth: true },
      { id: `${p}_mlp`, label: `MLP ${b + 1}`, role: "mlp", neurons: hidden, depth: true, activation: "positive" },
      { id: `${p}_a2`, label: "+", role: "add", neurons: width, depth: true },
    );
    links.push(
      { from: prev, to: `${p}_n1`, pattern: "identity" },
      { from: `${p}_n1`, to: `${p}_att`, pattern: "dense", density, toSlab: "all" },
      { within: `${p}_att`, pattern: "attention", causal: true },
      { from: `${p}_att`, to: `${p}_a1`, pattern: "dense", density },
      { from: prev, to: `${p}_a1`, pattern: "residual" },
      { from: `${p}_a1`, to: `${p}_n2`, pattern: "identity" },
      { from: `${p}_n2`, to: `${p}_mlp`, pattern: "dense", density },
      { from: `${p}_mlp`, to: `${p}_a2`, pattern: "dense", density },
      { from: `${p}_a1`, to: `${p}_a2`, pattern: "residual" },
    );
    prev = `${p}_a2`;
  }
  sections.push({ id: "out", label: "Uscita", role: "output", neurons: 32, depth: true });
  links.push({ from: prev, to: "out", pattern: "dense", density });
  return {
    title: "Prova di carico della rete neurale",
    intro: "Una rete grande, senza blocchi compressi, per misurare quanto costa disegnarla. Non è un modello vero.",
    scale: `${blocks} blocchi × ${tokens} token, ${width} neuroni per vettore e ${hidden} nell'MLP.`,
    seed: 3,
    axes: { y: "neuroni", z: "token" },
    depth: { count: tokens, label: "token" },
    sections,
    links,
    groups: Array.from({ length: blocks }, (_, b) => ({ label: `Blocco ${b + 1}`, from: `b${b}_n1`, to: `b${b}_a2` })),
  };
}

/** Elenco per la pagina di prova (tests/neuralnet-demo.html) e per i test. */
export const DEMO_SPECS = Object.freeze({
  llm: LLM_SPEC,
  unet: UNET_SPEC,
  audio: AUDIO_SPEC,
});
