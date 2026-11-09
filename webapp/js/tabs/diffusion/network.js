/**
 * Rappresentazione "Rete neurale" del tab Immagini e video: la specifica (soli dati) che il renderer del
 * nucleo disegna in orizzontale con la profondità (js/core/neuralNet/, formato in ARCHITECTURE.md,
 * "Rete neurale orizzontale"). La carica core/representations.js con un import dinamico.
 *
 * Da sinistra a destra, come la pipeline dell'Architettura:
 *   text encoder CLIP ViT-L/14   i token del prompt in profondità, attention causale, × 12 layer
 *   U-Net di Stable Diffusion 1.x la griglia del latente (righe su y, colonne su z, un canale per lastra):
 *                                convoluzioni tra vicini, la griglia che si dimezza e raddoppia, skip
 *                                connection, self-attention tra posizioni e cross-attention verso il testo
 *                                nello stesso blocco (come il blocco viola della scena), il passo t diffuso a
 *                                ogni ResNet block (broadcast)
 *   scheduler e VAE decoder       dal latente ai pixel
 *   un blocco di denoiser video   la griglia frame × patch: attention spaziale (lungo le righe) e
 *                                temporale (lungo le colonne)
 *
 * La profondità predefinita (le "fette" della scheda) sono i token del prompt: scegliendo una parola si
 * segue il suo percorso nel text encoder e i suoi archi di cross-attention dentro la U-Net.
 *
 * Il disegno è compatto apposta (di lato la rete deve stare sullo schermo): ogni sezione è un passo che
 * si vede, e i pezzi senza un meccanismo proprio stanno nel testo della sezione accanto (le LayerNorm del
 * testo, downsample e upsample dentro il collegamento che cambia la griglia, il feed-forward del blocco di
 * attention). La frase `scale` dice tutte le riduzioni.
 *
 * Numeri dell'architettura: dalle fonti del tab (content.js) e dal codice ufficiale, letto e citato in
 * `sources` (configurazione v1-inference.yaml, openaimodel.py, attention.py, model.py e il modello di
 * CLIP, a commit fissati).
 *
 * Valori veri (calcolati qui con le funzioni del tab, math.js): il rumore iniziale dell'azione
 * "Generazione" e i seni e coseni del passo t. Tutto il resto è simulato dal nucleo con un seme e la vista
 * lo dichiara ("simulazione didattica"): il denoiser del tab è simulato (denoiserSim.js), nel browser non
 * c'è una U-Net addestrata. Nel blocco video gli archi dell'attention sono la STRUTTURA vera
 * dell'attention fattorizzata (chi può guardare chi), con pesi uguali dichiarati come tali.
 *
 * Modulo puro: niente three.js né DOM (lo provano i test in tests/diffusion-reps.test.mjs).
 */
import { fillGaussian, mulberry32, samplingTimesteps, SCHEDULES, timestepEmbedding } from "./math.js";

// ------------------------------------------------------------------------------------------ costanti

/** Impostazioni predefinite dell'azione "Generazione" (tab.settings in index.js: un test le confronta). */
export const GENERATION = Object.freeze({ seed: 42, steps: 20 });
/** Il primo passo del campionatore con quelle impostazioni (spaziatura "leading" di diffusers): 951. */
export const FIRST_T = samplingTimesteps(SCHEDULES.sd.T, GENERATION.steps)[0];

/** Dimensioni vere (v1-inference.yaml e codice ufficiale): usate nei testi e nei campi `real`. */
export const SD = Object.freeze({
  latentChannels: 4,
  latentSide: 64, // 512 × 512 pixel / fattore 8 del VAE
  modelChannels: 320,
  timeEmbed: 1280, // model_channels × 4 (openaimodel.py)
  heads: 8,
  textTokens: 77,
  textWidth: 768,
  textLayers: 12,
  textHeads: 12,
  textMlp: 3072, // 4 × 768 (c_fc di CLIP)
  vocab: 49408, // 256 + 256 + 48.894 fusioni BPE + 2 token speciali (simple_tokenizer.py)
  vaeChannels: 512, // ch 128 × ch_mult[-1] 4
});

/**
 * Riduzione del disegno. La griglia di ogni livello della U-Net è divisa per 8 in ogni lato (64 → 8,
 * 32 → 4, 16 → 2), ma non scende sotto 2 × 2: il livello più piccolo (8 × 8) diviso per 8 avrebbe una
 * posizione sola, e al centro non si vedrebbe più l'attention tra posizioni; così il terzo dimezzamento
 * (16 → 8) e il primo raddoppio del decoder non si vedono, e lo si dice. Con una griglia più fitta gli
 * archi della cross-attention (fino a 4 per posizione) coprirebbero tutta la rete.
 * Si disegna 1 canale ogni 320: 1, 2, 4, 4 lastre, proprio channel_mult [1, 2, 4, 4].
 */
export const GRID_DIVISOR = 8;
export const MIN_SIDE = 2;
export const CHANNEL_DIVISOR = 320;
const side = (res) => Math.max(MIN_SIDE, res / GRID_DIVISOR);
const shown = (channels) => channels / CHANNEL_DIVISOR;
const grid = (res) => ({ grid: [side(res), side(res)], label: "griglia del latente" });

/** I quattro livelli della U-Net (channel_mult [1, 2, 4, 4], attention_resolutions [4, 2, 1]). */
export const UNET_LEVELS = Object.freeze([
  Object.freeze({ level: 0, res: 64, channels: 320, attention: true }),
  Object.freeze({ level: 1, res: 32, channels: 640, attention: true }),
  Object.freeze({ level: 2, res: 16, channels: 1280, attention: true }),
  Object.freeze({ level: 3, res: 8, channels: 1280, attention: false }),
]);

/**
 * Il prompt del tab (pipeline.js) in profondità: 10 dei 77 token. Suddivisione indicativa, come nella
 * scena: il tokenizer vero spezzerebbe le parole italiane in frammenti.
 */
export const TEXT_TOKENS = Object.freeze(["<inizio>", "un", "faro", "su", "una", "scogliera", "al", "tramonto", "<fine>", "riempimento"]);

/** Video: 4 frame latenti (righe) × 4 patch (colonne) = 16 token. */
export const VIDEO_FRAMES = 4;
export const VIDEO_PATCHES = 4;
const VIDEO_GRID = { grid: [VIDEO_FRAMES, VIDEO_PATCHES], label: "frame × patch" };

/** Migliaia col punto, senza dipendere dalla lingua del sistema: 49408 → "49.408". */
const thousands = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ".");

// ------------------------------------------------------------------------------------------ valori veri

/**
 * Il rumore iniziale z_T dell'azione "Generazione" (actionGenerate.js: fillGaussian con il seed del tab
 * sul latente 4 × 32 × 32 del VAE simulato), ritagliato all'angolo in alto a sinistra: 8 × 8 per canale.
 * Restituisce i valori nell'ordine del nucleo: [canale][riga × 8 + colonna].
 */
export function initialNoiseCrop({ seed = GENERATION.seed, latentSide = 32, crop = side(SD.latentSide), channels = SD.latentChannels } = {}) {
  const noise = fillGaussian(new Float32Array(channels * latentSide * latentSide), mulberry32(seed));
  const out = new Float32Array(channels * crop * crop);
  for (let c = 0; c < channels; c++) {
    for (let r = 0; r < crop; r++) {
      for (let col = 0; col < crop; col++) out[c * crop * crop + r * crop + col] = noise[c * latentSide * latentSide + r * latentSide + col];
    }
  }
  return out;
}

/**
 * Gli 8 numeri disegnati dei 320 dell'embedding sinusoidale del passo t (la funzione del tab è la stessa
 * di timestep_embedding nel codice di Stable Diffusion: metà coseni e metà seni, max_period 10.000).
 * Indici 0, 40, 80, 120 di ogni metà: frequenze 1, 1/10, 1/100, 1/1000 (10.000^(−i/160)).
 */
export const TIME_INDICES = Object.freeze([0, 40, 80, 120]);
export const TIME_NAMES = Object.freeze([
  ...["t", "t/10", "t/100", "t/1000"].map((a) => `cos(${a})`),
  ...["t", "t/10", "t/100", "t/1000"].map((a) => `sin(${a})`),
]);
export function timestepValues(t = FIRST_T) {
  const full = timestepEmbedding(new Float32Array(SD.modelChannels), t);
  const half = SD.modelChannels / 2;
  return Float32Array.from([...TIME_INDICES.map((i) => full[i]), ...TIME_INDICES.map((i) => full[half + i])]);
}

// ------------------------------------------------------------------------------------------ pesi illustrativi

/**
 * Le parole del prompt che nominano una zona dell'immagine del tab (indici in TEXT_TOKENS). Il mare non è
 * nel prompt: lo colora il sole che tramonta, come il cielo.
 */
export const ZONE_WORDS = Object.freeze({ faro: 2, scogliera: 5, tramonto: 7 });

/**
 * Zona di un punto dell'immagine del tab (u, v in [0, 1], v dall'alto in basso): "faro" (tetto, lanterna e
 * torre), "scogliera" o "tramonto" (cielo e mare). Le forme sono quelle di renderScene in imagery.js, che
 * non le esporta: un test confronta le zone con i colori dell'immagine disegnata, così se la scena cambia
 * il test se ne accorge.
 */
export function sceneZone(u, v) {
  const towerX = 0.24;
  const towerTop = 0.3;
  const cliffTop = 0.585;
  const dx = Math.abs(u - towerX);
  const roof = (towerTop - 0.04 - v) / 0.045;
  if (roof >= 0 && roof <= 1 && dx < 0.032 * (1 - roof)) return "faro";
  if (v > towerTop - 0.04 && v <= towerTop && dx < 0.026) return "faro";
  if (v > towerTop && v < cliffTop + 0.01 && dx < 0.022 + 0.01 * ((v - towerTop) / (cliffTop - towerTop))) return "faro";
  const edge = 0.42 + 0.05 * Math.sin(v * 23) + (v - cliffTop) * 0.35;
  const top = cliffTop + 0.015 * Math.sin(u * 40) + Math.max(0, u - 0.33) * 0.9;
  return u < edge && v > top ? "scogliera" : "tramonto";
}

/**
 * Pesi ILLUSTRATIVI della cross-attention verso una griglia side × side (1 head, [query][token], righe che
 * sommano a 1): ogni posizione divide il peso tra le parole delle zone che copre, in proporzione all'area
 * (samples × samples punti per cella). Scelti sulla scena del tab, non calcolati da un modello: in Stable
 * Diffusion la mappa di una parola che nomina un oggetto si concentra davvero dove l'oggetto compare
 * (Tang et al. 2022, DAAM), ma ogni posizione dà un po' di peso a tutti i 77 token. Perché non i pesi
 * simulati del nucleo: quelli scelgono parole a caso (la fetta «faro» cadrebbe in posizioni a caso) e con 4
 * archi per posizione la cross-attention diventa una tenda viola che copre la U-Net; così gli archi sono
 * circa un terzo e ogni parola entra nella sua zona a ogni livello.
 */
export function crossAttentionMap(side, samples = 16) {
  const T = TEXT_TOKENS.length;
  const out = new Float32Array(side * side * T);
  const share = 1 / (samples * samples);
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const row = (r * side + c) * T;
      for (let a = 0; a < samples; a++) {
        for (let b = 0; b < samples; b++) out[row + ZONE_WORDS[sceneZone((c + (b + 0.5) / samples) / side, (r + (a + 0.5) / samples) / side)]] += share;
      }
    }
  }
  return out;
}

/**
 * Pesi "di struttura" dell'attention fattorizzata del video (1 head, query × chiave, righe che sommano a
 * 1; posizione = frame × 4 + patch, come nella griglia): spaziale = le posizioni dello stesso frame,
 * temporale = la stessa patch negli altri frame. Pesi uguali: dicono chi può guardare chi, non quanto
 * (in un modello vero dipende dal contenuto).
 */
export function videoAttentionMask(kind) {
  const n = VIDEO_FRAMES * VIDEO_PATCHES;
  const out = new Float32Array(n * n);
  const frame = (i) => Math.floor(i / VIDEO_PATCHES);
  const patch = (i) => i % VIDEO_PATCHES;
  for (let q = 0; q < n; q++) {
    const keys = [];
    for (let k = 0; k < n; k++) if (kind === "spatial" ? frame(k) === frame(q) : patch(k) === patch(q)) keys.push(k);
    for (const k of keys) out[q * n + k] = 1 / keys.length;
  }
  return out;
}

// ------------------------------------------------------------------------------------------ testo (CLIP)

function textEncoder() {
  const sections = [
    {
      id: "txt_tokens", label: "Token del prompt", sub: `10 dei ${SD.textTokens} token · 12 delle ${thousands(SD.vocab)} voci`, role: "input", neurons: 12, depth: true, activation: "onehot", real: SD.vocab,
      text: `Il tokenizer BPE di CLIP spezza il prompt in pezzi di parola e aggiunge <inizio> e <fine>; il resto, fino a ${SD.textTokens} posizioni, è riempimento. Ogni token è un indice nel vocabolario: un solo neurone acceso, come un interruttore su ${thousands(SD.vocab)}.\n\nLa divisione in token è indicativa, come nella scena: il tokenizer vero spezzerebbe le parole italiane in frammenti.`,
    },
    {
      id: "txt_embed", label: "Embedding e posizione", sub: `${SD.textWidth} numeri per token · qui 12`, role: "embedding", neurons: 12, depth: true, real: SD.textWidth,
      text: `Ogni token prende la sua riga della tabella di embedding (${thousands(SD.vocab)} righe da ${SD.textWidth} numeri) e le si somma il vettore della sua posizione (${SD.textTokens} righe, imparate anche quelle). Come una carta d'identità: chi è il token e dove sta nella frase.`,
    },
    {
      id: "txt_attn", label: "Self-attention causale", sub: `dopo una LayerNorm · ${SD.textHeads} head, qui 4`, role: "attention", neurons: 12, heads: 4, slabs: 2, slabNames: ["key e value", "uscita"], depth: true, real: SD.textWidth,
      text: "Una LayerNorm riporta i numeri di ogni token su una scala comune; poi ogni token raccoglie informazioni dai token prima di lui, mai da quelli dopo: il text encoder di CLIP usa una maschera causale (nel codice: build_attention_mask), come il nostro GPT. È come leggere una frase da sinistra a destra senza sbirciare avanti.\n\nGira la camera: gli archi vanno sempre verso il passato. Un colore per head: CLIP ne ha 12, qui 4. Pesi di attention simulati.",
    },
    { id: "txt_add1", label: "+ residuo", role: "add", neurons: 12, depth: true, text: "L'uscita dell'attention si somma al vettore che era entrato nel layer (connessione residua): il layer aggiunge informazione, non la sostituisce." },
    {
      id: "txt_mlp", label: "MLP", sub: `dopo una LayerNorm · ${thousands(SD.textMlp)} neuroni nascosti, qui 48`, role: "mlp", neurons: 48, depth: true, activation: "positive", real: SD.textMlp,
      text: `Dopo una seconda LayerNorm ogni token passa da solo in una rete a due strati: ${SD.textWidth} → ${thousands(SD.textMlp)} → ${SD.textWidth}, con l'attivazione QuickGELU. Se l'attention è la riunione in cui i token si scambiano notizie, l'MLP è il momento in cui ognuno ci ripensa da solo: nessun arco in profondità.`,
    },
    { id: "txt_add2", label: "+ residuo", role: "add", neurons: 12, depth: true, text: "Di nuovo la connessione residua: x + MLP(x)." },
    {
      repeat: SD.textLayers, of: ["txt_attn", "txt_add1", "txt_mlp", "txt_add2"], id: "txt_x12", label: `× ${SD.textLayers} layer`,
      text: `Il text encoder di CLIP ViT-L/14 ha ${SD.textLayers} layer uguali, ognuno con i suoi pesi: ne vedi uno vero, le lastre trasparenti stanno per gli altri.`,
    },
    {
      id: "txt_out", label: "Testo codificato", sub: `LayerNorm finale · ${SD.textTokens} × ${SD.textWidth}`, role: "norm", neurons: 12, depth: true, real: SD.textWidth,
      text: `L'ultima LayerNorm dà ${SD.textTokens} vettori da ${SD.textWidth} numeri: è ciò che Stable Diffusion usa del testo (last_hidden_state). Si calcola una volta sola per immagine e poi resta fisso: la U-Net lo rilegge a ogni passo, in ogni cross-attention (gli archi lunghi che partono da qui). Il text encoder è congelato: non si addestra insieme alla U-Net.\n\nScegli una parola come fetta («faro», per esempio): restano accesi solo i suoi archi, e si vede in quali posizioni dell'immagine entra a ogni livello della U-Net (archi illustrativi: vedi il blocco di attention).`,
    },
  ];
  const links = [
    { from: "txt_tokens", to: "txt_embed", pattern: "lookup" }, // il token sceglie la sua riga della tabella
    { from: "txt_embed", to: "txt_attn", pattern: "dense", toSlab: 0, density: 0.3, label: "LayerNorm e proiezioni query, key, value" },
    { within: "txt_attn", pattern: "attention", causal: true }, // solo verso il passato (maschera causale di CLIP)
    { from: "txt_attn", to: "txt_add1", pattern: "dense", density: 0.15, label: "proiezione di uscita" },
    { from: "txt_embed", to: "txt_add1", pattern: "residual" },
    { from: "txt_add1", to: "txt_mlp", pattern: "dense", density: 0.12, label: "LayerNorm e primo strato" },
    { from: "txt_mlp", to: "txt_add2", pattern: "dense", density: 0.1 },
    { from: "txt_add1", to: "txt_add2", pattern: "residual" },
    { from: "txt_add2", to: "txt_out", pattern: "identity" },
  ];
  return { sections, links };
}

// ------------------------------------------------------------------------------------------ U-Net

/** Forma vera e forma disegnata di una feature map: "64 × 64 · 320 canali · qui 8 × 8 · 1". */
const mapSub = (res, channels) => `${res} × ${res} · ${thousands(channels)} canali · qui ${side(res)} × ${side(res)} · ${shown(channels)}`;
/** Una sezione della griglia di un livello (canali come lastre, righe su y, colonne su z). */
const mapSection = ({ res, channels }, o) => ({ neurons: shown(channels), depth: grid(res), real: channels, ...o });

/** Testo del ResNet block di un livello: il primo lo spiega, gli altri dicono cosa cambia. */
function resText(L, { decoder, entering }) {
  const pixels = 512 / L.res;
  const ENTER = {
    down: "All'ingresso la griglia si è dimezzata (Downsample: una convoluzione con passo 2) e i canali sono cresciuti. ",
    "down-hidden": `All'ingresso la griglia si dimezza ancora (16 → ${L.res}): qui non si vede, perché la griglia ridotta resta 2 × 2. `,
    up: "All'ingresso la griglia è raddoppiata (Upsample: ogni posizione copiata in un blocco 2 × 2, poi una convoluzione 3 × 3). ",
    "up-hidden": `All'ingresso la griglia raddoppia (8 → ${L.res}): qui non si vede, perché la griglia ridotta era già 2 × 2. `,
  };
  const enter = ENTER[entering] ?? "";
  if (decoder) return `${enter}Ogni ResNet block del decoder riceve, affiancati ai suoi, i canali che l'encoder aveva prodotto a ${L.res} × ${L.res}: è la skip connection, l'arco sopra la U. Le skip della U-Net non sommano, concatenano: così i dettagli fini dell'inizio non si perdono nel viaggio fino al fondo.`;
  if (L.level === 0) return "Due convoluzioni 3 × 3 (con GroupNorm e SiLU) e una scorciatoia che somma l'ingresso all'uscita. In mezzo si aggiunge il vettore del passo t, uguale in ogni posizione: le linee che arrivano dal «MLP del tempo».";
  return `${enter}Ogni posizione ora riassume una zona più grande dell'immagine finale (${pixels} × ${pixels} pixel): la rete passa dai dettagli alla composizione.`;
}

/**
 * Head disegnate della cross-attention: una sola, con i pesi illustrativi di crossAttentionMap. Con due
 * head gli archi dal testo (lunghi, sopra tutta la U-Net) raddoppierebbero e coprirebbero la rete quando
 * si gira la camera. La self-attention ne mostra due.
 */
const CROSS_HEADS = 1;

/** Testo del blocco di attention di un livello (self-attention + cross-attention + feed-forward). */
function attentionText(L, { first }) {
  const positions = L.res * L.res;
  if (!first) return `Lo stesso blocco del primo livello sulle ${thousands(positions)} posizioni di questo: self-attention tra le posizioni (gli archi ad ansa, pesi simulati) e cross-attention verso i ${SD.textTokens} token del testo (gli archi lunghi da sinistra, schema illustrativo: ogni parola entra nella sua zona dell'immagine).`;
  const pairs = String(Math.round((positions * positions) / 1e5) / 10).replace(".", ",");
  return `Il blocco transformer della U-Net (il blocco viola della scena), in tre passi.\n\nSelf-attention: ogni posizione guarda tutte le altre, senza maschera, perché un'immagine non ha un prima e un dopo; così due zone lontane si accordano, come i due occhi dello stesso viso. A ${L.res} × ${L.res} le posizioni sono ${thousands(positions)}: circa ${pairs} milioni di coppie per head (gli archi ad ansa).\n\nCross-attention: ogni posizione «chiede» ai ${SD.textTokens} vettori del testo quali parole la riguardano. Query dall'immagine, key e value dal testo: gli archi lunghi che arrivano dal «Testo codificato», una head sola. Gli archi sono uno schema illustrativo, non calcolato: ogni posizione guarda la parola della sua zona nell'immagine del tab (la torre «faro», la scogliera «scogliera», cielo e mare «tramonto»). In Stable Diffusion succede davvero che la mappa di una parola si concentri dove compare la cosa che nomina, ma ogni posizione dà un po' di peso a tutti i ${SD.textTokens} token, anche a «un», «su» e a <inizio>.\n\nPoi un feed-forward (GEGLU) per ogni posizione. La self-attention è simulata: le due head disegnate seguono schemi fissi (la posizione accanto, la prima posizione); in un modello addestrato i pesi dipendono dal contenuto dell'immagine.`;
}

/** ResNet block e blocco di attention di un livello, più la ripetizione (× 2 nell'encoder, × 3 nel decoder). */
function level(p, L, { decoder = false, entering = null } = {}) {
  const count = decoder ? 3 : 2;
  const sections = [mapSection(L, { id: `${p}_res`, label: decoder ? "ResNet block + skip" : "ResNet block", sub: mapSub(L.res, L.channels), role: "conv", text: resText(L, { decoder, entering }) })];
  const links = [];
  if (L.attention) {
    sections.push(
      mapSection(L, {
        id: `${p}_attn`, label: "Self- e cross-attention", sub: `${L.res} × ${L.res} posizioni · ${SD.heads} head, qui 2 (cross: 1) · key e value dal testo`, role: "attention", heads: 2,
        text: attentionText(L, { first: L.level === 0 && !decoder }),
      }),
    );
    links.push(
      { from: `${p}_res`, to: `${p}_attn`, pattern: "dense", density: 0.25, label: "GroupNorm e proiezione 1 × 1" },
      { within: `${p}_attn`, pattern: "attention" }, // nessuna maschera: tutte le posizioni con tutte
      { from: "txt_out", to: `${p}_attn`, pattern: "attention", heads: CROSS_HEADS, weights: crossAttentionMap(side(L.res)), label: "cross-attention al testo (schema illustrativo)" }, // dai token alle posizioni
    );
  }
  const ids = sections.map((s) => s.id);
  sections.push({
    repeat: count, of: ids, id: `${p}_x${count}`, label: `× ${count} a ${L.res} × ${L.res}`,
    text: decoder
      ? `A ogni livello del decoder ci sono ${count} ResNet block (num_res_blocks + 1)${L.attention ? ", ognuno seguito dal blocco di attention" : ""}. Ne vedi uno: le lastre trasparenti stanno per gli altri.`
      : `A ogni livello dell'encoder ci sono ${count} ResNet block (num_res_blocks: 2)${L.attention ? ", ognuno seguito dal blocco di attention" : ""}. Ne vedi uno: la lastra trasparente sta per l'altro.`,
  });
  return { sections, links, first: ids[0], last: ids[ids.length - 1] };
}

function unet() {
  const [L0, , , L3] = UNET_LEVELS;
  const latent = grid(SD.latentSide);
  const sections = [
    {
      id: "un_zt", label: "Latente rumoroso z_t", sub: `4 × ${SD.latentSide} × ${SD.latentSide} · qui 4 × ${side(SD.latentSide)} × ${side(SD.latentSide)}`, role: "input", neurons: SD.latentChannels, depth: latent, real: SD.latentChannels,
      values: initialNoiseCrop(),
      text: `Il punto di partenza: rumore gaussiano nel latente del VAE (4 canali, ${SD.latentSide} × ${SD.latentSide} posizioni per un'immagine 512 × 512). È come la neve di un vecchio televisore: dentro non c'è ancora niente.\n\nValori veri: il rumore con cui parte l'azione «Generazione» del tab (seed ${GENERATION.seed}), l'angolo in alto a sinistra di ognuno dei 4 canali.`,
    },
    {
      id: "un_t", label: `Passo t = ${FIRST_T}`, sub: `seni e coseni · 8 dei ${SD.modelChannels} numeri`, role: "input", neurons: 8, depth: false, real: SD.modelChannels,
      values: timestepValues(FIRST_T), neuronNames: [...TIME_NAMES],
      text: `La rete deve sapere quanto rumore c'è: il passo t diventa ${SD.modelChannels} numeri, coseni e seni a frequenze diverse (la funzione timestep_embedding del codice di Stable Diffusion, la stessa del tab). Come le lancette di un orologio che girano a velocità diverse: insieme dicono l'ora esatta.\n\nValori veri per t = ${FIRST_T}, il primo dei ${GENERATION.steps} passi dell'azione «Generazione»: qui i numeri con le frequenze 1, 1/10, 1/100 e 1/1000.`,
    },
    {
      id: "un_tmlp", label: "MLP del tempo", sub: `${SD.modelChannels} → ${thousands(SD.timeEmbed)} → ${thousands(SD.timeEmbed)} · qui 8`, role: "mlp", neurons: 8, depth: false, real: SD.timeEmbed,
      text: `Due strati densi con la SiLU in mezzo trasformano i seni e coseni in un vettore di ${thousands(SD.timeEmbed)} numeri. Da qui il passo t raggiunge OGNI ResNet block, in tutte le posizioni della griglia (broadcast, le linee lunghe): ogni blocco lo somma ai suoi canali.`,
    },
    mapSection(L0, {
      id: "un_conv_in", label: "Conv di ingresso", sub: `3 × 3 · 4 → ${SD.modelChannels} canali · qui ${shown(L0.channels)}`, role: "conv",
      text: `Una convoluzione 3 × 3: ogni posizione guarda solo le 9 posizioni attorno, con gli stessi pesi su tutta la griglia, come una lente che scorre sull'immagine. Porta i 4 canali del latente a ${SD.modelChannels}.`,
    }),
  ];
  const links = [
    { from: "un_zt", to: "un_conv_in", pattern: "conv", kernel: 3, density: 0.2 },
    { from: "un_t", to: "un_tmlp", pattern: "dense", density: 0.5 },
  ];
  const resBlocks = [];
  const add = (part) => {
    sections.push(...part.sections);
    links.push(...part.links);
    resBlocks.push(part.first);
    return part;
  };

  // --- encoder: tre livelli con attention e il più piccolo senza; tra un livello e l'altro la griglia si
  // dimezza (Downsample: conv con passo 2) dentro il collegamento "down"
  let previous = "un_conv_in";
  let previousSide = side(L0.res);
  const encoderLast = [];
  for (const L of UNET_LEVELS) {
    const halves = side(L.res) < previousSide; // il dimezzamento si vede solo se la griglia ridotta cambia
    const part = add(level(`un_e${L.level}`, L, { entering: L.level === 0 ? null : halves ? "down" : "down-hidden" }));
    if (L.level === 0) links.push({ from: previous, to: part.first, pattern: "conv", density: 0.15 });
    else if (halves) links.push({ from: previous, to: part.first, pattern: "down", density: 0.5, label: "Downsample: conv con passo 2" });
    else links.push({ from: previous, to: part.first, pattern: "conv", density: 0.2, label: "Downsample (qui la griglia resta 2 × 2)" });
    previous = part.last;
    previousSide = side(L.res);
    encoderLast.push(part.last);
  }

  // --- centro della U: ResNet, attention (self + cross), ResNet
  const mid = (o) => mapSection(L3, o);
  sections.push(
    mid({ id: "un_m_res1", label: "Centro: ResNet block", sub: mapSub(L3.res, L3.channels), role: "conv", text: "Il fondo della U: qui ogni posizione vede una zona grande dell'immagine e si decide la composizione d'insieme (dove sta il faro, dove il mare)." }),
    mid({ id: "un_m_attn", label: "Centro: self- e cross-attention", sub: `${L3.res * L3.res} posizioni · ${SD.heads} head, qui 2 (cross: 1)`, role: "attention", heads: 2, text: "Anche al centro c'è un blocco di attention: le posizioni del fondo della U si guardano tra loro (self-attention, pesi simulati) e interrogano il testo (cross-attention, schema illustrativo). Qui ogni posizione riassume un quarto dell'immagine, quindi divide il peso tra più parole." }),
    mid({ id: "un_m_res2", label: "Centro: ResNet block", sub: mapSub(L3.res, L3.channels), role: "conv", text: "Il secondo ResNet block del centro chiude il fondo della U: da qui si risale." }),
  );
  links.push(
    { from: previous, to: "un_m_res1", pattern: "conv", density: 0.2 },
    { from: "un_m_res1", to: "un_m_attn", pattern: "dense", density: 0.25, label: "GroupNorm e proiezione 1 × 1" },
    { within: "un_m_attn", pattern: "attention" },
    { from: "txt_out", to: "un_m_attn", pattern: "attention", heads: CROSS_HEADS, weights: crossAttentionMap(side(L3.res)), label: "cross-attention al testo (schema illustrativo)" },
    { from: "un_m_attn", to: "un_m_res2", pattern: "conv", density: 0.2 },
  );
  resBlocks.push("un_m_res1", "un_m_res2");

  // --- decoder: dal livello più piccolo al più grande; tra un livello e l'altro la griglia raddoppia
  // (Upsample dentro il collegamento "up"); ogni livello riceve la skip connection dall'encoder
  previous = "un_m_res2";
  previousSide = side(L3.res);
  for (const L of [...UNET_LEVELS].reverse()) {
    const doubles = side(L.res) > previousSide;
    const part = add(level(`un_d${L.level}`, L, { decoder: true, entering: L.level === 3 ? null : doubles ? "up" : "up-hidden" }));
    if (doubles) links.push({ from: previous, to: part.first, pattern: "up", density: 0.5, label: "Upsample: copia 2 × 2 e conv" });
    else links.push({ from: previous, to: part.first, pattern: "conv", density: 0.2, label: L.level === 3 ? "" : "Upsample (qui la griglia resta 2 × 2)" });
    links.push({ from: encoderLast[L.level], to: part.first, pattern: "residual", label: "skip connection" });
    previous = part.last;
    previousSide = side(L.res);
  }
  sections.push({
    id: "un_eps", label: "Rumore previsto ε̂", sub: `4 × ${SD.latentSide} × ${SD.latentSide} · GroupNorm, SiLU, conv 3 × 3`, role: "output", neurons: SD.latentChannels, depth: latent, activation: "signed", real: SD.latentChannels,
    text: "L'ultima convoluzione torna a 4 canali: è il rumore che la U-Net crede ci sia in z_t, posizione per posizione, della stessa forma del latente.\n\nValori simulati, come tutta la U-Net: al primo passo, quando c'è quasi solo rumore, il rumore previsto da una rete vera somiglierebbe molto al latente in ingresso.",
  });
  links.push({ from: previous, to: "un_eps", pattern: "conv", density: 0.05 });

  // il passo t raggiunge ogni ResNet block, in tutte le posizioni (emb_layers: lo stesso vettore ovunque)
  for (const id of resBlocks) links.push({ from: "un_tmlp", to: id, pattern: "broadcast", density: 0.04, label: "passo t" });
  return { sections, links };
}

// ------------------------------------------------------------------------------------------ scheduler e VAE

/** Il VAE raddoppia la griglia: qui 8 → 16 (64 → 128 nel modello vero), e l'immagine resta 16 × 16. */
const IMAGE_SIDE = 2 * side(SD.latentSide);
const IMAGE_DEPTH = { grid: [IMAGE_SIDE, IMAGE_SIDE], label: "griglia dell'immagine" };
const VAE_SHOWN = 2; // 1 canale ogni 256 dei 512 del decoder del VAE

function decoderToPixels() {
  const latent = grid(SD.latentSide);
  const sections = [
    {
      id: "sch", label: "Scheduler DDIM", sub: "z_t − rumore previsto → z_t−1 · nessun peso", role: "data", neurons: SD.latentChannels, depth: latent,
      text: `Lo scheduler non ha pesi: con la formula di DDIM toglie da z_t la parte di rumore prevista (l'arco che arriva da z_t, sopra tutta la U-Net) e calcola z_t−1, un po' più pulito. Poi z_t−1 torna all'ingresso della U-Net con un t più piccolo: con le impostazioni del tab ${GENERATION.steps} volte, e a ogni passo la U-Net gira due volte per la classifier-free guidance.\n\nCome un restauratore che ripassa lo stesso quadro: ogni passata toglie un po' di sporco. Dopo l'ultimo passo resta z_0, il latente pulito.`,
    },
    {
      id: "vae_mid", label: "VAE decoder: ingresso e centro", sub: `4 → ${SD.vaeChannels} canali · ResNet, attention, ResNet, 3 ResNet · qui ${VAE_SHOWN}`, role: "attention", heads: 1, neurons: VAE_SHOWN, depth: latent, real: SD.vaeChannels,
      text: `Il latente pulito z_0 viene diviso per 0,18215 (scale_factor: nel training lo si era moltiplicato per avere varianza circa 1) ed entra nel decoder del VAE, una rete separata, addestrata prima e congelata. Una convoluzione porta i 4 canali a ${SD.vaeChannels}; poi due ResNet block con in mezzo un'attention a una sola head tra tutte le posizioni, e altri tre ResNet block, ancora a ${SD.latentSide} × ${SD.latentSide}. Pesi simulati.`,
    },
    {
      id: "vae_up", label: "VAE: upsample e 3 ResNet block", sub: `64 → 128 · ${SD.vaeChannels} canali · qui ${side(SD.latentSide)} → ${IMAGE_SIDE} · ${VAE_SHOWN}`, role: "conv", neurons: VAE_SHOWN, depth: IMAGE_DEPTH, real: SD.vaeChannels,
      text: "La griglia raddoppia (copia 2 × 2 e convoluzione) e tre ResNet block «inventano» i dettagli che il latente aveva solo riassunto. Nel modello vero i raddoppi sono tre, 64 → 128 → 256 → 512, ognuno seguito da tre ResNet block, mentre i canali scendono 512 → 256 → 128: qui se ne disegna uno.",
    },
    {
      id: "vae_out", label: "Immagine", sub: `512 × 512 × 3 (rosso, verde, blu) · qui ${IMAGE_SIDE} × ${IMAGE_SIDE}`, role: "output", neurons: 3, depth: IMAGE_DEPTH, activation: "signed", real: 3,
      text: `Gli altri due raddoppi, ognuno con i suoi tre ResNet block, poi GroupNorm, SiLU e una convoluzione 3 × 3 finale arrivano a 3 canali: rosso, verde e blu, con valori circa tra −1 e 1 (poi riportati a 0-255). Qui una sfera ogni ${512 / IMAGE_SIDE} × ${512 / IMAGE_SIDE} pixel.\n\nIl decoder del VAE si usa una volta sola, alla fine: per questo conviene far girare la diffusione nel latente, 48 volte più piccolo dei pixel (16.384 numeri contro 786.432).`,
    },
  ];
  const links = [
    { from: "un_eps", to: "sch", pattern: "identity", density: 0.35 },
    { from: "un_zt", to: "sch", pattern: "residual", label: "z_t entra nello scheduler" },
    { from: "sch", to: "vae_mid", pattern: "conv", density: 0.05, label: "dopo l'ultimo passo: z_0" },
    { within: "vae_mid", pattern: "attention" },
    { from: "vae_mid", to: "vae_up", pattern: "up", density: 0.3 },
    { from: "vae_up", to: "vae_out", pattern: "conv", density: 0.02, label: "altri due raddoppi, ResNet block e conv finale" },
  ];
  return { sections, links };
}

// ------------------------------------------------------------------------------------------ video

function videoBlock() {
  const cube = (o) => ({ neurons: 2, depth: VIDEO_GRID, ...o });
  const attention = (o) => cube({ role: "attention", heads: 1, ...o });
  const sections = [
    cube({
      id: "vid_in", label: "Video: patch e embedding", sub: "4 frame × 4 patch = 16 token · qui 3 numeri", role: "embedding", neurons: 3,
      text: "Un modello video lavora su un cubo di latenti: tempo × altezza × larghezza (nella scena del tab, 4 latenti 8 × 8 da 16 frame). Il cubo si taglia in patch, ogni patch diventa un vettore e riceve la sua posizione nello spazio e nel tempo.\n\nQui ogni lastra è la griglia frame × patch: una riga per frame (dall'alto, il frame 1), una colonna per patch. 16 token invece dei 64 della scena del tab e delle decine di migliaia dei modelli veri. È un modello a parte: non continua la U-Net qui a sinistra.",
    }),
    attention({
      id: "vid_spatial", label: "Attention spaziale", sub: "dentro ogni frame: lungo le righe",
      text: "Ogni token guarda solo i token dello STESSO frame, cioè della sua riga: un'immagine alla volta, come nella U-Net. Si capisce la composizione di ogni istante.\n\nGli archi mostrano chi può guardare chi: è la struttura vera dell'attention fattorizzata. Sono tutti uguali perché qui non c'è un modello addestrato: i pesi veri dipendono dal contenuto.",
    }),
    cube({ id: "vid_add1", label: "+ residuo", role: "add", text: "x + attention spaziale(x)." }),
    attention({
      id: "vid_temporal", label: "Attention temporale", sub: "la stessa patch negli altri frame: lungo le colonne",
      text: "Ogni token guarda la stessa patch negli ALTRI frame, cioè la sua colonna: così il faro resta al suo posto da un frame all'altro. È l'attention temporale di Video Diffusion Models, che la alterna a quella spaziale.\n\nI modelli aperti più recenti (HunyuanVideo, Wan 2.1) usano invece un'attention completa: ogni token guarda tutti gli altri, nello spazio e nel tempo. Più potente, ma con N token servono N² confronti. Archi con pesi uguali, come sopra.",
    }),
    cube({ id: "vid_add2", label: "+ residuo", role: "add", text: "x + attention temporale(x). Poi, come in ogni blocco transformer, un MLP per token con il suo residuo (non disegnato)." }),
    cube({
      id: "vid_out", label: "Rumore (o velocità) previsto", sub: "di nuovo patch del cubo · qui 3 numeri", role: "output", neurons: 3, activation: "signed",
      text: "L'ultimo strato riporta ogni token a una patch del cubo. Qui c'è un blocco solo: i modelli veri ne impilano molti (Wan 2.1: 30 nel modello da 1,3 miliardi di parametri, 40 in quello da 14) e lo scheduler li richiama per decine di passi. Alla fine il decoder del VAE 3D torna ai frame.",
    }),
  ];
  const links = [
    { from: "vid_in", to: "vid_spatial", pattern: "dense", density: 0.5 },
    { within: "vid_spatial", pattern: "attention", weights: videoAttentionMask("spatial") },
    { from: "vid_spatial", to: "vid_add1", pattern: "dense", density: 0.5 },
    { from: "vid_in", to: "vid_add1", pattern: "residual" },
    { from: "vid_add1", to: "vid_temporal", pattern: "dense", density: 0.5 },
    { within: "vid_temporal", pattern: "attention", weights: videoAttentionMask("temporal") },
    { from: "vid_temporal", to: "vid_add2", pattern: "dense", density: 0.5 },
    { from: "vid_add1", to: "vid_add2", pattern: "residual" },
    { from: "vid_add2", to: "vid_out", pattern: "dense", density: 0.5 },
  ];
  return { sections, links };
}

// ------------------------------------------------------------------------------------------ fonti

const CV = "https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229";
const CLIP_CODE = "https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6";

export const NETWORK_SOURCES = Object.freeze([
  { title: "CompVis/stable-diffusion: v1-inference.yaml", url: `${CV}/configs/stable-diffusion/v1-inference.yaml`, note: "U-Net: 4 canali in ingresso e in uscita, model_channels 320, channel_mult [1, 2, 4, 4], num_res_blocks 2, attention_resolutions [4, 2, 1], 8 head, context_dim 768. VAE: z_channels 4, ch 128, ch_mult [1, 2, 4, 4]. scale_factor 0,18215." },
  { title: "CompVis/stable-diffusion: openaimodel.py (la U-Net)", url: `${CV}/ldm/modules/diffusionmodules/openaimodel.py#L413-L741`, note: "time_embed 320 → 1.280 → 1.280 con SiLU; ResBlock con il passo t; Downsample con passo 2; Upsample nearest + conv; 3 ResBlock per livello del decoder; skip connection come concatenazione (th.cat)." },
  { title: "CompVis/stable-diffusion: attention.py", url: `${CV}/ldm/modules/attention.py#L196-L261`, note: "SpatialTransformer: GroupNorm, proiezione 1 × 1, self-attention, cross-attention verso il testo, feed-forward GEGLU, residuo." },
  { title: "CompVis/stable-diffusion: model.py (il decoder del VAE)", url: `${CV}/ldm/modules/diffusionmodules/model.py#L461-L560`, note: "Conv di ingresso, blocco centrale ResNet · attention · ResNet, 3 ResnetBlock per livello, upsample, conv finale a 3 canali." },
  { title: "CompVis/stable-diffusion: util.py (timestep_embedding)", url: `${CV}/ldm/modules/diffusionmodules/util.py#L151-L171`, note: "Embedding sinusoidale del passo: metà coseni e metà seni, max_period 10.000 (la stessa formula di math.js del tab)." },
  { title: "CompVis/stable-diffusion: FrozenCLIPEmbedder", url: `${CV}/ldm/modules/encoders/modules.py#L137-L162`, note: "Tokenizer con riempimento fino a 77 token; la U-Net riceve last_hidden_state (77 × 768)." },
  { title: "openai/CLIP: model.py", url: `${CLIP_CODE}/clip/model.py#L171-L356`, note: "Text transformer: maschera causale (build_attention_mask), LayerNorm prima di attention e MLP, MLP largo 4 volte con QuickGELU, LayerNorm finale (ln_final), embedding di posizione imparato; head = larghezza / 64." },
  { title: "openai/CLIP: simple_tokenizer.py", url: `${CLIP_CODE}/clip/simple_tokenizer.py#L61-L76`, note: "Vocabolario di 49.408 voci: 512 byte, 48.894 fusioni BPE e 2 token speciali." },
  { title: "Rombach et al. (2022), High-Resolution Image Synthesis with Latent Diffusion Models", url: "https://arxiv.org/abs/2112.10752", note: "Diffusione nel latente del VAE, U-Net con cross-attention verso il testo." },
  { title: "Radford et al. (2021), Learning Transferable Visual Models From Natural Language Supervision (CLIP)", url: "https://arxiv.org/abs/2103.00020", note: "Il text encoder di ViT-L/14: 12 layer, larghezza 768, 12 head." },
  { title: "Song, Meng, Ermon (2020), Denoising Diffusion Implicit Models", url: "https://arxiv.org/abs/2010.02502", note: "Il passo dello scheduler DDIM." },
  { title: "Ronneberger, Fischer, Brox (2015), U-Net", url: "https://arxiv.org/abs/1505.04597", note: "L'architettura a U con le skip connection." },
  { title: "Ho et al. (2022), Video Diffusion Models", url: "https://arxiv.org/abs/2204.03458", note: "Attention spaziale seguita da attention temporale (fattorizzata)." },
  { title: "Wan 2.1: README (architettura)", url: "https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/README.md#2-video-diffusion-dit", note: "Diffusion transformer video: 30 layer (1,3B) o 40 (14B), testo T5 con cross-attention in ogni blocco." },
  { title: "Wan 2.1: model.py (WanSelfAttention) e configurazioni", url: "https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L105-L160", note: "Self-attention su tutti i token del video (F × H × W, con RoPE 3D); le configurazioni t2v_1_3B e t2v_14B hanno window_size (−1, −1), cioè nessuna finestra locale: attention completa." },
  { title: "HunyuanVideo (codice e report)", url: "https://github.com/Tencent/HunyuanVideo", note: "VAE 3D causale (4× tempo, 8× spazio, 16 canali) e Full Attention su tutti i token." },
  { title: "Tang et al. (2022), What the DAAM: Interpreting Stable Diffusion Using Cross Attention", url: "https://arxiv.org/abs/2210.04885", note: "Le mappe di cross-attention di Stable Diffusion, parola per parola, si concentrano dove compare la cosa nominata (le mappe dei nomi segmentano gli oggetti): da qui l'idea dello schema illustrativo degli archi, che però non viene da un modello." },
]);

// ------------------------------------------------------------------------------------------ specifica

/** Costruisce la specifica completa (pura e deterministica: stessi dati, stessa rete). */
export function buildDiffusionNetworkSpec() {
  const parts = [textEncoder(), unet(), decoderToPixels(), videoBlock()];
  return {
    title: "Rete neurale di Stable Diffusion 1.x e di un blocco video (ridotte)",
    intro: "La pipeline del tab disegnata neurone per neurone, da sinistra a destra: il text encoder CLIP legge il prompt, la U-Net toglie il rumore dal latente, lo scheduler e il decoder del VAE arrivano ai pixel; in fondo, a parte, un blocco di un denoiser video.\n\nGira la camera («Guarda lo spessore»): nel testo l'attention collega ogni token ai precedenti, nella U-Net le convoluzioni guardano i vicini della griglia, la self-attention collega posizioni lontane e la cross-attention porta le parole del prompt dentro l'immagine; nel video l'attention lavora dentro i frame e tra i frame. La fascia viola sopra la U-Net sono gli archi della cross-attention: scegli una parola in «Fetta» («faro», per esempio) e restano solo i suoi.",
    scale: `1 sfera = 1 numero. U-Net: griglia divisa per ${GRID_DIVISOR} in ogni lato (64 × 64 → 8 × 8, 32 × 32 → 4 × 4, 16 × 16 → 2 × 2); il livello più piccolo (8 × 8) resta 2 × 2, quindi il terzo dimezzamento e il primo raddoppio non si vedono. 1 canale ogni ${CHANNEL_DIVISOR}: 1, 2, 4, 4 lastre invece di 320, 640, 1.280, 1.280 (proprio channel_mult [1, 2, 4, 4]); 2 delle 8 head (1 nella cross-attention); a ogni livello un ResNet block e un blocco di attention disegnati su 2 (encoder) o 3 (decoder), quindi una skip connection per livello invece di 12 e il passo t verso 10 ResNet block invece di 22. Testo: 10 dei 77 token, 12 dei 768 numeri (48 dei 3.072 nell'MLP), 4 delle 12 head, 1 dei 12 layer. VAE: 2 dei 512 canali, un raddoppio della griglia disegnato dei tre veri, immagine ${IMAGE_SIDE} × ${IMAGE_SIDE} invece di 512 × 512. Video: 16 token, pochi numeri per token, un blocco solo.`,
    note: `Valori veri, calcolati qui con le funzioni del tab (math.js): il rumore iniziale z_T dell'azione «Generazione» (seed ${GENERATION.seed}) e i seni e coseni del passo t = ${FIRST_T}. Tutto il resto è una simulazione didattica con un seme: il denoiser del tab è simulato (nel browser non c'è una U-Net addestrata), e così attivazioni e pesi della self-attention. Gli archi della cross-attention sono uno schema illustrativo scelto sulla scena del tab, non calcolato: ogni posizione guarda la parola della sua zona (la torre «faro», la scogliera «scogliera», cielo e mare «tramonto»), divisa in proporzione all'area dove una posizione ne copre più d'una. Nel blocco video gli archi mostrano la struttura vera dell'attention fattorizzata (chi può guardare chi) con pesi tutti uguali. Il Diffusion Transformer (vista «DiT» del tab) sostituirebbe la U-Net con blocchi transformer sulle patch del latente, come il blocco video senza il tempo.`,
    seed: 23,
    axes: {
      x: "flusso (dal prompt e dal rumore fino all'immagine; nella U-Net e nel VAE una lastra per canale)",
      y: "neuroni e righe (nel testo i numeri del vettore di ogni token; nella U-Net e nel VAE le righe della griglia; nel video i frame)",
      z: "token e colonne (nel testo i token del prompt; nella U-Net e nel VAE le colonne della griglia; nel video le patch)",
    },
    depth: { count: TEXT_TOKENS.length, label: "token del prompt", names: [...TEXT_TOKENS] },
    sections: parts.flatMap((p) => p.sections),
    links: parts.flatMap((p) => p.links),
    groups: [
      { label: "Text encoder CLIP", from: "txt_tokens", to: "txt_out", color: "#c4b5fd", text: "Il text encoder di CLIP ViT-L/14, circa 123 milioni di parametri. Legge il prompt una volta sola. È un transformer come il nostro GPT: 12 layer, 768 numeri per token, attention causale. Congelato: non si addestra con la U-Net." },
      { label: "U-Net: il denoiser (circa 860 milioni di parametri)", from: "un_zt", to: "un_eps", color: "#fcd34d", text: "Riceve il latente rumoroso z_t, il passo t e il testo; restituisce il rumore previsto, della stessa forma del latente. Gira a ogni passo del campionatore." },
      { label: "Encoder: la griglia si dimezza", from: "un_conv_in", to: "un_e3_x2", color: "#fdba74", text: "64 → 32 → 16 → 8 (qui 8 → 4 → 2 → 2) mentre i canali salgono 320 → 640 → 1.280: dai dettagli alla composizione." },
      { label: "Centro", from: "un_m_res1", to: "un_m_res2", color: "#fdba74", text: "ResNet block, attention e ResNet block a 8 × 8 (qui 2 × 2)." },
      { label: "Decoder: la griglia raddoppia", from: "un_d3_res", to: "un_eps", color: "#fdba74", text: "8 → 16 → 32 → 64 (qui 2 → 2 → 4 → 8). A ogni livello arrivano, con le skip connection, i canali dell'encoder della stessa risoluzione." },
      { label: "Scheduler e VAE decoder", from: "sch", to: "vae_out", color: "#7dd3fc", text: "Lo scheduler toglie il rumore previsto (a ogni passo); alla fine il decoder del VAE porta il latente ai pixel." },
      { label: "Video: un blocco a parte", from: "vid_in", to: "vid_out", color: "#67e8f9", text: "Un blocco di un denoiser video, un modello diverso dalla U-Net: token spazio-temporali (una riga per frame, una colonna per patch), attention spaziale e temporale." },
    ],
    sources: NETWORK_SOURCES.map((s) => ({ ...s })),
  };
}

export const networkSpec = buildDiffusionNetworkSpec();
