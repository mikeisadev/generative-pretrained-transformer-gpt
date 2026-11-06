/**
 * Indice della ricerca globale per il tab "Immagini e video" (contratto in ARCHITECTURE.md,
 * "Ricerca globale, link diretti e glossario").
 *
 * Solo dati leggeri: questo modulo viene caricato anche se il tab non è mai stato aperto, quindi
 * niente three.js né moduli della scena. I passi vengono dal Percorso (content.js, stessi id del
 * Menu); azioni e viste hanno gli stessi id usati da index.js (player.register e VIEW_IDS).
 */
import { PATH } from "./content.js";

/**
 * Sinonimi e parole che uno studente potrebbe cercare per arrivare a un passo (italiano e inglese).
 * La ricerca guarda già titolo e testo del passo: qui solo ciò che lì manca o è scritto diverso.
 */
const STEP_KEYWORDS = {
  "idea.cosa": ["diffusion model", "diffusione", "rumore", "noise", "denoising", "togliere il rumore", "generare immagini", "image generation", "inchiostro"],
  "idea.testo": ["autoregressivo", "autoregressive", "parallelo", "token per token", "GPT", "passi del campionatore", "sampling steps"],
  "idea.transformer": ["transformer", "CLIP", "text encoder", "DiT", "attention", "dove sta il transformer"],
  "forward.formula": ["processo diretto", "forward process", "aggiungere rumore", "add noise", "q_sample", "alpha bar", "ᾱ", "gaussiano", "gaussian noise", "x_t"],
  "forward.schedule": ["schedule", "noise schedule", "beta", "β", "lineare", "linear schedule", "T = 1000", "SNR", "signal to noise"],
  "forward.training": ["addestramento", "training", "loss", "MSE", "errore quadratico", "predire il rumore", "epsilon", "ε"],
  "reverse.step": ["processo inverso", "reverse process", "denoising", "DDIM", "DDPM", "sampler", "campionatore", "scheduler", "passo"],
  "reverse.cfg": ["CFG", "classifier-free guidance", "guidance scale", "scala di guidance", "prompt negativo", "negative prompt", "7.5"],
  "reverse.flow": ["rectified flow", "flow matching", "SD3", "Stable Diffusion 3", "linea retta", "velocità", "velocity"],
  "arch.text": ["tokenizer", "text encoder", "CLIP", "prompt", "77 token", "embedding del testo", "text embedding"],
  "arch.latent": ["VAE", "autoencoder", "latente", "latent space", "spazio latente", "compressione", "64×64×4", "encoder", "decoder"],
  "arch.unet": ["U-Net", "UNet", "encoder-decoder", "skip connection", "ResNet", "convoluzione", "convolution", "downsampling", "upsampling"],
  "arch.cross": ["cross-attention", "cross attention", "condizionamento", "conditioning", "query", "key", "value", "testo nell'immagine"],
  "arch.timestep": ["timestep", "passo t", "time embedding", "embedding sinusoidale", "sinusoidal", "seni e coseni"],
  "dit.patch": ["DiT", "Diffusion Transformer", "patch", "patchify", "token dell'immagine", "ViT", "vision transformer"],
  "dit.adaln": ["adaLN", "adaLN-Zero", "LayerNorm adattiva", "adaptive layer norm", "scale shift gate", "condizionamento"],
  "dit.sd3": ["SD3", "Stable Diffusion 3", "MMDiT", "multimodal", "attention congiunta", "joint attention", "testo e immagine"],
  "video.ipotesi": ["video", "frame", "fotogrammi", "un frame alla volta", "coerenza", "sfarfallio", "flicker"],
  "video.vdm": ["Video Diffusion Models", "U-Net 3D", "3D", "attention temporale", "temporal attention", "fattorizzata"],
  "video.sora": ["Sora", "OpenAI", "patch spazio-temporali", "spacetime patches", "world simulator", "diffusion transformer video"],
  "video.aperti": ["modelli aperti", "open source", "HunyuanVideo", "Wan", "CogVideoX", "VAE 3D", "causale", "causal VAE", "compressione temporale"],
  "video.attention": ["attention spaziale", "attention temporale", "full attention", "attention 3D", "spatial attention", "temporal attention", "costo quadratico"],
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

/** Le azioni animate (stessi id di player.register in action*.js e videoActions.js). */
const actions = [
  {
    id: "forward",
    kind: "action",
    title: "Processo diretto e training",
    text: "Animazione: la formula vera di DDPM aggiunge rumore a un'immagine per vari t (x_t = √ᾱ·x_0 + √(1−ᾱ)·ε), poi come questo diventa un passo di training.",
    keywords: ["aggiungere rumore", "add noise", "forward process", "training", "addestramento", "DDPM", "animazione"],
  },
  {
    id: "generate",
    kind: "action",
    title: "Generazione: dal rumore all'immagine",
    text: "Animazione: il ciclo di denoising con DDIM e classifier-free guidance, passo dopo passo, fino all'immagine. Denoiser simulato (simulazione didattica).",
    keywords: ["generazione", "generation", "sampling", "campionamento", "denoising", "CFG", "guidance", "prompt", "text-to-image", "animazione"],
  },
  {
    id: "vae",
    kind: "action",
    title: "Dal latente ai pixel (VAE)",
    text: "Animazione: il decoder del VAE trasforma 4 × 64 × 64 numeri in 512 × 512 × 3 pixel. Perché si lavora nel latente.",
    keywords: ["VAE", "decoder", "latente", "latent", "pixel", "upsampling", "scale factor", "animazione"],
  },
  {
    id: "inside",
    kind: "action",
    title: "Un passo dentro il denoiser",
    text: "Animazione: cosa succede in una sola chiamata alla U-Net (o al DiT, se è la vista attiva): ResNet block, attention, skip connection, passo t, testo.",
    keywords: ["U-Net", "UNet", "DiT", "denoiser", "dentro la rete", "ResNet", "skip connection", "adaLN", "patch", "animazione"],
  },
  {
    id: "video",
    kind: "action",
    title: "Video: tutti i frame insieme",
    text: "Animazione: il cubo spazio-temporale, il VAE 3D causale, le patch, attention spaziale contro temporale e la coerenza tra i frame.",
    keywords: ["video", "frame", "fotogrammi", "VAE 3D", "patch spazio-temporali", "attention temporale", "Sora", "animazione"],
  },
];

/** Viste del pannello "Viste" e livelli dello zoom (stessi id di VIEW_IDS in index.js). */
const views = [
  {
    id: "unet",
    kind: "view",
    title: "Stable Diffusion: U-Net",
    text: "Mostra la U-Net come denoiser: encoder-decoder con skip connection, ResNet block, self- e cross-attention.",
    keywords: ["U-Net", "UNet", "Stable Diffusion", "denoiser", "vista"],
  },
  {
    id: "dit",
    kind: "view",
    title: "Diffusion Transformer (DiT)",
    text: "Mostra il DiT come denoiser: il latente diventa una sequenza di patch, blocchi transformer con adaLN, come nel nostro GPT.",
    keywords: ["DiT", "Diffusion Transformer", "transformer", "patch", "denoiser", "vista"],
  },
  {
    id: "pipeline",
    kind: "view",
    title: "Tutta la pipeline (Blocchi)",
    text: "Vista d'insieme: dal prompt all'immagine, un blocco per componente; sotto, la parte video.",
    keywords: ["pipeline", "blocchi", "blocks", "vista d'insieme", "overview", "schema", "mappa", "Stable Diffusion"],
  },
  {
    id: "video",
    kind: "view",
    title: "Video: latenti spazio-temporali",
    text: "La parte video della scena: cubo di latenti, patch spazio-temporali, attention nello spazio e nel tempo.",
    keywords: ["video", "cubo", "latenti", "patch spazio-temporali", "frame", "vista"],
  },
  {
    id: "macro",
    kind: "view",
    title: "Panoramica: immagini e video in sei blocchi",
    text: "Lo schema più astratto: Testo → Denoiser × T passi → Immagine, e lo stesso per il video.",
    keywords: ["panoramica", "overview", "zoom out", "schema", "diagramma a blocchi", "vista d'insieme"],
  },
  {
    id: "detail",
    kind: "view",
    title: "Dettaglio: i componenti da vicino",
    text: "Lo zoom più vicino: i blocchi interni del denoiser, il latente, il passo t, il VAE (o il cubo del video se stai guardando la parte video).",
    keywords: ["dettaglio", "detail", "zoom in", "da vicino", "componenti", "blocchi interni"],
  },
];

export const searchIndex = [...steps, ...actions, ...views];
