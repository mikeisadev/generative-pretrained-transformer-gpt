/**
 * Indice della ricerca globale per il tab "Audio e musica" (contratto in ARCHITECTURE.md,
 * "Ricerca globale, link diretti e glossario").
 *
 * Solo dati leggeri: viene caricato anche se il tab non è mai stato aperto, quindi niente three.js,
 * niente scena e niente calcoli del laboratorio. I passi vengono dal Percorso (content.js, stessi id
 * del Menu); azioni, viste e livelli dello zoom dal catalogo (catalog.js, gli stessi nomi del tab).
 * Gli id sono quelli che index.js → goto() riconosce.
 */
import { ACTIONS, LEVELS, VIEWS } from "./catalog.js";
import { STEPS } from "./content.js";

/**
 * Sinonimi e parole che uno studente potrebbe cercare per arrivare a un passo (italiano e inglese).
 * La ricerca guarda anche titolo e testo del passo: qui solo ciò che lì non c'è o è scritto diverso.
 */
const STEP_KEYWORDS = {
  "sound.wave": ["waveform", "onda sonora", "campioni", "samples", "sample rate", "frequenza di campionamento", "16 kHz", "Nyquist", "sintesi additiva"],
  "sound.vs-text": ["audio vs testo", "discreto", "continuo", "tokenizzare l'audio", "audio tokens", "perché servono i token"],
  "freq.frames": ["frame", "finestra", "window", "Hann", "hop length", "sovrapposizione", "overlap"],
  "freq.fft": ["Fourier", "FFT", "STFT", "spectrogram", "trasformata di Fourier", "spettro", "frequenze", "armoniche", "prisma"],
  "freq.mel": ["Mel", "spettrogramma Mel", "mel spectrogram", "log-mel", "filterbank", "banco di filtri", "decibel", "percezione", "librosa", "Slaney"],
  "codec.encoder": ["codec", "neural codec", "EnCodec", "SoundStream", "compressione", "convoluzioni", "convolution", "75 frame al secondo"],
  "codec.rvq": ["RVQ", "residual vector quantization", "codebook", "quantizzazione", "k-means", "centroidi", "residuo", "token audio"],
  "codec.bitrate": ["bitrate", "kbps", "bit al secondo", "compressione", "quantizer dropout", "qualità"],
  "codec.decoder": ["decoder", "ricostruzione", "Griffin-Lim", "fase", "discriminatore", "GAN", "addestramento", "training"],
  "gen.next": ["next token", "prossimo token", "autoregressivo", "MusicGen", "transformer", "attention causale", "teste di uscita"],
  "gen.delay": ["delay pattern", "ritardi", "sfasamento", "interleaving", "codebook in parallelo", "MusicGen"],
  "gen.text": ["T5", "text encoder", "cross-attention", "prompt", "condizionamento", "conditioning", "classifier-free guidance", "CFG"],
  "gen.toy": ["simulazione", "modello a conteggi", "Markov", "temperatura", "temperature", "campionamento", "sampling", "seme", "seed"],
  "asr.input": ["Whisper", "speech recognition", "riconoscimento vocale", "ASR", "log-Mel", "30 secondi", "80 bande"],
  "asr.encoder": ["Whisper encoder", "conv1d", "convoluzioni", "bidirezionale", "positional embedding", "codifica di posizione"],
  "asr.decoder": ["Whisper decoder", "trascrizione", "transcription", "traduzione", "translate", "token speciali", "special tokens", "sottotitoli"],
  "family.audiolm": ["AudioLM", "token semantici", "token acustici", "w2v-BERT", "semantic tokens", "acoustic tokens"],
  "family.valle": ["VALL-E", "TTS", "text to speech", "sintesi vocale", "clonazione della voce", "voice cloning", "fonemi"],
  "family.diffusion": ["Stable Audio", "diffusione", "diffusion", "latent diffusion", "VAE", "spazio latente", "DiT"],
};

const steps = STEPS.flatMap((section) =>
  section.children.map((item) => ({
    id: item.id,
    kind: "step",
    title: item.title,
    text: item.text,
    keywords: [section.title, ...(STEP_KEYWORDS[item.id] ?? [])],
  })),
);

const ACTION_KEYWORDS = {
  sound: ["animazione", "campionamento", "FFT", "spettrogramma", "Mel", "log-Mel", "dal suono ai numeri"],
  rvq: ["animazione", "codec", "codebook", "RVQ", "k-means", "errore", "decoder", "ricostruzione"],
  generate: ["animazione", "MusicGen", "generare musica", "music generation", "delay pattern", "cross-attention", "prompt"],
  whisper: ["animazione", "Whisper", "trascrizione", "speech to text", "riconoscimento vocale", "token speciali"],
};

const actions = Object.entries(ACTIONS).map(([key, a]) => ({
  id: a.id,
  kind: "action",
  title: a.title,
  text: a.summary,
  keywords: [a.category, ...ACTION_KEYWORDS[key]],
}));

const VIEW_KEYWORDS = {
  overview: ["vista d'insieme", "overview", "tutta la pipeline", "mappa", "centra"],
  delayCore: ["MusicGen", "delay pattern", "transformer", "generazione"],
  rvq: ["codec", "codebook", "token", "RVQ"],
  whisperAll: ["Whisper", "speech to text", "riconoscimento vocale", "ASR"],
};

// "audio" e "musica" tra le parole chiave: viste e livelli hanno nomi simili negli altri tab
// ("panoramica audio" deve trovare questa).
const views = [
  ...VIEWS.map((v) => ({ id: v.id, kind: "view", title: v.label, text: v.text, keywords: ["vista", "view", "audio", "musica", ...VIEW_KEYWORDS[v.id]] })),
  ...LEVELS.map((l) => ({ id: l.id, kind: "view", title: `Zoom: ${l.label}`, text: l.text, keywords: ["zoom semantico", "livello", "level of detail", "LOD", "audio", "musica", l.label] })),
];

// Rappresentazioni del nucleo (core/representations.js: REP_LINKS), stessi id in ogni tab: goto le apre
// con reps.goto. Titoli con il nome del tab, perché ogni tab ha una "Rete neurale" e un "Cervello".
const representations = [
  {
    id: "rete",
    kind: "view",
    title: "Rete neurale (Audio e musica)",
    text: "Le reti del tab neurone per neurone, da sinistra a destra, con il tempo in profondità: EnCodec (convoluzioni con passo, quantizzazione residua, decoder), il codec del tab con i numeri veri (spettro, 4 codebook in cascata, voce scelta a ogni frame), MusicGen (4 codebook sfasati dal delay pattern, attention causale, cross-attention al testo) e Whisper (log-Mel vero, convoluzioni, attention bidirezionale nell'encoder, decoder causale con cross-attention).",
    keywords: [
      "rete neurale", "neural network", "neuroni", "rappresentazione", "profondità", "spessore", "3D", "frame", "tempo",
      "EnCodec", "SoundStream", "codec neurale", "convoluzione", "convolution", "stride", "passo", "LSTM", "latente",
      "RVQ", "residual vector quantization", "codebook", "voci del codebook", "residuo", "k-means", "token audio",
      "MusicGen", "delay pattern", "attention causale", "cross-attention", "T5", "teste lineari",
      "Whisper", "log-Mel", "attention bidirezionale", "encoder", "decoder", "codifica di posizione", "sinusoidi",
    ],
  },
  {
    id: "cervello",
    kind: "view",
    title: "Cervello (Audio e musica)",
    text: "Le analogie tra l'audio e la via uditiva del cervello, ognuna con cosa hanno in comune, dove non regge e le fonti: dalla coclea (cellule ciliate, membrana basilare) al nervo acustico, al tronco encefalico, al talamo e alla corteccia uditiva fino all'area di Wernicke; la scala Mel e le bande critiche, la musica e la dopamina; il modello 3D con le regioni, i percorsi e il microcircuito di neuroni.",
    keywords: [
      "cervello", "brain", "neuroscienze", "analogie", "via uditiva", "orecchio", "udito", "coclea", "cellule ciliate",
      "membrana basilare", "tonotopia", "nervo acustico", "ganglio spirale", "tronco encefalico", "nucleo cocleare",
      "oliva superiore", "collicolo inferiore", "talamo", "genicolato mediale", "MGN", "radiazione uditiva",
      "corteccia uditiva", "A1", "giro di Heschl", "giro temporale superiore", "Wernicke", "Broca", "fascicolo arcuato",
      "bande critiche", "scala Mel", "codifica efficiente", "dopamina", "striato", "aspettativa musicale", "neuroprotesi",
    ],
  },
];

export const searchIndex = [...steps, ...actions, ...views, ...representations];
