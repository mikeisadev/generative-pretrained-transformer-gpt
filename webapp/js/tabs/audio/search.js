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

export const searchIndex = [...steps, ...actions, ...views];
