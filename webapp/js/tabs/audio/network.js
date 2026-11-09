/**
 * Rappresentazione "Rete neurale" del tab Audio e musica: la specifica (soli dati) che il renderer del
 * nucleo disegna in orizzontale con la profondità (js/core/neuralNet/, formato in ARCHITECTURE.md,
 * "Rete neurale orizzontale"). La costruisce buildAudioNetworkSpec(lab, { style }) con i numeri che il
 * laboratorio del tab (lab.js) ha calcolato davvero nel browser; index.js la passa a
 * core/representations.js.
 *
 * Quattro reti in fila, da sinistra a destra, nell'ordine del Percorso (punti 3, 4 e 5):
 *   EnCodec 24 kHz    un codec neurale vero: encoder convoluzionale con i passi che accorciano il tempo,
 *                     quantizzazione residua (RVQ) con codebook da 1.024 voci, decoder simmetrico
 *   codec del tab     il codec giocattolo (codec.js) con i NUMERI VERI: campioni, spettro di ogni frame,
 *                     4 codebook da 64 voci in cascata (ognuno sul residuo del precedente), voce scelta e
 *                     somma; i collegamenti verso i codebook portano i numeri veri delle voci (centroidi)
 *   MusicGen small    il transformer che genera i token: i 4 codebook in ingresso sfasati dal delay
 *                     pattern (i token VERI del codec del tab), attention causale, cross-attention al
 *                     testo di T5, 24 layer, 4 teste d'uscita con le probabilità del modello a conteggi
 *   Whisper tiny      log-Mel VERO del brano → 2 convoluzioni (la seconda con passo 2) → encoder con
 *                     attention bidirezionale × 4 → decoder con attention causale tra i token del testo e
 *                     cross-attention ai frame dell'encoder × 4
 *
 * In profondità c'è il tempo: 12 frame da 20 ms. Nel codec del tab, in MusicGen e in Whisper sono gli
 * STESSI 240 ms del brano (la finestra comincia 6 frame prima dell'attacco della seconda nota, così il
 * codebook 1 cambia voce a metà), quindi una fetta della scheda è lo stesso istante in tutte le reti.
 *
 * Numeri dell'architettura: dalle fonti del tab (content.js, upstream.js) e dal codice ufficiale letto
 * il 9 novembre 2026 (ramo main; citato in NETWORK_SOURCES): encodec/modules/seanet.py e model.py,
 * audiocraft (model_scale/small.yaml, musicgen_lm.yaml, lm/default.yaml, lm.py, modules/transformer.py,
 * conditioners.py, docs/MUSICGEN.md), whisper/model.py e audio.py, i valori predefiniti di Whisper tiny
 * in transformers e whisper.cpp. Ricontrollati in modo indipendente (stessi file, stesso ramo) dalla verifica.
 *
 * Modulo puro: niente three.js né DOM (lo provano i test in tests/audio-reps.test.mjs).
 */
import { fmt } from "../../core/dom.js";
import { CODEC } from "./codec.js";
import { melFilterbank, WHISPER } from "./dsp.js";
import { formatPercent } from "./lab.js";
import { SAMPLE_RATE } from "./song.js";

// ------------------------------------------------------------------------------------------ costanti

/** EnCodec a 24 kHz (encodec/modules/seanet.py e model.py, valori predefiniti del modello 24 kHz). */
export const ENCODEC = Object.freeze({
  sampleRate: 24000,
  encoderStrides: Object.freeze([2, 4, 5, 8]), // ratios [8, 5, 4, 2] usati al contrario dall'encoder
  decoderStrides: Object.freeze([8, 5, 4, 2]),
  hop: 320, // 2 · 4 · 5 · 8
  frameRate: 75, // 24.000 / 320
  channels: Object.freeze([32, 64, 128, 256, 512]), // n_filters 32, raddoppiati a ogni passo
  dimension: 128,
  lstmLayers: 2,
  codebookSize: 1024,
  bandwidths: Object.freeze([1.5, 3, 6, 12, 24]), // kbps
  codebooks: Object.freeze([2, 4, 8, 16, 32]), // codebook per banda
});

/** MusicGen small (audiocraft: model_scale/small.yaml, musicgen_lm.yaml, lm/default.yaml, text2music.yaml). */
export const MUSICGEN = Object.freeze({
  layers: 24,
  dim: 1024,
  heads: 16,
  ffn: 4096, // hidden_scale 4
  codebooks: 4,
  card: 2048,
  frameRate: 50,
  delays: Object.freeze([0, 1, 2, 3]),
  textDim: 768, // t5-base (MODELS_DIMS in conditioners.py)
  cfg: 3,
});

/** Whisper tiny (whisper/model.py; valori predefiniti "openai/whisper-tiny" di transformers e di whisper.cpp). */
export const WHISPER_TINY = Object.freeze({
  layers: 4,
  width: 384,
  heads: 6,
  mlp: 1536, // 4 × larghezza
  vocab: 51865,
  audioPositions: 1500,
  textPositions: 448,
  largeLayers: 32,
});

/** Frame della finestra disegnata (20 ms ciascuno: 240 ms) e posizione dell'attacco della nota nella finestra. */
export const WINDOW_FRAMES = 12;
const ONSET_AT = 6;
/** Bin dello spettro del codec del tab disegnati: uno ogni 16, cioè 0, 250, …, 4.000 Hz (17 dei 257). */
export const FEATURE_STEP = 16;
export const FEATURE_BINS = Object.freeze(Array.from({ length: Math.floor((CODEC.bins - 1) / FEATURE_STEP) + 1 }, (_, i) => i * FEATURE_STEP));
/** Campioni disegnati per frame: 8 dei 320, uno ogni 40, nel tratto di 20 ms centrato sul frame. */
export const SAMPLES_SHOWN = 8;
/** Bande del log-Mel di Whisper disegnate: una ogni 5 (16 delle 80). */
export const MEL_STEP = 5;
/** Coppie seno/coseno della codifica di posizione di Whisper disegnate (6 delle 192 frequenze). */
export const POSITION_CHANNELS = Object.freeze([0, 32, 64, 96, 128, 160]);
/** Durata della finestra: 12 frame da 320 campioni a 16 kHz = 0,24 s. */
const WINDOW_SECONDS = (WINDOW_FRAMES * CODEC.hop) / SAMPLE_RATE;
/** Frame del log-Mel di Whisper per frame del codec (passo 160 contro 320): 2. */
const MEL_PER_FRAME = CODEC.hop / WHISPER.hop;
/** EnCodec negli stessi 240 ms: 18 frame (75 al secondo) e 36 passi da mezzo frame (un solo accorciamento, di 2). */
const EC_FRAMES = Math.round(WINDOW_SECONDS * ENCODEC.frameRate);
const EC_STEPS = 2 * EC_FRAMES;
/** Voci disegnate di ogni codebook di EnCodec (delle 1.024). */
const EC_VOICES = 16;

/**
 * Il testo d'uscita di Whisper: lo stesso esempio della scena (WHISPER_TOKENS di sceneWhisper.js, che
 * importa three.js e quindi qui non si può importare; un test controlla che siano uguali). I primi quattro
 * sono i token speciali veri del formato, le parole sono un esempio.
 */
export const WHISPER_TEXT = Object.freeze(["<|startoftranscript|>", "<|it|>", "<|transcribe|>", "<|notimestamps|>", " Ciao", " a", " tutti", "<|endoftext|>"]);

const frameMs = (lab, t) => Math.round((t * 1000) / lab.frameRate);
const hz = (f) => (f >= 1000 ? `${fmt(f / 1000, 1)} kHz` : `${Math.round(f)} Hz`);
const voiceNames = (k) => Array.from({ length: k }, (_, i) => `voce ${i}`);

// ------------------------------------------------------------------------------------------ valori veri

/** Frame del codec (50 al secondo) in cui attacca la seconda nota della melodia del brano. */
export function onsetFrame(lab, style) {
  return Math.round((lab.songs[style].notes[1]?.start ?? 0) * lab.frameRate);
}

/**
 * Primo frame della finestra: 6 frame prima dell'attacco della seconda nota del brano, così a metà della
 * finestra il suono cambia e il codebook 1 cambia voce. Mai prima del frame 3 (i 4 codebook di MusicGen,
 * sfasati di 0-3 frame, hanno tutti un token vero) e mai così tardi che l'ultimo passo non abbia un seguito.
 */
export function windowStart(lab, style) {
  const maxDelay = Math.max(...MUSICGEN.delays);
  return Math.min(lab.frames - WINDOW_FRAMES - 1, Math.max(maxDelay, onsetFrame(lab, style) - ONSET_AT));
}

/**
 * Il codec del tab sulla finestra, con gli stessi conti di codec.js (rvqEncode e rvqDecode, in Float32):
 * campioni, spettro, residui, voci scelte (one-hot sulle 64 voci) e somma delle voci, nell'ordine dei
 * `values` del nucleo ([neurone][frame]). `pairs[l]` sono i collegamenti verso il codebook l: dal numero n
 * del vettore alla voce c, con il peso vero C_l[c] alla frequenza del numero n. Solo verso le voci usate
 * nella finestra (`used[l]`): il confronto è con tutte e 64, ma 17 × 64 linee su ogni codebook coprirebbero
 * proprio la voce accesa, che è quello che conta vedere.
 */
export function codecWindow(lab, style, start = windowStart(lab, style)) {
  const T = lab.frames;
  const D = WINDOW_FRAMES;
  const { dim, k, codebooks } = lab.rvq;
  const levels = CODEC.used;
  const features = lab.features[style].features;
  const codes = lab.codes[style];
  const samples = lab.songs[style].samples;
  const B = FEATURE_BINS.length;
  const wave = new Float32Array(SAMPLES_SHOWN * D);
  const feature = new Float32Array(B * D);
  const residuals = Array.from({ length: levels - 1 }, () => new Float32Array(B * D));
  const onehots = Array.from({ length: levels }, () => new Float32Array(k * D));
  const choices = Array.from({ length: levels }, () => new Int32Array(D));
  const sum = new Float32Array(B * D);
  const r = new Float32Array(dim);
  const acc = new Float32Array(dim);
  const half = CODEC.hop / 2;
  const gap = CODEC.hop / SAMPLES_SHOWN;
  for (let d = 0; d < D; d++) {
    const t = start + d;
    // 8 campioni del tratto di 20 ms centrato sul frame (il frame t della STFT è centrato sul campione t · hop)
    for (let n = 0; n < SAMPLES_SHOWN; n++) wave[n * D + d] = samples[t * CODEC.hop - half + gap / 2 + n * gap] ?? 0;
    r.set(features.subarray(t * dim, t * dim + dim));
    acc.fill(0);
    for (let b = 0; b < B; b++) feature[b * D + d] = r[FEATURE_BINS[b]];
    for (let l = 0; l < levels; l++) {
      const cb = codebooks[l];
      const q = codes[l * T + t];
      choices[l][d] = q;
      onehots[l][q * D + d] = 1;
      for (let x = 0; x < dim; x++) {
        r[x] -= cb[q * dim + x];
        acc[x] += cb[q * dim + x];
      }
      if (l < levels - 1) for (let b = 0; b < B; b++) residuals[l][b * D + d] = r[FEATURE_BINS[b]];
    }
    for (let b = 0; b < B; b++) sum[b * D + d] = acc[FEATURE_BINS[b]];
  }
  const used = choices.map((q) => [...new Set(q)].sort((a, b) => a - b));
  const pairs = used.map((voices, l) => {
    const cb = codebooks[l];
    const n = B * voices.length;
    const from = new Int32Array(n);
    const to = new Int32Array(n);
    const weight = new Float32Array(n);
    let i = 0;
    for (let b = 0; b < B; b++) {
      for (const c of voices) {
        from[i] = b;
        to[i] = c;
        weight[i++] = cb[c * dim + FEATURE_BINS[b]];
      }
    }
    return { from, to, weight };
  });
  return { start, frames: Array.from({ length: D }, (_, d) => start + d), wave, feature, residuals, onehots, choices, used, sum, pairs };
}

/**
 * MusicGen sulla finestra, come durante l'addestramento: al passo s il codebook k legge il token vero del
 * frame s − k (delay pattern, ritardi 0-3) e la testa k dà le probabilità del token del passo dopo (frame
 * s + 1 − k). Le probabilità sono quelle del modello a conteggi del tab (TokenModel di codec.js, temperatura
 * 1), che nel browser sostituisce il transformer.
 */
export function musicgenWindow(lab, style, start = windowStart(lab, style)) {
  const T = lab.frames;
  const D = WINDOW_FRAMES;
  const k = lab.rvq.k;
  const Q = MUSICGEN.codebooks;
  const codes = lab.codes[style];
  const tokens = new Float32Array(Q * k * D);
  const probs = new Float32Array(Q * k * D);
  const read = Array.from({ length: Q }, () => new Int32Array(D)); // il token letto da ogni lastra a ogni passo
  for (let q = 0; q < Q; q++) {
    for (let d = 0; d < D; d++) {
      const s = start + d;
      const token = codes[q * T + (s - MUSICGEN.delays[q])];
      read[q][d] = token;
      tokens[(q * k + token) * D + d] = 1;
      const p = lab.model.distribution(style, q, codes, T, s + 1 - MUSICGEN.delays[q], 1);
      for (let i = 0; i < k; i++) probs[(q * k + i) * D + d] = p[i];
    }
  }
  return { start, tokens, probs, read };
}

/** Codifica di posizione dell'encoder di Whisper (sinusoids di whisper/model.py): seno nei primi canali/2, coseno negli altri. */
export function whisperSinusoid(position, channel, channels = WHISPER_TINY.width, maxTimescale = 10000) {
  const half = channels / 2;
  const i = channel % half;
  const angle = position * Math.exp((-Math.log(maxTimescale) / (half - 1)) * i);
  return channel < half ? Math.sin(angle) : Math.cos(angle);
}

/**
 * Whisper sulla finestra: 24 frame del log-Mel vero (lab.whisper, la formula di whisper/audio.py: un frame
 * ogni 10 ms, quindi i frame 2·start…), 16 delle 80 bande; la codifica di posizione vera delle 12 posizioni
 * dell'encoder (una ogni 20 ms dopo la convoluzione con passo 2); i token d'esempio come one-hot.
 */
export function whisperWindow(lab, style, start = windowStart(lab, style)) {
  const melFrames = MEL_PER_FRAME * WINDOW_FRAMES;
  const bands = WHISPER.nMels / MEL_STEP;
  const lm = lab.whisper(style);
  const mel = new Float32Array(bands * melFrames);
  for (let b = 0; b < bands; b++) for (let j = 0; j < melFrames; j++) mel[b * melFrames + j] = lm.values[(MEL_PER_FRAME * start + j) * WHISPER.nMels + b * MEL_STEP];
  const half = WHISPER_TINY.width / 2;
  const channels = [...POSITION_CHANNELS, ...POSITION_CHANNELS.map((c) => c + half)];
  const positions = new Float32Array(channels.length * WINDOW_FRAMES);
  channels.forEach((c, n) => {
    for (let d = 0; d < WINDOW_FRAMES; d++) positions[n * WINDOW_FRAMES + d] = whisperSinusoid(start + d, c);
  });
  const T = WHISPER_TEXT.length;
  const text = new Float32Array(T * T);
  for (let i = 0; i < T; i++) text[i * T + i] = 1; // token i → neurone i: gli 8 token d'esempio sono tutti diversi
  return { start, mel, positions, channels, text };
}

// ------------------------------------------------------------------------------------------ EnCodec

function encodecPart() {
  const steps = { count: EC_STEPS, label: "tempo (passi da 6,7 ms)" };
  const frames = { count: EC_FRAMES, label: "frame di EnCodec (13,3 ms)" };
  const ch = ENCODEC.channels;
  const sections = [
    {
      id: "ec_wave", label: "Forma d'onda", sub: "24.000 campioni/s · qui 8 ogni 160", role: "input", neurons: 8, activation: "signed", depth: steps,
      text: `Il suono entra come campioni: 24.000 numeri al secondo, su un solo canale (mono). In profondità il tempo: ${EC_STEPS} passi da 160 campioni (6,7 ms); di ogni passo disegniamo 8 campioni.\n\nAnalogia: il nastro di un registratore letto un pezzetto alla volta. Valori simulati: questa rete non ascolta il nostro brano.`,
    },
    {
      id: "ec_conv", label: "Conv iniziale", sub: `kernel 7 · 1 → ${ch[0]} canali · qui 8`, role: "conv", neurons: 8, depth: steps, real: ch[0],
      text: `Una convoluzione 1D: ogni istante guarda solo gli istanti vicini (7 campioni) e diventa ${ch[0]} canali, cioè ${ch[0]} «rilevatori» di piccoli motivi del suono. Gli stessi pesi scorrono su tutto il tempo, come una lente che passa lungo il nastro. Nel disegno la finestra unisce passi vicini da 160 campioni: quella vera, di 7 campioni, è molto più stretta di un passo.`,
    },
    {
      id: "ec_blocks", label: "4 blocchi con passo", sub: `${ENCODEC.encoderStrides.join(" · ")} = ${ENCODEC.hop} · qui uno, di 2`, role: "conv", neurons: 12, depth: frames, real: ch[4],
      text: `Ogni blocco ha una convoluzione residua e una convoluzione con passo (stride): con passo 2 due istanti diventano uno, con passo 8 otto diventano uno. In tutto ${ENCODEC.encoderStrides.join(" · ")} = ${ENCODEC.hop}: da 24.000 campioni al secondo a ${ENCODEC.frameRate} frame al secondo, uno ogni 13,3 ms. Intanto i canali raddoppiano: ${ch.join(", ")}.\n\nAnalogia: riassumere un libro frase per frase, poi paragrafo per paragrafo, poi capitolo per capitolo, con più concetti a ogni livello. Qui si vede un solo accorciamento (di 2): i passi veri non si possono disegnare in scala.`,
    },
    {
      id: "ec_latent", label: "LSTM e latente", sub: `${ENCODEC.dimension} numeri per frame · qui 8`, role: "latent", neurons: 8, depth: frames, real: ENCODEC.dimension,
      text: `${ENCODEC.lstmLayers} strati LSTM portano una memoria da un frame al successivo (la ricorrenza non è disegnata), poi una convoluzione finale riduce ogni frame a ${ENCODEC.dimension} numeri: ${ENCODEC.frameRate} vettori al secondo. È questo vettore che la quantizzazione trasforma in token.`,
    },
    {
      id: "ec_cb1", label: "Codebook 1", sub: `${fmt(ENCODEC.codebookSize)} voci (10 bit) · qui ${EC_VOICES}`, role: "codebook", neurons: EC_VOICES, depth: frames, real: ENCODEC.codebookSize,
      text: `Il vettore del frame si confronta con le ${fmt(ENCODEC.codebookSize)} voci del codebook ed è sostituito dalla più vicina (la voce accesa): basta salvare il suo numero, 10 bit. Come descrivere un colore con il nome del campione più simile di un catalogo.`,
    },
    {
      id: "ec_r1", label: "Residuo", sub: `${ENCODEC.dimension} numeri per frame · qui 8`, role: "latent", neurons: 8, depth: frames, real: ENCODEC.dimension,
      text: "Al vettore si toglie la voce scelta: resta un errore più piccolo, il residuo. Il codebook dopo quantizza proprio questo (Residual Vector Quantization).",
    },
    {
      id: "ec_cb2", label: "Codebook 2", sub: `${fmt(ENCODEC.codebookSize)} voci · qui ${EC_VOICES}`, role: "codebook", neurons: EC_VOICES, depth: frames, real: ENCODEC.codebookSize,
      text: "Il secondo codebook sceglie la voce più vicina al residuo: aggiunge dettaglio, come le monete dopo le banconote.",
    },
    {
      repeat: 7, of: ["ec_r1", "ec_cb2"], id: "ec_rvq_x7", label: "× 7 residuo → codebook",
      text: `A 6 kbps EnCodec usa 8 codebook: il primo più 7 coppie residuo → codebook come questa. ${ENCODEC.frameRate} frame × 8 codebook × 10 bit = 6.000 bit al secondo. Le bande del modello a 24 kHz sono ${ENCODEC.bandwidths.map((b) => fmt(b, b % 1 ? 1 : 0)).join(", ")} kbps, con ${ENCODEC.codebooks.join(", ")} codebook.`,
    },
    {
      id: "ec_quant", label: "Somma delle voci", sub: `${ENCODEC.dimension} numeri per frame · qui 8`, role: "add", neurons: 8, depth: frames, real: ENCODEC.dimension,
      text: "La somma delle voci scelte ricostruisce (quasi) il latente. Al decoder basta questo: 8 numeri interi per frame al posto di 320 campioni.",
    },
    {
      id: "ec_dec", label: "Conv e LSTM del decoder", sub: `${ENCODEC.dimension} → ${ch[4]} canali · qui 12`, role: "conv", neurons: 12, depth: frames, real: ch[4],
      text: `Il decoder è lo specchio dell'encoder: una convoluzione porta il latente a ${ch[4]} canali e ${ENCODEC.lstmLayers} strati LSTM riportano la memoria nel tempo.`,
    },
    {
      id: "ec_up", label: "4 blocchi trasposti", sub: `passo ${ENCODEC.decoderStrides.join(" · ")} · qui uno, di 2`, role: "conv", neurons: 8, depth: steps, real: ch[0],
      text: `Convoluzioni trasposte con passo ${ENCODEC.decoderStrides.join(", ")}: ogni frame torna a essere ${ENCODEC.hop} campioni, mentre i canali si dimezzano (${[...ch].reverse().join(", ")}). Qui si vede un solo allungamento, di 2.`,
    },
    {
      id: "ec_out", label: "Forma d'onda ricostruita", sub: "24.000 campioni/s", role: "output", neurons: 8, activation: "signed", depth: steps,
      text: "L'ultima convoluzione torna a un canale: il suono ricostruito. Encoder, RVQ e decoder si addestrano insieme: la loss confronta il suono ricostruito con l'originale (anche sugli spettrogrammi) e un discriminatore premia il suono che «sembra vero».",
    },
  ];
  const links = [
    { from: "ec_wave", to: "ec_conv", pattern: "conv", kernel: 3, density: 0.08, label: "finestra locale nel tempo" },
    { from: "ec_conv", to: "ec_blocks", pattern: "down", density: 0.12, label: "il tempo si accorcia" },
    { from: "ec_blocks", to: "ec_latent", pattern: "dense", density: 0.2 },
    { from: "ec_latent", to: "ec_cb1", pattern: "dense", density: 0.2, label: "confronto con le voci" },
    { from: "ec_cb1", to: "ec_r1", pattern: "lookup", label: "la voce scelta" },
    { from: "ec_latent", to: "ec_r1", pattern: "residual", label: "latente − voce scelta" },
    { from: "ec_r1", to: "ec_cb2", pattern: "dense", density: 0.2, label: "confronto con le voci" },
    { from: "ec_cb1", to: "ec_quant", pattern: "lookup", density: 0.5, label: "somma delle voci" },
    { from: "ec_cb2", to: "ec_quant", pattern: "lookup", density: 0.5, label: "somma delle voci" },
    { from: "ec_quant", to: "ec_dec", pattern: "dense", density: 0.2 },
    { from: "ec_dec", to: "ec_up", pattern: "up", density: 0.12, label: "il tempo si riallunga" },
    { from: "ec_up", to: "ec_out", pattern: "conv", kernel: 3, density: 0.08 },
  ];
  const groups = [
    { label: "EnCodec 24 kHz: un codec neurale vero", from: "ec_wave", to: "ec_out", color: "#fdba74", text: `Comprime il suono in pochi numeri interi per frame e lo ricostruisce: encoder convoluzionale, quantizzazione residua (RVQ) e decoder simmetrico, addestrati insieme. Struttura vera (codice ufficiale), valori simulati: questa rete non gira nel browser.` },
    { label: "Encoder", from: "ec_conv", to: "ec_latent", color: "#fdba74", text: `Da 24.000 campioni al secondo a ${ENCODEC.frameRate} vettori da ${ENCODEC.dimension} numeri.` },
    { label: "RVQ", from: "ec_cb1", to: "ec_quant", color: "#fde047", text: `Codebook in cascata da ${fmt(ENCODEC.codebookSize)} voci, ognuno sul residuo del precedente: da 2 a 32 a seconda della banda.` },
    { label: "Decoder", from: "ec_dec", to: "ec_out", color: "#fdba74", text: "Lo specchio dell'encoder: dal latente quantizzato di nuovo ai campioni." },
  ];
  return { sections, links, groups };
}

// ------------------------------------------------------------------------------------------ codec del tab

function codecPart(lab, style, v) {
  const k = lab.rvq.k;
  const B = FEATURE_BINS.length;
  const err = lab.relativeError();
  const binNames = FEATURE_BINS.map((b) => hz((b * lab.sampleRate) / CODEC.nFft));
  const onset = frameMs(lab, onsetFrame(lab, style));
  const vector = (o) => ({ role: "latent", neurons: B, depth: true, real: CODEC.bins, neuronNames: binNames, ...o });
  const codebook = (l, o) => ({ id: `toy_cb${l + 1}`, label: `Codebook ${l + 1}`, role: "codebook", neurons: k, depth: true, values: v.onehots[l], neuronNames: voiceNames(k), ...o });
  const residual = (l, o) => vector({ id: `toy_r${l + 1}`, label: `Residuo ${l + 1}`, sub: `${CODEC.bins} numeri per frame · qui ${B}`, values: v.residuals[l], ...o });
  const sections = [
    {
      id: "toy_wave", label: "Forma d'onda", sub: `${fmt(lab.sampleRate)} campioni/s · qui 8 dei ${CODEC.hop} per frame`, role: "input", activation: "signed", neurons: SAMPLES_SHOWN, depth: true, values: v.wave,
      text: `Il brano «${lab.songs[style].style.prompt}», sintetizzato qui con la sintesi additiva: ${fmt(lab.sampleRate)} campioni al secondo. Ogni frame dura 20 ms, cioè ${CODEC.hop} campioni; ne disegniamo 8 (uno ogni 40) con il loro valore vero, tra −1 e 1.`,
    },
    vector({
      id: "toy_feat", label: "Spettro del frame", sub: `${CODEC.bins} numeri (0-4 kHz) · qui ${B}`, values: v.feature,
      text: `L'encoder del nostro codec è fisso e non impara: per ogni frame fa la FFT di una finestra di ${fmt(CODEC.nFft)} campioni (64 ms, quindi anche un po' dei frame vicini: per questo i collegamenti arrivano dai frame accanto), prende l'ampiezza fino a 4 kHz e la comprime con log(1 + 20·|X|). Risultato: ${CODEC.bins} numeri per frame, ${fmt(lab.frameRate)} frame al secondo; qui ${B}, uno ogni 250 Hz.\n\nAnalogia: un equalizzatore grafico che fotografa le sue barre ogni 20 ms.`,
    }),
    codebook(0, {
      sub: `${k} voci (6 bit): ne vince una per frame`,
      text: `Il codebook 1 ha ${k} voci: i centroidi trovati dal k-means sui frame dei due brani. Il vettore del frame si confronta con tutte e vince la più vicina: il suo numero è il token del codebook 1. I collegamenti, sul frame scelto, vanno alle ${v.used[0].length} voci usate in questi 12 frame e hanno i numeri veri delle voci (gli altri confronti non sono disegnati).\n\nIn profondità si vede che la voce accesa resta la stessa mentre la nota suona uguale e cambia attorno a ${fmt(onset)} ms, quando attacca la nota successiva; poi cambia ancora qualche volta nei primi frame della nota nuova, finché il suo volume si assesta dopo l'attacco.`,
    }),
    residual(0, { text: "Allo spettro si toglie la voce scelta: resta l'errore del codebook 1. Il codebook 2 quantizza questo residuo, non lo spettro: ogni livello corregge il precedente. Come pagare 47,30 euro: prima le banconote, poi le monete, poi gli spiccioli." }),
    codebook(1, { sub: `${k} voci sul residuo 1`, text: `Altre ${k} voci, trovate con il k-means sui residui del livello 1: descrivono gli errori tipici del primo codebook. Anche qui vince la voce più vicina, frame per frame.` }),
    residual(1, { text: `L'errore che resta dopo due codebook. Errore relativo, misurato sui due brani: ${formatPercent(err[1])} con un codebook, ${formatPercent(err[2])} con due.` }),
    codebook(2, { sub: `${k} voci sul residuo 2`, text: "Il terzo livello corregge l'errore lasciato dai primi due." }),
    residual(2, { text: `Dopo tre codebook l'errore relativo è sceso a ${formatPercent(err[3])}.` }),
    codebook(3, {
      sub: `${k} voci sul residuo 3`,
      text: `L'ultimo dei ${CODEC.used} codebook usati dal tab (ne sono addestrati ${CODEC.levels}). Con ${CODEC.used} codebook l'errore relativo è ${formatPercent(err[CODEC.used])}: ogni frame è diventato ${CODEC.used} numeri interi tra 0 e ${k - 1}, cioè ${CODEC.used * Math.log2(k)} bit invece di ${CODEC.hop} campioni da 16 bit.`,
    }),
    vector({
      id: "toy_sum", label: "Somma delle 4 voci", sub: `${CODEC.bins} numeri per frame · qui ${B}`, role: "add", values: v.sum,
      text: "Il decoder del nostro codec somma le voci scelte e ottiene uno spettro quasi uguale all'originale (confrontalo con lo spettro più a sinistra); poi l'algoritmo di Griffin-Lim, che non è una rete, stima le fasi e torna al suono. Lo ascolti con «Ricostruito» nel pannello Laboratorio.",
    }),
  ];
  const links = [{ from: "toy_wave", to: "toy_feat", pattern: "conv", kernel: 3, density: 0.1, label: "FFT su 64 ms" }];
  for (let l = 0; l < CODEC.used; l++) {
    const input = l === 0 ? "toy_feat" : `toy_r${l}`;
    links.push({ from: input, to: `toy_cb${l + 1}`, pattern: "explicit", pairs: v.pairs[l], label: "confronto con le voci usate (numeri veri)" });
    if (l < CODEC.used - 1) {
      links.push({ from: `toy_cb${l + 1}`, to: `toy_r${l + 1}`, pattern: "lookup", label: "la voce scelta" });
      links.push({ from: input, to: `toy_r${l + 1}`, pattern: "residual", label: l === 0 ? "spettro − voce scelta" : "residuo − voce scelta" });
    }
  }
  for (let l = 0; l < CODEC.used; l++) links.push({ from: `toy_cb${l + 1}`, to: "toy_sum", pattern: "lookup", density: 0.5, label: "somma delle voci" });
  const groups = [
    { label: "Il codec del tab: numeri veri calcolati qui", from: "toy_wave", to: "toy_sum", color: "#67e8f9", text: `Il codec giocattolo del tab, con i numeri calcolati nel browser sul brano «${lab.songs[style].style.prompt}»: un encoder fisso (lo spettro di ogni frame) e una RVQ con ${CODEC.used} codebook da ${k} voci trovati con il k-means. Non è una rete addestrata: encoder e decoder sono calcoli fissi, solo i codebook sono «imparati».` },
    { label: `RVQ: ${CODEC.used} codebook in cascata`, from: "toy_cb1", to: "toy_cb4", color: "#fde047", text: "Ogni codebook quantizza il residuo lasciato dai precedenti (algoritmo 1 di SoundStream)." },
  ];
  return { sections, links, groups };
}

// ------------------------------------------------------------------------------------------ MusicGen

function musicgenPart(lab, style, v) {
  const k = lab.rvq.k;
  const words = lab.songs[style].style.prompt.split(" ");
  const Q = MUSICGEN.codebooks;
  const slabNames = (what) => Array.from({ length: Q }, (_, q) => `codebook ${q + 1}${what ? ` (ritardo ${MUSICGEN.delays[q]})` : ""}`);
  const width = (o) => ({ neurons: 16, depth: true, real: MUSICGEN.dim, ...o });
  const add = (id, text) => width({ id, label: "+ residuo", sub: `${fmt(MUSICGEN.dim)} numeri per passo · qui 16`, role: "add", text });
  const sections = [
    {
      id: "mg_tokens", label: "Token: 4 codebook sfasati", sub: `4 token per passo · ${fmt(MUSICGEN.card)} voci + S · qui ${k}`, role: "input", activation: "onehot", slabs: Q, slabNames: slabNames(true), neurons: k, depth: true, values: v.tokens, neuronNames: voiceNames(k),
      text: `A ogni passo il transformer legge un token per codebook, ma sfasati: il codebook 1 del frame s, il 2 del frame s − 1, il 3 del frame s − 2, il 4 del frame s − 3 (delay pattern, ritardi ${MUSICGEN.delays.join(", ")}). Così quando scrive il codebook 2 di un frame, il codebook 1 di quel frame è già pronto, e bastano ${MUSICGEN.frameRate} passi per secondo di audio invece di ${MUSICGEN.frameRate * Q}.\n\nSono i token veri del codec del tab (${k} voci invece delle ${fmt(MUSICGEN.card)} di MusicGen), letti come durante l'addestramento: il token del codebook 2 di un frame lo ritrovi nella seconda lastra un passo dopo, quello del codebook 3 nella terza due passi dopo. All'inizio di un brano, dove il frame non esiste ancora, MusicGen legge il token speciale S: qui siamo già dentro il brano.\n\nIl MusicGen vero legge i token di un altro EnCodec, a 32 kHz e addestrato sulla musica: ${Q} codebook da ${fmt(MUSICGEN.card)} voci, ${MUSICGEN.frameRate} frame al secondo. Quello a sinistra è il modello a ${fmt(ENCODEC.sampleRate / 1000)} kHz: ${ENCODEC.frameRate} frame al secondo e codebook da ${fmt(ENCODEC.codebookSize)} voci.`,
    },
    width({
      id: "mg_emb", label: "Somma dei 4 embedding", sub: `${fmt(MUSICGEN.dim)} numeri per passo · qui 16`, role: "embedding",
      text: `Ogni codebook ha la sua tabella di embedding (${fmt(MUSICGEN.card)} voci più il token speciale): i 4 vettori del passo si sommano in uno solo. Come mescolare 4 ingredienti in un impasto: il transformer vede la somma, non i 4 token separati. All'ingresso del transformer si somma anche la posizione del passo (sinusoidi fisse), non disegnata.`,
    }),
    {
      id: "mg_text", label: "Testo del prompt (T5-base)", sub: `${MUSICGEN.textDim} numeri per parola · qui 12`, role: "embedding", neurons: 12, real: MUSICGEN.textDim,
      depth: { count: words.length, label: "parola del prompt", names: words },
      slices: false, // le parole sono posizioni, non istanti: restano accese con ogni fetta
      text: `La descrizione («${lab.songs[style].style.prompt}») passa da T5-base, un text encoder già addestrato e congelato: non si riaddestra con MusicGen. Ogni pezzo di parola diventa un vettore di ${MUSICGEN.textDim} numeri, proiettato poi a ${fmt(MUSICGEN.dim)}. Qui una posizione per parola: il tokenizer vero spezza le parole in pezzi. Valori simulati.`,
    },
    width({
      id: "mg_attn", label: "Self-attention causale", sub: `${MUSICGEN.heads} head · qui 4`, role: "attention", heads: 4, slabs: 2, slabNames: ["key e value", "uscita"],
      text: `Ogni passo guarda se stesso e i passi prima, mai quelli dopo (maschera causale): gli archi vanno solo verso il passato, come nel GPT del corso. È qui che il modello «ricorda» la musica suonata finora. Un colore per head: MusicGen small ne ha ${MUSICGEN.heads}, qui 4; pesi simulati.`,
    }),
    add("mg_add1", "L'uscita dell'attention si somma al vettore che era entrato (connessione residua). Prima della self-attention, della cross-attention e del feed-forward c'è una LayerNorm (pre-norm), non disegnata."),
    width({
      id: "mg_cross", label: "Cross-attention al testo", sub: `${MUSICGEN.heads} head · qui 4 · key e value dal testo`, role: "attention", heads: 4,
      text: `Ogni passo audio chiede alle parole del prompt che cosa suonare: le query vengono dall'audio, chiavi e valori dal testo. È l'attention della lezione, ma guarda un'altra sequenza. Generando, MusicGen calcola anche la previsione senza testo e spinge nella direzione del testo (classifier-free guidance, coefficiente ${MUSICGEN.cfg}).`,
    }),
    add("mg_add2", "x + cross-attention(x): ciò che il passo ha letto nel prompt si aggiunge al flusso, non lo sostituisce."),
    width({ id: "mg_ffn", label: "Feed-forward", sub: `${fmt(MUSICGEN.ffn)} neuroni (GELU) · qui 32`, role: "mlp", neurons: 32, activation: "positive", real: MUSICGEN.ffn, text: `Ogni passo per conto suo: ${fmt(MUSICGEN.dim)} → ${fmt(MUSICGEN.ffn)} → ${fmt(MUSICGEN.dim)} numeri, con la GELU. Qui sta metà dei pesi del layer (2 × ${fmt(MUSICGEN.dim)} × ${fmt(MUSICGEN.ffn)}, circa ${fmt((2 * MUSICGEN.dim * MUSICGEN.ffn) / 1e6, 1)} milioni): l'altra metà è nella self-attention e nella cross-attention, 4 matrici ${fmt(MUSICGEN.dim)} × ${fmt(MUSICGEN.dim)} ciascuna.` }),
    add("mg_add3", "x + feed-forward(x). Le connessioni residue sono una strada principale con delle deviazioni: ogni blocco aggiunge qualcosa, ma il segnale di partenza passa comunque."),
    {
      repeat: MUSICGEN.layers, of: ["mg_attn", "mg_add1", "mg_cross", "mg_add2", "mg_ffn", "mg_add3"], id: "mg_x24", label: `× ${MUSICGEN.layers} layer`,
      text: `MusicGen small ha ${MUSICGEN.layers} layer uguali a questo (medium e large 48): ne vedi uno vero e poche lastre fantasma.`,
    },
    width({ id: "mg_norm", label: "LayerNorm finale", sub: `${fmt(MUSICGEN.dim)} numeri per passo · qui 16`, role: "norm", text: "L'ultima normalizzazione prima delle teste d'uscita." }),
    {
      id: "mg_heads", label: "4 teste: il passo dopo", sub: `${fmt(MUSICGEN.card)} voci per codebook · qui ${k}`, role: "output", activation: "probs", slabs: Q, slabNames: slabNames(false), neurons: k, depth: true, values: v.probs, neuronNames: voiceNames(k),
      text: `Quattro teste lineari, una per codebook, danno le probabilità dei 4 token del passo dopo. Le probabilità disegnate sono vere, ma vengono dal modello a conteggi del tab, che nel browser sostituisce il transformer (guarda solo il token precedente e il codebook sopra), calcolate sui token veri del brano. Generando, il modello sceglie un token, lo scrive e al passo dopo lo rilegge in ingresso.`,
    },
  ];
  const links = [
    ...Array.from({ length: Q }, (_, q) => ({ from: "mg_tokens", to: "mg_emb", pattern: "lookup", fromSlab: q, label: `embedding del codebook ${q + 1}` })),
    { from: "mg_emb", to: "mg_attn", pattern: "dense", toSlab: 0, density: 0.15 },
    { within: "mg_attn", pattern: "attention", causal: true },
    { from: "mg_attn", to: "mg_add1", pattern: "dense", density: 0.12 },
    { from: "mg_emb", to: "mg_add1", pattern: "residual" },
    { from: "mg_add1", to: "mg_cross", pattern: "dense", density: 0.12, label: "query" },
    { from: "mg_text", to: "mg_cross", pattern: "attention", label: "key e value dal testo" },
    { from: "mg_cross", to: "mg_add2", pattern: "dense", density: 0.12 },
    { from: "mg_add1", to: "mg_add2", pattern: "residual" },
    { from: "mg_add2", to: "mg_ffn", pattern: "dense", density: 0.08 },
    { from: "mg_ffn", to: "mg_add3", pattern: "dense", density: 0.08 },
    { from: "mg_add2", to: "mg_add3", pattern: "residual" },
    { from: "mg_add3", to: "mg_norm", pattern: "identity" },
    { from: "mg_norm", to: "mg_heads", pattern: "dense", toSlab: "all", density: 0.015 },
  ];
  const groups = [
    { label: "MusicGen small: il transformer che genera i token", from: "mg_tokens", to: "mg_heads", color: "#93c5fd", text: `Un transformer decoder come il GPT del corso, ma a ogni passo legge 4 token (uno per codebook) e ne prevede 4. Struttura vera (configurazione ufficiale: ${MUSICGEN.layers} layer, ${fmt(MUSICGEN.dim)} numeri, ${MUSICGEN.heads} head, circa 300 milioni di parametri); token del codec del tab e probabilità del suo modello a conteggi; neuroni interni simulati.` },
  ];
  return { sections, links, groups };
}

// ------------------------------------------------------------------------------------------ Whisper

function whisperPart(lab, v) {
  const W = WHISPER_TINY;
  const bank = melFilterbank({ sampleRate: WHISPER.sampleRate, nFft: WHISPER.nFft, nMels: WHISPER.nMels });
  const bandNames = Array.from({ length: WHISPER.nMels / MEL_STEP }, (_, b) => hz(bank.centers[b * MEL_STEP]));
  const periodNames = v.channels.map((c) => {
    const i = c % (W.width / 2);
    const seconds = ((2 * Math.PI) / Math.exp((-Math.log(10000) / (W.width / 2 - 1)) * i)) / lab.frameRate;
    return `${c < W.width / 2 ? "sin" : "cos"} · giro in ${fmt(seconds, seconds < 10 ? 2 : 0)} s`;
  });
  const melFrames = Array.from({ length: MEL_PER_FRAME * WINDOW_FRAMES }, (_, j) => `${fmt(((MEL_PER_FRAME * v.start + j) * 1000 * WHISPER.hop) / WHISPER.sampleRate)} ms`);
  const tokenNames = WHISPER_TEXT.map((t) => t.replace(/^ /, "·"));
  const text = { count: tokenNames.length, label: "token di testo", names: tokenNames };
  const enc = (o) => ({ neurons: 12, depth: true, real: W.width, ...o });
  // i token di testo sono posizioni, non istanti: non si accendono con la fetta (slices: false)
  const dec = (o) => ({ neurons: 12, depth: text, slices: false, real: W.width, ...o });
  const sections = [
    {
      id: "wh_mel", label: "log-Mel", sub: `${WHISPER.nMels} bande × ${fmt(WHISPER.nFrames)} frame (30 s) · qui 16 × 24`, role: "input", activation: "signed", neurons: WHISPER.nMels / MEL_STEP, values: v.mel, neuronNames: bandNames,
      depth: { count: melFrames.length, label: "frame da 10 ms", names: melFrames },
      text: `Whisper non usa token audio: riceve direttamente il log-Mel, calcolato qui con la stessa formula di whisper/audio.py (finestre di ${WHISPER.nFft} campioni, passo ${WHISPER.hop}, ${WHISPER.nMels} bande Mel, logaritmo, dinamica di 8 e riscalatura). Un frame ogni 10 ms: questi 24 sono gli stessi 240 ms delle reti a sinistra, con i loro valori veri (16 delle ${WHISPER.nMels} bande, una ogni ${MEL_STEP}). Il resto dei 30 secondi è silenzio di riempimento.`,
    },
    {
      id: "wh_conv1", label: "Conv1d + GELU", sub: `kernel 3 · ${WHISPER.nMels} → ${W.width} canali · qui 12`, role: "conv", neurons: 12, depth: { count: melFrames.length, label: "frame da 10 ms", names: melFrames }, real: W.width,
      text: `Una convoluzione 1D con kernel 3: ogni frame guarda il frame prima e quello dopo, e le ${WHISPER.nMels} bande diventano ${W.width} canali. Gli stessi pesi su tutto il tempo.`,
    },
    enc({
      id: "wh_conv2", label: "Conv1d con passo 2", sub: `${fmt(WHISPER.nFrames)} → ${fmt(W.audioPositions)} posizioni da 20 ms`, role: "conv",
      text: `La seconda convoluzione ha passo 2: due frame diventano una posizione. ${fmt(WHISPER.nFrames)} frame (30 s) diventano ${fmt(W.audioPositions)} posizioni da 20 ms, quelle che vedrà il transformer. Nel disegno ogni posizione riceve dai suoi 2 frame; nel modello vero la finestra è di 3 frame (kernel 3) e i canali si mescolano.`,
    }),
    enc({
      id: "wh_pos", label: "Posizione (sinusoidi)", sub: `fissa, non imparata · qui 12 dei ${W.width}`, role: "embedding", values: v.positions, neuronNames: periodNames,
      text: "L'encoder somma a ogni posizione un vettore di seni e coseni a frequenze diverse (sinusoids in whisper/model.py): fisso, non imparato. Le onde veloci distinguono posizioni vicine, quelle lente posizioni lontane, come le lancette dei secondi e delle ore di un orologio. Valori veri, calcolati con la stessa formula per le posizioni di questa finestra.",
    }),
    enc({ id: "wh_in", label: "+ posizione", sub: `${W.width} numeri · qui 12`, role: "add", text: "Convoluzione più posizione: ora ogni vettore sa che cosa si sente e quando." }),
    enc({
      id: "wh_attn", label: "Self-attention bidirezionale", sub: `${W.heads} head · qui 3`, role: "attention", heads: 3, slabs: 2, slabNames: ["key e value", "uscita"],
      text: `Ogni posizione guarda tutte le altre, prima e dopo: niente maschera causale, perché l'audio intero (30 s) è già disponibile. È la differenza con il GPT del corso, che guarda solo il passato. Un colore per head: Whisper tiny ne ha ${W.heads}, qui 3; pesi simulati.`,
    }),
    enc({ id: "wh_add1", label: "+ residuo", sub: `${W.width} numeri · qui 12`, role: "add", text: "x + attention(x). Prima dell'attention e dell'MLP c'è una LayerNorm, non disegnata." }),
    enc({ id: "wh_mlp", label: "MLP", sub: `${fmt(W.mlp)} neuroni (GELU) · qui 24`, role: "mlp", neurons: 24, activation: "positive", real: W.mlp, text: `Ogni posizione per conto suo: ${W.width} → ${fmt(W.mlp)} → ${W.width} numeri, con la GELU.` }),
    enc({ id: "wh_add2", label: "+ residuo", sub: `${W.width} numeri · qui 12`, role: "add", text: "x + MLP(x): come annotare a margine di un libro senza cancellare quello che c'era." }),
    { repeat: W.layers, of: ["wh_attn", "wh_add1", "wh_mlp", "wh_add2"], id: "wh_enc_x4", label: `× ${W.layers} layer`, text: `Whisper tiny ha ${W.layers} layer nell'encoder (large: ${W.largeLayers}). Ognuno: LayerNorm, attention, residuo; LayerNorm, MLP, residuo.` },
    enc({ id: "wh_enc_out", label: "LayerNorm finale", sub: `${fmt(W.audioPositions)} × ${W.width}: la memoria del decoder`, role: "norm", text: `L'uscita dell'encoder: ${fmt(W.audioPositions)} vettori da ${W.width} numeri, uno ogni 20 ms. Il decoder la consulta a ogni token con la cross-attention.` }),
    dec({
      id: "wh_tok", label: "Token di testo (esempio)", sub: `${fmt(W.vocab)} voci · qui 8`, role: "input", activation: "onehot", neurons: tokenNames.length, real: W.vocab, values: v.text, neuronNames: tokenNames, valueLabel: "(token d'esempio)",
      text: "Il decoder scrive il testo un token alla volta. Prima i token speciali che scelgono il compito (inizio, lingua, trascrivi, niente timestamp), poi il testo. Il formato è quello vero di Whisper; le parole sono un esempio, come nella scena: il nostro audio è musica senza voce e qui Whisper non viene eseguito. Il puntino segna lo spazio, che nel tokenizer fa parte del token.",
    }),
    dec({ id: "wh_temb", label: "Embedding + posizione", sub: `posizione imparata · qui 12 dei ${W.width}`, role: "embedding", text: `Ogni token prende la sua riga della tabella di embedding e le si somma la posizione, qui imparata (al massimo ${W.textPositions} token).` }),
    dec({ id: "wh_dattn", label: "Self-attention causale", sub: `${W.heads} head · qui 3`, role: "attention", heads: 3, slabs: 2, slabNames: ["key e value", "uscita"], text: "Ogni token guarda se stesso e i token prima, mai quelli dopo: il decoder è causale come il GPT del corso." }),
    dec({ id: "wh_dadd1", label: "+ residuo", sub: `${W.width} numeri · qui 12`, role: "add", text: "x + attention(x): il token resta se stesso e aggiunge ciò che ha letto nei token prima." }),
    dec({
      id: "wh_cross", label: "Cross-attention sull'audio", sub: `${W.heads} head · qui 3 · key e value dall'encoder`, role: "attention", heads: 3,
      text: `Ogni token di testo consulta l'uscita dell'encoder: le query vengono dal testo, chiavi e valori dalle ${fmt(W.audioPositions)} posizioni dell'audio (qui 12). Gli archi lunghi vanno dai frame dell'encoder ai token: è così che il testo «ascolta».`,
    }),
    dec({ id: "wh_dadd2", label: "+ residuo", sub: `${W.width} numeri · qui 12`, role: "add", text: "x + cross-attention(x): al token si aggiunge ciò che ha «sentito» nell'audio." }),
    dec({ id: "wh_dmlp", label: "MLP", sub: `${fmt(W.mlp)} neuroni (GELU) · qui 24`, role: "mlp", neurons: 24, activation: "positive", real: W.mlp, text: "Ogni token per conto suo, come nell'encoder." }),
    dec({ id: "wh_dadd3", label: "+ residuo", sub: `${W.width} numeri · qui 12`, role: "add", text: "x + MLP(x): l'ultimo ritocco del layer, sempre per somma." }),
    { repeat: W.layers, of: ["wh_dattn", "wh_dadd1", "wh_cross", "wh_dadd2", "wh_dmlp", "wh_dadd3"], id: "wh_dec_x4", label: `× ${W.layers} layer`, text: `Anche il decoder di Whisper tiny ha ${W.layers} layer (large: ${W.largeLayers}), con la cross-attention in ognuno.` },
    dec({ id: "wh_dln", label: "LayerNorm finale", sub: `${W.width} numeri · qui 12`, role: "norm", text: "L'ultima normalizzazione prima dell'uscita." }),
    dec({
      id: "wh_logits", label: "Prossimo token", sub: `${fmt(W.vocab)} voci · qui 8`, role: "output", activation: "probs", neurons: tokenNames.length, real: W.vocab, neuronNames: tokenNames,
      text: "Le probabilità del token successivo su tutto il vocabolario: Whisper riusa la tabella di embedding come matrice di uscita (pesi condivisi). Ogni posizione prevede il token dopo: dopo <|startoftranscript|> la lingua, poi il compito, poi le parole. Probabilità simulate.",
    }),
  ];
  const links = [
    { from: "wh_mel", to: "wh_conv1", pattern: "conv", kernel: 3, density: 0.05 },
    { from: "wh_conv1", to: "wh_conv2", pattern: "down", label: "passo 2" },
    { from: "wh_conv2", to: "wh_in", pattern: "residual", label: "convoluzione + posizione" },
    { from: "wh_pos", to: "wh_in", pattern: "identity" },
    { from: "wh_in", to: "wh_attn", pattern: "dense", toSlab: 0, density: 0.2 },
    { within: "wh_attn", pattern: "attention" }, // bidirezionale: nessuna maschera
    { from: "wh_attn", to: "wh_add1", pattern: "dense", density: 0.15 },
    { from: "wh_in", to: "wh_add1", pattern: "residual" },
    { from: "wh_add1", to: "wh_mlp", pattern: "dense", density: 0.12 },
    { from: "wh_mlp", to: "wh_add2", pattern: "dense", density: 0.12 },
    { from: "wh_add1", to: "wh_add2", pattern: "residual" },
    { from: "wh_add2", to: "wh_enc_out", pattern: "identity" },
    { from: "wh_tok", to: "wh_temb", pattern: "lookup" },
    { from: "wh_temb", to: "wh_dattn", pattern: "dense", toSlab: 0, density: 0.2 },
    { within: "wh_dattn", pattern: "attention", causal: true },
    { from: "wh_dattn", to: "wh_dadd1", pattern: "dense", density: 0.15 },
    { from: "wh_temb", to: "wh_dadd1", pattern: "residual" },
    { from: "wh_dadd1", to: "wh_cross", pattern: "dense", density: 0.15, label: "query" },
    { from: "wh_enc_out", to: "wh_cross", pattern: "attention", label: "key e value dall'encoder" },
    { from: "wh_cross", to: "wh_dadd2", pattern: "dense", density: 0.15 },
    { from: "wh_dadd1", to: "wh_dadd2", pattern: "residual" },
    { from: "wh_dadd2", to: "wh_dmlp", pattern: "dense", density: 0.12 },
    { from: "wh_dmlp", to: "wh_dadd3", pattern: "dense", density: 0.12 },
    { from: "wh_dadd2", to: "wh_dadd3", pattern: "residual" },
    { from: "wh_dadd3", to: "wh_dln", pattern: "identity" },
    { from: "wh_dln", to: "wh_logits", pattern: "dense", density: 0.3 },
  ];
  const groups = [
    { label: "Whisper tiny: dalla voce al testo", from: "wh_mel", to: "wh_logits", color: "#c4b5fd", text: `Il percorso opposto: ascolta e scrive. Struttura vera (codice ufficiale; tiny: ${W.layers} + ${W.layers} layer, ${W.width} numeri, ${W.heads} head, 39 milioni di parametri; large: ${W.largeLayers} + ${W.largeLayers} layer, 1,55 miliardi), log-Mel e posizioni veri del brano, il resto simulato: qui Whisper non viene eseguito.` },
    { label: "Encoder: attention in tutte le direzioni", from: "wh_conv1", to: "wh_enc_out", color: "#c4b5fd", text: "Due convoluzioni, la posizione, 4 layer con attention senza maschera." },
    { label: "Decoder: causale + cross-attention", from: "wh_tok", to: "wh_logits", color: "#c4b5fd", text: "Testo token per token: attention verso il passato e cross-attention verso l'audio." },
  ];
  return { sections, links, groups };
}

// ------------------------------------------------------------------------------------------ fonti

export const NETWORK_SOURCES = Object.freeze([
  { title: "facebookresearch/encodec: encodec/modules/seanet.py", url: "https://github.com/facebookresearch/encodec/blob/main/encodec/modules/seanet.py", note: "SEANetEncoder e SEANetDecoder: 32 filtri di base, ratios [8, 5, 4, 2] (l'encoder li usa al contrario: 2, 4, 5, 8), canali che raddoppiano fino a 512, 2 strati LSTM, latente di 128 numeri; decoder con convoluzioni trasposte. Ramo main, letto il 9 novembre 2026." },
  { title: "facebookresearch/encodec: encodec/model.py", url: "https://github.com/facebookresearch/encodec/blob/main/encodec/model.py", note: "Modello a 24 kHz: bande 1,5/3/6/12/24 kbps, codebook da 1.024 voci, al massimo 32 codebook (75 frame al secondo)." },
  { title: "Défossez, Copet, Synnaeve, Adi (2022), High Fidelity Neural Audio Compression (EnCodec)", url: "https://arxiv.org/abs/2210.13438", note: "Encoder e decoder convoluzionali con LSTM, RVQ, discriminatori." },
  { title: "Zeghidour et al. (2021), SoundStream: An End-to-End Neural Audio Codec", url: "https://arxiv.org/abs/2107.03312", note: "Quantizzazione vettoriale residua (algoritmo 1) addestrata con encoder e decoder." },
  { title: "facebookresearch/audiocraft: config/model/lm/model_scale/small.yaml", url: "https://github.com/facebookresearch/audiocraft/blob/main/config/model/lm/model_scale/small.yaml", note: "MusicGen small: dim 1.024, 16 head, 24 layer (circa 300 milioni di parametri); medium e large: 48 layer." },
  { title: "facebookresearch/audiocraft: config/model/lm/musicgen_lm.yaml e default.yaml", url: "https://github.com/facebookresearch/audiocraft/blob/main/config/model/lm/musicgen_lm.yaml", note: "4 codebook da 2.048 voci, delay pattern con ritardi [0, 1, 2, 3], pre-norm; in default.yaml causal: true, hidden_scale 4, GELU, posizioni sinusoidali." },
  { title: "facebookresearch/audiocraft: audiocraft/models/lm.py", url: "https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/lm.py", note: "Un embedding per codebook (card + 1 voci) sommati in ingresso, una testa lineare per codebook in uscita." },
  { title: "facebookresearch/audiocraft: audiocraft/modules/transformer.py", url: "https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/transformer.py", note: "Layer con norm_first: LayerNorm prima di self-attention, cross-attention e feed-forward; ogni attention ha 4 matrici dim × dim (query, key, value, uscita), il feed-forward due da dim × 4·dim; posizioni sinusoidali sommate all'ingresso." },
  { title: "facebookresearch/audiocraft: docs/MUSICGEN.md", url: "https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md", note: "MusicGen usa un EnCodec a 32 kHz addestrato sulla musica, con 4 codebook a 50 Hz (50 passi per secondo di audio); modelli da 300 milioni (small), 1,5 e 3,3 miliardi di parametri." },
  { title: "facebookresearch/audiocraft: config/conditioner/text2music.yaml e modules/conditioners.py", url: "https://github.com/facebookresearch/audiocraft/blob/main/config/conditioner/text2music.yaml", note: "T5-base congelato (768 numeri per token, proiettati alla larghezza del transformer), cross-attention, classifier-free guidance 3,0." },
  { title: "Copet et al. (2023), Simple and Controllable Music Generation (MusicGen)", url: "https://arxiv.org/abs/2306.05284", note: "Transformer a stadio singolo su 4 codebook EnCodec a 32 kHz e 50 Hz, delay pattern." },
  { title: "openai/whisper: whisper/model.py", url: "https://github.com/openai/whisper/blob/main/whisper/model.py", note: "AudioEncoder: due Conv1d con kernel 3 (la seconda con passo 2) e GELU, sinusoidi di posizione, blocchi senza maschera, ln_post. TextDecoder: embedding e posizione imparata, maschera causale, cross-attention in ogni blocco, MLP largo 4 volte, uscita con la tabella di embedding." },
  { title: "openai/whisper: whisper/audio.py", url: "https://github.com/openai/whisper/blob/main/whisper/audio.py", note: "16 kHz, n_fft 400, hop 160, 30 s = 3.000 frame, 80 bande Mel: la formula del log-Mel calcolato dal tab." },
  { title: "huggingface/transformers: configuration_whisper.py", url: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/configuration_whisper.py", note: "Valori predefiniti = openai/whisper-tiny: 4 + 4 layer, d_model 384, 6 head, MLP 1.536, vocabolario 51.865, 1.500 posizioni audio e 448 di testo." },
  { title: "ggml-org/whisper.cpp: src/whisper.cpp", url: "https://github.com/ggml-org/whisper.cpp/blob/master/src/whisper.cpp", note: "Iperparametri predefiniti di Whisper tiny e layer dell'encoder per taglia: tiny 4, base 6, small 12, medium 24, large 32." },
  { title: "openai/whisper: model-card.md", url: "https://github.com/openai/whisper/blob/main/model-card.md", note: "Parametri: tiny 39 M, base 74 M, small 244 M, medium 769 M, large 1.550 M." },
  { title: "Radford et al. (2022), Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)", url: "https://arxiv.org/abs/2212.04356", note: "Encoder-decoder transformer sul log-Mel, token speciali per i compiti." },
]);

// ------------------------------------------------------------------------------------------ specifica

/**
 * La specifica completa per il brano `style` (pura e deterministica: stesso laboratorio, stessa rete).
 * @param {object} lab il laboratorio del tab (createLab di lab.js), già calcolato
 * @param {{ style?: string }} [options] id del brano (predefinito: il primo)
 */
export function buildAudioNetworkSpec(lab, { style = Object.keys(lab.songs)[0] } = {}) {
  const start = windowStart(lab, style);
  const t0 = frameMs(lab, start);
  const t1 = frameMs(lab, start + WINDOW_FRAMES - 1);
  const prompt = lab.songs[style].style.prompt;
  const codec = codecWindow(lab, style, start);
  const parts = [encodecPart(), codecPart(lab, style, codec), musicgenPart(lab, style, musicgenWindow(lab, style, start)), whisperPart(lab, whisperWindow(lab, style, start))];
  return {
    title: "Le reti dell'audio: EnCodec, il codec del tab, MusicGen e Whisper (ridotte)",
    intro: `Quattro reti in fila, da sinistra a destra: EnCodec, un codec neurale vero; il codec del tab, con i numeri calcolati qui nel browser; il transformer di MusicGen, che genera i token; Whisper, che ascolta e scrive. In profondità c'è il tempo, 240 ms un frame dopo l'altro: nel codec del tab, in MusicGen e in Whisper sono gli stessi 12 frame del brano «${prompt}», da ${fmt(t0)} a ${fmt(t1)} ms.\n\nGira la camera («Guarda lo spessore»): le convoluzioni collegano frame vicini, in ogni codebook si accende una sola voce per frame, i 4 codebook di MusicGen entrano sfasati (delay pattern) e l'attention collega i frame tra loro: in tutte le direzioni nell'encoder di Whisper, solo verso il passato nei decoder.`,
    scale: `1 sfera = 1 numero. Tempo: 12 frame da 20 ms (240 ms); EnCodec 18 frame da 13,3 ms (e 36 passi da 6,7 ms prima dell'accorciamento), il log-Mel di Whisper 24 frame da 10 ms; le parole del prompt e i token di testo sono posizioni, non istanti. In profondità le sezioni con più posizioni (EnCodec, il log-Mel) hanno un passo più fitto: coprono gli stessi 240 ms con la stessa profondità, così un istante sta alla stessa z in tutte le reti, e la fetta scelta vi si abbina in proporzione. EnCodec: 8 dei 128 numeri del latente, 16 delle 1.024 voci di ogni codebook, 2 degli 8 codebook (a 6 kbps), un solo accorciamento del tempo, di 2, al posto dei quattro veri (2, 4, 5, 8). Codec del tab: 8 dei 320 campioni di ogni frame, 17 dei 257 numeri dello spettro (uno ogni 250 Hz), tutte le 64 voci dei 4 codebook. MusicGen small: 64 voci per codebook invece di 2.048 (quelle del codec del tab), 16 dei 1.024 numeri per passo, 4 delle 16 head, 32 dei 4.096 neuroni del feed-forward, 1 dei 24 layer; T5: 12 dei 768 numeri per parola. Whisper tiny: 16 delle 80 bande, 12 dei 384 numeri, 3 delle 6 head, 24 dei 1.536 neuroni dell'MLP, 1 dei 4 layer di encoder e decoder, 8 delle 51.865 voci.`,
    note: `Valori veri, calcolati qui nel browser dal laboratorio del tab sul brano «${prompt}» (frame da ${fmt(t0)} a ${fmt(t1)} ms): campioni, spettri, residui, voci scelte e somma del codec del tab, e i collegamenti verso i codebook con i numeri veri delle voci usate nella finestra (sul frame scelto, o su quello davanti); i token in ingresso a MusicGen (gli stessi del codec, sfasati dal delay pattern) e le probabilità in uscita, che però sono quelle del modello a conteggi del tab, perché nel browser non c'è un transformer addestrato; il log-Mel e le sinusoidi di posizione di Whisper. I token di testo di Whisper sono l'esempio della scena (formato vero, parole d'esempio). Tutto il resto è una simulazione didattica con un seme: EnCodec intero, i neuroni dentro MusicGen e Whisper, i pesi di attention. La fetta scelta è un istante: sulle parole del prompt e sui token di testo non ha un significato, e infatti restano accesi. Cambiando brano nel Laboratorio la rete si ricostruisce con i numeri del brano nuovo.`,
    seed: 20,
    axes: {
      x: "flusso: dal suono ai token e ritorno, poi dalla voce al testo",
      y: "numeri del vettore (campioni, frequenze, canali, voci del codebook)",
      z: "tempo: frame da 20 ms (nel decoder di Whisper i token di testo, nel prompt le parole)",
    },
    depth: { count: WINDOW_FRAMES, label: "frame", names: codec.frames.map((t) => `${fmt(frameMs(lab, t))} ms`) },
    sections: parts.flatMap((p) => p.sections),
    links: parts.flatMap((p) => p.links),
    // prima le parentesi delle quattro reti, poi le sotto-parti: a parità di priorità, quando due titoli
    // si sovrappongono vince quello che viene prima, e da lontano devono leggersi i nomi delle reti
    groups: [...parts.map((p) => p.groups[0]), ...parts.flatMap((p) => p.groups.slice(1))],
    sources: NETWORK_SOURCES.map((s) => ({ ...s })),
  };
}
