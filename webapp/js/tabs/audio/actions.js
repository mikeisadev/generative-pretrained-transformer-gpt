/**
 * Azioni animate del tab audio (ActionPlayer del nucleo): play/pausa, passo, velocità.
 *   1. Dal suono ai numeri         campionamento, finestre, FFT, Mel, log-Mel
 *   2. Comprimere il suono in token RVQ: codebook dopo codebook, errore che cala
 *   3. Generare musica token per token   prompt, cross-attention, delay pattern, decoder
 *   4. Dalla voce al testo (Whisper)
 * Ogni passo: enter() porta la scena allo stato iniziale del passo (così "indietro" funziona),
 * update(p) anima con p da 0 a 1 senza allocare.
 */
import { fmt } from "../../core/dom.js";
import { ACTIONS } from "./catalog.js";
import { CODEC } from "./codec.js";
import { formatPercent } from "./lab.js";
import { WHISPER } from "./dsp.js";
import { WHISPER_TOKENS } from "./sceneWhisper.js";


/** @param {object} tab vedi index.js */
export function installActions(tab) {
  const { ui, lab, parts, plots } = tab;
  const player = ui.player;
  const T = lab.frames;
  const P = parts;
  /**
   * Ogni passo dichiara il componente che deve essere inquadrato: così anche "passo indietro"
   * (o un salto con goTo) riporta la camera nel posto giusto. Se è già lì non riparte il volo.
   * tab.actionFocusId ricorda il bersaglio dell'azione anche quando lo studente guarda altrove
   * (camera libera): "Segui l'azione" torna lì (vedi index.js).
   */
  const frame = (id) => {
    tab.actionFocusId = id;
    if (tab.focusedId !== id) tab.focus(id);
  };

  /* ------------------------------------------------------------------ 1. dal suono ai numeri */
  player.register({
    ...ACTIONS.sound, // id, titolo, categoria e riassunto: catalog.js (li usa anche la ricerca)
    onStart: () => tab.resetScene(),
    onStop: () => tab.resetScene(),
    onEnd: () => tab.resetScene({ keepStage: true }),
    steps: [
      {
        title: "Il brano",
        duration: 4.8,
        caption: "Ecco il nostro brano: 4,8 secondi di melodia con accordi, generati qui con la sintesi additiva. La linea bianca scorre nel tempo come una testina.",
        enter: () => {
          frame("audio");
          tab.code("js:song.js#synthesize", ["addNote(out"]);
          P.encoder.setStage(0);
        },
        update: (p) => {
          const t = p * P.audio.duration;
          P.audio.setPlayhead(t);
          plots?.waveAt(t);
        },
        exit: () => {
          P.audio.setPlayhead(null);
          plots?.waveAt(null);
        },
      },
      {
        title: "Campionamento",
        duration: 4,
        caption: `Ingrandiamo 3 millisecondi: il computer non conserva la curva, ma ${fmt(lab.sampleRate)} misure al secondo. Ogni punto è un numero tra −1 e 1.`,
        enter: () => {
          frame("audio.lens");
          tab.code("js:song.js#addNote", ["const w = (2 * Math.PI * f) / sr", "if (f >= sr / 2) break"]);
        },
        update: (p) => P.audio.showLens("samples", 1.25 * lab.sampleRate, Math.min(1, p * 1.6)),
      },
      {
        title: "Una finestra alla volta",
        duration: 4.5,
        caption: `Un frame è un pezzo di ${CODEC.nFft} campioni (64 ms), moltiplicato per la finestra di Hann (in giallo): così i bordi vanno dolcemente a zero. Il frame successivo parte ${CODEC.hop} campioni dopo.`,
        enter: () => {
          frame("audio.lens");
          tab.code("js:dsp.js#hannWindow");
        },
        update: (p) => P.audio.showLens("frame", (60 + Math.floor(p * 6)) * CODEC.hop - CODEC.nFft / 2, Math.min(1, p * 2)),
      },
      {
        title: "FFT di ogni frame",
        duration: 6,
        caption: "La FFT scompone ogni frame nelle sue frequenze. Frame dopo frame nasce lo spettrogramma: tempo in orizzontale, frequenza in verticale, intensità come altezza e colore.",
        enter: () => {
          P.audio.showLens(null);
          frame("encoder");
          tab.code("js:dsp.js#stft", ["fft(re, im)", "Math.hypot"]);
          P.encoder.setStage(0);
          P.encoder.reveal(0);
          plots?.specUpTo(0, true);
        },
        update: (p) => {
          const n = Math.max(1, p * T);
          P.encoder.reveal(n);
          P.encoder.setSlice(Math.min(T - 1, n));
          P.audio.setPlayhead((n / T) * P.audio.duration);
          plots?.specUpTo(n);
        },
        exit: () => {
          P.encoder.reveal(T);
          P.encoder.setSlice(null);
          P.audio.setPlayhead(null);
          plots?.specUpTo(T, true);
        },
      },
      {
        title: "Asse in scala Mel",
        duration: 4,
        caption: "Ora cambiamo il righello delle frequenze: in scala Mel le basse frequenze (dove stanno le note) si allargano e le alte si stringono, come le sente l'orecchio. Guarda le etichette a sinistra.",
        enter: () => {
          frame("encoder");
          tab.code("js:dsp.js#hzToMel");
          P.encoder.setStage(0);
        },
        update: (p) => P.encoder.setStage(p),
      },
      {
        title: "Banco di filtri Mel",
        duration: 4,
        caption: "Ogni filtro triangolare somma l'energia delle frequenze vicine: 64 bande invece di 513 bin. I dettagli fini si fondono (vedi il grafico del banco di filtri).",
        enter: () => {
          frame("encoder");
          tab.code("js:dsp.js#melFilterbank", ["Math.min(up, down)"]);
          P.encoder.setStage(1);
        },
        update: (p) => P.encoder.setStage(1 + p),
      },
      {
        title: "Il logaritmo: log-Mel",
        duration: 4,
        caption: "Infine il logaritmo (decibel): anche i suoni deboli diventano visibili, come per l'orecchio. Questo log-Mel è l'ingresso tipico dei modelli vocali come Whisper.",
        enter: () => {
          frame("encoder");
          tab.code("js:dsp.js#powerToDb");
          P.encoder.setStage(2);
        },
        update: (p) => P.encoder.setStage(2 + p),
      },
      {
        title: "Dal suono ai numeri",
        duration: 4,
        caption: `Riassunto: ${fmt(lab.songs[tab.state.style].samples.length)} campioni sono diventati ${fmt(T)} frame × 64 bande = ${fmt(T * 64)} numeri, ordinati nel tempo. Ma sono ancora numeri reali, non token.`,
        enter: () => {
          frame("encoder");
          tab.code("js:dsp.js#applyMel");
          P.encoder.setStage(3);
        },
      },
    ],
  });

  /* ---------------------------------------------------------------------------- 2. RVQ */
  const rows = [0, 0, 0, 0]; // frame visibili per codebook, riusato dai passi RVQ (niente allocazioni per frame)
  const rvqLevel = (k, caption, title) => ({
    title,
    duration: k === 0 ? 7 : 5,
    caption,
    enter: () => {
      frame("rvq");
      tab.code("py:encodec.rvq", ["residual = residual - quantized"]);
      for (let i = 0; i < rows.length; i++) rows[i] = i < k ? T : 0;
      P.rvq.reveal(rows);
      plots?.errorUpTo(k, k || null);
    },
    update: (p) => {
      const n = Math.max(1, Math.round(p * T));
      for (let i = 0; i < rows.length; i++) rows[i] = i < k ? T : i === k ? n : 0;
      P.rvq.reveal(rows);
      P.rvq.setCursor(k, n - 1);
      P.rvq.setLens(n - 10, n - 1);
      P.encoder.setSlice(n - 1);
      if (k === 0) plots?.tokens(lab.codes[tab.state.style], n);
      if (p >= 1) plots?.errorUpTo(k + 1, k + 1);
    },
    exit: () => {
      P.rvq.setCursor(null);
      P.encoder.setSlice(null);
    },
  });
  player.register({
    ...ACTIONS.rvq, // id, titolo, categoria e riassunto: catalog.js (li usa anche la ricerca)
    onStart: () => tab.resetScene(),
    onStop: () => tab.resetScene(),
    onEnd: () => tab.resetScene(),
    steps: [
      {
        title: "Un vettore per frame",
        duration: 4,
        caption: `Il nostro encoder (fisso, non una rete) ogni 20 ms prende lo spettro di una finestra di 64 ms, fino a 4 kHz e in scala logaritmica: un vettore di ${CODEC.bins} numeri, ${fmt(lab.frameRate)} volte al secondo. EnCodec invece impara il suo vettore con una rete convoluzionale: 75 vettori al secondo a 24 kHz.`,
        enter: () => {
          frame("encoder");
          tab.code("js:codec.js#encodeFeatures");
          P.encoder.setStage(0);
          P.rvq.reveal(0);
          plots?.errorUpTo(0);
          plots?.tokens(lab.codes[tab.state.style], 0);
        },
        update: (p) => P.encoder.setSlice(Math.round(p * (T - 1))),
        exit: () => P.encoder.setSlice(null),
      },
      rvqLevel(0, `Codebook 1: ${lab.rvq.k} vettori tipici trovati con il k-means. Per ogni frame si sceglie il più vicino e si scrive il suo indice: è il token. Nella nuvola, il punto bianco è il frame e l'anello il centroide scelto.`, "Codebook 1: il centroide più vicino"),
      rvqLevel(1, `Codebook 2: non quantizza il frame ma l'errore rimasto (il residuo). L'errore relativo passa da ${formatPercent(lab.relativeError()[1])} a ${formatPercent(lab.relativeError()[2])}.`, "Codebook 2: il residuo"),
      rvqLevel(2, `Codebook 3: corregge l'errore lasciato dai primi due. Come pagare con banconote, poi monete, poi spiccioli.`, "Codebook 3"),
      rvqLevel(3, `Codebook 4: con 4 codebook l'errore è ${formatPercent(lab.relativeError()[4])}. MusicGen usa proprio 4 codebook (da 2048 voci ciascuno).`, "Codebook 4"),
      {
        title: "Più codebook, meno errore",
        duration: 5,
        caption: `Ogni frame ora è 4 numeri tra 0 e ${lab.rvq.k - 1}: ${fmt(lab.bitrate(4))} bit al secondo invece di ${fmt(lab.sampleRate * 16)}. Nei grafici (pulsante in alto a destra): con 8 codebook l'errore scende a ${formatPercent(lab.relativeError()[8])}. Più codebook = più bit = più qualità.`,
        enter: () => {
          frame("rvq");
          tab.code("js:codec.js#trainRvq", ["kmeans(residual"]);
          P.rvq.reveal(T);
          P.rvq.setLens(0);
          plots?.errorUpTo(CODEC.levels, 4);
        },
      },
      {
        title: "Il decoder: dai token al suono",
        duration: 5,
        caption: "Il decoder somma i centroidi scelti, ottiene uno spettro e ricostruisce le fasi con Griffin-Lim. In SoundStream ed EnCodec il decoder è invece una rete addestrata insieme all'encoder. Ascolta il risultato con «Ricostruito» nel Laboratorio.",
        enter: () => {
          frame("decoderOut");
          tab.code("js:codec.js#decodeFeatures");
          tab.showReconstruction(CODEC.used);
        },
        update: (p) => P.decoder.pulse(p),
      },
    ],
  });

  /* ---------------------------------------------------------------- 3. generazione */
  const SLOW_STEPS = 9;
  player.register({
    ...ACTIONS.generate, // id, titolo, categoria e riassunto: catalog.js (li usa anche la ricerca)
    onStart: () => tab.resetScene(),
    // a fine (o a metà) azione la scena mostra la generazione completa e il suo audio
    onStop: () => (tab.resetScene(), tab.showGenerated(tab.state.gen)),
    onEnd: () => (tab.resetScene(), tab.showGenerated(tab.state.gen)),
    build: async (ctx) => {
      const gen = tab.runGeneration(tab.state.genOptions);
      ctx.data.gen = gen;
      const words = lab.songs[gen.style].style.prompt.split(" ");
      const steps = [
        {
          title: "Il prompt",
          duration: 4,
          caption: `Si parte dal testo: "${lab.songs[gen.style].style.prompt}". Un text encoder pre-addestrato (in MusicGen è T5) lo trasforma in una sequenza di vettori, uno per pezzo di parola.`,
          enter: () => {
            frame("conditioning");
            tab.code("py:musicgen.t5", ["model: t5", "name: t5-base", "finetune: false"]);
            P.delay.show(null, -1);
          },
          update: (p) => {
            P.cond.highlightWord(Math.min(words.length - 1, Math.floor(p * words.length)));
            P.cond.glow(p);
          },
          exit: () => {
            P.cond.highlightWord(-1);
            P.cond.glow(0);
          },
        },
        {
          title: "Cross-attention sul testo",
          duration: 4,
          caption: "In ogni blocco del transformer la cross-attention consulta i vettori del testo: le query vengono dall'audio, chiavi e valori dal testo. È l'attention della lezione, ma guarda un'altra sequenza.",
          enter: () => {
            frame("genCore");
            tab.code("py:musicgen.lm", ["cross_attention_src", "input_ = sum("]);
          },
          update: (p) => {
            P.cross.userData.setStrength(0.5 + 0.5 * Math.sin(p * Math.PI * 4));
            P.transformer.pulse(p);
          },
          exit: () => {
            P.cross.userData.setStrength(0);
            P.transformer.pulse(0);
          },
        },
      ];
      for (let s = 0; s < SLOW_STEPS; s++) {
        const slots = gen.steps[s];
        const desc = slots.map(({ k, frame, token }) => (token === null ? `cb${k + 1}: S` : `cb${k + 1} → frame ${frame}: ${token}`)).join(" · ");
        steps.push({
          title: `Passo ${s}`,
          duration: s < 4 ? 2.4 : 1.4,
          caption:
            s === 0
              ? `Passo 0: solo il codebook 1 ha un frame (il frame 0); gli altri scrivono il token speciale S. ${desc}`
              : s < 4
                ? `Passo ${s}: il codebook k scrive il frame ${s} − k, così vede già il codebook sopra dello stesso frame. ${desc}`
                : `Passo ${s}: ${desc}`,
          enter: () => {
            frame("delay");
            tab.code(s === 0 ? "py:musicgen.delay" : "js:codec.js#TokenModel", s === 0 ? ["[S, 1, 2, 3, 4]"] : ["distribution(style", "delayPattern(T"]);
            P.delay.show(gen, s);
            const first = slots[0];
            if (first.p) plots?.nextToken(first.p, first.token);
          },
          update: (p) => P.transformer.pulse(Math.sin(p * Math.PI), s % 4),
          exit: () => P.transformer.pulse(0),
        });
      }
      let lastShown = -1;
      steps.push({
        title: "Tutti gli altri passi",
        duration: 6,
        caption: `E così via: ${gen.steps.length} passi per ${T} frame (4,8 s). Con il delay pattern i passi sono circa 50 al secondo di audio invece di 200 (MusicGen fa lo stesso).`,
        enter: () => {
          frame("delay");
          tab.code("js:codec.js#delayPattern");
          lastShown = -1;
        },
        update: (p) => {
          const s = Math.min(gen.steps.length - 1, SLOW_STEPS + Math.floor(p * (gen.steps.length - SLOW_STEPS)));
          if (s - lastShown < 3 && p < 1) return; // la lente si ridisegna al più ogni 3 passi
          lastShown = s;
          P.delay.show(gen, s);
          P.transformer.pulse(0.5 + 0.5 * Math.sin(s));
          const first = gen.steps[s][0];
          if (first.p) plots?.nextToken(first.p, first.token);
        },
        exit: () => P.transformer.pulse(0),
      });
      steps.push({
        title: "Decoder: ascolta il risultato",
        duration: 5,
        caption: "I token generati passano nel decoder del codec e tornano suono. Premi «Generato» nel Laboratorio per ascoltarlo: è un brano nuovo, ricombinato dai pezzi di suono che il modello conosce.",
        enter: () => {
          frame("decoderOut");
          tab.code("js:codec.js#decodeFeatures");
          P.delay.show(gen, gen.steps.length - 1);
          tab.showGenerated(gen);
        },
        update: (p) => P.decoder.pulse(p),
      });
      return steps;
    },
  });

  /* ------------------------------------------------------------------- 4. Whisper */
  player.register({
    ...ACTIONS.whisper, // id, titolo, categoria e riassunto: catalog.js (li usa anche la ricerca)
    onStart: () => tab.resetScene(),
    onStop: () => tab.resetScene(),
    onEnd: () => tab.resetScene(),
    steps: [
      {
        title: "30 secondi a 16 kHz",
        duration: 4,
        caption: `Whisper lavora sempre su blocchi da 30 secondi di audio a 16 kHz. Il log-Mel si calcola con finestre da ${WHISPER.nFft} campioni (25 ms) ogni ${WHISPER.hop} (10 ms): ${fmt(WHISPER.nFrames)} frame. Il nostro brano ne occupa ${lab.whisper(tab.state.style).signalFrames}, il resto è silenzio.`,
        enter: () => {
          frame("wMel");
          tab.code("py:whisper.params", ["N_FFT", "HOP_LENGTH", "CHUNK_LENGTH"]);
          P.whisper.revealMel(0);
        },
        update: (p) => P.whisper.revealMel(p),
        exit: () => P.whisper.revealMel(null),
      },
      {
        title: "log-Mel a 80 bande",
        duration: 4,
        caption: "Stessa formula di whisper/audio.py: potenza della STFT, 80 filtri Mel, log10, dinamica limitata a 8 (80 dB) e riscalatura. Il calcolo qui coincide con quello di PyTorch.",
        enter: () => {
          frame("wMel");
          tab.code("py:whisper.logmel", ["torch.stft", "filters @ magnitudes", "log_spec.max() - 8.0"]);
        },
      },
      {
        title: "Encoder",
        duration: 5,
        caption: "Due convoluzioni 1D (la seconda con passo 2: 3000 frame → 1500 posizioni), una codifica di posizione sinusoidale e una pila di blocchi transformer. L'attention è bidirezionale: ogni istante vede tutto l'audio.",
        enter: () => {
          frame("wEnc");
          tab.code("py:whisper.encoder", ["self.conv1", "stride=2", "positional_embedding"]);
        },
        update: (p) => P.whisper.pulse("enc", Math.sin(p * Math.PI)),
        exit: () => P.whisper.pulse("enc", 0),
      },
      {
        title: "Decoder e token speciali",
        duration: 7,
        caption: "Il decoder è causale come il GPT e consulta l'encoder con la cross-attention. Prima scrive i token speciali (inizio, lingua, compito, niente timestamp), poi il testo. Qui il testo è un esempio del formato.",
        enter: () => {
          frame("whisperOut"); // decoder (che pulsa) e testo insieme
          tab.code("py:whisper.tokens", ["<|startoftranscript|>", "<|transcribe|>", "<|notimestamps|>"]);
          P.whisper.showTokens(0);
        },
        update: (p) => {
          P.whisper.showTokens(Math.min(WHISPER_TOKENS.length, Math.floor(p * (WHISPER_TOKENS.length + 0.99))));
          P.whisper.pulse("dec", Math.sin(p * Math.PI * 3) * 0.5 + 0.5);
          P.whisper.pulse("cross", Math.sin(p * Math.PI * 3) * 0.5 + 0.5);
        },
        exit: () => {
          P.whisper.showTokens(WHISPER_TOKENS.length);
          P.whisper.pulse("dec", 0);
          P.whisper.pulse("cross", 0);
        },
      },
      {
        title: "Un modello, tanti compiti",
        duration: 5,
        caption: "Cambiando i token iniziali lo stesso modello trascrive, traduce verso l'inglese o riconosce la lingua. Whisper è stato addestrato su 680.000 ore di audio con trascrizioni prese dal web.",
        enter: () => {
          frame("whisperAll");
          tab.code("py:whisper.sot", ["sot_sequence.append"]);
        },
      },
    ],
  });
}
