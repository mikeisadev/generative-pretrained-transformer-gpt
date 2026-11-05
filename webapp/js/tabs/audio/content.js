/**
 * Testi del tab "Audio e musica": percorso didattico (menu a fisarmonica) e fonti.
 *
 * Ogni sottopunto: { id, title, text (HTML nostro), math?, focus (id del registro 3D),
 * code ("py:…" | "js:file#nome"), highlight? }. I segnaposto {{…}} vengono dai calcoli veri
 * del laboratorio (vedi index.js → vars).
 *
 * Regola: ogni numero sui modelli reali viene dai paper o dal codice ufficiale elencati in
 * SOURCES; i numeri del "nostro" codec sono calcolati nel browser.
 */

export const STEPS = [
  {
    id: "sound",
    title: "1 · Il suono è un'onda",
    children: [
      {
        id: "sound.wave",
        title: "Forma d'onda e campionamento",
        focus: "audio",
        code: "js:song.js#synthesize",
        highlight: ["addNote(out"],
        text: `<p>Il suono è aria che vibra: la pressione sale e scende centinaia o migliaia di volte al secondo.
          Un microfono trasforma questa pressione in una tensione elettrica, cioè in una curva continua: la <strong>forma d'onda</strong>.</p>
          <p>Il computer non può salvare una curva continua: la <strong>campiona</strong>, cioè ne misura l'altezza a intervalli regolari.
          Il nostro brano usa <strong>{{sampleRate}} campioni al secondo</strong> (16 kHz, la stessa frequenza che usa Whisper):
          {{duration}} secondi diventano <strong>{{samples}} numeri</strong>.</p>
          <p>Analogia: è come un cartone animato. Tante foto al secondo danno l'illusione del movimento continuo.</p>
          <p class="hint">Il brano è generato qui nel browser con la <em>sintesi additiva</em>: ogni nota è una somma di sinusoidi (la fondamentale e le armoniche) con un inviluppo di volume.</p>`,
        math: [
          { tex: "x[n] = x\\!\\left(\\tfrac{n}{f_s}\\right),\\quad n = 0, 1, 2, \\dots", note: "Il campione n è il valore della curva all'istante n/f_s." },
          { tex: "f_{\\max} = \\frac{f_s}{2} = \\frac{16\\,000}{2} = 8\\,000\\ \\text{Hz}", note: "Teorema del campionamento (Nyquist-Shannon): con f_s campioni al secondo si rappresentano frequenze fino a f_s/2. Per questo il sintetizzatore scarta le armoniche sopra 8 kHz." },
        ],
      },
      {
        id: "sound.vs-text",
        title: "Cosa cambia rispetto al testo",
        focus: "overview",
        code: "js:codec.js#encodeFeatures",
        text: `<p>Il GPT della lezione lavora su <strong>token discreti</strong>: 65 caratteri possibili, uno dopo l'altro. Il testo nasce già "a pezzi".</p>
          <p>L'audio invece è <strong>continuo</strong> e <strong>lunghissimo</strong>: {{samples}} campioni per {{duration}} secondi, mentre una frase di {{duration}} secondi ha qualche decina di caratteri.
          Per usare la stessa idea del "prossimo token" servono due passi in più:</p>
          <ul>
            <li><strong>comprimere nel tempo</strong>: un vettore ogni 20 ms invece di un numero ogni 1/16.000 di secondo;</li>
            <li><strong>discretizzare</strong>: trasformare ogni vettore in pochi numeri interi (i token) presi da un vocabolario fisso, il <em>codebook</em>.</li>
          </ul>
          <p>Questo lavoro lo fa un <strong>codec neurale</strong> (encoder + quantizzazione + decoder). Poi un transformer può generare token audio esattamente come il GPT genera caratteri, e il decoder del codec li trasforma di nuovo in suono.</p>`,
      },
    ],
  },
  {
    id: "freq",
    title: "2 · Dal tempo alle frequenze",
    children: [
      {
        id: "freq.frames",
        title: "Finestre e frame",
        focus: "audio.lens",
        code: "js:dsp.js#hannWindow",
        text: `<p>Una nota dura decine di millisecondi, un brano minuti: per capire "quali note suonano adesso" si guarda il segnale a pezzetti, i <strong>frame</strong>.</p>
          <p>Qui ogni frame è lungo {{nFft}} campioni (64 ms) e il successivo parte {{hop}} campioni dopo (20 ms): i frame si sovrappongono, così nessun istante resta scoperto.
          Ogni frame viene moltiplicato per la <strong>finestra di Hann</strong>, una campana che va a zero ai bordi: senza, il taglio netto creerebbe frequenze finte.</p>`,
        math: [{ tex: "w[n] = \\tfrac12 - \\tfrac12\\cos\\!\\left(\\frac{2\\pi n}{N}\\right),\\quad 0 \\le n < N", note: "Finestra di Hann periodica, la stessa di torch.hann_window." }],
      },
      {
        id: "freq.fft",
        title: "FFT e spettrogramma",
        focus: "encoder",
        code: "js:dsp.js#stft",
        highlight: ["fft(re, im)", "Math.hypot"],
        text: `<p>La <strong>trasformata di Fourier</strong> risponde alla domanda: "quanto di ogni frequenza c'è in questo frame?". È come un prisma che separa la luce bianca nei colori.</p>
          <p>La <strong>FFT</strong> (Fast Fourier Transform) è l'algoritmo veloce per calcolarla. Applicata a tutti i frame (STFT) dà lo <strong>spettrogramma</strong>: tempo in orizzontale, frequenza in verticale, intensità come altezza e colore.
          Nella scena vedi le righe parallele delle armoniche: ogni nota è una fondamentale più i suoi multipli.</p>
          <p>Il nostro spettrogramma ha {{frames}} frame × {{bins}} frequenze, calcolato qui con una FFT radix-2 scritta in JavaScript.</p>`,
        math: [
          { tex: "X[t,k] = \\sum_{n=0}^{N-1} x[n + tH]\\, w[n]\\, e^{-2\\pi i\\, k n / N}", note: "STFT: frame t (passo H = {{hop}}), bin k. La DFT di un frame è la somma dei campioni pesati da onde complesse." },
          { tex: "f_k = \\frac{k\\, f_s}{N} = k \\cdot \\frac{16\\,000}{1024} \\approx k \\cdot 15{,}6\\ \\text{Hz}", note: "Il bin k corrisponde a questa frequenza: la risoluzione è f_s/N." },
          { tex: "\\text{FFT: } O(N \\log N) \\quad\\text{invece di}\\quad O(N^2)", note: "Per N = 1024: circa 10.000 operazioni invece di un milione." },
        ],
      },
      {
        id: "freq.mel",
        title: "Scala Mel e log-Mel",
        focus: "encoder",
        code: "js:dsp.js#melFilterbank",
        highlight: ["hzToMel(fMin", "Math.min(up, down)"],
        text: `<p>L'orecchio non sente le frequenze in modo lineare: tra 100 e 200 Hz sentiamo un salto grande (un'ottava), tra 7.000 e 7.100 Hz quasi nulla.
          La <strong>scala Mel</strong> misura le frequenze "come le percepiamo": fitta in basso, rada in alto.</p>
          <p>Un <strong>banco di filtri Mel</strong> (triangoli sempre più larghi, vedi i grafici) somma l'energia dei bin vicini in {{mels}} bande. Poi si prende il <strong>logaritmo</strong>, perché anche il volume lo percepiamo in modo logaritmico (decibel).
          Il risultato, il <strong>log-Mel</strong>, è l'ingresso tipico dei modelli vocali come Whisper.</p>
          <p class="hint">Esistono due varianti diffuse della formula: HTK e Slaney (lineare sotto 1 kHz, logaritmica sopra). Qui usiamo Slaney, il default di librosa, la libreria con cui sono stati salvati i filtri di Whisper.</p>`,
        math: [
          { tex: "m_{\\text{HTK}}(f) = 2595 \\log_{10}\\!\\left(1 + \\frac{f}{700}\\right)", note: "Variante HTK." },
          { tex: "m_{\\text{Slaney}}(f) = \\begin{cases} \\dfrac{3f}{200} & f < 1000\\ \\text{Hz} \\\\[4pt] 15 + \\dfrac{27\\,\\ln(f/1000)}{\\ln 6{,}4} & f \\ge 1000\\ \\text{Hz} \\end{cases}", note: "Variante Slaney (librosa, htk=False)." },
          { tex: "S_{\\text{mel}}[t,m] = \\sum_k M[m,k]\\, |X[t,k]|^2, \\qquad L = \\log_{10}\\max(S_{\\text{mel}}, 10^{-10})", note: "Potenza filtrata e logaritmo: è il calcolo di whisper/audio.py." },
        ],
      },
    ],
  },
  {
    id: "codec",
    title: "3 · Il codec neurale: suono → token → suono",
    children: [
      {
        id: "codec.encoder",
        title: "Encoder: un vettore ogni pochi millisecondi",
        focus: "encoder",
        code: "py:encodec.24k",
        highlight: ["SEANetEncoder", "bins=1024"],
        text: `<p>Un codec neurale come <strong>SoundStream</strong> (Google, 2021) o <strong>EnCodec</strong> (Meta, 2022) ha un <strong>encoder convoluzionale</strong>: strati di convoluzioni che riducono via via la risoluzione nel tempo.
          In EnCodec a 24 kHz il fattore di riduzione è 8·5·4·2 = 320: esce <strong>un vettore ogni 320 campioni, cioè 75 al secondo</strong>. EnCodec aggiunge anche strati LSTM.</p>
          <p>Nel <strong>nostro codec giocattolo</strong> (simulazione didattica) l'encoder è fisso e non impara: per ogni frame da 20 ms prende lo spettro di ampiezza fino a 4 kHz in scala logaritmica, un vettore di {{bins}} numeri.
          Così si vede bene l'idea senza dover addestrare una rete.</p>`,
      },
      {
        id: "codec.rvq",
        title: "Quantizzazione vettoriale residua (RVQ)",
        focus: "rvq",
        code: "py:encodec.rvq",
        highlight: ["residual = residual - quantized", "quantized_out = quantized_out + quantized"],
        text: `<p>Un <strong>codebook</strong> è un elenco di K vettori "tipici" (centroidi). Quantizzare un vettore vuol dire sostituirlo con l'<strong>indice</strong> del centroide più vicino: quel numero intero è il <strong>token</strong>.</p>
          <p>Un solo codebook è grossolano. La RVQ ne usa diversi in cascata: il primo approssima il vettore, il secondo quantizza l'<strong>errore residuo</strong> lasciato dal primo, il terzo l'errore lasciato dai primi due, e così via.
          È come pagare 47,30 euro: prima 40 (banconote), poi 7 (monete grandi), poi 0,30 (spiccioli).</p>
          <p>Nel nostro codec i codebook hanno K = {{k}} centroidi, ottenuti con il <strong>k-means</strong> sui frame dei due brani. Con 1 codebook l'errore relativo è {{err1}}, con 4 scende a {{err4}}, con 8 a {{err8}} (vedi il grafico).</p>`,
        math: [
          { tex: "r_0 = z,\\qquad q_\\ell = \\arg\\min_{c} \\lVert r_{\\ell-1} - C_\\ell[c] \\rVert^2,\\qquad r_\\ell = r_{\\ell-1} - C_\\ell[q_\\ell]", note: "Ogni livello quantizza il residuo del precedente (algoritmo 1 di SoundStream)." },
          { tex: "\\hat z = \\sum_{\\ell=1}^{N_q} C_\\ell[q_\\ell]", note: "Il decoder riceve la somma dei centroidi scelti." },
        ],
      },
      {
        id: "codec.bitrate",
        title: "Quanti bit al secondo",
        focus: "rvq",
        code: "py:encodec.24k",
        highlight: ["n_q = int"],
        text: `<p>Ogni token di un codebook da K voci costa log<sub>2</sub>K bit. EnCodec a 24 kHz ha codebook da 1024 voci (10 bit): con 8 codebook a 75 frame al secondo fa 75 × 8 × 10 = <strong>6.000 bit/s</strong>, cioè la banda da 6 kbps del modello; con 2 codebook 1,5 kbps, con 32 codebook 24 kbps.</p>
          <p>Il nostro codec: {{frameRate}} frame/s × 4 codebook × 6 bit = <strong>{{bitrate4}} bit/s</strong>, contro i {{pcmBitrate}} bit/s dell'audio originale a 16 bit. Il prezzo è la qualità: ascolta la ricostruzione nel pannello Laboratorio.</p>
          <p>SoundStream applica il <em>quantizer dropout</em> durante l'addestramento (a volte usa meno codebook), così un solo modello funziona da 3 a 18 kbps.</p>`,
        math: [{ tex: "R = f_{\\text{frame}} \\cdot N_q \\cdot \\log_2 K", note: "Bit al secondo di un codec RVQ." }],
      },
      {
        id: "codec.decoder",
        title: "Decoder e addestramento",
        focus: "decoder",
        code: "js:dsp.js#griffinLim",
        text: `<p>Il <strong>decoder</strong> fa il cammino inverso: dai vettori quantizzati ricostruisce la forma d'onda con convoluzioni che aumentano la risoluzione nel tempo.
          In SoundStream ed EnCodec encoder, RVQ e decoder si addestrano <strong>insieme</strong>: la loss confronta l'audio ricostruito con l'originale (anche sugli spettrogrammi) e un <strong>discriminatore</strong> avversario premia il suono che "sembra vero".</p>
          <p>Il nostro decoder giocattolo non è una rete: rimette insieme lo spettro e stima le fasi mancanti con l'algoritmo di <strong>Griffin-Lim</strong>. Per questo suona più "metallico" di un codec vero.</p>`,
      },
    ],
  },
  {
    id: "gen",
    title: "4 · Generare audio con un transformer",
    children: [
      {
        id: "gen.next",
        title: "Lo stesso gioco del prossimo token",
        focus: "transformer",
        code: "py:musicgen.lm",
        highlight: ["input_ = sum(", "logits = torch.stack"],
        text: `<p>Una volta che l'audio è una sequenza di token, un transformer decoder lo genera <strong>un passo alla volta</strong>, esattamente come il GPT della lezione genera caratteri: guarda i token passati (attention causale) e produce la probabilità del prossimo.</p>
          <p>La differenza: a ogni istante ci sono <strong>più token</strong>, uno per codebook. MusicGen (Meta, 2023) usa 4 codebook di EnCodec a 32 kHz, 50 frame al secondo, con 2048 voci per codebook.
          In ingresso <strong>somma</strong> gli embedding dei 4 token; in uscita ha <strong>4 teste lineari</strong>, una per codebook (vedi il codice a destra).</p>`,
        math: [
          { tex: "p(x_{1:S} \\mid c) = \\prod_{s=1}^{S} p(x_s \\mid x_{<s},\\, c)", note: "Modello autoregressivo condizionato dal testo c: la stessa formula del GPT." },
          { tex: "h_s = \\sum_{k=1}^{4} E_k[x_{k,s}],\\qquad \\text{logits}_k = W_k\\, \\text{Transformer}(h)_s", note: "Embedding sommati, una testa per codebook (audiocraft/models/lm.py)." },
        ],
      },
      {
        id: "gen.delay",
        title: "Il delay pattern",
        focus: "delay",
        code: "py:musicgen.delay",
        highlight: ["[S, 1, 2, 3, 4]", "[S, S, 1, 2, 3]"],
        text: `<p>Generare i 4 codebook uno dopo l'altro costerebbe 4 passi per frame. Generarli tutti insieme ignorerebbe che il codebook 2 dipende dal codebook 1 dello stesso frame.</p>
          <p>MusicGen sceglie una via di mezzo: <strong>sfasa</strong> i codebook di un passo ciascuno (ritardi 0, 1, 2, 3). Al passo s il modello scrive il codebook 1 del frame s, il codebook 2 del frame s−1, il 3 del frame s−2, il 4 del frame s−3.
          Così quando scrive il codebook 2 di un frame, il codebook 1 di quel frame è già pronto. Risultato: circa <strong>50 passi per secondo di audio</strong> invece di 200.</p>
          <p>Nella scena la "scala" di celle mostra proprio questo sfasamento; le celle grigie <em>S</em> sono token speciali dove il frame non esiste ancora.</p>`,
        math: [{ tex: "\\text{passo } s:\\quad x_{k,\\; s - d_k},\\qquad d = [0, 1, 2, 3]", note: "Ritardi di MusicGen (config/model/lm/musicgen_lm.yaml)." }],
      },
      {
        id: "gen.text",
        title: "Condizionamento testuale (T5 e cross-attention)",
        focus: "t5",
        code: "py:musicgen.t5",
        highlight: ["model: t5", "name: t5-base", "cross: [description]", "finetune: false"],
        text: `<p>Come dire al modello "musica allegra"? La descrizione passa da un <strong>text encoder</strong> pre-addestrato: in MusicGen è <strong>T5</strong> (nella configurazione t5-base, non riaddestrato).
          I vettori del testo entrano nel transformer con la <strong>cross-attention</strong>: le query vengono dai token audio, le chiavi e i valori dal testo. È la stessa attention della lezione, ma "guarda" un'altra sequenza.</p>
          <p>Con la <strong>classifier-free guidance</strong> il modello calcola due previsioni, con e senza testo, e spinge nella direzione del testo (coefficiente 3 nella configurazione di default).</p>
          <p class="hint">Nel nostro modello giocattolo il "testo" sceglie semplicemente quali statistiche usare (stile allegro o calmo): è un condizionamento vero, ma molto più semplice di T5.</p>`,
        math: [{ tex: "\\ell = \\ell_{\\varnothing} + \\gamma\\,(\\ell_{c} - \\ell_{\\varnothing}),\\qquad \\gamma = 3", note: "Classifier-free guidance sui logits (audiocraft/models/lm.py)." }],
      },
      {
        id: "gen.toy",
        title: "Il nostro generatore (simulazione didattica)",
        focus: "delay",
        code: "js:codec.js#TokenModel",
        highlight: ["distribution(style", "delayPattern(T"],
        text: `<p>Un transformer vero richiede ore di addestramento su migliaia di ore di musica (MusicGen: 20.000 ore di musica con licenza). Qui lo sostituiamo con un <strong>modello a conteggi</strong> calcolato sui token dei due brani:</p>
          <ul>
            <li>per il codebook 1 guarda il token precedente;</li>
            <li>per gli altri guarda anche il codebook sopra nello stesso frame, che grazie al delay pattern è già stato generato.</li>
          </ul>
          <p>È poco più di una catena di Markov, quindi ricorda solo il passato immediato. Il meccanismo però è identico: probabilità del prossimo token, campionamento con temperatura, delay pattern, decoder del codec.</p>`,
      },
    ],
  },
  {
    id: "asr",
    title: "5 · Dalla voce al testo: Whisper",
    children: [
      {
        id: "asr.input",
        title: "L'ingresso: 30 secondi di log-Mel",
        focus: "wMel",
        code: "py:whisper.logmel",
        highlight: ["torch.stft", "filters @ magnitudes", "log_spec.max() - 8.0"],
        text: `<p><strong>Whisper</strong> (OpenAI, 2022) fa il percorso opposto: ascolta e scrive. Non usa token audio discreti: riceve direttamente il <strong>log-Mel</strong>.</p>
          <p>I parametri sono fissi nel codice: audio a 16 kHz, finestre da 400 campioni (25 ms), passo 160 (10 ms), 80 bande Mel, blocchi da <strong>30 secondi = 3000 frame</strong>. Se l'audio è più corto si aggiunge silenzio.</p>
          <p>Qui il log-Mel è calcolato con la stessa formula e gli stessi filtri: il nostro brano occupa {{whisperFrames}} frame, il resto è silenzio di riempimento (la parte scura).</p>`,
      },
      {
        id: "asr.encoder",
        title: "Encoder: convoluzioni + transformer",
        focus: "wEnc",
        code: "py:whisper.encoder",
        highlight: ["self.conv1", "stride=2", "positional_embedding"],
        text: `<p>L'encoder di Whisper applica due convoluzioni 1D (la seconda con passo 2, quindi 3000 frame diventano 1500 posizioni), aggiunge una codifica di posizione sinusoidale e poi una pila di blocchi transformer con attention <strong>bidirezionale</strong>: ogni posizione vede tutto l'audio.</p>
          <p>È la differenza con il GPT: lì l'attention è causale (si guarda solo il passato), qui l'encoder può guardare anche il "futuro" perché l'audio intero è già disponibile.</p>`,
      },
      {
        id: "asr.decoder",
        title: "Decoder: testo token per token",
        focus: "wDec",
        code: "py:whisper.tokens",
        highlight: ["<|startoftranscript|>", "<|transcribe|>", "<|notimestamps|>"],
        text: `<p>Il decoder è un transformer causale, come il GPT: genera il testo un token alla volta e, con la <strong>cross-attention</strong>, consulta l'uscita dell'encoder.</p>
          <p>Il compito si sceglie con <strong>token speciali</strong> all'inizio: <code>&lt;|startoftranscript|&gt;</code>, la lingua (es. <code>&lt;|it|&gt;</code>), il compito (<code>&lt;|transcribe|&gt;</code> o <code>&lt;|translate|&gt;</code>) e <code>&lt;|notimestamps|&gt;</code> se non servono i tempi.
          Un solo modello fa trascrizione, traduzione verso l'inglese e riconoscimento della lingua.</p>
          <p>È stato addestrato su 680.000 ore di audio con trascrizioni prese dal web (supervisione "debole"). Le dimensioni vanno da 39 milioni di parametri (tiny) a 1,55 miliardi (large).</p>
          <p class="hint">Nella scena il testo d'uscita è un esempio: il nostro audio è musica senza voce, e qui non eseguiamo Whisper.</p>`,
      },
    ],
  },
  {
    id: "family",
    title: "6 · Altri modi di generare audio",
    children: [
      {
        id: "family.audiolm",
        title: "AudioLM: token semantici e acustici",
        focus: "rvq",
        code: "py:encodec.rvq",
        text: `<p><strong>AudioLM</strong> (Google, 2022) usa due tipi di token. I <strong>token semantici</strong> vengono da un modello auto-supervisionato (w2v-BERT): si prendono le attivazioni di uno strato intermedio e si raggruppano con k-means. Catturano la struttura a lungo termine (cosa si dice, come prosegue la melodia) ma da soli ricostruiscono male il suono.</p>
          <p>I <strong>token acustici</strong> vengono da SoundStream e catturano i dettagli fini (timbro, voce). AudioLM genera prima i token semantici, poi quelli acustici "grossolani" e infine quelli "fini": tre stadi in cascata.
          MusicGen invece usa un solo stadio con il delay pattern.</p>`,
      },
      {
        id: "family.valle",
        title: "Sintesi vocale con token di codec (VALL-E)",
        focus: "transformer",
        code: "py:musicgen.lm",
        text: `<p><strong>VALL-E</strong> (Microsoft, 2023) tratta la sintesi vocale (TTS) come un problema di modello linguistico condizionato: in ingresso il testo (fonemi) e <strong>3 secondi di voce</strong> di chi deve parlare, in uscita i token di un codec neurale.</p>
          <p>Addestrato su 60.000 ore di parlato inglese (Libri-Light), riesce a imitare una voce mai vista, e anche l'emozione e l'ambiente acustico del campione. È lo stesso schema di questo tab: codec + "prossimo token".</p>`,
      },
      {
        id: "family.diffusion",
        title: "Diffusione latente per l'audio (Stable Audio)",
        focus: "decoder",
        code: "js:codec.js#decodeFeatures",
        text: `<p>Non tutti i generatori audio sono autoregressivi. <strong>Stable Audio</strong> (Stability AI, 2024) usa la <strong>diffusione latente</strong>, come i generatori di immagini: un autoencoder variazionale (VAE) convoluzionale comprime l'audio in un latente continuo, e un modello di diffusione impara a "togliere il rumore" da quel latente, guidato dal testo.</p>
          <p>In più riceve il <strong>tempo</strong> (inizio e durata) come condizionamento, così può generare brani di lunghezza scelta. Nel paper genera fino a 95 secondi stereo a 44,1 kHz in circa 8 secondi su una GPU A100.</p>
          <p class="hint">Vedi anche il tab "Immagini e video" per la diffusione.</p>`,
      },
    ],
  },
];

/** Fonti del tab. Le pagine arXiv sono gli abstract ufficiali; i repository GitHub il codice citato. */
export const SOURCES = [
  {
    group: "Codec neurali",
    items: [
      { title: "Zeghidour et al. (2021), SoundStream: An End-to-End Neural Audio Codec", url: "https://arxiv.org/abs/2107.03312", note: "Encoder/decoder convoluzionali + RVQ addestrati insieme; quantizer dropout: un modello da 3 a 18 kbps." },
      { title: "Défossez, Copet, Synnaeve, Adi (2022), High Fidelity Neural Audio Compression (EnCodec)", url: "https://arxiv.org/abs/2210.13438", note: "Encoder-decoder in streaming con spazio latente quantizzato e discriminatore multiscala." },
      { title: "facebookresearch/encodec (codice ufficiale)", url: "https://github.com/facebookresearch/encodec", note: "24 kHz: 1,5/3/6/12/24 kbps con 2/4/8/16/32 codebook; codebook da 1024 voci; riduzione 8·5·4·2 = 320 (75 frame/s)." },
    ],
  },
  {
    group: "Modelli linguistici su token audio",
    items: [
      { title: "Borsos et al. (2022), AudioLM: a Language Modeling Approach to Audio Generation", url: "https://arxiv.org/abs/2209.03143", note: "Token semantici (w2v-BERT + k-means) e acustici (SoundStream), tre stadi." },
      { title: "Copet et al. (2023), Simple and Controllable Music Generation (MusicGen)", url: "https://arxiv.org/abs/2306.05284", note: "Transformer a stadio singolo su 4 codebook EnCodec a 32 kHz e 50 Hz, delay pattern." },
      { title: "facebookresearch/audiocraft: documentazione di MusicGen", url: "https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md", note: "4 codebook a 50 Hz, 50 passi per secondo, 20.000 ore di musica, modelli da 300M a 3,3B." },
      { title: "audiocraft: musicgen_lm.yaml e text2music.yaml", url: "https://github.com/facebookresearch/audiocraft/tree/main/config", note: "Ritardi [0, 1, 2, 3], 2048 voci per codebook, T5-base con cross-attention, guidance 3,0." },
      { title: "Wang et al. (2023), Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)", url: "https://arxiv.org/abs/2301.02111", note: "TTS come modello linguistico su token di codec; 60.000 ore; prompt vocale di 3 secondi." },
    ],
  },
  {
    group: "Riconoscimento vocale e diffusione",
    items: [
      { title: "Radford et al. (2022), Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)", url: "https://arxiv.org/abs/2212.04356", note: "680.000 ore di audio multilingue e multitask." },
      { title: "openai/whisper (codice ufficiale)", url: "https://github.com/openai/whisper", note: "audio.py: 16 kHz, n_fft 400, hop 160, 30 s, 80 bande; model.py: due Conv1d (stride 2); tokenizer.py: token speciali." },
      { title: "Evans et al. (2024), Fast Timing-Conditioned Latent Audio Diffusion (Stable Audio)", url: "https://arxiv.org/abs/2402.04825", note: "Diffusione latente su VAE convoluzionale, condizionamento su testo e tempo." },
      { title: "Stability-AI/stable-audio-tools (codice ufficiale)", url: "https://github.com/Stability-AI/stable-audio-tools", note: "Codice di training e inferenza." },
    ],
  },
  {
    group: "Elaborazione del segnale",
    items: [
      { title: "librosa: mel_frequencies (varianti HTK e Slaney della scala Mel)", url: "https://librosa.org/doc/main/generated/librosa.mel_frequencies.html", note: "Formula HTK 2595·log10(1 + f/700); Slaney lineare sotto 1 kHz." },
    ],
  },
];
