/**
 * Testi della vista "Architettura LLM moderni completa" (SOLI DATI): il Percorso (14 passi) e le
 * schede di sintesi della scena. Ogni numero viene dalla ricerca verificata (full/models/*.js,
 * full/research.js); ciò che non è verificato è detto nel testo.
 *
 * Campi dei sottopunti del percorso (letti da full/index.js; nomi diversi da quelli della vista
 * d'insieme, così i test di quella restano validi):
 *   anchor  ancora della scena da inquadrare (full/model.js, anchorsOf)   layer  layer del Dettaglio
 *   play    id dell'azione animata (full/catalog.js)                       code   chiave di full/snippets.js
 *   charts  id della sezione di grafici
 * Il percorso segue il modello principale: scegliendo un passo con un altro modello attivo si torna a lui.
 */

export const FULL_PATH = [
  {
    id: "full-sec-model",
    title: "1. Un modello vero, smontato pezzo per pezzo",
    subtab: "full",
    children: [
      {
        id: "full-intro",
        title: "DeepSeek-V4.1-Flash, con i pesi aperti",
        anchor: "blocks:layers",
        code: "config",
        text: `<p>Nella vista d'insieme hai visto 16 meccanismi uno alla volta. Qui li vedi lavorare <strong>tutti insieme</strong> dentro un modello vero con i pesi aperti: <strong>DeepSeek-V4.1-Flash</strong> (checkpoint 1017, ottobre 2026, licenza MIT). È il più piccolo della nuova famiglia DeepSeek, eppure ha 552 miliardi di parametri nel «corpo» (backbone) più 196 miliardi in una memoria a parte (Engram).</p>
<p>Come si conoscono numeri così precisi? Il file di configurazione non era scaricabile da qui, quindi è stato <strong>ricostruito campo per campo</strong> dal codice ufficiale di DeepSeek (il motore DeepInfer), dalle configurazioni di NVIDIA NeMo e vLLM e dagli estratti del file ufficiale. La prova del nove: contando tensore per tensore si ottengono esattamente i <strong>96.085 tensori</strong> e i <strong>510,3 GB</strong> dichiarati. Ciò che non si è potuto verificare porta il segno <em>da verificare</em>.</p>
<p>È come studiare un'auto di serie con il manuale d'officina in mano, invece di guardarne solo la brochure. Con il selettore in alto a destra puoi smontare anche Kimi K3.1 Pro e gpt-oss-120b.</p>`,
      },
      {
        id: "full-zoom",
        title: "Tre zoom e 16 segnaposto",
        anchor: "blocks:layers",
        text: `<p>La scena ha tre livelli, come la vista d'insieme: da lontanissimo la <strong>Panoramica</strong> (dai dati al token generato), poi i <strong>Blocchi</strong> (l'architettura da sinistra a destra) e da vicino il <strong>Dettaglio</strong> (dentro un layer). Usa la rotellina, oppure il selettore dello zoom in alto a sinistra.</p>
<p>I cerchietti numerati sono i <strong>16 meccanismi</strong> della vista d'insieme, ognuno nel punto in cui vive in questo modello: il 2 (Mixture of Experts) sulla griglia degli esperti, il 5 (tokenizer BPE) sul tokenizer, il 6 (KV cache) sulla riga delle cache… Un cerchietto grigio tratteggiato vuol dire «qui vivrebbe, ma questo modello non lo usa».</p>
<p>Clic su un cerchietto: passi alla vista d'insieme, proprio su quel meccanismo; il pulsante <em>← Torna all'architettura completa</em> ti riporta qui, esattamente dove eri.</p>`,
      },
    ],
  },
  {
    id: "full-sec-weights",
    title: "2. Dai dati ai pesi",
    subtab: "full",
    children: [
      {
        id: "full-data",
        title: "Dati, pre-training e contesto lungo",
        anchor: "macro:data",
        play: "full-train",
        charts: "full-scala",
        text: `<p>Tutto parte dai dati: DeepSeek dichiara un pre-training da zero su <strong>45 trilioni di token</strong> di testo e immagini. Il compito è lo stesso del nostro GPT: prevedere il token successivo.</p>
<p>Quanti dati per quanti parametri? Le leggi di scala di Chinchilla suggerivano circa 20 token per parametro. Qui sono circa <strong>80 per parametro totale</strong> e oltre 2.800 per parametro attivo: il modello «studia» molto più a lungo di quanto chiederebbe Chinchilla. DeepSeek non spiega la scelta; un motivo diffuso è che un modello più piccolo ma addestrato più a lungo costa meno ogni volta che lo si usa.</p>
<p>Poi il <strong>contesto lungo</strong>: le posizioni dei layer compressi vengono stirate con YaRN (fattore 16 su 65.536) fino a 1.048.576 token. È come allargare le tacche di un righello per misurare distanze più grandi. Due dettagli che si leggono in giro (attention sparsa addestrata a 64K, estensione a 1M dopo 34T token) <em>non sono verificati</em>.</p>`,
      },
      {
        id: "full-post",
        title: "Post-training: SFT, RL, distillazione",
        anchor: "macro:post",
        play: "full-train",
        text: `<p>Il modello base sa continuare il testo, non ancora rispondere. Il post-training segue lo schema <strong>SFT → RL → distillazione on-policy (OPD)</strong>: prima esempi da imitare, poi tanti «specialisti» allenati con il rinforzo (in V4 con GRPO, l'algoritmo di DeepSeek-R1), infine un solo modello che impara da tutti generando le proprie risposte.</p>
<p>Come in una scuola di cucina: prima si copiano le ricette (SFT), poi ogni allievo si specializza assaggiando e correggendo i propri piatti (RL), infine un capo cuoco impara dagli specialisti cucinando lui stesso (OPD). La novità dichiarata di V4.1 non è l'algoritmo ma i dati: task e ambienti agentici sintetizzati automaticamente.</p>
<p>Che cosa cambia nei pesi? Uno studio del 2025 (Mukherjee e colleghi) mostra che l'RL aggiorna solo il <strong>5-30% dei parametri</strong>, e che le sottoreti trovate con seed e algoritmi diversi si sovrappongono molto più del caso.</p>`,
      },
      {
        id: "full-weights",
        title: "I pesi: 510 GB, 96.085 tensori",
        anchor: "macro:weights",
        code: "names",
        charts: "full-pesi",
        text: `<p>Il risultato del training è un file: il <strong>checkpoint</strong>, 510,3 GB in formato <em>safetensors</em> (8 byte con la lunghezza dell'intestazione, un JSON con nome, forma e posizione di ogni tensore, poi i numeri grezzi: niente codice eseguibile). Dentro ci sono 96.085 tensori; il numero di file (48 shard) <em>non è verificato</em>.</p>
<p>I numeri non hanno tutti la stessa precisione. Gli esperti, il 98,5% dei parametri del backbone, stanno in <strong>MXFP4</strong>: 4 bit per numero più una scala condivisa ogni 32, cioè 4,25 bit. Attention ed esperti condivisi in FP8 a blocchi 32 × 32; embedding, head e router in BF16; i pochi parametri delicati (mHC, sink, bias del router) in FP32.</p>
<p>È come un archivio dove i faldoni consultati di rado (i 384 esperti di ogni layer) sono compressi al massimo, mentre i documenti usati a ogni pagina restano in bella copia. Il pannello «Pesi» elenca i nomi veri dei tensori.</p>`,
      },
    ],
  },
  {
    id: "full-sec-arch",
    title: "3. L'architettura, blocco per blocco",
    subtab: "full",
    children: [
      {
        id: "full-tokens",
        title: "Tokenizer, embedding e 4 corsie",
        anchor: "blocks:tokenizer",
        play: "full-token",
        text: `<p>Il testo entra nel <strong>tokenizer BPE a byte</strong>: 128.000 voci più i token speciali, 129.280 in tutto. Le cifre si raggruppano a 3 a 3; ci sono token per il ragionamento (<code>&lt;think&gt;</code> … <code>&lt;/think&gt;</code>), per le chiamate a strumenti (<code>｜DSML｜</code>) e per le immagini.</p>
<p>Ogni token diventa una riga della tabella di <strong>embedding</strong>: 129.280 righe da 5.120 numeri (0,66 miliardi di parametri). Le immagini arrivano da un'altra strada: un ViT da 32 blocchi le trasforma in al massimo 1.024 token da 5.120, messi in fila accanto al testo.</p>
<p>Poi una novità rispetto al nostro GPT: il residuo si allarga a <strong>4 flussi</strong> (mHC). Pensa a un'autostrada a 4 corsie invece di una strada sola: ogni layer decide come spostare il traffico tra le corsie.</p>`,
      },
      {
        id: "full-stack",
        title: "40 layer, pochi tipi: il blocco «× N»",
        anchor: "blocks:layers",
        text: `<p>V4.1-Flash ha 40 layer, ma solo 6 tipi diversi. Invece di disegnarne 40, la scena li raggruppa: <strong>2 × SWA</strong> e <strong>3 × [Full + 5 Reuse]</strong> nell'encoder; <strong>Full ← H20</strong>, <strong>3 × Reuse</strong> e <strong>4 × [Reindex + 3 Reuse]</strong> nel decoder. Le lastre fantasma dietro un blocco e la parentesi «× N» dicono quante volte si ripete.</p>
<p>È il «blocco n-Transformer» che maschera le ripetizioni: clic sull'etichetta «× N» (oppure «Espandi le ripetizioni» nel pannello Viste) per srotolare il gruppo nei layer veri, ognuno con il suo numero. Ogni blocco mostra i suoi parametri: circa 13,8 miliardi per layer, di cui 13,6 negli esperti.</p>
<p>Il raggruppamento non è scritto a mano: una funzione cerca la descrizione più corta dello schema dei layer, e funziona allo stesso modo per Kimi (23 × [3 KDA + 1 Gated MLA] + 1) e per gpt-oss (18 × [finestra + densa]).</p>`,
      },
      {
        id: "full-ced",
        title: "Encoder e decoder: il prompt a metà prezzo",
        anchor: "blocks:segment-encoder",
        play: "full-ced",
        code: "ced",
        text: `<p>I 40 layer sono divisi in due metà: 20 di <strong>encoder causale</strong> e 20 di <strong>decoder</strong>. Causale vuol dire che ogni posizione vede solo il passato, come nel nostro GPT.</p>
<p>Il trucco: il KV globale di tutto il decoder è una proiezione dell'<strong>uscita dell'encoder</strong> (H20), non dello stato di ogni layer. Così, quando il modello legge il prompt (prefill), basta eseguire l'encoder: <strong>8 miliardi di parametri attivi</strong> per token invece di 16. Il decoder si riesegue solo sugli ultimi 128 token, per ricostruire le sue finestre.</p>
<p>Come in una riunione: la segretaria riassume i documenti una volta sola (encoder) e i dirigenti leggono il riassunto quando devono parlare (decoder). L'idea viene da YOCO (2024); il paper non pubblica un'ablazione del solo encoder-decoder.</p>`,
      },
      {
        id: "full-kv",
        title: "KV cache compressa: 890 byte per token",
        anchor: "blocks:cache-kv",
        play: "full-kv",
        code: "kvmath",
        charts: "full-kv",
        text: `<p>Nel nostro GPT ogni layer salva key e value di ogni token passato. V4.1 è costruito attorno al problema opposto: <strong>890 byte per token</strong> di KV globale, sempre nella memoria veloce della GPU.</p>
<p>Come? Una sola KV head da 512 valori, con <strong>K = V</strong>; nei layer dell'encoder 2 token diventano 1 voce; ogni voce è in <strong>FP4</strong> (512 × 4 bit + 32 scale = 288 byte); solo 4 layer (2, 8, 14 e 20) scrivono il KV globale, gli altri layer compressi lo <strong>riusano</strong> (i layer 0 e 1 hanno solo la finestra). Conto: 2,5 voci per token × (288 + 68 byte di chiave dell'indexer) = 890 byte.</p>
<p>DeepSeek-V3 (MLA) ne usava 70.272, circa 79 volte di più; DeepSeek LLM 67B (GQA) 389.120, <strong>437 volte</strong> di più. Invece di fotocopiare ogni pagina per ogni lettore, si tiene un riassunto compresso che tutti consultano.</p>`,
      },
    ],
  },
  {
    id: "full-sec-layer",
    title: "4. Dentro un layer",
    subtab: "full",
    children: [
      {
        id: "full-attention",
        title: "Attention: latente condiviso, finestra, indexer",
        anchor: "detail:core",
        layer: 2,
        code: "csa2",
        text: `<p>Dentro un layer «Full» (il 2, per esempio) l'attention ha quattro strade. La <strong>query</strong> passa da una strettoia (5.120 → 1.280) e torna larga: 64 head da 512. Il <strong>KV locale</strong> è un solo vettore da 512 (448 di contenuto + 64 con RoPE), condiviso da tutte le head: è MQA, il caso estremo di GQA.</p>
<p>Ogni token guarda i 128 vicini più recenti (la <strong>finestra</strong>, con un «sink» che permette a una head di non guardare nulla) e, più lontano, solo le <strong>512 voci</strong> che il lightning indexer giudica più utili: un punteggio veloce in FP4 con 32 piccole head. Dal layer 20 la scelta è a due livelli: prima 2.048 blocchi candidati da 8 voci, poi le 512 migliori.</p>
<p>Per scrivere una frase rileggi le ultime righe (finestra) e consulti l'indice analitico per saltare alle pagine giuste (indexer), invece di rileggere tutto il libro.</p>`,
      },
      {
        id: "full-moe",
        title: "MoE: 6 esperti su 384, più 1 condiviso",
        anchor: "detail:experts",
        layer: 2,
        play: "full-router",
        code: "router",
        charts: "full-pesi",
        text: `<p>Ogni layer ha <strong>384 esperti</strong> (piccole MLP SwiGLU da 5.120 → 2.304 → 5.120) e 1 esperto condiviso sempre attivo. Il router dà un punteggio a ogni esperto (sqrtsoftplus), aggiunge un piccolo bias di bilanciamento solo per scegliere, prende i <strong>6 migliori</strong>, normalizza i loro pesi e li moltiplica per 1,5.</p>
<p>Qui sta il 98,5% dei parametri del backbone: 543,6 miliardi su 551,9. Ma per ogni token lavorano solo 6 + 1 esperti per layer: per questo «16B attivi» su 552B. È un ospedale enorme in cui ogni paziente vede solo i 6 specialisti giusti più il medico di base.</p>
<p>Le scelte vere degli esperti per ogni token non sono pubblicate: nell'animazione sono una <em>simulazione didattica</em>.</p>`,
      },
      {
        id: "full-mhc",
        title: "mHC ed Engram: 4 corsie e un dizionario",
        anchor: "detail:residual",
        layer: 14,
        play: "full-engram",
        code: "engram",
        text: `<p>Le 4 corsie del residuo (<strong>mHC</strong>) si mescolano prima e dopo attention e MoE con matrici «doppiamente stocastiche» (ogni riga e ogni colonna sommano a 1), ottenute con 20 iterazioni di Sinkhorn: ogni corsia riceve una media pesata delle corsie, quindi la miscela da sola non amplifica il segnale lungo i 40 layer.</p>
<p>Nei layer 1 e 14 c'è un ramo in più: <strong>Engram</strong>, una memoria di n-gram. Gli ultimi 2, 3 e 4 token (normalizzati) passano per 8 funzioni di hash ciascuno e indicano 24 righe, una per ognuna delle 24 tabelle del layer (un numero primo di righe vicino a 16 milioni ciascuna, circa 384 milioni in tutto); le righe lette vengono proiettate e sommate alle corsie con un gate. Le due tabelle valgono 196,6 miliardi di parametri, ma l'indirizzo è deterministico: possono stare nella RAM del computer invece che nella GPU.</p>
<p>È un dizionario dei modi di dire: per completare «in bocca al…» non serve ragionare, basta cercare.</p>`,
      },
    ],
  },
  {
    id: "full-sec-out",
    title: "5. Uscita, inferenza e capacità",
    subtab: "full",
    children: [
      {
        id: "full-dspark",
        title: "DSpark, deployment e inferenza",
        anchor: "blocks:dspark",
        play: "full-dspark",
        code: "dspark",
        text: `<p>Alla fine del decoder le feature dei layer 37, 38 e 39 alimentano <strong>DSpark</strong>: 3 piccoli blocchi che propongono <strong>5 token</strong> per giro, con una testa che stima la probabilità che vengano accettati. Il modello li verifica tutti in un passaggio: è la decodifica speculativa della vista d'insieme, costruita dentro il modello.</p>
<p>In produzione il lavoro si divide: visione, prefill e decode girano su gruppi di GPU separati (nel paper; non nella release open di DeepInfer), e gli esperti sono distribuiti tra le GPU con copie ridondanti per bilanciare il carico (EPLB). vLLM indica almeno 614 GB di memoria; su 8 H100 bisogna spostare le tabelle Engram nella RAM dell'host.</p>
<p>Un token di prompt costa circa 15,8 GFLOP, un token generato circa 32,3.</p>`,
      },
      {
        id: "full-emerge",
        title: "Come emergono le capacità",
        anchor: "macro:capabilities",
        play: "full-emerge",
        charts: "full-scala",
        text: `<p>Nessun pezzo del modello «contiene» il ragionamento o il coding: le capacità emergono dall'addestramento. Cosa sappiamo, con le sue cautele. La loss scende in modo prevedibile con parametri, dati e calcolo (<strong>leggi di scala</strong>). Alcune abilità sembrano accendersi di colpo con la scala, ma Schaeffer e colleghi (2023) mostrano che spesso è un effetto della <strong>metrica</strong>: con una misura continua la crescita è dolce.</p>
<p>Presto nel training compaiono le <strong>induction head</strong>, che copiano schemi dal contesto; su piccoli problemi algoritmici la generalizzazione può arrivare molto dopo l'overfitting (<strong>grokking</strong>); l'RL rifinisce una sottorete del 5-30% dei parametri. Sono risultati su modelli studiati a fondo: nessuna delle fonti consultate misura questi fenomeni su V4.1.</p>
<p>Come un bambino che impara a leggere: non c'è un giorno preciso in cui «sa leggere», ma se lo misuri solo con «legge un libro intero senza errori» sembra succedere all'improvviso.</p>`,
      },
    ],
  },
];

/** Schede di sintesi sotto la Panoramica (titolo e testo brevi: si leggono da lontano). */
export const MACRO_NOTES = {
  capabilities: {
    title: "Come emergono le capacità",
    text: "Leggi di scala prevedibili · salti «emergenti» che spesso dipendono dalla metrica (Schaeffer 2023) · induction head presto nel training · grokking su problemi piccoli · l'RL rifinisce il 5-30% dei pesi.",
  },
  weightsShare: {
    title: "Dove sono i pesi",
    text: "Quasi tutti negli esperti: {{share}} del backbone. Il resto (attention, embedding, head, router, residuo) è una piccola parte.",
  },
};

/** Etichetta "da verificare" con il valore dubbio (per le pillole nella scena). */
export const UNVERIFIED_LABEL = "da verificare";
