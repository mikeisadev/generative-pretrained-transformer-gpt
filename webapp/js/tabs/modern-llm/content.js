/**
 * Testi del tab "LLM moderni": percorso didattico (Menu del nucleo), schede dei modelli aperti
 * e fonti. Ogni numero qui ha una fonte nel pannello "Fonti"; ciò che non è pubblico è dichiarato.
 *
 * I numeri del nostro GPT sono segnaposto {{chiave}} riempiti da index.js con i dati di
 * data/model.json (vedi ourModel.js).
 *
 * Campi extra dei sottopunti del percorso (letti da index.js):
 *   focus   id della regione da inquadrare        code    chiave dello snippet (snippets.js)
 *   action  id dell'azione animata suggerita      charts  id della sezione di grafici da aprire
 *
 * Il tab ha due sottotab con due Percorsi: OVERVIEW_PATH (la "Vista d'insieme", qui sotto) e
 * FULL_PATH (l'"Architettura completa", in full/content.js). PATH li unisce: è l'elenco di TUTTI i
 * passi del tab, con id univoci, quello che leggono la ricerca globale e i test.
 */
import { FULL_PATH } from "./full/content.js";

export const OVERVIEW_PATH = [
  {
    id: "idea",
    title: "1. Dal nostro GPT ai modelli di oggi",
    children: [
      {
        id: "idea.ponte",
        title: "Stessa ricetta, cucina industriale",
        focus: "compare",
        code: "ours:block",
        action: "compare",
        text: `<p>Il GPT di questa lezione ha {{params}} parametri, legge {{block}} caratteri e conosce {{vocab}} simboli. Un modello aperto come Llama 3 8B ha circa 8 miliardi di parametri, legge migliaia di token e ne conosce 128.256.</p>
<p>Eppure l'ossatura è la stessa: una pila di blocchi decoder, ognuno con <em>attention</em> (i token si parlano) e <em>MLP</em> (ogni token rielabora da solo), con normalizzazione prima e connessioni residue dopo. È come passare dalla cucina di casa a quella di un ristorante: le ricette di base non cambiano, cambiano gli attrezzi, le quantità e l'organizzazione.</p>
<p>Nella scena a sinistra c'è il nostro blocco, a destra uno moderno: ogni riga con la freccia è un pezzo sostituito. Avvia l'animazione per vederli cambiare uno alla volta.</p>`,
      },
      {
        id: "idea.chiusi",
        title: "Cosa sappiamo e cosa no",
        focus: "compare",
        text: `<p>Le architetture di GPT-4/5, Claude e Gemini <strong>non sono pubbliche</strong>: le aziende pubblicano risultati e indicazioni generali, non numero di layer, head o esperti. Quello che vedi in questo tab viene da modelli <strong>con pesi e codice aperti</strong> (Llama, Mistral e Mixtral, Qwen, DeepSeek), documentati da paper, configurazioni e codice ufficiali.</p>
<p>È ragionevole pensare che i modelli chiusi usino idee simili (sono tutte pubblicate e diffuse), ma non è confermato: quando un numero non ha una fonte verificabile, qui non lo usiamo.</p>
<p>Le animazioni con pesi o testi inventati sono marcate come <em>simulazione didattica</em>.</p>`,
      },
    ],
  },
  {
    id: "arch",
    title: "2. Architettura: i pezzi nuovi",
    children: [
      {
        id: "arch.bpe",
        title: "Tokenizer BPE a byte",
        focus: "bpe",
        code: "minbpe.train",
        action: "bpe",
        text: `<p>Il nostro GPT legge un carattere alla volta. I modelli moderni usano un <strong>BPE a byte</strong> (Byte Pair Encoding): il testo diventa la sequenza dei suoi byte UTF-8 (256 simboli possibili) e poi, durante l'addestramento del tokenizer, si fonde ripetutamente la coppia di simboli vicini più frequente in un simbolo nuovo.</p>
<p>Analogia: chi scrive spesso «per favore» finisce per abbreviarlo in «pf». Le parole e i pezzi di parola comuni diventano un solo token, quelli rari restano spezzati. Partendo dai byte nessun testo è mai «sconosciuto»: anche un'emoji o un carattere cinese si scrivono con i byte di partenza.</p>
<p>Il vocabolario finale è 256 byte + una voce per ogni fusione, più pochi token speciali (come il segnale di «fine testo»): GPT-2 ha 50.257 token (256 + 50.000 fusioni + 1 speciale), Llama 3 ne ha 128.256 (128.000 normali + 256 riservati a usi speciali). Meno token per frase significa più testo nello stesso contesto.</p>`,
      },
      {
        id: "arch.rmsnorm",
        title: "RMSNorm al posto di LayerNorm",
        focus: "compare",
        code: "llama.rmsnorm",
        text: `<p>LayerNorm toglie la media e divide per la deviazione standard, poi applica scala e spostamento appresi. <strong>RMSNorm</strong> (Zhang e Sennrich, 2019) fa solo la seconda metà: divide per la radice della media dei quadrati e moltiplica per una scala appresa. Niente media, niente spostamento.</p>
<p>Analogia: per regolare il volume di una canzone basta abbassare tutto della stessa quantità; non serve anche spostarla di tono. Gli autori riportano qualità simile a LayerNorm con un tempo di calcolo ridotto tra il 7% e il 64% a seconda del modello. Llama, Mistral, Qwen e DeepSeek la usano.</p>`,
        math: [
          { tex: "\\text{LayerNorm}(x) = \\frac{x-\\mu}{\\sqrt{\\sigma^2+\\varepsilon}}\\odot\\gamma+\\beta", note: "μ e σ² calcolati sulle d componenti del vettore." },
          { tex: "\\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\tfrac{1}{d}\\sum_{i=1}^{d} x_i^2+\\varepsilon}}\\odot g", note: "Nel codice di Llama: x * rsqrt(mean(x²) + eps) * weight." },
        ],
      },
      {
        id: "arch.rope",
        title: "RoPE: la posizione è una rotazione",
        focus: "rope",
        code: "llama.rope",
        action: "rope",
        text: `<p>Nel nostro GPT la posizione è una tabella appresa ({{block}} righe) sommata all'embedding: oltre la riga {{block}} il modello non sa cosa fare. <strong>RoPE</strong> (Su et al., 2021) invece prende query e key, le divide in coppie di numeri e <em>ruota</em> ogni coppia di un angolo proporzionale alla posizione.</p>
<p>Analogia: due lancette che girano alla stessa velocità. Se una parte alla posizione 3 e l'altra alla 7, l'angolo tra loro dipende solo dalla differenza 7 − 3, non da dove sono sull'orologio. Così il prodotto q·k dell'attention «sente» la <strong>distanza</strong> tra i token. Coppie diverse girano a velocità diverse: le veloci distinguono vicini, le lente le distanze grandi.</p>
<p>Llama 3 usa una base θ = 500.000 (il paper originale 10.000): rotazioni più lente aiutano i contesti lunghi.</p>`,
        math: [
          { tex: "\\begin{pmatrix}x'_{2i}\\\\x'_{2i+1}\\end{pmatrix}=\\begin{pmatrix}\\cos m\\theta_i & -\\sin m\\theta_i\\\\ \\sin m\\theta_i & \\cos m\\theta_i\\end{pmatrix}\\begin{pmatrix}x_{2i}\\\\x_{2i+1}\\end{pmatrix},\\qquad \\theta_i = b^{-2i/d}", note: "m = posizione del token, b = base (10.000 in RoFormer, 500.000 in Llama 3)." },
          { tex: "\\langle R_m\\,q,\\; R_n\\,k\\rangle = \\langle q,\\; R_{n-m}\\,k\\rangle", note: "Il punteggio di attention dipende solo dalla distanza n − m." },
        ],
      },
      {
        id: "arch.gqa",
        title: "MQA e GQA: K e V condivisi",
        focus: "gqa",
        code: "llama.repeat_kv",
        action: "gqa",
        charts: "kv",
        text: `<p>Nel nostro GPT ogni head ha le sue matrici K e V (multi-head attention, MHA). Durante la generazione K e V di tutti i token passati vanno tenuti in memoria (la KV cache, prossimo capitolo) e riletti a ogni token: con contesti lunghi spesso è proprio questa lettura dalla memoria, più dei calcoli, a limitare la velocità.</p>
<p><strong>Multi-query attention</strong> (Shazeer, 2019): tutte le head di query condividono UNA sola coppia K/V. <strong>Grouped-query attention</strong> (Ainslie et al., 2023) è la via di mezzo: le query sono divise in gruppi e ogni gruppo condivide una coppia K/V. Analogia: invece di una fotocopia degli appunti per ogni studente, una per banco.</p>
<p>Llama 3 8B ha 32 head di query e 8 head K/V: la cache è 4 volte più piccola che con MHA. Il paper GQA mostra anche che un modello MHA si può convertire con circa il 5% del calcolo del pre-training.</p>`,
        math: [{ tex: "\\text{KV cache} = 2 \\cdot L \\cdot n_{kv} \\cdot d_{head} \\cdot \\text{byte} \\cdot T", note: "Llama 3 8B in bf16: 2 · 32 · 8 · 128 · 2 = 131.072 byte = 128 KiB per token; a 8.192 token fa 1 GiB." }],
      },
      {
        id: "arch.swiglu",
        title: "SwiGLU al posto di ReLU",
        focus: "compare",
        code: "llama.ffn",
        text: `<p>La nostra MLP è Linear → ReLU → Linear. <strong>SwiGLU</strong> (Shazeer, 2020) usa tre matrici: una produce il segnale, un'altra un «cancello» che passa per la funzione SiLU (detta anche Swish), e i due si moltiplicano elemento per elemento prima della proiezione finale.</p>
<p>Analogia: invece di un interruttore acceso/spento (ReLU), un variatore di luce comandato dal contenuto stesso. Nel paper le varianti GLU danno perplessità migliori a parità di costo. Per non aumentare i parametri, lo strato nascosto si riduce a circa 2/3 di 4·d: nel codice di Llama si vede <code>int(2 * hidden_dim / 3)</code>. Llama 3 8B poi lo moltiplica per 1,3 (<code>ffn_dim_multiplier</code>) e lo arrotonda a un multiplo di 1024: 4·4096 → 10.922 → 14.198 → <strong>14.336</strong>.</p>`,
        math: [{ tex: "\\text{SwiGLU}(x) = W_2\\big(\\text{SiLU}(W_1x)\\odot W_3x\\big),\\qquad \\text{SiLU}(z)=z\\,\\sigma(z)", note: "Nel codice: w2(F.silu(w1(x)) * w3(x))." }],
      },
      {
        id: "arch.moe",
        title: "Mixture of Experts",
        focus: "moe",
        code: "mistral.moe",
        action: "moe",
        charts: "moe",
        text: `<p>In un layer <strong>Mixture of Experts</strong> la MLP è sostituita da tanti «esperti» (MLP indipendenti) e da un <strong>router</strong>, una piccola matrice che per ogni token dà un punteggio a ogni esperto. Si usano solo i migliori k e le loro uscite si sommano pesate.</p>
<p>Analogia: un pronto soccorso con un infermiere al triage che manda ogni paziente da due specialisti, non da tutti. L'ospedale ha tanti medici (parametri), ma ogni paziente ne impegna pochi (calcolo). Switch Transformer (2021) usa un solo esperto per token; Mixtral 8x7B ne sceglie 2 su 8 in ogni layer (47B parametri in totale, 13B attivi per token).</p>
<p>DeepSeek-V3 spinge l'idea: 256 esperti instradati più 1 esperto condiviso sempre attivo, 8 attivi per token, 671B parametri totali e 37B attivi. Il rischio è che il router mandi tutti dagli stessi esperti: servono strategie di bilanciamento (DeepSeek-V3 ne usa una senza loss ausiliaria).</p>`,
        math: [
          { tex: "g = \\operatorname{softmax}\\big(\\operatorname{TopK}(W_g\\,x,\\;k)\\big),\\qquad y = \\sum_{i \\in \\operatorname{TopK}} g_i\\,E_i(x)", note: "Come nel codice di Mixtral (k = 2 su 8): softmax solo sui k punteggi scelti. DeepSeek-V3 usa punteggi sigmoid (config ufficiale) e aggiunge l'esperto condiviso." },
        ],
      },
    ],
  },
  {
    id: "inf",
    title: "3. Inferenza efficiente",
    children: [
      {
        id: "inf.kv",
        title: "KV cache",
        focus: "kv",
        code: "llama.cache",
        action: "kv",
        charts: "kv",
        text: `<p>Il nostro <code>generate</code> rimette nel modello tutto il contesto a ogni nuovo carattere: ricalcola K e V di token già visti, che non cambiano mai (con la maschera causale un token passato non vede quelli futuri).</p>
<p>La <strong>KV cache</strong> li salva: a ogni passo si calcolano K e V solo del token nuovo e si rileggono gli altri. Analogia: in un'interrogazione non rileggi il libro dall'inizio a ogni domanda, consulti i tuoi appunti. Il prezzo è la memoria, che cresce con la lunghezza del contesto: per questo contano GQA, MQA e la MLA di DeepSeek (che salva un vettore latente compresso invece di K e V completi).</p>`,
        math: [{ tex: "\\underbrace{\\textstyle\\sum_{t=1}^{n} t = \\tfrac{n(n+1)}{2}}_{\\text{senza cache}}\\quad\\text{contro}\\quad \\underbrace{n}_{\\text{con cache}}", note: "Colonne K/V calcolate per generare n token (per layer)." }],
      },
      {
        id: "inf.flash",
        title: "FlashAttention, finestre e contesti lunghi",
        focus: "flash",
        action: "flash",
        text: `<p><strong>FlashAttention</strong> (Dao et al., 2022) calcola l'attention <em>esatta</em>, ma a blocchi: carica pezzi di Q, K, V nella memoria veloce sul chip (SRAM), calcola, accumula, e non scrive mai l'intera matrice T × T nella memoria grande e lenta (HBM). Analogia: cucinare sul piano di lavoro prendendo dal frigo solo gli ingredienti del passo corrente. Il paper riporta, per esempio, un addestramento di GPT-2 (contesto 1K) 3 volte più veloce rispetto alle implementazioni di riferimento.</p>
<p>L'<strong>attention a finestra scorrevole</strong> di Mistral 7B limita ogni layer agli ultimi W = 4096 token; impilando i layer l'informazione può comunque viaggiare più lontano. Oggi diversi modelli aperti dichiarano contesti fino a 128K token (Llama 3.1, DeepSeek-V3).</p>`,
        math: [{ tex: "\\text{accessi HBM: } \\Theta\\!\\left(\\tfrac{N^2 d^2}{M}\\right)\\ \\text{(FlashAttention)}\\quad\\text{contro}\\quad \\Theta(Nd + N^2)\\ \\text{(standard)}", note: "N = lunghezza, d = dimensione della head, M = dimensione della SRAM (dal paper)." }],
      },
      {
        id: "inf.spec",
        title: "Decodifica speculativa",
        focus: "spec",
        code: "sim:speculativeExpectedTokens",
        action: "spec",
        charts: "inferenza",
        text: `<p>Generare è lento perché ogni token richiede un passaggio completo del modello grande. Con la <strong>decodifica speculativa</strong> (Leviathan et al., 2023) un modello piccolo e veloce propone γ token; il modello grande li valuta <em>tutti insieme</em> in un passaggio, accetta il prefisso giusto e aggiunge un token suo.</p>
<p>Analogia: uno stagista scrive una bozza, il capo la legge una volta e corregge dal primo errore in poi. Con la regola di accettazione del paper la distribuzione dei token è identica a quella del modello grande da solo; su T5-XXL gli autori misurano 2-3× di velocità.</p>`,
        math: [{ tex: "\\mathbb{E}[\\text{token per passaggio}] = \\frac{1-\\alpha^{\\gamma+1}}{1-\\alpha}", note: "α = probabilità che una proposta sia accettata, γ = token proposti." }],
      },
    ],
  },
  {
    id: "train",
    title: "4. Addestramento",
    children: [
      {
        id: "train.scaling",
        title: "Leggi di scala (Chinchilla)",
        focus: "scaling",
        code: "sim:chinchillaOptimal",
        action: "scaling",
        charts: "scala",
        text: `<p>Le <strong>leggi di scala</strong> dicono come scende la loss quando crescono parametri (N), dati (D) e calcolo (C ≈ 6·N·D FLOP). Hoffmann et al. (2022) hanno addestrato centinaia di modelli e adattato una formula: la loss è una costante irriducibile più un termine che cala con N e uno che cala con D.</p>
<p>La conseguenza: con un budget fissato conviene far crescere N e D <em>insieme</em> (circa in proporzione). Chinchilla, 70B parametri e 1,4 trilioni di token, con lo stesso calcolo di Gopher (280B) lo supera su quasi tutti i test. Analogia: uno studente brillante con pochi libri impara meno di uno normale con una biblioteca.</p>
<p>Attenzione ai numeri esatti: il paper usa tre metodi. La formula qui sotto (il terzo, un fit parametrico) suggerisce modelli più piccoli degli altri due, che per il budget di Gopher indicano circa 63-67B parametri: da lì viene la scelta di 70B. Una replica (Besiroglu et al., 2024) ottiene coefficienti diversi; qui usiamo quelli pubblicati da Hoffmann et al. e li prendiamo come tendenza, non come ricetta.</p>`,
        math: [
          { tex: "L(N,D) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}},\\quad E=1{,}69,\\ A=406{,}4,\\ B=410{,}7,\\ \\alpha=0{,}34,\\ \\beta=0{,}28" },
          { tex: "N_{opt} \\propto C^{\\beta/(\\alpha+\\beta)} \\approx C^{0{,}45},\\qquad D_{opt} \\propto C^{\\alpha/(\\alpha+\\beta)} \\approx C^{0{,}55}", note: "Minimizzando L con il vincolo C = 6·N·D." },
        ],
      },
      {
        id: "train.pipeline",
        title: "Pre-training, SFT, preferenze",
        focus: "training",
        action: "pipeline",
        text: `<p><strong>Pre-training</strong>: lo stesso obiettivo del nostro GPT (prevedere il prossimo token) su trilioni di token (Llama 3: oltre 15 trilioni; DeepSeek-V3: 14,8). Il risultato è un modello base: continua il testo, ma non «risponde».</p>
<p><strong>SFT</strong> (fine-tuning supervisionato): si continua l'addestramento su dialoghi esempio scritti da persone, così il modello impara il formato domanda-risposta. Poi le <strong>preferenze</strong>: si confrontano coppie di risposte e si spinge il modello verso quelle giudicate migliori.</p>
<p>Analogia: prima si legge una biblioteca intera, poi si fa un tirocinio guardando esempi, infine un tutor dice «questa risposta è meglio di quella».</p>`,
      },
      {
        id: "train.align",
        title: "RLHF, DPO, Constitutional AI",
        focus: "training",
        code: "dpo.loss",
        action: "pipeline",
        text: `<p><strong>RLHF</strong> (InstructGPT, Ouyang et al., 2022): persone ordinano le risposte, si addestra un <em>reward model</em> che imita i loro giudizi e poi si ottimizza il modello con RL (PPO), senza allontanarlo troppo dall'originale. Nel paper le risposte di InstructGPT da 1,3B erano preferite a quelle di GPT-3 da 175B.</p>
<p><strong>DPO</strong> (Rafailov et al., 2023) ottiene lo stesso obiettivo senza reward model né RL: una loss di classificazione direttamente sulle coppie preferita/scartata. <strong>Constitutional AI</strong> (Bai et al., 2022, Anthropic): i giudizi vengono da un'AI guidata da una lista di principi (la «costituzione»): il modello critica e rivede le proprie risposte, poi si fa RL con feedback dell'AI (RLAIF).</p>`,
        math: [
          { tex: "\\max_{\\pi}\\ \\mathbb{E}\\,[\\,r(x,y)\\,] - \\beta\\,\\mathrm{KL}\\big(\\pi \\,\\|\\, \\pi_{ref}\\big)", note: "Obiettivo di RLHF: ricompensa alta, senza allontanarsi troppo dal modello di partenza." },
          { tex: "\\mathcal{L}_{DPO} = -\\log\\sigma\\!\\left(\\beta\\log\\frac{\\pi(y_w|x)}{\\pi_{ref}(y_w|x)} - \\beta\\log\\frac{\\pi(y_l|x)}{\\pi_{ref}(y_l|x)}\\right)", note: "y_w = risposta preferita, y_l = scartata." },
        ],
      },
    ],
  },
  {
    id: "reas",
    title: "5. Ragionamento e agenti",
    children: [
      {
        id: "reas.cot",
        title: "Catena di pensiero e calcolo all'inferenza",
        focus: "reasoning",
        action: "cot",
        charts: "inferenza",
        text: `<p>Wei et al. (2022) hanno mostrato che chiedere al modello di scrivere i <strong>passi intermedi</strong> (chain-of-thought) migliora molto i problemi di aritmetica e logica, soprattutto nei modelli grandi: con 8 esempi ragionati, un modello da 540B superava lo stato dell'arte su GSM8K.</p>
<p>Ogni token di ragionamento è calcolo in più: è il <strong>test-time compute</strong>. Si può spendere anche campionando più soluzioni e facendole scegliere a un verificatore o a un voto. Snell et al. (2024) mostrano che allocare bene questo calcolo può valere più che ingrandire il modello, almeno su problemi di difficoltà media. Analogia: in un compito in classe chi fa la brutta copia e ricontrolla sbaglia meno di chi scrive subito la risposta.</p>`,
        math: [{ tex: "P(\\text{almeno una giusta su }N) = 1-(1-p)^N", note: "Con N tentativi indipendenti e un verificatore perfetto." }],
      },
      {
        id: "reas.rl",
        title: "RL con ricompense verificabili (GRPO)",
        focus: "reasoning",
        code: "sim:grpoAdvantages",
        action: "grpo",
        text: `<p>Nei problemi con una risposta controllabile (matematica, codice con test) la ricompensa non deve venire da un giudice umano: la dà un programma. DeepSeek-R1 (2025) parte da DeepSeek-V3-Base; la variante R1-Zero è addestrata con RL <em>senza</em> SFT iniziale e sviluppa da sola catene di ragionamento lunghe, verifica e riflessione.</p>
<p>L'algoritmo è <strong>GRPO</strong> (introdotto con DeepSeekMath, 2024): per ogni domanda si campiona un gruppo di risposte, si dà a ognuna la sua ricompensa e si confronta con la media del gruppo. Non serve un secondo modello che stimi il valore (il critic di PPO). Analogia: non c'è un voto assoluto, conta come sei andato rispetto ai compagni della stessa verifica.</p>`,
        math: [{ tex: "\\hat{A}_i = \\frac{r_i - \\operatorname{mean}(r_1,\\dots,r_G)}{\\operatorname{std}(r_1,\\dots,r_G)}", note: "Le risposte sopra la media vengono rese più probabili, quelle sotto meno." }],
      },
      {
        id: "reas.agent",
        title: "Tool use e agenti",
        focus: "agent",
        action: "agent",
        text: `<p>Un modello di linguaggio sa solo produrre testo. Con il <strong>tool use</strong> (o <em>function calling</em>) gli si descrivono degli strumenti (nome, descrizione, parametri) e lui può rispondere con una chiamata strutturata invece che con una frase. Il <em>programma</em> esegue lo strumento e rimette il risultato nel contesto. Framework come Spring AI espongono lo stesso meccanismo come «tool».</p>
<p>Toolformer (Schick et al., 2023) ha mostrato che un modello può imparare da solo quando chiamare calcolatrice, ricerca o calendario; ReAct (Yao et al., 2022) alterna ragionamento e azioni. Un <strong>agente</strong> ripete il ciclo pensa → agisci → osserva finché il compito è finito.</p>`,
      },
      {
        id: "reas.coding",
        title: "Capacità di coding",
        focus: "coding",
        action: "coding",
        text: `<p>I modelli moderni programmano bene per tre motivi. <strong>Dati</strong>: molto codice nel pre-training. <strong>Fill-in-the-middle</strong> (Bavarian et al., 2022): una parte dei documenti viene spezzata in prefisso, centro e suffisso e riordinata, così il modello impara a riempire un buco nel mezzo di un file (quello che fa l'autocompletamento nell'editor) senza perdere la capacità di scrivere da sinistra a destra.</p>
<p><strong>Feedback di esecuzione</strong>: il codice si può eseguire, e i test dicono se è giusto. È una ricompensa verificabile, come nella matematica. Per misurare tutto questo su problemi veri c'è SWE-bench (Jimenez et al., 2023): 2.294 issue reali di 12 repository Python, da risolvere modificando il codice finché i test passano.</p>`,
      },
    ],
  },
  {
    id: "beyond",
    title: "6. Oltre il testo",
    children: [
      {
        id: "beyond.multi",
        title: "Multimodale: immagini come token",
        focus: "multi",
        action: "multi",
        text: `<p>Come fa un modello di testo a «vedere»? Nello schema di <strong>LLaVA</strong> (Liu et al., 2023) un encoder visivo già addestrato (CLIP ViT-L/14) trasforma l'immagine in vettori, uno per patch; un <strong>proiettore</strong> (una matrice in LLaVA, un piccolo MLP in LLaVA-1.5) li porta nello spazio degli embedding del modello. Il risultato sono «token visivi» messi in fila prima del testo.</p>
<p>Analogia: un interprete che traduce l'immagine nella lingua interna del modello. Da lì in poi è tutto attention, come per le parole.</p>`,
        math: [{ tex: "H_v = W \\cdot Z_v,\\qquad Z_v = g(X_v)", note: "g = encoder visivo, W = proiettore addestrabile (LLaVA)." }],
      },
      {
        id: "beyond.mesh",
        title: "Modellazione 3D",
        focus: "mesh",
        action: "mesh",
        text: `<p>Due strade documentate. La prima: il modello <strong>scrive codice</strong> per un programma 3D. Blender ha un'API Python (<code>bpy</code>) e progetti come MeshGen o MCP for Blender collegano un LLM a Blender, che esegue gli script.</p>
<p>La seconda: la <strong>mesh diventa una sequenza di token</strong>. MeshGPT (Siddiqui et al., 2023) addestra un transformer decoder su triangoli codificati; <strong>LLaMA-Mesh</strong> (Wang et al., 2024, NVIDIA e Tsinghua) scrive la mesh come testo OBJ: righe «v x y z» per i vertici e «f a b c» per i triangoli, con coordinate scalate in [0, 64] e arrotondate a interi, senza token nuovi. Così lo stesso modello parla e modella.</p>`,
      },
      {
        id: "beyond.emergent",
        title: "Capacità emergenti o miraggio?",
        focus: "scaling",
        action: "emergence",
        charts: "emergenza",
        text: `<p>Wei et al. (2022) chiamano <strong>emergente</strong> una capacità assente nei modelli piccoli e presente nei grandi, non prevedibile estrapolando. Schaeffer, Miranda e Koyejo (2023) obiettano: spesso il salto dipende dalla <strong>metrica</strong>. Se si misura «risposta esattamente giusta» su una risposta di molti token, un miglioramento graduale per token appare come un'improvvisa accensione; con una metrica continua la stessa crescita è dolce.</p>
<p>Analogia: un ponte di 10 assi che reggono ognuna con probabilità p. La probabilità di attraversarlo è p¹⁰: resta quasi zero e poi sale di colpo, anche se ogni asse migliora piano. Il grafico «Emergenza» lo mostra con una curva illustrativa.</p>`,
        math: [{ tex: "P(\\text{esatta su } L \\text{ token}) = p^{L}", note: "Metrica discreta: piccoli miglioramenti di p sembrano un salto." }],
      },
      {
        id: "beyond.interp",
        title: "Interpretabilità: aprire la scatola",
        focus: "interp",
        action: "interp",
        text: `<p>Le <strong>induction head</strong> (Olsson et al., 2022, Anthropic) sono head di attention che completano lo schema [A][B] … [A] → [B]: cercano la volta precedente in cui compariva il token attuale e copiano quello che lo seguiva. Si formano durante l'addestramento insieme a un salto della capacità di imparare dal contesto.</p>
<p>I singoli neuroni spesso mescolano più concetti. Con il <strong>dictionary learning</strong> (un autoencoder sparso addestrato sulle attivazioni) Anthropic ha trovato direzioni più leggibili, le «feature»: oltre 4.000 da un layer di 512 neuroni in un piccolo modello a un solo layer (2023), fino a 34 milioni in Claude 3 Sonnet (2024).</p>`,
        math: [{ tex: "f = \\operatorname{ReLU}(W_e x + b_e),\\quad \\hat{x} = W_d f + b_d,\\quad \\mathcal{L} = \\|x-\\hat{x}\\|^2 + \\lambda\\|f\\|_1", note: "Ricostruire x usando poche feature accese (il termine λ‖f‖₁ premia la sparsità)." }],
      },
    ],
  },
];

/** Tutti i passi del tab: vista d'insieme e architettura completa (vedi l'intestazione). */
export const PATH = [...OVERVIEW_PATH, ...FULL_PATH];

/** Schede dei modelli: solo numeri con fonte (il nostro GPT viene da data/model.json). */
export const MODEL_CARDS = [
  {
    name: "Llama 3 8B",
    who: "Meta · pesi aperti",
    rows: [["layer", "32"], ["dimensione", "4096"], ["head query / K-V", "32 / 8 (GQA)"], ["MLP", "SwiGLU, 14.336"], ["norm", "RMSNorm"], ["posizioni", "RoPE, θ = 500.000"], ["vocabolario", "128.256 (BPE)"], ["contesto", "8K (3.1: 128K)"], ["dati", "oltre 15T token"]],
  },
  {
    name: "Mistral 7B",
    who: "Mistral AI · Apache 2.0",
    rows: [["attention", "GQA + finestra scorrevole"], ["finestra", "W = 4096"], ["tokenizer", "BPE con byte fallback"]],
  },
  {
    name: "Mixtral 8x7B",
    who: "Mistral AI · Apache 2.0",
    rows: [["MLP", "MoE, 8 esperti"], ["esperti per token", "2 (in ogni layer)"], ["parametri", "47B totali, 13B attivi"], ["post-training", "SFT + DPO (Instruct)"]],
  },
  {
    name: "DeepSeek-V3",
    who: "DeepSeek · pesi aperti",
    rows: [["parametri", "671B totali, 37B attivi"], ["layer", "61 (3 densi)"], ["attention", "MLA, latente K/V 512 + RoPE 64"], ["MoE", "1 condiviso + 256 instradati, 8 attivi"], ["router", "punteggi sigmoid"], ["contesto", "128K"], ["dati", "14,8T token"], ["extra", "multi-token prediction"]],
  },
  {
    name: "DeepSeek-R1",
    who: "DeepSeek · pesi aperti",
    rows: [["base", "DeepSeek-V3-Base"], ["R1-Zero", "solo RL, senza SFT"], ["R1", "dati cold-start + 2 fasi RL + 2 SFT"], ["distillati", "6 modelli densi (Llama, Qwen)"]],
  },
  {
    name: "Qwen2 / Qwen2.5",
    who: "Alibaba · pesi aperti",
    rows: [["nel codice di transformers", "RMSNorm, RoPE, SwiGLU, GQA"], ["dettaglio", "bias nelle proiezioni Q, K, V"]],
  },
  {
    name: "GPT-4/5, Claude, Gemini",
    who: "modelli chiusi",
    rows: [["architettura", "non pubblicata"], ["cosa è ragionevole", "transformer decoder con idee simili"], ["cosa NON diciamo", "numeri da voci non ufficiali"]],
  },
];

export const SOURCES = [
  {
    title: "Architettura",
    items: [
      { title: "Neural Machine Translation of Rare Words with Subword Units (BPE)", who: "Sennrich et al. · 2015", url: "https://arxiv.org/abs/1508.07909", note: "BPE applicato alle sotto-parole." },
      { title: "karpathy/minbpe", who: "codice", url: "https://github.com/karpathy/minbpe", note: "BPE a byte minimale; stesso algoritmo dell'animazione del tab. GPT-2: 50.257 token = 256 byte + 50.000 fusioni + 1 speciale (tiktoken)." },
      { title: "Root Mean Square Layer Normalization", who: "Zhang, Sennrich · NeurIPS 2019", url: "https://arxiv.org/abs/1910.07467", note: "RMSNorm: niente ricentramento, tempo ridotto del 7-64%." },
      { title: "RoFormer: Enhanced Transformer with Rotary Position Embedding", who: "Su et al. · 2021", url: "https://arxiv.org/abs/2104.09864", note: "RoPE: rotazione di q e k, dipendenza dalla posizione relativa." },
      { title: "GLU Variants Improve Transformer", who: "Shazeer · 2020", url: "https://arxiv.org/abs/2002.05202", note: "SwiGLU, GEGLU, ReGLU nella MLP del transformer." },
      { title: "Fast Transformer Decoding: One Write-Head is All You Need", who: "Shazeer · 2019", url: "https://arxiv.org/abs/1911.02150", note: "Multi-query attention: K e V condivisi tra le head." },
      { title: "GQA: Training Generalized Multi-Query Transformer Models", who: "Ainslie et al. · EMNLP 2023", url: "https://arxiv.org/abs/2305.13245", note: "Grouped-query attention; conversione con circa il 5% del calcolo di pre-training." },
      { title: "Switch Transformers", who: "Fedus, Zoph, Shazeer · 2021", url: "https://arxiv.org/abs/2101.03961", note: "MoE con un solo esperto per token." },
      { title: "Mixtral of Experts", who: "Jiang et al. · 2024", url: "https://arxiv.org/abs/2401.04088", note: "2 esperti su 8 per token in ogni layer; 47B totali, 13B attivi." },
      { title: "transformers: modeling_qwen2.py", who: "Hugging Face, codice", url: "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py", note: "Qwen2: RMSNorm, rotary embedding, MLP gate/up/down con SiLU, num_key_value_heads (GQA), bias in q/k/v." },
      { title: "DeepSeek-V2 (MLA)", who: "DeepSeek-AI · 2024", url: "https://arxiv.org/abs/2405.04434", note: "Multi-head Latent Attention: compressione congiunta a basso rango di K e V." },
      { title: "DeepSeek-V3 Technical Report e config ufficiale", who: "DeepSeek-AI · 2024", url: "https://github.com/deepseek-ai/DeepSeek-V3", note: "671B/37B, MLA, DeepSeekMoE, bilanciamento senza loss ausiliaria, MTP, 14,8T token. Config: inference/configs/config_671B.json. Report: arXiv 2412.19437." },
    ],
  },
  {
    title: "Inferenza",
    items: [
      { title: "FlashAttention", who: "Dao et al. · NeurIPS 2022", url: "https://arxiv.org/abs/2205.14135", note: "Attention esatta a blocchi, consapevole dell'IO; 3× nell'addestramento di GPT-2 (contesto 1K)." },
      { title: "Mistral 7B", who: "Jiang et al. · 2023", url: "https://arxiv.org/abs/2310.06825", note: "GQA e sliding window attention (W = 4096), rolling buffer cache." },
      { title: "Fast Inference from Transformers via Speculative Decoding", who: "Leviathan, Kalman, Matias · ICML 2023", url: "https://arxiv.org/abs/2211.17192", note: "Stesse uscite del modello grande, 2-3× su T5-XXL; formula dei token attesi." },
      { title: "meta-llama/llama3: llama/model.py", who: "codice ufficiale", url: "https://github.com/meta-llama/llama3/blob/main/llama/model.py", note: "RMSNorm, RoPE, repeat_kv, KV cache, SwiGLU: il codice nel pannello." },
      { title: "meta-llama/llama-models: sku_list.py e model card", who: "configurazioni ufficiali", url: "https://github.com/meta-llama/llama-models", note: "Llama 3 8B: dim 4096, 32 layer, 32 head, 8 head K/V, vocab 128.256, rope θ 500.000, ffn_dim_multiplier 1,3, multiple_of 1024; 3.1 a 128K." },
      { title: "mistralai/mistral-inference: moe.py", who: "codice ufficiale", url: "https://github.com/mistralai/mistral-inference", note: "MoeLayer: topk sui logit del gate e softmax sui pesi scelti." },
    ],
  },
  {
    title: "Addestramento e allineamento",
    items: [
      { title: "Training Compute-Optimal Large Language Models (Chinchilla)", who: "Hoffmann et al. · 2022", url: "https://arxiv.org/abs/2203.15556", note: "Tre metodi; il fit parametrico L(N,D) = 1,69 + 406,4/N^0,34 + 410,7/D^0,28. Chinchilla 70B, 1,4T token." },
      { title: "Chinchilla Scaling: A replication attempt", who: "Besiroglu et al. · 2024", url: "https://arxiv.org/abs/2404.10102", note: "Ricostruisce i dati e ottiene coefficienti diversi: i numeri esatti sono discussi." },
      { title: "Training language models to follow instructions with human feedback (InstructGPT)", who: "Ouyang et al. · NeurIPS 2022", url: "https://arxiv.org/abs/2203.02155", note: "SFT + reward model + PPO; 1,3B preferito a GPT-3 175B." },
      { title: "Direct Preference Optimization", who: "Rafailov et al. · NeurIPS 2023", url: "https://arxiv.org/abs/2305.18290", note: "Loss di classificazione sulle preferenze, senza reward model né RL. Codice: eric-mitchell/direct-preference-optimization." },
      { title: "Constitutional AI: Harmlessness from AI Feedback", who: "Bai et al. · Anthropic 2022", url: "https://arxiv.org/abs/2212.08073", note: "Critica e revisione guidate da principi, poi RL da feedback dell'AI." },
      { title: "Llama 3 model card", who: "Meta", url: "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", note: "GQA in tutti i modelli, vocabolario 128K, oltre 15T token, SFT + RLHF." },
    ],
  },
  {
    title: "Ragionamento, agenti, coding",
    items: [
      { title: "Chain-of-Thought Prompting Elicits Reasoning", who: "Wei et al. · NeurIPS 2022", url: "https://arxiv.org/abs/2201.11903", note: "8 esempi ragionati, 540B, stato dell'arte su GSM8K." },
      { title: "Scaling LLM Test-Time Compute Optimally…", who: "Snell et al. · ICLR 2025", url: "https://arxiv.org/abs/2408.03314", note: "Calcolo all'inferenza allocato in base alla difficoltà." },
      { title: "DeepSeekMath (GRPO)", who: "Shao et al. · 2024", url: "https://arxiv.org/abs/2402.03300", note: "GRPO: niente critic, baseline dalla media del gruppo." },
      { title: "DeepSeek-R1", who: "DeepSeek-AI · 2025", url: "https://github.com/deepseek-ai/DeepSeek-R1", note: "R1-Zero solo RL; R1 con cold start; distillazione. Paper: arXiv 2501.12948." },
      { title: "TRL: GRPO Trainer", who: "Hugging Face", url: "https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md", note: "Formula del vantaggio normalizzato sul gruppo." },
      { title: "Toolformer", who: "Schick et al. · 2023", url: "https://arxiv.org/abs/2302.04761", note: "Il modello impara da solo quando e come chiamare API." },
      { title: "ReAct", who: "Yao et al. · ICLR 2023", url: "https://arxiv.org/abs/2210.03629", note: "Ragionamento e azioni alternati." },
      { title: "Efficient Training of Language Models to Fill in the Middle", who: "Bavarian et al. · 2022", url: "https://arxiv.org/abs/2207.14255", note: "FIM: spostare un pezzo dal centro alla fine non danneggia la generazione normale." },
      { title: "SWE-bench", who: "Jimenez et al. · ICLR 2024", url: "https://arxiv.org/abs/2310.06770", note: "2.294 problemi da issue reali di 12 repository Python. Repo: github.com/SWE-bench/SWE-bench." },
    ],
  },
  {
    title: "Oltre il testo e interpretabilità",
    items: [
      { title: "Emergent Abilities of Large Language Models", who: "Wei et al. · TMLR 2022", url: "https://arxiv.org/abs/2206.07682", note: "Definizione: assente nei piccoli, presente nei grandi." },
      { title: "Are Emergent Abilities of Large Language Models a Mirage?", who: "Schaeffer, Miranda, Koyejo · NeurIPS 2023", url: "https://arxiv.org/abs/2304.15004", note: "Le metriche discontinue creano salti apparenti." },
      { title: "Visual Instruction Tuning (LLaVA)", who: "Liu et al. · NeurIPS 2023", url: "https://arxiv.org/abs/2304.08485", note: "CLIP ViT-L/14 + proiezione + Vicuna. Codice: haotian-liu/LLaVA (MLP a 2 strati in 1.5)." },
      { title: "MeshGPT", who: "Siddiqui et al. · CVPR 2024", url: "https://arxiv.org/abs/2311.15475", note: "Mesh triangolari generate da un transformer decoder." },
      { title: "LLaMA-Mesh", who: "Wang et al. · 2024", url: "https://arxiv.org/abs/2411.09595", note: "Mesh come testo OBJ, coordinate intere in [0, 64]. Codice: nv-tlabs/LLaMA-Mesh." },
      { title: "MeshGen (Blender) e MCP for Blender", who: "Hugging Face; ahujasid", url: "https://github.com/huggingface/meshgen", note: "Add-on che collegano un LLM a Blender (anche ahujasid/blender-mcp)." },
      { title: "In-context Learning and Induction Heads", who: "Olsson et al. · Anthropic 2022", url: "https://arxiv.org/abs/2209.11895", note: "Head che completano [A][B] … [A] → [B]." },
      { title: "Towards Monosemanticity", who: "Anthropic · 2023", url: "https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning", note: "Dictionary learning: oltre 4.000 feature da un layer di 512 neuroni." },
      { title: "Scaling Monosemanticity", who: "Anthropic · 2024", url: "https://transformer-circuits.pub/2024/scaling-monosemanticity/", note: "Autoencoder sparsi fino a 34 milioni di feature su Claude 3 Sonnet." },
    ],
  },
];
