/**
 * Glossario della lezione: i termini chiave con una definizione semplice, un'analogia concreta
 * e il passo del Percorso in cui vederli (link "Guarda nel tab …" = app.goto(tab, go)).
 *
 * Voce: { id, term, aliases?: string[], definition, analogy, tab, go?: {kind: "step", id}, related?: [id] }
 * - `definition` e `analogy` sono testo semplice (niente HTML): vengono mostrati con textContent;
 * - `tab` + `go` devono esistere (lo controlla tests/search.test.mjs leggendo i content.js dei tab);
 * - i numeri citati sono quelli del GPT del corso (gpt/config.py, data/model.json) o delle fonti
 *   dei tab (pannello "Fonti").
 */

const step = (id) => ({ kind: "step", id });

export const GLOSSARY = [
  // --- dal testo ai vettori
  {
    id: "token",
    term: "Token",
    aliases: ["gettone", "vocabolario"],
    definition:
      "Il pezzetto di testo che il modello legge e scrive, uno alla volta. Nel nostro GPT un token è un carattere (65 in tutto); nei modelli grandi è di solito un pezzo di parola. Il modello non vede le lettere: vede solo il numero (id) di ogni token.",
    analogy: "Come i mattoncini di un set da costruzione: ogni testo viene smontato in pezzi standard, ognuno con il suo codice.",
    tab: "transformer",
    go: step("data.tokenizer"),
    related: ["tokenizer", "bpe", "embedding"],
  },
  {
    id: "tokenizer",
    term: "Tokenizer",
    aliases: ["tokenizzazione", "encode", "decode"],
    definition:
      "Il programma che trasforma il testo in una lista di id di token (encode) e gli id di nuovo in testo (decode). Il nostro è a caratteri: una tabella carattere → numero costruita sul dataset.",
    analogy: "Come una rubrica a doppia entrata: dal nome trovi il numero e dal numero trovi il nome.",
    tab: "transformer",
    go: step("data.tokenizer"),
    related: ["token", "bpe"],
  },
  {
    id: "bpe",
    term: "BPE (Byte-Pair Encoding)",
    aliases: ["byte pair encoding", "sottoparole", "subword"],
    definition:
      "Il tokenizer dei modelli moderni: parte dai singoli byte e fonde, una dopo l'altra, le coppie che compaiono più spesso nel testo di addestramento, creando token sempre più lunghi. Le parole comuni diventano un token solo, quelle rare restano spezzate.",
    analogy: "Come le abbreviazioni che nascono a forza di scrivere sempre le stesse parole: alla fine \"per favore\" diventa \"pf\".",
    tab: "modern-llm",
    go: step("arch.bpe"),
    related: ["token", "tokenizer"],
  },
  {
    id: "embedding",
    term: "Embedding",
    aliases: ["token embedding", "vettore", "rappresentazione"],
    definition:
      "Un vettore di numeri appreso che rappresenta un token. È una tabella: la riga i è il vettore del token i (nel nostro GPT 65 righe da 64 numeri). Durante il training i token che si usano in modo simile finiscono con vettori vicini.",
    analogy: "Come le coordinate su una mappa dei significati: parole simili abitano nello stesso quartiere.",
    tab: "transformer",
    go: step("embedding.token"),
    related: ["positional-embedding", "embedding-model"],
  },
  {
    id: "positional-embedding",
    term: "Positional embedding",
    aliases: ["position embedding", "posizione", "embedding di posizione"],
    definition:
      "Un secondo vettore appreso, uno per ogni posizione nel contesto (0, 1, 2…), che si somma all'embedding del token. Serve perché l'attention da sola non sa in che ordine arrivano i token.",
    analogy: "Come il numero del posto sul biglietto del treno: la stessa persona seduta in un posto diverso è un passeggero diverso.",
    tab: "transformer",
    go: step("embedding.position"),
    related: ["embedding", "rope", "contesto"],
  },
  {
    id: "contesto",
    term: "Contesto (block_size)",
    aliases: ["context window", "finestra di contesto", "block size", "lunghezza del contesto"],
    definition:
      "Il numero massimo di token che il modello guarda insieme per prevedere il successivo. Nel nostro GPT è block_size = 32 caratteri; i modelli moderni arrivano a centinaia di migliaia di token.",
    analogy: "Come la memoria a breve termine durante una telefonata: ricordi le ultime frasi, non tutta la conversazione dell'anno scorso.",
    tab: "transformer",
    go: step("data.batch"),
    related: ["positional-embedding", "kv-cache"],
  },
  // --- dentro il blocco transformer
  {
    id: "self-attention",
    term: "Self-attention",
    aliases: ["attention", "attenzione", "autoattenzione"],
    definition:
      "Il meccanismo con cui ogni token raccoglie informazioni dagli altri token del contesto: calcola quanto ognuno è rilevante per lui e ne fa una media pesata. I pesi non sono fissi: dipendono da cosa c'è scritto.",
    analogy: "Come in una riunione: ognuno ascolta di più chi ha detto cose utili per il proprio lavoro.",
    tab: "transformer",
    go: step("head.qkv"),
    related: ["qkv", "head", "maschera-causale", "softmax"],
  },
  {
    id: "qkv",
    term: "Query, Key, Value",
    aliases: ["q k v", "qkv", "query", "key", "value", "chiave", "valore"],
    definition:
      "Tre proiezioni lineari di ogni token. La query dice cosa cerca, la key con che cosa si presenta, il value quale informazione passa se viene scelto. Il punteggio tra due token è il prodotto query · key, diviso per la radice della dimensione della head.",
    analogy: "Come in biblioteca: la domanda che hai in testa (query), l'etichetta sul dorso del libro (key) e il contenuto del libro (value).",
    tab: "transformer",
    go: step("head.qkv"),
    related: ["self-attention", "head", "kv-cache"],
  },
  {
    id: "head",
    term: "Head (testa di attention)",
    aliases: ["attention head", "multi-head attention", "testa", "multi head"],
    definition:
      "Una singola attention con le sue matrici query, key e value. Più head lavorano in parallelo (4 nel nostro GPT), ognuna libera di notare relazioni diverse; i loro risultati vengono concatenati e rimescolati da una proiezione lineare.",
    analogy: "Come più evidenziatori di colori diversi sullo stesso testo: uno per i nomi, uno per i verbi, uno per la punteggiatura.",
    tab: "transformer",
    go: step("mha.parallel"),
    related: ["self-attention", "qkv"],
  },
  {
    id: "maschera-causale",
    term: "Maschera causale",
    aliases: ["causal mask", "tril", "maschera"],
    definition:
      "Una matrice triangolare che impedisce a ogni token di guardare i token che vengono dopo di lui: i loro punteggi diventano −∞ e, dopo la softmax, pesano 0. Così il modello impara davvero a prevedere il futuro invece di copiarlo.",
    analogy: "Come leggere un giallo coprendo con un foglio le righe successive: puoi usare solo quello che hai già letto.",
    tab: "transformer",
    go: step("head.mask"),
    related: ["self-attention", "softmax"],
  },
  {
    id: "softmax",
    term: "Softmax",
    aliases: ["probabilita", "normalizzazione esponenziale"],
    definition:
      "La funzione che trasforma numeri qualsiasi in probabilità: tutte positive e con somma 1. Usa l'esponenziale, quindi esalta i valori più grandi. Si usa nell'attention (per i pesi) e all'uscita (per il prossimo token).",
    analogy: "Come dividere una torta in base ai voti, ma con una regola che regala fette molto più grandi a chi ha preso più voti.",
    tab: "transformer",
    go: step("output.softmax"),
    related: ["logit", "temperatura", "maschera-causale"],
  },
  {
    id: "residual",
    term: "Connessione residua (residual stream)",
    aliases: ["residual", "residuo", "skip connection", "residual stream", "x + f(x)"],
    definition:
      "Ogni pezzo del blocco aggiunge il suo risultato al vettore che arriva (x = x + f(x)) invece di sostituirlo. Il vettore che scorre lungo tutta la rete si chiama residual stream. Aiuta il gradiente ad arrivare fino ai primi strati durante il training.",
    analogy: "Come un documento condiviso in cui ogni revisore aggiunge note a margine senza cancellare il testo originale.",
    tab: "transformer",
    go: step("block.residual"),
    related: ["layernorm", "backprop"],
  },
  {
    id: "layernorm",
    term: "LayerNorm",
    aliases: ["layer normalization", "normalizzazione"],
    definition:
      "Normalizza ogni vettore: sottrae la media dei suoi numeri e divide per la deviazione standard, poi applica una scala (γ) e uno spostamento (β) appresi. Tiene i valori in un intervallo stabile e rende il training più regolare.",
    analogy: "Come regolare il volume di ogni traccia prima di mandarla al mixer: nessuna troppo forte, nessuna troppo debole.",
    tab: "transformer",
    go: step("block.layernorm"),
    related: ["rmsnorm", "residual"],
  },
  {
    id: "rmsnorm",
    term: "RMSNorm",
    aliases: ["root mean square norm"],
    definition:
      "Una versione più semplice di LayerNorm usata da molti modelli moderni (per esempio Llama): divide il vettore per la radice della media dei quadrati dei suoi numeri, senza sottrarre la media e senza lo spostamento β. Costa meno e funziona altrettanto bene.",
    analogy: "Come regolare solo il volume di una traccia senza toccarne il tono.",
    tab: "modern-llm",
    go: step("arch.rmsnorm"),
    related: ["layernorm"],
  },
  {
    id: "feed-forward",
    term: "Feed-forward (MLP)",
    aliases: ["mlp", "ffn", "ffwd", "rete feed forward", "relu"],
    definition:
      "Una piccola rete a due strati lineari con una non linearità in mezzo (ReLU nel nostro GPT), applicata a ogni token separatamente. Lo strato centrale è 4 volte più largo del vettore (256 numeri contro 64).",
    analogy: "Dopo la riunione (l'attention) ognuno torna alla sua scrivania e rielabora da solo gli appunti presi.",
    tab: "transformer",
    go: step("ffwd.mlp"),
    related: ["self-attention", "moe"],
  },
  // --- uscita, generazione e training
  {
    id: "logit",
    term: "Logit",
    aliases: ["logits", "lm_head", "punteggi"],
    definition:
      "Il punteggio grezzo che il modello dà a ogni token del vocabolario come possibile prossimo token. Può essere qualsiasi numero, anche negativo: la softmax lo trasforma in probabilità. Li calcola l'ultimo strato lineare (lm_head).",
    analogy: "Come i voti grezzi di una giuria prima che vengano convertiti in percentuali.",
    tab: "transformer",
    go: step("output.lmhead"),
    related: ["softmax", "temperatura"],
  },
  {
    id: "temperatura",
    term: "Temperatura",
    aliases: ["temperature", "campionamento", "sampling"],
    definition:
      "Un numero che divide i logit prima della softmax. Sotto 1 le scelte diventano più sicure e ripetitive, sopra 1 più varie e caotiche; vicino a 0 il modello sceglie quasi sempre il token più probabile.",
    analogy: "Come la manopola dell'audacia di un cuoco: bassa segue la ricetta, alta improvvisa.",
    tab: "transformer",
    go: step("output.softmax"),
    related: ["softmax", "top-k", "logit"],
  },
  {
    id: "top-k",
    term: "Top-k (campionamento)",
    aliases: ["top k", "topk", "top-p", "nucleus sampling"],
    definition:
      "Durante la generazione si tengono solo i k token più probabili, si rinormalizzano le loro probabilità e si estrae tra quelli: così si evitano scelte assurde. Il generate del nostro GPT usa solo la temperatura; top-k è un'aggiunta comune nei modelli reali. Attenzione: nel tab RAG topK indica invece quanti chunk recuperare dal database.",
    analogy: "Come scegliere il vincitore solo tra i k finalisti, invece che tra tutti gli iscritti.",
    tab: "transformer",
    go: step("generation.loop"),
    related: ["temperatura", "softmax"],
  },
  {
    id: "loss",
    term: "Loss (cross-entropy)",
    aliases: ["cross entropy", "funzione di perdita", "perdita", "errore"],
    definition:
      "Il numero che misura quanto sbaglia il modello: è −log della probabilità che ha dato al token giusto, in media su tutti i token del batch. 0 vorrebbe dire certezza sempre corretta; tirando a indovinare tra 65 caratteri si parte da ln 65 ≈ 4,17.",
    analogy: "Come un voto di sorpresa: più eri sicuro di una risposta sbagliata, più punti perdi.",
    tab: "transformer",
    go: step("training.loss"),
    related: ["backprop", "overfitting", "adamw"],
  },
  {
    id: "backprop",
    term: "Backpropagation",
    aliases: ["retropropagazione", "gradiente", "backward", "loss.backward"],
    definition:
      "L'algoritmo che calcola, per ogni peso, quanto ha contribuito alla loss (il gradiente), applicando la regola della catena dall'uscita verso l'ingresso. In PyTorch è loss.backward(); poi l'ottimizzatore usa i gradienti per correggere i pesi.",
    analogy: "Come risalire una catena di montaggio dal prodotto difettoso per capire quanto ha sbagliato ogni stazione.",
    tab: "transformer",
    go: step("training.loop"),
    related: ["loss", "adamw"],
  },
  {
    id: "adamw",
    term: "AdamW",
    aliases: ["adam", "ottimizzatore", "optimizer", "learning rate", "weight decay"],
    definition:
      "L'ottimizzatore usato nel training del nostro GPT. Aggiorna ogni peso con una media mobile dei gradienti (il momento) divisa per una stima della loro grandezza, quindi ogni peso ha il suo passo; il weight decay, che rimpicciolisce un po' i pesi, è applicato a parte (\"decoupled\").",
    analogy: "Come un escursionista che scende a valle ricordando la direzione presa finora e accorciando il passo dove il terreno è irregolare.",
    tab: "transformer",
    go: step("training.loop"),
    related: ["backprop", "loss"],
  },
  {
    id: "overfitting",
    term: "Overfitting",
    aliases: ["sovradattamento", "validation", "validazione", "generalizzare"],
    definition:
      "Quando il modello impara a memoria i dati di training invece delle regole generali: la loss sul training continua a scendere, quella sui dati di validation (mai visti) smette di scendere o risale. Per accorgersene si tiene da parte una porzione del dataset.",
    analogy: "Come lo studente che memorizza le soluzioni degli esercizi svolti ma non sa risolverne uno nuovo.",
    tab: "transformer",
    go: step("training.loss"),
    related: ["loss"],
  },
  {
    id: "parametri",
    term: "Parametri (pesi)",
    aliases: ["pesi", "weights", "parameters"],
    definition:
      "I numeri che il modello impara durante il training: le tabelle di embedding, le matrici dei layer lineari, γ e β delle LayerNorm. Il nostro GPT ne ha 209.729; la maschera causale e il dropout non hanno pesi.",
    analogy: "Come le migliaia di manopole di un mixer gigante, regolate una per una dal training.",
    tab: "transformer",
    go: step("weights.all"),
    related: ["embedding", "adamw"],
  },
  // --- RAG e applicazioni
  {
    id: "chunk",
    term: "Chunk",
    aliases: ["chunking", "frammento", "spezzare il testo"],
    definition:
      "Un pezzo di documento indicizzato da solo: qualche centinaio di token (nella L17 il TokenTextSplitter di Spring AI ne usa 500). La ricerca restituisce chunk, non documenti interi, così nel prompt entra solo la parte utile.",
    analogy: "Come le schede di un archivio: non porti al tavolo tutto lo scaffale, solo le schede che servono.",
    tab: "rag",
    go: step("index.chunk"),
    related: ["embedding-model", "database-vettoriale", "rag"],
  },
  {
    id: "embedding-model",
    term: "Embedding model",
    aliases: ["modello di embedding", "sentence embedding", "vettorizzare"],
    definition:
      "Un modello che trasforma un testo intero (una frase, un chunk) in un solo vettore di lunghezza fissa, in modo che testi con significato simile abbiano vettori vicini. È diverso dagli embedding interni del GPT, che sono uno per token.",
    analogy: "Come un bibliotecario che riassume ogni testo con le coordinate del suo argomento sulla mappa della biblioteca.",
    tab: "rag",
    go: step("index.embed"),
    related: ["similarita-coseno", "database-vettoriale", "embedding"],
  },
  {
    id: "similarita-coseno",
    term: "Similarità del coseno",
    aliases: ["cosine similarity", "coseno", "similarita"],
    definition:
      "Misura quanto due vettori puntano nella stessa direzione: è il coseno dell'angolo tra loro, da −1 (opposti) a 1 (stessa direzione). Non dipende dalla lunghezza dei vettori, solo dalla direzione.",
    analogy: "Come due frecce sulla mappa che indicano la stessa città, anche se una è più lunga dell'altra.",
    tab: "rag",
    go: step("index.embed"),
    related: ["embedding-model", "database-vettoriale"],
  },
  {
    id: "database-vettoriale",
    term: "Database vettoriale",
    aliases: ["vector store", "vector database", "pgvector", "vector_store"],
    definition:
      "Un database che salva vettori insieme al loro testo e ai metadati e trova in fretta quelli più vicini a un vettore dato. Nel corso è PostgreSQL con l'estensione pgvector (tabella vector_store) usato da Spring AI.",
    analogy: "Come una biblioteca ordinata per argomento invece che per titolo: chiedi \"libri come questo\" e ti portano allo scaffale giusto.",
    tab: "rag",
    go: step("intro.db"),
    related: ["hnsw", "similarita-coseno", "rag"],
  },
  {
    id: "hnsw",
    term: "HNSW",
    aliases: ["hierarchical navigable small world", "indice", "ann", "vicini piu vicini"],
    definition:
      "Hierarchical Navigable Small World: un indice a grafo su più livelli per trovare i vettori più vicini senza confrontarli tutti. Si parte dal livello più alto, con pochi nodi e collegamenti lunghi, e si scende di livello in livello avvicinandosi alla risposta. Il risultato è approssimato ma molto veloce.",
    analogy: "Come un viaggio in auto: prima l'autostrada, poi la strada statale, infine le vie del quartiere.",
    tab: "rag",
    go: step("hnsw.simple"),
    related: ["database-vettoriale", "similarita-coseno"],
  },
  {
    id: "rag",
    term: "RAG (Retrieval-Augmented Generation)",
    aliases: ["retrieval augmented generation", "generazione aumentata dal recupero", "retrieval"],
    definition:
      "Prima si recuperano dal database vettoriale i chunk più pertinenti alla domanda, poi li si mette nel prompt: così l'LLM risponde basandosi su quei documenti, anche se non li ha mai visti durante il training.",
    analogy: "Come un esame a libro aperto: prima cerchi le pagine giuste, poi rispondi usando quelle.",
    tab: "rag",
    go: step("intro.flusso"),
    related: ["chunk", "database-vettoriale", "system-prompt"],
  },
  {
    id: "system-prompt",
    term: "System prompt",
    aliases: ["prompt di sistema", "istruzioni di sistema", "ruolo system"],
    definition:
      "Le istruzioni inviate al modello con il ruolo \"system\" prima dei messaggi dell'utente: fissano il comportamento, le regole e il formato delle risposte. Nel RAG dice per esempio di rispondere solo in base ai documenti forniti.",
    analogy: "Come il briefing che il responsabile dà al commesso prima di aprire il negozio.",
    tab: "rag",
    go: step("search.prompt"),
    related: ["rag", "prompt-injection", "tool-calling"],
  },
  {
    id: "prompt-injection",
    term: "Prompt injection",
    aliases: ["indirect prompt injection", "iniezione di prompt", "sicurezza"],
    definition:
      "Testo scritto apposta per far seguire al modello istruzioni diverse da quelle dello sviluppatore. È \"indiretta\" quando si nasconde dentro un documento recuperato: per questo i documenti vanno trattati come dati non fidati.",
    analogy: "Come un biglietto infilato in una pratica che dice all'impiegato \"ignora il tuo capo e fai questo\".",
    tab: "rag",
    go: step("security.injection"),
    related: ["system-prompt", "rag"],
  },
  {
    id: "tool-calling",
    term: "Function calling (tool use)",
    aliases: ["tool calling", "tool use", "strumenti", "@tool", "agenti", "agente"],
    definition:
      "Il modello non esegue codice: quando serve, risponde con una richiesta strutturata (nome dello strumento e argomenti in JSON). L'applicazione esegue lo strumento e gli rimanda il risultato, che il modello usa per continuare. In Spring AI gli strumenti si dichiarano con @Tool.",
    analogy: "Come il centralinista che compila il modulo per l'ufficio giusto e poi ti riferisce la risposta.",
    tab: "modern-llm",
    go: step("reas.agent"),
    related: ["system-prompt", "chain-of-thought"],
  },
  // --- immagini e video
  {
    id: "diffusione",
    term: "Modello di diffusione",
    aliases: ["diffusion", "diffusione", "denoising", "ddpm", "ddim"],
    definition:
      "Un modello generativo addestrato a togliere rumore da un'immagine. Per generare si parte da rumore puro e lo si toglie un po' alla volta, in molti passi, finché resta un'immagine coerente con il prompt. A differenza del GPT, produce tutta l'immagine insieme e la raffina.",
    analogy: "Come una fotografia che emerge piano piano dalla nebbia.",
    tab: "diffusion",
    go: step("idea.cosa"),
    related: ["rumore", "unet", "cfg"],
  },
  {
    id: "rumore",
    term: "Rumore (processo diretto)",
    aliases: ["noise", "rumore gaussiano", "forward process", "schedule", "epsilon"],
    definition:
      "Nel training si sporca un'immagine con rumore gaussiano secondo una schedule (1000 passi in DDPM). Una formula chiusa permette di saltare subito al passo t; la rete impara a indovinare il rumore che è stato aggiunto.",
    analogy: "Come la neve sullo schermo di un vecchio televisore che copre l'immagine sempre di più.",
    tab: "diffusion",
    go: step("forward.formula"),
    related: ["diffusione"],
  },
  {
    id: "vae",
    term: "VAE e spazio latente",
    aliases: ["latente", "latent", "autoencoder", "variational autoencoder", "spazio latente"],
    definition:
      "Un autoencoder variazionale: l'encoder comprime l'immagine in un latente molto più piccolo (in Stable Diffusion 1.x un'immagine 512×512 diventa 64×64×4), la diffusione lavora lì e il decoder riporta il risultato in pixel. Lavorare nel latente costa molto meno.",
    analogy: "Come fare il bozzetto in miniatura e stamparlo in grande solo alla fine.",
    tab: "diffusion",
    go: step("arch.latent"),
    related: ["unet", "diffusione"],
  },
  {
    id: "unet",
    term: "U-Net",
    aliases: ["unet", "u net", "skip connection", "denoiser"],
    definition:
      "La rete che toglie il rumore in Stable Diffusion: a forma di U, prima riduce la risoluzione (per capire la composizione), poi la aumenta (per i dettagli); le skip connection ricopiano i dettagli dalla discesa alla risalita. Il testo entra con la cross-attention.",
    analogy: "Come un pittore che guarda il quadro da lontano per la composizione e da vicino per le pennellate.",
    tab: "diffusion",
    go: step("arch.unet"),
    related: ["dit", "vae", "cfg"],
  },
  {
    id: "dit",
    term: "DiT (Diffusion Transformer)",
    aliases: ["diffusion transformer", "patch", "mmdit"],
    definition:
      "Un denoiser in cui la U-Net è sostituita da un transformer: il latente viene tagliato in patch e ogni patch diventa un token. È lo stesso transformer della lezione, usato per togliere rumore invece che per prevedere il prossimo carattere.",
    analogy: "Come tagliare una foto nelle tessere di un puzzle e trattare ogni tessera come una parola.",
    tab: "diffusion",
    go: step("dit.patch"),
    related: ["unet", "self-attention"],
  },
  {
    id: "cfg",
    term: "CFG (Classifier-Free Guidance)",
    aliases: ["classifier free guidance", "guidance", "guidance scale"],
    definition:
      "A ogni passo la rete fa due previsioni del rumore, con il prompt e senza, e si spinge nella direzione della differenza: ε = ε_senza + w · (ε_con − ε_senza). Con w alto l'immagine segue di più il prompt ma è meno varia.",
    analogy: "Come chiedere la strada \"verso il mare\" e \"a caso\", e poi esagerare la differenza tra le due risposte.",
    tab: "diffusion",
    go: step("reverse.cfg"),
    related: ["diffusione", "unet"],
  },
  // --- audio
  {
    id: "spettrogramma",
    term: "Spettrogramma",
    aliases: ["stft", "fft", "frequenze", "spectrogram"],
    definition:
      "Un'immagine del suono: il tempo in orizzontale, la frequenza in verticale, il colore indica l'energia. Si ottiene con la STFT: si taglia il segnale in finestre brevi e su ognuna si calcola la FFT.",
    analogy: "Come il rullo di carta di una pianola: dove c'è un buco, in quel momento suona quella nota.",
    tab: "audio",
    go: step("freq.fft"),
    related: ["mel", "codec-neurale"],
  },
  {
    id: "mel",
    term: "Scala Mel (log-Mel)",
    aliases: ["mel", "log mel", "mel spectrogram", "whisper"],
    definition:
      "Una scala delle frequenze che imita l'orecchio umano: fitta sui suoni bassi e sempre più rada su quelli acuti. Lo spettrogramma log-Mel (Mel + logaritmo dell'energia) è, per esempio, l'ingresso di Whisper.",
    analogy: "Come i tasti del pianoforte: salendo di un'ottava la frequenza raddoppia, ma noi sentiamo sempre lo stesso salto.",
    tab: "audio",
    go: step("freq.mel"),
    related: ["spettrogramma"],
  },
  {
    id: "codec-neurale",
    term: "Codec neurale",
    aliases: ["encodec", "soundstream", "codec", "token audio"],
    definition:
      "Una rete encoder-decoder che comprime l'audio in una sequenza di token discreti e poi lo ricostruisce. EnCodec a 24 kHz produce 75 frame al secondo, ognuno descritto da alcuni token (uno per codebook). Sono questi token che un transformer impara a generare.",
    analogy: "Come un MP3, ma imparato dai dati e fatto di \"parole\" che un modello linguistico può scrivere.",
    tab: "audio",
    go: step("codec.encoder"),
    related: ["rvq", "spettrogramma"],
  },
  {
    id: "rvq",
    term: "RVQ (quantizzazione vettoriale residua)",
    aliases: ["residual vector quantization", "codebook", "quantizzazione"],
    definition:
      "Ogni vettore dell'encoder viene sostituito dal più vicino di un elenco (codebook); un secondo codebook approssima l'errore rimasto, un terzo l'errore del secondo, e così via. Ogni frame diventa così una pila di token: più codebook, più qualità e più bit.",
    analogy: "Come descrivere un colore per gradi: \"blu\", poi \"un po' più chiaro\", poi \"con un filo di verde\".",
    tab: "audio",
    go: step("codec.rvq"),
    related: ["codec-neurale"],
  },
  // --- LLM moderni
  {
    id: "rope",
    term: "RoPE (Rotary Position Embedding)",
    aliases: ["rotary", "rotary position embedding", "rotazione"],
    definition:
      "Invece di sommare un vettore di posizione, ruota query e key di un angolo che cresce con la posizione (a coppie di numeri, con velocità diverse). Così il prodotto query · key dipende dalla distanza tra i due token, non dalla loro posizione assoluta.",
    analogy: "Come le lancette di un orologio: per sapere quanto tempo è passato conta l'angolo tra le due, non l'ora esatta.",
    tab: "modern-llm",
    go: step("arch.rope"),
    related: ["positional-embedding", "qkv"],
  },
  {
    id: "kv-cache",
    term: "KV cache",
    aliases: ["cache", "key value cache", "cache kv"],
    definition:
      "Durante la generazione si salvano key e value dei token già elaborati: a ogni nuovo token si calcolano solo i suoi, invece di rifare tutto il contesto. È molto più veloce, ma la memoria occupata cresce con la lunghezza del contesto.",
    analogy: "Come tenere gli appunti di una conversazione invece di riascoltarla da capo prima di ogni nuova frase.",
    tab: "modern-llm",
    go: step("inf.kv"),
    related: ["qkv", "contesto"],
  },
  {
    id: "moe",
    term: "MoE (Mixture of Experts)",
    aliases: ["mixture of experts", "esperti", "router", "experts"],
    definition:
      "Il feed-forward viene sostituito da tanti \"esperti\" e un router sceglie, per ogni token, solo pochi di loro (in Mixtral 2 su 8). Il modello ha moltissimi parametri in totale, ma per ogni token ne usa solo una parte, quindi costa meno calcolo.",
    analogy: "Come un pronto soccorso con molti specialisti: il triage manda ogni paziente solo da due di loro.",
    tab: "modern-llm",
    go: step("arch.moe"),
    related: ["feed-forward"],
  },
  {
    id: "rlhf",
    term: "RLHF e DPO",
    aliases: ["reinforcement learning from human feedback", "dpo", "preferenze", "allineamento", "post-training", "sft"],
    definition:
      "Fasi di addestramento dopo il pre-training. Delle persone confrontano coppie di risposte; con RLHF si addestra un modello di ricompensa su queste preferenze e poi si ottimizza l'LLM per massimizzarla; DPO usa direttamente le coppie, senza modello di ricompensa.",
    analogy: "Come una giuria di assaggiatori che dice quale di due piatti è migliore, e il cuoco che corregge le ricette di conseguenza.",
    tab: "modern-llm",
    go: step("train.align"),
    related: ["chain-of-thought"],
  },
  {
    id: "chain-of-thought",
    term: "Ragionamento (chain of thought)",
    aliases: ["catena di pensiero", "reasoning", "cot", "ragionare", "pensiero"],
    definition:
      "Il modello scrive passaggi intermedi prima della risposta finale, e ogni passaggio diventa contesto per il successivo. I modelli \"di ragionamento\" sono addestrati a farlo, anche con reinforcement learning su problemi con risposta verificabile, e spendono più calcolo al momento della risposta.",
    analogy: "Come fare i calcoli in colonna sul foglio invece che tutto a mente.",
    tab: "modern-llm",
    go: step("reas.cot"),
    related: ["rlhf", "tool-calling"],
  },
];

/** Voce del glossario per id (undefined se non esiste). */
export function glossaryTerm(id) {
  return GLOSSARY.find((t) => t.id === id);
}

/** Voci per l'indice della ricerca globale (kind "term"). */
export function glossarySearchIndex() {
  return GLOSSARY.map((t) => ({
    id: t.id,
    kind: "term",
    title: t.term,
    text: `${t.definition} ${t.analogy}`,
    keywords: t.aliases ?? [],
  }));
}
