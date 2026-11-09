/**
 * Indice della ricerca globale per il tab Transformer (contratto in webapp/ARCHITECTURE.md,
 * "Ricerca globale, link diretti e glossario").
 *
 * Solo dati leggeri: questo modulo viene caricato anche se il tab non è mai stato aperto, quindi
 * importa soltanto content.js (testi del Percorso) e mai three.js, la scena o index.js.
 *
 *   step    i passi del Percorso (stessi id del Menu, aperti con menu.reveal)
 *   action  le azioni animate (stessi id di player.register)
 *   view    le viste (Architettura, Pesi, Pesi numerici e la nascita dei pesi nel training, Rete neurale e la sua
 *           disposizione orizzontale, Contesto), le rappresentazioni del nucleo (rete, cervello), i livelli dello
 *           zoom semantico (zoom.detail | zoom.blocks | zoom.macro) e i passi del percorso della vista Contesto
 *           (ctx.*, aperti nel loro menu dentro la vista)
 *
 * I titoli con un segnaposto ({{n_head}} …) hanno qui un titolo leggibile senza numeri: i numeri
 * veri li mostra il tab, letti da model.json.
 */
import { STEPS } from "./content.js";

/** Titoli leggibili per i passi il cui titolo contiene un segnaposto ({{…}}). */
const STEP_TITLES = {
  "mha.parallel": "Più head in parallelo",
  "block.stack": "Blocchi impilati",
  "weights.all": "Dove sono i pesi",
  // il primo passo della self-attention: chi cerca "attention" nella lezione sul GPT deve trovare questo
  "head.qkv": "Attention: Query, Key, Value",
};

/** Sinonimi e parole che uno studente potrebbe cercare (anche in inglese), per id del passo. */
const STEP_KEYWORDS = {
  "data.dataset": ["dataset", "Shakespeare", "testo", "corpus", "dati di training", "SHA-256", "prossimo carattere"],
  "data.tokenizer": ["tokenizer", "tokenizzatore", "encode", "decode", "token id", "vocabolario", "vocab_size", "caratteri", "stoi", "itos"],
  "data.batch": ["batch", "block_size", "contesto", "finestra", "input e target", "get_batch", "x y"],
  "bigram.model": ["bigram", "bigramma", "tabella", "baseline", "modello più semplice"],
  "embedding.token": ["embedding", "token embedding", "vettore", "lookup", "nn.Embedding", "tok_emb", "n_embd"],
  "embedding.position": ["positional embedding", "position embedding", "posizione", "ordine dei token", "pos_emb"],
  "embedding.sum": ["residual stream", "flusso residuo", "somma", "nastro trasportatore", "broadcasting"],
  "trick.loop": ["media", "cicli", "bag of words", "trucco matematico", "comunicazione tra token"],
  "trick.matmul": ["moltiplicazione di matrici", "matmul", "matrice triangolare", "tril", "media pesata"],
  "trick.softmax": ["softmax", "maschera", "meno infinito", "-inf", "masked_fill"],
  "head.qkv": ["attention", "query", "key", "value", "self-attention", "attenzione", "head", "q k v", "head_size"],
  "head.scores": ["punteggi", "affinità", "prodotto scalare", "dot product", "scaling", "radice di head_size", "scaled dot-product"],
  "head.mask": ["maschera causale", "causal mask", "futuro", "softmax", "pattern di attention", "tril"],
  "head.values": ["media pesata", "value", "weighted sum", "uscita della head"],
  "mha.parallel": ["multi-head attention", "più teste", "head in parallelo", "n_head", "concat", "concatenazione"],
  "mha.proj": ["proiezione", "projection", "proj", "rimescola", "nn.Linear"],
  "ffwd.mlp": ["feed-forward", "MLP", "ReLU", "rete a due strati", "ffwd", "neuroni nascosti"],
  "block.residual": ["connessione residua", "residual connection", "skip connection", "x = x +", "gradiente"],
  "block.layernorm": ["LayerNorm", "normalizzazione", "layer normalization", "media e varianza", "pre-norm", "gamma beta"],
  "block.stack": ["blocchi", "strati", "layer", "n_layer", "profondità", "nn.Sequential"],
  "output.lmhead": ["lm_head", "ln_f", "LayerNorm finale", "logits", "punteggi", "unembedding"],
  "output.softmax": ["softmax", "probabilità", "distribuzione", "probabilities", "prossimo token"],
  "training.loss": ["loss", "cross-entropy", "entropia incrociata", "errore", "ln 65", "perdita"],
  "training.loop": ["training", "addestramento", "backpropagation", "retropropagazione", "AdamW", "ottimizzatore", "gradiente", "learning rate"],
  "generation.loop": ["generazione", "generate", "campionamento", "sampling", "temperatura", "multinomial", "un token alla volta", "autoregressivo"],
  "weights.all": ["pesi", "parametri", "parameters", "weights", "quanti parametri", "heatmap", "conteggio"],
};

const steps = STEPS.flatMap((section) =>
  section.children.map((item) => ({
    id: item.id,
    kind: "step",
    title: STEP_TITLES[item.id] ?? item.title,
    text: item.text,
    keywords: [section.title.replace(/^\d+\.\s*/, ""), ...(STEP_KEYWORDS[item.id] ?? [])],
  })),
);

/** Azioni animate: stessi id e titoli di player.register (actions/, lod.js, views/, context/). */
const actions = [
  {
    id: "prompt",
    title: "Dal prompt all'output",
    text: "Il tuo prompt attraversa davvero il modello: tokenizzazione, embedding, attention di ogni blocco, logits, temperatura ed estrazione del prossimo carattere. Poi i token successivi, più veloci.",
    keywords: ["inferenza", "inference", "forward", "prompt", "genera testo", "temperatura", "estrazione", "sampling", "ruota delle probabilità", "logit lens"],
  },
  {
    id: "training-step",
    title: "Uno step di training al microscopio",
    text: "Batch, forward, loss, backward e AdamW con i numeri veri del training registrato.",
    keywords: ["training step", "backward", "backpropagation", "gradiente", "AdamW", "ottimizzatore", "loss", "cross-entropy", "addestramento"],
  },
  {
    id: "training-run",
    title: "Il training completo in un minuto",
    text: "La storia vera del training (5000 step): heatmap dei pesi che cambiano, pattern di attention che emergono, testo generato a ogni snapshot e i grafici della loss.",
    keywords: ["training completo", "curva della loss", "snapshot", "evoluzione dei pesi", "loss curve", "addestramento", "train val"],
  },
  {
    id: "lod-zoom",
    title: "Zoom semantico: dal dettaglio alla panoramica",
    text: "La camera si allontana: le matrici diventano rettangoli con una descrizione, poi un diagramma a blocchi.",
    keywords: ["zoom", "LOD", "livelli di dettaglio", "panoramica", "blocchi", "semantic zoom"],
  },
  {
    id: "weights-tour",
    title: "Dove sono i parametri: il giro dei pesi",
    text: "Visita ogni componente, conta i suoi pesi e verifica che il totale torni con quello di PyTorch.",
    keywords: ["pesi", "parametri", "conteggio", "parameters", "count", "tour", "vista Pesi"],
  },
  {
    id: "neurons-forward",
    title: "Un passaggio in avanti, neurone per neurone",
    text: "I neuroni si accendono strato dopo strato, dal carattere in ingresso alle probabilità del prossimo.",
    keywords: ["neuroni", "rete neurale", "forward pass", "attivazioni", "neural network", "strati"],
  },
  {
    id: "context-pipeline",
    title: "Dalla domanda al contesto dell'LLM",
    text: "Guardrail, embedding, ricerca su pgvector, assemblaggio del prompt, strumenti e streaming SSE: il contesto si compone pezzo per pezzo ed entra nella pila.",
    keywords: ["RAG", "contesto", "context window", "system prompt", "pgvector", "Spring AI", "tool calling", "function calling", "streaming"],
  },
].map((a) => ({ ...a, kind: "action" }));

/** Viste del tab (stessi id di views.register) e livelli dello zoom semantico. */
const views = [
  {
    id: "architettura",
    title: "Vista Architettura",
    text: "Il modello come lo descrive il codice: matrici, head, blocchi e flusso dei dati.",
    keywords: ["architettura", "architecture", "pila", "stack", "vista d'insieme", "modello 3D"],
  },
  {
    id: "pesi",
    title: "Vista Pesi",
    text: "Dove stanno i parametri appresi: ogni tensore con nome, forma e numero di pesi. Clic su una matrice per il suo istogramma.",
    keywords: ["pesi", "parametri", "tensori", "weights", "parameters", "istogramma", "forma", "shape"],
  },
  {
    id: "numeri",
    title: "Vista Pesi numerici",
    text: "I numeri veri dentro le matrici: avvicinati a una matrice e i colori diventano i float32 che usa il motore. Tabella dei numeri di ogni tensore, statistiche, istogramma e il passaggio al Laboratorio pesi.",
    keywords: ["pesi numerici", "numeri dei pesi", "valori dei pesi", "tabella dei pesi", "matrice di numeri", "float32", "weights values", "tensore", "statistiche dei pesi"],
  },
  {
    id: "numeri.training",
    title: "I pesi che nascono durante il training",
    text: "Gli 11 snapshot veri del training su Tiny Shakespeare, dallo step 0 (numeri casuali) allo step 5000: heatmap e numeri cambiano davanti a te. Anche con la stima lineare tra due snapshot, dichiarata come tale.",
    keywords: ["snapshot", "pesi durante il training", "evoluzione dei pesi", "inizializzazione", "pesi casuali", "float16", "interpolazione", "Tiny Shakespeare", "nascita dei pesi", "guarda i pesi nascere"],
  },
  {
    id: "rete",
    title: "Rappresentazione Rete neurale",
    text: "Lo stesso GPT come rete di neuroni e pesi: a righe, un blocco per riga, oppure da sinistra a destra con i token del prompt in profondità.",
    keywords: ["rete neurale", "neural network", "neuroni", "rappresentazione", "selettore"],
  },
  {
    id: "cervello",
    title: "Cervello: da dove vengono le idee del GPT",
    text: "Analogie tra il GPT del corso e il cervello (neurone e ReLU, plasticità, attention, normalizzazione, predizione, gerarchia, cervelletto, ippocampo), ognuna con dove l'analogia non regge e le fonti; modello 3D del cervello e del microcircuito della corteccia.",
    keywords: ["cervello", "brain", "neuroscienze", "analogie", "neurone biologico", "corteccia", "microcircuito", "sinapsi", "plasticità"],
  },
  {
    id: "neuroni",
    title: "Vista Rete neurale",
    text: "Tutti i neuroni e le connessioni del modello; clic su un neurone per vedere il suo calcolo.",
    keywords: ["neuroni", "rete neurale", "a righe", "neural network", "connessioni", "attivazioni", "MLP"],
  },
  {
    id: "neuroni-profondita",
    title: "Rete neurale orizzontale con profondità",
    text: "La rete vera e completa del GPT da sinistra a destra, con tutti i token del prompt in profondità: archi di attention tra i token con la maschera causale, connessioni residue, LayerNorm token per token e i pesi veri della fetta scelta. Gira la camera per vedere lo spessore.",
    keywords: ["rete neurale orizzontale", "profondità", "spessore", "disposizione orizzontale", "token in profondità", "attention tra token", "maschera causale", "neural network 3D", "layer", "strati"],
  },
  {
    id: "contesto",
    title: "Vista Contesto",
    text: "Dal messaggio dell'utente al contesto dell'LLM: guardrail, RAG con pgvector, prompt, strumenti e streaming.",
    keywords: ["contesto", "context", "RAG", "prompt di sistema", "system prompt", "Spring AI", "chatbot"],
  },
  {
    id: "zoom.detail",
    title: "Zoom: livello Dettaglio",
    text: "Da vicino: ogni matrice di pesi, una cella per numero appreso, e i pattern di attention.",
    keywords: ["zoom semantico", "LOD", "dettaglio", "detail", "matrici"],
  },
  {
    id: "zoom.blocks",
    title: "Zoom: livello Blocchi",
    text: "Un rettangolo per componente con nome, numero di pesi e una frase su cosa fa.",
    keywords: ["zoom semantico", "LOD", "blocchi", "blocks", "componenti"],
  },
  {
    id: "zoom.macro",
    title: "Zoom: livello Panoramica",
    text: "Input, Transformer e Output con le frecce del flusso: il diagramma a blocchi del paper.",
    keywords: ["zoom semantico", "LOD", "panoramica", "overview", "diagramma a blocchi", "macro"],
  },
].map((v) => ({ ...v, kind: "view" }));

/**
 * Passi del percorso della vista Contesto (id del suo Menu, vedi context/texts.js pathSteps).
 * I testi veri dipendono dalla domanda scelta: qui una frase fissa che dice di cosa parla il passo.
 */
const CONTEXT_STEPS = [
  ["ctx.react", "L'utente scrive nella chat", "React invia la domanda al backend Spring Boot con l'ID della conversazione.", ["React", "frontend", "POST", "chat"]],
  ["ctx.guard", "Il guardrail controlla l'input", "InputGuard controlla lunghezza, tentativi di prompt injection e oscura IBAN e codici fiscali.", ["guardrail", "InputGuard", "prompt injection", "sicurezza", "security"]],
  ["ctx.embed", "La domanda diventa un vettore", "Il modello di embedding trasforma la domanda nelle coordinate del significato.", ["embedding", "text-embedding-3-small", "vettore", "embedding model"]],
  ["ctx.vectordb", "pgvector cerca i chunk vicini", "Ricerca per distanza del coseno con l'indice HNSW di PostgreSQL.", ["pgvector", "database vettoriale", "vector database", "HNSW", "similarità coseno", "cosine similarity"]],
  ["ctx.chunks", "topK e soglia scelgono i chunk", "Spring AI tiene al massimo topK chunk sopra la soglia di similarità, con i metadati delle fonti.", ["topK", "top-k", "soglia", "similarity threshold", "chunk", "metadati"]],
  ["ctx.prompt", "Si assembla la cartellina", "Gli advisor di Spring AI montano strumenti, system prompt, storico, domanda e documenti.", ["prompt", "advisor", "prompt caching", "contesto"]],
  ["ctx.prompt.tools", "Definizioni degli strumenti", "Le descrizioni dei metodi Java che l'LLM può chiedere di eseguire.", ["tool", "strumenti", "function calling", "@Tool"]],
  ["ctx.prompt.system", "System prompt", "Le istruzioni fisse che dicono al modello chi è e come rispondere.", ["system prompt", "prompt di sistema", "istruzioni"]],
  ["ctx.prompt.history", "Storico della conversazione", "I messaggi precedenti rimandati a ogni chiamata: il modello non ricorda nulla.", ["memoria", "chat memory", "history", "storico"]],
  ["ctx.prompt.documents", "Domanda e documenti", "La domanda dell'utente con i chunk recuperati dal database vettoriale.", ["documenti", "RAG", "chunk", "retrieval"]],
  ["ctx.tokenize", "Il testo diventa token", "Tutto il contesto viene spezzato in token: è ciò che legge davvero l'LLM.", ["token", "tokenizer", "context window", "finestra di contesto"]],
  ["ctx.llm", "Il contesto entra nella pila", "I token del contesto attraversano un transformer decoder come quello del tab.", ["LLM", "transformer", "decoder", "gpt-4.1-mini"]],
  ["ctx.toolcall", "L'LLM chiede uno strumento", "Invece di rispondere, il modello chiede di chiamare un metodo con certi argomenti.", ["tool call", "function calling", "chiamata a strumento"]],
  ["ctx.java", "Spring esegue il metodo Java", "Spring AI esegue il metodo annotato e ne raccoglie il risultato.", ["Java", "Spring", "@Tool", "metodo"]],
  ["ctx.toolresult", "Il risultato torna nel contesto", "Il risultato dello strumento si aggiunge al contesto e parte una seconda chiamata all'LLM.", ["tool result", "seconda chiamata", "agente"]],
  ["ctx.stream", "Streaming con Server-Sent Events", "La risposta arriva un token alla volta con SSE.", ["streaming", "SSE", "Server-Sent Events", "Flux"]],
  ["ctx.stream.react", "React mostra risposta e fonti", "Il frontend compone la risposta e mostra le fonti dei chunk usati.", ["React", "fonti", "citazioni", "frontend"]],
].map(([id, title, text, keywords]) => ({ id, kind: "view", title: `Contesto: ${title}`, text, keywords: ["vista Contesto", ...keywords] }));

export const searchIndex = [...steps, ...actions, ...views, ...CONTEXT_STEPS];
