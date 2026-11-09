/**
 * Testi del Laboratorio pesi (modulo di soli dati: lo importano i pannelli e search.js, che non può
 * caricare la scena). Comandi Python veri: README.md, sezione "Esportare un run e confrontarlo nel
 * Laboratorio pesi", e scripts/export_run.py, scripts/benchmark.py, scripts/record_training.py.
 */

export const TAB_HINT_MOUSE =
  "I 209.729 pesi del GPT del corso disposti come il codice Python. Rotellina: da lontano i moduli con i conteggi, da vicino le celle colorate e poi i numeri veri. Clic su una matrice per la sua tabella; trascina per ruotare, tasto destro o W A S D per spostarti, C per tornare a tutto il modello (? mostra tutti i comandi).";
export const TAB_HINT_TOUCH =
  "I 209.729 pesi del GPT del corso disposti come il codice Python. Pizzica per lo zoom: da lontano i moduli, da vicino le celle e i numeri veri. Tocca una matrice per la sua tabella; due dita per spostarti.";

export const INTRO = {
  title: "I pesi del tuo GPT, come nel codice",
  text: "Ogni rettangolo è un modulo di gpt/*.py con la riga Python che lo crea (es. self.key = nn.Linear(…)); dentro, ogni parametro è una matrice: 1 cella = 1 peso, arancione positivo, blu negativo, scuro vicino a zero. Il flusso va da sinistra a destra come il passaggio in avanti: embedding, i 4 blocchi, ln_f, lm_head. Avvicinandoti compaiono i numeri veri, gli stessi float32 che usa il motore.",
  analogy: "Come un grande foglio di calcolo diviso in schede: il codice Python dice come sono fatte le schede e quanto sono grandi, il training riempie le celle. Qui guardi le celle, scheda per scheda.",
};

export const LEGEND = [
  { swatch: "pos", text: "peso positivo" },
  { swatch: "neg", text: "peso negativo" },
  { swatch: "zero", text: "vicino a zero" },
];

/** Il flusso "prima e dopo" (README): modifica, riaddestra, esporta, carica, confronta. */
export const FLOW = [
  { id: "modifica", title: "Modifica", text: "Cambia il codice in gpt/ (architettura, attivazione, iperparametri) o il training: anche con un assistente come Claude." },
  { id: "riaddestra", title: "Riaddestra", text: "Registra la storia del training (loss, snapshot dei pesi) con scripts/record_training.py, sempre con --no-export." },
  { id: "esporta", title: "Esporta", text: "scripts/export_run.py scrive la cartella runs/<nome>/ con pesi, run.json, benchmark e storia." },
  { id: "carica", title: "Carica", text: "Scegli o trascina qui la cartella del run: i pesi si dispongono come il codice." },
  { id: "confronta", title: "Confronta", text: "Guarda i pesi, confrontali con il GPT del corso o tra fasi del training, leggi grafici e benchmark." },
];

/**
 * I comandi, dalla radice del repository. `--no-export` è importante: senza, record_training.py
 * riscriverebbe i dati della web app (webapp/data).
 */
export const EXPORT_STEPS = [
  {
    title: "0. Il metro di riferimento",
    note: "Il GPT del corso è già misurato in webapp/data/benchmark/builtin.json; per rifarlo:",
    command: "python scripts/benchmark.py webapp/data",
  },
  {
    title: "1. Riaddestra registrando la storia",
    note: "Dopo aver modificato gpt/ (qui un esempio: GELU al posto di ReLU in gpt/feedforward.py). Senza storia, più veloce: python train.py --checkpoint checkpoints/gelu/gpt.pt",
    command: "python scripts/record_training.py --checkpoint checkpoints/gelu/gpt.pt --out checkpoints/gelu/training --no-export",
  },
  {
    title: "2. Esporta il run",
    note: "Scrive runs/gpt-con-gelu/ (run.json, weights.json + weights.f32.bin, benchmark.json, training/). --benchmark misura con il codice modificato; --not-engine-compatible quando cambi il calcolo senza cambiare le forme (GELU): il browser mostra i pesi senza eseguirli.",
    command: 'python scripts/export_run.py checkpoints/gelu/gpt.pt --name "GPT con GELU" --description "FFN con GELU" --trace checkpoints/gelu/training --benchmark --not-engine-compatible',
  },
  {
    title: "3. Confronta in terminale",
    note: "Tabella A/B in italiano (A e B possono essere checkpoint, cartelle di run o benchmark.json già calcolati).",
    command: "python scripts/benchmark.py webapp/data/benchmark/builtin.json runs/gpt-con-gelu/benchmark.json",
  },
  {
    title: "4. Carica qui la cartella",
    note: "Pulsante «Scegli una cartella» e poi runs/gpt-con-gelu, oppure trascinala sulla scena. Su una macchina già occupata aggiungi --threads 1 ai comandi di benchmark: i numeri non cambiano.",
    command: "runs/gpt-con-gelu/",
  },
];

/** Che cosa si può caricare (formati di js/shared/weights/formats.js). */
export const ACCEPTED_FILES = [
  "una cartella esportata con scripts/export_run.py (run.json, weights.json + weights.f32.bin, benchmark.json, training/)",
  "più cartelle insieme (es. runs/): un run per sottocartella",
  "file .safetensors (F32, F16, BF16), anche in shard",
  "la storia di un training da sola (trace.json + snap_XXXXX.f16.bin): i pesi sono l'ultimo snapshot",
];

/** Onestà: cosa è vero e cosa no, in ogni vista del tab. */
export const HONESTY = {
  weights: "I numeri sono i pesi veri del run (float32 dei pesi finali, float16 degli snapshot): nessuna simulazione.",
  interpolation: "Interpolazione: tra due snapshot i pesi sono una media pesata, un'approssimazione (il training vero non va in linea retta). Gli snapshot veri restano gli unici pesi davvero salvati.",
  tril: "La maschera tril è un buffer: numeri fissi costruiti dal codice, non appresi e non salvati nei pesi.",
};

export const SOURCES = [
  {
    title: "Codice e dati del corso",
    items: [
      { title: "Karpathy, ng-video-lecture: gpt.py", url: "https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py", note: "Il GPT da cui è ricostruito gpt/ del corso: stessi moduli, stessi nomi dei pesi." },
      { title: "PyTorch: torch.nn.Linear", url: "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html", note: "y = x Wᵀ + b, peso di forma (out_features, in_features): per questo una matrice è alta quanto le sue uscite." },
      { title: "PyTorch: torch.nn.LayerNorm", url: "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html", note: "γ (weight) e β (bias), un numero per dimensione." },
      { title: "PyTorch: torch.nn.Embedding", url: "https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html", note: "Una tabella (num_embeddings, embedding_dim): il token sceglie la sua riga." },
      { title: "PyTorch: Module.register_buffer", url: "https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer", note: "I buffer (come tril) non sono parametri; con persistent=False non finiscono nello state_dict." },
    ],
  },
  {
    title: "Formati dei pesi",
    items: [
      { title: "Hugging Face: formato safetensors", url: "https://huggingface.co/docs/safetensors/index", note: "Intestazione JSON + dati grezzi, nessun codice eseguito al caricamento: per questo il Laboratorio lo accetta." },
      { title: "PyTorch: torch.load(weights_only=True)", url: "https://pytorch.org/docs/stable/generated/torch.load.html", note: "Come scripts/export_run.py legge i checkpoint senza eseguire codice nascosto nel file." },
    ],
  },
  {
    title: "Training e confronto",
    items: [
      { title: "Loshchilov e Hutter (2019), Decoupled Weight Decay Regularization", url: "https://arxiv.org/abs/1711.05101", note: "AdamW, l'ottimizzatore del training del corso (gpt/training.py)." },
      { title: "Vaswani et al. (2017), Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", note: "Il blocco transformer: attention con key, query, value e feed-forward." },
    ],
  },
];
