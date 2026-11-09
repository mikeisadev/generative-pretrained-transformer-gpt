/**
 * Indice della ricerca globale del Laboratorio pesi (SOLI DATI: niente three.js, niente scena). Lo carica
 * la palette anche se il tab non è mai stato aperto. La ricerca accetta solo voci step | action | view:
 * i tensori e gli step del training sono "viste" con id tensor:<nome> e step:<n> (links.js li risolve).
 * Le voci delle azioni, del confronto, dei grafici e del benchmark (fase 2) stanno in actions/search.js e
 * analysis/search.js.
 */
import { searchEntries as actionEntries } from "./actions/search.js";
import { searchEntries as analysisEntries } from "./analysis/search.js";
import { EXPORT_STEPS, INTRO } from "./content.js";

/** I tensori principali del GPT del corso, con i sinonimi che uno studente cercherebbe. */
const TENSORS = [
  { name: "token_embedding_table.weight", title: "Tabella degli embedding dei token (65 × 64)", text: "Una riga di 64 numeri per ogni carattere del vocabolario: la riga 43 è il carattere «e».", keywords: ["embedding", "wte", "token", "vocabolario", "nn.Embedding"] },
  { name: "position_embedding_table.weight", title: "Tabella degli embedding delle posizioni (32 × 64)", text: "Una riga per ogni posizione del contesto: dice al modello dove sta il token.", keywords: ["posizione", "positional embedding", "wpe"] },
  { name: "blocks.0.sa.heads.0.key.weight", title: "Key della head 0 del blocco 0 (16 × 64)", text: "self.key = nn.Linear(config.n_embd, self.head_size, bias=False): che cosa contiene ogni token.", keywords: ["key", "attention", "head", "nn.Linear", "Wk"] },
  { name: "blocks.0.sa.heads.0.query.weight", title: "Query della head 0 del blocco 0 (16 × 64)", text: "self.query = nn.Linear(…): che cosa cerca ogni token.", keywords: ["query", "attention", "Wq"] },
  { name: "blocks.0.sa.heads.0.value.weight", title: "Value della head 0 del blocco 0 (16 × 64)", text: "self.value = nn.Linear(…): che cosa comunica ogni token.", keywords: ["value", "attention", "Wv"] },
  { name: "blocks.0.sa.heads.0.tril", title: "La maschera causale tril (buffer, non è un peso)", text: "register_buffer(\"tril\", …): 1 sotto la diagonale, 0 sopra. Il futuro non si guarda.", keywords: ["tril", "maschera", "causale", "buffer", "mask"] },
  { name: "blocks.0.sa.proj.weight", title: "Proiezione dell'attention del blocco 0 (64 × 64)", text: "self.proj = nn.Linear(config.n_embd, config.n_embd): rimescola le uscite delle 4 head.", keywords: ["proj", "proiezione", "multi-head"] },
  { name: "blocks.0.ffwd.net.0.weight", title: "Feed-forward, primo strato del blocco 0 (256 × 64)", text: "nn.Linear(config.n_embd, hidden): 256 neuroni nascosti per token, poi la ReLU.", keywords: ["feed-forward", "ffn", "mlp", "hidden", "relu"] },
  { name: "blocks.0.ffwd.net.2.weight", title: "Feed-forward, secondo strato del blocco 0 (64 × 256)", text: "nn.Linear(hidden, config.n_embd): da 256 numeri di nuovo a 64.", keywords: ["feed-forward", "ffn", "mlp"] },
  { name: "blocks.0.ln1.weight", title: "γ della LayerNorm ln1 del blocco 0 (64)", text: "self.ln1 = nn.LayerNorm(config.n_embd): γ allarga o stringe ogni dimensione, β la sposta.", keywords: ["layernorm", "gamma", "beta", "normalizzazione", "ln1"] },
  { name: "ln_f.weight", title: "γ della LayerNorm finale ln_f (64)", text: "self.ln_f = nn.LayerNorm(config.n_embd): l'ultima normalizzazione prima di lm_head.", keywords: ["ln_f", "layernorm finale"] },
  { name: "lm_head.weight", title: "lm_head: dai vettori ai punteggi dei caratteri (65 × 64)", text: "self.lm_head = nn.Linear(config.n_embd, config.vocab_size): una riga per carattere, il suo logit.", keywords: ["lm_head", "logits", "uscita", "output", "unembedding"] },
];

export const searchIndex = [
  { id: "modello", kind: "view", title: "Laboratorio pesi: i moduli del codice Python in 3D", text: INTRO.text, keywords: ["pesi", "weights", "print(model)", "moduli", "GPTLanguageModel", "parametri", "209.729"] },
  { id: "tensore", kind: "view", title: "Numeri, statistiche e istogramma di un tensore", text: "Il pannello Tensore: la tabella dei numeri veri con il significato di righe e colonne, media, deviazione standard, istogramma.", keywords: ["tabella", "numeri", "statistiche", "istogramma", "distribuzione"] },
  { id: "run", kind: "view", title: "I run: il GPT del corso e quelli caricati", text: "Il registro dei run: pesi finali e snapshot del GPT del corso, i run caricati da file, quelli salvati nel browser con «Dimentica».", keywords: ["run", "modelli", "checkpoint", "salvati", "dimentica", "browser"] },
  { id: "carica", kind: "view", title: "Caricare un run: cartella, file o trascinamento", text: "Scegli la cartella esportata da scripts/export_run.py oppure trascinala sulla scena: weights.json, run.json, benchmark.json, training/, .safetensors.", keywords: ["upload", "carica", "trascina", "drag and drop", "safetensors", "cartella"] },
  { id: "esporta", kind: "view", title: "Come esportare un run dal Python (prima e dopo)", text: EXPORT_STEPS.map((s) => `${s.title}: ${s.command}`).join(" "), keywords: ["export_run.py", "benchmark.py", "record_training.py", "esportare", "migliorare", "claude", "prima e dopo"] },
  { id: "step", kind: "view", title: "Step del training: i pesi che nascono dal rumore", text: "Il cursore sugli 11 snapshot veri del training (step 0 … 5000): scena, tabella e statistiche cambiano insieme.", keywords: ["training", "snapshot", "step", "rumore", "inizializzazione", "evoluzione"] },
  { id: "step:0", kind: "view", title: "I pesi allo step 0: numeri casuali", text: "Lo snapshot dello step 0: i pesi appena inizializzati, prima di qualunque training.", keywords: ["inizializzazione", "random", "step 0", "rumore"] },
  { id: "step:finali", kind: "view", title: "I pesi finali del GPT del corso", text: "I pesi alla fine dei 5.000 step, in float32: gli stessi del motore JS.", keywords: ["pesi finali", "float32", "addestrato"] },
  ...TENSORS.map((t) => ({ id: `tensor:${t.name}`, kind: "view", title: t.title, text: t.text, keywords: [t.name, ...t.keywords] })),
  { id: "rete", kind: "view", title: "Rete neurale del run nel Laboratorio", text: "La rete vera del run scelto, da sinistra a destra con i token in profondità: attivazioni e pesi veri se il motore JS la sa eseguire.", keywords: ["rete neurale", "neuroni", "orizzontale", "profondità"] },
  { id: "cervello", kind: "view", title: "Pesi e sinapsi: il Laboratorio e il cervello", text: "Le analogie tra pesi e sinapsi, training e plasticità (LTP, cervelletto), con dove non reggono e le fonti.", keywords: ["cervello", "sinapsi", "plasticità", "hebb", "cervelletto", "biologia"] },
  ...actionEntries,
  ...analysisEntries,
];
