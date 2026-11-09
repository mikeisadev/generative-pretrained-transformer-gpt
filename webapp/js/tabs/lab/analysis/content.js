/**
 * Testi della fase 2 del Laboratorio pesi: confronto A/B, grafici, benchmark (modulo di SOLI DATI: lo
 * importa anche analysis/search.js, che la palette carica senza aprire il tab).
 *
 * Convenzione di tutto il modulo: A è il run nella scena (quello che guardi, spesso il "dopo"), B il
 * termine di paragone (spesso il "prima": il GPT del corso, lo step 0). Le differenze sono sempre
 * A − B, il cambio relativo è ‖A − B‖ / ‖B‖: "di quanto A si è allontanato da B, in proporzione a B".
 */

export const COMPARE = {
  intro: "A è il run nella scena, B il termine di paragone: due run (il GPT del corso e la tua versione migliorata) o due momenti dello stesso training (step 500 e step 5.000). Ogni tensore di A si confronta con quello di B che ha lo stesso nome.",
  analogy: "Come due foto dello stesso paesaggio scattate in giorni diversi: sovrapponendole si vede subito dove è cambiato qualcosa. Qui le «foto» sono le matrici dei pesi e la differenza si legge cella per cella.",
  empty: "Scegli B qui sopra (o con il pulsante «B» nel pannello Run): compaiono la tabella dei tensori più cambiati, la differenza A − B nella scena e gli istogrammi sovrapposti.",
  match: "Come si abbinano: per nome esatto, poi per nome senza i prefissi module. e _orig_mod. (aggiunti da DataParallel e torch.compile). Si confrontano numero per numero solo i tensori con la stessa forma: con forme diverse (es. una feed-forward da 192 neuroni invece di 256) la cella (i, j) di A e quella di B non sono lo stesso peso.",
  float16: "Le differenze sono minuscole: lo snapshot salva i pesi in float16 (circa 3 cifre significative), i pesi finali sono in float32. È l'arrotondamento, non un altro training: sono gli stessi pesi.",
  identical: "A e B sono identici, bit per bit: nessuna differenza da mostrare.",
};

/** Le colonne della tabella del confronto: intestazione breve e spiegazione (title e lettore di schermo). */
export const COLUMN_HELP = {
  name: "Il nome del tensore in PyTorch (A).",
  shape: "La forma del tensore: righe × colonne.",
  normA: "‖A‖: la norma L2 dei pesi di A, cioè la radice della somma dei quadrati. Quanto è «grande» il tensore.",
  normB: "‖B‖: la norma L2 dei pesi di B.",
  diffNorm: "‖A − B‖: la norma della differenza. 0 = identici.",
  relChange: "Cambio relativo ‖A − B‖ / ‖B‖: di quanto A si è allontanato da B, in proporzione a B. 100% = la differenza è grande quanto B.",
  cosine: "Coseno tra A e B visti come due vettori: 1 = stessa direzione (solo la scala può cambiare), 0 = scorrelati, −1 = opposti.",
  maxAbsDiff: "max |Δ|: la differenza più grande tra due pesi nella stessa posizione.",
};

export const DIFF = {
  toggle: "Mostra A − B nella scena",
  scaleCommon: "Scala comune",
  scaleCommonHelp: "Lo stesso colore vuol dire la stessa differenza in tutte le matrici: il colore pieno è la differenza più grande di tutto il confronto.",
  scaleWeights: "Scala dei pesi",
  scaleWeightsHelp: "Ogni matrice con la scala dei suoi pesi: il colore pieno è una differenza grande quanto il peso più grande di quella matrice. Si vede la struttura anche dove le differenze sono piccole.",
  positive: "A più grande di B",
  negative: "A più piccolo di B",
  zero: "nessuna differenza",
  veiled: "velati: tensori senza un gemello della stessa forma in B (non si confrontano)",
  numbers: "Avvicinandoti nella scena, nelle celle compaiono i numeri della differenza A − B.",
  offForAction: "La differenza A − B si è spenta: è partita un'azione, che mostra i pesi veri del run A.",
  offForOther: "La differenza A − B si è spenta: la scena ora mostra altri valori.",
};

export const HISTOGRAM = {
  title: "Distribuzione dei valori: A e B sovrapposti",
  hint: "Gli stessi intervalli per A e B: si vede se il training ha allargato, stretto o spostato i pesi. Le distribuzioni si confrontano anche con forme diverse.",
  none: "Scegli un tensore (riga della tabella, clic nella scena o pannello Modello) per vedere le sue distribuzioni in A e in B.",
};

export const PROMPT = {
  title: "Il prossimo carattere: A contro B",
  intro: "Scrivi un inizio di testo: il motore JS calcola, con i pesi di A e di B, la probabilità di ogni carattere che potrebbe venire dopo. Logits = i punteggi grezzi di lm_head prima della softmax.",
  incompatible: "il motore JS non esegue questo run",
  tooLong: "Più lungo del contesto: contano solo gli ultimi {{n}} caratteri (block_size), come durante la generazione.",
};

/** La frase d'esempio del training (recorder.example_prompt): come continua nel testo vero. */
export const EXAMPLE_CONTINUATIONS = Object.freeze({
  "ROMEO:\nBut, soft! what light thr": { next: "o", note: "Nel testo di Tiny Shakespeare (data/input.txt, riga 16.870, nella parte di training) la frase continua «…through yonder window breaks?»: il carattere giusto è «o»." },
});

export const CHARTS = {
  trainingTitle: "Training: loss e ottimizzazione",
  trainingDesc: "La storia vera registrata durante il training (trace.json di gpt/recording.py). A e B insieme quando entrambi hanno la storia; la linea verticale è lo step di A nella scena.",
  noTrace: "Nessuna storia del training (trace.json) né per A né per B: esporta il run con scripts/export_run.py --trace.",
  groupsTitle: "Training per gruppo di pesi",
  groupsDesc: "Gli stessi gruppi di gpt/recording.py: attention (head + proj), feed-forward, le due LayerNorm di ogni blocco, embedding, ln_f, lm_head.",
  snapsTitle: "Metriche degli snapshot",
  snapsDesc: "Misurate a ogni snapshot sullo stesso batch fisso di validation (64 sequenze, seme fisso): si possono confrontare tra step e tra run con lo stesso registratore.",
  promptTitle: "Il prossimo carattere sul tuo prompt",
  promptDesc: "Calcolato ora dal motore JS con i pesi di A e di B (solo run che il motore sa eseguire). Il prompt si scrive nel pannello Confronto.",
  benchTitle: "Benchmark",
  benchDesc: "Lo stesso metro per tutti i run (protocollo finestre-contigue-v1): dai benchmark.json del Python e dal calcolo nel browser.",
};

export const BENCH = {
  intro: "Un benchmark è lo stesso compito per tutti: la validation di Tiny Shakespeare (111.540 caratteri mai visti nel training), tagliata in finestre contigue, senza nessuna casualità. Due run misurati così si possono mettere uno accanto all'altro.",
  analogy: "Come un compito in classe uguale per tutti: stesse domande, stesse regole di correzione. Cambia solo lo studente (i pesi).",
  noBenchmark: "Nessun benchmark.json per questo run: esportalo con scripts/export_run.py … --benchmark (oppure misuralo con scripts/benchmark.py).",
  speed: "Token/s: dichiarati nel benchmark.json, misurati sulla macchina che l'ha calcolato. Dipendono dal computer e dal suo carico: non dicono quale modello è migliore.",
  sampled: "Il testo campionato (seme 1337) non si ricalcola qui: PyTorch e il motore JS usano generatori di numeri casuali diversi, quindi con lo stesso seme estraggono caratteri diversi. Il testo greedy (sempre il carattere più probabile) invece deve coincidere.",
  browserIntro: "Il motore JS di questa pagina ricalcola lo stesso protocollo sull'estratto della validation (val_excerpt.txt: 4.015 caratteri, 4.014 previsioni) e confronta i suoi numeri con quelli del Python.",
  browserIncompatible: "Il motore JS non esegue questo run, quindi qui niente calcolo nel browser: restano i numeri del Python.",
  consistencyRule: "✓ se la loss coincide entro 0,00001 e le previsioni giuste top-1 e top-5 sono le stesse; il testo greedy deve essere identico.",
  near: "≈ quasi: differenze entro le tolleranze di gpt/benchmark.py (loss entro 0,0001, ±2 previsioni giuste: un quasi pari merito tra due logits può rompersi in modo diverso).",
  interpolated: "Pesi interpolati tra due snapshot: un'approssimazione che il Python non ha mai misurato, quindi niente confronto.",
};

/** Che cosa vuol dire ogni metrica (righe della tabella del benchmark). */
export const METRIC_HELP = {
  loss: "Loss (nat): in media −ln p(carattere giusto). Più bassa è meglio. Tirare a caso tra 65 caratteri dà ln 65 = 4,17.",
  perplexity: "Perplexity = e^loss: è come se il modello esitasse tra questo numero di caratteri equiprobabili. Più bassa è meglio.",
  bpc: "Bit per carattere: la loss in bit (loss / ln 2). Quanti bit servirebbero per comprimere ogni carattere con questo modello.",
  top1: "Top-1: quante volte il carattere giusto è il più probabile.",
  top5: "Top-5: quante volte il carattere giusto è tra i 5 più probabili.",
  entropy: "Entropia media (nat): quanto il modello è indeciso, giusto o sbagliato che sia. Non ha un verso «meglio»: un modello sicuro e sbagliato ha entropia bassa.",
  params: "Parametri: quanti pesi ha il modello.",
  tokens_per_s: "Token al secondo in generazione, sulla macchina del benchmark.",
  eval_tokens_per_s: "Previsioni al secondo durante la valutazione, sulla macchina del benchmark.",
};

/** Le fonti della fase 2 (gruppo aggiunto al pannello Fonti). */
export const ANALYSIS_SOURCES = {
  title: "Confronto, grafici e benchmark",
  items: [
    { title: "Karpathy, «Building makemore Part 3»: activations, gradients, update ratio", url: "https://www.youtube.com/watch?v=P6sfmUTpUmc", note: "Il rapporto ‖Δw‖/‖w‖ per gruppo come termometro del training (attorno a 10⁻³ in scala log), lo stesso dei grafici per gruppo." },
    { title: "Shannon (1951), Prediction and Entropy of Printed English", url: "https://doi.org/10.1002/j.1538-7305.1951.tb01366.x", note: "Bit per carattere ed entropia di un testo: le unità del benchmark." },
    { title: "Jelinek et al. (1977), Perplexity: a measure of the difficulty of speech recognition tasks", url: "https://doi.org/10.1121/1.2016299", note: "La perplexity come «numero di scelte equiprobabili»." },
    { title: "IEEE 754 e il formato float16", url: "https://en.wikipedia.org/wiki/Half-precision_floating-point_format", note: "Perché uno snapshot float16 differisce dai pesi float32 di circa 10⁻³: 11 bit di mantissa." },
  ],
};
