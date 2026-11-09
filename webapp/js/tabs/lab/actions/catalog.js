/**
 * Catalogo delle azioni animate del Laboratorio pesi (SOLI DATI: lo importa anche search.js, che la
 * palette carica quando il tab non è mai stato aperto; quindi niente three.js, scena o motore).
 *
 * Ogni voce è la "scheda" di un'azione: l'id è quello di ui.player.register e dei link
 * #lab/go=action:<id>; titolo e riassunto compaiono nel pannello Azioni e nella ricerca. I numeri nei
 * testi sono quelli del GPT del corso (gpt/config.py): con un run compatibile di forma diversa le
 * didascalie delle azioni usano i numeri veri del run.
 */

/** Le categorie, nell'ordine del pannello. */
export const CATEGORIES = Object.freeze({
  forward: "Passaggio in avanti (numeri veri del motore)",
  training: "Training",
});

/**
 * @typedef {{id: string, category: keyof CATEGORIES, title: string, summary: string, keywords: string[],
 *   engine: boolean, history?: boolean}} ActionCard
 *   `engine`: serve il motore JS (attivazioni vere): solo i run compatibili; `history`: servono gli snapshot
 */
/** @type {readonly ActionCard[]} */
export const ACTIONS = Object.freeze([
  {
    id: "linear",
    category: "forward",
    title: "x · Wᵀ riga per colonna",
    summary: "Una nn.Linear al microscopio (la key di una head, o quella che scegli): ogni riga di W moltiplica x cella per cella, i prodotti si sommano, poi la bias se c'è.",
    keywords: ["nn.Linear", "prodotto matrice vettore", "matmul", "moltiplicazione", "riga per colonna", "key", "query", "value", "bias", "prodotto scalare"],
    engine: true,
  },
  {
    id: "attention",
    category: "forward",
    title: "q · kᵀ, scala, maschera e softmax",
    summary: "I punteggi di una head: ogni query incontra ogni key, la scala 1/√16, la maschera causale tril mette −∞ sul futuro, la softmax riga per riga dà i pesi di attention.",
    keywords: ["attention", "self-attention", "query", "key", "softmax", "maschera causale", "tril", "masked_fill", "scaled dot-product", "punteggi"],
    engine: true,
  },
  {
    id: "attention-v",
    category: "forward",
    title: "att · V, concatenazione, proj e residuo",
    summary: "La media dei value pesata dall'attention, le 4 head affiancate (torch.cat), la proiezione proj e la connessione residua x + sa(ln1(x)).",
    keywords: ["value", "concat", "torch.cat", "multi-head", "proj", "proiezione", "residuo", "residual", "skip connection"],
    engine: true,
  },
  {
    id: "layernorm",
    category: "forward",
    title: "LayerNorm: media, varianza, γ e β",
    summary: "Un token normalizzato passo per passo: la media μ, la varianza σ², i numeri riportati a media 0 e varianza 1, poi γ che allarga e β che sposta ogni dimensione.",
    keywords: ["layernorm", "normalizzazione", "media", "varianza", "gamma", "beta", "ln1", "ln2", "ln_f", "eps"],
    engine: true,
  },
  {
    id: "ffn",
    category: "forward",
    title: "Feed-forward: 64 → 256, ReLU, 256 → 64",
    summary: "I 256 neuroni nascosti di un token: il primo strato, la ReLU che spegne i negativi (quanti?), il secondo strato che torna a 64 numeri e il residuo.",
    keywords: ["feed-forward", "ffn", "mlp", "relu", "neuroni nascosti", "hidden", "attivazione", "residuo"],
    engine: true,
  },
  {
    id: "logits",
    category: "forward",
    title: "Logits e probabilità del prossimo carattere",
    summary: "lm_head trasforma i 64 numeri di ln_f in 65 punteggi, uno per carattere; la softmax li fa diventare probabilità che sommano a 1; il carattere scelto.",
    keywords: ["logits", "lm_head", "softmax", "probabilità", "prossimo token", "previsione", "unembedding", "argmax", "temperatura"],
    engine: true,
  },
  {
    id: "training-step",
    category: "training",
    title: "Un passo di training su lm_head",
    summary: "Un esempio vero: la loss, il gradiente esatto di lm_head (p − y e (p − y)·hᵀ), l'aggiornamento con il learning rate del training (SGD e il primo passo di AdamW) e la loss che scende.",
    keywords: ["training", "gradiente", "backpropagation", "loss", "cross-entropy", "SGD", "AdamW", "learning rate", "discesa del gradiente", "aggiornamento"],
    engine: true,
  },
  {
    id: "training-history",
    category: "training",
    title: "I pesi durante il training",
    summary: "Gli 11 snapshot veri, dallo step 0 (numeri casuali) ai pesi finali, sulla scena: la loss che scende e il testo generato a ogni fase.",
    keywords: ["training", "snapshot", "storia", "evoluzione dei pesi", "loss", "step", "inizializzazione", "testo generato"],
    engine: false,
    history: true,
  },
]);

/** Gli id delle azioni (per i controlli e i test). */
export const ACTION_IDS = Object.freeze(ACTIONS.map((a) => a.id));

/** La scheda di un'azione per id (null se non c'è). */
export function actionCard(id) {
  return ACTIONS.find((a) => a.id === id) ?? null;
}
