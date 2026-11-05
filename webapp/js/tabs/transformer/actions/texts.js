/**
 * Testi didattici del pannello "Azioni": introduzioni, approfondimenti con formule (KaTeX) e fonti.
 * Separati dal codice così si rileggono e si correggono in un posto solo.
 *
 * Le formule descrivono esattamente il codice del corso (gpt/*.py) e il motore JS (js/engine):
 * se cambia il codice, vanno ricontrollate qui.
 */

export const INTRO = `Qui il modello <strong>lavora</strong> davanti a te, con i suoi pesi veri. Le azioni sono di due tipi,
e conviene tenerli ben distinti:`;

/** Il segnaposto {{params}} si riempie con il dato vero di model.json (vedi pane.js). */
export const CATEGORIES = {
  prompt: {
    title: "Prompt (inferenza)",
    lead: `Il modello è già addestrato: i suoi {{params}} pesi sono <strong>fissi</strong>. Gli dai un testo e lui
      calcola, un carattere alla volta, che cosa viene dopo. È quello che succede ogni volta che scrivi a un
      chatbot: nessun peso cambia, il modello "legge" e "prevede".`,
  },
  training: {
    title: "Training",
    lead: `Qui invece si vede <strong>come sono stati trovati</strong> quei pesi. Il modello legge pezzi del testo di
      Shakespeare, prova a indovinare il carattere successivo, misura quanto ha sbagliato (la loss) e corregge
      di pochissimo ogni peso. 5000 volte. I numeri sono quelli del training vero, registrato da
      <code>scripts/record_training.py</code>.`,
  },
};

/** Riquadri "Approfondimento": {title, items: [{tex, note}]}. */
export const DEEP_PROMPT = [
  {
    title: "Approfondimento: dal testo ai vettori",
    items: [
      { tex: "\\text{idx}_t = \\text{stoi}(c_t), \\quad t < 32", note: "Ogni carattere diventa il suo numero (id) nel vocabolario. Il modello vede al massimo gli ultimi 32." },
      { tex: "x_t = E_{\\text{tok}}[\\text{idx}_t] + E_{\\text{pos}}[t]", note: "Lookup: si legge una riga della tabella dei token e una della tabella delle posizioni (64 numeri ciascuna) e si sommano." },
      { tex: "\\mathrm{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + 10^{-5}}} + \\beta", note: "LayerNorm: ogni vettore viene centrato e riscalato, poi γ e β (appresi) lo rimodellano." },
    ],
  },
  {
    title: "Approfondimento: attention e feed-forward",
    items: [
      { tex: "Q = X W_Q^{\\top},\\quad K = X W_K^{\\top},\\quad V = X W_V^{\\top}", note: "X contiene i T token del contesto, uno per riga (T × 64). Ogni head ha le sue tre matrici di forma 16 × 64, come i pesi di nn.Linear (che calcola x·Wᵀ): da 64 numeri a 16. Query = cosa cerco, key = cosa offro, value = cosa passo." },
      { tex: "\\mathrm{att} = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{16}} + M\\right)", note: "Punteggi scalati per √16 = 4, poi la maschera causale M e la softmax riga per riga." },
      { tex: "M_{ij} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}", note: "M vieta di guardare il futuro: dopo la softmax quei pesi valgono esattamente 0." },
      { tex: "\\text{out}_h = \\mathrm{att}_h\\, V_h", note: "Ogni head fa la media pesata dei value." },
      { tex: "x_t \\leftarrow x_t + W_O\\,[\\text{out}_0; \\dots; \\text{out}_3]_t + b_O", note: "Per ogni posizione t (vettore colonna da qui in giù): le 4 head si concatenano (4 × 16 = 64), la proiezione W_O (64 × 64) le rimescola e il risultato si somma al residual stream." },
      { tex: "h_t = \\mathrm{ReLU}(W_1\\,\\mathrm{LN}_2(x_t) + b_1)", note: "Feed-forward, primo strato: W_1 ha forma 256 × 64, quindi 64 → 256 neuroni; la ReLU azzera i negativi." },
      { tex: "x_t \\leftarrow x_t + W_2\\, h_t + b_2", note: "Secondo strato: 256 → 64, sommato al residual stream. Lavora su ogni posizione da sola." },
    ],
  },
  {
    title: "Approfondimento: dai logits al token",
    items: [
      { tex: "p_i = \\frac{e^{z_i / \\tau}}{\\sum_j e^{z_j / \\tau}}", note: "Softmax con temperatura τ: con τ < 1 la distribuzione si concentra sui token più probabili, con τ > 1 si appiattisce." },
      { tex: "\\text{token} = \\min \\{ k : u < \\textstyle\\sum_{i \\le k} p_i \\}", note: "La ruota: u è un numero casuale tra 0 e 1; le probabilità messe in fila (in ordine di id) formano spicchi e u cade in uno di essi." },
      { tex: "H(p) = -\\sum_i p_i \\ln p_i", note: "Entropia, in nat: 0 se il modello è sicurissimo, ln 65 ≈ 4,17 se non ha nessuna preferenza." },
    ],
  },
];

export const DEEP_TRAINING = [
  {
    title: "Approfondimento: la loss",
    items: [
      { tex: "\\mathcal{L} = -\\frac{1}{BT} \\sum_{b,t} \\ln p_\\theta(y_{b,t} \\mid x_{b,\\le t})", note: "Cross-entropy media su B = 16 finestre × T = 32 posizioni: 512 previsioni per step." },
      { tex: "-\\ln\\tfrac{1}{65} = \\ln 65 \\approx 4{,}17", note: "Un modello che non sa nulla dà 1/65 a ogni carattere. Il training vero parte da 4,41: l'inizializzazione casuale è appena peggio dell'uniforme." },
      { tex: "\\text{perplexity} = e^{\\mathcal{L}}", note: "\"Tra quanti caratteri, in media, il modello è indeciso\": circa 82 all'inizio, circa 6 alla fine (validation)." },
    ],
  },
  {
    title: "Approfondimento: backward e AdamW",
    items: [
      { tex: "\\frac{\\partial \\mathcal{L}}{\\partial z_i} = \\frac{p_i - \\mathbb{1}[i = y]}{BT}", note: "Il primo gradiente, sui logits, ha una forma semplicissima: probabilità meno il valore giusto." },
      { tex: "g_t = \\nabla_\\theta \\mathcal{L}(\\theta_{t-1})", note: "loss.backward() lo propaga con la regola della catena, strato per strato, dall'uscita fino agli embedding." },
      { tex: "\\theta \\leftarrow \\theta - \\gamma\\lambda\\theta", note: "AdamW (formule di PyTorch): prima il weight decay \"disaccoppiato\"…" },
      { tex: "m_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\, g_t", note: "…poi la media mobile del gradiente…" },
      { tex: "v_t = \\beta_2 v_{t-1} + (1-\\beta_2)\\, g_t^2", note: "…e del suo quadrato." },
      { tex: "\\theta_t = \\theta - \\gamma\\, \\frac{\\hat m_t}{\\sqrt{\\hat v_t} + \\epsilon}", note: "Con m̂ e v̂ corretti per i primi passi (divisi per 1 − βᵗ). Nel nostro training γ = 0,001, β = (0,9; 0,999), ε = 10⁻⁸, λ = 0,01: i valori predefiniti di torch.optim.AdamW." },
      { tex: "\\lVert \\Delta\\theta \\rVert / \\lVert \\theta \\rVert \\approx 10^{-3}", note: "Regola pratica (appunti CS231n): se il rapporto update/peso è molto più piccolo il learning rate è troppo basso, se è molto più grande è troppo alto." },
    ],
  },
];

/** Fonti: controllate durante lo sviluppo (vedi il report del modulo). */
export const SOURCES = [
  { title: "Vaswani et al., Attention Is All You Need (2017)", url: "https://arxiv.org/abs/1706.03762", note: "attention scalata per √d_k e maschera" },
  { title: "Karpathy, ng-video-lecture: gpt.py", url: "https://github.com/karpathy/ng-video-lecture", note: "il codice da cui nasce il GPT del corso" },
  { title: "Ba, Kiros, Hinton, Layer Normalization (2016)", url: "https://arxiv.org/abs/1607.06450", note: "LayerNorm" },
  { title: "Kingma, Ba, Adam (2014)", url: "https://arxiv.org/abs/1412.6980", note: "l'ottimizzatore Adam" },
  { title: "Loshchilov, Hutter, Decoupled Weight Decay Regularization (2017)", url: "https://arxiv.org/abs/1711.05101", note: "AdamW" },
  { title: "PyTorch, torch.optim.AdamW (codice e formule)", url: "https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py", note: "valori predefiniti e algoritmo" },
  { title: "CS231n, Neural Networks Part 3: rapporto update/peso", url: "https://cs231n.github.io/neural-networks-3/", note: "\"somewhere around 1e-3\"" },
  { title: "nostalgebraist, interpreting GPT: the logit lens (2020)", url: "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens", note: "la tecnica del grafico \"logit lens\"" },
];
