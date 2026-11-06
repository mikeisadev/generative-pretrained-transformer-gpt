/**
 * Contenuti didattici del menu, ordinati secondo il flusso dei dati.
 *
 * Ogni voce collega tre cose:
 *   - focus:   l'id dell'oggetto 3D su cui volare (vedi scene/architecture.js)
 *   - code:    la chiave del frammento Python esportato in data/model.json
 *   - weights: i pesi da elencare, come prefissi dei nomi PyTorch
 *
 * I numeri tra doppie graffe ({{n_embd}}, {{total}}, ...) vengono sostituiti
 * con i valori letti dal modello vero: il testo resta fedele al codice.
 */

/** @typedef {{label: string, prefix: string, buffer?: boolean}} WeightRef */

export const STEPS = [
  {
    id: "data",
    title: "1. Dati e tokenizzazione",
    children: [
      {
        id: "data.dataset",
        title: "Il dataset Tiny Shakespeare",
        focus: "tokens",
        code: "data.download",
        text: `<p>Il modello impara da un unico file di testo: circa 1,1 milioni di caratteri con le opere di Shakespeare.
          Il file viene scaricato automaticamente la prima volta e ne verifichiamo l'impronta SHA-256,
          così un download corrotto non passa inosservato.</p>
          <p>Il compito del modello è uno solo: <b>indovinare il prossimo carattere</b>.</p>`,
      },
      {
        id: "data.tokenizer",
        title: "Tokenizer a caratteri",
        focus: "tokens",
        code: "data.tokenizer",
        highlight: ["def encode", "def decode"],
        text: `<p>Le reti neurali lavorano con numeri, non con lettere. Il tokenizer assegna a ognuno dei
          <b>{{vocab_size}}</b> caratteri distinti un numero intero, il suo <i>token id</i>.</p>
          <p>Esempio: <code>encode("hii")</code> → <code>[46, 47, 47]</code> e <code>decode</code> fa il contrario.</p>
          <p>ChatGPT o Claude usano tokenizer a sotto-parole con decine di migliaia di token, ma il principio è identico:
          testo → interi → testo.</p>`,
        math: [
          { tex: String.raw`\text{encode}: \Sigma \to \{0,\dots,V-1\},\quad V = {{vocab_size}}`, note: "Σ è l'insieme dei caratteri del testo." },
        ],
      },
      {
        id: "data.batch",
        title: "Batch: input e target",
        focus: "tokens",
        code: "data.get_batch",
        highlight: ["x = torch.stack", "y = torch.stack"],
        text: `<p>Ritagliamo dal testo finestre lunghe al massimo <b>block_size = {{block_size}}</b> caratteri.
          Il target <code>y</code> è la stessa finestra spostata di una posizione: per ogni posizione il modello deve prevedere
          il carattere successivo.</p>
          <p>Una sola finestra contiene quindi {{block_size}} esempi di training: dato 1 carattere prevedi il 2°,
          dati 2 caratteri prevedi il 3°, e così via. Ne elaboriamo <b>B</b> in parallelo (il <i>batch</i>).</p>`,
        math: [
          { tex: String.raw`x \in \mathbb{N}^{B\times T},\quad y_{b,t} = x_{b,t+1}`, note: "T ≤ block_size è la lunghezza del contesto." },
        ],
      },
    ],
  },
  {
    id: "bigram",
    title: "2. Il modello bigram (punto di partenza)",
    children: [
      {
        id: "bigram.model",
        title: "Una tabella e basta",
        focus: "tok_emb",
        code: "bigram",
        text: `<p>Il primo modello del video è il più semplice possibile: una tabella {{vocab_size}} × {{vocab_size}}.
          La riga del carattere corrente contiene direttamente i punteggi del carattere successivo.</p>
          <p>Funziona male perché <b>ogni token guarda solo sé stesso</b>: dopo "th" e dopo "sh" prevede la stessa cosa,
          perché vede solo "h". Il transformer nasce per far <i>comunicare</i> i token tra loro.</p>
          <p>Nella scena non c'è: è stato sostituito dall'embedding qui evidenziato, che ha la stessa forma
          di tabella ma produce vettori di {{n_embd}} numeri invece che punteggi finali.</p>`,
        math: [
          { tex: String.raw`P(x_{t+1}\mid x_{\le t}) \approx P(x_{t+1}\mid x_t) = \mathrm{softmax}(W_{x_t,:})`, note: "Un bigram usa solo l'ultimo token: W ha forma (V, V)." },
        ],
      },
    ],
  },
  {
    id: "embedding",
    title: "3. Embedding: dai numeri ai vettori",
    children: [
      {
        id: "embedding.token",
        title: "Token embedding",
        focus: "tok_emb",
        code: "model.init",
        highlight: ["token_embedding_table"],
        weights: [{ label: "token_embedding_table.weight", prefix: "token_embedding_table." }],
        text: `<p>Ogni token id diventa un vettore di <b>{{n_embd}}</b> numeri: è la riga corrispondente di una tabella
          {{vocab_size}} × {{n_embd}} che il training impara.</p>
          <p>È lo stesso concetto di embedding visto nel RAG: vettori in cui caratteri "simili" finiscono vicini.
          Qui però lo spazio è piccolo ({{n_embd}} dimensioni contro le 1536 di <code>text-embedding-3-small</code>).</p>
          <p>Nella scena ogni cella colorata è un peso: arancione positivo, blu negativo, scuro vicino a zero.</p>`,
        math: [
          { tex: String.raw`\text{tok\_emb} = E_{\text{tok}}[x] \in \mathbb{R}^{B\times T\times C},\quad E_{\text{tok}}\in\mathbb{R}^{V\times C}`, note: "C = n_embd = {{n_embd}}. Un lookup equivale a moltiplicare un vettore one-hot per la matrice." },
        ],
      },
      {
        id: "embedding.position",
        title: "Position embedding",
        focus: "pos_emb",
        code: "model.forward",
        highlight: ["pos = torch.arange", "pos_emb ="],
        weights: [{ label: "position_embedding_table.weight", prefix: "position_embedding_table." }],
        text: `<p>L'attention, da sola, non sa in che ordine arrivano i token: "il cane morde l'uomo" e
          "l'uomo morde il cane" le sembrerebbero uguali. Per questo aggiungiamo un secondo vettore appreso
          che dice <b>in quale posizione</b> si trova il token (da 0 a {{block_size_minus_1}}).</p>`,
        math: [
          { tex: String.raw`\text{pos\_emb} = E_{\text{pos}}[0..T-1],\quad E_{\text{pos}}\in\mathbb{R}^{\text{block\_size}\times C}`, note: "Il paper del 2017 usava seni e coseni fissi; GPT (e questo codice) li fa imparare." },
        ],
      },
      {
        id: "embedding.sum",
        title: "Somma: il residual stream",
        focus: "embed_add",
        code: "model.forward",
        highlight: ["x = tok_emb + pos_emb"],
        text: `<p>I due vettori si sommano: il risultato <code>x</code>, di forma (B, T, C), contiene sia "che cosa" sia "dove".
          Da qui in poi <code>x</code> scorre lungo tutto il modello come un nastro trasportatore
          (il <i>residual stream</i>): ogni blocco legge da lì e ci aggiunge le sue correzioni.</p>`,
        math: [
          { tex: String.raw`x^{(0)} = \text{tok\_emb} + \text{pos\_emb}`, note: "La somma con un tensore (T, C) avviene per broadcasting su tutto il batch." },
        ],
      },
    ],
  },
  {
    id: "trick",
    title: "4. Il trucco matematico",
    children: [
      {
        id: "trick.loop",
        title: "v1: la media del passato con due cicli",
        focus: "blocks.0.heads.0.pattern",
        code: "trick.loop",
        text: `<p>Il modo più semplice per far comunicare i token: ogni token prende la <b>media</b> di sé stesso
          e di tutti i token precedenti. Mai quelli successivi: il futuro è proprio ciò che dobbiamo prevedere.</p>
          <p>È una comunicazione rozza (tutti pesano uguale), ma è il punto di partenza.</p>`,
        math: [
          { tex: String.raw`\bar{x}_t = \frac{1}{t+1}\sum_{i=0}^{t} x_i`, note: "Nel codice si chiama xbow: bag of words." },
        ],
      },
      {
        id: "trick.matmul",
        title: "v2: la stessa media con una matrice",
        focus: "blocks.0.heads.0.pattern",
        code: "trick.matmul",
        text: `<p>La stessa media si ottiene in un colpo solo moltiplicando per una matrice triangolare inferiore
          con le righe normalizzate. La forma a triangolo è quella che vedi nei pattern accanto a ogni head:
          sopra la diagonale c'è sempre zero.</p>`,
        math: [
          { tex: String.raw`W = \begin{pmatrix}1&0&0\\ \tfrac12&\tfrac12&0\\ \tfrac13&\tfrac13&\tfrac13\end{pmatrix},\qquad \bar{X} = W X`, note: "Esempio con T = 3." },
        ],
      },
      {
        id: "trick.softmax",
        title: "v3: maschera e softmax",
        focus: "blocks.0.heads.0.pattern",
        code: "trick.softmax",
        text: `<p>Terza versione: partiamo da punteggi tutti a zero, mettiamo <code>-inf</code> sopra la diagonale e
          applichiamo la softmax. Il risultato è di nuovo la media uniforme.</p>
          <p>Perché complicarsi la vita? Perché ora basta sostituire gli zeri con punteggi <b>calcolati dai dati</b>
          e otteniamo l'attention.</p>`,
        math: [
          { tex: String.raw`\mathrm{softmax}(z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}},\qquad e^{-\infty}=0`, note: "La softmax trasforma punteggi qualsiasi in pesi positivi che sommano a 1." },
        ],
      },
    ],
  },
  {
    id: "head",
    title: "5. Self-attention: una head",
    children: [
      {
        id: "head.qkv",
        title: "Query, Key, Value",
        focus: "blocks.0.heads.0",
        code: "head",
        highlight: ["self.key =", "self.query =", "self.value ="],
        weights: [
          { label: "query.weight", prefix: "blocks.0.sa.heads.0.query." },
          { label: "key.weight", prefix: "blocks.0.sa.heads.0.key." },
          { label: "value.weight", prefix: "blocks.0.sa.heads.0.value." },
          { label: "tril (maschera)", prefix: "blocks.0.sa.heads.0.tril", buffer: true },
        ],
        weightsNote: "Pesi di UNA head. Il modello ne ha {{n_head}} per blocco × {{n_layer}} blocchi.",
        text: `<p>Ogni token produce tre vettori di <b>{{head_size}}</b> numeri con tre matrici apprese:</p>
          <ul><li><b>Query</b>: "che cosa sto cercando?"</li>
          <li><b>Key</b>: "che cosa contengo?"</li>
          <li><b>Value</b>: "che cosa comunico se mi scegli?"</li></ul>
          <p>Analogia: in biblioteca la query è la tua domanda, le key sono le etichette sul dorso dei libri,
          i value sono i contenuti. Nella scena le tre matrici sono le "carte" sovrapposte di ogni head.</p>`,
        math: [
          { tex: String.raw`Q = xW_Q,\; K = xW_K,\; V = xW_V,\qquad W_{Q,K,V}\in\mathbb{R}^{C\times d_k}`, note: "d_k = head_size = n_embd / n_head = {{head_size}}. PyTorch salva i pesi trasposti: forma ({{head_size}}, {{n_embd}})." },
        ],
      },
      {
        id: "head.scores",
        title: "Punteggi e scaling",
        focus: "blocks.0.heads.0",
        code: "head",
        highlight: ["wei = q @ k.transpose"],
        text: `<p>Il prodotto scalare tra la query di un token e la key di un altro misura quanto sono "affini".
          Calcoliamo tutte le coppie insieme con una moltiplicazione di matrici: otteniamo una tabella T × T.</p>
          <p>Poi dividiamo per √{{head_size}}. Senza questa divisione i punteggi crescono con la dimensione della head,
          la softmax diventa "appuntita" e ogni token finisce per ascoltarne uno solo.</p>
          <p><b>Bug corretto</b>: il notebook dividiva per √n_embd (√{{n_embd}}) invece che per √head_size.</p>`,
        math: [
          { tex: String.raw`S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{T\times T}`, note: "Se q e k hanno varianza 1, q·k ha varianza d_k: dividere per √d_k la riporta a 1." },
        ],
      },
      {
        id: "head.mask",
        title: "Maschera causale e softmax",
        focus: "blocks.0.heads.0.pattern",
        code: "head",
        highlight: ["masked_fill", "F.softmax", "register_buffer"],
        weights: [{ label: "tril (maschera)", prefix: "blocks.0.sa.heads.0.tril", buffer: true }],
        text: `<p>Mettiamo <code>-inf</code> dove un token guarderebbe il futuro e applichiamo la softmax: ogni riga
          diventa una distribuzione di pesi che somma a 1.</p>
          <p>La maschera <code>tril</code> <b>non viene appresa</b>: è un triangolo di 1 fisso, salvato come <i>buffer</i>.
          Il pattern colorato nella scena invece è calcolato sul prompt d'esempio dal modello addestrato:
          righe = chi guarda, colonne = chi viene guardato, la zona tratteggiata è il futuro vietato.</p>`,
        math: [
          { tex: String.raw`A = \mathrm{softmax}\big(S + M\big),\quad M_{ij} = \begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}`, note: "Questo è un blocco «decoder». In un encoder (BERT) la maschera non c'è e tutti vedono tutti." },
        ],
      },
      {
        id: "head.values",
        title: "Media pesata dei value",
        focus: "blocks.0.heads.0",
        code: "head",
        highlight: ["return wei @ v", "v = self.value"],
        text: `<p>L'uscita di ogni token è la media dei value di tutti i token che può vedere, pesata con l'attention.
          Se il token "r" di "thr" assegna molto peso a "t" e "h", riceve soprattutto le loro informazioni.</p>`,
        math: [
          { tex: String.raw`\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big)V`, note: "La formula del paper «Attention Is All You Need» (2017), con la maschera causale." },
        ],
      },
    ],
  },
  {
    id: "mha",
    title: "6. Multi-head attention",
    children: [
      {
        id: "mha.parallel",
        title: "{{n_head}} head in parallelo",
        focus: "blocks.0.heads",
        code: "mha",
        highlight: ["torch.cat"],
        weights: [{ label: "{{n_head}} head (Q, K, V)", prefix: "blocks.0.sa.heads." }],
        text: `<p>Invece di una head grande ne usiamo <b>{{n_head}} piccole</b> da {{head_size}} dimensioni, in parallelo.
          Ognuna è libera di specializzarsi: una può seguire le vocali, un'altra l'inizio della parola, e così via.</p>
          <p>Le {{n_head}} uscite vengono concatenate: {{n_head}} × {{head_size}} = {{n_embd}}, di nuovo la dimensione C.</p>`,
        math: [
          { tex: String.raw`\mathrm{MultiHead}(x) = \big[\mathrm{head}_1;\dots;\mathrm{head}_h\big]\,W_O + b_O`, note: "h = n_head = {{n_head}}." },
        ],
      },
      {
        id: "mha.proj",
        title: "Proiezione di uscita",
        focus: "blocks.0.proj",
        code: "mha",
        highlight: ["self.proj", "self.dropout(self.proj"],
        weights: [{ label: "proj (weight + bias)", prefix: "blocks.0.sa.proj." }],
        text: `<p>Una matrice {{n_embd}} × {{n_embd}} rimescola gli "appunti" delle head prima di rimetterli nel residual stream.
          Il dropout (qui {{dropout}}) spegnerebbe a caso alcune attivazioni durante il training: <b>non ha pesi</b>.</p>`,
        math: [{ tex: String.raw`W_O\in\mathbb{R}^{C\times C},\; b_O\in\mathbb{R}^{C}` }],
      },
    ],
  },
  {
    id: "ffwd",
    title: "7. Feed-forward",
    children: [
      {
        id: "ffwd.mlp",
        title: "Ogni token elabora da solo",
        focus: "blocks.0.ffwd",
        code: "ffwd",
        weights: [
          { label: "net.0 (espansione)", prefix: "blocks.0.ffwd.net.0." },
          { label: "net.2 (compressione)", prefix: "blocks.0.ffwd.net.2." },
        ],
        text: `<p>Dopo aver comunicato, ogni token "ragiona" per conto suo: una piccola rete a due strati
          allarga il vettore da {{n_embd}} a <b>{{ffw}}</b> numeri, applica la ReLU e torna a {{n_embd}}.</p>
          <p>Nella scena si vede subito: queste due matrici sono le più grandi del blocco. Contengono circa due terzi
          dei suoi pesi.</p>
          <p>Analogia: l'attention è la riunione in cui ci si scambia informazioni, il feed-forward è il lavoro alla scrivania.</p>`,
        math: [
          { tex: String.raw`\mathrm{FFN}(x) = \max(0,\, xW_1 + b_1)\,W_2 + b_2`, note: "W₁ ∈ ℝ^{C×4C}, W₂ ∈ ℝ^{4C×C}. Il fattore 4 viene dal paper del 2017." },
        ],
      },
    ],
  },
  {
    id: "block",
    title: "8. Il blocco transformer",
    children: [
      {
        id: "block.residual",
        title: "Connessioni residue",
        focus: "blocks.0",
        code: "block",
        highlight: ["x = x + self.sa", "x = x + self.ffwd"],
        text: `<p>Ogni sotto-strato non sostituisce <code>x</code>: gli <b>aggiunge</b> una correzione (<code>x = x + ...</code>).
          Gli archi sopra il blocco nella scena sono queste "scorciatoie".</p>
          <p>Servono al training: il gradiente ha una strada diretta fino ai primi strati e una pila profonda resta addestrabile.</p>`,
        math: [
          { tex: String.raw`x \leftarrow x + \mathrm{MHA}(\mathrm{LN}_1(x)),\qquad x \leftarrow x + \mathrm{FFN}(\mathrm{LN}_2(x))` },
        ],
      },
      {
        id: "block.layernorm",
        title: "LayerNorm",
        focus: "blocks.0.ln1",
        code: "block",
        highlight: ["self.ln1", "self.ln2"],
        weights: [
          { label: "ln1 (γ, β)", prefix: "blocks.0.ln1." },
          { label: "ln2 (γ, β)", prefix: "blocks.0.ln2." },
        ],
        text: `<p>Prima di attention e feed-forward ogni vettore viene normalizzato: media 0 e varianza 1 sulle sue
          {{n_embd}} componenti. Poi due vettori appresi, γ (scala) e β (spostamento), lo riadattano.</p>
          <p>Qui la normalizzazione è <i>prima</i> di ogni sotto-strato (pre-norm), come nei GPT moderni.
          Nel paper del 2017 era dopo.</p>`,
        math: [
          { tex: String.raw`\mathrm{LN}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}} + \beta`, note: "μ e σ² sono calcolati sulle C feature di ogni singolo token." },
        ],
      },
      {
        id: "block.stack",
        title: "{{n_layer}} blocchi impilati",
        focus: "blocks",
        code: "model.init",
        highlight: ["self.blocks"],
        weights: [
          { label: "blocco 0", prefix: "blocks.0." },
          { label: "blocco 1", prefix: "blocks.1." },
          { label: "blocco 2", prefix: "blocks.2." },
          { label: "blocco 3", prefix: "blocks.3." },
        ],
        text: `<p>Il blocco si ripete <b>{{n_layer}} volte</b>, ognuno con pesi propri. I primi blocchi tendono a cogliere
          relazioni semplici (lettere vicine), gli ultimi combinazioni più astratte.</p>
          <p>GPT-3 ne ha 96: l'architettura è la stessa, cambiano le dimensioni.</p>`,
      },
    ],
  },
  {
    id: "output",
    title: "9. Dall'ultimo vettore al prossimo token",
    children: [
      {
        id: "output.lmhead",
        title: "LayerNorm finale e lm_head",
        focus: "lm_head",
        code: "model.forward",
        highlight: ["x = self.ln_f", "logits = self.lm_head"],
        weights: [
          { label: "ln_f (γ, β)", prefix: "ln_f." },
          { label: "lm_head (weight + bias)", prefix: "lm_head." },
        ],
        text: `<p>Dopo l'ultimo blocco, una LayerNorm finale e una matrice {{vocab_size}} × {{n_embd}} trasformano ogni vettore
          in <b>{{vocab_size}} punteggi</b> (i <i>logits</i>), uno per carattere del vocabolario.</p>`,
        math: [{ tex: String.raw`\text{logits} = \mathrm{LN}_f(x)\,W_{\text{head}}^\top + b \in \mathbb{R}^{B\times T\times V}` }],
      },
      {
        id: "output.softmax",
        title: "Softmax: le probabilità",
        focus: "logits",
        code: "generate",
        highlight: ["F.softmax", "logits = logits[:, -1, :]"],
        text: `<p>La softmax trasforma i punteggi in probabilità. Le barre nella scena sono le probabilità vere che il
          modello addestrato assegna al carattere dopo il prompt <code>{{prompt}}</code>.</p>`,
        math: [{ tex: String.raw`p_i = \frac{e^{\ell_i/\tau}}{\sum_j e^{\ell_j/\tau}}`, note: "τ è la temperature: τ < 1 rende le scelte più sicure, τ > 1 più creative." }],
      },
    ],
  },
  {
    id: "training",
    title: "10. Training",
    children: [
      {
        id: "training.loss",
        title: "La loss: quanto sbaglia il modello",
        focus: "logits",
        code: "loss",
        text: `<p>La cross-entropy misura quanta probabilità il modello ha dato al carattere giusto: più è bassa, meglio è.</p>
          <p>Un modello appena nato tira a indovinare tra {{vocab_size}} caratteri, quindi parte da circa
          ln({{vocab_size}}) ≈ <b>{{ln_vocab}}</b>. Dopo {{train_steps}} step il nostro arriva a <b>{{train_loss}}</b> sul train e <b>{{val_loss}}</b> sulla validation.</p>`,
        math: [{ tex: String.raw`\mathcal{L} = -\frac{1}{BT}\sum_{b,t}\log p_\theta\big(y_{b,t}\mid x_{b,\le t}\big)` }],
      },
      {
        id: "training.loop",
        title: "Il ciclo di training",
        focus: "overview",
        code: "train",
        highlight: ["loss.backward()", "optimizer.step()", "AdamW"],
        text: `<p>Quattro passi ripetuti 5000 volte: prendi un batch, calcola la loss, calcola il gradiente di ognuno dei
          <b>{{total}}</b> pesi (<code>backward</code>), sposta ogni peso un po' nella direzione che abbassa la loss
          (<code>step</code>).</p>
          <p>Il training è l'unico momento in cui i pesi cambiano. Quando usiamo il modello restano fissi.</p>`,
        math: [{ tex: String.raw`\theta \leftarrow \theta - \eta\,\hat{m}/(\sqrt{\hat{v}}+\epsilon)`, note: "Aggiornamento semplificato di Adam (η = learning rate = 1e-3). AdamW aggiunge il weight decay." }],
      },
    ],
  },
  {
    id: "generation",
    title: "11. Generazione",
    children: [
      {
        id: "generation.loop",
        title: "Un token alla volta",
        focus: "prediction",
        code: "generate",
        highlight: ["idx_cond =", "torch.multinomial", "torch.cat"],
        text: `<p>Per generare testo il modello prevede un carattere, lo estrae a sorte secondo le probabilità,
          lo aggiunge in fondo e ricomincia. Se il testo supera {{block_size}} caratteri, guarda solo gli ultimi {{block_size}}.</p>
          <p>È lo stesso "token by token" che vedevamo arrivare in streaming nella chat React della L16.</p>`,
        math: [{ tex: String.raw`x_{t+1} \sim p_\theta(\,\cdot \mid x_{t-\text{block\_size}+1..t})` }],
      },
    ],
  },
  {
    id: "weights",
    title: "12. I pesi del modello",
    children: [
      {
        id: "weights.all",
        title: "Dove sono i {{total}} pesi",
        focus: "overview",
        code: "weights.edit",
        weightsSummary: true,
        text: `<p>I pesi (o parametri) sono i numeri che il training impara: ogni cella colorata della scena è uno di loro.
          La tabella li conta per componente. In un GPT reale il conto arriva a miliardi, ma i "cassetti" sono gli stessi.</p>
          <p><b>Non appresi</b>: la maschera <code>tril</code> (un triangolo fisso per ogni head), il dropout (nessun peso)
          e gli iperparametri come n_embd o block_size, che scegliamo noi.</p>
          <p>Il codice a destra cambia un singolo peso e lo ripristina: è l'esperimento <code>06_edit_a_weight.py</code>.</p>`,
        math: [{ tex: String.raw`N = \underbrace{VC}_{\text{tok}} + \underbrace{T_{\max}C}_{\text{pos}} + L\,\underbrace{(12C^2+10C)}_{\text{blocco}} + \underbrace{2C}_{\mathrm{LN}_f} + \underbrace{CV+V}_{\text{head}}`, note: "Con V={{vocab_size}}, C={{n_embd}}, T_max={{block_size}}, L={{n_layer}}: N = {{total}}." }],
      },
    ],
  },
];

/**
 * Pannello "Fonti" del tab: da dove vengono il codice, l'architettura e le scelte di training.
 * Solo fonti pubbliche e verificabili; i numeri del nostro modello vengono da data/model.json.
 */
export const SOURCES = [
  {
    title: "Il codice del corso",
    items: [
      { title: "Andrej Karpathy, «Let's build GPT: from scratch, in code, spelled out» (video, 2023)", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", note: "Il video da cui nasce il pacchetto gpt/: bigram, trucco matematico, attention, blocchi." },
      { title: "karpathy/ng-video-lecture (GitHub)", url: "https://github.com/karpathy/ng-video-lecture", note: "Il notebook e lo script originali del video; i bug corretti sono elencati nel README del progetto." },
      { title: "karpathy/nanoGPT (GitHub)", url: "https://github.com/karpathy/nanoGPT", note: "La versione «grande» dello stesso codice, usata per riprodurre GPT-2." },
      { title: "Tiny Shakespeare (karpathy/char-rnn)", url: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", note: "Il file di testo del training (circa 1,1 milioni di caratteri)." },
    ],
  },
  {
    title: "Articoli scientifici",
    items: [
      { title: "Vaswani et al., «Attention Is All You Need» (2017)", url: "https://arxiv.org/abs/1706.03762", note: "Il transformer: scaled dot-product attention, multi-head, feed-forward 4×, connessioni residue." },
      { title: "Radford et al., «Language Models are Unsupervised Multitask Learners» (GPT-2, 2019)", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf", note: "Il transformer solo-decoder con LayerNorm prima di ogni sotto-strato (pre-norm), come il nostro." },
      { title: "He et al., «Deep Residual Learning for Image Recognition» (2015)", url: "https://arxiv.org/abs/1512.03385", note: "Le connessioni residue: x = x + f(x)." },
      { title: "Ba, Kiros, Hinton, «Layer Normalization» (2016)", url: "https://arxiv.org/abs/1607.06450", note: "La normalizzazione sulle feature di ogni token, con γ e β appresi." },
      { title: "Loshchilov, Hutter, «Decoupled Weight Decay Regularization» (AdamW, 2017)", url: "https://arxiv.org/abs/1711.05101", note: "L'ottimizzatore del training (torch.optim.AdamW)." },
    ],
  },
  {
    title: "Documentazione",
    items: [
      { title: "PyTorch: torch.nn", url: "https://pytorch.org/docs/stable/nn.html", note: "Embedding, Linear, LayerNorm, Dropout: i mattoni usati in gpt/." },
      { title: "Spring AI Reference", url: "https://docs.spring.io/spring-ai/reference/", note: "Advisor, ChatClient, tool calling e vector store della vista Contesto e del tab RAG (lezioni L16-L18)." },
    ],
  },
];
