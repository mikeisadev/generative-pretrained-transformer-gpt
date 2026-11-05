# Prompt — Transformer didattico (Lezione L20)

## 1. Ruolo

Agisci come senior software engineer con più di 30 anni di esperienza e come divulgatore tecnico. Lavori per un docente che deve spiegare il transformer a studenti alle prime armi.

## 2. Contesto

### Materiale nella repository

| File | Contenuto |
|---|---|
| `gpt_dev.py` | Export Python del notebook di Andrej Karpathy. Contiene ancora sintassi da notebook (es. `!wget`). |
| `gpt_dev.ipynb` | Lo stesso notebook in formato Jupyter, utile per seguire l'ordine originale delle celle. |

### Fonti

- Video: [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — leggi la descrizione del video: riassume i passaggi e i riferimenti principali.
- Notebook originale su Colab: <https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing>. Se non riesci ad accederci, i file locali sono equivalenti.

### Corso

- Lezione corrente: **L20 — Gli LLM da vicino: anatomia del transformer** (collegata a questa sessione).
- Lezioni precedenti, allegate come contesto:
  - **L16, L17, L18**: costruzione di un RAG.
  - **L19**: creazione di un LLM wiki.

## 3. Obiettivo

Riorganizzare il codice per una dimostrazione puramente didattica e creare due **medium**, cioè i mezzi con cui trasferirò i concetti agli studenti:

1. una web app 3D interattiva in Three.js;
2. la lezione L20 in Markdown.

## 4. Principi guida (validi per tutte le fasi)

- **Complessità nel codice, semplicità nelle spiegazioni.** Codice, dipendenze e commenti possono mostrare tutta la complessità reale. Le spiegazioni per gli studenti devono essere semplici, scorrevoli, senza fronzoli né ripetizioni.
- **Semplice non vuol dire superficiale.** La priorità è che ogni step sia spiegato in modo esauriente.
- **Fedeltà al codice.** Struttura, nomi, dimensioni e iperparametri mostrati nei medium devono corrispondere al codice riorganizzato.
- **Niente invenzioni.** Verifica notizie e dati sulle fonti. Se qualcosa non è verificabile, dichiaralo.
- **Lingua:** italiano. Mantieni in inglese i termini tecnici standard (token, embedding, attention, ecc.).

## 5. Attività

Eseguile in quest'ordine: i medium dipendono dal codice riorganizzato.

### Fase 1 — Ambiente, codice e test

#### 1.1 Ambiente virtuale

- Crea un ambiente virtuale autonomo con tutte le dipendenze necessarie e versioni fissate (es. `requirements.txt`).
- Deve funzionare su macOS (CPU e Apple Silicon/MPS) e su Linux con o senza GPU CUDA, rilevando il device in automatico.
- Il download del dataset (Tiny Shakespeare) deve essere automatico, senza comandi da notebook come `!wget`.

#### 1.2 Riorganizzazione del codice

- Struttura modulare che segue il percorso didattico del notebook:
  dati e tokenizzazione → modello bigram → "trucco matematico" della self-attention → singola head → multi-head attention → feed-forward → blocco transformer (residual + LayerNorm) → GPT completo → training → generazione.
- Separa gli esperimenti esplorativi dal modello finale ed elimina il codice duplicato.
- Commenti dettagliati: per ogni passaggio spiega cosa fa, perché serve e la forma dei tensori (`B, T, C`).

#### 1.3 Bug e test

- Individua e correggi bug e problemi. Verifica ogni correzione con test automatici scritti da te (es. `pytest`).
- Punti da controllare (lista non esaustiva):
  - `!wget` non è Python valido: il file `.py` non si avvia così com'è.
  - In `Head.forward` lo scaling dell'attention usa `C**-0.5` (con `C = n_embd`) invece di `head_size**-0.5`.
  - La classe del modello finale si chiama ancora `BigramLanguageModel`; `FeedFoward` contiene un refuso.
- Test minimi suggeriti:
  - forma dei tensori in uscita da ogni modulo;
  - maschera causale: un token non deve vedere quelli futuri;
  - loss iniziale ≈ ln(`vocab_size`);
  - lunghezza corretta dell'output di `generate`;
  - il modello riesce a fare overfitting su un batch piccolo.
- Per ogni bug riporta: causa, correzione e test che la verifica.

#### 1.4 Documentazione per il docente

- `README.md` che parte con un **avvio rapido**: pochi comandi copiabili per creare l'ambiente, lanciare training e generazione e avviare la web app.
- Dopo l'avvio rapido: struttura della repository e ruolo di ogni modulo.
- È per me, ma deve essere immediata: niente dettagli superflui.

### Fase 2 — Medium 1: web app 3D (Three.js)

#### Scena 3D

- Rappresenta l'esatta architettura del codice Python e ne segue gli iperparametri (oggi: `n_layer=4`, `n_head=4`, `n_embd=64`, `block_size=32`).
- È animata: mostra il flusso dei dati (token → embedding → blocchi transformer → logits → prossimo token).
- È navigabile liberamente (rotazione, zoom, pan).
- Quando seleziono uno step, la camera si sposta in modo fluido sul componente corrispondente e lo evidenzia.

#### Menu laterale sinistro (collassabile)

- Fisarmoniche con punti principali e sottopunti, ordinati secondo il flusso dei dati.
- Per ogni punto: spiegazione semplificata ed esauriente.
- Per ogni punto: riquadro **"Approfondimento"** con le formule matematiche in LaTeX (es. rese con KaTeX).

#### Finestra fluttuante a destra

- Mostra il frammento di codice Python dello step selezionato, preso dal codice riorganizzato, con evidenziazione della sintassi.
- È sincronizzata con il menu e con la scena 3D.

#### Pesi (parametri)

In questo modello i pesi ci sono: sono i parametri che il training impara. Integrali così:

- **Nel menu**, per ogni componente: quali pesi contiene (tabelle di embedding, matrici Q/K/V, proiezione, strati feed-forward, γ e β della LayerNorm, `lm_head`), con forma e numero di parametri. Indica anche il totale del modello.
- Distingui ciò che viene appreso da ciò che non lo è (maschera causale `tril`, dropout).
- **Nella scena 3D**: rendi visibili i pesi dove ha senso (es. matrici con dimensioni proporzionali ed etichette con la forma).

#### Requisiti tecnici

- Avviabile in locale con un solo comando, documentato nel README.

### Fase 3 — Medium 2: Lezione L20 (Markdown)

**Dove scriverla:** se hai accesso alla lezione L20 collegata a questa sessione, scrivi direttamente lì. Altrimenti consegna un file Markdown nella repository.

#### Struttura

1. **Introduzione semplice.**
2. **Cos'è un transformer.**
3. **Prima e dopo il transformer:** come si lavorava prima (es. RNN/LSTM) e come è stato usato nei primissimi modelli (traduzione automatica, primi GPT e BERT).
4. **Il paper "Attention Is All You Need" (2017):** cosa ha introdotto e perché è stato una svolta.
5. **Le parti del transformer**, descritte sommariamente.
6. **I pesi:**
   - cosa sono, collegandoli a questo progetto (es. quanti parametri ha il nostro modello e dove si trovano);
   - la notizia di **GPT-6 Astra**: ricercatori del MIT CSAIL lo hanno usato per modificare direttamente i pesi di un modello open (Qwen3.5-0.8B), migliorandone le prestazioni in matematica ("weight-space programming"). Verifica i dettagli sulle fonti (es. [VentureBeat](https://venturebeat.com/technology/like-gene-editing-for-neural-networks-mit-researchers-used-gpt-6-astra-to-edit-qwens-open-weights-directly-boosting-math-performance)) e riportali con le dovute cautele (i risultati non sono ancora stati replicati in modo indipendente);
   - un esempio semplicissimo di come si può modificare un peso, possibilmente sul nostro modello: cambiare un valore e osservare l'effetto sull'output.
7. **Esplora il transformer con questa repository:** il codice Python, da approfondire anche con l'aiuto di un LLM, e la web app interattiva.

#### Collegamenti con L16–L19

Inseriscili solo dove sono naturali (es. gli embedding già visti nel RAG, l'LLM wiki), senza appesantire la lezione.

## 6. Consegna finale

Al termine, riassumi:

- file creati o modificati;
- bug trovati e corretti, con i test relativi e il loro esito;
- comandi per avviare training, generazione e web app;
- eventuali punti aperti o limiti.
