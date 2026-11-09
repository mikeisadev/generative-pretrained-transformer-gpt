# GPT didattico — Lezione L20 "Anatomia del transformer"

Il GPT del video [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) di Andrej Karpathy, riorganizzato in moduli che seguono il percorso della lezione, con test automatici e una web app 3D interattiva.

## Avvio rapido

Requisiti: Python 3.10+ (verificato con 3.13). Funziona su macOS (CPU e Apple Silicon) e su Linux, con o senza GPU NVIDIA: il device viene scelto da solo.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python train.py                    # ~3 minuti su CPU, salva checkpoints/gpt.pt
python generate.py --prompt "ROMEO:" --max-new-tokens 300
python scripts/serve_webapp.py     # web app 3D: poi apri http://localhost:8000
python scripts/record_training.py  # rifà il training registrandolo per i grafici della web app (~4 min)
```

Prova veloce del training: `python train.py --max-iters 500`. Test: `pytest` e `node --test webapp/tests/*.test.mjs`.

Il dataset Tiny Shakespeare si scarica da solo al primo avvio (in `data/`). La web app funziona anche offline: le librerie sono in `webapp/vendor/`.

## Struttura

```
gpt/                    il modello, un file per ogni step della lezione
  config.py             iperparametri (GPTConfig, TrainConfig)
  device.py             scelta automatica CUDA → MPS → CPU
  data.py               1. download, tokenizer a caratteri, batch
  bigram.py             2. modello bigram (baseline)
  didactic/             3. "trucco matematico" e LayerNorm a mano (solo esplorazione)
  attention.py          4-5. Head e MultiHeadAttention
  feedforward.py        6. FeedForward
  block.py              7. Block: residui + LayerNorm
  model.py              8. GPTLanguageModel
  training.py           9. training, stima della loss, checkpoint (scritto in modo atomico: un'interruzione non lo rovina)
  generation.py         10. generazione token per token (con temperature)
  loss.py               cross-entropy condivisa da bigram e GPT
  weights.py            inventario dei pesi e modifica di un singolo peso
  export.py             genera webapp/data/model.json dal codice e dal checkpoint; capture_attention (anche per attention nuove)
  recording.py          registra la storia di un training (loss, metriche, snapshot, note), anche di un'architettura modificata
  runs.py               run esportabili per il Laboratorio pesi: pesi, storia del training, run.json con le note
  benchmark.py          benchmark deterministico e confrontabile (protocollo finestre-contigue-v1)
experiments/            script da proiettare in aula, nell'ordine del video
  01_data_and_tokenization.py … 06_edit_a_weight.py
tests/                  pytest: forme, maschera causale, loss, generazione, overfitting, bug, registrazione (anche di
                        architetture modificate e da riga di comando), run, benchmark, dati della web app
webapp/                 web app 3D a sei tab (Three.js, KaTeX, highlight.js inclusi; vedi webapp/ARCHITECTURE.md)
  index.html, js/app.js il guscio e il registro dei tab
  js/core/              nucleo: scena, navigazione libera, zoom semantico, ricerca, glossario, grafici, codice,
                        representations.js (selettore Architettura · Rete neurale · Cervello), matrixNumbers.js
                        (matrici con i numeri veri), ui/tensorTable.js e ui/histogram.js (numeri dei tensori)
  js/core/neuralNet/    rete neurale orizzontale con profondità, guidata da una specifica di soli dati
  js/core/brain/        cervello 3D (regioni, percorsi, piani di taglio, microcircuiti di neuroni) e pannello delle analogie
  js/engine/            motore del GPT in JavaScript (esegue i pesi veri nel browser)
  js/shared/rag/        libreria RAG didattica (chunking, embedding, HNSW, vector store)
  js/shared/weights/    pesi e run: formati, safetensors, statistiche, confronto, salvataggio nel browser
  js/tabs/<id>/         un tab per cartella (transformer, lab, rag, diffusion, audio, modern-llm), con
                        network.js (la sua rete neurale) e brain.js (il suo cervello), moduli di soli dati
  js/tabs/lab/          Laboratorio pesi: scena come il codice Python, run, azioni (actions/), confronto e benchmark (analysis/)
  js/tabs/modern-llm/full/  Architettura LLM moderni completa (DeepSeek-V4.1-Flash, Kimi K3.1 Pro, gpt-oss-120b)
  data/                 dati esportati dal Python: model.json, pesi, storia del training, benchmark del corso
  tests/                test JS (node --test), pagine di prova dei moduli, screenshot.mjs
scripts/
  serve_webapp.py       esporta i dati del modello e avvia la web app
  record_training.py    rifà il training registrandone la storia (webapp/data/training/, o una cartella tua)
  export_run.py         esporta un checkpoint come run da caricare nel Laboratorio pesi (runs/<nome>/)
  benchmark.py          misura un checkpoint o un run, oppure confronta due benchmark
  export_benchmark_data.py  rigenera (o controlla) webapp/data/benchmark/
train.py, generate.py   punti di ingresso
reference/              notebook ed export originali di Karpathy, non modificati
```

Gli esperimenti si lanciano dalla radice, per esempio `python experiments/03_attention_math_trick.py`. Il numero 06 modifica un solo peso del modello addestrato e mostra l'effetto sul testo generato (serve prima `python train.py`).

## Web app

```bash
python scripts/serve_webapp.py                 # riesporta i dati dal checkpoint e apre http://localhost:8000
python scripts/serve_webapp.py --no-export --port 8765   # usa i dati già presenti
python scripts/record_training.py              # rifà il training registrandolo (loss, metriche, snapshot) per i grafici
```

Il server ascolta solo su `localhost`. La web app non ha bisogno di internet né di installare nulla
(Three.js, KaTeX e highlight.js sono in `webapp/vendor/`). Un test (`tests/test_webapp_data.py`)
fallisce se `webapp/data/` non è allineato al codice: dopo aver modificato il pacchetto `gpt`, rilancia lo script.

### I tab

| Tab | Cosa mostra |
|---|---|
| **Transformer** | Il GPT del corso in 3D: ogni cella colorata è un peso vero (arancione positivo, blu negativo). Viste: *Architettura*, *Pesi* (dove stanno i 209.729 parametri), *Pesi numerici* (i numeri veri dentro le matrici, anche durante il training), *Rete neurale* (tutti i neuroni, le connessioni e i calcoli, con un clic su un neurone; *A righe* oppure *Orizzontale con profondità*), *Contesto* (domanda → database vettoriale → system prompt e tool → richiesta all'LLM, collegata al tab RAG). Azioni *Prompt* (dalla scrittura del prompt ai token, dentro i blocchi, fino al token estratto, calcolato dal motore JS con i pesi veri) e *Training* (uno step al microscopio e il training completo registrato) |
| **Laboratorio pesi** | I pesi del GPT organizzati come il codice Python (`GPTLanguageModel` → `blocks` → `Block` → `sa.heads[h].key`…), come celle colorate e da vicino come numeri; animazioni dei calcoli con i numeri veri; i run che esporti dal Python (anche con l'architettura modificata), il confronto A/B, i grafici del training e il benchmark |
| **Database vettoriale (RAG)** | La pipeline delle lezioni L16-L18 (Spring Boot + Spring AI + pgvector): database vuoto, upload, pulizia, chunking, embedding, tabella `vector_store`, indice HNSW; poi ricerca HNSW, prompt e risposta in streaming |
| **Immagini e video** | Modelli di diffusione: processo diretto e training, generazione (text encoder, U-Net o DiT, scheduler, VAE), video con VAE 3D e patch spazio-temporali |
| **Audio e musica** | Dal suono ai numeri (STFT, scala Mel), codec neurale con RVQ, generazione di musica a token, Whisper; laboratorio con suono vero |
| **LLM moderni** | Due sottotab: la *Vista d'insieme* di cosa cambia rispetto al nostro GPT (RoPE, GQA, MoE, KV cache, FlashAttention, BPE, leggi di scala, post-training, ragionamento, agenti, coding, capacità emergenti, multimodalità, modelli 3D) e l'*Architettura completa* di un modello aperto vero, DeepSeek-V4.1-Flash (con Kimi K3.1 Pro e gpt-oss-120b a confronto) |

In ogni tab, in alto nella scena, il selettore **Architettura · Rete neurale · Cervello** mostra lo stesso
sistema in tre modi (vedi "Novità della web app" qui sotto).

### Come si usa

- **Sidebar a sinistra**: *Percorso* (gli step in ordine, con spiegazione, formule e tabella dei pesi), *Azioni* (panoramica delle animazioni, divise per categoria: prompt, training…), *Viste*, *Cervello* (le analogie con il cervello), *Fonti*. Con la tastiera: ↑ ↓ cambiano step, lo spazio mette in pausa l'azione (in tutti i tab).
- **Selettore delle rappresentazioni** in alto nella scena: *Architettura* (la scena del tab), *Rete neurale*, *Cervello*. Con la tastiera: Tab fino al selettore, poi ← →. Un passo del Percorso, un'azione o un link riportano da soli all'Architettura.
- **Zoom semantico**: rotellina per avvicinarsi o allontanarsi, oppure il selettore in alto a sinistra. *Dettaglio* mostra ogni elemento, *Blocchi* un rettangolo con titolo e descrizione per componente, *Panoramica* il diagramma a blocchi (nel Transformer un unico rettangolo con input sotto e output sopra).
- **Azioni**: barra in basso con play/pausa, passo avanti/indietro, ricomincia, velocità e chiudi. Chiudendo un'azione la scena torna com'era.
- **Grafici**: il pulsante in alto a destra apre i grafici che usano AI engineer e data scientist (loss, learning rate, norma dei gradienti, distribuzioni, attention…), che si riempiono mentre le azioni avanzano.
- **Codice**: la finestra a destra mostra il codice vero dello step (Python del pacchetto `gpt`, Java delle lezioni Spring AI); "▁" la riduce.

### Muoversi nella scena

Ogni scena è come un plastico sul tavolo: ci giri attorno, ti sposti di lato, ti avvicini a un dettaglio e torni alla vista iniziale quando vuoi. Gli stessi comandi valgono in tutti i tab; `?` (o il pulsante con il punto di domanda) li mostra tutti.

| Per… | Mouse | Trackpad | Tastiera | Touch |
|---|---|---|---|---|
| spostarti a destra, sinistra, su, giù | tasto destro (o Shift) + trascina | due dita in orizzontale, oppure modalità **Sposta** e trascina | `W` `A` `S` `D`, `←` `→` | due dita |
| ruotare attorno al punto guardato | trascina | trascina | `Q` `E` (gira), `R` `F` (inclina) | un dito |
| zoom | rotella (verso il puntatore) | pizzico o due dita in verticale | `+` `−` | pizzico |
| centrare un punto | doppio clic | doppio clic | | doppio tocco |
| tornare alla vista iniziale | pulsante ◎ "Centra" | pulsante ◎ | `C` o `Home` | pulsante ◎ |

- Tenendo premuto un tasto o una freccia del pad ci si muove in continuo, con partenza e frenata morbide; `Shift` va più veloce, `Alt` più piano. `M` scambia Ruota e Sposta (scelta ricordata).
- **Pad a schermo** (in basso a destra; sul telefono un pulsante): frecce, rotazione, zoom, "Centra", aiuto, schermo intero, foto della scena in PNG.
- **Non ci si perde**: la scena ha dei bordi morbidi; se non si vede più niente compare "Ti sei perso? Torna al centro".
- **Durante un'azione** puoi guardarti intorno: la camera diventa libera e il chip **"Segui l'azione"** la riporta sull'animazione (lo fanno anche la fine dell'azione e "⟲ Ricomincia").

### Ricerca, link diretti, glossario e progressi

- **Cerca** (`Ctrl/⌘+K` o `/`): trova passi, azioni, viste e termini in tutti i tab, anche quelli mai aperti; tollera accenti e un errore di battitura ("sofmax"). Invio apre il risultato, Maiusc+Invio copia il link.
- **Link diretti**: `#transformer/go=step:head.qkv`, `#rag/go=action:ricerca`, `#diffusion/go=view:dit`… aprono il tab già sul punto giusto; ogni passo ha "Copia link a questo passo".
- **Glossario**: 47 termini (token, embedding, softmax, HNSW, CFG, RVQ, MoE, KV cache…) con definizione semplice, un'analogia e il pulsante "Guarda nel tab".
- **Progressi**: ✓ sui passi già visti e "n/m passi visti" per ogni tab (salvati nel browser; "Azzera" li cancella).

### Test della web app

```bash
node --test webapp/tests/*.test.mjs            # motore GPT in JS, RAG, grafici, navigazione, ricerca, rete neurale, cervello, pesi, tab
NODE_PATH=$(npm root -g) node webapp/tests/screenshot.mjs http://127.0.0.1:8765/ /tmp/shots --lod --nav
```

Lo script di screenshot (Playwright con Chromium) apre tutti e sei i tab e le loro viste principali, i livelli
dello zoom (`--lod`), prova la navigazione con i tasti e "Centra" (`--nav`), usa il formato telefono con
`--mobile` (o più dimensioni con `--sizes=1600x900,390x844`) e termina con errore se la console segnala errori o
warning. L'architettura del codice è descritta in `webapp/ARCHITECTURE.md`.

## Novità della web app: rete neurale, cervello, pesi numerici, Laboratorio pesi

Cinque cose nuove, pensate per vedere **lo stesso modello da più lati**: com'è costruito, quali neuroni lavorano e
con quali numeri, da quale idea del cervello viene, come cambia quando lo modifichi tu.

### Tre modi di guardare ogni tab: Architettura · Rete neurale · Cervello

In alto nella scena di **ogni** tab c'è il selettore *Architettura · Rete neurale · Cervello*. Pensa a una città:
*Architettura* è la mappa dei quartieri (i blocchi del modello), *Rete neurale* è l'impianto elettrico casa per
casa (ogni neurone e ogni collegamento), *Cervello* è la città vera da cui gli urbanisti hanno copiato le idee.
Ogni vista ha la sua navigazione (trascina, rotella, `W A S D`, `C` per tornare alla vista iniziale, `?` per
l'aiuto); un passo del Percorso, un'azione o un link diretto riportano da soli all'Architettura, esattamente dove
eri.

**La rete neurale orizzontale con profondità.** I neuroni sono disposti da sinistra a destra nell'ordine in cui
lavorano (ingresso → strati → uscita); in verticale ci sono i neuroni di ogni strato; **in profondità** c'è la cosa su
cui lavorano i meccanismi più "intricati": i token della frase (Transformer, RAG, LLM moderni), la griglia
dell'immagine (Immagini e video), i frame nel tempo (Audio). È come un libro visto di taglio: di lato leggi lo schema
classico, girandolo vedi le pagine (i token) una dietro l'altra, con gli archi dell'attention che le collegano.
- Nella scheda in alto a sinistra: **▶ Anima** (un'onda di attivazione da sinistra a destra), **Guarda lo spessore**
  (la camera di tre quarti), **Di lato**, **Fetta** (un solo token e il suo percorso), **Collegamenti** (quanti
  disegnarne). Passa sopra un neurone per il suo valore; clic su un neurone o sul nome di una sezione per la scheda
  con che cosa fa e da chi riceve.
- Le reti lunghe (LLM moderni, Audio, Immagini e video) partono da una **parte** leggibile: ◀ ▶ scorrono le parti, *Tutta la rete* le
  mostra insieme.
- Il **badge dell'onestà** dice da dove vengono i numeri: *Attivazioni vere* (calcolate dal modello, come nel
  Transformer e nel Laboratorio), *in parte vere* (per esempio nel RAG l'embedder e il database sono veri, l'LLM è
  illustrativo), *simulazione didattica*. Cliccalo per "Scala e onestà": quanto è ridotta la rete rispetto a quella vera.

**Il cervello.** Nel pannello *Cervello* della sidebar ogni analogia ha "In comune", **"Dove non regge"**, la forza
(forte, media, debole) e le fonti; *Mostra nel modello 3D* vola sulle regioni nel cervello stilizzato (coordinate MNI
approssimate, lo dice l'etichetta). Nella vista 3D: scala **Regioni** o **Neuroni** (il microcircuito del tab,
disposto da sinistra a destra come la rete AI), cursore *Corteccia* per vederci attraverso, *Taglio* sagittale,
coronale o assiale, **▶ Anima impulsi**.

### Transformer: i numeri veri dei pesi e la rete in profondità

- **Pesi numerici** (pannello *Viste*): avvicinandoti a una matrice i colori diventano i numeri veri (float32) che usa
  il modello. Come una foto: da lontano vedi i colori, da vicino i pixel, e qui ogni pixel è il numero vero. Scegli
  la sorgente: *Pesi finali* oppure uno **step del training** (gli 11 snapshot veri, da 0 a 5000); **▶ Guarda i pesi
  nascere** è il time-lapse dal rumore iniziale ai pesi addestrati. *Interpola* mostra anche gli step intermedi, ma
  li dichiara stime (in giallo). Nella sidebar: la riga Python che crea il tensore, la tabella dei numeri (frecce per
  muoversi, clic su una cella della scena per trovarla), statistiche, istogramma e **Apri nel Laboratorio pesi**.
- **Rete neurale**: l'interruttore *A righe* (com'era: una riga per blocco) / **Orizzontale con profondità** mostra
  tutta la rete vera del GPT del corso, 4 blocchi, con tutti i token del prompt in profondità e gli archi di attention
  veri tra i token (mai verso il futuro: la maschera causale si vede). Clic su un neurone = la stessa scheda con la
  formula e i numeri veri.

### Laboratorio pesi: il tuo GPT smontato come il codice

Un banco di laboratorio: i pesi sono sistemati come le classi Python (`GPTLanguageModel` → `blocks[0]` →
`sa.heads[0].key`), con la riga di codice che crea ogni peso; da lontano i moduli, poi le celle colorate, da vicino i
numeri. Pannelli: **Modello** (l'indice dei moduli, come `print(model)`, e lo step del training), **Tensore** (numeri,
statistiche, istogramma), **Run** (il GPT del corso e i run che carichi: *Ricordali in questo browser*, *Dimentica*),
**Azioni**, **Confronto**, **Benchmark**.
- **Azioni** con i numeri veri, sulle matrici della scena: x · Wᵀ riga per colonna, q · kᵀ con la maschera e la
  softmax, att · V e il residuo, LayerNorm, feed-forward con la ReLU, logits e probabilità, **un passo di training**
  (gradiente e aggiornamento, come farebbe PyTorch), **i pesi durante il training**. Una lavagna sopra la barra
  scrive ogni conto.
- **Confronto A/B**: A è il run nella scena, B il termine di paragone; *Mostra A − B nella scena* colora la
  differenza sulle matrici, come due fotografie in cui si colora solo ciò che è cambiato. I "confronti pronti" fanno
  tutto da soli (che cosa ha cambiato il training, step 5000 contro 500, il tuo run contro il GPT del corso).
- **Benchmark**: i numeri del Python e, per i run che il motore JavaScript sa eseguire, *Calcola per A*: lo stesso
  benchmark rifatto nel browser. Il verdetto è ✓ se i due calcoli coincidono (come due bilance che segnano lo stesso
  peso), ≈ se differiscono di un'inezia ammessa, ✗ se i due calcoli non tornano.

### LLM moderni: l'architettura completa di un modello aperto

Sopra la scena due sottotab: *Vista d'insieme* (i 16 meccanismi di sempre) e **Architettura completa**; il tasto
**V** passa dall'uno all'altro. L'architettura completa è DeepSeek-V4.1-Flash (selettore per Kimi K3.1 Pro e
gpt-oss-120b) in tre livelli di zoom: *Panoramica* (dai dati del pre-training fino ai **pesi**: 510 GB, 96.085
tensori), *Blocchi* (i 40 layer raggruppati in "× N", come "ripeti 18 volte" in una ricetta; *Espandi le ripetizioni*
li srotola), *Dettaglio* (dentro un layer: attention, KV cache, router e i 384 esperti, 6 accesi per token). I
segnaposto numerati dicono dove vive ognuno dei 16 meccanismi; ogni passo della vista d'insieme ha "Dove sta in
DeepSeek-V4.1-Flash". Ciò che le fonti non confermano è scritto "da verificare".

### Link diretti utili

| Link | Che cosa apre |
|---|---|
| `#transformer/go=view:numeri.training` | i Pesi numerici dallo step 0, pronti per "▶ Guarda i pesi nascere" |
| `#transformer/go=tensor:lm_head.weight@1000` | il tensore `lm_head.weight` allo step 1000 del training |
| `#transformer/go=view:neuroni-profondita` | la Rete neurale orizzontale con profondità del GPT del corso |
| `#lab/go=view:confronto:training` | che cosa ha cambiato il training: pesi finali contro step 0, con la differenza nella scena |
| `#lab/go=action:attention` | l'animazione di q · kᵀ, maschera e softmax con i numeri veri |
| `#lab/go=view:benchmark:browser` | il benchmark ricalcolato nel browser, con il controllo ✓ / ≈ |
| `#modern-llm/go=view:full` | l'Architettura LLM moderni completa |
| `#modern-llm/go=view:full-moe` | i layer MoE di DeepSeek-V4.1-Flash e i loro esperti |
| `#rag/go=view:rete`, `#audio/go=view:cervello` | la Rete neurale del RAG, il Cervello dell'Audio (vale `view:rete` / `view:cervello` in ogni tab) |

La ricerca (`Ctrl/⌘+K`) trova tutte queste viste: prova "pesi numerici", "rete neurale audio", "ippocampo".

## Esportare un run e confrontarlo nel Laboratorio pesi

Un *run* è una cartella con i pesi di un training nel formato che la web app sa caricare:
`run.json` (nome, descrizione, data, config, albero dei moduli con le forme, note),
`weights.json` + `weights.f32.bin` (lo stesso formato di `webapp/data`) e, a richiesta,
`benchmark.json` e la storia del training (`training/trace.json` con gli snapshot). Così vedi il
"prima e dopo" di ogni miglioramento: iperparametri, pesi o architettura, anche fatti con un
assistente come Claude. Il flusso completo, per esempio per provare GELU al posto di ReLU:

```bash
# 0. il metro di riferimento: il GPT del corso (già misurato in webapp/data/benchmark/builtin.json)
python scripts/benchmark.py webapp/data

# 1. modifica il codice in gpt/ (qui: nn.GELU() al posto di nn.ReLU() in gpt/feedforward.py)

# 2. riaddestra registrando la storia; --no-export è importante: non tocca i dati della web app
python scripts/record_training.py --checkpoint checkpoints/gelu/gpt.pt --out checkpoints/gelu/training --no-export
#    prova veloce (meno di un minuto): aggiungi --max-iters 200 --snapshot-steps 0,50,100,200 --threads 1
#    più veloce e senza storia: python train.py --checkpoint checkpoints/gelu/gpt.pt

# 3. esporta il run in runs/gpt-con-gelu/, misurato con il codice modificato
python scripts/export_run.py checkpoints/gelu/gpt.pt --name "GPT con GELU" --description "FFN con GELU" \
    --trace checkpoints/gelu/training --benchmark --not-engine-compatible

# 4. confronta in terminale con il GPT del corso
python scripts/benchmark.py webapp/data/benchmark/builtin.json runs/gpt-con-gelu/benchmark.json
```

Poi, nella web app, apri il tab **Laboratorio pesi** e carica la cartella `runs/gpt-con-gelu`
(scegliendola o trascinandola): i pesi organizzati come il codice Python, il confronto con il GPT del
corso tensore per tensore, i grafici del training e la tabella del benchmark. Se i pesi hanno nomi e
forme del GPT del corso, il browser ricalcola il benchmark sull'estratto con il motore JavaScript e
controlla che coincida con il Python. `--not-engine-compatible` serve quando cambi il *calcolo* senza
cambiare le forme (GELU, un'altra attivazione): il motore JS eseguirebbe ReLU e darebbe numeri sbagliati,
così il Laboratorio mostra i pesi senza eseguirli. I run esportati stanno in `runs/`, che git ignora.

### Un'architettura modificata, dall'idea al confronto

Lo stesso flusso vale per qualunque modifica del codice in `gpt/`: più blocchi, un feed-forward più
largo, RMSNorm al posto di LayerNorm, un'attention scritta in un altro modo, moduli con nomi nuovi. Per
esempio 6 blocchi con il feed-forward largo 3·C invece di 4·C:

```bash
# 1. modifico il codice: in gpt/config.py un campo nuovo, ffn_mult: int = 4, e in gpt/feedforward.py
#    hidden = config.ffn_mult * config.n_embd (ogni campo di GPTConfig diventa un'opzione dello script)

# 2. registro il training in una cartella mia (le opzioni vengono da GPTConfig e TrainConfig: --help)
python scripts/record_training.py --n-layer 6 --ffn-mult 3 --threads 1 \
    --checkpoint checkpoints/sei-blocchi/gpt.pt --out checkpoints/sei-blocchi/training --no-export

# 3. esporto il run con storia e benchmark, misurato dal codice modificato (prima di tornare indietro)
python scripts/export_run.py checkpoints/sei-blocchi/gpt.pt --name "6 blocchi, FFN 3C" \
    --trace checkpoints/sei-blocchi/training --benchmark

# 4. carico runs/6-blocchi-ffn-3c nel tab Laboratorio pesi e lo confronto con il GPT del corso;
#    in terminale: python scripts/benchmark.py webapp/data/benchmark/builtin.json runs/6-blocchi-ffn-3c/benchmark.json
```

Il flusso in cinque passi, con quello che vedi a ogni passo:

1. **Modifico** il codice in `gpt/` (un campo nuovo in `GPTConfig` diventa da solo un'opzione dello script).
2. **Registro** il training con `scripts/record_training.py … --out <cartella> --no-export`: loss, gradienti,
   snapshot dei pesi e, se serve, le note su ciò che è cambiato (i dati del corso restano protetti).
3. **Esporto** il run con `scripts/export_run.py … --trace <cartella> --benchmark`: in `runs/<nome>/` ci sono pesi,
   storia, benchmark e `run.json` con le note (le tue, quelle della registrazione, quelle dell'export).
4. **Carico** la cartella nel tab **Laboratorio pesi** (pannello *Run*: scegli la cartella, o trascinala sulla
   scena): il run diventa **A**, nella scena con i suoi moduli e la sua riga di codice, e **B** diventa da solo il
   GPT del corso. Le note sono nella scheda del run.
5. **Confronto**: pannello *Confronto* (tabella dei tensori più cambiati, *Mostra A − B nella scena*; i tensori che
   non hanno un gemello con la stessa forma, come la FFN da 192 neuroni contro 256, sono elencati e velati), pannello
   *Grafici* (loss train e validation di A e di B, gradienti e update ratio per gruppo di pesi, entropia
   dell'attention) e pannello *Benchmark* (A − B con la colonna "meglio"; per i run che il motore JavaScript sa
   eseguire anche il ricalcolo nel browser con il verdetto ✓ / ≈).

Due cose da sapere: due cartelle con **gli stessi pesi** sono lo stesso run (il Laboratorio li riconosce
dall'impronta dei pesi): la seconda risulta "Già caricato" e tiene nome e note della prima. E la regola dei gruppi
guarda un nome alla volta: un contenitore di blocchi annidato più in basso del terzo livello
(`model.decoder.layers.0…`) dà un solo gruppo, e un'attention con le head messe direttamente nel modello
(`sa_heads.heads.0.key`, il modello intermedio della lezione) dà un gruppo per head, che i grafici chiamano
"blocco 0…3".

- **La registrazione non si ferma per un nome nuovo** (`gpt/recording.py`). I parametri con i nomi del
  corso restano nei loro gruppi (`blocks.0.attention`, `blocks.0.ffwd`, `blocks.0.layernorm`, `ln_f`…);
  un nome nuovo va nel suo modulo: il sotto-modulo del blocco numerato (`blocks.0.norm1.weight` →
  `blocks.0.norm1`) o, fuori dai blocchi, i primi due livelli del nome (`transformer.wte`), con la stessa
  regola della web app. I gruppi stanno in `group_params` del trace e alimentano i grafici per gruppo
  (norma del gradiente, update ratio, deviazione standard dei pesi) e il confronto tensore per tensore.
- **L'entropia dell'attention per head** viene da ogni modulo che ha il metodo `attention_weights`, come
  `Head` del corso, ovunque sia nel modello: riceve gli argomenti di `forward` e restituisce le
  probabilità `(B, T, T)` di una head oppure `(B, H, T, T)` di tutte le head insieme (attention "fusa"),
  con il batch come prima dimensione (un `(B·H, T, T)` si segnala e si salta: si scrive
  `weights.view(B, H, T, T)`). Un blocco riusato in più strati (pesi condivisi, come ALBERT) dà una riga
  per passaggio. Un'attention che non espone il metodo (per esempio con `F.scaled_dot_product_attention`)
  salta solo questa metrica: loss train e validation, learning rate, gradienti, update ratio, snapshot
  dei pesi, testo generato e probabilità sul prompt ci sono sempre.
- **Le note** spiegano ogni adattamento (gruppi nuovi, pesi condivisi come il weight tying tra embedding
  e `lm_head`, attention letta da moduli nuovi o saltata, valori non finiti di un training divergente
  scritti come `null`): sono il diario di bordo del training. Stanno nel campo `notes` di
  `training/trace.json` e le stampa `record_training.py` alla fine; `export_run.py --trace` le **copia in
  `run.json`**, dopo le tue note (`--note`), con il prefisso «storia del training: » (e le stampa come
  `nota: storia del training: …`), e il **Laboratorio pesi le mostra nella scheda del run**. Così capisci
  perché un grafico manca: con un'attention scritta con `F.scaled_dot_product_attention` la registrazione non
  può misurare l'entropia dell'attention, e nei *Grafici* del Laboratorio, al posto della heatmap, la sezione
  "Entropia dell'attention di ogni head" dice proprio questo, con la nota tra virgolette. Il GPT del corso non
  ne ha: il suo trace è identico a prima.
- **I dati del corso sono protetti**: se il modello o il training non sono quelli di riferimento e il
  comando scriverebbe in `webapp/data` o su `checkpoints/gpt.pt`, `record_training.py` si ferma *prima*
  del training e dice che cosa aggiungere (`--out`, `--checkpoint`, `--no-export`). "Di riferimento"
  vuol dire stessi tensori registrati (anche un buffer persistente in più conta), nessun peso condiviso,
  stessa config, stesso training e stesso codice dei moduli: l'impronta di `gpt/model.py`, `gpt/block.py`,
  `gpt/attention.py` e `gpt/feedforward.py` si confronta con quella di `webapp/data/benchmark/builtin.json`,
  così si riconosce anche GELU al posto di ReLU, che non cambia nomi e forme dei pesi (chi cambia
  davvero il GPT del corso usa `--update-webapp`). Prima del training lo script controlla anche che
  `--out` e `--checkpoint` si possano scrivere, per non perdere i minuti di calcolo.
- **Una classe tua, fuori da `gpt/`**: usa `gpt.recording.TrainingRecorder` come callback di
  `gpt.training.train`; vale per ogni modello con l'interfaccia `logits, loss = model(idx, targets)`.

| Comando | Che cosa fa |
|---|---|
| `python scripts/record_training.py [--n-layer N …] [--max-iters N …] [--snapshot-steps 0,50,…] --checkpoint CKPT --out DIR --no-export` | riaddestra registrando la storia in `DIR` (`trace.json` + snapshot) per `export_run.py --trace`; le opzioni del modello e del training sono i campi di `GPTConfig` e `TrainConfig` |
| `python scripts/export_run.py CKPT --name NOME [--description …] [--note …] [--trace DIR] [--benchmark] [--out DIR] [--force]` | esporta un checkpoint in `runs/<nome>/`; `--force` sostituisce un run esistente cancellando solo i file del run |
| `python scripts/benchmark.py SORGENTE [--out benchmark.json]` | misura un checkpoint `.pt` o una cartella di run (anche `webapp/data`) |
| `python scripts/benchmark.py A B` | tabella di confronto A/B in italiano; A e B possono essere anche `benchmark.json` già calcolati |
| `python scripts/export_benchmark_data.py [--check]` | rigenera (o controlla) `webapp/data/benchmark/`: l'estratto per il browser e `builtin.json` |

Su una macchina già occupata da altri programmi aggiungi `--threads 1`: troppi thread di PyTorch si
ostacolano a vicenda e il benchmark può diventare decine di volte più lento (i numeri non cambiano).

**Il benchmark** (`gpt/benchmark.py`, protocollo `finestre-contigue-v1`) è deterministico e
confrontabile: la validation completa di Tiny Shakespeare (l'ultimo 10% di `data/input.txt`, 111.540
caratteri) si taglia in finestre contigue di `block_size` caratteri, e ogni carattere tranne il primo
viene previsto una volta sola. Misura loss, perplexity, bit per carattere, accuratezza top-1 e top-5 del
prossimo carattere, un testo generato con prompt e seme fissi (più la continuazione greedy, che il motore
JS riproduce identica), token al secondo, numero di parametri e durata; per una cartella con la storia
del training, anche ogni snapshot. I pesi si eseguono con la classe del codice **attuale**: un'architettura
migliorata si misura con lo stesso metro. Il GPT del corso, misurato così dai pesi committati:

| | loss (nat) | perplexity | bit per carattere | top-1 | top-5 |
|---|---|---|---|---|---|
| validation completa (111.539 previsioni) | 1.8145 | 6.138 | 2.618 | 45.62% | 79.70% |
| estratto per il browser (4.014 previsioni) | 1.7337 | 5.662 | 2.501 | 47.51% | 80.79% |

Un modello che tira a caso ha loss ln 65 = 4.17 nat: lo snapshot dello step 0 è lì vicino (4.40).

**Architettura modificata.** Per esportare non serve la classe del modello: `export_run.py` legge solo
i tensori dello state_dict, quindi esporta anche pesi con nomi o forme che il codice attuale non sa più
ricostruire. Per *misurarli*, invece, serve il codice che li esegue: lancia `--benchmark` con il codice
modificato, prima di tornare indietro, e confronta poi i `benchmark.json` (quello del GPT del corso è
`webapp/data/benchmark/builtin.json`). La storia del training di un'architettura modificata si registra
come quella del corso (vedi "Un'architettura modificata, dall'idea al confronto").

**Sicurezza.** I checkpoint si leggono con `torch.load(weights_only=True)`: un file scaricato non può
eseguire codice. I nomi dei file letti dai JSON (snapshot, binario dei pesi) devono stare accanto al
JSON, e `--force` cancella soltanto i file che l'export scrive (riesportando un run sul posto, con
`--trace runs/<nome>/training`, la sua storia del training resta dov'è). Nei JSON non finisce la tua cartella
personale: un checkpoint fuori dal progetto diventa `…/gpt.pt`.

## Bug corretti rispetto al notebook

| # | Problema | Correzione | Test |
|---|---|---|---|
| 1 | `!wget` non è Python: il `.py` esportato non si avvia | download automatico con verifica SHA-256 e scrittura atomica | `test_project::test_every_python_file_compiles`, `test_data::test_download_*` |
| 2 | In `Head` lo scaling usa `C**-0.5` (C = n_embd = 64) invece di `head_size**-0.5` (16) | `self.head_size**-0.5`, come nel paper | `test_attention::test_attention_scaling_uses_head_size` |
| 3 | Il GPT finale si chiama `BigramLanguageModel`; refuso `FeedFoward` | `GPTLanguageModel`, `FeedForward` | `test_project::test_class_names_are_fixed` |
| 4 | Le posizioni usano la variabile globale `device`: errore se il modello sta su un altro device | `torch.arange(T, device=idx.device)` | `test_attention::test_positions_are_created_on_the_input_device` |
| 5 | `generate` gira in modalità training (dropout attivo) e costruisce il grafo dei gradienti | `@torch.no_grad()` + `model.eval()`, poi ripristino della modalità | `test_model_behaviour::test_generate_runs_in_eval_mode_*` |
| 6 | Iperparametri globali: impossibile avere due modelli diversi nello stesso processo | `GPTConfig` passata esplicitamente a ogni modulo | `test_project::test_two_configs_can_coexist` |
| 7 | `LayerNorm1d` del notebook usa la varianza corretta (n−1) e non coincide con `nn.LayerNorm` | `unbiased=False` | `test_didactic::test_handmade_layernorm_matches_pytorch` |

Altre migliorie: checkpoint salvato e ricaricato in modo sicuro (`weights_only=True`), errore chiaro per caratteri fuori vocabolario o contesto più lungo di `block_size`, maschera `tril` non più salvata nel checkpoint (non è un peso).

Risultato del training di riferimento (seed 1337, 5000 step, 209.729 parametri): train loss 1.6592, val loss 1.8081 su Apple MPS (i pesi in `webapp/data/`); su CPU 1.6622 / 1.8208. Lo stesso seed dà numeri leggermente diversi su hardware diverso: le operazioni in virgola mobile non vengono eseguite nello stesso ordine. La web app legge questi valori da `webapp/data/training/trace.json`, quindi dopo un nuovo training si aggiornano da soli.
