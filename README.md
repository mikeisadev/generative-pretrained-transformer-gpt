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
  training.py           9. training, stima della loss, checkpoint
  generation.py         10. generazione token per token (con temperature)
  loss.py               cross-entropy condivisa da bigram e GPT
  weights.py            inventario dei pesi e modifica di un singolo peso
  export.py             genera webapp/data/model.json dal codice e dal checkpoint; capture_attention
  recording.py          registra la storia di un training (loss, metriche, snapshot), anche di un'architettura modificata
  runs.py               run esportabili per il Laboratorio pesi: pesi, storia del training, run.json
  benchmark.py          benchmark deterministico e confrontabile (protocollo finestre-contigue-v1)
experiments/            script da proiettare in aula, nell'ordine del video
  01_data_and_tokenization.py … 06_edit_a_weight.py
tests/                  pytest: forme, maschera causale, loss, generazione, overfitting, bug, registrazione, run, benchmark
webapp/                 web app 3D a tab (Three.js, KaTeX, highlight.js inclusi; vedi webapp/ARCHITECTURE.md)
  js/…                  nuove: core/neuralNet/ (rete neurale), core/brain/ (cervello 3D), shared/weights/ (pesi e run), tabs/lab/ (Laboratorio pesi)
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
| **Transformer** | Il GPT del corso in 3D: ogni cella colorata è un peso vero (arancione positivo, blu negativo). Viste: *Architettura*, *Pesi* (dove stanno i 209.729 parametri), *Rete neurale* (tutti i neuroni, le connessioni e i calcoli, con un clic su un neurone), *Contesto* (domanda → database vettoriale → system prompt e tool → richiesta all'LLM, collegata al tab RAG). Azioni *Prompt* (dalla scrittura del prompt ai token, dentro i blocchi, fino al token estratto, calcolato dal motore JS con i pesi veri) e *Training* (uno step al microscopio e il training completo registrato) |
| **Database vettoriale (RAG)** | La pipeline delle lezioni L16-L18 (Spring Boot + Spring AI + pgvector): database vuoto, upload, pulizia, chunking, embedding, tabella `vector_store`, indice HNSW; poi ricerca HNSW, prompt e risposta in streaming |
| **Immagini e video** | Modelli di diffusione: processo diretto e training, generazione (text encoder, U-Net o DiT, scheduler, VAE), video con VAE 3D e patch spazio-temporali |
| **Audio e musica** | Dal suono ai numeri (STFT, scala Mel), codec neurale con RVQ, generazione di musica a token, Whisper; laboratorio con suono vero |
| **LLM moderni** | Cosa cambia rispetto al nostro GPT: RoPE, GQA, MoE, KV cache, FlashAttention, BPE, leggi di scala, post-training, ragionamento, agenti, coding, capacità emergenti, multimodalità, modelli 3D |

### Come si usa

- **Sidebar a sinistra**: *Percorso* (gli step in ordine, con spiegazione, formule e tabella dei pesi), *Azioni* (panoramica delle animazioni, divise per categoria: prompt, training…), *Viste*, *Fonti*. Con la tastiera: ↑ ↓ cambiano step, lo spazio mette in pausa l'azione (in tutti i tab).
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
node --test webapp/tests/*.test.mjs            # motore GPT in JS, libreria RAG, HNSW, grafici, navigazione, ricerca
NODE_PATH=$(npm root -g) node webapp/tests/screenshot.mjs http://127.0.0.1:8765/ /tmp/shots --lod --nav
```

Lo script di screenshot (Playwright con Chromium) apre tutti i tab e i livelli dello zoom (`--lod`), prova
la navigazione con i tasti e "Centra" (`--nav`), usa il formato telefono con `--mobile` e termina con
errore se la console ne segnala. L'architettura del codice è descritta in `webapp/ARCHITECTURE.md`.

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

- **La registrazione non si ferma per un nome nuovo** (`gpt/recording.py`). I parametri con i nomi del
  corso restano nei loro gruppi (`blocks.0.attention`, `blocks.0.ffwd`, `blocks.0.layernorm`, `ln_f`…);
  un nome nuovo va nel suo modulo: il sotto-modulo del blocco numerato (`blocks.0.norm1.weight` →
  `blocks.0.norm1`) o, fuori dai blocchi, i primi due livelli del nome (`transformer.wte`), con la stessa
  regola della web app. I gruppi stanno in `group_params` del trace e alimentano i grafici per gruppo
  (norma del gradiente, update ratio, deviazione standard dei pesi) e il confronto tensore per tensore.
- **L'entropia dell'attention per head** viene da ogni modulo che ha il metodo `attention_weights`, come
  `Head` del corso, ovunque sia nel modello: riceve gli argomenti di `forward` e restituisce le
  probabilità `(B, T, T)` di una head oppure `(B, H, T, T)` di tutte le head insieme (attention "fusa").
  Un'attention che non lo espone (per esempio con `F.scaled_dot_product_attention`) salta solo questa
  metrica: loss train e validation, learning rate, gradienti, update ratio, snapshot dei pesi, testo
  generato e probabilità sul prompt ci sono sempre.
- **Le note** spiegano ogni adattamento (gruppi nuovi, attention letta da moduli nuovi o saltata, valori
  non finiti di un training divergente scritti come `null`): stanno nel campo `notes` di
  `training/trace.json` e le stampa `record_training.py` alla fine. Il GPT del corso non ne ha: il suo
  trace è identico a prima.
- **I dati del corso sono protetti**: se il modello o il training non sono quelli di riferimento e il
  comando scriverebbe in `webapp/data` o su `checkpoints/gpt.pt`, `record_training.py` si ferma *prima*
  del training e dice che cosa aggiungere (`--out`, `--checkpoint`, `--no-export`).
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
JSON, e `--force` cancella soltanto i file che l'export scrive.

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
