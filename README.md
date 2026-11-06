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
  export.py             genera webapp/data/model.json dal codice e dal checkpoint
experiments/            script da proiettare in aula, nell'ordine del video
  01_data_and_tokenization.py … 06_edit_a_weight.py
tests/                  pytest: forme, maschera causale, loss, generazione, overfitting, bug
webapp/                 web app 3D a tab (Three.js, KaTeX, highlight.js inclusi; vedi webapp/ARCHITECTURE.md)
scripts/serve_webapp.py esporta i dati del modello e avvia la web app
scripts/record_training.py rifà il training registrandone la storia (webapp/data/training/)
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
