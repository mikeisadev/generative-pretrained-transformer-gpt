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
python scripts/serve_webapp.py     # poi apri http://localhost:8000
```

Prova veloce del training: `python train.py --max-iters 500`. Test: `pytest`.

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
webapp/                 web app 3D (Three.js, KaTeX, highlight.js inclusi)
scripts/serve_webapp.py esporta i dati del modello e avvia la web app
train.py, generate.py   punti di ingresso
reference/              notebook ed export originali di Karpathy, non modificati
```

Gli esperimenti si lanciano dalla radice, per esempio `python experiments/03_attention_math_trick.py`. Il numero 06 modifica un solo peso del modello addestrato e mostra l'effetto sul testo generato (serve prima `python train.py`).

## Web app

`python scripts/serve_webapp.py` rigenera `webapp/data/model.json` dal codice Python e dal checkpoint (se esiste), poi serve la cartella `webapp/` solo su `localhost`.

- **Menu a sinistra**: gli step in ordine di flusso, con spiegazione, tabella dei pesi e riquadro "Approfondimento" con le formule.
- **Scena 3D**: ogni cella colorata è un peso vero (arancione positivo, blu negativo). I pattern di attention sono calcolati dal modello su un prompt d'esempio. Il flusso dei dati è animato.
- **Finestra a destra**: il codice Python dello step, preso dal sorgente del pacchetto.
- Frecce ↑ ↓ per passare da uno step all'altro.

Un test (`tests/test_webapp_data.py`) fallisce se `model.json` non è allineato al codice: dopo aver modificato il pacchetto `gpt`, rilancia lo script.

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

Risultato del training di riferimento (CPU, seed 1337, 5000 step, 162 s): train loss 1.6622, val loss 1.8208, 209.729 parametri.
