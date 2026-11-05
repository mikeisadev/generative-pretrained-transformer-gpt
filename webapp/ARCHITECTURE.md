# Architettura della web app (contratto per chi la estende)

Questo documento è il riferimento comune per tutti i moduli. Se un'API qui descritta non basta,
**non modificare file di altri**: segnala cosa serve nel tuo report finale.

## Principi

- **Fedeltà**: ciò che la web app dice sul GPT del corso deve coincidere con il codice Python
  (`gpt/`) e con i dati esportati (`webapp/data/`). Niente numeri copiati a mano quando esiste il dato vero.
- **Niente invenzioni**: per i tab concettuali (RAG, diffusione, audio, LLM moderni) ogni affermazione
  tecnica deve essere verificabile. Ogni tab ha un pannello "Fonti" con i link. Se qualcosa non è
  pubblico (es. l'architettura di un modello chiuso) lo si dice esplicitamente.
  Le simulazioni vanno dichiarate come tali ("simulazione didattica").
- **Lingua**: italiano semplice, tu/voi didattico, termini tecnici standard in inglese (token,
  embedding, attention…). Nessuna emoji nei testi didattici. Analogie concrete.
- **Codice da senior**: moduli piccoli, nomi chiari, commenti che spiegano il perché, niente
  duplicazione. JavaScript moderno (ES modules), nessun bundler, nessuna dipendenza da CDN:
  le librerie stanno in `webapp/vendor/`.
- **Prestazioni**: tante istanze uguali → `THREE.InstancedMesh`; tante linee → un solo
  `THREE.LineSegments` con `BufferGeometry`; libera le risorse (`geometry.dispose()`,
  `material.dispose()`, `texture.dispose()`) quando una vista viene distrutta. Animazioni a 60 fps:
  niente allocazioni dentro `onFrame` se evitabili.
- **Accessibilità**: pulsanti veri (`<button>`), `aria-label` sulle icone, testo leggibile su sfondo scuro.

## Struttura

```
webapp/
  index.html            guscio: barra dei tab, importmap ("three", "three/addons/…")
  ARCHITECTURE.md       questo file
  css/core.css          stili condivisi (layout, sidebar, pannelli, player, LOD)
  css/tabs/<id>.css     stili di un tab: classi con prefisso del tab (es. .rag-…)
  data/                 dati esportati dal Python (non modificarli a mano)
  vendor/               librerie incluse (three, katex, highlight.js)
  tests/                test JS (node --test) e script di screenshot
  js/app.js             registro dei tab e routing (#transformer, #rag, …)
  js/core/              nucleo condiviso (vedi sotto)
  js/engine/            motore di inferenza GPT in JavaScript
  js/shared/rag/        libreria RAG didattica (corpus, embedding, chunking, HNSW, vector store)
  js/tabs/<id>/         un tab per cartella; entry point index.js
```

Proprietà dei file: ogni agente modifica **solo** i file del proprio modulo, elencati nel suo incarico.

## Contratto di un tab

```js
// js/tabs/<id>/index.js
export async function createTab(root, app) {
  const ui = createTabLayout(root, { code: true, charts: true, player: true, hint: "…" });
  // … costruisci scena, sidebar, azioni …
  return { activate() { ui.host.start(); ui.host.resize(); }, deactivate() { ui.host.stop(); } };
}
```

Il guscio (`js/app.js`) importa il modulo solo alla prima apertura, mostra un tab alla volta e
chiama `deactivate()` su quello che si nasconde: un tab nascosto **non disegna** (il suo `SceneHost`
è fermo). Avanti/indietro del browser funzionano perché il tab attivo è nell'hash (`#rag`, `#audio`…).

| API di `app` | Uso |
|---|---|
| `app.bus` | `EventTarget` condiviso tra i tab (eventi sotto) |
| `await app.open(id)` | apre un tab (anche `"rag/q=…"`) e si risolve con la sua istanza quando è pronta |
| `await app.whenReady(id)` | attende che il tab sia stato creato, senza aprirlo (null se il caricamento fallisce) |

Un evento inviato a un tab mai aperto va perso (il suo modulo non è ancora caricato):
prima `await app.open(id)`, poi `app.bus.dispatchEvent(...)`.

### Eventi tra tab (`app.bus`)

| Evento | `detail` | Chi ascolta | Effetto |
|---|---|---|---|
| `rag:query` | `{ question, topK?, similarityThreshold?, source? }` | tab RAG | imposta la domanda e avvia l'azione "Ricerca" |
| `transformer:view` | `"contesto"` (id di una vista) oppure `{ view, question? }` | tab Transformer | apre il tab e la vista; con `question` la passa alla vista Contesto |
| `context:question` | `{ question }` | vista Contesto (Transformer) | imposta la domanda della vista (e la ricalcola se è aperta) |

Collegamenti già presenti: il pulsante "Approfondisci nel tab Database vettoriale" della vista Contesto
manda `rag:query`; il pulsante del tab RAG verso il Transformer manda `transformer:view`.
L'indirizzo `#rag/q=<domanda>` avvia una ricerca all'apertura del tab RAG.

## Nucleo: `js/core/`

| Modulo | Esporta | Uso |
|---|---|---|
| `dom.js` | `el(tag, attrs, ...children)`, `escapeHtml`, `fmt(n, digits)`, `fillTemplate`, `sleep`, `prefersReducedMotion` | `el("button", {onclick, className, dataset:{x}}, "testo")` |
| `binary.js` | `decodeBytes(b64)`, `fetchBuffer(url)`, `fetchJson(url)`, `halfToFloat`, `float16ToFloat32(buffer)` | dati binari |
| `sceneHost.js` | `SceneHost` | `host.scene`, `host.camera`, `host.controls`, `host.onFrame((dt, elapsed) => …)` (restituisce la funzione di rimozione; `dt` è limitato a 0,05 s), `host.start/stop`, `host.running`, `host.zoomDistance()`, `host.pick(mouseEvent, objects)`, `host.visibleAspect()` (larghezza/altezza della parte non coperta dai pannelli) |
| `director.js` | `CameraDirector`, `meshBounds(object, {visibleOnly})` | `await director.focus(objectOrBox3, {showFrame, margin, direction, minDistance, duration})`, `director.flyTo(pos, target, duration)`, `director.hideFrame()`, `director.cancel()`. I voli avanzano con il tempo reale (anche con frame lenti durano quanto chiesto); trascinare la scena li interrompe. Un oggetto con `userData.members = [...]` viene inquadrato come unione dei membri. `userData.noBounds`/`isFrame` esclude un mesh dal calcolo; `visibleOnly` salta i sotto-alberi nascosti |
| `lod.js` | `LODController`, `setGroupOpacity(group, alpha)`, `setBaseOpacity(material, opacity)` | vedi "LOD" |
| `diagram.js` | `makeBlock({title, subtitle, text, width, height, depth, color, fill})`, `makeArrow(from, to, {color, radius, headLength})`, `makeFlow(points, {color, radius, speed})` | blocchi 3D con testo (la descrizione si rimpicciolisce fino al 60% per entrare tutta; la faccia col testo ha il polygon offset contro lo z-fighting); `flow.userData.tick(dt)` va chiamato a ogni frame |
| `visuals.js` | `CELL`, `COLORS`, `weightTexture(tensor)`, `attentionTexture(b64, T)`, `matrixPlane(texture, rows, cols)`, `label(text, {size, color, sub})`, `addNode()`, `tokenCube(char, size)` | matrici di pesi come heatmap, etichette sprite |
| `layout.js` | `createTabLayout(root, {code, charts, player, hint, background})` | restituisce `{root, sidebar, stage, caption, host, director, codePanel, charts, player}` |
| `ui/sidebar.js` | `Sidebar` | `sidebar.addPane({id, title, hint}) → HTMLElement`, `sidebar.select(id)`, `sidebar.onSelect(fn)` |
| `ui/menu.js` | `Menu`, `mathBox(items, vars)` | fisarmonica di passi `{id, title, children:[{id, title, text(HTML), math:[{tex, note}], extra:(item)=>Node}]}`; `onSelect(item, title)` |
| `ui/codePanel.js` | `CodePanel` | `codePanel.show({title, file, startLine, code, language: "python"|"java"|"javascript"|"bash"|"json"|"xml", highlight:[substring…]})`, `codePanel.hide()`. Se lo studente ha ridotto il pannello con "▁", `show()` lo lascia ridotto (cambia solo l'intestazione); ridurlo da codice: `codePanel.root.classList.add("minimized")` |
| `ui/player.js` | `ActionPlayer` | vedi "Azioni animate" |
| `ui/charts.js` | `ChartsPanel` | vedi "Grafici" |

### LOD (zoom semantico)

```js
const lod = new LODController(host, {
  levels: [
    { id: "detail", label: "Dettaglio", maxDistance: 45, group: detailGroup },
    { id: "blocks", label: "Blocchi", maxDistance: 120, group: blocksGroup },
    { id: "macro", label: "Panoramica", maxDistance: Infinity, group: macroGroup },
  ],
  widgetParent: ui.stage,          // selettore Auto / Dettaglio / Blocchi / Panoramica
  onChange: (id) => {},            // a ogni cambio del livello attivo (anche da codice)
  onPick: (mode) => {},            // solo quando lo STUDENTE sceglie un livello: di solito per inquadrarlo
  distanceScale: () => 1,          // moltiplica le soglie (es. > 1 su schermi stretti)
});
lod.setMode("auto" | "detail" | …); // da codice (un'azione blocca un livello); lod.mode, lod.current
setBaseOpacity(material, 0.4);     // animare l'opacità di un materiale senza litigare con le sfumature
```
I gruppi dei livelli devono avere **materiali propri** (non condivisi tra livelli): l'opacità viene
sfumata modificando i materiali. Il livello attivo dipende da `host.zoomDistance()` (distanza
camera → punto guardato): più zoom out = livello più astratto. Livello "Blocchi": un `makeBlock` per
componente con titolo + descrizione di cosa fa. Livello "Panoramica": pochi blocchi grandi + frecce
del flusso (diagramma a blocchi).

Regole comuni a tutti i tab:
- scegliere un livello dal selettore inquadra quel livello (con `onPick` o in `onChange` se `lod.mode === id`);
  durante un'azione la regia della camera è dell'azione;
- le soglie vanno pensate per lo stage di riferimento (1600×900 con sidebar aperta, rapporto visibile
  ≈ 0,76) e scalate quando lo stage visibile è più stretto (`distanceScale` o equivalente nel tab);
- i livelli astratti usano `makeBlock`: titolo breve, descrizione di 1-3 frasi.

### Azioni animate (`ActionPlayer`)

```js
ui.player.register({
  id: "prompt", title: "Dal prompt all'output", category: "Prompt",
  summary: "Una frase per la panoramica",
  build: async (ctx) => [ /* passi */ ],      // oppure steps: [...]; ctx.isStale() è true se nel
                                              // frattempo l'azione è stata fermata o riavviata
  onStart(ctx) {}, onEnd(ctx) {}, onStop(ctx) {},
});
// passo: { title, caption, duration: 2, enter(ctx), update(p, dt, ctx), exit(ctx) }
ui.player.renderOverview(paneElement, { categories: ["Prompt", "Training"] });
ui.player.on("step", (ctx) => …);   // anche "state", "end"
```
`enter` deve impostare lo stato di inizio passo (così "passo indietro" funziona). `ctx.data` è condiviso
tra i passi di un'azione. La barra di controllo (play/pausa, avanti/indietro, velocità) è già pronta.
`onStop` (azione chiusa o interrotta) deve riportare la scena allo stato di riposo; `onEnd` decide se
lasciare il risultato finale. Un tab che si nasconde ferma il suo `SceneHost`, quindi l'azione si
congela e riprende al ritorno (RAG, Immagini e LLM moderni invece la chiudono in `deactivate`).

### Layout condiviso (`css/core.css`)

- Desktop: sidebar 380 px a sinistra; in alto a sinistra ☰ e selettore dello zoom, in alto a destra la
  barra strumenti del tab (`.stage-toolbar`) e "Grafici"; codice a destra (sotto i grafici se aperti);
  player in basso al centro, sopra il codice (z-index 16), centrato nella parte libera se i grafici
  sono aperti; durante un'azione il codice si accorcia per non coprire il player.
- Schermi ≤ 900 px: la sidebar è un cassetto (si chiude toccando la scena); prima fila ☰, barra
  strumenti, "Grafici"; seconda fila il selettore dello zoom; con la sidebar aperta i comandi dello
  stage si nascondono. Durante un'azione il codice sale sotto il selettore dello zoom e il player
  resta in basso; un tab che inquadra la scena sapendo che il codice è in basso aggiunge
  `.code-stays-bottom` alla sua radice (lo fa il Transformer).
- Niente `backdrop-filter` sui pannelli sopra la scena: con il rendering software costa 18-30 ms per frame.

### Grafici (`ChartsPanel`)

Pulsante fluttuante "Grafici" in alto a destra dello stage; apre un pannello laterale scorrevole.

```js
const section = ui.charts.section("training", "Training", "descrizione breve");   // crea o riusa
const loss = section.chart("loss", {
  type: "line", title: "Loss", description: "Perché conta: …", xLabel: "step", yLabel: "cross-entropy",
  yScale: "linear" | "log",
});
loss.setData({ series: [{ name: "train", points: [[0, 4.4], …] }, { name: "val", points: […] }] });
loss.append("train", [100, 3.1]);          // aggiunta incrementale (ridisegno a requestAnimationFrame)
loss.setHighlight({ x: 100 });             // linea verticale "siamo qui"
loss.clear();
section.clear();                           // svuota la sezione
ui.charts.open(); ui.charts.close(); ui.charts.toggle(); ui.charts.coveredWidth();
ui.charts.onResize(fn)                     // per ricalcolare il view offset della scena
```
Tipi e formati di `setData`:
- `line`: `{ series: [{ name, points: [[x, y], …] }] }`
- `bar`: `{ labels: [...], values: [...], highlight?: index }` (es. top-k probabilità)
- `histogram`: `{ values: number[] | Float32Array, bins?: 40 }`
- `heatmap`: `{ rows, cols, values: Float32Array (riga per riga), rowLabels?, colLabels?, min?, max?, diverging?: bool }`
- `scatter`: `{ points: [{ x, y, label?, group? }] }`

Ogni grafico ha titolo, descrizione ("a cosa serve") e legenda. Palette e regole: skill `dataviz`.

## Pannello codice

Il codice mostrato deve essere **vero**: per il GPT viene da `data/model.json` (chiave `snippets`;
in JS `spec.snippets[chiave] → {file, start_line, code}`, mostrato con `tab.showCode(chiave, titolo, highlight)`);
per Spring AI/Java si usa il codice delle lezioni L16-L18 del corso (vedi incarico) e si dichiara
"esempio" quando è semplificato.

## Dati esportati dal Python (`webapp/data/`)

| File | Contenuto | Prodotto da |
|---|---|---|
| `model.json` | config, pesi quantizzati per le heatmap, snippet di codice, esempio di attention | `gpt/export.py` |
| `weights.json` + `weights.f32.bin` | pesi addestrati float32 completi (per il motore JS) | `gpt/export.py` |
| `training/trace.json` + `training/snap_XXXXX.f16.bin` | storia del training vero: loss per step, metriche, snapshot dei pesi | `scripts/record_training.py` |

In JavaScript: `loadEngine("data/") → GPTEngine` (`js/engine/gpt.js`) con `encode/decode`,
`forward(ids, {trace})`, `nextTokenDistribution`, `generate(ids, n, {temperature, topK, seed})`,
`logitLens`, `loadTrace()`, `loadSnapshot(step)` / `setWeights` / `restoreWeights` (pesi del training).
Nel tab Transformer si usa `tab.getEngine()`, che lo carica una volta sola. La libreria RAG
(`js/shared/rag/`: corpus, pulizia, chunker, embedder, HNSW, vector store, prompt) è pura come il
motore e ha i suoi test in `tests/rag.test.mjs` e `tests/hnsw.test.mjs`.

Formati dettagliati: vedi le docstring di `gpt/export.py` e `gpt/recording.py`.

## Il tab Transformer: oggetto `tab`

`js/tabs/transformer/index.js` crea l'oggetto `tab` e chiama `install(tab)` dei moduli:

```
tab.app, tab.ui (output di createTabLayout), tab.spec (model.json), tab.vars
tab.arch = { root, rail, registry: Map(id → Object3D), stations, path }   // vedi architecture.js
tab.flow (DataFlow: pacchetto animato sul binario; flow.toggle(), flow.packet)
tab.events (EventTarget: evento "view" con detail = id vista)
tab.panes = { path, actions, views }   // elementi dei pannelli della sidebar
tab.menu (Menu del percorso)
tab.views.register({ id, label, description, order, enter(tab), exit(tab) }); tab.views.activate(id)
tab.showCode(snippetKey, title, highlight[]); tab.focus(registryIdOrObject, options); tab.setCaption(text)
tab.getEngine() → Promise<GPTEngine> (js/engine/gpt.js, caricato una volta e condiviso)
tab.arch.tensorMeshes: Map(nome tensore PyTorch → [mesh della heatmap])
tab.lod = { controller, levels: {detail, blocks, macro}, register({detail, blocks, macro}) → rimuovi,
            setDistances({detail, blocks}), resetDistances(), setBaseOpacity, arch, current }  // lod.js
tab.events: "view" (detail = id vista), "lod" (detail = id livello)
```

Moduli del tab (ognuno esporta `install…(tab)`):

| Cartella / file | Cosa aggiunge |
|---|---|
| `architecture.js`, `flow.js` | la pila 3D con i pesi veri e il pacchetto che scorre sul binario |
| `lod.js`, `views/archLevels.js`, `views/lodBlocks.js` | zoom semantico Dettaglio / Blocchi / Panoramica e l'azione "Zoom semantico" |
| `views/weights.js` | vista **Pesi**: dove stanno i 209.729 parametri, sovrapposta alla scena, con giro guidato |
| `views/neurons.js` (+ `neuronModel.js`, `neuronScene.js`) | vista **Rete neurale**: tutti i neuroni e le connessioni, forward animato, clic su un neurone per il suo calcolo |
| `actions/` | azioni **Prompt** ("Dal prompt all'output") e **Training** (uno step al microscopio, training completo registrato) con lavagne e grafici |
| `context/` | vista **Contesto**: domanda → guardrail → embedding → ricerca su pgvector → prompt di sistema, chunk, tool → richiesta all'LLM, collegata al tab RAG |
| `content.js` | testi del Percorso e fonti |

Id utili nel registro: `tokens`, `tok_emb`, `pos_emb`, `embed_add`, `blocks`, `blocks.<i>`,
`blocks.<i>.ln1|heads|heads.<h>|heads.<h>.pattern|proj|add1|ln2|ffwd|add2`, `ln_f`, `lm_head`,
`logits`, `next_token`, `prediction`, `overview`. La scena è una pila: riga 0 input, righe 1…n_layer
i blocchi, ultima riga l'uscita (`ROW_SPACING = 15`), centrata con `root.position.y`.

Viste registrate: `architettura` (base; tornandoci la camera, il codice e la didascalia tornano allo
step scelto nel Percorso). I moduli aggiungono `pesi`, `neuroni`, `contesto`. Scegliere a mano un
livello dello zoom inquadra quel livello (vedi `installLevelPicking` in `index.js`).

## Gli altri tab

| Tab | Cartella | Contenuto |
|---|---|---|
| Database vettoriale (RAG) | `tabs/rag/` | pipeline L17-L18 a stazioni: upload, estrazione, pulizia, chunking, embedding, `vector_store`, indice HNSW; poi ricerca HNSW, prompt, LLM, streaming. Azioni "Indicizzazione" e "Ricerca" |
| Immagini e video | `tabs/diffusion/` | DDPM/DDIM, Stable Diffusion (text encoder, VAE, U-Net, scheduler), DiT, video (VAE 3D, patch spazio-temporali). Azioni di training, generazione, VAE, denoiser, video |
| Audio e musica | `tabs/audio/` | forma d'onda, STFT, Mel, codec neurale con RVQ, generazione di token musicali (delay pattern), Whisper. Laboratorio con suono vero |
| LLM moderni | `tabs/modern-llm/` | 16 meccanismi (RMSNorm, RoPE, GQA, MoE, KV cache, FlashAttention, BPE, decodifica speculativa, scaling, post-training, ragionamento, agenti, coding, multimodale, mesh 3D, interpretabilità) con azioni animate |

Ogni tab ha `window.__<nome>Tab` (es. `__ragTab`, `__diffusionTab`, `__audioTab`, `__modernLlmTab`;
per il Transformer `__gptApp`) per la console e per gli script di verifica.

## Test

- Python: `pytest` (dalla radice).
- JavaScript: `node --test webapp/tests/*.test.mjs` (moduli puri senza DOM: motore GPT, libreria RAG, …).
- Visivo: `python scripts/serve_webapp.py --no-export --port 8765` poi
  `NODE_PATH=$(npm root -g) node webapp/tests/screenshot.mjs http://127.0.0.1:8765/ <cartella> [tab1,tab2,…] [--lod] [--mobile]`
  (senza elenco: tutti i tab; `--lod` aggiunge uno screenshot per livello dello zoom, `--mobile` usa 390×844;
  stampa gli errori della console e il codice di uscita è 1 se ce ne sono).
