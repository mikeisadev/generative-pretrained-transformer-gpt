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

`app.bus` è un `EventTarget` condiviso tra i tab; `app.open(id)` apre un altro tab.

## Nucleo: `js/core/`

| Modulo | Esporta | Uso |
|---|---|---|
| `dom.js` | `el(tag, attrs, ...children)`, `escapeHtml`, `fmt(n, digits)`, `fillTemplate`, `sleep`, `prefersReducedMotion` | `el("button", {onclick, className, dataset:{x}}, "testo")` |
| `binary.js` | `decodeBytes(b64)`, `fetchBuffer(url)`, `fetchJson(url)`, `halfToFloat`, `float16ToFloat32(buffer)` | dati binari |
| `sceneHost.js` | `SceneHost` | `host.scene`, `host.camera`, `host.controls`, `host.onFrame((dt, elapsed) => …)` (restituisce la funzione di rimozione), `host.start/stop`, `host.zoomDistance()`, `host.pick(mouseEvent, objects)`, `host.visibleAspect()` |
| `director.js` | `CameraDirector`, `meshBounds(object)` | `await director.focus(objectOrBox3, {showFrame, margin, direction, duration})`, `director.flyTo(pos, target, duration)`, `director.hideFrame()`. Un oggetto con `userData.members = [...]` viene inquadrato come unione dei membri. `userData.noBounds`/`isFrame` esclude un mesh dal calcolo |
| `lod.js` | `LODController`, `setGroupOpacity(group, alpha)` | vedi "LOD" |
| `diagram.js` | `makeBlock({title, subtitle, text, width, height, depth, color, fill})`, `makeArrow(from, to, {color, radius, headLength})`, `makeFlow(points, {color, radius, speed})` | blocchi 3D con testo; `flow.userData.tick(dt)` va chiamato a ogni frame |
| `visuals.js` | `CELL`, `COLORS`, `weightTexture(tensor)`, `attentionTexture(b64, T)`, `matrixPlane(texture, rows, cols)`, `label(text, {size, color, sub})`, `addNode()`, `tokenCube(char, size)` | matrici di pesi come heatmap, etichette sprite |
| `layout.js` | `createTabLayout(root, {code, charts, player, hint, background})` | restituisce `{root, sidebar, stage, caption, host, director, codePanel, charts, player}` |
| `ui/sidebar.js` | `Sidebar` | `sidebar.addPane({id, title, hint}) → HTMLElement`, `sidebar.select(id)`, `sidebar.onSelect(fn)` |
| `ui/menu.js` | `Menu`, `mathBox(items, vars)` | fisarmonica di passi `{id, title, children:[{id, title, text(HTML), math:[{tex, note}], extra:(item)=>Node}]}`; `onSelect(item, title)` |
| `ui/codePanel.js` | `CodePanel` | `codePanel.show({title, file, startLine, code, language: "python"|"java"|"javascript"|"bash"|"json"|"xml", highlight:[substring…]})` |
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
  onChange: (id) => {},
});
```
I gruppi dei livelli devono avere **materiali propri** (non condivisi tra livelli): l'opacità viene
sfumata modificando i materiali. Il livello attivo dipende da `host.zoomDistance()` (distanza
camera → punto guardato): più zoom out = livello più astratto. Livello "Blocchi": un `makeBlock` per
componente con titolo + descrizione di cosa fa. Livello "Panoramica": pochi blocchi grandi + frecce
del flusso (diagramma a blocchi).

### Azioni animate (`ActionPlayer`)

```js
ui.player.register({
  id: "prompt", title: "Dal prompt all'output", category: "Prompt",
  summary: "Una frase per la panoramica",
  build: async (ctx) => [ /* passi */ ],      // oppure steps: [...]
  onStart(ctx) {}, onEnd(ctx) {}, onStop(ctx) {},
});
// passo: { title, caption, duration: 2, enter(ctx), update(p, dt, ctx), exit(ctx) }
ui.player.renderOverview(paneElement, { categories: ["Prompt", "Training"] });
ui.player.on("step", (ctx) => …);   // anche "state", "end"
```
`enter` deve impostare lo stato di inizio passo (così "passo indietro" funziona). `ctx.data` è condiviso
tra i passi di un'azione. La barra di controllo (play/pausa, avanti/indietro, velocità) è già pronta.

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

Il codice mostrato deve essere **vero**: per il GPT viene da `data/model.json` (`spec.snippets`);
per Spring AI/Java si usa il codice delle lezioni L16-L18 del corso (vedi incarico) e si dichiara
"esempio" quando è semplificato.

## Dati esportati dal Python (`webapp/data/`)

| File | Contenuto | Prodotto da |
|---|---|---|
| `model.json` | config, pesi quantizzati per le heatmap, snippet di codice, esempio di attention | `gpt/export.py` |
| `weights.json` + `weights.f32.bin` | pesi addestrati float32 completi (per il motore JS) | `gpt/export.py` |
| `training/trace.json` + `training/snap_XXXXX.f16.bin` | storia del training vero: loss per step, metriche, snapshot dei pesi | `scripts/record_training.py` |

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
```
Id utili nel registro: `tokens`, `tok_emb`, `pos_emb`, `embed_add`, `blocks`, `blocks.<i>`,
`blocks.<i>.ln1|heads|heads.<h>|heads.<h>.pattern|proj|add1|ln2|ffwd|add2`, `ln_f`, `lm_head`,
`logits`, `next_token`, `prediction`, `overview`. La scena è una pila: riga 0 input, righe 1…n_layer
i blocchi, ultima riga l'uscita (`ROW_SPACING = 15`), centrata con `root.position.y`.

Viste registrate: `architettura` (base). I moduli aggiungono `pesi`, `neuroni`, `contesto`.

## Test

- Python: `pytest` (dalla radice).
- JavaScript: `node --test webapp/tests/*.test.mjs` (moduli puri senza DOM: motore GPT, libreria RAG, …).
- Visivo: `python scripts/serve_webapp.py --no-export --port 8765` poi
  `NODE_PATH=$(npm root -g) node webapp/tests/screenshot.mjs http://127.0.0.1:8765/ <cartella> transformer,rag`
  (stampa gli errori della console; il codice di uscita è 1 se ce ne sono).
