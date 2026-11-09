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
  css/<modulo>.css      stili dei moduli del nucleo: nav, palette, reps (rappresentazioni), neuralnet, brain
  data/                 dati esportati dal Python (non modificarli a mano)
  vendor/               librerie incluse (three, katex, highlight.js)
  tests/                test JS (node --test) e script di screenshot
  js/app.js             registro dei tab e routing (#transformer, #rag, …)
  js/core/              nucleo condiviso (vedi sotto)
  js/engine/            motore di inferenza GPT in JavaScript
  js/shared/rag/        libreria RAG didattica (corpus, embedding, chunking, HNSW, vector store)
  js/shared/weights/    libreria dei pesi e dei run (formati, safetensors, run, statistiche, confronto, salvataggio)
  js/tabs/<id>/         un tab per cartella; entry point index.js, indice della ricerca search.js
```

Proprietà dei file: ogni agente modifica **solo** i file del proprio modulo, elencati nel suo incarico.

## Contratto di un tab

```js
// js/tabs/<id>/index.js
export async function createTab(root, app) {
  const ui = createTabLayout(root, { code: true, charts: true, player: true, hint: "…" });
  // … costruisci scena, sidebar, azioni …
  return {
    activate() { ui.host.start(); ui.host.resize(); },
    deactivate() { ui.host.stop(); },
    async goto({ kind, id }) { /* facoltativo: vedi "Ricerca globale, link diretti e glossario" */ },
  };
}
```

Il guscio (`js/app.js`) importa il modulo solo alla prima apertura, mostra un tab alla volta e
chiama `deactivate()` su quello che si nasconde: un tab nascosto **non disegna** (il suo `SceneHost`
è fermo; la navigazione libera si ferma da sola, niente da fare nel tab). Avanti/indietro del browser
funzionano perché il tab attivo è nell'hash (`#rag`, `#audio`…).

Comandi uguali in tutti i tab: la navigazione libera arriva con `createTabLayout` (`ui.nav`, vedi
"Navigazione libera"); ↑ ↓ del Percorso e Spazio del player si collegano con una riga:
`installStepKeys({ root, ui, menu: () => tab.menu, pathPane })` (`core/stepKeys.js`).

| API di `app` | Uso |
|---|---|
| `app.bus` | `EventTarget` condiviso tra i tab (eventi sotto) |
| `await app.open(id)` | apre un tab (anche `"rag/q=…"`) e si risolve con la sua istanza quando è pronta |
| `await app.whenReady(id)` | attende che il tab sia stato creato, senza aprirlo (null se il caricamento fallisce) |
| `await app.goto(id, {kind, id})` | apre il tab e lo porta su un passo / azione / vista (chiama `instance.goto`), oppure mostra un termine del glossario (`kind: "term"`); aggiorna l'indirizzo in `#<tab>/go=<kind>:<id>`. → `true` se l'elemento è stato mostrato, `false` se si è solo aperto il tab. Senza il secondo argomento apre solo il tab |
| `app.openSearch(query?)` | apre la ricerca globale (eventualmente con un testo già scritto) |
| `app.showTerm(id)` | mostra la scheda di un termine del glossario senza cambiare tab (es. un link "che cos'è?" nel testo di un passo) |

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

### Ricerca globale, link diretti e glossario

La barra in alto ha il pulsante **Cerca** (anche `Ctrl/⌘+K`, oppure `/` fuori dai campi di testo):
apre una palette che cerca in **tutti** i tab (passi del Percorso, azioni, viste, tab e glossario),
anche in quelli mai aperti. Ricerca tollerante: maiuscole e accenti non contano ("perche" trova
"perché"), le parole parziali valgono ("embed" → "embedding"), un errore di battitura nelle parole
lunghe è perdonato ("sofmax"), le parole vuote ("come", "la", "di"…) si ignorano. Risultati
raggruppati per tab (prima il gruppo con il risultato migliore; il tab aperto ha una piccola spinta),
con il tipo (Passo / Azione / Vista / Termine / Tab), ✓ sui passi già visti e filtri per tipo.
Tastiera: ↑ ↓ scelgono, Invio apre, Maiusc+Invio copia il link, Esc chiude, Tab resta nella finestra.
Sul telefono la palette occupa tutto lo schermo.

| Modulo | Esporta | Ruolo |
|---|---|---|
| `core/search.js` | `normalize`, `queryTokens`, `plainText`, `prepareEntry`, `search(index, query, {filter, boost, limit})`, `groupResults`, `highlightRanges`, `excerpt`, `parseHash(hash)`, `formatHash(tab, {kind, id})`, `KINDS` | logica pura della ricerca e dei link (test in `tests/search.test.mjs`) |
| `core/glossary.js` | `GLOSSARY`, `glossaryTerm(id)`, `glossarySearchIndex()` | 47 termini: definizione semplice, analogia, tab e passo in cui vederli |
| `core/progress.js` | `progress` (`ProgressStore`: `mark(tab, id)`, `visited(tab)`, `count(tab, ids)`, `reset(tab)`, evento `"change"`) | passi visti per tab, nel localStorage (con try/catch: senza storage valgono per la sessione) |
| `core/ui/palette.js` | `SearchPalette` | la finestra di ricerca (caricata solo alla prima apertura) |
| `core/ui/share.js` | `copyLink(hash)`, `copyText(text)`, `showToast(text)`, `urlWithHash(hash)` | "Copia link" (anche fuori da https) e avvisi brevi |

**Link diretti.** `#<tab>/go=<kind>:<id>` (id codificato con `encodeURIComponent`), per esempio
`#transformer/go=step:head.qkv`, `#transformer/go=action:prompt`, `#transformer/go=view:pesi`,
`#modern-llm/go=term:rope`. Funziona al caricamento della pagina, quando l'hash cambia nella stessa
pagina e con avanti/indietro del browser (ogni salto di `app.goto` è una voce della cronologia).
Un link a un elemento che non esiste più apre il tab e mostra un avviso. `#rag/q=…` resta valido.
Per costruire un link: `formatHash(tab, {kind, id})` da `core/search.js`; per copiarlo: `copyLink(hash)`.

**Contratto per i tab (da implementare in ogni tab).**

1. `js/tabs/<id>/search.js` esporta l'indice, fatto **solo di dati leggeri**: può importare
   `content.js` e altri moduli di soli dati, **mai** three.js, la scena o `index.js` (viene caricato
   anche se il tab non è mai stato aperto).
   ```js
   // js/tabs/rag/search.js
   import { PATH } from "./content.js";
   export const searchIndex = [
     // un passo del Percorso per ogni sottopunto del Menu (stesso id del Menu)
     ...PATH.flatMap((s) => s.children.map((c) => ({ id: c.id, kind: "step", title: c.title, text: c.text, keywords: [s.title] }))),
     // le azioni animate (stesso id di player.register) e le viste (stesso id di views.register)
     { id: "ricerca", kind: "action", title: "Ricerca", text: "La domanda diventa un vettore…", keywords: ["query", "top-k"] },
   ];
   ```
   Voce: `{ id, kind: "step" | "action" | "view", title, text?, keywords? }`. `title` e `text`
   possono contenere l'HTML di `content.js` e i segnaposto `{{chiave}}`: la ricerca li ripulisce
   (`plainText`) e mostra tutto come testo. `keywords` sono sinonimi e parole che lo studente
   potrebbe cercare (anche in inglese). Niente duplicati `kind:id`. Il modulo può mancare o
   fallire: la palette funziona con gli altri tab (ma il browser segnala il 404 in console,
   quindi ogni tab dovrebbe averlo). `tests/search.test.mjs` controlla ogni `search.js` presente
   (voci valide, id dei passi esistenti in `content.js`).
2. L'istanza restituita da `createTab` può esporre `goto({kind, id}) → Promise<boolean|void>`:
   restituisce `false` se non conosce l'id (il guscio avvisa lo studente), qualsiasi altra cosa se
   l'ha mostrato. Viene chiamata dopo `activate()`, sia per i tab già aperti sia appena creati.
   Senza `goto`, `app.goto` apre soltanto il tab. Implementazione tipica:
   ```js
   async goto({ kind, id }) {
     if (kind === "step") return tab.menu.reveal(id, { sidebar: ui.sidebar });  // apre sidebar e pannello, scorre al passo
     if (kind === "action") { if (!actionIds.has(id)) return false; ui.sidebar.select("azioni"); ui.player.start(id); return true; }
     if (kind === "view") { if (!viewIds.has(id)) return false; tab.views.activate(id); return true; }
     return false;
   },
   ```
3. Il Menu del Percorso va creato con l'id del tab: `new Menu(pane, steps, { vars, onSelect, tabId: "<id>" })`.
   Così mostra i progressi (✓ sui passi visti, "n/m" in ogni sezione, "n/m passi visti" con barra e
   pulsante "Azzera" che chiede conferma) e il pulsante discreto **"Copia link a questo passo"** in
   fondo a ogni passo. Senza `tabId` il Menu funziona come prima (es. il menu della vista Contesto).
   `menu.select(id, {scroll})` apre la sezione, evidenzia il passo e lo porta in vista;
   `menu.reveal(id, {sidebar})` in più apre la sidebar (anche il cassetto sul telefono) e il pannello
   che contiene il menu: è quello che serve a `goto`. Entrambi restituiscono `false` se l'id non esiste.
   Ogni selezione (anche quella automatica all'avvio del tab) conta come passo visto.

**Glossario.** Ogni voce di `core/glossary.js` ha `{ id, term, aliases, definition, analogy, tab,
go: {kind: "step", id}, related }`: testo semplice (niente HTML), numeri verificati sul GPT del corso
o sulle fonti dei tab. Nella palette un termine si apre in una scheda con definizione, "Come
immaginarlo", "Vedi anche" e i pulsanti "Guarda nel tab …" (`app.goto(tab, go)`) e "Copia link".
Un nuovo termine deve puntare a un passo esistente (lo verifica `tests/search.test.mjs`).

## Nucleo: `js/core/`

| Modulo | Esporta | Uso |
|---|---|---|
| `dom.js` | `el(tag, attrs, ...children)`, `escapeHtml`, `fmt(n, digits)`, `fillTemplate`, `sleep`, `prefersReducedMotion` | `el("button", {onclick, className, dataset:{x}}, "testo")` |
| `binary.js` | `decodeBytes(b64)`, `fetchBuffer(url)`, `fetchJson(url)`, `halfToFloat`, `float16ToFloat32(buffer)` | dati binari |
| `sceneHost.js` | `SceneHost` | `host.scene`, `host.camera`, `host.controls` (`minDistance` predefinito 1), `host.onFrame((dt, elapsed) => …)` (restituisce la funzione di rimozione; `dt` è limitato a 0,05 s), `host.start/stop`, `host.running`, `host.onResize(fn)`, `host.onStop(fn)`, `host.snapshot()`, `host.zoomDistance()`, `host.pick(mouseEvent, objects)`, `host.visibleAspect()` (larghezza/altezza della parte non coperta dai pannelli) |
| `director.js` | `CameraDirector`, `meshBounds(object, {visibleOnly})` | `await director.focus(objectOrBox3, {showFrame, margin, direction, minDistance, duration})`, `director.flyTo(pos, target, duration)`, `director.hideFrame()`, `director.cancel()`, `director.flight` (volo in corso o null), `director.actionActive` (un'azione è in corso e non è finita). I voli avanzano con il tempo reale (anche con frame lenti durano quanto chiesto); trascinare la scena li interrompe. Inseguimento e camera libera durante le azioni: vedi "Navigazione libera". Un oggetto con `userData.members = [...]` viene inquadrato come unione dei membri. `userData.noBounds`/`isFrame` esclude un mesh dal calcolo; `visibleOnly` salta i sotto-alberi nascosti |
| `lod.js` | `LODController`, `setGroupOpacity(group, alpha)`, `setBaseOpacity(material, opacity)` | vedi "LOD" |
| `diagram.js` | `makeBlock({title, subtitle, text, width, height, depth, color, fill})`, `makeArrow(from, to, {color, radius, headLength})`, `makeFlow(points, {color, radius, speed})` | blocchi 3D con testo (la descrizione si rimpicciolisce fino al 60% per entrare tutta; la faccia col testo ha il polygon offset contro lo z-fighting); `flow.userData.tick(dt)` va chiamato a ogni frame |
| `visuals.js` | `CELL`, `COLORS`, `weightTexture(tensor)`, `attentionTexture(b64, T)`, `matrixPlane(texture, rows, cols)`, `label(text, {size, color, sub})`, `addNode()`, `tokenCube(char, size)` | matrici di pesi come heatmap, etichette sprite |
| `layout.js` | `createTabLayout(root, {code, charts, player, hint, background})`, `createSceneStack(stage, {root, player, codePanel, charts, coveredRight, background})` | `createTabLayout` restituisce `{root, sidebar, stage, caption, host, director, nav, codePanel, charts, player}` (`nav`: vedi "Navigazione libera"); `createSceneStack` monta solo `SceneHost` + `CameraDirector` (+ `ActionPlayer`) + `Navigation` + `NavPad` → `{host, director, nav, player}` (una seconda scena con la stessa navigazione: il livello delle rappresentazioni) |
| `ui/sidebar.js` | `Sidebar` | `sidebar.addPane({id, title, hint}) → HTMLElement`, `sidebar.select(id)`, `sidebar.onSelect(fn)` |
| `stepKeys.js` | `installStepKeys({root, ui, menu, pathPane, spaceIgnore})` | ↑ ↓ passo del Percorso (uno per pressione), Spazio pausa/riprendi; ascolto sulla finestra, solo con il tab visibile, mai nei campi, nelle finestre di dialogo, sul pad o con l'aiuto aperto; negli altri pannelli della sidebar le frecce li scorrono |
| `ui/menu.js` | `Menu`, `mathBox(items, vars)` | fisarmonica di passi `{id, title, children:[{id, title, text(HTML), math:[{tex, note}], extra:(item)=>Node}]}`; opzioni `{vars, onSelect(item, title), tabId}` (con `tabId`: progressi e "Copia link a questo passo"); `menu.select(id)`, `menu.reveal(id, {sidebar})`, `menu.move(±1)`, `menu.current`, `menu.items` (vedi "Ricerca globale") |
| `ui/codePanel.js` | `CodePanel` | `codePanel.show({title, file, startLine, code, language: "python"\|"java"\|"javascript"\|"bash"\|"json"\|"xml", highlight:[substring…]})`, `codePanel.hide()`. Se lo studente ha ridotto il pannello con "▁", `show()` lo lascia ridotto (cambia solo l'intestazione); ridurlo da codice: `codePanel.root.classList.add("minimized")` |
| `ui/player.js` | `ActionPlayer` | vedi "Azioni animate" |
| `ui/charts.js` | `ChartsPanel` | vedi "Grafici" |
| `representations.js` | `installRepresentations(ui, opts)`, `REP_IDS`, `REP_LINKS` | selettore "Architettura · Rete neurale · Cervello", livello alternativo con la sua pila di scena, ritorno automatico, pannello Cervello; vedi "Rappresentazioni" |
| `ui/repSwitch.js` | `RepSwitch`, `placeSwitch`, `freeIntervals`, `candidateRows`, `VARIANTS`, `ALT_ACTIVE_CLASS` | il selettore (radiogroup / menu compatto) e la scelta pura del suo posto (test in `tests/reps.test.mjs`) |
| `neuralNet/` | `view.js`: `createNetworkView(ctx)`; `model.js`: `buildNetworkModel(spec)`; `scene.js`: `NetworkScene`; `spec.js`: `normalizeSpec`, `NetworkSpecError`, `LIMITS`; `layout.js`, `links.js`, `activations.js` (puri); `roles.js`: `ROLES`, `PATTERNS` | rete orizzontale con profondità guidata da `tabs/<id>/network.js` (`networkSpec`); vedi "Rete neurale orizzontale" |
| `brain/` | `view.js`: `createBrainView(ctx)`; `pane.js`: `renderBrainPane(container, spec, {onShow3D, app})` (solo DOM); `spec.js`: `normalizeBrainSpec`, `BrainSpecError`; `atlas.js`: `REGIONS`, `MARKERS`, `PATHWAYS`, `atlasEntry`, `mniToScene`; `morphology.js`: `NEURON_TYPES`, `buildMorphology` | cervello 3D (regioni, percorsi, piani di taglio, microcircuito di neuroni modellati) e pannello delle analogie, guidati da `tabs/<id>/brain.js` (`brainSpec`); vedi "Cervello" |
| `matrixNumbers.js` | `NumericMatrix({rows, cols, data, cell, range, colormap, group})`, `NumbersGroup`, `attachNumbers(mesh, {rows, cols, getValue, group})`, `colormapLUT` | matrici di pesi come celle colorate con i numeri veri quando sono abbastanza grandi sullo schermo; vedi "Pesi e run" |

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

### Navigazione libera

Ogni tab la riceve da `createTabLayout` (`ui.nav`): nessun codice nel tab è necessario. Moduli:
`core/navigation.js` (logica: mouse, tastiera, touch, confini, vista iniziale), `core/ui/navPad.js`
(pad a schermo, aiuto, suggerimenti, chip "Camera libera"), `core/navMath.js` (regole numeriche pure,
testate in `tests/navigation.test.mjs`), stili in `css/nav.css`.

| Input | Effetto |
|---|---|
| trascina (sinistro) / un dito | ruota attorno al punto guardato; in modalità **Sposta** sposta |
| tasto destro, centrale, Shift/Ctrl/⌘ + trascina / due dita | sposta (pan nel piano dello schermo); due dita: anche pizzico = zoom 1:1 |
| rotella (anche due dita in verticale o pizzico sul trackpad) | zoom verso il puntatore (`zoomToCursor`), ~7% per tacca |
| due dita in orizzontale sul trackpad, Shift + rotella | sposta a destra/sinistra (1:1 con le dita, smorzato) |
| doppio clic / doppio tocco | volo che porta il punto al centro: su un oggetto avvicina anche del 28%, sul vuoto no (resta dentro i confini) |
| `W A S D`, `←` `→` | sposta, in continuo finché il tasto è premuto (accelera e frena dolcemente, velocità ∝ distanza) |
| `Q` `E` · `R` `F` · `+` `−` (`Z` `X`) | gira attorno · inclina · zoom; `Shift` più veloce, `Alt` più preciso |
| `C` / `Home` · `M` · `?` / `H` · `Esc` | vista iniziale · Ruota↔Sposta (ricordata) · aiuto · chiude l'aiuto |

I tasti agiscono solo sul tab visibile e mai mentre si scrive in un campo di testo o una tendina, né dentro
una finestra di dialogo (la ricerca); su un cursore o una casella le frecce e `Home` restano al controllo, ma
le lettere (`W A S D`…) muovono ancora la scena. Con l'aiuto aperto arrivano solo `?`, `H` ed `Esc`.
`↑` `↓` (menu del Percorso), `Spazio` (player), `/` e le scorciatoie con Ctrl/⌘ restano a chi le usava; sui
pulsanti del pad `Spazio`/`Invio` sono del pulsante (tenuti premuti sulle frecce = movimento continuo).
Un semplice clic sulla scena (per scegliere un neurone, una regione…) **non** è navigazione: lo diventa
oltre 4 px di trascinamento.

```js
ui.nav.setHome(() => tab.focus("overview"));  // "Centra" (C, Home, pulsante ◎, "Torna al centro").
                                              // Predefinito: la posa della camera a fine del primo volo del tab
ui.nav.setBounds(box3 | () => box3 | null);   // confini; predefinito: mesh visibili (meshBounds), allargati del 12%
ui.nav.setDragMode("rotate" | "pan");  ui.nav.toggleDragMode();  ui.nav.mode
ui.nav.home();  ui.nav.setEnabled(false);      // spegne mouse, tastiera e pad (es. una vista 2D a tutto schermo)
ui.nav.hold(axis, dir) → rilascia(); ui.nav.nudge(axis, dir)   // axis: panX | panY | orbit | tilt | zoom
ui.nav.panBy(dx, dy); ui.nav.stopMotion()     // sposta nel piano dello schermo (con i confini); ferma tutto
ui.nav.setOrbitLimits({ turn: 60, tilt: 45, minDistance: 6 })  // scene "a parete": gradi dalla vista
                                              // frontale (+z); niente testo a specchio né vista dall'alto
ui.nav.on("input", (source) => …)            // "pointer" | "keys" | "pad" | "dblclick" | "home"; anche "mode", "lost"
ui.nav.dispose()                              // toglie tutti i listener e il pad
```

- **Confini**: il punto guardato resta nel box del contenuto allargato di poco: il 12% della diagonale,
  ma mai più di mezza inquadratura alla distanza attuale (`navMath.boundsMargin`; da vicino, al livello
  Dettaglio, il 12% di una scena grande sarebbe già il vuoto). Il margine si ricalcola quando la distanza
  cambia di più del 10%. Oltre il bordo la spinta frena asse per asse (1/(1+o)², si "scivola lungo il
  muro": con la camera ruotata la componente laterale continua) e una molla riporta dentro. Lo zoom
  indietro (rotella e tasti) si ferma a 6 × il raggio della scena, o 1,5 × la vista iniziale se è già più
  lontana (telefono), mai sotto la distanza attuale; i voli dei tab (es. "Panoramica") possono andare
  oltre: durante un volo il limite si allarga, al primo input dello studente torna stretto.
  "Ti sei perso? Torna al centro" compare quando nell'inquadratura non c'è più nessun oggetto visibile
  (non basta il box: una scena a due file ha angoli vuoti) o la scena è ridotta a un puntino; il
  controllo gira al più ogni 250 ms e solo dopo un movimento (≤ 0,05 ms).
- **Pan e LOD**: il pan sposta camera e punto guardato insieme, quindi `host.zoomDistance()` (e il livello
  LOD) non cambia; lo zoom sì. Chi legge `controls.target` (es. "Dettaglio" della diffusione, regione più
  vicina in LLM moderni) riceve il punto spostato dallo studente.
- **Camera libera** (`CameraDirector`): `layout.js` collega lo stato del player a `director.setActionActive()`.
  Se lo studente si muove durante un'azione (`director.release()`), i voli chiesti dall'azione non muovono
  più la camera: si risolvono subito e l'ultimo viene ricordato; il chip "Camera libera · Segui l'azione"
  (sopra la barra del player) chiama `director.resumeFollow()` e vola lì. A fine azione o quando si ferma
  l'inseguimento torna da solo, e anche con "⟲ Ricomincia" (evento `restart` del player). Se l'azione
  non ha chiesto voli mentre la camera era libera, "Segui l'azione" torna sull'ultima inquadratura
  dell'azione (a meno che il tab, avvisato da `onFollowChange`, non faccia partire un volo suo). I voli
  chiesti dallo studente partono sempre: un clic (non un trascinamento) su menu, pulsanti o scena, o un
  tasto che non sia di navigazione, chiama `director.noteUserIntent()` e i voli dei 700 ms successivi
  sono "suoi"; mentre un volo dello studente è in corso, un volo dell'azione non lo sostituisce (es. il
  passo che scatta 300 ms dopo "Centra"). I clic sulla barra del player (passo avanti/indietro) restano
  dell'azione. API: `director.follow`, `director.actionActive`, `director.release()`,
  `director.resumeFollow()`, `director.noteUserIntent(ms)`, `director.onFollowChange(fn)`.
- **Tab nascosto**: quando il suo `SceneHost` si ferma (`host.onStop`), la navigazione azzera tasti tenuti,
  pulsanti e frenata: al ritorno la scena non scivola.
- **Pad**: in basso a destra della scena visibile; accanto al codice se questo arriva fin giù (solo se così
  resta nella metà destra della scena), a sinistra dei grafici aperti, sopra la barra del player se la
  toccherebbe (e sopra il chip "Segui l'azione", che resta sempre cliccabile); sul telefono, aperto, sopra il
  codice ridotto in fondo. Il chip sta sopra la barra del player, o sopra il codice ridotto se questo sta
  subito sopra la barra (telefono con `.code-stays-bottom`). Se non c'è un posto che lasci libera
  la scena (es. 1100 px con codice e player) si riduce da solo al pulsante unico (classe `squeezed`, non
  ricordata; il pulsante lo riapre comunque). Posizione ricalcolata solo a eventi di layout, mai per frame.
  Contiene Ruota/Sposta, frecce (tenute premute = movimento continuo, tocco breve = un passo), gira attorno,
  zoom, "Centra", aiuto, schermo intero (Fullscreen API su **tutta la pagina**: ricerca, glossario e avvisi
  stanno fuori dal tab e cambiare tab non lascia uno schermo nero), foto PNG (`host.snapshot()`: disegno e
  copia nello stesso istante, niente `preserveDrawingBuffer`; solo la parte di scena non coperta da codice o
  grafici). Si riduce a un pulsante (scelta ricordata); sul telefono parte ridotto, accanto alla didascalia.
  L'aiuto (`?`) è una finestra modale; sui touch screen mostra prima i gesti e i pulsanti a schermo.
- **Prestazioni**: a riposo il ciclo fa solo qualche confronto (nessuna allocazione); misurati ~0-8 µs per
  frame a riposo e ~40-70 µs in movimento. I confini (`meshBounds`) si ricalcolano al più ogni 3 s (0,05-0,2 ms).
- **Preferenze** (`localStorage`, sempre in try/catch): `nav.dragMode`, `nav.padCollapsed`, `nav.tipSeen`
  (il suggerimento di benvenuto compare una volta sola). `prefers-reduced-motion`: voli di al massimo 0,3 s e
  niente inerzia dopo il trascinamento (`enableDamping` spento).
- `SceneHost` aggiunge `host.onResize(fn)` e `host.snapshot()`; lo smorzamento delle OrbitControls è adattato
  al frame rate reale (`frameDamping`), così con frame lenti la scena non "scivola" più a lungo.

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
ui.player.on("step", (ctx) => …);   // anche "state", "end", "restart" (⟲ Ricomincia)
```
`enter` deve impostare lo stato di inizio passo (così "passo indietro" funziona). `ctx.data` è condiviso
tra i passi di un'azione. La barra di controllo (play/pausa, avanti/indietro, velocità) è già pronta.
`onStop` (azione chiusa o interrotta) deve riportare la scena allo stato di riposo; `onEnd` decide se
lasciare il risultato finale. Un tab che si nasconde ferma il suo `SceneHost`, quindi l'azione si
congela e riprende al ritorno (RAG, Immagini e LLM moderni invece la chiudono in `deactivate`).

### Layout condiviso (`css/core.css`)

- Desktop: sidebar 380 px a sinistra; in alto a sinistra ☰ e selettore dello zoom, in alto a destra la
  barra strumenti del tab (`.stage-toolbar`) e "Grafici" (tra 901 e 1280 px il selettore perde l'etichetta
  "Zoom:" e la barra strumenti si compatta, così non si coprono); codice a destra (sotto i grafici se aperti);
  player in basso al centro, sopra il codice (z-index 16), centrato nella parte libera se i grafici
  sono aperti; durante un'azione il codice si accorcia per non coprire il player.
- Schermi ≤ 900 px: la sidebar è un cassetto (si chiude toccando la scena); prima fila ☰, barra
  strumenti, "Grafici"; seconda fila il selettore dello zoom (regole in `core.css`: i tab non le ripetono); con la sidebar aperta i comandi dello
  stage si nascondono. Durante un'azione il codice sale sotto il selettore dello zoom e il player
  resta in basso; un tab che inquadra la scena sapendo che il codice è in basso aggiunge
  `.code-stays-bottom` alla sua radice (lo fa il Transformer).
- Il selettore delle rappresentazioni (`.rep-switch`) sta in alto nello stage nel primo posto libero (vedi
  "Rappresentazioni") e scrive `--rep-safe-top` sulla radice del tab; con una vista alternativa attiva i
  comandi del tab sono nascosti dalla classe `.rep-alt-active` sulla radice.
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

### Rappresentazioni (`core/representations.js`, `core/ui/repSwitch.js`, `css/reps.css`)

Ogni tab può mostrare lo stesso sistema in tre modi, scelti con un selettore in alto nello stage
("Architettura · Rete neurale · Cervello"): **Architettura** è la scena del tab (com'è costruito),
**Rete neurale** i neuroni disposti da sinistra a destra con la profondità (token, griglia, esperti),
**Cervello** le regioni e i neuroni che hanno ispirato l'architettura. Il selettore mostra solo le voci
che il tab dichiara (con la sola Architettura non compare).

```js
import { installRepresentations } from "../../core/representations.js";
const reps = installRepresentations(ui, {
  tabId: "rag", app,                                                          // app: passato alle viste
  network: { load: () => import("./network.js").then((m) => m.networkSpec) }, // vista del nucleo (livello)
  //   oppure personalizzata, sulla scena del tab (Transformer): { enter(options), exit(), focus?(target) }
  brain: { load: () => import("./brain.js").then((m) => m.brainSpec) },       // livello + pannello "Cervello"
  onChange: (id, previous) => {},                                             // anche reps.on("change", fn)
});
await reps.show("brain", { focus: ["hippocampus"] });   // → true a vista pronta (false: superata o errore)
reps.current;            // "main" | "network" | "brain"
reps.has("network");     // la voce esiste in questo tab
reps.goto({ kind, id }); // link diretti: view:rete | view:cervello → Promise<true>; altro → null (vedi sotto)
reps.sync("network");    // modalità personalizzata: il tab ha cambiato vista da solo
reps.view; reps.stack;   // la vista viva del livello e la sua pila { host, director, nav } (test, console) o null
reps.dispose();          // toglie tutto e rimette i metodi presi in carico (le pagine di prova)
```

| Opzione | Uso |
|---|---|
| `tabId` | id del tab (predefinito: `data-tab` della radice); dà anche il nome alle foto (`scena-rag-rete-….png`) |
| `app` | l'oggetto `app` del guscio: arriva alle viste (`ctx.app`) e al pannello Cervello |
| `main: { label, description }` | testi della voce Architettura (predefiniti: "Architettura" e una frase) |
| `network` / `brain: { load }` | `load()` → Promise della specifica di soli dati; la vista è del nucleo (`core/neuralNet/view.js`, `core/brain/view.js`), caricata alla prima apertura in un livello sovrapposto |
| `network` / `brain: { enter, exit, focus? }` | modalità personalizzata: la vista è del tab, sulla sua scena (il Transformer per la Rete neurale); il nucleo chiama solo `enter(options)`/`exit()` |
| `label`, `description` | in ogni voce: etichetta breve e frase del suggerimento |
| `onChange(id, previous)` | a ogni cambio di rappresentazione (anche da `sync`, dagli errori, dal ritorno automatico) |
| `loaders: { network, brain, pane }` | **solo per le pagine di prova**: sostituiscono i moduli veri (ognuna → Promise della funzione `create…View` / `renderBrainPane`) |

`installRepresentations` scrive anche `ui.reps` (la stessa API). Chiamala dopo aver creato i pannelli
del tab e **prima** di aggiungere "Fonti": così il pannello "Cervello" sta prima delle fonti.

**Il livello alternativo.** Una voce con `load` vive in `.rep-layer`, un livello dentro lo stage con la
sua pila di scena (`createSceneStack` di `layout.js`: `SceneHost`, `CameraDirector`, `Navigation`,
`NavPad`), quindi la navigazione libera è identica a quella dei tab (pad, aiuto, "Centra", foto, schermo
intero, tasti, touch). Mentre il livello è attivo:
- la scena del tab è **ferma e nascosta** (canvas compreso), la sua navigazione spenta
  (`ui.nav.setEnabled(false)`), l'azione in corso chiusa (`player.stop()`), un volo in corso portato
  subito alla meta e l'inerzia del trascinamento applicata: al ritorno la camera è ferma;
- i comandi del tab sono nascosti dalla classe `.rep-alt-active` sulla radice (`visibility: hidden`:
  selettore dello zoom, barra strumenti, "Grafici" e il pannello, codice, player, pad, didascalia e ogni
  altro elemento dello stage o della radice); restano la sidebar, ☰ e il selettore;
- tornando all'Architettura tutto torna **esattamente** com'era, camera del tab compresa (posizione e
  punto guardato salvati all'entrata), navigazione accesa solo se lo era.

Un solo livello vivo per tab. Lasciandolo, o nascondendo il tab, vista, navigazione e renderer si
liberano subito (`view.dispose()`, `nav.dispose()`, `host.dispose()`); il contesto WebGL si perde
(`forceContextLoss`) 2,5 s dopo (`RELEASE_DELAY_MS`), o subito se intanto si apre un altro livello.
Perché: subito dopo l'ultimo frame `loseContext()` blocca la pagina finché la GPU non finisce il lavoro
rimasto (misurato con il rendering software: 600-940 ms subito, 3-10 ms dopo 2-3 s); con una scena
pesante l'evento `webglcontextlost` può arrivare qualche secondo dopo la chiamata. Modulo e specifica
restano in memoria (riaprire è rapido) e, alla riapertura, la camera torna dov'era e "Centra" sulla
vista iniziale di prima.

Il nucleo prende in carico `start`/`stop` dello `SceneHost` del tab (i tab continuano a chiamarli in
`activate`/`deactivate`, niente da cambiare): con un livello attivo `host.start()` riapre il livello e
lascia ferma la scena del tab, `host.stop()` libera anche il livello.

**Ritorno automatico all'Architettura** quando lo studente chiede qualcosa della scena del tab mentre un
livello la copre:
- un'azione: `player.start(id)` (pannello Azioni, `rag:query`, link `action:…`) torna all'Architettura e
  poi parte;
- un volo della camera del tab **chiesto dallo studente**: `director.focus/flyTo` entro la finestra di
  `director.noteUserIntent` (clic su un passo del Percorso, un pulsante della sidebar, una vista; ↑ ↓
  anche con il focus sulla pagina), oppure dopo un link diretto o un risultato della ricerca verso questo
  tab (`app.goto` scrive `#<tab>/go=…` prima di chiamare il `goto` del tab);
- i voli **non chiesti** dallo studente (timer, continuazioni asincrone) si scartano senza muovere
  niente: al ritorno nessun movimento a sorpresa (anche il riquadro di evidenziazione resta com'era);
- i gesti sul livello, sul selettore, su ☰ e sulle schede della sidebar non contano come richieste per la
  scena del tab.

**Contratto delle viste del nucleo** (`core/neuralNet/view.js`, `core/brain/view.js`, importate
dinamicamente da `representations.js`):

```js
export function createNetworkView(ctx) { /* oppure createBrainView */ }
// ctx = { host, director, nav, layer, tabId, spec, app, reps, memory }
//   host/director/nav: la pila del livello (da usare, non da creare); layer: l'elemento .rep-layer
//   (ci sono già canvas, pad e aiuto: aggiungi, non svuotare); spec: la specifica del tab;
//   reps: l'API qui sopra (es. reps.show("main")); memory: oggetto che sopravvive alle riaperture
//   (facoltativo da usare: il cervello ci tiene scala, opacità e taglio)
// → { ready: Promise<void>, focus?(target), onShow?(), onHide?(), dispose() }
```
- `ready` si risolve a scena costruita: fino ad allora il livello mostra "Preparo la scena…".
  `onShow()` arriva dopo `ready`, `onHide()` prima di `dispose()`; `dispose()` può arrivare anche prima
  di `ready` (lo studente ha già cambiato vista). Un errore (anche in `create…View`) dà un avviso
  leggibile e riporta all'Architettura.
- `focus(target)` riceve il `focus` di `show` (es. id delle regioni per il cervello, id di una sezione
  per la rete). Con un `focus` la camera non viene rimessa dove lo studente l'aveva lasciata.
- Impostare `nav.setHome(fn)` ("Centra") e, con `InstancedMesh`, `nav.setBounds(box)` e inquadrature su
  un `Box3` esplicito: `meshBounds` di `director.js` conta un `InstancedMesh` come una sola istanza.
- In alto a sinistra ci sono ☰ e spesso il selettore: i comandi della vista in alto stanno sotto
  `top: var(--rep-safe-top, 56px)` (scritta dal selettore sulla radice del tab; la usano `.nn-card` e
  `.br-card`); in alto a destra c'è posto almeno quanto il pulsante "Grafici". Il livello ha un contesto
  di sovrapposizione suo (`z-index: 11`): qualunque `z-index` interno resta sotto ☰ (12) e il selettore
  (13, 24 con suggerimento o menu aperti); l'aiuto della navigazione del livello, aperto, sale sopra tutto.

**Modalità personalizzata (Transformer).** Con `{ enter, exit }` niente livello: il nucleo chiama
`enter(options)` scegliendo la voce, `exit()` lasciandola (anche per aprire il Cervello, che poi è un
livello). Quando il tab cambia vista da solo (pannello Viste, un passo del Percorso, un'azione) chiama
`reps.sync(id)`: il selettore si aggiorna senza richiamare `enter`/`exit`, e un livello aperto si chiude.
```js
const reps = installRepresentations(ui, {
  tabId: "transformer", app,
  network: { enter: () => tab.views.activate("neuroni"), exit: () => tab.views.activate("architettura") },
  brain: { load: () => import("./brain.js").then((m) => m.brainSpec) },
});
tab.events.addEventListener("view", (e) => reps.sync(e.detail === "neuroni" ? "network" : "main"));
```

**Pannello "Cervello".** Con `brain.load` il nucleo aggiunge subito il pannello "Cervello" alla sidebar;
il contenuto si carica la prima volta che lo studente lo apre, con `renderBrainPane(container, spec,
{ onShow3D, app })` di `core/brain/pane.js` (solo DOM, niente three.js). I suoi pulsanti chiamano
`onShow3D()` ("Mostra il cervello in 3D") o `onShow3D(regionIds)` ("Mostra nel modello 3D") →
`reps.show("brain", { focus })`. Sul telefono il cassetto si chiude per mostrare la scena. Errore di
caricamento: messaggio e "Riprova".

**Link diretti e ricerca.** `REP_LINKS = { rete: "network", cervello: "brain" }`: il `goto` del tab inizia
con `reps.goto(target)`, che per `view:rete` / `view:cervello` mostra la rappresentazione e per ogni
altro elemento chiude subito il livello e restituisce `null` (il `goto` prosegue come sempre). Le voci
vanno anche in `search.js` del tab (`{ id: "rete", kind: "view", … }`, `{ id: "cervello", kind: "view", … }`).
```js
async goto(target) {
  const shown = reps.goto(target);
  if (shown) return shown;
  // … passi, azioni, viste del tab come prima
}
```

**Selettore** (`RepSwitch`): radiogroup accessibile (frecce, Home/Fine, tabindex mobile, focus visibile,
`aria-describedby` con la descrizione, suggerimento al passaggio del mouse o col focus da tastiera); con
le frecce la vista cambia quando lo studente si ferma su una voce (160 ms); Spazio e Invio sono della
voce (non mettono in pausa l'azione). Il posto si ricalcola solo a eventi di layout, mai per frame
(~0,2 ms), con la logica pura di `placeSwitch` (`tests/reps.test.mjs`), in quest'ordine: prima fila dei
comandi (allineata a ☰, al centro della parte libera) con icone ed etichette, poi solo icone, poi la
seconda fila a sinistra sotto ☰, infine un menu compatto (`menu`/`menuitemradio`). Gli ostacoli sono gli
altri comandi del tab (figli posizionati di stage e radice); durante una vista alternativa restano
quelli di prima, così il selettore non salta, e contano anche i comandi della vista (figli del livello);
se così non c'è posto, i pannelli alti del tab nascosti (codice, grafici) liberano la loro area. Sul
telefono, a cassetto aperto, il selettore sale in cima alla sidebar; il foglio dei grafici a tutto stage
lo copre come copre gli altri comandi. Con il selettore in seconda fila i chip della navigazione
(`.nav-hud`) scendono sotto di lui (`.rep-switch-low`). Chi calcola l'area libera dello stage conta
`.rep-switch` tra i comandi in alto (es. `TOP_OVERLAYS` di `tabs/transformer/actions/stage.js`).

Regole:
- dichiarare solo le voci che il tab ha davvero (una rete onesta sulla scala, analogie con le fonti);
- un tab non tocca `.rep-layer` né la pila del livello: parla con le viste solo attraverso le specifiche;
- gesti nel tab, azioni e link diretti tornano all'Architettura da soli; per ciò che il tab mostra nella
  sua scena su una richiesta che non passa di lì (un evento del bus, es. `transformer:view` mandato dal
  tab RAG) il tab chiama prima `reps.show("main")` (o `reps.sync` nella modalità personalizzata): un
  volo senza gesto né link viene scartato;
- nessun lavoro per frame a riposo (il nucleo non registra `onFrame`), `prefers-reduced-motion`
  rispettato (nessuna animazione del caricamento, voli brevi del `CameraDirector`).

Esempio completo (un tab con livello per la rete e per il cervello):
```js
// js/tabs/rag/index.js (estratto)
import { installRepresentations } from "../../core/representations.js";

export async function createTab(root, app) {
  const ui = createTabLayout(root, { code: true, charts: true, player: true, hint: HINT });
  // … scena, LOD, Percorso, Azioni …
  const reps = installRepresentations(ui, {
    tabId: "rag",
    app,
    network: { load: () => import("./network.js").then((m) => m.networkSpec) },
    brain: { load: () => import("./brain.js").then((m) => m.brainSpec) },
  });
  sidebar.addPane({ id: "fonti", title: "Fonti" }).append(renderSources());
  // …
  return {
    activate() { ui.host.start(); ui.host.resize(); },   // invariato: con un livello attivo lo riapre
    deactivate() { ui.player.stop(); ui.host.stop(); },  // invariato: libera anche il livello
    async goto(target) {
      const shown = reps.goto(target);                    // view:rete, view:cervello
      if (shown) return shown;
      return gotoInTheScene(target);                      // passi, azioni, livelli dello zoom
    },
  };
}
```

**Prove.** `tests/reps.test.mjs` (posizionamento puro del selettore). `tests/reps-demo.html`: un finto tab
con codice, grafici, player, zoom e barra strumenti, viste "rete" e "cervello" finte e pulsanti per
provare tutto (ritorno automatico, volo non chiesto, tab nascosto, errore, link, contatore dei contesti
WebGL vivi). Parametri: `?viste=vere` (i moduli veri del nucleo con le specifiche di riferimento, come in un
tab vero: la rete di `neuralNet/demoSpecs.js`, `&spec=llm` predefinita, `unet`, `audio`, `load` = prova di
carico, `prova` = la rete piccola della pagina; il cervello di `brain/demoSpec.js`, `&cervello=prova` per
quello piccolo; più il pannello Cervello), `?rete=vera` (solo la rete vera, piccola), `?rete=personalizzata`
(modalità `{ enter, exit }`), `?errore=modulo` (modulo della rete che non si carica). `window.demo` per gli
script.

### Rete neurale orizzontale (`core/neuralNet/`, `css/neuralnet.css`)

La rappresentazione "Rete neurale" di un tab: tutti i neuroni di una rete ridotta, disposti in
orizzontale con la **profondità vera**. Il tab scrive solo una specifica di soli dati
(`tabs/<id>/network.js` esporta `networkSpec`); il nucleo la valida, la dispone, genera i collegamenti
e la disegna.

- **x** = flusso, da sinistra a destra (ingresso → strati → uscita); una sezione può avere più lastre in x
  (parti di un blocco, canali di una griglia, esperti di un MoE).
- **y** = neuroni della sezione (in una griglia 2D: le righe della griglia).
- **z** = la profondità vera, su cui lavorano i meccanismi "intricati": i token (LLM, RAG, Transformer),
  le colonne della griglia spaziale (diffusione), i frame nel tempo (audio). Il token 0 sta in fondo,
  l'ultimo davanti. Tutte le sezioni usano lo stesso passo in z: la posizione 3 sta alla stessa z
  ovunque e il suo percorso si segue a occhio.

Di lato la rete si legge come un diagramma classico; girando la camera ("Guarda lo spessore") si vede lo
spessore con l'attention tra token, le convoluzioni tra vicini, l'instradamento verso gli esperti, il
pooling che schiaccia la profondità, i residui che saltano le sezioni.

| Modulo (`js/core/neuralNet/`) | Esporta | Ruolo |
|---|---|---|
| `spec.js` | `normalizeSpec(spec)`, `NetworkSpecError` (`.errors`), `describeSpecError`, `LIMITS`, `DEFAULT_DENSITY`, `sliceUnits` | validazione e valori predefiniti; TUTTI gli errori in una volta, con il percorso del campo (puro) |
| `layout.js` | `computeLayout(net)`, `SPACING`, `unitIndex`, `decodeUnit`, `sliceIndex`, `unitPosition`, `boxOf`, `takesPartInSlices` | posizioni x/y/z di ogni neurone, lastre fantasma dei blocchi ripetuti, parentesi, altezze dei residui (puro) |
| `activations.js` | `resolveValues(net)`, `simulateAttention`, `attentionWeights`, `computeRouting`, `MODE` | valori veri o simulati con seme, instradamento top-k degli esperti, pesi di attention (puro) |
| `links.js` | `buildLinks(net, values, limits)`, `LINK_LIMITS`, `ARC_KIND`, `pairDepths` | segmenti e archi per schema, deterministici, ordinati per priorità, con tetto (puro) |
| `model.js` | `buildNetworkModel(spec, {limits})` | tutto il modello in un colpo: `{net, layout, values, links, timings}` (puro) |
| `pickIndex.js` | `ScreenIndex` | indice sullo schermo per il neurone sotto il mouse (puro) |
| `roles.js` | `ROLES`, `PATTERNS`, `HEAD_COLORS`, `ACTIVATION`, `LINK_COLORS`, `AXIS_COLORS` | vocabolario e colori (soli dati) |
| `scene.js` | `NetworkScene`, `LOD_PX`, `sizeText` | la scena three.js (vedi "Uso dentro la scena di un tab") |
| `geometry.js`, `shaders.js`, `labels.js` | — | geometrie (un oggetto per tutta la rete), shader con uniform condivise, etichette in un atlante |
| `card.js` | `NetworkCard` | la scheda (solo DOM) |
| `view.js` | `createNetworkView(ctx)`, `VIEW_DIRECTIONS` | la vista per `core/representations.js` |
| `demoSpecs.js` | `LLM_SPEC`, `UNET_SPEC`, `AUDIO_SPEC`, `loadTestSpec()`, `DEMO_SPECS` | tre specifiche di riferimento commentate (DeepSeek-V3 ridotto, U-Net di Stable Diffusion, EnCodec) + la prova di carico |

#### La vista (per `core/representations.js`)

```js
const view = createNetworkView({ host, director, nav, layer, tabId, spec, app, reps });
await view.ready;          // scena costruita, camera di lato su tutta la rete. Specifica sbagliata:
                           // view.error (NetworkSpecError) e la scheda elenca gli errori; ready si risolve lo stesso
view.focus("attn");        // oppure ["a", "b"] o { section: "attn", slice: 3 }: vola lì (stesso angolo) e mostra la scheda
view.onShow(); view.onHide();   // onHide ferma l'onda e nasconde l'etichetta
view.dispose();            // scena, texture, DOM, listener; rimette home, confini e limiti della navigazione
view.stats                 // { buildMs, model: {spec, layout, values, links}, tickMs, updateMs, updates, pickMs, units, segments, arcs }
view.play(); view.side(); view.thick(); view.setSlice(t | null); view.setDensity(f); view.select(id); view.measureIdle(n)
```
`host`, `director`, `nav` sono quelli del livello: `nav.setHome` = "Di lato" su tutta la rete,
`nav.setBounds` = la rete con i suoi assi, `nav.setOrbitLimits({ turn: 85, tilt: 82, minDistance: 0.5 })`
(mai da dietro: il flusso si leggerebbe al contrario). Nel `layer` la vista mette la scheda (`.nn-card`, in
alto a sinistra sotto `--rep-safe-top`; chi la ospita altrove può spostarla con `--nn-top` e, sotto 640 px,
`--nn-top-narrow`) e l'etichetta del mouse (`.nn-tip`). La scheda è il primo figlio del livello (prima di
canvas e pad): con Tab dal selettore si arriva subito ai suoi comandi. La camera inquadra la rete nella
parte dello stage non coperta dalla scheda, a destra o sotto, dove la rete viene più grande (conta la forma:
una rete lunga e bassa sta meglio sotto una scheda ridotta); sul telefono sotto l'altezza massima della
scheda (52vh), così il dettaglio aperto da un tocco non la copre.

Interazione: passando sopra un neurone, etichetta con sezione, lastra/head/esperto, indice, posizione in
profondità (nome del token), valore ("simulato" se lo è); clic su un neurone, sull'etichetta di una sezione,
di un gruppo o di un "× N" = scheda con il suo testo, i numeri disegnati e veri e i collegamenti in entrata e
in uscita (la sezione si evidenzia, le altre si spengono); clic sul nome di un token = quella fetta; clic sul
vuoto = chiude. Sul telefono un tocco fa lo stesso e mostra l'etichetta per 2,6 s; il dettaglio resta visibile
anche nella scheda ridotta. Scheda: titolo, badge
"Attivazioni: simulazione didattica" (o "vere"), introduzione, comandi (**▶ Anima** e velocità, **Guarda lo
spessore** = tre quarti, **Di lato**, **Fetta** = una posizione della profondità e il suo percorso, densità
dei **Collegamenti**), dettaglio, legenda costruita da ciò che la rete contiene, "Scala e onestà", indice
delle sezioni (da tastiera si arriva a tutto), fonti. Sul telefono parte ridotta ("Dettagli" la apre).

#### Specifica (`networkSpec`, soli dati)

```js
export const networkSpec = {
  title: "…",                 // obbligatorio
  intro: "…",                 // 1-3 frasi
  scale: "…",                 // onestà sulla scala: "16 numeri per token invece di 7.168" (senza: avviso)
  note: "…",                  // facoltativa, in "Scala e onestà"
  seed: 7,                    // seme delle simulazioni e dei campioni (predefinito 1)
  axes: { x: "…", y: "…", z: "…" },   // testi degli assi (z predefinito: l'etichetta della profondità)
  depth: { count: 8, label: "token", names: ["Il", "gatto", …] },  // profondità predefinita (depth: true)
  //     oppure depth: 8   oppure depth: { grid: [8, 8], label: "griglia del latente" }
  sections: [ /* sezioni e ripetizioni, da sinistra a destra */ ],
  links: [ /* collegamenti */ ],
  groups: [{ label: "Encoder", from: "a", to: "b", text: "…", color: "#7dd3fc" }],  // parentesi sopra
  sources: [{ title: "…", url: "https://…", note: "…" }],           // solo http(s)
};
```

**Sezione** `{ id, label, role, neurons, depth, … }`:

| Campo | Valori | Note |
|---|---|---|
| `id` | lettere, cifre, `_ . : -` | unico; i collegamenti lo citano |
| `label`, `sub`, `text` | testo | `text` = scheda della sezione (testo semplice, paragrafi separati da una riga vuota); `sub` = seconda riga dell'etichetta (predefinita: la forma, es. "16 × 8 token") |
| `role` | `input embedding norm attention mlp router experts conv latent codebook add output data` | colore, legenda e attivazione predefinita (`ROLES`) |
| `neurons` | 1…4096 | neuroni per posizione; **in una griglia: i canali** (una lastra per canale) |
| `depth` | `false`/assente, `true`, `n`, `[righe, colonne]`, `{count, label, names}`, `{grid, label}` | nessuna, quella predefinita, n in fila, griglia (righe su y, colonne su z) |
| `slabs`, `slabNames` | 1…256 | più lastre in x (es. "key e value", "uscita"); non con una griglia né con gli esperti |
| `heads` | 1…64 | le head dividono i neuroni in fasce (spazio tra le fasce, un colore per head negli archi) |
| `experts`, `active`, `shared`, `router` | solo `role: "experts"` | lastre = `shared` condivisi + `experts` instradati; `active` per token; `router`: id della sezione router (un neurone per esperto) da cui prendere i top-k |
| `routes`, `gates` | `count × active` indici interi (e pesi ≥ 0) | instradamento vero, posizione per posizione |
| `real` | numero | neuroni per posizione nel modello vero ("16 di 129.280") |
| `activation` | `signed positive probs onehot` | come leggere i valori (colore e simulazione); predefinito dal ruolo |
| `values` | `Float32Array` o elenco di numeri, lunghezza `slabs × neurons × count` | **attivazioni vere**, ordine `[lastra][neurone][posizione]`; in una griglia posizione = riga × colonne + colonna; negli esperti prima i condivisi |
| `neuronNames`, `valueLabel`, `color` | | nomi dei neuroni (es. caratteri dei logits), unità del valore, colore al posto di quello del ruolo |

**Ripetizione** `{ repeat: N, of: ["a", "b", …], id?, label?, text? }`: il blocco (sezioni consecutive, subito
prima) si disegna UNA volta; seguono al più 4 lastre fantasma con il suo ingombro, la parentesi sotto e "× N"
(id predefinito `repeat-<k>`). Per esempio i 58 layer MoE di DeepSeek-V3: `{ repeat: 58, of: [...8 sezioni] }`.

**Collegamento** `{ from, to, pattern, density?, fromSlab?, toSlab?, label?, … }` (`within` solo per la
self-attention). Si esce dall'ultima lastra di `from` e si entra nella prima di `to` (`"all"` o un indice per
cambiare; negli esperti tutte). `density` (0…1) è il campione disegnato; `label` compare nella scheda.

| `pattern` | Cosa disegna | Campi | Densità predefinita |
|---|---|---|---|
| `dense` | ogni unità verso quelle di `to` alla **stessa posizione** (un nn.Linear per token); pesi simulati | `slice` (`"all"` o un indice) | 0,15 |
| `identity` | uno a uno, stessa posizione (norm, attivazioni, copie) | | 1 |
| `lookup` | per ogni posizione l'unità accesa di `from` (il token) verso tutto il vettore di `to` | profondità uguali | 1 |
| `attention` | **archi tra posizioni**, uno per (head, query, chiave), dalla lastra delle chiavi/valori alla lastra di uscita; colore = head, luce = peso | `causal`, `heads`, `weights` (`heads × query × chiave`, veri) | — (le 4 chiavi più forti per query, peso ≥ 4%) |
| `attention` con `from` ≠ `to` | **cross-attention**: dalle posizioni di `from` (key/value, es. il testo) alle posizioni di `to` (query, es. la griglia) | `heads`, `weights` | — |
| `route` | verso gli esperti (dispatch) o dagli esperti (combine): **ogni token solo verso i suoi top-k + i condivisi**; dal router il neurone *e* va solo all'esperto *e* | una delle due sezioni `role: "experts"` | 0,35 |
| `conv` | ogni posizione di `to` riceve dalla finestra `kernel × kernel` attorno alla stessa posizione | `kernel` (dispari, 3) | 0,3 |
| `pool` | tutte le posizioni → una (profondità schiacciata) | `to` senza profondità | 1 |
| `broadcast` | una → tutte le posizioni | `from` senza profondità | 1 |
| `down` / `up` | la griglia (o la fila) si rimpicciolisce / si ingrandisce di un fattore intero | `factor` (ricavato dalle dimensioni) | 1 |
| `residual` | un arco per posizione **sopra le sezioni** saltate, all'altezza decisa dal layout (archi annidati salgono) | `from` prima di `to` | — |
| `explicit` | **coppie e pesi veri**: colore = segno, luce = intensità | `pairs` (`{from, to, weight}` o `[[from, to, peso]…]`, indici dentro la fetta = lastra × neuroni + neurone), `slice` (`"selected"` = sulla fetta scelta, predefinito; `"all"`; un indice) | 1 |

Regole:
- **Onestà**: `scale` dice di quanto è ridotta la rete; senza `values` le attivazioni sono simulate con il seme
  e la vista lo dichiara ("simulazione didattica"); i numeri del modello vero (`real`, `sub`, `text`) vengono
  dalle fonti in `sources`. I testi sono testo semplice (mai HTML), senza emoji.
- **Numeri veri controllati prima della copia**: `values`, `weights`, `routes`, `gates` e `pairs` devono
  essere numeri (interi dove servono indici): un `null` (es. un NaN passato da `JSON.stringify`), una
  stringa o un decimale in un indice sono errori con il percorso del campo, non zeri silenziosi.
- **Limiti** (`LIMITS`, `LINK_LIMITS`): 400 sezioni, 4.096 collegamenti, 250.000 neuroni, 4.096 per
  posizione, 1.024 posizioni, 160.000 segmenti dritti (oltre, tutte le densità si scalano dello stesso
  fattore, detto in "Scala e onestà"), 9.000 archi. Prova di carico: `loadTestSpec()` = 63.232 neuroni,
  131.123 segmenti, ~3.500 archi.
- **Fette**: le posizioni dell'asse z della profondità predefinita. Una sezione con più (o meno) posizioni si
  abbina in proporzione (es. 24 istanti prima di una conv con passo 2 ↔ 12 frame); le sezioni con una
  profondità di altro tipo (i token del prompt in una U-Net a griglia) non si accendono con la fetta.
- **Determinismo**: stessa specifica, stessi numeri e stessi collegamenti (generatori con seme per sezione e
  per collegamento: aggiungere una sezione non cambia le altre).
- **Prestazioni**: tutti i neuroni in un `THREE.Points` (disco nello shader), tutte le lastre-heatmap in una
  mesh con un atlante, i segmenti in un `LineSegments` ordinato per priorità (la densità è un `drawRange`),
  gli archi in un altro, le etichette in un `InstancedMesh` (vedi **Etichette**). LOD per sezione: sotto ~3 px
  tra neuroni vicini la sezione diventa una lastra con la heatmap (un texel per neurone). Onda, fetta, enfasi e LOD passano da
  uniform e da una texture di stato: nessun buffer si riscrive. A riposo, per frame, solo il confronto della
  matrice della camera (~1-2,6 µs, nessuna allocazione, 0 aggiornamenti a camera ferma). Misurato con il
  rendering software (1600×900): reti d'esempio 7-15 ms per frame, prova di carico 25-26 ms di lato e di tre
  quarti, 47 ms da vicino; costruzione 40-70 ms (esempi), ~120 ms (prova di carico); neurone sotto il mouse
  ~5 µs.
- **Etichette** (`labels.js`): ogni etichetta ha un'altezza nel mondo e un minimo e un massimo in pixel (riga
  di 14-20 px per le sezioni, cioè titolo di 11-16 px; 15-21 i gruppi, 19-30 il "× N", 12,5-16 token e assi):
  da lontano non scendono sotto una dimensione leggibile e, già al minimo, mostrano solo il titolo (la seconda
  riga, es. "128 head · qui 4", torna avvicinandosi). Quando due si toccano vince la più importante ("× N",
  gruppi e assi, sezioni, nomi dei token); i titoli dei gruppi possono salire di due altezze e quelli delle
  sezioni di una (`nudge`) prima di sparire, così la vista d'insieme dice sempre "Layer 1-3 … × 3 … × 58".
  Evitano i comandi DOM sopra il canvas (scheda, pad, chip della navigazione, ☰, selettore): `view.js` li passa
  a `NetworkScene.update(…, obstacles)` come rettangoli dello stage. Tutto solo quando la camera si muove o
  la scheda cambia dimensione (`ResizeObserver`): ~0,3 ms con la prova di carico.

#### Esempio completo

```js
// js/tabs/<id>/network.js — solo dati (lo carica representations.js con import dinamico)
export const networkSpec = {
  title: "Rete neurale di un piccolo encoder di frasi (esempio)",
  intro: "Ogni token della frase attraversa la rete alla sua profondità; l'attention mescola i token tra loro, il pooling li riassume in un solo vettore.",
  scale: "Esempio di formato: 12 numeri per token, 6 token. I numeri del modello vero vanno presi dalle sue fonti.",
  seed: 4,
  axes: { y: "numeri del vettore", z: "token della frase" },
  depth: { count: 6, label: "token", names: ["come", "funziona", "il", "chunking", "?", "[SEP]"] },
  sections: [
    { id: "tok", label: "Token", role: "input", neurons: 10, depth: true, activation: "onehot", text: "Un neurone acceso per token: la sua voce del vocabolario." },
    { id: "emb", label: "Embedding", role: "embedding", neurons: 12, depth: true, text: "La riga della tabella di ogni token." },
    { id: "attn", label: "Self-attention", role: "attention", neurons: 12, heads: 3, slabs: 2, slabNames: ["valori", "uscita"], depth: true, text: "Ogni token guarda tutti gli altri, prima e dopo: niente maschera causale." },
    { id: "add1", label: "+ residuo", role: "add", neurons: 12, depth: true, text: "x + attention(x)." },
    { id: "ffn", label: "Feed-forward", role: "mlp", neurons: 24, depth: true, activation: "positive", text: "Ogni token per conto suo." },
    { id: "add2", label: "+ residuo", role: "add", neurons: 12, depth: true, text: "x + feed-forward(x)." },
    { repeat: 6, of: ["attn", "add1", "ffn", "add2"], id: "enc_x6", label: "× 6 layer", text: "Sei layer uguali: ne vedi uno." },
    { id: "pool", label: "Vettore della frase", sub: "media dei token", role: "latent", neurons: 12, depth: false, text: "La media sui token: la profondità si schiaccia in un vettore solo." },
  ],
  links: [
    { from: "tok", to: "emb", pattern: "lookup" },
    { from: "emb", to: "attn", pattern: "dense", toSlab: 0, density: 0.3 },
    { within: "attn", pattern: "attention" },                       // bidirezionale: nessuna maschera
    { from: "attn", to: "add1", pattern: "dense", density: 0.2 },
    { from: "emb", to: "add1", pattern: "residual" },
    { from: "add1", to: "ffn", pattern: "dense", density: 0.15 },
    { from: "ffn", to: "add2", pattern: "dense", density: 0.15 },
    { from: "add1", to: "add2", pattern: "residual" },
    { from: "add2", to: "pool", pattern: "pool", label: "mean pooling" },   // la profondità si schiaccia
  ],
  groups: [{ label: "Encoder", from: "attn", to: "enc_x6" }],
  sources: [{ title: "Vaswani et al. (2017), Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762", note: "Il blocco transformer." }],
};
```
(576 neuroni, 1.085 segmenti, 64 archi.) Esempi grandi e realistici, con le fonti: `demoSpecs.js`.

**Valori e pesi veri** (Transformer, Laboratorio pesi):
```js
// attivazioni vere di una sezione { neurons: C, depth: T }: values[n * T + t] = attivazione del neurone n al token t
{ id: "ln1", label: "LayerNorm 1", role: "norm", neurons: C, depth: true, values: float32 }
// pesi veri di una nn.Linear W (out × in) tra due sezioni, disegnati sulla fetta scelta dallo studente
const pairs = { from: [], to: [], weight: [] };
for (let o = 0; o < OUT; o++) for (let i = 0; i < IN; i++) { pairs.from.push(i); pairs.to.push(o); pairs.weight.push(W[o * IN + i]); }
{ from: "ln1", to: "q", pattern: "explicit", pairs }              // slice: "selected" (predefinito)
// attention vera: weights[(h * T + query) * T + chiave] (righe che sommano a 1, futuro a zero)
{ within: "attn", pattern: "attention", causal: true, heads: H, weights: float32 }
```

#### Uso dentro la scena di un tab (es. "A righe / Orizzontale con profondità" del Transformer)

```js
import { buildNetworkModel } from "../../core/neuralNet/model.js";
import { NetworkScene } from "../../core/neuralNet/scene.js";
const model = buildNetworkModel(spec);        // puro: lancia NetworkSpecError con tutti gli errori
const net = new NetworkScene(model);          // { labels: false } per una rete senza etichette
tab.ui.host.scene.add(net.root);              // net.root si può spostare/scalare: LOD e mouse ne tengono conto
// quando la camera è cambiata (MAI a ogni frame a riposo: confronta la matrice della camera)
net.update(camera, stage.clientWidth, stage.clientHeight, renderer.getPixelRatio(), obstacles?);   // obstacles: [[x0, y0, x1, y1]…] da lasciare liberi
net.setSlice(t | null); net.setEmphasis(["attn"] | null); net.setDensity(0.4); net.setWave(x | null); net.setGlow(false);
const u = net.pick(event.clientX, event.clientY, camera, canvasRect);   // unità o -1 (µs; indice rifatto dopo un movimento)
net.describe(u);                              // { section, slab, neuron, depth, row, col, depthName, head, expert, value, simulated }
net.setSectionValues("ln1", float32);         // nuovi valori veri della stessa forma (un altro prompt): ricolora senza ricostruire
net.box(ids?) / net.framePoints(ids?);        // per inquadrare (THREE.Box3 / punti nel mondo)
net.dispose();
```
Dopo `setSlice`/`setEmphasis` si mette `net.needsUpdate = true` per ridisegnare le etichette al frame dopo.
Per cambiare la struttura (altre sezioni, altri pesi di attention) si ricostruisce: `buildNetworkModel` + una
nuova `NetworkScene` (~10-60 ms per le reti dei tab, ~120 ms per la prova di carico).

**Prove.** `tests/neuralnet.test.mjs` (specifica ed errori, disposizione, blocchi ripetuti, attivazioni e
instradamento, collegamenti deterministici, indice sullo schermo, esempi). `tests/neuralnet-demo.html`
(`?spec=llm|unet|audio|load|broken`, costi misurati nella sidebar, `window.__nnDemo`).

### Cervello (`core/brain/`, `css/brain.css`)

Ogni tab può affiancare alla sua architettura le **analogie con il cervello**: un pannello "Cervello" nella
sidebar (sempre disponibile, solo DOM) e un **modello 3D facoltativo**, la rappresentazione "Cervello" del
selettore (vedi "Rappresentazioni"). Il tab scrive solo una specifica di dati (`tabs/<id>/brain.js` esporta
`brainSpec`); pannello, scena, etichette, scheda e animazioni sono del nucleo.

Il modello 3D ha due scale nella stessa scena:
- **Regioni**: il cervello intero (emisferi con giri e solchi, scissura laterale e solco centrale, lobi,
  cervelletto con le folia, tronco encefalico, ippocampo, amigdala, talamo, striato, corpo calloso, VTA e
  substantia nigra, occhi con la via ottica, coclee con il nervo acustico), le regioni del tab evidenziate
  in arancione, i nuclei come marcatori, i percorsi come tubi luminosi, etichette con linee guida.
  Corteccia opaca o trasparente (cursore), piani di taglio sagittale / coronale / assiale con la sezione a
  colore pieno, colori dei lobi. Clic su qualunque punto: che cos'è, che cosa fa, coordinate MNI e fonte,
  analogie del tab che lo citano.
- **Neuroni**: il microcircuito del tab disposto **da sinistra a destra come la rete AI**, gruppi in colonne,
  neuroni modellati (soma, dendriti che si assottigliano, spine, assone con mielina e nodi di Ranvier,
  bottoni), fibre colorate per tipo di sinapsi, per ogni colonna il componente AI corrispondente. Il
  cervello si rimpicciolisce sopra il circuito e ogni colonna è legata da una linea alla sua regione: nel
  passaggio il circuito cresce dalla sua regione mentre la camera vola. "Anima impulsi": i gruppi scaricano
  in ordine, l'impulso salta tra i nodi di Ranvier e i bottoni si accendono all'arrivo.

Etichetta di onestà sempre visibile: "Modello stilizzato: forme e posizioni approssimate (coordinate MNI)";
nella scala Neuroni anche "neuroni generati al computer con forme tipiche, non in scala tra i tipi".

```js
// js/tabs/<id>/index.js: nessun import del cervello, basta la voce in installRepresentations
const reps = installRepresentations(ui, {
  tabId: "rag", app,
  brain: { load: () => import("./brain.js").then((m) => m.brainSpec) },   // pannello + livello 3D
});
await reps.show("brain", { focus: ["hippocampus", "perforant_path"] });   // vola lì e fa pulsare
await reps.show("brain", { focus: ["circuit:ca3"] });                      // scala Neuroni, gruppo ca3
```

| Modulo | Esporta | Uso |
|---|---|---|
| `view.js` | `createBrainView(ctx)`, `VIEW_DIRECTIONS`, `DEFAULT_OPACITY`, `detectQuality(renderer)` | la vista del livello (la importa `core/representations.js`) |
| `pane.js` | `renderBrainPane(container, spec, { onShow3D, app })` | il pannello della sidebar, solo DOM (si carica senza three.js: `tests/brain-pane.test.mjs`) |
| `spec.js` | `normalizeBrainSpec(spec, { strict })`, `BrainSpecError`, `EDGE_KINDS`, `STRENGTHS`, `SPEC_LIMITS`, `safeUrl`, `normalizeSource` | validazione e normalizzazione della specifica (pura) |
| `atlas.js` | `REGIONS`, `MARKERS`, `PATHWAYS`, `ATLAS_IDS`, `atlasEntry(id)`, `atlasKind(id)`, `placements(entry)`, `mniToScene(p)`, `sceneToMni(p)`, `mirrorMni`, `formatMni`, `shortName`, `LOBES`, `MM` | l'atlante del nucleo (puro, vedi sotto) |
| `morphology.js` | `NEURON_TYPES`, `resolveNeuronType(name)`, `buildMorphology(type, seed, { detail })`, `SEG`, `SPH` | morfologie procedurali dei neuroni (pure) |
| `circuitLayout.js` | `layoutCircuit(circuit, { detail })`, `DISPLAY_SCALE`, `FAMILY`, `FAMILY_COLORS`, `SPACING`, `FIRE_STEP` | disposizione del microcircuito (pura) |
| `shape.js`, `structures.js`, `geometry.js`, `noise.js` | `hemisphereSdf`, `surfaceNets`, `buildHemisphereBase`, `buildCortex`, `buildCerebellum`, `lobeOf`, `SULCI`…; strutture profonde; curve e tubi; rumore con seme | la forma del cervello in mm MNI (pura, provata in Node) |
| `anatomy.js` | `loadAnatomyData(quality)`, `BrainModel`, `CUTS`, `QUALITY` | il cervello three.js (dati in cache per tutta la pagina) |
| `regions.js`, `pathways.js`, `labels.js`, `neurons.js`, `circuit.js`, `card.js`, `shaders.js` | `RegionLayer`, `PathwayLayer`, `LabelOverlay`, `NeuronField`, `CircuitScene`, `BrainCard`, materiali | pezzi della vista |
| `demoSpec.js` | `demoBrainSpec` | specifica d'**esempio** (via uditiva, dati veri della ricerca): pagina di prova e test |

**La specifica del tab** (`brainSpec`, soli dati; nessun HTML: tutto è mostrato come testo):

```js
export const brainSpec = {
  title: "Il RAG e la memoria del cervello",
  intro: "Due o tre frasi: che cosa c'entra il cervello con questa architettura.",
  regions: ["hippocampus", "entorhinal", "pfc_dl"],       // evidenziate (id dell'atlante; anche marcatori)
  markers: ["pulvinar"],                                   // facoltativo: nuclei come punti
  pathways: ["perforant_path",                             // id dell'atlante (curva anatomica già pronta)…
             { id: "mio", label: "Collegamento", through: ["a1", "stg"], color: "#2dd4bf" }],  // …o nuovo
  circuit: {
    title: "Circuito trisinaptico dell'ippocampo",
    text: "Come scorre il segnale, in parole semplici.",
    caveat: "Dove il paragone con l'AI si ferma.",
    groups: [                                              // da sinistra a destra, come la rete AI
      { id: "ec", label: "Corteccia entorinale", region: "entorhinal", type: "stellate", count: 8,
        real: 650000, realText: "circa 650.000 nello strato II", ai: "Embedding della domanda" },
      { id: "dg", label: "Giro dentato", region: "hippocampus", type: "dentate", count: 12, ai: "Chunk con id univoci" },
    ],
    edges: [{ from: "ec", to: "dg", kind: "excitatory", label: "via perforante", pathway: "perforant_path", note: "…" }],
    sources: [{ title, authors, year, venue, url }],
  },
  analogies: [{
    id: "rag-indice", ai: "Indice HNSW", brain: "Indice ippocampale",
    regions: ["hippocampus"], markers: [], pathways: [],   // cosa mostrare con "Mostra nel modello 3D"
    text: "…", similar: "In comune…", different: "Dove non regge…",
    strength: "forte" | "media" | "debole",
    sources: [{ title, authors, year, venue, url }],       // url solo https/http (gli altri non diventano link)
  }],
  facts: [{ claim: "Neuroni nel cervello", value: "86,1 ± 8,1 miliardi", note: "…", source: { title, url } }],
  sources: [{ title, authors, year, venue, url }],         // fonti generali (il pannello le unisce senza doppioni)
};
```

| Campo | Regole |
|---|---|
| id delle regioni | quelli dell'atlante (sotto); un id sconosciuto è un avviso (`spec.warnings`) e si ignora, con `strict: true` un errore |
| `regions` | le regioni evidenziate; se manca si evidenziano tutte quelle usate da analogie e circuito |
| `type` dei gruppi | `pyramidal`, `stellate`, `relay`, `basket`, `granule` (cervelletto), `purkinje`, `dentate` (giro dentato), `photoreceptor`, `ganglion`, `hair`, `bipolar`, `projection` (forma generica dei nuclei del tronco), `spiny`, `pallidal`, `dopaminergic`; sconosciuto → forma generica e avviso |
| `count` | neuroni disegnati per gruppo, 1-18 (`SPEC_LIMITS`); `real`/`realText` dicono quanti sono davvero |
| `kind` delle sinapsi | `excitatory` (arancione), `inhibitory` (blu, contatto sul soma come i canestri), `modulatory` (viola) |
| analogie | `different` è obbligatorio nello spirito della web app (manca → avviso); almeno una fonte |
| limiti | 12 gruppi, 40 sinapsi, 30 analogie: oltre si taglia, con un avviso in `spec.warnings` |

**Nomi della ricerca accettati** (così `brain_data.json` si copia senza tradurre): `neuron_type` (chiavi
italiane: `piramidale`, `stellata_spinosa`, `stellata_entorinale`, `rele_talamico`, `canestro_pv`,
`granulare`, `purkinje`, `granulo_dentato`, `fotorecettore`, `gangliare_retinica`, `cellula_ciliata`,
`ganglio_spirale`, `neurone_di_proiezione`, `spinoso_medio`, `neurone_gangli_base`, `dopaminergico`),
`count_shown`, `real_count`, `real_count_text`, `analogy_map: [{ group, ai_component }]`, `kind:
"eccitatoria" | "inibitoria" | "modulatoria"`, `strength: "debole/metafora"`, fonti con `doi`.

**L'atlante** (`atlas.js`): 31 regioni (`pfc_dl, pfc_vm, acc, broca, premotor, motor, somato, parietal_post,
angular, wernicke, a1, stg, temporal_ant, it, ffa, v1, v2v4, mt, insula, entorhinal, parahippocampal,
hippocampus, amygdala, thalamus, striatum, vta_snc, cerebellum, brainstem, corpus_callosum, retina,
cochlea`), 6 marcatori (`lgn, mgn, pulvinar, cochlear_nucleus, superior_olive, inferior_colliculus`) e 5
percorsi (`arcuate, optic_radiation, auditory_radiation, mesolimbic, perforant_path`). Ogni voce: nome
italiano, centroide MNI in mm (emisfero sinistro; il destro è lo specchio), raggio, lobo, emisfero (`left`
per le funzioni lateralizzate come Broca e Wernicke, `right` per l'area dei volti, `both`, `midline`), tipo
(corticale, profonda, periferica, nucleo, fascio), testo e fonte del centroide (`aal`: centroidi AAL,
Tzourio-Mazoyer et al. 2002, dalla tabella di BrainNet Viewer; `lit`: studi di risonanza funzionale citati;
`approx`: posizione anatomica approssimata, da prendere come indicativa). I percorsi hanno punti di
passaggio anatomici. `mniToScene([x, y, z])` = 1 unità per cm, rotazione propria (niente specchio), camera
predefinita sull'emisfero sinistro con il lobo frontale a sinistra (la vista dei libri).

**`createBrainView(ctx)`** (contratto delle viste del nucleo) → `{ ready, focus(target), onShow(), onHide(),
dispose() }`, più comandi per i test e la console: `setScale("regions" | "neurons")`, `setOpacity(0…1)`,
`setCut(axis | null, mm, flip)`, `toggle("pathways" | "labels" | "lobes", on)`, `play(on)`, `select(id)`,
`selectGroup(id)`, `home()`, `measureIdle(n)`, `stats`, `state`.
- `focus(target)`: id dell'atlante (regioni, marcatori, percorsi; stringa o elenco) → scala Regioni, volo
  dalla parte in cui le regioni si affacciano con una finestra di almeno 12 cm sul piano dello schermo (si
  vedono i lobi attorno; misurata dopo la proiezione perché una regione nei due emisferi ha il box lungo
  proprio verso la camera), impulso per 4,5 s, scheda della prima; se c'è una struttura
  profonda o un percorso la corteccia diventa trasparente da sola (32%). `"circuit"` / `"circuit:<gruppo>"`
  → scala Neuroni (e il gruppo). Id sconosciuti: ignorati. Chiamato prima di `ready` aspetta la scena
  (mai dopo `dispose`).
- `ctx.memory` conserva scala, opacità, taglio e interruttori: riaprendo la vista (anche dopo aver cambiato
  tab) tutto è come prima. La geometria del cervello è in cache per tutta la pagina: il primo cervello costa
  150-450 ms, gli altri tab lo riusano.
- Qualità automatica: `detectQuality` legge il renderer WebGL (prima `RENDERER`, l'estensione solo se è
  mascherato: niente avvisi in Firefox); con il rendering software corteccia più rada, neuroni "cotti" in un
  buffer solo, cervello in miniatura liscio. `ctx.quality` ("high" | "low") la forza (solo pagine di prova).
- Comandi in alto a sinistra sotto `--rep-safe-top` (la scheda è il primo figlio del livello: Tab dal selettore
  arriva subito lì); la camera inquadra nella parte libera dalla scheda (a destra sul desktop, sotto sul
  telefono). Sul telefono la scheda parte ridotta (al più 52vh); un tocco su una regione mostra il dettaglio
  anche da ridotta. Le etichette con le linee guida evitano, oltre al bordo e alle altre etichette, i comandi
  DOM sopra la scena (scheda, pad, chip della navigazione, ☰, selettore).
- Navigazione libera del nucleo, intera: `nav.setHome` (vista d'insieme della scala attuale, o la sezione
  se c'è un taglio), `nav.setBounds` (box esplicito), nessun limite di rotazione (è un oggetto da girare).
- Clic sulla scena: raycast della corteccia e delle strutture (le parti tolte dal taglio non contano); la
  regione più vicina entro il suo raggio, altrimenti il lobo. Neuroni: proiezione dei somi (niente raycast
  su migliaia di istanze), suggerimento al passaggio del mouse, sopra il dito sul telefono.

**`renderBrainPane(container, spec, { onShow3D, app })`**: introduzione, "Mostra il cervello in 3D"
(`onShow3D()`), "Guarda i neuroni in 3D" (`onShow3D(["circuit"])`), legenda della forza, una scheda per
analogia (AI ↔ cervello, testo, "In comune", "Dove non regge", forza, regioni, fonti con link
`target="_blank" rel="noopener noreferrer"`, "Mostra nel modello 3D" = `onShow3D([regioni, nuclei,
percorsi])`), il circuito (gruppi con il componente AI, connessioni, avvertenza), i numeri e tutte le fonti
senza doppioni. Restituisce `{ spec, root }`. Testi sempre con nodi di testo, link solo http(s).

Regole:
- dati veri: numeri e fonti dalla ricerca verificata (`brain_data.json`) o da fonti primarie; ogni analogia
  dice dove non regge; niente HTML nei testi;
- il tab non tocca la scena: parla con la vista solo con la specifica e con `reps.show("brain", { focus })`;
- forme e dimensioni dei neuroni sono "tipiche", non misure: la vista lo dichiara; non scrivere nei testi
  numeri di forma che la ricerca segna come "non verificato";
- prestazioni (SwiftShader, macchina carica): ciclo per frame a riposo ~0,4 µs (solo il confronto della
  matrice della camera, 0 aggiornamenti delle etichette a camera ferma), con gli impulsi ~1 µs; etichette
  quando la camera si muove ~0,3 ms; costruzione ~240-350 ms (bassa qualità) e circuito 40-110 ms; disegno
  ~40 ms la scala Regioni e ~28 ms la scala Neuroni; `prefers-reduced-motion` rispettato (niente
  transizione animata, voli brevi), `dispose()` completo (0 geometrie e 0 texture rimaste nel renderer).

Esempio completo (un tab con un'analogia e un circuito di tre gruppi):

```js
// js/tabs/rag/brain.js (soli dati: niente three.js, niente DOM)
export const brainSpec = {
  title: "Il RAG e la memoria del cervello",
  intro: "Un LLM con i pesi congelati più un archivio esterno somiglia alla coppia neocorteccia (lenta) e ippocampo (veloce).",
  regions: ["hippocampus", "entorhinal", "pfc_dl"],
  pathways: ["perforant_path"],
  circuit: {
    title: "Circuito trisinaptico dell'ippocampo",
    text: "Dalla corteccia entorinale al giro dentato, a CA3 e a CA1.",
    caveat: "Nessun ippocampo calcola distanze del coseno: il paragone riguarda il ruolo, non il calcolo.",
    groups: [
      { id: "ec", label: "Corteccia entorinale", region: "entorhinal", type: "stellate", count: 6, ai: "Embedding" },
      { id: "dg", label: "Giro dentato", region: "hippocampus", type: "dentate", count: 10, ai: "Id univoci dei chunk" },
      { id: "ca3", label: "CA3", region: "hippocampus", type: "pyramidal", count: 6, ai: "Ricerca HNSW" },
    ],
    edges: [
      { from: "ec", to: "dg", kind: "excitatory", label: "via perforante", pathway: "perforant_path" },
      { from: "dg", to: "ca3", kind: "excitatory", label: "fibre muschiose" },
      { from: "ca3", to: "ca3", kind: "excitatory", label: "collaterali ricorrenti" },
    ],
  },
  analogies: [
    {
      id: "rag-cls", ai: "LLM congelato + database vettoriale", brain: "Sistemi di apprendimento complementari",
      regions: ["hippocampus", "pfc_dl"], text: "…", similar: "…", different: "…", strength: "media",
      sources: [{ title: "Why there are complementary learning systems in the hippocampus and neocortex", authors: "J. L. McClelland, B. L. McNaughton, R. C. O'Reilly", year: 1995, url: "https://doi.org/10.1037/0033-295X.102.3.419" }],
    },
  ],
  facts: [],
  sources: [],
};
```

**Prove.** `tests/brain.test.mjs` (atlante completo, trasformazione MNI ↔ scena, emisferi e lobi, funzione
distanza, surface nets chiusa e orientata, corteccia deterministica, cervelletto, curve e tubi, rumore,
morfologie, specifica anche nel formato della ricerca, limiti, link sicuri, microcircuito);
`tests/brain-pane.test.mjs` (pannello con un DOM finto: niente three.js, testi come testo, link solo
http(s), focus dei pulsanti). `tests/brain-demo.html`: il pannello nella sidebar e la vista in un livello
`.rep-layer` come in `representations.js`; parametri `?spec=demo|regioni|sbagliata`, `?q=low|high`,
`?scale=neurons`, `?focus=hippocampus,a1`; `window.__brain` per gli script. Integrazione:
`tests/reps-demo.html?viste=vere`.

### Pesi e run (`js/shared/weights/`, `core/matrixNumbers.js`, `gpt/runs.py`, `gpt/benchmark.py`)

Libreria pura (niente DOM salvo la lettura dei `File`, niente three.js) per leggere, controllare,
descrivere e confrontare i pesi del GPT del corso e dei run caricati dallo studente, più le matrici
3D con i **numeri veri** scritti nelle celle e, in Python, gli strumenti che producono i run e i
benchmark. Riusa il motore (`js/engine/weights.js`: `decodeFloat32`, `decodeFloat16`,
`expectedParameters`, `assertSameLayout`; `js/engine/gpt.js`: `GPTEngine`) e `core/binary.js`
(`halfToFloat`).

| Modulo | Esporta | Uso |
|---|---|---|
| `shared/weights/formats.js` | `parseWeightsManifest`, `decodeWeights`, `tensorsFromLayout`, `parseTrace`, `decodeSnapshot`, `parseRunInfo`, `parseBenchmark`; riesporta `parseSafetensors`, `safetensorsInfo`, `recognizeFiles`, `readJsonFile`, `readBinaryFile`, `filePath`, `LIMITS`, `WeightsFormatError`, `canonicalName` | i formati (tabella sotto) |
| `shared/weights/safetensors.js` | `parseSafetensors(bytes, {file, limits})`, `safetensorsInfo(metadata)`, `SAFETENSORS_DTYPES` | `.safetensors` F32/F16/BF16 → `Float32Array` |
| `shared/weights/files.js` | `recognizeFiles(files, {limits})`, `readJsonFile`, `readBinaryFile`, `filePath` | file caricati → gruppi di run, senza leggerli |
| `shared/weights/validate.js` | `LIMITS`, `WeightsFormatError`, `checkLayout`, `checkShape`, `checkName`, `checkRelativePath`, `cleanConfig`, `checkVocab`, `canonicalName`, `naturalCompare` | controlli comuni (dati non fidati) |
| `shared/weights/run.js` | `WeightRun`, `loadBuiltinRun`, `runsFromFiles`, `engineCompatibility`, `inferGptConfig`, `snapshotBracket`, `snapshotEntry`, `fingerprint`, `BUILTIN_RUN_ID` | l'oggetto run |
| `shared/weights/stats.js` | `tensorStats`, `histogram`, `percentiles`, `range`, `absMax`, `l2Norm`, `DEFAULT_PERCENTILES` | statistiche di un tensore |
| `shared/weights/compare.js` | `compareTensors(a, b, {top, tolerance, trace, groupOf})`, `tensorDiff`, `difference` | confronto A/B (run o fasi del training) |
| `shared/weights/groups.js` | `groupOf`, `groupsFromTrace`, `grouper` | gruppi di parametri come `gpt/recording.py` |
| `shared/weights/moduleTree.js` | `buildModuleTree(params, {snippets, config})`, `findNode`, `walkTree`, `pythonPath`, `ROOT_CLASS`, `ROOT_SNIPPET` | albero dei moduli con la riga Python vera |
| `shared/weights/pythonSource.js` | `parseModuleSource`, `moduleSources`, `splitComment` | lettura ristretta degli snippet Python di `model.json` |
| `shared/weights/store.js` | `RunStore`, `serializeRun`, `deserializeRun`, `RECORD_FORMAT` | run caricati salvati nel browser (IndexedDB) |
| `shared/weights/display.js` | `formatCell`, `formatPrecise`, `chooseCellFormat`, `textColorFor`, `textContrastFor`, `contrastRatio`, `relativeLuminance`, `quantizeSigned`, `quantizeUnsigned`, `matrixShape`, finestre di celle (`windowFromPoints`, `capWindow`, `expandWindow`, …), `TEXT_DARK`, `TEXT_LIGHT` | come si scrivono i numeri nelle celle (puro) |
| `core/matrixNumbers.js` | `NumericMatrix`, `attachNumbers`, `NumbersGroup`, `colormapLUT` | matrici 3D con i numeri (three.js) |

#### Formati supportati

| File | Contenuto | Controlli (errori in italiano con file e campo) |
|---|---|---|
| `weights.json` + `.bin` | manifest di `gpt/export.py` (anche di `scripts/export_run.py`), float32 little-endian, layout `{name, shape, offset, numel}` in NUMERO DI FLOAT | `format_version` 1, `dtype` float32, `byte_order` little; `file` relativo (niente `/…`, `..`, `\`, `C:`, `https:`); nomi unici e senza caratteri di controllo; forme di interi ≥ 0 (al più 8 dimensioni); `numel` = prodotto della forma; intervalli dentro il file, **senza sovrapposizioni né buchi** (somma dei `numel` = `total_params`); binario lungo esattamente `4 × total_params` byte; `config` (solo valori semplici, interi positivi) e `vocab` (stringhe uniche, lunghezza = `vocab_size`) facoltativi. **Qualunque architettura**: la compatibilità con il motore si valuta a parte |
| `trace.json` + `snap_XXXXX.f16.bin` | storia del training di `gpt/recording.py` | `format_version` 1, `snapshot_dtype` float16, layout come sopra; snapshot `{step, file}` con step interi unici (ordinati) e file accanto a `trace.json`; ogni snapshot lungo esattamente `2 × total_params` byte (conversione esatta da float16). Le metriche (`train_loss`, `evals`, `metrics`, …) restano quelle del file |
| `run.json` | descrizione di un run esportato | campi letti, tutti facoltativi: `name`, `description`, `created` (o `date`/`created_at`), `source`, `config`, `vocab`, `notes` (stringa o lista), `modules` (o `module_tree`, tenuto com'è), `engine_compatible` (booleano: `false` = calcolo diverso dal GPT del corso anche con forme uguali); il JSON originale in `raw` |
| `benchmark.json` | metriche di `scripts/benchmark.py` | basta un oggetto JSON: `metrics` raccoglie i numeri di primo livello e di `metrics` (tranne `format_version`); il resto in `raw` |
| `*.safetensors` | formato di Hugging Face: u64 little-endian con la lunghezza dell'intestazione, intestazione JSON UTF-8, dati | dtype **F32, F16, BF16** (→ `Float32Array`, BF16 esatto); gli altri (I64, F64, F8, …, e i nomi che non sono dtype) danno "dtype X non supportato: sono supportati F32, F16 e BF16"; intestazione ≤ 16 MB e dentro il file, UTF-8 valido, oggetto JSON; `__metadata__` solo stringhe (`config` e `vocab` come JSON vengono letti); `data_offsets` interi, nei limiti, coerenti con forma e dtype; **nessuna sovrapposizione, nessun buco, nessun byte dopo l'ultimo tensore** (le regole della libreria ufficiale); shard `nome-00001-of-00003.safetensors` uniti in un run |

Sicurezza: i file caricati sono dati non fidati. Niente `eval`, nessun oggetto indicizzato con nomi presi
dai file (solo `Map`: un tensore chiamato `__proto__` o `constructor` è un nome qualunque), nessuna
ricerca con `in` sulle tabelle di costanti (un dtype `"toString"` è sconosciuto), limiti di dimensione
prima di allocare o leggere. I nomi dei tensori, dei file e i testi di `run.json` vanno mostrati come
testo (`textContent`, o `escapeHtml` se si usa `innerHTML`). `LIMITS` (sovrascrivibili con l'opzione
`limits` di ogni funzione): `maxJsonBytes` 64 MB, `maxBinaryBytes` 1 GB per file, `maxTotalBytes` 1,5 GB
per caricamento, `maxSafetensorsHeaderBytes` 16 MB, `maxTensors` 100.000, `maxDims` 8, `maxParams`
268 M valori, `maxNameLength` 512.

#### Riconoscimento dei file caricati

`recognizeFiles(files)` accetta `FileList` (input con `webkitdirectory` o `multiple`), array di `File` o
oggetti "tipo File" `{name, size, text(), arrayBuffer()}` con il percorso in `webkitRelativePath`,
`relativePath` o `path` (chi converte le voci del drag & drop; le barre rovesciate di Windows diventano
`/`). Guarda solo nomi, percorsi e dimensioni e restituisce `{runs, ignored, totalBytes}`:
- una cartella con `weights.json`, `run.json` o un `.safetensors` è un run; `trace.json` e
  `snap_*.f16.bin` vanno al run della cartella o della cartella antenata più vicina (come `data/` e
  `data/training/` del corso); senza antenati formano un run da soli (pesi = ultimo snapshot);
- sottocartelle che sono run (`runs/a/`, `runs/b/`) = più run; più `.safetensors` nella stessa
  cartella = run diversi (col nome del file), tranne gli shard;
- ignorati con il motivo: tipi sconosciuti, `model.json` (serve alla scena del Transformer), percorsi
  con `..`, file fuori da una cartella di run; `.DS_Store` e simili in silenzio;
- un file oltre il limite del suo tipo finisce in `run.errors` (quel run non si carica); oltre
  `maxTotalBytes` in tutto, `WeightsFormatError` prima di leggere qualunque cosa.

#### Run

```js
import { loadBuiltinRun, runsFromFiles } from "../../shared/weights/run.js";

const run = await loadBuiltinRun();            // "GPT del corso" (data/): una volta per pagina; trace subito, snapshot a richiesta
run.id;                 // "builtin:gpt-corso" | "upload:<impronta dei pesi>" (stesso run caricato due volte = stesso id)
run.name; run.description; run.source;         // "builtin" | "upload"
run.config;             // {vocab_size, block_size, n_embd, n_head, n_layer, dropout} (dichiarata o ricavata dalle forme)
run.vocab;              // token id → carattere (o null)
run.tensors;            // Map nome → {name, shape, numel, dtype ("float32"|"float16"|"bfloat16"), data: Float32Array}
run.params;             // [{name, shape, numel, dtype}] nell'ordine di named_parameters
run.totalParams;        // 209729
run.tensor("blocks.0.sa.heads.0.key.weight");  // accetta anche i nomi con module. / _orig_mod.
run.flat();             // tutti i valori in un Float32Array
run.trace;              // trace.json controllato (parseTrace) o null
run.snapshots;          // [{step, file, meta (loss, top1, sample_text…), available, layoutMatches, load(), raw()}]
run.steps;              // step degli snapshot di cui ci sono i pesi: [0, 50, …, 5000]
await run.loadSnapshot(500);                   // {step, meta, flat, tensors, layout} (in cache; un errore non resta in cache)
await run.weightsAt(750);                      // tra due snapshot: media pesata, {exact: false, from: [500, 1000], t}
run.benchmark;          // benchmark.json (parseBenchmark) o null
run.info;               // run.json (parseRunInfo) o null
run.meta.warnings;      // avvisi da mostrare (snapshot mancanti, prefissi tolti, vocabolario del corso usato, …)
run.compatibleWithEngine;  run.engine;         // {compatible, config, vocab, vocabSource, issues[], missing[], extra[], mismatched[]}
const engine = run.createEngine();             // GPTEngine (errore chiaro se non compatibile); per il corso anche engine.loadSnapshot
engine.setWeights(run.engineFlat((await run.loadSnapshot(1000)).tensors));   // uno snapshot nel motore

const { runs, errors, ignored } = await runsFromFiles(input.files, { defaultVocab: run.vocab });
// errors: [{run, message}] per i run che non si caricano (gli altri sì)
```

`compatibleWithEngine` è vero se il run ha **esattamente** i nomi e le forme che `js/engine/gpt.js` sa
eseguire: `expectedParameters` della config ricavata dalle forme (`token_embedding_table` (V, C),
`position_embedding_table` (block_size, C), `blocks.<i>` da 0 senza salti, `blocks.0.sa.heads.<h>`). Il
motore accetta qualsiasi `vocab_size`, `block_size`, `n_embd`, `n_head`, `n_layer` con `n_embd` divisibile
per `n_head`; fissi sono feed-forward 4·n_embd, LayerNorm pre-norm con eps 1e-5, ReLU, bias come in
`gpt/*.py`. Serve anche un vocabolario della dimensione giusta: un safetensors senza vocabolario usa
`defaultVocab` (con avviso) se ha la stessa dimensione. Forme uguali non garantiscono lo stesso calcolo
(es. GELU al posto di ReLU): `run.json` può dire `"engine_compatible": false`. Le config dichiarate in
disaccordo con le forme sono un problema in `issues`.

#### Statistiche, confronto, gruppi, albero dei moduli

```js
import { tensorStats, histogram, range } from "../../shared/weights/stats.js";
const s = tensorStats(t.data);   // {count, finite, nan, inf, mean, std (popolazione, come recording.py), min, max, absMax,
                                 //  absMean, l2, rms, nearZero: {threshold: 0.001, count, fraction},
                                 //  percentiles: {1, 5, 25, 50, 75, 95, 99} (lineari come numpy), approximate, histogram}
const r = range([a.data, b.data]);                       // intervallo comune per due istogrammi sovrapposti
histogram(a.data, { bins: 40, ...r });                   // {min, max, binWidth, counts, below, above, total}

import { compareTensors, difference } from "../../shared/weights/compare.js";
const c = compareTensors(await run.loadSnapshot(0), run, { top: 10, trace: run.trace });
// c.matched: [{name, nameB, shape, numel, group, relChange ‖A−B‖/‖A‖, cosine, maxAbsDiff, meanAbsDiff, changedFraction, normA, normB, diffNorm}]
// c.onlyA, c.onlyB, c.shapeMismatch (architetture diverse: si confronta ciò che si può, il resto si riporta)
// c.groups (ordine di trace.groups se c'è), c.totals, c.mostChanged, c.identical
difference(a.data, b.data);                              // A − B elemento per elemento (modalità differenza)

import { buildModuleTree, findNode, walkTree } from "../../shared/weights/moduleTree.js";
const tree = buildModuleTree(run.params, { snippets: spec.snippets, config: run.config });   // spec = data/model.json
tree.label;            // "GPTLanguageModel" (senza snippet: "modello")
tree.architecture;     // "corso" | "modificata" | "sconosciuta";  tree.unexplained, tree.mismatched: nomi
findNode(tree, "blocks.0.sa.heads.0.key.weight").code;
// {snippet: "head", file: "gpt/attention.py", line: 29, text: "self.key = nn.Linear(config.n_embd, self.head_size, bias=False)",
//  comment, type: "nn.Linear", param: "weight", role: "matrice W (uscite × ingressi): y = x · Wᵀ + b"}
findNode(tree, "blocks.0").code;          // {className: "Block", text: "class Block(nn.Module):", createdBy: {text: "self.blocks = nn.Sequential(…)"}}
findNode(tree, "blocks.0.sa.heads.0").code.buffers;   // [{name: "tril", statement: "self.register_buffer(\"tril\", …)"}]
walkTree(tree, (node, depth) => …);
```
Abbinamento del confronto per nome esatto, poi per nome canonico (senza `module.`/`_orig_mod.`); i gruppi
sono quelli di `gpt/recording.py` (`group_params` del trace se c'è, altrimenti `groupOf`, che per
architetture sconosciute raggruppa per sotto-modulo). Le righe dell'albero non sono scritte a mano:
`pythonSource.js` legge gli snippet di `model.json` (assegnazioni `self.<nome> = …` dentro `__init__`, anche
su più righe, elementi di `nn.Sequential(…)`, comprensioni, `register_buffer`) e un test JS controlla che
ogni riga annotata sia davvero quella riga del file `gpt/*.py`. Per un run con l'architettura modificata
l'albero si costruisce dai nomi; `codeMatch: false` segnala un parametro che la riga del corso non crea.

#### Salvataggio nel browser

```js
import { RunStore } from "../../shared/weights/store.js";
const store = new RunStore();
await store.save(run);       // {ok: true, id, bytes} | {ok: false, reason} (quota controllata prima; il GPT del corso non si salva)
await store.list();          // [{id, name, description, savedAt, bytes, totalParams, tensors, config, steps, hasTrace, hasBenchmark, compatibleWithEngine, warnings}]
await store.load(id);        // WeightRun (snapshot letti dal browser quando servono) o null (store.lastError dice perché)
await store.remove(id);      // "Dimentica"
await store.estimate();      // {usage, quota} o null;   store.available, store.lastError
```
Due archivi IndexedDB (`gpt-didattico-pesi`): `runs` (descrizioni, per l'elenco) e `blobs` (pesi float32 e
snapshot float16 com'erano). Ogni salvataggio è **una sola transazione** (via i blob vecchi, poi descrizione
e blob nuovi): se il browser la annulla, per esempio per la quota esaurita a metà, resta intera la versione
di prima. Una connessione chiusa dal browser o da un'altra scheda si riapre alla prima operazione; un blob
che non torna con la descrizione è "danneggiato" (`load` → null) e non diventa mai un run con i numeri
sbagliati. Tutto in try/catch: senza IndexedDB (navigazione privata, `file://`) l'elenco è vuoto e `save`
risponde `{ok: false, reason}`; la pagina funziona con i run in memoria.

#### Matrici con i numeri (`core/matrixNumbers.js`)

```js
import { NumericMatrix, NumbersGroup, attachNumbers } from "../../core/matrixNumbers.js";

const numbers = new NumbersGroup({ maxOverlays: 4, maxCellsPerOverlay: 2500 });   // il budget della scena
const m = new NumericMatrix({ rows: 256, cols: 64, data: t.data, group: numbers });   // o NumericMatrix.fromTensor(t, {group})
//   opzioni: cell (lato nel mondo, predefinito CELL = 0,04), range (|valore| del colore pieno; predefinito il massimo),
//   colormap "diverging" | "attention", masked(r, c) + maskedText (es. "−∞"), rowLabel(r), colLabel(c)
scene.add(m.object3d);                                   // m.mesh è il piano da colpire con il raycaster
ui.host.onFrame(() => numbers.update(ui.host.camera, ui.host.container.clientHeight));   // una volta per frame
m.setData(snapshot.tensors.get(name).data);              // un altro step: la scala resta ({ range: "auto" } la ricalcola)
const cell = m.cellAt(ui.host.pick(event, [m.mesh])[0]); // {row, col, index, value, rowLabel?} o null
m.highlight(cell);                                       // riquadro sulla cella; null lo toglie
m.showingNumbers;  m.dispose();

// numeri sulle heatmap già esistenti (matrixPlane di core/visuals.js, riga 0 in alto):
const h = attachNumbers(mesh, { getValue: (r, c) => values[r * cols + c], group: numbers });   // rows/cols dalla texture
h.refresh();  h.cellAt(hit);  h.highlight(cell);  h.showing();  h.dispose();
```

- **Convenzioni di `core/visuals.js`**: 1 cella = 1 peso, un `nn.Linear` (out, in) è alto `out`, riga 0 in
  alto; i tensori 1D (bias, γ, β) sono una riga (`fromTensor`); oltre 2D `reshape(-1, ultima)`. Colori dalle
  stesse colormap (`colormapLUT` le ricava da `weightTexture` e `attentionTexture`).
- **Quando compaiono**: solo se la cella è abbastanza grande sullo schermo (≈ 36 px); i decimali crescono con
  la cella (2 a ~40 px, 3 a ~60, 4 a ~80, 6 oltre ~100, e da 110 px anche l'indice "riga,colonna"). Con
  `CELL` = 0,04 e fov 45° su uno stage alto 900 px significa camera a ≲ 1,2 unità: **abbassate
  `ui.host.controls.minDistance`** (es. 0,15-0,25) nella vista che mostra i numeri (il predefinito è 1).
- **Leggibilità**: virgola decimale e segno "−"; testo nero o bianco scelto per contrasto sul colore vero
  della cella (≥ 4,5:1 su ogni colore: verificato nei test); righe sottili tra le celle.
- **Solo la finestra visibile**: l'overlay (un piano figlio della matrice con una CanvasTexture) copre le
  celle visibili più un margine del 25% per lato, al più `maxCellsPerOverlay` celle; al più `maxOverlays`
  matrici con i numeri (le più grandi sullo schermo); `maxCanvasPixels` (4 M) per overlay e
  `maxTotalPixels` (8 M) per scena. Di spalle, nascosta (anche un genitore) o fuori dalla scena: niente numeri.
- **Nessun lavoro a riposo**: `update` confronta camera, viewport e versioni dei dati con l'ultima volta ed
  esce subito (0,2 µs per frame). `setData` / `setRange` / `refresh()` e, per `attachNumbers`,
  `texture.needsUpdate = true` sulla heatmap (come fa `SceneTextures.writeWeights` del Transformer)
  ridisegnano anche a camera ferma; matrici spostate da codice senza muovere la camera: `numbers.invalidate()`.
- L'overlay non si colpisce con il raycaster e ha `userData.noBounds`; segue opacità e visibilità della
  matrice (è suo figlio). Solo camere prospettiche.

Esempio completo (vista "Pesi numerici" del Transformer, con lo snapshot scelto):
```js
const numbers = new NumbersGroup();
let source = engine.paramViews;                      // Map nome → {data}: pesi finali (o le viste di uno snapshot)
const handles = [];
for (const [name, meshes] of tab.arch.tensorMeshes) {
  for (const mesh of meshes) {
    const { width: cols, height: rows } = mesh.material.map.image;     // (n, 1) per le barre dei vettori
    handles.push(attachNumbers(mesh, { rows, cols, group: numbers, getValue: (r, c) => source.get(name).data[r * cols + c] }));
  }
}
ui.host.controls.minDistance = 0.2;
const stop = ui.host.onFrame(() => numbers.update(ui.host.camera, ui.host.container.clientHeight));
async function showStep(step) {                      // cursore sugli step del training
  const snap = await run.loadSnapshot(step);         // run = await loadBuiltinRun()
  source = snap.tensors;
  sceneTextures.writeWeights(new Map([...snap.tensors].map(([n, t]) => [n, t.data])));   // heatmap: needsUpdate → numeri aggiornati
}
// uscendo dalla vista:
stop(); handles.forEach((h) => h.dispose()); numbers.dispose(); ui.host.controls.minDistance = 1;
```

Costi misurati (rendering software, 1600×900, solo JavaScript): `update` a riposo 0,2 µs/frame (0,9 µs con
94 matrici), in movimento 0,02-0,05 ms; ridisegno di un overlay 0,6-1,3 ms; 94 `NumericMatrix` create in
21 ms; `loadBuiltinRun` da disco 7 ms; `buildModuleTree` 5 ms.

#### Run esportabili e benchmark (Python: `gpt/runs.py`, `gpt/benchmark.py`)

Un *run* è una cartella con i pesi nel formato della web app (`weights.json` + `weights.f32.bin`, scritti
dalla stessa `gpt.export.export_weights` dei dati committati), la sua descrizione (`run.json`), a richiesta
il benchmark (`benchmark.json`) e la storia del training (`training/`). Per **esportare** non serve la
classe del modello (bastano i tensori dello state_dict: va bene anche un'architettura modificata); per
**misurare** sì, perché i pesi vanno eseguiti con il codice attuale. Gli schemi completi di `run.json` e
`benchmark.json` sono nelle docstring dei due moduli.

| Modulo | Esporta | Uso |
|---|---|---|
| `gpt/runs.py` | `read_checkpoint(path)`, `read_run_folder(path)`, `read_run_info(path)`, `weights_from_state_dict(sd, *, config, vocab, source, trained, notes)`, `weights_from_model(model, *, vocab, source)` | pesi di un run come `RunWeights` (`tensors` float32 nell'ordine dello state_dict, `config`, `vocab`, `source`, `source_dtypes`, `classes`, `buffers`, `skipped`, `notes`; `.sha256()`, `.with_module_info(model)`) |
| | `read_training_history(dir)` → `TrainingHistory` (`trace`, `layout`, `snapshots`, `missing`, `summary()`), `snapshot_tensors(history, entry)` | `trace.json` + snapshot float16 controllati |
| | `export_run(weights, out_dir, *, name, description, notes, training_dir, benchmark, engine_compatible, overwrite, created)` → `run.json` | scrive la cartella del run |
| | `module_tree(weights)`, `readable_json(value)`, `display_path(path)`, `slugify(name)`, `RUNS_DIR`, `RUN_FILE`, `BENCHMARK_FILE` | albero dei moduli, JSON leggibile, percorsi senza cartelle personali |
| `gpt/benchmark.py` | `benchmark_path(path, *, name, settings, text, excerpt_path, snapshots)`, `benchmark_model(model, tokenizer, *, name, source, validation, excerpt, history, weights, settings)` | il benchmark di un `.pt`, di una cartella di run o di un modello in memoria → contenuto di `benchmark.json` |
| | `BenchmarkSettings(prompt="ROMEO:\n", seed=1337, max_new_tokens=200, temperature=1.0, batch_windows=512, device="cpu", threads=None)` | le scelte fisse (cambiarle rende i numeri non confrontabili) |
| | `contiguous_windows(n, block_size)`, `score_tokens(model, ids, *, block_size, token_chars, batch_windows, surprisal)` → `Scores`, `generation_sample(...)`, `greedy_continuation(...)` | il protocollo, pezzo per pezzo |
| | `validation_text(text)`, `excerpt_bounds(validation)`, `read_excerpt(path, validation)`, `EvalText` | i testi valutati, con sha256 |
| | `open_source(path)`, `build_model(weights)`, `run_tokenizer(weights, default_vocab)`, `ArchitectureMismatch` | i pesi eseguiti dalla classe ATTUALE (nomi e forme identici, altrimenti errore che elenca le differenze) |
| | `comparison_rows(a, b)`, `comparison_warnings(a, b)`, `format_comparison(a, b)`, `format_summary(bench)`, `format_int(n)` | confronto A/B e testo per il terminale, in italiano |
| | `compute_benchmark_data()`, `write_benchmark_data()`, `benchmark_differences(committed, fresh)`, `load_benchmark(path)`, `write_json(path, data)` | i dati del benchmark per il browser e il controllo che siano aggiornati |

| Script | Che cosa fa |
|---|---|
| `scripts/export_run.py CKPT --name NOME [--description …] [--note …] [--trace DIR] [--benchmark] [--threads N] [--not-engine-compatible] [--out DIR] [--force]` | checkpoint → `runs/<slug del nome>/`. `--trace` copia la storia registrata da `scripts/record_training.py --out DIR`; `--benchmark` misura il run (fallisce prima di scrivere se il codice attuale non ricostruisce i pesi); `--force` sostituisce un run cancellando solo i file del run |
| `scripts/benchmark.py SORGENTE [--out FILE] [--name …] [--prompt …] [--seed …] [--tokens …] [--threads N] [--no-snapshots]` | misura un `.pt` o una cartella di run (anche `webapp/data`) e stampa il riassunto |
| `scripts/benchmark.py A B` | tabella di confronto; `A` e `B` possono essere anche `benchmark.json` già calcolati (si confronta così un run che il codice attuale non sa più eseguire) |
| `scripts/export_benchmark_data.py [--check] [--threads N]` | rigenera (o controlla, codice di uscita 1 se non aggiornati) `webapp/data/benchmark/` |

Cartella di un run (`runs/` è in `.gitignore`):
```
runs/<nome>/
  run.json                      nome, descrizione, data (ISO 8601 UTC), sorgente, config (anche chiavi nuove),
                                total_params, tensor_count, modules (albero), buffers, skipped, files,
                                training (riassunto di trace.json), engine_compatible (solo se dichiarato), notes
  weights.json                  manifest di gpt/export.py: per il modello del corso identico byte per byte
  weights.f32.bin               float32 little-endian, nell'ordine dello state_dict
  benchmark.json                (facoltativo) gpt/benchmark.py
  training/trace.json           (facoltativo) copia di scripts/record_training.py --out
  training/snap_XXXXX.f16.bin   (facoltativo) gli snapshot citati da trace.json
```
Albero `modules`: nodi `{path, key, kind: "module", class, params, tensors, children}` e foglie `{path, key,
kind: "param" | "buffer", shape, numel, dtype: "float32", source_dtype}`; `class` e i buffer vengono dal
codice attuale solo se questo ricostruisce i pesi (altrimenti `null`). Note automatiche in `notes`:
conversioni da float16/bfloat16, voci saltate (interi come `num_batches_tracked`), valori non finiti,
prefissi tolti, storia di un altro training, `benchmark.json` misurato su altri pesi, file estranei.

**Benchmark: protocollo `finestre-contigue-v1`.**
1. **Testo**: la validation di Tiny Shakespeare = ultimo 10% di `data/input.txt` (la regola di
   `train_val_split`), 111.540 caratteri; un carattere = un token.
2. **Finestre contigue** di `T = block_size` token che non si sovrappongono: la finestra che parte da `s`
   legge `ids[s … s+T)` e prevede `ids[s+1 … s+T+1)`; l'ultima può essere più corta. Ogni token tranne il
   primo è previsto **una volta**, dal contesto dei token precedenti nella sua finestra.
3. **Metriche**: `loss` = media di −ln p(giusto) (nat); `perplexity` = exp(loss); `bpc` = Σ nll / (ln 2 ·
   caratteri previsti); `top1`/`top5`: giusto se **meno di k logits sono strettamente maggiori** di quello
   del token giusto (a pari merito vince il giusto, uguale in ogni linguaggio); `entropy` = media di
   −Σ p ln p. Conteggi esatti in `counts`.
4. **Testo generato** con prompt `"ROMEO:\n"`, seme 1337, temperatura 1, 200 token: `sampled`
   (`torch.multinomial` con un `torch.Generator` suo) e `greedy` (argmax, a pari merito il primo indice).
   Il motore JS ha un altro generatore casuale: riproduce `greedy`, **non** `sampled`.
5. **Prestazioni** (dipendono dalla macchina): `tokens_per_s`, `eval_tokens_per_s`, `duration_s`. Su una
   macchina carica `threads=1` (più thread che core liberi rallentano un forward di centinaia di volte; i
   numeri restano identici).

| `benchmark.json` | Contenuto |
|---|---|
| `format_version`, `name`, `created`, `source` | 1; nome del run; data ISO 8601 UTC; percorso relativo al progetto |
| `protocol` | `{id: "finestre-contigue-v1", description, block_size, top_k: [1, 5], log_base: "e"}` |
| `data` | il testo valutato: `{name, file: "data/input.txt", offset: 1003854, chars: 111540, sha256}` |
| `model` | `{class, config, params, vocab_size, weights_sha256 (= sha256 di weights.f32.bin), code_sha256, code_files}` |
| `metrics` | `loss, perplexity, bpc, top1, top5, entropy, predictions, params, tokens_per_s, eval_tokens_per_s, duration_s` (validation completa) |
| `counts` | `{predictions, windows, chars, top1, top5}` |
| `excerpt` | lo stesso sull'estratto: `{file: "val_excerpt.txt", chars, sha256, offset, loss, perplexity, bpc, top1, top5, entropy, counts, surprisal_bits: [−log₂ p di ogni carattere previsto, 3 decimali]}` o null |
| `generation` | `{prompt, seed, temperature, max_new_tokens, sampled, greedy, tokens_per_s}` |
| `snapshots` | per una cartella con `training/`: `[{step, file, dtype: "float16", metrics: {…}, excerpt: {…, counts}}]` |
| `reference` | `{uniform_loss: ln V, uniform_bpc: log₂ V}`: il modello che tira a caso |
| `environment`, `notes` | `{python, torch, numpy, device, threads, system, machine}` (nessun nome di macchina o di utente); avvisi |

Numeri del GPT del corso (validation completa / estratto): loss 1.8145 / 1.7337, perplexity 6.138 / 5.662,
bpc 2.618 / 2.501, top-1 45.62% / 47.51%, top-5 79.70% / 80.79%; snapshot dello step 0: loss 4.4013 (il
caso: ln 65 = 4.1744).

**Controllo di coerenza nel browser** (Laboratorio pesi): per un run compatibile con il motore JS si
ricalcola l'estratto con lo stesso protocollo e si confronta con `excerpt` di `benchmark.json` (o di
`builtin.json` per il GPT del corso, anche per ogni snapshot). Tolleranze (le stesse di
`benchmark_differences`): `|Δ loss|`, `|Δ entropy|`, `|Δ bpc|` ≤ 1e-4; previsioni giuste top-1/top-5 entro
±2; `predictions` e `windows` identici; bit di sorpresa entro 2e-3; `greedy` identico. Misurato con il
motore vero: `|Δ loss|` ≤ 5e-7 sui pesi finali e su tutti gli 11 snapshot. Implementazione di riferimento
(quella che usa `tests/test_benchmark.py`):
```js
// Il protocollo "finestre-contigue-v1" con il motore del corso (js/engine/gpt.js).
export function scoreText(engine, ids) {
  const T = engine.config.block_size, V = engine.config.vocab_size;
  let nll = 0, entropy = 0, top1 = 0, top5 = 0, predictions = 0, windows = 0;
  const bits = [];                                       // sorpresa di ogni carattere, in bit
  for (let start = 0; start + 1 < ids.length; start += T) {
    const len = Math.min(T, ids.length - 1 - start);      // l'ultima finestra può essere più corta
    const { logits } = engine.forward(ids.slice(start, start + len));
    windows++;
    for (let t = 0; t < len; t++) {
      const off = t * V, target = ids[start + t + 1];
      let max = -Infinity, sum = 0, h = 0, greater = 0;
      for (let i = 0; i < V; i++) max = Math.max(max, logits[off + i]);
      for (let i = 0; i < V; i++) sum += Math.exp(logits[off + i] - max);
      const logZ = max + Math.log(sum);                  // log-softmax stabile
      for (let i = 0; i < V; i++) {
        const lp = logits[off + i] - logZ;
        h -= Math.exp(lp) * lp;
        if (logits[off + i] > logits[off + target]) greater++;   // strettamente maggiori
      }
      const lp = logits[off + target] - logZ;
      nll -= lp; entropy += h; predictions++; bits.push(-lp / Math.LN2);
      if (greater < 1) top1++;
      if (greater < 5) top5++;
    }
  }
  return { predictions, windows, loss: nll / predictions, perplexity: Math.exp(nll / predictions),
           bpc: nll / (Math.LN2 * predictions), entropy: entropy / predictions, top1, top5, bits };
}
// greedy: per 200 volte nextTokenDistribution(context).logits → argmax (il primo a pari merito).
```

Regole:
- **Deterministico**: niente casualità nella valutazione; due esecuzioni danno gli stessi numeri (anche
  cambiando `batch_windows` o i thread). Tra macchine diverse le differenze sono ~1e-6.
- **Stesso metro**: i pesi si eseguono con la classe ATTUALE (`build_model`: `GPTConfig(**config)` e
  nomi/forme identici allo `state_dict()`). Un'architettura modificata si misura con il codice modificato
  *prima* di tornare indietro; per confrontarla col GPT del corso si usa `builtin.json`. `code_sha256` dice
  se il codice è cambiato. Confrontabile solo con lo stesso `protocol.id` e lo stesso `data.sha256`
  (`comparison_warnings` lo segnala); le velocità non hanno mai un verdetto "meglio".
- **Sicurezza**: checkpoint con `torch.load(weights_only=True)` (un pickle malevolo non viene eseguito: c'è
  un test), nomi di file presi dai JSON solo se semplici e accanto al JSON, layout controllati, JSON sempre
  validi per il browser (`allow_nan=False`), nei JSON nessun percorso fuori dal progetto. Prima si controlla
  tutto, poi si scrive: un errore non lascia cartelle a metà; `--force` cancella solo i file del run.
- `builtin.json` va rigenerato (`scripts/export_benchmark_data.py`) se cambiano i pesi committati o il
  protocollo; `tests/test_benchmark.py` lo controlla.

```bash
python scripts/record_training.py --checkpoint checkpoints/gelu/gpt.pt --out checkpoints/gelu/training --no-export
python scripts/export_run.py checkpoints/gelu/gpt.pt --name "GPT con GELU" --trace checkpoints/gelu/training \
    --benchmark --not-engine-compatible
python scripts/benchmark.py webapp/data/benchmark/builtin.json runs/gpt-con-gelu/benchmark.json
```

**Prove.** `tests/weights.test.mjs` (formati, safetensors scritti byte per byte, riconoscimento dei file,
run, statistiche, confronto, albero dei moduli con le righe controllate sui file `gpt/*.py`, salvataggio,
formato dei numeri, `NumericMatrix`/`attachNumbers` con three.js da `vendor/` e un canvas finto);
`tests/weights-robustness.test.mjs` (file rotti o malevoli: intestazioni enormi, JSON rotti, offset fuori
dal file o sovrapposti, dtype del prototipo, file troncati, nomi strani; IndexedDB che fallisce, quota
esaurita a metà, connessione chiusa, blob danneggiato); `tests/test_runs.py`, `tests/test_benchmark.py`
(Python, anche il motore JS che riproduce `builtin.json`). Pagina di prova: `tests/matrix-demo.html`
(`window.__matrixDemo`).

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
| `benchmark/val_excerpt.txt` | estratto fisso della validation (4.015 caratteri = 4.014 previsioni in 126 finestre, ~0,8 s con il motore JS); byte esatti: il `.gitattributes` accanto impone `eol=lf` | `scripts/export_benchmark_data.py` |
| `benchmark/builtin.json` | `benchmark.json` del GPT del corso dai pesi committati: validation completa, estratto con i bit di sorpresa, testi, gli 11 snapshot (vedi "Pesi e run") | `scripts/export_benchmark_data.py` |

In JavaScript: `loadEngine("data/") → GPTEngine` (`js/engine/gpt.js`) con `encode/decode`,
`forward(ids, {trace})`, `nextTokenDistribution`, `generate(ids, n, {temperature, topK, seed})`,
`logitLens`, `loadTrace()`, `loadSnapshot(step)` / `setWeights` / `restoreWeights` (pesi del training).
Nel tab Transformer si usa `tab.getEngine()`, che lo carica una volta sola. La libreria RAG
(`js/shared/rag/`: corpus, pulizia, chunker, embedder, HNSW, vector store, prompt) è pura come il
motore e ha i suoi test in `tests/rag.test.mjs` e `tests/hnsw.test.mjs`.

Formati dettagliati: vedi le docstring di `gpt/export.py` e `gpt/recording.py` (run e benchmark: `gpt/runs.py`
e `gpt/benchmark.py`). Pesi, snapshot, run caricati, safetensors e benchmark nel browser: `js/shared/weights/`
(vedi "Pesi e run").

## Il tab Transformer: oggetto `tab`

`js/tabs/transformer/index.js` crea l'oggetto `tab` e chiama `install(tab)` dei moduli:

```
tab.app, tab.ui (output di createTabLayout), tab.spec (model.json), tab.vars
tab.arch = { root, rail, registry: Map(id → Object3D), stations, path }   // vedi architecture.js
tab.flow (DataFlow: pacchetto animato sul binario; flow.toggle(), flow.packet)
tab.events (EventTarget: evento "view" con detail = id vista)
tab.panes = { path, actions, views }   // elementi dei pannelli della sidebar
tab.menu (Menu del percorso)
tab.views.register({ id, label, description, order, enter(tab), exit(tab), home?(tab) }); tab.views.activate(id)
tab.views.home()   // "Centra" e "Vista d'insieme": l'inquadratura iniziale della vista attiva (home della vista)
tab.views.has(id); tab.contextView.reveal(stepId)   // apre un passo ctx.* del percorso della vista Contesto
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

Link diretti e ricerca (`search.js`, `goto`): `step:<id di content.js>` (torna all'Architettura, chiude
l'azione, zoom su Auto), `action:prompt|training-step|training-run|lod-zoom|weights-tour|neurons-forward|context-pipeline`,
`view:architettura|pesi|neuroni|contesto`, `view:zoom.detail|zoom.blocks|zoom.macro` (Architettura, livello
bloccato, pannello Viste aperto) e `view:ctx.<passo>` (un passo del percorso della vista Contesto, con lo
zoom su Auto; `tests/search.test.mjs` controlla che esista in `context/texts.js`).

## Gli altri tab

| Tab | Cartella | Contenuto |
|---|---|---|
| Database vettoriale (RAG) | `tabs/rag/` | pipeline L17-L18 a stazioni: upload, estrazione, pulizia, chunking, embedding, `vector_store`, indice HNSW; poi ricerca HNSW, prompt, LLM, streaming. Azioni "Indicizzazione" e "Ricerca". Link: `action:indicizzazione\|ricerca`, `view:detail\|blocks\|macro`, oltre a `#rag/q=<domanda>` |
| Immagini e video | `tabs/diffusion/` | DDPM/DDIM, Stable Diffusion (text encoder, VAE, U-Net, scheduler), DiT, video (VAE 3D, patch spazio-temporali). Azioni di training, generazione, VAE, denoiser, video. `sceneNav.js`: "Centra", confini per livello e didascalie delle due parti (immagini sopra, video sotto). Link: `action:forward\|generate\|vae\|inside\|video`, `view:unet\|dit\|pipeline\|video\|macro\|detail` |
| Audio e musica | `tabs/audio/` | forma d'onda, STFT, Mel, codec neurale con RVQ, generazione di token musicali (delay pattern), Whisper. Laboratorio con suono vero. `catalog.js` (solo dati): nomi e descrizioni di azioni e viste. Limiti di rotazione (`setOrbitLimits`). Link: `action:audio-sound\|audio-rvq\|audio-generate\|audio-whisper`, `view:overview\|delayCore\|rvq\|whisperAll\|detail\|blocks\|macro` |
| LLM moderni | `tabs/modern-llm/` | 16 meccanismi (RMSNorm, RoPE, GQA, MoE, KV cache, FlashAttention, BPE, decodifica speculativa, scaling, post-training, ragionamento, agenti, coding, multimodale, mesh 3D, interpretabilità) con azioni animate. Etichetta "Dove sei" (`regionCompass.js`) mentre si passeggia tra le regioni; moduli di soli dati `regionData.js`, `actionCatalog.js`, `palette.js` (usati anche da `search.js`). Limiti di rotazione (`setOrbitLimits`). Link: `action:<id>` (18), `view:macro\|blocks\|detail\|<regione>` |

Ogni tab ha `window.__<nome>Tab` (es. `__ragTab`, `__diffusionTab`, `__audioTab`, `__modernLlmTab`;
per il Transformer `__gptApp`) per la console e per gli script di verifica.

## Test

- Python: `pytest` (dalla radice).
- JavaScript: `node --test webapp/tests/*.test.mjs` (moduli puri senza DOM: motore GPT, libreria RAG, …;
  `brain-pane.test.mjs` con un DOM finto minimo, `weights.test.mjs` con three.js da `vendor/` e un canvas finto).
- Pagine di prova dei moduli del nucleo (aperte con lo stesso server): `tests/charts-demo.html`,
  `tests/reps-demo.html` (`?viste=vere` = integrazione con rete, cervello e pannello veri, sulle specifiche di
  riferimento del nucleo),
  `tests/neuralnet-demo.html`, `tests/brain-demo.html`, `tests/matrix-demo.html`.
- Visivo: `python scripts/serve_webapp.py --no-export --port 8765` poi
  `NODE_PATH=$(npm root -g) node webapp/tests/screenshot.mjs http://127.0.0.1:8765/ <cartella> [tab1,tab2,…] [--lod] [--nav] [--mobile]`
  (senza elenco: tutti i tab; `--lod` aggiunge uno screenshot per livello dello zoom, `--nav` uno dopo
  uno spostamento con D e W tenuti premuti e uno dopo C (deve tornare alla vista iniziale), `--mobile`
  usa 390×844; stampa gli errori della console e il codice di uscita è 1 se ce ne sono).
