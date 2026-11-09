# Architettura della web app (contratto per chi la estende)

Questo documento è il riferimento comune per tutti i moduli. Se un'API qui descritta non basta,
**non modificare file di altri**: segnala cosa serve nel tuo report finale.

## Indice

- [Principi](#principi) · [Struttura](#struttura) · [Contratto di un tab](#contratto-di-un-tab)
  ([eventi tra tab](#eventi-tra-tab-appbus), [ricerca, link diretti e glossario](#ricerca-globale-link-diretti-e-glossario))
- [Nucleo: js/core/](#nucleo-jscore): [LOD](#lod-zoom-semantico) · [navigazione libera](#navigazione-libera) ·
  [azioni animate](#azioni-animate-actionplayer) · [layout condiviso](#layout-condiviso-csscorecss) ·
  [grafici](#grafici-chartspanel) · [rappresentazioni](#rappresentazioni-corerepresentationsjs-coreuirepswitchjs-cssrepscss) ·
  [rete neurale orizzontale](#rete-neurale-orizzontale-coreneuralnet-cssneuralnetcss) ·
  [cervello](#cervello-corebrain-cssbraincss) ·
  [pesi e run](#pesi-e-run-jssharedweights-corematrixnumbersjs-gptrunspy-gptbenchmarkpy) (con i run esportabili, il
  benchmark e la storia del training in Python) ·
  [numeri dei tensori](#numeri-dei-tensori-tabella-statistiche-istogramma-coreuitensortablejs-coreuihistogramjs-csstensor-uicss)
- [Pannello codice](#pannello-codice) · [Dati esportati dal Python](#dati-esportati-dal-python-webappdata)
- [Il tab Transformer](#il-tab-transformer-oggetto-tab): [Pesi numerici](#vista-pesi-numerici-viewsnumbersjs) ·
  [Rete neurale a righe / orizzontale con profondità](#rete-neurale-a-righe--orizzontale-con-profondità-viewsneuronsjs-viewsneurondepthjs) ·
  [rappresentazioni e cervello](#rappresentazioni-e-cervello-del-transformer)
- [Il tab Laboratorio pesi](#il-tab-laboratorio-pesi-tabslab-csstabslabcss): [oggetto tab e stato](#loggetto-tab-e-lo-stato) ·
  [scena](#la-scena-tabscenescene-labscene) · [run caricati, note e limiti](#run-caricati-note-della-registrazione-e-limiti) ·
  [azioni animate](#azioni-animate-tabslabactions) · [confronto, grafici, benchmark](#confronto-grafici-benchmark-tabslabanalysis) ·
  [link](#link-diretti-e-ricerca-del-laboratorio) · [rappresentazioni](#rappresentazioni-del-laboratorio)
- [Il tab LLM moderni](#il-tab-llm-moderni-vista-dinsieme-e-architettura-completa-tabsmodern-llm) (vista d'insieme e
  architettura completa)
- [Gli altri tab](#gli-altri-tab) (la tabella dei sei tab, con moduli e link;
  [rappresentazioni di RAG, Immagini e video, Audio](#rappresentazioni-di-rag-immagini-e-video-audio)) · [Test](#test)

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
  css/tabs/<id>.css     stili di un tab: classi con prefisso del tab (es. .rag-…; il Laboratorio ha anche
                        lab-actions.css e lab-analysis.css, il Transformer transformer-actions/-context.css)
  css/<modulo>.css      stili dei moduli del nucleo: nav, palette, reps (rappresentazioni), neuralnet, brain,
                        tensor-ui (tabella e istogramma dei tensori), charts
  data/                 dati esportati dal Python (non modificarli a mano)
  vendor/               librerie incluse (three, katex, highlight.js)
  tests/                test JS (node --test), pagine di prova dei moduli (*-demo.html) e script di screenshot
  js/app.js             registro dei tab e routing (#transformer, #lab, #rag, …)
  js/core/              nucleo condiviso (vedi sotto): scena, navigazione, rappresentazioni, rete neurale
                        (core/neuralNet/), cervello 3D (core/brain/), matrici con i numeri, componenti DOM (core/ui/)
  js/engine/            motore di inferenza GPT in JavaScript
  js/shared/rag/        libreria RAG didattica (corpus, embedding, chunking, HNSW, vector store)
  js/shared/weights/    libreria dei pesi e dei run (formati, safetensors, run, statistiche, confronto, salvataggio)
  js/shared/gpt/        la rete "Orizzontale con profondità" del GPT del corso (depthSpec.js), di Transformer e
                        Laboratorio: un tab non importa i moduli di un altro (tests/structure.test.mjs)
  js/tabs/<id>/         un tab per cartella; entry point index.js, indice della ricerca search.js e, dove
                        ci sono, network.js (Rete neurale) e brain.js (Cervello), moduli di soli dati
```

I sei tab, nell'ordine della barra: `transformer`, `lab` (Laboratorio pesi), `rag` (Database vettoriale),
`diffusion` (Immagini e video), `audio` (Audio e musica), `modern-llm` (LLM moderni); `TABS` in `js/app.js`.

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
Oltre ai tipi della ricerca (`KINDS`: step, action, view, term, tab) un tab può riconoscere nel suo `goto` tipi
suoi, che non vanno nella ricerca: `tensor:<nome>[@<step>]` (Transformer e Laboratorio pesi, stesso formato e stesso
significato nei due tab) e `run:<id>` (Laboratorio). Le rappresentazioni del nucleo usano `view:rete` e
`view:cervello` in ogni tab (vedi "Rappresentazioni"); l'elenco dei link di ogni tab è in "Gli altri tab".

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
| `sceneHost.js` | `SceneHost` | `host.scene`, `host.camera`, `host.controls` (`minDistance` predefinito 1), `host.onFrame((dt, elapsed) => …)` (restituisce la funzione di rimozione; `dt` è limitato a 0,05 s), `host.start/stop`, `host.running`, `host.onResize(fn)`, `host.onStop(fn)`, `host.snapshot()`, `host.zoomDistance()`, `host.pick(event, objects, recursive = true)` (gli oggetti sotto il puntatore di un `MouseEvent`, o di un oggetto `{clientX, clientY}`, dal più vicino: è `raycaster.intersectObjects(objects, recursive)`; con `recursive = false` contano solo gli oggetti dell'elenco e non i loro figli, come serve quando si colpiscono i piani delle matrici senza gli overlay dei numeri che vi sono appesi: Pesi e Pesi numerici del Transformer, Laboratorio, segnaposto di LLM moderni), `host.visibleAspect()` (larghezza/altezza della parte non coperta dai pannelli) |
| `director.js` | `CameraDirector`, `meshBounds(object, {visibleOnly})`, `instancedBounds(mesh)` | `await director.focus(objectOrBox3, {showFrame, margin, direction, minDistance, duration})`, `director.flyTo(pos, target, duration)`, `director.hideFrame()`, `director.cancel()`, `director.flying` (un volo è in corso), `director.finish()` (porta subito alla meta il volo in corso → `true`, `false` se non volava), `director.actionActive` (un'azione è in corso e non è finita), `director.intentActive` e `director.lastIntent` (l'ultima intenzione dello studente vale ancora; contatore delle intenzioni: vedi "Navigazione libera"). I voli avanzano con il tempo reale (anche con frame lenti durano quanto chiesto); trascinare la scena li interrompe. Inseguimento e camera libera durante le azioni: vedi "Navigazione libera". Un oggetto con `userData.members = [...]` viene inquadrato come unione dei membri. `userData.noBounds`/`isFrame` esclude un mesh dal calcolo; `visibleOnly` salta i sotto-alberi nascosti. Un `InstancedMesh` conta **tutte** le sue istanze visibili (`instancedBounds`: box locale delle prime `count` istanze, quelle a scala zero saltate, in cache finché non cambiano `instanceMatrix`, `count` o la geometria) |
| `lod.js` | `LODController`, `setGroupOpacity(group, alpha)`, `setBaseOpacity(material, opacity)` | vedi "LOD" |
| `diagram.js` | `makeBlock({title, subtitle, text, width, height, depth, color, fill})`, `makeArrow(from, to, {color, radius, headLength})`, `makeFlow(points, {color, radius, speed})` | blocchi 3D con testo (la descrizione si rimpicciolisce fino al 60% per entrare tutta; la faccia col testo ha il polygon offset contro lo z-fighting); `flow.userData.tick(dt)` va chiamato a ogni frame |
| `visuals.js` | `CELL`, `COLORS`, `DIVERGING_LUT`, `ATTENTION_LUT`, `MASKED_LUT`, `weightTexture(tensor)`, `attentionTexture(b64, T)`, `matrixPlane(texture, rows, cols)`, `label(text, {size, color, sub})`, `addNode()`, `tokenCube(char, size)` | matrici di pesi come heatmap, etichette sprite. Le tabelle dei colori sono `Uint8Array` RGBA di sola lettura (per convenzione: non si possono congelare), le stesse di `weightTexture` e `attentionTexture`: `DIVERGING_LUT` 256 colori (indice = peso quantizzato come `gpt/export.py`, 128 = zero), `ATTENTION_LUT` 256 (0…255 = 0…1), `MASKED_LUT` i 4 colori del tratteggio del futuro mascherato (indice (riga + colonna) % 4). Chi colora celle o barre come le heatmap le usa invece di ricavarle disegnando una rampa (`matrixNumbers.js`, `tabs/transformer/actions/colormaps.js`) |
| `layout.js` | `createTabLayout(root, {code, charts, player, hint, background})`, `createSceneStack(stage, {root, player, codePanel, charts, coveredRight, background})` | `createTabLayout` restituisce `{root, sidebar, stage, caption, host, director, nav, codePanel, charts, player}` (`nav`: vedi "Navigazione libera"); `createSceneStack` monta solo `SceneHost` + `CameraDirector` (+ `ActionPlayer`) + `Navigation` + `NavPad` → `{host, director, nav, player}` (una seconda scena con la stessa navigazione: il livello delle rappresentazioni) |
| `ui/sidebar.js` | `Sidebar` | `sidebar.addPane({id, title, hint}) → HTMLElement`, `sidebar.select(id)` (ogni pannello ricorda il suo scorrimento; uno nuovo si apre dall'alto), `sidebar.onSelect(fn)`, `sidebar.current` (id del pannello aperto o `null`), `sidebar.setPaneHidden(id, hidden)` (nasconde la scheda di un pannello; se era aperto si apre il primo visibile), `sidebar.movePane(id, beforeId)` (sposta scheda e pannello prima di `beforeId`, `null` = in fondo; `false` se uno dei due manca: così "Cervello" resta prima di "Fonti" anche quando i pannelli nascono dopo). Sotto `.menu-hint` una riga separa il suggerimento dal contenuto che scorre |
| `stepKeys.js` | `installStepKeys({root, ui, menu, pathPane, spaceIgnore})` | ↑ ↓ passo del Percorso (uno per pressione), Spazio pausa/riprendi; ascolto sulla finestra, solo con il tab visibile, mai nei campi, nelle finestre di dialogo, sul pad o con l'aiuto aperto; negli altri pannelli della sidebar le frecce li scorrono |
| `ui/menu.js` | `Menu`, `mathBox(items, vars)` | fisarmonica di passi `{id, title, children:[{id, title, text(HTML), math:[{tex, note}], extra:(item)=>Node}]}`; opzioni `{vars, onSelect(item, title), tabId}` (con `tabId`: progressi e "Copia link a questo passo"); `menu.select(id)`, `menu.reveal(id, {sidebar})`, `menu.move(±1)`, `menu.current`, `menu.items` (vedi "Ricerca globale") |
| `ui/codePanel.js` | `CodePanel` | `codePanel.show({title, file, startLine, code, language: "python"\|"java"\|"javascript"\|"bash"\|"json"\|"xml", highlight:[substring…]})`, `codePanel.hide()`; vedi "Pannello codice" (riduzione, scelte dello studente, aggancio ai grafici) |
| `ui/player.js` | `ActionPlayer` | vedi "Azioni animate" |
| `ui/charts.js` | `ChartsPanel` | vedi "Grafici" |
| `ui/navPad.js`, `navMath.js` | `NavPad`; `padPlacement`, `captionPlacement`, `CAPTION_MIN_WIDTH`, `CAPTION_HIDE_WIDTH` e le regole della navigazione | pad, aiuto, chip; posto del pad e della didascalia (puri, `tests/navigation.test.mjs`); vedi "Navigazione libera" e "Layout condiviso" |
| `representations.js` | `installRepresentations(ui, opts)`; riesporta `REP_IDS`, `REP_LINKS`, `BRAIN_PANE_ID` | selettore "Architettura · Rete neurale · Cervello", livello alternativo con la sua pila di scena, ritorno automatico, pannello Cervello; vedi "Rappresentazioni" |
| `repLinks.js` (soli dati) | `REP_IDS`, `REP_LINKS`, `BRAIN_PANE_ID`, `repOfLink(target)`, `REP_SEARCH_KEYWORDS`, `withRepKeywords(item)` | id delle rappresentazioni, link diretti `view:rete` / `view:cervello`, id del pannello Cervello (`"cervello"`); le parole dei comandi comuni delle due viste ("guarda lo spessore", "anima impulsi"…) che la ricerca globale aggiunge alle voci `view:rete` e `view:cervello` di ogni indice: senza three.js, quindi importabili nei test Node e nei moduli di soli dati |
| `ui/repSwitch.js` | `RepSwitch`, `placeSwitch`, `freeIntervals`, `candidateRows`, `VARIANTS`, `ALT_ACTIVE_CLASS` | il selettore (radiogroup / menu compatto) e la scelta pura del suo posto (test in `tests/reps.test.mjs`) |
| `neuralNet/` | `view.js`: `createNetworkView(ctx)`; `model.js`: `buildNetworkModel(spec)`; `scene.js`: `NetworkScene`; `spec.js`: `normalizeSpec`, `NetworkSpecError`, `LIMITS`, `CARD_MODES`; `layout.js`, `links.js`, `activations.js` (puri); `framing.js` (puro): `readableHeight`, `walkParts`, `avoidRect` (reti lunghe a parti); `honesty.js` (puro): `valueSources`, `badgeOf`, `honestyLines` (onestà dei valori); `occlusion.js` (puro): `occluderOpacity` (lastre fantasma che si attenuano); `roles.js`: `ROLES`, `PATTERNS`, `linkSummary` | rete orizzontale con profondità guidata da `tabs/<id>/network.js` (`networkSpec`); vedi "Rete neurale orizzontale" |
| `brain/` | `view.js`: `createBrainView(ctx)`; `pane.js`: `renderBrainPane(container, spec, {onShow3D, app})` (solo DOM); `spec.js`: `normalizeBrainSpec`, `BrainSpecError`; `atlas.js`: `REGIONS`, `MARKERS`, `PATHWAYS`, `atlasEntry`, `mniToScene`; `morphology.js`: `NEURON_TYPES`, `buildMorphology` | cervello 3D (regioni, percorsi, piani di taglio, microcircuito di neuroni modellati) e pannello delle analogie, guidati da `tabs/<id>/brain.js` (`brainSpec`); vedi "Cervello" |
| `matrixNumbers.js` | `NumericMatrix({rows, cols, data, cell, range, colormap, group})`, `NumbersGroup`, `attachNumbers(mesh, {rows, cols, getValue, group})`, `colormapLUT` | matrici di pesi come celle colorate con i numeri veri quando sono abbastanza grandi sullo schermo; vedi "Pesi e run" |
| `ui/tensorTable.js`, `ui/histogram.js` | `TensorTable`, `renderTensorStats`; `Histogram` (logica pura in `ui/tensorTableMath.js` e `ui/histogramMath.js`) | i numeri veri di un tensore nella sidebar: tabella virtualizzata, statistiche, istogramma di una o più serie (Pesi numerici del Transformer, Laboratorio pesi); vedi "Numeri dei tensori" |

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
lod.setEnabled(false);             // due scene nello stesso SceneHost (LLM moderni): spento, update() esce subito,
                                   // il selettore è nascosto e i gruppi non si toccano più; lod.enabled
setBaseOpacity(material, 0.4);     // animare l'opacità di un materiale senza litigare con le sfumature
```
I gruppi dei livelli devono avere **materiali propri** (non condivisi tra livelli): l'opacità viene
sfumata modificando i materiali. Il livello attivo dipende da `host.zoomDistance()` (distanza
camera → punto guardato): più zoom out = livello più astratto. Livello "Blocchi": un `makeBlock` per
componente con titolo + descrizione di cosa fa. Livello "Panoramica": pochi blocchi grandi + frecce
del flusso (diagramma a blocchi).

**Niente lavoro a camera ferma.** `update()` esce dopo due confronti se la distanza della camera, le soglie
(`levels[i].maxDistance`) e le sfumature non sono cambiate: niente `distanceScale()`, nessun gruppo toccato.
`distanceScale` si rilegge solo quando la distanza cambia, quindi un fattore nuovo (es. il codice ridotto con
la camera ferma) vale dal primo movimento e il livello non cambia sotto gli occhi dello studente; una soglia
cambiata da un tab vale subito. Riacceso con `setEnabled(true)`, il LOD riparte senza dissolvenza dal livello
bloccato o da quello della distanza attuale. Il selettore porta `data-level` (livello attivo) e `data-mode`
(`"auto"` o l'id scelto), così CSS e script li leggono senza conoscere il tab. Test: `tests/lod.test.mjs`.

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
ui.nav.getHomePose(); ui.nav.setHomePose({ position, target } | null)   // la vista iniziale di "Centra" (in copia);
                                              // null = rifotografata alla fine del prossimo volo
const before = ui.nav.setBounds(box3 | () => box3 | null);   // confini (predefinito: mesh visibili, meshBounds,
                                              // allargati del 12%); restituisce la sorgente di prima, per rimetterla
ui.nav.getBounds();                           // la sorgente attuale: Box3, funzione o null (automatici)
ui.nav.setDragMode("rotate" | "pan");  ui.nav.toggleDragMode();  ui.nav.mode
ui.nav.home();  ui.nav.setEnabled(false);      // spegne mouse, tastiera e pad (es. una vista 2D a tutto schermo)
ui.nav.hold(axis, dir) → rilascia(); ui.nav.nudge(axis, dir)   // axis: panX | panY | orbit | tilt | zoom
ui.nav.panBy(dx, dy); ui.nav.stopMotion()     // sposta nel piano dello schermo (con i confini); ferma tutto
ui.nav.setOrbitLimits({ turn: 60, tilt: 45, minDistance: 6 })  // scene "a parete": gradi dalla vista
                                              // frontale (+z); niente testo a specchio né vista dall'alto
ui.nav.on("input", (source) => …)            // "pointer" | "keys" | "pad" | "dblclick" | "home"; anche "mode", "lost"
ui.nav.modalOpen                              // l'aiuto è aperto (i tasti dei tab non devono agire)
const remove = ui.nav.pad.addShortcut({ keys: "V", text: "Vista d'insieme ↔ Architettura completa", section: "Questo tab" });
                                              // una riga nell'aiuto ("?"), in una sezione in fondo alla griglia
                                              // (testo come testo); remove() la toglie, e la sezione se resta vuota
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
  `director.resumeFollow()`, `director.noteUserIntent(ms)`, `director.onFollowChange(fn)`,
  `director.intentActive` (la finestra dell'ultima intenzione è ancora aperta), `director.lastIntent`
  (contatore delle intenzioni: cambia a ogni gesto), `director.flying`, `director.finish()`. Sono le sole API
  che usa `core/representations.js` per il ritorno automatico all'Architettura (prima leggeva e scriveva
  campi interni del director e della navigazione); `director.flight` (il volo in corso o `null`) resta
  leggibile per i tab che lo usavano.
- **Tab nascosto**: quando il suo `SceneHost` si ferma (`host.onStop`), la navigazione azzera tasti tenuti,
  pulsanti e frenata: al ritorno la scena non scivola.
- **Pad**: in basso a destra della scena visibile; accanto al codice se questo arriva fin giù (solo se così
  resta nella metà destra della scena), a sinistra dei grafici aperti, sopra la barra del player se la
  toccherebbe (e sopra il chip "Segui l'azione", che resta sempre cliccabile); sul telefono, aperto, sopra il
  codice ridotto in fondo. Il chip sta sopra la barra del player, o sopra il codice ridotto se questo sta
  subito sopra la barra (telefono con `.code-stays-bottom`). Se l'unico posto copre il codice o il centro
  della scena visibile (es. 1100 px con codice e player, o 260 px di scena accanto ai grafici aperti) si
  riduce da solo al pulsante unico (classe `squeezed`, non ricordata; aperto a mano resta aperto) finché lo
  spazio torna. La regola è pura e pubblica, `padPlacement` di `navMath.js` (sotto): chi deve sapere in
  anticipo dove starà il pad la chiama invece di copiarla. Posizione ricalcolata solo a eventi di layout,
  mai per frame.
  ```js
  import { padPlacement, captionPlacement } from "../core/navMath.js";   // coordinate dello stage in px
  const spot = padPlacement({ sw, sh, size: { w, h }, bar, code, lowest, chartsWidth, edge: 14, base: 14 });
  // → { right, bottom, ok }: ok = non copre il codice né il centro della scena (altrimenti il pad si riduce)
  const cap = captionPlacement({ width, left, sh, height, pad: { right, bottom, w, h }, chartsWidth, bar, code, narrow, fabRoom: 56 });
  // → { right, bottom, hidden, abovePad }: il posto della didascalia (vedi "Layout condiviso")
  ```
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
- **Benvenuto dove lo studente guarda**: `NavPad.maybeShowWelcome()` → `"done" | "later" | "shown"`; "later" se
  i chip di quel pad non si vedono (sotto un livello aperto da un link diretto `view:rete`, dietro il cassetto
  del telefono): il pad riprova ogni mezzo secondo mentre la scena disegna, poi smette. In un livello delle
  rappresentazioni i chip in alto si rimettono a posto (`placeHud()`) a ogni figlio nuovo dello stage e a ogni
  suo cambio di misura (`MutationObserver` + `ResizeObserver`, coalescenti in un frame), perché la scheda della
  vista nasce dopo i chip.
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
ui.player.renderOverview(paneElement, { categories: ["Prompt", "Training"] });   // anche più elenchi: la scheda
                                    // dell'azione in corso si accende in tutti (LLM moderni ne ha due)
ui.player.highlightCard(id | null); // accende a mano una scheda in tutti gli elenchi
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
  "Zoom:" e la barra strumenti si compatta, così non si coprono); codice a destra; player in basso al
  centro, sopra il codice (z-index 16), centrato nella parte libera se i grafici sono aperti; durante
  un'azione il codice si accorcia per non coprire il player.
- **Chi sta dove** (un posto per ogni comando, in tutti i tab; misurato su 6 tab × viste × stati di sidebar e
  grafici a 1600×900, 1280×800, 1100×800 e 390×844: da 196 sovrapposizioni a 1):

  | Comando | Regola | Dove |
  |---|---|---|
  | Pad della navigazione | vedi "Navigazione libera"; nel DOM chip e pad sono gli ultimi figli dello stage di un tab (`NavPad.keepLast`, a ogni figlio nuovo, mai per frame): Tab arriva prima a ☰, zoom, selettore delle rappresentazioni e pulsanti del tab, poi ai 15 pulsanti del pad (nel RAG da 41 a 24 pressioni fino al selettore) | `navMath.padPlacement`, usato da `NavPad.layout` |
  | Didascalia (`.caption`) | sopra la barra del player se la toccherebbe; se il pad sta sulla stessa fascia, accanto al pad se resta larga almeno 320 px (`CAPTION_MIN_WIDTH`), altrimenti **sopra** il pad su tutta la parte visibile; mai sotto i grafici aperti; nascosta sotto 140 px (`CAPTION_HIDE_WIDTH`); sul telefono a sinistra del pulsante del pad e sopra il codice ridotto | `navMath.captionPlacement` → `--nav-caption-right`, `--nav-caption-bottom`, `.nav-caption-squeezed` sulla radice del tab |
  | Codice con i grafici aperti (≥ 901 px) | ridotto alla sua intestazione e **agganciato in fondo alla colonna dei grafici** (z-index 19, sopra i grafici); ▁ lo riapre lì (al più il 45% dell'altezza); il fondo dei grafici gli lascia posto (`--code-dock-h`); chiusi i grafici torna com'era, salvo una scelta dello studente con ▁ | `CodePanel.followCharts` (collegato da `createTabLayout`) + `core.css` |
  | Barra strumenti e selettore dello zoom con i grafici aperti | la barra scende nella seconda fila, sotto ☰; lo zoom resta a sinistra dei grafici, compatto, e scorre in orizzontale se non ci sta | `core.css` |
  | Chip in alto (benvenuto, avvisi, "Ti sei perso?") | nella parte non coperta (`right` = `coveredRight()`); in uno spazio stretto (container query ≤ 560 px) il testo sta sopra e i pulsanti sotto; in un livello delle rappresentazioni accanto alla scheda della vista se restano 360 px, altrimenti sotto | `css/nav.css`, `NavPad.placeHud()` |
  | Avviso globale (`.app-toast`) | mentre è visibile le didascalie dei tab tacciono (stavano sotto) | `css/palette.css` |
  | Pannelli | `--panel` è opaco (`#0f172a`): al 94% la scena traspariva sotto il codice (numeri delle matrici sopra le righe) | `css/core.css` |
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
ui.charts.setSectionVisible("full", false); // nascosta: non occupa posto e non disegna (senza sezioni visibili:
                                           // "Nessun grafico…"); serve a chi ha due scene nello stesso tab
ui.charts.reveal("training");              // apre il pannello sulla sezione (e la rende visibile); { open: false }
                                           // scorre soltanto; si riallinea dopo il primo disegno
loss.setSeriesColor("A", PALETTE[0]);      // colore fisso per una serie, anche prima dei dati (o color nel dato)
```
Colori automatici delle serie: lo slot della palette alla prima comparsa del nome, poi sempre quello (i
colori fissati a mano non consumano slot), così "A" e "B" non si scambiano quando arriva prima B. In scala
logaritmica le etichette dei tick troppo vicine si saltano tenendo prima quelle tonde (`tickPriority`,
`keepTickLabels` di `chartMath.js`, pure). `chartMath.histogram` conta con `histogram` di
`shared/weights/stats.js` (stessa regola dei bin, estremi compresi; i valori fuori dall'intervallo dichiarato
escono dai bin estremi): una sola regola per gli istogrammi di tutta la web app.
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
await reps.refresh("network");   // la specifica esce dalla cache; se il livello è aperto la vista si rifà sulla
                         // STESSA pila di scena, con la camera di prima → true (false: superato da un altro refresh)
reps.view; reps.stack;   // la vista viva del livello e la sua pila { host, director, nav } (test, console) o null
reps.dispose();          // toglie tutto e rimette i metodi presi in carico (le pagine di prova)
```

| Opzione | Uso |
|---|---|
| `tabId` | id del tab (predefinito: `data-tab` della radice); dà anche il nome alle foto (`scena-rag-rete-….png`) |
| `app` | l'oggetto `app` del guscio: arriva alle viste (`ctx.app`) e al pannello Cervello |
| `main: { label, description }` | testi della voce Architettura (predefiniti: "Architettura" e una frase) |
| `network` / `brain: { load }` | `load()` → Promise della specifica di soli dati; la vista è del nucleo (`core/neuralNet/view.js`, `core/brain/view.js`), caricata alla prima apertura in un livello sovrapposto |
| `network` / `brain: { load, cache: false }` | `load()` a **ogni** apertura del livello (predefinito `cache: true`: una volta per pagina). Per le specifiche che dipendono da ciò che lo studente ha fatto (la domanda del RAG, il brano dell'Audio, il run e lo step del Laboratorio, il modello scelto in LLM moderni); con il livello aperto il tab chiama `reps.refresh(id)` |
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
- Impostare `nav.setHome(fn)` ("Centra") e, se i confini automatici (`meshBounds`, che conta tutte le istanze
  visibili di un `InstancedMesh`) non vanno bene, `nav.setBounds(box)`; con un `THREE.Points` o geometrie
  scritte dallo shader, inquadrature e confini su un `Box3` esplicito (come fa la rete neurale).
- `reps.refresh(id)` rifà la vista sulla stessa pila: niente contesto WebGL nuovo né perso, la scheda dice
  "Aggiorno la scena…" nel frattempo, la camera e la posa di "Centra" restano quelle di prima; più refresh
  ravvicinati: vince l'ultimo.
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

**Link diretti e ricerca.** `REP_LINKS = { rete: "network", cervello: "brain" }` (in `core/repLinks.js`,
soli dati, con `REP_IDS`, `BRAIN_PANE_ID = "cervello"` e `repOfLink(target)`; `representations.js` li
riesporta): il `goto` del tab inizia con `reps.goto(target)`, che per `view:rete` / `view:cervello` mostra la
rappresentazione e per ogni altro elemento chiude subito il livello e restituisce `null` (il `goto` prosegue
come sempre). Le voci vanno anche in `search.js` del tab (`{ id: "rete", kind: "view", … }`,
`{ id: "cervello", kind: "view", … }`).
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
seconda fila a sinistra sotto ☰, poi un menu compatto (`menu`/`menuitemradio`); se le prime tre file sono
piene, un **posto di riserva** per il menu nella prima fila libera fino all'ottava (`RESERVE_ROWS`), il più a
sinistra; solo se non c'è neanche quello il menu va sopra un comando (`fits: false`). Gli ostacoli sono gli
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
  l'ultimo davanti. Le sezioni usano lo stesso passo in z: la posizione 3 sta alla stessa z ovunque e il
  suo percorso si segue a occhio. Eccezione: una sezione che campiona lo **stesso tempo** con un altro numero
  di posizioni (Audio: EnCodec 18 e 36 posizioni, log-Mel 24, contro 12 fette) si stende sulla stessa
  profondità con un passo proporzionale (`proportionalDepth`, `depthStep` di `layout.js`), così lo stesso
  istante sta alla stessa z anche nella geometria.

Di lato la rete si legge come un diagramma classico; girando la camera ("Guarda lo spessore") si vede lo
spessore con l'attention tra token, le convoluzioni tra vicini, l'instradamento verso gli esperti, il
pooling che schiaccia la profondità, i residui che saltano le sezioni.

| Modulo (`js/core/neuralNet/`) | Esporta | Ruolo |
|---|---|---|
| `spec.js` | `normalizeSpec(spec)`, `NetworkSpecError` (`.errors`), `describeSpecError`, `LIMITS`, `DEFAULT_DENSITY`, `CARD_MODES`, `sliceUnits` | validazione e valori predefiniti; TUTTI gli errori in una volta, con il percorso del campo (puro) |
| `layout.js` | `computeLayout(net)`, `SPACING`, `unitIndex`, `decodeUnit`, `sliceIndex`, `unitPosition`, `boxOf`, `takesPartInSlices`, `proportionalDepth`, `depthStep` | posizioni x/y/z di ogni neurone, lastre fantasma dei blocchi ripetuti, parentesi, altezze dei residui, passo in z proporzionale (puro) |
| `activations.js` | `resolveValues(net)`, `simulateAttention`, `attentionWeights`, `computeRouting`, `MODE` | valori veri o simulati con seme, instradamento top-k degli esperti, pesi di attention (puro) |
| `links.js` | `buildLinks(net, values, limits)`, `LINK_LIMITS`, `ARC_KIND`, `pairDepths` | segmenti e archi per schema, deterministici, ordinati per priorità, con tetto (puro) |
| `model.js` | `buildNetworkModel(spec, {limits})` | tutto il modello in un colpo: `{net, layout, values, links, timings}` (puro) |
| `pickIndex.js` | `ScreenIndex`, `hoverUnit` | indice sullo schermo per il neurone sotto il mouse; quale neurone raccontare al passaggio quando sotto il puntatore c'è anche un titolo (puro) |
| `framing.js` | `READABLE`, `readableHeight`, `walkParts`, `partTitle`, `partIndexAt`, `avoidRect`, `avoidRectVariants`, `fitInFreeRect` | inquadratura delle reti lunghe a parti e del pad da evitare; `fitInFreeRect({halfW, halfH, depth, rects, W, H, covered, tanV, margin})` → `{rect, distance, dx, dy}`: un riquadro di fronte nella parte libera dove viene più grande (la usa anche LLM moderni per la griglia) (puro; vedi "Reti lunghe") |
| `honesty.js` | `valueSources`, `badgeOf`, `honestyLines`, `attentionSource`, `plural`, `listText` | da dove vengono valori e pesi di attention: badge a tre stati e frasi di "Scala e onestà" (puro) |
| `occlusion.js` | `occluderOpacity`, `distanceToBox` | quanto attenuare le lastre fantasma che stanno tra la camera e il punto guardato (puro) |
| `roles.js` | `ROLES`, `PATTERNS`, `HEAD_COLORS`, `ACTIVATION`, `LINK_COLORS`, `AXIS_COLORS`, `linkSummary` | vocabolario e colori (soli dati); `linkSummary(links, pick)` raggruppa "Riceve da / Manda a" per sezione |
| `scene.js` | `NetworkScene`, `LOD_PX`, `sizeText` | la scena three.js (vedi "Uso dentro la scena di un tab"); `sizeText` con singolare e plurale ("1 neurone", "1 canale") |
| `geometry.js`, `shaders.js`, `labels.js` | `arcLines`, `ghostBox`, …; `createSharedUniforms`, …; `LabelLayer` | geometrie (un oggetto per tutta la rete; gli archi della cross-attention in un oggetto loro), shader con uniform condivise (anche `uGhostFade`), etichette in un atlante |
| `card.js` | `NetworkCard`, `startsCollapsed`, `LONG_NETWORK_RATIO` | la scheda (solo DOM) e quando parte ridotta (puro) |
| `view.js` | `createNetworkView(ctx)`, `VIEW_DIRECTIONS` | la vista per `core/representations.js` |
| `demoSpecs.js` | `LLM_SPEC`, `UNET_SPEC`, `AUDIO_SPEC`, `loadTestSpec()`, `DEMO_SPECS` | tre specifiche di riferimento commentate (DeepSeek-V3 ridotto, U-Net di Stable Diffusion, EnCodec) + la prova di carico |

#### La vista (per `core/representations.js`)

```js
const view = createNetworkView({ host, director, nav, layer, tabId, spec, app, reps });
await view.ready;          // scena costruita, camera di lato su tutta la rete. Specifica sbagliata:
                           // view.error (NetworkSpecError) e la scheda elenca gli errori; ready si risolve lo stesso
view.focus("attn");        // oppure ["a", "b"], { section: "attn", slice: 3 } o { group: "Encoder" | 2 }: vola lì
                           // (stesso angolo) e mostra la scheda; un gruppo sconosciuto → false
view.onShow(); view.onHide();   // onHide ferma l'onda e nasconde l'etichetta
view.dispose();            // scena, texture, DOM, listener; rimette home, confini e limiti della navigazione
view.stats                 // { buildMs, model: {spec, layout, values, links}, tickMs, updateMs, updates, pickMs, units, segments, arcs }
view.play(); view.side(); view.thick(); view.setSlice(t | null); view.setDensity(f); view.select(id); view.measureIdle(n)
view.home();               // "Centra": tutta la rete o, se è lunga, la parte 1 (nav.setHome(() => view.home()))
view.walk(+1 | -1);        // parte dopo / prima, con l'angolo di adesso; dalla vista d'insieme riparte dalla prima
view.whole();              // "Tutta la rete" di lato
view.parts                 // { long, parts, rect, key } per lo stage di adesso (null prima di ready)
```
`host`, `director`, `nav` sono quelli del livello: `nav.setHome` = la vista iniziale (sotto), `nav.setBounds`
= la rete con i suoi assi, `nav.setOrbitLimits({ turn: 85, tilt: 82, minDistance: 0.5 })` (mai da dietro: il
flusso si leggerebbe al contrario). Nel `layer` la vista mette la scheda (`.nn-card`, in alto a sinistra sotto
`--rep-safe-top`; chi la ospita altrove può spostarla con `--nn-top` e, sotto 640 px, `--nn-top-narrow`) e
l'etichetta del mouse (`.nn-tip`). La scheda è il primo figlio del livello (prima di canvas e pad): con Tab dal
selettore si arriva subito ai suoi comandi. La camera inquadra la rete nella parte dello stage non coperta
dalla scheda né dal pad, a destra o sotto, dove la rete viene più grande (conta la forma: una rete lunga e
bassa sta meglio sotto una scheda ridotta); sul telefono sotto l'altezza massima della scheda (52vh), così il
dettaglio aperto da un tocco non la copre (resta una fascia vuota tra scheda ridotta e rete: è voluto).

Interazione: passando sopra un neurone, etichetta con sezione, lastra/head/esperto, indice, posizione in
profondità (nome del token), valore ("simulato" se lo è); con `neuronNames` il nome viene prima del numero
(«voce 2» · 3 di 64). Il titolo di una sezione è più largo della sua colonna e di lato passa sopra i neuroni delle sezioni
accanto: lì l'etichetta al passaggio racconta il neurone sotto (`hoverUnit`; prima, nell'Audio a 1280 px, «Forma
d'onda» copriva senza etichetta il 40% di «Spettro del frame»), mentre il clic resta del titolo disegnato sopra e
l'etichetta lo dice («Clic qui: sceglie «…», il titolo disegnato sopra»). Clic su un neurone, sull'etichetta di una sezione, di un gruppo o di un "× N" = scheda
con il suo testo, i numeri disegnati e veri e i collegamenti in entrata e in uscita, una voce per sezione
collegata (`linkSummary`: dieci livelli uguali di una U-Net diventano "10 × …"; la sezione si evidenzia, le
altre si spengono); clic sul nome di un token = quella fetta; clic sul vuoto = chiude. Sul telefono un tocco fa
lo stesso e mostra l'etichetta per 2,6 s; il dettaglio resta visibile anche nella scheda ridotta. Scheda:
titolo, badge dell'onestà (sotto), introduzione, comandi (**▶ Anima** e velocità, **Guarda lo spessore** = tre
quarti, **Di lato**, **Fetta** = una posizione della profondità e il suo percorso, densità dei
**Collegamenti**), per le reti lunghe la riga delle parti, dettaglio, legenda costruita da ciò che la rete
contiene, "Scala e onestà", indice delle sezioni (da tastiera si arriva a tutto), fonti. Parte ridotta
secondo `card` della specifica (sotto); ridotta nasconde anche la riga "Collegamenti" ("Dettagli" la apre).

**Reti lunghe: vista iniziale leggibile a parti.** Una rete lunga e bassa (DeepSeek-V4.1-Flash 240 × 17
unità, Audio 133 × 21) vista intera di lato sarebbe una striscia di 25-90 px. La vista iniziale e "Centra"
sono allora una **finestra** larga quanto serve perché la rete sia alta `readableHeight(parte libera)` px (metà
della parte libera, tra 150 e 400, al più il 90%), che comincia dall'ingresso; ◀ ▶ della scheda scorrono le
parti, "Tutta la rete" mostra l'insieme. Una rete che entra quasi tutta (≤ 1,04 finestre) resta intera.
```js
import { readableHeight, walkParts, partIndexAt, avoidRect, avoidRectVariants } from "../core/neuralNet/framing.js";
walkParts({ items, span, width, groups })   // → [{ x0, x1, title }] in unità della rete; items: sezioni e
                                            //   ripetizioni { x0, x1, label } nel flusso; groups: le parentesi
partIndexAt(parts, x)                       // la parte con il centro più vicino a x
avoidRectVariants(free, pad)                // [sopra il pad, a sinistra del pad] (o [free] se non si toccano)
avoidRect(free, pad)                        // uno dei due: dove resta più area
card.setWalk({ index, count, title } | null)   // la riga "◀ Parte 1 di 4 ▶ · Tutta la rete" (index -1 = insieme)
```
Ogni parte comincia (0,6 unità prima) con il primo blocco che la precedente non mostrava per intero, almeno
mezza finestra più avanti; l'ultima finisce con l'uscita e sostituisce la penultima se quasi coincidono (< ¼
di finestra); il titolo sono le parentesi dei gruppi il cui centro cade nella parte (al più tre), altrimenti
"prima … ultima sezione". Il piano delle parti si rifà solo se cambiano lo stage o lo stato della scheda
(ridotta o no): "Centra" torna **esattamente** alla vista iniziale. Il punto guardato è quello al centro della
parte libera, alla profondità del punto attorno a cui gira la camera; se nella parte libera entrano più di 1,3
parti si sta guardando l'insieme ("Insieme: N parti", ◀ spento, ▶ riparte dall'ingresso). La riga di stato è
`aria-live` ("Parte 2 di 4: Layer Full, 5 layer Reuse."). Misurato (altezza della rete dopo "Centra", prima →
dopo, 1600×900): LLM moderni 40 → 162 px, Audio 87 → 195, Immagini e video 81 → 199, Laboratorio 122 → 213; sul
telefono 25-70 → 111-130 px.

**Onestà dei valori: badge a tre stati.** Il badge è un **pulsante** che apre "Scala e onestà" (la scheda si
apre, il riquadro va in vista, il focus va sul suo titolo): "Attivazioni vere" (verde), "Attivazioni: in parte
vere" (verde acqua, `.nn-badge.mixed`), "Attivazioni: simulazione didattica" (giallo). "In parte vere" vale
anche con tutte le attivazioni vere e qualche peso di attention illustrativo o simulato. Le frasi di "Scala e
onestà" elencano le sezioni vere ("Valori veri in 15 sezioni su 52: Forma d'onda, … e altre 3. Le altre 37
sono una simulazione didattica (seme 20)."); nel dettaglio di una sezione la riga "Attention" dice "veri,
calcolati dal modello" / "illustrativi: …" / "simulati con un seme".
```js
import { valueSources, badgeOf, honestyLines, attentionSource, plural, listText } from "../core/neuralNet/honesty.js";
const src = valueSources(model);   // { state: "real" | "partial" | "simulated", real, simulated,
                                   //   attention: { real, illustrative, simulated }, illustrativePairs }
badgeOf(src, seed);                // { text, className, title }
honestyLines(src, seed);           // frasi di "Scala e onestà": la prima sempre sui valori
attentionSource(link);             // "real" | "illustrative" | "simulated"
plural(1, "neurone", "neuroni");   // "1 neurone";  listText(items, 12) → "a, b e altre 3"
```

#### Specifica (`networkSpec`, soli dati)

```js
export const networkSpec = {
  title: "…",                 // obbligatorio
  intro: "…",                 // 1-3 frasi
  scale: "…",                 // onestà sulla scala: "16 numeri per token invece di 7.168" (senza: avviso)
  note: "…",                  // facoltativa, in "Scala e onestà"
  seed: 7,                    // seme delle simulazioni e dei campioni (predefinito 1)
  card: "auto",               // "auto" | "open" | "collapsed" (CARD_MODES): sul telefono sempre ridotta; "auto" anche
                              // sugli stage sotto 980 px e nelle reti lunghe ≥ 10 volte la loro altezza (LONG_NETWORK_RATIO)
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
| `slices` | `false` | le posizioni della sezione non sono quelle delle fette (i token di un encoder accanto ai chunk di un database, le parole del prompt accanto ai frame dell'audio): non si accende né si spegne con la fetta. Se nessuna sezione partecipa, la rete non ha fette |
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

**Collegamento** `{ from, to, pattern, density?, fromSlab?, toSlab?, label?, illustrative?, … }` (`within` solo
per la self-attention). Si esce dall'ultima lastra di `from` e si entra nella prima di `to` (`"all"` o un indice
per cambiare; negli esperti tutte). `density` (0…1) è il campione disegnato; `label` compare nella scheda.
`illustrative: true` su un `attention` con `weights` espliciti (o su un `explicit`) dice che quei numeri mostrano
lo schema e non vengono da un modello (LLM moderni, la cross-attention e il blocco video di Immagini e video): il
badge diventa "in parte vere". Non booleano = errore; su altri schemi, o su un'attention senza `weights` (già
simulata), un avviso e il campo si ignora. Lo schema `identity` collega le lastre scelte con `fromSlab`/`toSlab`
(con numeri di unità diversi si collegano le prime min, con un avviso).

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
  profondità di altro tipo (i token del prompt in una U-Net a griglia) o con `slices: false` non si accendono
  con la fetta (`takesPartInSlices`). I nomi della profondità (token, chunk, frame) stanno accanto alle sole
  sezioni che partecipano (`scene.sliceRun = { x0, x1 }`), non ai due estremi della rete.
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
- **Archi e lastre che non coprono la scena**: gli archi della cross-attention sono un oggetto loro
  (`scene.crossArcs`): di lato (`scene.sideness` ≈ 1, camera quasi sull'asse z) scendono al 30% e formano una
  fascia tenue invece di una tenda piena; di tre quarti, con una fetta o con una sezione scelta tornano pieni. Le
  lastre fantasma di un "× N" si attenuano fino al 12% quando stanno tra la camera e il punto guardato
  (`occluderOpacity(camera, focus, box, { roi: 1.5, floor: 0.12, samples: 24 })` di `occlusion.js`; restano
  piene di lato e se il blocco è scelto): l'attributo `aGhost` dice a ogni vertice fantasma quale colonna di
  `uGhostFade` leggere, quindi un cambio è una texture da 1 × ripetizioni, solo a camera mossa. Sulle griglie
  `simulateAttention` alterna head locali e di contenuto (niente head "prima posizione", che disegnava un
  ventaglio verso un angolo).

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
net.update(camera, stage.clientWidth, stage.clientHeight, renderer.getPixelRatio(), obstacles?, focus?);
//   obstacles: [[x0, y0, x1, y1]…] da lasciare liberi; focus: il punto guardato (mondo), per le lastre fantasma
net.setSlice(t | null); net.setEmphasis(["attn"] | null); net.setDensity(0.4); net.setWave(x | null); net.setGlow(false);
net.setSliceDensity(0.25);                    // pesi "explicit" della fetta: i più forti per |peso|, stessa quota per
net.setSliceDensity((link, n) => counts[link.index]);   // ogni collegamento, o quanti per collegamento; senza
net.sliceDrawn; net.sliceShown;               // ricostruire (2-3 ms): quanti si vedono, quale peso sta in ogni posto
const u = net.pick(event.clientX, event.clientY, camera, canvasRect);   // unità o -1 (µs; indice rifatto dopo un movimento)
net.describe(u);                              // { section, slab, neuron, depth, row, col, depthName, head, expert, value, simulated }
net.setSectionValues("ln1", float32);         // nuovi valori veri della stessa forma (un altro prompt): ricolora senza ricostruire
net.box(ids?) / net.framePoints(ids?);        // per inquadrare (THREE.Box3 / punti nel mondo)
net.dispose();
```
Dopo `setSlice`/`setEmphasis` si mette `net.needsUpdate = true` per ridisegnare le etichette al frame dopo.
`setSliceDensity` usa un rango per |peso| calcolato una volta e riscrive solo i posti visibili e il
`drawRange` (con quota 1 il disegno è quello di sempre): il Transformer, alla prima quota più alta di quella
costruita, ricostruisce la rete una volta alla quota massima e da lì cambia quota in 2-3 ms (prima ogni cambio
ricostruiva: 445-1045 ms con il rendering software). Per cambiare la struttura (altre sezioni, altri pesi di attention, un altro prompt) si ricostruisce:
`buildNetworkModel` + una nuova `NetworkScene` (~10-60 ms per le reti dei tab, ~120 ms per la prova di carico);
aggiornare sul posto archi di attention e lookup risparmierebbe ~40 ms ma cambierebbero numero di archi, nomi
dei token nell'atlante, valori e cache del pick, quindi non si fa.

**Prove.** `tests/neuralnet.test.mjs` (specifica ed errori, disposizione, blocchi ripetuti, attivazioni e
instradamento, collegamenti deterministici, indice sullo schermo, esempi; `identity` con le lastre, `slices:
false`, `illustrative` e `card`, passo proporzionale, onestà, reti lunghe, occlusione, `startsCollapsed`,
`linkSummary`); `tests/neuralnet-scene.test.mjs` (three.js da `vendor/`, scena senza etichette:
`setSliceDensity` coppia per coppia, archi della cross-attention, lastre fantasma, nomi delle fette, `meshBounds`
con le istanze, API del director e della navigazione). `tests/neuralnet-demo.html`
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
| `shared/weights/formats.js` | `parseWeightsManifest`, `decodeWeights`, `tensorsFromLayout`, `parseTrace`, `decodeSnapshot`, `parseRunInfo`, `parseBenchmark`, `MAX_RUN_NOTES`; riesporta `parseSafetensors`, `safetensorsInfo`, `recognizeFiles`, `readJsonFile`, `readBinaryFile`, `filePath`, `LIMITS`, `WeightsFormatError`, `canonicalName`, `withoutBidiControls`, `stripBidiControls` | i formati (tabella sotto) |
| `shared/weights/safetensors.js` | `parseSafetensors(bytes, {file, limits})`, `safetensorsInfo(metadata)`, `SAFETENSORS_DTYPES` | `.safetensors` F32/F16/BF16 → `Float32Array` |
| `shared/weights/files.js` | `recognizeFiles(files, {limits})`, `readJsonFile`, `readBinaryFile`, `filePath` | file caricati → gruppi di run, senza leggerli |
| `shared/weights/validate.js` | `LIMITS`, `WeightsFormatError`, `checkLayout`, `checkShape`, `checkName`, `checkRelativePath`, `cleanConfig`, `checkVocab`, `canonicalName`, `naturalCompare`, `withoutBidiControls`, `stripBidiControls` | controlli comuni (dati non fidati) |
| `shared/weights/run.js` | `WeightRun`, `loadBuiltinRun`, `runsFromFiles`, `engineCompatibility`, `inferGptConfig`, `snapshotBracket`, `snapshotEntry`, `fingerprint`, `BUILTIN_RUN_ID` | l'oggetto run |
| `shared/weights/stats.js` | `tensorStats`, `histogram`, `percentiles`, `range`, `absMax`, `l2Norm`, `DEFAULT_PERCENTILES` | statistiche di un tensore |
| `shared/weights/compare.js` | `compareTensors(a, b, {top, tolerance, trace, groupOf})`, `tensorDiff`, `difference` | confronto A/B (run o fasi del training) |
| `shared/weights/groups.js` | `groupOf`, `groupsFromTrace`, `grouper` | gruppi di parametri come `gpt/recording.py` |
| `shared/weights/moduleTree.js` | `buildModuleTree(params, {snippets, config})`, `findNode`, `walkTree`, `pythonPath`, `ROOT_CLASS`, `ROOT_SNIPPET` | albero dei moduli con la riga Python vera |
| `shared/weights/pythonSource.js` | `parseModuleSource`, `moduleSources`, `splitComment` | lettura ristretta degli snippet Python di `model.json` |
| `shared/weights/store.js` | `RunStore`, `serializeRun`, `deserializeRun`, `RECORD_FORMAT` | run caricati salvati nel browser (IndexedDB) |
| `shared/weights/display.js` | `formatCell`, `formatPrecise`, `chooseCellFormat`, `textColorFor`, `textContrastFor`, `contrastRatio`, `relativeLuminance`, `quantizeSigned`, `quantizeUnsigned`, `matrixShape`, finestre di celle (`windowFromPoints`, `capWindow`, `expandWindow`, …), `TEXT_DARK`, `TEXT_LIGHT`; `shownChar(c)` | come si scrivono i numeri nelle celle e i caratteri del vocabolario (␣ ↵ ⇥; un carattere invisibile o bidirezionale come «U+202E»: il vocabolario di un run caricato viene dai file dello studente); puro |
| `core/matrixNumbers.js` | `NumericMatrix`, `attachNumbers`, `NumbersGroup`, `colormapLUT` | matrici 3D con i numeri (three.js) |

#### Formati supportati

| File | Contenuto | Controlli (errori in italiano con file e campo) |
|---|---|---|
| `weights.json` + `.bin` | manifest di `gpt/export.py` (anche di `scripts/export_run.py`), float32 little-endian, layout `{name, shape, offset, numel}` in NUMERO DI FLOAT | `format_version` 1, `dtype` float32, `byte_order` little; `file` relativo (niente `/…`, `..`, `\`, `C:`, `https:`); nomi unici e senza caratteri di controllo; forme di interi ≥ 0 (al più 8 dimensioni); `numel` = prodotto della forma; intervalli dentro il file, **senza sovrapposizioni né buchi** (somma dei `numel` = `total_params`); binario lungo esattamente `4 × total_params` byte; `config` (solo valori semplici, interi positivi) e `vocab` (stringhe uniche, lunghezza = `vocab_size`) facoltativi. **Qualunque architettura**: la compatibilità con il motore si valuta a parte |
| `trace.json` + `snap_XXXXX.f16.bin` | storia del training di `gpt/recording.py` | `format_version` 1, `snapshot_dtype` float16, layout come sopra; snapshot `{step, file}` con step interi unici (ordinati) e file accanto a `trace.json`; ogni snapshot lungo esattamente `2 × total_params` byte (conversione esatta da float16). Le metriche (`train_loss`, `evals`, `metrics`, …) restano quelle del file |
| `run.json` | descrizione di un run esportato | campi letti, tutti facoltativi: `name`, `description`, `created` (o `date`/`created_at`), `source`, `config`, `vocab`, `notes` (stringa o lista), `modules` (o `module_tree`, tenuto com'è), `engine_compatible` (booleano: `false` = calcolo diverso dal GPT del corso anche con forme uguali); il JSON originale in `raw`. Nome, descrizione, data e sorgente escono senza i controlli bidirezionali espliciti (`withoutBidiControls`: finiscono dentro le frasi dell'interfaccia); le note restano com'erano |
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
run.snapshots;          // [{step, file, meta (loss, top1, sample_text…), available, layoutMatches, load(), unload(), loaded, raw()}]
run.steps;              // step degli snapshot di cui ci sono i pesi: [0, 50, …, 5000]
await run.loadSnapshot(500);                   // {step, meta, flat, tensors, layout} (in cache; un errore non resta in cache)
run.unloadSnapshots({ keep: [500] });          // run grandi: toglie dalla cache gli altri snapshot decodificati (→ quanti);
                                               // chi ne tiene un riferimento lo conserva, il prossimo load() li rilegge
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
// text è la riga com'è nel file, commento in coda compreso: per token_embedding_table
// "self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)  # (V, C)", con comment "(V, C)"
// (chi mostra solo il codice usa statement, o toglie il commento)
findNode(tree, "blocks.0").code;          // {className: "Block", text: "class Block(nn.Module):", createdBy: {text: "self.blocks = nn.Sequential(…)"}}
findNode(tree, "blocks.0.sa.heads.0").code.buffers;   // [{name: "tril", line, text, statement: "self.register_buffer(\"tril\", …)", snippet: "head", file: "gpt/attention.py"}]
findNode(tree, "blocks.0.sa").code.noWeights;         // [{attr: "dropout", type: "nn.Dropout", line: 75, text, snippet: "mha", file: "gpt/attention.py"}]
// buffers e noWeights dicono il file della LORO classe (il Dropout di sa sta in gpt/attention.py, non in gpt/block.py)
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
risponde `{ok: false, reason}`; la pagina funziona con i run in memoria. Il costruttore non lancia mai: dove
anche solo leggere `indexedDB` è vietato (iframe in sandbox, alcune navigazioni private: `SecurityError`)
`store.available` è false e `store.lastError` è `{fatal: true, message: "IndexedDB vietato dal browser
(SecurityError): …"}`.

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
- **Nessun lavoro a riposo**: `update` confronta camera, viewport e la **versione del gruppo** (un contatore
  che ogni `setData`/`touch` di una matrice incrementa; delle heatmap di `attachNumbers` si guardano solo le
  versioni delle texture) con l'ultima volta ed esce subito, senza interrogare le matrici una per una (con 110
  matrici a riposo). `setData` / `setRange` / `refresh()` e, per `attachNumbers`, `texture.needsUpdate = true`
  sulla heatmap (come fa `SceneTextures.writeWeights` del Transformer) ridisegnano anche a camera ferma;
  matrici spostate da codice senza muovere la camera: `numbers.invalidate()`.
- **Ridisegno parziale**: cambiando step (stessa finestra e risoluzione) l'overlay riscrive solo le celle il
  cui testo o colore è cambiato e ricarica la texture sulla GPU solo se almeno una è cambiata; ridisegno
  completo se cambia il formato (cifre, larghezza del testo più lungo). `numbers.stats.partialRedraws` accanto
  a `redraws`.
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
| | `training_notes(trace)`, `TRAINING_NOTE_PREFIX`, `MAX_TRAINING_NOTES` | le note del trace copiate in `run.json` (sotto) |
| | `module_tree(weights)`, `readable_json(value)`, `slugify(name)`, `RUNS_DIR`, `RUN_FILE`, `BENCHMARK_FILE` | albero dei moduli, JSON leggibile, nomi di cartella sicuri |
| `gpt/export.py` | `display_path(path)` (anche `gpt.runs.display_path`) | il percorso da scrivere nei JSON (`model.json`, `weights.json`, `run.json`, `benchmark.json`) senza la cartella personale di nessuno: relativo al progetto, oppure `…/gpt.pt` (la cartella solo dal terzo livello e se non è la home) |
| `gpt/benchmark.py` | `benchmark_path(path, *, name, settings, text, excerpt_path, snapshots)`, `benchmark_model(model, tokenizer, *, name, source, validation, excerpt, history, weights, settings)` | il benchmark di un `.pt`, di una cartella di run o di un modello in memoria → contenuto di `benchmark.json` |
| | `BenchmarkSettings(prompt="ROMEO:\n", seed=1337, max_new_tokens=200, temperature=1.0, batch_windows=512, device="cpu", threads=None)` | le scelte fisse (cambiarle rende i numeri non confrontabili) |
| | `contiguous_windows(n, block_size)`, `score_tokens(model, ids, *, block_size, token_chars, batch_windows, surprisal)` → `Scores`, `generation_sample(...)`, `greedy_continuation(...)` | il protocollo, pezzo per pezzo |
| | `validation_text(text)`, `excerpt_bounds(validation)`, `read_excerpt(path, validation)`, `EvalText` | i testi valutati, con sha256 |
| | `open_source(path)`, `build_model(weights)`, `run_tokenizer(weights, default_vocab)`, `ArchitectureMismatch` | i pesi eseguiti dalla classe ATTUALE (nomi e forme identici, altrimenti errore che elenca le differenze); un tensore intero o booleano dello `state_dict` (es. `num_batches_tracked` di una BatchNorm) non rende il run "non ricostruibile" e non fa saltare gli snapshot: resta il valore del codice, con una nota |
| | `code_files(model)`, `code_sha256(files)` (alias `_code_files`, `_code_sha256`) | i file dei moduli del modello e la loro impronta (`model.code_sha256`): la usa anche `scripts/record_training.py` |
| | `comparison_rows(a, b)`, `comparison_warnings(a, b)`, `format_comparison(a, b)`, `format_summary(bench)`, `format_int(n)`, `terminal_text(text)` | confronto A/B e testo per il terminale, in italiano (senza caratteri di controllo: vedi "Sicurezza") |
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
codice attuale solo se questo ricostruisce i pesi (altrimenti `null`).

**Le note di `run.json`** (`notes`, in quest'ordine): quelle dell'autore (`--note`); poi, con `--trace`, **le
note della storia del training** (`notes` di `training/trace.json`, scritte da `scripts/record_training.py`
quando ha dovuto adattarsi: gruppi nuovi, pesi condivisi, attention letta da moduli nuovi o saltata, valori
non finiti), una per nota con il prefisso `"storia del training: "` (`TRAINING_NOTE_PREFIX`); infine gli avvisi
automatici dell'export: conversioni da float16/bfloat16, voci saltate (interi come `num_batches_tracked`),
valori non finiti, prefissi tolti, storia di un altro training, `benchmark.json` misurato su altri pesi, file
estranei. Le note del trace sono testo letto da un file (`training_notes(trace)`): solo stringhe, su una riga
(spazi al posto dei caratteri di controllo, "�" al posto dei surrogati UTF-16 spaiati), al più 2.000 caratteri
ciascuna e al più 20 (`MAX_TRAINING_NOTES`; le altre contate in una nota); `training/trace.json` resta la copia
esatta e `export_run.py` le stampa (`nota: storia del training: …`). Ogni testo che finisce in `run.json`
(note, `device`, nome, descrizione, percorso del checkpoint) si può scrivere in UTF-8: un byte non UTF-8 della
riga di comando non lascia più un `run.json` vuoto dopo pesi e storia. Il Laboratorio le mostra nella scheda
del run (vedi "Il tab Laboratorio pesi": spiegano, per esempio, perché manca la heatmap dell'entropia
dell'attention).

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

**Controllo di coerenza nel browser** (Laboratorio pesi, `checkConsistency` di
`tabs/lab/analysis/protocol.js`): per un run compatibile con il motore JS si ricalcola l'estratto con lo stesso
protocollo e si confronta con `excerpt` di `benchmark.json` (o di `builtin.json` per il GPT del corso, anche per
ogni snapshot). Il verdetto ha **due livelli** di accordo (`TOLERANCES`), più il disaccordo:

| Verdetto | Condizione | Che cosa vuol dire |
|---|---|---|
| ✓ (`ok`) | `\|Δ loss\|` e `\|Δ entropy\|` ≤ 1e-5 (`\|Δ bpc\|` ≤ 1e-5 / ln 2), previsioni giuste top-1 e top-5 **identiche**, `windows` e `predictions` identici, bit di sorpresa carattere per carattere entro 2e-3 (il file li salva con 3 decimali), testo `greedy` identico | lo stesso calcolo, a meno dell'ordine delle somme in virgola mobile |
| ≈ (`near`) | almeno una misura oltre la soglia di ✓ ma entro le tolleranze di `benchmark_differences` di `gpt/benchmark.py`: `\|Δ loss\|`, `\|Δ entropy\|` ≤ 1e-4, previsioni giuste entro ±2 | coerente: un quasi pari merito tra due logits può cambiare una previsione |
| ✗ (`bad`) | oltre: anche una sola finestra, previsione, bit o carattere del greedy diverso | calcoli diversi (es. pesi letti male, un'altra attivazione) |

Il testo campionato non si confronta (PyTorch e il motore JS hanno generatori casuali diversi, lo si dice) e il
riferimento del Python vale solo se l'sha256 dell'estratto coincide con quello di `benchmark.json`. Misurato con il
motore vero: `|Δ loss|` ≤ 5e-7 sui pesi finali e su tutti gli 11 snapshot (✓ ovunque). Implementazione di
riferimento (quella che usa `tests/test_benchmark.py`):
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
  tutto, poi si scrive: un errore non lascia cartelle a metà; `--force` cancella solo i file del run, e con
  `--trace <run>/training` (riesportare un run sul posto) la storia resta dov'è: prima si cancellava prima di copiarla.
  Le stampe nel terminale (riepilogo e confronto di `gpt/benchmark.py`, riepilogo di `scripts/export_run.py`)
  passano da `terminal_text`: un ESC o un U+202E in un `benchmark.json` altrui (o nel vocabolario di un run
  scaricato) si legge come `\x1b`, non comanda il terminale. `build_model` controlla le forme su un modello
  costruito sul device `meta` (nessuna memoria) prima di costruire quello vero: una config enorme accanto a
  pesi piccoli è un `ArchitectureMismatch`, non gigabyte allocati.
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

#### Storia del training di qualunque architettura (Python: `gpt/recording.py`, `capture_attention`, `scripts/record_training.py`)

La storia del training (`trace.json` + snapshot, il formato che leggono `export_run.py --trace` e il
Laboratorio pesi) si registra per **qualunque** modello costruito dal codice di `gpt/`, anche modificato
(un'altra attivazione, più blocchi, RMSNorm, un'attention con un nome nuovo o "fusa", moduli nuovi): basta
l'interfaccia di `gpt.training.train`, `logits, loss = model(idx, targets)`. Le due cose che dipendevano dai
nomi del GPT del corso, i **gruppi** di parametri e l'**entropia dell'attention**, si adattano e il trace lo
dice in `notes`; tutto il resto (loss train/val e del batch, learning rate, norme dei gradienti, update ratio
e deviazione standard per gruppo, snapshot, testo generato, probabilità sul prompt) vale per ogni
architettura. Per il GPT del corso non cambia niente: trace, snapshot e `model.json` sono identici byte per
byte a prima.

| Modulo | Esporta | Uso |
|---|---|---|
| `gpt/recording.py` | `TrainingRecorder(tokenizer, val_data, block_size, RecorderConfig(...))` (`.save(dir)`, `.to_dict()`, `.notes()`), `RecorderConfig`, `snapshot_file`, `TRACE_FORMAT_VERSION`, `TRACE_FILE` | il callback di `train` che registra la storia |
| | `group_of(nome)`, `course_group(nome)` (None se il nome non è del corso), `module_group(nome)`, `parameter_groups(model)`, `grouping_note(nomi)` | gruppi di parametri, anche per nomi nuovi |
| | `shared_parameters(model)` → `{alias: originale}`, `sharing_note(shared, groups)` | pesi condivisi (weight tying, blocchi riusati) |
| | `attention_notes(sites, rows, problems)`, `snapshot_state(model)` | note sull'attention; i tensori di uno snapshot |
| `gpt/export.py` | `capture_attention(net, problems=None)`, `attention_sites(net)` → `[AttentionSite(path, layer, layer_path, module)]`, `attention_layer_path(path)`, `AttentionWeightsError`, `ATTENTION_METHOD` | probabilità di attention di ogni head, ovunque siano nel modello |
| `gpt/training.py` | `save_checkpoint(path, model, tokenizer)` | checkpoint scritto in modo atomico (sotto) |
| `scripts/record_training.py` | `parse_args(argv)`, `add_dataclass_options(group, cls, prefix)`, `overrides(args, prefix)`, `reference_differences(…)`, `same_layout_as_course(…)`, `webapp_problems(…)`, `output_problems(checkpoint, out)`, `main(argv)` | lo script (testabile in processo) |

**Gruppi** (`group_of`). I nomi del corso restano nei gruppi di sempre (`token_embedding`,
`position_embedding`, `blocks.<i>.attention` con head e proj, `blocks.<i>.ffwd`, `blocks.<i>.layernorm` con ln1 e
ln2, `ln_f`, `lm_head`). Un nome che il corso non ha va nel suo modulo con la regola di `module_group`, **la
stessa di `groupOf` in `shared/weights/groups.js`** (un test le confronta con Node; la verifica avversariale su
48.317 nomi): il sotto-modulo del blocco ripetuto oppure, fuori dai blocchi numerati, i primi due livelli del
nome; un parametro senza modulo è un gruppo da solo. L'indice del blocco è il primo livello numerato del nome
**se sta entro i primi tre livelli e ha ancora un modulo dopo di sé** (`_BLOCK_INDEX_MAX_LEVEL = 2`, in JS
`BLOCK_INDEX_MAX_LEVEL`): più in basso un indice numera le parti di un modulo (le head di `block.sa.heads.0`, gli
strati di `block.ffwd.net.0`). Esempi: `blocks.0.norm1.weight` → `blocks.0.norm1`, `transformer.h.3.attn.c_attn.bias`
→ `transformer.h.3.attn`, `model.layers.0.mlp.up_proj.weight` → `model.layers.0.mlp`, `transformer.wte.weight` →
`transformer.wte`, `logit_scale` → `logit_scale`; un blocco del corso riusato (`self.block`) dà 8 gruppi
`block.ln1`, `block.sa`, `block.ln2`, `block.ffwd`, …; `blocks.0.sa.c_attn.weight` → `blocks.0.attention`. Non fallisce
mai. Ordine di `groups`: quello di comparsa in `named_parameters` (per il corso l'ordine di prima, con
`blocks.<i>.layernorm` in fondo al suo blocco); la scelta finisce in `group_params` più una nota.
**Limiti della regola** (uguali in Python e in JS, perché guarda un nome alla volta): un contenitore di blocchi
più in basso del terzo livello (`model.decoder.layers.0…` di alcuni modelli di Hugging Face) dà un solo gruppo
grosso `model.decoder`; un'attention con le head messa direttamente nel modello, senza blocchi
(`sa_heads.heads.0.key.weight`, il modello intermedio della lezione), dà ancora un gruppo per head, perché lo 0 è
al terzo livello e dai nomi non si distingue da `transformer.h.0.ln_1`: i grafici del Laboratorio le chiamano
allora "blocco 0…3".

**Attention** (`capture_attention`). Un modulo è un'attention se la sua **classe** ha il **metodo**
`attention_weights` (come `Head` di `gpt/attention.py`), che riceve gli stessi argomenti di `forward` e
restituisce le probabilità dopo maschera e softmax, prima del dropout: `(B, T, T)` per una head oppure
`(B, H, T, T)` per H head calcolate insieme, con il batch come prima dimensione. I moduli si cercano in tutto il
modello (`named_modules`); il metodo si cerca sulla classe con `inspect.getattr_static` (un sotto-modulo, un
tensore o una property con quel nome non contano, e una property non viene eseguita); se un modulo con il
metodo ne contiene altri che lo hanno conta solo il più esterno. **Strato** = il blocco ripetuto che lo
contiene (`blocks.2.sa.heads.0` → `blocks.2`; fuori da blocchi numerati lo strato `""`); le head di uno strato si
numerano nell'ordine del forward; un modulo che lavora più volte nello stesso forward (un blocco riusato, pesi
condivisi come ALBERT) dà una riga per passaggio. Il risultato si controlla (tensore 3D o 4D con le ultime due
dimensioni uguali, finito, ≥ 0, righe che sommano a 1 ± 1e-2, prima dimensione = B: un `(B·H, T, T)` con head e
batch fusi si segnala suggerendo `weights.view(B, H, T, T)`), altrimenti `AttentionWeightsError`; con
`problems=[]` (lo usa il recorder) un modulo che fallisce si salta e il motivo finisce nella lista. Senza
probabilità (es. `F.scaled_dot_product_attention`) negli snapshot manca `attention_entropy` e il trace ha la
nota "attention: nessun modulo del modello espone il metodo attention_weights … le altre metriche ci sono tutte".

**Cambiamenti di `trace.json`** (`format_version` resta 1, tutto compatibile con `parseTrace`): `notes`, lista di
stringhe in italiano **solo se c'è qualcosa da dire** (mai per il GPT del corso); `snapshots[i].attention_entropy`
solo se il modello espone `attention_weights`, una riga per strato (anche di lunghezze diverse); `params`,
`total_params` e gli snapshot sono i tensori in virgola mobile dello `state_dict` nel suo ordine (anche i buffer
persistenti: gli stessi tensori che `export_run.py` esporta, così il Laboratorio abbina ogni snapshot ai pesi,
`layoutMatches`); `config` comprende i campi aggiunti dallo studente (es. `ffn_mult`); i numeri non finiti sono
`null` più una nota ("training divergente?"). `save` scrive ogni file in un temporaneo (`open(…, "xb")`) e poi
`os.replace`: un collegamento simbolico con quel nome viene sostituito, non seguito, e un'interruzione non lascia
file a metà. Anche `save_checkpoint` è atomico (cartella temporanea accanto, file con lo **stesso nome**, perché
`torch.save` lo scrive nell'archivio, poi `os.replace`; permessi del checkpoint di prima conservati; una cartella al
posto del file dà un errore in italiano).

**`scripts/record_training.py`**. Il modello è `gpt.model.GPTLanguageModel` del codice attuale (quello che
`export_run.py --benchmark` rimisura). Le opzioni del modello nascono dai campi di `GPTConfig` (`--n-layer`,
`--n-embd`, … e, da solo, un'opzione per ogni campo semplice aggiunto: `ffn_mult: int = 4` → `--ffn-mult 3`; bool
→ `--x/--no-x`), quelle del training dai campi di `TrainConfig` (`--max-iters`, `--eval-interval`, `--eval-iters`,
`--learning-rate`, `--seed`, …); poi `--log-every`, `--snapshot-steps 0,50,100`, `--threads N`, `--device`,
`--checkpoint`, `--out`, `--no-export`, `--update-webapp`. Errori delle opzioni in italiano, anche quelli di
argparse. **Protezione dei dati del corso**: prima del training lo script confronta con `webapp/data/` i tensori
registrati, i pesi condivisi, la config, il training e l'**impronta del codice dei moduli** (`code_sha256` contro
`model.code_sha256` di `builtin.json`: così riconosce anche GELU al posto di ReLU, a forme uguali); se qualcosa è
diverso e il comando scriverebbe in `webapp/data` o su `checkpoints/gpt.pt`, si ferma con l'elenco delle
differenze e le opzioni da aggiungere (`--out … --no-export --checkpoint …`; `--update-webapp` lo permette a chi
cambia davvero il GPT del corso). Controlla anche, prima del training, che `--out` e `--checkpoint` si possano
scrivere. Alla fine stampa le note e il comando `export_run.py … --trace … --benchmark` con i percorsi giusti.
Una classe fuori da `gpt/`: `TrainingRecorder` come callback di `gpt.training.train`, poi
`gpt.runs.export_run(weights_from_model(model, vocab=…), out, name=…, training_dir=…,
benchmark=gpt.benchmark.benchmark_model(model, tokenizer, …))`.

```bash
# il flusso per un'architettura modificata (README, "Un'architettura modificata, dall'idea al confronto")
python scripts/record_training.py --n-layer 6 --ffn-mult 3 --threads 1 \
    --checkpoint checkpoints/sei-blocchi/gpt.pt --out checkpoints/sei-blocchi/training --no-export
python scripts/export_run.py checkpoints/sei-blocchi/gpt.pt --name "6 blocchi, FFN 3C" \
    --trace checkpoints/sei-blocchi/training --benchmark
# nel Laboratorio: carica runs/6-blocchi-ffn-3c
```

Provato su copie del progetto con `gpt/` modificato come farebbe lo studente (200 step, poi `export_run.py --trace
--benchmark`, poi il caricamento con `runsFromFiles` e le funzioni dei grafici del Laboratorio): GELU, FFN 3·C con 6
blocchi, RMSNorm con un'attention fusa `(B, H, T, T)`, un'attention con `F.scaled_dot_product_attention`, weight
tying, multi-query attention in una `ModuleList` "layers", un blocco riusato 4 volte; in tutte i grafici pieni, ogni
tensore nel suo gruppo, gli snapshot abbinati ai pesi, nessun avviso. Limite: `attention_weights` si calcola una
seconda volta, ma solo negli snapshot. L'impronta del codice cambia anche per un commento nei quattro file dei
moduli: allora `record_training.py` senza `--out` chiede `--update-webapp` finché `builtin.json` non viene
rigenerato.

**Test.** `tests/test_recording_architectures.py` (gruppi e ordine del corso, `group_of` = `groupOf` di Node,
`capture_attention` sul corso e sulle architetture nuove, risultati sbagliati segnalati, blocco riusato, weight
tying, registrazione completa di un GPT modificato, NaN/∞ → `null`), `tests/test_recording.py` (`save` con
collegamenti simbolici), `tests/test_recording_cli.py` (opzioni generate, errori in italiano, protezione dei dati
del corso, uscite rifiutate prima del training), `tests/test_runs.py` (note del trace in `run.json`, testi non
fidati), `tests/test_training_and_weights.py` (checkpoint atomico). `tests/conftest.py` tiene PyTorch a 1 thread
durante i test (`GPT_TEST_THREADS=N` per cambiarli).

### Numeri dei tensori: tabella, statistiche, istogramma (`core/ui/tensorTable.js`, `core/ui/histogram.js`, `css/tensor-ui.css`)

Componenti DOM condivisi per **leggere i numeri veri di un tensore** nella sidebar o in un pannello: la vista
"Pesi numerici" del Transformer e il Laboratorio pesi li usano allo stesso modo. Niente three.js oltre a quello
che la pagina ha già (la tabella prende la colormap delle heatmap da `core/matrixNumbers.js`). Stili in
`css/tensor-ui.css` (classi `.tt-` e `.hist-`). Logica pura in `core/ui/tensorTableMath.js` e
`core/ui/histogramMath.js`; test in `tests/tensor-ui.test.mjs` (la logica pura e `TensorTable` su un DOM finto:
intestazioni dopo un cambio di tensore, virtualizzazione, tastiera, dispose).

| Modulo | Esporta | Uso |
|---|---|---|
| `core/ui/tensorTable.js` | `TensorTable`, `renderTensorStats` | tabella virtualizzata dei numeri; scheda delle statistiche |
| `core/ui/histogram.js` | `Histogram` | istogramma su canvas, una serie o più sovrapposte (A/B) |
| `core/ui/tensorTableMath.js` (puro) | `tableShape`, `axisWindow`, `scrollToShow`, `moveActive`, `GRID_KEYS`, `cellText`, `preciseText`, `scientificText`, `widestText`, `finiteAbsMax`, `headerText`, `describeCell`, `statsRows`, `MAX_SCROLL_PX`, `CELL_DECIMALS` | forma, finestra virtuale (anche compressa), tastiera, testi |
| `core/ui/histogramMath.js` (puro) | `histogramModel`, `binAt`, `binLabel`, `binRows`, `countHeight`, `logCountTicks`, `edgeDecimals`, `summaryText`, `countText`, `valuesText`, `seriesColor` | intervalli comuni, tacche, scala log, testi |

```js
import { TensorTable, renderTensorStats } from "../../core/ui/tensorTable.js";
import { Histogram } from "../../core/ui/histogram.js";
import { loadBuiltinRun } from "../../shared/weights/run.js";

const run = await loadBuiltinRun();
const t = run.tensor("blocks.0.ffwd.net.0.weight");          // (256, 64)
const table = new TensorTable(card, {
  caption: "Feed-forward, primo strato",                    // titolo (anche il nome accessibile della griglia)
  rowLabel: (i) => `n${i}`, colLabel: (j) => `${j}`,       // intestazioni (predefinite: gli indici)
  format: (v) => …,                                         // testo di una cella (predefinito: 4 decimali, "−0,1234")
  onHover: (cell) => …, onSelect: ({ i, j, value }) => …,   // sotto il puntatore / scelta: clic, frecce, Invio/Spazio
});
table.setTensor({ name: t.name, shape: t.shape, data: t.data, rowTitle: "neurone nascosto", colTitle: "dimensione d'ingresso" });
table.refresh();               // gli stessi dati cambiati sul posto (un altro step): si rimisura se serve
table.select(39, 3);           // cella attiva portata in vista ({ scroll: false } per non scorrere); table.selected
renderTensorStats(card, t.data, { label: "Statistiche" });  // una seconda chiamata sostituisce la scheda
const hist = new Histogram(card, { bins: 60, height: 120, label: `Valori di ${t.name}`, logY: false });
const step0 = await run.loadSnapshot(0);
hist.setData([{ label: "step 0", data: step0.tensors.get(t.name).data }, { label: "pesi finali", data: t.data }]);
// uscendo: table.dispose(); hist.dispose();
```

- **`TensorTable`**. `shape` `[righe, colonne]`; `[n]` = **una colonna** di n righe (un vettore si legge
  dall'alto in basso, come le barre dei bias nella scena); oltre 2D `reshape(-1, ultima)`; `data` è un
  riferimento, non una copia (almeno righe × colonne valori, altrimenti `RangeError`). Campi facoltativi di
  `setTensor`: `range` (|valore| del colore pieno; per gli step del training si passa la scala dei pesi finali,
  così i colori restano quelli della scena), `rowTitle` / `colTitle`, `caption`. **Ogni `setTensor` riscrive tutte
  le intestazioni**, anche quelle delle righe riciclate. Virtualizzata: solo le righe e le colonne visibili più un
  margine, riciclate mentre si scorre (render su `requestAnimationFrame` agli eventi `scroll`, mai per frame a
  riposo); oltre `MAX_SCROLL_PX` (8 milioni di px) l'area di scorrimento si comprime e ogni cella resta
  raggiungibile. Intestazioni ferme; sfondo dalla colormap divergente delle heatmap, testo nero o bianco per
  contrasto (≥ 4,5:1 su ogni colore), virgola italiana e segno "−", NaN e ±∞ scritti come tali; riga di stato e
  annuncio della cella con 6 decimali (`preciseText`, "3,2 · 10⁻⁷" per i minuscoli). Un solo punto di tabulazione
  (`role="grid"`, `aria-rowcount`/`aria-colcount`, `aria-activedescendant` con "carattere 39 ('a') · dimensione 3 ·
  −0,123457"); frecce, PagSu/PagGiù, Home/Fine (con Ctrl prima/ultima cella), Invio/Spazio = `onSelect`; le frecce
  usate dalla tabella non muovono né la pagina né la scena né il Percorso. Altezza: `--tt-max-height` (312 px).
- **`renderTensorStats(container, data, { label })`**: numero di valori, media, deviazione standard (di
  popolazione, come `gpt/recording.py`), minimo, massimo, media di |w|, % di |w| < 0,01, norma L2, da
  `tensorStats(data, { nearZero: 0.01 })`; NaN e ±∞ esclusi e dichiarati ("4 (2 non finiti esclusi)"); quattro
  cifre significative (`formatPrecise(v, 4)`).
- **`Histogram`**: intervalli comuni a tutte le serie (`[min, max]` dell'unione, `bins` uguali, l'ultimo comprende
  il massimo; NaN e ±∞ esclusi); una serie = barre, più serie = contorni a gradini e velatura al 16%, colori della
  palette dei grafici in ordine fisso; tacche all'italiana, linea dello zero; `logY` = conteggi in scala log(1 + n);
  legenda con due o più serie; canvas nitido (`devicePixelRatio` fino a 3). Accessibile: `role="img"` con il
  riassunto (`summaryText`), `Tab` sul grafico e frecce/Home/Fine/Esc per scorrere gli intervalli, annunciati da
  una regione `aria-live` ("[−0,0295; −0,0189)" e per ogni serie "667 valori · 4,1%").
- **Regole**: nomi di tensori e testi dei file caricati si passano così come sono (i componenti scrivono sempre
  con `textContent`); nessun lavoro per frame a riposo (ridisegnano solo a eventi); un contenitore nascosto non
  disegna, il `ResizeObserver` disegna quando diventa visibile; `dispose()` toglie listener, osservatori e DOM.
  Misure (rendering software, colonna di 356 px): `setTensor` di 65 × 64 in 8,6 ms la prima volta, render di una
  finestra mentre si scorre 0,2-1,1 ms, 112 celle nel DOM per una 256 × 64, 16 per un vettore di 65.536 valori.

## Pannello codice

Il codice mostrato deve essere **vero**: per il GPT viene da `data/model.json` (chiave `snippets`;
in JS `spec.snippets[chiave] → {file, start_line, code}`, mostrato con `tab.showCode(chiave, titolo, highlight)`);
per Spring AI/Java si usa il codice delle lezioni L16-L18 del corso (vedi incarico) e si dichiara
"esempio" quando è semplificato. Le righe Python che creano un peso (`buildModuleTree`, vedi "Pesi e run")
sono lette dagli snippet, non scritte a mano; per i modelli aperti di LLM moderni gli estratti veri di DeepInfer
hanno file e righe, il pseudocodice è marcato «NON è il codice ufficiale».

```js
codePanel.show({ title, file, startLine, code, language, highlight });   // se lo studente l'ha ridotto con "▁",
                                       // show() lo lascia ridotto (cambia solo l'intestazione)
codePanel.minimize(); codePanel.expand(); codePanel.minimized;   // da codice: non cambiano le scelte dello studente
codePanel.userMinimized; codePanel.userExpanded;   // l'ultima scelta dello studente con ▁ (show() la rispetta)
codePanel.minimizedForCharts;          // ridotto da followCharts (grafici aperti), non dallo studente
const off = codePanel.onToggle((minimized) => …);  // solo quando è lo studente a premere ▁
codePanel.toggleByUser();              // come il clic su ▁
codePanel.followCharts(charts);        // lo fa già createTabLayout (vedi "Layout condiviso") → funzione di rimozione
codePanel.coveredWidth();              // px coperti a destra (0 se ridotto, nascosto o sul telefono)
const stop = codePanel.onResize(fn);   // restituisce la funzione che smette di osservare
```
`aria-expanded` del pulsante ▁ segue la classe `minimized` anche quando un tab la cambia a mano (vecchia API,
`codePanel.root.classList.add("minimized")`). Con i grafici aperti `show()` lascia il pannello ridotto se lo
studente non l'ha espanso.

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
tab.views.register({ id, label, description, order, enter(tab), exit(tab), home?(tab), flow? }); tab.views.activate(id)
                   // flow: true = nella vista il flusso dei dati gira da solo sul binario (Architettura, Pesi):
                   // solo lì, e fuori dalle azioni, il pulsante "Flusso" è attivo
tab.views.home()   // "Centra" e "Vista d'insieme": l'inquadratura iniziale della vista attiva (home della vista)
tab.views.has(id); tab.contextView.reveal(stepId)   // apre un passo ctx.* del percorso della vista Contesto
tab.showCode(snippetKey, title, highlight[]); tab.focus(registryIdOrObject, options); tab.setCaption(text)
tab.getEngine() → Promise<GPTEngine> (js/engine/gpt.js, caricato una volta e condiviso)
tab.arch.tensorMeshes: Map(nome tensore PyTorch → [mesh della heatmap])
tab.lod = { controller, levels: {detail, blocks, macro}, register({detail, blocks, macro}) → rimuovi,
            setDistances({detail, blocks}), resetDistances(), setBaseOpacity, arch, current }  // lod.js
tab.neuronsView    // vista Rete neurale: arrangement, setArrangement(id), open(id), depth (vedi sotto)
tab.numbersView    // vista Pesi numerici: open({tensor, step, …}), loaded, source… (vedi sotto)
tab.reps           // l'API di installRepresentations (Rete neurale personalizzata, Cervello a livello)
tab.events: "view" (detail = id vista), "lod" (detail = id livello), "neurons-layout" ({arrangement, previous}),
            "numbers-source" (la sorgente dei pesi applicata nella vista Pesi numerici)
```

Moduli del tab (ognuno esporta `install…(tab)`):

| Cartella / file | Cosa aggiunge |
|---|---|
| `architecture.js`, `flow.js` | la pila 3D con i pesi veri e il pacchetto che scorre sul binario |
| `lod.js`, `views/archLevels.js`, `views/lodBlocks.js` | zoom semantico Dettaglio / Blocchi / Panoramica e l'azione "Zoom semantico" |
| `views/weights.js` | vista **Pesi**: dove stanno i 209.729 parametri, sovrapposta alla scena, con giro guidato |
| `views/numbers.js` (+ `numbersModel.js`, `numbersScene.js`, `numbersUi.js`, `numbersView.js`, `segmented.js`) | vista **Pesi numerici**: i float32 veri dentro le matrici della scena, ai pesi finali o a uno step del training (vedi sotto) |
| `views/neurons.js` (+ `neuronModel.js`, `neuronScene.js`; in profondità `neuronDepth.js`, `neuronDepthSpec.js`, `neuronDepthCamera.js`, `neuronDepthUi.js`) | vista **Rete neurale**: tutti i neuroni e le connessioni, forward animato, clic su un neurone per il suo calcolo; disposizione **A righe** o **Orizzontale con profondità** (vedi sotto) |
| `actions/` | azioni **Prompt** ("Dal prompt all'output") e **Training** (uno step al microscopio, training completo registrato) con lavagne e grafici |
| `context/` | vista **Contesto**: domanda → guardrail → embedding → ricerca su pgvector → prompt di sistema, chunk, tool → richiesta all'LLM, collegata al tab RAG |
| `brain.js` (soli dati) | il **Cervello** del tab: "Il GPT del corso e il cervello" (10 analogie, il microcircuito canonico della corteccia di Douglas e Martin, 15 regioni), generato dalla ricerca verificata |
| `content.js` | testi del Percorso e fonti |

Id utili nel registro: `tokens`, `tok_emb`, `pos_emb`, `embed_add`, `blocks`, `blocks.<i>`,
`blocks.<i>.ln1|heads|heads.<h>|heads.<h>.pattern|proj|add1|ln2|ffwd|add2`, `ln_f`, `lm_head`,
`logits`, `next_token`, `prediction`, `overview`. La scena è una pila: riga 0 input, righe 1…n_layer
i blocchi, ultima riga l'uscita (`ROW_SPACING = 15`), centrata con `root.position.y`.

Viste registrate: `architettura` (base; tornandoci la camera, il codice e la didascalia tornano allo
step scelto nel Percorso). I moduli aggiungono `pesi`, `numeri`, `neuroni`, `contesto`. Scegliere a mano un
livello dello zoom inquadra quel livello (vedi `installLevelPicking` in `index.js`).

Link diretti e ricerca (`search.js`, `goto`): il `goto` comincia con `reps.goto(target)` (`view:rete`,
`view:cervello`; per tutto il resto un Cervello aperto si chiude), poi `tensor:<nome>[@<step>]`, poi:
`step:<id di content.js>` (torna all'Architettura, chiude l'azione, zoom su Auto),
`action:prompt|training-step|training-run|lod-zoom|weights-tour|neurons-forward|context-pipeline`,
`view:architettura|pesi|numeri|numeri.training|neuroni|neuroni-profondita|contesto|rete|cervello`,
`view:zoom.detail|zoom.blocks|zoom.macro` (Architettura, livello bloccato, pannello Viste aperto) e
`view:ctx.<passo>` (un passo del percorso della vista Contesto, con lo zoom su Auto; `tests/search.test.mjs`
controlla che esista in `context/texts.js`).

| Link | Effetto |
|---|---|
| `view:numeri` | vista Pesi numerici (sorgente e tensore di prima) |
| `view:numeri.training` | Pesi numerici sul training, allo step 0, con "▶ Guarda i pesi nascere" pronto |
| `tensor:<nome>` | Pesi numerici su quel tensore, ai pesi finali (tabella aperta, volo sulla matrice) |
| `tensor:<nome>@<step>` | idem allo step: uno step di snapshot è un dato vero, uno intermedio una stima lineare dichiarata; fuori da 0…5000 o nome sconosciuto → `false` (il guscio avvisa). Lo stesso formato del Laboratorio pesi: un link a un tensore vale uguale nei due tab |
| `view:neuroni` · `view:neuroni-profondita` | la vista Rete neurale **a righe** (come sempre) · **orizzontale con profondità** (`NEURON_LINKS` di `views/neurons.js`) |
| `view:rete` · `view:cervello` | rappresentazioni del nucleo: la Rete neurale (la vista `neuroni`, nella disposizione scelta dallo studente) e il Cervello |

`tensor` non va nella ricerca (che indicizza passi, azioni e viste): ci vanno le viste `numeri`,
`numeri.training`, `neuroni-profondita`, `rete` e `cervello`.

### Vista "Pesi numerici" (`views/numbers*.js`)

I **numeri veri** dentro le matrici del GPT del corso: sulla scena dell'architettura, avvicinandosi a una
matrice, i colori delle heatmap diventano i float32 che usa il motore (`attachNumbers` di
`core/matrixNumbers.js`, un solo `NumbersGroup({ maxOverlays: 4 })`, `controls.minDistance` = 0,2 solo qui). La
sorgente si sceglie: i **pesi finali** oppure uno **step del training** (gli 11 snapshot veri; la stima lineare
tra due snapshot solo se lo studente accende "Interpola", dichiarata come tale); le heatmap si riscrivono con i
pesi dello step (`SceneTextures.writeWeights`, istanza propria) e i numeri seguono: si vedono i pesi nascere dal
rumore. La vista **mostra** soltanto: il motore e le altre viste restano sui pesi finali e, uscendo, le texture
tornano byte per byte quelle di prima.

| File | Esporta | Ruolo |
|---|---|---|
| `numbersModel.js` (puro) | `trainingSource`, `snapToSnapshot`, `sameSource`, `FINAL_SOURCE`, `sourceLabel`, `float16RoundingCheck`, `tensorAxes`, `tensorKind`, `cellMeaning`, `tensorGroups`, `shapeText`, `headCard`, `cardDepths`, `CARD_KINDS`, `parseTensorTarget`, `formatTensorTarget`, `labTarget`, `birthSchedule`, `shownChar`, `NUMBERS_VIEW_ID`, `BIRTH_VIEW_ID` | sorgente dei pesi, etichette oneste, significato di righe e colonne, elenco come il codice Python, link, calendario della "nascita" |
| `numbersScene.js` (three.js) | `NumbersScene`, `NUMBERS_MIN_DISTANCE` | numeri sulle heatmap, riscrittura delle heatmap, carta della head davanti, voli su tensore e cella, cella sotto il puntatore, attivazioni dei pesi finali attenuate negli step |
| `numbersUi.js` (solo DOM) | `sourceSwitch`, `stepControl`, `stepChips`, `playButton`, `sourceNote`, `tensorList`, `numbersBar`, `SOURCE_OPTIONS` | interruttore della sorgente, cursore degli step con le tacche, pulsanti degli snapshot, etichetta, elenco dei tensori, barra dello stage |
| `numbers.js` (nessun import) | `installNumbersView(tab)`, `NUMBERS_VIEW_ID`, `BIRTH_VIEW_ID` | registrazione **leggera**: la vista compare subito (pannello Viste, link, ricerca), i ~280 KB di sorgente che porta con sé (three.js dei numeri, tabella, istogramma, run, albero dei moduli) si caricano alla prima apertura; fino ad allora `tab.numbersView = { loaded: false, open }` |
| `numbersView.js` | `createNumbersView(tab)` → `{ enter, exit, home, api }` | la vista: stato, dati (run "GPT del corso"), scheda del tensore, "Apri nel Laboratorio pesi", animazione |
| `segmented.js` (solo DOM) | `segmentedSwitch` | radiogroup del tab (sorgente, disposizione della Rete neurale) |

- **Etichette oneste** (`sourceLabel` → `title`, `short`, `detail`, `tag`): "Pesi finali (float32)"; "Step 1000
  (float16)" (le cifre oltre la terza vengono dall'arrotondamento a 16 bit), "pesi iniziali" allo step 0; "Step
  5000 (float16: i pesi finali arrotondati)" **solo se la verifica sui file lo conferma** (`float16RoundingCheck`,
  nel browser al primo caricamento: vero per tutti i 209.729 valori di `snap_05000.f16.bin`); "≈ 750 stima" in
  giallo per una stima. Il cartellino accanto al cursore nella barra dello stage (`.tf-bar-step`) lo dice anche a
  sidebar chiusa. Senza "Interpola" il cursore si **aggancia** allo snapshot più vicino; i pulsanti degli 11
  snapshot scelgono lo step esatto. La scala dei colori resta quella dei pesi finali anche negli step: si vede
  quanto crescono.
- **"▶ Guarda i pesi nascere"** (`birthSchedule`): un tratto di 1,1 s per coppia di snapshot vicini (non sono
  equidistanti), tempo reale, riscritture al più 12 al secondo, statistiche e istogramma al più 4; con
  `prefers-reduced-motion` uno snapshot vero alla volta.
- **Attivazioni dei pesi finali attenuate negli step** (`dimFinalActivations`): i pattern di attention, le barre
  delle probabilità e il prossimo token sono calcolati con i pesi finali; accanto ai pesi di uno step scendono al
  20% dell'opacità (`setBaseOpacity`) e il pannello porta all'azione «Il training completo», dove nascono insieme
  ai pesi. Anche il flusso dei dati si ferma (moltiplicherebbe i colori); uscendo tutto torna com'era.
- **Puntatore**: al più un raycast per frame sui 94 piani (`host.pick(event, meshes, false)`), solo al livello
  Dettaglio, ripetuto anche quando la camera si muove sotto il mouse fermo; etichetta con tensore, "riga 39 =
  carattere 'a' · colonna 3 = dimensione dell'embedding 3" e valore; clic o tocco = la tabella su quella cella
  (sul telefono l'etichetta resta 2,6 s).
- **Sidebar** (pannello Viste): sorgente, elenco dei tensori **come il codice Python** (`tensorGroups`:
  Embedding, `blocks[i] · Block` con `ln1`, `sa.heads[h].key/query/value`, `sa.proj`, `ln2`, `ffwd.net[0]`,
  `ffwd.net[2]`, poi `ln_f` e `lm_head`), scheda del tensore con la **riga Python che lo crea** (`findNode` di
  `buildModuleTree`, evidenziata nel pannello del codice), che cosa sono righe e colonne (`tensorAxes`), la
  `TensorTable`, "Mostra nella scena", "Apri nel Laboratorio pesi", `renderTensorStats` e l'`Histogram` (in uno
  step: lo step sovrapposto ai pesi finali). "Apri nel Laboratorio pesi" → `app.goto("lab", { kind: "tensor", id:
  "<nome>[@<step>]" })`.
- **Barra dello stage** (`.tf-numbers-bar`, seconda fila a sinistra): sorgente, cursore, cartellino e "▶ Nascita";
  si ferma prima del codice e dei grafici, va a capo se serve; i chip della navigazione stanno sotto la sua altezza
  vera (`--tf-numbers-h`, da un `ResizeObserver` attivo solo nella vista).

```js
await tab.numbersView.open({ tensor: "lm_head.weight", step: 1000 });   // step: null = pesi finali (carica il modulo se serve)
await tab.numbersView.open({ tensor: "blocks.0.ln1.weight", step: 750, interpolate: true, cell: { row: 3, col: 0 } });
await tab.numbersView.open({ birth: true });                             // il training dallo step 0
tab.numbersView.setStep(500); tab.numbersView.setMode("final" | "training");
tab.numbersView.setInterpolate(true); tab.numbersView.togglePlay(); tab.numbersView.selectTensor(name, { fly, cell });
tab.numbersView.source;   // {kind: "final"} | {kind: "snapshot", step} | {kind: "interpolated", step, from, t}
```
Costi (rendering software): cambio di step con lo snapshot in memoria ~17 ms (di cui ~3 ms la riscrittura delle
94 heatmap), step stimato ~8 ms, `update` dei numeri a riposo sotto la risoluzione del timer; dopo 5-10 cicli di
entrata e uscita geometrie, texture, listener e callback per frame invariati.

### Rete neurale: "A righe" / "Orizzontale con profondità" (`views/neurons.js`, `views/neuronDepth*.js`)

La vista **Rete neurale** ha due disposizioni della stessa rete, scelte con un interruttore (radiogroup nel
riquadro della vista e nella barra dello stage `.tf-neuron-bar`): **A righe** (predefinita, invariata: una riga
per blocco, un token alla volta) e **Orizzontale con profondità**: la rete VERA e COMPLETA del GPT del corso da
sinistra a destra, con tutti i token del prompt in profondità, disegnata dal renderer del nucleo
(`NetworkScene`, vedi "Uso dentro la scena di un tab") montato sulla scena del tab. Prompt, posizione, scheda del
neurone e grafici sono condivisi: un clic su un neurone apre lo stesso dettaglio (formula con i numeri veri)
nelle due disposizioni.

- **x** = sezioni nell'ordine del flusso: one-hot del token (65) e della posizione (32) → embedding (due lastre) →
  x; per OGNI blocco LN 1 → self-attention (key, value, query, uscita delle head) → proiezione → + → LN 2 →
  nascosto 256 (ReLU) → uscita FFN → +; poi LN finale → logits (65) → softmax (65). I 4 blocchi si disegnano tutti.
  **y** = i neuroni della sezione. **z** = le posizioni del prompt (fino a 32) con le attivazioni vere di tutte le
  posizioni (`engine.forward(ids, { trace: true })`). Niente simulazioni: valori, pesi di attention e pesi delle
  `nn.Linear` vengono dal motore (badge "Attivazioni vere").
- **La self-attention è una sezione con quattro lastre** (`ATTENTION_SLABS`: key, value, query, uscita): il
  renderer disegna gli archi di attention dalla prima lastra (chi dà: key e value del token j) all'ultima (chi
  riceve: l'uscita del token i), così ogni arco j → i è `att[h][i][j]` del motore, causale. Sezioni:
  `tok_onehot`, `pos_onehot`, `emb`, `x0`, per ogni blocco `b<l>.ln1`, `b<l>.attn`, `b<l>.proj`, `b<l>.resid1`,
  `b<l>.ln2`, `b<l>.hidden`, `b<l>.ffnOut`, `b<l>.resid2`, poi `ln_f`, `logits`, `probs` (39); 4.067 neuroni per
  token (i 4.387 della disposizione a righe senza le 320 copie del vettore x), 130.144 unità con 32 token.
- **Collegamenti**: `lookup` (carattere e posizione scelgono la loro riga), `identity` (somme, LayerNorm token per
  token, softmax: collegamenti fissi), `residual`, `attention` (`causal: true`, `heads: 4`, pesi veri), `explicit`
  sulla fetta scelta per ogni `nn.Linear` (i pesi più forti per |w|, quota del cursore "Pesi della fetta":
  `DEPTH_FRACTIONS` 0,5%…25%, 2% predefinito) e per i γ delle LayerNorm. La quota si cambia **senza ricostruire**:
  la rete nasce con la quota chiesta; alla prima quota più alta si ricostruisce una volta con la quota massima
  (25%) e da lì `DepthLayout.applyFraction(f)` chiama `setSliceDensity` con `sliceCounts(spec, f)` (quanti per
  `nn.Linear`, i γ tutti) e aggiorna la nota (`depthNote(f)`); la regola è `depthBuildPlan` (pura, testata). Un
  prompt nuovo invece ricostruisce, tornando alla quota chiesta (gli archi di attention e i lookup dipendono dal
  prompt): solo allora la vista scrive «Ricostruisco la rete…» (`DepthLayout.needsRebuild`).
- **Interazione**: passando sopra un neurone, sezione, lastra, head, indice, token e valore vero
  (`unitDescription`, scritto come testo); clic = `depthUnitNeuron` porta l'unità al neurone della disposizione
  a righe e apre **la stessa scheda** di sempre, con in più la riga "Nella rete orizzontale: sezione · lastra ·
  neurone · head · token"; un anello di dimensione fissa segue il neurone (`neuronDepthUnit`). Fetta: i pulsanti dei
  token o un clic sul nome di un token. **▶ Anima**, **Guarda lo spessore**, **Di lato** nella barra dello stage.
  "Centra" = tutta la rete di lato; si gira fin quasi di fianco ma mai da dietro (`setOrbitLimits({ turn: 85,
  tilt: 82, minDistance: 0.5 })`); confini, limiti d'orbita e `minDistance` del tab si rimettono uscendo
  (`nav.setBounds` restituisce la sorgente di prima). La camera di ciascuna disposizione è ricordata.
- In profondità il selettore dello zoom e il pannello del codice si nascondono (la rete è lunga e bassa e le serve
  tutta la larghezza); tornando a righe tornano tali e quali. `neuronDepth.js` passa a `scene.update` i rettangoli
  coperti (barra, didascalia, grafici, pad) e il punto guardato.

| File | Esporta | Ruolo |
|---|---|---|
| `js/shared/gpt/depthSpec.js` (puro, condiviso con il Laboratorio) | `buildDepthSpec({config, vocab, trace, param, fraction, orders, values})`, `depthSections(config)`, `depthCounts(config, T)`, `depthSectionValues(config, trace)`, `attentionWeights(trace, l)`, `linearTensors(config)`, `weightOrders(config, param)`, `sliceCounts(spec, fraction)`, `depthNote(fraction)`, `unitDescription`, `sectionTitle`, `shownValue`, `ATTENTION_SLABS`, `DEFAULT_DEPTH_FRACTION`, `shownChar` | dalla configurazione e dalla traccia del motore alla `networkSpec` |
| `neuronDepthSpec.js` (puro) | riesporta tutto `js/shared/gpt/depthSpec.js` (gli import di prima valgono) più le parti della vista: `depthBuildPlan(…)`, `depthUnitNeuron`, `neuronDepthUnit`, `DEPTH_FRACTIONS`, `DEPTH_VIEW_ID` | quote del cursore e ricostruzione; mappatura unità ↔ neurone della disposizione a righe |
| `neuronDepthCamera.js` (puro) | `framePose(points, direction, {fov, aspect, top, bottom, margin, minDistance})`, `cameraBasis`, `DEPTH_DIRECTIONS` | inquadrature nella fascia libera tra i comandi in alto e la didascalia |
| `neuronDepth.js` | `DepthLayout` | three.js: la `NetworkScene` sulla scena del tab, mouse e tocco, fetta, onda, camera, navigazione, dispose |
| `neuronDepthUi.js` (solo DOM) | `ARRANGEMENTS`, `arrangementSwitch`, `stageBar`, `depthIntro`, `depthControls`, `depthWhere`, `depthLegend` | interruttore, barra dello stage, introduzione, comandi, riga della scheda, legenda |
| `neurons.js` | `installNeuronsView(tab)`, `NEURON_LINKS` | la vista: disposizione, riquadro condiviso, scheda del neurone, azione "passaggio in avanti" (anima la disposizione a righe) |

```js
tab.neuronsView.arrangement;                   // "rows" | "depth"
await tab.neuronsView.setArrangement("depth"); // fuori dalla vista la ricorda soltanto (entrando la applica)
await tab.neuronsView.open("depth");           // apre la vista Rete neurale in quella disposizione
tab.events.addEventListener("neurons-layout", (e) => e.detail);   // { arrangement, previous }
tab.neuronsView.depth;                         // DepthLayout (stats, scene, model): console e script di verifica
```
Fedeltà verificata con PyTorch (due prompt, tutte le unità: scarto massimo 4,8·10⁻⁶; archi di attention, residui, pesi
della fetta e schede del clic uguali a `gpt/`); a righe nessuna regressione (stato e scena identici pixel per pixel).
Costi (rendering software, 1600×900): costruzione 130-180 ms ("Preparo la rete…" disegnato prima), frame 48-56 ms di
lato e di tre quarti (a righe 18 ms), a riposo 1,2 µs (solo il confronto della camera), cambio di fetta 1,7-2,8 ms,
cambio di quota 2-3 ms; uscendo geometrie e texture tornano esattamente quelle di prima.

### Rappresentazioni e cervello del Transformer

`installReps(tab)` in `index.js`, dopo i pannelli del tab (anche Contesto) e **prima** di "Fonti":
- **Rete neurale** = modalità personalizzata `{ enter, exit }`: `enter` → `tab.views.activate("neuroni")` con la
  disposizione scelta dallo studente ("A righe" se non ne ha scelta una); `exit` → la vista di prima (o
  l'Architettura). **Cervello** = livello del nucleo con `brain.js`.
- `reps.sync` a ogni evento `view` di `tab.events` (pannello Viste, passi, azioni, link: tutti passano da
  `views.activate`) e nel `transformer:view` del bus prima di attivare la vista; durante `exit` della Rete neurale
  l'evento che segue non risincronizza.
- `brain.js` è generato dalla ricerca verificata (`brain_data.json → tabs.transformer`) senza riscrivere i testi:
  intro, 10 analogie (neurone e ReLU, plasticità, attention, normalizzazione, predizione, gerarchia, cervelletto,
  ippocampo, …), il microcircuito canonico della corteccia di Douglas e Martin (7 gruppi, 15 sinapsi, il componente
  del GPT per ogni gruppo), 15 regioni, 2 percorsi (arcuato, via perforante), il pulvinar, 7 numeri e 53 fonti;
  `normalizeBrainSpec(brainSpec, { strict: true })` senza avvisi.
- **Pulsante "Flusso"** (`installFlowButton`): `DataFlow.playing` è una proprietà con `onChange`, quindi l'etichetta
  segue sempre lo stato vero; attivo solo nelle viste con `flow: true` e fuori dalle azioni, con un titolo che spiega
  perché è disattivato.

## Il tab Laboratorio pesi (`tabs/lab/`, `css/tabs/lab*.css`)

I pesi del GPT in Python **organizzati come il codice** (`GPTLanguageModel` → `blocks` → `Block` →
`sa.heads[h].key/query/value`, `proj`, `ffwd.net`, `ln1/ln2`, `ln_f`, `lm_head`), come celle colorate e, da vicino,
come numeri veri; i run caricati dallo studente (anche con un'architettura modificata); le animazioni dei calcoli
con i pesi e le attivazioni veri (`actions/`); confronto A/B, grafici del training e benchmark (`analysis/`).
Registrato in `js/app.js` subito dopo `transformer` (id `lab`, link `#lab`). Il flusso che rende semplice:
modifico il codice o il training → riaddestro registrando la storia (`scripts/record_training.py`) → esporto il
run (`scripts/export_run.py --trace --benchmark`) → lo carico qui (diventa **A**, e **B** diventa da solo il GPT
del corso) → vedo i pesi, confronto prima e dopo, guardo grafici, note e benchmark.

| File | Esporta | Ruolo |
|---|---|---|
| `index.js` | `createTab(root, app)` | raccordo corto: layout, stato, pannelli, scena, azioni, analisi, rappresentazioni, Fonti |
| `tabObject.js` | `createTabObject({app, ui, spec, state})` | l'oggetto `tab` (sotto) |
| `state.js` (puro) | `LabState`, `COURSE_RUN_ID`, `MODES`, `EVENTS`, `SNAPSHOT_BUDGET`, `incompatibility(run)` | registro dei run, run A/B, step, scelta, modalità, pesi a uno step, motore JS (cache e memoria) |
| `steps.js` (puro) | `sliderModel(steps, {interpolate})`, `snapStep`, `stepLabel`, `shortStepLabel`, `INTERPOLATION_STEP` | il cursore "Step del training" |
| `links.js` (puro) | `resolveTarget({kind, id})`, `parseTensorTarget`, `formatTensorTarget`, `parseRunTarget`, `VIEW_IDS`, `RUN_ALIASES` | i link diretti (testo non fidato) |
| `goto.js` | `gotoTarget(tab, target)` | `instance.goto` |
| `scene/layout.js` (puro) | `layoutModel(tree, {config, snippets, classes})`, `macroItems`, `moduleItems`, `hitTest`, `subtractRects`, `coveredBy`, `classesFromRunInfo`, `codeLine`, `CELL`, `VECTOR_STRETCH`, `Z_STEP` | dall'albero dei moduli ai rettangoli della scena |
| `scene/roles.js`, `scene/pack.js`, `scene/labels.js`, `scene/frames.js` | `LAB_ROLES`, `roleOf`; `packShelves`; `LabelAtlas`, `labelMesh`; `buildFrames`, `buildArrows` | ruoli e colori (quelli di `core/neuralNet/roles.js`), atlante delle etichette, contenitori in due oggetti per livello |
| `scene/labScene.js`, `scene/index.js` | `LabScene`, `SCENE_LIMITS`; `installScene(tab)`, `LOD_DISTANCES` | la scena three.js (API sotto); scena + stato, LOD, mouse e tocco, navigazione |
| `inspect/meaning.js` (puro), `inspect/outline.js`, `inspect/modelPane.js`, `inspect/pane.js` | `tensorMeaning(name, shape, {vocab, config})`, `shownChar`; `createOutline`, `outlineRows`, `moduleArgs`; `createModelPane`; `createInspector` | che cosa sono righe, colonne e celle; l'indice dei moduli come `print(model)`; pannelli "Modello" e "Tensore" |
| `runs/describe.js` (puro) | `configDiff`, `architectureDiff`, `compatibilityText`, `trainingText`, `originText`, `formatDate`, `formatBytes`, `traceNotes(trace)`, `runNotes(run, {limit})`, `TRAINING_NOTE_PREFIX`, `isolate(text)` | come si racconta un run, comprese le note della registrazione (sotto) |
| `runs/pane.js`, `runs/card.js`, `runs/upload.js`, `runs/timeline.js` | `createRunsPane`, `runCard`, `filePicker`/`droppedFiles`/`dropZone`, `createTimeline`/`stageChip` | pannello "Run", scheda, caricamento, step del training e targhetta dello stage |
| `network.js` | `trueSpec`, `structureSpec`, `networkSpecFor(state)`, `createNetworkSource(tab)`, `courseDims`, `encodeWith`, `strongest`, `NETWORK_PROMPT` | rappresentazione "Rete neurale" del run A |
| `brain.js` (dati) | `brainSpec` | "Cervello": 10 analogie (pesi e sinapsi, LTP e regola di Hebb, cervelletto e segnale d'errore, …), il circuito del cervelletto (8 gruppi, 11 sinapsi), 22 fatti con fonte, dalla ricerca verificata |
| `content.js`, `search.js` (dati) | `INTRO`, `FLOW`, `EXPORT_STEPS`, `ACCEPTED_FILES`, `HONESTY`, `SOURCES`, …; `searchIndex` | testi e voci della ricerca globale (più quelle di `actions/search.js` e `analysis/search.js`) |
| `actions/index.js`, `analysis/index.js` | `installLabActions(tab)`, `installLabAnalysis(tab)` | le azioni animate e il confronto, i grafici, il benchmark (sotto) |

Stili: `css/tabs/lab.css` (`.lab-`), `lab-actions.css` (`.lab-act-`), `lab-analysis.css` (`.lab-an-`), caricati da
`index.html` in quest'ordine. Test: `tests/lab.test.mjs` (logica pura), `tests/lab-core-scene.test.mjs` (scena con
three.js e un canvas finto), `tests/lab-actions.test.mjs`, `tests/lab-analysis.test.mjs`, `tests/lab-notes.test.mjs`.

**Ordine di costruzione** (`index.js`): `createTabLayout`, `data/model.json` (gli snippet del codice vero),
`LabState`; pannelli **Modello**, **Tensore**, **Run**; `installScene(tab)`, barra dello stage (targhetta del run e
dello step), rilascio dei file sulla scena; `await state.init()` (il GPT del corso come run A, pesi finali);
`installLabActions(tab)` e `installLabAnalysis(tab)` (lo stato è già inizializzato: leggono `state.current` e
ascoltano gli eventi; i loro pannelli vanno dopo "Run" e prima di "Cervello"); `installRepresentations` (Rete
neurale, Cervello) e il pannello **Fonti** (`tab.sources`). Se un `install…` restituisce `{ dispose() }`, finisce in
`tab.disposers`. Lo store dei run si crea con `createStore()`: dove il browser vieta IndexedDB il tab si apre lo
stesso e la casella "Ricordali" è spenta e lo dice.

### L'oggetto `tab` e lo stato

```js
tab.app, tab.ui                  // ui = createTabLayout: { root, sidebar, stage, caption, host, director, nav, codePanel, charts, player }
tab.spec                         // data/model.json (spec.snippets[chiave] → {file, start_line, code}: il codice vero)
tab.state                        // LabState (sotto)
tab.scene                        // installScene: { scene (LabScene), lod, home(), focusTensor(name), focusModule(path),
                                 //   focusCell(name, i, j), pickAt(x, y), hideTip(), rebuild(run) }
tab.panes                        // { modello, tensore, run }: elementi dei pannelli della sidebar
tab.model, tab.inspector, tab.runs   // pannelli: modello { timeline, outline }; inspector { show(path), refresh(),
                                 //   selectCell(i, j), table (TensorTable), histogram }; runs { load(files),
                                 //   enableCompare(), openUpload(), openGuide() }
tab.engineFor(runId, step)       // = state.engineFor;  tab.weightsAt(runId, step, o) = state.weightsAt
tab.treeOf(run), tab.layoutOf(run)   // buildModuleTree(…) e layoutModel(…) in cache (la disposizione della scena)
tab.inspect(path, { cell, focus, code, source })   // sceglie tensore o modulo: riquadro, codice, pannello Tensore
tab.showCell(name, i, j, { fly }); tab.showCode(path, { expand }); tab.setCaption(text); tab.openPane(id, { focus })
tab.views.register(id, { title, open })            // viste per #lab/go=view:<id> (es. "confronto")
tab.links.register(kind, handler)                  // tipi nuovi #lab/go=<kind>:<id> → handler(id) → Promise<boolean>
tab.sources                                        // gruppi del pannello Fonti ({ title, items: [{title, url, note}] })
tab.reps, tab.network, tab.analysis, tab.stats     // dopo l'installazione; window.__labTab per la console
```

```js
const state = new LabState({ loadBuiltin: () => loadBuiltinRun(), store: new RunStore() });   // store: null = niente IndexedDB
await state.init();
state.entries;              // [{ id, name, description, origin: "builtin"|"upload"|"saved", loaded, saved, savedBytes, summary, error }]
state.run(id);              // WeightRun in memoria (o null se è solo salvato: await state.loadRun(id))
state.runA, state.step;     // il run nella scena e il suo step (null = pesi finali);  state.runB, state.stepB: il confronto
state.selected, state.tensor, state.kindOf(path);   // scelta: un tensore o un modulo (percorso PyTorch)
state.mode, state.modeLabel;  // "pesi" | "differenza" | "valori" (+ etichetta, es. "A − B")
state.interpolate; state.features;   // interpolazione tra snapshot (dichiarata); { compare } installato dall'analisi
state.current;              // { runId, step, view } = i pesi di A annunciati con "weights"
await state.setRunA(id, { step });  await state.setStep(step);       // step agganciato agli snapshot veri
await state.setRunB(id | null, { step });  state.setStepB(step);
state.select(path | null);  state.setMode("differenza", { label: "A − B" });  await state.setInterpolate(on);
state.enableFeature("compare");     // i pulsanti "B" compaiono accanto a ogni run nel pannello Run
state.addRuns(runs);  await state.saveRun(id);  await state.forget(id);  await state.storageUsage();
const v = await state.weightsAt(id, step);   // { runId, step, requested, exact, from, t, source: "final"|"snapshot"|"interpolated",
                                             //   tensors: Map(nome → {name, shape, numel, dtype, data}), dtype, label }
const { engine, reason, view } = await state.engineFor(id, step);   // GPTEngine con quei pesi, o null + motivo in italiano
state.release(id);  state.releaseSnapshots();  state.cacheInfo;     // libera viste e motori di un run; run grandi (sotto)
state.on(type, fn) → off                      // fn(detail, state)
```

| Evento | `detail` | Quando |
|---|---|---|
| `runs` | `{ reason: "init" \| "add" \| "load" \| "save" \| "forget" \| "saved", … }` | il registro cambia |
| `busy` | `{ what: "run" \| "weights", id, busy }` | un caricamento comincia o finisce |
| `runA` | `{ id, previous, run }` | il run A è cambiato (già caricato: la scena si ricostruisce qui) |
| `step` | `{ step, previous, runId }` | lo step di A è cambiato (i pesi arrivano con `weights`) |
| `weights` | `{ runId, step, view }` | i pesi di A da mostrare sono pronti (scena, tabella, statistiche si aggiornano) |
| `runB` / `stepB` | come `runA` / `step` | il confronto |
| `select` | `{ path, previous, kind: "tensor" \| "module" \| null }` | la scelta |
| `mode` | `{ mode, previous, label }` | la modalità della scena |
| `interpolate`, `features`, `error` | `{ interpolate }`, `{ name, enabled }`, `{ message, cause }` | |

Regole: ordine garantito `runA` → `step` → `weights`; una richiesta superata da una più recente non annuncia pesi
vecchi (gettoni); un errore di caricamento lascia lo stato com'era e manda `error`. Cache: viste dei pesi (al più 24,
interpolazioni entro 96 MB; quelle di A e B non si buttano), al più 3 motori; il run lasciato si libera
(`release`). **Run grandi**: se gli snapshot decodificati di un run superano `SNAPSHOT_BUDGET` (64 MB),
`releaseSnapshots()` (a ogni `weights`, `runB`, `stepB` e fine azione) toglie dalla memoria quelli degli step che non
servono (tutti tranne quelli di A e B, con l'interpolazione i due attorno) con `run.unloadSnapshots`; il GPT del corso
(9 MB) resta in cache. `engineFor` per lo stesso (run, step) restituisce lo stesso motore: non chiamate
`setWeights` su di esso (create il vostro con `run.createEngine()`).

### La scena (`tab.scene.scene`, `LabScene`)

Tre livelli del LOD ("Auto · Dettaglio · Moduli · Panoramica", soglie `LOD_DISTANCES` = 48 e 135 unità, scalate
sugli stage stretti): **Dettaglio** = contenitori annidati con la riga Python che li crea e una `NumericMatrix` per
parametro (un solo `NumbersGroup`: i numeri compaiono da soli da vicino); **Moduli** = un riquadro per modulo con i
parametri; **Panoramica** = le isole del modello con le quote e il flusso.

```js
const S = tab.scene.scene;
S.matrixOf(name);           // NumericMatrix del tensore (o del buffer tril: blocks.0.sa.heads.0.tril)
S.layoutOf(name);           // { rows, cols, cellW, cellH, rect: {x0, x1, y0, y1, z}, center, cellCenter(i, j) }: dove animare
S.boxOf(path); S.nodeOf(path);   // Box3 di un modulo o di un tensore (per director.focus); nodo della disposizione
S.setWeights(map);          // name → Float32Array | {data}: i pesi di un altro step (scala dei colori invariata)
S.showValues(map | null, { colormap: "divergente" | "pesi", label });   // valori qualsiasi sulle stesse matrici
                            //   (es. A − B), → { shown, missing, label }; null = tutto torna ai pesi
S.highlightCells(name, cells | null, { dim = 0.62 });   // vela il resto della matrice: [{i, j}] | {rows} | {cols} |
                            //   {r0, r1, c0, c1}; false se la matrice è troppo grande
S.hoverCell(name, {row, col} | null); S.setSelected(path);
S.customValues, S.valuesLabel, S.numbers, S.root, S.levels.{detail, blocks, macro}, S.cells
```
Coordinate: x verso destra (il flusso), y in alto, z verso la camera; ogni livello di annidamento `Z_STEP` (0,035)
più avanti. 1 cella = `CELL` (0,04); un `nn.Linear` (out, in) è alto `out`, riga 0 in alto; i bias sono **colonne**
accanto alle righe di W, γ e β delle LayerNorm **righe** (vettori stirati ×3 sul lato sottile). Oggetti delle
animazioni: `renderOrder` ≥ 9, materiali propri, `userData.noBounds = true`, `raycast = () => {}`, liberati in
`onStop`; durante un'azione il livello si blocca con `tab.scene.lod.setMode("detail")`. Le matrici fuori budget
(`SCENE_LIMITS`: 8 M di celle) sono riquadri grigi e i loro numeri restano nel pannello Tensore. Il mouse colpisce
solo i piani delle matrici (`host.pick(…, scene.pickables(), false)`). Nessun lavoro per frame a riposo: l'unica
chiamata per frame del tab è `numbers.update`, e solo nel Dettaglio; `distanceScale` si ricalcola ai
ridimensionamenti. Costi (rendering software): scena del corso (110 matrici) 30-96 ms, un altro run 31-43 ms, cambio
di step con lo snapshot già letto ~5 ms, a riposo ~19 µs per frame in Panoramica.

### Run caricati, note della registrazione e limiti

- **Caricamento**: cartella (`webkitdirectory`), file multipli o trascinamento (`recognizeFiles` + `runsFromFiles` di
  `shared/weights/`); il run nuovo diventa A subito e poi si salva nel browser ("Ricordali": IndexedDB, «Dimentica»
  lo toglie); vince l'ultimo caricamento chiesto. B diventa da solo il GPT del corso se era vuoto (avviso
  "Confronto pronto: … (A) contro il GPT del corso (B)").
- **Stessi pesi = stesso run.** L'id di un run caricato è `upload:<impronta dei pesi>` (`fingerprint` di `run.js`):
  due cartelle con gli stessi pesi (per esempio lo stesso export copiato con un altro `--name`, o con note diverse)
  sono lo stesso run; la seconda risulta "Già caricato «…»" e tiene **nome e note della prima**. Per vederle tutte e
  due, i pesi devono essere diversi (un altro training), oppure si toglie la prima con «Dimentica».
- **Note della registrazione** (richiesta del flusso "architettura modificata"): la scheda del run mostra le note
  di `run.json` (comprese quelle che `export_run.py` ha copiato dal trace con il prefisso `"storia del training: "`),
  poi quelle del trace che `run.json` non ha già (un run esportato prima, o una cartella con il solo trace) con lo
  stesso prefisso, poi gli avvisi della libreria; una nota copiata (anche accorciata o con un surrogato spaiato:
  `toWellFormed()` come `_one_line` del Python) non compare due volte; oltre 12, "… e altre N, non mostrate qui"
  (`runNotes(run, { limit })` → `{ notes, hidden }`, `traceNotes(trace)`). Così lo studente legge **perché** un
  grafico manca: per un trace senza `attention_entropy` la sezione "Entropia dell'attention di ogni head" dice, al
  posto della heatmap, «Nessuna heatmap per A (nome): la registrazione del training non ha misurato l'entropia
  dell'attention (in trace.json manca attention_entropy) e la sua nota dice perché: «…»» (`missingAttentionText`).
- **Testo non fidato**: nomi di file, run e tensori, testi di `run.json` e `benchmark.json`, prompt: sempre
  `textContent` / `el()` / `fillText`, mai `innerHTML` (un test lo controlla). `parseRunInfo` toglie una volta sola i
  controlli bidirezionali espliciti (U+202A–U+202E, U+2066–U+2069) da nome, descrizione, data e sorgente
  (`withoutBidiControls` di `shared/weights/validate.js`, riesportata da `formats.js`; un nome fatto solo di quelli
  torna il nome della cartella): un U+202E nel nome rovesciava la targhetta dello stage, la didascalia, lo stato del
  caricamento e le frasi del Confronto e del Benchmark. La stessa regola vale per tutto ciò che dai file finisce in
  una frase: i nomi di cartelle e file (`recognizeFiles`), i percorsi negli avvisi, negli errori e tra gli ignorati
  (`stripBidiControls`, anche nel messaggio di ogni `WeightsFormatError`), i valori della config (una chiave con un
  controllo si salta: pulita potrebbe coprire `n_embd`), le classi di `run.json` (`classesFromRunInfo`) e i run
  salvati nel browser da una versione precedente (`deserializeRun`, `RunStore.list`). Un **tensore** con un controllo
  bidirezionale nel nome si rifiuta (`checkName`; anche `gpt/runs.py` all'export): il nome compare in troppe frasi.
  I caratteri del vocabolario passano da `shownChar` (`shared/weights/display.js`: uno invisibile diventa «U+202E»).
  Le note restano com'erano (ognuna in un elemento suo, al più `MAX_RUN_NOTES` = 2.000 lette) e, dentro una frase,
  nomi e note vanno tra FSI e PDI (`isolate(text)`: U+2068 … U+2069).
- **Link a uno step che non è uno snapshot** (`tensor:<nome>@750`, anche dal Transformer con lo step interpolato): si
  aggancia allo snapshot più vicino e un avviso lo dice.

### Azioni animate (`tabs/lab/actions/`)

Otto azioni del player del nucleo mostrano **come i pesi lavorano insieme**, con i numeri VERI: i pesi del run A
allo step scelto e le attivazioni del motore JS sul prompt dello studente. Lavorano **sulle matrici della scena**:
celle evidenziate (`highlightCells`), numeri che compaiono da vicino, risultati in "carte" (`NumericMatrix`
temporanee) appoggiate accanto ai pesi e allineate perché "riga per colonna" si veda (x sopra W, x[j] sopra la
colonna j; i punteggi dell'attention sopra il buffer `tril`). Una **lavagna** DOM sopra la barra del player
scrive il calcolo con i numeri veri; il pannello del codice evidenzia la riga Python vera; la didascalia del player
ha un'analogia.

| id (`#lab/go=action:<id>`) | Titolo | Che cosa mostra |
|---|---|---|
| `linear` | x · Wᵀ riga per colonna | una `nn.Linear` scelta nel pannello (key/query/value, proj, i due strati della FFN, lm_head): l'ingresso vero, una riga colonna per colonna con la somma parziale, la bias, tutte le righe, il risultato uguale al motore |
| `attention` | q · kᵀ, scala, maschera e softmax | Q e Kᵀ, un punteggio, tutti (anche del futuro), la scala 1/√head_size, la maschera causale sul `tril` vero (−∞), la softmax riga per riga |
| `attention-v` | att · V, concatenazione, proj e residuo | la media dei value pesata, le head affiancate (`torch.cat`), proj, `x + sa(ln1(x))` |
| `layernorm` | LayerNorm: media, varianza, γ e β | μ, x − μ, σ² e √(σ² + ε), x̂, × γ e + β sulle righe vere della scena |
| `ffn` | Feed-forward: 64 → 256, ReLU, 256 → 64 | il primo strato, la ReLU (quanti neuroni si spengono), il secondo strato, il residuo |
| `logits` | Logits e probabilità del prossimo carattere | lm_head riga per carattere, la softmax, il carattere più probabile e quello vero (p e loss) |
| `training-step` | Un passo di training su lm_head | loss del motore; ∂L/∂logits = p − y; ∂L/∂W = (p − y)·hᵀ sopra W; passo SGD con il learning rate di `trace.json` (le celle di lm_head diventano i pesi dopo il passo); la loss che scende; il primo passo di AdamW a confronto; ∂L/∂h e i gradienti di γ e β di ln_f; i blocchi dichiarati, non mostrati |
| `training-history` | I pesi durante il training | gli snapshot veri del run A sulla scena (anche se il motore non lo esegue): lo step del tab si sposta davvero, lavagna con la curva vera della loss, metriche e testo di ogni snapshot; a fine azione lo step di prima |

Run che il motore JS non esegue: le sette azioni dei calcoli mostrano un passo solo che spiega perché (niente
attivazioni inventate) e il pannello offre "Usa il GPT del corso"; `training-history` funziona con ogni run che ha
gli snapshot. Run compatibili di forma diversa: tutto si ricava da `engine.config` e da `S.layoutOf(nome)`.

| File | Esporta | Ruolo |
|---|---|---|
| `index.js` | `installLabActions(tab)` → `{ stage, settings, dispose }` | pannello, registrazione delle 8 azioni, chiusura se run A o step cambiano da fuori |
| `catalog.js`, `search.js`, `code.js` (dati) | `ACTIONS`, `ACTION_IDS`, `CATEGORIES`, `actionCard(id)`; `searchEntries`; `CODE`, `codeFor(id, where)` | schede delle azioni, voci della ricerca, righe Python vere (un test le controlla negli snippet) |
| `settings.js` (puro) | `ActionSettings`, `checkPrompt(text, vocab, blockSize)`, `resolveToken`, `trainingExample`, `LINEAR_CHOICES`, `DEFAULT_PROMPT` | prompt, token, blocco, head, nn.Linear |
| `math.js` (puro) | `linearRow`, `linearForward`, `attentionScores`, `causalMask`, `softmaxRows`, `attentionOutput`, `concatHeads`, `layerNormSteps`, `relu`, `softmax`, `crossEntropy`, `lmHeadGradient`, `lmHeadInputGradient`, `sgdStep`, `adamwFirstStep`, `ADAMW_DEFAULTS`, `LAYER_NORM_EPS`, … | ogni passo nello stesso ordine delle operazioni dei kernel del motore: il risultato coincide bit per bit |
| `data.js`, `training.js` (puri) | `forwardData(engine, ids, {token, block, head})`, `linearData`, `attentionData`, …; `trainingStepData`, `learningRateAt(trace, step)`, `historySeries`, … | la traccia del motore ritagliata per le azioni; passo di training (copia privata del motore) e storia |
| `plan.js` (puro) | `cardRect`, `rowAbove`, `columnRight`, `framePose`, `deskLayout`, `readableColumns`, … | dove stanno le carte, come si inquadra la parte libera, dove stanno lavagna, codice e pad (il posto del pad con `padPlacement` del nucleo) |
| `stepKit.js`, `stage.js`, `cards.js`, `board.js`, `linearKit.js`, `format.js` | `makeSteps`, `progressCount`; `ActionStage`; `Card`, `TokenStrip`; `Board`; `linearLayout`, `linearParts`; `num`, `sci`, … | la regia, le carte, la lavagna (sempre `textContent`), i pezzi comuni di una nn.Linear, i numeri all'italiana |
| `linearAction.js`, `attentionAction.js`, `layerNormAction.js`, `ffnAction.js`, `logitsAction.js`, `trainingAction.js`, `historyAction.js`, `pane.js` | `…Steps(stage, …)`, `createActionsPane` | le azioni e il pannello "Azioni" |

Un passo si dichiara (`makeSteps`): `{ title, caption, duration, code, scene() { … }, board: () => [...], update(p),
frame: () => rect }`; `scene()` ricostruisce sempre tutto lo stato di partenza (`stage.step()` toglie carte e veli del
passo prima), quindi avanti, indietro e "⟲ Ricomincia" danno la stessa scena; con `prefers-reduced-motion` ogni passo
mostra subito lo stato finale. La regia (`ActionStage`): `begin()` fotografa camera, livello, codice e step;
`card(key, {…})`, `strip`, `highlight`, `showValues`, `board.show(blocchi)`, `code(…)`, `frame(rect)`; `end({ flyBack })`
libera tutto (geometrie, materiali, texture, canvas, osservatori, `onFrame`) e rimette camera, livello, codice, step e
modalità. Lavagna, codice e pad si dispongono a eventi (`deskLayout`, mai per frame), con il posto del pad calcolato
come lo calcola il nucleo: se la lavagna seguisse il pad vero si inseguirebbero. Run A o step cambiati dallo studente
durante un'azione la chiudono; "I pesi durante il training" cambia lo step da sé senza chiudersi; la differenza A − B
si spegne quando parte un'azione. Verificate contro PyTorch (180 carte, scarto relativo ≤ 6·10⁻⁶; passo di training =
autograd e `torch.optim.AdamW`); a riposo nessuna callback per frame, durante un'azione ~1 µs.

### Confronto, grafici, benchmark (`tabs/lab/analysis/`)

**Convenzione**: **A** è il run nella scena (spesso il "dopo"), **B** il termine di paragone (spesso il "prima": il
GPT del corso, lo step 0). Le differenze sono sempre **A − B** (lo stesso verso di `scripts/benchmark.py PRIMA DOPO`) e
il cambio relativo è **‖A − B‖ / ‖B‖**.

| File | Esporta | Ruolo |
|---|---|---|
| `index.js` | `installLabAnalysis(tab)` → `{ dispose() }` | pannelli Confronto e Benchmark, viste, link, B automatico; `tab.analysis` (`comparison`, `diff`, `applyPreset`, …) |
| `compareModel.js`, `comparison.js` (puri) | `compareAB(a, b, {trace})`, `sortRows`, `differenceMap`, `onlyHalfRounding`, …; `createComparison(state, {schedule})` | il confronto per tabella e scena sopra `compareTensors` di `shared/weights/compare.js`; il confronto **vivo** (ricalcolato quando cambiano A o B, con i gettoni) |
| `diffMode.js` | `createDiffMode({scene, state, actionRunning})`, `DIFF_LABEL`, `SCALES` | A − B nella scena (`scene.showValues`), veli sui tensori senza gemello |
| `series.js` (puro) | `lossSeries`, `metricSeries`, `groupMetricSeries`, `groupFamilies`, `blockFamily`, `moduleFamily`, `snapshotSeries`, `attentionEntropy`, `missingAttentionText`, `exampleProbSeries`, … | le serie dei grafici dal trace (solo numeri finiti e step interi) |
| `benchmarkModel.js`, `protocol.js` (puri) | `readBenchmark(raw)`, `comparisonRows`, `comparisonWarnings`, `METRIC_ROWS`, …; `ExcerptScore`, `scoreText`, `GreedyRun`, `checkConsistency`, `TOLERANCES`, `PROTOCOL_ID` | `benchmark.json` letto con prudenza (anche rotto); il protocollo `finestre-contigue-v1` e il controllo ✓/≈/✗ (vedi "Run esportabili e benchmark") |
| `runner.js` + `worker.js` | `BenchmarkRunner` (`run`, `cancel`, `dispose`, `busy`), `CancelledError` | il calcolo in un Worker modulo (senza Worker nella pagina, a pezzi da 8 ms) |
| `probe.js`, `presets.js`, `benchData.js`, `format.js` (puri o quasi) | `nextTokenProbe(engine, text, {top})`; `presets(…)`, `PRESET_IDS`; `createBenchSource(state, {courseId})`; `num`, `pct`, … | il prossimo carattere su un prompt; i confronti pronti; `builtin.json` e i `benchmark.json` dei run; numeri all'italiana |
| `comparePane.js`, `compareTable.js`, `promptBlock.js`, `benchmarkPane.js`, `benchTable.js`, `browserBench.js`, `charts.js` | `createComparePane`, `createCompareTable`, `createPromptBlock`, `createBenchmarkPane`, `renderBenchTable`, `createBrowserBench`, `installAnalysisCharts(tab, {bench})` | i pannelli e le sezioni dei grafici |

- **Pannello Confronto**: A e B (run e step, "⇅ Scambia A e B"; gli stessi di `state.runA/step`, `state.runB/stepB`);
  **confronti pronti** ("Che cosa ha cambiato il training" = pesi finali contro step 0, "Step 5.000 contro step 500",
  "Il tuo run contro il GPT del corso"); riepilogo con il tipo dei numeri (float32/float16: pesi finali contro lo
  snapshot dello step 5.000 differiscono solo per l'arrotondamento, max |Δ| ≈ 0,001, e lo si dice); **differenza nella
  scena** (interruttore: A − B sulle matrici di A con la colormap divergente, scala comune o dei pesi, legenda con la
  rampa di `colormapLUT("diverging")`, la targhetta dice "A − B"); **tabella** per tensore (cambio relativo con una
  barra, coseno, ‖A − B‖, max |Δ|, ‖A‖, ‖B‖, forma; ordinabile con `aria-sort`, i più cambiati in alto, i cambi infiniti
  in fondo con la spiegazione, 120 righe e poi "Mostra tutti"; clic su un nome = quel tensore e la camera ci vola);
  **istogrammi sovrapposti** (`Histogram` condiviso, stessi intervalli); **il prossimo carattere** su un prompt
  (probabilità e logit di A e di B dal motore). Abbinamento per nome esatto, poi senza `module.`/`_orig_mod.`; con
  un'architettura diversa i tensori con lo stesso nome e forma diversa si elencano ("192×64 in A, 256×64 in B") e nella
  scena si velano.
- **Pannello Benchmark**: i numeri del Python per A e B allo step scelto (`benchmark.json`; per il GPT del corso
  `builtin.json`, anche per gli 11 snapshot) con A − B e la colonna "meglio" (loss, perplexity, bit/carattere ↓;
  top-1, top-5 ↑; entropia, parametri e token/s senza verdetto, token/s dichiarati con l'ambiente), gli avvisi di
  confrontabilità e i testi generati; **ricalcolato nel browser** per i run compatibili ("Calcola per A", "Calcola per
  B", "Tutti gli snapshot di A", "Annulla") nel Worker con la barra di avanzamento, poi il greedy, e il controllo
  ✓/≈/✗. I risultati restano in memoria per (run, step), al più 40.
- **Grafici** (sezioni del `ChartsPanel` del nucleo): Training (loss train e validation, loss del batch, learning rate,
  norma del gradiente, update ratio con l'intervallo vero di A; `lab-an-training`), per gruppo di pesi (update ratio e
  deviazione standard per famiglia: attention, feed-forward, LayerNorm con una linea per blocco, fuori dai blocchi; per
  nomi nuovi `blockFamily` mette `*` al posto del solo indice del blocco, `moduleFamily` dà una famiglia ai gruppi di un
  modulo senza indice, come un blocco riusato; al più 8 linee; `lab-an-groups`; i limiti della regola dei gruppi sono
  in "Storia del training di qualunque architettura"), metriche degli snapshot
  (`lab-an-snapshots`), entropia dell'attention di ogni head (heatmap blocco·head × step per A e per B, o il perché
  manca; `lab-an-attention`), il prossimo carattere sul prompt (`lab-an-prompt`), benchmark (loss per snapshot, sorpresa
  lungo l'estratto; `lab-an-bench`). La linea verticale è lo step di A; "A" è sempre il primo colore e "B" il secondo;
  con lo stesso run a due step, B è un punto sulla curva di A. `installAnalysisCharts` → `setProbe`, `setSurprisal`,
  `reveal(sezione)`, `redraw`.
- **Chi possiede i valori della scena**: la differenza ricorda gli array che ha dato a ogni matrice; se un altro modulo
  cambia modalità (es. un'azione con `setMode("valori")`) si spegne e rimette i pesi solo dove la scena mostra ancora i
  suoi array. Nessun `onFrame`: tutto a eventi, al più un ricalcolo per frame. Misure: `compareAB` del corso 2,3 ms,
  differenza nella scena 7 ms, tabella 10 ms, benchmark dell'estratto nel Worker 0,8-2,4 s + greedy 1,1-5 s; GPT del
  corso: pesi finali e 11 snapshot ✓.

### Link diretti e ricerca del Laboratorio

`#lab/go=<kind>:<id>`; `goto` (`gotoTarget`) comincia con `reps.goto` (`view:rete`, `view:cervello`), poi i tipi
registrati con `tab.links.register`, poi:

| Link | Effetto |
|---|---|
| `tensor:<nome>[@<step>]` | il tensore del run A se ce l'ha, altrimenti del GPT del corso; `@<step>` = lo snapshot (agganciato), `@finali` o niente = pesi finali; camera, codice, pannello Tensore; anche il buffer `tensor:blocks.0.sa.heads.0.tril`. È il link di "Apri nel Laboratorio pesi" del Transformer |
| `run:<id>` | un run del registro come A (`run:corso` = il GPT del corso, `RUN_ALIASES`) e il pannello Run |
| `view:modello` · `tensore` · `run` · `carica` · `esporta` · `step` · `centro` | i pannelli, il caricamento, la guida all'esportazione, il cursore dello step, tutto il modello |
| `view:step:<n>` · `view:step:finali` | lo step del training (del run A, o del GPT del corso se A non ha la storia) |
| `view:tensor:<nome>[@<step>]`, `view:run:<id>` | come sopra, in forma di vista (la ricerca accetta solo `step` · `action` · `view`) |
| `view:confronto` · `confronto:training` · `confronto:fasi` · `confronto:corso` | il pannello Confronto; i confronti pronti con la differenza nella scena (`corso` senza run caricati: avviso) |
| `view:differenza` · `probabilita` · `grafici` · `benchmark` · `benchmark:browser` | la differenza A − B (senza B: il confronto "training"), il prompt del prossimo carattere, i grafici del training, il pannello Benchmark (con il fuoco sul calcolo nel browser) |
| `action:linear\|attention\|attention-v\|layernorm\|ffn\|logits\|training-step\|training-history` | le azioni animate |
| `view:rete` · `view:cervello` | rappresentazioni del nucleo |

Un id che non esiste → `false` (il guscio avvisa) e niente cambia (il run A resta quello che era).

### Rappresentazioni del Laboratorio

- **Rete neurale** (`network.js`, `cache: false`): run compatibile con il motore → `buildDepthSpec` del Transformer
  (in sola lettura) con la traccia vera del motore sul prompt d'esempio `"ROMEO:\nBut, soft!"` (17 token) e i pesi del
  run allo step scelto (badge "Attivazioni vere"); altri run → **sola struttura** dai nomi e dalle forme (misure lette
  dai pesi, es. FFN da 192), pesi veri delle `nn.Linear` sui collegamenti, attivazioni simulate e dichiarate ("Valori
  non disponibili: architettura diversa dal motore JS. Il motivo: …"); architettura sconosciuta → i moduli in fila
  nell'ordine dei pesi. `createNetworkSource(tab)` → `{ load, invalidate, attach, dispose }`: se run A o step cambiano
  con la Rete neurale aperta, `reps.refresh("network")` 450 ms dopo l'ultimo cambio (due clic: una ricostruzione,
  camera identica). Con un livello aperto il cambio di run non fa voli nella scena del tab: si inquadra al ritorno.
- **Cervello** (`brain.js`): vedi la tabella dei moduli; `normalizeBrainSpec(brainSpec, { strict: true })` senza avvisi.
- Altre regole: interpolazione solo se lo studente la accende, sempre "approssimata"; il buffer `tril` è "buffer, non è
  un peso"; i moduli che il codice del corso non crea lo dicono ("non nel codice del corso"); sul telefono un tocco su
  una matrice mostra valore e significato per 2,6 s e sceglie il tensore senza aprire il codice («Mostra il codice» del
  pannello Tensore); la didascalia dice il run nella scena (non lo step: lo dice la targhetta) e tace durante un'azione
  e sotto un avviso del nucleo.

## Il tab LLM moderni: vista d'insieme e architettura completa (`tabs/modern-llm/`)

Due sottotab, scelti da una barra in cima allo stage (`subtabs.js`, tasto **V** per passare all'altro):
- **Vista d'insieme tecnologie LLM moderni** (`overview`): i 16 meccanismi (RMSNorm, RoPE, GQA, MoE, KV cache,
  FlashAttention, BPE, decodifica speculativa, scaling, post-training, ragionamento, agenti, coding, multimodale, mesh
  3D, interpretabilità) su una griglia di regioni, con il Percorso di 22 passi e 18 azioni: identica a prima.
- **Architettura LLM moderni completa** (`full`): un modello aperto vero, **DeepSeek-V4.1-Flash** (alternative **Kimi
  K3.1 Pro** e **gpt-oss-120b**), in tre livelli di zoom: Panoramica (dati → pre-training → contesto lungo →
  post-training → **pesi** → deployment → inferenza), Blocchi (tokenizer → embedding → layer raggruppati in "× N" con
  lastre fantasma ed "Espandi le ripetizioni" → norm → head → softmax, rami e riga delle cache), Dettaglio (l'anatomia
  del layer scelto con i nomi veri dei tensori, la griglia degli esperti con quelli attivi per token, **scelta
  simulata** e dichiarata). Pannelli Percorso (14 passi), Azioni (8), Viste, Modelli, **Pesi** (ripartizione dei
  parametri, nomi dei tensori, shard), Fonti; i 16 meccanismi hanno un segnaposto numerato dove vivono (clic → la loro
  regione nella vista d'insieme con "← Torna all'architettura completa"; ogni passo della vista d'insieme ha "Dove sta
  in DeepSeek-V4.1-Flash →").

Tutto nello **stesso SceneHost** (nessun contesto WebGL in più): la vista completa aggiunge i suoi gruppi con un suo
`LODController` e il tab ne accende uno solo (`lod.setEnabled`); il suo codice si carica con un import dinamico al
primo ingresso. Cambiando sottotab (`showSubtab(id, { restore })`) l'azione si ferma, codice e didascalia si chiudono,
la posa della camera del sottotab lasciato si salva e quella dell'altro torna (`restorePose` alza prima
`controls.maxDistance` alla distanza della posa), i pannelli si scambiano restando sullo stesso tipo (`PANE_PAIRS`) e le
sezioni dei grafici `full-…` si mostrano solo nella vista completa (`charts.setSectionVisible`). In modalità completa
la camera usa `near = 0.5`, `far = 40000` (rimessi in `deactivate()`); la radice ha la classe `mllm-full`.

- **Barra dei sottotab**: `role="tablist"` (frecce, Home/Fine, focus visibile), etichette brevi sotto i 780 px, primo
  figlio della radice del tab (Tab ci arriva subito); è nel flusso, sopra lo stage: `.mllm-has-subtabs` sposta in
  basso di `--mllm-subtabs-h` (42 px; 40 a ≤ 900 px) stage, "Grafici" e i loro pannelli. Area di contesto a destra
  (`setContext({ back, link })`: "← Torna …", "Dove sta in DeepSeek-V4.1-Flash: «meccanismo» →"). Il tasto V
  (`SUBTAB_KEY = "KeyV"`) non agisce nei campi, nelle finestre di dialogo, con l'aiuto aperto (`nav.modalOpen`, anche
  quello di un livello) né con Ctrl/⌘/Alt; compare nell'aiuto con `nav.pad.addShortcut`. Un clic sulla scheda già
  attiva chiude un livello delle rappresentazioni aperto (`onReselect`).
- **Esperienza d'uso**: il livello scelto (selettore, Viste, `view:full-macro|blocks|detail`) resta bloccato da subito;
  le inquadrature della Panoramica la bloccano; i grafici si aprono da soli solo se restano 700 px di scena
  (altrimenti "Grafici" si illumina 2,6 s), il codice di un passo parte ridotto sotto 600 px; sul telefono in verticale
  la soglia del Dettaglio mostra ~4 blocchi e una didascalia insegna il pizzico; cambiando modello il codice si chiude
  (estratti e pseudocodice sono di V4.1).

| File | Esporta | Ruolo |
|---|---|---|
| `subtabs.js` (solo DOM) | `createSubtabs({root, items, current, onSelect, onReselect, isBlocked})` → `{element, current, setCurrent, setContext, dispose}`, `SUBTAB_KEY`, `RESET_TITLE` | barra dei sottotab, area di contesto, tasto V |
| `sceneClick.js` (solo DOM) | `installSceneClick(canvas, onClick)` → `off()` | clic veri sulla scena (≤ 5 px di spostamento): non scatta dopo un trascinamento |
| `full/models/deepseekV41Flash.js`, `full/models/kimiK31Pro.js`, `full/models/gptOss120b.js` (SOLI DATI), `full/models/index.js` | `DEEPSEEK_V41_FLASH`, `KIMI_K31_PRO`, `GPT_OSS_120B`; `MODELS`, `MODEL_BY_ID`, `PRIMARY_MODEL_ID` | un modulo per modello, con fonti e marcature "da verificare" |
| `full/research.js` (dati) | `RESEARCH_DATE`, `HOW_OBTAINED`, `NAME_CHECK`, `LANDSCAPE`, `CAPABILITY_FACTS`, `SCALING_POINTS` | come sono stati ottenuti i numeri, panorama degli open-weight, capacità emergenti con la critica di Schaeffer et al. |
| `full/model.js` (puro) | `compressLayers`, `layerList`, `layerParams`, `paramTotals`, `addOnTotal`, `buildStrip`, `blockOfLayer`, `anatomyCards`, `cardState`, `layerSummary`, `formatLayers`, … | ripetizioni, conteggi dei parametri, striscia dei Blocchi, schede dell'anatomia |
| `full/layout.js`, `full/tensors.js` (puri) | `stripLayout`, `macroLayout`, `anatomyLayout`, `STRIP`, `MACRO`, `ANATOMY`, …; `enumerateCheckpoint(model)` | posizioni dei tre livelli; il checkpoint ricostruito tensore per tensore |
| `full/catalog.js`, `full/content.js`, `full/snippets.js` (dati) | `FULL_ACTIONS`, `FULL_ACTION_IDS`, `FULL_LEVELS`, `FULL_SECTIONS`, `MARKER_LABELS`, …; `FULL_PATH`, `MACRO_NOTES`, `UNVERIFIED_LABEL`; `FULL_SNIPPETS` | azioni, livelli, sezioni, Percorso, codice (estratti VERI di DeepInfer con file e righe, pseudocodice dichiarato) |
| `full/draw.js`, `full/sceneMacro.js`, `full/sceneBlocks.js`, `full/sceneDetail.js` | `textPlate`, `LineBatch`, `boxBatch`, …; `buildMacro`; `buildBlocks`; `buildAnatomy`, `buildContext` | three.js: lastre di testo, linee e scatole in un solo oggetto; i tre livelli |
| `full/panes.js`, `full/plots.js`, `full/actions.js`, `full/index.js` | `renderViews`, `renderModels`, `renderWeights`, `renderSources`; `installFullPlots`, `FULL_CHART_SECTIONS`; `fullActions(fx)`; `createFullArchitecture(o)`, `FULL_PANES` | pannelli, grafici, le 8 azioni, il montaggio |
| `network.js`, `networkMechanisms.js` (puri) | `buildNetworkSpec(model)`, `networkSpec`, `networkSpecFor(model)`, `PROMPT`, `SHOWN`, `readableText`, `layerSectionIds`; `layerRecipe`, `drawnLayers`, `layerAttention`, `selectEntries`, `entryReady`, `entryCount`, `seededRandom` | la Rete neurale di DeepSeek-V4.1-Flash (sotto) |
| `brain.js` (SOLI DATI, generato) | `brainSpec` | il Cervello del tab |

**Formato di un modello** (`full/models/<id>.js`, soli dati; `model.js` ne ricava tutto il resto): identità (`id`,
`name`, `short`, `version`, `org`, `license`, `repo`, …), `config` (i campi usati, con il nome originale nei commenti),
`layers` (`kinds`: **un carattere per layer**, es. V4.1 `"SSFRRRRRFRRRRRFRRRRRDUUUXUUUXUUUXUUUXUUU"`; `base`; `flags`
come Engram nei layer 1 e 14; `segments` encoder/decoder), `kinds` (etichetta, colore e testo di ogni carattere),
`params` (per layer: attention per tipo, MoE o FFN densa, residuo, norme, flag, blocchi fuori dal backbone, `addOn`),
`input`, `output`, `branches`, `caches`, `pipeline`, `precision`, `checkpoint`, `kv`, `anatomy` (schede con `in`,
`ghost`, `when`, `alt`: `cardState(card, layer)` → `"on" | "ghost" | "off"`), `mechanisms` (per i 16 meccanismi: `at`,
`status: "present" | "absent" | "study"`, `text`), `tensorGroups` (nomi veri dei tensori), `unverified` (ciò che la
ricerca segna **da verificare**), `sources`. Aggiungere un modello = un modulo con questi campi + una riga in
`models/index.js`; i test controllano la coerenza. Le ripetizioni "× N" vengono da `compressLayers(keys, {maxUnit = 8})`
(programmazione dinamica: V4.1 `S×2 | 3×[F, R×5] | D×1 | U×3 | 4×[X, U×3]`, Kimi `23×[K×3, M] | M×1`, gpt-oss `18×[B,
G]`); i parametri da `paramTotals(model)` (V4.1: backbone 551.880.850.544, attivi 16,13B in decode, checkpoint di
96.085 tensori e 510.286.023.000 byte, ricostruito da DeepInfer: torna esatto).

**Fedeltà**: ogni numero, nome, forma e fonte confrontato con la ricerca verificata (`llm_specs.json`,
`llm_frontier.md`) da script indipendenti; un valore non verificato porta sempre "da verificare"; un'assenza si dice
rispetto alle fonti ("nelle fonti consultate non compare…"); le scelte degli esperti sono "simulazione didattica";
dove la ricerca non dà il nome di un tensore si scrive "nome non verificato".

**Rappresentazioni** (per tutto il tab: lo stesso selettore vale nei due sottotab). La **Rete neurale** è DeepSeek-
V4.1-Flash con la profondità vera (`network.js`, `cache: false`): x = token → embedding → residuo a 4 corsie (mHC) →
**un layer per ogni blocco della striscia dei Blocchi**, con le stesse ripetizioni (7 layer disegnati dei 40, i gruppi
come `{ repeat, of }`) → norm → head → softmax → DSpark; y = i numeri del vettore ridotti e dichiarati (`SHOWN`); z = i
12 token del prompt d'esempio (`PROMPT`, «Perché il cielo è blu?» con i token speciali del tokenizer di V4.1; divisione
e ordine del template dichiarati illustrativi). I pezzi di ogni layer vengono dal config (`layerRecipe`:
`compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`), con finestra locale, KV globale compressa,
indexer, router e **6 esperti attivi per token** su 16 dei 384 disegnati; pesi di attention illustrativi e causali
(`illustrative: true`). `networkSpecFor(model)` dà la rete del modello scelto nel selettore se il suo config la
descrive, altrimenti quella del principale con una frase che lo dice (oggi la descrive solo il principale). 29.992
neuroni, 14.044 segmenti, 1.211 archi. Il **Cervello** (`brain.js`, generato da `brain_data.json → tabs["modern-llm"]`):
10 analogie (MoE, KV cache, RL e dopamina, ragionamento, RoPE, leggi di scala, interpretabilità, multimodale, agenti,
BPE), l'anello corteccia – gangli della base – talamo con la dopamina (8 gruppi, 11 sinapsi), 19 regioni, 9 numeri, 53
fonti. La barra dei sottotab resta visibile sopra un livello (una regola più specifica di `.rep-alt-active` in
`modern-llm.css`); scegliere un sottotab chiude prima il livello; il pannello "Cervello" sta sempre prima delle fonti
del sottotab attivo (`sidebar.movePane(BRAIN_PANE_ID, "fonti" | "full-fonti")`).

| Link | Cosa fa |
|---|---|
| `view:overview`, `view:full` | apre il sottotab (`full`: dove eri, la prima volta la casa) |
| `view:full-macro`, `view:full-blocks`, `view:full-detail` | livello dello zoom, bloccato subito |
| `view:full-<sezione>` | un pezzo dell'architettura (19, `FULL_SECTIONS`): `full-data`, `full-context`, `full-post`, `full-weights`, `full-deploy`, `full-capabilities`, `full-tokenizer`, `full-embedding`, `full-layers`, `full-encoder`, `full-decoder`, `full-caches`, `full-dspark`, `full-vision`, `full-head`, `full-attention`, `full-moe`, `full-residual`, `full-engram` |
| `view:full-model-<id>` | cambia modello: `deepseek-v4.1-flash`, `kimi-k3.1-pro`, `gpt-oss-120b` |
| `view:full-layer-<n>` | Dettaglio del layer n del modello attivo |
| `view:full-mech-<regione>` | dove vive un meccanismo (`compare`, `moe`, `gqa`, `rope`, `bpe`, `kv`, `flash`, `spec`, `scaling`, `training`, `reasoning`, `interp`, `agent`, `coding`, `multi`, `mesh`) |
| `action:full-<id>` | `full-token`, `full-router`, `full-kv`, `full-ced`, `full-engram`, `full-dspark`, `full-train`, `full-emerge` |
| `step:full-<id>` | i 14 passi del Percorso della vista completa (`full-intro`, `full-zoom`, `full-data`, …, `full-emerge`) |
| `action:<id>`, `step:<id>`, `view:macro\|blocks\|detail\|<regione>` | la vista d'insieme, come prima (18 azioni) |
| `view:rete`, `view:cervello` | rappresentazioni del nucleo |

`goto` comincia con `reps.goto`, poi sceglie il sottotab con `isFullTarget(kind, id)` (id `full…` o passi di
`FULL_PATH`). API: `tab.subtabs` (`show(id, {restore})`, `current`, `crossToFull(regionId)`, `crossToOverview(regionId)`,
`goBack()`), `tab.full` dopo il primo ingresso (`activate`, `deactivate`, `goto`, `home`, `bounds()`, `focusMechanism`,
`setModel(id, {fly, stopAction})`, `selectLayer(n, {fly})`, `model`, `lod`, `menu`, `panes`, `actionIds`,
`chartSections`, `dispose`), `regionCompass.js` `setEnabled(enabled)`. Test: `tests/modern-llm-full.test.mjs`,
`modern-llm-full-fidelity.test.mjs`, `modern-llm-full-ux.test.mjs`, `modern-llm-reps.test.mjs`. Misure (rendering
software): primo ingresso 480-590 ms, cambio di modello 73-206 ms (dispose completo), a riposo 0,75 µs per frame per il
LOD della vista completa; `app.goto` verificato su tutte le voci della ricerca del tab, anche con un livello aperto.

## Gli altri tab

La tabella riassume tutti e sei i tab, nell'ordine della barra: cartella, moduli principali, rappresentazioni e link
diretti (`#<tab>/go=<kind>:<id>`; ogni tab ha anche `step:<id>` per i passi del suo Percorso). Transformer, Laboratorio
pesi e LLM moderni hanno la loro sezione qui sopra; RAG, Immagini e video e Audio hanno i dettagli delle
rappresentazioni qui sotto. In ogni tab il `goto` comincia con `reps.goto(target)`: `view:rete` e `view:cervello`
mostrano la rappresentazione, ogni altro link chiude il livello e prosegue come sempre.

| Tab | Cartella | Moduli principali | Rappresentazioni | Link diretti |
|---|---|---|---|---|
| Transformer | `tabs/transformer/` | `architecture.js`, `flow.js`, `lod.js`, `views/` (Pesi, **Pesi numerici** `numbers*.js`, Rete neurale `neurons.js` + `neuronDepth*.js`), `actions/`, `context/` | Rete neurale = la vista `neuroni` (modalità personalizzata), Cervello = `brain.js` | `action:prompt\|training-step\|training-run\|lod-zoom\|weights-tour\|neurons-forward\|context-pipeline`, `view:architettura\|pesi\|numeri\|numeri.training\|neuroni\|neuroni-profondita\|contesto\|zoom.<livello>\|ctx.<passo>\|rete\|cervello`, `tensor:<nome>[@<step>]` |
| Laboratorio pesi | `tabs/lab/` | `state.js`, `scene/`, `inspect/`, `runs/`, `actions/` (8 azioni), `analysis/` (Confronto, grafici, Benchmark), `network.js` | Rete neurale del run A (`network.js`: vera se il motore lo esegue, altrimenti la struttura), Cervello = `brain.js` | `tensor:<nome>[@<step>]`, `run:<id>`, `view:modello\|tensore\|run\|carica\|esporta\|step\|centro\|step:<n>\|confronto\|confronto:training\|confronto:fasi\|confronto:corso\|differenza\|probabilita\|grafici\|benchmark\|benchmark:browser\|rete\|cervello`, `action:linear\|attention\|attention-v\|layernorm\|ffn\|logits\|training-step\|training-history` |
| Database vettoriale (RAG) | `tabs/rag/` | pipeline L17-L18 a stazioni: upload, estrazione, pulizia, chunking, embedding, `vector_store`, indice HNSW; poi ricerca HNSW, prompt, LLM, streaming. Azioni "Indicizzazione" e "Ricerca" | `network.js` (embedder e database con i valori veri, encoder E5 e Llama 3 illustrativi), `brain.js` (la memoria: ippocampo) | `action:indicizzazione\|ricerca`, `view:detail\|blocks\|macro\|rete\|cervello`, oltre a `#rag/q=<domanda>` |
| Immagini e video | `tabs/diffusion/` | DDPM/DDIM, Stable Diffusion (text encoder, VAE, U-Net, scheduler), DiT, video (VAE 3D, patch spazio-temporali). Azioni di training, generazione, VAE, denoiser, video. `sceneNav.js`: "Centra", confini per livello e didascalie delle due parti | `network.js` (CLIP, U-Net con la griglia del latente in profondità, VAE, blocco video), `brain.js` (la via visiva) | `action:forward\|generate\|vae\|inside\|video`, `view:unet\|dit\|pipeline\|video\|macro\|detail\|rete\|cervello` |
| Audio e musica | `tabs/audio/` | forma d'onda, STFT, Mel, codec neurale con RVQ, generazione di token musicali (delay pattern), Whisper; laboratorio con suono vero; `catalog.js` (soli dati) e `whisperTokens.js` (i token d'esempio di Whisper, soli dati). Limiti di rotazione (`setOrbitLimits`) | `network.js` (EnCodec, il codec del tab con i numeri veri, MusicGen, Whisper, il tempo in profondità), `brain.js` (la via uditiva) | `action:audio-sound\|audio-rvq\|audio-generate\|audio-whisper`, `view:overview\|delayCore\|rvq\|whisperAll\|detail\|blocks\|macro\|rete\|cervello` |
| LLM moderni | `tabs/modern-llm/` | due sottotab (`subtabs.js`, tasto V): **Vista d'insieme** con i 16 meccanismi (regioni `region*.js`, "Dove sei" di `regionCompass.js`, dati in `regionData.js`, `actionCatalog.js`, `palette.js`) e **Architettura completa** (`full/`). La griglia e la Panoramica si inquadrano nella parte dello stage libera da zoom, selettore e pad (`poseInFreeRect` con `avoidRectVariants` e `fitInFreeRect` di `core/neuralNet/framing.js`: sopra il pad o alla sua sinistra, dove vengono più grandi), anche con "Centra" | `network.js` + `networkMechanisms.js` (DeepSeek-V4.1-Flash: 7 layer della striscia con "× N", token in profondità, MoE 6 attivi), `brain.js` | `action:<id>` (18) e `action:full-<id>` (8), `view:macro\|blocks\|detail\|<regione>\|overview\|full\|full-macro\|full-blocks\|full-detail\|full-<sezione>\|full-model-<id>\|full-layer-<n>\|full-mech-<regione>\|rete\|cervello`, `step:full-<id>` |

Ogni tab ha `window.__<nome>Tab` (es. `__labTab`, `__ragTab`, `__diffusionTab`, `__audioTab`, `__modernLlmTab`;
per il Transformer `__gptApp`) per la console e per gli script di verifica; il guscio è `window.__app`
(`__app.goto(tab, {kind, id})`).

### Rappresentazioni di RAG, Immagini e video, Audio

I tre tab dichiarano le rappresentazioni del nucleo con `installRepresentations` dopo i loro pannelli e prima di
"Fonti", con moduli di soli dati: `network.js` (puro, deterministico, niente three.js né DOM: costruisce la
`networkSpec`, spesso dai dati veri che il tab calcola) e `brain.js` (generato dalla ricerca verificata
`brain_data.json → tabs[<id>]` con uno script del round, senza riscrivere i testi: cambiano solo i nomi dei campi in
quelli canonici del nucleo; ogni numero che la verifica segna "da ricontrollare" porta la cautela nella nota, o è
scartato dal generatore di Immagini e video). Il ritorno automatico del nucleo copre tutto ciò che i tab fanno sulla
loro scena (passi, ↑ ↓, Viste, azioni, link, ricerca): nessuno chiama `reps.show("main")` da solo, salvo il RAG quando
la configurazione svuota il database.

- **RAG** (`buildRagNetworkSpec({ kb, run, origin })`, `cache: false`: la rete si rifà a ogni apertura con la domanda
  di adesso; `origin` = "panel" | "last" | "example" dice da dove viene, nell'introduzione). Quattro parti con due
  parentesi grandi, **"Valori veri"** e **"Illustrativo: valori simulati"**: 1 · Embedder (le feature della domanda in
  colonna, l'hashing con il segno come collegamenti `explicit` di peso ±1 verso la sua dimensione, il vettore
  normalizzato: veri, da `embedder.explain`); 2 · Database (i chunk **in profondità**, i prodotti q·c, il coseno = la
  ricerca esatta, top-k e soglia = i risultati di `similaritySearch`: veri); 3 · Encoder E5 (multilingual-e5-small:
  attention bidirezionale, mean pooling; illustrativo, `slices: false`); 4 · LLM Llama 3 8B (attention causale, solo
  l'ultimo token predice; illustrativo, `slices: false`). Numeri dei modelli veri in `ENCODER` e `GENERATOR`, con le
  fonti; badge "in parte vere" (7 sezioni vere su 22). `hnswNote(run, N, efSearch)` dice quanto ha lavorato davvero
  l'indice (con l'indice didattico rilegge alcuni nodi a ogni livello; con i valori di pgvector il grafo ha un livello
  e `ef_search` 40 legge tutti i 21 nodi una volta), come la didascalia dell'azione Ricerca (`hnswSearchCaption` di
  `hnswCost.js`). `brain.js`: 10 analogie, il circuito trisinaptico dell'ippocampo (8 gruppi, 14 sinapsi), 10
  regioni, la via perforante, 5 numeri. Test: `tests/rag-reps.test.mjs` (valori veri uguali alla pipeline, somma dei
  prodotti = coseno, top-k = HNSW, casi limite).
- **Immagini e video** (`networkSpec` = `buildDiffusionNetworkSpec()`): text encoder CLIP ViT-L/14 (attention causale) →
  U-Net (latente z_t con la **griglia spaziale in profondità**, conv 3 × 3, `down`/`up`, skip connection per livello,
  self-attention e cross-attention dal testo, il passo t a ogni ResNet block con `broadcast`) → scheduler DDIM → decoder
  del VAE, più un blocco di denoiser video (griglia frame × patch, attention spaziale e temporale fattorizzata). Fette
  = i 10 token del prompt (le sezioni a griglia non partecipano). Veri solo il rumore iniziale z_T dell'azione
  Generazione (`initialNoiseCrop`, seed 42) e 8 dei 320 numeri dell'embedding del passo t = 951 (`timestepValues`); la
  **cross-attention è uno schema illustrativo** (`crossAttentionMap`: ogni posizione guarda la parola della sua zona
  nell'immagine del tab, `sceneZone`: «faro», «scogliera», «tramonto»; `illustrative: true`), tutto il resto simulato e
  dichiarato. Scala in `scale` (griglia divisa per 8, 1 canale ogni 320, 2 delle 8 head…); 4.328 neuroni. Tornando
  all'Architettura con la camera sopra il banco di lavoro di un'azione chiusa si vola alla vista d'insieme. `brain.js`:
  10 analogie, la gerarchia visiva con il feedback (retina → genicolato → V1 → V2/V4 → IT → prefrontale), 13 regioni,
  la radiazione ottica. Test: `tests/diffusion-reps.test.mjs`.
- **Audio e musica** (`buildAudioNetworkSpec(lab, { style })`, `cache: false`; `tab.setStyle` chiama
  `reps.refresh("network")` se la rete è aperta, così cambiando brano la rete si rifà con la stessa camera). Quattro
  reti in fila con **lo stesso tempo in profondità**: le fette sono 12 frame da 20 ms (240 ms attorno all'attacco
  della seconda nota), gli stessi istanti in tutte le reti (EnCodec 18 e 36 posizioni e il log-Mel 24 con il passo in
  z proporzionale). EnCodec 24 kHz (struttura vera, valori simulati), il **codec del tab con i numeri veri** (spettro,
  voci dei 4 codebook scelte, residui, somma: `codecWindow`), MusicGen small (token del codec sfasati dal delay
  pattern e probabilità del modello a conteggi del tab, dichiarato; testo del prompt `slices: false`), Whisper tiny
  (log-Mel vero e posizioni sinusoidali, `whisperWindow`, `whisperSinusoid`; token di testo d'esempio da
  `whisperTokens.js`, `slices: false`); badge "in parte vere" (15 sezioni su 52). Numeri delle architetture in
  `ENCODEC`, `MUSICGEN`, `WHISPER_TINY`, dal codice ufficiale. `brain.js`: 9 analogie, la via uditiva dalla coclea
  alla corteccia (10 gruppi, 12 sinapsi), 14 regioni, 3 percorsi, 4 marcatori. Test: `tests/audio-reps.test.mjs`.

Le reti si inquadrano a parti quando sono lunghe (vedi "Reti lunghe" in "Rete neurale orizzontale"); i selettori non
coprono nessun comando del tab a 1600×900, 1280×800, 1100×800 e 390×844; dopo 10 aperture e chiusure listener,
geometrie e texture tornano quelli di prima e resta un solo contesto WebGL vivo.

## Test

- Python: `pytest` (dalla radice; su una macchina carica `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, e comunque
  `tests/conftest.py` tiene PyTorch a 1 thread durante i test). Tra gli altri: `test_runs.py`, `test_benchmark.py`
  (anche il motore JS che riproduce `builtin.json`), `test_recording_architectures.py`, `test_recording_cli.py`,
  `test_webapp_data.py` (i dati di `webapp/data/` allineati al codice: non vanno rigenerati a mano).
- JavaScript: `node --test webapp/tests/*.test.mjs` (moduli puri senza DOM: motore GPT, libreria RAG, ricerca,
  navigazione, rete neurale, cervello, pesi, i tab; `brain-pane.test.mjs`, `tensor-ui.test.mjs` e
  `lab-notes.test.mjs` con un DOM finto minimo; `weights.test.mjs`, `neuralnet-scene.test.mjs`,
  `lab-core-scene.test.mjs` e `lab-actions.test.mjs` con three.js da `vendor/` e un canvas finto). Le specifiche dei
  tab: `transformer-depth.test.mjs`, `transformer-numbers.test.mjs`, `transformer-reps.test.mjs`, `rag-reps.test.mjs`,
  `diffusion-reps.test.mjs`, `audio-reps.test.mjs`, `modern-llm-reps.test.mjs`, `modern-llm-full*.test.mjs`,
  `lab*.test.mjs`, `fidelity-fixes.test.mjs` (con `BRAIN_RESEARCH=<brain_data.json>` alcuni confrontano anche i
  testi del cervello con la ricerca, parola per parola; senza, quei controlli si saltano). Tre test del Laboratorio
  caricano run esportati con `scripts/export_run.py` come farebbe lo studente: pesano qualche MB e non stanno nel
  repository, quindi si indicano con `LAB_RUNS_DIR=<cartella>` (con `gpt-200-step`, `ffn3` e `gpt-con-gelu`, descritti
  in `tests/runFixtures.mjs`); senza, quei test si saltano dicendo perché.
- Pagine di prova dei moduli del nucleo (aperte con lo stesso server): `tests/charts-demo.html`,
  `tests/reps-demo.html` (`?viste=vere` = integrazione con rete, cervello e pannello veri, sulle specifiche di
  riferimento del nucleo),
  `tests/neuralnet-demo.html`, `tests/brain-demo.html`, `tests/matrix-demo.html`.
- Visivo: `python scripts/serve_webapp.py --no-export --port 8765` poi
  `NODE_PATH=$(npm root -g) node webapp/tests/screenshot.mjs http://127.0.0.1:8765/ <cartella> [tab1,tab2,…] [--quick] [--lod] [--nav] [--mobile] [--sizes=1600x900,1100x800,390x844]`.
  Tutti e sei i tab nella stessa pagina (si passa da un tab all'altro come lo studente), uno screenshot all'apertura e
  uno per ogni vista principale (Transformer: pesi, numeri, numeri.training, contesto, neuroni, neuroni-profondita,
  cervello; Laboratorio: confronto, benchmark, rete, cervello; RAG, Immagini e video, Audio: le loro viste, rete e
  cervello; LLM moderni: full, rete, cervello); un elemento dell'elenco può essere un link diretto
  (`transformer/go=view:numeri`). `--quick` = solo l'apertura; `--lod` uno screenshot per livello dello zoom; `--nav`
  uno dopo uno spostamento con D e W tenuti premuti e uno dopo C (deve tornare alla vista iniziale); `--mobile` =
  390×844 touch; `--sizes` più dimensioni (sotto 600 px telefono touch). Una pagina di riscaldamento assorbe gli
  avvisi "GPU stall due to ReadPixels" della prima pagina WebGL con il rendering software (niente filtri: uno stallo
  vero si deve vedere). Codice di uscita 1 se la console ha errori o warning o se una vista non si apre.
