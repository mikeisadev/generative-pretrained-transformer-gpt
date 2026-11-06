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
| `layout.js` | `createTabLayout(root, {code, charts, player, hint, background})` | restituisce `{root, sidebar, stage, caption, host, director, nav, codePanel, charts, player}` (`nav`: vedi "Navigazione libera") |
| `ui/sidebar.js` | `Sidebar` | `sidebar.addPane({id, title, hint}) → HTMLElement`, `sidebar.select(id)`, `sidebar.onSelect(fn)` |
| `stepKeys.js` | `installStepKeys({root, ui, menu, pathPane, spaceIgnore})` | ↑ ↓ passo del Percorso (uno per pressione), Spazio pausa/riprendi; ascolto sulla finestra, solo con il tab visibile, mai nei campi, nelle finestre di dialogo, sul pad o con l'aiuto aperto; negli altri pannelli della sidebar le frecce li scorrono |
| `ui/menu.js` | `Menu`, `mathBox(items, vars)` | fisarmonica di passi `{id, title, children:[{id, title, text(HTML), math:[{tex, note}], extra:(item)=>Node}]}`; opzioni `{vars, onSelect(item, title), tabId}` (con `tabId`: progressi e "Copia link a questo passo"); `menu.select(id)`, `menu.reveal(id, {sidebar})`, `menu.move(±1)`, `menu.current`, `menu.items` (vedi "Ricerca globale") |
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
| Database vettoriale (RAG) | `tabs/rag/` | pipeline L17-L18 a stazioni: upload, estrazione, pulizia, chunking, embedding, `vector_store`, indice HNSW; poi ricerca HNSW, prompt, LLM, streaming. Azioni "Indicizzazione" e "Ricerca". Link: `action:indicizzazione|ricerca`, `view:detail|blocks|macro`, oltre a `#rag/q=<domanda>` |
| Immagini e video | `tabs/diffusion/` | DDPM/DDIM, Stable Diffusion (text encoder, VAE, U-Net, scheduler), DiT, video (VAE 3D, patch spazio-temporali). Azioni di training, generazione, VAE, denoiser, video. `sceneNav.js`: "Centra", confini per livello e didascalie delle due parti (immagini sopra, video sotto). Link: `action:forward|generate|vae|inside|video`, `view:unet|dit|pipeline|video|macro|detail` |
| Audio e musica | `tabs/audio/` | forma d'onda, STFT, Mel, codec neurale con RVQ, generazione di token musicali (delay pattern), Whisper. Laboratorio con suono vero. `catalog.js` (solo dati): nomi e descrizioni di azioni e viste. Limiti di rotazione (`setOrbitLimits`). Link: `action:audio-sound|audio-rvq|audio-generate|audio-whisper`, `view:overview|delayCore|rvq|whisperAll|detail|blocks|macro` |
| LLM moderni | `tabs/modern-llm/` | 16 meccanismi (RMSNorm, RoPE, GQA, MoE, KV cache, FlashAttention, BPE, decodifica speculativa, scaling, post-training, ragionamento, agenti, coding, multimodale, mesh 3D, interpretabilità) con azioni animate. Etichetta "Dove sei" (`regionCompass.js`) mentre si passeggia tra le regioni; moduli di soli dati `regionData.js`, `actionCatalog.js`, `palette.js` (usati anche da `search.js`). Limiti di rotazione (`setOrbitLimits`). Link: `action:<id>` (18), `view:macro|blocks|detail|<regione>` |

Ogni tab ha `window.__<nome>Tab` (es. `__ragTab`, `__diffusionTab`, `__audioTab`, `__modernLlmTab`;
per il Transformer `__gptApp`) per la console e per gli script di verifica.

## Test

- Python: `pytest` (dalla radice).
- JavaScript: `node --test webapp/tests/*.test.mjs` (moduli puri senza DOM: motore GPT, libreria RAG, …).
- Visivo: `python scripts/serve_webapp.py --no-export --port 8765` poi
  `NODE_PATH=$(npm root -g) node webapp/tests/screenshot.mjs http://127.0.0.1:8765/ <cartella> [tab1,tab2,…] [--lod] [--nav] [--mobile]`
  (senza elenco: tutti i tab; `--lod` aggiunge uno screenshot per livello dello zoom, `--nav` uno dopo
  uno spostamento con D e W tenuti premuti e uno dopo C (deve tornare alla vista iniziale), `--mobile`
  usa 390×844; stampa gli errori della console e il codice di uscita è 1 se ce ne sono).
