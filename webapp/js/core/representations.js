/**
 * Rappresentazioni di un tab: "Architettura" (la scena del tab) e, se il tab le dichiara, "Rete neurale"
 * e "Cervello". Lo stesso sistema visto in tre modi: come è costruito, come neuroni disposti da sinistra
 * a destra (con la profondità: token, griglia, esperti…), e quale parte del cervello lo ha ispirato.
 *
 *   const reps = installRepresentations(ui, {
 *     tabId: "rag", app,
 *     network: { load: () => import("./network.js").then((m) => m.networkSpec) },  // vista del nucleo
 *     //   oppure { enter(options), exit(), focus?(target) }: vista del tab sulla sua scena (Transformer)
 *     brain: { load: () => import("./brain.js").then((m) => m.brainSpec) },
 *     onChange: (id, previous) => {},
 *   });
 *   await reps.show("brain", { focus: ["hippocampus"] });   // → true quando la vista è pronta
 *   reps.current;  reps.has("network");  reps.on("change", fn);  reps.goto({ kind, id });  reps.sync(id)
 *   await reps.refresh("network");   // specifica di nuovo da load() e, se è aperta, vista rifatta (stessa camera)
 *
 * La specifica di `load` resta in cache per tutta la pagina; con `cache: false` load() si richiama a ogni
 * apertura del livello (la rete segue lo stato del tab: la domanda del RAG, il brano dell'Audio).
 *
 * Una vista con `load` vive in un livello sovrapposto allo stage (.rep-layer) con la sua pila di scena
 * (createSceneStack: SceneHost, CameraDirector, Navigation e NavPad, quindi navigazione libera identica).
 * Mentre è attiva la scena del tab è ferma e nascosta, la sua navigazione spenta, i suoi comandi nascosti
 * (classe .rep-alt-active sulla radice) e l'azione in corso fermata; tornando all'Architettura tutto
 * torna com'era, camera compresa. Un solo livello vivo per tab: lasciandolo, o nascondendo il tab, il
 * renderer si libera (dispose subito, forceContextLoss appena la GPU ha finito: vedi RELEASE_DELAY_MS);
 * modulo e specifica restano in memoria, così riaprirlo è rapido, e la camera torna dov'era.
 *
 * Ritorno automatico all'Architettura quando lo studente chiede qualcosa della scena del tab mentre un
 * livello la copre: un'azione (player.start) o un volo della camera del tab chiesto da un suo gesto
 * (director.focus/flyTo entro la finestra di director.noteUserIntent: un passo del Percorso, una vista,
 * un pulsante della sidebar, ↑ ↓). I voli non chiesti dallo studente si scartano: al ritorno niente
 * movimenti a sorpresa. I gesti sul livello, sul selettore, su ☰ e sulle schede della sidebar non
 * contano come richieste per la scena del tab.
 *
 * Il nucleo prende in carico start/stop dello SceneHost del tab (il tab continua a chiamarli in
 * activate/deactivate come sempre): mentre è attiva un'alternativa, host.start() riapre il livello e
 * lascia ferma la scena del tab; host.stop() libera anche il livello.
 */
import { el } from "./dom.js";
import { createSceneStack } from "./layout.js";
import { BRAIN_PANE_ID, REP_IDS, REP_LINKS, repOfLink } from "./repLinks.js";
import { parseHash } from "./search.js";
import { ALT_ACTIVE_CLASS, RepSwitch } from "./ui/repSwitch.js";
import { showToast } from "./ui/share.js";

// le parti pure (id, link, pannello) stanno in repLinks.js, importabile anche dai test in Node
export { BRAIN_PANE_ID, REP_IDS, REP_LINKS };

const DEFAULTS = {
  main: {
    label: "Architettura",
    icon: "architecture",
    description: "La scena di questo tab: com'è costruito il sistema, blocco per blocco, con il codice e le azioni.",
  },
  network: {
    label: "Rete neurale",
    icon: "network",
    description: "Neuroni e collegamenti in fila da sinistra a destra. Girando la camera vedi lo spessore della rete: i token, la griglia o gli esperti su cui lavora.",
  },
  brain: {
    label: "Cervello",
    icon: "brain",
    description: "Il cervello in 3D: le regioni, i percorsi e i neuroni a cui si ispira questa architettura, con i limiti dell'analogia.",
  },
};

/** Nome breve della vista nella foto della scena (scena-<tab>-rete-….png). */
const PHOTO_SLUG = { network: "rete", brain: "cervello" };

/** Moduli del nucleo che implementano le viste (caricati alla prima apertura). */
const MODULES = {
  network: () => import("./neuralNet/view.js").then((m) => m.createNetworkView),
  brain: () => import("./brain/view.js").then((m) => m.createBrainView),
  pane: () => import("./brain/pane.js").then((m) => m.renderBrainPane),
};

/** Gesti che riguardano l'alternativa o il layout, non la scena del tab (vedi "ritorno automatico"). */
const NOT_SCENE_REQUESTS = ".rep-layer, .rep-switch, .sidebar-toggle, .sidebar-nav";

/**
 * Attesa prima di perdere il contesto WebGL di un livello chiuso (il renderer è già liberato con
 * dispose(), il canvas tolto). Subito dopo l'ultimo frame loseContext() blocca la pagina finché la GPU
 * non ha finito il lavoro rimasto: misurato con il rendering software, su una scena con InstancedMesh e
 * colori per istanza, 600-940 ms subito, 180-320 ms dopo 0,4 s, 3-10 ms dopo 2-3 s. Così il ritorno
 * all'Architettura è immediato e il contesto se ne va poco dopo senza scatti. Se nel frattempo si apre
 * un altro livello, la perdita avviene prima di crearlo (al più un livello vivo per tab).
 */
const RELEASE_DELAY_MS = 2500;

const BRAIN_HINT = "Analogie tra questa architettura e il cervello: cosa hanno in comune, dove l'analogia non regge e le fonti. Il modello 3D è facoltativo.";

/**
 * @param {ReturnType<import("./layout.js").createTabLayout>} ui layout del tab
 * @param {object} [opts]
 * @param {string} [opts.tabId] id del tab (predefinito: data-tab della radice)
 * @param {object} [opts.app] l'oggetto app del guscio (passato alle viste e al pannello Cervello)
 * @param {{label?: string, description?: string}} [opts.main] la scena del tab
 * @param {{load?: () => Promise<object>, cache?: boolean, enter?: Function, exit?: Function, focus?: Function, label?: string, description?: string}} [opts.network]
 *   cache: false = load() a ogni apertura del livello (predefinito true: la specifica si carica una volta)
 * @param {{load?: () => Promise<object>, cache?: boolean, enter?: Function, exit?: Function, focus?: Function, label?: string, description?: string}} [opts.brain]
 * @param {(id: string, previous: string) => void} [opts.onChange]
 * @param {{network?: Function, brain?: Function, pane?: Function}} [opts.loaders] solo per le pagine di
 *   prova: sostituiscono i moduli veri (ognuna → Promise della funzione create…View / renderBrainPane)
 */
export function installRepresentations(ui, opts = {}) {
  const { root, stage, sidebar, host, director, nav } = ui;
  const player = ui.player ?? null;
  const tabId = opts.tabId ?? root.dataset.tab ?? "tab";
  const app = opts.app ?? null;
  const loaders = { ...MODULES, ...(opts.loaders ?? {}) };
  const narrow = window.matchMedia("(max-width: 900px)");

  const reps = new Map();
  reps.set("main", { id: "main", kind: "main", ...DEFAULTS.main, ...labels(opts.main) });
  for (const id of ["network", "brain"]) {
    const o = opts[id];
    if (!o) continue;
    const custom = typeof o.enter === "function";
    if (!custom && typeof o.load !== "function") throw new TypeError(`installRepresentations: "${id}" vuole load() oppure enter()/exit()`);
    reps.set(id, { id, kind: custom ? "custom" : "layer", ...DEFAULTS[id], ...labels(o), load: o.load, cache: o.cache !== false, enter: o.enter, exit: o.exit, focus: o.focus });
  }

  const listeners = new Set(typeof opts.onChange === "function" ? [opts.onChange] : []);
  const cache = new Map(); // chiave → Promise: moduli e specifiche si caricano una volta sola
  const versions = new Map(); // id → numero della specifica (refresh): un caricamento superato si rifà
  const memory = new Map(); // id → { camera, data }: ciò che sopravvive alla chiusura del livello
  let current = "main";
  let live = null; // il livello alternativo vivo: { id, element, stack, view, ready, shown, closed }
  let suspended = null; // la scena del tab ferma sotto un livello: { navEnabled, position, target }
  let visible = host.running; // il tab è visibile: lo dicono le sue chiamate a host.start/stop
  let seq = 0; // ogni show() ha un numero: una richiesta superata da una più recente non tocca niente
  let consumedStamp = -1; // ultimo gesto dello studente già "speso" per l'alternativa (director.lastIntent)
  let consumedHash = location.hash; // indirizzo al momento dell'ultima richiesta di una rappresentazione
  let disposed = false;
  let brainPane = null;
  let pendingRelease = null; // renderer di un livello chiuso il cui contesto si perde tra poco: { renderer, timer }

  // ---------------------------------------------------------------- prese in carico (annullate da dispose)
  const unwraps = [
    // Il tab chiama host.start() in activate e host.stop() in deactivate: sono il segnale "tab visibile".
    wrap(host, "start", (start) =>
      function startTabOrLayer(...args) {
        visible = true;
        if (suspended) return void ensureLive(); // la scena del tab resta ferma: si riapre il livello
        return start.apply(this, args);
      },
    ),
    wrap(host, "stop", (stop) =>
      function stopTabAndLayer(...args) {
        visible = false;
        const result = stop.apply(this, args);
        closeLive(); // tab nascosto: il contesto WebGL del livello si libera, si ricrea al ritorno
        return result;
      },
    ),
    wrap(director, "focus", (focus) => function focusOrReturn(...args) {
      return guardFlight(() => focus.apply(this, args));
    }),
    wrap(director, "flyTo", (flyTo) => function flyToOrReturn(...args) {
      return guardFlight(() => flyTo.apply(this, args));
    }),
  ];
  if (player) {
    unwraps.push(
      wrap(player, "start", (start) =>
        function startInArchitecture(...args) {
          if (suspended) returnToMain(); // un'azione si guarda nella scena del tab
          return start.apply(this, args);
        },
      ),
    );
  }
  const original = { start: unwrapped(host, "start"), stop: unwrapped(host, "stop") };

  // ---------------------------------------------------------------- ritorno automatico
  /** Un volo della camera del tab mentre un livello copre la sua scena. */
  function guardFlight(fly) {
    if (!suspended) return fly();
    if (!studentAsked()) return Promise.resolve(); // non chiesto dallo studente: nessun movimento, né ora né al ritorno
    returnToMain();
    return fly();
  }

  /**
   * true se il volo l'ha chiesto lo studente: un suo gesto recente nel tab che non riguardava
   * l'alternativa, oppure un link diretto o un risultato della ricerca verso questo tab (app.goto
   * scrive #<tab>/go=… nell'indirizzo prima di chiamare il goto del tab, e il clic sta nella palette,
   * fuori dal tab). Così il ritorno funziona anche in un goto che non passa da reps.goto.
   */
  function studentAsked() {
    if (director.intentActive && director.lastIntent !== consumedStamp) return true;
    if (location.hash === consumedHash) return false;
    const { tab, go } = parseHash(location.hash);
    return tab === tabId && Boolean(go);
  }

  function consumeGesture() {
    consumedStamp = director.lastIntent;
    consumedHash = location.hash;
  }

  // La navigazione del tab segna ogni clic nella radice come "richiesta dello studente" (fase di cattura,
  // registrata prima di noi): qui si spendono i gesti sul livello, sul selettore, su ☰ e sulle schede
  // della sidebar. ↑ ↓ restano richieste (passo del Percorso), tranne sul selettore che le usa per sé.
  const onGesture = (e) => {
    if (!suspended) return;
    const target = e.target instanceof Element ? e.target : null;
    if (e.type === "keydown" && (e.key === "ArrowUp" || e.key === "ArrowDown") && !target?.closest(".rep-switch")) return;
    if (target?.closest(NOT_SCENE_REQUESTS)) consumeGesture();
  };
  // ↑ ↓ con il focus sulla pagina (non dentro il tab) arrivano al Percorso senza passare dalla radice.
  const onWindowKey = (e) => {
    if (!suspended || !visible || e.defaultPrevented || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if ((e.key === "ArrowUp" || e.key === "ArrowDown") && (e.target === document.body || e.target === document.documentElement)) director.noteUserIntent();
  };
  root.addEventListener("pointerup", onGesture, true);
  root.addEventListener("keydown", onGesture, true);
  window.addEventListener("keydown", onWindowKey, true);

  // ---------------------------------------------------------------- selettore e pannello Cervello
  const switcher =
    reps.size > 1
      ? new RepSwitch({
          root,
          stage,
          sidebar: sidebar?.root ?? null,
          options: [...reps.values()].map(({ id, label, description, icon }) => ({ id, label, description, icon })),
          current,
          onPick: (id) => {
            if (narrow.matches) root.classList.add("sidebar-collapsed"); // scelto dal cassetto: si guarda la scena
            void show(id);
          },
        })
      : null;
  if (reps.get("brain")?.load && sidebar) brainPane = installBrainPane();

  // ---------------------------------------------------------------- API
  const api = {
    /** "main" | "network" | "brain": la rappresentazione mostrata (o in preparazione). */
    get current() {
      return current;
    },
    /** La vista viva del livello alternativo (per i test e la console), o null. */
    get view() {
      return live?.view ?? null;
    },
    /** La pila di scena del livello vivo { host, director, nav } (per i test e la console), o null. */
    get stack() {
      return live?.stack ?? null;
    },
    has: (id) => reps.has(id),
    show,
    sync,
    goto,
    refresh,
    /** type "change": fn(id, previous) a ogni cambio. Restituisce la funzione di rimozione. */
    on(type, fn) {
      if (type !== "change" || typeof fn !== "function") return () => {};
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    dispose,
  };
  ui.reps = api;
  return api;

  // ---------------------------------------------------------------- transizioni
  /**
   * Mostra una rappresentazione. Si risolve con true quando la vista è pronta (anche subito, se il tab
   * è nascosto: il livello si apre quando torna visibile), con false se la richiesta è stata superata da
   * un'altra o se la vista non si è caricata (in quel caso c'è un avviso e si torna all'Architettura).
   * @param {string} id "main" | "network" | "brain"
   * @param {{focus?: any}} [options] `focus` passa alla vista (es. id delle regioni del cervello)
   */
  async function show(id, options = {}) {
    const rep = reps.get(id);
    if (!rep || disposed) return false;
    const token = ++seq;
    consumeGesture(); // il gesto che ha chiesto questa vista non vale come richiesta per la scena del tab
    if (id === current) return reveal(rep, options, token);
    const previous = reps.get(current);
    if (previous.kind === "custom") {
      setCurrent(id); // subito: il selettore risponde al clic anche se exit() richiede tempo
      try {
        await previous.exit?.();
      } catch (err) {
        report(previous, err, { fallback: false });
      }
      if (token !== seq || disposed) return false;
    }
    if (rep.kind === "layer") {
      closeLive();
      suspendTab();
      setCurrent(id);
      if (!visible) return true;
      const entry = buildLive(rep, options);
      return (await entry.ready) && token === seq && entry === live;
    }
    // l'Architettura o una vista del tab sulla sua scena: la scena del tab torna visibile
    closeLive();
    resumeTab();
    setCurrent(id);
    if (rep.kind !== "custom") return true;
    try {
      await rep.enter?.(options);
      return token === seq;
    } catch (err) {
      report(rep, err);
      if (token === seq && current === id) setCurrent("main");
      return false;
    }
  }

  /** La rappresentazione chiesta è già quella attuale: aspetta che sia pronta e passa il focus. */
  async function reveal(rep, options, token) {
    if (rep.kind === "custom") {
      if (options.focus !== undefined) await rep.focus?.(options.focus);
      return true;
    }
    if (rep.kind !== "layer" || !visible) return true;
    const entry = live ?? buildLive(rep, options);
    const ok = await entry.ready;
    if (!ok || token !== seq || entry !== live) return false;
    if (options.focus !== undefined && entry.focusApplied !== options.focus) entry.view.focus?.(options.focus);
    return true;
  }

  /**
   * Il tab ha cambiato rappresentazione da solo (vista personalizzata: es. il pannello Viste del
   * Transformer): il selettore si aggiorna senza chiamare enter/exit. Un livello aperto si chiude.
   */
  function sync(id) {
    const rep = reps.get(id);
    if (!rep || rep.kind === "layer" || disposed) return;
    closeLive();
    resumeTab();
    setCurrent(id);
  }

  /**
   * Link diretti e ricerca: view:rete e view:cervello mostrano la rappresentazione (→ Promise<true>);
   * ogni altro elemento (passo, azione, vista) sta nella scena del tab, quindi un livello aperto si
   * chiude subito e si restituisce null: il goto del tab prosegue come sempre.
   *   const shown = reps.goto(target); if (shown) return shown;
   */
  function goto({ kind, id } = {}) {
    const repId = repOfLink({ kind, id });
    if (repId && reps.has(repId)) {
      if (narrow.matches) root.classList.add("sidebar-collapsed"); // sul telefono conta la scena
      return show(repId).then(() => true); // un errore di caricamento ha già il suo avviso
    }
    returnToMain();
    return null;
  }

  /** Torna subito alla scena del tab (senza attese: chi chiede un volo o un'azione prosegue). */
  function returnToMain() {
    if (!suspended && !live) return;
    closeLive();
    resumeTab();
    setCurrent("main");
  }

  function setCurrent(id) {
    if (id === current) return;
    const previous = current;
    current = id;
    switcher?.setCurrent(id);
    for (const fn of [...listeners]) safely(() => fn(id, previous), "onChange");
  }

  // ---------------------------------------------------------------- scena del tab
  /** La scena del tab si ferma sotto il livello: niente rendering, niente navigazione, niente azione. */
  function suspendTab() {
    if (suspended) return;
    player?.stop(); // l'azione in corso si chiude (onStop riporta la scena a riposo, voli compresi)
    finishFlight(); // un volo in corso arriva subito alla meta: è lì che lo studente tornerà
    settleControls(host.controls); // niente inerzia residua: al ritorno la camera è ferma
    suspended = { navEnabled: nav.enabled, position: host.camera.position.clone(), target: host.controls.target.clone() };
    nav.setEnabled(false);
    original.stop.call(host); // ferma, ma il tab resta "visibile" (non è un deactivate)
    root.classList.add(ALT_ACTIVE_CLASS);
  }

  function resumeTab() {
    if (!suspended) return;
    const saved = suspended;
    suspended = null;
    root.classList.remove(ALT_ACTIVE_CLASS);
    host.camera.position.copy(saved.position);
    host.controls.target.copy(saved.target);
    nav.setEnabled(saved.navEnabled);
    if (visible) {
      original.start.call(host);
      host.resize();
    }
  }

  function finishFlight() {
    director.finish();
  }

  // ---------------------------------------------------------------- livello alternativo
  /** Riapre il livello della rappresentazione attuale (tab tornato visibile). */
  function ensureLive() {
    const rep = reps.get(current);
    if (live || !visible || rep?.kind !== "layer") return live;
    return buildLive(rep, {});
  }

  /**
   * Crea il livello: subito l'elemento con "Preparo la scena…", poi (modulo e specifica dalla cache)
   * la pila di scena e la vista. `entry.ready` si risolve con true a vista pronta, false se nel frattempo
   * il livello è stato chiuso o la vista non si è caricata.
   */
  function buildLive(rep, { focus } = {}) {
    const element = el(
      "div",
      { className: "rep-layer", role: "region", ariaLabel: `${rep.label} (rappresentazione alternativa)`, dataset: { rep: rep.id, tab: `${tabId}-${PHOTO_SLUG[rep.id] ?? rep.id}` } },
      loadingCard(rep),
    );
    stage.append(element);
    const entry = { id: rep.id, element, stack: null, view: null, ready: null, shown: false, closed: false, focusApplied: undefined };
    live = entry;
    switcher?.setBusy(rep.id);
    entry.ready = (async () => {
      try {
        let version = specVersion(rep.id);
        let [create, spec] = await Promise.all([once(`module:${rep.id}`, loaders[rep.id]), specFor(rep)]);
        // un refresh arrivato mentre la specifica si caricava: si legge quella nuova
        while (!entry.closed && version !== specVersion(rep.id)) {
          version = specVersion(rep.id);
          spec = await specFor(rep);
        }
        if (entry.closed) return false;
        if (typeof create !== "function") throw new TypeError(`il modulo della vista "${rep.id}" non esporta la funzione create…View`);
        releaseNow(); // al più un livello vivo: il contesto del livello precedente se ne va prima
        entry.stack = createSceneStack(element, { root: element });
        const { host: layerHost, director: layerDirector, nav: layerNav } = entry.stack;
        // come per la scena del tab: sul telefono un tocco sulla scena chiude il cassetto della sidebar
        layerHost.renderer.domElement.addEventListener("pointerdown", () => narrow.matches && root.classList.add("sidebar-collapsed"));
        const mem = memoryFor(rep.id);
        entry.view = create({ host: layerHost, director: layerDirector, nav: layerNav, layer: element, tabId, spec, app, reps: api, memory: mem.data });
        if (!entry.view || typeof entry.view.dispose !== "function") throw new TypeError(`la vista "${rep.id}" non restituisce { ready, dispose }`);
        layerHost.start();
        layerHost.resize();
        await entry.view.ready;
        if (entry.closed) return false;
        element.querySelector(":scope > .rep-loading")?.remove();
        entry.view.onShow?.();
        entry.shown = true;
        switcher?.scheduleLayout(); // la scheda della vista c'è adesso: il selettore non la deve coprire
        if (focus !== undefined) {
          entry.focusApplied = focus;
          entry.view.focus?.(focus);
        } else if (mem.camera) restoreCamera(entry.stack, mem.camera); // riaperta: la camera torna dov'era
        return true;
      } catch (err) {
        if (entry.closed) return false;
        report(rep, err);
        return false;
      } finally {
        if (live === entry) switcher?.setBusy(null);
      }
    })();
    return entry;
  }

  /** Chiude il livello vivo e libera tutto: vista, navigazione, renderer e il suo contesto WebGL. */
  function closeLive() {
    const entry = live;
    if (!entry) return;
    live = null;
    entry.closed = true;
    switcher?.setBusy(null);
    const hadFocus = entry.element.contains(document.activeElement);
    if (entry.stack && entry.shown) memoryFor(entry.id).camera = cameraOf(entry.stack);
    if (entry.shown) safely(() => entry.view.onHide?.(), "onHide");
    if (entry.view) safely(() => entry.view.dispose(), "dispose");
    if (entry.stack) {
      const { host: layerHost, director: layerDirector, nav: layerNav } = entry.stack;
      safely(() => layerNav.dispose(), "nav.dispose"); // toglie i listener, il pad, l'aiuto
      safely(() => layerHost.dispose(), "host.dispose"); // ferma il ciclo, renderer.dispose(), toglie il canvas
      releaseLater(layerHost.renderer); // il contesto WebGL non resta in vita (vedi RELEASE_DELAY_MS)
      layerDirector.highlight.geometry.dispose();
      layerDirector.highlight.material.dispose();
    }
    entry.element.remove();
    if (hadFocus) switcher?.focusCurrent();
  }

  /** Perde il contesto WebGL di un livello chiuso fra RELEASE_DELAY_MS (canvas già tolto, ciclo fermo). */
  function releaseLater(renderer) {
    releaseNow();
    pendingRelease = { renderer, timer: setTimeout(releaseNow, RELEASE_DELAY_MS) };
  }

  /** Perde subito il contesto in attesa, se c'è (nuovo livello in arrivo, pulizia finale). */
  function releaseNow() {
    const pending = pendingRelease;
    if (!pending) return;
    pendingRelease = null;
    clearTimeout(pending.timer);
    safely(() => pending.renderer.forceContextLoss(), "forceContextLoss");
  }

  function memoryFor(id) {
    if (!memory.has(id)) memory.set(id, { camera: null, data: {} });
    return memory.get(id);
  }

  /**
   * La specifica di `id` va riletta (lo stato del tab è cambiato: un altro brano, un altro run): esce
   * dalla cache e, se il livello di `id` è aperto, la vista si rifà sulla STESSA pila di scena (stesso
   * renderer: nessun contesto WebGL nuovo né perso, quindi nessuno scatto) con la camera di prima.
   * → Promise<boolean>: true a vista rifatta, o subito se il livello non è aperto (alla prossima apertura
   * la specifica sarà quella nuova); false se superata da un altro refresh o se il livello si chiude.
   */
  async function refresh(id) {
    const rep = reps.get(id);
    if (!rep || rep.kind !== "layer" || disposed) return false;
    cache.delete(`spec:${id}`);
    versions.set(id, specVersion(id) + 1);
    const entry = live;
    if (!entry || entry.id !== id) return true;
    if (!entry.stack) return entry.ready; // ancora in caricamento: buildLive legge la specifica nuova
    return rebuildView(entry, rep);
  }

  /** Rifà la vista del livello aperto con la specifica di adesso, tenendo pila, camera e "Centra". */
  async function rebuildView(entry, rep) {
    const version = specVersion(rep.id);
    const { host: layerHost, director: layerDirector, nav: layerNav } = entry.stack;
    const saved = cameraOf(entry.stack);
    const busy = loadingCard(rep, "Aggiorno la scena…");
    entry.element.append(busy);
    switcher?.setBusy(rep.id);
    try {
      const [create, spec] = await Promise.all([once(`module:${rep.id}`, loaders[rep.id]), specFor(rep)]);
      if (entry.closed || entry !== live || version !== specVersion(rep.id)) return false; // un refresh più recente fa il suo
      if (entry.shown) safely(() => entry.view.onHide?.(), "onHide");
      safely(() => entry.view.dispose(), "dispose"); // scheda, scena, listener; rimette home e confini della navigazione
      entry.shown = false;
      layerDirector.cancel();
      entry.view = create({ host: layerHost, director: layerDirector, nav: layerNav, layer: entry.element, tabId, spec, app, reps: api, memory: memoryFor(rep.id).data });
      await entry.view.ready;
      if (entry.closed || entry !== live) return false;
      entry.view.onShow?.();
      entry.shown = true;
      switcher?.scheduleLayout();
      restoreCamera(entry.stack, saved);
      return version === specVersion(rep.id);
    } catch (err) {
      if (!entry.closed) report(rep, err);
      return false;
    } finally {
      busy.remove();
      if (live === entry) switcher?.setBusy(null);
    }
  }

  /** Numero della specifica di una rappresentazione: cresce a ogni refresh. */
  function specVersion(id) {
    return versions.get(id) ?? 0;
  }

  /** La specifica di una rappresentazione: dalla cache, o di nuovo da load() con `cache: false`. */
  function specFor(rep) {
    return rep.cache ? once(`spec:${rep.id}`, rep.load) : Promise.resolve().then(rep.load);
  }

  /** Carica una volta sola (moduli, specifiche); un errore si può riprovare alla richiesta successiva. */
  function once(key, load) {
    if (!cache.has(key)) {
      const promise = Promise.resolve().then(load);
      promise.catch(() => cache.get(key) === promise && cache.delete(key));
      cache.set(key, promise);
    }
    return cache.get(key);
  }

  /** Avviso leggibile e, se serve, ritorno all'Architettura. */
  function report(rep, err, { fallback = true } = {}) {
    console.warn(`[rappresentazioni] «${rep.label}» non disponibile:`, err);
    const main = reps.get("main").label;
    showToast(fallback ? `Non è stato possibile aprire «${rep.label}»: torno a «${main}».` : `«${rep.label}» non si è chiusa del tutto: ricarica la pagina se la scena sembra strana.`, { duration: 5200 });
    if (fallback && current === rep.id) returnToMain();
  }

  // ---------------------------------------------------------------- pannello "Cervello"
  /**
   * Il pannello esiste subito (lo studente lo vede tra le schede); il contenuto si carica la prima
   * volta che serve, con renderBrainPane di core/brain/pane.js (solo DOM, nessun three.js).
   */
  function installBrainPane() {
    const pane = sidebar.addPane({ id: BRAIN_PANE_ID, title: "Cervello", hint: BRAIN_HINT });
    pane.classList.add("rep-brain-pane");
    pane.append(el("p", { className: "hint" }, "Le analogie con il cervello si caricano quando apri questo pannello."));
    let filling = null;
    const fill = () => {
      filling ??= (async () => {
        pane.replaceChildren(el("p", { className: "rep-pane-status", role: "status" }, "Carico le analogie con il cervello…"));
        const [render, spec] = await Promise.all([once("module:pane", loaders.pane), specFor(reps.get("brain"))]);
        if (disposed) return;
        pane.replaceChildren();
        render(pane, spec, { onShow3D: (focus) => showFromPane(focus), app });
      })().catch((err) => {
        filling = null; // "Riprova" ricomincia da capo
        console.warn("[rappresentazioni] pannello Cervello non disponibile:", err);
        pane.replaceChildren(
          el("p", { className: "rep-pane-status rep-pane-error", role: "status" }, "Non è stato possibile caricare le analogie con il cervello."),
          el("button", { type: "button", onclick: () => fill() }, "Riprova"),
        );
      });
      return filling;
    };
    const onSelect = (id) => !disposed && id === BRAIN_PANE_ID && fill();
    sidebar.onSelect(onSelect);
    if (!pane.hidden) void fill(); // unico pannello della sidebar: è già quello aperto
    return { pane, fill, onSelect };
  }

  /** "Mostra il cervello in 3D" / "Mostra nel modello 3D" dal pannello. */
  function showFromPane(focus) {
    if (narrow.matches) root.classList.add("sidebar-collapsed"); // sul telefono il cassetto coprirebbe la scena
    const regions = Array.isArray(focus) ? focus.filter((r) => typeof r === "string") : [];
    return show("brain", regions.length ? { focus: regions } : {});
  }

  // ---------------------------------------------------------------- pulizia
  function dispose() {
    if (disposed) return;
    closeLive();
    releaseNow();
    resumeTab();
    disposed = true;
    for (const undo of unwraps) undo();
    root.removeEventListener("pointerup", onGesture, true);
    root.removeEventListener("keydown", onGesture, true);
    window.removeEventListener("keydown", onWindowKey, true);
    switcher?.dispose();
    if (brainPane) {
      sidebar.nav.querySelector(`:scope > [data-pane="${BRAIN_PANE_ID}"]`)?.remove();
      brainPane.pane.remove();
      sidebar.panes?.delete?.(BRAIN_PANE_ID);
      sidebar.listeners?.delete?.(brainPane.onSelect); // Sidebar.onSelect non restituisce la funzione di rimozione
    }
    listeners.clear();
    if (ui.reps === api) delete ui.reps;
  }
}

// ---------------------------------------------------------------- utilità
/** Solo etichetta e descrizione: il resto delle opzioni di una rappresentazione ha un significato preciso. */
function labels(o = {}) {
  const out = {};
  if (typeof o.label === "string" && o.label) out.label = o.label;
  if (typeof o.description === "string" && o.description) out.description = o.description;
  return out;
}

/**
 * Sostituisce un metodo dell'istanza con `make(originale)`. Restituisce la funzione che lo rimette:
 * se nel frattempo qualcun altro l'ha avvolto di nuovo, non tocca niente (non si rompe la sua catena).
 */
function wrap(object, name, make) {
  const own = Object.prototype.hasOwnProperty.call(object, name);
  const before = object[name];
  const wrapper = make(before);
  wrapper.unwrapped = before;
  object[name] = wrapper;
  return () => {
    if (object[name] !== wrapper) return;
    if (own) object[name] = before;
    else delete object[name];
  };
}

/** Il metodo originale sotto un nostro wrap (per usarlo senza passare dal wrapper). */
function unwrapped(object, name) {
  return object[name].unwrapped ?? object[name];
}

/** L'inerzia del trascinamento si applica tutta adesso: la camera non "scivola" più quando la scena riparte. */
function settleControls(controls) {
  if (!controls.enableDamping) return;
  controls.enableDamping = false;
  controls.update();
  controls.enableDamping = true;
}

/**
 * La camera di una pila di scena e la sua vista iniziale fotografata dalla navigazione: riaprendo (o
 * rifacendo) la vista "Centra" deve tornare lì, non sul punto in cui lo studente aveva lasciato la camera.
 */
function cameraOf(stack) {
  const { camera, controls } = stack.host;
  return { position: camera.position.clone(), target: controls.target.clone(), home: stack.nav.getHomePose() };
}

/** Livello riaperto (o vista rifatta): camera dove lo studente l'aveva lasciata, "Centra" sulla vista iniziale di prima. */
function restoreCamera(stack, saved) {
  stack.director.cancel(); // l'inquadratura iniziale della vista ricreata non deve partire
  stack.host.camera.position.copy(saved.position);
  stack.host.controls.target.copy(saved.target);
  stack.host.controls.update();
  if (saved.home) stack.nav.setHomePose(saved.home);
}

function loadingCard(rep, text = "Preparo la scena…") {
  return el("div", { className: "rep-loading", role: "status" }, el("span", { className: "rep-loading-bar", ariaHidden: "true" }), el("strong", {}, rep.label), el("span", {}, text));
}

function safely(fn, what) {
  try {
    return fn();
  } catch (err) {
    console.warn(`[rappresentazioni] errore in ${what}:`, err);
    return undefined;
  }
}
