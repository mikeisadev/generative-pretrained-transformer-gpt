/**
 * Tab "Audio e musica": reti neurali che generano (e capiscono) audio e musica.
 *
 * Punto di raccordo; ogni parte sta in un modulo proprio:
 *   dsp.js / song.js / codec.js  calcoli veri (FFT, Mel, log-Mel di Whisper, k-means, RVQ,
 *                                delay pattern, Griffin-Lim, sintesi additiva) — moduli puri
 *   lab.js                       esegue i calcoli nel browser e tiene i risultati
 *   sceneCodec/Gen/Whisper.js    livello "Dettaglio" della scena 3D
 *   sceneLod.js                  livelli "Blocchi" e "Panoramica" + flussi animati
 *   actions.js                   azioni animate (ActionPlayer)
 *   plots.js                     grafici (ChartsPanel)
 *   labPane.js / content.js      sidebar: Laboratorio, Percorso, Fonti
 *   snippets.js / upstream.js    codice vero nel pannello di destra
 *   audioOut.js                  riproduzione con Web Audio
 *   catalog.js / search.js       nomi di azioni e viste; indice della ricerca globale (solo dati)
 *   network.js / brain.js        rappresentazioni "Rete neurale" (con i numeri veri del laboratorio) e
 *                                "Cervello" (soli dati, viste del nucleo: core/representations.js)
 *
 * Navigazione libera (ui.nav, dal nucleo): "Centra" = "Vista d'insieme", confini = tutta la pipeline.
 * Link diretti #audio/go=<kind>:<id> (ricerca globale) arrivano a goto():
 *   view:rete|cervello (rappresentazioni del nucleo), step:<id del Percorso>,
 *   action:audio-sound|audio-rvq|audio-generate|audio-whisper,
 *   view:overview|delayCore|rvq|whisperAll (pannello Viste) oppure detail|blocks|macro (livelli dello zoom).
 */
import * as THREE from "three";
import { fetchJson } from "../../core/binary.js";
import { el, fmt } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { installStepKeys } from "../../core/stepKeys.js";
import { LODController } from "../../core/lod.js";
import { installRepresentations } from "../../core/representations.js";
import { Menu } from "../../core/ui/menu.js";
import { installActions } from "./actions.js";
import { AudioOut } from "./audioOut.js";
import { ACTIONS, ACTION_CATEGORIES, LEVELS, VIEWS } from "./catalog.js";
import { CODEC } from "./codec.js";
import { SOURCES, STEPS } from "./content.js";
import { createLab, formatPercent } from "./lab.js";
import { buildLabPane } from "./labPane.js";
import { FOCUS, LOD_DISTANCES, SCENE_OFFSET } from "./layout3d.js";
import { createPlots } from "./plots.js";
import { buildAudio, buildEncoder, buildRvq } from "./sceneCodec.js";
import { buildConditioning, buildCrossAttention, buildDecoder, buildDelay, buildOutput, buildTransformer } from "./sceneGen.js";
import { buildBlocks, buildFlows, buildMacro } from "./sceneLod.js";
import { WHISPER_TOKENS, buildWhisper } from "./sceneWhisper.js";
import { showSnippet } from "./snippets.js";


const FRONT = new THREE.Vector3(0, 0.08, 1).normalize();
const SLIGHT = new THREE.Vector3(0.1, 0.12, 1).normalize();
const ACTION_IDS = new Set(Object.values(ACTIONS).map((a) => a.id));
const VIEW_IDS = new Set(VIEWS.map((v) => v.id));
const LEVEL_IDS = new Set(LEVELS.map((l) => l.id));
/** Telefono e tablet stretto: la sidebar è un cassetto che copre la scena (stessa soglia di core.css). */
const PHONE = "(max-width: 900px)";
/**
 * Rapporto larghezza/altezza della scena visibile su cui sono tarate le soglie del LOD (1600×900 con la
 * sidebar aperta). Su una scena più stretta (codice aperto, telefono) per inquadrare lo stesso componente
 * la camera si allontana di più: le soglie crescono nella stessa proporzione, così una vista mostra lo
 * stesso livello su ogni schermo (es. la vista d'insieme con il codice aperto resta "Blocchi").
 * La correzione si ferma a 2×: sul telefono (scena ~3 volte più stretta) il Dettaglio di mezza pipeline
 * sarebbe illeggibile e pesante da disegnare, meglio i blocchi con i titoli.
 */
const REFERENCE_ASPECT = 1.44;
const MAX_NARROWNESS = 2;
/**
 * Limiti della camera libera: distanza minima (il componente più piccolo, il decoder, riempie lo schermo
 * già a ~15) e di quanto si può girare attorno (°) o inclinare (°) rispetto alla vista frontale.
 */
const MIN_ZOOM_DISTANCE = 6;
const MAX_TURN_DEG = 60;
const MAX_TILT_DEG = 45;
/** Dispositivo senza mouse: le didascalie parlano di dita invece che di tasti. */
const TOUCH = "(pointer: coarse)";

export async function createTab(root, app) {
  const touch = window.matchMedia(TOUCH);
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint:
      "Scegli un punto del percorso o un'azione: la camera vola sul componente e a destra compare il codice. Più ti allontani, più la vista si semplifica (dettaglio → blocchi → panoramica). " +
      (touch.matches
        ? "Puoi anche muoverti da solo lungo la pipeline con due dita; il pulsante con le frecce in basso a destra apre i comandi."
        : "Puoi anche muoverti da solo lungo la pipeline: tasto destro (o Shift) e trascina, oppure A/D; C riporta alla vista d'insieme."),
  });
  const { host, director, sidebar, stage, codePanel } = ui;
  root.classList.add("audio-tab");

  // --- calcoli veri (circa 2 secondi): schermata di attesa con avanzamento
  const loading = el("div", { className: "audio-loading", role: "status" }, el("strong", {}, "Preparo il laboratorio audio"), el("p", {}, "Avvio…"), el("div", { className: "audio-progress" }, el("div")));
  stage.append(loading);
  // dimensione del vocabolario del GPT del corso: dal dato esportato, non scritta a mano
  const gptVocab = fetchJson(new URL("../../../data/weights.json", import.meta.url))
    .then((w) => w.config.vocab_size)
    .catch(() => null);
  const lab = await createLab({
    onProgress: (text, f) => {
      loading.querySelector("p").textContent = text;
      loading.querySelector(".audio-progress div").style.width = `${Math.round(f * 100)}%`;
    },
  });
  loading.remove();

  const styleIds = Object.keys(lab.songs);
  const state = { style: styleIds[0], gen: null, genOptions: { temperature: 0.9, seed: 1 } };

  // --- scena: tre livelli di dettaglio
  const sceneRoot = new THREE.Group();
  sceneRoot.position.set(SCENE_OFFSET.x, SCENE_OFFSET.y, 0);
  host.scene.add(sceneRoot);
  const grid = new THREE.GridHelper(240, 120, 0x1e293b, 0x111827);
  grid.position.y = -24;
  host.scene.add(grid);

  const parts = {
    audio: buildAudio(lab, state.style),
    encoder: buildEncoder(lab, state.style),
    rvq: buildRvq(lab, state.style, styleIds),
    cond: buildConditioning(lab, state.style),
    transformer: buildTransformer(),
    delay: buildDelay(lab),
    decoder: buildDecoder(),
    output: buildOutput(lab),
    whisper: buildWhisper(lab, state.style),
  };
  parts.cross = buildCrossAttention(parts.transformer);
  const flows = buildFlows();
  const detail = new THREE.Group();
  detail.add(...["audio", "encoder", "rvq", "cond", "transformer", "delay", "decoder", "output", "whisper"].map((k) => parts[k].group), parts.cross, flows.group);
  const blocks = buildBlocks();
  const macro = buildMacro();
  sceneRoot.add(detail, blocks.group, macro);

  // calcolato solo quando la scena cambia forma, non a ogni frame
  let narrowness = 1;
  const updateNarrowness = () => (narrowness = THREE.MathUtils.clamp(REFERENCE_ASPECT / host.visibleAspect(), 1, MAX_NARROWNESS));
  updateNarrowness();
  host.onResize(updateNarrowness);
  const lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: LOD_DISTANCES.detail, group: detail },
      { id: "blocks", label: "Blocchi", maxDistance: LOD_DISTANCES.blocks, group: blocks.group },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: macro },
    ],
    widgetParent: stage,
    distanceScale: () => narrowness,
    // Scelta a mano nel selettore: "Dettaglio" avvicina la camera al cuore del generatore,
    // gli altri livelli inquadrano tutta la pipeline (il livello resta quello scelto).
    onPick: (mode) => {
      if (mode !== "auto") tab.focus(mode === "detail" ? "genCore" : "overview", { byUser: true });
    },
  });

  const P = parts;
  // registro degli oggetti da inquadrare: rettangoli invisibili (mai disegnati) nella scena
  const focusRoot = new THREE.Group();
  sceneRoot.add(focusRoot);
  const focusMaterial = new THREE.MeshBasicMaterial({ visible: false });
  const registry = new Map(
    Object.entries(FOCUS).map(([id, [x0, x1, y0, y1]]) => {
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), focusMaterial);
      plane.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0);
      focusRoot.add(plane);
      return [id, plane];
    }),
  );
  sceneRoot.updateMatrixWorld(true); // i rettangoli servono subito (primo volo e confini), prima del primo disegno

  const plots = createPlots(ui.charts, lab);
  let pendingCode = Promise.resolve(); // pannello del codice in arrivo (vedi tab.focus)
  let stageTarget = null; // stadio dello spettrogramma scelto dal menu, raggiunto con una transizione
  const out = new AudioOut();

  const tab = {
    app,
    ui,
    lab,
    parts,
    plots,
    state,
    lod,
    /** Ultimo componente inquadrato (le azioni lo usano per non ripetere un volo inutile). */
    focusedId: null,
    /** Componente che l'azione in corso vuole inquadrare (per "Segui l'azione", vedi sotto). */
    actionFocusId: null,
    /**
     * Vola su un rettangolo del registro. Aspetta che il pannello del codice chiesto nello stesso
     * passo sia comparso (cambia la parte visibile dello stage) e, se un'azione è in corso, lascia
     * spazio in basso per la barra del player.
     * Con `byUser` il volo è una richiesta dello studente (menu, viste, "Centra", ricerca): parte anche
     * con la camera libera, anche se il codice ha impiegato un po' ad arrivare, e durante un'azione la
     * camera poi resta lì ("Camera libera · Segui l'azione") invece di essere riportata altrove dal
     * passo successivo dell'azione.
     */
    async focus(id, { byUser = false, ...options } = {}) {
      const target = registry.get(id);
      if (!target) return;
      tab.focusedId = id;
      await Promise.resolve(); // lascia a tab.code(), chiamato subito dopo, il tempo di registrarsi
      await pendingCode;
      if (byUser) {
        director.noteUserIntent();
        director.release(); // fuori da un'azione non fa niente
      }
      const box = new THREE.Box3().setFromObject(target);
      const h = box.max.y - box.min.y;
      box.max.y += h * 0.14; // i titoli non finiscono sotto la barra degli strumenti
      if (ui.player.action) box.min.y -= h * 0.42; // barra del player in basso
      if (id === "overview") return director.focus(box, { showFrame: false, margin: 1.06, direction: FRONT, ...options });
      return director.focus(box, { showFrame: false, margin: 1.15, direction: SLIGHT, ...options });
    },
    code(ref, highlight = [], title) {
      pendingCode = showSnippet(codePanel, ref, highlight, title).catch(() => {});
      return pendingCode;
    },
    setCaption(text) {
      ui.caption.textContent = text;
    },
    /** Riporta la scena allo stato di riposo (fine o interruzione di un'azione). */
    resetScene({ keepStage = false } = {}) {
      // Un'azione che parte (onStart) deve volare anche sul componente già inquadrato: lascia spazio
      // alla barra del player, e lo studente può aver spostato la camera dopo l'ultimo volo.
      tab.focusedId = null;
      P.audio.showLens(null);
      P.audio.setPlayhead(null);
      if (!keepStage) P.encoder.setStage(0);
      P.encoder.reveal(lab.frames);
      P.encoder.setSlice(null);
      P.rvq.reveal(lab.frames);
      P.rvq.setCursor(null);
      P.rvq.setLens(0);
      P.cond.highlightWord(-1);
      P.cond.glow(0);
      P.cross.userData.setStrength(0);
      P.transformer.pulse(0);
      P.decoder.pulse(0);
      P.whisper.revealMel(null);
      P.whisper.showTokens(WHISPER_TOKENS.length);
      for (const part of ["enc", "dec", "cross"]) P.whisper.pulse(part, 0);
      P.delay.show(state.gen, state.gen ? state.gen.steps.length - 1 : -1);
      plots?.specUpTo(lab.frames, true);
      plots?.errorUpTo(CODEC.levels);
      plots?.tokens(lab.codes[state.style], lab.frames);
      plots?.waveAt(null);
      director.hideFrame();
    },
    runGeneration(options = state.genOptions) {
      state.gen = lab.generate(state.style, options);
      labPane.enableGenerated(false);
      return state.gen;
    },
    async showGenerated(gen) {
      labPane.setStatus("Decodifica dei token generati…");
      const audio = await gen.audio();
      if (state.gen !== gen) return;
      P.output.setAudio(audio, "ready");
      labPane.enableGenerated(true);
      labPane.setStatus(`Brano generato (temperatura ${fmt(state.genOptions.temperature, 1)}, seme ${state.genOptions.seed}): premi «Generato».`);
    },
    async showReconstruction(q) {
      const audio = await lab.reconstruct(state.style, q);
      P.output.setAudio(audio, "rec");
      return audio;
    },
    setStyle(id) {
      if (!lab.songs[id] || id === state.style) return;
      // un'azione in corso ha didascalie e generazione costruite sul brano di prima: la chiudiamo
      ui.player.stop();
      out.stop();
      state.style = id;
      for (const k of ["audio", "encoder", "rvq", "cond", "whisper"]) P[k].setStyle(id);
      P.output.setAudio(null);
      tab.runGeneration(state.genOptions);
      P.delay.show(state.gen, state.gen.steps.length - 1);
      tab.showGenerated(state.gen);
      labPane.selectStyle(id);
      plots?.setStyle(id);
      labPane.setStatus(`Brano scelto: "${lab.songs[id].style.prompt}".`);
      // La "Rete neurale" mostra i numeri veri del brano: se è aperta si rifà subito con quelli del brano
      // nuovo, sulla stessa scena e con la stessa camera (reps.refresh); chiusa, la si costruisce
      // all'apertura (cache: false).
      if (reps.current === "network") void reps.refresh("network");
    },
  };

  // --- sidebar
  const panePath = sidebar.addPane({ id: "percorso", title: "Percorso" });
  const paneActions = sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Animazioni passo per passo: usa play/pausa, avanti/indietro e la velocità nella barra in basso." });
  const paneLab = sidebar.addPane({ id: "lab", title: "Laboratorio", hint: "Tutto calcolato qui nel browser: ascolta, cambia i codebook, genera." });
  const paneViews = sidebar.addPane({ id: "viste", title: "Viste" });
  // Rappresentazioni del nucleo: selettore "Architettura · Rete neurale · Cervello" e pannello "Cervello",
  // che deve stare prima di "Fonti". Le due viste vivono in un livello sopra lo stage; il nucleo torna da
  // solo all'Architettura quando lo studente chiede qualcosa della scena (un passo, una vista, un'azione,
  // "Genera subito", un link diretto): il tab non ha richieste che arrivino da altre strade (eventi del bus).
  const reps = installRepresentations(ui, {
    tabId: "audio",
    app,
    network: {
      // i numeri dipendono dal brano scelto nel Laboratorio: la specifica si rifà a ogni apertura (~5 ms)
      load: () => import("./network.js").then((m) => m.buildAudioNetworkSpec(lab, { style: state.style })),
      cache: false,
      description: "Quattro reti dell'audio neurone per neurone, da sinistra a destra: EnCodec, il codec del tab con i numeri veri, MusicGen e Whisper. Girando la camera vedi il tempo in profondità: convoluzioni tra frame vicini, voci dei codebook, delay pattern, attention tra i frame.",
    },
    brain: {
      load: () => import("./brain.js").then((m) => m.brainSpec),
      description: "La via uditiva del cervello (coclea, tronco encefalico, talamo, corteccia uditiva) e le analogie con l'audio, ognuna con i suoi limiti e le fonti.",
    },
  });
  const paneSources = sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Paper originali e codice ufficiale da cui vengono i numeri dei modelli reali." });

  const err = lab.relativeError();
    const vars = {
    sampleRate: fmt(lab.sampleRate),
    duration: fmt(lab.songs[state.style].duration, 1),
    samples: fmt(lab.songs[state.style].samples.length),
    nFft: fmt(CODEC.nFft),
    hop: fmt(CODEC.hop),
    frames: fmt(lab.frames),
    bins: fmt(CODEC.bins),
    mels: "64",
    k: fmt(lab.rvq.k),
    err1: formatPercent(err[1]),
    err4: formatPercent(err[4]),
    err8: formatPercent(err[8]),
    frameRate: fmt(lab.frameRate),
    bitrate4: fmt(lab.bitrate(4)),
    pcmBitrate: fmt(lab.sampleRate * 16),
    whisperFrames: fmt(lab.whisper(state.style).signalFrames),
    gptVocab: await gptVocab.then((v) => (v ? fmt(v) : "poche decine di")),
  };
  panePath.append(el("p", { className: "audio-intro" }, "Il GPT della lezione genera caratteri. Qui vediamo come la stessa idea, il prossimo token, genera suono: prima bisogna trasformare l'audio in token e ritorno."));
  tab.menu = new Menu(panePath, STEPS, {
    vars,
    tabId: "audio",
    onSelect: (item) => {
      tab.focus(item.focus, { byUser: true });
      tab.code(item.code, item.highlight ?? []);
      if (item.stage !== undefined && !ui.player.action) stageTarget = item.stage;
    },
  });
  panePath.append(
    el(
      "div",
      { className: "audio-legend" },
      el("span", { className: "audio-ramp" }),
      el("small", {}, "spettrogrammi: debole → forte"),
      el("small", {}, "Token: il colore indica quale token (identità), non una grandezza."),
    ),
  );

  installActions(tab);
  ui.player.renderOverview(paneActions, { categories: ACTION_CATEGORIES });
  paneActions.append(el("button", { type: "button", className: "audio-link", onclick: () => app.open("transformer") }, "Rivedi il transformer del GPT"));

  const labPane = buildLabPane(paneLab, lab, {
    onStyle: (id) => tab.setStyle(id),
    playOriginal: () => play(lab.songs[state.style].samples, "originale", P.audio),
    playReconstruction: async (q) => {
      labPane.setStatus(`Ricostruzione con ${q} codebook (Griffin-Lim)…`);
      const audio = await tab.showReconstruction(q);
      play(audio, `ricostruito con ${q} codebook`, P.output);
    },
    playGenerated: async () => {
      if (!state.gen) return;
      play(await state.gen.audio(), "generato", P.output);
    },
    stop: () => out.stop(),
    generate: async (options) => {
      state.genOptions = options;
      ui.player.stop();
      const gen = tab.runGeneration(options);
      P.delay.show(gen, gen.steps.length - 1);
      tab.focus("decoderOut");
      await tab.showGenerated(gen);
    },
    animate: (options) => {
      state.genOptions = options;
      ui.player.start("audio-generate");
    },
  });

  let playheadTarget = null;
  function play(samples, name, target) {
    playheadTarget = target;
    out.play(samples, lab.sampleRate, name).then(
      () => labPane.setStatus(`In riproduzione: ${name}.`),
      (e) => labPane.setStatus(`Audio non disponibile: ${e.message}`),
    );
  }
  out.addEventListener("change", (e) => {
    if (!e.detail.playing) {
      P.audio.setPlayhead(null);
      P.output.setPlayhead(null);
      plots?.waveAt(null);
    }
  });

  /**
   * Vista d'insieme, la vista iniziale del tab: la usano il pulsante "Vista d'insieme", "Centra" della
   * navigazione (C, Home, ◎, "Torna al centro") e la vista "overview". Anche lo zoom torna automatico:
   * con un livello bloccato a mano (es. "Dettaglio") da lontano si vedrebbero ancora i dettagli.
   */
  function showOverview() {
    if (lod.mode !== "auto") lod.setMode("auto");
    return tab.focus("overview", { byUser: true });
  }
  const showView = (id) => (id === "overview" ? showOverview() : tab.focus(id, { byUser: true }));

  // viste: la stessa scena guardata da punti diversi (catalog.js)
  paneViews.append(
    el(
      "div",
      { className: "view-switch" },
      ...VIEWS.map((v) => el("button", { type: "button", dataset: { view: v.id }, onclick: () => showView(v.id) }, el("strong", {}, v.label), el("small", {}, v.text))),
    ),
    el(
      "p",
      { className: "hint audio-nav-hint" },
      touch.matches
        ? "Da una vista puoi proseguire da solo: un dito per girare attorno, due dita per spostarti e avvicinarti, doppio tocco su un componente per centrarlo."
        : "Da una vista puoi proseguire da solo: trascina per girare attorno, tasto destro (o Shift) e trascina per spostarti, rotella per avvicinarti. Doppio clic su un componente per centrarlo.",
    ),
  );

  paneSources.append(
    ...SOURCES.map((g) =>
      el(
        "section",
        { className: "audio-sources" },
        el("h3", {}, g.group),
        el("ul", {}, ...g.items.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, s.title), el("small", {}, s.note)))),
      ),
    ),
    el("p", { className: "hint" }, "Il brano, il codec giocattolo e il modello a conteggi di questo tab sono simulazioni didattiche calcolate nel browser; i numeri dei modelli reali vengono dalle fonti qui sopra."),
  );

  // --- barra dello stage
  stage.append(
    el(
      "div",
      { className: "stage-toolbar" },
      el("button", { type: "button", title: "Torna alla vista iniziale (C)", onclick: showOverview }, "Vista d'insieme"),
      el("button", { type: "button", ariaLabel: "Ascolta il brano", onclick: () => play(lab.songs[state.style].samples, "originale", P.audio) }, "▶ Ascolta", el("span", { className: "audio-wide" }, " il brano")),
    ),
  );

  host.onFrame((dt) => {
    flows.tick(dt);
    if (stageTarget !== null && !ui.player.action) {
      const s = P.encoder.stage;
      const next = s + Math.sign(stageTarget - s) * Math.min(Math.abs(stageTarget - s), dt * 1.8);
      P.encoder.setStage(next);
      if (next === stageTarget) stageTarget = null;
    }
    const t = out.position();
    if (t !== null && playheadTarget) {
      playheadTarget.setPlayhead(t);
      if (playheadTarget === P.audio) plots?.waveAt(t);
    }
  });
  // --- navigazione libera (ui.nav): "Centra" = vista d'insieme. Confini: il punto guardato può arrivare
  // al centro di ogni componente (dall'Audio a sinistra all'Audio generato a destra, dal prompt in alto a
  // Whisper in basso) e poco oltre (il nucleo allarga il box del 12%): anche al bordo il componente
  // estremo resta in mezzo allo schermo, non si scivola nel vuoto. Box fisso, calcolato una volta: niente
  // visite di tutta la scena del Dettaglio (pesante) per ricalcolarlo.
  const navBox = new THREE.Box3();
  const center = new THREE.Vector3();
  for (const plane of registry.values()) navBox.expandByPoint(plane.getWorldPosition(center));
  ui.nav.setBounds(navBox);
  ui.nav.setHome(showOverview);
  // La pipeline è una parete verticale (piano x-y) guardata da davanti: da sopra o da sotto si vedrebbe
  // solo una linea, da dietro il testo a specchio, e oltre una certa vicinanza si finisce dentro i blocchi.
  // Si gira attorno e ci si avvicina liberamente, ma entro questi limiti (valgono per mouse, tasti, pad, touch).
  ui.nav.setOrbitLimits({ turn: MAX_TURN_DEG, tilt: MAX_TILT_DEG, minDistance: MIN_ZOOM_DISTANCE });
  // Lo studente ha mosso la camera: non è più sul componente dell'ultimo volo, quindi un passo che lo
  // inquadra di nuovo (actions.js → frame) deve rivolarci. "Centra" lo reimposta subito dopo (setHome).
  ui.nav.on("input", () => (tab.focusedId = null));
  // "Segui l'azione" dopo che lo studente si è mosso: se nel frattempo l'azione non ha chiesto voli
  // (il passo inquadra ancora lo stesso componente) il director non ha un bersaglio in sospeso, e
  // senza questo la camera resterebbe dov'è. Il controllo avviene dopo resumeFollow, che vola da solo
  // se il bersaglio c'è.
  director.onFollowChange((following) => {
    if (!following || !director.actionActive || !tab.actionFocusId) return;
    queueMicrotask(() => {
      if (!director.flight && director.actionActive) tab.focus(tab.actionFocusId);
    });
  });

  // Tastiera: ↑ ↓ scorrono i passi del Percorso, Spazio mette in pausa l'azione (core/stepKeys.js).
  // Le altre lettere e ← → sono della navigazione libera.
  installStepKeys({ root, ui, menu: () => tab.menu, pathPane: panePath });

  plots?.setStyle(state.style);
  // una generazione di partenza: la scala del delay pattern e l'uscita non restano vuote
  state.gen = lab.generate(state.style, state.genOptions);
  tab.resetScene();
  setTimeout(() => tab.showGenerated(state.gen), 300);
  const firstStep = state.gen.steps[1][0];
  plots?.nextToken(firstStep.p, firstStep.token);
  codePanel.hide(); // all'inizio la scena intera: il codice compare quando scegli un passo
  // Didascalia di base: la scena è una pipeline orizzontale, quindi spostarsi a destra e a sinistra
  // vuol dire "percorrerla" (dal suono ai token e di nuovo al suono). Testo diverso per il touch.
  const idleCaption = () =>
    touch.matches
      ? "Suono → token → suono: allarga due dita per il dettaglio, trascina con due dita per percorrere la pipeline."
      : "Suono → token → suono: rotella per il dettaglio o per il diagramma a blocchi; A/D (o tasto destro e trascina) per percorrere la pipeline.";
  const refreshCaption = () => tab.setCaption(ui.player.action ? "" : idleCaption());
  refreshCaption();
  // durante un'azione parla la barra del player: la didascalia di base si toglie
  ui.player.on("state", refreshCaption);
  touch.addEventListener?.("change", refreshCaption);
  tab.focus("overview", { duration: 0.01 });
  window.__audioTab = tab; // per il debug dalla console e per i test automatici

  const phone = window.matchMedia(PHONE);
  /**
   * Link diretto o risultato della ricerca globale: porta il tab su un passo, un'azione, una vista o
   * un livello dello zoom. Restituisce false per un id sconosciuto (il guscio avvisa lo studente).
   */
  async function goto(target) {
    // view:rete e view:cervello mostrano la rappresentazione; per tutto il resto un livello aperto si chiude
    // (si torna all'Architettura) e si prosegue come sempre
    const shown = reps.goto(target);
    if (shown) return shown;
    const { kind, id } = target;
    if (kind === "step") return tab.menu.reveal(id, { sidebar }); // apre il Percorso sul passo
    // Azioni e viste si guardano nella scena: sul telefono il cassetto della sidebar la coprirebbe.
    const showScene = () => phone.matches && sidebar.toggleCollapsed(true);
    if (kind === "action") {
      if (!ACTION_IDS.has(id)) return false;
      sidebar.select("azioni");
      showScene();
      ui.player.start(id);
      return true;
    }
    if (kind !== "view" || !(VIEW_IDS.has(id) || LEVEL_IDS.has(id))) return false;
    // come una scelta dal pannello Viste, ma è un salto: l'azione in corso si chiude
    ui.player.stop();
    showScene();
    if (LEVEL_IDS.has(id)) {
      lod.pick(id); // come il selettore dello zoom: blocca il livello e lo inquadra
    } else {
      sidebar.select("viste");
      await showView(id);
    }
    return true;
  }

  return {
    tab,
    goto,
    activate() {
      host.start();
      host.resize();
    },
    deactivate() {
      host.stop(); // la navigazione si ferma da sola (niente scivolate al ritorno)
      out.stop();
    },
  };
}
