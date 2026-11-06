/**
 * Tab "Immagini e video": i modelli di diffusione.
 *
 * Questo file collega i pezzi (ognuno in un modulo proprio):
 *   pipeline.js   livello Dettaglio: pipeline text-to-image (U-Net oppure DiT)
 *   videoScene.js livello Dettaglio della parte video (cubo spazio-temporale, patch, attention)
 *   overview.js   livelli Blocchi e Panoramica (immagine e video)
 *   levels.js     zoom semantico (LOD) e inquadrature dei livelli scelti a mano
 *   lab.js        "banco di lavoro" sopra la scena dove le azioni mostrano immagini e latenti
 *   actions.js    azioni animate su immagini (rumore, generazione, VAE, dentro il denoiser)
 *   denoiserSim.js denoiser simulato (dichiarato) per la generazione; imagery.js immagini e VAE simulato
 *   videoActions.js azione animata sul video
 *   plots.js      grafici (schedule, SNR, rumore per passo, guidance, coerenza tra frame)
 *   sceneNav.js   navigazione libera: "Centra", confini, "Dettaglio" dove si guarda, didascalie delle zone
 *   content.js    percorso didattico, fonti; snippets.js frammenti di codice veri
 *   search.js     indice della ricerca globale (passi, azioni, viste); i link diretti arrivano a goto()
 */
import * as THREE from "three";
import { el } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { installStepKeys } from "../../core/stepKeys.js";
import { parseHash } from "../../core/search.js";
import { Menu } from "../../core/ui/menu.js";
import { installActions } from "./actions.js";
import { PATH, SOURCES } from "./content.js";
import { renderScene, SimulatedVAE } from "./imagery.js";
import { Lab } from "./lab.js";
import { createLevels, FRONT } from "./levels.js";
import { fillGaussian, mulberry32 } from "./math.js";
import { DenoiserSim } from "./denoiserSim.js";
import { buildBlocks, buildMacro, VIDEO_Y } from "./overview.js";
import { buildPipeline } from "./pipeline.js";
import { framingBox } from "./shapes.js";
import { installPlots } from "./plots.js";
import { installSceneNav } from "./sceneNav.js";
import { SNIPPETS } from "./snippets.js";
import { installVideo } from "./videoActions.js";
import { buildVideoScene } from "./videoScene.js";

const IMAGE_SIZE = 256;

/** Direzione per i dettagli: un po' di lato e dall'alto per dare profondità, ma testo leggibile. */
const DETAIL_DIR = new THREE.Vector3(0.16, 0.2, 1).normalize();
/** Stessa soglia del nucleo (layout.js, codePanel.js) per il layout a una colonna. */
const NARROW = window.matchMedia("(max-width: 900px)");
/** Altezza del banco di lavoro: abbastanza sopra la pipeline da non sovrapporsi alle sue etichette. */
const LAB_Y = 48;
/** In cima al Percorso: come si legge la scena e dove sta la parte video (si scopre spostandosi). */
const PATH_HINT =
  "Scegli un passo: la scena vola sul pezzo di cui si parla. Rotellina o pizzico per lo zoom (da vicino i blocchi interni, da lontano lo schema); tasto destro, due dita o W A S D per spostarti: sotto le immagini c'è la parte video.";

export async function createTab(root, app) {
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint: PATH_HINT,
  });
  const { host, director, sidebar, codePanel, stage } = ui;
  let sceneNav = null; // installSceneNav, più sotto: serve già al selettore dello zoom

  // --- dati: immagine di riferimento e VAE simulato (tutto generato nel browser)
  const reference = renderScene(new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3), IMAGE_SIZE);
  const vae = new SimulatedVAE(reference, IMAGE_SIZE);
  const sim = new DenoiserSim(vae, reference);

  // --- scena: tre livelli di LOD
  const detail = new THREE.Group();
  const pipeline = buildPipeline();
  detail.add(pipeline.group);
  const video = buildVideoScene();
  video.group.position.y = VIDEO_Y;
  detail.add(video.group);
  const blocks = buildBlocks();
  const macro = buildMacro();
  host.scene.add(detail, blocks.group, macro.group);
  host.onFrame((dt) => {
    for (const f of pipeline.flows) f.userData.tick(dt);
  });
  // pavimento sotto la parte video: solo nel livello Dettaglio (negli schemi taglierebbe i blocchi)
  const grid = new THREE.GridHelper(400, 100, 0x1e293b, 0x111827);
  grid.position.y = VIDEO_Y - 16;
  detail.add(grid);

  // stato iniziale delle texture (vedi tab.resetPipeline): latente di puro rumore, immagine finale
  const initialNoise = fillGaussian(new Float32Array(vae.z0.length), mulberry32(1234));
  let shownCode = null; // firma del frammento di codice mostrato (vedi tab.showCode)

  const lab = new Lab({ textScale: 1.45 });
  lab.group.position.set(0, LAB_Y, 0);
  host.scene.add(lab.group);

  const levels = createLevels({
    host,
    director,
    groups: { detail, blocks: blocks.group, macro: macro.group },
    widgetParent: stage,
    // "Dettaglio" scelto a mano: il pezzo che si sta guardando (parte video, testo, denoiser o VAE)
    focusDetail: () => sceneNav.focusDetail(),
    onPick: (mode) => sceneNav.levelPicked(mode),
  });

  // --- oggetto condiviso dai moduli
  const tab = {
    app,
    ui,
    lod: levels.lod,
    lab,
    vae,
    sim,
    reference,
    settings: { seed: 42, scale: 7.5, steps: 20 },
    pipeline,
    video,
    blocks,
    denoiser: "unet",
    /**
     * Mostra un frammento di codice vero (chiave di SNIPPETS). Se è già quello mostrato non fa
     * nulla: i passi delle azioni possono chiamarlo sempre (così "passo indietro" ritrova il codice
     * giusto) senza riaprire il pannello che l'utente ha ridotto.
     */
    showCode(key, title, highlight = []) {
      const s = SNIPPETS[key];
      if (!s) return;
      const signature = `${key}|${title}|${highlight.join("|")}`;
      if (signature === shownCode && !codePanel.root.hidden) return;
      shownCode = signature;
      codePanel.show({ title, file: s.file, startLine: s.startLine, code: s.code, language: s.language, highlight });
      // Su schermi stretti il pannello è un foglio in basso che, insieme alla barra del player,
      // coprirebbe la scena: durante le azioni si apre ridotto (il titolo resta, si espande a mano).
      if (ui.player.action && NARROW.matches) codePanel.root.classList.add("minimized");
    },
    hideCode() {
      shownCode = null;
      codePanel.hide();
    },
    /** Blocca il livello del LOD durante un'azione ("detail") o lo libera ("auto") senza muovere la camera. */
    lockLevel: levels.lock,
    /** Riporta la pipeline allo stato iniziale (latente di rumore, immagine di riferimento, t = 999). */
    resetPipeline() {
      pipeline.latentTex.forEach((t, c) => t.set(initialNoise, c));
      pipeline.epsTex.forEach((t, c) => t.set(initialNoise, c));
      pipeline.imageTex.set(reference);
      pipeline.setTimestep(999);
    },
    /**
     * Vola su un componente del registro (id), su più componenti insieme (array di id), su un
     * oggetto o su un Box3.
     */
    focus(target, options) {
      const lookup = (id) => pipeline.registry.get(id) ?? video.registry.get(id);
      const targets = (Array.isArray(target) ? target : [target]).map((t) => (typeof t === "string" ? lookup(t) : t)).filter(Boolean);
      if (!targets.length) return Promise.resolve();
      // Calcolo rimandato di un microtask: se nello stesso istante si apre il pannello del codice,
      // l'inquadratura tiene già conto dello spazio che copre.
      return Promise.resolve().then(() => {
        const box = new THREE.Box3();
        for (const t of targets) box.union(t.isBox3 ? t : framingBox(t));
        return director.focus(box, { direction: DETAIL_DIR, ...options });
      });
    },
    setCaption(text) {
      ui.caption.textContent = text;
    },
    setDenoiser(kind) {
      if (kind === tab.denoiser) return;
      // "Un passo dentro il denoiser" ha i passi costruiti per il denoiser visibile all'avvio
      if (ui.player.action?.id === "inside") ui.player.stop();
      tab.denoiser = kind;
      pipeline.setDenoiser(kind);
      blocks.unet.visible = kind === "unet";
      blocks.dit.visible = kind === "dit";
      viewButtons?.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.view === kind));
    },
  };

  tab.resetPipeline();
  sceneNav = installSceneNav(tab, levels);

  // --- sidebar
  const panePath = sidebar.addPane({ id: "percorso", title: "Percorso", hint: PATH_HINT });
  const paneActions = sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Animazioni passo per passo: usa i controlli in basso per pausa, passo avanti/indietro e velocità." });
  const paneViews = sidebar.addPane({ id: "viste", title: "Viste", hint: "Scegli quale denoiser mostrare e dove guardare." });
  const paneSources = sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Paper, codice ufficiale e documentazione da cui vengono numeri e formule." });

  tab.menu = new Menu(panePath, PATH, {
    tabId: "diffusion",
    onSelect: (item, title) => {
      // Un passo scelto durante un'azione: lo studente passa a leggere il Percorso. L'azione si chiude,
      // altrimenti il suo passo successivo riporterebbe subito altrove camera, codice e didascalia.
      if (ui.player.action) ui.player.stop();
      if (item.denoiser) tab.setDenoiser(item.denoiser);
      if (item.focus) tab.focus(item.focus, { margin: item.margin ?? 1.25 });
      if (item.code) tab.showCode(item.code, item.codeTitle ?? title, item.highlight ?? []);
      else tab.hideCode();
      tab.setCaption(item.caption ?? "");
    },
  });

  const viewDefs = [
    { id: "unet", title: "Stable Diffusion: U-Net", text: "Encoder-decoder con skip connection, ResNet block, self- e cross-attention.", run: () => (tab.setDenoiser("unet"), tab.focus("unet")) },
    { id: "dit", title: "Diffusion Transformer (DiT)", text: "Il latente diventa una sequenza di patch: blocchi transformer con adaLN, come nel nostro GPT.", run: () => (tab.setDenoiser("dit"), tab.focus("dit")) },
    { id: "pipeline", title: "Tutta la pipeline", text: "Dal prompt all'immagine, vista d'insieme (livello Blocchi).", run: () => sceneNav.overview() },
    { id: "video", title: "Video: latenti spazio-temporali", text: "Cubo di latenti, patch spazio-temporali, attention nello spazio e nel tempo.", run: () => tab.focus(video.regions.intro(), { showFrame: false, margin: 1.02, direction: FRONT }) },
  ];
  const viewButtons = el(
    "div",
    { className: "view-switch" },
    ...viewDefs.map((v) =>
      el(
        "button",
        {
          type: "button",
          dataset: { view: v.id },
          className: v.id === "unet" ? "active" : "",
          onclick: () => showView(v),
        },
        el("strong", {}, v.title),
        el("small", {}, v.text),
      ),
    ),
  );
  paneViews.append(viewButtons);

  /** Una vista del pannello "Viste" (pulsante o link diretto). */
  function showView(v) {
    // come un passo del Percorso: l'azione in corso si chiude, altrimenti il suo passo successivo
    // riporterebbe subito la camera altrove
    if (ui.player.action) ui.player.stop();
    tab.hideCode(); // le viste servono a guardare la scena: niente pannello sopra
    tab.setCaption("");
    if (NARROW.matches) sidebar.toggleCollapsed(true); // sul telefono il cassetto coprirebbe la scena
    return v.run();
  }

  paneSources.append(renderSources());

  // --- moduli
  installPlots(tab);
  installActions(tab, paneActions);
  installVideo(tab, paneActions);
  ui.player.renderOverview(tab.actionsList, { categories: ["Training", "Generazione (prompt)", "Video"] });
  // Sul telefono la sidebar è un cassetto sopra la scena: un'azione appena avviata si guarda nella
  // scena, quindi il cassetto si chiude (da "Avvia" come da un link diretto).
  let runningAction = null;
  ui.player.on("state", () => {
    const id = ui.player.action?.id ?? null;
    if (id && id !== runningAction && NARROW.matches) sidebar.toggleCollapsed(true);
    runningAction = id;
  });

  // --- tasti: ↑ ↓ scorrono i passi del Percorso, Spazio mette in pausa l'azione (core/stepKeys.js)
  installStepKeys({ root, ui, menu: () => tab.menu, pathPane: panePath });

  tab.hideCode(); // si apre quando un passo del percorso o un'azione mostra del codice
  // Aperto da un link diretto a un passo: si parte da quel passo (goto lo mostrerà subito dopo), così
  // il primo passo del Percorso non viene segnato come visto senza che lo studente l'abbia letto.
  const initial = parseHash(location.hash);
  const linked = initial.tab === "diffusion" && initial.go?.kind === "step" && tab.menu.items.has(initial.go.id);
  tab.menu.select(linked ? initial.go.id : "idea.cosa", { scroll: false });
  levels.flyToLevel("blocks", 0.01);
  window.__diffusionTab = tab; // per il debug dalla console e per gli screenshot automatici

  const actionIds = new Set(ui.player.actions.keys());
  /** Viste raggiungibili da un link diretto: quelle del pannello più i due livelli estremi dello zoom. */
  const linkViews = new Map([
    ...viewDefs.map((v) => [v.id, () => showView(v)]),
    ["macro", () => showView({ run: () => sceneNav.overview("macro") })],
    ["detail", () => showView({ run: () => sceneNav.detail() })],
  ]);

  /** Link diretto o risultato della ricerca globale: porta il tab su un passo, un'azione o una vista. */
  async function goto({ kind, id }) {
    if (kind === "step") return tab.menu.reveal(id, { sidebar }); // apre il Percorso sul passo
    if (kind === "action") {
      if (!actionIds.has(id)) return false;
      sidebar.select("azioni");
      ui.player.start(id);
      return true;
    }
    if (kind === "view") {
      const show = linkViews.get(id);
      if (!show) return false;
      sidebar.select("viste");
      await show();
      return true;
    }
    return false;
  }

  return {
    tab,
    goto,
    activate() {
      host.start();
      host.resize();
    },
    deactivate() {
      ui.player.stop();
      host.stop();
    },
  };
}

/** Pannello "Fonti": elenco ragionato con link. */
function renderSources() {
  return el(
    "div",
    { className: "diff-sources" },
    ...SOURCES.map((group) =>
      el(
        "section",
        {},
        el("h3", {}, group.title),
        el(
          "ul",
          {},
          ...group.items.map((s) =>
            el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener" }, s.title), s.who ? el("span", { className: "muted" }, ` · ${s.who}`) : null, el("p", { className: "hint" }, s.note)),
          ),
        ),
      ),
    ),
  );
}
