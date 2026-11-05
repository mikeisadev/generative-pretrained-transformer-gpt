/**
 * Tab "Immagini e video": i modelli di diffusione.
 *
 * Questo file collega i pezzi (ognuno in un modulo proprio):
 *   pipeline.js   livello Dettaglio: pipeline text-to-image (U-Net oppure DiT)
 *   videoScene.js livello Dettaglio della parte video (cubo spazio-temporale, patch, attention)
 *   overview.js   livelli Blocchi e Panoramica (immagine e video)
 *   lab.js        "banco di lavoro" sopra la scena dove le azioni mostrano immagini e latenti
 *   actions.js    azioni animate su immagini (rumore, generazione, VAE)
 *   videoActions.js azione animata sul video
 *   plots.js      grafici (schedule, SNR, rumore per passo, guidance, coerenza tra frame)
 *   content.js    percorso didattico, fonti; snippets.js frammenti di codice veri
 */
import * as THREE from "three";
import { el } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { LODController } from "../../core/lod.js";
import { Menu } from "../../core/ui/menu.js";
import { installActions } from "./actions.js";
import { PATH, SOURCES } from "./content.js";
import { renderScene, SimulatedVAE } from "./imagery.js";
import { Lab } from "./lab.js";
import { fillGaussian, mulberry32 } from "./math.js";
import { DenoiserSim } from "./denoiserSim.js";
import { buildBlocks, buildMacro, VIDEO_Y } from "./overview.js";
import { buildPipeline } from "./pipeline.js";
import { installPlots } from "./plots.js";
import { SNIPPETS } from "./snippets.js";
import { installVideo } from "./videoActions.js";
import { buildVideoScene } from "./videoScene.js";

const IMAGE_SIZE = 256;

/** Vista (camera) di partenza per ogni livello del LOD quando l'utente lo sceglie a mano. */
const LEVEL_DISTANCE = { detail: 70, blocks: 150, macro: 300 };

export async function createTab(root, app) {
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint: "Segui il percorso o avvia un'azione. Rotellina per lo zoom: da vicino vedi i blocchi interni, da lontano lo schema.",
  });
  const { host, director, sidebar, codePanel, stage } = ui;

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
  const grid = new THREE.GridHelper(400, 100, 0x1e293b, 0x111827);
  grid.position.y = VIDEO_Y - 16;
  host.scene.add(grid);

  // stato iniziale delle texture: latente di puro rumore, immagine finale, embedding di t
  const noise = fillGaussian(new Float32Array(vae.z0.length), mulberry32(1234));
  pipeline.latentTex.forEach((t, c) => t.set(noise, c));
  pipeline.epsTex.forEach((t, c) => t.set(noise, c));
  pipeline.imageTex.set(reference);
  pipeline.setTimestep(999);

  const lab = new Lab(host);
  lab.group.position.set(0, 30, 0);
  host.scene.add(lab.group);

  // `lod` è dichiarato prima: il costruttore chiama subito onChange
  let lod = null;
  lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: 95, group: detail },
      { id: "blocks", label: "Blocchi", maxDistance: 230, group: blocks.group },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: macro.group },
    ],
    widgetParent: stage,
    onChange: (id) => {
      if (lod?.mode !== "auto" && lod?.mode === id) flyToLevel(id);
    },
  });
  function flyToLevel(id) {
    const target = host.controls.target.clone();
    const dir = host.camera.position.clone().sub(target).normalize();
    director.flyTo(target.clone().addScaledVector(dir, LEVEL_DISTANCE[id]), target, 1);
  }

  // --- oggetto condiviso dai moduli
  const tab = {
    app,
    ui,
    lod,
    lab,
    vae,
    sim,
    reference,
    settings: { seed: 42, scale: 7.5, steps: 20 },
    pipeline,
    video,
    blocks,
    denoiser: "unet",
    /** Mostra un frammento di codice vero (chiave di SNIPPETS). */
    showCode(key, title, highlight = []) {
      const s = SNIPPETS[key];
      if (!s) return;
      codePanel.show({ title, file: s.file, startLine: s.startLine, code: s.code, language: s.language, highlight });
    },
    /** Vola su un componente del registro o su un oggetto. */
    focus(target, options) {
      const object = typeof target === "string" ? pipeline.registry.get(target) ?? video.registry.get(target) : target;
      return object ? director.focus(object, options) : Promise.resolve();
    },
    setCaption(text) {
      ui.caption.textContent = text;
    },
    setDenoiser(kind) {
      tab.denoiser = kind;
      pipeline.setDenoiser(kind);
      blocks.unet.visible = kind === "unet";
      blocks.dit.visible = kind === "dit";
      viewButtons?.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.view === kind));
    },
  };

  // --- sidebar
  const panePath = sidebar.addPane({ id: "percorso", title: "Percorso" });
  const paneActions = sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Animazioni passo per passo: usa i controlli in basso per pausa, passo avanti/indietro e velocità." });
  const paneViews = sidebar.addPane({ id: "viste", title: "Viste", hint: "Scegli quale denoiser mostrare e dove guardare." });
  const paneSources = sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Paper, codice ufficiale e documentazione da cui vengono numeri e formule." });

  tab.menu = new Menu(panePath, PATH, {
    onSelect: (item, title) => {
      if (item.denoiser) tab.setDenoiser(item.denoiser);
      if (item.focus) tab.focus(item.focus, { margin: item.margin ?? 1.25 });
      if (item.code) tab.showCode(item.code, item.codeTitle ?? title, item.highlight ?? []);
      if (item.caption) tab.setCaption(item.caption);
    },
  });

  const viewDefs = [
    { id: "unet", title: "Stable Diffusion: U-Net", text: "Encoder-decoder con skip connection, ResNet block, self- e cross-attention.", run: () => (tab.setDenoiser("unet"), tab.focus("unet")) },
    { id: "dit", title: "Diffusion Transformer (DiT)", text: "Il latente diventa una sequenza di patch: blocchi transformer con adaLN, come nel nostro GPT.", run: () => (tab.setDenoiser("dit"), tab.focus("dit")) },
    { id: "pipeline", title: "Tutta la pipeline", text: "Dal prompt all'immagine, vista d'insieme.", run: () => tab.focus(pipeline.group, { showFrame: false, margin: 1.05 }) },
    { id: "video", title: "Video: latenti spazio-temporali", text: "Cubo di latenti, patch spazio-temporali, attention nello spazio e nel tempo.", run: () => tab.focus(video.group, { showFrame: false, margin: 1.1 }) },
  ];
  const viewButtons = el(
    "div",
    { className: "view-switch" },
    ...viewDefs.map((v) => el("button", { type: "button", dataset: { view: v.id }, className: v.id === "unet" ? "active" : "", onclick: v.run }, el("strong", {}, v.title), el("small", {}, v.text))),
  );
  paneViews.append(viewButtons);

  paneSources.append(renderSources());

  // --- moduli
  installPlots(tab);
  installActions(tab, paneActions);
  installVideo(tab, paneActions);
  ui.player.renderOverview(tab.actionsList, { categories: ["Immagini", "Video"] });

  // --- tasti
  root.tabIndex = -1;
  root.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowDown") tab.menu.move(+1);
    if (e.key === "ArrowUp") tab.menu.move(-1);
    if (e.key === " " && ui.player?.action) {
      e.preventDefault();
      ui.player.toggle();
    }
  });

  tab.menu.select("idea.cosa");
  tab.focus(pipeline.group, { showFrame: false, margin: 1.05 });
  window.__diffusionTab = tab; // per il debug dalla console e per gli screenshot automatici

  return {
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
