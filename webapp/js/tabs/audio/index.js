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
 */
import * as THREE from "three";
import { el } from "../../core/dom.js";
import { createTabLayout } from "../../core/layout.js";
import { LODController } from "../../core/lod.js";
import { Menu } from "../../core/ui/menu.js";
import { installActions } from "./actions.js";
import { AudioOut } from "./audioOut.js";
import { CODEC } from "./codec.js";
import { SOURCES, STEPS } from "./content.js";
import { createLab } from "./lab.js";
import { buildLabPane } from "./labPane.js";
import { LOD_DISTANCES, SCENE_OFFSET } from "./layout3d.js";
import { createPlots } from "./plots.js";
import { buildAudio, buildEncoder, buildRvq } from "./sceneCodec.js";
import { buildConditioning, buildCrossAttention, buildDecoder, buildDelay, buildOutput, buildTransformer } from "./sceneGen.js";
import { buildBlocks, buildFlows, buildMacro } from "./sceneLod.js";
import { buildWhisper } from "./sceneWhisper.js";
import { showSnippet } from "./snippets.js";

const fmtIt = (n, d = 0) => Number(n).toLocaleString("it-IT", { maximumFractionDigits: d, minimumFractionDigits: d });

/** Oggetto "virtuale" da inquadrare: l'unione dei membri (vedi CameraDirector.focus). */
function members(...objects) {
  const o = new THREE.Object3D();
  o.userData.members = objects.flat();
  return o;
}

export async function createTab(root, app) {
  const ui = createTabLayout(root, {
    code: true,
    charts: true,
    player: true,
    hint: "Scegli un punto del percorso o un'azione: la camera vola sul componente e a destra compare il codice. Rotellina: più ti allontani, più la vista si semplifica (dettaglio → blocchi → panoramica).",
  });
  const { host, director, sidebar, stage, codePanel } = ui;
  root.classList.add("audio-tab");

  // --- calcoli veri (circa 2 secondi): schermata di attesa con avanzamento
  const loading = el("div", { className: "audio-loading", role: "status" }, el("strong", {}, "Preparo il laboratorio audio"), el("p", {}, "Avvio…"), el("div", { className: "audio-progress" }, el("div")));
  stage.append(loading);
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

  const lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: LOD_DISTANCES.detail, group: detail },
      { id: "blocks", label: "Blocchi", maxDistance: LOD_DISTANCES.blocks, group: blocks.group },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: macro },
    ],
    widgetParent: stage,
  });

  // registro degli oggetti da inquadrare
  const P = parts;
  const registry = new Map([
    ["audio", P.audio.group],
    ["audio.lens", members(P.audio.lens, P.audio.group.children[1])],
    ["encoder", P.encoder.group],
    ["rvq", P.rvq.group],
    ["transformer", P.transformer.group],
    ["delay", P.delay.group],
    ["delayCore", members(P.transformer.group, P.delay.group)],
    ["genCore", members(P.cond.t5, P.transformer.group)],
    ["conditioning", members(P.cond.text, P.cond.t5)],
    ["t5", members(P.cond.text, P.cond.t5, P.transformer.group)],
    ["decoder", members(P.decoder.group)],
    ["decoderOut", members(P.decoder.group, P.output.group)],
    ["output", P.output.group],
    ["wMel", P.whisper.parts.wMel],
    ["wEnc", members(P.whisper.parts.wEnc, P.whisper.parts.wDec)],
    ["wDec", members(P.whisper.parts.wDec, P.whisper.parts.wText)],
    ["whisperOut", members(P.whisper.parts.wDec, P.whisper.parts.wText)],
    ["whisperAll", members(P.whisper.parts.wMel, P.whisper.parts.wText)],
    ["overview", blocks.group],
  ]);

  const plots = createPlots(ui.charts, lab);
  const out = new AudioOut();

  const tab = {
    app,
    ui,
    lab,
    parts,
    plots,
    state,
    lod,
    focus(id, options = {}) {
      const target = registry.get(id);
      if (!target) return Promise.resolve();
      if (id === "overview") return director.focus(target, { showFrame: false, margin: 1.08, ...options });
      return director.focus(target, { margin: 1.25, ...options });
    },
    code(ref, highlight = [], title) {
      return showSnippet(codePanel, ref, highlight, title);
    },
    setCaption(text) {
      ui.caption.textContent = text;
    },
    /** Riporta la scena allo stato di riposo (fine o interruzione di un'azione). */
    resetScene({ keepStage = false, keepGeneration = false } = {}) {
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
      P.whisper.showTokens(8);
      if (!keepGeneration) P.delay.show(state.gen, state.gen ? state.gen.steps.length - 1 : -1);
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
      labPane.setStatus(`Brano generato (temperatura ${fmtIt(gen.options?.temperature ?? state.genOptions.temperature, 1)}, seme ${state.genOptions.seed}): premi «Generato».`);
    },
    async showReconstruction(q) {
      const audio = await lab.reconstruct(state.style, q);
      P.output.setAudio(audio, "rec");
      return audio;
    },
    setStyle(id) {
      if (!lab.songs[id] || id === state.style) return;
      out.stop();
      state.style = id;
      state.gen = null;
      for (const k of ["audio", "encoder", "rvq", "cond", "whisper"]) P[k].setStyle(id);
      P.delay.show(null, -1);
      P.output.setAudio(null);
      labPane.enableGenerated(false);
      labPane.selectStyle(id);
      plots?.setStyle(id);
      labPane.setStatus(`Brano scelto: "${lab.songs[id].style.prompt}".`);
    },
  };

  // --- sidebar
  const panePath = sidebar.addPane({ id: "percorso", title: "Percorso" });
  const paneActions = sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Animazioni passo per passo: usa play/pausa, avanti/indietro e la velocità nella barra in basso." });
  const paneLab = sidebar.addPane({ id: "lab", title: "Laboratorio", hint: "Tutto calcolato qui nel browser: ascolta, cambia i codebook, genera." });
  const paneViews = sidebar.addPane({ id: "viste", title: "Viste" });
  const paneSources = sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Paper originali e codice ufficiale da cui vengono i numeri dei modelli reali." });

  const err = lab.relativeError();
  const pct = (v) => `${fmtIt(v * 100, v < 0.01 ? 2 : 1)} %`;
  const vars = {
    sampleRate: fmtIt(lab.sampleRate),
    duration: fmtIt(lab.songs[state.style].duration, 1),
    samples: fmtIt(lab.songs[state.style].samples.length),
    nFft: fmtIt(CODEC.nFft),
    hop: fmtIt(CODEC.hop),
    frames: fmtIt(lab.frames),
    bins: fmtIt(CODEC.bins),
    mels: "64",
    k: fmtIt(lab.rvq.k),
    err1: pct(err[1]),
    err4: pct(err[4]),
    err8: pct(err[8]),
    frameRate: fmtIt(lab.frameRate),
    bitrate4: fmtIt(lab.bitrate(4)),
    pcmBitrate: fmtIt(lab.sampleRate * 16),
    whisperFrames: fmtIt(lab.whisper(state.style).signalFrames),
  };
  panePath.append(el("p", { className: "audio-intro" }, "Il GPT della lezione genera caratteri. Qui vediamo come la stessa idea, il prossimo token, genera suono: prima bisogna trasformare l'audio in token e ritorno."));
  new Menu(panePath, STEPS, {
    vars,
    onSelect: (item) => {
      tab.focus(item.focus);
      tab.code(item.code, item.highlight ?? []);
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
  ui.player.renderOverview(paneActions, { categories: ["Rappresentare il suono", "Codec neurale", "Generazione", "Riconoscimento vocale"] });
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

  // viste: la stessa scena guardata da punti diversi
  const views = [
    { id: "overview", label: "Vista d'insieme", text: "Tutto il percorso: suono → token → suono, il prompt e Whisper." },
    { id: "delayCore", label: "Generazione musicale", text: "Transformer e delay pattern: il cuore di MusicGen." },
    { id: "rvq", label: "Codec: token audio", text: "Codebook, residui e griglia dei token." },
    { id: "whisperAll", label: "Whisper: dalla voce al testo", text: "Vista alternativa: log-Mel → encoder → decoder → testo." },
  ];
  paneViews.append(
    el(
      "div",
      { className: "view-switch" },
      ...views.map((v) => el("button", { type: "button", dataset: { view: v.id }, onclick: () => tab.focus(v.id) }, el("strong", {}, v.label), el("small", {}, v.text))),
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
      el("button", { type: "button", onclick: () => tab.focus("overview") }, "Vista d'insieme"),
      el("button", { type: "button", onclick: () => play(lab.songs[state.style].samples, "originale", P.audio) }, "▶ Ascolta il brano"),
    ),
  );

  host.onFrame((dt) => {
    flows.tick(dt);
    const t = out.position();
    if (t !== null && playheadTarget) {
      playheadTarget.setPlayhead(t);
      if (playheadTarget === P.audio) plots?.waveAt(t);
    }
  });
  root.tabIndex = -1;
  root.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === " " && ui.player.action) {
      e.preventDefault();
      ui.player.toggle();
    }
  });

  plots?.setStyle(state.style);
  tab.resetScene();
  tab.code("js:song.js#synthesize", ["addNote(out"], "Il brano: sintesi additiva (questo tab)");
  tab.setCaption("Suono → token → suono: zooma su un componente per il dettaglio, allontanati per il diagramma a blocchi.");
  tab.focus("overview", { duration: 0.01 });
  window.__audioTab = tab; // per il debug dalla console e per i test automatici

  return {
    tab,
    activate() {
      host.start();
      host.resize();
    },
    deactivate() {
      host.stop();
      out.stop();
    },
  };
}
