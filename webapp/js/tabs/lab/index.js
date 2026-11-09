/**
 * Tab "Laboratorio pesi": i pesi del GPT in Python organizzati come il codice, come numeri e celle
 * colorate; i run caricati dallo studente (anche con un'architettura modificata) e, nella fase 2,
 * animazioni, confronto, grafici e benchmark.
 *
 * Questo file è il raccordo: crea il layout, lo stato e l'oggetto `tab` (contratto in lab.md), poi
 * delega ai moduli:
 *   state.js           registro dei run, run A/B, step, scelta, modalità, pesi a uno step, motore JS
 *   scene/             la scena 3D (layout puro, contenitori, etichette, NumericMatrix, LOD, mouse)
 *   inspect/           pannelli "Modello" (indice dei moduli, step) e "Tensore" (numeri, statistiche)
 *   runs/              pannello "Run": caricamento, salvataggio nel browser, scheda, esportazione
 *   network.js         rappresentazione "Rete neurale" del run A;  brain.js: "Cervello" (soli dati)
 *   actions/ analysis/ fase 2: installLabActions(tab), installLabAnalysis(tab)
 *   links.js           link diretti (tensor:, run:, view:, action:);  search.js: indice della ricerca
 */
import { fetchJson } from "../../core/binary.js";
import { createTabLayout } from "../../core/layout.js";
import { el } from "../../core/dom.js";
import { installRepresentations } from "../../core/representations.js";
import { loadBuiltinRun } from "../../shared/weights/run.js";
import { RunStore } from "../../shared/weights/store.js";
import { installLabActions } from "./actions/index.js";
import { installLabAnalysis } from "./analysis/index.js";
import { TAB_HINT_MOUSE, TAB_HINT_TOUCH } from "./content.js";
import { gotoTarget } from "./goto.js";
import { createInspector } from "./inspect/pane.js";
import { createModelPane } from "./inspect/modelPane.js";
import { createNetworkSource } from "./network.js";
import { createRunsPane } from "./runs/pane.js";
import { stageChip } from "./runs/timeline.js";
import { dropZone } from "./runs/upload.js";
import { installScene } from "./scene/index.js";
import { LabState } from "./state.js";
import { createTabObject } from "./tabObject.js";

const narrow = window.matchMedia("(max-width: 900px)");
const touch = window.matchMedia("(pointer: coarse)");

export async function createTab(root, app) {
  root.classList.add("lab-tab");
  const ui = createTabLayout(root, { code: true, charts: true, player: true, hint: touch.matches ? TAB_HINT_TOUCH : TAB_HINT_MOUSE });
  const spec = await fetchJson("data/model.json"); // gli snippet del codice vero (gpt/export.py)
  const state = new LabState({ loadBuiltin: () => loadBuiltinRun(), store: new RunStore() });
  const tab = createTabObject({ app, ui, spec, state });

  // pannelli della sidebar: Modello, Tensore, Run (poi la fase 2, poi Cervello e Fonti)
  tab.panes.modello = ui.sidebar.addPane({ id: "modello", title: "Modello", hint: "I moduli del GPT come nel codice Python, lo step del training e come leggere la scena." });
  tab.panes.tensore = ui.sidebar.addPane({ id: "tensore", title: "Tensore", hint: "I numeri veri del tensore scelto: tabella, statistiche, istogramma." });
  tab.panes.run = ui.sidebar.addPane({ id: "run", title: "Run", hint: "Il GPT del corso e i run che carichi: cartelle esportate dal Python, file, trascinamento." });
  // i pannelli condividono lo scorrimento della sidebar: cambiando pannello si riparte dall'alto
  let paneShown = ui.sidebar.current;
  ui.sidebar.onSelect((id) => {
    if (id !== paneShown) ui.sidebar.body.scrollTop = 0;
    paneShown = id;
  });
  tab.scene = installScene(tab);
  tab.model = createModelPane(tab.panes.modello, tab);
  tab.inspector = createInspector(tab.panes.tensore, tab);
  tab.runs = createRunsPane(tab.panes.run, tab);

  // barra dello stage: quale run e quale step si stanno guardando; rilascio dei file su tutta la scena
  ui.stage.append(el("div", { className: "stage-toolbar lab-toolbar" }, stageChip(tab)));
  const dropHint = el("div", { className: "lab-drop-overlay", hidden: true, ariaHidden: "true" }, el("strong", {}, "Rilascia qui la cartella del run"), el("span", {}, "weights.json, run.json, training/, .safetensors"));
  ui.stage.append(dropHint);
  tab.disposers.push(dropZone(ui.stage, { onDrop: (dt) => tab.runs.loadDropped(dt), indicator: dropHint }));

  // il modello è molto più largo che alto: all'inizio il codice resta chiuso e la scena ha tutta la
  // larghezza; compare al primo clic su una matrice o su un modulo (tab.inspect → tab.showCode)
  ui.codePanel.hide();
  await state.init(); // GPT del corso come run A: la scena si costruisce (evento runA), i pannelli si riempiono
  tab.setCaption(narrow.matches ? "Pizzica per avvicinarti: celle e poi numeri. Tocca una matrice." : "Avvicinati con la rotellina: prima le celle colorate, poi i numeri veri. Clic su una matrice per la sua tabella.");

  // fase 2 (animazioni; confronto, grafici, benchmark): dopo scena e pannelli, prima di Cervello e Fonti
  for (const install of [installLabActions, installLabAnalysis]) {
    const installed = install(tab);
    if (installed?.dispose) tab.disposers.push(() => installed.dispose());
  }

  // rappresentazioni: Rete neurale del run A (renderer del nucleo) e Cervello (pannello + 3D)
  const network = createNetworkSource(tab);
  tab.reps = installRepresentations(ui, {
    tabId: "lab",
    app,
    network: { load: network.load, description: "La rete vera del run nella scena: neuroni da sinistra a destra, i token di un prompt d'esempio in profondità." },
    brain: { load: () => import("./brain.js").then((m) => m.brainSpec) },
  });
  network.attach(tab.reps);
  tab.scene.attachReps(tab.reps);
  for (const type of ["runA", "weights"]) state.on(type, () => network.invalidate());
  tab.network = network;

  ui.sidebar.addPane({ id: "fonti", title: "Fonti", hint: "Da dove vengono codice, formati e regole." }).append(renderSources(tab.sources));
  window.__labTab = tab; // per la console e gli script di verifica

  return {
    tab,
    activate() {
      ui.host.start();
      ui.host.resize();
    },
    deactivate: () => ui.host.stop(), // l'azione in corso si congela e riprende al ritorno (come nel Transformer)
    goto: (target) => gotoTarget(tab, target),
  };
}

/** Il pannello "Fonti" (stile condiviso .sources-list di core.css). */
function renderSources(groups) {
  return el(
    "div",
    { className: "sources-list" },
    ...groups.map((group) =>
      el(
        "section",
        {},
        el("h3", {}, group.title),
        el("ul", {}, ...group.items.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, s.title), el("p", { className: "hint" }, s.note)))),
      ),
    ),
  );
}
