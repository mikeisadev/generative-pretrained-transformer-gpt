/**
 * Vista "Contesto" del tab Transformer: dal prompt dell'utente al contesto completo che riceve
 * l'LLM, collegata al tab "Database vettoriale (RAG)".
 *
 *   pipeline.js       scenario calcolato con js/shared/rag (guardrail, embedding, HNSW, prompt, SSE)
 *   layout.js         slot delle fasi a sinistra della pila e collegamenti
 *   scene.js          livelli Dettaglio/Blocchi/Panoramica (levels.js, cards.js, painters.js), pacchetto, token
 *   action.js         azione animata "Dalla domanda al contesto dell'LLM" (categoria "Contesto")
 *   contextCharts.js  sezione "Contesto" del pannello Grafici
 *   pane.js           pannello "Contesto" della sidebar (percorso, approfondimenti, fonti)
 *   texts.js / snippets.js   testi didattici e codice Java/Spring AI delle lezioni L17-L18
 *
 * Contratto con il tab RAG (via tab.app.bus):
 *   - emettiamo "rag:query" con detail = { question, topK, similarityThreshold, source: "transformer" }
 *     DOPO aver aperto il tab "rag" e atteso che sia stato creato (il tab si carica in modo pigro);
 *   - il tab RAG apre questa vista con "transformer:view" e detail = "contesto" (ascoltato da index.js);
 *   - facoltativo: "context:question" con detail = { question } imposta la domanda di questa vista.
 */
import * as THREE from "three";
import { sleep } from "../../../core/dom.js";
import { registerContextAction } from "./action.js";
import { ContextCharts } from "./contextCharts.js";
import { CONTEXT_DISTANCES, DIAGRAM_IDS, overviewBox, slotsBox } from "./layout.js";
import { ContextPane } from "./pane.js";
import { DEMO_QUESTION, createContextModel } from "./pipeline.js";
import { ContextScene, FULL_STATE } from "./scene.js";
import { SNIPPETS, showSnippet } from "./snippets.js";
import { STAGE } from "./texts.js";

const VIEW_ID = "contesto";
const FRONT = new THREE.Vector3(0.03, 0.07, 1).normalize(); // quasi di fronte: le schede restano leggibili
const RAG_WAIT_MS = 10000;

export function installContext(tab) {
  const { ui } = tab;
  const { host, director } = ui;
  if (!tab.lod) throw new Error("Vista Contesto: serve tab.lod (lod.js) installato prima");

  const scene = new ContextScene(tab);
  const unregister = tab.lod.register(scene.levels);
  const charts = new ContextCharts(ui.charts);
  let model = null; // vector store del corso: costruito alla prima apertura (~30 ms)
  let offFrame = null;
  let flowSaved = null;

  /**
   * Durante un'azione la barra del player copre il fondo dello stage: allunghiamo il riquadro da
   * inquadrare verso il basso, così la scheda resta nella parte alta, visibile.
   */
  const roomForPlayer = (box) => {
    if (ui.player?.action) box.min.y -= (box.max.y - box.min.y) * 0.32;
    return box;
  };

  const view = {
    tab,
    scene,
    charts,
    scenario: null,
    question: DEMO_QUESTION,

    /** Scenario della domanda corrente (calcolato una volta per domanda). */
    async ensure() {
      if (view.scenario?.question === view.question) return view.scenario;
      model ??= createContextModel();
      view.scenario = model.run(view.question);
      scene.setScenario(view.scenario);
      pane.setScenario(view.scenario, tab.vars);
      return view.scenario;
    },

    focus(ids, { duration = 1.2, margin = 1.12 } = {}) {
      return director.focus(roomForPlayer(slotsBox(ids)), { showFrame: false, margin, direction: FRONT, duration, minDistance: 8 });
    },

    /** Inquadra la scheda dei token, il riquadro LLM e la parte bassa della pila (dove entrano i token). */
    focusLlm() {
      const box = slotsBox(["tokenize", "llm"]);
      box.expandByPoint(new THREE.Vector3(14, -46, 0)).expandByPoint(new THREE.Vector3(14, 0, 0));
      return director.focus(roomForPlayer(box), { showFrame: false, margin: 1.05, direction: FRONT, duration: 1.3 });
    },

    overview() {
      return director.focus(roomForPlayer(overviewBox()), { showFrame: false, margin: 1.02, direction: FRONT, duration: 1.6 });
    },

    showCode(key) {
      if (SNIPPETS[key]) showSnippet(ui.codePanel, key);
      else tab.showCode(key, key === "data.tokenizer" ? "Il tokenizzatore del nostro GPT (un token per carattere)" : key, ["encode", "stoi"]);
    },

    /** Durante l'azione il pacchetto della pila lo muoviamo noi (moveArchFlow); poi si ripristina. */
    lockFlow(on) {
      if (on && !flowSaved) {
        flowSaved = { playing: tab.flow.playing, visible: tab.flow.packet.visible };
        tab.flow.playing = false;
        tab.flow.packet.visible = true;
      } else if (!on && flowSaved) {
        tab.flow.playing = flowSaved.playing;
        tab.flow.packet.visible = flowSaved.visible;
        flowSaved = null;
      }
    },

    moveArchFlow(p) {
      tab.flow.moveTo(p * tab.flow.length);
    },
  };

  const pane = new ContextPane(tab, {
    onQuestion: async (question) => {
      if (ui.player?.action?.id === "context-pipeline") ui.player.stop();
      view.question = question;
      await tab.views.activate(VIEW_ID);
      const s = await view.ensure();
      scene.setState(FULL_STATE);
      charts.showAll(s);
      tab.setCaption(s.blocked ? `Domanda bloccata dal guardrail: «${s.guard.text}»` : `Contesto calcolato: ${s.search.results.length} chunk, ${s.prompt.tokens.total.toLocaleString("it-IT")} token, ${s.calls.length} ${s.calls.length === 1 ? "chiamata" : "chiamate"} all'LLM.`);
      view.focus(DIAGRAM_IDS, { duration: 1.2, margin: 1.05 });
    },
    onSelect: async (item, title) => {
      await tab.views.activate(VIEW_ID);
      scene.setActive(item.stage);
      view.focus([item.stage]);
      view.showCode(item.snippet);
      tab.setCaption(`${title}: ${STAGE[item.stage].blocks}`);
    },
    onPlay: () => ui.player?.start("context-pipeline"),
    onRag: () => openRag(tab, view),
    onShow: () => ui.sidebar.select("contesto"),
  });

  tab.views.register({
    id: VIEW_ID,
    label: "Contesto",
    description: "Dal messaggio dell'utente al contesto dell'LLM: guardrail, RAG con pgvector, prompt, strumenti e streaming.",
    order: 30,
    enter: async () => {
      tab.lod.setDistances(CONTEXT_DISTANCES);
      director.hideFrame();
      pane.setVisible(true);
      // l'architettura resta visibile: la pila del transformer è l'LLM del diagramma
      tab.arch.root.visible = true;
      tab.arch.rail.visible = true;
      // il pacchetto della pila resta fermo all'ingresso finché il contesto non arriva (lo muove l'azione);
      // così non sovrascrive nemmeno la didascalia con le stazioni della pila
      view.lockFlow(true);
      view.moveArchFlow(0);
      scene.setVisible(true);
      offFrame = host.onFrame((dt, elapsed) => scene.tick(dt, elapsed));
      const s = await view.ensure();
      charts.showAll(s);
      ui.sidebar.select("contesto");
      if (!ui.player?.action) {
        tab.setCaption("Contesto: a sinistra della pila, la fase che prepara l'input dell'LLM. Rotellina per lo zoom semantico, oppure avvia l'animazione dal pannello Contesto.");
        // prima la Panoramica (Contesto → LLM → Risposta) con la pila intera; con la rotellina si scende nei dettagli
        view.overview();
        view.showCode("llm");
      }
    },
    exit: () => {
      if (ui.player?.action?.id === "context-pipeline") ui.player.stop();
      pane.setVisible(false);
      scene.setVisible(false);
      scene.setActive(null);
      scene.hidePacket();
      scene.flowTokens(null);
      offFrame?.();
      offFrame = null;
      view.lockFlow(false);
      tab.lod.resetDistances();
    },
  });

  // Il tab RAG (o un altro tab) può passarci la sua domanda prima di aprire questa vista:
  // app.bus.dispatchEvent(new CustomEvent("context:question", { detail: { question } }))
  tab.app.bus.addEventListener("context:question", (e) => {
    const question = String(e.detail?.question ?? "").trim();
    if (!question || question === view.question) return;
    view.question = question;
    if (tab.views.current === VIEW_ID) pane.submit(question);
  });

  registerContextAction(view);
  tab.contextView = { view, unregister }; // per la console e per i test automatici
}

/**
 * Apre il tab "Database vettoriale (RAG)" con la stessa domanda. Il tab si crea alla prima
 * apertura (import dinamico): aspettiamo che esista, poi mandiamo "rag:query".
 */
async function openRag(tab, view) {
  const app = tab.app;
  const detail = { question: view.question, topK: view.scenario?.topK ?? 5, similarityThreshold: view.scenario?.threshold ?? 0.5, source: "transformer" };
  app.open("rag");
  const deadline = performance.now() + RAG_WAIT_MS;
  while (!app.instances?.get("rag")?.instance && performance.now() < deadline) await sleep(50);
  app.bus.dispatchEvent(new CustomEvent("rag:query", { detail }));
}
