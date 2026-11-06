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
import { registerContextAction } from "./action.js";
import { ContextCharts } from "./contextCharts.js";
import { CONTEXT_DISTANCES, DIAGRAM_IDS, STACK, overviewBox, slotsBox } from "./layout.js";
import { ContextPane } from "./pane.js";
import { DEMO_QUESTION, createContextModel } from "./pipeline.js";
import { ContextScene, FULL_STATE } from "./scene.js";
import { SNIPPETS, showSnippet } from "./snippets.js";
import { STAGE } from "./texts.js";

const VIEW_ID = "contesto";
const FRONT = new THREE.Vector3(0.03, 0.07, 1).normalize(); // quasi di fronte: le schede restano leggibili
/**
 * Rapporto larghezza/altezza della parte visibile dello stage per cui sono tarate le soglie del LOD
 * (1600×900 con il pannello del codice aperto). Su uno stage più stretto la camera deve stare più
 * lontana per inquadrare la stessa scheda: le soglie crescono in proporzione, così cliccare una fase
 * porta al Dettaglio anche su un telefono.
 */
const REFERENCE_ASPECT = 0.76;
/** Margine oltre la soglia dei Blocchi per la vista d'insieme (l'isteresi del LOD è dell'8%). */
const MACRO_MARGIN = 1.12;

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
  let leaving = false; // true mentre la vista si chiude: l'azione interrotta non tocca la didascalia
  let lodScale = 0;

  /** Soglie del LOD scalate sulla forma dello stage (vedi REFERENCE_ASPECT); ricalcolate solo se cambia. */
  const fitLodDistances = () => {
    const scale = THREE.MathUtils.clamp(REFERENCE_ASPECT / host.visibleAspect(), 1, 3);
    if (Math.abs(scale - lodScale) < 0.01) return;
    lodScale = scale;
    tab.lod.setDistances({ detail: CONTEXT_DISTANCES.detail * scale, blocks: CONTEXT_DISTANCES.blocks * scale });
  };

  /** Pixel coperti in basso nello stage da un elemento (0 se nascosto o assente). */
  const coveredBelow = (element, stageRect) => {
    if (!element || element.hidden) return 0;
    const r = element.getBoundingClientRect();
    return r.height > 0 ? THREE.MathUtils.clamp(stageRect.bottom - r.top, 0, stageRect.height * 0.7) : 0;
  };

  /**
   * La barra del player (durante un'azione) e, su schermi stretti, il pannello del codice coprono il
   * fondo dello stage. Allunghiamo il riquadro da inquadrare verso il basso della quantità giusta
   * perché il soggetto resti nella parte alta, quella visibile.
   *   altezza inquadrata nel mondo  H = max(h · S/(S − c), w / aspetto)   (S, c: pixel dello stage e coperti)
   *   allungamento                  e = H · c / S
   */
  const roomForOverlays = (box) => {
    const stageRect = host.container.getBoundingClientRect();
    // su schermi larghi il pannello del codice sta a destra (ne tiene conto il view offset della camera)
    const narrow = window.matchMedia("(max-width: 900px)").matches;
    const code = ui.codePanel?.root;
    const codeBelow = narrow && code && !code.classList.contains("minimized") ? code : null;
    const covered = Math.max(coveredBelow(ui.player?.action ? ui.player.bar : null, stageRect), coveredBelow(codeBelow, stageRect));
    if (covered <= 0 || stageRect.height <= covered) return box;
    const w = box.max.x - box.min.x;
    const h = box.max.y - box.min.y;
    const span = Math.max((h * stageRect.height) / (stageRect.height - covered), w / host.visibleAspect());
    box.min.y -= (span * covered) / stageRect.height;
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
      return director.focus(roomForOverlays(slotsBox(ids)), { showFrame: false, margin, direction: FRONT, duration, minDistance: 8 });
    },

    /** Inquadra la scheda dei token, il riquadro LLM e la metà bassa della pila (dove entrano i token). */
    focusLlm() {
      const box = slotsBox(["tokenize", "llm"]);
      box.expandByPoint(new THREE.Vector3(STACK.rightX, -46, 0)).expandByPoint(new THREE.Vector3(STACK.rightX, 0, 0));
      return director.focus(roomForOverlays(box), { showFrame: false, margin: 1.05, direction: FRONT, duration: 1.3 });
    },

    /** Contesto → pila → Risposta, abbastanza da lontano da essere sempre nel livello Panoramica. */
    overview() {
      fitLodDistances();
      const margin = 1.02;
      const minDistance = (tab.lod.controller.levels[1].maxDistance * MACRO_MARGIN) / margin;
      return director.focus(roomForOverlays(overviewBox()), { showFrame: false, margin, minDistance, direction: FRONT, duration: 1.6 });
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

    /** true mentre la vista si sta chiudendo (l'azione interrotta non deve riscrivere la didascalia). */
    get leaving() {
      return leaving;
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
      // come i passi del Percorso: la scheda si inquadra da vicino, quindi lo zoom torna su Auto anche se
      // lo studente aveva bloccato "Panoramica" (durante un'azione il livello lo decide l'azione)
      if (!ui.player?.action) tab.lod.controller.setMode("auto");
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
    // "Centra" e "Vista d'insieme": la Panoramica Contesto → LLM → Risposta, come all'apertura della vista
    home: () => view.overview(),
    enter: async () => {
      lodScale = 0;
      fitLodDistances();
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
        tab.setCaption("Contesto: a sinistra della pila, la fase che prepara l'input dell'LLM. Doppio clic su una fase per avvicinarti (lo zoom semantico mostra i dettagli), oppure avvia l'animazione dal pannello Contesto.");
        // prima la Panoramica (Contesto → LLM → Risposta) con la pila intera; con la rotellina si scende nei dettagli
        view.overview();
        view.showCode("llm");
      }
    },
    exit: () => {
      leaving = true;
      if (ui.player?.action?.id === "context-pipeline") ui.player.stop();
      leaving = false;
      pane.setVisible(false);
      scene.setVisible(false);
      scene.setActive(null);
      scene.hidePacket();
      scene.flowTokens(null);
      offFrame?.();
      offFrame = null;
      view.lockFlow(false);
      tab.lod.resetDistances();
      tab.setCaption(""); // le didascalie del Contesto non valgono per la vista successiva
      // la camera guardava il diagramma a sinistra della pila, ora nascosto: torna sulla pila
      // (se la vista successiva sceglie un'inquadratura sua, il suo volo sostituisce questo)
      tab.focus("overview");
    },
  });

  // Le soglie del LOD seguono la forma dello stage: finestra ridimensionata, pannelli aperti o chiusi.
  const refitLod = () => {
    if (tab.views.current === VIEW_ID) fitLodDistances();
  };
  new ResizeObserver(refitLod).observe(host.container);
  ui.codePanel?.onResize(refitLod);
  ui.charts?.onResize?.(refitLod);

  // Il tab RAG (o un altro tab) può passarci la sua domanda prima di aprire questa vista:
  // app.bus.dispatchEvent(new CustomEvent("context:question", { detail: { question } }))
  tab.app.bus.addEventListener("context:question", (e) => {
    const question = String(e.detail?.question ?? "").trim();
    if (!question || question === view.question) return;
    view.question = question;
    if (tab.views.current === VIEW_ID) pane.submit(question);
  });

  registerContextAction(view);
  tab.contextView = {
    view,
    unregister,
    /**
     * Apre la vista e un passo del suo percorso (ricerca globale, link #transformer/go=view:ctx.guard).
     * @returns {Promise<boolean>} false se il passo non esiste
     */
    async reveal(stepId) {
      await view.ensure(); // il menu del percorso nasce con il primo scenario
      // un passo che non esiste non cambia la vista: lo studente resta dov'era e il guscio lo avvisa
      if (!pane.menu?.items.has(stepId)) return false;
      await tab.views.activate(VIEW_ID);
      return pane.menu.reveal(stepId, { sidebar: ui.sidebar });
    },
  }; // anche per la console e per i test automatici
}

/**
 * Apre il tab "Database vettoriale (RAG)" con la stessa domanda: app.open attende che il tab
 * esista (si crea alla prima apertura), poi mandiamo "rag:query".
 */
async function openRag(tab, view) {
  const app = tab.app;
  const detail = { question: view.question, topK: view.scenario?.topK ?? 5, similarityThreshold: view.scenario?.threshold ?? 0.5, source: "transformer" };
  await app.open("rag"); // si risolve quando il tab (caricato alla prima apertura) è pronto
  app.bus.dispatchEvent(new CustomEvent("rag:query", { detail }));
}
