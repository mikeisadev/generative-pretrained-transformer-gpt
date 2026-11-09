/**
 * Fase 2 del Laboratorio pesi: confronto A/B, grafici e benchmark. Raccordo corto che collega i moduli:
 *
 *   comparison.js    il confronto vivo A/B (ricalcolato quando cambiano run o step)         puro
 *   diffMode.js      la differenza A − B nella scena (showValues, veli, chi possiede i valori)
 *   comparePane.js   pannello "Confronto": A/B, confronti pronti, tabella, istogrammi sovrapposti
 *   promptBlock.js   il prossimo carattere su un prompt: probabilità e logits di A e di B
 *   charts.js        i grafici del training, degli snapshot, del prompt e del benchmark
 *   benchmarkPane.js pannello "Benchmark": numeri del Python e calcolo nel browser (browserBench.js,
 *                    runner.js + worker.js, protocol.js), con il controllo di coerenza
 *   search.js        voci della ricerca (soli dati); links.js del tab riconosce le viste ANALYSIS_VIEWS
 *
 * Lo chiama index.js del tab dopo scena e pannelli (lo stato è già inizializzato) e prima di Cervello e
 * Fonti. Contratto (oggetto tab, eventi dello stato, API della scena): lab.md; questa fase: lab-analysis.md.
 *
 * @param {object} tab l'oggetto del Laboratorio (vedi lab.md)
 * @returns {{dispose(): void}}
 */
import { prefersReducedMotion } from "../../../core/dom.js";
import { showToast } from "../../../core/ui/share.js";
import { COURSE_RUN_ID } from "../state.js";
import { createBenchSource } from "./benchData.js";
import { createBenchmarkPane } from "./benchmarkPane.js";
import { installAnalysisCharts } from "./charts.js";
import { createComparePane } from "./comparePane.js";
import { createComparison } from "./comparison.js";
import { ANALYSIS_SOURCES } from "./content.js";
import { createDiffMode } from "./diffMode.js";
import { presets } from "./presets.js";
import { createPromptBlock } from "./promptBlock.js";
import { BenchmarkRunner } from "./runner.js";
import { ANALYSIS_VIEWS } from "./search.js";

const DEFAULT_PROMPT = "ROMEO:\n";

export function installLabAnalysis(tab) {
  const { state, ui } = tab;
  state.enableFeature("compare"); // i pulsanti "B" accanto a ogni run nel pannello Run

  const comparison = createComparison(state, { schedule: (fn) => requestAnimationFrame(fn) });
  const diff = createDiffMode({ scene: tab.scene.scene, state, actionRunning: () => Boolean(ui.player?.action) });
  const bench = createBenchSource(state, { courseId: COURSE_RUN_ID });
  const charts = installAnalysisCharts(tab, { bench });
  const runner = new BenchmarkRunner();
  const offs = [comparison.on((snapshot) => diff.update(snapshot))];

  // pannelli: dopo "Run" (e "Azioni"), prima di "Cervello" e "Fonti"
  const confronto = ui.sidebar.addPane({ id: "confronto", title: "Confronto", hint: "A (la scena) contro B: tensore per tensore, la differenza nella scena, le distribuzioni, il prossimo carattere." });
  const benchmark = ui.sidebar.addPane({ id: "benchmark", title: "Benchmark", hint: "Lo stesso metro per tutti i run: i numeri del Python e quelli ricalcolati qui dal motore JS." });
  const compare = createComparePane(confronto, { tab, comparison, diff, onPreset: (p) => applyPreset(p), onDiffToggle: () => diff.toggle() && revealDiff() });
  const prompt = createPromptBlock(compare.promptSlot, { tab, charts, defaultPrompt: state.builtin?.trace?.recorder?.example_prompt ?? DEFAULT_PROMPT });
  const benchPane = createBenchmarkPane(benchmark, { tab, bench, charts, runner });

  /**
   * La differenza si vede solo nel Dettaglio (da lontano le matrici sono meno di un pixel e la scena
   * mostra i moduli): accendendola, se la camera è lontana vola sul primo blocco, con tutte le sue matrici.
   */
  function revealDiff() {
    if (tab.scene.lod.current === "detail") return;
    ui.director.noteUserIntent();
    const S = tab.scene.scene;
    if (S.nodeOf("blocks.0")) return void tab.scene.focusModule("blocks.0");
    const top = comparison.current.result?.rows.find((r) => Number.isFinite(r.relChange) && S.nodeOf(r.name));
    if (top) void tab.scene.focusTensor(top.name);
  }

  /** Un confronto pronto: A e B come chiede, poi la differenza nella scena. */
  async function applyPreset(p) {
    ui.director.noteUserIntent(); // la camera inquadra il run A se cambia
    const okA = await state.setRunA(p.a.run, { step: p.a.step });
    const okB = okA && (await state.setRunB(p.b.run, { step: p.b.step }));
    if (!okB) return false;
    await comparison.refresh();
    if (diff.setEnabled(true)) revealDiff();
    showToast(`${p.title}: A − B nella scena.`);
    return true;
  }

  /** Il preset con quell'id, calcolato su ciò che c'è ora nel registro (null se non ha senso). */
  const presetById = (id) => presets({ runA: state.runA, courseId: COURSE_RUN_ID, stepsOf: (x) => state.stepsOf(x), entries: state.entries }).find((p) => p.id === id) ?? null;

  // la differenza si spegne quando parte un'azione: le azioni animano i pesi veri del run A
  if (ui.player) offs.push(ui.player.on("state", () => ui.player.action && diff.stopForAction()));

  // Prima e dopo, senza un clic in più: caricato un run (che diventa A), B diventa il GPT del corso
  let justAdded = null;
  offs.push(
    state.on("runs", ({ reason, added }) => {
      if (reason === "add" && added?.length) justAdded = added[0];
    }),
    state.on("runA", ({ id }) => {
      if (id !== justAdded) return;
      justAdded = null;
      if (state.runB || id === COURSE_RUN_ID) return;
      void state.setRunB(COURSE_RUN_ID).then((ok) => ok && showToast(`Confronto pronto: «${state.entry(id)?.name ?? ""}» (A) contro il GPT del corso (B).`, { duration: 4200 }));
    }),
  );

  // ---------------------------------------------------------------- viste (#lab/go=view:<id>, ricerca)
  const open = (id, then) => () => {
    tab.openPane(id);
    then?.();
    return true;
  };
  const focusLater = (node) =>
    requestAnimationFrame(() => {
      node?.scrollIntoView?.({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
      node?.focus?.({ preventScroll: true });
    });
  const views = {
    confronto: open("confronto"),
    differenza: async () => {
      tab.openPane("confronto");
      if (!state.runB) {
        const p = presetById("training");
        return p ? applyPreset(p) : false;
      }
      await comparison.refresh();
      if (diff.setEnabled(true)) revealDiff();
      return true;
    },
    probabilita: open("confronto", () => focusLater(prompt.input)),
    grafici: () => {
      charts.reveal("training");
      return true;
    },
    benchmark: open("benchmark"),
    "benchmark:browser": open("benchmark", () => focusLater(benchmark.querySelector(".lab-an-buttons button:not([hidden])"))),
  };
  for (const id of ["training", "fasi", "corso"]) {
    views[`confronto:${id}`] = async () => {
      tab.openPane("confronto");
      const p = presetById(id);
      if (!p) {
        showToast(id === "corso" ? "Carica prima un run (pannello Run): poi lo confronto con il GPT del corso." : "Questo confronto ha bisogno di un run con gli snapshot del training.", { duration: 4200 });
        return false;
      }
      return applyPreset(p);
    };
  }
  for (const [id, title] of Object.entries(ANALYSIS_VIEWS)) tab.views.register(id, { title, open: views[id] });

  tab.sources.push(ANALYSIS_SOURCES);
  tab.analysis = { comparison, diff, charts, runner, bench, compare, prompt, benchmark: benchPane, applyPreset }; // console e prove

  return {
    dispose() {
      for (const off of offs) off();
      prompt.dispose();
      compare.dispose();
      benchPane.dispose();
      runner.dispose();
      diff.dispose();
      charts.dispose();
      comparison.dispose();
      // la Sidebar del nucleo non toglie i pannelli: li svuotiamo e li nascondiamo
      for (const [id, pane] of [["confronto", confronto], ["benchmark", benchmark]]) {
        pane.replaceChildren();
        ui.sidebar.setPaneHidden?.(id, true);
      }
    },
  };
}
