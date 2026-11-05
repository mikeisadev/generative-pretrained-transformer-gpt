/**
 * Azioni animate del tab Transformer: "Prompt (inferenza)" e "Training", con i loro grafici.
 *
 *   pane.js          pannello "Azioni" della sidebar (form del prompt, impostazioni, formule, fonti)
 *   stage.js         regia nella scena: lavagna, camera, pacchetto, ripristino dello stato
 *   promptAction.js  "Dal prompt all'output"              (dati: promptRun.js → motore JS)
 *   trainingStep.js  "Uno step di training al microscopio" (dati: trainingData.js → trace + snapshot)
 *   trainingRun.js   "Il training completo in un minuto"
 *   promptCharts.js / trainingCharts.js   sezioni "Prompt" e "Training" del ChartsPanel
 *
 * Regola comune: ogni azione, alla fine o quando viene interrotta, rimette la scena com'era
 * (pesi addestrati, pattern d'esempio, barre d'esempio, flusso automatico, livello di zoom).
 */
import { ActionsPane } from "./pane.js";
import { createPromptAction } from "./promptAction.js";
import { PromptCharts } from "./promptCharts.js";
import { Stage } from "./stage.js";
import { TrainingCharts } from "./trainingCharts.js";
import { TrainingData } from "./trainingData.js";
import { createTrainingRunAction } from "./trainingRun.js";
import { createTrainingStepAction } from "./trainingStep.js";

/**
 * Se la preparazione di un'azione fallisce (rete, dati mancanti, prompt non valido) l'azione
 * mostra un solo passo con il messaggio, invece di lasciare un errore in console.
 */
function safely(action, stage, tab) {
  const build = action.build;
  return {
    ...action,
    // interrotta dall'utente: la scena è già ripristinata, la camera torna alla vista d'insieme
    onStop(ctx) {
      action.onStop?.(ctx);
      tab.focus("overview");
    },
    async build(ctx) {
      try {
        return await build(ctx);
      } catch (error) {
        console.warn(`Azione "${action.title}":`, error);
        stage.restore();
        return [{ title: "Impossibile avviare l'azione", caption: error.message, duration: 6 }];
      }
    },
  };
}

export function installActions(tab) {
  const { ui } = tab;
  if (!ui.player) return;
  const stage = new Stage(tab);
  const { n_layer: layers, n_head: heads } = tab.spec.config;
  const promptCharts = ui.charts ? new PromptCharts(ui.charts, { layers, heads }) : null;
  const trainingCharts = ui.charts ? new TrainingCharts(ui.charts) : null;
  const data = new TrainingData(() => tab.getEngine());

  // La storia del training serve ai grafici e alla scelta dello step: la carichiamo solo quando
  // serve davvero (pannello Azioni, pannello Grafici o avvio di un'azione), una volta sola.
  let trainingReady = null;
  const ensureTraining = () =>
    (trainingReady ??= Promise.all([data.trace(), tab.getEngine()])
      .then(([trace, engine]) => {
        pane.setSnapshotSteps(trace.snapshots.map((s) => s.step));
        if (trainingCharts) {
          trainingCharts.setTrace(trace);
          trainingCharts.setSnapshotWeights(trace.snapshots.at(-1).step, new Map([...engine.paramViews].map(([n, p]) => [n, p.data])));
        }
      })
      .catch((error) => {
        trainingReady = null;
        console.warn("Storia del training non disponibile:", error);
      }));

  const pane = new ActionsPane(tab, { onNeedTraining: ensureTraining });
  ui.charts?.onResize(() => ui.charts.isOpen && ensureTraining());

  const chartsOrNoop = (charts) => charts ?? new Proxy({}, { get: () => () => {} });
  const prompt = createPromptAction({
    tab,
    stage,
    charts: chartsOrNoop(promptCharts),
    getOptions: () => pane.promptOptions(),
    onOutput: (run, count) => pane.showOutput(run, count),
  });
  const step = createTrainingStepAction({ tab, stage, charts: chartsOrNoop(trainingCharts), data, getStartStep: () => pane.startStepValue() });
  const run = createTrainingRunAction({ tab, stage, charts: chartsOrNoop(trainingCharts), data });
  for (const action of [prompt, step, run]) {
    const wrapped = safely(action, stage, tab);
    if (action.category === "Training") {
      const build = wrapped.build;
      wrapped.build = async (ctx) => (await ensureTraining(), build(ctx));
    }
    ui.player.register(wrapped);
  }
  pane.render(true);
}
