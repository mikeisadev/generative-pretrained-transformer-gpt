/**
 * Fase 2 del Laboratorio pesi: le azioni animate (player del nucleo) di come i pesi interagiscono, con
 * i pesi e le attivazioni VERI del run A al suo step, sulle matrici della scena.
 *
 *   catalog.js     schede delle azioni (soli dati, anche per search.js)
 *   settings.js    prompt, token, blocco, head, nn.Linear scelti dallo studente (puro)
 *   data.js        il passaggio in avanti del motore JS ritagliato per le azioni (puro)
 *   math.js        ogni passo dei calcoli, con le stesse operazioni del motore (puro)
 *   training.js    passo di training (gradiente di lm_head, SGD, AdamW) e storia del training (puro)
 *   plan.js        dove mettere le carte e come inquadrarle (puro)
 *   stage.js       regia: carte, veli, linee, lavagna, codice, camera e ritorno alla scena di prima
 *   cards.js, board.js, stepKit.js, linearKit.js, format.js   pezzi della regia
 *   *Action.js     le azioni (un modulo per famiglia)
 *   pane.js        il pannello "Azioni" della sidebar
 *
 * Regola comune: ogni azione, chiusa o interrotta, rimette la scena com'era (pesi e valori mostrati,
 * step del training, camera, livello dello zoom, codice) e libera tutto ciò che ha creato.
 */
import { installStepKeys } from "../../../core/stepKeys.js";
import { incompatibility } from "../state.js";
import { ACTIONS, CATEGORIES } from "./catalog.js";
import { forwardData } from "./data.js";
import { attentionSteps, attentionValueSteps } from "./attentionAction.js";
import { ffnSteps } from "./ffnAction.js";
import { historySteps } from "./historyAction.js";
import { layerNormSteps } from "./layerNormAction.js";
import { linearSteps } from "./linearAction.js";
import { logitsSteps } from "./logitsAction.js";
import { createActionsPane } from "./pane.js";
import { ActionSettings, checkPrompt } from "./settings.js";
import { ActionStage } from "./stage.js";
import { makeSteps } from "./stepKit.js";
import { trainingStepSteps } from "./trainingAction.js";

const narrow = window.matchMedia("(max-width: 900px)");

/** Le azioni del passaggio in avanti: dai dati del motore ai passi dichiarati. */
const FORWARD = {
  linear: (stage, f, settings) => linearSteps(stage, f, settings.linear),
  attention: (stage, f) => attentionSteps(stage, f),
  "attention-v": (stage, f) => attentionValueSteps(stage, f),
  layernorm: (stage, f) => layerNormSteps(stage, f),
  ffn: (stage, f) => ffnSteps(stage, f),
  logits: (stage, f) => logitsSteps(stage, f),
};

/**
 * @param {object} tab l'oggetto del Laboratorio (lab.md)
 * @returns {{dispose(): void, stage: ActionStage, settings: ActionSettings}|void}
 */
export function installLabActions(tab) {
  const { ui, state } = tab;
  if (!ui.player) return;
  const settings = new ActionSettings();
  const stage = new ActionStage(tab);
  const pane = ui.sidebar.addPane({ id: "azioni", title: "Azioni", hint: "Animazioni dei calcoli con i numeri veri: x · Wᵀ, attention, LayerNorm, feed-forward, logits, un passo di training, la storia dei pesi." });
  const panel = createActionsPane(pane, tab, settings);
  const own = new Set(ACTIONS.map((a) => a.id));

  for (const card of ACTIONS) ui.player.register(wrap(card, { tab, stage, settings }));
  ui.player.renderOverview(panel.overview, { categories: panel.categories });

  // Se lo studente cambia run o step mentre un'azione gira, i numeri dell'azione non sono più quelli
  // della scena: l'azione si chiude (e rimette tutto com'era). Le modifiche dell'azione stessa no.
  const mine = () => stage.active && own.has(ui.player.action?.id);
  const offs = [
    state.on("runA", () => mine() && ui.player.stop()),
    state.on("step", () => mine() && !stage.ownStepChange && ui.player.stop()),
  ];

  // Spazio = pausa / riprendi l'azione (il tasto del player di tutti i tab; il Laboratorio non ha un Percorso)
  const offKeys = installStepKeys({ root: ui.root, ui, menu: () => null });

  tab.actions = { stage, settings, panel }; // per la console e gli script di verifica (window.__labTab.actions)
  return {
    stage,
    settings,
    dispose() {
      offKeys();
      for (const off of offs) off();
      panel.dispose();
      stage.dispose();
    },
  };
}

/** Un'azione del catalogo per il player: inizio e fine della regia, preparazione dei dati veri, errori leggibili. */
function wrap(card, { tab, stage, settings }) {
  return {
    id: card.id,
    title: card.title,
    category: CATEGORIES[card.category],
    summary: card.summary,
    onStart() {
      stage.begin();
      // sul telefono la sidebar copre la scena: l'azione si guarda a tutto schermo
      if (narrow.matches) tab.ui.sidebar.toggleCollapsed(true);
    },
    async build(ctx) {
      const t0 = performance.now();
      try {
        const specs = card.engine ? await forwardBuild(card, { tab, stage, settings }) : await historySteps(stage, tab, settings);
        if (ctx.isStale()) return [];
        stage.stats.builds.push({ id: card.id, ms: performance.now() - t0 });
        return makeSteps(stage, specs, { id: card.id });
      } catch (error) {
        if (ctx.isStale()) return [];
        console.warn(`Azione "${card.title}":`, error);
        return makeSteps(stage, [problemStep("Impossibile avviare l'azione", error.message)], { id: card.id });
      }
    },
    onStop: () => stage.end({ flyBack: true }),
  };
}

/** I passi di un'azione che usa il motore JS (attivazioni vere), o il passo che spiega perché non si può. */
async function forwardBuild(card, { tab, stage, settings }) {
  const { state } = tab;
  const run = await state.loadRun(state.runA);
  if (!run.compatibleWithEngine) return [notRunnable(run)];
  const { engine } = await tab.engineFor(state.runA, state.step);
  const check = checkPrompt(settings.prompt, engine.vocab, engine.config.block_size);
  if (!check.ok) return [problemStep("Il prompt non va", `${check.message} Correggilo nel pannello Azioni.`)];
  const { block, head } = settings.clamp(engine.config);
  if (card.id === "training-step") {
    const steps = trainingStepSteps(stage, { tab, engine, ids: check.ids, token: settings.token, run });
    return steps ?? [problemStep("Serve un prompt più lungo", "Un esempio di training è un contesto più il carattere che viene dopo: scrivi almeno 2 caratteri.")];
  }
  const f = forwardData(engine, check.ids, { token: settings.token, block, head });
  const steps = FORWARD[card.id](stage, f, settings);
  return steps ?? [problemStep("La scena non disegna questi pesi", "Le matrici di questo run sono troppo grandi per la scena (i numeri restano nel pannello Tensore).")];
}

/** Il passo unico per i run che il motore JS non esegue: perché, e che cosa si può fare. */
function notRunnable(run) {
  const reason = incompatibility(run);
  return {
    title: "Questo run non si può eseguire qui",
    caption: `Il motore JS non esegue «${run.name}»: ${reason}. Senza il motore le attivazioni andrebbero inventate, e qui non si inventa.`,
    duration: 8,
    board: () => [
      { title: "Niente attivazioni inventate", kicker: "Perché" },
      { line: [`Il motore JS esegue solo i run con gli stessi moduli e lo stesso calcolo del GPT del corso. Questo run: ${reason}.`] },
      { line: ["I suoi pesi restano tutti da esplorare: nella scena (celle e numeri), nel pannello Tensore, nel confronto con il GPT del corso. «I pesi durante il training» funziona se il run ha gli snapshot."] },
      { note: "Per vedere i calcoli: scegli il GPT del corso (pannello Run) o un run compatibile." },
    ],
  };
}

function problemStep(title, message) {
  return { title, caption: message, duration: 6, board: () => [{ title, kicker: "Azione" }, { line: [message] }] };
}
