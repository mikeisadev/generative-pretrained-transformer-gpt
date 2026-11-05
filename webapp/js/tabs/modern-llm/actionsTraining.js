/**
 * Azioni "Addestramento": leggi di scala (Chinchilla) e pipeline pre-training → SFT → preferenze.
 */
import { BUDGETS } from "./regionScaling.js";
import { phase } from "./shapes.js";

const CATEGORY = "Addestramento";

function scalingAction(tab) {
  const r = tab.regions.scaling;
  const fmtC = (C) => `10^${Math.log10(C)} FLOP`;
  return {
    id: "scaling",
    title: "Leggi di scala: quanti parametri, quanti dati",
    category: CATEGORY,
    summary: "Con la formula di Chinchilla: per ogni budget di calcolo, N e D ottimali e la loss prevista.",
    onStart() {
      tab.prepare("scaling", { charts: "scala" });
    },
    onStop() {
      r.set(1, -1);
      tab.plots.chinchilla.setHighlight(null);
      tab.plots.alloc.setHighlight(null);
    },
    steps: [
      {
        title: "Il budget",
        caption: "Il costo di un training si misura in FLOP: circa 6 · N · D (N parametri, D token). Con un budget fissato, meglio un modello più grande o più dati?",
        duration: 5,
        enter() {
          r.set(0, -1);
          tab.showCode("sim:chinchillaOptimal");
        },
        update(p) {
          r.set(phase(p, 0.3, 1), -1);
        },
      },
      ...BUDGETS.map((C, k) => ({
        title: fmtC(C),
        caption: `Budget ${fmtC(C)}: il fit di Chinchilla suggerisce circa ${r.values[k].N.toExponential(1).replace(".", ",")} parametri e ${r.values[k].D.toExponential(1).replace(".", ",")} token, con loss prevista ${r.values[k].loss.toFixed(2).replace(".", ",")}. Ogni ×10 di calcolo: N circa ×2,8 e D circa ×3,5.`,
        duration: 3.6,
        enter() {
          r.set(1, k);
          tab.plots.chinchilla.setHighlight({ x: Math.log10(C) });
          tab.plots.alloc.setHighlight({ x: Math.log10(C) });
        },
      })),
      {
        title: "La lezione di Chinchilla",
        caption: "Grafico «Loss prevista»: a parametri fissi la loss si appiattisce (mancano parametri), mentre crescendo insieme N e D continua a scendere. Chinchilla (70B, 1,4T token) ha battuto Gopher (280B) con lo stesso calcolo.",
        duration: 6,
        enter() {
          r.set(1, -1);
          tab.ui.charts.open();
          tab.plots.chinchilla.setHighlight({ x: Math.log10(5.76e23) });
        },
      },
    ],
  };
}

function pipelineAction(tab) {
  const r = tab.regions.training;
  const stage = (k, title, caption, code = null) => ({
    title,
    caption,
    duration: 5,
    enter() {
      r.setStage(k, 0);
      r.method(-1);
      if (code) tab.showCode(code);
    },
    update(p) {
      r.setStage(k, phase(p, 0, 0.6));
    },
  });
  const method = (i, title, caption, code) => ({
    title,
    caption,
    duration: 5.5,
    enter() {
      r.setStage(2, 1);
      r.method(i);
      if (code) tab.showCode(code);
    },
  });
  return {
    id: "pipeline",
    title: "Pre-training → SFT → preferenze",
    category: CATEGORY,
    summary: "Le tre fasi che trasformano un modello che continua il testo in un assistente; RLHF, DPO e Constitutional AI.",
    onStart() {
      tab.prepare("training");
    },
    onStop() {
      r.setStage(-1);
      r.method(-1);
    },
    steps: [
      stage(0, "Pre-training", "Fase 1, come il nostro GPT: prevedere il prossimo token, ma su trilioni di token (Llama 3: oltre 15T; DeepSeek-V3: 14,8T). Ne esce un modello base: sa continuare un testo, non conversare.", "ours:block"),
      stage(1, "SFT", "Fase 2, fine-tuning supervisionato: si continua l'addestramento su dialoghi esempio (domanda, risposta ideale). Il modello impara il formato e il tono di un assistente."),
      stage(2, "Preferenze", "Fase 3: per la stessa domanda si confrontano due risposte e si indica la migliore. Il modello viene spinto verso le risposte preferite."),
      method(0, "RLHF", "RLHF (InstructGPT): un reward model impara a imitare i giudizi umani, poi si ottimizza il modello con PPO senza allontanarlo troppo dall'originale. InstructGPT da 1,3B era preferito a GPT-3 da 175B.", null),
      method(1, "DPO", "DPO: stesso obiettivo, senza reward model e senza RL. Una loss di classificazione direttamente sulle coppie: aumenta la probabilità della risposta scelta rispetto a quella scartata.", "dpo.loss"),
      method(2, "Constitutional AI", "Constitutional AI (Anthropic): i giudizi vengono da un'AI che segue una lista di principi. Prima il modello critica e rivede le sue risposte, poi RL con feedback dell'AI (RLAIF)."),
      {
        title: "Il risultato",
        caption: "Un assistente: stesse equazioni del nostro GPT, ma addestrato in tre fasi. Mixtral 8x7B Instruct, per esempio, è dichiarato SFT + DPO.",
        duration: 4.5,
        enter() {
          r.setStage(2, 1);
          r.method(-1);
          r.note([{ text: "Modello base → assistente: cambia l'addestramento, non l'architettura.", bold: true }, { text: "Le fasi di ragionamento con RL verificabile sono nella regione «Ragionamento».", color: "#94a3b8" }]);
        },
        exit() {
          r.note([{ text: "Il nostro GPT si ferma alla prima fase: è un modello base che continua il testo.", bold: true }, { text: "Gli assistenti (ChatGPT, Claude, Llama Instruct…) passano anche dalle altre due.", color: "#94a3b8" }]);
        },
      },
    ],
  };
}

export function trainingActions(tab) {
  return [scalingAction(tab), pipelineAction(tab)];
}
