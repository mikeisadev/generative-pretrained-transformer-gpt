/**
 * Azioni "Addestramento": leggi di scala (Chinchilla) e pipeline pre-training → SFT → preferenze.
 */
import { actionInfo } from "./actionCatalog.js";
import { BUDGETS } from "./regionScaling.js";
import { emergenceIllustration } from "./sim.js";
import { phase } from "./shapes.js";

function scalingAction(tab) {
  const r = tab.regions.scaling;
  const fmtC = (C) => `10^${Math.log10(C)} FLOP`;
  return {
    ...actionInfo("scaling"),
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
        caption: "Grafico «Loss prevista»: a parametri fissi la loss si appiattisce (mancano parametri), mentre crescendo insieme N e D continua a scendere. Chinchilla (70B, 1,4T token) ha battuto Gopher (280B) con lo stesso calcolo; questo fit indicherebbe un modello più piccolo, gli altri metodi del paper circa 63-67B.",
        duration: 6,
        enter() {
          r.set(1, -1);
          tab.openCharts();
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
    ...actionInfo("pipeline"),
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

/**
 * Capacità «emergenti» o effetto della metrica? ILLUSTRAZIONE dichiarata dell'argomento di
 * Schaeffer et al. (2023): la curva per token è inventata, la relazione p^10 è esatta.
 */
function emergenceAction(tab) {
  const r = tab.regions.scaling;
  const X_MIN = 7;
  const X_MAX = 13;
  const LEVELS = 60; // il grafico e il testo cambiano a scatti di 0,1 sull'asse, non a ogni frame
  const pct = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0).replace(".", ",")}%`;
  /** Sposta il «siamo qui» al livello k (0..LEVELS) e scrive i due punteggi sotto le barre. */
  const showAt = (k, ctx) => {
    if (ctx.data.level === k) return;
    ctx.data.level = k;
    const x = X_MIN + ((X_MAX - X_MIN) * k) / LEVELS;
    const { perToken, exactMatch } = emergenceIllustration(x);
    tab.plots.emergence.setHighlight({ x });
    r.note([
      { text: `scala (illustrativa) 10^${x.toFixed(1).replace(".", ",")} parametri`, bold: true },
      { text: `per token: ${pct(perToken)}   ·   risposta esatta su 10 token: ${pct(exactMatch)}`, color: "#fde68a", bold: true },
      { text: "Curva per token inventata (illustrazione); p^10 è il calcolo esatto.", color: "#94a3b8" },
    ]);
  };
  const sweep = (title, caption, from, to, duration = 7) => ({
    title,
    caption,
    duration,
    enter(ctx) {
      ctx.data.level = -1;
      showAt(Math.round(from * LEVELS), ctx);
    },
    update(p, dt, ctx) {
      showAt(Math.round((from + (to - from) * phase(p, 0.05, 0.9)) * LEVELS), ctx);
    },
  });
  return {
    ...actionInfo("emergence"),
    onStart() {
      tab.prepare("scaling", { charts: "emergenza" });
    },
    onStop() {
      r.note();
      tab.plots.emergence.setHighlight(null);
    },
    steps: [
      {
        title: "La definizione",
        caption: "Wei et al. (2022) chiamano «emergente» una capacità assente nei modelli piccoli e presente nei grandi, che non si poteva prevedere guardando i piccoli. Esempi citati: aritmetica a più cifre, alcune prove di ragionamento.",
        duration: 5.5,
        enter(ctx) {
          ctx.data.level = -1;
          tab.hideCode();
          r.note([{ text: "Emergente = assente nei piccoli, presente nei grandi, non prevedibile.", bold: true }, { text: "Ma è la capacità a saltare, o il modo in cui la misuriamo?", color: "#94a3b8" }]);
          tab.plots.emergence.setHighlight(null);
        },
      },
      sweep("Per token: crescita dolce", "Illustrazione. Facciamo crescere la scala: la probabilità di azzeccare OGNI singolo token sale piano, senza salti (curva «per token» nel grafico «Stessa capacità, due metriche»).", 0, 0.5),
      sweep("Risposta esatta: il salto", "La risposta conta come giusta solo se TUTTI i 10 token sono giusti: p^10. Resta quasi a zero e poi sale di colpo, anche se p è cresciuta piano. Come un ponte di 10 assi: si attraversa solo se reggono tutte.", 0.5, 1),
      {
        title: "Cosa ne concludiamo",
        caption: "Schaeffer, Miranda e Koyejo (2023): con metriche continue molti salti spariscono. Non vuol dire che i modelli grandi non sappiano fare cose nuove, ma che prima di parlare di emergenza conviene guardare come si misura.",
        duration: 6,
        enter(ctx) {
          ctx.data.level = -1;
          showAt(LEVELS / 2 + 5, ctx);
        },
      },
    ],
  };
}

export function trainingActions(tab) {
  return [scalingAction(tab), pipelineAction(tab), emergenceAction(tab)];
}
