/**
 * Azioni "Ragionamento e agenti": catena di pensiero e verifica, RL con ricompense verificabili
 * (GRPO), ciclo agente con tool use, capacità di coding (FIM e feedback di esecuzione).
 */
import { AGENT_LEGS } from "./regionAgent.js";
import { CHAINS, REWARDS } from "./regionReasoning.js";
import { phase } from "./shapes.js";

const CATEGORY = "Ragionamento e agenti";

/** Passi visibili per catena, riusato a ogni frame (niente array nuovi in update). */
const shownSteps = [0, 0, 0, 0];
const setShown = (a, b, c, d) => {
  shownSteps[0] = a;
  shownSteps[1] = b;
  shownSteps[2] = c;
  shownSteps[3] = d;
  return shownSteps;
};

function cotAction(tab) {
  const r = tab.regions.reasoning;
  const all = CHAINS.map((c) => c.steps.length);
  return {
    id: "cot",
    title: "Catena di pensiero e verifica",
    category: CATEGORY,
    summary: "Passi intermedi, più tentativi, un verificatore: spendere più calcolo al momento della risposta.",
    onStart() {
      tab.prepare("reasoning", { charts: "inferenza" });
    },
    onStop() {
      r.setChains(all, true);
      r.verdict(1);
      r.grpo(0);
      tab.plots.passN.setHighlight(null);
    },
    steps: [
      {
        title: "La domanda",
        caption: "Copione didattico. Una domanda con una risposta controllabile: 17 × 24. Rispondere «di getto» con un solo token è facile da sbagliare.",
        duration: 4,
        enter() {
          r.setChains([0, 0, 0, 0], false);
          r.verdict(0);
          r.grpo(0);
          tab.showCode("sim:passAtN");
        },
      },
      {
        title: "Passi intermedi",
        caption: "Chain-of-thought: il modello scrive i passaggi (17 × 20, 17 × 4, la somma). Ogni token di ragionamento è calcolo in più che il modello usa per arrivare alla risposta.",
        duration: 5,
        enter() {
          r.setChains([0, 0, 0, 0], false);
        },
        update(p) {
          r.setChains(setShown(Math.ceil(phase(p, 0.05, 0.8) * 3), 0, 0, 0), false);
        },
      },
      {
        title: "Più tentativi",
        caption: "Si può spendere ancora più calcolo campionando altre catene. Alcune seguono strade diverse e corrette, altre sbagliano un conto o approssimano.",
        duration: 5,
        enter() {
          r.setChains([3, 0, 0, 0], false);
        },
        update(p) {
          r.setChains(setShown(3, Math.ceil(phase(p, 0, 0.4) * 3), Math.ceil(phase(p, 0.3, 0.7) * 3), Math.ceil(phase(p, 0.6, 0.9) * 1)), false);
        },
      },
      {
        title: "Le risposte",
        caption: "Quattro risposte: 408, 398, 408, 420. Quale scegliere? Si può votare (la più frequente) oppure usare un verificatore.",
        duration: 4,
        enter() {
          r.setChains(all, true);
          r.verdict(0);
        },
      },
      {
        title: "Il verificatore",
        caption: "Qui il verificatore è un programma che controlla davvero 17 × 24 = 408. Con N tentativi e un verificatore perfetto la probabilità di trovare la risposta giusta è 1 − (1 − p)^N.",
        duration: 5,
        enter() {
          r.setChains(all, true);
          r.verdict(0);
          tab.openCharts();
          tab.plots.passN.setHighlight({ x: 4 });
        },
        update(p) {
          r.verdict(phase(p, 0.2, 0.5));
        },
      },
      {
        title: "Test-time compute",
        caption: "Ragionare più a lungo, provare più strade, verificare: è il «calcolo al momento dell'inferenza». Secondo Snell et al. (2024), se ben allocato può valere più di un modello più grande su problemi di difficoltà media.",
        duration: 5.5,
        enter() {
          r.setChains(all, true);
          r.verdict(1);
        },
      },
    ],
  };
}

function grpoAction(tab) {
  const r = tab.regions.reasoning;
  const all = CHAINS.map((c) => c.steps.length);
  const advLabels = CHAINS.map((c, i) => `${i + 1}: ${c.answer}`);
  let lastLevel = -1;
  /** Il grafico si aggiorna a 20 livelli di crescita, non a ogni frame. */
  const showAdv = (t) => {
    const level = Math.round(t * 20);
    if (level === lastLevel) return;
    lastLevel = level;
    tab.plots.grpo.setData({ labels: advLabels, values: r.advantages.map((a) => (a * level) / 20) });
  };
  return {
    id: "grpo",
    title: "RL con ricompense verificabili (GRPO)",
    category: CATEGORY,
    summary: "Ricompensa da un verificatore, vantaggio = confronto con la media del gruppo: l'idea di DeepSeek-R1.",
    onStart() {
      tab.prepare("reasoning", { charts: "grpo" });
    },
    onStop() {
      r.grpo(0);
      r.note([{ text: "Più calcolo al momento dell'inferenza: passi intermedi, più tentativi, un verificatore.", bold: true }, { text: "Le catene sono un copione didattico; il verificatore controlla davvero 17 × 24 = 408.", color: "#94a3b8" }]);
    },
    steps: [
      {
        title: "Un gruppo di risposte",
        caption: "Durante l'addestramento, per una domanda si campiona un GRUPPO di risposte (qui 4) dal modello stesso.",
        duration: 4,
        enter() {
          r.setChains(all, true);
          r.verdict(0);
          r.grpo(0);
          showAdv(0);
          tab.showCode("sim:grpoAdvantages");
          r.note([{ text: "GRPO (DeepSeekMath, 2024): niente critic, il termine di paragone è il gruppo.", bold: true }]);
        },
      },
      {
        title: "Ricompense verificabili",
        caption: `Un programma controlla ogni risposta: 1 se è 408, 0 altrimenti. Ricompense: ${REWARDS.join(", ")}. Nessun giudizio umano: per matematica e codice la correttezza si verifica.`,
        duration: 4.5,
        enter() {
          r.verdict(0);
          r.grpo(0);
        },
        update(p) {
          r.verdict(phase(p, 0.1, 0.4));
          r.grpo(phase(p, 0.4, 0.8), 0);
        },
      },
      {
        title: "Vantaggio di gruppo",
        caption: `Vantaggio = (ricompensa − media) / deviazione standard del gruppo. Media ${(REWARDS.reduce((a, b) => a + b, 0) / REWARDS.length).toFixed(2).replace(".", ",")}: le risposte giuste hanno vantaggio positivo, le sbagliate negativo.`,
        duration: 5,
        enter() {
          r.verdict(1);
          r.grpo(1, 0);
          showAdv(0);
        },
        update(p) {
          const t = phase(p, 0.1, 0.7);
          r.grpo(1, t);
          showAdv(t);
        },
      },
      {
        title: "L'aggiornamento",
        caption: "Si aumenta la probabilità dei token delle risposte con vantaggio positivo e si riduce quella delle altre. Ripetuto su moltissime domande, il modello impara a ragionare meglio.",
        duration: 5,
        enter() {
          r.verdict(1);
          r.grpo(1);
          showAdv(1);
          tab.openCharts();
        },
      },
      {
        title: "DeepSeek-R1",
        caption: "DeepSeek-R1-Zero è stato addestrato così, solo con RL e senza SFT iniziale, partendo da DeepSeek-V3-Base: ha sviluppato da solo catene lunghe, verifica e riflessione. R1 aggiunge dati iniziali e altre fasi per leggibilità e allineamento.",
        duration: 6,
        enter() {
          r.verdict(1);
          r.grpo(1);
          r.note([{ text: "R1-Zero: RL puro su DeepSeek-V3-Base. R1: cold start + 2 fasi RL + 2 SFT.", bold: true }, { text: "Fonte: README ufficiale di DeepSeek-R1.", color: "#94a3b8" }]);
        },
      },
    ],
  };
}

const TOOL_EXAMPLE = `{
  "name": "meteo",
  "description": "Previsioni del tempo per una città",
  "parameters": {
    "type": "object",
    "properties": {
      "citta":  { "type": "string" },
      "giorno": { "type": "string", "enum": ["oggi", "domani"] }
    },
    "required": ["citta"]
  }
}`;

function agentAction(tab) {
  const r = tab.regions.agent;
  const captions = [
    "L'utente fa una domanda a cui il modello non può rispondere da solo: le previsioni di domani non sono nei suoi pesi.",
    "Il modello conosce la descrizione dello strumento «meteo» (nel pannello del codice, formato di esempio) e invece di una frase genera una chiamata strutturata: nome e argomenti.",
    "Il programma che usa il modello legge la chiamata ed esegue la funzione vera: un'API, una query al database, un comando. Il modello non esegue nulla da solo.",
    "Lo strumento restituisce un risultato strutturato.",
    "Il risultato viene aggiunto al contesto e il modello riprende a generare. Un agente può fare molti giri: pensa, agisce, osserva (lo schema di ReAct).",
    "Infine il modello risponde a parole, usando dati che non conosceva.",
  ];
  return {
    id: "agent",
    title: "Ciclo agente e tool use",
    category: CATEGORY,
    summary: "Il modello scrive una chiamata, il programma esegue lo strumento, il risultato torna nel contesto.",
    onStart() {
      tab.prepare("agent");
    },
    onStop() {
      r.leg(-1);
    },
    steps: AGENT_LEGS.map((leg, k) => ({
      title: ["Domanda", "Chiamata", "Esecuzione", "Risultato", "Di nuovo al modello", "Risposta"][k],
      caption: captions[k],
      duration: 4.2,
      enter() {
        r.leg(k, 0);
        // la descrizione dello strumento è nel contesto fin dall'inizio: il pannello resta aperto
        tab.showCode({ title: "Esempio: descrizione di uno strumento (formato semplificato)", file: "esempio", code: TOOL_EXAMPLE, language: "json", highlight: k === 1 ? ['"name"', '"required"'] : [] });
      },
      update(p) {
        r.leg(k, phase(p, 0.05, 0.6));
      },
    })),
  };
}

function codingAction(tab) {
  const r = tab.regions.coding;
  return {
    id: "coding",
    title: "Coding: fill-in-the-middle e test",
    category: CATEGORY,
    summary: "Il riordino FIM che insegna a riempire un buco; il ciclo scrivi → esegui i test → correggi.",
    onStart() {
      tab.prepare("coding");
    },
    onStop() {
      r.fim(1);
      r.loop(0, 5);
    },
    steps: [
      {
        title: "Un file con un buco",
        caption: "Nell'editor il cursore sta spesso in mezzo a un file: c'è codice prima (prefisso) e dopo (suffisso). Un modello che scrive solo da sinistra a destra non vede il suffisso.",
        duration: 4.5,
        enter() {
          r.fim(0);
          r.loop(0, 3);
          tab.showCode("sim:fimTransform");
        },
      },
      {
        title: "Il riordino FIM",
        caption: "Fill-in-the-middle (Bavarian et al., 2022): in una parte dei dati di training il centro viene spostato alla fine, con dei segnaposto. Leggendo da sinistra a destra, il modello genera il centro sapendo già il suffisso.",
        duration: 5.5,
        enter() {
          r.fim(0);
        },
        update(p) {
          r.fim(phase(p, 0.1, 0.7));
        },
      },
      {
        title: "Scrivi, esegui, correggi",
        caption: "Il codice si può eseguire: i test dicono cosa non va. Primo tentativo: 3 test su 5. L'errore torna al modello come testo.",
        duration: 5,
        enter() {
          r.fim(1);
          r.loop(0, 3);
        },
        update(p) {
          r.loop(phase(p, 0, 0.95), 3);
        },
      },
      {
        title: "Secondo giro",
        caption: "Il modello legge l'errore, corregge e riprova: 5 test su 5. È un feedback verificabile, usato sia dagli agenti di coding sia nell'addestramento con RL.",
        duration: 5,
        enter() {
          r.loop(0, 3);
        },
        update(p) {
          r.loop(phase(p, 0, 0.95), p > 0.4 ? 5 : 3);
        },
      },
      {
        title: "Misurare: SWE-bench",
        caption: "SWE-bench (Jimenez et al., 2023): 2.294 issue reali da 12 repository Python. Il modello deve modificare il codice finché i test del progetto passano.",
        duration: 5,
        enter() {
          r.loop(0, 5);
        },
      },
    ],
  };
}
export function reasoningActions(tab) {
  return [cotAction(tab), grpoAction(tab), agentAction(tab), codingAction(tab)];
}
