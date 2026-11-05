/**
 * Azioni "Architettura" (seconda parte, il lato inferenza): tokenizer BPE, KV cache,
 * FlashAttention e finestra scorrevole, decodifica speculativa.
 */
import { FLASH_N, FLASH_TILE } from "./regionFlash.js";
import { SPEC_ROUNDS } from "./regionSpec.js";
import { tokenText } from "./sim.js";
import { phase } from "./shapes.js";

const CATEGORY = "Architettura";
const shown = (s) => s.replace(/ /g, "␣");

function bpeAction(tab) {
  const r = tab.regions.bpe;
  return {
    id: "bpe",
    title: "Tokenizer BPE a byte, fusione dopo fusione",
    category: CATEGORY,
    summary: "Algoritmo vero: dai byte UTF-8 si fonde ogni volta la coppia più frequente.",
    onStart() {
      tab.prepare("bpe");
    },
    onStop() {
      r.show(r.history.length - 1);
    },
    build: async () => [
      {
        title: "Si parte dai byte",
        caption: "La frase diventa la sequenza dei suoi byte UTF-8: 256 simboli possibili. Le lettere comuni sono un byte; «È» ne usa due. Nessun testo può contenere un simbolo sconosciuto.",
        duration: 5,
        enter() {
          r.show(0);
          tab.showCode("minbpe.train");
        },
      },
      ...r.history.slice(1).map((step, i) => {
        const k = i + 1;
        const a = shown(tokenText(r.vocab.get(step.pair[0])));
        const b = shown(tokenText(r.vocab.get(step.pair[1])));
        return {
          title: `Fusione ${k}`,
          caption: `La coppia più frequente è «${a}» + «${b}» (${step.count} volte): diventa il token ${step.idx}. Ogni fusione aggiunge una voce al vocabolario e accorcia la sequenza.`,
          duration: k <= 2 ? 3.6 : 2.4,
          enter() {
            r.show(k, true);
          },
          update(p) {
            r.show(k, p < 0.5);
          },
        };
      }),
      {
        title: "Il risultato",
        caption: `Dopo ${r.history.length - 1} fusioni la frase è passata da ${r.history[0].ids.length} a ${r.history.at(-1).ids.length} token: le parti ripetute, come «␣gatt», sono diventate un token solo. I tokenizer veri fanno decine di migliaia di fusioni su moltissimo testo.`,
        duration: 5,
        enter() {
          r.show(r.history.length - 1);
        },
      },
    ],
  };
}

function kvAction(tab) {
  const r = tab.regions.kv;
  const N_NO = 6;
  const work = (n) => ({
    series: [
      { name: "senza cache", points: Array.from({ length: n }, (_, k) => [k + 1, ((k + 1) * (k + 2)) / 2]) },
      { name: "con cache", points: Array.from({ length: n }, (_, k) => [k + 1, k + 1]) },
    ],
  });
  const step = (n, cache, caption) => ({
    title: `${cache ? "Con" : "Senza"} cache · token ${n}`,
    caption,
    duration: cache ? 1.3 : 2,
    enter() {
      r.set(n, cache, 0);
      tab.plots.kvWork.setData(work(n));
    },
    update(p) {
      r.set(n, cache, Math.sin(Math.PI * Math.min(1, p * 1.4)));
    },
  });
  return {
    id: "kv",
    title: "KV cache: token dopo token",
    category: CATEGORY,
    summary: "Senza cache si ricalcola tutto il prefisso a ogni token; con la cache solo il token nuovo.",
    onStart() {
      tab.prepare("kv", { charts: "kv" });
    },
    onStop() {
      r.set(r.T, true);
    },
    steps: [
      {
        title: "Come genera il nostro GPT",
        caption: "Il nostro generate() rimette nel modello tutto il contesto per ogni nuovo carattere. Colonne = token, righe = layer; in alto le K, in basso le V.",
        duration: 4.5,
        enter() {
          r.set(1, false);
          tab.plots.kvWork.setData(work(1));
          tab.showCode("ours:generate");
        },
      },
      ...Array.from({ length: N_NO - 1 }, (_, i) => step(i + 2, false, i === 0 ? "Senza cache: per il secondo token si ricalcolano K e V anche del primo (in rosso), anche se non sono cambiati." : `Token ${i + 2}: si rifanno ${i + 2} colonne. In tutto ${((i + 2) * (i + 3)) / 2}: il lavoro cresce come n(n+1)/2.`)),
      {
        title: "L'idea della cache",
        caption: "Con la maschera causale K e V di un token passato non cambiano più. Allora si salvano (cache_k, cache_v nel codice di Llama) e a ogni passo si calcola solo la colonna nuova.",
        duration: 5,
        enter() {
          r.set(1, true);
          tab.plots.kvWork.setData(work(N_NO));
          tab.showCode("llama.cache");
        },
      },
      ...Array.from({ length: r.T - 1 }, (_, i) => step(i + 2, true, i < 2 ? "Con cache: si calcola solo la colonna del token nuovo (in chiaro); le altre si rileggono." : `Token ${i + 2}: sempre una colonna nuova. Il prezzo è la memoria, che cresce con il contesto.`)),
      {
        title: "Il conto",
        caption: "Grafico «Colonne K/V calcolate»: la cache trasforma un lavoro quadratico in lineare. In cambio occupa memoria: per questo esistono GQA, MQA e MLA.",
        duration: 5,
        enter() {
          r.set(r.T, true);
          tab.plots.kvWork.setData(work(r.T));
          tab.openCharts();
        },
      },
    ],
  };
}

function flashAction(tab) {
  const r = tab.regions.flash;
  const per = FLASH_N / FLASH_TILE;
  const blocks = [];
  for (let bi = 0; bi < per; bi++) for (let bj = 0; bj <= bi; bj++) blocks.push(bi * per + bj);
  return {
    id: "flash",
    title: "FlashAttention e finestra scorrevole",
    category: CATEGORY,
    summary: "Attention esatta calcolata a blocchi nella memoria veloce; poi la maschera a finestra di Mistral 7B.",
    onStart() {
      tab.prepare("flash");
    },
    onStop() {
      r.tile(-1);
      r.window(4);
      r.note();
    },
    steps: [
      {
        title: "Il problema",
        caption: "L'attention standard costruisce la matrice T × T dei punteggi e la scrive nella memoria grande ma lenta della GPU (HBM). Con contesti lunghi è il trasferimento di dati, non il calcolo, a rallentare.",
        duration: 5,
        enter() {
          r.tile(-1);
          r.window(FLASH_N);
          r.note([{ text: "Matrice intera: T² numeri scritti e riletti dalla HBM.", bold: true }, { text: "Qui T = 16; con 128K token (131.072) sarebbero circa 17 miliardi di punteggi per head e per layer.", color: "#94a3b8" }]);
        },
      },
      {
        title: "A blocchi, in SRAM",
        caption: "FlashAttention carica un blocco di query e uno di key nella SRAM (piccola e veloce), calcola, aggiorna un softmax «in corsa» e passa al blocco successivo. I blocchi sopra la diagonale si saltano.",
        duration: 8,
        enter() {
          r.window(FLASH_N);
          r.tile(blocks[0], 0);
          r.note([{ text: "Un blocco alla volta: la matrice intera non esiste mai in memoria.", bold: true }, { text: "Il risultato è identico all'attention normale (esatta, non approssimata).", color: "#94a3b8" }]);
        },
        update(p) {
          const f = Math.min(blocks.length - 1e-6, p * blocks.length);
          const k = Math.floor(f);
          r.tile(blocks[k], Math.min(1, (f - k) * 2.5));
        },
      },
      {
        title: "Fatto",
        caption: "Tutti i blocchi utili calcolati. Nel paper: addestramento di GPT-2 3 volte più veloce e memoria aggiuntiva che cresce in modo lineare con la lunghezza, non quadratico.",
        duration: 3.5,
        enter() {
          r.window(FLASH_N);
          r.tile(blocks.length + 99);
        },
      },
      {
        title: "Finestra scorrevole",
        caption: "Mistral 7B limita ogni layer agli ultimi W token (W = 4096). Qui W = 4: ogni riga (query) vede al massimo 4 colonne (key). In rosso scuro ciò che la finestra esclude.",
        duration: 7,
        enter() {
          r.tile(blocks.length + 99);
          r.window(4, 0);
        },
        update(p) {
          r.window(4, Math.min(FLASH_N - 1, Math.floor(p * FLASH_N)));
        },
      },
      {
        title: "Contesti lunghi",
        caption: "Impilando i layer l'informazione viaggia oltre la finestra. Oggi diversi modelli aperti dichiarano contesti fino a 128K token (Llama 3.1, DeepSeek-V3), con RoPE adattato e cache efficienti.",
        duration: 5,
        enter() {
          r.window(4);
          r.note([{ text: "Contesti lunghi = attention efficiente + KV cache piccola + RoPE adattato.", bold: true }, { text: "Llama 3.1 e DeepSeek-V3: 128K token (model card e README ufficiali).", color: "#94a3b8" }]);
        },
      },
    ],
  };
}

function specAction(tab) {
  const r = tab.regions.spec;
  const rounds = SPEC_ROUNDS.flatMap((R, k) => [
    {
      title: `Giro ${k + 1}: la bozza propone`,
      caption: `Il modello piccolo propone ${R.draft.length} token, uno alla volta: «${R.draft.join(" ")}». È veloce perché è piccolo.`,
      duration: 3.2,
      enter() {
        r.set(k, "draft", 0);
      },
      update(p) {
        r.set(k, "draft", Math.min(R.draft.length, Math.ceil(phase(p, 0.05, 0.85) * R.draft.length)));
      },
    },
    {
      title: `Giro ${k + 1}: il grande verifica`,
      caption:
        R.accepted === R.draft.length
          ? `Il modello grande valuta tutte le proposte in UN passaggio: le accetta tutte e, già che c'è, aggiunge un token suo («${R.bonus}»).`
          : `Un solo passaggio del modello grande: accetta ${R.accepted} proposte, rifiuta «${R.draft[R.accepted]}» e tutto ciò che segue, e mette al suo posto «${R.bonus}».`,
      duration: 4,
      enter() {
        r.set(k, "verify", R.draft.length);
      },
    },
  ]);
  return {
    id: "spec",
    title: "Decodifica speculativa",
    category: CATEGORY,
    summary: "Il piccolo propone, il grande verifica in un passaggio: più token per ogni passaggio costoso.",
    onStart() {
      tab.prepare("spec", { charts: "inferenza" });
    },
    onStop() {
      r.set(SPEC_ROUNDS.length);
      tab.plots.spec.setHighlight(null);
    },
    steps: [
      {
        title: "Il collo di bottiglia",
        caption: "Copione didattico. Normalmente ogni token richiede un passaggio completo del modello grande: per 11 token, 11 passaggi.",
        duration: 4,
        enter() {
          r.set(0, "commit");
          tab.showCode("sim:speculativeExpectedTokens");
        },
      },
      ...rounds,
      {
        title: "Il conto",
        caption: "3 passaggi del modello grande per 11 token. Con la regola di accettazione del paper la distribuzione è la stessa del modello grande da solo. Grafico: token attesi per passaggio in funzione di α.",
        duration: 6,
        enter() {
          r.set(SPEC_ROUNDS.length);
          tab.openCharts();
          tab.plots.spec.setHighlight({ x: 0.7 });
        },
      },
    ],
  };
}

export function inferenceActions(tab) {
  return [bpeAction(tab), kvAction(tab), flashAction(tab), specAction(tab)];
}
