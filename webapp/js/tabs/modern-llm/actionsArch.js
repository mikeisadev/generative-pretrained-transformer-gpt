/**
 * Azioni "Architettura" (prima parte): confronto dei blocchi, Mixture of Experts, GQA/MQA, RoPE.
 * Regola di tutte le azioni: `enter` ricostruisce lo stato di inizio passo (il passo indietro
 * funziona sempre), `update(p)` anima da lì; niente allocazioni in update.
 */
import { COMPARE_ROWS } from "./regionCompare.js";
import { MOE_TOKENS } from "./regionMoe.js";
import { RouterSim } from "./sim.js";
import { phase } from "./shapes.js";

const CATEGORY = "Architettura";

function compareAction(tab) {
  const r = tab.regions.compare;
  const captions = {
    pos: "Posizioni: il nostro GPT somma una tabella appresa di 32 posizioni. RoPE invece ruota query e key in base alla posizione: niente tabella, e conta la distanza tra i token.",
    norm: "Normalizzazione: LayerNorm toglie la media e divide per la deviazione standard. RMSNorm divide solo per la radice della media dei quadrati: più semplice, qualità simile.",
    attn: "Attention: nel nostro GPT ogni head ha i suoi K e V. Con GQA più head di query condividono K e V (Llama 3 8B: 32 query, 8 K/V), e una KV cache evita di ricalcolarli.",
    res: "Le connessioni residue e la struttura pre-norm restano identiche: x + attention(norm(x)), poi x + MLP(norm(x)).",
    ffn: "MLP: al posto di Linear-ReLU-Linear, SwiGLU (tre matrici e un cancello SiLU). Nei modelli MoE la MLP diventa un router con tanti esperti.",
    out: "Vocabolario e contesto: da 65 caratteri e 32 posizioni a 128.256 token BPE e migliaia di posizioni (8K in Llama 3, 128K in Llama 3.1).",
  };
  const code = { pos: "llama.rope", norm: "llama.rmsnorm", attn: "llama.init_kv", res: "llama.block", ffn: "llama.ffn", out: "minbpe.train" };
  return {
    id: "compare",
    title: "Dal nostro GPT a un blocco moderno",
    category: CATEGORY,
    summary: "I pezzi del blocco cambiano uno alla volta: RoPE, RMSNorm, GQA, SwiGLU, vocabolario.",
    onStart() {
      tab.prepare("compare");
      r.reveal(-1);
    },
    onStop() {
      r.reset();
    },
    steps: [
      {
        title: "Il nostro blocco",
        caption: "A sinistra il blocco del GPT che abbiamo addestrato in questa lezione. A destra, vuoto per ora, un blocco di Llama 3: lo riempiamo un pezzo alla volta.",
        duration: 4,
        enter() {
          r.reveal(-1);
          r.highlight(-1);
          tab.showCode("ours:block");
        },
      },
      ...COMPARE_ROWS.map((row, i) => ({
        title: row.modern[0],
        caption: captions[row.id],
        duration: 5,
        enter() {
          r.reveal(i - 1);
          r.highlight(i);
          r.fly(i, 0);
          tab.showCode(code[row.id]);
        },
        update(p) {
          r.fly(i, phase(p, 0.05, 0.45));
          if (p > 0.4) r.reveal(i);
        },
        exit() {
          r.fly(i, 1);
        },
      })),
      {
        title: "Il blocco completo",
        caption: "Ecco il blocco moderno completo. Ossatura identica, pezzi più efficienti: è questo che permette di arrivare a miliardi di parametri e contesti lunghi.",
        duration: 4,
        enter() {
          r.reveal(COMPARE_ROWS.length);
          r.highlight(-1);
          tab.showCode("llama.block");
        },
      },
    ],
  };
}

function moeAction(tab) {
  const r = tab.regions.moe;
  const labels = Array.from({ length: 8 }, (_, e) => `E${e + 1}`);
  let routes = [];
  let balance = [];
  let lastKey = "";
  const showLoad = (values, highlight) => {
    const key = values.join(",") + "|" + highlight;
    if (key === lastKey) return;
    lastKey = key;
    tab.plots.moeLoad.setData({ labels, values, highlight: highlight ?? null });
  };
  const countUpTo = (n) => {
    const counts = new Array(8).fill(0);
    for (let i = 0; i < n; i++) for (const e of routes[i].chosen) counts[e]++;
    return counts;
  };
  return {
    id: "moe",
    title: "Mixture of Experts: il router sceglie",
    category: CATEGORY,
    summary: "Ogni token va a 2 esperti su 8 (come Mixtral); poi il problema del bilanciamento e l'esperto condiviso di DeepSeek-V3.",
    onStart() {
      tab.prepare("moe", { charts: "moe" });
      const router = new RouterSim({ seed: 11 });
      routes = MOE_TOKENS.map(() => router.route(router.token(), 2));
      // bilanciamento: 24 istantanee del carico mentre i bias si aggiustano (simulazione)
      const skewed = new RouterSim({ seed: 5, skew: 1.2 });
      balance = [];
      for (let k = 0; k < 24; k++) {
        const counts = skewed.load(512);
        balance.push(counts);
        skewed.balanceStep(counts, 0.12);
      }
      lastKey = "";
    },
    onStop() {
      r.reset();
    },
    // build (non steps): le didascalie usano le scelte del router calcolate in onStart
    build: async () => [
      {
        title: "Un layer MoE",
        caption: "Simulazione didattica (pesi casuali). Al posto di una sola MLP ci sono 8 esperti e un router: una piccola matrice che dà un punteggio a ogni esperto per ogni token.",
        duration: 4,
        enter() {
          r.reset();
          showLoad(new Array(8).fill(0), undefined);
          tab.showCode("mistral.moe");
        },
      },
      ...MOE_TOKENS.map((tok, i) => ({
        title: `Token «${tok}»`,
        caption:
          i === 0
            ? `Il router calcola le probabilità degli 8 esperti per «${tok}», sceglie i 2 migliori (top-2) e somma le loro uscite pesate con un softmax sui due punteggi scelti.`
            : `«${tok}»: esperti ${routes[i]?.chosen.map((e) => e + 1).join(" e ") ?? ""}. Gli altri 6 non lavorano: ecco perché un MoE ha molti parametri ma usa poco calcolo per token.`,
        duration: i === 0 ? 5 : 2.6,
        enter(ctx) {
          ctx.data.loadShown = -1;
          r.setShared(false);
          r.setToken(i);
          r.setRoute(routes[i], 0);
          showLoad(countUpTo(i), undefined);
        },
        update(p, dt, ctx) {
          r.setRoute(routes[i], phase(p, 0.1, 0.75));
          if (p > 0.75 && ctx.data.loadShown !== i) {
            ctx.data.loadShown = i; // una sola volta per passo, non a ogni frame
            showLoad(countUpTo(i + 1), routes[i].chosen[0]);
          }
        },
      })),
      {
        title: "Il rischio: pochi esperti fanno tutto",
        caption: "Grafico «Carico degli esperti»: su 512 token un router sbilanciato manda quasi tutto ai primi esperti. Gli altri non imparano: è il problema del bilanciamento del carico.",
        duration: 5,
        enter() {
          r.reset();
          showLoad(balance[0], 0);
        },
      },
      {
        title: "Bilanciamento",
        caption: "Si corregge un piccolo bias per esperto: chi è sovraccarico viene scelto un po' meno, chi è scarico un po' di più (l'idea senza loss ausiliaria di DeepSeek-V3). Il carico si livella.",
        duration: 6,
        enter(ctx) {
          r.reset();
          ctx.data.balanceIndex = 0;
          showLoad(balance[0], undefined);
        },
        update(p, dt, ctx) {
          const k = Math.min(balance.length - 1, Math.floor(p * balance.length));
          if (k === ctx.data.balanceIndex) return;
          ctx.data.balanceIndex = k;
          showLoad(balance[k], undefined);
        },
      },
      {
        title: "Esperto condiviso",
        caption: "DeepSeek-V3: in ogni layer MoE 1 esperto condiviso sempre attivo più 8 scelti tra 256. In totale 671B parametri, ma solo 37B attivi per token.",
        duration: 5,
        enter() {
          r.setShared(true);
          r.setToken(2);
          r.setRoute(routes[2], 1);
        },
      },
    ],
  };
}

function gqaAction(tab) {
  const r = tab.regions.gqa;
  const morphStep = (from, to, title, caption) => ({
    title,
    caption,
    duration: 5,
    enter() {
      r.morph(from, from, 1);
      tab.plots.kvMemory.setHighlight({ x: 8 });
    },
    update(p) {
      r.morph(from, to, phase(p, 0.1, 0.6));
    },
  });
  return {
    id: "gqa",
    title: "GQA e MQA: query che condividono K e V",
    category: CATEGORY,
    summary: "Da una coppia K/V per head (MHA) a una per gruppo (GQA) a una sola (MQA): la KV cache si rimpicciolisce.",
    onStart() {
      tab.prepare("gqa", { charts: "kv" });
    },
    onStop() {
      r.set("gqa");
      tab.plots.kvMemory.setHighlight(null);
    },
    steps: [
      {
        title: "Multi-head attention",
        caption: "Come nel nostro GPT: ogni head di query (in alto) ha la sua coppia K/V (in basso). Tutte le K e V finiscono nella KV cache, per ogni token e ogni layer.",
        duration: 4.5,
        enter() {
          r.set("mha");
          tab.showCode("ours:mha");
          tab.plots.kvMemory.setHighlight({ x: 8 });
        },
      },
      morphStep("mha", "gqa", "Grouped-query attention", "GQA: le query si dividono in gruppi e ogni gruppo legge la stessa coppia K/V. Qui 8 query e 2 K/V, lo stesso rapporto 4:1 di Llama 3 8B (32 query, 8 K/V)."),
      {
        title: "Come lo fa il codice",
        caption: "Nel codice di Llama le K e V vengono calcolate solo per le 8 head K/V e poi «ripetute» (repeat_kv) per le 32 query: meno pesi e una cache 4 volte più piccola.",
        duration: 5,
        enter() {
          r.set("gqa");
          tab.showCode("llama.repeat_kv");
        },
        update(p) {
          r.pulse(p * 3 % 1);
        },
        exit() {
          r.pulse(0);
        },
      },
      morphStep("gqa", "mqa", "Multi-query attention", "MQA (Shazeer, 2019): tutte le query condividono UNA sola coppia K/V. Cache minima, ma un po' di qualità in meno: GQA è il compromesso che si usa di più."),
      {
        title: "Quanto pesa la cache",
        caption: "Grafico «Memoria della KV cache»: con l'architettura di Llama 3 8B a 8K token servono 4 GiB con MHA, 1 GiB con GQA, 128 MiB con MQA. DeepSeek-V3 con MLA salva un latente compresso.",
        duration: 6,
        enter() {
          r.set("gqa");
          tab.ui.charts.open();
          tab.plots.kvMemory.setHighlight({ x: 8 });
        },
      },
    ],
  };
}

function ropeAction(tab) {
  const r = tab.regions.rope;
  const angleNoteOf = (shift) => [
    { text: `Posizioni ${1 + shift} e ${3 + shift}: le frecce sono ruotate di ${1 + shift}θ e ${3 + shift}θ.`, bold: true },
    { text: "La differenza resta 2θ: il prodotto q·k dipende solo dalla distanza (2 posizioni).", color: "#94a3b8" },
  ];
  const ANGLE_NOTES = [0, 1, 2].map(angleNoteOf);
  const angleNote = (shift) => ANGLE_NOTES[shift];
  return {
    id: "rope",
    title: "RoPE: ruotare per dire la posizione",
    category: CATEGORY,
    summary: "Ogni coppia di dimensioni di q e k ruota di posizione × θ; conta solo la distanza tra i token.",
    onStart() {
      tab.prepare("rope");
    },
    onStop() {
      r.set(0, []);
      r.note([{ text: "Ogni coppia di dimensioni di q e k ruota di (posizione × θ).", bold: true }, { text: "Riga lenta disegnata con angoli × 10 per renderla visibile.", color: "#94a3b8" }]);
    },
    steps: [
      {
        title: "Lo stesso vettore in posizioni diverse",
        caption: "Prendiamo la stessa coppia di numeri di una query in 6 posizioni. Senza RoPE le frecce sarebbero tutte uguali: il vettore non sa dove si trova.",
        duration: 4,
        enter() {
          r.set(0, [], 0);
          r.note([{ text: "Senza RoPE: stesso vettore, stessa direzione, in ogni posizione.", bold: true }]);
          tab.showCode("ours:model.init");
        },
      },
      {
        title: "Ruotare di m · θ",
        caption: "RoPE ruota la coppia di un angolo m · θ, dove m è la posizione. Riga in alto: coppia veloce (θ = 1 radiante). Riga in basso: coppia lenta (θ = 0,01, disegnata 10 volte più ampia).",
        duration: 5,
        enter() {
          r.set(0, [], 0);
          tab.showCode("llama.rope");
          r.note([{ text: "θ_i = 10000^(−2i/d): coppie diverse girano a velocità diverse.", bold: true }, { text: "Llama 3 usa una base di 500.000 invece di 10.000.", color: "#94a3b8" }]);
        },
        update(p) {
          r.set(0, [], phase(p, 0.1, 0.8));
        },
      },
      {
        title: "Conta la distanza",
        caption: "Guardiamo le posizioni 1 e 3. Spostiamo tutta la frase avanti di 2: entrambe le frecce girano dello stesso angolo e la differenza tra loro (2θ) non cambia.",
        duration: 6,
        enter() {
          r.set(0, [1, 3]);
          r.note(angleNote(0));
        },
        update(p) {
          const shift = phase(p, 0.2, 0.8) * 2;
          r.set(shift, [1, 3]);
          r.note(angleNote(Math.round(shift)));
        },
      },
      {
        title: "Perché funziona",
        caption: "Il punteggio di attention è un prodotto scalare: ruotare q di mθ e k di nθ equivale a ruotare k di (n − m)θ. Il modello vede la posizione relativa, senza una tabella di posizioni.",
        duration: 5,
        enter() {
          r.set(0, [1, 3]);
          r.note([{ text: "⟨R_m q, R_n k⟩ = ⟨q, R_(n−m) k⟩", bold: true }, { text: "Vedi «Approfondimento» nel percorso.", color: "#94a3b8" }]);
        },
      },
    ],
  };
}

export function architectureActions(tab) {
  return [compareAction(tab), moeAction(tab), gqaAction(tab), ropeAction(tab)];
}
