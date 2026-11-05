/**
 * Grafici del tab (ChartsPanel del nucleo). Ognuno ha titolo, "a cosa serve" e legenda.
 *   scala      legge di Chinchilla (formula e coefficienti pubblicati) e allocazione ottimale
 *   moe        carico degli esperti (SIMULAZIONE: router con pesi casuali)
 *   kv         memoria della KV cache (calcolata dalle config ufficiali) e K/V calcolati
 *   inferenza  decodifica speculativa (formula del paper) e più tentativi (formula esatta)
 *   grpo       vantaggi del gruppo (formula GRPO) durante l'azione
 *   emergenza  metrica discreta contro continua (ILLUSTRAZIONE della critica di Schaeffer et al.)
 */
import {
  DEEPSEEK_V3,
  LLAMA3_8B,
  chinchillaLoss,
  chinchillaOptimal,
  emergenceIllustration,
  kvBytesPerToken,
  mlaBytesPerToken,
  passAtN,
  speculativeExpectedTokens,
} from "./sim.js";

const GIB = 1024 ** 3;
const range = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

export function installPlots(tab) {
  const charts = tab.ui.charts;
  const plots = {};

  // --- leggi di scala
  const scala = charts.section("scala", "Leggi di scala", "Formula di Hoffmann et al. (2022) con i coefficienti pubblicati: L(N, D) = 1,69 + 406,4/N^0,34 + 410,7/D^0,28.");
  plots.chinchilla = scala.chart("loss", {
    type: "line",
    title: "Loss prevista contro calcolo",
    description: "Serve a capire perché non basta ingrandire il modello: a parametri fissi la loss si ferma; con l'allocazione ottimale (N e D crescono insieme) continua a scendere. x = log10 dei FLOP di training (C ≈ 6·N·D).",
    xLabel: "log10 FLOP",
    yLabel: "loss (nats/token)",
  });
  const xs = range(19, 25, 49);
  // a N fisso tutto il calcolo in più diventa dati: D = C / (6N)
  const fixed = [1e9, 1e10, 7e10].map((N) => ({ name: `N fisso = ${N / 1e9}B`, points: xs.map((x) => [x, chinchillaLoss(N, Math.pow(10, x) / (6 * N))]).filter(([, y]) => y < 4) }));
  plots.chinchilla.setData({ series: [{ name: "allocazione ottimale", points: xs.map((x) => [x, chinchillaOptimal(Math.pow(10, x)).loss]) }, ...fixed] });
  plots.alloc = scala.chart("alloc", {
    type: "line",
    title: "Parametri e token ottimali",
    description: "Quanti parametri (N) e quanti token (D) usare per ogni budget di calcolo secondo lo stesso fit. La scala logaritmica mostra due rette: crescono con esponenti ≈ 0,45 e ≈ 0,55.",
    xLabel: "log10 FLOP",
    yLabel: "quantità",
    yScale: "log",
  });
  plots.alloc.setData({
    series: [
      { name: "parametri N", points: xs.map((x) => [x, chinchillaOptimal(Math.pow(10, x)).N]) },
      { name: "token D", points: xs.map((x) => [x, chinchillaOptimal(Math.pow(10, x)).D]) },
    ],
  });

  // --- MoE
  const moe = charts.section("moe", "Mixture of Experts", "Simulazione didattica: router con pesi casuali e seme fisso, 8 esperti, top-2.");
  plots.moeLoad = moe.chart("load", {
    type: "bar",
    title: "Carico degli esperti (simulazione)",
    description: "Quanti token riceve ogni esperto. Serve a vedere il problema del bilanciamento: se il router preferisce pochi esperti, gli altri restano inutilizzati. Si aggiorna durante l'azione «Mixture of Experts».",
    xLabel: "esperto",
    yLabel: "token assegnati",
  });
  plots.moeLoad.setData({ labels: Array.from({ length: 8 }, (_, e) => `E${e + 1}`), values: new Array(8).fill(0) });

  // --- KV cache
  const kv = charts.section("kv", "KV cache", "Memoria calcolata dalle configurazioni ufficiali, valori in bf16 (2 byte).");
  plots.kvMemory = kv.chart("memory", {
    type: "line",
    title: "Memoria della KV cache contro lunghezza del contesto",
    description: "A cosa serve: capire perché GQA, MQA e MLA esistono. Architettura di Llama 3 8B (32 layer, head da 128) con 32 (MHA), 8 (GQA, la scelta vera) o 1 (MQA) head K/V; per confronto DeepSeek-V3 con MLA (61 layer, latente 512 + 64), un modello molto più grande.",
    xLabel: "token nel contesto (migliaia)",
    yLabel: "GiB per sequenza",
  });
  const ctx = range(0, 128, 33);
  const mem = (bytes) => ctx.map((k) => [k, (bytes * k * 1024) / GIB]);
  plots.kvMemory.setData({
    series: [
      { name: "MHA (32 K/V)", points: mem(kvBytesPerToken({ ...LLAMA3_8B, kvHeads: 32 })) },
      { name: "GQA (8 K/V)", points: mem(kvBytesPerToken(LLAMA3_8B)) },
      { name: "MQA (1 K/V)", points: mem(kvBytesPerToken({ ...LLAMA3_8B, kvHeads: 1 })) },
      { name: "MLA DeepSeek-V3", points: mem(mlaBytesPerToken(DEEPSEEK_V3)) },
    ],
  });
  plots.kvWork = kv.chart("work", {
    type: "line",
    title: "Colonne K/V calcolate: con e senza cache",
    description: "Si riempie durante l'azione «KV cache»: senza cache il lavoro cresce come n(n+1)/2, con la cache come n. Per ogni layer.",
    xLabel: "token generati",
    yLabel: "colonne K/V calcolate",
  });

  // --- inferenza
  const inf = charts.section("inferenza", "Più calcolo all'inferenza", "Formule esatte sotto ipotesi dichiarate.");
  plots.spec = inf.chart("spec", {
    type: "line",
    title: "Decodifica speculativa: token per passaggio del modello grande",
    description: "Formula di Leviathan et al.: (1 − α^(γ+1)) / (1 − α). Serve a capire quando conviene: con una bozza che indovina spesso (α alto) ogni passaggio del modello grande produce più token.",
    xLabel: "α (probabilità di accettazione)",
    yLabel: "token attesi",
  });
  const alphas = range(0, 0.95, 39);
  plots.spec.setData({ series: [2, 4, 8].map((g) => ({ name: `γ = ${g}`, points: alphas.map((a) => [a, speculativeExpectedTokens(a, g)]) })) });
  plots.passN = inf.chart("pass", {
    type: "line",
    title: "Più tentativi con un verificatore",
    description: "1 − (1 − p)^N: probabilità che almeno uno di N tentativi indipendenti sia giusto, se un verificatore perfetto lo riconosce. Ipotesi ottimistiche: mostra il potenziale del test-time compute.",
    xLabel: "tentativi N",
    yLabel: "probabilità",
    yMin: 0,
    yMax: 1,
  });
  const ns = Array.from({ length: 16 }, (_, i) => i + 1);
  plots.passN.setData({ series: [0.1, 0.3, 0.6].map((p) => ({ name: `p = ${String(p).replace(".", ",")}`, points: ns.map((n) => [n, passAtN(p, n)]) })) });

  // --- GRPO
  const grpo = charts.section("grpo", "RL con ricompense verificabili", "Formula di GRPO applicata al gruppo dell'azione.");
  plots.grpo = grpo.chart("adv", {
    type: "bar",
    title: "Vantaggi del gruppo (GRPO)",
    description: "Ricompensa meno media del gruppo, diviso deviazione standard: positivo = risposta da rinforzare, negativo = da scoraggiare. Si riempie durante l'azione «RL con ricompense verificabili».",
    xLabel: "risposta",
    yLabel: "vantaggio",
  });
  plots.grpo.setData({ labels: ["1", "2", "3", "4"], values: [0, 0, 0, 0] });

  // --- emergenza
  const em = charts.section("emergenza", "Capacità «emergenti»", "Illustrazione dichiarata dell'argomento di Schaeffer et al. (2023): la curva per token è inventata, la relazione p^L è esatta.");
  plots.emergence = em.chart("metric", {
    type: "line",
    title: "Stessa capacità, due metriche",
    description: "A cosa serve: vedere che un salto può dipendere dalla metrica. La probabilità per token cresce dolcemente con la scala; la risposta esatta su 10 token (p^10) sembra accendersi all'improvviso.",
    xLabel: "log10 parametri (illustrativo)",
    yLabel: "punteggio",
    yMin: 0,
    yMax: 1,
  });
  const xe = range(7, 13, 49);
  plots.emergence.setData({
    series: [
      { name: "per token (continua)", points: xe.map((x) => [x, emergenceIllustration(x).perToken]) },
      { name: "esatta su 10 token (discreta)", points: xe.map((x) => [x, emergenceIllustration(x).exactMatch]) },
    ],
  });

  tab.plots = plots;
  return plots;
}
