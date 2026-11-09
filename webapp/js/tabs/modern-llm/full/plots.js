/**
 * Grafici della vista completa (ChartsPanel del nucleo), in sezioni con id "full-…" (index.js le
 * mostra solo in questo sottotab). Numeri dai dati dei modelli e dalla ricerca; l'unica curva
 * illustrativa (emergenza) è quella dichiarata del tab, con la relazione esatta p^L.
 *   full-pesi   parametri per componente, byte per precisione (modello scelto)
 *   full-kv     KV cache per token dei tre modelli e dei due confronti della ricerca (scala log)
 *   full-scala  token per parametro (scala log), quota di parametri attivi, metrica e "emergenza"
 */
import { emergenceIllustration } from "../sim.js";
import { MODELS, PRIMARY_MODEL_ID, MODEL_BY_ID } from "./models/index.js";
import { formatBytes, formatCount, paramTotals } from "./model.js";
import { SCALING_POINTS } from "./research.js";

export const FULL_CHART_SECTIONS = ["full-pesi", "full-kv", "full-scala"];

const SHORT = { "deepseek-v4.1-flash": "V4.1-Flash", "kimi-k3.1-pro": "Kimi K3.1", "gpt-oss-120b": "gpt-oss" };

export function installFullPlots(charts) {
  const plots = {};
  const pesi = charts.section("full-pesi", "Architettura completa · dove sono i pesi", "Conteggi esatti dai dati del modello scelto; byte del checkpoint per precisione.");
  plots.params = pesi.chart("params", {
    type: "bar",
    title: "Parametri per componente (miliardi)",
    description: "A cosa serve: vedere che quasi tutti i parametri sono negli esperti, che però lavorano pochi alla volta. Le altre parti sono barre quasi invisibili: è proprio questo il punto.",
    xLabel: "componente",
    yLabel: "miliardi di parametri",
    format: (v) => `${v.toLocaleString("it-IT", { maximumFractionDigits: 2 })}B`,
  });
  plots.precision = pesi.chart("precision", {
    type: "bar",
    title: "Byte del checkpoint per precisione (GB)",
    description: "Dove finiscono i byte del file dei pesi: MXFP4 = 4,25 bit per numero, FP8 = 8 bit (+ scale), BF16 = 16, FP32 = 32.",
    xLabel: "precisione",
    yLabel: "GB",
    format: (v) => `${v.toLocaleString("it-IT", { maximumFractionDigits: 1 })} GB`,
  });

  const kv = charts.section("full-kv", "Architettura completa · KV cache", "Byte salvati per ogni token del contesto (conti della ricerca).");
  plots.kv = kv.chart("bytes", {
    type: "bar",
    title: "KV cache per token, scala logaritmica",
    description: "Ogni tacca dell'asse è ×10. V4.1-Flash: 890 B, circa 437 volte meno di un modello GQA da 67B. Kimi: solo i 24 layer MLA (più uno stato fisso per sequenza); gpt-oss: i 18 layer densi.",
    xLabel: "modello",
    yLabel: "log10 byte per token",
    format: (v) => formatBytes(Math.round(Math.pow(10, v))),
  });
  const primary = MODEL_BY_ID.get(PRIMARY_MODEL_ID);
  const refs = primary.kv.compare.filter((c) => c.id !== "self");
  const kvRows = [...MODELS.map((m) => ({ label: SHORT[m.id], bytes: m.kv.perToken })), ...refs.map((r) => ({ label: r.id === "v3" ? "DeepSeek-V3" : "LLM 67B (GQA)", bytes: r.bytes }))];
  plots.kv.setData({ labels: kvRows.map((r) => r.label), values: kvRows.map((r) => Math.log10(r.bytes)), highlight: 0 });

  const scala = charts.section("full-scala", "Architettura completa · scala e capacità", "Rapporti tra dati, parametri e calcolo; la critica della metrica alle abilità «emergenti».");
  plots.tokens = scala.chart("tokens", {
    type: "bar",
    title: "Token di training per parametro (scala logaritmica)",
    description: "Chinchilla (2022): circa 20 token per parametro all'ottimo di calcolo. V4.1-Flash: 45T token per 552B totali (≈ 80) o 16B attivi (≈ 2.800). Kimi e gpt-oss non dichiarano i token.",
    xLabel: "",
    yLabel: "log10 token per parametro",
    format: (v) => Math.round(Math.pow(10, v)).toLocaleString("it-IT"),
  });
  plots.tokens.setData({ labels: SCALING_POINTS.map((p) => p.label), values: SCALING_POINTS.map((p) => Math.log10(p.tokensPerParam)) });
  plots.active = scala.chart("active", {
    type: "bar",
    title: "Quanta parte del modello lavora per ogni token",
    description: "Parametri attivi per token / parametri totali del backbone: è il «trucco» dei MoE. Embedding escluso (è una lettura).",
    xLabel: "modello",
    yLabel: "% attiva",
    format: (v) => `${v.toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`,
  });
  const activeRows = [];
  for (const m of MODELS) {
    const t = paramTotals(m);
    activeRows.push({ label: `${SHORT[m.id]}${t.prefill ? " decode" : ""}`, value: (100 * t.active) / t.backbone, note: `${formatCount(t.active)} su ${formatCount(t.backbone)}` });
    if (t.prefill) activeRows.push({ label: `${SHORT[m.id]} prefill`, value: (100 * t.prefill) / t.backbone });
  }
  plots.active.setData({ labels: activeRows.map((r) => r.label), values: activeRows.map((r) => r.value) });
  plots.emergence = scala.chart("emergence", {
    type: "line",
    title: "Stessa capacità, due metriche (illustrazione)",
    description: "La critica di Schaeffer et al. (2023): la probabilità per token cresce dolcemente; la risposta esatta su 10 token (p^10) sembra «accendersi». La curva per token è inventata, la relazione p^L è esatta.",
    xLabel: "log10 parametri (illustrativo)",
    yLabel: "punteggio",
    yMin: 0,
    yMax: 1,
  });
  const xs = Array.from({ length: 49 }, (_, i) => 7 + (6 * i) / 48);
  plots.emergence.setData({
    series: [
      { name: "per token (continua)", points: xs.map((x) => [x, emergenceIllustration(x).perToken]) },
      { name: "esatta su 10 token (discreta)", points: xs.map((x) => [x, emergenceIllustration(x).exactMatch]) },
    ],
  });

  /** Aggiorna i grafici che dipendono dal modello scelto. */
  plots.setModel = (model) => {
    const t = paramTotals(model);
    const comps = t.components.filter((c) => c.params >= 1e6);
    plots.params.setText(`Parametri per componente: ${model.name} (miliardi)`);
    plots.params.setData({ labels: comps.map((c) => c.label), values: comps.map((c) => c.params / 1e9), highlight: 0 });
    plots.precision.setText(`Byte del checkpoint per precisione: ${model.name} (GB${model.precision.some((p) => p.estimate) ? ", stima" : ""})`);
    plots.precision.setData({ labels: model.precision.map((p) => `${p.dtype} ${p.what.split(/[,(]/)[0].trim()}`), values: model.precision.map((p) => p.gb) });
    const index = MODELS.findIndex((m) => m.id === model.id);
    plots.kv.setData({ labels: kvRows.map((r) => r.label), values: kvRows.map((r) => Math.log10(r.bytes)), highlight: index });
  };
  return plots;
}
