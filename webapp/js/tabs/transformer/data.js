/**
 * Caricamento e decodifica di data/model.json (generato da gpt/export.py).
 */

/** Decodifica una stringa base64 in un array di byte. */
export function decodeBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function loadModelSpec(url = "data/model.json") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Impossibile caricare ${url} (HTTP ${response.status})`);
  const spec = await response.json();
  const cfg = spec.config;
  const result = await loadTrainingResult(new URL("training/trace.json", new URL(url, location.href)));
  // Valori derivati usati dai testi ({{...}}): calcolati una volta sola qui.
  spec.vars = {
    ...cfg,
    head_size: spec.head_size,
    ffw: 4 * cfg.n_embd,
    block_size_minus_1: cfg.block_size - 1,
    total: spec.total_params.toLocaleString("it-IT", { useGrouping: "always" }),
    ln_vocab: Math.log(cfg.vocab_size).toFixed(2).replace(".", ","),
    prompt: escapeHtml(JSON.stringify(spec.example.prompt).slice(1, -1)),
    // Esito del training registrato (cambia a ogni nuovo training: mai scritto a mano nei testi).
    train_steps: result ? (result.steps + 1).toLocaleString("it-IT") : "5000",
    train_loss: result ? italian(result.train) : "≈1,66",
    val_loss: result ? italian(result.val) : "≈1,81",
  };
  return spec;
}

const italian = (x) => x.toFixed(2).replace(".", ",");

/** Ultima valutazione del training registrato (data/training/trace.json), o null se manca. */
async function loadTrainingResult(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    const { evals } = await response.json();
    const last = evals.steps.length - 1;
    return { steps: evals.steps[last], train: evals.train[last], val: evals.val[last] };
  } catch {
    return null; // i testi useranno i valori indicativi
  }
}

/** Sostituisce {{chiave}} con i valori del modello. */
export function fillTemplate(text, vars) {
  return text.replace(/\{\{(\w+)\}\}/g, (match, key) => (key in vars ? String(vars[key]) : match));
}

/** Somma dei parametri appresi (o dei buffer) il cui nome inizia con prefix. */
export function summarize(spec, prefix, buffer = false) {
  const list = buffer ? spec.buffers : spec.params;
  const hits = list.filter((p) => p.name.startsWith(prefix));
  return {
    count: hits.reduce((acc, p) => acc + p.numel, 0),
    shapes: hits.map((p) => `${p.name.slice(prefix.lastIndexOf(".") + 1) || p.name} (${p.shape.join(", ")})`),
    hits,
  };
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
