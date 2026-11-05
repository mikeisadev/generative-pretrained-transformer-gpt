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
  // Valori derivati usati dai testi ({{...}}): calcolati una volta sola qui.
  spec.vars = {
    ...cfg,
    head_size: spec.head_size,
    ffw: 4 * cfg.n_embd,
    block_size_minus_1: cfg.block_size - 1,
    total: spec.total_params.toLocaleString("it-IT", { useGrouping: "always" }),
    ln_vocab: Math.log(cfg.vocab_size).toFixed(2).replace(".", ","),
    prompt: escapeHtml(JSON.stringify(spec.example.prompt).slice(1, -1)),
  };
  return spec;
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
