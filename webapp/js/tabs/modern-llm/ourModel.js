/**
 * Il nostro GPT nei testi del tab: i numeri (parametri, contesto, vocabolario, dimensioni) vengono
 * da data/model.json, esportato da gpt/export.py, e non sono copiati a mano. I testi usano i
 * segnaposto {{chiave}} di fillTemplate (core/dom.js) con le chiavi restituite da modelVars().
 */
import { fetchJson } from "../../core/binary.js";
import { fmt } from "../../core/dom.js";

const MODEL_URL = new URL("../../../data/model.json", import.meta.url);
let pending = null;

/** model.json, scaricato una sola volta e condiviso (null se non disponibile). */
export function loadModel() {
  pending ??= fetchJson(MODEL_URL).catch(() => null);
  return pending;
}

/** Valori per i segnaposto dei testi; {} se il modello non è disponibile (i segnaposto restano). */
export function modelVars(model) {
  const c = model?.config;
  if (!c) return {};
  return {
    params: fmt(model.total_params),
    block: c.block_size,
    vocab: c.vocab_size,
    embd: c.n_embd,
    heads: c.n_head,
    layers: c.n_layer,
    hidden: 4 * c.n_embd, // FeedForward del nostro GPT: hidden = 4 * n_embd
    headSize: model.head_size,
  };
}
