/**
 * Rete neurale "Orizzontale con profondità" del Transformer: le parti della VISTA (link diretto, quote del
 * cursore "Pesi della fetta", quando ricostruire la rete, corrispondenza con la disposizione "A righe" di
 * neuronModel.js). La specifica della rete del GPT del corso (sezioni, valori veri, pesi, buildDepthSpec) sta in
 * js/shared/gpt/depthSpec.js, perché la usa anche il Laboratorio pesi: qui si riesporta tutta, così chi la
 * importava da questo file (neuronDepth.js, neurons.js, i test) continua a funzionare.
 * Logica pura: tests/transformer-depth.test.mjs.
 */
import { ATTENTION_SLABS } from "../../../shared/gpt/depthSpec.js";

export * from "../../../shared/gpt/depthSpec.js";

/** Id della disposizione per i link diretti e la ricerca: #transformer/go=view:neuroni-profondita. */
export const DEPTH_VIEW_ID = "neuroni-profondita";

/** Quote dei pesi più forti di ogni matrice disegnate sulla fetta scelta (cursore "Pesi della fetta"). */
export const DEPTH_FRACTIONS = Object.freeze([0.005, 0.01, 0.02, 0.05, 0.1, 0.25]);
/** La quota con cui la rete si ricostruisce una volta sola quando lo studente sale oltre quella costruita. */
const MAX_DEPTH_FRACTION = Math.max(...DEPTH_FRACTIONS);

/**
 * Se la rete in profondità va ricostruita (DepthLayout.build) e con quale quota. Il modello si riusa con lo
 * stesso prompt e gli stessi pesi (stessa `key`; null = pesi personalizzati, sempre nuovi) se ha abbastanza
 * pesi della fetta per la quota chiesta: quella mostrata la sceglie setSliceDensity, in pochi ms. Con lo
 * stesso prompt e una quota più alta si ricostruisce alla quota massima, così i cambi seguenti non
 * ricostruiscono più; con un prompt nuovo si costruisce alla quota chiesta (più in fretta). Con un solo token
 * i pesi della fetta stanno tra i collegamenti normali: lì vale solo la quota esatta.
 * @param {{key: string|null, builtKey?: string|null, builtFraction: number, T: number, fraction: number}} o
 *   `builtKey`: la chiave del modello costruito (undefined se non ce n'è uno)
 * @returns {{rebuild: boolean, fraction: number}} `fraction`: la quota con cui costruire
 */
export function depthBuildPlan({ key, builtKey, builtFraction, T, fraction }) {
  const same = key !== null && key === builtKey;
  const scalable = T > 1;
  const rebuild = !(same && (scalable ? fraction <= builtFraction : fraction === builtFraction));
  return { rebuild, fraction: same && scalable ? Math.max(MAX_DEPTH_FRACTION, fraction) : fraction };
}

/**
 * Neurone della disposizione "A righe" (indice in neuronModel.buildNeuronLayout) che corrisponde a
 * un'unità della rete orizzontale: sezione, lastra e neurone (la posizione è la stessa: la profondità).
 * Serve al clic: lo stesso dettaglio (explainNeuron) nelle due disposizioni. -1 se non esiste.
 */
export function depthUnitNeuron(layout, sectionId, slab, neuron) {
  const C = layout.config.n_embd;
  if (sectionId === "emb") return layout.byId.get(slab === 0 ? "tok_emb" : "pos_emb").offset + neuron;
  const attention = /^b(\d+)\.attn$/.exec(sectionId);
  if (attention) {
    const part = ATTENTION_SLABS[slab]?.part;
    if (part === undefined) return -1;
    return part === null ? layout.byId.get(`b${attention[1]}.concat`).offset + neuron : layout.byId.get(`b${attention[1]}.qkv`).offset + part * C + neuron;
  }
  const column = layout.byId.get(sectionId);
  return column && neuron < column.n ? column.offset + neuron : -1;
}

/**
 * Al contrario: il neurone k della disposizione "A righe" → sezione, lastra e neurone della rete
 * orizzontale. Le copie del vettore x a inizio riga diventano il vettore da cui sono copiate (x0 o
 * l'uscita del blocco prima). null se k non è un neurone.
 */
export function neuronDepthUnit(layout, k) {
  const column = layout.columns[layout.columnOf[k]];
  if (!column || k < column.offset || k >= column.offset + column.n) return null;
  const i = k - column.offset;
  const C = layout.config.n_embd;
  switch (column.key) {
    case "tok_emb":
      return { section: "emb", slab: 0, neuron: i };
    case "pos_emb":
      return { section: "emb", slab: 1, neuron: i };
    case "qkv":
      return { section: `b${column.layer}.attn`, slab: ATTENTION_SLABS.findIndex((s) => s.part === Math.floor(i / C)), neuron: i % C };
    case "concat":
      return { section: `b${column.layer}.attn`, slab: ATTENTION_SLABS.length - 1, neuron: i };
    case "x": {
      const L = layout.config.n_layer;
      const source = column.id === "out.x" ? `b${L - 1}.resid2` : column.layer === 0 ? "x0" : `b${column.layer - 1}.resid2`;
      return { section: source, slab: 0, neuron: i };
    }
    default:
      return { section: column.id, slab: 0, neuron: i };
  }
}
