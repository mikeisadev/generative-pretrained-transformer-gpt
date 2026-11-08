/**
 * Modello della rete (logica pura, testata con node): dalla specifica di soli dati a tutto ciò che
 * serve per disegnarla, in un colpo solo.
 *
 *   const model = buildNetworkModel(networkSpec);   // lancia NetworkSpecError se la specifica è sbagliata
 *   model.net      specifica normalizzata (spec.js)
 *   model.layout   posizioni x/y/z di ogni neurone, lastre fantasma, parentesi, residui (layout.js)
 *   model.values   attivazioni (vere o simulate), modi di colore, instradamento degli esperti (activations.js)
 *   model.links    segmenti, archi e pesi della fetta scelta, già ordinati per priorità (links.js)
 *   model.timings  millisecondi di ogni fase (per la prova di carico)
 *
 * La scena (scene.js) e la vista (view.js) partono da qui; un tab che vuole la rete dentro la SUA
 * scena (es. il Transformer con l'interruttore "A righe / Orizzontale") usa buildNetworkModel +
 * NetworkScene direttamente.
 */
import { resolveValues } from "./activations.js";
import { computeLayout } from "./layout.js";
import { buildLinks } from "./links.js";
import { normalizeSpec } from "./spec.js";

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/**
 * @param {object} spec la specifica del tab
 * @param {{ limits?: object }} [options] limits: tetti dei collegamenti (vedi LINK_LIMITS)
 */
export function buildNetworkModel(spec, { limits } = {}) {
  const t0 = now();
  const net = normalizeSpec(spec);
  const t1 = now();
  const layout = computeLayout(net);
  const t2 = now();
  const values = resolveValues(net);
  const t3 = now();
  const links = buildLinks(net, values, limits);
  const t4 = now();
  return { net, layout, values, links, timings: { spec: t1 - t0, layout: t2 - t1, values: t3 - t2, links: t4 - t3, total: t4 - t0 } };
}
