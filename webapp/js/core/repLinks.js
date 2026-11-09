/**
 * Le parti pure delle rappresentazioni (soli dati, niente three.js né DOM): id, link diretti e pannello
 * Cervello. Stanno qui e non in representations.js (che importa la pila di scena e quindi three.js) perché
 * i test in Node e gli indici della ricerca le possano importare.
 */

/** Id delle rappresentazioni, nell'ordine del selettore "Architettura · Rete neurale · Cervello". */
export const REP_IDS = Object.freeze(["main", "network", "brain"]);

/** Link diretti #<tab>/go=view:rete | view:cervello → id della rappresentazione. */
export const REP_LINKS = Object.freeze({ rete: "network", cervello: "brain" });

/** Id del pannello "Cervello" che installRepresentations aggiunge alla sidebar (sidebar.select(BRAIN_PANE_ID)). */
export const BRAIN_PANE_ID = "cervello";

/** La rappresentazione chiesta da un link diretto { kind, id } ("network" | "brain"), o null. */
export function repOfLink({ kind, id } = {}) {
  return kind === "view" && typeof id === "string" && Object.hasOwn(REP_LINKS, id) ? REP_LINKS[id] : null;
}
