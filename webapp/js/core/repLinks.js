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

/**
 * Parole per la ricerca globale: i comandi che le viste Rete neurale e Cervello hanno in ogni tab (nella
 * scheda della vista). Lo studente cerca il pulsante con il suo nome ("guarda lo spessore", "anima
 * impulsi"): la palette le aggiunge alle voci view:rete e view:cervello di ogni indice (withRepKeywords).
 * Solo nomi distintivi: parole comuni come "regioni" o "fetta" porterebbero le 12 viste in cima a ricerche
 * che cercano altro.
 */
export const REP_SEARCH_KEYWORDS = Object.freeze({
  rete: Object.freeze(["guarda lo spessore", "tutta la rete", "scala e onestà", "attivazioni vere", "simulazione didattica"]),
  cervello: Object.freeze(["anima impulsi", "neuroni 3D", "sagittale", "coronale", "assiale", "mostra nel modello 3D"]),
});

/** Le parole chiave di una voce dell'indice di un tab, più quelle dei comandi comuni se è view:rete o view:cervello. */
export function withRepKeywords(item) {
  const extra = item?.kind === "view" && typeof item.id === "string" && Object.hasOwn(REP_SEARCH_KEYWORDS, item.id) ? REP_SEARCH_KEYWORDS[item.id] : null;
  return extra ? [...(Array.isArray(item.keywords) ? item.keywords : []), ...extra] : item?.keywords;
}
