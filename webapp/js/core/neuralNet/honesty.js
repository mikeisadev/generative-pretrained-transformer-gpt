/**
 * Onestà dei numeri di una rete (logica pura, testata con node): quali sezioni hanno attivazioni vere e
 * quali simulate, quali pesi espliciti sono veri e quali illustrativi. La scheda ne ricava il badge
 * ("Attivazioni vere", "in parte vere", "simulazione didattica") e l'elenco in "Scala e onestà".
 *
 * Perché un terzo stato: una rete mista (il RAG con l'embedder vero e l'encoder illustrativo, l'Audio con
 * 15 sezioni vere su 52) con il badge "simulazione didattica" direbbe falso sulle parti vere, con
 * "vere" sulle altre. I pesi di attention contano come attivazioni (sono una softmax calcolata dal
 * modello): espliciti ma illustrativi (`illustrative: true` nella specifica) non sono veri.
 */

/** Cifre all'italiana con il separatore delle migliaia sempre presente ("1.024"), come fmt di core/dom.js. */
export const int = (n) => Number(n).toLocaleString("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });

/** Il numero con il nome giusto: "1 neurone", "64 neuroni", "1.024 canali". */
export function plural(n, one, many) {
  return `${int(n)} ${n === 1 ? one : many}`;
}

/** Elenco leggibile: "a", "a e b", "a, b e c"; oltre `max` voci "… e altre N". */
export function listText(items, max = 12) {
  const shown = items.slice(0, max);
  const rest = items.length - shown.length;
  if (rest > 0) return `${shown.join(", ")} e altre ${int(rest)}`;
  return shown.length < 2 ? shown.join("") : `${shown.slice(0, -1).join(", ")} e ${shown.at(-1)}`;
}

/** Da dove vengono i pesi di attention di un collegamento: "real" (dati), "illustrative" (dati come schema), "simulated" (seme). */
export const attentionSource = (link) => (!link.weights ? "simulated" : link.illustrative ? "illustrative" : "real");

/**
 * @param {{ net: object, values: { simulated: Set<string> } }} model il modello (buildNetworkModel)
 * @returns {{ state: "real"|"partial"|"simulated", real: object[], simulated: object[],
 *   attention: { real: object[], illustrative: object[], simulated: object[] }, illustrativePairs: object[] }}
 */
export function valueSources({ net, values }) {
  const real = net.sections.filter((s) => !values.simulated.has(s.id));
  const simulated = net.sections.filter((s) => values.simulated.has(s.id));
  const attention = { real: [], illustrative: [], simulated: [] };
  for (const link of net.links) if (link.pattern === "attention") attention[attentionSource(link)].push(link);
  const illustrativePairs = net.links.filter((l) => l.pattern === "explicit" && l.illustrative);
  const anyReal = real.length > 0 || attention.real.length > 0;
  const allReal = simulated.length === 0 && attention.simulated.length === 0 && attention.illustrative.length === 0;
  return { state: allReal ? "real" : anyReal ? "partial" : "simulated", real, simulated, attention, illustrativePairs };
}

/** Badge della scheda: testo, classe e suggerimento (anche per chi non vede il colore). */
export function badgeOf(sources, seed) {
  if (sources.state === "real") return { text: "Attivazioni vere", className: "real", title: "I valori sono quelli calcolati dal modello." };
  if (sources.state === "simulated") return { text: "Attivazioni: simulazione didattica", className: "sim", title: `I valori non vengono da un modello: sono generati con un seme (${seed}) per mostrare il meccanismo.` };
  const total = sources.real.length + sources.simulated.length;
  const title = sources.simulated.length
    ? `Valori veri in ${plural(sources.real.length, "sezione", "sezioni")} su ${int(total)}; le altre sono una simulazione didattica (seme ${seed}). L'elenco è in «Scala e onestà».`
    : "Valori veri in tutte le sezioni; alcuni pesi di attention sono illustrativi o simulati: l'elenco è in «Scala e onestà».";
  return { text: "Attivazioni: in parte vere", className: "mixed", title };
}

/** Nome di un collegamento di attention per gli elenchi: "Attention (self)" o "Testo → Cross-attention". */
const attentionName = (l) => (l.within || l.from === l.to ? l.to.label : `${l.from.label} → ${l.to.label}`);
const unique = (items) => [...new Set(items)];

/**
 * Le frasi di "Scala e onestà" su che cosa è vero: testo semplice, una per riga, la prima sempre sui
 * valori (il badge porta qui: chi lo preme trova la risposta, anche quando è tutto vero o tutto simulato).
 */
export function honestyLines(sources, seed) {
  const lines = [];
  const { real, simulated, attention, illustrativePairs } = sources;
  if (sources.state === "real") lines.push(`Valori veri in tutte le sezioni: li ha calcolati il modello${attention.real.length ? ", come i pesi di attention" : ""}.`);
  if (sources.state === "simulated") lines.push(`Tutti i valori sono una simulazione didattica (seme ${seed}): mostrano il meccanismo, non le attivazioni di un modello vero.`);
  if (sources.state === "partial") {
    if (simulated.length) {
      lines.push(`Valori veri in ${plural(real.length, "sezione", "sezioni")} su ${int(real.length + simulated.length)}: ${listText(unique(real.map((s) => s.label)))}. ${simulated.length === 1 ? "L'altra è" : `Le altre ${int(simulated.length)} sono`} una simulazione didattica (seme ${seed}).`);
    } else lines.push("Valori veri in tutte le sezioni.");
  }
  if (sources.state !== "simulated" || attention.illustrative.length) {
    if (attention.illustrative.length) lines.push(`Pesi di attention illustrativi (mostrano lo schema, non vengono da un modello): ${listText(unique(attention.illustrative.map(attentionName)), 8)}.`);
    if (attention.simulated.length && sources.state !== "simulated") lines.push(`Pesi di attention simulati con il seme: ${listText(unique(attention.simulated.map(attentionName)), 8)}.`);
    if (attention.real.length && sources.state !== "real") lines.push(`Pesi di attention veri: ${listText(unique(attention.real.map(attentionName)), 8)}.`);
  }
  if (illustrativePairs.length) lines.push(`Pesi dei collegamenti illustrativi: ${listText(unique(illustrativePairs.map((l) => l.label || `${l.from.label} → ${l.to.label}`)), 8)}.`);
  return lines;
}
