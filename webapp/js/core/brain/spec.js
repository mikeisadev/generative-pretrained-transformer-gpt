/**
 * Specifica del cervello di un tab (tabs/<id>/brain.js esporta `brainSpec`, soli dati): validazione e
 * normalizzazione. Pura (niente three.js, niente DOM): la usano la vista 3D, il pannello e i test.
 *
 *   const spec = normalizeBrainSpec(brainSpec);          // avvisi in spec.warnings, mai eccezioni
 *   normalizeBrainSpec(brainSpec, { strict: true });     // BrainSpecError con l'elenco dei problemi
 *
 * Il formato è documentato in ARCHITECTURE.md ("Cervello"). Accetta anche i nomi dei campi della
 * ricerca (brain_data.json: neuron_type, count_shown, real_count, real_count_text, analogy_map, kind
 * "eccitatoria" | "inibitoria" | "modulatoria", strength "debole/metafora"), così i dati verificati si
 * copiano nei tab senza tradurli a mano.
 */
import { atlasEntry, atlasKind } from "./atlas.js";
import { resolveNeuronType } from "./morphology.js";

/** Tipi di sinapsi del microcircuito: etichetta, colore (legenda e 3D) e forma del contatto. */
export const EDGE_KINDS = Object.freeze({
  excitatory: { label: "eccitatoria", color: "#fb923c", text: "aumenta la probabilità che il neurone di arrivo scarichi (glutammato)." },
  inhibitory: { label: "inibitoria", color: "#60a5fa", text: "la diminuisce (GABA): frena e mette in fila nel tempo." },
  modulatory: { label: "modulatoria", color: "#c084fc", text: "non decide da sola lo scarico ma cambia il guadagno o la plasticità (es. dopamina, ritorno corticotalamico)." },
});

const KIND_ALIASES = new Map([
  ["excitatory", "excitatory"],
  ["eccitatoria", "excitatory"],
  ["eccitatorio", "excitatory"],
  ["inhibitory", "inhibitory"],
  ["inibitoria", "inhibitory"],
  ["inibitorio", "inhibitory"],
  ["modulatory", "modulatory"],
  ["modulatoria", "modulatory"],
  ["modulatorio", "modulatory"],
]);

/** Forza di un'analogia (criteri della ricerca: legame storico o sperimentale, principio comune, metafora). */
export const STRENGTHS = Object.freeze({
  forte: { label: "forte", text: "legame storico diretto (l'idea dell'AI è nata dalla biologia, o viceversa) o confronto sperimentale diretto." },
  media: { label: "media", text: "un principio comune solido, senza un legame diretto." },
  debole: { label: "debole (metafora)", text: "una somiglianza di nomi o di intuizione: utile per ricordare, non per spiegare." },
});

/** Limiti: oltre questi numeri la scena diventa illeggibile (o pesante con il rendering software). */
export const SPEC_LIMITS = Object.freeze({ groups: 12, neuronsPerGroup: 18, edges: 40, analogies: 30 });

/** Errore di una specifica non valida (modalità strict): `errors` = elenco leggibile. */
export class BrainSpecError extends Error {
  constructor(errors) {
    super(`Specifica del cervello non valida: ${errors.join("; ")}`);
    this.name = "BrainSpecError";
    this.errors = errors;
  }
}

const text = (v) => (typeof v === "string" ? v.trim() : "");
const asArray = (v) => (Array.isArray(v) ? v : v === undefined || v === null ? [] : [v]);

/** Solo link http(s): nel pannello un "javascript:" o un indirizzo relativo strano non diventano mai link. */
export function safeUrl(url) {
  if (typeof url !== "string") return null;
  try {
    const u = new URL(url.trim());
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch {
    return null;
  }
}

/** Una fonte: { title, authors, year, venue, url } (url solo se http/https). Stringa → solo titolo. */
export function normalizeSource(source) {
  if (typeof source === "string") return { title: source.trim(), authors: "", year: null, venue: "", url: null };
  if (!source || typeof source !== "object") return null;
  const title = text(source.title);
  if (!title) return null;
  const year = Number.isFinite(Number(source.year)) && source.year !== "" && source.year !== null ? Number(source.year) : null;
  return { id: text(source.id) || null, title, authors: text(source.authors), year, venue: text(source.venue), url: safeUrl(source.url ?? (source.doi ? `https://doi.org/${source.doi}` : null)), note: text(source.note) };
}

/**
 * Normalizza la specifica di un tab.
 * @param {object} raw
 * @param {{ strict?: boolean }} [o]
 * @returns {{ title, intro, regions: string[], markers: string[], pathways: object[], circuit: object | null,
 *   analogies: object[], facts: object[], sources: object[], highlight: string[], warnings: string[] }}
 */
export function normalizeBrainSpec(raw, { strict = false } = {}) {
  const warnings = [];
  const errors = [];
  const warn = (msg) => warnings.push(msg);
  if (!raw || typeof raw !== "object") {
    if (strict) throw new BrainSpecError(["la specifica non è un oggetto"]);
    raw = {};
    warn("specifica vuota");
  }

  // regioni e marcatori evidenziati (un id di marcatore dentro `regions` va bene: finisce tra i marcatori)
  const regions = [];
  const markers = [];
  const addId = (id, where) => {
    const kind = atlasKind(id);
    if (kind === "region") !regions.includes(id) && regions.push(id);
    else if (kind === "marker") !markers.includes(id) && markers.push(id);
    else {
      const msg = `${where}: "${id}" non è nell'atlante`;
      if (strict) errors.push(msg);
      else warn(msg);
    }
  };
  for (const id of asArray(raw.regions)) addId(id, "regions");
  for (const id of asArray(raw.markers)) addId(id, "markers");

  // percorsi: id dell'atlante (eventualmente con etichetta propria) o percorsi nuovi con `through`
  const pathways = [];
  for (const p of asArray(raw.pathways)) {
    const item = typeof p === "string" ? { id: p } : p;
    if (!item || typeof item !== "object" || !text(item.id)) {
      warn("pathways: voce senza id ignorata");
      continue;
    }
    const atlas = atlasKind(item.id) === "pathway" ? atlasEntry(item.id) : null;
    const through = asArray(item.through ?? atlas?.through).filter((id) => {
      const ok = atlasKind(id) === "region" || atlasKind(id) === "marker";
      if (!ok) warn(`pathways.${item.id}.through: "${id}" non è nell'atlante`);
      return ok;
    });
    if (!atlas && through.length < 2) {
      const msg = `pathways: "${item.id}" non è nell'atlante e non ha almeno due regioni in "through"`;
      if (strict) errors.push(msg);
      else warn(msg);
      continue;
    }
    pathways.push({
      id: item.id,
      label: text(item.label) || atlas?.name || item.id,
      text: text(item.text) || atlas?.text || "",
      through,
      color: /^#[0-9a-f]{6}$/i.test(item.color ?? "") ? item.color : atlas?.color ?? "#5eead4",
      atlas: Boolean(atlas),
    });
  }

  const circuit = normalizeCircuit(raw.circuit, { warn, error: (msg) => (strict ? errors.push(msg) : warn(msg)) });

  // analogie
  const analogies = [];
  const seen = new Set();
  const rawAnalogies = asArray(raw.analogies);
  for (const [index, a] of rawAnalogies.entries()) {
    if (analogies.length >= SPEC_LIMITS.analogies) {
      warn(`analogies: al più ${SPEC_LIMITS.analogies} analogie, ${rawAnalogies.length - index} ignorate`);
      break;
    }
    if (!a || typeof a !== "object") continue;
    const id = text(a.id) || `analogia-${index + 1}`;
    if (seen.has(id)) warn(`analogies: id ripetuto "${id}"`);
    seen.add(id);
    const strengthKey = normalizeStrength(a.strength);
    if (!strengthKey) warn(`analogies.${id}: forza "${a.strength}" sconosciuta (forte, media, debole)`);
    const aRegions = asArray(a.regions).filter((r) => {
      const ok = atlasKind(r) === "region" || atlasKind(r) === "marker";
      if (!ok) warn(`analogies.${id}.regions: "${r}" non è nell'atlante`);
      return ok;
    });
    const aMarkers = asArray(a.markers).filter((r) => atlasKind(r) === "marker");
    const aPathways = asArray(a.pathways).filter((r) => atlasKind(r) === "pathway" || pathways.some((p) => p.id === r));
    if (!text(a.ai) || !text(a.brain)) warn(`analogies.${id}: servono "ai" e "brain"`);
    if (!text(a.different)) warn(`analogies.${id}: manca "different" (dove l'analogia non regge)`);
    const sources = asArray(a.sources).map(normalizeSource).filter(Boolean);
    if (!sources.length) warn(`analogies.${id}: nessuna fonte`);
    analogies.push({
      id,
      ai: text(a.ai),
      aiSection: text(a.ai_section ?? a.aiSection),
      brain: text(a.brain),
      regions: aRegions,
      markers: aMarkers,
      pathways: aPathways,
      text: text(a.text),
      similar: text(a.similar),
      different: text(a.different),
      strength: strengthKey ?? "media",
      strengthLabel: STRENGTHS[strengthKey ?? "media"].label,
      sources,
    });
  }

  const facts = asArray(raw.facts)
    .map((f) => (f && typeof f === "object" && text(f.claim) ? { claim: text(f.claim), value: text(f.value), note: text(f.note), source: normalizeSource(f.source) } : null))
    .filter(Boolean);
  const sources = asArray(raw.sources).map(normalizeSource).filter(Boolean);

  if (errors.length) throw new BrainSpecError(errors);

  // le regioni da evidenziare: quelle dichiarate, oppure tutte quelle usate da analogie e circuito
  const used = new Set([...regions, ...analogies.flatMap((a) => a.regions.filter((r) => atlasKind(r) === "region")), ...(circuit?.groups ?? []).map((g) => g.region).filter(Boolean)]);
  const highlight = regions.length ? regions : [...used];
  const usedMarkers = new Set([...markers, ...analogies.flatMap((a) => [...a.markers, ...a.regions.filter((r) => atlasKind(r) === "marker")]), ...(circuit?.groups ?? []).map((g) => g.marker).filter(Boolean)]);
  return {
    title: text(raw.title) || "Il cervello",
    intro: text(raw.intro),
    regions,
    markers: [...usedMarkers],
    pathways,
    circuit,
    analogies,
    facts,
    sources,
    highlight,
    warnings,
  };
}

/** "forte" | "media" | "debole" (accetta "debole/metafora", maiuscole, spazi). */
export function normalizeStrength(value) {
  const v = text(value).toLowerCase();
  if (v.startsWith("forte") || v === "strong") return "forte";
  if (v.startsWith("media") || v === "medium") return "media";
  if (v.startsWith("debole") || v.includes("metafora") || v === "weak") return "debole";
  return null;
}

/** Tipo di sinapsi canonico (accetta l'italiano della ricerca). */
export function normalizeEdgeKind(value) {
  return KIND_ALIASES.get(text(value).toLowerCase()) ?? null;
}

/** Microcircuito: gruppi in ordine da sinistra a destra (come la rete AI), sinapsi tra gruppi. */
function normalizeCircuit(raw, { warn, error }) {
  if (!raw || typeof raw !== "object") return null;
  const aiOf = new Map(asArray(raw.analogy_map ?? raw.analogyMap).map((m) => [m?.group, text(m?.ai_component ?? m?.ai)]));
  const groups = [];
  const ids = new Set();
  for (const g of asArray(raw.groups)) {
    if (!g || typeof g !== "object" || !text(g.id)) {
      warn("circuit.groups: gruppo senza id ignorato");
      continue;
    }
    if (ids.has(g.id)) {
      error(`circuit.groups: id ripetuto "${g.id}"`);
      continue;
    }
    if (groups.length >= SPEC_LIMITS.groups) {
      warn(`circuit.groups: al più ${SPEC_LIMITS.groups} gruppi, "${g.id}" ignorato`);
      continue;
    }
    const typeName = g.type ?? g.neuron_type ?? g.neuronType;
    const type = resolveNeuronType(typeName);
    if (!type) warn(`circuit.groups.${g.id}: tipo di neurone "${typeName}" sconosciuto (uso la forma generica)`);
    const region = atlasKind(g.region) === "region" ? g.region : null;
    if (g.region && !region) warn(`circuit.groups.${g.id}: regione "${g.region}" non è nell'atlante`);
    const marker = atlasKind(g.marker) === "marker" ? g.marker : null;
    let count = Math.round(Number(g.count ?? g.count_shown ?? g.countShown ?? 6));
    if (!Number.isFinite(count) || count < 1) count = 1;
    if (count > SPEC_LIMITS.neuronsPerGroup) {
      warn(`circuit.groups.${g.id}: ${count} neuroni sono troppi da disegnare, ne mostro ${SPEC_LIMITS.neuronsPerGroup}`);
      count = SPEC_LIMITS.neuronsPerGroup;
    }
    const real = Number(g.real ?? g.real_count ?? g.realCount);
    ids.add(g.id);
    groups.push({
      id: g.id,
      label: text(g.label) || g.id,
      region,
      marker,
      type: type ?? "projection",
      count,
      real: Number.isFinite(real) && real > 0 ? real : null,
      realText: text(g.realText ?? g.real_count_text),
      ai: text(g.ai) || aiOf.get(g.id) || "",
      note: text(g.note),
    });
  }
  const edges = [];
  const rawEdges = asArray(raw.edges);
  for (const [index, e] of rawEdges.entries()) {
    if (edges.length >= SPEC_LIMITS.edges) {
      warn(`circuit.edges: al più ${SPEC_LIMITS.edges} sinapsi, ${rawEdges.length - index} ignorate`);
      break;
    }
    if (!e || typeof e !== "object") continue;
    if (!ids.has(e.from) || !ids.has(e.to)) {
      error(`circuit.edges: "${e.from}" → "${e.to}" collega un gruppo che non esiste`);
      continue;
    }
    const kind = normalizeEdgeKind(e.kind);
    if (!kind) warn(`circuit.edges: tipo "${e.kind}" sconosciuto in ${e.from} → ${e.to} (uso "eccitatoria")`);
    edges.push({ from: e.from, to: e.to, kind: kind ?? "excitatory", label: text(e.label), note: text(e.note), pathway: atlasKind(e.pathway) === "pathway" ? e.pathway : null });
  }
  if (!groups.length) return null;
  return {
    title: text(raw.title) || "Microcircuito",
    text: text(raw.text),
    caveat: text(raw.caveat),
    groups,
    edges,
    sources: asArray(raw.sources).map(normalizeSource).filter(Boolean),
  };
}
