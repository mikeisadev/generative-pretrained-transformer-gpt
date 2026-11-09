/**
 * Rete neurale del tab "LLM moderni": la rappresentazione "Rete neurale" del nucleo (core/neuralNet,
 * formato della `networkSpec` in ARCHITECTURE.md, "Rete neurale orizzontale").
 *
 * Modulo PURO (niente three.js, niente DOM): `buildNetworkSpec(model)` costruisce la specifica dal modulo
 * di soli dati del modello (full/models/*.js, già verificato per la vista completa): ogni numero dei
 * sottotitoli e dei testi viene da lì o è calcolato da lì, mai copiato a mano. index.js la carica con
 * `import("./network.js").then((m) => m.networkSpec)`: la rete del modello principale, DeepSeek-V4.1-Flash,
 * la stessa nei due sottotab.
 *
 * Da sinistra a destra (x): token → embedding → residuo a 4 corsie (mHC) → un layer per ogni blocco della
 * striscia dei Blocchi, nell'ordine vero e con i gruppi ripetuti compressi in «× N» (networkMechanisms.js,
 * drawnLayers) → norm finale → head → softmax → DSpark. Dall'alto in basso (y): i numeri del vettore di
 * ogni token, ridotti (SHOWN). In profondità (z): i token del prompt d'esempio (PROMPT).
 * Dentro ogni layer ci sono solo i pezzi che il config gli dà (layerRecipe): KV latente con la finestra,
 * KV globale compressa (2 → 1) o proiettata da H20, indexer, attention MQA, router ed esperti.
 *
 * Onestà: i pesi di V4.1 non sono nel browser. Attivazioni (simulate dal nucleo), pesi di attention,
 * voci scelte dall'indexer ed esperti scelti dal router sono ILLUSTRATIVI e ogni sezione lo dice; la
 * scala ridotta è dichiarata nei sottotitoli ("qui 32") e in "Scala e onestà".
 *
 * Gli altri due modelli della vista completa (Kimi K3.1 Pro, gpt-oss-120b) hanno un'attention che il
 * config non descrive con gli stessi campi (KDA lineare, GQA a finestre alterne): buildNetworkSpec li
 * rifiuta (drawnLayers lancia un errore) invece di disegnarli sbagliati.
 */
import { formatCount, formatLayers, formatShare, layerByNumber, layerParams, paramTotals } from "./full/model.js";
import { MODEL_BY_ID, PRIMARY_MODEL_ID } from "./full/models/index.js";
import { drawnLayers, layerAttention, selectEntries } from "./networkMechanisms.js";

/**
 * Il prompt in profondità: una domanda nel formato di chat di DeepSeek (token speciali del tokenizer di
 * V4.1: full/models/deepseekV41Flash.js → input, tokenizer) con l'inizio della risposta già generato.
 * La divisione delle parole in token è ILLUSTRATIVA: il tokenizer vero non è nel browser.
 */
export const PROMPT = Object.freeze({
  tokens: Object.freeze(["<｜begin▁of▁sentence｜>", "<｜User｜>", "Perché", "il", "cielo", "è", "blu", "?", "<｜Assistant｜>", "<think>", "La", "luce"]),
  /** I primi `prompt` token sono la domanda con il formato di chat; gli altri il modello li ha già scritti, uno alla volta. */
  prompt: 10,
});

/**
 * Quanto si disegna (il modello vero nel commento). Ridotto perché i 40 layer stiano in una scena
 * navigabile anche con il rendering software; i sottotitoli e "Scala e onestà" lo dichiarano.
 * `window` ed `entries` mettono in miniatura la finestra locale e le voci globali scelte: in un prompt di 12
 * token la finestra vera (128) li coprirebbe tutti e non si vedrebbe che i vicini si leggono nitidi e i
 * lontani attraverso i riassunti.
 */
export const SHOWN = Object.freeze({
  hidden: 32, // numeri del vettore di ogni token (5.120)
  stream: 8, // numeri di ognuna delle 4 corsie del residuo mHC (5.120 ciascuna)
  latent: 8, // KV latente (512)
  indexer: 4, // indexer (32 head × 128)
  heads: 4, // head di query (64)
  experts: 16, // esperti instradati (384)
  expertNeurons: 6, // neuroni nascosti di un esperto (2.304)
  vocab: 16, // voci del vocabolario (129.280)
  window: 4, // finestra locale (128 token)
  entries: 2, // voci globali scelte dall'indexer per ogni token (512)
  draftHeads: 2, // head disegnate nei blocchi draft di DSpark
});

/** Le parti di un layer, nell'ordine del flusso (quelle tra parentesi solo se la ricetta le prevede). */
const LAYER_PARTS = ["norm1", "kv", "(gkv)", "(idx)", "attn", "add1", "norm2", "router", "experts", "add2"];

const it = (n, digits = 0) => Number(n).toLocaleString("it-IT", { useGrouping: "always", minimumFractionDigits: digits, maximumFractionDigits: digits });
// percentuali scritte come nel resto del tab ("98,5%": formatShare della vista completa)
const percent = (x) => formatShare(x);
const list = (items) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} e ${items.at(-1)}`);
/** "ENCODER causale" → "Encoder causale" (le etichette dei segmenti nel modulo dati sono in maiuscolo). */
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
/** Articolo davanti a un numero in cifre, come si legge: «lo 0», «l'8», «l'11», «il 25». */
const theNumber = (n) => (n === 0 ? "lo 0" : /^(1|8|11|8\d|8\d\d)$/.test(String(n)) ? `l'${n}` : `il ${n}`);
const paragraphs = (...parts) => parts.filter(Boolean).join("\n\n");
const isSpecial = (token) => /^<.*>$/.test(token);

/** Il testo leggibile di una fila di token (senza i token speciali): "Perché il cielo è blu?". */
export function readableText(tokens) {
  return tokens
    .filter((t) => !isSpecial(t))
    .join(" ")
    .replace(/ ([?!.,;:])/g, "$1");
}

/** Frasi di onestà in fondo a ogni sezione: che cosa, lì, è illustrativo. */
function honesty(model) {
  const weights = `i pesi di ${model.short} non sono nel browser`;
  return {
    values: `Valori illustrativi: ${weights}.`,
    probs: `Probabilità illustrative: ${weights}.`,
    attention: `Pesi di attention illustrativi, calcolati con un seme: ${weights}.`,
    indexer: `Le voci scelte qui sono illustrative: la scelta vera dipende dai pesi, e ${weights}.`,
    routing: `Quali esperti si accendono è illustrativo: la scelta vera dipende dai pesi del router, e ${weights}.`,
    tokens: "Divisione in token illustrativa: il tokenizer vero potrebbe spezzare le parole in modo diverso.",
  };
}

/**
 * La specifica della rete di un modello della vista completa (soli dati, per core/neuralNet).
 * @param {object} [model] modulo dati di full/models (predefinito: il modello principale)
 */
export function buildNetworkSpec(model = MODEL_BY_ID.get(PRIMARY_MODEL_ID)) {
  const c = model.config;
  const T = PROMPT.tokens.length;
  const totals = paramTotals(model);
  const drawn = drawnLayers(model);
  const ctx = { model, c, T, totals, drawn, honest: honesty(model), prefixOf: new Map(), selections: new Map(), sections: [], links: [], groups: [] };

  inputPart(ctx);
  let previous = "mhc";
  for (const d of drawn.blocks) {
    previous = layerPart(ctx, d, previous);
    // il gruppo «× N» della striscia si chiude dopo il suo ultimo blocco: la ripetizione segue subito
    if (d.group?.to === d.index) ctx.sections.push(repeatOf(ctx, d.group));
  }
  outputPart(ctx, previous);
  draftPart(ctx, previous);
  segmentGroups(ctx);

  const routedShare = totals.components.find((x) => x.id === "routed")?.share ?? 0;
  const engram = model.layers.flags?.engram?.layers ?? [];
  const unverified = model.unverified.filter((u) => u.id === "kv-qat" || u.id === "clamp").map((u) => u.value);
  return {
    title: `Rete neurale di ${model.name} (ridotta)`,
    intro: paragraphs(
      `${model.name} disegnato come rete neurale: da sinistra a destra il flusso, dal tokenizer alla softmax; dall'alto in basso i numeri del vettore di ogni token; in profondità i ${T} token del prompt. Gira la camera («Guarda lo spessore»): vedi la finestra locale, la KV globale compressa con l'indexer che sceglie le voci, e il router che manda ogni token ai suoi ${c.topK} esperti.`,
      `Prompt d'esempio: la domanda «${readableText(PROMPT.tokens.slice(0, PROMPT.prompt))}» con i token speciali del tokenizer di ${model.short} (inizio, utente, assistente e ragionamento <think>) messi come nel formato di chat di DeepSeek, poi l'inizio della risposta già scritto dal modello: «${readableText(PROMPT.tokens.slice(PROMPT.prompt))}». I ${T} token sono scritti ai lati della rete e nel menu «Fetta». ${ctx.honest.tokens} Anche l'ordine dei token speciali è indicativo: il template di chat di ${model.short} non è tra le fonti verificate.`,
      `È il modello dell'«Architettura LLM moderni completa»: la stessa rete anche dalla vista d'insieme, qualunque modello sia scelto nel selettore.`,
    ),
    scale: [
      `1 sfera = 1 numero di un vettore. Qui ${SHOWN.hidden} numeri per token invece di ${it(c.hidden)}; le ${c.hcMult} corsie del residuo da ${SHOWN.stream} invece di ${it(c.hidden)}; ${SHOWN.latent} dei ${it(c.headDim)} numeri del KV latente; ${SHOWN.heads} delle ${c.heads} head; ${SHOWN.vocab} delle ${it(c.vocab)} voci del vocabolario.`,
      `Esperti: ${SHOWN.experts} dei ${it(c.experts)} instradati (più ${c.sharedExperts} condiviso), ${SHOWN.expertNeurons} dei ${it(c.expertHidden)} neuroni di ciascuno. Ogni token ne accende ${c.topK} come nel modello vero: su ${SHOWN.experts} sono il ${percent(c.topK / SHOWN.experts)}, su ${it(c.experts)} l'${percent(c.topK / c.experts)}.`,
      `Layer: ${drawn.blocks.length} disegnati dei ${c.layers}, uno per blocco della striscia dei Blocchi e nell'ordine vero; i gruppi ripetuti sono lastre fantasma con «× N».`,
      `Attention in miniatura: finestra locale di ${SHOWN.window} token invece di ${it(c.window)} e ${SHOWN.entries} voci globali scelte invece di ${it(c.indexTopK)}. In un prompt di ${T} token la finestra vera li coprirebbe tutti e non si vedrebbe la differenza tra i vicini, letti nitidi, e i lontani, letti attraverso i riassunti.`,
    ].join(" "),
    note: [
      `Valori illustrativi: i pesi di ${model.name} non sono nel browser, quindi attivazioni, pesi di attention, voci scelte dall'indexer ed esperti scelti dal router sono generati con un seme (badge «simulazione didattica»). Veri sono la struttura, l'ordine dei layer, i pezzi di ogni layer (dal config: compress_ratios, kv_source_layer_ids, index_source_layer_ids) e i numeri dei sottotitoli, dal modulo dati dell'Architettura completa.`,
      `Non disegnati: ${engram.length ? `la memoria Engram (layer ${list(engram.map(String))}), ` : ""}la visione (ViT), l'interno dei blocchi draft di DSpark e il replay del decoder sugli ultimi token. Gli esperti instradati sono il ${percent(routedShare)} dei parametri del backbone.`,
      unverified.length ? `Da verificare, come nella vista completa: ${list(unverified)}.` : "",
    ]
      .filter(Boolean)
      .join(" "),
    seed: 41,
    axes: { x: "flusso: dal prompt al prossimo token", y: `numeri del vettore (${SHOWN.hidden} dei ${it(c.hidden)})`, z: "token del prompt" },
    depth: { count: T, label: "token", names: [...PROMPT.tokens] },
    sections: ctx.sections,
    links: ctx.links,
    groups: ctx.groups,
    sources: model.sources.map((s) => ({ title: s.title, url: s.url, note: s.note })),
  };
}

// ------------------------------------------------------------------------------------------------
// Ingresso: token, embedding, residuo a 4 corsie
// ------------------------------------------------------------------------------------------------
function inputPart(ctx) {
  const { model, c, sections, links, groups, honest } = ctx;
  const embedding = model.input.find((b) => b.id === "embedding");
  const residual = model.input.find((b) => b.id === "residual");
  sections.push(
    {
      id: "tok",
      label: "Token (BPE)",
      sub: `${it(c.vocab)} voci · qui ${SHOWN.vocab}`,
      role: "input",
      neurons: SHOWN.vocab,
      depth: true,
      activation: "onehot",
      real: c.vocab,
      text: paragraphs(
        `Il testo entra a pezzi: il tokenizer BPE a byte ha ${it(c.vocab)} voci, token speciali compresi, e trasforma il prompt in una fila di numeri interi. Ogni token è un neurone acceso su ${it(c.vocab)}: qui il vocabolario è ridotto a ${SHOWN.vocab} voci, quindi il neurone acceso è solo indicativo.`,
        `I token speciali segnano il dialogo: <｜User｜> e <｜Assistant｜> dicono chi parla, <think> apre il ragionamento. Come un mosaico: le parole frequenti sono una tessera sola, quelle rare si spezzano in più tessere.`,
        honest.tokens,
      ),
    },
    {
      id: "emb",
      label: "Embedding",
      sub: `${embedding.sub} · qui ${SHOWN.hidden}`,
      role: "embedding",
      neurons: SHOWN.hidden,
      depth: true,
      real: c.hidden,
      text: paragraphs(
        `Una riga della tabella per ogni voce del vocabolario: ${embedding.sub} pesi, ${it(model.params.embedding / 1e6, 1)} milioni di parametri. Ogni token prende la sua riga, come una carta d'identità di ${it(c.hidden)} numeri; da qui ogni token è un vettore che attraversa la rete restando alla sua profondità.`,
        honest.values,
      ),
    },
    {
      id: "mhc",
      label: `mHC: 1 → ${c.hcMult} corsie`,
      sub: `${residual.sub} · ${c.hcMult} × ${it(c.hidden)} · qui ${c.hcMult} × ${SHOWN.stream}`,
      role: "add",
      neurons: SHOWN.stream,
      slabs: c.hcMult,
      slabNames: lanes(c.hcMult),
      depth: true,
      real: c.hcMult * c.hidden,
      text: paragraphs(
        `Novità rispetto al nostro GPT: il residuo non è una strada sola ma un'autostrada a ${c.hcMult} corsie (mHC, Manifold-Constrained Hyper-Connections), ognuna da ${it(c.hidden)} numeri: le vedi come ${c.hcMult} lastre una accanto all'altra. Lungo tutta la rete ogni layer legge un mix delle corsie e riscrive su tutte.`,
        `I mescolamenti sono matrici «doppiamente stocastiche»: ogni riga e ogni colonna sommano a 1 (${c.sinkhornIters} iterazioni di Sinkhorn), così nessuna corsia si svuota o trabocca lungo i ${c.layers} layer.`,
        honest.values,
      ),
    },
  );
  links.push(
    { from: "tok", to: "emb", pattern: "lookup", label: "ogni token prende la sua riga" },
    { from: "emb", to: "mhc", pattern: "dense", toSlab: "all", density: 0.03, label: `l'embedding entra nelle ${c.hcMult} corsie` },
  );
  groups.push({ label: "Ingresso", from: "tok", to: "mhc", text: "Dal testo ai vettori: token, embedding e le corsie del residuo." });
}

const lanes = (n) => Array.from({ length: n }, (_, i) => `corsia ${i + 1}`);

// ------------------------------------------------------------------------------------------------
// Un layer: il rappresentante di un blocco della striscia
// ------------------------------------------------------------------------------------------------
/** @returns {string} l'id della sezione che porta il residuo al layer dopo */
function layerPart(ctx, d, previous) {
  const { model, c, T, sections, links, groups, prefixOf, selections } = ctx;
  const r = d.recipe;
  const p = `l${d.rep}`;
  prefixOf.set(d.index, p);
  const kind = model.kinds[d.block.kind];
  const id = (part) => `${p}.${part}`;
  // in fondo a ogni sezione: di quali layer veri è il rappresentante ("Layer 20 (Full ← H20)." se è uno solo)
  const many = d.block.layers.length > 1;
  const layer = { where: many ? `Layer ${kind.label}: ${formatLayers(d.block.layers)}, ne vedi uno (${theNumber(d.rep)}).` : `Layer ${d.rep} (${kind.label}).` };

  // la scelta delle voci globali: calcolata dai layer sorgente dell'indexer, riusata dagli altri
  if (r.indexes) selections.set(d.index, selectEntries({ tokens: T, m: r.compress, k: SHOWN.entries, seed: `${model.id}:${d.rep}` }));
  const selection = d.indexBlock >= 0 ? selections.get(d.indexBlock) : null;
  const attention = layerAttention({ tokens: T, heads: SHOWN.heads, window: SHOWN.window, m: r.compress, selection, seed: `${model.id}:${d.rep}` });
  const kvSource = d.kvBlock >= 0 ? `${prefixOf.get(d.kvBlock)}.gkv` : null;

  // --- attention
  sections.push(normSection(ctx, id("norm1"), "attn", layer), kvSection(ctx, id("kv"), r, layer));
  links.push(
    { from: previous, to: id("norm1"), pattern: "dense", fromSlab: "all", density: 0.012, label: `mHC: le ${c.hcMult} corsie → 1` },
    { from: id("norm1"), to: id("kv"), pattern: "dense", density: 0.045, label: `wkv: ${it(c.hidden)} → ${it(c.headDim)}` },
  );
  if (r.writesKV) {
    sections.push(globalKvSection(ctx, id("gkv"), r, layer));
    if (r.compress > 1) links.push({ from: id("norm1"), to: id("gkv"), pattern: "down", density: 0.15, label: `compressore: ${r.compress} token → 1 voce` });
    else links.push({ from: encoderOutput(ctx, r), to: id("gkv"), pattern: "dense", fromSlab: "all", density: 0.03, label: `H${r.fromSegment.to + 1} (uscita dell'encoder) → KV globale` });
  }
  if (r.indexes) {
    sections.push(indexerSection(ctx, id("idx"), r, layer));
    links.push({ from: id("norm1"), to: id("idx"), pattern: "dense", density: 0.12, label: "query dell'indexer" });
  }
  sections.push(attentionSection(ctx, id("attn"), r, layer));
  links.push({ from: id("norm1"), to: id("attn"), pattern: "dense", toSlab: 0, density: 0.012, label: `query: wq_a ${it(c.hidden)} → ${it(c.qLora)} → wq_b ${c.heads} × ${it(c.headDim)}` });
  if (r.indexes) links.push({ from: id("idx"), to: id("attn"), pattern: "dense", toSlab: 0, density: 0.08, label: `voci scelte (top-${it(c.indexTopK)}, qui ${SHOWN.entries})` });
  links.push({ from: id("kv"), to: id("attn"), pattern: "attention", heads: SHOWN.heads, weights: attention.window, label: `finestra locale: ${it(c.window)} token, qui ${SHOWN.window}` });
  if (kvSource) links.push({ from: kvSource, to: id("attn"), pattern: "attention", heads: SHOWN.heads, weights: attention.entries, label: r.writesKV ? "voci globali di questo layer" : `voci globali riusate dal layer ${r.kvSource}` });
  sections.push(addSection(ctx, id("add1"), "attn", layer));
  links.push(
    { from: id("attn"), to: id("add1"), pattern: "dense", toSlab: "all", density: 0.01, label: `wo_a → wo_b (${c.oGroups} gruppi): ritorno a ${it(c.hidden)}, poi mHC 1 → ${c.hcMult}` },
    { from: previous, to: id("add1"), pattern: "residual", label: "residuo" },
  );

  // --- MoE
  sections.push(normSection(ctx, id("norm2"), "ffn", layer), routerSection(ctx, id("router"), layer), expertsSection(ctx, id("experts"), id("router"), layer), addSection(ctx, id("add2"), "ffn", layer));
  links.push(
    { from: id("add1"), to: id("norm2"), pattern: "dense", fromSlab: "all", density: 0.012, label: `mHC: le ${c.hcMult} corsie → 1` },
    { from: id("norm2"), to: id("router"), pattern: "dense", density: 0.022, label: `ffn.gate: ${it(c.hidden)} → ${it(c.experts)} punteggi` },
    { from: id("router"), to: id("experts"), pattern: "route", density: 0.08, label: `ogni token ai suoi ${c.topK} esperti (+ ${c.sharedExperts} condiviso)` },
    { from: id("experts"), to: id("add2"), pattern: "route", density: 0.04, label: `somma pesata × ${it(c.routedScaling, 1)}` },
    { from: id("add1"), to: id("add2"), pattern: "residual", label: "residuo" },
  );

  groups.push({ label: layerLabel(d, kind), from: id("norm1"), to: id("add2"), text: layerGroupText(ctx, d, kind) });
  return id("add2");
}

/** Le sezioni di un layer disegnato, nell'ordine del flusso (per le ripetizioni e i test). */
export function layerSectionIds(prefix, recipe) {
  return LAYER_PARTS.filter((part) => (part === "(gkv)" ? recipe.writesKV : part === "(idx)" ? recipe.indexes : true)).map((part) => `${prefix}.${part.replace(/[()]/g, "")}`);
}

/** Etichetta breve del layer disegnato: "Layer SWA", "5 layer Reuse" (un pezzo di un gruppo «× N»), … */
function layerLabel(d, kind) {
  return d.group?.kind === "group" && d.block.count > 1 ? `${d.block.count} layer ${kind.label}` : `Layer ${kind.label}`;
}

function layerGroupText(ctx, d, kind) {
  const { model } = ctx;
  const lp = layerParams(model, layerByNumber(model, d.rep));
  const engram = d.block.flags.filter((f) => f.id === "engram").map((f) => `${f.layers.length === 1 ? "Il layer" : "I layer"} ${list(f.layers.map(String))} ${f.layers.length === 1 ? "ha" : "hanno"} in più la memoria Engram (non disegnata).`);
  // i numeri veri dei layer (formatLayers): "25-27, 29-31, 33-35, 37-39", non "12 layer · 25 … 39", che
  // sembrerebbe una fila continua (i layer 28, 32 e 36 sono Reindex)
  const n = d.block.layers.length;
  const which = n > 1 ? `Layer ${formatLayers(d.block.layers)} (${n} layer): ne vedi uno, ${theNumber(d.rep)}.` : `Layer ${d.rep}: è l'unico di questo tipo.`;
  return paragraphs(`${kind.name}. ${kind.text}`, `${which} Circa ${formatCount(lp.total)} parametri per layer, ${formatCount(lp.parts.routed)} negli esperti.${engram.length ? ` ${engram.join(" ")}` : ""}`);
}

/** La sezione con l'uscita dell'encoder (H20): il residuo dopo il layer disegnato che contiene il suo ultimo layer. */
function encoderOutput(ctx, r) {
  const holder = ctx.drawn.blocks.find((b) => b.block.layers.includes(r.fromSegment.to));
  if (!holder) throw new Error(`nessun layer disegnato contiene il layer ${r.fromSegment.to}`);
  return `${ctx.prefixOf.get(holder.index)}.add2`;
}

function normSection(ctx, id, which, layer) {
  const { model, c, honest } = ctx;
  const tensor = which === "attn" ? model.anatomy.norm.attnTensor : model.anatomy.norm.ffnTensor;
  return {
    id,
    label: "RMSNorm",
    sub: `mHC ${c.hcMult} → 1 · ${it(c.hidden)} · qui ${SHOWN.hidden}`,
    role: "norm",
    neurons: SHOWN.hidden,
    depth: true,
    real: c.hidden,
    text: paragraphs(
      `Prima ${which === "attn" ? "dell'attention" : "del MoE"} le ${c.hcMult} corsie si fondono in un vettore solo (mHC ${c.hcMult} → 1), poi la RMSNorm lo riporta a una scala comune: divide per la radice della media dei quadrati e moltiplica per i pesi γ (${tensor}). Lavora token per token: nessun arco in profondità. Come regolare il volume di ogni voce prima di ascoltarla.`,
      layer.where,
      honest.values,
    ),
  };
}

function kvSection(ctx, id, r, layer) {
  const { model, c, honest } = ctx;
  const content = c.headDim - c.ropeDim;
  return {
    id,
    label: "KV latente",
    sub: `${it(c.headDim)} (${it(content)} + ${c.ropeDim} RoPE) · qui ${SHOWN.latent}`,
    role: "latent",
    neurons: SHOWN.latent,
    depth: true,
    real: c.headDim,
    text: paragraphs(
      `Proiezione nel latente: wkv (${it(c.hidden)} → ${it(c.headDim)}) e kv_norm danno a ogni token un solo vettore da ${it(c.headDim)} numeri (${it(content)} di contenuto + ${c.ropeDim} ruotati dalla RoPE). È MQA, il caso estremo di GQA: ${c.kvHeads === 1 ? "una sola KV head" : `${c.kvHeads} KV head`}, e lo stesso vettore fa da key e da value (K = V). Tutte le ${c.heads} head di query leggono questo latente.`,
      `Gli archi che partono da qui sono la finestra locale: ogni token guarda gli ultimi ${it(c.window)} token (qui ${SHOWN.window}), nitidi, come le ultime righe che rileggi mentre scrivi. Ogni layer tiene la sua finestra in FP8: ${it(model.kv.swaBytesPerTokenLayer)} B per token.${r.compress === 0 ? ` Questo layer ha solo la finestra: nessuna KV globale (RoPE con θ = ${it(c.ropeTheta)}).` : ""}`,
      layer.where,
      honest.values,
    ),
  };
}

function globalKvSection(ctx, id, r, layer) {
  const { model, c, T, totals, honest } = ctx;
  if (r.compress > 1) {
    const self = model.kv.compare.find((x) => x.id === "self");
    const gqa = model.kv.compare.find((x) => x.id === "llm67b");
    const qat = model.unverified.some((u) => u.id === "kv-qat") ? " (che la voce FP4 venga dal QAT è da verificare)" : "";
    return {
      id,
      label: "KV globale",
      sub: `${r.compress} token → 1 voce · ${it(model.kv.entryBytes)} B · qui ${SHOWN.latent}`,
      role: "latent",
      neurons: SHOWN.latent,
      depth: T / r.compress,
      real: c.headDim,
      text: paragraphs(
        `La memoria lunga del layer: il compressore (compressor.wkv + wgate) fonde ogni ${r.compress} token in 1 voce da ${it(c.headDim)} numeri, salvata in FP4: ${it(model.kv.entryBytes)} B${qat}. In profondità si vede: ${T} token diventano ${T / r.compress} voci. Come gli appunti a margine: ogni ${r.compress} righe un riassunto.`,
        `Solo i layer ${list(c.kvSourceLayers.map(String))} scrivono la KV globale; gli altri layer compressi la riusano: sono gli archi lunghi che partono da qui verso i layer Reuse. In tutto ${it(self.bytes)} B per token${gqa ? `, circa ${it(model.kv.ratio)} volte meno di ${gqa.label}` : ""}.`,
        layer.where,
        honest.values,
      ),
    };
  }
  const h = `H${r.fromSegment.to + 1}`;
  return {
    id,
    label: "KV globale",
    sub: `1 voce per token · da ${h} · qui ${SHOWN.latent}`,
    role: "latent",
    neurons: SHOWN.latent,
    depth: true,
    real: c.headDim,
    text: paragraphs(
      `Il decoder non comprime: ogni token ha la sua voce globale, ed è una proiezione dell'uscita dell'encoder (${h}, l'arco lungo che arriva da sinistra), non dello stato di questo layer. Si calcola una volta per tutto il prompt: per questo il prefill esegue solo l'encoder, ${formatCount(totals.prefill)} parametri attivi per token invece di ${formatCount(totals.active)}.`,
      `Come una segretaria che riassume i documenti una volta sola: tutti i layer del decoder leggono il suo riassunto.`,
      layer.where,
      honest.values,
    ),
  };
}

function indexerSection(ctx, id, r, layer) {
  const { c, honest } = ctx;
  const reindexLayers = c.indexSourceLayers.filter((n) => !c.kvSourceLayers.includes(n));
  const twoLevel = r.compress === 1 && c.candidateBlocks ? `Nel decoder la scelta è a due livelli: prima ${it(c.candidateBlocks)} blocchi candidati da ${c.candidateBlockSize} voci, poi le ${it(c.indexTopK)} migliori dentro i candidati.` : "";
  return {
    id,
    label: "Indexer",
    sub: `${c.indexHeads} × ${c.indexHeadDim} · qui ${SHOWN.indexer} · top-${it(c.indexTopK)}, qui ${SHOWN.entries}`,
    role: "attention",
    neurons: SHOWN.indexer,
    depth: true,
    real: c.indexHeads * c.indexHeadDim,
    text: paragraphs(
      r.writesKV
        ? `Il lightning indexer sceglie quali voci della KV globale leggere: per ogni token dà un punteggio veloce a ogni voce (${c.indexHeads} head da ${c.indexHeadDim}, in FP4: Σ w · ReLU(q · k)) e tiene le ${it(c.indexTopK)} migliori (qui ${SHOWN.entries}). È il bibliotecario che per ogni domanda ti porta le schede giuste dallo schedario, invece di farti rileggere tutto.`
        : `Reindex: questo layer non riscrive la KV globale (riusa quella del layer ${r.kvSource}) ma ricalcola quali voci leggere. Succede nei layer ${list(reindexLayers.map(String))}: ogni tanto il «bibliotecario» aggiorna le schede, e i layer Reuse dopo di lui usano la nuova scelta.`,
      twoLevel,
      layer.where,
      honest.indexer,
    ),
  };
}

function attentionSection(ctx, id, r, layer) {
  const { c, honest } = ctx;
  let how;
  if (r.compress === 0) how = `Solo finestra: in questo layer non c'è KV globale, gli archi arrivano solo dai ${it(c.window)} token più vicini (qui ${SHOWN.window}).`;
  else if (r.writesKV) how = "Gli archi arrivano dalla finestra (KV latente) e dalle voci globali di questo layer, quelle scelte dall'indexer.";
  // Reindex: la KV globale è di un altro layer, ma la scelta delle voci è sua (il suo indexer, a sinistra)
  else if (r.indexes) how = `Reindex: nessuna KV globale propria. Gli archi lunghi vengono dalla KV globale del layer ${r.kvSource}, ma le voci le sceglie l'indexer di questo layer, che ricalcola il top-${it(c.indexTopK)}; in più c'è la sua finestra, come in ogni layer.`;
  else how = `Reuse: nessuna KV globale propria. Gli archi lunghi vengono dalla KV globale del layer ${r.kvSource}, con le voci scelte dall'indexer del layer ${r.indexSource}; questo layer calcola solo la sua finestra.`;
  const theta = r.compress === 0 ? it(c.ropeTheta) : `${it(c.compressRopeTheta)} e YaRN`;
  const shared = c.heads / c.kvHeads > 1 ? ` Come ${c.heads} lettori con domande diverse che consultano la stessa copia degli appunti.` : "";
  return {
    id,
    label: "Attention MQA",
    sub: `${c.heads} query · ${c.kvHeads} KV head · qui ${SHOWN.heads}`,
    role: "attention",
    neurons: SHOWN.hidden,
    heads: SHOWN.heads,
    slabs: 2,
    slabNames: ["query", "uscita"],
    depth: true,
    real: c.heads * c.headDim,
    text: paragraphs(
      `Qui ogni token raccoglie informazioni dagli altri. Lastra «query»: la query del token (${it(c.hidden)} → ${it(c.qLora)} → ${c.heads} head da ${it(c.headDim)}, RoPE su ${c.ropeDim} dimensioni con θ = ${theta}). ${how} Poi l'uscita torna larga ${it(c.hidden)} con wo_a e wo_b (${c.oGroups} gruppi): la proiezione nel latente e il ritorno.`,
      `Causale: nessun arco va verso il futuro. Un colore per head (qui ${SHOWN.heads} delle ${c.heads}), più luce = più peso; un «sink» per head può prendersi parte del peso per non guardare nulla (non disegnato).${shared}`,
      layer.where,
      honest.attention,
    ),
  };
}

function addSection(ctx, id, which, layer) {
  const { c, honest } = ctx;
  return {
    id,
    label: "+ residuo (mHC)",
    sub: which === "ffn" ? `somma pesata × ${it(c.routedScaling, 1)} · ${c.hcMult} corsie · qui ${c.hcMult} × ${SHOWN.stream}` : `${c.hcMult} corsie × ${it(c.hidden)} · qui ${c.hcMult} × ${SHOWN.stream}`,
    role: "add",
    neurons: SHOWN.stream,
    slabs: c.hcMult,
    slabNames: lanes(c.hcMult),
    depth: true,
    real: c.hcMult * c.hidden,
    text: paragraphs(
      which === "attn"
        ? `L'uscita dell'attention si somma alle ${c.hcMult} corsie (mHC 1 → ${c.hcMult}); l'arco che salta il blocco è il residuo, il traffico che continua sulla sua corsia. Il layer aggiunge informazione, non la sostituisce. Le corsie si mescolano con una matrice doppiamente stocastica (pesi hc_attn_* in FP32).`
        : `La somma pesata degli esperti scelti (pesi normalizzati sui ${c.topK}, moltiplicati per ${it(c.routedScaling, 1)}) più l'esperto condiviso torna nelle ${c.hcMult} corsie, insieme al residuo. Qui finisce il layer.`,
      layer.where,
      honest.values,
    ),
  };
}

function routerSection(ctx, id, layer) {
  const { model, c, honest } = ctx;
  const r = model.anatomy.ffn.router;
  return {
    id,
    label: "Router",
    sub: `${c.scoring} → top-${c.topK} · ${it(c.experts)} · qui ${SHOWN.experts}`,
    role: "router",
    neurons: SHOWN.experts,
    depth: true,
    real: c.experts,
    text: paragraphs(
      `Per ogni token un punteggio per ognuno dei ${it(c.experts)} esperti (qui ${SHOWN.experts}): ${r.tensor}, funzione ${c.scoring}. Solo per scegliere si aggiunge un bias di bilanciamento (${r.tensor2}: senza loss ausiliaria, con un bias a parte per i token immagine) e vincono i ${c.topK} più alti. È lo smistamento della posta: ogni lettera va a ${c.topK} uffici su ${it(c.experts)}.`,
      layer.where,
      honest.routing,
    ),
  };
}

function expertsSection(ctx, id, router, layer) {
  const { model, c, totals, honest } = ctx;
  const e = model.anatomy.ffn.expert;
  const share = totals.components.find((x) => x.id === "routed")?.share ?? 0;
  const clamp = model.unverified.some((u) => u.id === "clamp") ? " (la forma esatta del clamp è da verificare)" : "";
  return {
    id,
    label: "Esperti (MoE)",
    sub: `${it(c.experts)} + ${c.sharedExperts} · qui ${SHOWN.experts} + ${c.sharedExperts} · ${c.topK} attivi`,
    role: "experts",
    neurons: SHOWN.expertNeurons,
    depth: true,
    experts: SHOWN.experts,
    active: c.topK,
    shared: c.sharedExperts,
    router,
    real: c.expertHidden,
    text: paragraphs(
      `${it(c.experts)} piccole MLP SwiGLU (${it(c.hidden)} → ${it(c.expertHidden)} → ${it(c.hidden)}, pesi in ${e.dtype}) più ${c.sharedExperts} esperto condiviso, sempre acceso: la prima lastra, separata dalle altre. Ogni token accende solo i suoi ${c.topK}: girando la camera vedi la griglia esperti × token. Come un ospedale enorme: ogni paziente vede i ${c.topK} specialisti giusti più il medico di base.`,
      `Qui sta il ${percent(share)} dei parametri del backbone, ma per ogni token ne lavorano pochi: ${formatCount(totals.active)} attivi su ${formatCount(totals.backbone)}. Limite di ${it(c.swigluLimit)} sulle pre-attivazioni${clamp}. Qui ${SHOWN.experts} dei ${it(c.experts)}: ${c.topK} accesi su ${SHOWN.experts} sembrano tanti, su ${it(c.experts)} sono l'${percent(c.topK / c.experts)}.`,
      layer.where,
      honest.routing,
    ),
  };
}

/** La ripetizione «× N» di un gruppo della striscia: le sezioni dei suoi layer disegnati, una volta sola. */
function repeatOf(ctx, group) {
  const { model, drawn, prefixOf } = ctx;
  const blocks = drawn.blocks.filter((b) => b.index >= group.from && b.index <= group.to);
  const of = blocks.flatMap((b) => layerSectionIds(prefixOf.get(b.index), b.recipe));
  const unit = blocks.map((b) => (b.block.count > 1 && group.kind === "group" ? `${b.block.count} ${model.kinds[b.block.kind].label}` : model.kinds[b.block.kind].label)).join(" + ");
  const layers = `layer ${formatLayers(group.layers)}`;
  return {
    repeat: group.repeat,
    of,
    id: `rep.${group.id}`,
    label: `× ${group.repeat} ${layers}`,
    text: paragraphs(
      group.kind === "group"
        ? `Il blocco [${unit}] si ripete ${group.repeat} volte (${layers}): ne vedi uno vero, le lastre fantasma dicono che dopo ce ne sono altri uguali, con la stessa forma e pesi diversi. Come nella striscia dei Blocchi.`
        : // con 2 layer il nucleo disegna una sola lastra fantasma: «l'altro», non «gli altri»
          `${group.repeat} layer ${unit} di fila (${layers}): ne vedi uno vero, ${group.repeat === 2 ? "la lastra fantasma è l'altro" : "le lastre fantasma sono gli altri"}, con la stessa forma e pesi diversi.`,
      `Parametri del gruppo: ${formatCount(group.params)}.`,
    ),
  };
}

/** Parentesi grandi sopra i segmenti del modello (encoder causale, decoder): dopo quelle dei layer. */
function segmentGroups(ctx) {
  const { model, drawn, prefixOf, groups } = ctx;
  for (const s of model.layers.segments ?? []) {
    const inside = drawn.blocks.filter((b) => b.block.layers.every((n) => n >= s.from && n <= s.to));
    if (!inside.length) continue;
    const last = inside.at(-1);
    const end = last.group?.to === last.index ? `rep.${last.group.id}` : `${prefixOf.get(last.index)}.add2`;
    groups.push({ label: capitalize(s.label), from: `${prefixOf.get(inside[0].index)}.norm1`, to: end, color: s.color, text: `${capitalize(s.label)}: ${s.sub}.` });
  }
}

// ------------------------------------------------------------------------------------------------
// Uscita e DSpark
// ------------------------------------------------------------------------------------------------
function outputPart(ctx, previous) {
  const { model, c, T, sections, links, groups, honest } = ctx;
  const head = model.output.find((b) => b.id === "head");
  sections.push(
    {
      id: "fnorm",
      label: "Norm finale",
      sub: `mHC ${c.hcMult} → 1 · RMSNorm · qui ${SHOWN.hidden}`,
      role: "norm",
      neurons: SHOWN.hidden,
      depth: true,
      real: c.hidden,
      text: paragraphs(`Dopo l'ultimo layer le ${c.hcMult} corsie tornano una (mHC ${c.hcMult} → 1) e una RMSNorm finale (norm.weight, ${it(c.hidden)} pesi) prepara il vettore di ogni token per l'uscita.`, honest.values),
    },
    {
      id: "head",
      label: "Head (lm_head)",
      sub: `${head.sub} · qui ${SHOWN.vocab}`,
      role: "output",
      neurons: SHOWN.vocab,
      activation: "signed",
      real: c.vocab,
      text: paragraphs(
        `La matrice di uscita (${head.sub}, non legata all'embedding) dà un punteggio, il logit, a ogni voce del vocabolario. Serve solo all'ultima posizione: è lei che predice il token dopo «${PROMPT.tokens[T - 1]}». Qui ${SHOWN.vocab} delle ${it(c.vocab)} voci.`,
        honest.values,
      ),
    },
    {
      id: "softmax",
      label: "Softmax",
      sub: `→ prossimo token · qui ${SHOWN.vocab} di ${it(c.vocab)}`,
      role: "output",
      neurons: SHOWN.vocab,
      real: c.vocab,
      text: paragraphs("La softmax trasforma i punteggi in probabilità che sommano a 1; se ne sceglie una: il prossimo token. Poi il token si aggiunge al prompt e si ricomincia, uno alla volta.", honest.probs),
    },
  );
  links.push(
    { from: previous, to: "fnorm", pattern: "dense", fromSlab: "all", density: 0.012, label: `mHC: le ${c.hcMult} corsie → 1` },
    { from: "fnorm", to: "head", pattern: "dense", slice: T - 1, density: 0.3, label: "lm_head, solo dall'ultima posizione" },
    { from: "head", to: "softmax", pattern: "identity", label: "softmax" },
  );
  groups.push({ label: "Uscita", from: "fnorm", to: "softmax", text: "Dal vettore dell'ultimo token alla probabilità di ogni voce del vocabolario." });
}

function draftPart(ctx, previous) {
  const { model, c, T, sections, links, groups, honest } = ctx;
  const ds = c.dspark;
  const branch = model.branches.find((b) => b.id === "dspark");
  if (!ds || !branch) return;
  const targets = ds.targets.map(String);
  const draft = layerAttention({ tokens: T, heads: SHOWN.draftHeads, window: SHOWN.window, seed: `${model.id}:draft` });
  sections.push(
    {
      id: "mtp.fuse",
      label: `Fusione ${targets[0]}-${targets.at(-1)}`,
      sub: `layer ${targets.join(", ")} · ${it(ds.targets.length * c.hidden)} → ${it(c.hidden)}`,
      role: "latent",
      neurons: SHOWN.hidden,
      slabs: ds.targets.length,
      slabNames: targets.map((n) => `layer ${n}`),
      depth: true,
      real: c.hidden,
      text: paragraphs(
        `${branch.title}: la predizione di più token alla volta, costruita dentro il modello. Le feature dei layer ${list(targets)} (le ${ds.targets.length} lastre) si fondono con main_proj in un vettore da ${it(c.hidden)} per token.`,
        honest.values,
      ),
    },
    {
      id: "mtp.block",
      label: "Blocco draft",
      sub: `finestra ${it(c.window)} + MoE ${ds.experts} esperti, top-${ds.topK}`,
      role: "attention",
      neurons: SHOWN.hidden,
      heads: SHOWN.draftHeads,
      depth: true,
      real: c.hidden,
      text: paragraphs(
        `Ogni blocco draft ha una finestra di ${it(c.window)} token con sink e un MoE più piccolo di quello dei layer (${ds.experts} esperti, ${ds.topK} attivi): qui ne vedi solo l'attention a finestra (qui ${SHOWN.window} token e ${SHOWN.draftHeads} head); norme, residuo e MoE non sono disegnati. I blocchi sono ${ds.layers}, in fila.`,
        honest.attention,
      ),
    },
    { repeat: ds.layers, of: ["mtp.block"], id: "rep.mtp", label: `× ${ds.layers} blocchi draft`, text: `${ds.layers} blocchi draft uguali in fila: ne vedi uno, gli altri sono lastre fantasma.` },
    {
      id: "mtp.out",
      label: `Bozza: ${ds.blockSize} token`,
      sub: `Markov (rango ${ds.markovRank}) + confidenza · qui ${SHOWN.vocab} voci`,
      role: "output",
      neurons: SHOWN.vocab,
      slabs: ds.blockSize,
      slabNames: Array.from({ length: ds.blockSize }, (_, i) => `+${i + 1}`),
      real: c.vocab,
      text: paragraphs(
        `A ogni giro DSpark propone ${ds.blockSize} token, una lastra per token; una testa di Markov di rango ${ds.markovRank} corregge i logit e una testa di confidenza stima quanti verranno accettati. Il modello grande li verifica tutti in un passaggio: è la decodifica speculativa della vista d'insieme. Come un assistente che scrive la bozza di una frase e il capo che la approva in un colpo solo.`,
        honest.probs,
      ),
    },
  );
  links.push(
    { from: previous, to: "mtp.fuse", pattern: "dense", fromSlab: "all", toSlab: "all", density: 0.006, label: `feature dei layer ${list(targets)}` },
    { from: "mtp.fuse", to: "mtp.block", pattern: "dense", fromSlab: "all", density: 0.012, label: `main_proj: ${it(ds.targets.length * c.hidden)} → ${it(c.hidden)}` },
    { within: "mtp.block", pattern: "attention", causal: true, heads: SHOWN.draftHeads, weights: draft.window, label: `finestra locale: ${it(c.window)} token, qui ${SHOWN.window}` },
    { from: "mtp.block", to: "mtp.out", pattern: "dense", slice: T - 1, toSlab: "all", density: 0.1, label: `${ds.blockSize} token proposti dall'ultima posizione` },
  );
  groups.push({ label: "DSpark (MTP)", from: "mtp.fuse", to: "mtp.out", color: branch.color, text: `${branch.title}: ${branch.text}` });
}

/** La rete del modello principale della vista completa (DeepSeek-V4.1-Flash). */
export const networkSpec = buildNetworkSpec();
