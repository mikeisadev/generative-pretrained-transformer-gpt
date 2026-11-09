/**
 * Rete neurale del tab "Database vettoriale (RAG)": la rappresentazione "Rete neurale" del nucleo
 * (core/neuralNet, formato della `networkSpec` in ARCHITECTURE.md, "Rete neurale orizzontale").
 *
 * Modulo puro (niente three.js, niente DOM): `buildRagNetworkSpec({ kb, run })` costruisce la
 * specifica dai calcoli VERI del tab (base di conoscenza di pipeline.js e ricerca della domanda
 * corrente), così i numeri delle parti 1 e 2 sono esattamente quelli che il tab usa. index.js la
 * ricalcola ogni volta che lo studente apre la rete (vedi `createNetworkSource` in index.js).
 *
 * Da sinistra a destra, quattro parti (gruppi), come la pipeline del tab:
 *   1. il nostro embedder (js/shared/rag/embedder.js, NON è una rete neurale): feature della domanda →
 *      hashing con il segno → normalizzazione L2. Valori veri.
 *   2. il database vettoriale: vettori dei chunk in profondità, confronto con la domanda, similarità
 *      del coseno, top-k e soglia. Valori veri (gli stessi della ricerca esatta e di HNSW del tab).
 *   3. un encoder neurale aperto (multilingual-e5-small), al posto del text-embedding-3-small delle
 *      lezioni, la cui architettura non è pubblica: token in profondità, attention bidirezionale,
 *      "× 12", mean pooling, normalizzazione. Illustrativo (attivazioni simulate dal nucleo).
 *   4. l'LLM che risponde (Llama 3 8B, decoder causale): i token del prompt aumentato in profondità.
 *      Illustrativo.
 *
 * Perché le feature della parte 1 stanno in colonna (y) e non in profondità: per il nostro embedder
 * la domanda è un "sacchetto" di feature senza ordine, e l'hashing è una matrice fissa con un solo ±1
 * per feature. Così ogni feature ha UN collegamento vero verso la sua dimensione (schema "explicit");
 * in profondità il nucleo non saprebbe collegare ogni posizione a una dimensione diversa, e la "Fetta"
 * (che qui segue i chunk) accenderebbe feature a caso.
 *
 * Numeri dei modelli veri: dai config e dal codice ufficiali (vedi ENCODER, GENERATOR e `sources`).
 */
import { SHARED_DIMENSION, UNKNOWN_DIMENSION } from "../../shared/rag/embedder.js";
import { cosineSimilarity } from "../../shared/rag/similarity.js";
import { tokenize } from "../../shared/rag/tokenizer.js";
import { hnswCost } from "./hnswCost.js";

/**
 * multilingual-e5-small (Microsoft), verificato su fonti ufficiali:
 *   layer 12 e vettore da 384: README di microsoft/unilm/e5; 12 head e architettura BERT, tokenizer di
 *   XLM-R: README di Multilingual-MiniLM-L12-H384 (il modello da cui parte, metadati MTEB);
 *   118 milioni di parametri, 96.014.208 nell'embedding, 512 token, coseno: metadati MTEB;
 *   media sui token ("avg") + F.normalize, prefissi "query: " / "passage: ": codice di valutazione E5.
 * Calcoli nostri: vocabolario 96.014.208 / 384 = 250.037; head da 384 / 12 = 32; feed-forward
 * 4 × 384 = 1.536 (la regola di BERT: con questo valore tornano i 21 milioni di parametri del
 * transformer dichiarati per Multilingual-MiniLM).
 */
export const ENCODER = Object.freeze({
  name: "multilingual-e5-small",
  layers: 12,
  width: 384,
  heads: 12,
  ffn: 1536,
  vocab: 250037,
  maxTokens: 512,
  params: 118_000_000,
  queryPrefix: "query: ",
});

/**
 * Llama 3 8B (Meta): dim 4096, 32 layer, 32 head di query e 8 di key/value (GQA), vocabolario 128.256,
 * RoPE θ 500.000, ffn_dim_multiplier 1,3 e multiple_of 1024 → SwiGLU da 14.336 (sku_list.py di
 * meta-llama/llama-models); contesto 8.192 token (model card). Gli stessi numeri del tab LLM moderni.
 */
export const GENERATOR = Object.freeze({ name: "Llama 3 8B", layers: 32, width: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 128256, context: 8192 });

/** Quanto si disegna: tutto il vero dove sta in scena, un campione dichiarato dove no. */
export const SHOWN = Object.freeze({
  features: 48, // feature della domanda (le più pesanti, se sono di più)
  dims: 48, // dimensioni del vettore: quelle in cui la domanda non è zero (le più grandi, se sono di più)
  vocab: 12, // voci del vocabolario nei one-hot dei token
  // le parti illustrative più basse della parte vera: si guarda prima il RAG del tab, poi i modelli veri
  encoder: Object.freeze({ width: 16, heads: 4, ffn: 24, tokens: 24 }),
  generator: Object.freeze({ width: 16, heads: 4, ffn: 24, tokens: 24 }),
});

/** Id delle sezioni, nell'ordine del flusso (le ripetizioni "× N" non sono sezioni). */
export const SECTION_IDS = Object.freeze({
  embedder: ["emb.features", "emb.raw", "emb.q"],
  database: ["db.chunks", "db.compare", "db.cosine", "db.topk"],
  encoder: ["enc.tokens", "enc.embed", "enc.attn", "enc.add1", "enc.ffn", "enc.add2", "enc.pool", "enc.vec"],
  generator: ["llm.prompt", "llm.embed", "llm.attn", "llm.add1", "llm.ffn", "llm.add2", "llm.next"],
});

const KIND_LABEL = Object.freeze({ word: "radice", bigram: "coppia", char: "trigramma" });
const KIND_PLURAL = Object.freeze({ word: "radici", bigram: "coppie", char: "trigrammi" });
const KIND_ORDER = Object.freeze({ word: 0, bigram: 1, char: 2 });
/** Prima dimensione libera per l'hashing: le precedenti sono riservate dall'embedder (comune, sconosciute). */
const FIRST_HASHED = Math.max(SHARED_DIMENSION, UNKNOWN_DIMENSION) + 1;
// come fmt di core/dom.js: separatore delle migliaia anche a 4 cifre ("4.096", non "4096"), uguale alla scheda del nucleo
const it = (n, digits = 0) => Number(n).toLocaleString("it-IT", { useGrouping: "always", minimumFractionDigits: digits, maximumFractionDigits: digits });
const quoted = (text, max = 90) => `«${text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text}»`;

/**
 * Da dove viene la domanda disegnata (index.js, `networkSearch`): il campo del pannello Azioni oppure,
 * se è vuoto, l'ultima ricerca mostrata o una domanda d'esempio. L'introduzione lo dice com'è.
 */
const ORIGIN_TEXT = Object.freeze({
  panel: "quella del pannello Azioni. Per un'altra domanda scrivila lì e riapri la Rete neurale: i numeri delle parti 1 e 2 si ricalcolano.",
  last: "l'ultima cercata: il campo del pannello Azioni è vuoto. Scrivi lì una domanda e riapri la Rete neurale per vedere la tua.",
  example: "una domanda d'esempio: il campo del pannello Azioni è vuoto. Scrivi lì una domanda e riapri la Rete neurale per vedere la tua.",
});

/** Etichetta breve di un chunk, come shortId della stazione Ricerca: "rimborsi-p2:1" → "rimborsi p2·1". */
export function chunkLabel(id) {
  const [doc, rest] = String(id).split("-p");
  return rest ? `${doc} p${rest.replace(":", "·")}` : String(id);
}

/** Nome di una feature per l'etichetta del neurone: "radice «spes»", "trigramma «#li» (mai vista)". */
export function featureLabel(feature) {
  return `${KIND_LABEL[feature.kind] ?? feature.kind} «${feature.value}»${feature.known ? "" : " (mai vista)"}`;
}

/** Nome di una dimensione del vettore (le prime due sono riservate dall'embedder). */
export function dimLabel(d) {
  if (d === SHARED_DIMENSION) return `dim ${d} · comune`;
  if (d === UNKNOWN_DIMENSION) return `dim ${d} · sconosciute`;
  return `dim ${d}`;
}

/**
 * Le dimensioni da disegnare: quelle in cui il vettore della domanda non è zero (le altre moltiplicano
 * per zero e non cambiano nessun punteggio), in ordine; se sono più di `max`, le più grandi in valore
 * assoluto. Con una domanda fatta solo di parole vuote il vettore è nullo: si disegna la dimensione 0.
 */
export function questionDims(vector, max = SHOWN.dims) {
  const nonzero = [];
  for (let d = 0; d < vector.length; d++) if (vector[d] !== 0) nonzero.push(d);
  if (!nonzero.length) return { dims: [SHARED_DIMENSION], total: 0 };
  if (nonzero.length <= max) return { dims: nonzero, total: nonzero.length };
  const kept = [...nonzero].sort((a, b) => Math.abs(vector[b]) - Math.abs(vector[a]) || a - b).slice(0, max);
  return { dims: kept.sort((a, b) => a - b), total: nonzero.length };
}

/** Le feature da disegnare: le più pesanti (al più `max`), poi raggruppate per tipo (radici, coppie, trigrammi). */
export function shownFeatures(features, max = SHOWN.features) {
  const kept = [...features].sort((a, b) => b.weight - a.weight || (a.feature < b.feature ? -1 : 1)).slice(0, max);
  return kept.sort((a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind] || b.weight - a.weight || (a.feature < b.feature ? -1 : 1));
}

/** Testi dei token di un testo (stima di tokenizer.js), senza spazi e a capo: i nomi delle posizioni. */
export function textTokens(text) {
  return tokenize(text)
    .filter((t) => t.kind !== "space" && t.kind !== "newline")
    .map((t) => {
      const s = t.text.trim();
      return s.length > 16 ? `${s.slice(0, 15)}…` : s;
    })
    .filter(Boolean);
}

/** Token dell'encoder: <s>, "query: " + domanda, </s> (al più `max`, l'ultimo tagliato diventa "…"). */
export function encoderTokens(question, max = SHOWN.encoder.tokens) {
  const body = textTokens(ENCODER.queryPrefix + question);
  const room = max - 2;
  const cut = body.length > room ? [...body.slice(0, room - 1), "…"] : body;
  return { names: ["<s>", ...cut, "</s>"], total: body.length + 2 };
}

/**
 * Le posizioni del prompt aumentato disegnate in profondità: la domanda (per intero se ci sta) e
 * l'inizio del testo di ogni chunk scelto, almeno due token a testa; senza chunk, il testo che il
 * prompt mette al loro posto. Sono token veri del prompt del tab (stimati), non tutti: strumenti,
 * system prompt e tag <documento> ci sono ma non si disegnano.
 */
export function promptSample(run, max = SHOWN.generator.tokens) {
  const docs = run.results.length ? run.results.map((r) => textTokens(r.text)) : [textTokens(run.prompt.parts.find((p) => p.id === "documents-empty")?.content || "Nessun documento pertinente trovato.")];
  const all = textTokens(run.question);
  const minDocs = 2 * docs.length;
  const question = all.slice(0, Math.min(all.length, max - minDocs));
  const each = Math.max(2, Math.floor((max - question.length) / docs.length));
  const names = [...question];
  for (const doc of docs) names.push(...doc.slice(0, each));
  const byRole = run.prompt.tokens.byRole;
  return { names: names.slice(0, max), total: run.prompt.tokens.total, fixed: (byRole.tools ?? 0) + (byRole.system ?? 0) };
}

/**
 * La specifica della rete per la domanda corrente.
 * @param {{ kb: ReturnType<import("./pipeline.js").buildKnowledgeBase>, run: ReturnType<import("./pipeline.js").runSearch>, origin?: keyof ORIGIN_TEXT }} input
 *   origin: da dove viene la domanda (predefinito: il campo del pannello Azioni)
 */
export function buildRagNetworkSpec({ kb, run, origin = "panel" }) {
  const chunks = kb.inserts.map((r) => ({ id: r.entry.id, vector: r.vector }));
  const N = chunks.length;
  const ex = run.explanation;
  const features = shownFeatures(ex.features);
  const { dims, total: nonzero } = questionDims(ex.vector);
  const cosine = chunks.map((c) => cosineSimilarity(ex.vector, c.vector));
  const chosen = run.results.map((r) => ({ position: chunks.findIndex((c) => c.id === r.id), id: r.id, score: r.score }));
  const embedder = { dims: kb.embedder.dimensions, shared: kb.embedder.sharedComponent };
  const ctx = { kb, run, N, ex, features, dims, nonzero, chunks, cosine, chosen, embedder, prompt: promptSample(run), tokens: encoderTokens(run.question) };

  // i token dell'encoder e dell'LLM non sono i chunk delle fette: con slices: false restano accesi
  // (prima la fetta li accendeva in proporzione, senza legame con il chunk scelto)
  const ownDepth = (part) => ({ ...part, sections: part.sections.map((s) => (s.depth ? { ...s, slices: false } : s)) });
  const parts = [embedderPart(ctx), databasePart(ctx), ownDepth(encoderPart(ctx)), ownDepth(generatorPart(ctx))];
  return {
    title: "Rete neurale del RAG: dalla domanda alla risposta",
    intro: [
      "Il RAG di questo tab come rete, da sinistra a destra. Parti 1 e 2, con i numeri veri della domanda qui sotto: il nostro embedder (che non è una rete neurale) la trasforma in un vettore, il vettore si confronta con ogni chunk del database, in profondità, e passano i migliori. Parti 3 e 4, illustrative: le due reti neurali di un RAG vero, l'encoder che fa gli embedding e l'LLM che legge il prompt e scrive la risposta.",
      `Domanda: ${quoted(run.question)}, ${ORIGIN_TEXT[origin] ?? ORIGIN_TEXT.panel}`,
    ].join("\n\n"),
    scale: [
      `Parti 1 e 2 (vere): ${features.length < ex.features.length ? `le ${features.length} feature più pesanti delle ${ex.features.length}` : `tutte le ${ex.features.length} feature`} della domanda; ${dims.length < nonzero ? `${dims.length} delle ${nonzero} dimensioni` : `le ${dims.length} dimensioni`} (delle ${embedder.dims}) in cui il suo vettore non è zero; tutti i ${N} chunk del database (${kb.config.splitterPreset.label}).`,
      `Parte 3 (illustrativa): ${SHOWN.encoder.width} dei ${it(ENCODER.width)} numeri per token, ${SHOWN.encoder.heads} delle ${ENCODER.heads} head, uno dei ${ENCODER.layers} layer; one-hot su ${SHOWN.vocab} voci invece di ${it(ENCODER.vocab)}.`,
      `Parte 4 (illustrativa): ${SHOWN.generator.width} dei ${it(GENERATOR.width)} numeri per token, ${SHOWN.generator.heads} delle ${GENERATOR.heads} head, uno dei ${GENERATOR.layers} layer, ${ctx.prompt.names.length} dei ${it(ctx.prompt.total)} token del prompt; one-hot su ${SHOWN.vocab} voci invece di ${it(GENERATOR.vocab)}.`,
    ].join(" "),
    note: [
      "Attivazioni: vere nelle parti 1 e 2 (js/shared/rag e la pipeline del tab), simulate con un seme nelle parti 3 e 4: il badge «in parte vere» in alto lo dice (un clic sul badge apre l'elenco delle sezioni vere), e passando sopra un neurone la parola «simulato» compare solo nelle parti 3 e 4.",
      "La «Fetta» segue i chunk del database (parte 2) e i loro nomi stanno ai due lati del database. Nelle parti 3 e 4 la profondità sono i token: con i chunk non c'entrano, quindi restano accesi qualunque fetta tu scelga.",
      "I token delle parti 3 e 4 sono quelli stimati dal tab (tokenizer.js, che approssima cl100k_base): i tokenizer veri di E5 e di Llama 3 spezzerebbero le parole in modo un po' diverso.",
    ].join(" "),
    seed: 23,
    axes: {
      x: "flusso: dalla domanda alla risposta",
      y: "neuroni: feature, dimensioni del vettore, numeri di ogni token",
      z: "chunk del database (parte 2) e token del testo (parti 3 e 4)",
    },
    depth: { count: N, label: "chunk", names: chunks.map((c) => chunkLabel(c.id)) },
    sections: parts.flatMap((p) => p.sections),
    links: parts.flatMap((p) => p.links),
    // Prima le parti, poi le parentesi grandi che le contengono: a parità di importanza il nucleo piazza le
    // etichette nell'ordine dell'elenco, così "1 · …" e "2 · …" stanno sulle loro parentesi e "Valori veri"
    // sopra di loro (nell'ordine opposto il titolo grande, piazzato per primo, spingeva su o nascondeva i piccoli).
    // Etichette corte: di lato, a rete intera, la parentesi della parte 2 è larga poche decine di pixel.
    groups: [
      { label: "1 · Embedder", from: "emb.features", to: "emb.q", text: "Testo → feature → hashing → normalizzazione: un embedder a regole, senza una rete neurale, che simula text-embedding-3-small in modo dichiarato." },
      { label: "2 · Database", from: "db.chunks", to: "db.topk", text: "La ricerca di RagService.search: la domanda contro ogni chunk della tabella vector_store, poi topK e soglia come in PgVectorStore." },
      { label: "Valori veri", from: "emb.features", to: "db.topk", color: "#86efac", text: "Le parti 1 e 2 sono il tab vero: ogni numero è calcolato nel browser da js/shared/rag e dalla pipeline del tab (pipeline.js), per la domanda scritta nell'introduzione." },
      { label: "3 · Encoder E5", from: "enc.tokens", to: "enc.vec", text: "Come un modello neurale trasforma la domanda in un vettore: token in fila, attention in entrambe le direzioni, media dei token." },
      { label: "4 · LLM Llama 3", from: "llm.prompt", to: "llm.next", text: "Il decoder che legge il prompt aumentato (domanda + chunk scelti) e scrive la risposta un token alla volta." },
      { label: "Illustrativo: valori simulati", from: "enc.tokens", to: "llm.next", color: "#fcd34d", text: "Le parti 3 e 4 mostrano due modelli aperti come esempio, con i numeri dei loro config ufficiali. Le attivazioni sono simulate con un seme: servono a vedere i meccanismi, non sono calcoli veri." },
    ],
    sources: SOURCES,
  };
}

// ------------------------------------------------------------------------------------- parte 1
function embedderPart({ ex, features, dims, nonzero, embedder }) {
  const K = dims.length;
  const at = new Map(dims.map((d, k) => [d, k]));
  const F = Math.max(1, features.length);
  const unknown = features.filter((f) => !f.known).length;
  const kinds = Object.keys(KIND_LABEL)
    .map((kind) => [kind, features.filter((f) => f.kind === kind).length])
    .filter(([, n]) => n)
    .map(([kind, n]) => `${n} ${n === 1 ? KIND_LABEL[kind] : KIND_PLURAL[kind]}`)
    .join(", ");
  const dimsNote = K < nonzero ? `${K} delle ${nonzero} non nulle (le più grandi)` : `${K} delle ${embedder.dims} · le altre sono zero`;
  // la normalizzazione dell'embedder (embedder.js, explain): la parte "specifica" del vettore va a lunghezza
  // √(1 − s²) e la dimensione comune vale s, così il vettore intero è lungo 1 (s = sharedComponent, 0,66)
  const s = embedder.shared;
  const rest = Math.sqrt(1 - s * s);
  // l'hashing come matrice fissa: un collegamento per feature, verso la sua dimensione, peso ±1
  const pairs = [];
  features.forEach((f, i) => {
    const k = at.get(f.dimension);
    if (k !== undefined) pairs.push([i, k, f.known ? f.sign : 1]);
  });
  return {
    sections: [
      {
        id: "emb.features",
        label: "Feature della domanda",
        sub: features.length ? `${kinds}${features.length < ex.features.length ? ` · ${features.length} delle ${ex.features.length}` : ""}` : "nessuna feature",
        role: "input",
        neurons: F,
        activation: "positive",
        values: features.length ? features.map((f) => f.weight) : [0],
        neuronNames: features.length ? features.map(featureLabel) : ["nessuna feature: solo parole vuote"],
        valueLabel: "(peso tf-idf)",
        ...(features.length < ex.features.length ? { real: ex.features.length } : {}),
        text: [
          "Il nostro embedder non legge la frase come un modello neurale: la spezza in pezzi riconoscibili, le feature. Radici delle parole («spes» per spesa e spese), coppie di radici vicine e trigrammi di caratteri, che riconoscono le varianti della stessa parola.",
          "Ogni feature ha un peso tf-idf: pesa di più se compare nella domanda e di rado nei documenti, come una parola sottolineata in un libro in cui quasi nessuno sottolinea.",
          `Niente profondità: per questo embedder l'ordine delle parole non conta (tranne nelle coppie vicine), la domanda è un «sacchetto» di feature e le vedi in colonna.${unknown ? ` ${unknown === 1 ? "Una feature non compare" : `${unknown} feature non compaiono`} in nessun documento: passandoci sopra leggi «(mai vista)».` : ""} Valori veri, calcolati nel browser.`,
        ].join("\n\n"),
      },
      {
        id: "emb.raw",
        label: "Hashing: vettore grezzo",
        sub: dimsNote,
        role: "latent",
        neurons: K,
        real: embedder.dims,
        values: dims.map((d) => ex.raw[d]),
        neuronNames: dims.map(dimLabel),
        text: [
          `Ogni feature cade in una delle ${embedder.dims} dimensioni, scelta da una funzione di hash, con un segno + o − deciso anch'esso dall'hash: è il «trucco dell'hashing». Come smistare la posta in tante caselle guardando solo il codice postale: nessuno ha scritto una tabella, la casella la decide una formula. Le feature viste nei documenti vanno nelle dimensioni dalla ${FIRST_HASHED} alla ${embedder.dims - 1}: la ${SHARED_DIMENSION} e la ${UNKNOWN_DIMENSION} sono riservate.`,
          "I collegamenti a sinistra sono proprio questa regola: un solo collegamento per feature, peso +1 (arancione) o −1 (blu). Non è un peso imparato. Se due feature cadono nella stessa casella i loro contributi si sommano (collisione).",
          `Le feature mai viste nei documenti vanno tutte nella dimensione ${UNKNOWN_DIMENSION} e si sommano «in quadratura» (radice della somma dei quadrati): allungano il vettore ma non lo avvicinano a nessun documento. La dimensione ${SHARED_DIMENSION} qui è ancora vuota: la riempie il passo dopo.`,
        ].join("\n\n"),
      },
      {
        id: "emb.q",
        label: "Vettore della domanda",
        sub: nonzero ? `normalizzato, lunghezza 1 · ${embedder.dims} numeri, qui ${K < nonzero ? `${K} dei ${nonzero}` : K} non nulli` : "vettore nullo: nessuna feature utile",
        role: "latent",
        neurons: K,
        real: embedder.dims,
        values: dims.map((d) => ex.vector[d]),
        neuronNames: dims.map(dimLabel),
        text: nonzero
          ? [
              `Il vettore grezzo viene allungato o accorciato fino a lunghezza ${it(rest, 2)}; poi la dimensione ${SHARED_DIMENSION}, uguale per tutti i testi, prende il valore ${it(s, 2)}. In tutto il vettore è lungo 1 (${it(rest, 2)}² + ${it(s, 2)}² ≈ 1), come i vettori di OpenAI: così la similarità del coseno diventa un semplice prodotto scalare.`,
              `La dimensione ${SHARED_DIMENSION} simula una cosa vera dei modelli neurali, i cui vettori condividono sempre una componente comune. Per questo due testi scollegati hanno comunque similarità circa ${it(s * s, 2)} (${it(s, 2)} × ${it(s, 2)}) e la soglia 0,5 delle lezioni ha senso.`,
              "È lo stesso embedder usato per i chunk: domanda e documenti devono stare sulla stessa mappa (L18 §12.1).",
            ].join("\n\n")
          : [
              "Questa domanda è fatta solo di parole vuote: il vettore grezzo è nullo e non ha una direzione, quindi resta tutto zero, anche la componente comune della dimensione 0 (similarità 0 con ogni chunk).",
              `Di solito invece il vettore viene riscalato a lunghezza ${it(rest, 2)} e la dimensione ${SHARED_DIMENSION} vale ${it(s, 2)}: in tutto lunghezza 1, come i vettori di OpenAI.`,
            ].join("\n\n"),
      },
    ],
    links: [
      { from: "emb.features", to: "emb.raw", pattern: "explicit", pairs, label: "hashing: un ±1 per feature" },
      { from: "emb.raw", to: "emb.q", pattern: "identity", label: `riscalato a lunghezza ${it(rest, 2)}, più ${it(s, 2)} nella dimensione ${SHARED_DIMENSION}` },
    ],
  };
}

// ------------------------------------------------------------------------------------- parte 2
function databasePart({ kb, run, N, ex, dims, nonzero, chunks, cosine, chosen, embedder }) {
  const K = dims.length;
  const vectorValues = new Float32Array(K * N);
  const products = new Float32Array(K * N);
  dims.forEach((d, k) => {
    for (let n = 0; n < N; n++) {
      vectorValues[k * N + n] = chunks[n].vector[d];
      products[k * N + n] = ex.vector[d] * chunks[n].vector[d];
    }
  });
  const passed = new Float32Array(N);
  for (const c of chosen) passed[c.position] = c.score;
  const best = [...cosine.keys()].sort((a, b) => cosine[b] - cosine[a])[0];
  const partial = K < nonzero;
  const list = chosen.map((c) => `${chunkLabel(c.id)} (${it(c.score, 2)})`).join(", ");
  const shared = it(ex.vector[SHARED_DIMENSION] * chunks[0].vector[SHARED_DIMENSION], 4);
  const example = dims.find((d) => d > UNKNOWN_DIMENSION) ?? SHARED_DIMENSION; // una dimensione vera della domanda
  return {
    sections: [
      {
        id: "db.chunks",
        label: "Vettori dei chunk",
        sub: `vector_store · ${N} chunk × ${K} delle ${embedder.dims} dimensioni`,
        role: "latent",
        neurons: K,
        depth: true,
        real: embedder.dims,
        values: vectorValues,
        neuronNames: dims.map(dimLabel),
        text: [
          "Il database pieno, come alla fine dell'azione Indicizzazione: una riga della tabella vector_store per chunk, in profondità uno dietro l'altro. Ogni chunk è passato per lo stesso embedder della domanda.",
          `Disegniamo solo le dimensioni in cui la domanda non è zero: nelle altre i chunk hanno numeri diversi da zero, ma moltiplicati per lo zero della domanda non cambiano il punteggio.`,
          `Qui i chunk sono ${N}; in un database vero possono essere milioni, e la profondità diventerebbe un corridoio lunghissimo da percorrere a ogni domanda. Per questo serve l'indice HNSW.`,
        ].join("\n\n"),
      },
      {
        id: "db.compare",
        label: "Domanda × chunk",
        sub: `prodotti dimensione per dimensione · ${N} chunk × ${K}`,
        role: "data",
        neurons: K,
        depth: true,
        values: products,
        neuronNames: dims.map(dimLabel),
        valueLabel: "(q × c)",
        text: [
          `La domanda raggiunge ogni chunk (i collegamenti che si aprono in profondità) e si moltiplica con lui numero per numero: dimensione 0 con dimensione 0, la ${example} con la ${example}. Come confrontare due ricette ingrediente per ingrediente: tanto dello stesso ingrediente in entrambe dà un prodotto grande e positivo, un ingrediente che una ha e l'altra no dà zero.`,
          nonzero ? `La riga della dimensione 0 vale ${shared} per tutti i chunk: è la componente comune, uguale in tutti i vettori.` : "Qui la domanda è un vettore nullo: tutti i prodotti sono zero.",
        ].join("\n\n"),
      },
      {
        id: "db.cosine",
        label: "Similarità del coseno",
        sub: "un numero per chunk · score = 1 − distanza",
        role: "data",
        neurons: 1,
        depth: true,
        values: Float32Array.from(cosine),
        valueLabel: "(similarità)",
        text: [
          `La somma dei prodotti è la similarità del coseno (i vettori sono lunghi 1): 1 = stessa direzione, 0 = niente in comune. È lo score che Spring AI restituisce, 1 − la distanza <=> di pgvector. ${nonzero ? `La più alta: ${chunkLabel(chunks[best].id)} (${it(cosine[best], 3)}).` : "Qui il vettore della domanda è nullo: tutte le similarità valgono 0."}${partial ? ` La somma conta tutte le ${nonzero} dimensioni non nulle, anche le ${nonzero - K} non disegnate.` : ""}`,
          "I collegamenti della somma si vedono sul chunk scelto con la «Fetta» (senza fetta, su quello davanti).",
          `Qui il confronto è disegnato con tutti i chunk, come la ricerca esatta. ${hnswNote(run, N, kb.store.index.efSearch)}`,
        ].join("\n\n"),
      },
      {
        id: "db.topk",
        label: "Top-k e soglia",
        sub: `scelti ${chosen.length} su ${N} · topK ${run.topK}, soglia ${it(run.similarityThreshold, 2)}`,
        role: "output",
        neurons: 1,
        depth: true,
        activation: "positive",
        values: passed,
        valueLabel: "(score)",
        text: chosen.length
          ? [
              `Passano solo i chunk con score sopra la soglia (${it(run.similarityThreshold, 2)}) e, tra questi, al più ${run.topK}: i collegamenti arrivano solo a loro. Come un concorso con un voto minimo e un numero di posti.`,
              `Scelti dall'indice HNSW: ${list}.`,
              "Il loro TESTO, non il vettore, entra nel prompt dell'LLM: qui la rete si interrompe, perché il vettore serve solo a trovare il chunk. Lo ritrovi come token nella parte 4.",
            ].join("\n\n")
          : [
              `Nessun chunk supera la soglia ${it(run.similarityThreshold, 2)}: la domanda è fuori tema rispetto ai documenti, e i collegamenti non arrivano a nessuno.`,
              "Il prompt dirà «Nessun documento pertinente trovato» e l'LLM dovrà rispondere che non ha informazioni (system prompt della L18).",
            ].join("\n\n"),
      },
    ],
    links: [
      { from: "emb.q", to: "db.compare", pattern: "broadcast", label: "la domanda raggiunge ogni chunk" },
      { from: "db.chunks", to: "db.compare", pattern: "identity", label: "numero per numero" },
      // la somma: K pesi uguali a 1, disegnati sulla fetta scelta (sono gli stessi per ogni chunk)
      { from: "db.compare", to: "db.cosine", pattern: "explicit", slice: "selected", pairs: dims.map((_, k) => [k, 0, 1]), label: "somma" },
      // top-k e soglia: un collegamento per chunk scelto, alla sua posizione, con il suo score
      ...chosen.map((c) => ({ from: "db.cosine", to: "db.topk", pattern: "explicit", slice: c.position, pairs: [[0, 0, c.score]], label: `scelto: ${chunkLabel(c.id)}` })),
    ],
  };
}

/**
 * Quanto ha lavorato davvero l'indice HNSW del tab (come la didascalia dell'azione Ricerca: gli stessi conti
 * di hnswCost.js). Con pochi chunk non conviene, per due motivi diversi che il testo distingue: con il grafo
 * didattico a più livelli rilegge alcuni nodi a ogni livello (più distanze dei nodi visitati); con i valori
 * di pgvector il grafo ha un livello solo ed ef_search (40) supera i chunk, quindi legge tutto il grafo, un
 * nodo una volta.
 * @param {{ comparisons: number, recall: number, frames?: { visited: number[] }[] }} run
 * @param {number} N chunk nel database
 * @param {number} [efSearch] ef_search dell'indice
 */
export function hnswNote(run, N, efSearch) {
  const { comparisons: n, rereads } = hnswCost(run);
  if (n < N) return `L'indice HNSW del tab ha calcolato ${it(n)} distanze invece di ${it(N)} (recall ${it(run.recall * 100)}%): salta tra pochi nodi del grafo invece di leggerli tutti.`;
  const why = rereads ? "rilegge alcuni nodi a ogni livello del grafo" : `${efSearch ? `con ef_search ${it(efSearch)} e ` : "con "}solo ${it(N)} chunk legge tutto il grafo`;
  return `L'indice HNSW del tab ha calcolato ${it(n)} distanze, la ricerca esatta ${it(N)}: con così pochi chunk l'indice non conviene (${why}). Il vantaggio arriva con migliaia di vettori, come misurano i Grafici del tab.`;
}

// ------------------------------------------------------------------------------------- parte 3
function encoderPart({ tokens, embedder }) {
  const E = SHOWN.encoder;
  const T = tokens.names.length;
  const depth = { count: T, label: "token", names: tokens.names };
  const cut = tokens.total > T ? ` · qui ${T} dei ${tokens.total}` : "";
  return {
    sections: [
      {
        id: "enc.tokens",
        label: "Token della domanda",
        sub: `«query: » + domanda · ${T} token${cut} · one-hot su ${SHOWN.vocab} delle ${it(ENCODER.vocab)} voci`,
        role: "input",
        neurons: SHOWN.vocab,
        depth,
        activation: "onehot",
        real: ENCODER.vocab,
        text: [
          "Un modello neurale legge token, non feature. Come esempio aperto usiamo multilingual-e5-small (Microsoft): l'architettura di text-embedding-3-small, il modello di OpenAI delle lezioni (1.536 numeri per testo), non è pubblica.",
          "E5 vuole un prefisso: «query: » davanti alla domanda, «passage: » davanti ai chunk, così sa se sta leggendo una domanda o un documento. <s> e </s> segnano l'inizio e la fine.",
          "I token qui sono quelli stimati dal tab: il tokenizer vero di E5, quello di XLM-R, spezzerebbe le parole in modo un po' diverso. Valori simulati.",
        ].join("\n\n"),
      },
      {
        id: "enc.embed",
        label: "Embedding",
        sub: `${it(ENCODER.width)} numeri per token · qui ${E.width}`,
        role: "embedding",
        neurons: E.width,
        depth,
        real: ENCODER.width,
        text: `Ogni token prende la sua riga di una tabella di ${it(ENCODER.vocab)} × ${ENCODER.width} pesi imparati (96 milioni dei ${it(ENCODER.params / 1e6)} milioni di parametri del modello), più un vettore per la posizione: come un dizionario in cui ogni parola ha la sua carta d'identità di ${ENCODER.width} numeri.`,
      },
      {
        id: "enc.attn",
        label: "Attention bidirezionale",
        sub: `${ENCODER.heads} head da ${ENCODER.width / ENCODER.heads} · qui ${E.heads} · nessuna maschera`,
        role: "attention",
        neurons: E.width,
        heads: E.heads,
        slabs: 2,
        slabNames: ["key e value", "uscita"],
        depth,
        real: ENCODER.width,
        text: [
          "Ogni token guarda TUTTI gli altri, prima e dopo di lui: nessuna maschera causale, è l'attention bidirezionale di BERT. Così ogni parola si legge nel contesto dell'intera frase: in «il limite di spesa per una cena in trasferta» la parola «limite» capisce subito di quale limite si parla, anche se «spesa» e «trasferta» vengono dopo. Come leggere tutta la frase prima di capirne una parola.",
          "È la differenza con l'LLM della parte 4, che guarda solo indietro perché scrive un token alla volta. Un colore per head, più luce = più peso (simulato).",
        ].join("\n\n"),
      },
      {
        id: "enc.add1",
        label: "+ residuo e LayerNorm",
        role: "add",
        neurons: E.width,
        depth,
        text: "L'uscita dell'attention si somma al vettore che era entrato nel layer (connessione residua, l'arco che salta l'attention): il layer aggiunge informazione, non la sostituisce. Subito dopo una LayerNorm riporta i numeri di ogni token su una scala comune: in BERT e in E5 la normalizzazione viene dopo la somma (post-LN), l'LLM della parte 4 la mette prima.",
      },
      {
        id: "enc.ffn",
        label: "Feed-forward (GELU)",
        sub: `${it(ENCODER.ffn)} neuroni nascosti · qui ${E.ffn}`,
        role: "mlp",
        neurons: E.ffn,
        depth,
        activation: "positive",
        real: ENCODER.ffn,
        text: `Due strati densi applicati a ogni token per conto suo, con ${it(ENCODER.ffn)} neuroni nascosti: 4 volte la larghezza, la regola di BERT (con questo valore tornano i 21 milioni di parametri del transformer di Multilingual-MiniLM, da cui E5 small parte). Qui ogni token rielabora ciò che ha raccolto dagli altri.`,
      },
      { id: "enc.add2", label: "+ residuo e LayerNorm", role: "add", neurons: E.width, depth, text: "Di nuovo la connessione residua, x + feed-forward(x), e la seconda LayerNorm del layer: il vettore di ogni token passa al layer dopo." },
      {
        repeat: ENCODER.layers,
        of: ["enc.attn", "enc.add1", "enc.ffn", "enc.add2"],
        id: "enc.x12",
        label: `× ${ENCODER.layers} layer`,
        text: `multilingual-e5-small ha ${ENCODER.layers} layer uguali: ne vedi uno, gli altri ${ENCODER.layers - 1} sono le lastre fantasma. Hanno la stessa forma e pesi diversi.`,
      },
      {
        id: "enc.pool",
        label: "Mean pooling",
        sub: `media sui token · ${ENCODER.width} numeri · qui ${E.width}`,
        role: "latent",
        neurons: E.width,
        real: ENCODER.width,
        text: "La profondità si schiaccia: il vettore della frase è la media dei vettori di tutti i token (nel codice di E5 il pooling si chiama proprio «avg»). Come riassumere un coro con una nota sola: la media delle voci.",
      },
      {
        id: "enc.vec",
        label: "Vettore della frase",
        sub: `normalizzato, lunghezza 1 · ${ENCODER.width} numeri, qui ${E.width}`,
        role: "latent",
        neurons: E.width,
        real: ENCODER.width,
        text: `Il vettore viene portato a lunghezza 1 (F.normalize nel codice di E5): è questo che un RAG vero salverebbe e cercherebbe al posto dei nostri ${embedder.dims} numeri, confrontandolo con i chunk codificati con «passage: ». A differenza del nostro, quasi nessuno dei suoi ${ENCODER.width} numeri è zero, e due testi con parole diverse ma lo stesso significato finiscono vicini.`,
      },
    ],
    links: [
      { from: "enc.tokens", to: "enc.embed", pattern: "lookup" },
      { from: "enc.embed", to: "enc.attn", pattern: "dense", toSlab: 0, density: 0.2 },
      { within: "enc.attn", pattern: "attention" }, // bidirezionale: nessuna maschera causale
      { from: "enc.attn", to: "enc.add1", pattern: "dense", density: 0.12 },
      { from: "enc.embed", to: "enc.add1", pattern: "residual" },
      { from: "enc.add1", to: "enc.ffn", pattern: "dense", density: 0.1 },
      { from: "enc.ffn", to: "enc.add2", pattern: "dense", density: 0.1 },
      { from: "enc.add1", to: "enc.add2", pattern: "residual" },
      { from: "enc.add2", to: "enc.pool", pattern: "pool", label: "mean pooling" },
      { from: "enc.pool", to: "enc.vec", pattern: "identity", label: "diviso per la lunghezza" },
    ],
  };
}

// ------------------------------------------------------------------------------------- parte 4
function generatorPart({ prompt }) {
  const G = SHOWN.generator;
  const P = prompt.names.length;
  const depth = { count: P, label: "token del prompt", names: prompt.names };
  return {
    sections: [
      {
        id: "llm.prompt",
        label: "Prompt aumentato",
        sub: `${P} dei ${it(prompt.total)} token · one-hot su ${SHOWN.vocab} delle ${it(GENERATOR.vocab)} voci`,
        role: "input",
        neurons: SHOWN.vocab,
        depth,
        activation: "onehot",
        real: GENERATOR.vocab,
        text: [
          `Il prompt che il tab costruisce davvero (stazione 9): strumenti e system prompt (${it(prompt.fixed)} token), poi la domanda e i chunk scelti, circa ${it(prompt.total)} token stimati. In profondità ne disegniamo ${P}: la domanda e l'inizio del testo di ogni chunk scelto.`,
          "Come esempio di LLM aperto usiamo Llama 3 8B (Meta): quello delle lezioni, chiamato via API, non ha un'architettura pubblica. Valori simulati.",
        ].join("\n\n"),
      },
      {
        id: "llm.embed",
        label: "Embedding",
        sub: `${it(GENERATOR.width)} numeri per token · qui ${G.width}`,
        role: "embedding",
        neurons: G.width,
        depth,
        real: GENERATOR.width,
        text: `La riga della tabella di ogni token: ${it(GENERATOR.vocab)} righe × ${it(GENERATOR.width)} numeri. La posizione non si somma qui: entra nell'attention, ruotando query e key (RoPE).`,
      },
      {
        id: "llm.attn",
        label: "Attention causale (GQA)",
        sub: `RMSNorm prima · ${GENERATOR.heads} head di query, ${GENERATOR.kvHeads} di key/value · qui ${G.heads}`,
        role: "attention",
        neurons: G.width,
        heads: G.heads,
        slabs: 2,
        slabNames: ["key e value", "uscita"],
        depth,
        real: GENERATOR.width,
        text: [
          "Ogni token guarda solo se stesso e quelli prima (maschera causale): gli archi vanno sempre verso il passato. Così l'LLM legge la domanda e i chunk e poi scrive un token alla volta, senza sbirciare il futuro: come scrivere un tema riga per riga, rileggendo solo quello che hai già scritto.",
          `Prima dell'attention una RMSNorm normalizza ogni token (pre-norm, al contrario di BERT). GQA: ${GENERATOR.heads} head di query condividono ${GENERATOR.kvHeads} coppie key/value, così la KV cache è ${GENERATOR.heads / GENERATOR.kvHeads} volte più piccola (vedi il tab LLM moderni).`,
        ].join("\n\n"),
      },
      { id: "llm.add1", label: "+ residuo", role: "add", neurons: G.width, depth, text: "x + attention(x): la connessione residua, l'arco che salta l'attention. Il vettore che esce va al feed-forward." },
      {
        id: "llm.ffn",
        label: "Feed-forward SwiGLU",
        sub: `RMSNorm prima · ${it(GENERATOR.ffn)} neuroni nascosti · qui ${G.ffn}`,
        role: "mlp",
        neurons: G.ffn,
        depth,
        activation: "positive",
        real: GENERATOR.ffn,
        text: `Ogni token per conto suo, dopo un'altra RMSNorm, con ${it(GENERATOR.ffn)} neuroni nascosti e un «variatore» (SwiGLU) al posto dell'interruttore della ReLU: qui sta la maggior parte dei pesi.`,
      },
      { id: "llm.add2", label: "+ residuo", role: "add", neurons: G.width, depth, text: "Di nuovo la connessione residua, x + feed-forward(x): il layer finisce qui e il vettore di ogni token passa al layer dopo." },
      {
        repeat: GENERATOR.layers,
        of: ["llm.attn", "llm.add1", "llm.ffn", "llm.add2"],
        id: "llm.x32",
        label: `× ${GENERATOR.layers} layer`,
        text: `Llama 3 8B ha ${GENERATOR.layers} layer uguali: ne vedi uno vero e poche lastre fantasma.`,
      },
      {
        id: "llm.next",
        label: "Prossimo token",
        sub: `RMSNorm, lm_head e softmax su ${it(GENERATOR.vocab)} voci · qui ${SHOWN.vocab}`,
        role: "output",
        neurons: SHOWN.vocab,
        real: GENERATOR.vocab,
        text: "Un'ultima RMSNorm, poi la matrice di uscita (lm_head) dà un punteggio a ogni voce del vocabolario e la softmax lo trasforma in probabilità. Solo l'ultima posizione predice il token che viene dopo, il primo pezzo della risposta; poi il token si aggiunge al prompt e si ricomincia, uno alla volta (nel tab arrivano in streaming SSE). Simulato: nel tab la risposta è composta con le frasi dei chunk recuperati (simulazione dichiarata), qui vedi come la scriverebbe un LLM.",
      },
    ],
    links: [
      { from: "llm.prompt", to: "llm.embed", pattern: "lookup" },
      { from: "llm.embed", to: "llm.attn", pattern: "dense", toSlab: 0, density: 0.2 },
      { within: "llm.attn", pattern: "attention", causal: true }, // solo verso il passato
      { from: "llm.attn", to: "llm.add1", pattern: "dense", density: 0.12 },
      { from: "llm.embed", to: "llm.add1", pattern: "residual" },
      { from: "llm.add1", to: "llm.ffn", pattern: "dense", density: 0.1 },
      { from: "llm.ffn", to: "llm.add2", pattern: "dense", density: 0.1 },
      { from: "llm.add1", to: "llm.add2", pattern: "residual" },
      { from: "llm.add2", to: "llm.next", pattern: "dense", slice: P - 1, density: 0.4, label: "lm_head, dall'ultima posizione" },
    ],
  };
}

/** Fonti della rete (solo link http/https: la scheda del nucleo li rende cliccabili). */
const SOURCES = Object.freeze([
  { title: "OpenAI: guida agli embedding", url: "https://platform.openai.com/docs/guides/embeddings", note: "text-embedding-3-small, il modello delle lezioni: 1.536 dimensioni, vettori di lunghezza 1; l'architettura non è pubblica. Il nostro embedder (js/shared/rag/embedder.js) lo simula senza rete neurale." },
  { title: "pgvector: README", url: "https://github.com/pgvector/pgvector", note: "Operatore <=> (distanza del coseno), indice HNSW, filtri dopo la scansione dell'indice." },
  { title: "Spring AI 2.0.2: PgVectorStore.java", url: "https://github.com/spring-projects/spring-ai/blob/v2.0.2/vector-stores/spring-ai-pgvector-store/src/main/java/org/springframework/ai/vectorstore/pgvector/PgVectorStore.java", note: "distance < 1 − similarityThreshold, ORDER BY distance LIMIT topK, score = 1 − distance." },
  { title: "microsoft/unilm: E5 Text Embeddings (README)", url: "https://github.com/microsoft/unilm/tree/master/e5", note: "multilingual-e5-small: 12 layer, vettori da 384." },
  { title: "microsoft/unilm: codice di valutazione di E5", url: "https://github.com/microsoft/unilm/blob/master/e5/mteb_beir_eval.py", note: "Prefissi «query: » e «passage: », pooling «avg» (utils.py, model_config.py) e F.normalize." },
  { title: "microsoft/unilm: Multilingual MiniLM", url: "https://github.com/microsoft/unilm/tree/master/minilm", note: "Multilingual-MiniLM-L12-H384: 12 layer, 384, 12 head, 21M parametri del transformer + 96M di embedding; tokenizer di XLM-R, architettura di BERT." },
  { title: "MTEB: metadati di intfloat/multilingual-e5-small", url: "https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/e5_models.py", note: "118M parametri, 96.014.208 nell'embedding (/ 384 = 250.037 voci), 512 token, coseno, adattato da Multilingual-MiniLM-L12-H384." },
  { title: "Wang et al. (2024), Multilingual E5 Text Embeddings: A Technical Report", url: "https://arxiv.org/abs/2402.05672", note: "Il rapporto tecnico dei modelli E5 multilingui." },
  { title: "google-research/bert (README)", url: "https://github.com/google-research/bert", note: "Encoder «deeply bidirectional»: ogni token guarda quelli prima e dopo." },
  { title: "meta-llama/llama-models: sku_list.py", url: "https://github.com/meta-llama/llama-models", note: "Llama 3 8B: dim 4096, 32 layer, 32 head, 8 head K/V, vocabolario 128.256, ffn_dim_multiplier 1,3, multiple_of 1024 → 14.336, RoPE θ 500.000." },
  { title: "meta-llama/llama3: MODEL_CARD.md", url: "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", note: "Decoder auto-regressivo con GQA, vocabolario di 128K token, sequenze di 8.192 token." },
  { title: "Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", url: "https://arxiv.org/abs/2005.11401", note: "L'idea del RAG: un retriever che trova i documenti e un generatore che li legge." },
]);
