/**
 * Meccanismi della "Rete neurale" del tab LLM moderni: logica PURA (niente three.js, niente DOM; test in
 * tests/modern-llm-reps.test.mjs). Da qui network.js prende due cose:
 *
 *   drawnLayers(model)   quali layer disegnare e in che ordine: i blocchi della striscia del livello
 *                        Blocchi (buildStrip di full/model.js, la stessa compressione «× N»), ognuno con
 *                        la ricetta della sua attention letta dal config (layerRecipe)
 *   layerAttention(o)    i pesi di attention ILLUSTRATIVI di un layer: finestra locale e voci globali in
 *                        un'unica softmax con il sink, causali per costruzione (selectEntries sceglie le
 *                        voci come farebbe l'indexer, solo tra quelle già complete)
 *
 * Perché la ricetta viene dal config e non dalla lettera del tipo di layer (S, F, R…): i campi del
 * config (compressRatios, kvSourceLayers, indexSourceLayers, la ricerca §4.2-4.4) sono la fonte; il tipo
 * è solo un nome. Così un errore nello schema dei tipi si vedrebbe nei test invece di diventare un
 * disegno sbagliato.
 *
 * I pesi dei modelli veri non sono nel browser: i valori qui sono generati con un seme, deterministici
 * (stessa specifica, stessi archi), e la rete lo dichiara in ogni sezione.
 */
import { blockOfLayer, buildStrip, layerByNumber } from "./full/model.js";

/**
 * Ricetta dell'attention di un layer, dai campi del config (DeepSeek-V4.1-Flash, ricerca §4.3-4.4):
 *   compress   compress_ratios: 0 = solo finestra, 2 = 2 token → 1 voce globale (encoder), 1 = una voce
 *              per token (decoder)
 *   writesKV   il layer calcola la KV globale (kv_source_layer_ids); gli altri layer compressi la riusano
 *   indexes    il layer calcola la scelta dell'indexer (index_source_layer_ids); gli altri la riusano
 *   kvSource / indexSource   il layer di cui riusa KV globale e scelta (l'ultimo sorgente prima di lui)
 *   fromSegment   nel decoder (m = 1) la voce globale è una proiezione dell'uscita del segmento prima
 *              (l'encoder: H20, eq. 1 del report), non dello stato del layer
 * @returns {null | {number, kind, compress, writesKV, indexes, kvSource, indexSource, fromSegment}}
 *   null se il modello non descrive l'attention con questi campi (Kimi K3.1 Pro, gpt-oss-120b)
 */
export function layerRecipe(model, number) {
  const c = model.config;
  const layer = layerByNumber(model, number);
  if (!layer || !Array.isArray(c.compressRatios) || !Array.isArray(c.kvSourceLayers) || !Array.isArray(c.indexSourceLayers)) return null;
  const compress = c.compressRatios[number - model.layers.base] ?? 0;
  const global = compress > 0;
  const writesKV = global && c.kvSourceLayers.includes(number);
  const indexes = global && c.indexSourceLayers.includes(number);
  const segments = model.layers.segments ?? [];
  const at = segments.findIndex((s) => number >= s.from && number <= s.to);
  return {
    number,
    kind: layer.kind,
    compress,
    writesKV,
    indexes,
    kvSource: global ? lastAtMost(c.kvSourceLayers, number) : null,
    indexSource: global ? lastAtMost(c.indexSourceLayers, number) : null,
    fromSegment: writesKV && compress === 1 && at > 0 ? segments[at - 1] : null,
  };
}

/** L'ultimo valore ≤ n di un elenco crescente (il layer sorgente più vicino prima di n), o null. */
function lastAtMost(list, n) {
  let best = null;
  for (const v of list) if (v <= n && (best === null || v > best)) best = v;
  return best;
}

/**
 * I layer da disegnare: un layer rappresentativo per ogni blocco della striscia dei Blocchi (gruppi
 * compressi in «× N», come nella vista completa), nell'ordine vero, con le ricette e i riferimenti ai
 * blocchi da cui riusano KV globale e scelta dell'indexer.
 * @returns {{ blocks: object[], groups: object[], strip: object }}
 *   blocks: { index (nella striscia), block, rep, recipe, kvBlock, indexBlock, group } (solo i layer)
 *   groups: i gruppi «× N» della striscia (from/to = indici della striscia)
 */
export function drawnLayers(model) {
  const strip = buildStrip(model);
  const groups = strip.groups;
  const blocks = [];
  strip.blocks.forEach((block, index) => {
    if (block.type !== "layer") return;
    const recipe = layerRecipe(model, block.rep);
    if (!recipe) throw new Error(`${model.name}: la rete neurale sa disegnare solo modelli con compress_ratios, kv_source e index_source nel config`);
    // tutti i layer del blocco devono avere la stessa ricetta: il blocco li rappresenta tutti
    for (const n of block.layers) {
      const other = layerRecipe(model, n);
      if (other.compress !== recipe.compress || other.writesKV !== recipe.writesKV || other.indexes !== recipe.indexes) throw new Error(`${model.name}: il layer ${n} non ha la stessa attention del layer ${block.rep}`);
    }
    // chi riusa KV globale e scelta delle voci deve avere la stessa compressione del layer sorgente: le
    // voci (m token → 1) e gli indici scelti dall'indexer sono quelli del sorgente
    for (const source of [recipe.kvSource, recipe.indexSource]) {
      if (source !== null && layerRecipe(model, source).compress !== recipe.compress) throw new Error(`${model.name}: il layer ${block.rep} riusa il layer ${source}, che ha un'altra compressione`);
    }
    blocks.push({
      index,
      block,
      rep: block.rep,
      recipe,
      kvBlock: recipe.kvSource === null ? -1 : blockOfLayer(strip, recipe.kvSource),
      indexBlock: recipe.indexSource === null ? -1 : blockOfLayer(strip, recipe.indexSource),
      group: groups.find((g) => index >= g.from && index <= g.to) ?? null,
    });
  });
  return { blocks, groups, strip };
}

// ------------------------------------------------------------------------------------------------
// Pesi di attention illustrativi
// ------------------------------------------------------------------------------------------------

/** Generatore pseudo-casuale con seme (mulberry32): stessi semi, stessi numeri, in ogni browser. */
export function seededRandom(...parts) {
  let h = 2166136261;
  for (const ch of parts.join("|")) h = Math.imul(h ^ ch.codePointAt(0), 16777619);
  let a = h >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Voce globale e (m token: e·m … e·m + m − 1) già completa per la query i: niente sbirciate nel futuro. */
export const entryReady = (e, i, m) => e * m + m - 1 <= i;

/** Numero di voci globali di un prompt di `tokens` token con compressione m (m token → 1 voce). */
export const entryCount = (tokens, m) => Math.floor(tokens / m);

/**
 * Le voci globali scelte per ogni query, come farebbe il lightning indexer: le `k` con il punteggio più
 * alto tra quelle già complete (causale). Punteggi illustrativi con seme.
 * @returns {number[][]} per ogni posizione i, gli indici delle voci scelte (al più k)
 */
export function selectEntries({ tokens, m, k, seed }) {
  const E = entryCount(tokens, m);
  const rand = seededRandom("indexer", seed);
  const score = Array.from({ length: E }, () => rand());
  const out = [];
  for (let i = 0; i < tokens; i++) {
    const ready = [];
    for (let e = 0; e < E; e++) if (entryReady(e, i, m)) ready.push(e);
    // un po' di variazione per query (l'indexer guarda anche la query), poi le k migliori
    const jitter = seededRandom("indexer", seed, i);
    const scored = ready.map((e) => ({ e, s: score[e] + 0.35 * jitter() }));
    scored.sort((a, b) => b.s - a.s || a.e - b.e);
    out.push(scored.slice(0, k).map((x) => x.e).sort((a, b) => a - b));
  }
  return out;
}

/**
 * Pesi di attention di un layer (ILLUSTRATIVI): per ogni head e ogni query, un'unica softmax sulle chiavi
 * della finestra locale, sulle voci globali scelte e sul sink (il logit appreso che permette a una head di
 * "non guardare nulla": la sua parte non si disegna). Le head hanno abitudini diverse, come nei transformer
 * veri: il token precedente, i vicini, l'inizio della finestra, un punto scelto dal contenuto.
 * @param {{ tokens: number, heads: number, window: number, m?: number, selection?: number[][] | null, seed: string }} o
 * @returns {{ window: Float32Array, entries: Float32Array | null, sink: Float32Array, entryCount: number }}
 *   window[(h·T + i)·T + j]: query i ← chiave j (j ≤ i, i − j < finestra);
 *   entries[(h·T + i)·E + e]: query i ← voce e (solo le scelte, già complete); sink[h·T + i]: il resto
 */
export function layerAttention({ tokens: T, heads: H, window: W, m = 1, selection = null, seed }) {
  const E = selection ? entryCount(T, m) : 0;
  const win = new Float32Array(H * T * T);
  const entries = selection ? new Float32Array(H * T * E) : null;
  const sink = new Float32Array(H * T);
  const rand = seededRandom("attention", seed);
  for (let h = 0; h < H; h++) {
    const style = h % 4;
    const sinkLogit = 0.4 + 1.2 * rand(); // quanto questa head tende a "non guardare nulla"
    for (let i = 0; i < T; i++) {
      const first = Math.max(0, i - W + 1);
      const keys = [];
      for (let j = first; j <= i; j++) {
        let z = 0.35 * (rand() - 0.5);
        if (style === 0) z += j === Math.max(0, i - 1) ? 3 : 0;
        else if (style === 1) z += -0.8 * (i - j);
        else if (style === 2) z += j === first ? 2.4 : 0;
        else z += rand() < 0.3 ? 2.2 : 0;
        keys.push({ kind: "window", j, z });
      }
      for (const e of selection?.[i] ?? []) keys.push({ kind: "entry", j: e, z: 1.2 + 1.4 * rand() + (style === 3 ? 0.8 : 0) });
      let max = sinkLogit;
      for (const k of keys) max = Math.max(max, k.z);
      let sum = Math.exp(sinkLogit - max);
      for (const k of keys) sum += Math.exp(k.z - max);
      sink[h * T + i] = Math.exp(sinkLogit - max) / sum;
      for (const k of keys) {
        const w = Math.exp(k.z - max) / sum;
        if (k.kind === "window") win[(h * T + i) * T + k.j] = w;
        else entries[(h * T + i) * E + k.j] = w;
      }
    }
  }
  return { window: win, entries, sink, entryCount: E };
}
