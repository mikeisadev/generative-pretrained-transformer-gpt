/**
 * Collegamenti della rete orizzontale, generati per schema con un seme (logica pura, testata con node).
 *
 * Due famiglie, due oggetti grafici (uno ciascuno, regola delle prestazioni):
 *   segmenti  linee dritte tra unità: dense, identity, lookup, route, conv, pool, broadcast, down/up,
 *             explicit. Ordinati per priorità: la densità scelta dallo studente è solo il numero di
 *             segmenti da disegnare (un prefisso), senza ricostruire niente. Priorità = 1 − |peso| per
 *             i collegamenti con un peso (prima i più forti), un numero casuale per quelli strutturali.
 *   archi     curve: attention (tra posizioni della profondità, una head per colore) e residui
 *             (sopra le sezioni). La forma la decide la scena; qui solo chi collega chi e con che peso.
 *
 * Tetto di sicurezza: se gli schemi chiederebbero più di `maxSegments` segmenti, le densità vengono
 * scalate tutte dello stesso fattore (`scale`, dichiarato nella scheda), poi si tagliano i meno
 * importanti. Ogni collegamento ha il suo generatore (seme della rete + indice): stessi input,
 * stessi segmenti, sempre.
 */
import { attentionWeights, headAnchorNeuron } from "./activations.js";
import { unitIndex } from "./layout.js";
import { gaussian, geometricSkip, rngFor } from "./random.js";
import { sliceUnits } from "./spec.js";

export const LINK_LIMITS = Object.freeze({
  maxSegments: 160_000, // linee dritte disegnabili (prova di carico: ~150.000 con il rendering software)
  maxArcs: 9_000, // archi (ognuno diventa una curva di qualche segmento)
  maxSliceSegments: 60_000, // pesi veri della fetta scelta
  arcMinWeight: 0.04, // sotto il 4% un arco di attention sarebbe invisibile
  arcKeysPerQuery: 4, // al massimo 4 chiavi per query e per head (le più forti): il disegno resta leggibile
});

export const ARC_KIND = Object.freeze({ attention: 0, cross: 1, residual: 2 });
const PRIORITY_BUCKETS = 2048;

/** Elenco di segmenti che cresce raddoppiando (array tipizzati: niente milioni di oggetti). */
class SegmentList {
  constructor(capacity = 4096) {
    this.count = 0;
    this.alloc(capacity);
  }

  alloc(capacity) {
    const grow = (Ctor, old) => {
      const a = new Ctor(capacity);
      if (old) a.set(old.subarray(0, this.count));
      return a;
    };
    this.from = grow(Uint32Array, this.from);
    this.to = grow(Uint32Array, this.to);
    this.weight = grow(Float32Array, this.weight);
    this.link = grow(Uint16Array, this.link);
    this.priority = grow(Float32Array, this.priority);
    this.extra = grow(Uint8Array, this.extra); // head (archi) o niente
    this.kind = grow(Uint8Array, this.kind);
    this.capacity = capacity;
  }

  push(from, to, weight, link, priority, extra = 0, kind = 0) {
    if (this.count === this.capacity) this.alloc(this.capacity * 2);
    const i = this.count++;
    this.from[i] = from;
    this.to[i] = to;
    this.weight[i] = weight;
    this.link[i] = link;
    this.priority[i] = priority;
    this.extra[i] = extra;
    this.kind[i] = kind;
  }

  /** Ordina per priorità crescente (counting sort su 2048 secchi: O(n)) e taglia a `max`. */
  finish(max = Infinity) {
    const n = this.count;
    const counts = new Uint32Array(PRIORITY_BUCKETS + 1);
    const bucket = new Uint16Array(n);
    for (let i = 0; i < n; i++) {
      const b = Math.min(PRIORITY_BUCKETS - 1, Math.max(0, Math.floor(this.priority[i] * PRIORITY_BUCKETS)));
      bucket[i] = b;
      counts[b + 1]++;
    }
    for (let b = 0; b < PRIORITY_BUCKETS; b++) counts[b + 1] += counts[b];
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[counts[bucket[i]]++] = i;
    const kept = Math.min(n, max);
    const pick = (src, Ctor) => {
      const out = new Ctor(kept);
      for (let i = 0; i < kept; i++) out[i] = src[order[i]];
      return out;
    };
    return {
      count: kept,
      dropped: n - kept,
      from: pick(this.from, Uint32Array),
      to: pick(this.to, Uint32Array),
      weight: pick(this.weight, Float32Array),
      link: pick(this.link, Uint16Array),
      head: pick(this.extra, Uint8Array),
      kind: pick(this.kind, Uint8Array),
      priority: pick(this.priority, Float32Array),
    };
  }
}

/** Lastre coinvolte: "all" o un indice. */
const slabsOf = (s, sel) => (sel === "all" ? Array.from({ length: s.slabs }, (_, i) => i) : [sel]);

/** Indici globali delle unità della fetta d (nelle lastre scelte), in ordine di lastra e neurone. */
function sliceList(s, slabs, d) {
  const out = new Uint32Array(slabs.length * s.neurons);
  let k = 0;
  for (const slab of slabs) for (let n = 0; n < s.neurons; n++) out[k++] = s.offset + unitIndex(s, slab, n, d);
  return out;
}

/** Coppie di posizioni (dA, dB): uguali una a una, una sola contro tutte, oppure in proporzione. */
export function pairDepths(DA, DB) {
  if (DA === DB) return Array.from({ length: DA }, (_, d) => [d, d]);
  if (DA === 1) return Array.from({ length: DB }, (_, d) => [0, d]);
  if (DB === 1) return Array.from({ length: DA }, (_, d) => [d, 0]);
  return Array.from({ length: DA }, (_, d) => [d, Math.min(DB - 1, Math.floor((d * DB) / DA))]);
}

/** Le coppie da usare secondo l'opzione `slice` ("all", "selected" = gestite a parte, o un indice). */
function pairsForSlice(link) {
  const pairs = pairDepths(link.from.depth.count, link.to.depth.count);
  if (link.slice === undefined || link.slice === "all" || link.slice === "selected") return pairs;
  return pairs.filter(([a, b]) => (link.to.depth.count > 1 ? b : a) === link.slice);
}

/** Peso simulato di un collegamento appreso: normale, stretto attorno a zero, tagliato a ±1. */
const simulatedWeight = (rand) => Math.max(-1, Math.min(1, 0.42 * gaussian(rand)));
const weightPriority = (w, rand) => Math.min(1, Math.max(0, 1 - Math.abs(w) + 0.02 * rand()));

/** Numero atteso di segmenti di un collegamento (prima del campionamento): serve al tetto. */
export function expectedSegments(link, routing) {
  const A = link.from;
  const B = link.to;
  const ua = (sel) => slabsOf(A, sel).length * A.neurons;
  const ub = (sel) => slabsOf(B, sel).length * B.neurons;
  const same = sliceUnits(A) === sliceUnits(B);
  switch (link.pattern) {
    case "dense":
      return pairsForSlice(link).length * ua(link.fromSlab) * ub(link.toSlab) * link.density;
    case "identity":
      return pairsForSlice(link).length * Math.min(ua(link.fromSlab), ub(link.toSlab)) * link.density;
    case "lookup":
      return B.depth.count * ub(link.toSlab) * link.density;
    case "route": {
      const E = link.experts;
      const k = E.active + E.shared;
      const other = link.direction === "dispatch" ? ua(link.fromSlab) : ub(link.toSlab);
      return E.depth.count * k * E.neurons * other * link.density;
    }
    case "conv": {
      const window = A.depth.kind === "grid" ? link.kernel ** 2 : link.kernel;
      return B.depth.count * ub(link.toSlab) * ua(link.fromSlab) * window * link.density;
    }
    case "pool":
      return A.depth.count * ua(link.fromSlab) * (same ? 1 : ub(link.toSlab) * link.density);
    case "broadcast":
      return B.depth.count * ub(link.toSlab) * (same ? 1 : ua(link.fromSlab) * link.density);
    case "down": {
      const window = A.depth.kind === "grid" ? link.factor ** 2 : link.factor;
      return B.depth.count * ub(link.toSlab) * window * (same ? 1 : ua(link.fromSlab) * link.density);
    }
    case "up":
      return B.depth.count * ub(link.toSlab) * (same ? 1 : ua(link.fromSlab) * link.density);
    case "explicit":
      return link.slice === "selected" && Math.max(A.depth.count, B.depth.count) > 1 ? 0 : link.pairs.count * pairsForSlice(link).length;
    default:
      return 0; // attention e residui sono archi
  }
}

/**
 * @param {import("./spec.js").NormalizedNetwork} net
 * @param {{ raw: Float32Array, routing: Map }} values da resolveValues
 * @param {Partial<typeof LINK_LIMITS>} [limits]
 */
export function buildLinks(net, values, limits = {}) {
  const lim = { ...LINK_LIMITS, ...limits };
  const expected = net.links.map((l) => expectedSegments(l, values.routing));
  const total = expected.reduce((a, b) => a + b, 0);
  const scale = total > lim.maxSegments ? lim.maxSegments / total : 1;
  const segments = new SegmentList();
  const sliceSegments = new SegmentList(256);
  const arcs = new SegmentList(1024);
  const stats = [];
  for (const link of net.links) {
    const rand = rngFor(net.seed, "link", link.index, link.pattern);
    const before = segments.count + sliceSegments.count + arcs.count;
    const density = link.pattern === "explicit" ? 1 : Math.min(1, link.density * scale);
    const generator = GENERATORS[link.pattern];
    generator({ net, link, rand, density, values, segments, sliceSegments, arcs, lim });
    stats.push({ link: link.index, pattern: link.pattern, generated: segments.count + sliceSegments.count + arcs.count - before, expected: expected[link.index], density });
  }
  const seg = segments.finish(lim.maxSegments);
  const arc = arcs.finish(lim.maxArcs);
  const slice = sliceSegments.finish(lim.maxSliceSegments);
  return { segments: seg, arcs: arc, sliceSegments: slice, stats, scale, capped: scale < 1 || seg.dropped > 0 || arc.dropped > 0 };
}

/** Campione di Bernoulli con probabilità p su `size` coppie, in O(campionate): chiama fn(indice). */
function sample(size, p, rand, fn) {
  if (p >= 1) {
    for (let i = 0; i < size; i++) fn(i);
    return;
  }
  for (let i = geometricSkip(rand, p); i < size; i += 1 + geometricSkip(rand, p)) fn(i);
}

const GENERATORS = {
  dense({ link, rand, density, segments }) {
    const fromSlabs = slabsOf(link.from, link.fromSlab);
    const toSlabs = slabsOf(link.to, link.toSlab);
    for (const [dA, dB] of pairsForSlice(link)) {
      const A = sliceList(link.from, fromSlabs, dA);
      const B = sliceList(link.to, toSlabs, dB);
      sample(A.length * B.length, density, rand, (i) => {
        const w = simulatedWeight(rand);
        segments.push(A[Math.floor(i / B.length)], B[i % B.length], w, link.index, weightPriority(w, rand));
      });
    }
  },

  identity({ link, rand, density, segments }) {
    const fromSlabs = slabsOf(link.from, link.fromSlab);
    const toSlabs = slabsOf(link.to, link.toSlab);
    for (const [dA, dB] of pairsForSlice(link)) {
      const A = sliceList(link.from, fromSlabs, dA);
      const B = sliceList(link.to, toSlabs, dB);
      sample(Math.min(A.length, B.length), density, rand, (i) => segments.push(A[i], B[i], 1, link.index, rand()));
    }
  },

  /** Il token sceglie la sua riga: dall'unità accesa (la più alta) di ogni posizione a tutto il vettore. */
  lookup({ link, rand, density, values, segments }) {
    const fromSlabs = slabsOf(link.from, link.fromSlab);
    const toSlabs = slabsOf(link.to, link.toSlab);
    for (let d = 0; d < link.to.depth.count; d++) {
      const A = sliceList(link.from, fromSlabs, d);
      let hot = A[0];
      for (const u of A) if (values.raw[u] > values.raw[hot]) hot = u;
      const B = sliceList(link.to, toSlabs, d);
      sample(B.length, density, rand, (i) => segments.push(hot, B[i], 1, link.index, rand()));
    }
  },

  /** Ogni token verso i suoi esperti (dispatch) o dagli esperti verso il token (combine). */
  route({ link, rand, density, values, segments }) {
    const E = link.experts;
    const route = values.routing.get(E.id);
    const D = E.depth.count;
    const other = link.direction === "dispatch" ? link.from : link.to;
    const otherSlabs = slabsOf(other, link.direction === "dispatch" ? link.fromSlab : link.toSlab);
    // dal router agli esperti: il neurone e del router (il punteggio dell'esperto e) va SOLO all'esperto e
    const fromRouter = link.direction === "dispatch" && other.role === "router" && sliceUnits(other) === E.experts;
    for (let d = 0; d < D; d++) {
      const all = sliceList(other, otherSlabs, other.depth.count === 1 ? 0 : d);
      for (let slab = 0; slab < E.slabs; slab++) {
        if (!route.mask[slab * D + d]) continue;
        const X = sliceList(E, [slab], d);
        const gate = slab < E.shared ? 1 : gateFor(route, d, slab - E.shared);
        const O = fromRouter && slab >= E.shared ? all.subarray(slab - E.shared, slab - E.shared + 1) : all;
        const p = fromRouter && slab >= E.shared ? 1 : density; // la scelta del router si vede tutta
        sample(O.length * X.length, p, rand, (i) => {
          const o = O[Math.floor(i / X.length)];
          const x = X[i % X.length];
          const pri = 0.5 * (1 - gate) + 0.5 * rand();
          if (link.direction === "dispatch") segments.push(o, x, gate, link.index, pri);
          else segments.push(x, o, gate, link.index, pri);
        });
      }
    }
  },

  /** Ogni unità di arrivo riceve dalla finestra kernel × kernel attorno alla stessa posizione. */
  conv({ link, rand, density, segments }) {
    const A = link.from;
    const B = link.to;
    const fromSlabs = slabsOf(A, link.fromSlab);
    const toSlabs = slabsOf(B, link.toSlab);
    const { rows, cols, kind } = A.depth;
    const half = (link.kernel - 1) / 2;
    const rowSpan = kind === "grid" ? half : 0;
    for (let d = 0; d < B.depth.count; d++) {
      const r = Math.floor(d / cols);
      const c = d % cols;
      const window = [];
      for (let dr = -rowSpan; dr <= rowSpan; dr++) {
        for (let dc = -half; dc <= half; dc++) {
          const rr = r + dr;
          const cc = c + dc;
          if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) window.push(rr * cols + cc);
        }
      }
      const outs = sliceList(B, toSlabs, d);
      // per ogni coppia (canale di arrivo, canale di partenza) scelta, TUTTA la finestra: si vede il ventaglio
      sample(outs.length * fromSlabs.length * A.neurons, density, rand, (i) => {
        const out = outs[Math.floor(i / (fromSlabs.length * A.neurons))];
        const k = i % (fromSlabs.length * A.neurons);
        const slab = fromSlabs[Math.floor(k / A.neurons)];
        const n = k % A.neurons;
        for (const p of window) {
          const w = simulatedWeight(rand);
          segments.push(A.offset + unitIndex(A, slab, n, p), out, w, link.index, weightPriority(w, rand));
        }
      });
    }
  },

  pool: (ctx) => collapse(ctx, "pool"),
  broadcast: (ctx) => collapse(ctx, "broadcast"),

  /** La griglia si rimpicciolisce: ogni posizione di arrivo riassume il suo blocco factor × factor. */
  down: (ctx) => resample(ctx, "down"),
  up: (ctx) => resample(ctx, "up"),

  explicit({ link, segments, sliceSegments }) {
    const { from, to, weight, count } = link.pairs;
    const deep = Math.max(link.from.depth.count, link.to.depth.count) > 1;
    if (link.slice === "selected" && deep) {
      // la posizione la sceglie lo studente: la scena li sposta sulla fetta scelta (indici dentro la fetta)
      for (let i = 0; i < count; i++) sliceSegments.push(from[i], to[i], weight[i], link.index, weightPriority(weight[i] / maxAbs(weight), () => 0));
      return;
    }
    const norm = maxAbs(weight);
    for (const [dA, dB] of pairsForSlice(link)) {
      for (let i = 0; i < count; i++) {
        const a = link.from.offset + sliceToUnit(link.from, from[i], dA);
        const b = link.to.offset + sliceToUnit(link.to, to[i], dB);
        segments.push(a, b, weight[i], link.index, weightPriority(weight[i] / norm, () => 0));
      }
    }
  },

  /** Archi tra posizioni: dalla chiave j (lastra delle chiavi/valori) alla query i (lastra di uscita). */
  attention({ net, link, lim, arcs }) {
    const { weights } = attentionWeights(net, link);
    const K = link.from;
    const Q = link.to;
    const Dq = Q.depth.count;
    const Dk = K.depth.count;
    const keySlab = 0;
    const querySlab = link.within ? Q.slabs - 1 : 0;
    const kind = link.within ? ARC_KIND.attention : ARC_KIND.cross;
    const keys = [];
    for (let h = 0; h < link.heads; h++) {
      const kn = headAnchorNeuron(K, h, link.heads);
      const qn = headAnchorNeuron(Q, h, link.heads);
      for (let i = 0; i < Dq; i++) {
        keys.length = 0;
        for (let j = 0; j < Dk; j++) {
          const w = weights[(h * Dq + i) * Dk + j];
          if (w >= lim.arcMinWeight) keys.push(j);
        }
        keys.sort((a, b) => weights[(h * Dq + i) * Dk + b] - weights[(h * Dq + i) * Dk + a]);
        for (let t = 0; t < Math.min(keys.length, lim.arcKeysPerQuery); t++) {
          const j = keys[t];
          const w = weights[(h * Dq + i) * Dk + j];
          arcs.push(K.offset + unitIndex(K, keySlab, kn, j), Q.offset + unitIndex(Q, querySlab, qn, i), w, link.index, 1 - w, h, kind);
        }
      }
    }
  },

  /**
   * Un arco per posizione sull'asse z (token, colonna della griglia), dalla cima di A alla cima di B:
   * in fila parte dal neurone 0 (il più in alto) dell'ultima lastra di A; in una griglia dalla riga in
   * alto dell'ultimo canale. Arriva al neurone 0 (o alla riga in alto del primo canale) di B.
   */
  residual({ link, arcs }) {
    const A = link.from;
    const B = link.to;
    const gridA = A.depth.kind === "grid";
    for (let c = 0; c < A.depth.cols; c++) {
      const from = gridA ? unitIndex(A, 0, A.neurons - 1, c) : unitIndex(A, A.slabs - 1, 0, c);
      const to = unitIndex(B, 0, 0, c);
      arcs.push(A.offset + from, B.offset + to, 1, link.index, 0, 0, ARC_KIND.residual);
    }
  },
};

const gateFor = (route, d, expert) => {
  for (let i = 0; i < route.k; i++) if (route.experts[d * route.k + i] === expert) return route.gates[d * route.k + i];
  return 0;
};

const maxAbs = (a) => {
  let m = 1e-12;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
  return m;
};

/** Indice di fetta (lastra × neuroni + neurone) → indice locale dell'unità alla posizione d. */
const sliceToUnit = (s, k, d) => k * s.depth.count + d;

/** pool: tutte le posizioni → una; broadcast: una → tutte. Stesse unità per fetta = uno a uno. */
function collapse({ link, rand, density, segments }, mode) {
  const deep = mode === "pool" ? link.from : link.to;
  const flat = mode === "pool" ? link.to : link.from;
  const deepSlabs = slabsOf(deep, mode === "pool" ? link.fromSlab : link.toSlab);
  const flatSlabs = slabsOf(flat, mode === "pool" ? link.toSlab : link.fromSlab);
  const F = sliceList(flat, flatSlabs, 0);
  for (let d = 0; d < deep.depth.count; d++) {
    const X = sliceList(deep, deepSlabs, d);
    const emit = (x, f) => (mode === "pool" ? segments.push(x, f, 1, link.index, rand()) : segments.push(f, x, 1, link.index, rand()));
    if (X.length === F.length) X.forEach((x, i) => (density >= 1 || rand() < density) && emit(x, F[i]));
    else sample(X.length * F.length, density, rand, (i) => emit(X[Math.floor(i / F.length)], F[i % F.length]));
  }
}

/** down/up: la posizione grande (r, c) ↔ la posizione piccola (r / f, c / f). */
function resample({ link, rand, density, segments }, mode) {
  const A = link.from;
  const B = link.to;
  const f = link.factor;
  const fromSlabs = slabsOf(A, link.fromSlab);
  const toSlabs = slabsOf(B, link.toSlab);
  const grid = A.depth.kind === "grid";
  const sameUnits = fromSlabs.length * A.neurons === toSlabs.length * B.neurons;
  for (let d = 0; d < B.depth.count; d++) {
    const r = grid ? Math.floor(d / B.depth.cols) : 0;
    const c = grid ? d % B.depth.cols : d;
    const sources = [];
    if (mode === "down") {
      for (let i = 0; i < (grid ? f : 1); i++) for (let j = 0; j < f; j++) sources.push(grid ? (r * f + i) * A.depth.cols + c * f + j : c * f + j);
    } else {
      sources.push(grid ? Math.floor(r / f) * A.depth.cols + Math.floor(c / f) : Math.floor(c / f));
    }
    const outs = sliceList(B, toSlabs, d);
    for (const p of sources) {
      const ins = sliceList(A, fromSlabs, p);
      if (sameUnits) ins.forEach((x, i) => (density >= 1 || rand() < density) && segments.push(x, outs[i], 1, link.index, rand()));
      else sample(ins.length * outs.length, density, rand, (i) => segments.push(ins[Math.floor(i / outs.length)], outs[i % outs.length], 1, link.index, rand()));
    }
  }
}
