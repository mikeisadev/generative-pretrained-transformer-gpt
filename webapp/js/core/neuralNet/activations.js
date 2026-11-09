/**
 * Valori della rete: attivazioni vere (section.values) oppure simulate con un seme, instradamento
 * degli esperti e pesi di attention (logica pura, testata con node).
 *
 * Simulazione didattica: i numeri NON vengono da un modello, ma hanno la forma giusta perché la
 * scena racconti il meccanismo: un ingresso one-hot per token, attivazioni che variano con
 * continuità tra token vicini, mappe spaziali lisce in una griglia (come le feature map di una
 * convoluzione), probabilità che sommano a 1, esperti spenti per i token che non li scelgono, head
 * di attention con "personalità" diverse (token precedente, primo token, vicini, contenuto). La
 * vista lo dichiara sempre ("simulazione didattica").
 */
import { gaussian, rngFor } from "./random.js";
import { sliceUnits } from "./spec.js";

/** Modo di colore di ogni unità (lo legge lo shader dei neuroni). */
export const MODE = Object.freeze({ signed: 0, probs: 1, onehot: 2, off: 3 });

/**
 * @param {import("./spec.js").NormalizedNetwork} net
 * @returns {{ raw: Float32Array, display: Float32Array, mode: Uint8Array, routing: Map<string, Routing>, simulated: Set<string> }}
 *   raw = il valore da mostrare al passaggio del mouse; display = -1…1 (o 0…1) per il colore.
 */
export function resolveValues(net) {
  const raw = new Float32Array(net.units);
  const display = new Float32Array(net.units);
  const mode = new Uint8Array(net.units);
  const simulated = new Set();
  const routing = new Map();
  // prima i router (i loro punteggi decidono gli esperti), poi tutto il resto, gli esperti per ultimi
  const order = [...net.sections].sort((a, b) => rank(a) - rank(b));
  for (const s of order) {
    let values = s.values;
    if (s.role === "experts") routing.set(s.id, computeRouting(net, s, raw));
    if (!values) {
      values = simulateSection(net, s, routing.get(s.id));
      simulated.add(s.id);
    }
    raw.set(values, s.offset);
    writeDisplay(s, values, display, mode, routing.get(s.id));
  }
  return { raw, display, mode, routing, simulated };
}

const rank = (s) => (s.role === "router" ? 0 : s.role === "experts" ? 2 : 1);

/** Valori di colore di una sezione: segno e intensità normalizzati sulla sezione (come neuronScene.js). */
export function writeDisplay(s, values, display, mode, route) {
  const o = s.offset;
  const D = s.depth.count;
  if (s.activation === "onehot") {
    for (let u = 0; u < s.units; u++) {
      display[o + u] = values[u] > 0.5 ? 1 : 0;
      mode[o + u] = MODE.onehot;
    }
    return;
  }
  let max = 1e-9;
  for (let u = 0; u < s.units; u++) max = Math.max(max, Math.abs(values[u]));
  const probs = s.activation === "probs";
  for (let u = 0; u < s.units; u++) {
    const v = values[u] / max;
    display[o + u] = probs ? Math.sqrt(Math.max(0, v)) : v;
    mode[o + u] = probs ? MODE.probs : MODE.signed;
  }
  if (route) {
    // un esperto non scelto da quel token è spento (non "a zero": proprio non calcolato)
    for (let u = 0; u < s.units; u++) {
      const d = u % D;
      const slab = Math.floor(u / (s.neurons * D));
      if (!route.mask[slab * D + d]) {
        display[o + u] = 0;
        mode[o + u] = MODE.off;
      }
    }
  }
}

/**
 * @typedef {{ k: number, shared: number, experts: Int32Array, gates: Float32Array, mask: Uint8Array, source: "spec"|"router"|"simulated" }} Routing
 *   experts[d * k + i] = i-esimo esperto instradato (0…experts-1) del token d; mask[slab * D + d] = 1 se la lastra lavora per d
 */
export function computeRouting(net, s, raw) {
  const D = s.depth.count;
  const k = s.active;
  const experts = new Int32Array(D * k);
  const gates = new Float32Array(D * k);
  let source = "spec";
  if (s.routes) {
    experts.set(s.routes);
    if (s.gates) gates.set(s.gates);
    else gates.fill(1 / k);
  } else {
    // punteggi: dal router collegato (se ha un neurone per esperto), altrimenti simulati con seme
    // il router: quello dichiarato (router: "id") o la sezione router che instrada qui con un "route"
    const routerSection = s.router ?? net.links.find((l) => l.pattern === "route" && l.to === s && l.from.role === "router" && sliceUnits(l.from) === s.experts)?.from ?? null;
    const rand = rngFor(net.seed, "route", s.id);
    const scores = new Float32Array(s.experts);
    source = routerSection ? "router" : "simulated";
    for (let d = 0; d < D; d++) {
      if (routerSection) {
        const r = routerSection;
        const rd = r.depth.count === 1 ? 0 : d;
        for (let e = 0; e < s.experts; e++) scores[e] = raw[r.offset + e * r.depth.count + rd];
      } else {
        for (let e = 0; e < s.experts; e++) scores[e] = gaussian(rand);
      }
      const top = topK(scores, k);
      // pesi dei k esperti scelti: le probabilità del router rinormalizzate sui k (come Mixtral e
      // DeepSeek); con punteggi simulati, la softmax dei k punteggi
      const weight = (e) => (routerSection ? Math.max(1e-6, scores[e]) : Math.exp(scores[e] - scores[top[0]]));
      let sum = 0;
      for (let i = 0; i < k; i++) sum += weight(top[i]);
      for (let i = 0; i < k; i++) {
        experts[d * k + i] = top[i];
        gates[d * k + i] = weight(top[i]) / sum;
      }
    }
  }
  const mask = new Uint8Array(s.slabs * D);
  for (let d = 0; d < D; d++) {
    for (let e = 0; e < s.shared; e++) mask[e * D + d] = 1;
    for (let i = 0; i < k; i++) mask[(s.shared + experts[d * k + i]) * D + d] = 1;
  }
  return { k, shared: s.shared, experts, gates, mask, source };
}

/** Indici dei k valori più grandi (in ordine decrescente). */
export function topK(values, k) {
  const idx = Array.from(values.keys());
  idx.sort((a, b) => values[b] - values[a] || a - b);
  return idx.slice(0, k);
}

/** Attivazioni simulate di una sezione, nell'ordine dei `values` (lastra, neurone, posizione). */
export function simulateSection(net, s, route) {
  const rand = rngFor(net.seed, "values", s.id);
  const D = s.depth.count;
  const N = s.neurons;
  const out = new Float32Array(s.units);
  if (s.activation === "onehot") {
    // un'unità accesa per posizione: per i token con un nome, sempre la stessa per lo stesso nome
    const units = sliceUnits(s);
    for (let d = 0; d < D; d++) {
      const name = s.depth.names?.[d];
      const hot = name ? rngFor("token", name)() * units : rand() * units;
      out[Math.floor(hot) * D + d] = 1;
    }
    return out;
  }
  if (s.activation === "probs") {
    const units = sliceUnits(s);
    const logits = new Float32Array(units);
    for (let d = 0; d < D; d++) {
      let max = -Infinity;
      for (let i = 0; i < units; i++) max = Math.max(max, (logits[i] = 1.4 * gaussian(rand)));
      logits[Math.floor(rand() * units)] += 2.5; // una scelta netta, come un modello sicuro
      let sum = 0;
      for (let i = 0; i < units; i++) sum += Math.exp(logits[i] - max - 2.5);
      for (let i = 0; i < units; i++) out[i * D + d] = Math.exp(logits[i] - max - 2.5) / sum;
    }
    return out;
  }
  if (s.depth.kind === "grid") {
    // feature map lisce: somma di poche onde per canale, più un filo di rumore
    const { rows, cols } = s.depth;
    for (let c = 0; c < N; c++) {
      const waves = [0, 1, 2].map(() => ({ fr: 0.25 + rand() * 0.9, fc: 0.25 + rand() * 0.9, phase: rand() * 6.283, amp: 0.5 + rand() }));
      for (let r = 0; r < rows; r++) {
        for (let col = 0; col < cols; col++) {
          let v = 0.25 * gaussian(rand);
          for (const w of waves) v += w.amp * Math.sin(w.fr * r + w.fc * col + w.phase);
          out[c * D + r * cols + col] = v;
        }
      }
    }
  } else {
    // in fila: una componente per neurone + una passeggiata lungo i token (token vicini si somigliano)
    for (let slab = 0; slab < s.slabs; slab++) {
      for (let n = 0; n < N; n++) {
        const base = 0.8 * gaussian(rand);
        let walk = gaussian(rand);
        for (let d = 0; d < D; d++) {
          walk = 0.7 * walk + 0.71 * gaussian(rand);
          out[(slab * N + n) * D + d] = base + 0.7 * walk;
        }
      }
    }
  }
  if (s.role === "norm") normalizePerSlice(s, out);
  if (s.activation === "positive") for (let i = 0; i < out.length; i++) out[i] = Math.max(0, out[i] - 0.35);
  if (route) {
    // l'esperto lavora solo per i token che l'hanno scelto, con più forza se il peso del router è alto
    for (let slab = 0; slab < s.slabs; slab++) {
      for (let d = 0; d < D; d++) {
        const on = route.mask[slab * D + d];
        const gate = slab < route.shared ? 1 : gateOf(route, d, slab - route.shared);
        for (let n = 0; n < N; n++) out[(slab * N + n) * D + d] *= on ? 0.6 + gate : 0;
      }
    }
  }
  return out;
}

const gateOf = (route, d, expert) => {
  for (let i = 0; i < route.k; i++) if (route.experts[d * route.k + i] === expert) return route.gates[d * route.k + i];
  return 0;
};

/** Una LayerNorm vera: media 0 e varianza 1 tra i neuroni di ogni posizione. */
function normalizePerSlice(s, out) {
  const D = s.depth.count;
  const units = sliceUnits(s);
  for (let d = 0; d < D; d++) {
    let mean = 0;
    for (let i = 0; i < units; i++) mean += out[i * D + d];
    mean /= units;
    let variance = 0;
    for (let i = 0; i < units; i++) variance += (out[i * D + d] - mean) ** 2;
    const std = Math.sqrt(variance / units) || 1;
    for (let i = 0; i < units; i++) out[i * D + d] = (out[i * D + d] - mean) / std;
  }
}

/** Stili delle head simulate: token precedente, primo token ("attention sink"), vicini, contenuto. */
const STYLE = Object.freeze({ previous: 0, sink: 1, local: 2, content: 3 });

/**
 * Pesi di attention simulati: Float32Array(heads × Dq × Dk), ogni riga (head, query) somma a 1.
 * Le head hanno comportamenti tipici osservati nei transformer veri: guardare il token precedente,
 * il primo token ("attention sink"), i vicini, oppure poche posizioni scelte dal contenuto.
 * Su una griglia (feature map di una U-Net) "il primo" e "il precedente" non hanno senso: la prima
 * posizione è l'angolo in alto a sinistra, e tutte le query che lo guardano disegnerebbero un ventaglio
 * verso l'angolo che domina le viste da vicino. Lì le head alternano schemi locali e di contenuto.
 * @param {{ causal?: boolean, qCols?: number, kCols?: number, seed?: number[] }} o
 *   qCols/kCols: per le griglie, quante colonne (la distanza tra posizioni è quella nella griglia)
 */
export function simulateAttention(heads, Dq, Dk, { causal = false, qCols = Dq, kCols = Dk, seed = [1] } = {}) {
  const rand = rngFor(...seed, "attention");
  const out = new Float32Array(heads * Dq * Dk);
  const logits = new Float32Array(Dk);
  const grid = qCols !== Dq || kCols !== Dk;
  const sameSpace = Dq === Dk;
  for (let h = 0; h < heads; h++) {
    const style = !sameSpace ? STYLE.content : grid ? (h % 2 === 0 ? STYLE.local : STYLE.content) : h % 4;
    const peaks = Array.from({ length: Dq }, () => [Math.floor(rand() * Dk), Math.floor(rand() * Dk)]);
    for (let i = 0; i < Dq; i++) {
      let max = -Infinity;
      for (let j = 0; j < Dk; j++) {
        if (causal && j > i) {
          logits[j] = -Infinity;
          continue;
        }
        const dist = grid ? Math.abs(Math.floor(i / qCols) - Math.floor(j / kCols)) + Math.abs((i % qCols) - (j % kCols)) : Math.abs(i - j);
        let z = 0.35 * gaussian(rand);
        if (style === STYLE.previous) z += j === i - 1 || (i === 0 && j === 0) ? 3.4 : j === i ? 1.2 : 0;
        else if (style === STYLE.sink) z += j === 0 ? 3 : 0.4 * (j === i);
        else if (style === STYLE.local) z += -0.9 * dist;
        else z += j === peaks[i][0] ? 3 : j === peaks[i][1] ? 2 : 0.6 * gaussian(rand);
        logits[j] = z;
        max = Math.max(max, z);
      }
      let sum = 0;
      for (let j = 0; j < Dk; j++) sum += logits[j] === -Infinity ? 0 : Math.exp(logits[j] - max);
      const row = (h * Dq + i) * Dk;
      for (let j = 0; j < Dk; j++) out[row + j] = logits[j] === -Infinity ? 0 : Math.exp(logits[j] - max) / sum;
    }
  }
  return out;
}

/** Pesi di attention di un collegamento: quelli veri della specifica o simulati (con il seme della rete). */
export function attentionWeights(net, link) {
  if (link.weights) return { weights: link.weights, simulated: false };
  const q = link.to;
  const k = link.from;
  const weights = simulateAttention(link.heads, q.depth.count, k.depth.count, {
    causal: link.causal,
    qCols: q.depth.kind === "grid" ? q.depth.cols : q.depth.count,
    kCols: k.depth.kind === "grid" ? k.depth.cols : k.depth.count,
    seed: [net.seed, "link", link.index],
  });
  return { weights, simulated: true };
}

/**
 * Neurone "centrale" della head h (lì partono e arrivano i suoi archi): il centro della sua fascia di
 * neuroni in una sezione in fila, il canale centrale del suo gruppo in una griglia.
 */
export function headAnchorNeuron(s, h, heads) {
  return Math.min(s.neurons - 1, Math.floor(((h + 0.5) * s.neurons) / heads));
}
