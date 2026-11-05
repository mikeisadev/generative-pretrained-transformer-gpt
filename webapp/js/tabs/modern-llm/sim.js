/**
 * Matematica pura del tab "LLM moderni" (niente DOM, niente Three.js).
 *
 * Due tipi di funzioni, e il commento di ognuna dice quale:
 * - FORMULE PUBBLICATE con numeri verificati (legge di Chinchilla, memoria della KV cache
 *   calcolata dalle configurazioni ufficiali, frequenze di RoPE, vantaggi di GRPO, BPE,
 *   numero atteso di token della decodifica speculativa);
 * - SIMULAZIONI DIDATTICHE dichiarate (router del MoE con pesi casuali, curva "emergente").
 * Il pannello del codice mostra il sorgente vero di alcune di queste funzioni.
 */

/** Generatore pseudo-casuale con seme (mulberry32): stesse animazioni a ogni avvio. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------------------------------
// Modelli aperti: numeri dalle configurazioni ufficiali (vedi pannello Fonti)
// ------------------------------------------------------------------------------------------

/** Llama 3 8B (meta-llama/llama-models, sku_list.py): dim 4096, 32 layer, 32 head, 8 head K/V. */
export const LLAMA3_8B = { name: "Llama 3 8B", layers: 32, heads: 32, kvHeads: 8, headDim: 4096 / 32, vocab: 128256, ropeTheta: 500000, ffn: 14336 };

/** DeepSeek-V3 (inference/configs/config_671B.json): MLA con latente K/V da 512 + chiave RoPE da 64. */
export const DEEPSEEK_V3 = { name: "DeepSeek-V3", layers: 61, heads: 128, kvLoraRank: 512, ropeHeadDim: 64, routedExperts: 256, sharedExperts: 1, activeExperts: 8 };

/**
 * FORMULA: byte della KV cache per UN token in tutto il modello.
 * Per ogni layer si salvano K e V (fattore 2) per ogni head K/V, ognuno lungo headDim.
 */
export function kvBytesPerToken({ layers, kvHeads, headDim }, bytesPerValue = 2) {
  return 2 * layers * kvHeads * headDim * bytesPerValue;
}

/**
 * FORMULA (DeepSeek-V2, MLA): per ogni layer si salva solo il latente compresso c_KV
 * (kv_lora_rank valori) e la chiave RoPE condivisa (qk_rope_head_dim valori).
 */
export function mlaBytesPerToken({ layers, kvLoraRank, ropeHeadDim }, bytesPerValue = 2) {
  return layers * (kvLoraRank + ropeHeadDim) * bytesPerValue;
}

// ------------------------------------------------------------------------------------------
// Leggi di scala: Chinchilla (Hoffmann et al. 2022, "approccio 3")
// ------------------------------------------------------------------------------------------

/** Coefficienti pubblicati del fit parametrico L(N, D) = E + A/N^α + B/D^β. */
export const CHINCHILLA = { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 };

/** FORMULA: loss prevista per un modello da N parametri addestrato su D token. */
export function chinchillaLoss(N, D, c = CHINCHILLA) {
  return c.E + c.A / Math.pow(N, c.alpha) + c.B / Math.pow(D, c.beta);
}

/**
 * FORMULA: allocazione ottimale di un budget di calcolo C ≈ 6·N·D FLOP (eq. 4 del paper):
 * N_opt = G·(C/6)^a, D_opt = (C/6)^b / G, con a = β/(α+β), b = α/(α+β), G = (αA/βB)^(1/(α+β)).
 */
export function chinchillaOptimal(C, c = CHINCHILLA) {
  const s = c.alpha + c.beta;
  const G = Math.pow((c.alpha * c.A) / (c.beta * c.B), 1 / s);
  const N = G * Math.pow(C / 6, c.beta / s);
  const D = Math.pow(C / 6, c.alpha / s) / G;
  return { N, D, loss: chinchillaLoss(N, D, c) };
}

// ------------------------------------------------------------------------------------------
// Mixture of Experts: router (SIMULAZIONE DIDATTICA con pesi casuali)
// ------------------------------------------------------------------------------------------

export function softmax(values, out = new Float64Array(values.length)) {
  let max = -Infinity;
  for (const v of values) max = Math.max(max, v);
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += out[i] = Math.exp(values[i] - max);
  for (let i = 0; i < values.length; i++) out[i] /= sum;
  return out;
}

/** Indici dei k valori più grandi, dal maggiore al minore. */
export function topK(values, k) {
  return [...values.keys()].sort((a, b) => values[b] - values[a]).slice(0, k);
}

/**
 * Router come in Mixtral (mistral-inference, moe.py): logit = gate(x), si prendono i top-k
 * logit e il softmax SOLO su quelli dà i pesi con cui sommare le uscite degli esperti.
 */
export function routeTopK(logits, k) {
  const chosen = topK(logits, k);
  const weights = softmax(chosen.map((i) => logits[i]));
  return { chosen, weights: [...weights], probs: [...softmax(logits)] };
}

/**
 * SIMULAZIONE: un piccolo router con embedding e pesi casuali (seme fisso).
 * `bias` (uno per esperto) sposta la SCELTA degli esperti senza cambiare i pesi della somma:
 * è l'idea del bilanciamento "senza loss ausiliaria" di DeepSeek-V3.
 */
export class RouterSim {
  constructor({ experts = 8, dim = 16, seed = 7, skew = 0 } = {}) {
    const rand = mulberry32(seed);
    const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    this.experts = experts;
    this.dim = dim;
    this.gauss = gauss;
    this.W = Array.from({ length: experts }, () => Float64Array.from({ length: dim }, gauss));
    // skew > 0: i primi esperti partono "favoriti", come capita a inizio training
    this.prior = Float64Array.from({ length: experts }, (_, e) => skew * Math.max(0, 1.5 - e));
    this.bias = new Float64Array(experts);
  }

  token() {
    return Float64Array.from({ length: this.dim }, this.gauss);
  }

  logits(x) {
    return this.W.map((w, e) => {
      let s = this.prior[e];
      for (let i = 0; i < this.dim; i++) s += (w[i] * x[i]) / Math.sqrt(this.dim);
      return s;
    });
  }

  route(x, k = 2) {
    const logits = this.logits(x);
    const chosen = topK(logits.map((l, e) => l + this.bias[e]), k);
    const weights = [...softmax(chosen.map((e) => logits[e]))];
    return { logits, chosen, weights, probs: [...softmax(logits)] };
  }

  /** Conta quanti token vanno a ogni esperto su `n` token casuali. */
  load(n = 512, k = 2) {
    const counts = new Array(this.experts).fill(0);
    for (let t = 0; t < n; t++) for (const e of this.route(this.token(), k).chosen) counts[e]++;
    return counts;
  }

  /** Un passo di bilanciamento: abbassa il bias degli esperti sovraccarichi, alza gli altri. */
  balanceStep(counts, gamma = 0.15) {
    const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
    counts.forEach((c, e) => (this.bias[e] += c > mean ? -gamma : gamma));
  }
}

// ------------------------------------------------------------------------------------------
// RoPE (Su et al. 2021)
// ------------------------------------------------------------------------------------------

/** FORMULA: frequenza della coppia i su d dimensioni: θ_i = base^(-2i/d). */
export function ropeFrequency(i, d, base = 10000) {
  return Math.pow(base, (-2 * i) / d);
}

/** FORMULA: ruota la coppia (x0, x1) dell'angolo m·θ (posizione m). */
export function ropeRotate(x0, x1, m, theta) {
  const a = m * theta;
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [x0 * c - x1 * s, x0 * s + x1 * c];
}

// ------------------------------------------------------------------------------------------
// Tokenizer BPE a byte (stesso algoritmo di karpathy/minbpe, BasicTokenizer)
// ------------------------------------------------------------------------------------------

/** Conta le coppie consecutive di id. */
export function pairCounts(ids) {
  const counts = new Map();
  for (let i = 0; i + 1 < ids.length; i++) {
    const key = `${ids[i]},${ids[i + 1]}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

/** Sostituisce ogni occorrenza della coppia con il nuovo id. */
export function mergePair(ids, [a, b], idx) {
  const out = [];
  for (let i = 0; i < ids.length; i++) {
    if (ids[i] === a && i + 1 < ids.length && ids[i + 1] === b) {
      out.push(idx);
      i++;
    } else out.push(ids[i]);
  }
  return out;
}

/**
 * Addestra un BPE a byte: si parte dai byte UTF-8 (256 simboli) e si fonde per `merges`
 * volte la coppia più frequente. Restituisce la storia di ogni fusione (per l'animazione).
 */
export function trainBpe(text, merges) {
  let ids = [...new TextEncoder().encode(text)];
  const vocab = new Map(Array.from({ length: 256 }, (_, i) => [i, [i]]));
  const history = [{ ids, pair: null, idx: null, count: 0 }];
  for (let m = 0; m < merges; m++) {
    const counts = pairCounts(ids);
    if (!counts.size) break;
    let best = null;
    for (const [key, count] of counts) if (!best || count > best.count) best = { key, count };
    if (best.count < 2) break; // fondere una coppia che compare una volta sola non comprime nulla
    const pair = best.key.split(",").map(Number);
    const idx = 256 + m;
    vocab.set(idx, [...vocab.get(pair[0]), ...vocab.get(pair[1])]);
    ids = mergePair(ids, pair, idx);
    history.push({ ids, pair, idx, count: best.count });
  }
  return { history, vocab };
}

/** Testo leggibile di un token: i byte che non formano un carattere completo diventano esadecimali. */
export function tokenText(bytes) {
  const text = new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(bytes));
  if (text.includes("�")) return bytes.map((b) => `0x${b.toString(16).toUpperCase()}`).join(" ");
  return text;
}

// ------------------------------------------------------------------------------------------
// Ragionamento e RL
// ------------------------------------------------------------------------------------------

/**
 * FORMULA (GRPO, Shao et al. 2024): per un gruppo di risposte alla stessa domanda il vantaggio
 * è la ricompensa meno la media del gruppo, diviso la deviazione standard del gruppo.
 * Niente critic: il "termine di paragone" è il gruppo stesso.
 */
export function grpoAdvantages(rewards, eps = 1e-4) {
  const n = rewards.length;
  const mean = rewards.reduce((a, b) => a + b, 0) / n;
  const variance = rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / Math.max(1, n - 1); // come torch.std
  const std = Math.sqrt(variance);
  return rewards.map((r) => (r - mean) / (std + eps));
}

/** FORMULA: probabilità che almeno una di N risposte indipendenti sia giusta (verificatore perfetto). */
export function passAtN(p, n) {
  return 1 - Math.pow(1 - p, n);
}

/**
 * FORMULA (Leviathan et al. 2023): numero atteso di token prodotti per ogni passaggio del modello
 * grande quando la bozza propone γ token e ognuno viene accettato con probabilità α.
 */
export function speculativeExpectedTokens(alpha, gamma) {
  if (alpha >= 1) return gamma + 1;
  return (1 - Math.pow(alpha, gamma + 1)) / (1 - alpha);
}

/**
 * ILLUSTRAZIONE della critica di Schaeffer et al. (2023): se la probabilità di azzeccare un
 * token cresce dolcemente con la scala, l'esatta corrispondenza su L token (p^L) sembra
 * "accendersi" all'improvviso. x = log10(parametri); la curva p(x) è inventata e dichiarata.
 */
export function emergenceIllustration(x, L = 10) {
  const perToken = 0.5 + 0.49 / (1 + Math.exp(-1.3 * (x - 10)));
  return { perToken, exactMatch: Math.pow(perToken, L) };
}

/**
 * FORMULA (Bavarian et al. 2022, modalità PSM): il documento viene tagliato in prefisso, centro e
 * suffisso e riscritto come <PRE> prefisso <SUF> suffisso <MID> centro. Il modello continua a
 * leggere da sinistra a destra, ma il centro arriva per ultimo: impara a "riempire il buco".
 */
export function fimTransform(doc, start, end) {
  const prefix = doc.slice(0, start);
  const middle = doc.slice(start, end);
  const suffix = doc.slice(end);
  return `<PRE>${prefix}<SUF>${suffix}<MID>${middle}`;
}
