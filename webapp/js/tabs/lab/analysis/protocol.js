/**
 * Il protocollo di benchmark "finestre-contigue-v1" nel browser, con il motore del corso
 * (js/engine/gpt.js), e il controllo di coerenza con i numeri del Python. Logica pura: gira uguale in
 * Node (tests/lab-analysis.test.mjs), in un Worker (worker.js) e, a pezzi, nella pagina (runner.js).
 *
 * È l'implementazione di riferimento di ARCHITECTURE.md ("Pesi e run", quella che usa anche
 * tests/test_benchmark.py), spezzata in finestre: `ExcerptScore.next()` fa UNA finestra, così chi la
 * usa può mostrare l'avanzamento e fermarsi senza bloccare la pagina.
 *
 *   const s = new ExcerptScore(ids, engine.config);
 *   while (s.next(engine)) {}            // oppure a pezzi, con un budget di tempo
 *   s.result()                           // { loss, perplexity, bpc, entropy, top1, top5, … }
 *   greedyText(engine, "ROMEO:\n", 200)  // sempre il carattere più probabile (a pari merito il primo)
 *   checkConsistency(browser, python)    // ✓ / ≈ / ✗ riga per riga
 */

export const PROTOCOL_ID = "finestre-contigue-v1";

/** Tolleranze del controllo di coerenza. `strict`: quelle chieste qui (✓); `python`: quelle di gpt/benchmark.py (≈). */
export const TOLERANCES = Object.freeze({
  strict: { loss: 1e-5, hits: 0 },
  python: { loss: 1e-4, hits: 2, bits: 2e-3 },
});

/** Le finestre contigue del protocollo: [{start, len}], la finestra legge ids[start…start+len) e prevede il token dopo ognuno. */
export function contiguousWindows(n, blockSize) {
  if (!Number.isSafeInteger(n) || !Number.isSafeInteger(blockSize) || blockSize < 1) throw new RangeError("contiguousWindows: servono interi positivi");
  const out = [];
  for (let start = 0; start + 1 < n; start += blockSize) out.push({ start, len: Math.min(blockSize, n - 1 - start) });
  return out;
}

/**
 * Il punteggio dell'estratto, una finestra alla volta. Somme in float64 (numeri di JS), log-softmax
 * stabile, top-k "senza ambiguità": giusto se MENO di k logits sono strettamente maggiori di quello del
 * token giusto (a pari merito vince il giusto, come in PyTorch con lo stesso criterio).
 */
export class ExcerptScore {
  /**
   * @param {ArrayLike<number>} ids il testo codificato
   * @param {{block_size: number, vocab_size: number}} config del motore
   * @param {{bits?: boolean}} [o] `bits`: tiene la sorpresa (−log₂ p) di ogni carattere previsto
   */
  constructor(ids, config, { bits = true } = {}) {
    this.ids = Array.from(ids);
    this.T = config.block_size;
    this.V = config.vocab_size;
    this.windowsList = contiguousWindows(this.ids.length, this.T);
    this.index = 0;
    this.nll = 0;
    this.entropy = 0;
    this.top1 = 0;
    this.top5 = 0;
    this.predictions = 0;
    this.bits = bits ? [] : null;
  }

  get total() {
    return this.windowsList.length;
  }

  get done() {
    return this.index;
  }

  get finished() {
    return this.index >= this.windowsList.length;
  }

  /** Fa la prossima finestra con `engine.forward`; false quando non ce ne sono più. */
  next(engine) {
    if (this.finished) return false;
    const { start, len } = this.windowsList[this.index++];
    const { logits } = engine.forward(this.ids.slice(start, start + len));
    const V = this.V;
    for (let t = 0; t < len; t++) {
      const off = t * V;
      const target = this.ids[start + t + 1];
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (logits[off + i] > max) max = logits[off + i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(logits[off + i] - max);
      const logZ = max + Math.log(sum);
      const right = logits[off + target];
      let h = 0;
      let greater = 0;
      for (let i = 0; i < V; i++) {
        const lp = logits[off + i] - logZ;
        h -= Math.exp(lp) * lp;
        if (logits[off + i] > right) greater++;
      }
      const lp = right - logZ;
      this.nll -= lp;
      this.entropy += h;
      this.predictions++;
      if (this.bits) this.bits.push(-lp / Math.LN2);
      if (greater < 1) this.top1++;
      if (greater < 5) this.top5++;
    }
    return !this.finished;
  }

  /** I numeri del protocollo (anche a metà: utile per l'avanzamento). */
  result() {
    const n = this.predictions;
    const loss = n ? this.nll / n : NaN;
    return {
      protocol: PROTOCOL_ID,
      predictions: n,
      windows: this.index,
      loss,
      perplexity: Math.exp(loss),
      bpc: n ? this.nll / (Math.LN2 * n) : NaN,
      entropy: n ? this.entropy / n : NaN,
      top1: this.top1,
      top5: this.top5,
      top1Rate: n ? this.top1 / n : NaN,
      top5Rate: n ? this.top5 / n : NaN,
      bits: this.bits,
    };
  }
}

/** Il protocollo tutto in una volta (riferimento di ARCHITECTURE.md). */
export function scoreText(engine, ids, options) {
  const s = new ExcerptScore(ids, engine.config, options);
  while (s.next(engine));
  return s.result();
}

/** Indice del massimo, a pari merito il primo (come argmax di PyTorch). */
export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * La continuazione greedy, un token alla volta: sempre il più probabile, con il contesto tagliato agli
 * ultimi block_size token (engine.nextTokenDistribution lo fa da solo), come greedy_continuation.
 */
export class GreedyRun {
  constructor(promptIds, maxNewTokens) {
    this.context = Array.from(promptIds);
    this.start = this.context.length;
    this.total = Math.max(0, maxNewTokens | 0);
  }

  get done() {
    return this.context.length - this.start;
  }

  get finished() {
    return this.done >= this.total;
  }

  next(engine) {
    if (this.finished) return false;
    const { logits } = engine.nextTokenDistribution(this.context);
    this.context.push(argmax(logits));
    return !this.finished;
  }

  ids() {
    return this.context.slice(this.start);
  }
}

/** Il testo greedy tutto in una volta. */
export function greedyText(engine, prompt, maxNewTokens) {
  const run = new GreedyRun(engine.encode(prompt), maxNewTokens);
  while (run.next(engine));
  return engine.decode(run.ids());
}

/** La prima posizione in cui due testi differiscono (−1 se uguali). */
export function firstDifference(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

/**
 * Il controllo di coerenza tra il calcolo del browser e il Python (excerpt di benchmark.json).
 * Livelli: "ok" (✓: loss entro 1e-5, previsioni giuste top-1/top-5 identiche, stesse finestre),
 * "near" (≈: entro le tolleranze di gpt/benchmark.py), "bad" (✗), "info" (solo da leggere).
 * @param {object} browser risultato di ExcerptScore.result() (+ `greedy` facoltativo)
 * @param {object|null} python l'excerpt di readBenchmark (con counts) e, facoltativo, il testo greedy
 * @param {{greedy?: string|null, pythonGreedy?: string|null}} [o]
 * @returns {{verdict: "ok"|"near"|"bad"|"none", rows: {id, label, python, browser, delta, level}[]}}
 */
export function checkConsistency(browser, python, { greedy = null, pythonGreedy = null } = {}) {
  if (!python) return { verdict: "none", rows: [] };
  const rows = [];
  const counts = python.counts ?? {};
  const add = (id, label, py, js, delta, level) => rows.push({ id, label, python: py, browser: js, delta, level });
  const same = (a, b) => a !== null && a !== undefined && a === b;

  add("windows", "finestre", counts.windows ?? null, browser.windows, null, counts.windows === null || counts.windows === undefined ? "info" : same(counts.windows, browser.windows) ? "ok" : "bad");
  add("predictions", "previsioni", counts.predictions ?? null, browser.predictions, null, counts.predictions === null || counts.predictions === undefined ? "info" : same(counts.predictions, browser.predictions) ? "ok" : "bad");
  const loss = level(python.loss, browser.loss, TOLERANCES.strict.loss, TOLERANCES.python.loss);
  add("loss", "loss (nat)", python.loss, browser.loss, loss.delta, loss.level);
  for (const k of ["top1", "top5"]) {
    const py = counts[k];
    const delta = Number.isSafeInteger(py) ? browser[k] - py : null;
    const lv = delta === null ? "info" : Math.abs(delta) <= TOLERANCES.strict.hits ? "ok" : Math.abs(delta) <= TOLERANCES.python.hits ? "near" : "bad";
    add(k, k === "top1" ? "previsioni giuste top-1" : "previsioni giuste top-5", py ?? null, browser[k], delta, lv);
  }
  const entropy = level(python.entropy, browser.entropy, TOLERANCES.strict.loss, TOLERANCES.python.loss);
  add("entropy", "entropia media (nat)", python.entropy, browser.entropy, entropy.delta, entropy.level);
  const bpc = level(python.bpc, browser.bpc, TOLERANCES.strict.loss / Math.LN2, TOLERANCES.python.loss / Math.LN2);
  add("bpc", "bit per carattere", python.bpc, browser.bpc, bpc.delta, bpc.level);
  if (Array.isArray(python.surprisalBits) && Array.isArray(browser.bits)) {
    let worst = python.surprisalBits.length === browser.bits.length ? 0 : Infinity;
    if (worst === 0) for (let i = 0; i < browser.bits.length; i++) worst = Math.max(worst, Math.abs(browser.bits[i] - python.surprisalBits[i]));
    // il Python salva i bit con 3 decimali: ±5·10⁻⁴ sono solo l'arrotondamento del file
    add("bits", "bit di sorpresa, carattere per carattere (max |Δ|)", python.surprisalBits.length, browser.bits.length, worst, worst <= TOLERANCES.python.bits ? "ok" : "bad");
  }
  if (typeof greedy === "string" && typeof pythonGreedy === "string") {
    const at = firstDifference(greedy, pythonGreedy);
    add("greedy", "testo greedy", pythonGreedy.length, greedy.length, at, at < 0 ? "ok" : "bad");
  }
  const levels = rows.map((r) => r.level).filter((l) => l !== "info");
  const verdict = !levels.length ? "none" : levels.includes("bad") ? "bad" : levels.includes("near") ? "near" : "ok";
  return { verdict, rows };
}

function level(py, js, strict, loose) {
  if (typeof py !== "number" || !Number.isFinite(py) || !Number.isFinite(js)) return { delta: null, level: "info" };
  const delta = js - py;
  const abs = Math.abs(delta);
  return { delta, level: abs <= strict ? "ok" : abs <= loose ? "near" : "bad" };
}
