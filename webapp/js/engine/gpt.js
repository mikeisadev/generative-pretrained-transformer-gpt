/**
 * Motore di inferenza del GPT del corso, in JavaScript puro (nessun DOM: browser e Node 22).
 *
 * Esegue ESATTAMENTE il forward di gpt/model.py con i pesi esportati da gpt/export.py:
 *
 *   ids (T)  →  token embedding + position embedding  →  x (T × C)
 *   per ogni blocco (gpt/block.py, "pre-norm"):
 *     x = x + MultiHeadAttention(LayerNorm1(x))     comunicazione tra token
 *     x = x + FeedForward(LayerNorm2(x))            elaborazione di ogni token
 *   LayerNorm finale  →  lm_head  →  logits (T × V)  →  softmax  →  prossimo token
 *
 * Un test (tests/test_js_engine.py) confronta logits e attivazioni con PyTorch.
 *
 * Uso nel browser:
 *   const engine = await loadEngine("data/");
 *   const { probs } = engine.nextTokenDistribution(engine.encode("ROMEO:"));
 * Uso in Node:
 *   const engine = GPTEngine.fromBuffers(JSON.parse(fs.readFileSync(json)), fs.readFileSync(bin));
 *
 * Tutte le matrici sono Float32Array piatte, riga per riga: l'elemento (r, c) di una
 * matrice (R × K) sta in `data[r * K + c]`. Simboli: T lunghezza della sequenza,
 * C = n_embd, V = vocab_size, hs = head_size = C / n_head, H4 = 4·C (strato nascosto FFN).
 */

import { fetchBuffer, fetchJson } from "../core/binary.js";
import { addInPlace, causalSelfAttention, layerNorm, linear, reluInPlace, softmax } from "./kernels.js";
import { distributionFromLogits, mulberry32, sample, samplingOptions } from "./sampling.js";
import {
  FFN_MULTIPLIER,
  assertSameLayout,
  decodeFloat16,
  decodeFloat32,
  expectedParameters,
  validateManifest,
} from "./weights.js";

export { expectedParameters, mulberry32, sample };

/** eps di `nn.LayerNorm` (default di PyTorch). */
export const LAYER_NORM_EPS = 1e-5;
/** Seed di default della generazione: lo stesso del training del corso. */
export const DEFAULT_SEED = 1337;
const WEIGHTS_MANIFEST = "weights.json";

/** Errore di `encode` per caratteri fuori vocabolario; `unknown` elenca i caratteri. */
export class VocabularyError extends Error {
  constructor(unknown) {
    const shown = unknown.map((ch) => JSON.stringify(ch)).join(", ");
    super(`caratteri non presenti nel vocabolario: ${shown}`);
    this.name = "VocabularyError";
    this.unknown = unknown;
  }
}

/**
 * @typedef {object} HeadTrace       attivazioni di una head di attention
 * @property {Float32Array} q        query (T × hs)
 * @property {Float32Array} k        key (T × hs)
 * @property {Float32Array} v        value (T × hs)
 * @property {Float32Array} scores   q·kᵀ·hs^-0.5 PRIMA della softmax (T × T); −Infinity sopra la diagonale
 * @property {Float32Array} att      pesi di attention dopo la softmax (T × T); ogni riga somma a 1
 * @property {Float32Array} out      att · v (T × hs)
 *
 * @typedef {object} BlockTrace      attivazioni di un blocco transformer
 * @property {Float32Array} ln1      LayerNorm1(x) (T × C): ingresso dell'attention
 * @property {HeadTrace[]} heads     una voce per head
 * @property {Float32Array} concat   uscite delle head affiancate (T × C), head h nelle colonne h·hs…
 * @property {Float32Array} proj     proiezione con bias = uscita della multi-head attention, "saOut" (T × C)
 * @property {Float32Array} resid1   x + proj: residual stream dopo l'attention (T × C)
 * @property {Float32Array} ln2      LayerNorm2(resid1) (T × C): ingresso della feed-forward
 * @property {Float32Array} ffnPre   primo Linear, prima della ReLU (T × H4)
 * @property {Float32Array} ffnPost  dopo la ReLU (T × H4)
 * @property {Float32Array} ffnOut   secondo Linear (T × C)
 * @property {Float32Array} resid2   resid1 + ffnOut: uscita del blocco (T × C)
 *
 * @typedef {object} ForwardTrace    tutte le attivazioni di un forward (copie, non cambiano dopo)
 * @property {number[]} ids          token id in ingresso
 * @property {number} T @property {number} C @property {number} V @property {number} headSize @property {number} hidden
 * @property {Float32Array} tokEmb   righe della token embedding table (T × C)
 * @property {Float32Array} posEmb   righe della position embedding table 0…T−1 (T × C)
 * @property {Float32Array} x0       tokEmb + posEmb: inizio del residual stream (T × C)
 * @property {BlockTrace[]} blocks   una voce per blocco
 * @property {Float32Array} lnF      LayerNorm finale (T × C)
 * @property {Float32Array} logits   lm_head (T × V)
 *
 * @typedef {{id: number, token: string, prob: number}} TokenProb
 */

export class GPTEngine {
  /**
   * Crea il motore dal manifest (weights.json già letto) e dai byte di weights.f32.bin.
   * @param {object} manifest contenuto di weights.json
   * @param {ArrayBuffer|ArrayBufferView} weightsData byte del file float32 (in Node va bene un Buffer)
   * @param {{baseUrl?: string, loadJson?: (url: string) => Promise<any>, loadBuffer?: (url: string) => Promise<ArrayBuffer>}} [sources]
   *   da dove caricare gli snapshot del training (`loadSnapshot`); in Node si possono passare
   *   funzioni che leggono da disco.
   */
  static fromBuffers(manifest, weightsData, sources = {}) {
    return new GPTEngine(manifest, weightsData, sources);
  }

  constructor(manifest, weightsData, { baseUrl = null, loadJson = null, loadBuffer = null } = {}) {
    this.layout = validateManifest(manifest);
    const cfg = manifest.config;
    /** Iperparametri del modello (come GPTConfig) più head_size. */
    this.config = Object.freeze({ ...cfg, head_size: cfg.n_embd / cfg.n_head });
    /** token id → carattere. */
    this.vocab = Object.freeze([...manifest.vocab]);
    this.paramCount = manifest.total_params;
    this.manifest = manifest;
    this._stoi = new Map(this.vocab.map((ch, i) => [ch, i]));
    this._sources = { baseUrl, loadJson, loadBuffer, training: manifest.training ?? null };

    // Copia propria dei pesi: setWeights la sovrascrive, le viste restano valide.
    this.weights = new Float32Array(decodeFloat32(weightsData, this.paramCount));
    this._initialWeights = this.weights.slice();
    /** "trained" (pesi di weights.f32.bin), "snapshot" (loadSnapshot) o "custom" (setWeights). */
    this.weightsSource = "trained";
    /** Step dello snapshot caricato, null se i pesi non vengono da uno snapshot. */
    this.snapshotStep = null;

    /**
     * Viste sui pesi: nome PyTorch → {name, shape, offset, numel, data}. `data` è una vista
     * "viva" su this.weights: dopo setWeights/loadSnapshot mostra i valori nuovi.
     * @type {Map<string, {name: string, shape: number[], offset: number, numel: number, data: Float32Array}>}
     */
    this.paramViews = new Map(
      this.layout.map((p) => [p.name, { ...p, data: this.weights.subarray(p.offset, p.offset + p.numel) }]),
    );
    this._net = this._structure();
    this._ws = this._workspace();
    this._trace = null; // promessa di trace.json, caricata una volta sola
    this._snapshotCache = new Map(); // step → Promise<Float32Array>
    this._weightsVersion = 0; // invalida i loadSnapshot superati da richieste più recenti
  }

  // ---------------------------------------------------------------- tokenizer

  /**
   * Testo → token id (un carattere = un token, come CharTokenizer in gpt/data.py).
   * @throws {VocabularyError} se ci sono caratteri fuori vocabolario
   */
  encode(text) {
    const ids = [];
    const unknown = new Set();
    for (const ch of String(text)) {
      const id = this._stoi.get(ch);
      if (id === undefined) unknown.add(ch);
      else ids.push(id);
    }
    if (unknown.size > 0) throw new VocabularyError([...unknown].sort());
    return ids;
  }

  /** Token id → testo. */
  decode(ids) {
    let text = "";
    for (const id of ids) {
      this._checkId(id);
      text += this.vocab[id];
    }
    return text;
  }

  // ---------------------------------------------------------------- inferenza

  /**
   * Forward completo, come `GPTLanguageModel.forward` (senza targets).
   * @param {ArrayLike<number>} ids da 1 a block_size token id
   * @param {{trace?: boolean}} [options] trace: true restituisce anche tutte le attivazioni
   * @returns {{logits: Float32Array, T: number, V: number, trace?: ForwardTrace}} logits (T × V)
   */
  forward(ids, { trace = false } = {}) {
    this._checkSequence(ids, { allowLonger: false });
    const T = ids.length;
    const V = this.config.vocab_size;
    const recorded = this._run(ids, 0, T, trace, true);
    const logits = this._ws.logits.slice(0, T * V);
    if (!recorded) return { logits, T, V };
    recorded.logits = logits;
    return { logits, T, V, trace: recorded };
  }

  /**
   * Distribuzione del prossimo token. Se `ids` è più lungo di block_size usa solo gli ultimi
   * block_size token, come gpt/generation.py.
   * @param {ArrayLike<number>} ids almeno un token id
   * @param {{temperature?: number, topK?: number|null}} [options]
   * @returns {{probs: Float32Array, logits: Float32Array}} probs (V) dopo temperatura e top-k; logits (V) grezzi
   */
  nextTokenDistribution(ids, options = {}) {
    const sampling = samplingOptions(options, this.config.vocab_size);
    this._checkSequence(ids, { allowLonger: true });
    return this._nextToken(ids, sampling, false);
  }

  /**
   * Generazione autoregressiva token per token, come gpt/generation.py: a ogni passo il
   * contesto è tagliato agli ultimi block_size token. È un generatore: ogni `next()` calcola
   * UN token, così un'animazione può mostrare il passo e poi chiedere il successivo.
   *
   * Con lo stesso seed (e gli stessi pesi) la sequenza è sempre la stessa. I token estratti
   * non coincidono con quelli di PyTorch (generatori casuali diversi), le probabilità sì.
   *
   * @param {ArrayLike<number>} ids prompt (almeno un token id)
   * @param {number} maxNewTokens quanti token generare
   * @param {{temperature?: number, topK?: number|null, seed?: number, rng?: () => number, trace?: boolean}} [options]
   *   rng sostituisce seed; trace: true aggiunge a ogni passo il ForwardTrace del contesto usato
   * @returns {Generator<{step: number, id: number, token: string, probs: Float32Array, logits: Float32Array, contextStart: number, trace?: ForwardTrace}>}
   *   `contextStart`: indice (nella sequenza completa prompt + generati) del primo token del contesto usato
   */
  generate(ids, maxNewTokens, { temperature = 1, topK = null, seed = DEFAULT_SEED, rng = null, trace = false } = {}) {
    // Controlli subito (non al primo next()): gli errori arrivano dove si chiama generate.
    const sampling = samplingOptions({ temperature, topK }, this.config.vocab_size);
    this._checkSequence(ids, { allowLonger: true });
    if (!Number.isInteger(maxNewTokens) || maxNewTokens < 0) {
      throw new RangeError(`maxNewTokens deve essere un intero ≥ 0, ricevuto ${maxNewTokens}`);
    }
    return this._generate(Array.from(ids), maxNewTokens, sampling, rng ?? mulberry32(seed), trace);
  }

  *_generate(context, maxNewTokens, sampling, rng, trace) {
    for (let step = 0; step < maxNewTokens; step++) {
      const contextStart = Math.max(0, context.length - this.config.block_size);
      const { probs, logits, trace: recorded } = this._nextToken(context, sampling, trace);
      const id = sample(probs, rng);
      context.push(id);
      const result = { step, id, token: this.vocab[id], probs, logits, contextStart };
      if (recorded) result.trace = recorded;
      yield result;
    }
  }

  /**
   * Logit lens: "che cosa pensa il modello strato per strato". Applica LayerNorm finale +
   * lm_head al residual stream dopo l'embedding e dopo ogni blocco. L'ultimo strato coincide
   * con la previsione vera del modello.
   * @param {ArrayLike<number>} ids da 1 a block_size token id
   * @param {{topK?: number}} [options] quanti token più probabili elencare per posizione
   * @returns {{T: number, V: number, layers: {layer: number, label: string, probs: Float32Array, top: TokenProb[][]}[]}}
   *   layers[0] = dopo l'embedding, layers[l] = dopo il blocco l; probs (T × V); top[t] in ordine decrescente
   */
  logitLens(ids, { topK = 5 } = {}) {
    const { V, C, T, x0, blocks } = this.forward(ids, { trace: true }).trace;
    const k = Math.min(Math.max(1, topK | 0), V);
    const stages = [{ label: "embedding", x: x0 }, ...blocks.map((b, l) => ({ label: `blocco ${l + 1}`, x: b.resid2 }))];
    const normed = new Float32Array(T * C);
    const logits = new Float32Array(T * V);
    const layers = stages.map(({ label, x }, layer) => {
      layerNorm(normed, x, this._net.lnF.weight, this._net.lnF.bias, C, LAYER_NORM_EPS, 0, T);
      linear(logits, normed, this._net.lmHead.weight, this._net.lmHead.bias, C, V, 0, T);
      const probs = new Float32Array(T * V);
      const top = [];
      for (let t = 0; t < T; t++) {
        softmax(probs, t * V, logits, t * V, V);
        top.push(this.topTokens(probs.subarray(t * V, (t + 1) * V), k));
      }
      return { layer, label, probs, top };
    });
    return { T, V, layers };
  }

  /**
   * I k token più probabili di una distribuzione, in ordine decrescente.
   * @param {ArrayLike<number>} probs (V)
   * @returns {TokenProb[]}
   */
  topTokens(probs, k = 10) {
    const order = Array.from({ length: probs.length }, (_, i) => i);
    order.sort((a, b) => probs[b] - probs[a] || a - b);
    return order.slice(0, k).map((id) => ({ id, token: this.vocab[id], prob: probs[id] }));
  }

  // ---------------------------------------------------------------- pesi e snapshot

  /** Sostituisce tutti i pesi (stesso layout di weights.f32.bin). */
  setWeights(values) {
    this._applyWeights(values, "custom", null);
  }

  /** Pesi da un buffer float16 con il layout degli snapshot (training/snap_XXXXX.f16.bin). */
  setWeightsFromFloat16(data) {
    this._applyWeights(decodeFloat16(data, this.paramCount), "custom", null);
  }

  /** Torna ai pesi addestrati caricati all'inizio (weights.f32.bin). */
  restoreWeights() {
    this._applyWeights(this._initialWeights, "trained", null);
  }

  /** trace.json del training (caricato una volta sola). */
  loadTrace() {
    if (!this._trace) {
      const { training } = this._sources;
      if (!training) return Promise.reject(new Error("storia del training non disponibile: weights.json non ha 'training'"));
      this._trace = this._load("loadJson", training.trace).then((trace) => {
        if (trace.snapshot_dtype !== "float16") throw new Error(`snapshot in ${trace.snapshot_dtype}, atteso float16`);
        assertSameLayout(trace.params, this.layout);
        return trace;
      });
      this._trace.catch(() => (this._trace = null)); // un errore di rete non resta in cache
    }
    return this._trace;
  }

  /**
   * Carica i pesi dello snapshot del training allo step indicato (0 = inizializzazione,
   * l'ultimo coincide con il checkpoint). Gli snapshot già scaricati restano in memoria.
   * Se nel frattempo arriva un'altra richiesta (o setWeights/restoreWeights), questa non
   * applica i pesi e restituisce null: vince sempre l'ultima richiesta.
   * @param {number} step uno degli step di trace.snapshots
   * @returns {Promise<object|null>} la voce di trace.snapshots (loss, sample_text, …) o null
   */
  async loadSnapshot(step) {
    const version = ++this._weightsVersion;
    const trace = await this.loadTrace();
    const entry = trace.snapshots.find((s) => s.step === step);
    if (!entry) {
      throw new Error(`snapshot dello step ${step} non disponibile (step: ${trace.snapshots.map((s) => s.step).join(", ")})`);
    }
    if (!this._snapshotCache.has(step)) {
      const pending = this._load("loadBuffer", `${this._sources.training.dir}/${entry.file}`).then((buf) =>
        decodeFloat16(buf, this.paramCount),
      );
      pending.catch(() => this._snapshotCache.delete(step));
      this._snapshotCache.set(step, pending);
    }
    const values = await this._snapshotCache.get(step);
    if (version !== this._weightsVersion) return null;
    this._applyWeights(values, "snapshot", step, false);
    return entry;
  }

  _applyWeights(values, source, step, invalidatePending = true) {
    if (!values || values.length !== this.paramCount) {
      throw new Error(`servono ${this.paramCount} pesi, ricevuti ${values?.length}`);
    }
    if (invalidatePending) this._weightsVersion++;
    this.weights.set(values);
    this.weightsSource = source;
    this.snapshotStep = step;
  }

  _load(kind, path) {
    const loader = this._sources[kind];
    if (!loader) {
      return Promise.reject(new Error(`motore creato senza ${kind}: impossibile leggere ${path} (usa loadEngine o passa le sorgenti a fromBuffers)`));
    }
    return Promise.resolve(loader(joinUrl(this._sources.baseUrl ?? "", path)));
  }

  // ---------------------------------------------------------------- interno

  /** Viste strutturate sui pesi, nell'ordine in cui le usa il forward. */
  _structure() {
    const w = (name) => this.paramViews.get(name).data;
    const linearLayer = (prefix) => ({ weight: w(`${prefix}.weight`), bias: w(`${prefix}.bias`) });
    const { n_layer: L, n_head: H } = this.config;
    return {
      tokEmb: w("token_embedding_table.weight"),
      posEmb: w("position_embedding_table.weight"),
      blocks: Array.from({ length: L }, (_, l) => {
        const p = `blocks.${l}`;
        return {
          ln1: linearLayer(`${p}.ln1`),
          heads: Array.from({ length: H }, (_, h) => ({
            query: w(`${p}.sa.heads.${h}.query.weight`),
            key: w(`${p}.sa.heads.${h}.key.weight`),
            value: w(`${p}.sa.heads.${h}.value.weight`),
          })),
          proj: linearLayer(`${p}.sa.proj`),
          ln2: linearLayer(`${p}.ln2`),
          fc1: linearLayer(`${p}.ffwd.net.0`),
          fc2: linearLayer(`${p}.ffwd.net.2`),
        };
      }),
      lnF: linearLayer("ln_f"),
      lmHead: linearLayer("lm_head"),
    };
  }

  /**
   * Buffer di lavoro dimensionati per block_size, allocati UNA volta: in generazione il
   * forward non crea array (si usano solo le prime T righe di ogni buffer).
   */
  _workspace() {
    const { block_size: B, n_embd: C, n_head: H, vocab_size: V, head_size: hs } = this.config;
    const perHead = (size) => Array.from({ length: H }, () => new Float32Array(size));
    return {
      x: new Float32Array(B * C), // residual stream
      ln: new Float32Array(B * C),
      q: perHead(B * hs),
      k: perHead(B * hs),
      v: perHead(B * hs),
      scores: perHead(B * B),
      att: perHead(B * B),
      out: perHead(B * hs),
      concat: new Float32Array(B * C),
      proj: new Float32Array(B * C),
      hidden: new Float32Array(B * FFN_MULTIPLIER * C),
      ffn: new Float32Array(B * C),
      lnF: new Float32Array(B * C),
      logits: new Float32Array(B * V),
      sortScratch: new Float32Array(V),
    };
  }

  /**
   * Il forward vero e proprio su ids[start … start+T). I risultati restano nel workspace;
   * con `allLogits` false calcola LayerNorm finale e lm_head solo sull'ultima posizione
   * (in generazione serve solo quella). Con `wantTrace` restituisce le copie delle attivazioni.
   * @returns {ForwardTrace|null}
   */
  _run(ids, start, T, wantTrace, allLogits) {
    const { n_embd: C, vocab_size: V, head_size: hs } = this.config;
    const hidden = FFN_MULTIPLIER * C;
    const net = this._net;
    const ws = this._ws;
    const x = ws.x;
    const n = T * C;
    const copy = (buffer, size) => buffer.slice(0, size);

    // Embedding: riga del token + riga della posizione (le posizioni ripartono da 0 nel contesto).
    for (let t = 0; t < T; t++) {
      const tokOff = ids[start + t] * C;
      const posOff = t * C;
      for (let i = 0; i < C; i++) x[posOff + i] = net.tokEmb[tokOff + i] + net.posEmb[posOff + i];
    }
    let trace = null;
    if (wantTrace) {
      const tokEmb = new Float32Array(n);
      for (let t = 0; t < T; t++) {
        const tokOff = ids[start + t] * C;
        tokEmb.set(net.tokEmb.subarray(tokOff, tokOff + C), t * C);
      }
      trace = {
        ids: Array.from({ length: T }, (_, t) => ids[start + t]),
        T, C, V, headSize: hs, hidden,
        tokEmb,
        posEmb: copy(net.posEmb, n),
        x0: copy(x, n),
        blocks: [],
      };
    }

    for (const block of net.blocks) {
      const record = wantTrace ? { heads: [] } : null;

      // Comunicazione: x = x + MultiHeadAttention(LayerNorm1(x))
      layerNorm(ws.ln, x, block.ln1.weight, block.ln1.bias, C, LAYER_NORM_EPS, 0, T);
      if (record) record.ln1 = copy(ws.ln, n);
      for (let h = 0; h < block.heads.length; h++) {
        const head = block.heads[h];
        const q = ws.q[h], k = ws.k[h], v = ws.v[h], out = ws.out[h];
        linear(q, ws.ln, head.query, null, C, hs, 0, T);
        linear(k, ws.ln, head.key, null, C, hs, 0, T);
        linear(v, ws.ln, head.value, null, C, hs, 0, T);
        causalSelfAttention(q, k, v, ws.scores[h], ws.att[h], out, T, hs);
        // torch.cat(..., dim=-1): la head h occupa le colonne h·hs … h·hs + hs − 1.
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < hs; d++) ws.concat[t * C + h * hs + d] = out[t * hs + d];
        }
        if (record) {
          record.heads.push({
            q: copy(q, T * hs), k: copy(k, T * hs), v: copy(v, T * hs),
            scores: copy(ws.scores[h], T * T), att: copy(ws.att[h], T * T), out: copy(out, T * hs),
          });
        }
      }
      linear(ws.proj, ws.concat, block.proj.weight, block.proj.bias, C, C, 0, T);
      addInPlace(x, ws.proj, n);
      if (record) Object.assign(record, { concat: copy(ws.concat, n), proj: copy(ws.proj, n), resid1: copy(x, n) });

      // Elaborazione: x = x + FeedForward(LayerNorm2(x)), FeedForward = Linear → ReLU → Linear
      layerNorm(ws.ln, x, block.ln2.weight, block.ln2.bias, C, LAYER_NORM_EPS, 0, T);
      linear(ws.hidden, ws.ln, block.fc1.weight, block.fc1.bias, C, hidden, 0, T);
      if (record) Object.assign(record, { ln2: copy(ws.ln, n), ffnPre: copy(ws.hidden, T * hidden) });
      reluInPlace(ws.hidden, T * hidden);
      linear(ws.ffn, ws.hidden, block.fc2.weight, block.fc2.bias, hidden, C, 0, T);
      addInPlace(x, ws.ffn, n);
      if (record) {
        Object.assign(record, { ffnPost: copy(ws.hidden, T * hidden), ffnOut: copy(ws.ffn, n), resid2: copy(x, n) });
        trace.blocks.push(record);
      }
    }

    const first = allLogits ? 0 : T - 1;
    const rows = allLogits ? T : 1;
    layerNorm(ws.lnF, x, net.lnF.weight, net.lnF.bias, C, LAYER_NORM_EPS, first, rows);
    linear(ws.logits, ws.lnF, net.lmHead.weight, net.lmHead.bias, C, V, first, rows);
    if (trace) trace.lnF = copy(ws.lnF, n);
    return trace;
  }

  /** Distribuzione del prossimo token sugli ultimi block_size token di `ids`. */
  _nextToken(ids, sampling, wantTrace) {
    const V = this.config.vocab_size;
    const start = Math.max(0, ids.length - this.config.block_size);
    const T = ids.length - start;
    const trace = this._run(ids, start, T, wantTrace, wantTrace);
    const logits = this._ws.logits.slice((T - 1) * V, T * V);
    const probs = new Float32Array(V);
    distributionFromLogits(probs, logits, 0, V, sampling, this._ws.sortScratch);
    if (!trace) return { probs, logits };
    trace.logits = this._ws.logits.slice(0, T * V);
    return { probs, logits, trace };
  }

  _checkId(id) {
    if (!Number.isInteger(id) || id < 0 || id >= this.config.vocab_size) {
      throw new RangeError(`token id non valido: ${id} (validi da 0 a ${this.config.vocab_size - 1})`);
    }
  }

  _checkSequence(ids, { allowLonger }) {
    if (!ids || typeof ids.length !== "number") throw new TypeError("ids deve essere un array di token id");
    if (ids.length === 0) throw new RangeError("serve almeno un token");
    if (!allowLonger && ids.length > this.config.block_size) {
      throw new RangeError(`sequenza lunga ${ids.length}, il massimo è block_size=${this.config.block_size}`);
    }
    for (let i = 0; i < ids.length; i++) this._checkId(ids[i]);
  }
}

/** Unisce base e percorso relativo ("data" + "x.json" → "data/x.json"). */
function joinUrl(base, path) {
  if (!base) return path;
  return base.endsWith("/") ? base + path : `${base}/${path}`;
}

/**
 * Carica il motore dai file esportati: `${baseUrl}weights.json` e il binario che indica.
 * @param {string} [baseUrl] cartella dei dati (nel browser, relativa alla pagina)
 * @param {{loadJson?: (url: string) => Promise<any>, loadBuffer?: (url: string) => Promise<ArrayBuffer>}} [loaders]
 *   di default fetch; in Node si possono passare funzioni che leggono da disco
 * @returns {Promise<GPTEngine>}
 */
export async function loadEngine(baseUrl = "data/", { loadJson = fetchJson, loadBuffer = fetchBuffer } = {}) {
  const manifest = await loadJson(joinUrl(baseUrl, WEIGHTS_MANIFEST));
  const buffer = await loadBuffer(joinUrl(baseUrl, manifest.file));
  return GPTEngine.fromBuffers(manifest, buffer, { baseUrl, loadJson, loadBuffer });
}
