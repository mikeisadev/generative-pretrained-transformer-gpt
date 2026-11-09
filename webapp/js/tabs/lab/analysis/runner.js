/**
 * Chi esegue il benchmark nel browser: un Worker (worker.js) quando il browser lo permette, altrimenti
 * la pagina stessa a pezzi da pochi millisecondi (stesso codice: protocol.js), così la scena e i cursori
 * non si bloccano in nessuno dei due casi. Un lavoro alla volta: uno nuovo annulla il precedente;
 * `cancel()` termina il worker (riparte pulito al lavoro dopo) o ferma i pezzi.
 *
 *   const runner = new BenchmarkRunner();
 *   const out = await runner.run({ manifest, weights, text, greedy: { prompt, tokens: 200 }, onProgress });
 *   // out = { score, greedy, timing, where: "worker" | "pagina" }; annullato → CancelledError
 */
import { GPTEngine } from "../../../engine/gpt.js";
import { ExcerptScore, GreedyRun } from "./protocol.js";

const SLICE_MS = 8; // un pezzo nella pagina: abbastanza corto da non far perdere un frame

export class CancelledError extends Error {
  constructor() {
    super("calcolo annullato");
    this.name = "CancelledError";
  }
}

export class BenchmarkRunner {
  /**
   * @param {{workerUrl?: URL|string|null, createWorker?: (url) => Worker}} [o] `workerUrl` null = sempre nella pagina
   */
  constructor({ workerUrl = new URL("./worker.js", import.meta.url), createWorker = (url) => new Worker(url, { type: "module" }) } = {}) {
    this.workerUrl = workerUrl;
    this.createWorker = createWorker;
    this.worker = null;
    this.workerBroken = !workerUrl || typeof Worker !== "function";
    this.job = 0;
    this.active = null; // { job, reject, cleanup }
  }

  get busy() {
    return Boolean(this.active);
  }

  /**
   * @param {{manifest: object, weights: Float32Array, text: string, greedy?: {prompt: string, tokens: number}|null,
   *   onProgress?: (p: {phase: string, done: number, total: number}) => void}} o `weights` viene ceduto al worker
   *   (dopo la chiamata non va più usato)
   */
  run({ manifest, weights: given, text, greedy = null, onProgress = () => {} }) {
    this.cancel();
    const job = ++this.job;
    // al worker si cede il buffer intero: deve contenere esattamente questi pesi
    const weights = given.byteOffset === 0 && given.buffer.byteLength === given.byteLength ? given : given.slice();
    return new Promise((resolve, reject) => {
      const finish = (fn) => (value) => {
        if (this.active?.job !== job) return;
        this.active.cleanup?.();
        this.active = null;
        fn(value);
      };
      this.active = { job, reject: finish(reject), cleanup: null, where: null };
      const done = finish(resolve);
      const fail = finish(reject);
      if (!this.workerBroken) {
        try {
          this._inWorker({ job, manifest, weights, text, greedy, onProgress, done, fail });
          return;
        } catch {
          this.workerBroken = true; // es. pagina aperta da file://: si lavora nella pagina
        }
      }
      this._inPage({ job, manifest, weights, text, greedy, onProgress, done, fail });
    });
  }

  _inWorker({ job, manifest, weights, text, greedy, onProgress, done, fail }) {
    this.worker ??= this.createWorker(this.workerUrl);
    const worker = this.worker;
    this.active.where = "worker";
    let answered = false;
    const onMessage = ({ data }) => {
      if (data?.job !== job) return;
      answered = true;
      if (data.type === "progress") onProgress(data);
      else if (data.type === "done") done({ score: data.score, greedy: data.greedy, timing: data.timing, where: "worker" });
      else if (data.type === "error") fail(new Error(data.message));
    };
    const onError = (event) => {
      event.preventDefault?.();
      // il worker non è nemmeno partito (moduli nei worker non supportati, file non trovato): nella pagina
      if (!answered) {
        this._dropWorker();
        this.workerBroken = true;
        this.active.cleanup = null;
        this.active.where = null;
        this._inPage({ job, manifest, weights: weightsCopy, text, greedy, onProgress, done, fail });
      } else fail(new Error(event.message || "errore nel worker del benchmark"));
    };
    // una copia resta qui per il ripiego nella pagina; l'originale passa al worker senza copiarlo
    const weightsCopy = weights.slice();
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    this.active.cleanup = () => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
    };
    worker.postMessage({ type: "run", job, manifest, weights: weights.buffer, text, greedy }, [weights.buffer]);
  }

  _inPage({ job, manifest, weights, text, greedy, onProgress, done, fail }) {
    this.active.where = "pagina";
    let engine;
    let score;
    try {
      engine = GPTEngine.fromBuffers(manifest, weights.buffer); // il motore si fa una copia sua
      score = new ExcerptScore(engine.encode(text), engine.config);
    } catch (error) {
      fail(error);
      return;
    }
    let g = null;
    let timer = 0;
    const timing = { scoreMs: 0, greedyMs: 0 };
    const slice = () => {
      timer = 0;
      if (this.active?.job !== job) return;
      const t0 = performance.now();
      try {
        while (performance.now() - t0 < SLICE_MS) {
          if (!score.finished) {
            score.next(engine);
            if (score.finished) timing.scoreMs += performance.now() - t0;
            continue;
          }
          if (greedy && !g) g = new GreedyRun(engine.encode(greedy.prompt), greedy.tokens);
          if (g && !g.finished) {
            const t1 = performance.now();
            g.next(engine);
            timing.greedyMs += performance.now() - t1;
            continue;
          }
          done({ score: score.result(), greedy: g ? engine.decode(g.ids()) : null, timing, where: "pagina" });
          return;
        }
      } catch (error) {
        fail(error);
        return;
      }
      if (!score.finished) timing.scoreMs += performance.now() - t0;
      onProgress(g ? { phase: "greedy", done: g.done, total: g.total } : { phase: "score", done: score.done, total: score.total });
      timer = setTimeout(slice, 0);
    };
    this.active.cleanup = () => clearTimeout(timer);
    timer = setTimeout(slice, 0);
  }

  /** Ferma il lavoro in corso (la Promise di run si rifiuta con CancelledError). */
  cancel() {
    if (!this.active) return;
    const { reject, where } = this.active;
    // il worker è a metà di un calcolo sincrono: l'unico modo di fermarlo subito è terminarlo
    if (where === "worker") this._dropWorker();
    reject(new CancelledError());
  }

  _dropWorker() {
    this.worker?.terminate();
    this.worker = null;
  }

  dispose() {
    this.cancel();
    this._dropWorker();
  }
}
