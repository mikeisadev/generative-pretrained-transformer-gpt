/**
 * Worker del benchmark nel browser: esegue il protocollo "finestre-contigue-v1" (protocol.js) con un
 * motore JS suo, così la pagina resta libera (scena, cursori, scorrimento) mentre si calcolano le
 * 126 finestre dell'estratto e i 200 token greedy. Un lavoro alla volta; per annullare, la pagina
 * termina il worker (runner.js).
 *
 * Messaggi:
 *   ← { type: "run", job, manifest, weights: ArrayBuffer, text, greedy: { prompt, tokens } | null }
 *   → { type: "progress", job, phase: "score" | "greedy", done, total }
 *   → { type: "done", job, score, greedy: string | null, timing: { scoreMs, greedyMs } }
 *   → { type: "error", job, message }
 */
import { GPTEngine } from "../../../engine/gpt.js";
import { ExcerptScore, GreedyRun } from "./protocol.js";

const PROGRESS_MS = 60; // non serve un messaggio per finestra: basta che la barra si muova

self.addEventListener("message", (event) => {
  const msg = event.data;
  if (msg?.type !== "run") return;
  try {
    self.postMessage(run(msg));
  } catch (error) {
    self.postMessage({ type: "error", job: msg.job, message: error?.message ?? String(error) });
  }
});

function run({ job, manifest, weights, text, greedy }) {
  const engine = GPTEngine.fromBuffers(manifest, weights);
  const ids = engine.encode(text);
  const score = new ExcerptScore(ids, engine.config);
  let last = 0;
  const progress = (phase, done, total) => {
    const now = performance.now();
    if (now - last < PROGRESS_MS && done < total) return;
    last = now;
    self.postMessage({ type: "progress", job, phase, done, total });
  };
  const t0 = performance.now();
  while (score.next(engine)) progress("score", score.done, score.total);
  const scoreMs = performance.now() - t0;
  let text2 = null;
  let greedyMs = 0;
  if (greedy) {
    const g = new GreedyRun(engine.encode(greedy.prompt), greedy.tokens);
    const t1 = performance.now();
    while (g.next(engine)) progress("greedy", g.done, g.total);
    greedyMs = performance.now() - t1;
    text2 = engine.decode(g.ids());
  }
  return { type: "done", job, score: score.result(), greedy: text2, timing: { scoreMs, greedyMs } };
}
