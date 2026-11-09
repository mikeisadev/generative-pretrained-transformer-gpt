// Chi esegue il benchmark nel browser (js/tabs/lab/analysis/runner.js): nel Worker o nella pagina a pezzi, un
// lavoro alla volta, "Annulla", il ripiego nella pagina quando il Worker non parte, e niente listener o worker
// lasciati indietro. In Node non c'è Worker: si usa un finto worker (createWorker) con le regole che contano
// (postMessage con trasferimento, eventi message ed error, terminate).
//   node --test webapp/tests/lab-runner.test.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { GPTEngine } from "../js/engine/gpt.js";
import { scoreText } from "../js/tabs/lab/analysis/protocol.js";
import { BenchmarkRunner, CancelledError } from "../js/tabs/lab/analysis/runner.js";

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const manifest = JSON.parse(await readFile(`${DATA_DIR}weights.json`, "utf8"));
const bytes = await readFile(`${DATA_DIR}weights.f32.bin`);
const weights = () => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const TEXT = "ROMEO:\nBut, soft! what light through yonder window breaks?\n".repeat(3); // 3 finestre da 64
const engine = GPTEngine.fromBuffers(manifest, weights().buffer);
const expected = scoreText(engine, engine.encode(TEXT));

/** Il worker finto: risponde come worker.js (stesso calcolo, nella stessa pagina) e tiene il conto dei listener. */
class FakeWorker {
  constructor({ failToStart = false, cloneError = false, silent = false } = {}) {
    Object.assign(this, { failToStart, cloneError, silent, terminated: false, listeners: { message: new Set(), error: new Set() }, posted: [] });
  }
  addEventListener(type, fn) {
    this.listeners[type].add(fn);
  }
  removeEventListener(type, fn) {
    this.listeners[type].delete(fn);
  }
  get listenerCount() {
    return this.listeners.message.size + this.listeners.error.size;
  }
  emit(type, event) {
    for (const fn of [...this.listeners[type]]) fn(event);
  }
  postMessage(msg, transfer) {
    if (this.cloneError) throw Object.assign(new Error("could not be cloned"), { name: "DataCloneError" });
    assert.ok(transfer?.[0] === msg.weights, "i pesi passano al worker senza copia (trasferiti)");
    this.posted.push(msg);
    if (this.silent) return;
    setTimeout(() => {
      if (this.terminated) return;
      if (this.failToStart) return this.emit("error", { message: "moduli nei worker non supportati", preventDefault() {} });
      const e = GPTEngine.fromBuffers(msg.manifest, msg.weights);
      this.emit("message", { data: { type: "progress", job: msg.job, phase: "score", done: 1, total: 3 } });
      this.emit("message", { data: { type: "done", job: msg.job, score: scoreText(e, e.encode(msg.text)), greedy: null, timing: { scoreMs: 1, greedyMs: 0 } } });
    }, 0);
  }
  terminate() {
    this.terminated = true;
  }
}

/**
 * Un runner con worker finti: ogni worker creato è il prossimo di `queue` (o uno nuovo con `options`) e finisce in
 * `made`. In Node `Worker` non esiste: lo si dichiara, altrimenti il runner lavorerebbe sempre nella pagina.
 */
function runnerWith(options = {}, queue = []) {
  const made = [];
  globalThis.Worker ??= class {};
  const createWorker = () => {
    const worker = queue.shift() ?? new FakeWorker(options);
    made.push(worker);
    return worker;
  };
  return { runner: new BenchmarkRunner({ workerUrl: "worker.js", createWorker }), made };
}

test("runner: nella pagina a pezzi dà gli stessi numeri del protocollo tutto in una volta", async () => {
  const runner = new BenchmarkRunner({ workerUrl: null });
  const out = await runner.run({ manifest, weights: weights(), text: TEXT });
  assert.equal(out.where, "pagina");
  assert.equal(out.score.loss, expected.loss);
  assert.equal(out.score.top1, expected.top1);
  assert.equal(runner.busy, false);
});

test("runner: nel worker i listener si tolgono a lavoro finito; un lavoro nuovo annulla il precedente", async () => {
  const { runner, made } = runnerWith();
  const out = await runner.run({ manifest, weights: weights(), text: TEXT });
  assert.equal(out.where, "worker");
  assert.equal(out.score.loss, expected.loss);
  assert.equal(made[0].listenerCount, 0, "nessun listener resta sul worker");
  // due lavori di fila: il primo (un worker che non risponde) si rifiuta con CancelledError e il suo worker, a
  // metà di un calcolo sincrono, si termina; il secondo va in un worker nuovo
  const twice = runnerWith({}, [new FakeWorker({ silent: true }), new FakeWorker()]);
  const first = twice.runner.run({ manifest, weights: weights(), text: TEXT });
  const second = twice.runner.run({ manifest, weights: weights(), text: TEXT });
  await assert.rejects(first, CancelledError);
  assert.equal(twice.made[0].terminated, true);
  assert.equal((await second).score.loss, expected.loss);
  assert.equal(twice.made.length, 2);
  assert.equal(twice.made[1].listenerCount, 0);
  twice.runner.dispose();
  assert.equal(twice.made[1].terminated, true, "dispose chiude il worker");
});

test("runner: «Annulla» rifiuta con CancelledError, sia nel worker sia nella pagina, e il runner riparte pulito", async () => {
  const { runner, made } = runnerWith({ silent: true });
  const job = runner.run({ manifest, weights: weights(), text: TEXT });
  runner.cancel();
  await assert.rejects(job, CancelledError);
  assert.equal(made[0].terminated, true);
  assert.equal(made[0].listenerCount, 0);
  const page = new BenchmarkRunner({ workerUrl: null });
  const inPage = page.run({ manifest, weights: weights(), text: TEXT });
  page.cancel();
  await assert.rejects(inPage, CancelledError);
  assert.equal((await page.run({ manifest, weights: weights(), text: TEXT })).score.loss, expected.loss);
});

test("runner: un worker che non parte o un postMessage rifiutato → il calcolo si fa nella pagina, senza listener orfani", async () => {
  const broken = runnerWith({ failToStart: true });
  const out = await broken.runner.run({ manifest, weights: weights(), text: TEXT });
  assert.equal(out.where, "pagina");
  assert.equal(out.score.loss, expected.loss);
  assert.equal(broken.made[0].listenerCount, 0);
  assert.equal(broken.runner.workerBroken, true, "le volte dopo si va subito nella pagina");
  // prima: con un postMessage che lancia, i due listener restavano sul worker (e il worker restava aperto)
  const clone = runnerWith({ cloneError: true });
  const out2 = await clone.runner.run({ manifest, weights: weights(), text: TEXT });
  assert.equal(out2.where, "pagina");
  assert.equal(out2.score.loss, expected.loss);
  assert.equal(clone.made[0].listenerCount, 0, "nessun listener resta sul worker");
  assert.equal(clone.made[0].terminated, true, "il worker inutile si chiude");
  assert.equal(clone.runner.worker, null);
});
