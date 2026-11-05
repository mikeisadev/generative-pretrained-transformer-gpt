// Ponte per tests/test_js_engine.py: esegue il motore JS e stampa le attivazioni in JSON,
// così pytest le confronta con PyTorch sullo stesso checkpoint.
//
// Uso: node webapp/tests/engine_fixture.mjs < richiesta.json > risposta.json
// Richiesta: {
//   "dataDir": "/percorso/webapp/data",
//   "snapshot": null | step,                // pesi di training/snap_XXXXX.f16.bin invece di quelli finali
//   "cases": [{ "text": "...", "temperature": 1, "topK": null }]
// }
// Risposta: { config, weightsSource, snapshotStep, results: [{ ids, next: {probs, logits},
//   forward?: {logits, trace}, lens? }] }; forward e lens solo se il testo sta in block_size.
// Nel JSON gli -Infinity (punteggi mascherati dell'attention) diventano null.
import { readFile } from "node:fs/promises";
import path from "node:path";

import { loadEngine } from "../js/engine/gpt.js";

const readStdin = async () => {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
};

const fsLoaders = {
  loadJson: async (file) => JSON.parse(await readFile(file, "utf8")),
  loadBuffer: async (file) => {
    const bytes = await readFile(file);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  },
};

// Float32Array (e strutture che le contengono) → valori JSON.
function plain(value) {
  if (value instanceof Float32Array) return Array.from(value, (v) => (Number.isFinite(v) ? v : null));
  if (Array.isArray(value)) return value.map(plain);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, plain(v)]));
  return value;
}

async function main() {
  const request = JSON.parse(await readStdin());
  const engine = await loadEngine(path.join(request.dataDir, path.sep), fsLoaders);
  if (request.snapshot !== null && request.snapshot !== undefined) {
    const entry = await engine.loadSnapshot(request.snapshot);
    if (!entry) throw new Error("snapshot non applicato");
  }
  const results = request.cases.map(({ text, temperature = 1, topK = null }) => {
    const ids = engine.encode(text);
    const result = { ids, next: plain(engine.nextTokenDistribution(ids, { temperature, topK })) };
    if (ids.length <= engine.config.block_size) {
      const { logits, trace } = engine.forward(ids, { trace: true });
      result.forward = { logits: plain(logits), trace: plain(trace) };
      result.lens = engine.logitLens(ids).layers.map((layer) => plain(layer.probs));
    }
    return result;
  });
  const output = { config: engine.config, weightsSource: engine.weightsSource, snapshotStep: engine.snapshotStep, results };
  process.stdout.write(JSON.stringify(output));
}

main().catch((error) => {
  console.error(error?.stack ?? String(error));
  process.exit(1);
});
