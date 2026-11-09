// Robustezza della libreria dei pesi (js/shared/weights/) contro file rotti o malevoli e contro un
// IndexedDB che fallisce:   node --test webapp/tests/*.test.mjs
// I file caricati dallo studente sono dati NON fidati: qui si provano i casi che un file scaricato da
// chissà dove, troncato o scritto apposta può presentare (intestazioni enormi, JSON rotti, offset fuori
// dal file o sovrapposti, dtype che vengono dal prototipo degli oggetti, nomi strani, file troncati) e
// un browser in cui salvare non riesce (navigazione privata, quota esaurita a metà, connessione chiusa).
// Niente DOM e niente three.js; IndexedDB è un finto minimo con le regole che contano (transazioni
// atomiche, richieste in ordine, annullamento).
import assert from "node:assert/strict";
import { test } from "node:test";

import { expectedParameters } from "../js/engine/gpt.js";
import { recognizeFiles, readBinaryFile, readJsonFile } from "../js/shared/weights/files.js";
import { decodeWeights, parseBenchmark, parseTrace, parseWeightsManifest } from "../js/shared/weights/formats.js";
import { runsFromFiles } from "../js/shared/weights/run.js";
import { parseSafetensors } from "../js/shared/weights/safetensors.js";
import { RunStore, deserializeRun, serializeRun } from "../js/shared/weights/store.js";
import { WeightsFormatError } from "../js/shared/weights/validate.js";

// ---------------------------------------------------------------- aiuti

const SMALL = { vocab_size: 5, block_size: 4, n_embd: 8, n_head: 2, n_layer: 1, dropout: 0 };
const VOCAB = ["\n", " ", "a", "b", "c"];

function synthetic(config = SMALL) {
  let offset = 0;
  const params = expectedParameters(config).map(({ name, shape }) => {
    const numel = shape.reduce((a, b) => a * b, 1);
    const entry = { name, shape, offset, numel };
    offset += numel;
    return entry;
  });
  const values = Float32Array.from({ length: offset }, (_, i) => Math.sin(i) * 0.1);
  const manifest = { format_version: 1, dtype: "float32", byte_order: "little", file: "weights.f32.bin", config, vocab: VOCAB, total_params: offset, params, training: null };
  return { manifest, values, bytes: new Uint8Array(values.buffer) };
}

/** Oggetto "tipo File" che conta le letture: un file oltre i limiti non va mai letto. */
function fileLike(path, content, { size } = {}) {
  const bytes = typeof content === "string" ? new TextEncoder().encode(content) : content;
  const reads = { count: 0 };
  return {
    name: path.split(/[\\/]/).pop(),
    size: size ?? bytes.byteLength,
    webkitRelativePath: path,
    reads,
    text: async () => (reads.count++, new TextDecoder().decode(bytes)),
    arrayBuffer: async () => (reads.count++, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)),
  };
}

function safetensors(header, data = new Uint8Array(0), { declaredLength = null, highWord = 0 } = {}) {
  const json = new TextEncoder().encode(typeof header === "string" ? header : JSON.stringify(header));
  const out = new Uint8Array(8 + json.length + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, declaredLength ?? json.length, true);
  view.setUint32(4, highWord, true);
  out.set(json, 8);
  out.set(data, 8 + json.length);
  return out;
}
const f32 = (...v) => new Uint8Array(new Float32Array(v).buffer);
const throwsFormat = (fn, pattern) => assert.throws(fn, (e) => e instanceof WeightsFormatError && pattern.test(e.message), `atteso ${pattern}`);
const rejectsFormat = (promise, pattern) => assert.rejects(promise, (e) => e instanceof WeightsFormatError && pattern.test(e.message), `atteso ${pattern}`);

// ---------------------------------------------------------------- safetensors

test("safetensors: un dtype che viene dal prototipo degli oggetti è sconosciuto, non un numero", () => {
  for (const dtype of ["toString", "constructor", "__proto__", "valueOf", "hasOwnProperty"]) {
    const header = `{"x":{"dtype":${JSON.stringify(dtype)},"shape":[2],"data_offsets":[0,8]}}`;
    throwsFormat(() => parseSafetensors(safetensors(header, f32(1, 2))), /non supportato \(dtype sconosciuto\)/);
  }
});

test("safetensors: offset, forme e intestazioni sbagliate in tutti i modi", () => {
  const one = (info) => safetensors({ x: { dtype: "F32", shape: [2], data_offsets: [0, 8], ...info } }, f32(1, 2));
  for (const offsets of [[0, 7.5], ["0", "8"], [-8, 0], [0], [0, 8, 16], [0, 2 ** 53 + 2], null, "0-8"]) {
    throwsFormat(() => parseSafetensors(one({ data_offsets: offsets })), /data_offsets/);
  }
  for (const shape of [["2"], [2.5], [[2]], [true], "2", null, [1, 1, 1, 1, 1, 1, 1, 1, 2]]) {
    throwsFormat(() => parseSafetensors(one({ shape })), /forma|dimensione/);
  }
  // l'intestazione deve essere un oggetto JSON
  for (const header of ["null", "3", '"x"', "true", "[]"]) throwsFormat(() => parseSafetensors(safetensors(header)), /deve essere un oggetto/);
  throwsFormat(() => parseSafetensors(safetensors({ x: "F32" }, f32(1))), /servono dtype, shape e data_offsets/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { shape: [1], data_offsets: [0, 4] } }, f32(1))), /dtype mancante/);
  // senza tensori va bene (file vuoto), ma niente byte in più
  assert.equal(parseSafetensors(safetensors({ __metadata__: { a: "b" } })).tensors.size, 0);
  throwsFormat(() => parseSafetensors(safetensors({}, f32(1))), /4 byte in più/);
});

test("safetensors: file troncati e lunghezze dichiarate enormi si rifiutano prima di allocare", () => {
  const header = { a: { dtype: "F32", shape: [4], data_offsets: [0, 16] } };
  const whole = safetensors(header, f32(1, 2, 3, 4));
  // troncato a metà dei dati, a metà dell'intestazione, dentro gli 8 byte della lunghezza
  throwsFormat(() => parseSafetensors(whole.subarray(0, whole.length - 6)), /escono dai dati/);
  throwsFormat(() => parseSafetensors(whole.subarray(0, 20)), /il file ne ha solo/);
  throwsFormat(() => parseSafetensors(whole.subarray(0, 7)), /almeno 8 byte/);
  // 1 TB di intestazione dichiarata in un file di 16 byte: rifiutato dal limite, nessun TextDecoder enorme
  throwsFormat(() => parseSafetensors(safetensors({}, new Uint8Array(0), { declaredLength: 0, highWord: 256 })), /limite/);
  throwsFormat(() => parseSafetensors(safetensors({}, new Uint8Array(0), { declaredLength: 16 * 1024 * 1024 + 1 })), /il limite è 16777216/);
  // una forma che dichiara 10^18 valori: errore sul limite, niente Float32Array gigante
  throwsFormat(() => parseSafetensors(safetensors({ a: { dtype: "F32", shape: [1e9, 1e9], data_offsets: [0, 4] } }, f32(1))), /limite/);
});

test("safetensors: nomi strani ma validi restano chiavi di una Map, quelli pericolosi si rifiutano", () => {
  const names = ["../../etc/passwd", "constructor", "ψ.weight", "a b\tc".replace("\t", " "), "<img src=x onerror=alert(1)>"];
  const header = Object.fromEntries(names.map((n, i) => [n, { dtype: "F32", shape: [1], data_offsets: [4 * i, 4 * i + 4] }]));
  const parsed = parseSafetensors(safetensors(header, f32(...names.map((_, i) => i))));
  assert.deepEqual([...parsed.tensors.keys()], names);
  assert.equal(parsed.tensors.get("constructor").data[0], 1);
  assert.equal(typeof {}.constructor, "function", "il prototipo degli oggetti non è stato toccato");
  throwsFormat(() => parseSafetensors(safetensors({ "a\u0007b": { dtype: "F32", shape: [1], data_offsets: [0, 4] } }, f32(1))), /caratteri di controllo/);
  throwsFormat(() => parseSafetensors(safetensors({ ["x".repeat(600)]: { dtype: "F32", shape: [1], data_offsets: [0, 4] } }, f32(1))), /lungo 600 caratteri/);
});

// ---------------------------------------------------------------- manifest, trace, benchmark

test("weights.json: percorsi del file binario che escono dalla cartella o non sono relativi", () => {
  const { manifest } = synthetic();
  for (const file of ["C:\\pesi\\w.bin", "https://example.com/w.bin", "file:///etc/passwd", "a/../../w.bin", "..", "/", "", "./", null, 7]) {
    throwsFormat(() => parseWeightsManifest({ ...manifest, file }), /percorso|file/);
  }
  assert.equal(parseWeightsManifest({ ...manifest, file: "./sub/./w.bin" }).file, "sub/w.bin");
  throwsFormat(() => parseWeightsManifest({ ...manifest, training: { dir: "../fuori", trace: "trace.json" } }), /\.\./);
});

test("weights.json: offset non interi, binario troppo lungo, troppi tensori", () => {
  const { manifest, bytes } = synthetic();
  const params = structuredClone(manifest.params);
  params[1].offset = 2.5;
  throwsFormat(() => parseWeightsManifest({ ...manifest, params }), /offset .* non valido/);
  const parsed = parseWeightsManifest(manifest);
  const longer = new Uint8Array(bytes.length + 4);
  longer.set(bytes);
  throwsFormat(() => decodeWeights(longer, parsed), /attesi .* byte, ricevuti/);
  throwsFormat(() => parseWeightsManifest(manifest, { limits: { maxTensors: 3 } }), /il massimo è 3/);
});

test("trace.json: layout diverso, snapshot troncato e nome del file che esce dalla cartella", async () => {
  const { manifest, bytes } = synthetic();
  const trace = { format_version: 1, snapshot_dtype: "float16", params: manifest.params, total_params: manifest.total_params, snapshots: [{ step: 0, file: "snap_00000.f16.bin" }] };
  const truncated = new Uint8Array(manifest.total_params * 2 - 2);
  const { runs, errors } = await runsFromFiles([
    fileLike("r/weights.json", JSON.stringify(manifest)),
    fileLike("r/weights.f32.bin", bytes),
    fileLike("r/trace.json", JSON.stringify(trace)),
    fileLike("r/snap_00000.f16.bin", truncated),
  ]);
  assert.deepEqual(errors, []);
  // lo snapshot troncato non rompe il run: l'errore arriva quando lo si chiede, e non resta in cache
  await rejectsFormat(runs[0].loadSnapshot(0), /attesi \d+ byte, ricevuti \d+/);
  await rejectsFormat(runs[0].loadSnapshot(0), /attesi/);
  throwsFormat(() => parseTrace({ ...trace, snapshots: [{ step: 0, file: "C:\\snap_00000.f16.bin" }] }), /percorso non ammesso/);
});

test("benchmark.json: format_version non è una metrica, i nomi del prototipo non diventano chiavi strane", () => {
  const bench = parseBenchmark(JSON.parse('{"format_version":1,"__proto__":5,"metrics":{"loss":1.81,"top1":0.456,"format_version":1}}'));
  assert.deepEqual(Object.keys(bench.metrics).sort(), ["loss", "top1"]);
  assert.equal(Object.getPrototypeOf(bench.metrics), Object.prototype);
});

// ---------------------------------------------------------------- file caricati

test("readJsonFile e readBinaryFile: JSON rotto con un messaggio chiaro, file troppo grandi mai letti", async () => {
  await rejectsFormat(readJsonFile(fileLike("r/run.json", '{"name": "x",')), /r\/run.json: non è un JSON valido/);
  await rejectsFormat(readJsonFile(fileLike("r/run.json", "\u0000\u0001")), /non è un JSON valido/);
  const big = fileLike("r/trace.json", "{}", { size: 65 * 1024 * 1024 });
  await rejectsFormat(readJsonFile(big), /troppo grande per un JSON/);
  const huge = fileLike("r/weights.f32.bin", new Uint8Array(4), { size: 2 * 1024 ** 3 });
  await rejectsFormat(readBinaryFile(huge), /oltre il limite/);
  assert.equal(big.reads.count + huge.reads.count, 0, "nessuna lettura dei file oltre i limiti");
});

test("recognizeFiles: oltre il limite totale non si legge niente; percorsi di Windows e con .. gestiti", async () => {
  const files = [fileLike("a/weights.json", "{}", { size: 900 * 1024 ** 2 }), fileLike("a/weights.f32.bin", new Uint8Array(4), { size: 900 * 1024 ** 2 })];
  assert.throws(() => recognizeFiles(files), (e) => e instanceof WeightsFormatError && /il limite per un caricamento/.test(e.message));
  await assert.rejects(runsFromFiles(files), /il limite per un caricamento/);
  assert.equal(files[0].reads.count + files[1].reads.count, 0);
  const { manifest, bytes } = synthetic();
  const r = recognizeFiles([fileLike("run\\sub\\weights.json", JSON.stringify(manifest)), fileLike("run\\sub\\weights.f32.bin", bytes), fileLike("run/../fuori/weights.json", "{}")]);
  assert.equal(r.runs.length, 1);
  assert.equal(r.runs[0].dir, "run/sub");
  assert.match(r.ignored[0].reason, /contiene ".."/);
});

test("runsFromFiles: un run rotto non blocca gli altri e dice perché", async () => {
  const { manifest, bytes } = synthetic();
  const short = bytes.subarray(0, bytes.length - 4);
  const { runs, errors } = await runsFromFiles([
    fileLike("buono/weights.json", JSON.stringify(manifest)),
    fileLike("buono/weights.f32.bin", bytes),
    fileLike("troncato/weights.json", JSON.stringify(manifest)),
    fileLike("troncato/weights.f32.bin", short),
    fileLike("json-rotto/weights.json", "{ format_version: 1"),
    fileLike("json-rotto/weights.f32.bin", bytes),
    fileLike("senza-bin/weights.json", JSON.stringify(manifest)),
    fileLike("st-rotto/modello.safetensors", safetensors({ x: { dtype: "F32", shape: [2], data_offsets: [0, 8] } }, f32(1))),
  ]);
  assert.deepEqual(runs.map((r) => r.name), ["buono"]);
  const byRun = Object.fromEntries(errors.map((e) => [e.run, e.message]));
  assert.match(byRun.troncato, /attesi \d+ byte, ricevuti \d+/);
  assert.match(byRun["json-rotto"], /non è un JSON valido/);
  assert.match(byRun["senza-bin"], /manca il file dei pesi/);
  assert.match(byRun["st-rotto"] ?? byRun.modello, /escono dai dati/);
});

// ---------------------------------------------------------------- IndexedDB che fallisce

/**
 * IndexedDB finto con le regole che servono qui: ogni transazione lavora su una copia e la applica solo
 * alla fine (atomica), le richieste girano in ordine e in modo asincrono, un errore annulla tutto.
 * `failures` simula i guasti: open che lancia, open che fallisce, put oltre la quota.
 */
function fakeIndexedDB(failures = {}) {
  const stores = new Map();
  const later = (fn) => setTimeout(fn, 0);
  const inRange = (range, key) => key >= range.lower && key <= range.upper;
  const dbs = [];
  const factory = {
    opens: 0,
    dbs,
    stores,
    open() {
      factory.opens++;
      if (failures.openThrows) throw Object.assign(new Error("The user denied permission to access the database."), { name: "SecurityError" });
      const req = { result: null, error: null };
      later(() => {
        if (failures.openFails) {
          req.error = Object.assign(new Error("Internal error opening backing store"), { name: "UnknownError" });
          return req.onerror?.();
        }
        const db = {
          closed: false,
          objectStoreNames: { contains: (n) => stores.has(n) },
          createObjectStore: (n) => stores.set(n, new Map()),
          close() {
            db.closed = true;
          },
          transaction(names, mode) {
            if (db.closed) throw Object.assign(new Error("The database connection is closing."), { name: "InvalidStateError" });
            const draft = new Map([...stores].map(([n, m]) => [n, new Map(m)]));
            const ops = [];
            const tx = { error: null, mode };
            const request = (run) => {
              const r = {};
              ops.push(() => {
                r.result = run();
                r.onsuccess?.();
              });
              if (ops.length === 1) later(flush);
              return r;
            };
            function flush() {
              try {
                for (const op of ops) op();
              } catch (error) {
                tx.error = error;
                tx.onerror?.();
                tx.onabort?.();
                return;
              }
              for (const [n, m] of draft) stores.set(n, m);
              tx.oncomplete?.();
            }
            tx.objectStore = (n) => {
              const m = draft.get(n);
              return {
                put: (value, key) => {
                  const copy = structuredClone(value); // come IndexedDB: un valore non clonabile lancia subito
                  return request(() => {
                    if (failures.quotaOver && (copy?.byteLength ?? 0) > failures.quotaOver) throw Object.assign(new Error("quota"), { name: "QuotaExceededError" });
                    m.set(key ?? copy.id, copy);
                  });
                },
                get: (key) => request(() => m.get(key)),
                getAll: () => request(() => [...m.values()]),
                count: (key) => request(() => (m.has(key) ? 1 : 0)),
                delete: (key) =>
                  request(() => {
                    for (const k of [...m.keys()]) if (typeof key === "object" ? inRange(key, k) : k === key) m.delete(k);
                  }),
                clear: () => request(() => m.clear()),
              };
            };
            return tx;
          },
        };
        dbs.push(db);
        req.result = db;
        req.onupgradeneeded?.();
        req.onsuccess?.();
      });
      return req;
    },
  };
  return factory;
}
const keyRange = { bound: (lower, upper) => ({ lower, upper }) };

async function uploadedRun(scale = 1) {
  const { manifest, values } = synthetic();
  const scaled = Float32Array.from(values, (v) => v * scale);
  const { runs } = await runsFromFiles([fileLike("mio/weights.json", JSON.stringify(manifest)), fileLike("mio/weights.f32.bin", new Uint8Array(scaled.buffer))]);
  return runs[0];
}

test("store: IndexedDB che lancia all'apertura o fallisce: risposte pulite, niente eccezioni", async () => {
  const run = await uploadedRun();
  for (const failures of [{ openThrows: true }, { openFails: true }]) {
    const store = new RunStore({ indexedDB: fakeIndexedDB(failures), keyRange, storage: null });
    assert.deepEqual(await store.list(), []);
    assert.equal(store.available, false, "dopo un errore di apertura lo si dice");
    const saved = await store.save(run);
    assert.equal(saved.ok, false);
    assert.match(saved.reason, /il browser non ha salvato il run/);
    assert.equal(await store.load(run.id), null);
    assert.equal(await store.remove(run.id), false);
    assert.equal(await store.has(run.id), false);
  }
});

test("store: quota esaurita a metà salvataggio → resta intera la versione di prima", async () => {
  const failures = {};
  const store = new RunStore({ indexedDB: fakeIndexedDB(failures), keyRange, storage: null });
  const first = await uploadedRun(1);
  assert.equal((await store.save(first)).ok, true);
  const second = await uploadedRun(2);
  second.id = first.id; // la stessa voce, da sostituire
  failures.quotaOver = 16; // il blob dei pesi non ci sta più: il browser annulla la transazione
  const result = await store.save(second);
  assert.deepEqual(result, { ok: false, reason: "spazio del browser esaurito: elimina qualche run salvato o libera spazio" });
  failures.quotaOver = 0;
  const back = await store.load(first.id);
  assert.ok(back, "la versione di prima si carica ancora (blob e descrizione nella stessa transazione)");
  for (const [name, t] of first.tensors) assert.deepEqual(back.tensor(name).data, t.data, name);
  assert.equal((await store.list()).length, 1);
});

test("store: connessione chiusa da un'altra scheda → la prossima operazione la riapre", async () => {
  const idb = fakeIndexedDB();
  const store = new RunStore({ indexedDB: idb, keyRange, storage: null });
  const run = await uploadedRun();
  assert.equal((await store.save(run)).ok, true);
  assert.equal(idb.opens, 1);
  idb.dbs[0].onversionchange(); // un'altra scheda aggiorna il database: questa chiude la sua connessione
  assert.equal(idb.dbs[0].closed, true);
  const list = await store.list();
  assert.equal(list.length, 1, "riaperta, l'elenco c'è ancora");
  assert.equal(idb.opens, 2);
  assert.equal(await store.remove(run.id), true);
  assert.deepEqual(await store.list(), []);
  assert.equal([...idb.stores.get("blobs").keys()].length, 0, "Dimentica toglie anche i blob");
});

test("store: blob dei pesi danneggiato → null con il motivo, mai un run con i numeri sbagliati", async () => {
  const run = await uploadedRun();
  const { record, blobs } = await serializeRun(run);
  const weights = blobs[0].data;
  assert.throws(() => deserializeRun(record, weights.slice(0, weights.byteLength - 8), async () => null), /danneggiati/);
  assert.throws(() => deserializeRun(record, weights.slice(0, weights.byteLength - 1), async () => null), /danneggiato/);
  assert.throws(() => deserializeRun({ ...record, format: 99 }, weights, async () => null), /formato sconosciuto/);
  const idb = fakeIndexedDB();
  const store = new RunStore({ indexedDB: idb, keyRange, storage: null });
  assert.equal((await store.save(run)).ok, true);
  idb.stores.get("blobs").set(`${run.id}|weights`, weights.slice(0, 16));
  assert.equal(await store.load(run.id), null);
  assert.match(store.lastError.message, /danneggiati/);
});

test("store: quota insufficiente secondo il browser → niente scrittura, si dice quanto serve", async () => {
  const idb = fakeIndexedDB();
  const store = new RunStore({ indexedDB: idb, keyRange, storage: { estimate: async () => ({ usage: 990, quota: 1000 }) } });
  const run = await uploadedRun();
  const result = await store.save(run);
  assert.equal(result.ok, false);
  assert.match(result.reason, /servono .* MB ma nel browser ne restano 0,0 MB/);
  assert.equal(idb.opens, 0, "il database non si apre nemmeno");
  // un estimate che lancia vale come "non si sa": il salvataggio si prova
  const flaky = new RunStore({ indexedDB: fakeIndexedDB(), keyRange, storage: { estimate: async () => Promise.reject(new Error("no")) } });
  assert.equal((await flaky.save(run)).ok, true);
});
