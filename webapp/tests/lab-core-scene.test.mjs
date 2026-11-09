// Test della scena del Laboratorio pesi (js/tabs/lab/scene/labScene.js) con three.js caricato da
// webapp/vendor e un canvas finto: costruzione del GPT del corso (una NumericMatrix per parametro, i 16
// buffer tril), API per la fase 2 (showValues, setWeights, highlightCells, layoutOf, boxOf), scelta col
// puntatore, livelli del LOD, pulizia completa.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const VENDOR = pathToFileURL(fileURLToPath(new URL("../vendor/three/", import.meta.url))).href;
register(
  "data:text/javascript," +
    encodeURIComponent(`export async function resolve(specifier, context, next) {
      if (specifier === "three") return { url: ${JSON.stringify(`${VENDOR}three.module.min.js`)}, shortCircuit: true };
      if (specifier.startsWith("three/addons/")) return { url: ${JSON.stringify(`${VENDOR}addons/`)} + specifier.slice(13), shortCircuit: true };
      return next(specifier, context);
    }`),
);
const THREE = await import("three");
const { LabScene } = await import("../js/tabs/lab/scene/labScene.js");
const { NumbersGroup } = await import("../js/core/matrixNumbers.js");
const { CELL: VISUALS_CELL } = await import("../js/core/visuals.js");
const { CELL, layoutModel } = await import("../js/tabs/lab/scene/layout.js");
const { buildModuleTree } = await import("../js/shared/weights/moduleTree.js");
const { loadBuiltinRun } = await import("../js/shared/weights/run.js");

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const loadJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const loadBuffer = async (file) => {
  const b = await readFile(file);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const spec = await loadJson(`${DATA_DIR}model.json`);
const run = await loadBuiltinRun({ baseUrl: DATA_DIR, loadJson, loadBuffer });

/** Canvas finto: misura il testo (0,6 em per carattere) e conta i testi scritti. */
function fakeCanvas() {
  const canvas = { width: 0, height: 0, texts: [] };
  let font = "10px sans-serif";
  const ctx = {
    get font() {
      return font;
    },
    set font(v) {
      font = v;
    },
    fillStyle: "#000",
    strokeStyle: "#000",
    lineWidth: 1,
    textBaseline: "alphabetic",
    measureText: (t) => ({ width: String(t).length * 0.6 * Number(/(\d+(?:\.\d+)?)px/.exec(font)?.[1] ?? 10) }),
    fillText: (t) => canvas.texts.push(String(t)),
    save() {},
    restore() {},
    beginPath() {},
    moveTo() {},
    arcTo() {},
    closePath() {},
    fill() {},
    stroke() {},
    clearRect() {},
    fillRect() {},
  };
  canvas.getContext = () => ctx;
  return canvas;
}

const canvases = [];
const createCanvas = () => {
  const c = fakeCanvas();
  canvases.push(c);
  return c;
};

function buildScene() {
  const tree = buildModuleTree(run.params, { snippets: spec.snippets, config: run.config });
  const layout = layoutModel(tree, { config: run.config, snippets: spec.snippets });
  const scene = new LabScene({ createCanvas, numbers: new NumbersGroup({ createCanvas }) });
  scene.build({ run, layout, view: { tensors: run.tensors } });
  return { scene, layout };
}

test("la cella del Laboratorio è quella di core/visuals.js", () => {
  assert.equal(CELL, VISUALS_CELL);
});

test("costruzione: una NumericMatrix per parametro (94) più i 16 buffer tril, etichette nell'atlante", () => {
  const { scene, layout } = buildScene();
  const s = scene.stats;
  assert.equal(s.matrices, 94 + 16);
  assert.equal(s.cells, 209729 + 16 * 32 * 32);
  assert.deepEqual(s.limited, []);
  const key = scene.matrixOf("blocks.0.sa.heads.0.key.weight");
  assert.deepEqual([key.rows, key.cols], [16, 64]);
  assert.equal(key.data, run.tensor("blocks.0.sa.heads.0.key.weight").data, "i numeri sono quelli veri (stesso Float32Array)");
  const tril = scene.matrixOf("blocks.0.sa.heads.0.tril");
  assert.equal(tril.data[0 * 32 + 1], 0);
  assert.equal(tril.data[5 * 32 + 3], 1);
  // un vettore in colonna accanto a W: celle stirate in x, una per peso
  const bias = scene.matrixOf("lm_head.bias");
  assert.deepEqual([bias.rows, bias.cols], [65, 1]);
  assert.ok(bias.mesh.scale.x > bias.mesh.scale.y);
  const texts = canvases.flatMap((c) => c.texts);
  assert.ok(texts.includes("self.key = nn.Linear(config.n_embd, self.head_size, bias=False)"));
  assert.ok(texts.includes("tril (32, 32) · buffer, non è un peso"));
  assert.ok(texts.includes("GPTLanguageModel · GPT del corso"));
  // la posizione della matrice nel mondo è quella della disposizione
  scene.root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(key.mesh);
  const r = layout.nodes.get("blocks.0.sa.heads.0.key.weight").matrix;
  assert.ok(Math.abs(box.min.x - r.x0) < 1e-4 && Math.abs(box.max.y - r.y1) < 1e-4);
  scene.dispose();
});

test("API della fase 2: setWeights, showValues (differenza e ritorno), highlightCells, layoutOf, boxOf", async () => {
  const { scene } = buildScene();
  const name = "blocks.1.ffwd.net.0.weight";
  const m = scene.matrixOf(name);
  const range = m.range;
  const snap = await run.loadSnapshot(0);
  scene.setWeights(snap.tensors);
  assert.equal(m.data, snap.tensors.get(name).data);
  assert.equal(m.range, range, "la scala resta quella dei pesi finali: si vede quanto crescono");
  const diff = new Float32Array(m.rows * m.cols).fill(0.01);
  diff[3] = -0.5;
  const r = scene.showValues(new Map([[name, diff], ["non.esiste", diff]]), { colormap: "divergente", label: "A − B" });
  assert.deepEqual(r, { shown: [name], missing: ["non.esiste"], label: "A − B" });
  assert.equal(m.data, diff);
  assert.equal(m.range, 0.5, "scala propria e simmetrica dei valori");
  scene.setWeights(run.tensors); // un altro step mentre la differenza è in scena: la differenza resta
  assert.equal(m.data, diff);
  scene.showValues(null);
  assert.equal(m.data, run.tensor(name).data);
  assert.equal(m.range, range);
  assert.equal(scene.valuesLabel, "");
  assert.equal(scene.highlightCells(name, { rows: [3] }), true);
  const veil = m.mesh.children.find((c) => c.material?.map?.isDataTexture && c !== m.mesh.children[0]);
  assert.ok(veil?.visible);
  const alpha = (row, col) => veil.material.map.image.data[((m.rows - 1 - row) * m.cols + col) * 4 + 3];
  assert.equal(alpha(3, 10), 0);
  assert.ok(alpha(4, 10) > 100);
  scene.highlightCells(name, [{ i: 7, j: 2 }]);
  assert.equal(alpha(7, 2), 0);
  assert.ok(alpha(3, 10) > 100);
  scene.highlightCells(name, null);
  assert.equal(veil.visible, false);
  const where = scene.layoutOf(name);
  assert.deepEqual([where.rows, where.cols], [256, 64]);
  const c = where.cellCenter(0, 0);
  assert.ok(Math.abs(c.x - (where.rect.x0 + CELL / 2)) < 1e-9 && Math.abs(c.y - (where.rect.y1 - CELL / 2)) < 1e-9);
  assert.equal(scene.layoutOf("blocks.1.ffwd"), null, "un modulo non è una matrice");
  assert.ok(scene.boxOf("blocks.1.ffwd").containsBox(scene.boxOf(name)));
  scene.dispose();
});

test("scelta col puntatore: matrice con la cella, contenitore più profondo, livelli del LOD", () => {
  const { scene, layout } = buildScene();
  const target = layout.nodes.get("blocks.2.sa.heads.1.query.weight").matrix;
  const ray = new THREE.Ray(new THREE.Vector3(target.x0 + 5.5 * CELL, target.y1 - 2.5 * CELL, 10), new THREE.Vector3(0, 0, -1));
  const raycaster = new THREE.Raycaster(ray.origin, ray.direction);
  scene.root.updateMatrixWorld(true);
  const hits = raycaster.intersectObjects(scene.pickables(), false);
  const hit = scene.pickFrom(ray, hits);
  assert.equal(hit.path, "blocks.2.sa.heads.1.query.weight");
  assert.deepEqual([hit.cell.row, hit.cell.col], [2, 5]);
  const sa = layout.nodes.get("blocks.2.sa");
  const ray2 = new THREE.Ray(new THREE.Vector3(sa.x0 + 0.05, sa.y0 + 0.05, 10), new THREE.Vector3(0, 0, -1));
  assert.equal(scene.pickFrom(ray2, []).path, "blocks.2.sa");
  scene.setLevel("macro");
  assert.equal(scene.cells.visible, false, "da lontano le matrici non si disegnano");
  assert.deepEqual(scene.pickables(), []);
  const blk = layout.nodes.get("blocks.3");
  assert.equal(scene.pickFrom(new THREE.Ray(new THREE.Vector3(blk.cx, blk.cy, 10), new THREE.Vector3(0, 0, -1)), []).path, "blocks.3");
  scene.setLevel("blocks");
  assert.equal(scene.pickFrom(ray2, []).path, "blocks.2.sa");
  scene.setLevel("detail");
  scene.setSelected("blocks.2.sa.heads.1.query.weight");
  assert.equal(scene.root.children.find((o) => o.isLineLoop)?.visible, true);
  scene.dispose();
});

test("pulizia: dispose libera matrici, etichette, contenitori e lascia la scena vuota", () => {
  const { scene } = buildScene();
  const parent = new THREE.Scene();
  parent.add(scene.root);
  let disposed = 0;
  scene.root.traverse((o) => {
    if (o.geometry) o.geometry.addEventListener("dispose", () => disposed++);
  });
  scene.dispose();
  assert.equal(scene.root.parent, null);
  assert.equal(scene.matrices.size, 0);
  assert.equal(scene.cells.children.length, 0);
  for (const g of Object.values(scene.levels)) assert.equal(g.children.length, 0);
  assert.ok(disposed > 200, `geometrie liberate: ${disposed}`);
});
