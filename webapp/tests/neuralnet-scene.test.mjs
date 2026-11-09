// Test della scena della rete neurale (js/core/neuralNet/scene.js) e della regia della camera
// (js/core/director.js) con three.js caricato da webapp/vendor, senza DOM (scena senza etichette):
//   node --test webapp/tests/*.test.mjs
// Pesi della fetta a quota (setSliceDensity), archi della cross-attention tenui di lato, lastre fantasma che
// si attenuano davanti alla sezione guardata, nomi delle fette accanto alle sole sezioni che partecipano,
// forme al singolare; meshBounds con tutte le istanze di un InstancedMesh, API pubbliche del director.
import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

// three.js sta in webapp/vendor (nel browser lo trova l'importmap): qui lo trova un hook di risoluzione.
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
const { NetworkScene, sizeText } = await import("../js/core/neuralNet/scene.js");
const { buildNetworkModel } = await import("../js/core/neuralNet/model.js");
const { DEMO_SPECS } = await import("../js/core/neuralNet/demoSpecs.js");
const { CameraDirector, instancedBounds, meshBounds } = await import("../js/core/director.js");

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} ≉ ${b}`);

/** Rete con pesi veri sulla fetta: due nn.Linear di forza diversa e un γ (tutti da tenere). */
function sliceSpec() {
  const pairs = (n, scale) => ({ from: Array.from({ length: n }, (_, i) => i % 8), to: Array.from({ length: n }, (_, i) => Math.floor(i / 8) % 8), weight: Array.from({ length: n }, (_, i) => scale * (i % 2 ? -1 : 1) * (1 + ((i * 7) % 13))) });
  return {
    title: "Pesi della fetta",
    scale: "prova",
    depth: 4,
    sections: [
      { id: "a", label: "A", role: "norm", neurons: 8, depth: true },
      { id: "b", label: "B", role: "mlp", neurons: 8, depth: true },
      { id: "c", label: "C", role: "mlp", neurons: 8, depth: true },
    ],
    links: [
      { from: "a", to: "b", pattern: "explicit", pairs: pairs(40, 1) },
      { from: "b", to: "c", pattern: "explicit", pairs: pairs(24, 0.01) },
      { from: "a", to: "c", pattern: "explicit", pairs: { from: [0, 1, 2], to: [0, 1, 2], weight: [1.1, 0.9, 1] } },
    ],
  };
}

test("setSliceDensity: per ogni collegamento i più forti, senza ricostruire; quota 1 = il disegno di sempre", () => {
  const model = buildNetworkModel(sliceSpec());
  const net = new NetworkScene(model, { labels: false });
  const list = model.links.sliceSegments;
  const g = net.sliceLinks.geometry;
  assert.equal(net.sliceDrawn, list.count);
  const before = g.attributes.position.array.slice(0, 6 * list.count);
  // il 25% di ogni collegamento (arrotondato in su), i γ tutti con una funzione
  const shown = net.setSliceDensity((link, n) => (link.index === 2 ? n : Math.ceil(0.25 * n)));
  assert.equal(shown, 10 + 6 + 3);
  assert.equal(g.drawRange.count, 2 * shown);
  const byLink = new Map();
  for (let slot = 0; slot < shown; slot++) {
    const i = net.sliceShown[slot];
    if (!byLink.has(list.link[i])) byLink.set(list.link[i], []);
    byLink.get(list.link[i]).push(Math.abs(list.weight[i]));
  }
  for (const [link, weights] of byLink) {
    const all = [];
    for (let i = 0; i < list.count; i++) if (list.link[i] === link) all.push(Math.abs(list.weight[i]));
    all.sort((x, y) => y - x);
    assert.deepEqual([...weights].sort((x, y) => y - x), all.slice(0, weights.length), `collegamento ${link}: i più forti`);
  }
  // l'ordine dei posti segue l'elenco (priorità): nessun riordino del disegno
  for (let slot = 1; slot < shown; slot++) assert.ok(net.sliceShown[slot] > net.sliceShown[slot - 1]);
  // la fetta riposiziona solo i pesi disegnati; la quota 1 torna al disegno di sempre
  net.setSlice(1);
  assert.equal(net.setSliceDensity(1), list.count);
  net.setSlice(null);
  assert.deepEqual(g.attributes.position.array.slice(0, 6 * list.count), before);
  assert.equal(net.setSliceDensity(0), 0);
  assert.equal(net.setSliceDensity(0.01), 3, "almeno uno per collegamento");
  net.dispose();
});

test("cross-attention: un oggetto suo, tenue di lato senza fetta né selezione, pieno di tre quarti", () => {
  const model = buildNetworkModel(DEMO_SPECS.unet);
  const net = new NetworkScene(model, { labels: false });
  assert.ok(net.crossArcs, "la U-Net ha archi della cross-attention");
  assert.ok(net.arcs.geometry.drawRange.count > 0, "l'attention dentro le sezioni e i residui restano nell'altro oggetto");
  const opacity = () => net.crossArcs.material.uniforms.uOpacity.value;
  const camera = new THREE.PerspectiveCamera(45, 1.5, 0.1, 2000);
  const look = (dir) => {
    camera.position.set(...dir).multiplyScalar(80);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    net.update(camera, 1200, 800, 1);
  };
  look([-0.07, 0.1, 1]);
  assert.ok(opacity() < 0.5, `di lato: ${opacity()}`);
  net.setSlice(2);
  assert.equal(opacity(), 1, "con una fetta i suoi archi si leggono");
  net.setSlice(null);
  net.setEmphasis(["text"]);
  assert.equal(opacity(), 1, "con una sezione scelta");
  net.setEmphasis(null);
  look([-0.366, 0.438, 0.821]);
  assert.equal(opacity(), 1, "di tre quarti la fascia non c'è");
  net.dispose();
});

test("lastre fantasma: attenuate davanti alla sezione guardata di tre quarti, piene di lato e se il blocco è scelto", () => {
  const model = buildNetworkModel(DEMO_SPECS.llm);
  const net = new NetworkScene(model, { labels: false });
  const r = model.layout.repeats.findIndex((x) => x.id === "moe_x58");
  const rep = model.layout.repeats[r];
  // la sezione subito dopo i fantasmi
  const after = model.layout.sections.find((b) => b.x0 > rep.x1);
  const focus = new THREE.Vector3(...after.center);
  const camera = new THREE.PerspectiveCamera(45, 1.5, 0.1, 2000);
  const look = (dir, distance) => {
    camera.position.copy(focus).addScaledVector(new THREE.Vector3(...dir).normalize(), distance);
    camera.lookAt(focus);
    camera.updateMatrixWorld();
    net.update(camera, 1200, 800, 1, null, focus);
    return net.ghostData[4 * r];
  };
  assert.ok(look([-0.6, 0.45, 0.66], 14) < 80, "di tre quarti da vicino coprono la sezione: si attenuano");
  assert.equal(look([-0.07, 0.1, 1], 14), 255, "di lato stanno accanto: piene");
  look([-0.6, 0.45, 0.66], 14);
  net.setEmphasis(model.net.repeats[r].of.map((s) => s.id));
  assert.equal(look([-0.6, 0.45, 0.66], 14), 255, "il blocco scelto: si guardano proprio loro");
  net.dispose();
});

test("nomi delle fette e confini: accanto alle sole sezioni che partecipano alla fetta", () => {
  const spec = {
    title: "t",
    scale: "s",
    depth: { count: 5, label: "chunk" },
    sections: [
      { id: "q", label: "Domanda", role: "input", neurons: 4 },
      { id: "db", label: "Chunk", role: "data", neurons: 4, depth: true },
      { id: "tok", label: "Token", role: "input", neurons: 4, depth: { count: 7, label: "token" }, slices: false },
    ],
    links: [{ from: "q", to: "db", pattern: "broadcast" }],
  };
  const model = buildNetworkModel(spec);
  const net = new NetworkScene(model, { labels: false });
  const db = model.layout.sections[model.net.byId.get("db").index];
  near(net.sliceRun.x0, db.box.min[0]);
  near(net.sliceRun.x1, db.box.max[0]);
  const box = net.box();
  near(box.max.x, Math.max(model.layout.bounds.max[0], db.box.max[0] + 2), 1e-4);
  net.dispose();
  // senza sezioni che partecipano niente nomi (e niente posto per loro)
  const flat = new NetworkScene(buildNetworkModel({ ...spec, sections: spec.sections.map((s) => ({ ...s, slices: false })) }), { labels: false });
  assert.equal(flat.sliceRun, null);
  flat.dispose();
});

test("forme al singolare: 1 neurone, 1 canale, 1 esperto", () => {
  const s = (o) => ({ depth: { kind: "none", count: 1, rows: 1, cols: 1, label: "" }, slabs: 1, shared: 0, role: "mlp", ...o });
  assert.equal(sizeText(s({ neurons: 1 })), "1 neurone");
  assert.equal(sizeText(s({ neurons: 64 })), "64 neuroni");
  assert.equal(sizeText(s({ neurons: 1, depth: { kind: "grid", count: 4, rows: 2, cols: 2, label: "griglia" } })), "1 canale · 2 × 2");
  assert.equal(sizeText(s({ role: "experts", experts: 1, neurons: 4 })), "1 esperto × 4");
});

test("meshBounds: tutte le istanze di un InstancedMesh (non quelle nascoste con scala zero), ricalcolate quando cambiano", () => {
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial(), 3);
  const m = new THREE.Matrix4();
  mesh.setMatrixAt(0, m.makeTranslation(-5, 0, 0));
  mesh.setMatrixAt(1, m.makeTranslation(5, 2, 0));
  mesh.setMatrixAt(2, m.makeScale(0, 0, 0).setPosition(100, 100, 100)); // nascosta: non allarga l'inquadratura
  mesh.instanceMatrix.needsUpdate = true;
  const group = new THREE.Group().add(mesh);
  group.position.set(0, 10, 0);
  const box = meshBounds(group);
  near(box.min.x, -5.5);
  near(box.max.x, 5.5);
  near(box.max.y, 12.5);
  assert.equal(instancedBounds(mesh), instancedBounds(mesh), "in cache finché le matrici non cambiano");
  mesh.setMatrixAt(1, m.makeTranslation(9, 0, 0));
  mesh.instanceMatrix.needsUpdate = true;
  near(meshBounds(group).max.x, 9.5);
  mesh.count = 1;
  near(meshBounds(group).max.x, -4.5, 1e-6);
});

test("director: volo in corso, finish() alla meta, intenzione dello studente", async (t) => {
  // flyTo chiede "riduci il movimento" al browser: qui un window minimo
  globalThis.window ??= { matchMedia: () => ({ matches: false }) };
  t.after(() => delete globalThis.window);
  const camera = new THREE.PerspectiveCamera(45, 1.5, 0.1, 100);
  camera.position.set(0, 0, 10);
  const frames = [];
  const host = { camera, controls: { target: new THREE.Vector3() }, scene: new THREE.Scene(), onFrame: (fn) => frames.push(fn), visibleAspect: () => 1.5 };
  const director = new CameraDirector(host);
  assert.equal(director.flying, false);
  const flight = director.flyTo(new THREE.Vector3(3, 4, 5), new THREE.Vector3(1, 1, 1), 1);
  assert.equal(director.flying, true);
  assert.equal(director.finish(), true);
  await flight;
  assert.deepEqual(camera.position.toArray(), [3, 4, 5]);
  assert.deepEqual(host.controls.target.toArray(), [1, 1, 1]);
  assert.equal(director.flying, false);
  assert.equal(director.finish(), false);
  const stamp = director.lastIntent;
  assert.equal(director.intentActive, false);
  director.noteUserIntent();
  assert.equal(director.intentActive, true);
  assert.equal(director.lastIntent, stamp + 1);
});
