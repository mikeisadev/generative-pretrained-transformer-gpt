// Test del LOD del nucleo (js/core/lod.js): node --test webapp/tests/lod.test.mjs
// Niente three.js: un host finto (distanza della camera e ciclo dei frame) e gruppi finti con materiali.
// Richiesta 41 del round E: a riposo update() esce subito, un nuovo distanceScale vale dal primo movimento
// della camera, le soglie cambiate da un tab (levels[i].maxDistance) valgono subito come prima.
import assert from "node:assert/strict";
import { test } from "node:test";

import { LODController } from "../js/core/lod.js";

/** Un gruppo finto: traverse sui materiali, come un THREE.Group. */
function group(name) {
  const material = { opacity: 1, transparent: false, userData: {}, needsUpdate: false };
  return { name, visible: true, userData: {}, traverse(fn) { fn({ material }); }, material_: material };
}

function setup({ distance = 10, scale = () => 1 } = {}) {
  const frames = new Set();
  const host = { distance, zoomDistance: () => host.distance, onFrame: (fn) => (frames.add(fn), () => frames.delete(fn)) };
  const levels = [
    { id: "detail", label: "Dettaglio", maxDistance: 45, group: group("detail") },
    { id: "blocks", label: "Blocchi", maxDistance: 120, group: group("blocks") },
    { id: "macro", label: "Panoramica", maxDistance: Infinity, group: group("macro") },
  ];
  const changes = [];
  let scaleCalls = 0;
  const lod = new LODController(host, { levels, onChange: (id) => changes.push(id), distanceScale: () => (scaleCalls++, scale()) });
  const tick = (n = 1, dt = 0.05) => {
    for (let i = 0; i < n; i++) for (const fn of frames) fn(dt);
  };
  return { host, lod, levels, changes, tick, scaleCalls: () => scaleCalls };
}

test("LOD: il livello segue la distanza, con le sfumature, e poi a riposo non lavora più", () => {
  const { host, lod, levels, changes, tick, scaleCalls } = setup({ distance: 10 });
  assert.equal(lod.current, "detail");
  tick(); // il primo frame applica le opacità iniziali
  assert.equal(levels[0].group.visible, true);
  assert.equal(levels[1].group.visible, false);
  host.distance = 80;
  tick();
  assert.equal(lod.current, "blocks");
  assert.deepEqual(changes, ["detail", "blocks"]);
  tick(10); // 3,5 alpha al secondo: in 0,5 s le sfumature sono finite
  assert.equal(levels[0].group.visible, false);
  assert.equal(levels[1].group.material_.opacity, 1);
  // A riposo: nessuna chiamata a distanceScale (può leggere il layout), nessun materiale toccato.
  const calls = scaleCalls();
  const mat = levels[1].group.material_;
  mat.opacity = 0.5; // se update toccasse i gruppi, rimetterebbe 1
  tick(100);
  assert.equal(scaleCalls(), calls, "distanceScale non chiamata a camera ferma");
  assert.equal(mat.opacity, 0.5, "nessun materiale riscritto a riposo");
  // Uno spostamento laterale (pan) non cambia la distanza: niente da ricalcolare.
  tick(5);
  assert.equal(scaleCalls(), calls);
});

test("LOD: un nuovo distanceScale vale dal primo movimento della camera, non a camera ferma", () => {
  let k = 1;
  const { host, lod, tick } = setup({ distance: 40, scale: () => k });
  tick(10);
  assert.equal(lod.current, "detail");
  k = 0.5; // es. il codice si riduce: la parte visibile si allarga
  tick(10);
  assert.equal(lod.current, "detail", "con la camera ferma il livello non cambia sotto gli occhi");
  host.distance = 40.5; // il primo movimento della rotellina
  tick();
  assert.equal(lod.current, "blocks", "40,5 / 0,5 = 81: Blocchi");
});

test("LOD: soglie cambiate da un tab, setMode e setEnabled valgono subito", () => {
  const { host, lod, levels, tick } = setup({ distance: 60 });
  tick(10);
  assert.equal(lod.current, "blocks");
  levels[0].maxDistance = 80; // es. il Transformer entra in una vista con altre soglie
  tick();
  assert.equal(lod.current, "detail", "le soglie nuove valgono anche a camera ferma");
  lod.setMode("macro");
  assert.equal(lod.current, "macro");
  tick(20);
  assert.equal(levels[2].group.visible, true);
  lod.setMode("auto");
  tick();
  assert.equal(lod.current, "detail", "tornando ad Auto il livello si ricalcola dalla distanza");
  lod.setEnabled(false);
  host.distance = 500;
  tick(5);
  assert.equal(lod.current, "detail", "spento non segue la camera");
  lod.setEnabled(true);
  assert.equal(lod.current, "macro", "riacceso riparte dal livello della distanza attuale");
});
