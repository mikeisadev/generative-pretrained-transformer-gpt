// Test delle forme del tab Immagini e video (js/tabs/diffusion/shapes.js) con three.js caricato da
// webapp/vendor, senza DOM:   node --test webapp/tests/*.test.mjs
// Le scatole di instancedBoxes sono un solo InstancedMesh senza figli: i riquadri di inquadratura (framingBox
// del tab, meshBounds del nucleo) contano tutte le istanze da soli, senza il riquadro invisibile che il tab
// aggiungeva quando la regia ne contava una sola.
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
      return next(specifier, context);
    }`),
);
const THREE = await import("three");
const { framingBox, instancedBoxes, lineSegments, recolor } = await import("../js/tabs/diffusion/shapes.js");
const { meshBounds } = await import("../js/core/director.js");

const corners = (box) => [box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z];

test("instancedBoxes: un InstancedMesh senza figli invisibili; framingBox e meshBounds contano tutte le istanze", () => {
  const items = [
    { x: -3, y: 0, z: 0, sx: 1, sy: 2, sz: 0.5, color: 0xff0000 },
    { x: 4, y: 1, z: 2, sx: 2, sy: 1, sz: 1, color: 0x00ff00 },
  ];
  const mesh = instancedBoxes(items);
  assert.equal(mesh.count, 2);
  assert.equal(mesh.children.length, 0, "niente riquadro invisibile");
  assert.notEqual(mesh.userData.noBounds, true, "la regia lo conta");
  const group = new THREE.Group();
  group.position.set(10, 0, 0);
  group.add(mesh);
  // le linee lunghe restano fuori dai riquadri (noBounds), come prima
  group.add(lineSegments([[new THREE.Vector3(-100, 0, 0), new THREE.Vector3(100, 0, 0)]], 0xffffff));
  const expected = [6.5, -1, -0.25, 15, 1.5, 2.5]; // le due scatole, spostate di 10 in x con il gruppo
  assert.deepEqual(corners(framingBox(group)), expected);
  assert.deepEqual(corners(meshBounds(group)), expected);
  // ricolorare non cambia le matrici: stesso riquadro
  recolor(mesh, () => 0x123456);
  assert.deepEqual(corners(framingBox(group)), expected);
  // un gruppo nascosto non si inquadra (framingBox guarda solo ciò che si vede)
  mesh.visible = false;
  assert.equal(framingBox(group).isEmpty(), true);
});
