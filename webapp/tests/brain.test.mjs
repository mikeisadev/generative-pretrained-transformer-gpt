// Test del cervello 3D del nucleo (js/core/brain/: moduli puri, niente three.js né DOM):
//   node --test webapp/tests/*.test.mjs
// Atlante completo e trasformazione MNI → scena, lobi, forma (funzione distanza, surface nets, giri),
// geometria (curve e tubi), rumore con seme, morfologie dei neuroni, specifica (anche nel formato della
// ricerca), disposizione del microcircuito.
import assert from "node:assert/strict";
import { test } from "node:test";

import { ATLAS_IDS, HEMISPHERES, KINDS, LOBES, MARKERS, MM, PATHWAYS, REGIONS, atlasEntry, atlasKind, formatMni, mirrorMni, mniToScene, placements, sceneToMni } from "../js/core/brain/atlas.js";
import { DISPLAY_SCALE, FAMILY, layoutCircuit } from "../js/core/brain/circuitLayout.js";
import { demoBrainSpec } from "../js/core/brain/demoSpec.js";
import { catmullRom, computeNormals, sampleCurve, sweepTube } from "../js/core/brain/geometry.js";
import { NEURON_TYPES, SEG, SPH, buildMorphology, resolveNeuronType } from "../js/core/brain/morphology.js";
import { createNoise3D, hashString, mulberry32 } from "../js/core/brain/noise.js";
import { HEMISPHERE_BOX, buildCerebellum, buildCortex, buildHemisphereBase, hemisphereSdf, lobeOf, mirrorMesh, surfaceNets } from "../js/core/brain/shape.js";
import { BrainSpecError, EDGE_KINDS, SPEC_LIMITS, normalizeBrainSpec, normalizeEdgeKind, normalizeStrength, safeUrl } from "../js/core/brain/spec.js";

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} ≉ ${b}`);

// Gli id che la specifica del round fissa per l'atlante del nucleo (gli stessi della ricerca).
const SPEC_REGIONS = ["pfc_dl", "pfc_vm", "acc", "broca", "premotor", "motor", "somato", "parietal_post", "angular", "wernicke", "a1", "stg", "temporal_ant", "it", "ffa", "v1", "v2v4", "mt", "insula", "entorhinal", "parahippocampal", "hippocampus", "amygdala", "thalamus", "striatum", "vta_snc", "cerebellum", "brainstem", "corpus_callosum", "retina", "cochlea"];
const SPEC_MARKERS = ["lgn", "mgn", "pulvinar", "cochlear_nucleus", "superior_olive", "inferior_colliculus"];
const SPEC_PATHWAYS = ["arcuate", "optic_radiation", "auditory_radiation", "mesolimbic", "perforant_path"];

/** Ogni spigolo in esattamente due triangoli con versi opposti: mesh chiusa e orientata. */
function badEdges(indices) {
  const edges = new Map();
  for (let f = 0; f < indices.length; f += 3) {
    for (let e = 0; e < 3; e++) {
      const k = `${indices[f + e]},${indices[f + ((e + 1) % 3)]}`;
      edges.set(k, (edges.get(k) ?? 0) + 1);
    }
  }
  let bad = 0;
  for (const [k, n] of edges) {
    const [a, b] = k.split(",");
    if (n !== 1 || edges.get(`${b},${a}`) !== 1) bad++;
  }
  return bad;
}

// ---------------------------------------------------------------- atlante
test("atlante: tutti gli id della specifica, con i campi completi", () => {
  assert.deepEqual(REGIONS.map((r) => r.id).sort(), [...SPEC_REGIONS].sort());
  assert.deepEqual(MARKERS.map((r) => r.id).sort(), [...SPEC_MARKERS].sort());
  assert.deepEqual(PATHWAYS.map((r) => r.id).sort(), [...SPEC_PATHWAYS].sort());
  assert.equal(new Set(ATLAS_IDS).size, ATLAS_IDS.length, "id unici");
  for (const r of [...REGIONS, ...MARKERS]) {
    assert.ok(r.name && r.text && r.note, `${r.id}: nome, testo e nota`);
    assert.equal(r.mni.length, 3, r.id);
    // dentro il cervello MNI (con occhio e coclea poco fuori dal cranio)
    assert.ok(r.mni[0] <= 0 || r.hemisphere === "midline", `${r.id}: coordinate dell'emisfero sinistro`);
    assert.ok(r.mni[0] >= -75 && r.mni[0] <= 5 && r.mni[1] >= -110 && r.mni[1] <= 75 && r.mni[2] >= -75 && r.mni[2] <= 85, `${r.id}: fuori dal cervello`);
    assert.ok(r.radius > 0 && r.radius < 30, r.id);
    assert.ok(r.hemisphere in HEMISPHERES, r.id);
    assert.ok(r.kind in KINDS, r.id);
    assert.ok(["aal", "lit", "approx"].includes(r.source), `${r.id}: fonte`);
  }
  for (const r of REGIONS) assert.ok(r.lobe in LOBES, `${r.id}: lobo`);
  for (const m of MARKERS) assert.equal(atlasKind(m.within), "region", `${m.id}: dentro una regione`);
  for (const p of PATHWAYS) {
    assert.ok(p.path.length >= 3 && /^#[0-9a-f]{6}$/i.test(p.color), p.id);
    for (const id of p.through) assert.ok(atlasEntry(id), `${p.id}: ${id}`);
  }
  assert.equal(atlasKind("a1"), "region");
  assert.equal(atlasKind("lgn"), "marker");
  assert.equal(atlasKind("arcuate"), "pathway");
  assert.equal(atlasKind("boh"), null);
});

test("atlante: da MNI alla scena e ritorno, senza specchi", () => {
  for (const p of [[0, 0, 0], [-25, -21, -10], [40, 60, -30], [12.5, -100, 70]]) {
    const back = sceneToMni(mniToScene(p));
    for (let a = 0; a < 3; a++) near(back[a], p[a], 1e-9);
  }
  // 1 cm = 1 unità; davanti (y+) a sinistra dello schermo (x−), sopra (z+) in alto (y+), sinistra (x−) verso la camera (z+)
  const o = mniToScene([0, 0, 0]);
  const d = (p) => mniToScene(p).map((v, i) => v - o[i]);
  assert.deepEqual(d([0, 10, 0]).map((v) => Math.round(v * 10) / 10), [-1, 0, 0]);
  assert.deepEqual(d([0, 0, 10]).map((v) => Math.round(v * 10) / 10), [0, 1, 0]);
  assert.deepEqual(d([-10, 0, 0]).map((v) => Math.round(v * 10) / 10), [0, 0, 1]);
  near(MM, 0.1);
  // rotazione propria: (immagine di x) × (immagine di y) = immagine di z (determinante +1)
  const ex = d([1, 0, 0]);
  const ey = d([0, 1, 0]);
  const ez = d([0, 0, 1]);
  const cross = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
  for (let a = 0; a < 3; a++) near(cross[a] / (MM * MM), ez[a] / MM, 1e-9);
  assert.equal(formatMni([-25.4, -21, 9.6]), "x −25, y −21, z 10");
  assert.deepEqual(mirrorMni([-25, 3, 4]), [25, 3, 4]);
});

test("atlante: quante volte si disegna una voce (emisferi)", () => {
  assert.equal(placements(atlasEntry("hippocampus")).length, 2);
  assert.deepEqual(placements(atlasEntry("broca")).map((p) => p.side), ["left"]);
  const ffa = placements(atlasEntry("ffa"));
  assert.equal(ffa.length, 1);
  assert.ok(ffa[0].mni[0] > 0, "l'area dei volti è a destra");
  assert.deepEqual(placements(atlasEntry("brainstem")).map((p) => p.side), ["midline"]);
});

test("lobi: le regioni corticali dell'atlante cadono nel loro lobo", () => {
  for (const r of REGIONS.filter((x) => x.kind === "cortical")) assert.equal(lobeOf(r.mni), r.lobe, `${r.id} (${r.mni})`);
  // lo stesso punto nell'altro emisfero
  assert.equal(lobeOf(mirrorMni(atlasEntry("broca").mni)), "frontale");
});

// ---------------------------------------------------------------- forma
test("forma: i centroidi delle regioni corticali e profonde stanno dentro l'emisfero liscio", () => {
  for (const r of REGIONS.filter((x) => x.kind === "cortical" || ["hippocampus", "amygdala", "thalamus", "striatum"].includes(x.id))) {
    assert.ok(hemisphereSdf(...r.mni) < 1.5, `${r.id}: fuori di ${hemisphereSdf(...r.mni).toFixed(1)} mm`);
  }
  // fuori: oltre la parete mediale, sopra il vertice, dentro il cervelletto
  assert.ok(hemisphereSdf(3, -20, 30) > 0);
  assert.ok(hemisphereSdf(-30, -20, 90) > 0);
  assert.ok(hemisphereSdf(-20, -70, -40) > 0);
});

test("surface nets: una sfera diventa una mesh chiusa con i vertici sulla superficie", () => {
  const R = 10;
  const sphere = (x, y, z) => Math.hypot(x, y, z) - R;
  const mesh = surfaceNets(sphere, { min: [-13, -13, -13], max: [13, 13, 13] }, 1.2);
  assert.ok(mesh.positions.length / 3 > 400);
  assert.equal(badEdges(mesh.indices), 0);
  let worst = 0;
  for (let i = 0; i < mesh.positions.length; i += 3) worst = Math.max(worst, Math.abs(Math.hypot(mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]) - R));
  assert.ok(worst < 0.7, `scarto massimo ${worst}`);
  // normali verso l'esterno: con la regola dei triangoli antiorari
  computeNormals(mesh);
  for (let i = 0; i < mesh.positions.length; i += 3 * 37) {
    const dot = mesh.normals[i] * mesh.positions[i] + mesh.normals[i + 1] * mesh.positions[i + 1] + mesh.normals[i + 2] * mesh.positions[i + 2];
    assert.ok(dot > 0, "normale verso l'esterno");
  }
});

test("emisfero: superficie chiusa nei limiti del cervello MNI, corteccia deterministica, specchio", () => {
  const base = buildHemisphereBase({ cell: 3 });
  const n = base.positions.length / 3;
  assert.ok(n > 2000 && n < 20000, `${n} vertici`);
  assert.ok(badEdges(base.indices) <= 4, "al più qualche spigolo ambiguo di surface nets");
  for (let i = 0; i < base.positions.length; i += 3) {
    for (let a = 0; a < 3; a++) assert.ok(base.positions[i + a] >= HEMISPHERE_BOX.min[a] && base.positions[i + a] <= HEMISPHERE_BOX.max[a]);
  }
  const a = buildCortex(base, { side: "left", seed: 11, gyrus: 6 });
  const b = buildCortex(base, { side: "left", seed: 11, gyrus: 6 });
  const c = buildCortex(base, { side: "left", seed: 12, gyrus: 6 });
  assert.deepEqual(a.positions, b.positions, "stesso seme, stessa corteccia");
  assert.notDeepEqual(a.positions, c.positions, "semi diversi, giri diversi");
  assert.equal(a.depth.length, n);
  assert.ok(a.depth.every((d) => d >= 0 && d <= 1));
  assert.ok(a.lobe.every((l) => l >= 0 && l < 6));
  const m = mirrorMesh(base);
  assert.equal(m.positions[0], -base.positions[0]);
  assert.equal(m.indices[1], base.indices[2], "triangoli girati");
  const right = buildCortex(m, { side: "right", seed: 23, gyrus: 6 });
  assert.ok(right.positions.every((v, i) => i % 3 !== 0 || v >= -3), "l'emisfero destro sta a destra");
});

test("cervelletto: sotto il tentorio, folia come anelli orizzontali", () => {
  const mesh = buildCerebellum({ rows: 60, cols: 24 });
  let maxZ = -Infinity;
  for (let i = 0; i < mesh.positions.length; i += 3) maxZ = Math.max(maxZ, mesh.positions[i + 2]);
  assert.ok(maxZ < -10, `cima a z = ${maxZ}`);
  assert.equal(mesh.depth.length, mesh.positions.length / 3);
  for (let i = 0; i < mesh.normals.length; i += 3) near(Math.hypot(mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]), 1, 1e-4);
});

// ---------------------------------------------------------------- geometria e rumore
test("geometria: Catmull-Rom passa per i punti, il campionamento è a passo costante, il tubo è chiuso", () => {
  const pts = [[0, 0, 0], [4, 1, 0], [8, 0, 2], [12, 3, 2]];
  for (const [t, p] of [[0, pts[0]], [1, pts[3]]]) catmullRom(pts, t).forEach((v, a) => near(v, p[a], 1e-6));
  const { points, lengths, total } = sampleCurve(pts, 25);
  assert.equal(points.length, 25);
  near(lengths[24], total, 1e-9);
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(...points[i].map((v, a) => v - points[i - 1][a]));
    near(d, total / 24, total / 24 * 0.08);
  }
  const tube = sweepTube(points, { radius: () => 0.5, radial: 8, lengths });
  // 25 anelli da 9 vertici (cucitura doppia: normali pulite) + 2 centri dei tappi; triangoli: lati + tappi
  assert.equal(tube.positions.length / 3, 25 * 9 + 2);
  assert.equal(tube.indices.length / 3, 24 * 8 * 2 + 2 * 8);
  for (let i = 0; i < tube.normals.length; i += 3) near(Math.hypot(tube.normals[i], tube.normals[i + 1], tube.normals[i + 2]), 1, 1e-5);
  assert.equal(tube.arc.length, tube.positions.length / 3);
});

test("rumore: deterministico, nei limiti, diverso per semi diversi", () => {
  const a = createNoise3D(5);
  const b = createNoise3D(5);
  const c = createNoise3D(6);
  let diff = 0;
  for (let i = 0; i < 200; i++) {
    const p = [i * 0.37, i * 0.11 - 3, Math.sin(i) * 4];
    const v = a(...p);
    assert.equal(v, b(...p));
    assert.ok(Math.abs(v) <= 1.1);
    diff += Math.abs(v - c(...p));
  }
  assert.ok(diff > 1);
  const r = mulberry32(42);
  const first = [r(), r(), r()];
  assert.ok(first.every((x) => x >= 0 && x < 1));
  assert.notEqual(hashString("piramidale:1"), hashString("piramidale:2"));
});

// ---------------------------------------------------------------- neuroni
test("morfologie: tutti i tipi, deterministici e plausibili", () => {
  for (const type of Object.keys(NEURON_TYPES)) {
    const m = buildMorphology(type, 3);
    assert.deepEqual(m.segments, buildMorphology(type, 3).segments, `${type}: stesso seme`);
    assert.equal(m.segments.length % 11, 0);
    assert.equal(m.spheres.length % 6, 0);
    assert.ok(m.segments.length / 11 >= 5, `${type}: segmenti`);
    for (let i = 0; i < m.segments.length; i += 11) {
      assert.ok(m.segments[i + 6] > 0 && m.segments[i + 7] > 0, `${type}: raggi positivi`);
      assert.ok(Object.values(SEG).includes(m.segments[i + 8]), `${type}: tipo di segmento`);
    }
    let somata = 0;
    for (let i = 0; i < m.spheres.length; i += 6) if (m.spheres[i + 4] === SPH.soma) somata++;
    assert.ok(somata >= 1, `${type}: un soma`);
    assert.ok(m.inputs.length >= 1 && m.axonEnd.every(Number.isFinite), type);
    assert.ok(m.extent.maxY - m.extent.minY > 0.3 && m.extent.maxY - m.extent.minY < 2.5, `${type}: altezza ${m.extent.maxY - m.extent.minY}`);
    assert.ok(DISPLAY_SCALE[type] > 0 && FAMILY[type], type);
  }
  const kinds = (type) => {
    const m = buildMorphology(type, 1);
    const set = new Set();
    for (let i = 0; i < m.segments.length; i += 11) set.add(m.segments[i + 8]);
    return set;
  };
  for (const type of ["pyramidal", "relay", "purkinje", "projection"]) assert.ok(kinds(type).has(SEG.myelin) && kinds(type).has(SEG.node), `${type}: mielina e nodi di Ranvier`);
  for (const type of ["granule", "dopaminergic", "basket"]) assert.ok(!kinds(type).has(SEG.myelin), `${type}: assone senza mielina`);
  assert.ok(kinds("hair").has(SEG.cilium), "stereociglia");
  // la cellula granulare si divide a T: la fibra parallela corre lungo z
  const g = buildMorphology("granule", 1);
  let zSpan = 0;
  for (let i = 0; i < g.segments.length; i += 11) zSpan = Math.max(zSpan, Math.abs(g.segments[i + 5]));
  assert.ok(zSpan > 0.4);
  // la Purkinje è piatta: il ventaglio sta nel piano x-y
  const p = buildMorphology("purkinje", 2);
  let maxZ = 0;
  for (let i = 0; i < p.segments.length; i += 11) if (p.segments[i + 8] === SEG.dendrite) maxZ = Math.max(maxZ, Math.abs(p.segments[i + 2]), Math.abs(p.segments[i + 5]));
  assert.ok(maxZ < 0.08, `ventaglio spesso ${maxZ}`);
  // il dopaminergico ha un cespuglio di varicosità
  const d = buildMorphology("dopaminergic", 1);
  let varicosities = 0;
  for (let i = 0; i < d.spheres.length; i += 6) if (d.spheres[i + 4] === SPH.varicosity) varicosities++;
  assert.ok(varicosities >= 8);
  // meno dettaglio con il rendering software
  assert.ok(buildMorphology("pyramidal", 1, { detail: 0.6 }).spheres.length < buildMorphology("pyramidal", 1).spheres.length);
  // alias della ricerca
  assert.equal(resolveNeuronType("piramidale"), "pyramidal");
  assert.equal(resolveNeuronType("canestro_pv"), "basket");
  assert.equal(resolveNeuronType("cellula_ciliata"), "hair");
  assert.equal(resolveNeuronType("neurone_gangli_base"), "pallidal");
  assert.equal(resolveNeuronType("boh"), null);
});

// ---------------------------------------------------------------- specifica
test("specifica: l'esempio è valido senza avvisi", () => {
  const spec = normalizeBrainSpec(demoBrainSpec, { strict: true });
  assert.deepEqual(spec.warnings, []);
  assert.equal(spec.circuit.groups.length, 10);
  assert.ok(spec.analogies.every((a) => a.different && a.sources.length && a.sources.every((s) => s.url?.startsWith("https://"))));
  assert.deepEqual(spec.pathways.map((p) => p.id), ["auditory_radiation", "arcuate"]);
  assert.ok(spec.markers.includes("mgn"));
});

test("specifica: accetta il formato della ricerca (brain_data.json)", () => {
  const spec = normalizeBrainSpec(
    {
      title: "Ricerca",
      regions: ["hippocampus", "pulvinar"],
      circuit: {
        title: "Trisinaptico",
        groups: [
          { id: "ec", label: "Entorinale", region: "entorhinal", neuron_type: "stellata_entorinale", count_shown: 8, real_count: 650000, real_count_text: "circa 650.000" },
          { id: "dg", label: "Giro dentato", region: "hippocampus", neuron_type: "granulo_dentato", count_shown: 30 },
        ],
        edges: [{ from: "ec", to: "dg", kind: "eccitatoria", pathway: "perforant_path" }],
        analogy_map: [{ group: "dg", ai_component: "Chunk con id univoci" }],
      },
      analogies: [{ id: "x", ai: "A", brain: "B", regions: ["hippocampus"], markers: ["pulvinar"], text: "t", similar: "s", different: "d", strength: "debole/metafora", sources: [{ title: "T", url: "https://doi.org/10.1/x", year: 2001 }] }],
    },
    { strict: true },
  );
  const [ec, dg] = spec.circuit.groups;
  assert.equal(ec.type, "stellate");
  assert.equal(ec.real, 650000);
  assert.equal(ec.realText, "circa 650.000");
  assert.equal(dg.type, "dentate");
  assert.equal(dg.count, 18, "contato ridotto al massimo disegnabile");
  assert.equal(dg.ai, "Chunk con id univoci");
  assert.equal(spec.circuit.edges[0].kind, "excitatory");
  assert.equal(spec.circuit.edges[0].pathway, "perforant_path");
  assert.equal(spec.analogies[0].strength, "debole");
  assert.deepEqual(spec.regions, ["hippocampus"]);
  assert.deepEqual(spec.markers, ["pulvinar"]);
  assert.ok(spec.warnings.some((w) => w.includes("18")), "avviso sul conteggio");
});

test("specifica: errori e avvisi leggibili, link sicuri", () => {
  const raw = { title: "X", regions: ["ippocampo", "a1"], pathways: ["boh", { id: "mio", through: ["a1", "stg"] }], circuit: { groups: [{ id: "a", type: "boh" }], edges: [{ from: "a", to: "z" }] } };
  const spec = normalizeBrainSpec(raw);
  assert.deepEqual(spec.regions, ["a1"]);
  assert.deepEqual(spec.pathways.map((p) => p.id), ["mio"]);
  assert.equal(spec.circuit.groups[0].type, "projection");
  assert.ok(spec.warnings.some((w) => w.includes("ippocampo")));
  assert.throws(() => normalizeBrainSpec(raw, { strict: true }), (e) => e instanceof BrainSpecError && e.errors.length >= 3);
  assert.equal(safeUrl("javascript:alert(1)"), null);
  assert.equal(safeUrl("ftp://x.org/a"), null);
  assert.equal(safeUrl("https://doi.org/10.1/x"), "https://doi.org/10.1/x");
  assert.equal(normalizeStrength("Forte"), "forte");
  assert.equal(normalizeStrength("debole/metafora"), "debole");
  assert.equal(normalizeEdgeKind("inibitoria"), "inhibitory");
  assert.equal(normalizeEdgeKind("modulatory"), "modulatory");
  assert.ok(Object.keys(EDGE_KINDS).length === 3);
  // senza regioni dichiarate si evidenziano quelle usate da analogie e circuito
  const implicit = normalizeBrainSpec({ title: "Y", analogies: [{ id: "a", ai: "x", brain: "y", regions: ["v1"], different: "d", strength: "media", sources: ["Libro"] }] });
  assert.deepEqual(implicit.highlight, ["v1"]);
  assert.equal(implicit.analogies[0].sources[0].url, null);
});

test("specifica: oltre i limiti si tagliano analogie e sinapsi, ma lo si dice (mai in silenzio)", () => {
  const analogy = (i) => ({ id: `a${i}`, ai: "x", brain: "y", regions: ["v1"], different: "d", strength: "media", sources: ["Libro"] });
  const groups = [{ id: "a", type: "pyramidal" }, { id: "b", type: "pyramidal" }];
  const edges = Array.from({ length: SPEC_LIMITS.edges + 5 }, () => ({ from: "a", to: "b", kind: "excitatory" }));
  const spec = normalizeBrainSpec({ title: "Z", analogies: Array.from({ length: SPEC_LIMITS.analogies + 3 }, (_, i) => analogy(i)), circuit: { groups, edges } });
  assert.equal(spec.analogies.length, SPEC_LIMITS.analogies);
  assert.equal(spec.circuit.edges.length, SPEC_LIMITS.edges);
  assert.ok(spec.warnings.some((w) => w.includes("3 ignorate") && w.startsWith("analogies")), spec.warnings.join("\n"));
  assert.ok(spec.warnings.some((w) => w.includes("5 ignorate") && w.startsWith("circuit.edges")), spec.warnings.join("\n"));
});

// ---------------------------------------------------------------- microcircuito
test("microcircuito: colonne da sinistra a destra, fibre per ogni sinapsi, deterministico", () => {
  const spec = normalizeBrainSpec(demoBrainSpec);
  const layout = layoutCircuit(spec.circuit, { detail: 0.6 });
  assert.equal(layout.columns.length, spec.circuit.groups.length);
  for (let i = 1; i < layout.columns.length; i++) assert.ok(layout.columns[i].left > layout.columns[i - 1].left, "colonne in ordine");
  for (let i = 1; i < layout.columns.length; i++) assert.ok(layout.columns[i].fire > layout.columns[i - 1].fire, "scaricano in ordine");
  assert.equal(layout.cells.length, spec.circuit.groups.reduce((n, g) => n + g.count, 0));
  for (const cell of layout.cells) {
    const col = layout.columns[cell.column];
    assert.ok(cell.position[0] >= col.left - 1e-9 && cell.position[0] <= col.right + 1e-9, "il soma sta nella sua colonna");
  }
  const perEdge = new Map();
  for (const f of layout.fibers) perEdge.set(f.edge, (perEdge.get(f.edge) ?? 0) + 1);
  assert.equal(perEdge.size, spec.circuit.edges.length);
  for (const n of perEdge.values()) assert.ok(n >= 2 && n <= 5);
  // le fibre all'indietro passano sopra le colonne
  const back = layout.fibers.filter((f) => f.direction === "backward");
  assert.ok(back.length > 0);
  for (const f of back) assert.ok(Math.max(...f.points.map((p) => p[1])) > layout.top);
  // stesso circuito, stessa disposizione
  const again = layoutCircuit(spec.circuit, { detail: 0.6 });
  assert.deepEqual(again.cells.map((c) => c.position), layout.cells.map((c) => c.position));
  assert.ok(layout.box.max[0] > layout.box.min[0]);
});
