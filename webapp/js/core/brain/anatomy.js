/**
 * Il cervello 3D (three.js) costruito dalla forma pura (shape.js, structures.js): due emisferi con giri
 * e solchi, cervelletto con le folia, tronco encefalico, strutture profonde, occhi con la via ottica e
 * coclee con il nervo acustico. Più le regole di disegno: corteccia trasparente regolabile, piani di
 * taglio con la sezione a colore pieno, evidenziazione per regione.
 *
 *   const data = await loadAnatomyData("high");   // array puri, in cache per tutta la pagina
 *   const brain = new BrainModel(data);            // oggetti three.js (da liberare con dispose)
 *   host.scene.add(brain.root);
 *   brain.setCortexOpacity(0.45); brain.setCut({ axis: "coronal", mm: -20 }); brain.setLobeColors(true);
 *
 * Perché la cache dei dati: generare la corteccia costa 150-450 ms (dipende dalla qualità); i tab sono
 * sei e ognuno può aprire il suo cervello: la forma è la stessa, si calcola una volta sola per pagina.
 */
import * as THREE from "three";
import { LOBES, MM, REGIONS, mniToScene, placements } from "./atlas.js";
import { mergeMeshes } from "./geometry.js";
import { MAX_PULSES, cortexDepthMaterial, cortexMaterial, solidMaterial } from "./shaders.js";
import { LOBE_ORDER, buildCerebellum, buildCortex, buildHemisphereBase, mirrorMesh } from "./shape.js";
import * as S from "./structures.js";

/** Qualità della geometria: "high" con una GPU, "low" con il rendering software (SwiftShader). */
export const QUALITY = Object.freeze({
  high: { cell: 1.25, minor: 4.2, gyrus: 4.2, cerebellum: { rows: 220, cols: 96 }, tube: 1 },
  low: { cell: 2.6, minor: 4.4, gyrus: 6.6, cerebellum: { rows: 120, cols: 48 }, tube: 0.65 },
});

/** Colori della corteccia (base chiara e calda, fondo dei solchi più scuro, sezione color sostanza bianca). */
export const CORTEX_COLORS = Object.freeze({
  base: "#d9cbc3",
  sulcus: "#9c8580",
  interior: "#efe6d8",
  lobeColors: LOBE_ORDER.map((id) => LOBES[id].color),
});

const cache = new Map(); // qualità → Promise dei dati

/** Lascia respirare la pagina tra un pezzo e l'altro della costruzione (la scheda "Preparo la scena…" resta fluida). */
const nextTask = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Dati puri del cervello (coordinate della scena), calcolati una volta per qualità e poi riusati.
 * @param {"high" | "low"} quality
 */
export function loadAnatomyData(quality = "high") {
  if (!cache.has(quality)) {
    const promise = buildAnatomyData(quality);
    promise.catch(() => cache.delete(quality));
    cache.set(quality, promise);
  }
  return cache.get(quality);
}

async function buildAnatomyData(quality) {
  const q = QUALITY[quality] ?? QUALITY.high;
  const timings = {};
  let t = performance.now();
  const lap = (name) => {
    const now = performance.now();
    timings[name] = now - t;
    t = now;
  };
  const base = buildHemisphereBase({ cell: q.cell });
  lap("base");
  await nextTask();
  const left = buildCortex(base, { side: "left", seed: 11, minor: q.minor, gyrus: q.gyrus });
  const right = buildCortex(mirrorMesh(base), { side: "right", seed: 23, minor: q.minor, gyrus: q.gyrus });
  lap("cortex");
  await nextTask();
  const cerebellum = buildCerebellum(q.cerebellum);
  lap("cerebellum");
  const deep = buildStructures(q);
  lap("deep");
  const cortex = { left: toScene(left, ["depth", "lobe"]), right: toScene(right, ["depth", "lobe"]) };
  const data = {
    quality,
    cortex,
    cerebellum: toScene(cerebellum, ["depth"]),
    deep: deep.map((s) => ({ ...s, mesh: toScene(s.mesh) })),
    timings,
    vertices: 0,
  };
  data.anchors = cortexAnchors(data.cortex);
  lap("anchors");
  data.vertices = [data.cortex.left, data.cortex.right, data.cerebellum, ...data.deep.map((s) => s.mesh)].reduce((n, m) => n + m.positions.length / 3, 0);
  return data;
}

/** Corteccia liscia e rada (miniatura): due emisferi senza giri e un cervelletto semplice, ~10.000 triangoli. */
function miniCortexMesh() {
  const base = buildHemisphereBase({ cell: 5 });
  return mergeMeshes([base, mirrorMesh(base), buildCerebellum({ rows: 36, cols: 24 })]);
}

/** Da mm MNI a coordinate della scena (posizioni e normali; la rotazione non cambia il verso dei triangoli). */
function toScene(mesh, extra = []) {
  const n = mesh.positions.length / 3;
  const positions = new Float32Array(n * 3);
  const normals = new Float32Array(n * 3);
  const p = [0, 0, 0];
  const out = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    p[0] = mesh.positions[i * 3];
    p[1] = mesh.positions[i * 3 + 1];
    p[2] = mesh.positions[i * 3 + 2];
    mniToScene(p, out);
    positions.set(out, i * 3);
    if (mesh.normals) {
      // stessa rotazione di mniToScene senza traslazione né scala: (x, y, z) → (−y, z, −x)
      normals[i * 3] = -mesh.normals[i * 3 + 1];
      normals[i * 3 + 1] = mesh.normals[i * 3 + 2];
      normals[i * 3 + 2] = -mesh.normals[i * 3];
    }
  }
  const result = { positions, normals, indices: mesh.indices };
  for (const name of extra) result[name] = mesh[name];
  return result;
}

/** Strutture profonde e organi di senso, lato sinistro e (specchiato) destro, unite per regione. */
function buildStructures(q) {
  const tubeScale = (n) => Math.max(8, Math.round(n * q.tube));
  const both = (mesh) => [mesh, mirrorStructure(mesh)];
  const eye = S.eyeMeshes();
  const cochlea = S.cochleaMeshes();
  const striatum = S.striatumMeshes();
  const vta = S.vtaSncMeshes();
  const list = [
    { id: "hippocampus", region: "hippocampus", color: S.STRUCTURE_COLORS.hippocampus, parts: both(S.hippocampusMesh({ samples: tubeScale(40), radial: tubeScale(14) })) },
    { id: "amygdala", region: "amygdala", color: S.STRUCTURE_COLORS.amygdala, parts: both(S.amygdalaMesh()) },
    { id: "thalamus", region: "thalamus", color: S.STRUCTURE_COLORS.thalamus, parts: both(S.thalamusMesh()) },
    { id: "striatum", region: "striatum", color: S.STRUCTURE_COLORS.striatum, parts: striatum.flatMap(both) },
    { id: "vta_snc", region: "vta_snc", color: S.STRUCTURE_COLORS.vta_snc, parts: vta.flatMap(both) },
    { id: "corpus_callosum", region: "corpus_callosum", color: S.STRUCTURE_COLORS.corpus_callosum, parts: [S.corpusCallosumMesh()] },
    { id: "brainstem", region: "brainstem", color: S.STRUCTURE_COLORS.brainstem, parts: [S.brainstemMesh({ samples: tubeScale(40), radial: tubeScale(20) })] },
    { id: "optic", region: null, color: S.STRUCTURE_COLORS.nerve, parts: both(S.opticPathMesh()), label: "Nervo e tratto ottico" },
    { id: "eye", region: null, color: S.STRUCTURE_COLORS.eye, parts: both(eye.sclera), glass: true, label: "Occhio" },
    { id: "iris", region: null, color: S.STRUCTURE_COLORS.iris, parts: both(eye.iris), label: "Iride" },
    { id: "retina", region: "retina", color: S.STRUCTURE_COLORS.retina, parts: both(eye.retina) },
    { id: "cochlea", region: "cochlea", color: S.STRUCTURE_COLORS.cochlea, parts: [cochlea.spiral, ...cochlea.canals].flatMap(both) },
    { id: "auditory_nerve", region: null, color: S.STRUCTURE_COLORS.nerve, parts: both(cochlea.nerve), label: "Nervo acustico" },
  ];
  return list.map(({ parts, ...rest }) => ({ ...rest, mesh: mergeMeshes(parts) }));
}

/** Una struttura nell'altro emisfero: x → −x e triangoli girati. */
function mirrorStructure(mesh) {
  const positions = mesh.positions.slice();
  const normals = mesh.normals.slice();
  for (let i = 0; i < positions.length; i += 3) {
    positions[i] = -positions[i];
    normals[i] = -normals[i];
  }
  const indices = mesh.indices.slice();
  for (let f = 0; f < indices.length; f += 3) {
    const t = indices[f + 1];
    indices[f + 1] = indices[f + 2];
    indices[f + 2] = t;
  }
  return { ...mesh, positions, normals, indices };
}

/**
 * Per ogni regione corticale (e ogni lato in cui compare) il vertice della corteccia più vicino al
 * centroide: lì sta la macchia evidenziata e il marcatore. Un centroide profondo (il giro di Heschl,
 * dentro la scissura laterale) finisce così sul bordo della scissura, dove la regione si affaccia.
 * @returns {Map<string, {side, point: number[], normal: number[], depthMm: number}[]>}
 */
function cortexAnchors(cortex) {
  const anchors = new Map();
  for (const region of REGIONS) {
    if (region.kind !== "cortical") continue;
    const list = [];
    for (const { side, mni } of placements(region)) {
      const mesh = side === "right" ? cortex.right : cortex.left;
      const target = mniToScene(mni);
      const { index, distance } = nearestVertex(mesh.positions, target);
      list.push({
        side,
        point: Array.from(mesh.positions.subarray(index * 3, index * 3 + 3)),
        normal: Array.from(mesh.normals.subarray(index * 3, index * 3 + 3)),
        centroid: target,
        depthMm: distance / MM,
      });
    }
    anchors.set(region.id, list);
  }
  return anchors;
}

function nearestVertex(positions, p) {
  let best = Infinity;
  let index = 0;
  for (let i = 0; i < positions.length; i += 3) {
    const dx = positions[i] - p[0];
    const dy = positions[i + 1] - p[1];
    const dz = positions[i + 2] - p[2];
    const d = dx * dx + dy * dy + dz * dz;
    if (d < best) {
      best = d;
      index = i / 3;
    }
  }
  return { index, distance: Math.sqrt(best) };
}

/** BufferGeometry da array puri (gli array sono condivisi con la cache: non vanno modificati). */
function geometryOf(mesh, attributes = {}) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(mesh.normals, 3));
  for (const [name, [array, size]] of Object.entries(attributes)) g.setAttribute(name, new THREE.BufferAttribute(array, size));
  g.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

/** Piani di taglio: asse anatomico → piano nella scena e lato tenuto (vedi setCut). */
export const CUTS = Object.freeze({
  sagittal: { label: "Sagittale", range: [-60, 60], value: 1, hint: "Taglia tra sinistra e destra: si vede la faccia interna dell'emisfero sinistro." },
  coronal: { label: "Coronale", range: [-95, 60], value: -20, hint: "Taglia tra davanti e dietro, come una fetta di pane: si vedono talamo, striato e ippocampo." },
  axial: { label: "Assiale", range: [-40, 70], value: 8, hint: "Taglia tra sopra e sotto: si guarda dall'alto dentro il cervello." },
});

export class BrainModel {
  /** @param {Awaited<ReturnType<typeof loadAnatomyData>>} data */
  constructor(data) {
    this.data = data;
    this.root = new THREE.Group();
    this.root.name = "brain";
    // un piano sempre presente (lontano quando non si taglia): cambiare il numero di piani
    // ricompilerebbe i programmi, con il rendering software ~100 ms per materiale
    this.clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 1e6);
    this.clipPlanes = [this.clipPlane];
    this.opacity = 1;
    this.cut = null;
    this.materials = new Set();
    this.geometries = new Set();
    this.pulsing = false;

    // corteccia: emisferi e cervelletto, ognuno con colore + passata di profondità (per la trasparenza)
    this.cortex = [];
    const hemis = [
      { id: "left", mesh: data.cortex.left, lobe: true },
      { id: "right", mesh: data.cortex.right, lobe: true },
      { id: "cerebellum", mesh: data.cerebellum, lobe: false },
    ];
    for (const h of hemis) {
      const n = h.mesh.positions.length / 3;
      const highlight = new Float32Array(n * 4);
      const region = new Float32Array(n).fill(-1);
      const lobe = h.lobe ? Float32Array.from(h.mesh.lobe) : new Float32Array(n).fill(-1);
      const geometry = geometryOf(h.mesh, {
        aDepth: [h.mesh.depth, 1],
        aLobe: [lobe, 1],
        aHighlight: [highlight, 4],
        aRegion: [region, 1],
      });
      const material = cortexMaterial(CORTEX_COLORS);
      material.depthFunc = THREE.LessEqualDepth;
      if (!h.lobe) material.uniforms.uBase.value.set("#cdbfc6"); // cervelletto un filo più freddo: si distingue dai lobi
      material.clippingPlanes = this.clipPlanes;
      const depthMaterial = cortexDepthMaterial();
      depthMaterial.transparent = true; // nella lista trasparente, dopo i fasci (vedi setCortexOpacity)
      depthMaterial.clippingPlanes = this.clipPlanes;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `cortex-${h.id}`;
      mesh.userData.cortex = h.id;
      const depth = new THREE.Mesh(geometry, depthMaterial);
      depth.userData.noBounds = true;
      depth.visible = false;
      depth.renderOrder = 9;
      mesh.renderOrder = 10;
      this.root.add(depth, mesh);
      this.cortex.push({ id: h.id, mesh, depth, geometry, material, depthMaterial, highlight, region });
      this.geometries.add(geometry);
      this.materials.add(material).add(depthMaterial);
    }

    // strutture profonde e organi di senso
    this.structures = new Map();
    for (const s of data.deep) {
      const material = solidMaterial(s.color, { opacity: s.glass ? 0.55 : 1, specular: s.glass ? 0.5 : 0.12 });
      if (s.glass) {
        material.side = THREE.FrontSide;
        material.depthWrite = false;
      }
      material.clippingPlanes = this.clipPlanes;
      const geometry = geometryOf(s.mesh);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `structure-${s.id}`;
      mesh.userData.structure = s.id;
      mesh.userData.region = s.region;
      if (s.glass) mesh.renderOrder = 1;
      this.root.add(mesh);
      this.structures.set(s.id, { ...s, mesh, material, geometry });
      this.geometries.add(geometry);
      this.materials.add(material);
    }
    this.setCortexOpacity(1);
  }

  // ---------------------------------------------------------------- aspetto
  /** Opacità della corteccia (emisferi e cervelletto): 1 = opaca, 0 = nascosta. */
  setCortexOpacity(value) {
    const a = Math.max(0, Math.min(1, value));
    this.opacity = a;
    const opaque = a >= 0.99;
    for (const c of this.cortex) {
      c.material.uniforms.uOpacity.value = a;
      c.mesh.visible = a > 0.01;
      c.depth.visible = !opaque && a > 0.01;
      if (c.material.transparent === opaque) {
        // trasparente: niente scrittura della profondità (l'ha già scritta la passata `depth`) e
        // LessEqual: passa solo il frammento più vicino, quello che la passata ha lasciato
        c.material.transparent = !opaque;
        c.material.depthWrite = opaque;
        c.material.needsUpdate = true;
      }
    }
  }

  /** Colori dei lobi (frontale blu, parietale giallo, temporale verde, occipitale rosa…) al posto del colore unico. */
  setLobeColors(on) {
    for (const c of this.cortex) if (c.id !== "cerebellum") c.material.uniforms.uLobeMix.value = on ? 0.72 : 0;
  }

  /**
   * Piano di taglio: axis "sagittal" | "coronal" | "axial", mm = posizione in mm MNI (x, y o z),
   * flip = tiene l'altra metà. null toglie il taglio. Le facce interne diventano visibili (sezione).
   */
  setCut(cut) {
    this.cut = cut && CUTS[cut.axis] ? { axis: cut.axis, mm: cut.mm ?? CUTS[cut.axis].value, flip: Boolean(cut.flip) } : null;
    const plane = this.clipPlane;
    if (!this.cut) plane.set(new THREE.Vector3(1, 0, 0), 1e6);
    else {
      const { axis, mm, flip } = this.cut;
      const s = flip ? -1 : 1;
      if (axis === "sagittal") {
        // x_mni < mm (si tiene la sinistra): z_scena = −x·MM ≥ −mm·MM
        const z0 = mniToScene([mm, 0, 0])[2];
        plane.set(new THREE.Vector3(0, 0, s), -s * z0);
      } else if (axis === "coronal") {
        // y_mni < mm (si tiene la parte posteriore): x_scena ≥ x0
        const x0 = mniToScene([0, mm, 0])[0];
        plane.set(new THREE.Vector3(s, 0, 0), -s * x0);
      } else {
        // z_mni < mm (si tiene la parte inferiore): y_scena ≤ y0
        const y0 = mniToScene([0, 0, mm])[1];
        plane.set(new THREE.Vector3(0, -s, 0), s * y0);
      }
    }
    const side = this.cut ? THREE.DoubleSide : THREE.FrontSide;
    for (const c of this.cortex) {
      if (c.material.side !== side) {
        c.material.side = side;
        c.depthMaterial.side = side;
        c.material.needsUpdate = true;
        c.depthMaterial.needsUpdate = true;
      }
    }
  }

  /** Normale del piano di taglio verso la parte tenuta (la camera guarda la sezione da lì), o null. */
  cutViewDirection() {
    if (!this.cut) return null;
    // la parte tenuta è nel semispazio positivo: la camera sta dalla parte TOLTA e guarda la sezione
    return this.clipPlane.normal.clone().negate();
  }

  // ---------------------------------------------------------------- regioni
  /**
   * Evidenzia le regioni: [{ id, color, index }] (index = posizione per gli impulsi, < MAX_PULSES).
   * Corticali: macchia sulla superficie attorno al punto d'aggancio (calcolata una volta qui, per
   * vertice); profonde: bagliore del materiale della struttura.
   */
  setHighlights(list) {
    const cortical = [];
    const deepGlow = new Map();
    for (const item of list) {
      const anchors = this.data.anchors.get(item.id);
      if (anchors) for (const a of anchors) cortical.push({ ...item, ...a, radius: item.radius });
      else deepGlow.set(item.id, item);
    }
    const tmp = new THREE.Color();
    for (const c of this.cortex) {
      const { highlight, region } = c;
      highlight.fill(0);
      region.fill(-1);
      if (c.id !== "cerebellum") {
        const positions = c.geometry.attributes.position.array;
        const mine = cortical.filter((r) => r.side === c.id);
        for (const r of mine) {
          tmp.set(r.color);
          const R = r.radius * MM;
          const inner = R * 0.5;
          const [ax, ay, az] = r.point;
          for (let v = 0; v < positions.length / 3; v++) {
            const dx = positions[v * 3] - ax;
            const dy = positions[v * 3 + 1] - ay;
            const dz = positions[v * 3 + 2] - az;
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d >= R) continue;
            const w = d <= inner ? 1 : 1 - smooth((d - inner) / (R - inner));
            if (w <= highlight[v * 4 + 3]) continue;
            highlight[v * 4] = tmp.r;
            highlight[v * 4 + 1] = tmp.g;
            highlight[v * 4 + 2] = tmp.b;
            highlight[v * 4 + 3] = w;
            region[v] = r.index;
          }
        }
      } else if (deepGlow.has("cerebellum")) {
        const r = deepGlow.get("cerebellum");
        tmp.set(r.color);
        for (let v = 0; v < region.length; v++) {
          highlight.set([tmp.r, tmp.g, tmp.b, 0.85], v * 4);
          region[v] = r.index;
        }
      }
      c.geometry.attributes.aHighlight.needsUpdate = true;
      c.geometry.attributes.aRegion.needsUpdate = true;
    }
    for (const s of this.structures.values()) {
      const item = s.region ? deepGlow.get(s.region) : null;
      s.material.uniforms.uGlow.value = item ? 0.85 : 0;
      if (item) s.material.uniforms.uGlowColor.value.set(item.color);
      s.pulseIndex = item ? item.index : -1;
    }
    this.highlightIndex = new Map(list.map((item) => [item.id, item.index]));
  }

  /** Intensità degli impulsi (0…1) per indice di regione: la vista li anima per qualche secondo. */
  setPulse(values) {
    for (const c of this.cortex) c.material.uniforms.uPulse.value.set(values.subarray(0, MAX_PULSES));
    for (const s of this.structures.values()) s.material.uniforms.uPulse.value = s.pulseIndex >= 0 ? values[s.pulseIndex] : 0;
  }

  /** Regione scelta con un clic: un alone chiaro attorno a un punto (corteccia) o il bagliore della struttura. */
  setSelection(selection) {
    const radius = selection?.point ? (selection.radius ?? 10) * MM : 0;
    for (const c of this.cortex) {
      c.material.uniforms.uSelRadius.value = c.id === "cerebellum" && selection?.id !== "cerebellum" ? 0 : radius;
      if (selection?.point) c.material.uniforms.uSelPos.value.fromArray(selection.point);
      if (selection?.id === "cerebellum") c.material.uniforms.uSelRadius.value = c.id === "cerebellum" ? 50 : 0;
    }
    for (const s of this.structures.values()) {
      const selected = selection && s.region === selection.id;
      if (selected) {
        s.material.uniforms.uGlow.value = Math.max(s.material.uniforms.uGlow.value, 0.65);
        s.material.uniforms.uGlowColor.value.set("#e0f2fe");
      } else if (s.wasSelected) {
        // torna l'evidenziazione del tab (setHighlights la riscrive tutta: qui basta spegnere)
        const index = s.region ? this.highlightIndex?.get(s.region) : undefined;
        s.material.uniforms.uGlow.value = index !== undefined ? 0.85 : 0;
      }
      s.wasSelected = selected;
    }
  }

  /** Una struttura profonda semitrasparente (es. il talamo, per vedere il pulvinar dentro). */
  setStructureSeeThrough(id, on) {
    const s = this.structures.get(id);
    if (!s || s.glass) return;
    const material = s.material;
    const want = on ? 0.45 : 1;
    if (material.uniforms.uOpacity.value === want) return;
    material.uniforms.uOpacity.value = want;
    material.transparent = on;
    material.depthWrite = !on;
    material.needsUpdate = true;
    s.mesh.renderOrder = on ? 1 : 0;
  }

  /**
   * Cervello in miniatura (scala Neuroni) con il rendering software: al posto della corteccia con i giri
   * (decine di migliaia di triangoli per un oggetto piccolo sullo schermo) una versione liscia e rada,
   * costruita la prima volta che serve. Con una GPU la miniatura resta quella vera.
   */
  setMiniature(on, { keep = null } = {}) {
    if (on && !this.mini) {
      const mesh = toScene(miniCortexMesh());
      const geometry = geometryOf(mesh);
      // semitrasparente: dentro si vedono le strutture profonde collegate alle colonne del circuito
      const material = solidMaterial(CORTEX_COLORS.base, { specular: 0.1, opacity: 0.5 });
      material.side = THREE.FrontSide;
      material.clippingPlanes = this.clipPlanes;
      this.mini = new THREE.Mesh(geometry, material);
      this.mini.name = "cortex-mini";
      this.mini.visible = false;
      this.root.add(this.mini);
      this.geometries.add(geometry);
      this.materials.add(material);
    }
    if (!this.mini) return;
    this.mini.visible = on;
    for (const c of this.cortex) {
      c.mesh.visible = !on && this.opacity > 0.01;
      c.depth.visible = !on && this.opacity < 0.99 && this.opacity > 0.01;
    }
    // nella miniatura solo le strutture che servono al circuito (ogni oggetto in più è un disegno in più)
    for (const s of this.structures.values()) s.mesh.visible = !on || (s.region !== null && Boolean(keep?.has(s.region)));
  }

  /** Box della scena di tutto il cervello (per inquadrare e per i confini della navigazione). */
  box(target = new THREE.Box3()) {
    this.root.updateMatrixWorld();
    target.makeEmpty();
    for (const c of this.cortex) target.union(c.geometry.boundingBox);
    for (const s of this.structures.values()) target.union(s.geometry.boundingBox);
    return target.applyMatrix4(this.root.matrixWorld);
  }

  dispose() {
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    this.root.removeFromParent();
  }
}

const smooth = (t) => {
  const x = Math.min(1, Math.max(0, t));
  return x * x * (3 - 2 * x);
};
