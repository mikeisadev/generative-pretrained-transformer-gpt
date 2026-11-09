/**
 * Neuroni 3D del microcircuito (three.js): tutte le morfologie (morphology.js) in pochi disegni
 * (segmenti che si assottigliano per dendriti, assoni, mielina, nodi di Ranvier e stereociglia; sfere
 * per somi, bottoni, spine e contatti) e gli impulsi luminosi calcolati nello shader.
 *
 *   const field = new NeuronField(layout, { quality: "high" });  // layout di circuitLayout.js
 *   scene.add(field.root); field.setPlaying(true); field.setTime(t);  // una uniform per frame
 *
 * Due modi, stesso aspetto:
 *  - con una GPU, InstancedMesh: cilindri veri e icosaedri, una matrice per pezzo;
 *  - con il rendering software (SwiftShader), geometrie "cotte": tutti i pezzi in un unico buffer, i
 *    segmenti come nastri rivolti alla camera con l'ombreggiatura di un cilindro (2 triangoli invece di
 *    8). Misurato: con SwiftShader ogni istanza costa ~18 µs anche se ha pochi triangoli (7.000 segmenti
 *    istanziati = ~130 ms per frame); cotti in un buffer solo costano come i triangoli che sono.
 * L'animazione (scarico di ogni gruppo, impulso che corre lungo l'assone e salta tra i nodi di Ranvier,
 * lampo dei bottoni all'arrivo) è una funzione del tempo nello shader: in JavaScript solo `uTime`.
 */
import * as THREE from "three";
import { place } from "./circuitLayout.js";
import { SEG, SPH } from "./morphology.js";
import { EDGE_KINDS } from "./spec.js";

/** Colori dei pezzi che non dipendono dalla famiglia del neurone. */
export const PART_COLORS = Object.freeze({ myelin: "#f4f1ea", node: "#fcd34d", cilium: "#fde68a", outer: "#f9a8d4", glow: "#fff4c2" });

/** Tempo (s) che un impulso impiega lungo una fibra tra due gruppi. */
export const FIBER_TRAVEL = 0.42;
/** Ritardo (s) tra lo scarico del neurone e la partenza dell'impulso sulla fibra verso il gruppo dopo. */
export const FIBER_DELAY = 0.32;

const CONTACT = 5; // tipo di sfera: contatto di una fibra tra gruppi

// ---------------------------------------------------------------- shader
const UNIFORMS = /* glsl */ `
uniform float uTime;
uniform float uPeriod;
uniform float uPlay;
uniform float uSpeed;
uniform float uFiber;
uniform float uTravel;
`;

/** Bagliore di un segmento a un certo istante (comune ai cilindri e ai nastri). */
const SEGMENT_GLOW = /* glsl */ `
float segmentGlow(vec4 seg, float t) {
  if (uPlay < 0.5) return 0.0;
  float kind = seg.x;
  float since = mod(uTime - seg.w, uPeriod); // secondi da quando questo neurone (o questa fibra) ha scaricato
  float arc = mix(seg.y, seg.z, t);
  if (uFiber > 0.5) {
    // fibra tra gruppi: arc in [0, 1], l'impulso la percorre in uTravel secondi
    return exp(-pow((arc - since / uTravel) / 0.07, 2.0)) * step(since, uTravel + 0.2);
  }
  if (kind > 0.5 && kind < 3.5) {
    // assone: un fronte che corre; nella mielina è debole, nei nodi di Ranvier forte (conduzione saltatoria)
    float g = exp(-pow((arc - since * uSpeed) / 0.07, 2.0));
    return g * (kind < 1.5 ? 1.0 : (kind < 2.5 ? 0.22 : 1.8));
  }
  if (kind < 0.5 || (kind > 3.5 && kind < 4.5)) {
    // dendriti: si accendono un attimo prima dello scarico (gli ingressi che arrivano), dalle punte al soma
    float before = mod(seg.w - uTime, uPeriod);
    return 0.6 * exp(-pow((before - 0.12 - arc * 0.35) / 0.1, 2.0));
  }
  return 0.0;
}
`;

const SEGMENT_VERTEX = /* glsl */ `
#include <common>
attribute vec2 aRadius;
attribute vec4 aSeg;
attribute vec3 aColor;
#ifdef BAKED
attribute vec3 aA;
attribute vec3 aB;
#endif
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vColor;
varying float vGlow;
varying float vU;
varying vec3 vFacing;
${UNIFORMS}
${SEGMENT_GLOW}
void main() {
  #ifdef BAKED
  // nastro rivolto alla camera tra A e B: position.x = lato (−1…1), position.y = lungo il segmento (0…1)
  vec3 A = (modelMatrix * vec4(aA, 1.0)).xyz;
  vec3 B = (modelMatrix * vec4(aB, 1.0)).xyz;
  float t = position.y;
  vec3 P = mix(A, B, t);
  vec3 V = normalize(cameraPosition - P);
  vec3 S = normalize(cross(normalize(B - A + vec3(1e-6)), V) + vec3(1e-6, 0.0, 0.0));
  P += S * position.x * mix(aRadius.x, aRadius.y, t);
  vec4 mv = viewMatrix * vec4(P, 1.0);
  vNormal = normalize((viewMatrix * vec4(S, 0.0)).xyz);
  vFacing = normalize((viewMatrix * vec4(V, 0.0)).xyz);
  vU = position.x;
  #else
  // cilindro che si assottiglia: il raggio si applica qui, la matrice d'istanza ha scala (1, L, 1)
  float t = position.y + 0.5;
  float r = mix(aRadius.x, aRadius.y, t);
  vec4 mv = viewMatrix * modelMatrix * instanceMatrix * vec4(position.x * r, position.y, position.z * r, 1.0);
  vNormal = normalize((viewMatrix * modelMatrix * instanceMatrix * vec4(position.x, 0.0, position.z, 0.0)).xyz);
  vFacing = vec3(0.0);
  vU = 2.0;
  #endif
  vViewPosition = -mv.xyz;
  vColor = aColor;
  vGlow = segmentGlow(aSeg, t);
  gl_Position = projectionMatrix * mv;
}
`;

const SPHERE_VERTEX = /* glsl */ `
#include <common>
attribute vec3 aSph;
attribute vec3 aColor;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vColor;
varying float vGlow;
varying float vU;
varying vec3 vFacing;
${UNIFORMS}
void main() {
  #ifdef BAKED
  vec4 mv = viewMatrix * modelMatrix * vec4(position, 1.0);
  vNormal = normalize((viewMatrix * modelMatrix * vec4(normal, 0.0)).xyz);
  #else
  vec4 mv = viewMatrix * modelMatrix * instanceMatrix * vec4(position, 1.0);
  vNormal = normalize((viewMatrix * modelMatrix * instanceMatrix * vec4(normal, 0.0)).xyz);
  #endif
  vViewPosition = -mv.xyz;
  vColor = aColor;
  vU = 2.0;
  vFacing = vec3(0.0);
  float glow = 0.0;
  if (uPlay > 0.5) {
    float kind = aSph.x;
    float since = mod(uTime - aSph.z, uPeriod);
    if (kind < 0.5) glow = 1.2 * exp(-pow(since / 0.11, 2.0)); // soma: il lampo dello scarico
    else if (kind < 1.5 || (kind > 2.5 && kind < 3.5)) glow = 1.4 * exp(-pow((aSph.y - since * uSpeed) / 0.06, 2.0)); // bottoni e varicosità
    else if (kind > 4.5) glow = 1.5 * exp(-pow((since - uTravel) / 0.07, 2.0)); // contatto di una fibra tra gruppi
    else {
      float before = mod(aSph.z - uTime, uPeriod);
      glow = 0.5 * exp(-pow((before - 0.15) / 0.1, 2.0)); // spine e artigli: con i dendriti
    }
  }
  vGlow = glow;
  gl_Position = projectionMatrix * mv;
}
`;

const FRAGMENT = /* glsl */ `
#include <common>
uniform vec3 uGlowColor;
uniform float uDim;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vColor;
varying float vGlow;
varying float vU;
varying vec3 vFacing;
void main() {
  // sul nastro (vU in −1…1) la normale di un cilindro si ricostruisce: di lato verso i bordi, verso la
  // camera al centro; per cilindri e sfere veri (vU = 2) è quella interpolata
  vec3 n = vU < 1.5 ? normalize(vNormal * vU + vFacing * sqrt(max(0.0, 1.0 - vU * vU))) : normalize(vNormal);
  vec3 v = normalize(vViewPosition);
  vec3 L = normalize(vec3(-0.35, 0.55, 0.78));
  float wrap = max((dot(n, L) + 0.45) / 1.45, 0.0);
  float rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
  vec3 color = vColor * (0.3 + 0.75 * wrap) + vColor * rim * 0.25;
  color = color * uDim + uGlowColor * vGlow;
  gl_FragColor = vec4(color, 1.0);
  #include <colorspace_fragment>
}
`;

function material(vertexShader, { fiber = false, baked = false } = {}) {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader: FRAGMENT,
    defines: baked ? { BAKED: "" } : {},
    side: baked ? THREE.DoubleSide : THREE.FrontSide, // il nastro non ha un "dietro"
    uniforms: {
      uTime: { value: 0 },
      uPeriod: { value: 10 },
      uPlay: { value: 0 },
      uSpeed: { value: 1.6 },
      uFiber: { value: fiber ? 1 : 0 },
      uTravel: { value: FIBER_TRAVEL },
      uGlowColor: { value: new THREE.Color(PART_COLORS.glow) },
      uDim: { value: 1 },
    },
  });
}

// ---------------------------------------------------------------- contenitori dei pezzi
const UP = new THREE.Vector3(0, 1, 0);

/** Segmenti istanziati (GPU): un cilindro aperto per pezzo. */
class InstancedSegments {
  constructor(count, mat, radial) {
    const geometry = new THREE.CylinderGeometry(1, 1, 1, radial, 1, true);
    for (const [name, size] of [["aRadius", 2], ["aSeg", 4], ["aColor", 3]]) geometry.setAttribute(name, new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * size), size));
    this.mesh = new THREE.InstancedMesh(geometry, mat, Math.max(1, count));
    this.mesh.count = count;
    this.triangles = radial * 2;
    this.m = new THREE.Matrix4();
    this.q = new THREE.Quaternion();
    this.mid = new THREE.Vector3();
    this.scale = new THREE.Vector3();
    this.dir = new THREE.Vector3();
  }

  put(i, a, b, ra, rb, seg, rgb) {
    this.dir.subVectors(b, a);
    const length = Math.max(1e-5, this.dir.length());
    this.q.setFromUnitVectors(UP, this.dir.multiplyScalar(1 / length));
    this.m.compose(this.mid.addVectors(a, b).multiplyScalar(0.5), this.q, this.scale.set(1, length, 1));
    this.mesh.setMatrixAt(i, this.m);
    const at = this.mesh.geometry.attributes;
    at.aRadius.setXY(i, ra, rb);
    at.aSeg.setXYZW(i, seg[0], seg[1], seg[2], seg[3]);
    at.aColor.setXYZ(i, rgb.r, rgb.g, rgb.b);
  }
}

/** Segmenti cotti (rendering software): 4 vertici e 2 triangoli per pezzo, tutti in un buffer. */
class BakedSegments {
  constructor(count, mat) {
    const n = Math.max(1, count) * 4;
    this.arrays = { position: new Float32Array(n * 3), aA: new Float32Array(n * 3), aB: new Float32Array(n * 3), aRadius: new Float32Array(n * 2), aSeg: new Float32Array(n * 4), aColor: new Float32Array(n * 3) };
    const index = new Uint32Array(Math.max(1, count) * 6);
    for (let i = 0; i < count; i++) {
      const v = i * 4;
      index.set([v, v + 1, v + 2, v + 2, v + 1, v + 3], i * 6);
      this.arrays.position.set([-1, 0, 0, 1, 0, 0, -1, 1, 0, 1, 1, 0], v * 3);
    }
    const geometry = new THREE.BufferGeometry();
    for (const [name, array] of Object.entries(this.arrays)) geometry.setAttribute(name, new THREE.BufferAttribute(array, array.length / n));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.setDrawRange(0, count * 6);
    this.mesh = new THREE.Mesh(geometry, mat);
    this.triangles = 2;
  }

  put(i, a, b, ra, rb, seg, rgb) {
    const { aA, aB, aRadius, aSeg, aColor } = this.arrays;
    for (let k = 0; k < 4; k++) {
      const v = i * 4 + k;
      aA[v * 3] = a.x;
      aA[v * 3 + 1] = a.y;
      aA[v * 3 + 2] = a.z;
      aB[v * 3] = b.x;
      aB[v * 3 + 1] = b.y;
      aB[v * 3 + 2] = b.z;
      aRadius[v * 2] = ra;
      aRadius[v * 2 + 1] = rb;
      aSeg.set(seg, v * 4);
      aColor[v * 3] = rgb.r;
      aColor[v * 3 + 1] = rgb.g;
      aColor[v * 3 + 2] = rgb.b;
    }
  }
}

/** Sfere istanziate (GPU). */
class InstancedSpheres {
  constructor(count, mat, template) {
    for (const [name, size] of [["aSph", 3], ["aColor", 3]]) template.setAttribute(name, new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, count) * size), size));
    this.mesh = new THREE.InstancedMesh(template, mat, Math.max(1, count));
    this.mesh.count = count;
    this.triangles = template.attributes.position.count / 3;
    this.m = new THREE.Matrix4();
    this.p = new THREE.Vector3();
    this.s = new THREE.Vector3();
    this.q = new THREE.Quaternion();
  }

  put(i, p, r, sph, rgb) {
    this.m.compose(this.p.fromArray(p), this.q, this.s.set(r, r, r));
    this.mesh.setMatrixAt(i, this.m);
    this.mesh.geometry.attributes.aSph.setXYZ(i, sph[0], sph[1], sph[2]);
    this.mesh.geometry.attributes.aColor.setXYZ(i, rgb.r, rgb.g, rgb.b);
  }
}

/** Sfere cotte (rendering software): i vertici del modello spostati e scalati qui, una volta. */
class BakedSpheres {
  constructor(count, mat, template) {
    this.template = template.attributes.position.array;
    this.templateNormals = template.attributes.normal.array;
    this.per = this.template.length / 3;
    const n = Math.max(1, count) * this.per;
    this.arrays = { position: new Float32Array(n * 3), normal: new Float32Array(n * 3), aSph: new Float32Array(n * 3), aColor: new Float32Array(n * 3) };
    const geometry = new THREE.BufferGeometry();
    for (const [name, array] of Object.entries(this.arrays)) geometry.setAttribute(name, new THREE.BufferAttribute(array, 3));
    geometry.setDrawRange(0, count * this.per);
    template.dispose();
    this.mesh = new THREE.Mesh(geometry, mat);
    this.triangles = this.per / 3;
  }

  put(i, p, r, sph, rgb) {
    const { position, normal, aSph, aColor } = this.arrays;
    const base = i * this.per;
    for (let k = 0; k < this.per; k++) {
      const v = (base + k) * 3;
      position[v] = p[0] + this.template[k * 3] * r;
      position[v + 1] = p[1] + this.template[k * 3 + 1] * r;
      position[v + 2] = p[2] + this.template[k * 3 + 2] * r;
      normal[v] = this.templateNormals[k * 3];
      normal[v + 1] = this.templateNormals[k * 3 + 1];
      normal[v + 2] = this.templateNormals[k * 3 + 2];
      aSph[v] = sph[0];
      aSph[v + 1] = sph[1];
      aSph[v + 2] = sph[2];
      aColor[v] = rgb.r;
      aColor[v + 1] = rgb.g;
      aColor[v + 2] = rgb.b;
    }
  }
}

// ---------------------------------------------------------------- campo di neuroni
/**
 * Tutti i neuroni e le fibre di un microcircuito.
 * @param {ReturnType<import("./circuitLayout.js").layoutCircuit>} layout
 * @param {{ quality?: "high" | "low", period?: number }} [o]
 */
export class NeuronField {
  constructor(layout, { quality = "high", period = 10 } = {}) {
    this.layout = layout;
    this.root = new THREE.Group();
    this.root.name = "neurons";
    this.materials = [];
    const baked = quality === "low";

    // ---------------- conteggio, per allocare una volta sola
    let segCount = 0;
    let somaCount = 0;
    let smallCount = 0;
    for (const cell of layout.cells) {
      segCount += cell.morph.segments.length / 11;
      for (let i = 0; i < cell.morph.spheres.length; i += 6) {
        if (cell.morph.spheres[i + 4] === SPH.soma) somaCount++;
        else smallCount++;
      }
    }
    const fiberSamples = baked ? 14 : 22;
    const fiberSegCount = layout.fibers.length * (fiberSamples - 1);
    smallCount += layout.fibers.length; // un contatto per fibra

    const segMat = material(SEGMENT_VERTEX, { baked });
    const fiberMat = material(SEGMENT_VERTEX, { fiber: true, baked });
    const sphereMat = material(SPHERE_VERTEX, { baked });
    const smallMat = material(SPHERE_VERTEX, { baked });
    this.materials.push(segMat, fiberMat, sphereMat, smallMat);
    const Segments = baked ? BakedSegments : InstancedSegments;
    const Spheres = baked ? BakedSpheres : InstancedSpheres;
    const segments = new Segments(segCount, segMat, 7);
    const fibers = new Segments(fiberSegCount, fiberMat, 6);
    // somi rotondi (icosaedro), bottoni, spine e contatti come ottaedri: piccoli sullo schermo e tanti
    const somata = new Spheres(somaCount, sphereMat, new THREE.IcosahedronGeometry(1, baked ? 1 : 2));
    const small = new Spheres(smallCount, smallMat, new THREE.OctahedronGeometry(1, baked ? 0 : 1));

    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const color = new THREE.Color();
    const tmp = new THREE.Color();
    const white = new THREE.Color("#ffffff");
    const family = new Map(layout.columns.map((c) => [c.index, c.color]));
    let si = 0;
    let bi = 0;
    let ri = 0;
    for (const cell of layout.cells) {
      const base = color.set(family.get(cell.column) ?? "#e2e8f0");
      const sc = cell.scale;
      const seg = cell.morph.segments;
      for (let i = 0; i < seg.length; i += 11) {
        a.fromArray(place(cell, [seg[i], seg[i + 1], seg[i + 2]]));
        b.fromArray(place(cell, [seg[i + 3], seg[i + 4], seg[i + 5]]));
        const kind = seg[i + 8];
        const rgb =
          kind === SEG.myelin ? tmp.set(PART_COLORS.myelin) : kind === SEG.node ? tmp.set(PART_COLORS.node) : kind === SEG.cilium ? tmp.set(PART_COLORS.cilium) : kind === SEG.outer ? tmp.set(PART_COLORS.outer) : kind === SEG.axon ? tmp.copy(base).multiplyScalar(0.92) : base;
        segments.put(si++, a, b, seg[i + 6] * sc, seg[i + 7] * sc, [kind, seg[i + 9], seg[i + 10], cell.fire], rgb);
      }
      const sph = cell.morph.spheres;
      for (let i = 0; i < sph.length; i += 6) {
        const kind = sph[i + 4];
        const p = place(cell, [sph[i], sph[i + 1], sph[i + 2]]);
        const rgb = kind === SPH.bouton || kind === SPH.varicosity ? tmp.copy(base).lerp(white, 0.25) : base;
        if (kind === SPH.soma) somata.put(bi++, p, sph[i + 3] * sc, [kind, sph[i + 5], cell.fire], rgb);
        else small.put(ri++, p, sph[i + 3] * sc * (kind === SPH.spine || kind === SPH.claw ? 1 : 1.15), [kind, sph[i + 5], cell.fire], rgb);
      }
    }

    // ---------------- fibre tra i gruppi (colore = tipo di sinapsi) e il contatto all'arrivo
    let fi = 0;
    for (const fiber of layout.fibers) {
      const curve = new THREE.CatmullRomCurve3(fiber.points.map((p) => new THREE.Vector3(...p)), false, "centripetal");
      const points = curve.getSpacedPoints(fiberSamples - 1);
      const rgb = tmp.set(EDGE_KINDS[fiber.kind]?.color ?? "#e2e8f0");
      const r = fiber.kind === "modulatory" ? 0.011 : 0.014;
      const fire = fiber.fire + FIBER_DELAY;
      for (let k = 0; k < points.length - 1; k++) fibers.put(fi++, points[k], points[k + 1], r, r, [SEG.axon, k / (points.length - 1), (k + 1) / (points.length - 1), fire], rgb);
      // contatto: bottone (eccitatoria), più largo sul soma (inibitoria a canestro), piccolo (modulatoria)
      const size = fiber.kind === "inhibitory" ? 0.034 : fiber.kind === "modulatory" ? 0.022 : 0.03;
      small.put(ri++, points[points.length - 1].toArray(), size * 1.2, [CONTACT, 0, fire], rgb);
    }

    this.meshes = { segments: segments.mesh, fibers: fibers.mesh, somata: somata.mesh, small: small.mesh };
    for (const mesh of Object.values(this.meshes)) {
      if (mesh.isInstancedMesh) mesh.instanceMatrix.needsUpdate = true;
      for (const attr of Object.values(mesh.geometry.attributes)) attr.needsUpdate = true;
      mesh.frustumCulled = false; // una sola istanza fuori schermo non deve far sparire tutto il campo
      this.root.add(mesh);
    }
    this.counts = {
      mode: baked ? "cotti" : "istanziati",
      segments: si,
      somata: bi,
      small: ri,
      fibers: fi,
      triangles: Math.round(si * segments.triangles + bi * somata.triangles + ri * small.triangles + fi * fibers.triangles),
    };
    this.setPeriod(period);
  }

  setPeriod(seconds) {
    for (const mat of this.materials) mat.uniforms.uPeriod.value = seconds;
  }

  setPlaying(on) {
    for (const mat of this.materials) mat.uniforms.uPlay.value = on ? 1 : 0;
  }

  setTime(t) {
    for (const mat of this.materials) mat.uniforms.uTime.value = t;
  }

  /** 1 = normale, < 1 = più scuro. */
  setDim(value) {
    for (const mat of this.materials) mat.uniforms.uDim.value = value;
  }

  dispose() {
    for (const mesh of Object.values(this.meshes)) {
      mesh.geometry.dispose();
      mesh.dispose?.();
    }
    for (const mat of this.materials) mat.dispose();
    this.root.removeFromParent();
  }
}
