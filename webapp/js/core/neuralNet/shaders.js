/**
 * Shader della rete orizzontale: neuroni (punti), lastre-heatmap, collegamenti, fantasmi, etichette.
 *
 * Perché shader propri: con decine di migliaia di neuroni e ~150.000 segmenti non si può toccare un
 * buffer a ogni frame. Tutto ciò che cambia nel tempo passa da poche uniform condivise da TUTTI i
 * materiali (un solo oggetto `uniforms`, gli stessi riferimenti):
 *   uFront / uWaveWidth / uGlow  l'onda di attivazione: il fronte è una x; ogni vertice confronta la
 *                                sua x (attributo position) con il fronte → nessun lavoro sulla CPU
 *   uSlice                       la fetta di profondità scelta (intervallo normalizzato 0…1)
 *   uState                       una texture RGBA8 con una colonna per sezione: R = scala dei punti
 *                                (LOD), G = opacità della lastra-heatmap, B = enfasi (sezione scelta),
 *                                A = opacità dei collegamenti. Il LOD scrive pochi byte, non i buffer.
 * Un vertice "spento" (sezione nascosta dal LOD) esce dal clip space: la GPU non rasterizza niente.
 *
 * Colori nello spazio dello schermo (sRGB), come le texture delle heatmap: niente conversioni.
 */
import * as THREE from "three";
import { ACTIVATION } from "./roles.js";

/** Componenti 0…1 di un colore #rrggbb, senza conversioni di spazio colore. */
export function rgb(hex) {
  const v = typeof hex === "number" ? hex : parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}
const vec3 = (hex) => new THREE.Vector3(...rgb(hex));

/** Le uniform condivise da tutti i materiali della rete (un'assegnazione le aggiorna ovunque). */
export function createSharedUniforms(stateTexture) {
  return {
    uState: { value: stateTexture },
    uFront: { value: 1e6 }, // a riposo il fronte è "lontanissimo": tutto è già calcolato
    uWaveWidth: { value: 2 },
    uGlow: { value: 1 },
    uSlice: { value: new THREE.Vector2(1, 0) }, // lo > hi = nessuna fetta scelta
    uSliceDim: { value: 0.2 },
    uLineSliceDim: { value: 0.07 },
    uPointScale: { value: 400 },
    uMaxPoint: { value: 26 }, // pixel del framebuffer (scene.update lo moltiplica per il pixel ratio)
    uPos: { value: vec3(ACTIVATION.positive) },
    uNeg: { value: vec3(ACTIVATION.negative) },
    uZero: { value: vec3(ACTIVATION.zero) },
    uPending: { value: vec3(ACTIVATION.pending) },
    uProb: { value: vec3(ACTIVATION.probability) },
    uOneHot: { value: vec3(ACTIVATION.onehot) },
    uOff: { value: vec3(ACTIVATION.off) },
    uGlowColor: { value: vec3(ACTIVATION.glow) },
  };
}

const COMMON = /* glsl */ `
uniform sampler2D uState;
uniform float uFront;
uniform float uWaveWidth;
uniform float uGlow;
uniform vec2 uSlice;
vec4 sectionState(float s) { return texelFetch(uState, ivec2(int(s + 0.5), 0), 0); }
// 0 prima che l'onda arrivi, 1 dopo: la x del vertice confrontata con il fronte
float wavePassed(float x) { return smoothstep(x - uWaveWidth, x, uFront); }
float waveGlow(float x) { float t = (uFront - x) / uWaveWidth; return uGlow * exp(-2.0 * t * t); }
// 1 se la posizione in profondità u (0…1, -1 = senza profondità) è nella fetta scelta
float inSlice(float u) { return (uSlice.x > uSlice.y || u < 0.0) ? 1.0 : step(uSlice.x, u) * step(u, uSlice.y); }
const vec4 OUTSIDE = vec4(2.0, 2.0, 2.0, 1.0);
`;

// ------------------------------------------------------------------------------------------ neuroni
const NEURON_VERTEX = /* glsl */ `
${COMMON}
attribute float aValue;
attribute float aMode;
attribute float aSection;
attribute float aDepthU;
attribute float aRadius;
uniform float uPointScale;
uniform float uMaxPoint;
uniform float uSliceDim;
uniform vec3 uPos;
uniform vec3 uNeg;
uniform vec3 uZero;
uniform vec3 uPending;
uniform vec3 uProb;
uniform vec3 uOneHot;
uniform vec3 uOff;
uniform vec3 uGlowColor;
varying vec3 vColor;
void main() {
  vec4 st = sectionState(aSection);
  float slice = inSlice(aDepthU);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // diametro vero di 2r unità del mondo: gl_PointSize = 2r · (altezza/2) / tan(fov/2) / distanza
  float size = 2.0 * aRadius * uPointScale * st.r * mix(0.62, 1.0, slice) / max(1e-4, -mv.z);
  if (st.r < 0.01 || size < 0.7) {
    gl_Position = OUTSIDE;
    gl_PointSize = 0.0;
    vColor = vec3(0.0);
    return;
  }
  gl_Position = projectionMatrix * mv;
  // da molto vicino i neuroni smettono di crescere: un disco enorme costa (riempimento) e coprirebbe gli archi
  gl_PointSize = min(size, uMaxPoint);
  float a = clamp(abs(aValue), 0.0, 1.0);
  vec3 c;
  if (aMode < 0.5) c = mix(uZero, aValue < 0.0 ? uNeg : uPos, pow(a, 0.6));
  else if (aMode < 1.5) c = mix(uZero, uProb, a);
  else if (aMode < 2.5) c = mix(uZero, uOneHot, a);
  else c = uOff;
  c = mix(uPending, c, wavePassed(position.x));
  float glow = aMode > 2.5 ? 0.0 : waveGlow(position.x) * (0.3 + 0.7 * a);
  c = mix(c, uGlowColor, clamp(glow, 0.0, 0.85));
  vColor = mix(uPending, c, st.b * mix(uSliceDim, 1.0, slice));
}
`;

const NEURON_FRAGMENT = /* glsl */ `
varying vec3 vColor;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float d = dot(p, p);
  if (d > 1.0) discard;
  // bordo più scuro: un disco piatto sembra una pallina anche senza luci
  gl_FragColor = vec4(vColor * mix(1.0, 0.62, smoothstep(0.3, 1.0, d)), 1.0);
}
`;

// ------------------------------------------------------------------------------------------ lastre-heatmap
const SLAB_VERTEX = /* glsl */ `
${COMMON}
attribute float aSection;
attribute float aDepthU;
varying vec2 vUv;
varying float vDepthU;
varying float vAlpha;
varying float vPassed;
varying float vGlow;
varying float vEmph;
void main() {
  vec4 st = sectionState(aSection);
  vAlpha = st.g;
  if (vAlpha < 0.01) {
    gl_Position = OUTSIDE;
    return;
  }
  vUv = uv;
  vDepthU = aDepthU;
  vEmph = st.b;
  vPassed = wavePassed(position.x);
  vGlow = waveGlow(position.x);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SLAB_FRAGMENT = /* glsl */ `
uniform sampler2D uAtlas;
uniform vec3 uPending;
uniform vec3 uGlowColor;
uniform vec2 uSlice;
uniform float uSliceDim;
varying vec2 vUv;
varying float vDepthU;
varying float vAlpha;
varying float vPassed;
varying float vGlow;
varying float vEmph;
void main() {
  vec3 c = mix(uPending, texture2D(uAtlas, vUv).rgb, vPassed);
  c = mix(c, uGlowColor, clamp(vGlow * 0.6, 0.0, 0.7));
  float slice = (uSlice.x > uSlice.y || vDepthU < 0.0) ? 1.0 : step(uSlice.x, vDepthU) * step(vDepthU, uSlice.y);
  gl_FragColor = vec4(mix(uPending, c, vEmph * mix(uSliceDim, 1.0, slice)), vAlpha);
}
`;

// ------------------------------------------------------------------------------------------ collegamenti
// Un segmento si spegne solo con valori uguali nei suoi due vertici (sezione e fetta del segmento):
// spostare fuori dal clip un vertice solo disegnerebbe una linea verso il bordo dello schermo.
const LINE_VERTEX = /* glsl */ `
${COMMON}
attribute vec4 aColor;
attribute float aSection;
attribute float aSection2;
attribute float aDepthU;
uniform float uOpacity;
uniform float uLineSliceDim;
uniform vec3 uGlowColor;
varying vec4 vColor;
void main() {
  // LOD dalla sezione di arrivo; enfasi dalla più "accesa" delle due sezioni che il collegamento tocca
  vec4 st = sectionState(aSection);
  float emph = max(st.b, sectionState(aSection2).b);
  float alpha = aColor.a * st.a * uOpacity * mix(uLineSliceDim, 1.0, inSlice(aDepthU)) * mix(0.12, 1.0, emph * emph);
  if (alpha < 0.004) {
    gl_Position = OUTSIDE;
    vColor = vec4(0.0);
    return;
  }
  float glow = waveGlow(position.x);
  alpha *= wavePassed(position.x);
  vColor = vec4(mix(aColor.rgb, uGlowColor, clamp(glow * 0.75, 0.0, 0.85)), clamp(alpha * (1.0 + 1.5 * glow), 0.0, 1.0));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const LINE_FRAGMENT = /* glsl */ `
varying vec4 vColor;
void main() {
  gl_FragColor = vColor;
}
`;

// ------------------------------------------------------------------------------------------ fantasmi e strutture
const GHOST_VERTEX = /* glsl */ `
${COMMON}
attribute vec4 aColor;
uniform vec3 uGlowColor;
varying vec4 vColor;
void main() {
  float glow = waveGlow(position.x);
  vColor = vec4(mix(aColor.rgb, uGlowColor, clamp(glow * 0.7, 0.0, 0.8)), clamp(aColor.a * (1.0 + 2.5 * glow), 0.0, 1.0));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// ------------------------------------------------------------------------------------------ etichette
// Un rettangolo istanziato per etichetta, sempre rivolto alla camera (come uno sprite), con la
// dimensione nel mondo scritta per istanza (labels.js la tiene tra un minimo e un massimo di pixel).
const LABEL_VERTEX = /* glsl */ `
attribute vec3 aCenter;
attribute vec2 aSize;
attribute vec2 aAnchor;
attribute vec4 aUv;
attribute float aAlpha;
varying vec2 vUv;
varying float vAlpha;
void main() {
  vAlpha = aAlpha;
  if (aAlpha < 0.01) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }
  vec4 mv = modelViewMatrix * vec4(aCenter, 1.0);
  mv.xy += (position.xy - aAnchor) * aSize;
  gl_Position = projectionMatrix * mv;
  vUv = mix(aUv.xy, aUv.zw, position.xy);
}
`;

const LABEL_FRAGMENT = /* glsl */ `
uniform sampler2D uAtlas;
varying vec2 vUv;
varying float vAlpha;
void main() {
  vec4 t = texture2D(uAtlas, vUv);
  float a = t.a * vAlpha;
  if (a < 0.02) discard;
  gl_FragColor = vec4(t.rgb, a);
}
`;

const material = (uniforms, vertexShader, fragmentShader, options = {}) =>
  new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader, ...options });

export const neuronMaterial = (uniforms) => material(uniforms, NEURON_VERTEX, NEURON_FRAGMENT);

export const slabMaterial = (uniforms, atlas) =>
  material({ ...uniforms, uAtlas: { value: atlas } }, SLAB_VERTEX, SLAB_FRAGMENT, { transparent: true });

/** Collegamenti: `opacity` è un'uniform propria (la densità li rende più o meno trasparenti). */
export const lineMaterial = (uniforms, opacity = 1) =>
  material({ ...uniforms, uOpacity: { value: opacity } }, LINE_VERTEX, LINE_FRAGMENT, { transparent: true, depthWrite: false });

export const ghostMaterial = (uniforms, { lines = false } = {}) =>
  material(uniforms, GHOST_VERTEX, LINE_FRAGMENT, { transparent: true, depthWrite: false, side: lines ? THREE.FrontSide : THREE.DoubleSide });

export const labelMaterial = (atlas) =>
  material({ uAtlas: { value: atlas } }, LABEL_VERTEX, LABEL_FRAGMENT, { transparent: true, depthTest: false, depthWrite: false });
