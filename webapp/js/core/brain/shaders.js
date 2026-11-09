/**
 * Materiali del cervello 3D (three.js ShaderMaterial): pochi calcoli per pixel, niente ombre, niente
 * mappe. Perché materiali propri e non MeshStandardMaterial: con il rendering software (SwiftShader) il
 * modello fisico costa molto di più, e qui servono cose che lo standard non fa da solo:
 *  - luce "frontale" nello spazio della camera: il rilievo dei giri si legge da qualunque lato giri il
 *    cervello (come una lampada da tavolo puntata sul plastico);
 *  - corteccia trasparente senza artefatti: una passata di sola profondità (cortexDepthMaterial) e poi
 *    il colore con depthFunc LessEqual: si vede solo lo strato più vicino, mai le pieghe una sull'altra;
 *  - piani di taglio leggibili: le facce posteriori, che si vedono solo attraverso il taglio, hanno un
 *    colore pieno (la "sezione"), così il taglio sembra una fetta di cervello e non un guscio vuoto;
 *  - evidenziazione delle regioni del tab (colore per vertice calcolato una volta) e impulsi che
 *    viaggiano lungo i fasci (una uniform: nessun lavoro in JavaScript per frame).
 */
import * as THREE from "three";

/** Al più tante regioni pulsano insieme (indice per vertice → uniform). */
export const MAX_PULSES = 24;

/** Illuminazione comune: luce nello spazio della camera, luce avvolgente, riflesso e bordo. */
const SHADE = /* glsl */ `
vec3 brainShade(vec3 base, vec3 n, vec3 viewDir, float occlusion, float specular) {
  vec3 L = normalize(vec3(-0.35, 0.55, 0.78));
  float ndl = dot(n, L);
  float wrap = max((ndl + 0.4) / 1.4, 0.0);
  vec3 H = normalize(L + viewDir);
  float spec = pow(max(dot(n, H), 0.0), 36.0) * specular;
  float rim = pow(1.0 - max(dot(n, viewDir), 0.0), 2.5);
  vec3 fill = vec3(0.45, 0.55, 0.75) * max(-n.y, 0.0) * 0.12; // un filo di luce fredda dal basso
  return base * (0.24 + 0.82 * wrap) * occlusion + spec + rim * 0.16 * base + fill * base;
}
`;

const CORTEX_VERTEX = /* glsl */ `
#include <common>
#include <clipping_planes_pars_vertex>
attribute float aDepth;
attribute float aLobe;
attribute vec4 aHighlight;
attribute float aRegion;
uniform float uPulse[${MAX_PULSES}];
uniform vec3 uSelPos;
uniform float uSelRadius;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vDepth;
varying float vLobe;
varying vec4 vHighlight;
varying float vPulse;
varying float vSel;
void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  vNormal = normalize(normalMatrix * normal);
  vDepth = aDepth;
  vLobe = aLobe;
  vHighlight = aHighlight;
  int r = int(aRegion + 0.5);
  vPulse = (r >= 0 && r < ${MAX_PULSES}) ? uPulse[r] : 0.0;
  vec3 world = (modelMatrix * vec4(position, 1.0)).xyz;
  vSel = uSelRadius > 0.0 ? 1.0 - smoothstep(uSelRadius * 0.55, uSelRadius, distance(world, uSelPos)) : 0.0;
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;

const CORTEX_FRAGMENT = /* glsl */ `
#include <common>
#include <clipping_planes_pars_fragment>
uniform vec3 uBase;
uniform vec3 uSulcus;
uniform vec3 uLobeColors[6];
uniform float uLobeMix;
uniform float uHighlightMix;
uniform vec3 uSelColor;
uniform float uOpacity;
uniform vec3 uInterior;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying float vDepth;
varying float vLobe;
varying vec4 vHighlight;
varying float vPulse;
varying float vSel;
${SHADE}
void main() {
  #include <clipping_planes_fragment>
  if (!gl_FrontFacing) {
    // si vede solo attraverso un piano di taglio: la sezione, a colore pieno (sostanza bianca)
    gl_FragColor = vec4(uInterior, max(uOpacity, 0.92));
    #include <colorspace_fragment>
    return;
  }
  int lobe = int(vLobe + 0.5);
  vec3 lobeColor = uLobeColors[0];
  if (lobe == 1) lobeColor = uLobeColors[1];
  else if (lobe == 2) lobeColor = uLobeColors[2];
  else if (lobe == 3) lobeColor = uLobeColors[3];
  else if (lobe == 4) lobeColor = uLobeColors[4];
  else if (lobe == 5) lobeColor = uLobeColors[5];
  vec3 base = mix(uBase, lobeColor, uLobeMix);
  // fondo dei solchi più scuro: si legge la forma anche con la corteccia trasparente
  base = mix(base, base * uSulcus, smoothstep(0.25, 1.0, vDepth) * smoothstep(0.25, 1.0, vDepth));
  float glow = vHighlight.a * uHighlightMix;
  float pulse = vPulse;
  base = mix(base, vHighlight.rgb, clamp(glow * (0.72 + 0.28 * pulse), 0.0, 1.0));
  base = mix(base, uSelColor, vSel * 0.65);
  vec3 n = normalize(vNormal);
  vec3 viewDir = normalize(vViewPosition);
  float occlusion = 1.0 - 0.42 * smoothstep(0.15, 1.0, vDepth);
  vec3 color = brainShade(base, n, viewDir, occlusion, 0.16);
  color += vHighlight.rgb * glow * (0.12 + 0.5 * pulse);
  color += uSelColor * vSel * 0.18;
  // trasparente: un "cervello di vetro". Lo sfondo scuro che passa attraverso spegnerebbe il colore
  // (marrone e torbido): una tinta più chiara e fredda lo compensa; il bordo resta più opaco (la sagoma
  // non si perde) e il fondo dei solchi pure (il disegno dei giri resta leggibile)
  float clear = 1.0 - uOpacity;
  color = mix(color, color * 1.25 + vec3(0.05, 0.06, 0.08), clear);
  float rim = pow(1.0 - max(dot(n, viewDir), 0.0), 1.6);
  float alpha = uOpacity >= 0.999 ? 1.0 : clamp(uOpacity + clear * (rim * 0.7 + smoothstep(0.45, 1.0, vDepth) * 0.35) + glow * 0.3, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
  #include <colorspace_fragment>
}
`;

/**
 * Materiale della corteccia. Le uniform si cambiano da fuori (setCortexOpacity, evidenziazioni…).
 * @param {{ base: string, sulcus: string, interior: string, lobeColors: string[] }} colors
 */
export function cortexMaterial(colors) {
  return new THREE.ShaderMaterial({
    vertexShader: CORTEX_VERTEX,
    fragmentShader: CORTEX_FRAGMENT,
    clipping: true,
    uniforms: {
      uBase: { value: new THREE.Color(colors.base) },
      uSulcus: { value: new THREE.Color(colors.sulcus) },
      uLobeColors: { value: colors.lobeColors.map((c) => new THREE.Color(c)) },
      uLobeMix: { value: 0 },
      uHighlightMix: { value: 1 },
      uPulse: { value: new Float32Array(MAX_PULSES) },
      uSelPos: { value: new THREE.Vector3() },
      uSelRadius: { value: 0 },
      uSelColor: { value: new THREE.Color("#e0f2fe") },
      uOpacity: { value: 1 },
      uInterior: { value: new THREE.Color(colors.interior) },
    },
  });
}

/** Passata di sola profondità della corteccia trasparente (stessi piani di taglio del colore). */
export function cortexDepthMaterial() {
  return new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true });
}

const SOLID_VERTEX = /* glsl */ `
#include <common>
#include <clipping_planes_pars_vertex>
varying vec3 vNormal;
varying vec3 vViewPosition;
void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;

const SOLID_FRAGMENT = /* glsl */ `
#include <common>
#include <clipping_planes_pars_fragment>
uniform vec3 uColor;
uniform vec3 uGlowColor;
uniform float uGlow;
uniform float uPulse;
uniform float uOpacity;
uniform float uSpecular;
varying vec3 vNormal;
varying vec3 vViewPosition;
${SHADE}
void main() {
  #include <clipping_planes_fragment>
  vec3 base = mix(uColor, uGlowColor, uGlow * (0.65 + 0.35 * uPulse));
  if (!gl_FrontFacing) {
    gl_FragColor = vec4(base * 0.82, 1.0); // sezione di una struttura tagliata: colore pieno
    #include <colorspace_fragment>
    return;
  }
  vec3 n = normalize(vNormal);
  vec3 viewDir = normalize(vViewPosition);
  vec3 color = brainShade(base, n, viewDir, 1.0, uSpecular);
  color += uGlowColor * uGlow * (0.18 + 0.45 * uPulse);
  gl_FragColor = vec4(color, uOpacity);
  #include <colorspace_fragment>
}
`;

/** Materiale delle strutture profonde e degli organi di senso: colore, bagliore di evidenziazione, impulso. */
export function solidMaterial(color, { glowColor = "#f59e0b", specular = 0.12, opacity = 1 } = {}) {
  return new THREE.ShaderMaterial({
    vertexShader: SOLID_VERTEX,
    fragmentShader: SOLID_FRAGMENT,
    clipping: true,
    side: THREE.DoubleSide,
    transparent: opacity < 1,
    depthWrite: opacity >= 1,
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uGlowColor: { value: new THREE.Color(glowColor) },
      uGlow: { value: 0 },
      uPulse: { value: 0 },
      uOpacity: { value: opacity },
      uSpecular: { value: specular },
    },
  });
}

const PATH_VERTEX = /* glsl */ `
#include <common>
#include <clipping_planes_pars_vertex>
attribute float arc;
varying float vArc;
varying vec3 vNormal;
varying vec3 vViewPosition;
void main() {
  vArc = arc;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  vNormal = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;

const PATH_FRAGMENT = /* glsl */ `
#include <common>
#include <clipping_planes_pars_fragment>
uniform vec3 uColor;
uniform float uTime;
uniform float uPulses;
uniform float uSpacing;
uniform float uSpeed;
uniform float uStrength;
varying float vArc;
varying vec3 vNormal;
varying vec3 vViewPosition;
void main() {
  #include <clipping_planes_fragment>
  vec3 n = normalize(vNormal);
  float facing = max(dot(n, normalize(vViewPosition)), 0.0);
  // tubo luminoso: più chiaro al centro, sfumato sui bordi
  float body = 0.35 + 0.65 * facing;
  // impulsi: pacchetti che scorrono lungo l'ascissa del fascio
  float k = fract(vArc / uSpacing - uTime * uSpeed / uSpacing);
  float pulse = uPulses * smoothstep(0.0, 0.06, k) * (1.0 - smoothstep(0.06, 0.32, k));
  vec3 color = uColor * body * (0.55 + 0.45 * uStrength) + vec3(1.0) * pulse * 0.85 + uColor * pulse;
  gl_FragColor = vec4(color, clamp(0.55 * uStrength + 0.25 + pulse, 0.0, 1.0));
  #include <colorspace_fragment>
}
`;

/** Materiale di un fascio di fibre (tubo) con impulsi che viaggiano. */
export function pathwayMaterial(color) {
  return new THREE.ShaderMaterial({
    vertexShader: PATH_VERTEX,
    fragmentShader: PATH_FRAGMENT,
    clipping: true,
    transparent: true,
    depthWrite: false,
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uTime: { value: 0 },
      uPulses: { value: 0 },
      uSpacing: { value: 2.2 },
      uSpeed: { value: 2.4 },
      uStrength: { value: 1 },
    },
  });
}

/** Sfere dei marcatori (InstancedMesh con colore per istanza): sempre accese, leggermente pulsanti. */
export function markerMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: /* glsl */ `
      #include <common>
      #include <clipping_planes_pars_vertex>
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying vec3 vColor;
      void main() {
        vColor = instanceColor;
        vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
        vViewPosition = -mvPosition.xyz;
        vNormal = normalize(normalMatrix * mat3(instanceMatrix) * normal);
        gl_Position = projectionMatrix * mvPosition;
        #include <clipping_planes_vertex>
      }`,
    fragmentShader: /* glsl */ `
      #include <common>
      #include <clipping_planes_pars_fragment>
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying vec3 vColor;
      void main() {
        #include <clipping_planes_fragment>
        float facing = max(dot(normalize(vNormal), normalize(vViewPosition)), 0.0);
        vec3 color = vColor * (0.55 + 0.6 * facing) + vec3(0.25) * pow(facing, 8.0);
        gl_FragColor = vec4(color, 1.0);
        #include <colorspace_fragment>
      }`,
    clipping: true,
  });
}
