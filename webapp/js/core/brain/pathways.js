/**
 * Fasci di fibre del tab (arcuato, radiazione ottica e uditiva, via mesolimbica, via perforante…) come
 * tubi luminosi lungo curve, con impulsi che li percorrono (shader: in JavaScript solo `uTime`).
 *
 *   const paths = new PathwayLayer(spec.pathways, { clipPlanes });
 *   brain.root.add(paths.root); paths.setPlaying(true); paths.setTime(t);
 *
 * Le curve dei percorsi dell'atlante passano per punti scelti sull'anatomia (atlas.js PATHWAYS.path),
 * dentro la sostanza bianca: il tubo non attraversa la superficie a caso. Un percorso nuovo di un tab
 * (solo `through`) passa per i centroidi delle sue regioni, con i punti intermedi spinti verso il
 * centro del cervello (così resta dentro e non taglia la corteccia in linea retta).
 */
import * as THREE from "three";
import { MNI_CENTER, atlasEntry, mirrorMni, mniToScene, placements } from "./atlas.js";
import { sampleCurve, sweepTube } from "./geometry.js";
import { pathwayMaterial } from "./shaders.js";

/** Punti MNI (mm) di un percorso, per lato: quelli dell'atlante o, per un percorso nuovo, i centroidi. */
export function pathwayControlPoints(pathway) {
  const atlas = pathway.atlas ? atlasEntry(pathway.id) : null;
  const sides = atlas?.hemisphere === "left" ? ["left"] : atlas?.hemisphere === "right" ? ["right"] : ["left", "right"];
  if (atlas?.path) return sides.map((side) => ({ side, points: side === "right" ? atlas.path.map(mirrorMni) : atlas.path.map((p) => p.slice()) }));
  const left = pathway.through.map((id) => placements(atlasEntry(id)).find((p) => p.side !== "right")?.mni).filter(Boolean);
  if (left.length < 2) return [];
  // tra due centroidi un punto a metà, spinto del 30% verso il centro del cervello: la curva resta dentro
  const points = [];
  left.forEach((p, i) => {
    if (i > 0) {
      const a = left[i - 1];
      const mid = [(a[0] + p[0]) / 2, (a[1] + p[1]) / 2, (a[2] + p[2]) / 2];
      points.push(mid.map((v, k) => v + (MNI_CENTER[k] - v) * 0.3));
    }
    points.push(p);
  });
  return sides.map((side) => ({ side, points: side === "right" ? points.map(mirrorMni) : points }));
}

export class PathwayLayer {
  /**
   * @param {object[]} pathways percorsi normalizzati (spec.js)
   * @param {{ clipPlanes?: THREE.Plane[], quality?: "high" | "low" }} [o]
   */
  constructor(pathways, { clipPlanes = [], quality = "high" } = {}) {
    this.root = new THREE.Group();
    this.root.name = "pathways";
    this.items = [];
    const samples = quality === "low" ? 36 : 64;
    const radial = quality === "low" ? 6 : 10;
    for (const pathway of pathways) {
      const material = pathwayMaterial(pathway.color);
      material.clippingPlanes = clipPlanes;
      const meshes = [];
      const anchors = [];
      for (const { side, points } of pathwayControlPoints(pathway)) {
        const scene = points.map((p) => mniToScene(p));
        const { points: sampled, lengths } = sampleCurve(scene, samples);
        const tube = sweepTube(sampled, { radius: (t) => 0.19 * (0.7 + 0.3 * Math.sin(Math.PI * t)), radial, lengths, caps: true });
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(tube.positions, 3));
        geometry.setAttribute("normal", new THREE.BufferAttribute(tube.normals, 3));
        geometry.setAttribute("arc", new THREE.BufferAttribute(tube.arc, 1));
        geometry.setIndex(new THREE.BufferAttribute(tube.indices, 1));
        geometry.computeBoundingSphere();
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `pathway-${pathway.id}-${side}`;
        mesh.userData.pathway = pathway.id;
        mesh.userData.noBounds = true; // dentro il cervello: non cambia i confini della navigazione
        mesh.renderOrder = 2; // prima della corteccia trasparente (vedi anatomy.js)
        this.root.add(mesh);
        meshes.push(mesh);
        anchors.push({ side, point: sampled[Math.floor(sampled.length * 0.5)], points: sampled });
      }
      this.items.push({ pathway, material, meshes, anchors });
    }
  }

  setPlaying(on) {
    for (const item of this.items) item.material.uniforms.uPulses.value = on ? 1 : 0;
  }

  setTime(t) {
    for (const item of this.items) item.material.uniforms.uTime.value = t;
  }

  /** Mette in risalto un percorso (gli altri si attenuano); null = tutti uguali. */
  emphasize(id) {
    for (const item of this.items) item.material.uniforms.uStrength.value = !id || item.pathway.id === id ? 1 : 0.35;
  }

  setVisible(on) {
    this.root.visible = on;
  }

  /** Box della scena di un percorso (per inquadrarlo). */
  box(id, target = new THREE.Box3()) {
    target.makeEmpty();
    for (const item of this.items) {
      if (item.pathway.id !== id) continue;
      for (const mesh of item.meshes) {
        mesh.geometry.computeBoundingBox();
        target.union(mesh.geometry.boundingBox);
      }
    }
    return target;
  }

  dispose() {
    for (const item of this.items) {
      item.material.dispose();
      for (const mesh of item.meshes) mesh.geometry.dispose();
    }
    this.root.removeFromParent();
  }
}
