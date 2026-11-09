/**
 * Regioni del tab sul cervello 3D: evidenziazione (macchia sulla corteccia o bagliore della struttura
 * profonda), marcatori dei nuclei (sfere), voci per le etichette e scelta della regione con un clic.
 *
 *   const regions = new RegionLayer(brain, spec);
 *   regions.apply({ focus: ["hippocampus"], selected: "a1" });   // colori e impulsi
 *   regions.labelItems()                                          // per LabelOverlay
 *   regions.pick(raycaster) → { id, kind, point } | null
 *
 * Colori: le regioni del tab in arancione (il colore dei "pesi imparati" in tutta la web app), quelle
 * chiamate da un'analogia ("Mostra nel modello 3D") in azzurro e pulsanti, la regione scelta con un
 * alone chiaro. Una sola tinta per le regioni del tab, non un arcobaleno: i nomi li dicono le etichette.
 */
import * as THREE from "three";
import { LOBES, MM, atlasEntry, atlasKind, mniToScene, placements, sceneToMni, shortName } from "./atlas.js";
import { MAX_PULSES, markerMaterial } from "./shaders.js";
import { lobeOf } from "./shape.js";

// shortName sta nell'atlante (puro): lo usa anche il pannello della sidebar, che non deve caricare three.js
export { shortName };

export const REGION_COLORS = Object.freeze({ tab: "#f59e0b", focus: "#38bdf8", marker: "#fbbf24", selected: "#e0f2fe" });

/** Strutture disegnate che non sono regioni dell'atlante: un nome per la scheda quando le si clicca. */
const STRUCTURE_NAMES = Object.freeze({ optic: "Nervo e tratto ottico", eye: "Occhio", iris: "Iride", auditory_nerve: "Nervo acustico (VIII)" });

export class RegionLayer {
  /**
   * @param {import("./anatomy.js").BrainModel} brain
   * @param {ReturnType<import("./spec.js").normalizeBrainSpec>} spec
   */
  constructor(brain, spec) {
    this.brain = brain;
    this.spec = spec;
    this.focus = [];
    this.selected = null;
    this.root = new THREE.Group();
    this.root.name = "regions";
    // marcatori dei nuclei (LGN, MGN, pulvinar, nuclei del tronco): una InstancedMesh, colore per istanza
    this.markerIds = [];
    for (const id of spec.markers) for (const p of placements(atlasEntry(id))) this.markerIds.push({ id, side: p.side, mni: p.mni });
    const geometry = new THREE.IcosahedronGeometry(1, 2);
    this.markers = new THREE.InstancedMesh(geometry, markerMaterial(), Math.max(1, this.markerIds.length));
    this.markers.count = this.markerIds.length;
    this.markers.material.clippingPlanes = brain.clipPlanes;
    this.markers.userData.noBounds = true;
    this.markers.renderOrder = 3;
    const m = new THREE.Matrix4();
    const color = new THREE.Color();
    this.markerIds.forEach((mk, i) => {
      const r = atlasEntry(mk.id).radius * MM;
      m.makeScale(r, r, r).setPosition(...mniToScene(mk.mni));
      this.markers.setMatrixAt(i, m);
      this.markers.setColorAt(i, color.set(REGION_COLORS.marker));
    });
    if (!this.markerIds.length) this.markers.setColorAt(0, color.set(REGION_COLORS.marker)); // instanceColor sempre presente (shader)
    this.markers.instanceMatrix.needsUpdate = true;
    this.markers.computeBoundingSphere();
    this.root.add(this.markers);
    this.pulse = new Float32Array(MAX_PULSES);
    this.apply({});
  }

  /** Regioni evidenziate: quelle del tab, più quelle a fuoco; ognuna con il suo indice per gli impulsi. */
  highlightList() {
    const list = [];
    const add = (id, color) => {
      if (list.some((r) => r.id === id) || list.length >= MAX_PULSES) return;
      const entry = atlasEntry(id);
      if (!entry || atlasKind(id) !== "region") return;
      list.push({ id, color, radius: entry.radius, index: list.length });
    };
    for (const id of this.focus) add(id, REGION_COLORS.focus);
    for (const id of this.spec.highlight) add(id, REGION_COLORS.tab);
    return list;
  }

  /**
   * Applica evidenziazione, fuoco e scelta. focus: id dell'atlante da far pulsare (regioni e
   * marcatori; i percorsi li gestisce la vista); selected: id della regione scelta (o null).
   */
  apply({ focus = this.focus, selected = this.selected } = {}) {
    this.focus = focus.filter((id) => atlasKind(id) === "region" || atlasKind(id) === "marker");
    this.selected = selected;
    const list = this.highlightList();
    this.brain.setHighlights(list);
    this.focusIndex = list.filter((r) => this.focus.includes(r.id)).map((r) => r.index);
    // un nucleo dentro una struttura (il pulvinar nel talamo, l'oliva nel ponte): la struttura si fa
    // semitrasparente, così il marcatore si vede
    const shown = new Set(this.markerIds.map((mk) => mk.id));
    this.brain.setStructureSeeThrough("thalamus", shown.has("pulvinar"));
    this.brain.setStructureSeeThrough("brainstem", shown.has("superior_olive"));
    // l'occhio diventa di vetro quando c'è da vedere la retina
    const eye = this.brain.structures.get("eye");
    if (eye) eye.material.uniforms.uOpacity.value = list.some((r) => r.id === "retina") ? 0.32 : 0.82;
    // marcatori: azzurri se a fuoco
    const color = new THREE.Color();
    this.markerIds.forEach((mk, i) => this.markers.setColorAt(i, color.set(this.focus.includes(mk.id) ? REGION_COLORS.focus : REGION_COLORS.marker)));
    if (this.markers.instanceColor) this.markers.instanceColor.needsUpdate = true;
    this.brain.setSelection(selected ? this.selectionShape(selected) : null);
  }

  /** Forma dell'alone della regione scelta: un punto sulla corteccia (o una struttura profonda). */
  selectionShape(id) {
    const anchors = this.brain.data.anchors.get(id);
    if (anchors?.length) return { id, point: anchors[0].point, radius: (atlasEntry(id)?.radius ?? 10) * 1.05 };
    return { id };
  }

  /** Intensità degli impulsi delle regioni a fuoco (0…1) a un certo tempo: la vista la chiama per qualche secondo. */
  setPulse(t, amount = 1) {
    this.pulse.fill(0);
    const v = amount * (0.5 + 0.5 * Math.sin(t * 5.2));
    for (const i of this.focusIndex ?? []) this.pulse[i] = v;
    this.brain.setPulse(this.pulse);
    // anche i marcatori a fuoco "respirano"
    if (this.focus.some((id) => atlasKind(id) === "marker")) {
      const m = new THREE.Matrix4();
      this.markerIds.forEach((mk, i) => {
        const r = atlasEntry(mk.id).radius * MM * (this.focus.includes(mk.id) ? 1 + 0.35 * v : 1);
        m.makeScale(r, r, r).setPosition(...mniToScene(mk.mni));
        this.markers.setMatrixAt(i, m);
      });
      this.markers.instanceMatrix.needsUpdate = true;
    }
  }

  /**
   * Voci delle etichette: regioni del tab e a fuoco, marcatori. anchors = un punto per lato (la vista
   * sceglie quello più vicino alla camera); priority 0 = a fuoco o scelta.
   */
  labelItems() {
    const items = [];
    const ids = [...new Set([...this.focus, ...this.spec.highlight, ...this.spec.markers, ...(this.selected ? [this.selected] : [])])];
    for (const id of ids) {
      const entry = atlasEntry(id);
      if (!entry) continue;
      const anchors = this.anchorsOf(id);
      if (!anchors.length) continue;
      const focused = this.focus.includes(id) || this.selected === id;
      const kind = atlasKind(id);
      items.push({
        id,
        text: shortName(entry),
        color: focused ? REGION_COLORS.focus : kind === "marker" ? REGION_COLORS.marker : REGION_COLORS.tab,
        anchors: anchors.map((a) => a.point),
        normals: anchors.map((a) => a.normal),
        kind: entry.kind,
        priority: focused ? 0 : kind === "marker" ? 2 : 1,
        data: { kind: "region", id },
      });
    }
    return items;
  }

  /** Punti d'aggancio di una voce (scena): corticale = sulla superficie, altrimenti il centroide. */
  anchorsOf(id) {
    const surface = this.brain.data.anchors.get(id);
    if (surface?.length) {
      return surface.map((a) => ({ side: a.side, point: [a.point[0] + a.normal[0] * 0.05, a.point[1] + a.normal[1] * 0.05, a.point[2] + a.normal[2] * 0.05], normal: a.normal }));
    }
    return placements(atlasEntry(id)).map((p) => ({ side: p.side, point: mniToScene(p.mni), normal: null }));
  }

  /** Box della scena che contiene le regioni indicate (per inquadrarle). */
  box(ids, target = new THREE.Box3()) {
    target.makeEmpty();
    const v = new THREE.Vector3();
    for (const id of ids) {
      const entry = atlasEntry(id);
      if (!entry) continue;
      const r = (entry.radius ?? 8) * MM;
      for (const a of this.anchorsOf(id)) {
        target.expandByPoint(v.fromArray(a.point).addScalar(r));
        target.expandByPoint(v.fromArray(a.point).addScalar(-r));
      }
      // una struttura profonda: tutta la sua mesh
      for (const s of this.brain.structures.values()) if (s.region === id) target.union(s.geometry.boundingBox);
    }
    return target;
  }

  /**
   * Che cosa c'è sotto il puntatore: una struttura profonda (se visibile davanti o se la corteccia è
   * trasparente), un marcatore, oppure una regione della corteccia (la più vicina al punto colpito, o il
   * lobo). Le parti tolte dal piano di taglio non contano.
   */
  pick(raycaster) {
    const plane = this.brain.clipPlane;
    const kept = (hit) => plane.distanceToPoint(hit.point) >= -1e-4;
    const cortexMeshes = this.brain.cortex.filter((c) => c.mesh.visible).map((c) => c.mesh);
    const structureMeshes = [...this.brain.structures.values()].filter((s) => s.mesh.visible).map((s) => s.mesh);
    const markerHit = raycaster.intersectObject(this.markers, false).find(kept);
    const cortexHit = this.brain.opacity > 0.01 ? raycaster.intersectObjects(cortexMeshes, false).find(kept) : null;
    const structureHit = raycaster.intersectObjects(structureMeshes, false).find(kept);
    const glass = this.brain.opacity < 0.6;
    if (markerHit && (glass || !cortexHit || markerHit.distance < cortexHit.distance)) {
      const mk = this.markerIds[markerHit.instanceId];
      if (mk) return { id: mk.id, kind: "marker", point: markerHit.point.toArray() };
    }
    if (structureHit && (glass || !cortexHit || structureHit.distance < cortexHit.distance)) {
      const s = structureHit.object.userData;
      if (s.region) return { id: s.region, kind: "region", point: structureHit.point.toArray() };
      return { id: s.structure, kind: "structure", name: STRUCTURE_NAMES[s.structure] ?? s.structure, point: structureHit.point.toArray() };
    }
    if (!cortexHit) return null;
    if (cortexHit.object.userData.cortex === "cerebellum") return { id: "cerebellum", kind: "region", point: cortexHit.point.toArray() };
    return this.regionAt(cortexHit.point.toArray());
  }

  /** La regione corticale più vicina a un punto della corteccia (entro il suo raggio), o il lobo. */
  regionAt(point) {
    let best = null;
    let bestScore = Infinity;
    for (const [id, anchors] of this.brain.data.anchors) {
      const r = (atlasEntry(id).radius ?? 10) * MM * 1.25;
      for (const a of anchors) {
        const d = Math.hypot(a.point[0] - point[0], a.point[1] - point[1], a.point[2] - point[2]);
        // a parità di distanza vincono le regioni del tab
        const score = d / r - (this.spec.highlight.includes(id) ? 0.15 : 0);
        if (d < r && score < bestScore) {
          bestScore = score;
          best = id;
        }
      }
    }
    if (best) return { id: best, kind: "region", point };
    const lobe = lobeOf(sceneToMni(point));
    return { id: lobe, kind: "lobe", name: LOBES[lobe]?.label ?? lobe, point };
  }

  dispose() {
    this.markers.geometry.dispose();
    this.markers.material.dispose();
    this.markers.dispose?.();
    this.root.removeFromParent();
  }
}
