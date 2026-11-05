/**
 * Zoom semantico del tab: tre livelli (Dettaglio, Blocchi, Panoramica) con il LODController del
 * nucleo, più la regia della camera quando l'utente sceglie un livello a mano.
 *
 * Le soglie di distanza sono misurate sullo stage "di riferimento" (1600 × 900 senza pannelli,
 * rapporto visibile ≈ 1,44). Su schermi stretti, o con il pannello del codice aperto, la stessa
 * distanza mostra meno scena: le soglie crescono in proporzione, così il livello cambia quando
 * cambia davvero ciò che si riesce a leggere.
 */
import * as THREE from "three";
import { LODController } from "../../core/lod.js";
import { framingBox } from "./shapes.js";

const THRESHOLDS = { detail: 118, blocks: 300 };
const REFERENCE_ASPECT = 1.44;
/** Distanza minima per i livelli scelti a mano (appena oltre la soglia del livello precedente). */
const MIN_DISTANCE = { blocks: 135, macro: 330 };
/** Direzione della camera per gli schemi: quasi di fronte, appena dall'alto. */
export const FRONT = new THREE.Vector3(0, 0.12, 1).normalize();

/**
 * @param {object} o
 * @param {import("../../core/sceneHost.js").SceneHost} o.host
 * @param {import("../../core/director.js").CameraDirector} o.director
 * @param {{detail: THREE.Object3D, blocks: THREE.Object3D, macro: THREE.Object3D}} o.groups
 * @param {HTMLElement} o.widgetParent dove disegnare il selettore Auto / Dettaglio / Blocchi / Panoramica
 * @param {() => Promise<void>} o.focusDetail inquadratura quando l'utente sceglie "Dettaglio"
 */
export function createLevels({ host, director, groups, widgetParent, focusDetail }) {
  // riquadri dei livelli astratti, calcolati ora che sono visibili (poi il LOD li nasconde)
  const levelBox = { blocks: framingBox(groups.blocks), macro: framingBox(groups.macro) };
  let programmatic = false; // true mentre un'azione blocca il livello: niente voli di camera
  let lod = null; // il costruttore chiama subito onChange: `lod` deve già esistere (null)
  lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: THRESHOLDS.detail, group: groups.detail },
      { id: "blocks", label: "Blocchi", maxDistance: THRESHOLDS.blocks, group: groups.blocks },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: groups.macro },
    ],
    widgetParent,
    onChange: (id) => {
      if (!programmatic && lod?.mode !== "auto" && lod?.mode === id) (id === "detail" ? focusDetail() : flyToLevel(id));
    },
  });

  /** Fattore (≥ 1) di cui la parte visibile dello stage è più stretta di quella di riferimento. */
  const narrowness = () => Math.max(1, REFERENCE_ASPECT / host.visibleAspect());
  host.onFrame(() => {
    const k = narrowness();
    lod.levels[0].maxDistance = THRESHOLDS.detail * k;
    lod.levels[1].maxDistance = THRESHOLDS.blocks * k;
  });

  /** Distanza a cui il riquadro `box` entra tutto nella parte visibile dello stage. */
  function fitDistance(box, margin) {
    const size = box.getSize(new THREE.Vector3());
    const tan = Math.tan(THREE.MathUtils.degToRad(host.camera.fov) / 2);
    return Math.max(size.y / 2 / tan, size.x / 2 / (tan * host.visibleAspect())) * margin;
  }

  /** Inquadra di fronte tutto il livello astratto `id` ("blocks" o "macro"), almeno alla sua distanza minima. */
  function flyToLevel(id, duration = 1) {
    director.hideFrame();
    const box = levelBox[id];
    const center = box.getCenter(new THREE.Vector3());
    const distance = Math.max(fitDistance(box, 1.06), MIN_DISTANCE[id] * narrowness());
    return director.flyTo(center.clone().addScaledVector(FRONT, distance), center, duration);
  }

  /** Blocca un livello ("detail"…) o torna ad "auto" senza muovere la camera (la regia è dell'azione). */
  function lock(mode) {
    programmatic = true;
    lod.setMode(mode);
    programmatic = false;
  }

  return { lod, flyToLevel, lock };
}
