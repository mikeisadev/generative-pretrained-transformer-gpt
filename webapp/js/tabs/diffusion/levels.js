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
/** Aria attorno ai livelli astratti inquadrati (1 = il contenuto tocca i bordi della parte visibile). */
const FIT_MARGIN = 1.14;
/** Direzione della camera per gli schemi: quasi di fronte, appena dall'alto. */
export const FRONT = new THREE.Vector3(0, 0.12, 1).normalize();

/**
 * @param {object} o
 * @param {import("../../core/sceneHost.js").SceneHost} o.host
 * @param {import("../../core/director.js").CameraDirector} o.director
 * @param {{detail: THREE.Object3D, blocks: THREE.Object3D, macro: THREE.Object3D}} o.groups
 * @param {HTMLElement} o.widgetParent dove disegnare il selettore Auto / Dettaglio / Blocchi / Panoramica
 * @param {() => Promise<void>} o.focusDetail inquadratura quando l'utente sceglie "Dettaglio"
 * @param {(mode: string) => void} [o.onPick] avvisato prima del volo quando lo studente sceglie un livello
 */
export function createLevels({ host, director, groups, widgetParent, focusDetail, onPick = () => {} }) {
  // riquadri dei livelli (immagini sopra, video sotto), calcolati ora che sono tutti visibili (poi il
  // LOD ne nasconde due): servono alle inquadrature e ai confini della navigazione libera
  const levelBox = { detail: framingBox(groups.detail), blocks: framingBox(groups.blocks), macro: framingBox(groups.macro) };
  const lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: THRESHOLDS.detail, group: groups.detail },
      { id: "blocks", label: "Blocchi", maxDistance: THRESHOLDS.blocks, group: groups.blocks },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: groups.macro },
    ],
    widgetParent,
    // Solo le scelte dello studente nel selettore inquadrano il livello, anche se è già quello attivo
    // (dopo uno spostamento o uno zoom libero "Blocchi" deve riportare lo schema in vista).
    // Le azioni bloccano il livello con lock(): lì la regia della camera è dell'azione.
    onPick: (mode) => {
      onPick(mode);
      if (mode === "detail") focusDetail();
      else if (mode !== "auto") flyToLevel(mode);
    },
  });

  /** Fattore (≥ 1) di cui la parte visibile dello stage è più stretta di quella di riferimento. */
  const narrowness = () => Math.max(1, REFERENCE_ASPECT / host.visibleAspect());
  // Ricalcolate solo quando lo stage cambia (finestra, codice, grafici: tutti passano da host.resize),
  // non a ogni frame: leggere le misure del DOM costa ~40 µs per frame anche a scena ferma.
  const scaleThresholds = () => {
    const k = narrowness();
    lod.levels[0].maxDistance = THRESHOLDS.detail * k;
    lod.levels[1].maxDistance = THRESHOLDS.blocks * k;
  };
  scaleThresholds();
  host.onResize(scaleThresholds);

  /** Distanza a cui il riquadro `box` entra tutto nella parte visibile dello stage. */
  function fitDistance(box, margin) {
    const size = box.getSize(new THREE.Vector3());
    const tan = Math.tan(THREE.MathUtils.degToRad(host.camera.fov) / 2);
    return Math.max(size.y / 2 / tan, size.x / 2 / (tan * host.visibleAspect())) * margin;
  }

  /**
   * Inquadra di fronte tutto il livello astratto `id` ("blocks" o "macro"), almeno alla sua distanza
   * minima. Il margine lascia un po' d'aria ai bordi: i blocchi più esterni non toccano il bordo e il
   * pad della navigazione (in basso a destra) copre al massimo un angolo vuoto.
   */
  function flyToLevel(id, duration = 1) {
    director.hideFrame();
    const box = levelBox[id];
    const center = box.getCenter(new THREE.Vector3());
    const distance = Math.max(fitDistance(box, FIT_MARGIN), MIN_DISTANCE[id] * narrowness());
    return director.flyTo(center.clone().addScaledVector(FRONT, distance), center, duration);
  }

  /**
   * Blocca un livello ("detail"…) o torna ad "auto" senza muovere la camera (la regia è dell'azione).
   * Il livello chiesto dall'azione viene ricordato: se lo studente ne sceglie un altro a mano durante
   * l'azione, tornando a seguirla si ritrova quello dell'azione (vedi sceneNav.js).
   */
  let actionMode = "auto";
  function lock(mode) {
    actionMode = mode;
    lod.setMode(mode);
  }

  return { lod, flyToLevel, lock, actionMode: () => actionMode, boxes: levelBox };
}
