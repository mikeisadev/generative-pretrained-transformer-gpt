/**
 * Mappa della scena: 16 "regioni" su una griglia 4 × 4, una per meccanismo.
 * Lo stesso posto ospita i tre livelli del LOD:
 *   Dettaglio  → la regione con i componenti interni (file region*.js)
 *   Blocchi    → un rettangolo con titolo e descrizione (overview.js, testo `block`)
 *   Panoramica → il diagramma Dati → Pre-training → Post-training → Modello → Inferenza
 * Le righe seguono le categorie delle azioni.
 */
import * as THREE from "three";
import { REGION_H, REGION_W, STEP_X, STEP_Y } from "./regionData.js";
import { frame, text } from "./shapes.js";

export { nearestRegionAt, REGION_BY_ID, REGION_H, REGION_W, REGIONS, ROWS, STEP_X, STEP_Y } from "./regionData.js";

/** Centro della regione in coordinate di scena (griglia centrata nell'origine). */
export function regionCenter(region) {
  return new THREE.Vector3((region.col - 1.5) * STEP_X, (1.5 - region.row) * STEP_Y, 0);
}

/** Box3 della regione (per inquadrarla con il CameraDirector). */
export function regionBox(region, pad = 0) {
  const c = regionCenter(region);
  return new THREE.Box3(new THREE.Vector3(c.x - REGION_W / 2 - pad, c.y - REGION_H / 2 - pad, -0.5), new THREE.Vector3(c.x + REGION_W / 2 + pad, c.y + REGION_H / 2 + pad, 0.5));
}

/** Riquadro di tutta la griglia. */
export function gridBox() {
  return new THREE.Box3(new THREE.Vector3(-2 * STEP_X, -2 * STEP_Y, -1), new THREE.Vector3(2 * STEP_X, 2 * STEP_Y + 9, 1));
}

/** Colore della cornice di una regione a riposo (quella "dove sei" prende il colore della regione). */
export const FRAME_COLOR = "#334155";

/** Cornice e titolo comuni a tutte le regioni del livello Dettaglio. Restituisce la cornice. */
export function regionShell(group, region) {
  const line = frame(group, REGION_W, REGION_H, FRAME_COLOR);
  text(group, region.title, 0, REGION_H / 2 - 1.3, { size: 0.95, color: region.color });
  return line;
}
