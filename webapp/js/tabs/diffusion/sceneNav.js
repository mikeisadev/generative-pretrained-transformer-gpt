/**
 * Navigazione libera nel tab "Immagini e video" (ui.nav del nucleo: vedi ARCHITECTURE.md,
 * "Navigazione libera"). Qui solo ciò che è proprio di questa scena:
 *   - "Centra" (C, Home, ◎): la vista d'insieme del tab, la stessa dell'avvio e di "Tutta la pipeline";
 *     durante un'azione invece torna a ciò che l'azione sta mostrando;
 *   - i confini: immagini e video del livello dello zoom attivo (ognuno ha i suoi: da vicino il
 *     Dettaglio è più stretto dei Blocchi, la Panoramica è la più grande) più il "banco di lavoro"
 *     delle azioni quando è acceso;
 *   - "Dettaglio" scelto a mano inquadra il pezzo che lo studente sta guardando, non sempre il denoiser;
 *   - un livello scelto a mano durante un'azione rende la camera libera; tornando a seguire l'azione
 *     torna anche il livello che l'azione usa;
 *   - una didascalia quando lo studente, spostandosi da solo, passa dalla parte immagini a quella video.
 *
 * La scena è come un edificio a due piani: sopra la pipeline delle immagini, sotto (VIDEO_Y) quella
 * del video; il banco di lavoro delle azioni sta ancora più su.
 */
import * as THREE from "three";
import { focusLab } from "./lab.js";
import { FRONT } from "./levels.js";
import { MACRO_VIDEO_Y, VIDEO_Y } from "./overview.js";

/**
 * Zone del punto guardato (y) per ogni livello dello zoom: la riga delle immagini sta a y = 0, quella del
 * video più in basso (nella Panoramica, ingrandita, molto più in basso). Con isteresi: la didascalia
 * cambia solo quando lo studente è entrato bene nell'altra parte (30% della distanza tra le righe oltre
 * il punto di mezzo). Le viste d'insieme (y ≈ −31 nei Blocchi, ≈ −49 nella Panoramica) stanno nella
 * fascia neutra; "Dettaglio" sceglie la parte video poco sotto il punto di mezzo.
 */
const zoneLimits = (videoY) => ({ imagesAbove: videoY * 0.2, videoBelow: videoY * 0.8, videoDetailBelow: videoY * 0.6 });
const ZONES = { detail: zoneLimits(VIDEO_Y), blocks: zoneLimits(VIDEO_Y), macro: zoneLimits(MACRO_VIDEO_Y) };

const ZONE_CAPTIONS = {
  images: "Parte immagini: la pipeline di Stable Diffusion, dal prompt (a sinistra) all'immagine (a destra). Più in basso c'è la parte video.",
  video: "Parte video: come per le immagini, ma il denoiser lavora su un cubo di latenti (tempo × altezza × larghezza) e tutti i frame nascono insieme.",
};

/**
 * Pezzi della pipeline delle immagini che "Dettaglio" può inquadrare: il più vicino (in orizzontale)
 * al punto guardato. `ids` può essere una funzione (il denoiser dipende dalla vista U-Net / DiT).
 */
const DETAIL_AREAS = [
  { ids: ["prompt", "tokens", "textEncoder", "textEmb"], margin: 1.15 },
  { ids: (tab) => tab.denoiser, margin: 1.15 },
  { ids: ["vae", "image"], margin: 1.3 },
];

/**
 * @param {object} tab l'oggetto condiviso del tab (index.js)
 * @param {{boxes: Record<string, THREE.Box3>, flyToLevel: Function, actionMode: () => string}} levels output di createLevels
 */
export function installSceneNav(tab, levels) {
  const { ui, lod, lab } = tab;
  const { nav, director, host, player } = ui;

  // --- confini: quelli del livello attivo più il banco di lavoro se un'azione lo usa. Ogni livello ha i
  // suoi perché da vicino si vede poco: nel Dettaglio i confini larghi dei Blocchi lascerebbero arrivare
  // dove sullo schermo non c'è più niente ("Ti sei perso?"). La navigazione li ricalcola al più ogni 3 s:
  // a ogni cambio di livello glieli si ripropone (setBounds), così valgono subito (e, se il punto
  // guardato ne resta fuori, il bordo elastico lo riporta dentro piano piano).
  const bounds = new THREE.Box3();
  const navBounds = () => {
    bounds.copy(levels.boxes[lod.current] ?? levels.boxes.blocks);
    if (lab.group.visible && lab.items.size) bounds.union(lab.bounds());
    return bounds;
  };
  nav.setBounds(navBounds);
  // Zoom avanti: 4 unità (da così vicino il blocco più piccolo della pipeline riempie già lo schermo).
  // Rotazione come negli altri tab "a parete": niente vista da dietro (testo a specchio) o dall'alto.
  nav.setOrbitLimits({ turn: 75, tilt: 60, minDistance: 4 });

  /**
   * Vista d'insieme ("blocks", come all'avvio) o "Panoramica" ("macro"). Lo zoom torna automatico:
   * un livello bloccato a mano resterebbe tale anche da lontano, e la rotella non lo cambierebbe più.
   */
  function overview(level = "blocks") {
    if (lod.mode !== "auto") lod.setMode("auto");
    return levels.flyToLevel(level);
  }

  /** Il livello Dettaglio da un link diretto: zoom automatico e il pezzo che si sta guardando. */
  function detail() {
    if (lod.mode !== "auto") lod.setMode("auto");
    return focusDetail();
  }

  // --- "Centra"
  nav.setHome(() => (player.action ? actionHome() : overview()));

  /**
   * Durante un'azione "Centra" vuol dire "rimettimi dove sta succedendo qualcosa": si torna a seguire
   * l'azione (il chip "Camera libera" sparisce) e si vola sull'ultima inquadratura che ha chiesto;
   * se non ne ha chieste da quando lo studente si è mosso, sulla zona che l'azione usa.
   */
  function actionHome() {
    const pending = director.pendingFlight;
    const resumed = director.resumeFollow();
    if (pending) return resumed;
    if (lab.group.visible && lab.items.size) return focusLab(tab);
    if (player.action.id === "video") return tab.focus(tab.video.regions.intro(), { showFrame: false, margin: 1.02, direction: FRONT });
    return tab.focus(tab.denoiser, { showFrame: false, margin: 1.15 });
  }

  // --- un livello scelto a mano durante un'azione: lo studente vuole guardare quel livello, quindi la
  // camera diventa libera (il passo successivo non la riporta indietro) e compare "Segui l'azione".
  // Tornando a seguire (chip, "Centra", fine dell'azione) torna anche il livello dell'azione: "Un passo
  // dentro il denoiser" e "Video" vivono nel Dettaglio, che uno zoom bloccato su "Blocchi" nasconderebbe.
  function levelPicked(mode) {
    if (mode !== "auto" && player.action) director.release();
  }
  director.onFollowChange((following) => {
    const wanted = levels.actionMode();
    if (following && player.action && lod.mode !== wanted) lod.setMode(wanted);
  });

  // --- un'azione che usava il banco di lavoro si chiude (✕, un'altra azione, un passo del Percorso, un
  // altro tab): il banco sparisce e una camera rimasta a guardarlo inquadrerebbe il vuoto sopra la scena.
  // Si torna alla vista d'insieme; chi ha chiuso l'azione per andare altrove (un passo, una vista, un'altra
  // azione) chiede subito il suo volo, che sostituisce questo.
  const sceneTop = Math.max(levels.boxes.detail.max.y, levels.boxes.blocks.max.y);
  player.on("state", () => {
    if (!player.action && host.controls.target.y > sceneTop) overview();
  });

  // --- didascalia quando lo studente passa da una parte all'altra (pochi confronti per frame). I voli
  // (menu, viste, azioni, "Centra") aggiornano la zona in silenzio e tolgono la didascalia di una zona
  // da cui portano via: "Centra" dalla parte video non deve lasciare scritto "Parte video".
  let zone = "both";
  let captionZone = null; // zona di cui è mostrata la didascalia (null: la didascalia è di altri)
  let level = lod.current;
  host.onFrame(() => {
    if (lod.current !== level) {
      level = lod.current;
      nav.setBounds(navBounds);
    }
    const y = host.controls.target.y;
    const limits = ZONES[level];
    if (director.flight || player.action) {
      zone = zoneOf(y, "both", limits);
      if (captionZone && zone !== captionZone) {
        // un'azione scrive le sue didascalie: lì non si tocca niente
        if (!player.action && ui.caption.textContent === ZONE_CAPTIONS[captionZone]) tab.setCaption("");
        captionZone = null;
      }
      return;
    }
    const next = zoneOf(y, zone, limits);
    if (next === zone) return;
    zone = next;
    captionZone = zone;
    tab.setCaption(ZONE_CAPTIONS[zone]);
  });

  /**
   * "Dettaglio" scelto a mano: il pezzo che lo studente sta guardando (il denoiser dalla vista d'insieme).
   * Le soglie sono quelle del livello che si vedeva fino a un attimo fa (`level`, aggiornato a ogni frame):
   * scegliendo "Dettaglio" il LOD passa subito al Dettaglio, ma la camera è ancora dov'era (es. Panoramica).
   */
  function focusDetail() {
    const target = host.controls.target;
    if (target.y < ZONES[level].videoDetailBelow) return tab.focus(tab.video.regions.intro(), { showFrame: false, margin: 1.05 });
    const box = new THREE.Box3();
    let best = DETAIL_AREAS[1];
    let bestDistance = Infinity;
    for (const area of DETAIL_AREAS) {
      const ids = typeof area.ids === "function" ? [area.ids(tab)] : area.ids;
      box.makeEmpty();
      for (const id of ids) {
        const object = tab.pipeline.registry.get(id);
        if (object) box.union(new THREE.Box3().setFromObject(object));
      }
      const distance = box.isEmpty() ? Infinity : Math.abs((box.min.x + box.max.x) / 2 - target.x);
      if (distance < bestDistance) [best, bestDistance] = [area, distance];
    }
    const ids = typeof best.ids === "function" ? best.ids(tab) : best.ids;
    return tab.focus(ids, { showFrame: false, margin: best.margin });
  }

  return { overview, detail, focusDetail, levelPicked };
}

/** Zona del punto guardato: "images", "video", oppure quella di prima nella fascia neutra. */
function zoneOf(y, previous, limits) {
  if (y > limits.imagesAbove) return "images";
  if (y < limits.videoBelow) return "video";
  return previous;
}
