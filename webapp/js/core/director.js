/**
 * Regia della camera: volo fluido verso un oggetto e riquadro pulsante di evidenziazione.
 *
 * Inseguimento e "camera libera": durante un'azione del player la regia segue l'azione (ogni passo
 * chiama focus/flyTo). Se lo studente si muove da solo (release(), chiamata dalla navigazione), la
 * camera diventa libera: i voli chiesti dall'AZIONE non muovono più la camera, si risolvono subito e
 * l'ultimo viene ricordato; resumeFollow() torna a seguire e vola lì. I voli chiesti dallo STUDENTE
 * (menu, pulsanti, "Centra": la navigazione li segnala con noteUserIntent()) partono sempre.
 * A fine azione (o quando si ferma) l'inseguimento torna automatico.
 */
import * as THREE from "three";
import { prefersReducedMotion } from "./dom.js";
import { COLORS } from "./visuals.js";

/** Con "riduci il movimento" attivo nel sistema, i voli durano al massimo così (secondi). */
const REDUCED_MOTION_MAX_FLIGHT = 0.3;

const DEFAULT_DIR = new THREE.Vector3(0.18, 0.22, 1).normalize(); // leggermente dall'alto e di lato

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * Bounding box dei soli mesh. Sono esclusi sprite/etichette, gli sfondi dei blocchi
 * (userData.isFrame) e gli oggetti marcati userData.noBounds.
 * Un oggetto con userData.members usa l'unione dei membri (gruppi "virtuali").
 * Con { visibleOnly: true } conta solo ciò che è visibile (utile per inquadrare un livello LOD).
 * Un InstancedMesh conta tutte le sue istanze (instancedBounds), non solo la geometria di una.
 */
export function meshBounds(object, { visibleOnly = false } = {}) {
  const box = new THREE.Box3();
  const part = new THREE.Box3();
  const visit = (child, isRoot) => {
    // visibleOnly: salta i sotto-alberi nascosti (ma non la radice, es. un livello LOD ancora in dissolvenza)
    if (visibleOnly && !isRoot && !child.visible) return;
    if (child.isMesh && !child.userData.isFrame && !child.userData.noBounds) {
      if (child.isInstancedMesh) {
        const local = instancedBounds(child);
        if (!local.isEmpty()) box.union(part.copy(local).applyMatrix4(child.matrixWorld));
      } else {
        if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
        box.union(part.copy(child.geometry.boundingBox).applyMatrix4(child.matrixWorld));
      }
    }
    for (const c of child.children) visit(c, false);
  };
  for (const target of object.userData?.members ?? [object]) {
    target.updateWorldMatrix(true, true);
    visit(target, true);
  }
  return box;
}

const instanceCache = new WeakMap(); // InstancedMesh → { version, count, geometry, box }
const instanceMatrix = new THREE.Matrix4();
const instanceBox = new THREE.Box3();

/**
 * Box (coordinate locali del mesh) di tutte le istanze visibili di un InstancedMesh: le prime `count`,
 * saltando quelle con scala zero (il modo comune di nasconderne una: il suo box sarebbe un punto, magari
 * nell'origine, e allargherebbe l'inquadratura verso il nulla). Ricalcolato solo se le matrici delle
 * istanze (versione dell'attributo), il loro numero o la geometria cambiano: la navigazione chiede i
 * confini ogni 3 s e un mesh con migliaia di istanze non deve costare a ogni richiesta.
 */
export function instancedBounds(mesh) {
  const geometry = mesh.geometry;
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  const cached = instanceCache.get(mesh);
  if (cached && cached.version === mesh.instanceMatrix.version && cached.count === mesh.count && cached.geometry === geometry) return cached.box;
  const box = cached?.box ?? new THREE.Box3();
  box.makeEmpty();
  for (let i = 0; i < mesh.count; i++) {
    mesh.getMatrixAt(i, instanceMatrix);
    if (Math.abs(instanceMatrix.determinant()) < 1e-12) continue; // istanza nascosta con scala zero
    box.union(instanceBox.copy(geometry.boundingBox).applyMatrix4(instanceMatrix));
  }
  instanceCache.set(mesh, { version: mesh.instanceMatrix.version, count: mesh.count, geometry, box });
  return box;
}

export class CameraDirector {
  /** @param {import("./sceneHost.js").SceneHost} host */
  constructor(host) {
    this.host = host;
    this.camera = host.camera;
    this.controls = host.controls;
    this.flight = null;
    this.actionActive = false; // un'azione del player è in corso (layout.js lo collega al player)
    this.following = true; // false = camera libera: i voli dell'azione non muovono la camera
    this.pendingFlight = null; // ultimo volo chiesto dall'azione mentre la camera era libera
    this.lastActionFlight = null; // ultimo volo dell'azione partito davvero (per "Segui l'azione" senza voli in sospeso)
    this.userIntentUntil = 0; // fino a quando (performance.now) un volo è "chiesto dallo studente"
    this.intentStamp = 0; // cresce a ogni gesto dello studente (noteUserIntent)
    this.followListeners = new Set();
    this.highlight = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: COLORS.highlight, transparent: true, opacity: 0.9 }),
    );
    this.highlight.visible = false;
    this.highlight.userData.noBounds = true;
    host.scene.add(this.highlight);
    host.onFrame((dt, elapsed) => this.update(dt, elapsed));
    // Trascinare la scena interrompe il volo: lo fa la navigazione (js/core/navigation.js) al primo
    // movimento vero del puntatore, così un semplice clic su un oggetto non lascia la camera a metà volo.
  }

  /**
   * Vola su un oggetto (o su un Box3) e lo incornicia.
   * @returns {Promise<void>} si risolve a fine volo
   */
  focus(target, { showFrame = true, margin = 1.35, direction = DEFAULT_DIR, minDistance = 4.5, duration = 1.2 } = {}) {
    const box = target?.isBox3 ? target : meshBounds(target);
    if (box.isEmpty()) return Promise.resolve();
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fitH = size.y / 2 / Math.tan(fov / 2);
    const fitW = size.x / 2 / (Math.tan(fov / 2) * this.host.visibleAspect());
    const distance = Math.max(fitH, fitW, minDistance) * margin + size.z;

    this.highlight.visible = showFrame;
    const pad = 0.25;
    this.highlight.position.copy(center);
    this.highlight.scale.set(size.x + pad, size.y + pad, Math.max(size.z, 0.1) + pad);
    return this.flyTo(center.clone().addScaledVector(direction.clone().normalize(), distance), center, duration);
  }

  /** Vola verso una posizione di camera precisa guardando target. */
  flyTo(position, target, duration = 1.2) {
    if (this.holdsActionFlights()) {
      // camera libera: il volo dell'azione non parte, ma lo ricordiamo per "Segui l'azione"
      this.pendingFlight = { position: position.clone(), target: target.clone(), duration };
      return Promise.resolve();
    }
    const byUser = performance.now() <= this.userIntentUntil;
    if (this.actionActive && !byUser) this.lastActionFlight = { position: position.clone(), target: target.clone(), duration };
    if (prefersReducedMotion()) duration = Math.min(duration, REDUCED_MOTION_MAX_FLIGHT);
    this.flight?.resolve?.();
    return new Promise((resolve) => {
      this.flight = {
        intent: byUser ? this.intentStamp : 0, // volo chiesto da questo gesto dello studente
        t: 0,
        duration: Math.max(0.01, duration),
        fromPos: this.camera.position.clone(),
        fromTarget: this.controls.target.clone(),
        toPos: position.clone(),
        toTarget: target.clone(),
        resolve,
      };
    });
  }

  hideFrame() {
    this.highlight.visible = false;
  }

  update(dt, elapsed) {
    if (this.highlight.visible) this.highlight.material.opacity = 0.55 + 0.4 * Math.sin(elapsed * 4);
    // Il volo avanza con il tempo reale: il dt del SceneHost è limitato a 0,05 s, e con frame lenti
    // (rendering software, macchina carica) un volo di 1,2 s durerebbe 3-4 s. Il limite di 0,25 s
    // evita un salto alla fine quando il tab torna visibile dopo una pausa.
    const now = performance.now();
    const realDt = this.lastUpdate === undefined ? dt : Math.min(0.25, Math.max(dt, (now - this.lastUpdate) / 1000));
    this.lastUpdate = now;
    if (!this.flight) return;
    const f = this.flight;
    f.t = Math.min(1, f.t + realDt / f.duration);
    const k = easeInOutCubic(f.t);
    this.camera.position.lerpVectors(f.fromPos, f.toPos, k);
    this.controls.target.lerpVectors(f.fromTarget, f.toTarget, k);
    if (f.t >= 1) {
      this.flight = null;
      f.resolve();
    }
  }

  /** Interrompe il volo in corso (lo studente si è mosso: trascinamento, rotella, tasti, pad). */
  cancel() {
    const f = this.flight;
    this.flight = null;
    f?.resolve?.();
  }

  /** true mentre un volo è in corso (la camera cambia a ogni frame). */
  get flying() {
    return this.flight !== null;
  }

  /**
   * Porta subito il volo in corso alla meta e lo risolve (es. una vista alternativa copre la scena: lo
   * studente tornerà lì, non a metà strada). Restituisce true se c'era un volo.
   */
  finish() {
    const f = this.flight;
    if (!f) return false;
    this.camera.position.copy(f.toPos);
    this.controls.target.copy(f.toTarget);
    this.cancel();
    return true;
  }

  /** true dentro la finestra dell'ultimo noteUserIntent: un volo chiesto adesso è "dello studente". */
  get intentActive() {
    return performance.now() <= this.userIntentUntil;
  }

  /** Numero dell'ultimo gesto dello studente (cresce a ogni noteUserIntent): per riconoscere un gesto già usato. */
  get lastIntent() {
    return this.intentStamp;
  }

  // ---------------------------------------------------------------- inseguimento / camera libera
  /** true se la camera segue l'azione (o se non c'è nessuna azione). */
  get follow() {
    return this.following;
  }

  /** Il player segnala se un'azione è in corso; ogni nuova azione riparte seguendo, e a fine azione si torna a seguire. */
  setActionActive(active) {
    active = Boolean(active);
    if (active === this.actionActive) return;
    this.actionActive = active;
    this.pendingFlight = null;
    this.lastActionFlight = null;
    this.setFollowing(true);
  }

  /** Lo studente si è mosso da solo: durante un'azione la camera diventa libera. */
  release() {
    if (this.actionActive) this.setFollowing(false);
  }

  /**
   * "Segui l'azione": torna a seguire e vola sull'ultimo bersaglio chiesto dall'azione mentre la camera
   * era libera. Se l'azione non ne ha chiesti (il passo inquadra ancora la stessa cosa), si torna
   * sull'ultima inquadratura dell'azione, a meno che il tab, avvisato da onFollowChange, non abbia già
   * fatto partire un volo suo.
   */
  resumeFollow() {
    const pending = this.pendingFlight;
    this.pendingFlight = null;
    const before = this.flight;
    this.setFollowing(true);
    const goal = pending ?? (this.actionActive && this.flight === before ? this.lastActionFlight : null);
    return goal ? this.flyTo(goal.position, goal.target, Math.min(goal.duration, 1)) : Promise.resolve();
  }

  /**
   * I voli chiesti nei prossimi `ms` millisecondi vengono dallo studente (un clic sul menu, su un
   * pulsante, "Centra"…) e partono anche con la camera libera.
   */
  noteUserIntent(ms = 700) {
    this.userIntentUntil = Math.max(this.userIntentUntil, performance.now() + ms);
    this.intentStamp++;
  }

  /** fn(following) a ogni cambio tra inseguimento e camera libera. Restituisce la funzione di rimozione. */
  onFollowChange(fn) {
    this.followListeners.add(fn);
    return () => this.followListeners.delete(fn);
  }

  /**
   * true se un volo chiesto ora va trattenuto: camera libera durante un'azione e nessun gesto recente
   * dello studente. Dentro la finestra del gesto vale comunque se il volo dello studente è ancora in
   * corso: il passo dell'azione che scatta 300 ms dopo "Centra" non deve sostituirlo.
   */
  holdsActionFlights() {
    if (!this.actionActive || this.following) return false;
    if (performance.now() > this.userIntentUntil) return true;
    return this.flight !== null && this.flight.intent === this.intentStamp;
  }

  setFollowing(following) {
    if (following === this.following) return;
    this.following = following;
    for (const fn of this.followListeners) fn(following);
  }
}
