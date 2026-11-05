/**
 * Regia della camera: volo fluido verso un oggetto e riquadro pulsante di evidenziazione.
 */
import * as THREE from "three";
import { COLORS } from "./visuals.js";

const DEFAULT_DIR = new THREE.Vector3(0.18, 0.22, 1).normalize(); // leggermente dall'alto e di lato

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/**
 * Bounding box dei soli mesh. Sono esclusi sprite/etichette, gli sfondi dei blocchi
 * (userData.isFrame) e gli oggetti marcati userData.noBounds.
 * Un oggetto con userData.members usa l'unione dei membri (gruppi "virtuali").
 * Con { visibleOnly: true } conta solo ciò che è visibile (utile per inquadrare un livello LOD).
 */
export function meshBounds(object, { visibleOnly = false } = {}) {
  const box = new THREE.Box3();
  const part = new THREE.Box3();
  const visit = (child, isRoot) => {
    // visibleOnly: salta i sotto-alberi nascosti (ma non la radice, es. un livello LOD ancora in dissolvenza)
    if (visibleOnly && !isRoot && !child.visible) return;
    if (child.isMesh && !child.userData.isFrame && !child.userData.noBounds) {
      if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
      box.union(part.copy(child.geometry.boundingBox).applyMatrix4(child.matrixWorld));
    }
    for (const c of child.children) visit(c, false);
  };
  for (const target of object.userData?.members ?? [object]) {
    target.updateWorldMatrix(true, true);
    visit(target, true);
  }
  return box;
}

export class CameraDirector {
  /** @param {import("./sceneHost.js").SceneHost} host */
  constructor(host) {
    this.host = host;
    this.camera = host.camera;
    this.controls = host.controls;
    this.flight = null;
    this.highlight = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: COLORS.highlight, transparent: true, opacity: 0.9 }),
    );
    this.highlight.visible = false;
    this.highlight.userData.noBounds = true;
    host.scene.add(this.highlight);
    host.onFrame((dt, elapsed) => this.update(dt, elapsed));
    this.controls.addEventListener("start", () => this.cancel());
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
    this.flight?.resolve?.();
    return new Promise((resolve) => {
      this.flight = {
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

  /** Se l'utente trascina la scena durante un volo, il volo si interrompe. */
  cancel() {
    const f = this.flight;
    this.flight = null;
    f?.resolve?.();
  }
}
