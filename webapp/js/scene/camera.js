/**
 * Volo fluido della camera verso un componente ed evidenziazione con un riquadro pulsante.
 */
import * as THREE from "three";
import { COLORS } from "./visuals.js";

const FLY_SECONDS = 1.2;
const VIEW_DIR = new THREE.Vector3(0.18, 0.22, 1).normalize(); // leggermente dall'alto e di lato

const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Bounding box dei soli mesh (le etichette sprite e gli sfondi dei blocchi non contano). */
export function meshBounds(object) {
  const box = new THREE.Box3();
  const targets = object.userData.members ?? [object];
  targets.forEach((target) => {
    target.updateWorldMatrix(true, true);
    target.traverse((child) => {
      if (!child.isMesh || child.userData.isFrame) return;
      child.geometry.computeBoundingBox();
      box.union(child.geometry.boundingBox.clone().applyMatrix4(child.matrixWorld));
    });
  });
  return box;
}

export class CameraDirector {
  /**
   * @param {() => number} visibleAspect rapporto larghezza/altezza della parte di scena
   *   non coperta dalla finestra del codice: l'inquadratura si adatta a quella.
   */
  constructor(camera, controls, scene, visibleAspect = () => camera.aspect) {
    this.camera = camera;
    this.visibleAspect = visibleAspect;
    this.controls = controls;
    this.flight = null;
    this.highlight = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: COLORS.highlight, transparent: true, opacity: 0.9 }),
    );
    this.highlight.visible = false;
    scene.add(this.highlight);
  }

  /** Vola su `object` e lo incornicia. */
  focus(object, { showFrame = true, margin = 1.35 } = {}) {
    const box = meshBounds(object);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fitH = size.y / 2 / Math.tan(fov / 2);
    const fitW = size.x / 2 / (Math.tan(fov / 2) * this.visibleAspect());
    const distance = Math.max(fitH, fitW, 4.5) * margin + size.z;

    this.flight = {
      t: 0,
      fromPos: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      toPos: center.clone().addScaledVector(VIEW_DIR, distance),
      toTarget: center,
    };

    this.highlight.visible = showFrame;
    const pad = 0.25;
    this.highlight.position.copy(center);
    this.highlight.scale.set(size.x + pad, size.y + pad, Math.max(size.z, 0.1) + pad);
  }

  update(dt, elapsed) {
    if (this.highlight.visible) this.highlight.material.opacity = 0.55 + 0.4 * Math.sin(elapsed * 4);
    if (!this.flight) return;
    const f = this.flight;
    f.t = Math.min(1, f.t + dt / FLY_SECONDS);
    const k = easeInOutCubic(f.t);
    this.camera.position.lerpVectors(f.fromPos, f.toPos, k);
    this.controls.target.lerpVectors(f.fromTarget, f.toTarget, k);
    if (f.t >= 1) this.flight = null;
  }

  /** Se l'utente trascina la scena durante un volo, il volo si interrompe. */
  cancel() {
    this.flight = null;
  }
}
