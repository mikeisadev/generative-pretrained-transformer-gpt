/**
 * Animazione del flusso dei dati: un "pacchetto" di token percorre il binario,
 * dai token di input fino al prossimo token. Quando attraversa un componente,
 * quel componente si illumina e la didascalia in basso dice che cosa sta succedendo.
 */
import * as THREE from "three";
import { COLORS } from "./visuals.js";

const SPEED = 12; // unità di scena al secondo
const PAUSE_AT_END = 1.5;

export class DataFlow {
  /**
   * @param {THREE.CurvePath} path il binario (vedi architecture.js → flowPath)
   * @param {{distance: number, caption: string, object: THREE.Object3D}[]} stations
   */
  constructor(scene, path, stations, onStation) {
    this.path = path;
    this.length = path.getLength();
    this.stations = [...stations].sort((a, b) => a.distance - b.distance);
    this.onStation = onStation;
    this.playing = true;
    this.distance = 0;
    this.pause = 0;
    this.current = null;

    this.packet = new THREE.Group();
    const dotGeo = new THREE.SphereGeometry(0.08, 12, 8);
    const dotMat = new THREE.MeshBasicMaterial({ color: COLORS.flow });
    for (let i = 0; i < 8; i++) {
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.x = -i * 0.16;
      this.packet.add(dot);
    }
    this.packet.add(new THREE.PointLight(COLORS.flow, 4, 4));
    scene.add(this.packet);
    this.moveTo(0);
  }

  toggle() {
    this.playing = !this.playing;
    return this.playing;
  }

  moveTo(distance) {
    const u = THREE.MathUtils.clamp(distance / this.length, 0, 1);
    this.packet.position.copy(this.path.getPointAt(u));
  }

  update(dt, elapsed) {
    this.packet.children.forEach((dot, i) => {
      if (dot.isMesh) dot.position.y = Math.sin(elapsed * 6 + i) * 0.1;
    });
    if (!this.playing) return;

    if (this.pause > 0) {
      this.pause -= dt;
      if (this.pause <= 0) this.distance = 0;
      return;
    }
    this.distance = Math.min(this.length, this.distance + SPEED * dt);
    this.moveTo(this.distance);

    // Stazione attiva = l'ultima superata dal pacchetto
    let active = null;
    for (const s of this.stations) if (s.distance <= this.distance + 1e-6) active = s;
    if (active !== this.current) {
      this.setPulse(this.current, false);
      this.current = active;
      this.setPulse(active, true);
      if (active) this.onStation(active);
    }
    if (this.distance >= this.length) this.pause = PAUSE_AT_END;
  }

  /** Illumina i materiali della stazione moltiplicandone il colore (e poi lo ripristina). */
  setPulse(station, on) {
    if (!station) return;
    station.object.traverse((child) => {
      if (!child.isMesh || child.userData.isFrame || !child.material?.color) return;
      child.userData.baseColor ??= child.material.color.clone();
      child.material.color.copy(child.userData.baseColor);
      if (on) child.material.color.multiplyScalar(1.7);
    });
  }
}
