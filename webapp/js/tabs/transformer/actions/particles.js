/**
 * Particelle della backward: scorrono sul binario del flusso AL CONTRARIO, dall'uscita verso gli
 * embedding, come i gradienti calcolati da loss.backward(). La luminosità di ogni particella
 * dipende dalla norma del gradiente del gruppo di parametri che sta attraversando (dato vero del
 * trace). Un solo THREE.Points con buffer preallocati: per frame si aggiornano solo i numeri.
 */
import * as THREE from "three";

const BINS = 512;

export class BackwardParticles {
  /**
   * @param {THREE.CurvePath} path il binario di architecture.js
   * @param {number} [count]
   */
  constructor(path, count = 1400) {
    this.path = path;
    this.length = path.getLength();
    this.count = count;
    this.positions = new Float32Array(count * 3);
    this.colors = new Float32Array(count * 3);
    this.distance = new Float32Array(count);
    this.offset = new Float32Array(count * 3);
    this.speed = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      this.distance[i] = Math.random() * this.length;
      this.speed[i] = 0.75 + Math.random() * 0.5;
      this.offset[i * 3] = (Math.random() - 0.5) * 2;
      this.offset[i * 3 + 1] = (Math.random() - 0.5) * 2;
      this.offset[i * 3 + 2] = (Math.random() - 0.5) * 0.8;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("color", new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
    this.points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ size: 0.42, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending }),
    );
    this.points.frustumCulled = false;
    this.points.userData.noBounds = true;
    this.points.visible = false;
    this.intensity = new Float32Array(BINS); // 0..1 lungo il binario
    this.base = new THREE.Color(0xf472b6);
    this.tmp = new THREE.Vector3();
    this.flow = this.length / 6; // unità al secondo: un passaggio completo in circa 6 secondi
  }

  /** `fn(distanza) → 0..1`: intensità del gradiente in ogni punto del binario. */
  setIntensity(fn) {
    for (let b = 0; b < BINS; b++) this.intensity[b] = fn(((b + 0.5) / BINS) * this.length);
  }

  /** Avanza verso l'inizio del binario (gli embedding); chi arriva in fondo riparte dall'uscita. */
  update(dt) {
    for (let i = 0; i < this.count; i++) {
      let d = this.distance[i] - dt * this.flow * this.speed[i];
      if (d < 0) d += this.length;
      this.distance[i] = d;
      const b = Math.min(BINS - 1, Math.max(0, Math.floor((d / this.length) * BINS)));
      const k = this.intensity[b];
      this.path.getPointAt(d / this.length, this.tmp);
      const spread = 0.25 + 1.6 * k;
      this.positions[i * 3] = this.tmp.x + this.offset[i * 3] * spread;
      this.positions[i * 3 + 1] = this.tmp.y + this.offset[i * 3 + 1] * spread;
      this.positions[i * 3 + 2] = this.tmp.z + this.offset[i * 3 + 2];
      const light = 0.3 + 0.7 * k;
      this.colors[i * 3] = this.base.r * light;
      this.colors[i * 3 + 1] = this.base.g * light;
      this.colors[i * 3 + 2] = this.base.b * light;
    }
    this.points.geometry.attributes.position.needsUpdate = true;
    this.points.geometry.attributes.color.needsUpdate = true;
  }

  dispose() {
    this.points.geometry.dispose();
    this.points.material.dispose();
  }
}
