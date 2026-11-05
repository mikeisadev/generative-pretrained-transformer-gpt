/**
 * SceneHost: tutto ciò che serve a una scena 3D di un tab.
 *
 * Renderer, scena, camera prospettica, OrbitControls, ridimensionamento, ciclo di
 * animazione e "view offset" (il centro della proiezione si sposta a sinistra quando
 * un pannello copre il lato destro, così l'oggetto inquadrato resta visibile).
 *
 * Ogni tab ha il proprio SceneHost. Quando il tab non è visibile il ciclo si ferma
 * (start/stop), così i tab nascosti non consumano GPU.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class SceneHost {
  /**
   * @param {HTMLElement} container elemento che ospita il canvas
   * @param {object} [options]
   * @param {number} [options.background] colore di sfondo
   * @param {() => number} [options.coveredRight] pixel coperti a destra da pannelli fluttuanti
   * @param {number} [options.fov]
   */
  constructor(container, { background = 0x0d1117, coveredRight = () => 0, fov = 45 } = {}) {
    this.container = container;
    this.coveredRight = coveredRight;
    this.updaters = new Set();
    this.clock = new THREE.Clock();
    this.running = false;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(background);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(10, 20, 15);
    this.scene.add(sun);

    this.camera = new THREE.PerspectiveCamera(fov, 1, 0.05, 2000);
    this.camera.position.set(0, 10, 120);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = true;

    this.raycaster = new THREE.Raycaster();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  /** Larghezza/altezza della parte di canvas non coperta dai pannelli. */
  visibleAspect() {
    const w = Math.max(1, this.container.clientWidth - this.coveredRight());
    return w / Math.max(1, this.container.clientHeight);
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.camera.aspect = w / h;
    const covered = this.coveredRight();
    if (covered > 0 && covered < w) this.camera.setViewOffset(w, h, covered / 2, 0, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** Registra una funzione chiamata a ogni frame: fn(dt, elapsed). Restituisce la funzione per rimuoverla. */
  onFrame(fn) {
    this.updaters.add(fn);
    return () => this.updaters.delete(fn);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.clock.getDelta();
    this.renderer.setAnimationLoop(() => {
      const dt = Math.min(this.clock.getDelta(), 0.05);
      const elapsed = this.clock.elapsedTime;
      for (const fn of this.updaters) fn(dt, elapsed);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  stop() {
    this.running = false;
    this.renderer.setAnimationLoop(null);
  }

  /** Distanza della camera dal punto che sta guardando: è lo "zoom" usato dal LOD. */
  zoomDistance() {
    return this.camera.position.distanceTo(this.controls.target);
  }

  /** Oggetti 3D sotto il puntatore (coordinate di un MouseEvent). */
  pick(event, objects, recursive = true) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(pointer, this.camera);
    return this.raycaster.intersectObjects(objects, recursive);
  }

  dispose() {
    this.stop();
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
