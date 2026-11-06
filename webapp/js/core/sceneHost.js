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
import { prefersReducedMotion } from "./dom.js";
import { frameDamping } from "./navMath.js";

/** Smorzamento delle OrbitControls a 60 fps (poi adattato al frame rate reale, vedi frameDamping). */
const DAMPING_60FPS = 0.08;

let softwareRenderer = null;

/**
 * true se WebGL gira senza GPU (SwiftShader, llvmpipe…: macchine virtuali, PC senza driver).
 * Lì l'antialiasing MSAA raddoppia il tempo di ogni frame (misurato: 40 → 22 ms nel tab
 * Transformer), mentre con una GPU vera costa poco: lo teniamo solo dove non pesa.
 */
function isSoftwareRenderer() {
  if (softwareRenderer !== null) return softwareRenderer;
  softwareRenderer = false;
  try {
    const gl = document.createElement("canvas").getContext("webgl2") ?? document.createElement("canvas").getContext("webgl");
    const info = gl?.getExtension("WEBGL_debug_renderer_info");
    const name = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl?.getParameter(gl.RENDERER);
    softwareRenderer = /swiftshader|llvmpipe|software|softpipe/i.test(String(name ?? ""));
    gl?.getExtension("WEBGL_lose_context")?.loseContext(); // libera subito il contesto di prova
  } catch {
    // nessun WebGL: ci penserà il renderer a segnalarlo
  }
  return softwareRenderer;
}

export class SceneHost {
  /**
   * @param {HTMLElement} container elemento che ospita il canvas
   * @param {object} [options]
   * @param {number} [options.background] colore di sfondo
   * @param {() => number} [options.coveredRight] pixel coperti a destra da pannelli fluttuanti
   * @param {number} [options.fov]
   * @param {boolean} [options.antialias] predefinito: sì con una GPU, no con il rendering software
   */
  constructor(container, { background = 0x0d1117, coveredRight = () => 0, fov = 45, antialias = !isSoftwareRenderer() } = {}) {
    this.container = container;
    this.coveredRight = coveredRight;
    this.updaters = new Set();
    this.resizeListeners = new Set();
    this.stopListeners = new Set();
    this.clock = new THREE.Clock();
    this.running = false;

    this.renderer = new THREE.WebGLRenderer({ antialias });
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
    // Comandi del mouse (la navigazione, js/core/navigation.js, li adatta alla modalità Ruota/Sposta):
    // sinistro = ruota, destro o centrale = sposta, Shift/Ctrl/⌘ + sinistro = sposta, rotella = zoom
    // verso il puntatore (si avvicina a ciò che stai indicando, come in una mappa).
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    // L'inerzia dopo il trascinamento è movimento non chiesto: con "riduci il movimento" si toglie.
    this.controls.enableDamping = !prefersReducedMotion();
    this.controls.dampingFactor = DAMPING_60FPS;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.zoomSpeed = 1.4; // rotella; la navigazione usa 1 per il pizzico a due dita
    this.controls.mouseButtons.MIDDLE = THREE.MOUSE.PAN;
    // Mai fino a distanza 0 dal punto guardato: lì la rotella non può più allontanare la camera (lo zoom
    // è proporzionale alla distanza). I tab con scene grandi alzano il minimo (es. dentro i blocchi).
    this.controls.minDistance = 1;

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
    for (const fn of this.resizeListeners) fn();
  }

  /** fn() dopo ogni ridimensionamento (anche quando un pannello cambia la parte coperta). Restituisce la rimozione. */
  onResize(fn) {
    this.resizeListeners.add(fn);
    return () => this.resizeListeners.delete(fn);
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
      this.controls.dampingFactor = frameDamping(DAMPING_60FPS, dt);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  /**
   * Foto della scena come data URL PNG. Il canvas WebGL non conserva l'immagine tra un frame e
   * l'altro (preserveDrawingBuffer costerebbe a ogni frame): si disegna e si copia nello stesso istante.
   */
  snapshot(type = "image/png") {
    this.renderer.render(this.scene, this.camera);
    const canvas = this.renderer.domElement;
    // Solo la parte che si vede: quella sotto il pannello del codice o dei grafici sarebbe una striscia vuota.
    const scale = canvas.width / Math.max(1, this.container.clientWidth);
    const covered = Math.round(this.coveredRight() * scale);
    if (covered <= 0 || covered >= canvas.width) return canvas.toDataURL(type);
    const crop = document.createElement("canvas");
    crop.width = canvas.width - covered;
    crop.height = canvas.height;
    crop.getContext("2d").drawImage(canvas, 0, 0, crop.width, crop.height, 0, 0, crop.width, crop.height);
    return crop.toDataURL(type);
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    this.renderer.setAnimationLoop(null);
    for (const fn of this.stopListeners) fn();
  }

  /** fn() quando il ciclo si ferma (tab nascosto). Restituisce la funzione di rimozione. */
  onStop(fn) {
    this.stopListeners.add(fn);
    return () => this.stopListeners.delete(fn);
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
