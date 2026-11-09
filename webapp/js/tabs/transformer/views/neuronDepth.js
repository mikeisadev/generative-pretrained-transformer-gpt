/**
 * Disposizione "Orizzontale con profondità" della vista Rete neurale: la rete VERA del GPT del corso
 * (neuronDepthSpec.js) disegnata dal renderer del nucleo (NetworkScene) sulla scena del tab.
 *
 *   const depth = new DepthLayout(tab, { onNeuron, onSlice, onStatus, onPlaying, onBuild, reserved });
 *   await depth.enter({ engine, trace, fraction, slice });  // costruisce, monta, inquadra (o rimette la camera di prima)
 *   await depth.update({ engine, trace, fraction });        // nuovo prompt o nuova quota di pesi: ricostruisce se serve
 *   depth.setSlice(t | null); depth.markNeuron({ section, slab, neuron }, t); depth.togglePlay();
 *   depth.frame("side" | "thick"); depth.home();
 *   depth.exit();                                           // libera la scena (GPU), rimette navigazione e limiti
 *
 * Perché si ricostruisce a ogni prompt invece di ricolorare (NetworkScene.setSectionValues): gli archi
 * di attention (i pesi veri del motore) e i lookup (il carattere di ogni token) cambiano con il prompt,
 * e il renderer li fissa quando costruisce la scena. Ricolorare soltanto lascerebbe archi del prompt
 * precedente. Il modello puro si costruisce in ~20 ms (quota 2%) e resta in memoria: tornando a questa
 * disposizione con lo stesso prompt si rifà solo la scena.
 *
 * Costi: a riposo, a ogni frame, solo il confronto della matrice della camera (nessuna allocazione);
 * quando la camera si muove, LOD ed etichette del renderer; durante l'onda una uniform. I tempi
 * misurati sono in `depth.stats`.
 */
import * as THREE from "three";
import { el, prefersReducedMotion } from "../../../core/dom.js";
import { unitIndex } from "../../../core/neuralNet/layout.js";
import { buildNetworkModel } from "../../../core/neuralNet/model.js";
import { HEAD_COLORS } from "../../../core/neuralNet/roles.js";
import { NetworkScene } from "../../../core/neuralNet/scene.js";
import { DEPTH_DIRECTIONS, framePose } from "./neuronDepthCamera.js";
import { buildDepthSpec, sectionTitle, unitDescription, weightOrders } from "./neuronDepthSpec.js";

export { sectionTitle };

const WAVE_SECONDS = 7; // una passata dell'onda, da sinistra a destra
const CLICK_PX = 5;
const CLICK_MS = 650;
const TOUCH_TIP_MS = 2600;
/** Si gira attorno alla rete fin quasi di fianco e dall'alto, mai da dietro (il flusso si leggerebbe al contrario). */
const ORBIT = { turn: 85, tilt: 82, minDistance: 0.5 };
const FLIGHT = 1.1;

/** Chiave dei pesi del motore: cambia con uno snapshot del training; null = pesi personalizzati (sempre nuovi). */
const weightsKey = (engine) => (engine.weightsSource === "custom" ? null : `${engine.weightsSource}:${engine.snapshotStep ?? ""}`);
const ema = (prev, x) => (prev ? prev * 0.9 + x * 0.1 : x);

export class DepthLayout {
  /**
   * @param {object} tab l'oggetto tab del Transformer
   * @param {object} on
   * @param {(hit: {section: string, slab: number, neuron: number, t: number, unit: number}) => void} on.onNeuron clic su un neurone
   * @param {(t: number|null) => void} on.onSlice clic sul nome di un token nella scena (fetta scelta o nessuna)
   * @param {(text: string) => void} on.onStatus una frase per la riga di stato
   * @param {(playing: boolean) => void} on.onPlaying l'onda parte o si ferma
   * @param {(section: object|null) => void} [on.onSection] clic sull'etichetta di una sezione (o sul vuoto)
   * @param {() => {top: number, bottom: number}} [on.reserved] pixel occupati in alto e in basso dello stage
   */
  constructor(tab, { onNeuron, onSlice, onStatus, onPlaying, onSection = () => {}, reserved = () => ({ top: 60, bottom: 44 }) }) {
    this.tab = tab;
    this.ui = tab.ui;
    this.on = { onNeuron, onSlice, onStatus, onPlaying, onSection, reserved };
    this.camera = this.ui.host.camera;
    this.canvas = this.ui.host.renderer.domElement;
    this.model = null;
    this.tokenNames = null; // caratteri del prompt (la profondità normalizzata li perde con un solo token)
    this.modelKey = undefined;
    this.orders = null;
    this.ordersKey = undefined;
    this.scene = null;
    this.slice = null;
    this.emphasis = null;
    this.savedPose = null; // camera di questa disposizione, ricordata uscendo
    this.saved = null; // navigazione del tab prima di entrare
    this.disposers = [];
    this.wave = { playing: false, front: -Infinity, from: 0, to: 0, lastNow: 0, section: -1 };
    this.pointer = { at: null, down: null, raf: 0, tipTimer: 0 };
    this.lastCamera = new Float64Array(21).fill(NaN); // doppia precisione: un Float32Array arrotonderebbe e il confronto fallirebbe sempre
    this.dirty = true;
    this.stats = { specMs: 0, modelMs: 0, sceneMs: 0, builds: 0, units: 0, segments: 0, sliceSegments: 0, arcs: 0, tickMs: 0, updateMs: 0, updates: 0, pickMs: 0 };
    this.tip = el("div", { className: "tf-depth-tip", role: "tooltip", hidden: true });
    this.ui.stage.append(this.tip);
    this.marker = selectionRing();
    this.marker.visible = false;
    this.mark = null; // neurone scelto { section, slab, neuron, t }: l'anello lo segue anche dopo una ricostruzione
  }

  /** true mentre la scena della disposizione è montata. */
  get active() {
    return this.scene !== null;
  }

  // ---------------------------------------------------------------- modello e scena
  /** Modello puro per questo prompt e questa quota: riusato se non è cambiato niente. */
  build(engine, trace, fraction) {
    const wkey = weightsKey(engine);
    const key = wkey === null ? null : `${wkey}|${fraction}|${trace.ids.join(",")}`;
    if (key !== null && key === this.modelKey && this.model) return false;
    const t0 = performance.now();
    const param = (name) => engine.paramViews.get(name).data;
    if (wkey === null || wkey !== this.ordersKey) {
      this.orders = weightOrders(engine.config, param); // dipende solo dai pesi: si rifà solo se cambiano
      this.ordersKey = wkey;
    }
    const spec = buildDepthSpec({ config: engine.config, vocab: engine.vocab, trace, param, fraction, orders: this.orders });
    const t1 = performance.now();
    this.model = buildNetworkModel(spec);
    this.tokenNames = spec.depth.names;
    this.modelKey = key;
    this.stats.specMs = t1 - t0;
    this.stats.modelMs = performance.now() - t1;
    this.stats.builds++;
    return true;
  }

  /** Crea la NetworkScene dal modello e la mette nella scena del tab, con lo stato visibile di prima. */
  mount() {
    const t0 = performance.now();
    const scene = new NetworkScene(this.model);
    const { min, max } = this.model.layout.bounds;
    // la rete al centro del mondo, come la pila delle altre viste (stesso pavimento, stessa luce)
    scene.root.position.set(-(min[0] + max[0]) / 2, 0, 0);
    if (prefersReducedMotion()) scene.setGlow(false); // l'onda resta (la chiede lo studente), senza bagliori
    // Tutti i collegamenti fissi (somme, copie, LayerNorm per token): il campione casuale predefinito del
    // nucleo ne toglierebbe a caso. Il costo è misurato in depth.stats e nel report.
    scene.setDensity(1);
    if (this.slice !== null) scene.setSlice(this.slice);
    if (this.emphasis && this.model.net.byId.has(this.emphasis)) scene.setEmphasis([this.emphasis]);
    else this.emphasis = null;
    this.ui.host.scene.add(scene.root);
    this.scene = scene;
    const [x0, x1] = scene.xRange;
    const w = scene.uniforms.uWaveWidth.value;
    this.wave.from = x0 - 2 * w;
    this.wave.to = x1 + 2 * w;
    const { links, net } = this.model;
    Object.assign(this.stats, { sceneMs: performance.now() - t0, units: net.units, segments: links.segments.count, sliceSegments: links.sliceSegments.count, arcs: links.arcs.count });
    this.dirty = true;
    this.placeMarker();
  }

  unmount() {
    if (!this.scene) return;
    this.marker.removeFromParent(); // prima: dispose() della scena libera tutto ciò che trova sotto la radice
    this.scene.dispose();
    this.scene = null;
  }

  // ---------------------------------------------------------------- ciclo di vita
  /** Entra nella disposizione: modello, scena, navigazione, camera. */
  async enter({ engine, trace, fraction, slice = null }) {
    if (this.scene) return;
    this.slice = slice;
    this.build(engine, trace, fraction);
    this.mount();
    this.adoptNavigation();
    this.disposers.push(this.ui.host.onFrame(() => this.tick()));
    this.bindPointer();
    if (this.savedPose) this.setPose(this.savedPose);
    else await this.frame("side", { whole: true, immediate: true });
  }

  /** Esce: ricorda la camera, libera tutto ciò che sta sulla GPU, rimette la navigazione del tab. */
  exit() {
    if (!this.scene) return;
    this.stopWave();
    this.hideTip();
    this.savedPose = { position: this.camera.position.clone(), target: this.ui.host.controls.target.clone() };
    for (const off of this.disposers.splice(0)) off();
    cancelAnimationFrame(this.pointer.raf);
    clearTimeout(this.pointer.tipTimer);
    this.pointer.raf = 0;
    this.canvas.style.cursor = "";
    this.unmount();
    this.restoreNavigation();
  }

  /** Nuovo prompt, nuovi pesi o nuova quota: ricostruisce scena e archi (camera, fetta ed enfasi restano). */
  update({ engine, trace, fraction }) {
    if (!this.scene) return false;
    if (!this.build(engine, trace, fraction)) return false;
    this.stopWave();
    const n = this.model.net.slices.count;
    if (this.slice !== null && this.slice >= n) this.slice = null;
    this.unmount();
    this.mount();
    this.ui.nav.setBounds(() => this.bounds()); // la rete può essere più o meno profonda: confini nuovi subito
    return true;
  }

  dispose() {
    this.exit();
    this.tip.remove();
    this.marker.geometry.dispose();
    this.marker.material.map.dispose();
    this.marker.material.dispose();
    this.model = null;
    this.orders = null;
  }

  /**
   * Navigazione per questa disposizione: confini sulla rete, si gira fin quasi di fianco (mai da
   * dietro), ci si avvicina fino ai neuroni più piccoli. Tutto torna com'era in restoreNavigation.
   */
  adoptNavigation() {
    const { nav, host } = this.ui;
    const c = host.controls;
    // Navigation non ha un getter dei confini: si ricorda la sorgente per rimetterla tale e quale (vedi report)
    this.saved = { bounds: nav.boundsSource ?? null, minAz: c.minAzimuthAngle, maxAz: c.maxAzimuthAngle, minPolar: c.minPolarAngle, maxPolar: c.maxPolarAngle, minDistance: c.minDistance };
    nav.setBounds(() => this.bounds());
    nav.setOrbitLimits(ORBIT);
  }

  restoreNavigation() {
    if (!this.saved) return;
    const { nav, host } = this.ui;
    const c = host.controls;
    const s = this.saved;
    nav.setBounds(s.bounds);
    c.minAzimuthAngle = s.minAz;
    c.maxAzimuthAngle = s.maxAz;
    c.minPolarAngle = s.minPolar;
    c.maxPolarAngle = s.maxPolar;
    c.minDistance = s.minDistance;
    this.saved = null;
  }

  /** Confini della navigazione: la rete con assi e nomi dei token, un filo più larga. */
  bounds() {
    return this.scene ? this.scene.box().expandByScalar(0.8) : new THREE.Box3();
  }

  // ---------------------------------------------------------------- camera
  setPose({ position, target }) {
    const { director, host } = this.ui;
    director.cancel();
    this.camera.position.copy(position);
    host.controls.target.copy(target);
    this.camera.lookAt(target);
    host.controls.update();
  }

  /** Frazioni dell'altezza dello stage occupate dai comandi in alto e dalla didascalia in basso. */
  reservedFractions() {
    const h = Math.max(1, this.ui.host.container.clientHeight);
    const { top, bottom } = this.on.reserved();
    return { top: Math.min(0.4, top / h), bottom: Math.min(0.4, bottom / h) };
  }

  /**
   * Inquadra di lato ("side") o di tre quarti ("thick") la sezione scelta o, con whole o senza scelta,
   * tutta la rete. immediate: senza volo (prima apertura).
   */
  frame(kind = "side", { whole = false, immediate = false } = {}) {
    if (!this.scene) return Promise.resolve();
    const part = whole ? null : this.emphasis;
    const points = this.scene.framePoints(part ? [part] : null).map((v) => [v.x, v.y, v.z]);
    const direction = kind === "thick" ? (part ? DEPTH_DIRECTIONS.thickPart : DEPTH_DIRECTIONS.thick) : DEPTH_DIRECTIONS.side;
    const pose = framePose(points, direction, { fov: this.camera.fov, aspect: this.ui.host.visibleAspect(), margin: part ? 1.3 : 1.04, ...this.reservedFractions() });
    const position = new THREE.Vector3(...pose.position);
    const target = new THREE.Vector3(...pose.target);
    if (immediate) {
      this.setPose({ position, target });
      return Promise.resolve();
    }
    this.ui.director.noteUserIntent(); // un volo chiesto dallo studente: parte anche con la camera libera
    return this.ui.director.flyTo(position, target, FLIGHT);
  }

  /** "Centra": tutta la rete di lato. */
  home() {
    return this.frame("side", { whole: true });
  }

  // ---------------------------------------------------------------- stato visibile
  /** Fetta scelta (posizione del prompt) o null = tutte accese; i pesi veri vanno sulla fetta (o sull'ultimo token). */
  setSlice(t) {
    this.slice = t;
    if (!this.scene) return;
    this.scene.setSlice(t);
    this.dirty = true; // i nomi dei token della fetta cambiano: etichette da rifare al prossimo frame
  }

  /** Mette in evidenza una sezione (id) o nessuna. */
  setEmphasis(id) {
    this.emphasis = id ?? null;
    this.scene?.setEmphasis(this.emphasis ? [this.emphasis] : null);
    this.dirty = true;
  }

  /** Anello sul neurone della scheda: sezione, lastra e neurone alla posizione t (null = nessuno). */
  markNeuron(target, t) {
    this.mark = target ? { ...target, t } : null;
    this.placeMarker();
  }

  placeMarker() {
    const s = this.mark && this.model?.net.byId.get(this.mark.section);
    if (!this.scene || !s || this.mark.t >= s.depth.count) {
      this.marker.visible = false;
      return;
    }
    const unit = s.offset + unitIndex(s, this.mark.slab, this.mark.neuron, this.mark.t);
    this.marker.geometry.attributes.position.array.set(this.model.layout.positions.subarray(3 * unit, 3 * unit + 3));
    this.marker.geometry.attributes.position.needsUpdate = true;
    this.scene.root.add(this.marker); // dentro la radice: segue la posizione della rete
    this.marker.visible = true;
  }

  // ---------------------------------------------------------------- ciclo per frame
  tick() {
    const t0 = performance.now();
    if (this.wave.playing) this.advanceWave(t0);
    if (this.cameraChanged()) {
      const t1 = performance.now();
      const { container, renderer } = this.ui.host;
      this.scene.update(this.camera, container.clientWidth, container.clientHeight, renderer.getPixelRatio());
      this.stats.updateMs = ema(this.stats.updateMs, performance.now() - t1);
      this.stats.updates++;
    }
    this.stats.tickMs = ema(this.stats.tickMs, performance.now() - t0);
  }

  /**
   * true se camera, proiezione (anche il view offset del codice) o dimensioni dello stage sono cambiate:
   * 21 numeri confrontati, nessuna allocazione. Con una tolleranza: le OrbitControls riscrivono la
   * posizione a ogni frame passando per le coordinate sferiche e le ultime cifre cambiano anche da ferme.
   */
  cameraChanged() {
    const cam = this.camera;
    cam.updateMatrixWorld();
    const m = cam.matrixWorld.elements;
    const last = this.lastCamera;
    let changed = this.dirty;
    for (let i = 0; i < 16; i++) {
      if (Math.abs(last[i] - m[i]) > 1e-6 * (1 + Math.abs(m[i])) || last[i] !== last[i]) {
        last[i] = m[i];
        changed = true;
      }
    }
    const p = cam.projectionMatrix.elements;
    const { container } = this.ui.host;
    changed = this.remember(16, p[0]) || changed;
    changed = this.remember(17, p[5]) || changed;
    changed = this.remember(18, p[8]) || changed; // view offset: il codice o i grafici coprono una parte dello stage
    changed = this.remember(19, container.clientWidth) || changed;
    changed = this.remember(20, container.clientHeight) || changed;
    this.dirty = false;
    return changed;
  }

  remember(i, value) {
    if (this.lastCamera[i] === value) return false;
    this.lastCamera[i] = value;
    return true;
  }

  // ---------------------------------------------------------------- onda di attivazione
  togglePlay() {
    if (!this.scene) return;
    if (this.wave.playing) {
      this.wave.playing = false;
      this.on.onPlaying(false);
      this.on.onStatus("Onda in pausa: premi di nuovo per continuare.");
      return;
    }
    if (!(this.wave.front > this.wave.from && this.wave.front < this.wave.to)) this.wave.front = this.wave.from; // ricomincia da capo
    this.wave.playing = true;
    this.wave.lastNow = performance.now();
    this.wave.section = -1;
    this.on.onPlaying(true);
  }

  stopWave() {
    if (!this.wave.playing && this.wave.front === -Infinity) return;
    this.wave.playing = false;
    this.wave.front = -Infinity;
    this.scene?.setWave(null);
    this.on.onPlaying(false);
  }

  advanceWave(now) {
    // tempo reale (come i voli della camera): con frame lenti l'onda non rallenta
    const dt = Math.min(0.25, (now - this.wave.lastNow) / 1000);
    this.wave.lastNow = now;
    this.wave.front += ((this.wave.to - this.wave.from) / WAVE_SECONDS) * dt;
    if (this.wave.front > this.wave.to) {
      this.stopWave();
      this.on.onStatus("L'onda è arrivata all'uscita: ogni neurone mostra il suo valore vero.");
      return;
    }
    this.scene.setWave(this.wave.front);
    // la sezione raggiunta dal fronte, detta solo quando cambia
    const boxes = this.model.layout.sections;
    let k = -1;
    for (let i = 0; i < boxes.length; i++) if (boxes[i].x0 <= this.wave.front) k = i;
    if (k >= 0 && k !== this.wave.section) {
      this.wave.section = k;
      this.on.onStatus(`L'onda è in: ${sectionTitle(this.model.net.sections[k])}.`);
    }
  }

  // ---------------------------------------------------------------- mouse e tocco
  bindPointer() {
    const on = (type, fn) => {
      this.canvas.addEventListener(type, fn);
      this.disposers.push(() => this.canvas.removeEventListener(type, fn));
    };
    on("pointermove", (e) => {
      if (e.pointerType === "touch") return;
      this.pointer.at = { x: e.clientX, y: e.clientY };
      if (!this.pointer.raf) this.pointer.raf = requestAnimationFrame(() => this.hover());
    });
    on("pointerleave", (e) => {
      this.pointer.at = null;
      // un dito "esce" sempre dopo il tocco: l'etichetta del tocco resta (si chiude da sola)
      if (e.pointerType !== "touch") this.hideTip();
    });
    on("pointerdown", (e) => {
      this.pointer.down = { x: e.clientX, y: e.clientY, t: performance.now(), type: e.pointerType };
      if (e.pointerType !== "touch") this.hideTip();
    });
    on("pointerup", (e) => {
      const d = this.pointer.down;
      this.pointer.down = null;
      if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > CLICK_PX || performance.now() - d.t > CLICK_MS) return;
      this.click(e.clientX, e.clientY, d.type);
    });
    on("pointercancel", () => (this.pointer.down = null));
  }

  hover() {
    this.pointer.raf = 0;
    if (!this.scene || !this.pointer.at || this.pointer.down) return;
    // durante un volo la camera cambia a ogni frame: l'indice sullo schermo andrebbe rifatto ogni volta
    if (this.ui.director.flight) return this.hideTip();
    const { x, y } = this.pointer.at;
    const rect = this.canvas.getBoundingClientRect();
    const label = this.scene.labels?.hit(x - rect.left, y - rect.top);
    const t0 = performance.now();
    const unit = label ? -1 : this.scene.pick(x, y, this.camera, rect);
    this.stats.pickMs = ema(this.stats.pickMs, performance.now() - t0);
    this.canvas.style.cursor = label?.ref || unit >= 0 ? "pointer" : "";
    if (unit < 0) return this.hideTip();
    this.showTip(this.scene.describe(unit), x, y);
  }

  click(x, y, pointerType) {
    if (!this.scene) return;
    const rect = this.canvas.getBoundingClientRect();
    const label = this.scene.labels?.hit(x - rect.left, y - rect.top);
    if (label?.ref?.kind === "slice") return this.on.onSlice(this.slice === label.ref.index ? null : label.ref.index);
    if (label?.ref?.kind === "section") {
      const section = this.model.net.byId.get(label.ref.id);
      this.setEmphasis(this.emphasis === section.id ? null : section.id);
      return this.on.onSection(this.emphasis ? section : null);
    }
    if (label?.ref?.kind === "group") return this.on.onSection({ group: this.model.net.groups[label.ref.index] });
    const unit = this.scene.pick(x, y, this.camera, rect);
    if (unit < 0) {
      if (this.emphasis) {
        this.setEmphasis(null);
        this.on.onSection(null);
      }
      return;
    }
    const info = this.scene.describe(unit);
    if (pointerType === "touch") {
      // sul telefono la scheda con il calcolo sta nel cassetto chiuso: l'etichetta dice dove trovarla
      this.showTip(info, x, y, { hint: "Il calcolo completo: menu ☰ › Viste" });
      clearTimeout(this.pointer.tipTimer);
      this.pointer.tipTimer = setTimeout(() => this.hideTip(), TOUCH_TIP_MS);
    }
    this.on.onNeuron({ section: info.section.id, slab: info.slab, neuron: info.neuron, t: info.depth, unit }); // la scheda rimette l'anello (markNeuron)
  }

  /** Righe di testo di un'unità (etichetta e scheda): vedi unitDescription in neuronDepthSpec.js. */
  describeUnit(info) {
    return unitDescription(info, this.tokenNames);
  }

  /** Righe di testo del neurone { section, slab, neuron } alla posizione t (per la scheda), o null. */
  describeTarget(target, t) {
    const s = target && this.model?.net.byId.get(target.section);
    if (!this.scene || !s || t >= s.depth.count) return null;
    return this.describeUnit(this.scene.describe(s.offset + unitIndex(s, target.slab, target.neuron, t)));
  }

  /** Etichetta al passaggio del mouse: sezione, lastra e head, indice, token e valore vero (testo, mai HTML). */
  showTip(info, clientX, clientY, { hint = "" } = {}) {
    const d = this.describeUnit(info);
    const head = d.head !== null ? el("span", { className: "tf-depth-head" }, el("i", { style: { background: HEAD_COLORS[d.head % HEAD_COLORS.length] } }), `head ${d.head}`) : null;
    const lines = [
      el("strong", {}, d.title),
      el("span", {}, d.neuron),
      head,
      el("span", {}, d.token),
      el("span", { className: "tf-depth-value" }, d.value),
      hint ? el("span", { className: "tf-depth-hint" }, hint) : null,
    ];
    this.tip.replaceChildren(...lines.filter(Boolean)); // replaceChildren scriverebbe "null" come testo
    this.tip.hidden = false;
    // vicino al puntatore (in basso a destra, mai sotto il dito), dentro lo stage
    const box = this.ui.stage.getBoundingClientRect();
    const px = clientX - box.left;
    const py = clientY - box.top;
    const w = this.tip.offsetWidth;
    const h = this.tip.offsetHeight;
    this.tip.style.left = `${Math.max(6, Math.min(box.width - w - 6, px + 16))}px`;
    this.tip.style.top = `${Math.max(6, py + 18 + h > box.height - 6 ? py - h - 14 : py + 18)}px`;
  }

  hideTip() {
    this.tip.hidden = true;
  }
}

/**
 * Anello attorno al neurone scelto: un punto solo con una texture ad anello e dimensione fissa sullo
 * schermo (i neuroni del renderer smettono di crescere a 20 px: un anello "nel mondo" da vicino
 * diventerebbe enorme, da lontano sparirebbe). Sopra a tutto (depthTest spento), escluso dai confini.
 */
function selectionRing() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 9;
  ctx.strokeStyle = "rgba(2, 6, 23, 0.85)"; // alone scuro: l'anello si legge anche sopra neuroni chiari
  ctx.beginPath();
  ctx.arc(32, 32, 24, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
  const ring = new THREE.Points(geometry, new THREE.PointsMaterial({ size: 30, sizeAttenuation: false, map: texture, transparent: true, depthTest: false, depthWrite: false }));
  ring.frustumCulled = false;
  ring.renderOrder = 6;
  ring.userData.noBounds = true;
  ring.raycast = () => {}; // il doppio clic della navigazione non deve "colpire" l'anello
  return ring;
}
