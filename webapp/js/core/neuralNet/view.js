/**
 * Vista "Rete neurale" del nucleo: la rete orizzontale con profondità dentro il livello di una
 * rappresentazione alternativa (core/representations.js la importa dinamicamente).
 *
 *   const view = createNetworkView({ host, director, nav, layer, tabId, spec, app, reps });
 *   await view.ready;            // scena costruita e inquadrata di lato (se la specifica è sbagliata:
 *                                // view.error, e la scheda dice cosa non va)
 *   view.focus("attn");          // o ["a", "b"], o { section: "attn", slice: 3 }: vola lì e la descrive
 *   view.onShow(); view.onHide(); view.dispose();
 *
 * host / director / nav sono quelli del livello (SceneHost, CameraDirector, Navigation del nucleo):
 * qui si usano soltanto, non si creano. `layer` è l'elemento in cui la vista mette scheda ed etichetta.
 *
 * Costi: a riposo, a ogni frame, solo il confronto della matrice della camera con l'ultima vista
 * (nessuna allocazione); quando la camera si muove, LOD ed etichette (~0,1-0,3 ms); durante l'onda
 * un'uniform. I numeri misurati sono in view.stats (la pagina tests/neuralnet-demo.html li mostra).
 */
import * as THREE from "three";
import { el, fmt, prefersReducedMotion } from "../dom.js";
import { MODE } from "./activations.js";
import { NetworkCard } from "./card.js";
import { buildNetworkModel } from "./model.js";
import { PATTERNS } from "./roles.js";
import { NetworkScene } from "./scene.js";
import { describeSpecError } from "./spec.js";

/**
 * Direzioni della camera. Di lato: con un filo di prospettiva (si intuisce che c'è spessore). Di tre
 * quarti la camera sta a sinistra e in alto, e x sale sullo schermo di sin(azimut)·sin(elevazione): per
 * tutta la rete un angolo dolce (il flusso resta quasi orizzontale), per una sezione sola uno più deciso
 * (le facce delle lastre, cioè neuroni × profondità, si vedono meglio).
 */
export const VIEW_DIRECTIONS = Object.freeze({
  side: Object.freeze([-0.07, 0.1, 1]),
  thick: Object.freeze([-0.366, 0.438, 0.821]), // azimut −24°, elevazione 26°
  thickSection: Object.freeze([-0.6, 0.45, 0.66]), // azimut −42°, elevazione 27°
});
const WAVE_SECONDS = 7; // una passata dell'onda a velocità 1×
const CLICK_PX = 5;
const CLICK_MS = 650;
const TOUCH_TIP_MS = 2600;

export function createNetworkView(ctx) {
  const { host, director, nav, layer, spec } = ctx;
  const camera = host.camera;
  const canvas = host.renderer.domElement;
  const disposers = [];
  const stats = { buildMs: 0, model: null, tickMs: 0, updateMs: 0, updates: 0, pickMs: 0, units: 0, segments: 0, arcs: 0 };
  const state = { playing: false, front: -Infinity, waveFrom: 0, waveTo: 0, speed: 1, lastNow: 0, waveSection: -1, selected: null, tipTimer: 0, hoverRaf: 0, pointer: null, down: null };
  const lastCamera = new Float64Array(19).fill(NaN); // doppia precisione: un Float32Array arrotonderebbe e il confronto fallirebbe sempre
  let model = null;
  let scene = null;
  let card = null;
  let disposed = false;
  const tip = el("div", { className: "nn-tip", role: "tooltip", hidden: true });
  layer.append(tip);

  const view = {
    ready: null,
    error: null,
    get model() {
      return model;
    },
    get scene() {
      return scene;
    },
    stats,
    focus,
    onShow,
    onHide,
    dispose,
    /** Comandi della vista, anche per gli script di verifica: play(), side(), thick(), setSlice(t), setDensity(f), select(id). */
    play: () => togglePlay(),
    side: () => frameNetwork("side"),
    thick: () => frameNetwork("thick"),
    setSlice: (t) => setSlice(t),
    setDensity: (f) => setDensity(f),
    select: (id) => select(model?.net.byId.get(id)?.kind ?? "section", id),
    /** Microsecondi per frame del ciclo a riposo (n giri sincroni: performance.now è troppo grossolano per un giro solo). */
    measureIdle(n = 20000) {
      if (!scene) return NaN;
      tick();
      const t0 = performance.now();
      for (let i = 0; i < n; i++) tick();
      return ((performance.now() - t0) * 1000) / n;
    },
  };

  view.ready = (async () => {
    // la costruzione è sincrona e pesa (fino a ~100 ms per 60.000 neuroni): prima lascia disegnare la scheda vuota
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    if (disposed) return;
    const t0 = performance.now();
    try {
      model = buildNetworkModel(spec);
    } catch (error) {
      view.error = error;
      card = new NetworkCard(layer, null, {});
      card.showError(spec?.title || "Rete neurale", describeSpecError(error));
      return;
    }
    scene = new NetworkScene(model);
    host.scene.add(scene.root);
    if (prefersReducedMotion()) scene.setGlow(false); // l'onda resta (la chiede lo studente), senza bagliori
    card = new NetworkCard(layer, model, {
      play: togglePlay,
      speed: (v) => (state.speed = v),
      slice: (t) => setSlice(t),
      density: (f) => setDensity(f),
      thick: () => frameNetwork("thick"),
      side: () => frameNetwork("side"),
      select: (kind, id, o) => select(kind, id, o),
      frame: (target, how) => frameTarget(target, how),
      clear: () => clearSelection(),
    });
    card.setDensity(scene.density, scene.drawn);
    stats.buildMs = performance.now() - t0;
    stats.model = model.timings;
    stats.units = model.net.units;
    stats.segments = model.links.segments.count;
    stats.arcs = model.links.arcs.count;

    // l'onda parte prima della prima sezione e finisce dopo l'ultima
    const [x0, x1] = scene.xRange;
    const w = scene.uniforms.uWaveWidth.value;
    state.waveFrom = x0 - 2 * w;
    state.waveTo = x1 + 2 * w;
    nav.setBounds(() => scene.box().expandByScalar(0.8));
    nav.setHome(() => frameNetwork("side", { whole: true })); // "Centra": sempre tutta la rete
    // si gira attorno alla rete fin quasi di fianco e dall'alto (la griglia esperti × token si legge dall'alto),
    // mai da dietro: il flusso si leggerebbe al contrario
    nav.setOrbitLimits({ turn: 85, tilt: 82, minDistance: 0.5 });
    frameNetwork("side", { immediate: true });
    disposers.push(host.onFrame(tick));
    bindPointer();
  })();

  // ---------------------------------------------------------------- inquadrature
  /**
   * Inquadra un insieme di punti dalla direzione `dir`, dentro la parte dello stage non coperta dalla
   * scheda: i punti si proiettano sulla base della camera (destra, su, profondità); si sceglie la
   * distanza che li fa entrare e si sposta il punto guardato perché il loro centro cada nel centro
   * della parte libera.
   */
  function framePoints(points, dir, { duration = 1.1, immediate = false, margin = 1.08 } = {}) {
    if (!points.length) return Promise.resolve();
    const W = Math.max(1, host.container.clientWidth);
    const H = Math.max(1, host.container.clientHeight);
    const free = card?.freeRect(host.container) ?? { x0: 0, y0: 0, x1: W, y1: H };
    const fw = Math.max(80, free.x1 - free.x0);
    const fh = Math.max(80, free.y1 - free.y0);
    const d = new THREE.Vector3(...dir).normalize();
    const upWorld = Math.abs(d.y) > 0.98 ? new THREE.Vector3(0, 0, -1) : new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(upWorld, d).normalize();
    const up = new THREE.Vector3().crossVectors(d, right).normalize();
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const p of points) {
      const a = [p.dot(right), p.dot(up), p.dot(d)];
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], a[k]);
        hi[k] = Math.max(hi[k], a[k]);
      }
    }
    // centro dell'ingombro nel piano della camera, a metà profondità
    const center = new THREE.Vector3().addScaledVector(right, (lo[0] + hi[0]) / 2).addScaledVector(up, (lo[1] + hi[1]) / 2).addScaledVector(d, (lo[2] + hi[2]) / 2);
    const halfR = (hi[0] - lo[0]) / 2;
    const halfU = (hi[1] - lo[1]) / 2;
    const halfD = (hi[2] - lo[2]) / 2;
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const tanH = tanV * (W / H);
    const distance = Math.max(2, Math.max(halfU / (tanV * (fh / H)), halfR / (tanH * (fw / W))) * margin + halfD);
    const cx = ((free.x0 + free.x1) / 2 / W) * 2 - 1;
    const cy = 1 - ((free.y0 + free.y1) / 2 / H) * 2;
    const target = center.clone().addScaledVector(right, -cx * distance * tanH).addScaledVector(up, -cy * distance * tanV);
    const position = target.clone().addScaledVector(d, distance);
    if (immediate) {
      director.cancel();
      camera.position.copy(position);
      host.controls.target.copy(target);
      camera.lookAt(target);
      host.controls.update();
      return Promise.resolve();
    }
    return director.flyTo(position, target, duration);
  }

  /** La sezione scelta (o, senza scelta o con whole, tutta la rete) di lato o di tre quarti. */
  function frameNetwork(how, { whole = false, ...options } = {}) {
    if (!scene) return Promise.resolve();
    director.noteUserIntent();
    const part = whole ? null : state.selected;
    const points = part ? scene.framePoints(selectionIds(part)) : scene.framePoints();
    const dir = how === "thick" && part ? VIEW_DIRECTIONS.thickSection : VIEW_DIRECTIONS[how] ?? VIEW_DIRECTIONS.side;
    return framePoints(points, dir, { margin: part ? 1.3 : 1.06, ...options });
  }

  function frameTarget(target, how) {
    const dir = how === "thick" ? VIEW_DIRECTIONS.thickSection : currentDirection();
    director.noteUserIntent();
    return framePoints(scene.framePoints(selectionIds(target)), dir, { margin: 1.3 });
  }

  const selectionIds = (target) => (target.kind === "group" ? groupSections(model.net.groups[target.id]).map((s) => s.id) : [target.id]);

  /** La direzione da cui guarda la camera adesso (per inquadrare senza cambiare angolo). */
  function currentDirection() {
    const d = new THREE.Vector3().subVectors(camera.position, host.controls.target).normalize();
    return [d.x, d.y, d.z];
  }

  /** Le sezioni sotto la parentesi di un gruppo (fino a una ripetizione compresa: il suo blocco sta prima). */
  const groupSections = (g) => model.net.sections.filter((s) => s.order >= g.from.order && s.order <= g.to.order);

  // ---------------------------------------------------------------- ciclo per frame
  function tick() {
    const t0 = performance.now();
    if (state.playing) advanceWave(t0);
    if (cameraChanged()) {
      const t1 = performance.now();
      scene.update(camera, host.container.clientWidth, host.container.clientHeight, host.renderer.getPixelRatio());
      stats.updateMs = ema(stats.updateMs, performance.now() - t1);
      stats.updates++;
    }
    stats.tickMs = ema(stats.tickMs, performance.now() - t0);
  }

  /**
   * true se camera, proiezione o dimensioni dello stage sono cambiate (19 numeri, nessuna allocazione).
   * Con una tolleranza: le OrbitControls riscrivono la posizione a ogni frame passando per le coordinate
   * sferiche, e l'arrotondamento cambia le ultime cifre anche a camera ferma.
   */
  function cameraChanged() {
    camera.updateMatrixWorld();
    const m = camera.matrixWorld.elements;
    let changed = scene.needsUpdate === true;
    for (let i = 0; i < 16; i++) {
      if (Math.abs(lastCamera[i] - m[i]) > 1e-6 * (1 + Math.abs(m[i])) || lastCamera[i] !== lastCamera[i]) {
        lastCamera[i] = m[i];
        changed = true;
      }
    }
    changed = remember(16, camera.projectionMatrix.elements[0]) || changed;
    changed = remember(17, host.container.clientWidth) || changed;
    changed = remember(18, host.container.clientHeight) || changed;
    scene.needsUpdate = false;
    return changed;
  }

  function remember(i, value) {
    if (lastCamera[i] === value) return false;
    lastCamera[i] = value;
    return true;
  }

  // ---------------------------------------------------------------- onda di attivazione
  function togglePlay() {
    if (!scene) return;
    if (state.playing) {
      state.playing = false;
      card.setPlaying(false);
      card.setStatus("Onda in pausa: premi di nuovo per continuare.");
      return;
    }
    if (!(state.front > state.waveFrom && state.front < state.waveTo)) state.front = state.waveFrom; // ricomincia da capo
    state.playing = true;
    state.lastNow = performance.now();
    state.waveSection = -1;
    card.setPlaying(true);
  }

  function advanceWave(now) {
    // tempo reale (come i voli della camera): con frame lenti l'onda non rallenta
    const dt = Math.min(0.25, (now - state.lastNow) / 1000);
    state.lastNow = now;
    state.front += ((state.waveTo - state.waveFrom) / WAVE_SECONDS) * state.speed * dt;
    if (state.front > state.waveTo) {
      state.playing = false;
      state.front = -Infinity;
      scene.setWave(null);
      card.setPlaying(false);
      card.setStatus("L'onda è arrivata all'uscita. Ogni neurone ora mostra il suo valore.");
      return;
    }
    scene.setWave(state.front);
    // la sezione raggiunta dal fronte, scritta nella scheda solo quando cambia
    const boxes = model.layout.sections;
    let k = -1;
    for (let i = 0; i < boxes.length; i++) if (boxes[i].x0 <= state.front) k = i;
    if (k !== state.waveSection && k >= 0) {
      state.waveSection = k;
      const s = model.net.sections[k];
      const deep = model.net.links.filter((l) => l.to === s && PATTERNS[l.pattern].depth).map((l) => PATTERNS[l.pattern].label.toLowerCase());
      card.setStatus(`L'onda è in: ${s.label}${deep.length ? ` (${[...new Set(deep)].join(", ")} in profondità)` : ""}.`);
    }
  }

  // ---------------------------------------------------------------- fetta, densità, selezione
  function setSlice(t) {
    if (!scene) return;
    scene.setSlice(t);
    scene.needsUpdate = true; // le etichette dei nomi della fetta cambiano
    card.setSlice(scene.slice);
    const names = model.net.slices.names;
    card.setStatus(scene.slice === null ? "" : `Fetta ${scene.slice + 1}: ${names[scene.slice]}. Segui il suo percorso da sinistra a destra; gli altri neuroni si spengono.`);
  }

  function setDensity(f) {
    if (!scene) return;
    const drawn = scene.setDensity(f);
    card.setDensity(scene.density, drawn);
  }

  function select(kind, id, { frame = false } = {}) {
    if (!scene) return false;
    const net = model.net;
    if (kind === "section") {
      const s = net.byId.get(id);
      if (!s || s.kind !== "section") return false;
      state.selected = { kind, id };
      scene.setEmphasis([s.index]);
      const describe = (l, other) => `${other.label} (${l.label || PATTERNS[l.pattern].label.toLowerCase()})`;
      card.showSection(s, {
        incoming: net.links.filter((l) => l.to === s && l.from !== s).map((l) => describe(l, l.from)),
        outgoing: net.links.filter((l) => l.from === s && l.to !== s).map((l) => describe(l, l.to)),
        simulated: model.values.simulated.has(s.id),
      });
    } else if (kind === "repeat") {
      const rep = net.repeats.find((r) => r.id === id);
      if (!rep) return false;
      state.selected = { kind, id };
      scene.setEmphasis(rep.of.map((s) => s.index));
      card.showRepeat(rep, model.layout.repeats[rep.index].shown);
    } else if (kind === "group") {
      const g = net.groups[id];
      if (!g) return false;
      state.selected = { kind, id };
      const sections = groupSections(g);
      scene.setEmphasis(sections.map((s) => s.index));
      card.showGroup(g, sections);
    } else return false;
    if (frame) frameTarget(state.selected, "keep");
    return true;
  }

  function clearSelection() {
    state.selected = null;
    scene?.setEmphasis(null);
    card?.clearDetail();
  }

  /**
   * focus(target): "id" | ["id", …] | { section: "id", slice: n }. Vola sulla sezione (o sull'unione)
   * mantenendo l'angolo della camera e ne mostra la scheda. false se non conosce l'id.
   */
  async function focus(target) {
    await view.ready;
    if (!scene) return false;
    const o = typeof target === "string" || Array.isArray(target) ? { section: target } : target ?? {};
    const ids = (Array.isArray(o.section) ? o.section : [o.section]).filter((id) => model.net.byId.has(id));
    if (o.slice !== undefined) setSlice(o.slice);
    if (!ids.length) return o.slice !== undefined;
    const first = model.net.byId.get(ids[0]);
    select(first.kind, first.id);
    if (ids.length > 1) scene.setEmphasis(ids.map((id) => model.net.byId.get(id)).flatMap((x) => (x.kind === "repeat" ? x.of.map((s) => s.index) : [x.index])));
    director.noteUserIntent();
    await framePoints(scene.framePoints(ids), currentDirection(), { margin: 1.3 });
    return true;
  }

  // ---------------------------------------------------------------- mouse, tocco
  function bindPointer() {
    const on = (target, type, fn, options) => {
      target.addEventListener(type, fn, options);
      disposers.push(() => target.removeEventListener(type, fn, options));
    };
    on(canvas, "pointermove", (e) => {
      if (e.pointerType === "touch") return;
      state.pointer = { x: e.clientX, y: e.clientY };
      if (!state.hoverRaf) state.hoverRaf = requestAnimationFrame(hover);
    });
    on(canvas, "pointerleave", (e) => {
      state.pointer = null;
      // un dito "esce" sempre dopo il tocco: l'etichetta del tocco resta (si chiude da sola dopo 2,6 s)
      if (e.pointerType !== "touch") hideTip();
    });
    on(canvas, "pointerdown", (e) => {
      state.down = { x: e.clientX, y: e.clientY, t: performance.now(), type: e.pointerType };
      if (e.pointerType !== "touch") hideTip();
    });
    on(canvas, "pointerup", (e) => {
      const d = state.down;
      state.down = null;
      if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > CLICK_PX || performance.now() - d.t > CLICK_MS) return;
      click(e.clientX, e.clientY, d.type);
    });
    on(canvas, "pointercancel", () => (state.down = null));
  }

  function hover() {
    state.hoverRaf = 0;
    if (!scene || !state.pointer || state.down) return;
    // durante un volo la camera cambia a ogni frame: l'indice sullo schermo andrebbe rifatto ogni volta
    if (director.flight) return hideTip();
    const { x, y } = state.pointer;
    const rect = canvas.getBoundingClientRect();
    const labelHit = scene.labels?.hit(x - rect.left, y - rect.top);
    canvas.style.cursor = labelHit ? "pointer" : "";
    const t0 = performance.now();
    const u = labelHit ? -1 : scene.pick(x, y, camera, rect);
    stats.pickMs = ema(stats.pickMs, performance.now() - t0);
    if (u < 0) return hideTip();
    showTip(scene.describe(u), x, y);
    canvas.style.cursor = "pointer";
  }

  function click(x, y, pointerType) {
    if (!scene) return;
    const rect = canvas.getBoundingClientRect();
    const hit = scene.labels?.hit(x - rect.left, y - rect.top);
    if (hit?.ref) {
      if (hit.ref.kind === "slice") return setSlice(scene.slice === hit.ref.index ? null : hit.ref.index);
      return select(hit.ref.kind, hit.ref.kind === "group" ? hit.ref.index : hit.ref.id);
    }
    const u = scene.pick(x, y, camera, rect);
    if (u < 0) return clearSelection();
    const info = scene.describe(u);
    select("section", info.section.id);
    if (pointerType === "touch") {
      showTip(info, x, y);
      clearTimeout(state.tipTimer);
      state.tipTimer = setTimeout(hideTip, TOUCH_TIP_MS);
    }
  }

  /** Etichetta al passaggio del mouse: sezione, indice, posizione in profondità, valore. x, y: coordinate client. */
  function showTip(info, clientX, clientY) {
    const s = info.section;
    const lines = [];
    let what;
    if (s.role === "experts") {
      const e = info.expert;
      what = `${e.shared ? `esperto condiviso ${e.index + 1}` : `esperto ${e.index + 1} di ${s.experts}`} · neurone ${info.neuron + 1} di ${s.neurons}`;
      lines.push(e.active ? `acceso per questo ${s.depth.label || "token"}${e.shared ? "" : ` (peso del router ${fmt(e.gate, 2)})`}` : `spento: questo ${s.depth.label || "token"} ha scelto altri esperti`);
    } else if (s.depth.kind === "grid") {
      what = `canale ${info.neuron + 1} di ${s.neurons} · riga ${info.row + 1}, colonna ${info.col + 1} della griglia ${s.depth.rows} × ${s.depth.cols}`;
    } else {
      const slab = s.slabs > 1 ? `${s.slabNames?.[info.slab] ?? `lastra ${info.slab + 1}`} · ` : "";
      const head = info.head !== undefined ? ` (head ${info.head + 1})` : "";
      const name = s.neuronNames?.[info.neuron] ? ` «${s.neuronNames[info.neuron]}»` : "";
      what = `${slab}neurone ${info.neuron + 1}${name} di ${s.neurons}${head}`;
    }
    if (info.depthName) lines.unshift(`${s.depth.label}: ${info.depth + 1} di ${s.depth.count} «${info.depthName}»`);
    const value = info.mode === MODE.off ? "—" : fmt(info.value, Math.abs(info.value) < 0.01 && info.value !== 0 ? 4 : 3);
    tip.replaceChildren(
      el("strong", {}, s.label),
      el("span", {}, what),
      ...lines.map((l) => el("span", {}, l)),
      el("span", { className: "nn-tip-value" }, `valore ${value}${s.valueLabel ? ` ${s.valueLabel}` : ""}${info.simulated ? " · simulato" : ""}`),
    );
    tip.hidden = false;
    // vicino al puntatore (in basso a destra, mai sotto il dito), ma dentro il livello
    const box = layer.getBoundingClientRect();
    const x = clientX - box.left;
    const y = clientY - box.top;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    tip.style.left = `${Math.max(6, Math.min(box.width - w - 6, x + 16))}px`;
    tip.style.top = `${Math.max(6, y + 18 + h > box.height - 6 ? y - h - 14 : y + 18)}px`;
  }

  function hideTip() {
    tip.hidden = true;
  }

  // ---------------------------------------------------------------- ciclo di vita
  function onShow() {
    lastCamera.fill(NaN); // ricalcola LOD ed etichette al primo frame
  }

  function onHide() {
    if (state.playing) {
      state.playing = false;
      card?.setPlaying(false);
    }
    hideTip();
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    for (const off of disposers.splice(0)) off();
    cancelAnimationFrame(state.hoverRaf);
    clearTimeout(state.tipTimer);
    canvas.style.cursor = "";
    scene?.dispose();
    card?.dispose();
    tip.remove();
    nav.setHome(null);
    nav.setBounds(null);
    nav.setOrbitLimits({});
    host.controls.minDistance = 1;
  }

  return view;
}

const ema = (prev, x) => (prev ? prev * 0.9 + x * 0.1 : x);
