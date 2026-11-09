/**
 * Vista "Rete neurale" del nucleo: la rete orizzontale con profondità dentro il livello di una
 * rappresentazione alternativa (core/representations.js la importa dinamicamente).
 *
 *   const view = createNetworkView({ host, director, nav, layer, tabId, spec, app, reps });
 *   await view.ready;            // scena costruita e inquadrata di lato (se la specifica è sbagliata:
 *                                // view.error, e la scheda dice cosa non va)
 *   view.focus("attn");          // o ["a", "b"], { section: "attn", slice: 3 }, { group: "Encoder" | 2 }: vola lì e la descrive
 *   view.onShow(); view.onHide(); view.dispose();
 *
 * host / director / nav sono quelli del livello (SceneHost, CameraDirector, Navigation del nucleo):
 * qui si usano soltanto, non si creano. `layer` è l'elemento in cui la vista mette scheda ed etichetta.
 *
 * Reti lunghe (framing.js): se la rete intera di lato verrebbe più bassa di un'altezza leggibile, la vista
 * iniziale e "Centra" sono la prima parte (l'ingresso con i primi strati, alta abbastanza da vedere i
 * neuroni); ◀ ▶ della scheda scorrono le parti, "Tutta la rete" mostra l'insieme.
 *
 * Costi: a riposo, a ogni frame, solo il confronto della matrice della camera con l'ultima vista
 * (nessuna allocazione); quando la camera si muove, LOD ed etichette (~0,1-0,3 ms); durante l'onda
 * un'uniform. I numeri misurati sono in view.stats (la pagina tests/neuralnet-demo.html li mostra).
 */
import * as THREE from "three";
import { el, fmt, prefersReducedMotion } from "../dom.js";
import { MODE } from "./activations.js";
import { NetworkCard } from "./card.js";
import { avoidRect, avoidRectVariants, partIndexAt, readableHeight, walkParts } from "./framing.js";
import { attentionSource } from "./honesty.js";
import { buildNetworkModel } from "./model.js";
import { hoverUnit } from "./pickIndex.js";
import { PATTERNS, linkSummary } from "./roles.js";
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
const WHOLE_MARGIN = 1.06; // aria attorno a tutta la rete
const PART_MARGIN = 1.04; // aria attorno a una parte di una rete lunga (la rete continua oltre i bordi)
/** Oltre questa larghezza vista (in parti) si sta guardando la rete d'insieme, non una parte. */
const WHOLE_VIEW = 1.3;
/** Invito a scorrere una rete lunga (riga di stato della scheda, letta anche dagli screen reader). */
const LONG_HINT = "Rete lunga: qui l'ingresso e i primi strati. ▶ prosegue; «Tutta la rete» per l'insieme.";

export function createNetworkView(ctx) {
  const { host, director, nav, layer, spec } = ctx;
  const camera = host.camera;
  const canvas = host.renderer.domElement;
  const disposers = [];
  const stats = { buildMs: 0, model: null, tickMs: 0, updateMs: 0, updates: 0, pickMs: 0, units: 0, segments: 0, arcs: 0 };
  const state = { playing: false, front: -Infinity, waveFrom: 0, waveTo: 0, speed: 1, lastNow: 0, waveSection: -1, selected: null, tipTimer: 0, hoverRaf: 0, pointer: null, down: null, walk: null, walkShown: "" };
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
    /** Comandi della vista, anche per gli script di verifica: play(), side(), thick(), whole(), walk(±1), home(), setSlice(t), setDensity(f), select(id). */
    play: () => togglePlay(),
    side: () => frameNetwork("side"),
    thick: () => frameNetwork("thick"),
    whole: () => showWhole(),
    walk: (step) => walk(step),
    home: () => home(),
    /** Le parti di una rete lunga per lo stage di adesso ({ long, parts, rect }) o null prima di ready. */
    get parts() {
      return scene ? walkPlan() : null;
    },
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
      walk: (step) => walk(step),
      whole: () => showWhole(),
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
    // "Centra": tutta la rete di lato o, se così sarebbe una striscia illeggibile, la sua prima parte
    nav.setHome(() => home());
    // si gira attorno alla rete fin quasi di fianco e dall'alto (la griglia esperti × token si legge dall'alto),
    // mai da dietro: il flusso si leggerebbe al contrario
    nav.setOrbitLimits({ turn: 85, tilt: 82, minDistance: 0.5 });
    // l'invito a scorrere cambia l'altezza della scheda: va scritto prima di calcolare la casa
    let plan = walkPlan();
    if (plan.long) {
      card.setStatus(LONG_HINT);
      state.walk = null;
      plan = walkPlan();
      if (!plan.long) card.setStatus("");
    }
    home({ immediate: true, plan });
    updateWalk();
    disposers.push(host.onFrame(tick));
    // la scheda che si apre o si riduce (e il pad) cambia lo spazio libero per le etichette anche a camera ferma
    const relabel = new ResizeObserver(() => scene && (scene.needsUpdate = true));
    relabel.observe(card.root);
    const pad = layer.querySelector(":scope > .nav-pad");
    if (pad) relabel.observe(pad);
    disposers.push(() => relabel.disconnect());
    bindPointer();
  })();

  // ---------------------------------------------------------------- inquadrature
  /**
   * Inquadra un insieme di punti dalla direzione `dir`, dentro la parte dello stage non coperta dalla
   * scheda: i punti si proiettano sulla base della camera (destra, su, profondità); si sceglie la
   * distanza che li fa entrare e si sposta il punto guardato perché il loro centro cada nel centro
   * della parte libera. Delle parti libere (a destra della scheda, sotto) si usa quella dove i punti
   * vengono più grandi: conta la forma, non solo l'area (una rete lunga e bassa sta meglio sotto).
   */
  function framePoints(points, dir, { duration = 1.1, immediate = false, margin = 1.08, rect = null } = {}) {
    if (!points.length) return Promise.resolve();
    const { W, H, tanV, tanH } = stageView();
    const e = extentOf(points, dir);
    let free = rect;
    let distance = rect ? distanceFor(e, rect, margin) : Infinity;
    if (!rect) {
      for (const r of freeRects({ variants: true })) {
        const dr = distanceFor(e, r, margin);
        if (dr < distance - 1e-6) {
          distance = dr;
          free = r;
        }
      }
    }
    const cx = ((free.x0 + free.x1) / 2 / W) * 2 - 1;
    const cy = 1 - ((free.y0 + free.y1) / 2 / H) * 2;
    const target = e.center.clone().addScaledVector(e.right, -cx * distance * tanH).addScaledVector(e.up, -cy * distance * tanV);
    const position = target.clone().addScaledVector(e.d, distance);
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

  /** Misure dello stage e del campo visivo (in pixel CSS). */
  function stageView() {
    const W = Math.max(1, host.container.clientWidth);
    const H = Math.max(1, host.container.clientHeight);
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    return { W, H, tanV, tanH: tanV * (W / H) };
  }

  /**
   * Le parti dello stage libere dalla scheda (a destra e sotto), o tutto lo stage, senza il pad della
   * navigazione: una striscia inquadrata più grande arriverebbe nell'angolo in basso a destra, sotto di lui.
   * variants: per ogni parte i due tagli attorno al pad (sopra e a sinistra), per chi sceglie quello dove
   * ciò che inquadra viene più grande; altrimenti un taglio solo, dove resta più area (avoidRect).
   */
  function freeRects({ variants = false } = {}) {
    const { W, H } = stageView();
    const rects = card?.freeRects(host.container) ?? [{ x0: 0, y0: 0, x1: W, y1: H }];
    const pad = layer.querySelector(":scope > .nav-pad");
    if (!pad || pad.hidden) return rects;
    const box = host.container.getBoundingClientRect();
    const r = pad.getBoundingClientRect();
    if (!r.width || !r.height) return rects;
    const block = { x0: r.left - box.left, y0: r.top - box.top, x1: r.right - box.left, y1: r.bottom - box.top };
    return variants ? rects.flatMap((free) => avoidRectVariants(free, block)) : rects.map((free) => avoidRect(free, block));
  }

  /**
   * Ingombro dei punti visti dalla direzione `dir`: la base della camera (destra, su, profondità), i
   * semiassi e il centro (nel piano della camera, a metà profondità).
   */
  function extentOf(points, dir) {
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
    const center = new THREE.Vector3().addScaledVector(right, (lo[0] + hi[0]) / 2).addScaledVector(up, (lo[1] + hi[1]) / 2).addScaledVector(d, (lo[2] + hi[2]) / 2);
    return { d, right, up, center, halfR: (hi[0] - lo[0]) / 2, halfU: (hi[1] - lo[1]) / 2, halfD: (hi[2] - lo[2]) / 2 };
  }

  /** Distanza dal centro che fa entrare l'ingombro `e` nella parte libera `r` (la faccia più vicina a filo). */
  function distanceFor(e, r, margin) {
    const { W, H, tanV, tanH } = stageView();
    const fw = Math.max(80, r.x1 - r.x0);
    const fh = Math.max(80, r.y1 - r.y0);
    return Math.max(2, Math.max(e.halfU / (tanV * (fh / H)), e.halfR / (tanH * (fw / W))) * margin + e.halfD);
  }

  /**
   * Piano delle reti lunghe (framing.js), ricalcolato solo se cambiano stage o scheda: per ogni parte
   * libera la larghezza (unità) di una finestra in cui la rete è alta `readableHeight` pixel; vince la
   * parte libera dove la finestra è più larga. Se la rete ci sta quasi tutta, una parte sola (long false).
   */
  function walkPlan() {
    const { W, H, tanV, tanH } = stageView();
    // si rifà quando cambiano lo stage o lo stato della scheda (ridotta o no), non per ogni riga di stato in
    // più: "Centra" deve tornare proprio alla vista iniziale
    const key = `${W}x${H}|${card?.collapsed ?? ""}`;
    if (state.walk?.key === key) return state.walk;
    const rects = freeRects();
    const points = scene.framePoints();
    const e = extentOf(points, VIEW_DIRECTIONS.side);
    let xMin = Infinity;
    let xMax = -Infinity;
    for (const p of points) {
      xMin = Math.min(xMin, p.x);
      xMax = Math.max(xMax, p.x);
    }
    const { layout, net } = model;
    const items = net.items.map((it) => (it.kind === "repeat" ? { x0: layout.repeats[it.index].blockX0, x1: layout.repeats[it.index].x1, label: "" } : { x0: layout.sections[it.index].box.min[0], x1: layout.sections[it.index].box.max[0], label: it.label }));
    const groups = layout.groups.map((g) => ({ x0: g.x0, x1: g.x1, label: g.label }));
    let best = null;
    for (const rect of rects) {
      const fw = Math.max(80, rect.x1 - rect.x0);
      const fh = Math.max(80, rect.y1 - rect.y0);
      // a che distanza la rete (la sua faccia più vicina) è alta quanto basta; quanto si vede in larghezza lì
      const near = (e.halfU * H) / (tanV * readableHeight(fh));
      const width = (2 * near * tanH * (fw / W)) / PART_MARGIN;
      const parts = walkParts({ items, span: [xMin, xMax], width, groups });
      if (!best || width > best.width) best = { width, parts, rect };
    }
    state.walk = { key, long: best.parts.length > 1, parts: best.parts, rect: best.rect };
    return state.walk;
  }

  /** I punti da inquadrare per la parte p: la sua fetta di rete, alta e profonda come tutta la rete. */
  function partPoints(p) {
    const { min, max } = model.layout.bounds;
    const all = scene.framePoints();
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const v of all) {
      y0 = Math.min(y0, v.y);
      y1 = Math.max(y1, v.y);
    }
    const points = [];
    for (let i = 0; i < 8; i++) points.push(new THREE.Vector3(i & 1 ? p.x1 : p.x0, i & 2 ? y1 : y0, i & 4 ? max[2] : min[2]));
    return points;
  }

  /** "Centra" e la prima inquadratura: tutta la rete, o la prima parte di una rete lunga (l'ingresso). */
  function home({ plan = null, ...options } = {}) {
    if (!scene) return Promise.resolve();
    const p = plan ?? walkPlan();
    if (!p.long) return frameNetwork("side", { whole: true, ...options });
    // la prima inquadratura ha già l'invito a scorrere; "Centra" dopo una passeggiata dice dove si torna
    if (!options.immediate) card?.setStatus(partStatus(p, 0));
    return framePart(0, VIEW_DIRECTIONS.side, options);
  }

  /** "Tutta la rete": l'insieme di lato; per una rete lunga la riga di stato dice come tornare alle parti. */
  function showWhole() {
    const plan = scene ? walkPlan() : null;
    if (plan?.long) card?.setStatus(`Tutta la rete: ${plan.parts.length} parti. ▶ riparte dall'ingresso.`);
    return frameNetwork("side", { whole: true });
  }

  const partStatus = (plan, i) => {
    const title = plan.parts[i].title;
    return `Parte ${i + 1} di ${plan.parts.length}${title ? `: ${title}` : ""}${title.endsWith("…") ? "" : "."}`;
  };

  function framePart(index, dir, options = {}) {
    const plan = walkPlan();
    const i = Math.max(0, Math.min(plan.parts.length - 1, index));
    director.noteUserIntent();
    return framePoints(partPoints(plan.parts[i]), dir, { margin: PART_MARGIN, rect: plan.rect, ...options });
  }

  /**
   * ◀ ▶ della scheda: la parte prima o dopo quella che si sta guardando, con lo stesso angolo della
   * camera; dalla vista d'insieme si riparte dalla prima (l'ingresso).
   */
  function walk(step) {
    if (!scene) return Promise.resolve();
    const plan = walkPlan();
    if (!plan.long) return Promise.resolve();
    const from = partAt(plan);
    const i = from < 0 ? 0 : Math.max(0, Math.min(plan.parts.length - 1, from + Math.sign(step)));
    card?.setStatus(partStatus(plan, i));
    return framePart(i, from < 0 ? VIEW_DIRECTIONS.side : currentDirection());
  }

  /**
   * La parte che si sta guardando: quella con il centro più vicino al punto al centro della parte libera
   * del piano (dove framePoints mette la parte inquadrata), alla profondità del punto attorno a cui gira
   * la camera. Non quel punto: con la scheda a sinistra sta più a sinistra del centro della parte, e con
   * due parti vicine (l'ultima si sovrappone alla penultima) si leggerebbe la parte sbagliata.
   * -1 se nella parte libera entra più di una parte e un terzo: si guarda la rete d'insieme.
   */
  function partAt(plan) {
    const { W, tanH } = stageView();
    const r = plan.rect;
    const depth = lookedAt(r, look.point);
    const seen = 2 * depth * tanH * ((r.x1 - r.x0) / W);
    const part = plan.parts[0].x1 - plan.parts[0].x0;
    return seen > WHOLE_VIEW * part ? -1 : partIndexAt(plan.parts, look.point.x);
  }

  /**
   * Il punto guardato: quello al centro del rettangolo `rect` dello stage (la parte libera dove le
   * inquadrature mettono ciò che si guarda), alla profondità del punto attorno a cui gira la camera.
   * Lo scrive in `out` e restituisce quella profondità (0 se la camera guarda altrove: allora `out` è il
   * punto attorno a cui gira la camera).
   */
  function lookedAt(rect, out) {
    const { W, H } = stageView();
    camera.updateMatrixWorld();
    const forward = camera.getWorldDirection(look.forward);
    const depth = look.toTarget.subVectors(host.controls.target, camera.position).dot(forward);
    const ray = look.ray.set((rect.x0 + rect.x1) / W - 1, 1 - (rect.y0 + rect.y1) / H, 0.5).unproject(camera).sub(camera.position);
    const along = ray.dot(forward);
    if (!(depth > 0) || !(along > 1e-9)) {
      out.copy(host.controls.target);
      return 0;
    }
    out.copy(camera.position).addScaledVector(ray, depth / along);
    return depth;
  }
  const look = { forward: new THREE.Vector3(), toTarget: new THREE.Vector3(), ray: new THREE.Vector3(), point: new THREE.Vector3(), focus: new THREE.Vector3() };

  /** La parte libera più grande dello stage (per il punto guardato a camera mossa dallo studente). */
  function largestFreeRect() {
    let best = null;
    for (const r of freeRects()) if (!best || (r.x1 - r.x0) * (r.y1 - r.y0) > (best.x1 - best.x0) * (best.y1 - best.y0)) best = r;
    return best;
  }

  /** La riga delle parti nella scheda segue la camera (solo quando cambia la parte, o il piano). */
  function updateWalk() {
    if (!scene || !card) return;
    const plan = walkPlan();
    const i = plan.long ? partAt(plan) : -1;
    const shown = plan.long ? `${plan.key}|${i}` : "";
    if (shown === state.walkShown) return;
    state.walkShown = shown;
    card.setWalk(plan.long ? { index: i, count: plan.parts.length, title: i < 0 ? "" : plan.parts[i].title } : null);
  }

  /**
   * La sezione scelta (o, senza scelta, la parte che si guarda di una rete lunga, o tutta la rete) di
   * lato o di tre quarti. whole: sempre tutta la rete di lato ("Tutta la rete").
   */
  function frameNetwork(how, { whole = false, ...options } = {}) {
    if (!scene) return Promise.resolve();
    director.noteUserIntent();
    const part = whole ? null : state.selected;
    const dir = how === "thick" && part ? VIEW_DIRECTIONS.thickSection : VIEW_DIRECTIONS[how] ?? VIEW_DIRECTIONS.side;
    if (!part && !whole) {
      const plan = walkPlan();
      const i = plan.long ? partAt(plan) : -1;
      if (i >= 0) return framePart(i, dir, options);
    }
    const points = part ? scene.framePoints(selectionIds(part)) : scene.framePoints();
    return framePoints(points, dir, { margin: part ? 1.3 : WHOLE_MARGIN, ...options });
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
      // il fuoco delle lastre fantasma è il punto al centro della parte libera, non quello attorno a cui
      // gira la camera: con la scheda a sinistra sta più a sinistra, a volte proprio dentro le lastre
      lookedAt(largestFreeRect(), look.focus);
      scene.update(camera, host.container.clientWidth, host.container.clientHeight, host.renderer.getPixelRatio(), coveredRects(), look.focus);
      updateWalk();
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

  /**
   * Rettangoli dello stage coperti da comandi DOM sopra il canvas: la scheda, il pad e i chip della
   * navigazione del livello, ☰ e il selettore delle rappresentazioni. Le etichette 3D li evitano (un
   * titolo mezzo nascosto sotto la scheda non si legge e toglie il posto a un altro). Letti solo quando
   * la camera o lo stage cambiano, non a ogni frame.
   */
  function coveredRects() {
    const box = host.container.getBoundingClientRect();
    const rects = [];
    const add = (node) => {
      if (!node || node.hidden) return;
      const r = node.getBoundingClientRect();
      if (r.width >= 1 && r.height >= 1) rects.push([r.left - box.left, r.top - box.top, r.right - box.left, r.bottom - box.top]);
    };
    add(card?.root);
    for (const node of layer.querySelectorAll(":scope > .nav-pad, :scope > .nav-hud > :not([hidden])")) add(node);
    for (const node of layer.parentElement?.querySelectorAll(":scope > .rep-switch, :scope > .sidebar-toggle") ?? []) add(node);
    return rects;
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
      const attention = net.links.filter((l) => l.pattern === "attention" && l.to === s);
      card.showSection(s, {
        incoming: linkSummary(net.links.filter((l) => l.to === s && l.from !== s), (l) => l.from),
        outgoing: linkSummary(net.links.filter((l) => l.from === s && l.to !== s), (l) => l.to),
        simulated: model.values.simulated.has(s.id),
        // da dove vengono i pesi di attention che arrivano qui (il più "debole" se sono di più tipi)
        attention: attention.length ? ["simulated", "illustrative", "real"].find((k) => attention.some((l) => attentionSource(l) === k)) : null,
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

  /** Un gruppo per indice o per etichetta (le parentesi non hanno un id). */
  function groupIndex(g) {
    const groups = model.net.groups;
    if (Number.isInteger(g)) return g >= 0 && g < groups.length ? g : -1;
    return groups.findIndex((x) => x.label === g);
  }

  /**
   * focus(target): "id" | ["id", …] | { section: "id", slice: n } | { group: "Encoder" | indice }. Vola
   * sulla sezione (o sull'unione, o sulle sezioni del gruppo) mantenendo l'angolo della camera e ne
   * mostra la scheda. false se non conosce l'id.
   */
  async function focus(target) {
    await view.ready;
    if (!scene) return false;
    const o = typeof target === "string" || Array.isArray(target) ? { section: target } : target ?? {};
    if (o.slice !== undefined) setSlice(o.slice);
    if (o.group !== undefined) {
      const g = groupIndex(o.group);
      if (g < 0) return o.slice !== undefined;
      select("group", g);
      director.noteUserIntent();
      await framePoints(scene.framePoints(selectionIds(state.selected)), currentDirection(), { margin: 1.3 });
      return true;
    }
    const ids = (Array.isArray(o.section) ? o.section : [o.section]).filter((id) => model.net.byId.has(id));
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
    if (director.flying) return hideTip();
    const { x, y } = state.pointer;
    const rect = canvas.getBoundingClientRect();
    const label = scene.labels?.hit(x - rect.left, y - rect.top) ?? null;
    canvas.style.cursor = label ? "pointer" : "";
    const t0 = performance.now();
    // sotto un titolo di sezione si cerca lo stesso il neurone: se è di un'altra sezione, il titolo lo copriva
    const picked = label && label.ref?.kind !== "section" ? -1 : scene.pick(x, y, camera, rect);
    stats.pickMs = ema(stats.pickMs, performance.now() - t0);
    const { net, layout } = scene.model;
    const u = hoverUnit(label, picked, (unit) => net.sections[layout.sectionOf[unit]].id);
    if (u < 0) return hideTip();
    // il clic resta del titolo disegnato sopra: l'etichetta lo dice, per non far credere che scelga questa sezione
    showTip(scene.describe(u), x, y, label ? `Clic qui: sceglie «${label.text}», il titolo disegnato sopra` : null);
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

  /**
   * Etichetta al passaggio del mouse: sezione, indice, posizione in profondità, valore. x, y: coordinate client;
   * `note`: una riga in fondo (es. che cosa sceglie il clic in quel punto), testo semplice.
   */
  function showTip(info, clientX, clientY, note = null) {
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
      // con un nome (una voce del codebook, un carattere) il nome prima e il contatore da 1 dopo:
      // "«voce 2» · 3 di 64", non "neurone 3 «voce 2» di 64" (3 accanto a un nome che conta da 0)
      const name = s.neuronNames?.[info.neuron];
      what = name ? `${slab}«${name}» · ${info.neuron + 1} di ${s.neurons}${head}` : `${slab}neurone ${info.neuron + 1} di ${s.neurons}${head}`;
    }
    if (info.depthName) lines.unshift(`${s.depth.label}: ${info.depth + 1} di ${s.depth.count} «${info.depthName}»`);
    const value = info.mode === MODE.off ? "—" : fmt(info.value, Math.abs(info.value) < 0.01 && info.value !== 0 ? 4 : 3);
    tip.replaceChildren(
      el("strong", {}, s.label),
      el("span", {}, what),
      ...lines.map((l) => el("span", {}, l)),
      el("span", { className: "nn-tip-value" }, `valore ${value}${s.valueLabel ? ` ${s.valueLabel}` : ""}${info.simulated ? " · simulato" : ""}`),
      note ? el("span", {}, note) : "",
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

