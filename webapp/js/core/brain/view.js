/**
 * Vista "Cervello" del nucleo: il cervello 3D del tab dentro il livello di una rappresentazione
 * alternativa (core/representations.js la importa dinamicamente).
 *
 *   const view = createBrainView({ host, director, nav, layer, tabId, spec, app, reps, memory });
 *   await view.ready;                       // scena costruita e inquadrata
 *   view.focus(["hippocampus"]);            // vola sulle regioni e le fa pulsare
 *   view.focus("circuit");                  // scala Neuroni ("circuit:<gruppo>" sceglie anche il gruppo)
 *   view.onShow(); view.onHide(); view.dispose();
 *
 * Due scale nella stessa scena:
 *  - Regioni: il cervello intero, le regioni del tab evidenziate, i percorsi, le etichette, la
 *    trasparenza della corteccia e i piani di taglio;
 *  - Neuroni: il microcircuito del tab, da sinistra a destra come la rete AI; il cervello si rimpicciolisce
 *    di lato e ogni colonna di neuroni è legata da una linea alla sua regione ("ingrandimento"). Nel
 *    passaggio il circuito cresce dalla sua regione mentre la camera vola: si capisce da dove viene.
 *
 * host / director / nav sono quelli del livello (SceneHost, CameraDirector, Navigation del nucleo):
 * qui si usano soltanto. Costi: a riposo, per frame, solo il confronto della matrice della camera
 * (nessuna allocazione); con la camera in movimento le etichette (~0,1-0,3 ms); durante gli impulsi
 * qualche uniform. La geometria del cervello si calcola una volta per pagina (anatomy.js).
 */
import * as THREE from "three";
import { prefersReducedMotion } from "../dom.js";
import { atlasEntry, atlasKind } from "./atlas.js";
import { BrainModel, CUTS, loadAnatomyData } from "./anatomy.js";
import { BrainCard } from "./card.js";
import { CircuitScene } from "./circuit.js";
import { LabelOverlay } from "./labels.js";
import { PathwayLayer } from "./pathways.js";
import { RegionLayer } from "./regions.js";
import { normalizeBrainSpec } from "./spec.js";

/** Direzioni della camera (verso la camera, coordinate della scena: +z = lato sinistro della testa). */
export const VIEW_DIRECTIONS = Object.freeze({
  home: Object.freeze([-0.3, 0.3, 1]), // emisfero sinistro di tre quarti, un po' da davanti e dall'alto
  neurons: Object.freeze([-0.3, 0.3, 1]),
});
/**
 * Opacità iniziale della corteccia: opaca, il cervello come lo si riconosce (giri, solchi, lobi). Le
 * etichette delle strutture interne sono attenuate e tratteggiate; quando lo studente vola su una di
 * loro (o su un percorso) la corteccia diventa trasparente da sola e il cursore lo mostra.
 */
export const DEFAULT_OPACITY = 1;
const DEEP_OPACITY = 0.32; // quando si vola su una struttura profonda la corteccia si fa più trasparente
const PULSE_SECONDS = 4.5;
const FOCUS_MIN_SIZE = 12; // cm di finestra inquadrata attorno alle regioni: si vedono i lobi attorno (il cervello è lungo ~17)
const TRANSITION_SECONDS = 1.25;
const MINI_SCALE = 0.34; // cervello in miniatura nella scala Neuroni
const CLICK_PX = 5;
const CLICK_MS = 650;

export function createBrainView(ctx) {
  const { host, director, nav, layer, memory = {} } = ctx;
  const camera = host.camera;
  const canvas = host.renderer.domElement;
  const spec = normalizeBrainSpec(ctx.spec);
  const quality = ctx.quality ?? detectQuality(host.renderer);
  const reduced = prefersReducedMotion();
  const state = { scale: "regions", opacity: DEFAULT_OPACITY, cut: null, pathways: true, labels: true, lobes: false, ...(memory.state ?? {}) };
  if (!spec.circuit) state.scale = "regions";
  const stats = { quality, buildMs: 0, anatomy: null, vertices: 0, circuitMs: 0, circuit: null, labelsMs: 0, labelUpdates: 0, tickUs: 0 };
  const disposers = [];
  const lastCamera = new Float64Array(17).fill(NaN);
  let brain = null;
  let regions = null;
  let pathways = null;
  let circuit = null;
  let labels = null;
  let card = null;
  let brainGroup = null;
  let disposed = false;
  let playing = false;
  let clock = 0;
  let lastTick = 0;
  let pulseUntil = 0;
  let transition = null;
  let selected = null; // { kind: "region" | "group", id }
  let labelsDirty = true;
  let down = null;
  let hoverRaf = 0;
  const tip = document.createElement("div");
  tip.className = "br-tip";
  tip.hidden = true;
  tip.setAttribute("role", "tooltip");
  layer.append(tip);

  const view = {
    ready: null,
    spec,
    stats,
    get scale() {
      return state.scale;
    },
    get brain() {
      return brain;
    },
    get circuit() {
      return circuit;
    },
    get state() {
      return { ...state, playing };
    },
    focus,
    onShow,
    onHide,
    dispose,
    /** Comandi della vista, anche per gli script di verifica. */
    setScale: (id) => setScale(id),
    setOpacity: (v) => setOpacity(v),
    setCut: (axis, mm, flip = false) => setCut(axis ? { axis, mm: mm ?? CUTS[axis]?.value, flip } : null, { fly: true }),
    toggle: (name, on) => toggle(name, on),
    play: (on) => setPlaying(on ?? !playing),
    select: (id) => selectAtlas(id),
    selectGroup: (id) => selectGroup(id),
    home: () => home(),
    /** Microsecondi per frame del ciclo a riposo (n giri sincroni: performance.now è troppo grossolano per uno solo). */
    measureIdle(n = 20000) {
      if (!brain) return NaN;
      tick(0);
      const t0 = performance.now();
      for (let i = 0; i < n; i++) tick(0);
      return ((performance.now() - t0) * 1000) / n;
    },
  };

  view.ready = (async () => {
    // lascia disegnare la scheda "Preparo la scena…" prima del lavoro pesante
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    if (disposed) return;
    const t0 = performance.now();
    const data = await loadAnatomyData(quality);
    if (disposed) return;
    stats.anatomy = data.timings;
    stats.vertices = data.vertices;
    brain = new BrainModel(data);
    brainGroup = new THREE.Group();
    brainGroup.name = "brain-scale";
    brainGroup.add(brain.root);
    regions = new RegionLayer(brain, spec);
    brainGroup.add(regions.root);
    pathways = new PathwayLayer(spec.pathways, { clipPlanes: brain.clipPlanes, quality });
    brainGroup.add(pathways.root);
    host.scene.add(brainGroup);
    host.renderer.localClippingEnabled = true;
    labels = new LabelOverlay(layer, { onPick: (item) => onLabelPick(item) });
    card = new BrainCard(layer, spec, handlers(), state);
    brain.setCortexOpacity(state.opacity);
    brain.setLobeColors(state.lobes);
    if (state.cut) brain.setCut(state.cut);
    pathways.setVisible(state.pathways);
    labels.setVisible(state.labels);
    card.renderLegend("regions", { cut: state.cut });
    nav.setBounds(() => boundsBox());
    nav.setHome(() => home());
    host.controls.minDistance = 1.2;
    stats.buildMs = performance.now() - t0;
    disposers.push(host.onFrame(tick));
    // la scheda che si apre o si riduce (e il pad) cambia lo spazio libero per le etichette anche a camera ferma
    const relabel = new ResizeObserver(() => (labelsDirty = true));
    relabel.observe(card.root);
    const pad = layer.querySelector(":scope > .nav-pad");
    if (pad) relabel.observe(pad);
    disposers.push(() => relabel.disconnect());
    bindPointer();
    if (state.scale === "neurons") setScale("neurons", { immediate: true, force: true });
    else frameHome({ immediate: true });
    rebuildLabels();
  })();

  // ---------------------------------------------------------------- comandi della scheda
  function handlers() {
    return {
      scale: (id) => setScale(id),
      play: () => setPlaying(!playing),
      opacity: (v) => setOpacity(v),
      cut: (axis) => setCut(axis ? { axis, mm: CUTS[axis].value, flip: false } : null, { fly: true }),
      cutPosition: (mm) => state.cut && setCut({ ...state.cut, mm }),
      flip: () => state.cut && setCut({ ...state.cut, flip: !state.cut.flip }, { fly: true }),
      toggle: (name, on) => toggle(name, on),
      focusRegion: (id) => focus([id]),
      selectGroup: (id, o = {}) => (o.scale ? focus(`circuit:${id}`) : selectGroup(id, { frame: true })),
      frameGroup: (id) => frameGroup(id),
      clear: () => clearSelection(),
    };
  }

  function remember() {
    memory.state = { scale: state.scale, opacity: state.opacity, cut: state.cut, pathways: state.pathways, labels: state.labels, lobes: state.lobes };
  }

  function setOpacity(value) {
    state.opacity = Math.max(0, Math.min(1, value));
    brain?.setCortexOpacity(state.opacity);
    card?.setOpacity(state.opacity);
    labelsDirty = true;
    remember();
  }

  function setCut(cut, { fly = false } = {}) {
    state.cut = cut;
    brain?.setCut(cut);
    card?.setCut(cut);
    card?.renderLegend(state.scale, { focus: regions?.focus ?? [], cut });
    labelsDirty = true;
    remember();
    // un taglio si guarda meglio con la corteccia opaca: la sezione è una fetta piena, come in un atlante
    if (cut && fly && state.opacity < 0.9) setOpacity(1);
    if (cut && fly && state.scale === "regions") {
      // la camera guarda la sezione di fronte, dalla parte tolta
      const dir = brain.cutViewDirection();
      if (cut.axis === "axial") dir.add(new THREE.Vector3(-0.05, 0, 0.25));
      else dir.add(new THREE.Vector3(0, 0.25, 0));
      frameBox(brain.box(), dir.normalize().toArray(), { margin: 1.02 });
    }
  }

  function toggle(name, on) {
    if (!(name in state)) return;
    state[name] = Boolean(on);
    card?.setToggle(name, state[name]);
    if (name === "pathways") pathways?.setVisible(state.pathways);
    if (name === "labels") labels?.setVisible(state.labels);
    if (name === "lobes") brain?.setLobeColors(state.lobes);
    labelsDirty = true;
    remember();
  }

  function setPlaying(on) {
    playing = Boolean(on);
    pathways?.setPlaying(playing);
    circuit?.setPlaying(playing);
    card?.setPlaying(playing);
  }

  // ---------------------------------------------------------------- scale
  function ensureCircuit() {
    if (circuit || !spec.circuit) return circuit;
    const t0 = performance.now();
    circuit = new CircuitScene(spec.circuit, { quality });
    circuit.root.visible = false;
    host.scene.add(circuit.root);
    host.scene.add(circuit.links);
    circuit.setPlaying(playing);
    stats.circuitMs = performance.now() - t0;
    stats.circuit = circuit.field.counts;
    return circuit;
  }

  /**
   * Dove va il cervello in miniatura nella scala Neuroni: sopra l'inizio del circuito (lì i primi gruppi,
   * di solito l'ingresso, sono bassi e c'è posto). Accanto al circuito allungherebbe la scena e
   * rimpicciolirebbe i neuroni.
   */
  function miniPlacement() {
    const box = circuit.localBox();
    const width = 17.6 * MINI_SCALE;
    const height = 14 * MINI_SCALE;
    return { scale: MINI_SCALE, position: new THREE.Vector3(box.min.x + width / 2 + 0.2, box.max.y + height / 2 + 0.5, -0.6) };
  }

  /** La regione "di casa" del circuito: quella del gruppo centrale (da lì il circuito cresce nella transizione). */
  function homeRegionPoint() {
    const groups = spec.circuit?.groups ?? [];
    const g = groups[Math.floor(groups.length / 2)] ?? groups[0];
    const id = g?.marker ?? g?.region;
    const anchor = id ? regions.anchorsOf(id)[0] : null;
    brainGroup.updateMatrixWorld();
    return anchor ? new THREE.Vector3().fromArray(anchor.point).applyMatrix4(brainGroup.matrixWorld) : new THREE.Vector3();
  }

  function setScale(scale, { immediate = false, force = false, after = null } = {}) {
    if (!brain || (scale === "neurons" && !spec.circuit)) return Promise.resolve();
    if (scale === state.scale && !force) return Promise.resolve(after?.());
    if (scale === "neurons") ensureCircuit();
    state.scale = scale;
    remember();
    card.setScale(scale);
    card.clearDetail();
    selected = null;
    regions.apply({ focus: [], selected: null });
    hideTip();
    const toNeurons = scale === "neurons";
    if (!toNeurons) {
      brain.setMiniature(false);
      pathways.setVisible(state.pathways);
    }
    const mini = circuit ? miniPlacement() : null;
    const origin = toNeurons ? homeRegionPoint() : null;
    const from = {
      brainScale: brainGroup.scale.x,
      brainPos: brainGroup.position.clone(),
      circuitScale: circuit?.root.visible ? circuit.root.scale.x : 0.04,
      circuitPos: circuit?.root.visible ? circuit.root.position.clone() : origin ?? new THREE.Vector3(),
    };
    const to = toNeurons
      ? { brainScale: mini.scale, brainPos: mini.position, circuitScale: 1, circuitPos: new THREE.Vector3() }
      : { brainScale: 1, brainPos: new THREE.Vector3(), circuitScale: 0.04, circuitPos: from.circuitPos.clone() };
    if (circuit) circuit.root.visible = true;
    transition = { from, to, t: 0, duration: immediate || reduced ? 0 : TRANSITION_SECONDS, toNeurons };
    if (immediate || reduced) stepTransition(1);
    host.controls.minDistance = toNeurons ? 0.6 : 1.2;
    card.renderLegend(scale, { focus: [], cut: state.cut });
    rebuildLabels();
    const flight = frameHome({ immediate, scale, final: true });
    return Promise.resolve(flight).then(() => after?.());
  }

  /** Avanza la transizione tra le scale (k = 0…1 già con l'andamento morbido). */
  function stepTransition(k) {
    const { from, to } = transition;
    const s = from.brainScale + (to.brainScale - from.brainScale) * k;
    brainGroup.scale.setScalar(s);
    brainGroup.position.lerpVectors(from.brainPos, to.brainPos, k);
    if (circuit) {
      circuit.root.scale.setScalar(Math.max(0.001, from.circuitScale + (to.circuitScale - from.circuitScale) * k));
      circuit.root.position.lerpVectors(from.circuitPos, to.circuitPos, k);
    }
    if (k >= 1) {
      if (circuit && !transition.toNeurons) circuit.root.visible = false;
      if (quality === "low") {
        brain.setMiniature(transition.toNeurons, { keep: new Set((spec.circuit?.groups ?? []).map((g) => g.region)) });
        pathways.setVisible(!transition.toNeurons && state.pathways);
      }
      transition = null;
      labelsDirty = true;
      rebuildLabels();
    }
    updateLinks();
  }

  function updateLinks() {
    if (!circuit) return;
    if (state.scale !== "neurons" || !circuit.root.visible) {
      circuit.setLinks(null);
      return;
    }
    brainGroup.updateMatrixWorld();
    const points = new Map();
    const v = new THREE.Vector3();
    for (const g of spec.circuit.groups) {
      for (const id of [g.region, g.marker]) {
        if (!id || points.has(id)) continue;
        const anchor = regions.anchorsOf(id)[0];
        if (anchor) points.set(id, v.fromArray(anchor.point).applyMatrix4(brainGroup.matrixWorld).toArray());
      }
    }
    circuit.setLinks(points);
  }

  // ---------------------------------------------------------------- focus e scelta
  /**
   * Vola su ciò che si chiede e lo fa pulsare: id di regioni, marcatori o percorsi (stringa o elenco),
   * oppure "circuit" / "circuit:<gruppo>" per la scala Neuroni.
   */
  function focus(target) {
    if (disposed) return;
    if (!brain) {
      // scena non ancora pronta: si riprova a scena costruita (mai in cerchio se la costruzione non
      // c'è stata: vista chiusa prima, o errore, che representations.js ha già segnalato)
      view.ready.then(() => brain && !disposed && focus(target), () => {});
      return;
    }
    const ids = (Array.isArray(target) ? target : [target]).filter((t) => typeof t === "string");
    const circuitTarget = ids.find((t) => t === "circuit" || t.startsWith("circuit:"));
    if (circuitTarget) {
      const group = circuitTarget.split(":")[1] ?? null;
      setScale("neurons", { after: () => group && selectGroup(group, { frame: true }) });
      return;
    }
    const known = ids.filter((id) => atlasKind(id));
    if (!known.length) {
      if (state.scale !== "regions") setScale("regions");
      return;
    }
    const run = () => focusAtlas(known);
    if (state.scale !== "regions") setScale("regions", { after: run });
    else run();
  }

  function focusAtlas(ids) {
    const regionIds = ids.filter((id) => atlasKind(id) !== "pathway");
    const pathwayIds = ids.filter((id) => atlasKind(id) === "pathway");
    regions.apply({ focus: regionIds, selected: null });
    pathways.emphasize(pathwayIds[0] ?? null);
    if (pathwayIds.length && !state.pathways) toggle("pathways", true);
    // le strutture profonde e i percorsi stanno dentro: la corteccia si fa trasparente (e lo si vede nel cursore)
    const inside = regionIds.some((id) => atlasEntry(id).kind !== "cortical") || pathwayIds.length > 0;
    if (inside && state.opacity > DEEP_OPACITY) setOpacity(DEEP_OPACITY);
    const box = regions.box(regionIds);
    for (const id of pathwayIds) box.union(pathways.box(id));
    pulseUntil = clock + PULSE_SECONDS;
    card.renderLegend("regions", { focus: regionIds, cut: state.cut });
    selected = { kind: "region", id: ids[0] };
    card.showRegion({ id: ids[0], kind: atlasKind(ids[0]) });
    rebuildLabels();
    // mai troppo vicino: la regione si deve vedere dentro il cervello, con i lobi attorno per orientarsi
    if (!box.isEmpty()) frameBox(box, facingDirection(regionIds), { margin: 1.15, minSize: FOCUS_MIN_SIZE });
  }

  /** Da che parte guardare delle regioni: dalla parte in cui si affacciano (normali), altrimenti di tre quarti. */
  function facingDirection(ids) {
    const dir = new THREE.Vector3();
    for (const id of ids) {
      const anchors = regions.anchorsOf(id);
      const a = anchors.find((x) => x.side !== "right") ?? anchors[0];
      if (a?.normal) dir.add(new THREE.Vector3().fromArray(a.normal));
    }
    if (dir.lengthSq() < 0.01) return VIEW_DIRECTIONS.home;
    dir.normalize().add(new THREE.Vector3(...VIEW_DIRECTIONS.home).normalize().multiplyScalar(0.7)).add(new THREE.Vector3(0, 0.25, 0));
    return dir.normalize().toArray();
  }

  function selectAtlas(id) {
    if (!brain || !atlasKind(id)) return;
    onPicked({ id, kind: atlasKind(id) });
  }

  /** Scelta con un clic (sulla scena o su un'etichetta): scheda e alone, senza muovere la camera. */
  function onPicked(pick) {
    selected = { kind: "region", id: pick.id };
    const region = atlasKind(pick.id) === "region" ? pick.id : null;
    regions.apply({ selected: region });
    pathways.emphasize(atlasKind(pick.id) === "pathway" ? pick.id : null);
    card.showRegion(pick);
    labels.mark(pick.id);
    rebuildLabels();
  }

  function selectGroup(id, { frame = false } = {}) {
    if (!spec.circuit) return;
    const group = spec.circuit.groups.find((g) => g.id === id);
    if (!group) return;
    if (state.scale !== "neurons") {
      setScale("neurons", { after: () => selectGroup(id, { frame }) });
      return;
    }
    selected = { kind: "group", id };
    const name = (gid) => spec.circuit.groups.find((g) => g.id === gid)?.label ?? gid;
    const incoming = spec.circuit.edges.filter((e) => e.to === id).map((e) => ({ ...e, other: e.from === id ? "sé stesso" : name(e.from) }));
    const outgoing = spec.circuit.edges.filter((e) => e.from === id && e.to !== id).map((e) => ({ ...e, other: name(e.to) }));
    card.showGroup(group, { incoming, outgoing });
    rebuildLabels();
    if (group.region) regions.apply({ focus: [group.marker ?? group.region], selected: null });
    pulseUntil = clock + PULSE_SECONDS;
    if (frame) frameGroup(id);
  }

  function frameGroup(id) {
    if (!circuit) return;
    const box = circuit.columnBox(id);
    if (!box.isEmpty()) frameBox(box, [-0.42, 0.28, 0.86], { margin: 1.5, minSize: 3.2 });
  }

  function clearSelection() {
    selected = null;
    card.clearDetail();
    labels.mark(null);
    regions.apply({ focus: [], selected: null });
    pathways.emphasize(null);
    rebuildLabels();
  }

  function onLabelPick(item) {
    director.noteUserIntent();
    if (item.data?.kind === "group") selectGroup(item.data.id);
    else if (item.data?.id) onPicked({ id: item.data.id, kind: atlasKind(item.data.id) });
  }

  // ---------------------------------------------------------------- inquadrature
  function frameHome({ immediate = false, scale = state.scale, final = false } = {}) {
    if (scale === "neurons" && circuit) {
      const mini = miniPlacement();
      const box = circuit.localBox();
      box.expandByPoint(mini.position.clone().add(new THREE.Vector3(-17.6 * mini.scale * 0.5, -7 * mini.scale, 0)));
      box.expandByPoint(mini.position.clone().add(new THREE.Vector3(17.6 * mini.scale * 0.5, 7 * mini.scale, 0)));
      return frameBox(box, VIEW_DIRECTIONS.neurons, { immediate, margin: 0.98 });
    }
    // a transizione finita il cervello sta nell'origine a scala 1: si inquadra lì anche durante il volo
    const box = brainBoxAt(final ? { scale: 1, position: new THREE.Vector3() } : null);
    return frameBox(box, VIEW_DIRECTIONS.home, { immediate, margin: 1.02 });
  }

  function home() {
    director.noteUserIntent();
    if (state.cut && state.scale === "regions") {
      const dir = brain.cutViewDirection().add(new THREE.Vector3(0, 0.25, 0)).normalize();
      return frameBox(brain.box(), dir.toArray(), { margin: 1.02 });
    }
    return frameHome();
  }

  /** Box del cervello; con `at` la posizione che avrà a transizione finita. */
  function brainBoxAt(at) {
    const box = brain.box();
    if (!at) return box;
    // box() è già trasformata con la posizione attuale: si riporta a scala 1 nell'origine
    const s = brainGroup.scale.x || 1;
    box.min.sub(brainGroup.position).divideScalar(s).multiplyScalar(at.scale).add(at.position);
    box.max.sub(brainGroup.position).divideScalar(s).multiplyScalar(at.scale).add(at.position);
    return box;
  }

  /** Confini della navigazione: il cervello (o circuito + miniatura), allargati. */
  function boundsBox() {
    if (!brain) return null;
    const box = brain.box();
    if (state.scale === "neurons" && circuit) box.union(circuit.localBox());
    return box.expandByScalar(1.5);
  }

  /**
   * Inquadra un box dalla direzione `dir`, dentro la parte dello stage non coperta dalla scheda: si
   * proiettano gli 8 angoli sulla base della camera, si sceglie la distanza che li fa entrare e si
   * sposta il punto guardato perché il loro centro cada nel centro della parte libera.
   * `minSize` (cm) è la finestra minima inquadrata sul piano dello schermo, in larghezza e in altezza:
   * si misura dopo la proiezione perché una regione presente nei due emisferi ha il box lungo proprio
   * nella direzione della camera (vista di lato), e allargare il box 3D non allontanerebbe la camera.
   */
  function frameBox(box, dir, { duration = 1.1, immediate = false, margin = 1.08, minSize = 0 } = {}) {
    if (!box || box.isEmpty()) return Promise.resolve();
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
    const p = new THREE.Vector3();
    for (let i = 0; i < 8; i++) {
      p.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z);
      const a = [p.dot(right), p.dot(up), p.dot(d)];
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], a[k]);
        hi[k] = Math.max(hi[k], a[k]);
      }
    }
    for (let k = 0; k < 2 && minSize > 0; k++) {
      const grow = Math.max(0, minSize - (hi[k] - lo[k])) / 2;
      lo[k] -= grow;
      hi[k] += grow;
    }
    const center = new THREE.Vector3().addScaledVector(right, (lo[0] + hi[0]) / 2).addScaledVector(up, (lo[1] + hi[1]) / 2).addScaledVector(d, (lo[2] + hi[2]) / 2);
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const tanH = tanV * (W / H);
    const halfD = (hi[2] - lo[2]) / 2;
    const distance = Math.max(2, Math.max((hi[1] - lo[1]) / 2 / (tanV * (fh / H)), (hi[0] - lo[0]) / 2 / (tanH * (fw / W))) * margin + halfD);
    const cx = ((free.x0 + free.x1) / 2 / W) * 2 - 1;
    const cy = 1 - ((free.y0 + free.y1) / 2 / H) * 2;
    const target = center.clone().addScaledVector(right, -cx * distance * tanH).addScaledVector(up, -cy * distance * tanV);
    const position = target.clone().addScaledVector(d, distance);
    labelsDirty = true;
    if (immediate) {
      director.cancel();
      camera.position.copy(position);
      host.controls.target.copy(target);
      camera.lookAt(target);
      host.controls.update();
      return Promise.resolve();
    }
    director.noteUserIntent();
    return director.flyTo(position, target, duration);
  }

  // ---------------------------------------------------------------- etichette
  function rebuildLabels() {
    if (!labels || !regions) return;
    let items;
    if (state.scale === "neurons" && circuit) {
      items = circuit.labelItems();
      // la colonna scelta mostra anche il componente AI corrispondente
      for (const item of items) if (selected?.kind === "group" && item.data.id === selected.id) item.sub = item.ai;
      // il cervello in miniatura non ha etichette: le linee dicono da dove viene ogni colonna e la
      // scheda della colonna scelta dice il nome della regione (dieci nomi in più sarebbero rumore)
    } else {
      items = regions.labelItems();
      for (const item of pathways.items) {
        const left = item.anchors.find((a) => a.side !== "right") ?? item.anchors[0];
        if (left) items.push({ id: item.pathway.id, text: item.pathway.label, color: item.pathway.color, anchors: item.anchors.map((a) => a.point), priority: 2, data: { kind: "region", id: item.pathway.id }, pathway: true });
      }
    }
    labels.set(items);
    if (selected) labels.mark(selected.kind === "group" ? `col:${selected.id}` : selected.id);
    labelsDirty = true;
  }

  function updateLabels() {
    const t0 = performance.now();
    const W = host.container.clientWidth;
    const H = host.container.clientHeight;
    camera.updateMatrixWorld();
    const camPos = camera.position;
    const neurons = state.scale === "neurons";
    const center = neurons ? null : brainGroup.position.toArray();
    labels.update(camera, W, H, {
      center,
      obstacles: coveredRects(),
      // dell'emisfero si etichetta il lato più vicino alla camera
      pick: (item) => {
        if (item.anchors.length < 2) return item.anchors[0];
        let best = item.anchors[0];
        let bestD = Infinity;
        for (const a of item.anchors) {
          const d = (a[0] - camPos.x) ** 2 + (a[1] - camPos.y) ** 2 + (a[2] - camPos.z) ** 2;
          if (d < bestD) {
            bestD = d;
            best = a;
          }
        }
        return best;
      },
      state: (item, anchor) => {
        if (neurons) return "on";
        if (state.cut && brain.clipPlane.distanceToPoint(new THREE.Vector3(...anchor)) < -0.05) return "off";
        const index = item.anchors.indexOf(anchor);
        const normal = item.normals?.[index];
        if (normal) {
          // corticale: attenuata se la superficie dove sta guarda dall'altra parte (e la corteccia è opaca)
          const facing = normal[0] * (camPos.x - anchor[0]) + normal[1] * (camPos.y - anchor[1]) + normal[2] * (camPos.z - anchor[2]);
          return facing < 0 && state.opacity > 0.55 ? "dim" : "on";
        }
        return state.opacity > 0.75 && item.kind !== "peripheral" ? "dim" : "on";
      },
    });
    stats.labelsMs = performance.now() - t0;
    stats.labelUpdates++;
  }

  /**
   * Rettangoli dello stage coperti da comandi DOM sopra la scena (scheda, pad e chip della navigazione,
   * ☰ e selettore delle rappresentazioni): le etichette ci girano attorno. Letti solo quando le etichette
   * si ricalcolano (camera mossa, scheda ridimensionata), mai a riposo.
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

  // ---------------------------------------------------------------- ciclo per frame
  function tick(frameDt) {
    if (!brain) return;
    // tempo reale, come i voli del CameraDirector: il dt del SceneHost è limitato a 0,05 s e con frame
    // lenti (rendering software) transizione e impulsi andrebbero al rallentatore
    const now = performance.now();
    const dt = frameDt > 0 ? Math.min(0.25, Math.max(frameDt, lastTick ? (now - lastTick) / 1000 : frameDt)) : 0;
    lastTick = frameDt > 0 ? now : lastTick;
    if (transition) {
      transition.t = transition.duration > 0 ? Math.min(1, transition.t + Math.max(dt, 1 / 60) / transition.duration) : 1;
      const t = transition.t;
      stepTransition(t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
      labelsDirty = true;
    }
    const animating = playing || clock < pulseUntil;
    if (animating) {
      clock += dt;
      if (playing) {
        pathways.setTime(clock);
        circuit?.setTime(clock);
      }
      if (clock < pulseUntil) regions.setPulse(clock, Math.min(1, (pulseUntil - clock) / 0.8));
      else regions.setPulse(clock, 0);
    } else if (dt > 0) clock += dt;
    // la camera si è mossa? (confronto della matrice: niente allocazioni)
    const e = camera.matrixWorld.elements;
    let moved = labelsDirty;
    for (let i = 0; i < 16; i++) {
      if (e[i] !== lastCamera[i]) {
        moved = true;
        lastCamera[i] = e[i];
      }
    }
    if (camera.aspect !== lastCamera[16]) {
      moved = true;
      lastCamera[16] = camera.aspect;
    }
    if (moved && state.labels) {
      labelsDirty = false;
      updateLabels();
    }
  }

  // ---------------------------------------------------------------- puntatore
  function bindPointer() {
    const onDown = (e) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      down = { x: e.clientX, y: e.clientY, t: performance.now() };
    };
    const onUp = (e) => {
      const d = down;
      down = null;
      if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > CLICK_PX || performance.now() - d.t > CLICK_MS) return;
      director.noteUserIntent();
      onClick(e);
    };
    const onMove = (e) => {
      if (state.scale !== "neurons" || !circuit || e.pointerType === "touch") return;
      if (hoverRaf) return;
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = 0;
        hover(e);
      });
    };
    const onLeave = () => hideTip();
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    disposers.push(() => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      cancelAnimationFrame(hoverRaf);
    });
  }

  function localPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top, w: rect.width, h: rect.height };
  }

  function onClick(e) {
    if (transition) return;
    const { x, y, w, h } = localPoint(e);
    if (state.scale === "neurons" && circuit) {
      const hit = circuit.pickCell(x, y, camera, w, h, e.pointerType === "touch" ? 30 : 22);
      if (hit) {
        selectGroup(hit.column.id);
        if (e.pointerType === "touch") showTip(hit, x, y, true);
        return;
      }
      // un clic sul cervello in miniatura: si torna alle regioni, su quella regione
      host.raycaster.setFromCamera(new THREE.Vector2((x / w) * 2 - 1, -(y / h) * 2 + 1), camera);
      const pick = regions.pick(host.raycaster);
      if (pick && atlasKind(pick.id)) focus([pick.id]);
      return;
    }
    host.raycaster.setFromCamera(new THREE.Vector2((x / w) * 2 - 1, -(y / h) * 2 + 1), camera);
    const pick = regions.pick(host.raycaster);
    if (pick) onPicked(pick);
    else if (selected) clearSelection();
  }

  function hover(e) {
    const { x, y, w, h } = localPoint(e);
    const hit = circuit.pickCell(x, y, camera, w, h);
    if (hit) showTip(hit, x, y);
    else hideTip();
    canvas.style.cursor = hit ? "pointer" : "";
  }

  function showTip(hit, x, y, touch = false) {
    const g = spec.circuit.groups.find((gr) => gr.id === hit.column.id);
    tip.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = g.label;
    const small = document.createElement("span");
    small.textContent = g.ai ? `AI: ${g.ai}` : "";
    tip.append(strong, small);
    tip.hidden = false;
    const W = layer.clientWidth;
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    // mai sotto il dito: sul telefono sopra il punto toccato
    const left = Math.min(W - tw - 8, Math.max(8, x + 14));
    const top = touch ? Math.max(8, y - th - 28) : Math.min(layer.clientHeight - th - 8, y + 16);
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }

  function hideTip() {
    tip.hidden = true;
    if (canvas.style.cursor) canvas.style.cursor = "";
  }

  // ---------------------------------------------------------------- ciclo di vita
  function onShow() {
    labelsDirty = true;
  }

  function onHide() {
    setPlaying(false);
    hideTip();
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    for (const off of disposers.splice(0)) off();
    labels?.dispose();
    card?.dispose();
    tip.remove();
    circuit?.dispose();
    pathways?.dispose();
    regions?.dispose();
    brain?.dispose();
    brainGroup?.removeFromParent();
  }

  return view;
}

/**
 * "high" con una GPU, "low" con il rendering software (SwiftShader, llvmpipe): lì una corteccia da
 * 150.000 triangoli costerebbe decine di millisecondi per frame. Si legge prima RENDERER (Firefox lo dà
 * già in chiaro, e l'estensione lì è deprecata e lo direbbe in console); solo se è mascherato (Chrome:
 * "WebKit WebGL") si chiede all'estensione.
 */
export function detectQuality(renderer) {
  try {
    const gl = renderer.getContext();
    let name = String(gl.getParameter(gl.RENDERER) ?? "");
    if (/^webkit webgl$/i.test(name)) {
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      if (ext) name = String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) ?? name);
    }
    return /swiftshader|llvmpipe|software|softpipe/i.test(name) ? "low" : "high";
  } catch {
    return "high";
  }
}
