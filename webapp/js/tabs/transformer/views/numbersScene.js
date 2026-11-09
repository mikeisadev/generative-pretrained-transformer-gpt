/**
 * Vista "Pesi numerici": la parte three.js sulla scena dell'architettura.
 *
 * - Numeri: un solo NumbersGroup (il budget del nucleo: al più 4 matrici con i numeri, le più grandi
 *   sullo schermo) e attachNumbers su ogni heatmap di tab.arch.tensorMeshes. I valori vengono dalla
 *   sorgente scelta (pesi finali o uno step del training): `setViews` la cambia.
 * - Heatmap: per uno step del training si riscrivono con SceneTextures.writeWeights (stessa colormap e
 *   stessa scala dei pesi finali, così si vede quanto crescono); i numeri seguono da soli perché la
 *   texture cambia versione. Tornando ai pesi finali, o uscendo dalla vista, le texture tornano
 *   esattamente quelle di prima (copie salvate da SceneTextures).
 * - Camera: minDistance abbassata (≈ 0,2) solo qui, perché i numeri compaiono da vicino (celle di
 *   ~36 px); voli sul tensore e sulla cella.
 * - Le tre carte di una head (query, key, value) sono sovrapposte: quella scelta passa davanti.
 * - Con uno step del training le attivazioni calcolate con i pesi finali (pattern di attention, probabilità,
 *   prossimo token) si attenuano: non corrispondono ai pesi mostrati.
 * - Puntatore: cella sotto il mouse (tensore, riga, colonna, valore), ricalcolata anche quando la camera si
 *   muove da sola sotto il mouse fermo (rotellina, tasti, voli), e clic su una cella.
 * Uscendo (exit) tutto torna com'era: texture, carte, opacità, minDistance, listener, overlay.
 */
import * as THREE from "three";
import { setBaseOpacity } from "../../../core/lod.js";
import { NumbersGroup, attachNumbers } from "../../../core/matrixNumbers.js";
import { CELL } from "../../../core/visuals.js";
import { SceneTextures } from "../actions/sceneTextures.js";
import { CARD_KINDS, cardDepths, headCard } from "./numbersModel.js";

/** minDistance della camera in questa vista: a 0,2 unità una cella (0,04) è grande quasi metà schermo. */
export const NUMBERS_MIN_DISTANCE = 0.2;
/** Altezza della cella sullo schermo quando la camera vola su una cella: 4 decimali leggibili. */
const CELL_TARGET_PX = 72;
const CLICK_SLOP = 5; // px: oltre è un trascinamento (navigazione), non un clic
/** Opacità (rispetto a quella normale) delle attivazioni dei pesi finali mentre la vista mostra uno step del training. */
const STALE_ACTIVATION_OPACITY = 0.2;

export class NumbersScene {
  /**
   * @param {object} tab l'oggetto tab del Transformer
   * @param {{onHover?: (hit: object|null, event?: PointerEvent) => void, onPick?: (hit: object, event: PointerEvent) => void}} [o]
   *   hit = { name, row, col, value }
   */
  constructor(tab, { onHover = () => {}, onPick = () => {} } = {}) {
    this.tab = tab;
    this.host = tab.ui.host;
    this.onHover = onHover;
    this.onPick = onPick;
    this.textures = new SceneTextures(tab);
    this.active = false;
    this.views = null;
    this.entries = []; // { name, mesh, handle }
    this.byMesh = new Map();
    this.byName = new Map();
    this.meshes = [];
    this.group = null;
    this.highlighted = null;
    this.cards = new Map(); // mesh → z originale (carte delle head spostate davanti)
    this.saved = null;
    this.dimmed = null; // materiale → opacità di base, mentre le attivazioni dei pesi finali sono attenuate
    this.pending = 0;
    this.downAt = null;
    this.lastMove = null;
    this.stats = { writeMs: 0, lastStepMs: 0 };
    const canvas = this.host.renderer.domElement;
    this.canvas = canvas;
    this.listeners = [
      ["pointermove", (e) => this.move(e)],
      ["pointerdown", (e) => (this.downAt = [e.clientX, e.clientY])],
      ["pointerup", (e) => this.up(e)],
      [
        "pointerleave",
        () => {
          this.lastMove = null;
          this.hover(null);
        },
      ],
    ];
    this.dragging = false;
    // Si trascina la scena: niente etichetta mentre gira (la cella sotto il puntatore cambia di continuo).
    this.onControlsStart = () => {
      this.dragging = true;
      this.hover(null);
    };
    // Fine del trascinamento o di una tacca della rotellina (le OrbitControls la avvolgono in start/end, con lo
    // zoom già applicato in mezzo): la cella sotto il puntatore si ricalcola.
    this.onControlsEnd = () => {
      this.dragging = false;
      if (this.lastMove) this.schedulePick();
    };
    // La camera si muove senza il mouse (rotellina, tasti, "Centra", "Mostra nella scena"): sotto il
    // puntatore fermo c'è un'altra cella e l'etichetta va ricalcolata, altrimenti ne descriverebbe una che
    // non c'è più. "change" delle OrbitControls arriva solo quando la camera si muove: niente lavoro a riposo.
    this.onCameraChange = () => {
      if (!this.dragging && this.lastMove) this.schedulePick();
    };
  }

  /** Entra nella vista con la sorgente `views` (Map nome → Float32Array). */
  enter(views) {
    if (this.active) return;
    this.active = true;
    this.views = views;
    const { host, tab } = this;
    this.group = new NumbersGroup({ maxOverlays: 4 });
    for (const [name, meshes] of tab.arch.tensorMeshes) {
      for (const mesh of meshes) {
        const { width: cols } = mesh.material.map.image; // (n, 1) per le barre dei vettori
        const handle = attachNumbers(mesh, { group: this.group, getValue: (r, c) => this.views.get(name)[r * cols + c] });
        const entry = { name, mesh, handle };
        this.entries.push(entry);
        this.byMesh.set(mesh, entry);
        if (!this.byName.has(name)) this.byName.set(name, entry);
        this.meshes.push(mesh);
      }
    }
    this.saved = { minDistance: host.controls.minDistance };
    host.controls.minDistance = NUMBERS_MIN_DISTANCE;
    // Una sola chiamata per frame: a riposo confronta camera e versioni ed esce (~1 µs, nessuna allocazione).
    this.offFrame = host.onFrame(() => this.group.update(host.camera, host.container.clientHeight));
    for (const [type, fn] of this.listeners) this.canvas.addEventListener(type, fn);
    host.controls.addEventListener("start", this.onControlsStart);
    host.controls.addEventListener("end", this.onControlsEnd);
    host.controls.addEventListener("change", this.onCameraChange);
  }

  /**
   * Nuova sorgente dei numeri. `heatmaps`: "final" rimette le texture originali (i pesi finali di
   * model.json), "write" le riscrive con questi valori (uno step del training).
   * @returns {number} millisecondi spesi a riscrivere le heatmap
   */
  setViews(views, { heatmaps }) {
    if (!this.active) return 0;
    const t0 = performance.now();
    this.views = views;
    if (heatmaps === "write") this.textures.writeWeights(views);
    else this.textures.restore();
    // le texture riscritte cambiano versione (i numeri si ridisegnano da soli); i valori cambiano anche
    // dove la texture resta uguale (es. stessi byte quantizzati), quindi si avvisa ogni bersaglio
    for (const e of this.entries) e.handle.refresh();
    this.group.invalidate();
    this.stats.writeMs = performance.now() - t0;
    return this.stats.writeMs;
  }

  /** Riquadro sulla cella (r, c) del tensore `name` (null toglie il riquadro). */
  highlight(name, cell) {
    const entry = name ? this.byName.get(name) : null;
    if (this.highlighted && this.highlighted !== entry) this.highlighted.handle.highlight(null);
    this.highlighted = entry && cell ? entry : null;
    if (entry) entry.handle.highlight(cell ? { row: cell.row, col: cell.col } : null);
  }

  /** La carta scelta di una head (query, key o value) passa davanti alle altre due. */
  bringToFront(name) {
    const card = headCard(name);
    if (!card || !this.active) return;
    const prefix = `blocks.${card.layer}.sa.heads.${card.head}.`;
    const meshes = Object.fromEntries(CARD_KINDS.map((k) => [k, this.byName.get(`${prefix}${k}.weight`)?.mesh]));
    if (!CARD_KINDS.every((k) => meshes[k])) return;
    for (const k of CARD_KINDS) if (!this.cards.has(meshes[k])) this.cards.set(meshes[k], meshes[k].position.z);
    const depths = cardDepths(Object.fromEntries(CARD_KINDS.map((k) => [k, this.cards.get(meshes[k])])), card.kind);
    for (const k of CARD_KINDS) meshes[k].position.z = depths[k];
    this.group.invalidate(); // matrici spostate da codice: i numeri si ricalcolano anche a camera ferma
  }

  /**
   * Pattern di attention, probabilità del prossimo carattere e prossimo token sono ATTIVAZIONI, non pesi: il
   * motore le ha calcolate con i pesi finali sul prompt d'esempio (model.json). Accanto ai pesi di uno step del
   * training non corrisponderebbero più (allo step 0 si vedrebbe l'attention del modello addestrato vicino a
   * pesi casuali, come se non dipendesse dai pesi): si attenuano finché la sorgente non torna ai pesi finali.
   * setBaseOpacity (core/lod.js) convive con le sfumature dello zoom semantico.
   */
  dimFinalActivations(on) {
    if (!this.active || on === Boolean(this.dimmed)) return;
    if (!on) {
      for (const [material, base] of this.dimmed) setBaseOpacity(material, base);
      this.dimmed = null;
      return;
    }
    this.dimmed = new Map();
    const { registry } = this.tab.arch;
    const { n_layer: layers, n_head: heads } = this.tab.spec.config;
    const roots = [registry.get("logits"), registry.get("next_token")];
    for (let l = 0; l < layers; l++) for (let h = 0; h < heads; h++) roots.push(registry.get(`blocks.${l}.heads.${h}.pattern`));
    for (const root of roots) {
      root?.traverse((o) => {
        const material = o.material;
        if (!material || Array.isArray(material) || this.dimmed.has(material)) return;
        const base = material.userData.baseOpacity ?? material.opacity;
        this.dimmed.set(material, base);
        setBaseOpacity(material, base * STALE_ACTIVATION_OPACITY);
      });
    }
  }

  /** Mesh della heatmap di un tensore (per inquadrarlo). */
  meshOf(name) {
    return this.byName.get(name)?.mesh ?? null;
  }

  /** Vola sul tensore intero (inquadratura stretta: minDistance bassa, la matrice riempie la vista). */
  focusTensor(name, { duration = 1 } = {}) {
    const mesh = this.meshOf(name);
    if (!mesh) return Promise.resolve();
    const box = new THREE.Box3().setFromObject(mesh, true);
    return this.tab.ui.director.focus(box, { showFrame: false, margin: 1.3, minDistance: 0.8, duration, direction: new THREE.Vector3(0, 0.04, 1) });
  }

  /** Vola davanti alla cella: abbastanza vicino da leggerne il numero (≈ 72 px per cella). */
  focusCell(name, cell, { duration = 0.9 } = {}) {
    const mesh = this.meshOf(name);
    if (!mesh || !cell) return Promise.resolve();
    const { host } = this;
    const image = mesh.material.map.image;
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    const bb = mesh.geometry.boundingBox;
    const cw = (bb.max.x - bb.min.x) / image.width;
    const ch = (bb.max.y - bb.min.y) / image.height;
    mesh.updateWorldMatrix(true, false);
    const target = new THREE.Vector3(bb.min.x + (cell.col + 0.5) * cw, bb.max.y - (cell.row + 0.5) * ch, 0).applyMatrix4(mesh.matrixWorld);
    const normal = new THREE.Vector3(0, 0, 1).transformDirection(mesh.matrixWorld);
    const fov = THREE.MathUtils.degToRad(host.camera.fov);
    const height = host.container.clientHeight || 800;
    const cellWorld = CELL * Math.abs(mesh.getWorldScale(new THREE.Vector3()).y);
    const distance = Math.max(NUMBERS_MIN_DISTANCE * 1.5, (cellWorld * height) / (2 * Math.tan(fov / 2) * CELL_TARGET_PX));
    return this.tab.ui.director.flyTo(target.clone().addScaledVector(normal, distance), target, duration);
  }

  /** Quante matrici mostrano i numeri ora e il costo dell'ultimo ridisegno (console, script di verifica). */
  get numbersStats() {
    return { ...this.group?.stats, shown: this.group?.shown().map((s) => ({ name: this.byMesh.get(s.target.mesh)?.name, window: s.window, decimals: s.decimals })) ?? [] };
  }

  // ---------------------------------------------------------------- puntatore

  /** Cella della heatmap sotto il puntatore (solo al livello Dettaglio, dove le matrici si vedono). */
  pick(event) {
    if (this.tab.lod.current !== "detail") return null;
    const hit = this.host.pick(event, this.meshes, false)[0];
    if (!hit) return null;
    const entry = this.byMesh.get(hit.object);
    const cell = entry?.handle.cellAt(hit);
    return cell ? { name: entry.name, row: cell.row, col: cell.col, value: cell.value } : null;
  }

  move(event) {
    if (event.pointerType === "touch") return; // sul telefono l'etichetta arriva con un tocco (up)
    this.lastMove = event;
    if (!this.dragging) this.schedulePick();
  }

  /** Cella sotto l'ultima posizione del mouse, al prossimo frame: un raycast per frame al più (94 piani: ~0,1 ms). */
  schedulePick() {
    if (this.pending) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      if (!this.active || !this.lastMove) return;
      this.hover(this.pick(this.lastMove), this.lastMove);
    });
  }

  hover(hit, event) {
    this.onHover(hit, event);
  }

  up(event) {
    if (!this.downAt || Math.hypot(event.clientX - this.downAt[0], event.clientY - this.downAt[1]) > CLICK_SLOP) return;
    this.downAt = null;
    const hit = this.pick(event);
    if (hit) this.onPick(hit, event);
  }

  /** Esce dalla vista: texture, carte, camera e listener come prima; overlay e numeri liberati. */
  exit() {
    if (!this.active) return;
    this.dimFinalActivations(false);
    this.active = false;
    cancelAnimationFrame(this.pending);
    this.pending = 0;
    this.offFrame?.();
    this.offFrame = null;
    for (const [type, fn] of this.listeners) this.canvas.removeEventListener(type, fn);
    this.host.controls.removeEventListener("start", this.onControlsStart);
    this.host.controls.removeEventListener("end", this.onControlsEnd);
    this.host.controls.removeEventListener("change", this.onCameraChange);
    this.lastMove = null;
    this.dragging = false;
    this.textures.restore();
    for (const [mesh, z] of this.cards) mesh.position.z = z;
    this.cards.clear();
    for (const e of this.entries) e.handle.dispose(); // toglie anche il riquadro della cella
    this.group.dispose();
    this.group = null;
    this.entries = [];
    this.byMesh.clear();
    this.byName.clear();
    this.meshes = [];
    this.highlighted = null;
    if (this.saved) this.host.controls.minDistance = this.saved.minDistance;
    this.saved = null;
    this.views = null;
  }
}
