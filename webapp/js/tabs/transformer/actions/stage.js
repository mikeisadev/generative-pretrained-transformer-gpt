/**
 * Regia delle azioni nella scena del Transformer: lavagna accanto al componente, camera,
 * pacchetto del flusso, oggetti della scena da nascondere o aggiornare, e il ripristino di tutto
 * a fine azione (o quando l'utente la interrompe).
 *
 * Ogni modifica alla scena di base passa da qui e viene annotata, così `restore()` rimette la
 * scena esattamente com'era: pesi addestrati, pattern d'esempio, probabilità d'esempio, flusso.
 */
import * as THREE from "three";
import { meshBounds } from "../../../core/director.js";
import { Board } from "./board.js";
import { SceneTextures } from "./sceneTextures.js";

const GAP = 0.8;

export class Stage {
  constructor(tab) {
    this.tab = tab;
    this.host = tab.ui.host;
    this.director = tab.ui.director;
    this.overlay = new THREE.Group();
    this.overlay.name = "transformer-actions-overlay";
    this.host.scene.add(this.overlay);
    this.board = new Board();
    this.overlay.add(this.board.mesh);
    this.textures = new SceneTextures(tab);
    this.stations = new Map(tab.arch.stations.map((s) => [s.id, s]));
    this.hidden = new Map(); // oggetto → visibilità originale
    this.bars = null; // stato originale delle barre di probabilità
    this.flowState = null;
    this.pulsed = null;
    this.tmp = new THREE.Vector3();
    this.pathLength = tab.arch.path.getLength();
    // la lavagna guarda sempre la camera: il testo resta dritto e leggibile da ogni angolazione
    this.host.onFrame(() => {
      if (this.board.mesh.visible) this.board.mesh.quaternion.copy(this.host.camera.quaternion);
    });
  }

  /* --------------------------------------------------------------- camera e lavagna */

  /** Oggetto del registro (id) o oggetto 3D. */
  resolve(target) {
    return typeof target === "string" ? this.tab.arch.registry.get(target) : target;
  }

  /** Box3 nel mondo di un id del registro, di un oggetto o di un Box3 già pronto. */
  bounds(target) {
    if (!target) return new THREE.Box3();
    if (target.isBox3) return target.clone();
    const object = this.resolve(target);
    return object ? meshBounds(object) : new THREE.Box3();
  }

  /**
   * Area dello stage davvero libera, in pixel: tolti i pannelli a destra (codice, grafici), la
   * barra degli strumenti in alto e la barra del player in basso.
   */
  freeArea() {
    const container = this.host.container;
    const width = Math.max(200, container.clientWidth - this.host.coveredRight());
    const height = Math.max(200, container.clientHeight);
    const top = 64;
    let bottom = 16;
    const bar = this.tab.ui.player?.bar;
    if (bar && !bar.hidden) bottom = Math.max(bottom, container.getBoundingClientRect().bottom - bar.getBoundingClientRect().top + 10);
    return { width, height, top, bottom, free: Math.max(120, height - top - bottom) };
  }

  /**
   * Inquadra `target` con la lavagna accanto; `draw(board)` disegna il contenuto.
   * La lavagna è più grande del componente (è lì che si leggono i numeri) e si mette a sinistra
   * o sopra, scegliendo la disposizione che la fa apparire più grande nell'area libera.
   * @param {object} [o]
   * @param {"auto"|"left"|"right"|"above"} [o.side="auto"]
   * @param {number} [o.scale=1.5] altezza della lavagna rispetto al componente
   * @param {Array<THREE.Object3D|THREE.Box3|string>} [o.extra] altro da includere nell'inquadratura
   */
  show(target, draw, { side = "auto", scale = 2, extra = [], duration = 1.1 } = {}) {
    const box = this.bounds(target);
    for (const o of extra) box.union(this.bounds(o));
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const area = this.freeArea();
    const board = this.board;
    const frame = box.clone();
    if (draw) {
      draw(board);
      board.commit();
      board.mesh.visible = true;
      const A = board.aspect;
      const h = THREE.MathUtils.clamp(Math.max(size.y, size.x / A) * scale, 3, 60);
      const w = h * A;
      // pixel per unità di scena con ogni disposizione: vince quella con la lavagna più grande
      const fit = (uw, uh) => Math.min(area.width / uw, area.free / uh);
      const options = {
        left: fit(w + GAP + size.x, Math.max(h, size.y)),
        right: fit(w + GAP + size.x, Math.max(h, size.y)),
        above: fit(Math.max(w, size.x), h + GAP + size.y),
      };
      const chosen = side === "auto" ? (options.left >= options.above ? "left" : "above") : side;
      board.setHeight(h);
      const z = box.max.z + 0.6;
      if (chosen === "left") board.mesh.position.set(box.min.x - GAP - w / 2, center.y, z);
      else if (chosen === "right") board.mesh.position.set(box.max.x + GAP + w / 2, center.y, z);
      else board.mesh.position.set(center.x, box.max.y + GAP + h / 2, z);
      const half = new THREE.Vector3(w / 2, h / 2, 0.01);
      frame.union(new THREE.Box3(board.mesh.position.clone().sub(half), board.mesh.position.clone().add(half)));
    } else {
      board.mesh.visible = false;
    }
    // Il riquadro da inquadrare si allunga in alto e in basso quanto le barre che coprono lo
    // stage: così il contenuto finisce nella parte libera e non sotto il player.
    const fsize = frame.getSize(new THREE.Vector3());
    const ppu = Math.min(area.width / Math.max(fsize.x, 1e-3), area.free / Math.max(fsize.y, 1e-3)) / 1.06;
    frame.max.y += area.top / ppu;
    frame.min.y -= area.bottom / ppu;
    const done = this.director.focus(frame, { showFrame: false, margin: 1.06, duration, minDistance: 2 });
    // il riquadro pulsante resta sul componente, non sulla lavagna
    if (!box.isEmpty()) {
      const hl = this.director.highlight;
      hl.visible = true;
      hl.position.copy(center);
      hl.scale.set(size.x + 0.25, size.y + 0.25, Math.max(size.z, 0.1) + 0.25);
    }
    return done;
  }

  /** Ridisegna la lavagna senza muovere la camera. */
  redraw(draw) {
    draw(this.board);
    this.board.commit();
  }

  hideBoard() {
    this.board.mesh.visible = false;
  }

  /* --------------------------------------------------------------- pacchetto del flusso */

  /** Il pacchetto del flusso passa sotto il controllo dell'azione (il giro automatico si ferma). */
  takeFlow() {
    const flow = this.tab.flow;
    if (this.flowState) return;
    this.flowState = { playing: flow.playing, distance: flow.distance };
    flow.playing = false;
    flow.setPulse(flow.current, false);
    flow.current = null;
  }

  /** Distanza lungo il binario di una stazione (id del registro). */
  distanceOf(id) {
    return this.stations.get(id)?.distance ?? 0;
  }

  /** Porta il pacchetto alla distanza d e illumina la stazione `id` (se data). */
  packetAt(d, id = null, { caption = true } = {}) {
    const flow = this.tab.flow;
    this.tab.arch.path.getPointAt(THREE.MathUtils.clamp(d / this.pathLength, 0, 1), this.tmp);
    flow.packet.position.copy(this.tmp);
    const station = id ? this.stations.get(id) : null;
    if (station !== this.pulsed) {
      flow.setPulse(this.pulsed, false);
      flow.setPulse(station, true);
      this.pulsed = station;
      if (station && caption) this.tab.setCaption(station.caption); // la didascalia in basso segue il pacchetto
    }
    flow.current = station; // il LOD (livelli Blocchi/Panoramica) illumina il rettangolo della stazione corrente
  }

  /** Movimento del pacchetto da una stazione all'altra (p 0..1). */
  packetBetween(fromId, toId, p) {
    const a = this.distanceOf(fromId);
    const b = this.distanceOf(toId);
    this.packetAt(a + (b - a) * p, p >= 0.98 ? toId : fromId);
  }

  /** Stazione attiva lungo tutto il binario in funzione della distanza (per i passaggi veloci). */
  stationAt(d) {
    let active = null;
    for (const s of this.tab.arch.stations) if (s.distance <= d + 1e-6) active = s;
    return active;
  }

  get length() {
    return this.pathLength;
  }

  /**
   * Blocca lo zoom semantico sul livello "Dettaglio" finché l'azione è attiva: le azioni mostrano
   * matrici e pattern veri, che esistono solo a quel livello. A fine azione torna com'era.
   */
  lockDetail() {
    const controller = this.tab.lod?.controller;
    if (!controller) return;
    this.lodMode ??= controller.mode;
    if (controller.mode !== "detail") controller.setMode("detail");
  }

  /* --------------------------------------------------------------- oggetti della scena */

  /** Nasconde un oggetto della scena di base (ricordando com'era). */
  hide(target) {
    const object = this.resolve(target);
    if (!object) return;
    if (!this.hidden.has(object)) this.hidden.set(object, object.visible);
    object.visible = false;
  }

  /**
   * Aggiorna le barre 3D "softmax → probabilità" di architecture.js con le probabilità vere del
   * prompt. Le etichette d'esempio (top-5 del prompt di model.json) vengono nascoste.
   */
  setSceneProbs(probs, chosen = -1) {
    const group = this.resolve("logits");
    if (!group) return;
    const bars = group.children.filter((c) => c.isMesh);
    if (!this.bars) {
      this.bars = bars.map((b) => ({
        mesh: b,
        scale: b.scale.x,
        x: b.position.x,
        color: b.material.color.clone(),
        emissive: b.material.emissive?.clone(),
        base: b.userData.baseColor?.clone(),
        len: b.geometry.parameters.width,
      }));
      group.children.filter((c) => c.isSprite && Math.abs(c.position.y) < 4.1).forEach((s) => this.hide(s));
    }
    const maxLen = 5; // come buildOutput di architecture.js
    this.bars.forEach((info, i) => {
      const len = Math.max(0.02, (probs[i] ?? 0) * maxLen);
      info.mesh.scale.x = len / info.len;
      info.mesh.position.x = len / 2;
      const color = i === chosen ? 0xfbbf24 : 0x475569;
      info.mesh.material.color.setHex(color);
      info.mesh.material.emissive?.setHex(i === chosen ? 0x92400e : 0);
      info.mesh.userData.baseColor = info.mesh.material.color.clone(); // il pulse del flusso parte da qui
    });
  }

  /* --------------------------------------------------------------- ripristino */

  /** Rimette la scena com'era prima dell'azione. Idempotente. */
  restore() {
    this.textures.restore();
    for (const [object, visible] of this.hidden) object.visible = visible;
    this.hidden.clear();
    if (this.bars) {
      for (const info of this.bars) {
        info.mesh.scale.x = info.scale;
        info.mesh.position.x = info.x;
        info.mesh.material.color.copy(info.color);
        if (info.emissive) info.mesh.material.emissive.copy(info.emissive);
        if (info.base) info.mesh.userData.baseColor = info.base;
        else delete info.mesh.userData.baseColor;
      }
      this.bars = null;
    }
    const flow = this.tab.flow;
    flow.setPulse(this.pulsed, false);
    this.pulsed = null;
    flow.current = null;
    if (this.lodMode) {
      this.tab.lod?.controller.setMode(this.lodMode);
      this.lodMode = null;
    }
    if (this.flowState) {
      flow.playing = this.flowState.playing;
      flow.distance = this.flowState.distance;
      flow.moveTo(flow.distance);
      this.flowState = null;
    }
    this.board.mesh.visible = false;
    this.director.hideFrame();
  }
}
