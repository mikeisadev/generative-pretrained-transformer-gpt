/**
 * Regia delle azioni nella scena del Laboratorio (three.js + DOM): le carte temporanee accanto ai pesi
 * veri, le celle evidenziate sulle matrici della scena, le linee che collegano i numeri, la lavagna, il
 * codice vero, la camera; e il RITORNO alla scena com'era a fine azione.
 *
 *   const stage = new ActionStage(tab);
 *   stage.begin();                                   // onStart: fotografa la scena (camera, zoom, codice, valori mostrati)
 *   stage.card("x", { rect, data, label });          // in enter(): le carte del passo (riusate per chiave)
 *   stage.highlight(name, { rows: [3] });            // velo sul resto della matrice della scena
 *   stage.frame(rect);                               // camera sulla parte libera dello stage
 *   stage.end({ flyBack: true });                    // onStop: tutto torna com'era, oggetti liberati
 *
 * Ogni passo dichiara TUTTO ciò che gli serve (stage.step() nasconde le carte e i veli del passo prima):
 * così "passo indietro" e "⟲ Ricomincia" funzionano sempre. Nessun lavoro per frame a riposo: il
 * NumbersGroup delle carte si aggiorna in un onFrame registrato solo mentre un'azione è aperta.
 */
import * as THREE from "three";
import { NumbersGroup } from "../../../core/matrixNumbers.js";
import { prefersReducedMotion } from "../../../core/dom.js";
import { Board } from "./board.js";
import { Card, LABEL_PX, TokenStrip } from "./cards.js";
import { padBottom } from "../../../core/navMath.js";
import { cardRect, codeAboveBoard, deskLayout, framePose, padRect, zAbove } from "./plan.js";

/** Comandi fluttuanti in alto nello stage: la camera non inquadra lì sotto. */
const TOP_OVERLAYS = ".sidebar-toggle, .lod-switch, .stage-toolbar, .charts-fab, .rep-switch";
const FRONT = new THREE.Vector3(0, 0.02, 1).normalize();
const _dir = new THREE.Vector3();
const _p = new THREE.Vector3();
const narrow = () => window.matchMedia("(max-width: 900px)").matches;

export class ActionStage {
  constructor(tab) {
    this.tab = tab;
    this.ui = tab.ui;
    this.group = new THREE.Group();
    this.group.name = "lab-actions";
    this.numbers = new NumbersGroup({ maxOverlays: 4, maxCellsPerOverlay: 1600 });
    this.cards = new Map(); // chiave → Card
    this.strips = new Map(); // chiave → { strip: TokenStrip, signature }
    this.used = new Set(); // chiavi usate dal passo in corso
    this.highlighted = new Set(); // matrici della scena con il velo
    this.links = null; // LineSegments delle linee tra i numeri
    this.outlines = null; // LineSegments dei riquadri attorno a pezzi della scena
    this.valueNames = null; // tensori della scena con valori dell'azione (showValues)
    this.board = new Board(tab.ui.stage);
    this.boardChoice = null; // ridotta o aperta per scelta dello studente (vale per tutta la visita)
    this.board.onToggle = (collapsed, user) => {
      if (user) this.boardChoice = collapsed;
      this.reframe();
    };
    this.saved = null;
    this.active = false;
    this.touchedStep = false; // l'azione ha cambiato lo step del training: a fine azione si rimette
    this.lastFrame = null;
    this.stats = { enters: [], updates: 0, updateMs: 0, maxUpdateMs: 0, builds: [] };
    this._stopFrame = null;
    this._resize = null;
    this._frameToken = 0;
    this._frameRaf = 0;
  }

  get scene() {
    return this.tab.scene.scene;
  }

  /** true se l'utente ha chiesto meno movimento: niente comparse progressive, voli brevi. */
  get reduced() {
    return prefersReducedMotion();
  }

  // ---------------------------------------------------------------- inizio e fine
  /** Fotografa la scena e prepara la regia (onStart di ogni azione). */
  begin() {
    if (this.active) this.end({ flyBack: false });
    const { ui, state } = this.tab;
    const S = this.scene;
    const code = ui.codePanel;
    // Valori mostrati da un altro modulo (la differenza A − B del Confronto): durante l'azione la scena
    // mostra i pesi veri del run A, quelli di cui parlano le didascalie. Chi possiede quei valori si
    // spegne da solo quando parte un'azione (analysis/diffMode.js: "stopForAction", e il suo pannello lo
    // dice): a fine azione non li rimettiamo noi, li riaccende il loro pulsante.
    const custom = S.customValues.size > 0;
    this.saved = {
      camera: ui.host.camera.position.clone(),
      target: ui.host.controls.target.clone(),
      lod: this.tab.scene.lod.mode,
      step: state.step,
      runA: state.runA,
      mode: state.mode,
      modeLabel: state.modeLabel,
      custom,
      code: code ? { hidden: code.root.hidden, minimized: code.root.classList.contains("minimized"), user: code.userMinimized } : null,
      caption: ui.caption.textContent,
      selected: state.selected,
    };
    if (custom) S.showValues(null);
    if (state.mode !== "pesi") state.setMode("pesi");
    this.tab.scene.hideTip?.();
    S.hoverCell(null, null);
    this.tab.scene.lod.setMode("detail");
    this.touchedStep = false;
    if (!this.group.parent) ui.host.scene.add(this.group);
    // per frame, solo mentre l'azione è aperta: i numeri delle carte e le etichette (escono subito a camera ferma)
    this._stopFrame ??= ui.host.onFrame(() => {
      this.numbers.update(ui.host.camera, ui.host.container.clientHeight);
      this.updateLabels();
    });
    this._resize ??= ui.host.onResize(() => this.reframe());
    // il pad aperto, ridotto o spostato dal nucleo cambia la larghezza della lavagna; il chip "Segui
    // l'azione" alza la lavagna (e il codice deve finire sopra di lei): a eventi, mai per frame
    this._padObserver ??= new MutationObserver(() => this.syncLayout());
    const pad = ui.stage.querySelector(".nav-pad");
    if (pad) this._padObserver.observe(pad, { attributes: true, attributeFilter: ["style", "class", "hidden"] });
    const chip = ui.root.querySelector(".nav-follow");
    if (chip) this._padObserver.observe(chip, { attributes: true, attributeFilter: ["hidden"] });
    // sul telefono la lavagna parte ridotta (la scena ha poco spazio; la didascalia del player dice
    // l'essenziale): un tocco la apre, e la scelta dello studente vale per le azioni dopo
    this.board.setCollapsed(this.boardChoice ?? narrow());
    ui.root.classList.add("lab-act-playing");
    ui.host.resize(); // il codice è più stretto durante l'azione (lab-actions.css): il view offset della camera cambia
    this.active = true;
  }

  /**
   * Fine dell'azione (chiusa, interrotta o sostituita): via carte, veli, linee e lavagna; tornano camera,
   * zoom, codice, valori mostrati e (per "I pesi durante il training") lo step del training.
   */
  end({ flyBack = true } = {}) {
    if (!this.active) return;
    this.active = false;
    const { ui, state } = this.tab;
    const S = this.scene;
    const saved = this.saved;
    this.saved = null;
    this._stopFrame?.();
    this._stopFrame = null;
    this._resize?.();
    this._resize = null;
    this._padObserver?.disconnect();
    this._frameToken++;
    cancelAnimationFrame(this._frameRaf);
    for (const card of this.cards.values()) card.dispose();
    this.cards.clear();
    for (const { strip } of this.strips.values()) strip.dispose();
    this.strips.clear();
    this.used.clear();
    this.valueNames = null; // i valori dell'azione si tolgono qui sotto (showValues(null)), con la modalità
    this.clearHighlights();
    this.clearLinks();
    this.group.removeFromParent();
    this.board.hide();
    ui.root.classList.remove("lab-act-playing");
    for (const v of ["--lab-act-board-h", "--lab-act-max-w", "--lab-act-player-h", "--lab-act-left-px", "--lab-act-code-max"]) ui.root.style.removeProperty(v);
    ui.host.resize();
    this.lastFrame = null;
    if (!saved) return;
    // i valori dell'azione sulla scena (es. lm_head dopo un passo) tornano ai pesi; la modalità di prima
    // torna se non era quella di un altro modulo che si è spento per l'azione (vedi begin)
    if (S.customValues.size) S.showValues(null);
    const mode = saved.custom ? "pesi" : saved.mode;
    const label = saved.custom ? "" : saved.modeLabel;
    if (state.mode !== mode || state.modeLabel !== label) state.setMode(mode, { label });
    if (this.tab.scene.lod.mode !== saved.lod) this.tab.scene.lod.setMode(saved.lod);
    // lo step del training di prima (solo se l'azione lo ha cambiato: "I pesi durante il training")
    if (this.touchedStep && state.runA === saved.runA && state.step !== saved.step) void state.setStep(saved.step);
    this.touchedStep = false;
    this.restoreCode(saved.code, saved.selected);
    ui.caption.textContent = saved.caption;
    // camera: dove lo studente era prima dell'azione. Non con la Rete neurale o il Cervello aperti: un
    // volo della scena del tab riporterebbe all'Architettura (ritorno automatico del nucleo)
    const covered = this.tab.reps && this.tab.reps.current !== "main";
    if (flyBack && !covered && state.runA === saved.runA) {
      ui.director.noteUserIntent();
      void ui.director.flyTo(saved.camera, saved.target, this.reduced ? 0.3 : 0.9);
    }
  }

  restoreCode(code, selected) {
    const panel = this.ui.codePanel;
    if (!panel || !code) return;
    if (code.hidden) {
      panel.hide();
    } else if (selected && this.scene.nodeOf(selected)) {
      this.tab.showCode(selected, { expand: !code.minimized });
    }
    if (!panel.root.hidden) panel.root.classList.toggle("minimized", code.minimized);
    panel.userMinimized = code.user;
  }

  // ---------------------------------------------------------------- un passo
  /** Inizio di un passo: le carte e i veli del passo prima si nascondono (il passo dichiara i suoi). */
  step() {
    this.used.clear();
    for (const card of this.cards.values()) card.visible = false;
    for (const { strip } of this.strips.values()) strip.object3d.visible = false;
    this.clearValues();
    this.clearHighlights();
    this.clearLinks();
    this.scene.hoverCell(null, null);
  }

  /**
   * Una carta temporanea (creata la prima volta, poi riusata per chiave con i dati del passo).
   * @param {string} key
   * @param {object} spec opzioni di Card (rect, data, colormap, range, label, labelSide, accent, maskedText, …)
   */
  card(key, spec) {
    let card = this.cards.get(key);
    const same = card && card.rows === spec.rect.rows && card.cols === spec.rect.cols && Math.abs(card.rect.x0 - spec.rect.x0) < 1e-6 && Math.abs(card.rect.y1 - spec.rect.y1) < 1e-6 && card.labelText === (spec.label ?? "");
    if (card && !same) {
      card.dispose();
      this.cards.delete(key);
      card = null;
    }
    if (!card) {
      // appena davanti a ciò che copre (plan.zAbove): vicina al piano della scena, niente parallasse
      const z = spec.z ?? zAbove(this.scene.layout?.boxes, padRect(cardRect(spec.rect), 0.06));
      card = new Card({ ...spec, z, group: this.numbers });
      card.labelText = spec.label ?? "";
      this.cards.set(key, card);
      this.group.add(card.object3d);
    } else if (spec.data) {
      card.setData(spec.data, spec.range > 0 ? { range: spec.range } : spec.range === "auto" ? { range: "auto" } : undefined);
    }
    card.visible = true;
    this._labelsDirty = true; // una carta (ri)comparsa: le etichette si riscalano al prossimo frame
    card.mark(null);
    card.configure({ reveal: spec.reveal ?? Infinity, mask: spec.mask ?? null });
    this.used.add(key);
    return card;
  }

  /**
   * Le etichette dei token lungo le righe o le colonne di una carta (riusate per chiave se uguali).
   * @param {string} key @param {string[]} texts @param {{x: number, y: number}[]} positions
   */
  strip(key, texts, positions, options = {}) {
    const signature = `${texts.join("\u0001")}|${positions.map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`).join(";")}|${options.marked ?? ""}|${options.height ?? ""}`;
    let entry = this.strips.get(key);
    if (entry && entry.signature !== signature) {
      entry.strip.dispose();
      this.strips.delete(key);
      entry = null;
    }
    if (!entry) {
      const xs = positions.map((p) => p.x);
      const ys = positions.map((p) => p.y);
      const z = options.z ?? zAbove(this.scene.layout?.boxes, { x0: Math.min(...xs) - 0.05, x1: Math.max(...xs) + 0.05, y0: Math.min(...ys) - 0.05, y1: Math.max(...ys) + 0.05 });
      entry = { strip: new TokenStrip(texts, positions, { ...options, z }), signature };
      this.strips.set(key, entry);
      this.group.add(entry.strip.object3d);
    }
    entry.strip.object3d.visible = true;
    return entry.strip;
  }

  /**
   * Valori diversi dai pesi sulle matrici VERE della scena (es. lm_head dopo un passo di training), con
   * la scala dei colori dei pesi: i numeri nelle celle cambiano, la targhetta dello stage lo dice.
   * @param {Map<string, Float32Array>} map nome → valori (stessa forma)
   */
  showValues(map, label) {
    this.scene.showValues(map, { colormap: "pesi", label });
    this.valueNames = [...map.keys()];
    this.tab.state.setMode("valori", { label });
  }

  /** I valori del passo prima tornano ai pesi (all'inizio di ogni passo e a fine azione). */
  clearValues() {
    if (!this.valueNames) return;
    this.scene.showValues(new Map(this.valueNames.map((n) => [n, null])));
    this.valueNames = null;
    this.tab.state.setMode("pesi");
  }

  /** Velo sulla matrice `name` della scena tranne le celle indicate (S.highlightCells). */
  highlight(name, cells, options) {
    if (!this.scene.matrixOf(name)) return false;
    const ok = this.scene.highlightCells(name, cells, options);
    if (ok) this.highlighted.add(name);
    return ok;
  }

  clearHighlights() {
    for (const name of this.highlighted) this.scene.highlightCells(name, null);
    this.highlighted.clear();
  }

  /** Linee luminose tra punti del mondo (es. dalla fine di una riga di W alla cella del risultato). */
  link(segments, { color = 0xfde047 } = {}) {
    if (this.links) {
      this.links.removeFromParent();
      this.links.geometry.dispose();
      this.links.material.dispose();
      this.links = null;
    }
    if (!segments.length) return;
    const pos = new Float32Array(segments.length * 6);
    const zOf = (p) => p.z ?? zAbove(this.scene.layout?.boxes, { x0: p.x, x1: p.x, y0: p.y, y1: p.y });
    segments.forEach(([a, b], k) => pos.set([a.x, a.y, zOf(a), b.x, b.y, zOf(b)], k * 6));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, depthWrite: false, toneMapped: false }));
    lines.raycast = () => {};
    lines.userData.noBounds = true;
    lines.renderOrder = 14;
    this.links = lines;
    this.group.add(lines);
  }

  clearLinks() {
    if (this.links) {
      this.links.removeFromParent();
      this.links.geometry.dispose();
      this.links.material.dispose();
      this.links = null;
    }
    if (this.outlines) {
      this.outlines.removeFromParent();
      this.outlines.geometry.dispose();
      this.outlines.material.dispose();
      this.outlines = null;
    }
  }

  /** Riquadri luminosi attorno a rettangoli del mondo (es. la riga γ della scena, il nodo della ReLU). */
  outline(rects, { color = 0xfde047, pad = 0.02, z = null } = {}) {
    const list = (Array.isArray(rects) ? rects : [rects]).filter(Boolean);
    z ??= Math.max(...list.map((r) => zAbove(this.scene.layout?.boxes, r, { lift: 0.004 })), 0);
    if (this.outlines) {
      this.outlines.removeFromParent();
      this.outlines.geometry.dispose();
      this.outlines.material.dispose();
      this.outlines = null;
    }
    if (!list.length) return;
    const pos = new Float32Array(list.length * 24);
    list.forEach((r, k) => {
      const [x0, x1, y0, y1] = [r.x0 - pad, r.x1 + pad, r.y0 - pad, r.y1 + pad];
      pos.set([x0, y0, z, x1, y0, z, x1, y0, z, x1, y1, z, x1, y1, z, x0, y1, z, x0, y1, z, x0, y0, z], k * 24);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 1, depthWrite: false, toneMapped: false }));
    lines.raycast = () => {};
    lines.userData.noBounds = true;
    lines.renderOrder = 14;
    this.outlines = lines;
    this.group.add(lines);
  }

  /**
   * Etichette delle carte a grandezza costante sullo schermo (LABEL_PX): si ricalcolano solo se la camera
   * o l'altezza dello stage sono cambiate (o se è comparsa una carta nuova). A camera ferma: un confronto.
   */
  updateLabels() {
    const cam = this.ui.host.camera;
    const H = this.ui.host.container.clientHeight;
    const e = cam.matrixWorld.elements;
    const last = (this._labelPose ??= new Float64Array(8).fill(NaN));
    if (!this._labelsDirty && last[0] === e[12] && last[1] === e[13] && last[2] === e[14] && last[3] === e[8] && last[4] === e[9] && last[5] === e[10] && last[6] === H && last[7] === cam.fov) return;
    last.set([e[12], e[13], e[14], e[8], e[9], e[10], H, cam.fov]);
    this._labelsDirty = false;
    const tanHalf = Math.tan((cam.fov * Math.PI) / 360);
    cam.getWorldDirection(_dir);
    for (const card of this.cards.values()) {
      if (!card.visible || !card.label) continue;
      _p.set((card.rect.x0 + card.rect.x1) / 2, (card.rect.y0 + card.rect.y1) / 2, card.z).sub(cam.position);
      const d = Math.max(0.02, _p.dot(_dir));
      card.scaleLabel(Math.min(2, Math.max(0.003, (LABEL_PX * 2 * d * tanHalf) / Math.max(1, H))));
    }
  }

  /** Il codice vero (snippet di data/model.json) con le righe evidenziate. */
  code(snippetKey, title, highlight = []) {
    const panel = this.ui.codePanel;
    const snippet = this.tab.spec.snippets?.[snippetKey];
    if (!panel || !snippet) return;
    panel.show({ title, file: snippet.file, startLine: snippet.start_line, code: snippet.code, language: "python", highlight });
    // sul telefono il codice sta sopra la scena: ridotto (resta l'intestazione con la riga), come nel Transformer
    if (narrow()) panel.root.classList.add("minimized");
  }

  // ---------------------------------------------------------------- camera
  /** La parte dello stage davvero libera, in pixel: { width, height, top, bottom }. */
  freeArea() {
    const { host, stage, root, player, codePanel } = this.ui;
    const box = stage.getBoundingClientRect();
    const width = Math.max(160, stage.clientWidth - host.coveredRight());
    const height = Math.max(160, stage.clientHeight);
    let top = 56;
    for (const element of root.querySelectorAll(TOP_OVERLAYS)) {
      const r = element.getBoundingClientRect();
      if (r.height && r.width && r.top - box.top < 150) top = Math.max(top, r.bottom - box.top + 8);
    }
    let bottom = 12;
    const cover = (element) => {
      if (!element || element.hidden) return;
      const r = element.getBoundingClientRect();
      if (r.height) bottom = Math.max(bottom, box.bottom - r.top + 10);
    };
    cover(player?.bar);
    cover(this.board.root);
    if (narrow() && codePanel && !codePanel.root.hidden) {
      const r = codePanel.root.getBoundingClientRect();
      if (r.top - box.top < height / 2) top = Math.max(top, r.bottom - box.top + 8);
      else cover(codePanel.root);
    }
    return { width, height, top, bottom };
  }

  /**
   * Inquadra un rettangolo del mondo nella parte libera dello stage (sopra lavagna e player, sotto i
   * comandi). Con la camera libera (lo studente si è mosso) il director ricorda il volo per "Segui l'azione".
   * Sul telefono il margine è più stretto: la parte libera è bassa e ogni pixel conta per i numeri nelle
   * celle (il nucleo li scrive da ~36 px di larghezza con due decimali).
   */
  frame(rect, { margin = narrow() ? 1.04 : 1.12, duration = 1.0, z = null } = {}) {
    if (!rect) return Promise.resolve();
    z ??= this.contentZ(rect);
    this.lastFrame = { rect, margin, z };
    const area = this.freeArea();
    const reserve = this.labelReserve(rect);
    const pose = framePose({ rect, z, fovDeg: this.ui.host.camera.fov, width: area.width, height: area.height, top: area.top + reserve.top, bottom: area.bottom + reserve.bottom, leftPx: reserve.left, margin });
    const target = new THREE.Vector3(pose.target.x, pose.target.y, pose.target.z);
    const position = target.clone().addScaledVector(FRONT, pose.distance);
    this.lastPose = { ...pose, area };
    return this.ui.director.flyTo(position, target, this.reduced ? 0.3 : duration);
  }

  /**
   * Inquadra al frame dopo: il player scrive la didascalia del passo DOPO enter(), e la sua altezza
   * (come quella della lavagna) decide la parte libera dello stage. Un passo più recente annulla questo.
   */
  frameSoon(rect, options = {}) {
    const token = ++this._frameToken;
    cancelAnimationFrame(this._frameRaf);
    this._frameRaf = requestAnimationFrame(() => {
      if (token !== this._frameToken || !this.active) return;
      this.syncLayout();
      void this.frame(rect, options);
    });
  }

  /**
   * La disposizione dei pannelli nelle variabili CSS (lab-actions.css): altezza della barra del player,
   * posizione e larghezza della lavagna, altezza massima del codice. A eventi (passi, ridimensionamenti, pad),
   * mai per frame.
   */
  syncLayout() {
    const { root, stage, codePanel } = this.ui;
    const bar = this.ui.player?.bar;
    const barOn = Boolean(bar && !bar.hidden);
    if (barOn) root.style.setProperty("--lab-act-player-h", `${Math.round(bar.offsetHeight)}px`);
    let codeMax = null;
    if (this.board.visible && !narrow()) {
      // schermo largo: lavagna allineata alla barra e finita prima del pad, codice sopra il posto del pad
      // nell'angolo e sopra la lavagna (deskLayout in plan.js: niente inseguimenti fra i tre)
      const s = stage.getBoundingClientRect();
      const rel = (r) => ({ left: r.left - s.left, right: r.right - s.left, top: r.top - s.top, bottom: r.bottom - s.top });
      const shown = (e) => Boolean(e && !e.hidden && e.offsetHeight > 0);
      const barRect = barOn ? rel(bar.getBoundingClientRect()) : null;
      const chip = root.querySelector(".nav-follow");
      const band = barRect && shown(chip) ? { ...barRect, top: Math.min(barRect.top, rel(chip.getBoundingClientRect()).top) } : barRect;
      const charts = root.querySelector(".charts-panel.open");
      const code = !charts && shown(codePanel?.root) ? rel(codePanel.root.getBoundingClientRect()) : null;
      const pad = stage.querySelector(".nav-pad");
      const padOn = Boolean(pad && pad.offsetWidth);
      // il nucleo piazza il pad con la misura da aperto anche quando è ridotto: la ricordiamo
      if (padOn && !pad.classList.contains("collapsed") && !pad.classList.contains("squeezed")) this._padSize = { w: pad.offsetWidth, h: pad.offsetHeight };
      const L = deskLayout({
        sw: stage.clientWidth,
        sh: stage.clientHeight,
        bar: barRect,
        band,
        chartsLeft: charts ? rel(charts.getBoundingClientRect()).left : null,
        code,
        pad: padOn ? rel(pad.getBoundingClientRect()) : null,
        padSize: this._padSize,
        padBottomFn: padBottom,
      });
      root.style.setProperty("--lab-act-left-px", `${L.left}px`);
      root.style.setProperty("--lab-act-max-w", `${L.maxW}px`);
      // la lavagna ha ora la sua larghezza (e quindi la sua altezza): il codice finisce sopra di lei
      codeMax = codeAboveBoard(L.codeMax, L.fallback ? null : code, rel(this.board.root.getBoundingClientRect()));
    } else {
      root.style.removeProperty("--lab-act-left-px");
      root.style.setProperty("--lab-act-max-w", "720px");
    }
    if (codeMax === null) root.style.removeProperty("--lab-act-code-max");
    else root.style.setProperty("--lab-act-code-max", `${codeMax}px`);
    root.style.setProperty("--lab-act-board-h", this.board.visible ? `${Math.round(this.board.root.offsetHeight)}px` : "0px");
  }

  /**
   * I pixel da lasciare liberi attorno a un rettangolo per le etichette delle carte che vi entrano (hanno
   * una grandezza fissa sullo schermo): a sinistra la più larga, sopra e sotto una riga di testo.
   */
  labelReserve(rect) {
    const out = { left: 0, top: 0, bottom: 0 };
    for (const card of this.cards.values()) {
      const b = card.box;
      if (!card.visible || !card.label || b.x1 < rect.x0 || b.x0 > rect.x1 || b.y1 < rect.y0 || b.y0 > rect.y1) continue;
      const px = card.labelReservePx();
      if (card.labelSide === "left") out.left = Math.max(out.left, px);
      else if (card.labelSide === "top") out.top = Math.max(out.top, px);
      else if (card.labelSide === "bottom") out.bottom = Math.max(out.bottom, px);
    }
    return out;
  }

  /** Il piano da inquadrare: la z media delle carte visibili nel rettangolo (o quella della scena lì sotto). */
  contentZ(rect) {
    let sum = 0;
    let n = 0;
    for (const card of this.cards.values()) {
      const b = card.box;
      if (!card.visible || b.x1 < rect.x0 || b.x0 > rect.x1 || b.y1 < rect.y0 || b.y0 > rect.y1) continue;
      sum += card.z;
      n++;
    }
    return n ? sum / n : zAbove(this.scene.layout?.boxes, rect, { lift: 0 });
  }

  /** Dopo un ridimensionamento (o la lavagna ridotta) la stessa inquadratura, ricalcolata. */
  reframe() {
    if (!this.active) return;
    this.syncLayout();
    if (!this.lastFrame || this.ui.director.actionActive === false) return;
    if (!this.ui.director.follow) return; // camera libera: lo studente guarda dove vuole
    const { rect, margin, z } = this.lastFrame;
    void this.frame(rect, { margin, z, duration: 0.35 });
  }

  // ---------------------------------------------------------------- misure
  /** Registra il costo di un enter (ms) per le prove. */
  noteEnter(id, index, ms) {
    this.stats.enters.push({ id, index, ms });
    if (this.stats.enters.length > 200) this.stats.enters.shift();
  }

  noteUpdate(ms) {
    this.stats.updates++;
    this.stats.updateMs += ms;
    this.stats.maxUpdateMs = Math.max(this.stats.maxUpdateMs, ms);
  }

  dispose() {
    this.end({ flyBack: false });
    this.numbers.dispose();
    this.board.dispose();
  }
}
