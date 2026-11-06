/**
 * Navigazione libera nella scena 3D di un tab: come muoversi in un plastico che hai sul tavolo.
 *
 *   mouse     trascina = ruota attorno al punto guardato; tasto destro, centrale o Shift/Ctrl/⌘ +
 *             trascina = sposta; rotella = zoom verso il puntatore; doppio clic = centra quel punto;
 *             trackpad: due dita in orizzontale (o Shift + rotella) = sposta a destra/sinistra
 *   tastiera  W A S D e ← → spostano, Q E girano attorno, R F inclinano, + − zoomano (movimento
 *             continuo con accelerazione e frenata morbide), Shift = veloce, Alt = preciso,
 *             C o Home = vista iniziale, M = Ruota/Sposta, ? o H = aiuto
 *   touch     un dito = ruota (o sposta in modalità Sposta), due dita = zoom e sposta, doppio tocco = centra
 *
 * Questo modulo è la logica (nessun elemento DOM creato qui): i comandi a schermo e l'aiuto stanno
 * in js/core/ui/navPad.js, le regole numeriche (velocità, frenata, bordi…) in js/core/navMath.js.
 * Ogni tab ne ha una (ui.nav, creata da createTabLayout); i tasti agiscono solo sul tab visibile.
 *
 *   ui.nav.setHome(() => tab.focus("overview"));   // cosa fa "Centra" (predefinito: la vista iniziale)
 *   ui.nav.setBounds(box3 | () => box3);           // confini (predefinito: i mesh visibili, allargati)
 *   ui.nav.setDragMode("rotate" | "pan");  ui.nav.home();  ui.nav.setEnabled(false);
 *   ui.nav.setOrbitLimits({ turn: 60, tilt: 45, minDistance: 6 }); // scene "a parete" (diagrammi piatti)
 *   ui.nav.on("input", (source) => …)              // lo studente si è mosso: "pointer", "keys", "pad", "dblclick", "home"
 *
 * Con un'azione del player in corso, muoversi rende la camera "libera" (vedi CameraDirector.release).
 * A riposo il costo per frame è un paio di confronti: niente allocazioni, niente calcoli.
 */
import * as THREE from "three";
import { meshBounds } from "./director.js";
import { AXES, SPEED, boundsMargin, classifyKey, dollyDistance, excessToBox, expandBox, isDoubleTap, isLost, maxUserDistance, resistMove, speedMultiplier, stepVelocity } from "./navMath.js";

const PREF_MODE = "nav.dragMode";
const DRAG_THRESHOLD = 4; // px: sotto è un clic, sopra un trascinamento
const MIN_TAP_SECONDS = 0.18; // un tocco breve su un tasto o un pulsante sposta comunque di un passo percepibile
const BOUNDS_MAX_AGE_MS = 3000; // i confini si ricalcolano al più ogni 3 s (la scena può cambiare)
const LOST_CHECK_MS = 250;
const LOST_DELAY_MS = 700; // il suggerimento "Ti sei perso?" aspetta un attimo: niente lampeggi
const SPRING_RATE = 6; // quanto in fretta il bordo riporta dentro (1/s)
const CENTER_APPROACH = 0.72; // doppio clic: ci si avvicina al punto del 28%
const CENTER_FLIGHT = 0.8;
const HOME_FLIGHT = 1.1;
// Confini: il punto guardato può uscire dal contenuto solo di poco: il 12% della diagonale della scena,
// ma mai più di mezza inquadratura (da vicino, al livello Dettaglio, il 12% di una scena grande sarebbe
// già il vuoto). Così anche al bordo metà dello schermo mostra ancora la scena. Vedi navMath.boundsMargin.
const BOUNDS_MARGIN = { fraction: 0.12, view: 0.5, minimum: 0.5 };
const MARGIN_REFRESH = 0.1; // il margine si ricalcola quando la distanza cambia di più del 10%
// Zoom indietro massimo: 6 raggi della scena, o 1,5 volte la vista iniziale se è già più lontana (telefono).
// I voli dei tab possono andare oltre (vedi allowFlightDistance).
const MAX_DISTANCE = { radii: 6, home: 1.5 };
// Rotella orizzontale (trackpad a due dita, Shift + rotella): pixel di scorrimento → spostamento sullo
// schermo 1:1, smorzato in ~0,08 s perché i salti da 100 px di una rotella vera non diano scatti.
const WHEEL_PAN_TAU = 0.08;
const WHEEL_LINE_PX = 16;
// Zoom: una tacca di rotella avvicina del ~7%; con due dita lo zoom segue le dita 1:1 (pizzico naturale).
const WHEEL_ZOOM_SPEED = 1.4;
const PINCH_ZOOM_SPEED = 1;

/** Preferenze condivise tra i tab: cambiare modalità in un tab la cambia anche negli altri. */
const shared = new EventTarget();

export function readPref(key, fallback) {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback; // navigazione privata o archiviazione bloccata: si usa il valore predefinito
  }
}

export function writePref(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // non poterlo ricordare non è un errore
  }
}

const TEXT_FIELDS = "textarea, select, [contenteditable=''], [contenteditable='true'], input:not([type]), input[type='text'], input[type='search'], input[type='number'], input[type='email'], input[type='url'], input[type='password'], input[type='tel']";
/** Tasti che i controlli non testuali (cursori, caselle, pulsanti radio…) usano già da soli. */
const CONTROL_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

/**
 * true se il tasto appartiene a un campo o a un controllo invece che alla scena:
 * - in un campo di testo (prompt, domanda, tendina) ogni tasto è testo;
 * - su un cursore o una casella le frecce e Home lo regolano, ma le lettere (W A S D…) restano alla
 *   scena: dopo aver mosso il cursore della guidance, D sposta ancora la vista;
 * - dentro una finestra di dialogo che non è l'aiuto della navigazione (es. la ricerca) niente scena.
 */
function keyBelongsToControl(e) {
  const target = e.target;
  if (!target?.closest) return false;
  if (target.isContentEditable || target.closest(TEXT_FIELDS)) return true;
  const dialog = target.closest("dialog, [role='dialog'], [aria-modal='true']");
  if (dialog && !dialog.closest(".nav-help")) return true;
  return CONTROL_KEYS.has(e.key) && Boolean(target.closest("input, [role='slider'], [role='radiogroup'], [role='tablist'], [role='listbox'], [role='menu']"));
}

export class Navigation {
  /**
   * @param {import("./sceneHost.js").SceneHost} host
   * @param {import("./director.js").CameraDirector} director
   * @param {object} o
   * @param {HTMLElement} o.root radice del tab (per sapere se è visibile e per i clic sui menu)
   * @param {import("./ui/player.js").ActionPlayer | null} [o.player]
   */
  constructor(host, director, { root, player = null }) {
    this.host = host;
    this.director = director;
    this.root = root;
    this.player = player;
    this.camera = host.camera;
    this.controls = host.controls;
    this.canvas = host.renderer.domElement;
    this.enabled = true;
    this.listeners = new Map();

    // stato del movimento continuo
    this.velocity = Object.fromEntries(AXES.map((a) => [a, 0]));
    this.wanted = Object.fromEntries(AXES.map((a) => [a, 0]));
    this.keys = new Map(); // e.code → { axis, dir, since }
    this.holds = new Set(); // pulsanti del pad tenuti premuti: { axis, dir }
    this.mods = { fast: false, precise: false };
    this.moving = false;
    this.wheelPanPx = 0; // scorrimento orizzontale ancora da applicare (px)
    this.modalOpen = false; // l'aiuto è aperto: solo ?, H ed Esc arrivano alla navigazione
    this.userMaxDistance = Infinity;
    this.engaged = false; // dopo un input dello studente: i confini "elastici" sono attivi
    this.pointerActive = false;
    this.lastUpdate = undefined;

    // confini e "ti sei perso?"
    this.boundsSource = null;
    this.contentBox = new THREE.Box3();
    this.baseBox = new THREE.Box3(); // contenuto + punto guardato della vista iniziale (prima del margine)
    this.boundsDiagonal = 0;
    this.marginDistance = 0; // distanza della camera a cui è stato calcolato il margine
    this.navBounds = { min: new THREE.Vector3(), max: new THREE.Vector3() };
    this.boundsAt = -Infinity;
    this.cameraDirty = false;
    this.lostCheckedAt = 0;
    this.lostSince = 0;
    this.lost = false;

    // vista iniziale ("Centra")
    this.homeFn = null;
    this.homePose = null;
    this.framesSinceStart = 0;

    // vettori riusati nel ciclo (niente allocazioni per frame)
    this.v = { offset: new THREE.Vector3(), right: new THREE.Vector3(), up: new THREE.Vector3(), move: new THREE.Vector3(), excess: new THREE.Vector3() };
    this.spherical = new THREE.Spherical();
    this.frustum = new THREE.Frustum();
    this.projScreen = new THREE.Matrix4();
    this.sphere = new THREE.Sphere();
    this.meshSphere = new THREE.Sphere(); // per anyVisibleInFrustum (this.sphere serve a checkLost nello stesso momento)

    this.setDragMode(readPref(PREF_MODE, "rotate"), { persist: false, broadcast: false });
    this.bind();
  }

  // ---------------------------------------------------------------- API pubblica
  /** Cosa fa "Centra" (tasto C/Home e pulsante): di solito il tab.focus("overview") del tab. */
  setHome(fn) {
    this.homeFn = typeof fn === "function" ? fn : null;
  }

  /** Confini della navigazione: un THREE.Box3 o una funzione che lo restituisce (null = automatici). */
  setBounds(source) {
    this.boundsSource = source ?? null;
    this.boundsAt = -Infinity;
  }

  /**
   * Limiti per le scene "a parete" (diagrammi piatti guardati da davanti): da dietro il testo sarebbe a
   * specchio, da sopra si vedrebbe solo una linea, troppo vicino si entra nei blocchi. Angoli in gradi
   * rispetto alla vista frontale (camera sull'asse +z); valgono per mouse, tasti, pad e touch.
   *   ui.nav.setOrbitLimits({ turn: 60, tilt: 45, minDistance: 6 });
   */
  setOrbitLimits({ turn = Infinity, tilt = 90, minDistance } = {}) {
    const deg = THREE.MathUtils.degToRad;
    const c = this.controls;
    c.minAzimuthAngle = Number.isFinite(turn) ? -deg(turn) : -Infinity;
    c.maxAzimuthAngle = Number.isFinite(turn) ? deg(turn) : Infinity;
    c.minPolarAngle = deg(Math.max(0, 90 - tilt));
    c.maxPolarAngle = deg(Math.min(180, 90 + tilt));
    if (minDistance !== undefined) c.minDistance = minDistance;
  }

  /** "rotate": il trascinamento sinistro ruota; "pan": sposta (comodo con il solo trackpad). */
  setDragMode(mode, { persist = true, broadcast = true } = {}) {
    mode = mode === "pan" ? "pan" : "rotate";
    this.mode = mode;
    this.controls.mouseButtons.LEFT = mode === "pan" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    this.controls.touches.ONE = mode === "pan" ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;
    this.canvas.dataset.navMode = mode;
    if (persist) writePref(PREF_MODE, mode);
    if (broadcast) shared.dispatchEvent(new CustomEvent("mode", { detail: { mode, from: this } }));
    this.emit("mode", mode);
  }

  toggleDragMode() {
    this.setDragMode(this.mode === "pan" ? "rotate" : "pan");
  }

  /** Torna alla vista iniziale del tab con un volo (anche durante un'azione: è una richiesta dello studente). */
  home() {
    this.noteInput("home");
    this.stopMotion();
    this.director.noteUserIntent();
    if (this.homeFn) return Promise.resolve(this.homeFn());
    if (this.homePose) return this.director.flyTo(this.homePose.position, this.homePose.target, HOME_FLIGHT);
    const box = this.computeContentBox();
    return box.isEmpty() ? Promise.resolve() : this.director.focus(box, { showFrame: false, margin: 1.15 });
  }

  /** Accende o spegne tutta la navigazione dello studente (mouse, tastiera, pad). */
  setEnabled(enabled) {
    this.enabled = Boolean(enabled);
    this.controls.enabled = this.enabled;
    if (!this.enabled) this.stopMotion();
    this.emit("enabled", this.enabled);
  }

  /** Tiene premuto un movimento (pulsante del pad): restituisce la funzione per rilasciarlo. */
  hold(axis, dir) {
    if (!this.enabled) return () => {};
    const token = { axis, dir };
    const since = performance.now();
    this.holds.add(token);
    this.moving = true;
    this.noteInput("pad");
    return () => {
      if (!this.holds.has(token)) return;
      // un tocco brevissimo diventa un piccolo passo, non un movimento impercettibile
      const remaining = MIN_TAP_SECONDS - (performance.now() - since) / 1000;
      if (remaining > 0) setTimeout(() => this.holds.delete(token), remaining * 1000);
      else this.holds.delete(token);
    };
  }

  /** Un passo breve in una direzione (pulsante del pad attivato da tastiera). */
  nudge(axis, dir) {
    this.hold(axis, dir)();
  }

  /** fn(detail) per "input", "mode", "lost", "help", "escape", "enabled", "dispose". Restituisce la rimozione. */
  on(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(fn);
    return () => this.listeners.get(type)?.delete(fn);
  }

  /** true se il tab è quello visibile e la navigazione è attiva. */
  isActive() {
    return this.enabled && this.host.running && !this.root.hidden && this.root.isConnected;
  }

  dispose() {
    this.unbind();
    this.emit("dispose");
    this.listeners.clear();
  }

  // ---------------------------------------------------------------- eventi
  bind() {
    const on = (target, type, fn, options) => {
      target.addEventListener(type, fn, options);
      this.disposers.push(() => target.removeEventListener(type, fn, options));
    };
    this.disposers = [];
    // Un clic sulla scena (per scegliere un neurone, una regione…) non è navigazione: lo diventa solo
    // quando il puntatore si muove davvero (DRAG_THRESHOLD px) o con la rotella.
    on(this.controls, "start", () => (this.pointerActive = true));
    on(this.controls, "end", () => {
      this.pointerActive = false;
      this.dragStart = null;
    });
    on(this.canvas, "pointermove", (e) => this.onCanvasMove(e));
    on(this.controls, "change", () => (this.cameraDirty = true));
    on(window, "keydown", (e) => this.onKeyDown(e));
    on(window, "keyup", (e) => this.onKeyUp(e));
    on(window, "blur", () => this.stopMotion());
    on(document, "visibilitychange", () => document.hidden && this.stopMotion());
    on(this.canvas, "pointerdown", (e) => this.onCanvasDown(e));
    on(this.canvas, "pointerup", (e) => this.onCanvasUp(e));
    on(this.canvas, "pointercancel", () => this.onCanvasCancel());
    on(this.canvas, "dblclick", (e) => this.centerOn(e));
    on(this.root, "wheel", (e) => this.onWheel(e), { capture: true, passive: false });
    // Un clic su un menu o un pulsante del tab (non sul pad né sulla barra del player) è una richiesta
    // dello studente: il volo che ne segue parte anche con la camera libera. Fase di cattura: arriva
    // prima dei gestori del tab, che chiedono il volo.
    on(this.root, "pointerdown", (e) => this.onRootDown(e), true);
    on(this.root, "pointerup", (e) => this.onRootUp(e), true);
    on(this.root, "keydown", (e) => this.onRootKey(e), true);
    on(shared, "mode", (e) => {
      if (e.detail.from !== this && e.detail.mode !== this.mode) this.setDragMode(e.detail.mode, { persist: false, broadcast: false });
    });
    this.disposers.push(this.host.onFrame((dt) => this.update(dt)));
    // Tab nascosto (il ciclo si ferma): un tasto tenuto premuto o la frenata in corso non devono far
    // scivolare la scena al ritorno.
    this.disposers.push(this.host.onStop(() => this.stopMotion()));
  }

  unbind() {
    for (const off of this.disposers ?? []) off();
    this.disposers = [];
    this.stopMotion();
  }

  emit(type, detail) {
    for (const fn of this.listeners.get(type) ?? []) fn(detail);
  }

  /** Qualsiasi input di navigazione: annulla il volo in corso, libera la camera durante un'azione. */
  noteInput(source) {
    if (!this.homePose) this.captureHome();
    this.director.cancel();
    this.director.release();
    this.refreshBounds();
    // Lo zoom indietro si ferma al limite, ma senza mai "tirare" la camera se un volo l'ha portata più in là.
    this.controls.maxDistance = Math.max(this.userMaxDistance, this.host.zoomDistance());
    this.engaged = true;
    this.emit("input", source);
  }

  onKeyDown(e) {
    this.mods.fast = e.shiftKey;
    this.mods.precise = e.altKey;
    if (!this.isActive() || e.defaultPrevented || keyBelongsToControl(e)) return;
    const k = classifyKey(e);
    if (!k) return;
    // Con l'aiuto aperto (finestra modale) la scena dietro non si muove: si può solo chiuderlo.
    if (this.modalOpen && !(k.type === "command" && (k.name === "help" || k.name === "escape"))) return;
    if (k.type === "move") {
      e.preventDefault();
      // Sulla tastiera americana "+" e "_" si scrivono con Shift: lì Shift non vuol dire "più veloce".
      if (e.shiftKey && (e.key === "+" || e.key === "_")) this.mods.fast = false;
      const id = e.code || k.id;
      if (!this.keys.has(id)) {
        this.keys.set(id, { axis: k.axis, dir: k.dir, since: performance.now() });
        this.moving = true;
        this.noteInput("keys");
      }
      return;
    }
    if (e.repeat) return;
    if (k.name === "home") {
      e.preventDefault();
      this.home();
    } else if (k.name === "mode") this.toggleDragMode();
    else if (k.name === "help") {
      e.preventDefault();
      this.emit("help");
    } else if (k.name === "escape") this.emit("escape", e);
  }

  onKeyUp(e) {
    this.mods.fast = e.shiftKey;
    this.mods.precise = e.altKey;
    const id = e.code || classifyKey(e)?.id;
    const k = this.keys.get(id);
    if (!k) return;
    const remaining = MIN_TAP_SECONDS - (performance.now() - k.since) / 1000;
    if (remaining > 0) setTimeout(() => this.keys.get(id) === k && this.keys.delete(id), remaining * 1000);
    else this.keys.delete(id);
  }

  /** Ferma tutto (tasti bloccati quando la finestra perde il focus, tab nascosto, volo verso "Centra"). */
  stopMotion() {
    this.keys.clear();
    this.holds.clear();
    for (const a of AXES) this.velocity[a] = 0;
    this.moving = false;
    this.wheelPanPx = 0;
  }

  onWheel(e) {
    if (e.target !== this.canvas || !this.enabled) return;
    this.controls.zoomSpeed = WHEEL_ZOOM_SPEED;
    // Scorrimento orizzontale (due dita sul trackpad, Shift + rotella): sposta a destra/sinistra invece
    // di fare zoom. Il pizzico del trackpad arriva con ctrlKey: resta zoom (OrbitControls).
    const horizontal = !e.ctrlKey && !e.metaKey && (Math.abs(e.deltaX) > Math.abs(e.deltaY) || (e.shiftKey && e.deltaY !== 0));
    if (horizontal) {
      e.preventDefault(); // niente "pagina indietro" col gesto orizzontale del trackpad
      e.stopPropagation(); // le OrbitControls (sul canvas) non lo trasformano in zoom
      const unit = e.deltaMode === 1 ? WHEEL_LINE_PX : e.deltaMode === 2 ? this.canvas.clientWidth : 1;
      this.wheelPanPx += (Math.abs(e.deltaX) >= Math.abs(e.deltaY) ? e.deltaX : e.deltaY) * unit;
    }
    this.noteInput("pointer");
  }

  onCanvasMove(e) {
    const start = this.dragStart;
    if (!start || !this.pointerActive || Math.hypot(e.clientX - start.x, e.clientY - start.y) < DRAG_THRESHOLD) return;
    this.dragStart = null; // una volta per trascinamento
    this.noteInput("pointer");
  }

  onCanvasDown(e) {
    this.dragStart = { x: e.clientX, y: e.clientY };
    if (e.pointerType === "touch") this.controls.zoomSpeed = PINCH_ZOOM_SPEED;
    else this.canvas.dataset.navDrag = this.dragKind(e);
    this.touches = e.pointerType === "touch" ? (this.touches ?? 0) + 1 : 0;
    this.press = { x: e.clientX, y: e.clientY, t: performance.now(), multi: this.touches > 1 };
    if (this.touches > 1 && this.firstPress) this.firstPress.multi = true;
    if (this.touches === 1) this.firstPress = this.press;
  }

  onCanvasUp(e) {
    delete this.canvas.dataset.navDrag;
    if (e.pointerType !== "touch") return;
    this.touches = Math.max(0, (this.touches ?? 1) - 1);
    // Doppio tocco (sul telefono non c'è il doppio clic): due tocchi brevi e fermi, vicini.
    const p = this.firstPress;
    if (!p || p.multi || this.touches > 0) return;
    const tap = { t: performance.now(), x: e.clientX, y: e.clientY };
    const still = Math.hypot(tap.x - p.x, tap.y - p.y) < 10 && tap.t - p.t < 350;
    if (!still) {
      this.lastTap = null;
      return;
    }
    if (isDoubleTap(this.lastTap, tap)) {
      this.lastTap = null;
      this.centerOn(e);
    } else this.lastTap = tap;
  }

  onCanvasCancel() {
    delete this.canvas.dataset.navDrag;
    this.touches = 0;
    this.lastTap = null;
  }

  /** Che cosa farà questo trascinamento (per il cursore): stessa regola delle OrbitControls. */
  dragKind(e) {
    if (e.button === 1 || e.button === 2) return "pan";
    const swap = e.ctrlKey || e.metaKey || e.shiftKey;
    const pan = this.mode === "pan" ? !swap : swap;
    return pan ? "pan" : "rotate";
  }

  onRootDown(e) {
    this.rootPress = e.target.closest?.(".nav-pad, .nav-follow, .player-bar") ? null : { x: e.clientX, y: e.clientY };
  }

  onRootUp(e) {
    const p = this.rootPress;
    this.rootPress = null;
    if (p && Math.hypot(e.clientX - p.x, e.clientY - p.y) < 6) this.director.noteUserIntent();
  }

  onRootKey(e) {
    if (classifyKey(e) || e.target.closest?.(".player-bar")) return;
    this.director.noteUserIntent();
  }

  // ---------------------------------------------------------------- doppio clic: centra il punto
  /**
   * Doppio clic / doppio tocco: il punto va al centro. Su un oggetto ci si avvicina anche (del 28%);
   * sul vuoto no: ripetuto, porterebbe la camera sempre più dentro il nulla tra una parte e l'altra
   * della scena. Il punto resta comunque dentro i confini.
   */
  centerOn(e) {
    if (!this.enabled) return;
    const now = performance.now();
    if (now - (this.lastCenterAt ?? 0) < 450) return; // doppio tocco già gestito (alcuni browser mandano anche dblclick)
    this.lastCenterAt = now;
    const pick = this.pickPoint(e);
    if (!pick) return;
    const { point, hit } = pick;
    this.noteInput("dblclick");
    if (this.hasBounds()) point.clamp(this.navBounds.min, this.navBounds.max);
    this.stopMotion();
    this.director.noteUserIntent();
    const offset = this.v.offset.subVectors(this.camera.position, this.controls.target);
    const current = offset.length();
    offset.normalize();
    const distance = hit ? Math.max(this.controls.minDistance, 1.5, this.camera.position.distanceTo(point) * CENTER_APPROACH) : current;
    const position = point.clone().addScaledVector(offset, distance);
    this.director.flyTo(position, point, CENTER_FLIGHT);
  }

  /**
   * Il punto della scena sotto il puntatore: il primo oggetto visibile colpito (salta i livelli LOD
   * sfumati, le linee e gli oggetti marcati noBounds), con hit = true; se non c'è niente, il punto sul
   * piano che passa per il punto guardato, di fronte alla camera (hit = false).
   */
  pickPoint(e) {
    for (const h of this.host.pick(e, this.host.scene.children)) {
      if (isPickable(h.object)) return { point: h.point.clone(), hit: true };
    }
    const normal = this.camera.getWorldDirection(new THREE.Vector3());
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, this.controls.target);
    const point = this.host.raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    return point ? { point, hit: false } : null;
  }

  // ---------------------------------------------------------------- ciclo per frame
  update(dt) {
    if (this.homePose === null) this.captureHomeWhenSettled();
    if (this.director.flight) this.allowFlightDistance(this.director.flight);
    if (!this.moving && !this.engaged && !this.cameraDirty && !this.lostSince && !this.wheelPanPx) {
      this.lastUpdate = undefined; // a riposo: niente lavoro (e il prossimo movimento riparte da un dt normale)
      return;
    }
    // Tempo reale, come i voli: con frame lenti (rendering software) il movimento non deve rallentare.
    // Il limite di 0,25 s evita un balzo dopo una pausa lunga.
    const now = performance.now();
    const realDt = this.lastUpdate === undefined ? dt : Math.min(0.25, Math.max(dt, (now - this.lastUpdate) / 1000));
    this.lastUpdate = now;
    if (this.moving) this.applyMotion(realDt);
    if (this.wheelPanPx) this.applyWheelPan(realDt);
    if (this.engaged) {
      this.updateMargin();
      this.applyBounds(realDt);
    }
    if (this.cameraDirty || this.lostSince) this.checkLost(now);
  }

  applyMotion(dt) {
    for (const a of AXES) this.wanted[a] = 0;
    for (const k of this.keys.values()) this.wanted[k.axis] += k.dir;
    for (const k of this.holds) this.wanted[k.axis] += k.dir;
    let pushing = false;
    for (const a of AXES) if (this.wanted[a] !== 0) pushing = true;
    if (this.director.flight) {
      // Un volo appena chiesto (menu, Centra, doppio clic) vince sulla frenata residua di un tasto già
      // rilasciato; un tasto o un pulsante ancora premuto invece annulla il volo.
      if (!pushing) {
        this.stopMotion();
        return;
      }
      this.director.cancel();
    }
    const multiplier = speedMultiplier(this.mods);
    let still = true;
    for (const a of AXES) {
      const wanted = Math.max(-1, Math.min(1, this.wanted[a])) * multiplier;
      this.velocity[a] = stepVelocity(this.velocity[a], wanted, dt);
      if (this.velocity[a] !== 0 || wanted !== 0) still = false;
    }
    if (still) {
      this.moving = false;
      return;
    }
    const { camera } = this;
    const target = this.controls.target;
    const { offset } = this.v;
    const vel = this.velocity;

    if (vel.panX || vel.panY) {
      const step = SPEED.pan * this.halfHeight() * dt;
      this.panBy(vel.panX * step, vel.panY * step);
    }
    if (vel.orbit || vel.tilt || vel.zoom) {
      offset.subVectors(camera.position, target);
      this.spherical.setFromVector3(offset);
      this.spherical.theta += vel.orbit * SPEED.orbit * dt;
      const minPhi = Math.max(0.05, this.controls.minPolarAngle);
      const maxPhi = Math.min(Math.PI - 0.05, this.controls.maxPolarAngle);
      this.spherical.phi = Math.min(maxPhi, Math.max(minPhi, this.spherical.phi - vel.tilt * SPEED.tilt * dt));
      if (vel.zoom) {
        // stesso limite della rotella (noteInput lo tiene ≥ della distanza di partenza: niente scatti)
        this.spherical.radius = dollyDistance(this.spherical.radius, vel.zoom * SPEED.zoom, dt, { min: Math.max(0.3, this.controls.minDistance), max: this.controls.maxDistance });
      }
      offset.setFromSpherical(this.spherical);
      camera.position.copy(target).add(offset);
      camera.lookAt(target);
    }
  }

  /** Metà dell'altezza inquadrata alla distanza del punto guardato (unità della scena). */
  halfHeight() {
    return this.camera.position.distanceTo(this.controls.target) * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
  }

  /** Sposta camera e punto guardato insieme nel piano dello schermo: la distanza (e il livello LOD) non cambia. */
  panBy(dx, dy) {
    const { camera } = this;
    const target = this.controls.target;
    const { right, up, move } = this.v;
    right.setFromMatrixColumn(camera.matrixWorld, 0);
    up.setFromMatrixColumn(camera.matrixWorld, 1);
    move.set(0, 0, 0).addScaledVector(right, dx).addScaledVector(up, dy);
    if (this.hasBounds()) resistMove(target, this.navBounds, move, this.softZone);
    camera.position.add(move);
    target.add(move);
  }

  /** Scorrimento orizzontale accumulato dalla rotella: applicato poco a poco, la scena segue le dita. */
  applyWheelPan(dt) {
    if (this.director.flight) {
      this.wheelPanPx = 0;
      return;
    }
    const k = 1 - Math.exp(-dt / WHEEL_PAN_TAU);
    const px = Math.abs(this.wheelPanPx) < 0.5 ? this.wheelPanPx : this.wheelPanPx * k;
    this.wheelPanPx -= px;
    const height = Math.max(1, this.canvas.clientHeight);
    this.panBy((px * 2 * this.halfHeight()) / height, 0);
  }

  /** Bordo "elastico": se il punto guardato è uscito dai confini, la scena lo riporta dentro dolcemente. */
  applyBounds(dt) {
    if (this.director.flight || !this.hasBounds()) {
      if (!this.director.flight) this.engaged = false;
      return;
    }
    const excess = excessToBox(this.controls.target, this.navBounds, this.v.excess);
    if (excess.lengthSq() < 1e-8) {
      if (!this.moving && !this.pointerActive) this.engaged = false;
      return;
    }
    excess.multiplyScalar(1 - Math.exp(-SPRING_RATE * dt));
    this.controls.target.add(excess);
    this.camera.position.add(excess);
  }

  hasBounds() {
    return !this.navBounds.min.equals(this.navBounds.max) && Number.isFinite(this.navBounds.min.x);
  }

  /** Ricalcola i confini se sono vecchi (la scena può essere cambiata: un'altra vista, un livello LOD…). */
  refreshBounds(force = false) {
    const now = performance.now();
    if (!force && now - this.boundsAt < BOUNDS_MAX_AGE_MS) return;
    this.boundsAt = now;
    const box = this.computeContentBox();
    if (box.isEmpty()) {
      this.baseBox.makeEmpty();
      this.navBounds.min.set(0, 0, 0);
      this.navBounds.max.set(0, 0, 0);
      return;
    }
    if (this.homePose) box.expandByPoint(this.homePose.target);
    this.baseBox.copy(box);
    this.boundsRadius = box.getBoundingSphere(this.sphere).radius;
    this.boundsDiagonal = 2 * this.boundsRadius;
    this.updateMargin(true);
    // Zoom indietro limitato: oltre una certa distanza la scena sarebbe un puntino (mai sotto la distanza
    // attuale: la camera non deve fare uno scatto in avanti).
    const homeDistance = this.homePose ? this.homePose.position.distanceTo(this.homePose.target) : 0;
    this.userMaxDistance = maxUserDistance({ radius: this.boundsRadius, homeDistance }, MAX_DISTANCE);
    this.controls.maxDistance = Math.max(this.userMaxDistance, this.host.zoomDistance());
  }

  /**
   * Confini = contenuto allargato di un margine che dipende dallo zoom (vedi BOUNDS_MARGIN): si
   * ricalcola solo quando la distanza cambia di più del 10%, quindi quasi mai mentre ci si sposta.
   */
  updateMargin(force = false) {
    if (this.baseBox.isEmpty()) return;
    const distance = this.host.zoomDistance();
    if (!force && Math.abs(distance - this.marginDistance) <= MARGIN_REFRESH * this.marginDistance) return;
    this.marginDistance = distance;
    const margin = boundsMargin({ diagonal: this.boundsDiagonal, halfHeight: this.halfHeight() }, BOUNDS_MARGIN);
    const expanded = expandBox(this.baseBox, { fraction: 0, minimum: margin });
    this.navBounds.min.copy(expanded.min);
    this.navBounds.max.copy(expanded.max);
    this.softZone = Math.max(0.25, Math.min(this.boundsRadius * 0.25, margin));
  }

  /**
   * Le OrbitControls tagliano la distanza a maxDistance a ogni frame: un volo del tab verso una vista
   * più lontana del limite dello studente (es. "Panoramica") verrebbe deformato. Il limite si allarga
   * per il volo (un confronto per frame, solo mentre si vola) e torna stretto al prossimo input.
   */
  allowFlightDistance(flight) {
    flight.navFar ??= 1.01 * Math.max(flight.toPos.distanceTo(flight.toTarget), flight.fromPos.distanceTo(flight.fromTarget));
    if (this.controls.maxDistance < flight.navFar) this.controls.maxDistance = flight.navFar;
  }

  computeContentBox() {
    const source = typeof this.boundsSource === "function" ? this.boundsSource() : this.boundsSource;
    if (source?.isBox3) return this.contentBox.copy(source);
    return this.contentBox.copy(meshBounds(this.host.scene, { visibleOnly: true }));
  }

  /** "Ti sei perso?": la scena non si vede più (fuori inquadratura o ridotta a un puntino). */
  checkLost(now) {
    if (now - this.lostCheckedAt < LOST_CHECK_MS) return;
    this.lostCheckedAt = now;
    this.cameraDirty = false;
    this.refreshBounds();
    const box = this.contentBox;
    let lost = false;
    if (!box.isEmpty()) {
      this.camera.updateMatrixWorld();
      this.projScreen.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
      this.frustum.setFromProjectionMatrix(this.projScreen);
      box.getBoundingSphere(this.sphere);
      lost = isLost({
        // il box dice "c'è qualcosa da quella parte"; ma una scena non è un rettangolo (es. due file di
        // stazioni): nell'angolo vuoto del box lo schermo è nero, quindi serve un oggetto visibile davvero
        inFrustum: this.frustum.intersectsBox(box) && this.anyVisibleInFrustum(this.host.scene),
        distance: this.camera.position.distanceTo(this.sphere.center),
        radius: this.sphere.radius,
        fovRad: THREE.MathUtils.degToRad(this.camera.fov),
      });
    }
    if (!lost) {
      this.lostSince = 0;
      if (this.lost) this.setLost(false);
      return;
    }
    if (!this.lostSince) this.lostSince = now;
    if (!this.lost && now - this.lostSince >= LOST_DELAY_MS) this.setLost(true);
    if (this.lost) this.lostSince = 0; // già segnalato: si ricontrolla solo quando la camera si muove
  }

  /**
   * true se almeno un oggetto visibile della scena entra nell'inquadratura (sfera di ingombro contro il
   * frustum). Salta i sotto-alberi nascosti, le linee, i livelli LOD sfumati e gli oggetti noBounds;
   * si ferma al primo trovato. Gira solo dentro checkLost (al più ogni 250 ms, dopo un movimento).
   */
  anyVisibleInFrustum(object) {
    if (!object.visible) return false;
    if (object.isMesh && !object.userData.isFrame && !object.userData.noBounds && isPickable(object)) {
      // InstancedMesh: la sfera di tutte le istanze (quella della geometria è di una sola)
      const owner = object.isInstancedMesh ? object : object.geometry;
      if (!owner.boundingSphere) owner.computeBoundingSphere();
      this.meshSphere.copy(owner.boundingSphere).applyMatrix4(object.matrixWorld);
      if (this.frustum.intersectsSphere(this.meshSphere)) return true;
    }
    for (const child of object.children) if (this.anyVisibleInFrustum(child)) return true;
    return false;
  }

  setLost(lost) {
    this.lost = lost;
    this.emit("lost", lost);
  }

  // ---------------------------------------------------------------- vista iniziale
  /** La vista iniziale è quella che il tab ha scelto all'apertura: la si fotografa a volo finito. */
  captureHomeWhenSettled() {
    if (++this.framesSinceStart < 4 || this.director.flight) return;
    this.captureHome();
  }

  captureHome() {
    const f = this.director.flight;
    this.homePose = {
      position: (f ? f.toPos : this.camera.position).clone(),
      target: (f ? f.toTarget : this.controls.target).clone(),
    };
  }
}

/** Un oggetto colpito dal doppio clic conta solo se si vede davvero. */
function isPickable(object) {
  if (object.isLine || object.userData.noBounds) return false;
  for (let o = object; o; o = o.parent) if (!o.visible) return false;
  const materials = Array.isArray(object.material) ? object.material : [object.material];
  return materials.some((m) => m && m.visible !== false && !(m.transparent && m.opacity < 0.05));
}
