/**
 * Matematica pura della navigazione libera (nessun DOM, nessun three.js: si testa con node --test).
 *
 * Qui stanno le regole che decidono "come ci si muove": quale tasto fa cosa, quanto velocemente,
 * come si accelera e si frena, quando il bordo della scena "fa resistenza", quando ci si è persi.
 * js/core/navigation.js le applica alla camera; js/core/ui/navPad.js le mostra a schermo.
 * I vettori sono oggetti {x, y, z} qualsiasi (vanno bene anche i THREE.Vector3).
 */

/** Gli assi del movimento continuo: tutti gli input (tasti, pulsanti tenuti premuti) finiscono qui. */
export const AXES = ["panX", "panY", "orbit", "tilt", "zoom"];

/**
 * Velocità a "1×" (Shift e Alt le moltiplicano).
 * - pan: mezze altezze dello schermo al secondo (quindi proporzionale alla distanza: da vicino ci si
 *   muove piano, da lontano in fretta, e la scena scorre sempre alla stessa velocità sullo schermo);
 * - orbit/tilt: radianti al secondo (circa 70°/s);
 * - zoom: velocità esponenziale (distanza × e^(−rate·t): in ~0,6 s la distanza si dimezza).
 */
export const SPEED = { pan: 0.9, orbit: 1.2, tilt: 0.9, zoom: 1.1 };
export const MODIFIERS = { fast: 2.5, precise: 0.3 };

/** Costanti di tempo (secondi) di accelerazione e frenata: piccole = scatto, grandi = "nave pesante". */
export const SMOOTHING = { accelerate: 0.12, brake: 0.16 };

/** Sotto questa velocità (in unità dell'asse) un asse è fermo: niente lavoro nel ciclo. */
export const REST_EPSILON = 1e-3;

/** Tasti di movimento continuo, per posizione fisica (e.code): W A S D restano WASD anche su tastiere diverse. */
const MOVE_CODES = {
  KeyW: ["panY", +1],
  KeyS: ["panY", -1],
  KeyA: ["panX", -1],
  KeyD: ["panX", +1],
  ArrowLeft: ["panX", -1],
  ArrowRight: ["panX", +1],
  KeyQ: ["orbit", -1],
  KeyE: ["orbit", +1],
  KeyR: ["tilt", +1],
  KeyF: ["tilt", -1],
  KeyZ: ["zoom", +1],
  KeyX: ["zoom", -1],
  NumpadAdd: ["zoom", +1],
  NumpadSubtract: ["zoom", -1],
};

/** "+" e "-" cambiano posto da una tastiera all'altra (su quella italiana "-" è dove l'americana ha "/"): si guarda il carattere. */
const MOVE_KEYS = { "+": ["zoom", +1], "=": ["zoom", +1], "-": ["zoom", -1], _: ["zoom", -1] };

const COMMAND_CODES = { KeyC: "home", Home: "home", KeyM: "mode", KeyH: "help" };
const COMMAND_KEYS = { "?": "help", Escape: "escape" };

/**
 * Cosa significa un tasto per la navigazione.
 * @param {{code?: string, key?: string, ctrlKey?: boolean, metaKey?: boolean}} e evento keydown/keyup (o simile)
 * @returns {{type: "move", axis: string, dir: number, id: string} | {type: "command", name: string} | null}
 *   `id` identifica il tasto premuto (per sapere quale rilasciare al keyup).
 *   Con Ctrl/⌘ niente: sono le scorciatoie del browser e dell'app (Ctrl+K, Ctrl+C, zoom della pagina…).
 */
export function classifyKey(e) {
  if (e.ctrlKey || e.metaKey) return null;
  const byKey = MOVE_KEYS[e.key];
  if (byKey) return { type: "move", axis: byKey[0], dir: byKey[1], id: `key:${e.key === "=" ? "+" : e.key === "_" ? "-" : e.key}` };
  const byCode = MOVE_CODES[e.code];
  if (byCode) return { type: "move", axis: byCode[0], dir: byCode[1], id: `code:${e.code}` };
  const command = COMMAND_KEYS[e.key] ?? COMMAND_CODES[e.code];
  return command ? { type: "command", name: command } : null;
}

/** Moltiplicatore di velocità: Shift = più veloce, Alt = più preciso (insieme si compensano). */
export function speedMultiplier({ fast = false, precise = false } = {}) {
  return (fast ? MODIFIERS.fast : 1) * (precise ? MODIFIERS.precise : 1);
}

/**
 * Avvicina `current` a `target` con un filtro esponenziale indipendente dal frame rate:
 * dopo `tau` secondi ha percorso il 63% della strada, qualunque sia il numero di frame.
 */
export function smoothTowards(current, target, dt, tau) {
  if (tau <= 0) return target;
  return current + (target - current) * (1 - Math.exp(-dt / tau));
}

/**
 * Un passo della velocità di un asse: accelera verso `wanted` quando c'è input, frena verso 0 quando non c'è.
 * Sotto REST_EPSILON torna 0 esatto (così il ciclo sa che può riposare).
 */
export function stepVelocity(velocity, wanted, dt, { accelerate = SMOOTHING.accelerate, brake = SMOOTHING.brake } = {}) {
  const tau = wanted !== 0 && Math.sign(wanted) === Math.sign(velocity || wanted) ? accelerate : brake;
  const next = smoothTowards(velocity, wanted, dt, tau);
  return wanted === 0 && Math.abs(next) < REST_EPSILON ? 0 : next;
}

/**
 * Fattore di "freno" vicino al bordo: 1 dentro i confini, poi scende dolcemente
 * (1/(1+o)² con o = sforamento/zona morbida). A una zona di distanza la spinta è già ridotta a 1/4.
 */
export function edgeResistance(overshoot, softZone) {
  if (overshoot <= 0 || softZone <= 0) return 1;
  const o = overshoot / softZone;
  return 1 / ((1 + o) * (1 + o));
}

/**
 * Frena lo spostamento `move` asse per asse quando porta `point` oltre il bordo di `box`: si "scivola
 * lungo il muro". Con la camera ruotata di lato uno spostamento a destra ha anche una componente in
 * profondità: se è quella a sbattere sul bordo, la componente laterale continua. Modifica `move`.
 */
export function resistMove(point, box, move, softZone) {
  for (const k of ["x", "y", "z"]) {
    const over = move[k] > 0 ? point[k] - box.max[k] : move[k] < 0 ? box.min[k] - point[k] : 0;
    if (over > 0) move[k] *= edgeResistance(over, softZone);
  }
  return move;
}

/**
 * Di quanto il punto guardato può uscire dal contenuto: una frazione della scena, ma mai più di
 * mezza inquadratura (`view` × metà altezza visibile alla distanza attuale). Da lontano conta la
 * scena, da vicino lo schermo: al bordo almeno metà dello schermo mostra ancora qualcosa, anche al
 * livello Dettaglio dove la scena intera è molto più grande dell'inquadratura.
 */
export function boundsMargin({ diagonal, halfHeight }, { fraction = 0.12, view = 0.5, minimum = 0.5 } = {}) {
  return Math.max(minimum, Math.min(diagonal * fraction, halfHeight * view));
}

/**
 * Zoom indietro massimo dello studente: `radii` raggi della scena (la scena occupa ancora ~1/4 dello
 * schermo), oppure un po' oltre la vista iniziale se questa è già più lontana (schermi stretti).
 */
export function maxUserDistance({ radius, homeDistance = 0 }, { radii = 6, home = 1.5 } = {}) {
  return Math.max(radius * radii, homeDistance * home);
}

/** Vettore che riporta `point` dentro `box` (zero se è già dentro). Scrive in `out` e lo restituisce. */
export function excessToBox(point, box, out = { x: 0, y: 0, z: 0 }) {
  for (const k of ["x", "y", "z"]) {
    out[k] = point[k] < box.min[k] ? box.min[k] - point[k] : point[k] > box.max[k] ? box.max[k] - point[k] : 0;
  }
  return out;
}

/**
 * Allarga un box del contenuto per farne i confini della navigazione: margine proporzionale alla
 * scena (frazione della diagonale) con un minimo assoluto, così anche una scena piatta ha "spessore".
 */
export function expandBox(box, { fraction = 0.35, minimum = 4 } = {}) {
  const dx = box.max.x - box.min.x;
  const dy = box.max.y - box.min.y;
  const dz = box.max.z - box.min.z;
  const pad = Math.max(minimum, Math.hypot(dx, dy, dz) * fraction);
  return {
    min: { x: box.min.x - pad, y: box.min.y - pad, z: box.min.z - pad },
    max: { x: box.max.x + pad, y: box.max.y + pad, z: box.max.z + pad },
  };
}

/** Nuova distanza dopo `dt` secondi di zoom continuo (dir +1 = avanti, −1 = indietro), dentro [min, max]. */
export function dollyDistance(distance, rate, dt, { min = 0.3, max = Infinity } = {}) {
  const next = distance * Math.exp(-rate * dt);
  return Math.min(max, Math.max(min, next));
}

/**
 * "Ti sei perso?": la scena è fuori dall'inquadratura, oppure è così lontana da essere un puntino
 * (occupa meno di `minFraction` dell'altezza dello schermo).
 */
export function isLost({ inFrustum, distance, radius, fovRad, minFraction = 0.06 }) {
  if (!inFrustum) return true;
  if (!(radius > 0) || !(distance > 0)) return false;
  const halfHeight = distance * Math.tan(fovRad / 2);
  return radius / halfHeight < minFraction;
}

/**
 * Smorzamento delle OrbitControls indipendente dal frame rate: `base` è il fattore pensato per 60 fps;
 * a 20 fps (rendering software) ogni frame deve smorzare di più, altrimenti la scena "scivola" 3 volte più a lungo.
 */
export function frameDamping(base, dt) {
  return 1 - Math.pow(1 - base, Math.max(0, dt) * 60);
}

/**
 * Doppio tocco: due tocchi brevi, vicini nel tempo e nello spazio.
 * @param {{t: number, x: number, y: number} | null} previous tocco precedente
 * @param {{t: number, x: number, y: number}} current tocco attuale (t in ms)
 */
export function isDoubleTap(previous, current, { maxDelay = 320, maxDistance = 30 } = {}) {
  if (!previous) return false;
  return current.t - previous.t <= maxDelay && Math.hypot(current.x - previous.x, current.y - previous.y) <= maxDistance;
}

/**
 * Distanza dal fondo dello stage a cui mettere il pad perché non copra la barra del player:
 * resta a `base` se non si sovrappongono, altrimenti sale sopra la barra.
 * @param {{left: number, right: number, height: number}} pad estensione del pad (px, coordinate dello stage)
 * @param {{left: number, right: number, top: number, bottom: number} | null} bar la barra (null se nascosta)
 * @param {number} stageHeight altezza dello stage
 * @param {number} [base] distanza normale dal fondo
 * @param {number} [gap] spazio tra barra e pad
 */
export function padBottom(pad, bar, stageHeight, base = 14, gap = 10) {
  if (!bar) return base;
  const top = stageHeight - base - pad.height;
  const bottom = stageHeight - base;
  const overlaps = pad.left < bar.right && pad.right > bar.left && top < bar.bottom && bottom > bar.top;
  return overlaps ? Math.max(base, stageHeight - bar.top + gap) : base;
}

/** Nome del file della foto: scena-<tab>-AAAA-MM-GG-hhmmss.png (ordinabile, senza caratteri strani). */
export function snapshotName(tabId, date = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  const safe = String(tabId || "scena").replace(/[^\w-]+/g, "-").replace(/^-+|-+$/g, "") || "scena";
  const stamp = `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
  return `scena-${safe}-${stamp}.png`;
}
