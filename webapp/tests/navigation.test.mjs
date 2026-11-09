// Test della matematica pura della navigazione libera (js/core/navMath.js):
//   node --test webapp/tests/*.test.mjs
// Niente DOM e niente three.js: tasti → movimenti, accelerazione e frenata indipendenti dal frame rate,
// bordo "elastico", zoom limitato, "ti sei perso?", doppio tocco, posizione del pad e della didascalia, nome della foto.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  AXES,
  CAPTION_HIDE_WIDTH,
  CAPTION_MIN_WIDTH,
  MODIFIERS,
  REST_EPSILON,
  captionPlacement,
  classifyKey,
  dollyDistance,
  edgeResistance,
  excessToBox,
  expandBox,
  frameDamping,
  isDoubleTap,
  isLost,
  maxUserDistance,
  padBottom,
  padPlacement,
  resistMove,
  boundsMargin,
  smoothTowards,
  snapshotName,
  speedMultiplier,
  stepVelocity,
} from "../js/core/navMath.js";

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} ≉ ${b}`);

/** Simula `seconds` secondi di velocità con frame di `dt` secondi; restituisce la strada percorsa. */
function travel(wanted, seconds, dt) {
  let v = 0;
  let x = 0;
  for (let t = 0; t < seconds - 1e-9; t += dt) {
    v = stepVelocity(v, wanted, dt);
    x += v * dt;
  }
  return { v, x };
}

test("tasti: W A S D e frecce ← → spostano, Q E R F ruotano, + − zoomano", () => {
  const move = (e) => {
    const k = classifyKey(e);
    return k && k.type === "move" ? [k.axis, k.dir] : k;
  };
  assert.deepEqual(move({ code: "KeyW", key: "w" }), ["panY", 1]);
  assert.deepEqual(move({ code: "KeyS", key: "s" }), ["panY", -1]);
  assert.deepEqual(move({ code: "KeyA", key: "a" }), ["panX", -1]);
  assert.deepEqual(move({ code: "KeyD", key: "d" }), ["panX", 1]);
  assert.deepEqual(move({ code: "ArrowLeft", key: "ArrowLeft" }), ["panX", -1]);
  assert.deepEqual(move({ code: "ArrowRight", key: "ArrowRight" }), ["panX", 1]);
  assert.deepEqual(move({ code: "KeyQ", key: "q" }), ["orbit", -1]);
  assert.deepEqual(move({ code: "KeyE", key: "e" }), ["orbit", 1]);
  assert.deepEqual(move({ code: "KeyR", key: "r" }), ["tilt", 1]);
  assert.deepEqual(move({ code: "KeyF", key: "f" }), ["tilt", -1]);
  // "+" e "-" si riconoscono dal carattere: sulla tastiera italiana "-" sta dove l'americana ha "/"
  assert.deepEqual(move({ code: "BracketRight", key: "+" }), ["zoom", 1]);
  assert.deepEqual(move({ code: "Slash", key: "-" }), ["zoom", -1]);
  assert.deepEqual(move({ code: "Equal", key: "=" }), ["zoom", 1]);
  assert.deepEqual(move({ code: "NumpadAdd", key: "+" }), ["zoom", 1]);
  assert.deepEqual(move({ code: "KeyZ", key: "z" }), ["zoom", 1]);
  assert.deepEqual(move({ code: "KeyX", key: "x" }), ["zoom", -1]);
  // con Alt sul Mac il carattere cambia (Alt+W = "∑"), la posizione fisica no
  assert.deepEqual(move({ code: "KeyW", key: "∑", altKey: true }), ["panY", 1]);
});

test("tasti: comandi, e niente che rubi i tasti degli altri (↑ ↓ Spazio, Ctrl/⌘, /)", () => {
  assert.deepEqual(classifyKey({ code: "KeyC", key: "c" }), { type: "command", name: "home" });
  assert.deepEqual(classifyKey({ code: "Home", key: "Home" }), { type: "command", name: "home" });
  assert.deepEqual(classifyKey({ code: "KeyM", key: "m" }), { type: "command", name: "mode" });
  assert.deepEqual(classifyKey({ code: "Minus", key: "?" }), { type: "command", name: "help" }); // "?" italiano: Shift + '
  assert.deepEqual(classifyKey({ code: "KeyH", key: "h" }), { type: "command", name: "help" });
  assert.deepEqual(classifyKey({ code: "Escape", key: "Escape" }), { type: "command", name: "escape" });
  for (const e of [
    { code: "ArrowUp", key: "ArrowUp" }, // menu del Percorso
    { code: "ArrowDown", key: "ArrowDown" },
    { code: "Space", key: " " }, // play/pausa del player
    { code: "Digit7", key: "/" }, // ricerca globale
    { code: "KeyK", key: "k", ctrlKey: true },
    { code: "KeyC", key: "c", metaKey: true }, // copia
    { code: "Equal", key: "+", ctrlKey: true }, // zoom della pagina
    { code: "Enter", key: "Enter" },
  ]) {
    assert.equal(classifyKey(e), null, JSON.stringify(e));
  }
  // la stessa direzione dà lo stesso id anche se Shift viene rilasciato prima del tasto ("+" → "=")
  assert.equal(classifyKey({ code: "Equal", key: "+" }).id, classifyKey({ code: "Equal", key: "=" }).id);
});

test("modificatori: Shift più veloce, Alt più preciso", () => {
  assert.equal(speedMultiplier({}), 1);
  assert.equal(speedMultiplier({ fast: true }), MODIFIERS.fast);
  assert.equal(speedMultiplier({ precise: true }), MODIFIERS.precise);
  assert.ok(speedMultiplier({ fast: true }) > 1 && speedMultiplier({ precise: true }) < 1);
  near(speedMultiplier({ fast: true, precise: true }), MODIFIERS.fast * MODIFIERS.precise);
});

test("smoothTowards: dopo tau secondi il 63% della strada, a qualunque frame rate", () => {
  near(smoothTowards(0, 1, 0.1, 0.1), 1 - Math.exp(-1));
  let a = 0;
  for (let i = 0; i < 6; i++) a = smoothTowards(a, 1, 1 / 60, 0.1); // 60 fps
  let b = 0;
  for (let i = 0; i < 2; i++) b = smoothTowards(b, 1, 1 / 20, 0.1); // 20 fps (rendering software)
  near(a, b, 1e-9);
  assert.equal(smoothTowards(0.3, 1, 0.5, 0), 1);
});

test("movimento continuo: stessa strada a 60 fps e a 15 fps, accelera e poi frena fino a fermarsi", () => {
  const fast = travel(1, 1, 1 / 60);
  const slow = travel(1, 1, 1 / 15);
  near(fast.x, slow.x, 0.03); // indipendente dal frame rate (a 15 fps l'integrazione a passi sbaglia di ~2%)
  assert.ok(fast.v > 0.99, "dopo 1 s la velocità è quasi piena");
  assert.ok(travel(1, 0.05, 1 / 60).v < 0.4, "all'inizio accelera, non scatta");
  // frenata: rilasciato il tasto la velocità torna esattamente a 0 in tempi brevi
  let v = 1;
  let t = 0;
  while (v !== 0 && t < 3) {
    v = stepVelocity(v, 0, 1 / 60);
    t += 1 / 60;
  }
  assert.equal(v, 0);
  assert.ok(t > 0.3 && t < 1.6, `frenata in ${t.toFixed(2)} s`);
  // invertire la direzione passa per lo zero senza salti
  const reversed = stepVelocity(1, -1, 1 / 60);
  assert.ok(reversed < 1 && reversed > 0);
  assert.equal(stepVelocity(REST_EPSILON / 2, 0, 1 / 60), 0);
  assert.deepEqual(AXES, ["panX", "panY", "orbit", "tilt", "zoom"]);
});

test("bordo elastico: dentro nessun freno, fuori la spinta cala dolcemente", () => {
  assert.equal(edgeResistance(0, 5), 1);
  assert.equal(edgeResistance(-3, 5), 1);
  near(edgeResistance(5, 5), 0.25);
  const samples = [0.5, 1, 2, 4, 8].map((o) => edgeResistance(o, 5));
  for (let i = 1; i < samples.length; i++) assert.ok(samples[i] < samples[i - 1], "decrescente");
  assert.ok(samples[0] > 0.8, "niente scatti appena oltre il bordo");

  const box = { min: { x: -10, y: -10, z: -10 }, max: { x: 10, y: 10, z: 10 } };
  // oltre il bordo destro: la spinta verso destra è frenata, quella verso il centro no
  const out = resistMove({ x: 12, y: 0, z: 0 }, box, { x: 1, y: 1, z: 0 }, 2);
  near(out.x, 0.25);
  assert.equal(out.y, 1, "lungo il muro si scivola: l'altra componente non è frenata");
  assert.deepEqual(resistMove({ x: 12, y: 0, z: 0 }, box, { x: -1, y: 0, z: 0 }, 2), { x: -1, y: 0, z: 0 });
  assert.deepEqual(resistMove({ x: 0, y: 0, z: 0 }, box, { x: 5, y: -5, z: 5 }, 2), { x: 5, y: -5, z: 5 }, "dentro: nessun freno");
  assert.deepEqual(excessToBox({ x: 12, y: 0, z: -11 }, box), { x: -2, y: 0, z: 1 });
  assert.deepEqual(excessToBox({ x: 1, y: 2, z: 3 }, box), { x: 0, y: 0, z: 0 });
});

test("expandBox: margine proporzionale alla scena, con un minimo anche per le scene piatte", () => {
  const flat = expandBox({ min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: 1, z: 0 } }, { fraction: 0.35, minimum: 4 });
  assert.equal(flat.max.z - flat.min.z, 8);
  const big = expandBox({ min: { x: -100, y: 0, z: 0 }, max: { x: 100, y: 0, z: 0 } }, { fraction: 0.25, minimum: 4 });
  assert.equal(big.max.x, 150);
  assert.equal(big.min.y, -50);
});

test("margine dei confini: da lontano conta la scena, da vicino mezza inquadratura", () => {
  // scena larga 300: da lontano il 12% della diagonale
  near(boundsMargin({ diagonal: 300, halfHeight: 400 }), 36);
  // al livello Dettaglio (metà altezza visibile 17) il bordo resta a metà dello schermo
  near(boundsMargin({ diagonal: 300, halfHeight: 17 }), 8.5);
  assert.equal(boundsMargin({ diagonal: 300, halfHeight: 0 }), 0.5, "mai zero");
});

test("zoom indietro dello studente: 6 raggi della scena, o poco oltre una vista iniziale già lontana", () => {
  assert.equal(maxUserDistance({ radius: 100, homeDistance: 200 }), 600);
  assert.equal(maxUserDistance({ radius: 100, homeDistance: 810 }), 1215, "telefono: non 3 volte la vista iniziale");
  assert.equal(maxUserDistance({ radius: 100 }), 600);
});

test("zoom continuo: esponenziale e dentro i limiti", () => {
  near(dollyDistance(100, Math.log(2), 1), 50);
  near(dollyDistance(100, -Math.log(2), 1), 200);
  assert.equal(dollyDistance(1, 10, 1, { min: 0.5 }), 0.5);
  assert.equal(dollyDistance(100, -10, 1, { max: 300 }), 300);
});

test("ti sei perso?: fuori inquadratura o scena ridotta a un puntino", () => {
  const fov = (45 * Math.PI) / 180;
  assert.equal(isLost({ inFrustum: false, distance: 10, radius: 5, fovRad: fov }), true);
  assert.equal(isLost({ inFrustum: true, distance: 100, radius: 30, fovRad: fov }), false);
  assert.equal(isLost({ inFrustum: true, distance: 10000, radius: 30, fovRad: fov }), true);
  assert.equal(isLost({ inFrustum: true, distance: 0, radius: 30, fovRad: fov }), false, "dentro la scena non sei perso");
});

test("smorzamento delle OrbitControls: stesso effetto al secondo a 60 e a 20 fps", () => {
  near(frameDamping(0.08, 1 / 60), 0.08);
  const left60 = Math.pow(1 - frameDamping(0.08, 1 / 60), 60);
  const left20 = Math.pow(1 - frameDamping(0.08, 1 / 20), 20);
  near(left60, left20, 1e-9);
  assert.equal(frameDamping(0.08, 0), 0);
});

test("doppio tocco: vicino nel tempo e nello spazio", () => {
  const a = { t: 1000, x: 100, y: 100 };
  assert.equal(isDoubleTap(null, a), false);
  assert.equal(isDoubleTap(a, { t: 1250, x: 110, y: 104 }), true);
  assert.equal(isDoubleTap(a, { t: 1500, x: 100, y: 100 }), false, "troppo lento");
  assert.equal(isDoubleTap(a, { t: 1200, x: 180, y: 100 }), false, "troppo lontano");
});

test("pad: sale sopra la barra del player solo se la toccherebbe", () => {
  const bar = { left: 250, right: 970, top: 680, bottom: 804 };
  // stage 1220×848: pad nell'angolo (x 1030…1206), barra al centro → nessuna sovrapposizione
  assert.equal(padBottom({ left: 1030, right: 1206, height: 160 }, bar, 848), 14);
  // stage stretto: il pad tocca la barra → sale sopra di essa
  assert.equal(padBottom({ left: 700, right: 886, height: 160 }, bar, 848), 848 - 680 + 10);
  assert.equal(padBottom({ left: 700, right: 886, height: 160 }, null, 848), 14);
  // sul telefono il pulsante unico sta sotto la barra (che parte da 44 px dal fondo): nessuno spostamento
  assert.equal(padBottom({ left: 342, right: 382, height: 40 }, { left: 16, right: 374, top: 650, bottom: 800 }, 844, 4), 4);
});

test("pad sul telefono: sale sopra il codice in basso e poi, se serve, sopra il player", () => {
  const pad = { left: 179, right: 382, height: 192 };
  const code = { left: 16, right: 374, top: 685, bottom: 744 }; // codice ridotto in fondo allo stage (792 px)
  const bottom = padBottom(pad, code, 792, 46);
  assert.equal(bottom, 792 - 685 + 10, "sopra il codice, non sopra la didascalia");
  // concatenando gli ostacoli (codice, poi barra) il pad finisce sopra il più alto dei due
  const bar = { left: 16, right: 374, top: 470, bottom: 620 };
  assert.equal(padBottom(pad, bar, 792, bottom), 792 - 470 + 10);
  assert.equal(padBottom(pad, null, 792, bottom), bottom, "senza player resta dov'è");
});

test("posto del pad (padPlacement): angolo, accanto ai grafici, sopra il player, accanto al codice, telefono", () => {
  const size = { w: 190, h: 146 };
  // stage 1220×848 libero: angolo in basso a destra
  assert.deepEqual(padPlacement({ sw: 1220, sh: 848, size }), { right: 14, bottom: 14, ok: true });
  // grafici aperti: a sinistra dei loro 460 px
  assert.deepEqual(padPlacement({ sw: 1220, sh: 848, size, chartsWidth: 460 }), { right: 472, bottom: 14, ok: true });
  // ... ma con 260 px di scena accanto ai grafici (1100 px, sidebar aperta) il pad aperto ne coprirebbe tre quarti:
  // ok false, navPad lo riduce al pulsante unico (che ci sta: 40 px)
  assert.equal(padPlacement({ sw: 720, sh: 748, size, chartsWidth: 460 }).ok, false);
  assert.equal(padPlacement({ sw: 720, sh: 748, size: { w: 40, h: 40 }, chartsWidth: 460 }).ok, true);
  assert.equal(padPlacement({ sw: 900, sh: 748, size, chartsWidth: 460 }).ok, true, "1280 px con la sidebar: 440 px di scena");
  // la barra del player sotto l'angolo: il pad sale sopra di lei (la regola di padBottom)
  const bar = { left: 700, right: 1210, top: 680, bottom: 804 };
  assert.deepEqual(padPlacement({ sw: 1220, sh: 848, size, bar }), { right: 14, bottom: 848 - 680 + 10, ok: true });
  // codice lungo fino in fondo: il pad va accanto al codice, che resta nella metà destra della scena visibile
  const code = { left: 766, right: 1206, top: 56, bottom: 800 };
  assert.deepEqual(padPlacement({ sw: 1220, sh: 848, size, code }), { right: 1220 - 766 + 12, bottom: 14, ok: true });
  // codice corto (non arriva all'angolo): il pad resta nell'angolo
  assert.deepEqual(padPlacement({ sw: 1220, sh: 848, size, code: { ...code, bottom: 500 } }), { right: 14, bottom: 14, ok: true });
  // stage stretto: accanto al codice finirebbe al centro della scena → ok false (navPad lo riduce al pulsante)
  assert.equal(padPlacement({ sw: 700, sh: 848, size, code: { left: 300, right: 686, top: 56, bottom: 800 } }).ok, false);
  // telefono: sopra il codice ridotto in fondo allo stage
  const phone = padPlacement({ sw: 390, sh: 792, size: { w: 203, h: 192 }, lowest: { left: 16, right: 374, top: 685, bottom: 744 }, edge: 8, base: 46 });
  assert.deepEqual(phone, { right: 8, bottom: 792 - 685 + 10, ok: true });
});

test("posto della didascalia (captionPlacement): accanto al pad, sopra il pad, mai sotto i grafici, sopra il player, telefono", () => {
  assert.ok(CAPTION_MIN_WIDTH > CAPTION_HIDE_WIDTH);
  const pad = { right: 14, bottom: 14, w: 190, h: 146 };
  // schermo largo: accanto al pad, larga più di CAPTION_MIN_WIDTH
  assert.deepEqual(captionPlacement({ width: 1600, left: 396, sh: 848, height: 23, pad }), { right: 14 + 190 + 10, bottom: 14, hidden: false, abovePad: false });
  // 1100 px con sidebar e grafici aperti: accanto al pad resterebbero 32 px (una parola per riga) → sopra il pad,
  // su tutta la parte visibile ma mai sotto i grafici
  const beside = { right: 472, bottom: 14, w: 190, h: 146 };
  assert.deepEqual(captionPlacement({ width: 1100, left: 396, sh: 748, height: 23, pad: beside, chartsWidth: 460 }), { right: 476, bottom: 14 + 146 + 8, hidden: false, abovePad: true });
  assert.equal(captionPlacement({ width: 1600, left: 396, sh: 848, height: 23, chartsWidth: 460 }).right, 476, "senza pad: il margine dei grafici");
  // nemmeno sopra il pad c'è posto per leggerla: tace (navPad mette .nav-caption-squeezed)
  assert.equal(captionPlacement({ width: 1000, left: 396, sh: 748, height: 23, pad: beside, chartsWidth: 460 }).hidden, true);
  // pad già alzato (sopra il player): non è sulla stessa riga, la didascalia resta in fondo
  assert.deepEqual(captionPlacement({ width: 1600, left: 396, sh: 848, height: 23, pad: { ...pad, bottom: 178 } }), { right: 16, bottom: 14, hidden: false, abovePad: false });
  // la barra del player in basso al centro: la didascalia sale sopra di lei
  const bar = { left: 500, right: 1100, top: 680, bottom: 834 };
  assert.equal(captionPlacement({ width: 1600, left: 396, sh: 848, height: 23, pad, bar }).bottom, 848 - 680 + 8);
  // telefono: a destra lo spazio del pulsante del pad; sopra il codice ridotto se gli finirebbe sotto
  const code = { left: 16, right: 374, top: 733, bottom: 778 };
  assert.deepEqual(captionPlacement({ width: 390, left: 16, sh: 792, height: 46, code, narrow: true }), { right: 56, bottom: 792 - 733 + 6, hidden: false, abovePad: false });
  assert.equal(captionPlacement({ width: 390, left: 16, sh: 792, height: 46, narrow: true }).bottom, 14, "senza codice: in fondo");
});

test("tasti: '+' scritto con Shift (tastiera americana) è zoom; l'id del tasto è lo stesso al rilascio", () => {
  const down = classifyKey({ key: "+", code: "Equal", shiftKey: true });
  assert.equal(down.axis, "zoom");
  assert.equal(down.dir, +1);
  // al rilascio Shift può essere già alzato: e.key diventa "=" ma il movimento è lo stesso
  assert.equal(classifyKey({ key: "=", code: "Equal" }).id, down.id);
  assert.equal(classifyKey({ key: "_", code: "Minus", shiftKey: true }).dir, -1);
});

test("nome della foto: ordinabile e senza caratteri strani", () => {
  const d = new Date(2026, 10, 6, 9, 5, 7);
  assert.equal(snapshotName("modern-llm", d), "scena-modern-llm-2026-11-06-090507.png");
  assert.equal(snapshotName("../x y", d), "scena-x-y-2026-11-06-090507.png");
  assert.equal(snapshotName("", d), "scena-scena-2026-11-06-090507.png");
});
