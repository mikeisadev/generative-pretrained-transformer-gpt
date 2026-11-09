// Test della logica pura del selettore delle rappresentazioni (js/core/ui/repSwitch.js):
//   node --test webapp/tests/*.test.mjs
// Niente DOM e niente three.js: dove mettere "Architettura · Rete neurale · Cervello" senza coprire
// gli altri comandi dello stage, a ogni larghezza (prima fila, seconda fila, solo icone, menu compatto).
// Gli ostacoli riproducono le misure vere della pagina di prova (tests/reps-demo.html).
import assert from "node:assert/strict";
import { test } from "node:test";

import { BRAIN_PANE_ID, REP_IDS, REP_LINKS, REP_SEARCH_KEYWORDS, repOfLink, withRepKeywords } from "../js/core/repLinks.js";
import { VARIANTS, candidateRows, freeIntervals, placeSwitch } from "../js/core/ui/repSwitch.js";

const SIZES = { full: 322, icons: 104, menu: 48, height: 36 };
const box = (left, top, width, height) => ({ left, top, right: left + width, bottom: top + height });
const overlaps = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
/** Il riquadro occupato dal selettore per un risultato di placeSwitch. */
const rectOf = (spot, sizes = SIZES) => box(spot.left, spot.top, sizes[spot.variant], sizes.height);

/** Comandi in alto di un tab desktop: ☰, selettore dello zoom, barra strumenti, "Grafici", codice. */
function desktopObstacles(width, { lod = 330, toolbar = 200, code = true } = {}) {
  const list = [box(10, 10, 38, 32), box(56, 10, lod, 36), box(width - 140 - toolbar, 10, toolbar, 32), box(width - 14 - 96, 10, 96, 32)];
  if (code) list.push(box(width - 14 - Math.min(560, 0.42 * (width + 380)), 56, Math.min(560, 0.42 * (width + 380)), 600));
  return list;
}

test("VARIANTS: dalla più ricca alla più compatta", () => {
  assert.deepEqual(VARIANTS, ["full", "icons", "menu"]);
});

test("freeIntervals: senza ostacoli tutta la fascia meno i bordi", () => {
  assert.deepEqual(freeIntervals({ width: 500, top: 10, height: 36 }), [{ start: 10, end: 490 }]);
});

test("freeIntervals: un ostacolo nella fascia la divide, allargato dello spazio minimo", () => {
  const free = freeIntervals({ width: 500, top: 10, height: 36, obstacles: [box(100, 10, 50, 30)], gap: 8 });
  assert.deepEqual(free, [
    { start: 10, end: 92 },
    { start: 158, end: 490 },
  ]);
});

test("freeIntervals: gli ostacoli sopra o sotto la fascia non contano (anche a filo dello spazio minimo)", () => {
  const below = box(0, 54, 500, 40); // inizia esattamente a top + height + gap
  const above = box(0, -40, 500, 42); // finisce esattamente a top - gap
  assert.deepEqual(freeIntervals({ width: 500, top: 10, height: 36, obstacles: [below, above], gap: 8 }), [{ start: 10, end: 490 }]);
});

test("freeIntervals: ostacoli sovrapposti e oltre i bordi", () => {
  const free = freeIntervals({ width: 400, top: 0, height: 30, obstacles: [box(-50, 0, 120, 30), box(60, 5, 40, 20), box(350, 0, 100, 30)], gap: 4, edge: 10 });
  assert.deepEqual(free, [{ start: 104, end: 346 }]);
});

test("candidateRows: la seconda fila sta sotto i comandi della prima; i pannelli alti non fanno fila", () => {
  const obstacles = [box(10, 10, 38, 32), box(56, 10, 300, 36), box(600, 56, 400, 700)];
  const rows = candidateRows({ obstacles, height: 36 });
  assert.equal(rows[0], 10);
  assert.equal(rows[1], 46 + 8); // sotto il selettore dello zoom (il più basso della prima fila)
  assert.equal(rows[2], rows[1] + 36 + 8); // nessun comando nella seconda: subito sotto il selettore
});

test("placeSwitch: 1600×900 con la sidebar aperta → prima fila, etichette, al centro della scena", () => {
  const width = 1220;
  const spot = placeSwitch({ width, stageHeight: 900, sizes: SIZES, height: SIZES.height, obstacles: desktopObstacles(width) });
  assert.equal(spot.variant, "full");
  assert.equal(spot.row, 0);
  assert.equal(spot.top, 10);
  assert.ok(Math.abs(spot.left + SIZES.full / 2 - width / 2) < 1, "centrato");
  for (const o of desktopObstacles(width)) assert.ok(!overlaps(rectOf(spot), o), "non copre niente");
});

test("placeSwitch: poco spazio nella prima fila → solo icone, senza cambiare fila", () => {
  const width = 900; // 1280 con la sidebar aperta e la barra strumenti del Transformer
  const obstacles = desktopObstacles(width, { lod: 262, toolbar: 176 });
  const spot = placeSwitch({ width, stageHeight: 800, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(spot.variant, "icons");
  assert.equal(spot.row, 0);
  for (const o of obstacles) assert.ok(!overlaps(rectOf(spot), o));
});

test("placeSwitch: prima fila piena → seconda fila a sinistra, sotto ☰", () => {
  const width = 720; // 1100 con la sidebar aperta
  const obstacles = desktopObstacles(width, { lod: 262, toolbar: 176, code: false });
  const spot = placeSwitch({ width, stageHeight: 800, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(spot.row, 1);
  assert.equal(spot.variant, "full");
  assert.equal(spot.left, 10);
  for (const o of obstacles) assert.ok(!overlaps(rectOf(spot), o));
});

test("placeSwitch: seconda fila stretta dal codice → solo icone", () => {
  const width = 720;
  const obstacles = desktopObstacles(width, { lod: 262, toolbar: 176, code: true });
  const spot = placeSwitch({ width, stageHeight: 800, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(spot.row, 1);
  assert.equal(spot.variant, "icons");
  for (const o of obstacles) assert.ok(!overlaps(rectOf(spot), o));
});

test("placeSwitch: telefono con barra strumenti e selettore dello zoom → menu compatto accanto allo zoom", () => {
  const width = 390;
  const obstacles = [box(10, 10, 38, 32), box(56, 10, 214, 32), box(280, 10, 96, 32), box(10, 54, 292, 34)];
  const spot = placeSwitch({ width, stageHeight: 790, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(spot.variant, "menu");
  assert.equal(spot.row, 1);
  for (const o of obstacles) assert.ok(!overlaps(rectOf(spot), o));
});

test("placeSwitch: telefono senza barra strumenti → icone nella prima fila", () => {
  const width = 390;
  const obstacles = [box(10, 10, 38, 32), box(280, 10, 96, 32), box(10, 54, 292, 34)];
  const spot = placeSwitch({ width, stageHeight: 790, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(spot.row, 0);
  assert.equal(spot.variant, "icons");
  for (const o of obstacles) assert.ok(!overlaps(rectOf(spot), o));
});

test("placeSwitch: con i grafici aperti il centro preferito è quello della parte libera", () => {
  const width = 1220;
  const charts = box(width - 460, 0, 460, 900);
  const obstacles = [box(10, 10, 38, 32), box(56, 10, 330, 36), charts];
  const spot = placeSwitch({ width, stageHeight: 900, sizes: SIZES, height: SIZES.height, obstacles, center: (width - 460) / 2 });
  assert.equal(spot.row, 0);
  assert.ok(!overlaps(rectOf(spot), charts));
  assert.ok(spot.left >= 386 + 8, "a destra del selettore dello zoom");
});

test("placeSwitch: nessun posto libero → menu compatto nella seconda fila, segnalato con fits: false", () => {
  const width = 300;
  const walls = [box(0, 0, 150, 900), box(150, 0, 150, 900)]; // due pannelli affiancati, nessuno è un foglio
  const spot = placeSwitch({ width, stageHeight: 900, sizes: SIZES, height: SIZES.height, obstacles: walls });
  assert.equal(spot.variant, "menu");
  assert.equal(spot.fits, false);
});

test("placeSwitch: una fila che uscirebbe dallo stage non vale", () => {
  const spot = placeSwitch({ width: 300, stageHeight: 60, sizes: SIZES, height: SIZES.height, obstacles: [box(0, 0, 150, 40), box(150, 0, 150, 40)] });
  assert.equal(spot.fits, false); // la seconda fila (48 + 36) esce da uno stage alto 60
});

test("placeSwitch: un foglio a tutto stage (grafici sul telefono) non sposta il selettore", () => {
  const width = 390;
  const controls = [box(10, 10, 38, 32), box(56, 10, 214, 32), box(280, 10, 96, 32), box(10, 54, 292, 34)];
  const sheet = box(0, 0, 390, 790);
  const without = placeSwitch({ width, stageHeight: 790, sizes: SIZES, height: SIZES.height, obstacles: controls });
  const withSheet = placeSwitch({ width, stageHeight: 790, sizes: SIZES, height: SIZES.height, obstacles: [...controls, sheet] });
  assert.deepEqual(withSheet, without);
  assert.equal(withSheet.fits, true);
});

test("placeSwitch: con una vista alternativa i pannelli nascosti del tab cedono il posto ai comandi della vista", () => {
  // 1100 × 800 con la sidebar: zoom e barra strumenti del tab (nascosti ma al loro posto), codice nascosto,
  // e la scheda della vista nel livello proprio sotto ☰
  const width = 720;
  const hiddenTab = [box(56, 10, 283, 36), box(400, 10, 180, 30), box(615, 10, 91, 35), box(244, 56, 462, 173)].map((o) => ({ ...o, soft: true }));
  const viewCard = box(12, 56, 270, 212);
  const obstacles = [box(10, 10, 34, 31), ...hiddenTab, viewCard];
  const spot = placeSwitch({ width, stageHeight: 800, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(spot.fits, true);
  assert.equal(spot.row, 1);
  assert.ok(!overlaps(rectOf(spot), viewCard), "non copre la scheda della vista");
  for (const o of hiddenTab.filter((o) => o.bottom - o.top <= 120)) assert.ok(!overlaps(rectOf(spot), o), "i comandi in alto del tab restano liberi");
  // senza la scheda il posto con gli ostacoli soft basta: nessuno spostamento inutile
  const quiet = placeSwitch({ width, stageHeight: 800, sizes: SIZES, height: SIZES.height, obstacles: [box(10, 10, 34, 31), ...hiddenTab] });
  assert.deepEqual(quiet, placeSwitch({ width, stageHeight: 800, sizes: SIZES, height: SIZES.height, obstacles: [box(10, 10, 34, 31), ...hiddenTab.map(({ soft, ...o }) => o)] }));
});

test("placeSwitch: prime tre file piene a sinistra e a destra → posto di riserva in fondo alla colonna, senza coprire niente", () => {
  // Transformer con i grafici aperti a 1280 px: ☰ e zoom, barre delle viste nella seconda e terza fila che
  // arrivano fino al pannello dei grafici (a destra, alto): nelle prime tre file non c'è posto nemmeno per il menu
  const width = 520;
  const charts = box(width - 330, 0, 330, 760);
  const left = width - 330 - 18; // i comandi a sinistra arrivano fino al pannello dei grafici
  const obstacles = [box(10, 10, left, 34), box(10, 52, left, 40), box(10, 100, left, 40), box(10, 148, left, 40), charts];
  const spot = placeSwitch({ width, stageHeight: 760, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(spot.variant, "menu");
  assert.equal(spot.fits, true);
  assert.equal(spot.reserve, true);
  assert.ok(spot.row >= 3);
  for (const o of obstacles) assert.ok(!overlaps(rectOf(spot), o), "non copre nessun comando");
  // uno stage troppo basso per la riserva: resta il menu nella seconda fila, segnalato
  const low = placeSwitch({ width, stageHeight: 180, sizes: SIZES, height: SIZES.height, obstacles });
  assert.equal(low.fits, false);
});

test("repLinks: id delle rappresentazioni, link diretti e pannello Cervello importabili senza three.js", () => {
  assert.deepEqual([...REP_IDS], ["main", "network", "brain"]);
  assert.deepEqual({ ...REP_LINKS }, { rete: "network", cervello: "brain" });
  assert.equal(BRAIN_PANE_ID, "cervello");
  assert.equal(repOfLink({ kind: "view", id: "rete" }), "network");
  assert.equal(repOfLink({ kind: "view", id: "cervello" }), "brain");
  assert.equal(repOfLink({ kind: "view", id: "toString" }), null, "niente chiavi del prototipo");
  assert.equal(repOfLink({ kind: "step", id: "rete" }), null);
  assert.equal(repOfLink(), null);
});

test("repLinks: la ricerca trova le viste Rete neurale e Cervello anche con i nomi dei loro comandi", () => {
  const rete = withRepKeywords({ kind: "view", id: "rete", keywords: ["rete neurale"] });
  assert.equal(rete[0], "rete neurale", "le parole del tab restano per prime");
  for (const word of ["guarda lo spessore", "tutta la rete", "scala e onestà"]) assert.ok(rete.includes(word), word);
  assert.ok(withRepKeywords({ kind: "view", id: "cervello" }).includes("anima impulsi"), "anche senza parole del tab");
  assert.deepEqual(withRepKeywords({ kind: "step", id: "rete", keywords: ["x"] }), ["x"], "solo le viste");
  assert.equal(withRepKeywords({ kind: "view", id: "pesi" }), undefined, "le altre viste restano come sono");
  assert.equal(withRepKeywords({ kind: "view", id: "constructor" }), undefined, "niente chiavi del prototipo");
  assert.ok(Object.isFrozen(REP_SEARCH_KEYWORDS.rete) && Object.isFrozen(REP_SEARCH_KEYWORDS.cervello));
});
