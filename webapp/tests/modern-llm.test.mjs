// Test del tab "LLM moderni" sui moduli di soli dati (niente three.js né DOM):
// mappa delle regioni, catalogo delle azioni, indice della ricerca e collegamenti del Percorso.
import assert from "node:assert/strict";
import test from "node:test";

import { ACTION_CATALOG, ACTION_IDS, actionInfo } from "../js/tabs/modern-llm/actionCatalog.js";
import { PATH } from "../js/tabs/modern-llm/content.js";
import { nearestRegionAt, REGION_BY_ID, REGIONS, ROWS, STEP_X, STEP_Y } from "../js/tabs/modern-llm/regionData.js";
import { searchIndex } from "../js/tabs/modern-llm/search.js";

const CATEGORIES = ["Architettura", "Addestramento", "Ragionamento e agenti", "Oltre il testo"];
const center = (r) => ({ x: (r.col - 1.5) * STEP_X, y: (1.5 - r.row) * STEP_Y });

test("regioni: griglia 4 × 4 completa, id unici, righe valide", () => {
  assert.equal(REGIONS.length, 16);
  assert.equal(REGION_BY_ID.size, 16);
  const cells = new Set(REGIONS.map((r) => `${r.row},${r.col}`));
  assert.equal(cells.size, 16);
  for (const r of REGIONS) assert.ok(ROWS[r.row], `${r.id}: riga ${r.row}`);
});

test("nearestRegionAt coincide con la ricerca esaustiva del centro più vicino", () => {
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  for (let i = 0; i < 2000; i++) {
    const x = rand() * 120;
    const y = rand() * 90;
    let best = null;
    let bestD = Infinity;
    for (const r of REGIONS) {
      const c = center(r);
      const d = (c.x - x) ** 2 + (c.y - y) ** 2;
      if (d < bestD - 1e-9) [best, bestD] = [r, d];
    }
    const got = nearestRegionAt(x, y);
    const c = center(got);
    // a parità di distanza (punto esattamente a metà) vale una qualsiasi delle due
    assert.ok(Math.abs((c.x - x) ** 2 + (c.y - y) ** 2 - bestD) < 1e-6, `(${x}, ${y}): ${got.id} invece di ${best.id}`);
  }
  // sul centro di ogni regione, la regione stessa; lontanissimo, quella dell'angolo
  for (const r of REGIONS) assert.equal(nearestRegionAt(center(r).x, center(r).y), r);
  assert.equal(nearestRegionAt(-1e6, 1e6).id, "compare");
  assert.equal(nearestRegionAt(1e6, -1e6).id, "mesh");
});

test("catalogo delle azioni: id unici, categorie e regioni esistenti, testi completi", () => {
  assert.equal(ACTION_IDS.size, ACTION_CATALOG.length);
  for (const a of ACTION_CATALOG) {
    assert.ok(CATEGORIES.includes(a.category), `${a.id}: categoria ${a.category}`);
    assert.ok(REGION_BY_ID.has(a.region), `${a.id}: regione ${a.region}`);
    assert.ok(a.title.length > 5 && a.summary.length > 20, `${a.id}: titolo o riassunto`);
    assert.ok(a.keywords.length >= 3, `${a.id}: parole chiave`);
  }
  assert.deepEqual(Object.keys(actionInfo("moe")).sort(), ["category", "id", "summary", "title"]);
  assert.throws(() => actionInfo("nessuna"));
});

test("Percorso: ogni regione e ogni azione citata esiste", () => {
  for (const section of PATH) {
    for (const item of section.children) {
      if (item.focus) assert.ok(REGION_BY_ID.has(item.focus), `${item.id}: regione ${item.focus}`);
      if (item.action) assert.ok(ACTION_IDS.has(item.action), `${item.id}: azione ${item.action}`);
    }
  }
});

test("indice della ricerca: tutti i passi, tutte le azioni, livelli e regioni", () => {
  const keys = new Set(searchIndex.map((e) => `${e.kind}:${e.id}`));
  assert.equal(keys.size, searchIndex.length, "voci duplicate");
  for (const section of PATH) for (const item of section.children) assert.ok(keys.has(`step:${item.id}`), `manca il passo ${item.id}`);
  for (const a of ACTION_CATALOG) assert.ok(keys.has(`action:${a.id}`), `manca l'azione ${a.id}`);
  for (const id of ["macro", "blocks", "detail", ...REGIONS.map((r) => r.id)]) assert.ok(keys.has(`view:${id}`), `manca la vista ${id}`);
  for (const e of searchIndex) assert.ok(e.keywords?.length, `${e.kind}:${e.id} senza parole chiave`);
});
