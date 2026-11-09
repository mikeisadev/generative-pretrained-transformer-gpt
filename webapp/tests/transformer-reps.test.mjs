// Rappresentazioni del tab Transformer: la specifica del cervello (js/tabs/transformer/brain.js, soli dati)
// passa la validazione stretta del nucleo, usa solo regioni, marcatori e percorsi dell'atlante, cita fonti
// che esistono; i link view:rete / view:cervello e le voci della ricerca ci sono; installRepresentations
// viene chiamata dopo i pannelli del tab e prima di "Fonti".
//   node --test webapp/tests/*.test.mjs
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

import { atlasKind } from "../js/core/brain/atlas.js";
import { normalizeBrainSpec, safeUrl } from "../js/core/brain/spec.js";
import { REP_LINKS } from "../js/core/repLinks.js";
import { brainSpec } from "../js/tabs/transformer/brain.js";
import { searchIndex } from "../js/tabs/transformer/search.js";

// La ricerca verificata sta fuori dal repository: per il confronto parola per parola si indica il file con
// BRAIN_RESEARCH=/percorso/brain_data.json (senza, il test si salta).
const RESEARCH = process.env.BRAIN_RESEARCH ?? "";

test("cervello: la specifica passa la validazione stretta del nucleo, senza avvisi", () => {
  const spec = normalizeBrainSpec(brainSpec, { strict: true });
  assert.deepEqual(spec.warnings, []);
  assert.equal(spec.analogies.length, 10);
  assert.equal(spec.circuit.groups.length, 7);
  assert.ok(spec.circuit.edges.length >= 10);
  assert.ok(spec.title && spec.intro.length > 80);
});

test("cervello: regioni, marcatori e percorsi esistono nell'atlante (anche quelli delle analogie e del circuito)", () => {
  for (const id of brainSpec.regions) assert.equal(atlasKind(id), "region", id);
  for (const id of brainSpec.markers) assert.equal(atlasKind(id), "marker", id);
  for (const p of brainSpec.pathways) assert.equal(atlasKind(typeof p === "string" ? p : p.id), "pathway", JSON.stringify(p));
  for (const a of brainSpec.analogies) {
    for (const id of a.regions) assert.ok(["region", "marker"].includes(atlasKind(id)), `${a.id}: ${id}`);
    for (const id of a.markers ?? []) assert.equal(atlasKind(id), "marker", `${a.id}: ${id}`);
    for (const id of a.pathways ?? []) assert.equal(atlasKind(id), "pathway", `${a.id}: ${id}`);
  }
  for (const g of brainSpec.circuit.groups) assert.equal(atlasKind(g.region), "region", g.id);
  for (const e of brainSpec.circuit.edges) if (e.pathway) assert.equal(atlasKind(e.pathway), "pathway", `${e.from}→${e.to}`);
});

test("cervello: ogni analogia dice dove non regge e cita fonti che esistono nell'elenco delle fonti", () => {
  const byId = new Map(brainSpec.sources.map((s) => [s.id, s]));
  assert.equal(byId.size, brainSpec.sources.length, "fonti senza doppioni");
  for (const s of brainSpec.sources) {
    assert.ok(s.title, JSON.stringify(s));
    // i dati del corso sono file della repository (il pannello li scrive come testo, non come link)
    if (s.id === "corso") assert.ok(existsSync(new URL(`../../${s.url}`, import.meta.url)), `${s.id}: ${s.url} non esiste`);
    else assert.ok(safeUrl(s.url), `${s.id}: link non valido ${s.url}`);
  }
  const cited = [...brainSpec.analogies.flatMap((a) => a.sources), ...brainSpec.circuit.sources, ...brainSpec.facts.map((f) => f.source)];
  for (const s of cited) {
    assert.ok(byId.has(s.id), `fonte citata ma non elencata: ${s.id}`);
    assert.equal(byId.get(s.id).url, s.url, s.id);
  }
  for (const a of brainSpec.analogies) {
    assert.ok(a.similar?.length > 20, `${a.id}: "In comune"`);
    assert.ok(a.different?.length > 20, `${a.id}: "Dove non regge"`);
    assert.ok(["forte", "media", "debole"].includes(a.strength), `${a.id}: forza ${a.strength}`);
    assert.ok(a.sources.length >= 1, `${a.id}: fonti`);
  }
  // ogni gruppo del circuito dice a che cosa corrisponde nel GPT (anche "nessun equivalente")
  for (const g of brainSpec.circuit.groups) assert.ok(g.ai, g.id);
  // niente HTML nei testi (il pannello li scrive come testo)
  const texts = JSON.stringify(brainSpec);
  assert.ok(!/<[a-z][^>]*>/i.test(texts), "niente tag HTML");
});

test("cervello: i testi sono quelli della ricerca verificata, parola per parola", { skip: !(RESEARCH && existsSync(RESEARCH)) && "BRAIN_RESEARCH non indicato (la ricerca sta fuori dal repository)" }, async () => {
  const research = JSON.parse(await readFile(RESEARCH, "utf8")).tabs.transformer;
  assert.equal(brainSpec.intro, research.intro);
  assert.deepEqual(brainSpec.analogies.map((a) => a.id), research.analogies.map((a) => a.id));
  for (const [i, a] of brainSpec.analogies.entries()) {
    const r = research.analogies[i];
    for (const k of ["ai", "brain", "text", "similar", "different"]) assert.equal(a[k], r[k], `${a.id}.${k}`);
    assert.deepEqual(a.regions, r.regions, `${a.id}.regions`);
  }
  assert.equal(brainSpec.circuit.text, research.circuit.text);
  assert.equal(brainSpec.circuit.caveat, research.circuit.caveat);
  assert.deepEqual(brainSpec.circuit.groups.map((g) => g.label), research.circuit.groups.map((g) => g.label));
});

test("link e ricerca: view:rete e view:cervello, Pesi numerici e i pesi che nascono", async () => {
  // REP_LINKS del nucleo: core/repLinks.js (soli dati; representations.js importa three.js, che qui non c'è)
  assert.deepEqual({ ...REP_LINKS }, { rete: "network", cervello: "brain" });
  const views = new Map(searchIndex.filter((e) => e.kind === "view").map((e) => [e.id, e]));
  for (const id of ["rete", "cervello", "numeri", "numeri.training", "neuroni", "neuroni-profondita", "pesi", "architettura", "contesto"]) assert.ok(views.has(id), id);
  // parole chiave utili ma poche: la ricerca globale non deve esserne inondata
  for (const id of ["rete", "cervello", "numeri", "numeri.training"]) assert.ok(views.get(id).keywords.length <= 10, id);
});

test("installRepresentations dopo i pannelli del tab e prima di \"Fonti\" (il pannello Cervello sta prima delle fonti)", async () => {
  const source = await readFile(new URL("../js/tabs/transformer/index.js", import.meta.url), "utf8");
  const create = source.slice(source.indexOf("export async function createTab"));
  const at = (needle) => {
    const i = create.indexOf(needle);
    assert.ok(i >= 0, `manca ${needle}`);
    return i;
  };
  assert.ok(at("installContext(tab);") < at("installReps(tab);"), "dopo l'ultimo pannello del tab (Contesto)");
  assert.ok(at("installReps(tab);") < at('id: "fonti"'), "prima di Fonti");
  assert.match(source, /network: \{[\s\S]*?enter: \(\) => tab\.views\.activate\("neuroni"\)/, "Rete neurale = la vista neuroni del tab (modalità personalizzata)");
  assert.match(source, /brain: \{ load: \(\) => import\("\.\/brain\.js"\)/, "Cervello = livello del nucleo con brain.js");
});
