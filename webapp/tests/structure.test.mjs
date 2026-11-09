// Regole di struttura del codice della web app, controllate sui file (niente DOM, niente three.js):
//   node --test webapp/tests/structure.test.mjs
// - un tab non importa i moduli di un altro tab: la logica che serve a due tab sta in js/shared/ o js/core/
//   (prima il Laboratorio importava tabs/transformer/views/neuronDepthSpec.js; ora js/shared/gpt/depthSpec.js);
// - il nucleo (js/core/), le librerie (js/shared/) e il motore (js/engine/) non importano i tab;
// - nei sorgenti nessun carattere invisibile o bidirezionale VERO (U+202E e simili, "Trojan Source"): dove
//   servono (regole, test) si scrivono come escape \u202e, così chi legge il codice vede quello che gira.
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const JS = fileURLToPath(new URL("../js/", import.meta.url));
const TESTS = fileURLToPath(new URL("./", import.meta.url));

function filesOf(dir, ext = /\.(js|mjs)$/) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...filesOf(path, ext));
    else if (ext.test(name)) out.push(path);
  }
  return out;
}

/** Gli specificatori relativi di import ed export … from, statici e dinamici, risolti in percorsi assoluti. */
function importsOf(file) {
  const text = readFileSync(file, "utf8");
  const specs = [...text.matchAll(/(?:\bimport|\bexport)\s[^;]*?\sfrom\s*["']([^"']+)["']|\bimport\s*\(\s*["']([^"']+)["']\s*\)|^\s*import\s*["']([^"']+)["']/gm)].map((m) => m[1] ?? m[2] ?? m[3]);
  return specs.filter((s) => s.startsWith(".")).map((s) => resolve(dirname(file), s));
}

const area = (path) => {
  const parts = relative(JS, path).split(sep);
  return parts[0] === "tabs" ? `tabs/${parts[1]}` : parts[0];
};

test("struttura: nessun tab importa i moduli di un altro tab", () => {
  const crossing = [];
  for (const file of filesOf(join(JS, "tabs"))) {
    const from = area(file);
    for (const target of importsOf(file)) {
      const to = area(target);
      if (to.startsWith("tabs/") && to !== from) crossing.push(`${relative(JS, file)} → ${relative(JS, target)}`);
    }
  }
  assert.deepEqual(crossing, []);
});

test("struttura: nucleo, librerie condivise e motore non importano i tab", () => {
  const wrong = [];
  for (const dir of ["core", "shared", "engine"]) {
    for (const file of filesOf(join(JS, dir))) {
      for (const target of importsOf(file)) if (area(target).startsWith("tabs/")) wrong.push(`${relative(JS, file)} → ${relative(JS, target)}`);
    }
  }
  assert.deepEqual(wrong, []);
});

test("struttura: niente caratteri invisibili o bidirezionali veri nei sorgenti JS e nei test", () => {
  // controlli C0 (tranne tab, a capo), DEL e C1, trattino morbido, U+061C, U+180E, larghezza zero e segni di
  // direzione (U+200B–U+200F), separatori e controlli bidirezionali (U+2028–U+202E), U+2060–U+206F, U+FEFF
  // eslint-disable-next-line no-control-regex
  const INVISIBLE = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u00ad\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff]/;
  const found = [];
  for (const file of [...filesOf(JS), ...filesOf(TESTS, /\.(mjs|html)$/)]) {
    const lines = readFileSync(file, "utf8").split("\n");
    lines.forEach((line, i) => {
      const m = INVISIBLE.exec(line);
      if (m) found.push(`${relative(join(JS, ".."), file)}:${i + 1} U+${m[0].charCodeAt(0).toString(16).padStart(4, "0")}`);
    });
  }
  assert.deepEqual(found, []);
});
