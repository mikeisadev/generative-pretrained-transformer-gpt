// Test del pannello "Cervello" della sidebar (js/core/brain/pane.js) con un DOM finto minimo:
//   node --test webapp/tests/*.test.mjs
// Il pannello è solo DOM: si deve caricare senza three.js (qui "three" non si risolve: l'import
// fallirebbe), mostrare ogni testo della specifica come TESTO (mai innerHTML: nel DOM finto lanciare
// è la prova), rendere link solo gli indirizzi http(s), con rel="noopener noreferrer", e chiamare
// onShow3D con il focus giusto da ogni pulsante.
import assert from "node:assert/strict";
import { test } from "node:test";

/** Elemento finto: le proprietà che el() di core/dom.js imposta, gli attributi, i figli, i clic. */
class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attributes = new Map();
    this.listeners = new Map();
    this.dataset = {};
    this.style = {};
    // proprietà "vere" del DOM che el() assegna come proprietà (key in node)
    Object.assign(this, { className: "", id: "", type: "", title: "", href: "", target: "", rel: "", hidden: false, ariaHidden: null, ariaLabel: null, role: null });
  }
  set innerHTML(_) {
    throw new Error("innerHTML vietato nel pannello: i testi della specifica sono dati, non HTML");
  }
  setAttribute(k, v) {
    this.attributes.set(k, String(v));
  }
  addEventListener(type, fn) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  append(...nodes) {
    for (const n of nodes) this.children.push(typeof n === "string" ? { text: n } : n);
  }
  replaceChildren(...nodes) {
    this.children = [];
    this.append(...nodes);
  }
  click() {
    for (const fn of this.listeners.get("click") ?? []) fn({ type: "click", target: this });
  }
  get textContent() {
    return this.children.map((c) => (c instanceof FakeElement ? c.textContent : c.text)).join("");
  }
  *walk() {
    yield this;
    for (const c of this.children) if (c instanceof FakeElement) yield* c.walk();
  }
  find(fn) {
    return [...this.walk()].filter(fn);
  }
}
globalThis.document = { createElement: (tag) => new FakeElement(tag) };

const { renderBrainPane } = await import("../js/core/brain/pane.js");
const { demoBrainSpec } = await import("../js/core/brain/demoSpec.js");

const textNodes = (root) => [...root.walk()].flatMap((e) => e.children.filter((c) => !(c instanceof FakeElement)).map((c) => c.text));

test("pannello Cervello: si carica senza three.js e mostra introduzione, analogie, circuito, fonti", () => {
  const container = new FakeElement("div");
  const calls = [];
  const { spec, root } = renderBrainPane(container, demoBrainSpec, { onShow3D: (focus) => calls.push(focus) });
  assert.equal(container.children[0], root);
  assert.equal(spec.analogies.length, demoBrainSpec.analogies.length);
  const text = root.textContent;
  assert.ok(text.includes(demoBrainSpec.intro));
  for (const a of demoBrainSpec.analogies) {
    assert.ok(text.includes(a.ai) && text.includes(a.brain), a.id);
    assert.ok(text.includes(a.different), `${a.id}: c'è sempre "Dove non regge"`);
  }
  assert.ok(text.includes("Dove non regge"));
  assert.ok(text.includes(demoBrainSpec.circuit.title));
  // pulsanti: cervello intero, neuroni, ogni analogia con le sue regioni
  const buttons = root.find((e) => e.tagName === "BUTTON");
  buttons.find((b) => b.textContent.endsWith("Mostra il cervello in 3D")).click(); // "◉ " è un segno aria-hidden
  buttons.find((b) => b.textContent === "Guarda i neuroni in 3D").click();
  const showAnalogy = buttons.filter((b) => b.textContent === "Mostra nel modello 3D");
  assert.equal(showAnalogy.length, demoBrainSpec.analogies.length);
  showAnalogy[0].click();
  assert.deepEqual(calls[0], undefined);
  assert.deepEqual(calls[1], ["circuit"]);
  for (const id of demoBrainSpec.analogies[0].regions) assert.ok(calls[2].includes(id), `focus dell'analogia: ${id}`);
});

test("pannello Cervello: HTML nei testi resta testo, link solo http(s) e sempre noopener", () => {
  const evil = "<img src=x onerror=alert(1)>";
  const spec = {
    title: evil,
    intro: `Introduzione ${evil}`,
    regions: ["hippocampus"],
    analogies: [
      {
        id: "a",
        ai: `AI ${evil}`,
        brain: "Ippocampo",
        regions: ["hippocampus"],
        text: evil,
        similar: evil,
        different: evil,
        strength: "media",
        sources: [
          { title: "Fonte sicura", url: "https://doi.org/10.1037/0735-7044.100.2.147" },
          { title: "Fonte con javascript", url: "javascript:alert(1)" },
          { title: "Fonte con data", url: "data:text/html,<script>alert(1)</script>" },
          { title: "Fonte relativa", url: "../../segreto" },
        ],
      },
    ],
    facts: [{ claim: evil, value: evil, source: { title: "Numero", url: "vbscript:x" } }],
  };
  const container = new FakeElement("div");
  const { root } = renderBrainPane(container, spec, {});
  // il testo malevolo c'è, ma solo come nodo di testo
  assert.ok(textNodes(root).some((t) => t.includes(evil)));
  assert.equal(root.find((e) => e.tagName === "IMG" || e.tagName === "SCRIPT").length, 0);
  const links = root.find((e) => e.tagName === "A");
  assert.ok(links.length >= 1);
  for (const a of links) {
    assert.match(a.href, /^https?:\/\//, `solo http(s): ${a.href}`);
    assert.equal(a.rel, "noopener noreferrer");
    assert.equal(a.target, "_blank");
  }
  assert.ok(root.textContent.includes("Fonte con javascript"), "la fonte resta, come testo senza link");
});
