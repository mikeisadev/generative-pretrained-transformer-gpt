/**
 * Pannello "Cervello" della sidebar (solo DOM, niente three.js): le analogie tra l'architettura del tab
 * e il cervello, ognuna con cosa hanno in comune, DOVE NON REGGE, la forza e le fonti; il microcircuito,
 * i numeri e le fonti. Si legge anche senza aprire il modello 3D, che resta facoltativo.
 *
 *   renderBrainPane(container, brainSpec, { onShow3D, app });
 *   //   onShow3D()            "Mostra il cervello in 3D"
 *   //   onShow3D(["a1", …])   "Mostra nel modello 3D" di un'analogia (regioni, nuclei, percorsi)
 *   //   onShow3D(["circuit"]) "Guarda i neuroni in 3D" (scala Neuroni)
 *
 * Testi sempre come testo (textContent), link solo http(s) e con rel="noopener noreferrer": la
 * specifica è un dato, mai HTML da interpretare. Leggibile nella sidebar da 380 px e sul telefono.
 */
import { el, fmt } from "../dom.js";
import { atlasEntry, shortName } from "./atlas.js";
import { NEURON_TYPES } from "./morphology.js";
import { EDGE_KINDS, STRENGTHS, normalizeBrainSpec } from "./spec.js";

let uid = 0;

/**
 * @param {HTMLElement} container il pannello della sidebar (viene svuotato)
 * @param {object} rawSpec la specifica del tab (brainSpec)
 * @param {{ onShow3D?: (focus?: string[]) => void, app?: object }} [o]
 * @returns {{ spec: object, root: HTMLElement }}
 */
export function renderBrainPane(container, rawSpec, { onShow3D = () => {} } = {}) {
  const spec = normalizeBrainSpec(rawSpec);
  const id = `brp${++uid}`;
  const root = el("div", { className: "br-pane" });
  const show = (focus) => onShow3D(focus);

  // introduzione e pulsante principale (append scriverebbe "null" per un pezzo che manca: si filtra)
  const head = [
    spec.intro ? el("p", { className: "br-pane-intro" }, spec.intro) : null,
    el(
      "div",
      { className: "br-pane-actions" },
      el("button", { type: "button", className: "br-pane-primary", onclick: () => show() }, el("span", { ariaHidden: "true" }, "◉ "), "Mostra il cervello in 3D"),
      spec.circuit ? el("button", { type: "button", className: "br-pane-secondary", onclick: () => show(["circuit"]) }, "Guarda i neuroni in 3D") : null,
    ),
    el("p", { className: "br-pane-note" }, "Il modello 3D è facoltativo: le analogie si leggono anche da qui. Ogni scheda dice anche dove il paragone smette di funzionare."),
  ];
  root.append(...head.filter(Boolean));

  // analogie
  if (spec.analogies.length) {
    const list = el("div", { className: "br-analogy-list" });
    for (const a of spec.analogies) list.append(analogyCard(a, id, show));
    root.append(sectionTitle(`Analogie (${spec.analogies.length})`), strengthLegend(), list);
  }

  // microcircuito
  if (spec.circuit) root.append(circuitSection(spec.circuit, show));

  // numeri
  if (spec.facts.length) {
    root.append(
      sectionTitle("Numeri"),
      el(
        "ul",
        { className: "br-facts-list" },
        ...spec.facts.map((f) =>
          el(
            "li",
            {},
            el("span", { className: "br-fact-claim" }, f.claim),
            f.value ? el("strong", { className: "br-fact-value" }, f.value) : null,
            f.note ? el("span", { className: "br-fact-note" }, f.note) : null,
            f.source ? el("span", { className: "br-fact-source" }, "Fonte: ", sourceLink(f.source)) : null,
          ),
        ),
      ),
    );
  }

  // fonti (tutte, senza doppioni)
  const all = dedupeSources([...spec.sources, ...spec.analogies.flatMap((a) => a.sources), ...(spec.circuit?.sources ?? []), ...spec.facts.map((f) => f.source).filter(Boolean)]);
  if (all.length) {
    root.append(
      el(
        "details",
        { className: "br-pane-sources" },
        el("summary", {}, `Tutte le fonti (${all.length})`),
        el("ul", {}, ...all.map((s) => el("li", {}, sourceLine(s)))),
      ),
    );
  }

  container.replaceChildren(root);
  return { spec, root };
}

function sectionTitle(text) {
  return el("h3", { className: "br-pane-h" }, text);
}

function strengthLegend() {
  return el(
    "details",
    { className: "br-strength-legend" },
    el("summary", {}, "Che cosa vuol dire la forza di un'analogia"),
    el("ul", {}, ...Object.entries(STRENGTHS).map(([key, s]) => el("li", {}, el("span", { className: `br-strength s-${key}` }, s.label), " ", s.text))),
  );
}

/** Una scheda di analogia: AI ↔ cervello, testo, "In comune", "Dove non regge", forza, regioni, fonti. */
function analogyCard(a, id, show) {
  const regionNames = [...a.regions, ...a.markers].map((r) => atlasEntry(r)).filter(Boolean).map(shortName);
  const pathwayNames = a.pathways.map((p) => atlasEntry(p)?.name ?? p);
  const focus = [...a.regions, ...a.markers, ...a.pathways];
  const titleId = `${id}-${a.id}`;
  return el(
    "article",
    { className: "br-analogy", "aria-labelledby": titleId, dataset: { analogy: a.id } },
    el(
      "header",
      { className: "br-analogy-head" },
      el(
        "h4",
        { className: "br-analogy-title", id: titleId },
        el("span", { className: "br-side br-side-ai" }, el("small", {}, "AI"), a.ai),
        el("span", { className: "br-arrow", ariaHidden: "true" }, "↔"),
        el("span", { className: "br-side br-side-brain" }, el("small", {}, "Cervello"), a.brain),
      ),
      el("span", { className: `br-strength s-${a.strength}`, title: STRENGTHS[a.strength].text }, `Analogia ${a.strengthLabel}`),
    ),
    a.text ? el("p", { className: "br-analogy-text" }, a.text) : null,
    a.similar ? el("div", { className: "br-analogy-block br-similar" }, el("h5", {}, "In comune"), el("p", {}, a.similar)) : null,
    a.different ? el("div", { className: "br-analogy-block br-different" }, el("h5", {}, "Dove non regge"), el("p", {}, a.different)) : null,
    regionNames.length || pathwayNames.length ? el("p", { className: "br-analogy-regions" }, el("span", {}, "Nel cervello: "), [...regionNames, ...pathwayNames].join(" · ")) : null,
    el(
      "div",
      { className: "br-analogy-foot" },
      focus.length ? el("button", { type: "button", className: "br-pane-show", onclick: () => show(focus) }, "Mostra nel modello 3D") : null,
      a.sources.length
        ? el("details", { className: "br-analogy-sources" }, el("summary", {}, `Fonti (${a.sources.length})`), el("ul", {}, ...a.sources.map((s) => el("li", {}, sourceLine(s)))))
        : el("span", { className: "br-muted" }, "Nessuna fonte indicata"),
    ),
  );
}

function circuitSection(c, show) {
  const name = (gid) => c.groups.find((g) => g.id === gid)?.label ?? gid;
  return el(
    "section",
    { className: "br-pane-circuit" },
    sectionTitle(c.title),
    c.text ? el("p", { className: "br-pane-text" }, c.text) : null,
    el(
      "ol",
      { className: "br-pane-groups" },
      ...c.groups.map((g) =>
        el(
          "li",
          {},
          el("strong", {}, g.label),
          el("span", { className: "br-pane-type" }, NEURON_TYPES[g.type]?.label ?? g.type, g.realText ? ` · veri: ${g.realText}` : g.real ? ` · veri: circa ${fmt(g.real)}` : ""),
          g.ai ? el("span", { className: "br-pane-ai" }, el("b", {}, "AI: "), g.ai) : null,
        ),
      ),
    ),
    c.edges.length
      ? el(
          "details",
          { className: "br-pane-edges" },
          el("summary", {}, `Le connessioni (${c.edges.length})`),
          el(
            "ul",
            {},
            ...c.edges.map((e) =>
              el(
                "li",
                {},
                el("i", { className: "br-line", style: { background: EDGE_KINDS[e.kind].color }, ariaHidden: "true" }),
                `${name(e.from)} → ${e.from === e.to ? "sé stesso" : name(e.to)} (${EDGE_KINDS[e.kind].label}${e.label ? `, ${e.label}` : ""})`,
                e.note ? el("small", {}, e.note) : null,
              ),
            ),
          ),
        )
      : null,
    c.caveat ? el("p", { className: "br-pane-caveat" }, el("b", {}, "Attenzione: "), c.caveat) : null,
    el("button", { type: "button", className: "br-pane-show", onclick: () => show(["circuit"]) }, "Guarda i neuroni in 3D"),
  );
}

/** "Autori (anno). Titolo. Rivista." con il titolo come link se c'è un URL sicuro. */
function sourceLine(s) {
  const meta = [s.authors, s.year ? `(${s.year})` : ""].filter(Boolean).join(" ");
  return el("span", { className: "br-source" }, meta ? `${meta}. ` : "", sourceLink(s), s.venue ? `. ${s.venue}` : "", s.note ? el("small", { className: "br-source-note" }, s.note) : null);
}

function sourceLink(s) {
  return s.url ? el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, s.title) : el("span", {}, s.title);
}

function dedupeSources(list) {
  const seen = new Set();
  const out = [];
  for (const s of list) {
    if (!s) continue;
    const key = s.url ?? s.title;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out.sort((a, b) => (a.authors || a.title).localeCompare(b.authors || b.title, "it"));
}
