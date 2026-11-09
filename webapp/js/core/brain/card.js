/**
 * Scheda della vista "Cervello" (solo DOM): titolo, etichetta di onestà, scala Regioni / Neuroni,
 * comandi (impulsi, trasparenza della corteccia, piani di taglio, percorsi, etichette, lobi), il
 * dettaglio di ciò che si è scelto (regione, nucleo, gruppo di neuroni), l'elenco delle regioni del tab
 * e la legenda. Sul telefono parte ridotta: titolo, scala e "Anima"; "Comandi" apre il resto.
 * Testi sempre come testo (textContent): la specifica è un dato, mai HTML da interpretare.
 */
import { el, fmt } from "../dom.js";
import { HEMISPHERES, KINDS, LOBES, atlasEntry, atlasKind, formatMni, placements, shortName, sourceLabel } from "./atlas.js";
import { CUTS } from "./anatomy.js";
import { FAMILY, FAMILY_COLORS } from "./circuitLayout.js";
import { NEURON_TYPES } from "./morphology.js";
import { PART_COLORS } from "./neurons.js";
import { REGION_COLORS } from "./regions.js";
import { EDGE_KINDS } from "./spec.js";

const NARROW = "(max-width: 640px)";
const COMPACT_STAGE = 980; // px di stage sotto i quali la scheda parte ridotta (come quella della rete neurale)
export const HONESTY = "Modello stilizzato: forme e posizioni approssimate (coordinate MNI).";
const NEURON_HONESTY =
  "Neuroni generati al computer con forme tipiche (non ricostruzioni di cellule vere); le dimensioni non sono in scala tra i tipi e ogni gruppo mostra pochi neuroni al posto di migliaia o milioni.";

let uid = 0;

export class BrainCard {
  /**
   * @param {HTMLElement} layer
   * @param {object} spec specifica normalizzata
   * @param {object} on gestori: scale(id), play(), opacity(v), cut(axis), cutPosition(mm), flip(), toggle(name, on),
   *   focusRegion(id), selectGroup(id), clear(), frame(), neuronsOf(regionId)
   * @param {object} state stato iniziale { scale, opacity, cut, pathways, labels, lobes }
   */
  constructor(layer, spec, on, state) {
    this.spec = spec;
    this.on = on;
    this.id = `br${++uid}`;
    this.narrow = window.matchMedia(NARROW);
    this.root = el("section", { className: "br-card", ariaLabel: "Cervello 3D: comandi, legenda e dettagli" });
    layer.append(this.root);
    this.build(state);
    this.onNarrow = () => this.setCollapsed(this.narrow.matches);
    this.narrow.addEventListener("change", this.onNarrow);
    // stage stretto (telefono, o finestra media con la sidebar aperta): ridotta, la scena ha più posto
    this.setCollapsed(this.narrow.matches || (layer.clientWidth > 0 && layer.clientWidth < COMPACT_STAGE));
  }

  build(state) {
    const spec = this.spec;
    this.collapseButton = el("button", { type: "button", className: "br-collapse", ariaExpanded: "true", onclick: () => this.setCollapsed(!this.collapsed) }, "Riduci");
    const head = el(
      "header",
      { className: "br-head" },
      el("div", { className: "br-head-text" }, el("h2", { className: "br-title" }, spec.title), el("p", { className: "br-honest" }, HONESTY)),
      this.collapseButton,
    );

    // scala: radiogroup accessibile (frecce, tabindex mobile)
    this.scaleButtons = ["regions", "neurons"].map((id) =>
      el(
        "button",
        {
          type: "button",
          role: "radio",
          className: "br-seg",
          dataset: { scale: id },
          disabled: id === "neurons" && !spec.circuit,
          title: id === "regions" ? "Il cervello intero: le regioni di questo tab, i percorsi e le strutture profonde." : "Il microcircuito: i neuroni modellati, da sinistra a destra come la rete AI.",
          onclick: () => this.on.scale(id),
          onkeydown: (e) => this.onScaleKey(e),
        },
        id === "regions" ? "Regioni" : "Neuroni",
      ),
    );
    const scale = el("div", { className: "br-segmented", role: "radiogroup", ariaLabel: "Scala" }, ...this.scaleButtons);
    this.playButton = el("button", { type: "button", className: "br-btn br-primary", ariaPressed: "false", onclick: () => this.on.play() }, "▶ Anima impulsi");
    this.playButton.title = "Impulsi che viaggiano lungo i percorsi e lungo gli assoni dei neuroni (saltando da un nodo di Ranvier all'altro).";
    const row1 = el("div", { className: "br-buttons" }, scale, this.playButton);

    // comandi della scala Regioni
    const opacityId = `${this.id}-op`;
    this.opacity = el("input", { type: "range", id: opacityId, className: "br-range", min: "0", max: "100", step: "5", value: String(Math.round(state.opacity * 100)), oninput: (e) => this.on.opacity(Number(e.target.value) / 100) });
    this.opacityText = el("span", { className: "br-range-value" }, "");
    const cutId = `${this.id}-cut`;
    this.cutSelect = el(
      "select",
      { id: cutId, className: "br-select", onchange: (e) => this.on.cut(e.target.value || null) },
      el("option", { value: "" }, "Nessun taglio"),
      ...Object.entries(CUTS).map(([id, c]) => el("option", { value: id }, c.label)),
    );
    this.cutRange = el("input", { type: "range", className: "br-range", min: "-60", max: "60", step: "1", ariaLabel: "Posizione del taglio (mm)", oninput: (e) => this.on.cutPosition(Number(e.target.value)) });
    this.cutText = el("span", { className: "br-range-value" }, "");
    this.flipButton = el("button", { type: "button", className: "br-btn", onclick: () => this.on.flip() }, "Inverti");
    this.flipButton.title = "Tiene l'altra metà del cervello.";
    this.cutRow = el("div", { className: "br-row br-cut-row" }, el("span", { className: "br-row-label" }, "Posizione"), this.cutRange, this.cutText, this.flipButton);
    const toggles = el(
      "div",
      { className: "br-toggles" },
      ...[
        ["pathways", "Percorsi"],
        ["labels", "Etichette"],
        ["lobes", "Colori dei lobi"],
      ].map(([name, label]) => {
        const input = el("input", { type: "checkbox", checked: Boolean(state[name]), onchange: (e) => this.on.toggle(name, e.target.checked) });
        this[`${name}Box`] = input;
        return el("label", { className: "br-check" }, input, label);
      }),
    );
    this.regionControls = el(
      "div",
      { className: "br-scale-controls" },
      el("div", { className: "br-row" }, el("label", { className: "br-row-label", htmlFor: opacityId }, "Corteccia"), this.opacity, this.opacityText),
      el("div", { className: "br-row br-row-2" }, el("label", { className: "br-row-label", htmlFor: cutId }, "Taglio"), this.cutSelect),
      this.cutRow,
      toggles,
    );
    this.status = el("p", { className: "br-status", role: "status" });
    this.controls = el("div", { className: "br-controls" }, row1, this.regionControls, this.status);

    // corpo: dettaglio, regioni del tab, circuito, legenda
    this.detail = el("div", { className: "br-detail", hidden: true, tabIndex: -1 });
    this.regionList = el("ul", { className: "br-chips" });
    this.regionSection = el("section", { className: "br-section" }, el("h3", { className: "br-h" }, "Regioni di questo tab"), this.regionList);
    this.circuitSection = el("section", { className: "br-section", hidden: true });
    this.legend = el("section", { className: "br-section br-legend" });
    this.body = el("div", { className: "br-body" }, this.detail, this.regionSection, this.circuitSection, this.legend);
    this.root.replaceChildren(head, this.controls, this.body);
    this.renderRegionList();
    this.renderCircuitSection();
    this.setOpacity(state.opacity);
    this.setCut(state.cut);
    this.setScale(state.scale);
  }

  renderRegionList() {
    const ids = [...this.spec.highlight, ...this.spec.markers];
    this.regionList.replaceChildren(
      ...ids.map((id) => {
        const entry = atlasEntry(id);
        const marker = atlasKind(id) === "marker";
        return el(
          "li",
          {},
          el(
            "button",
            { type: "button", className: "br-chip", dataset: { region: id }, onclick: () => this.on.focusRegion(id) },
            el("i", { className: "br-swatch", style: { background: marker ? REGION_COLORS.marker : REGION_COLORS.tab } }),
            shortName(entry),
          ),
        );
      }),
    );
    for (const p of this.spec.pathways) {
      this.regionList.append(
        el("li", {}, el("button", { type: "button", className: "br-chip", dataset: { pathway: p.id }, onclick: () => this.on.focusRegion(p.id) }, el("i", { className: "br-line", style: { background: p.color } }), p.label)),
      );
    }
  }

  renderCircuitSection() {
    const c = this.spec.circuit;
    if (!c) return;
    // replaceChildren scriverebbe "null" per i pezzi che mancano: si filtrano prima
    const parts = [
      el("h3", { className: "br-h" }, c.title),
      c.text ? el("p", { className: "br-text" }, c.text) : null,
      el(
        "ol",
        { className: "br-groups" },
        ...c.groups.map((g) =>
          el(
            "li",
            {},
            el(
              "button",
              { type: "button", className: "br-group", dataset: { group: g.id }, onclick: () => this.on.selectGroup(g.id) },
              el("span", { className: "br-group-label" }, g.label),
              g.ai ? el("span", { className: "br-group-ai" }, `AI: ${g.ai}`) : null,
            ),
          ),
        ),
      ),
      c.caveat ? el("p", { className: "br-text br-muted" }, c.caveat) : null,
      el("p", { className: "br-honest br-honest-block" }, NEURON_HONESTY),
    ];
    this.circuitSection.replaceChildren(...parts.filter(Boolean));
  }

  /** Legenda della scala attuale (costruita da ciò che c'è davvero nella scena). */
  renderLegend(scale, { focus = [], cut = null } = {}) {
    const key = (color, label, line = false) => el("span", { className: "br-key" }, el("i", { className: line ? "br-line" : "br-swatch", style: { background: color } }), label);
    if (scale === "neurons" && this.spec.circuit) {
      const families = new Set(this.spec.circuit.groups.map((g) => FAMILY[g.type] ?? "excitatory"));
      const kinds = new Set(this.spec.circuit.edges.map((e) => e.kind));
      this.legend.replaceChildren(
        el("h3", { className: "br-h" }, "Legenda"),
        el(
          "div",
          { className: "br-keys" },
          ...[
            ["excitatory", "neurone eccitatorio"],
            ["inhibitory", "neurone inibitorio"],
            ["modulatory", "neurone modulatorio"],
            ["sensory", "cellula sensoriale"],
          ]
            .filter(([f]) => families.has(f))
            .map(([f, label]) => key(FAMILY_COLORS[f], label)),
        ),
        el("div", { className: "br-keys" }, ...[...kinds].map((k) => key(EDGE_KINDS[k].color, `sinapsi ${EDGE_KINDS[k].label}`, true))),
        el("div", { className: "br-keys" }, key(PART_COLORS.myelin, "mielina"), key(PART_COLORS.node, "nodo di Ranvier"), key(PART_COLORS.glow, "impulso")),
        el("ul", { className: "br-kinds" }, ...[...kinds].map((k) => el("li", {}, el("b", {}, `${capitalize(EDGE_KINDS[k].label)}: `), EDGE_KINDS[k].text))),
      );
      return;
    }
    const keys = [key(REGION_COLORS.tab, "regioni di questo tab")];
    if (focus.length) keys.push(key(REGION_COLORS.focus, "regioni dell'analogia scelta"));
    if (this.spec.markers.length) keys.push(key(REGION_COLORS.marker, "nuclei (marcatori)"));
    for (const p of this.spec.pathways) keys.push(key(p.color, p.label, true));
    if (cut) keys.push(key("#efe6d8", "sezione del taglio (sostanza bianca)"));
    this.legend.replaceChildren(
      el("h3", { className: "br-h" }, "Legenda"),
      el("div", { className: "br-keys" }, ...keys),
      el("p", { className: "br-text br-muted" }, "Occhi e coclee sono fuori dal cranio e la coclea è ingrandita, per leggibilità. Clic su una parte del cervello: che cos'è e che cosa fa."),
    );
  }

  // ---------------------------------------------------------------- stato
  setCollapsed(collapsed) {
    this.collapsed = collapsed;
    this.root.classList.toggle("collapsed", collapsed);
    this.collapseButton.setAttribute("aria-expanded", String(!collapsed));
    this.collapseButton.textContent = collapsed ? (this.narrow.matches ? "Comandi" : "Apri") : "Riduci";
  }

  setScale(scale) {
    this.scale = scale;
    for (const b of this.scaleButtons) {
      const on = b.dataset.scale === scale;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    this.regionControls.hidden = scale !== "regions";
    this.regionSection.hidden = scale !== "regions";
    this.circuitSection.hidden = scale !== "neurons";
  }

  onScaleKey(e) {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const enabled = this.scaleButtons.filter((b) => !b.disabled);
    const i = enabled.findIndex((b) => b.dataset.scale === this.scale);
    const next = e.key === "Home" ? 0 : e.key === "End" ? enabled.length - 1 : (i + (e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1) + enabled.length) % enabled.length;
    enabled[next].focus();
    this.on.scale(enabled[next].dataset.scale);
  }

  setPlaying(playing) {
    this.playButton.textContent = playing ? "❚❚ Ferma gli impulsi" : "▶ Anima impulsi";
    this.playButton.setAttribute("aria-pressed", String(playing));
  }

  setOpacity(value) {
    this.opacity.value = String(Math.round(value * 100));
    const text = value >= 0.99 ? "opaca" : value <= 0.01 ? "nascosta" : `${fmt(value * 100)}%`;
    this.opacityText.textContent = text;
    this.opacity.setAttribute("aria-valuetext", `corteccia ${text}`);
  }

  setCut(cut) {
    this.cutSelect.value = cut?.axis ?? "";
    this.cutRow.hidden = !cut;
    if (!cut) return;
    const [min, max] = CUTS[cut.axis].range;
    this.cutRange.min = String(min);
    this.cutRange.max = String(max);
    this.cutRange.value = String(Math.round(cut.mm));
    const axis = cut.axis === "sagittal" ? "x" : cut.axis === "coronal" ? "y" : "z";
    this.cutText.textContent = `${axis} = ${fmt(cut.mm)} mm`;
    this.cutRange.setAttribute("aria-valuetext", `${axis} uguale a ${fmt(cut.mm)} millimetri`);
    this.cutSelect.title = CUTS[cut.axis].hint;
  }

  setToggle(name, on) {
    const box = this[`${name}Box`];
    if (box) box.checked = on;
  }

  setStatus(text) {
    this.status.textContent = text ?? "";
  }

  /** Evidenzia nell'elenco la regione o il gruppo scelti. */
  markActive({ region = null, group = null } = {}) {
    for (const b of this.regionList.querySelectorAll(".br-chip")) b.classList.toggle("active", b.dataset.region === region || b.dataset.pathway === region);
    for (const b of this.circuitSection.querySelectorAll(".br-group")) b.classList.toggle("active", b.dataset.group === group);
  }

  // ---------------------------------------------------------------- dettaglio
  clearDetail() {
    this.detail.hidden = true;
    this.detail.replaceChildren();
    this.markActive();
  }

  /** Dettaglio di una regione, di un nucleo, di un percorso o di un lobo. */
  showRegion(pick) {
    const entry = atlasEntry(pick.id);
    if (!entry) {
      this.renderDetail(pick.kind === "lobe" ? "Lobo" : "Struttura", pick.name ?? pick.id, [
        el("p", { className: "br-text" }, pick.kind === "lobe" ? `In questo punto non c'è una regione dell'atlante del corso: sei sul ${String(pick.name ?? "").toLowerCase()}.` : "Una struttura disegnata per orientarsi (non una regione dell'atlante del corso)."),
      ]);
      return;
    }
    const kind = atlasKind(pick.id);
    const facts = [];
    if (kind === "pathway") {
      facts.push(["Collega", entry.through.map((id) => shortName(atlasEntry(id))).join(" → ")]);
    } else {
      facts.push(["Tipo", KINDS[entry.kind] ?? entry.kind]);
      if (entry.lobe && LOBES[entry.lobe]) facts.push(["Dove", LOBES[entry.lobe].label]);
      facts.push(["Emisfero", HEMISPHERES[entry.hemisphere] ?? entry.hemisphere]);
      const place = placements(entry);
      if (place.length) facts.push(["Coordinate MNI", `${formatMni(place[0].mni)} mm (${sourceLabel(entry)})`]);
    }
    const analogies = this.spec.analogies.filter((a) => a.regions.includes(pick.id) || a.markers.includes(pick.id) || a.pathways.includes(pick.id));
    const groups = this.spec.circuit?.groups.filter((g) => g.region === pick.id || g.marker === pick.id) ?? [];
    const content = [
      el("p", { className: "br-text" }, entry.text),
      el("dl", { className: "br-facts" }, ...facts.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
      entry.note ? el("p", { className: "br-text br-muted br-note" }, entry.note) : null,
    ];
    if (analogies.length) {
      content.push(
        el("h4", { className: "br-h" }, "In questo tab"),
        el("ul", { className: "br-analogies" }, ...analogies.map((a) => el("li", {}, el("b", {}, `${a.ai} ↔ ${a.brain}`), el("span", { className: `br-strength s-${a.strength}` }, a.strengthLabel)))),
        el("p", { className: "br-text br-muted" }, "Il testo completo, con le fonti e i limiti dell'analogia, è nel pannello «Cervello» della barra laterale."),
      );
    } else if (this.spec.highlight.includes(pick.id) || this.spec.markers.includes(pick.id)) {
      content.push(el("p", { className: "br-text br-muted" }, "Fa parte delle regioni di questo tab."));
    } else {
      content.push(el("p", { className: "br-text br-muted" }, "In questo tab non ha un'analogia: è qui per orientarsi."));
    }
    const buttons = [el("button", { type: "button", className: "br-btn", onclick: () => this.on.focusRegion(pick.id) }, "Inquadra")];
    if (groups.length) buttons.push(el("button", { type: "button", className: "br-btn br-primary", onclick: () => this.on.selectGroup(groups[0].id, { scale: true }) }, "Guarda i neuroni"));
    content.push(el("div", { className: "br-buttons" }, ...buttons));
    const role = kind === "pathway" ? "Percorso" : kind === "marker" ? "Nucleo" : "Regione";
    this.renderDetail(role, entry.name, content);
    this.markActive({ region: pick.id });
  }

  /** Dettaglio di un gruppo di neuroni del microcircuito. */
  showGroup(group, { incoming, outgoing }) {
    const type = NEURON_TYPES[group.type];
    const region = group.region ? atlasEntry(group.region) : null;
    const marker = group.marker ? atlasEntry(group.marker) : null;
    const facts = [
      ["Neuroni", type?.label ?? group.type],
      ["Trasmettitore", type?.transmitter ?? "—"],
      ["Disegnati", `${fmt(group.count)}${group.realText ? ` (veri: ${group.realText})` : group.real ? ` (veri: circa ${fmt(group.real)})` : ""}`],
      region ? ["Regione", marker ? `${shortName(region)} · ${shortName(marker)}` : shortName(region)] : null,
    ].filter(Boolean);
    const edgeItem = (e, dir) =>
      el(
        "li",
        {},
        el("i", { className: "br-line", style: { background: EDGE_KINDS[e.kind].color } }),
        el("span", {}, `${dir} ${e.other} · ${EDGE_KINDS[e.kind].label}${e.label ? ` (${e.label})` : ""}`),
        e.note ? el("small", { className: "br-muted" }, e.note) : null,
      );
    const content = [
      type ? el("p", { className: "br-text" }, type.shape) : null,
      el("dl", { className: "br-facts" }, ...facts.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
      group.ai ? el("p", { className: "br-ai" }, el("b", {}, "Nell'AI: "), group.ai) : null,
      group.note ? el("p", { className: "br-text br-muted" }, group.note) : null,
      incoming.length || outgoing.length ? el("ul", { className: "br-edges" }, ...incoming.map((e) => edgeItem(e, "←")), ...outgoing.map((e) => edgeItem(e, "→"))) : null,
      el("div", { className: "br-buttons" }, region ? el("button", { type: "button", className: "br-btn", onclick: () => this.on.focusRegion(group.marker ?? group.region, { scale: true }) }, "Mostra nel cervello") : null, el("button", { type: "button", className: "br-btn", onclick: () => this.on.frameGroup(group.id) }, "Inquadra")),
    ];
    this.renderDetail("Gruppo di neuroni", group.label, content);
    this.markActive({ group: group.id });
  }

  renderDetail(role, title, content) {
    this.detail.replaceChildren(
      el("div", { className: "br-detail-head" }, el("span", { className: "br-role" }, role), el("button", { type: "button", className: "br-close", ariaLabel: "Chiudi il dettaglio", onclick: () => this.on.clear() }, "×")),
      el("h3", { className: "br-detail-title" }, title),
      ...content.filter(Boolean),
    );
    this.detail.hidden = false;
    if (this.collapsed && !this.narrow.matches) this.setCollapsed(false);
    this.body.scrollTop = 0;
  }

  /**
   * Parte dello stage libera dalla scheda, in pixel dello stage: la camera inquadra lì il cervello.
   * Desktop: a destra della scheda; telefono: sotto la scheda.
   */
  freeRect(stage) {
    const W = stage.clientWidth;
    const H = stage.clientHeight;
    const s = stage.getBoundingClientRect();
    const c = this.root.getBoundingClientRect();
    const free = { x0: 12, y0: 56, x1: W - 12, y1: H - 24 };
    if (!c.width || this.root.hidden) return free;
    const right = { ...free, x0: Math.min(W * 0.6, c.right - s.left + 16) };
    const below = { ...free, y0: Math.min(H * 0.6, c.bottom - s.top + 10) };
    const area = (r) => Math.max(0, r.x1 - r.x0) * Math.max(0, r.y1 - r.y0);
    return area(below) >= area(right) ? below : right;
  }

  dispose() {
    this.narrow.removeEventListener("change", this.onNarrow);
    this.root.remove();
  }
}

const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
