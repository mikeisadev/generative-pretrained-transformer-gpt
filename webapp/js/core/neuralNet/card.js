/**
 * Scheda della rete neurale orizzontale (solo DOM): titolo, introduzione, comandi, legenda, dettaglio
 * della sezione scelta, indice delle sezioni e fonti. La legenda si costruisce da ciò che la rete
 * contiene davvero (ruoli, meccanismi, colori presenti), così ogni tab ha la sua senza scriverla.
 *
 * Sul telefono la scheda si riduce a una fila con titolo e comandi principali; "Dettagli" la apre.
 * Testi sempre come testo (textContent): la specifica è un dato, mai HTML da interpretare.
 */
import { el, fmt } from "../dom.js";
import { MODE } from "./activations.js";
import { ACTIVATION, AXIS_COLORS, HEAD_COLORS, LINK_COLORS, PATTERNS, ROLES } from "./roles.js";

const NARROW = "(max-width: 640px)";
const COMPACT_STAGE = 980; // px di stage sotto i quali la scheda parte ridotta
const SPEEDS = [
  { value: 0.5, label: "0,5×" },
  { value: 1, label: "1×" },
  { value: 2, label: "2×" },
];

/** Colore del campione di linea di uno schema nella legenda. */
function patternSwatch(pattern) {
  if (pattern === "attention") return `linear-gradient(90deg, ${HEAD_COLORS.slice(0, 4).join(", ")})`;
  if (pattern === "dense" || pattern === "conv" || pattern === "explicit") return `linear-gradient(90deg, ${ACTIVATION.negative}, ${ACTIVATION.positive})`;
  if (pattern === "residual") return LINK_COLORS.residual;
  if (pattern === "route") return LINK_COLORS.route;
  if (pattern === "lookup") return LINK_COLORS.lookup;
  return LINK_COLORS.structural;
}

const paragraphs = (text, className = "nn-text") => (text ? text.split(/\n{2,}/).map((p) => el("p", { className }, p)) : []);

export class NetworkCard {
  /**
   * @param {HTMLElement} layer dove vive la scheda (il livello della rappresentazione)
   * @param {object} model il modello (buildNetworkModel) oppure null se la specifica è sbagliata
   * @param {object} on gestori: play, speed, slice, density, thick, side, select(kind, id, {frame}), frame(target, how), clear
   */
  constructor(layer, model, on) {
    this.layer = layer;
    this.on = on;
    this.narrow = window.matchMedia(NARROW);
    this.root = el("section", { className: "nn-card", ariaLabel: "Rete neurale: comandi, legenda e dettagli" });
    layer.append(this.root);
    this.onNarrow = () => this.setCollapsed(this.narrow.matches);
    this.narrow.addEventListener("change", this.onNarrow);
    if (model) this.build(model);
  }

  /** La specifica non è valida: si dice dove, per chi scrive il tab (gli studenti non la vedranno mai così). */
  showError(title, errors) {
    this.root.replaceChildren(
      el("header", { className: "nn-head" }, el("h2", { className: "nn-title" }, title)),
      el("p", { className: "nn-text" }, "La rete non si può disegnare: la specifica ha questi problemi."),
      el("ul", { className: "nn-errors" }, ...errors.slice(0, 12).map((e) => el("li", {}, e))),
    );
    this.root.classList.add("nn-card-error");
  }

  build(model) {
    const { net, links } = model;
    this.net = net;
    this.total = links.segments.count;
    const collapse = el("button", { type: "button", className: "nn-collapse", ariaExpanded: "true", onclick: () => this.setCollapsed(!this.collapsed) }, el("span", { className: "nn-collapse-text" }, "Riduci"));
    this.collapseButton = collapse;
    const badge = el("span", { className: `nn-badge ${net.simulated ? "sim" : "real"}` }, net.simulated ? "Attivazioni: simulazione didattica" : "Attivazioni vere");
    badge.title = net.simulated ? `I valori non vengono da un modello: sono generati con un seme (${net.seed}) per mostrare il meccanismo.` : "I valori sono quelli calcolati dal modello.";
    const head = el("header", { className: "nn-head" }, el("div", { className: "nn-head-text" }, el("h2", { className: "nn-title" }, net.title), badge), collapse);

    this.playButton = el("button", { type: "button", className: "nn-btn nn-primary", ariaPressed: "false", onclick: () => this.on.play() }, "▶ Anima");
    this.playButton.title = "Un'onda di attivazione scorre da sinistra a destra: archi di attention che si accendono, esperti scelti per ogni token.";
    const speed = el("select", { className: "nn-select", ariaLabel: "Velocità dell'onda", onchange: (e) => this.on.speed(Number(e.target.value)) }, ...SPEEDS.map((s) => el("option", { value: String(s.value), selected: s.value === 1 }, s.label)));
    const thick = el("button", { type: "button", className: "nn-btn", onclick: () => this.on.thick() }, "Guarda lo spessore");
    thick.title = "Camera di tre quarti: si vede la profondità (token, griglia, tempo) con i meccanismi che la attraversano.";
    const side = el("button", { type: "button", className: "nn-btn", onclick: () => this.on.side() }, "Di lato");
    side.title = "Di nuovo di lato: la rete si legge da sinistra a destra.";

    this.sliceSelect = el(
      "select",
      { className: "nn-select nn-slice", onchange: (e) => this.on.slice(e.target.value === "" ? null : Number(e.target.value)) },
      el("option", { value: "" }, "tutte"),
      ...net.slices.names.map((name, i) => el("option", { value: String(i) }, `${i + 1} · ${name}`)),
    );
    const sliceId = `nn-slice-${Math.random().toString(36).slice(2, 8)}`;
    this.sliceSelect.id = sliceId;
    const sliceRow = net.slices.count > 1
      ? el("label", { className: "nn-row", htmlFor: sliceId }, el("span", { className: "nn-row-label" }, `Fetta (${net.slices.label})`), this.sliceSelect)
      : null;

    this.density = el("input", { type: "range", min: "2", max: "100", step: "1", className: "nn-range", ariaLabel: "Densità dei collegamenti disegnati", oninput: (e) => this.on.density(Number(e.target.value) / 100) });
    this.densityText = el("span", { className: "nn-range-value" });
    const densityRow = this.total > 0 ? el("div", { className: "nn-row" }, el("span", { className: "nn-row-label" }, "Collegamenti"), this.density, this.densityText) : null;
    this.status = el("p", { className: "nn-status" });

    const controls = el(
      "div",
      { className: "nn-controls" },
      el("div", { className: "nn-buttons" }, this.playButton, speed, thick, side),
      sliceRow,
      densityRow,
      this.status,
    );

    this.detail = el("div", { className: "nn-detail", hidden: true, ariaLive: "polite" });
    const body = el(
      "div",
      { className: "nn-body" },
      ...paragraphs(net.intro),
      this.detail,
      this.legend(model),
      el("details", { className: "nn-more" }, el("summary", {}, "Scala e onestà"), el("p", { className: "nn-text" }, net.scale || "Rete ridotta."), net.note ? el("p", { className: "nn-text" }, net.note) : null, el("p", { className: "nn-text nn-muted" }, this.capNote(model))),
      this.index(net),
      net.sources.length ? el("details", { className: "nn-more" }, el("summary", {}, `Fonti (${net.sources.length})`), el("ul", { className: "nn-sources" }, ...net.sources.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, s.title), s.note ? el("span", { className: "nn-muted" }, ` · ${s.note}`) : null)))) : null,
    );
    this.body = body;
    this.root.replaceChildren(head, controls, body);
    // stage stretto (telefono, o finestra media con la sidebar aperta): si parte ridotti, la scena ha spazio
    this.setCollapsed(this.narrow.matches || this.layer.clientWidth < COMPACT_STAGE);
  }

  /** Legenda: assi, colori delle attivazioni presenti, ruoli e meccanismi della rete. */
  legend(model) {
    const { net, values } = model;
    const present = new Set(values.mode);
    const swatch = (color, label, title = "") => el("span", { className: "nn-key", title }, el("i", { className: "nn-swatch", style: { background: color } }), label);
    const activation = [swatch(ACTIVATION.negative, "negativo"), swatch(ACTIVATION.zero, "zero"), swatch(ACTIVATION.positive, "positivo")];
    if (present.has(MODE.probs)) activation.push(swatch(ACTIVATION.probability, "probabilità"));
    if (present.has(MODE.onehot)) activation.push(swatch(ACTIVATION.onehot, "acceso (one-hot)"));
    if (present.has(MODE.off)) activation.push(swatch(ACTIVATION.off, "esperto spento"));
    const roles = [...new Set(net.sections.map((s) => s.role))].map((r) => swatch(ROLES[r].color, ROLES[r].label, ROLES[r].text));
    const patterns = [...new Set(net.links.map((l) => l.pattern))].map((p) =>
      el("li", { className: "nn-pattern" }, el("i", { className: "nn-line", style: { background: patternSwatch(p) } }), el("strong", {}, PATTERNS[p].label), el("span", { className: "nn-muted" }, ` ${PATTERNS[p].text}`)),
    );
    const maxHeads = Math.max(0, ...net.links.filter((l) => l.pattern === "attention").map((l) => l.heads));
    const axes = el(
      "ul",
      { className: "nn-axes" },
      el("li", {}, el("b", { style: { color: AXIS_COLORS.x } }, "x"), ` ${net.axes.x}`),
      el("li", {}, el("b", { style: { color: AXIS_COLORS.y } }, "y"), ` ${net.axes.y}`),
      net.slices.count > 1 ? el("li", {}, el("b", { style: { color: AXIS_COLORS.z } }, "z"), ` ${net.axes.z}: la profondità vera. Di lato la vedi di taglio; gira la camera per vederla.`) : null,
    );
    return el(
      "details",
      { className: "nn-more nn-legend", open: !this.narrow.matches },
      el("summary", {}, "Legenda"),
      el("h3", { className: "nn-h" }, "Assi"),
      axes,
      el("h3", { className: "nn-h" }, "Attivazioni (colore dei neuroni)"),
      el("div", { className: "nn-keys" }, ...activation),
      el("h3", { className: "nn-h" }, "Ruoli delle sezioni"),
      el("div", { className: "nn-keys" }, ...roles),
      patterns.length ? el("h3", { className: "nn-h" }, "Meccanismi (collegamenti)") : null,
      patterns.length ? el("ul", { className: "nn-patterns" }, ...patterns) : null,
      maxHeads > 1 ? el("div", { className: "nn-keys" }, ...HEAD_COLORS.slice(0, Math.min(maxHeads, HEAD_COLORS.length)).map((c, h) => swatch(c, `head ${h + 1}`))) : null,
      net.repeats.length ? el("p", { className: "nn-text nn-muted" }, "× N: un blocco ripetuto è disegnato una volta sola; le lastre trasparenti dopo di lui (i \"fantasmi\") stanno per le altre ripetizioni.") : null,
      el("p", { className: "nn-text nn-muted" }, "Da lontano ogni sezione diventa una lastra colorata: un quadratino per neurone, con gli stessi colori. Avvicinati e tornano i neuroni."),
    );
  }

  /** Indice delle sezioni: anche da tastiera si arriva a ogni parte della rete. */
  index(net) {
    const items = net.items.map((item) =>
      el(
        "li",
        {},
        el(
          "button",
          { type: "button", className: "nn-index-item", onclick: () => this.on.select(item.kind, item.id, { frame: true }) },
          el("i", { className: "nn-swatch", style: { background: item.kind === "repeat" ? "transparent" : item.color, borderColor: item.kind === "repeat" ? "#94a3b8" : "transparent" } }),
          item.kind === "repeat" ? `× ${fmt(item.count)} · ${item.label.replace(/^×\s*[\d.]+\s*/, "")}` : item.label,
        ),
      ),
    );
    return el("details", { className: "nn-more" }, el("summary", {}, `Sezioni (${net.sections.length})`), el("ol", { className: "nn-index" }, ...items));
  }

  capNote(model) {
    const { links } = model;
    const parts = [`${fmt(model.net.units)} neuroni disegnati, ${fmt(links.segments.count)} collegamenti dritti e ${fmt(links.arcs.count)} archi.`];
    if (links.scale < 1) parts.push(`I collegamenti densi sono un campione: per restare fluida la scena ne disegna al più ${fmt(links.segments.count)} (densità ridotta al ${fmt(links.scale * 100, 0)}%).`);
    else parts.push("I collegamenti densi sono un campione (la densità di ogni schema è nella specifica): disegnarli tutti farebbe una macchia.");
    return parts.join(" ");
  }

  // ---------------------------------------------------------------- stato
  setCollapsed(collapsed) {
    if (!this.collapseButton) return;
    this.collapsed = collapsed;
    this.root.classList.toggle("collapsed", collapsed);
    this.collapseButton.setAttribute("aria-expanded", String(!collapsed));
    this.collapseButton.firstChild.textContent = collapsed ? "Dettagli" : "Riduci";
  }

  setPlaying(playing) {
    if (!this.playButton) return;
    this.playButton.textContent = playing ? "❚❚ Pausa" : "▶ Anima";
    this.playButton.setAttribute("aria-pressed", String(playing));
  }

  setStatus(text) {
    if (this.status) this.status.textContent = text;
  }

  setDensity(fraction, drawn) {
    if (!this.density) return;
    this.density.value = String(Math.round(fraction * 100));
    const text = `${fmt(fraction * 100)}% · ${fmt(drawn)}`;
    this.densityText.textContent = text;
    this.density.setAttribute("aria-valuetext", `${fmt(fraction * 100)} per cento, ${fmt(drawn)} collegamenti disegnati su ${fmt(this.total)}`);
  }

  setSlice(t) {
    if (this.sliceSelect) this.sliceSelect.value = t === null ? "" : String(t);
  }

  /** Dettaglio di una sezione (clic sulla scena o sull'indice). */
  showSection(s, { incoming, outgoing, simulated }) {
    const role = ROLES[s.role];
    const depth = s.depth.kind === "grid" ? `griglia ${s.depth.rows} × ${s.depth.cols}` : s.depth.count > 1 ? `${s.depth.count} ${s.depth.label}` : "nessuna profondità";
    const drawn = s.role === "experts"
      ? `${s.experts} esperti${s.shared ? ` + ${s.shared} condiviso` : ""} × ${s.neurons} neuroni, ${s.active} attivi per ${s.depth.label || "posizione"}`
      : s.depth.kind === "grid" ? `${s.neurons} canali × ${s.depth.rows} × ${s.depth.cols}` : `${s.slabs > 1 ? `${s.slabs} lastre × ` : ""}${s.neurons} neuroni × ${depth}`;
    const facts = [
      ["Disegnati", `${drawn} = ${fmt(s.units)}`],
      s.real ? ["Nel modello vero", `${fmt(s.real)} ${s.depth.kind === "grid" ? "canali" : "per posizione"}`] : null,
      s.heads > 1 ? ["Head", fmt(s.heads)] : null,
      ["Valori", simulated ? "simulazione didattica" : "veri"],
      incoming.length ? ["Riceve da", incoming.join(" · ")] : null,
      outgoing.length ? ["Manda a", outgoing.join(" · ")] : null,
    ].filter(Boolean);
    this.renderDetail(
      el("span", { className: "nn-role" }, el("i", { className: "nn-swatch", style: { background: s.color } }), role.label),
      s.label,
      s.sub,
      [...paragraphs(s.text), el("dl", { className: "nn-facts" }, ...facts.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]))],
      { kind: "section", id: s.id },
    );
  }

  showRepeat(rep, shown) {
    const names = rep.of.map((s) => s.label).join(" → ");
    this.renderDetail(
      el("span", { className: "nn-role" }, "Blocco ripetuto"),
      `× ${fmt(rep.count)}`,
      rep.label,
      [...paragraphs(rep.text), el("dl", { className: "nn-facts" }, el("dt", {}, "Blocco"), el("dd", {}, names), el("dt", {}, "Disegnati"), el("dd", {}, `1 vero + ${shown} fantasm${shown === 1 ? "a" : "i"} (al posto di ${fmt(rep.count)} copie)`))],
      { kind: "repeat", id: rep.id },
    );
  }

  showGroup(group, sections) {
    this.renderDetail(el("span", { className: "nn-role" }, "Gruppo"), group.label, "", [...paragraphs(group.text), el("p", { className: "nn-text nn-muted" }, sections.map((s) => s.label).join(" → "))], { kind: "group", id: group.index });
  }

  renderDetail(role, title, sub, content, target) {
    // replaceChildren trasformerebbe un null nel testo "null": si filtra prima
    const children = [
      el(
        "div",
        { className: "nn-detail-head" },
        role,
        el("button", { type: "button", className: "nn-close", ariaLabel: "Chiudi il dettaglio", onclick: () => this.on.clear() }, "×"),
      ),
      el("h3", { className: "nn-detail-title" }, title),
      sub ? el("p", { className: "nn-sub" }, sub) : null,
      ...content,
      el(
        "div",
        { className: "nn-buttons" },
        el("button", { type: "button", className: "nn-btn", onclick: () => this.on.frame(target, "keep") }, "Inquadra"),
        el("button", { type: "button", className: "nn-btn", onclick: () => this.on.frame(target, "thick") }, "Guarda lo spessore"),
      ),
    ];
    this.detail.replaceChildren(...children.filter(Boolean));
    this.detail.hidden = false;
    if (this.collapsed && !this.narrow.matches) this.setCollapsed(false);
    this.detail.scrollIntoView?.({ block: "nearest" });
  }

  clearDetail() {
    if (!this.detail) return;
    this.detail.hidden = true;
    this.detail.replaceChildren();
  }

  /**
   * Parte dello stage libera dalla scheda, in pixel dello stage: la camera inquadra lì la rete.
   * Desktop: a destra della scheda; telefono: sotto la scheda.
   */
  freeRect(stage) {
    const W = stage.clientWidth;
    const H = stage.clientHeight;
    const s = stage.getBoundingClientRect();
    const c = this.root.getBoundingClientRect();
    const free = { x0: 12, y0: 52, x1: W - 12, y1: H - 52 };
    if (!c.width || this.root.hidden) return free;
    // a destra della scheda o sotto: vince la parte più grande (scheda ridotta = bassa = meglio sotto)
    const right = { ...free, x0: Math.min(W * 0.62, c.right - s.left + 14) };
    const below = { ...free, y0: Math.min(H * 0.62, c.bottom - s.top + 8) };
    const area = (r) => Math.max(0, r.x1 - r.x0) * Math.max(0, r.y1 - r.y0);
    return area(below) >= area(right) ? below : right;
  }

  dispose() {
    this.narrow.removeEventListener("change", this.onNarrow);
    this.root.remove();
  }
}
