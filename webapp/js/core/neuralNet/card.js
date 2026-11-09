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
import { badgeOf, honestyLines, plural, valueSources } from "./honesty.js";
import { ACTIVATION, AXIS_COLORS, HEAD_COLORS, LINK_COLORS, PATTERNS, ROLES } from "./roles.js";

const NARROW = "(max-width: 640px)";
const COMPACT_STAGE = 980; // px di stage sotto i quali la scheda parte ridotta
const NARROW_MAX_VH = 0.52; // altezza massima della scheda sul telefono: la stessa di neuralnet.css (52vh)
/**
 * Rapporto lunghezza / altezza oltre il quale la scheda parte ridotta anche sugli schermi larghi (con
 * `card: "auto"`): la rete di DeepSeek-V4.1-Flash è lunga 240 unità e alta 17, e accanto alla scheda
 * aperta restava una striscia di 40 px. Ridotta, la rete prende tutta la larghezza dello stage.
 */
export const LONG_NETWORK_RATIO = 10;

/** true se la scheda deve partire ridotta (telefono sempre; poi la specifica; poi stage stretto o rete molto lunga). */
export function startsCollapsed({ narrow, stageWidth, mode = "auto", ratio = 0 }) {
  if (narrow || mode === "collapsed") return true;
  if (mode === "open") return false;
  return stageWidth < COMPACT_STAGE || ratio >= LONG_NETWORK_RATIO;
}
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

/** Area di un rettangolo {x0, y0, x1, y1} (0 se vuoto). */
const area = (r) => Math.max(0, r.x1 - r.x0) * Math.max(0, r.y1 - r.y0);

/** Da dove vengono i pesi di attention di una sezione, detto allo studente. */
const ATTENTION_TEXT = Object.freeze({
  real: "veri, calcolati dal modello",
  illustrative: "illustrativi: mostrano lo schema, non vengono da un modello",
  simulated: "simulati con un seme",
});

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
    // prima di canvas e pad nell'ordine del DOM: con Tab dal selettore si arriva subito ai comandi della
    // vista (▶ Anima, spessore, fetta…), non ai 13 pulsanti del pad (la posizione la decide il CSS)
    layer.prepend(this.root);
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
    // badge a tre stati (vere, in parte vere, simulazione): è un pulsante che apre "Scala e onestà",
    // dove c'è l'elenco di che cosa è vero (anche sul telefono, dove il suggerimento del mouse non c'è)
    const sources = valueSources(model);
    const info = badgeOf(sources, net.seed);
    const badge = el("button", { type: "button", className: `nn-badge ${info.className}`, title: info.title, ariaLabel: `${info.text}. ${info.title}`, onclick: () => this.showHonesty() }, info.text);
    this.badge = badge;
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
    this.status = el("p", { className: "nn-status", ariaLive: "polite" });

    // reti lunghe: la vista iniziale è l'ingresso con i primi strati; qui si scorre a parti (o si vede tutta)
    this.walkPrev = el("button", { type: "button", className: "nn-btn nn-walk-step", ariaLabel: "Parte precedente della rete", title: "Parte precedente (verso l'ingresso)", onclick: () => this.on.walk?.(-1) }, "◀");
    this.walkNext = el("button", { type: "button", className: "nn-btn nn-walk-step", ariaLabel: "Parte successiva della rete", title: "Parte successiva (verso l'uscita)", onclick: () => this.on.walk?.(1) }, "▶");
    this.walkLabel = el("span", { className: "nn-walk-label" });
    const whole = el("button", { type: "button", className: "nn-btn", title: "Tutta la rete di lato: si leggono le parti e le ripetizioni (i neuroni sono piccoli)", onclick: () => this.on.whole?.() }, "Tutta la rete");
    this.walk = el("div", { className: "nn-row nn-walk", role: "group", ariaLabel: "Parti della rete", hidden: true }, this.walkPrev, this.walkLabel, this.walkNext, whole);

    const controls = el(
      "div",
      { className: "nn-controls" },
      el("div", { className: "nn-buttons" }, this.playButton, speed, thick, side),
      this.walk,
      sliceRow,
      densityRow,
      this.status,
    );

    this.detail = el("div", { className: "nn-detail", hidden: true, ariaLive: "polite" });
    this.honesty = el(
      "details",
      { className: "nn-more nn-honesty" },
      el("summary", {}, "Scala e onestà"),
      ...honestyLines(sources, net.seed).map((line) => el("p", { className: "nn-text nn-truth" }, line)),
      el("p", { className: "nn-text" }, net.scale || "Rete ridotta."),
      net.note ? el("p", { className: "nn-text" }, net.note) : null,
      el("p", { className: "nn-text nn-muted" }, this.capNote(model)),
    );
    const body = el(
      "div",
      { className: "nn-body" },
      ...paragraphs(net.intro),
      this.detail,
      this.legend(model),
      this.honesty,
      this.index(net),
      net.sources.length ? el("details", { className: "nn-more" }, el("summary", {}, `Fonti (${net.sources.length})`), el("ul", { className: "nn-sources" }, ...net.sources.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, s.title), s.note ? el("span", { className: "nn-muted" }, ` · ${s.note}`) : null)))) : null,
    );
    this.body = body;
    this.root.replaceChildren(head, controls, body);
    // telefono, stage stretto, o rete molto più lunga che alta: si parte ridotti, la scena ha spazio
    const { min, max } = model.layout.bounds;
    const ratio = (max[0] - min[0]) / Math.max(1e-6, max[1] - min[1]);
    this.setCollapsed(startsCollapsed({ narrow: this.narrow.matches, stageWidth: this.layer.clientWidth, mode: net.card, ratio }));
  }

  /** Il badge apre "Scala e onestà": la scheda si apre e l'elenco di che cosa è vero va in vista. */
  showHonesty() {
    if (!this.honesty) return;
    if (this.collapsed) this.setCollapsed(false);
    this.honesty.open = true;
    this.honesty.scrollIntoView?.({ block: "nearest" });
    this.honesty.querySelector("summary")?.focus({ preventScroll: true });
  }

  /**
   * Riga delle parti di una rete lunga: { index, count, title } (index da 0; -1: si vede la rete
   * d'insieme, e ▶ riparte dalla prima parte) o null per nasconderla. I pulsanti ai due estremi si
   * spengono (la rete non continua da quella parte).
   */
  setWalk(walk) {
    if (!this.walk) return;
    this.walk.hidden = !walk;
    if (!walk) return;
    // nella riga il numero; il nome della parte nel suggerimento (e nella riga di stato quando si scorre)
    const whole = walk.index < 0;
    this.walkLabel.textContent = whole ? `Insieme: ${walk.count} parti` : `Parte ${walk.index + 1} di ${walk.count}`;
    this.walkLabel.title = whole ? "Si vede tutta la rete: ▶ riparte dalla prima parte, l'ingresso" : walk.title ? `Parte ${walk.index + 1} di ${walk.count}: ${walk.title}` : "";
    this.walkPrev.disabled = walk.index <= 0;
    this.walkNext.disabled = !whole && walk.index >= walk.count - 1;
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

  /**
   * Dettaglio di una sezione (clic sulla scena o sull'indice). `incoming`/`outgoing`: una voce per
   * sezione collegata (view.js raggruppa i collegamenti verso la stessa sezione); `attention`: da dove
   * vengono i pesi di attention che arrivano qui ("real" | "illustrative" | "simulated" | null).
   */
  showSection(s, { incoming, outgoing, simulated, attention = null }) {
    const role = ROLES[s.role];
    const depth = s.depth.kind === "grid" ? `griglia ${s.depth.rows} × ${s.depth.cols}` : s.depth.count > 1 ? `${s.depth.count} ${s.depth.label}` : "nessuna profondità";
    const drawn = s.role === "experts"
      ? `${plural(s.experts, "esperto", "esperti")}${s.shared ? ` + ${plural(s.shared, "condiviso", "condivisi")}` : ""} × ${plural(s.neurons, "neurone", "neuroni")}, ${plural(s.active, "attivo", "attivi")} per ${s.depth.label || "posizione"}`
      : s.depth.kind === "grid" ? `${plural(s.neurons, "canale", "canali")} × ${s.depth.rows} × ${s.depth.cols}` : `${s.slabs > 1 ? `${s.slabs} lastre × ` : ""}${plural(s.neurons, "neurone", "neuroni")} × ${depth}`;
    const facts = [
      ["Disegnati", `${drawn} = ${fmt(s.units)}`],
      s.real ? ["Nel modello vero", s.depth.kind === "grid" ? plural(s.real, "canale", "canali") : `${fmt(s.real)} per posizione`] : null,
      s.heads > 1 ? ["Head", fmt(s.heads)] : null,
      ["Valori", simulated ? "simulazione didattica" : "veri"],
      attention ? ["Pesi di attention", ATTENTION_TEXT[attention]] : null,
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
   * Parti dello stage libere dalla scheda, in pixel dello stage: a destra della scheda e sotto. La
   * camera inquadra la rete in quella dove viene più grande (view.js le prova tutte e due): una rete
   * lunga e bassa sta meglio sotto una scheda ridotta, larga quanto lo stage, che in una colonna a destra.
   */
  freeRects(stage) {
    const W = stage.clientWidth;
    const H = stage.clientHeight;
    const s = stage.getBoundingClientRect();
    const c = this.root.getBoundingClientRect();
    const free = { x0: 12, y0: 52, x1: W - 12, y1: H - 52 };
    if (!c.width || this.root.hidden) return [free];
    const rightX = c.right - s.left + 14;
    let belowY = c.bottom - s.top + 8;
    let belowLimit = 0.62;
    if (this.narrow.matches) {
      // telefono: la scheda ridotta cresce fino al suo max-height quando un tocco apre un dettaglio; la
      // rete (una striscia bassa) si inquadra sotto quel limite, così il dettaglio non la nasconde
      const top = c.top - s.top;
      const maxH = Math.min(NARROW_MAX_VH * window.innerHeight, H - top - 12);
      belowY = Math.max(belowY, top + maxH + 8);
      belowLimit = 0.72;
    }
    const right = { ...free, x0: Math.min(W * 0.62, rightX) };
    const below = { ...free, y0: Math.min(H * belowLimit, belowY) };
    // una parte tagliata al 62% dello stage passa sopra la scheda (scheda alta: "sotto" non esiste;
    // scheda larga quanto lo stage: "a destra" non esiste): vale solo se nessun'altra è libera
    const clear = [rightX <= W * 0.62 ? right : null, belowY <= H * belowLimit ? below : null].filter(Boolean);
    if (clear.length) return clear;
    return [area(below) >= area(right) ? below : right];
  }

  /** La più grande delle parti libere (per chi non sa la forma di ciò che inquadra). */
  freeRect(stage) {
    return this.freeRects(stage).reduce((best, r) => (area(r) >= area(best) ? r : best)); // a pari area, sotto
  }

  dispose() {
    this.narrow.removeEventListener("change", this.onNarrow);
    this.root.remove();
  }
}
