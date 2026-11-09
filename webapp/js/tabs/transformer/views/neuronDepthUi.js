/**
 * Interfaccia della disposizione "Orizzontale con profondità" (solo DOM, niente three.js):
 *   arrangementSwitch  l'interruttore "A righe / Orizzontale con profondità" (radiogroup accessibile),
 *                      usato due volte: nel riquadro della vista nella sidebar e nella barra dello stage
 *   stageBar           la barra in alto a sinistra dello stage: interruttore e, in profondità, ▶ Anima,
 *                      Guarda lo spessore, Di lato (sul telefono la sidebar è un cassetto chiuso)
 *   depthIntro         cosa si vede: assi, analogia per la profondità, onestà (rete vera, cosa è un campione)
 *   depthControls      quota dei pesi della fetta e comandi della camera, nel riquadro della vista
 *   depthWhere         riga della scheda del neurone: dove sta nella rete orizzontale
 *   depthLegend        colori dei neuroni, ruoli delle sezioni, collegamenti, head
 * Testi sempre come testo (textContent): i nomi dei token vengono dal prompt scritto dallo studente.
 */
import { el, fmt } from "../../../core/dom.js";
import { LINK_LIMITS } from "../../../core/neuralNet/links.js";
import { ACTIVATION, AXIS_COLORS, HEAD_COLORS, LINK_COLORS, ROLES } from "../../../core/neuralNet/roles.js";
import { DEPTH_FRACTIONS } from "./neuronDepthSpec.js";

/** Le due disposizioni della vista Rete neurale (id usati anche da tab.neuronsView e dall'evento "neurons-layout"). */
export const ARRANGEMENTS = Object.freeze([
  Object.freeze({ id: "rows", label: "A righe", short: "Righe", title: "La disposizione di sempre: una riga per blocco, dal basso verso l'alto, un token alla volta." }),
  Object.freeze({ id: "depth", label: "Orizzontale con profondità", short: "Profondità", title: "La rete vera da sinistra a destra, con tutti i token del prompt in profondità: gira la camera per vedere lo spessore." }),
]);

const SVG = "http://www.w3.org/2000/svg";
/** Icone disegnate con il DOM (niente innerHTML): righe impilate, oppure lastre in fila con lo spessore. */
function icon(id) {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 20 16");
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("tf-arrange-icon");
  const shapes = id === "rows"
    ? [[2, 2, 16, 3], [2, 6.5, 16, 3], [2, 11, 16, 3]]
    : [[1.5, 4, 3, 10], [6.5, 3, 3, 10], [11.5, 2, 3, 10], [16, 1, 2.5, 10]];
  for (const [x, y, w, h] of shapes) {
    const r = document.createElementNS(SVG, "rect");
    for (const [k, v] of Object.entries({ x, y, width: w, height: h, rx: 0.8 })) r.setAttribute(k, String(v));
    svg.append(r);
  }
  return svg;
}

/**
 * Interruttore della disposizione: un radiogroup con due voci (frecce, Home/Fine, tabindex mobile).
 * Spazio e Invio sono della voce: niente pausa del player né passo del Percorso (preventDefault).
 * @returns {{ element: HTMLElement, set(id: string): void }}
 */
export function arrangementSwitch({ value, onChange, compact = false }) {
  const group = el("div", { className: `tf-arrange${compact ? " compact" : ""}`, role: "radiogroup", ariaLabel: "Disposizione della rete neurale" });
  const buttons = ARRANGEMENTS.map((a) =>
    el(
      "button",
      { type: "button", role: "radio", className: "tf-arrange-option", dataset: { arrangement: a.id }, title: a.title, onclick: () => choose(a.id) },
      icon(a.id),
      el("span", { className: "tf-arrange-label" }, a.label),
      el("span", { className: "tf-arrange-short" }, a.short),
    ),
  );
  group.append(...buttons);
  const choose = (id, focus = false) => {
    set(id);
    if (focus) buttons.find((b) => b.dataset.arrangement === id)?.focus();
    onChange(id);
  };
  group.addEventListener("keydown", (e) => {
    const i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    const keys = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 };
    let next = null;
    if (e.key in keys) next = (i + keys[e.key] + buttons.length) % buttons.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = buttons.length - 1;
    else if (e.key === " " || e.key === "Enter") next = i;
    if (next === null) return;
    e.preventDefault(); // la pagina non scorre, il Percorso non cambia passo, il player non va in pausa
    choose(buttons[next].dataset.arrangement, true);
  });
  function set(id) {
    for (const b of buttons) {
      const on = b.dataset.arrangement === id;
      b.setAttribute("aria-checked", String(on));
      b.classList.toggle("active", on);
      b.tabIndex = on ? 0 : -1;
    }
  }
  set(value);
  return { element: group, set };
}

/**
 * Barra dello stage (in alto a sinistra, sotto ☰): l'interruttore sempre, i comandi della camera e
 * dell'onda solo in profondità. Le etichette si accorciano sugli schermi stretti (CSS).
 */
export function stageBar({ switchElement, onPlay, onThick, onSide }) {
  const play = el("button", { type: "button", className: "tf-bar-btn tf-bar-primary", ariaPressed: "false", title: "Un'onda di attivazione scorre da sinistra a destra: archi di attention che si accendono in profondità.", onclick: onPlay }, "▶ Anima");
  const thick = el("button", { type: "button", className: "tf-bar-btn", title: "Camera di tre quarti: si vedono i token in profondità e i meccanismi che li collegano.", onclick: onThick }, el("span", { className: "tf-arrange-label" }, "Guarda lo spessore"), el("span", { className: "tf-arrange-short" }, "Spessore"));
  const side = el("button", { type: "button", className: "tf-bar-btn", title: "Di nuovo di lato: la rete si legge da sinistra a destra.", onclick: onSide }, "Di lato");
  const tools = el("div", { className: "tf-bar-tools", role: "group", ariaLabel: "Comandi della disposizione orizzontale" }, play, thick, side);
  const element = el("div", { className: "tf-neuron-bar" }, switchElement, tools);
  return {
    element,
    setMode(arrangement) {
      tools.hidden = arrangement !== "depth";
    },
    setPlaying(playing) {
      play.textContent = playing ? "❚❚ Pausa" : "▶ Anima";
      play.setAttribute("aria-pressed", String(playing));
    },
  };
}

/**
 * Cosa si vede nella disposizione orizzontale: introduzione dalla specifica, assi, analogia per la
 * profondità e "Onestà" (rete vera, cosa è un campione). `spec` è la networkSpec costruita per il prompt.
 */
export function depthIntro(spec, counts) {
  const T = spec.depth.count;
  const axis = (key, text) => el("li", {}, el("b", { style: { color: AXIS_COLORS[key] } }, key), ` ${text}`);
  return [
    el("p", {}, spec.intro),
    el(
      "ul",
      { className: "tf-axes" },
      axis("x", "flusso: le sezioni da sinistra a destra, dall'ingresso alle probabilità"),
      axis("y", "i neuroni della sezione, dall'alto in basso"),
      axis("z", `le ${T} posizioni del prompt: il token 0 in fondo, l'ultimo davanti (di lato vedi quello)`),
    ),
    el(
      "p",
      { className: "tf-analogy" },
      el("b", {}, "Come immaginarla. "),
      `Una fabbrica con ${T} nastri trasportatori affiancati, uno per carattere del prompt. Ogni stazione (sezione) lavora su tutti i nastri insieme con gli stessi attrezzi (gli stessi pesi): di lato vedi una catena sola, girando la camera vedi i ${T} nastri uno dietro l'altro. L'attention è l'unico punto in cui i nastri si parlano: ogni carattere guarda indietro quelli prima di lui, mai quelli dopo.`,
    ),
    el(
      "details",
      { className: "deep tf-honesty" },
      el("summary", {}, "È la rete vera: cosa vedi davvero"),
      el("p", {}, `Ogni sfera è un neurone vero del modello: ${fmt(counts.perPosition)} per token, ${fmt(counts.units)} in tutto, con il valore che il motore ha calcolato per questo prompt. Sono gli stessi neuroni della disposizione a righe (lì ${fmt(counts.rowsCount)}, perché il vettore x è ripetuto a inizio riga: ${fmt(counts.copies)} copie).`),
      el("p", {}, `Ogni arco di attention è un peso di attention vero, calcolato dal motore confrontando query e key: per ogni token e ogni head si vedono i ${LINK_LIMITS.arcKeysPerQuery} più forti (sopra il ${LINK_LIMITS.arcMinWeight * 100}%), dalla key e dal value del token guardato all'uscita del token che guarda.`),
      el("p", {}, spec.note),
      el("p", {}, spec.scale),
    ),
  ];
}

/** Comandi nel riquadro della vista: quota dei pesi della fetta (i più forti di ogni matrice) e camera. */
export function depthControls({ fraction, onFraction, onPlay, onThick, onSide, onWhole }) {
  const index = Math.max(0, DEPTH_FRACTIONS.indexOf(fraction));
  const range = el("input", { type: "range", min: 0, max: DEPTH_FRACTIONS.length - 1, step: 1, value: index, ariaLabel: "Quota dei pesi più forti di ogni matrice, disegnati sulla fetta scelta" });
  const out = el("output");
  // si ricostruisce la rete: al rilascio del cursore (change), non a ogni passo del trascinamento
  range.addEventListener("input", () => (out.textContent = pct(DEPTH_FRACTIONS[Number(range.value)])));
  range.addEventListener("change", () => onFraction(DEPTH_FRACTIONS[Number(range.value)]));
  const play = el("button", { type: "button", ariaPressed: "false", onclick: onPlay }, "▶ Anima");
  const element = el(
    "div",
    { className: "tf-depth-controls" },
    el("div", { className: "tf-row" }, el("label", {}, "Pesi della fetta"), range, out),
    el("div", { className: "tf-row" }, play, el("button", { type: "button", onclick: onThick }, "Guarda lo spessore"), el("button", { type: "button", onclick: onSide }, "Di lato"), el("button", { type: "button", onclick: onWhole }, "Tutta la rete")),
  );
  return {
    element,
    setFraction(f, shown) {
      range.value = String(Math.max(0, DEPTH_FRACTIONS.indexOf(f)));
      out.textContent = shown === undefined ? pct(f) : `${pct(f)} · ${fmt(shown)}`;
      range.setAttribute("aria-valuetext", `${pct(f)} dei pesi di ogni matrice${shown === undefined ? "" : `, ${fmt(shown)} collegamenti`}`);
    },
    setPlaying(playing) {
      play.textContent = playing ? "❚❚ Pausa" : "▶ Anima";
      play.setAttribute("aria-pressed", String(playing));
    },
  };
}

const pct = (f) => `${(f * 100).toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`;

/**
 * Riga della scheda del neurone in profondità: dove sta il neurone nella rete orizzontale (sezione,
 * lastra, indice nella lastra, head, token). La scheda numera i neuroni come la disposizione a righe
 * (query, key e value sono una colonna sola da 192): questa riga lega i due numeri.
 * @param {ReturnType<import("./neuronDepthSpec.js").unitDescription>} d
 */
export function depthWhere(d) {
  const parts = [d.neuron, d.head !== null ? `head ${d.head}` : null, d.token].filter(Boolean).join(" · ");
  return el("p", { className: "tf-depth-where" }, "Nella rete orizzontale: ", el("b", {}, d.title), ` · ${parts}`);
}

/** Legenda della disposizione orizzontale (colori del renderer del nucleo, roles.js). */
export function depthLegend(heads) {
  const roles = ["input", "embedding", "norm", "attention", "mlp", "add", "output"];
  const item = (color, label, kind = "") => {
    const sw = el("span", { className: `tf-swatch ${kind}`.trim() });
    sw.style.setProperty("--sw", color); // le proprietà CSS personalizzate non passano da Object.assign
    return el("li", {}, sw, el("span", {}, label));
  };
  const list = (items) => el("ul", { className: "tf-legend" }, ...items);
  return el(
    "div",
    { className: "tf-depth-legend" },
    el("h4", {}, "Neuroni (colore = valore vero)"),
    list([
      item(ACTIVATION.positive, "positivo", "ring"),
      item(ACTIVATION.negative, "negativo", "ring"),
      item(ACTIVATION.zero, "zero (per esempio la ReLU che azzera)", "ring"),
      item(ACTIVATION.probability, "probabilità (softmax)", "ring"),
      item(ACTIVATION.onehot, "acceso (one-hot)", "ring"),
    ]),
    el("h4", {}, "Sezioni"),
    list(roles.map((r) => item(ROLES[r].color, ROLES[r].label))),
    el("h4", {}, "Collegamenti"),
    list([
      item(ACTIVATION.positive, "peso vero positivo della fetta scelta (più luce = |peso| più grande)", "line"),
      item(ACTIVATION.negative, "peso vero negativo", "line"),
      item(LINK_COLORS.lookup, "lookup: il carattere (o la posizione) sceglie la sua riga della tabella", "line"),
      item(LINK_COLORS.structural, "collegamento fisso, non appreso, token per token: somme (peso 1), LayerNorm e softmax", "line"),
      item(LINK_COLORS.residual, "connessione residua: un arco per token sopra il blocco", "line"),
      ...HEAD_COLORS.slice(0, heads).map((c, h) => item(c, `attention della head ${h} (luce = peso)`, "line")),
    ]),
    el("p", { className: "hint" }, "Da lontano ogni sezione diventa una lastra colorata: un quadratino per neurone e per token, con gli stessi colori. Avvicinati e tornano le sfere."),
  );
}
