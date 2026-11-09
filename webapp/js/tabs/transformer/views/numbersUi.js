/**
 * Interfaccia della vista "Pesi numerici" (solo DOM, niente three.js):
 *   sourceSwitch     "Pesi finali (float32) / Training (snapshot)" (radiogroup, segmented.js)
 *   stepControl      cursore sugli step 0…5000 che si aggancia agli snapshot veri, con le tacche
 *   stepChips        un pulsante per ogni snapshot (si sceglie lo step esatto anche col dito)
 *   playButton       "▶ Guarda i pesi nascere" / "❚❚ Pausa"
 *   sourceNote       l'etichetta onesta della sorgente (cosa è vero, arrotondato o stimato)
 *   tensorList       i tensori raggruppati come il codice Python, con forma e numero di valori
 *   numbersBar       la barra in alto a sinistra dello stage (sorgente, step, ▶)
 * Testi sempre con textContent: i nomi dei tensori vengono dai file dei pesi.
 */
import { el, fmt } from "../../../core/dom.js";
import { segmentedSwitch } from "./segmented.js";
import { shapeText, snapToSnapshot } from "./numbersModel.js";

export const SOURCE_OPTIONS = Object.freeze([
  Object.freeze({ id: "final", label: "Pesi finali (float32)", short: "Finali", title: "I numeri esatti che usa il motore della web app: il checkpoint alla fine del training." }),
  Object.freeze({ id: "training", label: "Training (snapshot)", short: "Training", title: "I pesi salvati durante il training su Tiny Shakespeare, dallo step 0 (numeri casuali) allo step 5000." }),
]);

/** Interruttore della sorgente dei pesi (due copie: pannello e barra dello stage, stesso stato). */
export function sourceSwitch({ value, onChange, compact = false }) {
  return segmentedSwitch({ options: SOURCE_OPTIONS, value, onChange, compact, ariaLabel: "Sorgente dei pesi", dataKey: "source" });
}

const INT = (n) => fmt(n);

/**
 * Cursore sugli step del training (0…ultimo) con le tacche degli snapshot veri. Senza interpolazione
 * si aggancia allo snapshot più vicino (lo decide chi ascolta, con trainingSource); con le frecce passa
 * da uno snapshot al successivo (con l'interpolazione: 50 step alla volta, PagSu/PagGiù 500).
 * @returns {{element: HTMLElement, input: HTMLInputElement, set(step: number, text?: string): void, setInterpolate(on: boolean): void}}
 */
export function stepControl({ steps, value, onInput, label = "Step del training", compact = false }) {
  const last = steps[steps.length - 1];
  const input = el("input", { type: "range", min: 0, max: last, step: 1, value, className: "tf-step-range", ariaLabel: label });
  let interpolate = false;
  const ticks = el(
    "div",
    { className: "tf-step-ticks", ariaHidden: "true" },
    ...steps.map((s) => {
      const tick = el("span", { className: `tf-step-tick${s % 1000 === 0 ? " major" : ""}` });
      tick.style.left = `${(s / last) * 100}%`;
      return tick;
    }),
  );
  const labels = compact
    ? null
    : el(
        "div",
        { className: "tf-step-labels", ariaHidden: "true" },
        ...steps
          .filter((s) => s % 1000 === 0)
          .map((s) => {
            const tag = el("span", {}, String(s)); // gli step sono nomi (snap_05000): niente separatore
            tag.style.left = `${(s / last) * 100}%`;
            return tag;
          }),
      );
  const track = el("div", { className: "tf-step-track" }, input, ticks);
  const element = el("div", { className: `tf-step${compact ? " compact" : ""}` }, track, labels);
  input.addEventListener("input", () => onInput(Number(input.value), { live: true }));
  input.addEventListener("change", () => onInput(Number(input.value), { live: false }));
  input.addEventListener("keydown", (e) => {
    const now = Number(input.value);
    let next = null;
    const prev = [...steps].reverse().find((s) => s < now);
    const after = steps.find((s) => s > now);
    const near = steps.indexOf(snapToSnapshot(now, steps)); // senza interpolazione PagSu/PagGiù saltano 3 snapshot
    if (e.key === "ArrowRight" || e.key === "ArrowUp") next = interpolate ? now + 50 : after ?? last;
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") next = interpolate ? now - 50 : prev ?? steps[0];
    else if (e.key === "PageUp") next = interpolate ? now + 500 : steps[Math.min(steps.length - 1, near + 3)];
    else if (e.key === "PageDown") next = interpolate ? now - 500 : steps[Math.max(0, near - 3)];
    else if (e.key === "Home") next = steps[0];
    else if (e.key === "End") next = last;
    if (next === null) return;
    e.preventDefault(); // le frecce sono del cursore: la scena e il Percorso non si muovono
    next = Math.max(steps[0], Math.min(last, next));
    input.value = String(next);
    onInput(next, { live: false });
  });
  return {
    element,
    input,
    set(step, text) {
      if (Number(input.value) !== step) input.value = String(step);
      if (text) input.setAttribute("aria-valuetext", text);
    },
    setInterpolate(on) {
      interpolate = on;
    },
  };
}

/** Un pulsante per ogni snapshot vero: lo step esatto anche dove le tacche sono fitte (0, 50, 100…). */
export function stepChips({ steps, onPick }) {
  const buttons = steps.map((s) => el("button", { type: "button", className: "tf-step-chip", ariaPressed: "false", title: `Snapshot dello step ${s}`, onclick: () => onPick(s) }, String(s)));
  const element = el("div", { className: "tf-step-chips", role: "group", ariaLabel: "Snapshot del training" }, ...buttons);
  return {
    element,
    set(step) {
      steps.forEach((s, i) => buttons[i].setAttribute("aria-pressed", String(s === step)));
    },
  };
}

/** "▶ Guarda i pesi nascere" (e la sua pausa); `short` per la barra dello stage. */
export function playButton({ onClick, short = false, className = "tf-start" }) {
  const idle = short ? "▶ Nascita" : "▶ Guarda i pesi nascere";
  const button = el("button", { type: "button", className, ariaPressed: "false", title: "Guarda i pesi nascere: gli snapshot veri dal rumore iniziale (step 0) ai pesi finali (step 5000), le heatmap e i numeri cambiano davanti a te.", onclick: onClick }, idle);
  return {
    element: button,
    setPlaying(playing) {
      button.textContent = playing ? "❚❚ Pausa" : idle;
      button.setAttribute("aria-pressed", String(playing));
    },
  };
}

/** L'etichetta onesta della sorgente: titolo in evidenza, poi la frase (stima dichiarata in giallo). */
export function sourceNote() {
  const title = el("strong", {});
  const detail = el("p", { className: "hint" });
  const element = el("div", { className: "tf-source-note", ariaLive: "polite" }, title, detail);
  return {
    element,
    set(label) {
      title.textContent = label.title;
      detail.textContent = label.detail;
      element.classList.toggle("estimate", label.estimate);
    },
  };
}

/**
 * Elenco dei tensori raggruppati come il codice Python: un gruppo aperto per l'embedding e l'uscita,
 * un <details> per blocco (22 tensori ciascuno). Ogni voce: percorso Python, forma, numero di valori.
 * @returns {{element: HTMLElement, setActive(name: string|null): void}}
 */
export function tensorList({ groups, onPick }) {
  const items = new Map();
  const sections = groups.map((g) => {
    const list = el(
      "ul",
      { className: "tf-tensor-list" },
      ...g.items.map((item) => {
        const button = el(
          "button",
          { type: "button", className: "tf-tensor-item", dataset: { tensor: item.name }, title: item.name, onclick: () => onPick(item.name) },
          el("span", { className: "tf-tensor-path" }, item.local),
          el("span", { className: "tf-tensor-shape" }, `${shapeText(item.shape)} · ${INT(item.numel)}`),
        );
        items.set(item.name, { button, group: g.id });
        return el("li", {}, button);
      }),
    );
    const summary = el("summary", {}, el("span", { className: "tf-tensor-group" }, g.title), el("small", {}, ` ${INT(g.count)} valori`));
    const details = el("details", { className: "tf-tensor-section", dataset: { group: g.id } }, summary, list);
    if (g.id === "embedding") details.open = true;
    return details;
  });
  const element = el("div", { className: "tf-tensor-groups" }, ...sections);
  return {
    element,
    setActive(name) {
      for (const [n, { button }] of items) {
        const on = n === name;
        button.classList.toggle("active", on);
        if (on) button.setAttribute("aria-current", "true");
        else button.removeAttribute("aria-current");
      }
      // il gruppo del tensore scelto si apre (un clic nella scena può scegliere un tensore di un blocco chiuso)
      const entry = name ? items.get(name) : null;
      if (entry) {
        const details = element.querySelector(`details[data-group="${entry.group}"]`);
        if (details && !details.open) details.open = true;
      }
    },
  };
}

/**
 * Barra in alto a sinistra dello stage (sotto ☰ e il selettore dello zoom): sorgente, step e ▶.
 * Sul telefono la sidebar è un cassetto: qui si scorre il training guardando la scena.
 * Accanto al cursore il cartellino dello step ("Step 1000", in giallo "≈ 750 stima"): senza, con la
 * sidebar chiusa o scorsa altrove la scena mostrerebbe una stima senza dirlo. Per gli screen reader lo
 * step è già nell'aria-valuetext del cursore: il cartellino è solo visivo (aria-hidden).
 */
export function numbersBar({ switchElement, step, play }) {
  const tag = el("span", { className: "tf-bar-step", ariaHidden: "true" });
  const stepBox = el("div", { className: "tf-bar-tools tf-numbers-step" }, step.element, tag, play.element);
  const element = el("div", { className: "tf-numbers-bar", role: "group", ariaLabel: "Sorgente dei pesi della vista Pesi numerici" }, switchElement, stepBox);
  return {
    element,
    setTraining(on) {
      stepBox.hidden = !on;
    },
    /** @param {{tag: string, title: string, estimate: boolean}} label da sourceLabel */
    setLabel(label) {
      tag.textContent = label.tag;
      tag.title = label.title;
      tag.classList.toggle("estimate", label.estimate);
    },
  };
}
