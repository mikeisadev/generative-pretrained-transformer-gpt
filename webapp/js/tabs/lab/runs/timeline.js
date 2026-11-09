/**
 * "Step del training" (solo DOM): il cursore sugli snapshot veri del run A, che aggiorna scena, tabella
 * e statistiche: si vedono i pesi nascere dal rumore (step 0) e prendere forma. Agganciato agli snapshot;
 * l'interpolazione tra snapshot si accende solo a richiesta ed è dichiarata come approssimazione.
 * Più la "targhetta" nella barra dello stage: quale run e quale step sta guardando la scena.
 */
import { el } from "../../../core/dom.js";
import { HONESTY } from "../content.js";
import { shortStepLabel, sliderModel } from "../steps.js";

export function createTimeline(container, tab) {
  const { state } = tab;
  const id = "lab-step-range";
  const range = el("input", { type: "range", id, min: "0", max: "1", step: "1", value: "1", className: "lab-range" });
  const output = el("output", { htmlFor: id, className: "lab-step-value" });
  const prev = el("button", { type: "button", className: "lab-small", ariaLabel: "Snapshot precedente", onclick: () => move(-1) }, "◂");
  const next = el("button", { type: "button", className: "lab-small", ariaLabel: "Snapshot successivo", onclick: () => move(1) }, "▸");
  const interp = el("input", { type: "checkbox", onchange: (e) => void state.setInterpolate(e.target.checked) });
  const ticks = el("div", { className: "lab-ticks", ariaHidden: "true" });
  const note = el("p", { className: "lab-step-note hint", ariaLive: "polite" });
  const noHistory = el("p", { className: "hint" }, "Questo run ha solo i pesi finali: per vedere il training, esportalo con --trace (pannello Run, «Come esportare un run»).");
  const controls = el(
    "div",
    { className: "lab-timeline-controls" },
    el("div", { className: "lab-step-head" }, el("label", { htmlFor: id }, "Step del training"), output),
    el("div", { className: "lab-step-row" }, prev, el("div", { className: "lab-range-box" }, range, ticks), next),
    el("label", { className: "lab-check" }, interp, " Anche gli step intermedi (interpolazione: approssimata)"),
    note,
  );
  const root = el("section", { className: "lab-timeline", ariaLabel: "Step del training" }, controls, noHistory);
  container.append(root);

  let model = sliderModel([]);
  let raf = 0;
  let pending = null;

  function render() {
    const steps = state.stepsOf(state.runA);
    const has = steps.length > 0;
    controls.hidden = !has;
    noHistory.hidden = has;
    if (!has) return;
    model = sliderModel(steps, { interpolate: state.interpolate });
    range.min = String(model.min);
    range.max = String(model.max);
    range.step = String(model.step);
    range.value = String(model.toValue(state.step));
    interp.checked = state.interpolate;
    interp.disabled = steps.length < 2;
    // tacche degli snapshot veri (anche con l'interpolazione: si vede dove stanno i pesi salvati)
    const span = model.max - model.min || 1;
    ticks.replaceChildren(
      ...steps.map((s) => el("span", { className: "lab-tick", style: { left: `${((model.toValue(s) - model.min) / span) * 100}%` } })),
      el("span", { className: "lab-tick lab-tick-final", style: { left: "100%" } }),
    );
    label();
  }

  function label() {
    const view = state.current?.runId === state.runA ? state.current.view : null;
    const text = shortStepLabel(state.step, view);
    output.textContent = text;
    range.setAttribute("aria-valuetext", text);
    const source = view?.source ?? (state.step === null ? "final" : "snapshot");
    note.textContent =
      source === "interpolated"
        ? HONESTY.interpolation
        : source === "final"
          ? "Pesi finali: quelli alla fine del training, in float32 (gli stessi del motore)."
          : state.step === 0
            ? "Step 0: i pesi appena inizializzati, numeri casuali. Il training non ha ancora imparato niente."
            : "Snapshot vero, salvato durante il training (float16).";
  }

  function apply(value) {
    pending = model.toStep(value);
    if (raf) return;
    // il cursore può mandare decine di eventi al secondo: uno per frame basta (gli altri si scavalcano)
    raf = requestAnimationFrame(() => {
      raf = 0;
      void state.setStep(pending);
    });
  }

  /** ◂ ▸: lo snapshot vero precedente o successivo (dopo l'ultimo, i pesi finali). */
  function move(dir) {
    const steps = state.stepsOf(state.runA);
    if (!steps.length) return;
    const stops = [...steps, null];
    const pos = state.step === null ? steps.length : steps.indexOf(state.step);
    let k;
    if (pos >= 0) k = Math.min(stops.length - 1, Math.max(0, pos + dir));
    else k = dir > 0 ? steps.findIndex((s) => s > state.step) : steps.findLastIndex((s) => s < state.step); // da uno step interpolato
    if (k < 0) k = dir > 0 ? stops.length - 1 : 0;
    void state.setStep(stops[k]);
  }

  range.addEventListener("input", () => apply(range.value));
  state.on("runA", render);
  state.on("step", () => {
    range.value = String(model.toValue(state.step));
    label();
  });
  state.on("interpolate", render);
  state.on("weights", label);
  render();

  return {
    root,
    render,
    focus: () => (controls.hidden ? noHistory : range).focus?.(),
    dispose: () => cancelAnimationFrame(raf),
  };
}

/** La targhetta della barra dello stage: «GPT del corso · step 500», apre il pannello Modello. */
export function stageChip(tab) {
  const { state } = tab;
  const name = el("strong", {});
  const step = el("span", {});
  const chip = el("button", { type: "button", className: "lab-chip", title: "Il run e lo step nella scena: clic per cambiarli", onclick: () => tab.openPane("modello", { focus: "step" }) }, name, step);
  const update = () => {
    const entry = state.entry(state.runA);
    name.textContent = entry?.name ?? "";
    const view = state.current?.runId === state.runA ? state.current.view : null;
    const values = state.mode !== "pesi" && state.modeLabel ? ` · ${state.modeLabel}` : "";
    step.textContent = ` · ${shortStepLabel(state.step, view)}${values}`;
    chip.setAttribute("aria-label", `Nella scena: ${entry?.name ?? ""}, ${shortStepLabel(state.step, view)}${values}. Cambia run o step`);
  };
  for (const type of ["runA", "step", "weights", "mode", "runs"]) state.on(type, update);
  update();
  return chip;
}
