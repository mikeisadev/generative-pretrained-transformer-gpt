/**
 * Pannello "Confronto" (solo DOM): A e B (run e step), i confronti pronti, il riepilogo dell'abbinamento,
 * la differenza A − B nella scena con la sua legenda, la tabella dei tensori e gli istogrammi sovrapposti
 * del tensore scelto. Legge il confronto vivo (comparison.js) e la modalità differenza (diffMode.js); il
 * prossimo carattere su un prompt sta in promptBlock.js. Testi dei file dello studente: solo textContent.
 *
 *   const pane = createComparePane(element, { tab, comparison, diff, onPreset, onDiffToggle });
 */
import { el } from "../../../core/dom.js";
import { colormapLUT } from "../../../core/matrixNumbers.js";
import { Histogram } from "../../../core/ui/histogram.js";
import { tensorStats } from "../../../shared/weights/stats.js";
import { shortStepLabel } from "../steps.js";
import { COURSE_RUN_ID } from "../state.js";
import { matchSummary, onlyHalfRounding, sortRows } from "./compareModel.js";
import { createCompareTable } from "./compareTable.js";
import { COMPARE, DIFF, HISTOGRAM } from "./content.js";
import { int, num, pct } from "./format.js";
import { presets } from "./presets.js";

const NONE = "";

export function createComparePane(pane, { tab, comparison, diff, onPreset, onDiffToggle = () => diff.toggle() }) {
  const { state } = tab;

  // ---------------------------------------------------------------- A e B
  const pickA = picker("A", "nella scena");
  const pickB = picker("B", "termine di paragone");
  const swap = el("button", { type: "button", className: "lab-small lab-an-swap", onclick: () => void swapAB(), ariaLabel: "Scambia A e B" }, "⇅ Scambia A e B");
  const presetBox = el("div", { className: "lab-an-presets", role: "group", ariaLabel: "Confronti pronti" });
  const status = el("div", { className: "lab-an-status", role: "status", ariaLive: "polite" });

  // ---------------------------------------------------------------- differenza nella scena
  const diffButton = el("button", { type: "button", className: "lab-an-toggle", onclick: () => onDiffToggle(), disabled: true }, el("span", { className: "lab-an-switch", ariaHidden: "true" }), DIFF.toggle);
  diffButton.setAttribute("aria-pressed", "false");
  const scaleInputs = Object.entries({ common: [DIFF.scaleCommon, DIFF.scaleCommonHelp], weights: [DIFF.scaleWeights, DIFF.scaleWeightsHelp] }).map(([value, [label, help]]) => {
    const input = el("input", { type: "radio", name: "lab-an-scale", value, checked: diff.scale === value, onchange: () => diff.setScale(value) });
    return { value, input, label: el("label", { className: "lab-an-radio", title: help }, input, ` ${label}`) };
  });
  const legend = el("div", { className: "lab-an-legend", hidden: true });
  const diffNote = el("p", { className: "hint lab-an-diff-note", ariaLive: "polite" });
  const diffBox = el("section", { className: "lab-an-diff", ariaLabel: "Differenza nella scena" }, diffButton, el("div", { className: "lab-an-scales", role: "radiogroup", ariaLabel: "Scala dei colori della differenza" }, ...scaleInputs.map((s) => s.label)), legend, diffNote);

  // ---------------------------------------------------------------- tabella e istogrammi
  const tableBox = el("div", { className: "lab-an-table-box" });
  const table = createCompareTable(tableBox, { onSelect: (name) => chooseTensor(name) });
  const matchBox = el("details", { className: "lab-details lab-an-match" }, el("summary", {}, "Che cosa si abbina e che cosa no"), el("p", { className: "hint" }, COMPARE.match), el("ul", { className: "lab-notes" }));
  const histTitle = el("p", { className: "lab-an-hist-name mono" });
  const histNote = el("p", { className: "hint lab-an-hist-note" });
  const histBox = el("div", { className: "lab-an-hist" });
  const histStats = el("dl", { className: "lab-an-hist-stats" });
  const hist = new Histogram(histBox, { bins: 60, height: 130, label: "" });
  const histSection = el("section", { className: "lab-an-hist-section", ariaLabel: HISTOGRAM.title }, el("h3", { className: "lab-h3" }, HISTOGRAM.title), el("p", { className: "hint" }, HISTOGRAM.hint), histTitle, histBox, histStats, histNote);
  const compared = el("div", { className: "lab-an-compared", hidden: true }, diffBox, el("h3", { className: "lab-h3" }, "Tensore per tensore"), tableBox, matchBox, histSection);
  const emptyNote = el("p", { className: "lab-empty" }, COMPARE.empty);

  const intro = el(
    "div",
    { className: "lab-an-intro" },
    el("p", { className: "lab-intro" }, COMPARE.intro),
    el("details", { className: "lab-details" }, el("summary", {}, "Un'analogia"), el("p", {}, COMPARE.analogy)),
  );
  const promptSlot = el("div", { className: "lab-an-prompt-slot" });
  pane.append(intro, el("div", { className: "lab-an-picks" }, pickA.root, pickB.root, swap), el("h3", { className: "lab-h3" }, "Confronti pronti"), presetBox, status, emptyNote, compared, promptSlot);

  // ---------------------------------------------------------------- comportamento
  function picker(side, role) {
    const run = el("select", { className: "lab-an-select", onchange: () => void pickRun(side, run.value) });
    const step = el("select", { className: "lab-an-select", onchange: () => void pickStep(side, step.value) });
    run.setAttribute("aria-label", `Run ${side}`);
    step.setAttribute("aria-label", `Step del training di ${side}`);
    const root = el("fieldset", { className: `lab-an-pick lab-an-pick-${side.toLowerCase()}` }, el("legend", {}, el("span", { className: `lab-an-tag lab-an-tag-${side.toLowerCase()}` }, side), ` ${role}`), run, step);
    return { root, run, step };
  }

  /** Gli step di un run: quelli in memoria, o quelli dichiarati da un run salvato nel browser non ancora letto. */
  const stepsOf = (id) => (state.run(id) ? state.stepsOf(id) : state.entry(id)?.summary?.steps ?? []);
  const stepValue = (step) => (step === null || step === undefined ? "finali" : String(step));
  const parseStep = (value) => (value === "finali" ? null : Number(value));

  function fillPicker(p, { id, step, allowNone }) {
    const entries = state.entries;
    p.run.replaceChildren(...(allowNone ? [el("option", { value: NONE }, "— nessuno: niente confronto —")] : []), ...entries.map((e) => el("option", { value: e.id }, e.name)));
    p.run.value = id ?? NONE;
    const steps = id ? stepsOf(id) : [];
    const options = [el("option", { value: "finali" }, "pesi finali")];
    for (const s of steps) options.push(el("option", { value: String(s) }, `step ${int(s)}${s === 0 ? " (inizio)" : ""}`));
    if (step !== null && step !== undefined && !steps.includes(step)) options.push(el("option", { value: String(step) }, `≈ step ${int(step)} (interpolato)`));
    p.step.replaceChildren(...options);
    p.step.value = stepValue(step);
    p.step.disabled = !id || steps.length === 0;
  }

  function renderPickers() {
    fillPicker(pickA, { id: state.runA, step: state.step, allowNone: false });
    fillPicker(pickB, { id: state.runB, step: state.stepB, allowNone: true });
    swap.disabled = !state.runB;
    renderPresets();
  }

  let presetKey = "";
  function renderPresets() {
    const list = presets({ runA: state.runA, courseId: COURSE_RUN_ID, stepsOf: (id) => stepsOf(id), entries: state.entries });
    // si ridisegnano solo se cambiano: un pulsante appena premuto da tastiera non deve sparire sotto il fuoco
    const key = JSON.stringify(list.map((p) => [p.id, p.text]));
    if (key === presetKey) return;
    presetKey = key;
    const focused = presetBox.contains(document.activeElement) ? document.activeElement.dataset.preset : null;
    presetBox.replaceChildren(
      ...list.map((p) =>
        el("button", { type: "button", className: "lab-an-preset", dataset: { preset: p.id }, onclick: () => void onPreset(p) }, el("strong", {}, p.title), el("span", {}, p.text)),
      ),
    );
    if (focused) presetBox.querySelector(`[data-preset="${focused}"]`)?.focus({ preventScroll: true });
    presetBox.hidden = !list.length;
  }

  async function pickRun(side, id) {
    tab.ui.director.noteUserIntent(); // cambiando A la camera inquadra il nuovo modello
    if (side === "A") await state.setRunA(id, { step: null });
    else await state.setRunB(id || null, { step: null });
  }

  async function pickStep(side, value) {
    const step = parseStep(value);
    if (side === "A") await state.setStep(step);
    else state.setStepB(step);
  }

  async function swapAB() {
    if (!state.runB) return;
    const a = { run: state.runA, step: state.step };
    const b = { run: state.runB, step: state.stepB };
    tab.ui.director.noteUserIntent();
    await state.setRunA(b.run, { step: b.step });
    await state.setRunB(a.run, { step: a.step });
  }

  /** Una riga della tabella: il tensore diventa la scelta (scena e pannello Tensore), la camera ci vola. */
  function chooseTensor(name) {
    state.select(name);
    tab.ui.director.noteUserIntent();
    void tab.scene.focusTensor(name);
  }

  // ---------------------------------------------------------------- confronto pronto
  let snapshot = comparison.current;
  let matchPair = "";
  function renderComparison(s) {
    // durante un ricalcolo (es. scorrendo gli step) resta il risultato di prima, un po' velato: niente
    // tabella che sparisce e ricompare, niente scorrimento perso
    if (s.status === "busy" && snapshot.status === "ready") {
      compared.classList.add("is-busy");
      return;
    }
    compared.classList.remove("is-busy");
    snapshot = s;
    const ready = s.status === "ready";
    emptyNote.hidden = s.status !== "none";
    compared.hidden = !ready;
    diffButton.disabled = !ready;
    if (s.status === "none") status.replaceChildren();
    else if (s.status === "busy") status.replaceChildren(el("p", { className: "lab-busy" }, "Confronto i pesi…"));
    else if (s.status === "error") status.replaceChildren(el("p", { className: "lab-error" }, `Confronto non riuscito: ${s.message}`));
    if (!ready) return;
    const c = s.result;
    const m = matchSummary(c);
    const lines = [
      el("p", { className: "lab-an-ab" }, el("span", { className: "lab-an-tag lab-an-tag-a" }, "A"), ` ${s.a.name} · ${sideLabel(s.a)}`),
      el("p", { className: "lab-an-ab" }, el("span", { className: "lab-an-tag lab-an-tag-b" }, "B"), ` ${s.b.name} · ${sideLabel(s.b)}`),
    ];
    if (c.identical) lines.push(el("p", { className: "lab-ok-line" }, COMPARE.identical));
    else if (m.matched) {
      lines.push(el("p", { className: "lab-an-total-line" }, `${int(m.matched)} tensori confrontati (${int(c.matchedParams)} pesi): cambio totale `, el("strong", {}, pct(c.totals.relChange)), ` · coseno `, el("strong", {}, num(c.totals.cosine))));
    } else lines.push(el("p", { className: "lab-error" }, "Nessun tensore con lo stesso nome e la stessa forma: A e B non si confrontano cella per cella (vedi sotto)."));
    if (onlyHalfRounding(c, s.a.dtype, s.b.dtype)) lines.push(el("p", { className: "lab-an-callout" }, COMPARE.float16));
    if (!m.complete) lines.push(el("p", { className: "lab-arch-note" }, m.lines.join(" ")));
    status.replaceChildren(...lines);
    matchBox.querySelector("ul").replaceChildren(...(m.lines.length ? m.lines : ["Tutti i tensori di A hanno un gemello della stessa forma in B, e viceversa."]).map((t) => el("li", {}, t)));
    // aperto da solo quando qualcosa non si abbina, ma solo cambiando run: scorrendo gli step resta come l'ha lasciato lo studente
    const pair = `${s.a.runId}|${s.b.runId}`;
    if (pair !== matchPair) matchBox.open = !m.complete;
    matchPair = pair;
    table.setData(c, { selected: state.tensor });
    renderHistogram();
  }

  function sideLabel(side) {
    return side.view?.label ?? shortStepLabel(side.step);
  }

  // ---------------------------------------------------------------- istogrammi sovrapposti
  /** Il tensore degli istogrammi: quello scelto in A, altrimenti il più cambiato (in cima alla tabella). */
  function histogramTensor() {
    const c = snapshot.result;
    if (!c) return null;
    const chosen = state.tensor;
    if (chosen && snapshot.a.view?.tensors.has(chosen)) return chosen;
    const best = sortRows(c.rows).find((r) => Number.isFinite(r.relChange)) ?? c.rows[0];
    return best?.name ?? null;
  }

  function renderHistogram() {
    const s = snapshot;
    const name = s.status === "ready" ? histogramTensor() : null;
    histSection.hidden = !name;
    if (!name) {
      hist.setData([]);
      return;
    }
    const c = s.result;
    const row = c.rows.find((r) => r.name === name);
    const mismatch = c.shapeMismatch.find((x) => x.name === name);
    const ta = s.a.view.tensors.get(name);
    const nameB = row?.nameB ?? mismatch?.nameB ?? name;
    const tb = s.b.view.tensors.get(nameB);
    histTitle.textContent = `${name}${state.tensor === name ? "" : " (il più cambiato)"}`;
    const series = [{ label: `A · ${sideLabel(s.a)}`, data: ta.data }];
    if (tb) series.push({ label: `B · ${sideLabel(s.b)}`, data: tb.data });
    hist.setData(series);
    const sa = tensorStats(ta.data, { histogram: false, percentiles: [] });
    const sb = tb ? tensorStats(tb.data, { histogram: false, percentiles: [] }) : null;
    const pair = (label, a, b) => el("div", {}, el("dt", {}, label), el("dd", {}, a), el("dd", {}, b ?? "—"));
    histStats.replaceChildren(
      el("div", { className: "lab-an-hist-head" }, el("dt", {}, ""), el("dd", {}, el("span", { className: "lab-an-tag lab-an-tag-a" }, "A")), el("dd", {}, el("span", { className: "lab-an-tag lab-an-tag-b" }, "B"))),
      pair("media", num(sa.mean), sb && num(sb.mean)),
      pair("dev. std", num(sa.std), sb && num(sb.std)),
      pair("min … max", `${num(sa.min)} … ${num(sa.max)}`, sb && `${num(sb.min)} … ${num(sb.max)}`),
      pair("norma ‖·‖", num(sa.l2), sb && num(sb.l2)),
    );
    histNote.textContent = row
      ? `‖A − B‖ = ${num(row.diffNorm)} (cambio relativo ${pct(row.relChange)}), coseno ${row.cosine === null ? "—" : num(row.cosine)}, max |Δ| = ${num(row.maxAbsDiff)}.`
      : mismatch
        ? `Forma diversa (${mismatch.shapeA.join(" × ")} in A, ${mismatch.shapeB.join(" × ")} in B): si confrontano le distribuzioni, non le celle.`
        : "B non ha questo tensore: c'è solo la distribuzione di A.";
  }

  // ---------------------------------------------------------------- differenza: pulsante e legenda
  function renderDiff(d) {
    diffButton.setAttribute("aria-pressed", String(d.on));
    diffButton.classList.toggle("is-on", d.on);
    for (const s of scaleInputs) s.input.checked = d.scale === s.value;
    legend.hidden = !d.on;
    if (d.on && d.legend) {
      const L = d.legend;
      const scaleText = d.scale === "common" ? `colore pieno = ±${num(L.maxAbs)}` : "colore pieno = il peso più grande di ogni matrice";
      legend.replaceChildren(
        el("div", { className: "lab-an-ramp", ariaHidden: "true", style: { background: rampCss() } }),
        el("div", { className: "lab-an-ramp-labels" }, el("span", {}, `${DIFF.negative}`), el("span", {}, "0"), el("span", {}, DIFF.positive)),
        el("p", { className: "lab-an-legend-text" }, `A − B · ${scaleText}. ${L.veiled ? `${int(L.veiled)} ${DIFF.veiled}.` : ""}`),
      );
    }
    diffNote.textContent = d.on ? DIFF.numbers : d.reason === "action" ? DIFF.offForAction : d.reason === "other" ? DIFF.offForOther : "";
  }

  const offs = [
    comparison.on(renderComparison),
    diff.on(renderDiff),
    ...["runs", "runA", "runB", "step", "stepB", "interpolate"].map((type) => state.on(type, renderPickers)),
    state.on("select", ({ path }) => {
      table.setSelected(path);
      if (path && snapshot.result?.rows.some((r) => r.name === path)) table.reveal(path);
      renderHistogram();
    }),
  ];
  renderPickers();
  renderComparison(comparison.current);
  renderDiff(diff.state);

  return {
    promptSlot,
    table,
    histogram: hist,
    render: renderPickers,
    dispose() {
      for (const off of offs) off();
      table.dispose();
      hist.dispose();
    },
  };
}

/** La rampa della legenda con gli STESSI colori delle matrici della scena (colormap divergente del nucleo). */
let ramp = null;
function rampCss() {
  if (ramp) return ramp;
  const lut = colormapLUT("diverging");
  const stops = [0, 32, 64, 96, 128, 160, 192, 224, 255].map((i) => `rgb(${lut[4 * i]}, ${lut[4 * i + 1]}, ${lut[4 * i + 2]}) ${((i / 255) * 100).toFixed(1)}%`);
  ramp = `linear-gradient(to right, ${stops.join(", ")})`;
  return ramp;
}
