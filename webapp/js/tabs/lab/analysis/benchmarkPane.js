/**
 * Pannello "Benchmark" (solo DOM): i numeri del Python per A e B allo step scelto (benchmark.json, per
 * il GPT del corso data/benchmark/builtin.json, anche snapshot per snapshot) con la colonna "meglio",
 * gli avvisi di confrontabilità e i testi generati; poi il calcolo nel browser con il controllo di
 * coerenza (browserBench.js). A e B sono quelli del pannello Confronto (stesso stato).
 */
import { el } from "../../../core/dom.js";
import { referenceAt } from "./benchmarkModel.js";
import { renderBenchTable, renderGeneration } from "./benchTable.js";
import { createBrowserBench } from "./browserBench.js";
import { BENCH } from "./content.js";
import { int } from "./format.js";

export function createBenchmarkPane(pane, { tab, bench, charts, runner }) {
  const { state } = tab;
  const who = el("div", { className: "lab-an-who" });
  const python = el("div", { className: "lab-an-python", ariaLive: "polite" });
  const browserBox = el("div", { className: "lab-an-browser" });
  pane.append(
    el("p", { className: "lab-intro" }, BENCH.intro),
    el("details", { className: "lab-details" }, el("summary", {}, "Un'analogia"), el("p", {}, BENCH.analogy)),
    who,
    el("h3", { className: "lab-h3" }, "I numeri del Python"),
    python,
    el("h3", { className: "lab-h3" }, "Ricalcolato nel browser"),
    browserBox,
    el("p", { className: "hint" }, "Grafici: la loss di ogni snapshot sulla validation completa e la sorpresa carattere per carattere lungo l'estratto."),
    el("button", { type: "button", className: "lab-small", onclick: () => charts.reveal("bench") }, "Vedi i grafici del benchmark"),
  );
  const browser = createBrowserBench(browserBox, { tab, bench, charts, runner });

  const label = (id, step) => {
    const name = state.entry(id)?.name ?? id;
    return `${name} · ${step === null || step === undefined ? "pesi finali" : `step ${int(step)}`}`;
  };

  function renderWho() {
    who.replaceChildren(
      el("p", { className: "lab-an-ab" }, el("span", { className: "lab-an-tag lab-an-tag-a" }, "A"), ` ${label(state.runA, state.step)}`),
      state.runB ? el("p", { className: "lab-an-ab" }, el("span", { className: "lab-an-tag lab-an-tag-b" }, "B"), ` ${label(state.runB, state.stepB)}`) : el("p", { className: "hint" }, "Nessun B: scegline uno per confrontare i benchmark."),
      el("button", { type: "button", className: "lab-link", onclick: () => tab.openPane("confronto") }, "Cambia A e B nel pannello Confronto"),
    );
  }

  let token = 0;
  async function renderPython() {
    const my = ++token;
    let a;
    let b;
    try {
      [a, b] = await Promise.all([bench.of(state.runA), state.runB ? bench.of(state.runB) : null]);
    } catch (error) {
      if (my === token) python.replaceChildren(el("p", { className: "lab-error" }, `Non riesco a leggere i benchmark: ${error.message}`));
      return;
    }
    if (my !== token) return;
    const refA = referenceAt(a, state.step);
    const refB = state.runB ? referenceAt(b, state.stepB) : null;
    const parts = [];
    const missing = (side, id, data, step) => {
      if (!data) return el("p", { className: "hint" }, `${side}: ${BENCH.noBenchmark}`);
      if (step !== null && step !== undefined) return el("p", { className: "hint" }, `${side}: il benchmark.json non ha lo step ${int(step)}${data.snapshots.length ? ` (ha ${data.snapshots.map((s) => int(s.step)).join(", ")})` : " (solo i pesi finali)"}.`);
      return null;
    };
    if (!refA) parts.push(missing("A", state.runA, a, state.step));
    if (state.runB && !refB) parts.push(missing("B", state.runB, b, state.stepB));
    for (const [side, data] of [["A", a], ["B", b]]) if (data?.problems.length) parts.push(el("p", { className: "lab-error" }, `${side}, benchmark.json: ${data.problems.slice(0, 4).join("; ")}${data.problems.length > 4 ? "…" : ""}`));
    if (refA || refB) {
      const labelA = label(state.runA, state.step);
      const labelB = state.runB ? label(state.runB, state.stepB) : "";
      // senza i numeri di A ma con quelli di B la colonna di A resta, con i trattini (A − B vuota)
      parts.unshift(renderBenchTable({ a, b, refA, refB, labelA, labelB }));
      parts.push(renderGeneration({ refA, refB, labelA, labelB }));
    }
    python.replaceChildren(...parts.filter(Boolean));
  }

  function render() {
    renderWho();
    void renderPython();
  }

  const offs = ["runA", "runB", "step", "stepB", "runs"].map((type) => state.on(type, render));
  render();

  return {
    browser,
    render,
    dispose() {
      token++;
      for (const off of offs) off();
      browser.dispose();
    },
  };
}
