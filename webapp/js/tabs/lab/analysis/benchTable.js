/**
 * La tabella dei benchmark del Python (solo DOM): una riga per metrica, le colonne A, B, A − B e
 * "meglio", con il verso di ogni metrica e gli avvisi di confrontabilità; più i testi generati (seme
 * fisso) di A e di B. I numeri vengono da benchmarkModel.js; i testi dei file sono sempre textContent.
 */
import { el } from "../../../core/dom.js";
import { METRIC_ROWS, comparisonRows, comparisonWarnings, environmentText } from "./benchmarkModel.js";
import { BENCH, METRIC_HELP } from "./content.js";
import { fixed, int, num, pct, signedFixed, signedInt, signedNum, signedPoints } from "./format.js";

/** Un valore nel formato della sua riga. */
export function formatMetric(format, v) {
  if (v === null || v === undefined) return "—";
  if (format === "fixed4") return fixed(v, 4);
  if (format === "fixed3") return fixed(v, 3);
  if (format === "pct") return pct(v, 2);
  if (format === "int") return int(v);
  return num(v, 3);
}

/** La differenza A − B nel formato della sua riga (le quote in punti percentuali). */
export function formatDelta(format, d) {
  if (d === null || d === undefined) return "—";
  if (format === "fixed4") return signedFixed(d, 4);
  if (format === "fixed3") return signedFixed(d, 3);
  if (format === "pct") return signedPoints(d, 2);
  if (format === "int") return signedInt(d);
  return signedNum(d, 3);
}

/**
 * @param {{a: object|null, b: object|null, refA: object|null, refB: object|null, labelA: string, labelB: string}} o
 *   a/b: benchmark normalizzati (readBenchmark); refA/refB: referenceAt (i numeri allo step scelto)
 */
export function renderBenchTable({ a, b, refA, refB, labelA, labelB }) {
  const hasB = Boolean(refB);
  const rows = comparisonRows(refA, refB);
  const head = el(
    "tr",
    {},
    el("th", { scope: "col" }, "metrica"),
    el("th", { scope: "col", className: "is-num" }, el("span", { className: "lab-an-tag lab-an-tag-a" }, "A")),
    hasB ? el("th", { scope: "col", className: "is-num" }, el("span", { className: "lab-an-tag lab-an-tag-b" }, "B")) : "",
    hasB ? el("th", { scope: "col", className: "is-num", title: "A − B (il run nella scena meno il termine di paragone, come scripts/benchmark.py PRIMA DOPO stampa dopo − prima) e chi è meglio" }, "A − B · meglio") : "",
  );
  const body = rows.map((r) => {
    const help = METRIC_HELP[r.key] ?? "";
    const noVerdict = r.machine || r.direction === 0;
    const verdictTitle = r.machine ? "Dipende dalla macchina e dal suo carico: nessun verdetto." : r.direction === 0 ? "Nessun verso «meglio» per questa misura." : r.direction < 0 ? "Più basso è meglio." : "Più alto è meglio.";
    const verdict = noVerdict || !r.better ? null : r.better === "=" ? el("span", { className: "lab-an-better lab-an-better-same", title: verdictTitle }, "pari") : el("span", { className: `lab-an-tag lab-an-tag-${r.better.toLowerCase()} lab-an-better`, title: `${verdictTitle} Meglio ${r.better}.` }, r.better);
    return el(
      "tr",
      { className: r.machine ? "is-machine" : "" },
      el("th", { scope: "row", title: help }, r.label, r.direction ? el("small", { className: "lab-an-dir", ariaHidden: "true" }, r.direction < 0 ? " ↓" : " ↑") : ""),
      el("td", { className: "is-num" }, formatMetric(r.format, r.a)),
      hasB ? el("td", { className: "is-num" }, formatMetric(r.format, r.b)) : "",
      hasB ? el("td", { className: "is-num lab-an-delta" }, el("span", {}, formatDelta(r.format, r.delta)), verdict ?? el("span", { className: "lab-an-better-none", title: verdictTitle }, el("span", { ariaHidden: "true" }, "—"), el("span", { className: "lab-an-sr" }, "nessun verdetto"))) : "",
    );
  });
  const table = el("table", { className: "lab-an-bench-table" }, el("caption", { className: "lab-an-sr" }, "Benchmark del Python: A e B a confronto"), el("thead", {}, head), el("tbody", {}, ...body));
  const legend = el("p", { className: "hint" }, "↓ più basso è meglio · ↑ più alto è meglio; top-1 e top-5 si confrontano in punti percentuali (pp). * Token al secondo dichiarati nel benchmark.json: misurati sulla macchina del Python, non qui.");
  const sources = el("ul", { className: "lab-an-sources" }, sourceLine("A", labelA, a, refA), hasB ? sourceLine("B", labelB, b, refB) : "");
  const warnings = a && b && hasB ? comparisonWarnings(a, b) : [];
  const warn = warnings.length ? el("ul", { className: "lab-an-warnings" }, ...warnings.map((w) => el("li", { className: `lab-an-warn-${w.level === "attenzione" ? "high" : "low"}` }, el("strong", {}, `${w.level === "attenzione" ? "Attenzione" : "Nota"}: `), w.text))) : "";
  return el("div", { className: "lab-an-bench" }, el("div", { className: "lab-an-table-wrap lab-an-bench-wrap", tabIndex: 0, role: "region", ariaLabel: "Tabella del benchmark" }, table), legend, sources, warn, el("p", { className: "hint" }, BENCH.speed));
}

function sourceLine(side, label, bench, ref) {
  if (!ref) return el("li", {}, el("span", { className: `lab-an-tag lab-an-tag-${side.toLowerCase()}` }, side), ` ${label}: nessun numero del Python per questi pesi.`);
  const count = ref?.source === "snapshot" ? ref.metrics.predictions : bench?.counts?.predictions ?? bench?.metrics?.predictions;
  const env = environmentText(bench);
  return el(
    "li",
    {},
    el("span", { className: `lab-an-tag lab-an-tag-${side.toLowerCase()}` }, side),
    ` ${label}: validation completa${Number.isFinite(count) ? ` (${int(count)} previsioni)` : ""}`,
    env ? `, misurato con PyTorch su ${env}` : "",
    bench?.created ? ` il ${bench.created.slice(0, 10)}` : "",
    ".",
  );
}

/** I testi generati dal Python (campionato e greedy) di A e di B. */
export function renderGeneration({ refA, refB, labelA, labelB }) {
  const block = (side, label, ref) => {
    const g = ref?.bench?.generation;
    if (!g || ref.source !== "final") return null;
    const prompt = g.prompt ?? "";
    return el(
      "div",
      { className: "lab-an-gen-side" },
      el("p", { className: "lab-an-gen-head" }, el("span", { className: `lab-an-tag lab-an-tag-${side.toLowerCase()}` }, side), ` ${label}`),
      g.sampled !== null ? el("figure", { className: "lab-an-gen" }, el("figcaption", {}, `campionato (seme ${g.seed ?? "?"}, temperatura ${g.temperature === null ? "?" : num(g.temperature, 2)})`), el("pre", {}, el("span", { className: "lab-an-gen-prompt" }, prompt), g.sampled)) : "",
      g.greedy !== null ? el("figure", { className: "lab-an-gen" }, el("figcaption", {}, "greedy (sempre il carattere più probabile)"), el("pre", {}, el("span", { className: "lab-an-gen-prompt" }, prompt), g.greedy)) : "",
    );
  };
  const blocks = [block("A", labelA, refA), block("B", labelB, refB)].filter(Boolean);
  if (!blocks.length) return el("p", { className: "hint" }, "Testi generati: solo per i pesi finali di un run con benchmark.json (il Python non genera testo per ogni snapshot).");
  const g = (refA?.source === "final" ? refA : refB)?.bench?.generation;
  const what = g ? ` (prompt «${String(g.prompt ?? "").replace(/\n/g, "↵")}», ${int(g.maxNewTokens ?? NaN)} caratteri)` : "";
  return el("details", { className: "lab-details lab-an-gen-box" }, el("summary", {}, `Testo generato dal Python${what}`), ...blocks, el("p", { className: "hint" }, BENCH.sampled));
}

export { METRIC_ROWS };
