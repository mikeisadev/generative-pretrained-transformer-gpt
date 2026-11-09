/**
 * Il benchmark ricalcolato nel browser (DOM + motore): per un run che il motore JS sa eseguire, il
 * protocollo finestre-contigue-v1 sull'estratto della validation (val_excerpt.txt) nel Worker del
 * runner, con l'avanzamento, "Annulla" e il controllo di coerenza con i numeri del Python (excerpt di
 * benchmark.json o di builtin.json, anche per ogni snapshot). Il testo greedy deve coincidere; quello
 * campionato no (generatori casuali diversi) e lo si dice.
 *
 *   const browser = createBrowserBench(box, { tab, bench, charts, runner });
 *   browser.compute("A");  browser.computeAll();  browser.cancel();  browser.render();
 */
import { el } from "../../../core/dom.js";
import { referenceAt } from "./benchmarkModel.js";
import { BENCH } from "./content.js";
import { fixed, int, num, pct, seconds, signedInt, signedNum } from "./format.js";
import { checkConsistency } from "./protocol.js";
import { CancelledError } from "./runner.js";

const MAX_RESULTS = 40;
/** Etichette brevi delle righe del controllo (la colonna sta nella sidebar). */
const SHORT = { windows: "finestre", predictions: "previsioni", loss: "loss (nat)", top1: "giuste top-1", top5: "giuste top-5", entropy: "entropia", bpc: "bit/carattere", bits: "bit di sorpresa", greedy: "testo greedy" };
const VERDICT = {
  ok: { mark: "✓", text: "Il motore JS e PyTorch danno gli stessi numeri", cls: "ok" },
  near: { mark: "≈", text: "Quasi: differenze entro le tolleranze del Python", cls: "near" },
  bad: { mark: "✗", text: "I numeri non coincidono: il calcolo JS non è quello del run", cls: "bad" },
  none: { mark: "·", text: "Nessun numero del Python con cui confrontare", cls: "none" },
};

/** sha256 esadecimale di un testo o di byte (null se il browser non lo offre: pagina non sicura). */
export async function sha256Hex(data) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const digest = await subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export function createBrowserBench(container, { tab, bench, charts, runner }) {
  const { state } = tab;
  const results = new Map(); // "runId|step" → risultato
  let job = null; // { side, label, phase, done, total, batch }
  let message = "";

  const buttonA = el("button", { type: "button", className: "lab-primary lab-small", onclick: () => void compute("A") }, "Calcola per A");
  const buttonB = el("button", { type: "button", className: "lab-small", onclick: () => void compute("B") }, "Calcola per B");
  const buttonAll = el("button", { type: "button", className: "lab-small", onclick: () => void computeAll() }, "Tutti gli snapshot di A");
  const cancel = el("button", { type: "button", className: "lab-small lab-danger", hidden: true, onclick: () => runner.cancel() }, "Annulla");
  const progress = el("progress", { className: "lab-an-progress", max: 1, value: 0, hidden: true });
  const progressText = el("p", { className: "lab-an-progress-text hint" });
  const live = el("p", { className: "lab-an-sr", ariaLive: "polite" });
  const cards = el("div", { className: "lab-an-browser-cards" });
  const batchBox = el("div", { className: "lab-an-batch" });
  container.append(
    el("p", { className: "hint" }, BENCH.browserIntro),
    el("p", { className: "hint" }, BENCH.consistencyRule),
    el("div", { className: "lab-an-buttons" }, buttonA, buttonB, buttonAll, cancel),
    progress,
    progressText,
    live,
    cards,
    batchBox,
  );

  const keyOf = (id, step) => `${id}|${step ?? "finali"}`;
  const sideOf = (side) => (side === "A" ? { id: state.runA, step: state.step } : { id: state.runB, step: state.stepB });

  /** Tutto quello che serve per un calcolo: manifest del motore, pesi in ordine del motore, riferimento del Python. */
  async function prepare(id, step) {
    const run = await state.loadRun(id);
    if (!run.compatibleWithEngine) return { error: BENCH.browserIncompatible };
    const [{ engine, reason }, view, text, data] = await Promise.all([state.engineFor(id, null), state.weightsAt(id, step), bench.excerpt(), bench.of(id)]);
    if (!engine) return { error: `${BENCH.browserIncompatible} (${reason})` };
    const interpolated = view.source === "interpolated";
    const ref = interpolated ? null : referenceAt(data, view.step ?? null);
    return { run, view, text, data, ref, manifest: engine.manifest, weights: run.engineFlat(view.tensors), interpolated, label: `${state.entry(id)?.name ?? id} · ${view.label}` };
  }

  async function runOne(id, step, { greedy = true, side, batch = false } = {}) {
    const p = await prepare(id, step);
    if (p.error) return { error: p.error };
    const generation = p.data?.generation;
    const wantGreedy = greedy ? { prompt: generation?.prompt ?? "ROMEO:\n", tokens: generation?.maxNewTokens ?? 200 } : null;
    job = { side, label: p.label, phase: "score", done: 0, total: 1, batch };
    renderProgress();
    const out = await runner.run({
      manifest: p.manifest,
      weights: p.weights,
      text: p.text,
      greedy: wantGreedy,
      onProgress: (x) => {
        job = { ...job, ...x };
        renderProgress();
      },
    });
    // il riferimento del Python vale solo se l'estratto è lo stesso (sha256 del file di benchmark.json)
    const sha = p.ref?.excerpt?.sha256 ? await sha256Hex(p.text) : null;
    const sameText = sha === null || !p.ref?.excerpt?.sha256 ? null : sha === p.ref.excerpt.sha256;
    const python = sameText === false ? null : p.ref?.excerpt ?? null;
    const check = checkConsistency(out.score, python, { greedy: out.greedy, pythonGreedy: p.ref?.greedy ?? null });
    const result = { key: keyOf(id, step), runId: id, step: p.view.step ?? null, label: p.label, interpolated: p.interpolated, score: out.score, greedy: out.greedy, pythonGreedy: p.ref?.greedy ?? null, sampled: p.ref?.sampled ?? null, python, sameText, check, timing: out.timing, where: out.where, at: Date.now() };
    results.delete(result.key);
    results.set(result.key, result);
    while (results.size > MAX_RESULTS) results.delete(results.keys().next().value);
    return result;
  }

  async function compute(side) {
    const { id, step } = sideOf(side);
    if (!id) return;
    message = "";
    try {
      const r = await runOne(id, step, { side });
      if (r.error) message = r.error;
      else live.textContent = `${side}: ${VERDICT[r.check.verdict].text}.`;
    } catch (error) {
      message = error instanceof CancelledError ? "Calcolo annullato." : `Calcolo non riuscito: ${error.message}`;
    } finally {
      job = null;
      render();
    }
  }

  /** Tutti gli snapshot di A (più i pesi finali) uno dopo l'altro, senza greedy: la tabella si riempie man mano. */
  async function computeAll() {
    const id = state.runA;
    const steps = [...state.stepsOf(id), null];
    message = "";
    const rows = [];
    try {
      for (const [i, step] of steps.entries()) {
        const r = await runOne(id, step, { greedy: step === null, side: "A", batch: { index: i, count: steps.length } });
        if (r.error) {
          message = r.error;
          break;
        }
        rows.push(r);
        renderBatch(rows, steps.length);
      }
      const ok = rows.filter((r) => r.check.verdict === "ok").length;
      if (rows.length) live.textContent = `${ok} su ${rows.length} coincidono con il Python.`;
    } catch (error) {
      message = error instanceof CancelledError ? `Annullato dopo ${rows.length} su ${steps.length}.` : `Calcolo non riuscito: ${error.message}`;
    } finally {
      job = null;
      render();
      renderBatch(rows, steps.length);
    }
  }

  function renderProgress() {
    const busy = Boolean(job);
    progress.hidden = !busy;
    cancel.hidden = !busy;
    for (const b of [buttonA, buttonB, buttonAll]) b.disabled = busy;
    if (!busy) {
      progressText.textContent = message;
      return;
    }
    progress.max = Math.max(1, job.total);
    progress.value = job.done;
    const what = job.phase === "greedy" ? `testo greedy: ${int(job.done)} di ${int(job.total)} caratteri` : `finestra ${int(job.done)} di ${int(job.total)}`;
    const batch = job.batch ? `${int(job.batch.index + 1)} di ${int(job.batch.count)} · ` : "";
    progressText.textContent = `${batch}${job.label}: ${what}`;
  }

  function card(side, r) {
    const v = VERDICT[r.check.verdict];
    const s = r.score;
    const rows = r.check.rows.map((x) => {
      const isCount = ["windows", "predictions", "top1", "top5", "bits", "greedy"].includes(x.id);
      const fmt = (value) => (value === null || value === undefined ? "—" : isCount ? int(value) : fixed(value, 6));
      const delta = x.id === "greedy" ? (x.delta < 0 ? "identico" : `diverso dal carattere ${int(x.delta + 1)}`) : x.id === "bits" ? num(x.delta, 2) : x.delta === null ? "" : isCount ? signedInt(x.delta) : signedNum(x.delta, 2);
      const mark = x.level === "ok" ? "✓" : x.level === "near" ? "≈" : x.level === "bad" ? "✗" : "";
      return el(
        "tr",
        { className: `is-${x.level}` },
        el("th", { scope: "row" }, SHORT[x.id] ?? x.label),
        el("td", { className: "is-num" }, x.id === "bits" ? `${int(x.python)} val.` : x.id === "greedy" ? `${int(x.python)} car.` : fmt(x.python)),
        el("td", { className: "is-num" }, x.id === "bits" ? `${int(x.browser)} val.` : x.id === "greedy" ? `${int(x.browser)} car.` : fmt(x.browser)),
        el("td", { className: "is-num lab-an-delta" }, el("span", {}, delta), el("span", { className: "lab-an-level", ariaHidden: "true" }, mark), mark ? el("span", { className: "lab-an-sr" }, x.level === "ok" ? " coincide" : x.level === "near" ? " quasi" : " diverso") : ""),
      );
    });
    const speed = r.timing.scoreMs > 0 ? `${int(Math.round((s.predictions / r.timing.scoreMs) * 1000))} previsioni/s` : "";
    const gen = r.timing.greedyMs > 0 && r.greedy ? `, ${int(Math.round((r.greedy.length / r.timing.greedyMs) * 1000))} token/s in generazione` : "";
    return el(
      "article",
      { className: `lab-an-card lab-an-verdict-${v.cls}` },
      el("header", {}, el("span", { className: `lab-an-tag lab-an-tag-${side.toLowerCase()}` }, side), el("strong", {}, ` ${r.label}`)),
      el("p", { className: "lab-an-verdict" }, el("span", { className: "lab-an-mark", ariaHidden: "true" }, v.mark), ` ${v.text}.`),
      el("p", { className: "lab-an-metrics" }, `loss ${fixed(s.loss, 4)} · perplexity ${fixed(s.perplexity, 3)} · ${fixed(s.bpc, 3)} bit/car. · top-1 ${pct(s.top1Rate, 2)} · top-5 ${pct(s.top5Rate, 2)} · entropia ${fixed(s.entropy, 3)}`),
      r.check.rows.length ? el("div", { className: "lab-an-table-wrap" }, el("table", { className: "lab-an-check" }, el("caption", { className: "lab-an-sr" }, "Controllo di coerenza con il Python"), el("thead", {}, el("tr", {}, el("th", { scope: "col" }, ""), el("th", { scope: "col", className: "is-num" }, "Python"), el("th", { scope: "col", className: "is-num" }, "browser"), el("th", { scope: "col", className: "is-num" }, "Δ"))), el("tbody", {}, ...rows))) : "",
      r.interpolated ? el("p", { className: "hint" }, BENCH.interpolated) : "",
      r.sameText === false ? el("p", { className: "lab-error" }, "Il benchmark.json è stato misurato su un altro estratto (sha256 diverso): niente confronto.") : "",
      r.check.verdict === "near" ? el("p", { className: "hint" }, BENCH.near) : "",
      r.greedy ? el("figure", { className: "lab-an-gen" }, el("figcaption", {}, `greedy calcolato qui${r.pythonGreedy ? (r.greedy === r.pythonGreedy ? ": identico a quello di PyTorch" : ": diverso da quello di PyTorch") : ""}`), el("pre", {}, r.greedy)) : "",
      el("p", { className: "hint lab-an-where" }, `Misurato in questo browser (${r.where === "worker" ? "in un Worker" : "nella pagina, a pezzi"}): ${int(s.predictions)} previsioni in ${seconds(r.timing.scoreMs / 1000)}${speed ? ` = ${speed}` : ""}${gen}. Dipende da questo computer.`),
    );
  }

  function renderBatch(rows, total) {
    if (!rows.length) {
      batchBox.replaceChildren();
      return;
    }
    const ok = rows.filter((r) => r.check.verdict === "ok").length;
    batchBox.replaceChildren(
      el("h4", { className: "lab-h4" }, `Tutti gli snapshot: ${ok} su ${rows.length} ✓${rows.length < total ? ` (di ${total})` : ""}`),
      el(
        "div",
        { className: "lab-an-table-wrap" },
        el(
          "table",
          { className: "lab-an-check lab-an-batch-table" },
          el("caption", { className: "lab-an-sr" }, "Benchmark nel browser di tutti gli snapshot"),
          el("thead", {}, el("tr", {}, ...["step", "loss Python", "loss browser", "Δ loss", "top-1", "top-5", ""].map((t, i) => el("th", { scope: "col", className: i ? "is-num" : "" }, t)))),
          el(
            "tbody",
            {},
            ...rows.map((r) => {
              const row = (id) => r.check.rows.find((x) => x.id === id);
              const loss = row("loss");
              const hits = (id) => (row(id) ? `${int(row(id).browser)}${row(id).delta ? ` (${signedInt(row(id).delta)})` : ""}` : int(r.score[id]));
              return el("tr", { className: `is-${r.check.verdict}` }, el("th", { scope: "row" }, r.step === null ? "finali" : int(r.step)), el("td", { className: "is-num" }, loss?.python === null || !loss ? "—" : fixed(loss.python, 6)), el("td", { className: "is-num" }, fixed(r.score.loss, 6)), el("td", { className: "is-num" }, loss?.delta === null || !loss ? "—" : signedNum(loss.delta, 2)), el("td", { className: "is-num" }, hits("top1")), el("td", { className: "is-num" }, hits("top5")), el("td", { className: "lab-an-level" }, VERDICT[r.check.verdict].mark));
            }),
          ),
        ),
      ),
    );
  }

  function render() {
    renderProgress();
    const a = state.runA ? results.get(keyOf(state.runA, state.step)) : null;
    const b = state.runB ? results.get(keyOf(state.runB, state.stepB)) : null;
    const runA = state.run(state.runA);
    const runB = state.runB ? state.run(state.runB) : null;
    buttonA.hidden = !runA?.compatibleWithEngine;
    buttonAll.hidden = !runA?.compatibleWithEngine || !state.stepsOf(state.runA).length;
    buttonB.hidden = !runB?.compatibleWithEngine;
    const notes = [];
    if (runA && !runA.compatibleWithEngine) notes.push(el("p", { className: "hint" }, `A: ${BENCH.browserIncompatible}`));
    if (runB && !runB.compatibleWithEngine) notes.push(el("p", { className: "hint" }, `B: ${BENCH.browserIncompatible}`));
    cards.replaceChildren(...notes, a ? card("A", a) : "", b ? card("B", b) : "");
    // la sorpresa carattere per carattere nei grafici: il browser se ha calcolato, altrimenti il Python (pesi finali)
    void surprisalFor(a, state.runA, state.step).then((sa) => surprisalFor(b, state.runB, state.stepB).then((sb) => charts.setSurprisal({ a: sa, b: sb })));
  }

  async function surprisalFor(result, id, step) {
    if (result?.score?.bits) return { bits: result.score.bits, label: "calcolato qui" };
    if (!id || step !== null) return null;
    const data = await bench.of(id).catch(() => null);
    return data?.excerpt?.surprisalBits ? { bits: data.excerpt.surprisalBits, label: "Python" } : null;
  }

  const offs = ["runA", "runB", "step", "stepB", "weights"].map((type) => state.on(type, () => render()));
  render();

  return {
    compute,
    computeAll,
    cancel: () => runner.cancel(),
    render,
    get busy() {
      return Boolean(job);
    },
    results,
    dispose() {
      for (const off of offs) off();
      runner.cancel();
    },
  };
}
