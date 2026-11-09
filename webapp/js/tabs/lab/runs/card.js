/**
 * Scheda di un run (solo DOM): nome, descrizione, data, sorgente, config con le differenze dal GPT del
 * corso, parametri, compatibilità con il motore JS (con il motivo), training, note e avvisi. Tutti i
 * testi dei file caricati passano da textContent (el() li mette come nodi di testo).
 */
import { el } from "../../../core/dom.js";
import { architectureDiff, compatibilityText, configDiff, configValue, fmtInt, formatDate, originText, trainingText } from "./describe.js";

export function runCard(entry, run, course) {
  const card = el("article", { className: "lab-run-card", ariaLabel: `Scheda del run ${entry.name}` });
  card.append(el("h4", { className: "lab-run-title" }, entry.name), entry.description ? el("p", { className: "lab-run-desc" }, entry.description) : "");
  if (!run) {
    card.append(el("p", { className: "hint" }, entry.error ?? "Salvato in questo browser: i pesi si leggono quando lo scegli."));
    return card;
  }
  const rows = [];
  const add = (label, ...value) => rows.push(el("div", { className: "lab-fact" }, el("dt", {}, label), el("dd", {}, ...value)));
  if (run.meta?.created ?? run.info?.created) add("Data", formatDate(run.meta?.created ?? run.info?.created));
  add("Sorgente", originText(entry, run));
  add("Parametri", `${fmtInt(run.totalParams)} in ${fmtInt(run.tensors.size)} tensori`);

  const diff = configDiff(run.config, course?.config);
  if (diff.length) {
    add(
      "Config",
      el(
        "ul",
        { className: "lab-config" },
        ...diff.map((d) =>
          el(
            "li",
            { className: d.changed || d.extra ? "changed" : "" },
            el("code", {}, d.key),
            ` = ${configValue(d.value)}`,
            d.changed ? el("span", { className: "lab-was" }, ` (corso: ${configValue(d.course)})`) : d.extra ? el("span", { className: "lab-was" }, " (nuova)") : "",
            el("small", {}, ` · ${d.label}`),
          ),
        ),
      ),
    );
  } else add("Config", "non dichiarata (ricavata dalle forme dei pesi dove si può)");

  if (course && run !== course) {
    const arch = architectureDiff(run.tensors, course.tensors, { engineDeclared: run.info?.engineCompatible ?? null });
    add("Architettura", arch.same ? "uguale al GPT del corso: stessi nomi e forme dei 94 tensori" : el("ul", { className: "lab-notes" }, ...arch.notes.map((n) => el("li", {}, n))));
  }
  const compat = compatibilityText(run);
  add("Motore JS", el("strong", { className: compat.ok ? "lab-ok" : "lab-no" }, compat.title), el("span", {}, `. ${compat.text}`), compat.reasons?.length ? el("ul", { className: "lab-notes" }, ...compat.reasons.map((r) => el("li", {}, r))) : "");
  add("Training", trainingText(run));
  if (run.benchmark?.metrics) {
    const m = run.benchmark.metrics;
    const parts = [];
    if (Number.isFinite(m.loss)) parts.push(`loss ${m.loss.toLocaleString("it-IT", { maximumFractionDigits: 4 })}`);
    if (Number.isFinite(m.top1)) parts.push(`top-1 ${(m.top1 * 100).toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`);
    if (Number.isFinite(m.top5)) parts.push(`top-5 ${(m.top5 * 100).toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`);
    add("Benchmark", parts.length ? `${parts.join(" · ")} (validation completa, benchmark.json)` : "benchmark.json presente");
  }
  const notes = [...(run.meta?.notes ?? []), ...(run.meta?.warnings ?? [])];
  if (notes.length) add("Note", el("ul", { className: "lab-notes" }, ...notes.slice(0, 12).map((n) => el("li", {}, n))));
  card.append(el("dl", { className: "lab-facts" }, ...rows));
  return card;
}
