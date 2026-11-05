/**
 * Pezzi di interfaccia condivisi dalle viste "Pesi" e "Rete neurale": il riquadro informativo
 * nella sidebar (pannello "Viste"), le formule KaTeX con i numeri all'italiana, legende e fonti.
 */
import katex from "../../../../vendor/katex/katex.mjs";
import { el } from "../../../core/dom.js";

/** Riquadro della vista nel pannello "Viste": visibile solo mentre la vista è attiva. */
export function createViewInfo(tab, id) {
  const box = el("section", { className: "tf-view-info", hidden: true, dataset: { view: id } });
  tab.panes.views.append(box);
  return box;
}

/** Numero all'italiana dentro una formula KaTeX: la virgola decimale va protetta con {,}. */
export function texNum(value, digits = 3) {
  if (!Number.isFinite(value)) return value > 0 ? "+\\infty" : "-\\infty";
  const text = value.toLocaleString("it-IT", { minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: false });
  return text.replace(",", "{,}").replace("-", "−");
}

/** Elemento con una formula KaTeX (display). */
export function formula(tex, { display = true, className = "formula" } = {}) {
  const holder = el("div", { className });
  katex.render(tex, holder, { displayMode: display, throwOnError: false, output: "html" });
  return holder;
}

/** Riquadro "Approfondimento" (stesso stile del menu del percorso). */
export function deepBox(title, ...children) {
  return el("details", { className: "deep" }, el("summary", {}, title), ...children);
}

/** Legenda: [{swatch: css colore | classe, label, dashed?, ring?}] */
export function legend(items) {
  return el(
    "ul",
    { className: "tf-legend" },
    ...items.map((item) => {
      const swatch = el("span", { className: `tf-swatch${item.dashed ? " dashed" : ""}${item.ring ? " ring" : ""}${item.line ? " line" : ""}` });
      swatch.style.setProperty("--sw", item.color); // le proprietà CSS personalizzate non passano da Object.assign
      return el("li", {}, swatch, el("span", {}, item.label));
    }),
  );
}

/** Pannello "Fonti": link esterni verificati. */
export function sources(list) {
  return el(
    "details",
    { className: "deep tf-sources" },
    el("summary", {}, "Fonti"),
    el("ul", {}, ...list.map(({ href, label, note }) => el("li", {}, el("a", { href, target: "_blank", rel: "noopener" }, label), note ? el("span", { className: "hint" }, ` — ${note}`) : null))),
  );
}

/**
 * Sezione dei grafici della vista attiva, portata in cima al pannello "Grafici":
 * chi apre il pannello mentre guarda questa vista trova subito i suoi grafici.
 */
export function topChartsSection(charts, id, title, description) {
  const section = charts.section(id, title, description);
  const parent = section.root?.parentElement;
  if (parent && parent.firstElementChild !== section.root) {
    const first = [...parent.children].find((c) => c.classList.contains("charts-section"));
    if (first && first !== section.root) parent.insertBefore(section.root, first);
  }
  return section;
}
