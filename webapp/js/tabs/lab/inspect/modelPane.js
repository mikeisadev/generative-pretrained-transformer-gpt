/**
 * Pannello "Modello" (solo DOM): come leggere la scena (colori, ruoli), il run nella scena con lo step
 * del training, che cosa cambia rispetto al GPT del corso e l'indice dei moduli (print(model)).
 */
import { el } from "../../../core/dom.js";
import { HONESTY, INTRO, LEGEND } from "../content.js";
import { architectureDiff } from "../runs/describe.js";
import { createTimeline } from "../runs/timeline.js";
import { LAB_ROLES } from "../scene/roles.js";
import { createOutline } from "./outline.js";

export function createModelPane(pane, tab) {
  const { state } = tab;
  const runLine = el("p", { className: "lab-run-line" });
  const archNote = el("div", { className: "lab-arch-note", hidden: true });
  const roles = ["embedding", "attention", "mlp", "norm", "output", "buffer"];
  pane.append(
    el("h3", { className: "lab-h3" }, INTRO.title),
    el("p", { className: "lab-intro" }, INTRO.text),
    el(
      "p",
      { className: "legend lab-legend" },
      ...LEGEND.flatMap((l) => [el("span", { className: `swatch ${l.swatch}` }), ` ${l.text} `]),
      " · 1 cella = 1 peso",
    ),
    el(
      "details",
      { className: "lab-details" },
      el("summary", {}, "Colori dei moduli e come leggere la scena"),
      el("ul", { className: "lab-role-legend" }, ...roles.map((r) => el("li", {}, el("span", { className: "lab-role-dot", style: { background: LAB_ROLES[r].color } }), el("strong", {}, LAB_ROLES[r].label), ` ${LAB_ROLES[r].text}`))),
      el("p", { className: "hint" }, INTRO.analogy),
      el("p", { className: "hint" }, `${HONESTY.weights} ${HONESTY.tril}`),
    ),
    runLine,
  );
  const timeline = createTimeline(pane, tab);
  pane.append(archNote, el("h3", { className: "lab-h3" }, "I moduli, come print(model)"), el("p", { className: "hint" }, "Clic su una riga: la camera ci vola, il codice compare a destra, i numeri nel pannello Tensore."));
  const outline = createOutline(pane, tab);

  function render() {
    const entry = state.entry(state.runA);
    runLine.replaceChildren(
      el("span", {}, "Nella scena: "),
      el("strong", {}, entry?.name ?? ""),
      " ",
      el("button", { type: "button", className: "lab-link", onclick: () => tab.openPane("run") }, "cambia run"),
    );
    const run = state.run(state.runA);
    const course = state.builtin;
    archNote.replaceChildren();
    archNote.hidden = true;
    if (run && course && run !== course) {
      const diff = architectureDiff(run.tensors, course.tensors, { engineDeclared: run.info?.engineCompatible ?? null });
      if (!diff.same) {
        archNote.hidden = false;
        archNote.append(el("strong", {}, "Che cosa cambia rispetto al GPT del corso"), el("ul", {}, ...diff.notes.map((n) => el("li", {}, n))));
        if (!diff.notes.length) archNote.append(el("p", { className: "hint" }, "Stessi moduli, pesi diversi."));
      }
    }
    outline.render();
  }
  state.on("runA", render);
  state.on("runs", () => {
    const strong = runLine.querySelector("strong"); // il nome del run A può cambiare (letto dal browser)
    const entry = state.entry(state.runA);
    if (strong && entry) strong.textContent = entry.name;
  });
  return { render, timeline, outline };
}
