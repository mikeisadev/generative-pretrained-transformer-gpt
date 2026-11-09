/**
 * "Il prossimo carattere: A contro B" (solo DOM): un prompt scritto dallo studente, la distribuzione del
 * prossimo carattere calcolata dal motore JS con i pesi di A e di B (run compatibili), i primi caratteri
 * di ciascuno con probabilità e logit; i grafici a barre e lo scatter dei logits vanno nel pannello dei
 * grafici (charts.setProbe). Il prompt è testo non fidato: textContent e messaggi, mai eccezioni.
 */
import { el } from "../../../core/dom.js";
import { shownChar } from "../inspect/meaning.js";
import { shortStepLabel } from "../steps.js";
import { PROMPT } from "./content.js";
import { num, pct } from "./format.js";
import { MAX_PROMPT, nextTokenProbe } from "./probe.js";

const DEBOUNCE_MS = 250;
const TOP = 5;

export function createPromptBlock(container, { tab, charts, defaultPrompt }) {
  const { state } = tab;
  const input = el("textarea", { className: "lab-an-prompt", rows: 2, maxLength: MAX_PROMPT, spellcheck: false, value: defaultPrompt });
  input.setAttribute("aria-label", "Prompt per il prossimo carattere");
  const outA = el("div", { className: "lab-an-next lab-an-next-a" });
  const outB = el("div", { className: "lab-an-next lab-an-next-b" });
  const note = el("p", { className: "hint lab-an-prompt-note", ariaLive: "polite" });
  const open = el("button", { type: "button", className: "lab-small", onclick: () => charts.reveal("prompt") }, "Vedi i grafici: barre e logits");
  const root = el(
    "section",
    { className: "lab-an-prompt-box", ariaLabel: PROMPT.title },
    el("h3", { className: "lab-h3" }, PROMPT.title),
    el("p", { className: "hint" }, PROMPT.intro),
    input,
    el("p", { className: "hint lab-an-prompt-help" }, "Invio va a capo: per il GPT del corso anche l'a capo è un carattere. Mentre scrivi i tasti della scena non si muovono."),
    el("div", { className: "lab-an-next-grid" }, outA, outB),
    note,
    open,
  );
  container.append(root);

  let timer = 0;
  let token = 0;

  async function probeSide(id, step) {
    if (!id) return { ok: false, reason: "", absent: true };
    const { engine, reason } = await state.engineFor(id, step);
    if (!engine) return { ok: false, reason: `${PROMPT.incompatible}: ${reason}` };
    return nextTokenProbe(engine, input.value, { top: 10 });
  }

  async function update() {
    timer = 0;
    const my = ++token;
    let a;
    let b;
    try {
      [a, b] = await Promise.all([probeSide(state.runA, state.step), probeSide(state.runB, state.stepB)]);
    } catch (error) {
      if (my === token) note.textContent = `Non riesco a calcolare il prossimo carattere: ${error.message}`;
      return;
    }
    if (my !== token) return;
    render(outA, "A", a);
    render(outB, "B", b);
    const T = (a.ok ? a : b.ok ? b : null);
    note.textContent = T?.cropped ? PROMPT.tooLong.replace("{{n}}", String(T.used)) : "";
    charts.setProbe(a, b, { prompt: input.value });
  }

  function render(box, side, p) {
    const id = side === "A" ? state.runA : state.runB;
    const step = side === "A" ? state.step : state.stepB;
    const head = el("p", { className: "lab-an-next-head", title: `${state.entry(id)?.name ?? ""} · ${shortStepLabel(step)}` }, el("span", { className: `lab-an-tag lab-an-tag-${side.toLowerCase()}` }, side), ` ${state.entry(id)?.name ?? ""} · ${shortStepLabel(step)}`);
    if (p.absent) {
      box.replaceChildren(head, el("p", { className: "hint" }, "Scegli B per confrontare."));
      return;
    }
    if (!p.ok) {
      box.replaceChildren(head, el("p", { className: "lab-error" }, p.reason));
      return;
    }
    box.replaceChildren(
      head,
      el(
        "ol",
        { className: "lab-an-next-list" },
        ...p.top.slice(0, TOP).map((t) =>
          el("li", {}, el("span", { className: "lab-an-char mono" }, shownChar(t.char)), el("span", { className: "lab-an-prob" }, pct(t.prob)), el("span", { className: "lab-an-logit mono" }, `logit ${num(t.logit, 3)}`), el("span", { className: "lab-an-pbar", ariaHidden: "true", style: { width: `${Math.round(100 * t.prob)}%` } })),
        ),
      ),
    );
  }

  function schedule(delay = DEBOUNCE_MS) {
    clearTimeout(timer);
    timer = setTimeout(() => void update(), delay);
  }

  // i tasti della scena (W A S D, C…) non partono mentre si scrive: la navigazione del nucleo ignora i campi di testo
  input.addEventListener("input", () => schedule());
  const offs = ["runA", "runB", "weights", "stepB"].map((type) => state.on(type, () => schedule(60)));
  schedule(0);

  return {
    root,
    input,
    refresh: () => schedule(0),
    dispose() {
      clearTimeout(timer);
      token++;
      for (const off of offs) off();
      root.remove();
    },
  };
}
