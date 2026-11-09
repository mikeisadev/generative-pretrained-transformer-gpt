/**
 * Pannello "Azioni" della sidebar (solo DOM): che cosa mostrano le animazioni, se il run A si può
 * eseguire, il prompt, il token da seguire, blocco e head, la nn.Linear di «x · Wᵀ», poi le schede delle
 * azioni (renderOverview del player del nucleo) e che cosa è vero.
 *
 * Il prompt è testo dello studente: si scrive solo con textContent / value, mai come HTML.
 */
import { el } from "../../../core/dom.js";
import { COURSE_RUN_ID, incompatibility } from "../state.js";
import { shortStepLabel } from "../steps.js";
import { CATEGORIES } from "./catalog.js";
import { DEFAULT_PROMPT, LINEAR_CHOICES, checkPrompt, resolveToken, visibleChar } from "./settings.js";

const INTRO = "Le animazioni mostrano come i pesi lavorano insieme, con i numeri veri: i pesi del run nella scena allo step scelto e le attivazioni calcolate dal motore JS sul tuo prompt. Le matrici temporanee che compaiono accanto ai pesi sono le attivazioni di un token; la lavagna in basso scrive il calcolo.";
const ANALOGY = "Come guardare una ricetta mentre la si cucina: il codice Python dice i passi, i pesi sono gli ingredienti misurati dal training, le attivazioni sono quello che c'è nella pentola a ogni passo.";
const HONEST = [
  "Pesi: quelli della scena (float32 dei pesi finali, float16 degli snapshot), senza arrotondamenti.",
  "Attivazioni: calcolate dal motore JS (js/engine/gpt.js), lo stesso forward di gpt/model.py; un test le confronta con PyTorch. Ogni passo intermedio mostrato qui (prodotti, media, varianza, punteggi del futuro prima della maschera) è ricalcolato con le stesse operazioni e coincide con il motore.",
  "Passo di training: gradiente esatto di lm_head per un esempio; il training vero usa AdamW su batch di 16 × 32 posizioni e aggiorna tutti i pesi.",
  "Run che il motore JS non sa eseguire (architettura diversa, GELU dichiarata): niente animazioni dei calcoli, perché le attivazioni andrebbero inventate. «I pesi durante il training» funziona con ogni run che ha gli snapshot.",
];

export function createActionsPane(pane, tab, settings) {
  const { state } = tab;
  const status = el("div", { className: "lab-act-status", ariaLive: "polite" });
  const prompt = el("textarea", { className: "lab-act-prompt", rows: 2, spellcheck: false, value: settings.prompt, "aria-describedby": "lab-act-prompt-msg" });
  const promptMessage = el("p", { className: "lab-act-message", id: "lab-act-prompt-msg", ariaLive: "polite" });
  const reset = el("button", { type: "button", className: "lab-small", onclick: () => setPrompt(DEFAULT_PROMPT) }, "prompt d'esempio");
  const tokens = el("div", { className: "lab-act-tokens", role: "radiogroup", ariaLabel: "Token da seguire" });
  const block = el("select", { className: "lab-act-select", ariaLabel: "Blocco" });
  const head = el("select", { className: "lab-act-select", ariaLabel: "Head" });
  const linear = el("select", { className: "lab-act-select lab-act-linear", ariaLabel: "nn.Linear di x · Wᵀ" }, ...LINEAR_CHOICES.map((c) => el("option", { value: c.id }, c.label)));
  const pending = el("p", { className: "lab-act-pending hint", hidden: true }, "Le nuove scelte valgono dal prossimo ▶ Avvia.");
  const overview = el("div", { className: "lab-act-overview" });
  const form = el(
    "section",
    { className: "lab-act-form", ariaLabel: "Scelte delle animazioni" },
    el("div", { className: "lab-act-field" }, el("label", { htmlFor: "lab-act-prompt" }, "Prompt "), reset),
    prompt,
    promptMessage,
    el("p", { className: "lab-act-label" }, "Token da seguire"),
    tokens,
    el("div", { className: "lab-act-row" }, el("label", {}, "Blocco ", block), el("label", {}, "Head ", head)),
    el("label", { className: "lab-act-wide" }, "nn.Linear di «x · Wᵀ» ", linear),
    pending,
  );
  prompt.id = "lab-act-prompt";
  pane.append(
    el("p", { className: "lab-intro" }, INTRO),
    el("p", { className: "hint lab-act-analogy" }, ANALOGY),
    status,
    form,
    overview,
    el("details", { className: "lab-details lab-act-honest" }, el("summary", {}, "Che cosa è vero (e che cosa no)"), el("ul", {}, ...HONEST.map((t) => el("li", {}, t)))),
  );

  // ---------------------------------------------------------------- stato del run A
  function vocabOf() {
    const run = state.run(state.runA);
    return run?.vocab ?? state.builtin?.vocab ?? null;
  }
  function configOf() {
    const run = state.run(state.runA);
    return run?.compatibleWithEngine ? run.engine.config : state.builtin?.config ?? null;
  }

  function renderStatus() {
    const run = state.run(state.runA);
    const entry = state.entry(state.runA);
    const view = state.current?.runId === state.runA ? state.current.view : null;
    const where = el("p", { className: "lab-act-where" }, "Run nella scena: ", el("strong", {}, entry?.name ?? "—"), ` · ${shortStepLabel(state.step, view)}`);
    if (!run) return status.replaceChildren(where);
    if (run.compatibleWithEngine) {
      status.replaceChildren(where, el("p", { className: "lab-ok-line" }, "Il motore JS lo esegue: le animazioni dei calcoli usano le sue attivazioni vere."));
      return;
    }
    const back = run.id !== COURSE_RUN_ID ? el("button", { type: "button", className: "lab-small lab-primary", onclick: () => void state.setRunA(COURSE_RUN_ID) }, "Usa il GPT del corso") : null;
    status.replaceChildren(
      where,
      el("p", { className: "lab-no" }, "Il motore JS non esegue questo run: ", el("span", {}, incompatibility(run)), "."),
      el("p", { className: "hint" }, "Le animazioni dei calcoli mostrerebbero attivazioni inventate: restano spente. «I pesi durante il training» funziona se il run ha gli snapshot."),
      back,
    );
  }

  // ---------------------------------------------------------------- scelte
  function setPrompt(text) {
    prompt.value = text;
    onPrompt();
  }

  function onPrompt() {
    const cfg = configOf();
    const check = checkPrompt(prompt.value, vocabOf(), cfg?.block_size ?? 32);
    promptMessage.textContent = check.message;
    promptMessage.classList.toggle("lab-error", !check.ok);
    prompt.setAttribute("aria-invalid", String(!check.ok));
    settings.set({ prompt: prompt.value.replace(/\r\n?/g, "\n") });
    renderTokens(check);
  }

  /** I caratteri del prompt come radiogroup (frecce, Home/Fine, tabindex mobile). */
  function renderTokens(check = checkPrompt(settings.prompt, vocabOf(), configOf()?.block_size ?? 32)) {
    if (!check.ok) {
      tokens.replaceChildren(el("span", { className: "hint" }, "Scrivi un prompt valido per scegliere il token."));
      return;
    }
    const current = resolveToken(settings.token, check.chars.length);
    tokens.replaceChildren(
      ...check.chars.map((c, i) => {
        const on = i === current;
        return el(
          "button",
          {
            type: "button",
            role: "radio",
            className: `lab-act-token${on ? " on" : ""}`,
            ariaChecked: String(on),
            tabIndex: on ? 0 : -1,
            title: `posizione ${i}: ${c === " " ? "spazio" : c === "\n" ? "a capo" : c}`,
            dataset: { index: String(i) },
            onclick: () => chooseToken(i, check.chars.length),
          },
          visibleChar(c),
        );
      }),
    );
  }

  function chooseToken(i, n, { focus = false } = {}) {
    settings.set({ token: i === n - 1 ? null : i }); // l'ultimo = "segui sempre l'ultimo", anche se il prompt cambia
    renderTokens();
    if (focus) tokens.querySelector(`[data-index="${i}"]`)?.focus();
  }

  tokens.addEventListener("keydown", (e) => {
    const buttons = [...tokens.querySelectorAll("[role=radio]")];
    if (!buttons.length) return;
    const n = buttons.length;
    const at = resolveToken(settings.token, n);
    const next = { ArrowRight: at + 1, ArrowDown: at + 1, ArrowLeft: at - 1, ArrowUp: at - 1, Home: 0, End: n - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    e.stopPropagation(); // le frecce non vanno al Percorso né alla scena
    chooseToken(Math.min(n - 1, Math.max(0, next)), n, { focus: true });
  });

  function renderSelects() {
    const cfg = configOf();
    const L = cfg?.n_layer ?? 1;
    const H = cfg?.n_head ?? 1;
    const { block: b, head: h } = settings.clamp(cfg);
    block.replaceChildren(...Array.from({ length: L }, (_, i) => el("option", { value: String(i), selected: i === b }, `blocks[${i}]`)));
    head.replaceChildren(...Array.from({ length: H }, (_, i) => el("option", { value: String(i), selected: i === h }, `heads[${i}]`)));
    linear.value = settings.linear;
  }

  block.addEventListener("change", () => settings.set({ block: Number(block.value) }));
  head.addEventListener("change", () => settings.set({ head: Number(head.value) }));
  linear.addEventListener("change", () => settings.set({ linear: linear.value }));
  prompt.addEventListener("input", onPrompt);
  // dentro il campo le frecce e lo spazio sono del testo, non del player né del Percorso
  prompt.addEventListener("keydown", (e) => e.stopPropagation());

  const offSettings = settings.on(() => {
    pending.hidden = !tab.ui.player?.action;
  });
  const offs = [
    state.on("runA", () => {
      renderStatus();
      renderSelects();
      onPrompt();
    }),
    state.on("weights", renderStatus),
    state.on("runs", renderStatus),
  ];
  tab.ui.player?.on("state", () => {
    if (!tab.ui.player.action) pending.hidden = true;
  });

  renderStatus();
  renderSelects();
  onPrompt();

  return {
    overview,
    categories: Object.values(CATEGORIES),
    refresh() {
      renderStatus();
      renderSelects();
      onPrompt();
    },
    dispose() {
      offSettings();
      for (const off of offs) off();
    },
  };
}
