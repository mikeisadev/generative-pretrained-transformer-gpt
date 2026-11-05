/**
 * Pannello "Azioni" della sidebar: panoramica delle azioni divise per categoria
 * ("Prompt (inferenza)" e "Training"), con il form del prompt, le impostazioni del training,
 * gli approfondimenti con le formule e le fonti.
 *
 * Le schede delle azioni le disegna ActionPlayer.renderOverview (così "Avvia" e l'evidenziazione
 * dell'azione attiva restano quelle del nucleo); qui aggiungiamo dentro ogni categoria i nostri
 * contenuti. Le azioni registrate da altri moduli compaiono in coda, nelle loro categorie.
 */
import { el, fillTemplate } from "../../../core/dom.js";
import { mathBox } from "../../../core/ui/menu.js";
import { VocabularyError } from "../../../engine/gpt.js";
import { DEFAULT_PROMPT, PROMPT_LIMITS, checkPrompt } from "./promptRun.js";
import { CATEGORIES, DEEP_PROMPT, DEEP_TRAINING, INTRO, SOURCES } from "./texts.js";

const html = (tag, className, markup) => {
  const node = el(tag, { className });
  node.innerHTML = markup; // testi nostri (texts.js), non input dell'utente
  return node;
};

/** Controllo del vocabolario senza aspettare il motore: model.json ha già i 65 caratteri. */
function vocabularyChecker(vocab) {
  const stoi = new Map(vocab.map((c, i) => [c, i]));
  return {
    encode(text) {
      const unknown = [...new Set([...text].filter((c) => !stoi.has(c)))].sort();
      if (unknown.length) throw new VocabularyError(unknown);
      return [...text].map((c) => stoi.get(c));
    },
  };
}

export class ActionsPane {
  constructor(tab, { onNeedTraining }) {
    this.tab = tab;
    this.player = tab.ui.player;
    this.pane = tab.panes.actions;
    this.vocab = vocabularyChecker(tab.spec.vocab);
    this.blockSize = tab.spec.config.block_size;
    this.onNeedTraining = onNeedTraining;
    this.buildPromptForm();
    this.buildTrainingSettings();
    this.overview = el("div", { className: "tfa-overview" });
    // riquadri con le formule: costruiti una volta (KaTeX), poi solo spostati a ogni render
    this.deepPrompt = DEEP_PROMPT.map((d) => mathBox(d.items, {}, d.title));
    this.deepTraining = DEEP_TRAINING.map((d) => mathBox(d.items, {}, d.title));
    this.sources = el(
      "details",
      { className: "deep tfa-sources" },
      el("summary", {}, "Fonti"),
      el("ul", {}, ...SOURCES.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener" }, s.title), el("span", { className: "hint" }, ` · ${s.note}`)))),
    );
    this.pane.classList.add("tfa");
    this.pane.append(html("p", "tfa-intro", INTRO), this.overview, this.sources);
    this.renderedCount = -1;
    tab.ui.sidebar.onSelect((id) => {
      if (id !== "azioni") return;
      this.render();
      this.onNeedTraining();
    });
  }

  /* ----------------------------------------------------------------- prompt */

  buildPromptForm() {
    const L = PROMPT_LIMITS;
    this.promptInput = el("textarea", { className: "tfa-prompt", rows: 6, spellcheck: false, value: DEFAULT_PROMPT, ariaLabel: "Prompt da continuare", oninput: () => this.validate() });
    this.promptMessage = el("p", { className: "tfa-message", ariaLive: "polite" });
    this.temperature = el("input", { type: "range", min: L.temperature.min, max: L.temperature.max, step: L.temperature.step, value: L.temperature.value, oninput: () => this.syncTemperature() });
    this.temperatureOut = el("output", { className: "mono" });
    this.tokens = el("input", { type: "number", min: L.tokens.min, max: L.tokens.max, value: L.tokens.value, className: "tfa-number", oninput: () => this.validate() });
    this.seed = el("input", { type: "number", min: L.seed.min, max: L.seed.max, value: L.seed.value, className: "tfa-number", oninput: () => this.validate() });
    const reseed = el("button", { type: "button", className: "tfa-small", title: "Nuovo seed casuale", onclick: () => ((this.seed.value = String(Math.floor(Math.random() * 100000))), this.validate()) }, "nuovo");
    const reset = el("button", { type: "button", className: "tfa-small", onclick: () => ((this.promptInput.value = DEFAULT_PROMPT), this.validate()) }, "prompt d'esempio");
    this.form = el(
      "div",
      { className: "tfa-form" },
      el("label", { className: "tfa-field" }, el("span", {}, "Prompt ", reset), this.promptInput),
      this.promptMessage,
      el("div", { className: "tfa-row" },
        el("label", { className: "tfa-field" }, el("span", {}, "Temperatura ", this.temperatureOut), this.temperature),
      ),
      el("div", { className: "tfa-row" },
        el("label", { className: "tfa-field" }, el("span", {}, "Token da generare"), this.tokens),
        el("label", { className: "tfa-field" }, el("span", {}, "Seed ", reseed), this.seed),
      ),
      el("p", { className: "hint" }, "Lo stesso seed con lo stesso prompt dà sempre lo stesso testo. Il modello conosce solo i 65 caratteri di Tiny Shakespeare: lettere inglesi senza accenti, spazio, a capo, la cifra 3 e ! $ & ' , - . : ; ?"),
    );
    this.output = el("div", { className: "tfa-output", hidden: true }, el("strong", {}, "Testo generato"), el("pre", { className: "tfa-output-text" }));
    this.syncTemperature();
    this.validate();
  }

  syncTemperature() {
    this.temperatureOut.value = Number(this.temperature.value).toFixed(2).replace(".", ",");
  }

  /** Controlla il prompt e i numeri; abilita o disabilita "Avvia". Restituisce le opzioni o null. */
  validate() {
    const text = this.promptInput.value.replace(/\r\n?/g, "\n");
    const result = checkPrompt(this.vocab, text);
    const L = PROMPT_LIMITS;
    const tokens = Math.round(Number(this.tokens.value));
    const seed = Math.round(Number(this.seed.value));
    let message = "";
    let ok = result.ok;
    if (!result.ok) message = result.message;
    else if (!(tokens >= L.tokens.min && tokens <= L.tokens.max)) {
      ok = false;
      message = `Il numero di token da generare va da ${L.tokens.min} a ${L.tokens.max}.`;
    } else if (!(seed >= L.seed.min && seed <= L.seed.max)) {
      ok = false;
      message = "Il seed deve essere un intero positivo.";
    } else {
      const n = result.ids.length;
      message = n > this.blockSize ? `${n} caratteri = ${n} token: il modello ne vedrà gli ultimi ${this.blockSize}.` : `${n} caratteri = ${n} token: stanno tutti nel contesto (${this.blockSize}).`;
    }
    this.promptMessage.textContent = message;
    this.promptMessage.classList.toggle("error", !ok);
    this.promptInput.classList.toggle("invalid", !result.ok);
    this.valid = ok;
    const button = this.overview?.querySelector('[data-action="prompt"] button');
    if (button) button.disabled = !ok;
    return ok ? { text, temperature: Number(this.temperature.value), tokens, seed } : null;
  }

  /** Opzioni correnti per l'azione (lancia un errore leggibile se non sono valide). */
  promptOptions() {
    const options = this.validate();
    if (!options) throw new Error(this.promptMessage.textContent);
    return options;
  }

  /** Testo generato nel riquadro della sidebar (prompt in grigio, generati evidenziati). */
  showOutput(run, count) {
    this.output.hidden = false;
    const pre = this.output.querySelector("pre");
    const generated = run.sequence.slice(run.promptIds.length, run.promptIds.length + count).map((id) => run.vocab[id]).join("");
    pre.replaceChildren(el("span", { className: "tfa-out-prompt" }, run.text), el("span", { className: "tfa-out-generated" }, generated));
  }

  /* ----------------------------------------------------------------- training */

  buildTrainingSettings() {
    this.startStep = el("select", { className: "tfa-select", ariaLabel: "Step di partenza del microscopio" }, el("option", { value: "0" }, "step 0 (pesi casuali)"));
    this.trainingSettings = el(
      "div",
      { className: "tfa-form" },
      el("label", { className: "tfa-field" }, el("span", {}, "Step di partenza per il microscopio"), this.startStep),
      el("p", { className: "hint" }, "Il microscopio parte dai pesi registrati a quello step e arriva allo snapshot successivo."),
    );
  }

  /** Popola la scelta dello step con gli snapshot veri (tutti tranne l'ultimo). */
  setSnapshotSteps(steps) {
    const current = this.startStep.value;
    this.startStep.replaceChildren(
      ...steps.slice(0, -1).map((s, i) => el("option", { value: String(s) }, s === 0 ? "step 0 (pesi casuali)" : `step ${s} → ${steps[i + 1]}`)),
    );
    this.startStep.value = steps.includes(Number(current)) ? current : "0";
  }

  startStepValue() {
    return Number(this.startStep.value) || 0;
  }

  /* ----------------------------------------------------------------- panoramica */

  /** (Ri)disegna la panoramica se nel frattempo altri moduli hanno registrato azioni. */
  render(force = false) {
    const count = this.player.actions.size;
    if (!force && count === this.renderedCount) return;
    this.renderedCount = count;
    const mine = [CATEGORIES.prompt.title, CATEGORIES.training.title];
    const others = [...new Set([...this.player.actions.values()].map((a) => a.category))].filter((c) => !mine.includes(c));
    this.player.renderOverview(this.overview, { categories: [...mine, ...others] });
    const section = (title) => [...this.overview.querySelectorAll(".action-group")].find((s) => s.querySelector("h3")?.textContent === title);
    const prompt = section(CATEGORIES.prompt.title);
    if (prompt) {
      prompt.classList.add("tfa-group", "tfa-group-prompt");
      const params = this.tab.spec.total_params.toLocaleString("it-IT");
      prompt.querySelector("h3").after(html("p", "tfa-lead", fillTemplate(CATEGORIES.prompt.lead, { params })), this.form);
      prompt.append(this.output, ...this.deepPrompt);
    }
    const training = section(CATEGORIES.training.title);
    if (training) {
      training.classList.add("tfa-group", "tfa-group-training");
      training.querySelector("h3").after(html("p", "tfa-lead", CATEGORIES.training.lead), this.trainingSettings);
      training.append(...this.deepTraining);
    }
    this.validate();
  }
}
