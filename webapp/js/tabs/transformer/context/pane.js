/**
 * Pannello "Contesto" della sidebar: spiegazione, scelta della domanda, avvio dell'animazione,
 * collegamento al tab "Database vettoriale (RAG)", percorso passo per passo (Menu del nucleo,
 * con formule KaTeX) e fonti. Più un riquadro breve nel pannello "Viste".
 */
import { el } from "../../../core/dom.js";
import { Menu } from "../../../core/ui/menu.js";
import { createViewInfo, deepBox, formula, legend, sources } from "../views/panel.js";
import { ROLE_STYLE, SUGGESTED_QUESTIONS } from "./pipeline.js";
import { INTRO, SOURCES, pathSteps } from "./texts.js";

export class ContextPane {
  /**
   * @param {object} tab
   * @param {{onQuestion: (q: string) => void, onSelect: (item: object, title: string) => void,
   *          onPlay: () => void, onRag: () => void, onShow: () => void}} handlers
   */
  constructor(tab, handlers) {
    this.tab = tab;
    this.handlers = handlers;
    this.pane = tab.ui.sidebar.addPane({ id: "contesto", title: "Contesto", hint: "Dal messaggio dell'utente al contesto completo che riceve l'LLM: la fase che prepara l'input della pila." });
    this.pane.classList.add("ctx-pane");
    this.input = el("textarea", { className: "ctx-question", rows: 3, spellcheck: false, ariaLabel: "Domanda dell'utente al chatbot del gestionale" });
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) this.submit();
    });
    this.status = el("p", { className: "ctx-status", ariaLive: "polite" });
    this.menuHost = el("div", { className: "ctx-menu" });
    this.render();
    this.renderViewInfo();
  }

  render() {
    const roles = ["tools", "system", "history", "question", "document", "tool-call", "tool-result"];
    this.pane.append(
      el("h3", { className: "ctx-title" }, "Il contesto: ciò che l'LLM legge davvero"),
      el("p", {}, INTRO),
      el(
        "p",
        { className: "ctx-note" },
        el("b", {}, "Simulazione didattica. "),
        "Guardrail, embedding, ricerca e prompt usano i moduli veri della libreria RAG del corso. La scelta dello strumento, i dati dei dipendenti e la risposta finale sono esempi: il nostro GPT è addestrato solo su Shakespeare, mentre nel corso risponde un LLM di produzione (gpt-4.1-mini). La struttura, un transformer decoder, è la stessa.",
      ),
      el(
        "div",
        { className: "ctx-form" },
        el("label", { className: "ctx-label" }, "Domanda al chatbot del gestionale", this.input),
        el(
          "div",
          { className: "ctx-chips", role: "group", ariaLabel: "Domande d'esempio" },
          ...SUGGESTED_QUESTIONS.map((q) => el("button", { type: "button", className: "ctx-chip", title: q.question, onclick: () => this.submit(q.question) }, q.label)),
        ),
        el("div", { className: "ctx-row" }, el("button", { type: "button", onclick: () => this.submit() }, "Calcola il contesto"), this.status),
      ),
      el("button", { type: "button", className: "tf-start ctx-play", onclick: () => this.handlers.onPlay() }, "▶ Animazione: dalla domanda al contesto dell'LLM"),
      el("button", { type: "button", className: "ctx-rag", onclick: () => this.handlers.onRag() }, "Approfondisci nel tab Database vettoriale"),
      el("p", { className: "hint" }, "Il tab Database vettoriale riceve la stessa domanda e mostra nel dettaglio chunking, embedding e ricerca HNSW."),
      el("h4", { className: "ctx-sub" }, "Le parti del contesto"),
      legend(roles.map((r) => ({ color: ROLE_STYLE[r].color, label: ROLE_STYLE[r].label })).filter((x, i, a) => a.findIndex((y) => y.label === x.label) === i)),
      el("h4", { className: "ctx-sub" }, "Percorso, passo per passo"),
      el("p", { className: "hint" }, "Clicca un passo: la camera vola sulla fase e a destra compare il codice Spring AI delle lezioni."),
      this.menuHost,
      deepBox(
        "Approfondimento: quanto costa una domanda",
        el("p", {}, "Il provider fa pagare i token letti e quelli scritti, a ogni chiamata. Con uno strumento le chiamate sono due e la seconda rilegge tutto il contesto della prima."),
        formula(String.raw`\text{token totali} = \sum_{k=1}^{K} \big(N^{(k)}_{\text{in}} + N^{(k)}_{\text{out}}\big),\qquad K = 1 + \text{giri di strumenti}`),
        el("p", { className: "hint" }, "Con il prompt caching la parte iniziale identica (strumenti e system prompt) costa meno, se supera la lunghezza minima richiesta dal provider (L18 §2.6)."),
      ),
      sources(SOURCES),
    );
  }

  renderViewInfo() {
    const info = createViewInfo(this.tab, "contesto");
    info.append(
      el("h3", {}, "Contesto: dalla domanda all'LLM"),
      el("p", {}, "A sinistra della pila compare la fase che prepara l'input: guardrail, embedding, ricerca su pgvector, assemblaggio del prompt, strumenti e streaming della risposta. Una freccia entra nei token di input del transformer."),
      el("p", { className: "hint" }, "Rotellina: da vicino le schede con i dati veri, poi un rettangolo per fase, da lontano il diagramma Contesto → LLM → Risposta."),
      el("button", { type: "button", className: "tf-start", onclick: () => this.handlers.onShow() }, "Apri il pannello Contesto"),
    );
    this.info = info;
  }

  setVisible(v) {
    if (this.info) this.info.hidden = !v;
  }

  submit(question = this.input.value) {
    const q = String(question).trim();
    this.input.value = q;
    if (!q) {
      this.setStatus("Scrivi una domanda.", true);
      return;
    }
    this.handlers.onQuestion(q);
  }

  setStatus(text, error = false) {
    this.status.textContent = text;
    this.status.classList.toggle("error", error);
  }

  /** Nuovo scenario: aggiorna la domanda mostrata e ricostruisce il percorso con i numeri veri. */
  setScenario(s, vars) {
    this.input.value = s.question;
    const first = s.blocked ? "Bloccata dal guardrail" : `${s.search.results.length} chunk · ${s.prompt.tokens.total.toLocaleString("it-IT")} token · ${s.calls.length} ${s.calls.length === 1 ? "chiamata" : "chiamate"}`;
    this.setStatus(first);
    const current = this.menu?.current;
    this.menuHost.replaceChildren();
    this.menu = new Menu(this.menuHost, pathSteps(s), { vars, onSelect: (item, title) => this.handlers.onSelect(item, title) });
    if (current) this.menu.items.get(current)?.wrapper.classList.add("active");
  }
}
