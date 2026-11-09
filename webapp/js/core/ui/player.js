/**
 * ActionPlayer: esegue "azioni" animate passo per passo, con barra di controllo
 * (play/pausa, passo avanti/indietro, ricomincia, velocità) e didascalia.
 *
 * Un'azione:
 *   {
 *     id, title, category,            // category raggruppa le azioni nella panoramica (es. "Prompt", "Training")
 *     summary,                        // una frase per la panoramica
 *     steps?: Step[],                 // passi fissi, oppure
 *     build?: async (ctx) => Step[],  // passi costruiti al momento (es. tanti passi quanti token generati);
 *                                     // ctx.isStale() diventa true se l'utente ferma o riavvia durante l'attesa
 *     onStart?(ctx), onEnd?(ctx), onStop?(ctx)
 *   }
 * Un passo:
 *   {
 *     title, caption,                 // caption: testo semplice mostrato nella barra
 *     duration = 2,                   // secondi a velocità 1×
 *     enter?(ctx), update?(p, dt, ctx), exit?(ctx),   // p va da 0 a 1 durante il passo
 *   }
 * enter deve portare la scena nello stato di inizio passo: così "passo indietro" funziona sempre.
 * ctx = { player, action, step, index, data } — `data` è un oggetto condiviso tra i passi dell'azione.
 */
import { el } from "../dom.js";

const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4];

export class ActionPlayer {
  /**
   * @param {import("../sceneHost.js").SceneHost} host il ciclo di animazione che fa avanzare i passi
   * @param {HTMLElement} parent dove disegnare la barra di controllo (di solito lo stage del tab)
   */
  constructor(host, parent) {
    this.host = host;
    this.actions = new Map();
    this.action = null;
    this.steps = [];
    this.index = -1;
    this.t = 0;
    this.playing = false;
    this.speed = 1;
    this.data = {};
    this.run = 0; // cresce a ogni start/stop: una preparazione (build) superata non deve partire
    this.listeners = { step: new Set(), state: new Set(), end: new Set(), restart: new Set() };
    // Gli elenchi disegnati da renderOverview (un tab può averne più di uno, es. i due sottotab di LLM
    // moderni): la scheda dell'azione in corso si accende in tutti.
    this.overviews = new Set();
    this.renderBar(parent);
    host.onFrame((dt) => this.tick(dt));
  }

  // ---------------------------------------------------------------- registro
  register(action) {
    this.actions.set(action.id, action);
    return action;
  }

  /** Panoramica delle azioni raggruppate per categoria, con pulsante "Avvia". */
  renderOverview(container, { categories } = {}) {
    const groups = new Map();
    for (const action of this.actions.values()) {
      if (!groups.has(action.category)) groups.set(action.category, []);
      groups.get(action.category).push(action);
    }
    const order = categories ?? [...groups.keys()];
    container.replaceChildren(
      ...order.filter((c) => groups.has(c)).map((category) =>
        el(
          "section",
          { className: "action-group" },
          el("h3", {}, category),
          ...groups.get(category).map((action) =>
            el(
              "article",
              { className: "action-card", dataset: { action: action.id } },
              el("div", {}, el("strong", {}, action.title), el("p", { className: "hint" }, action.summary ?? "")),
              el("button", { type: "button", onclick: () => this.start(action.id) }, "▶ Avvia"),
            ),
          ),
        ),
      ),
    );
    this.overview = container; // l'ultimo disegnato (compatibilità)
    this.overviews.add(container);
    this.highlightCard(this.action?.id ?? null);
  }

  /** Accende la scheda dell'azione `id` (null: nessuna) in ogni elenco ancora nella pagina. */
  highlightCard(id) {
    for (const list of this.overviews) {
      if (!list.isConnected) {
        this.overviews.delete(list);
        continue;
      }
      list.querySelectorAll(".action-card").forEach((c) => c.classList.toggle("active", c.dataset.action === id));
    }
  }

  // ---------------------------------------------------------------- controllo
  async start(id) {
    const action = this.actions.get(id);
    if (!action) throw new Error(`azione sconosciuta: ${id}`);
    this.stop();
    const run = ++this.run;
    this.action = action;
    this.data = {};
    this.bar.hidden = false;
    this.title.textContent = action.title;
    this.caption.textContent = "Preparazione…";
    this.highlightCard(id);
    const ctx = this.ctx();
    ctx.isStale = () => run !== this.run; // per le build lunghe: smetti di toccare la scena se true
    await action.onStart?.(ctx);
    if (run !== this.run) return; // fermata o sostituita durante onStart
    const steps = action.steps ?? (await action.build?.(ctx)) ?? [];
    if (run !== this.run) return; // fermata o sostituita durante la preparazione
    this.steps = steps;
    this.index = -1;
    this.playing = true;
    this.goTo(0);
    this.emit("state");
  }

  stop() {
    if (!this.action) return;
    this.run++;
    this.currentStep()?.exit?.(this.ctx());
    this.action.onStop?.(this.ctx());
    this.action = null;
    this.steps = [];
    this.index = -1;
    this.playing = false;
    this.bar.hidden = true;
    this.highlightCard(null);
    this.emit("state");
  }

  toggle() {
    if (!this.action) return;
    if (this.index >= this.steps.length - 1 && this.t >= 1) {
      this.goTo(0);
      this.playing = true;
    } else this.playing = !this.playing;
    this.emit("state");
  }

  next() {
    if (this.index < this.steps.length - 1) this.goTo(this.index + 1);
  }

  prev() {
    this.goTo(Math.max(0, this.index - (this.t > 0.15 ? 0 : 1)));
  }

  restart() {
    if (!this.action) return;
    this.emit("restart"); // prima del primo passo: chi ascolta (la regia) torna a seguire l'azione
    this.goTo(0);
    this.playing = true;
    this.emit("state");
  }

  setSpeed(speed) {
    this.speed = speed;
  }

  goTo(index) {
    const ctx = this.ctx();
    const current = this.currentStep();
    if (current) {
      current.update?.(1, 0, ctx);
      current.exit?.(ctx);
    }
    this.index = index;
    this.t = 0;
    const step = this.currentStep();
    if (!step) return;
    step.enter?.(this.ctx());
    step.update?.(0, 0, this.ctx());
    this.stepLabel.textContent = `${index + 1}/${this.steps.length} · ${step.title ?? ""}`;
    this.caption.textContent = step.caption ?? "";
    this.emit("step");
  }

  currentStep() {
    return this.steps[this.index] ?? null;
  }

  ctx() {
    return { player: this, action: this.action, step: this.currentStep(), index: this.index, data: this.data };
  }

  tick(dt) {
    if (!this.action || !this.playing) return;
    const step = this.currentStep();
    if (!step) return;
    const duration = Math.max(0.05, step.duration ?? 2);
    this.t = Math.min(1, this.t + (dt * this.speed) / duration);
    step.update?.(this.t, dt * this.speed, this.ctx());
    this.progress.style.width = `${((this.index + this.t) / this.steps.length) * 100}%`;
    if (this.t >= 1) {
      if (this.index < this.steps.length - 1) this.goTo(this.index + 1);
      else {
        this.playing = false;
        step.exit?.(this.ctx());
        this.action.onEnd?.(this.ctx());
        this.emit("end");
        this.emit("state");
      }
    }
  }

  // ---------------------------------------------------------------- eventi
  /** type: "step" | "state" | "end" | "restart" ("⟲ Ricomincia") */
  on(type, fn) {
    this.listeners[type].add(fn);
    return () => this.listeners[type].delete(fn);
  }

  emit(type) {
    if (type === "state") this.playButton.textContent = this.playing ? "⏸" : "▶";
    for (const fn of this.listeners[type]) fn(this.ctx());
  }

  // ---------------------------------------------------------------- UI
  renderBar(parent) {
    this.title = el("strong");
    this.stepLabel = el("span", { className: "muted" });
    this.caption = el("p", { className: "player-caption", ariaLive: "polite" });
    this.progress = el("div", { className: "player-progress-fill" });
    this.playButton = el("button", { type: "button", title: "Play/pausa (spazio)", onclick: () => this.toggle() }, "▶");
    const speed = el(
      "select",
      { title: "Velocità", onchange: (e) => this.setSpeed(Number(e.target.value)) },
      ...SPEEDS.map((s) => el("option", { value: String(s), selected: s === 1 }, `${s}×`)),
    );
    this.bar = el(
      "div",
      { className: "player-bar", hidden: true },
      el("div", { className: "player-head" }, this.title, this.stepLabel),
      this.caption,
      el("div", { className: "player-progress" }, this.progress),
      el(
        "div",
        { className: "player-controls" },
        el("button", { type: "button", title: "Ricomincia", onclick: () => this.restart() }, "⟲"),
        el("button", { type: "button", title: "Passo indietro", onclick: () => this.prev() }, "⏮"),
        this.playButton,
        el("button", { type: "button", title: "Passo avanti", onclick: () => this.next() }, "⏭"),
        speed,
        el("button", { type: "button", title: "Chiudi l'azione", onclick: () => this.stop() }, "✕"),
      ),
    );
    parent.append(this.bar);
  }
}
