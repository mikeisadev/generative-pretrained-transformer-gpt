/**
 * "Dove sei": mentre lo studente passeggia nella scena (WASD, tasto destro, pad, doppio clic…),
 * un'etichetta in basso a sinistra dice su quale dei 16 meccanismi si trova, e la cornice di quella
 * regione si accende del suo colore. Come la targa davanti a ogni sala di un museo: ti muovi
 * liberamente e sai sempre dove sei; da lì un clic per inquadrarla, leggerne il passo o animarla.
 *
 * Costo: niente per frame. Si aggiorna solo all'evento "change" delle OrbitControls (cioè quando la
 * camera si è mossa davvero) e solo mentre è visibile; trovare la regione più vicina è un
 * arrotondamento (nearestRegionAt), il DOM cambia solo quando cambia la regione.
 */
import { Color } from "three";
import { el } from "../../core/dom.js";
import { FRAME_COLOR, nearestRegionAt, ROWS } from "./regions.js";

/** Quanto colore della regione entra nello sfondo della sua cornice quando "ci sei sopra". */
const BACK_TINT = 0.07;
const TINT = new Color();

export class RegionCompass {
  /**
   * @param {object} o
   * @param {object} o.ui layout del tab (stage, host, nav, director)
   * @param {import("../../core/lod.js").LODController} o.lod
   * @param {Record<string, THREE.LineLoop>} o.frames cornice di ogni regione (livello Dettaglio)
   * @param {(regionId: string) => void} o.onFrame "Inquadra"
   * @param {(regionId: string) => string|null} o.stepFor id del passo del Percorso sulla regione
   * @param {(regionId: string) => string|null} o.actionFor id dell'azione animata della regione
   * @param {(stepId: string) => void} o.onRead "Leggi": apre il passo
   * @param {(actionId: string) => void} o.onAnimate "▶": avvia l'azione
   */
  constructor({ ui, lod, frames, onFrame, stepFor, actionFor, onRead, onAnimate }) {
    this.ui = ui;
    this.lod = lod;
    this.frames = frames;
    this.stepFor = stepFor;
    this.actionFor = actionFor;
    this.region = null;
    this.visible = false;
    this.enabled = true; // spenta nell'altro sottotab del tab (architettura completa)

    this.dot = el("span", { className: "mllm-where-dot", ariaHidden: "true" });
    this.title = el("strong");
    this.row = el("small", { className: "mllm-where-row" });
    this.readButton = el("button", { type: "button", onclick: () => this.region && onRead(this.stepFor(this.region.id)) }, "Leggi");
    this.animateButton = el("button", { type: "button", className: "mllm-where-play", onclick: () => this.region && onAnimate(this.actionFor(this.region.id)) }, "▶ Anima");
    this.root = el(
      "div",
      { className: "mllm-where", hidden: true, role: "group", ariaLabel: "Dove sei nella scena" },
      this.dot,
      el("span", { className: "mllm-where-text", ariaLive: "polite" }, el("small", {}, "Sei su "), this.title, this.row),
      el(
        "span",
        { className: "mllm-where-actions" },
        el("button", { type: "button", title: "Inquadra questo meccanismo", onclick: () => this.region && onFrame(this.region.id) }, "Inquadra"),
        this.readButton,
        this.animateButton,
      ),
    );
    ui.stage.append(this.root);

    this.offs = [
      ui.nav.on("input", (source) => source !== "home" && this.show()),
      () => ui.host.controls.removeEventListener("change", this.onChange),
      () => window.removeEventListener("resize", this.place),
    ];
    ui.host.controls.addEventListener("change", this.onChange);
    window.addEventListener("resize", this.place);
    // Il pad della navigazione (in basso a destra) si apre, si chiude e si sposta sopra il codice:
    // se l'etichetta finirebbe sotto di lui, sale sopra. Solo a questi eventi, mai per frame.
    const pad = ui.nav.pad?.pad;
    if (pad) {
      const observer = new MutationObserver(this.place);
      observer.observe(pad, { attributes: true, attributeFilter: ["class", "style"] });
      this.offs.push(() => observer.disconnect());
    }
  }

  /** Tiene l'etichetta fuori dal pad: se si sovrappongono, la porta appena sopra. */
  place = () => {
    if (!this.visible) return;
    this.root.style.bottom = "";
    const pad = this.ui.nav.pad?.pad;
    if (!pad) return;
    const p = pad.getBoundingClientRect();
    const c = this.root.getBoundingClientRect();
    const overlap = p.width > 0 && c.right > p.left && c.left < p.right && c.bottom > p.top && c.top < p.bottom;
    if (overlap) this.root.style.bottom = `${Math.round(this.ui.stage.getBoundingClientRect().bottom - p.top + 8)}px`;
  };

  /** La camera si è mossa: se l'etichetta è visibile, si aggiorna la regione. */
  onChange = () => {
    if (!this.visible) return;
    if (this.lod.current === "macro") return this.hide(); // nella Panoramica non ci sono regioni
    const t = this.ui.host.controls.target;
    this.setRegion(nearestRegionAt(t.x, t.y));
  };

  /** Lo studente si muove da solo: compare l'etichetta (non durante un'azione, che ha già la sua regione). */
  show() {
    if (!this.enabled || this.ui.director.actionActive || this.lod.current === "macro") return;
    const wasVisible = this.visible;
    this.visible = true;
    this.root.hidden = false;
    this.onChange();
    if (!wasVisible) this.place();
  }

  /** Il tab inquadra qualcosa da sé (passo, vista, azione, "Centra"): l'etichetta non serve più. */
  hide() {
    if (!this.visible) return;
    this.visible = false;
    this.root.hidden = true;
    this.setRegion(null);
  }

  /** Accende o spegne l'etichetta (spenta: nascosta e sorda ai movimenti della camera). */
  setEnabled(enabled) {
    this.enabled = Boolean(enabled);
    if (!this.enabled) this.hide();
  }

  setRegion(region) {
    if (region === this.region) return;
    if (this.region) this.paintFrame(this.region, false);
    this.region = region;
    if (!region) return;
    this.paintFrame(region, true);
    this.dot.style.background = region.color;
    this.title.textContent = region.title;
    this.title.style.color = region.color;
    this.row.textContent = ` · ${ROWS[region.row].title}`;
    const action = this.actionFor(region.id);
    this.animateButton.hidden = !action;
    this.animateButton.title = action ? "Avvia l'animazione di questo meccanismo" : "";
    this.readButton.title = "Apri il passo del Percorso su questo meccanismo";
    this.place(); // il titolo nuovo può allargare l'etichetta
  }

  /** Accende (o spegne) la cornice della regione: linea del suo colore e sfondo appena tinto. */
  paintFrame(region, on) {
    const line = this.frames[region.id];
    if (!line) return;
    line.material.color.set(on ? region.color : FRAME_COLOR);
    const back = line.userData.back;
    if (!back) return;
    back.userData.baseColor ??= back.material.color.clone();
    back.material.color.copy(back.userData.baseColor);
    // la miscela si fa nello spazio sRGB (come la vede l'occhio): in quello lineare un 7% di rosa
    // su uno sfondo quasi nero diventerebbe una macchia viola
    if (on) back.material.color.convertLinearToSRGB().lerp(TINT.set(region.color).convertLinearToSRGB(), BACK_TINT).convertSRGBToLinear();
  }

  dispose() {
    this.hide();
    for (const off of this.offs) off();
    this.root.remove();
  }
}
