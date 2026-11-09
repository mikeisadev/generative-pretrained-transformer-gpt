/**
 * LOD semantico ("Level Of Detail"): cosa si vede dipende da quanto sei lontano.
 *
 * Esempio nel tab Transformer:
 *   vicino  → "Dettaglio": ogni matrice di pesi, ogni head
 *   medio   → "Blocchi":   un rettangolo per componente, con titolo e descrizione
 *   lontano → "Panoramica": un solo rettangolo "Transformer" con input e output
 *
 * Ogni livello è un THREE.Group. Il controller misura la distanza camera → punto
 * guardato (lo zoom delle OrbitControls) e sfuma i gruppi tra loro. L'utente può
 * anche bloccare un livello con il selettore (Auto / Dettaglio / Blocchi / Panoramica).
 */
import { el } from "./dom.js";

const HYSTERESIS = 0.08; // ±8% attorno alle soglie: niente sfarfallio vicino al confine
const FADE_PER_SECOND = 3.5;

/** Applica un'opacità (0..1) a tutti i materiali del gruppo, ricordando quella originale. */
export function setGroupOpacity(group, alpha) {
  group.visible = alpha > 0.01;
  if (!group.visible) return;
  group.traverse((child) => {
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    for (const m of materials) {
      if (m.userData.baseOpacity === undefined) {
        m.userData.baseOpacity = m.opacity;
        m.userData.baseTransparent = m.transparent;
      }
      m.opacity = m.userData.baseOpacity * alpha;
      const transparent = m.userData.baseTransparent || alpha < 0.999;
      if (m.transparent !== transparent) {
        m.transparent = transparent;
        m.needsUpdate = true;
      }
    }
  });
}

/**
 * Cambia l'opacità "di base" di un materiale gestito dal LOD (per animazioni di opacità che devono
 * convivere con le sfumature tra livelli): il LOD calcola opacity = baseOpacity × alpha del livello,
 * quindi qui si aggiorna la base e si riapplica l'alpha in corso.
 */
export function setBaseOpacity(material, opacity) {
  const ud = material.userData;
  const base = ud.baseOpacity ?? material.opacity;
  const alpha = base > 1e-6 ? material.opacity / base : 1;
  if (ud.baseOpacity === undefined) ud.baseTransparent = material.transparent;
  ud.baseOpacity = opacity;
  material.opacity = opacity * alpha;
  const transparent = ud.baseTransparent || opacity < 0.999 || alpha < 0.999;
  if (material.transparent !== transparent) {
    material.transparent = transparent;
    material.needsUpdate = true;
  }
}

export class LODController {
  /**
   * @param {import("./sceneHost.js").SceneHost} host
   * @param {object} options
   * @param {{id: string, label: string, maxDistance: number, group: THREE.Object3D}[]} options.levels
   *        dal più vicino al più lontano; l'ultimo livello di solito ha maxDistance = Infinity
   * @param {HTMLElement} [options.widgetParent] dove disegnare il selettore dei livelli
   * @param {(id: string) => void} [options.onChange] a ogni cambio del livello attivo
   * @param {(mode: string) => void} [options.onPick] quando lo STUDENTE sceglie un livello nel selettore
   *        (non quando lo fa il codice con setMode): di solito per inquadrare quel livello
   * @param {() => number} [options.distanceScale] moltiplica tutte le soglie (es. > 1 su schermi stretti,
   *        dove per far entrare lo stesso oggetto la camera deve stare più lontana)
   */
  constructor(host, { levels, widgetParent = null, onChange = () => {}, onPick = null, distanceScale = () => 1 }) {
    this.host = host;
    this.levels = levels;
    this.onChange = onChange;
    this.onPick = onPick;
    this.distanceScale = distanceScale;
    this.mode = "auto";
    this.enabled = true;
    this.alphas = new Map(levels.map((l) => [l.id, 0]));
    this.current = null;
    this.widget = widgetParent ? this.renderWidget(widgetParent) : null;
    this.unsubscribe = host.onFrame((dt) => this.update(dt));
    this.setActive(this.pickByDistance(host.zoomDistance()), true);
  }

  /**
   * Accende o spegne il controller, per un tab con due scene alternative nello stesso SceneHost (es. i
   * due sottotab di "LLM moderni"): spento non tocca più i suoi gruppi a ogni frame e nasconde il suo
   * selettore; chi lo spegne decide se nascondere i gruppi. Riacceso, riparte senza dissolvenza dal
   * livello giusto (quello bloccato, o quello della distanza attuale).
   */
  setEnabled(enabled) {
    enabled = Boolean(enabled);
    if (enabled === this.enabled) return;
    this.enabled = enabled;
    if (this.widget) this.widget.style.display = enabled ? "" : "none";
    if (!enabled) return;
    this.current = null;
    this.setActive(this.mode === "auto" ? this.pickByDistance(this.host.zoomDistance()) : this.mode, true);
    for (const level of this.levels) {
      level.group.userData.lodAlpha = this.alphas.get(level.id);
      setGroupOpacity(level.group, this.alphas.get(level.id));
    }
  }

  pickByDistance(distance) {
    const d = distance / Math.max(1e-6, this.distanceScale());
    const index = this.levels.findIndex((l) => l.id === this.current);
    for (let i = 0; i < this.levels.length; i++) {
      const level = this.levels[i];
      // isteresi: per uscire dal livello corrente serve superare la soglia di un 8% in più
      const slack = i === index ? 1 + HYSTERESIS : i === index - 1 ? 1 - HYSTERESIS : 1;
      if (d <= level.maxDistance * slack) return level.id;
    }
    return this.levels.at(-1).id;
  }

  /** "auto" oppure l'id di un livello da bloccare. */
  setMode(mode) {
    this.mode = mode;
    this.widget?.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    if (mode !== "auto") this.setActive(mode);
  }

  /** Scelta dal selettore: come setMode, più il richiamo onPick (es. per inquadrare il livello). */
  pick(mode) {
    this.setMode(mode);
    this.onPick?.(mode);
  }

  setActive(id, immediate = false) {
    if (id === this.current) return;
    this.current = id;
    if (immediate) for (const l of this.levels) this.alphas.set(l.id, l.id === id ? 1 : 0);
    this.widget?.querySelector("[data-current]")?.replaceChildren(this.levels.find((l) => l.id === id).label);
    this.onChange(id);
  }

  update(dt) {
    if (!this.enabled) return;
    if (this.mode === "auto") this.setActive(this.pickByDistance(this.host.zoomDistance()));
    for (const level of this.levels) {
      const target = level.id === this.current ? 1 : 0;
      const a = this.alphas.get(level.id);
      const next = target > a ? Math.min(target, a + dt * FADE_PER_SECOND) : Math.max(target, a - dt * FADE_PER_SECOND);
      if (next !== a || level.group.userData.lodAlpha === undefined) {
        this.alphas.set(level.id, next);
        level.group.userData.lodAlpha = next;
        setGroupOpacity(level.group, next);
      }
    }
  }

  renderWidget(parent) {
    const buttons = [{ mode: "auto", label: "Auto" }, ...this.levels.map((l) => ({ mode: l.id, label: l.label }))];
    const widget = el(
      "div",
      { className: "lod-switch", role: "group", ariaLabel: "Livello di dettaglio" },
      el("span", { className: "lod-title" }, "Zoom: ", el("strong", { dataset: { current: "" } })),
      ...buttons.map((b) =>
        el("button", { type: "button", className: b.mode === "auto" ? "active" : "", dataset: { mode: b.mode }, onclick: () => this.pick(b.mode) }, b.label),
      ),
    );
    parent.append(widget);
    return widget;
  }

  dispose() {
    this.unsubscribe();
    this.widget?.remove();
  }
}
