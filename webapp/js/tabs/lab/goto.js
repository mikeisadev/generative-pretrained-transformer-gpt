/**
 * Link diretti e ricerca globale del Laboratorio pesi (app.goto → instance.goto): tensor:, run:, view:,
 * action: (links.js li riconosce) più i tipi registrati dalla fase 2 con tab.links.register.
 */
import { showToast } from "../../core/ui/share.js";
import { resolveTarget } from "./links.js";
import { COURSE_RUN_ID } from "./state.js";

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });

const narrow = window.matchMedia("(max-width: 900px)");

export /**
 * Link diretti e ricerca globale (app.goto → instance.goto). Prima le rappresentazioni del nucleo
 * (view:rete, view:cervello), poi i tipi registrati dalla fase 2, poi quelli del tab (links.js).
 * false = non trovato (il guscio avvisa lo studente).
 */
async function gotoTarget(tab, target = {}) {
  const { ui, state } = tab;
  const shown = tab.reps.goto(target);
  if (shown) return shown;
  ui.director.noteUserIntent(1500);
  const custom = tab.links.handler(target.kind);
  if (custom) return custom(target.id);
  if (target.kind === "view" && tab.views.has(target.id)) {
    await tab.views.open(target.id);
    return true;
  }
  const t = resolveTarget(target);
  if (!t) return false;
  if (t.type === "action") {
    if (!ui.player?.actions.has(t.id)) return false;
    if (narrow.matches) ui.sidebar.toggleCollapsed(true);
    void ui.player.start(t.id);
    return true;
  }
  if (t.type === "run") {
    if (!state.has(t.id)) return false;
    const ok = await state.setRunA(t.id);
    tab.openPane("run");
    return ok;
  }
  if (t.type === "tensor") return gotoTensor(tab, t.name, t.step);
  if (t.type === "step") {
    let id = state.runA;
    if (t.step !== null && !state.stepsOf(id).length) id = COURSE_RUN_ID; // il run A non ha la storia: quella del corso
    if (id !== state.runA) await state.setRunA(id, { step: t.step });
    else await state.setStep(t.step);
    tab.openPane("modello", { focus: "step" });
    return true;
  }
  // t.type === "view"
  if (t.id === "centro") {
    if (narrow.matches) ui.sidebar.toggleCollapsed(true);
    await tab.scene.home();
    return true;
  }
  if (t.id === "carica" || t.id === "esporta") {
    tab.openPane("run");
    if (t.id === "carica") tab.runs.openUpload();
    else tab.runs.openGuide();
    return true;
  }
  tab.openPane(t.id === "step" ? "modello" : t.id, { focus: t.id === "step" ? "step" : undefined });
  return true;
}

/**
 * tensor:<nome>[@<step>]: il tensore del run A se ce l'ha, altrimenti del GPT del corso; con lo step,
 * lo snapshot (agganciato a quelli veri). Camera sul tensore, codice, numeri nel pannello Tensore.
 */
async function gotoTensor(tab, name, step) {
  const { state } = tab;
  /** Il nome del tensore (o del buffer tril, che non sta nei pesi ma nella disposizione) in un run, o null. */
  const find = (id) => {
    const run = state.run(id);
    if (!run) return null;
    const t = run.tensor(name); // accetta anche i prefissi module./_orig_mod.
    if (t) return t.name;
    return tab.layoutOf(run).nodes.get(name)?.kind === "buffer" ? name : null;
  };
  let id = state.runA;
  let path = find(id);
  if (!path) {
    id = COURSE_RUN_ID;
    path = find(id);
    if (!path) return false; // nessuno ha quel tensore: il run A resta quello che era
  }
  if (id !== state.runA) await state.setRunA(id, { step });
  else if (step !== state.step) await state.setStep(step);
  // uno step che non è uno snapshot salvato (es. il 750 interpolato della vista "Pesi numerici" del
  // Transformer) si aggancia al più vicino: lo si dice, invece di mostrare in silenzio altri pesi
  if (step !== null && state.step !== step) {
    const where = state.step === null ? "i pesi finali (il run non ha la storia del training)" : `lo snapshot più vicino, step ${INT.format(state.step)}`;
    showToast(`Lo step ${INT.format(step)} non è uno snapshot salvato: mostro ${where}. Gli step intermedi: interpolazione nel pannello Modello.`, { duration: 5200 });
  }
  if (narrow.matches) tab.ui.sidebar.toggleCollapsed(true);
  return tab.inspect(path, { focus: true, code: true, source: "link" });
}

