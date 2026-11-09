/**
 * Link diretti del Laboratorio pesi (logica pura): che cosa vuol dire `#lab/go=<kind>:<id>`.
 *
 *   tensor:<nome>[@<step>]   un tensore (es. tensor:blocks.0.sa.heads.0.key.weight@500): del run A se
 *                            ce l'ha, altrimenti del GPT del corso. "@finali" (o niente) = pesi finali.
 *                            Il tab Transformer lo usa per "Apri nel Laboratorio pesi".
 *   run:<id>                 un run del registro come run A (run:corso = il GPT del corso)
 *   view:<id>                un pannello o una vista del tab (VIEW_IDS); view:rete e view:cervello sono
 *                            le rappresentazioni del nucleo; view:tensor:<nome>[@<step>] e view:step:<n>
 *                            sono le stesse cose di sopra in forma di "vista", perché la ricerca globale
 *                            accetta solo voci step | action | view
 *   action:<id>              un'azione animata (fase 2: ui.player.register)
 *
 * Gli id arrivano dall'indirizzo: sono testo non fidato. Qui si controlla la forma; se il tensore o il
 * run esistono lo decide il tab.
 */

import { BUILTIN_RUN_ID } from "../../shared/weights/run.js";

/** Le viste del tab raggiungibili con view:<id> (oltre a rete e cervello del nucleo). */
export const VIEW_IDS = Object.freeze({
  modello: "il pannello Modello: i moduli del codice Python e lo step del training",
  tensore: "il pannello Tensore: numeri, statistiche e istogramma del tensore scelto",
  run: "il pannello Run: il GPT del corso e i run caricati",
  carica: "il caricamento di un run (cartella, file, trascinamento)",
  esporta: "come esportare un run dal Python",
  step: "il cursore dello step del training",
  centro: "tutto il modello inquadrato (come C)",
});

/** Alias leggibili degli id dei run nei link (run:corso). */
export const RUN_ALIASES = Object.freeze({ corso: BUILTIN_RUN_ID, "gpt-corso": BUILTIN_RUN_ID, builtin: BUILTIN_RUN_ID });

const MAX_ID = 600;
// eslint-disable-next-line no-control-regex
const CONTROL = /[\u0000-\u001f\u007f]/;
const FINAL = new Set(["finali", "final", "finale"]);

/**
 * `<nome>[@<step>]` → { name, step } (step null = pesi finali), oppure null se la forma non va.
 * Il nome è quello di PyTorch (anche con i prefissi module./_orig_mod.: li toglie chi cerca il tensore).
 */
export function parseTensorTarget(id) {
  if (typeof id !== "string" || !id || id.length > MAX_ID || CONTROL.test(id)) return null;
  const at = id.lastIndexOf("@");
  const name = (at < 0 ? id : id.slice(0, at)).trim();
  if (!name || /\s/.test(name)) return null;
  if (at < 0) return { name, step: null };
  const tail = id.slice(at + 1).trim().toLowerCase();
  if (FINAL.has(tail)) return { name, step: null };
  if (!/^\d{1,9}$/.test(tail)) return null;
  return { name, step: Number(tail) };
}

/** Il contrario di parseTensorTarget: `formatTensorTarget("lm_head.weight", 500)` → "lm_head.weight@500". */
export function formatTensorTarget(name, step = null) {
  return step === null || step === undefined ? String(name) : `${name}@${step}`;
}

/** `run:<id>` → id del registro (con gli alias), oppure null. */
export function parseRunTarget(id) {
  if (typeof id !== "string" || !id || id.length > MAX_ID || CONTROL.test(id)) return null;
  const clean = id.trim();
  return RUN_ALIASES[clean.toLowerCase()] ?? clean;
}

/**
 * Un link diretto del tab, normalizzato:
 *   { type: "tensor", name, step } | { type: "run", id } | { type: "view", id } | { type: "step", step }
 *   | { type: "action", id } | { type: "rep", id } (view:rete | view:cervello, del nucleo) | null
 * `step` null = pesi finali. null se kind o id non si capiscono (il tab risponde false: "non trovato").
 */
export function resolveTarget({ kind, id } = {}) {
  if (typeof id !== "string" || !id) return null;
  if (kind === "tensor") {
    const t = parseTensorTarget(id);
    return t ? { type: "tensor", ...t } : null;
  }
  if (kind === "run") {
    const runId = parseRunTarget(id);
    return runId ? { type: "run", id: runId } : null;
  }
  if (kind === "action") return id.length <= MAX_ID && !CONTROL.test(id) ? { type: "action", id } : null;
  if (kind !== "view") return null;
  if (id === "rete" || id === "cervello") return { type: "rep", id };
  if (id.startsWith("tensor:")) return resolveTarget({ kind: "tensor", id: id.slice(7) });
  if (id.startsWith("run:")) return resolveTarget({ kind: "run", id: id.slice(4) });
  const step = /^step:(\d{1,9}|finali)$/.exec(id);
  if (step) return { type: "step", step: step[1] === "finali" ? null : Number(step[1]) };
  return Object.hasOwn(VIEW_IDS, id) ? { type: "view", id } : null;
}
