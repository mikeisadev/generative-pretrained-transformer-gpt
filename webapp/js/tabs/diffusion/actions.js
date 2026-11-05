/**
 * Azioni animate sulle immagini (ActionPlayer del nucleo), una per file:
 *   actionForward.js  processo diretto: aggiungere rumore (formula vera di DDPM)
 *   actionGenerate.js generazione: dal rumore all'immagine (DDIM + guidance, denoiser simulato)
 *   actionVae.js      dal latente ai pixel (VAE)
 *   actionInside.js   un passo dentro il denoiser (U-Net o DiT, secondo la vista)
 * Ogni passo ha `enter` che ricostruisce lo stato di inizio passo: "passo indietro" funziona sempre.
 * Qui: pannello dei parametri (seed, guidance scale, passi) e registrazione.
 */
import { el, fmt } from "../../core/dom.js";
import { forwardAction } from "./actionForward.js";
import { generateAction } from "./actionGenerate.js";
import { insideAction } from "./actionInside.js";
import { vaeAction } from "./actionVae.js";

/** Controlli (seed, scala di guidance, numero di passi) e registrazione delle azioni. */
export function installActions(tab, pane) {
  const s = tab.settings;
  const scaleOut = el("output", {}, fmt(s.scale, 1));
  const controls = el(
    "div",
    { className: "diff-controls" },
    el("h3", {}, "Parametri della generazione"),
    el(
      "label",
      {},
      "Guidance scale s ",
      scaleOut,
      el("input", {
        type: "range",
        min: "0",
        max: "15",
        step: "0.5",
        value: String(s.scale),
        oninput: (e) => {
          s.scale = Number(e.target.value);
          scaleOut.textContent = fmt(s.scale, 1);
          tab.plots.setGuidance(s.scale);
        },
      }),
    ),
    el(
      "label",
      {},
      "Passi del campionatore ",
      el("select", { onchange: (e) => (s.steps = Number(e.target.value)) }, ...[10, 20, 50].map((n) => el("option", { value: String(n), selected: n === s.steps }, String(n)))),
    ),
    el("label", {}, "Seed ", el("input", { type: "number", min: "0", value: String(s.seed), onchange: (e) => (s.seed = Math.max(0, Math.floor(Number(e.target.value) || 0))) })),
    el("p", { className: "hint" }, "Valgono dal prossimo avvio di 'Generazione'. Il denoiser è una simulazione didattica: conosce già l'immagine finale."),
  );
  tab.actionsList = el("div");
  pane.append(controls, tab.actionsList);
  for (const make of [forwardAction, generateAction, vaeAction, insideAction]) tab.ui.player.register(make(tab));
}
