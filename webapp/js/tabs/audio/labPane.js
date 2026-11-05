/**
 * Pannello "Laboratorio" della sidebar: scegli il brano/prompt, ascolta l'originale, la
 * ricostruzione con N codebook e il brano generato; genera con temperatura e seme.
 * Mostra i numeri veri calcolati dal laboratorio. Non conosce la scena: comunica con
 * index.js tramite le callback.
 */
import { el, fmt } from "../../core/dom.js";
import { CODEC } from "./codec.js";

const pct = (v) => `${(v * 100).toLocaleString("it-IT", { maximumFractionDigits: v < 0.01 ? 2 : 1 })} %`;

/**
 * @param {HTMLElement} pane
 * @param {object} lab
 * @param {object} cb { onStyle(id), playOriginal(), playReconstruction(q), playGenerated(), stop(),
 *                     generate({temperature, seed}), animate() }
 */
export function buildLabPane(pane, lab, cb) {
  const status = el("p", { className: "audio-status", ariaLive: "polite" }, "Premi un pulsante per ascoltare.");
  const styles = Object.values(lab.songs).map(({ style }) => style);
  const styleInputs = styles.map((s, i) =>
    el(
      "label",
      { className: "audio-choice" },
      el("input", { type: "radio", name: "audio-style", value: s.id, checked: i === 0, onchange: () => cb.onStyle(s.id) }),
      el("span", {}, el("strong", {}, `"${s.prompt}"`), el("small", {}, `${s.chords.map((c) => c[0]).join(" · ")} · ${s.bpm} bpm`)),
    ),
  );

  const relErr = lab.relativeError();
  const qInput = el("input", { type: "range", min: 1, max: CODEC.levels, value: CODEC.used, ariaLabel: "Numero di codebook" });
  const qInfo = el("span", { className: "mono" });
  const updateQ = () => {
    const q = Number(qInput.value);
    qInfo.textContent = `${q} codebook · ${fmt(lab.bitrate(q))} bit/s · errore ${pct(relErr[q])}`;
  };
  qInput.addEventListener("input", updateQ);
  updateQ();

  const tempInput = el("input", { type: "range", min: 0.3, max: 1.6, step: 0.1, value: 0.9, ariaLabel: "Temperatura" });
  const tempInfo = el("span", { className: "mono" }, "0,9");
  tempInput.addEventListener("input", () => (tempInfo.textContent = Number(tempInput.value).toLocaleString("it-IT")));
  const seedInput = el("input", { type: "number", min: 1, max: 9999, value: 1, className: "audio-seed", ariaLabel: "Seme casuale" });
  const genButton = el("button", { type: "button", disabled: true, onclick: () => cb.playGenerated() }, "Generato");

  const options = () => ({ temperature: Number(tempInput.value), seed: Number(seedInput.value) || 1 });
  const pcm = lab.sampleRate * 16;

  pane.append(
    el(
      "section",
      { className: "audio-box" },
      el("h3", {}, "1 · Brano e prompt"),
      el("p", { className: "hint" }, "Due brani generati qui con la sintesi additiva. Il prompt sceglie lo stile anche per la generazione."),
      ...styleInputs,
    ),
    el(
      "section",
      { className: "audio-box" },
      el("h3", {}, "2 · Ascolta"),
      el(
        "div",
        { className: "audio-buttons" },
        el("button", { type: "button", onclick: () => cb.playOriginal() }, "Originale"),
        el("button", { type: "button", onclick: () => cb.playReconstruction(Number(qInput.value)) }, "Ricostruito"),
        genButton,
        el("button", { type: "button", className: "audio-stop", onclick: () => cb.stop(), ariaLabel: "Ferma l'audio" }, "Stop"),
      ),
      el("label", { className: "audio-range" }, el("span", {}, "Codebook per la ricostruzione"), qInput, qInfo),
      el("p", { className: "hint" }, "Meno codebook = meno bit, più distorsione. Il nostro decoder (Griffin-Lim) aggiunge un timbro \"metallico\" che un decoder neurale non avrebbe."),
      status,
    ),
    el(
      "section",
      { className: "audio-box" },
      el("h3", {}, "3 · Genera un brano nuovo"),
      el("label", { className: "audio-range" }, el("span", {}, "Temperatura"), tempInput, tempInfo),
      el("label", { className: "audio-range" }, el("span", {}, "Seme"), seedInput),
      el(
        "div",
        { className: "audio-buttons" },
        el("button", { type: "button", onclick: () => cb.animate(options()) }, "Guarda l'animazione"),
        el("button", { type: "button", onclick: () => cb.generate(options()) }, "Genera subito"),
      ),
      el("p", { className: "hint" }, "Simulazione didattica: al posto del transformer c'è un modello a conteggi sui token dei due brani (vedi Percorso, punto 4)."),
    ),
    el(
      "section",
      { className: "audio-box" },
      el("h3", {}, "Numeri veri di questo laboratorio"),
      el(
        "table",
        { className: "audio-table" },
        el(
          "tbody",
          {},
          row("Campioni del brano", fmt(lab.songs[styles[0].id].samples.length)),
          row("Frame del codec (20 ms)", fmt(lab.frames)),
          row("Numeri per frame (feature)", fmt(CODEC.bins)),
          row("Token per secondo", `${fmt(lab.frameRate)} × ${CODEC.used} = ${fmt(lab.frameRate * CODEC.used)}`),
          row("Bit/s con 4 codebook", fmt(lab.bitrate(CODEC.used))),
          row("Bit/s dell'audio a 16 bit", fmt(pcm)),
          row("Compressione", `circa ${fmt(pcm / lab.bitrate(CODEC.used))} volte`),
        ),
      ),
    ),
  );

  return {
    setStatus(text) {
      status.textContent = text;
    },
    enableGenerated(on = true) {
      genButton.disabled = !on;
    },
    selectStyle(id) {
      styleInputs.forEach((l) => {
        const input = l.querySelector("input");
        input.checked = input.value === id;
      });
    },
    options,
  };
}

function row(name, value) {
  return el("tr", {}, el("th", { scope: "row" }, name), el("td", { className: "mono" }, value));
}
