/**
 * Grafici del tab (ChartsPanel del nucleo). Ogni grafico ha titolo, descrizione e legenda.
 *
 * - Rumore nel tempo: β_t, ᾱ_t e SNR(t) (scala log) per le schedule di DDPM e Stable Diffusion
 * - Generazione: quanto rumore resta a ogni passo, effetto della guidance scale
 * - Processo diretto: distribuzione dei valori dei pixel di x_t
 * - Video: differenza tra frame consecutivi (uno alla volta contro tutti insieme)
 */
import { meanAbsDiff, meanStd, SCHEDULES, snr } from "./math.js";
import { REFERENCE_SCALE } from "./denoiserSim.js";

const EVERY = 10; // un punto ogni 10 passi: 100 punti per curva bastano

function curve(schedule, f) {
  const pts = [];
  for (let t = 0; t < schedule.T; t += EVERY) pts.push([t, f(schedule, t)]);
  pts.push([schedule.T - 1, f(schedule, schedule.T - 1)]);
  return pts;
}

export function installPlots(tab) {
  const charts = tab.ui.charts;
  const { ddpm, sd } = SCHEDULES;

  // --- rumore nel tempo
  const noise = charts.section("schedule", "Il rumore nel tempo", "Come cresce il rumore lungo i T = 1000 passi del processo diretto.");
  const beta = noise.chart("beta", {
    type: "line",
    title: "Schedule di β_t",
    description: "Quanto rumore si aggiunge in UN passo. DDPM: retta da 0,0001 a 0,02. Stable Diffusion: 'scaled linear' da 0,00085 a 0,012 (si interpolano le radici).",
    xLabel: "passo t",
    yLabel: "β_t",
  });
  beta.setData({ series: [{ name: "DDPM (lineare)", points: curve(ddpm, (s, t) => s.betas[t]) }, { name: "Stable Diffusion (scaled linear)", points: curve(sd, (s, t) => s.betas[t]) }] });
  const alphaBar = noise.chart("alphabar", {
    type: "line",
    title: "ᾱ_t: quanta immagine resta",
    description: "Prodotto di tutti gli (1 − β): √ᾱ_t è il peso dell'immagine originale dentro x_t, √(1 − ᾱ_t) quello del rumore. Scende da 1 a quasi 0.",
    xLabel: "passo t",
    yLabel: "ᾱ_t",
  });
  alphaBar.setData({ series: [{ name: "DDPM", points: curve(ddpm, (s, t) => s.alphaBar[t]) }, { name: "Stable Diffusion", points: curve(sd, (s, t) => s.alphaBar[t]) }] });
  const snrChart = noise.chart("snr", {
    type: "line",
    title: "SNR(t) = ᾱ_t / (1 − ᾱ_t), scala log",
    description: "Rapporto segnale/rumore. Serve a confrontare schedule diverse: la scala logaritmica mostra che il segnale cala di molti ordini di grandezza.",
    xLabel: "passo t",
    yLabel: "SNR",
    yScale: "log",
  });
  snrChart.setData({ series: [{ name: "DDPM", points: curve(ddpm, (s, t) => snr(s.alphaBar[t])) }, { name: "Stable Diffusion", points: curve(sd, (s, t) => snr(s.alphaBar[t])) }] });
  const hist = noise.chart("hist", {
    type: "histogram",
    title: "Valori dei pixel di x_t",
    description: "Durante l'azione 'Processo diretto': la distribuzione dei pixel passa da quella dell'immagine a una gaussiana (la campana del rumore).",
    xLabel: "valore del pixel",
    yLabel: "quanti pixel",
  });

  // --- generazione
  const gen = charts.section("generazione", "Generazione", "Si aggiornano durante l'azione 'Generazione: dal rumore all'immagine'.");
  const perStep = gen.chart("perstep", {
    type: "line",
    title: "Quanto rumore c'è a ogni passo",
    description: "Peso del rumore √(1 − ᾱ_t) e peso del segnale √ᾱ_t nel latente, passo dopo passo del campionatore (schedule di Stable Diffusion).",
    xLabel: "passo di generazione",
    yLabel: "peso",
  });
  const guidance = gen.chart("guidance", {
    type: "line",
    title: "Effetto della guidance scale (simulazione)",
    description: `Contrasto dell'immagine finale e distanza dall'immagine di riferimento al variare di s. Simulazione didattica tarata su s = ${REFERENCE_SCALE}: s basso dà un'immagine slavata, s alto la satura.`,
    xLabel: "guidance scale s",
    yLabel: "valore",
  });
  {
    const sim = tab.sim;
    const z = new Float32Array(sim.vae.z0.length);
    const px = new Float32Array(tab.reference.length);
    const contrast = [];
    const distance = [];
    for (let s = 0; s <= 15.001; s += 0.5) {
      sim.vae.decode(px, sim.finalLatent(z, s));
      contrast.push([s, meanStd(px).std]);
      distance.push([s, meanAbsDiff(px, tab.reference)]);
    }
    guidance.setData({ series: [{ name: "contrasto (dev. std. dei pixel)", points: contrast }, { name: "distanza dal riferimento", points: distance }] });
  }

  // --- video
  const vid = charts.section("video", "Video", "Coerenza tra frame: perché i modelli video generano i frame insieme.");
  const flicker = vid.chart("flicker", {
    type: "line",
    title: "Differenza tra frame consecutivi (simulazione)",
    description: "Media della differenza assoluta tra un frame e il successivo. Se ogni frame nasce per conto suo, la scena 'salta'; se nascono insieme, cambia solo ciò che si muove davvero.",
    xLabel: "coppia di frame",
    yLabel: "differenza media",
  });

  tab.plots = {
    /** Linea verticale "siamo qui" sui grafici della schedule. */
    highlightT(t) {
      beta.setHighlight({ x: t });
      alphaBar.setHighlight({ x: t });
      snrChart.setHighlight({ x: t });
    },
    histogram(values) {
      hist.setData({ values, bins: 40 });
    },
    resetGeneration() {
      perStep.setData({ series: [{ name: "rumore √(1 − ᾱ)", points: [] }, { name: "segnale √ᾱ", points: [] }] });
    },
    /** alphaBars[k] = ᾱ al passo k: disegna i passi 0…upTo (anche tornando indietro con il player). */
    showGeneration(alphaBars, upTo) {
      const shown = alphaBars.slice(0, upTo + 1);
      perStep.setData({
        series: [
          { name: "rumore √(1 − ᾱ)", points: shown.map((ab, k) => [k, Math.sqrt(1 - ab)]) },
          { name: "segnale √ᾱ", points: shown.map((ab, k) => [k, Math.sqrt(ab)]) },
        ],
      });
      perStep.setHighlight({ x: upTo });
    },
    setGuidance(scale) {
      guidance.setHighlight({ x: scale });
    },
    showFlicker() {
      const a = tab.video.indep.map((f) => f.data);
      const b = tab.video.joint.map((f) => f.data);
      const pairs = (frames) => frames.slice(1).map((f, i) => [i + 1, meanAbsDiff(frames[i], f)]);
      flicker.setData({ series: [{ name: "un frame alla volta", points: pairs(a) }, { name: "tutti insieme", points: pairs(b) }] });
    },
    open: () => charts.open(),
  };
  tab.plots.resetGeneration();
  tab.plots.showFlicker();
}
