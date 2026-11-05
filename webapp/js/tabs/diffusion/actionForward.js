/**
 * Azione "Processo diretto: aggiungere rumore": la formula di DDPM applicata a un'immagine vera
 * (generata nel browser), per vari t, con le miniature e l'istogramma dei pixel; poi come lo
 * stesso calcolo diventa un passo di training (categoria "Training" della panoramica).
 */
import { fmt } from "../../core/dom.js";
import { focusLab } from "./lab.js";
import { fillGaussian, mulberry32, qSample, SCHEDULES } from "./math.js";
import { RGBTexture } from "./textures.js";

const f2 = (x) => fmt(x, 2);

/** Formula in alto nel banco: il processo diretto, oppure la loss nel passo di training. */
const FORMULA = {
  forward: ["x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε", "DDPM: β lineare da 0,0001 a 0,02 · T = 1000"],
  training: ["loss = media di (ε − ε̂)²,   ε̂ = rete(x_t, t)", "la rete vede x_t e t; la risposta giusta è ε (il pannello centrale)"],
};
/** Passo "estratto a caso" per l'esempio di training (fisso, così l'animazione è ripetibile). */
const TRAIN_T = 640;

export function forwardAction(tab) {
  const { lab, reference } = tab;
  const schedule = SCHEDULES.ddpm;
  const TS = [50, 100, 200, 400, 600, 800, 999];
  const N = reference.length;
  let xt, eps, texX0, texEps, texXt, thumbs, last;

  const show = (t) => {
    if (t === last) return;
    last = t;
    const ab = t < 0 ? 1 : schedule.alphaBar[t];
    qSample(xt, reference, eps, ab);
    texXt.set(xt);
    lab.get("xt").title.set(t < 0 ? "x_0 (nessun rumore)" : `x_t con t = ${t}`);
    lab.get("x0").title.set(`x_0  ·  √ᾱ = ${f2(Math.sqrt(ab))}`);
    lab.get("eps").title.set(`ε  ·  √(1 − ᾱ) = ${f2(Math.sqrt(1 - ab))}`);
    tab.plots.highlightT(Math.max(0, t));
  };

  return {
    id: "forward",
    title: "Processo diretto e training",
    category: "Training",
    summary: "La formula vera di DDPM applicata a un'immagine (x_t = √ᾱ·x_0 + √(1−ᾱ)·ε, per vari t) e come diventa un passo di training.",
    onStart(ctx) {
      lab.clear();
      xt = new Float32Array(N);
      eps = fillGaussian(new Float32Array(N), mulberry32(tab.settings.seed));
      texX0 = lab.own(new RGBTexture(256).set(reference));
      texEps = lab.own(new RGBTexture(256).set(eps.map((v) => v / 3)));
      texXt = lab.own(new RGBTexture(256));
      lab.text("formula", FORMULA.forward[0], -4, 15.6, { height: 1.6, color: "#fde68a" });
      lab.text("formula-sub", FORMULA.forward[1], -4, 13.4, { height: 0.9, color: "#94a3b8" });
      lab.panel("x0", { texture: texX0.texture, x: -22, y: 2.5, w: 9, title: "x_0", titleHeight: 0.9 });
      lab.text("plus", "+", -15, 2.5, { height: 2, color: "#e2e8f0" });
      lab.panel("eps", { texture: texEps.texture, x: -8, y: 2.5, w: 9, title: "ε", titleHeight: 0.9 });
      lab.text("eq", "=", -1.2, 2.5, { height: 2, color: "#e2e8f0" });
      lab.panel("xt", { texture: texXt.texture, x: 9.5, y: 2.5, w: 13, title: "x_t", titleHeight: 1.1 });
      thumbs = TS.map((t, i) => {
        const tex = lab.own(new RGBTexture(256).set(qSample(new Float32Array(N), reference, eps, schedule.alphaBar[t])));
        return lab.panel(`thumb${i}`, { texture: tex.texture, x: -22 + i * 5.3, y: -11, w: 4.6, title: `t = ${t}`, titleHeight: 0.75 });
      });
      lab.text("thumbs-title", "Lo stesso ε, sempre più forte:", -26.5, -5.6, { height: 0.85, color: "#94a3b8", align: "left" });
      lab.show();
      last = undefined;
      show(-1);
      tab.plots.histogram(reference);
      tab.plots.reveal("schedule");
      ctx.data.view = "lab";
      focusLab(tab);
    },
    onStop() {
      lab.clear();
      tab.hideCode();
    },
    build: () => {
      const setFormula = ([main, sub]) => {
        lab.get("formula").label.set(main);
        lab.get("formula-sub").label.set(sub);
      };
      const showQSample = () => tab.showCode("ddpm_qsample", "DDPM: q_sample (processo diretto)", ["sqrt_alphas_cumprod", "noise"]);
      const showThumbs = (k) => thumbs.forEach((th, i) => (th.mesh.visible = th.title.sprite.visible = i < k));
      const steps = [
        {
          title: "L'immagine di partenza",
          caption: "Un faro al tramonto, disegnato pixel per pixel nel browser. Ogni pixel ha 3 numeri (rosso, verde, blu) scalati tra −1 e 1, come in DDPM.",
          duration: 3.5,
          enter() {
            show(-1);
            showThumbs(0);
            tab.plots.histogram(reference);
            showQSample();
          },
        },
        {
          title: "Il rumore gaussiano ε",
          caption: `ε è rumore gaussiano: un numero casuale per ogni valore dell'immagine, preso da una "campana" con media 0 e varianza 1. Il seed ${tab.settings.seed} lo rende ripetibile.`,
          duration: 3.5,
          enter() {
            show(-1);
            showThumbs(0);
            tab.plots.histogram(eps);
            showQSample();
          },
        },
        ...TS.map((t, i) => {
          const from = i === 0 ? 0 : TS[i - 1];
          const ab = schedule.alphaBar[t];
          const captions = {
            50: `t = 50: resta il ${fmt(Math.sqrt(ab) * 100)}% del segnale. Non serve applicare il rumore 50 volte: la formula salta direttamente al passo t.`,
            200: `t = 200: √ᾱ = ${f2(Math.sqrt(ab))}, √(1 − ᾱ) = ${f2(Math.sqrt(1 - ab))}. Le forme grandi si vedono ancora, i dettagli fini sono già spariti.`,
            600: `t = 600: il rumore pesa ${f2(Math.sqrt(1 - ab))}, il segnale solo ${f2(Math.sqrt(ab))}. Si intuisce appena dov'era il sole.`,
            999: `t = 999 (l'ultimo): √ᾱ = ${fmt(Math.sqrt(ab), 3)}. È quasi solo rumore: da qui parte la generazione.`,
          };
          return {
            title: `t = ${t}`,
            caption: captions[t] ?? `t = ${t}: √ᾱ = ${f2(Math.sqrt(ab))} (peso dell'immagine), √(1 − ᾱ) = ${f2(Math.sqrt(1 - ab))} (peso del rumore).`,
            duration: 2.4,
            enter() {
              showThumbs(i);
              show(from);
              showQSample();
            },
            update(p, dt, ctx) {
              // t scorre con continuità; l'istogramma si ricalcola solo ogni tanto (è il pezzo costoso)
              const now = Math.round(from + (t - from) * Math.min(1, p * 1.4));
              show(now);
              ctx.data.histT = (ctx.data.histT ?? 0) + dt;
              if (ctx.data.histT > 0.3 || p >= 1) {
                ctx.data.histT = 0;
                tab.plots.histogram(xt);
              }
              if (p >= 1) showThumbs(i + 1);
            },
          };
        }),
        {
          title: "Un passo di training",
          caption: `Training: si sceglie un t a caso (qui t = ${TRAIN_T}) e un rumore ε, si crea x_t con la formula e la rete, guardando solo x_t e t, deve indovinare ε. La loss è la media dei quadrati di (ε − ε̂); poi la discesa del gradiente corregge i pesi, come nel nostro GPT.`,
          duration: 7,
          formula: FORMULA.training,
          enter() {
            showThumbs(TS.length);
            show(TRAIN_T);
            tab.showCode("ddpm_config", "DDPM: iperparametri del training (CIFAR-10)", ["num_diffusion_timesteps", "model_mean_type"]);
          },
        },
        {
          title: "E in Stable Diffusion?",
          caption: "Stessa idea, ma il rumore si aggiunge al latente 4×64×64 prodotto dal VAE, non ai pixel; e la schedule è 'scaled linear' (β da 0,00085 a 0,012).",
          duration: 5,
          enter() {
            showThumbs(TS.length);
            show(TS.at(-1));
            tab.showCode("sd_config", "Stable Diffusion 1.x: v1-inference.yaml", ["linear_start", "linear_end", "timesteps", "channels: 4"]);
          },
        },
      ];
      // ogni passo rimette la formula giusta in alto (così "passo indietro" funziona sempre)
      return steps.map((step) => ({
        ...step,
        enter(ctx) {
          setFormula(step.formula ?? FORMULA.forward);
          step.enter(ctx);
        },
      }));
    },
  };
}
