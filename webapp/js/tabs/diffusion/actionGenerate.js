/**
 * Azione "Generazione: dal rumore all'immagine": ciclo DDIM con classifier-free guidance.
 * Il denoiser è SIMULATO (denoiserSim.js) e lo si dichiara nella scena e nelle didascalie.
 * La traiettoria si precalcola all'avvio: così si può andare avanti e indietro tra i passi.
 */
import { fmt } from "../../core/dom.js";
import { ensureLab, latentGrid } from "./lab.js";
import { fillGaussian, meanStd, mulberry32, samplingTimesteps, SCHEDULES } from "./math.js";
import { ChannelTexture, RGBTexture } from "./textures.js";

const f2 = (x) => fmt(x, 2);
const SIM_NOTE = "Simulazione didattica: questo denoiser conosce già l'immagine finale.\nQuello vero è una rete addestrata (U-Net o DiT) che la deve indovinare.";

export function generateAction(tab) {
  const { lab, sim, vae, pipeline } = tab;
  const schedule = SCHEDULES.sd;
  let traj; // stati precalcolati: così si può andare avanti e indietro tra i passi
  let gridZ, gridEps;
  let texEu, texEc, texPreview;
  let shownZ, preview, lastBlend, lastPreview;

  /** Precalcola tutta la traiettoria con i parametri correnti (seed, passi, scala). */
  function computeTrajectory({ seed, steps, scale }) {
    const ts = samplingTimesteps(schedule.T, steps);
    const n = vae.z0.length;
    const states = [fillGaussian(new Float32Array(n), mulberry32(seed))];
    const preds = [];
    for (let k = 0; k < ts.length; k++) {
      const ab = schedule.alphaBar[ts[k]];
      const abPrev = k + 1 < ts.length ? schedule.alphaBar[ts[k + 1]] : 1; // l'ultimo passo arriva a z_0
      const next = new Float32Array(n);
      sim.step(next, states[k], ab, abPrev, scale);
      preds.push({ epsU: sim.epsU.slice(), epsC: sim.epsC.slice(), eps: sim.eps.slice(), x0hat: sim.x0hat.slice() });
      states.push(next);
    }
    return { ts, states, preds, alphaBars: ts.map((t) => schedule.alphaBar[t]), scale };
  }

  /** Mostra lo stato a metà tra il passo k e il successivo (blend 0…1). */
  function display(k, blend) {
    const key = k * 1000 + Math.round(blend * 40);
    if (key === lastBlend) return;
    lastBlend = key;
    const a = traj.states[k];
    const b = traj.states[Math.min(k + 1, traj.states.length - 1)];
    for (let i = 0; i < shownZ.length; i++) shownZ[i] = a[i] + (b[i] - a[i]) * blend;
    gridZ.set(shownZ);
    pipeline.latentTex.forEach((t, c) => t.set(shownZ, c));
    // anteprima: decodifica la stima di z_0 di questo passo (cosa "vede" il modello in fondo al rumore).
    // Dipende solo dal passo, non dalla sfumatura: si ricalcola una volta per passo.
    const kp = Math.min(k, traj.preds.length - 1);
    if (kp === lastPreview) return;
    lastPreview = kp;
    vae.decode(preview, traj.preds[kp].x0hat);
    texPreview.set(preview);
  }

  function invalidateDisplay() {
    lastBlend = lastPreview = undefined;
  }

  function showPredictions(k) {
    const pred = traj.preds[k];
    texEu.set(pred.epsU, 0);
    texEc.set(pred.epsC, 0);
    gridEps.set(pred.eps);
    pipeline.epsTex.forEach((t, c) => t.set(pred.eps, c));
  }

  return {
    id: "generate",
    title: "Generazione: dal rumore all'immagine",
    category: "Generazione (prompt)",
    summary: "Il ciclo di denoising con DDIM e classifier-free guidance (due previsioni e la scala s). Denoiser simulato.",
    onStart() {
      lab.clear();
      traj = computeTrajectory(tab.settings);
      shownZ = new Float32Array(vae.z0.length);
      preview = new Float32Array(tab.reference.length);
      texEu = lab.own(new ChannelTexture(32));
      texEc = lab.own(new ChannelTexture(32));
      texPreview = lab.own(new RGBTexture(256));
      gridZ = latentGrid(lab, "z", -23, 1, 9, "z_t · 4 canali", 1);
      lab.panel("eu", { texture: texEu.texture, x: -10.5, y: 6, w: 5.5, title: "ε senza prompt", titleHeight: 0.95 });
      lab.panel("ec", { texture: texEc.texture, x: -10.5, y: -4.4, w: 5.5, title: "ε con il prompt", titleHeight: 0.95 });
      gridEps = latentGrid(lab, "e", 2.5, 1, 9, `ε̂ = ε_u + s·(ε_c − ε_u)\ns = ${fmt(traj.scale, 1)}`, 1);
      lab.panel("preview", { texture: texPreview.texture, x: 18, y: 1, w: 12, title: "anteprima di ẑ_0", titleHeight: 1 });
      lab.text("status", "", -2.5, 15.5, { height: 1.3, color: "#fde68a" });
      lab.text("note", SIM_NOTE, -2.5, -11.8, { height: 0.85, color: "#fca5a5" });
      lab.text("canale", "(canale 0)", -10.5, -7.9, { height: 0.6, color: "#94a3b8" });
      lab.show();
      invalidateDisplay();
      tab.plots.resetGeneration();
      tab.plots.setGuidance(traj.scale);
      tab.plots.reveal("generazione");
    },
    onStop() {
      lab.clear();
      tab.resetPipeline();
      tab.hideCode();
    },
    build: () => {
      const { ts, alphaBars } = traj;
      const status = () => lab.get("status").label;
      const steps = [
        {
          title: "Il prompt diventa vettori",
          caption: `"un faro su una scogliera al tramonto": il tokenizer lo spezza in token, il text encoder CLIP (un transformer) produce 77 vettori da 768 numeri.`,
          duration: 4,
          enter(ctx) {
            ctx.data.view = "text";
            tab.focus(["prompt", "tokens", "textEncoder", "textEmb"], { margin: 1.08, showFrame: false });
            tab.showCode("sd_encode", "diffusers: encode_prompt (tokenizer + text encoder)", ["self.tokenizer(", "max_length", "self.text_encoder("]);
          },
        },
        {
          title: "Rumore iniziale",
          caption: `Si parte da puro rumore gaussiano nel latente 4×64×64, scelto dal seed ${tab.settings.seed}: stesso seed e stesso prompt danno la stessa immagine.`,
          duration: 3.5,
          enter(ctx) {
            ensureLab(tab, ctx);
            display(0, 0);
            texEu.set(new Float32Array(vae.z0.length), 0);
            texEc.set(new Float32Array(vae.z0.length), 0);
            gridEps.set(new Float32Array(vae.z0.length));
            texPreview.set(new Float32Array(tab.reference.length));
            invalidateDisplay(); // l'anteprima è stata cambiata a mano: il prossimo display() la ricalcola
            status().set(`seed ${tab.settings.seed} · ${ts.length} passi · t parte da ${ts[0]}`);
            pipeline.setTimestep(ts[0]);
            tab.showCode("sd_loop", "diffusers: il ciclo di denoising di Stable Diffusion", ["for i, t in enumerate(timesteps)"]);
            tab.plots.showGeneration(alphaBars, 0);
          },
        },
        ...ts.map((t, k) => ({
          title: `Passo ${k + 1}/${ts.length}`,
          caption:
            k === 0
              ? `Passo 1 (t = ${t}): il denoiser predice il rumore DUE volte, senza e con il prompt. La guidance le combina: ε̂ = ε_u + ${fmt(traj.scale, 1)}·(ε_c − ε_u). Lo scheduler DDIM toglie la parte prevista.`
              : k === 1
                ? `Passo 2 (t = ${t}): l'anteprima mostra la stima di ẑ_0, cioè cosa il modello "vede" sotto il rumore. All'inizio è vaga, poi si precisa.`
                : `Passo ${k + 1}/${ts.length} · t = ${t} · peso del rumore √(1 − ᾱ) = ${f2(Math.sqrt(1 - alphaBars[k]))}`,
          duration: k < 2 ? 4.5 : ts.length > 30 ? 0.5 : 1.1,
          enter(ctx) {
            ensureLab(tab, ctx);
            showPredictions(k);
            display(k, 0);
            pipeline.setTimestep(t);
            status().set(`passo ${k + 1}/${ts.length} · t = ${t}`);
            tab.plots.showGeneration(alphaBars, k);
            if (k === 1) tab.showCode("ddim_step", "diffusers: passo DDIM (eta = 0)", ["pred_original_sample = (sample", "prev_sample ="]);
            else tab.showCode("sd_loop", "diffusers: guidance e passo dello scheduler", ["noise_pred_uncond, noise_pred_text", "guidance_scale", "scheduler.step"]);
          },
          update(p) {
            // prima metà: arrivano le previsioni; seconda metà: il latente diventa z_{t−1}
            display(k, p < 0.4 ? 0 : (p - 0.4) / 0.6);
          },
        })),
        {
          title: "Dal latente ai pixel",
          caption: "Dopo l'ultimo passo il latente z_0 (4×64×64) passa nel decoder del VAE e diventa un'immagine 512×512 a colori.",
          duration: 4,
          enter(ctx) {
            ctx.data.view = "image";
            showPredictions(ts.length - 1);
            display(ts.length, 0);
            vae.decode(preview, traj.states.at(-1));
            texPreview.set(preview);
            invalidateDisplay();
            pipeline.imageTex.set(preview);
            lab.get("preview").title.set("immagine finale");
            status().set("fine: VAE decoder");
            tab.focus(["vae", "image"], { margin: 1.3, showFrame: false });
            tab.showCode("sd_decode", "diffusers: decodifica con il VAE", ["vae.decode", "scaling_factor"]);
          },
        },
        {
          title: "Risultato",
          caption: `Fatto: ${ts.length} passi, ${ts.length * 2} previsioni del denoiser (due per passo per la guidance; diffusers le calcola in un unico batch). Prova a cambiare la scala s o il seed nel pannello Azioni e riavvia.`,
          duration: 5,
          enter(ctx) {
            showPredictions(ts.length - 1);
            display(ts.length, 0);
            vae.decode(preview, traj.states.at(-1));
            texPreview.set(preview);
            invalidateDisplay();
            pipeline.imageTex.set(preview);
            lab.get("preview").title.set("immagine finale");
            tab.showCode("sd_decode", "diffusers: decodifica con il VAE", ["vae.decode"]);
            const { std } = meanStd(preview);
            status().set(`s = ${fmt(traj.scale, 1)} · contrasto ${f2(std)}`);
            ensureLab(tab, ctx);
          },
        },
      ];
      return steps;
    },
    onEnd() {
      pipeline.imageTex.set(preview);
    },
  };
}
