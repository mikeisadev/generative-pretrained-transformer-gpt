/**
 * Azioni animate sulle immagini (ActionPlayer del nucleo):
 *   1. Processo diretto: aggiungere rumore (formula vera di DDPM su un'immagine generata nel browser)
 *   2. Generazione: dal rumore all'immagine (DDIM + classifier-free guidance, denoiser simulato)
 *   3. Dal latente ai pixel (VAE)
 *   4. Un passo dentro il denoiser (U-Net o DiT, secondo la vista)
 * Ogni passo ha `enter` che ricostruisce lo stato di inizio passo: "passo indietro" funziona sempre.
 */
import { el, fmt } from "../../core/dom.js";
import { fillGaussian, meanStd, mulberry32, qSample, samplingTimesteps, SCHEDULES } from "./math.js";
import { recolor } from "./shapes.js";
import { ChannelTexture, RGBTexture } from "./textures.js";

const f2 = (x) => fmt(x, 2);
const SIM_NOTE = "Simulazione didattica: questo denoiser conosce già l'immagine finale.\nQuello vero è una rete addestrata (U-Net o DiT) che la deve indovinare.";

/** Quattro canali di un latente in una griglia 2 × 2 nel banco di lavoro. */
function latentGrid(lab, key, x, y, w, title) {
  const textures = Array.from({ length: 4 }, () => lab.own(new ChannelTexture(32)));
  const half = w / 2 + 0.15;
  textures.forEach((t, c) => {
    lab.panel(`${key}${c}`, { texture: t.texture, x: x + (c % 2 ? half / 2 + 0.1 : -half / 2 - 0.1), y: y + (c < 2 ? half / 2 + 0.1 : -half / 2 - 0.1), w: w / 2, title: "" });
  });
  lab.text(`${key}-title`, title, x, y + w / 2 + 1.2, { height: 0.5 });
  return { set: (z) => textures.forEach((t, c) => t.set(z, c)) };
}

function focusLab(tab, margin = 1.08) {
  return tab.ui.director.focus(tab.lab.bounds(), { showFrame: false, margin });
}

// ---------------------------------------------------------------------------------------------
// 1. Processo diretto
// ---------------------------------------------------------------------------------------------
function forwardAction(tab) {
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
    title: "Processo diretto: aggiungere rumore",
    category: "Immagini",
    summary: "La formula vera di DDPM applicata a un'immagine: x_t = √ᾱ·x_0 + √(1−ᾱ)·ε, per vari t.",
    onStart() {
      lab.clear();
      xt = new Float32Array(N);
      eps = fillGaussian(new Float32Array(N), mulberry32(tab.settings.seed));
      texX0 = lab.own(new RGBTexture(256).set(reference));
      texEps = lab.own(new RGBTexture(256).set(eps.map((v) => v / 3)));
      texXt = lab.own(new RGBTexture(256));
      lab.text("formula", "x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε", -2, 13, { height: 0.9, color: "#fde68a" });
      lab.text("formula-sub", "DDPM: schedule lineare di β da 0,0001 a 0,02, T = 1000 passi", -2, 11.6, { height: 0.45, color: "#94a3b8" });
      lab.panel("x0", { texture: texX0.texture, x: -21, y: 1, w: 8, title: "x_0" });
      lab.text("plus", "+", -14.5, 1, { height: 1.4, color: "#e2e8f0" });
      lab.panel("eps", { texture: texEps.texture, x: -8, y: 1, w: 8, title: "ε" });
      lab.text("eq", "=", -1.6, 1, { height: 1.4, color: "#e2e8f0" });
      lab.panel("xt", { texture: texXt.texture, x: 8, y: 1, w: 12, title: "x_t", titleHeight: 0.6 });
      thumbs = TS.map((t, i) => {
        const tex = lab.own(new RGBTexture(256).set(qSample(new Float32Array(N), reference, eps, schedule.alphaBar[t])));
        return lab.panel(`thumb${i}`, { texture: tex.texture, x: -21 + i * 4.9, y: -8.6, w: 4, title: `t = ${t}`, titleHeight: 0.38 });
      });
      lab.text("thumbs-title", "Lo stesso ε, sempre più forte:", -24.5, -5.6, { height: 0.42, color: "#94a3b8", align: "left" });
      lab.show();
      last = undefined;
      show(-1);
      tab.plots.histogram(reference);
      focusLab(tab);
    },
    onStop() {
      lab.clear();
    },
    build: () => {
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
            tab.showCode("ddpm_qsample", "DDPM: q_sample (processo diretto)", ["sqrt_alphas_cumprod", "noise"]);
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
          title: "Come si addestra la rete",
          caption: "Training: si sceglie un t a caso, si crea x_t con questa formula e si chiede alla rete di indovinare ε. La loss è l'errore quadratico tra ε vero e ε previsto.",
          duration: 5,
          enter() {
            showThumbs(TS.length);
            show(TS.at(-1));
            tab.showCode("ddpm_config", "DDPM: iperparametri del training (CIFAR-10)", ["num_diffusion_timesteps", "model_mean_type"]);
          },
        },
        {
          title: "E in Stable Diffusion?",
          caption: "Stessa idea, ma il rumore si aggiunge al latente 4×64×64 prodotto dal VAE, non ai pixel; e la schedule è 'scaled linear' (β da 0,00085 a 0,012).",
          duration: 5,
          enter() {
            showThumbs(TS.length);
            tab.showCode("sd_config", "Stable Diffusion 1.x: v1-inference.yaml", ["linear_start", "linear_end", "timesteps", "channels: 4"]);
          },
        },
      ];
      return steps;
    },
  };
}

// ---------------------------------------------------------------------------------------------
// 2. Generazione
// ---------------------------------------------------------------------------------------------
function generateAction(tab) {
  const { lab, sim, vae, pipeline } = tab;
  const schedule = SCHEDULES.sd;
  let traj; // stati precalcolati: così si può andare avanti e indietro tra i passi
  let gridZ, gridEps;
  let texEu, texEc, texPreview;
  let shownZ, preview, lastBlend;

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
    const pred = traj.preds[Math.min(k, traj.preds.length - 1)];
    // anteprima: decodifica la stima di z_0 di questo passo (cosa "vede" il modello in fondo al rumore)
    vae.decode(preview, pred.x0hat);
    texPreview.set(preview);
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
    category: "Immagini",
    summary: "Il ciclo di denoising con DDIM e classifier-free guidance (due previsioni e la scala s). Denoiser simulato.",
    onStart() {
      lab.clear();
      traj = computeTrajectory(tab.settings);
      shownZ = new Float32Array(vae.z0.length);
      preview = new Float32Array(tab.reference.length);
      texEu = lab.own(new ChannelTexture(32));
      texEc = lab.own(new ChannelTexture(32));
      texPreview = lab.own(new RGBTexture(256));
      gridZ = latentGrid(lab, "z", -21, 1, 8, "z_t (latente, 4 canali)");
      lab.panel("eu", { texture: texEu.texture, x: -9, y: 4.6, w: 5, title: "ε senza prompt", titleHeight: 0.42 });
      lab.panel("ec", { texture: texEc.texture, x: -9, y: -3.4, w: 5, title: "ε con il prompt", titleHeight: 0.42 });
      gridEps = latentGrid(lab, "e", 2.5, 1, 7, `ε̂ = ε_u + s·(ε_c − ε_u),  s = ${fmt(traj.scale, 1)}`);
      lab.panel("preview", { texture: texPreview.texture, x: 16, y: 1, w: 10, title: "anteprima: ẑ_0 decodificato", titleHeight: 0.5 });
      lab.text("status", "", -2, 13, { height: 0.8, color: "#fde68a" });
      lab.text("note", SIM_NOTE, -2, -7.6, { height: 0.42, color: "#fca5a5" });
      lab.text("canale", "(canale 0)", -9, -6.8, { height: 0.32, color: "#94a3b8" });
      lab.show();
      lastBlend = undefined;
      tab.plots.resetGeneration();
      tab.plots.setGuidance(traj.scale);
    },
    onStop() {
      lab.clear();
    },
    build: () => {
      const { ts, alphaBars } = traj;
      const status = () => lab.get("status");
      const steps = [
        {
          title: "Il prompt diventa vettori",
          caption: `"un faro su una scogliera al tramonto": il tokenizer lo spezza in token, il text encoder CLIP (un transformer) produce 77 vettori da 768 numeri.`,
          duration: 4,
          enter() {
            tab.focus("textEncoder", { margin: 1.6 });
            tab.showCode("sd_encode", "diffusers: encode_prompt (tokenizer + text encoder)", ["self.tokenizer(", "max_length", "self.text_encoder("]);
          },
        },
        {
          title: "Rumore iniziale",
          caption: `Si parte da puro rumore gaussiano nel latente 4×64×64, scelto dal seed ${tab.settings.seed}: stesso seed e stesso prompt danno la stessa immagine.`,
          duration: 3.5,
          enter() {
            display(0, 0);
            texEu.set(new Float32Array(vae.z0.length), 0);
            texEc.set(new Float32Array(vae.z0.length), 0);
            gridEps.set(new Float32Array(vae.z0.length));
            texPreview.set(new Float32Array(tab.reference.length));
            status().set(`seed ${tab.settings.seed} · ${ts.length} passi · t parte da ${ts[0]}`);
            pipeline.setTimestep(ts[0]);
            focusLab(tab);
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
          enter() {
            showPredictions(k);
            display(k, 0);
            pipeline.setTimestep(t);
            status().set(`passo ${k + 1}/${ts.length} · t = ${t}`);
            tab.plots.showGeneration(alphaBars, k);
            if (k === 0) tab.showCode("sd_loop", "diffusers: guidance e passo dello scheduler", ["noise_pred_uncond, noise_pred_text", "guidance_scale", "scheduler.step"]);
            if (k === 1) tab.showCode("ddim_step", "diffusers: passo DDIM (eta = 0)", ["pred_original_sample = (sample", "prev_sample ="]);
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
          enter() {
            display(ts.length, 0);
            vae.decode(preview, traj.states.at(-1));
            texPreview.set(preview);
            pipeline.imageTex.set(preview);
            lab.get("preview").title.set("immagine finale (VAE decoder)");
            status().set("fine: VAE decoder");
            tab.focus("image", { margin: 2.2 });
            tab.showCode("sd_decode", "diffusers: decodifica con il VAE", ["vae.decode", "scaling_factor"]);
          },
        },
        {
          title: "Risultato",
          caption: `Fatto: ${ts.length} passi, ${ts.length * 2} chiamate al denoiser (due per passo per la guidance). Prova a cambiare la scala s o il seed nel pannello Azioni e riavvia.`,
          duration: 5,
          enter() {
            pipeline.imageTex.set(preview);
            const { std } = meanStd(preview);
            status().set(`s = ${fmt(traj.scale, 1)} · contrasto ${f2(std)}`);
            focusLab(tab);
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

// ---------------------------------------------------------------------------------------------
// 3. VAE
// ---------------------------------------------------------------------------------------------
function vaeAction(tab) {
  const { lab, vae, reference } = tab;
  let stages;

  /** Immagine ridotta di un fattore f (media dei blocchi) e riportata a 256 per il disegno. */
  function stageImage(f) {
    const size = 256;
    const out = new Float32Array(size * size * 3);
    for (let j = 0; j < size; j++)
      for (let i = 0; i < size; i++) {
        const bj = Math.floor(j / f) * f;
        const bi = Math.floor(i / f) * f;
        for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let y = 0; y < f; y++) for (let x = 0; x < f; x++) s += reference[((bj + y) * size + bi + x) * 3 + c];
          out[(j * size + i) * 3 + c] = s / (f * f);
        }
      }
    return out;
  }

  const reveal = (k, p = 1) =>
    stages.forEach((s, i) => {
      s.mesh.visible = s.title.sprite.visible = i <= k;
      s.mesh.material.opacity = i < k ? 1 : i === k ? p : 0;
      s.mesh.material.transparent = true;
    });

  return {
    id: "vae",
    title: "Dal latente ai pixel (VAE)",
    category: "Immagini",
    summary: "Il decoder del VAE: 4 × 64 × 64 numeri diventano 512 × 512 × 3 pixel. Perché si lavora nel latente.",
    onStart() {
      lab.clear();
      const grid = latentGrid(lab, "z", -24, 0, 7, "latente z_0 · 4 × 64 × 64");
      grid.set(vae.z0);
      const defs = [
        { f: 8, w: 4.2, x: -13.5, title: "64 × 64\n512 canali" },
        { f: 4, w: 6, x: -6.5, title: "128 × 128\n512 canali" },
        { f: 2, w: 8, x: 2.6, title: "256 × 256\n256 canali" },
        { f: 1, w: 10.5, x: 15, title: "512 × 512 × 3\nimmagine" },
      ];
      stages = defs.map((d, i) => {
        const tex = lab.own(new RGBTexture(256, { nearest: d.f > 1 }).set(stageImage(d.f)));
        return lab.panel(`s${i}`, { texture: tex.texture, x: d.x, y: 0, w: d.w, title: d.title, titleHeight: 0.42 });
      });
      lab.text("title", "VAE decoder: la risoluzione sale, i canali scendono", -4, 11.5, { height: 0.75, color: "#7dd3fc" });
      lab.text("note", "Simulazione didattica: le feature intermedie vere hanno centinaia di canali;\nqui mostriamo solo l'immagine a quella risoluzione.", -4, -8.4, { height: 0.4, color: "#94a3b8" });
      lab.show();
      reveal(-1);
      focusLab(tab);
    },
    onStop() {
      lab.clear();
    },
    steps: [
      {
        title: "Il latente",
        caption: "Il latente di Stable Diffusion: 4 canali × 64 × 64 = 16.384 numeri. Un'immagine 512×512 a colori ne ha 786.432: 48 volte di più.",
        duration: 4.5,
        enter() {
          reveal(-1);
          tab.showCode("sd_config", "Stable Diffusion 1.x: configurazione dell'autoencoder", ["z_channels: 4", "ch_mult", "embed_dim: 4"]);
        },
      },
      ...["Il primo blocco del decoder porta il latente a 512 canali alla stessa risoluzione 64×64.", "ResNet block e upsampling: 128×128. La rete 'inventa' i dettagli che il latente aveva solo riassunto.", "Ancora un raddoppio: 256×256, con meno canali (256).", "Ultimo raddoppio e una convoluzione finale a 3 canali: l'immagine 512×512 a colori."].map((caption, k) => ({
        title: ["Primo blocco", "Upsampling ×2", "Upsampling ×2", "Immagine"][k],
        caption,
        duration: 3,
        enter() {
          reveal(k, 0);
        },
        update(p) {
          reveal(k, Math.min(1, p * 2));
        },
      })),
      {
        title: "Perché lavorare nel latente",
        caption: "Il denoiser gira 20-50 volte (il doppio con la guidance): farlo su 64×64 invece che su 512×512 costa molto meno. Il VAE decoder invece si usa una volta sola, alla fine.",
        duration: 5,
        enter() {
          reveal(3, 1);
          tab.showCode("sd_decode", "diffusers: decodifica del latente", ["vae.decode"]);
        },
      },
      {
        title: "Il fattore di scala",
        caption: "Prima del decoder il latente viene diviso per 0,18215 (scale_factor): durante il training lo si era moltiplicato per avere varianza circa 1, comoda per la diffusione.",
        duration: 5,
        enter() {
          reveal(3, 1);
          tab.showCode("sd_decode", "diffusers: il latente viene riscalato", ["scaling_factor"]);
        },
      },
      {
        title: "E l'encoder?",
        caption: "L'encoder del VAE fa il viaggio inverso (immagine → latente). Serve nel training e per partire da una foto (image-to-image), non nella generazione da testo.",
        duration: 5,
        enter() {
          reveal(3, 1);
          tab.focus("vae", { margin: 1.8 });
        },
      },
    ],
  };
}

// ---------------------------------------------------------------------------------------------
// 4. Un passo dentro il denoiser
// ---------------------------------------------------------------------------------------------
function insideAction(tab) {
  const { pipeline } = tab;
  const dim = 0x1e293b;
  const paintUNet = (pred) => recolor(pipeline.unet.mesh, (i, it) => (pred(it, i) ? it.color : dim));
  const paintDiT = (pred) => recolor(pipeline.dit.mesh, (i, it) => (pred(it, i) ? it.color : dim));
  const all = () => true;

  const unetSteps = () => {
    const { rowY, middleY, cross, tLines, skips } = pipeline.unet;
    const lines = (c, t, s) => {
      cross.visible = c;
      tLines.visible = t;
      skips.visible = s;
    };
    return [
      { title: "Ingresso", caption: "La U-Net riceve il latente rumoroso z_t (4×64×64), il passo t e gli embedding del testo. Deve restituire il rumore previsto, della stessa forma.", focus: "unet", paint: all, lines: [true, true, true], code: ["sd_config", "Configurazione della U-Net", ["in_channels: 4", "out_channels: 4", "model_channels: 320"]] },
      { title: "ResNet block e passo t", caption: "I blocchi arancioni sono ResNet block: due convoluzioni con una scorciatoia. L'embedding del passo t viene sommato dentro ognuno: così la rete sa quanto rumore aspettarsi.", paint: (it) => it.kind === "res", lines: [false, true, false], code: ["ddpm_resnet", "DDPM: ResNet block con il passo t", ["temb_proj"]] },
      { title: "Encoder: si scende", caption: "A ogni livello la risoluzione si dimezza (64 → 32 → 16 → 8) e i canali crescono (320 → 640 → 1280): la rete passa dai dettagli alla composizione d'insieme.", paint: (it) => it.x < -1.7 && it.y > middleY + 1, lines: [false, false, false], code: ["sd_config", "channel_mult", ["channel_mult", "num_res_blocks"]] },
      { title: "Attention: self e cross", caption: "I blocchi viola sono transformer: self-attention tra le posizioni dell'immagine e cross-attention verso il testo (query dall'immagine, key e value dai 77 vettori del prompt).", paint: (it) => it.kind === "attn", lines: [true, false, false], code: ["sd_config", "attention_resolutions e context_dim", ["attention_resolutions", "context_dim: 768", "use_spatial_transformer"]] },
      { title: "Il fondo della U", caption: "Al centro (8×8, 1280 canali) un ResNet, un'attention e un altro ResNet: qui ogni posizione 'vede' quasi tutta l'immagine.", paint: (it) => Math.abs(it.y - middleY) < 0.1, lines: [false, false, false] },
      { title: "Decoder e skip connection", caption: "Il decoder risale fino a 64×64. Le skip connection portano le feature dell'encoder dello stesso livello: i dettagli fini non vanno persi nel passaggio dal fondo.", paint: (it) => it.x > 1.7 && Math.abs(it.y - middleY) > 0.1, lines: [false, false, true] },
      { title: "Uscita: il rumore previsto", caption: "L'ultima convoluzione torna a 4 canali: è ε̂, il rumore che la rete crede ci sia in z_t. Lo scheduler lo userà per fare un passo verso l'immagine.", paint: (it) => it.kind === "conv", lines: [false, false, false], focus: "eps" },
    ].map((s) => ({
      title: s.title,
      caption: s.caption,
      duration: 4.5,
      enter() {
        paintUNet(s.paint);
        lines(...(s.lines ?? [true, true, true]));
        if (s.focus) tab.focus(s.focus, { margin: 1.3 });
        if (s.code) tab.showCode(...s.code);
      },
      exit() {
        paintUNet(all);
        lines(true, true, true);
      },
    }));
  };

  const ditSteps = () =>
    [
      { title: "Patchify", caption: "Il latente viene tagliato in patch 2×2: ognuna diventa un token (un vettore). Con un latente 32×32, come in DiT a 256×256, sono 16×16 = 256 token.", paint: (it) => it.kind === "patch" || it.kind === "token", code: ["dit_forward", "DiT: forward", ["x_embedder"]] },
      { title: "Condizionamento adaLN", caption: "Il passo t (e la classe, o il testo in SD3) diventa un vettore c. Un MLP ne ricava scale, shift e gate per ogni blocco: è l'adaLN-Zero di DiT.", paint: (it) => it.kind === "ada", code: ["dit_block", "DiT: blocco con adaLN-Zero", ["adaLN_modulation", "modulate("]] },
      { title: "Self-attention", caption: "Come nel nostro GPT, ogni token guarda gli altri con l'attention. Differenza: nessuna maschera causale, ogni patch vede tutte le patch (sopra e sotto, prima e dopo).", paint: (it) => it.kind === "attn", code: ["dit_block", "DiT: attention e MLP con connessioni residue", ["self.attn("]] },
      { title: "MLP", caption: "Poi un MLP per ogni token, con connessione residua: la stessa struttura dei blocchi del transformer della lezione.", paint: (it) => it.kind === "mlp", code: ["dit_block", "DiT: MLP", ["self.mlp("]] },
      { title: "Il testo in SD3 (MMDiT)", caption: "In Stable Diffusion 3 i token del testo hanno pesi propri ma entrano nella stessa attention dei token dell'immagine: l'informazione scorre nei due sensi.", paint: (it) => it.kind === "textStream" || it.kind === "attn", code: ["sd3_forward", "diffusers: SD3Transformer2DModel.forward", ["time_text_embed", "context_embedder", "block("]] },
      { title: "Unpatchify", caption: "L'ultimo strato riporta ogni token a una patch 2×2 e ricompone il latente: è la previsione (rumore ε e varianza in DiT; velocità nei modelli a rectified flow come SD3).", paint: (it) => it.kind === "patchOut", code: ["dit_forward", "DiT: final layer e unpatchify", ["final_layer", "unpatchify"]] },
    ].map((s) => ({
      title: s.title,
      caption: s.caption,
      duration: 4.5,
      enter() {
        paintDiT(s.paint);
        if (s.code) tab.showCode(...s.code);
      },
      exit() {
        paintDiT(all);
      },
    }));

  return {
    id: "inside",
    title: "Un passo dentro il denoiser",
    category: "Immagini",
    summary: "Cosa succede in UNA chiamata alla U-Net (o al DiT, se è la vista attiva): blocchi, passo t, testo.",
    onStart() {
      tab.lab.clear();
      tab.lod.setMode("detail");
      tab.focus(tab.denoiser === "dit" ? "dit" : "unet", { margin: 1.15, showFrame: false });
    },
    onStop() {
      tab.lod.setMode("auto");
    },
    onEnd() {
      tab.lod.setMode("auto");
    },
    build: () => (tab.denoiser === "dit" ? ditSteps() : unetSteps()),
  };
}

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
