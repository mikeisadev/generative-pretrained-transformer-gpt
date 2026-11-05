/**
 * Azione animata sul video: perché i modelli video generano i frame INSIEME.
 *
 * Usa la scena di videoScene.js: cubo di frame, VAE 3D causale, latenti, patch spazio-temporali,
 * attention spaziale / temporale / completa, denoising congiunto e confronto con la generazione
 * "un frame alla volta" (simulazione didattica).
 */
import { fmt } from "../../core/dom.js";
import { fillGaussian, mulberry32, qSample, SCHEDULES } from "./math.js";
import { FRAMES, LATENT_FRAMES, TOKENS } from "./videoScene.js";

const QUERY = 22; // token di esempio: latente 1, riga 1, colonna 2
const FPS = 8;

export function installVideo(tab) {
  const v = tab.video;
  const schedule = SCHEDULES.sd;
  const size = v.frames[0].data.length;
  let noises = null;
  let work = null;

  const reset = () => {
    v.tokens.visible = false;
    v.seqLabel.visible = false;
    v.patchGrid.visible = false;
    v.setAttention(null, QUERY);
    v.frames.forEach((f) => {
      f.mesh.scale.setScalar(1);
      f.tex.set(f.data);
    });
    v.latents.forEach((l) => l.mesh.scale.setScalar(1));
  };

  /** Mostra il frame k (dei frame "insieme" o "uno alla volta") nel lettore. */
  const playerFrame = (list, k, title) => {
    v.player.tex.set(list[k].data);
    v.player.label.set(`${title}\nframe ${k + 1}/${list.length}`);
  };

  const steps = [
    {
      title: "Un transformer per ogni frame?",
      caption: "Ipotesi naturale: generare un'immagine per ogni frame. I modelli video recenti invece generano tutti i frame insieme, come un unico blocco: vediamo come.",
      duration: 5,
      enter() {
        reset();
        tab.focus(v.group, { showFrame: false, margin: 1.05 });
      },
    },
    {
      title: "Il video è un cubo",
      caption: "Un video è una pila di frame: larghezza × altezza × tempo. Qui 16 frame generati nel browser: il fascio del faro ruota e il sole scende.",
      duration: 4,
      enter() {
        reset();
        tab.focus("videoFrames", { margin: 1.5 });
      },
      update(p) {
        // un'onda di evidenziazione percorre il tempo
        const k = Math.floor(p * FRAMES * 0.999);
        v.frames.forEach((f, i) => f.mesh.scale.setScalar(i === k ? 1.08 : 1));
      },
    },
    {
      title: "VAE 3D causale",
      caption: "Un VAE 3D comprime spazio E tempo. In HunyuanVideo: 8 volte nello spazio, 4 nel tempo, 16 canali. 'Causale': ogni latente dipende solo dai frame passati, non dai futuri.",
      duration: 6,
      enter() {
        reset();
        tab.focus("videoVae", { margin: 2.6 });
        tab.showCode("causal_conv3d", "diffusers: convoluzione 3D causale (VAE di HunyuanVideo)", ["time_causal_padding", "kernel_size[2] - 1"]);
      },
      update(p) {
        // ogni gruppo di 4 frame si accende insieme al latente che lo riassume
        const g = Math.min(LATENT_FRAMES - 1, Math.floor(p * LATENT_FRAMES));
        v.frames.forEach((f, i) => f.mesh.scale.setScalar(Math.floor(i / 4) === g ? 1.07 : 1));
        v.latents.forEach((l, i) => l.mesh.scale.setScalar(i === g ? 1.1 : 1));
      },
    },
    {
      title: "Patch spazio-temporali",
      caption: "Il cubo di latenti viene tagliato in cubetti (patch spazio-temporali): ognuno diventa un token. Il report di Sora (OpenAI, 2024) usa proprio questa idea.",
      duration: 4.5,
      enter() {
        reset();
        v.patchGrid.visible = true;
        v.tokens.visible = true;
        v.setTokenLayout(0, 0.01);
        tab.focus("videoLatent", { margin: 1.4 });
        tab.showCode("dit_forward", "DiT: le patch diventano token (stessa idea, in 2D)", ["x_embedder"]);
      },
      update(p) {
        v.setTokenLayout(0, Math.min(1, p * 1.6));
      },
    },
    {
      title: "Una sequenza di token",
      caption: `I token si mettono in fila, come i caratteri nel nostro GPT: qui ${TOKENS} (4 latenti × 16 patch). In un modello vero sono decine di migliaia.`,
      duration: 4,
      enter() {
        reset();
        v.patchGrid.visible = true;
        v.tokens.visible = true;
        v.seqLabel.visible = true;
        v.setTokenLayout(0);
        tab.focus(v.group, { showFrame: false, margin: 1.05 });
      },
      update(p) {
        const e = p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2;
        v.setTokenLayout(e);
      },
    },
    ...[
      {
        mode: "spatial",
        color: 0xc4b5fd,
        title: "Attention spaziale",
        caption: "Attention spaziale: il token evidenziato guarda solo i token dello STESSO latente (stesso istante). Così si capisce la composizione di ogni frame.",
      },
      {
        mode: "temporal",
        color: 0x67e8f9,
        title: "Attention temporale",
        caption: "Attention temporale: lo stesso punto guarda la stessa posizione negli ALTRI istanti. Video Diffusion Models (Ho et al., 2022) alterna attention spaziale e temporale.",
        code: ["temporal_attn", "diffusers: attention temporale (ogni pixel guarda i frame)", ["batch_size * height * width, num_frames"]],
      },
      {
        mode: "full",
        color: 0xfde68a,
        title: "Attention completa 3D",
        caption: `Attention completa: ogni token guarda tutti gli altri, nello spazio e nel tempo (HunyuanVideo la chiama "Full Attention"). Costa di più: con N token servono N² confronti (qui ${TOKENS}² = ${fmt(TOKENS * TOKENS)}).`,
      },
    ].map((s) => ({
      title: s.title,
      caption: s.caption,
      duration: 4.5,
      enter() {
        reset();
        v.tokens.visible = true;
        v.patchGrid.visible = true;
        v.setTokenLayout(0);
        v.attention.material.color.setHex(s.color);
        v.setAttention(s.mode, QUERY, 0);
        tab.focus("videoLatent", { margin: 1.4 });
        if (s.code) tab.showCode(...s.code);
      },
      update(p) {
        v.setAttention(s.mode, QUERY, Math.min(1, p * 1.5));
      },
    })),
    {
      title: "Denoising di tutti i frame",
      caption: "Il denoiser lavora sull'intero cubo a ogni passo: tutti i frame escono dal rumore INSIEME e restano coerenti (qui mostrato sui pixel per chiarezza; il modello vero lo fa sui latenti).",
      duration: 6,
      enter() {
        reset();
        if (!noises) {
          noises = v.frames.map((_, k) => fillGaussian(new Float32Array(size), mulberry32(500 + k)));
          work = new Float32Array(size);
        }
        tab.focus("videoFrames", { margin: 1.5 });
        tab.showCode("temporal_attn", "diffusers: attention temporale", ["reshape(batch_size * height * width, num_frames"]);
      },
      update(p) {
        const t = Math.round((1 - p) * (schedule.T - 1));
        const ab = schedule.alphaBar[t];
        v.frames.forEach((f, k) => f.tex.set(qSample(work, f.data, noises[k], ab)));
      },
      exit() {
        v.frames.forEach((f) => f.tex.set(f.data));
      },
    },
    {
      title: "Video coerente",
      caption: "Il risultato scorre senza salti: il faro resta dov'è, cambia solo ciò che si muove davvero (il fascio di luce, il sole, le onde).",
      duration: 5,
      enter() {
        reset();
        tab.focus("videoOut", { margin: 1.8 });
      },
      update(p, dt, ctx) {
        ctx.data.clock = (ctx.data.clock ?? 0) + dt;
        const k = Math.floor(ctx.data.clock * FPS) % FRAMES;
        playerFrame(v.frames, k, "Video generato (tutti insieme)");
      },
    },
    {
      title: "Perché non un frame alla volta",
      caption: "Se ogni frame nascesse per conto suo, con il suo rumore, la scena cambierebbe a ogni frame (righe in alto): faro che salta, colori che cambiano. Vedi il grafico 'Differenza tra frame consecutivi'.",
      duration: 7,
      enter() {
        reset();
        tab.plots.showFlicker();
        tab.focus("videoCompare", { margin: 1.25 });
      },
      update(p, dt, ctx) {
        ctx.data.clock = (ctx.data.clock ?? 0) + dt;
        const k = Math.floor(ctx.data.clock * 4) % v.indep.length;
        playerFrame(v.indep, k, "Un frame alla volta");
      },
      exit() {
        playerFrame(v.frames, 0, "Video generato");
      },
    },
  ];

  tab.ui.player.register({
    id: "video",
    title: "Video: tutti i frame insieme",
    category: "Video",
    summary: "Cubo spazio-temporale, VAE 3D causale, patch, attention spaziale contro temporale, coerenza tra frame.",
    steps,
    onStart() {
      tab.lab.clear();
      tab.lod.setMode("detail");
    },
    onStop() {
      reset();
      tab.lod.setMode("auto");
    },
    onEnd() {
      tab.lod.setMode("auto");
    },
  });
}
