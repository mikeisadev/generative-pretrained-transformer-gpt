/**
 * Azione "Dal latente ai pixel (VAE)": il decoder porta 4 × 64 × 64 numeri a 512 × 512 × 3 pixel.
 */
import { avgPool } from "./imagery.js";
import { ensureLab, focusLab, latentGrid } from "./lab.js";
import { RGBTexture } from "./textures.js";

export function vaeAction(tab) {
  const { lab, vae, reference } = tab;
  let stages;

  const reveal = (k, p = 1) =>
    stages.forEach((s, i) => {
      s.mesh.visible = s.title.sprite.visible = i <= k;
      s.mesh.material.opacity = i < k ? 1 : i === k ? p : 0;
      s.mesh.material.transparent = true;
    });

  return {
    id: "vae",
    title: "Dal latente ai pixel (VAE)",
    category: "Generazione (prompt)",
    summary: "Il decoder del VAE: 4 × 64 × 64 numeri diventano 512 × 512 × 3 pixel. Perché si lavora nel latente.",
    onStart(ctx) {
      lab.clear();
      const grid = latentGrid(lab, "z", -24, 0, 7, "latente z_0\n4 × 64 × 64");
      grid.set(vae.z0);
      const defs = [
        { f: 8, w: 4.2, x: -13.5, title: "64 × 64\n512 canali" },
        { f: 4, w: 6, x: -6.5, title: "128 × 128\n512 canali" },
        { f: 2, w: 8, x: 2.6, title: "256 × 256\n256 canali" },
        { f: 1, w: 10.5, x: 15, title: "512 × 512 × 3\nimmagine" },
      ];
      stages = defs.map((d, i) => {
        // l'immagine alla risoluzione dello stadio (media dei blocchi f × f), disegnata a pixel netti
        const tex = lab.own(new RGBTexture(vae.size / d.f, { nearest: d.f > 1 }).set(avgPool(reference, vae.size, d.f)));
        return lab.panel(`s${i}`, { texture: tex.texture, x: d.x, y: 0, w: d.w, title: d.title, titleHeight: 0.8 });
      });
      lab.text("title", "VAE decoder: la risoluzione sale, i canali scendono", -4.5, 12.5, { height: 1.3, color: "#7dd3fc" });
      lab.text("note", "Simulazione didattica: le feature vere hanno centinaia di canali;\nqui vediamo solo l'immagine a quella risoluzione.", -4.5, -8.6, { height: 0.75, color: "#94a3b8" });
      lab.show();
      reveal(-1);
      ctx.data.view = "lab";
      focusLab(tab);
    },
    onStop() {
      lab.clear();
      tab.hideCode();
    },
    steps: [
      {
        title: "Il latente",
        caption: "Il latente di Stable Diffusion: 4 canali × 64 × 64 = 16.384 numeri. Un'immagine 512×512 a colori ne ha 786.432: 48 volte di più.",
        duration: 4.5,
        enter(ctx) {
          ensureLab(tab, ctx);
          reveal(-1);
          tab.showCode("sd_config", "Stable Diffusion 1.x: configurazione dell'autoencoder", ["z_channels: 4", "ch_mult", "embed_dim: 4"]);
        },
      },
      ...["Il primo blocco del decoder porta il latente a 512 canali alla stessa risoluzione 64×64.", "ResNet block e upsampling: 128×128. La rete 'inventa' i dettagli che il latente aveva solo riassunto.", "Ancora un raddoppio: 256×256, con meno canali (256).", "Ultimo raddoppio e una convoluzione finale a 3 canali: l'immagine 512×512 a colori."].map((caption, k) => ({
        title: ["Primo blocco", "Upsampling ×2", "Upsampling ×2", "Immagine"][k],
        caption,
        duration: 3,
        enter(ctx) {
          ensureLab(tab, ctx);
          reveal(k, 0);
          tab.showCode("sd_config", "Stable Diffusion 1.x: configurazione dell'autoencoder", ["z_channels: 4", "ch_mult", "embed_dim: 4"]);
        },
        update(p) {
          reveal(k, Math.min(1, p * 2));
        },
      })),
      {
        title: "Perché lavorare nel latente",
        caption: "Il denoiser gira 20-50 volte (il doppio con la guidance): farlo su 64×64 invece che su 512×512 costa molto meno. Il VAE decoder invece si usa una volta sola, alla fine.",
        duration: 5,
        enter(ctx) {
          ensureLab(tab, ctx);
          reveal(3, 1);
          tab.showCode("sd_decode", "diffusers: decodifica del latente", ["vae.decode"]);
        },
      },
      {
        title: "Il fattore di scala",
        caption: "Prima del decoder il latente viene diviso per 0,18215 (scale_factor): durante il training lo si era moltiplicato per avere varianza circa 1, comoda per la diffusione.",
        duration: 5,
        enter(ctx) {
          ensureLab(tab, ctx);
          reveal(3, 1);
          tab.showCode("sd_decode", "diffusers: il latente viene riscalato", ["scaling_factor"]);
        },
      },
      {
        title: "E l'encoder?",
        caption: "L'encoder del VAE fa il viaggio inverso (immagine → latente). Serve nel training e per partire da una foto (image-to-image), non nella generazione da testo.",
        duration: 5,
        enter(ctx) {
          ctx.data.view = "vae";
          reveal(3, 1);
          tab.focus(["vae", "image"], { margin: 1.4 });
          tab.showCode("sd_config", "Stable Diffusion 1.x: l'autoencoder (encoder e decoder)", ["first_stage_config", "AutoencoderKL"]);
        },
      },
    ],
  };
}
