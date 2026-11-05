/**
 * Azione "Un passo dentro il denoiser": illumina i blocchi della U-Net (o del DiT, se è la vista
 * attiva) uno alla volta, ricolorando le istanze degli InstancedMesh (nessuna geometria nuova).
 */
import { recolor } from "./shapes.js";

export function insideAction(tab) {
  const { pipeline } = tab;
  const dim = 0x1e293b;
  const paintUNet = (pred) => recolor(pipeline.unet.mesh, (i, it) => (pred(it, i) ? it.color : dim));
  const paintDiT = (pred) => recolor(pipeline.dit.mesh, (i, it) => (pred(it, i) ? it.color : dim));
  const all = () => true;
  /** I passi senza codice proprio tengono quello del passo precedente (stato deterministico). */
  const withPreviousCode = (s, i, list) => ({ ...s, code: s.code ?? list.slice(0, i).findLast((p) => p.code).code });

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
      { title: "Il fondo della U", caption: "Al centro (8×8, 1280 canali) un ResNet, un'attention e un altro ResNet. Ogni posizione riassume una zona grande (64×64 pixel dell'immagine finale): qui si decide la composizione d'insieme.", paint: (it) => Math.abs(it.y - middleY) < 0.1, lines: [false, false, false] },
      { title: "Decoder e skip connection", caption: "Il decoder risale fino a 64×64. Le skip connection portano le feature dell'encoder dello stesso livello: i dettagli fini non vanno persi nel passaggio dal fondo.", paint: (it) => it.x > 1.7 && Math.abs(it.y - middleY) > 0.1, lines: [false, false, true] },
      { title: "Uscita: il rumore previsto", caption: "L'ultima convoluzione torna a 4 canali: è ε̂, il rumore che la rete crede ci sia in z_t. Lo scheduler lo userà per fare un passo verso l'immagine.", paint: (it) => it.kind === "conv" && it.x > 0, lines: [false, false, false], focus: ["eps", "scheduler"], margin: 1.6 },
    ].map(withPreviousCode).map((s) => ({
      title: s.title,
      caption: s.caption,
      duration: 4.5,
      enter() {
        paintUNet(s.paint);
        lines(...s.lines);
        // ogni passo decide inquadratura e codice: così "passo indietro" ritrova lo stato giusto
        tab.focus(s.focus ?? "unet", { margin: s.margin ?? 1.15, showFrame: false });
        tab.showCode(...s.code);
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
        tab.showCode(...s.code);
      },
      exit() {
        paintDiT(all);
      },
    }));

  return {
    id: "inside",
    title: "Un passo dentro il denoiser",
    category: "Generazione (prompt)",
    summary: "Cosa succede in UNA chiamata alla U-Net (o al DiT, se è la vista attiva): blocchi, passo t, testo.",
    onStart() {
      tab.lab.clear();
      tab.lockLevel("detail");
      tab.focus(tab.denoiser === "dit" ? "dit" : "unet", { margin: 1.15, showFrame: false });
    },
    onStop() {
      tab.lockLevel("auto");
      tab.hideCode();
    },
    onEnd() {
      tab.lockLevel("auto");
    },
    build: () => (tab.denoiser === "dit" ? ditSteps() : unetSteps()),
  };
}
