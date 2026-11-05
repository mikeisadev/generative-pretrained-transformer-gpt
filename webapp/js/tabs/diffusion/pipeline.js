/**
 * Livello "Dettaglio" della scena: la pipeline text-to-image a diffusione latente
 * (Stable Diffusion 1.x), da sinistra a destra:
 *
 *   prompt → tokenizer → text encoder CLIP → embedding del testo (77 × 768)
 *   latente di rumore z_t (4 × 64 × 64) + passo t
 *   denoiser: U-Net (vista base) oppure Diffusion Transformer (vista alternativa)
 *   ε̂ previsto → scheduler → z_{t−1} (anello ripetuto) → VAE decoder → immagine 512 × 512
 *
 * Le dimensioni della U-Net vengono dal file di configurazione ufficiale v1-inference.yaml
 * (model_channels 320, channel_mult [1, 2, 4, 4], num_res_blocks 2, attention_resolutions [4, 2, 1]).
 */
import * as THREE from "three";
import { makeBlock, makeFlow } from "../../core/diagram.js";
import { mulberry32, timestepEmbedding } from "./math.js";
import { PALETTE, instancedBoxes, lineSegments, placeLabel, placeText, texturedPlane } from "./shapes.js";
import { ChannelTexture, RGBTexture, matrixTexture } from "./textures.js";

export const PROMPT = "un faro su una scogliera al tramonto";
// Suddivisione indicativa: il tokenizer BPE di CLIP lavora su pezzi di parola e spezzerebbe
// le parole italiane in frammenti; qui mostriamo parole intere per leggibilità.
export const TOKENS = ["<inizio>", "un", "faro", "su", "una", "scogliera", "al", "tramonto", "<fine>", "… fino a 77"];

/** Livelli della U-Net di Stable Diffusion 1.x (latente 64 × 64). */
export const UNET_LEVELS = [
  { res: 64, ch: 320, attn: true },
  { res: 32, ch: 640, attn: true },
  { res: 16, ch: 1280, attn: true },
  { res: 8, ch: 1280, attn: false },
];

const HEIGHT = { 64: 2.6, 32: 2.0, 16: 1.5, 8: 1.1 };
const DEPTH = { 320: 1.2, 640: 1.8, 1280: 2.5 };

/** Posizioni principali (riusate dai livelli Blocchi/Panoramica e dalle azioni). */
export const LAYOUT = {
  prompt: new THREE.Vector3(-55, 0, 0),
  encoder: new THREE.Vector3(-45, 0, 0),
  textEmb: new THREE.Vector3(-36.5, 0, 0),
  latent: new THREE.Vector3(-28, 1, 0),
  timestep: new THREE.Vector3(-28, -8.5, 0),
  denoiser: new THREE.Vector3(0, 0, 0),
  eps: new THREE.Vector3(22.5, 1, 0),
  scheduler: new THREE.Vector3(0, -13.5, 0),
  vae: new THREE.Vector3(33, 0, 0),
  image: new THREE.Vector3(46, 0, 0),
};

/** Quattro piani (uno per canale) sfalsati in diagonale: il "cubo" del latente. */
function latentStack(textures, size, title, sub) {
  const group = new THREE.Group();
  const step = size * 0.13;
  textures.forEach((t, c) => {
    const plane = texturedPlane(t.texture, size, size);
    plane.position.set((c - 1.5) * step, (1.5 - c) * step, (1.5 - c) * 0.9);
    group.add(plane);
  });
  placeLabel(group, title, 0, size / 2 + 2, 0, { size: 0.95, sub });
  return group;
}

function buildTextSide(group, registry) {
  // prompt
  const prompt = new THREE.Group();
  placeLabel(prompt, `"${PROMPT}"`, LAYOUT.prompt.x + 4, 13.4, 0, { size: 0.95, color: "#e0e7ff" });
  placeLabel(prompt, "Prompt", LAYOUT.prompt.x - 6.5, 15, 0, { size: 0.8, color: "#c4b5fd" });
  group.add(prompt);
  registry.set("prompt", prompt);

  // token: un InstancedMesh di "gettoni" con etichetta
  const tokenGroup = new THREE.Group();
  const items = TOKENS.map((_, i) => ({ x: LAYOUT.prompt.x, y: 6.1 - i * 1.35, z: 0, sx: 5.4, sy: 1.1, sz: 0.4, color: i === TOKENS.length - 1 ? 0x334155 : 0x312e81 }));
  const chips = instancedBoxes(items);
  tokenGroup.add(chips);
  TOKENS.forEach((tok, i) => placeLabel(tokenGroup, tok, LAYOUT.prompt.x, items[i].y, 0.3, { size: 0.6, color: i === TOKENS.length - 1 ? "#94a3b8" : "#e0e7ff" }));
  placeLabel(tokenGroup, "Tokenizer", LAYOUT.prompt.x, 9.2, 0, { size: 0.95, color: "#c4b5fd", sub: "testo → token" });
  group.add(tokenGroup);
  registry.set("tokens", tokenGroup);

  // text encoder: 12 layer transformer (CLIP ViT-L/14: 12 layer, larghezza 768)
  const encGroup = new THREE.Group();
  const layers = Array.from({ length: 12 }, (_, i) => ({ x: LAYOUT.encoder.x - 3.6 + i * 0.65, y: 0, z: 0, sx: 0.36, sy: 10, sz: 4, color: i % 2 ? 0x7c3aed : 0x6d28d9 }));
  encGroup.add(instancedBoxes(layers));
  placeLabel(encGroup, "Text encoder", LAYOUT.encoder.x, 7.6, 0, { size: 0.95, color: "#c4b5fd", sub: "CLIP · 12 layer" });
  group.add(encGroup);
  registry.set("textEncoder", encGroup);

  // embedding del testo: 77 righe (token) × 768 colonne (larghezza compressa nel disegno)
  const rng = mulberry32(11);
  const embTex = matrixTexture(77, 48, (r) => (r > 9 ? (rng() - 0.5) * 0.35 : (rng() - 0.5) * 2.2));
  const emb = texturedPlane(embTex, 3, 10);
  emb.position.copy(LAYOUT.textEmb);
  const embGroup = new THREE.Group();
  embGroup.add(emb);
  placeLabel(embGroup, "Embedding", LAYOUT.textEmb.x, 7.6, 0, { size: 0.95, color: "#c4b5fd", sub: "77 × 768" });
  group.add(embGroup);
  registry.set("textEmb", embGroup);
}

/** U-Net: encoder a scalini a sinistra, blocco centrale in basso, decoder a destra, skip connection. */
function buildUNet(textAnchor) {
  const group = new THREE.Group();
  const items = [];
  const rowY = [7, 3.4, -0.2, -3.8];
  const middleY = -7.2;
  const gap = 0.28;
  const ends = []; // per le skip connection: fine encoder e inizio decoder di ogni livello
  const attnTops = [];
  const resBottoms = [];
  const push = (kind, x, y, level, w) => {
    const h = HEIGHT[level.res] * (kind === "sample" ? 0.6 : kind === "conv" ? 0.85 : 1);
    const d = DEPTH[level.ch] * (kind === "attn" ? 0.92 : 1);
    const color = kind === "res" ? PALETTE.resnet : kind === "attn" ? PALETTE.attn : PALETTE.sample;
    items.push({ x: x + w / 2, y, z: 0, sx: w, sy: h, sz: d, color, kind });
    if (kind === "attn") attnTops.push(new THREE.Vector3(x + w / 2, y + h / 2, 0));
    if (kind === "res") resBottoms.push(new THREE.Vector3(x + w / 2, y - h / 2, 0));
    return x + w + gap;
  };
  const W = { res: 0.75, attn: 0.6, sample: 0.36, conv: 0.36 };

  // encoder
  UNET_LEVELS.forEach((level, i) => {
    let x = -18.5 + i * 1.7;
    const y = rowY[i];
    const start = x;
    if (i === 0) x = push("conv", x, y, level, W.conv);
    for (let r = 0; r < 2; r++) {
      x = push("res", x, y, level, W.res);
      if (level.attn) x = push("attn", x, y, level, W.attn);
    }
    if (i < UNET_LEVELS.length - 1) x = push("sample", x, y, level, W.sample);
    ends[i] = { encStart: start, encEnd: x - gap, y };
  });
  // blocco centrale: ResNet, attention, ResNet a 8 × 8
  {
    const level = UNET_LEVELS[3];
    let x = -1.6;
    x = push("res", x, middleY, level, W.res);
    x = push("attn", x, middleY, { ...level, ch: 1280 }, W.attn);
    push("res", x, middleY, level, W.res);
  }
  // decoder (dal livello più basso al più alto): 3 ResNet (+ attention) e un upsample
  for (let i = UNET_LEVELS.length - 1; i >= 0; i--) {
    const level = UNET_LEVELS[i];
    const y = rowY[i];
    const count = 3 * (W.res + gap) + (level.attn ? 3 * (W.attn + gap) : 0) + (W.sample + gap);
    let x = 18.5 - i * 1.7 - count;
    ends[i].decStart = x;
    for (let r = 0; r < 3; r++) {
      x = push("res", x, y, level, W.res);
      if (level.attn) x = push("attn", x, y, level, W.attn);
    }
    x = push(i === 0 ? "conv" : "sample", x, y, level, W.sample);
    ends[i].decEnd = x - gap;
  }
  const mesh = instancedBoxes(items);
  group.add(mesh);

  // collegamenti: discesa nell'encoder, risalita nel decoder, skip connection orizzontali
  const flowPairs = [];
  for (let i = 0; i < 3; i++) {
    flowPairs.push([new THREE.Vector3(ends[i].encEnd, rowY[i], 0), new THREE.Vector3(ends[i + 1].encStart + 0.2, rowY[i + 1], 0)]);
    flowPairs.push([new THREE.Vector3(ends[i + 1].decEnd, rowY[i + 1], 0), new THREE.Vector3(ends[i].decStart, rowY[i], 0)]);
  }
  flowPairs.push([new THREE.Vector3(ends[3].encEnd, rowY[3], 0), new THREE.Vector3(-1.6, middleY, 0)]);
  flowPairs.push([new THREE.Vector3(1.9, middleY, 0), new THREE.Vector3(ends[3].decStart, rowY[3], 0)]);
  group.add(lineSegments(flowPairs, PALETTE.flow, { opacity: 0.8 }));
  const skipPairs = ends.map((e) => [new THREE.Vector3(e.encEnd + 0.1, e.y, 0), new THREE.Vector3(e.decStart - 0.1, e.y, 0)]);
  const skips = lineSegments(skipPairs, PALETTE.skip, { opacity: 0.45 });
  group.add(skips);

  // cross-attention: dal testo a ogni blocco di attention (Q dall'immagine, K e V dal testo)
  const cross = lineSegments(attnTops.map((p) => [textAnchor, p]), PALETTE.text, { opacity: 0.22 });
  group.add(cross);
  // passo t: un "bus" sotto la U-Net e un trattino verso ogni ResNet block
  const busY = -9.6;
  const tPairs = [[new THREE.Vector3(LAYOUT.timestep.x + 2.6, LAYOUT.timestep.y, 0), new THREE.Vector3(-19, busY, 0)], [new THREE.Vector3(-19, busY, 0), new THREE.Vector3(19, busY, 0)]];
  for (const p of resBottoms) tPairs.push([p, new THREE.Vector3(p.x, p.y - 0.55, 0)]);
  const tLines = lineSegments(tPairs, PALETTE.time, { opacity: 0.55 });
  group.add(tLines);

  // etichette
  placeLabel(group, "U-Net", 0, 13.8, 0, { size: 1.6, color: "#fde68a", sub: "circa 860 milioni di parametri" });
  placeLabel(group, "Encoder", -14.5, 10.4, 0, { size: 1.1, color: "#e2e8f0" });
  placeLabel(group, "Decoder", 14.5, 10.4, 0, { size: 1.1, color: "#e2e8f0" });
  UNET_LEVELS.forEach((l, i) => placeLabel(group, `${l.res}×${l.res} · ${l.ch} canali`, 0, rowY[i] + 0.85, 0, { size: 0.9, color: "#94a3b8" }));
  placeLabel(group, "blocco centrale 8×8", 0, middleY - 1.9, 0, { size: 0.9, color: "#94a3b8" });
  placeLabel(group, "skip connection", 0, rowY[0] - 0.85, 0, { size: 0.85, color: "#67e8f9" });
  placeLabel(group, "passo t → ogni ResNet", 0, busY - 1, 0, { size: 0.9, color: "#86efac" });
  placeLabel(group, "cross-attention", -21.5, 10.4, 0, { size: 0.9, color: "#c4b5fd" });

  return { group, mesh, items, cross, tLines, skips, rowY, middleY };
}

/** Diffusion Transformer: patchify → N blocchi transformer con adaLN → unpatchify. */
function buildDiT(textAnchor) {
  const group = new THREE.Group();
  const items = [];
  // patchify: griglia 8 × 8 di patch (rappresentativa)
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) items.push({ x: -17.5 + c * 0.62, y: 2.2 - r * 0.62, z: 0, sx: 0.54, sy: 0.54, sz: 0.25, color: 0x0e7490, kind: "patch" });
  // sequenza di token
  for (let i = 0; i < 8; i++) items.push({ x: -11.2, y: 4.6 - i * 0.95, z: 0, sx: 0.8, sy: 0.75, sz: 0.8, color: 0x0891b2, kind: "token" });
  // 8 blocchi DiT, ognuno con 4 sotto-strati (dal basso): adaLN, self-attention, adaLN, MLP
  const sub = [
    { kind: "ada", color: PALETTE.time, h: 1.0 },
    { kind: "attn", color: PALETTE.attn, h: 3.4 },
    { kind: "ada", color: PALETTE.time, h: 1.0 },
    { kind: "mlp", color: PALETTE.resnet, h: 3.4 },
  ];
  const blockX = [];
  for (let b = 0; b < 8; b++) {
    const x = -8 + b * 2.35;
    blockX.push(x);
    let y = -5.2;
    for (const s of sub) {
      items.push({ x, y: y + s.h / 2, z: 0, sx: 1.1, sy: s.h, sz: 3, color: s.color, kind: s.kind });
      y += s.h + 0.18;
    }
    // flusso del testo (MMDiT di SD3): pesi propri, si unisce all'attention dell'immagine
    items.push({ x, y: 6.6, z: 0, sx: 0.8, sy: 1.6, sz: 1.6, color: PALETTE.text, kind: "textStream" });
  }
  // strato finale + unpatchify
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) items.push({ x: 12.6 + c * 0.62, y: 2.2 - r * 0.62, z: 0, sx: 0.54, sy: 0.54, sz: 0.25, color: 0x155e75, kind: "patchOut" });
  const mesh = instancedBoxes(items);
  group.add(mesh);

  const pairs = [];
  // condizionamento: c = emb(t) + emb(classe o testo "pooled") → adaLN di ogni blocco
  const condY = -8.6;
  for (const x of blockX) pairs.push([new THREE.Vector3(x, condY + 0.6, 0), new THREE.Vector3(x, -5.2, 0)]);
  const condLines = lineSegments(pairs, PALETTE.time, { opacity: 0.6 });
  group.add(condLines);
  const joint = lineSegments(blockX.map((x) => [new THREE.Vector3(x, 5.8, 0), new THREE.Vector3(x, 3.1, 0)]), PALETTE.text, { opacity: 0.7 });
  group.add(joint);
  const textIn = lineSegments([[textAnchor, new THREE.Vector3(blockX[0] - 0.6, 6.6, 0)]], PALETTE.text, { opacity: 0.35 });
  group.add(textIn);
  const flow = lineSegments(
    [
      [new THREE.Vector3(-12.9, 0, 0), new THREE.Vector3(-11.7, 0, 0)],
      [new THREE.Vector3(-10.7, 0, 0), new THREE.Vector3(-8.6, 0, 0)],
      [new THREE.Vector3(blockX.at(-1) + 0.6, 0, 0), new THREE.Vector3(12.2, 0, 0)],
    ],
    PALETTE.flow,
    { opacity: 0.9 },
  );
  group.add(flow);

  const cond = makeBlock({ title: "c = emb(t) + emb(classe o testo) → MLP → scale, shift, gate", width: 17 / 2.6, height: 3.4 / 2.6, color: "#86efac" });
  cond.scale.setScalar(2.6);
  cond.position.set(0.2, condY - 0.7, 0);
  group.add(cond);

  placeLabel(group, "Diffusion Transformer", 0.3, 13.8, 0, { size: 1.6, color: "#fde68a", sub: "DiT-XL/2: 28 blocchi · qui 8" });
  placeLabel(group, "Patchify", -15.3, 4.8, 0, { size: 1, sub: "patch 2×2" });
  placeLabel(group, "Unpatchify", 14.8, 4.8, 0, { size: 1, sub: "token → latente" });
  placeText(group, "ogni blocco, dal basso:\nadaLN · attention\nadaLN · MLP", -15.3, -4.9, 0, { height: 0.75 });
  placeLabel(group, "testo (MMDiT, SD3): attention congiunta", 0.3, 8.8, 0, { size: 0.9, color: "#c4b5fd" });
  return { group, mesh, items, condLines, joint, blockX };
}

/** Decoder del VAE: la risoluzione sale (64 → 512) mentre i canali scendono (512 → 128). */
function buildVAE() {
  const group = new THREE.Group();
  const stages = [
    { res: 64, ch: 512, h: 3, d: 2.4 },
    { res: 128, ch: 512, h: 4.2, d: 2.4 },
    { res: 256, ch: 256, h: 5.6, d: 1.7 },
    { res: 512, ch: 128, h: 7.2, d: 1.1 },
  ];
  const items = stages.map((s, i) => ({ x: LAYOUT.vae.x - 4.2 + i * 2.8, y: 0, z: 0, sx: 1.2, sy: s.h, sz: s.d, color: PALETTE.vae, kind: "vae" }));
  group.add(instancedBoxes(items));
  stages.forEach((s, i) => placeLabel(group, String(s.res), items[i].x, -s.h / 2 - 0.8, 0, { size: 0.62, color: "#94a3b8" }));
  placeLabel(group, "VAE decoder", LAYOUT.vae.x, 6.6, 0, { size: 0.95, color: "#7dd3fc", sub: "fattore 8" });
  return { group, items };
}

/**
 * Costruisce il livello di dettaglio. Restituisce i riferimenti usati da viste e azioni.
 */
export function buildPipeline() {
  const group = new THREE.Group();
  const registry = new Map();
  buildTextSide(group, registry);
  const textAnchor = new THREE.Vector3(LAYOUT.textEmb.x + 1.6, LAYOUT.textEmb.y + 3, 0);

  // latente di rumore e passo t
  const latentTex = Array.from({ length: 4 }, () => new ChannelTexture(32));
  const latent = latentStack(latentTex, 5.2, "Latente z_t", "4 × 64 × 64");
  latent.position.copy(LAYOUT.latent);
  group.add(latent);
  registry.set("latent", latent);
  const tTex = new ChannelTexture(32, { height: 1, range: 1 });
  const tEmb = new Float32Array(32);
  const tPlane = texturedPlane(tTex.texture, 5.2, 0.9);
  const tGroup = new THREE.Group();
  tPlane.position.copy(LAYOUT.timestep);
  tGroup.add(tPlane);
  placeLabel(tGroup, "Passo t", LAYOUT.timestep.x, LAYOUT.timestep.y + 1.9, 0, { size: 0.8, color: "#86efac", sub: "seni e coseni" });
  group.add(tGroup);
  registry.set("timestep", tGroup);

  // denoiser: due alternative, una visibile alla volta
  const unet = buildUNet(textAnchor);
  const dit = buildDiT(textAnchor);
  dit.group.visible = false;
  group.add(unet.group, dit.group);
  registry.set("unet", unet.group);
  registry.set("dit", dit.group);

  // rumore previsto
  const epsTex = Array.from({ length: 4 }, () => new ChannelTexture(32));
  const eps = latentStack(epsTex, 3.6, "Rumore ε̂", "previsto");
  eps.position.copy(LAYOUT.eps);
  group.add(eps);
  registry.set("eps", eps);

  // scheduler e anello del campionamento
  const scheduler = makeBlock({ title: "Scheduler (DDIM)", subtitle: "z_t, ε̂ → z_t−1", text: "Toglie il rumore previsto. Si ripete a ogni passo.", width: 11 / 1.8, height: 3.4 / 1.8, color: "#a5f3fc" });
  scheduler.scale.setScalar(1.8);
  scheduler.position.copy(LAYOUT.scheduler);
  group.add(scheduler);
  registry.set("scheduler", scheduler);
  const flows = [
    makeFlow([new THREE.Vector3(LAYOUT.eps.x, -2.2, 0), new THREE.Vector3(LAYOUT.eps.x, LAYOUT.scheduler.y, 0), new THREE.Vector3(5.6, LAYOUT.scheduler.y, 0)], { radius: 0.09 }),
    makeFlow([new THREE.Vector3(-5.6, LAYOUT.scheduler.y, 0), new THREE.Vector3(LAYOUT.latent.x - 3.4, LAYOUT.scheduler.y, 0), new THREE.Vector3(LAYOUT.latent.x - 3.4, -2.2, 0)], { radius: 0.09 }),
    makeFlow([new THREE.Vector3(LAYOUT.latent.x + 3.2, LAYOUT.latent.y, 0), new THREE.Vector3(-19.5, LAYOUT.latent.y, 0)], { radius: 0.09 }),
    makeFlow([new THREE.Vector3(18.8, 1, 0), new THREE.Vector3(LAYOUT.eps.x - 2.5, 1, 0)], { radius: 0.09 }),
    makeFlow([new THREE.Vector3(-51.7, 0, 0), new THREE.Vector3(-48.9, 0, 0)], { radius: 0.09 }),
    makeFlow([new THREE.Vector3(-41.3, 0, 0), new THREE.Vector3(-38.2, 0, 0)], { radius: 0.09 }),
    makeFlow([new THREE.Vector3(LAYOUT.eps.x + 2.6, LAYOUT.scheduler.y + 0.6, 0), new THREE.Vector3(LAYOUT.vae.x - 4.2, LAYOUT.scheduler.y + 0.6, 0), new THREE.Vector3(LAYOUT.vae.x - 4.2, -1.8, 0)], { radius: 0.07, color: "#7dd3fc" }),
    makeFlow([new THREE.Vector3(LAYOUT.vae.x + 5, 0, 0), new THREE.Vector3(LAYOUT.image.x - 4.6, 0, 0)], { radius: 0.09 }),
  ];
  flows.forEach((f) => group.add(f));
  placeLabel(group, "z_t−1: di nuovo, con t più piccolo", -17, LAYOUT.scheduler.y - 1.2, 0, { size: 0.68, color: "#a5f3fc" });
  placeLabel(group, "alla fine: z_0", LAYOUT.vae.x - 4.5, LAYOUT.scheduler.y + 1.8, 0, { size: 0.68, color: "#7dd3fc" });

  const vae = buildVAE();
  group.add(vae.group);
  registry.set("vae", vae.group);

  const imageTex = new RGBTexture(256);
  const image = texturedPlane(imageTex.texture, 8.4, 8.4);
  image.position.copy(LAYOUT.image);
  const imageGroup = new THREE.Group();
  imageGroup.add(image);
  placeLabel(imageGroup, "Immagine", LAYOUT.image.x, 6.6, 0, { size: 0.95, sub: "512 × 512 × 3" });
  group.add(imageGroup);
  registry.set("image", imageGroup);

  return {
    group,
    registry,
    latentTex,
    epsTex,
    tTex,
    imageTex,
    unet,
    dit,
    flows,
    /** Aggiorna la striscia dell'embedding sinusoidale del passo t. */
    setTimestep(t) {
      tTex.set(timestepEmbedding(tEmb, t), 0);
    },
    /** "unet" oppure "dit" */
    setDenoiser(kind) {
      unet.group.visible = kind === "unet";
      dit.group.visible = kind === "dit";
    },
  };
}
