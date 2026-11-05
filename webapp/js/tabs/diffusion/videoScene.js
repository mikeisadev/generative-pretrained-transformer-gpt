/**
 * Livello "Dettaglio" della parte video (coordinate locali; il gruppo sta sotto la pipeline).
 *
 *   16 frame in pixel → VAE 3D causale (in HunyuanVideo: 4× nel tempo, 8× nello spazio)
 *   → cubo di 4 latenti 8 × 8 → patch spazio-temporali (token) → attention spazio/tempo
 *   → frame in uscita, più un confronto "un frame alla volta" contro "tutti insieme".
 *
 * Le immagini sono generate nel browser (imagery.js); le parti "latenti" e il confronto
 * sono una simulazione didattica.
 */
import * as THREE from "three";
import { renderScene, sceneVariant, SimulatedVAE } from "./imagery.js";
import { mulberry32 } from "./math.js";
import { PALETTE, instancedBoxes, lineSegments, placeLabel, texturedPlane } from "./shapes.js";
import { DynamicLabel, RGBTexture } from "./textures.js";

export const FRAMES = 16;
export const FRAME_SIZE = 64;
export const LATENT_FRAMES = 4; // 16 frame / 4 (compressione temporale)
export const LATENT_SIZE = 8; // 64 / 8 (compressione spaziale)
export const PATCH = 2; // patch 2 × 2 nello spazio, 1 nel tempo → 4 × 4 token per latente
const GRID = LATENT_SIZE / PATCH;
export const TOKENS = LATENT_FRAMES * GRID * GRID;

const PIX = { x: -44, y: 1, w: 6, dx: 0.42, dy: 0.26, dz: -0.75 };
const LAT = { x: -13, y: 1, w: 5, dx: 1.7, dy: 0.95, dz: -2.4 };

/** Centro della cella (r, c) del latente f, in coordinate del gruppo. */
export function tokenPosition(f, r, c, out = new THREE.Vector3()) {
  const cell = LAT.w / GRID;
  return out.set(
    LAT.x + (f - (LATENT_FRAMES - 1) / 2) * LAT.dx - LAT.w / 2 + cell * (c + 0.5),
    LAT.y + (f - (LATENT_FRAMES - 1) / 2) * LAT.dy + LAT.w / 2 - cell * (r + 0.5),
    (f - (LATENT_FRAMES - 1) / 2) * LAT.dz + 0.25,
  );
}

/** Posizione del token i nella sequenza (riga sotto il cubo). */
export function sequencePosition(i, out = new THREE.Vector3()) {
  return out.set(-24 + i * 0.36 + Math.floor(i / (GRID * GRID)) * 0.6, -9.2, 0);
}

/** Frame "falsi colori" di un latente simulato (canali 0, 1, 3 → R, G, B). */
function latentFalseColor(z, s) {
  const n = s * s;
  const rgb = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    rgb[k * 3] = Math.tanh(z[k] * 0.7);
    rgb[k * 3 + 1] = Math.tanh(z[n + k] * 0.7);
    rgb[k * 3 + 2] = Math.tanh(z[3 * n + k] * 0.7);
  }
  return rgb;
}

export function buildVideoScene() {
  const group = new THREE.Group();
  const registry = new Map();

  // --- frame in pixel: un video "vero" generato nel browser (il fascio del faro ruota, il sole scende)
  const frames = [];
  const pixelGroup = new THREE.Group();
  for (let k = 0; k < FRAMES; k++) {
    const data = renderScene(new Float32Array(FRAME_SIZE * FRAME_SIZE * 3), FRAME_SIZE, { time: k / (FRAMES - 1) });
    const tex = new RGBTexture(FRAME_SIZE).set(data);
    const mesh = texturedPlane(tex.texture, PIX.w, PIX.w);
    const o = k - (FRAMES - 1) / 2;
    mesh.position.set(PIX.x + o * PIX.dx, PIX.y + o * PIX.dy, o * PIX.dz);
    pixelGroup.add(mesh);
    frames.push({ data, tex, mesh });
  }
  group.add(pixelGroup);
  placeLabel(pixelGroup, "16 frame (pixel)", PIX.x, 8.6, 0, { size: 0.5, sub: "larghezza × altezza × tempo" });
  registry.set("videoFrames", pixelGroup);

  // --- VAE 3D causale: tre stadi che riducono spazio e tempo
  const vaeItems = [0, 1, 2].map((i) => ({ x: -29.5 + i * 2.2, y: 1, z: 0, sx: 1.2, sy: 6 - i * 1.5, sz: 5 - i * 1.2, color: PALETTE.vae }));
  const vaeGroup = new THREE.Group();
  vaeGroup.add(instancedBoxes(vaeItems));
  placeLabel(vaeGroup, "VAE 3D causale", -27.3, 7, 0, { size: 0.5, color: "#7dd3fc", sub: "HunyuanVideo: 4× tempo, 8× spazio" });
  group.add(vaeGroup);
  registry.set("videoVae", vaeGroup);

  // --- cubo di latenti
  const vae64 = new SimulatedVAE(frames[0].data, FRAME_SIZE);
  const latentGroup = new THREE.Group();
  const latents = [];
  for (let f = 0; f < LATENT_FRAMES; f++) {
    // il latente f riassume 4 frame consecutivi (compressione temporale 4×)
    const avg = new Float32Array(FRAME_SIZE * FRAME_SIZE * 3);
    for (let k = 0; k < 4; k++) frames[f * 4 + k].data.forEach((v, i) => (avg[i] += v / 4));
    const z = vae64.encode(avg);
    const tex = new RGBTexture(LATENT_SIZE, { nearest: true }).set(latentFalseColor(z, LATENT_SIZE));
    const mesh = texturedPlane(tex.texture, LAT.w, LAT.w);
    const o = f - (LATENT_FRAMES - 1) / 2;
    mesh.position.set(LAT.x + o * LAT.dx, LAT.y + o * LAT.dy, o * LAT.dz);
    latentGroup.add(mesh);
    latents.push({ z, tex, mesh });
  }
  // griglia delle patch (2 × 2 celle latenti ciascuna)
  const gridPairs = [];
  for (let f = 0; f < LATENT_FRAMES; f++) {
    const o = f - (LATENT_FRAMES - 1) / 2;
    const cx = LAT.x + o * LAT.dx;
    const cy = LAT.y + o * LAT.dy;
    const z = o * LAT.dz + 0.02;
    for (let g = 1; g < GRID; g++) {
      const t = -LAT.w / 2 + (LAT.w / GRID) * g;
      gridPairs.push([new THREE.Vector3(cx + t, cy - LAT.w / 2, z), new THREE.Vector3(cx + t, cy + LAT.w / 2, z)]);
      gridPairs.push([new THREE.Vector3(cx - LAT.w / 2, cy + t, z), new THREE.Vector3(cx + LAT.w / 2, cy + t, z)]);
    }
  }
  const patchGrid = lineSegments(gridPairs, 0xe2e8f0, { opacity: 0.7 });
  patchGrid.visible = false; // compare con il passo "patch spazio-temporali"
  latentGroup.add(patchGrid);
  group.add(latentGroup);
  placeLabel(latentGroup, "Cubo di latenti", LAT.x, 8.6, 0, { size: 0.5, sub: "4 latenti × 8 × 8 (colori falsi)" });
  registry.set("videoLatent", latentGroup);

  // --- token: uno per patch spazio-temporale (InstancedMesh)
  const tokenItems = [];
  const p = new THREE.Vector3();
  for (let f = 0; f < LATENT_FRAMES; f++)
    for (let r = 0; r < GRID; r++)
      for (let c = 0; c < GRID; c++) {
        tokenPosition(f, r, c, p);
        tokenItems.push({ x: p.x, y: p.y, z: p.z, sx: 0.5, sy: 0.5, sz: 0.5, color: [0x22d3ee, 0x38bdf8, 0x818cf8, 0xc084fc][f], f, r, c });
      }
  const tokens = instancedBoxes(tokenItems);
  tokens.visible = false;
  group.add(tokens);
  const seqLabel = placeLabel(group, "sequenza di 64 token (4 latenti × 16 patch)", -12, -10.6, 0, { size: 0.4, color: "#67e8f9" });
  seqLabel.visible = false;

  // linee di attention da un token "query" (max TOKENS − 1 segmenti, una sola draw call)
  const attention = lineSegments([], 0xfde68a, { opacity: 0.85, maxSegments: TOKENS });
  group.add(attention);

  // --- denoiser video
  const blockItems = [
    { x: 6, y: -2.2, z: 0, sx: 2.2, sy: 2.4, sz: 3.6, color: PALETTE.attn, kind: "spatial" },
    { x: 6, y: 0.6, z: 0, sx: 2.2, sy: 2.4, sz: 3.6, color: 0x22d3ee, kind: "temporal" },
    { x: 6, y: 3.4, z: 0, sx: 2.2, sy: 2.4, sz: 3.6, color: PALETTE.resnet, kind: "mlp" },
  ];
  const denoiser = new THREE.Group();
  const blockMesh = instancedBoxes(blockItems);
  denoiser.add(blockMesh);
  placeLabel(denoiser, "Transformer video", 6, 6.8, 0, { size: 0.5, color: "#fde68a", sub: "× N blocchi, ripetuto per T passi" });
  placeLabel(denoiser, "attention spaziale", 9.6, -2.2, 0, { size: 0.32, color: "#c4b5fd" });
  placeLabel(denoiser, "attention temporale", 9.8, 0.6, 0, { size: 0.32, color: "#67e8f9" });
  placeLabel(denoiser, "MLP", 8.4, 3.4, 0, { size: 0.32, color: "#fcd34d" });
  group.add(denoiser);
  registry.set("videoDenoiser", denoiser);

  // --- uscita: lettore dei frame e confronto
  const playerTex = new RGBTexture(FRAME_SIZE).set(frames[0].data);
  const player = texturedPlane(playerTex.texture, 8, 8);
  player.position.set(22, 1, 0);
  const outGroup = new THREE.Group();
  outGroup.add(player);
  const playerLabel = new DynamicLabel("Video generato\nframe 1/16", { height: 0.5 });
  playerLabel.sprite.position.set(22, 6.6, 0);
  outGroup.add(playerLabel.sprite);
  group.add(outGroup);
  registry.set("videoOut", outGroup);

  // confronto: 8 frame "uno alla volta" (ogni frame con un suo seme) e 8 "insieme"
  const compare = new THREE.Group();
  const indep = [];
  const joint = [];
  for (let k = 0; k < 8; k++) {
    const time = (k * 2) / (FRAMES - 1);
    const a = renderScene(new Float32Array(FRAME_SIZE * FRAME_SIZE * 3), FRAME_SIZE, { time, variant: sceneVariant(mulberry32(100 + k)) });
    const b = frames[k * 2].data;
    const ta = new RGBTexture(FRAME_SIZE).set(a);
    const tb = new RGBTexture(FRAME_SIZE).set(b);
    const ma = texturedPlane(ta.texture, 3.2, 3.2);
    const mb = texturedPlane(tb.texture, 3.2, 3.2);
    ma.position.set(31 + k * 3.5, 3.4, 0);
    mb.position.set(31 + k * 3.5, -2.2, 0);
    compare.add(ma, mb);
    indep.push({ data: a, tex: ta });
    joint.push({ data: b, tex: tb });
  }
  placeLabel(compare, "Un frame alla volta, ognuno per conto suo", 43.3, 6, 0, { size: 0.42, color: "#fca5a5" });
  placeLabel(compare, "Tutti i frame insieme", 43.3, 0.4, 0, { size: 0.42, color: "#86efac" });
  placeLabel(compare, "simulazione didattica", 43.3, -4.7, 0, { size: 0.32, color: "#94a3b8" });
  group.add(compare);
  registry.set("videoCompare", compare);

  placeLabel(group, "Video: tutti i frame nascono insieme", 0, 13, 0, { size: 0.9, color: "#e2e8f0", sub: "latenti spazio-temporali, patch e attention nello spazio e nel tempo" });

  // --- metodi per le azioni (nessuna allocazione nei metodi chiamati a ogni frame)
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const positions = attention.geometry.attributes.position;

  return {
    group,
    registry,
    frames,
    latents,
    tokens,
    tokenItems,
    patchGrid,
    seqLabel,
    attention,
    blockMesh,
    blockItems,
    player: { mesh: player, tex: playerTex, label: playerLabel },
    indep,
    joint,
    /** p = 0: token nel cubo; p = 1: token in fila come una sequenza. */
    setTokenLayout(p, scale = 1) {
      tokenItems.forEach((it, i) => {
        sequencePosition(i, b);
        a.set(it.x, it.y, it.z).lerp(b, p);
        s.setScalar(0.5 * scale);
        m.compose(a, q, s);
        tokens.setMatrixAt(i, m);
      });
      tokens.instanceMatrix.needsUpdate = true;
    },
    /**
     * Linee di attention dal token `query` (indice) secondo `mode`:
     * "spatial" stesso latente, "temporal" stessa posizione negli altri latenti, "full" tutti.
     * `reveal` (0…1) mostra le linee progressivamente.
     */
    setAttention(mode, query, reveal = 1) {
      const qi = tokenItems[query];
      let n = 0;
      tokenItems.forEach((it, i) => {
        if (i === query || !mode) return;
        const ok = mode === "full" || (mode === "spatial" && it.f === qi.f) || (mode === "temporal" && it.r === qi.r && it.c === qi.c);
        if (!ok) return;
        positions.setXYZ(n * 2, qi.x, qi.y, qi.z);
        positions.setXYZ(n * 2 + 1, it.x, it.y, it.z);
        n++;
      });
      positions.needsUpdate = true;
      attention.geometry.setDrawRange(0, Math.floor(n * reveal) * 2);
      return n;
    },
  };
}
