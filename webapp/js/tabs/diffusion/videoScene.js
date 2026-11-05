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
import { avgPool, renderScene, sceneVariant } from "./imagery.js";
import { mulberry32 } from "./math.js";
import { PALETTE, instancedBoxes, lineSegments, placeLabel, recolor, texturedPlane } from "./shapes.js";
import { DynamicLabel, RGBTexture } from "./textures.js";

export const FRAMES = 16;
export const FRAME_SIZE = 64;
export const LATENT_FRAMES = 4; // 16 frame / 4 (compressione temporale)
export const LATENT_SIZE = 8; // 64 / 8 (compressione spaziale)
export const PATCH = 2; // patch 2 × 2 nello spazio, 1 nel tempo → 4 × 4 token per latente
const GRID = LATENT_SIZE / PATCH;
export const TOKENS = LATENT_FRAMES * GRID * GRID;

const PIX = { x: -44, y: 1, w: 6, dx: 0.42, dy: 0.26, dz: -0.75 };
const LAT = { x: -12.5, y: 1, w: 5, dx: 5.6, dy: 0.5, dz: -1.2 };

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
  placeLabel(pixelGroup, "16 frame", PIX.x, 9.4, 0, { size: 0.95, sub: "larghezza × altezza × tempo" });
  registry.set("videoFrames", pixelGroup);

  // --- VAE 3D causale: tre stadi che riducono spazio e tempo
  const vaeItems = [0, 1, 2].map((i) => ({ x: -29.5 + i * 2.2, y: 1, z: 0, sx: 1.2, sy: 6 - i * 1.5, sz: 5 - i * 1.2, color: PALETTE.vae }));
  const vaeGroup = new THREE.Group();
  vaeGroup.add(instancedBoxes(vaeItems));
  placeLabel(vaeGroup, "VAE 3D causale", -27.3, 7.4, 0, { size: 0.95, color: "#7dd3fc", sub: "4× tempo, 8× spazio" });
  group.add(vaeGroup);
  registry.set("videoVae", vaeGroup);

  // --- cubo di latenti (simulazione: ogni latente è la media di 4 frame ridotta 8 volte,
  // mostrata con i suoi colori approssimati; un latente vero ha 16 canali non leggibili a occhio)
  const latentGroup = new THREE.Group();
  const latents = [];
  for (let f = 0; f < LATENT_FRAMES; f++) {
    // il latente f riassume 4 frame consecutivi (compressione temporale 4×)
    const avg = new Float32Array(FRAME_SIZE * FRAME_SIZE * 3);
    for (let k = 0; k < 4; k++) frames[f * 4 + k].data.forEach((v, i) => (avg[i] += v / 4));
    const small = avgPool(avg, FRAME_SIZE, FRAME_SIZE / LATENT_SIZE);
    const tex = new RGBTexture(LATENT_SIZE, { nearest: true }).set(small);
    const mesh = texturedPlane(tex.texture, LAT.w, LAT.w);
    const o = f - (LATENT_FRAMES - 1) / 2;
    mesh.position.set(LAT.x + o * LAT.dx, LAT.y + o * LAT.dy, o * LAT.dz);
    latentGroup.add(mesh);
    latents.push({ tex, mesh });
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
  placeLabel(latentGroup, "Cubo di latenti", LAT.x, 7.6, 0, { size: 0.95, sub: "4 istanti × 8 × 8 (simulazione)" });
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
  const seqLabel = placeLabel(group, "sequenza di 64 token (4 latenti × 16 patch)", -12, -11, 0, { size: 0.8, color: "#67e8f9" });
  seqLabel.visible = false;

  // linee di attention da un token "query" (max TOKENS − 1 segmenti, una sola draw call)
  const attention = lineSegments([], 0xfde68a, { opacity: 1, maxSegments: TOKENS });
  attention.material.depthTest = false; // le linee restano visibili anche dietro ai piani
  attention.renderOrder = 20;
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
  placeLabel(denoiser, "Transformer", 6, 7.4, 0, { size: 0.95, color: "#fde68a", sub: "× N blocchi, T passi" });
  placeLabel(denoiser, "spaziale", 9.3, -2.2, 0, { size: 0.62, color: "#c4b5fd" });
  placeLabel(denoiser, "temporale", 9.3, 0.6, 0, { size: 0.62, color: "#67e8f9" });
  placeLabel(denoiser, "MLP", 8.6, 3.4, 0, { size: 0.62, color: "#fcd34d" });
  group.add(denoiser);
  registry.set("videoDenoiser", denoiser);

  // --- uscita: lettore dei frame e confronto
  const playerTex = new RGBTexture(FRAME_SIZE).set(frames[0].data);
  const player = texturedPlane(playerTex.texture, 8, 8);
  player.position.set(22, 1, 0);
  const outGroup = new THREE.Group();
  outGroup.add(player);
  const playerLabel = new DynamicLabel("Video generato\nframe 1/16", { height: 0.85 });
  playerLabel.sprite.position.set(22, 6.9, 0);
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
  placeLabel(compare, "Un frame alla volta, ognuno per conto suo", 43.3, 6.1, 0, { size: 0.8, color: "#fca5a5" });
  placeLabel(compare, "Tutti i frame insieme", 43.3, 0.5, 0, { size: 0.8, color: "#86efac" });
  placeLabel(compare, "simulazione didattica", 43.3, -4.8, 0, { size: 0.62, color: "#94a3b8" });
  group.add(compare);
  registry.set("videoCompare", compare);

  placeLabel(group, "Video: tutti i frame nascono insieme", 2, 15, 0, { size: 1.9, color: "#e2e8f0" });

  // --- metodi per le azioni (nessuna allocazione nei metodi chiamati a ogni frame)
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const positions = attention.geometry.attributes.position;

  /** Riquadro (coordinate del mondo) di una zona della scena video, per la regia. */
  const region = (x1, x2, y1, y2) => {
    group.updateWorldMatrix(true, false);
    const y0 = group.position.y;
    return new THREE.Box3(new THREE.Vector3(x1, y0 + y1, 0), new THREE.Vector3(x2, y0 + y2, 0));
  };

  return {
    group,
    registry,
    regions: {
      intro: () => region(-50, 30, -6, 15),
      cube: () => region(-26, 2, -12, 9),
      tokens: () => region(-26, 1, -6, 9),
    },
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
      // query in bianco, token "guardati" nel colore delle linee, gli altri spenti
      const hex = attention.material.color.getHex();
      recolor(tokens, (i, it) => {
        if (!mode) return it.color;
        if (i === query) return 0xffffff;
        const seen = mode === "full" || (mode === "spatial" && it.f === qi.f) || (mode === "temporal" && it.r === qi.r && it.c === qi.c);
        return seen ? hex : 0x334155;
      });
      return n;
    },
  };
}
