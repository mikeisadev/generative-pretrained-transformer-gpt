/**
 * Livello "Dettaglio", prima metà: forma d'onda (con lente sui campioni e sui frame),
 * spettrogramma 3D (STFT → Mel → log-Mel) e RVQ (nuvole dei codebook, griglia dei token).
 *
 * Ogni builder restituisce { group, ... } con metodi che le azioni chiamano per animare.
 * Tutti i buffer sono allocati una volta: gli aggiornamenti per frame riscrivono array esistenti.
 */
import * as THREE from "three";
import { label } from "../../core/visuals.js";
import { fmt } from "../../core/dom.js";
import { CODEC } from "./codec.js";
import { envelope, hannWindow, hzToMel } from "./dsp.js";
import { SLOT } from "./layout3d.js";
import { canvasPanel, dynamicLine, placeLabel, rampColor, tokenColor, tokenCss, tokenGrid, waveformMesh } from "./widgets.js";

const smooth = (t) => t * t * (3 - 2 * t);
/** Scrive un punto 3D in un Float32Array senza creare array temporanei (si usa dentro le animazioni). */
function put(arr, i, x, y, z) {
  arr[i * 3] = x;
  arr[i * 3 + 1] = y;
  arr[i * 3 + 2] = z;
}

/* ======================================================================== forma d'onda === */

export function buildAudio(lab, initialStyle) {
  const { w, h } = SLOT.audio;
  const group = new THREE.Group();
  group.position.set(SLOT.audio.x, SLOT.audio.y, 0);
  const amp = h - 1; // ±0,8 di picco → circa ±2 unità
  const wave = waveformMesh(envelope(lab.songs[initialStyle].samples, 700), { width: w, height: amp / 0.8 });
  const axis = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-w / 2, 0, 0), new THREE.Vector3(w / 2, 0, 0)]),
    new THREE.LineBasicMaterial({ color: 0x334155, transparent: true }),
  );
  group.add(axis, wave);
  const title = label("Forma d'onda", { size: 0.55, sub: `${fmt(lab.sampleRate)} campioni al secondo` });
  title.position.set(0, h / 2 + 0.9, 0);
  group.add(title);
  const duration = lab.songs[initialStyle].duration;
  placeLabel(group, "0 s", -w / 2, -h / 2 - 0.25, 0, { size: 0.3, color: "#94a3b8" });
  placeLabel(group, `${fmt(duration, 1)} s`, w / 2, -h / 2 - 0.25, 0, { size: 0.3, color: "#94a3b8" });
  placeLabel(group, "tempo →", 0, -h / 2 - 0.25, 0, { size: 0.3, color: "#94a3b8" });

  const playhead = new THREE.Mesh(new THREE.BoxGeometry(0.05, h - 0.4, 0.05), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 }));
  playhead.visible = false;
  playhead.userData.noBounds = true;
  const frameWidth = (CODEC.nFft / lab.songs[initialStyle].samples.length) * w;
  const marker = new THREE.Mesh(new THREE.BoxGeometry(frameWidth, h - 0.4, 0.08), new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.35, depthWrite: false }));
  marker.visible = false;
  marker.userData.noBounds = true;
  group.add(playhead, marker);

  // --- lente: ingrandimento di un tratto del segnale, sopra la forma d'onda
  const lens = new THREE.Group();
  const LW = 12;
  const LH = 3.6;
  lens.position.set(0, h / 2 + 4.4, 0);
  const lensBg = new THREE.Mesh(new THREE.PlaneGeometry(LW + 0.6, LH + 0.6), new THREE.MeshBasicMaterial({ color: 0x0b1220, transparent: true, opacity: 0.92, depthWrite: false }));
  lensBg.position.z = -0.05;
  lensBg.userData.isFrame = true;
  const lensEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(LW + 0.6, LH + 0.6)), new THREE.LineBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.8 }));
  const raw = dynamicLine(CODEC.nFft + 2, { color: 0x67e8f9, opacity: 0.9 });
  const hann = dynamicLine(CODEC.nFft + 2, { color: 0xfbbf24 });
  const windowed = dynamicLine(CODEC.nFft + 2, { color: 0xf472b6 });
  const DOTS = 48;
  const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.07, 10, 8), new THREE.MeshBasicMaterial({ color: 0x67e8f9, transparent: true }), DOTS);
  const stemPos = new Float32Array(DOTS * 6);
  const stemsGeo = new THREE.BufferGeometry();
  stemsGeo.setAttribute("position", new THREE.BufferAttribute(stemPos, 3));
  const stems = new THREE.LineSegments(stemsGeo, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.6 }));
  stems.frustumCulled = false;
  dots.frustumCulled = false;
  const lensTitle = { samples: label("48 campioni = 3 millisecondi", { size: 0.38, sub: "il computer vede solo questi punti" }), frame: label(`un frame: ${CODEC.nFft} campioni (64 ms)`, { size: 0.38, sub: "segnale × finestra di Hann" }) };
  for (const t of Object.values(lensTitle)) {
    t.position.set(0, LH / 2 + 0.85, 0);
    lens.add(t);
  }
  const legend = label("segnale   ·   finestra di Hann   ·   segnale × finestra", { size: 0.26, color: "#cbd5e1" });
  legend.position.set(0, -LH / 2 - 0.55, 0);
  lens.add(lensBg, lensEdge, raw.line, hann.line, windowed.line, stems, dots, legend);
  // linee che collegano il tratto ingrandito alla lente
  const linkPos = new Float32Array(12);
  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute("position", new THREE.BufferAttribute(linkPos, 3));
  const links = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.5 }));
  links.frustumCulled = false;
  links.userData.noBounds = true;
  group.add(lens, links);
  lens.visible = false;
  links.visible = false;
  lens.userData.members = [lensBg];

  let samples = lab.songs[initialStyle].samples;
  const window = hannWindow(CODEC.nFft);
  const matrix = new THREE.Matrix4();
  const xOf = (i) => -w / 2 + (i / samples.length) * w;

  /** Disegna nella lente `count` campioni da `start`; mode "samples" (punti) o "frame" (finestra). */
  function showLens(mode, start = 0, progress = 1) {
    lens.visible = mode !== null;
    links.visible = mode !== null;
    marker.visible = mode !== null;
    if (!mode) return;
    for (const [k, t] of Object.entries(lensTitle)) t.visible = k === mode;
    legend.visible = mode === "frame";
    const count = mode === "samples" ? DOTS : CODEC.nFft;
    start = Math.max(0, Math.min(samples.length - count, Math.round(start)));
    const yScale = (LH / 2 - 0.15) / 0.8;
    const x = (i) => -LW / 2 + (i / (count - 1)) * LW;
    const shown = Math.max(2, Math.round(count * progress));
    for (let i = 0; i < count; i++) {
      const v = samples[start + i];
      put(raw.positions, i, x(i), v * yScale, 0);
      if (mode === "frame") {
        put(hann.positions, i, x(i), window[i] * (LH / 2 - 0.15), 0.01);
        put(windowed.positions, i, x(i), v * window[i] * yScale, 0.02);
      }
    }
    raw.commit(mode === "samples" ? count : shown);
    hann.commit(mode === "frame" ? shown : 0);
    windowed.commit(mode === "frame" ? shown : 0);
    dots.count = mode === "samples" ? Math.min(DOTS, shown) : 0;
    stems.visible = mode === "samples";
    if (mode === "samples") {
      for (let i = 0; i < DOTS; i++) {
        const y = samples[start + i] * yScale;
        matrix.makeTranslation(x(i), y, 0.02);
        dots.setMatrixAt(i, matrix);
        put(stemPos, 2 * i, x(i), 0, 0);
        put(stemPos, 2 * i + 1, x(i), i < shown ? y : 0, 0);
      }
      dots.instanceMatrix.needsUpdate = true;
      stemsGeo.attributes.position.needsUpdate = true;
    }
    // marcatore sul segnale intero e "raggi" verso la lente
    const a = xOf(start);
    const b = xOf(start + count);
    marker.scale.x = Math.max(0.15, (b - a) / frameWidth);
    marker.position.x = (a + b) / 2;
    const top = lens.position.y - (LH + 0.6) / 2;
    put(linkPos, 0, a, h / 2 - 0.4, 0);
    put(linkPos, 1, -LW / 2 - 0.3, top, 0);
    put(linkPos, 2, b, h / 2 - 0.4, 0);
    put(linkPos, 3, LW / 2 + 0.3, top, 0);
    linkGeo.attributes.position.needsUpdate = true;
  }

  return {
    group,
    lens,
    duration,
    setStyle(id) {
      samples = lab.songs[id].samples;
      wave.userData.setEnvelope(envelope(samples, 700));
    },
    /** t in secondi (null = nascosta). */
    setPlayhead(t) {
      playhead.visible = t !== null && t !== undefined;
      if (playhead.visible) playhead.position.x = -w / 2 + (t / duration) * w;
    },
    showLens,
    xOfSample: xOf,
  };
}

/* ====================================================================== spettrogramma 3D === */

const FREQ_TICKS = [0, 500, 1000, 2000, 4000, 8000];

export function buildEncoder(lab, initialStyle) {
  const { w, h } = SLOT.encoder;
  const H = h - 1; // altezza dell'asse delle frequenze
  const Z = 1.6; // altezza del rilievo
  const group = new THREE.Group();
  group.position.set(SLOT.encoder.x, SLOT.encoder.y, 0);
  const d0 = lab.display[initialStyle];
  const T = d0.frames;
  const R = d0.rows;
  const melMax = hzToMel(lab.sampleRate / 2);
  const yLin = Float32Array.from(d0.rowHz, (f) => -H / 2 + (f / (lab.sampleRate / 2)) * H);
  const yMel = Float32Array.from({ length: R }, (_, r) => -H / 2 + (r / (R - 1)) * H);

  const positions = new Float32Array(T * R * 3);
  const colors = new Float32Array(T * R * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  // indici in ordine di TEMPO: drawRange rivela lo spettrogramma frame dopo frame
  const index = new Uint32Array((T - 1) * (R - 1) * 6);
  let o = 0;
  for (let t = 0; t < T - 1; t++) {
    for (let r = 0; r < R - 1; r++) {
      const a = t * R + r;
      const b = (t + 1) * R + r;
      index.set([a, b, a + 1, b, b + 1, a + 1], o);
      o += 6;
    }
  }
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  for (let t = 0; t < T; t++) for (let r = 0; r < R; r++) positions[(t * R + r) * 3] = -w / 2 + (t / (T - 1)) * w;
  const surface = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true }));
  const base = new THREE.Mesh(new THREE.PlaneGeometry(w + 0.4, H + 0.4), new THREE.MeshBasicMaterial({ color: 0x0b1220, transparent: true, opacity: 0.9 }));
  base.position.z = -0.03;
  base.userData.isFrame = true;
  // il "foglio" è inclinato all'indietro: il rilievo (intensità) si legge come un paesaggio
  const plot = new THREE.Group();
  plot.rotation.x = -0.5;
  group.add(plot);
  plot.add(base, surface);

  const slice = new THREE.Mesh(new THREE.BoxGeometry(w / T + 0.03, H + 0.3, Z + 0.3), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25, depthWrite: false }));
  slice.position.z = Z / 2;
  slice.visible = false;
  slice.userData.noBounds = true;
  plot.add(slice);

  const titles = [
    label("Spettrogramma (STFT)", { size: 0.5, sub: "ampiezza · frequenze in Hz (asse lineare)" }),
    label("Asse in scala Mel", { size: 0.5, sub: "le basse frequenze si allargano" }),
    label("Banco di filtri Mel", { size: 0.5, sub: "energia sommata in 64 bande" }),
    label("log-Mel", { size: 0.5, sub: "decibel: come percepiamo il volume" }),
  ];
  for (const t of titles) {
    t.position.set(0, H / 2 + 1.1, 0);
    group.add(t);
  }
  const ticks = FREQ_TICKS.map((f) => {
    const tag = label(f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`, { size: 0.3, color: "#cbd5e1" });
    plot.add(tag);
    return { f, tag };
  });
  placeLabel(plot, "tempo →", 0, -H / 2 - 0.55, 0, { size: 0.32, color: "#94a3b8" });

  let data = d0;
  let stage = 0;
  const color = new THREE.Color();

  /** stage: 0 STFT lineare → 1 asse Mel → 2 filtri Mel → 3 log-Mel (valori intermedi = morph). */
  function setStage(s) {
    stage = Math.max(0, Math.min(3, s));
    const axis = smooth(Math.min(1, stage));
    const fMel = smooth(Math.max(0, Math.min(1, stage - 1)));
    const fLog = smooth(Math.max(0, Math.min(1, stage - 2)));
    const { linear, mel, logMel } = data;
    for (let t = 0; t < T; t++) {
      for (let r = 0; r < R; r++) {
        const i = t * R + r;
        const v1 = linear[i] + (mel[i] - linear[i]) * fMel;
        const v = v1 + (logMel[i] - v1) * fLog;
        const shown = fLog > 0 ? v : Math.pow(v, 0.6); // in ampiezza i dettagli deboli si vedono appena: un po' di gamma
        positions[i * 3 + 1] = yLin[r] + (yMel[r] - yLin[r]) * axis;
        positions[i * 3 + 2] = shown * Z;
        rampColor(shown, color);
        color.setRGB(color.r, color.g, color.b, THREE.SRGBColorSpace); // la rampa è in sRGB, i vertex color sono lineari
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      }
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
    const active = Math.min(3, Math.round(stage));
    titles.forEach((t, i) => (t.visible = i === active));
    for (const { f, tag } of ticks) {
      const lin = -H / 2 + (f / (lab.sampleRate / 2)) * H;
      const m = -H / 2 + (hzToMel(f) / melMax) * H;
      tag.position.set(-w / 2 - 0.9, lin + (m - lin) * axis, 0);
    }
  }
  setStage(0);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return {
    group,
    frames: T,
    setStyle(id) {
      data = lab.display[id];
      setStage(stage);
    },
    setStage,
    get stage() {
      return stage;
    },
    /** Mostra solo i primi n frame (animazione "FFT frame per frame"). */
    reveal(n) {
      const cols = Math.max(0, Math.min(T - 1, Math.round(n) - 1));
      geometry.setDrawRange(0, cols * (R - 1) * 6);
    },
    setSlice(t) {
      slice.visible = t !== null && t !== undefined;
      if (slice.visible) slice.position.x = -w / 2 + (t / (T - 1)) * w;
    },
  };
}

/* ============================================================================= RVQ === */

export function buildRvq(lab, initialStyle, styleIds) {
  const { w } = SLOT.rvq;
  const T = lab.frames;
  const Q = CODEC.used;
  const group = new THREE.Group();
  group.position.set(SLOT.rvq.x, SLOT.rvq.y, 0);

  // --- nuvole dei codebook: centroidi (colorati come i loro token) e frame (punti grigi)
  const P = 1.45; // mezzo lato del riquadro di una nuvola
  const clouds = [];
  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();
  for (let k = 0; k < Q; k++) {
    const c = lab.clouds[k];
    const holder = new THREE.Group();
    holder.position.set(-w / 2 + 1.75 + k * 3.5, 3.6, 0);
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(2 * P + 0.2, 2 * P + 0.2), new THREE.MeshBasicMaterial({ color: 0x0b1220, transparent: true, opacity: 0.9, depthWrite: false }));
    bg.position.z = -0.04;
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(bg.geometry), new THREE.LineBasicMaterial({ color: 0x334155, transparent: true }));
    const clampP = (v) => Math.max(-P, Math.min(P, v)); // i pochi punti lontanissimi restano sul bordo
    const pts = new Float32Array((c.points.length / 2) * 3);
    for (let i = 0; i < c.points.length / 2; i++) pts.set([clampP((c.points[2 * i] / c.scale) * P), clampP((c.points[2 * i + 1] / c.scale) * P), 0], i * 3);
    const pointGeo = new THREE.BufferGeometry();
    pointGeo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
    const points = new THREE.Points(pointGeo, new THREE.PointsMaterial({ color: 0x64748b, size: 0.05, transparent: true, opacity: 0.8 }));
    const cents = new THREE.InstancedMesh(new THREE.SphereGeometry(0.075, 10, 8), new THREE.MeshBasicMaterial({ transparent: true }), lab.rvq.k);
    for (let i = 0; i < lab.rvq.k; i++) {
      matrix.makeTranslation(clampP((c.centroids[2 * i] / c.scale) * P), clampP((c.centroids[2 * i + 1] / c.scale) * P), 0.03);
      cents.setMatrixAt(i, matrix);
      cents.setColorAt(i, tokenColor(i, color));
    }
    cents.instanceMatrix.needsUpdate = true;
    cents.instanceColor.needsUpdate = true;
    const tag = label(`codebook ${k + 1}`, { size: 0.3, sub: k === 0 ? "il frame" : "il residuo", color: k === 0 ? "#e2e8f0" : "#cbd5e1" });
    tag.position.set(0, P + 0.55, 0);
    holder.add(bg, edge, points, cents, tag);
    group.add(holder);
    clouds.push({ holder, bg, edge, data: c, pts });
  }
  // cursore: il frame corrente in una nuvola e la linea verso il centroide scelto
  const cursor = new THREE.Mesh(new THREE.SphereGeometry(0.12, 14, 10), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true }));
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.15, 0.21, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, side: THREE.DoubleSide }));
  const link = dynamicLine(2, { color: 0xffffff });
  cursor.visible = ring.visible = link.line.visible = false;
  cursor.userData.noBounds = ring.userData.noBounds = true;
  group.add(cursor, ring, link.line);

  // --- griglia dei token: 4 righe × 241 frame
  const grid = tokenGrid(Q, T, { width: w, rowHeight: 0.62, gap: 0.12 });
  grid.group.position.y = -0.55;
  group.add(grid.group);
  for (let k = 0; k < Q; k++) placeLabel(group, `cb ${k + 1}`, -w / 2 - 0.65, -0.55 + grid.rowY(k), 0, { size: 0.26, color: "#94a3b8" });
  const gridTitle = label("Token: 4 numeri interi per ogni frame da 20 ms", { size: 0.32, color: "#e2e8f0" });
  gridTitle.position.set(0, 1.35, 0);
  group.add(gridTitle);
  const frameMark = new THREE.Mesh(new THREE.BoxGeometry(grid.cell * 1.6, 3.2, 0.3), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3, depthWrite: false }));
  frameMark.position.y = -0.55;
  frameMark.visible = false;
  frameMark.userData.noBounds = true;
  group.add(frameMark);

  // --- lente con i numeri: 16 frame a partire da `lensStart`
  const LENS = 16;
  const lensState = { start: 0, cursor: -1, rows: [T, T, T, T] };
  let codes = lab.codes[initialStyle];
  const lens = canvasPanel(w, 2.7, (ctx, cw, ch) => {
    const padL = 120;
    const cellW = (cw - padL - 10) / LENS;
    const cellH = (ch - 44) / Q;
    ctx.font = "600 22px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillStyle = "#94a3b8";
    for (let i = 0; i < LENS; i++) ctx.fillText(`${lensState.start + i}`, padL + (i + 0.5) * cellW, 18);
    ctx.textAlign = "left";
    ctx.fillText("frame", 8, 18);
    for (let k = 0; k < Q; k++) {
      ctx.fillStyle = "#94a3b8";
      ctx.textAlign = "left";
      ctx.fillText(`codebook ${k + 1}`, 8, 40 + (k + 0.5) * cellH);
      for (let i = 0; i < LENS; i++) {
        const t = lensState.start + i;
        const x = padL + i * cellW + 3;
        const y = 40 + k * cellH + 3;
        const shown = t < lensState.rows[k];
        ctx.fillStyle = shown ? tokenCss(codes[k * T + t]) : "#1e293b";
        ctx.globalAlpha = shown ? 0.9 : 0.6;
        ctx.beginPath();
        ctx.roundRect(x, y, cellW - 6, cellH - 6, 6);
        ctx.fill();
        ctx.globalAlpha = 1;
        if (shown) {
          ctx.fillStyle = "#0b1220";
          ctx.textAlign = "center";
          ctx.font = "700 24px 'JetBrains Mono', ui-monospace, monospace";
          ctx.fillText(String(codes[k * T + t]), x + (cellW - 6) / 2, y + (cellH - 6) / 2);
          ctx.font = "600 22px Inter, system-ui, sans-serif";
        }
        if (t === lensState.cursor) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 4;
          ctx.strokeRect(x - 2, 40 + 1, cellW - 2, Q * cellH - 2);
        }
      }
    }
  });
  lens.position.set(0, -3.95, 0);
  group.add(lens);

  const styleOffset = (id) => styleIds.indexOf(id) * T;
  let style = initialStyle;

  /** Cursore sul frame t al livello k (k = null nasconde). */
  function setCursor(k, t) {
    const on = k !== null && k !== undefined;
    cursor.visible = ring.visible = link.line.visible = frameMark.visible = on;
    clouds.forEach((c, i) => c.edge.material.color.set(on && i === k ? 0xfbbf24 : 0x334155));
    if (!on) return;
    const c = clouds[k];
    const idx = styleOffset(style) + t;
    const px = c.pts[idx * 3] + c.holder.position.x;
    const py = c.pts[idx * 3 + 1] + c.holder.position.y;
    const tok = codes[k * T + t];
    const cx = Math.max(-P, Math.min(P, (c.data.centroids[2 * tok] / c.data.scale) * P)) + c.holder.position.x;
    const cy = Math.max(-P, Math.min(P, (c.data.centroids[2 * tok + 1] / c.data.scale) * P)) + c.holder.position.y;
    cursor.position.set(px, py, 0.1);
    ring.position.set(cx, cy, 0.1);
    ring.material.color.copy(tokenColor(tok, color));
    put(link.positions, 0, px, py, 0.08);
    put(link.positions, 1, cx, cy, 0.08);
    link.commit(2);
    frameMark.position.x = grid.frameX(t);
  }

  grid.setCodes(codes, T);
  return {
    group,
    grid,
    clouds: clouds.map((c) => c.holder),
    lens,
    setStyle(id) {
      style = id;
      codes = lab.codes[id];
      grid.setCodes(codes, T);
      lens.userData.redraw();
    },
    /** Quanti frame mostrare per ogni codebook (array di 4) o un numero per tutti. */
    reveal(perRow) {
      const rows = Array.isArray(perRow) ? perRow : [perRow, perRow, perRow, perRow];
      grid.reveal(0, rows);
      lensState.rows = rows.slice();
      lens.userData.redraw();
    },
    setCursor,
    setLens(start, cursorFrame = -1) {
      lensState.start = Math.max(0, Math.min(T - LENS, start));
      lensState.cursor = cursorFrame;
      lens.userData.redraw();
    },
  };
}
