/**
 * Livello "Dettaglio", seconda metà: prompt testuale → T5 → cross-attention → transformer →
 * delay pattern (lente sui passi) → token generati → decoder del codec → audio generato.
 *
 * Il transformer è disegnato come in MusicGen (embedding sommati dei 4 codebook, blocchi,
 * 4 teste lineari); i token che "escono" vengono dal nostro modello a conteggi
 * (simulazione didattica, vedi codec.js → TokenModel).
 */
import * as THREE from "three";
import { addNode, label } from "../../core/visuals.js";
import { CODEC } from "./codec.js";
import { envelope } from "./dsp.js";
import { SLOT } from "./layout3d.js";
import { canvasPanel, placeLabel, slab, tokenCss, tokenGrid, waveformMesh } from "./widgets.js";

const CB_COLORS = [0x38bdf8, 0xa78bfa, 0xf472b6, 0xfbbf24]; // un colore per codebook (righe della scala)
const CB_CSS = ["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24"];

/* ===================================================================== testo e T5 === */

export function buildConditioning(lab, initialStyle) {
  const group = new THREE.Group();
  let words = lab.songs[initialStyle].style.prompt.split(" ");
  const state = { active: -1 };
  const text = canvasPanel(SLOT.text.w, 1.5, (ctx, cw, ch) => {
    ctx.font = "600 34px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    const gap = 14;
    const widths = words.map((wd) => ctx.measureText(wd).width + 28);
    let x = Math.max(6, (cw - widths.reduce((s, v) => s + v + gap, -gap)) / 2);
    words.forEach((wd, i) => {
      const on = state.active < 0 || i <= state.active;
      ctx.fillStyle = on ? "#1e3a5f" : "#111827";
      ctx.strokeStyle = on ? "#38bdf8" : "#334155";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.roundRect(x, 18, widths[i], ch - 36, 12);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = on ? "#e0f2fe" : "#64748b";
      ctx.fillText(wd, x + 14, ch / 2);
      x += widths[i] + gap;
    });
  });
  text.position.set(SLOT.text.x, SLOT.text.y, 0);
  const textTitle = label("Prompt testuale", { size: 0.42, sub: "parole intere (semplificato: T5 usa sotto-parole)" });
  textTitle.position.set(SLOT.text.x, SLOT.text.y + 1.6, 0);
  group.add(text, textTitle);

  const t5 = slab({ width: SLOT.t5.w, height: 2.2, depth: 1.4, emissive: 0x7c3aed, intensity: 0.3 });
  t5.position.set(SLOT.t5.x, SLOT.t5.y, 0);
  const t5Title = label("T5 · text encoder", { size: 0.42, sub: "pre-addestrato, non riaddestrato" });
  t5Title.position.set(SLOT.t5.x, SLOT.t5.y + 1.85, 0);
  group.add(t5, t5Title);
  // un vettore per parola sotto T5: sono le chiavi/valori della cross-attention
  const VEC = 8;
  const vectors = new THREE.InstancedMesh(new THREE.BoxGeometry(0.22, 0.22, 0.22), new THREE.MeshStandardMaterial({ roughness: 0.5, transparent: true }), 8 * VEC);
  const m = new THREE.Matrix4();
  const c = new THREE.Color();
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < VEC; j++) {
      m.makeTranslation(SLOT.t5.x - 3.6 + i * 1.03 + (j % 4) * 0.24 - 0.36, SLOT.t5.y - 1.55 - Math.floor(j / 4) * 0.26, 0.4);
      vectors.setMatrixAt(i * VEC + j, m);
      vectors.setColorAt(i * VEC + j, c.setHSL(0.72 + 0.05 * Math.sin(i * 3 + j), 0.6, 0.45 + 0.2 * Math.cos(i + j * 1.7)));
    }
  }
  vectors.instanceMatrix.needsUpdate = true;
  vectors.instanceColor.needsUpdate = true;
  vectors.computeBoundingBox();
  vectors.computeBoundingSphere();
  group.add(vectors);

  const setWords = (w) => {
    words = w;
    vectors.count = Math.min(8, words.length) * VEC;
    text.userData.redraw();
  };
  setWords(words);
  return {
    group,
    text,
    t5,
    members: { text: [text], t5: [t5, vectors] },
    setStyle(id) {
      setWords(lab.songs[id].style.prompt.split(" "));
    },
    /** Evidenzia le parole fino a i (−1 = tutte normali). */
    highlightWord(i) {
      state.active = i;
      text.userData.redraw();
    },
    glow(v) {
      t5.material.emissiveIntensity = 0.3 + 0.9 * v;
    },
  };
}

/* ===================================================================== transformer === */

export function buildTransformer() {
  const group = new THREE.Group();
  group.position.set(SLOT.transformer.x, SLOT.transformer.y, 0);
  const embeds = CB_COLORS.map((col, k) => {
    const e = slab({ width: 1.7, height: 0.45, depth: 0.8, color: 0x0f172a, emissive: col, intensity: 0.45 });
    e.position.set(-2.7 + k * 1.8, -3.8, 0);
    group.add(e);
    return e;
  });
  placeLabel(group, "embedding dei 4 token del passo precedente", 0, -4.55, 0, { size: 0.28, color: "#94a3b8" });
  const sum = addNode(0.32);
  sum.position.set(0, -2.85, 0);
  group.add(sum);
  const layers = [0, 1, 2].map((i) => {
    const l = slab({ width: 7.4, height: 1.05, depth: 2.2, emissive: 0x0e7490, intensity: 0.25 });
    l.position.set(0, -1.6 + i * 1.45, 0);
    group.add(l);
    return l;
  });
  const dots = label("⋮", { size: 0.5, color: "#94a3b8" });
  dots.position.set(0, 2.55, 1.2);
  group.add(dots);
  const tag = label("Blocchi transformer", { size: 0.38, sub: "self-attention causale · cross-attention · MLP" });
  tag.position.set(0, -0.15, 1.3);
  group.add(tag);
  const heads = CB_COLORS.map((col, k) => {
    const hd = slab({ width: 1.7, height: 0.55, depth: 0.8, color: 0x0f172a, emissive: col, intensity: 0.45 });
    hd.position.set(-2.7 + k * 1.8, 3.25, 0);
    group.add(hd);
    return hd;
  });
  placeLabel(group, "4 teste lineari: una per codebook", 0, 4.05, 0, { size: 0.3, color: "#e2e8f0" });
  const title = label("Transformer decoder", { size: 0.5, sub: "prevede i prossimi token audio" });
  title.position.set(0, 5.2, 0);
  group.add(title);
  return {
    group,
    layers,
    heads,
    /** Intensità dei blocchi (0…1) e codebook "attivo" nelle teste. */
    pulse(v, activeHead = -1) {
      layers.forEach((l, i) => (l.material.emissiveIntensity = 0.25 + 0.8 * v * (0.6 + 0.4 * Math.sin(i * 1.3 + v * 6))));
      heads.forEach((hd, k) => (hd.material.emissiveIntensity = activeHead < 0 || activeHead === k ? 0.45 + 0.6 * v : 0.2));
      embeds.forEach((e) => (e.material.emissiveIntensity = 0.45 + 0.4 * v));
    },
  };
}

/* =============================================================== cross-attention (linee) === */

/** Linee dalla T5 ai blocchi del transformer: la cross-attention "consulta" il testo. */
export function buildCrossAttention(transformer, nWords = 8) {
  const segments = transformer.layers.length * nWords;
  const positions = new Float32Array(segments * 6);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.25 }));
  lines.userData.noBounds = true;
  let s = 0;
  transformer.layers.forEach((layer) => {
    for (let i = 0; i < nWords; i++) {
      const x0 = SLOT.t5.x - 3.6 + i * 1.03;
      positions.set([x0, SLOT.t5.y - 1.9, 0.4, SLOT.transformer.x + 3.7, SLOT.transformer.y + layer.position.y, 0], s * 6);
      s++;
    }
  });
  lines.userData.setStrength = (v) => (lines.material.opacity = 0.12 + 0.75 * v);
  return lines;
}

/* ================================================================= delay pattern === */

const LENS_STEPS = 14;

export function buildDelay(lab) {
  const { w } = SLOT.delay;
  const T = lab.frames;
  const Q = CODEC.used;
  const group = new THREE.Group();
  group.position.set(SLOT.delay.x, SLOT.delay.y, 0);
  const state = { gen: null, step: -1, start: 0 };

  // lente: colonne = passi di generazione, righe = codebook; la cella (k, s) è il frame s − k
  const lens = canvasPanel(w, 5.2, (ctx, cw, ch) => {
    const padL = 150;
    const top = 70;
    const cellW = (cw - padL - 8) / LENS_STEPS;
    const cellH = (ch - top - 54) / Q;
    ctx.textBaseline = "middle";
    ctx.font = "600 22px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "left";
    ctx.fillText("passo", 8, top - 22);
    for (let i = 0; i < LENS_STEPS; i++) {
      const s = state.start + i;
      ctx.textAlign = "center";
      ctx.fillStyle = s === state.step ? "#ffffff" : "#94a3b8";
      ctx.fillText(String(s), padL + (i + 0.5) * cellW, top - 22);
    }
    for (let k = 0; k < Q; k++) {
      ctx.textAlign = "left";
      ctx.fillStyle = CB_CSS[k];
      ctx.fillText(`cb ${k + 1} (ritardo ${k})`, 8, top + (k + 0.5) * cellH);
      for (let i = 0; i < LENS_STEPS; i++) {
        const s = state.start + i;
        const frame = s - k;
        const x = padL + i * cellW + 3;
        const y = top + k * cellH + 3;
        const done = state.gen && s <= state.step;
        const special = frame < 0 || frame >= T;
        ctx.beginPath();
        ctx.roundRect(x, y, cellW - 6, cellH - 6, 6);
        if (!done) {
          ctx.strokeStyle = "#1e293b";
          ctx.lineWidth = 2;
          ctx.stroke();
          continue;
        }
        const tok = special ? null : state.gen.codes[k * T + frame];
        ctx.fillStyle = special ? "#334155" : tokenCss(tok);
        ctx.fill();
        ctx.fillStyle = special ? "#cbd5e1" : "#0b1220";
        ctx.textAlign = "center";
        ctx.font = "700 22px 'JetBrains Mono', ui-monospace, monospace";
        ctx.fillText(special ? "S" : String(tok), x + (cellW - 6) / 2, y + (cellH - 6) / 2 - 7);
        ctx.font = "500 15px Inter, system-ui, sans-serif";
        ctx.fillText(special ? "" : `f${frame}`, x + (cellW - 6) / 2, y + (cellH - 6) / 2 + 15);
        ctx.font = "600 22px Inter, system-ui, sans-serif";
      }
    }
    if (state.step >= state.start && state.step < state.start + LENS_STEPS) {
      const i = state.step - state.start;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 4;
      ctx.strokeRect(padL + i * cellW, top - 2, cellW, Q * cellH + 2);
    }
    ctx.textAlign = "left";
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "500 21px Inter, system-ui, sans-serif";
    ctx.fillText("Ogni colonna è un passo: il codebook k scrive il frame (passo − k). S = token speciale.", 8, ch - 24);
  });
  lens.position.set(0, 1.9, 0);
  group.add(lens);
  const title = label("Delay pattern", { size: 0.5, sub: "i 4 codebook sfasati di un passo" });
  title.position.set(0, 5.4, 0);
  group.add(title);

  const grid = tokenGrid(Q, T, { width: w, rowHeight: 0.42, gap: 0.08 });
  grid.group.position.y = -2.85;
  grid.reveal(0);
  group.add(grid.group);
  placeLabel(group, "token generati: 241 frame × 4 codebook", 0, -1.55, 0, { size: 0.3, color: "#e2e8f0" });

  return {
    group,
    lens,
    grid,
    /** Imposta la generazione da mostrare e il passo corrente (le celle fino a quel passo sono piene). */
    show(gen, step) {
      state.gen = gen;
      state.step = step;
      state.start = Math.max(0, Math.min(step - Math.floor(LENS_STEPS / 2), (gen ? gen.steps.length : LENS_STEPS) - LENS_STEPS));
      if (step < LENS_STEPS - 2) state.start = 0;
      lens.userData.redraw();
      if (gen) {
        grid.setCodes(gen.codes, T);
        grid.reveal(0, [0, 1, 2, 3].map((k) => Math.max(0, Math.min(T, step - k + 1))));
      } else grid.reveal(0);
    },
  };
}

/* ======================================================================= decoder === */

export function buildDecoder() {
  const group = new THREE.Group();
  group.position.set(SLOT.decoder.x, SLOT.decoder.y, 0);
  // piastre sempre più alte: il decoder AUMENTA la risoluzione nel tempo (upsampling)
  const plates = [1.6, 2.6, 3.7, 4.9].map((h, i) => {
    const p = slab({ width: 0.55, height: h, depth: 1.6, emissive: 0x059669, intensity: 0.3 });
    p.position.set(-2.4 + i * 1.6, 0, 0);
    group.add(p);
    return p;
  });
  const title = label("Decoder del codec", { size: 0.48, sub: "dai token al suono" });
  title.position.set(0, 3.6, 0);
  group.add(title);
  placeLabel(group, "qui: somma dei centroidi + Griffin-Lim", 0, -3.2, 0, { size: 0.27, color: "#94a3b8" });
  return {
    group,
    pulse(v) {
      plates.forEach((p, i) => (p.material.emissiveIntensity = 0.3 + 0.9 * Math.max(0, Math.sin(v * Math.PI * 2 - i * 0.6))));
    },
  };
}

/* ================================================================== audio generato === */

export function buildOutput(lab) {
  const { w, h } = SLOT.output;
  const group = new THREE.Group();
  group.position.set(SLOT.output.x, SLOT.output.y, 0);
  const amp = (h - 1) / 0.8;
  const wave = waveformMesh(new Float32Array(1400), { width: w, height: amp, color: 0x34d399, capacity: 700 });
  const axis = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-w / 2, 0, 0), new THREE.Vector3(w / 2, 0, 0)]),
    new THREE.LineBasicMaterial({ color: 0x334155, transparent: true }),
  );
  const box = new THREE.Mesh(new THREE.PlaneGeometry(w, h - 0.6), new THREE.MeshBasicMaterial({ color: 0x0b1220, transparent: true, opacity: 0.6, depthWrite: false }));
  box.position.z = -0.05;
  group.add(box, axis, wave);
  const titles = {
    empty: label("Audio generato", { size: 0.55, sub: "avvia \"Generare musica token per token\"" }),
    ready: label("Audio generato", { size: 0.55, sub: "premi Ascolta nel Laboratorio" }),
    rec: label("Audio ricostruito", { size: 0.55, sub: "dai token del brano originale" }),
  };
  for (const t of Object.values(titles)) {
    t.position.set(0, h / 2 + 0.9, 0);
    t.visible = false;
    group.add(t);
  }
  titles.empty.visible = true;
  const playhead = new THREE.Mesh(new THREE.BoxGeometry(0.05, h - 0.4, 0.05), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 }));
  playhead.visible = false;
  playhead.userData.noBounds = true;
  group.add(playhead);
  const duration = lab.songs[Object.keys(lab.songs)[0]].duration;
  return {
    group,
    /** samples = null svuota; kind: "ready" (generato) | "rec" (ricostruito) */
    setAudio(samples, kind = "ready") {
      wave.userData.setEnvelope(samples ? envelope(samples, 700) : new Float32Array(1400));
      for (const [k, t] of Object.entries(titles)) t.visible = k === (samples ? kind : "empty");
    },
    setPlayhead(t) {
      playhead.visible = t !== null && t !== undefined;
      if (playhead.visible) playhead.position.x = -w / 2 + (t / duration) * w;
    },
  };
}

export { CB_COLORS };
