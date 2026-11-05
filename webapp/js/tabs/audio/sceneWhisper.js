/**
 * Ramo Whisper (vista alternativa): audio → log-Mel (30 s, 3000 × 80, calcolato davvero con la
 * formula di whisper/audio.py) → encoder (2 Conv1d + blocchi) → decoder causale con
 * cross-attention → token di testo. Il testo d'uscita è un ESEMPIO del formato (il nostro audio
 * è musica e Whisper qui non viene eseguito).
 */
import * as THREE from "three";
import { label } from "../../core/visuals.js";
import { WHISPER } from "./dsp.js";
import { SLOT } from "./layout3d.js";
import { canvasPanel, disposeTree, placeLabel, rampColor, slab } from "./widgets.js";

/** Token d'uscita d'esempio: i primi quattro sono quelli veri del formato di Whisper. */
export const WHISPER_TOKENS = ["<|startoftranscript|>", "<|it|>", "<|transcribe|>", "<|notimestamps|>", " Ciao", " a", " tutti", "<|endoftext|>"];

export function buildWhisper(lab, initialStyle) {
  const group = new THREE.Group();

  // --- log-Mel: texture 3000 × 80
  const { w: MW, h: MH0 } = SLOT.wMel;
  const MH = MH0 - 0.8;
  const rgba = new Uint8Array(WHISPER.nFrames * WHISPER.nMels * 4);
  const texture = new THREE.DataTexture(rgba, WHISPER.nFrames, WHISPER.nMels, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  const mel = new THREE.Mesh(new THREE.PlaneGeometry(MW, MH), new THREE.MeshBasicMaterial({ map: texture, transparent: true }));
  mel.position.set(SLOT.wMel.x, SLOT.wMel.y, 0);
  // velo che copre la parte non ancora "calcolata" durante l'animazione
  const veil = new THREE.Mesh(new THREE.PlaneGeometry(1, MH + 0.04), new THREE.MeshBasicMaterial({ color: 0x0d1117, transparent: true, opacity: 0.94, depthWrite: false }));
  veil.position.set(SLOT.wMel.x, SLOT.wMel.y, 0.02);
  veil.userData.noBounds = true;
  veil.visible = false;
  // bordo dei 30 secondi: la parte di solo silenzio è scura come lo sfondo, il bordo la rende visibile
  const melEdge = new THREE.LineSegments(new THREE.EdgesGeometry(mel.geometry), new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true }));
  melEdge.position.copy(mel.position);
  group.add(melEdge);
  const melTitle = label("log-Mel di Whisper", { size: 0.5, sub: "30 s = 3000 frame × 80 bande, calcolato qui" });
  melTitle.position.set(SLOT.wMel.x, SLOT.wMel.y + MH / 2 + 1, 0);
  group.add(mel, veil, melTitle);
  let signalFrames = 0;
  const brace = new THREE.Group();
  group.add(brace);
  const color = new THREE.Color();

  function setStyle(id) {
    const lm = lab.whisper(id);
    signalFrames = lm.signalFrames;
    // valori tipici in −1…1,5: li riportiamo in 0…1 per la rampa di colore
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of lm.values) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
    for (let t = 0; t < WHISPER.nFrames; t++) {
      for (let m = 0; m < WHISPER.nMels; m++) {
        rampColor((lm.values[t * WHISPER.nMels + m] - lo) / (hi - lo || 1), color);
        const o = (m * WHISPER.nFrames + t) * 4; // riga = banda (dal basso), colonna = frame
        rgba[o] = color.r * 255;
        rgba[o + 1] = color.g * 255;
        rgba[o + 2] = color.b * 255;
        rgba[o + 3] = 255;
      }
    }
    texture.needsUpdate = true;
    disposeTree(brace); // le etichette del brano precedente non servono più
    brace.clear();
    const xEnd = SLOT.wMel.x - MW / 2 + (signalFrames / WHISPER.nFrames) * MW;
    const y = SLOT.wMel.y - MH / 2 - 0.35;
    const tagA = label(`il nostro brano: ${signalFrames} frame`, { size: 0.28, color: "#fbbf24" });
    tagA.position.set((SLOT.wMel.x - MW / 2 + xEnd) / 2 + 0.9, y - 0.25, 0);
    const tagB = label("silenzio di riempimento fino a 30 s", { size: 0.28, color: "#94a3b8" });
    tagB.position.set((xEnd + SLOT.wMel.x + MW / 2) / 2, y - 0.25, 0);
    const mark = new THREE.Mesh(new THREE.BoxGeometry(0.04, MH + 0.3, 0.04), new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true }));
    mark.position.set(xEnd, SLOT.wMel.y, 0.03);
    mark.userData.noBounds = true;
    brace.add(tagA, tagB, mark);
  }
  setStyle(initialStyle);

  // --- encoder: 2 convoluzioni sottili + blocchi
  const enc = new THREE.Group();
  enc.position.set(SLOT.wEnc.x, SLOT.wEnc.y, 0);
  const convs = [0, 1].map((i) => {
    const c = slab({ width: 7, height: 0.4, depth: 1.8, emissive: 0xd97706, intensity: 0.35 });
    c.position.y = -2.6 + i * 0.6;
    enc.add(c);
    return c;
  });
  const encBlocks = [0, 1, 2, 3].map((i) => {
    const b = slab({ width: 7, height: 0.75, depth: 1.8, emissive: 0x0e7490, intensity: 0.25 });
    b.position.y = -1.1 + i * 1.0;
    enc.add(b);
    return b;
  });
  placeLabel(enc, "Conv1d ×2 (la seconda con passo 2): 3000 → 1500", 0, -3.3, 0, { size: 0.26, color: "#fcd34d" });
  const encTitle = label("Encoder", { size: 0.48, sub: "attention bidirezionale" });
  encTitle.position.set(0, 3.5, 0);
  enc.add(encTitle);
  group.add(enc);

  // --- decoder
  const dec = new THREE.Group();
  dec.position.set(SLOT.wDec.x, SLOT.wDec.y, 0);
  const decBlocks = [0, 1, 2, 3].map((i) => {
    const b = slab({ width: 7, height: 0.75, depth: 1.8, emissive: 0x7c3aed, intensity: 0.25 });
    b.position.y = -1.6 + i * 1.0;
    dec.add(b);
    return b;
  });
  placeLabel(dec, "token di testo già scritti", 0, -2.75, 0, { size: 0.26, color: "#94a3b8" });
  const decTitle = label("Decoder", { size: 0.48, sub: "causale + cross-attention sull'audio" });
  decTitle.position.set(0, 3.5, 0);
  dec.add(decTitle);
  group.add(dec);

  // cross-attention: dalle uscite dell'encoder (lato destro) ai blocchi del decoder
  const segs = decBlocks.length * 4;
  const pos = new Float32Array(segs * 6);
  let s = 0;
  for (const b of decBlocks) {
    for (let j = 0; j < 4; j++) {
      pos.set([SLOT.wEnc.x + 3.5, SLOT.wEnc.y + encBlocks[j].position.y, 0, SLOT.wDec.x - 3.5, SLOT.wDec.y + b.position.y, 0], s * 6);
      s++;
    }
  }
  const cross = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(pos, 3)), new THREE.LineBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.3 }));
  cross.userData.noBounds = true;
  group.add(cross);

  // --- token d'uscita
  const tstate = { shown: WHISPER_TOKENS.length };
  const text = canvasPanel(SLOT.wText.w, SLOT.wText.h, (ctx, cw, ch) => {
    // font senza legature: in alcuni monospace "<|" diventerebbe un simbolo unico
    ctx.font = "600 40px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    let x = 10;
    let y = 60;
    WHISPER_TOKENS.forEach((tok, i) => {
      const wd = ctx.measureText(tok).width + 22;
      if (x + wd > cw - 10) (x = 10), (y += 96);
      const on = i < tstate.shown;
      const special = tok.startsWith("<|");
      ctx.fillStyle = on ? (special ? "#3b1f5e" : "#123b33") : "#111827";
      ctx.strokeStyle = on ? (special ? "#a78bfa" : "#34d399") : "#1e293b";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.roundRect(x, y - 38, wd, 76, 12);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = on ? "#f1f5f9" : "#334155";
      ctx.fillText(tok.replace(" ", "·"), x + 11, y);
      x += wd + 12;
    });
    ctx.font = "500 32px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("viola = token speciali (formato vero) · verde = testo d'esempio", 10, ch - 30);
  });
  text.position.set(SLOT.wText.x, SLOT.wText.y, 0);
  const textTitle = label("Testo trascritto", { size: 0.5, sub: "esempio del formato: qui Whisper non viene eseguito" });
  textTitle.position.set(SLOT.wText.x, SLOT.wText.y + SLOT.wText.h / 2 + 1, 0);
  group.add(text, textTitle);

  return {
    group,
    parts: { wMel: mel, wEnc: enc, wDec: dec, wText: text },
    setStyle,
    /** Rivela il log-Mel fino alla frazione f (0…1) dei frame del brano; null = tutto visibile. */
    revealMel(f) {
      veil.visible = f !== null && f !== undefined;
      if (!veil.visible) return;
      const x0 = SLOT.wMel.x - MW / 2 + (f * signalFrames * MW) / WHISPER.nFrames;
      const width = SLOT.wMel.x + MW / 2 - x0;
      veil.scale.x = Math.max(0.001, width);
      veil.position.x = x0 + width / 2;
    },
    /** Quanti token d'uscita mostrare. */
    showTokens(n) {
      tstate.shown = n;
      text.userData.redraw();
    },
    pulse(part, v) {
      const list = part === "enc" ? [...convs, ...encBlocks] : part === "dec" ? decBlocks : [];
      list.forEach((b, i) => (b.material.emissiveIntensity = 0.25 + 0.9 * v * (0.5 + 0.5 * Math.sin(v * 8 - i))));
      if (part === "cross") cross.material.opacity = 0.2 + 0.7 * v;
    },
  };
}
