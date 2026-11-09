/**
 * Livello Panoramica della vista completa: la vita del modello da sinistra a destra (dati →
 * pre-training → contesto lungo → post-training → PESI → deployment → inferenza), con la barra dei
 * byte per precisione sotto i pesi, i passi del post-training, le pillole "da verificare" e due
 * schede di sintesi (capacità, dove sono i pesi). Tutto dai dati del modello (full/models/*.js).
 *
 *   const macro = buildMacro(group, model, layout, totals);
 *   macro.anchors   Map("macro:<id>" → Box3)
 *   macro.highlight(stageId | null); macro.fillPrecision(t)   per le azioni
 */
import * as THREE from "three";
import { glow } from "../shapes.js";
import { MACRO_NOTES, UNVERIFIED_LABEL } from "./content.js";
import { arrow, bigBlock, flatBox, rectBox, textPlate } from "./draw.js";
import { formatCount, formatGB, formatShare } from "./model.js";
import { label } from "../../../core/visuals.js";

/** Colori delle fasi: addestramento (verde), pesi (ambra, come i pesi appresi del corso), uso (ciano). */
const STAGE_COLORS = { data: "#4ade80", pretrain: "#4ade80", context: "#4ade80", post: "#4ade80", weights: "#fbbf24", deploy: "#22d3ee", infer: "#22d3ee" };
/** Colori delle precisioni (gli stessi del pannello "Pesi" e dei grafici). */
export const DTYPE_COLORS = { MXFP4: "#a78bfa", FP8: "#22d3ee", E8M0: "#64748b", BF16: "#fbbf24", FP32: "#f472b6" };

export function buildMacro(group, model, layout, totals) {
  const anchors = new Map();
  const stageBlocks = new Map();
  const k = layout.stages[0].w / 112; // fattore di scala della Panoramica (testi compresi)
  /** Sulla fase solo la prima frase: la Panoramica si legge da lontano; il resto è nelle azioni e nei pannelli. */
  const firstSentence = (text) => text.split(/(?<=\.)\s/)[0];

  const title = label(`La vita di ${model.name}: dai dati al token`, { size: 15 * k, color: "#e2e8f0" });
  title.position.set(layout.title.x, layout.title.y, 0);
  group.add(title);

  model.pipeline.forEach((stage, i) => {
    const r = layout.stages[i];
    const block = bigBlock(group, { title: stage.title, subtitle: stage.sub, text: firstSentence(stage.text), color: STAGE_COLORS[stage.id] ?? "#38bdf8" }, { x: r.x, y: r.y, w: r.w, h: r.h, S: 27 * k });
    stageBlocks.set(stage.id, block);
    anchors.set(`macro:${stage.id}`, rectBox(r, 4 * k));
    if (i > 0) {
      const prev = layout.stages[i - 1];
      arrow(group, [[prev.x + prev.w / 2 + 1.5 * k, 0], [r.x - r.w / 2 - 1.5 * k, 0]], { color: 0x67e8f9, radius: 0.9 * k, head: 5 * k, z: 0.5 });
    }
    // pillole "da verificare" sotto la fase: il valore dubbio con la parola chiave
    const doubts = (stage.unverified ?? []).map((id) => model.unverified.find((u) => u.id === id)).filter(Boolean);
    doubts.forEach((u, j) => {
      const plate = textPlate({ w: r.w * 0.98, h: 10 * k, size: 4.6 * k, align: "center", lines: [{ text: `${UNVERIFIED_LABEL}: ${u.value}`, color: "#fde68a", bold: true }], background: "rgba(120, 53, 15, 0.55)", border: "#f59e0b" });
      plate.place(group, r.x, r.y - r.h / 2 - 7.5 * k - j * 11.5 * k, 0.4);
    });
    // passi del post-training (SFT → RL → OPD) come pillole sotto la fase
    if (stage.steps) {
      const plate = textPlate({ w: r.w * 0.98, h: 11 * k, size: 6 * k, align: "center", mono: true, lines: [{ text: stage.steps.join("  →  "), color: "#bbf7d0", bold: true }], background: "rgba(20, 83, 45, 0.6)", border: "#4ade80" });
      plate.place(group, r.x, r.y - r.h / 2 - 8 * k - doubts.length * 11.5 * k, 0.4);
    }
  });

  // barra dei byte per precisione sotto la fase dei pesi (segmenti proporzionali ai GB)
  const precisionBars = [];
  const weights = layout.stages.find((s) => s.id === "weights");
  if (weights && model.precision?.length) {
    const total = model.precision.reduce((a, p) => a + p.gb, 0);
    const barY = weights.y - weights.h / 2 - 9 * k - ((model.pipeline.find((s) => s.id === "weights")?.unverified?.length ?? 0) * 11.5 * k);
    const barH = 8 * k;
    let x = weights.x - weights.w / 2;
    for (const p of model.precision) {
      const w = (p.gb / total) * weights.w;
      const mesh = flatBox(group, { x: x + w / 2, y: barY, w: Math.max(0.15, w - 0.25 * k), h: barH, d: 0.6 }, { color: DTYPE_COLORS[p.dtype] ?? "#94a3b8", emissive: DTYPE_COLORS[p.dtype] ?? "#94a3b8", intensity: 0.15 });
      precisionBars.push({ mesh, x0: x, w });
      x += w;
    }
    const big = model.precision.filter((p) => p.gb / total > 0.12);
    const legend = textPlate({
      w: weights.w,
      h: 8 * k + big.length * 6.4 * k,
      size: 4.6 * k,
      lines: [{ text: `${formatGB(total)}${model.precision.some((p) => p.estimate) ? " (stima)" : ""} per precisione:`, color: "#e2e8f0", bold: true }, ...big.map((p) => ({ text: `${p.dtype} · ${p.what}: ${formatGB(p.gb)}`, color: DTYPE_COLORS[p.dtype] }))],
    });
    legend.place(group, weights.x, barY - barH / 2 - legend.opts.h / 2 - 1.5 * k, 0.4);
    anchors.set("macro:weights", rectBox({ x: weights.x, y: (weights.y + weights.h / 2 + barY - barH) / 2, w: weights.w, h: weights.h / 2 + weights.y - barY + barH + 14 * k }, 4 * k));
  }

  // schede di sintesi in basso
  const notes = [
    ["capabilities", layout.capabilities, MACRO_NOTES.capabilities.title, MACRO_NOTES.capabilities.text, "#c084fc"],
    ["weights-share", layout.weightsShare, MACRO_NOTES.weightsShare.title, MACRO_NOTES.weightsShare.text.replace("{{share}}", formatShare(totals.components.find((c) => c.id === "routed")?.share ?? 0)), "#fbbf24"],
  ];
  for (const [id, r, noteTitle, text, color] of notes) {
    const subtitle = id === "weights-share" ? `${formatCount(totals.backbone, { digits: 2 })} nel backbone · ${formatShare(totals.components[0]?.share ?? 0)} in ${totals.components[0]?.label.toLowerCase() ?? ""}` : "5 fatti, ognuno con la sua cautela";
    bigBlock(group, { title: noteTitle, subtitle, text, color }, { x: r.x, y: r.y, w: r.w, h: r.h, S: 27 * k });
    anchors.set(`macro:${id}`, rectBox(r, 4 * k));
  }

  const box = layout.box;
  anchors.set("macro:all", new THREE.Box3(new THREE.Vector3(box.minX, box.minY, -1), new THREE.Vector3(box.maxX, box.maxY, 1)));

  let lit = null;
  return {
    anchors,
    /** Accende una fase (le azioni la usano per dire "siamo qui"); null le spegne. */
    highlight(id) {
      if (lit === id) return;
      for (const [sid, block] of stageBlocks) block.traverse((o) => o.isMesh && o.material.emissive && glow(o, sid === id ? 0.35 : 0, STAGE_COLORS[sid]));
      lit = id;
    },
    /** La barra delle precisioni che si riempie da sinistra (0 = vuota, 1 = piena). */
    fillPrecision(t) {
      const end = precisionBars.length ? precisionBars[0].x0 + t * precisionBars.reduce((a, b) => a + b.w, 0) : 0;
      for (const { mesh, x0, w } of precisionBars) {
        const visible = Math.max(0, Math.min(w, end - x0));
        mesh.visible = visible > 0.05;
        mesh.scale.x = Math.max(0.001, visible / Math.max(0.001, w));
        mesh.position.x = x0 + visible / 2;
      }
    },
  };
}
