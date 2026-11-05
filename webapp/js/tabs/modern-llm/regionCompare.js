/**
 * Regione "confronto": il blocco del nostro GPT (a sinistra) e un blocco decoder moderno
 * (a destra, i componenti di Llama 3) riga per riga. Le righe che cambiano hanno una freccia;
 * l'azione fa "volare" il componente vecchio verso quello nuovo.
 */
import { fillTemplate } from "../../core/dom.js";
import { arrow, box, card, ease, PALETTE, Plate, text } from "./shapes.js";

/**
 * Ogni riga: il pezzo del nostro GPT e quello moderno. `same` = non cambia.
 * I numeri del nostro GPT sono segnaposto riempiti con data/model.json (ourModel.js).
 */
const ROW_TEMPLATES = [
  { id: "pos", ours: ["Posizioni apprese", "tabella {{block}} × {{embd}} sommata all'ingresso"], modern: ["RoPE", "ruota q e k: conta la distanza"] },
  { id: "norm", ours: ["LayerNorm", "toglie la media, divide per σ"], modern: ["RMSNorm", "divide solo per la radice della media dei quadrati"] },
  { id: "attn", ours: ["Multi-head attention", "{{heads}} head, ognuna con i suoi K e V"], modern: ["GQA + KV cache", "32 head di query, 8 coppie K/V"] },
  { id: "res", ours: ["Connessione residua", "x + strato(norm(x))"], modern: ["Connessione residua", "identica: pre-norm + residuo"], same: true },
  { id: "ffn", ours: ["MLP con ReLU", "{{embd}} → {{hidden}} → {{embd}}"], modern: ["SwiGLU (o MoE)", "gate e up → down; router + esperti"] },
  { id: "out", ours: ["{{vocab}} caratteri, contesto {{block}}", "tokenizer a caratteri"], modern: ["128.256 token BPE", "contesto 8K (Llama 3), 128K (3.1)"] },
];

/** Righe del confronto con i numeri veri del nostro GPT (vars di modelVars()). */
export function compareRows(vars = {}) {
  return ROW_TEMPLATES.map((row) => ({ ...row, ours: row.ours.map((t) => fillTemplate(t, vars)) }));
}

const X_OURS = -8.6;
const X_MODERN = 8.6;
const CARD_W = 15;
const CARD_H = 2.45;
const rowY = (i) => 6.9 - i * 3.05;

export function buildCompare(group, vars) {
  const rows = compareRows(vars);
  text(group, "Il nostro GPT (L20)", X_OURS, 9.35, { size: 0.62, color: PALETTE.ours });
  text(group, "Blocco moderno (es. Llama 3)", X_MODERN, 9.35, { size: 0.62, color: PALETTE.modern });

  const ours = [];
  const modern = [];
  rows.forEach((row, i) => {
    const y = rowY(i);
    ours.push(card(group, { x: X_OURS, y, w: CARD_W, h: CARD_H, title: row.ours[0], sub: row.ours[1], color: PALETTE.ours, size: 0.46 }));
    modern.push(card(group, { x: X_MODERN, y, w: CARD_W, h: CARD_H, title: row.modern[0], sub: row.modern[1], color: row.same ? PALETTE.muted : PALETTE.modern, size: 0.46 }));
    if (!row.same) arrow(group, [[-0.95, y], [0.95, y]], { color: 0x38bdf8, radius: 0.07, head: 0.5 });
    else text(group, "=", 0, y, { size: 0.7, color: PALETTE.muted });
  });

  const note = new Plate({ w: 34, h: 1.6, size: 0.55, align: "center", color: PALETTE.muted }).place(group, 0, -11.6);
  note.set("Stessa ossatura (norm → attention → residuo → norm → MLP → residuo): cambiano i pezzi.");

  // "Fantasma": copia del componente vecchio che vola verso il nuovo durante l'azione
  const ghost = box({ w: CARD_W, h: CARD_H, d: 0.36, color: "#f59e0b", emissive: "#f59e0b", opacity: 0.55 });
  ghost.visible = false;
  ghost.userData.noBounds = true;
  group.add(ghost);

  return {
    rows,
    ours,
    modern,
    note,
    rowY,
    /** Accende la riga i (o nessuna con -1); le altre restano spente. */
    highlight(i) {
      ours.forEach((c, k) => c.set(k === i ? 0.8 : 0));
      modern.forEach((c, k) => c.set(k === i ? 0.8 : 0));
    },
    /** Mostra le card moderne fino alla riga `upTo` compresa (le altre restano "spente"). */
    reveal(upTo) {
      modern.forEach((c, k) => {
        c.mesh.visible = k <= upTo;
        if (c.label) c.label.visible = k <= upTo;
      });
    },
    /** t in 0..1: il fantasma va dalla card del nostro GPT a quella moderna della riga i. */
    fly(i, t) {
      ghost.visible = i >= 0 && t > 0 && t < 1;
      const k = ease(t);
      ghost.position.set(X_OURS + (X_MODERN - X_OURS) * k, rowY(i), 0.4 + Math.sin(Math.PI * k) * 2.5);
      ghost.material.color.set(k < 0.5 ? "#f59e0b" : "#38bdf8");
      ghost.material.emissive.set(k < 0.5 ? "#f59e0b" : "#38bdf8");
      ghost.material.emissiveIntensity = 0.6;
    },
    reset() {
      this.highlight(-1);
      this.reveal(rows.length);
      ghost.visible = false;
      note.set("Stessa ossatura (norm → attention → residuo → norm → MLP → residuo): cambiano i pezzi.");
    },
  };
}
