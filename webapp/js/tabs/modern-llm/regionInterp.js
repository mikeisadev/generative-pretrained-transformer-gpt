/**
 * Regione "Interpretabilità".
 * In alto: una induction head (Olsson et al. 2022) — sullo schema [A][B] … [A] guarda il token
 * che seguiva la volta precedente e lo copia. In basso: un autoencoder sparso (dictionary
 * learning, Anthropic 2023-2024) — un vettore di attivazioni diventa poche "feature" accese.
 * I nomi delle feature sono esempi illustrativi (dichiarati).
 */
import * as THREE from "three";
import { Cells, Links, PALETTE, Plate, TokenStrip, arrow, text } from "./shapes.js";

export const INDUCTION_TOKENS = ["Mario", "Rossi", "arriva", "in", "ufficio", ".", "Poi", "Mario", "?"];
const A_FIRST = 0;
const B_FIRST = 1;
const A_LAST = 7;
const TOKEN_W = 3.55;
const tokX = (i) => -14.2 + i * TOKEN_W;
const TOK_Y = 6.4;

const ACT = 10; // dimensione del vettore di attivazioni
const FEAT_COLS = 12;
const FEAT_ROWS = 3;
export const FEATURES = [
  { index: 4, name: "codice Python" },
  { index: 17, name: "nomi di città" },
  { index: 29, name: "frasi in italiano" },
];

const FEATURE_LINES = FEATURES.map((f) => ({ text: `feature ${f.index}: «${f.name}»`, color: "#fef08a" }));

export function buildInterp(group) {
  text(group, "Induction head: [A][B] … [A] → [B]", -6.5, 9.1, { size: 0.48, color: "#f0abfc" });
  const strips = INDUCTION_TOKENS.map((t, i) => new TokenStrip({ w: TOKEN_W - 0.1, h: 1.8, font: 0.72, align: "center", mono: false }).place(group, tokX(i), TOK_Y));
  const arc = new Links(24);
  arc.setCount(24);
  group.add(arc.line);
  // arco da "Mario" (ultimo) a "Rossi" (dopo il primo "Mario"): curva campionata in 24 segmenti
  const from = new THREE.Vector2(tokX(A_LAST), TOK_Y - 0.8);
  const to = new THREE.Vector2(tokX(B_FIRST), TOK_Y - 0.8);
  const curve = (t) => [from.x + (to.x - from.x) * t, from.y - Math.sin(Math.PI * t) * 2.2];
  for (let k = 0; k < 24; k++) {
    const [ax, ay] = curve(k / 24);
    const [bx, by] = curve((k + 1) / 24);
    arc.set(k, ax, ay, 0.3, bx, by, 0.3);
  }
  const prediction = new Plate({ w: 15, h: 1.3, size: 0.55, align: "center" }).place(group, 6.2, 2.4);

  text(group, "Autoencoder sparso: attivazioni → poche feature", 0, 0.2, { size: 0.48, color: "#f0abfc" });
  const act = new Cells(ACT, { w: 0.9, h: 0.62, d: 0.2 });
  const rec = new Cells(ACT, { w: 0.9, h: 0.62, d: 0.2 });
  const feats = new Cells(FEAT_COLS * FEAT_ROWS, { w: 1.05, h: 1.05, d: 0.2 });
  const rand = (i) => Math.sin(i * 12.9898) * 0.5 + 0.5;
  for (let i = 0; i < ACT; i++) {
    act.place(i, -14.6, -1.7 - i * 0.72, 0);
    rec.place(i, 14.6, -1.7 - i * 0.72, 0);
  }
  for (let r = 0; r < FEAT_ROWS; r++) for (let c = 0; c < FEAT_COLS; c++) feats.place(r * FEAT_COLS + c, -6.6 + c * 1.2, -2.6 - r * 1.25, 0);
  group.add(act.mesh, rec.mesh, feats.mesh);
  text(group, "attivazioni", -14.6, -9.4, { size: 0.36, color: PALETTE.muted });
  text(group, "ricostruzione", 14.6, -9.4, { size: 0.36, color: PALETTE.muted });
  text(group, "dizionario di feature (molte più dei neuroni)", 0, -6.5, { size: 0.36, color: PALETTE.muted });
  arrow(group, [[-13.6, -4.5], [-7.6, -4.5]], { color: 0xf0abfc, radius: 0.05 });
  arrow(group, [[7.6, -4.5], [13.6, -4.5]], { color: 0xf0abfc, radius: 0.05 });
  text(group, "encoder", -10.6, -3.9, { size: 0.32, color: PALETTE.muted });
  text(group, "decoder", 10.6, -3.9, { size: 0.32, color: PALETTE.muted });
  const featureNames = new Plate({ w: 15, h: 2.2, size: 0.49, align: "center" }).place(group, 0, -8.2);
  const note = new Plate({ w: 34, h: 1.6, size: 0.47, align: "center", color: PALETTE.muted }).place(group, 0, -11.3);
  note.set("Nomi delle feature: esempi illustrativi. Il metodo è quello di Anthropic (Towards / Scaling Monosemanticity).");

  const api = {
    /** t: 0 = solo la frase, 0.5 = arco di attention, 1 = predizione copiata. */
    induction(t) {
      for (let k = 0; k < 24; k++) arc.color(k, t > 0.4 && k / 24 <= (t - 0.4) / 0.4 ? "#f0abfc" : "#0d1117");
      const stage = t >= 1 ? 3 : t > 0.4 ? 2 : t > 0 ? 1 : 0;
      if (stage === api.stage) return;
      api.stage = stage;
      INDUCTION_TOKENS.forEach((tok, i) => {
        const isA = i === A_FIRST || i === A_LAST;
        const isB = i === B_FIRST;
        const shown = i === INDUCTION_TOKENS.length - 1 && t >= 1 ? "Rossi" : tok;
        strips[i].set([{ text: shown, fill: isA && t > 0 ? "#4a044e" : isB && t > 0.4 ? "#1e3a8a" : i === 8 && t >= 1 ? "#14532d" : "#1e293b", stroke: isA && t > 0 ? "#f0abfc" : isB && t > 0.4 ? "#93c5fd" : "#475569" }]);
      });
      prediction.set(t >= 1 ? [{ text: "Prossimo token previsto: «Rossi» (copiato dal contesto)", bold: true, color: "#bbf7d0" }] : t > 0.4 ? "La head guarda il token dopo il «Mario» precedente…" : "");
    },
    /** t: 0 = spento, 0.5 = attivazioni, 1 = feature accese e ricostruzione. */
    sae(t) {
      for (let i = 0; i < ACT; i++) {
        act.mix(i, "#1e293b", rand(i) > 0.5 ? "#f0abfc" : "#7e22ce", Math.min(1, t * 2));
        rec.mix(i, "#1e293b", rand(i) > 0.5 ? "#f0abfc" : "#7e22ce", Math.max(0, t * 2 - 1));
      }
      for (let k = 0; k < FEAT_COLS * FEAT_ROWS; k++) feats.mix(k, "#1e293b", "#fde047", FEATURES.some((f) => f.index === k) ? Math.max(0, t * 2 - 0.8) : 0);
      featureNames.set(t > 0.6 ? FEATURE_LINES : "");
    },
  };
  api.induction(1);
  api.sae(1);
  return api;
}
