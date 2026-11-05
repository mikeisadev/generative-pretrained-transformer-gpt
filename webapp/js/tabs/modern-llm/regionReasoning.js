/**
 * Regione "Ragionamento": una domanda verificabile, quattro catene di pensiero campionate,
 * un verificatore e (per GRPO) ricompense e vantaggi del gruppo.
 * Le catene sono un copione didattico; i vantaggi sono calcolati con la formula vera (sim.js).
 */
import { grpoAdvantages } from "./sim.js";
import { card, Cells, PALETTE, Plate, text } from "./shapes.js";

export const QUESTION = "Quanto fa 17 × 24?";
/** Quattro tentativi: passi di ragionamento (brevi) e risposta finale. 17 × 24 = 408. */
export const CHAINS = [
  { steps: ["17 × 20 = 340", "17 × 4 = 68", "340 + 68"], answer: 408 },
  { steps: ["17 × 25 = 425", "tolgo 27", "425 − 27"], answer: 398 },
  { steps: ["24 × 10 = 240", "24 × 7 = 168", "240 + 168"], answer: 408 },
  { steps: ["circa 20 × 21"], answer: 420 },
];
export const CORRECT = 408;
export const REWARDS = CHAINS.map((c) => (c.answer === CORRECT ? 1 : 0));

const STEP_SLOTS = 3;
const rowY = (r) => 4.9 - r * 3.1;
const stepX = (j) => -14.6 + j * 4.5;
const X_ANSWER = 0.9;
const X_REWARD = 6.4;
const X_ADV = 12.6;

export function buildReasoning(group) {
  const question = new Plate({ w: 16, h: 1.6, size: 0.81, align: "center", background: "#1a2e05", border: "#a3e635" }).place(group, -8.4, 8.9).set([{ text: QUESTION, bold: true }]);
  const verifier = card(group, { x: 9.6, y: 8.9, w: 12, h: 1.9, title: "Verificatore", sub: "controlla il numero finale", color: "#a3e635", size: 0.42 });
  text(group, "passi di pensiero", -10, 6.75, { size: 0.38, color: PALETTE.muted });
  text(group, "risposta", X_ANSWER, 6.75, { size: 0.38, color: PALETTE.muted });
  const rewardHead = text(group, "ricompensa", X_REWARD, 6.75, { size: 0.38, color: PALETTE.muted });
  const advHead = text(group, "vantaggio", X_ADV, 6.75, { size: 0.38, color: PALETTE.muted });

  const steps = CHAINS.map((c, r) => Array.from({ length: STEP_SLOTS }, (_, j) => new Plate({ w: 4.3, h: 1.6, size: 0.43, align: "center", background: "#14231a", border: "#3f6212" }).place(group, stepX(j), rowY(r))));
  const answers = CHAINS.map((c, r) => new Plate({ w: 3.6, h: 1.6, size: 0.72, align: "center", background: "#0f172a", border: "#475569" }).place(group, X_ANSWER, rowY(r)));
  const bars = new Cells(CHAINS.length * 2, { w: 1, h: 1.2, d: 0.3 });
  group.add(bars.mesh);
  const values = CHAINS.map(() => new Plate({ w: 2.4, h: 1, size: 0.55, align: "center" }));
  values.forEach((v, r) => v.place(group, X_ADV + 3.8, rowY(r)));
  const note = new Plate({ w: 34, h: 3.6, size: 0.55, align: "center" }).place(group, 0, -9.8);
  const advantages = grpoAdvantages(REWARDS);

  const api = {
    advantages,
    /** shown[r] = quanti passi della catena r sono visibili; answers = mostra le risposte. */
    setChains(shown, showAnswers = false) {
      const key = shown.join(",") + (showAnswers ? "+" : "-");
      if (key === api.chainsKey) return;
      api.chainsKey = key;
      api.verdictOn = undefined; // le risposte vengono riscritte: il verdetto va riapplicato
      CHAINS.forEach((c, r) => {
        steps[r].forEach((plate, j) => {
          const visible = j < Math.min(shown[r] ?? 0, c.steps.length);
          plate.mesh.visible = visible;
          if (visible) plate.set(c.steps[j]);
        });
        answers[r].mesh.visible = showAnswers;
        answers[r].set([{ text: String(c.answer), bold: true }]);
      });
    },
    /** Colora le risposte in base al verificatore (t > 0 = verdetto visibile). */
    verdict(t) {
      verifier.set(t > 0 ? Math.min(1, t * 1.5) : 0);
      if ((t > 0) === api.verdictOn) return;
      api.verdictOn = t > 0;
      CHAINS.forEach((c, r) => {
        const ok = c.answer === CORRECT;
        answers[r].opts.background = t > 0 ? (ok ? "#14532d" : "#7f1d1d") : "#0f172a";
        answers[r].opts.border = t > 0 ? (ok ? "#4ade80" : "#f87171") : "#475569";
        answers[r].key = null; // forza il ridisegno con i nuovi colori
        answers[r].set([{ text: `${c.answer}${t > 0 ? (ok ? "  giusta" : "  errata") : ""}`, bold: true }]);
      });
    },
    /** Barre di ricompensa (cresciute di t) e vantaggio (cresciute di a); t = 0 nasconde tutto. */
    grpo(t, a = t) {
      rewardHead.visible = t > 0;
      advHead.visible = a > 0;
      CHAINS.forEach((c, r) => {
        const rw = REWARDS[r] * 3 * t;
        const adv = advantages[r] * 2.4 * a;
        if (t <= 0) {
          bars.hide(r * 2);
          bars.hide(r * 2 + 1);
          values[r].mesh.visible = false;
          return;
        }
        bars.place(r * 2, X_REWARD - 1.5 + Math.max(rw, 0.05) / 2, rowY(r), 0, Math.max(rw, 0.05), 1, 1);
        bars.color(r * 2, REWARDS[r] ? "#4ade80" : "#475569");
        if (a > 0) bars.place(r * 2 + 1, X_ADV + adv / 2, rowY(r), 0, Math.max(Math.abs(adv), 0.05), 1, 1);
        else bars.hide(r * 2 + 1);
        bars.color(r * 2 + 1, adv >= 0 ? "#4ade80" : "#f87171");
        values[r].mesh.visible = a > 0.5;
        if (values[r].ref) return; // il testo del vantaggio non cambia: basta scriverlo una volta
        values[r].set([{ text: (advantages[r] >= 0 ? "+" : "") + advantages[r].toFixed(2).replace(".", ","), color: adv >= 0 ? "#bbf7d0" : "#fecaca", bold: true }]);
      });
    },
    note(lines) {
      note.set(lines);
    },
    question,
  };
  api.setChains([3, 3, 3, 3], true);
  api.verdict(1);
  api.grpo(0);
  api.note([{ text: "Più calcolo al momento dell'inferenza: passi intermedi, più tentativi, un verificatore.", bold: true }, { text: "Le catene sono un copione didattico; il verificatore controlla davvero 17 × 24 = 408.", color: PALETTE.muted }]);
  return api;
}
