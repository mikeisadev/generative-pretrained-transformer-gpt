/**
 * Regione "Decodifica speculativa": il modello piccolo propone γ token, quello grande li verifica
 * tutti in un solo passaggio; si tengono quelli accettati più un token del modello grande.
 * Le proposte sono un copione didattico (dichiarato); la regola di accettazione è quella del paper.
 */
import { card, PALETTE, Plate, TokenStrip, arrow } from "./shapes.js";

/**
 * Copione: per ogni giro, le proposte della bozza, quante ne accetta il grande e il token che
 * il grande aggiunge (correzione del primo rifiutato, oppure un token in più se le accetta tutte).
 */
export const SPEC_ROUNDS = [
  { draft: ["gatto", "dorme", "sul", "letto"], accepted: 3, bonus: "divano" },
  { draft: ["vicino", "alla", "finestra", "aperta"], accepted: 4, bonus: "mentre" },
  { draft: ["piove", "forte", "e", "tuona"], accepted: 1, bonus: "fuori" },
];
export const SPEC_PROMPT = ["Il"];

const COLORS = {
  done: { fill: "#1e293b", stroke: "#475569" },
  draft: { fill: "#713f12", stroke: "#facc15" },
  ok: { fill: "#14532d", stroke: "#4ade80" },
  bad: { fill: "#7f1d1d", stroke: "#f87171", color: "#fecaca" },
  bonus: { fill: "#1e3a8a", stroke: "#93c5fd" },
};

export function buildSpec(group) {
  const draft = card(group, { x: -11.5, y: 6.4, w: 11, h: 2.6, title: "Bozza: modello piccolo", sub: "veloce, propone γ = 4 token uno alla volta", color: PALETTE.warn, size: 0.42 });
  const target = card(group, { x: -11.5, y: -3.2, w: 11, h: 2.6, title: "Modello grande", sub: "verifica tutte le proposte in 1 passaggio", color: "#f472b6", size: 0.42 });
  arrow(group, [[-5.6, 6.4], [-3.4, 4.4]], { color: 0xfacc15 });
  arrow(group, [[-5.6, -3.2], [-3.4, -0.6]], { color: 0xf472b6 });
  const strip = new TokenStrip({ w: 20.5, h: 7.4, font: 0.78, gap: 0.22 }).place(group, 7.3, 1.9);
  const legend = new Plate({ w: 34, h: 1.9, size: 0.52, align: "center", color: PALETTE.muted }).place(group, 0, -6.4);
  legend.set("grigio = testo già fissato · giallo = proposta della bozza\nverde = accettata · rosso = rifiutata · blu = token del modello grande");
  const counter = new Plate({ w: 34, h: 3.4, size: 0.6, align: "center" }).place(group, 0, -9.6);

  const api = {
    /**
     * Stato dopo `round` giri completati e, nel giro corrente, fase:
     * "draft" (k proposte visibili), "verify" (esito), "commit" (tenuto il risultato).
     */
    set(round, phase = "commit", k = 4) {
      const key = round * 100 + (phase === "draft" ? 10 : phase === "verify" ? 20 : 30) + k; // numero: niente stringhe a ogni frame
      if (key === api.key) return;
      api.key = key;
      const tokens = SPEC_PROMPT.map((t) => ({ text: t, ...COLORS.done }));
      let passes = 0;
      for (let r = 0; r < round; r++) {
        const R = SPEC_ROUNDS[r];
        R.draft.slice(0, R.accepted).forEach((t) => tokens.push({ text: t, ...COLORS.done }));
        tokens.push({ text: R.bonus, ...COLORS.done });
        passes++;
      }
      const R = SPEC_ROUNDS[round];
      if (R && phase !== "commit") {
        R.draft.slice(0, k).forEach((t, i) => {
          if (phase === "draft") tokens.push({ text: t, ...COLORS.draft });
          else tokens.push({ text: t, ...(i < R.accepted ? COLORS.ok : COLORS.bad) });
        });
        if (phase === "verify") tokens.push({ text: R.bonus, ...COLORS.bonus });
      }
      strip.set(tokens);
      draft.set(phase === "draft" ? 0.8 : 0);
      target.set(phase === "verify" ? 0.8 : 0);
      const produced = tokens.length - SPEC_PROMPT.length;
      const verifying = phase === "verify" ? 1 : 0;
      counter.set([
        { text: `passaggi del modello grande: ${passes + verifying}   ·   token generati: ${phase === "verify" ? produced - (R.draft.length - R.accepted) : produced}`, bold: true },
        { text: "Senza bozza servirebbe un passaggio del modello grande per OGNI token.", color: PALETTE.muted },
        { text: "Copione didattico: le proposte sono scritte a mano per mostrare il meccanismo.", color: PALETTE.muted },
      ]);
    },
  };
  api.set(SPEC_ROUNDS.length);
  return api;
}
