/**
 * Regione "Leggi di scala": per cinque budget di calcolo, le barre di parametri N e token D
 * ottimali secondo il fit di Chinchilla (altezza = log10, quindi ogni tacca = ×10).
 * Numeri calcolati con chinchillaOptimal (coefficienti pubblicati da Hoffmann et al. 2022).
 */
import { chinchillaOptimal } from "./sim.js";
import { Cells, PALETTE, Plate, text } from "./shapes.js";

export const BUDGETS = [1e20, 1e21, 1e22, 1e23, 1e24];
const BASE_Y = -5.2;
const UNIT = 1.25; // altezza per ogni potenza di 10 sopra 10^7
const colX = (k) => -12.4 + k * 6.2;
const barH = (v) => (Math.log10(v) - 7) * UNIT;

const human = (v) => {
  const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"]];
  for (const [u, s] of units) if (v >= u) return `${(v / u).toFixed(v / u < 10 ? 1 : 0).replace(".", ",")}${s}`;
  return String(Math.round(v));
};

/** Testo di riposo sotto le barre (l'azione «Capacità emergenti» lo sostituisce e poi lo ripristina). */
const DEFAULT_NOTE = [
  { text: "Fit di Chinchilla: L(N, D) = 1,69 + 406,4 / N^0,34 + 410,7 / D^0,28, con C ≈ 6·N·D", bold: true },
  { text: "Questo fit dà modelli più piccoli degli altri due metodi del paper (63-67B per il budget di Gopher).", color: PALETTE.muted },
  { text: "Da quelli viene Chinchilla: 70B parametri e 1,4T token, stesso calcolo di Gopher (280B), risultati migliori.", color: PALETTE.muted },
];

export function buildScaling(group) {
  const bars = new Cells(BUDGETS.length * 2, { w: 1.9, h: 1, d: 0.6 });
  group.add(bars.mesh);
  const values = BUDGETS.map((C) => chinchillaOptimal(C));
  const tops = [];
  BUDGETS.forEach((C, k) => {
    text(group, `10^${Math.log10(C)} FLOP`, colX(k), BASE_Y - 1, { size: 0.42, color: PALETTE.muted });
    tops.push(new Plate({ w: 5.8, h: 2.6, size: 0.52, align: "center" }).place(group, colX(k), 7.6));
  });
  text(group, "■ parametri N", -11.4, -7.9, { size: 0.45, color: "#fb923c" });
  text(group, "■ token D", -5.4, -7.9, { size: 0.45, color: "#f472b6" });
  text(group, "altezza in scala log: ogni tacca = ×10", 6.2, -7.9, { size: 0.4, color: PALETTE.muted });
  const note = new Plate({ w: 34, h: 3.4, size: 0.55, align: "center" }).place(group, 0, -10.5);

  const api = {
    values,
    /** grow (0..1) fa crescere le barre; focus = budget evidenziato (-1 nessuno). */
    set(grow = 1, focus = -1) {
      values.forEach(({ N, D, loss }, k) => {
        const hN = Math.max(0.001, barH(N) * grow);
        const hD = Math.max(0.001, barH(D) * grow);
        bars.place(k * 2, colX(k) - 1.05, BASE_Y + hN / 2, 0, 1, hN, 1);
        bars.place(k * 2 + 1, colX(k) + 1.05, BASE_Y + hD / 2, 0, 1, hD, 1);
        const on = focus === k;
        bars.color(k * 2, on ? "#fdba74" : "#c2410c");
        bars.color(k * 2 + 1, on ? "#f9a8d4" : "#be185d");
        if (focus === api.focus) return;
        tops[k].set([
          { text: `N ≈ ${human(N)}`, color: "#fdba74", bold: on },
          { text: `D ≈ ${human(D)}`, color: "#f9a8d4", bold: on },
          { text: `L ≈ ${loss.toFixed(2).replace(".", ",")}`, color: on ? "#fef3c7" : PALETTE.muted },
        ]);
      });
      api.focus = focus;
    },
    /** Testo sotto le barre; senza argomenti torna quello di riposo. */
    note(lines = DEFAULT_NOTE) {
      note.set(lines);
    },
  };
  api.set(1, -1);
  api.note();
  return api;
}
