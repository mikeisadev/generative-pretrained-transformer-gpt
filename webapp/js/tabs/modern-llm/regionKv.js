/**
 * Regione "KV cache": colonne = token generati, righe = layer (K sopra, V sotto).
 * Con la cache a ogni passo si accende solo la colonna nuova; senza cache si ricalcolano tutte
 * (lampeggio rosso). Celle in un solo InstancedMesh.
 */
import { Cells, PALETTE, Plate, TokenStrip, text } from "./shapes.js";

export const KV_TOKENS = ["Il", "gatto", "dorme", "sul", "divano", "e", "sogna", "un", "topo", "di", "for", "maggio"];
const T = KV_TOKENS.length;
const LAYERS = 4;
const colX = (t) => -10.6 + t * 2.15;
const kY = (l) => 4.6 - l * 1.0;
const vY = (l) => -1.2 - l * 1.0;

export function buildKv(group) {
  const header = new TokenStrip({ w: 33, h: 1.9, font: 0.74, gap: 0.14, align: "center" }).place(group, 0, 7.6);
  text(group, "K", -14.6, kY(1.5), { size: 0.9, color: "#22d3ee" });
  text(group, "V", -14.6, vY(1.5), { size: 0.9, color: "#f472b6" });
  text(group, "4 layer", -14.6, kY(1.5) - 1.2, { size: 0.36, color: PALETTE.muted });
  text(group, "4 layer", -14.6, vY(1.5) - 1.2, { size: 0.36, color: PALETTE.muted });

  const cells = new Cells(T * LAYERS * 2, { w: 1.9, h: 0.82, d: 0.3 });
  const idx = (t, l, isV) => (t * LAYERS + l) * 2 + (isV ? 1 : 0);
  for (let t = 0; t < T; t++)
    for (let l = 0; l < LAYERS; l++) {
      cells.place(idx(t, l, false), colX(t), kY(l), 0);
      cells.place(idx(t, l, true), colX(t), vY(l), 0);
    }
  group.add(cells.mesh);
  const counter = new Plate({ w: 34, h: 3.6, size: 0.62, align: "center" }).place(group, 0, -8.7);

  const api = {
    T,
    /**
     * n = token presenti; `cache` = usa la cache; flash (0..1) = lampeggio del calcolo del passo.
     * computedWith / computedWithout: totale di colonne K/V calcolate finora nei due casi.
     */
    set(n, cache = true, flash = 0) {
      const textKey = n * 2 + (cache ? 1 : 0);
      const textChanged = textKey !== api.textKey;
      api.textKey = textKey;
      if (textChanged) header.set(KV_TOKENS.map((tok, t) => ({ text: tok, fill: t < n ? (t === n - 1 ? "#155e75" : "#1e293b") : "#0f172a", stroke: t < n ? (t === n - 1 ? "#67e8f9" : "#475569") : "#1e293b", color: t < n ? "#f1f5f9" : "#334155" })));
      for (let t = 0; t < T; t++)
        for (let l = 0; l < LAYERS; l++)
          for (let v = 0; v < 2; v++) {
            const isV = v === 1;
            const i = idx(t, l, isV);
            if (t >= n) {
              cells.hide(i);
              continue;
            }
            cells.place(i, colX(t), isV ? vY(l) : kY(l), 0);
            const base = isV ? "#9d174d" : "#155e75";
            const fresh = t === n - 1;
            const recomputed = !cache && !fresh;
            if (fresh) cells.mix(i, base, isV ? "#f9a8d4" : "#a5f3fc", flash);
            else if (recomputed) cells.mix(i, base, "#ef4444", flash);
            else cells.color(i, base);
          }
      if (!textChanged) return;
      const without = (n * (n + 1)) / 2;
      counter.set([
        { text: `${n} token · colonne K/V calcolate in tutto (per layer):`, color: PALETTE.muted },
        { text: `con la cache ${n}   ·   senza cache ${without}`, bold: true, color: cache ? "#a5f3fc" : "#fca5a5" },
        { text: cache ? "Con la cache si calcola solo la colonna nuova; le altre si rileggono dalla memoria." : "Senza cache: a ogni passo si rifà tutto il prefisso (in rosso).", color: PALETTE.muted },
      ]);
    },
  };
  api.set(T, true);
  return api;
}
