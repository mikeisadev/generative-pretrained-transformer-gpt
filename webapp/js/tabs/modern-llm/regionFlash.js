/**
 * Regione "Attention efficiente": a sinistra la matrice di attention calcolata a blocchi
 * (idea di FlashAttention: i blocchi passano dalla memoria veloce SRAM, la matrice intera non
 * viene mai scritta nella HBM); a destra la maschera a finestra scorrevole (Mistral 7B).
 */
import { card, Cells, PALETTE, Plate, text } from "./shapes.js";

export const FLASH_N = 16;
export const FLASH_TILE = 4;
const CELL = 0.62;
const LEFT_X = -8.4;
const RIGHT_X = 9.2;
const Y0 = 2.6;
const cx = (j, x0) => x0 + (j - (FLASH_N - 1) / 2) * CELL;
const cy = (i) => Y0 + ((FLASH_N - 1) / 2 - i) * CELL;
/** Testo di riposo della regione (ripristinato anche alla fine dell'azione). */
const DEFAULT_NOTE = [
  { text: "Il risultato è identico all'attention normale: cambia solo come si muovono i dati.", bold: true },
  { text: "Contesti lunghi: Llama 3.1 e DeepSeek-V3 arrivano a 128K token.", color: PALETTE.muted },
];

export function buildFlash(group) {
  text(group, "FlashAttention: a blocchi", LEFT_X, 8.9, { size: 0.5, color: "#5eead4" });
  text(group, "Finestra scorrevole", RIGHT_X, 8.9, { size: 0.5, color: "#5eead4" });
  const left = new Cells(FLASH_N * FLASH_N, { w: CELL * 0.92, h: CELL * 0.92, d: 0.15 });
  const right = new Cells(FLASH_N * FLASH_N, { w: CELL * 0.92, h: CELL * 0.92, d: 0.15 });
  for (let i = 0; i < FLASH_N; i++)
    for (let j = 0; j < FLASH_N; j++) {
      left.place(i * FLASH_N + j, cx(j, LEFT_X), cy(i), 0);
      right.place(i * FLASH_N + j, cx(j, RIGHT_X), cy(i), 0);
    }
  group.add(left.mesh, right.mesh);
  text(group, "query ↓   key →", LEFT_X, -2.9, { size: 0.36, color: PALETTE.muted });
  const sram = card(group, { x: -12.6, y: -5.6, w: 7.6, h: 1.8, title: "SRAM: veloce, piccola", color: "#5eead4", size: 0.36 });
  const hbm = card(group, { x: -4.2, y: -5.6, w: 7.6, h: 1.8, title: "HBM: grande, lenta", color: PALETTE.muted, size: 0.36 });
  const windowNote = new Plate({ w: 16, h: 2, size: 0.55, align: "center" }).place(group, RIGHT_X, -4.6);
  const note = new Plate({ w: 34, h: 3.4, size: 0.57, align: "center" }).place(group, 0, -9.7);

  const api = {
    /** Colora la matrice di sinistra: blocchi già calcolati, blocco corrente acceso (k = indice del blocco). */
    tile(k, glowT = 1) {
      const per = FLASH_N / FLASH_TILE;
      for (let i = 0; i < FLASH_N; i++)
        for (let j = 0; j < FLASH_N; j++) {
          const c = i * FLASH_N + j;
          if (j > i) {
            left.color(c, "#111827");
            continue;
          }
          const b = Math.floor(i / FLASH_TILE) * per + Math.floor(j / FLASH_TILE);
          if (k < 0) left.color(c, "#134e4a");
          else if (b < k) left.color(c, "#0f766e");
          else if (b === k) left.mix(c, "#0f766e", "#99f6e4", glowT);
          else left.color(c, "#1f2937");
        }
      sram.set(k >= 0 ? glowT : 0);
      hbm.set(0);
    },
    /** Maschera causale con finestra W (W ≥ FLASH_N = attention piena). Riga `row` evidenziata. */
    window(W, row = -1) {
      for (let i = 0; i < FLASH_N; i++)
        for (let j = 0; j < FLASH_N; j++) {
          const c = i * FLASH_N + j;
          const visible = j <= i && i - j < W;
          right.color(c, visible ? (i === row ? "#99f6e4" : "#0f766e") : j <= i ? "#3f1d1d" : "#111827");
        }
      if (W === api.lastW) return;
      api.lastW = W;
      windowNote.set(W >= FLASH_N ? "Senza finestra: ogni token vede tutti i precedenti" : `W = ${W}: ogni token vede solo gli ultimi ${W}\n(Mistral 7B: W = 4096)`);
    },
    /** Testo sotto le matrici; senza argomenti torna quello di riposo. */
    note(lines = DEFAULT_NOTE) {
      note.set(lines);
    },
  };
  api.tile(-1);
  api.window(4);
  api.note();
  return api;
}
