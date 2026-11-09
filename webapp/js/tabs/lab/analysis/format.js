/**
 * Come si scrivono i numeri del confronto, dei grafici e del benchmark (logica pura, testata).
 *
 * Stesse convenzioni del resto del Laboratorio: virgola decimale, segno meno tipografico "−",
 * separatore delle migliaia sempre presente, numeri minuscoli in notazione scientifica leggibile
 * ("3,2 · 10⁻⁷") invece di "0,0000", che farebbe credere a uno zero vero. Riusa i formattatori del
 * nucleo (display.js, tensorTableMath.js) invece di riscriverli.
 */
import { scientificText } from "../../../core/ui/tensorTableMath.js";
import { MINUS, formatPrecise } from "../../../shared/weights/display.js";

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const isNumber = (v) => typeof v === "number";

/** Un intero all'italiana ("111.539"); "—" se non è un numero. */
export function int(value) {
  return isNumber(value) && Number.isFinite(value) ? INT.format(value) : "—";
}

/**
 * Un numero con `digits` cifre significative: `num(0.89812)` → "0,898", `num(-1.2e-6)` → "−1,2 · 10⁻⁶",
 * `num(209729)` → "209.729". "—" per i valori mancanti; NaN e ±∞ scritti come tali (dati rotti: meglio
 * vederli che nasconderli).
 */
export function num(value, digits = 3) {
  if (!isNumber(value)) return "—";
  if (Number.isNaN(value)) return "NaN";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : `${MINUS}∞`;
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 1e-3) return scientificText(value, Math.min(digits, 2));
  if (abs >= 1e7) return scientificText(value, digits);
  const rounded = Number(value.toPrecision(digits));
  // 999,96 con 3 cifre diventerebbe "1e3": oltre le cifre chieste si scrive l'intero
  if (Math.abs(rounded) >= 10 ** digits) return `${value < 0 ? MINUS : ""}${INT.format(Math.round(abs))}`;
  return formatPrecise(rounded, digits);
}

/** Un numero con `decimals` decimali fissi ("1,8145"), per le metriche del benchmark. */
export function fixed(value, decimals = 4) {
  if (!isNumber(value) || !Number.isFinite(value)) return "—";
  const rounded = Number(value.toFixed(decimals));
  const text = Math.abs(rounded).toLocaleString("it-IT", { minimumFractionDigits: decimals, maximumFractionDigits: decimals, useGrouping: "always" });
  return rounded < 0 ? MINUS + text : text;
}

/**
 * Una quota (0…1 o più) come percentuale con i decimali che servono: 0,473 → "47,3%", 1,853 → "185%",
 * 0,00031 → "0,031%". `decimals` forza i decimali.
 */
export function pct(value, decimals = null) {
  if (!isNumber(value) || Number.isNaN(value)) return "—";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : `${MINUS}∞`;
  const p = value * 100;
  const abs = Math.abs(p);
  if (p !== 0 && abs < 0.001 && decimals === null) return `${scientificText(p, 2)}%`;
  const d = decimals ?? (abs === 0 ? 0 : abs < 0.1 ? 3 : abs < 1 ? 2 : abs < 100 ? 1 : 0);
  const rounded = Number(p.toFixed(d));
  const text = Math.abs(rounded).toLocaleString("it-IT", { minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: "always" });
  return `${rounded < 0 ? MINUS : ""}${text}%`;
}

/** Una differenza con il segno sempre scritto: "+0,0123", "−0,0123", "0,0000" (stesse regole di `fixed`). */
export function signedFixed(value, decimals = 4) {
  if (!isNumber(value) || !Number.isFinite(value)) return "—";
  const rounded = Number(value.toFixed(decimals));
  const text = fixed(value, decimals);
  return rounded > 0 ? `+${text}` : text;
}

/** Una differenza in notazione adattiva con il segno: "+3,2 · 10⁻⁷", "−0,0021", "0". */
export function signedNum(value, digits = 2) {
  if (!isNumber(value) || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const text = num(value, digits);
  return value > 0 ? `+${text}` : text;
}

/** Una differenza intera con il segno: "+2", "−1", "0". */
export function signedInt(value) {
  if (!isNumber(value) || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return `${value > 0 ? "+" : MINUS}${INT.format(Math.abs(value))}`;
}

/** Una differenza tra due quote in punti percentuali: 0,1685 → "+16,85 pp". */
export function signedPoints(value, decimals = 2) {
  if (!isNumber(value) || !Number.isFinite(value)) return "—";
  const p = Number((value * 100).toFixed(decimals));
  if (p === 0) return "0 pp";
  const text = Math.abs(p).toLocaleString("it-IT", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return `${p > 0 ? "+" : MINUS}${text} pp`;
}

/** Una forma di tensore: [16, 64] → "16 × 64", [64] → "64". */
export function shape(s) {
  return Array.isArray(s) && s.length ? s.map((v) => INT.format(v)).join(" × ") : "scalare";
}

/** Durata in secondi: 0,82 → "820 ms", 12,3 → "12,3 s". */
export function seconds(value) {
  if (!isNumber(value) || !Number.isFinite(value) || value < 0) return "—";
  if (value < 1) return `${num(value * 1000, 2)} ms`;
  return `${num(value, 3)} s`;
}
