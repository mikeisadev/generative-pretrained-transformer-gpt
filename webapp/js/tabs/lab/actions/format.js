/**
 * Come si scrivono i numeri delle azioni (logica pura, testata): virgola decimale e segno "−" come nelle
 * celle della scena (shared/weights/display.js), percentuali e interi all'italiana.
 */
import { formatCell, formatPrecise } from "../../../shared/weights/display.js";

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const MINUS = "−";

/** Un numero con `digits` decimali: num(-0.12345) → "−0,1235"; sotto 10^-digits in notazione leggibile. */
export function num(value, digits = 4) {
  if (!Number.isFinite(value)) return formatCell(value, 0);
  if (value !== 0 && Math.abs(value) < 0.5 * 10 ** -digits) return sci(value, 2);
  return formatCell(value, digits);
}

/** Come num, ma con il segno anche per i positivi (per le differenze): "+0,0312", "−0,0120". */
export function signed(value, digits = 4) {
  const text = num(value, digits);
  return value > 0 ? `+${text}` : text;
}

/**
 * Notazione scientifica leggibile: sci(0.000034) → "3,4 · 10⁻⁵". La mantissa si arrotonda con
 * toExponential, che sposta anche l'esponente: sci(0.000999) → "1,0 · 10⁻³" (non "10,0 · 10⁻⁴").
 */
export function sci(value, digits = 2) {
  if (!Number.isFinite(value)) return formatCell(value, 0);
  if (value === 0) return "0";
  const [mantissa, exponent] = Math.abs(value).toExponential(Math.max(0, digits - 1)).split("e");
  const sup = String(Number(exponent)).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  return `${value < 0 ? MINUS : ""}${mantissa.replace(".", ",")} · 10${sup}`;
}

/** Il valore con tutta la precisione utile di un float32 ("−0,01234568"). */
export const precise = (value) => formatPrecise(value);

/** Una probabilità come percentuale: pct(0.4123) → "41,2%"; sotto lo 0,1% con due cifre significative. */
export function pct(p, digits = 1) {
  if (!Number.isFinite(p)) return "—";
  const v = p * 100;
  if (v !== 0 && Math.abs(v) < 0.1) return `${formatCell(v, 2).replace(/0+$/, "").replace(/,$/, "")}%`;
  return `${formatCell(v, digits)}%`;
}

/** Un intero con il separatore delle migliaia: int(209729) → "209.729". */
export const int = (n) => INT.format(n);

/** Un fattore tra parentesi se negativo, per i prodotti scritti per esteso: "(−0,0341)", "0,8123". */
export function factor(value, digits = 4) {
  const text = num(value, digits);
  return value < 0 ? `(${text})` : text;
}

/** I primi valori di un vettore, "[0,81; −1,20; 0,33; …]" (punto e virgola: la virgola è il decimale). */
export function vectorPreview(values, { count = 4, digits = 2 } = {}) {
  const n = values.length;
  const shown = Array.from(values.slice(0, Math.min(count, n)), (v) => num(v, digits));
  return `[${shown.join("; ")}${n > count ? "; …" : ""}]`;
}

export { MINUS };
