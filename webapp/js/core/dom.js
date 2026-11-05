/**
 * Piccole utility DOM condivise da tutti i tab.
 */

/** Crea un elemento: el("div", {className: "x"}, figlio1, "testo", ...). */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

/** Escape per inserire testo NON fidato dentro innerHTML. */
export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Formatta un numero all'italiana con separatore delle migliaia sempre presente. */
export function fmt(n, digits = 0) {
  return Number(n).toLocaleString("it-IT", { useGrouping: "always", maximumFractionDigits: digits, minimumFractionDigits: digits });
}

/** Sostituisce {{chiave}} con i valori di vars (lascia intatte le chiavi sconosciute). */
export function fillTemplate(text, vars) {
  return text.replace(/\{\{(\w+)\}\}/g, (match, key) => (key in vars ? String(vars[key]) : match));
}

/** Promessa che si risolve dopo ms millisecondi. */
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** true se l'utente preferisce animazioni ridotte (accessibilità). */
export const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
