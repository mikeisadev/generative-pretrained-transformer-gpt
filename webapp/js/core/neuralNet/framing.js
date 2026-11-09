/**
 * Inquadrature delle reti lunghe (logica pura, testata con node).
 *
 * Una rete lunga e bassa (DeepSeek-V4.1-Flash: 240 unità × 17; l'Audio: 133 × 21) vista tutta intera di
 * lato è una striscia di 25-90 px: non si leggono né i neuroni né le etichette delle sezioni. La vista
 * iniziale ("Centra") diventa allora una FINESTRA larga quanto serve perché la rete sia alta almeno
 * `readableHeight` pixel, che comincia dall'ingresso; le altre finestre ("parti") si scorrono con ◀ ▶
 * della scheda, e "Tutta la rete" resta la vista d'insieme. Le parti cominciano dove comincia una
 * sezione (o una ripetizione) che la parte prima non mostrava per intero, così ogni parte ha i suoi
 * blocchi interi; l'ultima finisce con l'uscita.
 */

/**
 * Altezza leggibile della rete sullo schermo: metà della parte libera, tra 150 e 400 px (mai più del 90%
 * della parte libera). A 1600 × 900 i neuroni di una sezione da 32 distano ~6 px: si vedono uno per uno.
 */
export const READABLE = Object.freeze({ fraction: 0.5, minPx: 150, maxPx: 400, maxOfFree: 0.9 });

/** Pixel di altezza voluti per la rete in una parte libera alta `freeHeight` pixel. */
export function readableHeight(freeHeight, r = READABLE) {
  return Math.min(r.maxOfFree * freeHeight, Math.max(r.minPx, Math.min(r.maxPx, r.fraction * freeHeight)));
}

const PAD = 0.6; // unità prima della sezione con cui comincia una parte: non sta a filo del bordo

/**
 * Le parti di una rete per una finestra larga `width` unità.
 * @param {object} o
 * @param {{x0: number, x1: number, label?: string}[]} o.items sezioni e ripetizioni nell'ordine del flusso
 * @param {[number, number]} o.span estensione orizzontale di tutto ciò che si inquadra (assi e nomi compresi)
 * @param {number} o.width larghezza della finestra (unità)
 * @param {{x0: number, x1: number, label: string}[]} [o.groups] parentesi dei gruppi (per i titoli)
 * @returns {{x0: number, x1: number, title: string}[]} almeno una parte; una sola se la rete ci sta
 */
export function walkParts({ items, span, width, groups = [] }) {
  const [s0, s1] = span;
  if (!(width > 0) || s1 - s0 <= width * 1.04) return [{ x0: s0, x1: s1, title: partTitle(items, groups, s0, s1) }];
  const parts = [];
  let start = s0;
  for (let guard = 0; guard < 1000; guard++) {
    const end = start + width;
    if (end >= s1 - 1e-6) break;
    parts.push({ x0: start, x1: end });
    // la prossima comincia con il primo blocco che questa non mostra per intero (almeno mezza finestra
    // avanti); se quel blocco comincia dopo la fine di questa, in mezzo c'è solo spazio vuoto
    const next = items.find((it) => it.x1 > end + 1e-6);
    start = Math.max(start + 0.5 * width, next ? next.x0 - PAD : end);
  }
  // l'ultima finisce con l'uscita; se quasi coincide con la penultima, la sostituisce
  const last = { x0: Math.max(s0, s1 - width), x1: s1 };
  if (parts.length && last.x0 - parts.at(-1).x0 < 0.25 * width) parts.pop();
  parts.push(last);
  return parts.map((p) => ({ ...p, title: partTitle(items, groups, p.x0, p.x1) }));
}

/**
 * Titolo di una parte: le parentesi dei gruppi il cui centro sta nella parte (al più tre, escluse quelle
 * più larghe della parte: un "Encoder" di venti layer non è il nome di una parte), altrimenti la prima
 * e l'ultima sezione che si vedono.
 */
export function partTitle(items, groups, x0, x1) {
  const width = x1 - x0;
  const names = [];
  for (const g of [...groups].sort((a, b) => a.x0 - b.x0)) {
    const c = (g.x0 + g.x1) / 2;
    if (c >= x0 && c <= x1 && g.x1 - g.x0 <= 0.8 * width && g.label && !names.includes(g.label)) names.push(g.label);
  }
  if (names.length) return names.length > 3 ? `${names.slice(0, 3).join(", ")}…` : names.join(", ");
  const inside = items.filter((it) => it.label && it.x0 >= x0 - 1e-6 && it.x1 <= x1 + 1e-6);
  if (!inside.length) return "";
  return inside.length === 1 ? inside[0].label : `${inside[0].label} … ${inside.at(-1).label}`;
}

/** La parte che si sta guardando: quella con il centro più vicino al punto guardato (x). */
export function partIndexAt(parts, x) {
  let best = 0;
  for (let i = 1; i < parts.length; i++) if (Math.abs((parts[i].x0 + parts[i].x1) / 2 - x) < Math.abs((parts[best].x0 + parts[best].x1) / 2 - x)) best = i;
  return best;
}

/**
 * I due modi di togliere alla parte libera `free` il rettangolo `block` (il pad in un angolo): tagliata
 * sopra di lui o alla sua sinistra. [free] se non si toccano. Tutto in pixel dello stage { x0, y0, x1, y1 }.
 * Chi inquadra una forma nota (tutta la rete, una sezione) li prova tutti e due e tiene quello dove la
 * forma viene più grande: una rete larga sta meglio sopra il pad, una alta alla sua sinistra.
 */
export function avoidRectVariants(free, block, gap = 8) {
  if (block.x1 <= free.x0 || block.x0 >= free.x1 || block.y1 <= free.y0 || block.y0 >= free.y1) return [free];
  const above = { ...free, y1: Math.max(free.y0, Math.min(free.y1, block.y0 - gap)) };
  const left = { ...free, x1: Math.max(free.x0, Math.min(free.x1, block.x0 - gap)) };
  return [above, left];
}

/**
 * Uno solo dei due tagli (avoidRectVariants): dove resta più area. È quello delle parti di una rete
 * lunga, che non hanno una forma da far entrare: con il pad del desktop vince di solito il taglio a
 * sinistra, resta tutta l'altezza e la rete viene più alta (si legge meglio) a costo di una parte in più.
 */
export function avoidRect(free, block, gap = 8) {
  const [a, b = a] = avoidRectVariants(free, block, gap);
  const area = (r) => (r.x1 - r.x0) * (r.y1 - r.y0);
  return area(a) >= area(b) ? a : b;
}

/**
 * Inquadratura di fronte di un riquadro (largo 2·halfW, alto 2·halfH, profondo `depth`, in unità di scena)
 * dentro la parte libera dello stage dove viene più grande, fra `rects` (pixel { x0, y0, x1, y1 } di uno
 * stage W × H, per esempio i due tagli di avoidRectVariants attorno al pad). I `covered` pixel a destra sono
 * coperti da pannelli: il punto principale della camera sta al centro del resto, come in SceneHost.
 * Con lo stage intero e covered 0 è la distanza di CameraDirector.focus (senza minDistance).
 * @returns {{ rect: object, distance: number, dx: number, dy: number }} la parte scelta, la distanza della
 *   camera dal centro del riquadro e lo spostamento che porta il centro al centro della parte (unità di
 *   scena lungo destra e su della camera): il punto guardato è centro − destra·dx − su·dy
 */
export function fitInFreeRect({ halfW, halfH, depth = 0, rects, W, H, covered = 0, tanV, margin = 1 }) {
  let best = null;
  for (const rect of rects) {
    const fw = Math.max(80, rect.x1 - rect.x0);
    const fh = Math.max(80, rect.y1 - rect.y0);
    const distance = Math.max((halfW * H) / (fw * tanV), (halfH * H) / (fh * tanV)) * margin + depth;
    if (!best || distance < best.distance) best = { rect, distance };
  }
  const k = (2 * best.distance * tanV) / H; // unità di scena per pixel, nel piano del centro
  return {
    ...best,
    dx: ((best.rect.x0 + best.rect.x1) / 2 - (W - covered) / 2) * k,
    dy: (H / 2 - (best.rect.y0 + best.rect.y1) / 2) * k,
  };
}
