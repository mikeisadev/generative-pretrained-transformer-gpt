/**
 * Impacchettamento a scaffali (shelf packing) dei riquadri di testo in pagine di un atlante (puro).
 *
 * Perché un atlante: la scena del Laboratorio ha centinaia di etichette (titolo, riga di codice e
 * conteggi di ogni contenitore). Una texture per etichetta vorrebbe dire centinaia di disegni per frame,
 * troppi con il rendering software; con un atlante tutte le etichette di un livello del LOD sono UN
 * mesh per pagina. I testi uguali (key/query/value di tutte le head, per esempio) occupano un posto solo.
 *
 *   const { pages, places } = packShelves([{ w: 600, h: 80 }, …], { pageWidth: 2048, maxPageHeight: 2048 });
 *   places[i] = { page, x, y }   (pixel, origine in alto a sinistra); pages[k] = { width, height } usati davvero
 */

/**
 * @param {{w: number, h: number}[]} sizes riquadri in pixel (interi, w ≤ pageWidth)
 * @param {{pageWidth?: number, maxPageHeight?: number, padding?: number}} [o]
 * @returns {{pages: {width: number, height: number}[], places: {page: number, x: number, y: number}[]}}
 */
export function packShelves(sizes, { pageWidth = 2048, maxPageHeight = 2048, padding = 2 } = {}) {
  const order = sizes.map((s, i) => i).sort((a, b) => sizes[b].h - sizes[a].h || sizes[b].w - sizes[a].w || a - b);
  const places = new Array(sizes.length);
  const pages = [];
  let page = -1;
  let shelfY = 0;
  let shelfH = 0;
  let x = 0;
  const newPage = () => {
    page++;
    pages.push({ width: pageWidth, height: 0 });
    shelfY = 0;
    shelfH = 0;
    x = 0;
  };
  newPage();
  for (const i of order) {
    const w = Math.min(pageWidth, Math.ceil(sizes[i].w)) + padding;
    const h = Math.ceil(sizes[i].h) + padding;
    if (x + w > pageWidth) {
      // scaffale pieno: se ne apre uno sotto (o una pagina nuova)
      shelfY += shelfH;
      shelfH = 0;
      x = 0;
    }
    if (shelfY + h > maxPageHeight && (shelfY > 0 || x > 0)) newPage();
    places[i] = { page, x, y: shelfY };
    x += w;
    shelfH = Math.max(shelfH, h);
    pages[page].height = Math.max(pages[page].height, shelfY + shelfH);
  }
  for (const p of pages) p.height = Math.min(maxPageHeight, nextPow2(p.height));
  return { pages, places };
}

/** Altezze delle pagine a potenze di 2: le texture si ripetono e filtrano senza sorprese su ogni GPU. */
function nextPow2(n) {
  let p = 16;
  while (p < n) p *= 2;
  return p;
}
