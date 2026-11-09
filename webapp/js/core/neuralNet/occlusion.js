/**
 * Che cosa copre ciò che la camera guarda (logica pura, testata con node).
 *
 * Le lastre fantasma di un "× N" hanno l'ingombro intero del blocco ripetuto (alte e profonde quanto
 * lui): guardando di tre quarti da vicino la sezione che viene dopo, stanno proprio tra la camera e lei e
 * coprono metà dell'inquadratura. La scena le attenua quando il raggio dalla camera al punto guardato le
 * attraversa (o le sfiora, dentro un cono che si allarga fino a `roi` attorno al punto guardato). Di lato
 * il raggio corre lungo z e passa accanto ai fantasmi: lì restano piene. Se il punto guardato sta proprio
 * sui fantasmi (lo studente guarda loro) restano piene anche loro.
 */

/** Distanza di un punto [x, y, z] da un riquadro { min, max } (0 se è dentro). */
export function distanceToBox(p, box) {
  let s = 0;
  for (let k = 0; k < 3; k++) {
    const d = Math.max(box.min[k] - p[k], 0, p[k] - box.max[k]);
    s += d * d;
  }
  return Math.sqrt(s);
}

/**
 * Opacità da dare a un riquadro che potrebbe coprire il punto guardato.
 * @param {ArrayLike<number>} camera posizione della camera [x, y, z]
 * @param {ArrayLike<number>} focus punto guardato [x, y, z] (stesso sistema di coordinate del riquadro)
 * @param {{min: number[], max: number[]}} box il riquadro
 * @param {{ roi?: number, floor?: number, samples?: number }} [o] roi: raggio attorno al punto guardato
 *   (unità del mondo) che deve restare libero; floor: opacità quando copre del tutto
 * @returns {number} 1 = non copre; `floor` = sta proprio tra la camera e il punto guardato
 */
export function occluderOpacity(camera, focus, box, { roi = 1.5, floor = 0.12, samples = 24 } = {}) {
  const d = [focus[0] - camera[0], focus[1] - camera[1], focus[2] - camera[2]];
  const len = Math.hypot(d[0], d[1], d[2]);
  if (!(len > 1e-6) || distanceToBox(focus, box) <= roi) return 1;
  let occ = 0;
  const p = [0, 0, 0];
  for (let i = 1; i < samples && occ < 1; i++) {
    const t = i / samples;
    if ((1 - t) * len < roi) break; // davanti al punto guardato di almeno roi: quello che gli sta accanto non lo copre
    for (let k = 0; k < 3; k++) p[k] = camera[k] + d[k] * t;
    const gap = distanceToBox(p, box);
    occ = Math.max(occ, gap <= 0 ? 1 : 1 - gap / (roi * t));
  }
  return 1 - (1 - floor) * Math.min(1, Math.max(0, occ) / 0.6);
}
