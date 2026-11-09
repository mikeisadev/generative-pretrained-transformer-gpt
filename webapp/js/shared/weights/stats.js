/**
 * Statistiche di un tensore di pesi (logica pura, Float32Array o array di numeri).
 *
 * Convenzioni, uguali al Python del corso:
 * - deviazione standard "di popolazione" (divisa per n, `std(correction=0)` come in
 *   gpt/recording.py per weight_std_by_group);
 * - percentili con interpolazione lineare tra i valori ordinati (il metodo predefinito di numpy);
 * - NaN e ±Infinity non entrano nelle statistiche: si contano a parte (`nan`, `inf`), perché un
 *   solo NaN renderebbe NaN anche media e norma e lo studente non capirebbe dove sta il problema.
 * Le somme usano numeri JavaScript (float64): sui 16.384 pesi di una matrice l'errore è trascurabile.
 */

/** Oltre questa dimensione i percentili si stimano dall'istogramma fine (ordinare costerebbe troppo). */
const EXACT_PERCENTILES_MAX = 2_000_000;
const FINE_BINS = 4096;
export const DEFAULT_PERCENTILES = Object.freeze([1, 5, 25, 50, 75, 95, 99]);

/**
 * @param {ArrayLike<number>} data
 * @param {{nearZero?: number, bins?: number, percentiles?: number[], histogram?: boolean}} [o]
 *   nearZero: soglia di |w| sotto cui un peso conta come "quasi zero" (predefinita 0,001)
 * @returns {{count: number, finite: number, nan: number, inf: number, mean: number, std: number,
 *   min: number, max: number, absMax: number, absMean: number, l2: number, rms: number,
 *   nearZero: {threshold: number, count: number, fraction: number},
 *   percentiles: Record<number, number>, approximate: boolean,
 *   histogram: {min: number, max: number, binWidth: number, counts: Uint32Array}|null}}
 */
export function tensorStats(data, { nearZero = 1e-3, bins = 40, percentiles: ps = DEFAULT_PERCENTILES, histogram: wantHistogram = true } = {}) {
  const n = data.length;
  let finite = 0;
  let nan = 0;
  let inf = 0;
  let sum = 0;
  let sumAbs = 0;
  let sumSq = 0;
  let min = Infinity;
  let max = -Infinity;
  let small = 0;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    if (v !== v) {
      nan++;
      continue;
    }
    if (v === Infinity || v === -Infinity) {
      inf++;
      continue;
    }
    finite++;
    sum += v;
    const a = v < 0 ? -v : v;
    sumAbs += a;
    sumSq += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (a < nearZero) small++;
  }
  const mean = finite ? sum / finite : NaN;
  // Secondo passaggio per la varianza: più stabile di E[x²] − E[x]² quando la media è grande.
  let sq = 0;
  if (finite) {
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (Number.isFinite(v)) sq += (v - mean) * (v - mean);
    }
  }
  const empty = finite === 0;
  const result = {
    count: n,
    finite,
    nan,
    inf,
    mean,
    std: empty ? NaN : Math.sqrt(sq / finite),
    min: empty ? NaN : min,
    max: empty ? NaN : max,
    absMax: empty ? NaN : Math.max(-min, max),
    absMean: empty ? NaN : sumAbs / finite,
    l2: Math.sqrt(sumSq),
    rms: empty ? NaN : Math.sqrt(sumSq / finite),
    nearZero: { threshold: nearZero, count: small, fraction: empty ? NaN : small / finite },
    percentiles: {},
    approximate: false,
    histogram: null,
  };
  if (!empty && ps.length) {
    const exact = finite <= EXACT_PERCENTILES_MAX;
    result.percentiles = exact ? exactPercentiles(data, finite, ps) : approximatePercentiles(data, min, max, finite, ps);
    result.approximate = !exact;
  }
  if (wantHistogram && !empty) result.histogram = histogram(data, { bins, min, max });
  return result;
}

/** Percentili esatti (interpolazione lineare, come numpy.percentile). */
export function percentiles(data, ps = DEFAULT_PERCENTILES) {
  let finite = 0;
  for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i])) finite++;
  return finite ? exactPercentiles(data, finite, ps) : Object.fromEntries(ps.map((p) => [p, NaN]));
}

function exactPercentiles(data, finite, ps) {
  const sorted = new Float64Array(finite);
  let k = 0;
  for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i])) sorted[k++] = data[i];
  sorted.sort();
  const out = {};
  for (const p of ps) {
    const pos = (Math.min(100, Math.max(0, p)) / 100) * (finite - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(finite - 1, lo + 1);
    out[p] = sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  return out;
}

function approximatePercentiles(data, min, max, finite, ps) {
  const h = histogram(data, { bins: FINE_BINS, min, max });
  const out = {};
  for (const p of ps) {
    const target = (Math.min(100, Math.max(0, p)) / 100) * (finite - 1);
    let seen = 0;
    let b = 0;
    while (b < FINE_BINS - 1 && seen + h.counts[b] <= target) seen += h.counts[b++];
    const within = h.counts[b] ? (target - seen) / h.counts[b] : 0;
    out[p] = h.min + (b + within) * h.binWidth;
  }
  return out;
}

/**
 * Istogramma su [min, max] con `bins` intervalli uguali; l'ultimo comprende max. I valori fuori
 * dall'intervallo (se min/max sono dati, es. per sovrapporre due tensori) vanno nel primo o
 * nell'ultimo intervallo e si contano anche in `below` / `above`.
 * @returns {{min: number, max: number, binWidth: number, counts: Uint32Array, below: number, above: number, total: number}}
 */
export function histogram(data, { bins = 40, min = null, max = null } = {}) {
  if (min === null || max === null) {
    const r = range([data]);
    min ??= r.min;
    max ??= r.max;
  }
  const counts = new Uint32Array(Math.max(1, bins | 0));
  const nb = counts.length;
  let below = 0;
  let above = 0;
  let total = 0;
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: NaN, max: NaN, binWidth: NaN, counts, below, above, total };
  const width = max > min ? (max - min) / nb : 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    total++;
    let b;
    if (width === 0) b = 0;
    else {
      b = Math.floor((v - min) / width);
      if (v < min) below++;
      else if (v > max) above++;
    }
    counts[b < 0 ? 0 : b >= nb ? nb - 1 : b]++;
  }
  return { min, max, binWidth: width, counts, below, above, total };
}

/** Minimo e massimo (valori finiti) di più array insieme: l'intervallo comune per sovrapporre istogrammi. */
export function range(arrays) {
  let min = Infinity;
  let max = -Infinity;
  for (const data of arrays) {
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < min) min = v; // NaN non passa nessun confronto
      if (v > max) max = v;
    }
  }
  if (min === Infinity) return { min: NaN, max: NaN };
  // ±Infinity non sono un intervallo utile: si ricalcola senza.
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = Infinity;
    max = -Infinity;
    for (const data of arrays) for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i])) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }
    if (min === Infinity) return { min: NaN, max: NaN };
  }
  return { min, max };
}

/** Il massimo di |v| (valori finiti): la scala simmetrica delle heatmap divergenti. */
export function absMax(data) {
  let m = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    const a = v < 0 ? -v : v;
    if (a > m && a !== Infinity) m = a;
  }
  return m;
}

/** Norma L2 (radice della somma dei quadrati), ignorando i valori non finiti. */
export function l2Norm(data) {
  let s = 0;
  for (let i = 0; i < data.length; i++) if (Number.isFinite(data[i])) s += data[i] * data[i];
  return Math.sqrt(s);
}
