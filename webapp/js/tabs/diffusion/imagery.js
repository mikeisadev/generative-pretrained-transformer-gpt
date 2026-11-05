/**
 * Immagini generate proceduralmente nel browser (nessun file scaricato) e un VAE SIMULATO.
 *
 * - renderScene(): "un faro su una scogliera al tramonto", disegnato pixel per pixel.
 *   Valori RGB in [−1, 1] come nei modelli di diffusione (DDPM scala i pixel da 0…255 a −1…1).
 * - encodeLatent() / LatentDecoder: un finto autoencoder con fattore 8 e 4 canali, la stessa FORMA
 *   del latente di Stable Diffusion. Il VAE vero è una rete addestrata e il suo latente non è
 *   leggibile a occhio: qui usiamo una media 8×8 e una matrice fissa di colori, così il latente
 *   resta "guardabile". È dichiarato come simulazione didattica in tutta l'interfaccia.
 * Nessuna dipendenza da DOM: si può provare in Node.
 */

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
const mix = (a, b, t) => a + (b - a) * t;

/** Parametri casuali di una "variante" della scena (usati per simulare frame generati separatamente). */
export function sceneVariant(rng) {
  return {
    towerX: 0.24 + (rng() - 0.5) * 0.08,
    sunHue: (rng() - 0.5) * 0.35,
    stripes: 3 + Math.floor(rng() * 3),
    cloudShift: rng(),
    seaTint: (rng() - 0.5) * 0.25,
  };
}

const BASE_VARIANT = { towerX: 0.24, sunHue: 0, stripes: 4, cloudShift: 0.3, seaTint: 0 };

/**
 * Disegna la scena in `out` (RGB intercalato, lunghezza size·size·3, valori in [−1, 1]).
 * @param {number} time 0…1: fa ruotare il fascio del faro, scendere il sole e muovere le onde (video)
 */
export function renderScene(out, size, { time = 0, variant = BASE_VARIANT } = {}) {
  const v0 = variant;
  const horizon = 0.62;
  const sunX = 0.7;
  const sunY = 0.47 + time * 0.08;
  const sunR = 0.085;
  const beam = -0.5 + time * 1.4; // angolo del fascio (radianti)
  const towerTop = 0.3;
  const cliffTop = 0.585;
  const lantern = { x: v0.towerX, y: towerTop - 0.02 };
  for (let j = 0; j < size; j++) {
    const v = (j + 0.5) / size;
    for (let i = 0; i < size; i++) {
      const u = (i + 0.5) / size;
      let r, g, b;
      if (v < horizon) {
        // cielo: viola in alto, arancione all'orizzonte
        const k = smooth(0.0, horizon, v);
        r = mix(0.2, 1.0, k);
        g = mix(0.12, 0.5 + v0.sunHue * 0.3, k * k);
        b = mix(0.38, 0.28, k);
        // nuvole allungate
        const cy = 0.2 + 0.06 * Math.sin((u + v0.cloudShift) * 9.0);
        const cloud = Math.exp(-(((v - cy) / 0.025) ** 2)) * smooth(0.25, 0.75, Math.sin((u * 3 + v0.cloudShift * 5) * 2.1) * 0.5 + 0.5);
        r = mix(r, 0.95, cloud * 0.55);
        g = mix(g, 0.55, cloud * 0.55);
        b = mix(b, 0.6, cloud * 0.55);
        // sole con alone
        const d = Math.hypot(u - sunX, (v - sunY) * 1.05);
        const glow = Math.exp(-((d / 0.2) ** 2)) * 0.45;
        r += glow;
        g += glow * (0.7 + v0.sunHue);
        b += glow * 0.3;
        if (d < sunR) {
          r = 1.0;
          g = 0.86 + v0.sunHue * 0.2;
          b = 0.5;
        }
      } else {
        // mare: blu scuro con riflesso del sole a strisce (onde)
        const k = smooth(horizon, 1, v);
        r = mix(0.2, 0.05, k);
        g = mix(0.2, 0.12, k) + v0.seaTint * 0.2;
        b = mix(0.38, 0.28, k) + v0.seaTint * 0.3;
        const waves = 0.5 + 0.5 * Math.sin(v * 160 - Math.sin(u * 30) * 1.5 + time * Math.PI * 4);
        const reflect = Math.exp(-(((u - sunX) / (0.05 + (v - horizon) * 0.35)) ** 2)) * waves * (1 - k * 0.7);
        r += reflect * 0.9;
        g += reflect * 0.55;
        b += reflect * 0.2;
      }
      // fascio di luce del faro (additivo, attenuato con la distanza)
      const dx = u - lantern.x;
      const dy = v - lantern.y;
      const ang = Math.atan2(dy, dx);
      const dist = Math.hypot(dx, dy);
      const beamA = Math.exp(-(((ang - beam) / 0.07) ** 2)) * Math.exp(-dist * 2.2) * smooth(0.02, 0.06, dist);
      r += beamA * 0.6;
      g += beamA * 0.55;
      b += beamA * 0.35;
      // scogliera: silhouette a sinistra, il bordo scende verso il mare
      const edge = 0.42 + 0.05 * Math.sin(v * 23) + (v - cliffTop) * 0.35;
      const top = cliffTop + 0.015 * Math.sin(u * 40) + Math.max(0, u - 0.33) * 0.9;
      if (u < edge && v > top) {
        const shade = 0.5 + 0.5 * Math.sin(u * 60 + v * 25);
        r = 0.16 + shade * 0.05;
        g = 0.09 + shade * 0.03;
        b = 0.13 + shade * 0.04;
      }
      // faro: torre rastremata a strisce bianche e rosse, lanterna e tetto
      const half = mix(0.022, 0.032, (v - towerTop) / (cliffTop - towerTop));
      if (v > towerTop && v < cliffTop + 0.01 && Math.abs(u - v0.towerX) < half) {
        const band = Math.floor(((v - towerTop) / (cliffTop - towerTop)) * v0.stripes * 2) % 2;
        const side = 0.75 + 0.25 * ((u - v0.towerX) / half);
        if (band === 0) {
          r = 0.92 * side;
          g = 0.88 * side;
          b = 0.84 * side;
        } else {
          r = 0.78 * side;
          g = 0.16 * side;
          b = 0.16 * side;
        }
      }
      if (v > towerTop - 0.04 && v <= towerTop && Math.abs(u - v0.towerX) < 0.026) {
        r = 1.0;
        g = 0.92;
        b = 0.55;
      }
      const roofH = (towerTop - 0.04 - v) / 0.045;
      if (roofH >= 0 && roofH <= 1 && Math.abs(u - v0.towerX) < 0.032 * (1 - roofH)) {
        r = 0.12;
        g = 0.1;
        b = 0.14;
      }
      const o = (j * size + i) * 3;
      out[o] = clamp(r) * 2 - 1;
      out[o + 1] = clamp(g) * 2 - 1;
      out[o + 2] = clamp(b) * 2 - 1;
    }
  }
  return out;
}

/** Media su blocchi f×f di un'immagine RGB intercalata: restituisce (size/f)²·3 valori. */
export function avgPool(img, size, f) {
  const s = size / f;
  const out = new Float32Array(s * s * 3);
  const n = f * f;
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const o = ((Math.floor(j / f) * s + Math.floor(i / f)) * 3);
      const p = (j * size + i) * 3;
      out[o] += img[p] / n;
      out[o + 1] += img[p + 1] / n;
      out[o + 2] += img[p + 2] / n;
    }
  }
  return out;
}

/** Ingrandimento bilineare di una mappa (s×s, `ch` canali intercalati) a size×size: scrive in out. */
function upsampleBilinear(out, src, s, size, ch) {
  const scale = s / size;
  for (let j = 0; j < size; j++) {
    const y = clamp((j + 0.5) * scale - 0.5, 0, s - 1);
    const y0 = Math.floor(y);
    const y1 = Math.min(s - 1, y0 + 1);
    const fy = y - y0;
    for (let i = 0; i < size; i++) {
      const x = clamp((i + 0.5) * scale - 0.5, 0, s - 1);
      const x0 = Math.floor(x);
      const x1 = Math.min(s - 1, x0 + 1);
      const fx = x - x0;
      for (let c = 0; c < ch; c++) {
        const a = src[(y0 * s + x0) * ch + c];
        const b = src[(y0 * s + x1) * ch + c];
        const d = src[(y1 * s + x0) * ch + c];
        const e = src[(y1 * s + x1) * ch + c];
        out[(j * size + i) * ch + c] = mix(mix(a, b, fx), mix(d, e, fx), fy);
      }
    }
  }
  return out;
}

// Matrice fissa RGB → 3 canali "di colore" del latente simulato (luminanza, caldo/freddo, verde/magenta).
const M = [
  [0.6, 0.8, 0.3],
  [0.9, 0.0, -0.9],
  [-0.4, 0.9, -0.4],
];
function invert3(m) {
  const [a, b, c] = m[0];
  const [d, e, f] = m[1];
  const [g, h, i] = m[2];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}
const MINV = invert3(M);

/**
 * VAE simulato con fattore 8 e 4 canali. Il latente è "channels first": [c][y][x], lunghezza 4·s·s.
 * Il canale 3 misura quanto è dettagliata (contrastata) ogni zona 8×8. I canali sono riscalati per avere deviazione standard ~1, come fa il
 * `scale_factor` 0,18215 di Stable Diffusion con il latente vero.
 */
export class SimulatedVAE {
  constructor(reference, size, factor = 8) {
    this.size = size;
    this.factor = factor;
    this.s = size / factor;
    this.ref = reference;
    const pooled = avgPool(reference, size, factor);
    this.baseUp = upsampleBilinear(new Float32Array(size * size * 3), pooled, this.s, size, 3);
    this.detail = new Float32Array(size * size * 3);
    for (let k = 0; k < this.detail.length; k++) this.detail[k] = reference[k] - this.baseUp[k];
    this.scales = [1, 1, 1, 1];
    const raw = this.rawEncode(reference);
    for (let c = 0; c < 4; c++) {
      const n = this.s * this.s;
      let m = 0;
      let v = 0;
      for (let k = 0; k < n; k++) m += raw[c * n + k] / n;
      for (let k = 0; k < n; k++) v += (raw[c * n + k] - m) ** 2 / n;
      this.scales[c] = 1 / Math.max(Math.sqrt(v), 1e-3);
    }
    this.z0 = this.encode(reference);
    // buffer di lavoro del decoder (niente allocazioni durante le animazioni)
    this.small = new Float32Array(this.s * this.s * 3);
    this.up = new Float32Array(size * size * 3);
    this.gainSmall = new Float32Array(this.s * this.s);
    this.gainUp = new Float32Array(size * size);
  }

  rawEncode(img) {
    const { size, factor, s } = this;
    const pooled = avgPool(img, size, factor);
    const out = new Float32Array(4 * s * s);
    const n = s * s;
    for (let k = 0; k < n; k++) {
      const rgb = [pooled[k * 3], pooled[k * 3 + 1], pooled[k * 3 + 2]];
      for (let c = 0; c < 3; c++) out[c * n + k] = M[c][0] * rgb[0] + M[c][1] * rgb[1] + M[c][2] * rgb[2];
    }
    // canale 3: contrasto locale della luminanza dentro ogni blocco f×f
    for (let by = 0; by < s; by++) {
      for (let bx = 0; bx < s; bx++) {
        let m = 0;
        let q = 0;
        for (let y = 0; y < factor; y++) {
          for (let x = 0; x < factor; x++) {
            const p = ((by * factor + y) * size + bx * factor + x) * 3;
            const l = 0.3 * img[p] + 0.59 * img[p + 1] + 0.11 * img[p + 2];
            m += l;
            q += l * l;
          }
        }
        const nn = factor * factor;
        m /= nn;
        out[3 * n + by * s + bx] = Math.sqrt(Math.max(0, q / nn - m * m)) * 3 - 0.3;
      }
    }
    return out;
  }

  /** Immagine → latente (4 × s × s). */
  encode(img) {
    const out = this.rawEncode(img);
    const n = this.s * this.s;
    for (let c = 0; c < 4; c++) for (let k = 0; k < n; k++) out[c * n + k] *= this.scales[c];
    return out;
  }

  /**
   * Latente → immagine RGB (size × size). Ricostruisce colori e forme dal latente; i dettagli fini
   * vengono dall'immagine di riferimento e compaiono solo dove il latente è vicino a quello vero:
   * con il latente pulito restituisce esattamente l'immagine.
   */
  decode(out, z) {
    const { s, size, small, up, gainSmall, gainUp, scales, z0 } = this;
    const n = s * s;
    for (let k = 0; k < n; k++) {
      const c0 = z[k] / scales[0];
      const c1 = z[n + k] / scales[1];
      const c2 = z[2 * n + k] / scales[2];
      for (let c = 0; c < 3; c++) small[k * 3 + c] = MINV[c][0] * c0 + MINV[c][1] * c1 + MINV[c][2] * c2;
      // i dettagli fini compaiono solo dove il latente somiglia a quello vero (simulazione)
      let err = 0;
      for (let c = 0; c < 4; c++) err += Math.abs(z[c * n + k] - z0[c * n + k]) / 4;
      gainSmall[k] = clamp(1 - 1.2 * err, 0, 1);
    }
    upsampleBilinear(up, small, s, size, 3);
    upsampleBilinear(gainUp, gainSmall, s, size, 1);
    const detail = this.detail;
    for (let p = 0; p < size * size; p++) {
      const gain = gainUp[p];
      for (let c = 0; c < 3; c++) out[p * 3 + c] = clamp(up[p * 3 + c] + gain * detail[p * 3 + c], -1, 1);
    }
    return out;
  }
}

/**
 * Immagine "generica" (sfocata, poco satura): ciò che il denoiser simulato prevede SENZA prompt.
 * Serve alla classifier-free guidance simulata.
 */
export function genericImage(img, size, radius = Math.round(size / 10)) {
  const tmp = new Float32Array(img.length);
  const out = new Float32Array(img.length);
  const blur = (src, dst, dx, dy) => {
    for (let j = 0; j < size; j++) {
      for (let i = 0; i < size; i++) {
        let r = 0;
        let g = 0;
        let b = 0;
        let n = 0;
        for (let k = -radius; k <= radius; k++) {
          const x = clamp(i + k * dx, 0, size - 1);
          const y = clamp(j + k * dy, 0, size - 1);
          const p = (y * size + x) * 3;
          r += src[p];
          g += src[p + 1];
          b += src[p + 2];
          n++;
        }
        const o = (j * size + i) * 3;
        dst[o] = r / n;
        dst[o + 1] = g / n;
        dst[o + 2] = b / n;
      }
    }
  };
  blur(img, tmp, 1, 0);
  blur(tmp, out, 0, 1);
  // desatura e avvicina al grigio medio
  for (let p = 0; p < size * size; p++) {
    const l = 0.3 * out[p * 3] + 0.59 * out[p * 3 + 1] + 0.11 * out[p * 3 + 2];
    for (let c = 0; c < 3; c++) out[p * 3 + c] = mix(out[p * 3 + c], l, 0.65) * 0.6 - 0.1;
  }
  return out;
}
