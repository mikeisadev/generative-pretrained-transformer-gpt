/**
 * Denoiser SIMULATO per le animazioni (simulazione didattica, dichiarata nell'interfaccia).
 *
 * Una rete vera, dopo l'addestramento, stima il rumore ε guardando z_t, t e il testo.
 * Qui non c'è nessuna rete: il simulatore conosce già il latente finale e calcola il rumore
 * "perfetto" con la formula inversa del processo diretto (math.oracleEps). Per mostrare la
 * classifier-free guidance servono due previsioni diverse:
 *   - senza prompt  → punta a un'immagine generica (sfocata, poco satura)
 *   - con il prompt → punta a metà strada fra la generica e il riferimento,
 * scelte in modo che con la scala di guidance di riferimento (7,5, il default dello script
 * txt2img di Stable Diffusion 1.x) si ottenga esattamente l'immagine di riferimento.
 * Così si vede l'effetto della scala: s = 1 dà un'immagine slavata, s molto alto la satura.
 *
 * Realismo: una rete addestrata, quando il rumore è tanto, non può sapere i dettagli e prevede
 * una media sfocata; i dettagli arrivano man mano che il rumore cala. Il simulatore imita questo
 * comportamento sfocando la sua stima di x0 in proporzione al rapporto rumore/segnale.
 */
import { genericImage } from "./imagery.js";
import { cfgCombine, ddimStep, oracleEps } from "./math.js";

export const REFERENCE_SCALE = 7.5;

export class DenoiserSim {
  /** @param {import("./imagery.js").SimulatedVAE} vae @param {Float32Array} reference */
  constructor(vae, reference) {
    this.vae = vae;
    const n = vae.z0.length;
    this.zUncond = vae.encode(genericImage(reference, vae.size));
    this.zCond = new Float32Array(n);
    for (let i = 0; i < n; i++) this.zCond[i] = this.zUncond[i] + (vae.z0[i] - this.zUncond[i]) / REFERENCE_SCALE;
    // buffer di lavoro riutilizzati a ogni passo
    this.epsU = new Float32Array(n);
    this.epsC = new Float32Array(n);
    this.eps = new Float32Array(n);
    this.x0hat = new Float32Array(n);
    this.x0u = new Float32Array(n);
    this.x0c = new Float32Array(n);
    this.tmp = new Float32Array(n);
  }

  /**
   * Stima "realistica" di x0: più rumore → più sfocata e più vicina alla media di ogni canale.
   * È lineare in `target`, quindi la guidance si combina come con le previsioni di rumore.
   */
  estimateX0(out, target, alphaBar) {
    const ratio = Math.sqrt((1 - alphaBar) / alphaBar); // rumore / segnale
    const radius = Math.min(10, Math.round(ratio * 2));
    const keep = 1 / (1 + 0.15 * ratio * ratio);
    const s = this.vae.s;
    const n = s * s;
    boxBlur(out, target, this.tmp, s, 4, radius);
    for (let c = 0; c < 4; c++) {
      let mean = 0;
      for (let k = 0; k < n; k++) mean += target[c * n + k] / n;
      for (let k = 0; k < n; k++) out[c * n + k] = mean + (out[c * n + k] - mean) * keep;
    }
    return out;
  }

  /** Le due previsioni e la loro combinazione con la scala s. Restituisce this.eps. */
  predict(zt, alphaBar, scale) {
    oracleEps(this.epsU, zt, this.estimateX0(this.x0u, this.zUncond, alphaBar), alphaBar);
    oracleEps(this.epsC, zt, this.estimateX0(this.x0c, this.zCond, alphaBar), alphaBar);
    return cfgCombine(this.eps, this.epsU, this.epsC, scale);
  }

  /** Un passo completo: previsione con guidance + passo DDIM. Scrive z_{t−1} in `out`. */
  step(out, zt, alphaBar, alphaBarPrev, scale) {
    this.predict(zt, alphaBar, scale);
    return ddimStep(out, zt, this.eps, alphaBar, alphaBarPrev, this.x0hat);
  }

  /** Latente finale che si otterrebbe con la scala s (forma chiusa: ẑ0 = z_u + s · (z_c − z_u)). */
  finalLatent(out, scale) {
    for (let i = 0; i < out.length; i++) out[i] = this.zUncond[i] + scale * (this.zCond[i] - this.zUncond[i]);
    return out;
  }
}

/** Sfocatura a box separabile di un tensore [c][y][x] (s × s); `tmp` è un buffer di lavoro. */
function boxBlur(out, src, tmp, s, channels, r) {
  if (r <= 0) {
    out.set(src);
    return out;
  }
  const n = s * s;
  const pass = (from, to, horizontal) => {
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < s; y++) {
        for (let x = 0; x < s; x++) {
          let sum = 0;
          let count = 0;
          for (let k = -r; k <= r; k++) {
            const xx = horizontal ? x + k : x;
            const yy = horizontal ? y : y + k;
            if (xx < 0 || yy < 0 || xx >= s || yy >= s) continue;
            sum += from[c * n + yy * s + xx];
            count++;
          }
          to[c * n + y * s + x] = sum / count;
        }
      }
    }
  };
  pass(src, tmp, true);
  pass(tmp, out, false);
  return out;
}
