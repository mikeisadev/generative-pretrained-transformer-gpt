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
  }

  /** Le due previsioni e la loro combinazione con la scala s. Restituisce this.eps. */
  predict(zt, alphaBar, scale) {
    oracleEps(this.epsU, zt, this.zUncond, alphaBar);
    oracleEps(this.epsC, zt, this.zCond, alphaBar);
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
