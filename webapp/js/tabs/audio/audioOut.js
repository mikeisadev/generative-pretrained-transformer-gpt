/**
 * Riproduzione con la Web Audio API: un solo AudioContext per il tab, creato al primo clic
 * (i browser bloccano l'audio che parte senza un gesto dell'utente).
 *
 *   const out = new AudioOut();
 *   await out.play(samples, 16000, "Originale");
 *   out.position()   // secondi dall'inizio, o null se non sta suonando (per la testina nella scena)
 */
export class AudioOut extends EventTarget {
  constructor() {
    super();
    this.context = null;
    this.source = null;
    this.startedAt = 0;
    this.duration = 0;
    this.label = "";
  }

  /** Suona un Float32Array mono. Interrompe ciò che stava suonando. */
  async play(samples, sampleRate, label = "") {
    this.stop();
    this.context ??= new AudioContext();
    if (this.context.state === "suspended") await this.context.resume();
    const buffer = this.context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    const gain = this.context.createGain();
    gain.gain.value = 0.9;
    source.connect(gain).connect(this.context.destination);
    source.onended = () => {
      if (this.source !== source) return;
      this.source = null;
      this.dispatchEvent(new CustomEvent("change", { detail: { playing: false, label } }));
    };
    source.start();
    this.source = source;
    this.startedAt = this.context.currentTime;
    this.duration = buffer.duration;
    this.label = label;
    this.dispatchEvent(new CustomEvent("change", { detail: { playing: true, label } }));
  }

  stop() {
    if (!this.source) return;
    const source = this.source;
    this.source = null;
    try {
      source.stop();
    } catch {
      /* già fermo */
    }
    this.dispatchEvent(new CustomEvent("change", { detail: { playing: false, label: this.label } }));
  }

  get playing() {
    return this.source !== null;
  }

  /** Secondi trascorsi dall'inizio della riproduzione (null se non suona). */
  position() {
    if (!this.source || !this.context) return null;
    return Math.min(this.duration, this.context.currentTime - this.startedAt);
  }

  dispose() {
    this.stop();
    this.context?.close();
  }
}
