/**
 * Due brevi brani generati con la sintesi additiva: ogni nota è una somma di sinusoidi
 * (fondamentale + armoniche) con un inviluppo di volume. Servono come "dataset" del codec
 * giocattolo e del modello che genera token.
 *
 * Modulo PURO: restituisce Float32Array, la riproduzione è in player.js.
 */
import { normalizePeak } from "./dsp.js";

export const SAMPLE_RATE = 16000; // la stessa frequenza di campionamento che usa Whisper

/** Numero MIDI → frequenza in Hz (la4 = 69 = 440 Hz, temperamento equabile). */
export const midiToHz = (m) => 440 * Math.pow(2, (m - 69) / 12);

const NOTE_NAMES = ["do", "do#", "re", "re#", "mi", "fa", "fa#", "sol", "sol#", "la", "la#", "si"];
export const noteName = (m) => `${NOTE_NAMES[m % 12]}${Math.floor(m / 12) - 1}`;

/**
 * I due stili. Gli accordi sono [nome, note MIDI]; la melodia è una lista di [nota, durata in battiti]
 * (null = pausa). Entrambi durano 8 battiti = 4,8 s a 100 bpm.
 */
export const STYLES = {
  allegra: {
    id: "allegra",
    prompt: "melodia allegra in do maggiore, note veloci",
    bpm: 100,
    chords: [
      ["Do", [48, 52, 55]],
      ["Sol", [43, 47, 50]],
      ["La-", [45, 48, 52]],
      ["Fa", [41, 45, 48]],
    ],
    melody: [
      [72, 0.5], [76, 0.5], [79, 0.5], [76, 0.5],
      [74, 0.5], [71, 0.5], [74, 0.5], [79, 0.5],
      [72, 0.5], [76, 0.5], [81, 0.5], [76, 0.5],
      [77, 0.5], [72, 0.5], [69, 0.5], [72, 0.5],
    ],
  },
  calma: {
    id: "calma",
    prompt: "melodia calma in la minore, note lunghe",
    bpm: 100,
    chords: [
      ["La-", [45, 48, 52]],
      ["Fa", [41, 45, 48]],
      ["Do", [48, 52, 55]],
      ["Mi", [40, 44, 47]],
    ],
    melody: [
      [69, 1], [72, 1],
      [65, 1.5], [64, 0.5],
      [67, 1], [64, 1],
      [68, 1.5], [null, 0.5],
    ],
  },
};

/** Inviluppo ADSR semplificato: attacco rapido, decadimento, sostegno, rilascio. */
function adsr(t, length, { attack = 0.01, decay = 0.12, sustain = 0.55, release = 0.08 } = {}) {
  if (t < 0 || t > length + release) return 0;
  if (t < attack) return t / attack;
  if (t < attack + decay) return 1 - ((1 - sustain) * (t - attack)) / decay;
  if (t <= length) return sustain;
  return sustain * (1 - (t - length) / release);
}

/** Aggiunge una nota (somma di armoniche) al buffer. */
function addNote(out, { midi, start, length, gain, harmonics, rolloff, envelope }) {
  const f0 = midiToHz(midi);
  const sr = SAMPLE_RATE;
  const a = Math.max(0, Math.floor(start * sr));
  const b = Math.min(out.length, Math.ceil((start + length + (envelope?.release ?? 0.08)) * sr));
  for (let h = 1; h <= harmonics; h++) {
    const f = f0 * h;
    if (f >= sr / 2) break; // oltre Nyquist l'armonica si ripiegherebbe (aliasing)
    const amp = gain / Math.pow(h, rolloff);
    const w = (2 * Math.PI * f) / sr;
    for (let i = a; i < b; i++) out[i] += amp * adsr(i / sr - start, length, envelope) * Math.sin(w * (i - a));
  }
}

/**
 * Sintetizza il brano di uno stile.
 * @returns {{samples: Float32Array, sampleRate: number, duration: number, notes: object[], chords: object[]}}
 *          notes/chords: tempi in secondi, per le etichette della scena
 */
export function synthesize(style) {
  const beat = 60 / style.bpm;
  const beats = style.melody.reduce((s, [, d]) => s + d, 0);
  const duration = beats * beat;
  const out = new Float32Array(Math.round(duration * SAMPLE_RATE));
  const chords = [];
  const chordBeats = beats / style.chords.length;
  style.chords.forEach(([name, notes], i) => {
    const start = i * chordBeats * beat;
    const length = chordBeats * beat - 0.06;
    for (const midi of notes) addNote(out, { midi, start, length, gain: 0.16, harmonics: 4, rolloff: 1.6, envelope: { attack: 0.03, decay: 0.3, sustain: 0.6, release: 0.06 } });
    addNote(out, { midi: notes[0] - 12, start, length, gain: 0.22, harmonics: 3, rolloff: 1.2 }); // basso: la fondamentale un'ottava sotto
    chords.push({ name, start, end: start + chordBeats * beat });
  });
  const notes = [];
  let t = 0;
  for (const [midi, d] of style.melody) {
    if (midi !== null) {
      addNote(out, { midi, start: t, length: d * beat - 0.04, gain: 0.32, harmonics: 6, rolloff: 1.3 });
      notes.push({ midi, name: noteName(midi), start: t, end: t + d * beat });
    }
    t += d * beat;
  }
  normalizePeak(out, 0.8);
  return { samples: out, sampleRate: SAMPLE_RATE, duration, notes, chords };
}
