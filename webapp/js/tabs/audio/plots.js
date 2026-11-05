/**
 * Grafici del tab audio (ChartsPanel del nucleo). Tutti i dati sono calcolati dal laboratorio;
 * durante le azioni i grafici si aggiornano insieme alla scena (con un limite di frequenza:
 * ridisegnare una heatmap a ogni frame sarebbe uno spreco).
 */
import { CODEC } from "./codec.js";
import { WHISPER, envelope } from "./dsp.js";
import { DISPLAY_MELS } from "./lab.js";

const MEL_SHOWN = [2, 10, 18, 26, 34, 42, 50, 58]; // 8 filtri su 64: uno per colore della palette
const fmtHz = (f) => (f >= 1000 ? `${(f / 1000).toLocaleString("it-IT", { maximumFractionDigits: 1 })} kHz` : `${Math.round(f)} Hz`);

export function createPlots(charts, lab) {
  if (!charts) return null;
  const T = lab.frames;
  const sound = charts.section("audio-sound", "Il suono", "Calcolati qui nel browser sul brano scelto nel Laboratorio.");
  const wave = sound.chart("wave", {
    type: "line",
    title: "Forma d'onda",
    description: "A cosa serve: è il segnale grezzo, l'altezza dell'onda di pressione nel tempo. Si vedono gli attacchi delle note e il volume.",
    xLabel: "tempo (s)",
    yLabel: "ampiezza",
    yMin: -1,
    yMax: 1,
  });
  const spec = sound.chart("spectrogram", {
    type: "heatmap",
    title: "Spettrogramma log-Mel",
    description: "A cosa serve: mostra quali frequenze suonano in ogni istante (64 bande Mel, decibel). Le righe orizzontali sono le note e le loro armoniche.",
    xLabel: "frame (20 ms)",
    yLabel: "banda Mel",
  });
  const bank = sound.chart("mel-bank", {
    type: "line",
    title: "Banco di filtri Mel (8 dei 64 filtri)",
    description: "A cosa serve: ogni triangolo somma l'energia delle frequenze vicine. In alto i triangoli sono più larghi (l'orecchio distingue meno) e più bassi (normalizzazione Slaney: area costante).",
    xLabel: "frequenza (Hz)",
    yLabel: "peso (×1000)",
  });

  const tok = charts.section("audio-tokens", "Token", "Codec giocattolo: k-means e RVQ calcolati nel browser.");
  const err = tok.chart("rvq-error", {
    type: "line",
    yScale: "log",
    title: "Errore di ricostruzione vs numero di codebook",
    description: "A cosa serve: dice quanto si perde comprimendo. 1 = rispondere sempre con il frame medio. Ogni codebook in più quantizza il residuo e l'errore scende.",
    xLabel: "codebook usati",
    yLabel: "errore relativo",
  });
  const dist = tok.chart("token-dist", {
    type: "bar",
    title: "Distribuzione dei token (codebook 1)",
    description: "A cosa serve: quante volte compare ciascuno dei 64 token nel brano. Pochi token molto frequenti = note tenute a lungo; i token mai usati servono all'altro brano.",
    xLabel: "token",
    yLabel: "frame",
    highlightLabel: "il più frequente",
  });
  const next = tok.chart("next-token", {
    type: "bar",
    percent: true,
    title: "Probabilità del prossimo token (codebook 1)",
    description: "A cosa serve: è l'uscita del modello a ogni passo, come le probabilità del prossimo carattere nel GPT. Il token scelto (campionato) è evidenziato.",
    xLabel: "token candidati (i 10 più probabili)",
    yLabel: "probabilità",
    highlightLabel: "scelto",
  });

  const ws = charts.section("audio-whisper", "Whisper", "L'ingresso di Whisper, calcolato con i suoi parametri.");
  const wmel = ws.chart("whisper-mel", {
    type: "heatmap",
    title: "log-Mel di Whisper (30 s × 80 bande)",
    description: "A cosa serve: è esattamente ciò che entra nell'encoder di Whisper. A destra il silenzio di riempimento: Whisper lavora sempre su blocchi da 30 secondi.",
    xLabel: "frame (10 ms)",
    yLabel: "banda Mel",
  });

  // --- dati fissi: banco di filtri e righe della heatmap
  const anyStyle = Object.keys(lab.songs)[0];
  const fb = lab.display[anyStyle].bank;
  const binHz = lab.sampleRate / CODEC.nFft;
  bank.setData({
    series: MEL_SHOWN.map((m) => ({
      name: `filtro ${m + 1} (${fmtHz(fb.centers[m])})`,
      points: Array.from({ length: fb.bins }, (_, k) => [k * binHz, fb.weights[m * fb.bins + k] * 1000]).filter(([, v], k, arr) => v > 0 || (arr[k + 1] && arr[k + 1][1] > 0) || (arr[k - 1] && arr[k - 1][1] > 0)),
    })),
  });
  const rowLabels = Array.from({ length: DISPLAY_MELS }, (_, i) => fmtHz(fb.centers[DISPLAY_MELS - 1 - i]));
  const specValues = new Float32Array(DISPLAY_MELS * T);
  let specStyle = null;
  let specShown = -1;
  let lastSpec = 0;

  function drawSpec(id, upTo) {
    const melDb = lab.display[id].melDb;
    for (let r = 0; r < DISPLAY_MELS; r++) {
      const band = DISPLAY_MELS - 1 - r; // riga 0 in alto = banda più acuta
      for (let t = 0; t < T; t++) specValues[r * T + t] = t < upTo ? melDb[t * DISPLAY_MELS + band] : NaN;
    }
    spec.setData({ rows: DISPLAY_MELS, cols: T, values: specValues, rowLabels });
    specShown = upTo;
  }

  const relErr = lab.relativeError();
  const errPoints = (upTo) => relErr.slice(0, upTo + 1).map((v, l) => [l, v]).filter(([l]) => l >= 1);

  let style = anyStyle;
  return {
    setStyle(id) {
      style = id;
      const env = envelope(lab.songs[id].samples, 480);
      const dur = lab.songs[id].duration;
      const points = [];
      for (let c = 0; c < 480; c++) {
        const x = (c / 479) * dur;
        points.push([x, env[2 * c]], [x, env[2 * c + 1]]);
      }
      wave.setData({ series: [{ name: "campioni (min e max ogni 10 ms)", points }] });
      specStyle = id;
      drawSpec(id, T);
      this.tokens(lab.codes[id], T);
      err.setData({ series: [{ name: "codec giocattolo", points: errPoints(CODEC.levels) }] });
      const lm = lab.whisper(id);
      const rows = WHISPER.nMels;
      const cols = WHISPER.nFrames;
      const values = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) for (let t = 0; t < cols; t++) values[r * cols + t] = lm.values[t * rows + (rows - 1 - r)];
      wmel.setData({ rows, cols, values });
    },
    /** Testina sulla forma d'onda (secondi) o null. */
    waveAt(t) {
      wave.setHighlight(t === null ? null : { x: t });
    },
    /** Spettrogramma rivelato fino al frame n (al più 12 ridisegni al secondo). */
    specUpTo(n, force = false) {
      const now = performance.now();
      if (!force && (n === specShown || now - lastSpec < 80)) return;
      lastSpec = now;
      drawSpec(specStyle ?? style, Math.round(n));
    },
    /** Errore con i primi `levels` codebook; evidenzia `q`. */
    errorUpTo(levels, q = null) {
      err.setData({ series: [{ name: "codec giocattolo", points: errPoints(levels) }] });
      err.setHighlight(q ? { x: q } : null);
    },
    /** Conteggio dei token del codebook 1 nei primi n frame. */
    tokens(codes, n) {
      const counts = new Array(lab.rvq.k).fill(0);
      for (let t = 0; t < n; t++) counts[codes[t]]++;
      let best = 0;
      counts.forEach((c, i) => c > counts[best] && (best = i));
      dist.setData({ labels: counts.map((_, i) => String(i)), values: counts, highlight: n > 0 ? best : undefined });
    },
    /** Distribuzione del prossimo token: i 10 più probabili, evidenziato quello scelto. */
    nextToken(p, chosen) {
      if (!p) return;
      const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 10);
      if (!order.includes(chosen)) order[9] = chosen;
      next.setData({ labels: order.map((i) => `tok ${i}`), values: order.map((i) => p[i]), highlight: order.indexOf(chosen) });
    },
  };
}
