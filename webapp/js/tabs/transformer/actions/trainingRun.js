/**
 * Azione "Il training completo (5000 step) in un minuto": riscorre la storia VERA del training.
 *
 * - Le heatmap dei pesi passano da uno snapshot al successivo (11 snapshot registrati; tra due
 *   snapshot i valori sono interpolati linearmente: lo diciamo nella didascalia).
 * - I pattern di attention nella scena sono calcolati davvero, con i pesi di ogni snapshot, sul
 *   prompt d'esempio di model.json; le barre di probabilità sono example_next_token_probs del trace.
 * - I grafici della sezione Training si riempiono man mano che lo step avanza.
 * - La lavagna mostra il testo che il modello genera a ogni snapshot (sample_text del trace).
 */
import * as T from "./trainingBoards.js";
import { TrainingData } from "./trainingData.js";
import { lerpInto } from "./math.js";

const SEGMENT_SECONDS = 5.2;
const UPLOADS_PER_SECOND = 12;
const CHART_UPDATES_PER_SECOND = 6;
/** Dove guarda la camera in ogni tratto: prima ciò che cambia di più, poi il resto del modello. */
const CAMERA = ["embeddings", "blocks.0.heads", "blocks.0.ffwd", "lm_head", "blocks.1.heads", "blocks.2.ffwd", "blocks.3.heads", "blocks.1.heads", "embeddings", "prediction"];

export function createTrainingRunAction({ tab, stage, charts, data }) {
  async function prepare() {
    const [trace, snapshots, engine] = await Promise.all([data.trace(), data.allSnapshots(), data.trainingEngine()]);
    const ids = engine.encode(tab.spec.example.prompt);
    const entries = trace.snapshots;
    // attention vera del prompt d'esempio con i pesi di ogni snapshot
    const attention = entries.map((s) => {
      engine.setWeights(snapshots.get(s.step));
      return engine.forward(ids, { trace: true }).trace.blocks.map((b) => b.heads.map((h) => h.att));
    });
    engine.restoreWeights();
    const { n_layer: L, n_head: H } = engine.config;
    return {
      trace,
      engine,
      snapshots: entries,
      weights: entries.map((s) => snapshots.get(s.step)),
      views: entries.map((s) => TrainingData.views(engine, snapshots.get(s.step))),
      attention,
      T: ids.length,
      L,
      H,
      mix: new Float32Array(engine.paramCount),
      attMix: new Float32Array(ids.length * ids.length),
      probsMix: new Float32Array(engine.config.vocab_size),
    };
  }

  /** Scena allo stato interpolato tra gli snapshot a e a+1 (t in 0..1). */
  function applyMix(r, a, t) {
    const b = Math.min(a + 1, r.snapshots.length - 1);
    lerpInto(r.mix, r.weights[a], r.weights[b], t);
    stage.textures.writeWeights(r.mixViews);
    for (let l = 0; l < r.L; l++) {
      for (let h = 0; h < r.H; h++) {
        lerpInto(r.attMix, r.attention[a][l][h], r.attention[b][l][h], t);
        stage.textures.writePattern(l, h, r.attMix, r.T);
      }
    }
    lerpInto(r.probsMix, r.snapshots[a].example_next_token_probs, r.snapshots[b].example_next_token_probs, t);
    let best = 0;
    for (let i = 1; i < r.probsMix.length; i++) if (r.probsMix[i] > r.probsMix[best]) best = i;
    stage.setSceneProbs(r.probsMix, best);
  }

  function base() {
    stage.takeFlow();
    stage.lockDetail();
  }

  function segment(r, a) {
    const sa = r.snapshots[a].step;
    const sb = r.snapshots[a + 1].step;
    let lastUpload = -1;
    let lastChart = -1;
    return {
      title: `Step ${sa} → ${sb}`,
      caption: `Dallo step ${sa} allo step ${sb}: ${sb - sa} passi di AdamW. Le heatmap passano dallo snapshot ${sa} al ${sb} (valori intermedi interpolati); pattern di attention e probabilità sono quelli veri di ogni snapshot.`,
      duration: SEGMENT_SECONDS,
      enter: () => {
        base();
        lastUpload = -1;
        lastChart = -1;
        applyMix(r, a, 0);
        charts.fill(sa);
        charts.setSnapshotWeights(sa, r.views[a]);
        stage.packetAt(0, null);
        stage.show(CAMERA[a % CAMERA.length], (board) => T.drawRunSegment(board, r, a, a + 1), { duration: 1.4 });
        tab.showCode("train", "train(): il ciclo completo", ["for step in range(train_cfg.max_iters)", "optimizer.step()"]);
      },
      update: (p, dt, ctx) => {
        const step = Math.round(sa + (sb - sa) * p);
        // il pacchetto fa un giro per ogni "colpo d'occhio": forward e backward si ripetono di continuo
        const d = ((p * 3) % 1) * stage.length;
        stage.packetAt(d, stage.stationAt(d)?.id, { caption: false });
        const upload = Math.floor(p * SEGMENT_SECONDS * UPLOADS_PER_SECOND);
        if (upload !== lastUpload || p >= 1) {
          lastUpload = upload;
          applyMix(r, a, p);
          const loss = r.trace.train_loss[Math.min(step, r.trace.train_loss.length - 1)];
          tab.setCaption(`Training: step ${step} di ${r.trace.train_config.max_iters} · loss del batch ${loss.toFixed(3).replace(".", ",")}`);
        }
        const tick = Math.floor(p * SEGMENT_SECONDS * CHART_UPDATES_PER_SECOND);
        if (tick !== lastChart || p >= 1) {
          lastChart = tick;
          charts.fill(step);
        }
      },
    };
  }

  function steps(r) {
    const last = r.snapshots.length - 1;
    const list = [
      {
        title: "Pesi casuali",
        caption: "Step 0: i pesi sono numeri casuali. Apri i Grafici: si riempiranno con la storia vera di questo training.",
        duration: 4,
        enter: () => {
          base();
          applyMix(r, 0, 0);
          charts.fill(0);
          charts.setSnapshotWeights(0, r.views[0]);
          stage.packetAt(0, null);
          stage.show("embeddings", (board) => T.drawRunSegment(board, r, 0, 1));
          tab.showCode("model.init", "I pesi iniziali", ["nn.Embedding", "nn.Linear"]);
        },
      },
    ];
    for (let a = 0; a < last; a++) list.push(segment(r, a));
    list.push({
      title: "Pesi finali",
      caption: "Fine: 5000 step. Questi pesi coincidono con quelli del checkpoint usato da tutta la web app.",
      duration: 6,
      enter: () => {
        base();
        applyMix(r, last, 0);
        charts.fill(Infinity);
        charts.setSnapshotWeights(r.snapshots[last].step, r.views[last]);
        stage.packetAt(stage.distanceOf("next_token"), "next_token");
        stage.show("prediction", (board) => T.drawRunEnd(board, r));
        tab.setCaption("Training completato: 5000 step.");
        tab.showCode("train", "train(): fine", ["return history"]);
      },
    });
    return list;
  }

  function cleanup() {
    stage.restore();
    charts.fill(Infinity);
  }

  return {
    id: "training-run",
    title: "Il training completo (5000 step) in un minuto",
    category: "Training",
    summary: "La storia vera del training: heatmap dei pesi che cambiano, pattern di attention che emergono, testo generato a ogni snapshot e tutti i grafici.",
    async build(ctx) {
      cleanup();
      await tab.views.activate("architettura");
      const r = await prepare();
      r.mixViews = TrainingData.views(r.engine, r.mix);
      ctx.data.r = r;
      tab.ui.charts?.open();
      charts.reveal();
      return steps(r);
    },
    onEnd: cleanup,
    onStop: cleanup,
  };
}
