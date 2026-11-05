/**
 * Azione "Uno step di training al microscopio": batch → forward → loss → backward → AdamW,
 * con i dati veri del training registrato.
 *
 * Che cosa è vero e che cosa no (lo diciamo anche nelle didascalie):
 * - pesi: gli snapshot veri (step di partenza scelto e snapshot successivo);
 * - batch: finestre estratte da un estratto vero di input.txt (non è il batch esatto usato
 *   quello step, che il training non ha salvato); loss e gradiente sui logits sono calcolati
 *   davvero dal motore JS su questo batch;
 * - norme dei gradienti e rapporti update/peso: misurati durante il training (trace.json);
 * - l'aggiornamento dei pesi mostrato nella scena copre i passi veri tra due snapshot, perché
 *   un singolo passo cambia i pesi dello 0,1% circa e non si vedrebbe.
 */
import * as T from "./trainingBoards.js";
import { BackwardParticles } from "./particles.js";
import { TrainingData, batchLoss, groupOfStation, metricIndex, sampleBatch } from "./trainingData.js";
import { TRAINING_EXCERPT } from "./excerpt.js";
import { ContextColumn, disposeObject } from "./tokens3d.js";
import { lerpInto, logitsGradient, rateLimiter, visibleChar } from "./math.js";
import * as THREE from "three";

/** Aggiornamenti al secondo delle heatmap mentre i pesi si animano (scrivere tutte le texture costa ~5 ms). */
const UPLOADS_PER_SECOND = 12;
const OPTIMIZER_SECONDS = 7;
/** Numeri all'italiana nelle didascalie. */
const it = (v) => v.toFixed(3).replace(".", ",");

export function createTrainingStepAction({ tab, stage, charts, data, getStartStep }) {
  let scene = null;

  async function prepare() {
    const trace = await data.trace();
    const steps = trace.snapshots.map((s) => s.step);
    const s0 = getStartStep();
    const i0 = Math.max(0, Math.min(steps.length - 2, steps.indexOf(s0)));
    const [wA, wB, engine] = await Promise.all([data.snapshot(steps[i0]), data.snapshot(steps[i0 + 1]), data.trainingEngine()]);
    const blockSize = engine.config.block_size;
    const ids = engine.encode(TRAINING_EXCERPT);
    const windows = sampleBatch(ids, blockSize, trace.train_config.batch_size, trace.train_config.seed + steps[i0]);
    engine.setWeights(wA);
    const lossA = batchLoss(engine, windows);
    engine.setWeights(wB);
    const lossB = batchLoss(engine, windows);
    const V = engine.config.vocab_size;
    const w0 = windows[0];
    const last = blockSize - 1;
    const grad = logitsGradient(lossA.perWindow[0].logits, last * V, V, w0.y[last], windows.length * blockSize);
    const mi = metricIndex(trace.metrics.steps, steps[i0]);
    const pick = (byGroup) => Object.fromEntries(trace.groups.map((g) => [g, byGroup[g][mi]]));
    return {
      trace,
      engine,
      s0: steps[i0],
      s1: steps[i0 + 1],
      snapA: trace.snapshots[i0],
      snapB: trace.snapshots[i0 + 1],
      wA,
      wB,
      viewsA: TrainingData.views(engine, wA),
      viewsB: TrainingData.views(engine, wB),
      mix: new Float32Array(wA.length),
      windows,
      lossA,
      lossB,
      blockSize,
      vocab: engine.vocab,
      text: TRAINING_EXCERPT,
      logitGrad: { grad, target: w0.y[last], context: w0.x.slice(-8).map((id) => visibleChar(engine.vocab[id])).join("") },
      groups: trace.groups,
      gradStep: trace.metrics.steps[mi],
      gradByGroup: pick(trace.metrics.grad_norm_by_group),
      updateByGroup: pick(trace.metrics.update_ratio_by_group),
      updateGlobal: trace.metrics.update_ratio[mi],
    };
  }

  function buildScene(m) {
    const particles = new BackwardParticles(tab.arch.path);
    const maxGrad = Math.max(...Object.values(m.gradByGroup));
    // intensità lungo il binario: la norma del gruppo della stazione attraversata (dato vero)
    particles.setIntensity((d) => {
      const station = stage.stationAt(d);
      const g = station ? groupOfStation(station.id) : "token_embedding";
      return Math.sqrt((m.gradByGroup[g] ?? 0) / maxGrad);
    });
    const column = new ContextColumn(m.blockSize);
    column.group.position.copy(tab.arch.registry.get("tokens").getWorldPosition(new THREE.Vector3()));
    column.group.visible = false;
    stage.overlay.add(particles.points, column.group);
    return { particles, column };
  }

  function disposeScene() {
    if (!scene) return;
    stage.overlay.remove(scene.particles.points, scene.column.group);
    scene.particles.dispose();
    disposeObject(scene.column.group);
    scene = null;
  }

  /** Stato comune a ogni passo (serve anche dopo "Ricomincia" a fine azione). */
  function base(m, weights = "A") {
    stage.takeFlow();
    stage.lockDetail();
    stage.hide("tokens");
    stage.textures.writeWeights(weights === "A" ? m.viewsA : m.viewsB);
    scene.particles.points.visible = false;
    charts.highlight(weights === "A" ? m.s0 : m.s1);
  }

  function showColumn(m) {
    const w = m.windows[0];
    scene.column.group.visible = true;
    scene.column.set(w.x.map((id) => m.vocab[id]), w.x);
  }

  function steps(m) {
    const uploads = rateLimiter(UPLOADS_PER_SECOND);
    return [
      {
        title: `Step ${m.s0}: il punto di partenza`,
        caption: `Carichiamo i pesi veri dello snapshot ${m.s0}: le heatmap della scena ora mostrano quei pesi. È il modello "a metà strada" su cui faremo uno step di training.`,
        duration: 5,
        enter: () => {
          base(m);
          scene.column.group.visible = false;
          stage.packetAt(0, null);
          stage.show("blocks.0", (b) => T.drawStart(b, m));
          tab.showCode("train", "train(): il ciclo di training", ["for step in range(train_cfg.max_iters)"]);
        },
      },
      {
        title: "Estrazione del batch",
        caption: `get_batch: ${m.windows.length} finestre casuali di ${m.blockSize + 1} caratteri dal testo di training (qui un estratto vero di input.txt).`,
        duration: 5,
        enter: () => {
          base(m);
          scene.column.group.visible = false;
          stage.packetAt(0, null);
          stage.show("embeddings", (b) => T.drawBatch(b, m));
          tab.showCode("data.get_batch", "get_batch", ["ix = torch.randint", "x = torch.stack", "y = torch.stack"]);
        },
      },
      {
        title: "Input e target",
        caption: "x sono i caratteri letti, y gli stessi spostati di uno: ogni posizione deve prevedere il carattere che viene dopo.",
        duration: 5,
        enter: () => {
          base(m);
          showColumn(m);
          stage.packetAt(0, "tokens");
          stage.show(scene.column.plane.mesh, (b) => T.drawInputTarget(b, m));
          tab.showCode("data.get_batch", "get_batch: i target", ["y = torch.stack", "y[b, t] è il token che segue"]);
        },
      },
      {
        title: "Forward",
        caption: `Forward: il batch attraversa tutto il modello con i pesi dello step ${m.s0}. Escono ${m.windows.length} × ${m.blockSize} distribuzioni di ${m.vocab.length} logits.`,
        duration: 5,
        enter: () => {
          base(m);
          showColumn(m);
          stage.show("prediction", (b) => T.drawForward(b, m));
          tab.showCode("model.forward", "forward con i target", ["logits = self.lm_head(x)", "next_token_loss(logits, targets)"]);
        },
        update: (p) => {
          const d = Math.min(1, p / 0.85) * stage.length;
          stage.packetAt(d, stage.stationAt(d)?.id);
        },
      },
      {
        title: "Loss",
        caption: `Cross-entropy: −ln della probabilità data al carattere giusto, in media su ${m.windows.length * m.blockSize} previsioni. Su questo batch vale ${it(m.lossA.mean)}.`,
        duration: 6,
        enter: () => {
          base(m);
          stage.packetAt(stage.distanceOf("logits"), "logits");
          stage.show("prediction", (b) => T.drawLoss(b, m));
          tab.showCode("loss", "next_token_loss: la cross-entropy", ["F.cross_entropy", "ln(V)"]);
        },
      },
      {
        title: "Gradiente sui logits",
        caption: "Il primo passo della backward: per ogni logit, quanto e in che direzione spostarlo. Il carattere giusto riceve una spinta verso l'alto.",
        duration: 5,
        enter: () => {
          base(m);
          stage.packetAt(stage.distanceOf("lm_head"), "lm_head");
          stage.show("lm_head", (b) => T.drawLogitGrad(b, m));
          tab.showCode("train", "train(): backward", ["loss.backward()"]);
        },
      },
      {
        title: "Backward",
        caption: "loss.backward(): i gradienti scendono dall'uscita verso gli embedding (particelle). La luminosità segue la norma vera del gradiente di ogni gruppo.",
        duration: 7,
        enter: () => {
          base(m);
          scene.particles.points.visible = true;
          scene.particles.update(0);
          stage.packetAt(stage.length, null);
          stage.show("blocks.1", (b) => T.drawBackward(b, m), { extra: [tab.arch.registry.get("blocks.2"), tab.arch.registry.get("blocks.0")] });
          tab.showCode("train", "train(): backward", ["optimizer.zero_grad", "loss.backward()"]);
        },
        update: (p, dt) => {
          scene.particles.points.visible = true;
          scene.particles.update(dt);
          stage.packetAt(stage.length * (1 - p), stage.stationAt(stage.length * (1 - p))?.id);
        },
        exit: () => (scene.particles.points.visible = false),
      },
      {
        title: "AdamW: i pesi cambiano",
        caption: `optimizer.step(): ogni peso si sposta un poco contro il suo gradiente. Un passo solo non si vedrebbe: la scena anima i ${m.s1 - m.s0} passi veri fino allo snapshot ${m.s1}.`,
        duration: OPTIMIZER_SECONDS,
        enter: () => {
          base(m);
          uploads.reset();
          stage.packetAt(0, null);
          stage.show("blocks.0", (b) => T.drawOptimizer(b, m));
          tab.showCode("train", "train(): passo dell'ottimizzatore", ["optimizer.step()", "torch.optim.AdamW"]);
        },
        update: (p) => {
          const t = Math.min(1, Math.max(0, (p - 0.1) / 0.75));
          if (!uploads.ready(p >= 1)) return;
          lerpInto(m.mix, m.wA, m.wB, t);
          stage.textures.writeWeights(m.mixViews);
          charts.highlight(Math.round(m.s0 + (m.s1 - m.s0) * t));
        },
      },
      {
        title: "Di nuovo la loss",
        caption: `Stesso batch, pesi dello step ${m.s1}: loss ${it(m.lossB.mean)} (prima ${it(m.lossA.mean)}). Questo, ripetuto ${m.trace.train_config.max_iters} volte, è il training.`,
        duration: 7,
        enter: () => {
          base(m, "B");
          stage.packetAt(stage.distanceOf("logits"), "logits");
          stage.show("prediction", (b) => T.drawResult(b, m));
          tab.showCode("train", "train(): e si ricomincia", ["for step in range(train_cfg.max_iters)", "estimate_loss"]);
        },
      },
    ];
  }

  function cleanup() {
    stage.restore();
    disposeScene();
    data.trainingEngine().then((e) => e.restoreWeights()).catch(() => {});
  }

  return {
    id: "training-step",
    title: "Uno step di training al microscopio",
    category: "Training",
    summary: "Batch, forward, loss, backward e AdamW con i numeri veri del training registrato. Scegli sotto lo step di partenza.",
    async build(ctx) {
      await tab.views.activate("architettura");
      const m = await prepare();
      if (ctx.isStale?.()) return [];
      cleanup();
      m.mixViews = TrainingData.views(m.engine, m.mix);
      ctx.data.m = m;
      scene = buildScene(m);
      charts.fill(Infinity);
      return steps(m);
    },
    // a fine azione la scena torna ai pesi addestrati; "Ricomincia" la riprepara dal primo passo
    onEnd: () => {
      stage.restore();
      if (scene) scene.column.group.visible = scene.particles.points.visible = false;
    },
    onStop: cleanup,
  };
}
