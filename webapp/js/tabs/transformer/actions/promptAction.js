/**
 * Azione "Dal prompt all'output": il forward vero del modello sul prompt dell'utente, passo per
 * passo, dentro la scena 3D. Il primo token si segue in ogni dettaglio (tokenizzazione,
 * embedding, ogni blocco, uscita, estrazione); i token successivi scorrono più veloci.
 *
 * I dati sono calcolati tutti in `build` (promptRun.js); i passi si limitano a mostrarli.
 * Ogni `enter` imposta lo stato completo del passo, così "passo indietro" funziona sempre.
 */
import * as THREE from "three";
import * as B from "./promptBoards.js";
import { runPrompt } from "./promptRun.js";
import { ContextColumn, FlyingToken, LinkLines, RowMarks, StoiTable, TokenTape, disposeObject } from "./tokens3d.js";
import { CELL } from "../../../core/visuals.js";
import { visibleChar } from "./math.js";

/** Durate (secondi a velocità 1×): il primo blocco con calma, gli altri più svelti. */
const SLOW = 4.2;
const MEDIUM = 2.6;
const FAST = 1.9;

export function createPromptAction({ tab, stage, charts, getOptions, onOutput }) {
  let scene = null; // oggetti 3D dell'azione, creati in build e distrutti a fine azione
  let patternsKey = ""; // quali pattern di attention del prompt sono ora nella scena

  /**
   * Pattern di attention nella scena: quelli VERI del token k per i primi `blocks` blocchi,
   * quelli d'esempio di model.json per gli altri. Ogni passo dichiara il suo stato, così anche
   * saltando avanti e indietro la scena mostra solo i blocchi che il forward ha già attraversato.
   */
  function scenePatterns(run, k, blocks) {
    const key = `${k}:${blocks}`;
    if (key === patternsKey) return;
    patternsKey = key;
    stage.textures.restore(); // in questa azione le uniche texture toccate sono i pattern
    const s = run.steps[k];
    for (let l = 0; l < Math.min(blocks, s.attention.length); l++) s.attention[l].forEach((att, h) => stage.textures.writePattern(l, h, att, s.T));
  }

  /** Crea gli oggetti 3D e li posiziona rispetto alla colonna dei token di architecture.js. */
  function buildScene(run) {
    const tokens = tab.arch.registry.get("tokens");
    const anchor = tokens.getWorldPosition(new THREE.Vector3());
    const tape = new TokenTape(run.chars.length + run.steps.length, { cols: 16 });
    tape.setText(run.sequence.map((id) => run.vocab[id]), run.sequence);
    const { width, height } = tape.size();
    tape.group.position.set(anchor.x - width - 4.5, anchor.y + Math.max(height / 2, 2.5), 0.4);
    const stoi = new StoiTable(run.vocab);
    stoi.group.position.set(tape.group.position.x + width / 2 - 0.3, tape.group.position.y - height - stoi.h / 2 - 0.6, 0.4);
    const column = new ContextColumn(run.config.blockSize);
    column.group.position.copy(anchor);
    const links = new LinkLines(96);
    const tokMarks = new RowMarks(run.config.blockSize, 0xfbbf24);
    const posMarks = new RowMarks(run.config.blockSize, 0x22d3ee);
    const flyer = new FlyingToken(0.5);
    stage.overlay.add(tape.group, stoi.group, column.group, links.lines, tokMarks.mesh, posMarks.mesh, flyer.group);
    stoi.group.visible = false;
    column.group.visible = false;
    return { tape, stoi, column, links, tokMarks, posMarks, flyer };
  }

  function disposeScene() {
    if (!scene) return;
    for (const o of [scene.tape.group, scene.stoi.group, scene.column.group, scene.links.lines, scene.tokMarks.mesh, scene.posMarks.mesh, scene.flyer.group]) {
      stage.overlay.remove(o);
      disposeObject(o);
    }
    scene = null;
  }

  /** Stato della striscia del testo quando sono già stati generati `k` token. */
  function tapeState(run, k, { ids = true, current = -1 } = {}) {
    const n = run.promptIds.length;
    const total = n + k;
    const start = Math.max(0, total - run.config.blockSize);
    scene.tape.setVisible(total, 1);
    scene.tape.setStates((i) => (i === current ? "current" : i < start ? "outside" : i >= n ? "generated" : "context"));
    scene.tape.setIds(ids ? total : 0);
    scene.tape.setBracket(start, total);
    scene.tape.redraw();
  }

  /** Colonna dei token che il modello vede per il token k. */
  function columnState(run, k, highlight = -1) {
    const s = run.steps[k];
    scene.column.group.visible = true;
    scene.column.set(s.contextIds.map((id) => run.vocab[id]), s.contextIds, highlight);
  }

  /** Fili e righe evidenziate per il lookup negli embedding. */
  function prepareLookup(run, which) {
    const s = run.steps[0];
    const table = tab.arch.registry.get(which === "tok" ? "tok_emb" : "pos_emb");
    const marks = which === "tok" ? scene.tokMarks : scene.posMarks;
    const rows = which === "tok" ? s.contextIds : s.contextIds.map((_, t) => t);
    const ends = marks.mark(table, which === "tok" ? run.config.V : run.config.blockSize, rows, CELL);
    scene.links.clear();
    const a = new THREE.Vector3();
    ends.forEach((end, t) => scene.links.add(scene.column.worldPosition(t, a), end));
    scene.links.lines.material.color.setHex(which === "tok" ? 0xfbbf24 : 0x22d3ee);
    return ends.length;
  }

  function clearLookup() {
    scene.links.clear();
    scene.tokMarks.reveal(0, 0);
    scene.posMarks.reveal(0, 0);
  }

  /** Passo generico "componente + lavagna + codice + pacchetto". */
  function componentStep({ title, caption, duration, focus, station, draw, code, side = "auto", patterns, onEnter }) {
    return {
      title,
      caption,
      duration,
      patterns: [0, patterns ?? Infinity],
      enter: (ctx) => {
        const run = ctx.data.run;
        clearLookup();
        stage.hideBoard();
        charts.showToken(0, { appendSeries: false }); // il forward del primo token: nessuna estrazione ancora
        if (station) stage.packetAt(stage.distanceOf(station), station);
        onEnter?.(run, ctx);
        stage.show(focus, draw ? (board) => draw(board, run) : null, { side });
        if (code) tab.showCode(code[0], code[1], code[2] ?? []);
      },
    };
  }

  function blockSteps(run, l) {
    const p = `blocks.${l}`;
    const d = l === 0 ? SLOW : MEDIUM;
    const name = `Blocco ${l}`;
    const before = { patterns: l }; // la softmax di questo blocco non è ancora avvenuta
    const after = { patterns: l + 1 };
    const steps = [
      componentStep({ ...before, title: `${name}: LayerNorm 1`, caption: `${name}. LayerNorm 1: ogni vettore viene normalizzato (media 0, varianza 1) prima dell'attention.`, duration: d, focus: `${p}.ln1`, station: `${p}.ln1`, draw: (b, r) => B.drawLayerNorm(b, r, l, 1), code: ["block", "Block.forward: LayerNorm 1", ["self.ln1(x)"]], onEnter: () => charts.setBlock(l) }),
      componentStep({ ...before, title: `${name}: Q, K, V`, caption: `${name}. Ogni head calcola query, key e value con le sue tre matrici 64 → 16.`, duration: d, focus: `${p}.heads`, station: `${p}.heads`, draw: (b, r) => B.drawQKV(b, r, l), code: ["head", "Head: key, query, value", ["self.key(x)", "self.query(x)"]] }),
      componentStep({ ...before, title: `${name}: punteggi q·k`, caption: `${name}. Punteggi di affinità: query · key, divisi per √16 = 4.`, duration: d, focus: `${p}.heads`, draw: (b, r) => B.drawScores(b, r, l, false), code: ["head", "Head: punteggi scalati", ["wei = q @ k.transpose"]] }),
      componentStep({ ...before, title: `${name}: maschera causale`, caption: `${name}. Maschera causale: il futuro vale −∞ e non può essere guardato.`, duration: d, focus: `${p}.heads`, draw: (b, r) => B.drawScores(b, r, l, true), code: ["head", "Head: maschera causale", ["masked_fill"]] }),
      componentStep({
        ...after,
        title: `${name}: softmax`,
        caption: `${name}. Softmax: ogni riga diventa un insieme di pesi che sommano a 1. I pattern nella scena ora sono quelli VERI del tuo prompt.`,
        duration: d,
        focus: `${p}.heads`,
        draw: (b, r) => B.drawSoftmax(b, r, l),
        code: ["head", "Head: softmax", ["F.softmax(wei"]],
      }),
      componentStep({ ...after, title: `${name}: media dei value`, caption: `${name}. Ogni token raccoglie i value dei token che ascolta: out = att · V.`, duration: d, focus: `${p}.heads`, draw: (b, r) => B.drawWeighted(b, r, l), code: ["head", "Head: media pesata dei value", ["return wei @ v"]] }),
      componentStep({ ...after, title: `${name}: concat e proiezione`, caption: `${name}. Le 4 head si concatenano (4 × 16 = 64) e la proiezione le rimescola.`, duration: d, focus: `${p}.proj`, station: `${p}.proj`, draw: (b, r) => B.drawProjection(b, r, l), code: ["mha", "MultiHeadAttention.forward", ["torch.cat", "self.proj(out)"]] }),
      componentStep({ ...after, title: `${name}: residuo 1`, caption: `${name}. Connessione residua: x = x + attention.`, duration: d, focus: `${p}.add1`, station: `${p}.add1`, draw: (b, r) => B.drawResidual(b, r, l, 1), code: ["block", "Block.forward: residuo", ["x = x + self.sa"]] }),
      componentStep({ ...after, title: `${name}: LayerNorm 2`, caption: `${name}. LayerNorm 2 prima del feed-forward.`, duration: d * 0.8, focus: `${p}.ln2`, station: `${p}.ln2`, draw: (b, r) => B.drawLayerNorm(b, r, l, 2), code: ["block", "Block.forward: LayerNorm 2", ["self.ln2(x)"]] }),
      componentStep({ ...after, title: `${name}: feed-forward`, caption: `${name}. Feed-forward: 64 → 256 neuroni (ReLU) → 64, per ogni token separatamente.`, duration: d, focus: `${p}.ffwd`, station: `${p}.ffwd`, draw: (b, r) => B.drawFeedForward(b, r, l), code: ["ffwd", "FeedForward", ["nn.Linear(config.n_embd, hidden)", "nn.ReLU()", "nn.Linear(hidden, config.n_embd)"]] }),
      componentStep({ ...after, title: `${name}: residuo 2`, caption: `${name}. x = x + feed-forward: è l'uscita del blocco.`, duration: d * 0.8, focus: `${p}.add2`, station: `${p}.add2`, draw: (b, r) => B.drawResidual(b, r, l, 2), code: ["block", "Block.forward: residuo", ["x = x + self.ffwd"]] }),
    ];
    return steps;
  }

  function wheelStep(run, k, { title, caption, duration, fast = false }) {
    let wheel = null;
    return {
      title,
      caption,
      duration,
      patterns: [k, Infinity], // nei passi veloci i pattern seguono il contesto di ogni nuovo token
      enter: (ctx) => {
        clearLookup();
        const s = run.steps[k];
        tapeState(run, k);
        columnState(run, k);
        stage.setSceneProbs(s.probs, s.id);
        charts.showToken(k);
        onOutput(run, k + 1);
        scene.flyer.group.visible = false;
        stage.show("prediction", (board) => (wheel = fast ? B.drawFastToken(board, run, k) : B.drawWheel(board, run, k)));
        tab.showCode("generate", "generate: estrazione del token", ["torch.multinomial"]);
        if (fast) stage.packetAt(0, "tokens");
        else stage.packetAt(stage.distanceOf("next_token"), "next_token");
      },
      update: (p) => {
        if (!wheel) return;
        // la "pallina" gira qualche volta e si ferma su u
        const spin = Math.min(1, p / (fast ? 0.7 : 0.8));
        const eased = 1 - (1 - spin) ** 3;
        const pos = (wheel.u + 2 * (1 - eased)) % 1; // due giri, poi si ferma esattamente su u
        stage.board.needleAt(wheel.x0 + pos * wheel.w, wheel.top - 14, wheel.bottom + 14);
        if (fast) {
          // il pacchetto ripercorre tutto il modello: un forward completo per ogni token
          const d = Math.min(1, p / 0.7) * stage.length;
          stage.packetAt(d, stage.stationAt(d)?.id);
        }
      },
    };
  }

  function appendStep(run, k) {
    const from = new THREE.Vector3();
    return {
      title: `«${visibleChar(run.steps[k].char)}» entra nel contesto`,
      caption: "Il token estratto si aggiunge al testo e al contesto: al giro successivo il modello ricomincia il forward da capo.",
      duration: 3.2,
      patterns: [k, Infinity],
      enter: () => {
        clearLookup();
        const s = run.steps[k];
        tapeState(run, k);
        columnState(run, k);
        tab.arch.registry.get("next_token").getWorldPosition(from);
        scene.flyer.from.copy(from);
        scene.tape.worldPosition(run.promptIds.length + k, scene.flyer.to);
        scene.flyer.set(s.char);
        scene.flyer.group.visible = true;
        scene.flyer.fly(0);
        stage.show(scene.tape.bounds(run.promptIds.length + k + 1), (board) => B.drawAppend(board, run, k), { scale: 1.6 });
        tab.showCode("generate", "generate: il token si aggiunge", ["torch.cat((idx, idx_next)"]);
      },
      update: (p) => {
        scene.flyer.fly(Math.min(1, p / 0.75));
        if (p >= 0.75 && scene.tape.visible !== run.promptIds.length + k + 1) {
          scene.flyer.group.visible = false;
          tapeState(run, k + 1);
          columnState(run, Math.min(k + 1, run.steps.length - 1));
        }
      },
    };
  }

  function buildSteps(run) {
    const n = run.promptIds.length;
    const steps = [
      {
        title: "Il prompt",
        patterns: [0, 0],
        caption: "1. Il prompt: per il modello è solo una fila di caratteri. Ognuno diventerà un cubo, cioè un token.",
        duration: Math.min(5, Math.max(3, n * 0.05)),
        enter: () => {
          clearLookup();
          scene.stoi.group.visible = false;
          scene.column.group.visible = false;
          scene.tape.setIds(0);
          scene.tape.setBracket(0, 0);
          scene.tape.setStates(() => "idle");
          scene.tape.setVisible(0);
          stage.packetAt(0, null);
          stage.show(scene.tape.bounds(n), (board) => B.drawPrompt(board, run), { scale: 1.6 });
          tab.showCode("data.tokenizer", "CharTokenizer: un carattere = un token", ["def encode"]);
        },
        update: (p) => {
          const shown = p * n;
          scene.tape.setVisible(Math.min(n, Math.floor(shown) + 1), shown - Math.floor(shown));
        },
      },
      {
        title: "Tokenizzazione",
        patterns: [0, 0],
        caption: "2. Tokenizzazione: la tabella stoi assegna a ogni carattere il suo numero (id). Il testo diventa una lista di interi.",
        duration: Math.min(6, Math.max(3.5, n * 0.07)),
        enter: () => {
          clearLookup();
          scene.tape.setVisible(n);
          scene.tape.setStates(() => "idle");
          scene.tape.setBracket(0, 0);
          scene.stoi.group.visible = true;
          scene.stoi.draw(new Set());
          stage.show(scene.tape.bounds(n), null, { extra: [scene.stoi.plane.mesh] });
          tab.showCode("data.tokenizer", "CharTokenizer.encode", ["self.stoi: dict[str, int] = {ch: i", "return [self.stoi[ch]"]);
        },
        update: (p) => {
          const k = Math.min(n, Math.floor(p * (n + 1)));
          if (k === scene.tape.idsShown) return;
          scene.tape.setStates((i) => (i === k - 1 ? "current" : i < k ? "context" : "idle"));
          scene.tape.setIds(k);
          scene.stoi.draw(new Set(run.promptIds.slice(0, k)), k > 0 ? run.promptIds[k - 1] : -1);
        },
      },
      {
        title: "Taglio al contesto",
        patterns: [0, 0],
        caption: n > run.config.blockSize ? `3. Il modello vede al massimo ${run.config.blockSize} token: teniamo gli ultimi ${run.config.blockSize} del prompt.` : "3. Il prompt sta tutto nel contesto di 32 token.",
        duration: 4,
        enter: () => {
          clearLookup();
          scene.stoi.group.visible = true;
          scene.stoi.draw(new Set(run.promptIds));
          tapeState(run, 0);
          scene.column.group.visible = false;
          stage.show(scene.tape.bounds(n), (board) => B.drawCut(board, run), { scale: 1.6 });
          tab.showCode("generate", "generate: il contesto si taglia a block_size", ["idx_cond = idx"]);
        },
      },
      {
        title: "I token entrano nel modello",
        patterns: [0, 0],
        caption: "I token del contesto diventano l'ingresso del modello: idx, una colonna di id (B = 1, T ≤ 32).",
        duration: 3,
        enter: () => {
          clearLookup();
          tapeState(run, 0);
          columnState(run, 0);
          charts.showToken(0, { appendSeries: false });
          stage.packetAt(0, "tokens");
          stage.show(scene.column.plane.mesh, null);
          tab.showCode("model.forward", "GPTLanguageModel.forward", ["B, T = idx.shape"]);
        },
      },
      {
        title: "Token embedding",
        patterns: [0, 0],
        caption: "4. Lookup nella token embedding table: ogni id legge la SUA riga (evidenziata). Nessun calcolo, solo una lettura.",
        duration: SLOW,
        enter: () => {
          columnState(run, 0);
          scene.links.total = prepareLookup(run, "tok");
          scene.links.reveal(0);
          scene.tokMarks.reveal(0, 0);
          stage.packetAt(stage.distanceOf("embeddings"), "embeddings");
          stage.show("tok_emb", (board) => B.drawLookup(board, run, "tok"), { extra: [scene.column.plane.mesh] });
          tab.showCode("model.forward", "forward: token embedding", ["tok_emb = self.token_embedding_table(idx)"]);
        },
        update: (p) => {
          const k = Math.ceil(Math.min(1, p / 0.8) * scene.links.total);
          scene.links.reveal(k);
          scene.tokMarks.reveal(k, scene.links.total);
        },
      },
      {
        title: "Position embedding",
        patterns: [0, 0],
        caption: "5. Lookup nella position embedding table: la posizione t legge la riga t. Così il modello sa l'ordine dei caratteri.",
        duration: SLOW,
        enter: () => {
          columnState(run, 0);
          scene.tokMarks.reveal(0, 0);
          scene.links.total = prepareLookup(run, "pos");
          scene.links.reveal(0);
          stage.packetAt(stage.distanceOf("embeddings"), "embeddings");
          stage.show("pos_emb", (board) => B.drawLookup(board, run, "pos"), { extra: [scene.column.plane.mesh] });
          tab.showCode("model.forward", "forward: position embedding", ["pos = torch.arange", "pos_emb = self.position_embedding_table(pos)"]);
        },
        update: (p) => {
          const k = Math.ceil(Math.min(1, p / 0.8) * scene.links.total);
          scene.links.reveal(k);
          scene.posMarks.reveal(k, scene.links.total);
        },
      },
      componentStep({ patterns: 0, title: "Somma degli embedding", caption: "6. Somma: x = tok_emb + pos_emb. Questo vettore attraverserà tutti i blocchi.", duration: SLOW, focus: "embed_add", station: "embed_add", draw: (b, r) => B.drawSum(b, r), code: ["model.forward", "forward: somma", ["x = tok_emb + pos_emb"]] }),
    ];
    for (let l = 0; l < run.config.L; l++) steps.push(...blockSteps(run, l));
    steps.push(
      componentStep({ title: "LayerNorm finale", caption: "LayerNorm finale (ln_f): l'ultima normalizzazione prima della previsione.", duration: MEDIUM, focus: "ln_f", station: "ln_f", draw: (b, r) => B.drawFinalNorm(b, r), code: ["model.forward", "forward: LayerNorm finale", ["x = self.ln_f(x)"]] }),
      componentStep({ title: "lm_head → logits", caption: "lm_head: 65 punteggi (logits) per ogni posizione. Per generare serve solo l'ultima.", duration: SLOW, focus: "lm_head", station: "lm_head", draw: (b, r) => B.drawLogits(b, r), code: ["model.forward", "forward: lm_head", ["logits = self.lm_head(x)"]] }),
      componentStep({ title: "Temperatura", caption: `Divisione per la temperatura (${run.options.temperature}): decide quanto la distribuzione sarà "appuntita".`, duration: SLOW, focus: "logits", station: "logits", draw: (b, r) => B.drawTemperature(b, r, 0), code: ["generate", "generate: temperatura", ["logits = logits[:, -1, :] / temperature"]] }),
      componentStep({
        title: "Softmax",
        caption: "Softmax: 65 probabilità che sommano a 1. Le barre nella scena ora mostrano la distribuzione VERA del tuo prompt.",
        duration: SLOW,
        focus: "logits",
        station: "logits",
        draw: (b, r) => B.drawSoftmaxOut(b, r, 0),
        code: ["generate", "generate: softmax", ["probs = F.softmax"]],
        onEnter: (r) => stage.setSceneProbs(r.steps[0].probs, -1),
      }),
      wheelStep(run, 0, { title: "Estrazione del token", caption: "Estrazione: un numero casuale u tra 0 e 1 cade su uno spicchio della ruota. Gli spicchi più grandi vengono scelti più spesso, ma non sempre.", duration: 5.5 }),
      appendStep(run, 0),
    );
    for (let k = 1; k < run.steps.length; k++) {
      steps.push(wheelStep(run, k, { title: `Token ${k + 1}: «${visibleChar(run.steps[k].char)}»`, caption: `Token ${k + 1} di ${run.steps.length}: un forward completo sul contesto aggiornato, poi una nuova estrazione. Ora a passo veloce.`, duration: FAST, fast: true }));
    }
    steps.push({
      title: "Testo finale",
      patterns: [run.steps.length - 1, Infinity],
      caption: `Fatto: ${run.steps.length} caratteri generati, uno alla volta, ognuno con un forward completo. Il modello non "pianifica" la frase: sceglie solo il prossimo carattere.`,
      duration: 6,
      enter: () => {
        clearLookup();
        tapeState(run, run.steps.length, { ids: false });
        scene.stoi.group.visible = false;
        onOutput(run, run.steps.length);
        charts.showToken(run.steps.length - 1);
        stage.packetAt(stage.distanceOf("next_token"), "next_token");
        stage.show(scene.tape.bounds(run.sequence.length), (board) => B.drawFinal(board, run), { scale: 1.6 });
        tab.showCode("generate", "generate: il ciclo completo", ["for _ in range(max_new_tokens)"]);
      },
    });
    return steps;
  }

  /** Stato globale dell'azione: lo riapplica ogni passo (serve anche dopo "Ricomincia" a fine azione). */
  function activate(step, run) {
    stage.takeFlow();
    stage.lockDetail();
    stage.hide("tokens");
    scene.tape.group.visible = true;
    if (step.patterns) scenePatterns(run, ...step.patterns);
  }

  /** Avvolge l'enter di ogni passo con activate(). */
  const withActivate = (steps) => steps.map((step) => ({ ...step, enter: (ctx) => (activate(step, ctx.data.run), step.enter?.(ctx)) }));

  /** Fine azione: la scena di base torna com'era; gli oggetti dell'azione si nascondono (non si distruggono). */
  function finish() {
    stage.restore();
    patternsKey = "";
    if (!scene) return;
    for (const o of [scene.tape.group, scene.stoi.group, scene.column.group, scene.flyer.group]) o.visible = false;
    clearLookup();
    stage.board.hideNeedle();
  }

  function cleanup() {
    stage.restore();
    patternsKey = "";
    disposeScene();
  }

  return {
    id: "prompt",
    title: "Dal prompt all'output",
    category: "Prompt (inferenza)",
    summary: "Il tuo prompt attraversa davvero il modello: token, embedding, attention di ogni blocco, logits, temperatura, estrazione. Poi i token successivi, più veloci.",
    async build(ctx) {
      const options = getOptions();
      const engine = await tab.getEngine();
      await tab.views.activate("architettura");
      if (ctx.isStale?.()) return [];
      cleanup();
      const run = runPrompt(engine, options.text, options);
      ctx.data.run = run;
      charts.reset(run);
      charts.reveal(); // il pannello potrebbe essere rimasto sulla sezione Training
      scene = buildScene(run);
      return withActivate(buildSteps(run));
    },
    onEnd: finish,
    onStop: cleanup,
  };
}
