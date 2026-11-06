/**
 * Vista "Rete neurale": tutti i neuroni del GPT del corso e tutte le connessioni con peso.
 *
 *   - colore dei neuroni = attivazione vera per il prompt e la posizione scelti (motore JS,
 *     engine.forward(ids, {trace: true}));
 *   - archi = i pesi veri (207.552 nel modello del corso) che collegano due neuroni (matrici + γ delle LayerNorm),
 *     con una soglia "mostra solo i più forti" e la possibilità di vederli tutti;
 *   - attention = archi tratteggiati calcolati dal pattern vero: NON sono pesi appresi;
 *   - clic su un neurone → scheda nella sidebar con la formula e i numeri veri.
 * Il modello (layout, archi, spiegazioni) è in neuronModel.js, la grafica in neuronScene.js.
 */
import * as THREE from "three";
import { el, fmt } from "../../../core/dom.js";
import { VocabularyError } from "../../../engine/gpt.js";
import { setBaseOpacity } from "../lod.js";
import { NeuronScene, HEAD_COLORS } from "./neuronScene.js";
import { buildEdges, buildNeuronLayout, explainNeuron, neuronValues, parameterBreakdown, ROLE } from "./neuronModel.js";
import { createViewInfo, deepBox, formula, legend, sources, texNum, topChartsSection } from "./panel.js";

const DISTANCES = { detail: 30, blocks: 95 }; // la rete è alta ~103 unità, colonne a 3-4 unità di distanza
/**
 * Rapporto larghezza/altezza della scena visibile per cui sono tarate le soglie (1600×900 con il codice
 * aperto). Su uno stage più stretto (telefono) la camera deve stare più lontana per inquadrare le stesse
 * colonne: le soglie crescono in proporzione, così la vista iniziale mostra gli strati anche lì
 * (regola "LOD" di ARCHITECTURE.md; la vista Contesto fa lo stesso).
 */
const REFERENCE_ASPECT = 0.76;
const FRACTIONS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1];
const DEFAULT_FRACTION = 3; // 2%
const FRONT = new THREE.Vector3(0.04, 0.1, 1).normalize(); // quasi di fronte: le colonne restano dritte
const shownChar = (c) => (c === " " ? "␣" : c === "\n" ? "↵" : c);
const pctText = (f) => `${(f * 100).toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`;

export function installNeuronsView(tab) {
  const { ui, spec, arch, flow } = tab;
  const { host, director } = ui;
  const info = createViewInfo(tab, "neuroni");
  // Archi con peso = tutti i parametri tranne bias e β (che appartengono al neurone): vedi neuronModel.js.
  const EDGE_COUNT = spec.params.filter((p) => !p.name.endsWith(".bias")).reduce((a, p) => a + p.numel, 0);
  const state = {
    active: false,
    layout: buildNeuronLayout(spec.config), // deterministico e leggero (4.387 posizioni): pronto subito
    scene: null,
    edges: null,
    edgesKey: null,
    engine: null,
    prompt: spec.example.prompt,
    ids: null,
    trace: null,
    t: 0,
    values: null,
    selected: null,
    fraction: FRACTIONS[DEFAULT_FRACTION],
    saved: null,
    unregister: null,
    offFrame: null,
  };

  // ---------------------------------------------------------------- interfaccia nella sidebar
  const status = el("p", { className: "tf-status", ariaLive: "polite" });
  // textarea e non input: il prompt d'esempio contiene un a capo, che un <input> cancellerebbe.
  const promptInput = el("textarea", { rows: 2, value: state.prompt, maxLength: spec.config.block_size, ariaLabel: `Prompt (al massimo ${spec.config.block_size} caratteri)`, spellcheck: false });
  const positions = el("div", { className: "tf-positions", role: "group", ariaLabel: "Posizione da visualizzare" });
  const fractionInput = el("input", { type: "range", min: 0, max: FRACTIONS.length - 1, step: 1, value: DEFAULT_FRACTION, ariaLabel: "Quota di archi da mostrare" });
  const fractionOut = el("output");
  const card = el("div", { className: "tf-card", hidden: true });
  const summary = el("p", { className: "hint" });
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runPrompt(promptInput.value);
  });
  fractionInput.addEventListener("input", () => setFraction(FRACTIONS[Number(fractionInput.value)]));

  function renderInfo() {
    const C = spec.config.n_embd;
    const hs = C / spec.config.n_head;
    info.replaceChildren(
      el("h3", {}, "Rete neurale: tutti i neuroni"),
      el("p", {}, "Qui lo stesso modello è disegnato come una rete neurale classica. Ogni pallino è un neurone: un singolo numero calcolato mentre il modello legge il prompt. Ogni linea è un peso: il numero appreso che dice quanto un neurone influenza il successivo."),
      el("p", {}, el("b", {}, "Neuroni"), " = attivazioni, cambiano a ogni prompt. ", el("b", {}, "Pesi"), " = archi, fissi dopo il training. ", el("b", {}, "Bias"), " = un numero appreso che appartiene al neurone e si somma alla fine, come un punto di partenza."),
      el("p", { className: "hint" }, "Da lontano vedi i rettangoli degli strati; avvicinati con la rotellina (o con un doppio clic sulla colonna che ti interessa) per vedere i neuroni. Clicca un neurone per vedere il suo calcolo."),
      summary,
      el(
        "div",
        { className: "tf-controls" },
        el("div", { className: "tf-row" }, el("label", {}, "Prompt"), promptInput, el("button", { type: "button", onclick: () => runPrompt(promptInput.value) }, "Calcola")),
        el("div", { className: "tf-row" }, el("label", {}, "Posizione t (il token di cui vedi i neuroni):")),
        positions,
        el("div", { className: "tf-row" }, el("label", {}, "Archi"), fractionInput, fractionOut),
        el("div", { className: "tf-row" }, el("button", { type: "button", title: `${fmt(EDGE_COUNT)} linee: fluido con una scheda grafica, lento con il rendering software`, onclick: () => setFraction(1, true) }, "Mostra tutti i pesi"), el("button", { type: "button", onclick: () => setFraction(FRACTIONS[DEFAULT_FRACTION], true) }, "Solo i più forti")),
        status,
      ),
      el("button", { type: "button", className: "tf-start", onclick: () => ui.player?.start("neurons-forward") }, "▶ Animazione: un passaggio in avanti"),
      legend([
        { color: "#e0f2fe", ring: true, label: "input layer: one-hot (un solo neurone acceso)" },
        { color: "#fb923c", ring: true, label: "neurone hidden: attivazione positiva" },
        { color: "#3b82f6", ring: true, label: "neurone hidden: attivazione negativa" },
        { color: "#67e8f9", ring: true, label: "output layer: probabilità (più chiaro = più probabile)" },
        { color: "#fb923c", line: true, label: "peso positivo (opacità = |peso|)" },
        { color: "#3b82f6", line: true, label: "peso negativo" },
        { color: HEAD_COLORS[0], line: true, dashed: true, label: "attention: dipende dai dati, non è un peso" },
        { color: "#64748b", line: true, label: "somma o copia: peso fisso 1" },
        { color: "#0ea5e9", line: true, label: "connessione residua" },
      ]),
      card,
      deepBox(
        "Approfondimento: cosa calcola un neurone",
        formula(String.raw`y_j = f\Big(\sum_i w_{j,i}\,x_i + b_j\Big)`),
        el("p", { className: "hint" }, "w sono gli archi (pesi), b il bias del neurone, f la funzione di attivazione: ReLU nella feed-forward (f(z) = max(0, z)), nessuna (identità) nelle altre nn.Linear."),
        formula(String.raw`\text{LayerNorm: } y_j = \gamma_j\,\frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_j,\quad \mu = \tfrac{1}{${C}}\sum_i x_i`),
        el("p", { className: "hint" }, "Nella LayerNorm ogni neurone dipende da tutti gli ingressi solo attraverso media μ e varianza σ² (non apprese); i parametri sono γ (disegnato come arco uno a uno) e β (parametro del neurone). ε = 0,00001."),
        formula(String.raw`\text{head}_h(t) = \sum_{s \le t} a_{t,s}\, v_s,\qquad a_{t,s} = \operatorname{softmax}_s\!\Big(\tfrac{q_t \cdot k_s}{\sqrt{${hs}}}\Big)\quad(\sqrt{${hs}}:\ \text{head\_size})`),
        el("p", { className: "hint" }, "I coefficienti a non sono pesi: si ricalcolano a ogni prompt confrontando query e key. Per questo l'attention è tratteggiata e cambia quando cambi prompt o posizione."),
      ),
      sources([
        { href: "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html", label: "PyTorch: nn.Linear", note: "y = xAᵀ + b" },
        { href: "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html", label: "PyTorch: nn.LayerNorm", note: "ε = 1e-5, varianza biased" },
        { href: "https://arxiv.org/abs/1607.06450", label: "Ba, Kiros, Hinton (2016): Layer Normalization" },
        { href: "https://arxiv.org/abs/1706.03762", label: "Vaswani et al. (2017): Attention Is All You Need", note: "scaled dot-product attention" },
        { href: "https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py", label: "Karpathy, ng-video-lecture (gpt.py)" },
      ]),
    );
  }

  function renderPositions() {
    positions.replaceChildren(
      ...state.ids.map((id, t) =>
        el("button", { type: "button", className: t === state.t ? "active" : "", title: `posizione ${t}`, onclick: () => setPosition(t) }, shownChar(state.engine.vocab[id])),
      ),
    );
  }

  // ---------------------------------------------------------------- dati
  async function ensureModel() {
    state.engine ??= await tab.getEngine();
    state.layout ??= buildNeuronLayout(spec.config);
    if (!state.scene) {
      state.scene = new NeuronScene(state.layout, spec);
      state.unregister = tab.lod.register({ detail: state.scene.detail, blocks: state.scene.blocks, macro: state.scene.macro });
    }
    const e = state.engine;
    const key = e.weightsSource === "custom" ? Symbol("custom") : `${e.weightsSource}:${e.snapshotStep ?? ""}`;
    if (key !== state.edgesKey) {
      status.textContent = `Costruisco i ${fmt(EDGE_COUNT)} archi dai pesi veri…`;
      await new Promise((r) => setTimeout(r, 0));
      state.edges = buildEdges(state.layout, (name) => e.paramViews.get(name).data);
      state.scene.setEdges(state.edges);
      state.edgesKey = key;
      const b = parameterBreakdown(state.edges, spec.params);
      const neurons = state.layout.columns.filter((c) => c.role !== ROLE.copy).reduce((a, c) => a + c.n, 0);
      summary.textContent = `Conti veri: ${fmt(neurons)} neuroni (più ${fmt(state.layout.count - neurons)} copie del vettore x a inizio riga). Parametri: ${fmt(b.matrix)} archi delle matrici + ${fmt(b.gamma)} γ delle LayerNorm (archi uno a uno) + ${fmt(b.neuron)} bias e β (parametri dei neuroni) = ${fmt(b.total)}.`;
    }
    setFraction(state.fraction);
  }

  function runPrompt(text) {
    const e = state.engine;
    try {
      const ids = e.encode(text).slice(-spec.config.block_size);
      if (!ids.length) throw new Error("Scrivi almeno un carattere.");
      state.prompt = text;
      state.ids = ids;
      state.trace = e.forward(ids, { trace: true }).trace;
      status.classList.remove("error");
      status.textContent = `Pesi ${e.weightsSource === "trained" ? "addestrati" : `dello step ${e.snapshotStep}`} · ${ids.length} token.`;
      setPosition(ids.length - 1);
    } catch (err) {
      status.classList.add("error");
      status.textContent = err instanceof VocabularyError ? `Caratteri non nel vocabolario: ${err.unknown.map((c) => JSON.stringify(c)).join(" ")}` : err.message;
    }
  }

  function setPosition(t) {
    state.t = t;
    state.values = neuronValues(state.layout, state.trace, t);
    state.scene.setValues(state.values);
    state.scene.setAttention(state.trace, t, state.engine.vocab);
    renderPositions();
    updateProbChart();
    if (state.selected !== null) selectNeuron(state.selected);
  }

  function setFraction(f, syncSlider = false) {
    state.fraction = f;
    if (syncSlider) fractionInput.value = String(FRACTIONS.indexOf(f));
    if (!state.scene?.edgeGeometry) return;
    const shown = state.scene.setFraction(f, setBaseOpacity);
    fractionOut.textContent = `${pctText(f)} · ${fmt(shown)}`;
  }

  // ---------------------------------------------------------------- selezione e scheda del neurone
  function selectNeuron(k) {
    state.selected = k;
    state.scene.dimEdges = k !== null;
    setFraction(state.fraction);
    state.scene.select(k, state.edges, state.values);
    if (k === null) {
      card.hidden = true;
      return;
    }
    const e = state.engine;
    const x = explainNeuron(state.layout, k, state.trace, state.t, (name) => e.paramViews.get(name).data, e.vocab);
    card.hidden = false;
    card.replaceChildren(...neuronCard(x, e, state));
    updateContribChart(x);
    ui.sidebar.select("viste");
    card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // ---------------------------------------------------------------- grafici
  let charts = null;
  function ensureCharts() {
    if (!ui.charts) return;
    const section = topChartsSection(ui.charts, "neuroni", "Rete neurale", "Numeri veri del modello per il prompt e la posizione scelti.");
    if (charts) return;
    charts = {
      probs: section.chart("probs", { type: "bar", title: "Probabilità del prossimo carattere", description: "Le 10 probabilità più alte dello strato di uscita (softmax) per la posizione scelta: è ciò che il modello prevede.", xLabel: "carattere", yLabel: "probabilità", percent: true }),
      contrib: section.chart("contrib", { type: "bar", title: "Contributi al neurone selezionato", description: "Gli addendi w·x più grandi (in valore assoluto) della somma che calcola il neurone cliccato, più il bias: mostra quali ingressi lo spingono su o giù.", xLabel: "ingresso", yLabel: "w · x", highlightLabel: "bias" }),
      layer: section.chart("layer", { type: "histogram", title: "Attivazioni dello strato", description: "Distribuzione dei valori dei neuroni della colonna selezionata (o della colonna che l'animazione sta calcolando).", xLabel: "attivazione", yLabel: "neuroni" }),
      norm: section.chart("norm", { type: "line", title: "Ampiezza del vettore lungo la rete", description: "Radice della media dei quadrati (RMS) di ogni strato man mano che l'animazione avanza: le LayerNorm la riportano vicino a 1, i residui la fanno crescere.", xLabel: "strato calcolato", yLabel: "RMS" }),
    };
  }
  function updateProbChart() {
    if (!charts) return;
    const top = state.engine.topTokens(state.values.subarray(state.layout.byId.get("probs").offset, state.layout.byId.get("probs").offset + spec.config.vocab_size), 10);
    charts.probs.setData({ labels: top.map((x) => `'${shownChar(x.token)}'`), values: top.map((x) => x.prob), highlight: 0 });
  }
  function updateContribChart(x) {
    if (!charts) return;
    const col = x.column;
    charts.layer.setData({ values: state.values.subarray(col.offset, col.offset + col.n), bins: 30 });
    charts.layer.setHighlight({ x: x.value, label: "neurone selezionato" });
    if (x.kind === "linear") {
      const top = x.terms.slice(0, 10);
      const labels = top.map((s) => `${x.input.short}[${s.k}]`);
      const values = top.map((s) => s.product);
      if (x.bias !== undefined) {
        labels.push("bias");
        values.push(x.bias);
      }
      charts.contrib.setData({ labels, values, highlight: x.bias !== undefined ? labels.length - 1 : undefined });
    } else if (x.kind === "attention") {
      charts.contrib.setData({ labels: x.terms.slice(0, 10).map((s) => `'${shownChar(s.token)}' ${s.s}`), values: x.terms.slice(0, 10).map((s) => s.product) });
    } else charts.contrib.clear();
  }

  // ---------------------------------------------------------------- puntatore
  let downAt = null;
  const canvas = host.renderer.domElement;
  const onDown = (e) => (downAt = [e.clientX, e.clientY]);
  function onUp(event) {
    if (!downAt || Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 5) return;
    if (tab.lod.current !== "detail" || !state.values) return;
    host.pick(event, []); // imposta host.raycaster dal puntatore
    const k = state.scene.pick(host.raycaster.ray);
    if (k !== null) selectNeuron(k);
  }

  // ---------------------------------------------------------------- vista
  function setSceneVisible(v) {
    if (!state.scene) return;
    state.scene.detail.visible = v;
    state.scene.blocks.visible = v;
    state.scene.macro.visible = v;
  }

  /** Soglie del LOD della vista, scalate sulla forma dello stage visibile (vedi REFERENCE_ASPECT). */
  const fitDistances = () => {
    const scale = THREE.MathUtils.clamp(REFERENCE_ASPECT / host.visibleAspect(), 1, 3);
    tab.lod.setDistances({ detail: DISTANCES.detail * scale, blocks: DISTANCES.blocks * scale });
  };

  /** Inquadratura iniziale della vista: l'ingresso e il primo blocco (livello Blocchi: gli strati con i nomi). */
  const frameStart = () => {
    fitDistances(); // lo stage può essere cambiato (finestra, codice, grafici) dall'ultima volta
    return director.focus(state.scene.boxOf(["b0.x", "b0.resid2"]), { showFrame: false, margin: 1.5, direction: FRONT });
  };

  tab.views.register({
    id: "neuroni",
    label: "Rete neurale",
    description: `Tutti i neuroni e i ${fmt(EDGE_COUNT)} pesi che li collegano; clic su un neurone per il suo calcolo.`,
    order: 20,
    enter: async () => {
      state.active = true;
      state.saved = { root: arch.root.visible, rail: arch.rail.visible, packet: flow.packet.visible, playing: flow.playing };
      arch.root.visible = false;
      arch.rail.visible = false;
      flow.packet.visible = false;
      flow.playing = false;
      fitDistances();
      director.hideFrame();
      if (!info.childNodes.length) renderInfo();
      info.hidden = false;
      status.textContent = "Carico i pesi del modello…";
      ensureCharts();
      try {
        await ensureModel();
      } catch (err) {
        status.classList.add("error");
        status.textContent = `Impossibile caricare i pesi: ${err.message}`;
        return;
      }
      if (!state.active) return;
      setSceneVisible(true);
      if (!state.trace) runPrompt(state.prompt);
      else setPosition(state.t);
      state.offFrame = host.onFrame((dt) => state.scene.tick(dt));
      canvas.addEventListener("pointerdown", onDown);
      canvas.addEventListener("pointerup", onUp);
      tab.setCaption("Rete neurale: doppio clic su una colonna per avvicinarti, poi clic su un neurone per il suo calcolo.");
      tab.showCode("model.forward", "Rete neurale: il forward che accende i neuroni", ["tok_emb =", "pos_emb =", "x = self.blocks(x)", "x = self.ln_f(x)", "logits = self.lm_head(x)"]);
      frameStart();
    },
    // "Centra" e "Vista d'insieme": il primo blocco, come all'apertura della vista
    home: () => (state.scene ? frameStart() : Promise.resolve()),
    exit: () => {
      state.active = false;
      // L'azione "passaggio in avanti" anima questa vista: lasciarla correre a vista nascosta
      // muoverebbe la camera verso oggetti invisibili.
      if (ui.player?.action?.id === "neurons-forward") ui.player.stop();
      info.hidden = true;
      setSceneVisible(false);
      state.offFrame?.();
      state.offFrame = null;
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      tab.lod.resetDistances();
      if (state.saved) {
        arch.root.visible = state.saved.root;
        arch.rail.visible = state.saved.rail;
        flow.packet.visible = state.saved.packet;
        flow.playing = state.saved.playing;
      }
    },
  });

  tab.neuronsView = state; // stato della vista (layout, archi, trace…): utile da console e per i test automatici
  registerForwardAction(tab, state, { ensureCharts: () => (ensureCharts(), charts), setPosition });
}

// ==================================================================== scheda del neurone
function neuronCard(x, engine, state) {
  const col = x.column;
  const role = col.role === ROLE.input ? "input layer" : col.role === ROLE.output ? "output layer" : col.role === ROLE.copy ? "copia del vettore (riga sotto)" : "hidden";
  const head = [
    el("h4", {}, "Neurone selezionato"),
    el("p", {}, el("b", {}, col.title), ` · neurone ${x.index} di ${col.n}`),
    el(
      "p",
      { className: "tf-meta" },
      el("span", {}, "ruolo ", el("b", {}, role)),
      el("span", {}, "valore ", el("b", {}, fmt(x.relu ? x.post : x.value, 4))),
      el("span", {}, "posizione ", el("b", {}, `${state.t} '${shownChar(engine.vocab[state.ids[state.t]])}'`)),
      el("span", {}, "parametri che lo alimentano ", el("b", {}, fmt(x.params))),
    ),
  ];
  const n = (v, d = 3) => texNum(v, d);
  const body = [];
  switch (x.kind) {
    case "linear": {
      const top = x.terms.slice(0, 4);
      const rest = x.terms.slice(4).reduce((a, s) => a + s.product, 0);
      const pair = (a, b) => [a, b].filter(Boolean).map((t) => `(${n(t.w)})(${n(t.x)})`).join(" + ");
      const bias = x.bias !== undefined ? ` + \\underbrace{${n(x.bias)}}_{b}` : "";
      // Una formula su più righe (aligned): nella sidebar stretta una riga sola uscirebbe dal riquadro.
      body.push(
        formula(String.raw`z = \sum_{i=0}^{${x.terms.length - 1}} w_i\,x_i${x.bias !== undefined ? " + b" : ""}`),
        formula(String.raw`\begin{aligned} z &= ${pair(top[0], top[1])} \\ &\quad + ${pair(top[2], top[3])} \\ &\quad + \underbrace{${n(rest)}}_{\text{altri } ${x.terms.length - 4}}${bias} \\ &= ${n(x.sum)} \end{aligned}`),
      );
      if (x.relu) body.push(formula(String.raw`h = \operatorname{ReLU}(z) = \max(0,\,${n(x.sum)}) = ${n(x.post)}`));
      body.push(
        el("p", { className: "hint" }, `Pesi da ${x.tensor} (${x.terms.length} archi in ingresso${x.bias !== undefined ? " + 1 bias del neurone" : ", nessun bias"}). Ingressi: colonna "${x.input.short}". I contributi più grandi:`),
        contribTable(x.terms.slice(0, 8), x.input.short),
      );
      if (x.token !== undefined) body.push(el("p", { className: "hint" }, `È il punteggio (logit) del carattere '${shownChar(x.token)}'.`));
      break;
    }
    case "layernorm":
      body.push(
        formula(String.raw`\mu = ${n(x.mean)},\quad \sigma^2 = ${n(x.variance, 4)}\quad(\text{su tutti i ${x.input.n} ingressi})`),
        formula(String.raw`\begin{aligned} y &= \gamma\,\frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta \\ &= ${n(x.gamma)}\cdot\frac{${n(x.x)} - (${n(x.mean)})}{\sqrt{${n(x.variance, 4)} + 10^{-5}}} + (${n(x.beta)}) \\ &= ${n(x.sum)} \end{aligned}`),
        el("p", { className: "hint" }, "Parametri del neurone: γ (l'arco uno a uno che arriva) e β. Media e varianza non sono apprese: si calcolano sul vettore del token."),
      );
      break;
    case "attention": {
      const top = x.terms.slice(0, 3).map((s) => `(${n(s.att)})(${n(s.v)})`).join(" + ");
      body.push(
        formula(String.raw`\begin{aligned} c &= \sum_{s=0}^{${state.t}} a_{${state.t},s}\, v_s[${x.d}] \\ &= ${top}${x.terms.length > 3 ? " + \\dots" : ""} \\ &= ${n(x.sum)} \end{aligned}`),
        el("p", { className: "hint" }, `Head ${x.head}, componente ${x.d}. 0 parametri: i coefficienti a (che sommano a 1) vengono da softmax(q·k/√${state.layout.headSize}) e cambiano con il prompt. Le posizioni guardate di più:`),
        el(
          "table",
          { className: "tf-contrib" },
          el("tbody", {}, ...x.terms.slice(0, 6).map((s) => el("tr", {}, el("td", {}, `${s.s} '${shownChar(s.token)}'`), el("td", { className: "num" }, `a = ${fmt(s.att, 3)}`), el("td", { className: "num" }, `v = ${fmt(s.v, 3)}`), el("td", { className: `num ${s.product >= 0 ? "pos" : "neg"}` }, fmt(s.product, 3))))),
        ),
      );
      break;
    }
    case "add":
      body.push(formula(String.raw`y = ${x.parts.map(([, v]) => `(${n(v)})`).join(" + ")} = ${n(x.sum)}`), el("p", { className: "hint" }, `Somma uno a uno: ${x.parts.map(([name]) => name).join(" + ")}. Nessun parametro: il collegamento ha peso fisso 1.`));
      break;
    case "lookup":
      body.push(
        formula(String.raw`e = \text{${x.table.split(".")[0].replace(/_/g, "\\_")}}[${x.row}, ${x.index}] = ${n(x.sum)}`),
        el("p", { className: "hint" }, `Lettura dalla riga ${x.row} (${shownChar(x.rowLabel)}) della tabella: equivale a moltiplicare il vettore one-hot per la matrice. La colonna ${x.index} della tabella ha ${x.params} numeri, ne serve uno solo.`),
      );
      break;
    case "onehot":
      body.push(
        formula(String.raw`x = ${x.active ? 1 : 0}`),
        el("p", { className: "hint" }, x.token !== undefined ? `Neurone del carattere '${shownChar(x.token)}': vale 1 solo se il token in ingresso è questo.` : `Neurone della posizione ${x.position}: vale 1 solo alla posizione ${x.position}.`),
      );
      break;
    case "softmax":
      body.push(
        formula(String.raw`p = \frac{e^{z - z_{\max}}}{\sum_u e^{z_u - z_{\max}}} = \frac{e^{${n(x.z)} - ${n(x.max)}}}{${n(x.denom)}} = ${n(x.sum, 4)}`),
        el("p", { className: "hint" }, `Probabilità che il prossimo carattere sia '${shownChar(x.token)}'. Nessun parametro; sottrarre il massimo non cambia il risultato ma evita numeri enormi.`),
      );
      break;
    default:
      body.push(el("p", { className: "hint" }, "Stesso valore dell'ultima colonna della riga sotto: è il vettore x che entra in questo blocco (lo vedi ripetuto per chiarezza, non è un neurone in più)."));
  }
  return [...head, ...body];
}

function contribTable(terms, inputName) {
  return el(
    "table",
    { className: "tf-contrib" },
    el("tbody", {}, ...terms.map((s) => el("tr", {}, el("td", {}, `${inputName}[${s.k}]`), el("td", { className: "num" }, `w = ${fmt(s.w, 3)}`), el("td", { className: "num" }, `x = ${fmt(s.x, 3)}`), el("td", { className: `num ${s.product >= 0 ? "pos" : "neg"}` }, fmt(s.product, 3))))),
  );
}

// ==================================================================== azione: un passaggio in avanti
function registerForwardAction(tab, state, { ensureCharts }) {
  const { ui, spec } = tab;
  if (!ui.player) return;
  const { n_layer: L, n_embd: C, n_head: H, vocab_size: V, block_size: T } = spec.config;
  const stepsDef = [
    { cols: ["tok_onehot", "pos_onehot"], title: "Input one-hot", code: "model.forward", hl: ["B, T = idx.shape"], caption: `Input layer: per il carattere scelto si accende un solo neurone su ${V}, e uno solo dei ${T} neuroni di posizione.` },
    { cols: ["tok_emb", "pos_emb", "x0"], title: "Embedding", code: "model.forward", hl: ["tok_emb =", "pos_emb =", "x = tok_emb + pos_emb"], caption: `Ogni neurone acceso sceglie una riga di pesi: il token dà ${C} numeri, la posizione altri ${C}; sommati diventano il vettore x.` },
    { cols: ["b0.x", "b0.ln1"], title: "Blocco 0 · LayerNorm", code: "block", hl: ["self.ln1(x)"], caption: `LayerNorm: i ${C} numeri vengono riportati a media 0 e varianza 1, poi scalati da γ e spostati da β.` },
    { cols: ["b0.qkv"], title: "Blocco 0 · query, key, value", code: "head", hl: ["k = self.key(x)", "q = self.query(x)", "v = self.value(x)"], caption: `Tre gruppi di ${C} neuroni: query (cosa cerco), key (cosa offro), value (cosa passo). Ognuno è una somma pesata dei ${C} ingressi, senza bias.` },
    { cols: ["b0.ctx", "b0.concat"], title: "Blocco 0 · attention", code: "head", hl: ["wei = q @ k.transpose", "F.softmax", "return wei @ v"], caption: "Qui non ci sono pesi: le linee tratteggiate dicono quanto questo token guarda ogni token precedente, calcolato confrontando query e key." },
    { cols: ["b0.proj", "b0.resid1"], title: "Blocco 0 · proiezione e residuo", code: "mha", hl: ["self.proj(out)"], caption: `La proiezione rimescola le ${H} head; il risultato si somma al vettore di partenza (connessione residua).` },
    { cols: ["b0.ln2", "b0.hidden"], title: `Blocco 0 · feed-forward (${4 * C} neuroni)`, code: "ffwd", hl: ["nn.Linear(config.n_embd, hidden)", "nn.ReLU()"], caption: `Dopo una seconda LayerNorm, ${4 * C} neuroni nascosti: ognuno somma ${C} ingressi pesati più un bias, poi la ReLU azzera i negativi.` },
    { cols: ["b0.ffnOut", "b0.resid2"], title: "Blocco 0 · uscita e residuo", code: "block", hl: ["x = x + self.ffwd(self.ln2(x))"], caption: `La seconda matrice riporta ${4 * C} numeri a ${C}; si sommano di nuovo al vettore: è l'uscita del blocco 0.` },
    ...[...Array(L - 1).keys()].map((i) => ({ cols: rowCols(state, i + 1), title: `Blocco ${i + 1}`, code: "block", hl: ["x = x + self.sa", "x = x + self.ffwd"], caption: `Il blocco ${i + 1} ripete gli stessi calcoli con pesi diversi: LayerNorm, attention, feed-forward, residui.` })),
    { cols: ["out.x", "ln_f"], title: "LayerNorm finale", code: "model.forward", hl: ["x = self.ln_f(x)"], caption: "Ultima normalizzazione del vettore." },
    { cols: ["logits"], title: "Output layer: logits", code: "model.forward", hl: ["logits = self.lm_head(x)"], caption: `${V} neuroni di uscita, uno per carattere: ognuno somma ${C} ingressi pesati più un bias.` },
    { cols: ["probs"], title: "Softmax: probabilità", code: "generate", hl: ["softmax"], caption: "La softmax trasforma i punteggi in probabilità che sommano a 1: il neurone più chiaro è il carattere più probabile." },
  ];
  ui.player.register({
    id: "neurons-forward",
    title: "Un passaggio in avanti, neurone per neurone",
    category: "Viste",
    summary: "I neuroni si accendono strato dopo strato, dal carattere in ingresso alle probabilità del prossimo.",
    onStart: async (ctx) => {
      await tab.views.activate("neuroni");
      ui.sidebar.select("viste");
      if (!state.values) return;
      tab.lod.controller.setMode("detail");
      const { columns, byId } = state.layout;
      ctx.data.firstCol = stepsDef.map((s) => Math.min(...s.cols.map((id) => columns.indexOf(byId.get(id)))));
      ctx.data.lastCol = stepsDef.map((s) => Math.max(...s.cols.map((id) => columns.indexOf(byId.get(id)))));
      ctx.data.p = 0;
      ctx.data.step = 0;
      // Una sola funzione per tutta l'azione (niente allocazioni per frame): legge passo e avanzamento da ctx.data.
      ctx.data.reveal = (ci) => (ci < ctx.data.firstCol[ctx.data.step] ? 1 : ci > ctx.data.lastCol[ctx.data.step] ? 0 : ctx.data.p);
    },
    onStop: () => {
      tab.lod.controller.setMode("auto");
      if (state.values) state.scene.setValues(state.values);
    },
    onEnd: () => tab.lod.controller.setMode("auto"),
    steps: stepsDef.map((s, i) => ({
      title: s.title,
      caption: s.caption,
      duration: i < 2 ? 3.5 : 4.5,
      enter: (ctx) => {
        if (!state.values) return;
        ctx.data.step = i;
        ctx.data.p = 0;
        state.scene.setValues(state.values, ctx.data.reveal);
        tab.ui.director.focus(state.scene.boxOf(s.cols), { showFrame: true, margin: s.cols.length > 4 ? 1.05 : 1.6, duration: 1.1, direction: FRONT });
        tab.showCode(s.code, `Rete neurale: ${s.title}`, s.hl);
        drawCharts(ensureCharts(), state, stepsDef, i);
      },
      update: (p, dt, ctx) => {
        if (!state.values || !ctx.data.reveal) return;
        const q = Math.min(1, p * 1.6); // i neuroni si accendono nella prima parte del passo
        if (q === ctx.data.p) return;
        ctx.data.p = q;
        state.scene.setValues(state.values, ctx.data.reveal);
      },
    })),
  });
}

function rowCols(state, row) {
  return state.layout.columns.filter((c) => c.row === row).map((c) => c.id);
}

/** Grafici dell'animazione: RMS degli strati già calcolati e istogramma dello strato corrente. */
function drawCharts(charts, state, stepsDef, i) {
  if (!charts) return;
  const { byId } = state.layout;
  const points = [];
  let idx = 0;
  for (let s = 0; s <= i; s++) {
    for (const id of stepsDef[s].cols) {
      const col = byId.get(id);
      if (!col.n || col.role === ROLE.input || col.key === "probs") continue;
      let sq = 0;
      for (let k = 0; k < col.n; k++) sq += state.values[col.offset + k] ** 2;
      points.push([++idx, Math.sqrt(sq / col.n)]);
    }
  }
  charts.norm.setData({ series: [{ name: "RMS dello strato", points }] });
  if (points.length) charts.norm.setHighlight({ x: points.at(-1)[0], label: stepsDef[i].title });
  const main = [...stepsDef[i].cols].reverse().map((id) => byId.get(id)).find((c) => c.n);
  charts.layer.setData({ values: state.values.subarray(main.offset, main.offset + main.n), bins: 30 });
}
