/**
 * Vista "Pesi numerici": il codice vero, caricato alla prima apertura da numbers.js (la registrazione
 * leggera della vista). Qui three.js (numbersScene.js), la tabella e l'istogramma condivisi, il run
 * "GPT del corso" e l'albero dei moduli.
 *
 * Sulla scena dell'architettura, avvicinandosi a una matrice, i colori delle heatmap diventano i numeri
 * float32 che usa il motore (core/matrixNumbers.js, un solo budget per la scena). La sorgente si sceglie:
 * i pesi finali oppure uno step del training su Tiny Shakespeare (gli 11 snapshot veri, float16, con la
 * stima lineare tra due snapshot solo se lo studente la chiede, dichiarata come tale). Le heatmap si
 * riscrivono con i pesi dello step e i numeri seguono: si vedono i pesi nascere dal rumore.
 * La vista MOSTRA soltanto: il motore e le altre viste restano sui pesi finali, e uscendo le texture
 * tornano esattamente quelle di prima.
 *
 * Sidebar (pannello Viste): sorgente e cursore degli step, l'etichetta onesta (vero / arrotondato /
 * stimato), i tensori raggruppati come il codice Python, la tabella dei numeri del tensore scelto
 * (core/ui/tensorTable.js) con statistiche e istogramma, "Apri nel Laboratorio pesi".
 * Logica pura in numbersModel.js, three.js in numbersScene.js, DOM in numbersUi.js.
 *
 * API (tab.numbersView, sostituita a quella leggera di numbers.js appena il modulo è caricato):
 * open({ tensor, step, interpolate, cell, birth }), .source, .selected, .state, .scene, .table,
 * .setStep(step), .setMode(id), .setInterpolate(on), .togglePlay(), .selectTensor(name, o);
 * evento "numbers-source" su tab.events.
 */
import { el, fmt, prefersReducedMotion } from "../../../core/dom.js";
import { Histogram } from "../../../core/ui/histogram.js";
import { showToast } from "../../../core/ui/share.js";
import { TensorTable, renderTensorStats } from "../../../core/ui/tensorTable.js";
import { preciseText } from "../../../core/ui/tensorTableMath.js";
import { buildModuleTree, findNode } from "../../../shared/weights/moduleTree.js";
import { lerpInto } from "../actions/math.js";
import { NumbersScene } from "./numbersScene.js";
import { FINAL_SOURCE, NUMBERS_VIEW_ID, birthSchedule, cellMeaning, float16RoundingCheck, labTarget, sameSource, shapeText, sourceLabel, tensorAxes, tensorGroups, trainingSource } from "./numbersModel.js";
import { numbersBar, playButton, sourceNote, sourceSwitch, stepChips, stepControl, tensorList } from "./numbersUi.js";
import { createViewInfo, deepBox, sources } from "./panel.js";

const DEFAULT_TENSOR = "token_embedding_table.weight";
const LAB_TAB = "lab";
/** Riscritture delle heatmap al secondo mentre si trascina il cursore o scorre l'animazione (~5 ms l'una). */
const LIVE_WRITES_PER_SECOND = 12;
/** Aggiornamenti al secondo di statistiche e istogramma durante l'animazione (il resto del tempo: subito). */
const PANEL_UPDATES_PER_SECOND = 4;
const narrowScreen = () => window.matchMedia("(max-width: 900px)").matches;
const canHover = () => window.matchMedia("(hover: hover)").matches;
const CAPTION = {
  hover: "Avvicinati a una matrice: i colori diventano numeri. Passa sopra una cella per il valore, clic per la tabella.",
  // sul telefono una riga sola: sopra c'è l'intestazione del codice ridotto
  touch: "Avvicinati: i colori diventano numeri.",
};

/**
 * Crea la vista (una volta sola, alla prima apertura): stato, interfaccia, scena.
 * @returns {{enter: () => Promise<void>, exit: () => void, home: () => Promise<void>, api: object}}
 */
export function createNumbersView(tab) {
  const { ui, spec, arch, flow } = tab;
  const { host } = ui;
  const info = createViewInfo(tab, NUMBERS_VIEW_ID);
  const state = {
    active: false,
    run: null,
    runPromise: null,
    tree: null,
    steps: [],
    mode: "final", // "final" | "training"
    step: 0, // valore del cursore del training
    interpolate: false,
    source: FINAL_SOURCE,
    views: null,
    finalViews: null,
    snapshots: new Map(), // step → Promise<{views: Map nome → Float32Array, flat}>
    loaded: new Set(), // step degli snapshot già in memoria (niente "Carico…" per loro)
    mix: null,
    mixViews: null,
    lastCheck: null,
    selected: DEFAULT_TENSOR,
    cell: null, // { row, col } attiva nella tabella
    cardFor: null, // il tensore della scheda disegnata
    tableData: null,
    seq: 0,
    playing: false,
    play: null,
    saved: null,
    lastWrite: 0,
    lastPanel: 0,
    stepMs: [],
  };

  const scene = new NumbersScene(tab, {
    onHover: (hit, event) => showTip(hit, event),
    onPick: (hit, event) => pickCell(hit, event),
  });

  // ---------------------------------------------------------------- interfaccia (costruita alla prima apertura)
  const status = el("p", { className: "tf-status", ariaLive: "polite" });
  const note = sourceNote();
  const switches = [sourceSwitch({ value: state.mode, onChange: (id) => setMode(id) }), sourceSwitch({ value: state.mode, onChange: (id) => setMode(id), compact: true })];
  let panelStep = null;
  let barStep = null;
  let chips = null;
  const plays = [playButton({ onClick: () => togglePlay() }), playButton({ onClick: () => togglePlay(), short: true, className: "tf-bar-btn tf-bar-primary" })];
  const interpolateBox = el("input", { type: "checkbox", id: "tf-num-interp", onchange: () => setInterpolate(interpolateBox.checked) });
  const trainingBox = el("div", { className: "tf-training-controls", hidden: true });
  const listBox = el("div", { className: "tf-tensors" });
  const card = el("section", { className: "tf-card tf-num-card", ariaLabel: "Tensore scelto" });
  const cardHead = el("div", { className: "tf-num-head" });
  const tableBox = el("div", { className: "tf-num-table" });
  const statsBox = el("div", { className: "tf-num-stats" });
  const histBox = el("div", { className: "tf-num-hist" });
  const showCell = el("button", { type: "button", disabled: true, title: "La camera vola davanti alla cella scelta nella tabella, abbastanza vicino da leggerne il numero.", onclick: () => flyToCell() }, "Mostra nella scena");
  const openLab = el("button", { type: "button", title: "Apre questo tensore nel tab Laboratorio pesi (alla stessa sorgente: pesi finali o step del training).", onclick: () => openInLab() }, "Apri nel Laboratorio pesi");
  card.append(cardHead, tableBox, el("div", { className: "tf-row tf-num-actions" }, showCell, openLab), statsBox, histBox);
  let list = null;
  let table = null;
  let histogram = null;
  let bar = null;
  let built = false;
  // Altezza vera della barra dello stage (va a capo quando codice o grafici stringono la scena): il CSS ci
  // mette sotto i suggerimenti della navigazione (--tf-numbers-h). Solo a ridimensionamento, mai a riposo.
  const barObserver = new ResizeObserver(() => {
    const h = bar?.element.offsetHeight ?? 0;
    if (h > 0) ui.root.style.setProperty("--tf-numbers-h", `${h}px`);
  });

  // etichetta visiva (gli stessi numeri sono nella tabella, accessibile da tastiera): niente annunci a ogni passaggio
  const tip = el("div", { className: "tf-num-tip", hidden: true, ariaHidden: "true" });
  ui.stage.append(tip);
  let tipTimer = 0;

  function build() {
    if (built) return;
    built = true;
    const steps = state.steps;
    panelStep = stepControl({ steps, value: state.step, onInput: (s, o) => setStep(s, o) });
    barStep = stepControl({ steps, value: state.step, onInput: (s, o) => setStep(s, o), compact: true, label: "Step del training (barra della scena)" });
    chips = stepChips({ steps, onPick: (s) => setStep(s, { live: false }) });
    trainingBox.append(
      el("div", { className: "tf-step-head" }, el("span", {}, "Step del training"), el("output", { className: "tf-step-value" })),
      panelStep.element,
      chips.element,
      el("label", { className: "tf-check", htmlFor: interpolateBox.id }, interpolateBox, el("span", {}, "Interpola tra gli snapshot ", el("small", {}, "(stima lineare: non è un vero step del training)"))),
      plays[0].element,
      el(
        "p",
        { className: "hint" },
        "Qui cambiano solo i pesi. Pattern di attention e probabilità della scena sono attivazioni calcolate con i pesi finali sul prompt d'esempio: negli step del training restano attenuati. Li vedi nascere insieme ai pesi nell'azione ",
        el("button", { type: "button", className: "tf-text-link", onclick: () => startTrainingAction() }, "«Il training completo»"),
        ".",
      ),
    );
    bar = numbersBar({ switchElement: switches[1].element, step: barStep, play: plays[1] });
    bar.element.hidden = true;
    ui.stage.append(bar.element);
    list = tensorList({ groups: tensorGroups(state.run.params), onPick: (name) => selectTensor(name, { fly: true, reveal: false }) });
    listBox.replaceChildren(list.element);
    table = new TensorTable(tableBox, {
      caption: "",
      rowLabel: (i) => currentAxes().rowLabel(i),
      colLabel: (j) => currentAxes().colLabel(j),
      onSelect: (cell) => tableCell(cell),
      onHover: (cell) => scene.highlight(state.selected, cell ? { row: cell.i, col: cell.j } : state.cell),
    });
    histogram = new Histogram(histBox, { bins: 60, height: 118, label: "Valori del tensore" });
    renderInfo();
  }

  function renderInfo() {
    info.replaceChildren(
      el("h3", {}, "Pesi numerici: i numeri dentro le matrici"),
      el("p", {}, "Ogni cella colorata della scena è un peso del modello, cioè un numero. Come in una foto: da lontano vedi i colori, avvicinandoti vedi i singoli pixel e qui, al loro posto, il numero vero."),
      el("p", { className: "hint" }, "Avvicinati con la rotellina (o con il pizzico) a una matrice: quando una cella è grande abbastanza sullo schermo compare il suo valore, con più decimali più ti avvicini. Con «Pesi finali» sono i float32 che il motore della web app usa davvero; negli step del training sono i numeri degli snapshot salvati durante l'addestramento. Passa sopra una cella (o toccala) per riga, colonna e valore; un clic la apre nella tabella qui sotto."),
      el("h4", {}, "Da dove vengono i pesi"),
      el("div", { className: "tf-arrange-row" }, switches[0].element),
      trainingBox,
      note.element,
      status,
      el("h4", {}, "I tensori, come nel codice Python"),
      el("p", { className: "hint" }, "Raggruppati come li crea GPTLanguageModel: gli embedding, un Block per ogni blocks[i] (ln1, le head della self-attention, proj, ln2, il feed-forward), poi ln_f e lm_head. Clic su un nome (o su una matrice nella scena) per la sua tabella."),
      listBox,
      card,
      deepBox(
        "Approfondimento: float32, float16 e la stima lineare",
        el("p", {}, "Un float32 tiene circa 7 cifre significative: è il formato dei pesi finali e dei calcoli del motore. Gli snapshot del training sono salvati in float16 (circa 3 cifre significative) per occupare metà spazio: 0,123456789 diventa 0,12347 in float16. Per questo i numeri di uno snapshot hanno le ultime cifre «arrotondate»."),
        el("p", {}, "Tra due snapshot non sappiamo che cosa sia successo: il training ha fatto centinaia di passi di AdamW che non sono stati salvati. «Interpola» mostra una media pesata dei due snapshot vicini: utile per vedere il cambiamento in modo continuo, ma è una stima, non un dato."),
        el("p", { className: "hint" }, "La scala dei colori resta quella dei pesi finali anche negli step del training: così si vede quanto i pesi crescono (colori più accesi) o si restringono."),
      ),
      sources([
        { href: "https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype", label: "PyTorch: torch.dtype", note: "float32 e float16 (half)" },
        { href: "https://en.wikipedia.org/wiki/Half-precision_floating-point_format", label: "Formato in mezza precisione (IEEE 754 binary16)", note: "11 bit di mantissa: ~3,3 cifre decimali" },
        { href: "https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py", label: "Karpathy, ng-video-lecture (gpt.py)", note: "il modello e i nomi dei tensori" },
      ]),
    );
  }

  // ---------------------------------------------------------------- dati
  /** Il run "GPT del corso" (pesi finali + 11 snapshot a richiesta), caricato una volta per pagina. */
  function ensureRun() {
    state.runPromise ??= import("../../../shared/weights/run.js")
      .then((m) => m.loadBuiltinRun())
      .then((run) => {
        state.run = run;
        state.steps = run.steps;
        state.finalViews = new Map([...run.tensors].map(([name, t]) => [name, t.data]));
        state.mix = new Float32Array(run.totalParams);
        const layout = run.trace?.layout ?? run.params.map((p, i, all) => ({ ...p, offset: all.slice(0, i).reduce((a, q) => a + q.numel, 0) }));
        state.mixViews = new Map(layout.map((p) => [p.name, state.mix.subarray(p.offset, p.offset + p.numel)]));
        state.tree = buildModuleTree(run.params, { snippets: spec.snippets, config: run.config });
        return run;
      });
    state.runPromise.catch(() => (state.runPromise = null)); // un errore di rete non resta in cache
    return state.runPromise;
  }

  /** Pesi di uno snapshot come Map nome → Float32Array (una volta per step). */
  function snapshotViews(step) {
    if (!state.snapshots.has(step)) {
      const pending = state.run.loadSnapshot(step).then((snap) => {
        if (step === state.steps[state.steps.length - 1] && !state.lastCheck) state.lastCheck = float16RoundingCheck(state.run.flat(), snap.flat);
        state.loaded.add(step);
        return { views: new Map([...snap.tensors].map(([name, t]) => [name, t.data])), flat: snap.flat };
      });
      pending.catch(() => state.snapshots.delete(step));
      state.snapshots.set(step, pending);
    }
    return state.snapshots.get(step);
  }

  /** I valori da mostrare per una sorgente (lo stesso Map per la stima lineare: si riscrive sul posto). */
  async function viewsFor(source) {
    if (source.kind === "final") return state.finalViews;
    if (source.kind === "snapshot") return (await snapshotViews(source.step)).views;
    const [a, b] = await Promise.all(source.from.map((s) => snapshotViews(s)));
    lerpInto(state.mix, a.flat, b.flat, source.t);
    return state.mixViews;
  }

  // ---------------------------------------------------------------- sorgente e step
  function setMode(mode) {
    if (mode !== "final" && mode !== "training") return;
    stopPlay();
    state.mode = mode;
    syncSourceUi();
    void applySource(currentSource());
  }

  function currentSource() {
    return state.mode === "final" || !state.steps.length ? FINAL_SOURCE : trainingSource(state.step, state.steps, { interpolate: state.interpolate });
  }

  /** Il cursore (o un pulsante di snapshot) sceglie uno step; senza interpolazione si aggancia allo snapshot. */
  function setStep(step, { live = false, fromPlay = false } = {}) {
    if (!fromPlay) stopPlay();
    if (state.mode !== "training") {
      state.mode = "training";
      syncSourceUi();
    }
    const source = trainingSource(step, state.steps, { interpolate: state.interpolate });
    // mentre si trascina il cursore l'agganciamento si vede solo al rilascio (il pollice non salta)
    state.step = live && !state.interpolate ? step : source.step;
    syncStepUi(source);
    if (sameSource(source, state.source) && source.kind !== "interpolated") return; // stesso snapshot: niente da riscrivere
    if (live && !throttle("lastWrite", LIVE_WRITES_PER_SECOND)) {
      scheduleTrailing(source); // troppo presto: l'ultimo valore arriva comunque, dopo l'intervallo
      return;
    }
    clearTimeout(trailing);
    void applySource(source, { live });
  }

  function setInterpolate(on) {
    state.interpolate = Boolean(on);
    interpolateBox.checked = state.interpolate;
    panelStep?.setInterpolate(state.interpolate);
    barStep?.setInterpolate(state.interpolate);
    if (state.mode === "training") setStep(state.step, { live: false });
  }

  /** true se è passato abbastanza tempo dall'ultima volta (e lo segna). */
  function throttle(key, perSecond) {
    const now = performance.now();
    if (now - state[key] < 1000 / perSecond) return false;
    state[key] = now;
    return true;
  }

  // l'ultimo valore di un trascinamento rapido arriva comunque (dopo l'intervallo)
  let trailing = 0;
  function scheduleTrailing(source) {
    clearTimeout(trailing);
    trailing = setTimeout(() => void applySource(source, { live: true }), 1000 / LIVE_WRITES_PER_SECOND);
  }

  /** Applica una sorgente: numeri, heatmap, tabella, statistiche e istogramma (l'ultima richiesta vince). */
  async function applySource(source, { live = false } = {}) {
    if (!state.active || !state.run) return;
    const token = ++state.seq;
    const needed = source.kind === "final" ? [] : source.kind === "snapshot" ? [source.step] : source.from;
    if (needed.some((s) => !state.loaded.has(s))) setStatus("Carico i pesi dello snapshot…");
    let views;
    try {
      views = await viewsFor(source);
    } catch (err) {
      if (token === state.seq) setStatus(`Impossibile caricare lo snapshot: ${err.message}`, true);
      return;
    }
    if (token !== state.seq || !state.active) return;
    const t0 = performance.now();
    const changed = !sameSource(source, state.source) || source.kind === "interpolated";
    state.source = source;
    state.views = views;
    if (changed) scene.setViews(views, { heatmaps: source.kind === "final" ? "final" : "write" });
    scene.dimFinalActivations(source.kind !== "final");
    const writeMs = scene.stats.writeMs;
    refreshCard({ heavy: !live || throttle("lastPanel", PANEL_UPDATES_PER_SECOND) });
    syncLabel();
    const ms = performance.now() - t0;
    if (changed) state.stepMs.push({ source: source.kind, writeMs, totalMs: ms });
    if (state.stepMs.length > 60) state.stepMs.shift();
    if (!state.playing) setStatus("");
    tab.events.dispatchEvent(new CustomEvent("numbers-source", { detail: { ...source } }));
  }

  function syncSourceUi() {
    for (const s of switches) s.set(state.mode);
    trainingBox.hidden = state.mode !== "training";
    bar?.setTraining(state.mode === "training");
  }

  function syncStepUi(source = state.source) {
    const label = sourceLabel(source, { steps: state.steps, lastCheck: state.lastCheck, format: preciseText });
    for (const s of [panelStep, barStep]) s?.set(state.step, label.title);
    bar?.setLabel(label);
    chips?.set(source.kind === "snapshot" ? source.step : null);
    const out = trainingBox.querySelector(".tf-step-value");
    if (out) out.textContent = label.short;
  }

  function syncLabel() {
    note.set(sourceLabel(state.source, { steps: state.steps, lastCheck: state.lastCheck, format: preciseText }));
    if (state.mode === "training") syncStepUi(state.source);
  }

  function setStatus(text, error = false) {
    status.textContent = text;
    status.classList.toggle("error", error);
  }

  // ---------------------------------------------------------------- "Guarda i pesi nascere"
  function togglePlay() {
    if (state.playing) stopPlay();
    else void startPlay();
  }

  async function startPlay() {
    if (!state.active || !state.steps.length) return;
    if (state.mode !== "training") {
      state.mode = "training";
      syncSourceUi();
    }
    state.playing = true;
    for (const p of plays) p.setPlaying(true);
    setStatus("Carico gli 11 snapshot del training…");
    try {
      await Promise.all(state.steps.map((s) => snapshotViews(s)));
    } catch (err) {
      setStatus(`Impossibile caricare gli snapshot: ${err.message}`, true);
      stopPlay();
      return;
    }
    if (!state.playing || !state.active) return;
    const reduced = prefersReducedMotion();
    // con il movimento ridotto niente scorrimento continuo: uno snapshot alla volta, più lento
    const schedule = birthSchedule(state.steps, { interpolate: state.interpolate && !reduced, segment: reduced ? 1.6 : 1.1 });
    const last = state.steps[state.steps.length - 1];
    let elapsed = state.step >= last ? 0 : schedule.timeOf(state.step); // riprende da dove si era fermi
    let before = performance.now();
    setStatus(state.interpolate && !reduced ? "I pesi nascono: snapshot veri, stimati linearmente tra l'uno e l'altro." : "I pesi nascono: uno snapshot vero dopo l'altro.");
    state.play = host.onFrame(() => {
      const now = performance.now();
      // tempo reale anche con frame lenti; un tab nascosto (niente frame) al ritorno non fa saltare l'animazione
      elapsed += Math.min(0.25, (now - before) / 1000);
      before = now;
      const step = schedule.stepAt(elapsed);
      const done = elapsed >= schedule.duration;
      // setStep limita da sé le riscritture (12 al secondo) e recupera l'ultimo valore
      if (step !== state.step && !done) setStep(step, { live: true, fromPlay: true });
      if (done) {
        setStep(last, { live: false, fromPlay: true });
        stopPlay();
        // si dice che sono i pesi finali solo se la verifica sui file lo conferma
        setStatus(state.lastCheck?.matches ? `Fine: i pesi dello step ${last} sono quelli finali, arrotondati a 16 bit. Riparti o scegli uno step.` : `Fine: l'ultimo snapshot del training (step ${last}). Riparti o scegli uno step.`);
      }
    });
  }

  function stopPlay() {
    if (!state.playing) return;
    state.playing = false;
    state.play?.();
    state.play = null;
    for (const p of plays) p.setPlaying(false);
  }

  // ---------------------------------------------------------------- tensore scelto e tabella
  /** Righe e colonne del tensore scelto (in cache: le intestazioni della tabella le chiedono a ogni riga). */
  let axesCache = { name: null, axes: null };
  function currentAxes() {
    if (axesCache.name !== state.selected || !axesCache.axes) {
      const p = state.run?.tensor(state.selected);
      const axes = p ? tensorAxes(state.selected, p.shape, { vocab: state.run.vocab, config: state.run.config }) : tensorAxes(state.selected, [1], {});
      axesCache = { name: state.selected, axes };
    }
    return axesCache.axes;
  }

  /** Sceglie il tensore: scheda, tabella, carta della head davanti, (fly) volo sulla matrice. */
  function selectTensor(name, { fly = false, cell = null, reveal = false } = {}) {
    if (!state.run?.tensor(name)) return false;
    const changed = name !== state.cardFor;
    if (state.selected && state.selected !== name) scene.highlight(state.selected, null);
    state.selected = name;
    state.cell = cell;
    list?.setActive(name);
    scene.bringToFront(name);
    if (changed) renderCard();
    else refreshCard({ heavy: true });
    if (cell) table?.select(cell.row, cell.col);
    scene.highlight(name, cell);
    showCell.disabled = !cell;
    const node = state.tree ? findNode(state.tree, name) : null;
    if (node?.code?.snippet) {
      tab.showCode(node.code.snippet, `Pesi numerici: ${node.pyPath ?? name}`, [node.code.text]);
      // sul telefono il codice è un foglio che coprirebbe la scena: resta ridotto (come in ViewManager)
      if (narrowScreen()) ui.codePanel?.root.classList.add("minimized");
    }
    if (fly) void scene.focusTensor(name);
    if (reveal) revealCard();
    return true;
  }

  /** Scheda del tensore (testo, tabella, statistiche, istogramma) per la sorgente attuale. */
  function renderCard() {
    const t = state.run.tensor(state.selected);
    const axes = currentAxes();
    const node = state.tree ? findNode(state.tree, state.selected) : null;
    const code = node?.code;
    cardHead.replaceChildren(
      el("h4", {}, "Tensore scelto"),
      el("p", { className: "mono tf-tensor-name" }, state.selected),
      el("p", { className: "tf-meta" }, el("span", {}, "forma ", el("b", {}, shapeText(t.shape))), el("span", {}, el("b", {}, fmt(t.numel)), " valori"), el("span", { title: "Il colore pieno corrisponde a questo valore assoluto (il massimo |w| dei pesi finali): la stessa scala in tutti gli step del training." }, "colore pieno a ±", el("b", {}, fmt(spec.tensors[state.selected]?.scale ?? 0, 3)))),
      code?.text ? el("p", { className: "tf-pyline" }, el("code", {}, code.text), el("small", {}, ` ${code.file}:${code.line}`)) : null,
      el("p", { className: "hint" }, axes.meaning),
    );
    state.cardFor = state.selected;
    state.tableData = new Float32Array(t.numel);
    state.tableData.set(state.views?.get(state.selected) ?? t.data);
    // il nome è già in cima alla scheda: la tabella dice solo che cosa contiene (e la forma)
    table.setTensor({ name: "", shape: t.shape, data: state.tableData, range: spec.tensors[state.selected]?.scale, rowTitle: axes.rowTitle, colTitle: axes.colTitle || "colonna", caption: `I numeri di ${node?.pyPath ?? state.selected}` });
    refreshCard({ heavy: true, table: false });
  }

  /** Nuovi valori per lo stesso tensore (un altro step): tabella sul posto, statistiche e istogramma. */
  function refreshCard({ heavy = true, table: withTable = true } = {}) {
    // la scheda di un altro tensore (cambio di tensore in corso): la ridisegna renderCard
    if (!state.tableData || !state.views || state.cardFor !== state.selected) return;
    const data = state.views.get(state.selected);
    if (!data || data.length !== state.tableData.length) return;
    if (withTable) {
      state.tableData.set(data);
      table.refresh();
    }
    if (!heavy) return;
    const label = sourceLabel(state.source, { steps: state.steps, lastCheck: state.lastCheck, format: preciseText });
    renderTensorStats(statsBox, data, { label: `Statistiche · ${label.short}` });
    const final = state.finalViews.get(state.selected);
    if (state.source.kind === "final") histogram.setData([{ label: "pesi finali", data: final }]);
    else histogram.setData([{ label: label.short, data }, { label: "pesi finali", data: final }]);
  }

  /** Cella scelta nella tabella (clic, frecce, Invio): si illumina nella scena. */
  function tableCell(cell) {
    state.cell = { row: cell.i, col: cell.j };
    scene.highlight(state.selected, state.cell);
    showCell.disabled = false;
  }

  function flyToCell() {
    if (!state.cell) return;
    if (narrowScreen()) ui.sidebar.toggleCollapsed(true); // sul telefono conta la scena
    ui.director.noteUserIntent();
    void scene.focusCell(state.selected, state.cell);
  }

  /** Porta la scheda in vista nella sidebar (desktop); sul telefono il cassetto resta chiuso. */
  function revealCard() {
    if (narrowScreen()) return;
    ui.sidebar.select("viste");
    card.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }

  /** Clic (o tocco) su una cella nella scena: la tabella va su quella cella e la evidenzia. */
  function pickCell(hit, event) {
    selectTensor(hit.name, { cell: { row: hit.row, col: hit.col }, reveal: true });
    if (event?.pointerType === "touch" || !canHover()) showTip(hit, event, { sticky: true });
  }

  /** L'azione del training completo (heatmap, attention e probabilità vere di ogni snapshot): torna all'Architettura da sé. */
  function startTrainingAction() {
    ui.sidebar.select("azioni");
    if (narrowScreen()) ui.sidebar.toggleCollapsed(true);
    void ui.player?.start("training-run");
  }

  async function openInLab() {
    const target = labTarget(state.selected, state.source);
    const shown = await tab.app.goto(LAB_TAB, target);
    // tab non ancora registrato: app.goto non apre niente e l'indirizzo resta qui
    if (!shown && !location.hash.startsWith(`#${LAB_TAB}`)) showToast("Il Laboratorio pesi non è ancora disponibile in questa versione della web app.", { duration: 3500 });
  }

  // ---------------------------------------------------------------- etichetta della cella nella scena
  function showTip(hit, event, { sticky = false } = {}) {
    if (!hit || !state.active) {
      // L'etichetta di un tocco resta fino al suo timer: sul telefono a ogni tocco segue un pointerleave
      // (il dito si stacca), che non deve né spegnerla né fermarne il timer (resterebbe per sempre).
      if (tip.dataset.sticky) return;
      clearTimeout(tipTimer);
      tip.hidden = true;
      return;
    }
    clearTimeout(tipTimer);
    const p = state.run.tensor(hit.name);
    const axes = tensorAxes(hit.name, p.shape, { vocab: state.run.vocab, config: state.run.config });
    const label = sourceLabel(state.source, { steps: state.steps, lastCheck: state.lastCheck, format: preciseText });
    tip.replaceChildren(
      el("strong", {}, hit.name),
      el("span", {}, cellMeaning(axes, hit.row, hit.col)),
      el("span", {}, "valore ", el("b", { className: "tf-num-value" }, preciseText(hit.value)), el("small", {}, ` · ${label.short}`)),
      el("small", { className: "tf-num-hint" }, sticky ? (narrowScreen() ? "Tabella dei numeri: menu ☰ › Viste" : "Aperta nella tabella della sidebar") : "clic: apri nella tabella"),
    );
    tip.hidden = false;
    const rect = ui.stage.getBoundingClientRect();
    const x = (event?.clientX ?? rect.left + rect.width / 2) - rect.left;
    const y = (event?.clientY ?? rect.top + rect.height / 2) - rect.top;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    tip.style.left = `${Math.max(6, Math.min(rect.width - w - 6, x + 14))}px`;
    tip.style.top = `${Math.max(6, Math.min(rect.height - h - 6, y + 14 + h > rect.height ? y - h - 14 : y + 14))}px`;
    if (sticky) {
      tip.dataset.sticky = "1";
      tipTimer = setTimeout(() => {
        delete tip.dataset.sticky;
        tip.hidden = true;
      }, 2600);
    } else delete tip.dataset.sticky;
  }

  // ---------------------------------------------------------------- la vista (registrata da numbers.js)
  const view = {
    enter: async () => {
      // Un'azione in corso scrive nelle stesse texture: si chiude PRIMA di salvarle (al ritorno tornano pulite).
      if (ui.player?.action) ui.player.stop();
      state.active = true;
      arch.root.visible = true;
      arch.rail.visible = true;
      // Il flusso dei dati moltiplica i colori delle matrici che attraversa e riscrive la didascalia: qui si ferma.
      state.saved = { packet: flow.packet.visible, playing: flow.playing };
      flow.playing = false;
      flow.setPulse(flow.current, false);
      flow.current = null;
      flow.packet.visible = false;
      ui.root.classList.add("tf-numbers-on");
      info.hidden = false;
      tab.setCaption(canHover() ? CAPTION.hover : CAPTION.touch);
      if (!built) info.replaceChildren(el("h3", {}, "Pesi numerici"), el("p", { className: "tf-status" }, "Carico i pesi del modello…"));
      try {
        await ensureRun();
      } catch (err) {
        if (state.active) info.replaceChildren(el("h3", {}, "Pesi numerici"), el("p", { className: "tf-status error" }, `Impossibile caricare i pesi: ${err.message}`));
        return;
      }
      if (!state.active) return;
      build();
      bar.element.hidden = false;
      barObserver.observe(bar.element);
      syncSourceUi();
      state.views = state.finalViews;
      state.source = FINAL_SOURCE;
      scene.enter(state.views);
      selectTensor(state.selected, { cell: state.cell });
      syncLabel();
      if (state.mode === "training") await applySource(currentSource());
      if (!state.active) return;
      if (state.cell) void scene.focusCell(state.selected, state.cell, { duration: 1.1 });
      else void scene.focusTensor(state.selected, { duration: 1.1 });
    },
    // "Centra" e "Vista d'insieme": il tensore scelto, intero
    home: () => (state.active && state.run ? scene.focusTensor(state.selected) : tab.focus("overview")),
    exit: () => {
      state.active = false;
      state.seq++;
      stopPlay();
      clearTimeout(trailing);
      clearTimeout(tipTimer);
      delete tip.dataset.sticky;
      scene.exit();
      state.source = FINAL_SOURCE;
      info.hidden = true;
      tip.hidden = true;
      if (bar) bar.element.hidden = true;
      barObserver.disconnect();
      ui.root.style.removeProperty("--tf-numbers-h");
      ui.root.classList.remove("tf-numbers-on");
      if (state.saved) {
        flow.packet.visible = state.saved.packet;
        flow.playing = state.saved.playing;
        state.saved = null;
      }
      tab.setCaption("");
    },
  };

  /**
   * Apre la vista (link diretti, ricerca, Laboratorio): tensore, step del training ed eventuale cella.
   * step: null = pesi finali; uno step di snapshot = dato vero; uno step intermedio = stima lineare.
   * @returns {Promise<boolean>} false se il tensore non esiste
   */
  async function open({ tensor = null, step = undefined, interpolate = false, cell = null, birth = false } = {}) {
    await ensureRun();
    if (tensor && !state.run.tensor(tensor)) return false;
    if (tensor) {
      state.selected = tensor;
      state.cell = cell;
    }
    if (birth) {
      state.mode = "training";
      state.step = state.steps[0];
    } else if (step === null) {
      state.mode = "final";
    } else if (step !== undefined) {
      state.mode = "training";
      state.step = step;
      state.interpolate = Boolean(interpolate);
      interpolateBox.checked = state.interpolate;
      panelStep?.setInterpolate(state.interpolate);
      barStep?.setInterpolate(state.interpolate);
    }
    if (tab.views.current !== NUMBERS_VIEW_ID) {
      await tab.views.activate(NUMBERS_VIEW_ID); // entrando applica sorgente, tensore e cella, e inquadra
      return state.active;
    }
    stopPlay();
    syncSourceUi();
    selectTensor(state.selected, { cell: state.cell }); // prima la scheda del tensore nuovo, poi i valori dello step
    await applySource(currentSource());
    if (state.cell) void scene.focusCell(state.selected, state.cell);
    else void scene.focusTensor(state.selected);
    return true;
  }

  // API della vista (goto, ricerca, console e script di verifica)
  const api = {
    loaded: true,
    open,
    get state() {
      return state;
    },
    get source() {
      return state.source;
    },
    get selected() {
      return state.selected;
    },
    get table() {
      return table;
    },
    scene,
    setStep: (step, o) => setStep(step, o),
    setMode,
    setInterpolate,
    togglePlay,
    selectTensor,
  };
  return { ...view, api };
}
