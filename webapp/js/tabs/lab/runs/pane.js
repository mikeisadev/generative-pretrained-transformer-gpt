/**
 * Pannello "Run" (solo DOM): il GPT del corso e i run caricati, il caricamento (cartella, file,
 * trascinamento sul pannello o sulla scena), il salvataggio nel browser con "Dimentica" e lo spazio
 * occupato, la scheda del run A e "Come esportare un run" con i comandi Python veri.
 *
 *   const runs = createRunsPane(pane, tab);   runs.load(files)  runs.render()  runs.enableCompare()
 */
import { el } from "../../../core/dom.js";
import { copyText, showToast } from "../../../core/ui/share.js";
import { runsFromFiles } from "../../../shared/weights/run.js";
import { LIMITS, megabytes } from "../../../shared/weights/validate.js";
import { ACCEPTED_FILES, EXPORT_STEPS, FLOW } from "../content.js";
import { runCard } from "./card.js";
import { fmtInt, formatBytes } from "./describe.js";
import { droppedFiles, dropZone, filePicker } from "./upload.js";

const REMEMBER_KEY = "lab.rememberRuns";

export function createRunsPane(pane, tab) {
  const { state } = tab;
  let loadToken = 0; // cresce a ogni caricamento: il più recente decide il run A
  const list = el("ul", { className: "lab-run-list", ariaLabel: "Run disponibili" });
  const status = el("div", { className: "lab-upload-status", role: "status", ariaLive: "polite" });
  const storage = el("p", { className: "lab-storage hint" });
  const cardBox = el("div", { className: "lab-card-box" });
  // senza salvataggio nel browser (IndexedDB vietato) la casella non promette niente: spenta e detta
  const remember = el("input", { type: "checkbox", checked: Boolean(state.store) && readRemember(), disabled: !state.store, onchange: (e) => writeRemember(e.target.checked) });
  const folderInput = filePicker({ directory: true, onFiles: (files) => load(files) });
  const filesInput = filePicker({ directory: false, onFiles: (files) => load(files) });
  const dropArea = el(
    "section",
    { className: "lab-drop-area", ariaLabel: "Carica un run" },
    el("h3", { className: "lab-h3" }, "Carica un run"),
    el("p", { className: "hint" }, "La cartella esportata da scripts/export_run.py (per esempio runs/gpt-con-gelu): scegli la cartella, scegli i file o trascinali qui o sulla scena."),
    el(
      "div",
      { className: "lab-upload-buttons" },
      el("button", { type: "button", className: "lab-primary", onclick: () => folderInput.click() }, "Scegli una cartella"),
      el("button", { type: "button", onclick: () => filesInput.click() }, "Scegli i file"),
    ),
    el("label", { className: "lab-check" }, remember, state.store ? " Ricordali in questo browser (restano solo qui, sul tuo computer)" : " Ricordali in questo browser: non disponibile qui (i run restano finché la pagina è aperta)"),
    el(
      "details",
      { className: "lab-accepted" },
      el("summary", {}, "Che cosa posso caricare?"),
      el("ul", {}, ...ACCEPTED_FILES.map((t) => el("li", {}, t))),
      // i file sono dati non fidati: la libreria li controlla (forme, offset, nomi) prima di allocare memoria
      el("p", { className: "hint" }, `Limiti: ${binarySize(LIMITS.maxBinaryBytes)} per file di pesi, ${binarySize(LIMITS.maxJsonBytes)} per file JSON, ${binarySize(LIMITS.maxTotalBytes)} per caricamento. I file restano nel tuo browser: niente viene inviato a un server.`),
    ),
    folderInput,
    filesInput,
    status,
  );
  const guide = exportGuide();
  pane.append(
    el("p", { className: "hint lab-pane-intro" }, "A è il run nella scena: «Mostra» lo cambia. Il GPT del corso c'è sempre."),
    list,
    storage,
    dropArea,
    el("h3", { className: "lab-h3" }, "Il run nella scena"),
    cardBox,
    guide,
  );
  const undrop = dropZone(dropArea, { onDrop: (dt) => loadDropped(dt) });

  // ---------------------------------------------------------------- elenco
  function render() {
    const entries = state.entries;
    list.replaceChildren(...entries.map((e) => item(e)));
    renderCard();
    void renderStorage();
  }

  function item(e) {
    const isA = e.id === state.runA;
    const isB = e.id === state.runB;
    const s = e.summary;
    const facts = s
      ? [`${fmtInt(s.totalParams)} parametri`, s.steps?.length ? `${fmtInt(s.steps.length)} snapshot` : "solo pesi finali", s.compatibleWithEngine ? "motore JS: sì" : "motore JS: no"]
      : ["salvato nel browser"];
    if (e.saved && e.savedBytes) facts.push(`nel browser: ${formatBytes(e.savedBytes)}`);
    const actions = [];
    if (!isA) actions.push(el("button", { type: "button", className: "lab-small", onclick: () => choose(e.id), ariaLabel: `Mostra ${e.name} nella scena (run A)` }, "Mostra"));
    else actions.push(el("span", { className: "lab-badge lab-badge-a", title: "Il run nella scena" }, "A · nella scena"));
    if (state.features.compare) {
      if (isB) actions.push(el("button", { type: "button", className: "lab-small", onclick: () => void state.setRunB(null), ariaLabel: `Togli ${e.name} dal confronto` }, "B ✕"));
      else actions.push(el("button", { type: "button", className: "lab-small", onclick: () => void state.setRunB(e.id), ariaLabel: `Confronta con ${e.name} (run B)` }, "B"));
    }
    if (e.origin !== "builtin" && !e.saved && e.loaded) actions.push(el("button", { type: "button", className: "lab-small", onclick: () => save(e.id), ariaLabel: `Salva ${e.name} in questo browser` }, "Salva"));
    if (e.origin !== "builtin") actions.push(el("button", { type: "button", className: "lab-small lab-danger", onclick: () => forget(e), ariaLabel: `Dimentica ${e.name}` }, "Dimentica"));
    return el(
      "li",
      { className: `lab-run${isA ? " is-a" : ""}${isB ? " is-b" : ""}` },
      el("div", { className: "lab-run-head" }, el("strong", {}, e.name), e.origin === "builtin" ? el("span", { className: "lab-badge" }, "del corso") : ""),
      el("p", { className: "lab-run-facts" }, facts.join(" · ")),
      e.error ? el("p", { className: "lab-error" }, e.error) : "",
      el("div", { className: "lab-run-actions" }, ...actions),
    );
  }

  function renderCard() {
    const entry = state.entry(state.runA);
    cardBox.replaceChildren(entry ? runCard(entry, state.run(entry.id), state.builtin) : el("p", { className: "hint" }, "Nessun run."));
  }

  async function renderStorage() {
    if (!state.store) {
      storage.textContent = "Il salvataggio nel browser non è disponibile qui: i run restano finché la pagina è aperta.";
      return;
    }
    const usage = await state.storageUsage();
    const saved = state.entries.filter((e) => e.saved);
    const quota = usage.quota ? ` · spazio del sito: ${formatBytes(usage.usage ?? 0)} di ${formatBytes(usage.quota)}` : "";
    storage.textContent = saved.length ? `Salvati in questo browser: ${fmtInt(saved.length)} run, ${formatBytes(usage.saved)}${quota}.` : `Nessun run salvato in questo browser${quota}.`;
  }

  /** Il run A (pulsante «Mostra» o un caricamento). Con `token` vale solo se nessun caricamento più recente è partito. */
  async function choose(id, token = null) {
    if (token !== null && token !== loadToken) return false;
    tab.ui.director.noteUserIntent(); // la camera vola sul nuovo modello anche se lo studente si stava muovendo
    return state.setRunA(id);
  }

  // La didascalia dice quale run c'è nella scena, comunque sia cambiato (pannello Run, Confronto, confronti
  // pronti, link, «Dimentica» del run A): scritta solo nel pulsante «Mostra», restava il nome del run di prima.
  // Niente step: la targhetta in alto lo dice e resta aggiornata (la didascalia no, se si muove il cursore).
  let captionRun = null;
  state.on("weights", ({ runId }) => {
    if (runId === captionRun) return;
    const first = captionRun === null;
    captionRun = runId;
    if (first || tab.ui.player?.action) return; // all'avvio la didascalia è di index.js; un'azione ha la sua
    const entry = state.entry(runId);
    const run = state.run(runId);
    tab.setCaption(`Nella scena: «${entry?.name ?? ""}»${run ? `, ${fmtInt(run.totalParams)} parametri` : ""}. Avvicinati per le celle colorate e poi i numeri veri.`);
  });

  async function save(id) {
    const result = await state.saveRun(id);
    showStatus(result.ok ? [`«${state.entry(id)?.name}» salvato in questo browser (${formatBytes(result.bytes)}).`] : [], result.ok ? [] : [`Non salvato: ${result.reason}.`]);
  }

  async function forget(e) {
    const what = e.saved ? "dalla pagina e da questo browser" : "dalla pagina";
    if (!window.confirm(`Dimenticare «${e.name}»? Lo tolgo ${what}; potrai ricaricarlo dai file.`)) return;
    const ok = await state.forget(e.id);
    if (ok) showToast(`«${e.name}» dimenticato.`);
  }

  // ---------------------------------------------------------------- caricamento
  async function loadDropped(dataTransfer) {
    try {
      const { files, truncated } = await droppedFiles(dataTransfer);
      if (truncated) showStatus([], [`Troppi file nel trascinamento: letti i primi ${fmtInt(files.length)}. Trascina solo la cartella del run.`]);
      if (files.length) await load(files);
    } catch (error) {
      showStatus([], [`Non riesco a leggere i file trascinati: ${error.message}`]);
    }
  }

  /**
   * Carica dei file (FileList, array di File o "tipo File"): run riconosciuti, validati e aggiunti al
   * registro; il primo nuovo diventa il run A. Errori per run, in italiano, con file e campo.
   */
  async function load(files) {
    // il gettone si prende subito: se lo studente carica altro mentre questi file si leggono, vince l'ultimo
    // caricamento chiesto (per il run A e per la riga di stato); i run di quello prima entrano comunque nell'elenco
    const token = ++loadToken;
    const latest = () => token === loadToken;
    const total = files.reduce((s, f) => s + (f.size ?? 0), 0);
    status.replaceChildren(el("p", { className: "lab-busy" }, `Leggo ${fmtInt(files.length)} file (${megabytes(total)})…`));
    tab.ui.sidebar.select("run");
    let result;
    try {
      result = await runsFromFiles(files, { defaultVocab: state.builtin?.vocab ?? null });
    } catch (error) {
      showStatus([], [error.message], null, latest()); // oltre il limite totale: nessun file letto
      return { added: [], errors: [error.message] };
    }
    const { added, duplicates } = state.addRuns(result.runs);
    const ok = [];
    // i file sciolti (senza cartella) formano il run "."
    const bad = result.errors.map((e) => `${!e.run || e.run === "." ? "File scelti" : `«${e.run}»`}: ${e.message}`);
    for (const run of result.runs) {
      const isNew = added.includes(run.id);
      ok.push(`${isNew ? "Caricato" : "Già caricato"} «${run.name}»: ${fmtInt(run.totalParams)} parametri, ${run.steps.length ? `${fmtInt(run.steps.length)} snapshot` : "solo pesi finali"}${run.compatibleWithEngine ? "" : ", il motore JS non lo esegue"}.`);
      for (const w of run.meta.warnings) ok.push(`Avviso per «${run.name}»: ${w}.`);
    }
    if (result.ignored.length) ok.push(`Ignorati ${fmtInt(result.ignored.length)} file: ${result.ignored.slice(0, 4).map((i) => `${i.path} (${i.reason})`).join("; ")}${result.ignored.length > 4 ? "…" : ""}.`);
    if (!result.runs.length && !bad.length) bad.push("Nessun run riconosciuto: serve una cartella con weights.json (+ weights.f32.bin), run.json, un .safetensors oppure trace.json con gli snapshot.");
    const saving = remember.checked && state.store && added.length > 0;
    showStatus(ok, bad, saving ? "Salvo in questo browser…" : null, latest());
    // Prima si mostra il run (lo studente lo vede subito nella scena), poi si salva nel browser: salvare
    // un run grande richiede secondi e "Caricato «…»" con la scena ancora sul run di prima confonde.
    const first = added[0] ?? duplicates[0];
    if (first) await choose(first, token);
    if (saving) {
      const failed = [];
      for (const [k, id] of added.entries()) {
        if (!state.has(id)) continue; // dimenticato nel frattempo
        if (latest() && added.length > 1) showStatus(ok, bad, `Salvo in questo browser: ${fmtInt(k + 1)} di ${fmtInt(added.length)}…`);
        const saved = await state.saveRun(id);
        if (!saved.ok) failed.push(`«${state.entry(id)?.name ?? id}» non salvato nel browser: ${saved.reason}.`);
      }
      if (latest()) showStatus(ok, [...bad, ...failed]);
      else if (failed.length) showStatus([], failed, null, false);
    }
    return { added, duplicates, errors: result.errors };
  }

  /** La riga di stato del caricamento; `replace: false` (un caricamento superato) aggiunge invece di sostituire. */
  function showStatus(ok, bad, busy = null, replace = true) {
    const lines = [...ok.map((t) => el("p", { className: "lab-ok-line" }, t)), ...bad.map((t) => el("p", { className: "lab-error" }, t)), ...(busy ? [el("p", { className: "lab-busy" }, busy)] : [])];
    if (replace) status.replaceChildren(...lines);
    else status.append(...lines);
  }

  state.on("runs", () => render());
  state.on("runA", () => render());
  state.on("runB", () => render());
  state.on("features", () => render());
  state.on("error", ({ message }) => showStatus([], [message]));
  render();

  return {
    render,
    load,
    loadDropped,
    /** La fase 2 (confronto) accende i pulsanti "B" accanto a ogni run. */
    enableCompare: () => state.enableFeature("compare"),
    openGuide() {
      guide.open = true;
      guide.scrollIntoView({ block: "start", behavior: "smooth" });
      guide.querySelector("summary")?.focus();
    },
    openUpload() {
      dropArea.scrollIntoView({ block: "start", behavior: "smooth" });
      dropArea.querySelector("button")?.focus();
    },
    dispose: () => undrop(),
  };
}

function exportGuide() {
  return el(
    "details",
    { className: "lab-guide" },
    el("summary", {}, "Come esportare un run dal Python"),
    el("p", { className: "hint" }, "Il flusso «prima e dopo»: ogni miglioramento (pesi, iperparametri o architettura) diventa una cartella che carichi qui accanto al GPT del corso."),
    el("ol", { className: "lab-flow" }, ...FLOW.map((f) => el("li", {}, el("strong", {}, f.title), ` ${f.text}`))),
    ...EXPORT_STEPS.map((s) =>
      el(
        "div",
        { className: "lab-cmd" },
        el("h4", { className: "lab-h4" }, s.title),
        el("p", { className: "hint" }, s.note),
        el("div", { className: "lab-cmd-row" }, el("code", {}, s.command), el("button", { type: "button", className: "lab-small", onclick: () => copyText(s.command).then(() => showToast("Comando copiato.")), ariaLabel: `Copia il comando: ${s.title}` }, "Copia")),
      ),
    ),
    el("p", { className: "hint" }, "I run esportati stanno in runs/, che git ignora. Formati e regole: README, sezione «Esportare un run e confrontarlo nel Laboratorio pesi»."),
  );
}

/** I limiti della libreria sono in multipli di 1024: "1 GB", "64 MB", "1,5 GB". */
function binarySize(bytes) {
  const MiB = 1024 * 1024;
  const n = (v) => v.toLocaleString("it-IT", { maximumFractionDigits: 1 });
  return bytes >= 1024 * MiB ? `${n(bytes / (1024 * MiB))} GB` : `${n(bytes / MiB)} MB`;
}

function readRemember() {
  try {
    return localStorage.getItem(REMEMBER_KEY) !== "0";
  } catch {
    return true;
  }
}
function writeRemember(on) {
  try {
    localStorage.setItem(REMEMBER_KEY, on ? "1" : "0");
  } catch {
    // senza localStorage la scelta vale per questa pagina
  }
}
