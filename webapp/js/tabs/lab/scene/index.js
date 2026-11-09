/**
 * La scena del Laboratorio collegata al resto del tab: stato (run A, step, scelta), LOD con il suo
 * selettore, mouse e tocco (passaggio = valore e significato della cella, clic = scelta), navigazione
 * libera ("Centra" = tutto il modello, confini, limiti di rotazione da scena "a parete"), numeri nelle
 * celle (un NumbersGroup per la scena, aggiornato a ogni frame: a riposo esce subito).
 *
 *   const scene = installScene(tab);   // tab.scene = l'API di LabScene + focusTensor/focusModule/focusCell/home
 */
import * as THREE from "three";
import { LODController } from "../../../core/lod.js";
import { el, prefersReducedMotion } from "../../../core/dom.js";
import { formatPrecise } from "../../../shared/weights/display.js";
import { pythonPath } from "../../../shared/weights/moduleTree.js";
import { tensorMeaning } from "../inspect/meaning.js";
import { LabScene } from "./labScene.js";
import { fmtInt } from "./layout.js";

/** Soglie del LOD (distanza camera → punto guardato) per lo stage di riferimento 1600 × 900. */
export const LOD_DISTANCES = Object.freeze({ detail: 48, blocks: 135 });
const REFERENCE_ASPECT = 1.44;
const CLICK_PX = 5;
const CLICK_MS = 650;
const TOUCH_TIP_MS = 2600;
const FRONT = new THREE.Vector3(0, 0.02, 1).normalize();

export function installScene(tab) {
  const { ui, state } = tab;
  const { host, director, nav, stage } = ui;
  const scene = new LabScene();
  host.scene.add(scene.root);
  const narrow = window.matchMedia("(max-width: 900px)");
  // l'etichetta del mouse serve già al LOD (che annuncia il livello iniziale nel costruttore)
  const tip = el("div", { className: "lab-tip", role: "tooltip", hidden: true });
  stage.append(tip);
  let tipTimer = 0;

  // --- LOD: Dettaglio (matrici e numeri) / Moduli / Panoramica
  // Su uno stage più stretto (telefono, codice aperto) per vedere lo stesso modulo la camera sta più lontana:
  // le soglie si moltiplicano per un fattore. Il LOD lo chiede a ogni frame e visibleAspect() legge il layout
  // (misurato: 40-75 µs per frame a riposo): si ricalcola solo dopo un ridimensionamento, e vale dal primo
  // movimento della camera che cambia la distanza (rotellina, volo). Ridurre il codice o chiudere i grafici
  // allarga la parte visibile ma lascia le celle grandi uguali sullo schermo: cambiare subito il fattore
  // faceva sparire le celle (Dettaglio → Moduli) con la camera ferma.
  const scaleNow = () => Math.min(3, Math.max(1, REFERENCE_ASPECT / Math.max(0.2, host.visibleAspect())));
  let distanceScale = scaleNow();
  let pendingScale = false;
  let distanceAtResize = 0;
  host.onResize(() => {
    pendingScale = true;
    distanceAtResize = host.zoomDistance();
  });
  const currentScale = () => {
    if (pendingScale) {
      const d = host.zoomDistance();
      if (Math.abs(d - distanceAtResize) > 1e-3 * Math.max(1, d)) {
        distanceScale = scaleNow();
        pendingScale = false;
      }
    }
    return distanceScale;
  };
  const lod = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: LOD_DISTANCES.detail, group: scene.levels.detail },
      { id: "blocks", label: "Moduli", maxDistance: LOD_DISTANCES.blocks, group: scene.levels.blocks },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: scene.levels.macro },
    ],
    widgetParent: stage,
    onChange: (id) => {
      scene.setLevel(id);
      hideTip();
    },
    onPick: (mode) => pickLevel(mode),
    distanceScale: currentScale,
  });

  // --- numeri nelle celle: una chiamata per frame, che a camera ferma esce subito; solo nel Dettaglio (negli
  // altri livelli le matrici sono nascoste e setLevel invalida il gruppo: tornando giù i numeri si rifanno)
  host.onFrame(() => {
    if (scene.level === "detail") scene.numbers.update(host.camera, host.container.clientHeight);
  });

  // --- navigazione: una parete da guardare di fronte (niente testo a specchio, niente vista dall'alto),
  // e abbastanza vicino da leggere i numeri (doc di matrixNumbers: minDistance 0,15-0,25)
  nav.setOrbitLimits({ turn: 65, tilt: 50, minDistance: 0.18 });
  nav.setBounds(() => (scene.layout ? scene.bounds().expandByScalar(1.5) : null));
  nav.setHome(() => home());

  // --- costruzione per il run A (anche con un'architettura diversa: l'albero si fa dai nomi)
  let builtFor = null;
  function rebuild(run, view) {
    const t0 = performance.now();
    const layout = tab.layoutOf(run);
    scene.build({ run, layout, view, config: run.config, vocab: run.vocab });
    builtFor = run.id;
    scene.setSelected(state.selected);
    tab.stats.buildMs = performance.now() - t0;
  }
  // Con la Rete neurale o il Cervello aperti la scena si ricostruisce senza voli: un volo chiesto qui
  // riporterebbe all'Architettura (ritorno automatico del nucleo). Si inquadra il nuovo run al ritorno.
  let homeOnReturn = false;
  state.on("runA", ({ run }) => {
    hideTip();
    rebuild(run, null);
    if (tab.reps && tab.reps.current !== "main") {
      homeOnReturn = true;
      return;
    }
    void home({ duration: prefersReducedMotion() ? 0.01 : 0.9 });
  });
  state.on("weights", ({ runId, view }) => {
    if (runId !== builtFor) return; // una ricostruzione per un altro run è in arrivo
    scene.setWeights(view.tensors);
  });
  state.on("select", ({ path }) => scene.setSelected(path));

  // --- inquadrature
  function frame(box, { margin = 1.2, minDistance = 0.6, duration } = {}) {
    if (!box || box.isEmpty()) return Promise.resolve();
    return director.focus(box, { showFrame: false, margin, minDistance, direction: FRONT, duration: duration ?? (prefersReducedMotion() ? 0.3 : 1.1) });
  }
  function home(options = {}) {
    if (!scene.layout) return Promise.resolve();
    if (lod.mode !== "auto") lod.setMode("auto");
    return frame(scene.bounds(), { margin: narrow.matches ? 1.12 : 1.04, minDistance: 1, ...options });
  }
  function focusTensor(name, options = {}) {
    return frame(scene.boxOf(name), { margin: 1.35, minDistance: 0.5, ...options });
  }
  function focusModule(path, options = {}) {
    if (!path) return home(options);
    return frame(scene.boxOf(path), { margin: 1.12, minDistance: 0.8, ...options });
  }
  /** Vola vicino a una cella, abbastanza da leggerne il numero (celle di ~70 px sullo schermo). */
  function focusCell(name, i, j, { duration } = {}) {
    const where = scene.layoutOf(name);
    if (!where) return Promise.resolve();
    const target = where.cellCenter(i, j);
    const fov = THREE.MathUtils.degToRad(host.camera.fov);
    const cellPx = 70;
    const distance = Math.max(0.2, (Math.min(where.cellW, where.cellH) * host.container.clientHeight) / (2 * Math.tan(fov / 2) * cellPx));
    if (lod.mode !== "auto" && lod.mode !== "detail") lod.setMode("auto");
    return director.flyTo(target.clone().addScaledVector(FRONT, distance), target, duration ?? (prefersReducedMotion() ? 0.3 : 0.9));
  }
  function pickLevel(mode) {
    if (mode === "auto" || ui.player?.action) return;
    if (mode === "macro") return void frame(scene.bounds(), { margin: 1.04, minDistance: 1 });
    if (mode === "blocks") return void focusModule(scene.layout?.nodes.has("blocks.0") ? "blocks.0" : "", {});
    const sel = state.selected && scene.nodeOf(state.selected) ? state.selected : scene.layout?.nodes.has("blocks.0.sa.heads.0") ? "blocks.0.sa.heads.0" : null;
    return void (sel ? focusModule(sel) : home());
  }

  // --- mouse e tocco: passaggio = che cosa c'è sotto, clic = scelta
  let hoverRaf = 0;
  let pointer = null;
  let down = null;
  const canvas = host.renderer.domElement;
  const pickAt = (x, y) => {
    const hits = host.pick({ clientX: x, clientY: y }, scene.pickables(), false);
    return scene.pickFrom(host.raycaster.ray, hits);
  };
  function describe(hit) {
    if (!hit) return null;
    const node = hit.node;
    const name = pythonPath(hit.path);
    if ((hit.kind === "tensor" || hit.kind === "buffer") && hit.cell) {
      const meaning = tensorMeaning(hit.path, node?.shape ?? [], { vocab: scene.vocab, config: scene.config });
      const vector = node?.matrix?.vector;
      const i = vector === "row" ? hit.cell.col : hit.cell.row;
      const j = vector ? null : hit.cell.col;
      const custom = scene.customValues.has(hit.path) ? ` (${scene.valuesLabel || "valori mostrati"})` : "";
      const lines = [`${name}  [${j === null ? i : `${i}, ${j}`}]`, meaning.rows.describe(i)];
      if (j !== null && meaning.cols) lines.push(meaning.cols.describe(j));
      lines.push(`valore${custom}: ${hit.cell.masked ? "0 (futuro, mascherato)" : formatPrecise(hit.cell.value)}`);
      return { lines, cell: hit.cell };
    }
    if (hit.kind === "tensor") return { lines: [name, `${fmtInt(node?.params ?? 0)} pesi · ${scene.limited.includes(hit.path) ? "troppo grande per la scena: i numeri sono nel pannello Tensore" : ""}`] };
    const title = node?.header?.title ?? node?.title ?? name;
    return { lines: [name || title, node?.kind === "virtual" ? node.note : `${node?.type ?? "modulo"} · ${fmtInt(node?.params ?? 0)} parametri`] };
  }
  function showTip(info, x, y, { touch = false } = {}) {
    if (!info) return hideTip();
    tip.replaceChildren(...info.lines.filter(Boolean).map((line, k) => el(k === 0 ? "strong" : "span", {}, line)));
    tip.hidden = false;
    const rect = stage.getBoundingClientRect();
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = x - rect.left + 14;
    let top = y - rect.top + (touch ? -h - 22 : 16);
    if (left + w > rect.width - 8) left = Math.max(8, x - rect.left - w - 14);
    if (top + h > rect.height - 8) top = Math.max(8, y - rect.top - h - 14);
    if (top < 8) top = 8;
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
  function hideTip() {
    tip.hidden = true;
    scene.hoverCell(null, null);
    clearTimeout(tipTimer);
  }
  function hover() {
    hoverRaf = 0;
    if (!pointer || !host.running) return;
    const hit = pickAt(pointer.x, pointer.y);
    const info = describe(hit);
    scene.hoverCell(info?.cell ? hit.path : null, info?.cell ?? null);
    if (info) showTip(info, pointer.x, pointer.y);
    else hideTip();
  }
  const listeners = [
    [canvas, "pointermove", (e) => {
      if (e.pointerType === "touch" || down?.moved) return;
      pointer = { x: e.clientX, y: e.clientY };
      if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > CLICK_PX) {
        down.moved = true;
        hideTip();
        return;
      }
      if (!hoverRaf) hoverRaf = requestAnimationFrame(hover);
    }],
    [canvas, "pointerleave", (e) => {
      pointer = null;
      if (e.pointerType !== "touch") hideTip();
    }],
    [canvas, "pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY, t: performance.now(), type: e.pointerType, moved: false };
    }],
    [canvas, "pointerup", (e) => {
      const d = down;
      down = null;
      if (!d || d.moved || Math.hypot(e.clientX - d.x, e.clientY - d.y) > CLICK_PX || performance.now() - d.t > CLICK_MS) return;
      const hit = pickAt(e.clientX, e.clientY);
      if (d.type === "touch") {
        // sul telefono non c'è il passaggio del mouse: il tocco mostra anche il valore per qualche secondo
        const info = describe(hit);
        if (info) {
          showTip(info, e.clientX, e.clientY, { touch: true });
          scene.hoverCell(info.cell ? hit.path : null, info.cell ?? null);
          tipTimer = setTimeout(hideTip, TOUCH_TIP_MS);
        }
      }
      if (!hit) return;
      director.noteUserIntent();
      tab.inspect(hit.path, { cell: hit.cell ? cellIndex(hit) : null, source: "scene" });
    }],
    [canvas, "pointercancel", () => (down = null)],
  ];
  for (const [target, type, fn] of listeners) target.addEventListener(type, fn);
  host.onStop?.(() => hideTip());

  /** Riga e colonna della cella nel tensore (un vettore in riga ha l'elemento nelle colonne). */
  function cellIndex(hit) {
    const vector = hit.node?.matrix?.vector;
    if (vector === "row") return { i: hit.cell.col, j: 0 };
    return { i: hit.cell.row, j: hit.cell.col };
  }

  const api = {
    /** Collega il selettore delle rappresentazioni (dopo installRepresentations). */
    attachReps(reps) {
      reps.on("change", (id) => {
        if (id !== "main" || !homeOnReturn) return;
        homeOnReturn = false;
        void home({ duration: prefersReducedMotion() ? 0.01 : 0.9 });
      });
    },
    scene,
    lod,
    home,
    focusTensor,
    focusModule,
    focusCell,
    rebuild,
    hideTip,
    /** Per i test e la console: il percorso sotto un punto dello schermo. */
    pickAt,
    dispose() {
      for (const [target, type, fn] of listeners) target.removeEventListener(type, fn);
      cancelAnimationFrame(hoverRaf);
      clearTimeout(tipTimer);
      tip.remove();
      lod.dispose();
      scene.dispose();
      scene.numbers.dispose();
    },
  };
  return api;
}
