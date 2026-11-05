/**
 * Vista "Pesi": DOVE stanno i parametri del modello (209.729 per il GPT del corso).
 *
 * È una sovrapposizione sulla scena dell'architettura:
 *   - ogni tensore appreso resta acceso, con il bordo color ambra e un'etichetta per componente
 *     (nome PyTorch, forma, numero di parametri, quota del totale);
 *   - ciò che NON è un peso (token, somme, attivazioni, pattern di attention, pacchetto del flusso)
 *     viene opacizzato;
 *   - le maschere `tril` (buffer, non apprese) compaiono grigie e tratteggiate sopra i pattern;
 *   - nei livelli astratti del LOD: "indicatori di riempimento" proporzionali ai pesi (Blocchi)
 *     e una barra impilata con la ripartizione del totale (Panoramica);
 *   - passando col mouse su una matrice compare il suo nome; un clic la descrive nella sidebar
 *     e ne disegna l'istogramma dei valori veri (pesi float32 del motore JS).
 * Uscendo dalla vista tutto torna esattamente come prima.
 */
import * as THREE from "three";
import { meshBounds } from "../../../core/director.js";
import { el, fmt } from "../../../core/dom.js";
import { label } from "../../../core/visuals.js";
import { setBaseOpacity } from "../lod.js";
import { KIND_COLORS } from "./archLevels.js";
import { createViewInfo, deepBox, formula, legend, sources, topChartsSection } from "./panel.js";
import { componentOf, componentTotals, countParams, partsSummary, shapeText } from "./paramGroups.js";

const DIM = 0.12; // opacità relativa di ciò che non è un peso
const AMBER = 0xf59e0b;
/** Intero con il punto delle migliaia dentro KaTeX (il punto va protetto con {.}). */
const texInt = (n) => fmt(n).replace(/\./g, "{.}");
const pct = (n, total) => `${((n / total) * 100).toLocaleString("it-IT", { maximumFractionDigits: n / total < 0.01 ? 2 : 1 })}%`;

/** Maschera tril (T×T): 1 sotto e sulla diagonale (grigio), 0 sopra (tratteggio scuro). */
function trilTexture(T) {
  const rgba = new Uint8Array(T * T * 4);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const o = ((T - 1 - i) * T + j) * 4; // DataTexture parte dal basso: riga 0 in alto
      if (j <= i) rgba.set([148, 163, 184, 235], o);
      else {
        const stripe = (i + j) % 4 < 2 ? 70 : 30;
        rgba.set([stripe, stripe + 6, stripe + 14, 200], o);
      }
    }
  }
  const tex = new THREE.DataTexture(rgba, T, T, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** Componenti con un'etichetta nella scena di dettaglio (id del registro). */
function labelledComponents(cfg) {
  const ids = ["tok_emb", "pos_emb"];
  for (let l = 0; l < cfg.n_layer; l++) ids.push(...["ln1", "heads", "proj", "ln2", "ffwd"].map((c) => `blocks.${l}.${c}`));
  ids.push("ln_f", "lm_head");
  return ids;
}

/** Testo dell'etichetta di un componente: nomi dei tensori e conteggio. */
function componentLabel(id, entry, total) {
  const names = entry.tensors.map((t) => t.name.replace(/^blocks\.\d+\./, ""));
  let name;
  if (id.endsWith(".heads")) name = `sa.heads.*.query/key/value · ${entry.tensors.length} tensori`;
  else if (names.length === 1) name = names[0];
  else name = names.map((n) => n.split(".").slice(0, -1).join(".")).filter((v, i, a) => a.indexOf(v) === i).join(" + ") + " (weight + bias)";
  const shapes = entry.tensors.length <= 4 ? entry.tensors.map((t) => shapeText(t.shape)).join(" + ") : `${entry.tensors.length / 3} head × 3 × ${shapeText(entry.tensors[0].shape)}`;
  return { title: name, sub: `${shapes} = ${fmt(entry.count)} · ${pct(entry.count, total)}` };
}

export function installWeightsView(tab) {
  const { ui, spec, arch, flow } = tab;
  const { host } = ui;
  const cfg = spec.config;
  const total = spec.total_params;
  const totals = componentTotals(spec.params);
  const parts = partsSummary(spec.params, cfg.n_layer);
  const info = createViewInfo(tab, "pesi");

  // ---------------------------------------------------------------- oggetti della sovrapposizione (creati una volta)
  let overlay = null;
  function buildOverlay() {
    const detail = new THREE.Group();
    detail.name = "weights-overlay-detail";
    // Etichette per componente, sotto il componente (sopra c'è già l'etichetta originale).
    const inv = new THREE.Matrix4().copy(arch.root.matrixWorld).invert();
    for (const id of labelledComponents(cfg)) {
      const object = arch.registry.get(id);
      const entry = totals.get(id);
      if (!object || !entry) continue;
      const box = meshBounds(object).applyMatrix4(inv);
      const { title, sub } = componentLabel(id, entry, total);
      const tag = label(title, { sub, size: id.endsWith(".ln1") || id.endsWith(".ln2") || id === "ln_f" ? 0.14 : 0.2, color: "#fbbf24", subColor: "#fde68a" });
      tag.position.set((box.min.x + box.max.x) / 2, box.min.y - 0.55, 0.8);
      detail.add(tag);
    }
    // Maschere tril sopra i pattern di attention: buffer, non apprese.
    const tex = trilTexture(cfg.block_size);
    const trilMaterial = new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false });
    const trils = [];
    for (let l = 0; l < cfg.n_layer; l++) {
      for (let h = 0; h < cfg.n_head; h++) {
        const pattern = arch.registry.get(`blocks.${l}.heads.${h}.pattern`);
        const plane = new THREE.Mesh(pattern.geometry, trilMaterial);
        plane.userData.noBounds = true;
        const world = pattern.getWorldPosition(new THREE.Vector3()).applyMatrix4(inv);
        plane.position.copy(world).setZ(world.z + 0.03);
        detail.add(plane);
        trils.push(plane);
        if (h === cfg.n_head - 1) {
          const tag = label(`tril (${cfg.block_size}, ${cfg.block_size}): buffer, NON appreso`, { sub: "0 parametri · 1 = può guardare, tratteggio = futuro", size: 0.13, color: "#cbd5e1" });
          tag.position.set(plane.position.x, plane.position.y - 1.0, 0.9);
          detail.add(tag);
        }
      }
    }
    arch.root.add(detail);

    // Livello Blocchi: un indicatore ambra dentro ogni rettangolo, alto in proporzione ai pesi.
    const blocks = new THREE.Group();
    blocks.name = "weights-overlay-blocks";
    const rects = tab.lod.arch.rects.blocks;
    // Le stazioni del flusso hanno gli stessi id dei componenti, tranne "embeddings" (token + posizione).
    const countOf = (stationId) => (stationId === "embeddings" ? totals.get("tok_emb").count + totals.get("pos_emb").count : totals.get(stationId)?.count ?? 0);
    const max = Math.max(...[...rects.keys()].map(countOf));
    for (const [stationId, r] of rects) {
      const n = countOf(stationId);
      const h = Math.max(0.12, (n / max) * r.h);
      const gauge = new THREE.Mesh(new THREE.PlaneGeometry(r.w - 0.2, h), new THREE.MeshBasicMaterial({ color: n ? AMBER : 0x64748b, transparent: true, opacity: n ? 0.28 : 0.5, depthWrite: false }));
      gauge.position.set(r.x, r.y - r.h / 2 + h / 2, 0.45);
      blocks.add(gauge);
    }
    tab.lod.arch.blocks.add(blocks);

    // Panoramica: barra impilata con la ripartizione del totale.
    const macro = new THREE.Group();
    macro.name = "weights-overlay-macro";
    const m = tab.lod.arch.macroInfo;
    const x = m.width / 2 + 4.5;
    const span = m.yTop - m.yBottom;
    let y = m.yBottom;
    const segments = [
      ["Input (embedding)", m.counts.input, 0xf59e0b],
      [`${cfg.n_layer} blocchi transformer`, m.counts.blocks, 0x38bdf8],
      ["Output (ln_f + lm_head)", m.counts.output, 0xf472b6],
    ];
    for (const [name, n, color] of segments) {
      const h = (n / total) * span;
      const seg = new THREE.Mesh(new THREE.BoxGeometry(3, Math.max(h, 0.5), 0.4), new THREE.MeshBasicMaterial({ color }));
      seg.position.set(x, y + h / 2, 0);
      macro.add(seg);
      const tag = label(`${name}: ${fmt(n)} (${pct(n, total)})`, { size: 2.6, color: "#e2e8f0" });
      tag.center.set(0, 0.5);
      // le due fette sottili (input, output) hanno l'etichetta spostata fuori dalla barra
      const ty = h < 6 ? (y < m.yBottom + 1 ? y - 2 : y + h + 2) : y + h / 2;
      tag.position.set(x + 2.5, ty, 0.5);
      macro.add(tag);
      y += h;
    }
    const head = label(`Dove sono i ${fmt(total)} parametri`, { size: 3, color: "#fbbf24" });
    head.center.set(0, 0.5);
    head.position.set(x - 1.5, m.yTop + 5, 0.5);
    macro.add(head);
    tab.lod.arch.macro.add(macro);

    overlay = { detail, blocks, macro, trils };
    setOverlayVisible(false);
  }

  function setOverlayVisible(visible) {
    if (!overlay) return;
    overlay.detail.visible = visible;
    overlay.blocks.visible = visible;
    overlay.macro.visible = visible;
  }

  // ---------------------------------------------------------------- opacizzazione di ciò che non è un peso
  const saved = []; // { material, base, color? }
  function remember(material) {
    if (saved.some((s) => s.material === material)) return null;
    const record = { material, base: material.userData.baseOpacity ?? material.opacity };
    saved.push(record);
    return record;
  }

  /** Un oggetto "parla di pesi"? (tensori, loro cornici, etichette dei componenti con pesi) */
  const weightGroups = new Set();
  function isWeight(object) {
    if (object.userData.tensorName) return true;
    if (object.parent?.userData.tensorName) return true; // cornice o etichetta di una matrice
    return object.isSprite && weightGroups.has(object.parent);
  }

  function applyDim() {
    weightGroups.clear();
    for (const id of labelledComponents(cfg)) weightGroups.add(arch.registry.get(id));
    for (let l = 0; l < cfg.n_layer; l++) {
      for (let h = 0; h < cfg.n_head; h++) weightGroups.add(arch.registry.get(`blocks.${l}.heads.${h}`));
      weightGroups.add(arch.registry.get(`blocks.${l}`)); // etichetta "Block l · N pesi"
    }
    weightGroups.add(arch.registry.get("embeddings"));
    arch.root.traverse((o) => {
      if (!o.material || overlay?.detail === o || isOverlay(o)) return;
      if (isWeight(o)) {
        // Cornice della matrice in ambra: "questo è un parametro appreso".
        if (o.isLineSegments && o.parent?.userData.tensorName) {
          const r = remember(o.material);
          if (r) {
            r.color = o.material.color.getHex();
            o.material.color.setHex(AMBER);
          }
        }
        return;
      }
      const r = remember(o.material);
      if (r) setBaseOpacity(o.material, r.base * DIM);
    });
    for (const o of [arch.rail, ...flow.packet.children]) {
      if (!o.material) continue;
      const r = remember(o.material);
      if (r) setBaseOpacity(o.material, r.base * DIM);
    }
    flow.packet.children.forEach((c) => c.isLight && (c.userData.savedIntensity = c.intensity, (c.intensity = 0.4)));
  }

  function isOverlay(o) {
    for (let p = o; p; p = p.parent) if (p === overlay?.detail) return true;
    return false;
  }

  function restoreDim() {
    for (const { material, base, color } of saved) {
      setBaseOpacity(material, base);
      if (color !== undefined) material.color.setHex(color);
    }
    saved.length = 0;
    flow.packet.children.forEach((c) => c.isLight && c.userData.savedIntensity !== undefined && (c.intensity = c.userData.savedIntensity));
  }

  // ---------------------------------------------------------------- puntatore: nome al passaggio, scheda al clic
  const tooltip = el("div", { className: "tf-tooltip", hidden: true });
  ui.stage.append(tooltip);
  const pickables = [...arch.tensorMeshes.values()].flat();
  let hoverName = null;
  let pending = null;
  let downAt = null;
  const canvas = host.renderer.domElement;

  function tensorAt(event) {
    if (tab.lod.current !== "detail") return null;
    const hit = host.pick(event, pickables, false)[0];
    return hit?.object.userData.tensorName ?? null;
  }
  const hideTooltip = () => {
    tooltip.hidden = true;
    hoverName = null;
  };
  function onMove(event) {
    if (pending) return;
    pending = requestAnimationFrame(() => {
      pending = null;
      const name = tensorAt(event);
      if (!name) {
        tooltip.hidden = true;
        hoverName = null;
        return;
      }
      const p = spec.params.find((x) => x.name === name);
      if (name !== hoverName) {
        hoverName = name;
        tooltip.replaceChildren(el("strong", {}, name), el("span", {}, `${shapeText(p.shape)} = ${fmt(p.numel)} parametri · appreso`), el("small", {}, "clic: dettagli e istogramma"));
      }
      const rect = ui.stage.getBoundingClientRect();
      tooltip.style.left = `${event.clientX - rect.left + 14}px`;
      tooltip.style.top = `${event.clientY - rect.top + 14}px`;
      tooltip.hidden = false;
    });
  }
  const onDown = (e) => (downAt = [e.clientX, e.clientY]);
  function onUp(event) {
    if (!downAt || Math.hypot(event.clientX - downAt[0], event.clientY - downAt[1]) > 5) return;
    const name = tensorAt(event);
    if (name) selectTensor(name, true);
  }

  // ---------------------------------------------------------------- sidebar e grafici
  const card = el("div", { className: "tf-card", hidden: true });
  let statsChart = null;
  let partsChart = null;

  /** @param {boolean} [openCharts] apre il pannello dei grafici (solo su clic dell'utente, non durante l'azione) */
  let selectToken = 0; // solo l'ultima selezione aggiorna la scheda (i clic rapidi non si sovrascrivono a vicenda)
  async function selectTensor(name, openCharts = false) {
    const token = ++selectToken;
    const p = spec.params.find((x) => x.name === name);
    const component = componentOf(name);
    card.hidden = false;
    card.replaceChildren(
      el("h4", {}, "Tensore selezionato"),
      el("p", { className: "mono tf-tensor-name" }, name),
      el("p", {}, `Forma ${shapeText(p.shape)} → ${fmt(p.numel)} parametri (${pct(p.numel, total)} del modello). Componente: ${component}.`),
      el("p", { className: "hint" }, "Caricamento dei valori veri…"),
    );
    const highlight = parts.findIndex((part) => partOfTensor(part, name));
    partsChart?.setData({ labels: parts.map((x) => x.short), values: parts.map((x) => x.count), highlight, highlightLabel: parts[highlight]?.label });
    const engine = await tab.getEngine();
    const view = engine.paramViews.get(name);
    if (!view || token !== selectToken) return;
    const v = view.data;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
      min = Math.min(min, v[i]);
      max = Math.max(max, v[i]);
      sum += v[i];
    }
    const mean = sum / v.length;
    let sq = 0;
    for (let i = 0; i < v.length; i++) sq += (v[i] - mean) ** 2;
    const std = Math.sqrt(sq / v.length);
    card.lastChild.replaceWith(
      el(
        "table",
        { className: "weights" },
        el("tbody", {}, ...[["minimo", min], ["massimo", max], ["media", mean], ["deviazione standard", std]].map(([k, x]) => el("tr", {}, el("td", {}, k), el("td", { className: "num" }, fmt(x, 4))))),
      ),
      el("p", { className: "hint" }, `Valori ${engine.weightsSource === "trained" ? "addestrati" : `dello snapshot (step ${engine.snapshotStep})`} da data/weights.f32.bin.`),
    );
    if (statsChart) {
      statsChart.setData({ values: v, bins: 40 });
      statsChart.setHighlight({ x: 0, label: "zero" });
    }
    if (openCharts) ui.charts?.open?.();
  }

  const partOfTensor = (part, name) => {
    if (part.id === "tok_emb") return name.startsWith("token_embedding_table.");
    if (part.id === "pos_emb") return name.startsWith("position_embedding_table.");
    if (part.id === "ln_f" || part.id === "lm_head") return name.startsWith(`${part.id}.`);
    const [, l, kind] = part.id.split(".");
    const p = `blocks.${l}.`;
    if (kind === "attention") return name.startsWith(`${p}sa.`);
    if (kind === "ffwd") return name.startsWith(`${p}ffwd.`);
    return name.startsWith(`${p}ln1.`) || name.startsWith(`${p}ln2.`);
  };

  function ensureCharts() {
    if (!ui.charts) return;
    const section = topChartsSection(ui.charts, "pesi", "Vista Pesi", "Quanti parametri ha ogni parte del modello e che valori hanno.");
    partsChart = section.chart("parti", {
      type: "bar",
      title: "Parametri per componente",
      description: "Mostra dove si concentra la memoria del modello: le feed-forward pesano più di tutto il resto, le LayerNorm quasi nulla. Etichette: tok/pos = embedding, A = attention, F = feed-forward, N = LayerNorm (il numero è il blocco), lm = lm_head.",
      xLabel: "componente",
      yLabel: "parametri",
      highlightLabel: "componente selezionato",
      format: (v) => fmt(v),
    });
    partsChart.setData({ labels: parts.map((x) => x.short), values: parts.map((x) => x.count) });
    statsChart = section.chart("valori", {
      type: "histogram",
      title: "Valori del tensore selezionato",
      description: "Clicca una matrice nella scena: l'istogramma mostra come sono distribuiti i suoi numeri addestrati (quasi tutti vicini a zero, pochi grandi).",
      xLabel: "valore del peso",
      yLabel: "quanti pesi",
    });
  }

  function renderInfo() {
    const bars = el(
      "div",
      { className: "tf-bars" },
      ...[
        ["Token embedding", totals.get("tok_emb").count, KIND_COLORS.embedding],
        ["Position embedding", totals.get("pos_emb").count, KIND_COLORS.embedding],
        ...[...Array(cfg.n_layer).keys()].map((l) => [`Blocco ${l}`, countParams(spec.params, `blocks.${l}.`), KIND_COLORS.attention]),
        ["LayerNorm finale", totals.get("ln_f").count, KIND_COLORS.norm],
        ["lm_head", totals.get("lm_head").count, KIND_COLORS.output],
      ].map(([name, n, color]) =>
        el(
          "div",
          { className: "tf-bar" },
          el("span", { className: "tf-bar-label" }, name),
          el("span", { className: "tf-bar-track" }, el("span", { className: "tf-bar-fill", style: { width: `${(n / total) * 100 * 4}%`, background: color } })),
          el("span", { className: "tf-bar-num" }, `${fmt(n)} · ${pct(n, total)}`),
        ),
      ),
    );
    const block = countParams(spec.params, "blocks.0.");
    const ffwd0 = countParams(spec.params, "blocks.0.ffwd.");
    const { vocab_size: V, block_size: T, n_embd: C, n_layer: L } = cfg;
    info.replaceChildren(
      el("h3", {}, "Dove stanno i parametri"),
      el("p", {}, `I ${fmt(total)} numeri che il modello ha imparato sono tutti in queste matrici e barre colorate: una cella per numero. Il resto della scena (token, somme, pattern di attention) non contiene niente di appreso: sono dati che cambiano a ogni prompt, per questo ora è opaco.`),
      legend([
        { color: "#f59e0b", label: "bordo ambra: tensore appreso (parametro)" },
        { color: "#94a3b8", dashed: true, label: "grigio tratteggiato: maschera tril, buffer NON appreso" },
        { color: "rgba(148,163,184,0.25)", label: "opaco: dati e attivazioni (non sono pesi)" },
      ]),
      el("p", { className: "hint" }, "Passa col mouse su una matrice per leggerne nome e forma; clicca per vederne i valori."),
      el("button", { type: "button", className: "tf-start", onclick: () => ui.player?.start("weights-tour") }, "▶ Animazione: dove sono i parametri"),
      card,
      el("h4", {}, "Parametri per parte del modello"),
      bars,
      el("p", { className: "hint" }, `Le barre sono in scala ×4 rispetto al totale. Ogni blocco ha ${fmt(block)} parametri: il ${pct(ffwd0, block)} sta nella feed-forward, che ha due matrici ${C}×${4 * C}.`),
      deepBox(
        "Approfondimento: contare i parametri",
        el("p", {}, `Con V = ${V} caratteri, T = ${T} posizioni, C = ${C} numeri per vettore e L = ${L} blocchi:`),
        // Su due righe (aligned): nella sidebar stretta una riga sola uscirebbe dal riquadro.
        formula(String.raw`\begin{aligned} N ={}& \underbrace{VC}_{\text{token emb.}} + \underbrace{TC}_{\text{pos. emb.}} + L\,(12C^2 + 10C) \\ &+ \underbrace{2C}_{\text{ln\_f}} + \underbrace{VC + V}_{\text{lm\_head}} \end{aligned}`),
        formula(String.raw`N = ${V * C} + ${T * C} + ${L} \cdot ${12 * C * C + 10 * C} + ${2 * C} + ${V * C + V} = ${texInt(V * C + T * C + L * (12 * C * C + 10 * C) + 2 * C + V * C + V)}`),
        el("p", { className: "hint" }, `Il risultato coincide con il totale contato da PyTorch sui tensori veri: ${fmt(total)}.`),
        el("p", { className: "hint" }, "In un blocco: 3C² per query, key e value (le 4 head insieme), C² + C per la proiezione, 4C per le due LayerNorm (γ e β), 4C² + 4C e 4C² + C per le due matrici della feed-forward."),
        formula(String.raw`\begin{aligned} &\text{nn.Linear}(n_{in}, n_{out}):\; y = x W^{\top} + b \\ &W \in \mathbb{R}^{n_{out}\times n_{in}},\quad b \in \mathbb{R}^{n_{out}} \end{aligned}`),
        el("p", { className: "hint" }, "Query, key e value sono nn.Linear con bias=False: niente bias. La maschera tril è registrata con register_buffer: viaggia col modello ma non compare tra i parametri e l'ottimizzatore non la tocca."),
        el("p", { className: "hint" }, "Nei GPT più grandi (per esempio nanoGPT) lm_head condivide la matrice con la token embedding (weight tying); in questo modello sono due matrici separate, come nel codice della lezione."),
      ),
      sources([
        { href: "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html", label: "PyTorch: nn.Linear", note: "y = xAᵀ + b, peso (out, in)" },
        { href: "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html", label: "PyTorch: nn.LayerNorm", note: "γ e β di forma (C,)" },
        { href: "https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer", label: "PyTorch: register_buffer", note: "buffer non appresi" },
        { href: "https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py", label: "Karpathy, ng-video-lecture (gpt.py)", note: "il modello da cui parte il corso" },
        { href: "https://github.com/karpathy/nanoGPT/blob/master/model.py", label: "Karpathy, nanoGPT (model.py)", note: "esempio di weight tying" },
      ]),
    );
  }

  // ---------------------------------------------------------------- registrazione della vista
  tab.views.register({
    id: "pesi",
    label: "Pesi",
    description: "Dove stanno i parametri appresi: ogni tensore con nome, forma e numero di pesi.",
    order: 10,
    enter: () => {
      arch.root.visible = true;
      arch.rail.visible = true;
      flow.packet.visible = true;
      overlay ?? buildOverlay();
      renderInfo();
      info.hidden = false;
      setOverlayVisible(true);
      applyDim();
      ensureCharts();
      // Il codice che CREA i parametri: ogni nn.Embedding / nn.Linear / nn.LayerNorm porta i suoi tensori.
      tab.showCode("model.init", "Vista Pesi: dove nascono i parametri", ["nn.Embedding", "nn.Linear", "nn.LayerNorm"]);
      canvas.addEventListener("pointermove", onMove);
      canvas.addEventListener("pointerdown", onDown);
      canvas.addEventListener("pointerup", onUp);
      canvas.addEventListener("pointerleave", hideTooltip);
      host.controls.addEventListener("start", hideTooltip);
    },
    exit: () => {
      if (ui.player?.action?.id === "weights-tour") ui.player.stop(); // il giro dei pesi ha senso solo in questa vista
      info.hidden = true;
      card.hidden = true;
      tooltip.hidden = true;
      setOverlayVisible(false);
      restoreDim();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointerleave", hideTooltip);
      host.controls.removeEventListener("start", hideTooltip);
    },
  });

  registerWeightsAction(tab, { selectTensor, hideTooltip });
}

/**
 * Azione "Dove sono i parametri": visita i componenti in ordine di flusso, li conta e
 * fa crescere il grafico del totale cumulato fino al totale di PyTorch (spec.total_params).
 */
function registerWeightsAction(tab, { selectTensor, hideTooltip }) {
  const { ui, spec } = tab;
  if (!ui.player) return;
  const cfg = spec.config;
  const { vocab_size: V, block_size: T, n_embd: C, n_head: H, n_layer: L } = cfg;
  const hs = C / H;
  const total = spec.total_params;
  const c = (prefix) => countParams(spec.params, prefix);
  const otherLayers = [...Array(L).keys()].slice(1); // i blocchi oltre il primo, contati insieme
  const visits = [
    { focus: "tok_emb", label: "token_embedding_table", n: c("token_embedding_table."), code: "model.init", hl: ["token_embedding_table"], tensor: "token_embedding_table.weight", caption: (n) => `Token embedding: ${V} righe × ${C} colonne = ${fmt(n)} parametri. Ogni carattere ha la sua riga di ${C} numeri.` },
    { focus: "pos_emb", label: "position_embedding_table", n: c("position_embedding_table."), code: "model.init", hl: ["position_embedding_table"], tensor: "position_embedding_table.weight", caption: (n) => `Position embedding: ${T} posizioni × ${C} = ${fmt(n)} parametri. Una riga per ogni posto nel contesto.` },
    { focus: "blocks.0.heads", label: "blocco 0: query, key, value", n: c("blocks.0.sa.heads."), code: "head", hl: ["self.key", "self.query", "self.value"], tensor: "blocks.0.sa.heads.0.query.weight", caption: (n) => `Blocco 0, attention: ${H} head × 3 matrici (query, key, value) da ${hs}×${C} = ${fmt(n)} parametri, senza bias.` },
    { focus: "blocks.0.proj", label: "blocco 0: proiezione", n: c("blocks.0.sa.proj."), code: "mha", hl: ["self.proj"], tensor: "blocks.0.sa.proj.weight", caption: (n) => `Proiezione dopo le head: matrice ${C}×${C} più ${C} bias = ${fmt(n)} parametri.` },
    { focus: "blocks.0.ln1", label: "blocco 0: LayerNorm", n: c("blocks.0.ln1.") + c("blocks.0.ln2."), code: "block", hl: ["self.ln1", "self.ln2"], tensor: "blocks.0.ln1.weight", caption: (n) => `Le due LayerNorm del blocco: γ e β da ${C} numeri ciascuna, ${fmt(n)} parametri in tutto. Poche, ma indispensabili per un training stabile.` },
    { focus: "blocks.0.ffwd", label: "blocco 0: feed-forward", n: c("blocks.0.ffwd."), code: "ffwd", hl: ["nn.Linear(config.n_embd, hidden)", "nn.Linear(hidden, config.n_embd)"], tensor: "blocks.0.ffwd.net.0.weight", caption: (n) => `Feed-forward: ${C}→${4 * C} e ${4 * C}→${C} con i bias = ${fmt(n)} parametri. È la parte più pesante di ogni blocco.` },
    { focus: "blocks", label: `blocchi ${otherLayers.join(", ")}`, n: otherLayers.reduce((a, l) => a + c(`blocks.${l}.`), 0), code: "model.init", hl: ["self.blocks"], tensor: null, caption: (n) => `Gli altri ${L - 1} blocchi sono identici nella forma (ma con numeri diversi): altri ${fmt(n)} parametri.` },
    { focus: "ln_f", label: "ln_f", n: c("ln_f."), code: "model.init", hl: ["self.ln_f"], tensor: "ln_f.weight", caption: (n) => `LayerNorm finale: ${fmt(n)} parametri.` },
    { focus: "lm_head", label: "lm_head", n: c("lm_head."), code: "model.init", hl: ["self.lm_head"], tensor: "lm_head.weight", caption: (n) => `lm_head: ${V}×${C} più ${V} bias = ${fmt(n)} parametri. Da qui escono i ${V} punteggi.` },
    { focus: "blocks.0.heads.0.pattern", label: "maschere tril", n: 0, code: "head", hl: ["register_buffer", "tril"], tensor: null, caption: () => `Le ${spec.buffers.length} maschere tril (${T}×${T}) non sono parametri: sono buffer fissi, il training non le cambia. Contano 0.` },
  ];
  let chart = null;
  const series = (upto) => {
    let acc = 0;
    return visits.slice(0, upto + 1).map((v, i) => [i + 1, (acc += v.n)]);
  };
  const draw = (i) => {
    if (!ui.charts) return;
    const section = ui.charts.section("pesi", "Vista Pesi", "Quanti parametri ha ogni parte del modello e che valori hanno.");
    chart ??= section.chart("cumulato", {
      type: "line",
      title: "Parametri contati finora",
      description: `Somma progressiva mentre visitiamo i componenti: deve arrivare esattamente a ${fmt(total)}, il numero che stampa PyTorch.`,
      xLabel: "componente visitato",
      yLabel: "parametri",
      format: (v) => fmt(v),
    });
    chart.setData({ series: [{ name: "cumulato", points: series(i) }, { name: "totale del modello", points: [[1, total], [visits.length, total]] }] });
    chart.setHighlight({ x: i + 1, label: visits[i].label });
  };

  ui.player.register({
    id: "weights-tour",
    title: `Dove sono i ${fmt(total)} parametri`,
    category: "Viste",
    summary: "Visita ogni componente, conta i suoi pesi e verifica che il totale torni con quello di PyTorch.",
    onStart: async () => {
      await tab.views.activate("pesi");
      tab.ui.sidebar.select("viste");
    },
    steps: visits.map((v, i) => ({
      title: v.label,
      duration: 4,
      caption: v.caption(v.n),
      enter: () => {
        hideTooltip(); // la camera si muove: il nome sotto il puntatore non sarebbe più quello giusto
        tab.lod.controller.setMode("auto");
        tab.focus(v.focus, { duration: 1.2 });
        tab.showCode(v.code, `Pesi: ${v.label}`, v.hl);
        draw(i);
        if (v.tensor) selectTensor(v.tensor);
      },
    })),
  });
}
