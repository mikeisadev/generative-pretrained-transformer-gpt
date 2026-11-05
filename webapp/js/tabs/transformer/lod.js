/**
 * Zoom semantico (LOD) del tab Transformer: più ti allontani, più la scena si semplifica.
 *
 *   Dettaglio   (vicino) → la scena di architecture.js: ogni matrice di pesi, ogni head
 *   Blocchi     (medio)  → un rettangolo per componente con titolo, pesi e una frase
 *   Panoramica  (lontano)→ Input, Transformer (4 blocchi), Output e le frecce del flusso
 *
 * API esposta come `tab.lod` (per gli altri moduli del tab, es. la vista "Contesto"):
 *
 *   tab.lod.controller                 LODController di core/lod.js (setMode("auto"|"detail"|"blocks"|"macro"), current)
 *   tab.lod.levels                     { detail, blocks, macro }: i THREE.Group dei tre livelli
 *   tab.lod.register({ detail, blocks, macro })
 *                                      aggiunge i tuoi gruppi come figli dei gruppi di livello
 *                                      (ognuno opzionale) e restituisce una funzione che li rimuove.
 *                                      Regola di core/lod.js: materiali PROPRI, non condivisi tra livelli,
 *                                      perché l'opacità viene sfumata modificando i materiali.
 *                                      La visibilità dei TUOI gruppi la decidi tu (es. solo nella tua vista).
 *   tab.lod.setDistances({ detail, blocks })   soglie di distanza (camera → punto guardato) per la
 *                                      scena corrente; tab.lod.resetDistances() torna a quelle dell'architettura.
 *   tab.lod.setBaseOpacity(material, opacity)  cambia l'opacità "di base" di un materiale in modo
 *                                      compatibile con le sfumature del LOD (vedi sotto).
 *   tab.lod.arch                       { blocks, macro, rects, macroInfo }: i gruppi dei rettangoli dell'architettura
 *                                      (coordinate di arch.root) e il rettangolo di ogni stazione del flusso,
 *                                      per chi vuole sovrapporre qualcosa ai livelli astratti (es. vista "Pesi").
 *   tab.events: evento "lod" con detail = id del livello attivo.
 *
 * Regole di convivenza:
 *   - arch.root sta dentro il livello Dettaglio; i rettangoli dell'architettura nei livelli Blocchi e
 *     Panoramica seguono arch.root.visible: una vista che nasconde arch.root nasconde l'architettura
 *     a tutti i livelli, senza dover conoscere questo modulo.
 *   - Il binario e il pacchetto del flusso (tab.flow) restano nella scena a ogni livello: i rettangoli
 *     occupano gli stessi slot della scena di dettaglio, e il rettangolo che il pacchetto sta
 *     attraversando si illumina (come i componenti nel dettaglio).
 */
import * as THREE from "three";
import { el, fmt } from "../../core/dom.js";
import { LODController } from "../../core/lod.js";
import { buildArchitectureLevels } from "./views/archLevels.js";
import { makeOutline } from "./views/lodBlocks.js";
import { topChartsSection } from "./views/panel.js";

/** Soglie dell'architettura (pila alta ~75 unità; la vista d'insieme sta a ~110-125 unità). */
export const ARCH_DISTANCES = { detail: 42, blocks: 150 };

/**
 * Imposta l'opacità di base di un materiale tenendo conto della sfumatura LOD in corso:
 * core/lod.js calcola opacity = baseOpacity × alpha, quindi aggiorniamo entrambi.
 */
export function setBaseOpacity(material, opacity) {
  const ud = material.userData;
  const base = ud.baseOpacity ?? material.opacity;
  const alpha = base > 1e-6 ? material.opacity / base : 1;
  if (ud.baseOpacity === undefined) ud.baseTransparent = material.transparent;
  ud.baseOpacity = opacity;
  material.opacity = opacity * alpha;
  const transparent = ud.baseTransparent || opacity < 0.999 || alpha < 0.999;
  if (material.transparent !== transparent) {
    material.transparent = transparent;
    material.needsUpdate = true;
  }
}

/** Evidenzia il rettangolo attraversato dal pacchetto del flusso, in un livello. */
class FlowHighlight {
  constructor(parent, rects) {
    this.rects = rects;
    this.outline = makeOutline(0x67e8f9);
    this.outline.renderOrder = 5;
    parent.add(this.outline);
    this.current = null;
  }

  show(stationId) {
    const r = stationId ? this.rects.get(stationId) : null;
    if (r === this.current) return;
    this.current = r;
    if (!r) {
      this.outline.visible = false;
      return;
    }
    this.outline.position.set(r.x, r.y, 0.75);
    this.outline.scale.set(r.w + 0.35, r.h + 0.35, 1);
    this.outline.visible = true;
  }

  pulse(elapsed) {
    if (this.outline.visible) this.outline.material.color.setScalar(0.65 + 0.35 * Math.sin(elapsed * 5));
  }
}

export function installLod(tab) {
  const { ui, arch, flow } = tab;
  const { host } = ui;

  const levels = { detail: new THREE.Group(), blocks: new THREE.Group(), macro: new THREE.Group() };
  for (const [id, g] of Object.entries(levels)) {
    g.name = `lod-${id}`;
    host.scene.add(g);
  }

  // Dettaglio = la scena esistente (spostarla in un gruppo all'origine non cambia la sua posizione nel mondo).
  levels.detail.add(arch.root);

  // Blocchi e Panoramica dell'architettura, nelle stesse coordinate della pila.
  const archLevels = buildArchitectureLevels(tab);
  const archBlocks = new THREE.Group();
  const archMacro = new THREE.Group();
  archBlocks.position.copy(arch.root.position);
  archMacro.position.copy(arch.root.position);
  archBlocks.add(archLevels.blocks);
  archMacro.add(archLevels.macro);
  levels.blocks.add(archBlocks);
  levels.macro.add(archMacro);
  const highlights = { blocks: new FlowHighlight(archBlocks, archLevels.rects.blocks), macro: new FlowHighlight(archMacro, archLevels.rects.macro) };

  const distances = { ...ARCH_DISTANCES };
  const controller = new LODController(host, {
    levels: [
      { id: "detail", label: "Dettaglio", maxDistance: distances.detail, group: levels.detail },
      { id: "blocks", label: "Blocchi", maxDistance: distances.blocks, group: levels.blocks },
      { id: "macro", label: "Panoramica", maxDistance: Infinity, group: levels.macro },
    ],
    widgetParent: ui.stage,
    onChange: (id) => tab.events.dispatchEvent(new CustomEvent("lod", { detail: id })),
  });
  controller.widget?.setAttribute("title", "Zoom semantico: con la rotellina (Auto) o scegliendo un livello");

  const setDistances = ({ detail = distances.detail, blocks = distances.blocks } = {}) => {
    controller.levels[0].maxDistance = detail;
    controller.levels[1].maxDistance = blocks;
  };

  host.onFrame((dt, elapsed) => {
    // I livelli astratti dell'architettura seguono la visibilità della scena di dettaglio.
    const visible = arch.root.visible;
    archBlocks.visible = visible;
    archMacro.visible = visible;
    if (!visible) return;
    const stationId = flow.packet.visible ? flow.current?.id ?? null : null;
    highlights.blocks.show(stationId);
    highlights.macro.show(stationId);
    highlights.blocks.pulse(elapsed);
    highlights.macro.pulse(elapsed);
  });

  tab.lod = {
    controller,
    levels,
    register(groups) {
      const added = [];
      for (const id of ["detail", "blocks", "macro"]) {
        if (!groups[id]) continue;
        levels[id].add(groups[id]);
        added.push(groups[id]);
      }
      return () => added.forEach((g) => g.removeFromParent());
    },
    setDistances,
    resetDistances: () => setDistances(ARCH_DISTANCES),
    setBaseOpacity,
    /** Livelli astratti dell'architettura: gruppi (coordinate di arch.root), rettangoli per stazione, geometria della Panoramica. */
    arch: { blocks: archBlocks, macro: archMacro, rects: archLevels.rects, macroInfo: archLevels.macroInfo },
    get current() {
      return controller.current;
    },
  };

  registerZoomAction(tab);
  renderLodHelp(tab);
}

/** Spiegazione dello zoom semantico nel pannello "Viste", sotto l'elenco delle viste. */
function renderLodHelp(tab) {
  const row = (name, text) => el("li", {}, el("b", {}, name), ` ${text}`);
  tab.panes.views.append(
    el(
      "details",
      { className: "deep tf-lod-help" },
      el("summary", {}, "Zoom semantico: cosa vedi a ogni livello"),
      el("p", {}, "Come una mappa: da vicino le strade, da lontano solo le regioni. Usa la rotellina (selettore su Auto) oppure scegli un livello in alto a sinistra."),
      el(
        "ul",
        {},
        row("Dettaglio:", "ogni matrice di pesi, una cella per numero appreso, e i pattern di attention."),
        row("Blocchi:", "un rettangolo per componente con nome, numero di pesi e una frase su cosa fa."),
        row("Panoramica:", "Input, Transformer e Output con le frecce del flusso: il diagramma del paper."),
      ),
      el("p", { className: "hint" }, "Il pacchetto luminoso del flusso attraversa i rettangoli a ogni livello; il rettangolo che sta attraversando si illumina."),
      el("button", { type: "button", className: "tf-start", onclick: () => tab.ui.player?.start("lod-zoom") }, "▶ Animazione: zoom semantico"),
    ),
  );
}

/**
 * Azione animata "Zoom semantico": la camera si allontana passo per passo e il livello cambia.
 * Il grafico "Quanto c'è da leggere" mostra quanti elementi compaiono a ogni livello.
 */
function registerZoomAction(tab) {
  const { ui, arch, spec } = tab;
  if (!ui.player) return;
  const lod = tab.lod;
  const center = new THREE.Vector3(0, 0, 0);
  const dir = new THREE.Vector3(0.12, 0.16, 1).normalize();
  const tensors = spec.params.length;
  // Rettangoli del livello Blocchi: uno per slot, l'embedding diviso in due (token, posizione), un contenitore per blocco.
  const blockCount = arch.slots.size + 1 + spec.config.n_layer;
  const counts = [
    { label: "Dettaglio: tensori", value: tensors },
    { label: "Blocchi: rettangoli", value: blockCount },
    { label: "Panoramica: rettangoli", value: 3 },
  ];
  let chart = null;
  const showChart = (index) => {
    if (!ui.charts) return;
    const section = topChartsSection(ui.charts, "lod", "Zoom semantico", "Quanta informazione vedi a ogni livello di zoom.");
    chart ??= section.chart("lod-count", {
      type: "bar",
      title: "Quante cose leggere a ogni livello",
      description: "Da vicino ogni tensore di pesi è un oggetto separato; allontanandosi lo stesso modello si riassume in pochi rettangoli. Serve a capire perché la vista si semplifica.",
      xLabel: "livello",
      yLabel: "elementi in scena",
      highlightLabel: "livello attivo",
    });
    chart.setData({ labels: counts.map((c) => c.label), values: counts.map((c) => c.value), highlight: index });
  };
  const fly = (distance, target = center) => tab.ui.director.flyTo(target.clone().addScaledVector(dir, distance), target, 1.6);

  ui.player.register({
    id: "lod-zoom",
    title: "Zoom semantico: dal dettaglio alla panoramica",
    category: "Viste",
    summary: "La camera si allontana: le matrici diventano rettangoli con una descrizione, poi un diagramma a blocchi.",
    onStart: () => tab.views.activate("architettura"),
    onStop: () => lod.controller.setMode("auto"),
    onEnd: () => lod.controller.setMode("auto"),
    steps: [
      {
        title: "Dettaglio",
        duration: 4.5,
        caption: `Da vicino vedi ogni matrice di pesi del blocco 0: una cella colorata per ogni numero appreso (${tensors} tensori in tutto il modello).`,
        enter: () => {
          lod.controller.setMode("detail");
          tab.focus("blocks.0", { duration: 1.4 });
          tab.showCode("block", "Il blocco transformer", ["x = x + self.sa", "x = x + self.ffwd"]);
          showChart(0);
        },
      },
      {
        title: "Blocchi",
        duration: 5,
        caption: `Allontanandoti, ogni componente diventa un rettangolo: nome, numero di pesi e una frase su cosa fa. Il pacchetto del flusso continua a passare dentro i rettangoli.`,
        enter: () => {
          lod.controller.setMode("blocks");
          fly(105);
          tab.showCode("block", "Il blocco transformer", ["self.ln1", "self.sa", "self.ln2", "self.ffwd"]);
          showChart(1);
        },
      },
      {
        title: "Panoramica",
        duration: 5,
        caption: `Ancora più lontano restano tre rettangoli: Input, Transformer (${spec.config.n_layer} blocchi, ${fmt(spec.total_params)} parametri in tutto) e Output, con le frecce del flusso.`,
        enter: () => {
          lod.controller.setMode("macro");
          fly(185);
          tab.showCode("model.forward", "Il forward del modello", ["tok_emb =", "x = self.blocks(x)", "logits = self.lm_head(x)"]);
          showChart(2);
        },
      },
      {
        title: "Tocca a te",
        duration: 3.5,
        caption: "Il selettore in alto a sinistra torna su Auto: usa la rotellina per avvicinarti e allontanarti, il livello cambia da solo.",
        enter: () => {
          lod.controller.setMode("auto");
          tab.focus("overview", { duration: 1.4 });
        },
      },
    ],
  });
}
