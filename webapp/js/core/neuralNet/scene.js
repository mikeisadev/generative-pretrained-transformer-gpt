/**
 * Scena 3D della rete neurale orizzontale con profondità (three.js), costruita da un modello puro.
 *
 *   const model = buildNetworkModel(networkSpec);            // model.js (puro)
 *   const net = new NetworkScene(model);
 *   host.scene.add(net.root);
 *   net.update(camera, width, height, pixelRatio, obstacles?, focus?); // SOLO quando la camera è cambiata: LOD ed etichette
 *   net.setSlice(3); net.setEmphasis(["attn"]); net.setDensity(0.4); net.setSliceDensity(0.2); net.setWave(x | null);
 *   const u = net.pick(clientX, clientY, camera, canvasRect); net.describe(u);
 *   net.setSectionValues("attn", float32);                   // valori veri aggiornati (es. un altro prompt)
 *   net.dispose();
 *
 * Il livello di dettaglio è per sezione: quando i neuroni di una sezione sullo schermo distano meno
 * di ~3 px, i punti spariscono e compare la lastra con la heatmap (stessi colori, un texel per
 * neurone); avvicinandosi tornano i neuroni singoli. Il LOD scrive pochi byte in una texture di stato
 * letta dagli shader: nessun buffer si riscrive (vedi shaders.js). A riposo questa classe non fa niente.
 */
import * as THREE from "three";
import { writeDisplay } from "./activations.js";
import { arcLines, axesGizmo, ghostBox, neuronPoints, outline, repaintSection, segmentLines, slabMesh, structure } from "./geometry.js";
import { LabelLayer } from "./labels.js";
import { plural } from "./honesty.js";
import { decodeUnit, headOf, takesPartInSlices } from "./layout.js";
import { ARC_KIND } from "./links.js";
import { occluderOpacity } from "./occlusion.js";
import { ScreenIndex } from "./pickIndex.js";
import { createSharedUniforms } from "./shaders.js";

/** Soglie del LOD in pixel di distanza tra neuroni vicini sullo schermo. */
export const LOD_PX = Object.freeze({ pointsFrom: 2.4, pointsFull: 4.6, slabsFrom: 3.2, slabsUntil: 6.4 });
/** Quanti collegamenti si vedono bene insieme: oltre, ognuno diventa più trasparente (l'"inchiostro" resta costante). */
const INK_SEGMENTS = 9000;
const DEFAULT_SEGMENTS = 16000; // collegamenti disegnati all'apertura in una rete grande
const MAX_SLICE_LABELS = 48; // nomi delle posizioni scritti per lato, al più
const MAX_POINT_PX = 20; // diametro massimo di un neurone sullo schermo (pixel CSS): oltre costa e copre gli archi
const DIMMED = 70; // enfasi (0…255) delle sezioni non scelte
const EMPHASIS_LINKS = 0.45; // opacità dei collegamenti dritti mentre una sezione è scelta
const DIMMED_LABEL = 0.4; // opacità dei titoli delle sezioni non scelte
/**
 * Opacità degli archi della cross-attention guardati di lato, senza fetta né sezione scelta: da lì gli
 * archi tra due sezioni si proiettano tutti sulla stessa striscia e fanno una fascia piena che copre la
 * rete. Di tre quarti, o con una fetta (una parola) o una sezione scelta, tornano pieni.
 */
const CROSS_SIDE_OPACITY = 0.3;
const SLICE_LABEL_ROOM = 2; // unità oltre la sezione occupate dai nomi delle fette (testo di ~1,6 unità)

const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export class NetworkScene {
  /** @param {ReturnType<import("./model.js").buildNetworkModel>} model */
  constructor(model, { labels = true } = {}) {
    this.model = model;
    const { net, layout, links } = model;
    this.root = new THREE.Group();
    this.root.name = "neural-network";
    const S = net.sections.length;
    this.stateData = new Uint8Array(4 * S).fill(255);
    this.state = dataTexture(this.stateData, S);
    // opacità delle lastre fantasma, una colonna per ripetizione (vedi updateGhosts)
    this.ghostData = new Uint8Array(4 * Math.max(1, layout.repeats.length)).fill(255);
    this.ghostTexture = dataTexture(this.ghostData, Math.max(1, layout.repeats.length));
    this.ghostBoxes = layout.repeats.map((r) => ghostBox(r));
    this.uniforms = createSharedUniforms(this.state, this.ghostTexture);
    this.width = layout.bounds.max[0] - layout.bounds.min[0];
    this.uniforms.uWaveWidth.value = Math.max(1.2, this.width / 40);
    // le sezioni che si accendono con la fetta: i nomi delle fette stanno accanto a loro (non agli estremi
    // della rete: nel RAG i chunk accanto all'LLM, in Immagini e video i token accanto al blocco video)
    const sliced = net.slices.count > 1 ? net.sections.filter((s) => takesPartInSlices(s, net)) : [];
    this.sliceRun = sliced.length ? { x0: layout.sections[sliced[0].index].box.min[0], x1: layout.sections[sliced.at(-1).index].box.max[0] } : null;

    this.slabs = slabMesh(model, this.uniforms);
    this.neurons = neuronPoints(model, this.uniforms);
    this.links = segmentLines(model, this.uniforms, links.segments);
    this.arcs = arcLines(model, this.uniforms, { kinds: [ARC_KIND.attention, ARC_KIND.residual], keepEmpty: true });
    this.crossArcs = arcLines(model, this.uniforms, { kinds: [ARC_KIND.cross], name: "nn-cross-arcs" });
    this.sideness = 0; // 1 = camera di lato (vedi update): gli archi della cross-attention si attenuano
    this.sliceLinks = links.sliceSegments.count ? this.buildSliceLinks() : null;
    const struct = structure(model, this.uniforms);
    this.axes = axesGizmo(model);
    this.selection = outline(12, "#e0f2fe", 0.9);
    this.sliceMarker = outline(4 * net.sections.length, "#67e8f9", 0.7); // un taglio per sezione
    this.root.add(this.slabs.mesh, this.neurons, this.links, this.arcs, this.axes.group, this.selection, this.sliceMarker);
    if (this.crossArcs) this.root.add(this.crossArcs);
    if (this.sliceLinks) this.root.add(this.sliceLinks);
    if (struct.mesh) this.root.add(struct.mesh);
    if (struct.lines) this.root.add(struct.lines);
    this.labels = labels ? this.buildLabels() : null;
    if (this.labels) this.root.add(this.labels.object);

    this.pickIndex = new ScreenIndex();
    this.pickDirty = true;
    this.pickRadius = new Float32Array(S); // raggio di presa per sezione (dipende dal LOD)
    // distanza tra neuroni vicini di ogni sezione (y, z se ha profondità, x se ha più lastre): la misura del LOD
    this.spacing = layout.sections.map((b) => Math.min(b.dy, net.sections[b.index].depth.count > 1 ? b.dz : Infinity, b.plates > 1 ? b.xs[1] - b.xs[0] : Infinity));
    this.tmp = { m: new THREE.Matrix4(), inv: new THREE.Matrix4(), v: new THREE.Vector3(), cam: new THREE.Vector3(), f: new THREE.Vector3() };
    this.slice = null;
    this.emphasis = null;
    this.drawn = links.segments.count;
    this.setDensity(this.defaultDensity());
    this.updateCrossOpacity();
  }

  // ---------------------------------------------------------------- costruzione
  /**
   * Pesi veri della fetta scelta (link "explicit" con slice: "selected"): posizioni riscritte a ogni cambio
   * di fetta. Il buffer ha posto per tutti; `sliceShown` dice quale peso dell'elenco sta in ogni posto
   * (setSliceDensity ne mostra solo una parte, i più forti di ogni collegamento, nei primi posti).
   */
  buildSliceLinks() {
    const { links } = this.model;
    const list = links.sliceSegments;
    this.sliceSlot = this.model.net.slices.count - 1; // senza fetta scelta: quella davanti (l'ultimo token)
    this.sliceShown = Uint32Array.from({ length: list.count }, (_, i) => i);
    this.sliceDrawn = list.count;
    this.sliceSource = null; // colori e sezioni per indice dell'elenco: copiati alla prima setSliceDensity
    return segmentLines(this.model, this.uniforms, list, {
      place: (i, pos, o) => this.placeSliceSegment(i, pos, o),
      owner: () => -1, // sempre in evidenza: sono già sulla fetta scelta
    });
  }

  /**
   * Quanti pesi della fetta disegnare: per ogni collegamento i più forti (|peso|), senza ricostruire
   * niente (il Transformer prima ricostruiva tutta la rete per cambiare la quota: 234-282 ms al 25%).
   *   share: 0…1 = la stessa quota di ogni collegamento (⌈quota × n⌉, almeno uno se > 0), oppure una
   *          funzione (link, n) → quanti (es. le nn.Linear in quota, i γ delle LayerNorm tutti)
   * L'ordine dei posti segue quello dell'elenco (priorità), quindi con share = 1 il disegno è quello di
   * sempre. Restituisce quanti pesi si vedono.
   */
  setSliceDensity(share) {
    if (!this.sliceLinks) return 0;
    const { net, links } = this.model;
    const list = links.sliceSegments;
    if (!this.sliceSource) this.prepareSliceDensity();
    const want = new Int32Array(net.links.length);
    for (const [link, size] of this.sliceSizes) {
      let k;
      if (typeof share === "function") k = Math.round(Number(share(net.links[link], size)) || 0);
      else k = share >= 1 ? size : share > 0 ? Math.max(1, Math.ceil(share * size)) : 0;
      want[link] = Math.max(0, Math.min(size, k));
    }
    let m = 0;
    for (let i = 0; i < list.count; i++) if (this.sliceRank[i] < want[list.link[i]]) this.sliceShown[m++] = i;
    const g = this.sliceLinks.geometry;
    const { color, secA, secB } = this.sliceSource;
    const pos = g.attributes.position.array;
    const aColor = g.attributes.aColor.array;
    const aSecA = g.attributes.aSection.array;
    const aSecB = g.attributes.aSection2.array;
    for (let slot = 0; slot < m; slot++) {
      const i = this.sliceShown[slot];
      this.placeSliceSegment(i, pos, 6 * slot);
      aColor.set(color.subarray(8 * i, 8 * i + 8), 8 * slot);
      aSecA[2 * slot] = aSecA[2 * slot + 1] = secA[i];
      aSecB[2 * slot] = aSecB[2 * slot + 1] = secB[i];
    }
    for (const name of ["position", "aColor", "aSection", "aSection2"]) g.attributes[name].needsUpdate = true;
    g.setDrawRange(0, 2 * m);
    this.sliceDrawn = m;
    return m;
  }

  /** Una volta: colori e sezioni di ogni peso, e il suo rango nel collegamento (|peso| decrescente). */
  prepareSliceDensity() {
    const list = this.model.links.sliceSegments;
    const g = this.sliceLinks.geometry;
    const color = g.attributes.aColor.array.slice(0, 8 * list.count);
    const secA = new Float32Array(list.count);
    const secB = new Float32Array(list.count);
    for (let i = 0; i < list.count; i++) {
      const link = this.model.net.links[list.link[i]];
      secA[i] = link.to.index;
      secB[i] = link.from.index;
    }
    // il buffer ha i pesi nell'ordine dell'elenco (nessuna setSliceDensity prima di questa): i colori sono quelli
    this.sliceSource = { color, secA, secB };
    const byLink = new Map();
    for (let i = 0; i < list.count; i++) {
      if (!byLink.has(list.link[i])) byLink.set(list.link[i], []);
      byLink.get(list.link[i]).push(i);
    }
    this.sliceRank = new Uint32Array(list.count);
    this.sliceSizes = new Map();
    for (const [link, items] of byLink) {
      items.sort((a, b) => Math.abs(list.weight[b]) - Math.abs(list.weight[a]) || a - b);
      items.forEach((i, rank) => (this.sliceRank[i] = rank));
      this.sliceSizes.set(link, items.length);
    }
  }

  placeSliceSegment(i, pos, o) {
    const { net, layout, links } = this.model;
    const list = links.sliceSegments;
    const link = net.links[list.link[i]];
    const unit = (s, k) => s.offset + k * s.depth.count + this.depthOfSlice(s, this.sliceSlot);
    const a = unit(link.from, list.from[i]);
    const b = unit(link.to, list.to[i]);
    const P = layout.positions;
    for (let k = 0; k < 3; k++) {
      pos[o + k] = P[3 * a + k];
      pos[o + 3 + k] = P[3 * b + k];
    }
  }

  /**
   * Indice di profondità di una sezione per la fetta t della rete (in proporzione se le profondità
   * differiscono). In una griglia la fetta è una colonna: si usa la sua riga 0.
   */
  depthOfSlice(s, t) {
    if (s.depth.count <= 1) return 0;
    const cols = s.depth.cols;
    return Math.min(cols - 1, Math.floor(((t + 0.5) * cols) / Math.max(1, this.model.net.slices.count)));
  }

  buildLabels() {
    const { net, layout } = this.model;
    const labels = new LabelLayer();
    this.sectionLabels = [];
    for (const s of net.sections) {
      const box = layout.sections[s.index];
      const front = s.depth.count > 1 ? box.z1 + box.dz / 2 : 0;
      this.sectionLabels[s.index] = labels.add({
        text: s.label,
        sub: s.sub || sizeText(s),
        chip: s.color,
        position: [(box.x0 + box.x1) / 2, box.yTop + box.dy / 2 + 0.3, front],
        world: 0.36,
        px: [14, 20],
        priority: 2,
        nudge: 1,
        ref: { kind: "section", id: s.id },
      });
    }
    for (const r of layout.repeats) {
      const rep = net.repeats[r.index];
      const sub = rep.label.replace(/^×\s*[\d.]+\s*/, "");
      labels.add({ text: `× ${rep.count.toLocaleString("it-IT")}`, sub, big: true, position: [(r.x0 + r.x1) / 2, r.yTop + 0.45, r.z1 + 0.2], world: 0.62, px: [19, 30], priority: 0, ref: { kind: "repeat", id: r.id } });
    }
    for (const g of layout.groups) labels.add({ text: g.label, position: [(g.x0 + g.x1) / 2, g.y + 0.12, 0], world: 0.4, px: [15, 21], priority: 1, nudge: 2, ref: { kind: "group", index: g.index } });
    // nomi delle posizioni in profondità, ai due lati delle sezioni che si accendono con la fetta: dal
    // davanti (vince quando si sovrappongono, di lato); con centinaia di posizioni se ne scrive una ogni
    // `step` (la scheda le elenca tutte)
    const names = this.sliceRun ? net.slices.names : [];
    const step = Math.max(1, Math.ceil(names.length / MAX_SLICE_LABELS));
    this.sliceLabels = [];
    for (let t = names.length - 1; t >= 0; t--) {
      if ((names.length - 1 - t) % step !== 0) {
        this.sliceLabels[t] = [];
        continue;
      }
      const z = this.sliceZ(t);
      const left = labels.add({ text: names[t], position: [this.sliceRun.x0 - 0.7, 0, z], anchor: [1, 0.5], world: 0.27, px: [12.5, 16], priority: 3, ref: { kind: "slice", index: t }, color: "#a5f3fc" });
      const right = labels.add({ text: names[t], position: [this.sliceRun.x1 + 0.7, 0, z], anchor: [0, 0.5], world: 0.27, px: [12.5, 16], priority: 3, ref: { kind: "slice", index: t }, color: "#a5f3fc" });
      this.sliceLabels[t] = [left, right];
    }
    // assi: etichette corte (il testo intero è nella legenda della scheda)
    const tips = this.axes.tips;
    labels.add({ text: "x · flusso", position: tips.x.toArray(), anchor: [0, 0.5], world: 0.26, px: [12.5, 16], priority: 1, color: "#e2e8f0" });
    labels.add({ text: `y · ${shortAxis(net.axes.y)}`, position: tips.y.toArray(), anchor: [0, 0], world: 0.26, px: [12.5, 16], priority: 1, color: "#cbd5e1" });
    if (net.slices.count > 1) labels.add({ text: `z · ${shortAxis(net.axes.z)}`, position: tips.z.toArray(), anchor: [0, 1], world: 0.26, px: [12.5, 16], priority: 1, color: "#a5f3fc" });
    labels.build();
    return labels;
  }

  /** z della fetta t della rete (stesso passo di tutte le sezioni, centrata su 0). */
  sliceZ(t) {
    const n = this.model.net.slices.count;
    return (t - (n - 1) / 2) * this.model.layout.dz;
  }

  // ---------------------------------------------------------------- stato visibile
  /**
   * Densità predefinita: tutti i collegamenti se sono pochi, altrimenti ~16.000 (i più forti). Con il
   * rendering software le linee sono lo strato che costa di più (misurato: ~1 ms ogni 1.000 linee
   * lunghe); di più non si leggerebbero comunque. Il cursore della scheda arriva a tutti.
   */
  defaultDensity() {
    const n = this.model.links.segments.count;
    return n <= DEFAULT_SEGMENTS ? 1 : Math.max(0.02, DEFAULT_SEGMENTS / n);
  }

  /** Mostra la frazione f (0…1) dei collegamenti, i più importanti prima. Restituisce quanti. */
  setDensity(f) {
    const n = this.model.links.segments.count;
    this.density = Math.min(1, Math.max(0, f));
    this.drawn = n ? Math.max(1, Math.round(n * this.density)) : 0;
    this.links.geometry.setDrawRange(0, this.drawn * 2);
    this.linkOpacity = Math.min(1, Math.max(0.14, Math.sqrt(INK_SEGMENTS / Math.max(1, this.drawn))));
    this.links.material.uniforms.uOpacity.value = this.linkOpacity * (this.emphasis ? EMPHASIS_LINKS : 1);
    return this.drawn;
  }

  /** Fronte dell'onda di attivazione (x nel sistema della rete), o null = a riposo. */
  setWave(front) {
    this.uniforms.uFront.value = front ?? 1e6;
  }

  setGlow(on) {
    this.uniforms.uGlow.value = on ? 1 : 0;
  }

  /** x minima e massima della rete (per far scorrere l'onda dal primo all'ultimo neurone). */
  get xRange() {
    const { min, max } = this.model.layout.bounds;
    return [min[0], max[0]];
  }

  /**
   * Evidenzia una fetta di profondità (indice della rete: token, colonna della griglia…) o null.
   * I neuroni fuori dalla fetta si scuriscono e rimpiccioliscono, i collegamenti fuori quasi spariscono.
   */
  setSlice(t) {
    const n = this.model.net.slices.count;
    this.slice = t === null || t === undefined || n <= 1 ? null : Math.min(n - 1, Math.max(0, t));
    if (this.slice === null) this.uniforms.uSlice.value.set(1, 0);
    else this.uniforms.uSlice.value.set(this.slice / n, (this.slice + 1) / n);
    // pesi veri: sulla fetta scelta (o davanti); solo quelli disegnati (setSliceDensity)
    if (this.sliceLinks) {
      this.sliceSlot = this.slice ?? n - 1;
      const pos = this.sliceLinks.geometry.attributes.position;
      for (let slot = 0; slot < this.sliceDrawn; slot++) this.placeSliceSegment(this.sliceShown[slot], pos.array, 6 * slot);
      pos.needsUpdate = true;
    }
    this.updateCrossOpacity();
    // la fetta in ogni sezione: un taglio (riquadro x-y) alla z delle sue posizioni, anche quando una
    // sezione ha più posizioni (il tempo prima del passo 2) o meno (la griglia in fondo alla U)
    this.sliceMarker.visible = this.slice !== null;
    if (this.slice !== null) {
      const { net, layout } = this.model;
      let k = 0;
      for (const s of net.sections) {
        if (!takesPartInSlices(s, net)) continue;
        const box = layout.sections[s.index];
        const cols = s.depth.cols;
        // colonne con il centro dentro la fetta (una sola, o due se la sezione ne ha il doppio)
        const c0 = Math.max(0, Math.ceil((this.slice / n) * cols - 0.5));
        const c1 = Math.min(cols - 1, Math.max(c0, Math.ceil(((this.slice + 1) / n) * cols - 0.5) - 1));
        const z = box.z0 + ((c0 + c1) / 2) * box.dz;
        const pad = 0.12 + box.radius;
        writeRect(this.sliceMarker, [box.x0 - pad, box.yBottom - pad, z], [box.x1 + pad, box.yTop + pad], k++);
      }
      this.sliceMarker.geometry.setDrawRange(0, k * 8);
    }
    if (this.labels) {
      // con una fetta scelta si leggono solo i nomi di quella fetta
      this.sliceLabels.forEach((pair, i) => pair.forEach((k) => this.labels.setHidden(k, this.slice !== null && i !== this.slice)));
    }
    this.pickDirty = true;
  }

  /** Mette in risalto alcune sezioni (id o indici), le altre si scuriscono; null = tutte normali. */
  setEmphasis(ids) {
    const { net, layout } = this.model;
    const chosen = ids === null || ids === undefined ? null : new Set((Array.isArray(ids) ? ids : [ids]).map((id) => (typeof id === "number" ? id : net.byId.get(id)?.index)).filter((i) => i !== undefined));
    this.emphasis = chosen && chosen.size ? chosen : null;
    for (let s = 0; s < net.sections.length; s++) this.stateData[4 * s + 2] = !this.emphasis || this.emphasis.has(s) ? 255 : DIMMED;
    this.state.needsUpdate = true;
    // con una sezione scelta i collegamenti dritti fanno un passo indietro: gli archi in profondità
    // (attention, residui) sono il meccanismo da guardare
    this.links.material.uniforms.uOpacity.value = this.linkOpacity * (this.emphasis ? EMPHASIS_LINKS : 1);
    this.updateCrossOpacity();
    this.selection.visible = Boolean(this.emphasis);
    // i titoli delle altre sezioni si attenuano (da vicino passerebbero sopra i neuroni della scelta)
    if (this.labels) for (let s = 0; s < net.sections.length; s++) this.labels.setDim(this.sectionLabels[s], !this.emphasis || this.emphasis.has(s) ? 1 : DIMMED_LABEL);
    this.needsUpdate = true; // le etichette si ricalcolano al prossimo frame anche a camera ferma
    if (this.emphasis) {
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (const s of this.emphasis) {
        const b = layout.sections[s].box;
        for (let k = 0; k < 3; k++) {
          min[k] = Math.min(min[k], b.min[k] - 0.18);
          max[k] = Math.max(max[k], b.max[k] + 0.18);
        }
      }
      writeBox(this.selection, min, max);
    }
  }

  /**
   * Da chiamare quando la camera (o la dimensione dello stage) è cambiata: LOD per sezione, scala dei
   * punti, etichette. Costa ~0,1 ms per 60 sezioni; non va chiamata a ogni frame a riposo.
   * `obstacles`: rettangoli [x0, y0, x1, y1] dello stage coperti da pannelli DOM (scheda, pad…), che le
   * etichette lasciano liberi (facoltativo).
   * `focus`: il punto guardato (THREE.Vector3 del mondo, di solito controls.target; facoltativo): le
   * lastre fantasma dei "× N" che stanno tra la camera e lui si attenuano.
   */
  update(camera, width, height, pixelRatio = 1, obstacles = null, focus = null) {
    const { net, layout } = this.model;
    const { inv, cam } = this.tmp;
    this.root.updateWorldMatrix(true, false);
    const scale = this.root.matrixWorld.getMaxScaleOnAxis();
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    this.uniforms.uPointScale.value = (height * pixelRatio) / 2 / tanHalf;
    this.uniforms.uMaxPoint.value = MAX_POINT_PX * pixelRatio;
    inv.copy(this.root.matrixWorld).invert();
    cam.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(inv);
    // di lato (camera lungo z) gli archi della cross-attention fanno una fascia: si attenuano
    const forward = this.tmp.v.setFromMatrixColumn(camera.matrixWorld, 2).transformDirection(inv);
    const sideness = smoothstep(0.9, 0.98, Math.abs(forward.z));
    if (Math.abs(sideness - this.sideness) > 0.01) {
      this.sideness = sideness;
      this.updateCrossOpacity();
    }
    if (focus && this.ghostBoxes.length) this.updateGhosts(cam, this.tmp.f.copy(focus).applyMatrix4(inv));
    let changed = false;
    for (let s = 0; s < net.sections.length; s++) {
      const b = layout.sections[s].box;
      // distanza dalla scatola della sezione (non dal centro: una sezione grande vista da vicino ha i suoi neuroni)
      const dx = Math.max(b.min[0] - cam.x, 0, cam.x - b.max[0]);
      const dy = Math.max(b.min[1] - cam.y, 0, cam.y - b.max[1]);
      const dz = Math.max(b.min[2] - cam.z, 0, cam.z - b.max[2]);
      const distance = Math.max(0.3, Math.hypot(dx, dy, dz)) * scale;
      const px = (this.spacing[s] * scale * height) / (2 * distance * tanHalf);
      const points = smoothstep(LOD_PX.pointsFrom, LOD_PX.pointsFull, px);
      const slab = points < 0.02 ? 1 : 1 - smoothstep(LOD_PX.slabsFrom, LOD_PX.slabsUntil, px);
      const R = Math.round(255 * points);
      const G = Math.round(255 * slab);
      const A = Math.round(255 * (0.45 + 0.55 * points));
      const o = 4 * s;
      if (this.stateData[o] !== R || this.stateData[o + 1] !== G || this.stateData[o + 3] !== A) {
        this.stateData[o] = R;
        this.stateData[o + 1] = G;
        this.stateData[o + 3] = A;
        changed = true;
      }
      // presa: il neurone con i punti, la cella della heatmap con le lastre
      this.pickRadius[s] = points > 0.5 ? 1.6 * layout.sections[s].radius : 0.55 * this.spacing[s];
    }
    if (changed) this.state.needsUpdate = true;
    this.labels?.update(camera, width, height, this.root, obstacles);
    this.pickDirty = true;
  }

  /** Opacità degli archi della cross-attention: tenui di lato, pieni con una fetta o una sezione scelta. */
  updateCrossOpacity() {
    if (!this.crossArcs) return;
    const full = this.slice !== null || this.emphasis !== null;
    this.crossArcs.material.uniforms.uOpacity.value = full ? 1 : 1 - (1 - CROSS_SIDE_OPACITY) * this.sideness;
  }

  /**
   * Le lastre fantasma che stanno tra la camera e il punto guardato si attenuano (occluderOpacity):
   * guardando di tre quarti da vicino la sezione dopo un "× N" coprivano metà dell'inquadratura. Quelle
   * di un blocco scelto restano piene (le si sta guardando). Coordinate della rete; solo a camera mossa.
   */
  updateGhosts(cam, focus) {
    const { net } = this.model;
    const c = [cam.x, cam.y, cam.z];
    const f = [focus.x, focus.y, focus.z];
    let changed = false;
    this.ghostBoxes.forEach((box, r) => {
      const chosen = this.emphasis && net.repeats[r].of.every((s) => this.emphasis.has(s.index));
      const value = Math.round(255 * (chosen ? 1 : occluderOpacity(c, f, box)));
      if (this.ghostData[4 * r] !== value) {
        this.ghostData[4 * r] = value;
        changed = true;
      }
    });
    if (changed) this.ghostTexture.needsUpdate = true;
  }

  // ---------------------------------------------------------------- neurone sotto il puntatore
  /**
   * L'unità sotto il punto (coordinate client) o -1. L'indice sullo schermo si ricostruisce solo se la
   * camera si è mossa dall'ultima volta (pochi ms per 60.000 neuroni), poi ogni ricerca costa µs.
   */
  pick(clientX, clientY, camera, rect) {
    const { net, layout } = this.model;
    if (this.pickDirty || this.pickIndex.width !== rect.width || this.pickIndex.height !== rect.height) {
      const { m } = this.tmp;
      camera.updateMatrixWorld();
      m.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(this.root.matrixWorld);
      const scale = this.root.matrixWorld.getMaxScaleOnAxis();
      const sectionOf = layout.sectionOf;
      this.pickIndex.build(layout.positions, net.units, m.elements, rect.width, rect.height, {
        pickRadius: (u) => this.pickRadius[sectionOf[u]] * scale,
        pxPerUnit: (rect.height / 2) * camera.projectionMatrix.elements[5],
        minPx: 4,
        maxPx: 30,
      });
      this.pickDirty = false;
    }
    const prefer = this.slice === null ? null : (u) => this.inSlice(u);
    return this.pickIndex.query(clientX - rect.left, clientY - rect.top, { prefer, maxPx: 30 });
  }

  inSlice(u) {
    const d = this.model.layout.depthU[u];
    const n = this.model.net.slices.count;
    return d < 0 || (d >= this.slice / n && d <= (this.slice + 1) / n);
  }

  /** Tutto quello che serve a raccontare un'unità (etichetta al passaggio del mouse). */
  describe(u) {
    const { net, layout, values } = this.model;
    const s = net.sections[layout.sectionOf[u]];
    const local = u - s.offset;
    const info = decodeUnit(s, local);
    const out = { unit: u, section: s, ...info, value: values.raw[u], display: values.display[u], mode: values.mode[u], simulated: values.simulated.has(s.id) };
    if (s.depth.kind === "line") out.depthName = s.depth.names?.[info.depth] ?? `${s.depth.label} ${info.depth + 1}`;
    if (s.heads > 1 && s.depth.kind !== "grid") out.head = headOf(s, info.neuron);
    if (s.role === "experts") {
      const route = values.routing.get(s.id);
      const shared = info.slab < s.shared;
      const e = info.slab - s.shared;
      let gate = shared ? 1 : 0;
      if (!shared) for (let k = 0; k < route.k; k++) if (route.experts[info.depth * route.k + k] === e) gate = route.gates[info.depth * route.k + k];
      out.expert = { shared, index: shared ? info.slab : e, active: route.mask[info.slab * s.depth.count + info.depth] === 1, gate };
    }
    return out;
  }

  // ---------------------------------------------------------------- dati nuovi
  /** Valori veri nuovi per una sezione (stessa forma): ricolora neuroni e heatmap, senza ricostruire. */
  setSectionValues(id, data) {
    const { net, values } = this.model;
    const s = net.byId.get(id);
    if (!s || s.kind !== "section" || data.length !== s.units) throw new Error(`setSectionValues: sezione "${id}" sconosciuta o ${data?.length} valori invece di ${s?.units}`);
    values.raw.set(data, s.offset);
    values.simulated.delete(s.id);
    writeDisplay(s, data, values.display, values.mode, values.routing.get(s.id));
    const g = this.neurons.geometry;
    const value = g.attributes.aValue;
    const mode = g.attributes.aMode;
    for (let u = s.offset; u < s.offset + s.units; u++) mode.array[u] = values.mode[u];
    for (const attr of [value, mode]) {
      attr.clearUpdateRanges();
      attr.addUpdateRange(s.offset, s.units);
      attr.needsUpdate = true;
    }
    repaintSection(this.slabs, values, s);
  }

  // ---------------------------------------------------------------- inquadrature
  /** Riquadro (THREE.Box3, coordinate del mondo) di sezioni o ripetizioni per id; tutta la rete senza argomenti. */
  box(ids = null) {
    const { layout, net } = this.model;
    const box = new THREE.Box3();
    const add = (b) => box.expandByPoint(this.tmp.v.fromArray(b.min)).expandByPoint(this.tmp.v.fromArray(b.max));
    if (!ids) {
      add(layout.bounds);
      box.expandByPoint(this.axes.origin).expandByPoint(this.axes.tips.x).expandByPoint(this.axes.tips.y);
      // nomi delle fette ai lati delle sezioni che si accendono con la fetta: ~1,6 unità di testo oltre
      if (this.sliceRun) box.expandByPoint(this.tmp.v.set(this.sliceRun.x0 - SLICE_LABEL_ROOM, 0, 0)).expandByPoint(this.tmp.v.set(this.sliceRun.x1 + SLICE_LABEL_ROOM, 0, 0));
    }
    else {
      for (const id of Array.isArray(ids) ? ids : [ids]) {
        const item = net.byId.get(id);
        if (!item) continue;
        if (item.kind === "repeat") {
          const r = layout.repeats.find((x) => x.id === id);
          add({ min: [r.blockX0, r.yBottom, r.z0], max: r.box.max });
        } else add(layout.sections[item.index].box);
      }
    }
    if (box.isEmpty()) return box;
    this.root.updateWorldMatrix(true, false);
    return box.applyMatrix4(this.root.matrixWorld);
  }

  /**
   * Punti da far entrare nell'inquadratura (coordinate del mondo): gli spigoli delle sezioni (o di quelle
   * scelte), delle lastre fantasma, i culmini di parentesi e residui, gli assi. Più stretti di un unico
   * riquadro: visto di tre quarti, il riquadro di tutta la rete ha grandi angoli vuoti.
   */
  framePoints(ids = null) {
    const { layout, net } = this.model;
    const boxes = [];
    if (!ids) {
      boxes.push(...layout.sections.map((b) => b.box), ...layout.repeats.map((r) => r.box));
      // il culmine di parentesi e residui; ai lati, il posto dei nomi delle fette (se ci sono)
      const top = layout.bounds.max[1];
      const x0 = Math.min(layout.bounds.min[0], (this.sliceRun?.x0 ?? Infinity) - SLICE_LABEL_ROOM);
      const x1 = Math.max(layout.bounds.max[0], (this.sliceRun?.x1 ?? -Infinity) + SLICE_LABEL_ROOM);
      boxes.push({ min: [x0, top, 0], max: [x1, top, 0] });
      boxes.push({ min: this.axes.origin.toArray(), max: this.axes.tips.y.toArray() });
    } else {
      for (const id of Array.isArray(ids) ? ids : [ids]) {
        const item = net.byId.get(id);
        if (!item) continue;
        if (item.kind === "repeat") {
          const r = layout.repeats.find((x) => x.id === id);
          boxes.push(...item.of.map((s) => layout.sections[s.index].box), r.box);
        } else boxes.push(layout.sections[item.index].box);
      }
    }
    this.root.updateWorldMatrix(true, false);
    const points = [];
    for (const b of boxes) {
      for (let i = 0; i < 8; i++) {
        points.push(new THREE.Vector3(i & 1 ? b.max[0] : b.min[0], i & 2 ? b.max[1] : b.min[1], i & 4 ? b.max[2] : b.min[2]).applyMatrix4(this.root.matrixWorld));
      }
    }
    return points;
  }

  dispose() {
    this.labels?.dispose();
    this.root.traverse((o) => {
      o.geometry?.dispose();
      if (o.material) {
        for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
          m.uniforms?.uAtlas?.value?.dispose?.();
          m.dispose();
        }
      }
    });
    this.slabs.texture.dispose();
    this.state.dispose();
    this.ghostTexture.dispose();
    this.root.removeFromParent();
  }
}

/** Texture di stato RGBA8 di una riga (una colonna per sezione o per ripetizione), letta con texelFetch. */
function dataTexture(data, width) {
  const texture = new THREE.DataTexture(data, width, 1, THREE.RGBAFormat);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/** Testo dell'asse fino alla prima parentesi o al primo ";" (al più 26 caratteri): "numeri del vettore (16 dei 7.168)" → "numeri del vettore". */
export function shortAxis(text) {
  const cut = text.split(/\s\(|;|:/)[0].trim();
  return cut.length > 26 ? `${cut.slice(0, 25).trimEnd()}…` : cut;
}

/** "16 × 8 token", "4 canali · 8 × 8", "17 esperti × 6 · 8 token": la forma disegnata, se la specifica non dice altro. */
export function sizeText(s) {
  const depth = s.depth.kind === "grid" ? `${s.depth.rows} × ${s.depth.cols}` : s.depth.count > 1 ? `${s.depth.count} ${s.depth.label}` : "";
  if (s.role === "experts") return `${s.experts}${s.shared ? ` + ${s.shared}` : ""} ${s.experts + s.shared === 1 ? "esperto" : "esperti"} × ${s.neurons}${depth ? ` · ${depth}` : ""}`;
  if (s.depth.kind === "grid") return `${plural(s.neurons, "canale", "canali")} · ${depth}`;
  const n = s.slabs > 1 ? `${s.slabs} × ${s.neurons}` : `${s.neurons}`;
  return depth ? `${n} × ${depth}` : s.slabs > 1 ? `${n} neuroni` : plural(s.neurons, "neurone", "neuroni");
}

/** Scrive i 12 spigoli di una scatola in un contorno di 12 segmenti. */
function writeBox(lines, min, max) {
  const c = [[min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [min[0], max[1], min[2]], [min[0], min[1], max[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]]];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  const pos = lines.geometry.attributes.position;
  edges.forEach(([a, b], i) => pos.array.set([...c[a], ...c[b]], 6 * i));
  pos.needsUpdate = true;
}

/** Scrive il rettangolo numero `slot` (4 segmenti) nel piano z = costante. */
function writeRect(lines, [x0, y0, z], [x1, y1], slot = 0) {
  const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  const pos = lines.geometry.attributes.position;
  for (let i = 0; i < 4; i++) {
    const [a, b] = [pts[i], pts[(i + 1) % 4]];
    pos.array.set([a[0], a[1], z, b[0], b[1], z], 24 * slot + 6 * i);
  }
  pos.needsUpdate = true;
}
