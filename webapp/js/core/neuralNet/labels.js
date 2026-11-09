/**
 * Etichette della rete: tutte in UNA texture (atlante) e disegnate con UN InstancedMesh di rettangoli
 * rivolti alla camera.
 *
 * Perché non uno sprite per etichetta (core/visuals.js label): con 100-200 etichette sarebbero
 * 100-200 disegni separati, ~7-10 ms per frame con il rendering software. Qui sono uno solo.
 *
 * Leggibili da vicino e da lontano: ogni etichetta ha un'altezza "nel mondo", ma sullo schermo resta
 * tra un minimo e un massimo di pixel (da lontano non diventa un puntino, da vicino non copre tutto).
 * Da lontano (etichetta già al minimo) resta solo il titolo: la seconda riga sarebbe di 7-9 px, illeggibile,
 * e occuperebbe il posto di altri titoli; avvicinandosi torna. Per questo ogni etichetta con una seconda
 * riga ha due riquadri nell'atlante (completo e solo titolo) e l'update sceglie quale disegnare.
 * Quando si sovrappongono vince la più importante (gruppi e "× N", poi le sezioni, poi i nomi dei token):
 * di lato i nomi dei token stanno uno sopra l'altro e se ne vede uno, girando la camera si aprono.
 * Le etichette evitano anche i rettangoli `obstacles` (la scheda, il pad, il selettore: DOM sopra il
 * canvas), così un titolo non finisce mezzo nascosto sotto un pannello togliendo il posto a un altro.
 * Tutto questo si ricalcola solo quando la camera si muove (update), mai a riposo.
 */
import * as THREE from "three";
import { roundRect } from "../diagram.js";
import { labelMaterial } from "./shaders.js";

const FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
const PAD = 10; // margine dentro il riquadro di un'etichetta (px della texture)
const GAP = 6; // spazio tra le etichette nell'atlante: le mipmap non mescolano i vicini
const MAX_ATLAS = 4096;
const NUDGE_GAP = 4; // px tra un'etichetta e quella che l'ha fatta salire
const SUB_RATIO = 0.82; // seconda riga rispetto al titolo: a 14 px di riga il titolo è di 11 px, la seconda di 9

/**
 * @typedef {object} LabelSpec
 * @property {string} text   prima riga (titolo)
 * @property {string} [sub]  seconda riga, più piccola e grigia
 * @property {string} [chip] colore #rrggbb del quadratino a sinistra (il ruolo della sezione)
 * @property {[number, number, number]} position punto della scena a cui è agganciata
 * @property {[number, number]} [anchor] quale punto dell'etichetta sta in `position` (0…1; [0.5, 0] = centro in basso)
 * @property {number} [world] altezza di una riga nel mondo
 * @property {[number, number]} [px] altezza di una riga sullo schermo: minimo e massimo (pixel CSS)
 * @property {number} [priority] 0 = la più importante
 * @property {number} [nudge] quante volte può salire di un'altezza se tocca un'etichetta più importante
 *   (es. il titolo di un gruppo sopra il "× N"): resta legata alla sua parentesi invece di sparire
 * @property {boolean} [big] testo grande (il "× N" delle ripetizioni)
 * @property {string} [color] colore del testo (predefinito: inchiostro chiaro, mai il colore del ruolo)
 * @property {*} [ref] dato del chiamante (es. { kind: "section", id })
 */
export class LabelLayer {
  constructor() {
    /** @type {(LabelSpec & { rect: number[], visible: boolean, region: object })[]} */
    this.items = [];
    this.object = null;
    this.tmp = new THREE.Vector3();
    this.hidden = new Set(); // etichette nascoste dal chiamante (es. il nome della fetta non scelta)
    this.dims = new Map(); // indice → opacità (< 1) delle etichette attenuate dal chiamante
  }

  /** @param {LabelSpec} spec @returns {number} indice dell'etichetta */
  add(spec) {
    this.items.push({ anchor: [0.5, 0], world: 0.4, px: [14, 20], priority: 2, ...spec, rect: [0, 0, 0, 0], visible: false, region: null, short: null, compact: false });
    return this.items.length - 1;
  }

  /** Disegna l'atlante e crea il mesh istanziato (una volta, dopo tutti gli add). */
  build(fontPx = 32) {
    const measure = document.createElement("canvas").getContext("2d");
    const full = this.items.map((item) => this.measure(measure, item, fontPx));
    // riquadro "solo titolo" per le etichette con la seconda riga (da lontano)
    const short = this.items.map((item) => (item.sub ? this.measure(measure, { ...item, sub: "" }, fontPx) : null));
    const layout = shelfPack([...full, ...short.filter(Boolean)], 2048);
    if (layout.height > MAX_ATLAS && fontPx > 16) return this.build(Math.floor(fontPx * 0.75)); // troppe: si rimpicciolisce
    const canvas = document.createElement("canvas");
    canvas.width = layout.width;
    canvas.height = Math.min(MAX_ATLAS, layout.height);
    const ctx = canvas.getContext("2d");
    this.items.forEach((item, i) => {
      this.draw(ctx, item, full[i], fontPx);
      if (short[i]) this.draw(ctx, { ...item, sub: "" }, short[i], fontPx);
    });
    this.canvas = canvas;
    this.texture = new THREE.CanvasTexture(canvas);
    this.texture.colorSpace = THREE.NoColorSpace; // shader nello spazio dello schermo (vedi shaders.js)
    this.texture.minFilter = THREE.LinearMipmapLinearFilter;
    this.texture.anisotropy = 4;

    const n = this.items.length;
    const quad = new THREE.InstancedBufferGeometry();
    quad.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], 3));
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    const center = new Float32Array(3 * n);
    this.anchors = new Float32Array(2 * n);
    this.uv = new Float32Array(4 * n);
    this.size = new Float32Array(2 * n);
    this.alpha = new Float32Array(n);
    this.items.forEach((item, i) => {
      center.set(item.position, 3 * i);
      this.anchors.set(item.anchor, 2 * i);
      item.region = full[i];
      item.short = short[i];
      item.compact = false;
      this.writeUv(i, full[i]);
    });
    quad.setAttribute("aCenter", new THREE.InstancedBufferAttribute(center, 3));
    this.anchorAttr = new THREE.InstancedBufferAttribute(this.anchors, 2);
    this.anchorAttr.setUsage(THREE.DynamicDrawUsage); // cambia solo quando un'etichetta sale per non toccarne un'altra
    quad.setAttribute("aAnchor", this.anchorAttr);
    this.uvAttr = new THREE.InstancedBufferAttribute(this.uv, 4);
    this.uvAttr.setUsage(THREE.DynamicDrawUsage); // cambia solo quando un'etichetta passa a "solo titolo" o torna completa
    quad.setAttribute("aUv", this.uvAttr);
    this.sizeAttr = new THREE.InstancedBufferAttribute(this.size, 2);
    this.alphaAttr = new THREE.InstancedBufferAttribute(this.alpha, 1);
    this.sizeAttr.setUsage(THREE.DynamicDrawUsage);
    this.alphaAttr.setUsage(THREE.DynamicDrawUsage);
    quad.setAttribute("aSize", this.sizeAttr);
    quad.setAttribute("aAlpha", this.alphaAttr);
    quad.instanceCount = n;
    this.object = new THREE.Mesh(quad, labelMaterial(this.texture));
    this.object.frustumCulled = false;
    this.object.renderOrder = 20; // sopra neuroni e collegamenti
    this.object.userData.noBounds = true; // non conta per i confini né per "Ti sei perso?"
    this.object.raycast = () => {}; // il doppio clic della navigazione non deve "colpire" le etichette
    this.order = this.items.map((_, i) => i).sort((a, b) => this.items[a].priority - this.items[b].priority || a - b);
    this.fontPx = fontPx;
    return this.object;
  }

  /** Coordinate nell'atlante del riquadro `r` per l'etichetta i (la texture di una canvas parte dall'alto: v = 1 - y / altezza). */
  writeUv(i, r) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    this.uv.set([r.x / W, 1 - (r.y + r.h) / H, (r.x + r.w) / W, 1 - r.y / H], 4 * i);
  }

  measure(ctx, item, F) {
    // nel "× N" il titolo è 1,6 volte più grande: la seconda riga resta sopra i 9 px quando l'etichetta è al massimo
    const subF = Math.round(F * (item.big ? 0.62 : SUB_RATIO));
    const titleF = item.big ? Math.round(F * 1.6) : F;
    ctx.font = `700 ${titleF}px ${FONT}`;
    const tw = ctx.measureText(item.text).width;
    ctx.font = `500 ${subF}px ${FONT}`;
    const sw = item.sub ? ctx.measureText(item.sub).width : 0;
    const chip = item.chip ? Math.round(F * 0.55) + 10 : 0;
    const lineH = Math.round(titleF * 1.25);
    const h = PAD * 2 + lineH + (item.sub ? Math.round(subF * 1.3) : 0);
    // l'altezza di "una riga" dell'etichetta, in pixel della texture: serve a passare dai px dello schermo al mondo
    return { w: Math.ceil(PAD * 2 + chip + Math.max(tw, sw)), h, x: 0, y: 0, lineH: titleF, titleF, subF, chip };
  }

  draw(ctx, item, r, F) {
    ctx.save();
    ctx.translate(r.x, r.y);
    // fondo scuro semitrasparente: il testo resta leggibile anche sopra un fascio di collegamenti
    ctx.fillStyle = item.big ? "rgba(13,17,23,0.55)" : "rgba(13,17,23,0.78)";
    roundRect(ctx, 0, 0, r.w, r.h, Math.min(12, r.h / 3));
    ctx.fill();
    if (item.chip) {
      ctx.fillStyle = item.chip;
      const s = Math.round(F * 0.55);
      roundRect(ctx, PAD, PAD + (r.titleF * 1.25 - s) / 2, s, s, 3);
      ctx.fill();
    }
    ctx.textBaseline = "middle";
    ctx.fillStyle = item.color ?? "#e2e8f0";
    ctx.font = `700 ${r.titleF}px ${FONT}`;
    ctx.fillText(item.text, PAD + r.chip, PAD + (r.titleF * 1.25) / 2);
    if (item.sub) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = `500 ${r.subF}px ${FONT}`;
      ctx.fillText(item.sub, PAD + r.chip, PAD + r.titleF * 1.25 + (r.subF * 1.3) / 2);
    }
    ctx.restore();
  }

  /**
   * Attenua un'etichetta (opacità 0…1; 1 = normale), es. le sezioni non scelte: da vicino i loro titoli
   * passano sopra i neuroni della sezione scelta e farebbero rumore. Vale dal prossimo update.
   */
  setDim(index, opacity) {
    if (opacity >= 1) this.dims.delete(index);
    else this.dims.set(index, Math.max(0, opacity));
  }

  /** Nasconde (o rimostra) un'etichetta a prescindere dalla camera; vale dal prossimo update. */
  setHidden(index, hidden) {
    if (hidden) this.hidden.add(index);
    else this.hidden.delete(index);
  }

  /**
   * Dimensioni e visibilità per la camera attuale (chiamata solo quando la camera è cambiata).
   * @param {THREE.PerspectiveCamera} camera @param {number} width @param {number} height pixel CSS dello stage
   * @param {THREE.Object3D} root il gruppo della rete (le posizioni sono nel suo sistema)
   * @param {number[][] | null} [obstacles] rettangoli [x0, y0, x1, y1] dello stage da lasciare liberi
   *   (pannelli DOM sopra il canvas): un'etichetta che li tocca non si mostra
   */
  update(camera, width, height, root, obstacles = null) {
    if (!this.object) return;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const accepted = obstacles ? obstacles.filter((o) => o[2] > o[0] && o[3] > o[1]) : [];
    const v = this.tmp;
    let uvChanged = false;
    let anchorChanged = false;
    root.updateWorldMatrix(true, false);
    for (const i of this.order) {
      const item = this.items[i];
      let visible = false;
      if (!this.hidden.has(i)) {
        v.set(item.position[0], item.position[1], item.position[2]).applyMatrix4(root.matrixWorld).applyMatrix4(camera.matrixWorldInverse);
        const depth = -v.z;
        if (depth > camera.near) {
          const pxPerUnit = height / (2 * depth * tanHalf);
          // da lontano (l'etichetta sarebbe più piccola del minimo) solo il titolo
          const compact = item.short !== null && item.world * pxPerUnit < item.px[0];
          if (compact !== item.compact) {
            item.compact = compact;
            this.writeUv(i, compact ? item.short : item.region);
            uvChanged = true;
          }
          const r = compact ? item.short : item.region;
          const linePx = Math.min(item.px[1], Math.max(item.px[0], item.world * pxPerUnit));
          const hPx = (linePx * r.h) / (r.titleF * 1.25);
          const wPx = (hPx * r.w) / r.h;
          // centro sullo schermo (pixel, y in giù)
          v.applyMatrix4(camera.projectionMatrix);
          const sx = (v.x + 1) * 0.5 * width;
          const sy = (1 - v.y) * 0.5 * height;
          const x0 = sx - item.anchor[0] * wPx;
          const y1 = sy + item.anchor[1] * hPx;
          // al suo posto o, se tocca un'etichetta più importante, fino a `nudge` altezze più su
          for (let k = 0; k <= (item.nudge ?? 0) && !visible; k++) {
            const lift = k * (hPx + NUDGE_GAP);
            const rect = [x0, y1 - hPx - lift, x0 + wPx, y1 - lift];
            // tutta dentro in orizzontale (un'etichetta tagliata dal bordo non si legge), almeno in parte in verticale
            const onScreen = rect[0] >= 2 && rect[2] <= width - 2 && rect[3] > 0 && rect[1] < height;
            if (!onScreen || accepted.some((o) => overlaps(o, rect))) continue;
            visible = true;
            accepted.push(rect);
            item.rect = rect;
            this.size[2 * i] = wPx / pxPerUnit;
            this.size[2 * i + 1] = hPx / pxPerUnit;
            // lo shader mette il punto `anchor` del rettangolo nel centro: abbassarlo alza il rettangolo
            const ay = item.anchor[1] - lift / hPx;
            if (this.anchors[2 * i + 1] !== ay) {
              this.anchors[2 * i + 1] = ay;
              anchorChanged = true;
            }
          }
        }
      }
      item.visible = visible;
      this.alpha[i] = visible ? this.dims.get(i) ?? 1 : 0;
    }
    this.sizeAttr.needsUpdate = true;
    this.alphaAttr.needsUpdate = true;
    if (uvChanged) this.uvAttr.needsUpdate = true;
    if (anchorChanged) this.anchorAttr.needsUpdate = true;
  }

  /** L'etichetta visibile sotto il punto (pixel dello stage), o null: le etichette si possono cliccare. */
  hit(x, y) {
    for (const i of this.order) {
      const item = this.items[i];
      if (item.visible && item.ref && x >= item.rect[0] && x <= item.rect[2] && y >= item.rect[1] && y <= item.rect[3]) return item;
    }
    return null;
  }

  dispose() {
    this.object?.geometry.dispose();
    this.object?.material.dispose();
    this.texture?.dispose();
    this.object?.removeFromParent();
    this.object = null;
    if (this.canvas) this.canvas.width = this.canvas.height = 0; // libera subito la memoria della canvas
  }
}

/** Rettangoli che si toccano (con 3 px di respiro). */
const overlaps = (a, b) => a[0] < b[2] + 3 && b[0] < a[2] + 3 && a[1] < b[3] + 3 && b[1] < a[3] + 3;

/** Impacchetta i riquadri a scaffali (dal più alto): x, y di ognuno; restituisce larghezza e altezza totali. */
export function shelfPack(regions, maxWidth) {
  const order = regions.map((_, i) => i).sort((a, b) => regions[b].h - regions[a].h);
  let x = GAP;
  let y = GAP;
  let shelf = 0;
  let width = 0;
  for (const i of order) {
    const r = regions[i];
    if (x + r.w + GAP > maxWidth && x > GAP) {
      x = GAP;
      y += shelf + GAP;
      shelf = 0;
    }
    r.x = x;
    r.y = y;
    x += r.w + GAP;
    shelf = Math.max(shelf, r.h);
    width = Math.max(width, x);
  }
  return { width: Math.min(maxWidth, Math.max(16, width)), height: Math.max(16, y + shelf + GAP) };
}

