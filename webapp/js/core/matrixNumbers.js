/**
 * Matrici di pesi in 3D con i NUMERI VERI scritti nelle celle quando sono abbastanza grandi
 * sullo schermo.
 *
 *   const m = new NumericMatrix({ rows: 256, cols: 64, data: tensor.data });   // celle colorate
 *   scene.add(m.object3d);
 *   host.onFrame(() => m.update(host.camera, host.container.clientHeight));     // economico a riposo
 *   m.setData(snapshot.tensors.get(name).data);                                 // un altro step del training
 *   const cell = m.cellAt(host.pick(event, [m.mesh])[0]);                       // {row, col, value}
 *
 *   // sulle heatmap già esistenti (core/visuals.js, riga 0 in alto), con un budget comune:
 *   const numbers = new NumbersGroup({ maxOverlays: 4 });
 *   const handle = attachNumbers(mesh, { rows, cols, getValue: (r, c) => w[r * cols + c], group: numbers });
 *   host.onFrame(() => numbers.update(host.camera, host.container.clientHeight));
 *
 * Come funziona (perché costa poco):
 * - le celle sono una DataTexture (1 texel = 1 peso, colormap divergente di core/visuals.js:
 *   stessi colori delle heatmap del Transformer) su un piano di `matrixPlane`;
 * - i numeri stanno in un "overlay": un piano figlio della matrice con una CanvasTexture che
 *   copre SOLO la finestra di celle visibili (più un margine), al più `maxCellsPerOverlay` celle;
 *   un `NumbersGroup` assegna al più `maxOverlays` overlay alle matrici più grandi sullo schermo
 *   (il budget per scena) e li ricicla;
 * - `update` confronta la camera con quella dell'ultima volta: se non si è mossa (e i dati non
 *   sono cambiati) esce subito, senza lavoro né allocazioni. Se si è mossa ridisegna l'overlay
 *   solo quando la finestra esce dal margine o cambiano precisione o risoluzione: il testo è
 *   nella texture, quindi zoomando cresce insieme alla cella senza ridisegnare;
 * - precisione adattiva, virgola decimale, segno "−", testo nero o bianco scelto per contrasto
 *   (≥ 4,5:1 su ogni colore della colormap): vedi js/shared/weights/display.js.
 */
import * as THREE from "three";
import {
  TEXT_DARK,
  TEXT_LIGHT,
  canvasCellSize,
  capWindow,
  chooseCellFormat,
  expandWindow,
  formatCell,
  quantizeSigned,
  quantizeUnsigned,
  textColorFor,
  windowAround,
  windowCells,
  windowContains,
  windowFromPoints,
} from "../shared/weights/display.js";
import { absMax } from "../shared/weights/stats.js";
import { CELL, attentionTexture, matrixPlane, weightTexture } from "./visuals.js";

const FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', 'Liberation Mono', monospace";
const GRID_COLOR = "rgba(3, 6, 12, 0.5)";
/** Lato massimo della texture delle celle: oltre si riassume a blocchi (il valore più estremo del blocco). */
const MAX_TEXTURE_SIDE = 4096;

// ---------------------------------------------------------------- colormap (le stesse di visuals.js)

const LUTS = new Map();

function toBase64(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

/**
 * Tabella di 256 colori RGBA della colormap: "diverging" (pesi, 128 = zero, blu negativo →
 * scuro → arancione positivo) o "attention" (0…1, scuro → azzurro). Si ottiene facendo disegnare
 * una rampa a weightTexture / attentionTexture di core/visuals.js: stessi colori delle heatmap
 * esistenti senza riscrivere le formule. "masked" sono i 4 colori del tratteggio del futuro.
 * @returns {Uint8Array} 256 × 4 byte (per "masked": 4 × 4)
 */
export function colormapLUT(name = "diverging") {
  if (LUTS.has(name)) return LUTS.get(name);
  const ramp = Uint8Array.from({ length: 256 }, (_, i) => i);
  if (name === "diverging") {
    const tex = weightTexture({ shape: [1, 256], data: toBase64(ramp) });
    LUTS.set(name, new Uint8Array(tex.image.data));
    tex.dispose();
  } else if (name === "attention" || name === "masked") {
    // Matrice 256×256 con l'ultima riga = 0…255: tutta sotto la diagonale (colori dell'attention);
    // la riga 0 dalla colonna 1 è sopra la diagonale (tratteggio). La texture parte dal basso.
    const T = 256;
    const bytes = new Uint8Array(T * T);
    bytes.set(ramp, (T - 1) * T);
    const tex = attentionTexture(toBase64(bytes), T);
    const data = tex.image.data;
    LUTS.set("attention", new Uint8Array(data.subarray(0, 256 * 4)));
    const row0 = (T - 1) * T * 4;
    const masked = new Uint8Array(16);
    for (let k = 0; k < 4; k++) masked.set(data.subarray(row0 + (4 + k) * 4, row0 + (5 + k) * 4), k * 4); // (0 + j) % 4 con j = 4…7
    LUTS.set("masked", masked);
    tex.dispose();
  } else {
    throw new Error(`colormap sconosciuta: ${name} (disponibili: diverging, attention)`);
  }
  return LUTS.get(name);
}

const textLUTs = new Map();
/** Colore del testo per ogni indice della colormap (contrasto massimo). */
function textLUT(name) {
  if (!textLUTs.has(name)) {
    const lut = colormapLUT(name);
    textLUTs.set(name, Array.from({ length: lut.length / 4 }, (_, i) => textColorFor(lut[4 * i], lut[4 * i + 1], lut[4 * i + 2])));
  }
  return textLUTs.get(name);
}

const defaultPixelRatio = () => Math.min(globalThis.devicePixelRatio || 1, 2);
function defaultCanvas() {
  if (typeof document !== "undefined") return document.createElement("canvas");
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(1, 1);
  throw new Error("serve un canvas 2D per scrivere i numeri (passa createCanvas a NumbersGroup)");
}

// ---------------------------------------------------------------- temporanei (niente allocazioni per frame)

const _inv = new THREE.Matrix4();
const _normal = new THREE.Matrix3();
const _n = new THREE.Vector3();
const _p0 = new THREE.Vector3();
const _o = new THREE.Vector3();
const _d = new THREE.Vector3();
const _v = new THREE.Vector3();
const _hit = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _corners = [
  [-1, -1],
  [1, -1],
  [1, 1],
  [-1, 1],
];
const _points = _corners.map(() => ({ x: 0, y: 0 }));

/** Visibile e dentro una scena: un mesh tolto dalla scena (es. vista ricostruita) non consuma il budget. */
function visibleInScene(object) {
  let o = object;
  for (; o.parent; o = o.parent) if (!o.visible) return false;
  return o.visible && o.isScene === true;
}

// ---------------------------------------------------------------- bersaglio: una matrice da annotare

/**
 * Una matrice su cui scrivere i numeri: un mesh piano (asse x = colonne, y = righe, riga 0 in
 * alto) più come leggere valore e colore di ogni cella. Lo usano NumericMatrix e attachNumbers.
 */
class NumbersTarget {
  constructor(mesh, { rows, cols, getValue, cellColor, paintBackground = false, masked = null, maskedText = "", rowLabel = null, colLabel = null }) {
    if (!mesh?.isMesh || !mesh.geometry) throw new TypeError("attachNumbers: serve un THREE.Mesh piano (es. matrixPlane di core/visuals.js)");
    if (!(Number.isInteger(rows) && rows > 0 && Number.isInteger(cols) && cols > 0)) throw new RangeError(`righe e colonne devono essere interi positivi, ricevuti ${rows} × ${cols}`);
    this.mesh = mesh;
    this.rows = rows;
    this.cols = cols;
    this.getValue = getValue;
    this.cellColor = cellColor; // (r, c, out[3]) → scrive il colore di sfondo, restituisce il colore del testo
    this.paintBackground = paintBackground;
    this.masked = masked;
    this.maskedText = maskedText;
    this.rowLabel = rowLabel;
    this.colLabel = colLabel;
    this._version = 0;
    this._maxAbsVersion = -1;
    this._maxAbs = 0;
    this._rgb = [0, 0, 0];
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    const bb = mesh.geometry.boundingBox;
    this.rect = { minX: bb.min.x, maxX: bb.max.x, minY: bb.min.y, maxY: bb.max.y };
    this.highlightLine = null;
  }

  /** Cambia quando cambiano i dati (o la texture della heatmap): l'overlay si ridisegna. */
  version() {
    return this._version;
  }

  touch() {
    this._version++;
  }

  /** |valore| massimo, per sapere quanto è largo il numero più lungo (in cache per versione). */
  maxAbs() {
    const v = this.version();
    if (v !== this._maxAbsVersion) {
      let m = 0;
      for (let r = 0; r < this.rows; r++) {
        for (let c = 0; c < this.cols; c++) {
          const x = Math.abs(this.getValue(r, c));
          if (x > m && x !== Infinity) m = x;
        }
      }
      this._maxAbs = m;
      this._maxAbsVersion = v;
    }
    return this._maxAbs;
  }

  /** Larghezza / altezza di una cella nel mondo (le barre dei vettori sono scalate in x). */
  worldAspect() {
    const s = this.mesh.getWorldScale(_a);
    const cw = ((this.rect.maxX - this.rect.minX) / this.cols) * Math.abs(s.x);
    const ch = ((this.rect.maxY - this.rect.minY) / this.rows) * Math.abs(s.y);
    return ch > 0 ? cw / ch : 1;
  }

  /** Punto del piano (coordinate locali) colpito dal raggio della camera per il punto NDC (x, y), o null. */
  _rayLocal(camera, x, y) {
    _d.set(x, y, 0.5).unproject(camera).sub(_o).normalize();
    const denom = _d.dot(_n);
    if (Math.abs(denom) < 1e-9) return null;
    const t = _v.subVectors(_p0, _o).dot(_n) / denom;
    if (!(t > 0)) return null;
    return _hit.copy(_o).addScaledVector(_d, t).applyMatrix4(_inv);
  }

  /**
   * Quanto è grande e visibile la matrice per questa camera: null se è nascosta, di spalle,
   * fuori dallo schermo o con celle troppo piccole per i numeri.
   */
  measure(camera, width, height) {
    const mesh = this.mesh;
    if (!visibleInScene(mesh)) return null;
    mesh.updateWorldMatrix(true, false);
    const mw = mesh.matrixWorld;
    _inv.copy(mw).invert();
    _p0.setFromMatrixPosition(mw);
    _n.set(0, 0, 1).applyMatrix3(_normal.getNormalMatrix(mw)).normalize();
    _o.setFromMatrixPosition(camera.matrixWorld);
    if (_v.subVectors(_o, _p0).dot(_n) <= 0) return null; // la camera la guarda da dietro
    const { rect, rows, cols } = this;
    const cw = (rect.maxX - rect.minX) / cols;
    const ch = (rect.maxY - rect.minY) / rows;
    // La cella al centro dello schermo (o la più vicina): lì si misura la grandezza delle celle.
    const center = this._rayLocal(camera, 0, 0);
    let fx = (rect.minX + rect.maxX) / 2;
    let fy = (rect.minY + rect.maxY) / 2;
    if (center) {
      fx = Math.min(Math.max(center.x, rect.minX + cw / 2), rect.maxX - cw / 2);
      fy = Math.min(Math.max(center.y, rect.minY + ch / 2), rect.maxY - ch / 2);
    }
    _a.set(fx, fy, 0).applyMatrix4(mw);
    if (_b.copy(_a).applyMatrix4(camera.matrixWorldInverse).z > -camera.near) return null; // dietro la camera
    _a.project(camera);
    _b.set(fx + cw, fy, 0).applyMatrix4(mw).project(camera);
    const cellW = Math.hypot(((_b.x - _a.x) * width) / 2, ((_b.y - _a.y) * height) / 2);
    _b.set(fx, fy + ch, 0).applyMatrix4(mw).project(camera);
    const cellH = Math.hypot(((_b.x - _a.x) * width) / 2, ((_b.y - _a.y) * height) / 2);
    // Prima la prova più economica: se non ci sta nemmeno "−0,00" non serve scorrere i valori
    // (conta quando chi aggiorna le heatmap le riscrive tutte molte volte al secondo).
    if (!chooseCellFormat({ cellW, cellH, maxAbs: 0 })) return null;
    const format = chooseCellFormat({ cellW, cellH, maxAbs: this.maxAbs() });
    if (!format) return null;
    const focus = {
      row: Math.min(rows - 1, Math.max(0, Math.floor((rect.maxY - fy) / ch))),
      col: Math.min(cols - 1, Math.max(0, Math.floor((fx - rect.minX) / cw))),
    };
    // Celle visibili: dove i raggi degli angoli dello schermo colpiscono il piano.
    let visible = null;
    let all = true;
    for (let i = 0; i < 4; i++) {
      const p = this._rayLocal(camera, _corners[i][0], _corners[i][1]);
      if (!p) {
        all = false;
        break;
      }
      _points[i].x = p.x;
      _points[i].y = p.y;
    }
    if (all) visible = windowFromPoints(_points, rect, rows, cols);
    else visible = windowAround(focus.row, focus.col, height / cellH / 2 + 1, width / cellW / 2 + 1, rows, cols);
    if (!visible) return null;
    return { score: windowCells(visible) * cellW * cellH, cellW, cellH, format, focus, visible };
  }

  /** Cella colpita da un'intersezione del raycaster (usa le uv del piano), o null. */
  cellAt(hit) {
    if (!hit || hit.object !== this.mesh || !hit.uv) return null;
    const col = Math.min(this.cols - 1, Math.max(0, Math.floor(hit.uv.x * this.cols)));
    const row = Math.min(this.rows - 1, Math.max(0, Math.floor((1 - hit.uv.y) * this.rows)));
    const cell = { row, col, index: row * this.cols + col, value: this.getValue(row, col) };
    if (this.masked?.(row, col)) cell.masked = true;
    if (this.rowLabel) cell.rowLabel = this.rowLabel(row);
    if (this.colLabel) cell.colLabel = this.colLabel(col);
    return cell;
  }

  /** Riquadro attorno a una cella (quella sotto il mouse); null lo nasconde. */
  highlight(cell) {
    if (!cell) {
      if (this.highlightLine) this.highlightLine.visible = false;
      return;
    }
    if (!this.highlightLine) {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, -0.5, 0), new THREE.Vector3(0.5, 0.5, 0), new THREE.Vector3(-0.5, 0.5, 0)]);
      this.highlightLine = new THREE.LineLoop(geo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthWrite: false }));
      this.highlightLine.raycast = () => {};
      this.highlightLine.userData.noBounds = true;
      this.highlightLine.renderOrder = 5;
      this.mesh.add(this.highlightLine);
    }
    const { rect, rows, cols } = this;
    const cw = (rect.maxX - rect.minX) / cols;
    const ch = (rect.maxY - rect.minY) / rows;
    this.highlightLine.position.set(rect.minX + (cell.col + 0.5) * cw, rect.maxY - (cell.row + 0.5) * ch, 2e-4 * Math.min(cw, ch) + 1e-5);
    this.highlightLine.scale.set(cw, ch, 1);
    this.highlightLine.visible = true;
  }

  dispose() {
    if (this.highlightLine) {
      this.highlightLine.removeFromParent();
      this.highlightLine.geometry.dispose();
      this.highlightLine.material.dispose();
      this.highlightLine = null;
    }
  }
}

// ---------------------------------------------------------------- overlay: il canvas con i numeri

class NumbersOverlay {
  constructor(group) {
    this.group = group;
    this.canvas = group.createCanvas();
    this.ctx = this.canvas.getContext("2d");
    this.texture = null;
    this.material = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4, toneMapped: false });
    this.mesh = new THREE.Mesh(group.plane, this.material);
    this.mesh.raycast = () => {}; // i clic arrivano alla matrice sotto, non al testo
    this.mesh.userData.noBounds = true;
    this.mesh.userData.numbersOverlay = true;
    this.mesh.visible = false;
    this.state = null;
  }

  attach(target) {
    if (this.mesh.parent !== target.mesh) target.mesh.add(this.mesh);
    this.mesh.renderOrder = target.mesh.renderOrder + 1;
    this.mesh.visible = true;
  }

  detach() {
    this.mesh.removeFromParent();
    this.mesh.visible = false;
    this.state = null;
  }

  /** Mostra i numeri della misura `m`; ridisegna solo se serve. @returns {boolean} ridisegnato */
  show(target, m) {
    const g = this.group;
    const { rows, cols } = target;
    const required = capWindow(m.visible, m.focus, g.maxCellsPerOverlay);
    const aspect = target.worldAspect();
    const st = this.state;
    if (st && st.target === target && st.version === target.version() && st.decimals === m.format.decimals && st.showIndex === m.format.showIndex && windowContains(st.window, required)) {
      const want = canvasCellSize({ cellH: m.cellH, aspect, pixelRatio: g.pixelRatio, rows: st.window.r1 - st.window.r0, cols: st.window.c1 - st.window.c0, maxPixels: g.pixelBudget });
      // Stessa risoluzione (a meno di un gradino in più o in meno): non si ridisegna.
      if (want.ph <= st.ph * 1.2 && want.ph >= st.ph / 2) return false;
    }
    const drawn = capWindow(expandWindow(required, g.margin, 2, rows, cols), m.focus, g.maxCellsPerOverlay);
    const size = canvasCellSize({ cellH: m.cellH, aspect, pixelRatio: g.pixelRatio, rows: drawn.r1 - drawn.r0, cols: drawn.c1 - drawn.c0, maxPixels: g.pixelBudget });
    this.draw(target, drawn, m.format, size);
    // Il piano dell'overlay copre esattamente le celle disegnate (coordinate locali della matrice).
    const { rect } = target;
    const cw = (rect.maxX - rect.minX) / cols;
    const ch = (rect.maxY - rect.minY) / rows;
    const x0 = rect.minX + drawn.c0 * cw;
    const x1 = rect.minX + drawn.c1 * cw;
    const y1 = rect.maxY - drawn.r0 * ch;
    const y0 = rect.maxY - drawn.r1 * ch;
    this.mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, 1e-4 * Math.min(cw, ch) + 1e-6);
    this.mesh.scale.set(x1 - x0, y1 - y0, 1);
    this.state = { target, window: drawn, version: target.version(), decimals: m.format.decimals, showIndex: m.format.showIndex, ph: size.ph };
    return true;
  }

  draw(target, w, format, { pw, ph }) {
    const t0 = performance.now();
    const nr = w.r1 - w.r0;
    const nc = w.c1 - w.c0;
    const width = nc * pw;
    const height = nr * ph;
    const { canvas, ctx } = this;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      // Una texture non cambia dimensione sulla GPU: se ne crea una nuova.
      this.texture?.dispose();
      this.texture = null;
    }
    ctx.clearRect(0, 0, width, height);
    const rgb = target._rgb;
    const colors = new Array(nr * nc);
    for (let r = 0; r < nr; r++) {
      for (let c = 0; c < nc; c++) {
        const color = target.cellColor(w.r0 + r, w.c0 + c, rgb);
        colors[r * nc + c] = color;
        if (target.paintBackground) {
          ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
          ctx.fillRect(c * pw, r * ph, pw, ph);
        }
      }
    }
    // Righe sottili tra le celle: due celle vicine di colore simile restano distinguibili.
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = Math.max(1, Math.round(ph / 40));
    ctx.beginPath();
    for (let c = 1; c < nc; c++) {
      ctx.moveTo(c * pw, 0);
      ctx.lineTo(c * pw, height);
    }
    for (let r = 1; r < nr; r++) {
      ctx.moveTo(0, r * ph);
      ctx.lineTo(width, r * ph);
    }
    ctx.stroke();

    const { decimals, showIndex } = format;
    const widest = formatCell(-target.maxAbs(), decimals);
    let fontPx = Math.min(ph * 0.42, (pw * 0.86) / (widest.length * 0.62));
    ctx.font = `600 ${fontPx.toFixed(1)}px ${FONT}`;
    const measured = ctx.measureText(widest).width;
    if (measured > pw * 0.9) {
      fontPx *= (pw * 0.9) / measured; // il font del sistema è più largo del previsto
      ctx.font = `600 ${fontPx.toFixed(1)}px ${FONT}`;
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const yText = showIndex ? 0.58 : 0.5;
    let current = null;
    let count = 0;
    for (let r = 0; r < nr; r++) {
      for (let c = 0; c < nc; c++) {
        const row = w.r0 + r;
        const col = w.c0 + c;
        const color = colors[r * nc + c];
        if (color !== current) {
          ctx.fillStyle = color;
          current = color;
        }
        const text = target.masked?.(row, col) ? target.maskedText : formatCell(target.getValue(row, col), decimals);
        if (text) ctx.fillText(text, (c + 0.5) * pw, (r + yText) * ph);
        count++;
      }
    }
    if (showIndex) {
      // Celle grandi: in piccolo, in alto a sinistra, riga e colonna (come si indicizza in Python).
      ctx.font = `500 ${(ph * 0.15).toFixed(1)}px ${FONT}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.globalAlpha = 0.7;
      current = null;
      for (let r = 0; r < nr; r++) {
        for (let c = 0; c < nc; c++) {
          const color = colors[r * nc + c];
          if (color !== current) {
            ctx.fillStyle = color;
            current = color;
          }
          ctx.fillText(`${w.r0 + r},${w.c0 + c}`, c * pw + ph * 0.07, r * ph + ph * 0.06);
        }
      }
      ctx.globalAlpha = 1;
    }
    if (!this.texture) {
      this.texture = new THREE.CanvasTexture(canvas);
      this.texture.colorSpace = THREE.SRGBColorSpace;
      this.texture.generateMipmaps = false;
      this.texture.minFilter = THREE.LinearFilter;
      this.texture.magFilter = THREE.LinearFilter;
      const first = !this.material.map;
      this.material.map = this.texture;
      if (first) this.material.needsUpdate = true;
    } else {
      this.texture.needsUpdate = true;
    }
    const g = this.group;
    g.stats.redraws++;
    g.stats.lastRedrawMs = performance.now() - t0;
    g.stats.maxRedrawMs = Math.max(g.stats.maxRedrawMs, g.stats.lastRedrawMs);
    g.stats.lastRedrawCells = count;
  }

  dispose() {
    this.detach();
    this.texture?.dispose();
    this.material.dispose();
    this.texture = null;
  }
}

// ---------------------------------------------------------------- gruppo: il budget per scena

/**
 * Le matrici di una scena che condividono il budget dei numeri: al più `maxOverlays` matrici
 * mostrano i numeri insieme (le più grandi sullo schermo), ognuna al più `maxCellsPerOverlay`
 * celle. Chiamare `update` una volta per frame (anche più volte: dopo la prima esce subito).
 */
export class NumbersGroup {
  /**
   * @param {object} [o]
   * @param {number} [o.maxOverlays] matrici con i numeri contemporaneamente (predefinito 4)
   * @param {number} [o.maxCellsPerOverlay] celle con il numero per matrice (predefinito 2.500)
   * @param {number} [o.maxCanvasPixels] pixel massimi del canvas di un overlay (predefinito 4 M = 16 MB)
   * @param {number} [o.maxTotalPixels] pixel massimi di tutti gli overlay insieme (predefinito 8 M = 32 MB):
   *   con più matrici vicine ognuna riceve la sua parte
   * @param {number} [o.margin] celle in più per lato da disegnare, come frazione della finestra
   *   visibile: un piccolo spostamento non obbliga a ridisegnare (predefinito 0,25)
   * @param {number} [o.pixelRatio] predefinito devicePixelRatio (al più 2)
   * @param {() => HTMLCanvasElement|OffscreenCanvas} [o.createCanvas] per i test senza DOM
   */
  constructor({ maxOverlays = 4, maxCellsPerOverlay = 2500, maxCanvasPixels = 4_194_304, maxTotalPixels = 8_388_608, margin = 0.25, pixelRatio = defaultPixelRatio(), createCanvas = defaultCanvas } = {}) {
    this.maxOverlays = maxOverlays;
    this.maxCellsPerOverlay = maxCellsPerOverlay;
    this.maxCanvasPixels = maxCanvasPixels;
    this.maxTotalPixels = maxTotalPixels;
    this.pixelBudget = maxCanvasPixels; // parte di ogni overlay, ricalcolata a ogni assegnazione
    this.margin = margin;
    this.pixelRatio = pixelRatio;
    this.createCanvas = createCanvas;
    this.plane = new THREE.PlaneGeometry(1, 1);
    this.targets = new Set();
    this.active = new Map(); // bersaglio → overlay
    this.pool = [];
    this._signature = new Float64Array(34).fill(NaN);
    this._versions = new Map();
    this._dirty = true;
    this._candidates = [];
    this.stats = { updates: 0, idle: 0, recomputes: 0, redraws: 0, lastRecomputeMs: 0, lastRedrawMs: 0, maxRedrawMs: 0, lastRedrawCells: 0, overlays: 0 };
  }

  add(target) {
    this.targets.add(target);
    this._dirty = true;
  }

  remove(target) {
    this._release(target);
    this.targets.delete(target);
    this._versions.delete(target);
    this._dirty = true;
  }

  /** Ricalcola al prossimo update anche se la camera è ferma (es. matrici spostate da codice). */
  invalidate() {
    this._dirty = true;
  }

  /**
   * @param {THREE.PerspectiveCamera} camera
   * @param {number} viewportHeightPx altezza in px CSS del canvas (la larghezza viene da camera.aspect)
   * @returns {boolean} true se ha ricalcolato qualcosa
   */
  update(camera, viewportHeightPx) {
    this.stats.updates++;
    camera.updateMatrixWorld();
    const sig = this._signature;
    const e = camera.matrixWorld.elements;
    const p = camera.projectionMatrix.elements;
    let changed = this._dirty;
    for (let i = 0; i < 16; i++) {
      if (sig[i] !== e[i]) {
        sig[i] = e[i];
        changed = true;
      }
      if (sig[16 + i] !== p[i]) {
        sig[16 + i] = p[i];
        changed = true;
      }
    }
    if (sig[32] !== viewportHeightPx || sig[33] !== camera.aspect) {
      sig[32] = viewportHeightPx;
      sig[33] = camera.aspect;
      changed = true;
    }
    for (const t of this.targets) {
      const v = t.version();
      if (this._versions.get(t) !== v) {
        this._versions.set(t, v);
        changed = true;
      }
    }
    if (!changed) {
      this.stats.idle++;
      return false;
    }
    this._dirty = false;
    this._recompute(camera, viewportHeightPx);
    return true;
  }

  _recompute(camera, height) {
    const t0 = performance.now();
    const width = height * camera.aspect;
    const list = this._candidates;
    list.length = 0;
    for (const t of this.targets) {
      const m = t.measure(camera, width, height);
      if (m) list.push({ target: t, m });
    }
    list.sort((a, b) => b.m.score - a.m.score);
    const granted = list.slice(0, this.maxOverlays);
    this.pixelBudget = Math.min(this.maxCanvasPixels, this.maxTotalPixels / Math.max(1, granted.length));
    const keep = new Set(granted.map((g) => g.target));
    for (const target of [...this.active.keys()]) if (!keep.has(target)) this._release(target);
    for (const { target, m } of granted) {
      let overlay = this.active.get(target);
      if (!overlay) {
        overlay = this.pool.pop() ?? new NumbersOverlay(this);
        overlay.state = null;
        this.active.set(target, overlay);
      }
      overlay.attach(target);
      overlay.show(target, m);
    }
    this.stats.overlays = this.active.size;
    this.stats.recomputes++;
    this.stats.lastRecomputeMs = performance.now() - t0;
  }

  _release(target) {
    const overlay = this.active.get(target);
    if (!overlay) return;
    overlay.detach();
    this.active.delete(target);
    this.pool.push(overlay);
  }

  /** Matrici che in questo momento mostrano i numeri, con la finestra di celle disegnata. */
  shown() {
    return [...this.active].map(([target, o]) => ({ target, window: o.state?.window ?? null, decimals: o.state?.decimals ?? null }));
  }

  dispose() {
    for (const target of [...this.active.keys()]) this._release(target);
    for (const o of this.pool) o.dispose();
    this.pool.length = 0;
    this.targets.clear();
    this.plane.dispose();
  }
}

// ---------------------------------------------------------------- matrice con i numeri

/**
 * Una matrice di pesi come celle colorate con i numeri veri.
 * Righe e colonne come in PyTorch: un nn.Linear (out, in) è alto `out`, la riga 0 è in alto.
 * Un tensore 1D (bias, γ, β) va passato come una riga: `rows: 1, cols: n` (o con fromTensor).
 */
export class NumericMatrix {
  /**
   * @param {object} o
   * @param {number} o.rows
   * @param {number} o.cols
   * @param {ArrayLike<number>} o.data valori riga per riga (almeno rows × cols)
   * @param {number} [o.cell] lato di una cella nel mondo (predefinito CELL di core/visuals.js)
   * @param {number|null} [o.range] scala dei colori: |valore| che dà il colore pieno (predefinito il massimo |valore|)
   * @param {"diverging"|"attention"} [o.colormap]
   * @param {NumbersGroup|null} [o.group] budget condiviso con altre matrici della scena
   * @param {(r: number, c: number) => boolean} [o.masked] celle "mascherate" (tratteggio, es. il futuro dell'attention)
   * @param {string} [o.maskedText] testo delle celle mascherate (es. "−∞" per i punteggi)
   * @param {(r: number) => string} [o.rowLabel] etichetta di una riga (es. il token di una riga dell'embedding)
   * @param {(c: number) => string} [o.colLabel]
   */
  constructor({ rows, cols, data, cell = CELL, range = null, colormap = "diverging", group = null, masked = null, maskedText = "", rowLabel = null, colLabel = null } = {}) {
    if (!(Number.isInteger(rows) && rows > 0 && Number.isInteger(cols) && cols > 0)) throw new RangeError(`righe e colonne devono essere interi positivi, ricevuti ${rows} × ${cols}`);
    if (!data || data.length < rows * cols) throw new RangeError(`servono ${rows * cols} valori, ricevuti ${data?.length}`);
    this.rows = rows;
    this.cols = cols;
    this.data = data;
    this.colormap = colormap;
    this.lut = colormapLUT(colormap);
    this.texts = textLUT(colormap);
    this.masked = masked;
    this.range = range ?? this._autoRange(data);
    // Matrici enormi (es. un embedding da 50.000 righe): la texture riassume a blocchi, i numeri restano esatti.
    this.blockY = Math.ceil(rows / MAX_TEXTURE_SIDE);
    this.blockX = Math.ceil(cols / MAX_TEXTURE_SIDE);
    const tw = Math.ceil(cols / this.blockX);
    const th = Math.ceil(rows / this.blockY);
    this.texture = new THREE.DataTexture(new Uint8Array(tw * th * 4), tw, th, THREE.RGBAFormat);
    this.texture.magFilter = THREE.NearestFilter; // celle nette, come core/visuals.js
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.flipY = false;
    this.mesh = matrixPlane(this.texture, rows, cols);
    if (cell !== CELL) this.mesh.scale.set(cell / CELL, cell / CELL, 1);
    this.mesh.userData.numericMatrix = this;
    this.object3d = new THREE.Group();
    this.object3d.add(this.mesh);

    const maskedRGB = masked ? colormapLUT("masked") : null;
    const downsampled = this.blockX > 1 || this.blockY > 1;
    this.target = new NumbersTarget(this.mesh, {
      rows,
      cols,
      getValue: (r, c) => this.data[r * this.cols + c],
      paintBackground: downsampled,
      masked,
      maskedText,
      rowLabel,
      colLabel,
      cellColor: (r, c, out) => {
        if (maskedRGB && masked(r, c)) {
          const k = ((r + c) % 4) * 4;
          out[0] = maskedRGB[k];
          out[1] = maskedRGB[k + 1];
          out[2] = maskedRGB[k + 2];
          return textColorFor(out[0], out[1], out[2]);
        }
        const i = this._index(this.data[r * this.cols + c]);
        out[0] = this.lut[4 * i];
        out[1] = this.lut[4 * i + 1];
        out[2] = this.lut[4 * i + 2];
        return this.texts[i];
      },
    });
    this.group = group ?? new NumbersGroup({ maxOverlays: 1 });
    this._ownGroup = !group;
    this.group.add(this.target);
    this._writeTexture();
  }

  /** Da un tensore `{shape, data}` (1D → una riga, oltre 2D → reshape(-1, ultima)). */
  static fromTensor(tensor, options = {}) {
    const shape = tensor.shape ?? [tensor.data.length];
    const cols = shape.length ? shape[shape.length - 1] : 1;
    const rows = shape.length > 1 ? shape.slice(0, -1).reduce((a, b) => a * b, 1) : 1;
    return new NumericMatrix({ rows, cols, data: tensor.data, ...options });
  }

  _autoRange(data) {
    if (this.colormap === "attention") {
      let m = 0;
      for (let i = 0; i < this.rows * this.cols; i++) if (data[i] > m && data[i] !== Infinity) m = data[i];
      return m > 0 ? m : 1;
    }
    return absMax(data.length === this.rows * this.cols ? data : Array.prototype.slice.call(data, 0, this.rows * this.cols)) || 1;
  }

  _index(v) {
    return this.colormap === "attention" ? quantizeUnsigned(v, this.range) : quantizeSigned(v, this.range);
  }

  /** Scrive i colori delle celle nella DataTexture (riga 0 in alto: la texture parte dal basso). */
  _writeTexture() {
    const { rows, cols, blockX, blockY, data, lut } = this;
    const tex = this.texture.image;
    const out = tex.data;
    const maskedRGB = this.masked ? colormapLUT("masked") : null;
    for (let ty = 0; ty < tex.height; ty++) {
      for (let tx = 0; tx < tex.width; tx++) {
        // Il valore più estremo del blocco (1 × 1 per le matrici normali): gli outlier restano visibili.
        let best = 0;
        let bestAbs = -1;
        let isMasked = false;
        const r0 = ty * blockY;
        const c0 = tx * blockX;
        for (let r = r0; r < Math.min(rows, r0 + blockY); r++) {
          for (let c = c0; c < Math.min(cols, c0 + blockX); c++) {
            if (maskedRGB && this.masked(r, c)) {
              isMasked = true;
              continue;
            }
            const v = data[r * cols + c];
            const a = Math.abs(v);
            if (a > bestAbs) {
              bestAbs = a;
              best = v;
            }
          }
        }
        const o = ((tex.height - 1 - ty) * tex.width + tx) * 4;
        if (isMasked && bestAbs < 0) {
          const k = ((r0 + c0) % 4) * 4;
          out[o] = maskedRGB[k];
          out[o + 1] = maskedRGB[k + 1];
          out[o + 2] = maskedRGB[k + 2];
        } else {
          const i = 4 * this._index(best);
          out[o] = lut[i];
          out[o + 1] = lut[i + 1];
          out[o + 2] = lut[i + 2];
        }
        out[o + 3] = 255;
      }
    }
    this.texture.needsUpdate = true;
  }

  /** Nuovi valori (stessa forma), es. un altro step del training. La scala resta, a meno di `{range}` ("auto" = ricalcola). */
  setData(data, { range } = {}) {
    if (!data || data.length < this.rows * this.cols) throw new RangeError(`servono ${this.rows * this.cols} valori, ricevuti ${data?.length}`);
    this.data = data;
    if (range === "auto") this.range = this._autoRange(data);
    else if (typeof range === "number" && range > 0) this.range = range;
    this._writeTexture();
    this.target.touch();
  }

  /** Cambia la scala dei colori (|valore| per il colore pieno). */
  setRange(range) {
    this.range = range > 0 ? range : this._autoRange(this.data);
    this._writeTexture();
    this.target.touch();
  }

  /** Aggiorna i numeri per questa camera; esce subito se nulla è cambiato. */
  update(camera, viewportHeightPx) {
    return this.group.update(camera, viewportHeightPx);
  }

  /** Cella di un'intersezione del raycaster (`host.pick(event, [m.mesh])[0]`): {row, col, index, value} o null. */
  cellAt(intersection) {
    return this.target.cellAt(intersection);
  }

  /** Riquadro sulla cella (es. quella sotto il mouse); null lo toglie. */
  highlight(cell) {
    this.target.highlight(cell);
  }

  /** true se in questo momento i numeri sono scritti sulla matrice. */
  get showingNumbers() {
    return this.group.active.has(this.target);
  }

  dispose() {
    this.group.remove(this.target);
    this.target.dispose();
    if (this._ownGroup) this.group.dispose();
    this.object3d.removeFromParent();
    this.mesh.traverse((o) => {
      if (o === this.mesh) return;
      o.geometry?.dispose();
      o.material?.dispose();
    });
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.texture.dispose();
  }
}

// ---------------------------------------------------------------- numeri sulle heatmap esistenti

/**
 * Aggiunge i numeri a una heatmap già esistente (piano di matrixPlane con la DataTexture di
 * weightTexture o simile: riga 0 in alto). Il colore del testo segue il colore VERO di ogni
 * cella letto dalla texture; se la texture viene riscritta (es. pesi di uno snapshot del
 * training, `texture.needsUpdate = true`) i numeri si ridisegnano da soli al prossimo update.
 * @param {THREE.Mesh} mesh
 * @param {{rows?: number, cols?: number, getValue: (r: number, c: number) => number, range?: number,
 *   colormap?: "diverging"|"attention", group?: NumbersGroup, masked?: (r, c) => boolean, maskedText?: string,
 *   rowLabel?: (r) => string, colLabel?: (c) => string}} o
 *   rows/cols: predefiniti dalle dimensioni della texture; range e colormap servono solo se la
 *   texture non si può leggere (allora il colore si ricalcola dal valore)
 * @returns {{target: object, update: (camera, h) => boolean, refresh: () => void, cellAt: (hit) => object|null,
 *   highlight: (cell) => void, showing: () => boolean, dispose: () => void}}
 */
export function attachNumbers(mesh, { rows, cols, getValue, range = null, colormap = "diverging", group = null, masked = null, maskedText = "", rowLabel = null, colLabel = null } = {}) {
  if (typeof getValue !== "function") throw new TypeError("attachNumbers: serve getValue(riga, colonna)");
  const texture = mesh?.material?.map ?? null;
  const image = texture?.image;
  rows ??= image?.height;
  cols ??= image?.width;
  const lut = colormapLUT(colormap);
  const texts = textLUT(colormap);
  const readable = Boolean(image?.data && image.width === cols && image.height === rows && image.data.length >= rows * cols * 4);
  const cache = new Map();
  let scale = range;
  const target = new NumbersTarget(mesh, {
    rows,
    cols,
    getValue,
    masked,
    maskedText,
    rowLabel,
    colLabel,
    cellColor: (r, c, out) => {
      if (readable) {
        const o = ((rows - 1 - r) * cols + c) * 4; // DataTexture parte dal basso
        const d = image.data;
        out[0] = d[o];
        out[1] = d[o + 1];
        out[2] = d[o + 2];
        const key = (out[0] << 16) | (out[1] << 8) | out[2];
        let color = cache.get(key);
        if (!color) {
          color = textColorFor(out[0], out[1], out[2]);
          cache.set(key, color);
        }
        return color;
      }
      scale ??= target.maxAbs() || 1;
      const v = getValue(r, c);
      const i = colormap === "attention" ? quantizeUnsigned(v, scale) : quantizeSigned(v, scale);
      out[0] = lut[4 * i];
      out[1] = lut[4 * i + 1];
      out[2] = lut[4 * i + 2];
      return texts[i];
    },
  });
  // Versione = cambi della texture (riscritta da chi aggiorna la heatmap) + refresh espliciti.
  let own = 0;
  target.version = () => own + (texture?.version ?? 0) * 1e6;
  target.touch = () => {
    own++;
  };
  const g = group ?? new NumbersGroup({ maxOverlays: 1 });
  g.add(target);
  return {
    target,
    mesh,
    update: (camera, viewportHeightPx) => g.update(camera, viewportHeightPx),
    /** I valori sono cambiati senza toccare la texture: ridisegna al prossimo update. */
    refresh: () => target.touch(),
    cellAt: (hit) => target.cellAt(hit),
    highlight: (cell) => target.highlight(cell),
    showing: () => g.active.has(target),
    dispose() {
      g.remove(target);
      target.dispose();
      if (!group) g.dispose();
    },
  };
}

export { TEXT_DARK, TEXT_LIGHT };
