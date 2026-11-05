/**
 * Aggiornamento delle texture della scena con dati veri (pesi del training, attention del prompt)
 * e ripristino dello stato originale a fine azione.
 *
 * Le heatmap dei pesi sono DataTexture create da core/visuals.js (1 texel = 1 peso, riga 0 in alto).
 * Qui scriviamo direttamente nei loro byte: nessuna nuova texture, nessuna allocazione per frame.
 * Prima di toccare una texture ne salviamo una copia, così `restore()` rimette esattamente i
 * pesi addestrati e i pattern d'esempio di data/model.json.
 */
import { colormaps } from "./colormaps.js";
import { quantizeSigned } from "./math.js";

export class SceneTextures {
  /** @param {object} tab l'oggetto tab del Transformer (vedi ARCHITECTURE.md) */
  constructor(tab) {
    this.tab = tab;
    this.saved = new Map(); // texture → copia dei byte originali
    this.patternSize = tab.spec.config.block_size;
  }

  save(texture) {
    if (!this.saved.has(texture)) this.saved.set(texture, new Uint8Array(texture.image.data));
  }

  /**
   * Ridisegna le heatmap di tutti i tensori con i valori di `views` (nome → Float32Array).
   * La scala resta quella dei pesi addestrati (spec.tensors[nome].scale): così si vede anche
   * quanto i pesi crescono o si restringono durante il training.
   */
  writeWeights(views) {
    // Un texel RGBA = un Uint32: si copia il colore della LUT con una sola scrittura. Durante il
    // training completo questa funzione gira 12 volte al secondo su ~210.000 texel.
    const lut = this.signed32 ??= new Uint32Array(colormaps().signed.slice().buffer);
    for (const [name, meshes] of this.tab.arch.tensorMeshes) {
      const values = views.get(name);
      const info = this.tab.spec.tensors[name];
      if (!values || !info) continue;
      for (const mesh of meshes) {
        const texture = mesh.material?.map;
        if (!texture?.image?.data) continue;
        this.save(texture);
        const { width: cols, height: rows } = texture.image;
        const texels = this.texels(texture);
        for (let r = 0; r < rows; r++) {
          const base = (rows - 1 - r) * cols; // DataTexture parte dal basso
          const row = r * cols;
          for (let c = 0; c < cols; c++) texels[base + c] = lut[quantizeSigned(values[row + c], info.scale)];
        }
        texture.needsUpdate = true;
      }
    }
  }

  /** Vista Uint32 (un elemento per texel RGBA) dei byte di una DataTexture, creata una volta. */
  texels(texture) {
    this.views ??= new WeakMap();
    let view = this.views.get(texture);
    if (!view) {
      const data = texture.image.data;
      view = new Uint32Array(data.buffer, data.byteOffset, data.length >> 2);
      this.views.set(texture, view);
    }
    return view;
  }

  /** Mesh del pattern di attention di una head (registro di architecture.js). */
  patternMesh(layer, head) {
    return this.tab.arch.registry.get(`blocks.${layer}.heads.${head}.pattern`);
  }

  /**
   * Scrive nel pattern (block_size × block_size) l'attention vera `att` (T × T, T ≤ block_size).
   * Le righe oltre T (posizioni non usate da questo prompt) restano scure.
   */
  writePattern(layer, head, att, T) {
    const texture = this.patternMesh(layer, head)?.material?.map;
    if (!texture) return;
    this.save(texture);
    const { attention, masked } = colormaps();
    const N = this.patternSize;
    const data = texture.image.data;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const o = ((N - 1 - i) * N + j) * 4;
        let src;
        let k;
        if (j > i) {
          src = masked;
          k = ((i + j) % 4) * 4;
        } else {
          src = attention;
          const v = i < T ? att[i * T + j] : 0;
          k = Math.max(0, Math.min(255, Math.round(v * 255))) * 4;
        }
        data[o] = src[k];
        data[o + 1] = src[k + 1];
        data[o + 2] = src[k + 2];
      }
    }
    texture.needsUpdate = true;
  }

  /** Rimette tutte le texture toccate com'erano (pesi addestrati, pattern d'esempio). */
  restore() {
    for (const [texture, bytes] of this.saved) {
      texture.image.data.set(bytes);
      texture.needsUpdate = true;
    }
    this.saved.clear();
  }
}
