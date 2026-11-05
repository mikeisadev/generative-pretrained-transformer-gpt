/**
 * Costruisce la scena 3D a partire da data/model.json.
 *
 * Il flusso dei dati sale dal basso verso l'alto, come nella figura del paper:
 *   riga 0:  token → embedding (+)
 *   riga 1…n_layer: un blocco transformer per riga, percorso da sinistra a destra
 *   ultima riga: ln_f → lm_head → probabilità → prossimo token
 * Ogni matrice è disegnata con 1 cella per peso, quindi le dimensioni sono proporzionali
 * alla forma vera del tensore PyTorch.
 *
 * Restituisce un registro { id → Object3D } usato da menu e camera, e la lista
 * delle "stazioni" attraversate dall'animazione del flusso.
 *
 * Ogni componente occupa uno "slot" orizzontale largo almeno quanto serve al suo rettangolo
 * nel livello "Blocchi" del LOD (lod.js): così rettangoli, binario e pacchetto del flusso
 * restano allineati a qualunque livello di zoom. `slots` (id → {row, x, width}) e `rows`
 * (centro e larghezza di ogni riga, coordinate di root) servono proprio a questo.
 */
import * as THREE from "three";
import { CELL, COLORS, addNode, attentionTexture, label, matrixPlane, tokenCube, weightTexture } from "../../core/visuals.js";

const GAP = 0.9;
const BAR_W = 0.12;

/** Un vettore 1D (γ, β o bias) disegnato come barra verticale: (1, n) → (n, 1), stessi byte. */
function vectorBar(spec, name) {
  const tensor = spec.tensors[name];
  if (!tensor) throw new Error(`tensore mancante nel JSON: ${name}`);
  const n = tensor.shape[1];
  const column = { ...tensor, shape: [n, 1] };
  const mesh = matrixPlane(weightTexture(column), n, 1);
  mesh.scale.x = BAR_W / CELL;
  mesh.userData.tensorName = name;
  mesh.userData.vector = true; // disegnato come colonna (n, 1): stessi byte di (1, n)
  return mesh;
}

function matrix(spec, name) {
  const t = spec.tensors[name];
  if (!t) throw new Error(`tensore mancante nel JSON: ${name}`);
  const mesh = matrixPlane(weightTexture(t), t.shape[0], t.shape[1]);
  mesh.userData.tensorName = name;
  return mesh;
}

function withLabel(obj, text, sub, height, { above = true, size = 0.26 } = {}) {
  const tag = label(text, { sub, size });
  tag.position.y = above ? height / 2 + 0.45 : -height / 2 - 0.45;
  obj.add(tag);
  return obj;
}

function shape(t) {
  return `(${t.shape.join(", ")})`;
}

export const ROW_SPACING = 15;

/**
 * Larghezza minima dello slot di ogni tipo di componente: lascia spazio ai rettangoli del
 * livello "Blocchi" (titolo leggibile) senza cambiare le dimensioni delle matrici (1 cella = 1 peso).
 */
const SLOT = { tokens: 4.6, embeddings: 6, add: 2.5, ln: 3.5, heads: 4.8, proj: 4.0, ln_f: 3.5, lm_head: 4.6, logits: 6.4, next: 3.8 };

class Layout {
  constructor(root) {
    this.root = root;
    this.registry = new Map();
    this.stations = [];
    this.rows = [];
    this.slots = new Map();
  }

  /** Apre una nuova riga all'altezza `index * ROW_SPACING`. */
  startRow(index) {
    this.row = new THREE.Group();
    this.row.position.y = index * ROW_SPACING;
    this.root.add(this.row);
    this.rowIndex = index;
    this.rowSlots = [];
    this.x = 0;
  }

  /** Chiude la riga centrandola sull'asse x. */
  endRow() {
    const width = this.x - GAP;
    this.row.position.x = -width / 2;
    this.rows.push({ group: this.row, width, index: this.rowIndex, y: this.row.position.y });
    for (const slot of this.rowSlots) slot.x -= width / 2; // centro dello slot in coordinate di root
  }

  /**
   * Posiziona `obj` largo `width` al centro di uno slot largo max(width, slot) e avanza.
   * Lo slot viene registrato in `slots` (stesso id del registro).
   */
  place(id, obj, width, caption, { parent = this.row, slot = 0 } = {}) {
    const slotWidth = Math.max(width, slot);
    obj.position.x = this.x + slotWidth / 2 - (parent === this.row ? 0 : parent.position.x);
    parent.add(obj);
    this.registry.set(id, obj);
    if (caption) this.stations.push({ id, caption, object: obj, row: this.rowIndex });
    const record = { row: this.rowIndex, x: this.x + slotWidth / 2, width: slotWidth, y: this.row.position.y };
    this.slots.set(id, record);
    this.rowSlots.push(record);
    this.x += slotWidth + GAP;
    return obj;
  }
}

function buildTokens(spec) {
  const group = new THREE.Group();
  const { prompt } = spec.example;
  const T = prompt.length;
  const size = 0.22;
  const step = 0.27;
  [...prompt].forEach((ch, t) => {
    const cube = tokenCube(ch, size);
    cube.position.y = ((T - 1) / 2 - t) * step;
    group.add(cube);
  });
  const height = T * step;
  withLabel(group, "idx: token di input", `(B, T)  T = ${T} = block_size`, height);
  return group;
}

function buildEmbeddings(spec, layout) {
  const tok = matrix(spec, "token_embedding_table.weight");
  const tokT = spec.tensors["token_embedding_table.weight"];
  withLabel(tok, "token_embedding_table", `${shape(tokT)}  appreso`, tokT.shape[0] * CELL);
  tok.position.y = 2.1;

  const pos = matrix(spec, "position_embedding_table.weight");
  const posT = spec.tensors["position_embedding_table.weight"];
  withLabel(pos, "position_embedding_table", `${shape(posT)}  appreso`, posT.shape[0] * CELL, { above: false });
  pos.position.y = -1.9;

  const width = tokT.shape[1] * CELL;
  const group = new THREE.Group();
  group.add(tok, pos);
  layout.place("embeddings", group, width, "Embedding: token + posizione", { slot: SLOT.embeddings });
  layout.registry.set("tok_emb", tok);
  layout.registry.set("pos_emb", pos);
  layout.place("embed_add", addNode(), 0.56, "x = tok_emb + pos_emb   (B, T, C)", { slot: SLOT.add });
}

function buildHead(spec, prefix, layer, h) {
  const group = new THREE.Group();
  const cards = ["query", "key", "value"];
  cards.forEach((kind, i) => {
    const m = matrix(spec, `${prefix}.${kind}.weight`);
    m.position.set(i * 0.14, -i * 0.1, 0.25 - i * 0.25); // carte sovrapposte, leggermente sfalsate
    group.add(m);
  });
  const t = spec.tensors[`${prefix}.query.weight`];
  const cardsW = t.shape[1] * CELL;
  const tag = label(`head ${h}: Q · K · V`, { sub: `3 × ${shape(t)}`, size: 0.2 });
  tag.position.set(0.14, t.shape[0] * CELL / 2 + 0.38, 0.3);
  group.add(tag);

  const T = spec.config.block_size;
  const pattern = matrixPlane(attentionTexture(spec.example.attention[`${layer}.${h}`], T), T, T);
  pattern.position.x = cardsW / 2 + 0.55 + (T * CELL) / 2;
  const ptag = label("attention (calcolata)", { sub: "tril: non appreso", size: 0.16 });
  ptag.position.y = -(T * CELL) / 2 - 0.3;
  pattern.add(ptag);
  group.add(pattern);
  return { group, pattern, width: cardsW + 0.55 + T * CELL + 0.3 };
}

function buildBlock(spec, layout, layer) {
  const cfg = spec.config;
  const p = `blocks.${layer}`;
  const startX = layout.x;
  const block = new THREE.Group();
  block.position.x = startX;
  layout.row.add(block);
  layout.registry.set(p, block);

  const sub = (id, obj, width, caption, slot = 0) => {
    layout.place(id, obj, width, caption, { parent: block, slot });
  };

  // LayerNorm 1: γ e β
  const ln1 = new THREE.Group();
  const g1 = vectorBar(spec, `${p}.ln1.weight`);
  const b1 = vectorBar(spec, `${p}.ln1.bias`);
  b1.position.x = 0.22;
  ln1.add(g1, b1);
  withLabel(ln1, "ln1", "γ, β (64,)".replace("64", cfg.n_embd), cfg.n_embd * CELL, { size: 0.2 });
  sub(`${p}.ln1`, ln1, 0.34, `Blocco ${layer} · LayerNorm 1`, SLOT.ln);

  // Multi-head attention: le head impilate in verticale
  const heads = new THREE.Group();
  const spacing = 1.75;
  let headsWidth = 0;
  for (let h = 0; h < cfg.n_head; h++) {
    const { group, pattern, width } = buildHead(spec, `${p}.sa.heads.${h}`, layer, h);
    group.position.y = ((cfg.n_head - 1) / 2 - h) * spacing;
    group.position.x = -width / 2 + (cfg.n_embd * CELL) / 2;
    heads.add(group);
    layout.registry.set(`${p}.heads.${h}`, group);
    layout.registry.set(`${p}.heads.${h}.pattern`, pattern);
    headsWidth = width;
  }
  sub(`${p}.heads`, heads, headsWidth, `Blocco ${layer} · ${cfg.n_head} head di self-attention`, SLOT.heads);

  // Proiezione
  const proj = new THREE.Group();
  const pw = matrix(spec, `${p}.sa.proj.weight`);
  const pb = vectorBar(spec, `${p}.sa.proj.bias`);
  pb.position.x = (cfg.n_embd * CELL) / 2 + 0.2;
  proj.add(pw, pb);
  withLabel(proj, "proj", `${shape(spec.tensors[`${p}.sa.proj.weight`])} + bias`, cfg.n_embd * CELL, { size: 0.22 });
  sub(`${p}.proj`, proj, cfg.n_embd * CELL + 0.3, `Blocco ${layer} · proiezione`, SLOT.proj);

  const add1 = addNode();
  sub(`${p}.add1`, add1, 0.56, `Blocco ${layer} · x = x + attention`, SLOT.add);

  const ln2 = new THREE.Group();
  const g2 = vectorBar(spec, `${p}.ln2.weight`);
  const b2 = vectorBar(spec, `${p}.ln2.bias`);
  b2.position.x = 0.22;
  ln2.add(g2, b2);
  withLabel(ln2, "ln2", `γ, β (${cfg.n_embd},)`, cfg.n_embd * CELL, { size: 0.2 });
  sub(`${p}.ln2`, ln2, 0.34, `Blocco ${layer} · LayerNorm 2`, SLOT.ln);

  // Feed-forward: espansione 4C, ReLU, compressione. I bias sono barre accanto alla loro matrice.
  const ffwd = new THREE.Group();
  const w1T = spec.tensors[`${p}.ffwd.net.0.weight`];
  const w2T = spec.tensors[`${p}.ffwd.net.2.weight`];
  const w1 = matrix(spec, `${p}.ffwd.net.0.weight`);
  const w1W = w1T.shape[1] * CELL;
  const fb1 = vectorBar(spec, `${p}.ffwd.net.0.bias`);
  fb1.position.x = w1W / 2 + 0.2;
  const relu = label("ReLU", { size: 0.22, color: "#a5f3fc" });
  relu.position.x = w1W / 2 + 0.8;
  const w2 = matrix(spec, `${p}.ffwd.net.2.weight`);
  const w2W = w2T.shape[1] * CELL;
  w2.position.x = w1W / 2 + 1.4 + w2W / 2;
  const fb2 = vectorBar(spec, `${p}.ffwd.net.2.bias`);
  fb2.position.x = w2.position.x + w2W / 2 + 0.2;
  withLabel(w1, "net.0", `${shape(w1T)} + bias`, w1T.shape[0] * CELL, { size: 0.22 });
  withLabel(w2, "net.2", `${shape(w2T)} + bias`, w2T.shape[0] * CELL, { size: 0.22 });
  ffwd.add(w1, fb1, relu, w2, fb2);
  const ffwdW = w1W + 1.4 + w2W + 0.3;
  ffwd.children.forEach((c) => (c.position.x -= ffwdW / 2 - w1W / 2));
  sub(`${p}.ffwd`, ffwd, ffwdW, `Blocco ${layer} · feed-forward ${cfg.n_embd}→${4 * cfg.n_embd}→${cfg.n_embd}`);

  const add2 = addNode();
  sub(`${p}.add2`, add2, 0.56, `Blocco ${layer} · x = x + feed-forward`, SLOT.add);

  // Sfondo del blocco e archi delle connessioni residue
  const endX = layout.x - GAP;
  const top = (cfg.n_head * spacing) / 2 + 0.9;
  const ffTop = (w1T.shape[0] * CELL) / 2 + 0.9;
  const height = Math.max(top, ffTop) * 2 + 0.6;
  const frame = new THREE.Mesh(
    new THREE.BoxGeometry(endX - startX + 0.8, height, 2.2),
    new THREE.MeshBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0.18, depthWrite: false }),
  );
  frame.position.set((endX - startX) / 2, 0, -0.2);
  frame.userData.isFrame = true;
  block.add(frame);
  const frameTag = label(`Block ${layer}`, { sub: `${(spec.params.filter((x) => x.name.startsWith(`${p}.`)).reduce((a, x) => a + x.numel, 0)).toLocaleString("it-IT")} pesi`, size: 0.4, color: "#7dd3fc" });
  frameTag.position.set((endX - startX) / 2, height / 2 + 0.7, 0);
  block.add(frameTag);

  const arcY = Math.max(top, ffTop) + 0.2;
  block.add(residualArc(0, add1.position.x, arcY * 0.62));
  block.add(residualArc(add1.position.x, add2.position.x, arcY));
}

/** Arco della connessione residua da x0 a x1 (coordinate del blocco), usato anche dal LOD. */
export function residualArc(x0, x1, height, { color = 0x0ea5e9, opacity = 0.55, radius = 0.03, z = 0.6 } = {}) {
  const curve = new THREE.CubicBezierCurve3(
    new THREE.Vector3(x0, 0, z),
    new THREE.Vector3(x0, height, z),
    new THREE.Vector3(x1, height, z),
    new THREE.Vector3(x1, 0, z),
  );
  return new THREE.Mesh(
    new THREE.TubeGeometry(curve, 48, radius, 8, false),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity }),
  );
}

function buildOutput(spec, layout) {
  const cfg = spec.config;
  const lnf = new THREE.Group();
  const g = vectorBar(spec, "ln_f.weight");
  const b = vectorBar(spec, "ln_f.bias");
  b.position.x = 0.22;
  lnf.add(g, b);
  withLabel(lnf, "ln_f", `γ, β (${cfg.n_embd},)`, cfg.n_embd * CELL, { size: 0.2 });
  layout.place("ln_f", lnf, 0.34, "LayerNorm finale", { slot: SLOT.ln_f });

  const head = new THREE.Group();
  const hwT = spec.tensors["lm_head.weight"];
  const hw = matrix(spec, "lm_head.weight");
  const hb = vectorBar(spec, "lm_head.bias");
  hb.position.x = (hwT.shape[1] * CELL) / 2 + 0.2;
  head.add(hw, hb);
  withLabel(head, "lm_head", `${shape(hwT)} + bias`, hwT.shape[0] * CELL);
  layout.place("lm_head", head, hwT.shape[1] * CELL + 0.3, "lm_head: un punteggio per carattere (logits)", { slot: SLOT.lm_head });

  // Probabilità del prossimo carattere (dal modello vero, sul prompt d'esempio)
  const probs = spec.example.next_token_probs;
  const bars = new THREE.Group();
  const step = 0.12;
  const maxLen = 5;
  const top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  probs.forEach((p, i) => {
    const len = Math.max(0.02, p * maxLen);
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(len, step * 0.75, 0.08),
      new THREE.MeshStandardMaterial({ color: i === top[0] ? 0x38bdf8 : 0x475569, emissive: i === top[0] ? 0x0369a1 : 0 }),
    );
    bar.position.set(len / 2, ((probs.length - 1) / 2 - i) * step, 0);
    bars.add(bar);
    if (top.slice(0, 5).includes(i)) {
      const ch = spec.vocab[i] === " " ? "␣" : spec.vocab[i] === "\n" ? "↵" : spec.vocab[i];
      const tag = label(`'${ch}' ${(p * 100).toFixed(1)}%`, { size: 0.16 });
      tag.position.set(len + 0.55, bar.position.y, 0);
      bars.add(tag);
    }
  });
  withLabel(bars, "softmax → probabilità", `${cfg.vocab_size} caratteri`, probs.length * step);
  bars.children.forEach((c) => {
    if (c.isSprite && c.position.x === 0) c.position.x = maxLen / 2;
  });
  layout.place("logits", bars, maxLen + 1, "softmax: probabilità del prossimo carattere", { slot: SLOT.logits });

  const best = top[0];
  const next = tokenCube(spec.vocab[best], 0.6, { color: 0x0c4a6e, emissive: 0x0369a1 });
  withLabel(next, "prossimo token", `'${spec.vocab[best]}' (${(probs[best] * 100).toFixed(1)}%)`, 0.6);
  layout.place("next_token", next, 0.6, "prossimo token → si aggiunge al contesto", { slot: SLOT.next });
}

/**
 * Percorso del flusso: passa per ogni stazione e, a fine riga, risale alla riga
 * successiva come il carrello di una macchina da scrivere.
 */
function flowPath(layout) {
  layout.root.updateWorldMatrix(true, true);
  const points = [];
  let previousRow = null;
  // Centro (y nel mondo) della riga: la stazione sta nella riga in cui è stata posizionata.
  // Nota: non si arrotonda p.y / ROW_SPACING, perché la pila è centrata (root.position.y = −37,5)
  // e l'arrotondamento spostava il binario di mezza riga, tra una riga e l'altra.
  const rowY = (row) => layout.root.position.y + row * ROW_SPACING;
  layout.stations.forEach((station) => {
    const p = station.object.getWorldPosition(new THREE.Vector3());
    p.z = 0.6;
    const row = station.row;
    if (previousRow !== null && row !== previousRow) {
      const last = points.at(-1);
      const midY = rowY(row - 0.5);
      points.push(new THREE.Vector3(last.x + 1.5, last.y, 0.6), new THREE.Vector3(last.x + 1.5, midY, 0.6));
      points.push(new THREE.Vector3(p.x - 1.5, midY, 0.6), new THREE.Vector3(p.x - 1.5, rowY(row), 0.6));
    }
    p.y = rowY(row); // il binario corre al centro della riga
    points.push(p);
    station.point = p;
    previousRow = row;
  });
  const path = new THREE.CurvePath();
  for (let i = 1; i < points.length; i++) path.add(new THREE.LineCurve3(points[i - 1], points[i]));

  // Distanza lungo il percorso di ogni stazione: serve all'animazione.
  const lengths = path.getCurveLengths();
  let pointIndex = 0;
  layout.stations.forEach((station) => {
    while (points[pointIndex] !== station.point) pointIndex++;
    station.distance = pointIndex === 0 ? 0 : lengths[pointIndex - 1];
  });
  return path;
}

export function buildArchitecture(spec) {
  const root = new THREE.Group();
  const layout = new Layout(root);
  const nLayer = spec.config.n_layer;

  layout.startRow(0);
  layout.place("tokens", buildTokens(spec), 0.3, "Token di input (B, T)", { slot: SLOT.tokens });
  buildEmbeddings(spec, layout);
  layout.endRow();

  for (let layer = 0; layer < nLayer; layer++) {
    layout.startRow(layer + 1);
    buildBlock(spec, layout, layer);
    layout.endRow();
  }

  layout.startRow(nLayer + 1);
  buildOutput(spec, layout);
  layout.endRow();

  // Gruppi "virtuali" per il volo della camera
  const blocks = new THREE.Group();
  blocks.userData.members = [...Array(nLayer).keys()].map((i) => layout.registry.get(`blocks.${i}`));
  layout.registry.set("blocks", blocks);
  const prediction = new THREE.Group();
  prediction.userData.members = [layout.registry.get("logits"), layout.registry.get("next_token")];
  layout.registry.set("prediction", prediction);
  const overview = new THREE.Group();
  overview.userData.members = [root];
  layout.registry.set("overview", overview);

  // Centriamo la pila in verticale
  root.position.y = -((nLayer + 1) * ROW_SPACING) / 2;

  const path = flowPath(layout);
  const rail = new THREE.Mesh(
    new THREE.TubeGeometry(path, 600, 0.02, 6, false),
    new THREE.MeshBasicMaterial({ color: 0x155e75 }),
  );
  rail.userData.isFrame = true; // non conta per l'inquadratura
  // Ogni mesh di pesi è rintracciabile con il nome PyTorch del tensore (es. "blocks.0.ffwd.net.0.weight"):
  // serve a chi aggiorna le heatmap (snapshot del training) o le evidenzia (vista Pesi).
  const tensorMeshes = new Map();
  root.traverse((o) => {
    const name = o.userData?.tensorName;
    if (!name) return;
    if (!tensorMeshes.has(name)) tensorMeshes.set(name, []);
    tensorMeshes.get(name).push(o);
  });
  return { root, rail, registry: layout.registry, stations: layout.stations, path, tensorMeshes, slots: layout.slots, rows: layout.rows, rowSpacing: ROW_SPACING };
}
