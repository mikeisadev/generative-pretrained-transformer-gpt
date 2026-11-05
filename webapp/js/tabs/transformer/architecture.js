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

const ROW_SPACING = 15;

class Layout {
  constructor(root) {
    this.root = root;
    this.registry = new Map();
    this.stations = [];
    this.rows = [];
  }

  /** Apre una nuova riga all'altezza `index * ROW_SPACING`. */
  startRow(index) {
    this.row = new THREE.Group();
    this.row.position.y = index * ROW_SPACING;
    this.root.add(this.row);
    this.x = 0;
  }

  /** Chiude la riga centrandola sull'asse x. */
  endRow() {
    const width = this.x - GAP;
    this.row.position.x = -width / 2;
    this.rows.push({ group: this.row, width });
  }

  /** Posiziona `obj` largo `width` alla posizione corrente della riga e avanza. */
  place(id, obj, width, caption, { parent = this.row } = {}) {
    obj.position.x = this.x + width / 2 - (parent === this.row ? 0 : parent.position.x);
    parent.add(obj);
    this.registry.set(id, obj);
    if (caption) this.stations.push({ id, caption, object: obj });
    this.x += width + GAP;
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
  layout.place("embeddings", group, width, "Embedding: token + posizione");
  layout.registry.set("tok_emb", tok);
  layout.registry.set("pos_emb", pos);
  layout.place("embed_add", addNode(), 0.56, "x = tok_emb + pos_emb   (B, T, C)");
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

  const sub = (id, obj, width, caption) => {
    layout.place(id, obj, width, caption, { parent: block });
  };

  // LayerNorm 1: γ e β
  const ln1 = new THREE.Group();
  const g1 = vectorBar(spec, `${p}.ln1.weight`);
  const b1 = vectorBar(spec, `${p}.ln1.bias`);
  b1.position.x = 0.22;
  ln1.add(g1, b1);
  withLabel(ln1, "ln1", "γ, β (64,)".replace("64", cfg.n_embd), cfg.n_embd * CELL, { size: 0.2 });
  sub(`${p}.ln1`, ln1, 0.34, `Blocco ${layer} · LayerNorm 1`);

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
  sub(`${p}.heads`, heads, headsWidth, `Blocco ${layer} · ${cfg.n_head} head di self-attention`);

  // Proiezione
  const proj = new THREE.Group();
  const pw = matrix(spec, `${p}.sa.proj.weight`);
  const pb = vectorBar(spec, `${p}.sa.proj.bias`);
  pb.position.x = (cfg.n_embd * CELL) / 2 + 0.2;
  proj.add(pw, pb);
  withLabel(proj, "proj", `${shape(spec.tensors[`${p}.sa.proj.weight`])} + bias`, cfg.n_embd * CELL, { size: 0.22 });
  sub(`${p}.proj`, proj, cfg.n_embd * CELL + 0.3, `Blocco ${layer} · proiezione`);

  const add1 = addNode();
  sub(`${p}.add1`, add1, 0.56, `Blocco ${layer} · x = x + attention`);

  const ln2 = new THREE.Group();
  const g2 = vectorBar(spec, `${p}.ln2.weight`);
  const b2 = vectorBar(spec, `${p}.ln2.bias`);
  b2.position.x = 0.22;
  ln2.add(g2, b2);
  withLabel(ln2, "ln2", `γ, β (${cfg.n_embd},)`, cfg.n_embd * CELL, { size: 0.2 });
  sub(`${p}.ln2`, ln2, 0.34, `Blocco ${layer} · LayerNorm 2`);

  // Feed-forward: espansione 4C, ReLU, compressione
  const ffwd = new THREE.Group();
  const w1T = spec.tensors[`${p}.ffwd.net.0.weight`];
  const w2T = spec.tensors[`${p}.ffwd.net.2.weight`];
  const w1 = matrix(spec, `${p}.ffwd.net.0.weight`);
  const w1W = w1T.shape[1] * CELL;
  const relu = label("ReLU", { size: 0.22, color: "#a5f3fc" });
  relu.position.x = w1W / 2 + 0.6;
  const w2 = matrix(spec, `${p}.ffwd.net.2.weight`);
  const w2W = w2T.shape[1] * CELL;
  w2.position.x = w1W / 2 + 1.2 + w2W / 2;
  withLabel(w1, "net.0", `${shape(w1T)} + bias`, w1T.shape[0] * CELL, { size: 0.22 });
  withLabel(w2, "net.2", `${shape(w2T)} + bias`, w2T.shape[0] * CELL, { size: 0.22 });
  ffwd.add(w1, relu, w2);
  const ffwdW = w1W + 1.2 + w2W;
  ffwd.children.forEach((c) => (c.position.x -= ffwdW / 2 - w1W / 2));
  sub(`${p}.ffwd`, ffwd, ffwdW, `Blocco ${layer} · feed-forward ${cfg.n_embd}→${4 * cfg.n_embd}→${cfg.n_embd}`);

  const add2 = addNode();
  sub(`${p}.add2`, add2, 0.56, `Blocco ${layer} · x = x + feed-forward`);

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

function residualArc(x0, x1, height) {
  const curve = new THREE.CubicBezierCurve3(
    new THREE.Vector3(x0, 0, 0.6),
    new THREE.Vector3(x0, height, 0.6),
    new THREE.Vector3(x1, height, 0.6),
    new THREE.Vector3(x1, 0, 0.6),
  );
  return new THREE.Mesh(
    new THREE.TubeGeometry(curve, 48, 0.03, 8, false),
    new THREE.MeshBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.55 }),
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
  layout.place("ln_f", lnf, 0.34, "LayerNorm finale");

  const head = new THREE.Group();
  const hwT = spec.tensors["lm_head.weight"];
  const hw = matrix(spec, "lm_head.weight");
  const hb = vectorBar(spec, "lm_head.bias");
  hb.position.x = (hwT.shape[1] * CELL) / 2 + 0.2;
  head.add(hw, hb);
  withLabel(head, "lm_head", `${shape(hwT)} + bias`, hwT.shape[0] * CELL);
  layout.place("lm_head", head, hwT.shape[1] * CELL + 0.3, "lm_head: un punteggio per carattere (logits)");

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
  layout.place("logits", bars, maxLen + 1, "softmax: probabilità del prossimo carattere");

  const best = top[0];
  const next = tokenCube(spec.vocab[best], 0.6, { color: 0x0c4a6e, emissive: 0x0369a1 });
  withLabel(next, "prossimo token", `'${spec.vocab[best]}' (${(probs[best] * 100).toFixed(1)}%)`, 0.6);
  layout.place("next_token", next, 0.6, "prossimo token → si aggiunge al contesto");
}

/**
 * Percorso del flusso: passa per ogni stazione e, a fine riga, risale alla riga
 * successiva come il carrello di una macchina da scrivere.
 */
function flowPath(layout) {
  layout.root.updateWorldMatrix(true, true);
  const points = [];
  let previousRow = null;
  layout.stations.forEach((station) => {
    const p = station.object.getWorldPosition(new THREE.Vector3());
    p.z = 0.6;
    const row = Math.round(p.y / ROW_SPACING);
    if (previousRow !== null && row !== previousRow) {
      const last = points.at(-1);
      const midY = (row - 0.5) * ROW_SPACING;
      points.push(new THREE.Vector3(last.x + 1.5, last.y, 0.6), new THREE.Vector3(last.x + 1.5, midY, 0.6));
      points.push(new THREE.Vector3(p.x - 1.5, midY, 0.6), new THREE.Vector3(p.x - 1.5, row * ROW_SPACING, 0.6));
    }
    p.y = row * ROW_SPACING; // il binario corre al centro della riga
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
  layout.place("tokens", buildTokens(spec), 0.3, "Token di input (B, T)");
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
  return { root, rail, registry: layout.registry, stations: layout.stations, path, tensorMeshes };
}
