/**
 * Stazione 7: l'indice HNSW di pgvector come grafo a livelli in 3D, più la nuvola dei vettori.
 *
 *  - ogni livello è un piano orizzontale; un nodo compare nel suo livello e in tutti quelli
 *    sotto (stessa posizione x, z, collegate da un filo verticale);
 *  - la posizione nel piano è la proiezione PCA dell'embedding: chunk simili stanno vicini;
 *  - archi di ogni livello in un unico LineSegments, nodi in un unico InstancedMesh: si
 *    aggiornano i buffer solo quando cambia lo stato (nodo inserito, passo della ricerca).
 *
 * Accanto, la "nuvola": gli stessi vettori proiettati in 3 dimensioni (PCA), colorati per
 * documento, con la domanda dell'utente quando parte la ricerca.
 */
import * as THREE from "three";
import { fmt } from "../../core/dom.js";
import { label } from "../../core/visuals.js";
import { Board, panel, paragraph } from "./board.js";
import { DOC_COLORS, makeStation } from "./scene3d.js";

const LW = 28; // larghezza dei piani (x)
const LD = 17; // profondità dei piani (z)
const STACK_X = -13;
const CLOUD_X = 19;
const CLOUD_R = 8;

const C = {
  visited: new THREE.Color("#64748b"),
  candidate: new THREE.Color("#facc15"),
  found: new THREE.Color("#4ade80"),
  current: new THREE.Color("#ffffff"),
  result: new THREE.Color("#22c55e"),
  edge: [new THREE.Color("#334155"), new THREE.Color("#475569"), new THREE.Color("#64748b"), new THREE.Color("#94a3b8")],
  edgeNew: new THREE.Color("#38bdf8"),
  path: new THREE.Color("#fde047"),
  dim: new THREE.Color("#1e293b"),
};

export function buildHnsw(kb) {
  const st = makeStation("hnsw");
  const { inserts, points, graphStates, maxLevel } = kb;
  const n = inserts.length;
  const gap = maxLevel > 0 ? Math.min(6, 19 / maxLevel) : 0;
  const baseY = -8.5;
  const docColors = DOC_COLORS.map((c) => new THREE.Color(c));

  // coordinate nel piano: PCA (componenti 0 e 1) normalizzate per riempire il piano
  const maxX = Math.max(...points.map((p) => Math.abs(p[0])), 1e-6);
  const maxZ = Math.max(...points.map((p) => Math.abs(p[1])), 1e-6);
  const planeXZ = points.map((p) => [STACK_X + (p[0] / maxX) * LW * 0.43, (p[1] / maxZ) * LD * 0.42]);
  const layerY = (l) => baseY + l * gap;
  const project = (xz, l, out) => out.set(xz[0], layerY(l), xz[1]);

  // --- piani
  const planes = new THREE.Group();
  for (let l = 0; l <= maxLevel; l++) {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(LW, LD),
      new THREE.MeshBasicMaterial({ color: l === 0 ? 0x0c4a6e : 0x1e3a5f, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false }),
    );
    plane.rotation.x = -Math.PI / 2;
    plane.position.set(STACK_X, layerY(l) - 0.05, 0);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(plane.geometry), new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.5 }));
    edge.rotation.copy(plane.rotation);
    edge.position.copy(plane.position);
    const tag = label(`livello ${l}${l === 0 ? " (tutti i nodi)" : ""}`, { size: 0.9, color: "#bae6fd" });
    tag.position.set(STACK_X - LW / 2 - 0.4, layerY(l) + 0.6, LD / 2);
    tag.center.set(1, 0.5);
    planes.add(plane, edge, tag);
  }
  st.group.add(planes);

  // --- nodi (una copia per livello)
  const copyStart = [];
  let copies = 0;
  for (const r of inserts) {
    copyStart.push(copies);
    copies += r.level + 1;
  }
  copyStart.push(copies);
  const nodes = new THREE.InstancedMesh(new THREE.SphereGeometry(0.42, 16, 12), new THREE.MeshBasicMaterial({ transparent: true }), copies);
  nodes.userData.noBounds = true;
  const dummy = new THREE.Object3D();
  const v = new THREE.Vector3();
  const w = new THREE.Vector3();
  inserts.forEach((r, i) => {
    for (let l = 0; l <= r.level; l++) {
      project(planeXZ[i], l, dummy.position);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      nodes.setMatrixAt(copyStart[i] + l, dummy.matrix);
      nodes.setColorAt(copyStart[i] + l, docColors[r.upload % docColors.length]);
    }
  });
  st.group.add(nodes);

  // --- fili verticali tra le copie dello stesso nodo
  const vertical = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0.45 }));
  {
    const pos = [];
    inserts.forEach((r, i) => {
      for (let l = 0; l < r.level; l++) {
        project(planeXZ[i], l, v);
        project(planeXZ[i], l + 1, w);
        pos.push(v.x, v.y, v.z, w.x, w.y, w.z);
      }
    });
    vertical.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  }
  const verticalStart = []; // numero di vertici dei fili per i primi k nodi
  {
    let acc = 0;
    for (const r of inserts) {
      verticalStart.push(acc);
      acc += r.level * 2;
    }
    verticalStart.push(acc);
  }
  st.group.add(vertical);

  // --- archi (buffer preallocato per lo stato più grande)
  const maxEdges = Math.max(...graphStates.map((s) => s.layers.reduce((sum, e) => sum + e.length, 0)), 1);
  const edgePos = new Float32Array(maxEdges * 6);
  const edgeCol = new Float32Array(maxEdges * 6);
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));
  edgeGeo.setAttribute("color", new THREE.BufferAttribute(edgeCol, 3));
  const edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
  edges.frustumCulled = false;
  st.group.add(edges);

  // --- percorso della ricerca e domanda
  const MAX_PATH = 96;
  const pathPos = new Float32Array(MAX_PATH * 6);
  const pathGeo = new THREE.BufferGeometry();
  pathGeo.setAttribute("position", new THREE.BufferAttribute(pathPos, 3));
  const path = new THREE.LineSegments(pathGeo, new THREE.LineBasicMaterial({ color: C.path, transparent: true }));
  path.frustumCulled = false;
  const query = new THREE.Mesh(new THREE.OctahedronGeometry(0.85), new THREE.MeshBasicMaterial({ color: 0xfde047, transparent: true }));
  query.userData.noBounds = true;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.85, 0.09, 8, 32), new THREE.MeshBasicMaterial({ color: 0xf472b6, transparent: true }));
  ring.rotation.x = Math.PI / 2;
  ring.userData.noBounds = true;
  st.group.add(path, query, ring);

  // --- nuvola 3D
  const cloud = new THREE.Group();
  cloud.position.set(CLOUD_X, 0, 0);
  const cloudPts = new THREE.InstancedMesh(new THREE.SphereGeometry(0.34, 12, 8), new THREE.MeshBasicMaterial({ transparent: true }), n);
  cloudPts.userData.noBounds = true;
  inserts.forEach((r, i) => {
    dummy.position.set(points[i][0] * CLOUD_R, points[i][2] * CLOUD_R, points[i][1] * CLOUD_R);
    dummy.updateMatrix();
    cloudPts.setMatrixAt(i, dummy.matrix);
    cloudPts.setColorAt(i, docColors[r.upload % docColors.length]);
  });
  const sphere = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(CLOUD_R * 1.05, 1)),
    new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.35 }),
  );
  const cloudQuery = new THREE.Mesh(new THREE.OctahedronGeometry(0.6), new THREE.MeshBasicMaterial({ color: 0xfde047, transparent: true }));
  const linkPos = new Float32Array(16 * 6);
  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute("position", new THREE.BufferAttribute(linkPos, 3));
  const links = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: 0x4ade80, transparent: true }));
  links.frustumCulled = false;
  const explained = kb.projection.explainedVariance.reduce((s, x) => s + x, 0);
  const cloudTitle = label("Nuvola dei vettori (PCA 3D)", { size: 1.0, color: "#e2e8f0", sub: `3 direzioni su ${kb.embedder.dimensions}: ${fmt(explained * 100)}% della varianza` });
  cloudTitle.position.set(0, CLOUD_R + 2.4, 0);
  cloud.add(cloudPts, sphere, cloudQuery, links, cloudTitle);
  st.group.add(cloud);

  // --- lavagna con le spiegazioni del passo corrente
  // inclinata verso la camera, che guarda la pila dall'alto
  const info = new Board({ width: 32, height: 6.6, ppu: 50 });
  info.mesh.position.set(STACK_X + 2, layerY(maxLevel) + 6.2, -LD / 2 + 1);
  info.mesh.rotation.x = -0.45;
  st.group.add(info.mesh);
  st.title.position.set(STACK_X + 2, layerY(maxLevel) + 11.4, -LD / 2 + 1);

  // ----------------------------------------------------------------- stato del grafo

  let shown = ""; // chiave dello stato disegnato: i buffer si riscrivono solo se cambia
  function writeEdges(k, newNode, fade = 1) {
    const layers = graphStates[k].layers;
    let o = 0;
    layers.forEach((list, l) => {
      const base = C.edge[Math.min(l, C.edge.length - 1)];
      for (const [a, b] of list) {
        project(planeXZ[a], l, v);
        project(planeXZ[b], l, w);
        const c = a === newNode || b === newNode ? C.edgeNew : base;
        edgePos[o] = v.x;
        edgePos[o + 1] = v.y;
        edgePos[o + 2] = v.z;
        edgePos[o + 3] = w.x;
        edgePos[o + 4] = w.y;
        edgePos[o + 5] = w.z;
        edgeCol[o] = edgeCol[o + 3] = c.r * fade;
        edgeCol[o + 1] = edgeCol[o + 4] = c.g * fade;
        edgeCol[o + 2] = edgeCol[o + 5] = c.b * fade;
        o += 6;
      }
    });
    edgeGeo.setDrawRange(0, o / 3);
    edgeGeo.attributes.position.needsUpdate = true;
    edgeGeo.attributes.color.needsUpdate = true;
  }

  function resetNodeColors(k) {
    for (let i = 0; i < k; i++) {
      for (let l = 0; l <= inserts[i].level; l++) {
        nodes.setColorAt(copyStart[i] + l, docColors[inserts[i].upload % docColors.length]);
        setScale(copyStart[i] + l, i, l, 1);
      }
    }
    nodes.instanceColor.needsUpdate = true;
    nodes.instanceMatrix.needsUpdate = true;
  }

  function setScale(copy, node, l, s) {
    project(planeXZ[node], l, dummy.position);
    dummy.scale.setScalar(s);
    dummy.updateMatrix();
    nodes.setMatrixAt(copy, dummy.matrix);
  }

  /** Entry point dopo k inserimenti: il primo nodo con il livello massimo. */
  function entryAfter(k) {
    let ep = -1;
    let top = -1;
    for (let i = 0; i < k; i++) {
      if (inserts[i].level > top) {
        top = inserts[i].level;
        ep = i;
      }
    }
    return { ep, top };
  }

  function placeRing(k) {
    const { ep, top } = entryAfter(k);
    ring.visible = ep >= 0;
    if (ep >= 0) {
      project(planeXZ[ep], top, ring.position);
    }
  }

  /**
   * Mostra i primi k nodi (k = 0: indice vuoto). Con newest = true il nodo k-1 e i suoi archi
   * sono evidenziati (inserimento appena avvenuto).
   */
  function setGraph(k, { newest = false } = {}) {
    const key = `g${k}${newest}`;
    if (key === shown) return;
    shown = key;
    nodes.count = copyStart[k];
    vertical.geometry.setDrawRange(0, verticalStart[k]);
    writeEdges(k, newest ? k - 1 : -1);
    resetNodeColors(k);
    if (newest && k > 0) {
      for (let l = 0; l <= inserts[k - 1].level; l++) {
        nodes.setColorAt(copyStart[k - 1] + l, C.current);
        setScale(copyStart[k - 1] + l, k - 1, l, 1.6);
      }
    }
    placeRing(k);
    cloudPts.count = k;
    setSearchOverlay(null);
  }

  /** Animazione d'ingresso del nodo k-1: le sue copie crescono da un punto (t ∈ [0, 1]). */
  function popNewest(k, t) {
    if (k <= 0) return;
    const s = 0.05 + 1.55 * (1 - Math.pow(1 - Math.min(1, t), 3));
    for (let l = 0; l <= inserts[k - 1].level; l++) setScale(copyStart[k - 1] + l, k - 1, l, s);
    nodes.instanceMatrix.needsUpdate = true;
  }

  function setSearchOverlay(run) {
    query.visible = !!run;
    cloudQuery.visible = !!run;
    path.visible = !!run;
    links.visible = false;
    if (!run) pathGeo.setDrawRange(0, 0);
  }

  /** Testo della lavagna durante l'inserimento del nodo k-1. */
  function insertInfo(k) {
    const r = inserts[k - 1];
    const start = r.events.find((e) => e.type === "insert-start");
    const end = r.events.find((e) => e.type === "insert-end");
    const connects = r.events.filter((e) => e.type === "connect");
    const prunes = r.events.filter((e) => e.type === "prune");
    const m = r.entry.metadata;
    info.redraw(`I${k}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, { title: `Inserimento del nodo ${k - 1}: ${m.original_filename}, pag. ${m.page_number}`, border: "#38bdf8" });
      const lines = [];
      lines.push(`Livello estratto a caso: ⌊−ln(U) · mL⌋ = ${r.level}${start.entryPoint < 0 ? " · indice vuoto: diventa il punto di ingresso" : ""}.`);
      if (connects.length) lines.push(`Collegato a: ${connects.map((c) => `livello ${c.layer} → ${c.neighbors.length} vicini`).join(", ")}${prunes.length ? ` · ${prunes.length} vicini troppo connessi potati` : ""}.`);
      if (end.entryPointChanged && start.entryPoint >= 0) lines.push("Ha il livello più alto: è il nuovo punto di ingresso (anello rosa).");
      paragraph(ctx, lines.join("\n"), box.x, box.y, box.w, { px: 30, color: "#e2e8f0", maxLines: 4 });
    });
  }

  function genericInfo(k) {
    info.redraw(`G${k}`, (ctx, W, H) => {
      const box = panel(ctx, W, H, { title: k ? `Indice HNSW: ${k} nodi, ${maxLevel + 1} ${maxLevel ? "livelli" : "livello"}` : "Indice HNSW vuoto", border: "#38bdf8" });
      const { M, efConstruction, efSearch } = kb.config.hnswPreset.options;
      paragraph(ctx, `m = ${M} vicini per nodo (${2 * M} al livello 0), ef_construction = ${efConstruction}, ef_search = ${efSearch}. Colori = documenti; anello rosa = punto di ingresso.`, box.x, box.y, box.w, { px: 30, color: "#cbd5e1", maxLines: 3 });
    });
  }

  // ----------------------------------------------------------------- ricerca

  /**
   * Disegna il fotogramma f della ricerca `run` (vedi pipeline.searchFrames).
   * @param {object} run risultato di runSearch
   * @param {number} f indice del fotogramma
   * @param {{t?: number, showResults?: boolean}} [o] t ∈ [0,1] per il volo della domanda verso il nodo corrente
   */
  function setSearch(run, f, { showResults = false } = {}) {
    const frame = run.frames[Math.max(0, Math.min(f, run.frames.length - 1))];
    const key = `S${run.question}|${f}|${showResults}`;
    query.visible = true;
    cloudQuery.visible = true;
    path.visible = true;
    // domanda: nel piano del livello corrente, alla sua posizione proiettata
    const qx = STACK_X + (run.point[0] / maxX) * LW * 0.43;
    const qz = (run.point[1] / maxZ) * LD * 0.42;
    query.position.set(qx, layerY(frame.layer) + 1.4, qz);
    cloudQuery.position.set(run.point[0] * CLOUD_R, run.point[2] * CLOUD_R, run.point[1] * CLOUD_R);
    if (shown === key) return frame;
    shown = key;
    nodes.count = copyStart[n];
    vertical.geometry.setDrawRange(0, verticalStart[n]);
    cloudPts.count = n;
    placeRing(n);

    // colori dei nodi: prima tutti spenti, poi visitati, candidati, trovati, corrente
    for (let i = 0; i < n; i++) {
      for (let l = 0; l <= inserts[i].level; l++) {
        nodes.setColorAt(copyStart[i] + l, C.dim);
        setScale(copyStart[i] + l, i, l, 0.8);
      }
    }
    const paint = (node, color, layer, s = 1.15) => {
      if (inserts[node].level < layer) return;
      nodes.setColorAt(copyStart[node] + layer, color);
      setScale(copyStart[node] + layer, node, layer, s);
    };
    const layer = frame.layer;
    for (const node of frame.visited) for (let l = 0; l <= inserts[node].level; l++) paint(node, C.visited, l, 1);
    for (const node of frame.candidates) paint(node, C.candidate, layer, 1.2);
    for (const fnd of frame.found) paint(fnd.index, C.found, layer, 1.25);
    if (showResults) {
      run.results.forEach((r) => {
        nodes.setColorAt(copyStart[r.node], C.result);
        setScale(copyStart[r.node], r.node, 0, 1.7);
      });
    }
    paint(frame.current, C.current, layer, 1.7);
    nodes.instanceColor.needsUpdate = true;
    nodes.instanceMatrix.needsUpdate = true;
    // archi del grafo attenuati, percorso in giallo
    writeEdges(n, -1, 0.55);
    let o = 0;
    for (const [l, a, b] of frame.path.slice(-MAX_PATH)) {
      project(planeXZ[a], l, v);
      project(planeXZ[b], l, w);
      v.y += 0.08;
      w.y += 0.08;
      v.toArray(pathPos, o);
      w.toArray(pathPos, o + 3);
      o += 6;
    }
    pathGeo.setDrawRange(0, o / 3);
    pathGeo.attributes.position.needsUpdate = true;
    // nuvola: collegamenti domanda → risultati
    links.visible = showResults;
    if (showResults) {
      let q = 0;
      for (const r of run.results.slice(0, 16)) {
        const p = points[r.node];
        linkPos.set([cloudQuery.position.x, cloudQuery.position.y, cloudQuery.position.z, p[0] * CLOUD_R, p[2] * CLOUD_R, p[1] * CLOUD_R], q);
        q += 6;
      }
      linkGeo.setDrawRange(0, q / 3);
      linkGeo.attributes.position.needsUpdate = true;
    }
    info.redraw(key, (ctx, W, H) => {
      const titles = { start: "Partenza dal punto di ingresso", layer: `Livello ${layer}`, step: `Livello ${layer}: esploro i vicini del nodo ${frame.current}`, end: `Fine del livello ${layer}`, done: "Ricerca finita" };
      const box = panel(ctx, W, H, { title: titles[frame.kind], border: "#fde047", accent: "#fde047" });
      const best = frame.found
        .slice(0, 5)
        .map((x) => `${x.index} (${(1 - x.distance).toFixed(2)})`)
        .join(", ");
      const mode = layer > 0 ? "discesa golosa: ef = 1, tengo solo il nodo più vicino" : `ricerca a fascio con ef = ${run.frames[0].ef}`;
      paragraph(ctx, `${mode}. Calcoli di distanza finora: ${frame.comparisons} (ricerca esatta: ${n}).\nMigliori (nodo e score): ${best || "—"}`, box.x, box.y, box.w, { px: 29, color: "#e2e8f0", maxLines: 3 });
    });
    return frame;
  }

  // registro per la camera
  const parts = {
    stack: makeFocusBox(STACK_X + 1, (baseY - 1 + layerY(maxLevel) + 10) / 2, LW + 4, layerY(maxLevel) + 10 - baseY + 1, LD),
    cloud: makeFocusBox(CLOUD_X, 0, CLOUD_R * 2.2, CLOUD_R * 2.2 + 3, CLOUD_R * 2.2),
  };
  st.group.add(parts.stack, parts.cloud);

  setGraph(0);
  genericInfo(0);
  return { ...st, setGraph, popNewest, setSearch, insertInfo, genericInfo, parts, layerCount: maxLevel + 1 };
}

function makeFocusBox(x, y, w, h, d) {
  const box = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshBasicMaterial({ visible: false }));
  box.visible = false;
  box.position.set(x, y, 0);
  return box;
}
