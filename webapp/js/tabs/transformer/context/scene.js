/**
 * Scena 3D della vista "Contesto": i tre livelli dello zoom semantico, il pacchetto che corre
 * lungo il percorso e gli elementi animati del Dettaglio.
 *
 *   Dettaglio  → una scheda per fase con i dati veri (cards.js + painters.js), il vettore della
 *                domanda (InstancedMesh di barre), il grafo HNSW (InstancedMesh + LineSegments),
 *                i token che entrano nella pila (InstancedMesh), frecce con trattini che scorrono.
 *   Blocchi    → levels.js (un rettangolo per fase)
 *   Panoramica → levels.js (Contesto → LLM → Risposta)
 *
 * Stato dell'animazione (`state`, vedi FULL_STATE): ogni passo dell'azione lo imposta in enter()
 * e lo fa avanzare in update(); le schede si ridisegnano solo quando la loro parte cambia.
 * Nel ciclo per frame non si allocano oggetti: vettori, matrici e colori di lavoro sono riusati.
 */
import * as THREE from "three";
import { makeFlow } from "../../../core/diagram.js";
import { disposeTree, makeOutline } from "../views/lodBlocks.js";
import { Card } from "./cards.js";
import { SLOTS, connections } from "./layout.js";
import { buildBlocksLevel, buildMacroLevel } from "./levels.js";
import { PAINTERS } from "./painters.js";
import { contextSegments } from "./pipeline.js";
import { STAGES } from "./texts.js";

/** Stato "tutto visibile": quello della vista quando non c'è un'animazione in corso. */
export const FULL_STATE = Object.freeze({
  checks: 99, // controlli del guardrail mostrati
  embed: 1, // crescita delle barre del vettore (0..1)
  visited: 999, // nodi HNSW visitati
  retrieved: true, // chunk recuperati evidenziati nel grafo
  rows: 99, // righe della tabella dei chunk
  segments: 99, // parti del contesto mostrate
  call: 2, // chiamata all'LLM in corso o completata (0 = nessuna)
  tool: 3, // 0 nessuno, 1 richiesta, 2 metodo Java, 3 risultato
  streamed: 9999, // token della risposta arrivati a React (-1 = conversazione vuota)
});

const NODE = { idle: new THREE.Color("#475569"), visited: new THREE.Color("#38bdf8"), hit: new THREE.Color("#34d399") };
const EMBED_BARS = { pos: new THREE.Color("#fb923c"), neg: new THREE.Color("#3b82f6") };
const TOKEN_TILES = 120;
const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();

export class ContextScene {
  constructor(tab) {
    this.tab = tab;
    this.root = new THREE.Group(); // pacchetto e token: fuori dai livelli, visibili in Dettaglio e Blocchi
    this.root.name = "context-overlay";
    this.levels = { detail: new THREE.Group(), blocks: new THREE.Group(), macro: new THREE.Group() };
    this.levels.detail.name = "context-detail";
    this.state = { ...FULL_STATE };
    this.drawn = {}; // ultimo stato disegnato di barre e grafo
    this.scenario = null;
    this.flows = [];
    this.routes = new Map();
    const { spec } = tab;
    this.info = { layers: spec.config.n_layer, params: spec.total_params.toLocaleString("it-IT"), vocab: spec.config.vocab_size, block: spec.config.block_size };

    this.buildDetail();
    const blocks = buildBlocksLevel();
    this.levels.blocks.add(blocks.group);
    this.blockById = blocks.blocks;
    this.levels.macro.add(buildMacroLevel().group);
    this.buildOutlines();
    this.buildPacket();
    this.buildTokenTiles();
    tab.ui.host.scene.add(this.root);
    this.setVisible(false);
  }

  // ---------------------------------------------------------------- costruzione

  buildDetail() {
    const d = this.levels.detail;
    this.cards = new Map();
    for (const stage of STAGES) {
      const slot = SLOTS[stage.id];
      const card = new Card({ w: slot.w, h: slot.h, color: stage.color });
      card.mesh.position.set(slot.x, slot.y, 0);
      d.add(card.mesh);
      this.cards.set(stage.id, card);
    }
    for (const c of connections()) {
      const flow = makeFlow(c.points, { color: c.decorative ? "#7dd3fc" : "#67e8f9", radius: c.decorative ? 0.06 : 0.1, speed: 0.9 });
      flow.userData.noBounds = true;
      d.add(flow);
      this.flows.push(flow);
      if (!c.decorative) {
        const path = new THREE.CurvePath();
        for (let i = 1; i < c.points.length; i++) path.add(new THREE.LineCurve3(c.points[i - 1], c.points[i]));
        this.routes.set(c.id, path);
      }
    }

    // vettore della domanda: una barra per dimensione (256), nella metà bassa della scheda "embed"
    const e = SLOTS.embed;
    this.embedArea = { x0: e.x - e.w / 2 + 0.7, width: e.w - 1.4, y: e.y - 1.9, scale: 4.2 };
    this.embedBars = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 0.12), new THREE.MeshBasicMaterial({ transparent: true }), 256);
    this.embedBars.position.z = 0.15;
    this.embedBars.count = 0;
    d.add(this.embedBars);

    // grafo HNSW: nodi e archi del piano 0, posizioni = proiezione PCA dei vettori dei chunk
    const v = SLOTS.vectordb;
    this.graphArea = { x: v.x, y: v.y - 0.7, w: v.w - 2.4, h: v.h - 5 };
    this.nodes = new THREE.InstancedMesh(new THREE.SphereGeometry(0.34, 16, 12), new THREE.MeshBasicMaterial({ transparent: true }), 64);
    this.nodes.position.z = 0.3;
    this.nodes.count = 0;
    this.edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.45 }));
    this.edges.position.z = 0.2;
    this.query = new THREE.Mesh(new THREE.OctahedronGeometry(0.5), new THREE.MeshBasicMaterial({ color: 0xc084fc, transparent: true }));
    this.query.position.z = 0.4;
    this.queryLinks = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.8 }));
    this.queryLinks.position.z = 0.25;
    d.add(this.edges, this.nodes, this.query, this.queryLinks);
  }

  buildOutlines() {
    // un contorno per livello (materiali propri): evidenzia la fase attiva
    this.outlines = { detail: makeOutline(0x67e8f9), blocks: makeOutline(0x67e8f9) };
    this.levels.detail.add(this.outlines.detail);
    this.levels.blocks.add(this.outlines.blocks);
    this.active = null;
  }

  buildPacket() {
    const packet = new THREE.Group();
    packet.add(new THREE.Mesh(new THREE.SphereGeometry(0.42, 18, 12), new THREE.MeshBasicMaterial({ color: 0xe0f2fe })));
    const halo = new THREE.Mesh(new THREE.SphereGeometry(0.9, 18, 12), new THREE.MeshBasicMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.35, depthWrite: false }));
    packet.add(halo);
    packet.userData.noBounds = true;
    packet.traverse((o) => (o.userData.noBounds = true));
    packet.visible = false;
    this.halo = halo;
    this.packet = packet;
    this.packetRoute = [];
    this.root.add(packet);
  }

  buildTokenTiles() {
    this.tiles = new THREE.InstancedMesh(new THREE.BoxGeometry(0.75, 0.75, 0.75), new THREE.MeshBasicMaterial(), TOKEN_TILES);
    this.tiles.count = 0;
    this.tiles.frustumCulled = false;
    this.tiles.userData.noBounds = true;
    this.root.add(this.tiles);
    const c = connections().find((x) => x.id === "tokenize-llm").points;
    this.tileFrom = c[0].clone();
    this.tileTo = c[1].clone();
  }

  // ---------------------------------------------------------------- dati

  /** Nuovo scenario (domanda cambiata): ricalcola barre, grafo, colori dei token e schede. */
  setScenario(s) {
    this.scenario = s;
    const vec = s.embedding?.vector;
    this.embedBars.count = vec ? vec.length : 0;
    if (vec) for (let i = 0; i < vec.length; i++) this.embedBars.setColorAt(i, vec[i] >= 0 ? EMBED_BARS.pos : EMBED_BARS.neg);
    if (this.embedBars.instanceColor) this.embedBars.instanceColor.needsUpdate = true;

    const r = s.search;
    if (r) {
      const xs = r.points.map((p) => p[0]).concat(r.query[0]);
      const ys = r.points.map((p) => p[1]).concat(r.query[1]);
      const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const a = this.graphArea;
      const sx = a.w / Math.max(1e-6, maxX - minX);
      const sy = a.h / Math.max(1e-6, maxY - minY);
      const place = (p) => [a.x - a.w / 2 + (p[0] - minX) * sx, a.y - a.h / 2 + (p[1] - minY) * sy];
      this.nodePos = r.points.map(place);
      this.nodes.count = this.nodePos.length;
      this.nodePos.forEach(([x, y], i) => this.nodes.setMatrixAt(i, _m.makeTranslation(x, y, 0)));
      this.nodes.instanceMatrix.needsUpdate = true;
      const seg = new Float32Array(r.edges.length * 6);
      r.edges.forEach(([i, j], k) => seg.set([...this.nodePos[i], 0, ...this.nodePos[j], 0], k * 6));
      this.edges.geometry.dispose();
      this.edges.geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(seg, 3));
      const q = place(r.query);
      this.query.position.set(q[0], q[1], 0.4);
      const links = new Float32Array(r.results.length * 6);
      r.results.forEach((res, k) => links.set([q[0], q[1], 0, ...this.nodePos[res.node], 0], k * 6));
      this.queryLinks.geometry.dispose();
      this.queryLinks.geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(links, 3));
      this.hits = new Set(r.results.map((x) => x.node));
      this.visitRank = new Map(r.visitOrder.map((n, i) => [n, i]));
    } else {
      this.nodes.count = 0;
      this.nodePos = [];
      this.edges.geometry.setDrawRange(0, 0);
      this.queryLinks.geometry.setDrawRange(0, 0);
    }
    this.query.visible = !!r;

    // colori dei token: proporzionali ai token di ogni parte del contesto (1 cubetto ≈ N token)
    this.tileColors = [[], []];
    for (const call of [1, 2]) {
      const segs = s.prompt ? contextSegments(s, { includeTool: call === 2 }) : [];
      const total = segs.reduce((acc, g) => acc + g.tokens, 0) || 1;
      let k = 0;
      for (const g of segs) {
        const n = Math.round((g.tokens / total) * TOKEN_TILES);
        for (let i = 0; i < n && k < TOKEN_TILES; i++, k++) this.tileColors[call - 1].push(new THREE.Color(g.color));
      }
      while (this.tileColors[call - 1].length < TOKEN_TILES) this.tileColors[call - 1].push(new THREE.Color(segs.at(-1)?.color ?? "#67e8f9"));
    }
    for (const card of this.cards.values()) card.key = null; // forza il ridisegno
    this.drawn = {};
    this.applyState();
  }

  /** Aggiorna lo stato dell'animazione (anche parziale) e ridisegna ciò che è cambiato. */
  setState(partial) {
    // le update() dell'azione chiamano setState a ogni frame: se nulla cambia non si fa nulla
    let changed = false;
    for (const k in partial) if (this.state[k] !== partial[k]) changed = true;
    if (!changed) return;
    Object.assign(this.state, partial);
    this.applyState();
  }

  applyState() {
    const s = this.scenario;
    if (!s) return;
    const st = this.state;
    for (const [id, card] of this.cards) {
      const key = stateKey(id, st, s);
      card.draw(key, (pen) => PAINTERS[id](pen, s, st, this.info));
    }
    // barre e grafo si ricaricano sulla GPU solo se la loro parte di stato è cambiata
    if (this.drawn.embed !== st.embed) this.updateEmbedBars((this.drawn.embed = st.embed));
    const graphKey = `${Math.min(st.visited, 999)}:${st.retrieved}`;
    if (this.drawn.graph !== graphKey) {
      this.drawn.graph = graphKey;
      this.updateGraph(st.visited, st.retrieved);
    }
  }

  updateEmbedBars(progress) {
    const vec = this.scenario?.embedding?.vector;
    if (!vec) return;
    const a = this.embedArea;
    const w = a.width / vec.length;
    for (let i = 0; i < vec.length; i++) {
      // le barre crescono da sinistra a destra: ogni dimensione parte un po' dopo la precedente
      const local = THREE.MathUtils.clamp(progress * 1.6 - (i / vec.length) * 0.6, 0, 1);
      const h = Math.max(0.02, Math.abs(vec[i]) * a.scale * local);
      _p.set(a.x0 + (i + 0.5) * w, a.y + (vec[i] >= 0 ? h / 2 : -h / 2), 0);
      _s.set(w * 0.8, h, 1);
      this.embedBars.setMatrixAt(i, _m.compose(_p, _q.identity(), _s));
    }
    this.embedBars.instanceMatrix.needsUpdate = true;
  }

  updateGraph(visited, retrieved) {
    if (!this.nodePos?.length) return;
    for (let i = 0; i < this.nodePos.length; i++) {
      const rank = this.visitRank.get(i) ?? Infinity;
      const color = retrieved && this.hits.has(i) ? NODE.hit : rank < visited ? NODE.visited : NODE.idle;
      this.nodes.setColorAt(i, color);
    }
    this.nodes.instanceColor.needsUpdate = true;
    this.queryLinks.visible = retrieved;
  }

  // ---------------------------------------------------------------- fase attiva, pacchetto, token

  /** Evidenzia la fase (contorno pulsante in Dettaglio e Blocchi); null per nessuna. */
  setActive(id) {
    this.active = id;
    for (const [level, outline] of Object.entries(this.outlines)) {
      const slot = id ? SLOTS[id] : null;
      outline.visible = !!slot;
      if (!slot) continue;
      outline.position.set(slot.x, slot.y, level === "blocks" ? 0.4 : 0.1);
      outline.scale.set(slot.w + 0.6, slot.h + 0.6, 1);
    }
  }

  /** Imposta il tragitto del pacchetto: elenco di id di connections() percorsi uno dopo l'altro. */
  route(ids) {
    this.packetRoute = ids.map((id) => this.routes.get(id)).filter(Boolean);
    this.routeLengths = this.packetRoute.map((p) => p.getLength());
    this.routeTotal = this.routeLengths.reduce((a, b) => a + b, 0);
    this.packet.visible = this.packetRoute.length > 0;
    this.movePacket(0);
  }

  /** Porta il pacchetto alla frazione p (0..1) del tragitto. */
  movePacket(p) {
    if (!this.packetRoute.length) return;
    let d = THREE.MathUtils.clamp(p, 0, 1) * this.routeTotal;
    for (let i = 0; i < this.packetRoute.length; i++) {
      const len = this.routeLengths[i];
      if (d <= len || i === this.packetRoute.length - 1) {
        this.packetRoute[i].getPointAt(Math.min(1, d / Math.max(len, 1e-6)), this.packet.position);
        this.packet.position.z = 0.9;
        return;
      }
      d -= len;
    }
  }

  hidePacket() {
    this.packet.visible = false;
    this.packetRoute = [];
  }

  /**
   * Token che entrano nella pila: cubetti colorati per ruolo che corrono dalla scheda
   * "Tokenizzazione" ai token di input. p in 0..1; call = 1 o 2 (con richiesta e risultato).
   */
  flowTokens(p, call = 1) {
    if (p === null) {
      this.tiles.count = 0;
      return;
    }
    const colors = this.tileColors?.[call - 1];
    if (!colors) return;
    this.tiles.count = TOKEN_TILES;
    for (let i = 0; i < TOKEN_TILES; i++) {
      const start = (i / TOKEN_TILES) * 0.65; // partenze scaglionate
      const local = THREE.MathUtils.clamp((p - start) / 0.35, 0, 1);
      const visible = local > 0 && local < 1;
      _p.lerpVectors(this.tileFrom, this.tileTo, local);
      _p.y += Math.sin(local * Math.PI) * 1.2 * ((i % 5) - 2) * 0.35;
      _p.z = 0.8;
      const k = visible ? 1 : 0.0001;
      _s.set(k, k, k);
      this.tiles.setMatrixAt(i, _m.compose(_p, _q.identity(), _s));
      this.tiles.setColorAt(i, colors[i]);
    }
    this.tiles.instanceMatrix.needsUpdate = true;
    this.tiles.instanceColor.needsUpdate = true;
  }

  // ---------------------------------------------------------------- ciclo per frame e visibilità

  tick(dt, elapsed) {
    for (const f of this.flows) f.userData.tick(dt);
    const pulse = 0.6 + 0.4 * Math.sin(elapsed * 5);
    for (const o of Object.values(this.outlines)) if (o.visible) o.material.color.setRGB(0.4 * pulse, 0.91 * pulse, 0.98 * pulse);
    if (this.packet.visible) this.halo.scale.setScalar(1 + 0.25 * Math.sin(elapsed * 8));
    // pacchetto e token: solo dove ha senso (in Panoramica il percorso è riassunto dalle frecce)
    const macro = this.tab.lod?.current === "macro";
    this.root.visible = this.visible && !macro;
  }

  setVisible(v) {
    this.visible = v;
    for (const g of Object.values(this.levels)) g.visible = v;
    this.root.visible = v;
  }

  dispose() {
    for (const card of this.cards.values()) card.dispose();
    for (const g of Object.values(this.levels)) disposeTree(g);
    disposeTree(this.root);
    this.root.removeFromParent();
  }
}

/** Riassunto dello stato che interessa a una scheda: se non cambia, la scheda non si ridisegna. */
function stateKey(id, st, s) {
  const n = s.answer.tokens.length;
  switch (id) {
    case "react":
      return `${Math.min(st.streamed, n)}`;
    case "guard":
      return `${Math.min(st.checks, s.guard.checks.length)}`;
    case "vectordb":
      return `${Math.min(st.visited, s.search?.visitOrder.length ?? 0)}`;
    case "chunks":
      return `${Math.min(st.rows, 7)}`;
    case "prompt":
      return `${Math.min(st.segments, 99)}:${st.call}`;
    case "tokenize":
    case "llm":
      return `${st.call}`;
    case "toolcall":
    case "java":
    case "toolresult":
      return `${st.tool}`;
    case "stream":
      return `${Math.min(st.streamed, n)}`;
    default:
      return "static";
  }
}
