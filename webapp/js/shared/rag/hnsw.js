// HNSW (Hierarchical Navigable Small World), fedele all'articolo:
//   Yu. A. Malkov, D. A. Yashunin, "Efficient and robust approximate nearest neighbor search
//   using Hierarchical Navigable Small World graphs", arXiv:1603.09320 (2016),
//   IEEE TPAMI 42(4), 2020.
// I nomi delle funzioni ricalcano gli algoritmi dell'articolo:
//   Algoritmo 1 INSERT, 2 SEARCH-LAYER, 3 SELECT-NEIGHBORS-SIMPLE,
//   4 SELECT-NEIGHBORS-HEURISTIC, 5 K-NN-SEARCH.
//
// L'idea in breve: un grafo a più piani. Al piano 0 ci sono tutti i punti, ognuno collegato ai
// suoi vicini; ogni piano superiore contiene una frazione sempre più piccola di punti (scelti a
// caso, con probabilità che cala in modo esponenziale) e collegamenti più "lunghi". La ricerca
// parte dall'alto, dove pochi salti attraversano tutto lo spazio, e scende piano per piano
// affinando la posizione, come passare dalla cartina dell'Italia a quella della città.
//
// Parametri con i valori predefiniti di pgvector (l'indice usato nelle lezioni):
// m = 16, ef_construction = 64, hnsw.ef_search = 40.

import { createRng } from "./random.js";
import { cosineDistance } from "./similarity.js";

/** Valori predefiniti di pgvector per l'indice HNSW. */
export const PGVECTOR_HNSW_DEFAULTS = Object.freeze({ M: 16, efConstruction: 64, efSearch: 40 });

/**
 * Coda con priorità (heap binario). `compare(a, b) < 0` significa "a esce prima di b".
 * Serve all'algoritmo 2: un min-heap per i candidati da esplorare e un max-heap per i
 * migliori trovati finora (così il peggiore si scarta in O(log n)).
 */
class Heap {
  constructor(compare) {
    this.compare = compare;
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  peek() {
    return this.items[0];
  }

  push(item) {
    const items = this.items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(items[i], items[parent]) >= 0) break;
      [items[i], items[parent]] = [items[parent], items[i]];
      i = parent;
    }
  }

  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const left = 2 * i + 1;
        const right = left + 1;
        let best = i;
        if (left < items.length && this.compare(items[left], items[best]) < 0) best = left;
        if (right < items.length && this.compare(items[right], items[best]) < 0) best = right;
        if (best === i) break;
        [items[i], items[best]] = [items[best], items[i]];
        i = best;
      }
    }
    return top;
  }

  toArray() {
    return [...this.items];
  }
}

/**
 * Ordine "dal più vicino": distanza crescente, a parità di distanza indice crescente.
 * Lo spareggio sull'indice rende l'algoritmo deterministico anche con distanze uguali.
 */
const nearestFirst = (a, b) => a.distance - b.distance || a.index - b.index;
const furthestFirst = (a, b) => -nearestFirst(a, b);

/**
 * @typedef {object} Candidate
 * @property {number} index    posizione del nodo nell'indice
 * @property {number} distance distanza dal vettore cercato
 */

/**
 * @typedef {object} HnswNode
 * @property {number} index
 * @property {string|number} id   identificativo esterno (es. id del chunk)
 * @property {ArrayLike<number>} vector
 * @property {number} level       piano più alto in cui compare il nodo
 * @property {number[][]} neighbors neighbors[l] = vicini al piano l (per l = 0…level)
 */

/**
 * Eventi emessi per l'animazione (tutti hanno il campo `type`):
 * - "insert-start"  {node, level, entryPoint, maxLevel}
 * - "layer-start"   {phase, layer, ef, entryPoints}
 * - "layer-step"    {phase, layer, current, currentDistance, newlyVisited, candidates, found, visitedCount}
 * - "layer-end"     {phase, layer, visited, found}
 * - "connect"       {layer, node, neighbors}             archi creati da INSERT
 * - "prune"         {layer, node, before, after, removed} vicini potati di un nodo troppo connesso
 * - "insert-end"    {node, level, entryPoint, maxLevel, entryPointChanged}
 * - "search-start"  {k, ef, entryPoint, maxLevel}
 * - "search-end"    {results, visitedCount}
 * `phase` vale "insert" o "search"; `candidates` sono indici (coda C dell'articolo),
 * `found` sono {index, distance} (lista W), ordinati dal più vicino.
 * @typedef {{type: string} & Record<string, unknown>} HnswEvent
 */

/** Indice HNSW. */
export class HNSWIndex {
  /**
   * @param {object} [options]
   * @param {number} [options.M=16]                 connessioni create per ogni nuovo nodo (e massimo ai piani ≥ 1)
   * @param {number} [options.M0=2*M]               massimo di connessioni al piano 0 (l'articolo consiglia 2M)
   * @param {number} [options.efConstruction=64]    ampiezza della ricerca durante l'inserimento
   * @param {number} [options.efSearch=40]          ampiezza della ricerca durante le query
   * @param {number} [options.mL=1/ln(M)]           fattore di normalizzazione dei livelli
   * @param {number} [options.seed=42]              seed per l'estrazione dei livelli
   * @param {"heuristic"|"simple"} [options.neighborSelection="heuristic"] algoritmo 4 o 3
   * @param {boolean} [options.extendCandidates=false]       opzione dell'algoritmo 4
   * @param {boolean} [options.keepPrunedConnections=false]  opzione dell'algoritmo 4
   * @param {(a: ArrayLike<number>, b: ArrayLike<number>) => number} [options.distance=cosineDistance]
   */
  constructor(options = {}) {
    const M = options.M ?? PGVECTOR_HNSW_DEFAULTS.M;
    if (!Number.isInteger(M) || M < 2) throw new RangeError("M deve essere un intero >= 2");
    this.M = M;
    this.M0 = options.M0 ?? 2 * M;
    this.efConstruction = Math.max(options.efConstruction ?? PGVECTOR_HNSW_DEFAULTS.efConstruction, 1);
    this.efSearch = Math.max(options.efSearch ?? PGVECTOR_HNSW_DEFAULTS.efSearch, 1);
    this.mL = options.mL ?? 1 / Math.log(M);
    this.seed = options.seed ?? 42;
    this.neighborSelection = options.neighborSelection ?? "heuristic";
    if (!["heuristic", "simple"].includes(this.neighborSelection)) {
      throw new RangeError(`neighborSelection sconosciuto: ${this.neighborSelection}`);
    }
    this.extendCandidates = !!options.extendCandidates;
    this.keepPrunedConnections = !!options.keepPrunedConnections;
    this.distance = options.distance ?? cosineDistance;
    this.rng = createRng(this.seed);
    /** @type {HnswNode[]} */
    this.nodes = [];
    /** Nodo da cui partono tutte le ricerche (-1 se l'indice è vuoto). */
    this.entryPoint = -1;
    this.maxLevel = -1;
    this.dimensions = null;
  }

  get size() {
    return this.nodes.length;
  }

  /**
   * Livello casuale di un nuovo nodo: l = ⌊-ln(U) · mL⌋ con U uniforme in (0, 1].
   * Con mL = 1/ln(M) la probabilità di arrivare al piano l è M^(-l): ogni piano ha circa
   * M volte meno nodi di quello sotto.
   *
   * @returns {number}
   */
  randomLevel() {
    return Math.floor(-Math.log(1 - this.rng()) * this.mL);
  }

  /** Distanza tra due nodi già inseriti. */
  nodeDistance(a, b) {
    return this.distance(this.nodes[a].vector, this.nodes[b].vector);
  }

  /** Massimo di connessioni ammesso a un piano. */
  maxConnections(layer) {
    return layer === 0 ? this.M0 : this.M;
  }

  /**
   * Algoritmo 2, SEARCH-LAYER: ricerca "a fascio" con ampiezza `ef` in un solo piano.
   *
   * @param {ArrayLike<number>} query
   * @param {Candidate[]} entryPoints
   * @param {number} ef
   * @param {number} layer
   * @param {{phase: string, emit?: (e: HnswEvent) => void, visitedAll?: Set<number>}} context
   * @returns {Candidate[]} i fino a `ef` nodi più vicini trovati, dal più vicino
   */
  searchLayer(query, entryPoints, ef, layer, context) {
    const { emit, phase } = context;
    const visited = new Set(entryPoints.map((e) => e.index));
    const candidates = new Heap(nearestFirst); // C: da esplorare, il più vicino in cima
    const found = new Heap(furthestFirst); // W: i migliori finora, il peggiore in cima
    for (const entry of entryPoints) {
      candidates.push(entry);
      found.push(entry);
    }
    emit?.({ type: "layer-start", phase, layer, ef, entryPoints: entryPoints.map((e) => e.index) });

    while (candidates.size > 0) {
      const current = candidates.pop();
      const furthest = found.peek();
      // Se il candidato più vicino è più lontano del peggiore già trovato, nessun altro
      // candidato può migliorare W: la ricerca in questo piano è finita.
      if (current.distance > furthest.distance) break;
      const newlyVisited = [];
      for (const neighbor of this.nodes[current.index].neighbors[layer]) {
        if (visited.has(neighbor)) continue;
        visited.add(neighbor);
        newlyVisited.push(neighbor);
        const distance = this.distance(query, this.nodes[neighbor].vector);
        if (found.size < ef || distance < found.peek().distance) {
          const candidate = { index: neighbor, distance };
          candidates.push(candidate);
          found.push(candidate);
          if (found.size > ef) found.pop();
        }
      }
      if (emit) {
        emit({
          type: "layer-step",
          phase,
          layer,
          current: current.index,
          currentDistance: current.distance,
          newlyVisited,
          candidates: candidates.toArray().sort(nearestFirst).map((c) => c.index),
          found: found.toArray().sort(nearestFirst),
          visitedCount: visited.size,
        });
      }
    }
    const result = found.toArray().sort(nearestFirst);
    if (context.visitedAll) for (const v of visited) context.visitedAll.add(v);
    emit?.({ type: "layer-end", phase, layer, visited: [...visited], found: result });
    return result;
  }

  /**
   * Algoritmo 3, SELECT-NEIGHBORS-SIMPLE: gli `m` candidati più vicini.
   *
   * @param {Candidate[]} candidates
   * @param {number} m
   * @returns {Candidate[]}
   */
  selectNeighborsSimple(candidates, m) {
    return [...candidates].sort(nearestFirst).slice(0, m);
  }

  /**
   * Algoritmo 4, SELECT-NEIGHBORS-HEURISTIC: un candidato entra tra i vicini solo se è più
   * vicino al nodo base che a ogni vicino già scelto. Così i collegamenti puntano in direzioni
   * diverse (anche verso gruppi lontani) invece di ammassarsi tutti nello stesso grappolo.
   *
   * @param {ArrayLike<number>} base   vettore del nodo a cui scegliamo i vicini
   * @param {Candidate[]} candidates   distanze già calcolate rispetto a `base`
   * @param {number} m
   * @param {number} layer
   * @param {number} [exclude=-1]      nodo da non considerare (il nodo base stesso)
   * @returns {Candidate[]}
   */
  selectNeighborsHeuristic(base, candidates, m, layer, exclude = -1) {
    const working = new Heap(nearestFirst);
    const seen = new Set();
    for (const c of candidates) {
      if (c.index === exclude || seen.has(c.index)) continue;
      seen.add(c.index);
      working.push(c);
    }
    if (this.extendCandidates) {
      for (const c of candidates) {
        for (const adjacent of this.nodes[c.index].neighbors[layer] ?? []) {
          if (adjacent === exclude || seen.has(adjacent)) continue;
          seen.add(adjacent);
          working.push({ index: adjacent, distance: this.distance(base, this.nodes[adjacent].vector) });
        }
      }
    }
    const result = [];
    const discarded = new Heap(nearestFirst);
    while (working.size > 0 && result.length < m) {
      const e = working.pop();
      const closerToBase = result.every((r) => e.distance < this.nodeDistance(e.index, r.index));
      if (closerToBase) result.push(e);
      else discarded.push(e);
    }
    if (this.keepPrunedConnections) {
      while (discarded.size > 0 && result.length < m) result.push(discarded.pop());
    }
    return result;
  }

  /**
   * Sceglie i vicini con l'algoritmo configurato.
   *
   * @param {ArrayLike<number>} base
   * @param {Candidate[]} candidates
   * @param {number} m
   * @param {number} layer
   * @param {number} [exclude=-1]
   */
  selectNeighbors(base, candidates, m, layer, exclude = -1) {
    if (this.neighborSelection === "simple") {
      return this.selectNeighborsSimple(candidates.filter((c) => c.index !== exclude), m);
    }
    return this.selectNeighborsHeuristic(base, candidates, m, layer, exclude);
  }

  /**
   * Algoritmo 1, INSERT: aggiunge un vettore all'indice.
   *
   * @param {ArrayLike<number>} vector
   * @param {{id?: string|number, level?: number, onEvent?: (e: HnswEvent) => void}} [options]
   *   `level` forza il livello (utile nei test e nelle dimostrazioni); di norma è casuale
   * @returns {number} indice del nuovo nodo
   */
  insert(vector, { id, level: forcedLevel, onEvent } = {}) {
    if (this.dimensions === null) this.dimensions = vector.length;
    else if (vector.length !== this.dimensions) {
      throw new RangeError(`Dimensione ${vector.length} diversa da quella dell'indice (${this.dimensions})`);
    }
    // Il livello si estrae sempre, anche se forzato, così la sequenza casuale non dipende
    // dalle dimostrazioni fatte in mezzo.
    const drawn = this.randomLevel();
    const level = forcedLevel ?? drawn;
    const index = this.nodes.length;
    const node = { index, id: id ?? index, vector, level, neighbors: Array.from({ length: level + 1 }, () => []) };
    this.nodes.push(node);
    const emit = onEvent;
    emit?.({ type: "insert-start", node: index, level, entryPoint: this.entryPoint, maxLevel: this.maxLevel });

    if (this.entryPoint === -1) {
      this.entryPoint = index;
      this.maxLevel = level;
      emit?.({ type: "insert-end", node: index, level, entryPoint: index, maxLevel: level, entryPointChanged: true });
      return index;
    }

    const context = { phase: "insert", emit };
    let entryPoints = [{ index: this.entryPoint, distance: this.distance(vector, this.nodes[this.entryPoint].vector) }];
    // Discesa golosa (ef = 1) nei piani più alti del livello del nuovo nodo.
    for (let layer = this.maxLevel; layer > level; layer--) {
      entryPoints = this.searchLayer(vector, entryPoints, 1, layer, context).slice(0, 1);
    }
    // Dal livello del nodo in giù: ricerca ampia, scelta dei vicini, archi nei due versi.
    for (let layer = Math.min(level, this.maxLevel); layer >= 0; layer--) {
      const found = this.searchLayer(vector, entryPoints, this.efConstruction, layer, context);
      const neighbors = this.selectNeighbors(vector, found, this.M, layer, index);
      node.neighbors[layer] = neighbors.map((n) => n.index);
      emit?.({ type: "connect", layer, node: index, neighbors: [...node.neighbors[layer]] });

      for (const neighbor of neighbors) {
        const links = this.nodes[neighbor.index].neighbors[layer];
        links.push(index);
        if (links.length > this.maxConnections(layer)) this.shrinkConnections(neighbor.index, layer, emit);
      }
      entryPoints = found;
    }

    const entryPointChanged = level > this.maxLevel;
    if (entryPointChanged) {
      this.entryPoint = index;
      this.maxLevel = level;
    }
    emit?.({ type: "insert-end", node: index, level, entryPoint: this.entryPoint, maxLevel: this.maxLevel, entryPointChanged });
    return index;
  }

  /**
   * Riduce i vicini di un nodo che ha superato il massimo, riscegliendoli con lo stesso
   * algoritmo di selezione (riga "shrink connections" dell'algoritmo 1).
   */
  shrinkConnections(index, layer, emit) {
    const node = this.nodes[index];
    const before = node.neighbors[layer];
    const candidates = before.map((n) => ({ index: n, distance: this.nodeDistance(index, n) }));
    const after = this.selectNeighbors(node.vector, candidates, this.maxConnections(layer), layer, index).map((c) => c.index);
    node.neighbors[layer] = after;
    if (emit) {
      const kept = new Set(after);
      emit({ type: "prune", layer, node: index, before: [...before], after: [...after], removed: before.filter((n) => !kept.has(n)) });
    }
  }

  /**
   * Algoritmo 5, K-NN-SEARCH: i `k` vicini approssimati di `query`.
   *
   * @param {ArrayLike<number>} query
   * @param {number} k
   * @param {{ef?: number, onEvent?: (e: HnswEvent) => void}} [options]
   *   ef effettivo = max(ef, k): con meno di k candidati non si potrebbero restituire k risultati
   * @returns {{index: number, id: string|number, distance: number}[]}
   */
  search(query, k, { ef, onEvent } = {}) {
    if (!Number.isInteger(k) || k < 1) throw new RangeError("k deve essere un intero >= 1");
    if (this.entryPoint === -1) return [];
    if (query.length !== this.dimensions) {
      throw new RangeError(`Dimensione ${query.length} diversa da quella dell'indice (${this.dimensions})`);
    }
    const width = Math.max(ef ?? this.efSearch, k);
    const emit = onEvent;
    const visitedAll = new Set();
    const context = { phase: "search", emit, visitedAll };
    emit?.({ type: "search-start", k, ef: width, entryPoint: this.entryPoint, maxLevel: this.maxLevel });

    let entryPoints = [{ index: this.entryPoint, distance: this.distance(query, this.nodes[this.entryPoint].vector) }];
    for (let layer = this.maxLevel; layer >= 1; layer--) {
      entryPoints = this.searchLayer(query, entryPoints, 1, layer, context).slice(0, 1);
    }
    const found = this.searchLayer(query, entryPoints, width, 0, context);
    const results = found.slice(0, k).map((c) => ({ index: c.index, id: this.nodes[c.index].id, distance: c.distance }));
    emit?.({ type: "search-end", results, visitedCount: visitedAll.size });
    return results;
  }

  /**
   * Ricerca esatta (forza bruta): confronta la query con tutti i nodi. È il riferimento per
   * misurare la recall dell'indice ed è ciò che fa pgvector senza indice.
   *
   * @param {ArrayLike<number>} query
   * @param {number} k
   */
  exactSearch(query, k) {
    return this.nodes
      .map((node) => ({ index: node.index, distance: this.distance(query, node.vector) }))
      .sort(nearestFirst)
      .slice(0, k)
      .map((c) => ({ index: c.index, id: this.nodes[c.index].id, distance: c.distance }));
  }

  /**
   * Struttura del grafo per disegnarla: nodi con livello e, per ogni piano, gli archi.
   * Gli archi di HNSW sono orientati (A può avere B tra i vicini senza il viceversa, dopo una
   * potatura): `edges` li riporta una volta sola come coppie [a, b] con a < b, `mutual` dice se
   * il collegamento esiste nei due versi.
   *
   * @returns {{nodes: {index: number, id: string|number, level: number}[],
   *   layers: {level: number, nodes: number[], edges: [number, number][], mutual: boolean[]}[],
   *   entryPoint: number, maxLevel: number}}
   */
  toGraph() {
    const layers = [];
    for (let level = 0; level <= this.maxLevel; level++) {
      const members = this.nodes.filter((n) => n.level >= level).map((n) => n.index);
      const pairs = new Map();
      for (const index of members) {
        for (const neighbor of this.nodes[index].neighbors[level]) {
          const a = Math.min(index, neighbor);
          const b = Math.max(index, neighbor);
          const key = a * this.nodes.length + b;
          pairs.set(key, { a, b, count: (pairs.get(key)?.count ?? 0) + 1 });
        }
      }
      const sorted = [...pairs.values()].sort((x, y) => x.a - y.a || x.b - y.b);
      layers.push({
        level,
        nodes: members,
        edges: sorted.map(({ a, b }) => [a, b]),
        mutual: sorted.map(({ count }) => count === 2),
      });
    }
    return {
      nodes: this.nodes.map(({ index, id, level }) => ({ index, id, level })),
      layers,
      entryPoint: this.entryPoint,
      maxLevel: this.maxLevel,
    };
  }

  /**
   * Statistiche dell'indice: nodi per piano, archi orientati, grado medio e massimo.
   */
  stats() {
    const levels = [];
    for (let level = 0; level <= this.maxLevel; level++) {
      const members = this.nodes.filter((n) => n.level >= level);
      const degrees = members.map((n) => n.neighbors[level].length);
      const links = degrees.reduce((sum, d) => sum + d, 0);
      levels.push({
        level,
        nodes: members.length,
        directedEdges: links,
        averageDegree: members.length ? links / members.length : 0,
        maxDegree: degrees.length ? Math.max(...degrees) : 0,
        maxAllowed: this.maxConnections(level),
      });
    }
    return {
      nodes: this.nodes.length,
      maxLevel: this.maxLevel,
      entryPoint: this.entryPoint,
      M: this.M,
      M0: this.M0,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      mL: this.mL,
      levels,
    };
  }
}
