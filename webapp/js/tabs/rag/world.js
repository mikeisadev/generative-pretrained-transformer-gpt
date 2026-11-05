/**
 * Il livello "Dettaglio" della scena: tutte le stazioni, il binario che le collega e gli stati
 * d'insieme ("database vuoto", "database pieno", "risultato di una ricerca").
 *
 * Le stazioni dipendono dalla base di conoscenza (configurazione di splitter e HNSW): quando
 * la configurazione cambia, `rebuild` libera le risorse della versione precedente e ricostruisce.
 */
import * as THREE from "three";
import { makeFlow } from "../../core/diagram.js";
import { disposeTree } from "./board.js";
import { ROW_BOTTOM, ROW_TOP, STATIONS, STATION_H } from "./scene3d.js";
import { buildHnsw } from "./stationHnsw.js";
import { buildAnswer, buildLLM, buildPrompt, buildSearch } from "./stationsAnswer.js";
import { buildClean, buildExtract, buildUpload } from "./stationsIngest.js";
import { buildChunk, buildEmbed, buildTable } from "./stationsStore.js";

export class World {
  constructor(host) {
    this.host = host;
    this.group = new THREE.Group();
    this.group.name = "rag-detail";
    this.stations = null;
    this.flow = null;
    this.kb = null;
    /** Ricerca mostrata per intero ({run, answer}) oppure null. */
    this.search = null;
    host.onFrame((dt) => this.flow?.userData.tick(dt));
  }

  /** Costruisce (o ricostruisce) tutte le stazioni per la base di conoscenza `kb`. */
  rebuild(kb) {
    if (this.stations) {
      for (const child of [...this.group.children]) {
        this.group.remove(child);
        disposeTree(child);
      }
    }
    this.kb = kb;
    const s = {
      upload: buildUpload(kb),
      extract: buildExtract(kb),
      clean: buildClean(kb),
      chunk: buildChunk(kb),
      embed: buildEmbed(kb),
      table: buildTable(kb),
      hnsw: buildHnsw(kb),
      search: buildSearch(kb),
      prompt: buildPrompt(kb),
      llm: buildLLM(kb),
      answer: buildAnswer(kb),
    };
    for (const station of Object.values(s)) this.group.add(station.group);
    this.stations = s;
    // binario: sotto la riga dell'indicizzazione, giù a destra, sotto la riga della ricerca
    const yTop = ROW_TOP - 17.5; // sotto le stazioni più alte (Pulizia, Chunking)
    const yBottom = ROW_BOTTOM - STATION_H / 2 - 1.2;
    const xRight = STATIONS.hnsw.x + STATIONS.hnsw.w / 2 + 3;
    const points = [
      new THREE.Vector3(STATIONS.upload.x - STATIONS.upload.w / 2, yTop, 0),
      new THREE.Vector3(xRight, yTop, 0),
      new THREE.Vector3(xRight, yBottom, 0),
      new THREE.Vector3(STATIONS.answer.x - STATIONS.answer.w / 2, yBottom, 0),
    ];
    this.flow = makeFlow(points, { radius: 0.18, speed: 0.6 });
    this.flow.userData.noBounds = true;
    this.group.add(this.flow);
    this.indexed = 0;
    return s;
  }

  /** Database vuoto: tutte le stazioni allo stato iniziale. */
  reset() {
    const s = this.stations;
    s.upload.set("idle");
    s.extract.set(0);
    s.clean.set(-1);
    s.chunk.set(0, 0);
    s.embed.set(0, 0);
    s.table.set(0);
    s.hnsw.setGraph(0);
    s.hnsw.genericInfo(0);
    this.clearSearch();
    this.indexed = 0;
  }

  /** Database pieno (indicizzazione completata), senza animazione. */
  showIndexed() {
    const s = this.stations;
    const n = this.kb.inserts.length;
    s.upload.set("batch", 1);
    s.extract.set(1);
    s.clean.set(s.clean.pageNumber ? this.kb.featured.cleaning.steps.length - 1 : -1);
    s.chunk.set(this.kb.featured.pages.length, 1);
    s.embed.set(this.kb.featured.chunks.length - 1, 1);
    s.table.set(n);
    s.hnsw.setGraph(n);
    s.hnsw.genericInfo(n);
    this.indexed = n;
  }

  /** Stazioni della ricerca allo stato "nessuna domanda". */
  clearSearch() {
    const s = this.stations;
    s.search.setQuestion(null);
    s.search.setScores(null, 0);
    s.prompt.set(null);
    s.llm.set("idle");
    s.answer.set(null, null);
    this.search = null;
  }

  /** Mostra il risultato completo di una ricerca (es. dopo la fine dell'azione o da un passo del percorso). */
  showSearch(run, answer) {
    const s = this.stations;
    if (this.indexed < this.kb.inserts.length) this.showIndexed();
    s.search.setQuestion(run, 1);
    s.search.setScores(run, 1);
    s.hnsw.setSearch(run, run.frames.length - 1, { showResults: true });
    s.prompt.set(run);
    s.llm.set("idle");
    s.answer.set(run, answer, answer.tokens.length);
    this.search = { run, answer };
  }
}
