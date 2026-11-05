/**
 * Azione "Ricerca": dalla domanda dell'utente alla risposta in streaming nel frontend React.
 * La ricerca nell'indice HNSW è quella vera (eventi registrati dalla libreria), riprodotta
 * fotogramma per fotogramma; poi topK e soglia, prompt, ChatClient e streaming SSE.
 */
import { shortId } from "./stationsAnswer.js";
import { isFinished } from "./actionIndex.js";
import { phase } from "./scene3d.js";

/** Token della risposta aggiunti a ogni ridisegno della chat durante lo streaming. */
const STREAM_BATCH = 3;

/** Raggruppa i fotogrammi della ricerca per livello (dall'alto verso il livello 0). */
function framesByLayer(frames) {
  const groups = [];
  frames.forEach((f, i) => {
    const last = groups.at(-1);
    if (last && last.layer === f.layer) last.to = i;
    else groups.push({ layer: f.layer, from: i, to: i });
  });
  return groups;
}

export function searchAction(tab) {
  let run = null;
  let answer = null;
  let before = null; // stato della scena prima dell'azione: si ripristina se viene interrotta

  /** Stato di partenza di ogni passo: database pieno, stazioni della ricerca fino alla fase `stage`. */
  const baseline = (stage) => {
    const { world, plots, kb } = tab;
    const s = world.stations;
    tab.ensureIndexed();
    s.search.setQuestion(stage >= 1 ? run : null, 1);
    s.search.setScores(stage >= 3 ? run : null, 1);
    s.prompt.set(stage >= 4 ? run : null);
    s.llm.set("idle");
    s.answer.set(stage >= 0 ? run : null, answer, stage >= 6 ? answer.tokens.length : -1);
    if (stage < 2) s.hnsw.setGraph(kb.inserts.length);
    plots.setProjection(kb, kb.inserts.length, stage >= 1 ? run : null);
    plots.setSearch(kb, stage >= 2 ? run : null, stage >= 3 ? run.comparisons : 0);
  };

  return {
    id: "ricerca",
    title: "Ricerca: dalla domanda alla risposta",
    category: "Ricerca",
    summary: "Embedding della domanda → ricerca HNSW (discesa e fascio) contro ricerca esatta → topK e soglia → prompt → LLM → streaming SSE.",
    onStart() {
      before = tab.snapshot();
      tab.prepareAction();
      run = tab.currentSearch();
      answer = tab.answerFor(run);
    },
    onEnd() {
      tab.world.showSearch(run, answer);
    },
    onStop({ player }) {
      if (!isFinished(player)) tab.restore(before);
    },
    build() {
      const { kb } = tab;
      const N = kb.inserts.length;
      const steps = [];
      const groups = framesByLayer(run.frames);
      const topLayer = groups[0]?.layer ?? 0;
      const ef = run.frames[0]?.ef ?? 0;

      steps.push({
        title: "La domanda",
        caption: `"${run.question}": React la invia al backend (POST /api/chat/rag/stream); il ChatBotController la passa al ChatService.`,
        duration: 3,
        enter: () => {
          baseline(0);
          tab.focus("answer");
          tab.code("chatController", ["@PostMapping", "streamRagMessage"]);
        },
      });
      steps.push({
        title: "Embedding della domanda",
        caption: "RagService.search chiama similaritySearch: Spring AI trasforma la domanda in un vettore con lo stesso EmbeddingModel dei chunk.",
        duration: 3.5,
        enter: () => {
          baseline(1);
          tab.focus("search");
          tab.code("ragService", ["SearchRequest.builder()", "vectorStore.similaritySearch"]);
        },
        update: (p) => tab.world.stations.search.setQuestion(run, p),
      });
      groups.forEach((g, gi) => {
        const isBase = g.layer === 0;
        const frames = g.to - g.from + 1;
        steps.push({
          title: isBase ? `Livello 0: ricerca a fascio (ef = ${ef})` : gi === 0 ? `Punto di ingresso, livello ${g.layer}` : `Livello ${g.layer}: discesa golosa`,
          caption: isBase
            ? `Al livello 0 la ricerca tiene i ${ef} migliori (verdi) e una coda di candidati (gialli): esplora i vicini del candidato migliore finché nessun candidato può migliorare la lista.`
            : gi === 0
              ? `La ricerca parte sempre dal punto di ingresso, il nodo nel livello più alto (${topLayer}). Il rombo giallo è la domanda, proiettata nel piano.`
              : `Nel livello ${g.layer} si guarda solo il nodo più vicino (ef = 1): ci si sposta sul vicino più simile alla domanda finché si migliora, poi si scende.`,
          duration: Math.max(2.5, frames * (isBase ? 0.45 : 0.8)),
          enter: () => {
            baseline(2);
            tab.focus("hnsw.stack");
            tab.code("lib:searchLayer", isBase ? ["found.size > ef", "current.distance > furthest.distance"] : ["visited.add"]);
            tab.world.stations.hnsw.setSearch(run, g.from);
          },
          update: (p) => {
            const f = g.from + Math.min(frames - 1, Math.floor(p * frames));
            const frame = tab.world.stations.hnsw.setSearch(run, f);
            tab.plots.setSearch(kb, run, frame.comparisons);
          },
        });
      });
      const saved = N - run.comparisons;
      steps.push({
        title: "HNSW contro ricerca esatta",
        caption:
          saved > 0
            ? `HNSW ha calcolato ${run.comparisons} distanze invece di ${N} (${saved} in meno), recall ${Math.round(run.recall * 100)}%. Con milioni di chunk il risparmio diventa enorme: vedi i Grafici.`
            : `Con soli ${N} chunk HNSW ha calcolato ${run.comparisons} distanze, la ricerca esatta ${N}: su così pochi vettori l'indice non conviene (rilegge alcuni nodi a ogni livello). Il vantaggio arriva con tanti vettori: vedi i Grafici.`,
        duration: 5.5,
        enter: () => {
          baseline(3);
          tab.world.stations.hnsw.setSearch(run, run.frames.length - 1, { showResults: true });
          tab.focus("hnsw");
          tab.code("lib:exact", ["this.nodes"]);
        },
      });
      steps.push({
        title: "topK e soglia di similarità",
        caption: run.results.length
          ? `Score = 1 − distanza del coseno. Passano i chunk con score sopra ${run.similarityThreshold}, al massimo ${run.topK}: ne restano ${run.results.length}. Il migliore: ${shortId(run.results[0].id)} (${run.results[0].score.toFixed(2)}).`
          : `Nessun chunk supera la soglia ${run.similarityThreshold}: la domanda è fuori tema rispetto ai documenti. Il prompt avrà "Nessun documento pertinente trovato".`,
        duration: 5,
        enter: () => {
          baseline(3);
          tab.world.stations.hnsw.setSearch(run, run.frames.length - 1, { showResults: true });
          tab.focus("search");
          tab.code("pgSearch", ["double distance = 1 - request.getSimilarityThreshold()", "COSINE_DISTANCE("]);
        },
        update: (p) => tab.world.stations.search.setScores(run, p),
      });
      steps.push({
        title: "Chunk recuperati",
        caption: run.results.length
          ? `Recuperati: ${run.results.map((r) => shortId(r.id)).join(", ")}. Nella nuvola le linee verdi collegano la domanda ai chunk scelti.`
          : "Nessun chunk recuperato.",
        duration: 3.5,
        enter: () => {
          baseline(3);
          tab.world.stations.hnsw.setSearch(run, run.frames.length - 1, { showResults: true });
          tab.focus("hnsw.cloud");
          tab.code("lib:storeSearch", ["filter((c) => c.distance < maxDistance)", ".slice(0, topK)"]);
        },
      });
      const parts = run.prompt.parts.length;
      const flagged = run.results.some((r) => r.metadata.flagged);
      steps.push({
        title: "Costruzione del prompt",
        caption: `Prompt strutturato: definizioni degli strumenti, system prompt, domanda e ${run.results.length} documenti nei tag <documento>, circa ${run.prompt.tokens.total} token.${flagged ? " Attenzione: tra i documenti c'è il chunk segnalato in pulizia: entra come dato, con < e > neutralizzati." : ""}`,
        duration: 2 + parts * 0.7,
        enter: () => {
          baseline(4);
          tab.world.stations.prompt.set(run, 0);
          tab.focus("prompt");
          tab.code(flagged ? "formatAsContext" : "contextAdvisor", flagged ? ["neutralizeTags"] : ["TEMPLATE", "augmentUserMessage"]);
        },
        update: (p) => tab.world.stations.prompt.set(run, Math.ceil(phase(p, 0, 0.85) * parts)),
      });
      steps.push({
        title: "Chiamata all'LLM",
        caption: "ChatClient manda system prompt e messaggio utente al provider; l'LLM (un transformer) legge tutto il contesto e comincia a generare.",
        duration: 3.5,
        enter: () => {
          baseline(5);
          tab.focus("llm");
          tab.code("chatService", [".system(systemPrompt)", ".user(userPrompt)", ".stream()"]);
        },
        update: (p) => tab.world.stations.llm.set("prompt", p),
      });
      steps.push({
        title: "Risposta in streaming (SSE)",
        caption: "Ogni token esce dal Flux e diventa un evento SSE: React lo aggiunge alla bolla. Risposta simulata: frasi dei chunk recuperati, con la fonte.",
        duration: Math.min(14, 2 + answer.tokens.length * 0.1),
        enter: () => {
          baseline(5);
          tab.world.stations.answer.set(run, answer, 0);
          tab.focus("answer");
          tab.code("reactStream", ["getReader", "onToken"]);
        },
        update: (p) => {
          tab.world.stations.llm.set("stream", p);
          // la chat si ridisegna a gruppi di STREAM_BATCH token: ogni ridisegno ricarica una texture
          // grande, farlo a ogni token rallentava l'animazione senza rendere lo streaming più chiaro
          const n = answer.tokens.length;
          const shown = Math.min(n, Math.ceil((phase(p, 0.05, 0.95) * n) / STREAM_BATCH) * STREAM_BATCH);
          tab.world.stations.answer.set(run, answer, shown);
        },
      });
      steps.push({
        title: "Fine",
        caption: "Il contesto (system + documenti + domanda) è ciò che l'LLM legge: per vederlo entrare in un transformer usa il pulsante \"Guarda come il contesto entra nell'LLM\".",
        duration: 3,
        enter: () => {
          baseline(6);
          tab.world.stations.hnsw.setSearch(run, run.frames.length - 1, { showResults: true });
          tab.focus("answer");
          tab.code("chatServiceL18", ["ChatEvent.Sources", "ChatEvent.Token"]);
        },
      });
      return steps;
    },
  };
}
