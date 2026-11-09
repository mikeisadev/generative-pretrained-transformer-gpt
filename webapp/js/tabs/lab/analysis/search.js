/**
 * Voci della ricerca globale di confronto, grafici e benchmark del Laboratorio (fase 2). SOLI DATI:
 * search.js del tab lo importa anche quando il tab non è mai stato aperto (niente three.js né scena).
 * Voce: { id, kind: "view", title, text?, keywords? }; gli id li apre tab.views.open(id) (registrati da
 * analysis/index.js con tab.views.register) e links.js li riconosce da ANALYSIS_VIEWS (#lab/go=view:<id>).
 */
import { BENCH, COMPARE } from "./content.js";

/** Le viste della fase 2 raggiungibili con #lab/go=view:<id> (id → che cosa apre). */
export const ANALYSIS_VIEWS = Object.freeze({
  confronto: "il pannello Confronto: A e B, la tabella dei tensori più cambiati, gli istogrammi sovrapposti",
  "confronto:training": "il confronto pronto «che cosa ha cambiato il training»: pesi finali contro step 0, con la differenza nella scena",
  "confronto:fasi": "il confronto pronto tra due fasi del training (step 5.000 contro step 500)",
  "confronto:corso": "il confronto pronto tra il tuo run e il GPT del corso",
  differenza: "la differenza A − B disegnata nella scena (con un B predefinito se manca)",
  probabilita: "il prossimo carattere su un prompt: probabilità e logits di A e di B",
  grafici: "i grafici del training: loss, learning rate, gradiente, update ratio, metriche degli snapshot",
  benchmark: "il pannello Benchmark: loss, perplexity, bit per carattere, top-1, top-5 dai benchmark.json",
  "benchmark:browser": "il benchmark ricalcolato nel browser sull'estratto, con il controllo di coerenza rispetto al Python",
});

export const searchEntries = [
  { id: "confronto", kind: "view", title: "Confronto A/B dei pesi, tensore per tensore", text: COMPARE.intro, keywords: ["confronto", "compare", "differenza", "prima e dopo", "norma", "coseno", "cambio relativo", "tensori più cambiati", "diff"] },
  { id: "confronto:training", kind: "view", title: "Che cosa ha cambiato il training: pesi finali contro step 0", text: "Il GPT del corso alla fine dei 5.000 step (A) contro i suoi pesi casuali dello step 0 (B): quali matrici si sono mosse di più.", keywords: ["training", "inizializzazione", "step 0", "pesi finali", "evoluzione", "apprendimento"] },
  { id: "confronto:fasi", kind: "view", title: "Fasi del training: step 5.000 contro step 500", text: "Due snapshot dello stesso training: che cosa cambia tra l'inizio e la fine, pesi dopo pesi.", keywords: ["snapshot", "fasi", "step 500", "step 5000", "durante il training"] },
  { id: "confronto:corso", kind: "view", title: "Il tuo run contro il GPT del corso (prima e dopo)", text: "Un run caricato (A) contro il GPT del corso (B): che cosa ha cambiato la tua modifica, anche con un'architettura diversa.", keywords: ["run caricato", "upload", "miglioramento", "architettura modificata", "claude", "opus"] },
  { id: "differenza", kind: "view", title: "La differenza A − B nella scena, cella per cella", text: "Le matrici della scena colorate con A − B (blu: A più piccolo, arancione: A più grande) e, da vicino, i numeri della differenza.", keywords: ["A - B", "differenza", "colormap divergente", "heatmap della differenza", "delta"] },
  { id: "probabilita", kind: "view", title: "Probabilità e logits del prossimo carattere: A contro B", text: "Un prompt scritto da te, i due modelli a confronto: quale carattere si aspettano e con quale sicurezza.", keywords: ["logits", "probabilità", "softmax", "prossimo token", "prompt", "next token"] },
  { id: "grafici", kind: "view", title: "Grafici del training: loss train e validation, learning rate, gradiente", text: "Le curve vere del training di A e di B: loss di train e di validation sovrapposte, learning rate, norma del gradiente, update ratio e deviazione standard dei pesi per gruppo, metriche degli snapshot, entropia dell'attention per head.", keywords: ["loss", "loss function", "curve", "validation", "overfitting", "learning rate", "gradient norm", "update ratio", "entropia", "attention entropy", "grafico"] },
  { id: "benchmark", kind: "view", title: "Benchmark: loss, perplexity, bit per carattere, top-1, top-5", text: BENCH.intro, keywords: ["benchmark", "perplexity", "bpc", "accuratezza", "top-1", "top-5", "valutazione", "benchmark.json", "metriche"] },
  { id: "benchmark:browser", kind: "view", title: "Benchmark nel browser e controllo di coerenza con il Python", text: BENCH.browserIntro, keywords: ["coerenza", "motore JS", "verifica", "estratto", "val_excerpt", "greedy", "riproducibilità"] },
];
