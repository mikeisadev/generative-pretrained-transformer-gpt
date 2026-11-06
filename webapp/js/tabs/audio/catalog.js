/**
 * Catalogo del tab audio: nomi e descrizioni brevi delle azioni animate e delle viste.
 *
 * Solo dati (nessun import): lo usano sia il tab (actions.js, index.js) sia l'indice della ricerca
 * globale (search.js), che viene caricato anche quando il tab non è mai stato aperto. Così un titolo
 * si cambia in un posto solo e la ricerca trova sempre lo stesso nome che lo studente vede.
 */

/** Azioni del player: stessi id di player.register (actions.js), nell'ordine del pannello "Azioni". */
export const ACTIONS = Object.freeze({
  sound: {
    id: "audio-sound",
    title: "Dal suono ai numeri",
    category: "Rappresentare il suono",
    summary: "Campionamento, finestre, FFT, scala Mel e logaritmo: come un'onda diventa una tabella di numeri.",
  },
  rvq: {
    id: "audio-rvq",
    title: "Comprimere il suono in token (RVQ)",
    category: "Codec neurale",
    summary: "Ogni frame diventa 4 numeri interi: codebook dopo codebook l'errore cala. Poi il decoder torna al suono.",
  },
  generate: {
    id: "audio-generate",
    title: "Generare musica token per token",
    category: "Generazione",
    summary: "Prompt, cross-attention, delay pattern: il transformer sceglie il prossimo token, ma i token sono pezzi di suono.",
  },
  whisper: {
    id: "audio-whisper",
    title: "Dalla voce al testo (Whisper)",
    category: "Riconoscimento vocale",
    summary: "log-Mel di 30 secondi, encoder con convoluzioni e transformer, decoder che scrive token di testo.",
  },
});

/** Categorie del pannello "Azioni", nell'ordine delle azioni. */
export const ACTION_CATEGORIES = [...new Set(Object.values(ACTIONS).map((a) => a.category))];

/**
 * Viste del pannello "Viste": la stessa scena guardata da punti diversi. L'id è un rettangolo del
 * registro da inquadrare (FOCUS in layout3d.js). "overview" è anche la vista iniziale ("Centra").
 */
export const VIEWS = Object.freeze([
  { id: "overview", label: "Vista d'insieme", text: "Tutto il percorso: suono → token → suono, il prompt e Whisper." },
  { id: "delayCore", label: "Generazione musicale", text: "Transformer e delay pattern: il cuore di MusicGen." },
  { id: "rvq", label: "Codec: token audio", text: "Codebook, residui e griglia dei token." },
  { id: "whisperAll", label: "Whisper: dalla voce al testo", text: "Vista alternativa: log-Mel → encoder → decoder → testo." },
]);

/** I tre livelli dello zoom semantico (stessi id del LODController in index.js). */
export const LEVELS = Object.freeze([
  { id: "detail", label: "Dettaglio", text: "Ogni componente da vicino, con i dati calcolati nel browser: forma d'onda, spettrogramma, codebook, griglia dei token, delay pattern, log-Mel di Whisper." },
  { id: "blocks", label: "Blocchi", text: "Un blocco per componente, con una frase su cosa fa: audio, encoder, RVQ, transformer, delay pattern, decoder, Whisper." },
  { id: "macro", label: "Panoramica", text: "Lo schema più astratto in cinque blocchi: suono → token → suono, il testo del prompt e il percorso opposto di Whisper (voce → testo)." },
]);
