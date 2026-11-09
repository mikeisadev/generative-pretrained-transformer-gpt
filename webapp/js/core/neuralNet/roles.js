/**
 * Vocabolario della rete neurale orizzontale: ruoli delle sezioni, schemi dei collegamenti e colori.
 * Modulo di soli dati (niente three.js né DOM): lo usano la validazione, la scena, la legenda e i test.
 *
 * Colori dei ruoli: quelli già in uso nel tab Transformer (KIND_COLORS di views/archLevels.js) per i
 * ruoli in comune, così la stessa rete ha gli stessi colori nella disposizione "A righe" e in quella
 * orizzontale; i ruoli nuovi sono stati scelti con il validatore della skill dataviz in modo che le
 * sezioni vicine nel flusso (es. conv → codebook → conv → uscita) restino distinguibili. Il colore
 * non è mai l'unico segnale: ogni sezione ha la sua etichetta, la scheda e il passaggio del mouse.
 */

/** Ruolo didattico di una sezione: etichetta per la legenda, colore e una frase di spiegazione. */
export const ROLES = Object.freeze({
  input: { label: "Ingresso", color: "#cbd5e1", text: "I dati che entrano: token, pixel del latente, campioni audio." },
  embedding: { label: "Embedding", color: "#f59e0b", text: "Una tabella di pesi: ogni token prende la sua riga (un vettore)." },
  norm: { label: "Normalizzazione", color: "#a78bfa", text: "LayerNorm, RMSNorm, GroupNorm: riporta i numeri su una scala comune, posizione per posizione." },
  attention: { label: "Attention", color: "#38bdf8", text: "Ogni posizione raccoglie informazioni dalle altre posizioni: gli archi in profondità." },
  mlp: { label: "Feed-forward (MLP)", color: "#34d399", text: "Strati densi applicati a ogni posizione separatamente: dove sta gran parte dei pesi." },
  router: { label: "Router (MoE)", color: "#f87171", text: "Un punteggio per esperto e per token: sceglie i top-k esperti di ogni token." },
  experts: { label: "Esperti (MoE)", color: "#a3e635", text: "Tante MLP in parallelo: ogni token ne usa solo poche (quelle accese)." },
  conv: { label: "Convoluzione", color: "#fb923c", text: "Ogni posizione guarda solo i vicini nella griglia o nel tempo, con gli stessi pesi ovunque." },
  latent: { label: "Latente / vettore", color: "#93c5fd", text: "Una rappresentazione compressa: il vettore che riassume l'ingresso." },
  codebook: { label: "Codebook (quantizzazione)", color: "#fde047", text: "Un dizionario di vettori: ogni posizione viene sostituita dalla voce più vicina." },
  add: { label: "Somma (residuo)", color: "#94a3b8", text: "x + f(x): l'uscita del blocco si somma al suo ingresso, numero per numero. Nessun peso." },
  output: { label: "Uscita", color: "#f472b6", text: "Il risultato: logits e probabilità, rumore previsto, campioni ricostruiti." },
  data: { label: "Dati / operazione", color: "#64748b", text: "Operazioni senza pesi (concatenazione, softmax, similarità) o dati esterni." },
});
export const ROLE_IDS = Object.freeze(Object.keys(ROLES));

/**
 * Schemi dei collegamenti. `depth`: true se il meccanismo lavora ATTRAVERSO la profondità (tra token,
 * tra posizioni della griglia, tra frame): sono quelli che si scoprono girando la camera.
 */
export const PATTERNS = Object.freeze({
  dense: { label: "Collegamenti densi", depth: false, text: "Ogni neurone verso quelli della sezione dopo, alla stessa posizione (un nn.Linear applicato a ogni token). Ne disegniamo un campione." },
  identity: { label: "Uno a uno", depth: false, text: "Il neurone i va nel neurone i, alla stessa posizione: normalizzazioni, attivazioni, copie." },
  lookup: { label: "Lookup", depth: false, text: "Per ogni posizione l'unico neurone acceso dell'ingresso (il token) sceglie la sua riga: tutto il vettore della posizione." },
  attention: { label: "Attention", depth: true, text: "Archi tra le posizioni della profondità, un colore per head; causale = solo verso il passato. Spessore e luce = peso." },
  route: { label: "Instradamento MoE", depth: true, text: "Ogni token accende i suoi top-k esperti (più quelli condivisi): gli altri esperti restano spenti per quel token." },
  residual: { label: "Connessione residua", depth: false, text: "Archi sopra le sezioni: l'ingresso salta il blocco e si somma alla sua uscita (peso fisso 1)." },
  conv: { label: "Convoluzione", depth: true, text: "Ogni posizione riceve dai vicini nella griglia (o nel tempo) della sezione prima: un piccolo ventaglio locale." },
  pool: { label: "Pooling", depth: true, text: "Tutte le posizioni confluiscono in una: la profondità si schiaccia (es. media dei token)." },
  broadcast: { label: "Broadcast", depth: true, text: "Un vettore senza profondità raggiunge tutte le posizioni (es. il passo t in ogni pixel)." },
  down: { label: "Downsampling", depth: true, text: "La griglia si rimpicciolisce: ogni posizione riassume un blocco di posizioni (es. 2 × 2)." },
  up: { label: "Upsampling", depth: true, text: "La griglia si ingrandisce: ogni posizione si copia in un blocco di posizioni." },
  explicit: { label: "Pesi veri", depth: false, text: "Coppie di neuroni con il loro peso vero: arancione positivo, blu negativo, più luce = peso più grande." },
});
export const PATTERN_IDS = Object.freeze(Object.keys(PATTERNS));

/** Colori delle head di attention: i primi quattro come nel tab Transformer (neuronScene.js). */
export const HEAD_COLORS = Object.freeze(["#c084fc", "#34d399", "#f472b6", "#facc15", "#38bdf8", "#fb923c", "#a3e635", "#f87171"]);

/**
 * Attivazioni: mappa divergente come le heatmap dei pesi del Transformer (core/visuals.js): blu
 * negativo, grigio a zero, arancione positivo (due tinte e un grigio al centro, regola dataviz).
 * Colori sRGB: gli shader della rete lavorano nello spazio dello schermo, come le texture delle heatmap.
 */
export const ACTIVATION = Object.freeze({
  positive: "#fb923c",
  negative: "#3b82f6",
  zero: "#5b6577", // un neurone a zero resta visibile sullo sfondo scuro
  pending: "#1e2532", // non ancora raggiunto dall'onda, o fuori dalla fetta scelta
  probability: "#67e8f9", // uscite che sono probabilità (azzurro)
  onehot: "#f1f5f9", // ingresso one-hot acceso (quasi bianco)
  off: "#141922", // esperto spento per quel token
  glow: "#fff4d6", // il fronte dell'onda
});

/** Come leggere i valori di una sezione (colore e simulazione). */
export const ACTIVATION_KINDS = Object.freeze(["signed", "positive", "probs", "onehot"]);

/** Colori dei collegamenti che non sono pesi appresi. */
export const LINK_COLORS = Object.freeze({
  structural: "#64748b", // uno a uno, pooling, broadcast, down/up: copie e medie, non pesi
  residual: "#67e8f9", // come il flusso del binario nella vista Architettura
  route: "#a3e635", // come gli esperti
  lookup: "#f59e0b", // come l'embedding
});

/** Colori degli assi (gizmo 3D e chip della scheda): x flusso, y neuroni, z profondità. */
export const AXIS_COLORS = Object.freeze({ x: "#e2e8f0", y: "#94a3b8", z: "#67e8f9" });

/**
 * Le sezioni collegate a una sezione, una voce per sezione: "Embedding (lookup dei token; lookup delle
 * posizioni)" invece di ripetere la sezione per ogni collegamento. Sezioni diverse con lo stesso nome e
 * gli stessi collegamenti (i livelli della U-Net, i layer di un blocco) diventano una voce sola con il
 * numero davanti: "8 × Self- e cross-attention (…)" e non otto voci uguali.
 * @param {object[]} links collegamenti della specifica normalizzata
 * @param {(link: object) => {label: string}} other la sezione dall'altra parte del collegamento
 * @returns {string[]}
 */
export function linkSummary(links, other) {
  const bySection = new Map();
  for (const l of links) {
    const s = other(l);
    if (!bySection.has(s)) bySection.set(s, []);
    const what = l.label || PATTERNS[l.pattern]?.label.toLowerCase() || l.pattern;
    if (!bySection.get(s).includes(what)) bySection.get(s).push(what);
  }
  const counts = new Map();
  for (const [s, what] of bySection) {
    const text = `${s.label} (${what.join("; ")})`;
    counts.set(text, (counts.get(text) ?? 0) + 1);
  }
  return [...counts].map(([text, n]) => (n > 1 ? `${n} × ${text}` : text));
}

