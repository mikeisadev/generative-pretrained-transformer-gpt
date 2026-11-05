/**
 * Regione "Tool use e agenti": il ciclo utente → modello → chiamata di strumento → programma che
 * la esegue → risultato → modello → risposta. Un pacchetto percorre il ciclo; sotto, il messaggio
 * che viaggia in quel momento (formato di esempio, semplificato).
 */
import { alongPath, arrow, card, packet, PALETTE, Plate } from "./shapes.js";

const NODES = {
  user: { x: -11.5, y: 5.6, title: "Utente", sub: "fa una domanda" },
  llm: { x: 0, y: 5.6, title: "LLM", sub: "rispondo o uso uno strumento?" },
  call: { x: 11.5, y: 5.6, title: "Chiamata", sub: "nome + argomenti (JSON)" },
  tool: { x: 11.5, y: -3.4, title: "Strumento", sub: "il TUO codice: API, database…" },
  result: { x: 0, y: -3.4, title: "Risultato", sub: "torna nel contesto" },
  answer: { x: -11.5, y: -3.4, title: "Risposta", sub: "testo finale" },
};

/** Le tappe del ciclo, con il messaggio che viaggia in quella tappa. */
export const AGENT_LEGS = [
  { from: "user", to: "llm", message: [{ text: "utente:", color: PALETTE.muted }, { text: "«Che tempo fa domani a Torino?»" }] },
  { from: "llm", to: "call", message: [{ text: "il modello genera una chiamata strutturata:", color: PALETTE.muted }, { text: '{ "name": "meteo", "arguments": { "citta": "Torino", "giorno": "domani" } }' }] },
  { from: "call", to: "tool", message: [{ text: "il programma (non il modello!) esegue la funzione:", color: PALETTE.muted }, { text: 'meteo(citta="Torino", giorno="domani")' }] },
  { from: "tool", to: "result", message: [{ text: "risultato dello strumento:", color: PALETTE.muted }, { text: '{ "previsione": "pioggia", "max": 14, "min": 8 }' }] },
  { from: "result", to: "llm", message: [{ text: "il risultato entra nel contesto e il modello riprende a generare", color: PALETTE.muted }, { text: "(un agente può ripetere il ciclo molte volte)" }] },
  { from: "llm", to: "answer", message: [{ text: "risposta:", color: PALETTE.muted }, { text: "«Domani a Torino è prevista pioggia, tra 8 e 14 gradi.»" }] },
];

const edge = (a, b) => {
  const A = NODES[a];
  const B = NODES[b];
  if (a === "result" && b === "llm") return [[A.x + 1.5, A.y + 1.6], [B.x + 1.5, B.y - 1.6]];
  if (a === "llm" && b === "answer") return [[A.x - 1.5, A.y - 1.6], [A.x - 1.5, 0.9], [B.x, 0.9], [B.x, B.y + 1.6]];
  const dx = Math.sign(B.x - A.x);
  const dy = Math.sign(B.y - A.y);
  return [[A.x + dx * 4.6, A.y + dy * 1.6], [B.x - dx * 4.6, B.y - dy * 1.6]];
};

export function buildAgent(group) {
  const cards = new Map(Object.entries(NODES).map(([id, n]) => [id, card(group, { x: n.x, y: n.y, w: 9.2, h: 3.2, title: n.title, sub: n.sub, color: id === "llm" ? PALETTE.modern : id === "tool" || id === "call" ? PALETTE.tool : PALETTE.muted, size: 0.5 })]));
  const paths = AGENT_LEGS.map((leg) => edge(leg.from, leg.to));
  paths.forEach((p) => arrow(group, p, { color: 0x34d399, radius: 0.06 }));
  const message = new Plate({ w: 34, h: 2.6, size: 0.6, align: "center", mono: true, background: "#0f172a", border: "#334155" }).place(group, 0, -8.2);
  const note = new Plate({ w: 34, h: 1.3, size: 0.47, align: "center", color: PALETTE.muted }).place(group, 0, -11.2);
  note.set("Formato di esempio: ogni API (Anthropic, OpenAI, Spring AI…) ha il suo, l'idea è la stessa.");
  const ball = packet("#34d399", 0.5);
  group.add(ball);
  const tmp = { x: 0, y: 0 };

  const api = {
    /** Tappa k del ciclo, con il pacchetto a frazione t del percorso. */
    leg(k, t = 1) {
      const leg = AGENT_LEGS[Math.max(0, Math.min(AGENT_LEGS.length - 1, k))];
      for (const [id, c] of cards) c.set(k < 0 ? 0 : id === leg.to && t > 0.85 ? 0.9 : id === leg.from ? 0.35 : 0);
      alongPath(paths[AGENT_LEGS.indexOf(leg)], t, tmp);
      ball.position.set(tmp.x, tmp.y, 0.6);
      ball.visible = k >= 0;
      message.set(k >= 0 ? leg.message : [{ text: "Il modello non esegue nulla: scrive una richiesta.", bold: true }, { text: "È il programma che chiama lo strumento e gli restituisce il risultato.", color: PALETTE.muted }]);
    },
  };
  api.leg(-1);
  return api;
}
