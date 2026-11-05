/**
 * Regione "Pipeline di addestramento": pre-training → SFT → preferenze/RL, con un esempio di
 * dati per ogni fase e i tre metodi di allineamento (RLHF, DPO, Constitutional AI).
 * Il "modello" è una sfera che passa da una fase all'altra.
 */
import { alongPath, arrow, card, packet, PALETTE, Plate } from "./shapes.js";

export const STAGES = [
  {
    title: "Pre-training",
    sub: "trilioni di token, prossimo token",
    data: [{ text: "Testo da web, libri, codice:", color: PALETTE.muted }, { text: "«il gatto dorme sul» → «divano»" }, { text: "Risultato: un modello base", color: PALETTE.muted }],
  },
  {
    title: "SFT",
    sub: "fine-tuning supervisionato",
    data: [{ text: "Dialoghi scritti da persone:", color: PALETTE.muted }, { text: "Utente: riassumi questo testo" }, { text: "Assistente: ecco il riassunto…" }],
  },
  {
    title: "Preferenze / RL",
    sub: "quale risposta è migliore?",
    data: [{ text: "Coppie confrontate:", color: PALETTE.muted }, { text: "risposta A  >  risposta B" }, { text: "Il modello impara a preferire A", color: PALETTE.muted }],
  },
];

export const METHODS = [
  { title: "RLHF (InstructGPT)", sub: "reward model + PPO" },
  { title: "DPO", sub: "loss diretta, senza reward model" },
  { title: "Constitutional AI", sub: "feedback di un'AI guidata da principi" },
];

const stageX = (k) => -11.6 + k * 11.6;
const PATH = [[-16.5, 5.6], [stageX(0), 5.6], [stageX(1), 5.6], [stageX(2), 5.6], [16.5, 5.6]];

export function buildTraining(group) {
  const stages = STAGES.map((s, k) => card(group, { x: stageX(k), y: 5.6, w: 9.6, h: 3.6, title: s.title, sub: s.sub, color: PALETTE.ok, size: 0.55 }));
  arrow(group, [[stageX(0) + 4.9, 5.6], [stageX(1) - 4.9, 5.6]], { color: 0x4ade80 });
  arrow(group, [[stageX(1) + 4.9, 5.6], [stageX(2) - 4.9, 5.6]], { color: 0x4ade80 });
  const data = STAGES.map((s, k) => new Plate({ w: 10.6, h: 3.8, size: 0.49, align: "center", background: "#0f172a", border: "#334155" }).place(group, stageX(k), -0.4).set(s.data));
  const methods = METHODS.map((m, k) => card(group, { x: stageX(k), y: -6.4, w: 10.6, h: 2.4, title: m.title, sub: m.sub, color: "#86efac", size: 0.4 }));
  arrow(group, [[stageX(2), 3.6], [stageX(2), 1.7]], { color: 0x86efac, radius: 0.05 });
  const note = new Plate({ w: 34, h: 2.8, size: 0.55, align: "center" }).place(group, 0, -10.4);
  const ball = packet("#fde047", 0.55);
  group.add(ball);
  const tmp = { x: 0, y: 0 };

  const api = {
    /** Sposta il "modello" lungo la pipeline (t in 0..1) e accende la fase k. */
    setStage(k, t = 1) {
      stages.forEach((s, i) => s.set(i === k ? 0.8 : 0));
      data.forEach((d, i) => (d.mesh.visible = k < 0 || i <= k));
      alongPath(PATH, k < 0 ? 0 : (k + t) / 3, tmp);
      ball.position.set(tmp.x, tmp.y + 2.3, 0.6);
      ball.visible = k >= 0;
    },
    method(i) {
      methods.forEach((m, j) => m.set(j === i ? 0.8 : 0));
    },
    note(lines) {
      note.set(lines);
    },
  };
  api.setStage(-1);
  api.method(-1);
  api.note([{ text: "Il nostro GPT si ferma alla prima fase: è un modello base che continua il testo.", bold: true }, { text: "Gli assistenti (ChatGPT, Claude, Llama Instruct…) passano anche dalle altre due.", color: PALETTE.muted }]);
  return api;
}
