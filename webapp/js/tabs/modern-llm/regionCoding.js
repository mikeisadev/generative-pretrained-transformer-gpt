/**
 * Regione "Capacità di coding".
 * In alto: la trasformazione fill-in-the-middle (Bavarian et al. 2022): il documento viene
 * tagliato in prefisso, centro e suffisso e riordinato in PRE · SUF · MID, così un modello che
 * scrive da sinistra a destra impara a riempire un buco. In basso: il ciclo scrivi → esegui i
 * test → leggi l'errore → correggi (feedback di esecuzione).
 */
import { alongPath, arrow, box, card, ease, packet, PALETTE, Plate, text } from "./shapes.js";

export const FIM_PARTS = [
  { id: "pre", label: "prefisso", code: "def area(r):", color: "#3b82f6", w: 9 },
  { id: "mid", label: "centro (da riempire)", code: "    return 3.14159 * r * r", color: "#eab308", w: 7 },
  { id: "suf", label: "suffisso", code: "print(area(2))", color: "#22c55e", w: 8 },
];
const Y_FIM = 6.3;
const GAP = 0.6;

/** x del centro di ogni parte nell'ordine dato (sentinelle comprese). */
function layout(order) {
  const total = order.reduce((s, id) => s + FIM_PARTS.find((p) => p.id === id).w, 0) + GAP * (order.length - 1);
  let x = -total / 2;
  const out = {};
  for (const id of order) {
    const w = FIM_PARTS.find((p) => p.id === id).w;
    out[id] = x + w / 2;
    x += w + GAP;
  }
  return out;
}
const DOC = layout(["pre", "mid", "suf"]);
const FIM = layout(["pre", "suf", "mid"]);

const LOOP = [
  { title: "Scrivi il codice", x: -11.5 },
  { title: "Esegui i test", x: 0 },
  { title: "Leggi l'errore", x: 11.5 },
];
const Y_LOOP = -4.4;

const CODE_DOC = [{ text: "def area(r):", color: "#93c5fd" }, { text: "    return 3.14159 * r * r", color: "#fde047" }, { text: "print(area(2))", color: "#86efac" }];
const CODE_FIM = [{ text: "<PRE>def area(r):", color: "#93c5fd" }, { text: "<SUF>print(area(2))", color: "#86efac" }, { text: "<MID>    return 3.14159 * r * r", color: "#fde047" }];
const STATUS = new Map();
/** Righe di stato dei test, create una volta per ogni esito. */
const statusLines = (passed, total) => {
  const key = passed * 100 + total;
  if (!STATUS.has(key))
    STATUS.set(key, [
      { text: `test superati: ${passed} / ${total}`, bold: true, color: passed === total ? "#86efac" : "#fca5a5" },
      { text: "Il feedback viene da un programma vero (i test), non da un'opinione.", color: PALETTE.muted },
    ]);
  return STATUS.get(key);
};

export function buildCoding(group) {
  const bars = FIM_PARTS.map((p) => {
    const b = box({ w: p.w, h: 1.5, d: 0.4, color: p.color, emissive: p.color });
    b.material.emissiveIntensity = 0.25;
    b.position.set(DOC[p.id], Y_FIM, 0);
    group.add(b);
    const tag = text(group, p.label, DOC[p.id], Y_FIM, { size: 0.42, z: 0.5, color: "#0b1220" });
    return { part: p, bar: b, tag };
  });
  const sentinels = new Plate({ w: 34, h: 1.1, size: 0.55, align: "center", mono: true, color: PALETTE.muted }).place(group, 0, 8.4);
  const code = new Plate({ w: 34, h: 3.4, size: 0.55, align: "center", mono: true, background: "#0f172a", border: "#334155" }).place(group, 0, 2.6);

  const cards = LOOP.map((l) => card(group, { x: l.x, y: Y_LOOP, w: 9, h: 2.6, title: l.title, color: "#93c5fd", size: 0.48 }));
  arrow(group, [[-6.9, Y_LOOP], [-4.6, Y_LOOP]], { color: 0x93c5fd });
  arrow(group, [[4.6, Y_LOOP], [6.9, Y_LOOP]], { color: 0x93c5fd });
  arrow(group, [[11.5, Y_LOOP - 1.4], [11.5, -7.2], [-11.5, -7.2], [-11.5, Y_LOOP - 1.4]], { color: 0x93c5fd });
  text(group, "correggi e riprova", 0, -6.6, { size: 0.38, color: PALETTE.muted });
  const status = new Plate({ w: 34, h: 2.6, size: 0.57, align: "center" }).place(group, 0, -9.9);
  const ball = packet("#93c5fd", 0.45);
  group.add(ball);
  const loopPath = [[-11.5, Y_LOOP + 1.5], [0, Y_LOOP + 1.5], [11.5, Y_LOOP + 1.5], [11.5, -7.2], [-11.5, -7.2], [-11.5, Y_LOOP + 1.5]];
  const tmp = { x: 0, y: 0 };

  const api = {
    /** t: 0 = documento in ordine, 1 = riordinato per FIM. */
    fim(t) {
      const k = ease(t);
      for (const { part, bar, tag } of bars) {
        const x = DOC[part.id] + (FIM[part.id] - DOC[part.id]) * k;
        const lift = part.id === "mid" ? Math.sin(Math.PI * k) * 2.6 : 0;
        bar.position.set(x, Y_FIM + lift, 0);
        tag.position.set(x, Y_FIM + lift, 0.5);
      }
      sentinels.set(t > 0.5 ? "<PRE> prefisso  <SUF> suffisso  <MID> centro → il modello genera il centro per ultimo" : "documento originale: prefisso · centro · suffisso");
      code.set(t > 0.5 ? CODE_FIM : CODE_DOC);
    },
    /** Ciclo di esecuzione: giro (0..), t lungo il percorso, esito del test di quel giro. */
    loop(t, passed, total = 5) {
      alongPath(loopPath, t, tmp);
      ball.position.set(tmp.x, tmp.y, 0.6);
      ball.visible = t > 0;
      const stage = t < 0.2 ? 0 : t < 0.4 ? 1 : t < 0.6 ? 2 : 0;
      cards.forEach((c, i) => c.set(t > 0 && i === stage ? 0.8 : 0));
      status.set(statusLines(passed, total));
    },
  };
  api.fim(1);
  api.loop(0, 5);
  return api;
}
