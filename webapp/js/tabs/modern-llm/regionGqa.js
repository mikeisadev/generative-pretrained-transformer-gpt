/**
 * Regione "MHA, GQA, MQA": 8 head di query in alto, le head K/V in basso.
 * Le 8 istanze K/V esistono sempre: in GQA si sovrappongono a gruppi di 4, in MQA tutte in una.
 * Così il passaggio tra le modalità è un'interpolazione di posizioni, senza ricostruire nulla.
 */
import { LLAMA3_8B, kvBytesPerToken } from "./sim.js";
import { Cells, ease, Links, PALETTE, Plate, text } from "./shapes.js";

const N = 8;
const Y_Q = 4.6;
const Y_KV = -3.6;
const qX = (i) => -11.2 + i * 3.2;
const GROUP_COLORS = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb7185", "#22d3ee", "#f97316"];

/** Per ogni modalità: numero di head K/V (su 8 query) e x di arrivo della head K/V i. */
export const GQA_MODES = {
  mha: { kv: 8, title: "Multi-head (MHA): ogni query ha la sua coppia K/V", x: (i) => qX(i) },
  gqa: { kv: 2, title: "Grouped-query (GQA): 4 query condividono una coppia K/V", x: (i) => (i < 4 ? -6.4 : 6.4) },
  mqa: { kv: 1, title: "Multi-query (MQA): una sola coppia K/V per tutte", x: () => 0 },
};

/** KiB per token di Llama 3 8B (32 layer, head da 128, bf16) con 32, 8 o 1 head K/V. */
function cacheLine(mode) {
  const kib = (kv) => kvBytesPerToken({ ...LLAMA3_8B, kvHeads: kv }) / 1024;
  return [
    { text: "KV cache per token, architettura di Llama 3 8B (bf16):", color: PALETTE.muted },
    { text: `MHA (32 K/V): ${kib(32)} KiB   ·   GQA (8 K/V, la scelta vera): ${kib(8)} KiB   ·   MQA (1 K/V): ${kib(1)} KiB`, bold: true, color: mode === "gqa" ? "#bfdbfe" : PALETTE.text },
  ];
}

const CACHE_LINES = Object.fromEntries(Object.keys(GQA_MODES).map((m) => [m, cacheLine(m)]));

export function buildGqa(group) {
  const title = new Plate({ w: 34, h: 1.3, size: 0.72, align: "center", color: "#bfdbfe" }).place(group, 0, 8.6);
  text(group, "query (una per head)", -12.4, 6.65, { size: 0.42, color: PALETTE.muted });
  text(group, "K e V (finiscono nella KV cache)", -9.6, -5.9, { size: 0.42, color: PALETTE.muted });

  const queries = new Cells(N, { w: 2.2, h: 2.2, d: 0.6 });
  const kv = new Cells(N, { w: 2.6, h: 2.6, d: 0.7 });
  for (let i = 0; i < N; i++) {
    queries.place(i, qX(i), Y_Q, 0);
    kv.place(i, qX(i), Y_KV, 0);
  }
  group.add(queries.mesh, kv.mesh);
  for (let i = 0; i < N; i++) text(group, `q${i + 1}`, qX(i), Y_Q, { size: 0.5, z: 0.5 });
  const kvLabels = Array.from({ length: N }, (_, i) => text(group, "K V", qX(i), Y_KV, { size: 0.5, z: 0.6 }));

  const links = new Links(N);
  links.setCount(N);
  group.add(links.line);
  const memory = new Plate({ w: 34, h: 2.2, size: 0.59, align: "center" }).place(group, 0, -10.3);

  const api = {
    mode: "gqa",
    /** Interpolazione dalla modalità `from` a `to` (t in 0..1). */
    morph(from, to, t) {
      const a = GQA_MODES[from];
      const b = GQA_MODES[to];
      const k = ease(t);
      const target = k < 0.5 ? from : to;
      const groupsOf = GQA_MODES[target].kv;
      for (let i = 0; i < N; i++) {
        const x = a.x(i) + (b.x(i) - a.x(i)) * k;
        kv.place(i, x, Y_KV, 0);
        kvLabels[i].position.x = x;
        const g = Math.floor((i * groupsOf) / N);
        // istanze sovrapposte: l'etichetta resta solo sulla prima del gruppo
        kvLabels[i].visible = (i * groupsOf) % N < groupsOf;
        const color = GROUP_COLORS[g % GROUP_COLORS.length];
        kv.color(i, color);
        queries.color(i, color);
        links.set(i, qX(i), Y_Q - 1.1, 0, x, Y_KV + 1.3, 0);
        links.color(i, color);
      }
      api.mode = target;
      title.set(GQA_MODES[target].title);
      memory.set(CACHE_LINES[target]);
    },
    set(mode) {
      api.morph(mode, mode, 1);
    },
    /** Lampeggio delle head che leggono la stessa K/V (attenzione "condivisa"). */
    pulse(t) {
      kv.mesh.material.emissive.set("#ffffff");
      kv.mesh.material.emissiveIntensity = 0.25 * Math.max(0, Math.sin(t * Math.PI));
    },
  };
  api.set("gqa");
  return api;
}
