/**
 * Regione "Mixture of Experts": token → router → 8 esperti (+ esperto condiviso alla DeepSeek).
 * Le probabilità del router sono barre (InstancedMesh), i collegamenti un solo LineSegments.
 */
import { card, Cells, Links, PALETTE, Plate, text, TokenStrip } from "./shapes.js";

export const MOE_TOKENS = ["Il", "gatto", "dorme", "sul", "divano", "."];
const N_EXPERTS = 8;
const X_TOKENS = -14.2;
const X_ROUTER = -7.6;
const X_EXPERT = 3.2;
const X_BAR = 7.2;
const BAR_MAX = 6.5;
const expertY = (e) => 8.2 - e * 2.3;
const tokenY = (i) => 6.5 - i * 2.6;

export function buildMoe(group) {
  text(group, "token", X_TOKENS, 9.4, { size: 0.5, color: PALETTE.muted });
  const tokens = MOE_TOKENS.map((t, i) => {
    const strip = new TokenStrip({ w: 4.2, h: 1.6, font: 0.62, align: "center", mono: false }).place(group, X_TOKENS, tokenY(i));
    strip.set([{ text: t }]);
    return strip;
  });

  const router = card(group, { x: X_ROUTER, y: 0.4, w: 5.4, h: 3.4, title: "Router", sub: "gate: 1 Linear", color: PALETTE.moe, size: 0.55 });
  const output = card(group, { x: X_ROUTER, y: -8.2, w: 6.2, h: 2.6, title: "Somma pesata", sub: "Σ peso × esperto(x)", color: PALETTE.ok, size: 0.45 });
  const experts = Array.from({ length: N_EXPERTS }, (_, e) => card(group, { x: X_EXPERT, y: expertY(e), w: 5.6, h: 1.85, title: `Esperto ${e + 1}`, sub: "MLP SwiGLU", color: PALETTE.moe, size: 0.38 }));
  const shared = card(group, { x: X_EXPERT, y: -10.6, w: 8.4, h: 1.7, title: "Esperto condiviso", sub: "sempre attivo (DeepSeek-V3)", color: PALETTE.warn, size: 0.36 });
  shared.mesh.visible = shared.label.visible = false;

  text(group, "probabilità del router", X_BAR + 3.2, 9.6, { size: 0.42, color: PALETTE.muted });
  const bars = new Cells(N_EXPERTS, { w: 1, h: 1.2, d: 0.25 });
  for (let e = 0; e < N_EXPERTS; e++) bars.place(e, X_BAR, expertY(e), 0, 0, 1, 1);
  group.add(bars.mesh);
  const percents = Array.from({ length: N_EXPERTS }, (_, e) => new Plate({ w: 2.8, h: 1, size: 0.65, align: "right", mono: true }).place(group, 16.2, expertY(e)));

  // collegamenti: token → router (1), router → esperti (8), esperti scelti → somma (2)
  const links = new Links(N_EXPERTS + 3);
  group.add(links.line);
  for (let e = 0; e < N_EXPERTS; e++) links.set(1 + e, X_ROUTER + 2.7, 0.4, 0.2, X_EXPERT - 2.8, expertY(e), 0.2);
  links.setCount(N_EXPERTS + 3);

  const state = { token: -1, route: undefined, shown: null };
  const api = {
    tokens,
    experts,
    shared,
    router,
    output,
    /** Evidenzia il token i e lo collega al router. */
    setToken(i) {
      state.token = i;
      tokens.forEach((s, k) => s.set([{ text: MOE_TOKENS[k], fill: k === i ? "#4c1d95" : "#1e293b", stroke: k === i ? "#c4b5fd" : "#475569" }]));
      if (i >= 0) links.set(0, X_TOKENS + 2.1, tokenY(i), 0.2, X_ROUTER - 2.7, 0.4, 0.2);
      links.color(0, i >= 0 ? "#c4b5fd" : "#000000");
      router.set(i >= 0 ? 0.6 : 0);
    },
    /**
     * Mostra la decisione del router: barre delle probabilità (cresciute di t), esperti scelti accesi
     * e collegamenti verso la somma pesata quando t > 0.6.
     */
    setRoute(route, t = 1) {
      const chosen = route?.chosen ?? [];
      for (let e = 0; e < N_EXPERTS; e++) {
        const p = route ? route.probs[e] : 0;
        const len = Math.max(0.0001, p * BAR_MAX * t);
        bars.place(e, X_BAR + len / 2, expertY(e), 0, len, 1, 1);
        const on = chosen.includes(e);
        bars.color(e, on ? "#c4b5fd" : "#475569");
        experts[e].set(on ? Math.min(1, t * 1.5) : 0);
        links.color(1 + e, on ? "#c4b5fd" : route ? "#334155" : "#1e293b");
      }
      const shown = t > 0.6 && chosen.length > 0;
      if (route === state.route && shown === state.shown) return; // il resto non cambia con t
      state.route = route;
      state.shown = shown;
      percents.forEach((plate, e) => plate.set([{ text: route ? `${(route.probs[e] * 100).toFixed(0)}%` : "", color: chosen.includes(e) ? "#ede9fe" : "#64748b", bold: chosen.includes(e) }]));
      chosen.forEach((e, k) => {
        links.set(1 + N_EXPERTS + k, X_EXPERT - 2.8, expertY(e), 0.25, X_ROUTER + 3.1, -8.2, 0.25);
        links.color(1 + N_EXPERTS + k, t > 0.6 ? "#4ade80" : "#000000");
      });
      if (!chosen.length) for (let k = 0; k < 2; k++) links.color(1 + N_EXPERTS + k, "#000000");
      output.set(t > 0.6 && chosen.length ? 0.7 : 0);
    },
    setShared(on) {
      shared.mesh.visible = shared.label.visible = on;
      shared.set(on ? 0.7 : 0);
    },
    reset() {
      api.setToken(-1);
      api.setRoute(null, 0);
      api.setShared(false);
    },
  };
  api.reset();
  return api;
}
