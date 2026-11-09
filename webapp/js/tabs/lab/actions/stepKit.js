/**
 * Dai passi "dichiarati" delle azioni ai passi dell'ActionPlayer del nucleo.
 *
 * Un passo dichiara il suo stato di partenza per intero (carte, veli, lavagna, codice, inquadratura):
 * enter() lo ricostruisce sempre da capo, così avanti, indietro e "⟲ Ricomincia" danno la stessa scena.
 *   {
 *     title, caption, duration,
 *     scene()      carte e veli del passo (stage.card, stage.highlight, stage.link)
 *     board()      blocchi della lavagna (vedi board.js)
 *     code         [chiave dello snippet, titolo, [righe da evidenziare]]
 *     frame()      rettangolo del mondo da inquadrare (plan.js), o null = la camera resta
 *     update(p)    comparsa progressiva (p da 0 a 1); con "riduci il movimento" si salta alla fine
 *   }
 * Ogni enter e ogni update si misurano (stage.stats): il costo per passo delle prove.
 */

/** I passi del player da quelli dichiarati. */
export function makeSteps(stage, specs, { id }) {
  return specs.filter(Boolean).map((spec, index) => ({
    title: spec.title,
    caption: spec.caption,
    duration: spec.duration ?? 4,
    enter: () => {
      const t0 = performance.now();
      stage.step();
      spec.scene?.();
      if (spec.board) stage.board.show(spec.board());
      if (spec.code) stage.code(...spec.code);
      if (spec.reduced !== false && stage.reduced) spec.update?.(1);
      else spec.update?.(0);
      const rect = spec.frame?.();
      if (rect) stage.frameSoon(rect, spec.frameOptions);
      else stage.syncLayout();
      stage.noteEnter(id, index, performance.now() - t0);
    },
    update: (p) => {
      if (!spec.update || (spec.reduced !== false && stage.reduced)) return;
      const t0 = performance.now();
      spec.update(p);
      stage.noteUpdate(performance.now() - t0);
    },
  }));
}

/**
 * Un contatore che cambia solo quando cambia il suo valore intero: le comparse progressive ridisegnano
 * una carta (o la lavagna) solo quando c'è davvero una cella nuova, non a ogni frame.
 */
export function onChange(fn) {
  let last = null;
  return (value) => {
    if (value === last) return;
    last = value;
    fn(value);
  };
}

/** p ∈ [0, 1] → intero in [0, n] (la quota di elementi scoperti), con una piccola pausa alla fine. */
export function progressCount(p, n, { hold = 0.15 } = {}) {
  return Math.min(n, Math.floor((Math.min(1, p / (1 - hold)) * n)));
}
