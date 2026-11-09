/**
 * Voci della ricerca globale delle azioni del Laboratorio (fase 2). SOLI DATI: search.js del tab lo
 * importa anche quando il tab non è mai stato aperto, quindi qui niente three.js, scena o index.js.
 * Voce: { id, kind: "action", title, text?, keywords? } con id = quello di ui.player.register.
 */
import { ACTIONS, CATEGORIES } from "./catalog.js";

export const searchEntries = ACTIONS.map((a) => ({
  id: a.id,
  kind: "action",
  title: a.title,
  text: a.summary,
  keywords: [...a.keywords, CATEGORIES[a.category], "laboratorio pesi", "animazione"],
}));
