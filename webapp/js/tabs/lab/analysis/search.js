/**
 * Voci della ricerca globale di confronto, grafici e benchmark del Laboratorio (fase 2). SOLI DATI:
 * search.js del tab lo importa anche quando il tab non è mai stato aperto (niente three.js né scena).
 * Voce: { id, kind: "view" | "action", title, text?, keywords? }; gli id "view" li apre tab.views.open(id)
 * (registrati con tab.views.register, vedi lab.md).
 */
export const searchEntries = [];
