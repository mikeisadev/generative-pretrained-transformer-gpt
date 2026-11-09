/**
 * Caricamento dei run dal disco dello studente (DOM): una cartella (input con webkitdirectory), più file
 * (input multiple) o il trascinamento sulla scena o sul pannello. Il trascinamento di una cartella si
 * legge con webkitGetAsEntry (anche le sottocartelle): ogni file diventa un oggetto "tipo File" con il
 * percorso relativo, come quelli di webkitdirectory, così la libreria (js/shared/weights/files.js) li
 * riconosce nello stesso modo. Nessun file viene letto qui: solo elencato.
 */

/** Al più tanti file per un trascinamento (una cartella sbagliata, es. la home, non blocca la pagina). */
export const MAX_DROPPED_FILES = 5000;

/** Un input file nascosto, aperto da un pulsante vero (accessibile); `onFiles(FileList)` alla scelta. */
export function filePicker({ directory = false, onFiles }) {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.hidden = true;
  input.tabIndex = -1;
  if (directory) {
    input.webkitdirectory = true;
    input.setAttribute("webkitdirectory", "");
  } else {
    input.accept = ".json,.bin,.safetensors";
  }
  input.addEventListener("change", () => {
    const files = [...(input.files ?? [])];
    input.value = ""; // la stessa cartella scelta di nuovo deve ricaricare
    if (files.length) onFiles(files);
  });
  return input;
}

/** true se il trascinamento porta file (non testo o link). */
export function draggingFiles(e) {
  return Boolean(e.dataTransfer && [...(e.dataTransfer.types ?? [])].includes("Files"));
}

/**
 * I file di un evento drop, cartelle comprese (percorso relativo in `relativePath`).
 * @returns {Promise<{files: object[], truncated: boolean}>}
 */
export async function droppedFiles(dataTransfer) {
  const items = [...(dataTransfer?.items ?? [])].filter((it) => it.kind === "file");
  const entries = items.map((it) => it.webkitGetAsEntry?.() ?? null);
  if (!entries.some(Boolean)) return { files: [...(dataTransfer?.files ?? [])], truncated: false };
  const files = [];
  let truncated = false;
  const visit = async (entry, prefix) => {
    if (files.length >= MAX_DROPPED_FILES) {
      truncated = true;
      return;
    }
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      files.push(withPath(file, `${prefix}${entry.name}`));
      return;
    }
    if (entry.isDirectory) {
      const reader = entry.createReader();
      // readEntries restituisce i file a blocchi (100 alla volta in Chrome): si legge fino a un blocco vuoto
      for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await visit(child, `${prefix}${entry.name}/`);
        if (files.length >= MAX_DROPPED_FILES) {
          truncated = true;
          break;
        }
      }
    }
  };
  for (const entry of entries) if (entry) await visit(entry, "");
  return { files, truncated };
}

/** Un File con il suo percorso relativo (i File non si possono modificare: un piccolo involucro). */
function withPath(file, relativePath) {
  return {
    name: file.name,
    size: file.size,
    relativePath,
    text: () => file.text(),
    arrayBuffer: () => file.arrayBuffer(),
  };
}

/**
 * Rende `target` una zona di rilascio: evidenzia (classe `cls`) mentre si trascinano file sopra e
 * chiama `onDrop(dataTransfer)`. Senza preventDefault su dragover/drop il browser aprirebbe il file
 * lasciando la pagina. Restituisce la funzione che toglie i listener.
 */
export function dropZone(target, { onDrop, cls = "lab-dropping", indicator = null }) {
  let depth = 0;
  const on = (on) => {
    target.classList.toggle(cls, on);
    if (indicator) indicator.hidden = !on;
  };
  const enter = (e) => {
    if (!draggingFiles(e)) return;
    e.preventDefault();
    depth++;
    on(true);
  };
  const over = (e) => {
    if (!draggingFiles(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  };
  const leave = (e) => {
    if (!draggingFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) on(false);
  };
  const drop = (e) => {
    if (!draggingFiles(e)) return;
    e.preventDefault();
    e.stopPropagation(); // la zona più interna gestisce il rilascio una volta sola
    depth = 0;
    on(false);
    onDrop(e.dataTransfer);
  };
  target.addEventListener("dragenter", enter);
  target.addEventListener("dragover", over);
  target.addEventListener("dragleave", leave);
  target.addEventListener("drop", drop);
  return () => {
    target.removeEventListener("dragenter", enter);
    target.removeEventListener("dragover", over);
    target.removeEventListener("dragleave", leave);
    target.removeEventListener("drop", drop);
  };
}
