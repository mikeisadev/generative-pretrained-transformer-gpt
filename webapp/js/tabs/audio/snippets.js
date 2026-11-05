/**
 * Codice mostrato nel pannello a destra, sempre VERO:
 * - "py:<chiave>" → frammenti dei progetti originali (Whisper, EnCodec, AudioCraft) in upstream.js
 * - "js:<file>#<nome>" → il codice JavaScript di questo tab che ha appena fatto il calcolo,
 *   letto dal file sorgente stesso (così non può divergere da ciò che gira).
 */
import { PY_SNIPPETS } from "./upstream.js";

const sources = new Map();

async function sourceOf(file) {
  if (!sources.has(file)) sources.set(file, fetch(new URL(`./${file}`, import.meta.url)).then((r) => r.text()));
  return sources.get(file);
}

/**
 * Estrae una funzione, una classe o un metodo dal sorgente, con il commento JSDoc che la precede.
 * Il corpo finisce alla parentesi graffa che chiude quella di apertura.
 */
export function extractDefinition(text, name) {
  const patterns = [
    new RegExp(`^export (async )?function ${name}\\b`, "m"),
    new RegExp(`^(export )?class ${name}\\b`, "m"),
    new RegExp(`^(async )?function ${name}\\b`, "m"),
    new RegExp(`^  (async )?${name}\\(`, "m"),
    new RegExp(`^export const ${name}\\b`, "m"),
  ];
  let start = -1;
  for (const p of patterns) {
    const m = p.exec(text);
    if (m) {
      start = m.index;
      break;
    }
  }
  if (start < 0) return null;
  // il commento /** … */ subito sopra fa parte della spiegazione
  const before = text.slice(0, start);
  const docStart = before.lastIndexOf("/**");
  const isDoc = docStart >= 0 && /^\/\*\*(?:(?!\*\/)[\s\S])*\*\/\s*$/.test(before.slice(docStart));
  let from = isDoc ? docStart : start;
  from = text.lastIndexOf("\n", from - 1) + 1;
  // i parametri possono contenere graffe (destrutturazione { a, b } = {}): si saltano le tonde
  let bodyFrom = start;
  const paren = text.indexOf("(", start);
  const brace = text.indexOf("{", start);
  if (paren >= 0 && (brace < 0 || paren < brace)) {
    let p = 0;
    for (let i = paren; i < text.length; i++) {
      if (text[i] === "(") p++;
      else if (text[i] === ")" && --p === 0) {
        bodyFrom = i + 1;
        break;
      }
    }
  }
  let depth = 0;
  let end = text.length;
  let opened = false;
  for (let i = bodyFrom; i < text.length; i++) {
    const c = text[i];
    if (c === "{") (depth++, (opened = true));
    else if (c === "}") depth--;
    else if (c === ";" && !opened && depth === 0) {
      end = i + 1;
      break;
    }
    if (opened && depth === 0) {
      end = i + 1;
      break;
    }
  }
  const startLine = text.slice(0, from).split("\n").length;
  return { code: text.slice(from, end).replace(/^\n+/, ""), startLine };
}

/**
 * Mostra un frammento nel CodePanel.
 * @param {import("../../core/ui/codePanel.js").CodePanel} panel
 * @param {string} ref "py:whisper.logmel" oppure "js:dsp.js#stft"
 * @param {string[]} [highlight]
 * @param {string} [title] titolo alternativo
 */
export async function showSnippet(panel, ref, highlight = [], title) {
  if (!panel || !ref) return;
  if (ref.startsWith("py:")) {
    const s = PY_SNIPPETS[ref.slice(3)];
    if (!s) return;
    panel.show({ title: title ?? s.title, file: s.file, startLine: s.start, code: s.code, language: s.language, highlight });
    return;
  }
  const [file, name] = ref.slice(3).split("#");
  const found = extractDefinition(await sourceOf(file), name);
  if (!found) return;
  panel.show({ title: title ?? `${name} (questo tab)`, file: `webapp/js/tabs/audio/${file}`, startLine: found.startLine, code: found.code, language: "javascript", highlight });
}
