/**
 * Lettura del codice Python dei moduli del GPT dagli snippet di data/model.json (chiave
 * `snippets`, esportati da gpt/export.py con `inspect` dai file gpt/*.py: un test Python
 * controlla che coincidano con il sorgente). Così la riga che crea un peso, per esempio
 * `self.key = nn.Linear(config.n_embd, self.head_size, bias=False)`, è letta dal codice vero.
 *
 * Non è un parser Python generale: riconosce le assegnazioni `self.<nome> = …` dentro
 * `__init__` (anche su più righe), i loro argomenti, gli elementi di un `nn.Sequential(…)`
 * scritti uno per riga, le liste per comprensione `[Block(config) for _ in …]` e i
 * `self.register_buffer("nome", …)`. Basta per i moduli del corso e per quelli scritti nello
 * stesso stile; ciò che non riconosce resta semplicemente senza annotazione.
 */

/**
 * Separa il codice dal commento `# …` di una riga, ignorando i `#` dentro le stringhe.
 * @returns {{code: string, comment: string}}
 */
export function splitComment(line) {
  let quote = null;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quote) {
      if (ch === "\\") i++;
      else if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (ch === "#") return { code: line.slice(0, i).trimEnd(), comment: line.slice(i + 1).trim() };
  }
  return { code: line.trimEnd(), comment: "" };
}

const indentOf = (line) => line.length - line.trimStart().length;

/** Un'istruzione su più righe scritta su una riga: `f(\n    a, b\n)` → `f(a, b)`. */
function oneLine(code) {
  return code
    .replace(/([([{])\s*\n\s*/g, "$1")
    .replace(/,?\s*\n\s*([)\]}])/g, "$1")
    .replace(/\s*\n\s*/g, " ")
    .trim();
}
const OPEN = "([{";
const CLOSE = ")]}";

/** Variazione della profondità delle parentesi su un pezzo di codice (stringhe escluse). */
function depthDelta(code) {
  let depth = 0;
  let quote = null;
  for (let i = 0; i < code.length; i++) {
    const ch = code[i];
    if (quote) {
      if (ch === "\\") i++;
      else if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (OPEN.includes(ch)) depth++;
    else if (CLOSE.includes(ch)) depth--;
  }
  return depth;
}

/**
 * Chiamata `callee(args…)` all'inizio di `text` (posizioni relative a `text`).
 * @returns {{callee: string, argsStart: number, argsEnd: number}|null} argsEnd = indice della ")"
 */
function parseCall(text) {
  const m = /^\s*([A-Za-z_][\w.]*)\s*\(/.exec(text);
  if (!m) return null;
  const argsStart = m[0].length;
  let depth = 1;
  let quote = null;
  for (let i = argsStart; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === "\\") i++;
      else if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") quote = ch;
    else if (OPEN.includes(ch)) depth++;
    else if (CLOSE.includes(ch) && --depth === 0) return { callee: m[1], argsStart, argsEnd: i };
  }
  return null;
}

/** Divide gli argomenti alle virgole di primo livello; ogni pezzo con la sua posizione iniziale. */
function splitArgs(text, start, end) {
  const out = [];
  let depth = 0;
  let quote = null;
  let from = start;
  for (let i = start; i <= end; i++) {
    const ch = i < end ? text[i] : ",";
    if (quote) {
      if (ch === "\\") i++;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") quote = ch;
    else if (OPEN.includes(ch)) depth++;
    else if (CLOSE.includes(ch)) depth--;
    else if (ch === "," && depth === 0) {
      const raw = text.slice(from, i);
      const lead = raw.length - raw.trimStart().length;
      if (raw.trim()) out.push({ text: raw.trim(), at: from + lead });
      from = i + 1;
    }
  }
  return out;
}

/**
 * Legge uno snippet `{file, start_line, code}` di model.json.
 * @param {string} key chiave dello snippet (es. "head")
 * @param {{file: string, start_line: number, code: string}} snippet
 * @returns {{key: string, file: string, startLine: number, className: string|null,
 *   classLine: {text: string, line: number}|null, attrs: Map<string, object>, buffers: object[]}}
 *   ogni attributo: `{attr, line, text, code, comment, statement, callee, args, items?, element?}`
 *   con `line` numero di riga nel file (1 = prima riga), `text` la riga com'è nel sorgente (senza
 *   indentazione), `statement` il codice dell'istruzione intera (anche su più righe, senza commenti)
 */
export function parseModuleSource(key, snippet) {
  const lines = String(snippet.code ?? "").split("\n");
  const startLine = Number(snippet.start_line) || 1;
  const first = lines[0] ?? "";
  const classMatch = /^class\s+([A-Za-z_]\w*)\s*[(:]/.exec(first.trim());
  const out = {
    key,
    file: snippet.file,
    startLine,
    className: classMatch ? classMatch[1] : null,
    classLine: classMatch ? { text: first.trim(), line: startLine } : null,
    attrs: new Map(),
    buffers: [],
  };
  // Corpo di __init__: le righe più indentate della riga "def __init__(".
  let from = 0;
  let to = lines.length;
  const def = lines.findIndex((l) => /^\s*def\s+__init__\s*\(/.test(l));
  if (def < 0) return out;
  from = def + 1;
  const defIndent = indentOf(lines[def]);
  for (let i = from; i < lines.length; i++) {
    if (lines[i].trim() && indentOf(lines[i]) <= defIndent) {
      to = i;
      break;
    }
  }
  for (let i = from; i < to; i++) {
    const { code, comment } = splitComment(lines[i]);
    const assign = /^\s*self\.([A-Za-z_]\w*)\s*=\s*(?!=)/.exec(code);
    const buffer = /^\s*self\.register_buffer\s*\(/.exec(code);
    if (!assign && !buffer) continue;
    // L'istruzione continua finché le parentesi non si chiudono.
    const parts = [{ code, comment, line: startLine + i, offset: 0 }];
    let depth = depthDelta(code);
    let j = i;
    let joined = code;
    while (depth > 0 && j + 1 < to) {
      j++;
      const next = splitComment(lines[j]);
      parts.push({ ...next, line: startLine + j, offset: joined.length + 1 });
      joined += `\n${next.code}`;
      depth += depthDelta(next.code);
    }
    const lineAt = (pos) => parts.reduce((found, p) => (p.offset <= pos ? p : found), parts[0]);
    const base = { line: startLine + i, text: lines[i].trim(), code: code.trim(), comment, statement: oneLine(joined) };
    if (buffer) {
      const name = /register_buffer\s*\(\s*(["'])([^"']+)\1/.exec(joined);
      out.buffers.push({ ...base, name: name ? name[2] : null });
      i = j;
      continue;
    }
    const rhsAt = assign[0].length;
    const rhs = joined.slice(rhsAt);
    const call = parseCall(rhs);
    const stmt = { ...base, attr: assign[1], callee: call?.callee ?? null, args: call ? rhs.slice(call.argsStart, call.argsEnd) : null };
    if (call) {
      const args = splitArgs(rhs, call.argsStart, call.argsEnd);
      const comprehension = args.length === 1 && /^\*?\[/.test(args[0].text) && /\sfor\s/.test(args[0].text);
      if (comprehension) {
        const element = /^\*?\[\s*([A-Za-z_][\w.]*)\s*\(/.exec(args[0].text);
        if (element) stmt.element = element[1];
      } else if (stmt.callee === "nn.Sequential") {
        stmt.items = args.map((arg, index) => {
          const where = lineAt(rhsAt + arg.at);
          const itemCall = parseCall(arg.text);
          const source = lines[where.line - startLine];
          return {
            index,
            callee: itemCall?.callee ?? null,
            args: itemCall ? arg.text.slice(itemCall.argsStart, itemCall.argsEnd) : null,
            code: arg.text,
            line: where.line,
            text: source.trim(),
            comment: where.comment,
          };
        });
      }
    }
    out.attrs.set(stmt.attr, stmt);
    i = j;
  }
  return out;
}

/**
 * Tutti gli snippet di model.json che definiscono un modulo: classi (`class Head(nn.Module):`)
 * per nome e il costruttore del modello (`model.init`, cioè GPTLanguageModel.__init__).
 * @param {Record<string, {file: string, start_line: number, code: string}>} snippets
 * @returns {Map<string, ReturnType<typeof parseModuleSource>>} nome della classe → sorgente letto
 */
export function moduleSources(snippets) {
  const out = new Map();
  for (const [key, snippet] of Object.entries(snippets ?? {})) {
    if (!snippet || typeof snippet.code !== "string") continue;
    const source = parseModuleSource(key, snippet);
    if (source.className && source.attrs.size) out.set(source.className, source);
  }
  return out;
}
