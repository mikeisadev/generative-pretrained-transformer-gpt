/**
 * Albero dei moduli dai nomi puntati dei parametri, come li scrive PyTorch:
 *
 *   blocks.0.sa.heads.2.key.weight  →  GPTLanguageModel → blocks → [0] Block → sa MultiHeadAttention
 *                                       → heads → [2] Head → key nn.Linear → weight (16, 64)
 *
 * Ogni nodo ha i conteggi (parametri e tensori sotto di lui, quota sul totale). Con gli snippet
 * di data/model.json (codice vero di gpt/*.py) i nodi dell'architettura del corso ricevono
 * l'annotazione `code`: la chiave dello snippet, il file, il numero di riga e la riga Python che
 * crea quel modulo o quel peso (`self.key = nn.Linear(config.n_embd, self.head_size, bias=False)`).
 * Per un run con l'architettura modificata l'albero si costruisce lo stesso dai nomi: i nodi che
 * il codice del corso non spiega restano senza `code`, e `codeMatch: false` segnala un parametro
 * che quella riga non creerebbe (es. un `key.bias` quando il codice dice `bias=False`).
 */

import { expectedParameters } from "../../engine/weights.js";
import { moduleSources, parseModuleSource } from "./pythonSource.js";

/** Il modello del corso: il costruttore sta nello snippet "model.init" (gpt/export.py, SNIPPETS). */
export const ROOT_CLASS = "GPTLanguageModel";
export const ROOT_SNIPPET = "model.init";

/** Parametri che i moduli di PyTorch creano, in base agli argomenti della riga che li costruisce. */
const CREATES = {
  "nn.Linear": (args) => (/\bbias\s*=\s*False\b/.test(args) ? ["weight"] : ["weight", "bias"]),
  "nn.LayerNorm": (args) => (/\belementwise_affine\s*=\s*False\b/.test(args) ? [] : /\bbias\s*=\s*False\b/.test(args) ? ["weight"] : ["weight", "bias"]),
  "nn.Embedding": () => ["weight"],
};

/** Che cosa è ogni parametro dei moduli standard (semantica di PyTorch). */
const ROLES = {
  "nn.Linear": { weight: "matrice W (uscite × ingressi): y = x · Wᵀ + b", bias: "bias b: un numero per ogni uscita" },
  "nn.LayerNorm": { weight: "γ (gamma): riscala ogni dimensione dopo la normalizzazione", bias: "β (beta): sposta ogni dimensione dopo la normalizzazione" },
  "nn.Embedding": { weight: "tabella: una riga di numeri appresi per ogni indice" },
};

const isIndex = (key) => /^\d+$/.test(key);

/**
 * @param {Iterable<{name: string, shape: number[], numel?: number, dtype?: string}>|Map} params
 *   i parametri in ordine (es. `run.params`, `run.tensors` o il layout di weights.json)
 * @param {{snippets?: Record<string, object>|null, config?: object|null}} [o]
 *   `snippets`: `spec.snippets` di model.json; `config`: per confrontare le forme con quelle del corso
 * @returns {object} radice `{path: "", key: "", label, pyPath, kind: "module", children, params, tensors, share, code?}`
 *   più `architecture` ("corso" | "modificata" | "sconosciuta"), `unexplained` (parametri senza
 *   annotazione) e `mismatched` (parametri che il codice non creerebbe o con forma diversa)
 */
export function buildModuleTree(params, { snippets = null, config = null } = {}) {
  const list = params instanceof Map ? [...params.values()] : [...params];
  const root = node("", "", "", "module");
  const byPath = new Map([["", root]]);
  for (const p of list) {
    const parts = String(p.name).split(".");
    let parent = root;
    for (let i = 0; i < parts.length; i++) {
      const path = parts.slice(0, i + 1).join(".");
      let child = byPath.get(path);
      const leaf = i === parts.length - 1;
      if (!child) {
        child = node(path, parts[i], pyPathOf(parent, parts[i]), leaf ? "param" : "module");
        parent.children.push(child);
        byPath.set(path, child);
      } else if (leaf || child.kind === "param") {
        throw new Error(`nome di parametro in conflitto con un modulo: ${p.name}`);
      }
      parent = child;
    }
    const numel = p.numel ?? p.shape.reduce((a, b) => a * b, 1);
    Object.assign(parent, { shape: [...p.shape], numel, dtype: p.dtype ?? null });
  }
  count(root);
  const total = root.params;
  walkTree(root, (n) => (n.share = total ? n.params / total : 0));

  root.label = "modello";
  root.architecture = "sconosciuta";
  root.unexplained = [];
  root.mismatched = [];
  if (snippets?.[ROOT_SNIPPET]) annotate(root, snippets, config);
  return root;
}

function node(path, key, pyPath, kind) {
  return { path, key, label: isIndex(key) ? `[${key}]` : key, pyPath, kind, index: isIndex(key), children: [], params: 0, tensors: 0, share: 0 };
}

function pyPathOf(parent, key) {
  if (isIndex(key)) return `${parent.pyPath}[${key}]`;
  return parent.pyPath ? `${parent.pyPath}.${key}` : key;
}

function count(n) {
  if (n.kind === "param") {
    n.params = n.numel;
    n.tensors = 1;
    return;
  }
  for (const c of n.children) count(c);
  n.params = n.children.reduce((s, c) => s + c.params, 0);
  n.tensors = n.children.reduce((s, c) => s + c.tensors, 0);
}

/** Visita in profondità (prima il nodo, poi i figli nell'ordine). */
export function walkTree(root, fn, depth = 0) {
  fn(root, depth);
  for (const c of root.children) walkTree(c, fn, depth + 1);
}

/** Il nodo con quel nome puntato ("" = radice), o null. */
export function findNode(root, path) {
  if (!path) return root;
  let n = root;
  for (const key of path.split(".")) {
    n = n.children.find((c) => c.key === key);
    if (!n) return null;
  }
  return n;
}

/** `blocks.0.sa.heads.2.key.weight` → `blocks[0].sa.heads[2].key.weight` (come si scrive in Python). */
export function pythonPath(name) {
  return String(name)
    .split(".")
    .reduce((acc, key) => (isIndex(key) ? `${acc}[${key}]` : acc ? `${acc}.${key}` : key), "");
}

// ---------------------------------------------------------------- annotazioni dal codice

function codeOf(source, stmt, extra = {}) {
  return { snippet: source.key, file: source.file, line: stmt.line, text: stmt.text, comment: stmt.comment || null, ...extra };
}

/** Che tipo di contesto apre l'istruzione `self.x = …` per i nomi che seguono. */
function contextOf(stmt, source, classes) {
  if (classes.has(stmt.callee)) return { kind: "class", source: classes.get(stmt.callee) };
  if ((stmt.callee === "nn.ModuleList" || stmt.callee === "nn.Sequential") && stmt.element) return { kind: "list", element: stmt.element, stmt, source };
  if (stmt.callee === "nn.Sequential" && stmt.items) return { kind: "sequence", items: stmt.items, stmt, source };
  if (stmt.callee) return { kind: "leaf", callee: stmt.callee, args: stmt.args ?? "", stmt, source };
  return null;
}

function annotate(root, snippets, config) {
  const classes = moduleSources(snippets);
  const rootSource = parseModuleSource(ROOT_SNIPPET, snippets[ROOT_SNIPPET]);
  root.label = ROOT_CLASS;
  // La radice non è creata da una riga del modello: il suo codice è il costruttore __init__.
  const initLine = String(snippets[ROOT_SNIPPET].code ?? "").split("\n")[0].trim();
  root.code = { snippet: ROOT_SNIPPET, file: rootSource.file, line: rootSource.startLine, text: initLine, comment: null, type: ROOT_CLASS, className: ROOT_CLASS };
  let expected = null;
  if (config) {
    try {
      expected = new Map(expectedParameters(config).map((p) => [p.name, p.shape]));
    } catch {
      expected = null; // config non valida per il GPT del corso: niente confronto delle forme
    }
  }
  visit(root, { kind: "class", source: rootSource }, classes);

  let explained = 0;
  let leaves = 0;
  walkTree(root, (n) => {
    if (n.kind !== "param") return;
    leaves++;
    if (n.code) explained++;
    else root.unexplained.push(n.path);
    if (expected) {
      const shape = expected.get(n.path);
      n.expectedShape = shape ?? null;
      n.shapeMatch = shape ? shape.length === n.shape.length && shape.every((v, i) => v === n.shape[i]) : null;
    }
    if (n.codeMatch === false || n.shapeMatch === false) root.mismatched.push(n.path);
  });
  const sameSet = expected ? leaves === expected.size : true;
  root.architecture = explained === 0 ? "sconosciuta" : explained === leaves && root.mismatched.length === 0 && sameSet ? "corso" : "modificata";
}

function visit(n, ctx, classes) {
  for (const child of n.children) {
    let code = null;
    let next = null;
    if (ctx.kind === "class" && !child.index) {
      const stmt = ctx.source.attrs.get(child.key);
      if (stmt) {
        const isClass = classes.has(stmt.callee);
        code = codeOf(ctx.source, stmt, { type: stmt.callee, className: isClass ? stmt.callee : null });
        if (stmt.element) code.element = stmt.element;
        if (stmt.items) code.items = stmt.items.map((it) => ({ index: it.index, type: it.callee, line: it.line, text: it.text, comment: it.comment || null }));
        next = contextOf(stmt, ctx.source, classes);
        if (next?.kind === "class") addClassDetails(code, next.source);
      }
    } else if (ctx.kind === "list" && child.index) {
      const cls = classes.get(ctx.element);
      if (cls) {
        code = { snippet: cls.key, file: cls.file, line: cls.classLine.line, text: cls.classLine.text, comment: null, type: ctx.element, className: ctx.element, createdBy: codeOf(ctx.source, ctx.stmt) };
        addClassDetails(code, cls);
        next = { kind: "class", source: cls };
      }
    } else if (ctx.kind === "sequence" && child.index) {
      const item = ctx.items[Number(child.key)];
      if (item) {
        code = { snippet: ctx.source.key, file: ctx.source.file, line: item.line, text: item.text, comment: item.comment || null, type: item.callee, className: null, createdBy: codeOf(ctx.source, ctx.stmt) };
        next = item.callee ? { kind: "leaf", callee: item.callee, args: item.args ?? "", stmt: item, source: ctx.source } : null;
      }
    } else if (ctx.kind === "leaf" && child.kind === "param") {
      const creates = CREATES[ctx.callee]?.(ctx.args);
      code = { snippet: ctx.source.key, file: ctx.source.file, line: ctx.stmt.line, text: ctx.stmt.text, comment: ctx.stmt.comment || null, type: ctx.callee, param: child.key, role: ROLES[ctx.callee]?.[child.key] ?? null };
      child.codeMatch = creates ? creates.includes(child.key) : null;
    }
    if (code) child.code = code;
    if (next && child.kind === "module") visit(child, next, classes);
  }
}

/** Buffer e sotto-moduli senza pesi della classe (es. la maschera `tril`, i Dropout): utili da mostrare. */
function addClassDetails(code, source) {
  if (source.buffers.length) code.buffers = source.buffers.map((b) => ({ name: b.name, line: b.line, text: b.text, statement: b.statement }));
  const withoutWeights = [...source.attrs.values()].filter((s) => s.callee && /^nn\.(Dropout|ReLU|GELU|Identity)$/.test(s.callee));
  if (withoutWeights.length) code.noWeights = withoutWeights.map((s) => ({ attr: s.attr, type: s.callee, line: s.line, text: s.text }));
}
