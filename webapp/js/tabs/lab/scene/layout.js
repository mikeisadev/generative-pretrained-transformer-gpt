/**
 * Disposizione della scena del Laboratorio pesi (logica pura: niente three.js, niente DOM; test in
 * tests/lab.test.mjs). Dall'albero dei moduli (js/shared/weights/moduleTree.js, con il codice vero di
 * gpt/*.py) ai rettangoli della scena, nello stesso annidamento del codice Python:
 *
 *   GPTLanguageModel → [token_embedding_table, position_embedding_table] → blocks → Block ×N → ln_f → lm_head
 *   Block → ln1 → sa (MultiHeadAttention: heads → Head ×H → key, query, value + buffer tril, proj) → ln2 → ffwd.net
 *
 * Regole (pensate per il GPT del corso, ma ricavate dai NOMI, così valgono anche per un'architettura
 * modificata: più blocchi, una FFN più larga, moduli nuovi):
 * - il flusso va da sinistra a destra nell'ordine dei parametri (= l'ordine di __init__, che nel
 *   codice del corso è l'ordine del passaggio in avanti); i moduli in parallelo vanno in colonna:
 *   le head (ModuleList chiamata heads/experts), le proiezioni gemelle di una head (key, query,
 *   value: stessi tipo e forma) e le tabelle di embedding iniziali (si sommano);
 * - ogni parametro è una matrice: 1 cella = 1 peso, un nn.Linear (out, in) alto `out` (le stesse
 *   convenzioni di core/visuals.js); il bias è una colonna accanto alle righe di W (b[r] si somma
 *   all'uscita r), γ e β delle LayerNorm sono righe (una per dimensione di x). I vettori sono
 *   "stirati" sul lato sottile (VECTOR_STRETCH) per vedersi e cliccarsi: resta una cella per peso;
 * - i moduli senza pesi del codice (nn.ReLU, nn.Dropout, il buffer tril) sono nodi a sé, dichiarati.
 *
 * Coordinate del mondo: x verso destra, y verso l'alto, z verso la camera (ogni livello di annidamento
 * un poco più avanti, così i contenitori si leggono a strati girando la camera). Tutto il modello è
 * centrato nell'origine.
 */

import { moduleSources, splitComment } from "../../../shared/weights/pythonSource.js";
import { withoutBidiControls } from "../../../shared/weights/validate.js";
import { roleOf } from "./roles.js";

/** Lato di una cella nel mondo: lo stesso CELL di core/visuals.js (il test lo controlla). */
export const CELL = 0.04;
/** I vettori (bias, γ, β) sono stirati sul lato sottile: una riga di 64 celle alta 3 celle. */
export const VECTOR_STRETCH = 3;
/** Oltre questo lato una matrice si disegna con celle più piccole (la scena resta navigabile). */
export const MAX_MATRIX_SIDE = 1024;
/** Avanzamento in z di ogni livello di annidamento. */
export const Z_STEP = 0.035;

const MONO_EM = 0.6; // larghezza di un carattere monospazio (in altezze del carattere)
const SANS_EM = 0.6; // stima prudente per il grassetto proporzionale
const PARAM_LABEL_H = 0.07;
const LABEL_GAP = 0.035;
const MIN_CODE_H = 0.04;

/** Spaziature per profondità di annidamento (0 = GPTLanguageModel): margine interno, distanza tra figli, titolo. */
const PAD = [0.7, 0.42, 0.34, 0.26, 0.2, 0.15, 0.11, 0.09];
const GAP = [1.1, 0.8, 0.6, 0.42, 0.3, 0.22, 0.16, 0.12];
const TITLE = [0.95, 0.5, 0.4, 0.28, 0.2, 0.15, 0.115, 0.095];
const at = (list, depth) => list[Math.min(depth, list.length - 1)];

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const PCT = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
export const fmtInt = (n) => INT.format(n);
export const fmtPct = (f) => `${PCT.format(f * 100)}%`;

/** "gpt/attention.py" → "attention.py". */
const baseName = (file) => String(file ?? "").split("/").pop();

/** La riga di codice da scrivere nella scena: l'istruzione senza il commento (il commento è nel pannello del codice). */
export function codeLine(code) {
  if (!code?.text) return "";
  return splitComment(String(code.text)).code;
}

/**
 * Tipo PyTorch da mostrare: dal codice del corso (className, type), dall'albero di run.json (classi
 * del codice attuale al momento dell'esportazione) o, se non si sa, null.
 */
function typeOf(node, classes) {
  return node.code?.className ?? node.code?.type ?? classes?.get(node.path) ?? null;
}

/** Nome del nodo come si scrive in Python: blocks[0], heads[2], net[1]. */
function titleOf(node, parent) {
  if (!node.path) return node.label === "modello" ? "modello" : node.label;
  return node.index ? `${parent?.key || parent?.label || ""}[${node.key}]` : node.key;
}

/** Forma (righe, colonne) da disegnare e orientamento dei vettori. */
function matrixOf(shape, vector = "row") {
  let rows;
  let cols;
  if (!shape?.length) [rows, cols] = [1, 1];
  else if (shape.length === 1) [rows, cols] = vector === "column" ? [shape[0], 1] : [1, shape[0]];
  else [rows, cols] = [shape.slice(0, -1).reduce((a, b) => a * b, 1), shape.at(-1)];
  const isVector = !shape || shape.length <= 1;
  const cell = CELL * Math.min(1, MAX_MATRIX_SIDE / Math.max(rows, cols));
  const cellW = isVector && vector === "column" ? cell * VECTOR_STRETCH : cell;
  const cellH = isVector && vector !== "column" ? cell * VECTOR_STRETCH : cell;
  return { rows, cols, cell, cellW, cellH, w: cols * cellW, h: rows * cellH, vector: isVector ? vector : null };
}

/** Etichetta di un parametro: "weight (16, 64)", "weight = γ (64)", "bias (256)". */
function paramLabel(node, parentType) {
  const shape = `(${node.shape.join(", ")})`;
  if (/LayerNorm|RMSNorm/.test(parentType ?? "") || node.code?.type === "nn.LayerNorm") {
    if (node.key === "weight") return `weight = γ ${shape}`;
    if (node.key === "bias") return `bias = β ${shape}`;
  }
  return `${node.key} ${shape}`;
}

// ---------------------------------------------------------------- dall'albero ai nodi della scena

function lnode(fields) {
  return { kind: "module", children: [], dir: "row", align: "center", depth: 0, code: null, params: 0, share: 0, title: "", type: null, role: "block", ...fields };
}

const isLeafModule = (n) => n.kind === "module" && n.children.length > 0 && n.children.every((c) => c.kind === "param");
const shapeKey = (n) => n.children.map((c) => `${c.key}:${(c.shape ?? []).join("x")}`).join("|");

/**
 * Figli di un modulo nell'ordine della scena, con i raggruppamenti (embedding iniziali, proiezioni
 * gemelle) e i nodi senza pesi presi dal codice (elementi di nn.Sequential, buffer, Dropout).
 */
function buildChildren(node, ctx) {
  const kids = node.children.map((c) => convert(c, node, ctx));
  const type = typeOf(node, ctx.classes);

  // Elementi di un nn.Sequential senza pesi (nn.ReLU, nn.Dropout): nodi a sé, al loro posto.
  const items = node.code?.items;
  if (items?.length) {
    for (const item of items) {
      if (kids.some((k) => k.key === String(item.index))) continue;
      kids.push(
        lnode({
          kind: "virtual",
          path: `${node.path}.${item.index}`,
          key: String(item.index),
          index: Number(item.index),
          title: `${node.key}[${item.index}]`,
          type: item.type,
          role: /ReLU|GELU|SiLU|Tanh|Sigmoid/.test(item.type ?? "") ? roleOf(node.path, type) : "block",
          code: { snippet: node.code.snippet, file: node.code.file, line: item.line, text: item.text },
          note: "nessun peso",
        }),
      );
    }
    kids.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  }

  // Le proiezioni gemelle di una head (key, query, value: stessi tipo e forma) stanno in colonna;
  // accanto, il buffer della classe (tril) e i suoi moduli senza pesi (Dropout).
  const leafKids = kids.filter((k) => k.kind === "module");
  const twins = leafKids.length >= 2 && leafKids.length === kids.length && leafKids.every((k) => k.leaf && k.shapeKey === leafKids[0].shapeKey && k.type === leafKids[0].type);
  const extras = [];
  // buffer e moduli senza pesi stanno nel sorgente della CLASSE (es. MultiHeadAttention in attention.py),
  // non nel file della riga che crea il modulo (block.py): il file giusto viene dagli snippet
  const cls = ctx.sources?.get(node.code?.className) ?? null;
  const where = { snippet: cls?.key ?? node.code?.snippet, file: cls?.file ?? node.code?.file };
  for (const b of node.code?.buffers ?? []) extras.push(bufferNode(node, b, ctx, where));
  for (const w of node.code?.noWeights ?? []) extras.push(noWeightNode(node, w, type, where));

  if (twins) {
    const column = lnode({ kind: "group", path: `${node.path}#parallel`, dir: "column", align: "start", children: kids });
    if (!extras.length) return [column];
    return [column, lnode({ kind: "group", path: `${node.path}#extra`, dir: "column", align: "start", children: extras })];
  }
  // Le tabelle di embedding all'inizio del modello si sommano: in colonna, una sopra l'altra.
  if (!node.path) {
    let i = 0;
    while (i < kids.length && kids[i].role === "embedding") i++;
    if (i >= 2) kids.splice(0, i, lnode({ kind: "group", path: "#embeddings", dir: "column", align: "start", role: "embedding", children: kids.slice(0, i) }));
  }
  // Negli altri moduli i nodi senza pesi vanno sotto l'ultimo figlio (es. il Dropout dopo proj in
  // MultiHeadAttention: self.dropout(self.proj(out))), così non allungano la fila.
  if (extras.length && kids.length) {
    const last = kids.pop();
    kids.push(lnode({ kind: "group", path: `${node.path}#tail`, dir: "column", align: "start", children: [last, ...extras] }));
    return kids;
  }
  return [...kids, ...extras];
}

function bufferNode(node, b, ctx, where) {
  const block = ctx.config?.block_size;
  // Il buffer tril del codice del corso: torch.tril(torch.ones(block_size, block_size)). Gli altri buffer
  // non si possono ricostruire dai pesi: restano un nodo con la riga di codice, senza celle.
  const isTril = b.name === "tril" && Number.isInteger(block) && block > 0 && /torch\.tril\(torch\.ones/.test(b.statement ?? "");
  const shape = isTril ? [block, block] : null;
  return lnode({
    kind: isTril ? "buffer" : "virtual",
    path: `${node.path}.${b.name}`,
    key: b.name,
    title: b.name,
    type: "buffer",
    role: "buffer",
    shape,
    matrix: shape ? matrixOf(shape) : null,
    label: shape ? `${b.name} (${shape.join(", ")}) · buffer, non è un peso` : `${b.name} · buffer`,
    // `text` = l'istruzione intera su una riga (per la scena); `first` = la sua prima riga nel file (per evidenziarla nel codice)
    code: { snippet: where.snippet, file: where.file, line: b.line, text: b.statement ?? b.text, first: b.text },
    note: "buffer: non è un peso, il training non lo cambia",
  });
}

function noWeightNode(node, w, parentType, where) {
  return lnode({
    kind: "virtual",
    path: `${node.path}.${w.attr}`,
    key: w.attr,
    title: w.attr,
    type: w.type,
    role: "block",
    code: { snippet: where.snippet, file: where.file, line: w.line, text: w.text },
    note: /Dropout/.test(w.type) ? "nessun peso (p = dropout della config)" : "nessun peso",
    parentType,
  });
}

function convert(node, parent, ctx) {
  const type = typeOf(node, ctx.classes);
  const role = node.kind === "param" ? roleOf(parent?.path ?? "", typeOf(parent ?? {}, ctx.classes)) : roleOf(node.path, type);
  if (node.kind === "param") {
    return lnode({
      kind: "param",
      path: node.path,
      key: node.key,
      title: node.key,
      node,
      role,
      shape: node.shape,
      params: node.params,
      share: node.share,
      code: node.code ?? null,
      label: paramLabel(node, typeOf(parent ?? {}, ctx.classes)),
      // un peso fuori da un modulo con soli pesi (es. un tensore senza punti nel nome) resta una matrice;
      // arrangeLeaf la ridispone (bias in colonna) quando sta in un nn.Linear
      matrix: matrixOf(node.shape, "row"),
    });
  }
  const l = lnode({ kind: "module", path: node.path, key: node.key, index: node.index ? Number(node.key) : undefined, title: titleOf(node, parent), type, node, role, params: node.params, share: node.share, code: node.code ?? null });
  l.leaf = isLeafModule(node);
  l.shapeKey = l.leaf ? shapeKey(node) : null;
  if (l.leaf) {
    l.children = node.children.map((c) => convert(c, node, ctx));
    arrangeLeaf(l, type);
  } else {
    l.children = buildChildren(node, ctx);
    // Le liste di moduli in parallelo (le head, gli esperti) vanno in colonna; le altre (blocks, net) in fila.
    l.dir = node.children.every((c) => c.index) && /head|expert/i.test(node.key) ? "column" : "row";
    if (l.dir === "column") l.align = "start";
  }
  return l;
}

/** Un modulo con soli pesi (nn.Linear, nn.LayerNorm, nn.Embedding…): come sono disposti i suoi parametri. */
function arrangeLeaf(l, type) {
  const params = l.children;
  const weight = params.find((p) => p.key === "weight" && p.shape?.length === 2);
  const bias = params.find((p) => p.key === "bias" && p.shape?.length === 1);
  if (weight && bias && bias.shape[0] === weight.shape[0] && params.length === 2) {
    // y = x · Wᵀ + b: il bias è una colonna accanto alle righe di W (b[r] va con la riga r)
    weight.matrix = matrixOf(weight.shape);
    bias.matrix = matrixOf(bias.shape, "column");
    l.children = [weight, bias];
    l.dir = "row";
    l.align = "top";
    return;
  }
  for (const p of params) p.matrix = matrixOf(p.shape, "row");
  l.children = params;
  l.dir = params.length === 1 ? "row" : "column";
  l.align = "start";
  l.leafType = type;
}

// ---------------------------------------------------------------- misure e posizioni

/**
 * Testo dell'intestazione di un contenitore: titolo, etichetta file:riga, riga di codice, conteggi.
 * La riga mostrata è quella che CREA il modulo: per un elemento di una lista per comprensione
 * (Block, Head) la riga della comprensione; il file:riga è sempre quello della riga mostrata.
 */
function header(n, total) {
  const title = n.kind === "virtual" ? `${n.title} · ${n.type ?? "?"}` : n.type ? `${n.title} · ${n.type}` : n.title;
  const created = n.kind === "module" && n.code?.createdBy && /^class\s/.test(String(n.code.text ?? ""));
  const shown = created ? n.code.createdBy : n.code;
  const tag = shown?.file && Number.isInteger(shown.line) ? `${baseName(shown.file)}:${shown.line}` : "";
  const code = shown ? codeLine(shown) : "";
  // un modulo che il codice del corso non crea (architettura modificata) lo dice nella sua intestazione
  const foreign = n.kind === "module" && n.path && !n.code ? "non nel codice del corso · " : "";
  const sub = n.kind === "virtual" ? n.note : `${foreign}${fmtInt(n.params)} parametri${n.path && total ? ` · ${fmtPct(n.share)}` : ""}`;
  return { title, tag, code, sub };
}

/** Larghezza minima del contenuto di un contenitore e quanto può allargarsi per il suo titolo. */
const MIN_W = [0, 4, 5, 2.6, 2.2, 1.8, 1.4, 1.2];
const WIDEN = 1.6;
const MIN_TITLE = 0.06;

function measure(n, depth, total) {
  n.depth = depth;
  if (n.kind === "param" || n.kind === "buffer") {
    const m = n.matrix;
    const labelW = n.label.length * MONO_EM * PARAM_LABEL_H;
    n.w = Math.max(m.w, labelW);
    n.h = PARAM_LABEL_H + LABEL_GAP + m.h;
    return;
  }
  if (n.kind === "group") {
    for (const c of n.children) measure(c, depth, total);
    sizeContent(n, at(GAP, depth));
    n.w = n.contentW;
    n.h = n.contentH;
    return;
  }
  for (const c of n.children) measure(c, depth + 1, total);
  const pad = at(PAD, depth);
  const h = header(n, total);
  if (n.kind === "virtual") {
    // Nodo senza pesi: compatto (titolo, riga di codice con file:riga, "nessun peso").
    const titleH = at(TITLE, depth) * 0.75;
    const codeText = h.code ? `${h.code}  ${h.tag}` : h.tag;
    n.header = { ...h, titleH, codeH: titleH * 0.62, subH: titleH * 0.62, tagInline: true };
    const w = Math.max(h.title.length * SANS_EM * titleH, codeText.length * MONO_EM * n.header.codeH, h.sub.length * SANS_EM * n.header.subH);
    n.w = w + 2 * pad;
    n.h = 2 * pad + titleH * 1.15 + n.header.codeH * 1.45 + n.header.subH * 1.2;
    n.innerW = w;
    n.headerH = n.h - 2 * pad;
    n.pad = pad;
    return;
  }
  sizeContent(n, at(GAP, depth + 1));
  const content = n.contentW;
  // Il titolo si rimpicciolisce quando il contenuto è stretto (fino a MIN_TITLE); il contenitore può
  // allargarsi un poco per lui (WIDEN), mai diventare molto più largo di quello che contiene.
  const titleChars = h.title.length + (h.tag ? h.tag.length * 0.7 + 3 : 0);
  const fullTitle = titleChars * SANS_EM * at(TITLE, depth);
  let inner = Math.max(content, Math.min(fullTitle, Math.max(content, at(MIN_W, depth)) * WIDEN));
  let titleH = Math.min(at(TITLE, depth), inner / (titleChars * SANS_EM));
  if (titleH < MIN_TITLE) {
    titleH = MIN_TITLE;
    inner = Math.max(inner, titleChars * SANS_EM * MIN_TITLE);
  }
  const subH = Math.min(titleH * 0.62, inner / Math.max(1, h.sub.length * SANS_EM));
  // La riga di codice si adatta alla larghezza (mai sotto MIN_CODE_H: allora si allarga il contenitore).
  let codeH = 0;
  if (h.code) {
    codeH = Math.max(MIN_CODE_H, Math.min(titleH * 0.62, inner / (h.code.length * MONO_EM)));
    inner = Math.max(inner, h.code.length * MONO_EM * codeH);
  }
  n.header = { ...h, titleH, subH, codeH };
  n.innerW = inner;
  n.headerH = titleH * 1.15 + (codeH ? codeH * 1.45 : 0) + subH * 1.25;
  n.w = inner + 2 * pad;
  n.h = pad + n.headerH + at(GAP, depth + 1) * 0.6 + n.contentH + pad;
  n.pad = pad;
}

function sizeContent(n, gap) {
  const kids = n.children;
  if (!kids.length) {
    n.contentW = 0;
    n.contentH = 0;
    return;
  }
  const sum = (f) => kids.reduce((s, k) => s + f(k), 0);
  const max = (f) => Math.max(...kids.map(f));
  const g = n.align === "top" ? Math.max(gap * 0.5, CELL * 2) : gap;
  n.gap = g;
  if (n.dir === "row") {
    n.contentW = sum((k) => k.w) + g * (kids.length - 1);
    n.contentH = max((k) => k.h);
  } else {
    n.contentW = max((k) => k.w);
    n.contentH = sum((k) => k.h) + g * (kids.length - 1);
  }
}

/** Posiziona `n` con l'angolo in alto a sinistra in (x, top) (top = y del bordo alto, y verso l'alto). */
function place(n, x, top, out) {
  n.x0 = x;
  n.x1 = x + n.w;
  n.y1 = top;
  n.y0 = top - n.h;
  n.cx = (n.x0 + n.x1) / 2;
  n.cy = (n.y0 + n.y1) / 2;
  n.z = n.depth * Z_STEP + (n.kind === "param" || n.kind === "buffer" ? Z_STEP * 0.5 : 0);
  if (n.kind === "param" || n.kind === "buffer") {
    const m = n.matrix;
    m.x0 = x;
    m.y1 = top - PARAM_LABEL_H - LABEL_GAP;
    m.x1 = m.x0 + m.w;
    m.y0 = m.y1 - m.h;
    m.cx = (m.x0 + m.x1) / 2;
    m.cy = (m.y0 + m.y1) / 2;
    n.labelBox = { x0: x, y1: top, h: PARAM_LABEL_H, w: n.w };
    out.push(n);
    return;
  }
  let cx = x;
  let cy = top;
  if (n.kind === "module") {
    cx = x + n.pad;
    cy = top - n.pad - n.headerH - at(GAP, n.depth + 1) * 0.6;
    n.headerBox = { x0: x + n.pad, y1: top - n.pad, w: n.innerW, h: n.headerH };
  } else if (n.kind === "virtual") {
    n.headerBox = { x0: x + at(PAD, n.depth), y1: top - at(PAD, n.depth), w: n.w - 2 * at(PAD, n.depth), h: n.h - 2 * at(PAD, n.depth) };
  }
  out.push(n);
  const kids = n.children;
  if (!kids.length) return;
  if (n.dir === "row") {
    let xx = cx;
    for (const k of kids) {
      const dy = n.align === "center" ? (n.contentH - k.h) / 2 : 0;
      place(k, xx, cy - dy, out);
      xx += k.w + n.gap;
    }
  } else {
    let yy = cy;
    for (const k of kids) {
      place(k, cx, yy, out);
      yy -= k.h + n.gap;
    }
  }
}

/**
 * La disposizione completa della scena.
 * @param {object} tree radice di buildModuleTree (con `code` se sono stati passati gli snippet)
 * @param {{config?: object, classes?: Map<string, string>, snippets?: object}} [o] `config` per il buffer tril
 *   (block_size); `classes`: percorso → classe Python (da run.json `modules`) per i moduli che il codice del
 *   corso non spiega; `snippets` (data/model.json) per il file delle classi (buffer e moduli senza pesi)
 * @returns {{root, nodes: Map<string, object>, params: object[], boxes: object[], bounds, total: number}}
 */
export function layoutModel(tree, { config = null, classes = null, snippets = null } = {}) {
  let sources = null;
  try {
    sources = snippets ? moduleSources(snippets) : null;
  } catch {
    sources = null; // snippet illeggibili: i nodi senza pesi tengono il file del modulo che li contiene
  }
  const ctx = { config, classes, sources };
  const root = convert(tree, null, ctx);
  root.title = tree.label;
  root.type = null;
  root.role = "block";
  measure(root, 0, tree.params);
  const boxes = [];
  place(root, -root.w / 2, root.h / 2, boxes);
  const nodes = new Map();
  for (const b of boxes) if (b.kind !== "group") nodes.set(b.path, b);
  const params = boxes.filter((b) => b.kind === "param");
  return { root, nodes, params, boxes, bounds: { x0: root.x0, x1: root.x1, y0: root.y0, y1: root.y1, z1: Math.max(...boxes.map((b) => b.z)) }, total: tree.params, architecture: tree.architecture };
}

/**
 * Le classi Python dell'albero di run.json (`modules`, scritto da scripts/export_run.py): percorso → classe.
 * run.json è un file dello studente: la classe finisce nelle etichette e nei titoli, quindi senza controlli
 * bidirezionali (una con solo quelli non c'è).
 */
export function classesFromRunInfo(modules) {
  const out = new Map();
  const visit = (n, depth = 0) => {
    if (!n || typeof n !== "object" || depth > 64) return;
    const cls = typeof n.class === "string" ? withoutBidiControls(n.class.slice(0, 80)) : null;
    if (typeof n.path === "string" && cls) out.set(n.path, cls);
    if (Array.isArray(n.children)) for (const c of n.children.slice(0, 4096)) visit(c, depth + 1);
  };
  visit(modules);
  return out;
}

// ---------------------------------------------------------------- velature senza sovrapposizioni

/**
 * Un rettangolo meno dei "buchi" disgiunti (i figli disegnati sopra): strisce orizzontali che coprono
 * esattamente la parte scoperta. Perché: con il rendering software ogni livello di velatura che copre lo
 * schermo costa (7 contenitori annidati = 7 volte lo schermo); così ogni pixel si colora una volta sola.
 * @param {{x0: number, x1: number, y0: number, y1: number}} r
 * @param {{x0: number, x1: number, y0: number, y1: number}[]} holes rettangoli che non si sovrappongono
 * @returns {{x0: number, x1: number, y0: number, y1: number}[]}
 */
export function subtractRects(r, holes, eps = 1e-6) {
  const cut = holes
    .map((h) => ({ x0: Math.max(r.x0, h.x0), x1: Math.min(r.x1, h.x1), y0: Math.max(r.y0, h.y0), y1: Math.min(r.y1, h.y1) }))
    .filter((h) => h.x1 - h.x0 > eps && h.y1 - h.y0 > eps);
  const ys = [...new Set([r.y0, r.y1, ...cut.flatMap((h) => [h.y0, h.y1])])].sort((a, b) => a - b);
  const strips = [];
  for (let k = 0; k < ys.length - 1; k++) {
    const y0 = ys[k];
    const y1 = ys[k + 1];
    if (y1 - y0 <= eps) continue;
    const spans = cut.filter((h) => h.y0 <= y0 + eps && h.y1 >= y1 - eps).sort((a, b) => a.x0 - b.x0);
    let x = r.x0;
    for (const h of spans) {
      if (h.x0 > x + eps) strips.push({ x0: x, x1: h.x0, y0, y1 });
      x = Math.max(x, h.x1);
    }
    if (r.x1 > x + eps) strips.push({ x0: x, x1: r.x1, y0, y1 });
  }
  // strisce una sopra l'altra con gli stessi lati si uniscono (meno triangoli)
  const merged = [];
  for (const s of strips.sort((a, b) => a.x0 - b.x0 || a.x1 - b.x1 || a.y0 - b.y0)) {
    const last = merged.at(-1);
    if (last && Math.abs(last.x0 - s.x0) <= eps && Math.abs(last.x1 - s.x1) <= eps && Math.abs(last.y1 - s.y0) <= eps) last.y1 = s.y1;
    else merged.push({ ...s });
  }
  return merged;
}

/** I rettangoli disegnati sopra la velatura di un contenitore: i figli (i gruppi si attraversano), le matrici dei pesi. */
export function coveredBy(n) {
  const out = [];
  for (const c of n.children ?? []) {
    if (c.kind === "group") out.push(...coveredBy(c));
    else if (c.kind === "param" || c.kind === "buffer") out.push(c.matrix);
    else out.push(c);
  }
  return out;
}

// ---------------------------------------------------------------- per il LOD e la scelta

/** Il nodo più profondo che contiene il punto (x, y) del piano, tra quelli che si possono scegliere. */
export function hitTest(layout, x, y, { kinds = ["param", "buffer", "virtual", "module"] } = {}) {
  let best = null;
  for (const b of layout.boxes) {
    if (!kinds.includes(b.kind)) continue;
    if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
    if (!best || b.depth > best.depth || (b.depth === best.depth && (b.kind === "param" || b.kind === "buffer"))) best = b;
  }
  return best;
}

/**
 * Le "isole" della Panoramica: i figli della radice, con i blocchi uno per uno (blocks → Block ×N).
 * Stessi rettangoli dei contenitori del Dettaglio, così zoomando niente cambia di posto.
 */
export function macroItems(layout) {
  const items = [];
  for (const c of layout.root.children) {
    if (c.kind === "group") {
      items.push({ path: c.path, members: c.children.map((k) => k.path), title: c.children.every((k) => k.role === "embedding") ? "Embedding" : c.children.map((k) => k.title).join(" + "), sub: `${c.children.map((k) => k.title).join(" + ")}`, params: c.children.reduce((s, k) => s + k.params, 0), role: c.role, x0: c.x0, x1: c.x1, y0: c.y0, y1: c.y1, z: c.children[0].z });
    } else if (c.kind === "module" && c.children.length && c.children.every((k) => k.kind === "module" && k.index !== undefined) && c.children.length > 1) {
      for (const k of c.children) items.push({ path: k.path, members: [k.path], title: `${k.title}${k.type ? ` · ${k.type}` : ""}`, sub: "", params: k.params, role: k.role, x0: k.x0, x1: k.x1, y0: k.y0, y1: k.y1, z: k.z });
    } else {
      items.push({ path: c.path, members: [c.path], title: `${c.title}${c.type ? ` · ${c.type}` : ""}`, sub: "", params: c.params, role: c.role, x0: c.x0, x1: c.x1, y0: c.y0, y1: c.y1, z: c.z });
    }
  }
  const total = layout.total || 1;
  for (const it of items) it.share = it.params / total;
  return items;
}

/**
 * I moduli del livello "Moduli": dentro ogni isola della Panoramica i suoi figli diretti (per un
 * Block: ln1, sa, ln2, ffwd), oppure l'isola stessa se non ha figli che sono moduli.
 */
export function moduleItems(layout) {
  const out = [];
  for (const it of macroItems(layout)) {
    for (const path of it.members) {
      const n = layout.nodes.get(path);
      if (!n) continue;
      const kids = n.children.filter((k) => k.kind === "module" || k.kind === "virtual");
      if (n.leaf || !kids.length) out.push(n);
      else out.push(...kids);
    }
  }
  return out;
}
