/**
 * La tabella del confronto A/B (solo DOM): una riga per tensore abbinato, colonne ordinabili (pulsanti
 * nelle intestazioni, aria-sort), i più cambiati in alto, una barra nel cambio relativo per leggerla a
 * colpo d'occhio, riga del totale. Clic (o Invio) sul nome = sceglie il tensore (scena, istogrammi).
 * Le intestazioni restano ferme scorrendo, la colonna del nome anche in orizzontale (sidebar stretta,
 * telefono). I nomi vengono dai file dello studente: sempre testo (el() crea nodi di testo).
 *
 *   const table = createCompareTable(box, { onSelect: (name) => … });
 *   table.setData(result, { selected });   table.setSelected(name);   table.dispose();
 */
import { el } from "../../../core/dom.js";
import { COLUMNS, DEFAULT_SORT, nextSort, sortRows } from "./compareModel.js";
import { COLUMN_HELP } from "./content.js";
import { int, num, pct, shape as shapeText, sig } from "./format.js";

const PAGE = 120; // righe mostrate subito: con migliaia di tensori il resto arriva a richiesta

/** "blocks.0.sa.heads.0.key.weight" → { tail: "key.weight", head: "blocks.0.sa.heads.0" }: in una colonna stretta si leggono entrambe. */
export function splitName(name) {
  const parts = String(name).split(".");
  // l'ultimo modulo con il nome del parametro; un indice da solo ("2.weight") non dice niente: si
  // prende anche il modulo che lo contiene ("net.2.weight")
  let k = Math.min(2, parts.length);
  if (parts.length > k && /^\d+$/.test(parts[parts.length - k])) k++;
  if (parts.length <= k) return { head: "", tail: String(name) };
  return { head: parts.slice(0, -k).join("."), tail: parts.slice(-k).join(".") };
}

export function createCompareTable(container, { onSelect = () => {} } = {}) {
  let sort = { ...DEFAULT_SORT };
  let result = null;
  let selected = null;
  let limit = PAGE;
  const caption = el("caption", { className: "lab-an-sr" }, "Confronto tensore per tensore tra A e B");
  const head = el("thead");
  const body = el("tbody");
  const foot = el("tfoot");
  const table = el("table", { className: "lab-an-table" }, caption, head, body, foot);
  const wrap = el("div", { className: "lab-an-table-wrap", role: "region", ariaLabel: "Tabella del confronto: scorre in orizzontale e in verticale", tabIndex: 0 }, table);
  const more = el("button", { type: "button", className: "lab-small lab-an-more", hidden: true, onclick: () => showMore() });
  const note = el("p", { className: "hint lab-an-table-note", hidden: true });
  container.append(wrap, more, note);

  function renderHead() {
    head.replaceChildren(
      el(
        "tr",
        {},
        ...COLUMNS.map((c) => {
          const active = sort.key === c.id;
          const th = el(
            "th",
            { scope: "col", className: `lab-an-col-${c.id}${c.numeric ? " is-num" : ""}`, title: COLUMN_HELP[c.id] ?? "" },
            el("button", { type: "button", className: "lab-an-sort", onclick: () => setSort(nextSort(sort, c.id)), ariaLabel: `${c.label}: ordina${active ? (sort.dir === "asc" ? ", ora crescente" : ", ora decrescente") : ""}` }, c.label, el("span", { className: "lab-an-arrow", ariaHidden: "true" }, active ? (sort.dir === "asc" ? "▴" : "▾") : "")),
          );
          th.setAttribute("aria-sort", active ? (sort.dir === "asc" ? "ascending" : "descending") : "none");
          return th;
        }),
      ),
    );
  }

  function renderBody() {
    const rows = result ? sortRows(result.rows, sort.key, sort.dir) : [];
    let maxRel = 0; // un ciclo: con decine di migliaia di tensori Math.max(...) traboccherebbe
    for (const r of rows) if (Number.isFinite(r.relChange) && r.relChange > maxRel) maxRel = r.relChange;
    if (!maxRel) maxRel = 1;
    body.replaceChildren(...rows.slice(0, limit).map((r) => row(r, maxRel)));
    more.hidden = rows.length <= limit;
    more.textContent = `Mostra tutti i ${int(rows.length)} tensori`;
    const infinite = rows.filter((r) => r.relChange === Infinity).length;
    note.hidden = !infinite;
    note.textContent = infinite ? `∞ (${int(infinite)} ${infinite === 1 ? "tensore" : "tensori"}, in fondo alla tabella): in B il tensore è tutto zero, per esempio i bias delle LayerNorm appena inizializzati, quindi non c'è una scala con cui dividere. Per loro guarda ‖A − B‖.` : "";
    const t = result?.totals;
    const totals = t ? { relChange: pct(t.relChange), cosine: t.cosine === null ? "—" : sig(t.cosine), diffNorm: sig(t.diffNorm), maxAbsDiff: "—", normA: sig(t.normA), normB: sig(t.normB), shape: "—" } : null;
    foot.replaceChildren(
      t && rows.length
        ? el(
            "tr",
            {},
            el("th", { scope: "row" }, el("span", { className: "lab-an-total" }, "Tutti i pesi abbinati"), el("small", {}, `${int(result.rows.length)} tensori · ${int(result.matchedParams)} pesi`)),
            ...COLUMNS.slice(1).map((c) => el("td", { className: "is-num" }, totals[c.id])),
          )
        : "",
    );
  }

  function row(r, maxRel) {
    const { head: prefix, tail } = splitName(r.name);
    const isSel = r.name === selected;
    const rel = r.relChange;
    const width = rel === Infinity ? 100 : Number.isFinite(rel) && maxRel > 0 ? Math.max(2, Math.round((100 * rel) / maxRel)) : 0;
    const button = el(
      "button",
      { type: "button", className: "lab-an-name", title: r.nameB ? `${r.name} (in B: ${r.nameB})` : r.name, onclick: () => onSelect(r.name), ariaLabel: `${r.name}: cambio relativo ${pct(rel)}, coseno ${num(r.cosine)}. Mostra il tensore` },
      el("span", { className: "lab-an-tail" }, tail),
      prefix ? el("span", { className: "lab-an-head" }, prefix) : "",
    );
    if (isSel) button.setAttribute("aria-current", "true");
    const cells = {
      relChange: el("td", { className: "is-num lab-an-rel" }, el("span", { className: "lab-an-bar", ariaHidden: "true", style: { width: `${width}%` } }), el("span", { className: "lab-an-rel-text" }, pct(rel))),
      cosine: el("td", { className: "is-num" }, r.cosine === null ? "—" : sig(r.cosine)),
      diffNorm: el("td", { className: "is-num" }, sig(r.diffNorm)),
      maxAbsDiff: el("td", { className: "is-num" }, sig(r.maxAbsDiff)),
      normA: el("td", { className: "is-num" }, sig(r.normA)),
      normB: el("td", { className: "is-num" }, sig(r.normB)),
      shape: el("td", { className: "is-num lab-an-shape" }, shapeText(r.shape)),
    };
    return el("tr", { className: isSel ? "is-selected" : "", dataset: { name: r.name } }, el("th", { scope: "row" }, button), ...COLUMNS.slice(1).map((c) => cells[c.id]));
  }

  function setSort(next) {
    sort = next;
    renderHead();
    renderBody();
  }

  function showMore() {
    limit = Infinity;
    renderBody();
  }

  renderHead();

  return {
    setData(next, { selected: sel = selected } = {}) {
      result = next;
      selected = sel;
      if (!next) limit = PAGE;
      renderBody();
    },
    setSelected(name) {
      if (name === selected) return;
      selected = name;
      for (const tr of body.children) {
        const on = tr.dataset.name === name;
        tr.classList.toggle("is-selected", on);
        const b = tr.querySelector(".lab-an-name");
        if (on) b?.setAttribute("aria-current", "true");
        else b?.removeAttribute("aria-current");
      }
    },
    /** Porta in vista la riga di un tensore (dopo una scelta nella scena). */
    reveal(name) {
      const tr = [...body.children].find((x) => x.dataset.name === name);
      if (!tr) return;
      const top = tr.offsetTop - wrap.clientHeight / 2 + tr.offsetHeight / 2;
      wrap.scrollTop = Math.max(0, top);
    },
    get sort() {
      return { ...sort };
    },
    root: wrap,
    dispose() {
      wrap.remove();
      more.remove();
      note.remove();
    },
  };
}
