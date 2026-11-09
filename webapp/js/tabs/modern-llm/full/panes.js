/**
 * Pannelli della sidebar della vista completa (solo DOM, testo sempre con textContent: niente HTML
 * da dati). Ogni funzione restituisce un elemento da mettere nel pannello e, dove serve, un metodo
 * update(stato) per tenerlo allineato al modello scelto.
 *   renderViews    livelli, pezzi dell'architettura, layer del Dettaglio, ripetizioni, 16 meccanismi
 *   renderModels   selettore dei 3 modelli, nomi chiesti, panorama degli open-weight
 *   renderWeights  ripartizione dei parametri, byte per precisione, tensori, formato del checkpoint
 *   renderSources  come sono stati ottenuti i numeri, fonti del modello, cose da verificare
 */
import { el } from "../../../core/dom.js";
import { CAPABILITY_FACTS, HOW_OBTAINED, LANDSCAPE, NAME_CHECK, RESEARCH_DATE } from "./research.js";
import { FULL_LEVELS, FULL_SECTIONS } from "./catalog.js";
import { enumerateCheckpoint } from "./tensors.js";
import { addOnTotal, anchorsOf, formatCount, formatGB, formatLayers, formatShare, layerList, paramTotals } from "./model.js";
import { DTYPE_COLORS } from "./sceneMacro.js";
import { MODELS } from "./models/index.js";

const STATUS_TEXT = { present: "qui", absent: "qui vivrebbe, ma questo modello non lo usa", study: "non è un pezzo del modello" };

/** Pillola "da verificare" con la spiegazione nel tooltip e nel testo accessibile. */
export function doubtBadge(u) {
  return el("span", { className: "mllm-doubt", title: `${u.value}: ${u.why}`, tabIndex: 0, ariaLabel: `Da verificare: ${u.value}. ${u.why}` }, "da verificare");
}

/** Pulsante a scheda (titolo + riga piccola), come nel pannello Viste della vista d'insieme. */
function viewButton(title, text, onclick, extra = null) {
  return el("button", { type: "button", onclick }, el("strong", {}, title), text ? el("small", {}, text) : null, extra);
}

/**
 * Pannello Viste.
 * @param {object} o {onLevel(level), onSection(section), onLayer(n), onToggleExpand(), onMechanism(regionId), onOverview(regionId), regions}
 */
export function renderViews({ onLevel, onSection, onLayer, onToggleExpand, onMechanism, onOverview, regions }) {
  const levels = el("div", { className: "view-switch mllm-views" }, ...FULL_LEVELS.map((l) => viewButton(l.title, l.text, () => onLevel(l.level))));
  const sections = el("div", { className: "view-switch mllm-views" });
  const expand = el("button", { type: "button", className: "mllm-run", onclick: () => onToggleExpand() }, "Espandi le ripetizioni");
  const layerSelect = el("select", { className: "mllm-select", ariaLabel: "Layer da guardare nel Dettaglio", onchange: (e) => onLayer(Number(e.target.value)) });
  const layerNote = el("p", { className: "hint" });
  const mechanisms = el("ol", { className: "mllm-mechs" });
  const root = el(
    "div",
    { className: "mllm-full-pane" },
    el("h3", { className: "mllm-h3" }, "Livelli dello zoom"),
    levels,
    el("h3", { className: "mllm-h3" }, "Ripetizioni (il blocco «× N»)"),
    el("p", { className: "hint" }, "I layer uguali sono raggruppati in blocchi «× N» con le lastre fantasma: srotolali per vedere i layer veri, uno per uno."),
    expand,
    el("h3", { className: "mllm-h3" }, "Dentro un layer"),
    el("label", { className: "mllm-field" }, el("span", {}, "Layer"), layerSelect),
    layerNote,
    el("h3", { className: "mllm-h3" }, "Pezzi dell'architettura"),
    sections,
    el("h3", { className: "mllm-h3" }, "I 16 meccanismi in questo modello"),
    el("p", { className: "hint" }, "Ogni meccanismo della vista d'insieme ha un segnaposto numerato nel punto dove vive. «Mostra» lo inquadra qui; «Vista d'insieme» apre la sua spiegazione."),
    mechanisms,
  );
  return {
    root,
    /** Allinea il pannello al modello, alle ripetizioni srotolate e al layer scelto. */
    update({ model, expanded, layer }) {
      expand.textContent = expanded ? "Comprimi le ripetizioni in «× N»" : "Espandi le ripetizioni (layer veri)";
      expand.setAttribute("aria-pressed", String(Boolean(expanded)));
      const list = layerList(model);
      if (layerSelect.dataset.model !== model.id) {
        layerSelect.dataset.model = model.id;
        layerSelect.replaceChildren(...list.map((l) => el("option", { value: String(l.number) }, `Layer ${l.number} · ${model.kinds[l.kind].name}${[...l.flags].map((f) => ` · ${model.layers.flags[f].label}`).join("")}`)));
        const anchors = anchorsOf(model);
        sections.replaceChildren(
          ...FULL_SECTIONS.map((s) => {
            const present = anchors.has(s.anchor);
            const b = viewButton(s.title, present ? null : "non c'è in questo modello", () => onSection(s));
            b.disabled = !present;
            b.dataset.view = s.id;
            return b;
          }),
        );
        mechanisms.replaceChildren(
          ...regions.map((r, i) => {
            const m = model.mechanisms[r.id];
            return el(
              "li",
              { className: `mllm-mech ${m.status}` },
              el("span", { className: "mllm-mech-num", style: { background: m.status === "absent" ? "#334155" : r.color } }, String(i + 1)),
              el("div", {}, el("strong", {}, r.title), el("small", {}, `${STATUS_TEXT[m.status]}: ${m.text}`)),
              el("span", { className: "mllm-mech-actions" }, el("button", { type: "button", onclick: () => onMechanism(r.id) }, "Mostra"), el("button", { type: "button", onclick: () => onOverview(r.id) }, "Vista d'insieme")),
            );
          }),
        );
      }
      layerSelect.value = String(layer);
      const l = list.find((x) => x.number === layer);
      layerNote.textContent = l ? model.kinds[l.kind].text : "";
    },
  };
}

/** Pannello Modelli: selettore, schede con i numeri, nomi chiesti, panorama degli open-weight. */
export function renderModels({ onModel }) {
  const cards = el("div", { className: "mllm-models" });
  const root = el(
    "div",
    { className: "mllm-full-pane" },
    el("p", { className: "mllm-note" }, `Tre modelli con pesi aperti, smontati con gli stessi dati e lo stesso codice. Ricerca verificata dell'${RESEARCH_DATE}.`),
    cards,
    el("h3", { className: "mllm-h3" }, "I nomi chiesti esistono?"),
    el("ul", { className: "mllm-checks" }, ...NAME_CHECK.map((n) => el("li", {}, el("span", { className: n.exists ? "ok" : "no" }, n.exists ? "sì" : "no"), el("strong", {}, n.asked), el("small", {}, n.what)))),
    el("h3", { className: "mllm-h3" }, "Panorama degli open-weight di frontiera"),
    el("p", { className: "hint" }, "Verifica: P = fonte primaria, R = ricetta o documentazione vLLM, S = solo stampa. Le righe «da verificare» si basano solo sulla stampa."),
    el(
      "div",
      { className: "mllm-landscape" },
      ...LANDSCAPE.map((row) =>
        el(
          "article",
          { className: row.verified ? "" : "unverified" },
          el("header", {}, el("strong", {}, row.model), el("span", { className: "muted" }, ` · ${row.family} · ${row.date}`)),
          el("p", {}, `${row.size} · ${row.arch}`),
          el("p", { className: "hint" }, `Licenza: ${row.license} · verifica: ${row.check}`, row.verified ? null : el("span", { className: "mllm-doubt", title: "Solo stampa: non riverificato" }, "da verificare")),
        ),
      ),
    ),
  );
  return {
    root,
    update({ model }) {
      cards.replaceChildren(
        ...MODELS.map((m) => {
          const t = paramTotals(m);
          const addOn = addOnTotal(m);
          const active = m.id === model.id;
          const rows = [
            ["parametri", `${formatCount(t.backbone, { digits: 2 })} nel backbone${t.outside.length ? ` (+ ${t.outside.map((o) => `${o.label} ${o.approx ? "≈ " : ""}${formatCount(o.params)}`).join(", ")})` : ""}${addOn ? `; in più ${addOn.label}: ${addOn.layers} × ${formatCount(addOn.perLayer)} = ${formatCount(addOn.params)}` : ""}`],
            ["attivi per token", `${formatCount(t.active, { digits: 2 })}${t.prefill ? ` in decode, ${formatCount(t.prefill, { digits: 2 })} in prefill` : ""}`],
            ["layer", `${t.layers}: ${Object.entries(t.byKind).map(([k, v]) => `${v.layers} ${m.kinds[k].name}`).join(", ")}`],
            ["esperti", `${m.params.moe.routed} instradati, ${m.params.moe.active} attivi${m.params.moe.shared ? ` + ${m.params.moe.shared} condivis${m.params.moe.shared > 1 ? "i" : "o"}` : ", nessun condiviso"}`],
            ["contesto", `${m.config.maxPositions.toLocaleString("it-IT")} token`],
            ["KV cache", `${m.kv.perToken.toLocaleString("it-IT")} B per token${m.kv.stateNote ? `, ${m.kv.stateNote}` : ""}`],
            ["uscito", m.released],
            ["licenza", m.license],
          ];
          return el(
            "article",
            { className: `mllm-model${active ? " active" : ""}` },
            el("h3", {}, m.name, m.version ? el("span", { className: "muted" }, ` · ${m.version}`) : null),
            el("p", { className: "hint" }, `${m.org} · ${m.role}`),
            el("p", { className: "mllm-why" }, m.why),
            el("table", {}, el("tbody", {}, ...rows.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, v))))),
            el("p", { className: "hint" }, `Fonte dei numeri: ${m.params.official}.`),
            active ? el("p", { className: "mllm-active" }, "Modello mostrato nella scena") : el("button", { type: "button", className: "mllm-run", onclick: () => onModel(m.id) }, `Mostra ${m.name}`),
          );
        }),
      );
    },
  };
}

/** Pannello Pesi: dove sono i parametri, in quali precisioni, con quali nomi; il formato del file. */
export function renderWeights() {
  const body = el("div", {});
  const root = el("div", { className: "mllm-full-pane" }, body);
  return {
    root,
    update({ model }) {
      const t = paramTotals(model);
      const addOn = addOnTotal(model);
      const maxShare = t.components[0]?.share ?? 1;
      const bars = t.components.map((c) =>
        el(
          "li",
          {},
          el("span", { className: "mllm-bar-label" }, c.label),
          el("span", { className: "mllm-bar" }, el("span", { style: { width: `${Math.max(0.6, (100 * c.share) / maxShare)}%` } })),
          el("span", { className: "mllm-bar-value" }, `${formatCount(c.params)} · ${formatShare(c.share, c.share < 0.001 ? 3 : 1)}`),
        ),
      );
      const precision = (model.precision ?? []).map((p) => el("li", {}, el("span", { className: "mllm-dot", style: { background: DTYPE_COLORS[p.dtype] ?? "#94a3b8" } }), el("strong", {}, p.dtype), ` ${p.what}: ${formatGB(p.gb)}`));
      const doubt = (id) => model.unverified.find((u) => u.id === id);
      const checkpointLine = el(
        "p",
        {},
        `Checkpoint ${model.checkpoint.format}: ${formatGB(model.checkpoint.gb)}${model.checkpoint.gib ? ` (${model.checkpoint.gib.toLocaleString("it-IT")} GiB)` : ""}`,
        model.checkpoint.tensors ? `, ${model.checkpoint.tensors.toLocaleString("it-IT")} tensori` : "",
        model.checkpoint.shards ? `, ${model.checkpoint.shards} shard ` : ". ",
        model.checkpoint.shards && doubt("shards") ? doubtBadge(doubt("shards")) : null,
      );
      const recon = model.config.compressRatios ? enumerateCheckpoint(model) : null;
      // replaceChildren (a differenza di el) scriverebbe "null" per le righe che mancano: si filtrano
      const children = [
        el("p", { className: "mllm-note" }, `Dove sono i ${formatCount(t.backbone, { digits: 2 })} parametri del backbone di ${model.name}: la barra è proporzionale alla quota più grande.`),
        el("ul", { className: "mllm-bars" }, ...bars),
        t.outside.length ? el("p", { className: "hint" }, `Fuori dal backbone: ${t.outside.map((o) => `${o.label} ${formatCount(o.params)}${o.approx ? " (circa)" : ""}`).join(" · ")}. ${model.params.note ?? ""}`) : null,
        addOn ? el("p", { className: "hint" }, `In più, non compreso nei conteggi ufficiali: ${addOn.label}, ${addOn.layers} layer × ${formatCount(addOn.perLayer)} = ${formatCount(addOn.params)}.`) : null,
        el("h3", { className: "mllm-h3" }, "Byte per precisione"),
        el("ul", { className: "mllm-precision" }, ...precision),
        model.precision?.some((p) => p.estimate) ? el("p", { className: "hint" }, "Stime: parametri × bit per valore (MXFP4 = 4,25 bit, cioè 4 bit più una scala ogni 32; BF16 = 16 bit).") : null,
        checkpointLine,
        el("p", { className: "hint" }, model.checkpoint.dtypes),
        recon
          ? el(
              "details",
              { className: "mllm-recon" },
              el("summary", {}, `Ricostruzione tensore per tensore: ${recon.tensors.toLocaleString("it-IT")} tensori, ${(recon.bytes / 1e9).toLocaleString("it-IT", { maximumFractionDigits: 1 })} GB`),
              el("p", { className: "hint" }, "Contati dai moduli del motore ufficiale DeepInfer e dal config ricostruito: coincidono con i 96.085 tensori e i 510,3 GB della ricetta vLLM."),
              el("table", {}, el("tbody", {}, ...recon.groups.map((g) => el("tr", {}, el("th", {}, g.label), el("td", {}, `${g.tensors.toLocaleString("it-IT")} tensori · ${(g.bytes / 1e9).toLocaleString("it-IT", { maximumFractionDigits: 2 })} GB`))))),
            )
          : null,
        el("h3", { className: "mllm-h3" }, "Nomi veri dei tensori"),
        ...model.tensorGroups.map((g) =>
          el(
            "details",
            { className: "mllm-tensors" },
            el("summary", {}, g.label),
            el("ul", {}, ...g.tensors.map(([name, shape, dtype]) => el("li", {}, el("code", {}, name), shape ? ` ${shape}` : "", dtype ? el("span", { className: "mllm-dtype", style: { color: DTYPE_COLORS[dtype.split(" ")[0]] ?? "#94a3b8" } }, ` ${dtype}`) : null))),
            g.note ? el("p", { className: "hint" }, g.note) : null,
          ),
        ),
        model.tensorNote ? el("p", { className: "hint" }, model.tensorNote) : null,
        el("h3", { className: "mllm-h3" }, "Il formato safetensors"),
        el("p", {}, CAPABILITY_FACTS.find((f) => f.id === "safetensors").text),
      ];
      body.replaceChildren(...children.filter(Boolean));
    },
  };
}

/** Pannello Fonti: come sono stati ottenuti i numeri, fonti del modello, elenco "da verificare". */
export function renderSources() {
  const body = el("div", {});
  const root = el(
    "div",
    { className: "mllm-full-pane mllm-sources" },
    el("h3", {}, "Come sono stati ottenuti i numeri"),
    ...HOW_OBTAINED.map((t) => el("p", { className: "hint" }, t)),
    body,
    el("h3", {}, "Come il training produce i pesi e come emergono le capacità"),
    el(
      "ul",
      {},
      ...CAPABILITY_FACTS.map((f) =>
        el(
          "li",
          {},
          el("a", { href: f.url, target: "_blank", rel: "noopener" }, f.title),
          el("span", { className: "muted" }, ` · ${f.source}`),
          el("p", { className: "hint" }, f.text),
          f.caution ? el("p", { className: "hint mllm-caution" }, `Cautela: ${f.caution}`) : null,
          ...(f.links ?? []).map((l) => el("p", { className: "hint" }, "Vedi anche: ", el("a", { href: l.url, target: "_blank", rel: "noopener" }, l.title))),
        ),
      ),
    ),
  );
  return {
    root,
    update({ model }) {
      body.replaceChildren(
        el("h3", {}, `Fonti di ${model.name}`),
        el("ul", {}, ...model.sources.map((s) => el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener" }, s.title), el("p", { className: "hint" }, s.note)))),
        el("h3", {}, "Da verificare"),
        el("ul", {}, ...model.unverified.map((u) => el("li", {}, el("strong", {}, u.value), el("p", { className: "hint" }, u.why)))),
      );
    },
  };
}

/** Riga di layer per una didascalia: "layer 2-7". */
export const layersText = (numbers) => `layer ${formatLayers(numbers)}`;
