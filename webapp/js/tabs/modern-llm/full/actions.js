/**
 * Azioni animate della vista completa (ActionPlayer del nucleo). Le didascalie si costruiscono dai
 * dati del modello scelto all'avvio (build): le azioni generiche (token, router, training,
 * capacità) valgono per tutti e tre i modelli; quelle su pezzi propri di DeepSeek-V4.1-Flash (KV
 * compressa, encoder-decoder, Engram, DSpark) tornano prima al modello principale.
 * Regole come nel resto del tab: `enter` ricostruisce lo stato di inizio passo (il passo indietro
 * funziona), `update(p)` anima senza allocare, `onStop` riporta la scena a riposo.
 *
 * `fx` (da full/index.js): model, strip, blocks, anatomy(), macro, focus(anchor, o), setLayer(n),
 * ensurePrimary(), showCode(key), openCharts(id), plots, segmentBlocks(id), blockIndex(id).
 */
import { mulberry32 } from "../sim.js";
import { phase } from "../shapes.js";
import { fullActionInfo } from "./catalog.js";
import { CAPABILITY_FACTS } from "./research.js";
import { formatCount, layerList, paramTotals } from "./model.js";

/** Spazio in basso per la barra del player (unità di scena, scalate dal fx sulla distanza). */
const PAD = true;

/** Esperti accesi per un token: SIMULAZIONE DIDATTICA (seme fisso), le scelte vere non sono pubblicate. */
function simulatedExperts(count, k, seed) {
  const rand = mulberry32(seed);
  const chosen = new Set();
  while (chosen.size < k) chosen.add(Math.floor(rand() * count));
  return [...chosen];
}

function tokenAction(fx) {
  return {
    ...fullActionInfo("full-token"),
    onStart() {
      fx.prepare("blocks");
    },
    onStop() {
      fx.rest();
    },
    build: async () => {
      const m = fx.model;
      const c = m.config;
      const strip = fx.strip;
      const io = (id) => fx.blockIndex(id);
      const steps = [];
      const visit = (title, caption, indices, { duration = 3.6, anchor = null } = {}) =>
        steps.push({
          title,
          caption,
          duration,
          enter() {
            fx.blocks.highlight(indices);
            fx.focusBlocks(indices, { anchor });
            fx.packetAt(indices[0]);
          },
          update(p) {
            if (indices.length > 1) fx.packetAlong(indices, phase(p, 0.05, 0.95));
          },
        });
      visit("Il testo diventa token", `Il tokenizer spezza il testo in token: ${c.vocab.toLocaleString("it-IT")} voci possibili. Ogni token è un numero, l'indirizzo di una riga della tabella di embedding.`, [io("tokenizer")]);
      visit("Embedding", `La riga di quel token: ${c.hidden.toLocaleString("it-IT")} numeri, da una tabella [${c.vocab.toLocaleString("it-IT")} × ${c.hidden.toLocaleString("it-IT")}]. È il vettore che attraverserà il modello.`, [io("embedding")]);
      const streams = m.anatomy.residual.streams;
      visit("Il residuo", streams > 1 ? `Il vettore entra nel residuo a ${streams} flussi (mHC): ${streams} corsie da ${c.hidden.toLocaleString("it-IT")} che ogni layer mescola prima e dopo attention e MoE.` : m.input.find((b) => b.id === "residual").text, [io("residual")]);
      for (const seg of strip.segments) {
        const indices = fx.segmentBlocks(seg.id);
        const kinds = [...new Set(indices.map((i) => strip.blocks[i].kind))].map((k) => m.kinds[k].label).join(", ");
        // strip.segments ha indici di blocco: i numeri di layer stanno nel segmento del modello
        const range = m.layers.segments.find((s) => s.id === seg.id);
        const dense = m.params.dense?.layers.filter((n) => n >= range.from && n <= range.to) ?? [];
        const ffn = dense.length ? `nei layer MoE passa da ${m.params.moe.active} esperti su ${m.params.moe.routed}; il layer ${dense.join(", ")} ha invece una FFN densa` : `passa da ${m.params.moe.active} esperti su ${m.params.moe.routed} (MoE)`;
        visit(seg.label.charAt(0) + seg.label.slice(1).toLowerCase(), `${seg.sub}. Tipi di layer: ${kinds}. In ogni layer il token guarda gli altri (attention) e ${ffn}.`, indices, { duration: 6 });
      }
      visit("Norm e head", `Norm finale e head: ${c.vocab.toLocaleString("it-IT")} punteggi (logit), uno per ogni token del vocabolario.`, [io("norm"), io("head")]);
      visit("Il token scelto", "La softmax trasforma i punteggi in probabilità e se ne sceglie uno: il nuovo token rientra dall'inizio per generare il successivo.", [io("softmax")], { duration: 4 });
      if (m.branches.some((b) => b.id === "dspark")) steps.push({ title: "Intanto DSpark", caption: "Le feature degli ultimi 3 layer vanno anche a DSpark, che propone i 5 token successivi (azione «DSpark: 5 token proposti e verificati»).", duration: 4, enter() { fx.blocks.highlight([]); fx.focus("blocks:dspark", { pad: PAD }); fx.packetAt(-1); } });
      return steps;
    },
  };
}

function routerAction(fx) {
  return {
    ...fullActionInfo("full-router"),
    onStart() {
      fx.prepare("detail", { charts: "full-pesi" });
    },
    onStop() {
      fx.rest();
    },
    build: async () => {
      const m = fx.model;
      const moe = m.params.moe;
      const f = m.anatomy.ffn;
      const layer = layerList(m).find((l) => !l.flags.has("dense") && !l.flags.has("engram")) ?? layerList(m)[0];
      const first = simulatedExperts(moe.routed, moe.active, 7);
      const second = simulatedExperts(moe.routed, moe.active, 19);
      const t = paramTotals(m);
      const steps = [
        {
          title: "Un layer MoE",
          caption: `Nel layer ${layer.number} la FFN è un MoE: ${moe.routed} esperti instradati${moe.shared ? ` e ${moe.shared} condivis${moe.shared > 1 ? "i" : "o"} sempre attiv${moe.shared > 1 ? "i" : "o"}` : ", nessun esperto condiviso"}. ${f.expert.title}: ${f.expert.sub}.`,
          duration: 5,
          enter() {
            fx.setLayer(layer.number);
            fx.anatomy().route([], 0);
            fx.anatomy().highlight("experts");
            fx.focus("detail:experts", { layer: layer.number, pad: PAD, wide: true });
            fx.showCode(m.id === "deepseek-v4.1-flash" ? "router" : null);
          },
        },
        {
          title: "Il router dà un punteggio",
          caption: `Il router (${f.router.tensor}) calcola un punteggio per ogni esperto: ${f.router.sub}.${m.id === "deepseek-v4.1-flash" ? " Il bias di bilanciamento conta solo per scegliere, non per pesare." : ""}`,
          duration: 5,
          enter() {
            fx.anatomy().route([], 0);
            fx.anatomy().highlight(f.router.id);
          },
        },
        {
          title: `I ${moe.active} esperti scelti`,
          caption: `Si accendono i ${moe.active} esperti con il punteggio più alto. Simulazione didattica: le scelte vere per ogni token non sono pubblicate. Gli altri ${(moe.routed - moe.active).toLocaleString("it-IT")} non lavorano per questo token.`,
          duration: 5.5,
          enter() {
            fx.anatomy().highlight("experts");
            fx.anatomy().route(first, 0);
          },
          update(p) {
            fx.anatomy().route(first, phase(p, 0.05, 0.5));
          },
        },
      ];
      if (moe.shared) steps.push({ title: "L'esperto condiviso", caption: `${moe.shared > 1 ? `I ${moe.shared} esperti condivisi lavorano` : "L'esperto condiviso lavora"} per ogni token, qualunque sia la scelta del router.`, duration: 4, enter() { fx.anatomy().route(first, 1); fx.anatomy().highlight(f.shared[0].id); } });
      steps.push(
        // LatentMoE (Kimi): u = Σ p_i E_i(z) nello spazio latente, poi y = Σ condivisi(x) + W↑ RMSNorm(u) (ricerca §5)
        { title: "Somma pesata", caption: `Le uscite degli esperti scelti si sommano pesate (${f.sum.sub})${moe.latent ? `, nello spazio latente da ${m.config.latent.toLocaleString("it-IT")}: RMSNorm e W↑ le riportano a ${m.config.hidden.toLocaleString("it-IT")}` : ""}${moe.shared ? `, più ${moe.shared > 1 ? `i ${moe.shared} esperti condivisi` : "l'esperto condiviso"}` : ""}, e tornano nel residuo.`, duration: 4.5, enter() { fx.anatomy().route(first, 1); fx.anatomy().highlight(f.sum.id); } },
        {
          title: "Un altro token, altri esperti",
          caption: `Ogni token sceglie i suoi esperti: per questo servono tutti in memoria, ma il calcolo per token resta piccolo, ${formatCount(t.active)} attivi su ${formatCount(t.backbone)} (${((100 * t.active) / t.backbone).toLocaleString("it-IT", { maximumFractionDigits: 1 })}%).`,
          duration: 5,
          enter() {
            fx.anatomy().highlight("experts");
            fx.anatomy().route(second, 0);
            fx.openCharts("full-pesi");
          },
          update(p) {
            fx.anatomy().route(second, phase(p, 0.05, 0.5));
          },
        },
      );
      return steps;
    },
  };
}

function kvAction(fx) {
  return {
    ...fullActionInfo("full-kv"),
    async onStart() {
      await fx.ensurePrimary();
      fx.prepare("blocks", { charts: "full-kv" });
    },
    onStop() {
      fx.rest();
    },
    steps: [
      { title: "Senza compressione", caption: "Un modello GQA da 67B (DeepSeek LLM) salva 389.120 byte per ogni token del contesto: K e V di 8 head in 95 layer, in BF16. Con un milione di token sarebbero centinaia di GB.", duration: 5, enter() { fx.blocks.highlight([]); fx.focus("blocks:cache-kv", { pad: PAD }); } },
      { title: "Una sola KV head, con K = V", caption: "In V4.1 c'è un solo vettore KV da 512 (448 di contenuto + 64 con RoPE), uguale per K e V e condiviso da tutte le 64 head di query.", duration: 5, enter() { fx.setLayer(2); fx.anatomy().highlight("latent"); fx.focus("detail:latent", { layer: 2, pad: PAD, wide: true }); } },
      { title: "2 token → 1 voce", caption: "Nei layer Full dell'encoder (2, 8 e 14) il compressore fonde ogni coppia di token in una voce del KV globale, con un pooling pesato (compressor.wkv + compressor.wgate).", duration: 5, enter() { fx.setLayer(2); fx.anatomy().highlight("compressor"); fx.focus("detail:compressor", { layer: 2, pad: PAD, wide: true }); } },
      { title: "Ogni voce in FP4: 288 byte", caption: "La voce sta in FP4: 512 numeri da 4 bit più una scala ogni 16 = 288 byte (dal codice ufficiale, cache_engine.py). La chiave dell'indexer ne aggiunge 68.", duration: 5, enter() { fx.setLayer(2); fx.anatomy().highlight("entry"); fx.focus("detail:entry", { layer: 2, pad: PAD, wide: true }); fx.showCode("cache"); } },
      { title: "Scritto da 4 layer, riusato dagli altri", caption: "Solo i layer 2, 8, 14 e 20 scrivono il KV globale (sorgente); gli altri layer compressi lo riusano (Reuse) o rifanno solo la scelta delle 512 voci (Reindex). I layer 0 e 1 hanno solo la finestra.", duration: 5.5, enter() { fx.blocks.highlight(fx.layerBlocks([2, 8, 14, 20])); fx.focus("blocks:layers", { pad: PAD }); } },
      { title: "890 byte per token", caption: "Conto: 2,5 voci per token × (288 + 68) byte = 890 byte, sempre nella memoria della GPU. Contro i 389.120 del modello GQA da 67B: 437 volte meno.", duration: 6, enter() { fx.blocks.highlight([]); fx.focus("blocks:cache-kv", { pad: PAD }); fx.showCode("kvmath"); fx.openCharts("full-kv"); } },
    ],
  };
}

function cedAction(fx) {
  return {
    ...fullActionInfo("full-ced"),
    async onStart() {
      await fx.ensurePrimary();
      fx.prepare("blocks", { charts: "full-scala" });
    },
    onStop() {
      fx.rest();
    },
    steps: [
      { title: "Prefill: solo l'encoder", caption: "Il prompt passa solo dai 20 layer dell'encoder causale: 7,89B parametri attivi per token (8B dichiarati). Il decoder per ora aspetta.", duration: 5.5, enter() { fx.blocks.highlight(fx.segmentBlocks("encoder")); fx.focus("blocks:segment-encoder", { pad: PAD }); fx.showCode("ced"); } },
      { title: "H20: il riassunto dell'encoder", caption: "L'uscita dell'encoder (H20) diventa, con una proiezione, il KV globale di tutto il decoder: lo calcola il layer 20 per l'intero prompt.", duration: 5, enter() { fx.blocks.highlight([...fx.segmentBlocks("encoder").slice(-1), fx.layerBlocks([20])[0]]); fx.focus("blocks:layers", { pad: PAD }); } },
      { title: "Replay della finestra", caption: "Il decoder si riesegue solo sugli ultimi 128 token del prompt, per ricostruire le sue finestre (DeepInfer tiene 140 token: la finestra più un margine). Il risultato non è identico al calcolo completo: il post-training simula questo replay.", duration: 6, enter() { fx.blocks.highlight(fx.segmentBlocks("decoder")); fx.focus("blocks:segment-decoder", { pad: PAD }); } },
      { title: "Decode: tutto il modello", caption: "Per ogni token generato girano tutti i 40 layer: 16,13B parametri attivi (16B dichiarati).", duration: 5, enter() { fx.blocks.highlight([...fx.segmentBlocks("encoder"), ...fx.segmentBlocks("decoder")]); fx.focus("blocks:layers", { pad: PAD }); } },
      { title: "Perché conviene", caption: "Leggere un prompt lungo costa circa la metà: ≈ 15,8 GFLOP per token di prompt contro ≈ 32,3 per token generato. Il paper non pubblica un'ablazione del solo encoder-decoder.", duration: 5.5, enter() { fx.blocks.highlight(fx.segmentBlocks("encoder")); fx.openCharts("full-scala"); } },
    ],
  };
}

function engramAction(fx) {
  const step = (title, caption, card, extra = {}) => ({ title, caption, duration: extra.duration ?? 4.5, enter() { fx.setLayer(14); fx.anatomy().highlight(card); fx.focus(extra.anchor ?? `detail:${card}`, { layer: 14, pad: PAD, wide: true }); if (extra.code) fx.showCode(extra.code); } });
  return {
    ...fullActionInfo("full-engram"),
    async onStart() {
      await fx.ensurePrimary();
      fx.prepare("detail");
    },
    onStop() {
      fx.rest();
    },
    steps: [
      step("Layer 14: un ramo in più", "Nei layer 1 e 14 c'è Engram, una memoria di n-gram: si consulta per indirizzo, come un dizionario, invece di calcolare.", "ngram", { anchor: "detail:layer", duration: 5 }),
      step("Gli ultimi 2, 3 e 4 token", "Si prendono gli ultimi 2, 3 e 4 token, normalizzati in un vocabolario compresso di 99.092 classi (NFKC, minuscole, senza accenti, spazi compattati).", "ngram", { code: "engram" }),
      step("8 hash per ordine", "Ogni n-gram passa per 8 funzioni di hash: 3 ordini × 8 = 24 indirizzi, ognuno in una tabella con un numero primo di righe vicino a 16 milioni.", "hash"),
      step("24 righe lette", "Si leggono 24 righe da 256 numeri da tabelle che in tutto hanno circa 384 milioni di righe per layer, in FP8. Le due tabelle valgono 196,6B parametri (183 GiB): possono stare nella RAM dell'host.", "table", { duration: 6 }),
      step("Proiezione", "Le 24 righe (6.144 numeri) passano da engram.wkv: 6.144 → 25.600, cioè 5.120 × (4 + 1).", "proj"),
      step("Gate e residuo", "Un gate decide quanto di questa memoria entra nelle 4 corsie del residuo, all'inizio del layer, prima dell'attention.", "gate", { duration: 5 }),
    ],
  };
}

function dsparkAction(fx) {
  return {
    ...fullActionInfo("full-dspark"),
    async onStart() {
      await fx.ensurePrimary();
      fx.prepare("blocks");
    },
    onStop() {
      fx.rest();
    },
    steps: [
      { title: "Le feature degli ultimi 3 layer", caption: "Le uscite dei layer 37, 38 e 39 vengono fuse da main_proj (15.360 → 5.120): sono il punto di partenza della bozza.", duration: 5, enter() { fx.blocks.highlight(fx.layerBlocks([37, 38, 39])); fx.focus("blocks:dspark", { pad: PAD, with: fx.layerBlocks([39]) }); fx.showCode("dspark"); } },
      { title: "3 blocchi draft", caption: "DSpark ha 3 blocchi piccoli: finestra di 128 token e un MoE da 128 esperti, 3 attivi più 1 condiviso. In tutto 14,2B parametri, fuori dal backbone.", duration: 5, enter() { fx.blocks.highlight([fx.blockIndex("dspark")].filter((i) => i >= 0)); fx.focus("blocks:dspark", { pad: PAD }); } },
      { title: "5 token proposti", caption: "DSpark propone 5 token per giro: una testa di Markov (rango 256) dà una correzione di primo ordine ai logit e una confidence head stima la probabilità che vengano accettati.", duration: 5, enter() { fx.focus("blocks:dspark", { pad: PAD }); } },
      { title: "Il modello verifica", caption: "Il modello li verifica tutti in un solo passaggio e tiene il prefisso giusto: è la decodifica speculativa della vista d'insieme, costruita dentro il modello (vLLM: method dspark, 5 token).", duration: 5.5, enter() { fx.blocks.highlight([fx.blockIndex("head"), fx.blockIndex("softmax")]); fx.focus("blocks:head", { pad: PAD, with: [fx.blockIndex("dspark")] }); } },
    ],
  };
}

function trainAction(fx) {
  return {
    ...fullActionInfo("full-train"),
    onStart() {
      fx.prepare("macro", { charts: "full-pesi" });
    },
    onStop() {
      fx.rest();
    },
    build: async () => {
      const m = fx.model;
      return m.pipeline.map((stage) => {
        const doubts = (stage.unverified ?? []).map((id) => m.unverified.find((u) => u.id === id)?.value).filter(Boolean);
        const weights = stage.id === "weights";
        return {
          title: `${stage.title}: ${stage.sub}`,
          caption: `${stage.text}${doubts.length ? ` Da verificare: ${doubts.join("; ")}.` : ""}`,
          duration: weights ? 7 : 5,
          enter() {
            fx.macro.highlight(stage.id);
            fx.macro.fillPrecision(weights ? 0 : 1);
            fx.focus(`macro:${stage.id}`, { pad: PAD });
            if (weights) {
              fx.showCode(m.id === "deepseek-v4.1-flash" ? "names" : "safetensors");
              fx.openCharts("full-pesi");
            }
          },
          update(p) {
            if (weights) fx.macro.fillPrecision(phase(p, 0.1, 0.7));
          },
        };
      });
    },
  };
}

function emergeAction(fx) {
  const facts = ["scaling", "emergent", "induction", "grokking", "rl-subnet"].map((id) => CAPABILITY_FACTS.find((f) => f.id === id));
  return {
    ...fullActionInfo("full-emerge"),
    onStart() {
      fx.prepare("macro", { charts: "full-scala" });
    },
    onStop() {
      fx.rest();
    },
    steps: facts.map((f) => ({
      title: f.title,
      caption: `${f.text}${f.caution ? ` Cautela: ${f.caution}` : ""} (${f.source})`,
      duration: 7,
      enter() {
        fx.macro.highlight(null);
        fx.focus("macro:capabilities", { pad: PAD });
        if (f.id === "scaling" || f.id === "emergent") fx.openCharts("full-scala");
      },
    })),
  };
}

export function fullActions(fx) {
  return [tokenAction(fx), routerAction(fx), kvAction(fx), cedAction(fx), engramAction(fx), dsparkAction(fx), trainAction(fx), emergeAction(fx)];
}
