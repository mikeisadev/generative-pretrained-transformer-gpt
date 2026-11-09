/**
 * Registro dei modelli della vista "Architettura LLM moderni completa" (SOLI DATI).
 *
 * Ogni modulo di modello esporta un oggetto con questi campi (tutti dati, nessuna funzione):
 *   id, name, short, checkpoint, org, role, released, license, repo, repoUrl, headline, why
 *   config        campi del config usati dalla vista (con il nome originale nei commenti)
 *   layers        { base, kinds: stringa con un carattere per layer, flags: {nome: {layers, label}},
 *                   segments: [{id, label, sub, from, to, color}] }  (from/to: numeri di layer)
 *   kinds         { codice: {label, name, color, text} } per ogni carattere di layers.kinds
 *   params        conteggi per layer e globali (vedi full/model.js: layerParams, paramBreakdown)
 *   input, output righe del livello Blocchi [{id, title, sub, text, params, color, narrow?}]
 *   branches      rami fuori linea [{id, …, attach: "embedding" | "last", side: "top" | "bottom"}]
 *   caches        riga delle cache [{id, title, sub, text, layers: [n…] | "all" | codice, color}]
 *   pipeline      livello Panoramica [{id, title, sub, text, steps?, unverified?: [id…]}]
 *   precision     byte per precisione [{id, dtype, what, gb, estimate?}]
 *   checkpoint, kv, flops, memory   numeri del checkpoint, della KV cache e del calcolo
 *   anatomy       livello Dettaglio: residual, norm, attention {rows, core, output}, ffn, branch
 *   mechanisms    dove vive ognuno dei 16 meccanismi della vista d'insieme (regionData.js)
 *   tensorGroups  nomi veri dei tensori per componente [{id, label, tensors: [[nome, forma, dtype]]}]
 *   unverified    ciò che la ricerca segna come NON verificato [{id, value, why}]
 *   sources       fonti [{title, url, note}]
 * La coerenza con la ricerca è controllata da tests/modern-llm-full.test.mjs.
 */
import { DEEPSEEK_V41_FLASH } from "./deepseekV41Flash.js";
import { GPT_OSS_120B } from "./gptOss120b.js";
import { KIMI_K31_PRO } from "./kimiK31Pro.js";

/** Nell'ordine del selettore: il principale per primo. */
export const MODELS = [DEEPSEEK_V41_FLASH, KIMI_K31_PRO, GPT_OSS_120B];
export const MODEL_BY_ID = new Map(MODELS.map((m) => [m.id, m]));
export const PRIMARY_MODEL_ID = DEEPSEEK_V41_FLASH.id;
