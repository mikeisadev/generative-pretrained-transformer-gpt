// Run esportati con scripts/export_run.py che alcuni test del Laboratorio pesi caricano come farebbe lo
// studente. Pesano qualche MB ciascuno, quindi non stanno nel repository: si indicano con
// LAB_RUNS_DIR=<cartella>, che deve contenere
//   gpt-200-step  200 step del GPT del corso (record_training.py, poi export_run.py --trace --benchmark)
//   ffn3          feed-forward da 3·C invece di 4·C (forme diverse: il motore JS non lo esegue)
//   gpt-con-gelu  GELU invece di ReLU (stesse forme, calcolo diverso: --not-engine-compatible --benchmark)
// Senza la variabile, o se manca una cartella, i test che li usano si saltano dicendo perché.
// Non è un file di test (non finisce in *.test.mjs): lo importano lab.test.mjs e lab-analysis.test.mjs.
import { existsSync } from "node:fs";
import { join } from "node:path";

export const RUN_NAMES = Object.freeze({ gpt200: "gpt-200-step", ffn3: "ffn3", gelu: "gpt-con-gelu" });

const ROOT = process.env.LAB_RUNS_DIR ?? "";

/** Cartella del run indicato (stringa vuota se LAB_RUNS_DIR non è impostata). */
export function runDir(key) {
  return ROOT ? join(ROOT, RUN_NAMES[key]) : "";
}

/** Motivo per saltare un test che usa questi run (per l'opzione `skip` di node:test), oppure false se ci sono tutti. */
export function missingRuns(...keys) {
  if (!ROOT) return "LAB_RUNS_DIR non indicato (i run esportati stanno fuori dal repository)";
  const missing = keys.filter((key) => !existsSync(runDir(key))).map((key) => RUN_NAMES[key]);
  return missing.length > 0 ? `mancano in LAB_RUNS_DIR: ${missing.join(", ")}` : false;
}
