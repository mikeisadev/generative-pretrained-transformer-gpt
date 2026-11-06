// Screenshot di verifica della web app (usa il Chromium preinstallato di Playwright).
//   node webapp/tests/screenshot.mjs <url> <cartella> [tab1,tab2,...] [--lod] [--mobile]
//     senza elenco dei tab: tutti, uno dopo l'altro (verifica anche il passaggio tra tab)
//     --lod     uno screenshot in più per ogni livello dello zoom semantico (selettore in alto)
//     --mobile  viewport 390×844 invece di 1600×900
//     --nav     navigazione libera: spostamento con D e W tenuti premuti (screenshot "-nav-moved"),
//               poi C (Centra) e uno screenshot "-nav-home" che deve tornare alla vista iniziale
// Stampa gli errori e i warning della console e termina con codice 1 se ce ne sono.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const ALL_TABS = ["transformer", "rag", "diffusion", "audio", "modern-llm"];
const flags = new Set(process.argv.slice(2).filter((a) => a.startsWith("--")));
const [url = "http://127.0.0.1:8765/", out = ".", hashes = ALL_TABS.join(",")] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const viewport = flags.has("--mobile") ? { width: 390, height: 844 } : { width: 1600, height: 900 };

const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport });
const errors = [];
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning") errors.push(`[${m.type()}] ${m.text()}`);
});
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

const prefix = `${viewport.width}x${viewport.height}`;
await page.goto(`${url}#${hashes.split(",")[0]}`, { waitUntil: "networkidle" });
for (const hash of hashes.split(",")) {
  // cambio di hash nella stessa pagina: è il percorso dello studente che passa da un tab all'altro
  await page.evaluate((h) => (location.hash = h), hash);
  await page.waitForTimeout(4000);
  const name = hash.replace(/[^\w-]/g, "_");
  await page.screenshot({ path: `${out}/${prefix}-${name}.png` });
  if (flags.has("--nav")) {
    await page.keyboard.down("d");
    await page.waitForTimeout(1200);
    await page.keyboard.up("d");
    await page.keyboard.down("w");
    await page.waitForTimeout(500);
    await page.keyboard.up("w");
    await page.waitForTimeout(1200); // frenata e bordo elastico
    await page.screenshot({ path: `${out}/${prefix}-${name}-nav-moved.png` });
    await page.keyboard.press("c");
    await page.waitForTimeout(2200); // volo di "Centra"
    await page.screenshot({ path: `${out}/${prefix}-${name}-nav-home.png` });
  }
  if (!flags.has("--lod")) continue;
  const root = `.tab-root[data-tab="${hash.split("/")[0]}"]`;
  const modes = await page.$$eval(`${root} .lod-switch button`, (bs) => bs.map((b) => b.dataset.mode).filter((m) => m !== "auto"));
  for (const mode of modes) {
    await page.click(`${root} .lod-switch button[data-mode="${mode}"]`);
    await page.waitForTimeout(2500); // volo della camera + sfumatura tra i livelli
    await page.screenshot({ path: `${out}/${prefix}-${name}-lod-${mode}.png` });
  }
  if (modes.length) await page.click(`${root} .lod-switch button[data-mode="auto"]`);
}
console.log(errors.length ? `ERRORI:\n${errors.join("\n")}` : "nessun errore in console");
await browser.close();
process.exit(errors.length ? 1 : 0);
