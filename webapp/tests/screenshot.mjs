// Screenshot di verifica della web app (usa il Chromium preinstallato di Playwright).
//   node webapp/tests/screenshot.mjs <url> <cartella> [hash1,hash2,...] [azioniJS]
// Stampa gli errori della console e termina con codice 1 se ce ne sono.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [url = "http://127.0.0.1:8765/", out = ".", hashes = "transformer"] = process.argv.slice(2);
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
for (const hash of hashes.split(",")) {
  await page.goto(`${url}#${hash}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${out}/${hash.replace(/[^\w-]/g, "_")}.png` });
}
console.log(errors.length ? `ERRORI:\n${errors.join("\n")}` : "nessun errore in console");
await browser.close();
process.exit(errors.length ? 1 : 0);
