// Screenshot di verifica della web app (usa il Chromium preinstallato di Playwright): il controllo finale di un
// round, sui 6 tab, sulle loro viste principali e sulle rappresentazioni (Rete neurale, Cervello).
//   node webapp/tests/screenshot.mjs <url> <cartella> [tab1,tab2,...] [--quick] [--lod] [--nav] [--mobile] [--sizes=LxA,...]
//     senza elenco dei tab: tutti e 6, uno dopo l'altro nella stessa pagina (verifica anche il passaggio tra
//               tab, come fa lo studente); un elemento dell'elenco può essere anche un link diretto
//               (es. transformer/go=view:numeri): allora solo quello screenshot
//     per ogni tab: uno screenshot all'apertura e uno per ogni vista principale (VIEWS qui sotto: le viste del
//               Transformer con Pesi numerici e la rete con profondità, Confronto e Benchmark del Laboratorio,
//               l'architettura completa di LLM moderni, Rete neurale e Cervello di ogni tab)
//     --quick   solo lo screenshot all'apertura di ogni tab (com'era prima del round E)
//     --lod     uno screenshot in più per ogni livello dello zoom semantico (selettore in alto)
//     --nav     navigazione libera: spostamento con D e W tenuti premuti (screenshot "-nav-moved"),
//               poi C (Centra) e uno screenshot "-nav-home" che deve tornare alla vista iniziale
//     --mobile  telefono: 390×844 con il touch (come --sizes=390x844)
//     --sizes   una o più dimensioni, es. --sizes=1600x900,1100x800,390x844 (sotto 600 px di larghezza il
//               contesto è un telefono touch); senza: 1600×900
// Prima della web app si apre una pagina di riscaldamento: la prima pagina WebGL di una sessione con il
// rendering software (SwiftShader) stampa alcuni avvisi "GPU stall due to ReadPixels" qualunque cosa
// disegni; la pagina di riscaldamento li assorbe, così ogni avviso che resta è davvero della web app
// (niente filtri sui messaggi: uno stallo causato dal nostro codice si deve vedere).
// Stampa gli errori e i warning della console e i passi falliti (vista che non si apre, tab che non si
// carica) e termina con codice 1 se ce ne sono.
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const ALL_TABS = ["transformer", "lab", "rag", "diffusion", "audio", "modern-llm"];
/** Viste principali di ogni tab (id di app.goto(tab, {kind: "view", id})), con le rappresentazioni per ultime. */
const VIEWS = {
  transformer: ["pesi", "numeri", "numeri.training", "contesto", "neuroni", "neuroni-profondita", "cervello"],
  lab: ["confronto", "benchmark", "rete", "cervello"],
  rag: ["rete", "cervello"],
  diffusion: ["dit", "video", "rete", "cervello"],
  audio: ["rvq", "whisperAll", "rete", "cervello"],
  "modern-llm": ["full", "rete", "cervello"],
};
/** Viste che costruiscono una scena a parte (rete, cervello): la prima volta servono più fotogrammi. */
const HEAVY = new Set(["rete", "cervello", "neuroni", "neuroni-profondita", "full"]);
/** Attese in ms: apertura del tab (volo iniziale della camera), cambio di vista, vista pesante. */
const SETTLE = { tab: 3500, view: 2500, heavy: 5000 };

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith("--") && !a.includes("=")));
const option = (name) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const [url = "http://127.0.0.1:8765/", out = ".", list = ALL_TABS.join(",")] = args.filter((a) => !a.startsWith("--"));
const sizes = (option("sizes") ?? (flags.has("--mobile") ? "390x844" : "1600x900")).split(",").map((s) => s.split("x").map(Number));
const hashes = list.split(",").filter(Boolean);
mkdirSync(out, { recursive: true });

/** La pagina di riscaldamento: un contesto WebGL2 che disegna e rilegge un pixel, poi si chiude. */
async function warmUp(context) {
  const page = await context.newPage();
  await page.setContent(
    "<canvas id=c></canvas><script>const g=document.getElementById('c').getContext('webgl2');g.clearColor(0,0,0,1);g.clear(g.COLOR_BUFFER_BIT);g.readPixels(0,0,1,1,g.RGBA,g.UNSIGNED_BYTE,new Uint8Array(4));</script>",
  );
  await page.waitForTimeout(300);
  await page.close();
}

const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const problems = [];
let shots = 0;
for (const [width, height] of sizes) {
  const prefix = `${width}x${height}`;
  const phone = width < 600;
  const context = await browser.newContext({ viewport: { width, height }, ...(phone ? { isMobile: true, hasTouch: true } : {}) });
  await warmUp(context);
  const page = await context.newPage();
  let where = "caricamento";
  const report = (text) => problems.push(`${prefix} · ${where}: ${text}`);
  page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && report(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => report(`pageerror: ${e.message}`));
  const shoot = async (name) => {
    await page.screenshot({ path: `${out}/${prefix}-${name.replace(/[^\w.-]+/g, "_")}.png` });
    shots++;
  };

  await page.goto(`${url}#${hashes[0]}`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.__app?.whenReady, null, { timeout: 60000 });
  for (const hash of hashes) {
    const tab = hash.split("/")[0];
    const name = hash.replace(/[^\w-]/g, "_");
    where = hash;
    // cambio di hash nella stessa pagina: è il percorso dello studente che passa da un tab all'altro
    await page.evaluate((h) => (location.hash = h), hash);
    const ready = await page.evaluate(async (t) => Boolean(await window.__app.whenReady(t)), tab);
    if (!ready) {
      report("il tab non si carica");
      continue;
    }
    // un link diretto: il salto è finito (un salto fallito lo dice la console, raccolta sopra)
    await page.evaluate(() => window.__app.routing).catch((error) => report(`salto fallito: ${error.message.split("\n")[0]}`));
    await page.waitForTimeout(SETTLE.tab);
    await shoot(name);
    if (flags.has("--nav")) {
      await page.keyboard.down("d");
      await page.waitForTimeout(1200);
      await page.keyboard.up("d");
      await page.keyboard.down("w");
      await page.waitForTimeout(500);
      await page.keyboard.up("w");
      await page.waitForTimeout(1200); // frenata e bordo elastico
      await shoot(`${name}-nav-moved`);
      await page.keyboard.press("c");
      await page.waitForTimeout(2200); // volo di "Centra"
      await shoot(`${name}-nav-home`);
    }
    if (flags.has("--lod")) {
      const root = `.tab-root[data-tab="${tab}"]`;
      // solo i pulsanti visibili (su un telefono il selettore può essere ridotto)
      const modes = await page.$$eval(`${root} .lod-switch button`, (bs) => bs.filter((b) => b.checkVisibility?.() ?? b.offsetParent !== null).map((b) => b.dataset.mode).filter((m) => m && m !== "auto"));
      for (const mode of modes) {
        await page.click(`${root} .lod-switch button[data-mode="${mode}"]`);
        await page.waitForTimeout(2500); // volo della camera + sfumatura tra i livelli
        await shoot(`${name}-lod-${mode}`);
      }
      if (modes.length) await page.click(`${root} .lod-switch button[data-mode="auto"]`);
    }
    if (flags.has("--quick") || hash.includes("/")) continue;
    for (const view of VIEWS[tab] ?? []) {
      where = `${tab} · vista ${view}`;
      const shown = await page.evaluate(([t, id]) => window.__app.goto(t, { kind: "view", id }), [tab, view]).catch((error) => report(`errore: ${error.message.split("\n")[0]}`));
      if (shown === false) report("la vista non si apre");
      if (!shown) continue;
      await page.waitForTimeout(HEAVY.has(view) ? SETTLE.heavy : SETTLE.view);
      await shoot(`${name}-${view}`);
    }
  }
  await context.close();
}
await browser.close();
console.log(`${shots} screenshot in ${out}`);
console.log(problems.length ? `PROBLEMI (${problems.length}):\n${problems.join("\n")}` : "nessun errore né warning in console, tutte le viste aperte");
process.exit(problems.length ? 1 : 0);
