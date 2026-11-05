/**
 * Punto di ingresso: carica i dati del modello, costruisce scena, menu e pannello
 * del codice e li tiene sincronizzati.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STEPS } from "./content.js";
import { loadModelSpec } from "./data.js";
import { buildArchitecture } from "./scene/architecture.js";
import { CameraDirector } from "./scene/camera.js";
import { DataFlow } from "./scene/flow.js";
import { COLORS } from "./scene/visuals.js";
import { CodePanel } from "./ui/codePanel.js";
import { Menu } from "./ui/menu.js";

function createRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);
  return renderer;
}

function createScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.background);
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(10, 20, 15);
  scene.add(sun);
  const grid = new THREE.GridHelper(220, 110, 0x1e293b, 0x111827);
  grid.position.y = -48;
  scene.add(grid);
  return scene;
}

function renderHeader(spec) {
  const cfg = spec.config;
  document.querySelector("[data-hparams]").textContent =
    `n_layer=${cfg.n_layer} · n_head=${cfg.n_head} · n_embd=${cfg.n_embd} · block_size=${cfg.block_size} · vocab_size=${cfg.vocab_size} · ${spec.total_params.toLocaleString("it-IT")} parametri`;
  document.querySelector("[data-source]").textContent = spec.trained ? "pesi addestrati" : "pesi casuali (lancia train.py)";
}

async function main() {
  const spec = await loadModelSpec();
  renderHeader(spec);
  // Su schermi stretti il menu parte chiuso, per lasciare spazio alla scena.
  if (window.matchMedia("(max-width: 900px)").matches) document.body.classList.add("menu-collapsed");

  const container = document.querySelector("#scene");
  const renderer = createRenderer(container);
  const scene = createScene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.05, 600);
  camera.position.set(0, 10, 120);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.screenSpacePanning = true;

  const { root, rail, registry, stations, path } = buildArchitecture(spec);
  scene.add(root, rail);

  // La finestra del codice copre il lato destro: spostiamo il centro della proiezione
  // verso sinistra, così il componente inquadrato resta nella parte visibile.
  const codeRoot = document.querySelector("#code-panel");
  const coveredWidth = () => {
    const wide = window.matchMedia("(min-width: 901px)").matches;
    return wide && !codeRoot.classList.contains("minimized") ? codeRoot.offsetWidth + 14 : 0;
  };
  const updateViewport = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    const covered = coveredWidth();
    if (covered > 0) camera.setViewOffset(w, h, covered / 2, 0, w, h);
    else camera.clearViewOffset();
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  };
  new MutationObserver(updateViewport).observe(codeRoot, { attributes: true, attributeFilter: ["class"] });

  const director = new CameraDirector(camera, controls, scene, () => (container.clientWidth - coveredWidth()) / container.clientHeight);
  controls.addEventListener("start", () => director.cancel());
  const showOverview = () => director.focus(registry.get("overview"), { showFrame: false, margin: 1.12 });

  const caption = document.querySelector("[data-caption]");
  const flow = new DataFlow(scene, path, stations, (station) => (caption.textContent = station.caption));

  const codePanel = new CodePanel(document.querySelector("#code-panel"), spec.snippets);
  const menu = new Menu(document.querySelector("#menu-steps"), STEPS, spec, (item, title) => {
    const target = registry.get(item.focus);
    if (item.focus === "overview") showOverview();
    else if (target) director.focus(target);
    codePanel.show(item.code, title, item.highlight ?? []);
  });

  // Controlli dell'interfaccia
  const playButton = document.querySelector("[data-play]");
  playButton.addEventListener("click", () => {
    playButton.textContent = flow.toggle() ? "⏸ Pausa flusso" : "▶ Avvia flusso";
  });
  document.querySelector("[data-overview]").addEventListener("click", showOverview);
  document.querySelector("[data-menu-toggle]").addEventListener("click", () => document.body.classList.toggle("menu-collapsed"));
  window.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea")) return;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") menu.move(+1);
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") menu.move(-1);
  });

  new ResizeObserver(updateViewport).observe(container);
  updateViewport();

  menu.select("data.dataset");
  showOverview(); // all'avvio: vista d'insieme, con il primo step già aperto nel menu

  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.elapsedTime;
    director.update(dt, elapsed);
    flow.update(dt, elapsed);
    controls.update();
    renderer.render(scene, camera);
  });
  window.__gptApp = { spec, registry, director, menu }; // utile per il debug dalla console
}

main().catch((err) => {
  console.error(err);
  document.querySelector("[data-caption]").textContent = `Errore: ${err.message}. Avvia la web app con: python scripts/serve_webapp.py`;
});
