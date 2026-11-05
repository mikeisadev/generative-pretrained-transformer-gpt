/** SEGNAPOSTO: tab "rag", implementato nel workflow B (vedi webapp/ARCHITECTURE.md). */
import { el } from "../../core/dom.js";

export async function createTab(root) {
  root.append(el("p", { className: "tab-error" }, "Tab in costruzione."));
  return { activate() {}, deactivate() {} };
}
