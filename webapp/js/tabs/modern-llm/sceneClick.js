/**
 * Clic sulla scena che NON è navigazione: un solo puntatore (niente pizzico a due dita), tasto
 * sinistro o tocco, spostamento entro 5 px tra pressione e rilascio (oltre è un trascinamento della
 * navigazione libera). Lo usano i due sottotab del tab, ognuno con il suo gestore.
 *
 *   const off = installSceneClick(canvas, (event) => { … });
 */
const CLICK_SLOP_PX = 5;

export function installSceneClick(canvas, onClick) {
  const presses = new Map();
  let multiTouch = false;
  const down = (e) => {
    presses.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (presses.size > 1) multiTouch = true;
  };
  const cancel = (e) => {
    presses.delete(e.pointerId);
    if (!presses.size) multiTouch = false;
  };
  const up = (e) => {
    const start = presses.get(e.pointerId);
    presses.delete(e.pointerId);
    const gesture = multiTouch;
    if (!presses.size) multiTouch = false;
    if (!start || gesture || e.button !== 0 || Math.hypot(e.clientX - start.x, e.clientY - start.y) > CLICK_SLOP_PX) return;
    onClick(e);
  };
  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointercancel", cancel);
  canvas.addEventListener("pointerup", up);
  return () => {
    canvas.removeEventListener("pointerdown", down);
    canvas.removeEventListener("pointercancel", cancel);
    canvas.removeEventListener("pointerup", up);
  };
}
