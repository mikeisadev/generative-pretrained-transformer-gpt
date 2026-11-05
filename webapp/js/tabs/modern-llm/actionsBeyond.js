/**
 * Azioni "Oltre il testo": immagini come token (LLaVA), mesh 3D come testo (LLaMA-Mesh),
 * interpretabilità (induction head e autoencoder sparsi).
 */
import { MESH_VERTICES, OBJ_LINES } from "./regionMesh.js";
import { phase } from "./shapes.js";

const CATEGORY = "Oltre il testo";

function multiAction(tab) {
  const r = tab.regions.multi;
  const part = (a, b, title, caption, duration = 4.5) => ({
    title,
    caption,
    duration,
    enter() {
      r.set(a === 0 ? 0 : a);
    },
    update(p) {
      r.set(a + (b - a) * phase(p, 0.05, 0.9));
    },
  });
  return {
    id: "multi",
    title: "Un'immagine diventa token",
    category: CATEGORY,
    summary: "Schema di LLaVA: patch → encoder visivo → proiettore → token visivi accanto al testo.",
    onStart() {
      tab.prepare("multi");
    },
    onStop() {
      r.set(0);
    },
    steps: [
      part(0.001, 0.25, "Patch", "L'immagine viene tagliata in quadratini (patch). Qui 4 × 4; un encoder come CLIP ViT-L/14 usa patch di 14 × 14 pixel."),
      part(0.25, 0.55, "Encoder e proiettore", "L'encoder visivo (già addestrato, es. CLIP) trasforma ogni patch in un vettore. Il proiettore (una matrice in LLaVA, un piccolo MLP in LLaVA-1.5) lo porta nello spazio degli embedding del modello di testo."),
      part(0.55, 0.85, "Token visivi", "Il risultato sono «token visivi»: vettori della stessa forma degli embedding delle parole, messi in fila prima della domanda."),
      part(0.85, 1, "Il modello legge tutto", "Da qui in poi è il solito transformer: la stessa attention mette in relazione token visivi e parole, e la risposta esce token per token.", 4),
    ],
  };
}

function meshAction(tab) {
  const r = tab.regions.mesh;
  const nV = MESH_VERTICES.length;
  const nAll = OBJ_LINES.length;
  return {
    id: "mesh",
    title: "Una mesh 3D scritta come testo",
    category: CATEGORY,
    summary: "Come LLaMA-Mesh: vertici e triangoli nel formato OBJ, coordinate intere; la forma si costruisce riga dopo riga.",
    onStart() {
      tab.prepare("mesh");
    },
    onStop() {
      r.set(nAll);
    },
    steps: [
      {
        title: "Le righe «v»",
        caption: "Esempio scritto a mano. Ogni riga «v x y z» è un vertice. LLaMA-Mesh scala la mesh in [0, 64] e arrotonda le coordinate a interi: numeri corti, quindi pochi token.",
        duration: 6,
        enter() {
          r.set(0);
          tab.showCode({ title: "Esempio: la casetta in formato OBJ (scritta a mano)", file: "esempio", code: OBJ_LINES.join("\n"), language: "bash", highlight: ["v 8 52 24", "f 5 6 10"] });
        },
        update(p, dt) {
          r.set(Math.round(phase(p, 0.05, 0.9) * nV));
          r.spin(dt);
        },
      },
      {
        title: "Le righe «f»",
        caption: "Ogni riga «f a b c» unisce tre vertici (numerati da 1) in un triangolo. È testo normale: il modello non ha bisogno di token nuovi, solo di imparare la «grammatica» delle mesh.",
        duration: 7,
        enter() {
          r.set(nV);
        },
        update(p, dt) {
          r.set(nV + Math.round(phase(p, 0.05, 0.9) * (nAll - nV)));
          r.spin(dt);
        },
      },
      {
        title: "Testo e 3D insieme",
        caption: "LLaMA-Mesh è un LLM addestrato con SFT su dialoghi che mescolano testo e mesh: può generare, spiegare e modificare oggetti conversando. MeshGPT invece codifica i triangoli con un tokenizer dedicato.",
        duration: 5,
        enter() {
          r.set(nAll);
        },
        update(p, dt) {
          r.spin(dt);
        },
      },
      {
        title: "L'altra strada: scrivere codice",
        caption: "Un LLM può anche scrivere uno script Python per Blender (API bpy) che crea la scena: progetti come MeshGen e MCP for Blender collegano il modello al programma, che esegue il codice.",
        duration: 5,
        enter() {
          r.set(nAll);
        },
        update(p, dt) {
          r.spin(dt);
        },
      },
    ],
  };
}

function interpAction(tab) {
  const r = tab.regions.interp;
  return {
    id: "interp",
    title: "Interpretabilità: induction head e feature",
    category: CATEGORY,
    summary: "Una head che copia lo schema [A][B] … [A] → [B]; un autoencoder sparso che trova concetti leggibili.",
    onStart() {
      tab.prepare("interp");
    },
    onStop() {
      r.induction(1);
      r.sae(1);
    },
    steps: [
      {
        title: "Lo schema",
        caption: "«Mario Rossi arriva in ufficio. Poi Mario …»: che parola viene dopo? Anche un bambino direbbe «Rossi», perché l'ha già letta dopo «Mario».",
        duration: 4.5,
        enter() {
          r.induction(0);
          r.sae(0);
          tab.showCode("ours:mha");
        },
      },
      {
        title: "Induction head",
        caption: "Olsson et al. (2022): una induction head, dal secondo «Mario», guarda il token che seguiva il primo «Mario» e ne copia l'informazione. Così il modello impara dal contesto senza cambiare i pesi.",
        duration: 6,
        enter() {
          r.induction(0.01);
        },
        update(p) {
          r.induction(p < 0.85 ? 0.01 + phase(p, 0, 0.8) * 0.79 : 1);
        },
      },
      {
        title: "Neuroni che mescolano concetti",
        caption: "Guardando un vettore di attivazioni, i singoli neuroni spesso rispondono a più concetti insieme: difficili da interpretare uno per uno.",
        duration: 4.5,
        enter() {
          r.induction(1);
          r.sae(0);
        },
        update(p) {
          r.sae(phase(p, 0, 0.9) * 0.5);
        },
      },
      {
        title: "Autoencoder sparso",
        caption: "Il dictionary learning addestra un autoencoder a ricostruire le attivazioni usando poche «feature» accese su un dizionario molto più grande. Anthropic: oltre 4.000 feature da 512 neuroni (2023), fino a 34 milioni su Claude 3 Sonnet (2024). Nomi qui illustrativi.",
        duration: 6,
        enter() {
          r.sae(0.5);
        },
        update(p) {
          r.sae(0.5 + phase(p, 0.05, 0.8) * 0.5);
        },
      },
    ],
  };
}

export function beyondActions(tab) {
  return [multiAction(tab), meshAction(tab), interpAction(tab)];
}
