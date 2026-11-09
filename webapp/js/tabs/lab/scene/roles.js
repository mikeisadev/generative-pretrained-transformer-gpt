/**
 * Ruolo di un modulo o di un peso nella scena del Laboratorio (soli dati e funzioni pure).
 *
 * Il ruolo decide il colore sobrio del contenitore (bordo, titolo, velatura) e la voce della legenda.
 * Colori: gli stessi dei ruoli della rete neurale del nucleo (js/core/neuralNet/roles.js, a loro volta
 * quelli del tab Transformer), così embedding, attention, feed-forward, normalizzazioni e uscita hanno
 * lo stesso colore in tutte le rappresentazioni. Il colore non è mai l'unico segnale: ogni contenitore
 * ha il suo nome e la riga Python.
 */
import { ROLES as NET_ROLES } from "../../../core/neuralNet/roles.js";

export const LAB_ROLES = Object.freeze({
  embedding: { label: "Embedding", color: NET_ROLES.embedding.color, text: "Tabelle di pesi: ogni token (o posizione) prende la sua riga." },
  attention: { label: "Attention", color: NET_ROLES.attention.color, text: "Key, query, value di ogni head e la proiezione: i token si scambiano informazioni." },
  mlp: { label: "Feed-forward", color: NET_ROLES.mlp.color, text: "Due nn.Linear con la ReLU in mezzo: ogni token elaborato per conto suo." },
  norm: { label: "Normalizzazione", color: NET_ROLES.norm.color, text: "LayerNorm: γ e β riscalano e spostano ogni dimensione." },
  output: { label: "Uscita", color: NET_ROLES.output.color, text: "lm_head: un punteggio (logit) per ogni carattere del vocabolario." },
  block: { label: "Contenitore", color: "#94a3b8", text: "Moduli che ne contengono altri (GPTLanguageModel, blocks, Block)." },
  buffer: { label: "Buffer (non è un peso)", color: "#64748b", text: "Numeri fissi del modulo, che il training non cambia (la maschera tril)." },
  other: { label: "Altro modulo", color: "#cbd5e1", text: "Un modulo che il codice del corso non ha: viene da un'architettura modificata." },
});

/** Ruolo di un nodo dell'albero (o di un nodo senza pesi) dal percorso e dal tipo PyTorch. */
export function roleOf(path, type = "") {
  const p = String(path ?? "");
  const t = String(type ?? "");
  const last = p.split(".").filter((k) => !/^\d+$/.test(k)).at(-1) ?? "";
  if (/Embedding/.test(t) || /(^|_)(emb|embed|embedding|wte|wpe)(_|$)|embedding/.test(last)) return "embedding";
  if (/LayerNorm|RMSNorm|GroupNorm|BatchNorm/.test(t) || /^(ln|norm)|_?norm\d*$|^ln_?\w*$/.test(last)) return "norm";
  if (/^(lm_head|output|head_out|unembed)$/.test(last)) return "output";
  if (/(^|\.)(sa|attn|attention|self_attn|heads|mha)(\.|$)/.test(p) || /Attention|^Head$/.test(t)) return "attention";
  if (/(^|\.)(ffwd|mlp|ffn|feed_forward|net|fc\d*|c_fc|c_proj)(\.|$)/.test(p) || /FeedForward|MLP/.test(t)) return "mlp";
  if (p === "" || /^blocks?$|^layers?$|^h$/.test(last) || /Block|Sequential|ModuleList|GPT/.test(t)) return "block";
  return "other";
}
