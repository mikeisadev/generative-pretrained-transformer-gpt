/**
 * Regione "Tokenizer BPE a byte": una frase come sequenza di token, che si accorciano fusione
 * dopo fusione. L'algoritmo è vero (trainBpe in sim.js, come karpathy/minbpe).
 */
import { trainBpe, tokenText } from "./sim.js";
import { PALETTE, Plate, TokenStrip } from "./shapes.js";

export const BPE_TEXT = "il gatto e la gatta guardano il gattino. È tardi!";
export const BPE_MERGES = 10;

const tokenColor = (id) => (id < 256 ? { fill: "#1e293b", stroke: "#475569" } : { fill: `hsl(${(id * 47) % 360} 55% 24%)`, stroke: `hsl(${(id * 47) % 360} 70% 62%)` });
const shown = (s) => s.replace(/ /g, "␣");

export function buildBpe(group) {
  const { history, vocab } = trainBpe(BPE_TEXT, BPE_MERGES);
  const strip = new TokenStrip({ w: 34, h: 12.5, font: 0.95, gap: 0.24 }).place(group, 0, 1.7);
  const header = new Plate({ w: 34, h: 2.4, size: 0.65, align: "center" }).place(group, 0, -6.6);
  const footer = new Plate({ w: 34, h: 2.2, size: 0.55, align: "center", color: PALETTE.muted }).place(group, 0, -10.4);
  footer.set("Vocabolario = 256 byte + una voce per ogni fusione.\nGPT-2: 50.257 token · Llama 3: 128.256 token.");

  const api = {
    history,
    vocab,
    /**
     * Mostra il passo k della storia. Con `pending` = true evidenzia la coppia che sta per
     * essere fusa (nei token del passo precedente).
     */
    show(k, pending = false) {
      const key = k * 2 + (pending ? 1 : 0);
      if (key === api.key) return;
      api.key = key;
      const step = history[Math.min(k, history.length - 1)];
      const ids = pending && k > 0 ? history[k - 1].ids : step.ids;
      const pair = step.pair;
      const tokens = ids.map((id, i) => {
        const bytes = vocab.get(id);
        const inPair = pending && pair && ((id === pair[0] && ids[i + 1] === pair[1]) || (id === pair[1] && ids[i - 1] === pair[0]));
        const fresh = !pending && pair && id === step.idx;
        const c = tokenColor(id);
        return { text: shown(tokenText(bytes)), fill: inPair ? "#713f12" : c.fill, stroke: inPair ? "#facc15" : fresh ? "#fde047" : c.stroke, strokeWidth: inPair || fresh ? 4 : 2 };
      });
      strip.set(tokens);
      const bytes = history[0].ids.length;
      if (!pair) header.set([{ text: `Partenza: ${bytes} byte UTF-8 = ${bytes} token`, bold: true }, { text: "«È» occupa 2 byte (0xC3 0x88): ogni testo è rappresentabile, nessun token sconosciuto.", color: PALETTE.muted }]);
      else {
        const a = shown(tokenText(vocab.get(pair[0])));
        const b = shown(tokenText(vocab.get(pair[1])));
        header.set([
          { text: `Fusione ${k}: «${a}» + «${b}» → token ${step.idx} «${shown(tokenText(vocab.get(step.idx)))}» (compare ${step.count} volte)`, bold: true },
          { text: `token: ${bytes} byte → ${ids.length} token${pending ? "" : ""}`, color: PALETTE.muted },
        ]);
      }
    },
  };
  api.show(history.length - 1);
  return api;
}
