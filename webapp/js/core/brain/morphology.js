/**
 * Morfologie procedurali dei neuroni: forme tipiche generate con un seme (deterministiche), non
 * ricostruzioni di neuroni veri. Puro: restituisce segmenti (cilindri che si assottigliano) e sfere;
 * three.js li disegna istanziati (neurons.js). Test in tests/brain.test.mjs.
 *
 *   const m = buildMorphology("pyramidal", 7);   // { segments, spheres, axonEnd, inputs, soma, height }
 *   resolveNeuronType("piramidale") → "pyramidal" (accetta anche le chiavi italiane della ricerca)
 *
 * Unità: altezza tipica della cellula ≈ 1, soma nell'origine, "su" = +y (verso la superficie della
 * corteccia), uscita verso +x (il microcircuito scorre da sinistra a destra). Le forme seguono le
 * descrizioni anatomiche di base (e i suggerimenti di modellazione della ricerca, brain_data.json):
 * le dimensioni non verificate sono scelte grafiche, e la vista lo dichiara.
 */
import { hashString, mulberry32 } from "./noise.js";

/** Tipi di segmento (attributo per istanza: colore e comportamento degli impulsi). */
export const SEG = Object.freeze({ dendrite: 0, axon: 1, myelin: 2, node: 3, spine: 4, cilium: 5, body: 6, outer: 7 });
/** Tipi di sfera. */
export const SPH = Object.freeze({ soma: 0, bouton: 1, spine: 2, varicosity: 3, claw: 4 });

/**
 * Tipi di neurone: nome, neurotrasmettitore, forma in una frase (anatomia di base) e alias (le chiavi
 * di brain_data.json della ricerca, così le specifiche dei tab possono usare l'una o l'altra).
 */
export const NEURON_TYPES = Object.freeze({
  pyramidal: {
    label: "Neurone piramidale",
    transmitter: "glutammato (eccitatorio)",
    shape: "Soma a piramide, un dendrite apicale lungo con un ciuffo in cima, cinque o sei alberi basali, spine fitte su tutti i dendriti e un assone con guaina di mielina interrotta dai nodi di Ranvier.",
    aliases: ["piramidale"],
  },
  stellate: {
    label: "Cellula stellata",
    transmitter: "glutammato (eccitatorio)",
    shape: "Soma tondeggiante e dendriti spinosi a stella, senza un vero dendrite apicale; l'assone sale verso gli strati superficiali.",
    aliases: ["stellata_spinosa", "stellata_entorinale", "spiny_stellate"],
  },
  relay: {
    label: "Neurone di relè talamico",
    transmitter: "glutammato (eccitatorio)",
    shape: "Soma medio con dendriti radiali corti e a ciuffo; l'assone mielinizzato sale verso la corteccia.",
    aliases: ["rele_talamico", "thalamic_relay"],
  },
  basket: {
    label: "Interneurone a canestro (PV+)",
    transmitter: "GABA (inibitorio)",
    shape: "Soma medio-piccolo, dendriti lisci a raggiera e un assone ramificatissimo che avvolge di sinapsi il soma dei neuroni vicini (il «canestro»).",
    aliases: ["canestro_pv", "interneuron", "interneurone"],
  },
  granule: {
    label: "Cellula granulare del cervelletto",
    transmitter: "glutammato (eccitatorio)",
    shape: "Soma piccolissimo, circa quattro dendriti corti con un «artiglio» in punta e un assone che sale e si divide a T in una fibra parallela lunghissima, senza mielina.",
    aliases: ["granulare", "cerebellar_granule"],
  },
  purkinje: {
    label: "Cellula di Purkinje",
    transmitter: "GABA (inibitorio)",
    shape: "Soma grande a pera e un albero dendritico enorme, piatto come un ventaglio e coperto di spine; le fibre parallele lo attraversano di traverso, come i fili di un telaio.",
    aliases: [],
  },
  dentate: {
    label: "Cellula granulare del giro dentato",
    transmitter: "glutammato (eccitatorio)",
    shape: "Soma piccolo e tondeggiante, dendriti spinosi a cono verso l'alto e un assone (fibra muschiosa) con grandi bottoni sinaptici.",
    aliases: ["granulo_dentato", "dentate_granule"],
  },
  photoreceptor: {
    label: "Fotorecettore (bastoncello o cono)",
    transmitter: "glutammato",
    shape: "Cellula allungata: segmento esterno pieno di dischi di pigmento (a bastoncino o a cono), segmento interno, nucleo e un terminale sinaptico; niente dendriti.",
    aliases: ["fotorecettore"],
  },
  ganglion: {
    label: "Cellula gangliare della retina",
    transmitter: "glutammato (eccitatorio)",
    shape: "Dendriti piatti a raggiera nello strato più interno della retina e un assone lungo che va al nervo ottico.",
    aliases: ["gangliare_retinica", "retinal_ganglion"],
  },
  hair: {
    label: "Cellula ciliata",
    transmitter: "glutammato (verso il nervo acustico)",
    shape: "Cellula cilindrica con in cima un fascio di stereociglia a gradini; niente assone e niente dendriti: la sinapsi sta alla base della cellula.",
    aliases: ["cellula_ciliata", "hair_cell"],
  },
  bipolar: {
    label: "Neurone bipolare del ganglio spirale",
    transmitter: "glutammato (eccitatorio)",
    shape: "Due prolungamenti opposti: uno verso la cellula ciliata, uno più lungo verso il tronco encefalico.",
    aliases: ["ganglio_spirale", "spiral_ganglion"],
  },
  projection: {
    label: "Neurone di proiezione (forma generica)",
    transmitter: "secondo il nucleo (glutammato, GABA o glicina)",
    shape: "Forma generica, la stessa per i nuclei del tronco encefalico e del cervelletto: soma medio, pochi dendriti e un assone mielinizzato.",
    aliases: ["neurone_di_proiezione"],
  },
  spiny: {
    label: "Neurone spinoso medio dello striato",
    transmitter: "GABA (inibitorio)",
    shape: "Soma medio con dendriti radiali fittissimi di spine (da cui il nome); l'assone va al pallido o alla substantia nigra.",
    aliases: ["spinoso_medio", "medium_spiny"],
  },
  pallidal: {
    label: "Neurone del pallido o del subtalamico (forma generica)",
    transmitter: "GABA (pallido) o glutammato (nucleo subtalamico)",
    shape: "Forma generica: soma medio e pochi dendriti lunghi e lisci.",
    aliases: ["neurone_gangli_base"],
  },
  dopaminergic: {
    label: "Neurone dopaminergico",
    transmitter: "dopamina (modulatoria)",
    shape: "Soma grande con pochi dendriti lunghi e un assone sottilissimo, senza mielina, che nello striato si apre in un cespuglio enorme di rami e varicosità.",
    aliases: ["dopaminergico", "dopamine"],
  },
});

const ALIASES = new Map();
for (const [id, t] of Object.entries(NEURON_TYPES)) {
  ALIASES.set(id, id);
  for (const a of t.aliases) ALIASES.set(a, id);
}

/** Id canonico di un tipo di neurone (accetta gli alias della ricerca), o null. */
export function resolveNeuronType(name) {
  if (typeof name !== "string") return null;
  return ALIASES.get(name.trim().toLowerCase()) ?? null;
}

// ---------------------------------------------------------------- costruttore
class Builder {
  constructor(seed, detail) {
    this.rand = mulberry32(seed);
    this.detail = detail;
    this.segments = []; // [ax, ay, az, bx, by, bz, ra, rb, kind, arcA, arcB]
    this.spheres = []; // [x, y, z, r, kind, arc]
    this.inputs = [];
    this.axonEnd = [0.6, 0, 0];
  }

  /** Numero casuale in [a, b). */
  range(a, b) {
    return a + (b - a) * this.rand();
  }

  seg(a, b, ra, rb, kind, arcA = 0, arcB = 0) {
    this.segments.push(a[0], a[1], a[2], b[0], b[1], b[2], ra, rb, kind, arcA, arcB);
  }

  sphere(p, r, kind, arc = 0) {
    this.spheres.push(p[0], p[1], p[2], r, kind, arc);
  }

  /** Direzione perturbata (curvatura casuale) e tirata verso `bias`. */
  wiggle(dir, amount, bias = null, pull = 0) {
    let x = dir[0] + (this.rand() - 0.5) * amount;
    let y = dir[1] + (this.rand() - 0.5) * amount;
    let z = dir[2] + (this.rand() - 0.5) * amount;
    if (bias) {
      x += bias[0] * pull;
      y += bias[1] * pull;
      z += bias[2] * pull;
    }
    const l = Math.hypot(x, y, z) || 1;
    return [x / l, y / l, z / l];
  }

  /**
   * Un ramo dendritico che cresce a passi, si assottiglia e si biforca (regola di Rall: le figlie
   * hanno raggio ≈ 0,7 della madre). `planar`: resta nel piano x-y (Purkinje). Restituisce le punte.
   */
  tree(start, dir, o) {
    const { length, radius, spines = 0, curl = 0.35, bias = null, pull = 0, planar = false, kind = SEG.dendrite, spread = 0.75, decay = 0.72, arc0 = 0 } = o;
    // con poco dettaglio (rendering software) un livello di biforcazioni in meno negli alberi fitti e
    // rami più corti a passi più lunghi: la forma resta, i segmenti si dimezzano
    const low = this.detail < 0.6;
    const levels = low && o.levels >= 3 ? o.levels - 1 : o.levels;
    const tips = [];
    const grow = (p, d, len, r, level, arc) => {
      const steps = Math.max(low ? 1 : 2, Math.round((len / 0.045) * Math.min(1, 0.4 + 0.6 * this.detail)));
      const step = len / steps;
      let pos = p;
      let dd = d;
      let rr = r;
      let s = arc;
      for (let i = 0; i < steps; i++) {
        dd = this.wiggle(dd, curl, bias, pull);
        if (planar) dd = flatten(dd);
        const next = [pos[0] + dd[0] * step, pos[1] + dd[1] * step, pos[2] + dd[2] * step];
        const rNext = Math.max(0.0035, rr * (1 - 0.18 / steps));
        this.seg(pos, next, rr, rNext, kind, s, s + step);
        // spine: tante quante la densità chiede (rese come densità, non una per una come nel neurone vero)
        if (spines > 0 && level > 0) {
          // con meno dettaglio (rendering software) le spine calano col quadrato: sono migliaia di istanze
          const count = this.rand() < spines * step * this.detail * this.detail * 26 ? 1 + (this.rand() < spines * 0.5 ? 1 : 0) : 0;
          for (let k = 0; k < count; k++) {
            const side = this.wiggle(perpendicular(dd, this.rand()), 0.2);
            const base = [pos[0] + dd[0] * step * this.rand(), pos[1] + dd[1] * step * this.rand(), pos[2] + dd[2] * step * this.rand()];
            const head = [base[0] + side[0] * 0.016, base[1] + side[1] * 0.016, base[2] + side[2] * 0.016];
            this.sphere(head, 0.0075, SPH.spine, s);
          }
        }
        pos = next;
        rr = rNext;
        s += step;
      }
      if (level >= levels) {
        tips.push(pos);
        return;
      }
      const children = this.rand() < 0.18 && level < levels - 1 ? 3 : 2;
      for (let c = 0; c < children; c++) {
        const axis = planar ? [0, 0, 1] : perpendicular(dd, this.rand());
        const angle = (c - (children - 1) / 2) * spread * (0.75 + 0.5 * this.rand());
        let nd = rotate(dd, axis, angle);
        if (planar) nd = flatten(nd);
        grow(pos, nd, len * decay * (0.8 + 0.4 * this.rand()), rr * 0.74, level + 1, s);
      }
    };
    grow(start, dir, length, radius, 0, arc0);
    return tips;
  }

  /**
   * Assone: tratto iniziale nudo, poi (se mielinizzato) internodi bianchi separati da nodi di Ranvier,
   * lungo una polilinea; restituisce la lunghezza percorsa. Gli impulsi saltano da un nodo all'altro.
   */
  axon(points, { radius = 0.008, myelin = true, internode = 0.085, node = 0.012, arc0 = 0 } = {}) {
    let arc = arc0;
    let inSheath = false;
    let left = 0.05; // tratto iniziale (segmento iniziale dell'assone) senza mielina
    for (let i = 0; i < points.length - 1; i++) {
      let a = points[i];
      const b = points[i + 1];
      let remaining = dist(a, b);
      const dir = norm(sub(b, a));
      while (remaining > 1e-6) {
        const piece = Math.min(remaining, left);
        const end = [a[0] + dir[0] * piece, a[1] + dir[1] * piece, a[2] + dir[2] * piece];
        if (myelin && inSheath) this.seg(a, end, radius * 2.3, radius * 2.3, SEG.myelin, arc, arc + piece);
        else this.seg(a, end, radius, radius, myelin && arc > arc0 + 0.04 ? SEG.node : SEG.axon, arc, arc + piece);
        arc += piece;
        remaining -= piece;
        left -= piece;
        a = end;
        if (left <= 1e-6) {
          if (myelin) {
            inSheath = !inSheath;
            left = inSheath ? internode : node;
          } else left = Infinity;
        }
      }
    }
    return arc;
  }

  /** Rami terminali dell'assone con i bottoni sinaptici. */
  terminals(at, dir, { count = 3, length = 0.12, radius = 0.005, bouton = 0.014, arc0 = 0, kind = SPH.bouton } = {}) {
    for (let i = 0; i < count; i++) {
      const d = this.wiggle(rotate(dir, perpendicular(dir, this.rand()), (i - (count - 1) / 2) * 0.55), 0.3);
      const mid = [at[0] + d[0] * length * 0.5, at[1] + d[1] * length * 0.5, at[2] + d[2] * length * 0.5];
      const d2 = this.wiggle(d, 0.6);
      const end = [mid[0] + d2[0] * length * 0.5, mid[1] + d2[1] * length * 0.5, mid[2] + d2[2] * length * 0.5];
      this.seg(at, mid, radius, radius * 0.85, SEG.axon, arc0, arc0 + length * 0.5);
      this.seg(mid, end, radius * 0.85, radius * 0.7, SEG.axon, arc0 + length * 0.5, arc0 + length);
      this.sphere(end, bouton, kind, arc0 + length);
    }
  }

  result(extra = {}) {
    const segs = Float32Array.from(this.segments);
    let minY = 0;
    let maxY = 0;
    let minX = 0;
    let maxX = 0;
    for (let i = 0; i < segs.length; i += 11) {
      minY = Math.min(minY, segs[i + 1], segs[i + 4]);
      maxY = Math.max(maxY, segs[i + 1], segs[i + 4]);
      minX = Math.min(minX, segs[i], segs[i + 3]);
      maxX = Math.max(maxX, segs[i], segs[i + 3]);
    }
    return {
      segments: segs,
      spheres: Float32Array.from(this.spheres),
      inputs: this.inputs,
      axonEnd: this.axonEnd,
      extent: { minX, maxX, minY, maxY },
      ...extra,
    };
  }
}

// ---------------------------------------------------------------- piccola algebra
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const norm = (a) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const flatten = (d) => norm([d[0], d[1], d[2] * 0.05]);
function perpendicular(d, t) {
  const ref = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const a = norm([d[1] * ref[2] - d[2] * ref[1], d[2] * ref[0] - d[0] * ref[2], d[0] * ref[1] - d[1] * ref[0]]);
  const b = [d[1] * a[2] - d[2] * a[1], d[2] * a[0] - d[0] * a[2], d[0] * a[1] - d[1] * a[0]];
  const ang = t * Math.PI * 2;
  return norm([a[0] * Math.cos(ang) + b[0] * Math.sin(ang), a[1] * Math.cos(ang) + b[1] * Math.sin(ang), a[2] * Math.cos(ang) + b[2] * Math.sin(ang)]);
}
function rotate(v, axis, angle) {
  // formula di Rodrigues
  const [kx, ky, kz] = norm(axis);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const dot = v[0] * kx + v[1] * ky + v[2] * kz;
  return norm([
    v[0] * c + (ky * v[2] - kz * v[1]) * s + kx * dot * (1 - c),
    v[1] * c + (kz * v[0] - kx * v[2]) * s + ky * dot * (1 - c),
    v[2] * c + (kx * v[1] - ky * v[0]) * s + kz * dot * (1 - c),
  ]);
}
// ---------------------------------------------------------------- forme
const SHAPES = {
  pyramidal(b) {
    // soma a piramide: una sfera e un cono che sale verso il dendrite apicale
    b.sphere([0, 0, 0], 0.062, SPH.soma);
    b.seg([0, 0.01, 0], [0, 0.13, 0], 0.056, 0.022, SEG.dendrite, 0, 0.12);
    // dendrite apicale con rami obliqui e il ciuffo in cima
    let p = [0, 0.13, 0];
    let d = [0, 1, 0];
    const top = 0.86;
    let arc = 0.12;
    const obliques = Math.round(4 + 2 * b.detail);
    for (let i = 0; i < 9; i++) {
      d = b.wiggle(d, 0.18, [0, 1, 0], 0.5);
      const next = [p[0] + d[0] * (top - 0.13) / 9, p[1] + d[1] * (top - 0.13) / 9, p[2] + d[2] * (top - 0.13) / 9];
      const r = 0.021 - 0.011 * (i / 9);
      b.seg(p, next, r, r - 0.0012, SEG.dendrite, arc, arc + 0.08);
      arc += 0.08;
      if (i > 1 && i % Math.max(1, Math.round(9 / obliques)) === 0) {
        const side = rotate([1, 0, 0], [0, 1, 0], b.range(0, Math.PI * 2));
        b.tree(next, norm([side[0], 0.55, side[2]]), { length: b.range(0.12, 0.2), radius: 0.008, levels: 1, spines: 0.9, arc0: arc });
      }
      p = next;
    }
    const tuft = b.tree(p, [0, 1, 0], { length: 0.16, radius: 0.011, levels: 3, spines: 0.9, spread: 0.85, arc0: arc, bias: [0, 1, 0], pull: 0.15 });
    b.inputs.push(...tuft.slice(0, 4));
    // dendriti basali: 5-6 alberi attorno alla base del soma (5,55 ± 1,47 nelle piramidali umane)
    const basal = 5 + (b.rand() < 0.55 ? 1 : 0);
    for (let i = 0; i < basal; i++) {
      const a = (i / basal) * Math.PI * 2 + b.range(-0.3, 0.3);
      const dir = norm([Math.cos(a), -0.55 - b.range(0, 0.35), Math.sin(a)]);
      const tips = b.tree([dir[0] * 0.05, -0.02, dir[2] * 0.05], dir, { length: b.range(0.14, 0.2), radius: 0.013, levels: 2, spines: 0.8, spread: 0.7 });
      if (i < 2) b.inputs.push(tips[0]);
    }
    // assone: scende verso la sostanza bianca con la mielina e piega verso destra (l'area successiva)
    const end = [
      [0, -0.06, 0],
      [0.01, -0.3, 0],
      [0.05, -0.48, 0],
      [0.22, -0.56, 0],
      [0.5, -0.56, 0],
    ];
    const length = b.axon(end, { radius: 0.0085, myelin: true });
    // collaterale locale (le piramidali parlano anche con i vicini)
    b.terminals([0.01, -0.2, 0], norm([-0.6, -0.2, 0.4]), { count: 2, length: 0.14, arc0: 0.18 });
    b.terminals(end[end.length - 1], [1, 0, 0], { count: 3, length: 0.1, arc0: length });
    b.axonEnd = end[end.length - 1];
  },

  stellate(b) {
    b.sphere([0, 0, 0], 0.056, SPH.soma);
    const n = 7;
    for (let i = 0; i < n; i++) {
      const dir = norm([b.range(-1, 1), b.range(-0.7, 0.7), b.range(-1, 1)]);
      const tips = b.tree([dir[0] * 0.045, dir[1] * 0.045, dir[2] * 0.045], dir, { length: b.range(0.14, 0.2), radius: 0.011, levels: 2, spines: 0.85, spread: 0.8, curl: 0.45 });
      if (i < 3) b.inputs.push(tips[0]);
    }
    // l'assone sale verso gli strati 2/3 della stessa colonna
    const path = [
      [0, 0.05, 0],
      [0.03, 0.25, 0],
      [0.12, 0.42, 0],
      [0.3, 0.5, 0],
    ];
    const length = b.axon(path, { radius: 0.006, myelin: false });
    b.terminals(path[path.length - 1], [1, 0.2, 0], { count: 3, length: 0.12, arc0: length });
    b.axonEnd = path[path.length - 1];
  },

  relay(b) {
    b.sphere([0, 0, 0], 0.066, SPH.soma);
    for (let i = 0; i < 7; i++) {
      const dir = norm([b.range(-1, 0.5), b.range(-1, 1), b.range(-1, 1)]);
      const tips = b.tree([dir[0] * 0.055, dir[1] * 0.055, dir[2] * 0.055], dir, { length: b.range(0.08, 0.12), radius: 0.012, levels: 3, spines: 0.25, spread: 0.9, decay: 0.65, curl: 0.5 });
      if (i < 3) b.inputs.push(tips[0]);
    }
    const path = [
      [0.05, 0.04, 0],
      [0.18, 0.2, 0],
      [0.42, 0.3, 0],
      [0.66, 0.3, 0],
    ];
    const length = b.axon(path, { radius: 0.009, myelin: true });
    b.terminals(path[path.length - 1], [1, 0, 0], { count: 3, length: 0.1, arc0: length });
    b.axonEnd = path[path.length - 1];
  },

  basket(b) {
    b.sphere([0, 0, 0], 0.05, SPH.soma);
    for (let i = 0; i < 5; i++) {
      const dir = norm([b.range(-1, 1), b.range(-0.8, 1), b.range(-1, 1)]);
      const tips = b.tree([dir[0] * 0.04, dir[1] * 0.04, dir[2] * 0.04], dir, { length: b.range(0.16, 0.24), radius: 0.009, levels: 2, spines: 0, spread: 0.6, curl: 0.25 });
      if (i < 2) b.inputs.push(tips[0]);
    }
    // assone ramificatissimo nel piano orizzontale, pieno di bottoni (i «canestri» attorno ai somi vicini)
    const path = [
      [0, -0.05, 0],
      [0.06, -0.14, 0],
      [0.2, -0.16, 0],
    ];
    const length = b.axon(path, { radius: 0.005, myelin: false });
    for (let k = 0; k < 4; k++) {
      const dir = norm([b.range(0.2, 1), b.range(-0.2, 0.2), b.range(-1, 1)]);
      const start = [0.06 + k * 0.04, -0.15, 0];
      const tips = b.tree(start, dir, { length: 0.16, radius: 0.004, levels: 2, spines: 0, kind: SEG.axon, spread: 0.9, curl: 0.4, arc0: length });
      for (const t of tips) b.sphere(t, 0.012, SPH.bouton, length + 0.3);
    }
    b.axonEnd = path[path.length - 1];
  },

  granule(b) {
    b.sphere([0, 0, 0], 0.034, SPH.soma);
    // circa 4 dendriti corti con l'«artiglio» in punta (4,17 in media, nel gatto)
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + b.range(-0.4, 0.4);
      const dir = norm([Math.cos(a), b.range(-0.6, 0.1), Math.sin(a)]);
      const end = [dir[0] * 0.09, dir[1] * 0.09, dir[2] * 0.09];
      b.seg([dir[0] * 0.03, dir[1] * 0.03, dir[2] * 0.03], end, 0.006, 0.004, SEG.dendrite, 0, 0.06);
      for (let k = 0; k < 3; k++) {
        const c = b.wiggle(dir, 1.6);
        b.sphere([end[0] + c[0] * 0.012, end[1] + c[1] * 0.012, end[2] + c[2] * 0.012], 0.007, SPH.claw);
      }
      if (i < 2) b.inputs.push(end);
    }
    // assone ascendente e biforcazione a T: la fibra parallela corre lungo z, perpendicolare al
    // ventaglio delle cellule di Purkinje (che sta nel piano x-y)
    const rise = [
      [0, 0.03, 0],
      [0.01, 0.35, 0],
      [0.02, 0.62, 0],
    ];
    const len = b.axon(rise, { radius: 0.004, myelin: false });
    const t = rise[rise.length - 1];
    const fiber = 0.55;
    b.axon([t, [t[0], t[1], t[2] + fiber]], { radius: 0.004, myelin: false, arc0: len });
    b.axon([t, [t[0], t[1], t[2] - fiber]], { radius: 0.004, myelin: false, arc0: len });
    for (let k = 1; k <= 4; k++) {
      b.sphere([t[0], t[1], t[2] + (fiber * k) / 4.5], 0.008, SPH.varicosity, len + (fiber * k) / 4.5);
      b.sphere([t[0], t[1], t[2] - (fiber * k) / 4.5], 0.008, SPH.varicosity, len + (fiber * k) / 4.5);
    }
    b.axonEnd = t;
  },

  purkinje(b) {
    b.sphere([0, 0, 0], 0.075, SPH.soma);
    // albero a ventaglio, tutto nel piano x-y (piatto), fitto di spine
    const primaries = 1 + (b.rand() < 0.5 ? 1 : 0);
    for (let i = 0; i < primaries; i++) {
      const start = [0, 0.06, 0];
      const dir = norm([(i - (primaries - 1) / 2) * 0.5, 1, 0]);
      const tips = b.tree(start, dir, { length: 0.2, radius: 0.022, levels: Math.round(4 + b.detail), spines: 1.1, planar: true, spread: 0.62, decay: 0.78, curl: 0.25, bias: [0, 1, 0], pull: 0.12 });
      b.inputs.push(...tips.filter((_, k) => k % 5 === 0).slice(0, 5));
    }
    // assone mielinizzato verso i nuclei cerebellari profondi
    const path = [
      [0, -0.07, 0],
      [0.02, -0.3, 0],
      [0.15, -0.45, 0],
      [0.42, -0.5, 0],
    ];
    const length = b.axon(path, { radius: 0.009, myelin: true });
    b.terminals(path[path.length - 1], [1, 0, 0], { count: 3, length: 0.1, arc0: length });
    b.axonEnd = path[path.length - 1];
  },

  dentate(b) {
    b.sphere([0, 0, 0], 0.042, SPH.soma);
    for (let i = 0; i < 3; i++) {
      const dir = norm([(i - 1) * 0.45 + b.range(-0.1, 0.1), 1, b.range(-0.35, 0.35)]);
      const tips = b.tree([0, 0.035, 0], dir, { length: 0.2, radius: 0.009, levels: 3, spines: 0.9, spread: 0.45, bias: [0, 1, 0], pull: 0.2 });
      b.inputs.push(tips[0]);
    }
    // fibra muschiosa con i grandi bottoni (pochi contatti molto forti con CA3)
    const path = [
      [0, -0.04, 0],
      [0.04, -0.2, 0],
      [0.25, -0.26, 0],
      [0.55, -0.26, 0],
    ];
    const length = b.axon(path, { radius: 0.005, myelin: false });
    for (let k = 1; k <= 3; k++) b.sphere([0.12 + k * 0.13, -0.26, 0], 0.022, SPH.bouton, length * (0.4 + k * 0.18));
    b.axonEnd = path[path.length - 1];
  },

  photoreceptor(b) {
    const cone = b.rand() < 0.35;
    // segmento esterno (dischi di pigmento): a bastoncino o a cono
    b.seg([0, 0.48, 0], [0, 0.9, 0], cone ? 0.05 : 0.032, cone ? 0.012 : 0.03, SEG.outer, 0, 0);
    b.seg([0, 0.26, 0], [0, 0.48, 0], 0.036, cone ? 0.05 : 0.034, SEG.body, 0, 0);
    b.sphere([0, 0.14, 0], 0.045, SPH.soma);
    const path = [
      [0, 0.1, 0],
      [0, -0.2, 0],
      [0.06, -0.3, 0],
    ];
    const length = b.axon(path, { radius: 0.007, myelin: false });
    b.sphere(path[path.length - 1], 0.03, SPH.bouton, length);
    b.inputs.push([0, 0.85, 0]);
    b.axonEnd = path[path.length - 1];
  },

  ganglion(b) {
    b.sphere([0, 0, 0], 0.058, SPH.soma);
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 + b.range(-0.3, 0.3);
      const dir = norm([Math.cos(a), 0.35, Math.sin(a)]);
      const tips = b.tree([dir[0] * 0.05, 0.03, dir[2] * 0.05], dir, { length: 0.16, radius: 0.009, levels: 2, spines: 0.1, spread: 0.7, bias: [0, 0.2, 0], pull: 0.1 });
      if (i < 3) b.inputs.push(tips[0]);
    }
    const path = [
      [0.05, -0.02, 0],
      [0.3, -0.06, 0],
      [0.72, -0.08, 0],
    ];
    const length = b.axon(path, { radius: 0.008, myelin: true });
    b.terminals(path[path.length - 1], [1, 0, 0], { count: 2, length: 0.08, arc0: length });
    b.axonEnd = path[path.length - 1];
  },

  hair(b) {
    const outer = b.rand() < 0.5;
    const h = outer ? 0.46 : 0.34;
    b.seg([0, 0, 0], [0, h, 0], 0.055, 0.058, SEG.body, 0, 0);
    b.sphere([0, h * 0.35, 0], 0.04, SPH.soma);
    // stereociglia a gradini: tre file di altezza crescente (a "V" nelle cellule esterne)
    const rows = [0.05, 0.08, 0.115];
    rows.forEach((len, r) => {
      for (let k = -3; k <= 3; k++) {
        const x = (k / 3) * 0.04;
        const z = outer ? -Math.abs(k) * 0.008 + r * 0.014 : r * 0.014 - 0.014;
        b.seg([x, h, z], [x, h + len, z], 0.0045, 0.0035, SEG.cilium, 0, 0);
      }
    });
    // sinapsi a nastro alla base: lì il segnale passa alla fibra del nervo acustico
    b.sphere([0.03, -0.01, 0], 0.018, SPH.bouton, 0);
    b.inputs.push([0, h + 0.1, 0]);
    b.axonEnd = [0.06, -0.02, 0];
  },

  bipolar(b) {
    b.sphere([0, 0, 0], 0.05, SPH.soma);
    // prolungamento periferico verso la cellula ciliata (a sinistra) e assone centrale verso il tronco
    const peripheral = [
      [-0.04, 0.02, 0],
      [-0.2, 0.12, 0],
      [-0.36, 0.2, 0],
    ];
    b.axon(peripheral, { radius: 0.006, myelin: true });
    b.inputs.push(peripheral[peripheral.length - 1]);
    const central = [
      [0.05, -0.01, 0],
      [0.3, -0.06, 0],
      [0.6, -0.08, 0],
    ];
    const length = b.axon(central, { radius: 0.008, myelin: true });
    b.terminals(central[central.length - 1], [1, 0, 0], { count: 3, length: 0.08, arc0: length });
    b.axonEnd = central[central.length - 1];
  },

  projection(b) {
    b.sphere([0, 0, 0], 0.06, SPH.soma);
    for (let i = 0; i < 5; i++) {
      const dir = norm([b.range(-1, 0.3), b.range(-1, 1), b.range(-1, 1)]);
      const tips = b.tree([dir[0] * 0.05, dir[1] * 0.05, dir[2] * 0.05], dir, { length: b.range(0.16, 0.22), radius: 0.011, levels: 2, spines: 0.2, spread: 0.7 });
      if (i < 3) b.inputs.push(tips[0]);
    }
    const path = [
      [0.05, 0, 0],
      [0.25, 0.04, 0],
      [0.6, 0.04, 0],
    ];
    const length = b.axon(path, { radius: 0.009, myelin: true });
    b.terminals(path[path.length - 1], [1, 0, 0], { count: 3, length: 0.09, arc0: length });
    b.axonEnd = path[path.length - 1];
  },

  spiny(b) {
    b.sphere([0, 0, 0], 0.052, SPH.soma);
    for (let i = 0; i < 6; i++) {
      const dir = norm([b.range(-1, 1), b.range(-1, 1), b.range(-1, 1)]);
      const tips = b.tree([dir[0] * 0.045, dir[1] * 0.045, dir[2] * 0.045], dir, { length: b.range(0.15, 0.2), radius: 0.01, levels: 2, spines: 1.6, spread: 0.75 });
      if (i < 3) b.inputs.push(tips[0]);
    }
    const path = [
      [0.04, -0.03, 0],
      [0.25, -0.1, 0],
      [0.55, -0.12, 0],
    ];
    const length = b.axon(path, { radius: 0.006, myelin: false });
    b.terminals(path[path.length - 1], [1, 0, 0], { count: 3, length: 0.09, arc0: length });
    b.axonEnd = path[path.length - 1];
  },

  pallidal(b) {
    b.sphere([0, 0, 0], 0.064, SPH.soma);
    for (let i = 0; i < 4; i++) {
      const dir = norm([b.range(-1, 0.4), b.range(-1, 1), b.range(-1, 1)]);
      const tips = b.tree([dir[0] * 0.055, dir[1] * 0.055, dir[2] * 0.055], dir, { length: b.range(0.28, 0.36), radius: 0.011, levels: 1, spines: 0, spread: 0.5, curl: 0.2 });
      if (i < 2) b.inputs.push(tips[0]);
    }
    const path = [
      [0.05, 0, 0],
      [0.3, 0.03, 0],
      [0.6, 0.03, 0],
    ];
    const length = b.axon(path, { radius: 0.008, myelin: true });
    b.terminals(path[path.length - 1], [1, 0, 0], { count: 3, length: 0.09, arc0: length });
    b.axonEnd = path[path.length - 1];
  },

  dopaminergic(b) {
    b.sphere([0, 0, 0], 0.075, SPH.soma);
    for (let i = 0; i < 4; i++) {
      const dir = norm([b.range(-1, 0.2), b.range(-0.8, 0.8), b.range(-1, 1)]);
      const tips = b.tree([dir[0] * 0.065, dir[1] * 0.065, dir[2] * 0.065], dir, { length: b.range(0.3, 0.38), radius: 0.012, levels: 1, spines: 0, spread: 0.45, curl: 0.25 });
      if (i < 2) b.inputs.push(tips[0]);
    }
    // assone sottilissimo e senza mielina che si apre in un cespuglio enorme di varicosità
    const path = [
      [0.06, 0.02, 0],
      [0.3, 0.1, 0],
      [0.55, 0.12, 0],
    ];
    const length = b.axon(path, { radius: 0.0045, myelin: false });
    const bush = b.tree(path[path.length - 1], [1, 0.1, 0], { length: 0.12, radius: 0.004, levels: Math.round(3 + b.detail), spines: 0, kind: SEG.axon, spread: 1.0, curl: 0.6, decay: 0.85, arc0: length });
    for (const t of bush) b.sphere(t, 0.009, SPH.varicosity, length + 0.4);
    b.axonEnd = path[path.length - 1];
  },
};

/**
 * Morfologia di un neurone: segmenti e sfere in coordinate locali (soma nell'origine).
 * @param {string} type id canonico o alias (resolveNeuronType)
 * @param {number | string} seed seme (stesso seme = stesso neurone)
 * @param {{ detail?: number }} [o] 1 = completo, ~0,6 con il rendering software (meno spine e rami)
 * @returns {{ type, segments: Float32Array, spheres: Float32Array, inputs: number[][], axonEnd: number[], extent }}
 *   segments: 11 numeri per segmento (ax ay az bx by bz ra rb kind arcA arcB), kind in SEG;
 *   spheres: 6 numeri per sfera (x y z r kind arc), kind in SPH; arc = distanza lungo l'assone (impulsi)
 */
export function buildMorphology(type, seed = 1, { detail = 1 } = {}) {
  const id = resolveNeuronType(type) ?? "projection";
  const b = new Builder(hashString(`${id}:${seed}`), detail);
  SHAPES[id](b);
  return { type: id, ...b.result() };
}
