/**
 * Rappresentazione "Cervello" del tab Immagini e video: la specifica (soli dati) per il pannello "Cervello"
 * e il modello 3D del nucleo (js/core/brain/, formato in ARCHITECTURE.md, "Cervello").
 *
 * FILE GENERATO: non modificarlo a mano. I dati vengono dalla ricerca verificata dell'8 novembre 2026
 * (scratchpad/research/brain_data.json, generata il 2026-11-08): tabs["diffusion"] copiato senza
 * tradurre (il nucleo accetta i nomi della ricerca: neuron_type, count_shown, real_count, analogy_map,
 * kind "eccitatoria", strength "debole/metafora", fonti con doi): introduzione, 10 analogie con
 * "In comune", "Dove non regge", forza e fonti, il circuito (gerarchia visiva con feedback) senza il campo
 * descrittivo "layout", regioni, percorsi e marcatori. In più: i fatti globali che citano le fonti dei
 * conteggi del circuito, quattro dati verificati dei suoi tipi di neurone (neuron_types, con la loro
 * fonte) e le due fonti citate dall'introduzione.
 *
 * Per rigenerarlo (dalla radice del repository):
 *   node <scratchpad>/reps-diffusion/gen_brain.mjs <scratchpad>/research/brain_data.json webapp/js/tabs/diffusion/brain.js
 * tests/diffusion-reps.test.mjs controlla che passi normalizeBrainSpec(…, { strict: true }) senza avvisi.
 */
export const brainSpec = {
  "title": "La diffusione e la via visiva del cervello",
  "intro": "I modelli di diffusione generano immagini partendo dal rumore e togliendolo un passo alla volta. Attenzione: l'idea della diffusione viene dalla fisica (termodinamica fuori equilibrio, Sohl-Dickstein et al. 2015), non dal cervello. Le reti dentro il modello (convoluzioni, U-Net) hanno invece radici dirette nello studio della visione, e un esperimento recente collega Stable Diffusion all'attività della corteccia visiva.",
  "regions": [
    "v1",
    "v2v4",
    "it",
    "pfc_dl",
    "parietal_post",
    "retina",
    "thalamus",
    "hippocampus",
    "entorhinal",
    "temporal_ant",
    "ffa",
    "parahippocampal",
    "mt"
  ],
  "markers": [
    "lgn"
  ],
  "pathways": [
    "optic_radiation"
  ],
  "circuit": {
    "title": "Gerarchia visiva con feedback",
    "text": "Dalla retina il segnale passa al nucleo genicolato laterale del talamo e, lungo la radiazione ottica, a V1; poi a V2/V4 e alla corteccia inferotemporale, dove si riconoscono gli oggetti. Ogni passo in avanti ha connessioni che tornano indietro: le aree alte rimandano previsioni e contesto a quelle basse, e la prefrontale porta gli obiettivi.",
    "groups": [
      {
        "id": "photo",
        "label": "Retina: fotorecettori (coni e bastoncelli)",
        "region": "retina",
        "neuron_type": "fotorecettore",
        "count_shown": 12,
        "real_count": 96600000,
        "real_count_text": "circa 4,6 milioni di coni e 92 milioni di bastoncelli per retina",
        "real_count_source": "curcio1990"
      },
      {
        "id": "rgc",
        "label": "Retina: cellule gangliari (nervo ottico)",
        "region": "retina",
        "neuron_type": "gangliare_retinica",
        "count_shown": 8,
        "real_count_text": "da 0,7 a 1,5 milioni per retina",
        "real_count_source": "curcio1990rgc"
      },
      {
        "id": "lgn",
        "label": "Talamo: nucleo genicolato laterale",
        "region": "thalamus",
        "neuron_type": "rele_talamico",
        "count_shown": 6,
        "marker": "lgn"
      },
      {
        "id": "v1",
        "label": "V1: cellule semplici e complesse",
        "region": "v1",
        "neuron_type": "piramidale",
        "count_shown": 12,
        "real_count": 140000000,
        "real_count_text": "circa 1,4 × 10^8 neuroni per emisfero (intervallo 0,9-2 × 10^8)",
        "real_count_source": "leuba1994"
      },
      {
        "id": "v2v4",
        "label": "V2/V4: forme e texture",
        "region": "v2v4",
        "neuron_type": "piramidale",
        "count_shown": 10
      },
      {
        "id": "it",
        "label": "Corteccia inferotemporale: oggetti e categorie",
        "region": "it",
        "neuron_type": "piramidale",
        "count_shown": 8
      },
      {
        "id": "pfc",
        "label": "Prefrontale: obiettivi e aspettative",
        "region": "pfc_dl",
        "neuron_type": "piramidale",
        "count_shown": 6
      }
    ],
    "edges": [
      {
        "from": "photo",
        "to": "rgc",
        "kind": "eccitatoria",
        "label": "via le cellule bipolari (non disegnate)",
        "note": "Semplificazione: tra fotorecettori e cellule gangliari ci sono cellule bipolari, orizzontali e amacrine. Il fotorecettore rilascia glutammato al buio e meno con la luce; le bipolari ON invertono il segno e le OFF no, quindi la via ha due canali di segno opposto."
      },
      {
        "from": "rgc",
        "to": "lgn",
        "kind": "eccitatoria",
        "label": "nervo e tratto ottico"
      },
      {
        "from": "lgn",
        "to": "v1",
        "kind": "eccitatoria",
        "label": "radiazione ottica",
        "pathway": "optic_radiation"
      },
      {
        "from": "v1",
        "to": "v2v4",
        "kind": "eccitatoria"
      },
      {
        "from": "v2v4",
        "to": "it",
        "kind": "eccitatoria"
      },
      {
        "from": "it",
        "to": "pfc",
        "kind": "eccitatoria"
      },
      {
        "from": "pfc",
        "to": "it",
        "kind": "modulatoria",
        "label": "obiettivi e aspettative"
      },
      {
        "from": "it",
        "to": "v2v4",
        "kind": "modulatoria",
        "label": "feedback"
      },
      {
        "from": "v2v4",
        "to": "v1",
        "kind": "modulatoria",
        "label": "feedback"
      },
      {
        "from": "v1",
        "to": "lgn",
        "kind": "modulatoria",
        "label": "ritorno corticotalamico",
        "note": "Glutammatergico, ma con effetto soprattutto modulatorio (Sherman e Guillery 1998)."
      }
    ],
    "analogy_map": [
      {
        "group": "photo",
        "ai_component": "I pixel dell'immagine (512×512×3)"
      },
      {
        "group": "rgc",
        "ai_component": "Encoder del VAE: compressione (fattore 8 per lato)"
      },
      {
        "group": "lgn",
        "ai_component": "Il latente 4×64×64 che entra nella U-Net"
      },
      {
        "group": "v1",
        "ai_component": "Primi blocchi dell'encoder della U-Net: bordi e orientamenti"
      },
      {
        "group": "v2v4",
        "ai_component": "Blocchi intermedi: texture e forme"
      },
      {
        "group": "it",
        "ai_component": "Fondo della U e cross-attention con il testo: il significato"
      },
      {
        "group": "pfc",
        "ai_component": "Il prompt e la classifier-free guidance"
      }
    ],
    "caveat": "Semplificato: nella retina mancano cellule bipolari, orizzontali e amacrine, e le connessioni di ritorno sono glutammatergiche ma agiscono soprattutto da modulatori. Il paragone con decoder e skip connection della U-Net riguarda il flusso di informazione dall'alto in basso, non un'equivalenza anatomica.",
    "sources": [
      {
        "id": "felleman1991",
        "title": "Distributed hierarchical processing in the primate cerebral cortex",
        "authors": "Daniel J. Felleman, David C. Van Essen",
        "year": 1991,
        "venue": "Cerebral Cortex 1(1):1-47",
        "url": "https://doi.org/10.1093/cercor/1.1.1",
        "doi": "10.1093/cercor/1.1.1"
      },
      {
        "id": "sherman1998",
        "title": "On the actions that one nerve cell can have on another: distinguishing «drivers» from «modulators»",
        "authors": "S. Murray Sherman, R. W. Guillery",
        "year": 1998,
        "venue": "PNAS 95(12):7121-7126",
        "url": "https://doi.org/10.1073/pnas.95.12.7121",
        "doi": "10.1073/pnas.95.12.7121"
      },
      {
        "id": "curcio1990",
        "title": "Human photoreceptor topography",
        "authors": "Christine A. Curcio, Kenneth R. Sloan, Robert E. Kalina, Anita E. Hendrickson",
        "year": 1990,
        "venue": "Journal of Comparative Neurology 292(4):497-523",
        "url": "https://doi.org/10.1002/cne.902920402",
        "doi": "10.1002/cne.902920402"
      },
      {
        "id": "curcio1990rgc",
        "title": "Topography of ganglion cells in human retina",
        "authors": "Christine A. Curcio, Kimberly A. Allen",
        "year": 1990,
        "venue": "Journal of Comparative Neurology 300(1):5-25",
        "url": "https://doi.org/10.1002/cne.903000103",
        "doi": "10.1002/cne.903000103"
      },
      {
        "id": "leuba1994",
        "title": "Changes in volume, surface estimate, three-dimensional shape and total number of neurons of the human primary visual cortex from midgestation until old age",
        "authors": "Geneviève Leuba, Rudolf Kraftsik",
        "year": 1994,
        "venue": "Anatomy and Embryology 190(4):351-366",
        "url": "https://bionumbers.hms.harvard.edu/bionumber.aspx?id=105617",
        "note": "Valore (1,4 × 10^8 neuroni in V1 per emisfero, intervallo 0,9-2 × 10^8) letto tramite la voce BioNumbers che cita l'articolo (p. 364)."
      },
      {
        "id": "bastos2012",
        "title": "Canonical microcircuits for predictive coding",
        "authors": "Andre M. Bastos, W. Martin Usrey, Rick A. Adams, George R. Mangun, Pascal Fries, Karl J. Friston",
        "year": 2012,
        "venue": "Neuron 76(4):695-711",
        "url": "https://doi.org/10.1016/j.neuron.2012.10.038",
        "doi": "10.1016/j.neuron.2012.10.038"
      },
      {
        "id": "takagi2023",
        "title": "High-resolution image reconstruction with latent diffusion models from human brain activity",
        "authors": "Yu Takagi, Shinji Nishimoto",
        "year": 2023,
        "venue": "CVPR 2023, pp. 14453-14463",
        "url": "https://doi.org/10.1109/CVPR52729.2023.01389",
        "doi": "10.1109/CVPR52729.2023.01389"
      }
    ]
  },
  "analogies": [
    {
      "id": "dif-unet",
      "ai": "Encoder della U-Net: convoluzioni e ResNet block, la risoluzione scende (64×64 → 32×32 → …)",
      "brain": "Via visiva ventrale: V1 → V2/V4 → corteccia inferotemporale (IT)",
      "regions": [
        "v1",
        "v2v4",
        "it"
      ],
      "text": "Hubel e Wiesel (1962) scoprirono nella V1 del gatto cellule «semplici», che rispondono a una barra con un certo orientamento in un punto preciso, e cellule «complesse», che tollerano piccoli spostamenti. Fukushima (1980) tradusse l'idea nel Neocognitron, da cui discendono le reti convoluzionali (LeCun et al. 1989) e quindi l'encoder della U-Net, che passa da bordi a forme a oggetti mentre la risoluzione scende. Yamins et al. (2014) hanno poi mostrato che gli strati intermedi e alti di una rete convoluzionale predicono bene l'attività di V4 e della corteccia inferotemporale della scimmia.",
      "similar": "Gerarchia: caratteristiche locali e semplici in basso, complesse e più invarianti in alto; gli stessi filtri ripetuti su tutta l'immagine, come neuroni con la stessa preferenza in punti diversi del campo visivo.",
      "different": "La corteccia visiva non usa esattamente gli stessi pesi in ogni punto, ha moltissime connessioni laterali e all'indietro, e vede bene solo al centro del campo visivo. La U-Net di un modello di diffusione è addestrata a predire il rumore, non a riconoscere oggetti.",
      "strength": "forte",
      "sources": [
        {
          "id": "hubel1962",
          "title": "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex",
          "authors": "David H. Hubel, Torsten N. Wiesel",
          "year": 1962,
          "venue": "Journal of Physiology 160(1):106-154",
          "url": "https://doi.org/10.1113/jphysiol.1962.sp006837",
          "doi": "10.1113/jphysiol.1962.sp006837"
        },
        {
          "id": "fukushima1980",
          "title": "Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position",
          "authors": "Kunihiko Fukushima",
          "year": 1980,
          "venue": "Biological Cybernetics 36(4):193-202",
          "url": "https://doi.org/10.1007/BF00344251",
          "doi": "10.1007/BF00344251"
        },
        {
          "id": "lecun1989",
          "title": "Backpropagation applied to handwritten zip code recognition",
          "authors": "Yann LeCun, Bernhard Boser, John S. Denker, et al.",
          "year": 1989,
          "venue": "Neural Computation 1(4):541-551",
          "url": "https://doi.org/10.1162/neco.1989.1.4.541",
          "doi": "10.1162/neco.1989.1.4.541"
        },
        {
          "id": "yamins2014",
          "title": "Performance-optimized hierarchical models predict neural responses in higher visual cortex",
          "authors": "Daniel L. K. Yamins, Ha Hong, Charles F. Cadieu, Ethan A. Solomon, Darren Seibert, James J. DiCarlo",
          "year": 2014,
          "venue": "PNAS 111(23):8619-8624",
          "url": "https://doi.org/10.1073/pnas.1403112111",
          "doi": "10.1073/pnas.1403112111"
        },
        {
          "id": "ronneberger2015",
          "title": "U-Net: convolutional networks for biomedical image segmentation",
          "authors": "Olaf Ronneberger, Philipp Fischer, Thomas Brox",
          "year": 2015,
          "venue": "MICCAI 2015 (arXiv:1505.04597)",
          "url": "https://arxiv.org/abs/1505.04597"
        },
        {
          "id": "curcio1990",
          "title": "Human photoreceptor topography",
          "authors": "Christine A. Curcio, Kenneth R. Sloan, Robert E. Kalina, Anita E. Hendrickson",
          "year": 1990,
          "venue": "Journal of Comparative Neurology 292(4):497-523",
          "url": "https://doi.org/10.1002/cne.902920402",
          "doi": "10.1002/cne.902920402"
        }
      ],
      "ai_section": "4 · L'architettura (La U-Net: encoder, decoder, skip connection)"
    },
    {
      "id": "dif-inferenza",
      "ai": "Processo inverso: dal rumore all'immagine, un passo di denoising alla volta (DDIM)",
      "brain": "Percezione come inferenza: le aspettative dall'alto completano i segnali dal basso (codifica predittiva)",
      "regions": [
        "v1",
        "v2v4",
        "it",
        "pfc_dl"
      ],
      "text": "Rao e Ballard (1999) hanno descritto la percezione come codifica predittiva: le aree alte mandano previsioni verso il basso e quelle basse rimandano in alto solo l'errore. Kok, Jehee e de Lange (2012) hanno visto che un'aspettativa rende più nitida la rappresentazione di uno stimolo in V1. Il denoiser fa qualcosa di simile nello spirito: ripulisce un'immagine rumorosa usando ciò che ha imparato su come sono fatte le immagini, un passo alla volta.",
      "similar": "In entrambi i casi un'ipotesi su «cosa c'è davvero» corregge un segnale rumoroso, per approssimazioni successive.",
      "different": "Il cervello non parte da rumore gaussiano puro e non esegue una serie numerata di passi: percepisce in poche centinaia di millisecondi, in modo continuo. La diffusione è nata dalla termodinamica, non dalla codifica predittiva: il paragone riguarda il principio (usare un modello del mondo per togliere il rumore), non il meccanismo.",
      "strength": "media",
      "sources": [
        {
          "id": "rao1999",
          "title": "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects",
          "authors": "Rajesh P. N. Rao, Dana H. Ballard",
          "year": 1999,
          "venue": "Nature Neuroscience 2:79-87",
          "url": "https://doi.org/10.1038/4580",
          "doi": "10.1038/4580"
        },
        {
          "id": "kok2012",
          "title": "Less is more: expectation sharpens representations in the primary visual cortex",
          "authors": "Peter Kok, Janneke F. M. Jehee, Floris P. de Lange",
          "year": 2012,
          "venue": "Neuron 75(2):265-270",
          "url": "https://doi.org/10.1016/j.neuron.2012.04.034",
          "doi": "10.1016/j.neuron.2012.04.034"
        },
        {
          "id": "sohl2015",
          "title": "Deep unsupervised learning using nonequilibrium thermodynamics",
          "authors": "Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli",
          "year": 2015,
          "venue": "ICML 2015, PMLR 37:2256-2265",
          "url": "https://proceedings.mlr.press/v37/sohl-dickstein15.html"
        },
        {
          "id": "ho2020",
          "title": "Denoising diffusion probabilistic models",
          "authors": "Jonathan Ho, Ajay Jain, Pieter Abbeel",
          "year": 2020,
          "venue": "NeurIPS 2020 (arXiv:2006.11239)",
          "url": "https://arxiv.org/abs/2006.11239"
        }
      ],
      "ai_section": "3 · Il processo inverso: generare"
    },
    {
      "id": "dif-immaginazione",
      "ai": "Generazione da un prompt (text-to-image)",
      "brain": "Immaginazione visiva (mental imagery)",
      "regions": [
        "v1",
        "v2v4",
        "parietal_post",
        "pfc_dl"
      ],
      "text": "Quando immagini una mela, la corteccia visiva primaria si attiva in parte come se la stessi vedendo, guidata dall'alto da aree frontali e parietali (rassegna di Pearson 2019). Non tutti però ci riescono: alcune persone non hanno immagini mentali volontarie (afantasia, Zeman, Dewar e Della Sala 2015). Un modello text-to-image fa qualcosa che chiameremmo immaginazione: produce un'immagine nuova partendo da una descrizione.",
      "similar": "Si genera un contenuto visivo senza stimolo esterno, a partire da conoscenze e da una «richiesta».",
      "different": "Le immagini mentali sono di solito più deboli e instabili della percezione; il modello produce sempre un'immagine definita in pixel, senza esperienza cosciente. L'afantasia mostra che si può pensare bene anche senza immagini mentali.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "pearson2019",
          "title": "The human imagination: the cognitive neuroscience of visual mental imagery",
          "authors": "Joel Pearson",
          "year": 2019,
          "venue": "Nature Reviews Neuroscience 20:624-634",
          "url": "https://doi.org/10.1038/s41583-019-0202-9",
          "doi": "10.1038/s41583-019-0202-9"
        },
        {
          "id": "zeman2015",
          "title": "Lives without imagery – Congenital aphantasia",
          "authors": "Adam Zeman, Michaela Dewar, Sergio Della Sala",
          "year": 2015,
          "venue": "Cortex 73:378-380",
          "url": "https://doi.org/10.1016/j.cortex.2015.05.019",
          "doi": "10.1016/j.cortex.2015.05.019"
        }
      ],
      "ai_section": "Generazione: dal rumore all'immagine"
    },
    {
      "id": "dif-rumore",
      "ai": "Rumore gaussiano ε e campionamento: ogni seed dà un'immagine diversa",
      "brain": "Rumore nel sistema nervoso e ipotesi del campionamento neurale",
      "regions": [
        "v1",
        "retina"
      ],
      "text": "Il sistema nervoso è rumoroso a ogni livello: canali ionici che si aprono a caso, sinapsi che a volte non rilasciano il neurotrasmettitore, fotoni che arrivano in modo irregolare (Faisal, Selen e Wolpert 2008). Secondo un'ipotesi, parte di questa variabilità rappresenterebbe «campioni» delle possibili interpretazioni di uno stimolo (Orbán et al. 2016). Nei modelli di diffusione il rumore è voluto: cambiando il rumore iniziale si ottiene un'immagine diversa per lo stesso prompt.",
      "similar": "Il caso serve a esplorare molte possibilità invece di dare sempre la stessa risposta.",
      "different": "Nel cervello il rumore è soprattutto un limite fisico da gestire, e l'ipotesi del campionamento è discussa, non dimostrata. Nella diffusione il rumore è gaussiano, dosato da una schedule (T = 1000 passi nel training) e riproducibile con lo stesso seed.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "faisal2008",
          "title": "Noise in the nervous system",
          "authors": "A. Aldo Faisal, Luc P. J. Selen, Daniel M. Wolpert",
          "year": 2008,
          "venue": "Nature Reviews Neuroscience 9:292-303",
          "url": "https://doi.org/10.1038/nrn2258",
          "doi": "10.1038/nrn2258"
        },
        {
          "id": "orban2016",
          "title": "Neural variability and sampling-based probabilistic representations in the visual cortex",
          "authors": "Gergő Orbán, Pietro Berkes, József Fiser, Máté Lengyel",
          "year": 2016,
          "venue": "Neuron 92(2):530-543",
          "url": "https://doi.org/10.1016/j.neuron.2016.09.038",
          "doi": "10.1016/j.neuron.2016.09.038"
        }
      ],
      "ai_section": "2 · Il processo diretto (Il rumore gaussiano ε)"
    },
    {
      "id": "dif-vae",
      "ai": "VAE: l'immagine 512×512×3 diventa un latente 4×64×64 (fattore 8 per lato)",
      "brain": "Compressione nella retina e codifica efficiente",
      "regions": [
        "retina",
        "thalamus",
        "v1"
      ],
      "text": "Ogni retina umana ha circa 4,6 milioni di coni e 92 milioni di bastoncelli (Curcio et al. 1990), ma il nervo ottico porta i segnali di sole 0,7–1,5 milioni di cellule gangliari (Curcio e Allen 1990): una forte compressione prima ancora della corteccia. Il VAE di Stable Diffusion comprime i 786.432 numeri di un'immagine 512×512×3 in 16.384 numeri (4×64×64), 48 volte meno, e la generazione avviene in questo spazio compresso. Olshausen e Field (1996) hanno mostrato che cercando un codice sparso ed efficiente per le immagini naturali emergono filtri simili ai campi recettivi di V1.",
      "similar": "Prima si comprime togliendo ridondanza, poi si lavora sul codice compatto.",
      "different": "La retina non deve ricostruire l'immagine e non comprime in modo uniforme: le cellule gangliari sono fittissime vicino alla fovea (fino a 32.000–38.000 per mm²) e rade in periferia. Il VAE invece è addestrato proprio a ricostruire i pixel; la sua idea (una rete che riconosce e una che genera, addestrate insieme) è parente della «macchina di Helmholtz» (Dayan et al. 1995), che si rifà all'idea di Helmholtz della percezione come inferenza, non alla retina.",
      "strength": "media",
      "sources": [
        {
          "id": "curcio1990",
          "title": "Human photoreceptor topography",
          "authors": "Christine A. Curcio, Kenneth R. Sloan, Robert E. Kalina, Anita E. Hendrickson",
          "year": 1990,
          "venue": "Journal of Comparative Neurology 292(4):497-523",
          "url": "https://doi.org/10.1002/cne.902920402",
          "doi": "10.1002/cne.902920402"
        },
        {
          "id": "curcio1990rgc",
          "title": "Topography of ganglion cells in human retina",
          "authors": "Christine A. Curcio, Kimberly A. Allen",
          "year": 1990,
          "venue": "Journal of Comparative Neurology 300(1):5-25",
          "url": "https://doi.org/10.1002/cne.903000103",
          "doi": "10.1002/cne.903000103"
        },
        {
          "id": "olshausen1996",
          "title": "Emergence of simple-cell receptive field properties by learning a sparse code for natural images",
          "authors": "Bruno A. Olshausen, David J. Field",
          "year": 1996,
          "venue": "Nature 381:607-609",
          "url": "https://doi.org/10.1038/381607a0",
          "doi": "10.1038/381607a0"
        },
        {
          "id": "dayan1995",
          "title": "The Helmholtz machine",
          "authors": "Peter Dayan, Geoffrey E. Hinton, Radford M. Neal, Richard S. Zemel",
          "year": 1995,
          "venue": "Neural Computation 7(5):889-904",
          "url": "https://doi.org/10.1162/neco.1995.7.5.889",
          "doi": "10.1162/neco.1995.7.5.889"
        },
        {
          "id": "kingma2014",
          "title": "Auto-encoding variational Bayes",
          "authors": "Diederik P. Kingma, Max Welling",
          "year": 2014,
          "venue": "ICLR 2014 (arXiv:1312.6114)",
          "url": "https://arxiv.org/abs/1312.6114"
        },
        {
          "id": "rombach2022",
          "title": "High-resolution image synthesis with latent diffusion models",
          "authors": "Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer",
          "year": 2022,
          "venue": "CVPR 2022 (arXiv:2112.10752)",
          "url": "https://arxiv.org/abs/2112.10752"
        }
      ],
      "ai_section": "4 · L'architettura (Il latente e il VAE)",
      "pathways": [
        "optic_radiation"
      ],
      "markers": [
        "lgn"
      ]
    },
    {
      "id": "dif-concetti",
      "ai": "Tokenizer e text encoder (CLIP) + cross-attention: come entra il testo",
      "brain": "Concept cells e neuroni multimodali",
      "regions": [
        "hippocampus",
        "entorhinal",
        "it",
        "temporal_ant"
      ],
      "text": "Quiroga et al. (2005) hanno trovato nel lobo temporale mediale umano neuroni che rispondono a una persona famosa in foto diverse, in un disegno e al suo nome scritto. Goh et al. (2021) hanno cercato qualcosa di simile in CLIP, il modello che collega immagini e testo da cui deriva il text encoder di Stable Diffusion: hanno trovato neuroni «multimodali», per esempio uno che risponde a foto di ragni, alla parola «spider» e all'Uomo Ragno, e li hanno paragonati esplicitamente alle concept cells.",
      "similar": "Un'unità risponde al concetto, comunque si presenti: foto, disegno o parola.",
      "different": "Nel cervello le concept cells stanno in strutture legate alla memoria e il codice è sparso; in CLIP le unità multimodali emergono dall'addestramento su coppie immagine-testo e molte rispondono a più concetti insieme. La cross-attention che inietta il testo nella U-Net non ha un equivalente anatomico noto.",
      "strength": "media",
      "sources": [
        {
          "id": "quiroga2005",
          "title": "Invariant visual representation by single neurons in the human brain",
          "authors": "R. Quian Quiroga, L. Reddy, G. Kreiman, C. Koch, I. Fried",
          "year": 2005,
          "venue": "Nature 435:1102-1107",
          "url": "https://doi.org/10.1038/nature03687",
          "doi": "10.1038/nature03687"
        },
        {
          "id": "goh2021",
          "title": "Multimodal neurons in artificial neural networks",
          "authors": "Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, Chris Olah",
          "year": 2021,
          "venue": "Distill 6(3)",
          "url": "https://doi.org/10.23915/distill.00030",
          "doi": "10.23915/distill.00030"
        },
        {
          "id": "radford2021",
          "title": "Learning transferable visual models from natural language supervision (CLIP)",
          "authors": "Alec Radford, Jong Wook Kim, Chris Hallacy, et al.",
          "year": 2021,
          "venue": "ICML 2021 (arXiv:2103.00020)",
          "url": "https://arxiv.org/abs/2103.00020"
        }
      ],
      "ai_section": "4 · L'architettura (Tokenizer e text encoder; Cross-attention)"
    },
    {
      "id": "dif-takagi",
      "ai": "Il latente z del VAE e il condizionamento testuale c di Stable Diffusion",
      "brain": "Ricostruzione di immagini viste a partire dall'attività fMRI",
      "regions": [
        "v1",
        "v2v4",
        "it",
        "ffa",
        "parahippocampal"
      ],
      "text": "Takagi e Nishimoto (2023) hanno preso registrazioni fMRI già esistenti di persone che guardavano foto (il Natural Scenes Dataset) e hanno ricostruito immagini simili usando Stable Diffusion, senza riaddestrarlo. Con semplici regressioni lineari, l'attività delle prime aree visive (come V1) prevedeva il latente z, cioè la disposizione grossolana dell'immagine, e quella delle aree visive alte prevedeva il condizionamento c, cioè il significato. Le due parti del modello si «agganciano» così a due livelli della gerarchia visiva.",
      "similar": "Il dettaglio spaziale corrisponde alle aree visive basse, il significato a quelle alte, come la divisione tra latente e testo in Stable Diffusion.",
      "different": "Il cervello non contiene un latente di Stable Diffusion: è il modello lineare a tradurre un'attività nell'altra. La qualità delle ricostruzioni dipende molto dal modello generativo, che aggiunge dettagli plausibili non presenti nei dati cerebrali.",
      "strength": "forte",
      "sources": [
        {
          "id": "takagi2023",
          "title": "High-resolution image reconstruction with latent diffusion models from human brain activity",
          "authors": "Yu Takagi, Shinji Nishimoto",
          "year": 2023,
          "venue": "CVPR 2023, pp. 14453-14463",
          "url": "https://doi.org/10.1109/CVPR52729.2023.01389",
          "doi": "10.1109/CVPR52729.2023.01389"
        }
      ],
      "ai_section": "4 · L'architettura di Stable Diffusion"
    },
    {
      "id": "dif-guidance",
      "ai": "Classifier-free guidance: spingere la previsione verso il prompt",
      "brain": "Attenzione e aspettative dall'alto che favoriscono ciò che si cerca",
      "regions": [
        "pfc_dl",
        "parietal_post",
        "v2v4"
      ],
      "text": "Quando cerchi le chiavi sul tavolo, i neuroni visivi che rispondono a quell'oggetto vengono favoriti rispetto agli altri: è la competizione «pilotata» dall'alto descritta da Desimone e Duncan (1995). La classifier-free guidance (Ho e Salimans 2022) fa un'operazione simile nello spirito: calcola il rumore previsto con e senza prompt e amplifica la differenza di un fattore scelto da chi genera, così l'immagine segue di più la richiesta.",
      "similar": "Un obiettivo dall'alto amplifica ciò che è coerente con esso.",
      "different": "La guidance è una formula (un'estrapolazione tra due previsioni) applicata a ogni passo; l'attenzione biologica è un processo dinamico con limiti di capacità. Un fattore di guidance troppo alto produce immagini innaturali, un effetto senza un chiaro equivalente percettivo.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "desimone1995",
          "title": "Neural mechanisms of selective visual attention",
          "authors": "Robert Desimone, John Duncan",
          "year": 1995,
          "venue": "Annual Review of Neuroscience 18:193-222",
          "url": "https://doi.org/10.1146/annurev.ne.18.030195.001205",
          "doi": "10.1146/annurev.ne.18.030195.001205"
        },
        {
          "id": "ho2022cfg",
          "title": "Classifier-free diffusion guidance",
          "authors": "Jonathan Ho, Tim Salimans",
          "year": 2022,
          "venue": "arXiv:2207.12598",
          "url": "https://arxiv.org/abs/2207.12598"
        }
      ],
      "ai_section": "3 · Il processo inverso (Classifier-free guidance)"
    },
    {
      "id": "dif-video",
      "ai": "Video: frame coerenti nel tempo (strati o attention lungo l'asse del tempo)",
      "brain": "Area MT/V5: la percezione del movimento",
      "regions": [
        "mt",
        "v1",
        "parietal_post"
      ],
      "text": "Nella corteccia esiste un'area specializzata per il movimento, MT/V5, che Zeki et al. (1991) localizzarono nell'uomo con la PET. Una paziente con lesioni in questa zona di entrambi gli emisferi non percepiva più il movimento in modo continuo (Zihl, von Cramon e Mai 1983). Anche i modelli video devono collegare istanti successivi: molti aggiungono a una rete per immagini strati che lavorano lungo l'asse del tempo (Blattmann et al. 2023).",
      "similar": "Il movimento richiede un'elaborazione dedicata che metta in relazione istanti successivi.",
      "different": "MT è un'area precisa che misura direzione e velocità del movimento (Born e Bradley 2005); nei modelli video non esiste un modulo del movimento separato, e la coerenza nel tempo emerge dall'addestramento. Le reti «a due flussi» per riconoscere azioni (Simonyan e Zisserman 2014) dichiarano una somiglianza con l'ipotesi delle due vie visive (oggetti e movimento), pur senza approfondirla; i modelli generativi di video in genere non fanno nemmeno questo.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "zeki1991",
          "title": "A direct demonstration of functional specialization in human visual cortex",
          "authors": "S. Zeki, J. D. G. Watson, C. J. Lueck, K. J. Friston, C. Kennard, R. S. J. Frackowiak",
          "year": 1991,
          "venue": "Journal of Neuroscience 11(3):641-649",
          "url": "https://doi.org/10.1523/JNEUROSCI.11-03-00641.1991",
          "doi": "10.1523/JNEUROSCI.11-03-00641.1991"
        },
        {
          "id": "zihl1983",
          "title": "Selective disturbance of movement vision after bilateral brain damage",
          "authors": "J. Zihl, D. von Cramon, N. Mai",
          "year": 1983,
          "venue": "Brain 106(2):313-340",
          "url": "https://doi.org/10.1093/brain/106.2.313",
          "doi": "10.1093/brain/106.2.313"
        },
        {
          "id": "born2005",
          "title": "Structure and function of visual area MT",
          "authors": "Richard T. Born, David C. Bradley",
          "year": 2005,
          "venue": "Annual Review of Neuroscience 28:157-189",
          "url": "https://doi.org/10.1146/annurev.neuro.26.041002.131052",
          "doi": "10.1146/annurev.neuro.26.041002.131052"
        },
        {
          "id": "blattmann2023",
          "title": "Align your latents: high-resolution video synthesis with latent diffusion models",
          "authors": "Andreas Blattmann, Robin Rombach, Huan Ling, et al.",
          "year": 2023,
          "venue": "CVPR 2023 (arXiv:2304.08818)",
          "url": "https://arxiv.org/abs/2304.08818"
        },
        {
          "id": "simonyan2014",
          "title": "Two-stream convolutional networks for action recognition in videos",
          "authors": "Karen Simonyan, Andrew Zisserman",
          "year": 2014,
          "venue": "NeurIPS 2014 (arXiv:1406.2199)",
          "url": "https://arxiv.org/abs/1406.2199"
        }
      ],
      "ai_section": "Video"
    },
    {
      "id": "dif-sogno",
      "ai": "Generare senza un'immagine in ingresso, partendo dal solo rumore",
      "brain": "Il sogno e l'algoritmo «veglia-sonno»",
      "regions": [
        "v1",
        "v2v4",
        "it"
      ],
      "text": "Horikawa et al. (2013) hanno svegliato più volte volontari appena addormentati e hanno decodificato dall'attività della corteccia visiva alcune categorie di ciò che riferivano di aver sognato. Già Hinton et al. (1995) avevano proposto per le reti generative un algoritmo «veglia-sonno»: nella fase di veglia i dati veri passano per le connessioni di riconoscimento e la rete aggiusta quelle generative; nella fase di sonno la rete genera esempi inventati e su quelli aggiusta le connessioni di riconoscimento. Un'ipotesi recente (Hoel 2021) vede i sogni come dati «strani» prodotti per evitare l'overfitting.",
      "similar": "Il cervello e i modelli generativi possono produrre contenuti visivi senza uno stimolo esterno.",
      "different": "La funzione del sogno è ancora discussa e l'idea dei sogni come regolarizzazione è un'ipotesi; l'algoritmo veglia-sonno non è quello usato da Stable Diffusion. Un modello di diffusione genera solo quando glielo chiediamo e non ha un ciclo di sonno.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "horikawa2013",
          "title": "Neural decoding of visual imagery during sleep",
          "authors": "T. Horikawa, M. Tamaki, Y. Miyawaki, Y. Kamitani",
          "year": 2013,
          "venue": "Science 340(6132):639-642",
          "url": "https://doi.org/10.1126/science.1234330",
          "doi": "10.1126/science.1234330"
        },
        {
          "id": "hinton1995",
          "title": "The «wake-sleep» algorithm for unsupervised neural networks",
          "authors": "Geoffrey E. Hinton, Peter Dayan, Brendan J. Frey, Radford M. Neal",
          "year": 1995,
          "venue": "Science 268(5214):1158-1161",
          "url": "https://doi.org/10.1126/science.7761831",
          "doi": "10.1126/science.7761831"
        },
        {
          "id": "hoel2021",
          "title": "The overfitted brain: dreams evolved to assist generalization",
          "authors": "Erik Hoel",
          "year": 2021,
          "venue": "Patterns 2(5):100244",
          "url": "https://doi.org/10.1016/j.patter.2021.100244",
          "doi": "10.1016/j.patter.2021.100244"
        }
      ],
      "ai_section": "Generazione"
    }
  ],
  "facts": [
    {
      "claim": "Fotorecettori per retina",
      "value": "4,6 milioni di coni (4,08–5,29) e 92 milioni di bastoncelli (77,9–107,3)",
      "source": {
        "id": "curcio1990",
        "title": "Human photoreceptor topography",
        "authors": "Christine A. Curcio, Kenneth R. Sloan, Robert E. Kalina, Anita E. Hendrickson",
        "year": 1990,
        "venue": "Journal of Comparative Neurology 292(4):497-523",
        "url": "https://doi.org/10.1002/cne.902920402",
        "doi": "10.1002/cne.902920402"
      }
    },
    {
      "claim": "Cellule gangliari per retina",
      "value": "da 0,7 a 1,5 milioni",
      "source": {
        "id": "curcio1990rgc",
        "title": "Topography of ganglion cells in human retina",
        "authors": "Christine A. Curcio, Kimberly A. Allen",
        "year": 1990,
        "venue": "Journal of Comparative Neurology 300(1):5-25",
        "url": "https://doi.org/10.1002/cne.903000103",
        "doi": "10.1002/cne.903000103"
      }
    },
    {
      "claim": "Neuroni in V1 per emisfero",
      "value": "circa 1,4 × 10^8 (da 0,9 a 2 × 10^8)",
      "source": {
        "id": "leuba1994",
        "title": "Changes in volume, surface estimate, three-dimensional shape and total number of neurons of the human primary visual cortex from midgestation until old age",
        "authors": "Geneviève Leuba, Rudolf Kraftsik",
        "year": 1994,
        "venue": "Anatomy and Embryology 190(4):351-366",
        "url": "https://bionumbers.hms.harvard.edu/bionumber.aspx?id=105617",
        "note": "Valore (1,4 × 10^8 neuroni in V1 per emisfero, intervallo 0,9-2 × 10^8) letto tramite la voce BioNumbers che cita l'articolo (p. 364)."
      }
    },
    {
      "claim": "Cellule gangliari della retina: densità massima",
      "value": "Massimo di 32.000–38.000 cellule/mm² in un anello a 0,4–2 mm dal centro della fovea.",
      "source": {
        "id": "curcio1990rgc",
        "title": "Topography of ganglion cells in human retina",
        "authors": "Christine A. Curcio, Kimberly A. Allen",
        "year": 1990,
        "venue": "Journal of Comparative Neurology 300(1):5-25",
        "url": "https://doi.org/10.1002/cne.903000103",
        "doi": "10.1002/cne.903000103"
      }
    },
    {
      "claim": "Coni per cellula gangliare",
      "value": "Rapporto coni:gangliari da 2,9 a 7,5 nelle diverse retine.",
      "source": {
        "id": "curcio1990rgc",
        "title": "Topography of ganglion cells in human retina",
        "authors": "Christine A. Curcio, Kimberly A. Allen",
        "year": 1990,
        "venue": "Journal of Comparative Neurology 300(1):5-25",
        "url": "https://doi.org/10.1002/cne.903000103",
        "doi": "10.1002/cne.903000103"
      }
    },
    {
      "claim": "Talamo (genicolato laterale): ingressi «driver» e «modulatori»",
      "value": "Gli ingressi dalla retina sono «driver», quelli di ritorno dallo strato 6 della corteccia sono soprattutto modulatori.",
      "source": {
        "id": "sherman1998",
        "title": "On the actions that one nerve cell can have on another: distinguishing «drivers» from «modulators»",
        "authors": "S. Murray Sherman, R. W. Guillery",
        "year": 1998,
        "venue": "PNAS 95(12):7121-7126",
        "url": "https://doi.org/10.1073/pnas.95.12.7121",
        "doi": "10.1073/pnas.95.12.7121"
      }
    },
    {
      "claim": "Neuroni di V1: dalla tensione agli impulsi",
      "value": "Relazione soglia-lineare tra potenziale e frequenza: soglia ≈ −54,4 mV, guadagno ≈ 7,2 spike/s per mV (gatto, V1).",
      "source": {
        "id": "carandini2000",
        "title": "Membrane potential and firing rate in cat primary visual cortex",
        "authors": "Matteo Carandini, David Ferster",
        "year": 2000,
        "venue": "Journal of Neuroscience 20(1):470-484",
        "url": "https://doi.org/10.1523/JNEUROSCI.20-01-00470.2000",
        "doi": "10.1523/JNEUROSCI.20-01-00470.2000"
      }
    }
  ],
  "sources": [
    {
      "id": "sohl2015",
      "title": "Deep unsupervised learning using nonequilibrium thermodynamics",
      "authors": "Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli",
      "year": 2015,
      "venue": "ICML 2015, PMLR 37:2256-2265",
      "url": "https://proceedings.mlr.press/v37/sohl-dickstein15.html"
    },
    {
      "id": "takagi2023",
      "title": "High-resolution image reconstruction with latent diffusion models from human brain activity",
      "authors": "Yu Takagi, Shinji Nishimoto",
      "year": 2023,
      "venue": "CVPR 2023, pp. 14453-14463",
      "url": "https://doi.org/10.1109/CVPR52729.2023.01389",
      "doi": "10.1109/CVPR52729.2023.01389"
    }
  ]
};
