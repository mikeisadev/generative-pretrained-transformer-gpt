/**
 * Il cervello del Laboratorio pesi (soli dati: niente three.js, niente DOM). Generato da
 * scratchpad/tab-lab-core/make_brain.py a partire dalla ricerca verificata (research/brain_data.json,
 * tabs["lab"]): analogie, circuito del cervelletto, fatti e fonti copiati senza riscriverli. Le cautele
 * della verifica sui numeri "da ricontrollare" stanno in fondo alla nota di ogni fonte segnata e di ogni
 * numero che la cita. Formato: ARCHITECTURE.md, "Cervello" (normalizeBrainSpec di core/brain/spec.js,
 * provato con strict: true).
 */
export const brainSpec = {
  "title": "Pesi e sinapsi: il Laboratorio e il cervello",
  "intro": "Il Laboratorio pesi mostra i 209.729 numeri del GPT del corso e come cambiano durante il training. Nel cervello l'equivalente più vicino sono le sinapsi e la loro plasticità: le schede confrontano pesi e sinapsi, training e apprendimento, modelli diversi e cervelli diversi. E dicono dove il confronto non regge, a cominciare dal fatto che un peso non è una sinapsi.",
  "regions": [
    "hippocampus",
    "entorhinal",
    "cerebellum",
    "brainstem",
    "v1",
    "thalamus",
    "mt",
    "parietal_post",
    "temporal_ant",
    "pfc_dl",
    "motor"
  ],
  "markers": [
    "lgn"
  ],
  "pathways": [
    "perforant_path"
  ],
  "circuit": {
    "title": "Cervelletto: apprendimento con un segnale d'errore",
    "text": "Dalla corteccia, attraverso i nuclei pontini, arrivano le fibre muschiose; le cellule granulari le espandono e, con le fibre parallele, raggiungono le cellule di Purkinje. Dall'oliva inferiore arriva a ogni cellula di Purkinje una fibra rampicante che segnala l'errore e modifica le sinapsi fibra parallela–Purkinje; l'uscita passa dai nuclei cerebellari profondi e dal talamo fino alla corteccia motoria.",
    "groups": [
      {
        "id": "pont",
        "label": "Nuclei pontini: ingresso dalla corteccia",
        "region": "brainstem",
        "neuron_type": "neurone_di_proiezione",
        "count_shown": 6
      },
      {
        "id": "grc",
        "label": "Strato granulare: cellule granulari (espansione)",
        "region": "cerebellum",
        "neuron_type": "granulare",
        "count_shown": 16,
        "real_count": 101000000000,
        "real_count_text": "circa 101 miliardi (stima stereologica, Andersen et al. 1992)",
        "real_count_source": "andersen1992"
      },
      {
        "id": "mli",
        "label": "Interneuroni dello strato molecolare (canestro e stellate)",
        "region": "cerebellum",
        "neuron_type": "canestro_pv",
        "count_shown": 4,
        "note": "Le cellule a canestro del cervelletto sono un tipo diverso da quelle della corteccia: le accomuna l'inibizione vicino al soma."
      },
      {
        "id": "pkj",
        "label": "Cellule di Purkinje: i «pesi» che imparano",
        "region": "cerebellum",
        "neuron_type": "purkinje",
        "count_shown": 6,
        "real_count": 30500000,
        "real_count_text": "circa 30,5 milioni (Andersen et al. 1992)",
        "real_count_source": "andersen1992"
      },
      {
        "id": "io",
        "label": "Oliva inferiore: il segnale d'errore",
        "region": "brainstem",
        "neuron_type": "neurone_di_proiezione",
        "count_shown": 4
      },
      {
        "id": "dcn",
        "label": "Nuclei cerebellari profondi: l'uscita",
        "region": "cerebellum",
        "neuron_type": "neurone_di_proiezione",
        "count_shown": 4
      },
      {
        "id": "thal",
        "label": "Talamo (nucleo ventrolaterale)",
        "region": "thalamus",
        "neuron_type": "rele_talamico",
        "count_shown": 4
      },
      {
        "id": "m1",
        "label": "Corteccia motoria primaria: il movimento",
        "region": "motor",
        "neuron_type": "piramidale",
        "count_shown": 6
      }
    ],
    "edges": [
      {
        "from": "pont",
        "to": "grc",
        "kind": "eccitatoria",
        "label": "fibre muschiose"
      },
      {
        "from": "pont",
        "to": "dcn",
        "kind": "eccitatoria",
        "label": "collaterali delle fibre muschiose"
      },
      {
        "from": "grc",
        "to": "pkj",
        "kind": "eccitatoria",
        "label": "fibre parallele",
        "note": "Circa 175.000 sinapsi da fibre parallele per cellula di Purkinje nel ratto (Napper e Harvey 1988): sono queste a cambiare con l'errore."
      },
      {
        "from": "grc",
        "to": "mli",
        "kind": "eccitatoria"
      },
      {
        "from": "mli",
        "to": "pkj",
        "kind": "inibitoria"
      },
      {
        "from": "io",
        "to": "pkj",
        "kind": "eccitatoria",
        "label": "fibra rampicante: segnale d'errore",
        "note": "Circa 1 impulso al secondo (Streng et al. 2018); con le fibre parallele attive insieme produce depressione a lungo termine (Ito e Kano 1982)."
      },
      {
        "from": "pkj",
        "to": "dcn",
        "kind": "inibitoria"
      },
      {
        "from": "dcn",
        "to": "thal",
        "kind": "eccitatoria"
      },
      {
        "from": "thal",
        "to": "m1",
        "kind": "eccitatoria"
      },
      {
        "from": "m1",
        "to": "pont",
        "kind": "eccitatoria",
        "label": "l'anello si chiude"
      },
      {
        "from": "dcn",
        "to": "io",
        "kind": "inibitoria",
        "label": "via nucleo-olivare",
        "note": "I nuclei profondi mandano fibre GABAergiche all'oliva inferiore e ne regolano l'attività: l'anello dell'errore è chiuso anche nel cervello (dato anatomico di base)."
      }
    ],
    "analogy_map": [
      {
        "group": "pont",
        "ai_component": "I dati di training in ingresso (il batch di caratteri)"
      },
      {
        "group": "grc",
        "ai_component": "Attivazioni nascoste: tante unità, ognuna con pochi ingressi (espansione)"
      },
      {
        "group": "mli",
        "ai_component": "Nessun equivalente diretto (inibizione che regola i tempi)"
      },
      {
        "group": "pkj",
        "ai_component": "Lo strato di uscita (lm_head): i pesi che il segnale d'errore corregge"
      },
      {
        "group": "io",
        "ai_component": "La loss e il suo gradiente: quanto si è sbagliato"
      },
      {
        "group": "dcn",
        "ai_component": "L'uscita del modello (i logits)"
      },
      {
        "group": "thal",
        "ai_component": "Il passaggio verso la generazione"
      },
      {
        "group": "m1",
        "ai_component": "Il comportamento: il carattere generato"
      }
    ],
    "caveat": "È il circuito biologico più vicino all'idea di «loss che corregge i pesi», ma l'errore corregge un solo stadio e non viene propagato all'indietro attraverso molti layer come nella backpropagation. I numeri reali sono stime per il cervelletto umano; le connessioni per cellula sono misurate nel ratto.",
    "sources": [
      {
        "id": "marr1969",
        "title": "A theory of cerebellar cortex",
        "authors": "David Marr",
        "year": 1969,
        "venue": "Journal of Physiology 202(2):437-470",
        "url": "https://doi.org/10.1113/jphysiol.1969.sp008820",
        "doi": "10.1113/jphysiol.1969.sp008820"
      },
      {
        "id": "albus1971",
        "title": "A theory of cerebellar function",
        "authors": "James S. Albus",
        "year": 1971,
        "venue": "Mathematical Biosciences 10(1-2):25-61",
        "url": "https://doi.org/10.1016/0025-5564(71)90051-4",
        "doi": "10.1016/0025-5564(71)90051-4"
      },
      {
        "id": "itokano1982",
        "title": "Long-lasting depression of parallel fiber-Purkinje cell transmission induced by conjunctive stimulation of parallel fibers and climbing fibers in the cerebellar cortex",
        "authors": "Masao Ito, Masanobu Kano",
        "year": 1982,
        "venue": "Neuroscience Letters 33(3):253-258",
        "url": "https://doi.org/10.1016/0304-3940(82)90380-9",
        "doi": "10.1016/0304-3940(82)90380-9"
      },
      {
        "id": "raymond2018",
        "title": "Computational principles of supervised learning in the cerebellum",
        "authors": "Jennifer L. Raymond, Javier F. Medina",
        "year": 2018,
        "venue": "Annual Review of Neuroscience 41:233-253",
        "url": "https://doi.org/10.1146/annurev-neuro-080317-061948",
        "doi": "10.1146/annurev-neuro-080317-061948"
      },
      {
        "id": "napper1988",
        "title": "Number of parallel fiber synapses on an individual Purkinje cell in the cerebellum of the rat",
        "authors": "Ruth M. A. Napper, Robert J. Harvey",
        "year": 1988,
        "venue": "Journal of Comparative Neurology 274(2):168-177",
        "url": "https://doi.org/10.1002/cne.902740204",
        "doi": "10.1002/cne.902740204"
      },
      {
        "id": "andersen1992",
        "title": "A quantitative study of the human cerebellum with unbiased stereological techniques",
        "authors": "Birgitte Bo Andersen, Lise Korbo, Bente Pakkenberg",
        "year": 1992,
        "venue": "Journal of Comparative Neurology 326(4):549-560",
        "url": "https://doi.org/10.1002/cne.903260405",
        "doi": "10.1002/cne.903260405",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifre 101 miliardi / 30,5 milioni coerenti; da ricontrollare la descrizione del campione («5 uomini anziani»)."
      },
      {
        "id": "streng2018",
        "title": "Complex spike wars: a new hope",
        "authors": "Martha L. Streng, Laurentiu S. Popa, Timothy J. Ebner",
        "year": 2018,
        "venue": "The Cerebellum 17(6):735-746",
        "url": "https://doi.org/10.1007/s12311-018-0960-3",
        "doi": "10.1007/s12311-018-0960-3",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente; divergenza 5–10 cellule e frequenze da ricontrollare sul testo."
      },
      {
        "id": "litwinkumar2017",
        "title": "Optimal degrees of synaptic connectivity",
        "authors": "Ashok Litwin-Kumar, Kameron Decker Harris, Richard Axel, Haim Sompolinsky, L. F. Abbott",
        "year": 2017,
        "venue": "Neuron 93(5):1153-1164",
        "url": "https://doi.org/10.1016/j.neuron.2017.01.030",
        "doi": "10.1016/j.neuron.2017.01.030"
      }
    ],
    "layout": "orizzontale, da sinistra a destra nell'ordine dei gruppi (come la rete AI corrispondente)"
  },
  "analogies": [
    {
      "id": "lab-sinapsi",
      "ai": "I 209.729 pesi del GPT del corso",
      "brain": "Sinapsi: la forza di una connessione",
      "regions": [
        "hippocampus"
      ],
      "text": "Nel cervello ciò che più somiglia a un peso è la forza di una sinapsi, legata anche alla sua dimensione. Bartol et al. (2015) hanno ricostruito al microscopio elettronico sinapsi dell'ippocampo del ratto e stimato almeno 26 livelli di forza distinguibili, cioè circa 4,7 bit per sinapsi. Un peso del GPT in float32 occupa 32 bit; la sola neocorteccia umana ha circa 1,5–1,6 × 10^14 sinapsi (Tang et al. 2001; Pakkenberg et al. 2003).",
      "similar": "Il comportamento dipende dalla forza delle connessioni, e imparare vuol dire cambiarla.",
      "different": "Una sinapsi non è un numero: cambia nel tempo (si affatica, si potenzia per secondi o per ore) e rilascia il neurotrasmettitore in modo probabilistico (Faisal et al. 2008). Per questo confrontare «parametri» e «sinapsi» come se fossero la stessa unità è fuorviante.",
      "strength": "media",
      "sources": [
        {
          "id": "bartol2015",
          "title": "Nanoconnectomic upper bound on the variability of synaptic plasticity",
          "authors": "Thomas M. Bartol Jr, Cailey Bromer, Justin Kinney, Michael A. Chirillo, Jennifer N. Bourne, Kristen M. Harris, Terrence J. Sejnowski",
          "year": 2015,
          "venue": "eLife 4:e10778",
          "url": "https://doi.org/10.7554/eLife.10778",
          "doi": "10.7554/eLife.10778"
        },
        {
          "id": "tang2001",
          "title": "Total regional and global number of synapses in the human brain neocortex",
          "authors": "Yong Tang, Jens R. Nyengaard, Didima M. G. De Groot, Hans Jørgen G. Gundersen",
          "year": 2001,
          "venue": "Synapse 41(3):258-273",
          "url": "https://doi.org/10.1002/syn.1083",
          "doi": "10.1002/syn.1083",
          "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifra 164 × 10^12 coerente; da ricontrollare la descrizione del campione («5 giovani maschi»)."
        },
        {
          "id": "pakkenberg2003",
          "title": "Aging and the human neocortex",
          "authors": "Bente Pakkenberg, Dorte Pelvig, Lisbeth Marner, et al.",
          "year": 2003,
          "venue": "Experimental Gerontology 38(1-2):95-99",
          "url": "https://doi.org/10.1016/S0531-5565(02)00151-1",
          "doi": "10.1016/S0531-5565(02)00151-1"
        },
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
          "id": "corso",
          "title": "Configurazione e traccia del training del GPT del corso",
          "authors": "Repository del corso (GPT didattico, dal notebook di Andrej Karpathy)",
          "year": 2026,
          "venue": "webapp/data/model.json e webapp/data/training/trace.json",
          "url": "webapp/data/training/trace.json",
          "note": "Dati esportati dal Python del corso: 209.729 parametri, 5000 passi, loss di validazione 4,40 → 1,81, durata 286,7 s, dropout 0,0, AdamW con lr 1e-3."
        }
      ],
      "ai_section": "I pesi del modello"
    },
    {
      "id": "lab-ltp",
      "ai": "Il training: in 5000 passi la loss di validazione scende da 4,40 a 1,81 (snapshot agli step 0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000)",
      "brain": "Potenziamento a lungo termine (LTP) e regola di Hebb",
      "regions": [
        "hippocampus",
        "entorhinal"
      ],
      "text": "Nel 1973 Bliss e Lømo stimolarono ad alta frequenza la via perforante, che va dalla corteccia entorinale al giro dentato, e videro che le risposte restavano potenziate per ore: la prima descrizione completa di un cambiamento duraturo nella forza delle sinapsi, il tipo di meccanismo che Hebb (1949) aveva ipotizzato. Nel laboratorio vedi il corrispettivo artificiale, con un meccanismo molto diverso: ogni passo di AdamW sposta un poco ciascun peso, e confrontando gli snapshot si vede quali matrici cambiano di più.",
      "similar": "Cambiamenti piccoli e cumulativi delle connessioni producono un comportamento nuovo.",
      "different": "LTP dipende dall'attività dei due neuroni collegati (una regola locale); la backpropagation calcola per ogni peso quanto ha contribuito all'errore finale, usando informazioni che una sinapsi reale non ha (Lillicrap et al. 2020). Il cervello impara di continuo, mentre il GPT del corso ha imparato in 286,7 secondi e poi è rimasto fisso.",
      "strength": "media",
      "sources": [
        {
          "id": "bliss1973",
          "title": "Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path",
          "authors": "T. V. P. Bliss, T. Lømo",
          "year": 1973,
          "venue": "Journal of Physiology 232(2):331-356",
          "url": "https://doi.org/10.1113/jphysiol.1973.sp010273",
          "doi": "10.1113/jphysiol.1973.sp010273"
        },
        {
          "id": "hebb1949",
          "title": "The Organization of Behavior: A Neuropsychological Theory",
          "authors": "Donald O. Hebb",
          "year": 1949,
          "venue": "Wiley, New York (DOI della ristampa Erlbaum/Psychology Press 2002)",
          "url": "https://doi.org/10.4324/9781410612403",
          "doi": "10.4324/9781410612403"
        },
        {
          "id": "lillicrap2020",
          "title": "Backpropagation and the brain",
          "authors": "Timothy P. Lillicrap, Adam Santoro, Luke Marris, Colin J. Akerman, Geoffrey Hinton",
          "year": 2020,
          "venue": "Nature Reviews Neuroscience 21:335-346",
          "url": "https://doi.org/10.1038/s41583-020-0277-3",
          "doi": "10.1038/s41583-020-0277-3"
        },
        {
          "id": "loshchilov2019",
          "title": "Decoupled weight decay regularization",
          "authors": "Ilya Loshchilov, Frank Hutter",
          "year": 2019,
          "venue": "ICLR 2019 (arXiv:1711.05101)",
          "url": "https://arxiv.org/abs/1711.05101"
        },
        {
          "id": "corso",
          "title": "Configurazione e traccia del training del GPT del corso",
          "authors": "Repository del corso (GPT didattico, dal notebook di Andrej Karpathy)",
          "year": 2026,
          "venue": "webapp/data/model.json e webapp/data/training/trace.json",
          "url": "webapp/data/training/trace.json",
          "note": "Dati esportati dal Python del corso: 209.729 parametri, 5000 passi, loss di validazione 4,40 → 1,81, durata 286,7 s, dropout 0,0, AdamW con lr 1e-3."
        }
      ],
      "ai_section": "Fasi di training",
      "pathways": [
        "perforant_path"
      ]
    },
    {
      "id": "lab-cervelletto",
      "ai": "La loss e il gradiente: il segnale che dice quanto si è sbagliato",
      "brain": "Il cervelletto: le fibre rampicanti portano il segnale d'errore",
      "regions": [
        "cerebellum",
        "brainstem"
      ],
      "text": "Il cervelletto è il caso più chiaro di apprendimento supervisionato nel cervello. Marr (1969) e Albus (1971) proposero (Albus con un paragone esplicito con il percettrone) che le sinapsi tra fibre parallele e cellule di Purkinje cambino quando arriva un segnale «insegnante» dalle fibre rampicanti, che partono dall'oliva inferiore (più tardi interpretato come segnale d'errore). Ito e Kano (1982) trovarono la depressione a lungo termine prevista da Albus (Marr aveva ipotizzato un rafforzamento): se fibre parallele e fibra rampicante sono attive insieme, quelle sinapsi si indeboliscono. Quando un movimento manca il bersaglio, l'errore corregge le sinapsi giuste, come la loss corregge i pesi (Raymond e Medina 2018).",
      "similar": "C'è un segnale esplicito di errore che guida il cambiamento delle connessioni giuste.",
      "different": "La fibra rampicante dà un segnale grossolano, circa 1 impulso al secondo (Streng et al. 2018), che arriva a tutta la cellula, non un gradiente preciso per ogni sinapsi; e corregge un solo stadio, non una catena di layer. Di norma ogni cellula di Purkinje riceve una sola fibra rampicante, ma ci sono eccezioni (Busch e Hansel 2023).",
      "strength": "forte",
      "sources": [
        {
          "id": "marr1969",
          "title": "A theory of cerebellar cortex",
          "authors": "David Marr",
          "year": 1969,
          "venue": "Journal of Physiology 202(2):437-470",
          "url": "https://doi.org/10.1113/jphysiol.1969.sp008820",
          "doi": "10.1113/jphysiol.1969.sp008820"
        },
        {
          "id": "albus1971",
          "title": "A theory of cerebellar function",
          "authors": "James S. Albus",
          "year": 1971,
          "venue": "Mathematical Biosciences 10(1-2):25-61",
          "url": "https://doi.org/10.1016/0025-5564(71)90051-4",
          "doi": "10.1016/0025-5564(71)90051-4"
        },
        {
          "id": "itokano1982",
          "title": "Long-lasting depression of parallel fiber-Purkinje cell transmission induced by conjunctive stimulation of parallel fibers and climbing fibers in the cerebellar cortex",
          "authors": "Masao Ito, Masanobu Kano",
          "year": 1982,
          "venue": "Neuroscience Letters 33(3):253-258",
          "url": "https://doi.org/10.1016/0304-3940(82)90380-9",
          "doi": "10.1016/0304-3940(82)90380-9"
        },
        {
          "id": "raymond2018",
          "title": "Computational principles of supervised learning in the cerebellum",
          "authors": "Jennifer L. Raymond, Javier F. Medina",
          "year": 2018,
          "venue": "Annual Review of Neuroscience 41:233-253",
          "url": "https://doi.org/10.1146/annurev-neuro-080317-061948",
          "doi": "10.1146/annurev-neuro-080317-061948"
        },
        {
          "id": "streng2018",
          "title": "Complex spike wars: a new hope",
          "authors": "Martha L. Streng, Laurentiu S. Popa, Timothy J. Ebner",
          "year": 2018,
          "venue": "The Cerebellum 17(6):735-746",
          "url": "https://doi.org/10.1007/s12311-018-0960-3",
          "doi": "10.1007/s12311-018-0960-3",
          "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente; divergenza 5–10 cellule e frequenze da ricontrollare sul testo."
        },
        {
          "id": "busch2023",
          "title": "Climbing fiber multi-innervation of mouse Purkinje dendrites with arborization common to human",
          "authors": "Silas E. Busch, Christian Hansel",
          "year": 2023,
          "venue": "Science 381(6656):420-427",
          "url": "https://doi.org/10.1126/science.adi1024",
          "doi": "10.1126/science.adi1024"
        }
      ],
      "ai_section": "Loss"
    },
    {
      "id": "lab-init",
      "ai": "Step 0: pesi inizializzati a caso (loss 4,40, peggio che tirare a indovinare tra 65 caratteri)",
      "brain": "Il cervello alla nascita: cablaggio innato prodotto dall'evoluzione",
      "regions": [
        "brainstem",
        "v1",
        "cerebellum"
      ],
      "text": "Il GPT parte da numeri casuali: allo step 0 la sua loss (4,40) è persino peggiore di quella di chi tira a indovinare tra 65 caratteri (ln 65 ≈ 4,17). Un cervello non parte mai da zero: Zador (2019) ricorda che molti animali mostrano comportamenti complessi subito dopo la nascita, perché il genoma codifica regole di cablaggio selezionate dall'evoluzione. Zador chiama «collo di bottiglia genomico» il fatto che queste regole debbano stare, compresse, in un genoma relativamente piccolo.",
      "similar": "In entrambi i casi l'esperienza modella una struttura che c'è già.",
      "different": "Il punto di partenza è opposto: casuale nel GPT, ricco di struttura nel cervello. Questa scheda serve soprattutto a capire dove l'analogia non regge.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "zador2019",
          "title": "A critique of pure learning and what artificial neural networks can learn from animal brains",
          "authors": "Anthony M. Zador",
          "year": 2019,
          "venue": "Nature Communications 10:3770",
          "url": "https://doi.org/10.1038/s41467-019-11786-6",
          "doi": "10.1038/s41467-019-11786-6"
        },
        {
          "id": "corso",
          "title": "Configurazione e traccia del training del GPT del corso",
          "authors": "Repository del corso (GPT didattico, dal notebook di Andrej Karpathy)",
          "year": 2026,
          "venue": "webapp/data/model.json e webapp/data/training/trace.json",
          "url": "webapp/data/training/trace.json",
          "note": "Dati esportati dal Python del corso: 209.729 parametri, 5000 passi, loss di validazione 4,40 → 1,81, durata 286,7 s, dropout 0,0, AdamW con lr 1e-3."
        }
      ],
      "ai_section": "Step 0"
    },
    {
      "id": "lab-periodi",
      "ai": "Fasi di training: cosa impara il modello all'inizio e cosa dopo (confronto tra snapshot)",
      "brain": "Periodi critici dello sviluppo",
      "regions": [
        "v1",
        "thalamus"
      ],
      "text": "Wiesel e Hubel (1963) chiusero un occhio a gattini nei primi mesi di vita: quell'occhio restava poi quasi incapace di attivare la corteccia visiva, mentre la stessa chiusura negli adulti non aveva questo effetto. Esistono finestre di tempo in cui l'esperienza modella i circuiti in modo durevole (Hensch 2005). Achille, Rovere e Soatto (2019) hanno trovato qualcosa di simile nelle reti profonde: sfocare le immagini nelle prime epoche di training lascia un danno permanente, sfocarle più tardi no.",
      "similar": "Le prime fasi dell'apprendimento fissano strutture difficili da cambiare dopo.",
      "different": "Nel cervello i periodi critici sono regolati da meccanismi biologici precisi, per esempio la maturazione dell'inibizione (Hensch 2005); nelle reti l'effetto dipende dalla dinamica dell'ottimizzazione. Il GPT del corso non è stato studiato da questo punto di vista: nel laboratorio puoi solo osservare come cambiano i pesi tra gli snapshot.",
      "strength": "media",
      "sources": [
        {
          "id": "wiesel1963",
          "title": "Single-cell responses in striate cortex of kittens deprived of vision in one eye",
          "authors": "Torsten N. Wiesel, David H. Hubel",
          "year": 1963,
          "venue": "Journal of Neurophysiology 26(6):1003-1017",
          "url": "https://doi.org/10.1152/jn.1963.26.6.1003",
          "doi": "10.1152/jn.1963.26.6.1003"
        },
        {
          "id": "hensch2005",
          "title": "Critical period plasticity in local cortical circuits",
          "authors": "Takao K. Hensch",
          "year": 2005,
          "venue": "Nature Reviews Neuroscience 6:877-888",
          "url": "https://doi.org/10.1038/nrn1787",
          "doi": "10.1038/nrn1787"
        },
        {
          "id": "achille2019",
          "title": "Critical learning periods in deep networks",
          "authors": "Alessandro Achille, Matteo Rovere, Stefano Soatto",
          "year": 2019,
          "venue": "ICLR 2019 (arXiv:1711.08856)",
          "url": "https://arxiv.org/abs/1711.08856"
        }
      ],
      "ai_section": "Confronto tra fasi di training",
      "markers": [
        "lgn"
      ]
    },
    {
      "id": "lab-prima-dopo",
      "ai": "Confronto prima/dopo: differenza dei pesi tra due snapshot",
      "brain": "Cambiamenti strutturali misurati prima e dopo un apprendimento",
      "regions": [
        "hippocampus",
        "mt",
        "parietal_post"
      ],
      "text": "Woollett e Maguire (2011) hanno seguito con la risonanza magnetica 79 aspiranti tassisti di Londra mentre studiavano la mappa della città: chi ha superato l'esame mostrava più materia grigia nell'ippocampo posteriore, chi non l'ha superato e i 31 controlli no. Draganski et al. (2004) hanno visto aumentare la materia grigia in un'area visiva del movimento (hMT/V5) e nel solco intraparietale dopo tre mesi di giocoleria, con una riduzione quando si smetteva. Il laboratorio fa lo stesso tipo di confronto prima/dopo, ma sui pesi.",
      "similar": "Si misura lo stesso sistema prima e dopo l'apprendimento e si cerca dove è cambiato.",
      "different": "La risonanza misura volumi di tessuto, cioè milioni di cellule insieme, e non dice quali sinapsi siano cambiate; nel laboratorio conosciamo ogni singolo peso, prima e dopo, con precisione assoluta.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "woollett2011",
          "title": "Acquiring «the Knowledge» of London's layout drives structural brain changes",
          "authors": "Katherine Woollett, Eleanor A. Maguire",
          "year": 2011,
          "venue": "Current Biology 21(24):2109-2114",
          "url": "https://doi.org/10.1016/j.cub.2011.11.018",
          "doi": "10.1016/j.cub.2011.11.018"
        },
        {
          "id": "maguire2000",
          "title": "Navigation-related structural change in the hippocampi of taxi drivers",
          "authors": "Eleanor A. Maguire, David G. Gadian, Ingrid S. Johnsrude, et al.",
          "year": 2000,
          "venue": "PNAS 97(8):4398-4403",
          "url": "https://doi.org/10.1073/pnas.070039597",
          "doi": "10.1073/pnas.070039597"
        },
        {
          "id": "draganski2004",
          "title": "Neuroplasticity: changes in grey matter induced by training",
          "authors": "Bogdan Draganski, Christian Gaser, Volker Busch, Gerhard Schuierer, Ulrich Bogdahn, Arne May",
          "year": 2004,
          "venue": "Nature 427:311-312",
          "url": "https://doi.org/10.1038/427311a",
          "doi": "10.1038/427311a"
        }
      ],
      "ai_section": "Confronto"
    },
    {
      "id": "lab-connettoma",
      "ai": "Leggere tutti i pesi del modello, matrice per matrice",
      "brain": "Connettomica: mappe complete delle connessioni",
      "regions": [
        "temporal_ant",
        "v1"
      ],
      "text": "Il primo sistema nervoso mappato per intero è quello del verme C. elegans: 302 neuroni, circa 5.000 sinapsi chimiche, 2.000 giunzioni neuromuscolari e 600 giunzioni elettriche (White et al. 1986). Nel 2024 FlyWire ha ricostruito il cervello di un moscerino adulto: 139.255 neuroni e circa 50 milioni di sinapsi chimiche (Dorkenwald et al. 2024). Un millimetro cubo di corteccia temporale umana contiene circa 57.000 cellule e 150 milioni di sinapsi e occupa 1,4 petabyte di immagini (Shapson-Coe et al. 2024).",
      "similar": "Avere la mappa di chi è collegato a chi è il primo passo per capire come un sistema calcola.",
      "different": "Nel GPT conosciamo ogni peso con 32 bit di precisione e possiamo eseguire il modello; un connettoma dice chi è collegato a chi e quante sinapsi ci sono, ma non dà direttamente la forza di ogni sinapsi né la neuromodulazione. Anche il millimetro cubo di corteccia visiva di topo del MICrONS (circa 524 milioni di sinapsi) è una frazione minuscola di un cervello.",
      "strength": "media",
      "sources": [
        {
          "id": "white1986",
          "title": "The structure of the nervous system of the nematode Caenorhabditis elegans",
          "authors": "J. G. White, E. Southgate, J. N. Thomson, S. Brenner",
          "year": 1986,
          "venue": "Philosophical Transactions of the Royal Society B 314(1165):1-340",
          "url": "https://doi.org/10.1098/rstb.1986.0056",
          "doi": "10.1098/rstb.1986.0056"
        },
        {
          "id": "dorkenwald2024",
          "title": "Neuronal wiring diagram of an adult brain",
          "authors": "Sven Dorkenwald, Arie Matsliah, Amy R. Sterling, et al. (FlyWire Consortium)",
          "year": 2024,
          "venue": "Nature 634:124-138",
          "url": "https://doi.org/10.1038/s41586-024-07558-y",
          "doi": "10.1038/s41586-024-07558-y"
        },
        {
          "id": "shapsoncoe2024",
          "title": "A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution",
          "authors": "Alexander Shapson-Coe, Michał Januszewski, Daniel R. Berger, et al. (ultimi autori Viren Jain, Jeff W. Lichtman)",
          "year": 2024,
          "venue": "Science 384(6696):eadk4858",
          "url": "https://doi.org/10.1126/science.adk4858",
          "doi": "10.1126/science.adk4858"
        },
        {
          "id": "microns2025",
          "title": "Functional connectomics spanning multiple areas of mouse visual cortex",
          "authors": "The MICrONS Consortium",
          "year": 2025,
          "venue": "Nature 640:435-447",
          "url": "https://doi.org/10.1038/s41586-025-08790-w",
          "doi": "10.1038/s41586-025-08790-w",
          "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente; i 524 milioni di sinapsi vengono da comunicati e sito del progetto, non dall'articolo."
        }
      ],
      "ai_section": "I pesi del modello"
    },
    {
      "id": "lab-specie",
      "ai": "Confronto tra modelli di dimensioni diverse",
      "brain": "Anatomia comparata: quanti neuroni hanno specie diverse",
      "regions": [
        "cerebellum",
        "pfc_dl"
      ],
      "text": "Contare non basta, ma aiuta a farsi un'idea delle scale: il verme C. elegans ha 302 neuroni, il moscerino circa 139.000, il topo circa 71 milioni (Herculano-Houzel, Mota e Lent 2006), l'essere umano circa 86 miliardi (Azevedo et al. 2009) e l'elefante africano circa 257 miliardi, quasi tutti nel cervelletto (Herculano-Houzel et al. 2014). Il GPT del corso ha 209.729 parametri, distribuiti in 4 blocchi con vettori da 64 numeri.",
      "similar": "Sistemi più grandi possono rappresentare più cose, e confrontare taglie diverse aiuta a capire cosa cambia con la scala.",
      "different": "Neuroni e parametri sono unità diverse: un parametro somiglia più a una sinapsi che a un neurone, e anche questo paragone zoppica. Tra le specie le capacità dipendono da dove stanno i neuroni e da come sono collegati: l'elefante ne ha tre volte più di noi, ma nella corteccia un terzo dei nostri.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "white1986",
          "title": "The structure of the nervous system of the nematode Caenorhabditis elegans",
          "authors": "J. G. White, E. Southgate, J. N. Thomson, S. Brenner",
          "year": 1986,
          "venue": "Philosophical Transactions of the Royal Society B 314(1165):1-340",
          "url": "https://doi.org/10.1098/rstb.1986.0056",
          "doi": "10.1098/rstb.1986.0056"
        },
        {
          "id": "dorkenwald2024",
          "title": "Neuronal wiring diagram of an adult brain",
          "authors": "Sven Dorkenwald, Arie Matsliah, Amy R. Sterling, et al. (FlyWire Consortium)",
          "year": 2024,
          "venue": "Nature 634:124-138",
          "url": "https://doi.org/10.1038/s41586-024-07558-y",
          "doi": "10.1038/s41586-024-07558-y"
        },
        {
          "id": "herculano2006",
          "title": "Cellular scaling rules for rodent brains",
          "authors": "Suzana Herculano-Houzel, Bruno Mota, Roberto Lent",
          "year": 2006,
          "venue": "PNAS 103(32):12138-12143",
          "url": "https://doi.org/10.1073/pnas.0604911103",
          "doi": "10.1073/pnas.0604911103",
          "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e circa 71 milioni coerente; la deviazione ±10,41 milioni va ricontrollata."
        },
        {
          "id": "azevedo2009",
          "title": "Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain",
          "authors": "Frederico A. C. Azevedo, Ludmila R. B. Carvalho, Lea T. Grinberg, et al. (ultimo autore Suzana Herculano-Houzel)",
          "year": 2009,
          "venue": "Journal of Comparative Neurology 513(5):532-541",
          "url": "https://doi.org/10.1002/cne.21974",
          "doi": "10.1002/cne.21974"
        },
        {
          "id": "herculano2014",
          "title": "The elephant brain in numbers",
          "authors": "Suzana Herculano-Houzel, Kamilla Avelino-de-Souza, Kleber Neves, et al.",
          "year": 2014,
          "venue": "Frontiers in Neuroanatomy 8:46",
          "url": "https://doi.org/10.3389/fnana.2014.00046",
          "doi": "10.3389/fnana.2014.00046"
        },
        {
          "id": "corso",
          "title": "Configurazione e traccia del training del GPT del corso",
          "authors": "Repository del corso (GPT didattico, dal notebook di Andrej Karpathy)",
          "year": 2026,
          "venue": "webapp/data/model.json e webapp/data/training/trace.json",
          "url": "webapp/data/training/trace.json",
          "note": "Dati esportati dal Python del corso: 209.729 parametri, 5000 passi, loss di validazione 4,40 → 1,81, durata 286,7 s, dropout 0,0, AdamW con lr 1e-3."
        }
      ],
      "ai_section": "Confronto tra modelli"
    },
    {
      "id": "lab-decay",
      "ai": "Weight decay di AdamW: a ogni passo i pesi vengono ristretti un poco",
      "brain": "Omeostasi sinaptica: le sinapsi vengono ridimensionate per restare in equilibrio",
      "regions": [
        "pfc_dl",
        "hippocampus"
      ],
      "text": "Turrigiano (2008) descrive lo scaling sinaptico: se un neurone è troppo attivo abbassa in proporzione la forza delle sue sinapsi eccitatorie, se è troppo poco attivo la alza. Secondo l'ipotesi dell'omeostasi sinaptica (Tononi e Cirelli 2014) il sonno servirebbe anche a ridimensionare le sinapsi rafforzate durante la veglia. Il weight decay di AdamW (0,01 è il valore predefinito di PyTorch, che il corso non cambia) fa qualcosa di simile nello spirito: a ogni passo restringe un poco tutti i pesi, così nessuno cresce senza controllo.",
      "similar": "Un meccanismo generale tiene sotto controllo la crescita delle connessioni mentre l'apprendimento le modifica.",
      "different": "Anche il weight decay moltiplica i pesi per uno stesso fattore, ma sempre verso il basso e senza guardare l'attività; lo scaling sinaptico invece è guidato dall'attività del neurone, può alzare o abbassare le sue sinapsi e riguarda le sinapsi di quel neurone, non tutta la rete. L'idea che il sonno «ridimensioni» le sinapsi è un'ipotesi ancora discussa.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "turrigiano2008",
          "title": "The self-tuning neuron: synaptic scaling of excitatory synapses",
          "authors": "Gina G. Turrigiano",
          "year": 2008,
          "venue": "Cell 135(3):422-435",
          "url": "https://doi.org/10.1016/j.cell.2008.10.008",
          "doi": "10.1016/j.cell.2008.10.008"
        },
        {
          "id": "tononi2014",
          "title": "Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration",
          "authors": "Giulio Tononi, Chiara Cirelli",
          "year": 2014,
          "venue": "Neuron 81(1):12-34",
          "url": "https://doi.org/10.1016/j.neuron.2013.12.025",
          "doi": "10.1016/j.neuron.2013.12.025"
        },
        {
          "id": "loshchilov2019",
          "title": "Decoupled weight decay regularization",
          "authors": "Ilya Loshchilov, Frank Hutter",
          "year": 2019,
          "venue": "ICLR 2019 (arXiv:1711.05101)",
          "url": "https://arxiv.org/abs/1711.05101"
        },
        {
          "id": "pytorch_adamw",
          "title": "torch.optim.AdamW (documentazione ufficiale)",
          "authors": "PyTorch",
          "year": 2024,
          "venue": "Documentazione PyTorch",
          "url": "https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html",
          "note": "Valore predefinito weight_decay = 0,01."
        }
      ],
      "ai_section": "Training (AdamW)"
    },
    {
      "id": "lab-potatura",
      "ai": "Pesi vicini a zero e potatura (pruning) delle reti",
      "brain": "Potatura sinaptica durante lo sviluppo",
      "regions": [
        "pfc_dl"
      ],
      "text": "Huttenlocher (1979) contò le sinapsi nel giro frontale medio umano dalla nascita alla vecchiaia: la densità sale nell'infanzia, tocca il massimo verso 1–2 anni (circa il 50% sopra la media adulta), scende fino ai 16 anni e poi resta stabile nell'età adulta. Nelle reti artificiali si possono eliminare molti pesi piccoli con poca perdita: una delle prime tecniche si chiamava proprio «Optimal Brain Damage» (LeCun, Denker e Solla 1990). Nel laboratorio puoi vedere quanti pesi del GPT del corso sono vicini a zero.",
      "similar": "Si parte da molte connessioni e si tengono quelle utili.",
      "different": "La potatura nel cervello è guidata dall'attività e dallo sviluppo; quella delle reti è una scelta di chi progetta, fatta con un criterio come la grandezza del peso o, in Optimal Brain Damage, una stima di quanto cresce la loss togliendo quel peso (Frankle e Carbin 2019 mostrano che dentro una rete grande esistono sottoreti piccole addestrabili). Il GPT del corso non è stato potato.",
      "strength": "media",
      "sources": [
        {
          "id": "huttenlocher1979",
          "title": "Synaptic density in human frontal cortex – developmental changes and effects of aging",
          "authors": "Peter R. Huttenlocher",
          "year": 1979,
          "venue": "Brain Research 163(2):195-205",
          "url": "https://doi.org/10.1016/0006-8993(79)90349-4",
          "doi": "10.1016/0006-8993(79)90349-4"
        },
        {
          "id": "lecun1990",
          "title": "Optimal brain damage",
          "authors": "Yann LeCun, John S. Denker, Sara A. Solla",
          "year": 1990,
          "venue": "Advances in Neural Information Processing Systems 2 (NIPS 1989)",
          "url": "https://papers.nips.cc/paper_files/paper/1989",
          "note": "L'url porta all'indice del volume NIPS 2."
        },
        {
          "id": "frankle2019",
          "title": "The lottery ticket hypothesis: finding sparse, trainable neural networks",
          "authors": "Jonathan Frankle, Michael Carbin",
          "year": 2019,
          "venue": "ICLR 2019 (arXiv:1803.03635)",
          "url": "https://arxiv.org/abs/1803.03635"
        }
      ],
      "ai_section": "I pesi del modello"
    }
  ],
  "facts": [
    {
      "claim": "Neuroni nel cervello umano adulto (maschi)",
      "value": "86,1 ± 8,1 miliardi",
      "source": {
        "id": "azevedo2009",
        "title": "Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain",
        "authors": "Frederico A. C. Azevedo, Ludmila R. B. Carvalho, Lea T. Grinberg, et al. (ultimo autore Suzana Herculano-Houzel)",
        "year": 2009,
        "venue": "Journal of Comparative Neurology 513(5):532-541",
        "url": "https://doi.org/10.1002/cne.21974",
        "doi": "10.1002/cne.21974"
      }
    },
    {
      "claim": "Cellule non neuronali (glia e altre)",
      "value": "84,6 ± 9,8 miliardi",
      "source": {
        "id": "azevedo2009",
        "title": "Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain",
        "authors": "Frederico A. C. Azevedo, Ludmila R. B. Carvalho, Lea T. Grinberg, et al. (ultimo autore Suzana Herculano-Houzel)",
        "year": 2009,
        "venue": "Journal of Comparative Neurology 513(5):532-541",
        "url": "https://doi.org/10.1002/cne.21974",
        "doi": "10.1002/cne.21974"
      }
    },
    {
      "claim": "Neuroni nella corteccia cerebrale",
      "value": "circa 16,3 miliardi (circa il 19% del totale)",
      "source": {
        "id": "azevedo2009",
        "title": "Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain",
        "authors": "Frederico A. C. Azevedo, Ludmila R. B. Carvalho, Lea T. Grinberg, et al. (ultimo autore Suzana Herculano-Houzel)",
        "year": 2009,
        "venue": "Journal of Comparative Neurology 513(5):532-541",
        "url": "https://doi.org/10.1002/cne.21974",
        "doi": "10.1002/cne.21974"
      },
      "note": "Valore ripreso anche in Herculano-Houzel et al. 2014."
    },
    {
      "claim": "Neuroni nel cervelletto",
      "value": "69,03 ± 6,65 miliardi (circa l'80% del totale)",
      "source": {
        "id": "azevedo2009",
        "title": "Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain",
        "authors": "Frederico A. C. Azevedo, Ludmila R. B. Carvalho, Lea T. Grinberg, et al. (ultimo autore Suzana Herculano-Houzel)",
        "year": 2009,
        "venue": "Journal of Comparative Neurology 513(5):532-541",
        "url": "https://doi.org/10.1002/cne.21974",
        "doi": "10.1002/cne.21974"
      }
    },
    {
      "claim": "Neuroni nel resto del cervello (tronco, talamo, gangli della base…)",
      "value": "circa 0,69 miliardi",
      "source": {
        "id": "azevedo2009",
        "title": "Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain",
        "authors": "Frederico A. C. Azevedo, Ludmila R. B. Carvalho, Lea T. Grinberg, et al. (ultimo autore Suzana Herculano-Houzel)",
        "year": 2009,
        "venue": "Journal of Comparative Neurology 513(5):532-541",
        "url": "https://doi.org/10.1002/cne.21974",
        "doi": "10.1002/cne.21974"
      },
      "note": "Coerente con 86,06 − 16,34 − 69,03."
    },
    {
      "claim": "Cervelletto umano, stima stereologica (5 uomini anziani)",
      "value": "101 miliardi di cellule granulari e 30,5 milioni di cellule di Purkinje",
      "source": {
        "id": "andersen1992",
        "title": "A quantitative study of the human cerebellum with unbiased stereological techniques",
        "authors": "Birgitte Bo Andersen, Lise Korbo, Bente Pakkenberg",
        "year": 1992,
        "venue": "Journal of Comparative Neurology 326(4):549-560",
        "url": "https://doi.org/10.1002/cne.903260405",
        "doi": "10.1002/cne.903260405",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifre 101 miliardi / 30,5 milioni coerenti; da ricontrollare la descrizione del campione («5 uomini anziani»)."
      },
      "note": "Metodi diversi danno numeri diversi: per questo non coincide con i 69 miliardi di Azevedo. Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifre 101 miliardi / 30,5 milioni coerenti; da ricontrollare la descrizione del campione («5 uomini anziani»)."
    },
    {
      "claim": "Sinapsi nella neocorteccia (5 giovani maschi)",
      "value": "164 × 10^12 (circa 1,6 × 10^14)",
      "source": {
        "id": "tang2001",
        "title": "Total regional and global number of synapses in the human brain neocortex",
        "authors": "Yong Tang, Jens R. Nyengaard, Didima M. G. De Groot, Hans Jørgen G. Gundersen",
        "year": 2001,
        "venue": "Synapse 41(3):258-273",
        "url": "https://doi.org/10.1002/syn.1083",
        "doi": "10.1002/syn.1083",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifra 164 × 10^12 coerente; da ricontrollare la descrizione del campione («5 giovani maschi»)."
      },
      "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifra 164 × 10^12 coerente; da ricontrollare la descrizione del campione («5 giovani maschi»)."
    },
    {
      "claim": "Sinapsi nella neocorteccia (rassegna)",
      "value": "circa 0,15 × 10^15 (1,5 × 10^14)",
      "source": {
        "id": "pakkenberg2003",
        "title": "Aging and the human neocortex",
        "authors": "Bente Pakkenberg, Dorte Pelvig, Lisbeth Marner, et al.",
        "year": 2003,
        "venue": "Experimental Gerontology 38(1-2):95-99",
        "url": "https://doi.org/10.1016/S0531-5565(02)00151-1",
        "doi": "10.1016/S0531-5565(02)00151-1"
      }
    },
    {
      "claim": "Sinapsi in tutto il cervello",
      "value": "nessun conteggio diretto: la cifra 10^14–10^15 che si legge spesso è una stima d'ordine di grandezza; il dato misurato riguarda solo la neocorteccia",
      "source": {
        "id": "tang2001",
        "title": "Total regional and global number of synapses in the human brain neocortex",
        "authors": "Yong Tang, Jens R. Nyengaard, Didima M. G. De Groot, Hans Jørgen G. Gundersen",
        "year": 2001,
        "venue": "Synapse 41(3):258-273",
        "url": "https://doi.org/10.1002/syn.1083",
        "doi": "10.1002/syn.1083",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifra 164 × 10^12 coerente; da ricontrollare la descrizione del campione («5 giovani maschi»)."
      },
      "note": "Da dire esplicitamente nel pannello. Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifra 164 × 10^12 coerente; da ricontrollare la descrizione del campione («5 giovani maschi»)."
    },
    {
      "claim": "Sinapsi per neurone corticale (ordine di grandezza)",
      "value": "circa 10^4 (calcolo nostro: 1,6 × 10^14 sinapsi / 1,6 × 10^10 neuroni, da due studi diversi)",
      "source": {
        "id": "tang2001",
        "title": "Total regional and global number of synapses in the human brain neocortex",
        "authors": "Yong Tang, Jens R. Nyengaard, Didima M. G. De Groot, Hans Jørgen G. Gundersen",
        "year": 2001,
        "venue": "Synapse 41(3):258-273",
        "url": "https://doi.org/10.1002/syn.1083",
        "doi": "10.1002/syn.1083",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifra 164 × 10^12 coerente; da ricontrollare la descrizione del campione («5 giovani maschi»)."
      },
      "note": "Calcolo indicativo che combina Tang et al. 2001 e Azevedo et al. 2009. Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e cifra 164 × 10^12 coerente; da ricontrollare la descrizione del campione («5 giovani maschi»)."
    },
    {
      "claim": "Informazione in una sinapsi (ippocampo di ratto)",
      "value": "almeno 26 livelli distinguibili ≈ 4,7 bit",
      "source": {
        "id": "bartol2015",
        "title": "Nanoconnectomic upper bound on the variability of synaptic plasticity",
        "authors": "Thomas M. Bartol Jr, Cailey Bromer, Justin Kinney, Michael A. Chirillo, Jennifer N. Bourne, Kristen M. Harris, Terrence J. Sejnowski",
        "year": 2015,
        "venue": "eLife 4:e10778",
        "url": "https://doi.org/10.7554/eLife.10778",
        "doi": "10.7554/eLife.10778"
      }
    },
    {
      "claim": "Sinapsi da fibre parallele su una cellula di Purkinje",
      "value": "circa 175.000 (ratto)",
      "source": {
        "id": "napper1988",
        "title": "Number of parallel fiber synapses on an individual Purkinje cell in the cerebellum of the rat",
        "authors": "Ruth M. A. Napper, Robert J. Harvey",
        "year": 1988,
        "venue": "Journal of Comparative Neurology 274(2):168-177",
        "url": "https://doi.org/10.1002/cne.902740204",
        "doi": "10.1002/cne.902740204"
      }
    },
    {
      "claim": "Frequenze delle cellule di Purkinje",
      "value": "spike semplici 50–150/s; spike complessi circa 0,5–2/s",
      "source": {
        "id": "streng2018",
        "title": "Complex spike wars: a new hope",
        "authors": "Martha L. Streng, Laurentiu S. Popa, Timothy J. Ebner",
        "year": 2018,
        "venue": "The Cerebellum 17(6):735-746",
        "url": "https://doi.org/10.1007/s12311-018-0960-3",
        "doi": "10.1007/s12311-018-0960-3",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente; divergenza 5–10 cellule e frequenze da ricontrollare sul testo."
      },
      "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente; divergenza 5–10 cellule e frequenze da ricontrollare sul testo."
    },
    {
      "claim": "Densità sinaptica nella corteccia frontale umana",
      "value": "massimo a 1–2 anni, circa il 50% sopra la media adulta; nell'adulto 11,05 × 10^8 sinapsi per mm³",
      "source": {
        "id": "huttenlocher1979",
        "title": "Synaptic density in human frontal cortex – developmental changes and effects of aging",
        "authors": "Peter R. Huttenlocher",
        "year": 1979,
        "venue": "Brain Research 163(2):195-205",
        "url": "https://doi.org/10.1016/0006-8993(79)90349-4",
        "doi": "10.1016/0006-8993(79)90349-4"
      }
    },
    {
      "claim": "C. elegans (primo connettoma completo)",
      "value": "302 neuroni, circa 5.000 sinapsi chimiche, 2.000 giunzioni neuromuscolari, 600 giunzioni elettriche",
      "source": {
        "id": "white1986",
        "title": "The structure of the nervous system of the nematode Caenorhabditis elegans",
        "authors": "J. G. White, E. Southgate, J. N. Thomson, S. Brenner",
        "year": 1986,
        "venue": "Philosophical Transactions of the Royal Society B 314(1165):1-340",
        "url": "https://doi.org/10.1098/rstb.1986.0056",
        "doi": "10.1098/rstb.1986.0056"
      }
    },
    {
      "claim": "Drosophila adulta (FlyWire)",
      "value": "139.255 neuroni, circa 5 × 10^7 sinapsi chimiche",
      "source": {
        "id": "dorkenwald2024",
        "title": "Neuronal wiring diagram of an adult brain",
        "authors": "Sven Dorkenwald, Arie Matsliah, Amy R. Sterling, et al. (FlyWire Consortium)",
        "year": 2024,
        "venue": "Nature 634:124-138",
        "url": "https://doi.org/10.1038/s41586-024-07558-y",
        "doi": "10.1038/s41586-024-07558-y"
      }
    },
    {
      "claim": "1 mm³ di corteccia temporale umana (H01)",
      "value": "circa 57.000 cellule, 150 milioni di sinapsi, 230 mm di vasi sanguigni, 1,4 petabyte di dati",
      "source": {
        "id": "shapsoncoe2024",
        "title": "A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution",
        "authors": "Alexander Shapson-Coe, Michał Januszewski, Daniel R. Berger, et al. (ultimi autori Viren Jain, Jeff W. Lichtman)",
        "year": 2024,
        "venue": "Science 384(6696):eadk4858",
        "url": "https://doi.org/10.1126/science.adk4858",
        "doi": "10.1126/science.adk4858"
      }
    },
    {
      "claim": "1 mm³ di corteccia visiva di topo (MICrONS)",
      "value": "oltre 200.000 cellule e circa 524 milioni di sinapsi",
      "source": {
        "id": "microns2025",
        "title": "Functional connectomics spanning multiple areas of mouse visual cortex",
        "authors": "The MICrONS Consortium",
        "year": 2025,
        "venue": "Nature 640:435-447",
        "url": "https://doi.org/10.1038/s41586-025-08790-w",
        "doi": "10.1038/s41586-025-08790-w",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente; i 524 milioni di sinapsi vengono da comunicati e sito del progetto, non dall'articolo."
      },
      "note": "Valori diffusi con la pubblicazione (comunicati e sito del progetto). Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente; i 524 milioni di sinapsi vengono da comunicati e sito del progetto, non dall'articolo."
    },
    {
      "claim": "Topo",
      "value": "circa 71 milioni di neuroni (70,89 ± 10,41 milioni)",
      "source": {
        "id": "herculano2006",
        "title": "Cellular scaling rules for rodent brains",
        "authors": "Suzana Herculano-Houzel, Bruno Mota, Roberto Lent",
        "year": 2006,
        "venue": "PNAS 103(32):12138-12143",
        "url": "https://doi.org/10.1073/pnas.0604911103",
        "doi": "10.1073/pnas.0604911103",
        "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e circa 71 milioni coerente; la deviazione ±10,41 milioni va ricontrollata."
      },
      "note": "Da ricontrollare (verifica della ricerca dell'8 novembre 2026): Riferimento coerente e circa 71 milioni coerente; la deviazione ±10,41 milioni va ricontrollata."
    },
    {
      "claim": "Elefante africano",
      "value": "circa 257 miliardi di neuroni, il 97,5% nel cervelletto, 5,6 miliardi nella corteccia",
      "source": {
        "id": "herculano2014",
        "title": "The elephant brain in numbers",
        "authors": "Suzana Herculano-Houzel, Kamilla Avelino-de-Souza, Kleber Neves, et al.",
        "year": 2014,
        "venue": "Frontiers in Neuroanatomy 8:46",
        "url": "https://doi.org/10.3389/fnana.2014.00046",
        "doi": "10.3389/fnana.2014.00046"
      },
      "note": "Un solo esemplare."
    },
    {
      "claim": "GPT del corso",
      "value": "209.729 parametri; 4 blocchi, 4 head, vettori da 64, contesto di 32 caratteri; 5000 passi in 286,7 s; loss di validazione da 4,40 a 1,81",
      "source": {
        "id": "corso",
        "title": "Configurazione e traccia del training del GPT del corso",
        "authors": "Repository del corso (GPT didattico, dal notebook di Andrej Karpathy)",
        "year": 2026,
        "venue": "webapp/data/model.json e webapp/data/training/trace.json",
        "url": "webapp/data/training/trace.json",
        "note": "Dati esportati dal Python del corso: 209.729 parametri, 5000 passi, loss di validazione 4,40 → 1,81, durata 286,7 s, dropout 0,0, AdamW con lr 1e-3."
      }
    },
    {
      "claim": "Caratteri visti dal GPT del corso durante il training",
      "value": "5000 passi × 16 sequenze × 32 caratteri = 2.560.000 caratteri (calcolo dalla configurazione)",
      "source": {
        "id": "corso",
        "title": "Configurazione e traccia del training del GPT del corso",
        "authors": "Repository del corso (GPT didattico, dal notebook di Andrej Karpathy)",
        "year": 2026,
        "venue": "webapp/data/model.json e webapp/data/training/trace.json",
        "url": "webapp/data/training/trace.json",
        "note": "Dati esportati dal Python del corso: 209.729 parametri, 5000 passi, loss di validazione 4,40 → 1,81, durata 286,7 s, dropout 0,0, AdamW con lr 1e-3."
      }
    }
  ],
  "sources": []
};
