/**
 * Rappresentazione "Cervello" del tab Audio e musica: la specifica di SOLI DATI per il pannello "Cervello"
 * della sidebar e per il modello 3D facoltativo del nucleo (js/core/brain/, formato in ARCHITECTURE.md,
 * sezione "Cervello"). Niente three.js, niente DOM: tutti i testi sono mostrati come testo.
 *
 * FILE GENERATO: non modificarlo a mano. Da dove vengono i dati: la ricerca verificata dell'8 novembre
 * 2026 (scratchpad/research/brain_data.json, generata il 2026-11-08), tabs["audio"] copiato senza
 * tradurre né riscrivere (il nucleo accetta i nomi della ricerca: neuron_type, count_shown, real_count,
 * real_count_text, analogy_map, kind "eccitatoria" | "inibitoria" | "modulatoria", strength
 * "debole/metafora", ai_section, fonti con doi):
 *   - introduzione, 9 analogie (ognuna con "In comune", "Dove non regge", forza e fonti);
 *   - il circuito "Via uditiva: dalla coclea alla corteccia" (10 gruppi, 12 sinapsi, con
 *     l'avvertenza della ricerca), senza il campo descrittivo "layout";
 *   - regioni, percorsi e marcatori (id dell'atlante del nucleo);
 *   - facts: i 2 numeri globali che citano le fonti dei conteggi del circuito e 7 dati verificati dei
 *     suoi tipi di neurone (neuron_types, ognuno con la sua fonte; l'etichetta breve è nostra, il testo è
 *     quello della ricerca);
 *   - sources: le due fonti che sostengono l'introduzione (le altre sono nelle analogie, nel circuito e nei
 *     fatti: il pannello le unisce senza doppioni).
 * Niente analogie nuove e nessun numero nuovo: solo quelli della ricerca.
 *
 * Per rigenerarlo (dalla radice del repository; lo script sta nello scratchpad del round E, fuori dal
 * repository):
 *   node <scratchpad>/reps-audio/gen_brain.mjs <scratchpad>/research/brain_data.json webapp/js/tabs/audio/brain.js
 * tests/audio-reps.test.mjs controlla che passi normalizeBrainSpec(…, { strict: true }) senza avvisi, che
 * regioni, percorsi e marcatori siano nell'atlante e che ogni fonte citata abbia un link.
 */
export const brainSpec = {
  "title": "L'audio e la via uditiva del cervello",
  "intro": "Prima di diventare token, il suono passa per trasformazioni che imitano l'orecchio in modo dichiarato: la scala Mel, per esempio, nasce da esperimenti di percezione. La coclea scompone il suono in frequenze come uno spettrogramma, e le reti per il parlato sviluppano rappresentazioni che somigliano a quelle della corteccia uditiva. Ma l'orecchio è un organo meccanico attivo e non lineare, molto diverso da una FFT.",
  "regions": [
    "cochlea",
    "a1",
    "brainstem",
    "stg",
    "thalamus",
    "wernicke",
    "broca",
    "striatum",
    "vta_snc",
    "amygdala",
    "hippocampus",
    "motor",
    "somato",
    "premotor"
  ],
  "markers": [
    "cochlear_nucleus",
    "inferior_colliculus",
    "mgn",
    "superior_olive"
  ],
  "pathways": [
    "arcuate",
    "auditory_radiation",
    "mesolimbic"
  ],
  "circuit": {
    "title": "Via uditiva: dalla coclea alla corteccia",
    "text": "Le cellule ciliate della coclea trasformano la vibrazione in segnali per il nervo acustico; il segnale sale nel tronco encefalico (nucleo cocleare, oliva superiore, collicolo inferiore), passa dal genicolato mediale del talamo e arriva, lungo la radiazione uditiva, alla corteccia uditiva primaria e poi al giro temporale superiore. Ci sono anche vie discendenti: dall'oliva superiore partono fibre che regolano le cellule ciliate esterne.",
    "groups": [
      {
        "id": "ohc",
        "label": "Coclea: cellule ciliate esterne (amplificatore)",
        "region": "cochlea",
        "neuron_type": "cellula_ciliata",
        "count_shown": 9,
        "real_count": 11000,
        "real_count_text": "circa 11.000 per coclea",
        "real_count_source": "ashmore2008"
      },
      {
        "id": "ihc",
        "label": "Coclea: cellule ciliate interne (sensori)",
        "region": "cochlea",
        "neuron_type": "cellula_ciliata",
        "count_shown": 6,
        "real_count": 3500,
        "real_count_text": "circa 3.500 per coclea",
        "real_count_source": "ashmore2008"
      },
      {
        "id": "sgn",
        "label": "Nervo acustico: neuroni del ganglio spirale",
        "region": "cochlea",
        "neuron_type": "ganglio_spirale",
        "count_shown": 8,
        "real_count": 33679,
        "real_count_text": "in media circa 33.700 nel primo decennio di vita, circa 22.400 nel decimo",
        "real_count_source": "makary2011"
      },
      {
        "id": "cn",
        "label": "Nucleo cocleare",
        "region": "brainstem",
        "neuron_type": "neurone_di_proiezione",
        "count_shown": 6,
        "marker": "cochlear_nucleus"
      },
      {
        "id": "so",
        "label": "Oliva superiore (confronto tra le due orecchie)",
        "region": "brainstem",
        "neuron_type": "neurone_di_proiezione",
        "count_shown": 6,
        "marker": "superior_olive"
      },
      {
        "id": "ic",
        "label": "Collicolo inferiore",
        "region": "brainstem",
        "neuron_type": "neurone_di_proiezione",
        "count_shown": 6,
        "marker": "inferior_colliculus"
      },
      {
        "id": "mgn",
        "label": "Talamo: genicolato mediale",
        "region": "thalamus",
        "neuron_type": "rele_talamico",
        "count_shown": 6,
        "marker": "mgn"
      },
      {
        "id": "a1",
        "label": "Corteccia uditiva primaria (giro di Heschl)",
        "region": "a1",
        "neuron_type": "piramidale",
        "count_shown": 10
      },
      {
        "id": "stg",
        "label": "Giro temporale superiore: fonemi e sillabe",
        "region": "stg",
        "neuron_type": "piramidale",
        "count_shown": 10
      },
      {
        "id": "wern",
        "label": "Area di Wernicke: parole",
        "region": "wernicke",
        "neuron_type": "piramidale",
        "count_shown": 8
      }
    ],
    "edges": [
      {
        "from": "ohc",
        "to": "ihc",
        "kind": "modulatoria",
        "label": "amplificazione meccanica",
        "note": "Non è una sinapsi: le cellule esterne amplificano il moto della membrana basilare (Ashmore 2008)."
      },
      {
        "from": "ihc",
        "to": "sgn",
        "kind": "eccitatoria",
        "label": "sinapsi sulle fibre del nervo acustico"
      },
      {
        "from": "sgn",
        "to": "cn",
        "kind": "eccitatoria"
      },
      {
        "from": "cn",
        "to": "so",
        "kind": "eccitatoria"
      },
      {
        "from": "cn",
        "to": "ic",
        "kind": "eccitatoria"
      },
      {
        "from": "so",
        "to": "ic",
        "kind": "eccitatoria",
        "note": "Semplificato: dall'oliva partono anche proiezioni inibitorie (Grothe et al. 2010)."
      },
      {
        "from": "ic",
        "to": "mgn",
        "kind": "eccitatoria"
      },
      {
        "from": "mgn",
        "to": "a1",
        "kind": "eccitatoria",
        "label": "radiazione uditiva",
        "pathway": "auditory_radiation"
      },
      {
        "from": "a1",
        "to": "stg",
        "kind": "eccitatoria"
      },
      {
        "from": "stg",
        "to": "wern",
        "kind": "eccitatoria"
      },
      {
        "from": "a1",
        "to": "mgn",
        "kind": "modulatoria",
        "label": "ritorno corticotalamico"
      },
      {
        "from": "so",
        "to": "ohc",
        "kind": "inibitoria",
        "label": "efferenti olivococleari mediali (acetilcolina)",
        "note": "Il cervello regola il guadagno della coclea stessa: l'acetilcolina delle fibre olivococleari mediali riduce l'amplificazione delle cellule ciliate esterne (Guinan 2006)."
      }
    ],
    "analogy_map": [
      {
        "group": "ohc",
        "ai_component": "Compressione della dinamica (il logaritmo del log-Mel)"
      },
      {
        "group": "ihc",
        "ai_component": "Campionamento + banco di filtri: ogni posizione della coclea ≈ una banda di frequenza"
      },
      {
        "group": "sgn",
        "ai_component": "Encoder del codec: eventi discreti nel tempo, come i token"
      },
      {
        "group": "cn",
        "ai_component": "Nessun equivalente diretto (elaborazione temporale precoce)"
      },
      {
        "group": "so",
        "ai_component": "Nessun equivalente: Whisper lavora su audio mono, il cervello confronta le due orecchie"
      },
      {
        "group": "ic",
        "ai_component": "Prime convoluzioni dell'encoder di Whisper (corrispondenza solo ipotetica)"
      },
      {
        "group": "mgn",
        "ai_component": "Ingresso al transformer dell'encoder"
      },
      {
        "group": "a1",
        "ai_component": "Layer iniziali dell'encoder di Whisper"
      },
      {
        "group": "stg",
        "ai_component": "Layer profondi dell'encoder: fonemi (Li et al. 2023)"
      },
      {
        "group": "wern",
        "ai_component": "Decoder di Whisper: le parole, token per token"
      }
    ],
    "caveat": "Le corrispondenze con codec e Whisper riguardano la gerarchia complessiva, dimostrata per la corteccia (Li et al. 2023), non i singoli nuclei del tronco, che non hanno un equivalente diretto. Nel tronco encefalico mancano numeri verificati e sono indicati solo come marcatori.",
    "sources": [
      {
        "id": "ashmore2008",
        "title": "Cochlear outer hair cell motility",
        "authors": "Jonathan Ashmore",
        "year": 2008,
        "venue": "Physiological Reviews 88(1):173-210",
        "url": "https://doi.org/10.1152/physrev.00044.2006",
        "doi": "10.1152/physrev.00044.2006",
        "note": "Circa 3.500 cellule ciliate interne e circa 11.000 esterne per coclea umana (valori citati da BioNumbers, voce 117129, da questa rassegna)."
      },
      {
        "id": "makary2011",
        "title": "Age-related primary cochlear neuronal degeneration in human temporal bones",
        "authors": "Chadi A. Makary, Jennifer Shin, Sharon G. Kujawa, M. Charles Liberman, Saumil N. Merchant",
        "year": 2011,
        "venue": "Journal of the Association for Research in Otolaryngology 12(6):711-717",
        "url": "https://doi.org/10.1007/s10162-011-0283-2",
        "doi": "10.1007/s10162-011-0283-2"
      },
      {
        "id": "grothe2010",
        "title": "Mechanisms of sound localization in mammals",
        "authors": "Benedikt Grothe, Michael Pecka, David McAlpine",
        "year": 2010,
        "venue": "Physiological Reviews 90(3):983-1012",
        "url": "https://doi.org/10.1152/physrev.00026.2009",
        "doi": "10.1152/physrev.00026.2009"
      },
      {
        "id": "guinan2006",
        "title": "Olivocochlear efferents: anatomy, physiology, function, and the measurement of efferent effects in humans",
        "authors": "John J. Guinan Jr",
        "year": 2006,
        "venue": "Ear and Hearing 27(6):589-607",
        "url": "https://doi.org/10.1097/01.aud.0000240507.83072.e7",
        "doi": "10.1097/01.aud.0000240507.83072.e7"
      },
      {
        "id": "li2023",
        "title": "Dissecting neural computations in the human auditory pathway using deep neural networks for speech",
        "authors": "Yuanning Li, Gopala K. Anumanchipalli, Abdelrahman Mohamed, et al. (ultimo autore Edward F. Chang)",
        "year": 2023,
        "venue": "Nature Neuroscience 26(12):2213-2225",
        "url": "https://doi.org/10.1038/s41593-023-01468-4",
        "doi": "10.1038/s41593-023-01468-4"
      },
      {
        "id": "formisano2003",
        "title": "Mirror-symmetric tonotopic maps in human primary auditory cortex",
        "authors": "Elia Formisano, Dae-Shik Kim, Francesco Di Salle, Pierre-Francois van de Moortele, Kamil Ugurbil, Rainer Goebel",
        "year": 2003,
        "venue": "Neuron 40(4):859-869",
        "url": "https://doi.org/10.1016/S0896-6273(03)00669-X",
        "doi": "10.1016/S0896-6273(03)00669-X"
      }
    ]
  },
  "analogies": [
    {
      "id": "aud-campionamento",
      "ai": "Campionamento: la forma d'onda diventa numeri (in Whisper 30 secondi a 16 kHz)",
      "brain": "Cellule ciliate e nervo acustico: dalla vibrazione agli impulsi",
      "regions": [
        "cochlea"
      ],
      "text": "Nella coclea circa 3.500 cellule ciliate interne trasformano le vibrazioni in segnali elettrici e li passano alle fibre del nervo acustico (Ashmore 2008); i neuroni del ganglio spirale sono in media circa 33.700 nel primo decennio di vita e circa 22.400 nel decimo (Makary et al. 2011). Le fibre scaricano impulsi agganciati alla fase dell'onda fino a qualche kHz (Johnson 1980, nel gatto). Un microfono con un convertitore fa lo stesso lavoro in un altro modo: per Whisper 16.000 campioni al secondo, tutti a intervalli regolari.",
      "similar": "Un segnale continuo (la pressione dell'aria) diventa una sequenza di eventi discreti che il resto del sistema può elaborare.",
      "different": "Il campionamento è uniforme e lineare; l'orecchio no: ogni cellula ciliata «ascolta» una banda diversa, le fibre hanno soglie e frequenze spontanee diverse (Liberman 1978) e scaricano in modo irregolare, e le cellule ciliate esterne amplificano attivamente i suoni deboli.",
      "strength": "debole/metafora",
      "sources": [
        {
          "id": "ashmore2008",
          "title": "Cochlear outer hair cell motility",
          "authors": "Jonathan Ashmore",
          "year": 2008,
          "venue": "Physiological Reviews 88(1):173-210",
          "url": "https://doi.org/10.1152/physrev.00044.2006",
          "doi": "10.1152/physrev.00044.2006",
          "note": "Circa 3.500 cellule ciliate interne e circa 11.000 esterne per coclea umana (valori citati da BioNumbers, voce 117129, da questa rassegna)."
        },
        {
          "id": "makary2011",
          "title": "Age-related primary cochlear neuronal degeneration in human temporal bones",
          "authors": "Chadi A. Makary, Jennifer Shin, Sharon G. Kujawa, M. Charles Liberman, Saumil N. Merchant",
          "year": 2011,
          "venue": "Journal of the Association for Research in Otolaryngology 12(6):711-717",
          "url": "https://doi.org/10.1007/s10162-011-0283-2",
          "doi": "10.1007/s10162-011-0283-2"
        },
        {
          "id": "johnson1980",
          "title": "The relationship between spike rate and synchrony in responses of auditory-nerve fibers to single tones",
          "authors": "Don H. Johnson",
          "year": 1980,
          "venue": "Journal of the Acoustical Society of America 68(4):1115-1122",
          "url": "https://doi.org/10.1121/1.384982",
          "doi": "10.1121/1.384982"
        },
        {
          "id": "liberman1978",
          "title": "Auditory-nerve response from cats raised in a low-noise chamber",
          "authors": "M. Charles Liberman",
          "year": 1978,
          "venue": "Journal of the Acoustical Society of America 63(2):442-455",
          "url": "https://doi.org/10.1121/1.381736",
          "doi": "10.1121/1.381736"
        },
        {
          "id": "radford2022whisper",
          "title": "Robust speech recognition via large-scale weak supervision (Whisper)",
          "authors": "Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever",
          "year": 2022,
          "venue": "ICML 2023 (arXiv:2212.04356)",
          "url": "https://arxiv.org/abs/2212.04356"
        }
      ],
      "ai_section": "1 · Il suono è un'onda (Campionamento)"
    },
    {
      "id": "aud-stft",
      "ai": "Finestre, FFT e spettrogramma (STFT)",
      "brain": "La membrana basilare: un analizzatore di frequenze ordinato (tonotopia)",
      "regions": [
        "cochlea",
        "a1"
      ],
      "text": "Georg von Békésy (premio Nobel 1961) scoprì che il suono crea un'onda che viaggia lungo la membrana basilare e ha il massimo in un punto diverso per ogni frequenza: gli acuti vicino alla base, i gravi verso l'apice. Greenwood (1990) ha descritto con una formula quale frequenza corrisponde a ogni posizione, e mappe ordinate per frequenza si ritrovano anche nella corteccia uditiva primaria (Formisano et al. 2003). Lo spettrogramma fa una scomposizione simile con la FFT: per ogni finestra di tempo, quanta energia c'è a ogni frequenza.",
      "similar": "Scomporre il suono in bande di frequenza nel tempo e disporre le frequenze in ordine.",
      "different": "La coclea è un filtro meccanico attivo e non lineare: le cellule ciliate esterne amplificano i suoni deboli e comprimono quelli forti (Ruggero et al. 1997), e i suoi filtri non sono equispaziati. La FFT è lineare, usa bande tutte uguali e lavora finestra per finestra.",
      "strength": "media",
      "sources": [
        {
          "id": "bekesy1961",
          "title": "The Nobel Prize in Physiology or Medicine 1961 – Georg von Békésy",
          "authors": "The Nobel Foundation",
          "year": 1961,
          "venue": "NobelPrize.org",
          "url": "https://www.nobelprize.org/prizes/medicine/1961/summary/",
          "note": "Motivazione: le scoperte sul meccanismo fisico di stimolazione nella coclea."
        },
        {
          "id": "greenwood1990",
          "title": "A cochlear frequency-position function for several species – 29 years later",
          "authors": "Donald D. Greenwood",
          "year": 1990,
          "venue": "Journal of the Acoustical Society of America 87(6):2592-2605",
          "url": "https://doi.org/10.1121/1.399052",
          "doi": "10.1121/1.399052"
        },
        {
          "id": "formisano2003",
          "title": "Mirror-symmetric tonotopic maps in human primary auditory cortex",
          "authors": "Elia Formisano, Dae-Shik Kim, Francesco Di Salle, Pierre-Francois van de Moortele, Kamil Ugurbil, Rainer Goebel",
          "year": 2003,
          "venue": "Neuron 40(4):859-869",
          "url": "https://doi.org/10.1016/S0896-6273(03)00669-X",
          "doi": "10.1016/S0896-6273(03)00669-X"
        },
        {
          "id": "ruggero1997",
          "title": "Basilar-membrane responses to tones at the base of the chinchilla cochlea",
          "authors": "Mario A. Ruggero, Nola C. Rich, Alberto Recio, S. Shyamla Narayan, Luis Robles",
          "year": 1997,
          "venue": "Journal of the Acoustical Society of America 101(4):2151-2163",
          "url": "https://doi.org/10.1121/1.418265",
          "doi": "10.1121/1.418265"
        }
      ],
      "ai_section": "2 · Dal tempo alle frequenze (FFT e spettrogramma)"
    },
    {
      "id": "aud-mel",
      "ai": "Scala Mel e banco di filtri Mel (log-Mel a 80 bande in Whisper)",
      "brain": "Percezione dell'altezza (pitch) e bande critiche dell'udito",
      "regions": [
        "cochlea",
        "a1"
      ],
      "text": "La scala Mel non è un'invenzione degli informatici: Stevens, Volkmann e Newman (1937) chiesero ad ascoltatori di regolare un tono finché sembrasse «alto la metà» di un altro e costruirono una scala in cui distanze uguali sembrano uguali all'orecchio (un tono di 1000 Hz vale 1000 mel). Fletcher (1940) e Zwicker (1961) misurarono le «bande critiche»: l'orecchio raggruppa le frequenze vicine in circa 24 bande, strette ai gravi e larghe agli acuti. Il banco di filtri Mel usato da Whisper (80 bande) riprende proprio questa idea.",
      "similar": "Più risoluzione dove l'orecchio distingue meglio (le basse frequenze), meno dove distingue peggio.",
      "different": "La scala Mel è una media su ascoltatori in condizioni di laboratorio e ne esistono più formule; i filtri uditivi veri cambiano con l'intensità del suono (Glasberg e Moore 1990). I filtri triangolari Mel sono un'approssimazione comoda, non un modello della coclea.",
      "strength": "forte",
      "sources": [
        {
          "id": "stevens1937",
          "title": "A scale for the measurement of the psychological magnitude pitch",
          "authors": "S. S. Stevens, J. Volkmann, E. B. Newman",
          "year": 1937,
          "venue": "Journal of the Acoustical Society of America 8(3):185-190",
          "url": "https://doi.org/10.1121/1.1915893",
          "doi": "10.1121/1.1915893"
        },
        {
          "id": "fletcher1940",
          "title": "Auditory patterns",
          "authors": "Harvey Fletcher",
          "year": 1940,
          "venue": "Reviews of Modern Physics 12(1):47-65",
          "url": "https://doi.org/10.1103/RevModPhys.12.47",
          "doi": "10.1103/RevModPhys.12.47"
        },
        {
          "id": "zwicker1961",
          "title": "Subdivision of the audible frequency range into critical bands (Frequenzgruppen)",
          "authors": "E. Zwicker",
          "year": 1961,
          "venue": "Journal of the Acoustical Society of America 33(2):248",
          "url": "https://doi.org/10.1121/1.1908630",
          "doi": "10.1121/1.1908630"
        },
        {
          "id": "glasberg1990",
          "title": "Derivation of auditory filter shapes from notched-noise data",
          "authors": "Brian R. Glasberg, Brian C. J. Moore",
          "year": 1990,
          "venue": "Hearing Research 47(1-2):103-138",
          "url": "https://doi.org/10.1016/0378-5955(90)90170-T",
          "doi": "10.1016/0378-5955(90)90170-T"
        },
        {
          "id": "radford2022whisper",
          "title": "Robust speech recognition via large-scale weak supervision (Whisper)",
          "authors": "Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever",
          "year": 2022,
          "venue": "ICML 2023 (arXiv:2212.04356)",
          "url": "https://arxiv.org/abs/2212.04356"
        }
      ],
      "ai_section": "2 · Dal tempo alle frequenze (Scala Mel e log-Mel)"
    },
    {
      "id": "aud-log",
      "ai": "Il logaritmo: log-Mel",
      "brain": "Compressione della dinamica nella coclea",
      "regions": [
        "cochlea",
        "brainstem"
      ],
      "text": "L'orecchio gestisce un'enorme gamma di intensità, dal fruscio al concerto. Parte del merito è delle cellule ciliate esterne, che amplificano molto i suoni deboli e poco quelli forti: misurando la membrana basilare del cincillà, Ruggero et al. (1997) hanno trovato una crescita della risposta fortemente compressiva. Il log-Mel ottiene una compressione simile con il logaritmo: le differenze tra suoni forti pesano meno.",
      "similar": "Comprimere i valori grandi per rappresentare insieme suoni deboli e forti.",
      "different": "La compressione della coclea dipende dalla frequenza e dal livello ed è regolata dal cervello attraverso le fibre efferenti olivococleari (Guinan 2006); il logaritmo è una funzione fissa, uguale per tutte le bande.",
      "strength": "media",
      "sources": [
        {
          "id": "ruggero1997",
          "title": "Basilar-membrane responses to tones at the base of the chinchilla cochlea",
          "authors": "Mario A. Ruggero, Nola C. Rich, Alberto Recio, S. Shyamla Narayan, Luis Robles",
          "year": 1997,
          "venue": "Journal of the Acoustical Society of America 101(4):2151-2163",
          "url": "https://doi.org/10.1121/1.418265",
          "doi": "10.1121/1.418265"
        },
        {
          "id": "ashmore2008",
          "title": "Cochlear outer hair cell motility",
          "authors": "Jonathan Ashmore",
          "year": 2008,
          "venue": "Physiological Reviews 88(1):173-210",
          "url": "https://doi.org/10.1152/physrev.00044.2006",
          "doi": "10.1152/physrev.00044.2006",
          "note": "Circa 3.500 cellule ciliate interne e circa 11.000 esterne per coclea umana (valori citati da BioNumbers, voce 117129, da questa rassegna)."
        },
        {
          "id": "guinan2006",
          "title": "Olivocochlear efferents: anatomy, physiology, function, and the measurement of efferent effects in humans",
          "authors": "John J. Guinan Jr",
          "year": 2006,
          "venue": "Ear and Hearing 27(6):589-607",
          "url": "https://doi.org/10.1097/01.aud.0000240507.83072.e7",
          "doi": "10.1097/01.aud.0000240507.83072.e7"
        }
      ],
      "ai_section": "2 · Dal tempo alle frequenze (Il logaritmo: log-Mel)",
      "markers": [
        "superior_olive"
      ]
    },
    {
      "id": "aud-codec",
      "ai": "Codec neurale: encoder convoluzionale + quantizzazione vettoriale residua (RVQ) + decoder",
      "brain": "Codifica efficiente dei suoni naturali",
      "regions": [
        "cochlea",
        "brainstem"
      ],
      "text": "Lewicki (2002) e Smith e Lewicki (2006) si sono chiesti quale codice rappresenterebbe nel modo più economico i suoni naturali, come versi di animali, rumori ambientali e parlato. Lasciando che un algoritmo lo imparasse, hanno ottenuto filtri simili a quelli misurati nelle fibre del nervo acustico, e per il parlato un codice a impulsi più efficiente delle rappresentazioni di Fourier e delle wavelet. Anche un codec neurale (SoundStream, EnCodec) impara dai dati un codice compatto, poi lo trasforma in token con più codebook in cascata (RVQ).",
      "similar": "Imparare dai dati un codice compatto ed efficiente per i suoni reali.",
      "different": "Il codec è addestrato a ricostruire la forma d'onda e a produrre token da un vocabolario fisso; il nervo acustico non ha codebook e il cervello non deve ricostruire il suono. La RVQ, in cui ogni codebook corregge l'errore del precedente, non ha un equivalente biologico noto.",
      "strength": "media",
      "sources": [
        {
          "id": "lewicki2002",
          "title": "Efficient coding of natural sounds",
          "authors": "Michael S. Lewicki",
          "year": 2002,
          "venue": "Nature Neuroscience 5:356-363",
          "url": "https://doi.org/10.1038/nn831",
          "doi": "10.1038/nn831"
        },
        {
          "id": "smith2006",
          "title": "Efficient auditory coding",
          "authors": "Evan C. Smith, Michael S. Lewicki",
          "year": 2006,
          "venue": "Nature 439:978-982",
          "url": "https://doi.org/10.1038/nature04485",
          "doi": "10.1038/nature04485"
        },
        {
          "id": "zeghidour2021",
          "title": "SoundStream: an end-to-end neural audio codec",
          "authors": "Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, Marco Tagliasacchi",
          "year": 2021,
          "venue": "IEEE/ACM TASLP (arXiv:2107.03312)",
          "url": "https://arxiv.org/abs/2107.03312"
        },
        {
          "id": "defossez2022",
          "title": "High fidelity neural audio compression (EnCodec)",
          "authors": "Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi",
          "year": 2022,
          "venue": "arXiv:2210.13438",
          "url": "https://arxiv.org/abs/2210.13438"
        }
      ],
      "ai_section": "3 · Il codec neurale: suono → token → suono",
      "markers": [
        "cochlear_nucleus"
      ]
    },
    {
      "id": "aud-encoder",
      "ai": "Encoder di Whisper: convoluzioni + transformer sul log-Mel",
      "brain": "Gerarchia uditiva: corteccia uditiva primaria (giro di Heschl) → giro temporale superiore",
      "regions": [
        "a1",
        "stg",
        "thalamus"
      ],
      "text": "Li et al. (2023) hanno confrontato registrazioni intracraniche della corteccia uditiva (pazienti con elettrodi impiantati) con reti neurali per il parlato: gli strati più profondi delle reti somigliano di più alle aree uditive di livello alto, e i modelli auto-supervisionati funzionano almeno quanto quelli supervisionati. Millet et al. (2022) hanno trovato la stessa corrispondenza di gerarchia confrontando wav2vec 2.0 con la fMRI di 412 persone, con sole 600 ore di parlato non etichettato per l'addestramento. Kell et al. (2018) avevano già visto che una rete addestrata a riconoscere parole e generi musicali sviluppa una gerarchia simile a quella della corteccia uditiva.",
      "similar": "Le prime elaborazioni sono vicine al suono, le successive ai fonemi e alle parole: l'ordine dei layer segue l'ordine delle aree.",
      "different": "La somiglianza è una correlazione misurata con modelli lineari, non un'identità; il cervello elabora il suono in tempo reale e con feedback dall'alto, mentre l'encoder di Whisper guarda 30 secondi tutti insieme. Questi studi hanno usato altre reti per il parlato: per Whisper il confronto diretto è nello studio di Goldstein et al. (2025), nella scheda successiva.",
      "strength": "forte",
      "sources": [
        {
          "id": "li2023",
          "title": "Dissecting neural computations in the human auditory pathway using deep neural networks for speech",
          "authors": "Yuanning Li, Gopala K. Anumanchipalli, Abdelrahman Mohamed, et al. (ultimo autore Edward F. Chang)",
          "year": 2023,
          "venue": "Nature Neuroscience 26(12):2213-2225",
          "url": "https://doi.org/10.1038/s41593-023-01468-4",
          "doi": "10.1038/s41593-023-01468-4"
        },
        {
          "id": "millet2022",
          "title": "Toward a realistic model of speech processing in the brain with self-supervised learning",
          "authors": "Juliette Millet, Charlotte Caucheteux, Pierre Orhan, et al. (ultimo autore Jean-Rémi King)",
          "year": 2022,
          "venue": "NeurIPS 2022 (arXiv:2206.01685)",
          "url": "https://arxiv.org/abs/2206.01685"
        },
        {
          "id": "kell2018",
          "title": "A task-optimized neural network replicates human auditory behavior, predicts brain responses, and reveals a cortical processing hierarchy",
          "authors": "Alexander J. E. Kell, Daniel L. K. Yamins, Erica N. Shook, Sam V. Norman-Haignere, Josh H. McDermott",
          "year": 2018,
          "venue": "Neuron 98(3):630-644",
          "url": "https://doi.org/10.1016/j.neuron.2018.03.044",
          "doi": "10.1016/j.neuron.2018.03.044"
        }
      ],
      "ai_section": "5 · Dalla voce al testo: Whisper (Encoder)",
      "pathways": [
        "auditory_radiation"
      ],
      "markers": [
        "mgn"
      ]
    },
    {
      "id": "aud-decoder",
      "ai": "Decoder di Whisper: testo token per token (con i token speciali)",
      "brain": "Giro temporale superiore e area di Wernicke: dai tratti fonetici alle parole",
      "regions": [
        "stg",
        "wernicke",
        "broca"
      ],
      "text": "Mesgarani et al. (2014) hanno scoperto che singoli elettrodi sul giro temporale superiore rispondono a tratti fonetici, organizzati soprattutto per modo di articolazione (occlusive, fricative, nasali e così via). Goldstein et al. (2025) hanno registrato 100 ore di conversazioni reali di quattro pazienti e hanno usato proprio Whisper: i suoi embedding acustici, del parlato e delle parole predicono l'attività cerebrale a ogni livello, prima di parlare e dopo aver ascoltato.",
      "similar": "Dal suono si passa a unità linguistiche sempre più astratte, fino alle parole.",
      "different": "Whisper trascrive scegliendo token di testo da un vocabolario; il cervello non scrive testo: capisce, risponde e intanto prevede. Le aree del linguaggio comunicano in entrambe le direzioni (il fascicolo arcuato collega le zone temporali e frontali) e servono anche a parlare, non solo ad ascoltare.",
      "strength": "forte",
      "sources": [
        {
          "id": "mesgarani2014",
          "title": "Phonetic feature encoding in human superior temporal gyrus",
          "authors": "Nima Mesgarani, Connie Cheung, Keith Johnson, Edward F. Chang",
          "year": 2014,
          "venue": "Science 343(6174):1006-1010",
          "url": "https://doi.org/10.1126/science.1245994",
          "doi": "10.1126/science.1245994"
        },
        {
          "id": "goldstein2025",
          "title": "A unified acoustic-to-speech-to-language embedding space captures the neural basis of natural language processing in everyday conversations",
          "authors": "Ariel Goldstein, et al. (laboratorio di Uri Hasson, Princeton, con Google Research)",
          "year": 2025,
          "venue": "Nature Human Behaviour 9(5)",
          "url": "https://doi.org/10.1038/s41562-025-02105-9",
          "doi": "10.1038/s41562-025-02105-9"
        },
        {
          "id": "radford2022whisper",
          "title": "Robust speech recognition via large-scale weak supervision (Whisper)",
          "authors": "Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever",
          "year": 2022,
          "venue": "ICML 2023 (arXiv:2212.04356)",
          "url": "https://arxiv.org/abs/2212.04356"
        },
        {
          "id": "hickok2007",
          "title": "The cortical organization of speech processing",
          "authors": "Gregory Hickok, David Poeppel",
          "year": 2007,
          "venue": "Nature Reviews Neuroscience 8:393-402",
          "url": "https://doi.org/10.1038/nrn2113",
          "doi": "10.1038/nrn2113"
        }
      ],
      "ai_section": "5 · Dalla voce al testo: Whisper (Decoder)",
      "pathways": [
        "arcuate"
      ]
    },
    {
      "id": "aud-musica",
      "ai": "Generare musica token per token (MusicGen, delay pattern)",
      "brain": "Aspettativa musicale, sorpresa e dopamina",
      "regions": [
        "striatum",
        "vta_snc",
        "a1",
        "amygdala",
        "hippocampus"
      ],
      "text": "Cheung et al. (2019) hanno analizzato con un modello statistico 80.000 accordi di canzoni pop: il piacere dipende dalla combinazione di incertezza e sorpresa, e amigdala, ippocampo e corteccia uditiva riflettono questa combinazione. Salimpoor et al. (2011) hanno misurato un rilascio di dopamina nello striato legato ai momenti di massima emozione musicale: nel nucleo caudato durante l'attesa, nel nucleo accumbens al culmine. MusicGen genera la musica prevedendo il pezzo successivo, e anche chi ascolta fa di continuo previsioni su cosa verrà dopo.",
      "similar": "La musica si costruisce, e si gusta, su previsioni: cosa verrà dopo e quanto sarà sorprendente.",
      "different": "Il modello non prova piacere né brividi: rende più probabili i token simili a quelli visti nel training. Nel cervello la musica attiva circuiti di ricompensa ed emozione e, secondo Norman-Haignere, Kanwisher e McDermott (2015), popolazioni della corteccia uditiva selettive per la musica e distinte da quelle per il parlato; MusicGen usa per la musica lo stesso tipo di codec e di transformer usato per altri suoni.",
      "strength": "media",
      "sources": [
        {
          "id": "cheung2019",
          "title": "Uncertainty and surprise jointly predict musical pleasure and amygdala, hippocampus, and auditory cortex activity",
          "authors": "Vincent K. M. Cheung, Peter M. C. Harrison, Lars Meyer, Marcus T. Pearce, John-Dylan Haynes, Stefan Koelsch",
          "year": 2019,
          "venue": "Current Biology 29(23):4084-4092",
          "url": "https://doi.org/10.1016/j.cub.2019.09.067",
          "doi": "10.1016/j.cub.2019.09.067"
        },
        {
          "id": "salimpoor2011",
          "title": "Anatomically distinct dopamine release during anticipation and experience of peak emotion to music",
          "authors": "Valorie N. Salimpoor, Mitchel Benovoy, Kevin Larcher, Alain Dagher, Robert J. Zatorre",
          "year": 2011,
          "venue": "Nature Neuroscience 14:257-262",
          "url": "https://doi.org/10.1038/nn.2726",
          "doi": "10.1038/nn.2726"
        },
        {
          "id": "normanhaignere2015",
          "title": "Distinct cortical pathways for music and speech revealed by hypothesis-free voxel decomposition",
          "authors": "Sam Norman-Haignere, Nancy G. Kanwisher, Josh H. McDermott",
          "year": 2015,
          "venue": "Neuron 88(6):1281-1296",
          "url": "https://doi.org/10.1016/j.neuron.2015.11.035",
          "doi": "10.1016/j.neuron.2015.11.035"
        },
        {
          "id": "copet2023",
          "title": "Simple and controllable music generation (MusicGen)",
          "authors": "Jade Copet, Felix Kreuk, Itai Gat, et al.",
          "year": 2023,
          "venue": "NeurIPS 2023 (arXiv:2306.05284)",
          "url": "https://arxiv.org/abs/2306.05284"
        }
      ],
      "ai_section": "4 · Generare audio con un transformer",
      "pathways": [
        "mesolimbic"
      ]
    },
    {
      "id": "aud-voce",
      "ai": "Sintesi vocale con token di codec (VALL-E)",
      "brain": "Produzione del linguaggio nella corteccia sensorimotoria e neuroprotesi vocali",
      "regions": [
        "motor",
        "somato",
        "premotor",
        "broca"
      ],
      "text": "Bouchard et al. (2013) hanno mostrato che la corteccia sensorimotoria ventrale controlla labbra, lingua, mandibola e laringe con una mappa ordinata. Nel 2023 due gruppi hanno decodificato con reti neurali il parlato tentato da persone paralizzate: Willett et al. hanno raggiunto 62 parole al minuto (testo), Metzger et al. una mediana di 78 parole al minuto con testo, voce sintetizzata e un avatar. Qui l'AI non imita il cervello: lo «legge» e gli restituisce una voce.",
      "similar": "In comune c'è il passo finale: trasformare una rappresentazione interna (attività neurale o token) in voce. Il legame sperimentale riguarda reti neurali usate per leggere la corteccia, non VALL-E: per questo la forza è «media» e non «forte».",
      "different": "VALL-E genera voce dal testo e da 3 secondi di registrazione di esempio, senza alcun segnale cerebrale; le neuroprotesi traducono l'attività registrata da elettrodi impiantati. Sono studi clinici sperimentali su pochi pazienti, con vocabolari ed errori diversi tra loro.",
      "strength": "media",
      "sources": [
        {
          "id": "bouchard2013",
          "title": "Functional organization of human sensorimotor cortex for speech articulation",
          "authors": "Kristofer E. Bouchard, Nima Mesgarani, Keith Johnson, Edward F. Chang",
          "year": 2013,
          "venue": "Nature 495:327-332",
          "url": "https://doi.org/10.1038/nature11911",
          "doi": "10.1038/nature11911"
        },
        {
          "id": "willett2023",
          "title": "A high-performance speech neuroprosthesis",
          "authors": "Francis R. Willett, Erin M. Kunz, Chaofei Fan, et al.",
          "year": 2023,
          "venue": "Nature 620:1031-1036",
          "url": "https://doi.org/10.1038/s41586-023-06377-x",
          "doi": "10.1038/s41586-023-06377-x"
        },
        {
          "id": "metzger2023",
          "title": "A high-performance neuroprosthesis for speech decoding and avatar control",
          "authors": "Sean L. Metzger, Kaylo T. Littlejohn, Alexander B. Silva, et al. (ultimo autore Edward F. Chang)",
          "year": 2023,
          "venue": "Nature 620:1037-1046",
          "url": "https://doi.org/10.1038/s41586-023-06443-4",
          "doi": "10.1038/s41586-023-06443-4"
        },
        {
          "id": "wang2023valle",
          "title": "Neural codec language models are zero-shot text to speech synthesizers (VALL-E)",
          "authors": "Chengyi Wang, Sanyuan Chen, Yu Wu, et al.",
          "year": 2023,
          "venue": "arXiv:2301.02111",
          "url": "https://arxiv.org/abs/2301.02111"
        }
      ],
      "ai_section": "6 · Altri modi di generare audio (Sintesi vocale con token di codec)"
    }
  ],
  "facts": [
    {
      "claim": "Cellule ciliate della coclea umana",
      "value": "circa 3.500 interne e 11.000 esterne",
      "note": "Altre fonti danno fino a 15.000 cellule esterne: i conteggi variano.",
      "source": {
        "id": "ashmore2008",
        "title": "Cochlear outer hair cell motility",
        "authors": "Jonathan Ashmore",
        "year": 2008,
        "venue": "Physiological Reviews 88(1):173-210",
        "url": "https://doi.org/10.1152/physrev.00044.2006",
        "doi": "10.1152/physrev.00044.2006",
        "note": "Circa 3.500 cellule ciliate interne e circa 11.000 esterne per coclea umana (valori citati da BioNumbers, voce 117129, da questa rassegna)."
      }
    },
    {
      "claim": "Neuroni del ganglio spirale (nervo acustico)",
      "value": "33.679 in media nel primo decennio di vita, 22.444 nel decimo",
      "source": {
        "id": "makary2011",
        "title": "Age-related primary cochlear neuronal degeneration in human temporal bones",
        "authors": "Chadi A. Makary, Jennifer Shin, Sharon G. Kujawa, M. Charles Liberman, Saumil N. Merchant",
        "year": 2011,
        "venue": "Journal of the Association for Research in Otolaryngology 12(6):711-717",
        "url": "https://doi.org/10.1007/s10162-011-0283-2",
        "doi": "10.1007/s10162-011-0283-2"
      }
    },
    {
      "claim": "Cellule ciliate esterne: l'amplificatore della coclea",
      "value": "Le cellule esterne cambiano lunghezza con il potenziale (elettromotilità) e amplificano il moto della membrana basilare.",
      "source": {
        "id": "ashmore2008",
        "title": "Cochlear outer hair cell motility",
        "authors": "Jonathan Ashmore",
        "year": 2008,
        "venue": "Physiological Reviews 88(1):173-210",
        "url": "https://doi.org/10.1152/physrev.00044.2006",
        "doi": "10.1152/physrev.00044.2006",
        "note": "Circa 3.500 cellule ciliate interne e circa 11.000 esterne per coclea umana (valori citati da BioNumbers, voce 117129, da questa rassegna)."
      }
    },
    {
      "claim": "Membrana basilare: una risposta compressiva",
      "value": "La risposta della membrana basilare cresce in modo fortemente compressivo con l'intensità.",
      "source": {
        "id": "ruggero1997",
        "title": "Basilar-membrane responses to tones at the base of the chinchilla cochlea",
        "authors": "Mario A. Ruggero, Nola C. Rich, Alberto Recio, S. Shyamla Narayan, Luis Robles",
        "year": 1997,
        "venue": "Journal of the Acoustical Society of America 101(4):2151-2163",
        "url": "https://doi.org/10.1121/1.418265",
        "doi": "10.1121/1.418265"
      }
    },
    {
      "claim": "Il cervello regola la coclea",
      "value": "Le cellule esterne ricevono fibre efferenti dall'oliva superiore (sistema olivococleare).",
      "source": {
        "id": "guinan2006",
        "title": "Olivocochlear efferents: anatomy, physiology, function, and the measurement of efferent effects in humans",
        "authors": "John J. Guinan Jr",
        "year": 2006,
        "venue": "Ear and Hearing 27(6):589-607",
        "url": "https://doi.org/10.1097/01.aud.0000240507.83072.e7",
        "doi": "10.1097/01.aud.0000240507.83072.e7"
      }
    },
    {
      "claim": "Fibre del nervo acustico: impulsi agganciati alla fase",
      "value": "Gli impulsi seguono la fase dell'onda sonora fino a qualche kHz (gatto).",
      "source": {
        "id": "johnson1980",
        "title": "The relationship between spike rate and synchrony in responses of auditory-nerve fibers to single tones",
        "authors": "Don H. Johnson",
        "year": 1980,
        "venue": "Journal of the Acoustical Society of America 68(4):1115-1122",
        "url": "https://doi.org/10.1121/1.384982",
        "doi": "10.1121/1.384982"
      }
    },
    {
      "claim": "Fibre del nervo acustico: tre classi di scarica spontanea",
      "value": "Tre classi di fibre per frequenza spontanea: bassa (< 0,5/s), media (0,5–18/s) e alta (> 18/s) (gatto).",
      "source": {
        "id": "liberman1978",
        "title": "Auditory-nerve response from cats raised in a low-noise chamber",
        "authors": "M. Charles Liberman",
        "year": 1978,
        "venue": "Journal of the Acoustical Society of America 63(2):442-455",
        "url": "https://doi.org/10.1121/1.381736",
        "doi": "10.1121/1.381736"
      }
    },
    {
      "claim": "Oliva superiore: il confronto tra le due orecchie",
      "value": "L'oliva superiore confronta i tempi di arrivo del suono alle due orecchie, con differenze di decine di microsecondi.",
      "source": {
        "id": "grothe2010",
        "title": "Mechanisms of sound localization in mammals",
        "authors": "Benedikt Grothe, Michael Pecka, David McAlpine",
        "year": 2010,
        "venue": "Physiological Reviews 90(3):983-1012",
        "url": "https://doi.org/10.1152/physrev.00026.2009",
        "doi": "10.1152/physrev.00026.2009"
      }
    },
    {
      "claim": "Neuroni di relè del talamo: due modi di scaricare",
      "value": "Due modalità di scarica: tonica (impulsi singoli) e a raffiche (burst).",
      "source": {
        "id": "sherman2001",
        "title": "Tonic and burst firing: dual modes of thalamocortical relay",
        "authors": "S. Murray Sherman",
        "year": 2001,
        "venue": "Trends in Neurosciences 24(2):122-126",
        "url": "https://doi.org/10.1016/S0166-2236(00)01714-8",
        "doi": "10.1016/S0166-2236(00)01714-8"
      }
    }
  ],
  "sources": [
    {
      "id": "stevens1937",
      "title": "A scale for the measurement of the psychological magnitude pitch",
      "authors": "S. S. Stevens, J. Volkmann, E. B. Newman",
      "year": 1937,
      "venue": "Journal of the Acoustical Society of America 8(3):185-190",
      "url": "https://doi.org/10.1121/1.1915893",
      "doi": "10.1121/1.1915893"
    },
    {
      "id": "li2023",
      "title": "Dissecting neural computations in the human auditory pathway using deep neural networks for speech",
      "authors": "Yuanning Li, Gopala K. Anumanchipalli, Abdelrahman Mohamed, et al. (ultimo autore Edward F. Chang)",
      "year": 2023,
      "venue": "Nature Neuroscience 26(12):2213-2225",
      "url": "https://doi.org/10.1038/s41593-023-01468-4",
      "doi": "10.1038/s41593-023-01468-4"
    }
  ]
};
