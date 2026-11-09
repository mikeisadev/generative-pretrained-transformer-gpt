/**
 * Atlante del cervello stilizzato del nucleo: regioni, marcatori dei nuclei e percorsi (fasci di fibre),
 * con il nome italiano, il centroide in coordinate MNI (millimetri), un raggio indicativo, il lobo,
 * l'emisfero, il tipo (corticale, profonda, periferica) e una descrizione breve. Solo dati e funzioni
 * pure: lo usano la vista 3D, il pannello e i test (tests/brain.test.mjs).
 *
 * Coordinate MNI: x da sinistra (−) a destra (+), y da dietro (−) ad avanti (+), z da sotto (−) a sopra
 * (+), origine nella commissura anteriore. I valori sono quelli dell'emisfero SINISTRO: il modello è
 * simmetrico, quindi per il destro si specchia x (mirrorMni).
 *
 * Fonti dei centroidi (riportate in `source` di ogni voce):
 *  - "aal": centroidi delle regioni AAL (Tzourio-Mazoyer et al. 2002, NeuroImage 15:273-289,
 *    doi:10.1006/nimg.2001.0978) come nella tabella dei 90 nodi distribuita con BrainNet Viewer
 *    (Xia, Wang e He 2013, PLoS ONE 8:e68910, doi:10.1371/journal.pone.0068910), arrotondati al mm.
 *    Quando una regione del corso unisce più regioni AAL il centroide è la media indicata nella nota.
 *  - "lit": coordinate di articoli che hanno definito la regione con la risonanza funzionale (citati
 *    nella nota); se erano in spazio di Talairach sono convertite in MNI in modo approssimato.
 *  - "approx": posizione anatomica approssimata, scelta da noi guardando l'anatomia descritta nella
 *    nota (piccoli nuclei del tronco e del talamo, occhio, coclea): va presa come indicativa.
 * In ogni caso la vista lo dichiara: "modello stilizzato: forme e posizioni approssimate (coordinate MNI)".
 */

/** Quante unità di scena per millimetro: 1 unità = 1 cm (il cervello intero è lungo ~17 unità). */
export const MM = 0.1;

/** Punto MNI (mm) che finisce nell'origine della scena: circa il centro del cervello. */
export const MNI_CENTER = Object.freeze([0, -18, 5]);

/**
 * Da MNI (mm) alle coordinate della scena (three.js, y in alto). Rotazione propria (niente specchio:
 * sinistra resta sinistra): x_scena = −(y − y0) (davanti a sinistra dello schermo quando guardi
 * l'emisfero sinistro), y_scena = z (in alto), z_scena = −x (l'emisfero sinistro verso la camera
 * predefinita, che sta su +z). Così la vista iniziale è quella dei libri: emisfero sinistro di lato,
 * lobo frontale a sinistra, occipitale a destra, area di Broca davanti e area di Wernicke dietro.
 */
export function mniToScene(p, out = [0, 0, 0]) {
  const x = p[0];
  const y = p[1];
  const z = p[2];
  out[0] = -(y - MNI_CENTER[1]) * MM;
  out[1] = (z - MNI_CENTER[2]) * MM;
  out[2] = -(x - MNI_CENTER[0]) * MM;
  return out;
}

/** Inversa di mniToScene. */
export function sceneToMni(p, out = [0, 0, 0]) {
  const X = p[0];
  const Y = p[1];
  const Z = p[2];
  out[0] = -Z / MM + MNI_CENTER[0];
  out[1] = -X / MM + MNI_CENTER[1];
  out[2] = Y / MM + MNI_CENTER[2];
  return out;
}

/** Lo stesso punto nell'altro emisfero. */
export const mirrorMni = (p) => [-p[0], p[1], p[2]];

/** Lobi e parti del cervello (per la scheda e per i colori dei lobi). */
export const LOBES = Object.freeze({
  frontale: { label: "Lobo frontale", color: "#6f9bd8" },
  parietale: { label: "Lobo parietale", color: "#d8bf5f" },
  temporale: { label: "Lobo temporale", color: "#6fbf8f" },
  occipitale: { label: "Lobo occipitale", color: "#d585b0" },
  insula: { label: "Insula", color: "#d99a5f" },
  limbico: { label: "Sistema limbico (lobo temporale mediale e cingolo)", color: "#c9a0dc" },
  profondo: { label: "Strutture profonde", color: "#9fb4c7" },
  cervelletto: { label: "Cervelletto", color: "#b8a5d8" },
  tronco: { label: "Tronco encefalico", color: "#b5aaa0" },
  periferia: { label: "Organi di senso", color: "#cfd8dc" },
});

/** Tipi di voce dell'atlante. */
export const KINDS = Object.freeze({
  cortical: "corticale",
  deep: "profonda",
  peripheral: "periferica",
  nucleus: "nucleo",
  tract: "fascio di fibre",
});

/** Emisferi: "left" e "right" per le funzioni lateralizzate, "both" per le regioni doppie, "midline" al centro. */
export const HEMISPHERES = Object.freeze({ left: "sinistro", right: "destro", both: "entrambi gli emisferi", midline: "linea mediana" });

/**
 * Regioni. Campi: id, name (nome italiano), mni (mm, emisfero sinistro), radius (mm, indicativo:
 * quanto è grande la macchia evidenziata), lobe, hemisphere, kind, text (cosa fa, in breve),
 * source ("aal" | "lit" | "approx") e note (da dove viene il centroide).
 */
export const REGIONS = Object.freeze([
  {
    id: "pfc_dl",
    name: "Corteccia prefrontale dorsolaterale",
    mni: [-38, 34, 30],
    radius: 15,
    lobe: "frontale",
    hemisphere: "both",
    kind: "cortical",
    text: "Tiene a mente le informazioni mentre ci lavori (memoria di lavoro), pianifica e controlla il comportamento verso un obiettivo.",
    source: "aal",
    note: "Giro frontale medio: centroide AAL Frontal_Mid_L (−33, 33, 35), spostato verso l'area 46 (più laterale e più in basso).",
  },
  {
    id: "pfc_vm",
    name: "Corteccia prefrontale ventromediale e orbitofrontale",
    mni: [-6, 46, -12],
    radius: 14,
    lobe: "frontale",
    hemisphere: "both",
    kind: "cortical",
    text: "Dà un valore alle opzioni e alle loro conseguenze: entra nelle decisioni, nelle ricompense e nelle emozioni legate alle scelte.",
    source: "aal",
    note: "Media dei centroidi AAL Frontal_Med_Orb_L (−5, 54, −7) e Rectus_L (−5, 37, −18).",
  },
  {
    id: "acc",
    name: "Corteccia cingolata anteriore",
    mni: [-4, 35, 14],
    radius: 11,
    lobe: "limbico",
    hemisphere: "both",
    kind: "cortical",
    text: "Sulla faccia interna dell'emisfero, sopra il corpo calloso: segnala conflitti ed errori e regola lo sforzo quando qualcosa non va come previsto.",
    source: "aal",
    note: "Centroide AAL Cingulum_Ant_L.",
  },
  {
    id: "broca",
    name: "Area di Broca",
    mni: [-47, 21, 17],
    radius: 12,
    lobe: "frontale",
    hemisphere: "left",
    kind: "cortical",
    text: "Giro frontale inferiore sinistro (aree 44 e 45): produzione del linguaggio, sintassi e pianificazione delle parole.",
    source: "aal",
    note: "Media dei centroidi AAL Frontal_Inf_Oper_L (−48, 13, 19) e Frontal_Inf_Tri_L (−46, 30, 14).",
  },
  {
    id: "premotor",
    name: "Corteccia premotoria e area motoria supplementare",
    mni: [-26, 2, 58],
    radius: 13,
    lobe: "frontale",
    hemisphere: "both",
    kind: "cortical",
    text: "Davanti alla corteccia motoria: prepara e mette in sequenza i movimenti prima che vengano eseguiti.",
    source: "aal",
    note: "Tra il centroide AAL Supp_Motor_Area_L (−5, 5, 61, sulla faccia interna) e la premotoria dorsale laterale.",
  },
  {
    id: "motor",
    name: "Corteccia motoria primaria (M1)",
    mni: [-39, -6, 51],
    radius: 13,
    lobe: "frontale",
    hemisphere: "both",
    kind: "cortical",
    text: "Giro precentrale, davanti al solco centrale: comanda i movimenti volontari; ogni parte del corpo ha la sua zona (omuncolo motorio).",
    source: "aal",
    note: "Centroide AAL Precentral_L.",
  },
  {
    id: "somato",
    name: "Corteccia somatosensoriale primaria (S1)",
    mni: [-42, -23, 49],
    radius: 13,
    lobe: "parietale",
    hemisphere: "both",
    kind: "cortical",
    text: "Giro postcentrale, dietro al solco centrale: riceve tatto, posizione del corpo e dolore, con una mappa ordinata del corpo.",
    source: "aal",
    note: "Centroide AAL Postcentral_L.",
  },
  {
    id: "parietal_post",
    name: "Corteccia parietale posteriore (solco intraparietale)",
    mni: [-33, -53, 53],
    radius: 14,
    lobe: "parietale",
    hemisphere: "both",
    kind: "cortical",
    text: "Integra vista e movimento nello spazio e dirige l'attenzione verso ciò che conta in quel momento.",
    source: "aal",
    note: "Media dei centroidi AAL Parietal_Sup_L (−23, −60, 59) e Parietal_Inf_L (−43, −46, 47), ai due lati del solco intraparietale.",
  },
  {
    id: "angular",
    name: "Giro angolare",
    mni: [-44, -61, 36],
    radius: 11,
    lobe: "parietale",
    hemisphere: "both",
    kind: "cortical",
    text: "Snodo tra lettura, significato delle parole, numeri e memoria degli eventi.",
    source: "aal",
    note: "Centroide AAL Angular_L.",
  },
  {
    id: "wernicke",
    name: "Area di Wernicke",
    mni: [-55, -42, 12],
    radius: 11,
    lobe: "temporale",
    hemisphere: "left",
    kind: "cortical",
    text: "Parte posteriore del giro temporale superiore sinistro: comprensione dei suoni del linguaggio e delle parole.",
    source: "lit",
    note: "Terzo posteriore del giro temporale superiore sinistro (area 22 posteriore); i confini dell'area variano tra gli autori (Binder 2015, Neurology 85:2170-2175, doi:10.1212/WNL.0000000000002219).",
  },
  {
    id: "a1",
    name: "Corteccia uditiva primaria (giro di Heschl)",
    mni: [-42, -19, 10],
    radius: 9,
    lobe: "temporale",
    hemisphere: "both",
    kind: "cortical",
    text: "Nascosta nella scissura laterale, sulla faccia superiore del lobo temporale: la prima stazione corticale dell'udito, ordinata per frequenze (tonotopia).",
    source: "aal",
    note: "Centroide AAL Heschl_L.",
  },
  {
    id: "stg",
    name: "Giro temporale superiore",
    mni: [-53, -21, 7],
    radius: 13,
    lobe: "temporale",
    hemisphere: "both",
    kind: "cortical",
    text: "Analizza suoni complessi (voci, fonemi, sillabe); insieme al solco temporale superiore integra udito e vista.",
    source: "aal",
    note: "Centroide AAL Temporal_Sup_L.",
  },
  {
    id: "temporal_ant",
    name: "Lobo temporale anteriore (polo temporale)",
    mni: [-38, 15, -27],
    radius: 12,
    lobe: "temporale",
    hemisphere: "both",
    kind: "cortical",
    text: "Concetti e conoscenze sul significato (memoria semantica), anche sulle persone che conosci.",
    source: "aal",
    note: "Media dei centroidi AAL Temporal_Pole_Sup_L (−40, 15, −20) e Temporal_Pole_Mid_L (−36, 15, −34).",
  },
  {
    id: "it",
    name: "Corteccia temporale inferiore",
    mni: [-50, -28, -23],
    radius: 13,
    lobe: "temporale",
    hemisphere: "both",
    kind: "cortical",
    text: "Ultima stazione della via visiva ventrale: riconosce oggetti e categorie anche se cambiano posizione e dimensione.",
    source: "aal",
    note: "Centroide AAL Temporal_Inf_L.",
  },
  {
    id: "ffa",
    name: "Area fusiforme dei volti",
    mni: [-40, -56, -15],
    radius: 8,
    lobe: "temporale",
    hemisphere: "right",
    kind: "cortical",
    text: "Nel giro fusiforme, sulla faccia inferiore del lobo temporale, soprattutto a destra: risponde in modo selettivo ai volti.",
    source: "lit",
    // mni come sempre per l'emisfero sinistro: con hemisphere "right" placements() lo specchia a destra
    note: "Kanwisher, McDermott e Chun 1997 (J Neurosci 17:4302-4311, doi:10.1523/JNEUROSCI.17-11-04302.1997): FFA destra a (40, −55, −10) in Talairach, ≈ (40, −56, −15) in MNI.",
  },
  {
    id: "v1",
    name: "Corteccia visiva primaria (V1)",
    mni: [-7, -79, 6],
    radius: 12,
    lobe: "occipitale",
    hemisphere: "both",
    kind: "cortical",
    text: "Lungo la scissura calcarina, sulla faccia interna del lobo occipitale: neuroni che rispondono a bordi e orientamenti in punti precisi del campo visivo.",
    source: "aal",
    note: "Centroide AAL Calcarine_L.",
  },
  {
    id: "v2v4",
    name: "Corteccia visiva extrastriata (V2, V4)",
    mni: [-30, -82, -8],
    radius: 13,
    lobe: "occipitale",
    hemisphere: "both",
    kind: "cortical",
    text: "Combina i bordi in contorni, forme, texture e colori.",
    source: "aal",
    note: "Vicino al centroide AAL Occipital_Inf_L (−36, −78, −8), verso il giro linguale (V4 umana sta sulla faccia inferiore del lobo occipitale).",
  },
  {
    id: "mt",
    name: "Area MT/V5 del movimento",
    mni: [-46, -77, 0],
    radius: 9,
    lobe: "occipitale",
    hemisphere: "both",
    kind: "cortical",
    text: "Al confine tra lobo occipitale e temporale: specializzata nel movimento visivo (direzione e velocità).",
    source: "lit",
    note: "Dumoulin et al. 2000 (Cereb Cortex 10:454-463, doi:10.1093/cercor/10.5.454): V5/MT+ a sinistra (−47, −76, 2) in Talairach, ≈ (−46, −77, 0) in MNI.",
  },
  {
    id: "insula",
    name: "Insula",
    mni: [-35, 7, 3],
    radius: 12,
    lobe: "insula",
    hemisphere: "both",
    kind: "cortical",
    text: "Nascosta in fondo alla scissura laterale: sensazioni dall'interno del corpo, gusto, emozioni e consapevolezza dello stato del corpo.",
    source: "aal",
    note: "Centroide AAL Insula_L.",
  },
  {
    id: "entorhinal",
    name: "Corteccia entorinale",
    mni: [-23, -9, -30],
    radius: 8,
    lobe: "limbico",
    hemisphere: "both",
    kind: "cortical",
    text: "La porta d'ingresso dell'ippocampo, sulla parte anteriore del giro paraippocampale: qui ci sono le cellule di griglia e da qui parte la via perforante.",
    source: "approx",
    note: "Parte anteriore del giro paraippocampale, sotto amigdala e testa dell'ippocampo (Insausti et al. 1998, AJNR 19:659-671); posizione approssimata (in AAL è dentro ParaHippocampal_L, centroide −21, −16, −21).",
  },
  {
    id: "parahippocampal",
    name: "Corteccia paraippocampale",
    mni: [-25, -40, -12],
    radius: 9,
    lobe: "limbico",
    hemisphere: "both",
    kind: "cortical",
    text: "Risponde a luoghi e scene (l'area dei luoghi, PPA) e lega il contesto ai ricordi.",
    source: "lit",
    note: "Parte posteriore del giro paraippocampale, dove Epstein e Kanwisher 1998 (Nature 392:598-601, doi:10.1038/33402) trovano l'area dei luoghi; posizione approssimata.",
  },
  {
    id: "hippocampus",
    name: "Ippocampo",
    mni: [-25, -21, -10],
    radius: 10,
    lobe: "limbico",
    hemisphere: "both",
    kind: "deep",
    text: "A forma di cavalluccio marino, nel lobo temporale mediale: forma i nuovi ricordi degli eventi e costruisce mappe dello spazio.",
    source: "aal",
    note: "Centroide AAL Hippocampus_L.",
  },
  {
    id: "amygdala",
    name: "Amigdala",
    mni: [-23, -1, -17],
    radius: 7,
    lobe: "limbico",
    hemisphere: "both",
    kind: "deep",
    text: "A forma di mandorla, davanti all'ippocampo: dà peso emotivo agli stimoli, soprattutto alla paura, e rafforza i ricordi emotivi.",
    source: "aal",
    note: "Centroide AAL Amygdala_L.",
  },
  {
    id: "thalamus",
    name: "Talamo",
    mni: [-11, -18, 8],
    radius: 10,
    lobe: "profondo",
    hemisphere: "both",
    kind: "deep",
    text: "La stazione di smistamento al centro del cervello: quasi tutti i segnali dei sensi passano di qui prima di arrivare alla corteccia.",
    source: "aal",
    note: "Centroide AAL Thalamus_L.",
  },
  {
    id: "striatum",
    name: "Striato (caudato e putamen)",
    mni: [-18, 7, 6],
    radius: 12,
    lobe: "profondo",
    hemisphere: "both",
    kind: "deep",
    text: "Parte dei gangli della base: impara il valore delle azioni, aiuta a sceglierle e a formare le abitudini.",
    source: "aal",
    note: "Media dei centroidi AAL Caudate_L (−11, 11, 9) e Putamen_L (−24, 4, 2).",
  },
  {
    id: "vta_snc",
    name: "Area tegmentale ventrale e substantia nigra compatta",
    mni: [-7, -18, -13],
    radius: 5,
    lobe: "tronco",
    hemisphere: "both",
    kind: "deep",
    text: "Nel mesencefalo: i neuroni della dopamina, che segnalano l'errore di predizione della ricompensa.",
    source: "approx",
    note: "Mesencefalo ventrale: l'area tegmentale ventrale vicino alla linea mediana e la substantia nigra di lato (atlante della VTA di Murty et al. 2014, NeuroImage 100:580-589, doi:10.1016/j.neuroimage.2014.06.047); posizione approssimata.",
  },
  {
    id: "cerebellum",
    name: "Cervelletto",
    mni: [-24, -62, -34],
    radius: 22,
    lobe: "cervelletto",
    hemisphere: "both",
    kind: "deep",
    text: "Sotto il lobo occipitale: coordina e affina i movimenti imparando dai propri errori; contiene circa l'80% dei neuroni del cervello.",
    source: "approx",
    note: "Centro approssimato dell'emisfero cerebellare sinistro (regioni Cerebelum_* di AAL).",
  },
  {
    id: "brainstem",
    name: "Tronco encefalico",
    mni: [0, -27, -30],
    radius: 14,
    lobe: "tronco",
    hemisphere: "midline",
    kind: "deep",
    text: "Mesencefalo, ponte e bulbo: collega cervello e midollo spinale, controlla respiro, battito e veglia e ospita le prime stazioni dell'udito.",
    source: "approx",
    note: "Centro approssimato del ponte, sulla linea mediana.",
  },
  {
    id: "corpus_callosum",
    name: "Corpo calloso",
    mni: [0, -6, 20],
    radius: 12,
    lobe: "profondo",
    hemisphere: "midline",
    kind: "tract",
    text: "Il grande fascio di fibre che collega i due emisferi.",
    source: "approx",
    note: "Centro approssimato del corpo calloso, sulla linea mediana (dal ginocchio, davanti, allo splenio, dietro).",
  },
  {
    id: "retina",
    name: "Retina (occhio)",
    mni: [-32, 44, -40],
    radius: 9,
    lobe: "periferia",
    hemisphere: "both",
    kind: "peripheral",
    text: "Lo strato di neuroni in fondo all'occhio: fotorecettori, cellule bipolari e gangliari; gli assoni delle gangliari formano il nervo ottico.",
    source: "approx",
    note: "Fondo dell'occhio nell'orbita; posizione approssimata.",
  },
  {
    id: "cochlea",
    name: "Coclea (orecchio interno)",
    mni: [-42, -18, -44],
    radius: 7,
    lobe: "periferia",
    hemisphere: "both",
    kind: "peripheral",
    text: "A spirale, nell'orecchio interno: le cellule ciliate trasformano le vibrazioni in segnali e ogni tratto della spirale risponde a una frequenza.",
    source: "approx",
    note: "Rocca petrosa dell'osso temporale, sotto il lobo temporale; posizione approssimata e coclea ingrandita nel modello per leggibilità.",
  },
]);

/** Marcatori dei nuclei: piccoli punti dentro una regione (talamo, tronco encefalico). */
export const MARKERS = Object.freeze([
  {
    id: "lgn",
    name: "Nucleo genicolato laterale",
    within: "thalamus",
    mni: [-22, -25, -5],
    radius: 3.5,
    hemisphere: "both",
    kind: "nucleus",
    text: "Nel talamo: riceve gli assoni della retina e manda il segnale a V1 con la radiazione ottica.",
    source: "approx",
    note: "Talamo posterolaterale; posizione approssimata.",
  },
  {
    id: "mgn",
    name: "Nucleo genicolato mediale",
    within: "thalamus",
    mni: [-15, -26, -5],
    radius: 3,
    hemisphere: "both",
    kind: "nucleus",
    text: "Nel talamo: la stazione dell'udito che proietta alla corteccia uditiva primaria con la radiazione uditiva.",
    source: "approx",
    note: "Talamo posteriore, medialmente al genicolato laterale; posizione approssimata (vedi l'atlante delle stazioni uditive di Sitek et al. 2019, eLife 8:e48932).",
  },
  {
    id: "pulvinar",
    name: "Pulvinar",
    within: "thalamus",
    mni: [-15, -29, 5],
    radius: 4.5,
    hemisphere: "both",
    kind: "nucleus",
    text: "Il nucleo più grande del talamo, nella parte posteriore: collegato a molte aree visive, partecipa all'attenzione.",
    source: "approx",
    note: "Parte posteriore del talamo; posizione approssimata.",
  },
  {
    id: "cochlear_nucleus",
    name: "Nucleo cocleare",
    within: "brainstem",
    mni: [-12, -41, -40],
    radius: 3,
    hemisphere: "both",
    kind: "nucleus",
    text: "Al confine tra ponte e bulbo: la prima stazione dell'udito nel cervello, dove arrivano le fibre del nervo acustico.",
    source: "approx",
    note: "Giunzione ponte-bulbo, di lato; posizione approssimata (vedi Sitek et al. 2019, eLife 8:e48932).",
  },
  {
    id: "superior_olive",
    name: "Oliva superiore",
    within: "brainstem",
    mni: [-9, -33, -36],
    radius: 2.8,
    hemisphere: "both",
    kind: "nucleus",
    text: "Nel ponte: confronta tempi e intensità del suono alle due orecchie per capire da dove arriva.",
    source: "approx",
    note: "Ponte inferiore; posizione approssimata (vedi Sitek et al. 2019, eLife 8:e48932).",
  },
  {
    id: "inferior_colliculus",
    name: "Collicolo inferiore",
    within: "brainstem",
    mni: [-5, -36, -9],
    radius: 3.2,
    hemisphere: "both",
    kind: "nucleus",
    text: "Sul dorso del mesencefalo: raccoglie quasi tutte le vie uditive del tronco prima del talamo.",
    source: "approx",
    note: "Tetto del mesencefalo; posizione approssimata (vedi Sitek et al. 2019, eLife 8:e48932).",
  },
]);

/**
 * Percorsi (fasci di fibre). `through`: regioni o marcatori che il fascio collega, in ordine; `path`:
 * punti di passaggio in mm MNI (emisfero sinistro) scelti seguendo l'anatomia descritta nel testo, così
 * il tubo corre dentro la sostanza bianca e non attraversa la superficie a caso. `color`: colore del tubo.
 */
export const PATHWAYS = Object.freeze([
  {
    id: "arcuate",
    name: "Fascicolo arcuato",
    through: ["broca", "wernicke"],
    hemisphere: "left",
    color: "#a78bfa",
    text: "Fascio di fibre che collega le aree del linguaggio davanti (Broca) e dietro (Wernicke), girando attorno all'estremità posteriore della scissura laterale.",
    path: [
      [-44, 20, 15],
      [-38, 6, 25],
      [-36, -14, 29],
      [-38, -34, 28],
      [-42, -46, 18],
      [-50, -44, 9],
    ],
  },
  {
    id: "optic_radiation",
    name: "Radiazione ottica",
    through: ["lgn", "v1"],
    hemisphere: "both",
    color: "#4ade80",
    text: "Dal genicolato laterale alla corteccia visiva primaria: le fibre fanno un'ansa in avanti nel lobo temporale (ansa di Meyer) e poi corrono all'indietro accanto al ventricolo laterale.",
    path: [
      [-22, -25, -5],
      [-30, -16, -6],
      [-33, -30, 0],
      [-30, -52, 4],
      [-20, -70, 6],
      [-9, -80, 6],
    ],
  },
  {
    id: "auditory_radiation",
    name: "Radiazione uditiva",
    through: ["mgn", "a1"],
    hemisphere: "both",
    color: "#f472b6",
    text: "Dal genicolato mediale alla corteccia uditiva primaria, passando sotto il nucleo lenticolare.",
    path: [
      [-15, -26, -5],
      [-24, -25, -1],
      [-33, -22, 5],
      [-41, -19, 9],
    ],
  },
  {
    id: "mesolimbic",
    name: "Via mesolimbica",
    through: ["vta_snc", "striatum", "pfc_vm"],
    hemisphere: "both",
    color: "#facc15",
    text: "Assoni della dopamina dall'area tegmentale ventrale allo striato ventrale (nucleo accumbens) e alla prefrontale; nel modello rappresenta anche le vie nigrostriatale e mesocorticale.",
    path: [
      [-5, -17, -13],
      [-7, -6, -11],
      [-10, 6, -8],
      [-12, 14, -6],
      [-9, 30, -8],
      [-6, 42, -10],
    ],
  },
  {
    id: "perforant_path",
    name: "Via perforante",
    through: ["entorhinal", "hippocampus"],
    hemisphere: "both",
    color: "#2dd4bf",
    text: "Dalla corteccia entorinale al giro dentato e a CA3 dell'ippocampo: la via in cui Bliss e Lømo scoprirono il potenziamento a lungo termine.",
    path: [
      [-23, -9, -29],
      [-26, -14, -24],
      [-28, -19, -17],
      [-26, -23, -11],
    ],
  },
]);

const BY_ID = new Map([...REGIONS, ...MARKERS, ...PATHWAYS].map((entry) => [entry.id, entry]));

/** La voce dell'atlante (regione, marcatore o percorso) con questo id, o null. */
export function atlasEntry(id) {
  return BY_ID.get(id) ?? null;
}

/** "region" | "marker" | "pathway" | null */
export function atlasKind(id) {
  const entry = BY_ID.get(id);
  if (!entry) return null;
  if (REGIONS.includes(entry)) return "region";
  if (MARKERS.includes(entry)) return "marker";
  return "pathway";
}

/** Tutti gli id dell'atlante (regioni, marcatori, percorsi). */
export const ATLAS_IDS = Object.freeze([...BY_ID.keys()]);

/**
 * Le posizioni MNI in cui disegnare una voce: una sola per le regioni lateralizzate o sulla linea
 * mediana, due (sinistra e destra) per quelle doppie. `side`: "left" | "right" | "midline".
 */
export function placements(entry) {
  if (!entry?.mni) return [];
  const left = { side: "left", mni: entry.mni };
  const right = { side: "right", mni: mirrorMni(entry.mni) };
  if (entry.hemisphere === "midline") return [{ side: "midline", mni: entry.mni }];
  if (entry.hemisphere === "left") return [left];
  if (entry.hemisphere === "right") return [right];
  return [left, right];
}

/** Coordinate MNI scritte per la scheda: "x −25, y −21, z −10". */
export function formatMni(p) {
  const n = (v) => `${v < 0 ? "−" : ""}${Math.abs(Math.round(v))}`;
  return `x ${n(p[0])}, y ${n(p[1])}, z ${n(p[2])}`;
}

/** Nome breve di una voce per etichette ed elenchi (senza le parentesi esplicative). */
export function shortName(entry) {
  return entry.name.replace(/\s*\(.*\)\s*$/, "");
}

/** Testo della fonte di una voce per la scheda. */
export function sourceLabel(entry) {
  if (entry?.source === "aal") return "centroide dell'atlante AAL";
  if (entry?.source === "lit") return "coordinate da studi di risonanza funzionale";
  return "posizione anatomica approssimata";
}
