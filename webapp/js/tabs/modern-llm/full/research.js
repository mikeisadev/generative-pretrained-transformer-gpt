/**
 * Fatti della ricerca verificata (8/11/2026) che non appartengono a un solo modello: come sono stati
 * ottenuti i config, i nomi chiesti dallo studente, il panorama degli open-weight e i fatti su come il
 * training produce i pesi e come emergono le capacità. SOLI DATI: li usano i pannelli della vista
 * completa e l'indice della ricerca globale.
 */

/** Data della ricerca: tutti i "di oggi" della vista si riferiscono a questa. */
export const RESEARCH_DATE = "8 novembre 2026";

/** Come sono stati ottenuti i numeri (pannello Fonti). */
export const HOW_OBTAINED = [
  "Da questa sessione huggingface.co, arxiv.org, deepseek.com e kimi.ai non erano raggiungibili: nessun config.json e nessun model.safetensors.index.json è stato letto direttamente.",
  "I config sono stati ricostruiti campo per campo da quattro fonti indipendenti e leggibili: il codice ufficiale DeepInfer (parser del config.json di V4.1 e mappa dei nomi dei tensori), le classi di config di NVIDIA NeMo AutoModel (dichiarano di riprodurre il rilascio), gli estratti del config.json ufficiale restituiti dal motore di ricerca e le ricette vLLM.",
  "I conti tornano con i totali ufficiali: per DeepSeek-V4.1-Flash 96.085 tensori su 96.085, 510,3 GB, 551,88B di backbone (552B dichiarati), 16,13B e 7,89B attivi (16B e 8B); per Kimi K3 2.779,5B e 104,2B (Table 1); per gpt-oss-120b 116,83B e 5,13B (model card).",
  "Una verifica avversariale indipendente (≈ 500 controlli) ha corretto 1 errore numerico, 1 errore di stima e diverse attribuzioni troppo forti; ciò che resta incerto è marcato «da verificare».",
];

/** I nomi chiesti dallo studente esistono? */
export const NAME_CHECK = [
  { asked: "DeepSeek V4.1 Flash", exists: true, what: "DeepSeek-V4.1-Flash (anteprima API dall'8-9/09, annuncio ufficiale 10/09/2026, checkpoint 0909) e DeepSeek-V4.1-Flash-1017 (17/10/2026), pesi MIT." },
  { asked: "DeepSeek V4.1 Pro", exists: false, what: "Non rilasciato all'8/11/2026: solo indiscrezioni e test. L'open «Pro» più vicino è DeepSeek-V4-Pro(-0813)." },
  { asked: "Kimi K3.1 Pro", exists: true, what: "API dal 23/10/2026, pesi moonshotai/Kimi-K3.1-Pro tra il 02 e il 04/11/2026 (fonti discordi), ≈ 1,56 TB in MXFP4." },
  { asked: "Kimi K3", exists: true, what: "API dal 16/07/2026, pesi dal 26-27/07/2026: 2,8T totali, 104B attivi, report tecnico completo su GitHub." },
  { asked: "DeepSeek V4", exists: true, what: "DeepSeek-V4-Pro (1,6T / 49B) e DeepSeek-V4-Flash (284B / 13B), 24/04/2026." },
];

/**
 * Panorama degli open-weight di frontiera (novembre 2026). `check`: P = fonte primaria, R = ricetta
 * o documentazione vLLM, S = solo stampa; `verified: false` = marcato "da verificare".
 */
export const LANDSCAPE = [
  { family: "DeepSeek", model: "DeepSeek-V4.1-Flash-1017", date: "17/10/2026", size: "552B + 196B Engram / 8B-16B", arch: "MoE 384e, encoder-decoder causale, CSA2 + SWA, Engram, mHC, DSpark, visione", license: "MIT", check: "P + R", verified: true },
  { family: "Moonshot", model: "Kimi K3.1 Pro", date: "pesi 02-04/11/2026", size: "2,8T / 104B", arch: "MoE 896e top-16 (LatentMoE), KDA + Gated MLA con DSA, AttnRes, MXFP4", license: "Kimi K3.1-Pro (MIT modificata)", check: "P (K3) + R", verified: true },
  { family: "Qwen", model: "Qwen3.8-2.4T-A95B", date: "10-26/08/2026", size: "2,4T / 95B", arch: "MoE 512e, attention lineare su 69 dei 92 layer, MTP, 262K → 1M", license: "personalizzata (27B: Apache 2.0)", check: "R + S", verified: true },
  { family: "Z.ai GLM", model: "GLM-5.3", date: "08/2026", size: "743B / 39B", arch: "MoE con sparse attention DSA (dedotta), MTP, 1M", license: "da verificare", check: "R", verified: true },
  { family: "MiniMax", model: "MiniMax-M3", date: "06/2026", size: "427B / 26B", arch: "MoE + MiniMax Sparse Attention, multimodale, 1M", license: "minimax-community (stampa)", check: "R", verified: true },
  { family: "OpenAI", model: "gpt-oss-120b / -20b", date: "05/08/2025", size: "116,8B / 5,1B", arch: "MoE 128e top-4, GQA, finestra 128 alternata, sink, MXFP4", license: "Apache 2.0", check: "P", verified: true },
  { family: "Meta", model: "Llama 4 Scout / Maverick", date: "05/04/2025", size: "109B / 17B (Scout)", arch: "MoE multimodale", license: "Llama 4 Community", check: "R", verified: true },
  { family: "Mistral", model: "Mistral Large 4", date: "pesi 29/10/2026", size: "~1T / 52B (dal nome del repo)", arch: "MoE", license: "fonti discordi", check: "S", verified: false },
  { family: "NVIDIA", model: "Nemotron 3 Ultra", date: "04/06/2026", size: "~550B / ~55B", arch: "MoE (famiglia ibrida Mamba-Transformer; per Ultra non confermato)", license: "licenza aperta NVIDIA (stampa)", check: "S", verified: false },
  { family: "Xiaomi", model: "MiMo-V2.5-Pro", date: "27/04/2026", size: "1,02T / 42B", arch: "MoE, attention ibrida, FP8, MTP", license: "da verificare", check: "R", verified: true },
  { family: "Meituan", model: "LongCat-2.0", date: "30/06-05/07/2026", size: "1,6T / ~33-56B dinamici", arch: "MoE ad attivazione dinamica, LongCat Sparse Attention", license: "MIT (stampa)", check: "S", verified: false },
  { family: "StepFun", model: "Step-5-Preview", date: "15/10/2026", size: "604B / non indicato", arch: "MoE multimodale, sparse/SWA, MTP, 1M", license: "da verificare", check: "R", verified: true },
  { family: "Tencent", model: "Hy3", date: "06/07/2026", size: "295B / 21B", arch: "MoE 192 esperti top-8 + 1 condiviso, 80 layer, MTP, 256K", license: "Apache 2.0 (stampa)", check: "R", verified: true },
  { family: "Baidu", model: "ERNIE 5.5 Flash Base", date: "inizio 11/2026", size: "280B / 14B", arch: "MoE, 1M, solo base", license: "restrittiva (stampa)", check: "S", verified: false },
  { family: "inclusionAI", model: "Ling-3.0-flash", date: "09/2026", size: "124B / 5,5B", arch: "MoE 512 esperti (8 attivi), attention ibrida MLA/KDA, MTP", license: "da verificare", check: "R", verified: true },
];

/**
 * Come il training produce i pesi e come emergono le capacità (ricerca §7 e capabilities_facts).
 * Ogni fatto ha la sua fonte; `caution` è la cautela da dire insieme al fatto; `links` sono le altre
 * fonti citate nel testo o nella cautela (es. la critica di Schaeffer et al. alle abilità emergenti).
 */
export const CAPABILITY_FACTS = [
  { id: "scaling", title: "Leggi di scala", text: "La loss scende come una legge di potenza in parametri N, dati D e calcolo C (esponenti ≈ 0,076, 0,095, 0,050). A calcolo fissato conviene crescere N e D insieme: circa 20 token per parametro, e Chinchilla 70B su 1,4T token batte Gopher 280B.", source: "Kaplan et al. 2020; Hoffmann et al. 2022", url: "https://arxiv.org/abs/2203.15556", links: [{ title: "Kaplan et al. 2020, Scaling Laws for Neural Language Models", url: "https://arxiv.org/abs/2001.08361" }], caution: "Oggi si va molto oltre: DeepSeek-V4.1-Flash ha circa 80 token per parametro totale." },
  { id: "emergent", title: "Abilità «emergenti»", text: "Un'abilità si dice emergente se manca nei modelli piccoli, c'è nei grandi e non si prevede estrapolando i piccoli.", source: "Wei et al. 2022 (TMLR)", url: "https://arxiv.org/abs/2206.07682", links: [{ title: "Schaeffer, Miranda, Koyejo 2023, Are Emergent Abilities of LLMs a Mirage?", url: "https://arxiv.org/abs/2304.15004" }], caution: "Critica di Schaeffer, Miranda e Koyejo (NeurIPS 2023): gran parte dei salti dipende da metriche discontinue (risposta esatta, scelta multipla); con metriche continue la curva è liscia, e oltre il 92% delle emergenze di BIG-Bench usa due sole metriche." },
  { id: "induction", title: "Induction head e in-context learning", text: "Nei transformer con almeno 2 layer, presto nel training, compare una «phase change» (una gobba nella loss): si formano le induction head e arriva gran parte dell'apprendimento dal contesto.", source: "Olsson et al. 2022", url: "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/", caution: "Osservato su modelli piccoli e medi studiati a fondo: nessuna delle fonti consultate lo misura su V4.1." },
  { id: "grokking", title: "Grokking", text: "Su piccoli dataset algoritmici la generalizzazione può passare dal caso al 100% molto dopo l'overfitting; con meno dati serve più ottimizzazione.", source: "Power et al. 2022", url: "https://arxiv.org/abs/2201.02177", caution: "È un fenomeno di dataset piccoli e sintetici: non dice che un LLM «capisce all'improvviso»." },
  { id: "rl-subnet", title: "L'RL cambia una sottorete", text: "L'RL aggiorna solo il 5-30% dei parametri; le sottoreti ottenute con seed, dati e algoritmi diversi (7 algoritmi, tra cui PPO, GRPO, DPO) si sovrappongono molto più del caso, su 10 LLM di famiglie diverse.", source: "Mukherjee, Yuan, Hakkani-Tür, Peng, NeurIPS 2025", url: "https://arxiv.org/abs/2505.11711", caution: "Le sottoreti non sono identiche: si sovrappongono più del caso." },
  { id: "rl-scaling", title: "Più calcolo di RL, più passi agentici", text: "In Kimi K3 aumentare i FLOP di RL migliora in modo consistente i punteggi e il numero medio di passi agentici; gli esperti per dominio ed effort vengono poi fusi con distillazione on-policy.", source: "Kimi K3 Technical Report, Fig. 8 ed eq. 15", url: "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf", caution: "Risultato di un solo laboratorio sul proprio modello." },
  { id: "qat", title: "Precisione decisa nel training", text: "Kimi K3 e la famiglia DeepSeek V4 addestrano con QAT: la stessa griglia FP4 vale in training e in inferenza. MXFP4 = blocchi di 32 valori E2M1 con una scala UE8M0 condivisa, cioè 4,25 bit per valore.", source: "Kimi K3 Technical Report §4.1.4; ricerca §7", url: "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf", caution: "Per V4.1 il QAT del KV in FP4 non è confermato." },
  { id: "safetensors", title: "Il formato dei pesi", text: "Un file safetensors è: 8 byte con la lunghezza N dell'intestazione, N byte di JSON {nome: {dtype, shape, data_offsets}}, poi i numeri grezzi. Niente codice eseguibile (a differenza di pickle) e caricamento zero-copy.", source: "huggingface/safetensors (README)", url: "https://github.com/huggingface/safetensors", caution: "" },
];

/** Grandezze usate dai grafici della vista completa (con la loro fonte). */
export const SCALING_POINTS = [
  { id: "chinchilla", label: "Chinchilla (ottimo)", tokensPerParam: 20, source: "Hoffmann et al. 2022" },
  { id: "v41-total", label: "V4.1-Flash, per parametro totale", tokensPerParam: 45e12 / 552e9, source: "45T token / 552B" },
  { id: "v41-active", label: "V4.1-Flash, per parametro attivo", tokensPerParam: 45e12 / 16e9, source: "45T token / 16B" },
];
