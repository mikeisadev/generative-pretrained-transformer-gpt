/**
 * Contenuti didattici del tab "Immagini e video": percorso (menu a fisarmonica) e fonti.
 *
 * Ogni sottopunto può indicare:
 *   focus     id del registro su cui volare (pipeline.js / videoScene.js)
 *   code      chiave di SNIPPETS (codice vero, a commit fissati) + highlight
 *   denoiser  "unet" | "dit" per cambiare il denoiser mostrato
 *   math      formule per il riquadro "Approfondimento" (KaTeX)
 * I numeri citati sono verificati sulle fonti elencate in SOURCES.
 */

const note = (html) => `<div class="diff-note">${html}</div>`;
const sim = (html) => `<div class="diff-sim">${html}</div>`;

export const PATH = [
  {
    id: "idea",
    title: "1 · L'idea: dal rumore all'immagine",
    children: [
      {
        id: "idea.cosa",
        title: "Che cos'è un modello di diffusione",
        text: `<p>Immagina una goccia d'inchiostro in un bicchiere d'acqua: piano piano si <strong>diffonde</strong> finché l'acqua è tutta grigia. Il nome viene da lì (dalla fisica dei processi di diffusione).</p>
<p>Un modello di diffusione impara a fare il film <strong>al contrario</strong>:</p>
<ul><li>durante l'addestramento si prende un'immagine e la si copre di rumore, un po' alla volta, fino a renderla irriconoscibile;</li>
<li>la rete impara a togliere <em>un po'</em> di rumore alla volta;</li>
<li>per generare si parte da puro rumore e si ripete il "togli un po' di rumore" decine di volte: alla fine resta un'immagine nuova.</li></ul>
<p>È come un restauratore che, passata dopo passata, fa emergere un dipinto da una tela sporca: solo che qui il dipinto sotto non c'era, lo inventa guidato dal testo.</p>
${note(`<strong>"Diffusione lineare"?</strong> Il nome corretto è <em>modelli di diffusione</em> (diffusion models). La parola "lineare" però non è del tutto fuori strada: nel paper DDPM la quantità di rumore aggiunta a ogni passo (β) cresce <em>in modo lineare</em>, e l'immagine rumorosa è una <em>combinazione lineare</em> di immagine e rumore. Sono dettagli del metodo, non il suo nome.`)}`,
      },
      {
        id: "idea.testo",
        title: "Testo token per token, immagini tutte insieme",
        focus: "unet",
        code: "sd_loop",
        codeTitle: "diffusers: il ciclo di denoising",
        highlight: ["for i, t in enumerate(timesteps)", "self.unet("],
        text: `<p>Il nostro GPT genera il testo <strong>un token alla volta</strong>, da sinistra a destra: ogni nuovo carattere richiede un passaggio nel modello e, una volta scritto, non cambia più.</p>
<p>Un modello di diffusione lavora su <strong>tutta l'immagine in parallelo</strong>: a ogni passo ritocca tutti i pixel (anzi, tutti i valori del latente) insieme. Non c'è un "primo pixel".</p>
<ul><li><strong>GPT</strong>: tanti passaggi quanti sono i token da generare.</li>
<li><strong>Diffusione</strong>: tanti passaggi quanti sono i passi del campionatore (es. 50), indipendentemente da cosa c'è nell'immagine. Con la classifier-free guidance ogni passo chiama la rete due volte.</li></ul>
<p>Analogia: GPT scrive una frase parola per parola; la diffusione mette a fuoco una foto sfocata, tutta insieme, un giro di ghiera alla volta.</p>`,
      },
      {
        id: "idea.transformer",
        title: "Dove sta il transformer",
        focus: "textEncoder",
        margin: 1.6,
        code: "sd_config",
        codeTitle: "Stable Diffusion 1.x: v1-inference.yaml",
        highlight: ["use_spatial_transformer", "context_dim", "FrozenCLIPEmbedder"],
        text: `<p>Il transformer della lezione compare in più punti:</p>
<ul><li><strong>Text encoder</strong>: in Stable Diffusion 1.x è il text encoder di CLIP ViT-L/14, un transformer di 12 layer largo 768 che legge al massimo 77 token.</li>
<li><strong>Dentro la U-Net</strong>: i blocchi di attention sono piccoli transformer (self-attention, cross-attention verso il testo, MLP).</li>
<li><strong>Diffusion Transformer (DiT)</strong>: la U-Net viene sostituita del tutto da una pila di blocchi transformer che lavora su "patch" del latente. È la strada dei modelli più recenti (SD3, molti modelli video).</li></ul>`,
      },
    ],
  },
  {
    id: "forward",
    title: "2 · Il processo diretto: aggiungere rumore",
    children: [
      {
        id: "forward.formula",
        title: "La formula del rumore",
        focus: "latent",
        margin: 2.2,
        code: "ddpm_qsample",
        codeTitle: "DDPM: q_sample",
        highlight: ["sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"],
        caption: "Avvia l'azione 'Processo diretto' per vedere la formula applicata a un'immagine.",
        text: `<p>Dopo t passi di rumore l'immagine diventa:</p>
<p style="text-align:center"><strong>x<sub>t</sub> = √ᾱ<sub>t</sub> · x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) · ε</strong></p>
<ul><li><strong>x<sub>0</sub></strong> è l'immagine pulita, <strong>ε</strong> è rumore gaussiano (media 0, varianza 1);</li>
<li><strong>√ᾱ<sub>t</sub></strong> è quanto resta dell'immagine, <strong>√(1 − ᾱ<sub>t</sub>)</strong> quanto pesa il rumore;</li>
<li>con t piccolo domina l'immagine, con t vicino a T domina il rumore.</li></ul>
<p>Il trucco: non serve applicare il rumore t volte di fila, la formula <strong>salta direttamente</strong> a qualsiasi t. Durante il training è comodissimo.</p>`,
        math: [
          { tex: "q(x_t \\mid x_{t-1}) = \\mathcal{N}\\big(x_t;\\ \\sqrt{1-\\beta_t}\\,x_{t-1},\\ \\beta_t I\\big)", note: "Un singolo passo: si riduce un po' il segnale e si aggiunge rumore di varianza β_t." },
          { tex: "\\alpha_t = 1-\\beta_t,\\qquad \\bar\\alpha_t = \\prod_{s=1}^{t}\\alpha_s", note: "ᾱ_t riassume tutti i passi fino a t." },
          { tex: "x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon,\\qquad \\varepsilon \\sim \\mathcal{N}(0, I)", note: "Forma chiusa: da x_0 a x_t in un colpo solo (Ho et al. 2020)." },
        ],
      },
      {
        id: "forward.schedule",
        title: "La schedule di β (T = 1000)",
        focus: "timestep",
        margin: 3,
        code: "ddpm_beta",
        codeTitle: "DDPM: get_beta_schedule",
        highlight: ["linear"],
        text: `<p>Quanto rumore aggiungere a ogni passo è deciso in anticipo dalla <strong>schedule</strong>:</p>
<ul><li><strong>DDPM</strong> (Ho et al. 2020): T = 1000 passi, β cresce in linea retta da 0,0001 a 0,02.</li>
<li><strong>Stable Diffusion 1.x</strong>: T = 1000, schedule "scaled linear" da 0,00085 a 0,012 (in linea retta sono le radici di β).</li></ul>
<p>Apri il pannello <strong>Grafici</strong>: vedrai β<sub>t</sub>, ᾱ<sub>t</sub> e il rapporto segnale/rumore SNR(t) in scala logaritmica per entrambe.</p>
<p>La rete riceve anche <strong>t</strong>, trasformato in un vettore con seni e coseni (lo stesso trucco delle positional encoding del transformer): così sa quanto rumore aspettarsi.</p>`,
        math: [
          { tex: "\\beta_t = \\beta_1 + \\frac{t-1}{T-1}(\\beta_T-\\beta_1),\\quad \\beta_1 = 10^{-4},\\ \\beta_T = 0{,}02", note: "Schedule lineare di DDPM." },
          { tex: "\\mathrm{SNR}(t) = \\frac{\\bar\\alpha_t}{1-\\bar\\alpha_t}", note: "Rapporto segnale/rumore: utile per confrontare schedule diverse." },
        ],
      },
      {
        id: "forward.training",
        title: "Come si addestra la rete",
        focus: "unet",
        code: "ddpm_config",
        codeTitle: "DDPM: iperparametri (CIFAR-10)",
        highlight: ["num_diffusion_timesteps", "model_mean_type='eps'", "loss_type='mse'"],
        text: `<p>Il training è sorprendentemente semplice. A ogni esempio:</p>
<ol><li>prendi un'immagine x<sub>0</sub> (in Stable Diffusion: il suo latente);</li>
<li>scegli un passo t a caso e un rumore ε a caso;</li>
<li>calcola x<sub>t</sub> con la formula;</li>
<li>chiedi alla rete di <strong>indovinare ε</strong> guardando x<sub>t</sub> e t;</li>
<li>la loss è l'errore quadratico medio tra ε vero e ε previsto.</li></ol>
<p>È lo stesso schema del nostro GPT (previsione + loss + discesa del gradiente), ma la "risposta giusta" è il rumore invece del token successivo.</p>`,
        math: [{ tex: "L_{\\text{simple}} = \\mathbb{E}_{t,\\,x_0,\\,\\varepsilon}\\Big[\\big\\lVert \\varepsilon - \\varepsilon_\\theta\\big(\\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon,\\ t\\big)\\big\\rVert^2\\Big]", note: "La loss \"semplificata\" di DDPM: predire il rumore." }],
      },
    ],
  },
  {
    id: "reverse",
    title: "3 · Il processo inverso: generare",
    children: [
      {
        id: "reverse.step",
        title: "Un passo di denoising (DDIM)",
        focus: "scheduler",
        margin: 2.2,
        code: "ddim_step",
        codeTitle: "diffusers: passo DDIM",
        highlight: ["pred_original_sample = (sample", "prev_sample ="],
        text: `<p>Per generare si parte da rumore puro e a ogni passo:</p>
<ol><li>la rete predice il rumore ε̂ contenuto in x<sub>t</sub>;</li>
<li>da ε̂ si stima l'immagine pulita x̂<sub>0</sub> (invertendo la formula);</li>
<li>lo <strong>scheduler</strong> calcola x<sub>t−1</sub>: un po' meno rumoroso.</li></ol>
<p><strong>DDPM</strong> campiona con tutti i 1000 passi e aggiunge un po' di rumore nuovo a ogni passo. <strong>DDIM</strong> (Song, Meng, Ermon 2020) usa lo stesso modello addestrato ma può saltare passi ed essere deterministico (η = 0): bastano poche decine di passi. Nella pipeline di Stable Diffusion di diffusers il default è 50.</p>`,
        math: [
          { tex: "\\hat x_0 = \\frac{x_t - \\sqrt{1-\\bar\\alpha_t}\\,\\hat\\varepsilon}{\\sqrt{\\bar\\alpha_t}}", note: "Stima dell'immagine pulita dal rumore previsto." },
          { tex: "x_{t-1} = \\sqrt{\\bar\\alpha_{t-1}}\\,\\hat x_0 + \\sqrt{1-\\bar\\alpha_{t-1}}\\,\\hat\\varepsilon", note: "Passo DDIM deterministico (η = 0)." },
        ],
      },
      {
        id: "reverse.cfg",
        title: "Classifier-free guidance",
        focus: "eps",
        margin: 2.4,
        code: "sd_loop",
        codeTitle: "diffusers: classifier-free guidance",
        highlight: ["noise_pred_uncond, noise_pred_text", "guidance_scale * (noise_pred_text - noise_pred_uncond)"],
        text: `<p>Per seguire meglio il prompt, a ogni passo la rete viene chiamata <strong>due volte</strong>: una con il testo (ε<sub>c</sub>) e una con un prompt vuoto (ε<sub>u</sub>). Poi si "spinge" nella direzione del testo:</p>
<p style="text-align:center"><strong>ε̂ = ε<sub>u</sub> + s · (ε<sub>c</sub> − ε<sub>u</sub>)</strong></p>
<ul><li>s = 1: solo la previsione con il testo;</li>
<li>s &gt; 1: si esagera la differenza, l'immagine aderisce di più al prompt (lo script di Stable Diffusion 1.x usa 7,5 di default);</li>
<li>s troppo alto: immagini sature e innaturali (lo osservano Saharia et al. nel paper di Imagen).</li></ul>
<p>Perché funzioni, durante il training il testo viene tolto in una parte degli esempi (per Stable Diffusion 1.x il 10%): la stessa rete impara anche la versione "senza prompt". È l'idea di Ho e Salimans (2022).</p>
${sim("Nell'azione 'Generazione' puoi cambiare s dal pannello Azioni e guardare il grafico 'Effetto della guidance scale'.")}`,
        math: [{ tex: "\\hat\\varepsilon = \\varepsilon_\\theta(x_t, \\varnothing) + s\\,\\big(\\varepsilon_\\theta(x_t, c) - \\varepsilon_\\theta(x_t, \\varnothing)\\big)", note: "Convenzione di Stable Diffusion/diffusers. Nel paper di Ho e Salimans la stessa regola è scritta con w = s − 1." }],
      },
      {
        id: "reverse.flow",
        title: "Rectified flow (Stable Diffusion 3)",
        focus: "scheduler",
        margin: 2.2,
        code: "flow_step",
        codeTitle: "diffusers: FlowMatchEulerDiscreteScheduler.step",
        highlight: ["dt = sigma_next - sigma", "prev_sample = sample + dt * model_output"],
        text: `<p>Stable Diffusion 3 (Esser et al. 2024) usa una variante chiamata <strong>rectified flow</strong>: immagine e rumore sono uniti in linea retta</p>
<p style="text-align:center"><strong>x<sub>t</sub> = (1 − t) · x<sub>0</sub> + t · ε</strong>, con t tra 0 e 1,</p>
<p>e la rete predice la <strong>velocità</strong> lungo questa retta. Per generare si fanno piccoli passi di Eulero dal rumore verso l'immagine. Il paper mostra anche che conviene allenare la rete più spesso sui livelli di rumore intermedi, quelli che contano di più per l'aspetto dell'immagine.</p>`,
        math: [
          { tex: "x_t = (1-t)\\,x_0 + t\\,\\varepsilon,\\qquad v = \\frac{dx_t}{dt} = \\varepsilon - x_0", note: "Il percorso è una retta: la velocità è costante." },
          { tex: "x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i)\\,v_\\theta(x_{t_i}, t_i)", note: "Passo di Eulero (nel codice: prev_sample = sample + dt * model_output)." },
        ],
      },
    ],
  },
  {
    id: "arch",
    title: "4 · L'architettura text-to-image (Stable Diffusion)",
    children: [
      {
        id: "arch.text",
        title: "Tokenizer e text encoder",
        focus: "textEncoder",
        margin: 1.4,
        code: "sd_encode",
        codeTitle: "diffusers: encode_prompt",
        highlight: ["self.tokenizer(", "max_length=self.tokenizer.model_max_length", "self.text_encoder("],
        text: `<p>Il prompt viene spezzato in token dal tokenizer BPE di CLIP (come il nostro GPT, ma con un vocabolario di pezzi di parola invece che di caratteri) e riempito fino a <strong>77 token</strong>.</p>
<p>Il <strong>text encoder</strong> di CLIP ViT-L/14 (12 layer, larghezza 768, circa 123 milioni di parametri) produce un vettore da 768 numeri per ogni token: una matrice 77 × 768. Il text encoder è <strong>congelato</strong>: non viene addestrato insieme alla U-Net.</p>
<p>La suddivisione in token mostrata nella scena è indicativa: il tokenizer vero spezzerebbe le parole italiane in frammenti.</p>`,
      },
      {
        id: "arch.latent",
        title: "Il latente e il VAE",
        focus: "vae",
        margin: 1.8,
        code: "sd_config",
        codeTitle: "Stable Diffusion 1.x: configurazione dell'autoencoder",
        highlight: ["z_channels: 4", "ch_mult", "scale_factor"],
        text: `<p>Lavorare direttamente sui pixel è costoso. L'idea della <strong>latent diffusion</strong> (Rombach et al. 2022) è comprimere prima l'immagine con un autoencoder (un VAE):</p>
<ul><li>fattore di riduzione <strong>8</strong>: 512×512 diventa 64×64;</li>
<li><strong>4 canali</strong>: il latente è 4 × 64 × 64 = 16.384 numeri, contro 786.432 dell'immagine;</li>
<li>la diffusione avviene tutta nel latente; solo alla fine il <strong>decoder</strong> del VAE ricostruisce i pixel.</li></ul>
<p>Il latente viene moltiplicato per 0,18215 (scale_factor) per avere varianza vicina a 1, e diviso per lo stesso numero prima del decoder.</p>`,
      },
      {
        id: "arch.unet",
        title: "La U-Net: encoder, decoder, skip connection",
        denoiser: "unet",
        focus: "unet",
        margin: 1.1,
        code: "sd_config",
        codeTitle: "Stable Diffusion 1.x: configurazione della U-Net",
        highlight: ["model_channels: 320", "channel_mult", "num_res_blocks: 2", "attention_resolutions"],
        caption: "Arancione: ResNet block · viola: attention · verde acqua: down/upsample · linee azzurre: skip connection",
        text: `<p>La <strong>U-Net</strong> (circa 860 milioni di parametri in SD 1.x) ha la forma di una U:</p>
<ul><li><strong>encoder</strong>: la risoluzione scende 64 → 32 → 16 → 8 e i canali salgono 320 → 640 → 1280 → 1280;</li>
<li><strong>blocco centrale</strong> a 8×8;</li>
<li><strong>decoder</strong>: risale fino a 64×64;</li>
<li><strong>skip connection</strong>: ogni livello del decoder riceve le feature del livello corrispondente dell'encoder, così i dettagli fini non si perdono.</li></ul>
<p>Ogni livello ha 2 <strong>ResNet block</strong> (3 nel decoder) e, nei tre livelli più grandi, un <strong>blocco di attention</strong> dopo ogni ResNet.</p>
${sim("Zoom: avvicinati per vedere ogni blocco, allontanati per lo schema. L'azione 'Un passo dentro il denoiser' li illumina uno per uno.")}`,
      },
      {
        id: "arch.cross",
        title: "Cross-attention: come entra il testo",
        denoiser: "unet",
        focus: "unet",
        code: "sd_config",
        codeTitle: "context_dim: la dimensione degli embedding del testo",
        highlight: ["context_dim: 768", "conditioning_key: crossattn", "transformer_depth: 1"],
        text: `<p>Nei blocchi di attention della U-Net c'è la <strong>cross-attention</strong>: è l'attention della lezione, ma con query e key/value da due sorgenti diverse.</p>
<ul><li><strong>query</strong>: dalle posizioni dell'immagine (il latente);</li>
<li><strong>key e value</strong>: dai 77 vettori del testo.</li></ul>
<p>Ogni zona dell'immagine "chiede" al testo quali parole la riguardano: la zona del cielo può dare peso a "tramonto", quella della torre a "faro". Le linee viola nella scena vanno dal testo a ogni blocco di attention.</p>`,
        math: [{ tex: "\\mathrm{Attention}(Q,K,V) = \\mathrm{softmax}\\!\\Big(\\frac{QK^\\top}{\\sqrt{d}}\\Big)V,\\quad Q = W_Q\\,\\varphi(z_t),\\ K = W_K\\,\\tau(y),\\ V = W_V\\,\\tau(y)", note: "φ(z_t): feature dell'immagine dentro la U-Net; τ(y): embedding del testo (Rombach et al. 2022)." }],
      },
      {
        id: "arch.timestep",
        title: "Il passo t dentro la rete",
        denoiser: "unet",
        focus: "timestep",
        margin: 3,
        code: "ddpm_resnet",
        codeTitle: "DDPM: ResNet block con il passo t",
        highlight: ["temb_proj", "add in timestep embedding"],
        text: `<p>La stessa rete deve funzionare con poco rumore (t piccolo) e con tantissimo (t vicino a 1000). Per questo riceve il passo t:</p>
<ul><li>t diventa un vettore con <strong>seni e coseni</strong> a frequenze diverse (come le positional encoding del transformer originale);</li>
<li>un piccolo MLP lo trasforma;</li>
<li>il risultato viene <strong>sommato dentro ogni ResNet block</strong> (le linee verdi nella scena).</li></ul>`,
        math: [{ tex: "\\mathrm{emb}(t) = \\big[\\cos(t\\,\\omega_0),\\dots,\\cos(t\\,\\omega_{d/2-1}),\\ \\sin(t\\,\\omega_0),\\dots\\big],\\quad \\omega_i = 10000^{-2i/d}", note: "Embedding sinusoidale del passo (come nel codice di DiT e GLIDE)." }],
      },
    ],
  },
  {
    id: "dit",
    title: "5 · Diffusion Transformer: il legame con la lezione",
    children: [
      {
        id: "dit.patch",
        title: "Patch come token",
        denoiser: "dit",
        focus: "dit",
        margin: 1.1,
        code: "dit_forward",
        codeTitle: "DiT: forward",
        highlight: ["x_embedder", "for block in self.blocks", "unpatchify"],
        text: `<p>Peebles e Xie (2023) hanno sostituito la U-Net con un <strong>transformer</strong>: il <strong>Diffusion Transformer (DiT)</strong>.</p>
<ul><li>il latente viene tagliato in <strong>patch</strong> (DiT-XL/2: patch 2×2);</li>
<li>ogni patch diventa un <strong>token</strong> (un vettore) con la sua positional embedding: con un latente 32×32 sono 256 token;</li>
<li>i token attraversano una pila di blocchi transformer (DiT-XL/2: 28 blocchi, larghezza 1152, 16 head);</li>
<li>alla fine ogni token torna patch e si ricompone il latente (<em>unpatchify</em>).</li></ul>
<p>Il paper mostra che più calcolo (più profondità, larghezza o token) dà immagini migliori in modo regolare: la stessa "legge di scala" dei transformer per il testo.</p>`,
      },
      {
        id: "dit.adaln",
        title: "adaLN: il passo t senza cross-attention",
        denoiser: "dit",
        focus: "dit",
        code: "dit_block",
        codeTitle: "DiT: blocco con adaLN-Zero",
        highlight: ["adaLN_modulation", "modulate(", "gate_msa"],
        text: `<p>Ogni blocco DiT è quello del nostro GPT (LayerNorm → attention → somma residua → LayerNorm → MLP → somma residua), con due differenze:</p>
<ul><li><strong>niente maschera causale</strong>: ogni patch vede tutte le altre, non solo "quelle prima";</li>
<li>il condizionamento (passo t + classe) entra con <strong>adaLN-Zero</strong>: un MLP calcola scale, shift e gate che modulano le LayerNorm e i rami residui. Il gate parte da zero, così all'inizio ogni blocco è l'identità.</li></ul>`,
        math: [{ tex: "x \\leftarrow x + \\alpha_1 \\odot \\mathrm{Attn}\\big(\\mathrm{LN}(x)\\,(1+\\gamma_1) + \\beta_1\\big),\\quad (\\gamma_i,\\beta_i,\\alpha_i) = \\mathrm{MLP}(c)", note: "c = embedding(t) + embedding(classe). Lo stesso per l'MLP con γ₂, β₂, α₂." }],
      },
      {
        id: "dit.sd3",
        title: "SD3 e MMDiT: testo e immagine nella stessa attention",
        denoiser: "dit",
        focus: "dit",
        code: "sd3_forward",
        codeTitle: "diffusers: SD3Transformer2DModel",
        highlight: ["time_text_embed", "context_embedder", "block("],
        text: `<p>Stable Diffusion 3 (Esser et al. 2024) usa un'architettura chiamata <strong>MMDiT</strong>:</p>
<ul><li>i token del testo e quelli dell'immagine hanno <strong>pesi separati</strong> ma vengono uniti nella <strong>stessa attention</strong>: l'informazione scorre in entrambe le direzioni;</li>
<li>il passo t e un riassunto del testo ("pooled") entrano con adaLN;</li>
<li>i testi sono codificati da tre encoder: due CLIP e un T5;</li>
<li>nel codice di diffusers il transformer di SD3 riceve latenti a 16 canali, con patch 2×2;</li>
<li>l'addestramento usa il rectified flow (punto 3).</li></ul>`,
      },
    ],
  },
  {
    id: "video",
    title: "6 · Video: tutti i frame insieme",
    children: [
      {
        id: "video.ipotesi",
        title: "Un transformer per ogni frame?",
        focus: "videoFrames",
        margin: 1.6,
        code: "temporal_attn",
        codeTitle: "diffusers: attention temporale",
        highlight: ["batch_size * height * width, num_frames"],
        text: `<p>Un'ipotesi naturale è generare un video come tante immagini, un frame alla volta. I modelli recenti <strong>non fanno così</strong>: generano tutti i frame <strong>insieme</strong>, come un unico blocco di dati.</p>
<p>Il motivo è la <strong>coerenza</strong>: se ogni frame nascesse dal suo rumore, il faro potrebbe spostarsi, cambiare colore o forma da un frame all'altro. Raffinando tutti i frame insieme, il modello può "accordare" ogni frame con gli altri a ogni passo.</p>
${sim("Guarda il confronto nella scena e il grafico 'Differenza tra frame consecutivi' (simulazione didattica).")}`,
      },
      {
        id: "video.vdm",
        title: "Video Diffusion Models: U-Net 3D",
        focus: "videoDenoiser",
        margin: 2.4,
        code: "temporal_attn",
        codeTitle: "diffusers: TransformerTemporalModel",
        highlight: ["permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)"],
        text: `<p>Video Diffusion Models (Ho et al. 2022) estende la U-Net al video:</p>
<ul><li>le convoluzioni 2D diventano 3D ma solo nello spazio (kernel 1×3×3);</li>
<li><strong>attention spaziale</strong>: ogni frame guarda dentro sé stesso;</li>
<li>dopo ogni attention spaziale c'è un'<strong>attention temporale</strong>: ogni posizione guarda la stessa posizione negli altri frame;</li>
<li>spegnendo l'attention temporale, lo stesso modello si allena anche su immagini singole.</li></ul>
<p>Si allena su clip brevi (per esempio 16 frame) e poi si estende a video più lunghi.</p>`,
      },
      {
        id: "video.sora",
        title: "Sora e le patch spazio-temporali",
        focus: "videoLatent",
        margin: 1.5,
        code: "dit_forward",
        codeTitle: "DiT: patch → token (in 2D)",
        highlight: ["x_embedder"],
        text: `<p>Il report tecnico di Sora (OpenAI, febbraio 2024) descrive l'idea generale:</p>
<ul><li>una rete comprime il video in un <strong>latente</strong> più piccolo nello spazio e nel tempo;</li>
<li>il latente viene tagliato in <strong>patch spazio-temporali</strong> che fanno da token;</li>
<li>un <strong>diffusion transformer</strong> condizionato dal testo le raffina tutte insieme.</li></ul>
<p>I dettagli dell'architettura di Sora (dimensioni, numero di layer, tipo di attention) <strong>non sono pubblici</strong>.</p>`,
      },
      {
        id: "video.aperti",
        title: "Modelli aperti: VAE 3D causali",
        focus: "videoVae",
        margin: 2.6,
        code: "causal_conv3d",
        codeTitle: "diffusers: convoluzione 3D causale (HunyuanVideo)",
        highlight: ["time_causal_padding", "kernel_size[2] - 1"],
        text: `<p>Nei modelli aperti recenti i dettagli sono pubblici:</p>
<ul><li><strong>HunyuanVideo</strong> (Tencent): VAE 3D con convoluzioni causali, compressione 4× nel tempo, 8× nello spazio, 16 canali; transformer con "Full Attention" su testo e video.</li>
<li><strong>Wan 2.1</strong> (Alibaba): Wan-VAE, un VAE 3D causale; diffusion transformer addestrato con flow matching, testo (encoder T5) inserito con cross-attention in ogni blocco.</li>
<li><strong>CogVideoX</strong> (Zhipu AI / Tsinghua): VAE 3D causale e un "expert transformer".</li></ul>
<p><strong>Causale</strong> vuol dire che un frame latente dipende solo dai frame passati: nel codice la convoluzione 3D aggiunge padding solo "prima" nel tempo.</p>`,
      },
      {
        id: "video.attention",
        title: "Attention spaziale, temporale, completa",
        focus: "videoLatent",
        margin: 1.4,
        text: `<p>Con N token l'attention costa circa N² confronti. Un video ha moltissimi token (frame × altezza × larghezza), quindi ci sono due strade:</p>
<ul><li><strong>fattorizzata</strong>: attention spaziale dentro ogni frame + attention temporale tra frame (Video Diffusion Models). Costa meno;</li>
<li><strong>completa 3D</strong>: ogni token guarda tutti gli altri (HunyuanVideo). Più potente, più costosa.</li></ul>
<p>L'azione "Video" mostra le tre varianti con linee che partono da un token di esempio.</p>`,
        math: [{ tex: "\\underbrace{F\\,(HW)^2 + HW\\,F^2}_{\\text{fattorizzata}} \\ \\ll\\ \\underbrace{(F\\,HW)^2}_{\\text{completa}}", note: "F frame (latenti), H×W token per frame. Esempio: F = 4, HW = 16 → 1.088 contro 4.096 confronti." }],
      },
    ],
  },
];

export const SOURCES = [
  {
    title: "Paper fondamentali",
    items: [
      { title: "Denoising Diffusion Probabilistic Models (DDPM)", who: "Ho, Jain, Abbeel · NeurIPS 2020", url: "https://arxiv.org/abs/2006.11239", note: "Processo diretto in forma chiusa, schedule lineare di β (1e-4 → 0,02), T = 1000, predizione del rumore ε, loss semplificata." },
      { title: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", who: "Sohl-Dickstein et al. · ICML 2015", url: "https://arxiv.org/abs/1503.03585", note: "Il lavoro che introduce l'idea: distruggere la struttura con un processo di diffusione e imparare a ricostruirla." },
      { title: "Denoising Diffusion Implicit Models (DDIM)", who: "Song, Meng, Ermon · ICLR 2021", url: "https://arxiv.org/abs/2010.02502", note: "Campionamento deterministico (η = 0) con molti meno passi, stesso modello addestrato." },
      { title: "High-Resolution Image Synthesis with Latent Diffusion Models", who: "Rombach et al. · CVPR 2022", url: "https://arxiv.org/abs/2112.10752", note: "Diffusione nel latente di un autoencoder, U-Net con cross-attention verso il testo: la base di Stable Diffusion." },
      { title: "Classifier-Free Diffusion Guidance", who: "Ho, Salimans · 2022", url: "https://arxiv.org/abs/2207.12598", note: "Due previsioni (con e senza condizione) combinate con una scala; condizione tolta in parte del training." },
      { title: "Photorealistic Text-to-Image Diffusion Models (Imagen)", who: "Saharia et al. · NeurIPS 2022", url: "https://arxiv.org/abs/2205.11487", note: "Osserva che pesi di guidance alti producono immagini sature e innaturali." },
      { title: "Scalable Diffusion Models with Transformers (DiT)", who: "Peebles, Xie · ICCV 2023", url: "https://arxiv.org/abs/2212.09748", note: "Patch come token, blocchi transformer con adaLN-Zero, DiT-XL/2 (28 blocchi, 1152, 16 head)." },
      { title: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)", who: "Esser et al. · ICML 2024", url: "https://arxiv.org/abs/2403.03206", note: "Rectified flow, architettura MMDiT con pesi separati per testo e immagine e attention congiunta." },
      { title: "Learning Transferable Visual Models From Natural Language Supervision (CLIP)", who: "Radford et al. · 2021", url: "https://arxiv.org/abs/2103.00020", note: "Text encoder di ViT-L/14: 12 layer, larghezza 768, 12 head." },
    ],
  },
  {
    title: "Video",
    items: [
      { title: "Video Diffusion Models", who: "Ho et al. · NeurIPS 2022", url: "https://arxiv.org/abs/2204.03458", note: "U-Net 3D fattorizzata: attention spaziale seguita da attention temporale; training congiunto su immagini e video." },
      { title: "Video generation models as world simulators (report tecnico di Sora)", who: "OpenAI · febbraio 2024", url: "https://openai.com/index/video-generation-models-as-world-simulators/", note: "Rete di compressione video, patch spazio-temporali, diffusion transformer. Dettagli dell'architettura non pubblicati." },
      { title: "HunyuanVideo (codice e report)", who: "Tencent · 2024", url: "https://github.com/Tencent/HunyuanVideo", note: "VAE 3D con CausalConv3D: compressione 4 (tempo), 8 (spazio), 16 canali; Full Attention. Report: arXiv 2412.03603." },
      { title: "Wan 2.1 (codice e report)", who: "Alibaba · 2025", url: "https://github.com/Wan-Video/Wan2.1", note: "Wan-VAE 3D causale, diffusion transformer con flow matching, encoder T5 e cross-attention. Report: arXiv 2503.20314." },
      { title: "CogVideoX (codice)", who: "Zhipu AI / Tsinghua · 2024", url: "https://github.com/THUDM/CogVideo", note: "3D Causal VAE ed expert transformer. Paper: arXiv 2408.06072." },
    ],
  },
  {
    title: "Codice e configurazioni (pannello del codice)",
    items: [
      { title: "hojonathanho/diffusion", who: "codice ufficiale DDPM", url: "https://github.com/hojonathanho/diffusion", note: "q_sample, get_beta_schedule, ResNet block con il passo t, iperparametri del training." },
      { title: "CompVis/stable-diffusion: v1-inference.yaml", who: "configurazione ufficiale", url: "https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml", note: "U-Net (320 canali, channel_mult [1,2,4,4], attention a 3 livelli, context_dim 768), VAE (4 canali, fattore 8), schedule, scale_factor 0,18215." },
      { title: "CompVis/stable-diffusion: README", who: "model card", url: "https://github.com/CompVis/stable-diffusion", note: "U-Net da 860M, text encoder CLIP ViT-L/14 da 123M, guidance 7,5 di default, 10% di prompt tolti in training." },
      { title: "huggingface/diffusers", who: "libreria di riferimento", url: "https://github.com/huggingface/diffusers", note: "Pipeline di Stable Diffusion e SD3, scheduler DDPM/DDIM/flow matching, attention temporale, VAE di HunyuanVideo." },
      { title: "facebookresearch/DiT", who: "codice ufficiale DiT", url: "https://github.com/facebookresearch/DiT", note: "DiTBlock con adaLN-Zero, embedding sinusoidale del passo, configurazione DiT-XL/2." },
      { title: "openai/CLIP", who: "codice ufficiale CLIP", url: "https://github.com/openai/CLIP", note: "Lunghezza di contesto del tokenizer: 77 token." },
    ],
  },
];
