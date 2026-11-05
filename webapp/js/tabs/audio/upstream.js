/**
 * Frammenti di codice VERO dei progetti originali, copiati alla lettera (ramo main dei
 * repository GitHub, consultati a novembre 2026) con il numero della prima riga.
 * Generato da uno script: non modificare a mano i frammenti.
 */
export const PY_SNIPPETS = {
 "whisper.logmel": {
  "title": "Whisper: log-Mel dello spettrogramma",
  "file": "openai/whisper · whisper/audio.py",
  "start": 110,
  "code": "def log_mel_spectrogram(\n    audio: Union[str, np.ndarray, torch.Tensor],\n    n_mels: int = 80,\n    padding: int = 0,\n    device: Optional[Union[str, torch.device]] = None,\n):\n    \"\"\"\n    Compute the log-Mel spectrogram of\n\n    Parameters\n    ----------\n    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)\n        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz\n\n    n_mels: int\n        The number of Mel-frequency filters, only 80 and 128 are supported\n\n    padding: int\n        Number of zero samples to pad to the right\n\n    device: Optional[Union[str, torch.device]]\n        If given, the audio tensor is moved to this device before STFT\n\n    Returns\n    -------\n    torch.Tensor, shape = (n_mels, n_frames)\n        A Tensor that contains the Mel spectrogram\n    \"\"\"\n    if not torch.is_tensor(audio):\n        if isinstance(audio, str):\n            audio = load_audio(audio)\n        audio = torch.from_numpy(audio)\n\n    if device is not None:\n        audio = audio.to(device)\n    if padding > 0:\n        audio = F.pad(audio, (0, padding))\n    window = torch.hann_window(N_FFT).to(audio.device)\n    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)\n    magnitudes = stft[..., :-1].abs() ** 2\n\n    filters = mel_filters(audio.device, n_mels)\n    mel_spec = filters @ magnitudes\n\n    log_spec = torch.clamp(mel_spec, min=1e-10).log10()\n    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)\n    log_spec = (log_spec + 4.0) / 4.0\n    return log_spec",
  "language": "python"
 },
 "whisper.params": {
  "title": "Whisper: iperparametri audio",
  "file": "openai/whisper · whisper/audio.py",
  "start": 12,
  "code": "# hard-coded audio hyperparameters\nSAMPLE_RATE = 16000\nN_FFT = 400\nHOP_LENGTH = 160\nCHUNK_LENGTH = 30\nN_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk\nN_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input\n\nN_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2\nFRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame\nTOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token",
  "language": "python"
 },
 "whisper.encoder": {
  "title": "Whisper: AudioEncoder",
  "file": "openai/whisper · whisper/model.py",
  "start": 181,
  "code": "class AudioEncoder(nn.Module):\n    def __init__(\n        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int\n    ):\n        super().__init__()\n        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)\n        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)\n        self.register_buffer(\"positional_embedding\", sinusoids(n_ctx, n_state))\n\n        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(\n            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]\n        )\n        self.ln_post = LayerNorm(n_state)\n\n    def forward(self, x: Tensor):\n        \"\"\"\n        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)\n            the mel spectrogram of the audio\n        \"\"\"\n        x = F.gelu(self.conv1(x))\n        x = F.gelu(self.conv2(x))\n        x = x.permute(0, 2, 1)\n\n        assert x.shape[1:] == self.positional_embedding.shape, \"incorrect audio shape\"\n        x = (x + self.positional_embedding).to(x.dtype)\n\n        for block in self.blocks:\n            x = block(x)\n\n        x = self.ln_post(x)\n        return x",
  "language": "python"
 },
 "whisper.tokens": {
  "title": "Whisper: token speciali",
  "file": "openai/whisper · whisper/tokenizer.py",
  "start": 340,
  "code": "    specials = [\n        \"<|endoftext|>\",\n        \"<|startoftranscript|>\",\n        *[f\"<|{lang}|>\" for lang in list(LANGUAGES.keys())[:num_languages]],\n        \"<|translate|>\",\n        \"<|transcribe|>\",\n        \"<|startoflm|>\",\n        \"<|startofprev|>\",\n        \"<|nospeech|>\",\n        \"<|notimestamps|>\",\n        *[f\"<|{i * 0.02:.2f}|>\" for i in range(1501)],\n    ]",
  "language": "python"
 },
 "whisper.sot": {
  "title": "Whisper: sequenza iniziale del decoder",
  "file": "openai/whisper · whisper/tokenizer.py",
  "start": 147,
  "code": "        sot: int = self.special_tokens[\"<|startoftranscript|>\"]\n        translate: int = self.special_tokens[\"<|translate|>\"]\n        transcribe: int = self.special_tokens[\"<|transcribe|>\"]\n\n        langs = tuple(LANGUAGES.keys())[: self.num_languages]\n        sot_sequence = [sot]\n        if self.language is not None:\n            sot_sequence.append(sot + 1 + langs.index(self.language))\n        if self.task is not None:\n            task_token: int = transcribe if self.task == \"transcribe\" else translate\n            sot_sequence.append(task_token)\n\n        self.sot_sequence = tuple(sot_sequence)",
  "language": "python"
 },
 "encodec.rvq": {
  "title": "EnCodec: quantizzazione vettoriale residua",
  "file": "facebookresearch/encodec · encodec/quantization/core_vq.py",
  "start": 319,
  "code": "class ResidualVectorQuantization(nn.Module):\n    \"\"\"Residual vector quantization implementation.\n    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf\n    \"\"\"\n    def __init__(self, *, num_quantizers, **kwargs):\n        super().__init__()\n        self.layers = nn.ModuleList(\n            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]\n        )\n\n    def forward(self, x, n_q: tp.Optional[int] = None):\n        quantized_out = 0.0\n        residual = x\n\n        all_losses = []\n        all_indices = []\n\n        n_q = n_q or len(self.layers)\n\n        for layer in self.layers[:n_q]:\n            quantized, indices, loss = layer(residual)\n            residual = residual - quantized\n            quantized_out = quantized_out + quantized\n\n            all_indices.append(indices)\n            all_losses.append(loss)\n\n        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))\n        return quantized_out, out_indices, out_losses\n\n    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:\n        residual = x\n        all_indices = []\n        n_q = n_q or len(self.layers)\n        for layer in self.layers[:n_q]:\n            indices = layer.encode(residual)\n            quantized = layer.decode(indices)\n            residual = residual - quantized\n            all_indices.append(indices)\n        out_indices = torch.stack(all_indices)\n        return out_indices",
  "language": "python"
 },
 "encodec.24k": {
  "title": "EnCodec 24 kHz: bande e codebook",
  "file": "facebookresearch/encodec · encodec/model.py",
  "start": 229,
  "code": "        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)\n        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)\n        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))\n        quantizer = qt.ResidualVectorQuantizer(\n            dimension=encoder.dimension,\n            n_q=n_q,\n            bins=1024,\n        )",
  "language": "python"
 },
 "musicgen.delay": {
  "title": "MusicGen: delay pattern",
  "file": "facebookresearch/audiocraft · audiocraft/modules/codebooks_patterns.py",
  "start": 305,
  "code": "class DelayedPatternProvider(CodebooksPatternProvider):\n    \"\"\"Provider for delayed pattern across delayed codebooks.\n    Codebooks are delayed in the sequence and sequence steps will contain codebooks\n    from different timesteps.\n\n    Example:\n        Taking timesteps=4 and n_q=3, delays=None, the multi-codebook sequence:\n        [[1, 2, 3, 4],\n        [1, 2, 3, 4],\n        [1, 2, 3, 4]]\n        The resulting sequence obtained from the returned pattern is:\n        [[S, 1, 2, 3, 4],\n        [S, S, 1, 2, 3],\n        [S, S, S, 1, 2]]\n        (with S being a special token)\n\n    Args:\n        n_q (int): Number of codebooks.\n        delays (list of int, optional): Delay for each of the codebooks.\n            If delays not defined, each codebook is delayed by 1 compared to the previous one.\n        flatten_first (int): Flatten the first N timesteps.\n        empty_initial (int): Prepend with N empty list of coordinates.\n    \"\"\"",
  "language": "python"
 },
 "musicgen.lm": {
  "title": "MusicGen: forward del transformer",
  "file": "facebookresearch/audiocraft · audiocraft/models/lm.py",
  "start": 242,
  "code": "        B, K, S = sequence.shape\n        assert K == self.num_codebooks, \"Sequence shape must match the specified number of codebooks\"\n        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])\n        if condition_tensors is None:\n            assert not self._is_streaming, \"Conditions tensors should be precomputed when streaming.\"\n            # apply dropout modules\n            conditions = self.cfg_dropout(conditions)\n            conditions = self.att_dropout(conditions)\n            tokenized = self.condition_provider.tokenize(conditions)\n            # encode conditions and fuse, both have a streaming cache to not recompute when generating.\n            condition_tensors = self.condition_provider(tokenized)\n        else:\n            assert not conditions, \"Shouldn't pass both conditions and condition_tensors.\"\n\n        input_, cross_attention_input = self.fuser(input_, condition_tensors)\n\n        out = self.transformer(input_, cross_attention_src=cross_attention_input,\n                               src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None))  # type: ignore\n        if self.out_norm:\n            out = self.out_norm(out)\n        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]",
  "language": "python"
 },
 "musicgen.config": {
  "title": "MusicGen: 4 codebook, delay [0,1,2,3]",
  "file": "facebookresearch/audiocraft · config/model/lm/musicgen_lm.yaml",
  "start": 1,
  "code": "# @package __global__\n\ndefaults:\n  - lm/default\n  - override /conditioner: text2music\n  - override /model/lm/model_scale: small # prefer this group to set model scale instead of transformer_lm keys directly\n\nlm_model: transformer_lm\n\ncodebooks_pattern:\n  modeling: delay\n  delay:\n    delays: [0, 1, 2, 3]\n    flatten_first: 0\n    empty_initial: 0\n  unroll:\n    flattening: [0, 1, 2, 3]\n    delays: [0, 0, 0, 0]\n  music_lm:\n    group_by: 2\n  coarse_first:\n    delays: [0, 0, 0]\n\ntransformer_lm:\n  n_q: 4\n  card: 2048\n  memory_efficient: true\n  bias_proj: false",
  "language": "yaml"
 },
 "musicgen.t5": {
  "title": "MusicGen: condizionamento testuale con T5",
  "file": "facebookresearch/audiocraft · config/conditioner/text2music.yaml",
  "start": 1,
  "code": "# @package __global__\n\nclassifier_free_guidance:\n  training_dropout: 0.3\n  inference_coef: 3.0\n\nattribute_dropout: {}\n\nfuser:\n  cross_attention_pos_emb: false\n  cross_attention_pos_emb_scale: 1\n  sum: []\n  prepend: []\n  cross: [description]\n  input_interpolate: []\n\nconditioners:\n  description:\n    model: t5\n    t5:\n      name: t5-base\n      finetune: false\n      word_dropout: 0.3\n      normalize_text: false\n",
  "language": "yaml"
 },
 "musicgen.cfg": {
  "title": "MusicGen: classifier-free guidance",
  "file": "facebookresearch/audiocraft · audiocraft/models/lm.py",
  "start": 390,
  "code": "            condition_tensors = cfg_conditions\n            if condition_tensors:\n                # Preparing for CFG, predicting both conditional and unconditional logits.\n                sequence = torch.cat([sequence, sequence], dim=0)\n            all_logits = model(\n                sequence,\n                conditions=[], condition_tensors=condition_tensors)\n            if condition_tensors:\n                cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]\n                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef\n            else:\n                logits = all_logits",
  "language": "python"
 }
};
