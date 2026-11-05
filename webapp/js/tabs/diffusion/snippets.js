/**
 * Frammenti di codice VERI copiati dai repository ufficiali, a commit fissati (vedi `url`).
 * File generato da uno script di estrazione: non modificare a mano i campi `code`.
 * Licenze: diffusers Apache-2.0, DiT CC-BY-NC 4.0, Stable Diffusion CreativeML OpenRAIL-M;
 * gli estratti sono brevi e citati a scopo didattico.
 */
export const SNIPPETS = {
  "ddpm_beta": {
    "repo": "hojonathanho/diffusion",
    "file": "hojonathanho/diffusion/diffusion_tf/diffusion_utils_2.py",
    "startLine": 23,
    "language": "python",
    "license": "nessuna licenza dichiarata nel repository: breve estratto citato a scopo didattico",
    "url": "https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L23-L27",
    "code": "def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):\n  if beta_schedule == 'quad':\n    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2\n  elif beta_schedule == 'linear':\n    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)"
  },
  "ddpm_qsample": {
    "repo": "hojonathanho/diffusion",
    "file": "hojonathanho/diffusion/diffusion_tf/diffusion_utils_2.py",
    "startLine": 101,
    "language": "python",
    "license": "nessuna licenza dichiarata nel repository: breve estratto citato a scopo didattico",
    "url": "https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L101-L111",
    "code": "  def q_sample(self, x_start, t, noise=None):\n    \"\"\"\n    Diffuse the data (t == 0 means diffused for 1 step)\n    \"\"\"\n    if noise is None:\n      noise = tf.random_normal(shape=x_start.shape)\n    assert noise.shape == x_start.shape\n    return (\n        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +\n        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise\n    )"
  },
  "ddpm_config": {
    "repo": "hojonathanho/diffusion",
    "file": "hojonathanho/diffusion/scripts/run_cifar.py",
    "startLine": 132,
    "language": "python",
    "license": "nessuna licenza dichiarata nel repository: breve estratto citato a scopo didattico",
    "url": "https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/scripts/run_cifar.py#L132-L138",
    "code": "def train(\n    exp_name, tpu_name, bucket_name_prefix, model_name='unet2d16b2', dataset='cifar10',\n    optimizer='adam', total_bs=128, grad_clip=1., lr=2e-4, warmup=5000,\n    num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear',\n    model_mean_type='eps', model_var_type='fixedlarge', loss_type='mse',\n    dropout=0.1, randflip=1,\n    tfds_data_dir='tensorflow_datasets', log_dir='logs', keep_checkpoint_max=2"
  },
  "ddpm_resnet": {
    "repo": "hojonathanho/diffusion",
    "file": "hojonathanho/diffusion/diffusion_tf/models/unet.py",
    "startLine": 37,
    "language": "python",
    "license": "nessuna licenza dichiarata nel repository: breve estratto citato a scopo didattico",
    "url": "https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L37-L63",
    "code": "def resnet_block(x, *, temb, name, out_ch=None, conv_shortcut=False, dropout):\n  B, H, W, C = x.shape\n  if out_ch is None:\n    out_ch = C\n\n  with tf.variable_scope(name):\n    h = x\n\n    h = nonlinearity(normalize(h, temb=temb, name='norm1'))\n    h = nn.conv2d(h, name='conv1', num_units=out_ch)\n\n    # add in timestep embedding\n    h += nn.dense(nonlinearity(temb), name='temb_proj', num_units=out_ch)[:, None, None, :]\n\n    h = nonlinearity(normalize(h, temb=temb, name='norm2'))\n    h = tf.nn.dropout(h, rate=dropout)\n    h = nn.conv2d(h, name='conv2', num_units=out_ch, init_scale=0.)\n\n    if C != out_ch:\n      if conv_shortcut:\n        x = nn.conv2d(x, name='conv_shortcut', num_units=out_ch)\n      else:\n        x = nn.nin(x, name='nin_shortcut', num_units=out_ch)\n\n    assert x.shape == h.shape\n    print('{}: x={} temb={}'.format(tf.get_default_graph().get_name_scope(), x.shape, temb.shape))\n    return x + h"
  },
  "sd_config": {
    "repo": "CompVis/stable-diffusion",
    "file": "CompVis/stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
    "startLine": 1,
    "language": "yaml",
    "license": "CreativeML OpenRAIL-M",
    "url": "https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L1-L71",
    "code": "model:\n  base_learning_rate: 1.0e-04\n  target: ldm.models.diffusion.ddpm.LatentDiffusion\n  params:\n    linear_start: 0.00085\n    linear_end: 0.0120\n    num_timesteps_cond: 1\n    log_every_t: 200\n    timesteps: 1000\n    first_stage_key: \"jpg\"\n    cond_stage_key: \"txt\"\n    image_size: 64\n    channels: 4\n    cond_stage_trainable: false   # Note: different from the one we trained before\n    conditioning_key: crossattn\n    monitor: val/loss_simple_ema\n    scale_factor: 0.18215\n    use_ema: False\n\n    scheduler_config: # 10000 warmup steps\n      target: ldm.lr_scheduler.LambdaLinearScheduler\n      params:\n        warm_up_steps: [ 10000 ]\n        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases\n        f_start: [ 1.e-6 ]\n        f_max: [ 1. ]\n        f_min: [ 1. ]\n\n    unet_config:\n      target: ldm.modules.diffusionmodules.openaimodel.UNetModel\n      params:\n        image_size: 32 # unused\n        in_channels: 4\n        out_channels: 4\n        model_channels: 320\n        attention_resolutions: [ 4, 2, 1 ]\n        num_res_blocks: 2\n        channel_mult: [ 1, 2, 4, 4 ]\n        num_heads: 8\n        use_spatial_transformer: True\n        transformer_depth: 1\n        context_dim: 768\n        use_checkpoint: True\n        legacy: False\n\n    first_stage_config:\n      target: ldm.models.autoencoder.AutoencoderKL\n      params:\n        embed_dim: 4\n        monitor: val/rec_loss\n        ddconfig:\n          double_z: true\n          z_channels: 4\n          resolution: 256\n          in_channels: 3\n          out_ch: 3\n          ch: 128\n          ch_mult:\n          - 1\n          - 2\n          - 4\n          - 4\n          num_res_blocks: 2\n          attn_resolutions: []\n          dropout: 0.0\n        lossconfig:\n          target: torch.nn.Identity\n\n    cond_stage_config:\n      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder"
  },
  "sd_encode": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py",
    "startLine": 396,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L396-L424",
    "code": "            text_inputs = self.tokenizer(\n                prompt,\n                padding=\"max_length\",\n                max_length=self.tokenizer.model_max_length,\n                truncation=True,\n                return_tensors=\"pt\",\n            )\n            text_input_ids = text_inputs.input_ids\n            untruncated_ids = self.tokenizer(prompt, padding=\"longest\", return_tensors=\"pt\").input_ids\n\n            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(\n                text_input_ids, untruncated_ids\n            ):\n                removed_text = self.tokenizer.batch_decode(\n                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]\n                )\n                logger.warning(\n                    \"The following part of your input was truncated because CLIP can only handle sequences up to\"\n                    f\" {self.tokenizer.model_max_length} tokens: {removed_text}\"\n                )\n\n            if hasattr(self.text_encoder.config, \"use_attention_mask\") and self.text_encoder.config.use_attention_mask:\n                attention_mask = text_inputs.attention_mask.to(device)\n            else:\n                attention_mask = None\n\n            if clip_skip is None:\n                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)\n                prompt_embeds = prompt_embeds[0]"
  },
  "sd_loop": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py",
    "startLine": 1032,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1032-L1062",
    "code": "            for i, t in enumerate(timesteps):\n                if self.interrupt:\n                    continue\n\n                # expand the latents if we are doing classifier free guidance\n                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents\n                if hasattr(self.scheduler, \"scale_model_input\"):\n                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)\n\n                # predict the noise residual\n                noise_pred = self.unet(\n                    latent_model_input,\n                    t,\n                    encoder_hidden_states=prompt_embeds,\n                    timestep_cond=timestep_cond,\n                    cross_attention_kwargs=self.cross_attention_kwargs,\n                    added_cond_kwargs=added_cond_kwargs,\n                    return_dict=False,\n                )[0]\n\n                # perform guidance\n                if self.do_classifier_free_guidance:\n                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)\n                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)\n\n                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:\n                    # Based on 3.4. in https://huggingface.co/papers/2305.08891\n                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)\n\n                # compute the previous noisy sample x_t -> x_t-1\n                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]"
  },
  "sd_decode": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py",
    "startLine": 1084,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1084-L1087",
    "code": "        if not output_type == \"latent\":\n            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[\n                0\n            ]"
  },
  "ddim_step": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/schedulers/scheduling_ddim.py",
    "startLine": 444,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/schedulers/scheduling_ddim.py#L444-L491",
    "code": "        # 1. get previous step value (=t-1)\n        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps\n\n        # 2. compute alphas, betas\n        alpha_prod_t = self.alphas_cumprod[timestep]\n        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod\n\n        beta_prod_t = 1 - alpha_prod_t\n\n        # 3. compute predicted original sample from predicted noise also called\n        # \"predicted x_0\" of formula (12) from https://huggingface.co/papers/2010.02502\n        if self.config.prediction_type == \"epsilon\":\n            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)\n            pred_epsilon = model_output\n        elif self.config.prediction_type == \"sample\":\n            pred_original_sample = model_output\n            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)\n        elif self.config.prediction_type == \"v_prediction\":\n            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output\n            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample\n        else:\n            raise ValueError(\n                f\"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or\"\n                \" `v_prediction`\"\n            )\n\n        # 4. Clip or threshold \"predicted x_0\"\n        if self.config.thresholding:\n            pred_original_sample = self._threshold_sample(pred_original_sample)\n        elif self.config.clip_sample:\n            pred_original_sample = pred_original_sample.clamp(\n                -self.config.clip_sample_range, self.config.clip_sample_range\n            )\n\n        # 5. compute variance: \"sigma_t(η)\" -> see formula (16)\n        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)\n        variance = self._get_variance(timestep, prev_timestep)\n        std_dev_t = eta * variance ** (0.5)\n\n        if use_clipped_model_output:\n            # the pred_epsilon is always re-derived from the clipped x_0 in Glide\n            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)\n\n        # 6. compute \"direction pointing to x_t\" of formula (12) from https://huggingface.co/papers/2010.02502\n        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon\n\n        # 7. compute x_t without \"random noise\" of formula (12) from https://huggingface.co/papers/2010.02502\n        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction"
  },
  "flow_noise": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    "startLine": 230,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L230-L236",
    "code": "        sigma = sigmas[step_indices].flatten()\n        while len(sigma.shape) < len(sample.shape):\n            sigma = sigma.unsqueeze(-1)\n\n        sample = sigma * noise + (1.0 - sigma) * sample\n\n        return sample"
  },
  "flow_step": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    "startLine": 506,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L506-L520",
    "code": "        else:\n            sigma_idx = self.step_index\n            sigma = self.sigmas[sigma_idx]\n            sigma_next = self.sigmas[sigma_idx + 1]\n\n            current_sigma = sigma\n            next_sigma = sigma_next\n            dt = sigma_next - sigma\n\n        if self.config.stochastic_sampling:\n            x0 = sample - current_sigma * model_output\n            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)\n            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise\n        else:\n            prev_sample = sample + dt * model_output"
  },
  "sd3_forward": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/models/transformers/transformer_sd3.py",
    "startLine": 289,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/models/transformers/transformer_sd3.py#L289-L320",
    "code": "        height, width = hidden_states.shape[-2:]\n\n        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.\n        # pos_embed output is non-contiguous due to flatten+transpose in PatchEmbed (BCHW -> BNC).\n        hidden_states = hidden_states.contiguous()\n        temb = self.time_text_embed(timestep, pooled_projections)\n        encoder_hidden_states = self.context_embedder(encoder_hidden_states)\n\n        if joint_attention_kwargs is not None and \"ip_adapter_image_embeds\" in joint_attention_kwargs:\n            ip_adapter_image_embeds = joint_attention_kwargs.pop(\"ip_adapter_image_embeds\")\n            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)\n\n            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)\n\n        for index_block, block in enumerate(self.transformer_blocks):\n            # Skip specified layers\n            is_skip = True if skip_layers is not None and index_block in skip_layers else False\n\n            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:\n                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(\n                    block,\n                    hidden_states,\n                    encoder_hidden_states,\n                    temb,\n                    joint_attention_kwargs,\n                )\n            elif not is_skip:\n                encoder_hidden_states, hidden_states = block(\n                    hidden_states=hidden_states,\n                    encoder_hidden_states=encoder_hidden_states,\n                    temb=temb,\n                    joint_attention_kwargs=joint_attention_kwargs,"
  },
  "dit_modulate": {
    "repo": "facebookresearch/DiT",
    "file": "facebookresearch/DiT/models.py",
    "startLine": 19,
    "language": "python",
    "license": "CC-BY-NC 4.0",
    "url": "https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L19-L20",
    "code": "def modulate(x, shift, scale):\n    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)"
  },
  "dit_temb": {
    "repo": "facebookresearch/DiT",
    "file": "facebookresearch/DiT/models.py",
    "startLine": 27,
    "language": "python",
    "license": "CC-BY-NC 4.0",
    "url": "https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27-L60",
    "code": "class TimestepEmbedder(nn.Module):\n    \"\"\"\n    Embeds scalar timesteps into vector representations.\n    \"\"\"\n    def __init__(self, hidden_size, frequency_embedding_size=256):\n        super().__init__()\n        self.mlp = nn.Sequential(\n            nn.Linear(frequency_embedding_size, hidden_size, bias=True),\n            nn.SiLU(),\n            nn.Linear(hidden_size, hidden_size, bias=True),\n        )\n        self.frequency_embedding_size = frequency_embedding_size\n\n    @staticmethod\n    def timestep_embedding(t, dim, max_period=10000):\n        \"\"\"\n        Create sinusoidal timestep embeddings.\n        :param t: a 1-D Tensor of N indices, one per batch element.\n                          These may be fractional.\n        :param dim: the dimension of the output.\n        :param max_period: controls the minimum frequency of the embeddings.\n        :return: an (N, D) Tensor of positional embeddings.\n        \"\"\"\n        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py\n        half = dim // 2\n        freqs = torch.exp(\n            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half\n        ).to(device=t.device)\n        args = t[:, None].float() * freqs[None]\n        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)\n        if dim % 2:\n            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)\n        return embedding"
  },
  "dit_block": {
    "repo": "facebookresearch/DiT",
    "file": "facebookresearch/DiT/models.py",
    "startLine": 101,
    "language": "python",
    "license": "CC-BY-NC 4.0",
    "url": "https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L123",
    "code": "class DiTBlock(nn.Module):\n    \"\"\"\n    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.\n    \"\"\"\n    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):\n        super().__init__()\n        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)\n        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)\n        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)\n        mlp_hidden_dim = int(hidden_size * mlp_ratio)\n        approx_gelu = lambda: nn.GELU(approximate=\"tanh\")\n        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)\n        self.adaLN_modulation = nn.Sequential(\n            nn.SiLU(),\n            nn.Linear(hidden_size, 6 * hidden_size, bias=True)\n        )\n\n    def forward(self, x, c):\n        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)\n        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))\n        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))\n        return x"
  },
  "dit_forward": {
    "repo": "facebookresearch/DiT",
    "file": "facebookresearch/DiT/models.py",
    "startLine": 233,
    "language": "python",
    "license": "CC-BY-NC 4.0",
    "url": "https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L233-L248",
    "code": "    def forward(self, x, t, y):\n        \"\"\"\n        Forward pass of DiT.\n        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)\n        t: (N,) tensor of diffusion timesteps\n        y: (N,) tensor of class labels\n        \"\"\"\n        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2\n        t = self.t_embedder(t)                   # (N, D)\n        y = self.y_embedder(y, self.training)    # (N, D)\n        c = t + y                                # (N, D)\n        for block in self.blocks:\n            x = block(x, c)                      # (N, T, D)\n        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)\n        x = self.unpatchify(x)                   # (N, out_channels, H, W)\n        return x"
  },
  "dit_xl2": {
    "repo": "facebookresearch/DiT",
    "file": "facebookresearch/DiT/models.py",
    "startLine": 328,
    "language": "python",
    "license": "CC-BY-NC 4.0",
    "url": "https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L328-L329",
    "code": "def DiT_XL_2(**kwargs):\n    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)"
  },
  "temporal_attn": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/models/transformers/transformer_temporal.py",
    "startLine": 163,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/models/transformers/transformer_temporal.py#L163-L197",
    "code": "        # 1. Input\n        batch_frames, channel, height, width = hidden_states.shape\n        batch_size = batch_frames // num_frames\n\n        residual = hidden_states\n\n        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)\n        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)\n\n        hidden_states = self.norm(hidden_states)\n        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)\n\n        hidden_states = self.proj_in(hidden_states)\n\n        # 2. Blocks\n        for block in self.transformer_blocks:\n            hidden_states = block(\n                hidden_states,\n                encoder_hidden_states=encoder_hidden_states,\n                timestep=timestep,\n                cross_attention_kwargs=cross_attention_kwargs,\n                class_labels=class_labels,\n            )\n\n        # 3. Output\n        hidden_states = self.proj_out(hidden_states)\n        hidden_states = (\n            hidden_states[None, None, :]\n            .reshape(batch_size, height, width, num_frames, channel)\n            .permute(0, 3, 4, 1, 2)\n            .contiguous()\n        )\n        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)\n\n        output = hidden_states + residual"
  },
  "causal_conv3d": {
    "repo": "huggingface/diffusers",
    "file": "huggingface/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py",
    "startLine": 46,
    "language": "python",
    "license": "Apache-2.0",
    "url": "https://github.com/huggingface/diffusers/blob/a91eccd57264c3bb17cdc0721bbd82010e81c076/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py#L46-L76",
    "code": "class HunyuanVideoCausalConv3d(nn.Module):\n    def __init__(\n        self,\n        in_channels: int,\n        out_channels: int,\n        kernel_size: int | tuple[int, int, int] = 3,\n        stride: int | tuple[int, int, int] = 1,\n        padding: int | tuple[int, int, int] = 0,\n        dilation: int | tuple[int, int, int] = 1,\n        bias: bool = True,\n        pad_mode: str = \"replicate\",\n    ) -> None:\n        super().__init__()\n\n        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size\n\n        self.pad_mode = pad_mode\n        self.time_causal_padding = (\n            kernel_size[0] // 2,\n            kernel_size[0] // 2,\n            kernel_size[1] // 2,\n            kernel_size[1] // 2,\n            kernel_size[2] - 1,\n            0,\n        )\n\n        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)\n\n    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:\n        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)\n        return self.conv(hidden_states)"
  }
};
