# PROBE RESULT — vLLM 0.27.1 × Olmo3 × YARN correctness (settles Codex press-point (g))

VERDICT: vLLM 0.27.1 Olmo3+YARN is **CORRECT CONDITIONAL ON the installed transformers version**.
The plan's existing G6 floor (`transformers >= 5.13.0`) DOES cover the generation half — but only
because of a non-obvious dependency chain, and only if the floor binds in the venv vLLM runs in.
UNCONDITIONALLY vLLM 0.27.1 is NOT immune: with transformers 5.5.3-5.12.x installed (which
vLLM's OWN floor permits), vLLM silently inherits the identical per-layer-type YARN defect.
Confidence: HIGH — direct source reads at the exact pinned tag PLUS empirical config-object dumps
under both a buggy and a fixed transformers version. Nothing rests on absence-of-bug-report.

## MECHANISM — why "vLLM has its own RoPE" does not decide this
vLLM has its own RoPE KERNELS but NOT its own CONFIG PARSING.
`vllm/model_executor/models/olmo3.py` @ v0.27.1 imports `Olmo3Config` from the INSTALLED
transformers (L33, asserted L77) and builds per-layer rotary embeddings from
`self.config.rope_parameters` (L139-147):

    # Rotary embeddings. Rope scaling is only applied on full attention layers.
    rope_parameters = self.config.rope_parameters
    attn_type = "full_attention" if sliding_window is None else "sliding_attention"
    rope_parameters = rope_parameters.get(attn_type, rope_parameters)

- **transformers >= 5.13.0** (fix PR huggingface/transformers#46911 added
  `Olmo3Config.convert_rope_params_to_dict`): `rope_parameters` splits per layer type —
  `{"sliding_attention": {rope_type "default", theta 500000.0},
    "full_attention": {rope_type "yarn", factor 8.0, orig_max_pos 8192, beta_fast 32,
                       beta_slow 1, attention_factor 1.2079441541679836, theta 500000}}`.
  vLLM then gives full-attention layers YARN and sliding layers plain RoPE. Matches the
  published OLMo3 design (YaRN exclusively on full-attention layers), corroborated independently
  by the llama.cpp Olmo3 PR and rasbt/LLMs-from-scratch issue #939.
- **transformers 5.12.0**: `rope_parameters` is a FLAT yarn dict with NO layer-type keys. vLLM's
  `.get(attn_type, rope_parameters)` FALLBACK returns that flat dict for EVERY layer, so the
  sliding layers — 3 of every 4, i.e. 24/32 for the 7B and 48/64 for the 32B — receive YaRN
  frequency interpolation plus the 1.2079 mscale they must not have. Loads fine, generates fine,
  silently wrong. Same failure class as the HF path, now on the generation half.

vLLM's own YaRN math is fine for these checkpoints: `YaRNScalingRotaryEmbedding` computes
mscale = `yarn_get_mscale(8.0) * attn_factor(=1)` = 1.2079441541679836, numerically identical to
the configs' `attention_factor`; beta_fast/beta_slow forwarded and equal the defaults;
max_position_embeddings 65536 = 8192 × 8 consistent both sides.

## EVIDENCE
- https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/olmo3.py
  (import L33, isinstance assert L77, rope block L139-147); rotary_embedding/__init__.py yarn
  branch L243; yarn_scaling_rope.py mscale computation.
- https://github.com/huggingface/transformers/pull/46911 — the `configuration_olmo3.py` hunk adds
  `default_theta = 500000.0` and `convert_rope_params_to_dict`, merging legacy `rope_scaling`
  into `full_attention` ONLY.
- Empirical config dumps this session from the SAME `allenai/Olmo-3-7B-Instruct` config.json:
  5.13.0 → per-layer-type dict; 5.12.0 → flat yarn dict.
- All four checkpoint configs fetched live: `Olmo3ForCausalLM`, identical YARN rope_scaling
  (factor 8.0, orig 8192, attention_factor 1.2079441541679836), rope_theta 500000, sliding_window
  4096, alternating layer_types (3× sliding, 1× full; 32 layers 7B, 64 layers 32B), and NO
  per-layer rope fields in the JSON itself — the split is CREATED BY THE CONFIG CLASS, which is
  exactly why the installed transformers version decides correctness.
- vLLM 0.27.1 `requirements/common.txt`: `transformers >= 5.5.3`, NO upper pin — the buggy range
  is installable right next to vLLM 0.27.1.
- OLMo-core issue #685 does not mention vLLM either way; it pinned the regression to the
  transformers RoPE refactor and confirmed wrong outputs vs paper table 3.

## WHAT THE PLAN SHOULD DO
1. **Cite, do not re-litigate.** State in G6, in one sentence, that the floor covers vLLM
   generation BECAUSE vLLM 0.27.1 builds Olmo3 rope from the installed transformers
   `Olmo3Config.rope_parameters` — so nobody later "optimizes" the generation venv to a separate
   unfloored transformers install. The floor must bind in EVERY venv that runs vLLM, not only the
   capture venv.
2. **Add the structural probe** to the cell-driver prologue beside G6. O(seconds), CPU-only, no
   weights, and it ALSO guards future transformers >5.13 API drift, since vLLM 0.27.1 has no
   upper pin. It asserts the exact contract vLLM's L142 consumes, which is strictly better than
   asserting a version number:

       from transformers import AutoConfig
       cfg = AutoConfig.from_pretrained(model_id)   # each OLMo id
       rp = cfg.rope_parameters
       assert set(rp) >= {"full_attention", "sliding_attention"}, f"flat/malformed rope_parameters: {rp}"
       assert rp["full_attention"].get("rope_type") == "yarn", rp
       assert rp["sliding_attention"].get("rope_type") == "default", rp

3. **HF-vs-vLLM numerical parity probe — OPTIONAL** (items 1-2 pin the mechanism directly).
   ~10-15 min one-time on the first OLMo cell (the 7B pod already loads the model; one extra HF
   bf16 load + a few hundred teacher-forced positions). The defect engages at SHORT prompts too
   (YaRN alters inv_freq at every nonzero position and mscale multiplies all cos/sin), so ~512
   tokens suffices — no need to exceed the 4096 sliding window. Gate: top-1 agreement ≥ 98% AND
   median per-token KL < 1e-2 on fp32 logprobs. Loose enough for fp/kernel noise (which gives
   per-token KL ~1e-4-1e-3 and top-1 ≥ ~99%), orders of magnitude below the bug's signature.

## RESIDUAL, stated for honesty
The code was verified at tag v0.27.1 and the config classes at 5.12.0/5.13.0 empirically, but no
end-to-end vLLM forward was run (no GPU provisioning, per brief). The conditional-correctness
chain is SOURCE-COMPLETE; item 3 is what would convert it to a measured end-to-end fact.
