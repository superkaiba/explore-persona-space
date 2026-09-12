# Source audit for faithful paired J/R lenses

Read-only research audit, 2026-09-11. No model inference/training, repo changes, tasks, or Claude/Anthropic service calls. Sources were extracted through Parallel CLI, inspected in read-only temporary Git clones, and checked against HTTP range-read artifact pickle metadata without loading tensors.

## Exact J estimator and calibration

For prompt p with tokenized length T, define M={skip_first,...,T−2}. The official code computes

\[
J^{(p)}_\ell = |M|^{-1}\sum_{t\in M}\sum_{t'\in M}\frac{\partial h_{L,t'}}{\partial h_{\ell,t}},\qquad
J_\ell = N^{-1}\sum_{p\in\text{successful prompts}} J^{(p)}_\ell.
\]

Causality makes terms with t'<t zero. This is a **sum over target positions, mean over source positions**, then equal mean over successful prompts. It is not a uniform mean over all causal token pairs, nor a self-position Jacobian. For each output dimension, one-hot cotangents are injected at all valid target positions simultaneously. Rows are averaged only over valid source positions. `dim_batch` replicates a prompt across batch to calculate multiple output coordinates per backward call. Per-prompt matrices and accumulation are fp32; `JacobianLens.save` defaults fp16. [Official fitting.py](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py), [official lens.py](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/lens.py)

The article's main mathematical expectation notation suppresses this normalization detail; its appendix pseudocode agrees with the sum-then-mean implementation. Paper default: 1,000 sequences of 128 tokens, pretraining-like distribution; default Sonnet analysis targets penultimate residual output. It reports calibration-size experiments down to 1 prompt and defines sparse decompositions with varying k (often around 25), **not a universal k=10**. [Workspace paper](https://transformer-circuits.pub/2026/workspace/)

The official API defaults differ from the release recipe: `target_layer=None` resolves to n_layers−1 and `skip_first=16`; max_seq_len=128. For release-style matched pairs these must explicitly become n_layers−2 and 4. Short sequences have no valid positions and are skipped. The code catches any ValueError from the whole per-prompt call and logs a skip; this should be tightened in a replication wrapper so genuine failures are not mislabeled as short prompts. Resume checkpoints validate only source_layers, target_layer, skip_first, and contain sums/counts/next_idx: they do **not** validate checkpoint hash, text/token manifests, norm rules, dtype, or tokenizer. Use separate files per arm and external manifest validation. [Official fitting.py](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py)

## Hook convention and readout

`ActivationRecorder` uses a forward hook on each numbered transformer block's **output** (`output` or `output[0]`), not block input. Its layer 0 is the residual after block 0; when using ordinary HF hidden-state tuples this is generally `hidden_states[1]`. The fitted source layers are strictly before target. The minimum source output is marked requires_grad to start the graph. Final readout applies `J_l @ h` followed by model-native final norm and LM head, including final softcap when configured. Ordinary forward retains native attention and uses `use_cache=False`; tokenizer may enable BOS through `from_hf(force_bos=True)`. [Official hooks.py](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hooks.py), [official hf.py](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hf.py)

`p_l(h)=softmax(W_U norm(J_l h))`. The paper names dictionary rows of W_U J_l. For exact native RMSNorm readout direction, algebraically fold the final norm's learned gain: `v_t=(W_U[t,:] * gamma) @ J_l`, then unit-normalize each nonzero dictionary row for pursuit (coefficient scale absorbs normalization). This folding is an inference from native readout equations, not an explicit normalization convention documented by the workspace sparse-pursuit implementation (none was released). With true LayerNorm the linear part also includes the centering operator; norm bias and LM-head bias are offsets, not feature directions. Positive scalar RMS denominator/monotone logit softcap do not change token ordering. Do not use the inverse J map or an SVD row-span projector as the sparse J-space. [Official hf.py](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/hf.py), [workspace definition](https://transformer-circuits.pub/2026/workspace/)

## R local backward rules

R uses the same estimator, forward checkpoint, hooks, prompts, token masks, targets, averaging and readout; the derivative is replaced by a product of local propagation coefficients. R is **not the actual Jacobian of the unchanged model**, and cannot be recovered from an averaged J matrix. Build it by rerunning native forward/backward with the local rules. [R-lens post](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)

Let sg denote stop-gradient. The forward-preserving rules are:

- RMSNorm: `y=gamma*x*sg(rsqrt(mean(x*x)+eps))`; local backward w.r.t. x is the fixed diagonal gamma*rsqrt. Preserve native precision/casting/epsilon and gain parameterization. True LayerNorm retains differentiation through `x−mean(x)` but detaches only the denominator, so the local matrix is `diag(gamma)/std * (I−11ᵀ/d)`.
- SiLU: `x*sg(sigmoid(x))`; derivative coefficient is sigmoid(x), not the full SiLU derivative.
- GELU: `x*sg(Phi(x))` with the **native GELU factor**. Exact GELU uses Gaussian CDF; tanh-approximate GELU uses its matching tanh factor. Do not literally divide GELU(x)/x at x=0 without analytic limiting handling.
- Gated MLP: a=activation(gate_proj(x)) under the identity rule, b=up_proj(x), z=a*b; replace product by `0.5*z + 0.5*sg(z)`. Thus both branch derivatives receive 0.5. This is not equal to detaching one branch entirely. Linear layers remain ordinary autograd (LRP 0-rule).

[RelP v2 rule table](https://arxiv.org/html/2508.21258v2), [original RelP normalization code](https://github.com/FarnoushRJ/RelP/blob/8219d6dc417c3fd7f318342cf61cd2a0c20b7250/TransformerLens/transformer_lens/components/rms_norm.py), [original LayerNorm code](https://github.com/FarnoushRJ/RelP/blob/8219d6dc417c3fd7f318342cf61cd2a0c20b7250/TransformerLens/transformer_lens/components/layer_norm.py), [original gated MLP code](https://github.com/FarnoushRJ/RelP/blob/8219d6dc417c3fd7f318342cf61cd2a0c20b7250/TransformerLens/transformer_lens/components/mlps/gated_mlp.py)

The **published dense R-lens** applies these to residual-stream RMSNorms and gated MLPs, explicitly leaving attention and q/k norms unchanged. Generic RelP's optional AH-rule detaches attention matrices, but applying that here would introduce a new R variant. The R post does not publish a derivation/rule for gated-delta recurrence or other hybrid attention internals. [R-lens methods](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)

## Qwen3.5 implementation boundary (important)

Current official HF Qwen3.5 source was inspected at file commit `bd15bc95a89e728bbc1224084eb3b5829428c353` (fetched from main; pin the actual intended runtime version separately).

- `Qwen3_5DecoderLayer.input_layernorm` and `.post_attention_layernorm` are `Qwen3_5RMSNorm`; it normalizes in fp32, multiplies by **1+weight.float()**, then casts to input dtype. Final text-model norm uses the same class. Zero-centered gain is not only a Gemma concern.
- `Qwen3_5MLP.forward` is `down_proj(act_fn(gate_proj(x))*up_proj(x))`.
- `Qwen3_5GatedDeltaNet.norm` is a different `Qwen3_5RMSNormGated`: RMS normalization, learned **weight** (not 1+weight), then SiLU on a separate gate. Released dense R metadata explicitly says `gated_norms=false`.
- Gated-delta recurrence, convolution/its activation, L2 q/k normalization, beta/decay gates, and full-attention output sigmoid gating remain ordinary in published dense-R scope. Native differentiable paths must still be validated for J and R; do not replace hybrid attention with softmax attention to get a backward pass.
- Q/k norm modules share the residual RMSNorm class but are excluded by path/scope. Patching all modules by class would violate released R scope.

[Official Qwen3.5 source](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## Exact nonnegative gradient pursuit, k=10

The workspace reference links Nanda et al.'s summary; its **full post** gives the algorithm. For dictionary rows D:[V,d], start a=0. Repeat **10 steps**:

1. r=x−aD; q=D r.
2. S=(a != 0); set S[argmax(q)]=true. Selection uses q, **not absolute q**, and can reselect an already-active atom.
3. g=S⊙q; c=gD; eta=(c·r)/(c·c).
4. a=max(a+eta*g,0).

Return x_sparse=aD and x_remainder=x−x_sparse. This has at most 10 positive coefficients, possibly fewer due reselection or clipping. It is not NNLS refitting and not one permanently new atom per step. The pseudocode omits zero-denominator handling; implement explicit documented guards. Stop safely for zero restricted update norm, or empty support with no positive correlation. Do not stop solely because max(q)<=0 while active coefficients remain: negative restricted gradients can reduce those coefficients. Full-vocabulary selection is the faithful baseline; fixed top-token candidate preselection is a separate approximation. [Primary GDM full post](https://www.lesswrong.com/s/AtTZjoDm8q3DbDT8Z/p/C5KAZQib3bzzpeyrg), [cited summary](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/summary-progress-update-1-from-the-gdm-mech-interp-team), [Gradient Pursuits DOI](https://doi.org/10.1109/TSP.2007.916124)

Important limitation: the paper does not release exact sparse-decomposition code or dictionary normalization flags. The above is the exact cited **nonnegative GP algorithm**, applied to the user's k=10 choice; it should not be described as identical unpublished Anthropic code. Because dictionary vectors are correlated and pursuit is approximate, the remainder is generally **not globally orthogonal** to the dictionary and should not be labeled “J-orthogonal”. [Workspace paper](https://transformer-circuits.pub/2026/workspace/), [GDM GP source](https://www.lesswrong.com/s/AtTZjoDm8q3DbDT8Z/p/C5KAZQib3bzzpeyrg)

The third-party idhantgulati implementation uses only the top 512 lens tokens and clips an unconstrained least-squares solution. That is not true NNLS or the above GP. Do not use it as the decomposition authority. [Third-party source](https://github.com/idhantgulati/j-lens/blob/main/jlens.py)

## Released lens artifacts: directly audited metadata

HF repository observed revision: `d740106d1e0f95456dc8718fba2895e9c8ffd6ef`. Data from **all 16 `lens.pt` data.pkl entries**, not merely the model card. All have `n_prompts=25`, `docs_consumed=25`, `t_max=128`, `skip_first=4`, `weighting='uniform'`, `corpus_mode='pretrain'`, dataset `NeelNanda/pile-10k`, fp16 per-layer [d,d] dictionaries. `J` is a dictionary keyed by layer (not a single stacked tensor). Keys run 0..target, including an anchor row that the model card calls identity; tensor values were not downloaded/validated by this metadata audit. [Released repository](https://huggingface.co/camilablank/workspace-lenses/tree/d740106d1e0f95456dc8718fba2895e9c8ffd6ef)

| Model directory | model_id | d_model | target / stored layer count | R metadata |
|---|---|---:|---:|---|
| qwen3.5-4b | Qwen/Qwen3.5-4B | 2560 | 30 / 31 | LN, identity, half; beta=.5; qk=false; gated_norms=false |
| qwen3.5-9b | Qwen/Qwen3.5-9B | 4096 | 30 / 31 | same dense config |
| qwen3.5-27b | Qwen/Qwen3.5-27B | 5120 | 62 / 63 | same dense config |
| qwen3.6-27b | Qwen/Qwen3.6-27B | 5120 | 62 / 63 | LN, identity, half; qk=false (beta/gated_norms absent) |
| gemma-3-27b-it | google/gemma-3-27b-it | 5376 | 60 / 61 | same dense config as Qwen3.5 |
| qwen3.5-122b-a10b | Qwen/Qwen3.5-122B-A10B | 3072 | 46 / 47 | LN, identity, half; qk=false; routed_experts/router_detach/router_half/shared_gate_detach all false |
| qwen3.6-35b-a3b | Qwen/Qwen3.6-35B-A3B | 2048 | 38 / 39 | routed_experts=true, router_detach=true, router_half=false, shared_gate_detach=true, shared_expert_scale=4 |
| deepseek-v4-flash | deepseek-ai/DeepSeek-V4-Flash | 4096 | 41 / 42 | arm all-c4; routed_experts=true, router_detach=true, router_half=false, shared_gate_detach=false, mhc_detach=true, shared_expert_scale=4 |

Every row also enables LN/identity/half, excludes qk norms. Sources are corresponding `j-lens/lens.pt` and `r-lens/lens.pt` under the model directory at the pinned repository URL above; exact URLs and all artifact LFS SHA256s in `/tmp/jr-hf-all-lens-metadata.json`. [Metadata API](https://huggingface.co/api/models/camilablank/workspace-lenses/tree/main?recursive=true&expand=false)

### Provenance obstacles

1. **Every artifact has `git_commit='modal'` and `n_positions=0.0`.** Contrary to the card's generic promise of “full per-artifact provenance (git commit...)”, these do not identify source implementation or measured valid-position counts. No exact source checkpoint revision, tokenizer revision, dataset revision, selected document IDs/text/token hashes, seed, or residual hook/collapse specification is embedded. Recipe fields match within pairs; exact bitwise calibration/checkpoint provenance is not proven.
2. R card points to `global_workspace` code and DeepSeek origin card names `docs/project/experiments/relp_jlens/replication.md`, but no public implementation URL is provided. Referenced `agu18dec/relp-jlens-matrix` and `agu18dec/qwen3.6-27b-relp-jlens` API reads returned 401. A 401 is inaccessible, not evidence of nonexistence. The accessible DeepSeek origin has no published source code in its file tree.
3. DeepSeek's mHC residual is [4,d], but artifact matrices are [d,d]. The collapse/injection representation is not specified in metadata; a naive ordinary residual hook does not establish faithful consumption or rebuild. It requires the intended native-state read/write convention, along with native quantization/backward support.
4. Two MoE releases use swept shared-expert backward scaling 4, an extra model-specific choice rather than just the dense three rules. Do not silently transplant c=4 to another MoE checkpoint. Qwen122 release explicitly does not enable routed-expert rules.
5. The card claims forward bit-identity; this audit only verified metadata, not forward equality. A fresh builder should check forward outputs with rules off/on, native-hook alignment, finite matrices, anchor identity, and exact calibration manifests for both arms. Rebuilding matched lenses on pinned current checkpoints is clearer than asserting missing release provenance.

[Released model card](https://huggingface.co/camilablank/workspace-lenses), [DeepSeek origin](https://huggingface.co/camilablank/deepseek-elens-jlens), [R methods](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)

## Local evidence files

- `/tmp/jr-lens-primary-extract.json`, `/tmp/jr-source-workspace-full.txt`: main paper and release card.
- `/tmp/jr-r-lens-post-retry.json`, `/tmp/jr-r-lens-post-0.txt`: full R post (initial LessWrong read 429, subsequent retry succeeded).
- `/tmp/jr-hf-all-lens-metadata.json`, `/tmp/jr-hf-tree.json`, `/tmp/jr-hf-model-api.json`: audited artifact metadata and file hashes.
- `/tmp/jr-read-hf-metadata.py`: repeatable HTTP range-read metadata audit; whitelist unpickler replaces tensor construction with storage/shape metadata; never reads full tensor data.
- `/tmp/jr-modeling-qwen3_5.py`: inspected native HF source.
- `/tmp/jr-official-jacobian-lens`: source checkout SHA 581d398613e5602a5af361e1c34d3a92ea82ba8e.
- `/tmp/jr-primary-relp`: source checkout SHA 8219d6dc417c3fd7f318342cf61cd2a0c20b7250.
- `/tmp/jr-gdm-full-api-extract.json`, `/tmp/jr-gdm-full-post.txt`: successfully extracted primary GP full post through LessWrong Markdown API. Canonical URL extraction `/tmp/jr-gdm-full-gp-extract.json` failed HTTP 429; web read and API extraction succeeded.

Sources:
- [Workspace paper](https://transformer-circuits.pub/2026/workspace/)
- [R-lens post](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)
- [RelP v2](https://arxiv.org/html/2508.21258v2)
- [Official J code](https://github.com/anthropics/jacobian-lens)
- [Original RelP code](https://github.com/FarnoushRJ/RelP)
- [GDM full GP post](https://www.lesswrong.com/s/AtTZjoDm8q3DbDT8Z/p/C5KAZQib3bzzpeyrg)
- [GDM summary](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/summary-progress-update-1-from-the-gdm-mech-interp-team)
- [Gradient Pursuits](https://doi.org/10.1109/TSP.2007.916124)
- [Released artifacts](https://huggingface.co/camilablank/workspace-lenses)
- [DeepSeek origin](https://huggingface.co/camilablank/deepseek-elens-jlens)
- [Qwen3.5 official HF source](https://github.com/huggingface/transformers/blob/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Third-party J example (not algorithm authority)](https://github.com/idhantgulati/j-lens/blob/main/jlens.py)
