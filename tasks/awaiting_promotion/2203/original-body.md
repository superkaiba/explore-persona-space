---
title: Assistant-axis capping at the context vector vs all tokens
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-08T16:23:11Z'
has_clean_result: false
origin_prompt: We've found a lot of persona info is stored at the context vector.
  The assistant axis paper caps activations along the assistant axis at all tokens;
  a more efficient way might be to cap (or patch) only at the context vector / prefix
  vector. Reproduce the capping experiment and compare position sets + patching variants.
workflow: v1
goal: Determine whether assistant-axis activation capping (and its query-preserving
  patch generalization) applied ONLY at the context-vector position recovers the jailbreak-reduction
  / persona-stabilization effect that Lu et al. (arXiv 2601.10387) get by capping
  at every token, and whether prefix-only capping fails — via a position ladder (prefix-end
  / context-end / all-prompt / all-tokens) x intervention type (cap / axis-component-replace
  / full-replace) over a fixed mid-late layer band, on Qwen-2.5-7B (in-house axis)
  with a Qwen-3-32B faithful anchor, scored on co-primary judged jailbreak-harm and
  role-susceptibility rates.
relates_to:
- spec-context-as-vector
- spec-steering
- spec-sysprompt-vs-drift
---
# Assistant-axis capping at the context vector vs all tokens

## Provenance
Originating prompt (Thomas, chat, 2026-08-08): "We've found that a lot of persona
information is stored at the context vector. One application of controlling personas is
preventing the model from straying too far from the assistant persona. The assistant axis
[Lu et al. 2026, arXiv 2601.10387] does this by capping the model's activation along the
assistant axis. One more efficient way might be to just cap **at the context vector** (or
patch?). Reproduce the activation-capping experiment and compare: capping at all tokens
(like they did) / only at the context vector / only at the prefix vector; plus patching the
default assistant prefix/context vector at subsequent positions (while maintaining query
info)."

## Goal
Determine whether activation capping — and its patching generalizations — applied **only at
the context-vector position** recovers the persona-stabilization / jailbreak-reduction effect
that Lu et al. (arXiv 2601.10387) obtain by capping **at every token**, and whether
**prefix-only** capping fails.

Formally: let `v` be the (unit, per-layer) Assistant Axis extracted as
`mean(default-assistant activation) − mean(fully-role-playing role vectors)`. Capping updates a
capped layer's post-MLP residual `h` as

    h ← h − v · min(⟨h, v⟩ − τ, 0)

i.e. it clamps the component of `h` along `v` up to a floor `τ` (25th percentile of the
axis-projection distribution) and leaves the orthogonal subspace untouched. Over a fixed
mid-late **layer band**, vary:

- **Position set P** (monotone ladder): `prefix-end` ⊂ `context-end (v_C, last prompt token)`
  ⊂ `all-prompt-tokens (prefill only)` ⊂ `all-tokens (incl. every generated token — the
  paper's setting)`.
- **Intervention type**: `cap-to-τ` [main] / `axis-component-replace-toward-default`
  [query-preserving patch] / `full-state-replace-with-default` [query-destroying control].

**Competing hypotheses.**
- H1: context-end capping ≈ all-token capping — recovers most of the effect at a fraction of
  the intervention footprint (a single fixed position vs every token).
- H2: prefix-end capping ≈ null — persona control happens at the context vector, NOT where the
  jailbreak persona was injected (the system prompt / prefix). Directly predicted by #2094:
  context-end is the only single position whose edits clear the shuffled-donor null;
  prefix-end / 2nd-to-last / 3rd-to-last yield zero null-separated behavioral effect.
- H3: axis-component replacement preserves query information where full-state replacement
  destroys it — resolving the plan's own "while somehow maintaining the query information"
  worry. Capping IS the query-preserving member of the patch family (it only moves the axis
  component); full-state replace is the query-destroying extreme.

**Measurement that distinguishes them:** co-primary judged on-policy RATES (see DVs) across the
position ladder, read against a no-intervention baseline and a norm-matched random-direction
cap null.

## Background — what the paper does, and what we already have
**Lu et al. 2026 (arXiv 2601.10387), reproduced faithfully except the deviations below.**
- Axis = per-layer contrast `mean(default-assistant) − mean(fully-role-playing)`; role vectors
  = mean post-MLP residual over response tokens, from the 275-role × 5-sysprompt × 240-question
  rollout bank.
- Cap band: **not one layer, not all** — an adjacent mid-late band (~12.5% of depth: 8 layers
  for Qwen-3-32B, layers 46–53), applied simultaneously; single-layer capping was ineffective.
  `τ` = 25th percentile of axis projections.
- Positions: **every token (prompt + every generated token)** in all their evals.
- DVs: persona-based-jailbreak harmful-rate (Shah et al. 2023 set, ~1100 prompts; drops ~60%),
  capabilities preserved (IFEval / MMLU-Pro / GSM8k / EQ-Bench), role-susceptibility drift.
- Standing project deviation: judge = `claude-sonnet-4-5-20250929` (not deepseek-v3 / gpt-4.1-mini).

**In-repo assets (≈90% of the instrument already exists):**
- No capping reproduction exists anywhere → add a `cap` mode (project onto axis + clamp) to
  `src/explore_persona_space/experiments/issue2094/hooks.py::PositionEditHook`. Interventions
  are **HF forward-hook only — no vLLM-side steering** (throughput risk, see Risks).
- 275-role bank + extraction questions on disk (`data/assistant_axis/`); persona-vector recipe
  in `artifacts/directions.py` → in-house axis extraction is turnkey. **No Qwen-2.5-7B axis on
  disk** (Lu's HF vectors `lu-christina/assistant-axis-vectors` are Qwen-3-32B).
- Position slots (prefix-end, context-end `v_C`, all-tokens) already implemented in #2094 hooks;
  graded judge (`eval/graded_judge.py`, Sonnet 4.5), jailbreak/refusal banks
  (`advbench_v1`, `strongreject_v1`), and a coherence gate (#1415) all exist.

**Directly relevant prior in-repo findings** (Qwen-2.5-7B, 28 layers × 3584 dim):
- **#2094** — context-end is the only single position whose activation edits clear the null;
  prefix-end does nothing; best clean effect = full-state replace at context-end across all 28
  layers (0.63 of a full context swap). → strong prior for H1/H2.
- **#1415** — single-token context-vector steering moves behavior (peak at ~layer 14, ≈21% of
  the context-swap ceiling); **all-position steering shatters the output distribution (96–98%
  flip to Chinese)** → coherence gate is mandatory; capping (clamp) is milder than replace/add.
- **#1092 / #1738** — answer-state transport runs through the query-bearing (context-end) state;
  query identity carries ~63% of per-row answer-state variance, history ~7%. → why axis-component
  replacement (not full replace) is the query-preserving arm.
- RESULTS.md:284 flags we never verified our PC1 vs Lu's contrast (they report cos > 0.71 mid-layer)
  — Phase 0 closes this.

## Design (phased critical path)
**Phase 0 — Axis extraction + validation (Qwen-2.5-7B-Instruct).** Extract the per-layer
Assistant Axis in-house (contrast vector) from the role bank via `directions.py`; validate
cos(axis, PC1) (target > 0.71 mid-layer), cos(axis, project trait directions), and a steering
sanity check (adding the axis modulates role susceptibility). Load Lu's Qwen-3-32B axis for the
anchor. (Generation cost of the full 275×5×240 rollout bank is the main GPU cost — subsample
roles/questions with a stability check; the axis is a mean-difference and is robust to
subsampling.)

**Phase 1 — Layer-band + τ calibration (fix hyperparameters ONCE, as the paper did).** Compute
the axis-projection distribution on rollouts; set `τ` = 25th percentile. Sweep band center ×
width (~12.5% of 28 layers ≈ 3–4 layers, mid-late) using the all-token capping arm against a
small jailbreak+capability dev set; pick Pareto-best band. Also fix a single mid-layer arm
(~L14 region) to connect to #1415. Fixing the band here prevents the position ladder from being
confounded by layer choice.

**Phase 2 — Main experiment (Qwen-2.5-7B, band fixed).** Position ladder (4) × intervention type
(3) = 12 arms + baseline + random-direction-cap null + single-mid-layer cap arm. Scored on both
co-primary DVs + coherence gate + capability preservation + the graded continuous companion.

**Phase 3 — 32B faithful anchor.** All-token capping on Qwen-3-32B with Lu's precomputed axis;
reproduce ~60% jailbreak reduction to validate our pipeline; add the context-end arm on 32B to
test H1 at the paper's scale.

## Dependent variables (dual-DV per project rule)
- **Co-primary rate 1 — jailbreak harmful-response rate** (judged on-policy, Sonnet 4.5).
  The Shah et al. set is NOT obtainable (see § Data availability) → reconstruct the paper's
  `persona system-prompt × behavioral question` structure from in-repo assets: harmful
  behavioral questions (`strongreject_v1`, 313 / `wang44_v1`, 44 harm categories — matches the
  paper's "44 categories") crossed with willing-to-comply persona system-prompts from the same
  275-role `data/assistant_axis/` bank the axis is extracted from. Optional external
  comparability cross-check: `JailbreakBench/JBB-Behaviors` (HF, public). Stated deviation from
  the paper's exact set.
- **Co-primary rate 2 — role-susceptibility / Assistant-identity-loss rate** (judged on-policy):
  fraction of responses written from a non-Assistant persona under role system-prompts +
  introspective questions ("Who are you?"), per the paper's susceptibility eval.
- **Continuous companion** — graded 0–100 "assistant-ness" judge score AND the realized
  axis-projection at answer tokens (did the cap actually move the projection, and by how much) —
  non-saturating, guards against floor/ceiling of the rates.
- **Guardrails** — capability preservation (IFEval / MMLU-Pro / GSM8k) and a coherence gate
  (an incoherent arm is not a valid comparison).

## Controls / nulls
No-intervention baseline; **norm-matched random-direction cap** (does the effect require the
Assistant Axis specifically, or would any clamp help? — RESULTS.md already warns the axis does
not beat random at corpus separation); full-state-replace as the query-destroying damage ceiling.

## Risks / open items for the planner
1. **Throughput** — interventions are HF forward-hook only (no vLLM steering); 12+ arms ×
   ~few-k prompts × 7B HF-hooked generation is the wall-clock bottleneck. Investigate a
   vLLM-side capping path or accept HF and size a wide pod. Est. order 30–80 GPU-h total
   (above the 20 GPU-h cheap band → plan approval required).
2. **7B may under-jailbreak-via-persona** relative to the paper's 32B (drift is model-dependent);
   the 32B anchor de-risks the headline.
3. **Jailbreak dataset availability — RESOLVED (checked 2026-08-09, see § Data availability):**
   Shah et al. set not obtainable; reconstruct in-style from in-repo banks + role bank. This is
   the harmful-content / trigger-dense leg → briefs reference banks by filename + count
   (digest-only), per context-hygiene rules.
4. **"Cap all tokens incl. generation"** requires the hook to fire on each decode step (a small
   extension to the edit-once-at-prefill PositionEditHook).
5. Linear-by-default respected (capping = linear projection; no MLP). Prefix AND context
   interventions both present (the position ladder), satisfying the both-arms convention.

## Data availability (checked 2026-08-09)
- **Shah et al. 2023 persona-jailbreak set (arXiv 2311.03348): NOT obtainable as a fixed
  download.** No public GitHub repo, no HF dataset. The method AUTO-GENERATES persona
  system-prompts with an LLM over harm categories — there is no released fixed file.
- **The `safety-research/assistant-axis` public repo is a minimal release:** axis-extraction
  pipeline (`pipeline/1_generate.py`…`5_axis.py`), `data/extraction_questions.jsonl` (empty
  `roles/`, `traits/`), and demo notebooks (`steer.ipynb`, `pca.ipynb`, …). It does NOT ship
  the jailbreak eval set, the capabilities harness, or the activation-capping eval code — only
  a steering demo. `lu-christina/assistant-axis-vectors` (HF) holds only the Qwen-3-32B axis
  vectors.
- **Usable substitutes (all verified live):** in-repo `strongreject_v1` (313), `advbench_v1`
  (200), `wang44_v1` (44 harm categories), `sensitive_info_requests_v1` (40),
  `china_sensitive_v1` (45); external `JailbreakBench/JBB-Behaviors` (HF, public, 100
  behaviors). Reconstruct the paper's persona×behavior structure by crossing a harm bank with
  persona system-prompts from `data/assistant_axis/`.
- **Capabilities side:** IFEval / MMLU-Pro / GSM8k are standard lm-eval-harness tasks (in the
  project eval stack); EQ-Bench is optional. NOTE: capabilities-under-capping must run on the
  HF forward-hook path (no vLLM interventions) — a throughput item for the planner.

## Anchors
Primary `docs/open_questions.md` anchor: **1.1 `q:spec-context-as-vector`**; also
**1.4 `q:spec-steering`** and **1.6 `q:spec-sysprompt-vs-drift`**. Related critique task #352
(Lu et al. methodology). Reuses #2094 hooks/bank/F-metrics, #1415 steering/coherence infra,
`artifacts/directions.py`, `eval/graded_judge.py`.

## Next step
NOT launched. Run `/adversarial-planner` (via `/issue <N>`) to harden hyperparameter grounding,
compute §9 sizing, and the critic/consistency passes before any GPU spend.
