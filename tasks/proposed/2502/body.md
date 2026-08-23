---
title: Fit context→answer map on a mega-diverse weird-behavior corpus, replicated
  on Qwen2.5-7B-Instruct + Qwen3.5-9B
kind: experiment
tags: []
created_at: '2026-08-23T17:17:32Z'
has_clean_result: false
origin_prompt: I want to try to fit a mapping on a very very large and VERY VERY diverse
  set of contexts and answers. It should be both on qwen2.5-7B instruct and on some
  newer qwen ideally that has no thinking. Get a subagent to find a large diversity
  of datasets including weird behaviors to train on. Then get another one to propose
  the other model.
workflow: v1
goal: 'Fit the context→answer activation map M_{C,A} on a very large, maximally diverse
  context/answer corpus (heavily weighted toward weird / OOD / red-team / jailbreak
  regimes), and test whether the map replicates across model generations: Qwen2.5-7B-Instruct
  and Qwen3.5-9B (thinking disabled). An answer = held-out R^2 plus the mandatory
  identity+learned-bias baseline and kNN-retrieval reads for the fitted map, measured
  under massive context diversity and compared across the two models.'
---
## Goal

Fit the context→answer activation map M_{C,A} on a very large, maximally diverse context/answer corpus (heavily weighted toward weird / OOD / red-team / jailbreak regimes), and test whether the map replicates across model generations: Qwen2.5-7B-Instruct and Qwen3.5-9B (thinking disabled). An answer = held-out R^2 plus the mandatory identity+learned-bias baseline and kNN-retrieval reads for the fitted map, measured under massive context diversity and compared across the two models.

## Provenance

Proposed 2026-08-23 from an interactive chat. Two scout subagents ran the scoping:
a dataset scout (diverse context/answer corpora, weird-behavior-weighted per Thomas's
steer) and a model scout (newer non-thinking Qwen). This body is pure capture — NOT
yet planned or dispatched. Next execution step is `/issue 2502` (which runs the
adversarial-planner + critic stack), never inline. Scale target and gated-access are
open decisions for Thomas (see § Open decisions).

## Models (decided)

- **Model A (fixed):** `Qwen/Qwen2.5-7B-Instruct` — dense, 28 layers x 3584 hidden.
- **Model B (Thomas's pick, 2026-08-23):** `Qwen/Qwen3.5-9B` with `enable_thinking=False`
  — newest Qwen generation. 9.65B, text backbone 32 layers x 4096, dense MLP, Apache-2.0.

**Known confound Thomas accepted at pick time:** Qwen3.5-9B is multimodal (vision tower
baked in) and uses hybrid attention — only 8 of 32 layers are full softmax attention,
the other 24 are Gated-DeltaNet linear attention, plus a 1-layer MTP head. A per-layer
map comparison against Qwen2.5's 28 uniform-softmax layers is architecturally confounded.
**Planned mitigation:** restrict the cross-model per-layer comparison to Qwen3.5-9B's 8
full-attention layers (the layers most comparable to Qwen2.5's attention), rather than
treating all 32 layers uniformly. No Qwen3-era model matches 3584 hidden, so a hidden-dim
mismatch is unavoidable regardless of pick.

**Thinking-disable mechanics (verified):** `enable_thinking=False` makes the chat template
emit a pre-closed empty `<think>\n\n</think>\n\n` pair in the prompt (deterministic, not
sampled); default (`True`) always emits a `<think>` block, so the flag must be set on
EVERY call. Activation-position bookkeeping must account for the empty think-pair tokens
at the start of each assistant turn. Card's vLLM recipe:
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`; offline path = apply
the template ourselves with `enable_thinking=False` then `LLM.generate`.

Runners-up considered and rejected: `Qwen/Qwen3-4B-Instruct-2507` (zero thinking-mode
confound, dense uniform attention — but only 4B / 2560 hidden) and `Qwen/Qwen3-8B` with
`enable_thinking=False` (best 7B size match, dense uniform attention). Qwen3.5/3.6/3.8 all
share the multimodal hybrid-attention class; nothing dense-uniform exists below 9B newer
than Qwen3. Full model report in events (model-scout, 2026-08-23).

## Data plan (dataset menu — proposed, to be finalized at plan time)

Already the backbone (wired in repo): `lmsys/lmsys-chat-1m`, `allenai/WildChat-1M`,
`HuggingFaceH4/ultrachat_200k`, `lmsys/toxic-chat`, `allenai/tulu-3-sft-mixture`,
`Anthropic/hh-rlhf`, `proj-persona/PersonaHub`. Realism tiers per
`.claude/rules/data-realism.md` (T1 real-world … T4 programmatic). Contexts supply the
map's input side; answers generated on-policy from each model (shipped answers a bonus
for teacher-forced variants only).

**Scale: TARGET = 150k contexts (Thomas, 2026-08-23).** Sits at the top of the linear
map's held-out-R² plateau (n_train ≈ 10k–50k knee, flat to ~1M; #779 sweep, dedup-audited
by #1775 — see the `epm:progress` scaling-evidence note). Going higher buys ~nothing for a
linear fit; 150k gives headroom for the diversity axis + optional nonlinear fits. Per-source
budgets below scale to the 150k target.

**Proposed top-~12 shortlist (weird/OOD-weighted, ~55% weird):**

1. `allenai/WildChat-1M-Full` (gated; keeps toxic/NSFW/jailbreak turns) — fallback
   `allenai/WildChat-4.8M` (ungated). T1.
2. `TrustAIRLab/in-the-wild-jailbreak-prompts` (15,140 prompts; real DAN/base64/leet). T1.
3. `Anthropic/hh-rlhf` red_team_attempts (multi-turn human red-team). T1.
4. `lmsys/lmsys-chat-1m` (fresh disjoint rows, moderation-flagged over-sampled). T1.
5. `allenai/wildjailbreak` (gated) + `JailbreakBench/JBB-Behaviors` + `walledai/AdvBench`
   + `walledai/HarmBench` (adversarial + matched-benign twins). T2/T3.
6. `Anthropic/model-written-evals` + `meg-tong/sycophancy-eval` + `cais/MASK`
   (sycophancy / deception / trait-persona). T2/T3.
7. `bench-llm/or-bench` + `allenai/coconot` + `walledai/XSTest` + `LibrAI/do-not-answer`
   (over-refusal boundary). T2/T3.
8. `PygmalionAI/PIPPA` (filtered) + `Norquinal/OpenCAI` (real roleplay-as-someone-else). T1.
9. `euclaise/writingprompts` + `Gryphe/Opus-WritingPrompts` + `google/IFEval`
   + `INK-USC/riddle_sense` + `jdpressman/retro-ascii-art-v1` + text-adventures
   (surreal / constrained-form / cipher-ASCII / text-game). T1–T4.
10. `PKU-Alignment/BeaverTails` + `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
    + `AI-Secure/DecodingTrust` (harm-category + trust-perspective). T3.
11. `allenai/tulu-3-sft-mixture` stratified by internal source tag (broad-instruction
    balance so the map isn't all-weird). T2.
12. Persona/system-prompt prefixes: `proj-persona/PersonaHub` + `nvidia/Nemotron-Personas-USA`
    + `fka/prompts.chat`, crossed with a fixed query bank. T1/T3.

Domain fillers (in-distribution corners): `ise-uiuc/Magicoder-OSS-Instruct-75K`,
`AI-MO/NuminaMath-CoT`, `qiaojin/PubMedQA`, `lavita/ChatDoctor-HealthCareMagic-100k`,
`nguha/legalbench`, `deepmind/pg19`/`pile-of-law` long-prefix excerpts. Emotional-support
register: `facebook/empathetic_dialogues`, `thu-coai/esconv` (NC — research use).

**License/handling flags carried forward:** NC/SA/unknown-license sets (toxic-chat,
BeaverTails, PersonaHub, DecodingTrust, several roleplay sets, `mlabonne/harmful_behaviors`,
`teknium/OpenHermes-2.5`, `cais/MASK`, ChatDoctor) usable as research artifacts (Tulu-3
takes this posture) but must NOT be redistributed inside any released dataset artifact.
Harmful item text stays under `guard_harmful_bank_read.sh` + trigger-dense-brief rules
(filename + index only; contexts embeddable for on-policy generation, never paged into an
agent context). Not on HF (GitHub/ParlAI only): LIGHT, `jujumilk3/leaked-system-prompts`,
Apollo sandbagging evals; no clean standalone cipher/base64/leet corpus (present inside
WildChat + jailbreak sets).

## Open decisions

1. **Scale target — DECIDED: 150k contexts** (Thomas, 2026-08-23). At the top of the
   linear-map R² plateau; see the Data plan § and the scaling-evidence `epm:progress` note.
2. **Gated-dataset access.** (STILL OPEN.) WildChat-1M-Full, wildjailbreak, HarmBench, AdvBench,
   DecodingTrust, lmsys-chat-1m, chatbot_arena all need access requests under `superkaiba1`.
   Request them, or run ungated-only (WildChat-4.8M fallback for slot 1)?
3. **Map regime + measurement** to be locked at plan time per project rules: linear map
   by default; dual-DV / on-policy-completion / identity+learned-bias baseline + kNN
   retrieval all mandatory; cross-model comparison restricted to matched (full-attention)
   layers for Model B.
