---
title: 'Fix C-axis preflight gates and re-run long-system stratum to measure the persona-framing
  factor (follow-up to #397)'
kind: experiment
tags: []
created_at: '2026-05-31T21:43:40Z'
has_clean_result: false
parent_id: 397
goal: 'Measure the persona-framing (C) factor''s matched-pair selectivity delta in
  the #397 single-token-marker recipe screen by fixing the two over-strict C-axis
  preflight gates (exact-token-match and Jaccard >= 0.15) that silently killed every
  long-system x neutral-framing cell, then re-running the full long-system (A=1) stratum
  self-contained (C=0 and C=1 in one run).'
---
---
title: "Fix C-axis preflight gates and re-run long-system stratum to measure the persona-framing factor (follow-up to #397)"
kind: experiment
parent_id: 397
goal: "Measure the persona-framing (C) factor's matched-pair selectivity Δ in the #397 single-token-marker recipe screen by fixing the two over-strict C-axis preflight gates (exact-token-match + Jaccard ≥ 0.15) that silently killed every long-system × neutral-framing cell, then re-running the full long-system (A=1) stratum self-contained (C=0 and C=1 in one run)."
---

# Fix C-axis preflight gates and re-run long-system stratum to measure the persona-framing factor (follow-up to #397)

## Goal

Measure the persona-framing (C) factor's matched-pair selectivity delta in the #397 single-token-marker recipe screen by fixing the two over-strict C-axis preflight gates (exact-token-match and Jaccard >= 0.15) that silently killed every long-system x neutral-framing cell, then re-running the full long-system (A=1) stratum self-contained (C=0 and C=1 in one run).

## Background / why

Parent #397 (single-token `※` marker + lr 1e-4 re-run of the #383 five-factor recipe screen) reported the persona-framing (C) factor as **entirely unmeasured**: all long-system × neutral-framing cells (A=1 × C=1) failed at training-data preparation. Diagnosis (reproduced locally with the real Qwen-2.5-7B-Instruct tokenizer) shows the failure is **two** over-strict gates in `factor_screen_365`'s C-axis preflight, not the one "padding bug" the #397 write-up surfaced:

1. **Exact token-match.** `render_nonpersona_prompt` + `run_c_axis_preflight` (line 244) demand the C=1 neutral prompt tokenize to *exactly* the same Qwen-token count as the paired C=0 persona prompt. The neutral prompt is assembled from atomic ~18-token clauses, so it cannot hit an arbitrary target exactly. Persona ≈ 344–378 tok; closest neutral settles 5–13 tok away → `CPaddingError` → `CAxisPreflightError`.
2. **Jaccard ≥ 0.15** (`MIN_C_JACCARD`, hidden behind gate 1 so the #397 analyzer never saw it). The closest-achievable neutral prompts score **0.086–0.138 for all three sources — all below 0.15**. Fixing only gate 1 whack-a-moles straight into gate 2.

Both gates are mis-calibrated for the verbose long-persona prompts (the persona prose carries ~100 unique non-lexicon words, inflating the Jaccard union; exact token equality is unreachable with atomic clauses). The neutral prompt is itself sound: it carries the domain lexicon terms, passes the role-adoption lint (no role phrases), and is length-matched to within ~8 tokens. Parent **#383 measured this exact C-axis** (+26.9 pp persona-framing selectivity, "n=24, long-system cells only") with the same personas/lexicons, so the contrast is valid; the current gates are simply stricter than what produced #383's number (the strict preflight landed in `867ff51e`, task #365 r2).

## The fix (shared library code → experiment-implementer + code-reviewer)

Touches `src/explore_persona_space/experiments/factor_screen_365/prompts.py` + `data_prep.py` (trunk-tier shared code; needs the code-reviewer ensemble):

- **`render_nonpersona_prompt`**: replace the oscillating padding loop with a deterministic closest-achievable scan over clause counts; accept the count minimizing |tokens − target| if within a tolerance (default ≈ one clause, ~20 tok); raise `CPaddingError` only if it cannot get within tolerance. Do NOT disable padding (target=None yields a ~900-tok neutral prompt → reintroduces a large length confound).
- **`run_c_axis_preflight`**: token-equality check → `abs(nonpersona_tokens − persona_tokens) ≤ tolerance` instead of `!= `. Convert the hard Jaccard gate to a recorded diagnostic with a low floor (the neutral prompt is domain-relevant by construction from `SOURCE_LEXICONS`, so the 0.15 hard gate is redundant; keep a small floor, e.g. 0.05, to still catch a pathological off-domain prompt). Record both the residual token gap and the actual Jaccard into the per-cell `prepared_dataset.json` manifest so the C0/C1 match quality is quantified in the write-up.
- Keep the role-adoption lint unchanged (it already passes).
- Add/extend unit tests in `tests/experiments/` covering: A=1×C=1 preflight now passes for all three sources; residual token gap ≤ tolerance; manifest carries the recorded gap + Jaccard; A=0×C=1 (dropped corner) still raises loudly if ever invoked.

## Setup (re-run)

- **Self-contained A=1 stratum**, per the scope decision: re-run all long-system cells — A=1 × B∈{0,1} × C∈{0,1} × D∈{0,1} × E∈{0,1,2} × 3 sources = **72 cells** — in ONE run so the persona-framing contrast (C=0 vs C=1) is fully internal, no cross-run seam with #397. Requires a new A=1-stratum cell filter on `dispatch_factor_screen_397.py` (e.g. `--only-a 1`) and a fresh output dir (`eval_results/issue_<N>/`) so `--resume` does not skip the already-complete #397 cells.
- **Everything else identical to #397**: marker `※`, lr 1e-4, warmup-ratio 0.10, seed 42, `--pos-per-source 400`, LoRA r=32/α=64, Qwen-2.5-7B-Instruct, #383 pools (`--reuse-pool-from-issue 383`; pools are marker/seed/framing-agnostic), train-matched eval panel (recipe-fix step 5b), 24-persona × 20-question × 5-completion vLLM eval, `max_new_tokens=2048`.
- **Raw-completion upload ON** this run (cheap add; addresses the #397 "context-free prefix" text-audit gap for the C=1 cells too). Persist + upload raw vLLM completions to the HF data repo per Upload Policy.

## Eval / analysis

Compute the persona-framing (C) matched-pair selectivity Δ over the A=1 stratum (n = 12 matched (B,D,E,source) tuples per the E0+E2 binary contrast, or full E triple for the ordinal view), with bootstrap intervals. Report C0 vs C1 source/bystander rates per loss-mask level. Re-state the cross-experiment sign-and-ordering test against #383 now that the C factor is available (the 5-factor extended ordering #397 could not compute).

## Success criterion

All 72 A=1 cells train and evaluate (no preflight deaths); the persona-framing matched-pair selectivity Δ is computed with its sign reported; the C0/C1 token-gap + Jaccard residuals are recorded in the write-up. (No effect-size threshold gate — this fills a measurement gap; the sign + magnitude are the deliverable.)

## Kill criterion

If A=1×C=1 cells still fail preflight after the fix (a third hidden gate), stop and report rather than disabling the preflight wholesale.

## Compute / pod

~8–10 GPU-h on 1× H100 (`lora-7b` intent; #397's 72-cell sweep was ~14h wall but carried heavy resume/rework overhead). Sequential train + vLLM eval passes with single-engine teardown between (mind the vLLM-teardown-OOM gotcha + MooseFS quota → `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1`).

## References

- Parent #397 (single-token marker recipe screen; this fills its unmeasured C factor).
- Grandparent #383 (measured C-axis +26.9 pp with same personas/lexicons; the target this re-run reproduces under `※` + lr 1e-4).
- #365 (`task-365-recipe-fix-v1`, commit `32ce24ef`; introduced the strict C-axis preflight at `867ff51e`).
- Module: `src/explore_persona_space/experiments/factor_screen_365/{prompts,data_prep}.py`; dispatcher `scripts/dispatch_factor_screen_397.py`.
