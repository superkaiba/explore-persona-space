---
name: Claude round-N fix-walkdown misses a registered-statistic operand mismatch in untouched code
description: On round-2/3 code-review, Claude's PASS verifies the prior round's fixes ✓ but doesn't re-check a registered statistic's OPERAND against its own doc-string/plan definition; Codex catches the wrong-vector substitution. Side with Codex (FAIL).
type: feedback
---

When a round-N (N≥2) code-review disagreement is Claude PASS vs Codex FAIL on a
NEW finding, and Claude's PASS rests on an item-by-item walk-down of the PRIOR
round's blockers all verified ✓, the gap is the same one Claude's walk-down
structurally cannot catch: a REGISTERED statistic computed on the WRONG OPERAND
in code the prior round never touched. Re-check the operand against its own
definition before crediting the PASS.

**Why:** Claude's round-N fix-verification walk-down checks "did each round-(N-1)
blocker get fixed?" — it does not re-derive every registered statistic from its
plan/doc-string definition. A wrong-but-stable operand (a different-but-adjacent
captured vector substituted for the mandated one) survives every fix round
because no prior blocker pointed at it. Codex's from-scratch read catches it.

**How to apply (the #667 r2 canonical case, `a37-frac-ctx-uses-tneg`):**
- The registered R3-1 stat `frac_ctx = ||v0(C) - v0(C_neg)|| / ||delta_contra||`
  (doc string `gate_chain.py:266-267`; plan v2 §3/§6/§11; theory-doc §1.5)
  expects `v0_cneg = v0(C_neg)` = base-CONTEXT activation under the negative
  persona (assistant slot, NO answer). The caller passed `v0_cneg = t_neg` =
  teacher-forced ANSWER-span activation under the negative persona. Adjacent
  object, different vector → numerator wrong → registered headline stat corrupt.
- The tell that it's worse than a one-line typo: GREP the extractor's persisted
  payload (NPZ `np.savez` field list) for the MANDATED operand. In #667 the
  correct `v0(C_neg)` was **not captured anywhere** — only `v0` (source-ctx),
  `t_pos`, `t_neg`, `c_C`/`c_Cp`. The analysis substituted the nearest available
  vector. No-correct-operand-captured ⇒ the fix is a re-EXTRACTION, not a
  rerun-of-analysis. Verify the operand is even in the data before believing any
  analysis-side read of it.
- Verdict: side with Codex, FAIL. A registered statistic on the wrong operand is
  Real-blocking ("false PASS lands a numerically wrong headline and propagates").

**Second canonical case (#664 r2, `sycophancy-refusal-store-battery-mismatch`):**
the operand mismatch generalizes from the wrong VECTOR to the wrong SCORING SURFACE
— the prompt battery the activations are extracted over. The registered PRIMARY
gate DV `ĝ^real = ŵᵀΔv(C')/ŵᵀŵ`, `Δv(C')=v⁺(C')−v0(C')` (plan §6.1 line 185)
requires the activations be measured "over the behavior's OWN scoring prompts (B6)"
(plan line 71: sycophancy → Sharma wrong-claims, refusal → XSTest/OR-Bench /
#390 pool; §13.1 + §6.5 make `v0(C')` on the own battery a required persisted
tensor). `issue664_extract_store.py:99` routed sycophancy/refusal to
`fetch_preregistered_probes(48)` (the generic Betley probes) instead → `v_plus`,
`v0`, `t_CB`, `r_plus`, `v_plus_probe`, `v0_probe` (extract_cell:274 →
_extract_all:385/455) all on the wrong surface for 2 of 5 content behaviors.
Two independent tells confirmed it: (1) the SAME repo's training-data builder
keys sycophancy positives by Sharma `wrong_claim` and refusal by the #390 pool
(`issue664_build_training_data.py`), so the eval/judge surface ≠ the extraction
surface; (2) the extractor's OWN docstring (`:88`) claimed "refusal/sycophancy
use their own claim/request pools" while the body at `:99` did the opposite —
a docstring-vs-body contradiction is a high-signal smell, grep the body, never
trust the docstring. Not analyzer-recoverable: the store captured the wrong
battery's activations, so it's a re-EXTRACTION (same conclusion as #667). Verdict
FAIL; persist a BLOCKER concern naming the line + the §6.1/B6 contract for the
Step 5c-ter dispatch gate. Bundled with a GROUNDED smoke-topology blocker
(`--live-judge-smoke` unreachable because the documented `--cells 1 --smoke`
canary is a `marker` cell ∉ `CONTENT_BEHAVIORS`, so `_live_judge_smoke` returns
N/A and the production fleet judge branch is never exercised — Claude's
"`--live-judge` forces `dry_run=False`" verified the library-direct call, not the
launcher path). General lesson: on a round-N PASS-vs-FAIL, re-derive each
registered DV's OPERAND **and** its measurement SURFACE/battery from the plan/
doc-string, and prefer grepping the persisted extractor payload + reading the
function body over trusting docstrings or the walk-down.

**Companion downgrade in the same adjudication (`frozen-r-cache-not-used`):** a
documented compute-deviation (greedy temp=0 vLLM REGEN of a frozen base response
R instead of LOADING the prior issue's R cache) is Real-but-non-blocking when (a)
greedy decode is deterministic on identical model+tokenizer+prompt+max_tokens, (b)
the deviation is carried in the plan's Scope/caveats, and (c) the only residual
risk is HF↔vLLM greedy kernel divergence (typically zero) + unverified cache
existence. Downgrade BLOCKER→CONCERN, PERSIST via `task.py raise-concern` (severity
CONCERN), name the analyzer as the surface that must carry it as a scope caveat.
Do NOT let it carry a round-N bounce on its own. Pattern: "load the cache or fail"
is over-hardening when a deterministic regeneration is already documented and the
cache existence is unverified (sibling of the codex_hardening_beyond_minimal_port
family, but here the over-hardening is a LOAD-vs-REGEN equivalence demand).
