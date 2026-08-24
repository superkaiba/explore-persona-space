---
title: Step 10d lint gate outer timeout fence is a launch-time constant while inner
  mapped-leg fences derive from TG_T — gate kills itself
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T14:43:46Z'
has_clean_result: false
origin_prompt: '#2327''s Step 10d gate died rc=124 with an empty verdict harvest after
  ~2.5h wall: outer fence 6000s vs inner mapped-leg fences TG_T+420=6480s and TG_T=6060s
  (TG_T from the gate''s own runtime selector call) plus 4x1800s lint legs = 19740s
  sum. Outer fence was smaller than either mapped leg individually; failure was determined
  at launch. Recipe derives inner fences but leaves the outer one to the launching
  agent as a constant.'
workflow: v1
---
---
kind: infra
tags:
  - workflow-fix
---

# Step 10d lint gate: the outer `timeout` fence is chosen by the launching agent as a constant, but the inner mapped-leg fences are DERIVED at runtime — so the gate routinely kills itself

## Goal

Make the Step 10d pre-push workflow-lint gate's OUTER fence derive from the same runtime quantity its INNER fences derive from, so an outer fence can never be smaller than the sum of the fences it wraps. Today the recipe leaves the outer fence to the launching agent's judgement while computing the inner ones itself, which makes a self-killing gate the default outcome on any branch whose own-diff maps to a large test selection.

## The defect

`.claude/skills/issue/steps/18-step-10d.md` § "Pre-push workflow-lint gate" computes the mapped invariant-test leg fences from the selector's `recommended-timeout-s` **at runtime**:

- `TG_T=$(grep -oE 'recommended-timeout-s=[0-9]+' … )` (work-script line ~267)
- mapped BASELINE leg: `timeout --kill-after=30s $((TG_T + 420))s …` (~line 327)
- mapped GATED leg: `timeout --kill-after=30s ${TG_T}s …` (~line 340)
- plus four lint legs at `timeout --kill-after=60s 1800s` each (~lines 97, 100, 238, 241)

The two mapped legs run SEQUENTIALLY, so the workload's worst-case duration is `(TG_T + 420) + TG_T + 4×1800 + overhead`. But the recipe gives the launching agent no rule for the outer fence and no formula tying it to `TG_T` — the agent picks a number. The recipe's own prose offers a "~9-12 min idle, 30-40 min under fleet load" figure, which invites a fence in the 1-2 h range and is unrelated to `TG_T`.

Result: whenever `TG_T` is large, the outer fence silently becomes the binding constraint and kills the gate mid-leg. The failure is **determined at launch**, before any work runs, and presents as `rc=124` with an EMPTY verdict harvest — which is correctly treated as INCONCLUSIVE, so the round cannot merge and must re-run the whole gate (including its fleet queue).

## Observed instance (#2327, 2026-08-24)

Measured values from the killed run:

| Fence | Value |
|---|---|
| `TG_T` (gate's own selector call, 25-file own-diff) | **6060 s** |
| mapped BASELINE leg | `TG_T + 420` = **6480 s** |
| mapped GATED leg | `TG_T` = **6060 s** |
| 4 × lint legs | 1800 s each = 7200 s |
| **Σ inner** | **≈ 19,740 s** |
| **outer fence chosen at launch** | **6000 s** |

The outer fence was smaller than EITHER mapped leg on its own. Wall arithmetic from file mtimes: leader start ~12:08Z, fleet queue 2,738 s then fail-open at ~12:54Z, rc sentinel written 14:33:56Z ⇒ workload elapsed ≈ **5,996 s**, i.e. the 6,000 s fence to within sampling error. Corroboration that it died mid-leg: the baseline leg's output file was still growing at 14:16Z (8,342 B, ~4,900 s into its 6,480 s budget) when the fence killed the group.

Cost: ~2.5 h wall (45 min fleet queue + 100 min doomed run) producing NO verdict, plus a full re-run at a corrected 21,600 s fence — which re-enters the fleet queue from zero.

The launching agent's stated reasoning is worth recording because it is the reasoning the recipe invites: "the recipe's structural worst case (every bounded leg wedged) is ~78 min = 4680 s; 6000 s is ≥1.28× that". That is a defensible reading of the recipe's prose and still wrong, because `TG_T` alone exceeded it. The agent was not careless; the recipe gave it no way to be right except by reading the work-script's fence arithmetic and re-deriving `TG_T` by hand before launch.

## Scope to investigate

1. **Derive the outer fence in the recipe, not at the call site.** The gate already computes `TG_T`; the runner should compute its own fence as `Σ inner + margin` (e.g. `(TG_T + 420) + TG_T + 4*LINT_LEG_T + PAD`) rather than accepting a literal. Since `TG_T` is only known after the selector call inside the workload, this likely means either (a) hoisting the selector call ahead of the fence, or (b) dropping the single outer fence in favour of per-stage fences that are each already derived — the inner legs are all individually fenced today, so the outer fence may be redundant belt-and-braces whose only live effect is this failure mode.
2. **Fail LOUD on an incoherent fence pair at launch.** Whatever the outer fence ends up being, assert `outer > Σ inner` and refuse to launch otherwise, with the arithmetic printed. A gate that cannot possibly finish should never start — 2.5 h of wall is too expensive a way to discover it.
3. **Write the verdict harvest on the timeout path.** The runner writes both rc sentinels on a fence kill (good) but the verdict file was never created, so a consumer sees `rc=124` + absent verdict and must reconstruct intent. Have the runner write `crash` (or a distinct `fence-kill`) plus the elapsed and the fence value, so the inconclusive state is self-describing.
4. **State the outer-fence rule in the brief-facing prose**, so a launching agent that does not read the work-script arithmetic still gets it right: "the outer fence must exceed the sum of the inner fences; the inner mapped-leg fences are `TG_T + 420` and `TG_T`, where `TG_T` is the selector's `recommended-timeout-s` for the branch's own-diff — read it before choosing a fence."
5. Check whether the sibling Step 9c gate and the Step 9a-ter inline payload lint gate share the constant-outer-fence-over-derived-inner-fences shape.

## Non-goals

Do not remove or shrink the inner leg fences — they are what keep a wedged leg bounded. Do not "fix" this by raising the recipe's suggested wall-time prose; the defect is a missing DERIVATION, not a wrong constant, and any new constant has the same failure mode on a branch with a larger `TG_T`. Do not weaken the INCONCLUSIVE-is-not-clean rule: treating `rc=124` as anything but inconclusive would convert this into a silently-unverified merge, which is far worse than the wasted wall.

## Provenance

Diagnosed by the #2327 orchestrator after its own Step 10d gate died at `rc=124` with an empty verdict; full arithmetic and mtime evidence are in that task's `epm:progress` fence-diagnosis marker. Confidence: high — every fence value was read first-hand from the launched scripts, and the elapsed matches the outer fence to within sampling error. Dedup target: `.claude/skills/issue/steps/18-step-10d.md` § Pre-push workflow-lint gate (fence arithmetic), distinct from #2523 (concern-row grammar) and #2533 (ownership-probe predicate).
