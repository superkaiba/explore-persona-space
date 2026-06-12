---
name: Codex Critical blocker on sandbox-unreadable pod-side data artifact
description: Codex code-reviewer FAILs "cached-artifact-coverage-unverified" when a data file (e.g. R_train.json) is pod-side/auto-downloaded and unreadable from its sandbox; verify producer exit-assertion + pinned-revision download + parent-run empirical proof before believing FAIL.
type: feedback
---

Codex code-reviewer raises a Critical `cached-artifact-coverage-unverified` blocker
when a runtime data dependency is NOT in git (pod-side, auto-downloaded) and its
sandbox cannot read it — the verdict literally says "I could not read <path> in this
worktree." The mechanical crash claim is usually true (an unguarded consume of a
missing key WOULD crash), but the blocker framing collapses once you trace three
links the sandbox hid:

1. **Producer-side coverage assertion** — open the artifact GENERATOR (e.g.
   `r_generate.py:28` "EXIT assertion: bank ∪ {no_persona} ⊆ R_train.keys()"); if the
   producer asserts the exact coverage the consumer needs, the gap is structural-impossible.
2. **Pinned-revision download** — grep for `prepare_data_dependencies` /
   `hf_hub_download(revision=...)`; a pinned SHA means the bytes are frozen to what the
   parent run used (`contrastive_neg_geometry_530/data_deps.py:57`).
3. **Parent-run empirical proof** — if the parent worker on main (`i530_run_cell.py:375`)
   consumed the same artifact through the same builder as a HARD dependency and
   production succeeded, the child's DIAGNOSTIC-only consume is a strict weakening, not
   a regression. Bonus: if the authoritative bytes (HF pool) were THEMSELVES built from
   the artifact, coverage is proven by construction.

Failure mode if all three somehow fail: loud KeyError pre-training = wasted launch
(infra-class, recoverable), NOT silent corruption — verify the consume path is
diagnostic (WARN-only byte-compare; authoritative bytes come from elsewhere) vs
authoritative before classifying.

**How to apply:** for any Codex Critical citing an unreadable data file, do NOT take
"unverified" as "unverifiable" — trace producer assertion, revision pin, and parent
production. Companion patterns: `feedback_codex_passes_when_sandbox_blocks_data.md`
(interp-critic flavor), `feedback_codex_plan_section_in_scope.md` (the same #534
round also had Codex reading plan §13 ANALYZER-notes — preamble: "the analyzer
applies them over diagnostics the plan already computes" — as pipeline-code
requirements; flags-computed-and-surfaced satisfies the plan; auto-exclusion is a
standing rec, not a blocker).

Origin: task #534 round-1 (2026-06-09), blockers `cached-r-train-coverage` +
`manifest-exclusion-not-enforced`, both demoted; PASS.
