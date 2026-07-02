---
name: Codex catches production-only crash/sign bug Claude's smoke never exercises
description: When Claude PASSes on a green smoke but Codex FAILs on a registry-vs-persona-key KeyError / sign-blind metric on the un-smoked production path, trace the production launch path and uphold FAIL.
type: feedback
---

When Claude code-review PASSes on the strength of a green smoke and Codex FAILs
on a bug on the PRODUCTION path the smoke structurally never reaches, go to the
launch path and trace it — do NOT defer to "the smoke passed." Two recurring bug
CLASSES live in this gap; both are Real-blocking and carry FAIL on their own.

**Why:** the unified/CPU smoke deliberately uses a single in-registry source and
stubs the judge, so it cannot exercise (a) the alternate-key resolution branch or
(b) the real judge dispatch. Claude reads the green smoke as coverage of the whole
dispatcher; it is not. Claude tends to UNDER-classify the production-only crash as
a "fragile but correct" robustness note — that under-classification is the tell.
(11+ incident family with #518/#522/#504 et al. on the broader "Claude under-classes
silent failures" line; this file is the smoke-coverage-gap specialization.)

**How to apply:**

1. **Registry-vs-alternate-key resolver gap (KeyError on the production launch).**
   The smoke source is an in-registry cid; a production arm feeds a key from a
   DIFFERENT namespace (a PERSONAS key, a config alias, a derived id) that only a
   specialized resolver branch handles. If the production code does a bare
   `registry[source]` / `dict[key]` lookup on the alternate-key path while the
   resolver branch exists ONLY in one phase (e.g. P0's `_resolve_ctx`), the
   production run KeyErrors. Confirm by: (i) the alternate keys are NOT in the
   registry (read the key-list constant + its docstring); (ii) the selector returns
   that alternate key; (iii) the bare lookup is on the launch path (grep the
   `--phase run` / production entry → the lookup line); (iv) the resolver wrapper
   is absent there. A sibling silent-bypass often rides along: a hard invariant
   assert (disjointness, coverage) built from `[d[c] for c in keys if c in d]`
   SILENTLY DROPS the alternate keys via the `if c in d` guard, so the assert never
   sees the very object it was hardened to check. Fix = ONE shared resolver used by
   build/run/eval AND the assert. (#641 r1: Arm-B matched neutral was a PERSONAS key;
   `phase_run`/`_eval_checkpoint` did bare `registry[source]` → KeyError; the §4.7
   disjointness assert filtered it out via `if c in registry`. FAIL.)

2. **Sign-blind decision rule in a symmetric metric (directional verdict from a
   magnitude test).** A classifier whose verdict is DIRECTIONAL (e.g. "X plateaus
   BELOW Y" → ceiling) but whose code uses sign-blind primitives —
   `excludes_zero = (lo > 0) or (hi < 0)` plus `gap = min(abs(lo), abs(hi))` — will
   return the directional label for the WRONG sign. The function's own docstring is
   the oracle: if it states the convention and which sign means the label, and the
   code admits both signs, it's a real bug. Fix = pin the sign (`hi < 0 and abs(hi)
   >= GAP`), not `excludes_zero` + `min(abs(...))`. (#641 r1 M5: `classify_h5`
   returned H5b for a POSITIVE asymptote gap; convention `(resistant − non-resistant)`
   means H5b needs `hi < 0`. FAIL.) Same family as the sign-blind |ρ| decision-rule
   memory.

3. **Judge/scorer hardcoded to the wrong rubric for one branch.** A dispatcher that
   loops MULTIPLE probe categories through a SINGLE judge helper that hardcodes one
   rubric (`judge_request_for_row("em", …)`) scores every category with that rubric.
   If the plan calls for a category-specific judge and the result feeds selection
   (matched control, threshold), it's a corrupted covariate — Major, Real-blocking.
   The smoke misses it because `--smoke` stubs the judge. (#641 r1 M4.)

4. **Silent aggregate-gate variant of item 1 (the resolver-fix backfire).** The
   round-1 fix for item 1 — making the source resolver TOTAL so a PERSONAS-key
   source resolves instead of `KeyError`-ing — can CONVERT the same bug into a
   SILENT failure two rounds later if a DOWNSTREAM aggregate gate still keys on
   the registry-cid constant. Pattern: `phase_run` carries the raw `--sources`
   token through build/train/eval and writes it as the records key
   (`"source": source`); the aggregate loader keys by `r["source"]`;
   `phase_aggregate` gates the headline read on `if HEADLINE_CONST in records`
   where `HEADLINE_CONST` is the registry cid (`sp_teacher_ho`). Launching with
   the PLAN-BODY / package-docstring slug (`kindergarten_teacher`, a valid
   PERSONAS key) resolves via the TOTAL resolver (no crash), keys records under
   the persona slug, the gate misses, and the load-bearing headline silently
   drops to `None` with no error. This is a documented-contract-vs-implementation
   divergence (plan body + `__init__.py` framing use the persona slug; the code
   constant + dispatcher docstring use the registry cid — the two documented
   conventions CONFLICT), so "the orchestrator will type the right slug" is
   contract-laundering, not a correctness guarantee — uphold FAIL. The cap-3
   FAIL routes to the trivial one-line fix (canonicalize the headline source to
   the constant at records-keying / accept both keys in the gate / assert the
   source resolves to the constant), a tractable-bug pivot, NOT an
   `/adversarial-planner` re-plan. (#641 r3, reconciled FAIL over Claude PASS:
   same source-key-namespace class as r1 item 1, third recurrence in the task.)

When confirming items 1/2/3/4 in code, the C3-style "smoke carve-out lacks
command+exit+digest" mechanical item does NOT need adjudication and does NOT trigger
the Step 5c-bis strip — the substantive findings carry FAIL regardless.
