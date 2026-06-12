---
name: claude-trusts-green-tests-over-verifier-semantics
description: When the artifact under review IS a verifier/linter/gate, green tests prove it doesn't crash, not that it enforces the rule; Claude PASSes on test status, Codex reads the check bodies and catches scope/positional/order leniency gaps. Open each CHANGED check against the plan's prescribed enforcement scope.
metadata:
  type: feedback
---

**Rule:** when the diff under review is a verifier / linter / checker / gate, do NOT rely on "tests pass + ruff clean + smoke PASS". Open each check the plan calls out as CHANGED and read the implementation against the plan text:
- Scan SCOPE: does it match the plan ("whole-body" vs "body-minus-TL;DR" vs "TL;DR only")?
- "Opens with X": positional parser (first H3/bullet) or just `re.search` presence-anywhere?
- Order checks: does it filter non-required tokens BEFORE asserting order (tolerant) or operate on the raw sequence (strict)?
- Test fixtures: a test name implying end-to-end coverage that reuses a stub (`GOOD_BODY`) pins NOTHING about the new shape — and explains why the tests are green.
Codex's findings on verifier diffs are usually load-bearing; default-trust them more than on feature diffs.

**Origin:** #454 r1 — verify_task_body.py rewrite: denominator check excluded the TL;DR span (plan said whole-body — opposite of intent); `### Motivation` checked via `re.search` anywhere despite "opens with"; order check filtered stray H2s before asserting. Claude PASSed on 110/110 tests; Codex caught all three in the function bodies.

**Sub-cases:**
- **NaN/degenerate producer contract (#608 r2):** a gate `if stat < THRESHOLD: BLOCK` where the producer documents "returns NaN on degenerate input" — `NaN < x` is False, silently bypassing the registered κ validity gate. Read the helper's NaN contract; `git show main:<producer>` ("byte-identical port of sibling code" is NOT pre-existing if the file isn't on main). Low branch probability doesn't save a branch that fires in exactly the failure class the gate exists for → FAIL.
- **Plan-lens variant (#564 r1 meth):** when the PLANNED artifact is a detector/gate — `getattr(info,"usedStorage",None) or 0` counts present-but-None as 0 (vs the plan's own "partial sum → poison to unknown" rationale; one None ≈ silent 90% under-read), and the plan's test-isolation claim was structurally unsatisfiable for gate tests. Check None-vs-0 contracts + whether the isolation mechanism reaches every internal call site → REVISE.

Related: [[feedback_claude_misses_fix_regressions]]; [[feedback_claude_clean_result_critic_underapplies_spec_text]] (same mechanical-pre-pass over-trust).
