---
title: 'workflow-fix: lint cross-module sha-pin domain agreement (#1776/#1491 class)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b2f4d8aa5352
created_at: '2026-08-05T04:57:31Z'
has_clean_result: false
origin_prompt: 'P0 crash on #1491: index-domain sha pins asserted against prompt digests
  — second recurrence of the #1776 class on identical constants; prose-only enforcement
  in gotchas.md + artifact-reuse.md check (f) did not catch it through planner, critic
  ensemble, and five code-review rounds.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on
task #1491 (emitting agent: orchestrator, during the P0 crash-fix round).

## Goal

Add a `workflow_lint.py` check that flags a 64-hex sha pin duplicated across modules
whose binding names carry no agreeing domain token (INDEX vs PROMPT), so the
#1776/#1491 wrong-domain pin class is caught at lint time instead of in production.

## Workflow gap

- **Bug observed:** #1491 copied #1776's `fixed_split` INDEX-array sha pins
  byte-identical (`2e307fb2…` / `b9377786…`) and asserted PROMPT-string digests
  against them; the resulting assert could never pass on any input, and the crash
  reached production on pod-1491 after the planner, the critic ensemble, and five
  code-review rounds had all passed the code.
- **Why it is a workflow gap:** the rule already exists in TWO always-on/on-demand
  surfaces — `.claude/rules/gotchas.md` § "A sha pin lives in a DOMAIN" (which quotes
  the exact pin `b9377786…` from the #1776 incident) and
  `.claude/rules/artifact-reuse.md` check (f) — and both were ineffective because
  enforcement is prose-only. This is the SECOND task to trip the identical class on
  the identical constants (#1776 first, #1491 second), so the recurrence is
  demonstrated, not hypothetical. The propagation vector is mechanical: the pins are
  copy-pasted from a sibling module whose binding name does not encode the domain.
- **Confidence (emitter):** medium — the CLASS and its recurrence are proven; the
  precise lint predicate below is a starting point the planner should refine
  (see Constraints for the known false-positive risk).
- verified-at-filing: `grep -c 'sha_pin_domain\|SHA_PIN_DOMAIN\|check-sha-pin' scripts/workflow_lint.py`
  → **0 hits in 1 file** (absence-of-guard claim; the 0-hit in-target result IS the
  evidence). Corroborating scan, same session:
  `grep -rln '2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613' scripts/ src/ tests/`
  → **3 files** (`issue1482_error_analysis.py`, `issue1776_contexts.py`,
  `issue779_ffc_n50k_fits.py`), under the non-agreeing names `pinned_val_sha256`,
  `VAL_400_SHA`, `ORIG_VAL_SHA256` — none carrying a domain token. Also
  `git log --oneline --since='7 days ago' -- scripts/workflow_lint.py` → no landed
  sha-pin-domain fix. (2026-08-05 UTC)
- **State note (post-mutation, per clause (e)):** #1491's own occurrence was renamed
  to `VAL_400_INDEX_SHA` / `TEST_1000_INDEX_SHA` by the crash fix
  `9f43b03e430285c8fd1688ff8b22684b500e027f` earlier in this same session, so a grep
  today shows #1491 already domain-correct. The three sibling sites above are
  unmutated and are the live signal.

## Proposed change (candidate diff sketch — refine in planning)

```
+ # `--check-sha-pin-domain`: a 64-hex string literal bound in >=2 modules under
+ # `scripts/` must carry an agreeing DOMAIN token in each binding name (or an
+ # adjacent `# SHA_PIN_DOMAIN: <domain>` comment). Vocabulary: INDEX | IDS |
+ # PROMPT | BYTES | CONTENT. A duplicated pin whose sites disagree — or where no
+ # site declares a domain — FAILs.
+ def check_sha_pin_domain(*, scripts_dir: Path | None = None) -> list[str]:
+     # 1. collect 64-hex literals -> {hex: [(file, binding_name, lineno)]}
+     # 2. keep those appearing in >= 2 distinct modules
+     # 3. for each, extract a domain token from the binding name or an
+     #    adjacent SHA_PIN_DOMAIN comment; FAIL on absent or disagreeing tokens
+     # waiver: `# SHA_PIN_DOMAIN_EXEMPT: <reason>`
```

Bundle into the no-flags default run only if the false-positive rate on the current
tree is acceptable (see Constraints).

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (plus a pin test under `tests/`).
- The three existing non-domain-named sites are the natural first fixtures; whether
  to RENAME them (a cross-module rename touching `issue779_ffc_n50k_fits.py`,
  `issue1482_error_analysis.py`, `issue1776_contexts.py`) or to annotate them with
  `# SHA_PIN_DOMAIN: INDEX` comments is a planner call — annotation is far lower
  risk, since `ORIG_VAL_SHA256` is imported by name elsewhere.

## Constraints / invariants

- **Known false-positive risk the planner must measure BEFORE bundling into the
  no-flags default run:** legitimate duplicated hex (a model revision pinned in two
  places, a dataset commit sha, a test fixture) has no meaningful "domain" and would
  be flagged. Run the candidate predicate across the current tree, count the hits,
  and calibrate the vocabulary/waiver before making it a default gate. A gate that
  fires on every dataset revision sha is worse than no gate.
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`. Do not
  change any pin VALUE; this is a naming/annotation + lint change only.
- `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: b2f4d8aa5352

<!-- workflow-fix-candidate v1 -->
target_file: scripts/workflow_lint.py
bug_observed: #1491 copied #1776 fixed_split INDEX-array sha pins byte-identical and asserted PROMPT-string digests against them; the crash reached production after the planner, critic ensemble and five code-review rounds all passed it.
why_workflow_gap: The rule exists in gotchas.md ("A sha pin lives in a DOMAIN", quoting the exact pin) and artifact-reuse.md check (f), but enforcement is prose-only — so the identical class recurred on the identical constants, #1776 then #1491.
proposed_change: Add a workflow_lint check flagging a 64-hex sha pin duplicated across modules whose binding names carry no agreeing domain token (INDEX vs PROMPT), the #1776/#1491 wrong-domain pin class.
diff_sketch: |
  + def check_sha_pin_domain(*, scripts_dir: Path | None = None) -> list[str]:
  +     # collect 64-hex literals bound in >=2 modules under scripts/
  +     # require an agreeing domain token (INDEX|IDS|PROMPT|BYTES|CONTENT) in each
  +     # binding name or an adjacent `# SHA_PIN_DOMAIN: <domain>` comment
  +     # waiver: `# SHA_PIN_DOMAIN_EXEMPT: <reason>`
confidence: medium
related_task: #1491
<!-- /workflow-fix-candidate -->
