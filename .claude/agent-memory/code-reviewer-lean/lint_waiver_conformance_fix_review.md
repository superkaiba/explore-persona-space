---
name: lint-waiver-conformance-fix-review
description: Review recipe for lint-conformance fix commits (waiver comments, guard-tuple broadenings) + the dead-session promised-gate-note residual
metadata:
  type: feedback
---

Rule: for a "lint-conformance fix" commit, certify three things yourself: (1) the WAIVER
RECOGNIZER's placement semantics in the check fn (e.g. `_prod_import_lint_waiver_present`
at workflow_lint.py:9689 — waiver on the import's first physical line OR the immediately
preceding NON-BLANK line, reason ≥ 10 chars, site-only) against where the diff put the
comment — a line-above waiver with a blank between is inert; (2) for an exception-tuple
broadening, the SUBCLASS relations (UnicodeDecodeError ⊂ ValueError, not OSError/
JSONDecodeError; the decode fires in `read_text`, not `json.loads(str)`) AND that the
catch routes to a designed repair path, not a silent swallow; (3) re-run the targeted
check flags yourself (`--check-json-guard-unicode` fast; `--check-prod-import-lockfile`
~4.5 min — background it past the 120s Bash cap).

**Why:** #2587 r2 g7 — both fixes were correct, but the claims were only certifiable by
reproducing the instruments; the waiver-placement grammar is the failure point a diff
read alone cannot settle.

**How to apply:** also check dead-implementer rounds for PROMISED-but-never-posted gate
results: a marker (c) saying "full no-flags re-run in flight, result in a follow-up
epm:progress note" with no such note in post-marker events is a CONCERN (surface, don't
block — targeted re-checks + the Step 10d landing gate bound the risk). In split-review
sub-scopes the brief's "do NOT mutate task state" overrides `raise-concern` — record the
concern in the verdict FILE and say it is unpersisted.
