---
name: envelope-brace-residue-and-intraround-rename
description: Prompt-file residue validation must exclude the marker-body envelope (git ^{{commit}} syntax is legit content); an intra-round symbol rename does not fire Step 3.75 on the net diff — attest it as a compose-time fact
metadata:
  type: feedback
---

Two compose nuances from #2198 r1 (2026-08-19, `kind: infra`, verify_report.py diff):

1. **Scope the `{{...}}` residue check OUTSIDE the marker envelope.** Marker
   bodies can legitimately contain double-brace pairs — #2198's `epm:results`
   note carried `cat-file -e <sha>^{{commit}}` (git revision syntax). A naive
   whole-file `re.findall(r'\{\{[^}]+\}\}')` flags it as unsubstituted
   composer residue and provokes a pointless re-compose. Validate residue on
   `content[:envelope_start] + content[envelope_end:]` only; anything between
   `---BEGIN/END IMPLEMENTATION MARKER BODY---` (and plan envelopes) is
   verbatim payload.

2. **Intra-round rename ⇒ Step 3.75 N/A, attested at compose time.** When a
   symbol is ADDED in commit 1 and RENAMED in commit 2 of the SAME round
   (#2198: `_resolve_companion` → `_resolve_companion_text`), the net
   round-base...HEAD diff shows only the fresh ADD — the Step 3.75
   module-exported-rename trigger does NOT fire. But a per-commit read (`git
   show <sha2>`) SHOWS a rename, so an unattested Codex may false-fire
   `symbol-rename-grep-absent` against the marker. Put a compose-time fact in
   the prompt: net-diff adds the symbol fresh, trigger N/A, plus any unrelated
   same-name symbols elsewhere (`scripts/codex_task.py::_resolve_companion`,
   monkeypatched by 4 test files — untouched, not dangling).

**Why:** both fired on the same compose; each costs either a wasted re-compose
or a false Codex blocker the reconciler has to strip.
**How to apply:** every compose's Step 3 validation (nuance 1); any round whose
commits include a rename of a symbol the same round introduced (nuance 2).
Related: [[infra-wf-fix-lint-gate-compose]].
