---
name: codex-conflates-marker-format-with-code
description: Codex code-reviewer escalates implementer-marker FORMAT (subsection style, label ordering, smoke-evidence presentation) or stale-file reads of marker EXISTENCE to FAIL and declines to review the diff. Out-of-rubric — discard, lean on Claude's substantive review with independent spot-checks.
metadata:
  type: feedback
---

**Rule:** when Codex's FAIL is exclusively a Step 0.5 mechanical-contract trigger AND its Major/Minor sections are empty ("diff not reviewed"), classify the finding Unverified/Discarded, independently verify 2-3 load-bearing spot-checks of Claude's substantive review against the actual diff, and PASS if those hold. Marker shape/existence is the orchestrator's Step 0.5/5c-bis surface, not a code-review finding; Step 0.7 rule 1 forbids FAIL on a `marker-shape`-only tag with no `substantive` tag, and Step 0.5 carves "present but imperfect ordering = CONCERNS, never standalone FAIL".

**Flavors:**
- **A — marker-shape nit:** inline backticks vs fenced block, H3 labels renamed/permuted from canonical `(a)-(d)` while all four sections' CONTENT is present with copy-pasteable commands. Incidents: #375 r2; #391 r3; #401 r2; #506 r4 (cap-3 round — another label-only FAIL would have burned the override on presentation; exactly the gate-hopping Step 0.5/0.7 prohibits). All PASS + standing rec to use canonical ordering.
- **B — stale-file false alarm:** Codex reads the WORKTREE's `tasks/<status>/<N>/events.jsonl`, sees the implementation marker "missing", FAILs without reviewing. Verify against canonical main-branch state: `jq -r 'select(.kind=="epm:experiment-implementation")' <main>/tasks/<status>/<N>/events.jsonl | tail`. Incident: #382 r1 (marker existed, note_len=12370). PASS.
- **C — smoke-evidence presentation nit:** evidence presented narratively rather than fenced-command + exit-0 + digest, while the smoke demonstrably exists and reproduces. Incident: #451 r1 — reconciler independently reproduced the preflight numbers character-for-character + re-ran the 9/9 test file + verified cherry-pick parity; FAIL was about FORMAT, not absence. PASS.

This is the inverse of [[feedback_claude_underclasses_silent_failures]]: Claude under-classes real runtime bugs; Codex over-classes prose nits / stale reads. The reconciler corrects each side to the rubric.
