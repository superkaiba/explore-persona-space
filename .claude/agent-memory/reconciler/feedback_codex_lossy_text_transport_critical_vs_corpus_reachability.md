---
name: codex-lossy-text-transport-critical-vs-corpus-reachability
description: "#2654 r1: Codex Critical on text=True/errors=replace lossy decode feeding a string-equality check — mechanism real (live-probe it), but severity turns on CORPUS reachability (scan the actual input blobs for the required byte shapes) + git content-filter config + loud-vs-silent resolution; advisory-label consumer with diff shown = CONCERN, not blocker"
metadata:
  type: feedback
---

When Codex FAILs a round because a content-equality check reads its inputs
through `subprocess.run(..., text=True, errors="replace")` (universal-newline
translation + U+FFFD many-to-one collapse ⇒ byte-distinct inputs can compare
EQUAL), verify in three ordered steps before accepting Critical:

1. **Live-probe the mechanism** (2 lines: raw `A\r\nB\rC\nD` → `'A\nB\nC\nD'`;
   `b'\xff'`/`b'\xfe'` → identical `'X�Y'`). In #2654 both legs confirmed —
   and Claude's PASS-side Minor had named only the `errors="replace"` leg,
   missing the newline leg entirely (Claude under-describes, Codex
   over-tiers: both calibrations in one finding).
2. **Scan the ACTUAL input corpus for the required byte shapes** — not the
   hypothetical. #2654's inputs were committed `tasks/**/body.md` blobs:
   0/2,887 at HEAD and 0/7,265 blob versions over 120 days contained any CR
   byte or invalid UTF-8; `core.autocrlf=input` cleans CRLF at commit on the
   only machine that commits `tasks/`; the writers are Python text-mode UTF-8
   writes (structurally cannot emit either shape). Also check the
   self-guarding conjunction: whole-file EOL normalization equalizes BOTH
   halves post-translation, so the mode-only guard returns None — the false
   label needs body-ONLY ending bytes + a real frontmatter text edit in one
   window.
3. **Check loud-vs-silent resolution + what the artifact CLAIMS.** An
   advisory label ("never a verdict input", verdict STALE regardless) whose
   consumer still displays the full diff beside it resolves LOUD ⇒ CONCERN
   tier per [[stopping-rule-false-claim-overrides-nit-severity]] r13
   direction-sensitivity. And grep the deliverable for the fidelity word
   ("byte") — a test docstring accurately describing its own byte-identical
   fixture is NOT an unbounded byte-fidelity claim; only an unbounded
   claims-exceed-enforcement gloss triggers the false-claim escalation.

**Why:** #2654 r1 — Codex Critical `byte-faithful-endpoint-read` vs Claude
PASS. Binding reconcile: PASS, finding persisted as CONCERN with the cheap
fix direction (bytes-mode capture for the two `git show` content reads,
keeping log/diff/subject decoding fail-soft per the pre-existing
`errors="replace"` rationale the docstring records from #2384 r2 blocker 3).
Codex's proposed fix was sound; its severity was overreach — the acceptance
bar (task `## Acceptance`) nowhere required byte fidelity and all criteria
held by Claude's executed tests (Codex ran none: sandbox).

**How to apply:** any reviewer split where one side calls lossy text
transport (newline translation, replace-decoding, NFC normalization,
`.strip()`) standing in for byte equality a blocker: probe → corpus-scan →
loud/silent + claim-grep. Sibling overreach entries:
[[codex-hardening-beyond-minimal-port-contract]],
[[codex-test-covers-easy-case-not-realistic-failure-class]].
