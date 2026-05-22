---
name: codex-conflates-marker-format-with-code
description: Codex twin code-reviewer escalates implementer marker-body formatting (e.g., inline backticks vs fenced code block in subsection (c)) to FAIL verdicts even when all code blockers are fixed. Discard such findings as out-of-rubric for code-reviewer.
metadata:
  type: feedback
---

When the Codex twin code-reviewer runs round 2+ on a clean diff (all round-1 blockers fixed at named line numbers, verified by both reviewers), it sometimes invents a single new "blocker" about the IMPLEMENTER'S MARKER REPORT FORMAT — typical examples:
- "Subsection `### (c) How to verify` uses inline backticks rather than a fenced ```bash``` block"
- "Fenced launch command is outside the (c) subsection"
- "Status header missing required field X"

These are real-but-non-blocking marker-shape observations, NOT code review findings.

**Why:** The `code-reviewer` rubric (`.claude/agents/code-reviewer.md`) covers the diff against base — code correctness, plan adherence, tests, lint, security. Marker-shape conformance is a Step-0.5 structural check the orchestrator owns separately. Escalating marker formatting to FAIL forces an unnecessary revision round on a clean code diff.

**How to apply:** When reconciling code-reviewer disagreements:
1. If both reviewers agreed all code blockers are fixed at the same line numbers, AND
2. Codex's sole FAIL finding is about implementer-report formatting (not about code, tests, plan, or security)
3. Classify the Codex finding as Real-nonblocking (Discarded weight), issue PASS verdict, and surface the formatting observation as a Standing Recommendation, not a blocker.

This is the inverse of [[feedback_claude_underclasses_silent_failures]]: Claude under-classes real runtime bugs as CONCERNS; Codex over-classes prose nits as FAIL. The pattern is the same — each side's calibration is biased and the reconciler corrects to the rubric.

Observed: task #375 round 2 (2026-05-21) — 11/11 code blockers fixed, Codex FAIL'd on subsection (c) using inline backticks instead of a fenced block. Adjudicated PASS.
