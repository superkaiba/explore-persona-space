---
name: smoke-arch-verdict-binds-per-arm-rows-only
description: "Step 0.55: PASS_UNIFIED's no-FALLBACK binding covers per-arm-resolution rows ONLY — FALLBACK <reason> in resume-matrix:/production-outroot-unit: sub-blocks is legal Step 0.6 attestation vocabulary (CONCERNS at most), not a marker-shape contradiction (#2378 R2 g1)"
metadata:
  type: feedback
---

Rule: when adjudicating a smoke-architecture marker's verdict↔row consistency
(Step 0.55), scope the "any FALLBACK row under PASS_UNIFIED/PASS_CANARY is a
marker-shape blocker" rule to the `per-arm-resolution:` rows ONLY. The
`resume-matrix:` and `production-outroot-unit:` sub-blocks use the Step 0.6
attestation vocabulary `REAL / FALLBACK <reason> / N/A` by design — a FALLBACK
there WITH a reason + closest-proxy named routes to CONCERNS at most
(code-reviewer-section-reference.md § Step 0.55: "every per-arm row must read
REAL or N/A"; § Step 0.6 resume-matrix/production-outroot-unit bullet:
"Missing either → FAIL smoke-run-missing … unless (d) explains why + names the
closest proxy — then CONCERNS").

**Why:** on #2378 R2 g1 the round-1 Codex twin FAILed PASS_UNIFIED as
"contradicting two FALLBACK rows" that were both sub-block attestations
(`p4_topup g2a_report.json re-entry` — fail-loud leg fired REAL, report-consuming
leg pod-side-only; `production-outroot-unit: FALLBACK` — P1-pilot-is-smoke by
plan design), and the R2 brief inherited that presumption ("if those rows still
read FALLBACK … FAIL marker-shape"). The spec grammar says otherwise; the
correct adjudication was CONSISTENT + a CONCERNS note that the carve-outs ride
to Step 6d.0/production.

**How to apply:** any Step 0.55 audit where a FALLBACK token appears outside
the per-arm span. First run `task.py check-smoke-arch-registry <N> --repo-root
<worktree>` ([[smoke-arch-marker-2176-grammar-pitfalls]]), then check which
SPAN each FALLBACK sits in before tagging `marker-shape`; sub-block FALLBACKs
need reason + proxy (else Step 0.6 `smoke-run-missing`, still not marker-shape).
A brief's presumed adjudication is an input, not the spec — cite the
section-reference text when overriding it.
