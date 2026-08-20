---
name: amendment-round-stale-literal-sweep
description: Amendment-round splits over surviving old-value literals — grep the numerics yourself across ALL amendment-dependent sections (§9 sizing included); adjudicate each literal by whether the registered decision path binds to a realized quantity elsewhere (fail-loud vs silently-wrong)
metadata:
  type: feedback
---

On a targeted-amendment plan round (vN = vN-1 + anchor/replace edits changing one
gate + its dependent arithmetic), the two statistics reviewers can agree on every
substantive sub-question and split ONLY on surviving stale literals. #823 v11 shape:
Claude's literal sweep found §4.1's `max_tokens 1024` but missed §9's fixed
`(10,000 × 4,990)` bootstrap GEMM shape (v10's expected mask; v11's is ≈4,683);
Codex found both but inflated the §9 one to "invalidates the CI-driven headline
lattice".

**Why:** Claude sweeps tend to stop at operational sections and skip §9 sizing
statements; Codex flags every hit but assigns worst-case severity without tracing
reachability. Both errors are cheap to correct by direct verification.

**How to apply:** (1) Run the numeric grep yourself (`grep -n '4,\?990'` style, old
AND new values) over the WHOLE plan — §9 draw-battery/sizing statements are inside
the amendment's dependent-arithmetic scope. (2) Severity per literal turns on two
probes: does the REGISTERED decision path bind to a realized quantity elsewhere
(#823: §6 registered the bootstrap "over the persisted per-context (ss_res, ss_tot)
arrays" + a fail-loud realized-key set-check, so the stale §9 shape could only
dimension-fail loudly — the wrong-population branch was unreachable since no
old-n-row array exists), and is the statement's CONCLUSION sensitive to the value
(the MAC estimate moved 1.96e10→1.84e10 — insensitive)? Fail-loud + insensitive ⇒
Real-non-blocking CONCERN (consistency fix), not a BLOCKER. (3) An OPERATIONAL
literal an implementer would follow (a generation-config `max_tokens` in the
provenance spec) that restores a pilot-measured, manipulated-variable-correlated
censoring IS in-lens blocking for Statistics (measurement validity) even when the
Methodology lens already REVISEd on the same line — the [[cross-lens-defect-refiled-per-lens]]
out-of-lens discard does NOT apply to censoring/measurement literals; redundant
binding is harmless (the union dedupes). Origin: #823 plan-v11 statistics reconcile
(REVISE; Codex verdict upheld, its §9 severity adjudicated down).
