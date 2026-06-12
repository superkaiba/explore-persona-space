---
name: cross-loop-ci-conflict
description: Codex clean-result-critic FAILs the bracketed-CI form ([low, high]) in TL;DR prose, but the upstream interpretation-critic explicitly asked for it as a powered-null statement. The Lens 7 "test-definition" exception + reconciler memory entry G save the CI; do not let Codex strip quantitative content the prior reviewer required.
metadata:
  type: feedback
---

# Rule

When Codex clean-result-critic flags bracketed-CI form (`95% CI [−0.22, −0.02]`, `95% CI [+1.79, +2.03]`) in TL;DR prose / captions as a Lens 7 violation, BEFORE classifying as Real-blocking, check whether the upstream interpretation-critic (round-1 verdict marker) EXPLICITLY required the CI as part of a powered-null statement or a non-detection-honesty justification.

If yes, the bracketed CI falls under SPEC Lens 7's "test-definition" exception ("a 'Why this test' sentence inside a result H3 that explicitly names the CI as part of the test definition") AND reconciler memory entry G (added 2026-06-03 from task #468):

> When the CI is load-bearing for non-detection honesty (e.g. paired-difference 95% CI to communicate "we can't detect a difference" vs "the recipes are tied"), keeping the bracket is defensible. DISCARD unless the same value also appears as `value ± err` somewhere.

**Why:** The two loops have different rule cultures. Interp-critic enforces honesty-of-claim (require the CI numbers so the powered-null isn't naked); clean-result-critic enforces register (no bracketed CIs in prose). Without cross-loop awareness, the analyzer gets bounced between contradictory demands and the powered-null statement gets stripped on round-N+1 to satisfy clean-result-critic, then re-added on round-N+2 to satisfy interp-critic.

**How to apply:** In the reconcile, after listing Codex's bracketed-CI hits, grep events.jsonl for `epm:interp-critique` markers (round 1 in particular) and search for "powered", "CI", "[−0", "[+1", or the specific bracketed values. If interp-critic's recommendation text uses the exact bracketed form, DISCARD Codex's finding with explicit rationale citing both the spec exception and the cross-loop conflict. Add a "Do NOT remove" line to the binding fix list naming the line numbers that must keep their quantitative content.

Suggest a register-only cosmetic alternative ("95% CI from −0.22 to −0.02" spaced form, no brackets) as optional but not required. The numbers stay either way.

## Caption carve-out — no interp-critic trail needed (added 2026-06-09, task #509)

Bracketed CIs in a FIGURE CAPTION are out of Lens 7's FAIL scope entirely, independent of any interp-critic requirement. Two spec anchors:

1. Lens 7's FAIL condition (clean-result-critic.md:210-211) enumerates "result-H3 setup/read paragraphs or the Confidence sentence" — captions are NOT in the enumeration, and the banned-list line says "chart error bars fine".
2. clean-result-critic.md ~:1162 verbatim: "**Don't suggest stripping numbers from Details or the figure caption** — the design narrative carries the precision-laden expansion."

When the caption's CI text DECODES error bars actually drawn on the chart ("Error bars = 5000-rep bootstrap 95% CIs ... CI [a, b]"), Codex's "qualitative-ize the caption" revision request is precisely the suggestion the spec bans. DISCARD without needing the interp-critic grep, and add a "Do NOT change — caption CIs stay" line to the binding fix list. (Origin: #509 follow-up re-gate round-1; Codex tagged `bracketed-ci-in-new-tldr-caption`, Claude's chart-error-bar-carve-out read was correct.) Reproducibility-table CIs are likewise always sanctioned.

## Prior-gate house-style extension — no fresh interp-critic trail needed (added 2026-06-10, task #464)

When a PRIOR clean-result-critique gate on the SAME body explicitly adjudicated bracketed CIs in finding read prose as Lens 7 PASS with a load-bearing rationale (e.g. #464's 2026-06-03 gate: "the headline statistic IS the paired-bootstrap CI"), a follow-up re-gate's NEW finding H4 that follows that adjudicated house style inherits the carve-out — even when the re-fold ran with NO interp-critique round (same-issue follow-up loops go analyzer → clean-result-critique directly, so there is no fresh interp trail to grep). Test the new CI on its own terms against entry G (headline statistic / powered-null honesty; no `± err` duplication) and check the prior gate's Lens 7 verdict text in events.jsonl. Stripping the new section's CIs while the parent's explicitly-PASSed CIs stay would make the register internally inconsistent — worse than the literal-spec deviation.

Asymmetry vs the sentence cap: REGISTER rules (Lens 7) are body-level house style, so a prior explicit adjudication binds new sections; STRUCTURE rules (Lens 2 sentence cap) are enforced fresh per new section (with entry-R severity). Origin: #464 minimal_content re-gate round-1 — Codex re-filed both line-84 (settled parent prose) and line-149 (new H4 following the settled style); both discarded/non-blocking, PASS.

## Origin

Task #478 round-1 clean-result reconcile (2026-06-04). Codex flagged bracketed CIs at body.md:90 + captions 162/178/180; the interp-critic's round-1 recommendation #5 was literally *"observed slope −0.12, CI [−0.22,−0.02] — opposite direction, not just NS"* and the round-2 interp verification praised the analyzer for adding it ("CLEAN"). Reconciling toward Codex without checking the interp-critic trail would have stripped the powered-null statement.

Related: [[claude-clean-result-critic-underapplies-spec-text]] (entry G specifically), the broader pattern of clean-result-critic spec-text-only rules.
