---
name: claude-misses-lens7-statistical-framing
description: Claude clean-result-critic frequently PASSes bodies that violate Lens 7, statistical framing (named statistical tests + derived intervals in narrative prose); always check for "Wilson", "Fisher", "Mann-Whitney", "t-test", "±", and explicit % upper bounds in prose before trusting a Claude PASS
metadata:
  type: feedback
---

When reconciling a Claude PASS vs Codex REVISE on `clean-result-critic`,
always independently re-scan the body for Lens 7 (statistical-framing
rule) violations even if Claude reports all lenses PASS. Claude
critics frequently miss this lens — possibly because the verifier
script `verify_task_body.py` doesn't grep-check for it (it's a semantic
lens, not a mechanical one).

(Numbering note: statistical framing was Lens 11 before the v2-spec lens
renumbering; under the current 15-lens spec it is Lens 7, and Lens 11 is
"raw alongside processed". Historical verdicts — including the quote
below — may still say "Lens 11" and mean statistical framing.)

**Why:** Observed 2026-05-23 on task #378 reconcile. Claude reported
"All 11 lenses PASS" but missed `"Wilson 95% upper bound on a 0/200
proportion is approximately 1.8%"` in Details and `"Error bars are
95% Wilson confidence intervals"` in caption. Both are textbook
statistical-framing (now Lens 7)
violations: named statistical procedure in prose + derived numerical
interval in prose. Codex caught both.

**Lens 7 trip-wires to scan for:**
- Named statistical tests in prose: `Wilson`, `Fisher (exact)`,
  `Mann-Whitney`, `Wilcoxon`, `paired t-test`, `bootstrap test`,
  `chi-square`, `Kolmogorov-Smirnov`, `binomial test`.
- Derived intervals or effect sizes in prose: `±`, `[X, Y]`,
  `Cohen's d`, `η²`, `r =`, "upper bound of X%", "credence X to Y%".
- The caption is prose for this rule. "Error bars are 95% Wilson CIs"
  in a caption violates Lens 7; "Error bars are 95% CIs" does not.

**How to apply:** In every Claude-PASS vs Codex-FAIL reconcile on
`clean-result-critic`, before classifying any Codex statistical-framing
finding, grep the body for the trip-wires above. If any match, the
Codex finding is Real-blocking and the binding verdict is REVISE
regardless of Claude's broader PASS. The L7 lens is exactly the
load-bearing lens the clean-result-critic absorbed from the retired
`reviewer` agent — missing it is a category error that the reconciler
must catch.

Related: [[claude-underclasses-silent-failures]] — same pattern of
Claude reviewer correctly flagging an issue but mis-classing the verdict.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude misses Lens 7 statistical framing](feedback_claude_misses_lens7_statistical_framing.md) — always re-scan for named tests (Wilson/Fisher/Mann-Whitney), ±, derived intervals in prose/captions before trusting a Claude PASS (Lens 11 in pre-renumbering verdicts). #378.
