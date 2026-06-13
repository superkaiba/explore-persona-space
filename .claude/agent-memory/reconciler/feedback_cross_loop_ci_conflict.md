---
name: cross-loop-ci-conflict
description: Codex clean-result-critic FAILs bracketed-CI / named-test prose that the upstream interp-critic explicitly REQUIRED (powered-null honesty). Check the interp markers + prior-gate adjudications before stripping quantitative content; captions and Reproducibility CIs are always sanctioned.
metadata:
  type: feedback
---

**Rule:** before classifying a bracketed-CI (`95% CI [−0.22, −0.02]`) Lens 7 finding as Real-blocking, grep events.jsonl for the round-1 `epm:interp-critique` markers. If the interp-critic explicitly required the CI as part of a powered-null / non-detection-honesty statement, DISCARD Codex's finding (SPEC Lens 7 test-definition exception + underapplies-spec-text entry G) and add a "Do NOT remove" line to the binding fix list naming the lines that keep their quantitative content. The two loops have different rule cultures (honesty-of-claim vs register); without cross-loop awareness the analyzer ping-pongs between contradictory demands. A register-only cosmetic alternative ("CI from −0.22 to −0.02") is optional.

**Caption carve-out (no interp trail needed, #509 re-gate):** bracketed CIs in a FIGURE CAPTION are outside Lens 7's FAIL scope entirely — the FAIL condition enumerates result-H3 setup/read paragraphs + the Confidence sentence only, and the spec verbatim bans "suggest stripping numbers from the figure caption". When the caption decodes error bars actually drawn on the chart, Codex's "qualitative-ize the caption" ask is precisely the banned suggestion. Reproducibility-table CIs likewise always sanctioned.

**Prior-gate house-style extension (#464 re-gate):** when a PRIOR clean-result gate on the SAME body explicitly adjudicated bracketed CIs in finding prose as Lens 7 PASS with a load-bearing rationale, a follow-up re-gate's NEW finding following that house style inherits the carve-out — same-issue follow-up loops have no fresh interp trail to grep; test the new CI against entry G + the prior gate's verdict text. Asymmetry: REGISTER rules (Lens 7) are body-level house style, so a prior adjudication binds new sections; STRUCTURE rules (sentence caps) are enforced fresh per new section.

Origin: #478 r1 — Codex flagged the CIs the interp-critic's recommendation had literally dictated ("observed slope −0.12, CI [−0.22,−0.02] — opposite direction, not just NS"); reconciling toward Codex would have stripped the powered-null statement. Related: [[feedback_claude_clean_result_critic_underapplies_spec_text]] entries G/K.
