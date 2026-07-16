---
name: Simulated-user brief realism vs token-budget attrition
description: An LLM user-simulator template without measured real-user turn constraints elicits 3x longer subject answers and blows multi-turn token budgets; ground the brief on the corpus's turn stats and version-key it into the resume fingerprint (#825 r11 G-B)
type: feedback
---

An LLM user-simulator (Haiku generating "user" turns for multi-turn rollouts)
whose system template lacks explicit real-user constraints drifts to long,
multi-question, expansive turns — #825 r11 pilot: 124 words median vs 24 for
real corpus user turns, ~2 questions/turn vs 0 — which elicit subject answers
~3x longer than real turns do (instruct 464 vs 152 words median; 17% at the
gen cap). The cumulative render then overflows the engine/capture windows and
the failure surfaces as a COMPLETION-gate failure (100% window/capture-overflow
died_reasons) that looks like "conversations die early", inviting wrong
closer/end-detection hypotheses. A pretrained arm with a `"\n\n"` stop token is
immune (answers truncate at one paragraph) — an instruct-vs-base completion-rate
asymmetry at depth is the signature of this class.

**Why:** the plan's budget grounding used REAL-context answer lengths; the
simulator silently broke that premise. Diagnose from died_reason histograms +
per-depth user/answer word medians (metadata only) before touching thresholds.

**How to apply:** (1) write simulator briefs pinned to the MEASURED real-user
distribution (1-3 sentences / word bound from the corpus median, exactly ONE
focused question, no pleasantries, never close the conversation); (2) the
template text is an output-affecting regime key — version it into the resume
fingerprint (#722 r3) so old-template dirs refuse resume; (3) persist per-depth
user/answer length telemetry + finish_reason in rollout diagnostics so budget
attrition is attributable without a rerun; (4) a wave-level hard-fail raise
must carry a failure-REASON histogram, not just a rate. Fix the simulator, not
the pre-registered gate thresholds. Validation shape: live small-N smoke with
canned subject + real simulator, read user_words_median (v2 gave 22-24 vs
v1's 124, matching real u1's 24).
