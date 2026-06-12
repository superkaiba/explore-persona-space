---
name: sonnet-strong-nl-length-band-too-tight
description: ±20% char-length band on Sonnet-authored prompt rewrites (issue467_author_strong_nl.py) is too tight — first-attempt rewrites consistently land at ~33% frac_dev, downgrading ALL cells to FAIL_LENGTH. Same family as the ±10% audit-gate-drift problem.
metadata:
  type: feedback
---

When a "strong-NL" prompt author uses Sonnet to rewrite a literal target prompt
into a natural-language equivalent against a fixed `target` char length, the
realistic frac_dev distribution sits around ±30-40%, not ±20%. Task #467 SMOKE
round-6 demonstrated this with a 2/2 FAIL_LENGTH rate where the leak-check
itself passed cleanly (leak_score=0.0 both cells):

- `aesthetic_popular`: char_len=1782, target=1305, frac_dev=+0.366
- `aesthetic_unpopular`: char_len=1208, target=911, frac_dev=+0.326

The author phase reports `Strong-NL authoring complete. Status counts:
{FAIL_LENGTH: 2}` and the downstream elicitation phase errors with `No cells
with PASS strong-NL prompts; run author script first.` — making the gate
indistinguishable from a hard upstream failure for the operator.

**Why:** Sonnet expands prose to convey the same intent — it almost never
matches an arbitrary target length pulled from the lit baseline. The ±20%
band assumes "natural" drift but the real distribution after a single-shot
rewrite is wider. Same root cause as `feedback_audit_gate_arm_drift.md`
(±10% audit gate caught ~15% cross-prompt BPE drift in #389), one numeric
band size up.

**How to apply:**
1. If a SMOKE run shows ALL cells downgraded to `FAIL_LENGTH` with
   `leak_score=0.0`, it is NOT a quality failure — it is the gate. Do
   not retry SMOKE-as-is; bounce code-class to `experiment-implementer`.
2. Recommend in the failure note one of three fixes, in order of preference:
   (a) re-author retry loop with an explicit "shorter please, target N chars"
   instruction when frac_dev > band (best — preserves content quality),
   (b) widen the band to ±40% (cheap — but loses content/length parity
   guarantee), (c) truncate post-author to target (lossy — last resort).
3. Pre-launch check: when reviewing the brief, grep the author script for
   any `frac_dev` / `±0.20` / `LENGTH_BAND` constant. If present at ≤0.20,
   flag in the launch note as "assumption: tight length band may eat all
   Sonnet rewrites on first pass."
4. Don't post `epm:failure infra` — this is a code-tunable gate, not an
   environmental failure.
