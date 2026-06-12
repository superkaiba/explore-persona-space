---
name: same-text shift extraction must persist response texts
description: Trained-own-text teacher-forced cross-context geometry DVs entangle response-content composition with write direction; the #551 rig drops texts by default, making the content channel unweighable post-pod — Must-Fix is "persist texts/ids" (tiny metadata), never a gate
type: feedback
---

When a plan's geometry DV compares activation shifts ACROSS contexts under the
`same`-text variant (both models teacher-forced on the TRAINED model's own
greedy response per context), the texts differ systematically across contexts:
the source response asserts the implanted behavior, bystander responses mostly
don't, and the source assertion rate can be co-monotone with the plan's IV
(#603: teacher self-assertion 97/94/72% monotone in teacher prior, the exact
regression axis). A predicted direction-composition gradient (e.g. CMF-on-prior)
is then mimicable by "shifts read on implant-bearing text point differently than
shifts read on generic text" — a measurement-channel artifact, sharper than the
generic teacher-property attribution cap the plan already concedes.

**Why:** the #551 extraction rig (`activation_shift.py`, schema v2) persists
only delta tensors + counts — generated response texts/ids are discarded, and
post-pod regeneration needs a fresh GPU pod + bitwise-greedy luck. Surfaced on
#603 plan v1 (Methodology lens, 2026-06-11).

**How to apply:** for any same-text cross-context shift plan, check whether
per-(context, question) response texts or token ids ship in the payload/sidecar
(~10 MB for 10k cells). Present → the content channel is a Concern with a
prescribed analysis (split per-question shifts by assertion content; condition
û on leak-bearing vs clean bystander texts). Absent → Must-Fix: add the text
dump (metadata, not a gate). The cheap follow-up disambiguator is a
source-context-only `base`-text read (base model never asserts → implant-free
text, ~4-5% of extraction cost), which the plan can name without running.
Related: feedback_completion_source_swap_mediation.md (#563),
feedback_rank1_mechanism_test_confounds.md (dump-missing → Must-Fix rule).

Two additions from the Alternatives lens (#603 v1, same round): (1)
cross-family "replication" does NOT break this confound — refusal/EM extension
arms have the same structure (trained sources express on their own text at
rates plausibly co-monotone with their base-rate IV; #518 EM trained rates vary
widely per cell), so the channel replicates wherever expression tracks the IV;
(2) the stratified read is thin exactly at high-expression cells (97% assertion
⇒ ~0-1 non-expressing responses of 20) — mid-expression families carry the real
discriminating power; and attenuation/noise alternatives shrink |cos| toward 0,
so a significantly NEGATIVE source cosine is attenuation-incompatible — read
the sign, not just the ordering.
