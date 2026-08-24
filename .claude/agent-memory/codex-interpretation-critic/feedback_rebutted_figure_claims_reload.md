---
name: feedback-rebutted-figure-claims-reload
description: When a revision REBUTS the Codex twin's own round-1 figure claims (blob-verified), the r2 prompt mandates a fresh PNG reload + an explicit RETRACT-or-stand adjudication — never a from-memory re-assertion
metadata:
  type: feedback
---

When an interpretation revision REBUTS (rather than fixes) a round-1 Codex
figure finding — e.g. #2474 r1: Codex claimed `prefit_perm_band_caps.png`
"omits all arm labels and the legend" and `prefit_layers_caps.png` "omits the
y-axis label and legend", while the Claude twin decoded every row of the same
PNGs and the analyzer verified the committed blobs byte-identical to the
copies read — the round-2 prompt must:

1. Blob-verify at compose time that the worktree PNGs match EVERY pin cited
   (the revision's rebuttal pin AND the round pin), and SAY so in the prompt,
   so "different bytes" is off the table as an explanation.
2. Mark those figures RELOAD MANDATORY and require a binary adjudication:
   `round-1 claim RETRACTED — rebuttal accepted` vs `rebuttal FAILS — <what
   is actually visible>`, with no split-the-difference option.
3. Forbid re-asserting the round-1 claim without a fresh successful image
   load — if the load fails, the line is `rebuttal re-verification BLOCKED —
   image load failed`, never a from-memory repeat (a Codex sandbox image-load
   failure silently truncating to a partial render is the likely root cause
   of divergent twin figure reads).

**Why:** twin-vs-twin figure-read divergence is otherwise unresolvable by the
reconciler — both claims are "loaded: yes" assertions about the same bytes.
Forcing the fresh load + explicit retract/stand converts it into evidence.

**How to apply:** in any r2+ composition where the revision ledger contains a
REBUTTED item targeting a Codex figure claim (see also
[[feedback-lens7-carried-forward-on-revision-rounds]] for the HF-only
sub-check analogue).
