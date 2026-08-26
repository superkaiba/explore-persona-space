---
name: sibling-figure-palette-pair-rebinding
description: On a multi-figure commit, enumerate each figure's color→factor binding; the same palette pair (e.g. role primary/accent) bound to DIFFERENT factors across sibling figures violates the paper-plots §2 color↔meaning row — and a baseline rendered as different artists/colors across figures is the same defect
metadata:
  type: feedback
---

When one commit renders a figure SET for one writeup, build a small table
figure → {color/pair → factor} before judging any single figure. The
paper-plots SKILL §2 row pins ONE color↔meaning assignment across EVERY
figure of a writeup and explicitly bans reusing the same palette pair for
a DIFFERENT factor in a sibling figure (#1092 incident: same two colors =
model identity in one figure, arm in another → user had to ask).

**Why:** #2546 R1 g4 (`issue2546_figures.py`): role colors primary/accent
were bound to necessity strata in hero3, post-vs-pre read in the n1m
exploratory panel, and transfer-R² vs acc@1 in the OOD panel — while the
strata everywhere else were hatch-coded grey; the identity(+bias) baseline
was a black tick in heroes 1–2 but a role-colored bar in the baseline-
decomposition panel. Each figure alone looked fine; only the cross-figure
table exposed the rebinding. Flagged Major (mechanical fix).

**How to apply:** (1) grep the figure script for every color source
(palette slices, role calls, literal colors) and list per figure what each
maps to; (2) flag any pair→factor rebinding across figures, and any single
quantity (a baseline, a chance line) drawn with different visual identity
in different figures; (3) `paper_palette(n)` is style-independent (Wong)
but `paper_palette_role` resolves at CALL time against the ACTIVE style —
module-import-time role capture vs in-function calls can silently mix
palettes. Sibling: [[figure-populated-assert-reference-artists]] (the
non-empty-axes guard on the same class of commits — re-fired verbatim on
#2546: artist-count scan passes on all-NaN bars + axhline).
