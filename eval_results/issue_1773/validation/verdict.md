# Issue #1773 — judged-axis trust verdict (acceptance criterion 5)

Registered lattice (plan §3, DISJOINT + exhaustive): TRUSTWORTHY <=> detection >= 0.7 AND fuzzing >= 0.7 AND discrimination >= 0.5 AND kappa >= 0.6 AND shuffled-label detection <= 0.55; SEARCH-INDEX-ONLY otherwise.

Detection/fuzzing/discrimination score the per-feature DESCRIPTION (shared across axis rows); kappa is the axis-differentiating conjunct. identity_disposition headlines additionally require precision >= 0.5 on the human-annotated subset (proxy reads are labeled, never the gate).

## abstraction: **SEARCH-INDEX-ONLY**
- detection=0.704 fuzzing=0.694 discrimination=0.322 kappa=0.682 shuffled_detection=0.514
- NEAR-THRESHOLD: detection is within 2 SE of its bar (|0.704-0.7| <= 2x0.008); fuzzing is within 2 SE of its bar (|0.694-0.7| <= 2x0.010) — read with the SE, not as a hard flip

## speaker_property: **SEARCH-INDEX-ONLY**
- detection=0.704 fuzzing=0.694 discrimination=0.322 kappa=0.512 shuffled_detection=0.514
- NEAR-THRESHOLD: detection is within 2 SE of its bar (|0.704-0.7| <= 2x0.008); fuzzing is within 2 SE of its bar (|0.694-0.7| <= 2x0.010) — read with the SE, not as a hard flip

## content_type: **SEARCH-INDEX-ONLY**
- detection=0.704 fuzzing=0.694 discrimination=0.322 kappa=0.665 shuffled_detection=0.514
- NEAR-THRESHOLD: detection is within 2 SE of its bar (|0.704-0.7| <= 2x0.008); fuzzing is within 2 SE of its bar (|0.694-0.7| <= 2x0.010) — read with the SE, not as a hard flip

## functional_role: **SEARCH-INDEX-ONLY**
- detection=0.704 fuzzing=0.694 discrimination=0.322 kappa=0.310 shuffled_detection=0.514
- NEAR-THRESHOLD: detection is within 2 SE of its bar (|0.704-0.7| <= 2x0.008); fuzzing is within 2 SE of its bar (|0.694-0.7| <= 2x0.010) — read with the SE, not as a hard flip

## interpretable: **SEARCH-INDEX-ONLY**
- detection=0.704 fuzzing=0.694 discrimination=0.322 kappa=0.650 shuffled_detection=0.514
- NEAR-THRESHOLD: detection is within 2 SE of its bar (|0.704-0.7| <= 2x0.008); fuzzing is within 2 SE of its bar (|0.694-0.7| <= 2x0.010) — read with the SE, not as a hard flip

Random-init control (REPORTED, not gated — the 2410.13928-vs-2501.17727 contradiction): randinit_detection=0.6718346253229973

Judged-label freeze on #1482/#1092/#1738 CONTINUES: no axis passed its lattice row (itself a valid completion — acceptance criterion 5).
